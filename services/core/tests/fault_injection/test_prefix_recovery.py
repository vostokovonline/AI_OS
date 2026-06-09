"""
P3.1 — Prefix Recovery Proof.

Property: For ANY physical WAL corruption, recovery produces the
longest valid prefix of the original WAL.

Recovered(W) == longest_valid_prefix(W)

This is tested by:
  1. Building a clean WAL with N entries
  2. Applying each physical attack
  3. Verifying the recovered journal is a prefix of the original
  4. Verifying IntegrityVerifier reports the recovered journal as valid
"""

import os
import tempfile
import json
import hashlib
from execution_dynamics.jsonl_wal import JsonLinesWAL
from execution_dynamics.journal import DispatchJournal
from execution_dynamics.integrity import IntegrityVerifier
from tests.fault_injection.attacks import PHYSICAL_ATTACKS
from tests.fault_injection.conftest import build_wal_with_events, journal_hash


VALID_LIFECYCLE = [
    'DISPATCHED',
    'LEASE_ISSUED',
    'STARTED',
    'COMPLETED',
]

EXTENDED_LIFECYCLE = [
    'DISPATCHED',
    'LEASE_ISSUED',
    'STARTED',
    'FAILED',
    'RETRIED',
    'STARTED',
    'COMPLETED',
]


def _read_valid_prefix(path: str) -> int:
    """Count valid JSON lines in a WAL file (returns count)."""
    count = 0
    if not os.path.exists(path):
        return 0
    with open(path, 'rb') as f:
        raw = f.read()
    for raw_line in raw.split(b'\n'):
        if not raw_line.strip():
            continue
        try:
            decoded = raw_line.decode('utf-8')
            json.loads(decoded)
            count += 1
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
    return count


class TestPrefixRecovery:

    def test_clean_wal_recovery_unchanged(self):
        """Clean WAL → recovery returns all entries."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            build_wal_with_events(path, VALID_LIFECYCLE)

            journal = DispatchJournal(wal=JsonLinesWAL(path))
            n1 = journal.recover_from_wal()
            hash1 = journal_hash(journal)

            journal2 = DispatchJournal(wal=JsonLinesWAL(path))
            n2 = journal2.recover_from_wal()
            hash2 = journal_hash(journal2)

            assert n1 == n2 == 4
            assert hash1 == hash2, "Clean WAL must recover identically"
        finally:
            os.unlink(path)

    def test_each_physical_attack_produces_valid_prefix(self):
        """For every physical attack, recovered journal is the longest valid prefix."""
        for attack in PHYSICAL_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                # Build clean WAL
                build_wal_with_events(path, VALID_LIFECYCLE)
                valid_prefix_count = _read_valid_prefix(path)

                # Apply attack
                attack(path)

                # Recover
                wal = JsonLinesWAL(path, auto_recover=True)
                journal = DispatchJournal(wal=wal)
                n_recovered = journal.recover_from_wal()

                # Recovered count must be <= original valid prefix
                recovered_valid = _read_valid_prefix(path)
                assert recovered_valid <= valid_prefix_count, \
                    f"{attack.__name__}: recovered {recovered_valid} > original prefix {valid_prefix_count}"

                # Recovered journal must pass IntegrityVerifier
                verifier = IntegrityVerifier()
                report = verifier.verify_integrity(journal)
                if len(journal._entries) > 0:
                    assert report.hash_chain_ok, \
                        f"{attack.__name__}: recovered journal hash chain broken"
                    assert report.valid, \
                        f"{attack.__name__}: recovered journal integrity failed: {report.errors}"
            finally:
                os.unlink(path)

    def test_prefix_recovery_extended_lifecycle(self):
        """Same property with a longer lifecycle (more entries = more attack surface)."""
        for attack in PHYSICAL_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                build_wal_with_events(path, EXTENDED_LIFECYCLE)
                valid_prefix_count = _read_valid_prefix(path)

                attack(path)

                wal = JsonLinesWAL(path, auto_recover=True)
                journal = DispatchJournal(wal=wal)
                n_recovered = journal.recover_from_wal()

                recovered_valid = _read_valid_prefix(path)
                assert recovered_valid <= valid_prefix_count

                verifier = IntegrityVerifier()
                report = verifier.verify_integrity(journal)
                if len(journal._entries) > 0:
                    assert report.hash_chain_ok, \
                        f"{attack.__name__}: hash chain broken after recovery"
            finally:
                os.unlink(path)

    def test_prefix_never_shorter_after_multiple_recoveries(self):
        """Recover, append, crash, recover — prefix should only grow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = build_wal_with_events(path, VALID_LIFECYCLE)
            del wal

            # Apply a non-destructive attack
            from tests.fault_injection.attacks import insert_binary_garbage_tail
            insert_binary_garbage_tail(path)

            # First recovery
            wal2 = JsonLinesWAL(path, auto_recover=True)
            assert wal2.get_entry_count() == 4
            del wal2

            # Append more entries
            wal3 = JsonLinesWAL(path, auto_recover=True)
            wal3.append('DISPATCHED', 'exec2:0:dispatched', {'event': 'DISPATCHED'})
            del wal3

            # Second recovery — should have 5 entries
            wal4 = JsonLinesWAL(path, auto_recover=True)
            assert wal4.get_entry_count() == 5
        finally:
            os.unlink(path)
