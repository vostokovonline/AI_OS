"""
P3.2 — Replay Determinism Under Damage.

Core invariant:

  state_original == state_after_corruption_and_recovery

For ANY journal and ANY physical attack:
  - Let S0 = replay(clean_journal)
  - Apply attack to WAL file
  - Recover WAL (truncate to valid prefix)
  - Let S1 = replay(recovered_journal)
  - Then S1 is a prefix of S0 (first k entries identical)
"""

import os
import tempfile
import json
import hashlib
from execution_dynamics.jsonl_wal import JsonLinesWAL
from execution_dynamics.journal import DispatchJournal
from tests.fault_injection.attacks import PHYSICAL_ATTACKS
from tests.fault_injection.conftest import build_wal_with_events, jhash


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


def replay_state_hash(wal) -> str:
    """Deterministic hash of all journal entries replayed from WAL."""
    journal = DispatchJournal(wal=wal)
    journal.recover_from_wal()
    return jhash(journal._entries)


class TestReplayDeterminism:

    def test_clean_replay_is_idempotent(self):
        """Replaying the same WAL twice produces the same state hash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            build_wal_with_events(path, VALID_LIFECYCLE)

            hash_a = replay_state_hash(JsonLinesWAL(path))
            hash_b = replay_state_hash(JsonLinesWAL(path))

            assert hash_a == hash_b
        finally:
            os.unlink(path)

    def test_replay_reopen_is_idempotent(self):
        """Close, reopen, replay — same hash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            path = f.name
        try:
            wal = build_wal_with_events(path, VALID_LIFECYCLE)
            hash_a = replay_state_hash(wal)
            del wal

            hash_b = replay_state_hash(JsonLinesWAL(path))
            assert hash_a == hash_b
        finally:
            os.unlink(path)

    def test_recovery_prefix_matches_original_prefix(self):
        """For each physical attack, recovered state equals first N entries of original."""
        for attack in PHYSICAL_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                # Build clean WAL
                build_wal_with_events(path, EXTENDED_LIFECYCLE)

                # Get original state (already loaded into journal)
                wal_orig = JsonLinesWAL(path)
                journal_orig = DispatchJournal(wal=wal_orig)
                journal_orig.recover_from_wal()
                n_original = len(journal_orig._entries)
                orig_hash = jhash(journal_orig._entries)
                del wal_orig

                # Apply attack
                attack(path)

                # Recover and replay
                wal_recovered = JsonLinesWAL(path, auto_recover=True)
                journal_rec = DispatchJournal(wal=wal_recovered)
                n_recovered = journal_rec.recover_from_wal()
                recovered_hash = jhash(journal_rec._entries)

                # Recovered state must be a prefix of original
                # (recovered entries up to n_recovered must match original's first n_recovered)
                recovered_entries = journal_rec._entries
                original_entries = journal_orig._entries[:n_recovered]
                rec_hash = jhash(recovered_entries)
                orig_prefix_hash = jhash(original_entries)
                assert rec_hash == orig_prefix_hash, \
                    f"{attack.__name__}: recovered entries don't match original prefix"

                # If no entries were lost, full state must match
                if n_recovered == n_original:
                    assert recovered_hash == orig_hash, \
                        f"{attack.__name__}: full state mismatch after non-destructive attack"
            finally:
                os.unlink(path)

    def test_recovery_determinism_triple_replay(self):
        """Recover from corrupted WAL three times — all three must produce same state."""
        for attack in PHYSICAL_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                build_wal_with_events(path, EXTENDED_LIFECYCLE)
                attack(path)

                hashes = []
                for _ in range(3):
                    h = replay_state_hash(JsonLinesWAL(path, auto_recover=True))
                    hashes.append(h)

                assert hashes[0] == hashes[1] == hashes[2], \
                    f"{attack.__name__}: non-deterministic recovery"
            finally:
                os.unlink(path)
