"""
P3.3 — Adversarial Lifecycle.

Verifies IntegrityVerifier.lifecycle checks across categories:

  B. Structural attacks → IntegrityVerifier detects sequence violations
  C. Cryptographic attacks → IntegrityVerifier detects hash chain violations
  D. Semantic attacks → IntegrityVerifier detects lifecycle violations

Plus state machine fuzzing: random event sequences, verify
    sequence valid ⇔ IntegrityVerifier accepts
"""

import os
import tempfile
import itertools
import random
from execution_dynamics.jsonl_wal import JsonLinesWAL
from execution_dynamics.journal import DispatchJournal, JournalEntry
from execution_dynamics.integrity import IntegrityVerifier, _VALID_TRANSITIONS, _VALID_INITIAL_EVENTS
from tests.fault_injection.attacks import (
    STRUCTURAL_ATTACKS,
    CRYPTOGRAPHIC_ATTACKS,
    SEMANTIC_ATTACKS,
    ALL_ATTACKS,
)
from tests.fault_injection.conftest import build_wal_with_events


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

ALL_EVENTS = list(_VALID_TRANSITIONS.keys())


def _verify_from_file(path) -> tuple:
    """Load journal from WAL file and run IntegrityVerifier. Returns (report, journal)."""
    wal = JsonLinesWAL(path, auto_recover=True)
    journal = DispatchJournal(wal=wal)
    n = journal.recover_from_wal()
    verifier = IntegrityVerifier()
    report = verifier.verify_integrity(journal)
    return report, journal


class TestStructuralAttacks:

    def test_each_structural_attack_caught_by_sequence_check(self):
        """Structural attacks (delete, duplicate, swap, reorder) → sequence failure."""
        for attack in STRUCTURAL_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                # Use extended lifecycle (7 entries) so reorder_range has room to work
                build_wal_with_events(path, EXTENDED_LIFECYCLE)
                attack(path)

                report, _ = _verify_from_file(path)
                # Sequence check must detect the structural damage
                assert not report.sequence_ok, \
                    f"{attack.__name__}: sequence_ok should be False"
                # Report must be invalid
                assert not report.valid, \
                    f"{attack.__name__}: report should be invalid"
            finally:
                os.unlink(path)


class TestCryptographicAttacks:

    def test_each_crypto_attack_caught_by_hash_chain(self):
        """Cryptographic attacks (hash corruption) → hash chain failure."""
        for attack in CRYPTOGRAPHIC_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                build_wal_with_events(path, VALID_LIFECYCLE)
                # Need hash chain to exist first
                attack(path)

                report, _ = _verify_from_file(path)
                assert not report.hash_chain_ok, \
                    f"{attack.__name__}: hash_chain_ok should be False"
            finally:
                os.unlink(path)


class TestSemanticAttacks:

    def test_each_semantic_attack_caught_by_lifecycle(self):
        """Semantic attacks (invalid transitions) → lifecycle failure."""
        for attack in SEMANTIC_ATTACKS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                path = f.name
            try:
                build_wal_with_events(path, VALID_LIFECYCLE)
                attack(path)

                report, _ = _verify_from_file(path)
                assert not report.lifecycle_ok, \
                    f"{attack.__name__}: lifecycle_ok should be False"
            finally:
                os.unlink(path)


class TestStateMachineFuzzing:

    def test_known_valid_sequences(self):
        """Pre-constructed valid sequences must pass lifecycle check."""
        valid_sequences = [
            ['DISPATCHED'],
            ['DISPATCHED', 'ABANDONED'],
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'],
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'FAILED'],
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'FAILED', 'RETRIED', 'STARTED', 'COMPLETED'],
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'PREEMPTED', 'RETRIED', 'STARTED', 'COMPLETED'],
            ['DISPATCHED', 'LEASE_ISSUED', 'LEASE_EXPIRED'],
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'CANCELLING', 'CANCELLED'],
        ]
        verifier = IntegrityVerifier()
        for seq in valid_sequences:
            journal = _journal_from_events(seq)
            report = verifier.verify_integrity(journal)
            assert report.lifecycle_ok and report.valid, \
                f"Valid sequence rejected: {seq} errors={report.errors}"

    def test_known_invalid_sequences(self):
        """Pre-constructed invalid sequences must fail lifecycle check."""
        invalid_sequences = [
            ['STARTED'],  # orphan
            ['COMPLETED'],  # orphan
            ['DISPATCHED', 'COMPLETED'],  # missing LEASE_ISSUED, STARTED
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED', 'STARTED'],  # terminal→active
            ['DISPATCHED', 'LEASE_ISSUED', 'COMPLETED'],  # missing STARTED
            ['DISPATCHED', 'DISPATCHED'],  # DISPATCHED→DISPATCHED
            ['ABANDONED'],  # orphan terminal
            ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'CANCELLED', 'STARTED'],  # terminal→active
        ]
        verifier = IntegrityVerifier()
        for seq in invalid_sequences:
            journal = _journal_from_events(seq)
            report = verifier.verify_integrity(journal)
            assert not report.lifecycle_ok, \
                f"Invalid sequence accepted: {seq}"
            assert not report.valid, \
                f"Invalid sequence reported as valid: {seq}"

    def test_fuzz_bounded_random_sequences(self):
        """
        Fuzz test: generate random sequences of length 1..8 using
        valid event types. Check that lifecycle_ok and valid are consistent:
          lifecycle_ok ⇔ the sequence has no illegal transitions + valid start.
        """
        verifier = IntegrityVerifier()
        for length in range(1, 9):
            for _ in range(50):
                seq = [random.choice(ALL_EVENTS) for _ in range(length)]
                journal = _journal_from_events(seq)
                report = verifier.verify_integrity(journal)

                # Manual validation: check transitions
                manual_valid = _is_lifecycle_valid(seq)
                if manual_valid:
                    assert report.lifecycle_ok, \
                        f"Manually valid sequence rejected: {seq}"
                else:
                    assert not report.lifecycle_ok, \
                        f"Manually invalid sequence accepted: {seq}"

    def test_fuzz_all_pairs(self):
        """
        Exhaustive pair test: every (event_a, event_b) combination.
        Verify that IntegrityVerifier matches _VALID_TRANSITIONS.
        """
        verifier = IntegrityVerifier()
        for event_a in ALL_EVENTS:
            for event_b in ALL_EVENTS:
                journal = _journal_from_events([event_a, event_b])
                report = verifier.verify_integrity(journal)

                expected_valid = _is_transition_valid(event_a, event_b)
                if expected_valid:
                    msg = f"VALID transition rejected: {event_a}->{event_b}"
                    # Must have no lifecycle error unless initial event is wrong
                    # Two-element seq: if event_a IS a valid initial event, then
                    # a->b valid means lifecycle_ok
                    if event_a in _VALID_INITIAL_EVENTS:
                        assert report.lifecycle_ok, f"{msg} {report.errors}"
                else:
                    msg = f"INVALID transition accepted: {event_a}->{event_b}"
                    # If event_a is a valid initial event but transition is bad
                    if event_a in _VALID_INITIAL_EVENTS:
                        assert not report.lifecycle_ok, msg
                    # If event_a is NOT a valid initial event, initial check catches it


# ============================================================================
# Helpers
# ============================================================================

def _journal_from_events(events: list) -> DispatchJournal:
    """Build an in-memory DispatchJournal from a list of event types."""
    journal = DispatchJournal()
    for i, ev in enumerate(events):
        entry = JournalEntry(
            event=ev,
            goal_id='goal_1',
            execution_id='exec1',
            lease_id='lease_1',
            timestamp=1000.0 + i,
        )
        journal.append(entry)
    return journal


def _is_transition_valid(prev_event: str, curr_event: str) -> bool:
    """Check if curr_event is a valid transition from prev_event."""
    allowed = _VALID_TRANSITIONS.get(prev_event, [])
    return curr_event in allowed


def _is_lifecycle_valid(events: list) -> bool:
    """Full lifecycle validation (same logic as IntegrityVerifier)."""
    if not events:
        return True
    # Check initial event
    if events[0] not in _VALID_INITIAL_EVENTS:
        return False
    # Check transitions
    for i in range(1, len(events)):
        if not _is_transition_valid(events[i - 1], events[i]):
            return False
    return True
