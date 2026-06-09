"""
Journal Integrity Verifier Tests — P2.6.

4 verification levels:
  1. Hash-chain — SHA256 chain integrity
  2. Sequence — monotonic entry_id index
  3. Causal links — bridge edges reference valid journal entries
  4. Lifecycle — valid state machine transitions
"""

import copy
import hashlib
import json
from execution_dynamics.journal import DispatchJournal, JournalEntry
from execution_dynamics.integrity import IntegrityVerifier, IntegrityReport, IntegrityError


def make_entry(event: str, execution_id: str = "exec1", goal_id: str = "goal1",
               sequence: int = 0, entry_hash: str = "", prev_hash: str = "") -> JournalEntry:
    return JournalEntry(
        event=event,
        goal_id=goal_id,
        execution_id=execution_id,
        lease_id=f"lease_{execution_id}",
        timestamp=1000.0 + sequence,
        entry_id=f"{execution_id}:{sequence}:{event.lower()}",
        dispatch_epoch=sequence,
        entry_hash=entry_hash,
        prev_hash=prev_hash,
    )


def journal_with_events(events: list) -> DispatchJournal:
    """Build a journal from a list of event strings using proper append()."""
    j = DispatchJournal()
    for i, event in enumerate(events):
        entry = JournalEntry(
            event=event,
            goal_id="goal1",
            execution_id="exec1",
            lease_id="lease1",
            timestamp=1000.0 + i,
            entry_id=f"exec1:{i}:{event.lower()}",
            dispatch_epoch=i,
        )
        j.append(entry)
    return j


class TestHashChain:

    def test_clean_hash_chain(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        v = IntegrityVerifier()
        errors = v.verify_hash_chain(j)
        assert errors == []

    def test_tampered_entry_hash(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        # Corrupt the second entry's hash
        j._entries[1].entry_hash = "deadbeef" * 8
        v = IntegrityVerifier()
        errors = v.verify_hash_chain(j)
        # Cascade: entry 1 hash fails, entry 2 prev_hash fails, entry 2 hash fails
        assert len(errors) >= 1
        assert errors[0].type == 'hash_chain'

    def test_broken_prev_hash(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        # Corrupt prev_hash of the third entry
        j._entries[2].prev_hash = "deadbeef" * 8
        v = IntegrityVerifier()
        errors = v.verify_hash_chain(j)
        assert len(errors) == 1
        assert errors[0].type == 'hash_chain'

    def test_tampered_event_type(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        # Change event type after append (simulates storage corruption)
        j._entries[2].event = 'FAILED'
        v = IntegrityVerifier()
        errors = v.verify_hash_chain(j)
        # Should fail because hash no longer matches
        assert len(errors) >= 1

    def test_empty_journal(self):
        j = DispatchJournal()
        v = IntegrityVerifier()
        errors = v.verify_hash_chain(j)
        assert errors == []

    def test_single_entry(self):
        j = DispatchJournal()
        entry = JournalEntry(event='DISPATCHED', goal_id='g1', execution_id='e1',
                             lease_id='l1', timestamp=1000.0)
        j.append(entry)
        v = IntegrityVerifier()
        errors = v.verify_hash_chain(j)
        assert errors == []


class TestSequenceVerification:

    def test_clean_sequence(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        v = IntegrityVerifier()
        errors = v.verify_sequence(j)
        assert errors == []

    def test_gap_in_sequence(self):
        j = DispatchJournal()
        entries = [
            JournalEntry(event='DISPATCHED', goal_id='g1', execution_id='e1',
                         lease_id='l1', timestamp=1000.0,
                         entry_id='e1:0:dispatched'),
            JournalEntry(event='COMPLETED', goal_id='g1', execution_id='e1',
                         lease_id='l1', timestamp=1001.0,
                         entry_id='e1:5:completed'),  # gap: 1→5
        ]
        for e in entries:
            j._entries.append(e)
        v = IntegrityVerifier()
        errors = v.verify_sequence(j)
        assert len(errors) == 1
        assert errors[0].type == 'sequence'

    def test_reversed_sequence(self):
        j = DispatchJournal()
        entries = [
            JournalEntry(event='COMPLETED', goal_id='g1', execution_id='e1',
                         lease_id='l1', timestamp=1000.0,
                         entry_id='e1:3:completed'),
            JournalEntry(event='DISPATCHED', goal_id='g1', execution_id='e1',
                         lease_id='l1', timestamp=999.0,
                         entry_id='e1:0:dispatched'),
        ]
        for e in entries:
            j._entries.append(e)
        v = IntegrityVerifier()
        errors = v.verify_sequence(j)
        assert len(errors) >= 1

    def test_non_parseable_entry_id(self):
        j = DispatchJournal()
        j._entries.append(JournalEntry(event='DISPATCHED', goal_id='g1',
                                        execution_id='e1', lease_id='l1',
                                        timestamp=1000.0,
                                        entry_id='bad-id-format'))
        v = IntegrityVerifier()
        errors = v.verify_sequence(j)
        assert len(errors) == 1
        assert errors[0].type == 'sequence'


class TestCausalLinks:

    def test_clean_causal_links(self):
        """Bridge edges reference valid execution_ids in journal."""
        j = journal_with_events(['DISPATCHED', 'STARTED', 'COMPLETED'])

        class MockEdge:
            def __init__(self, eid, exec_id):
                self.edge_id = eid
                self.execution_entry_id = exec_id

        class MockGraph:
            def __init__(self):
                self.edges = [
                    MockEdge('ce:1', 'exec1'),
                    MockEdge('ce:2', 'exec1'),
                ]

        v = IntegrityVerifier()
        errors = v.verify_causal_links(j, MockGraph())
        assert errors == []

    def test_broken_causal_link(self):
        """Bridge edge references non-existent execution_id."""
        j = journal_with_events(['DISPATCHED', 'COMPLETED'])

        class MockEdge:
            def __init__(self, eid, exec_id):
                self.edge_id = eid
                self.execution_entry_id = exec_id

        class MockGraph:
            def __init__(self):
                self.edges = [
                    MockEdge('ce:1', 'nonexistent_exec'),
                ]

        v = IntegrityVerifier()
        errors = v.verify_causal_links(j, MockGraph())
        assert len(errors) == 1
        assert errors[0].type == 'causal_link'

    def test_no_bridge_graph_skips_check(self):
        j = journal_with_events(['DISPATCHED', 'COMPLETED'])
        v = IntegrityVerifier()
        errors = v.verify_causal_links(j, bridge_graph=None)
        assert errors == []


class TestLifecycle:

    def test_clean_lifecycle(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        v = IntegrityVerifier()
        errors = v.verify_lifecycle(j)
        assert errors == []

    def test_clean_retry_lifecycle(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED',
                                  'FAILED', 'RETRIED', 'STARTED', 'COMPLETED'])
        v = IntegrityVerifier()
        errors = v.verify_lifecycle(j)
        assert errors == []

    def test_invalid_transition(self):
        j = DispatchJournal()
        # DISPATCHED → COMPLETED (missing LEASE_ISSUED, STARTED)
        j.append(JournalEntry(event='DISPATCHED', goal_id='g1', execution_id='e1',
                              lease_id='l1', timestamp=1000.0))
        j.append(JournalEntry(event='COMPLETED', goal_id='g1', execution_id='e1',
                              lease_id='l1', timestamp=1001.0))
        v = IntegrityVerifier()
        errors = v.verify_lifecycle(j)
        assert len(errors) >= 1
        assert errors[0].type == 'lifecycle'
        assert 'invalid transition' in errors[0].detail

    def test_dispatched_abandoned_valid(self):
        j = journal_with_events(['DISPATCHED', 'ABANDONED'])
        v = IntegrityVerifier()
        errors = v.verify_lifecycle(j)
        assert errors == []

    def test_started_before_dispatched_invalid(self):
        """STARTED without preceding DISPATCHED → invalid lifecycle."""
        j = journal_with_events(['STARTED', 'COMPLETED'])
        v = IntegrityVerifier()
        errors = v.verify_lifecycle(j)
        assert len(errors) >= 1

    def test_completed_to_started_invalid(self):
        """COMPLETED→STARTED is not a valid transition."""
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED',
                                  'COMPLETED', 'STARTED'])
        v = IntegrityVerifier()
        errors = v.verify_lifecycle(j)
        assert len(errors) >= 1
        assert any(e.type == 'lifecycle' for e in errors)


class TestIntegrityReport:

    def test_clean_report(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        v = IntegrityVerifier()
        report = v.verify_integrity(j)
        assert report.valid is True
        assert report.hash_chain_ok is True
        assert report.sequence_ok is True
        assert report.causal_links_ok is True
        assert report.lifecycle_ok is True
        assert len(report.errors) == 0

    def test_corrupted_report(self):
        j = journal_with_events(['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'])
        # Tamper with hash and create invalid transition
        j._entries[2].entry_hash = "dead" * 16
        # Manually create invalid sequence
        j._entries[2].entry_id = 'exec1:99:started'
        v = IntegrityVerifier()
        report = v.verify_integrity(j)
        assert report.valid is False
        assert report.hash_chain_ok is False
        assert len(report.errors) > 0

    def test_report_to_dict(self):
        j = journal_with_events(['DISPATCHED', 'STARTED', 'COMPLETED',
                                  'DISPATCHED'])  # DISPATCHED after COMPLETED is invalid
        v = IntegrityVerifier()
        report = v.verify_integrity(j)
        d = report.to_dict()
        assert 'valid' in d
        assert 'hash_chain_ok' in d
        assert 'n_errors' in d
        assert isinstance(d['errors'], list)
