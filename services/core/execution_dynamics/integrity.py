"""
Journal Integrity Verifier — validates consistency of the execution journal.

4 verification levels:
  1. Hash-chain — SHA256 chain integrity (tamper detection)
  2. Sequence — monotonic entry_id index (gap detection)
  3. Causal links — bridge edges reference valid journal entries
  4. Lifecycle — valid state machine transitions per execution_id

Usage:
    verifier = IntegrityVerifier()
    report = verifier.verify_integrity(journal, bridge_graph=None)
    if not report.valid:
        for err in report.errors:
            print(err)
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from .journal import DispatchJournal, JournalEntry


# Valid lifecycle transitions (state machine)
_VALID_TRANSITIONS: Dict[str, List[str]] = {
    # Initial events (no prior event needed)
    'DISPATCHED':     ['LEASE_ISSUED', 'ABANDONED'],
    'LEASE_ISSUED':   ['STARTED', 'ABANDONED', 'LEASE_EXPIRED', 'LEASE_REVOKED'],
    'STARTED':        ['COMPLETED', 'FAILED', 'PREEMPTED', 'CANCELLING', 'LEASE_EXPIRED'],
    'COMPLETED':      [],  # terminal
    'FAILED':         ['RETRIED', 'ABANDONED'],
    'RETRIED':        ['STARTED'],
    'PREEMPTED':      ['RETRIED', 'ABANDONED'],
    'ABANDONED':      [],  # terminal
    'CANCELLING':     ['CANCELLED'],
    'CANCELLED':      [],  # terminal
    'LEASE_EXPIRED':  ['RETRIED', 'ABANDONED'],
    'LEASE_REVOKED':  [],  # terminal
}

# Events that are terminal (no valid transition out)
_TERMINAL_EVENTS = {'COMPLETED', 'ABANDONED', 'CANCELLED', 'LEASE_REVOKED'}

# Valid initial events (must be the first event in a lifecycle sequence)
_VALID_INITIAL_EVENTS = {'DISPATCHED', 'RECOVERED'}


@dataclass
class IntegrityError:
    type: str          # 'hash_chain', 'sequence', 'causal_link', 'lifecycle'
    detail: str
    entry_id: str = ""


@dataclass
class IntegrityReport:
    valid: bool = True
    hash_chain_ok: bool = True
    sequence_ok: bool = True
    causal_links_ok: bool = True
    lifecycle_ok: bool = True
    errors: List[IntegrityError] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'valid': self.valid,
            'hash_chain_ok': self.hash_chain_ok,
            'sequence_ok': self.sequence_ok,
            'causal_links_ok': self.causal_links_ok,
            'lifecycle_ok': self.lifecycle_ok,
            'errors': [{'type': e.type, 'detail': e.detail, 'entry_id': e.entry_id} for e in self.errors],
            'n_errors': len(self.errors),
        }


class IntegrityVerifier:
    """
    Verifies journal integrity across 4 dimensions.

    Pure function — no state mutation.
    Can verify any journal, including recovered WAL replays.
    """

    def verify_hash_chain(self, journal: DispatchJournal) -> List[IntegrityError]:
        """Verify SHA256 hash chain across all entries in order."""
        errors = []
        prev_hash = ""
        for entry in journal._entries:
            expected = entry.compute_hash(prev_hash)
            if entry.entry_hash and entry.entry_hash != expected:
                errors.append(IntegrityError(
                    type='hash_chain',
                    detail=f"Entry {entry.entry_id}: hash mismatch. "
                           f"expected={expected[:16]}... stored={entry.entry_hash[:16]}...",
                    entry_id=entry.entry_id,
                ))
            if entry.prev_hash and entry.prev_hash != prev_hash:
                errors.append(IntegrityError(
                    type='hash_chain',
                    detail=f"Entry {entry.entry_id}: prev_hash mismatch. "
                           f"expected={prev_hash[:16] if prev_hash else 'empty'} "
                           f"stored={entry.prev_hash[:16] if entry.prev_hash else 'empty'}",
                    entry_id=entry.entry_id,
                ))
            prev_hash = entry.entry_hash or expected
        return errors

    def verify_sequence(self, journal: DispatchJournal) -> List[IntegrityError]:
        """Verify monotonic sequence indices in entry_ids."""
        errors = []
        pattern = re.compile(r'^(.+):(\d+):(.+)$')
        expected_idx = 0
        for entry in journal._entries:
            m = pattern.match(entry.entry_id)
            if not m:
                errors.append(IntegrityError(
                    type='sequence',
                    detail=f"Entry {entry.entry_id}: cannot parse sequence index",
                    entry_id=entry.entry_id,
                ))
                continue
            idx = int(m.group(2))
            if idx != expected_idx:
                errors.append(IntegrityError(
                    type='sequence',
                    detail=f"Entry {entry.entry_id}: expected sequence index {expected_idx}, got {idx}",
                    entry_id=entry.entry_id,
                ))
            expected_idx = idx + 1
        return errors

    def verify_causal_links(
        self,
        journal: DispatchJournal,
        bridge_graph: Optional[Any] = None,
    ) -> List[IntegrityError]:
        """
        Verify that all bridge causality edges reference valid journal entries.

        Each bridge edge has execution_entry_id which must match at least
        one journal entry's execution_id.

        If no bridge_graph is provided, this check is skipped.
        """
        errors = []
        if bridge_graph is None:
            return errors

        # Collect all execution_ids from journal
        journal_exec_ids = set()
        for entry in journal._entries:
            if entry.execution_id:
                journal_exec_ids.add(entry.execution_id)

        # Check each bridge edge
        if hasattr(bridge_graph, 'edges'):
            for edge in bridge_graph.edges:
                exec_id = getattr(edge, 'execution_entry_id', None) or getattr(edge, 'execution_goal_id', None)
                if exec_id and exec_id not in journal_exec_ids and len(journal_exec_ids) > 0:
                    errors.append(IntegrityError(
                        type='causal_link',
                        detail=f"Bridge edge {getattr(edge, 'edge_id', '?')}: "
                               f"execution_entry_id={exec_id} not found in journal",
                        entry_id=getattr(edge, 'edge_id', ''),
                    ))

        return errors

    def verify_lifecycle(self, journal: DispatchJournal) -> List[IntegrityError]:
        """Verify valid state machine transitions per execution_id."""
        errors = []

        # Group events by execution_id
        exec_events: Dict[str, List[JournalEntry]] = {}
        for entry in journal._entries:
            eid = entry.execution_id
            if eid:
                if eid not in exec_events:
                    exec_events[eid] = []
                exec_events[eid].append(entry)

        for exec_id, events in exec_events.items():
            # Sort by sequence index (not timestamp — sequence is canonical)
            events.sort(key=lambda e: _parse_seq(e.entry_id))

            # Check first event is a valid initial event
            if events and events[0].event not in _VALID_INITIAL_EVENTS:
                errors.append(IntegrityError(
                    type='lifecycle',
                    detail=f"Execution {exec_id}: invalid initial event "
                           f"'{events[0].event}'. "
                           f"Valid initial events: {sorted(_VALID_INITIAL_EVENTS)}",
                    entry_id=events[0].entry_id,
                ))

            for i in range(1, len(events)):
                prev_event = events[i - 1].event
                curr_event = events[i].event

                allowed = _VALID_TRANSITIONS.get(prev_event, [])
                if curr_event not in allowed:
                    errors.append(IntegrityError(
                        type='lifecycle',
                        detail=f"Execution {exec_id}: invalid transition "
                               f"'{prev_event}' -> '{curr_event}'. "
                               f"Allowed from '{prev_event}': {allowed}",
                        entry_id=events[i].entry_id,
                    ))

            # Check terminal event isn't followed by anything
            if events and events[-1].event in _TERMINAL_EVENTS:
                # It's fine — terminal event is the last event for this execution
                pass

        return errors

    def verify_integrity(
        self,
        journal: DispatchJournal,
        bridge_graph: Optional[Any] = None,
    ) -> IntegrityReport:
        """
        Run all 4 integrity checks and return a consolidated report.

        Args:
            journal: the DispatchJournal to verify
            bridge_graph: optional CausalityGraph for causal link checking

        Returns:
            IntegrityReport with per-check status + all errors
        """
        all_errors: List[IntegrityError] = []

        hash_errors = self.verify_hash_chain(journal)
        seq_errors = self.verify_sequence(journal)
        causal_errors = self.verify_causal_links(journal, bridge_graph)
        lifecycle_errors = self.verify_lifecycle(journal)

        all_errors.extend(hash_errors)
        all_errors.extend(seq_errors)
        all_errors.extend(causal_errors)
        all_errors.extend(lifecycle_errors)

        return IntegrityReport(
            valid=len(all_errors) == 0,
            hash_chain_ok=len(hash_errors) == 0,
            sequence_ok=len(seq_errors) == 0,
            causal_links_ok=len(causal_errors) == 0,
            lifecycle_ok=len(lifecycle_errors) == 0,
            errors=all_errors,
        )


def _parse_seq(entry_id: str) -> int:
    """Extract sequence index from entry_id: 'exec_id:idx:event' -> idx."""
    m = re.match(r'^.+:(\d+):.+$', entry_id)
    return int(m.group(1)) if m else -1
