"""
Truth Journal — append-only log of truth mutations.

SEPARATION OF CONCERNS:
  - DispatchJournal: records execution lifecycle (DISPATCHED, STARTED, COMPLETED...)
  - TruthJournal: records state mutations (status transition, artifact created...)

The kernel is the ONLY component that appends to TruthJournal.
Executors return proposed mutations; the kernel validates and commits them.

TRUTH ENTRY LIFECYCLE:
  1. PROPOSED — executor returned this mutation (not yet committed)
  2. COMMITTED — kernel validated and applied the mutation
  3. REJECTED — kernel rejected the mutation (validation failed)
  4. COMPENSATED — mutation was rolled back via compensation

STORAGE:
  - WAL (durable, append-only) for replayability
  - PostgreSQL for queryability
  - In-memory buffer for fast access
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
import logging
import time
import uuid

from .wal import WriteAheadLog

logger = logging.getLogger(__name__)


# ============================================================================
# Mutation Types (what can be proposed)
# ============================================================================

TRUTH_ENTRY_PROPOSED = 'proposed'
TRUTH_ENTRY_COMMITTED = 'committed'
TRUTH_ENTRY_REJECTED = 'rejected'
TRUTH_ENTRY_COMPENSATED = 'compensated'

TRUTH_ENTRY_STATES = {TRUTH_ENTRY_PROPOSED, TRUTH_ENTRY_COMMITTED,
                      TRUTH_ENTRY_REJECTED, TRUTH_ENTRY_COMPENSATED}


@dataclass
class ProposedMutation:
    """
    A single atomic truth mutation proposed by an executor.

    Types:
      - status_transition: goal.status → new_state
      - goal_update: goal.field → value (progress, trace, evaluation)
      - artifact_create: new artifact record
      - execution_record: new GoalExecution record
      - skill_stats_update: update skill statistics
      - raw_sql: direct SQL mutation (last resort)
    """
    mutation_type: str  # status_transition | goal_update | artifact_create | ...
    entity_type: str    # goal | artifact | execution | skill_stats | ...
    entity_id: str      # UUID of the entity being mutated
    field: str          # field name being changed (or 'new' for creates)
    old_value: Any = None
    new_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProposedEvent:
    """
    A domain event emitted by the executor during execution.

    Events are NOT mutations — they are signals about what happened.
    The kernel records them alongside the mutations for auditing.
    """
    event_type: str
    entity_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class Evidence:
    """
    Evidence supporting a proposed mutation.

    Examples:
      - Artifact content (file contents, LLM response)
      - Verification result (passed/failed checks)
      - Execution trace (steps taken, decisions made)
    """
    evidence_type: str  # artifact_content | verification_result | execution_trace | ...
    content: Any = None
    content_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Truth Entry (single record in the journal)
# ============================================================================

@dataclass
class TruthEntry:
    """
    Single entry in the Truth Journal.

    Each entry represents ONE atomic truth mutation that was
    proposed, committed, rejected, or compensated.
    """
    entry_id: str
    execution_id: str
    lease_id: str
    goal_id: str

    mutation: ProposedMutation
    events: List[ProposedEvent] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    state: str = TRUTH_ENTRY_PROPOSED
    timestamp: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'entry_id': self.entry_id,
            'execution_id': self.execution_id,
            'lease_id': self.lease_id,
            'goal_id': self.goal_id,
            'mutation_type': self.mutation.mutation_type,
            'entity_type': self.mutation.entity_type,
            'entity_id': self.mutation.entity_id,
            'field': self.mutation.field,
            'state': self.state,
            'timestamp': self.timestamp,
            'error': self.error,
            'events': [e.event_type for e in self.events],
            'evidence_count': len(self.evidence),
        }


# ============================================================================
# Truth Journal
# ============================================================================

class TruthJournal:
    """
    Append-only journal for truth mutations.

    Executors return proposed mutations.
    Kernel calls commit() to validate and persist.
    """

    def __init__(self, wal: Optional[Any] = None, db_session=None):
        self._wal = wal
        self._db = db_session
        self._entries: Dict[str, TruthEntry] = {}
        self._execution_entries: Dict[str, List[str]] = {}  # execution_id -> [entry_id]

    # ------------------------------------------------------------------
    # Propose (executor returns proposed mutations)
    # ------------------------------------------------------------------

    def propose(
        self,
        execution_id: str,
        lease_id: str,
        goal_id: str,
        mutations: List[ProposedMutation],
        events: Optional[List[ProposedEvent]] = None,
        evidence: Optional[List[Evidence]] = None,
    ) -> List[TruthEntry]:
        """
        Register proposed mutations from an executor.

        Returns list of TruthEntry objects (one per mutation).
        All entries start in PROPOSED state.
        """
        entries = []
        now = time.time()

        for mutation in mutations:
            entry_id = str(uuid.uuid4())
            entry = TruthEntry(
                entry_id=entry_id,
                execution_id=execution_id,
                lease_id=lease_id,
                goal_id=goal_id,
                mutation=mutation,
                events=events or [],
                evidence=evidence or [],
                state=TRUTH_ENTRY_PROPOSED,
                timestamp=now,
            )
            self._entries[entry_id] = entry
            entries.append(entry)

            if execution_id not in self._execution_entries:
                self._execution_entries[execution_id] = []
            self._execution_entries[execution_id].append(entry_id)

            # WAL: durable append
            if self._wal:
                self._wal.append(
                    'truth_proposed',
                    entry_id,
                    {
                        'execution_id': execution_id,
                        'goal_id': goal_id,
                        'mutation_type': mutation.mutation_type,
                        'entity_type': mutation.entity_type,
                        'entity_id': mutation.entity_id,
                        'field': mutation.field,
                        'old_value': str(mutation.old_value) if mutation.old_value else None,
                        'new_value': str(mutation.new_value) if mutation.new_value else None,
                        'timestamp': now,
                    },
                )

            # DB persist (if available)
            if self._db:
                self._persist_entry(entry)

        if entries:
            logger.info(
                f"truth_proposed execution_id={execution_id} "
                f"count={len(entries)} goal_id={goal_id}"
            )

        return entries

    # ------------------------------------------------------------------
    # Commit (kernel validates and persists)
    # ------------------------------------------------------------------

    def commit(self, entry_id: str) -> bool:
        """
        Commit a proposed truth entry.

        Validates that the entry is in PROPOSED state,
        then marks it as COMMITTED.

        Returns True if committed.
        """
        entry = self._entries.get(entry_id)
        if not entry:
            logger.warning(f"truth_entry_not_found entry_id={entry_id}")
            return False
        if entry.state != TRUTH_ENTRY_PROPOSED:
            logger.warning(f"truth_entry_not_proposed entry_id={entry_id} state={entry.state}")
            return False

        entry.state = TRUTH_ENTRY_COMMITTED
        entry.timestamp = time.time()

        if self._wal:
            self._wal.append(
                'truth_committed',
                entry_id,
                {
                    'execution_id': entry.execution_id,
                    'timestamp': entry.timestamp,
                },
            )

        if self._db:
            self._persist_state(entry_id, TRUTH_ENTRY_COMMITTED)

        return True

    def commit_all(self, execution_id: str) -> dict:
        """
        Commit all PROPOSED entries for an execution.

        Returns summary: {'committed': N, 'failed': N, 'errors': [...]}
        """
        entry_ids = self._execution_entries.get(execution_id, [])
        summary = {'committed': 0, 'failed': 0, 'errors': []}

        for eid in entry_ids:
            entry = self._entries.get(eid)
            if entry and entry.state == TRUTH_ENTRY_PROPOSED:
                if self.commit(eid):
                    summary['committed'] += 1
                else:
                    summary['failed'] += 1
                    summary['errors'].append(eid)

        return summary

    # ------------------------------------------------------------------
    # Reject (kernel validation failed)
    # ------------------------------------------------------------------

    def reject(self, entry_id: str, reason: str = "validation_failed") -> bool:
        """Reject a proposed mutation."""
        entry = self._entries.get(entry_id)
        if not entry or entry.state != TRUTH_ENTRY_PROPOSED:
            return False
        entry.state = TRUTH_ENTRY_REJECTED
        entry.error = reason
        if self._db:
            self._persist_state(entry_id, TRUTH_ENTRY_REJECTED)
        return True

    def reject_all(self, execution_id: str, reason: str = "execution_failed") -> dict:
        """Reject all PROPOSED entries for an execution."""
        entry_ids = self._execution_entries.get(execution_id, [])
        count = 0
        for eid in entry_ids:
            entry = self._entries.get(eid)
            if entry and entry.state == TRUTH_ENTRY_PROPOSED:
                if self.reject(eid, reason):
                    count += 1
        return {'rejected': count}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_entries(self, execution_id: Optional[str] = None) -> List[TruthEntry]:
        """Get entries, optionally filtered by execution_id."""
        if execution_id:
            entry_ids = self._execution_entries.get(execution_id, [])
            return [self._entries[eid] for eid in entry_ids if eid in self._entries]
        return list(self._entries.values())

    def get_stats(self) -> dict:
        """Get journal statistics."""
        states = {}
        for entry in self._entries.values():
            states[entry.state] = states.get(entry.state, 0) + 1
        return {
            'total_entries': len(self._entries),
            'total_executions': len(self._execution_entries),
            'state_counts': states,
        }

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_entry(self, entry: TruthEntry):
        """Persist truth entry to PostgreSQL."""
        if not self._db:
            return
        try:
            from models import TruthJournalEntry
            db_entry = TruthJournalEntry(
                entry_id=entry.entry_id,
                execution_id=entry.execution_id,
                lease_id=entry.lease_id,
                goal_id=entry.goal_id,
                mutation_type=entry.mutation.mutation_type,
                entity_type=entry.mutation.entity_type,
                entity_id=entry.mutation.entity_id,
                field=entry.mutation.field,
                old_value=str(entry.mutation.old_value) if entry.mutation.old_value else None,
                new_value=str(entry.mutation.new_value) if entry.mutation.new_value else None,
                state=entry.state,
                metadata=entry.mutation.metadata,
            )
            self._db.add(db_entry)
            self._db.flush()
        except Exception as e:
            logger.warning(f"truth_persist_failed entry_id={entry.entry_id} error={e}")

    def _persist_state(self, entry_id: str, state: str):
        """Update state of a truth entry in PostgreSQL."""
        if not self._db:
            return
        try:
            from sqlalchemy import update
            from models import TruthJournalEntry
            stmt = update(TruthJournalEntry).where(
                TruthJournalEntry.entry_id == entry_id
            ).values(state=state)
            self._db.execute(stmt)
            self._db.flush()
        except Exception as e:
            logger.warning(f"truth_state_update_failed entry_id={entry_id} error={e}")
