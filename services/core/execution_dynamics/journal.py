"""
Dispatch Journal — immutable, append-only causal chain for execution dispatch.

Every execution event (DISPATCHED, STARTED, COMPLETED, FAILED, PREEMPTED,
RETRIED, ABANDONED) is recorded with full context:

  - Execution pressure snapshot (why this goal was chosen)
  - Coordination field snapshot (what the execution context looked like)
  - Lease ID (authority token)
  - Execution ID (correlation ID)
  - Goal ID (which goal)
  - Timestamp (when)

The journal enables:
  - Causal reconstruction of execution decisions
  - Execution audit (who decided what and why)
  - Replay for debugging and learning
  - Recovery after system restart
  - Future RL training dataset generation

STORAGE:
  - Priority: PostgreSQL (durable) with in-memory cache (fast access)
  - Redis is NOT used — journal is source of truth, not ephemeral
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class BootResult:
    """Result of the DispatchJournal boot sequence (P2.9 + P2.10 metrics)."""
    method: str                         # 'snapshot_restore' or 'full_replay'
    entries_restored: int               # total entries in journal after boot
    tail_replayed: int                  # entries replayed from WAL tail (0 for full_replay)
    duration_ms: float                  # wall-clock time for boot
    integrity_valid: bool               # passed IntegrityVerifier?
    integrity_checks: Dict[str, bool] = field(default_factory=dict)  # per-check status

    # Recovery metrics (P2.10) — set when source files are accessible
    snapshot_size_bytes: int = 0        # bytes on disk (0 if no snapshot)
    wal_size_bytes: int = 0             # bytes on disk
    snapshot_entry_count: int = 0       # entries in snapshot (0 if no snapshot)
    wal_entry_count: int = 0            # total entries in WAL

    def to_dict(self) -> dict:
        return {
            'method': self.method,
            'entries_restored': self.entries_restored,
            'tail_replayed': self.tail_replayed,
            'duration_ms': round(self.duration_ms, 2),
            'integrity_valid': self.integrity_valid,
            'integrity_checks': dict(self.integrity_checks),
            'snapshot_size_bytes': self.snapshot_size_bytes,
            'wal_size_bytes': self.wal_size_bytes,
            'snapshot_entry_count': self.snapshot_entry_count,
            'wal_entry_count': self.wal_entry_count,
        }


# ============================================================================
# Event types (immutable enum-like set)
# ============================================================================

DISPATCH_EVENTS = {
    'DISPATCHED',      # Kernel accepted dispatch request
    'LEASE_ISSUED',    # Execution lease was issued
    'STARTED',         # Execution began (executor started)
    'COMPLETED',       # Execution finished successfully
    'FAILED',          # Execution finished with error
    'PREEMPTED',       # Execution was preempted by higher-pressure goal
    'RETRIED',         # Execution was retried after failure
    'ABANDONED',       # Execution was abandoned (max retries exceeded)
    'CANCELLING',      # Cooperative cancellation requested (not terminal)
    'CANCELLED',       # Execution was cancelled (user or system)
    'LEASE_EXPIRED',   # Lease timed out without completion
    'LEASE_REVOKED',   # Lease was explicitly revoked
}


# ============================================================================
# Journal Entry
# ============================================================================

@dataclass
class JournalEntry:
    """
    Single immutable entry in the dispatch journal.

    Each entry is append-only. Once written, it is never modified.
    Includes cryptographic hash chain for integrity verification.
    """
    event: str                # From DISPATCH_EVENTS
    goal_id: str
    execution_id: str
    lease_id: str
    timestamp: float

    # Causal context
    dispatch_epoch: int = 0
    group_id: Optional[str] = None

    # Why this decision was made (snapshot at decision time)
    pressure_snapshot: Optional[Dict[str, float]] = None
    execution_score: float = 0.0
    field_snapshot: Optional[Dict[str, float]] = None

    # Outcome
    success: Optional[bool] = None
    duration_ms: float = 0.0
    error: Optional[str] = None

    # Previous event in causal chain (for traversal)
    prev_entry_id: Optional[str] = None

    # Lease state embedded for atomic lease+journal persistence
    lease_state: Optional[Dict[str, Any]] = None

    # Idempotent dispatch (P2.12a)
    dispatch_id: str = ""

    entry_id: str = ""

    # Hash chain — for integrity verification
    prev_hash: str = ""       # SHA256 of previous entry (global journal order)
    entry_hash: str = ""      # SHA256 of this entry = sha256(prev_hash + entry_id + execution_id + event + payload_canonical)

    def compute_hash(self, prev_hash: str = "") -> str:
        """Compute SHA256 hash of this entry."""
        import hashlib, json
        payload = self._hash_payload()
        raw = f"{prev_hash}|{self.entry_id}|{self.execution_id}|{self.event}|{json.dumps(payload, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode('utf-8')).hexdigest()

    def _hash_payload(self) -> dict:
        """Canonical payload for hash computation (excludes hash fields)."""
        d = self.to_dict()
        d.pop('entry_hash', None)
        d.pop('prev_hash', None)
        return d

    def to_dict(self) -> dict:
        d = {
            'entry_id': self.entry_id,
            'event': self.event,
            'goal_id': self.goal_id,
            'execution_id': self.execution_id,
            'lease_id': self.lease_id,
            'timestamp': self.timestamp,
            'dispatch_epoch': self.dispatch_epoch,
            'dispatch_id': self.dispatch_id,
            'group_id': self.group_id,
            'pressure_snapshot': self.pressure_snapshot,
            'execution_score': self.execution_score,
            'field_snapshot': self.field_snapshot,
            'success': self.success,
            'duration_ms': self.duration_ms,
            'error': self.error,
            'prev_entry_id': self.prev_entry_id,
            'prev_hash': self.prev_hash,
            'entry_hash': self.entry_hash,
        }
        if self.lease_state:
            d['lease_state'] = self.lease_state
        return d


# ============================================================================
# Dispatch Journal
# ============================================================================

class DispatchJournal:
    """
    Immutable, append-only journal of all execution dispatch events.

    PROPERTIES:
      - Append-only: events are never modified after writing
      - Causal chains: each event links to previous event for the same goal
      - Temporal index: events can be queried by time range
      - Goal index: all events for a specific goal form a causal chain

    USAGE:
        journal = DispatchJournal()
        journal.append(JournalEntry(event='DISPATCHED', goal_id='...', ...))

        # Get causal chain for a goal
        chain = journal.get_chain('goal_123')

        # Get latest status for a goal
        latest = journal.get_latest('goal_123')
    """

    def __init__(self, db_session=None, wal=None):
        self._db = db_session
        self._wal = wal  # Write-Ahead Log (source of truth)
        self._entries: List[JournalEntry] = []
        self._goal_index: Dict[str, List[str]] = {}  # goal_id -> [entry_ids]
        self._last_lsn: Optional[str] = None

    # ------------------------------------------------------------------
    # Append (write-through WAL)
    # ------------------------------------------------------------------

    def append(self, entry: JournalEntry) -> str:
        """
        Append an immutable journal entry with cryptographic hash chain.

        WRITE ORDER (durability guarantee):
          1. Compute hash chain (prev_hash + entry_hash)
          2. WAL append (durable storage)
          3. In-memory cache update

        Returns the entry_id for reference.
        Raises ValueError if event type is invalid.
        """
        if entry.event not in DISPATCH_EVENTS:
            raise ValueError(f"Invalid dispatch event: {entry.event}. Must be one of {DISPATCH_EVENTS}")

        if not entry.entry_id:
            entry.entry_id = f"{entry.execution_id}:{len(self._entries)}:{entry.event.lower()}"

        # Link to previous event for this goal
        goal_entries = self._goal_index.get(entry.goal_id, [])
        if goal_entries:
            entry.prev_entry_id = goal_entries[-1]

        # Compute hash chain
        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry.prev_hash = prev_hash
        entry.entry_hash = entry.compute_hash(prev_hash)

        # 1. WAL append FIRST (durable storage)
        if self._wal:
            self._last_lsn = self._wal.append(
                entry_type=entry.event,
                entry_id=entry.entry_id,
                payload=entry.to_dict(),
            )

        # 2. In-memory cache SECOND
        self._entries.append(entry)
        if entry.goal_id not in self._goal_index:
            self._goal_index[entry.goal_id] = []
        self._goal_index[entry.goal_id].append(entry.entry_id)

        # Legacy DB persistence (deprecated — WAL is now source of truth)
        if self._db and not self._wal:
            self._persist(entry)

        return entry.entry_id

    # ------------------------------------------------------------------
    # Recovery from WAL
    # ------------------------------------------------------------------

    def recover_from_wal(self) -> int:
        """
        Reconstruct full in-memory journal state by replaying the WAL.

        This is deterministic — given the same WAL, the result is identical.

        Steps:
          1. Clear in-memory state
          2. Replay all WAL entries in order
          3. Rebuild _entries, _goal_index from replayed data

        Returns the number of entries recovered.
        """
        if not self._wal:
            return 0

        # Clear in-memory state
        self._entries = []
        self._goal_index = {}

        # Replay WAL entries
        wal_entries = self._wal.replay()
        for wal_entry in wal_entries:
            payload = wal_entry.payload
            je = JournalEntry(
                event=payload.get('event', wal_entry.entry_type),
                goal_id=payload.get('goal_id', ''),
                execution_id=payload.get('execution_id', ''),
                lease_id=payload.get('lease_id', ''),
                timestamp=payload.get('timestamp', wal_entry.timestamp),
                dispatch_epoch=payload.get('dispatch_epoch', 0),
                group_id=payload.get('group_id'),
                dispatch_id=payload.get('dispatch_id', ''),
                pressure_snapshot=payload.get('pressure_snapshot'),
                execution_score=payload.get('execution_score', 0.0),
                field_snapshot=payload.get('field_snapshot'),
                success=payload.get('success'),
                duration_ms=payload.get('duration_ms', 0.0),
                error=payload.get('error'),
                prev_entry_id=payload.get('prev_entry_id'),
                entry_id=wal_entry.entry_id,
                lease_state=payload.get('lease_state'),
                prev_hash=payload.get('prev_hash', '') or getattr(wal_entry, 'prev_hash', '') or '',
                entry_hash=payload.get('entry_hash', '') or getattr(wal_entry, 'entry_hash', '') or '',
            )
            self._entries.append(je)
            if je.goal_id not in self._goal_index:
                self._goal_index[je.goal_id] = []
            self._goal_index[je.goal_id].append(je.entry_id)

        logger.info(f"journal_recovered_from_wal n_entries={len(self._entries)} n_goals={len(self._goal_index)}")
        return len(self._entries)

    def boot(self, snapshot_mgr=None) -> BootResult:
        """
        Full boot sequence for deterministic journal recovery.

        Flow:
          1. Snapshot restore (if snapshot_mgr provided and snapshot exists)
          2. Fallback to full WAL replay if no snapshot
          3. IntegrityVerifier on restored state
          4. Return BootResult with timing and integrity status

        The resulting journal is guaranteed:
          - Deterministic replay (same WAL → same state)
          - Continuous hash chain
          - Passes IntegrityVerifier (hash_chain, sequence, lifecycle)

        Args:
            snapshot_mgr: optional SnapshotManager for fast recovery.
                          If None or no snapshot, falls back to full_replay.

        Returns:
            BootResult with method used, entry counts, timing, integrity status,
            and recovery metrics (file sizes, counts).
        """
        import os as _os
        start = time.time()

        # Phase 1: Recover journal state
        tail_replayed = 0
        method = 'full_replay'
        pre_entries = len(self._entries)
        wal_entry_count = 0

        if snapshot_mgr is not None and self._wal is not None:
            tail_replayed = snapshot_mgr.restore(self)

        if len(self._entries) > pre_entries:
            method = 'snapshot_restore'
        else:
            self.recover_from_wal()
            method = 'full_replay'
            tail_replayed = 0

        # Phase 2: Integrity verification
        from .integrity import IntegrityVerifier
        v = IntegrityVerifier()
        report = v.verify_integrity(self)

        duration_ms = (time.time() - start) * 1000.0

        # Phase 3: Recovery metrics (P2.10)
        snapshot_size_bytes = 0
        snapshot_entry_count = 0
        wal_bytes = 0
        if snapshot_mgr is not None:
            snap_path = snapshot_mgr._snapshot_path
            if snap_path and _os.path.exists(snap_path):
                snapshot_size_bytes = _os.path.getsize(snap_path)
            loaded = snapshot_mgr.load_snapshot()
            if loaded is not None:
                snapshot_entry_count = loaded.entry_count
        if self._wal and hasattr(self._wal, 'path'):
            try:
                wal_bytes = _os.path.getsize(self._wal.path)
            except (FileNotFoundError, OSError):
                pass
        if self._wal:
            wal_entry_count = len(self._wal.replay())

        return BootResult(
            method=method,
            entries_restored=len(self._entries),
            tail_replayed=tail_replayed,
            duration_ms=duration_ms,
            integrity_valid=report.valid,
            integrity_checks={
                'hash_chain_ok': report.hash_chain_ok,
                'sequence_ok': report.sequence_ok,
                'lifecycle_ok': report.lifecycle_ok,
            },
            snapshot_size_bytes=snapshot_size_bytes,
            wal_size_bytes=wal_bytes,
            snapshot_entry_count=snapshot_entry_count,
            wal_entry_count=wal_entry_count,
        )

    def _persist(self, entry: JournalEntry):
        """Legacy persist — only used when WAL is not available."""
        try:
            from models import DispatchJournalEntry as DBEntry
            db_entry = DBEntry(
                entry_id=entry.entry_id,
                event=entry.event,
                goal_id=entry.goal_id,
                execution_id=entry.execution_id,
                lease_id=entry.lease_id,
                timestamp=entry.timestamp,
                dispatch_epoch=entry.dispatch_epoch,
                group_id=entry.group_id,
                pressure_snapshot=entry.pressure_snapshot,
                execution_score=entry.execution_score,
                field_snapshot=entry.field_snapshot,
                success=entry.success,
                duration_ms=entry.duration_ms,
                error=entry.error,
                prev_entry_id=entry.prev_entry_id,
            )
            self._db.add(db_entry)
            self._db.flush()
        except Exception as e:
            logger.warning(f"journal_persist_failed entry_id={entry.entry_id} error={e}")

    # ------------------------------------------------------------------
    # Read (causal chain traversal)
    # ------------------------------------------------------------------

    def get_chain(self, goal_id: str) -> List[JournalEntry]:
        """
        Get the full causal chain for a goal, in chronological order.

        Returns all events for this goal from first dispatch to latest.
        """
        entry_ids = self._goal_index.get(goal_id, [])
        id_map = {e.entry_id: e for e in self._entries}
        return [id_map[eid] for eid in entry_ids if eid in id_map]

    def get_latest(self, goal_id: str) -> Optional[JournalEntry]:
        """Get the most recent journal entry for a goal."""
        entry_ids = self._goal_index.get(goal_id, [])
        if not entry_ids:
            return None
        id_map = {e.entry_id: e for e in self._entries}
        latest_id = entry_ids[-1]
        return id_map.get(latest_id)

    def get_latest_event(self, goal_id: str) -> Optional[str]:
        """Get the most recent event type for a goal (e.g., 'COMPLETED', 'FAILED')."""
        latest = self.get_latest(goal_id)
        return latest.event if latest else None

    def get_events_since(self, since_timestamp: float) -> List[JournalEntry]:
        """Get all events since a given timestamp."""
        return [e for e in self._entries if e.timestamp >= since_timestamp]

    def get_recent(self, limit: int = 100) -> List[JournalEntry]:
        """Get the N most recent entries across all goals."""
        return self._entries[-limit:]

    # ------------------------------------------------------------------
    # Query by lease
    # ------------------------------------------------------------------

    def get_by_lease(self, lease_id: str) -> List[JournalEntry]:
        """Get all events for a specific lease."""
        return [e for e in self._entries if e.lease_id == lease_id]

    def get_lease_status(self, lease_id: str) -> Optional[str]:
        """Get the latest event for a lease."""
        events = self.get_by_lease(lease_id)
        return events[-1].event if events else None

    # ------------------------------------------------------------------
    # Chain analysis
    # ------------------------------------------------------------------

    def get_chain_duration_ms(self, goal_id: str) -> float:
        """
        Get total execution duration from DISPATCHED to COMPLETED/FAILED.
        """
        chain = self.get_chain(goal_id)
        if not chain:
            return 0.0

        start = next((e for e in chain if e.event == 'DISPATCHED'), None)
        end = next((e for e in reversed(chain)
                     if e.event in ('COMPLETED', 'FAILED', 'ABANDONED')), None)

        if start and end:
            return (end.timestamp - start.timestamp) * 1000
        return 0.0

    def get_retry_count(self, goal_id: str) -> int:
        """Count how many times a goal has been retried."""
        chain = self.get_chain(goal_id)
        return sum(1 for e in chain if e.event == 'RETRIED')

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        wal_stats = self._wal.get_stats() if self._wal else {'n_entries': 0, 'has_db': False}
        if not self._entries:
            return {
                'total_entries': 0,
                'n_goals': 0,
                'wal': wal_stats,
                'last_lsn': self._last_lsn,
            }

        event_counts = {}
        for e in self._entries:
            event_counts[e.event] = event_counts.get(e.event, 0) + 1

        return {
            'total_entries': len(self._entries),
            'n_goals': len(self._goal_index),
            'event_counts': event_counts,
            'oldest_entry': self._entries[0].timestamp if self._entries else 0,
            'newest_entry': self._entries[-1].timestamp if self._entries else 0,
            'wal': wal_stats,
            'last_lsn': self._last_lsn,
        }
