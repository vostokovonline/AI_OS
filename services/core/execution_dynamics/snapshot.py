"""
Snapshot Manager — materialized Journal state for fast recovery.

P2.8 architecture:
  Snapshot = materialized Journal state (entries + goal_index + metadata)
  WAL     = source of truth

  Recovery:
    restore(snapshot) + replay_tail → state

  Invariant:
    state_now == restore(snapshot) + replay(WAL_from_LSN_forward)
    IntegrityVerifier passes on restored journal
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class JournalSnapshot:
    """Materialized Journal state at a point in time.

    Stores serialized entries so restore produces a fully functional Journal
    with intact hash chain, goal_index, and all query methods.
    """
    last_lsn: str                       # LSN of the last entry included
    last_entry_hash: str                # entry_hash of the last entry (redundant with entries[-1])
    goal_index: Dict[str, List[str]]    # reconstructed goal index
    entry_count: int                    # number of entries in the snapshot
    entries: List[dict] = field(default_factory=list)  # serialized JournalEntry dicts
    snapshot_timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            'last_lsn': self.last_lsn,
            'last_entry_hash': self.last_entry_hash,
            'goal_index': self.goal_index,
            'entry_count': self.entry_count,
            'entries': self.entries,
            'snapshot_timestamp': self.snapshot_timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'JournalSnapshot':
        return cls(
            last_lsn=d['last_lsn'],
            last_entry_hash=d['last_entry_hash'],
            goal_index=d.get('goal_index', {}),
            entry_count=d.get('entry_count', 0),
            entries=d.get('entries', []),
            snapshot_timestamp=d.get('snapshot_timestamp', 0.0),
        )


class SnapshotManager:
    """Manages creation, storage, loading, and validation of snapshots.

    Usage:
        mgr = SnapshotManager(wal=jsonl_wal, snapshot_path="/tmp/exec_snapshot.json")
        mgr.create_snapshot(journal)
        mgr.restore(journal)  # returns restored journal with full state
    """

    def __init__(self, wal=None, snapshot_path: str = "",
                 db_session=None, snapshot_interval: int = 0):
        self._wal = wal
        self._snapshot_path = snapshot_path
        self._db = db_session
        self._snapshot_interval = snapshot_interval  # P2.9+
        if db_session is not None:
            logger.info("snapshot_manager_db_session_deprecated")

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_snapshot(self, journal) -> JournalSnapshot:
        """Create a snapshot from current journal state (entries + metadata).

        Prune sealed segments explicitly after creating the snapshot:

            snap = mgr.create_snapshot(journal)
            if hasattr(wal, 'prune_segments'):
                result = wal.prune_segments(snap.last_lsn)
        """
        if not journal._entries:
            raise ValueError("Cannot create snapshot from empty journal")

        last_entry = journal._entries[-1]
        last_lsn = journal._last_lsn or (self._wal.get_lsn() if self._wal else '')
        snapshot = JournalSnapshot(
            last_lsn=last_lsn,
            last_entry_hash=last_entry.entry_hash,
            goal_index=dict(journal._goal_index),
            entry_count=len(journal._entries),
            entries=[e.to_dict() for e in journal._entries],
            snapshot_timestamp=__import__('time').time(),
        )
        self._write(snapshot)
        logger.info("snapshot_created lsn=%s entries=%d", last_lsn, snapshot.entry_count)
        return snapshot

    def _write(self, snapshot: JournalSnapshot):
        dirpath = os.path.dirname(self._snapshot_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(self._snapshot_path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_snapshot(self) -> Optional[JournalSnapshot]:
        """Load snapshot from file. Returns None if not found."""
        if not os.path.exists(self._snapshot_path):
            return None
        try:
            with open(self._snapshot_path, 'r') as f:
                data = json.load(f)
            return JournalSnapshot.from_dict(data)
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.warning("snapshot_load_failed error=%s", e)
            return None

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate(self, snapshot: JournalSnapshot) -> bool:
        """
        Validate snapshot against the current WAL.

        Checks that WAL[last_lsn].entry_hash matches snapshot.last_entry_hash.
        """
        if not self._wal:
            return False

        entries = self._wal.replay()
        for wal_entry in entries:
            if wal_entry.lsn == snapshot.last_lsn:
                stored_hash = (wal_entry.entry_hash
                               or wal_entry.payload.get('entry_hash', ''))
                if stored_hash == snapshot.last_entry_hash:
                    return True
                logger.warning(
                    "snapshot_hash_mismatch expected=%s.. got=%s..",
                    snapshot.last_entry_hash[:16], stored_hash[:16],
                )
                return False

        logger.warning("snapshot_lsn_not_found lsn=%s", snapshot.last_lsn)
        return False

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self, journal) -> int:
        """
        Restore full journal state from snapshot + WAL tail replay.

        Loads snapshot entries into journal._entries, then replays
        WAL entries after snapshot.last_lsn. The resulting journal
        has a continuous hash chain and passes IntegrityVerifier.

        Returns number of tail entries replayed (0 if snapshot at end).
        Returns 0 if no snapshot found (caller should use full WAL recovery).
        """
        snapshot = self.load_snapshot()
        if snapshot is None:
            return 0

        if not self.validate(snapshot):
            logger.warning("snapshot_validation_failed lsn=%s", snapshot.last_lsn)
            return 0

        # Reconstruct JournalEntry objects from snapshot entries
        from .journal import JournalEntry
        journal._entries = []
        for ed in snapshot.entries:
            je = JournalEntry(
                event=ed.get('event', ''),
                goal_id=ed.get('goal_id', ''),
                execution_id=ed.get('execution_id', ''),
                lease_id=ed.get('lease_id', ''),
                timestamp=ed.get('timestamp', 0.0),
                dispatch_epoch=ed.get('dispatch_epoch', 0),
                group_id=ed.get('group_id'),
                pressure_snapshot=ed.get('pressure_snapshot'),
                execution_score=ed.get('execution_score', 0.0),
                field_snapshot=ed.get('field_snapshot'),
                success=ed.get('success'),
                duration_ms=ed.get('duration_ms', 0.0),
                error=ed.get('error'),
                prev_entry_id=ed.get('prev_entry_id'),
                entry_id=ed.get('entry_id', ''),
                lease_state=ed.get('lease_state'),
                prev_hash=ed.get('prev_hash', ''),
                entry_hash=ed.get('entry_hash', ''),
            )
            journal._entries.append(je)

        journal._goal_index = dict(snapshot.goal_index)
        journal._last_lsn = snapshot.last_lsn

        # Replay WAL entries from snapshot point forward
        tail_entries = []
        if self._wal:
            wal_entries = self._wal.replay()
            found = False
            for wal_entry in wal_entries:
                if not found:
                    if wal_entry.lsn == snapshot.last_lsn:
                        found = True
                    continue
                payload = wal_entry.payload
                je = JournalEntry(
                    event=payload.get('event', wal_entry.entry_type),
                    goal_id=payload.get('goal_id', ''),
                    execution_id=payload.get('execution_id', ''),
                    lease_id=payload.get('lease_id', ''),
                    timestamp=payload.get('timestamp', wal_entry.timestamp),
                    dispatch_epoch=payload.get('dispatch_epoch', 0),
                    group_id=payload.get('group_id'),
                    pressure_snapshot=payload.get('pressure_snapshot'),
                    execution_score=payload.get('execution_score', 0.0),
                    field_snapshot=payload.get('field_snapshot'),
                    success=payload.get('success'),
                    duration_ms=payload.get('duration_ms', 0.0),
                    error=payload.get('error'),
                    prev_entry_id=payload.get('prev_entry_id'),
                    entry_id=wal_entry.entry_id,
                    lease_state=payload.get('lease_state'),
                    prev_hash=payload.get('prev_hash', '') or wal_entry.prev_hash or '',
                    entry_hash=payload.get('entry_hash', '') or wal_entry.entry_hash or '',
                )
                journal._entries.append(je)
                if je.goal_id not in journal._goal_index:
                    journal._goal_index[je.goal_id] = []
                journal._goal_index[je.goal_id].append(je.entry_id)
                tail_entries.append(je)

        journal._last_lsn = self._wal.get_lsn() if self._wal else snapshot.last_lsn
        logger.info(
            "snapshot_restored snapshot_entries=%d tail_replayed=%d total=%d",
            snapshot.entry_count, len(tail_entries), len(journal._entries),
        )
        return len(tail_entries)

    # ------------------------------------------------------------------
    # COMPATIBILITY PATH STUBS (frozen, rollback only)
    #
    # These exist for the legacy ExecutionKernel path (WriteAheadLog mode).
    # They are no-ops — new code should use create_snapshot() / load_snapshot() / restore().
    # Do not extend. Do not improve.
    # See CLAUDE.md "Persistence Layer: Kernel Migration" for removal policy.
    # ------------------------------------------------------------------

    def write_snapshot(self, **kwargs):
        """COMPATIBILITY PATH — use create_snapshot(journal) instead."""
        if not hasattr(self, '_deprecated_warned'):
            logger.warning("snapshot_write_deprecated_use_create_snapshot")
            self._deprecated_warned = True
        return None

    def load_latest(self):
        """COMPATIBILITY PATH — use load_snapshot() instead."""
        if not hasattr(self, '_deprecated_warned'):
            logger.warning("snapshot_load_latest_deprecated_use_load_snapshot")
            self._deprecated_warned = True
        return None

    def should_snapshot(self, *args) -> bool:
        """COMPATIBILITY PATH — no-op. Always returns False."""
        return False

    def prune(self, **kwargs):
        """COMPATIBILITY PATH — no-op."""
        pass


# ------------------------------------------------------------------
# COMPATIBILITY PATH — kept for legacy ExecutionKernel snapshot()
# ------------------------------------------------------------------

def build_kernel_state(kernel) -> dict:
    """
    COMPATIBILITY PATH — returns empty state dict.

    Previously reconstructed kernel metadata for snapshots.
    The new SnapshotManager.create_snapshot() materializes
    Journal state directly instead.
    """
    logger.warning("build_kernel_state_deprecated_use_create_snapshot")
    return {
        'active_executions': {},
        'seen_dispatches': {},
        'capability_epoch': 0,
    }
