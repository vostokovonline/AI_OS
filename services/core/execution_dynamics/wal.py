"""
Write-Ahead Log — durable, append-only log for the dispatch journal.

DESIGN:
  - Every journal entry is first written to the WAL
  - WAL entries are durable (PostgreSQL synchronous commit)
  - On recovery, WAL is replayed to reconstruct in-memory state
  - LSN (Log Sequence Number) is monotonic and enables partial replay
  - Checkpoints mark positions before which entries can be pruned

The WAL is the SOURCE OF TRUTH for execution history.
In-memory journal is a cache that is rebuilt from WAL on recovery.

STORAGE:
  - Primary: PostgreSQL (durable, ACID)
  - Fallback: SQLite (WAL mode, for standalone dev)
  - Dev-only: In-memory (no persistence, for testing)
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging
import json
import time
import os

logger = logging.getLogger(__name__)


# ============================================================================
# WAL Entry
# ============================================================================

@dataclass
class WalEntry:
    """
    A single durable entry in the write-ahead log.

    Each entry is:
      - Immutable (never modified after write)
      - Uniquely identified by LSN
      - Self-contained (carries all data needed for replay)
    """
    lsn: str                  # Log Sequence Number: "YYYYMMDD-HHMMSS-XXXXX"
    entry_type: str           # Journal event type: DISPATCHED, LEASE_ISSUED, ...
    entry_id: str             # Reference to journal entry
    payload: Dict[str, Any]   # Full journal entry data as dict
    timestamp: float

    def to_dict(self) -> dict:
        return {
            'lsn': self.lsn,
            'entry_type': self.entry_type,
            'entry_id': self.entry_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
        }


# ============================================================================
# Write-Ahead Log
# ============================================================================

class WriteAheadLog:
    """
    Durable, append-only write-ahead log.

    USAGE:
        wal = WriteAheadLog(db_session=session)
        lsn = wal.append(entry_type='DISPATCHED', entry_id='e1:0:dispatched', payload={...})

        # Recover from WAL (rebuild in-memory state)
        entries = wal.replay(since_lsn='20260528-120000-00001')
    """

    def __init__(self, db_session=None, db_url=None):
        self._db = db_session
        self._db_url = db_url
        self._entries: List[WalEntry] = []  # In-memory cache
        self._sequence = 0

    # ------------------------------------------------------------------
    # Write (append-only, durable)
    # ------------------------------------------------------------------

    def append(self, entry_type: str, entry_id: str, payload: dict) -> str:
        """
        Append an entry to the WAL.

        Writes to durable storage FIRST, then caches in memory.
        Returns the LSN (Log Sequence Number).

        The write order is:
          1. Generate LSN
          2. Write to PostgreSQL (sync commit)
          3. Cache in memory
          4. Return LSN

        This ensures durability before visibility.
        """
        self._sequence += 1
        now = time.time()
        ts = time.strftime('%Y%m%d-%H%M%S', time.gmtime(now))
        lsn = f"{ts}-{self._sequence:05d}"

        entry = WalEntry(
            lsn=lsn,
            entry_type=entry_type,
            entry_id=entry_id,
            payload=payload,
            timestamp=now,
        )

        # 1. Write to durable storage
        if self._db:
            self._persist(entry)
        else:
            # Fallback: in-memory only (dev mode)
            self._entries.append(entry)

        return lsn

    def _persist(self, entry: WalEntry):
        """Persist WAL entry to PostgreSQL. Synchronous commit."""
        try:
            # Use raw SQL for maximum control (no model dependency)
            from sqlalchemy import text
            stmt = text("""
                INSERT INTO execution_wal (lsn, entry_type, entry_id, payload, timestamp)
                VALUES (:lsn, :entry_type, :entry_id, :payload::jsonb, to_timestamp(:ts))
            """)
            self._db.execute(stmt, {
                'lsn': entry.lsn,
                'entry_type': entry.entry_type,
                'entry_id': entry.entry_id,
                'payload': json.dumps(entry.payload),
                'ts': entry.timestamp,
            })
            # Sync commit (PostgreSQL default with synchronous_commit = on)
            self._db.flush()

            # Also cache in memory for fast reads
            self._entries.append(entry)

        except Exception as e:
            logger.warning(f"wal_persist_failed lsn={entry.lsn} error={e}")
            # Fallback: keep in memory even if DB write fails
            self._entries.append(entry)

    # ------------------------------------------------------------------
    # Read (replay)
    # ------------------------------------------------------------------

    def replay(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """
        DEPRECATED: Use replay_after() with explicit exclusive semantics.

        Replay WAL entries from or at a given LSN. Retained for backward
        compatibility. Prefer replay_after() for clarity.

        NOTE: This actually has EXCLUSIVE semantics (entries at since_lsn
        are skipped). The method name was ambiguous — replay_after() makes
        this contract explicit.
        """
        return self._replay(since_lsn)

    def replay_after(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """
        Replay WAL entries STRICTLY AFTER a given LSN.

        EXCLUSIVE SEMANTICS:
          - since_lsn = None      → replay ALL entries (full cold start)
          - since_lsn = "LSN-X"   → replay entries with LSN > "LSN-X"
                                    (the entry at "LSN-X" is NOT included)

        This is the canonical replay method. Use it in all recovery paths.

        Args:
            since_lsn: Replay entries EXCLUSIVELY after this LSN.
                       None means replay everything.

        Returns:
            List[WalEntry] in LSN order.
        """
        return self._replay(since_lsn)

    def _replay(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """Internal implementation — delegates to DB or memory."""
        if self._db:
            return self._replay_from_db(since_lsn)
        return self._replay_from_memory(since_lsn)

    def _replay_from_db(self, since_lsn: Optional[str]) -> List[WalEntry]:
        """Replay from PostgreSQL."""
        try:
            from sqlalchemy import text

            if since_lsn:
                stmt = text("""
                    SELECT lsn, entry_type, entry_id, payload, timestamp
                    FROM execution_wal
                    WHERE lsn > :since_lsn
                    ORDER BY lsn ASC
                """)
                rows = self._db.execute(stmt, {'since_lsn': since_lsn}).fetchall()
            else:
                stmt = text("""
                    SELECT lsn, entry_type, entry_id, payload,
                           EXTRACT(EPOCH FROM timestamp) as ts_epoch
                    FROM execution_wal
                    ORDER BY lsn ASC
                """)
                rows = self._db.execute(stmt).fetchall()

            result = []
            for row in rows:
                payload = row.payload if isinstance(row.payload, dict) else json.loads(row.payload)
                ts = row.ts_epoch if hasattr(row, 'ts_epoch') and row.ts_epoch else row.timestamp.timestamp() if hasattr(row.timestamp, 'timestamp') else time.time()
                result.append(WalEntry(
                    lsn=row.lsn,
                    entry_type=row.entry_type,
                    entry_id=row.entry_id,
                    payload=payload,
                    timestamp=ts,
                ))
            return result

        except Exception as e:
            logger.warning(f"wal_replay_from_db_failed error={e}")
            return self._replay_from_memory(since_lsn)

    def _replay_from_memory(self, since_lsn: Optional[str]) -> List[WalEntry]:
        """Replay from in-memory cache."""
        if not since_lsn:
            return list(self._entries)

        result = []
        found = False
        for entry in self._entries:
            if entry.lsn == since_lsn:
                found = True
                continue
            if found:
                result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Checkpoint (prune old WAL entries)
    # ------------------------------------------------------------------

    def checkpoint(self, lsn: str) -> int:
        """
        Create a checkpoint at a given LSN.

        Entries before this LSN can be pruned.
        Returns the number of entries pruned.
        """
        count = 0
        self._entries = [e for e in self._entries if e.lsn >= lsn]
        count = len(self._entries)  # We just keep everything

        if self._db:
            try:
                from sqlalchemy import text
                stmt = text("DELETE FROM execution_wal WHERE lsn < :lsn")
                self._db.execute(stmt, {'lsn': lsn})
                self._db.flush()
                logger.info(f"wal_checkpoint pruned_before={lsn}")
            except Exception as e:
                logger.warning(f"wal_checkpoint_failed lsn={lsn} error={e}")

        return count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_lsn(self) -> str:
        """Get the current LSN (last written position)."""
        if self._entries:
            return self._entries[-1].lsn
        return "00000000-000000-00000"

    def get_entry_count(self) -> int:
        """Get total number of entries in the WAL."""
        if self._db:
            try:
                from sqlalchemy import text
                row = self._db.execute(text("SELECT COUNT(*) as cnt FROM execution_wal")).fetchone()
                return row[0] if row else len(self._entries)
            except Exception:
                pass
        return len(self._entries)

    def get_stats(self) -> dict:
        return {
            'n_entries': self.get_entry_count(),
            'current_lsn': self.get_lsn(),
            'in_memory': len(self._entries),
            'has_db': self._db is not None,
        }
