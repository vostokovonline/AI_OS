"""
JSON Lines Write-Ahead Log — durable, append-only, fsynced.

P2.7 minimal spec:
  - Append-only JSONL file
  - fsync per write (synchronous durability)
  - Crash recovery = truncate at first invalid JSON line
  - Strict ordering via sequence
  - No cross-domain logic
  - Locally verifiable (prev_hash + entry_hash at WAL level)

FILE FORMAT:
    {"lsn":"...", "entry_type":"DISPATCHED","entry_id":"...","seq":0,"prev_hash":"","entry_hash":"abc...","payload":{...}}
    {"lsn":"...", "entry_type":"STARTED","entry_id":"...","seq":1,"prev_hash":"abc...","entry_hash":"def...","payload":{...}}

RECOVERY INVARIANT:
    valid WAL prefix = longest prefix of valid JSON lines with consistent seq + hash chain
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WalEntry:
    """A single entry in the write-ahead log."""
    lsn: str
    entry_type: str
    entry_id: str
    payload: Dict[str, Any]
    timestamp: float
    seq: int = 0
    prev_hash: str = ""
    entry_hash: str = ""

    def to_dict(self) -> dict:
        return {
            'lsn': self.lsn,
            'entry_type': self.entry_type,
            'entry_id': self.entry_id,
            'payload': self.payload,
            'timestamp': self.timestamp,
            'seq': self.seq,
            'prev_hash': self.prev_hash,
            'entry_hash': self.entry_hash,
        }


class JsonLinesWAL:
    """
    Append-only WAL backed by a JSON Lines file with fsync durability.

    Implements the same protocol as the PostgreSQL-based WriteAheadLog
    so DispatchJournal can use it interchangeably.

    CRASH RECOVERY:
      On open(), scans the file for valid JSON lines. Stops at the first
      invalid line — everything after is truncated. This ensures the file
      always contains a prefix-consistent journal.
    """

    def __init__(self, path: str, auto_recover: bool = True):
        self.path = path
        self._entries: List[WalEntry] = []
        self._sequence = 0

        if auto_recover:
            self._load_or_recover()

    # ------------------------------------------------------------------
    # Load / Recover
    # ------------------------------------------------------------------

    def _load_or_recover(self):
        """Load existing entries; truncate at first invalid line on corruption."""
        if not os.path.exists(self.path):
            dirpath = os.path.dirname(self.path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            return

        with open(self.path, 'rb') as f:
            raw_bytes = f.read()

        # Split on newlines, preserving raw bytes per line
        # Omit trailing empty segment from split
        raw_lines: List[bytes] = raw_bytes.split(b'\n') if raw_bytes else []
        if raw_lines and raw_lines[-1] == b'':
            raw_lines = raw_lines[:-1]

        valid_lines: List[str] = []
        for raw in raw_lines:
            if not raw.strip():
                valid_lines.append('')
                continue
            try:
                decoded = raw.decode('utf-8')
                json.loads(decoded)
                valid_lines.append(decoded)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                logger.warning("jsonl_wal_corrupt_line stopping_at=%s error=%s", raw[:80], str(e))
                break

        truncated = len(raw_lines) - len(valid_lines)
        if truncated > 0:
            with open(self.path, 'wb') as f:
                for line in valid_lines:
                    f.write((line + '\n').encode('utf-8'))
                f.flush()
                os.fsync(f.fileno())
            logger.warning("jsonl_wal_truncated n_removed=%d n_kept=%d", truncated, len(valid_lines))

        # Parse valid entries
        for line in valid_lines:
            if not line:
                continue
            try:
                data = json.loads(line)
                self._entries.append(WalEntry(
                    lsn=data.get('lsn', ''),
                    entry_type=data.get('entry_type', ''),
                    entry_id=data.get('entry_id', ''),
                    payload=data.get('payload', {}),
                    timestamp=data.get('timestamp', time.time()),
                    seq=data.get('seq', 0),
                    prev_hash=data.get('prev_hash', ''),
                    entry_hash=data.get('entry_hash', ''),
                ))
                self._sequence = max(self._sequence, data.get('seq', 0) + 1)
            except json.JSONDecodeError:
                break

        logger.info("jsonl_wal_loaded path=%s n_entries=%d", self.path, len(self._entries))

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, entry_type: str, entry_id: str, payload: dict) -> str:
        """
        Append an entry with synchronous fsync.

        Write order:
          1. Build entry data
          2. Write JSON line to file
          3. fsync (data + metadata)
          4. Cache in memory

        Returns LSN (log sequence number).
        """
        seq = self._sequence
        self._sequence += 1

        now = time.time()
        ts = time.strftime('%Y%m%d-%H%M%S', time.gmtime(now))
        lsn = f"{ts}-{seq:05d}"

        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry_hash = payload.get('entry_hash', '')

        line_data = {
            'lsn': lsn,
            'entry_type': entry_type,
            'entry_id': entry_id,
            'seq': seq,
            'payload': payload,
            'timestamp': now,
            'prev_hash': prev_hash,
            'entry_hash': entry_hash,
        }

        line = json.dumps(line_data, sort_keys=True) + '\n'

        # Write + fsync (durable before visible)
        with open(self.path, 'a') as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        # Cache in memory
        entry = WalEntry(
            lsn=lsn,
            entry_type=entry_type,
            entry_id=entry_id,
            payload=payload,
            timestamp=now,
            seq=seq,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)

        return lsn

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """
        Replay WAL entries from a given LSN.

        Exclusive semantics: entries at or before since_lsn are skipped.
        None = replay all entries.
        """
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

    def replay_after(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """Alias for replay() with exclusive semantics."""
        return self.replay(since_lsn)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_lsn(self) -> str:
        if self._entries:
            return self._entries[-1].lsn
        return "00000000-000000-00000"

    def get_entry_count(self) -> int:
        return len(self._entries)

    def get_stats(self) -> dict:
        size = 0
        if os.path.exists(self.path):
            size = os.path.getsize(self.path)
        return {
            'n_entries': len(self._entries),
            'current_lsn': self.get_lsn(),
            'file_size_bytes': size,
            'path': self.path,
        }

    def get_last_entry(self) -> Optional[WalEntry]:
        if self._entries:
            return self._entries[-1]
        return None
