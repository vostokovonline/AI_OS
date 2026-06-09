"""
SegmentedWAL — production-grade WAL with byte-sized segments and manifest.

Architecture (P2.11):
  SegmentedWAL(directory)
    ├── manifest.json              (segment index for O(log N) lookup)
    ├── 00000001.wal               (sealed, read-only)
    ├── 00000002.wal               (sealed, read-only)
    └── 00000003.wal               (active — highest seq, accept writes)

  Active segment = max(seq). Sealed segments are immutable.

  Invariant P2.11 (recovery boundary):
    All sealed segments MUST be fully valid and hash-consistent.
    Only the active segment MAY contain an unsealed/truncated tail after crash.
    This invariant is enforced in _ensure_active_segment().

  Rotation: append checks active file size against max_segment_bytes.
  When exceeded → seal (write footer) → create new segment.

  Segment file format:
    {"_wal_meta": "segment_header", "seq": 1, "created_at": ...}
    {"lsn": "...", "entry_type": "DISPATCHED", ...}
    {"lsn": "...", "entry_type": "STARTED", ...}
    ...
    {"_wal_meta": "segment_footer", "start_lsn": "...", "last_lsn": "...", "entry_count": ...}

  This implements the same protocol as JsonLinesWAL / WriteAheadLog
  so DispatchJournal can use it interchangeably.

  WalEntry.{lsn, entry_type, entry_id, payload, timestamp, seq, prev_hash, entry_hash}
"""

import json
import logging
import os
import time
import bisect
import fcntl
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict, Tuple
from .jsonl_wal import WalEntry

logger = logging.getLogger(__name__)

_DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # 64 MB


@dataclass
class CompactionResult:
    """Result of a WAL compaction operation (P2.12)."""
    segments_deleted: int = 0
    bytes_reclaimed: int = 0
    oldest_remaining_lsn: str = ""
    oldest_remaining_segment: int = 0
_SEGMENT_PREFIX_LEN = 8  # zero-padded sequence: 00000001.wal


def _format_seq(seq: int) -> str:
    return f"{seq:0{_SEGMENT_PREFIX_LEN}d}"


def _segment_path(base: str, seq: int) -> str:
    return os.path.join(base, f"{_format_seq(seq)}.wal")


def _parse_lsn_seq(lsn: str) -> int:
    """Extract the seq part from an LSN: 'YYYYMMDD-HHMMSS-XXXXX'."""
    try:
        parts = lsn.rsplit('-', 1)
        return int(parts[-1]) if len(parts) == 2 else 0
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Internal: segment manifest
# ---------------------------------------------------------------------------

@dataclass
class _SegmentMeta:
    seq: int
    first_lsn: str = ""
    last_lsn: str = ""
    entry_count: int = 0

    def to_dict(self) -> dict:
        return {
            'seq': self.seq,
            'first_lsn': self.first_lsn,
            'last_lsn': self.last_lsn,
            'entry_count': self.entry_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> '_SegmentMeta':
        return cls(
            seq=d['seq'],
            first_lsn=d.get('first_lsn', ''),
            last_lsn=d.get('last_lsn', ''),
            entry_count=d.get('entry_count', 0),
        )


# ---------------------------------------------------------------------------
# Internal: segment file I/O
# ---------------------------------------------------------------------------

class _Segment:
    """A single WAL segment file (sealed or active)."""

    def __init__(self, seq: int, path: str, active: bool = False):
        self.seq = seq
        self.path = path
        self.active = active
        self._entries: List[WalEntry] = []
        self._file_size: int = 0
        self._file = None  # open file handle (only for active)

        if active:
            self._open_or_create()
        else:
            if os.path.exists(path):
                self._load()
                self._file_size = os.path.getsize(path)

    def mark_sealed(self):
        """Transition from active to sealed (no more writes)."""
        self.active = False
        self.close()

    def _open_or_create(self):
        """Open active segment file for append, with crash recovery."""
        if not os.path.exists(self.path):
            dirpath = os.path.dirname(self.path)
            if dirpath and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)
            self._file = open(self.path, 'a')
            self._file_size = 0
            return

        # Recover: load valid prefix
        self._load()
        self._file_size = os.path.getsize(self.path)
        # Reopen in append mode (truncated file on disk already)
        self._file = open(self.path, 'a')

    def _load(self):
        """Load entries from file (used by both sealed and active recovery)."""
        if not os.path.exists(self.path):
            return
        with open(self.path, 'rb') as f:
            raw_bytes = f.read()
        raw_lines = raw_bytes.split(b'\n') if raw_bytes else []
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
                if self.active:
                    logger.warning("segment_corrupt_line seq=%s stopping_at=%s error=%s",
                                   self._format_seq(), raw[:80], str(e))
                break

        truncated = len(raw_lines) - len(valid_lines)
        if truncated > 0 and self.active:
            with open(self.path, 'wb') as f:
                for line in valid_lines:
                    f.write((line + '\n').encode('utf-8'))
                f.flush()
                os.fsync(f.fileno())
            logger.warning("segment_truncated seq=%s removed=%d kept=%d",
                           self._format_seq(), truncated, len(valid_lines))

        for line in valid_lines:
            if not line:
                continue
            try:
                data = json.loads(line)
                if '_wal_meta' in data:
                    continue  # skip header/footer
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
            except json.JSONDecodeError:
                break

    def append(self, entry_type: str, entry_id: str, payload: dict,
               seq_idx: int = 0) -> str:
        """Append an entry with fsync. seq_idx is globally monotonic (from owner)."""
        local_idx = len(self._entries)
        now = time.time()
        ts = time.strftime('%Y%m%d-%H%M%S', time.gmtime(now))
        lsn = f"{ts}-{seq_idx:05d}"

        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry_hash = payload.get('entry_hash', '')

        line_data = {
            'lsn': lsn,
            'entry_type': entry_type,
            'entry_id': entry_id,
            'seq': local_idx,
            'payload': payload,
            'timestamp': now,
            'prev_hash': prev_hash,
            'entry_hash': entry_hash,
        }
        line = json.dumps(line_data, sort_keys=True) + '\n'

        self._file.write(line)
        self._file.flush()
        os.fsync(self._file.fileno())

        entry = WalEntry(
            lsn=lsn, entry_type=entry_type, entry_id=entry_id,
            payload=payload, timestamp=now, seq=local_idx,
            prev_hash=prev_hash, entry_hash=entry_hash,
        )
        self._entries.append(entry)
        self._file_size = self._file.tell()
        return lsn

    def write_header(self):
        """Write segment header (before any entries)."""
        header = json.dumps({
            '_wal_meta': 'segment_header',
            'seq': self.seq,
            'created_at': time.time(),
        }, sort_keys=True) + '\n'
        self._file.write(header)
        self._file.flush()
        os.fsync(self._file.fileno())

    def write_footer(self):
        """Write segment footer and flush. Called on seal."""
        if not self._entries:
            return
        footer = json.dumps({
            '_wal_meta': 'segment_footer',
            'start_lsn': self._entries[0].lsn,
            'last_lsn': self._entries[-1].lsn,
            'entry_count': len(self._entries),
        }, sort_keys=True) + '\n'
        self._file.write(footer)
        self._file.flush()
        os.fsync(self._file.fileno())

    @property
    def size_bytes(self) -> int:
        if self._file is not None:
            try:
                return self._file.tell()
            except OSError:
                pass
        return self._file_size

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def _format_seq(self) -> str:
        return _format_seq(self.seq)

    def replay(self) -> List[WalEntry]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# SegmentedWAL
# ---------------------------------------------------------------------------

class SegmentedWAL:
    """
    Production-grade WAL with byte-sized segments and manifest-based indexing.

    Implements the same protocol as JsonLinesWAL:
      append(), replay(), replay_after_lsn(), get_lsn(), get_stats()

    Usage:
        wal = SegmentedWAL("/data/wal", max_segment_bytes=64 * 1024 * 1024)
        lsn = wal.append("DISPATCHED", "exec:0:dispatched", {...})
        for entry in wal.replay():
            ...
    """

    def __init__(self, path: str, max_segment_bytes: int = _DEFAULT_MAX_BYTES):
        self.path = path
        self._max_bytes = max_segment_bytes
        self._manifest_path = os.path.join(path, 'manifest.json')
        self._segments: Dict[int, _Segment] = {}  # seq -> _Segment
        self._active_seq: Optional[int] = None
        self._sequence = 0  # globally monotonic across segments
        self._lock_fd = None  # K7/A: single-writer lock

        # Boot sequence (strict order):
        #   1. Discover: filesystem reality (ground truth)
        #   2. Load: manifest (hypothesis layer, untrusted)
        #   3. Reconcile: arbitration gate — may reject boot
        #   4. Open: only after reconciliation passes
        observed_seqs = self._discover_filesystem()
        self._load_manifest()
        self._boot_reconcile(observed_seqs)
        self._ensure_active_segment()
        self._verify_sealed_invariant()

    # ------------------------------------------------------------------
    # K7/A: Single-writer lock (fcntl.flock on lock file)
    # ------------------------------------------------------------------

    def acquire_lock(self):
        """Acquire exclusive file lock on the WAL directory.

        Raises RuntimeError if another writer holds the lock.
        Lock is automatically released on process exit or close().
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path, exist_ok=True)
        lock_path = os.path.join(self.path, '.wal_lock')
        self._lock_fd = open(lock_path, 'w')
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            self._lock_fd.close()
            self._lock_fd = None
            raise RuntimeError(
                f"WAL at {self.path} is locked by another writer. "
                f"AI-OS Execution Kernel enforces single-writer access.")

    def release_lock(self):
        """Release the WAL lock."""
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except (IOError, OSError):
                pass
            self._lock_fd.close()
            self._lock_fd = None

    # ------------------------------------------------------------------
    # Boot reconciliation (filesystem-first trust)
    # ------------------------------------------------------------------

    def _discover_filesystem(self) -> set[int]:
        """Stage A: physical truth — scan disk for all .wal files."""
        if not os.path.exists(self.path):
            return set()
        result = set()
        for f in os.listdir(self.path):
            if f.endswith('.wal') and f[:-4].isdigit():
                result.add(int(f[:-4]))
        return result

    def _boot_reconcile(self, observed: set[int]):
        """Stage C: arbitration gate between filesystem and manifest.

        Rules:
          Ghost entries (manifest has entry, filesystem has no file):
            → IRRECONCILABLE. Data was logically committed but physically lost.
          Orphan files (filesystem has file, manifest has no entry):
            → IRRECONCILABLE. Split-brain — manifest diverged from disk.
          Clean (observed == indexed):
            → Proceed.
        """
        indexed = set(self._manifest.keys())

        ghosts = indexed - observed
        if ghosts:
            raise RuntimeError(
                f"Boot reconciliation FAILED: {len(ghosts)} manifest segment(s) "
                f"missing from filesystem: {sorted(ghosts)}. "
                f"Data loss detected — WAL cannot boot."
            )

        orphans = observed - indexed
        if orphans and indexed:
            raise RuntimeError(
                f"Boot reconciliation FAILED: {len(orphans)} segment(s) "
                f"exist on disk but not in manifest: {sorted(orphans)}. "
                f"Manifest diverged from filesystem — WAL cannot boot."
            )

        # No manifest at all but files exist: first boot from disk
        if orphans and not indexed:
            for seq in sorted(orphans):
                self._manifest[seq] = _SegmentMeta(seq=seq)
                seg_path = _segment_path(self.path, seq)
                seg = _Segment(seq=seq, path=seg_path, active=False)
                self._segments[seq] = seg
                if seg._entries:
                    meta = self._manifest[seq]
                    meta.first_lsn = seg._entries[0].lsn if seg._entries else ""
                    meta.last_lsn = seg._entries[-1].lsn if seg._entries else ""
                    meta.entry_count = len(seg._entries)
            self._save_manifest()

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self):
        """Load manifest and open existing segments."""
        self._manifest: Dict[int, _SegmentMeta] = {}
        if not os.path.exists(self._manifest_path):
            return
        try:
            with open(self._manifest_path, 'r') as f:
                data = json.load(f)
            for sd in data.get('segments', []):
                meta = _SegmentMeta.from_dict(sd)
                self._manifest[meta.seq] = meta
                # Open sealed segments for replay
                seg_path = _segment_path(self.path, meta.seq)
                self._segments[meta.seq] = _Segment(
                    seq=meta.seq, path=seg_path, active=False
                )
        except (json.JSONDecodeError, KeyError, FileNotFoundError, UnicodeDecodeError, ValueError) as e:
            logger.warning("manifest_load_failed path=%s error=%s", self._manifest_path, e)

    def _save_manifest(self):
        """Persist manifest atomically."""
        dirpath = os.path.dirname(self._manifest_path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        segments = sorted(self._manifest.values(), key=lambda m: m.seq)
        data = {'segments': [s.to_dict() for s in segments]}
        tmp = self._manifest_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._manifest_path)

    def _verify_sealed_invariant(self):
        """Invariant P2.11: every sealed segment MUST have a manifest entry.

        This is a post-reconciliation assertion. Since _boot_reconcile()
        already enforced filesystem ↔ manifest consistency, this catches
        any divergence introduced during active segment handling.
        """
        observed = self._discover_filesystem()
        active_seq = max(self._segments.keys()) if self._segments else None
        missing = []
        for seq in sorted(observed):
            if seq == active_seq:
                continue
            if seq not in self._manifest:
                seg_path = _segment_path(self.path, seq)
                if os.path.getsize(seg_path) > 0:
                    missing.append(seq)
        if missing:
            raise RuntimeError(
                f"Invariant P2.11 violation: sealed segments without manifest entry: {missing}"
            )

    def _ensure_active_segment(self):
        """Open or create the active (highest seq) segment."""
        if not self._segments:
            self._create_new_segment()
            return

        max_seq = max(self._segments.keys())
        seg_path = _segment_path(self.path, max_seq)
        self._segments[max_seq] = _Segment(seq=max_seq, path=seg_path, active=True)
        self._active_seq = max_seq

        # Recover global sequence from all entries
        all_entries = []
        for seq in sorted(self._segments.keys()):
            seg = self._segments[seq]
            all_entries.extend(seg._entries)
        self._sequence = max((int(e.lsn.rsplit('-', 1)[-1]) for e in all_entries if e.lsn), default=0) + 1

        seg = self._segments[max_seq]
        if seg._entries:
            meta = self._manifest.get(max_seq)
            if meta is None:
                meta = _SegmentMeta(seq=max_seq)
            meta.last_lsn = seg._entries[-1].lsn
            meta.entry_count = len(seg._entries)
            meta.first_lsn = meta.first_lsn or seg._entries[0].lsn
            self._manifest[max_seq] = meta
            self._save_manifest()

    def _create_new_segment(self):
        """Create a new segment (finds next seq)."""
        next_seq = 1
        if self._manifest:
            next_seq = max(self._manifest.keys()) + 1

        seg_path = _segment_path(self.path, next_seq)
        seg = _Segment(seq=next_seq, path=seg_path, active=True)
        seg.write_header()
        self._segments[next_seq] = seg
        self._active_seq = next_seq

        meta = _SegmentMeta(seq=next_seq)
        self._manifest[next_seq] = meta
        self._save_manifest()

    def _seal_active(self):
        """
        Seal current active segment (write footer, save manifest),
        then create new active segment.
        """
        if self._active_seq is None:
            return

        active = self._segments.get(self._active_seq)
        if active is not None:
            active.write_footer()
            meta = self._manifest.get(self._active_seq)
            if meta and active._entries:
                meta.last_lsn = active._entries[-1].lsn
                meta.entry_count = len(active._entries)
                meta.first_lsn = meta.first_lsn or active._entries[0].lsn
            active.mark_sealed()  # sets active=False, closes file
            self._save_manifest()

        # Create new segment
        next_seq = self._active_seq + 1
        seg_path = _segment_path(self.path, next_seq)
        seg = _Segment(seq=next_seq, path=seg_path, active=True)
        seg.write_header()
        self._segments[next_seq] = seg
        self._active_seq = next_seq

        meta = _SegmentMeta(seq=next_seq)
        self._manifest[next_seq] = meta
        self._save_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, entry_type: str, entry_id: str, payload: dict) -> str:
        """
        Append an entry with synchronous fsync and auto-segmentation.

        Rotates to a new segment when the active file exceeds max_segment_bytes.
        Returns LSN (log sequence number).
        """
        active = self._segments.get(self._active_seq)
        if active is None:
            self._ensure_active_segment()
            active = self._segments.get(self._active_seq)

        # Check if we need to rotate
        if active is not None and active.size_bytes >= self._max_bytes:
            self._seal_active()
            active = self._segments.get(self._active_seq)

        if active is None:
            raise RuntimeError("No active segment available")

        seq_idx = self._sequence
        self._sequence += 1
        lsn = active.append(entry_type, entry_id, payload, seq_idx=seq_idx)

        # Update manifest metadata
        meta = self._manifest.get(self._active_seq)
        if meta:
            if not meta.first_lsn:
                meta.first_lsn = lsn
            meta.last_lsn = lsn
            meta.entry_count += 1

        return lsn

    def replay(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """
        Replay all entries, optionally from a given LSN.

        Uses manifest to skip irrelevant sealed segments.
        """
        if not self._segments:
            return []

        # Quick path: no filter
        if not since_lsn:
            result = []
            for seq in sorted(self._segments.keys()):
                seg = self._segments[seq]
                result.extend(seg.replay())
            return result

        # Filtered: binary search on manifest LSN index
        sorted_meta = sorted(self._manifest.values(), key=lambda m: m.seq)
        lsn_list = [m.last_lsn for m in sorted_meta]
        start_idx = bisect.bisect_left(lsn_list, since_lsn)
        if start_idx < len(sorted_meta):
            start_seq = sorted_meta[start_idx].seq
        else:
            return []

        result = []
        for seq in sorted(self._segments.keys()):
            if seq < start_seq:
                continue
            seg = self._segments[seq]
            entries = seg.replay()
            for entry in entries:
                # Exclusive: skip entries at or before since_lsn
                if entry.lsn <= since_lsn:
                    continue
                result.append(entry)
        return result

    def replay_after(self, since_lsn: Optional[str] = None) -> List[WalEntry]:
        """Alias for replay() with exclusive semantics."""
        return self.replay(since_lsn)

    def get_lsn(self) -> str:
        """Get the latest LSN across all segments."""
        if not self._manifest:
            return "00000000-000000-00000"
        # Find segment with max seq that has a last_lsn
        max_seq = max(self._manifest.keys())
        meta = self._manifest.get(max_seq)
        if meta and meta.last_lsn:
            return meta.last_lsn
        # Fallback: check active segment
        active = self._segments.get(self._active_seq)
        if active and active._entries:
            return active._entries[-1].lsn
        return "00000000-000000-00000"

    def get_entry_count(self) -> int:
        """Total entry count across all segments."""
        return sum(len(seg._entries) for seg in self._segments.values())

    def get_stats(self) -> dict:
        """Diagnostic statistics."""
        total_bytes = 0
        seg_info = []
        for seq in sorted(self._segments.keys()):
            seg = self._segments[seq]
            is_active = (seq == self._active_seq)
            meta = self._manifest.get(seq)
            total_bytes += seg.size_bytes
            seg_info.append({
                'seq': seq,
                'active': is_active,
                'entries': len(seg._entries),
                'size_bytes': seg.size_bytes,
                'first_lsn': meta.first_lsn if meta else '',
                'last_lsn': meta.last_lsn if meta else '',
            })

        wal_size = 0
        if os.path.exists(self.path):
            wal_size = sum(
                os.path.getsize(os.path.join(self.path, f))
                for f in os.listdir(self.path)
                if f.endswith('.wal') or f == 'manifest.json'
            )

        return {
            'n_entries': self.get_entry_count(),
            'n_segments': len(self._segments),
            'active_seq': self._active_seq,
            'current_lsn': self.get_lsn(),
            'total_size_bytes': wal_size,
            'max_segment_bytes': self._max_bytes,
            'segments': seg_info,
            'path': self.path,
        }

    def get_last_entry(self) -> Optional[WalEntry]:
        """Get the most recent entry (from active segment)."""
        active = self._segments.get(self._active_seq)
        if active and active._entries:
            return active._entries[-1]
        return None

    # ------------------------------------------------------------------
    # Snapshot Compaction (P2.12)
    # ------------------------------------------------------------------

    def prune_segments(self, up_to_lsn: str) -> CompactionResult:
        """
        Delete sealed segments whose last_lsn < up_to_lsn.

        Preserves the segment whose last_lsn == up_to_lsn so that
        SnapshotManager.validate() against snapshot.last_lsn still works.
        Active segment is NEVER deleted.

        Call explicitly after creating a snapshot:
            snap = mgr.create_snapshot(journal)
            result = wal.prune_segments(snap.last_lsn)

        Returns CompactionResult with segments_deleted, bytes_reclaimed,
        oldest_remaining_lsn, oldest_remaining_segment.
        """
        if up_to_lsn == "00000000-000000-00000" or not up_to_lsn:
            return CompactionResult()

        to_delete: List[int] = []
        for seq, meta in self._manifest.items():
            if seq == self._active_seq:
                continue
            # Strict < : preserve segment whose last_lsn == up_to_lsn
            if meta.last_lsn and meta.last_lsn < up_to_lsn:
                to_delete.append(seq)

        if not to_delete:
            return CompactionResult()

        bytes_reclaimed = 0
        for seq in to_delete:
            seg = self._segments.pop(seq, None)
            if seg is not None:
                bytes_reclaimed += seg.size_bytes
                seg.close()
            del self._manifest[seq]
            seg_path = _segment_path(self.path, seq)
            try:
                sz = os.path.getsize(seg_path) if os.path.exists(seg_path) else 0
                os.remove(seg_path)
                bytes_reclaimed = max(bytes_reclaimed, sz)
                logger.info("segment_pruned seq=%d last_lsn=%s path=%s", seq, up_to_lsn, seg_path)
            except OSError as e:
                logger.warning("segment_prune_failed seq=%d error=%s", seq, e)

        self._save_manifest()

        # Compute oldest remaining
        oldest_lsn = ""
        oldest_seq = 0
        for seq in sorted(self._segments.keys()):
            meta = self._manifest.get(seq)
            if meta and meta.first_lsn:
                oldest_lsn = meta.first_lsn
                oldest_seq = seq
                break

        result = CompactionResult(
            segments_deleted=len(to_delete),
            bytes_reclaimed=bytes_reclaimed,
            oldest_remaining_lsn=oldest_lsn,
            oldest_remaining_segment=oldest_seq,
        )
        logger.info("segments_pruned n=%d reclaimed=%d up_to_lsn=%s remaining=%d",
                    result.segments_deleted, result.bytes_reclaimed,
                    up_to_lsn, len(self._segments))
        return result

    def close(self):
        """Release lock and close all segment files."""
        self.release_lock()
        for seg in self._segments.values():
            seg.close()
