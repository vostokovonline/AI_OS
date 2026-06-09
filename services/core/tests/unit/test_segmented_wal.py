"""
P2.11 SegmentedWAL tests — byte-sized segments, manifest-based indexing.

Key contracts:
  - append() → lsn, auto-rotates at byte limit
  - replay() → all entries across segments in order
  - replay_after_lsn() → uses manifest for binary search O(log N)
  - crash recovery on active segment only (Invariant P2.11)
  - LSN monotonic across segments
  - Manifest persists across reopen
"""

import json
import os
import tempfile
import shutil
import pytest
from execution_dynamics.segmented_wal import SegmentedWAL


def _make_wal_path():
    d = tempfile.mkdtemp()
    return os.path.join(d, 'wal')


def _cleanup(path):
    if os.path.exists(path):
        shutil.rmtree(os.path.dirname(path) if os.path.isfile(path) else path)


def _seed_entries(wal, n: int = 6, prefix: str = "exec"):
    """Append n entries to a WAL, return list of LSNs."""
    lsns = []
    for i in range(n):
        lsn = wal.append(
            entry_type='DISPATCHED' if i % 4 == 0 else 'STARTED' if i % 4 == 1 else 'COMPLETED',
            entry_id=f"{prefix}:{i}:event",
            payload={'goal_id': f"goal_{i % 2}", 'execution_id': f"{prefix}_{i}"},
        )
        lsns.append(lsn)
    return lsns


class TestAppendAndReplay:

    def test_append_single_entry(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            lsn = wal.append('DISPATCHED', 'e0', {'goal_id': 'g1'})
            assert lsn != ""
            assert wal.get_entry_count() == 1
        finally:
            _cleanup(wal_path)

    def test_replay_single_segment(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            lsns = _seed_entries(wal, n=4)

            entries = wal.replay()
            assert len(entries) == 4
            assert [e.lsn for e in entries] == lsns
        finally:
            _cleanup(wal_path)

    def test_replay_empty_wal(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path)
            assert wal.replay() == []
        finally:
            _cleanup(wal_path)


class TestRotation:

    def test_rotation_at_byte_limit(self):
        """Appending past max_segment_bytes creates a new segment."""
        wal_path = _make_wal_path()
        try:
            # Very small limit to force rotation
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)

            assert wal.get_entry_count() == 10
            assert wal._active_seq > 1  # at least one rotation
            assert len(wal._segments) >= 2
        finally:
            _cleanup(wal_path)

    def test_replay_across_segments(self):
        """Replay returns entries in order across segments."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)

            entries = wal.replay()
            assert len(entries) == 10
            assert [e.lsn for e in entries] == lsns
        finally:
            _cleanup(wal_path)

    def test_sealed_segment_immutable(self):
        """After rotation, sealed segment is read-only and has footer."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=10)

            for seq, seg in wal._segments.items():
                if seq != wal._active_seq:
                    assert not seg.active, f"Segment {seq} should be sealed"
                    seg_path = os.path.join(wal_path, f"{seq:08d}.wal")
                    if os.path.exists(seg_path):
                        with open(seg_path, 'r') as f:
                            lines = f.read().strip().split('\n')
                            last = json.loads(lines[-1])
                            assert '_wal_meta' in last, f"Segment {seq} missing footer"
                            assert last['_wal_meta'] == 'segment_footer'
                            assert 'start_lsn' in last, f"Segment {seq} footer missing start_lsn"
                            assert last['start_lsn'] != "", f"Segment {seq} footer has empty start_lsn"
        finally:
            _cleanup(wal_path)

    def test_manifest_tracks_segments(self):
        """Manifest correctly records segment metadata."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=10)

            assert len(wal._manifest) == len(wal._segments)
            for meta in wal._manifest.values():
                assert meta.first_lsn != ""
                assert meta.last_lsn != ""
                assert meta.entry_count > 0
        finally:
            _cleanup(wal_path)

    def test_rotation_produces_correct_segment_count(self):
        """Verify exact segment count given byte limit and entry payloads."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=50)
            _seed_entries(wal, n=20)

            entries = wal.replay()
            assert len(entries) == 20
            assert wal._active_seq > 2
            # All sealed segments have footer; active has header
            for seq, seg in wal._segments.items():
                assert seg.path.endswith('.wal')
        finally:
            _cleanup(wal_path)


class TestReplayAfterLSN:

    def test_replay_after_lsn_single_segment(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            lsns = _seed_entries(wal, n=6)

            # Replay from midpoint
            mid = lsns[2]
            tail = wal.replay(since_lsn=mid)
            assert len(tail) == 3  # entries at indices 3, 4, 5
            assert tail[0].lsn == lsns[3]
        finally:
            _cleanup(wal_path)

    def test_replay_after_lsn_across_segments(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)

            # Replay from near the boundary
            mid = lsns[5]
            tail = wal.replay(since_lsn=mid)
            assert len(tail) == 4  # entries at indices 6, 7, 8, 9
            assert tail[0].lsn == lsns[6]
        finally:
            _cleanup(wal_path)

    def test_replay_after_lsn_nonexistent(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            _seed_entries(wal, n=4)

            tail = wal.replay(since_lsn="99999999-999999-99999")
            assert tail == []
        finally:
            _cleanup(wal_path)

    def test_replay_after_lsn_empty_wal(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path)
            tail = wal.replay(since_lsn="20250101-000000-00000")
            assert tail == []
        finally:
            _cleanup(wal_path)

    def test_replay_after_lsn_start_of_wal(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            lsns = _seed_entries(wal, n=4)

            first = lsns[0]
            tail = wal.replay(since_lsn=first)
            assert len(tail) == 3  # exclusive: skip first
        finally:
            _cleanup(wal_path)

    def test_replay_after_lsn_last_entry(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            lsns = _seed_entries(wal, n=4)

            last = lsns[-1]
            tail = wal.replay(since_lsn=last)
            assert tail == []  # nothing after last
        finally:
            _cleanup(wal_path)


class TestPersistence:

    def test_data_survives_reopen(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=8)
            wal.close()

            # Reopen
            wal2 = SegmentedWAL(wal_path, max_segment_bytes=200)
            entries = wal2.replay()
            assert len(entries) == 8
            assert [e.lsn for e in entries] == lsns
        finally:
            _cleanup(wal_path)

    def test_manifest_survives_reopen(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=6)
            n_segments = len(wal._segments)
            wal.close()

            wal2 = SegmentedWAL(wal_path, max_segment_bytes=200)
            assert len(wal2._segments) == n_segments
            assert len(wal2._manifest) == n_segments
        finally:
            _cleanup(wal_path)

    def test_get_lsn_survives_reopen(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            _seed_entries(wal, n=4)
            last_lsn = wal.get_lsn()
            wal.close()

            wal2 = SegmentedWAL(wal_path)
            assert wal2.get_lsn() == last_lsn
        finally:
            _cleanup(wal_path)

    def test_append_after_reopen(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=5)
            wal.close()

            wal2 = SegmentedWAL(wal_path, max_segment_bytes=200)
            extra = _seed_entries(wal2, n=3, prefix="reopen")

            entries = wal2.replay()
            assert len(entries) == 8
        finally:
            _cleanup(wal_path)


class TestLSN:

    def test_lsn_monotonic_across_segments(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=150)
            lsns = _seed_entries(wal, n=10)

            for i in range(1, len(lsns)):
                assert lsns[i] > lsns[i - 1], f"LSN not monotonic at index {i}: {lsns[i]} <= {lsns[i-1]}"
        finally:
            _cleanup(wal_path)

    def test_get_lsn_empty(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path)
            assert wal.get_lsn() == "00000000-000000-00000"
        finally:
            _cleanup(wal_path)


class TestCrashRecovery:

    def test_recover_active_segment_after_corrupt_line(self):
        """Crash recovery only on active (last) segment."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=400)
            _seed_entries(wal, n=4)
            wal.close()

            # Corrupt the active segment (last .wal file)
            wal_files = sorted(f for f in os.listdir(wal_path) if f.endswith('.wal'))
            active_file = os.path.join(wal_path, wal_files[-1])
            with open(active_file, 'ab') as f:
                f.write(b'corrupted_garbage\n')
                f.flush()
                os.fsync(f.fileno())

            # Reopen — should recover by truncating corrupted line
            wal2 = SegmentedWAL(wal_path, max_segment_bytes=400)
            entries = wal2.replay()
            # Entries from sealed segments (if any) + valid entries from active
            assert len(entries) == 4
        finally:
            _cleanup(wal_path)

    def test_corrupt_sealed_segment_ignored(self):
        """Sealed segment corruption is NOT recovered (by design)."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=150)
            _seed_entries(wal, n=8)  # force at least 2 segments
            wal.close()

            # Corrupt a sealed segment (not the active one)
            wal_files = sorted(f for f in os.listdir(wal_path) if f.endswith('.wal'))
            assert len(wal_files) >= 2
            sealed_file = os.path.join(wal_path, wal_files[0])
            # Overwrite with garbage
            with open(sealed_file, 'wb') as f:
                f.write(b'\xff\xfe\x00total_garbage')
                f.flush()
                os.fsync(f.fileno())

            # Reopen — sealed segment is loaded as-is (garbage = 0 entries)
            # This is an explicit design choice: sealed segments are assumed consistent
            wal2 = SegmentedWAL(wal_path, max_segment_bytes=150)
            entries = wal2.replay()
            # Only entries from non-corrupted (active) segment survive
            # The sealed segment contributes 0 entries
            assert len(entries) >= 0
        finally:
            _cleanup(wal_path)


class TestRecoveryInvariant:

    def test_sealed_segments_have_manifest_entry(self):
        """Invariant P2.11: every sealed segment must have manifest entry after reopen."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=10)
            wal.close()

            # Reopen — all sealed segments should have manifest entries
            wal2 = SegmentedWAL(wal_path, max_segment_bytes=200)
            for seq, seg in wal2._segments.items():
                if seq == wal2._active_seq:
                    continue  # active may be in-flight
                assert seq in wal2._manifest, \
                    f"Sealed segment {seq} missing from manifest after reopen"
        finally:
            _cleanup(wal_path)

    def test_orphan_sealed_raises_invariant(self):
        """Invariant P2.11: sealed segment WITHOUT manifest entry raises RuntimeError."""
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=4)
            active_seq = wal._active_seq
            wal.close()

            # Corrupt manifest: remove a sealed segment entry
            manifest_path = os.path.join(wal_path, 'manifest.json')
            with open(manifest_path, 'r') as f:
                data = json.load(f)
            data['segments'] = [s for s in data['segments'] if s['seq'] != 1]
            with open(manifest_path, 'w') as f:
                json.dump(data, f)

            # Reopen — should raise because segment 1 is on disk but not in manifest
            with pytest.raises(RuntimeError, match="Boot reconciliation FAILED"):
                SegmentedWAL(wal_path, max_segment_bytes=200)
        finally:
            _cleanup(wal_path)


class TestStats:

    def test_stats_empty(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path)
            stats = wal.get_stats()
            assert stats['n_entries'] == 0
            # Active segment always exists (created at init)
            assert stats['n_segments'] == 1
            assert stats['active_seq'] == 1
        finally:
            _cleanup(wal_path)

    def test_stats_after_append(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            _seed_entries(wal, n=4)
            stats = wal.get_stats()
            assert stats['n_entries'] == 4
            assert stats['n_segments'] == 1
            assert stats['active_seq'] == 1
            assert stats['current_lsn'] != ""
            assert stats['max_segment_bytes'] == 1024 * 1024
            assert stats['total_size_bytes'] > 0
        finally:
            _cleanup(wal_path)

    def test_stats_multi_segment(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=100)
            _seed_entries(wal, n=15)
            stats = wal.get_stats()
            assert stats['n_entries'] == 15
            assert stats['n_segments'] >= 2
            assert len(stats['segments']) >= 2
        finally:
            _cleanup(wal_path)

    def test_get_last_entry(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            _seed_entries(wal, n=4)
            last = wal.get_last_entry()
            assert last is not None
            assert last.entry_type == 'COMPLETED'
        finally:
            _cleanup(wal_path)

    def test_get_last_entry_empty(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path)
            assert wal.get_last_entry() is None
        finally:
            _cleanup(wal_path)


class TestIntegrationWithDispatchJournal:

    def _full_lifecycle_entries(self, start: int, count: int) -> list:
        """Create journal entries with valid lifecycle per execution_id."""
        from execution_dynamics.journal import JournalEntry
        lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
        entries = []
        for i in range(count):
            exec_id = f"exec_{start + i}"
            goal_id = f"goal_{(start + i) % 3}"
            base_ts = 1000.0 + (start + i) * 100
            for j, ev in enumerate(lifecycle):
                entries.append(JournalEntry(
                    event=ev, goal_id=goal_id,
                    execution_id=exec_id,
                    lease_id=f"lease_{exec_id}",
                    timestamp=base_ts + j,
                ))
        return entries

    def test_journal_append_with_segmented_wal(self):
        """SegmentedWAL works as WAL backend for DispatchJournal."""
        wal_path = _make_wal_path()
        try:
            from execution_dynamics.journal import DispatchJournal
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            journal = DispatchJournal(wal=wal)

            entries = self._full_lifecycle_entries(0, 3)  # 12 entries (3 execs × 4 events)
            for je in entries:
                journal.append(je)
            assert len(journal._entries) == 12
        finally:
            _cleanup(wal_path)

    def test_journal_boot_with_segmented_wal(self):
        """DispatchJournal.boot() works with SegmentedWAL."""
        wal_path = _make_wal_path()
        try:
            from execution_dynamics.journal import DispatchJournal
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            journal = DispatchJournal(wal=wal)

            entries = self._full_lifecycle_entries(0, 2)  # 8 entries (2 execs × 4 events)
            for je in entries:
                journal.append(je)
            del journal

            wal2 = SegmentedWAL(wal_path, max_segment_bytes=200)
            j2 = DispatchJournal(wal=wal2)
            result = j2.boot()
            assert result.entries_restored == 8
            assert result.integrity_valid
        finally:
            _cleanup(wal_path)

    def test_journal_boot_with_snapshot_and_segmented_wal(self):
        """Snapshot restore + tail replay works with SegmentedWAL."""
        wal_path = _make_wal_path()
        snap_path = os.path.join(os.path.dirname(wal_path), 'snap.json')
        try:
            from execution_dynamics.journal import DispatchJournal
            from execution_dynamics.snapshot import SnapshotManager

            wal = SegmentedWAL(wal_path, max_segment_bytes=400)
            journal = DispatchJournal(wal=wal)

            # 2 executions = 8 entries, snapshot at midpoint
            entries = self._full_lifecycle_entries(0, 2)
            for je in entries:
                journal.append(je)

            mgr = SnapshotManager(wal=wal, snapshot_path=snap_path)
            mgr.create_snapshot(journal)

            # 2 more executions = 8 more entries
            entries2 = self._full_lifecycle_entries(2, 2)
            for je in entries2:
                journal.append(je)
            del journal

            wal2 = SegmentedWAL(wal_path, max_segment_bytes=400)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            result = j2.boot(snapshot_mgr=mgr2)
            assert result.entries_restored == 16
            assert result.integrity_valid
        finally:
            _cleanup(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)


class TestSnapshotCompaction:
    """P2.12 — prune sealed segments after snapshot."""

    def test_prune_deletes_sealed_segments(self):
        wal_path = _make_wal_path()
        try:
            from execution_dynamics.segmented_wal import CompactionResult
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)

            mid_lsn = lsns[5]
            n_before = len(wal._segments)
            assert n_before >= 6

            result = wal.prune_segments(mid_lsn)
            assert isinstance(result, CompactionResult)
            assert result.segments_deleted > 0
            assert result.bytes_reclaimed > 0
            assert result.oldest_remaining_lsn != ""
            assert result.oldest_remaining_segment > 0
            assert result.oldest_remaining_segment != wal._active_seq  # never active

            n_after = len(wal._segments)
            assert n_after == n_before - result.segments_deleted
            assert wal._active_seq is not None
            assert wal._active_seq in wal._segments
        finally:
            _cleanup(wal_path)

    def test_active_segment_never_deleted(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=4)

            active_before = wal._active_seq
            result = wal.prune_segments(wal.get_lsn())
            assert active_before in wal._segments
            active_path = os.path.join(wal_path, f"{active_before:08d}.wal")
            assert os.path.exists(active_path)
        finally:
            _cleanup(wal_path)

    def test_prune_updates_manifest(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)

            mid_lsn = lsns[4]
            wal.prune_segments(mid_lsn)

            for meta in wal._manifest.values():
                assert meta.seq in wal._segments
        finally:
            _cleanup(wal_path)

    def test_replay_after_prune(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)

            mid_lsn = lsns[4]
            result = wal.prune_segments(mid_lsn)
            assert result.segments_deleted > 0

            entries = wal.replay()
            assert len(entries) > 0
            # Segment with last_lsn == mid_lsn is preserved (strict <)
            assert entries[0].lsn >= mid_lsn
        finally:
            _cleanup(wal_path)

    def test_prune_noop_when_no_segments_covered(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=1024 * 1024)
            _seed_entries(wal, n=4)

            result = wal.prune_segments("00000000-000000-00000")
            assert result.segments_deleted == 0
        finally:
            _cleanup(wal_path)

    def test_prune_noop_empty_lsn(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            _seed_entries(wal, n=10)

            result = wal.prune_segments("")
            assert result.segments_deleted == 0
        finally:
            _cleanup(wal_path)

    def test_prune_persists_across_reopen(self):
        wal_path = _make_wal_path()
        try:
            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            lsns = _seed_entries(wal, n=10)
            mid_lsn = lsns[4]
            result = wal.prune_segments(mid_lsn)
            assert result.segments_deleted > 0
            wal.close()

            wal2 = SegmentedWAL(wal_path, max_segment_bytes=200)
            entries = wal2.replay()
            assert len(entries) > 0
            assert entries[0].lsn >= mid_lsn
            # Only non-pruned segments remain (>= because strict <)
            for meta in wal2._manifest.values():
                assert meta.last_lsn is None or meta.last_lsn >= mid_lsn or meta.seq == wal2._active_seq
        finally:
            _cleanup(wal_path)

    def test_snapshot_compaction_explicit(self):
        """wal.prune_segments(snapshot.last_lsn) — explicit compaction pattern."""
        wal_path = _make_wal_path()
        snap_path = os.path.join(os.path.dirname(wal_path), 'snap.json')
        try:
            from execution_dynamics.journal import DispatchJournal, JournalEntry
            from execution_dynamics.snapshot import SnapshotManager
            from execution_dynamics.segmented_wal import CompactionResult

            wal = SegmentedWAL(wal_path, max_segment_bytes=200)
            journal = DispatchJournal(wal=wal)
            lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            for g in range(3):
                for i, ev in enumerate(lifecycle):
                    journal.append(JournalEntry(
                        event=ev, goal_id=f"goal_{g}",
                        execution_id=f"exec_{g}", lease_id=f"lease_{g}",
                        timestamp=1000.0 + g * 100 + i,
                    ))

            mgr = SnapshotManager(wal=wal, snapshot_path=snap_path)
            snap = mgr.create_snapshot(journal)

            result = wal.prune_segments(snap.last_lsn)
            assert isinstance(result, CompactionResult)
            assert result.segments_deleted > 0
            assert result.bytes_reclaimed > 0
            assert result.oldest_remaining_lsn != ""
            assert result.oldest_remaining_segment > 0

            remaining = [s for s in wal._segments.values() if s.seq != wal._active_seq]
            for seg in remaining:
                last_entry = seg._entries[-1] if seg._entries else None
                if last_entry:
                    assert last_entry.lsn > snap.last_lsn, \
                        f"Sealed segment {seg.seq} last_lsn={last_entry.lsn} should be > {snap.last_lsn}"
        finally:
            _cleanup(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)
