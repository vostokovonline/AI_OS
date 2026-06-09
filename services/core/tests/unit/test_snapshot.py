"""
P2.8 Snapshot tests — full Journal state materialization.

Core invariant:
    state_hash(full_replay(WAL)) == state_hash(restore(snapshot + tail_replay))

Where state_hash covers:
    - All entries (hash chain intact)
    - goal_index
    - last_lsn

IntegrityVerifier passes on restored journal.
"""

import json
import os
import tempfile
import hashlib
from typing import Optional
from execution_dynamics.jsonl_wal import JsonLinesWAL
from execution_dynamics.journal import DispatchJournal, JournalEntry
from execution_dynamics.snapshot import SnapshotManager, JournalSnapshot
from execution_dynamics.integrity import IntegrityVerifier


def build_full_journal(path: str, n_goals: int = 3, events_per_goal: int = 4) -> DispatchJournal:
    """Create a journal with multiple goals and write to WAL."""
    wal = JsonLinesWAL(path)
    journal = DispatchJournal(wal=wal)

    lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
    for g in range(n_goals):
        exec_id = f"exec_{g}"
        for i, ev in enumerate(lifecycle):
            je = JournalEntry(
                event=ev,
                goal_id=f"goal_{g}",
                execution_id=exec_id,
                lease_id=f"lease_{exec_id}",
                timestamp=1000.0 + g * 100 + i,
            )
            journal.append(je)
    return journal


def journal_state_hash(journal: DispatchJournal) -> str:
    """Deterministic SHA256 of journal's full queryable state."""
    raw = json.dumps({
        'entry_count': len(journal._entries),
        'entries': [e.to_dict() for e in journal._entries],
        'goal_index': journal._goal_index,
        'last_lsn': journal._last_lsn,
    }, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def compute_full_replay_hash(wal_path: str) -> str:
    """Full WAL replay hash for comparison."""
    wal = JsonLinesWAL(wal_path)
    j = DispatchJournal(wal=wal)
    j.recover_from_wal()
    return journal_state_hash(j)


class TestCreateSnapshot:

    def test_create_snapshot_basic(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=2)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)
            snap = mgr.create_snapshot(journal)

            assert snap.entry_count == 8
            assert snap.last_lsn != ""
            assert snap.last_entry_hash != ""
            assert len(snap.goal_index) == 2
            assert 'goal_0' in snap.goal_index
            assert 'goal_1' in snap.goal_index
            assert len(snap.entries) == 8
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_create_snapshot_empty_journal_raises(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            wal = JsonLinesWAL(wal_path)
            journal = DispatchJournal(wal=wal)
            mgr = SnapshotManager(wal=wal, snapshot_path=snap_path)
            try:
                mgr.create_snapshot(journal)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_snapshot_persists_and_loads(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=2)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)
            mgr.create_snapshot(journal)

            loaded = mgr.load_snapshot()
            assert loaded is not None
            assert loaded.entry_count == 8
            assert loaded.last_entry_hash == journal._entries[-1].entry_hash
            assert len(loaded.entries) == 8
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_snapshot_entries_match_journal(self):
        """Every entry in snapshot matches the source journal entry_hash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=2)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)
            snap = mgr.create_snapshot(journal)

            for i, (src, sd) in enumerate(zip(journal._entries, snap.entries)):
                assert src.entry_hash == sd['entry_hash'], \
                    f"Entry {i} hash mismatch: {src.entry_hash[:12]} != {sd['entry_hash'][:12]}"
                assert src.entry_id == sd['entry_id']
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)


class TestValidateSnapshot:

    def test_validate_valid_snapshot(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=1)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)
            snap = mgr.create_snapshot(journal)

            assert mgr.validate(snap) is True
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_validate_corrupted_snapshot_hash(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=1)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)
            snap = mgr.create_snapshot(journal)

            bad_snap = JournalSnapshot(
                last_lsn=snap.last_lsn,
                last_entry_hash='0' * 64,
                goal_index=snap.goal_index,
                entry_count=snap.entry_count,
                entries=snap.entries,
            )
            assert mgr.validate(bad_snap) is False
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_validate_nonexistent_lsn(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=1)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)

            snap = JournalSnapshot(
                last_lsn='NONEXISTENT',
                last_entry_hash='hash',
                goal_index={},
                entry_count=0,
                entries=[],
            )
            assert mgr.validate(snap) is False
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)


class TestRestoreSnapshot:

    def test_restore_produces_same_state_as_full_replay(self):
        """Core invariant: snapshot + tail_replay == full_replay (bit-exact)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            original = build_full_journal(wal_path, n_goals=3, events_per_goal=4)
            full_hash = journal_state_hash(original)
            del original

            wal_for_snap = JsonLinesWAL(wal_path)
            j_for_snap = DispatchJournal(wal=wal_for_snap)
            j_for_snap.recover_from_wal()
            mgr = SnapshotManager(wal=wal_for_snap, snapshot_path=snap_path)
            snap = mgr.create_snapshot(j_for_snap)
            del j_for_snap, wal_for_snap

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            restored = mgr2.restore(j2)

            assert restored == 0
            assert len(j2._entries) == snap.entry_count
            snapshot_hash = journal_state_hash(j2)
            assert snapshot_hash == full_hash, \
                "Journal after snapshot restore must be bit-identical to full replay"
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_restore_with_tail_replay(self):
        """Snapshot at midpoint + tail entries → full state with intact hash chain."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            wal1 = JsonLinesWAL(wal_path)
            j1 = DispatchJournal(wal=wal1)

            lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            for g in range(2):
                for i, ev in enumerate(lifecycle):
                    j1.append(JournalEntry(
                        event=ev, goal_id=f"goal_{g}",
                        execution_id=f"exec_{g}", lease_id=f"lease_{g}",
                        timestamp=1000.0 + g * 100 + i,
                    ))

            mgr = SnapshotManager(wal=wal1, snapshot_path=snap_path)
            snap = mgr.create_snapshot(j1)

            for i, ev in enumerate(lifecycle):
                j1.append(JournalEntry(
                    event=ev, goal_id="goal_2",
                    execution_id="exec_2", lease_id="lease_2",
                    timestamp=1200.0 + i,
                ))

            full_hash = journal_state_hash(j1)
            full_n = len(j1._entries)
            del j1, wal1

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            restored = mgr2.restore(j2)

            assert restored == 4
            assert len(j2._entries) == snap.entry_count + restored
            snapshot_hash = journal_state_hash(j2)
            assert snapshot_hash == full_hash, \
                "Journal after snapshot + tail replay must be bit-identical"
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_restore_preserves_hash_chain(self):
        """IntegrityVerifier passes on restored journal — no broken chains."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            wal1 = JsonLinesWAL(wal_path)
            j1 = DispatchJournal(wal=wal1)

            lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            for g in range(2):
                for i, ev in enumerate(lifecycle):
                    j1.append(JournalEntry(
                        event=ev, goal_id=f"goal_{g}",
                        execution_id=f"exec_{g}", lease_id=f"lease_{g}",
                        timestamp=1000.0 + g * 100 + i,
                    ))

            mgr = SnapshotManager(wal=wal1, snapshot_path=snap_path)
            mgr.create_snapshot(j1)

            for i, ev in enumerate(lifecycle):
                j1.append(JournalEntry(
                    event=ev, goal_id="goal_2",
                    execution_id="exec_2", lease_id="lease_2",
                    timestamp=1200.0 + i,
                ))
            del j1, wal1

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            mgr2.restore(j2)

            v = IntegrityVerifier()
            report = v.verify_integrity(j2)
            assert report.hash_chain_ok, "Hash chain must be intact after restore"
            assert report.sequence_ok, "Sequence must be intact after restore"
            assert report.lifecycle_ok, "Lifecycle must be valid after restore"
            assert report.valid, "IntegrityVerifier must pass after restore"
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_restore_preserves_journal_queries(self):
        """get_chain, get_latest_event, get_stats work after restore."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=2)
            mgr = SnapshotManager(wal=journal._wal, snapshot_path=snap_path)
            mgr.create_snapshot(journal)
            del journal

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            mgr2.restore(j2)

            chain = j2.get_chain("goal_0")
            assert len(chain) == 4

            latest_event = j2.get_latest_event("goal_0")
            assert latest_event == 'COMPLETED'

            stats = j2.get_stats()
            assert stats['total_entries'] == 8
            assert stats['n_goals'] == 2
            assert 'goal_0' in j2._goal_index
            assert 'goal_1' in j2._goal_index
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_restore_with_no_snapshot_returns_zero(self):
        """No snapshot file → restore returns 0 (fallback to full replay)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.nosnap'
        try:
            wal = JsonLinesWAL(wal_path)
            j = DispatchJournal(wal=wal)
            j.append(JournalEntry(event='DISPATCHED', goal_id='g1',
                                   execution_id='e1', lease_id='l1',
                                   timestamp=1000.0))
            mgr = SnapshotManager(wal=wal, snapshot_path=snap_path)
            restored = mgr.restore(j)
            assert restored == 0
        finally:
            os.unlink(wal_path)

    def test_restore_after_corrupted_wal_tail(self):
        """
        Snapshot is valid, WAL tail is corrupted.
        Snapshot entries provide intact prefix, WAL recovery truncates tail.
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            wal1 = JsonLinesWAL(wal_path)
            j1 = DispatchJournal(wal=wal1)
            lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            for g in range(2):
                for i, ev in enumerate(lifecycle):
                    j1.append(JournalEntry(event=ev, goal_id=f"goal_{g}",
                                            execution_id=f"exec_{g}",
                                            lease_id=f"lease_{g}",
                                            timestamp=1000.0 + g * 100 + i))
            mgr = SnapshotManager(wal=wal1, snapshot_path=snap_path)
            snap = mgr.create_snapshot(j1)

            for i, ev in enumerate(lifecycle):
                j1.append(JournalEntry(event=ev, goal_id="goal_2",
                                        execution_id="exec_2", lease_id="lease_2",
                                        timestamp=1200.0 + i))
            del j1, wal1

            with open(wal_path, 'ab') as f:
                f.write(b'\xff\xfe\x00garbage\x00')
                f.flush()
                os.fsync(f.fileno())

            wal2 = JsonLinesWAL(wal_path, auto_recover=True)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            restored = mgr2.restore(j2)

            assert len(j2._entries) == snap.entry_count + restored
            assert 'goal_0' in j2._goal_index
            assert 'goal_1' in j2._goal_index
            j2._last_lsn = snap.last_lsn
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)


class TestBootSequence:

    def test_boot_with_snapshot_restore(self):
        """boot() with valid snapshot returns snapshot_restore method."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=2)
            wal_for_snap = JsonLinesWAL(wal_path)
            j_for_snap = DispatchJournal(wal=wal_for_snap)
            j_for_snap.recover_from_wal()
            mgr = SnapshotManager(wal=wal_for_snap, snapshot_path=snap_path)
            mgr.create_snapshot(j_for_snap)
            del j_for_snap, wal_for_snap, journal

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            result = j2.boot(snapshot_mgr=mgr2)

            assert result.method == 'snapshot_restore'
            assert result.entries_restored == 8
            assert result.tail_replayed == 0
            assert result.integrity_valid
            assert result.integrity_checks['hash_chain_ok']
            assert result.integrity_checks['sequence_ok']
            assert result.duration_ms > 0
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_boot_with_tail_replay(self):
        """boot() replays WAL tail after snapshot restore."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            wal1 = JsonLinesWAL(wal_path)
            j1 = DispatchJournal(wal=wal1)
            lifecycle = ['DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED']
            for g in range(2):
                for i, ev in enumerate(lifecycle):
                    j1.append(JournalEntry(event=ev, goal_id=f"goal_{g}",
                                            execution_id=f"exec_{g}",
                                            lease_id=f"lease_{g}",
                                            timestamp=1000.0 + g * 100 + i))
            mgr = SnapshotManager(wal=wal1, snapshot_path=snap_path)
            mgr.create_snapshot(j1)
            for i, ev in enumerate(lifecycle):
                j1.append(JournalEntry(event=ev, goal_id="goal_2",
                                        execution_id="exec_2", lease_id="lease_2",
                                        timestamp=1200.0 + i))
            del j1, wal1

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            result = j2.boot(snapshot_mgr=mgr2)

            assert result.method == 'snapshot_restore'
            assert result.entries_restored == 12
            assert result.tail_replayed == 4
            assert result.integrity_valid
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_boot_without_snapshot_falls_back_to_full_replay(self):
        """boot() without snapshot falls back to full WAL replay."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.nosnap'
        try:
            build_full_journal(wal_path, n_goals=2)
            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            result = j2.boot(snapshot_mgr=mgr2)

            assert result.method == 'full_replay'
            assert result.entries_restored == 8
            assert result.tail_replayed == 0
            assert result.integrity_valid
        finally:
            os.unlink(wal_path)

    def test_boot_without_snapshot_mgr_falls_back_to_full_replay(self):
        """boot() with no snapshot_mgr falls back to full WAL replay."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        try:
            build_full_journal(wal_path, n_goals=2)
            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            result = j2.boot(snapshot_mgr=None)

            assert result.method == 'full_replay'
            assert result.entries_restored == 8
            assert result.integrity_valid
        finally:
            os.unlink(wal_path)

    def test_boot_metrics_populated(self):
        """boot() populates recovery metrics (file sizes, counts)."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        snap_path = wal_path + '.snap'
        try:
            journal = build_full_journal(wal_path, n_goals=2)
            wal_for_snap = JsonLinesWAL(wal_path)
            j_for_snap = DispatchJournal(wal=wal_for_snap)
            j_for_snap.recover_from_wal()
            mgr = SnapshotManager(wal=wal_for_snap, snapshot_path=snap_path)
            mgr.create_snapshot(j_for_snap)
            del j_for_snap, wal_for_snap, journal

            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            mgr2 = SnapshotManager(wal=wal2, snapshot_path=snap_path)
            result = j2.boot(snapshot_mgr=mgr2)

            assert result.snapshot_size_bytes > 0
            assert result.wal_size_bytes > 0
            assert result.snapshot_entry_count == 8
            assert result.wal_entry_count == 8
            assert result.to_dict()['method'] == 'snapshot_restore'
        finally:
            os.unlink(wal_path)
            if os.path.exists(snap_path):
                os.unlink(snap_path)

    def test_boot_metrics_full_replay(self):
        """boot() with full_replay populates WAL metrics correctly."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        try:
            build_full_journal(wal_path, n_goals=2)
            wal2 = JsonLinesWAL(wal_path)
            j2 = DispatchJournal(wal=wal2)
            result = j2.boot(snapshot_mgr=None)

            assert result.snapshot_size_bytes == 0
            assert result.snapshot_entry_count == 0
            assert result.wal_size_bytes > 0
            assert result.wal_entry_count == 8
        finally:
            os.unlink(wal_path)

    def test_boot_empty_journal(self):
        """boot() on empty journal returns full_replay with 0 entries."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            wal_path = f.name
        try:
            wal = JsonLinesWAL(wal_path)
            j = DispatchJournal(wal=wal)
            result = j.boot(snapshot_mgr=None)
            assert result.method == 'full_replay'
            assert result.entries_restored == 0
            assert result.integrity_valid
        finally:
            os.unlink(wal_path)
