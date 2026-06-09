"""
ExecutionKernel Recovery Validation.

Verifies that both the PRIMARY (SegmentedWAL + boot()) and COMPATIBILITY
(WriteAheadLog + legacy replay) recovery paths produce IDENTICAL state
for identical scenarios, and that the new path survives stress.

Migration policy: see CLAUDE.md "Persistence Layer: Kernel Migration"
"""

import json
import os
import time
import hashlib
import tempfile
import shutil
import pytest

from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.journal import JournalEntry


# ============================================================================
# Helpers
# ============================================================================

# ============================================================================
# Recovery Model: three ontological layers
# ============================================================================
# Layer 1 — State (truth):
#   Business journal entries replayed from WAL. Deterministic.
# Layer 2 — Repair (recovery-side effects):
#   Synthetic entries (ABANDONED, LEASE_EXPIRED) injected during recovery.
#   Event content is deterministic, timestamp is not.
# Layer 3 — Diagnostics (observability):
#   status=ok/corrupt, invariant violations, boot method.
#   Semantic comparison only (not hash).
#
# Recovery equality tests compare Layer 1 + Layer 2's deterministic footprint.
# Layer 3 is checked semantically in dedicated tests.
# ============================================================================

REPAIR_EVENTS = frozenset({'ABANDONED', 'LEASE_EXPIRED'})


def _state_view_hash(kernel: ExecutionKernel) -> str:
    """Deterministic hash of Layer 1 (business state) + Layer 2 (derived state).

    Layer 1 — Business entries (WAL-replayed, deterministically sorted).
      Hash-chain fields are stripped (differ between WAL implementations).

    Layer 2 — Derived state (pure function of full journal):
      active_executions, lease registry, epoch.
      Identical given identical business entries + identical repair decisions.

    Repair events (ABANDONED, LEASE_EXPIRED) are excluded from the hash
    because their timestamps are non-deterministic (time.time() at recovery).
    Their event content (type, goal_id, lease_id) IS deterministic given
    identical business entries, but is verified implicitly through the
    derived state (repair events produce identical active_executions/leases).
    """
    parts = {
        'business_entries': [
            {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
            for e in kernel.journal._entries if e.event not in REPAIR_EVENTS
        ],
        'active_executions': dict(kernel._active_executions),
        'leases': kernel.registry.get_stats() if hasattr(kernel.registry, 'get_stats') else {},
        'epoch': getattr(kernel.registry, '_epoch', 0),
    }
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _repair_trace(kernel: ExecutionKernel) -> list[dict]:
    """Layer 2 trace: repair events with non-deterministic fields removed.

    Used for debugging/audit — not for equality comparison between paths.
    """
    return [
        {k: v for k, v in e.to_dict().items() if k not in ('timestamp', 'prev_hash', 'entry_hash')}
        for e in kernel.journal._entries if e.event in REPAIR_EVENTS
    ]


def _populate_scenario(kernel: ExecutionKernel, scenario: list[dict]) -> None:
    """Append a list of journal entry dicts to the kernel's journal."""
    for kwargs in scenario:
        entry = JournalEntry(**kwargs)
        kernel.journal.append(entry)


def _make_new_kernel(tmpdir: str) -> ExecutionKernel:
    """Create a PRIMARY-path ExecutionKernel (SegmentedWAL + boot())."""
    wal_dir = os.path.join(tmpdir, "wal")
    snap_file = os.path.join(tmpdir, "snapshot.json")
    os.makedirs(wal_dir, exist_ok=True)
    return ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
        )
    )


def _make_legacy_kernel() -> ExecutionKernel:
    """Create a COMPATIBILITY-path ExecutionKernel (WriteAheadLog + legacy)."""
    return ExecutionKernel(redis_client=None)


# ============================================================================
# Standard lifecycle scenario
# ============================================================================

CLEAN_LIFECYCLE = [
    dict(event='DISPATCHED', goal_id='goal_a', execution_id='exec_a1',
         lease_id='lease_a1', timestamp=1000.0, dispatch_epoch=1),
    dict(event='LEASE_ISSUED', goal_id='goal_a', execution_id='exec_a1',
         lease_id='lease_a1', timestamp=1001.0, dispatch_epoch=1),
    dict(event='STARTED', goal_id='goal_a', execution_id='exec_a1',
         lease_id='lease_a1', timestamp=1002.0, dispatch_epoch=1),
    dict(event='COMPLETED', goal_id='goal_a', execution_id='exec_a1',
         lease_id='lease_a1', timestamp=1003.0, dispatch_epoch=1, success=True, duration_ms=500.0),
]

DANGLING_LIFECYCLE = [
    dict(event='DISPATCHED', goal_id='goal_b', execution_id='exec_b1',
         lease_id='lease_b1', timestamp=2000.0, dispatch_epoch=2),
    dict(event='LEASE_ISSUED', goal_id='goal_b', execution_id='exec_b1',
         lease_id='lease_b1', timestamp=2001.0, dispatch_epoch=2),
    dict(event='STARTED', goal_id='goal_b', execution_id='exec_b1',
         lease_id='lease_b1', timestamp=2002.0, dispatch_epoch=2),
    # No COMPLETED/FAILED — dangling STARTED (should be ABANDONED)
]

EXPIRED_LIFECYCLE = [
    dict(event='DISPATCHED', goal_id='goal_c', execution_id='exec_c1',
         lease_id='lease_c1', timestamp=3000.0, dispatch_epoch=3),
    dict(event='LEASE_ISSUED', goal_id='goal_c', execution_id='exec_c1',
         lease_id='lease_c1', timestamp=3001.0, dispatch_epoch=3),
    # No STARTED — lease expired (should be LEASE_EXPIRED)
]

FAILED_LIFECYCLE = [
    dict(event='DISPATCHED', goal_id='goal_d', execution_id='exec_d1',
         lease_id='lease_d1', timestamp=4000.0, dispatch_epoch=4),
    dict(event='LEASE_ISSUED', goal_id='goal_d', execution_id='exec_d1',
         lease_id='lease_d1', timestamp=4001.0, dispatch_epoch=4),
    dict(event='STARTED', goal_id='goal_d', execution_id='exec_d1',
         lease_id='lease_d1', timestamp=4002.0, dispatch_epoch=4),
    dict(event='FAILED', goal_id='goal_d', execution_id='exec_d1',
         lease_id='lease_d1', timestamp=4003.0, dispatch_epoch=4, success=False, error='timeout'),
]

RETRY_LIFECYCLE = [
    dict(event='DISPATCHED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e1', timestamp=5000.0, dispatch_epoch=5),
    dict(event='LEASE_ISSUED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e1', timestamp=5001.0, dispatch_epoch=5),
    dict(event='STARTED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e1', timestamp=5002.0, dispatch_epoch=5),
    dict(event='FAILED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e1', timestamp=5003.0, dispatch_epoch=5, error='transient'),
    dict(event='RETRIED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e1', timestamp=5004.0, dispatch_epoch=6),
    dict(event='STARTED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e2', timestamp=5005.0, dispatch_epoch=6),
    dict(event='COMPLETED', goal_id='goal_e', execution_id='exec_e1',
         lease_id='lease_e2', timestamp=5006.0, dispatch_epoch=6, success=True, duration_ms=800.0),
]


# ============================================================================
# C.1 + C.2: Baseline legacy + new path equivalence
# ============================================================================

class TestBaselineRecovery:

    @pytest.mark.asyncio
    async def test_legacy_recovery_clean(self):
        """C.1: Legacy path recovers clean lifecycle without errors."""
        kernel = _make_legacy_kernel()
        _populate_scenario(kernel, CLEAN_LIFECYCLE)

        result = await kernel.recover()
        assert result['status'] == 'ok', f"Legacy recovery failed: {result}"
        assert result['journal_recovered'] >= len(CLEAN_LIFECYCLE)

    @pytest.mark.asyncio
    async def test_legacy_recovery_deterministic(self):
        """C.1: Two recoveries on same legacy kernel produce identical state."""
        kernel = _make_legacy_kernel()
        _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE)

        await kernel.recover()
        h1 = _state_view_hash(kernel)
        await kernel.recover()
        h2 = _state_view_hash(kernel)
        assert h1 == h2

    @pytest.mark.asyncio
    async def test_new_recovery_clean(self):
        """C.2: New path recovers clean lifecycle via boot()."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE)

            result = await kernel.recover()
            assert result['status'] == 'ok', f"New recovery failed: {result}"
            assert result['boot_method'] in ('snapshot_restore', 'full_replay')
            assert result['journal_recovered'] >= len(CLEAN_LIFECYCLE)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_new_recovery_deterministic(self):
        """C.2: Two recoveries on same new kernel produce identical state."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE)

            await kernel.recover()
            h1 = _state_view_hash(kernel)
            await kernel.recover()
            h2 = _state_view_hash(kernel)
            assert h1 == h2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_legacy_vs_new_state_identical(self):
        """C.2: Legacy and new paths produce identical recovery state for same scenario."""
        tmpdir = tempfile.mkdtemp()
        try:
            full_scenario = CLEAN_LIFECYCLE + DANGLING_LIFECYCLE + EXPIRED_LIFECYCLE + FAILED_LIFECYCLE + RETRY_LIFECYCLE

            legacy = _make_legacy_kernel()
            _populate_scenario(legacy, full_scenario)
            await legacy.recover()
            legacy_state = _state_view_hash(legacy)

            new_ = _make_new_kernel(tmpdir)
            _populate_scenario(new_, full_scenario)
            await new_.recover()
            new_state = _state_view_hash(new_)

            assert legacy_state == new_state, \
                "Legacy and new recovery paths produce different state"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# C.2: Dangling lease detection equivalence
# ============================================================================

class TestDanglingLeases:

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scenario,expected_abandoned,expected_expired', [
        (DANGLING_LIFECYCLE, 1, 0),
        (EXPIRED_LIFECYCLE, 0, 1),
    ])
    async def test_legacy_dangling_detection(self, scenario, expected_abandoned, expected_expired):
        """Legacy path correctly detects dangling leases."""
        kernel = _make_legacy_kernel()
        _populate_scenario(kernel, scenario)
        result = await kernel.recover()
        assert result['leases_abandoned'] == expected_abandoned, f"Expected {expected_abandoned} abandoned"
        assert result['leases_expired'] == expected_expired, f"Expected {expected_expired} expired"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scenario,expected_abandoned,expected_expired', [
        (DANGLING_LIFECYCLE, 1, 0),
        (EXPIRED_LIFECYCLE, 0, 1),
    ])
    async def test_new_dangling_detection(self, scenario, expected_abandoned, expected_expired):
        """New path correctly detects dangling leases."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, scenario)
            result = await kernel.recover()
            assert result['leases_abandoned'] == expected_abandoned
            assert result['leases_expired'] == expected_expired
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_dangling_equivalence(self):
        """Both paths detect the SAME dangling leases."""
        tmpdir = tempfile.mkdtemp()
        try:
            scenario = DANGLING_LIFECYCLE + EXPIRED_LIFECYCLE

            legacy = _make_legacy_kernel()
            _populate_scenario(legacy, scenario)
            r1 = await legacy.recover()

            new_ = _make_new_kernel(tmpdir)
            _populate_scenario(new_, scenario)
            r2 = await new_.recover()

            assert r1['leases_abandoned'] == r2['leases_abandoned']
            assert r1['leases_expired'] == r2['leases_expired']
            assert r1['leases_restored'] == r2['leases_restored']
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# C.3: Crash consistency
# ============================================================================

class TestCrashConsistency:

    @pytest.mark.asyncio
    async def test_crash_mid_append_then_recover(self):
        """
        C.3: Append to new path WAL, simulate crash by deleting journal in-memory,
        recover from WAL. State must match original.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE)

            # Capture original state hash
            await kernel.recover()
            original_hash = _state_view_hash(kernel)

            # Simulate crash: destroy in-memory journal, keep WAL untouched
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None

            # Recover from WAL via boot()
            result = await kernel.recover()
            assert result['status'] == 'ok'
            assert result['journal_recovered'] >= len(CLEAN_LIFECYCLE)

            recovered_hash = _state_view_hash(kernel)
            assert recovered_hash == original_hash, \
                "Crash-mid-append: recovered state differs from original"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_crash_during_snapshot_then_recover(self):
        """
        C.3: Append, snapshot, simulate crash, recover from WAL.
        State must match original.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE)
            await kernel.recover()
            original_hash = _state_view_hash(kernel)

            # Create snapshot (this is what would survive after crash)
            snap = kernel.snapshots.create_snapshot(kernel.journal)

            # Destroy in-memory state
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None

            # Recover from snapshot + WAL tail via boot()
            result = await kernel.recover()
            assert result['status'] == 'ok'
            assert result['journal_recovered'] >= len(CLEAN_LIFECYCLE + DANGLING_LIFECYCLE)

            recovered_hash = _state_view_hash(kernel)
            assert recovered_hash == original_hash, \
                "Snapshot crash: recovered state differs from original"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_open_journal_append_after_recovery(self):
        """
        C.3: Recover -> append -> recover again. Append must survive second recovery.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE)

            # First recovery
            r1 = await kernel.recover()
            assert r1['status'] == 'ok'

            # Append more events
            _populate_scenario(kernel, FAILED_LIFECYCLE)

            # Simulate crash
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None

            # Second recovery -- must include both CLEAN and FAILED
            r2 = await kernel.recover()
            assert r2['status'] == 'ok'
            assert r2['journal_recovered'] >= len(CLEAN_LIFECYCLE + FAILED_LIFECYCLE), \
                "Append after recovery lost in second recovery"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# C.4: Compaction safety
# ============================================================================

class TestCompactionSafety:

    @pytest.mark.asyncio
    async def test_snapshot_compact_recover_equivalence(self):
        """
        C.4: Snapshot -> prune_segments -> recover.
        Recovered state must be identical to pre-compaction state.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE + FAILED_LIFECYCLE)

            await kernel.recover()
            original_hash = _state_view_hash(kernel)

            # Snapshot + compact
            snap = kernel.snapshots.create_snapshot(kernel.journal)
            comp = kernel.wal.prune_segments(snap.last_lsn)
            assert comp.segments_deleted > 0 or comp.bytes_reclaimed >= 0

            # Destroy in-memory, recover from compacted WAL
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None
            result = await kernel.recover()
            assert result['status'] == 'ok'

            recovered_hash = _state_view_hash(kernel)
            assert recovered_hash == original_hash, \
                "Snapshot+compaction: recovered state differs from original"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_multi_cycle_compaction(self):
        """
        C.4: Three compaction cycles, each followed by recovery.
        State must be identical to original across all cycles.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)

            # Round 1: append some, snapshot, compact, recover
            _populate_scenario(kernel, CLEAN_LIFECYCLE)
            await kernel.recover()
            original_hash = _state_view_hash(kernel)
            snap = kernel.snapshots.create_snapshot(kernel.journal)
            kernel.wal.prune_segments(snap.last_lsn)
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None
            await kernel.recover()
            assert _state_view_hash(kernel) == original_hash, "Round 1 lost data"

            # Round 2: append more, snapshot, compact, recover
            _populate_scenario(kernel, DANGLING_LIFECYCLE)
            await kernel.recover()
            original_hash = _state_view_hash(kernel)
            snap = kernel.snapshots.create_snapshot(kernel.journal)
            kernel.wal.prune_segments(snap.last_lsn)
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None
            await kernel.recover()
            assert _state_view_hash(kernel) == original_hash, "Round 2 lost data"

            # Round 3: append more, snapshot, compact, recover
            _populate_scenario(kernel, FAILED_LIFECYCLE)
            await kernel.recover()
            original_hash = _state_view_hash(kernel)
            snap = kernel.snapshots.create_snapshot(kernel.journal)
            comp = kernel.wal.prune_segments(snap.last_lsn)
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None
            await kernel.recover()
            assert _state_view_hash(kernel) == original_hash, "Round 3 lost data"
            assert comp.segments_deleted >= 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# C.5: Corruption tolerance
# ============================================================================

class TestCorruptionTolerance:

    @pytest.mark.asyncio
    async def test_corrupted_wal_tail_recovery(self):
        """
        C.5: Truncate the active segment of the new path WAL.
        Recovery must produce a valid prefix of the original state.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE + FAILED_LIFECYCLE)

            await kernel.recover()
            original_hash = _state_view_hash(kernel)
            original_count = len(kernel.journal._entries)

            # Corrupt the active segment: truncate last 50 bytes
            for seg in kernel.wal._segments.values():
                if seg.active:
                    with open(seg.path, 'ab') as f:
                        f.truncate(max(0, os.path.getsize(seg.path) - 50))
                    break

            # Recover -- must produce a prefix of original state
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None
            result = await kernel.recover()
            assert result['status'] == 'ok', f"Recovery failed after corruption: {result}"

            recovered_count = len(kernel.journal._entries)
            assert recovered_count <= original_count, \
                f"Recovered {recovered_count} > original {original_count}"
            assert recovered_count > 0, \
                "Recovery after truncation produced empty state"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_corrupted_snapshot_fallback_to_full_replay(self):
        """
        C.5: Corrupted snapshot file. Recovery must fall back to full WAL replay
        and produce state identical to original.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE + FAILED_LIFECYCLE)

            await kernel.recover()
            original_hash = _state_view_hash(kernel)

            # Create snapshot
            snap = kernel.snapshots.create_snapshot(kernel.journal)

            # Corrupt the snapshot file
            with open(kernel.snapshots._snapshot_path, 'w') as f:
                f.write('{"corrupted": true}')

            # Destroy in-memory and recover
            kernel.journal._entries.clear()
            kernel.journal._goal_index.clear()
            kernel.journal._last_lsn = None
            result = await kernel.recover()
            assert result['status'] == 'ok'
            assert result['boot_method'] == 'full_replay', \
                "Expected full WAL replay fallback after corrupted snapshot"

            recovered_hash = _state_view_hash(kernel)
            assert recovered_hash == original_hash, \
                "Corrupted snapshot: full replay produced different state"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_empty_wal_recovery(self):
        """C.5: Empty WAL must recover with status 'ok' and 0 entries."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            result = await kernel.recover()
            assert result['status'] == 'ok'
            assert result['journal_recovered'] == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)



# ============================================================================
# C.6: Mixed mode stability
# ============================================================================

class TestMixedMode:

    @pytest.mark.asyncio
    async def test_dispatch_and_switch_to_new(self):
        """
        C.6: Populate legacy kernel, export journal entries,
        inject into new kernel, recover -- state must match.
        """
        tmpdir = tempfile.mkdtemp()
        try:
            scenario = CLEAN_LIFECYCLE + DANGLING_LIFECYCLE + FAILED_LIFECYCLE

            # Legacy: populate and recover
            legacy = _make_legacy_kernel()
            _populate_scenario(legacy, scenario)
            legacy_result = await legacy.recover()
            legacy_hash = _state_view_hash(legacy)
            assert legacy_result['status'] == 'ok'

            # New: export journal entries from legacy, append to new journal
            new_ = _make_new_kernel(tmpdir)
            for entry in legacy.journal._entries:
                new_.journal.append(JournalEntry(
                    event=entry.event,
                    goal_id=entry.goal_id,
                    execution_id=entry.execution_id,
                    lease_id=entry.lease_id,
                    timestamp=entry.timestamp,
                    dispatch_epoch=entry.dispatch_epoch,
                    success=getattr(entry, 'success', None),
                    duration_ms=getattr(entry, 'duration_ms', 0.0),
                    error=getattr(entry, 'error', None),
                ))
            new_result = await new_.recover()
            assert new_result['status'] == 'ok'
            new_hash = _state_view_hash(new_)
            assert legacy_hash == new_hash, \
                "Mixed-mode: legacy->new migration produces different state"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_both_paths_handle_empty_gracefully(self):
        """Both paths must handle empty kernel gracefully."""
        legacy = _make_legacy_kernel()
        r1 = await legacy.recover()
        assert r1['status'] == 'ok'
        assert r1['journal_recovered'] == 0

        tmpdir = tempfile.mkdtemp()
        try:
            new_ = _make_new_kernel(tmpdir)
            r2 = await new_.recover()
            assert r2['status'] == 'ok'
            assert r2['journal_recovered'] == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Invariant verification
# ============================================================================

class TestRecoveryInvariants:

    @pytest.mark.asyncio
    async def test_recovery_preserves_invariants(self):
        """Recovery must pass all post-recovery invariant checks."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            _populate_scenario(kernel, CLEAN_LIFECYCLE + DANGLING_LIFECYCLE + FAILED_LIFECYCLE)

            result = await kernel.recover()
            assert result['status'] == 'ok', f"Invariant violation: {result.get('fatal_violation', 'unknown')}"
            assert 'invariant_violations' not in result or result['invariant_violations'] == 0, \
                f"Post-recovery invariants failed: {result}"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# C.7: Idempotent Dispatch (P2.12a)
# ============================================================================


class TestIdempotentDispatch:

    def test_rebuild_seen_dispatches_populates_from_journal(self):
        """_rebuild_seen_dispatches() populates _seen_dispatches from DISPATCHED entries."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel.config.dedup_window_seconds = 3600
            journal = kernel.journal
            now = time.time()

            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g1', execution_id='e1',
                lease_id='l1', timestamp=now - 100, dispatch_epoch=1,
                dispatch_id='dd1',
            ))
            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g2', execution_id='e2',
                lease_id='l2', timestamp=now - 50, dispatch_epoch=2,
                dispatch_id='dd2',
            ))
            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g3', execution_id='e3',
                lease_id='l3', timestamp=now, dispatch_epoch=3,
                dispatch_id='dd3',
            ))
            journal.append(JournalEntry(
                event='COMPLETED', goal_id='g1', execution_id='e1',
                lease_id='l1', timestamp=now + 10, dispatch_epoch=1,
                success=True, duration_ms=100.0,
            ))

            kernel._rebuild_seen_dispatches()

            assert 'dd1' in kernel._seen_dispatches
            assert 'dd2' in kernel._seen_dispatches
            assert 'dd3' in kernel._seen_dispatches
            assert kernel._seen_dispatches['dd1'] == pytest.approx(now - 100, abs=0.001)
            assert kernel._seen_dispatches['dd2'] == pytest.approx(now - 50, abs=0.001)
            assert len(kernel._seen_dispatches) == 3
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_rebuild_skips_entries_without_dispatch_id(self):
        """DISPATCHED entries without dispatch_id are not added to dedup index."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel.config.dedup_window_seconds = 3600
            journal = kernel.journal

            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g1', execution_id='e1',
                lease_id='l1', timestamp=time.time(), dispatch_epoch=1,
            ))
            kernel._rebuild_seen_dispatches()
            assert len(kernel._seen_dispatches) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_rebuild_excludes_old_entries_beyond_dedup_window(self):
        """DISPATCHED entries older than dedup_window_seconds are excluded."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel.config.dedup_window_seconds = 60
            journal = kernel.journal

            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g1', execution_id='e1',
                lease_id='l1', timestamp=100.0, dispatch_epoch=1,
                dispatch_id='old_dd',
            ))
            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g2', execution_id='e2',
                lease_id='l2', timestamp=time.time(), dispatch_epoch=2,
                dispatch_id='fresh_dd',
            ))

            kernel._rebuild_seen_dispatches()

            assert 'fresh_dd' in kernel._seen_dispatches
            assert 'old_dd' not in kernel._seen_dispatches
            assert len(kernel._seen_dispatches) == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dedup_index_survives_recovery(self):
        """After recover(), _seen_dispatches is rebuilt from journal DISPATCHED entries."""
        import asyncio
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel.config.dedup_window_seconds = 3600
            journal = kernel.journal

            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g1', execution_id='e1',
                lease_id='l1', timestamp=time.time(), dispatch_epoch=1,
                dispatch_id='survive_dd',
            ))
            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g2', execution_id='e2',
                lease_id='l2', timestamp=time.time(), dispatch_epoch=2,
                dispatch_id='also_survive',
            ))

            asyncio.run(kernel.recover())

            assert 'survive_dd' in kernel._seen_dispatches
            assert 'also_survive' in kernel._seen_dispatches
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_dedup_index_empty_after_clean_start(self):
        """Fresh kernel (no recovery) has empty _seen_dispatches."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            assert len(kernel._seen_dispatches) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_rebuild_clears_previous_index(self):
        """_rebuild_seen_dispatches() clears old entries before rebuilding."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel._seen_dispatches['stale_dd'] = 1.0
            kernel._seen_dispatches['also_stale'] = 2.0

            kernel._rebuild_seen_dispatches()
            assert len(kernel._seen_dispatches) == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_recovery_metrics_include_dedup_index_size(self):
        """recover() returns dedup_index_size in metrics."""
        import asyncio
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel.config.dedup_window_seconds = 3600
            journal = kernel.journal

            journal.append(JournalEntry(
                event='DISPATCHED', goal_id='g1', execution_id='e1',
                lease_id='l1', timestamp=time.time(), dispatch_epoch=1,
                dispatch_id='metric_dd',
            ))

            result = asyncio.run(kernel.recover())
            assert 'dedup_index_size' in result
            assert result['dedup_index_size'] == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_legacy_path_can_rebuild_dedup_index(self):
        """COMPATIBILITY path's _rebuild_seen_dispatches() works with pre-appended entries."""
        import asyncio
        kernel = _make_legacy_kernel()
        kernel.config.dedup_window_seconds = 3600
        journal = kernel.journal

        journal.append(JournalEntry(
            event='DISPATCHED', goal_id='g1', execution_id='e1',
            lease_id='l1', timestamp=time.time(), dispatch_epoch=1,
            dispatch_id='legacy_dd',
        ))

        # COMPATIBILITY path recover() clears journal._entries via recover_from_wal()
        # when WriteAheadLog is present. Test _rebuild_seen_dispatches directly.
        kernel._rebuild_seen_dispatches()
        assert 'legacy_dd' in kernel._seen_dispatches


# ============================================================================
# C.8: Lease Heartbeat Lifecycle (P2.12b)
# ============================================================================


class TestLeaseHeartbeatLifecycle:

    def test_heartbeat_extends_lease(self):
        """heartbeat() extends expires_at for an active lease."""
        from execution_dynamics.lease import LeaseRegistry
        registry = LeaseRegistry(default_ttl=3600)
        lease = registry.acquire(goal_id='g1', execution_id='e1')
        original_expiry = lease.expires_at
        time.sleep(0.001)
        ok = registry.heartbeat(lease.lease_id)
        assert ok
        assert lease.expires_at > original_expiry

    def test_detect_stale_finds_idle_lease(self):
        """detect_stale() returns lease that never received heartbeat."""
        from execution_dynamics.lease import LeaseRegistry
        registry = LeaseRegistry(default_ttl=3600)
        lease = registry.acquire(goal_id='g1', execution_id='e1')
        stale = registry.detect_stale(idle_threshold=0)
        assert len(stale) >= 1
        assert any(s['lease_id'] == lease.lease_id for s in stale)

    def test_expire_stale_journals_and_cleans_active_executions(self):
        """expire_stale_leases() journals LEASE_EXPIRED and removes stale from _active_executions."""
        import asyncio
        tmpdir = tempfile.mkdtemp()
        try:
            kernel = _make_new_kernel(tmpdir)
            kernel.config.stale_timeout_seconds = 0  # immediate stale
            from execution_dynamics.lease import LeaseRegistry

            # Acquire a lease directly into the kernel's registry
            lease = kernel.registry.acquire(
                goal_id='g1', execution_id='e1',
                ttl=999999,  # long TTL — won't expire naturally
            )
            kernel._active_executions['g1'] = lease.lease_id

            count = kernel.expire_stale_leases(0)
            assert count >= 1

            # Verify LEASE_EXPIRED was journaled
            le_entries = [e for e in kernel.journal._entries if e.event == 'LEASE_EXPIRED']
            assert len(le_entries) >= 1
            assert le_entries[-1].goal_id == 'g1'

            # Verify removed from active_executions
            assert 'g1' not in kernel._active_executions
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
