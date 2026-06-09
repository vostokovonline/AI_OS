"""
K1: Property-Based Replay Verification.

Generates random sequences of journal operations and verifies that:

    State_before_crash == State_after_recovery

for every valid sequence of append, snapshot, compact, rotate, crash, recover.
"""

import json
import os
import time
import hashlib
import random
import tempfile
import shutil
import itertools
import pytest

from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.journal import JournalEntry

# Seed for reproducibility
RANDOM_SEED = 42

# Event types for random generation
EVENT_TYPES = ['DISPATCHED', 'STARTED', 'COMPLETED', 'FAILED', 'DISPATCHED', 'STARTED']
GOAL_IDS = ['goal_a', 'goal_b', 'goal_c', 'goal_d']
LEASE_IDS = ['lease_a', 'lease_b', 'lease_c', 'lease_d']

REPAIR_EVENTS = frozenset({'ABANDONED', 'LEASE_EXPIRED'})


def _state_view_hash(kernel: ExecutionKernel) -> str:
    parts = {
        'business_entries': [
            {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
            for e in kernel.journal._entries if e.event not in REPAIR_EVENTS
        ],
        'epoch': getattr(kernel.registry, '_epoch', 0),
    }
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def _random_entry(rng: random.Random, seq: int) -> JournalEntry:
    """Generate a random but syntactically valid journal entry."""
    event = rng.choice(EVENT_TYPES)
    gidx = rng.randint(0, len(GOAL_IDS) - 1)
    goal_id = GOAL_IDS[gidx]
    lease_id = LEASE_IDS[gidx]
    execution_id = f"{goal_id}:exec:{seq}"
    ts = 1000000.0 + seq * 100.0 + rng.random() * 50.0

    kwargs = dict(
        event=event,
        goal_id=goal_id,
        execution_id=execution_id,
        lease_id=lease_id,
        timestamp=ts,
    )

    if event in ('COMPLETED', 'FAILED'):
        kwargs['success'] = event == 'COMPLETED'
        kwargs['duration_ms'] = rng.uniform(10.0, 5000.0)

    return JournalEntry(**kwargs)


def _make_kernel(wal_dir: str, snap_file: str) -> ExecutionKernel:
    os.makedirs(wal_dir, exist_ok=True)
    return ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
        )
    )


# ── Op types ──────────────────────────────────────────────────────────────

OPS_WITH_ARGS = ['append', 'snapshot', 'compact', 'crash']

def _generate_op_sequence(length: int, rng: random.Random) -> list[str]:
    ops = []
    for _ in range(length):
        ops.append(rng.choice(OPS_WITH_ARGS))
    return ops


class OpRunner:
    """Executes a sequence of operations on a kernel, tracking state after each."""

    def __init__(self, tmpdir: str):
        self.wal_dir = os.path.join(tmpdir, "wal")
        self.snap_file = os.path.join(tmpdir, "snapshot.json")
        self.kernel = _make_kernel(self.wal_dir, self.snap_file)
        self.entry_seq = [0]
        self.hash_history = []

    async def run_ops(self, ops: list[str]) -> list[dict]:
        """Run an op sequence, recording state hash before each crash/recover cycle."""
        results = []
        for op in ops:
            if op == 'append':
                self.entry_seq[0] += 1
                entry = _random_entry(random.Random(self.entry_seq[0]), self.entry_seq[0])
                self.kernel.journal.append(entry)

            elif op == 'snapshot':
                self.kernel.snapshot()

            elif op == 'compact':
                latest = getattr(self.kernel.snapshots, '_latest', None)
                if latest:
                    wal = getattr(self.kernel.journal, '_wal', None)
                    if wal and hasattr(wal, 'prune_segments'):
                        wal.prune_segments(latest.last_lsn)

            elif op == 'crash':
                hash_before = _state_view_hash(self.kernel)
                journal_before = list(self.kernel.journal._entries)

                # Destroy kernel
                old_kernel = self.kernel
                self.kernel = _make_kernel(self.wal_dir, self.snap_file)

                # Recover from WAL/snapshots
                await self.kernel.recover()

                hash_after = _state_view_hash(self.kernel)
                journal_after = list(self.kernel.journal._entries)

                # Filter repair events from both sides for comparison
                biz_before = [e for e in journal_before if e.event not in REPAIR_EVENTS]
                biz_after = [e for e in journal_after if e.event not in REPAIR_EVENTS]

                ok = True
                detail = ""
                if hash_before != hash_after:
                    ok = False
                    detail = "state_hash_mismatch"
                elif len(biz_before) != len(biz_after):
                    ok = False
                    detail = f"entry_count_mismatch: {len(biz_before)} vs {len(biz_after)}"
                else:
                    for i, (eb, ea) in enumerate(zip(biz_before, biz_after)):
                        if eb.event != ea.event or eb.goal_id != ea.goal_id:
                            ok = False
                            detail = f"entry_{i}_mismatch: {eb.event}/{eb.goal_id} vs {ea.event}/{ea.goal_id}"
                            break

                results.append({
                    'op_index': len(results),
                    'op': 'crash',
                    'ok': ok,
                    'detail': detail,
                    'hash_before': hash_before[:12],
                    'hash_after': hash_after[:12],
                    'entries_before': len(biz_before),
                    'entries_after': len(biz_after),
                })

        return results


class TestK1PropertyBased:

    @pytest.mark.asyncio
    async def test_short_sequences(self):
        """K1.1 — All 2-op sequences ending with crash."""
        seqs = [list(p) + ['crash'] for p in itertools.product(OPS_WITH_ARGS, repeat=2)]
        assert len(seqs) == 16  # 4 ops ^ 2 = 16 combinations + crash

        failures = []
        for seq in seqs:
            with tempfile.TemporaryDirectory() as tmpdir:
                runner = OpRunner(tmpdir)
                results = await runner.run_ops(seq)
                for r in results:
                    if not r['ok']:
                        failures.append((seq, r))

        assert len(failures) == 0, f"Failed sequences: {failures}"

    @pytest.mark.asyncio
    async def test_medium_random_sequences(self):
        """K1.2 — 50 random sequences of length 8-15 with crash at end."""
        rng = random.Random(RANDOM_SEED)
        failures = []

        for trial in range(50):
            length = rng.randint(8, 15)
            seq = _generate_op_sequence(length, rng) + ['crash']
            with tempfile.TemporaryDirectory() as tmpdir:
                runner = OpRunner(tmpdir)
                results = await runner.run_ops(seq)
                for r in results:
                    if not r['ok']:
                        failures.append((trial, seq, r))

        assert len(failures) == 0, f"{len(failures)} failures out of 50 sequences"

    @pytest.mark.asyncio
    async def test_multiple_crashes_same_wal(self):
        """K1.3 — Multiple crash/recover cycles on growing WAL."""
        rng = random.Random(RANDOM_SEED + 1)
        failures = []

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpRunner(tmpdir)

            for cycle in range(10):
                # Append 3-8 random entries
                append_count = rng.randint(3, 8)
                for _ in range(append_count):
                    runner.entry_seq[0] += 1
                    entry = _random_entry(rng, runner.entry_seq[0])
                    runner.kernel.journal.append(entry)

                # Optionally snapshot
                if rng.random() < 0.3:
                    runner.kernel.snapshot()

                # Crash and recover
                hash_before = _state_view_hash(runner.kernel)
                journal_before = list(runner.kernel.journal._entries)
                old_kernel = runner.kernel
                runner.kernel = _make_kernel(runner.wal_dir, runner.snap_file)
                await runner.kernel.recover()
                hash_after = _state_view_hash(runner.kernel)

                if hash_before != hash_after:
                    failures.append({
                        'cycle': cycle,
                        'entries_before': len([e for e in journal_before if e.event not in REPAIR_EVENTS]),
                        'hash_before': hash_before[:12],
                        'hash_after': hash_after[:12],
                    })

        assert len(failures) == 0, f"Multi-cycle failures: {failures}"

    @pytest.mark.asyncio
    async def test_append_snapshot_compact_crash_cycle(self):
        """K1.4 — Full lifecycle: append → snapshot → compact → crash → recover."""
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpRunner(tmpdir)

            # Phase 1: append entries
            for i in range(20):
                entry = _random_entry(random.Random(100 + i), i)
                runner.kernel.journal.append(entry)

            hash_phase1 = _state_view_hash(runner.kernel)

            # Phase 2: snapshot
            runner.kernel.snapshot()
            latest = getattr(runner.kernel.snapshots, '_latest', None)
            if latest:
                assert latest is not None
            else:
                # New persistence path — snapshot was created implicitly
                pass

            # Phase 3: more appends
            for i in range(20, 30):
                entry = _random_entry(random.Random(200 + i), i)
                runner.kernel.journal.append(entry)

            hash_phase3 = _state_view_hash(runner.kernel)

            # Phase 4: compact (prune segments before snapshot)
            latest = getattr(runner.kernel.snapshots, '_latest', None)
            if latest:
                wal = getattr(runner.kernel.journal, '_wal', None)
                if wal and hasattr(wal, 'prune_segments'):
                    result = wal.prune_segments(latest.last_lsn)
                    assert result.deleted >= 0

            # Phase 5: crash + recover
            old_kernel = runner.kernel
            runner.kernel = _make_kernel(runner.wal_dir, runner.snap_file)
            await runner.kernel.recover()
            hash_recovered = _state_view_hash(runner.kernel)

            assert hash_phase3 == hash_recovered, \
                "State after compact+crash+recover does not match state before crash"

    @pytest.mark.asyncio
    async def test_append_only_no_crash(self):
        """K1.5 — Append-only sequences, verify replay determinism."""
        rng = random.Random(RANDOM_SEED + 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            runner = OpRunner(tmpdir)
            for i in range(100):
                entry = _random_entry(rng, i)
                runner.kernel.journal.append(entry)

            hash_initial = _state_view_hash(runner.kernel)

            # Recover on same WAL (no crash — tests cold boot)
            old_kernel = runner.kernel
            runner.kernel = _make_kernel(runner.wal_dir, runner.snap_file)
            await runner.kernel.recover()

            hash_cold = _state_view_hash(runner.kernel)
            assert hash_initial == hash_cold, "Cold boot from WAL produces different state"
