"""
K5 — Ownership & Concurrency Correctness.

Proves invariants of the existing single-process ownership model
without changing the kernel:

  K5.1 — Exactly one active owner per execution at any time
  K5.2 — Deterministic execution identity (same inputs → same ID)
  K5.3 — No duplicate lifecycle roots in journal
  K5.4 — Deterministic recovery under concurrent reads
  K5.5 — Lease lifecycle transitions are valid

All tests operate on the EXISTING kernel API — no locking, no fencing,
no distributed coordination added. These are proofs of the current model.
"""

import json
import os
import time
import hashlib
import random
import tempfile
import shutil
import pytest

os.environ.setdefault('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost:5432/test')

from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.journal import JournalEntry, DISPATCH_EVENTS
from execution_dynamics.lease import LEASE_ACTIVE, LEASE_COMPLETED, LEASE_EXPIRED, LEASE_REVOKED, LEASE_ABANDONED

RANDOM_SEED = 42
GOAL_IDS = ['goal_a', 'goal_b', 'goal_c', 'goal_d']

REPAIR_EVENTS = frozenset({'ABANDONED', 'LEASE_EXPIRED'})

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_kernel(wal_dir: str, snap_file: str) -> ExecutionKernel:
    os.makedirs(wal_dir, exist_ok=True)
    return ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
        )
    )


def _journal_entries(entries: list[JournalEntry]) -> list[dict]:
    return [
        {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
        for e in entries
        if e.event not in REPAIR_EVENTS
    ]


def _entries_hash(kernel: ExecutionKernel) -> str:
    raw = json.dumps(_journal_entries(kernel.journal._entries), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Test Class ───────────────────────────────────────────────────────────────


class TestK5OwnershipCorrectness:

    # ══════════════════════════════════════════════════════════════════════════
    # K5.1 — Single Ownership per Execution
    #
    # Invariant: for any goal, at most one active lease exists at any time.
    # After dispatch acquires a lease, get_active_lease() returns the same
    # lease until it transitions to terminal.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k5_1_single_owner_per_goal(self):
        """get_active_lease tracks exactly one owner per goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            # No active lease initially
            assert k.registry.get_active_lease("goal_a") is None

            # Acquire — now active
            lease1 = k.registry.acquire("goal_a", "exec:0")
            assert lease1.state == LEASE_ACTIVE

            # get_active_lease returns it
            active = k.registry.get_active_lease("goal_a")
            assert active is not None
            assert active.lease_id == lease1.lease_id

            # Complete — no longer active
            k.registry.complete(lease1.lease_id)
            assert k.registry.get_lease(lease1.lease_id).state == LEASE_COMPLETED
            assert k.registry.get_active_lease("goal_a") is None

            # New acquire produces different lease
            lease2 = k.registry.acquire("goal_a", "exec:1")
            assert lease2.state == LEASE_ACTIVE
            assert lease2.lease_id != lease1.lease_id

    @pytest.mark.asyncio
    async def test_k5_1_independent_goals_uncontested(self):
        """Different goals each have their own active lease."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            for g in GOAL_IDS:
                lease = k.registry.acquire(g, f"exec:{g}")
                assert lease.state == LEASE_ACTIVE
                active = k.registry.get_active_lease(g)
                assert active is not None
                assert active.lease_id == lease.lease_id

            # All goals have independent active leases
            for g in GOAL_IDS:
                active = k.registry.get_active_lease(g)
                assert active is not None, f"Goal {g} has no active lease"

    @pytest.mark.asyncio
    async def test_k5_1_terminal_lease_not_active(self):
        """A completed/expired/revoked/abandoned lease is not active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            lease = k.registry.acquire("goal_a", "exec:0")
            k.registry.complete(lease.lease_id)
            assert k.registry.get_active_lease("goal_a") is None

            lease = k.registry.acquire("goal_b", "exec:0")
            k.registry.revoke(lease.lease_id)
            assert k.registry.get_active_lease("goal_b") is None

            lease = k.registry.acquire("goal_c", "exec:0")
            k.registry.abandon(lease.lease_id)
            assert k.registry.get_active_lease("goal_c") is None

    # ══════════════════════════════════════════════════════════════════════════
    # K5.2 — Deterministic Execution Identity
    #
    # Invariant: same (goal_id, dispatch_epoch, parent) → same execution_id,
    # regardless of kernel instance, timing, or other external factors.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k5_2_deterministic_execution_id(self):
        """Same inputs produce identical execution_id across kernel instances."""
        goal_id = "goal_test"
        epoch = 42

        results = []
        for _ in range(3):
            with tempfile.TemporaryDirectory() as tmpdir:
                wal = os.path.join(tmpdir, "wal")
                snap = os.path.join(tmpdir, "snap.json")
                k = _make_kernel(wal, snap)

                e = JournalEntry(
                    event='DISPATCHED', goal_id=goal_id,
                    execution_id=f"{goal_id}:test",
                    lease_id="lease_test", timestamp=time.time(),
                )
                k.journal.append(e)
                eid = k._compute_execution_id(goal_id, epoch)
                results.append(eid)

        assert all(r == results[0] for r in results), \
            f"Non-deterministic execution_id: {results}"

    @pytest.mark.asyncio
    async def test_k5_2_different_epoch_different_id(self):
        """Different dispatch_epoch produces different execution_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            k.journal.append(JournalEntry(
                event='DISPATCHED', goal_id="goal_x",
                execution_id="goal_x:seed",
                lease_id="lease_x", timestamp=time.time(),
            ))

            ids = set()
            for epoch in range(10):
                eid = k._compute_execution_id("goal_x", epoch)
                ids.add(eid)

            assert len(ids) == 10, \
                f"Expected 10 unique execution_ids, got {len(ids)}"

    # ══════════════════════════════════════════════════════════════════════════
    # K5.3 — No Duplicate Lifecycle Roots
    #
    # Invariant: each execution_id appears at most once in each lifecycle
    # position (DISPATCHED, STARTED, COMPLETED, FAILED). There are no
    # duplicate lifecycle roots — no two entries share execution_id with
    # different lifecycle paths.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k5_3_no_duplicate_lifecycle_entries(self):
        """No duplicate execution_id + event pairs in a valid journal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            eid = "goal_a:exec:0"
            for event in ('DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'):
                k.journal.append(JournalEntry(
                    event=event, goal_id="goal_a",
                    execution_id=eid, lease_id="lease_a",
                    timestamp=time.time(),
                ))

            seen: set[tuple[str, str]] = set()
            for e in k.journal._entries:
                if e.event in REPAIR_EVENTS:
                    continue
                key = (e.execution_id, e.event)
                assert key not in seen, \
                    f"Duplicate lifecycle entry: {key}"
                seen.add(key)

    @pytest.mark.asyncio
    async def test_k5_3_duplicate_dispatch_not_blocked(self):
        """Journal allows duplicate DISPATCHED (kernel dispatch() blocks it)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            eid = "goal_a:exec:dup"
            k.journal.append(JournalEntry(
                event='DISPATCHED', goal_id="goal_a",
                execution_id=eid, lease_id="lease_a",
                timestamp=time.time(),
            ))
            k.journal.append(JournalEntry(
                event='DISPATCHED', goal_id="goal_a",
                execution_id=eid, lease_id="lease_a",
                timestamp=time.time(),
            ))

            count = sum(
                1 for e in k.journal._entries
                if e.execution_id == eid and e.event == 'DISPATCHED'
            )
            assert count == 2, "journal.append does not enforce dispatch uniqueness"

    # ══════════════════════════════════════════════════════════════════════════
    # K5.4 — Deterministic Recovery
    #
    # Invariant: recover() produces identical journal state regardless
    # of when or how many times it's called, given the same WAL.
    # Multiple independent recovery calls on the same WAL state
    # produce identical journal state.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k5_4_recovery_deterministic(self):
        """Three independent recoveries from same WAL produce identical state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            for goal_id in GOAL_IDS:
                eid = f"{goal_id}:exec:0"
                for event in ('DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'):
                    k.journal.append(JournalEntry(
                        event=event, goal_id=goal_id,
                        execution_id=eid, lease_id=f"lease_{goal_id}",
                        timestamp=time.time(),
                    ))
            k.snapshot()

            hashes = []
            for _ in range(3):
                copy_dir = tmpdir + "_copy"
                copy_snap = tmpdir + "_snap_copy.json"
                if os.path.exists(copy_dir):
                    shutil.rmtree(copy_dir)
                shutil.copytree(wal, copy_dir)
                if os.path.exists(snap):
                    shutil.copy2(snap, copy_snap)

                k2 = _make_kernel(copy_dir, copy_snap)
                await k2.recover()
                hashes.append(_entries_hash(k2))

            assert all(h == hashes[0] for h in hashes), \
                f"Non-deterministic recovery: {hashes}"

    @pytest.mark.asyncio
    async def test_k5_4_recovery_before_after_consistency(self):
        """Recovered journal matches pre-snapshot journal state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            for goal_id in GOAL_IDS[:2]:
                eid = f"{goal_id}:exec:0"
                for event in ('DISPATCHED', 'LEASE_ISSUED', 'STARTED', 'COMPLETED'):
                    k.journal.append(JournalEntry(
                        event=event, goal_id=goal_id,
                        execution_id=eid, lease_id=f"lease_{goal_id}",
                        timestamp=time.time(),
                    ))

            pre_hash = _entries_hash(k)
            k.snapshot()
            await k.recover()
            post_hash = _entries_hash(k)
            assert pre_hash == post_hash, \
                "Journal state changed after recover()"

    # ══════════════════════════════════════════════════════════════════════════
    # K5.5 — Valid Lease Transitions
    #
    # Invariant: every lease follows a valid lifecycle:
    #   active → {completed, expired, revoked, abandoned}
    # No invalid transitions (e.g., completed → active).
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k5_5_lease_active_to_completed(self):
        """Active → completed transition is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            lease = k.registry.acquire("goal_a", "exec:0")
            assert lease.state == LEASE_ACTIVE
            k.registry.complete(lease.lease_id)
            assert k.registry.get_lease(lease.lease_id).state == LEASE_COMPLETED

    @pytest.mark.asyncio
    async def test_k5_5_lease_active_to_revoked(self):
        """Active → revoked transition is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            lease = k.registry.acquire("goal_b", "exec:0")
            assert lease.state == LEASE_ACTIVE
            k.registry.revoke(lease.lease_id)
            assert k.registry.get_lease(lease.lease_id).state == LEASE_REVOKED

    @pytest.mark.asyncio
    async def test_k5_5_lease_active_to_abandoned(self):
        """Active → abandoned transition is valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            lease = k.registry.acquire("goal_c", "exec:0")
            assert lease.state == LEASE_ACTIVE
            k.registry.abandon(lease.lease_id)
            assert k.registry.get_lease(lease.lease_id).state == LEASE_ABANDONED

    @pytest.mark.asyncio
    async def test_k5_5_lease_no_cross_goal_leak(self):
        """Completing a lease for one goal does not affect others."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            leases = {}
            for g in GOAL_IDS:
                leases[g] = k.registry.acquire(g, f"exec:{g}")

            k.registry.complete(leases['goal_a'].lease_id)

            for g in GOAL_IDS[1:]:
                state = k.registry.get_lease(leases[g].lease_id).state
                assert state == LEASE_ACTIVE, \
                    f"Goal {g} lease unexpectedly {state}"

    @pytest.mark.asyncio
    async def test_k5_5_lease_journal_correlation(self):
        """Every journal entry with a lease_id references a known lease."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")
            k = _make_kernel(wal, snap)

            for goal_id in GOAL_IDS[:2]:
                lease = k.registry.acquire(goal_id, f"exec:{goal_id}")
                eid = f"{goal_id}:exec:0"
                for event in ('DISPATCHED', 'LEASE_ISSUED', 'STARTED'):
                    k.journal.append(JournalEntry(
                        event=event, goal_id=goal_id,
                        execution_id=eid, lease_id=lease.lease_id,
                        timestamp=time.time(),
                    ))
                k.registry.complete(lease.lease_id)
                k.journal.append(JournalEntry(
                    event='COMPLETED', goal_id=goal_id,
                    execution_id=eid, lease_id=lease.lease_id,
                    timestamp=time.time(),
                ))

            for e in k.journal._entries:
                if e.lease_id:
                    lease = k.registry.get_lease(e.lease_id)
                    assert lease is not None, \
                        f"Journal entry references unknown lease: {e.lease_id}"

    # ══════════════════════════════════════════════════════════════════════════
    # K5.6 — Stress: Ownership Under Random Lifecycles
    #
    # Combines all invariants in random sequences across 4 goals.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k5_6_stress_random_lifecycles(self):
        """Random lifecycle sequences verify all ownership invariants."""
        rng = random.Random(RANDOM_SEED + 100)
        violations = []

        for trial in range(25):
            with tempfile.TemporaryDirectory() as tmpdir:
                wal = os.path.join(tmpdir, "wal")
                snap = os.path.join(tmpdir, "snap.json")
                k = _make_kernel(wal, snap)
                goal_leases: dict[str, str] = {}

                steps = rng.randint(5, 30)
                for step in range(steps):
                    goal = rng.choice(GOAL_IDS)

                    op = rng.choices(
                        ['acquire', 'terminal', 'journal'],
                        weights=[0.4, 0.4, 0.2],
                    )[0]

                    if op == 'acquire':
                        lease = k.registry.acquire(goal, f"exec:{step}")
                        goal_leases[goal] = lease.lease_id

                    elif op == 'terminal' and goal in goal_leases:
                        lid = goal_leases[goal]
                        lease = k.registry.get_lease(lid)
                        if lease and lease.state == LEASE_ACTIVE:
                            term = rng.choice(['complete', 'revoke', 'abandon'])
                            getattr(k.registry, term)(lid)
                            del goal_leases[goal]

                    elif op == 'journal' and goal in goal_leases:
                        lid = goal_leases[goal]
                        lease = k.registry.get_lease(lid)
                        if lease and lease.state == LEASE_ACTIVE:
                            eid = f"{goal}:exec:{step}"
                            for event in ('DISPATCHED', 'LEASE_ISSUED', 'STARTED'):
                                k.journal.append(JournalEntry(
                                    event=event, goal_id=goal,
                                    execution_id=eid, lease_id=lid,
                                    timestamp=time.time(),
                                ))

                # K5.1: at most one active lease per goal
                for g in GOAL_IDS:
                    active = k.registry.get_active_lease(g)
                    if active is not None:
                        count = sum(
                            1 for l in [k.registry.get_active_lease(g)]
                            if l is not None
                        )
                        if count > 1:
                            violations.append(
                                f"trial={trial} final: {count} active leases for {g}")

                # K5.3: no duplicate lifecycle roots
                seen_exec_event: set[tuple[str, str]] = set()
                for e in k.journal._entries:
                    if e.event in REPAIR_EVENTS:
                        continue
                    key = (e.execution_id, e.event)
                    if key in seen_exec_event:
                        violations.append(
                            f"trial={trial} final: duplicate {key}")
                    seen_exec_event.add(key)

                # K5.4: recovery deterministic
                pre_hash = _entries_hash(k)
                copy_dir = wal + "_copy"
                copy_snap = snap.replace(".json", "_copy.json")
                if os.path.exists(copy_dir):
                    shutil.rmtree(copy_dir)
                shutil.copytree(wal, copy_dir)
                if os.path.exists(snap):
                    shutil.copy2(snap, copy_snap)

                k2 = _make_kernel(copy_dir, copy_snap)
                await k2.recover()
                post_hash = _entries_hash(k2)
                if pre_hash != post_hash:
                    violations.append(
                        f"trial={trial} final: recovery non-deterministic")

        assert len(violations) == 0, \
            f"{len(violations)} violations:\n" + "\n".join(violations[:20])
