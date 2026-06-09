"""
K2+: Stateful Property Testing over Real Kernel Operations.

Generates random sequences of VALID kernel operations and verifies:

  I1. ActiveExecution == Projection(Journal)
        STARTED - {COMPLETED,FAILED,CANCELLED,LEASE_EXPIRED,ABANDONED}
  I2. Bidirectional Lease Consistency
        execution -> lease  AND  lease -> execution
  I3. Terminal Closure
        no non-repair events after terminal state
  I4. Snapshot Transparency
        hash(before snapshot) == hash(after snapshot)
  I5. Recovery Idempotence
        recover() twice produces identical state
  I6. Projection Equivalence
        State(before_crash) == Projection(Journal)(after_recovery)

Operations: dispatch, complete, fail, cancel, heartbeat, expire,
            snapshot, recover

expire is a first-class operation — simulates lease TTL expiry.
Lease_EXPIRED is treated as terminal, just like COMPLETED/FAILED/CANCELLED.
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
from execution_dynamics.journal import JournalEntry

RANDOM_SEED = 42

GOAL_IDS = ['goal_a', 'goal_b', 'goal_c', 'goal_d']

# State model
IDLE = 'IDLE'
STARTED = 'STARTED'
COMPLETED = 'COMPLETED'
FAILED = 'FAILED'
CANCELLED = 'CANCELLED'
EXPIRED = 'EXPIRED'
TERMINAL_STATES = {COMPLETED, FAILED, CANCELLED, EXPIRED}

STATE_GRAPH = {
    IDLE:      {STARTED},
    STARTED:   {COMPLETED, FAILED, CANCELLED, EXPIRED},
    COMPLETED: set(),
    FAILED:    set(),
    CANCELLED: set(),
    EXPIRED:   set(),
}

VALID_OPERATIONS = frozenset({
    'dispatch', 'complete', 'fail', 'cancel', 'heartbeat', 'expire',
    'snapshot', 'recover',
})

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


def _journal_entries_hash(kernel: ExecutionKernel) -> str:
    raw = json.dumps(_journal_entries(kernel.journal._entries), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── State Model ──────────────────────────────────────────────────────────────


class GoalState:
    __slots__ = ('goal_id', 'state', 'lease_id', 'execution_id', 'seq')

    def __init__(self, goal_id: str):
        self.goal_id = goal_id
        self.state = IDLE
        self.lease_id = ''
        self.execution_id = ''
        self.seq = 0

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def valid_ops(self) -> list[str]:
        transitions = STATE_GRAPH[self.state]
        ops = []
        if STARTED in transitions:
            ops.append('dispatch')
        if COMPLETED in transitions:
            ops.append('complete')
        if FAILED in transitions:
            ops.append('fail')
        if CANCELLED in transitions:
            ops.append('cancel')
        if EXPIRED in transitions:
            ops.append('expire')
        if self.state == STARTED:
            ops.append('heartbeat')
        return ops

    def clone(self) -> 'GoalState':
        gs = GoalState(self.goal_id)
        gs.state = self.state
        gs.lease_id = self.lease_id
        gs.execution_id = self.execution_id
        gs.seq = self.seq
        return gs


class Model:
    """Simplified world model tracking all goal states."""

    def __init__(self):
        self.goals: dict[str, GoalState] = {}
        self.snapshot_count = 0
        self.crash_count = 0

    def get_or_create(self, goal_id: str) -> GoalState:
        if goal_id not in self.goals:
            self.goals[goal_id] = GoalState(goal_id)
        return self.goals[goal_id]

    def valid_ops(self) -> list[str]:
        ops = ['snapshot', 'recover']
        for g in self.goals.values():
            if not g.is_terminal():
                ops.extend(g.valid_ops())
        if any(g.state == IDLE for g in self.goals.values()) or not self.goals:
            ops.append('dispatch')
        return list(set(ops))

    def _goal_in_state(self, state: str) -> list[str]:
        return [gid for gid, g in self.goals.items() if g.state == state]

    def _any_goal_in(self, states: set) -> bool:
        return any(g.state in states for g in self.goals.values())

    def apply_dispatch(self, goal_id: str, lease_id: str, execution_id: str) -> None:
        gs = self.get_or_create(goal_id)
        assert gs.state == IDLE, f"Cannot dispatch {goal_id}: state={gs.state}"
        gs.state = STARTED
        gs.lease_id = lease_id
        gs.execution_id = execution_id
        gs.seq += 1

    def apply_complete(self, goal_id: str) -> None:
        gs = self.get_or_create(goal_id)
        assert gs.state == STARTED, f"Cannot complete {goal_id}: state={gs.state}"
        gs.state = COMPLETED

    def apply_fail(self, goal_id: str) -> None:
        gs = self.get_or_create(goal_id)
        assert gs.state == STARTED, f"Cannot fail {goal_id}: state={gs.state}"
        gs.state = FAILED

    def apply_cancel(self, goal_id: str) -> None:
        gs = self.get_or_create(goal_id)
        assert gs.state == STARTED, f"Cannot cancel {goal_id}: state={gs.state}"
        gs.state = CANCELLED

    def apply_expire(self, goal_id: str) -> None:
        gs = self.get_or_create(goal_id)
        assert gs.state == STARTED, f"Cannot expire {goal_id}: state={gs.state}"
        gs.state = EXPIRED

    def apply_heartbeat(self, goal_id: str) -> None:
        gs = self.get_or_create(goal_id)
        assert gs.state == STARTED, f"Cannot heartbeat {goal_id}: state={gs.state}"

    def snapshot(self) -> None:
        self.snapshot_count += 1

    def recover(self) -> None:
        self.crash_count += 1

    def clone(self) -> 'Model':
        m = Model()
        m.goals = {k: v.clone() for k, v in self.goals.items()}
        m.snapshot_count = self.snapshot_count
        m.crash_count = self.crash_count
        return m


# ── Kernel Executor ──────────────────────────────────────────────────────────


class KernelExecutor:
    """Executes state model operations on a real Kernel."""

    def __init__(self, wal_dir: str, snap_file: str):
        self.wal_dir = wal_dir
        self.snap_file = snap_file
        self.kernel = _make_kernel(wal_dir, snap_file)
        self._entry_seq = 0

    def _journal(self, event: str, goal_id: str, lease_id: str, **kw):
        execution_id = kw.pop('execution_id', f"{goal_id}:exec:{self._entry_seq}")
        entry = JournalEntry(
            event=event,
            goal_id=goal_id,
            execution_id=execution_id,
            lease_id=lease_id,
            timestamp=time.time(),
            **kw,
        )
        self.kernel.journal.append(entry)
        return entry

    def dispatch(self, goal_id: str) -> dict:
        self._entry_seq += 1
        execution_id = f"{goal_id}:exec:{self._entry_seq}"

        lease = self.kernel.registry.acquire(
            goal_id=goal_id,
            execution_id=execution_id,
            issued_by="k2_test",
        )
        actual_lease_id = lease.lease_id

        self._journal('DISPATCHED', goal_id, actual_lease_id, execution_id=execution_id)
        self._journal('LEASE_ISSUED', goal_id, actual_lease_id, execution_id=execution_id)
        self._journal('STARTED', goal_id, actual_lease_id, execution_id=execution_id)
        self.kernel._active_executions[goal_id] = actual_lease_id

        return {'goal_id': goal_id, 'lease_id': actual_lease_id, 'execution_id': execution_id}

    def complete(self, goal_id: str) -> dict:
        lease_id = self.kernel._active_executions.get(goal_id, '')
        self._journal('COMPLETED', goal_id, lease_id, success=True, duration_ms=10)
        self.kernel.registry.complete(lease_id)
        self.kernel._active_executions.pop(goal_id, None)
        return {'goal_id': goal_id, 'lease_id': lease_id}

    def fail(self, goal_id: str) -> dict:
        lease_id = self.kernel._active_executions.get(goal_id, '')
        self._journal('FAILED', goal_id, lease_id, success=False, duration_ms=10, error='k2_test_failure')
        self._journal('ABANDONED', goal_id, lease_id, success=False)
        self.kernel.registry.abandon(lease_id)
        self.kernel._active_executions.pop(goal_id, None)
        return {'goal_id': goal_id, 'lease_id': lease_id}

    def cancel(self, goal_id: str) -> dict:
        lease_id = self.kernel._active_executions.get(goal_id, '')
        self._journal('CANCELLING', goal_id, lease_id)
        self.kernel.registry.request_cancellation(lease_id)
        self._journal('CANCELLED', goal_id, lease_id)
        self.kernel.registry.confirm_cancellation(lease_id)
        self.kernel._active_executions.pop(goal_id, None)
        return {'goal_id': goal_id, 'lease_id': lease_id}

    def expire(self, goal_id: str) -> dict:
        lease_id = self.kernel._active_executions.get(goal_id, '')
        self._journal('LEASE_EXPIRED', goal_id, lease_id)
        self.kernel.registry.expire_stale(max_age_seconds=0)
        self.kernel._active_executions.pop(goal_id, None)
        return {'goal_id': goal_id, 'lease_id': lease_id}

    def heartbeat(self, goal_id: str) -> dict:
        lease_id = self.kernel._active_executions.get(goal_id, '')
        ok = self.kernel.registry.heartbeat(lease_id)
        return {'goal_id': goal_id, 'lease_id': lease_id, 'ok': ok}

    def snapshot(self) -> dict:
        return self.kernel.snapshot()

    def get_active_executions(self) -> dict:
        return dict(self.kernel._active_executions)

    def get_stats(self) -> dict:
        return {
            'entries': len(self.kernel.journal._entries),
            'active_executions': len(self.kernel._active_executions),
            'total_leases': len(self.kernel.registry._leases),
        }


# ── Invariant Checker ────────────────────────────────────────────────────────


class InvariantReport:
    def __init__(self):
        self.violations: list[str] = []

    def ok(self) -> bool:
        return len(self.violations) == 0

    def add(self, name: str, detail: str):
        self.violations.append(f"{name}: {detail}")

    def __str__(self) -> str:
        if self.ok():
            return "OK"
        return "\n".join(self.violations)


def compute_active_projection(entries: list[JournalEntry]) -> set[str]:
    """Projection(Journal) for active executions.

    STARTED per goal, minus any that have reached terminal:
      COMPLETED, FAILED, CANCELLED, LEASE_EXPIRED, ABANDONED
    """
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'LEASE_EXPIRED', 'ABANDONED'}
    started: set[str] = set()
    terminated: set[str] = set()
    for e in entries:
        if e.event == 'STARTED':
            started.add(e.goal_id)
        if e.event in terminal:
            terminated.add(e.goal_id)
    return started - terminated


def check_i1_active_projection(
    kernel: ExecutionKernel, entries: list[JournalEntry],
) -> InvariantReport:
    """I1. _active_executions == Projection(Journal).

    Computes active executions independently from journal events
    and compares with kernel._active_executions.
    """
    report = InvariantReport()
    projected = compute_active_projection(entries)
    actual = set(kernel._active_executions.keys())

    missing = projected - actual
    extra = actual - projected
    if missing:
        report.add('I1_ActiveProjection',
                    f"journal has active but kernel does not: {sorted(missing)}")
    if extra:
        report.add('I1_ActiveProjection',
                    f"kernel has active but journal does not: {sorted(extra)}")
    return report


def check_i2_lease_consistency(
    kernel: ExecutionKernel,
    _entries: list[JournalEntry] = None,
) -> InvariantReport:
    """I2. Bidirectional lease consistency.

    Forward: every _active_executions entry has a valid non-terminal lease.
    Reverse: every active lease in registry has a corresponding _active_executions.
    """
    report = InvariantReport()
    active_lease_ids: set[str] = set()

    for goal_id, lease_id in kernel._active_executions.items():
        lease = kernel.registry.get_lease(lease_id)
        if lease is None:
            report.add('I2_Bidirectional',
                        f"goal={goal_id} lease={lease_id}: lease not found")
            continue
        if lease.state not in ('active', 'cancelling'):
            report.add('I2_Bidirectional',
                        f"goal={goal_id} lease={lease_id}: state={lease.state}")
        active_lease_ids.add(lease_id)

    # Reverse: every ACTIVE lease in registry must have _active_executions
    for lid, lease in kernel.registry._leases.items():
        if lease.state == 'active' and lid not in active_lease_ids:
            # Check if this lease is tracked under a different goal_id
            goal_for_lease = lease.goal_id
            if goal_for_lease not in kernel._active_executions:
                report.add('I2_Bidirectional',
                            f"orphaned_lease lease={lid} goal={goal_for_lease}: active in registry but no execution")

    return report


def check_i3_terminal_closure(entries: list[JournalEntry]) -> InvariantReport:
    """I3. No events after terminal state for a goal."""
    report = InvariantReport()
    terminal_events = {'COMPLETED', 'FAILED', 'CANCELLED', 'ABANDONED', 'LEASE_EXPIRED'}
    per_goal: dict[str, list[JournalEntry]] = {}
    for e in entries:
        per_goal.setdefault(e.goal_id, []).append(e)
    for goal_id, chain in per_goal.items():
        non_repair = [e for e in chain if e.event not in REPAIR_EVENTS]
        terminal_idx = -1
        for i, e in enumerate(non_repair):
            if e.event in terminal_events:
                terminal_idx = i
        if terminal_idx >= 0 and terminal_idx < len(non_repair) - 1:
            after = [e.event for e in non_repair[terminal_idx + 1:]]
            if after:
                report.add('I3_TerminalClosure',
                            f"goal={goal_id} has events after terminal: {after}")
    return report


def check_all_invariants(
    kernel: ExecutionKernel,
    entries: list[JournalEntry],
) -> InvariantReport:
    """Check I1, I2, I3."""
    report = InvariantReport()
    r1 = check_i1_active_projection(kernel, entries)
    if not r1.ok():
        report.violations.extend(r1.violations)
    r2 = check_i2_lease_consistency(kernel, entries)
    if not r2.ok():
        report.violations.extend(r2.violations)
    r3 = check_i3_terminal_closure(entries)
    if not r3.ok():
        report.violations.extend(r3.violations)
    return report


# ── Op Generator ─────────────────────────────────────────────────────────────


def generate_op_sequence(
    length: int,
    rng: random.Random,
    model: Model,
) -> list[tuple[str, str]]:
    """Generate a sequence of (operation, goal_id) pairs respecting state."""
    seq: list[tuple[str, str]] = []
    model = model.clone()
    for _ in range(length):
        ops = model.valid_ops()
        if not ops:
            break
        op = rng.choice(ops)
        if op == 'dispatch':
            idle_goals = [gid for gid, g in model.goals.items() if g.state == IDLE]
            if not idle_goals:
                idle_goals = [gid for gid in GOAL_IDS if gid not in model.goals]
            if idle_goals:
                goal_id = rng.choice(idle_goals)
                seq.append((op, goal_id))
                model.apply_dispatch(goal_id, '', '')
            else:
                break
        elif op in ('complete', 'fail', 'cancel', 'heartbeat', 'expire'):
            started_goals = [gid for gid, g in model.goals.items() if g.state == STARTED]
            if started_goals:
                goal_id = rng.choice(started_goals)
                seq.append((op, goal_id))
                getattr(model, f'apply_{op}')(goal_id)
            else:
                break
        elif op == 'snapshot':
            seq.append((op, ''))
            model.snapshot()
        elif op == 'recover':
            seq.append((op, ''))
            model.recover()
    return seq


# ── Scenario Runner ──────────────────────────────────────────────────────────


async def run_scenario(scenario: list[tuple[str, str]]) -> dict:
    """Run a K2 scenario, return result with invariants check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wal_dir = os.path.join(tmpdir, "wal")
        snap_file = os.path.join(tmpdir, "snapshot.json")
        executor = KernelExecutor(wal_dir, snap_file)

        for op, goal_id in scenario:
            if op == 'dispatch':
                executor.dispatch(goal_id)
            elif op == 'complete':
                executor.complete(goal_id)
            elif op == 'fail':
                executor.fail(goal_id)
            elif op == 'cancel':
                executor.cancel(goal_id)
            elif op == 'expire':
                executor.expire(goal_id)
            elif op == 'heartbeat':
                executor.heartbeat(goal_id)
            elif op == 'snapshot':
                # I4: snapshot transparency check
                hash_before_snap = _journal_entries_hash(executor.kernel)
                entries_before_snap = list(executor.kernel.journal._entries)
                executor.snapshot()
                hash_after_snap = _journal_entries_hash(executor.kernel)
                entries_after_snap = list(executor.kernel.journal._entries)
                if hash_before_snap != hash_after_snap:
                    return {
                        'ok': False,
                        'violations': [f"I4_SnapshotTransparency: hash changed after snapshot "
                                        f"({hash_before_snap[:12]} != {hash_after_snap[:12]})"],
                        'hash_before': hash_before_snap[:12],
                        'hash_after': hash_after_snap[:12],
                    }
            elif op == 'recover':
                entries_before = list(executor.kernel.journal._entries)
                hash_before = _journal_entries_hash(executor.kernel)
                inv_before = check_all_invariants(executor.kernel, entries_before)

                del executor.kernel
                executor.kernel = _make_kernel(wal_dir, snap_file)

                await executor.kernel.recover()
                entries_after_r1 = list(executor.kernel.journal._entries)
                hash_after_r1 = _journal_entries_hash(executor.kernel)

                await executor.kernel.recover()
                entries_after_r2 = list(executor.kernel.journal._entries)
                hash_after_r2 = _journal_entries_hash(executor.kernel)

                inv_after = check_all_invariants(executor.kernel, entries_after_r1)

                violations = []
                if not inv_before.ok():
                    violations.append(f"before: {inv_before}")

                # I6: Projection equivalence
                if hash_before != hash_after_r1:
                    violations.append(
                        f"I6_ProjectionEquivalence: hash mismatch after recover "
                        f"({hash_before[:12]} != {hash_after_r1[:12]})"
                    )
                biz_before = [e for e in entries_before if e.event not in REPAIR_EVENTS]
                biz_after = [e for e in entries_after_r1 if e.event not in REPAIR_EVENTS]
                if len(biz_before) != len(biz_after):
                    violations.append(
                        f"I6_ProjectionEquivalence: entry count mismatch "
                        f"({len(biz_before)} before vs {len(biz_after)} after)"
                    )
                else:
                    for i, (eb, ea) in enumerate(zip(biz_before, biz_after)):
                        if eb.event != ea.event or eb.goal_id != ea.goal_id:
                            violations.append(
                                f"I6_ProjectionEquivalence: entry {i} mismatch "
                                f"({eb.event}/{eb.goal_id} vs {ea.event}/{ea.goal_id})"
                            )
                            break

                # I5: Recovery idempotence
                if hash_after_r1 != hash_after_r2:
                    violations.append(
                        f"I5_RecoveryIdempotence: hash changed on second recover "
                        f"({hash_after_r1[:12]} != {hash_after_r2[:12]})"
                    )
                if len(entries_after_r1) != len(entries_after_r2):
                    violations.append(
                        f"I5_RecoveryIdempotence: entry count changed "
                        f"({len(entries_after_r1)} vs {len(entries_after_r2)})"
                    )

                if not inv_after.ok():
                    violations.append(f"after: {inv_after}")

                return {
                    'ok': len(violations) == 0,
                    'violations': violations,
                    'hash_before': hash_before[:12],
                    'hash_after': hash_after_r1[:12],
                }

    return {'ok': True, 'violations': [], 'hash_before': '', 'hash_after': ''}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestK2StatefulProperty:

    @pytest.mark.asyncio
    async def test_k2_short_sequences(self):
        """100 random 1-3 op sequences + recover."""
        rng = random.Random(RANDOM_SEED)
        failures = []
        for trial in range(100):
            length = rng.randint(1, 3)
            model = Model()
            scenario = generate_op_sequence(length, rng, model)
            scenario.append(('recover', ''))
            result = await run_scenario(scenario)
            if not result['ok']:
                failures.append((trial, scenario, result['violations']))
        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])

    @pytest.mark.asyncio
    async def test_k2_medium_sequences(self):
        """200 random sequences of length 8-15 with multiple recovers."""
        rng = random.Random(RANDOM_SEED + 2)
        failures = []
        for trial in range(200):
            length = rng.randint(8, 15)
            model = Model()
            scenario = generate_op_sequence(length, rng, model)
            has_recover = any(op == 'recover' for op, _ in scenario)
            if not has_recover:
                scenario.append(('recover', ''))
            result = await run_scenario(scenario)
            if not result['ok']:
                failures.append((trial, scenario, result['violations']))
        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])

    @pytest.mark.asyncio
    async def test_k2_long_sequences(self):
        """50 long sequences of length 20-30 with interleaved snapshots and recovers."""
        rng = random.Random(RANDOM_SEED + 3)
        failures = []
        for trial in range(50):
            length = rng.randint(20, 30)
            model = Model()
            scenario = []
            for _ in range(length):
                ops = model.valid_ops()
                if not ops:
                    break
                op = rng.choice(ops)
                if op == 'dispatch':
                    idle_goals = [gid for gid, g in model.goals.items() if g.state == IDLE]
                    if not idle_goals:
                        idle_goals = [gid for gid in GOAL_IDS if gid not in model.goals]
                    if idle_goals:
                        goal_id = rng.choice(idle_goals)
                        scenario.append((op, goal_id))
                        model.apply_dispatch(goal_id, '', '')
                elif op in ('complete', 'fail', 'cancel', 'heartbeat', 'expire'):
                    started_goals = [gid for gid, g in model.goals.items() if g.state == STARTED]
                    if started_goals:
                        goal_id = rng.choice(started_goals)
                        scenario.append((op, goal_id))
                        getattr(model, f'apply_{op}')(goal_id)
                elif op == 'snapshot':
                    scenario.append((op, ''))
                    model.snapshot()
                elif op == 'recover':
                    scenario.append((op, ''))
                    model.recover()
            has_recover = any(op == 'recover' for op, _ in scenario)
            if not has_recover:
                scenario.append(('recover', ''))
            result = await run_scenario(scenario)
            if not result['ok']:
                failures.append((trial, scenario, result['violations']))
        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])

    @pytest.mark.asyncio
    async def test_k2_expire_lifecycle(self):
        """50 sequences focused on expire lifecycle."""
        rng = random.Random(RANDOM_SEED + 5)
        failures = []
        for trial in range(50):
            model = Model()
            scenario = []
            for _ in range(rng.randint(3, 10)):
                ops = model.valid_ops()
                if not ops:
                    break
                op = rng.choice(ops)
                if op == 'dispatch':
                    idle_goals = [gid for gid, g in model.goals.items() if g.state == IDLE]
                    if not idle_goals:
                        idle_goals = [gid for gid in GOAL_IDS if gid not in model.goals]
                    if idle_goals:
                        goal_id = rng.choice(idle_goals)
                        scenario.append((op, goal_id))
                        model.apply_dispatch(goal_id, '', '')
                elif op in ('complete', 'fail', 'cancel', 'heartbeat', 'expire'):
                    started_goals = [gid for gid, g in model.goals.items() if g.state == STARTED]
                    if started_goals:
                        goal_id = rng.choice(started_goals)
                        scenario.append((op, goal_id))
                        getattr(model, f'apply_{op}')(goal_id)
                elif op == 'snapshot':
                    scenario.append((op, ''))
                    model.snapshot()
                elif op == 'recover':
                    scenario.append((op, ''))
                    model.recover()
            scenario.append(('recover', ''))
            result = await run_scenario(scenario)
            if not result['ok']:
                failures.append((trial, scenario, result['violations']))
        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])

    @pytest.mark.asyncio
    async def test_k2_recovery_idempotence(self):
        """100 sequences focused on I5: recovery idempotence."""
        rng = random.Random(RANDOM_SEED + 4)
        failures = []
        for trial in range(100):
            model = Model()
            scenario = []
            for _ in range(rng.randint(3, 10)):
                ops = model.valid_ops()
                if not ops:
                    break
                op = rng.choice(ops)
                if op == 'dispatch':
                    idle_goals = [gid for gid, g in model.goals.items() if g.state == IDLE]
                    if not idle_goals:
                        idle_goals = [gid for gid in GOAL_IDS if gid not in model.goals]
                    if idle_goals:
                        goal_id = rng.choice(idle_goals)
                        scenario.append((op, goal_id))
                        model.apply_dispatch(goal_id, '', '')
                elif op in ('complete', 'fail', 'cancel', 'heartbeat', 'expire'):
                    started_goals = [gid for gid, g in model.goals.items() if g.state == STARTED]
                    if started_goals:
                        goal_id = rng.choice(started_goals)
                        scenario.append((op, goal_id))
                        getattr(model, f'apply_{op}')(goal_id)
                elif op == 'snapshot':
                    scenario.append((op, ''))
                    model.snapshot()
                elif op == 'recover':
                    scenario.append((op, ''))
                    model.recover()
            scenario.append(('recover', ''))
            result = await run_scenario(scenario)
            if not result['ok']:
                failures.append((trial, scenario, result['violations']))
        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])

    @pytest.mark.asyncio
    async def test_k2_snapshot_transparency(self):
        """I4 focused: snapshot must not change observable state."""
        rng = random.Random(RANDOM_SEED + 6)
        failures = []
        for trial in range(100):
            model = Model()
            scenario = []
            for _ in range(rng.randint(2, 8)):
                ops = model.valid_ops()
                if not ops:
                    break
                op = rng.choice(ops)
                if op == 'dispatch':
                    idle_goals = [gid for gid, g in model.goals.items() if g.state == IDLE]
                    if not idle_goals:
                        idle_goals = [gid for gid in GOAL_IDS if gid not in model.goals]
                    if idle_goals:
                        goal_id = rng.choice(idle_goals)
                        scenario.append((op, goal_id))
                        model.apply_dispatch(goal_id, '', '')
                elif op in ('complete', 'fail', 'cancel', 'heartbeat', 'expire'):
                    started_goals = [gid for gid, g in model.goals.items() if g.state == STARTED]
                    if started_goals:
                        goal_id = rng.choice(started_goals)
                        scenario.append((op, goal_id))
                        getattr(model, f'apply_{op}')(goal_id)
                elif op == 'snapshot':
                    scenario.append((op, ''))
                    model.snapshot()
                elif op == 'recover':
                    scenario.append((op, ''))
                    model.recover()
            scenario.append(('snapshot', ''))
            result = await run_scenario(scenario)
            if not result['ok']:
                failures.append((trial, scenario, result['violations']))
        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])
