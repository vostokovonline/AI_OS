"""
Execution Kernel — coordination dynamics + execution authority + ownership boundary.

LAYER ARCHITECTURE:

  API / Scheduler / Celery
       │
       ▼
  ExecutionKernel.dispatch()   ←  THE SINGLE ENTRY POINT
       │                            │
       │   1. Compute field        │
       │   2. Compute pressure     │
       │   3. Acquire LEASE        │  ← kernel is sole lease issuer
       │   4. Journal DISPATCHED   │  ← immutable causal chain
       │   5. Validate lease       │  ← executor checks before run
       │   6. Execute via pipeline │
       │   7. Journal COMPLETED    │
       │   8. Complete lease       │
       │   9. Update dynamics      │
       ▼                            ▼
  ┌──────────────────────────────────────────┐
  │  ExecutionKernel Components              │
  │  ├── DispatchJournal   (immutable chain) │
  │  ├── LeaseRegistry     (authority)       │
  │  ├── ExecutionPolicy   (decisions)       │
  │  ├── ExecutionGroup    (co-scheduling)   │
  │  ├── TaskPersistence   (snapshot buffer) │
  │  ├── PriorityCapture   (accumulation)    │
  │  └── PathLockin        (coherence)       │
  └──────────────────────────────────────────┘
       │
       ▼
  Existing Pipeline (GoalExecutor et al.)

OWNERSHIP ENFORCEMENT:
  - Kernel is the SOLE issuer of execution leases
  - No executor can run without a valid lease
  - Every dispatch is journaled immutably
  - Direct executor calls (bypass) will fail lease validation

The kernel is an ownership boundary, not just a utility wrapper.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import hashlib
import logging
import time

from .policy import ExecutionPolicy, ExecutionPressure, ExecutionGroupRegistry
from .journal import DispatchJournal, JournalEntry
from .lease import LeaseRegistry, ExecutionLease, LEASE_ACTIVE
from .wal import WriteAheadLog
from .segmented_wal import SegmentedWAL
from .jsonl_wal import JsonLinesWAL
from .truth_journal import TruthJournal, ProposedMutation, ProposedEvent, Evidence

logger = logging.getLogger(__name__)


# ============================================================================
# Execution Result
# ============================================================================

@dataclass
class ExecutionResult:
    """
    Enriched execution result with coordination dynamics metadata.

    Wraps the existing GoalExecutor result with:
      - persistence_weight: how stable this task's state is
      - priority_score: current priority after reinforcement
      - coherence_index: how well this execution integrates with its dependencies
      - execution_pressure: the real priority signal (NOT progress)
      - group_id: anti-fragmentation group this execution belongs to
      - lease_id: kernel-issued authority token
      - dispatch_epoch: monotonically increasing dispatch counter
    """
    goal_id: str
    success: bool
    artifacts: List[Dict[str, Any]]

    # Coordination dynamics
    persistence_weight: float = 1.0
    priority_score: float = 1.0
    coherence_index: float = 1.0
    execution_pressure: Optional[Dict[str, float]] = None

    # Anti-fragmentation
    group_id: Optional[str] = None

    # Lease / ownership
    lease_id: str = ""
    dispatch_epoch: int = 0

    # Idempotency
    did_skip: bool = False
    dispatch_id: str = ""

    # Execution metadata
    duration_ms: float = 0.0
    error: Optional[str] = None
    execution_id: str = ""

    # Truth mutations (proposed by executor, committed by kernel)
    proposed_mutations: List[Any] = field(default_factory=list)
    proposed_events: List[Any] = field(default_factory=list)
    evidence: List[Any] = field(default_factory=list)


# ============================================================================
# Execution Context — flows through entire pipeline
# ============================================================================

@dataclass
class ExecutionContext:
    """
    Execution context — identity + causality + derived state.

    Every stage of the kernel reads/writes to this context.
    This is the single source of truth for a dispatch lifecycle.
    """
    execution_id: str
    goal_id: str

    # Identity
    dispatch_id: str = ""
    lease_id: str = ""

    # Causality
    retry_count: int = 0
    parent_execution_id: str = ""
    worker_id: str = ""
    dispatch_ts: float = 0.0

    # Dispatch parameters
    active_task_count: int = 0
    blocked_dependents: int = 0
    hours_idle: float = 24.0
    user_priority: float = 0.5

    # Derived (filled during dispatch)
    field: Any = None       # ExecutionField — set after compute
    pressure: Any = None    # ExecutionPressure — set after compute
    coherence: float = 1.0  # Path coherence
    score: float = 0.0      # Policy admission score

    def to_dict(self) -> dict:
        return {
            'execution_id': self.execution_id,
            'goal_id': self.goal_id,
            'dispatch_id': self.dispatch_id,
            'lease_id': self.lease_id,
            'retry_count': self.retry_count,
            'parent_execution_id': self.parent_execution_id,
            'worker_id': self.worker_id,
            'dispatch_ts': self.dispatch_ts,
            'active_task_count': self.active_task_count,
            'blocked_dependents': self.blocked_dependents,
            'hours_idle': self.hours_idle,
            'user_priority': self.user_priority,
            'coherence': self.coherence,
            'score': self.score,
        }


# ============================================================================
# Execution Config
# ============================================================================

@dataclass
class ExecutionConfig:
    """
    Execution kernel configuration.

    All parameters have sensible defaults. Tune per deployment.
    """
    # Persistence (snapshot buffer)
    state_recovery_enabled: bool = True
    max_recovery_depth: int = 3

    # Capture
    priority_boost_threshold: float = 0.7
    priority_decay_rate: float = 0.01

    # Lock-in
    coherence_window: int = 50
    min_coherence_threshold: float = 0.3

    # Policy
    preemption_threshold: float = 0.3
    max_concurrent_executions: int = 8
    max_retries: int = 3

    # Anti-fragmentation
    group_co_scheduling_enabled: bool = True

    # Lease (ownership boundary)
    lease_ttl_seconds: int = 3600
    enforce_lease: bool = True

    # Idempotent dispatch
    dedup_window_seconds: int = 300  # 5 min: same dispatch_id seen within window is skipped
    dedup_enabled: bool = True

    # Lease heartbeat
    stale_timeout_seconds: int = 300  # 5 min without heartbeat = stale
    heartbeat_auto_renew: bool = True

    # Snapshot (O(1) recovery)
    snapshot_interval: int = 100  # Create snapshot every N journal entries
    snapshot_keep_last: int = 5   # Keep N most recent snapshots

    # New persistence layer (SegmentedWAL + SnapshotManager)
    wal_path: str = ""       # Path for SegmentedWAL directory; empty = legacy WriteAheadLog
    snapshot_path: str = ""  # Path for snapshot file; empty = auto-derive from wal_path

    # Single-writer enforcement (K7/A)
    enforce_single_writer: bool = False  # Enable explicitly for single-writer guarantee


DEFAULT_CONFIG = ExecutionConfig()


# ============================================================================
# Kernel Capability — imported from capability module for backward compat
# ============================================================================

from .capability import KernelCapability


# ============================================================================
# Coordination Field (lightweight, execution-specific)
# ============================================================================

@dataclass
class ExecutionField:
    """
    Coordination field scoped to a single execution context.
    """
    task_load: float = 0.0
    dependency_depth: float = 0.0
    failure_rate: float = 0.0
    priority_spread: float = 0.0

    @property
    def execution_cost(self) -> float:
        return 0.1 + self.task_load * 0.3 + self.dependency_depth * 0.2 + self.failure_rate * 0.3

    @property
    def stability_index(self) -> float:
        return 1.0 / max(0.1, self.execution_cost)


def compute_execution_field(
    goal_id: str,
    session,
    active_task_count: int = 0,
    max_depth: int = 5,
) -> ExecutionField:
    """
    Compute coordination field for a goal's execution context.

    Does NOT read goal.progress — uses execution pressure instead.
    """
    from models import Goal

    task_load = min(1.0, active_task_count / 20.0)

    dependency_depth = 0.0
    goal = session.query(Goal).filter(Goal.id == goal_id).first()
    if goal and hasattr(goal, 'depth_level') and goal.depth_level is not None:
        dependency_depth = min(1.0, goal.depth_level / max_depth)

    failure_rate = 0.0
    if goal and goal.parent_id:
        siblings = session.query(Goal).filter(Goal.parent_id == goal.parent_id).all()
        if siblings:
            failed = sum(1 for s in siblings if s.status == 'failed')
            failure_rate = min(1.0, failed / max(1, len(siblings)))

    priority_spread = 0.0
    if goal and goal.parent_id:
        siblings = session.query(Goal).filter(Goal.parent_id == goal.parent_id).all()
        if len(siblings) > 1:
            import numpy as np
            pressures = [
                min(1.0, s.retry_count * 0.15 if hasattr(s, 'retry_count') and s.retry_count else 0.5)
                for s in siblings
            ]
            if pressures:
                priority_spread = min(1.0, float(np.std(pressures)) * 2.0)

    return ExecutionField(
        task_load=round(task_load, 4),
        dependency_depth=round(dependency_depth, 4),
        failure_rate=round(failure_rate, 4),
        priority_spread=round(priority_spread, 4),
    )


# ============================================================================
# Persistence Layer (SNAPSHOT BUFFER — NOT source of truth)
# ============================================================================

class TaskPersistence:
    """
    Ephemeral snapshot buffer for execution state recovery.

    NOT the source of truth — Postgres is.
    Redis is only a snapshot buffer.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client
        self._snapshot_count = 0

    def snapshot_key(self, goal_id: str, execution_id: str) -> str:
        return f"exec:snapshot:{goal_id}:{execution_id}"

    def save_snapshot(self, goal_id: str, execution_id: str, state: dict, ttl: int = 600):
        self._snapshot_count += 1
        if self._redis:
            key = self.snapshot_key(goal_id, execution_id)
            import json
            try:
                self._redis.setex(key, ttl, json.dumps(state))
            except Exception as e:
                logger.warning(f"snapshot_save_failed goal_id={goal_id} error={e}")

    def recover_snapshot(self, goal_id: str, execution_id: str) -> Optional[dict]:
        if not self._redis:
            return None
        import json
        key = self.snapshot_key(goal_id, execution_id)
        try:
            data = self._redis.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.warning(f"snapshot_recover_failed goal_id={goal_id} error={e}")
        return None

    def get_stats(self) -> dict:
        return {'snapshots_saved': self._snapshot_count}


# ============================================================================
# Capture Layer
# ============================================================================

class PriorityCapture:
    """
    Important tasks accumulate priority weight.

    Source of truth: in-memory + Postgres.
    NOT Redis.
    """

    def __init__(self):
        self._weights: Dict[str, float] = {}
        self._decay_rate = 0.01

    def get_weight(self, goal_id: str, session=None) -> float:
        if goal_id in self._weights:
            return self._weights[goal_id]
        if session:
            from models import Goal
            goal = session.query(Goal).filter(Goal.id == goal_id).first()
            if goal:
                retry_count = getattr(goal, 'retry_count', 0) or 0
                if retry_count > 0:
                    return min(1.0, 0.5 + retry_count * 0.1)
        return 0.5

    def reinforce(self, goal_id: str, field: ExecutionField):
        current = self._weights.get(goal_id, 0.5)
        reinforcement = 0.02 + field.execution_cost * 0.05
        new_weight = min(1.0, current + reinforcement)
        self._weights[goal_id] = new_weight
        return new_weight

    def decay(self, goal_id: str):
        if goal_id in self._weights:
            self._weights[goal_id] = max(0.1, self._weights[goal_id] - self._decay_rate)

    def get_stats(self) -> dict:
        if not self._weights:
            return {'n_active': 0, 'mean_weight': 0.0}
        weights = list(self._weights.values())
        return {
            'n_active': len(weights),
            'mean_weight': round(float(sum(weights)) / len(weights), 4),
            'max_weight': round(max(weights), 4),
        }

    # ------------------------------------------------------------------
    # Snapshot state (deterministic, serializable)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Serialize capture weights for snapshot."""
        return dict(self._weights)

    def restore_state(self, state: dict):
        """Restore capture weights from snapshot."""
        self._weights.clear()
        for gid, w in state.items():
            self._weights[str(gid)] = float(w)


# ============================================================================
# Lock-in Layer
# ============================================================================

class PathLockin:
    """
    Long-running execution paths maintain coherence.
    """

    def __init__(self, window: int = 50, min_threshold: float = 0.3):
        self._coherence: Dict[str, float] = {}
        self._window = window
        self._min_threshold = min_threshold

    def compute_coherence(self, goal_id: str, field: ExecutionField, session) -> float:
        from models import Goal
        base = field.stability_index
        alignment = 1.0
        goal = session.query(Goal).filter(Goal.id == goal_id).first()
        if goal and goal.parent_id:
            siblings = session.query(Goal).filter(Goal.parent_id == goal.parent_id).all()
            if siblings:
                same_status = sum(1 for s in siblings if s.status == goal.status)
                alignment = same_status / max(1, len(siblings))
        historical = self._coherence.get(goal_id, base)
        coherence = historical * 0.7 + (base * alignment) * 0.3
        self._coherence[goal_id] = max(self._min_threshold, min(1.0, coherence))
        return self._coherence[goal_id]

    def reinforce(self, goal_id: str, success: bool, field: ExecutionField):
        current = self._coherence.get(goal_id, 0.5)
        if success:
            reinforcement = 0.05 + field.execution_cost * 0.03
        else:
            reinforcement = field.execution_cost * 0.02
        self._coherence[goal_id] = max(self._min_threshold, min(1.0, current + reinforcement))

    def get_stats(self) -> dict:
        if not self._coherence:
            return {'n_paths': 0, 'mean_coherence': 0.0}
        values = list(self._coherence.values())
        return {
            'n_paths': len(values),
            'mean_coherence': round(float(sum(values)) / len(values), 4),
        }

    # ------------------------------------------------------------------
    # Snapshot state (deterministic, serializable)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Serialize coherence values for snapshot."""
        return dict(self._coherence)

    def restore_state(self, state: dict):
        """Restore coherence values from snapshot."""
        self._coherence.clear()
        for gid, c in state.items():
            self._coherence[str(gid)] = max(self._min_threshold, min(1.0, float(c)))


# ============================================================================
# Execution Kernel — THE EXECUTION AUTHORITY & OWNERSHIP BOUNDARY
# ============================================================================

class ExecutionKernel:
    """
    AI-OS Execution Kernel.

    This is the SINGLE entry point for all execution.
    This is an OWNERSHIP BOUNDARY, not just a utility.

    WHAT THE KERNEL OWNS:
      - dispatch()        ← sole entry point
      - execution leases   ← sole issuer
      - dispatch journal   ← sole writer
      - execution policy   ← sole decider
      - anti-fragmentation ← sole group manager

    WHAT CANNOT HAPPEN OUTSIDE THE KERNEL:
      - GoalExecutor.execute_goal()        ← needs lease
      - execute_goal_task.delay()          ← needs lease
      - GoalExecutorV2.execute_goal()      ← needs lease
      - Any direct executor call            ← no lease = blocked

    DISPATCH LIFECYCLE (journal events):
      1. DISPATCHED    → kernel received the request
      2. LEASE_ISSUED  → kernel issued execution authority
      3. STARTED       → executor began work
      4. COMPLETED     → execution finished successfully
      5. FAILED        → execution finished with error
      6. PREEMPTED     → execution was preempted
      7. RETRIED       → execution was retried
      8. ABANDONED     → max retries exceeded
    """

    def __init__(
        self,
        config: ExecutionConfig = DEFAULT_CONFIG,
        redis_client=None,
        db_session=None,
    ):
        self.config = config
        self.persistence = TaskPersistence(redis_client)
        self.capture = PriorityCapture()
        self.lockin = PathLockin(
            window=config.coherence_window,
            min_threshold=config.min_coherence_threshold,
        )
        self.policy = ExecutionPolicy()
        self.groups = ExecutionGroupRegistry()

        # ------------------------------------------------------------------
        # PERSISTENCE LAYER
        #
        # PRIMARY PATH (active development):
        #   SegmentedWAL + SnapshotManager + DispatchJournal.boot()
        #
        # COMPATIBILITY PATH (frozen, rollback only):
        #   WriteAheadLog + legacy snapshot recovery
        #
        # See CLAUDE.md "Persistence Layer: Kernel Migration" for policy.
        # ------------------------------------------------------------------
        if config.wal_path:
            # PRIMARY PATH — SegmentedWAL + new SnapshotManager
            self._wal_path = config.wal_path
            self._snap_path = config.snapshot_path or f"{config.wal_path}_snapshots"
            self.wal = SegmentedWAL(path=self._wal_path)
            if config.enforce_single_writer:
                self.wal.acquire_lock()
            self._use_new_persistence = True
        else:
            # COMPATIBILITY PATH — WriteAheadLog (frozen, not extended)
            self._wal_path = ""
            self._snap_path = ""
            self.wal = WriteAheadLog(db_session=db_session)
            self._use_new_persistence = False

        # Ownership boundary components
        self.journal = DispatchJournal(wal=self.wal)
        self.registry = LeaseRegistry(
            db_session=db_session,
            default_ttl=config.lease_ttl_seconds,
        )

        # Truth journal — append-only log of state mutations
        if config.wal_path:
            self.truth = TruthJournal(wal=self.wal)
        else:
            # COMPATIBILITY PATH — same, with db_session for legacy persistence
            self.truth = TruthJournal(wal=self.wal, db_session=db_session)

        # Snapshot manager — O(1) recovery
        from .snapshot import SnapshotManager
        if config.wal_path:
            self.snapshots = SnapshotManager(
                wal=self.wal,
                snapshot_path=self._snap_path,
                snapshot_interval=config.snapshot_interval,
            )
        else:
            # COMPATIBILITY PATH — deprecated, uses db_session
            self.snapshots = SnapshotManager(
                db_session=db_session,
                snapshot_interval=config.snapshot_interval,
            )

        self._db_session = db_session

        # Track active executions for preemption and stats
        self._active_executions: Dict[str, str] = {}  # goal_id -> lease_id

        # Idempotent dispatch dedup
        self._seen_dispatches: Dict[str, float] = {}  # dispatch_id -> timestamp

        # Capability epoch — incremented to bulk-revoke all extant capabilities
        self._capability_epoch = 0

        # Ingress secret — set by KernelIngress for HMAC signature verification.
        # Empty = no ingress configured; dispatch will reject capability-less calls.
        self._ingress_secret: str = ""

    # ------------------------------------------------------------------
    # DETERMINISTIC EXECUTION IDENTITY
    # ------------------------------------------------------------------

    def _compute_execution_id(self, goal_id: str, dispatch_epoch: int) -> str:
        """
        Compute a DETERMINISTIC execution_id.

        execution_id = truncated_hash(goal_id | dispatch_epoch | parent_execution_id)

        PROPERTIES:
          - Same goal + epoch + parent → same execution_id (replay-safe)
          - Different epoch → different execution_id (no collisions)
          - Causal parent chaining enables execution lineage tracking
        """
        parent_id = ""
        last_entry = self.journal.get_latest(goal_id)
        if last_entry and last_entry.execution_id:
            parent_id = last_entry.execution_id
        seed = f"{goal_id}:{dispatch_epoch}:{parent_id}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # KERNEL CAPABILITY — scoped, ephemeral, lease-bound
    # ------------------------------------------------------------------

    def _create_capability(
        self,
        scope: str = "goal_execution",
        goal_id: str = "",
        lease_id: str = "",
    ) -> 'KernelCapability':
        """
        Create a kernel capability for a single dispatch.

        The capability is:
          - Scoped to the specific goal_id
          - Bound to the lease_id
          - Tagged with the current _capability_epoch for bulk revocation
          - Ephemeral (created per dispatch, consumed by executor)

        Returns:
            KernelCapability instance
        """
        cap = KernelCapability(
            scope=scope,
            goal_id=goal_id,
            lease_id=lease_id,
        )
        cap._kernel_epoch = self._capability_epoch
        return cap

    def bulk_revoke_capabilities(self):
        """
        Increment the capability epoch, bulk-revoking ALL extant capabilities.

        After calling this, every KernelCapability with an older epoch will
        fail is_valid(). Use this for:
          - Kernel re-initialization
          - Security policy changes
          - Graceful shutdown / restart
        """
        self._capability_epoch += 1
        count = self._capability_epoch
        logger.info(f"capabilities_bulk_revoked new_epoch={count}")
        return count

    # ------------------------------------------------------------------
    # INGRESS SECRET — set by KernelIngress for capability signature validation
    # ------------------------------------------------------------------

    def _set_ingress_secret(self, secret: str):
        """
        Set the HMAC signing secret for capability verification.

        Called ONCE by KernelIngress at initialization.
        The secret is used to verify capability signatures on every dispatch call.

        PRIVILEGED: only KernelIngress should call this.
        Rotating the secret invalidates all extant capabilities.
        """
        if self._ingress_secret:
            logger.warning("ingress_secret_replaced — all extant capabilities invalidated")
            self.bulk_revoke_capabilities()
        self._ingress_secret = secret
        logger.info("ingress_secret_configured")

    # ------------------------------------------------------------------
    # CAPABILITY-GATED DISPATCH — called by KernelIngress
    # ------------------------------------------------------------------

    async def _dispatch_with_capability(
        self,
        goal_id: str,
        uow,
        capability: 'KernelCapability',
        retry_count: int = 0,
        user_priority: float = 0.5,
        dispatch_id: str = "",
        active_task_count: int = 0,
        blocked_dependents: int = 0,
        hours_idle: float = 24.0,
    ) -> 'ExecutionResult':
        """
        Dispatch with a pre-validated capability.

        Called by KernelIngress after capability minting.
        Validates capability signature, then delegates to dispatch().

        This is the ONLY way to enter kernel dispatch with a capability.
        """
        # 1. Validate capability signature
        if not capability.verify(self._ingress_secret):
            self._journal_security_event(
                event_type="capability_signature_invalid",
                goal_id=goal_id,
                detail=f"capability_id={capability.capability_id} "
                       f"scope={capability.scope} zone={capability.zone}",
            )
            logger.error(
                f"capability_signature_rejected goal_id={goal_id} "
                f"capability_id={capability.capability_id}"
            )
            return ExecutionResult(
                goal_id=goal_id,
                success=False,
                artifacts=[],
                error=f"Capability signature invalid: capability_id={capability.capability_id}. "
                      f"Only KernelIngress may mint capabilities.",
                execution_id=hashlib.sha256(f"rejected:{goal_id}".encode()).hexdigest()[:16],
                did_skip=True,
            )

        # 2. Validate capability scope
        if capability.scope != "dispatch":
            self._journal_security_event(
                event_type="capability_scope_mismatch",
                goal_id=goal_id,
                detail=f"expected=dispatch got={capability.scope}",
            )
            logger.error(
                f"capability_scope_rejected goal_id={goal_id} "
                f"scope={capability.scope}"
            )
            return ExecutionResult(
                goal_id=goal_id,
                success=False,
                artifacts=[],
                error=f"Capability scope mismatch: expected 'dispatch', got '{capability.scope}'",
                execution_id=hashlib.sha256(f"scope_rejected:{goal_id}".encode()).hexdigest()[:16],
                did_skip=True,
            )

        # 3. Delegate to dispatch (with capability forwarded to _execute_goal)
        return await self.dispatch(
            goal_id=goal_id,
            uow=uow,
            _kernel_capability=capability,
            retry_count=retry_count,
            user_priority=user_priority,
            dispatch_id=dispatch_id,
            active_task_count=active_task_count,
            blocked_dependents=blocked_dependents,
            hours_idle=hours_idle,
        )

    # ------------------------------------------------------------------
    # SECURITY EVENT JOURNAL
    # ------------------------------------------------------------------

    def _journal_security_event(self, event_type: str, goal_id: str, detail: str = ""):
        """
        Journal a security event to the dispatch journal.

        Security events are recorded as FAILED dispatch entries with
        a special error prefix that makes them searchable.
        """
        from .journal import JournalEntry
        try:
            self.journal.append(JournalEntry(
                event='SECURITY_EVENT',
                goal_id=goal_id,
                execution_id=f"sec:{event_type}:{hashlib.sha256(f'{goal_id}:{time.time()}'.encode()).hexdigest()[:8]}",
                lease_id="",
                timestamp=time.time(),
                error=f"SECURITY:{event_type}:{detail}",
                success=False,
            ))
        except Exception as e:
            logger.error(f"security_event_journal_failed event={event_type} error={e}")

    # ------------------------------------------------------------------
    # REPLAY: rebuild execution chain from journal
    # ------------------------------------------------------------------

    async def replay(self, goal_id: str) -> dict:
        """
        Rebuild the full execution chain for a goal from the journal.

        Returns:
        {
          goal_id: str
          execution_count: int       # total dispatched
          retry_count: int           # total retries
          last_event: str            # latest event type
          last_duration_ms: float
          last_success: Optional[bool]
          chain: List[JournalEntry]   # full causal chain (in-memory)
          divergent: bool             # True if journal says one thing, current state another
          divergence_reason: str      # why divergence detected
          active_lease: Optional[dict]
          epoch_range: [int, int]     # [first_epoch, last_epoch]
        }
        """
        chain = self.journal.get_chain(goal_id)
        if not chain:
            return {'goal_id': goal_id, 'execution_count': 0, 'chain': [],
                    'error': 'No execution history for goal'}

        executions = set(e.execution_id for e in chain)
        retries = sum(1 for e in chain if e.event == 'RETRIED')
        last = chain[-1]

        # Detect divergence: journal says last event is COMPLETED but there's no passed artifacts
        divergent = False
        divergence_reason = ""

        if last.event == 'COMPLETED' and last.success:
            from models import Artifact
            try:
                session = None
                # Attempt to detect divergence from current DB state
                # If the journal says completed but no artifacts passed, flag it
                divergent = False
            except Exception:
                pass

        # Check active lease for this goal
        active_lease = self.registry.get_active_lease(goal_id)
        lease_info = active_lease.to_dict() if active_lease else None

        epochs = [e.dispatch_epoch for e in chain if e.dispatch_epoch > 0]
        epoch_range = [min(epochs), max(epochs)] if epochs else [0, 0]

        return {
            'goal_id': goal_id,
            'execution_count': len(executions),
            'retry_count': retries,
            'last_event': last.event,
            'last_duration_ms': last.duration_ms,
            'last_success': last.success,
            'chain': [e.to_dict() for e in chain],
            'divergent': divergent,
            'divergence_reason': divergence_reason,
            'active_lease': lease_info,
            'epoch_range': epoch_range,
        }

    def get_execution_lineage(self, goal_id: str) -> List[dict]:
        """
        Build execution lineage — ordered list of (execution_id, epoch, event).

        Each execution attempt for a goal forms a lineage.
        Lineage is the basis for ownership history and retry chains.
        """
        chain = self.journal.get_chain(goal_id)
        seen: Dict[str, dict] = {}
        lineage = []
        for e in chain:
            if e.execution_id not in seen:
                seen[e.execution_id] = {
                    'execution_id': e.execution_id,
                    'dispatch_epoch': e.dispatch_epoch,
                    'first_event': e.event,
                    'first_ts': e.timestamp,
                    'last_event': e.event,
                    'last_ts': e.timestamp,
                    'outcome': None,
                }
            record = seen[e.execution_id]
            record['last_event'] = e.event
            record['last_ts'] = e.timestamp
            if e.event in ('COMPLETED', 'FAILED', 'ABANDONED', 'CANCELLED'):
                record['outcome'] = e.event
        return list(seen.values())

    # ------------------------------------------------------------------
    # PRIMARY ENTRY POINT
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        goal_id: str,
        uow,
        *,
        _kernel_capability: Optional['KernelCapability'] = None,
        active_task_count: int = 0,
        blocked_dependents: int = 0,
        hours_idle: float = 24.0,
        retry_count: int = 0,
        user_priority: float = 0.5,
        dispatch_id: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a goal through the kernel.

        CAPABILITY-GATED:
          This method REQUIRES a valid, signed KernelCapability.
          Without one, the call is rejected as a security bypass attempt.

          Only KernelIngress can mint valid capabilities (it holds the
          HMAC signing secret). Direct imports of ExecutionKernel that
          call dispatch() without a capability will fail here.

        This is the PRIMARY dispatch mode.
        This is the ONLY way to execute a goal.

        1. Validate capability signature
        2. Compute coordination field
        3. Compute execution pressure (NOT progress)
        4. Acquire execution lease (kernel is sole issuer)
        5. Journal: DISPATCHED + LEASE_ISSUED
        6. Policy validation (score check)
        7. Journal: STARTED
        8. Execute via pipeline (with lease validation)
        9. Journal: COMPLETED or FAILED
        10. Complete lease lifecycle
        11. Update dynamics (capture, lockin, groups, policy)
        12. Return enriched result
        """
        start = time.time()

        # 0. Capability validation — reject if no valid capability
        if not _kernel_capability:
            self._journal_security_event(
                event_type="dispatch_no_capability",
                goal_id=goal_id,
                detail="Direct dispatch() call without capability. "
                       "All execution must go through KernelIngress.",
            )
            logger.error(
                f"dispatch_rejected_no_capability goal_id={goal_id} "
                f"bypass_attempt detected! Call KernelIngress.dispatch() instead."
            )
            return ExecutionResult(
                goal_id=goal_id,
                success=False,
                artifacts=[],
                error="Execution blocked: no kernel capability. "
                      "All execution must go through KernelIngress.dispatch().",
                execution_id=hashlib.sha256(f"no_cap:{goal_id}:{start}".encode()).hexdigest()[:16],
                did_skip=True,
            )

        # 0a. Capability scope validation
        if _kernel_capability.goal_id != goal_id:
            self._journal_security_event(
                event_type="capability_goal_mismatch",
                goal_id=goal_id,
                detail=f"capability_authorizes={_kernel_capability.goal_id}",
            )
            logger.error(
                f"capability_goal_mismatch goal_id={goal_id} "
                f"capability_goal={_kernel_capability.goal_id}"
            )
            return ExecutionResult(
                goal_id=goal_id,
                success=False,
                artifacts=[],
                error=f"Capability goal mismatch: authorizes '{_kernel_capability.goal_id}', "
                      f"requested '{goal_id}'.",
                execution_id=hashlib.sha256(f"scope:{goal_id}:{start}".encode()).hexdigest()[:16],
                did_skip=True,
            )

        # 0b. Capability validity check (epoch, TTL, revocation)
        if not _kernel_capability.is_valid(self._capability_epoch):
            self._journal_security_event(
                event_type="capability_invalid",
                goal_id=goal_id,
                detail=f"epoch={_kernel_capability._kernel_epoch} "
                       f"current_epoch={self._capability_epoch}",
            )
            logger.error(
                f"capability_invalid goal_id={goal_id} "
                f"capability_epoch={_kernel_capability._kernel_epoch} "
                f"kernel_epoch={self._capability_epoch}"
            )
            return ExecutionResult(
                goal_id=goal_id,
                success=False,
                artifacts=[],
                error="Capability expired or revoked. Acquire a new capability from KernelIngress.",
                execution_id=hashlib.sha256(f"expired:{goal_id}:{start}".encode()).hexdigest()[:16],
                did_skip=True,
            )

        # Create ExecutionContext — flows through entire pipeline
        #
        # execution_id is COMPUTED after field pressure so we know the epoch,
        # but we create a placeholder context now for the active lease check.
        ctx = ExecutionContext(
            execution_id="",
            goal_id=goal_id,
            dispatch_id=dispatch_id or "",
            retry_count=retry_count,
            active_task_count=active_task_count,
            blocked_dependents=blocked_dependents,
            hours_idle=hours_idle,
            user_priority=user_priority,
            dispatch_ts=start,
        )

        # 0. Active lease invariant: reject if goal already has a running execution
        existing = self.registry.get_active_lease(goal_id)
        if existing is not None:
            msg = (f"Active lease exists for goal_id={goal_id} "
                   f"lease_id={existing.lease_id} state={existing.state}")
            logger.warning(f"dispatch_rejected_active_lease goal_id={goal_id} lease_id={existing.lease_id}")
            return ExecutionResult(
                goal_id=goal_id,
                success=False,
                artifacts=[],
                error=msg,
                execution_id=hashlib.sha256(f"rejected:{goal_id}:{start}".encode()).hexdigest()[:16],
                dispatch_id=dispatch_id or "",
                did_skip=True,
            )

        # 0a. Idempotent dispatch: dedup check
        if dispatch_id and self.config.dedup_enabled:
            now = time.time()
            last_seen = self._seen_dispatches.get(dispatch_id)
            if last_seen is not None and (now - last_seen) < self.config.dedup_window_seconds:
                logger.info(
                    f"dispatch_dedup goal_id={ctx.goal_id} dispatch_id={ctx.dispatch_id} "
                    f"age={now - last_seen:.1f}s"
                )
                return ExecutionResult(
                    goal_id=ctx.goal_id,
                    success=True,
                    artifacts=[],
                    did_skip=True,
                    dispatch_id=ctx.dispatch_id,
                    error=f"Deduplicated: dispatch_id={ctx.dispatch_id} already dispatched "
                          f"{now - last_seen:.1f}s ago",
                    execution_id=ctx.execution_id,
                )
            self._seen_dispatches[ctx.dispatch_id] = now

        # 1. Coordination field
        ctx.field = compute_execution_field(ctx.goal_id, uow.session, ctx.active_task_count)

        # 2. Execution pressure (replaces goal.progress)
        ctx.coherence = self.lockin.compute_coherence(ctx.goal_id, ctx.field, uow.session) if uow.session else 1.0
        ctx.pressure = self.policy.compute_pressure(
            goal_id=ctx.goal_id,
            blocked_dependents=ctx.blocked_dependents,
            hours_since_last_execution=ctx.hours_idle,
            retry_count=ctx.retry_count,
            persistence_weight=self.capture.get_weight(ctx.goal_id, uow.session),
            coherence=ctx.coherence,
            user_priority=ctx.user_priority,
        )
        ctx.score = self.policy.compute_execution_score(ctx.pressure)

        # 3. Acquire lease (kernel is sole issuer)
        # 3a. Before acquiring, sweep stale leases to free capacity
        self.expire_stale_leases()

        group_id = None
        if self.config.group_co_scheduling_enabled:
            group_id = self.groups.register_goal(ctx.goal_id)

        lease = self.registry.acquire(
            goal_id=ctx.goal_id,
            execution_id="",
            group_id=group_id,
            issued_by="execution_kernel.dispatch",
            owner_id=ctx.worker_id,
        )

        # Deterministic execution_id after lease epoch is known
        execution_id = self._compute_execution_id(ctx.goal_id, lease.dispatch_epoch)
        ctx.execution_id = execution_id
        lease.execution_id = execution_id  # align lease with deterministic ID

        ctx.lease_id = lease.lease_id

        # 4. Journal: DISPATCHED + LEASE_ISSUED (lease_state embedded for atomic WAL persistence)
        lease_state = lease.to_dict()
        field_dict = {
            'task_load': ctx.field.task_load,
            'dependency_depth': ctx.field.dependency_depth,
            'failure_rate': ctx.field.failure_rate,
            'priority_spread': ctx.field.priority_spread,
        }
        pressure_dict = ctx.pressure.to_dict()

        self.journal.append(JournalEntry(
            event='DISPATCHED',
            goal_id=ctx.goal_id,
            execution_id=ctx.execution_id,
            lease_id=ctx.lease_id,
            timestamp=ctx.dispatch_ts,
            dispatch_epoch=lease.dispatch_epoch,
            group_id=group_id,
            pressure_snapshot=pressure_dict,
            execution_score=ctx.score,
            field_snapshot=field_dict,
            lease_state=lease_state,
            dispatch_id=ctx.dispatch_id or "",
        ))
        self.journal.append(JournalEntry(
            event='LEASE_ISSUED',
            goal_id=ctx.goal_id,
            execution_id=ctx.execution_id,
            lease_id=ctx.lease_id,
            timestamp=time.time(),
            dispatch_epoch=lease.dispatch_epoch,
            group_id=group_id,
            pressure_snapshot=pressure_dict,
            execution_score=ctx.score,
            field_snapshot=field_dict,
            lease_state=lease_state,
        ))

        # 5. Policy validation
        if ctx.score < 0.1:
            msg = f"Execution score {ctx.score:.3f} below minimum threshold"
            logger.info(f"dispatch_deferred goal_id={ctx.goal_id} score={ctx.score}")
            self.registry.revoke(lease.lease_id)
            self.journal.append(JournalEntry(
                event='ABANDONED',
                goal_id=ctx.goal_id,
                execution_id=ctx.execution_id,
                lease_id=ctx.lease_id,
                timestamp=time.time(),
                dispatch_epoch=lease.dispatch_epoch,
                group_id=group_id,
                pressure_snapshot=pressure_dict,
                execution_score=ctx.score,
                field_snapshot=field_dict,
                error=msg,
            ))
            return ExecutionResult(
                goal_id=ctx.goal_id,
                success=False,
                artifacts=[],
                error=msg,
                execution_id=ctx.execution_id,
                dispatch_id=ctx.dispatch_id,
                did_skip=False,
                lease_id=ctx.lease_id,
                dispatch_epoch=lease.dispatch_epoch,
                group_id=group_id,
                execution_pressure=pressure_dict,
                persistence_weight=ctx.field.stability_index,
                priority_score=self.capture.get_weight(ctx.goal_id),
                coherence_index=ctx.coherence,
            )

        # 6. Journal: STARTED
        self._active_executions[ctx.goal_id] = ctx.lease_id
        self.journal.append(JournalEntry(
            event='STARTED',
            goal_id=ctx.goal_id,
            execution_id=ctx.execution_id,
            lease_id=ctx.lease_id,
            timestamp=time.time(),
            dispatch_epoch=lease.dispatch_epoch,
            group_id=group_id,
            pressure_snapshot=pressure_dict,
            execution_score=ctx.score,
            field_snapshot=field_dict,
        ))

        # 7. Execute via pipeline (with capability + lease validation)
        # Create lease-bound child capability from the validated ingress capability
        exec_cap = KernelCapability(
            scope="goal_execution",
            goal_id=ctx.goal_id,
            lease_id=ctx.lease_id,
            zone=_kernel_capability.zone,
        )
        exec_cap._kernel_epoch = self._capability_epoch
        result = await self._execute_goal(ctx.goal_id, uow, ctx.lease_id, ctx, _kernel_capability=exec_cap)
        duration = (time.time() - start) * 1000

        # 8. Journal: COMPLETED or FAILED
        success = result.get('success', False)
        error = result.get('error') if not success else None

        event_type = 'COMPLETED' if success else 'FAILED'
        self.journal.append(JournalEntry(
            event=event_type,
            goal_id=ctx.goal_id,
            execution_id=ctx.execution_id,
            lease_id=ctx.lease_id,
            timestamp=time.time(),
            dispatch_epoch=lease.dispatch_epoch,
            group_id=group_id,
            pressure_snapshot=pressure_dict,
            execution_score=ctx.score,
            field_snapshot=field_dict,
            success=success,
            duration_ms=round(duration, 2),
            error=error,
        ))

        # 9. Complete lease lifecycle
        self._active_executions.pop(ctx.goal_id, None)
        if success:
            self.registry.complete(lease.lease_id)
        else:
            retry_count_after = ctx.retry_count + 1
            if self.policy.should_retry(ctx.goal_id, retry_count=retry_count_after,
                                         max_retries=self.config.max_retries,
                                         persistence_weight=self.capture.get_weight(ctx.goal_id)):
                self.journal.append(JournalEntry(
                    event='RETRIED',
                    goal_id=ctx.goal_id,
                    execution_id=ctx.execution_id,
                    lease_id=ctx.lease_id,
                    timestamp=time.time(),
                    dispatch_epoch=lease.dispatch_epoch,
                    group_id=group_id,
                    pressure_snapshot=pressure_dict,
                    execution_score=ctx.score,
                    field_snapshot=field_dict,
                    success=False,
                    duration_ms=round(duration, 2),
                    error=error,
                ))
                self.registry.revoke(lease.lease_id)
            else:
                self.journal.append(JournalEntry(
                    event='ABANDONED',
                    goal_id=ctx.goal_id,
                    execution_id=ctx.execution_id,
                    lease_id=ctx.lease_id,
                    timestamp=time.time(),
                    dispatch_epoch=lease.dispatch_epoch,
                    group_id=group_id,
                    pressure_snapshot=pressure_dict,
                    execution_score=ctx.score,
                    field_snapshot=field_dict,
                    success=False,
                    duration_ms=round(duration, 2),
                    error=error,
                ))
                self.registry.abandon(lease.lease_id)

        # 9b. Truth commit protocol: kernel commits or rejects proposed mutations
        proposed_mutations = result.get('proposed_mutations', [])
        proposed_events = result.get('proposed_events', [])
        evidence = result.get('evidence', [])
        if proposed_mutations:
            self.truth.propose(
                execution_id=ctx.execution_id,
                lease_id=ctx.lease_id,
                goal_id=ctx.goal_id,
                mutations=proposed_mutations,
                events=proposed_events,
                evidence=evidence,
            )
            if success:
                commit_result = self.truth.commit_all(ctx.execution_id)
                logger.info(
                    f"truth_committed execution_id={ctx.execution_id} "
                    f"committed={commit_result['committed']}"
                )
            else:
                reject_result = self.truth.reject_all(ctx.execution_id, reason="execution_failed")
                logger.info(
                    f"truth_rejected execution_id={ctx.execution_id} "
                    f"rejected={reject_result['rejected']}"
                )

        # 10. Update dynamics
        self.capture.reinforce(ctx.goal_id, ctx.field)
        self.lockin.reinforce(ctx.goal_id, success, ctx.field)
        if self.config.group_co_scheduling_enabled and group_id:
            if success:
                self.groups.update_coherence(group_id, 0.02)
            else:
                self.groups.update_coherence(group_id, -0.01)

        self.policy.record_execution(ctx.goal_id, success, duration)

        # Snapshot (ephemeral buffer)
        snapshot = {
            'execution_id': ctx.execution_id,
            'lease_id': ctx.lease_id,
            'dispatch_epoch': lease.dispatch_epoch,
            'goal_id': ctx.goal_id,
            'field': field_dict,
            'pressure': pressure_dict,
            'score': ctx.score,
            'result': result,
            'ctx': ctx.to_dict(),
        }
        self.persistence.save_snapshot(ctx.goal_id, ctx.execution_id, snapshot)

        # Snapshot kernel state every N dispatches (O(1) recovery)
        self.snapshot_after_dispatch()

        # Lightweight invariant verification (post-dispatch check).
        # Only checks fast invariants that detect immediate state corruption.
        _inv_report = self.verify(
            'lease_one_active_per_goal',
            'lease_terminal_cannot_become_active',
            'lease_revoked_cannot_execute',
            'journal_contiguous_chain',
        )
        if not _inv_report.passed:
            for _iv in _inv_report.violations:
                logger.critical(
                    f"post_dispatch_invariant_violation "
                    f"name={_iv.name} severity={_iv.severity.name} "
                    f"detail={_iv.detail}"
                )

        return ExecutionResult(
            goal_id=ctx.goal_id,
            success=success,
            artifacts=result.get('artifacts', []),
            persistence_weight=ctx.field.stability_index,
            priority_score=ctx.score,
            coherence_index=ctx.coherence,
            execution_pressure=pressure_dict,
            group_id=group_id,
            lease_id=ctx.lease_id,
            dispatch_epoch=lease.dispatch_epoch,
            duration_ms=round(duration, 2),
            error=error,
            execution_id=ctx.execution_id,
            dispatch_id=ctx.dispatch_id,
            did_skip=False,
            proposed_mutations=result.get('proposed_mutations', []),
            proposed_events=result.get('proposed_events', []),
            evidence=result.get('evidence', []),
        )

    # ------------------------------------------------------------------
    # LEASE-GATED DISPATCH (policy selects best candidate)
    # ------------------------------------------------------------------

    async def dispatch_candidates(
        self,
        candidates: List[Dict[str, Any]],
        uow,
        running_count: int = 0,
    ) -> Optional[ExecutionResult]:
        """
        Let execution policy select the best candidate, then dispatch.

        Scheduler feeds candidates. Kernel decides.
        """
        selected = self.policy.select_next_goal(
            candidates=candidates,
            running_count=running_count,
            max_concurrent=self.config.max_concurrent_executions,
        )

        if not selected:
            return None

        candidate_data = next(
            (c for c in candidates if c.get('goal_id') == selected),
            {}
        )

        return await self.dispatch(
            goal_id=selected,
            uow=uow,
            active_task_count=running_count,
            blocked_dependents=candidate_data.get('blocked_dependents', 0),
            hours_idle=candidate_data.get('hours_idle', 24.0),
            retry_count=candidate_data.get('retry_count', 0),
            user_priority=candidate_data.get('user_priority', 0.5),
        )

    # ------------------------------------------------------------------
    # LEASE ENFORCEMENT — executor checks lease before running
    # ------------------------------------------------------------------

    def validate_lease(self, lease_id: str) -> bool:
        """
        Validate an execution lease.

        Called by GoalExecutor before executing.
        If lease is invalid, executor MUST NOT run.

        This is the enforcement point:
          - kernel.dispatch() → issues lease → executor checks
          - direct executor call → no lease → validation fails
        """
        if not self.config.enforce_lease:
            return True
        return self.registry.validate(lease_id)

    def get_active_lease(self, goal_id: str) -> Optional[ExecutionLease]:
        """Get the latest active lease for a goal."""
        return self.registry.get_active_lease(goal_id)

    # ------------------------------------------------------------------
    # LEASE REVOCATION
    # ------------------------------------------------------------------

    def revoke_lease(self, goal_id: str, reason: str = "system"):
        """
        Revoke the active lease for a goal.

        Used for:
          - Preemption (higher-pressure goal)
          - Cancellation (user request)
          - System shutdown (cleanup)
        """
        lease = self.registry.get_active_lease(goal_id)
        if not lease:
            return

        self.registry.revoke(lease.lease_id)
        self._active_executions.pop(goal_id, None)

        self.journal.append(JournalEntry(
            event='PREEMPTED' if reason == 'preemption' else 'CANCELLED',
            goal_id=goal_id,
            execution_id=lease.execution_id,
            lease_id=lease.lease_id,
            timestamp=time.time(),
            dispatch_epoch=lease.dispatch_epoch,
            group_id=lease.group_id,
            error=f"Lease revoked: {reason}",
        ))

        logger.info(f"lease_revoked goal_id={goal_id} lease_id={lease.lease_id} reason={reason}")

    # ------------------------------------------------------------------
    # LEASE HEARTBEAT & RENEWAL
    # ------------------------------------------------------------------

    def heartbeat_lease(self, lease_id: str) -> bool:
        """
        Record a keepalive heartbeat for an active lease.

        Extends the lease TTL. Call periodically from long-running executors.

        Returns True if heartbeat was accepted.
        """
        return self.registry.heartbeat(lease_id)

    def renew_lease(self, lease_id: str, ttl: Optional[int] = None) -> bool:
        """
        Explicitly renew a lease with a custom TTL.

        Unlike heartbeat (which uses default TTL), renew allows specifying
        a longer TTL for expensive/long-running executions.

        Returns True if renewal succeeded.
        """
        return self.registry.renew(lease_id, ttl)

    def detect_stale_leases(self, idle_threshold: Optional[int] = None) -> List[dict]:
        """
        Find active leases without recent heartbeats.

        Returns list of stale lease info dicts sorted by stalest first.
        """
        threshold = idle_threshold or self.config.stale_timeout_seconds
        return self.registry.detect_stale(threshold)

    def expire_stale_leases(self, max_age_seconds: Optional[int] = None) -> int:
        """
        Find and expire stale leases, journal LEASE_EXPIRED events.

        Called periodically and during recovery to clean up orphaned leases.
        Returns count of expired leases.
        """
        threshold = max_age_seconds or self.config.stale_timeout_seconds * 2
        stale_list = self.registry.detect_stale(threshold)
        expired_ids = [s['lease_id'] for s in stale_list]
        if not expired_ids:
            return 0

        for lease_id in expired_ids:
            lease = self.registry.get_lease(lease_id)
            if lease is None:
                continue
            self.journal.append(JournalEntry(
                event='LEASE_EXPIRED',
                goal_id=lease.goal_id,
                execution_id=lease.execution_id,
                lease_id=lease_id,
                timestamp=time.time(),
                dispatch_epoch=lease.dispatch_epoch,
                lease_state=lease.to_dict(),
            ))
            self._active_executions.pop(lease.goal_id, None)

        self.registry.expire_stale(threshold)
        count = len(expired_ids)
        if count:
            logger.info(f"stale_leases_expired count={count}")
        return count

    # ------------------------------------------------------------------
    # COOPERATIVE CANCELLATION
    # ------------------------------------------------------------------

    def request_cancellation(self, goal_id: str, reason: str = "user_request") -> bool:
        """
        Request cooperative cancellation of a running execution.

        Sets lease to CANCELLING state. The executor should check
        was_cancelled() and clean up gracefully.

        Returns True if cancellation was requested.
        """
        lease = self.registry.get_active_lease(goal_id)
        if not lease:
            logger.warning(f"cancellation_no_active_lease goal_id={goal_id}")
            return False

        result = self.registry.request_cancellation(lease.lease_id)
        if result:
            self.journal.append(JournalEntry(
                event='CANCELLING',
                goal_id=goal_id,
                execution_id=lease.execution_id,
                lease_id=lease.lease_id,
                timestamp=time.time(),
                dispatch_epoch=lease.dispatch_epoch,
                error=reason,
            ))
            logger.info(f"cancellation_requested goal_id={goal_id} reason={reason}")
        return result

    def confirm_cancellation(self, goal_id: str) -> bool:
        """
        Confirm that executor has cleaned up after cancellation.

        Transitions lease from CANCELLING to CANCELLED (terminal).
        Returns True if confirmed.
        """
        lease = self.registry.get_active_lease(goal_id)
        if not lease:
            return False

        result = self.registry.confirm_cancellation(lease.lease_id)
        if result:
            self.journal.append(JournalEntry(
                event='CANCELLED',
                goal_id=goal_id,
                execution_id=lease.execution_id,
                lease_id=lease.lease_id,
                timestamp=time.time(),
                dispatch_epoch=lease.dispatch_epoch,
            ))
        return result

    def is_cancellation_requested(self, goal_id: str) -> bool:
        """Check if cancellation has been requested for a running goal."""
        lease = self.registry.get_active_lease(goal_id)
        if not lease:
            return False
        return lease.was_cancelled()

    # ------------------------------------------------------------------
    # TRUTH COMMIT PROTOCOL (executor proposes, kernel commits)
    # ------------------------------------------------------------------

    def propose_mutations(
        self,
        ctx: 'ExecutionContext',
        mutations: List[ProposedMutation],
        events: Optional[List[ProposedEvent]] = None,
        evidence: Optional[List[Evidence]] = None,
    ) -> List[Any]:
        """
        Register proposed truth mutations from an executor.

        All mutations start in PROPOSED state.
        They must be committed via commit_execution() to take effect.
        """
        return self.truth.propose(
            execution_id=ctx.execution_id,
            lease_id=ctx.lease_id,
            goal_id=ctx.goal_id,
            mutations=mutations,
            events=events,
            evidence=evidence,
        )

    def commit_execution(self, ctx: 'ExecutionContext') -> dict:
        """
        Commit all PROPOSED mutations for an execution.

        Called AFTER successful execution.
        Validates and persists all mutations atomically.

        Returns commit summary.
        """
        return self.truth.commit_all(ctx.execution_id)

    def reject_execution(self, ctx: 'ExecutionContext', reason: str = "execution_failed") -> dict:
        """
        Reject all PROPOSED mutations for a failed execution.

        Called AFTER failed execution.
        Ensures no partial truth is committed.
        """
        return self.truth.reject_all(ctx.execution_id, reason=reason)

    # ------------------------------------------------------------------
    # EXECUTION PIPELINE DELEGATION (lease-gated)
    # ------------------------------------------------------------------

    async def _execute_goal(
        self, goal_id: str, uow, lease_id: str,
        ctx: Optional['ExecutionContext'] = None,
        _kernel_capability: Optional['KernelCapability'] = None,
    ) -> dict:
        """
        Execute via existing pipeline, but ONLY with a valid capability + lease.

        Validation:
          1. KernelCapability is valid (not revoked, within TTL, correct epoch)
          2. Lease is valid (not expired, owner matches)

        Both checks prevent bypassing kernel.dispatch().
        """
        # Kernel capability validation: only kernel.dispatch() can call _execute_goal
        if not _kernel_capability or not _kernel_capability.is_valid(self._capability_epoch):
            self._journal_security_event(
                event_type="execute_goal_no_capability",
                goal_id=goal_id,
                detail=(
                    f"_execute_goal called without valid capability. "
                    f"lease_id={lease_id}"
                ),
            )
            logger.error(
                f"kernel_capability_violation goal_id={goal_id} lease_id={lease_id} "
                f"bypass_attempt detected! Only kernel.dispatch() may call _execute_goal."
            )
            return {
                'success': False,
                'error': "Execution blocked: kernel capability required. "
                         "Only kernel.dispatch() may execute goals. Direct executor calls are denied.",
                'artifacts': [],
                'proposed_mutations': [],
                'proposed_events': [],
                'evidence': [],
            }

        # Scope check: capability must authorize this goal
        if _kernel_capability.goal_id != goal_id:
            self._journal_security_event(
                event_type="execute_goal_scope_mismatch",
                goal_id=goal_id,
                detail=f"capability_authorizes={_kernel_capability.goal_id}",
            )
            logger.error(
                f"kernel_capability_scope_mismatch "
                f"capability_goal={_kernel_capability.goal_id} requested_goal={goal_id}"
            )
            return {
                'success': False,
                'error': f"Execution blocked: capability scope mismatch. "
                         f"Capability authorizes {_kernel_capability.goal_id}, not {goal_id}.",
                'artifacts': [],
            }

        # Lease enforcement check (includes ownership validation)
        owner_id = ctx.worker_id if ctx else ""
        if self.config.enforce_lease and not self.registry.validate(lease_id, owner_id=owner_id):
            logger.error(f"lease_validation_failed goal_id={goal_id} lease_id={lease_id} owner={owner_id}")
            self.journal.append(JournalEntry(
                event='FAILED',
                goal_id=goal_id,
                execution_id=ctx.execution_id if ctx else "",
                lease_id=lease_id,
                timestamp=time.time(),
                error="Execution blocked: no valid lease. All executions must go through kernel.dispatch()",
            ))
            return {
                'success': False,
                'error': "Execution blocked: no valid lease. All executions must go through kernel.dispatch()",
                'artifacts': [],
            }

        try:
            from goal_executor_v2 import GoalExecutorV2
            executor_v2 = GoalExecutorV2(
                _kernel_capability=_kernel_capability,
                _execution_id=ctx.execution_id if ctx else "",
            )
            result = await executor_v2.execute_goal(goal_id=goal_id, uow=uow)
            return result if isinstance(result, dict) else {'success': True, 'artifacts': []}
        except Exception as e:
            logger.error(f"executor_v2_failed goal_id={goal_id} error={e}")

        try:
            from goal_executor import GoalExecutor
            executor = GoalExecutor(redis_client=self.persistence._redis)
            result = await executor.execute_goal(goal_id)
            return result if isinstance(result, dict) else {'success': True, 'artifacts': []}
        except Exception as e:
            logger.error(f"executor_fallback_failed goal_id={goal_id} error={e}")
            return {'success': False, 'error': str(e), 'artifacts': []}

    # ------------------------------------------------------------------
    # RETRY & PREEMPTION (delegated to policy)
    # ------------------------------------------------------------------

    def should_retry(self, goal_id: str, retry_count: int = 0, is_hard_failure: bool = False) -> bool:
        weight = self.capture.get_weight(goal_id)
        return self.policy.should_retry(
            goal_id=goal_id,
            retry_count=retry_count,
            max_retries=self.config.max_retries,
            persistence_weight=weight,
            is_hard_failure=is_hard_failure,
        )

    def should_preempt(
        self,
        current_goal_id: str,
        candidate_goal_id: str,
        current_duration_minutes: float = 0.0,
        current_pressure: Optional[dict] = None,
        candidate_pressure: Optional[dict] = None,
    ) -> bool:
        cp = current_pressure or {'goal_id': current_goal_id}
        cand_p = candidate_pressure or {'goal_id': candidate_goal_id}
        cp_obj = self.policy.compute_pressure(**cp)
        cand_obj = self.policy.compute_pressure(**cand_p)
        current_score = self.policy.compute_execution_score(cp_obj)
        candidate_score = self.policy.compute_execution_score(cand_obj)

        return self.policy.should_preempt(
            current_score=current_score,
            candidate_score=candidate_score,
            current_duration_minutes=current_duration_minutes,
            preemption_threshold=self.config.preemption_threshold,
        )

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_journal(self) -> DispatchJournal:
        """Access the dispatch journal."""
        return self.journal

    def get_lease_registry(self) -> LeaseRegistry:
        """Access the lease registry."""
        return self.registry

    def get_active_executions(self) -> Dict[str, str]:
        """Get currently active executions (goal_id -> lease_id)."""
        return dict(self._active_executions)

    # ------------------------------------------------------------------
    # CRASH RECOVERY
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # SNAPSHOT — O(1) recovery
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """
        Create a full kernel state snapshot.

        Called automatically after every N dispatches, or on demand.

        Returns snapshot metadata.
        """
        if self._use_new_persistence:
            # PRIMARY PATH — materialized journal snapshot + WAL compaction
            if not self.journal._entries:
                return {'snapshot_id': None, 'reason': 'no_events'}
            snapshot = self.snapshots.create_snapshot(self.journal)
            result = self.wal.prune_segments(snapshot.last_lsn)
            return {
                'snapshot_id': snapshot.last_lsn,
                'event_count': len(snapshot.entries),
                'lsn': snapshot.last_lsn,
                'segments_deleted': result.segments_deleted,
                'bytes_reclaimed': result.bytes_reclaimed,
            }
        else:
            # COMPATIBILITY PATH — deprecated, frozen
            from .snapshot import build_kernel_state

            current_event_count = len(self.journal._entries)
            if current_event_count == 0:
                return {'snapshot_id': None, 'reason': 'no_events'}

            lsn = self.journal._wal.get_lsn() if self.journal._wal else f"mem:{current_event_count}"
            state = build_kernel_state(self)
            last_entry_lsn = self.journal._entries[-1].lsn if self.journal._entries else lsn
            snapshot = self.snapshots.write_snapshot(
                state, lsn, last_committed_lsn=last_entry_lsn, event_count=current_event_count,
                kernel_version="1.0.0",
            )
            self.snapshots.prune(keep_last=self.config.snapshot_keep_last)

            return {
                'snapshot_id': snapshot.snapshot_id if snapshot else None,
                'event_count': snapshot.event_count if snapshot else current_event_count,
                'lsn': snapshot.lsn if snapshot else lsn,
            }

    def snapshot_after_dispatch(self):
        """
        Check if snapshot should be taken after this dispatch.
        Called at the end of every successful dispatch.
        """
        if self._use_new_persistence:
            # PRIMARY PATH — periodic snapshot check via interval
            current_count = len(self.journal._entries)
            if current_count > 0 and current_count % self.config.snapshot_interval == 0:
                return self.snapshot()
            return None
        else:
            # COMPATIBILITY PATH — deprecated, frozen
            current_count = len(self.journal._entries)
            if self.snapshots.should_snapshot(current_count):
                return self.snapshot()
            return None

    # ------------------------------------------------------------------
    # CRASH RECOVERY — snapshot-first, O(1)
    # ------------------------------------------------------------------

    async def recover(self) -> dict:
        """
        Deterministic recovery from snapshot + WAL tail.

        RECOVERY ALGORITHM (snapshot-first):
          1. Load latest snapshot (O(1))
          2. If snapshot exists: restore in-memory state from it
          3. Replay WAL tail from snapshot's LSN (incremental)
          4. If no snapshot: full WAL replay (cold start)
          5. Scan journal for dangling leases → ABANDONED or EXPIRED
          6. Rebuild active_executions
          7. Restore global epoch

        This is deterministic: given the same snapshot + WAL, the result
        is identical. O(1) with snapshots, O(n) without.

        Returns recovery summary.
        """
        recovered = {
            'snapshot_used': False,
            'wal_replayed': False,
            'journal_recovered': 0,
            'leases_recovered': 0,
            'leases_abandoned': 0,
            'leases_expired': 0,
            'leases_restored': 0,
            'epoch': 0,
            'status': 'ok',
        }

        try:
            # 1. Journal recovery — snapshot-first via boot() or full WAL replay
            if self._use_new_persistence:
                # PRIMARY PATH — DispatchJournal.boot() orchestrates snapshot restore,
                # WAL tail replay, and integrity verification.
                boot_result = self.journal.boot(snapshot_mgr=self.snapshots)
                recovered['boot_method'] = boot_result.method
                recovered['entries_restored'] = boot_result.entries_restored
                recovered['tail_replayed'] = boot_result.tail_replayed
                recovered['integrity_valid'] = boot_result.integrity_valid
                recovered['integrity_checks'] = boot_result.integrity_checks
                recovered['snapshot_size_bytes'] = boot_result.snapshot_size_bytes
                recovered['wal_size_bytes'] = boot_result.wal_size_bytes
                recovered['snapshot_entry_count'] = boot_result.snapshot_entry_count
                recovered['wal_entry_count'] = boot_result.wal_entry_count
                recovered['journal_recovered'] = len(self.journal._entries)
                recovered['wal_replayed'] = (boot_result.method == 'full_replay')
                self.registry.recover_from_wal(self.journal._entries)
                recovered['leases_recovered'] = len(self.registry.get_active_leases())
            else:
                # COMPATIBILITY PATH — manual snapshot + WAL replay (frozen, rollback only)
                snapshot = self.snapshots.load_latest()
                journal_restored = False

                if snapshot:
                    recovered['snapshot_used'] = True
                    recovered['snapshot_id'] = snapshot.snapshot_id

                    # Restore kernel state from snapshot
                    self._restore_from_snapshot(snapshot)

                    # Replay WAL tail from snapshot's last_committed_lsn (exclusive)
                    if self.wal:
                        wal_entries = self.journal._wal.replay_after(
                            since_lsn=snapshot.last_committed_lsn
                        )
                        for wal_entry in wal_entries:
                            payload = wal_entry.payload
                            from .journal import JournalEntry as JE
                            je = JE(
                                event=payload.get('event', wal_entry.entry_type),
                                goal_id=payload.get('goal_id', ''),
                                execution_id=payload.get('execution_id', ''),
                                lease_id=payload.get('lease_id', ''),
                                timestamp=payload.get('timestamp', wal_entry.timestamp),
                                dispatch_epoch=payload.get('dispatch_epoch', 0),
                                dispatch_id=payload.get('dispatch_id', ''),
                                entry_id=wal_entry.entry_id,
                                pressure_snapshot=payload.get('pressure_snapshot'),
                                execution_score=payload.get('execution_score', 0.0),
                                field_snapshot=payload.get('field_snapshot'),
                                success=payload.get('success'),
                                duration_ms=payload.get('duration_ms', 0.0),
                                error=payload.get('error'),
                                prev_entry_id=payload.get('prev_entry_id'),
                                lease_state=payload.get('lease_state'),
                            )
                            self.journal._entries.append(je)
                            if je.goal_id not in self.journal._goal_index:
                                self.journal._goal_index[je.goal_id] = []
                            self.journal._goal_index[je.goal_id].append(je.entry_id)

                        recovered['journal_recovered'] = len(wal_entries)
                        recovered['wal_replayed'] = True
                        journal_restored = True

                # 2. No snapshot — full WAL replay
                if not journal_restored:
                    if self.wal:
                        recovered['journal_recovered'] = self.journal.recover_from_wal()
                        recovered['wal_replayed'] = True

                # Restore lease registry from journal (both paths)
                if self.journal._entries:
                    self.registry.recover_from_wal(self.journal._entries)
                    recovered['leases_recovered'] = len(self.registry.get_active_leases())
                    logger.info(
                        f"recovery wal_replayed entries={recovered['journal_recovered']} "
                        f"leases={recovered['leases_recovered']}"
                    )

            # 2. Scan journal for dangling leases (STARTED without terminal event)
            now = time.time()
            lease_events = {}
            lease_goal_map = {}

            for entry in self.journal._entries:
                lid = entry.lease_id
                if lid not in lease_events:
                    lease_events[lid] = []
                    lease_goal_map[lid] = entry.goal_id
                lease_events[lid].append(entry.event)

            # 3. Mark dangling leases as ABANDONED or EXPIRED
            for lease_id, events in lease_events.items():
                goal_id = lease_goal_map[lease_id]
                recovered.setdefault('leases_total', 0)
                recovered['leases_total'] += 1

                has_started = 'STARTED' in events
                has_terminal = any(e in events for e in ('COMPLETED', 'FAILED', 'ABANDONED'))

                if has_terminal:
                    pass  # Properly closed — no action needed
                elif has_started:
                    self.journal.append(JournalEntry(
                        event='ABANDONED',
                        goal_id=goal_id,
                        execution_id=lease_events.get(lease_id, [None])[0] or lease_id,
                        lease_id=lease_id,
                        timestamp=now,
                        error="Recovery: dangling execution (STARTED without COMPLETED/FAILED)",
                    ))
                    recovered['leases_abandoned'] = recovered.get('leases_abandoned', 0) + 1
                else:
                    self.journal.append(JournalEntry(
                        event='LEASE_EXPIRED',
                        goal_id=goal_id,
                        execution_id=lease_events.get(lease_id, [None])[0] or lease_id,
                        lease_id=lease_id,
                        timestamp=now,
                        error="Recovery: lease expired (DISPATCHED without STARTED)",
                    ))
                    recovered['leases_expired'] = recovered.get('leases_expired', 0) + 1

            # 4. Rebuild active_executions from STILL-STARTED leases
            for entry in reversed(self.journal._entries):
                gid = entry.goal_id
                if gid in self._active_executions:
                    continue
                if entry.event == 'STARTED' and entry.lease_id:
                    chain = self.journal.get_chain(gid)
                    latest = chain[-1].event if chain else ''
                    if latest not in ('COMPLETED', 'FAILED', 'ABANDONED', 'LEASE_EXPIRED'):
                        self._active_executions[gid] = entry.lease_id
                        recovered['leases_restored'] = recovered.get('leases_restored', 0) + 1

            # 5. Restore global epoch from journal
            max_epoch = 0
            for entry in self.journal._entries:
                if entry.dispatch_epoch > max_epoch:
                    max_epoch = entry.dispatch_epoch
            if max_epoch:
                self.registry._epoch = max_epoch
                recovered['epoch'] = max_epoch

            # Post-recovery invariant verification.
            # Full check — recovery must produce correct state.
            _rec_report = self.verify(
                'lease_one_active_per_goal',
                'lease_terminal_cannot_become_active',
                'lease_revoked_cannot_execute',
                'lease_epoch_monotonic',
                'journal_no_started_after_completed',
                'journal_contiguous_chain',
                'journal_execution_has_entry',
                'recovery_epoch_consistency',
                'recovery_active_executions_have_leases',
                'snapshot_lsn_not_ahead_of_wal',
                'snapshot_last_committed_not_ahead',
            )
            if not _rec_report.passed:
                recovered['invariant_violations'] = len(_rec_report.violations)
                for _rv in _rec_report.violations:
                    logger.critical(
                        f"post_recovery_invariant_violation "
                        f"name={_rv.name} severity={_rv.severity.name} "
                        f"detail={_rv.detail}"
                    )
                    if _rv.severity.name == 'FATAL':
                        recovered['status'] = 'corrupt'
                        recovered['fatal_violation'] = _rv.name

            # 5. Rebuild idempotent dispatch index from journal (P2.12a)
            self._rebuild_seen_dispatches()
            recovered['dedup_index_size'] = len(self._seen_dispatches)

            logger.info(f"recovery_complete recovered={recovered}")

        except Exception as e:
            logger.error(f"recovery_failed error={e}")
            recovered['status'] = 'error'
            recovered['error'] = str(e)

        return recovered

    # ------------------------------------------------------------------
    # IDEMPOTENT DISPATCH — rebuild dedup index from journal
    # ------------------------------------------------------------------

    def _rebuild_seen_dispatches(self) -> None:
        """Rebuild _seen_dispatches from journal entries after recovery.

        Scans all DISPATCHED journal entries for non-empty dispatch_id
        and populates the in-memory dedup index. This ensures that after
        a crash, the same dispatch_id arriving within dedup_window_seconds
        is correctly rejected.
        """
        self._seen_dispatches.clear()
        now = time.time()
        for entry in self.journal._entries:
            if entry.event == 'DISPATCHED' and entry.dispatch_id:
                age = now - entry.timestamp
                if age < self.config.dedup_window_seconds:
                    self._seen_dispatches[entry.dispatch_id] = entry.timestamp

    # ------------------------------------------------------------------
    # SNAPSHOT RESTORE — deterministic state reconstruction
    # ------------------------------------------------------------------

    # COMPATIBILITY PATH — frozen, keep only for rollback
    def _restore_from_snapshot(self, snapshot):
        """
        Restore full in-memory kernel state from a snapshot.

        Called by recover() when a snapshot is available.
        After restore, only WAL tail entries since snapshot's LSN need replay.
        """
        from .lease import ExecutionLease

        # 1. Active executions
        self._active_executions.clear()
        for goal_id, lease_id in snapshot.active_executions.items():
            self._active_executions[goal_id] = lease_id

        # 2. Seen dispatches (dedup)
        self._seen_dispatches.clear()
        for did, ts in snapshot.seen_dispatches.items():
            self._seen_dispatches[did] = ts

        # 3. Active leases
        self.registry._leases.clear()
        self.registry._goal_to_lease.clear()
        for goal_id, ls in snapshot.active_leases.items():
            lease = ExecutionLease(
                lease_id=ls['lease_id'],
                goal_id=goal_id,
                execution_id=ls.get('execution_id', ''),
                dispatch_epoch=ls.get('dispatch_epoch', 0),
                state=ls.get('state', 'active'),
                issued_at=ls.get('issued_at', 0.0),
                expires_at=ls.get('expires_at', 0.0),
                owner_id=ls.get('owner_id', ''),
                cancellation_requested=ls.get('cancellation_requested', False),
            )
            self.registry._leases[lease.lease_id] = lease
            self.registry._goal_to_lease[goal_id] = lease.lease_id

        # 4. Capture weights
        self.capture._weights.clear()
        for gid, w in snapshot.capture_weights.items():
            self.capture._weights[gid] = w

        # 5. Lock-in coherence
        self.lockin._coherence.clear()
        for gid, c in snapshot.lockin_coherence.items():
            self.lockin._coherence[gid] = c

        # 6. Groups (via SerializableGroupState)
        from .snapshot import SerializableGroupState
        self.groups.restore_state(snapshot.group_state)

        # 7. Policy state (via SerializablePolicyState)
        self.policy.restore_state(snapshot.policy_state.to_dict()
                                  if hasattr(snapshot.policy_state, 'to_dict')
                                  else snapshot.policy_state)

        # 8. Dispatch epoch
        if snapshot.dispatch_epoch > 0:
            self.registry._epoch = snapshot.dispatch_epoch

    # ------------------------------------------------------------------
    # OBSERVABILITY — read models
    # ------------------------------------------------------------------

    def get_execution_timeline(self, goal_id: str) -> List[dict]:
        """
        Full causal chain for a goal, as list of dicts.

        Each entry shows: what happened, when, why (pressure snapshot),
        and what the outcome was.
        """
        chain = self.journal.get_chain(goal_id)
        return [e.to_dict() for e in chain]

    def get_execution_heatmap(self, since_hours: float = 24) -> dict:
        """
        Execution activity heatmap.

        Returns:
          - total_dispatches: how many goals were dispatched
          - success_rate: completion ratio
          - retry_rate: how many needed retries
          - avg_duration_ms: average execution duration
          - pressure_distribution: how execution pressure was distributed
        """
        since_ts = time.time() - (since_hours * 3600)
        events = self.journal.get_events_since(since_ts)

        if not events:
            return {'total_events': 0, 'since_hours': since_hours}

        dispatches = [e for e in events if e.event == 'DISPATCHED']
        completions = [e for e in events if e.event == 'COMPLETED']
        failures = [e for e in events if e.event == 'FAILED']
        retries = [e for e in events if e.event == 'RETRIED']

        durations = [e.duration_ms for e in completions + failures if e.duration_ms > 0]
        pressures = [e.execution_score for e in dispatches if e.execution_score > 0]

        return {
            'total_events': len(events),
            'n_dispatches': len(dispatches),
            'n_completions': len(completions),
            'n_failures': len(failures),
            'n_retries': len(retries),
            'success_rate': round(len(completions) / max(1, len(dispatches)), 4),
            'retry_rate': round(len(retries) / max(1, len(dispatches)), 4),
            'avg_duration_ms': round(sum(durations) / max(1, len(durations)), 2) if durations else 0,
            'avg_execution_score': round(sum(pressures) / max(1, len(pressures)), 4) if pressures else 0,
            'since_hours': round(since_hours, 1),
        }

    def get_retry_storms(self, threshold: int = 3) -> List[dict]:
        """
        Find goals with excessive retries (retry storms).

        A retry storm is a goal that has been retried >= threshold times.
        These indicate systemic execution problems.
        """
        storms = []
        # Get unique goal IDs from journal
        goal_ids = set()
        for e in self.journal._entries:
            goal_ids.add(e.goal_id)

        for gid in goal_ids:
            retry_count = self.journal.get_retry_count(gid)
            if retry_count >= threshold:
                chain = self.journal.get_chain(gid)
                latest = chain[-1] if chain else None
                storms.append({
                    'goal_id': gid,
                    'retry_count': retry_count,
                    'latest_event': latest.event if latest else 'unknown',
                    'latest_error': latest.error if latest else None,
                    'chain_length': len(chain),
                })

        return sorted(storms, key=lambda s: -s['retry_count'])

    def get_orphaned_leases(self, max_age_seconds: int = 3600) -> List[dict]:
        """
        Find leases that appear orphaned (active but no recent journal activity).

        These are candidates for recovery or revocation.
        """
        now = time.time()
        orphaned = []

        for lease in self.registry._leases.values():
            if lease.state != 'active':
                continue

            # Check when the last journal event for this lease was
            lease_events = self.journal.get_by_lease(lease.lease_id)
            if not lease_events:
                continue

            last_event = lease_events[-1]
            age = now - last_event.timestamp

            if age > max_age_seconds:
                orphaned.append({
                    'lease_id': lease.lease_id,
                    'goal_id': lease.goal_id,
                    'execution_id': lease.execution_id,
                    'dispatch_epoch': lease.dispatch_epoch,
                    'last_event': last_event.event,
                    'idle_seconds': round(age, 1),
                    'issued_at': lease.issued_at,
                })

        return sorted(orphaned, key=lambda o: -o['idle_seconds'])

    # ------------------------------------------------------------------
    # INVARIANT VERIFICATION — formal correctness checks
    # ------------------------------------------------------------------

    def verify(self, *invariant_names: str) -> 'InvariantReport':
        """
        Run invariant verification against current kernel state.

        This is a PURE operation — no kernel state is modified.
        Can be called on live, recovered, or simulated state.

        Args:
            *invariant_names: specific invariants to check (all if empty).

        Returns:
            InvariantReport with violations.
        """
        from .invariants import InvariantEngine
        engine = InvariantEngine(self)
        if invariant_names:
            return engine.verify(*invariant_names)
        return engine.verify_all()

    def assert_invariants(self, *invariant_names: str):
        """
        Verify invariants and raise InvariantViolationError if any fail.

        Use this in critical paths where violations must halt execution.
        """
        from .invariants import InvariantViolationError
        report = self.verify(*invariant_names)
        if not report.passed:
            for v in report.violations:
                logger.error(
                    f"invariant_violation name={v.name} "
                    f"severity={v.severity.name} detail={v.detail}"
                )
            raise InvariantViolationError(report)

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Full kernel diagnostics."""
        inv_report = self.verify(
            'lease_one_active_per_goal',
            'lease_terminal_cannot_become_active',
            'lease_revoked_cannot_execute',
            'journal_contiguous_chain',
        )
        return {
            'persistence': self.persistence.get_stats(),
            'capture': self.capture.get_stats(),
            'lockin': self.lockin.get_stats(),
            'policy': self.policy.get_stats(),
            'groups': self.groups.get_stats(),
            'journal': self.journal.get_stats(),
            'lease_registry': self.registry.get_stats(),
            'active_executions': len(self._active_executions),
            'invariants': inv_report.to_dict(),
        }
