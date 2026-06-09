"""
Execution Policy — governance layer for scheduling decisions.

Makes execution authority out of coordination dynamics:

  - select_next_goal() — which goal runs next, based on execution pressure
  - compute_execution_score() — unified score: capture + coherence + starvation
  - should_preempt() — whether a high-pressure goal interrupts current execution
  - should_retry() — whether persistence physics justifies another attempt

This replaces:
  - Flat Celery queue ordering
  - Progress-based pseudo-priority
  - Blind retry logic

Execution pressure is NOT progress.
Execution pressure = f(blocked_dependents, retry_history, persistence_weight,
                       time_starvation, user_priority, coherence_decay)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import math
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Execution Pressure — the real priority signal
# ============================================================================

@dataclass
class ExecutionPressure:
    """
    Unified execution priority signal.

    Computed from coordination dynamics, NOT from goal.progress.

    Components:
      - dependency_pressure: how many goals are blocked on this one
      - starvation: how long since last execution attempt (normalized)
      - persistence_weight: accumulated importance from execution history
      - retry_pressure: failed attempts increase urgency
      - coherence_decay: how much the execution path has degraded
      - user_priority: explicit user-assigned priority (if available)
    """
    dependency_pressure: float = 0.0
    starvation: float = 0.0
    persistence_weight: float = 0.5
    retry_pressure: float = 0.0
    coherence_decay: float = 0.0
    user_priority: float = 0.5

    @property
    def total(self) -> float:
        """Aggregate pressure score."""
        return (
            self.dependency_pressure * 0.25
            + self.starvation * 0.20
            + self.persistence_weight * 0.20
            + self.retry_pressure * 0.15
            + self.coherence_decay * 0.10
            + self.user_priority * 0.10
        )

    def to_dict(self) -> dict:
        return {
            'total': round(self.total, 4),
            'dependency_pressure': round(self.dependency_pressure, 4),
            'starvation': round(self.starvation, 4),
            'persistence_weight': round(self.persistence_weight, 4),
            'retry_pressure': round(self.retry_pressure, 4),
            'coherence_decay': round(self.coherence_decay, 4),
            'user_priority': round(self.user_priority, 4),
        }


# ============================================================================
# Anti-fragmentation Groups
# ============================================================================

@dataclass
class ExecutionGroup:
    """
    Group of tasks that should be co-scheduled to prevent fragmentation.

    Tasks in the same group:
      - Share coherence tracking
      - Are preferentially scheduled together
      - Resist queue scattering

    A group can be:
      - A goal tree (root goal + all subgoals)
      - An explicit user-defined group
      - An automatically detected execution cluster
    """
    group_id: str
    goal_ids: List[str]
    coherence: float = 1.0
    created_at: float = 0.0
    last_scheduled: float = 0.0

    def to_dict(self) -> dict:
        return {
            'group_id': self.group_id,
            'n_goals': len(self.goal_ids),
            'coherence': round(self.coherence, 4),
            'last_scheduled': self.last_scheduled,
        }


class ExecutionGroupRegistry:
    """
    Manages anti-fragmentation groups.

    Lightweight — stored in memory + PostgreSQL (not Redis).
    """

    def __init__(self):
        self._groups: Dict[str, ExecutionGroup] = {}
        self._goal_to_group: Dict[str, str] = {}

    def register_goal(self, goal_id: str, group_id: Optional[str] = None) -> str:
        """Assign a goal to a group. Creates new group if needed."""
        if goal_id in self._goal_to_group:
            return self._goal_to_group[goal_id]

        gid = group_id or f"exec_group_{goal_id[:8]}"

        if gid not in self._groups:
            self._groups[gid] = ExecutionGroup(
                group_id=gid,
                goal_ids=[],
                created_at=time.time(),
            )

        self._groups[gid].goal_ids.append(goal_id)
        self._goal_to_group[goal_id] = gid
        return gid

    def get_group(self, goal_id: str) -> Optional[ExecutionGroup]:
        """Get the group for a goal."""
        gid = self._goal_to_group.get(goal_id)
        if gid and gid in self._groups:
            return self._groups[gid]
        return None

    def get_group_ids(self) -> List[str]:
        return list(self._groups.keys())

    def get_coherence(self, group_id: str) -> float:
        group = self._groups.get(group_id)
        return group.coherence if group else 1.0

    def update_coherence(self, group_id: str, delta: float):
        group = self._groups.get(group_id)
        if group:
            group.coherence = max(0.1, min(1.0, group.coherence + delta))
            group.last_scheduled = time.time()

    def remove_goal(self, goal_id: str):
        """Remove a goal from its group."""
        gid = self._goal_to_group.pop(goal_id, None)
        if gid and gid in self._groups:
            group = self._groups[gid]
            if goal_id in group.goal_ids:
                group.goal_ids.remove(goal_id)
            if not group.goal_ids:
                del self._groups[gid]

    def get_stats(self) -> dict:
        if not self._groups:
            return {'n_groups': 0}
        coherences = [g.coherence for g in self._groups.values()]
        return {
            'n_groups': len(self._groups),
            'mean_coherence': round(sum(coherences) / len(coherences), 4),
            'min_coherence': round(min(coherences), 4),
            'max_coherence': round(max(coherences), 4),
        }

    # ------------------------------------------------------------------
    # Snapshot state (deterministic, serializable)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Serialize group registry for snapshot."""
        result = {}
        for gid, group in self._groups.items():
            result[str(gid)] = {
                'coherence': group.coherence,
                'member_count': len(group.goal_ids),
                'member_ids': [str(g) for g in group.goal_ids],
                'last_scheduled': getattr(group, 'last_scheduled', 0.0),
            }
        return result

    def restore_state(self, state: dict):
        """Restore group registry from snapshot."""
        self._groups.clear()
        self._goal_to_group.clear()
        for gid, gs in state.items():
            from .snapshot import SerializableGroupState
            sg = SerializableGroupState.from_dict({'group_id': gid, **gs})
            # Reconstruct ExecutionGroup from SerializableGroupState
            group = ExecutionGroup(
                group_id=gid,
                goal_ids=list(gs.get('member_ids', [])),
                coherence=sg.coherence,
                last_scheduled=gs.get('last_scheduled', 0.0),
            )
            self._groups[str(gid)] = group
            # Rebuild goal_to_group index
            for goal_id in group.goal_ids:
                self._goal_to_group[goal_id] = gid


# ============================================================================
# Execution Policy
# ============================================================================

class ExecutionPolicy:
    """
    Execution governance — decisions based on coordination dynamics.

    This is the authority layer between Scheduler and GoalExecutor.
    Every execution decision goes through this policy.

    Responsibilities:
      - Compute execution scores from pressure + dynamics
      - Select next goal from candidates
      - Decide preemption
      - Decide retry
    """

    def __init__(self, group_registry: Optional[ExecutionGroupRegistry] = None):
        self._group_registry = group_registry or ExecutionGroupRegistry()
        self._execution_history: Dict[str, List[float]] = {}
        self._decision_log: List[dict] = []

    # ------------------------------------------------------------------
    # Execution Pressure Computation
    # ------------------------------------------------------------------

    def compute_pressure(
        self,
        goal_id: str,
        goal_status: str = "pending",
        progress: float = 0.0,
        blocked_dependents: int = 0,
        hours_since_last_execution: float = 24.0,
        retry_count: int = 0,
        persistence_weight: float = 0.5,
        coherence: float = 1.0,
        user_priority: float = 0.5,
    ) -> ExecutionPressure:
        """
        Compute execution pressure from available signals.

        Pure function — no side effects.
        """
        # Dependency pressure
        dep_pressure = min(1.0, blocked_dependents / 10.0)

        # Starvation: how long since last attempt
        starvation = min(1.0, hours_since_last_execution / 72.0)

        # Retry pressure: failed attempts increase urgency
        retry_pressure = min(1.0, retry_count * 0.15)

        # Coherence decay: if coherence is dropping, pressure rises
        coherence_decay = max(0.0, 1.0 - coherence)

        return ExecutionPressure(
            dependency_pressure=dep_pressure,
            starvation=starvation,
            persistence_weight=persistence_weight,
            retry_pressure=retry_pressure,
            coherence_decay=coherence_decay,
            user_priority=user_priority,
        )

    # ------------------------------------------------------------------
    # Goal Selection
    # ------------------------------------------------------------------

    def compute_execution_score(self, pressure: ExecutionPressure) -> float:
        """Unified score — higher = should execute sooner."""
        # Base: total pressure
        score = pressure.total

        # Starvation bonus: long-waiting tasks get non-linear boost
        if pressure.starvation > 0.7:
            score += pressure.starvation * 0.2

        # Persistence bonus: important tasks get edge
        if pressure.persistence_weight > 0.7:
            score += pressure.persistence_weight * 0.1

        return round(score, 4)

    def select_next_goal(
        self,
        candidates: List[Dict[str, Any]],
        running_count: int = 0,
        max_concurrent: int = 8,
    ) -> Optional[str]:
        """
        Select the next goal to execute from candidates.

        Higher execution score = selected first.
        Respects group coherence (co-schedule group members).
        """
        if not candidates:
            return None

        if running_count >= max_concurrent:
            return None

        scored = []
        for c in candidates:
            pressure = self.compute_pressure(
                goal_id=c.get('goal_id', ''),
                goal_status=c.get('status', 'pending'),
                progress=c.get('progress', 0.0),
                blocked_dependents=c.get('blocked_dependents', 0),
                hours_since_last_execution=c.get('hours_idle', 24.0),
                retry_count=c.get('retry_count', 0),
                persistence_weight=c.get('persistence_weight', 0.5),
                coherence=c.get('coherence', 1.0),
                user_priority=c.get('user_priority', 0.5),
            )
            score = self.compute_execution_score(pressure)

            # Anti-fragmentation bonus: prefer goals from recently-scheduled groups
            group = self._group_registry.get_group(c.get('goal_id', ''))
            if group and group.last_scheduled > 0:
                recency = min(1.0, (time.time() - group.last_scheduled) / 3600.0)
                if recency < 0.5:
                    score += 0.05  # Small bonus for group continuity

            scored.append((score, c.get('goal_id', '')))

        # Sort descending by score
        scored.sort(key=lambda x: -x[0])
        selected = scored[0][1]

        self._decision_log.append({
            'time': time.time(),
            'selected': selected,
            'n_candidates': len(candidates),
            'top_score': scored[0][0] if scored else 0,
            'running': running_count,
        })

        return selected

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def should_preempt(
        self,
        current_score: float,
        candidate_score: float,
        current_duration_minutes: float = 0.0,
        preemption_threshold: float = 0.3,
    ) -> bool:
        """
        Should a high-pressure goal preempt current execution?

        Preempt only if:
          - Candidate score is significantly higher
          - Current execution has been running long enough to checkpoint
          - The gap exceeds threshold
        """
        if candidate_score <= current_score:
            return False

        gap = candidate_score - current_score
        if gap < preemption_threshold:
            return False

        # Don't preempt very short executions (under 30s)
        if current_duration_minutes < 0.5:
            return False

        # Don't preempt if current is near completion (>2x threshold gap)
        if current_score > 0.8 and gap < preemption_threshold * 2:
            return False

        return True

    # ------------------------------------------------------------------
    # Retry Decision
    # ------------------------------------------------------------------

    def should_retry(
        self,
        goal_id: str,
        retry_count: int = 0,
        max_retries: int = 3,
        persistence_weight: float = 0.5,
        is_hard_failure: bool = False,
    ) -> bool:
        """
        Should a failed execution be retried?

        Persistence physics:
          - High-persistence tasks get more retries
          - Hard failures (invalid input, unauthorized) skip retry
          - Retries decay in priority
        """
        if is_hard_failure:
            return False

        if retry_count >= max_retries:
            return False

        # Persistence-weighted retry budget
        effective_max = max_retries + int(persistence_weight * 2)
        if retry_count >= effective_max:
            return False

        return True

    # ------------------------------------------------------------------
    # Execution History
    # ------------------------------------------------------------------

    def record_execution(self, goal_id: str, success: bool, duration_ms: float):
        """Record execution outcome for starvation/pressure tracking."""
        if goal_id not in self._execution_history:
            self._execution_history[goal_id] = []
        self._execution_history[goal_id].append(time.time())

        # Prune old entries
        cutoff = time.time() - 86400 * 7  # 7 days
        self._execution_history[goal_id] = [
            t for t in self._execution_history[goal_id] if t > cutoff
        ]

    def hours_since_last_execution(self, goal_id: str) -> float:
        """Hours since last execution attempt."""
        history = self._execution_history.get(goal_id, [])
        if not history:
            return 999.0
        return (time.time() - history[-1]) / 3600.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        n_decisions = len(self._decision_log)
        recent = self._decision_log[-100:] if len(self._decision_log) > 100 else self._decision_log
        return {
            'n_decisions': n_decisions,
            'n_recent_scores': len(recent),
            'mean_top_score': round(
                sum(d['top_score'] for d in recent) / max(1, len(recent)), 4
            ) if recent else 0.0,
            'n_active_tracked': len(self._execution_history),
        }

    # ------------------------------------------------------------------
    # Snapshot state (deterministic, serializable)
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """Serialize policy state for snapshot."""
        return {
            'execution_history': {
                str(gid): list(ts) for gid, ts in self._execution_history.items()
            },
            'decision_log': self._decision_log[-500:],  # Keep last 500 for context
        }

    def restore_state(self, state: dict):
        """Restore policy state from snapshot."""
        self._execution_history.clear()
        for gid, history in state.get('execution_history', {}).items():
            self._execution_history[str(gid)] = list(history)
        self._decision_log.clear()
        for d in state.get('decision_log', []):
            self._decision_log.append(dict(d))
