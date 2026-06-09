"""
Two-Loop Architecture for AI_OS

Architecture:
1. External Goals (user's commands) - HIGHEST authority
2. Internal Goals (system maintenance) - ONLY when idle or degraded

Key principle:
System maintains cognitive health BETWEEN user commands,
NOT instead of executing them.
"""

import asyncio
import sys
import time
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import uuid

sys.path.insert(0, '/home/onor/ai_os_final/services/core/experience')

from cognitive_loop import (
    CognitiveLoopState,
    CognitiveLoopConfig,
    FilterConfig,
    detect_pressure_tension,
    add_tension,
    resolve_tension,
    get_top_tensions,
    TensionState
)


class GoalPriority(Enum):
    """Goal priority hierarchy - EXTERNAL always wins"""
    USER_CRITICAL = 100
    USER_HIGH = 90
    USER_NORMAL = 80
    SYSTEM_RECOVERY = 40
    SYSTEM_MAINTENANCE = 30
    SYSTEM_REFLECTION = 20
    SYSTEM_EXPLORATION = 10


@dataclass
class ExternalGoal:
    """A user-provided goal - HIGHEST priority"""
    goal_id: str
    title: str
    description: str
    priority: GoalPriority = GoalPriority.USER_NORMAL
    created_at: str = ""
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


@dataclass
class InternalGoal:
    """A system maintenance goal - ONLY when idle"""
    goal_id: str
    title: str
    description: str
    priority: GoalPriority
    created_at: str = ""
    max_runtime_seconds: float = 30.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


@dataclass
class GoalExecutionResult:
    goal_id: str
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    authority: str = "external"


class GoalArbitrationLayer:
    """
    Goal Priority Arbitration

    User goals almost ALWAYS win.
    Self-goals only run when:
    - No pending user goals
    - System is idle
    - System is degraded (needs recovery)
    """

    MAX_SELF_GENERATED_GOALS = 3
    MAX_REFLECTION_DEPTH = 2
    IDLE_THRESHOLD = 0.7

    def __init__(self):
        self.external_goals: List[ExternalGoal] = []
        self.internal_goals: List[InternalGoal] = []
        self.completed_goals: List[GoalExecutionResult] = []
        self.self_goals_this_cycle = 0
        self.reflection_depth = 0

    def add_external_goal(self, goal: ExternalGoal):
        """Add a user goal - ALWAYS accepted"""
        self.external_goals.append(goal)

    def add_internal_goal(self, goal: InternalGoal):
        """Add a system goal - only if within quota"""
        if self.self_goals_this_cycle >= self.MAX_SELF_GENERATED_GOALS:
            return
        self.internal_goals.append(goal)
        self.self_goals_this_cycle += 1

    def get_next_goal(self) -> Optional[GoalExecutionResult]:
        """Get next goal - EXTERNAL wins"""
        # External goals ALWAYS win
        for goal in self.external_goals:
            if goal.priority in [GoalPriority.USER_CRITICAL, GoalPriority.USER_HIGH]:
                self.external_goals.remove(goal)
                return GoalExecutionResult(
                    goal_id=goal.goal_id,
                    success=False,
                    authority="external",
                    output=goal.title
                )

        # User normal
        if self.external_goals:
            goal = self.external_goals[0]
            self.external_goals.remove(goal)
            return GoalExecutionResult(
                goal_id=goal.goal_id,
                success=False,
                authority="external",
                output=goal.title
            )

        # System goals ONLY if idle
        if self._is_idle():
            # Recovery
            recovery = [g for g in self.internal_goals
                        if g.priority == GoalPriority.SYSTEM_RECOVERY]
            if recovery:
                goal = recovery[0]
                self.internal_goals.remove(goal)
                return GoalExecutionResult(
                    goal_id=goal.goal_id,
                    success=False,
                    authority="internal",
                    output=goal.title
                )

            # Maintenance
            maintenance = [g for g in self.internal_goals
                           if g.priority == GoalPriority.SYSTEM_MAINTENANCE]
            if maintenance:
                goal = maintenance[0]
                self.internal_goals.remove(goal)
                return GoalExecutionResult(
                    goal_id=goal.goal_id,
                    success=False,
                    authority="internal",
                    output=goal.title
                )

            # Light reflection
            if self.reflection_depth < self.MAX_REFLECTION_DEPTH:
                reflection = [g for g in self.internal_goals
                              if g.priority == GoalPriority.SYSTEM_REFLECTION]
                if reflection:
                    goal = reflection[0]
                    self.internal_goals.remove(goal)
                    self.reflection_depth += 1
                    return GoalExecutionResult(
                        goal_id=goal.goal_id,
                        success=False,
                        authority="internal",
                        output=goal.title
                    )

        return None

    def _is_idle(self) -> bool:
        """Check if system is idle"""
        return not self.external_goals

    def complete_goal(self, result: GoalExecutionResult):
        self.completed_goals.append(result)

    def reset_cycle(self):
        self.self_goals_this_cycle = 0
        self.reflection_depth = 0


class CognitiveMaintenanceSystem:
    """System that maintains cognitive health BETWEEN user commands"""

    MAX_MEMORY_TIME = 10.0
    MAX_FAILURE_TIME = 15.0
    MAX_PRESSURE_TIME = 20.0

    def __init__(self):
        self.pressure_accumulated = 0.0
        self.failures_pending = 0
        self.memory_fragmentation = 0.0

    def needs_maintenance(self) -> bool:
        return (self.pressure_accumulated > 0.5 or
                self.failures_pending > 0 or
                self.memory_fragmentation > 0.6)

    def is_idle(self) -> bool:
        return (self.pressure_accumulated < 0.3 and
                self.failures_pending == 0 and
                self.memory_fragmentation < 0.4)

    def generate_maintenance_goals(self) -> List[InternalGoal]:
        goals = []
        if self.failures_pending > 0:
            goals.append(InternalGoal(
                goal_id=f"recovery_{uuid.uuid4().hex[:8]}",
                title=f"Recover from {self.failures_pending} failures",
                description="Analyze and resolve pending failures",
                priority=GoalPriority.SYSTEM_RECOVERY,
                max_runtime_seconds=self.MAX_FAILURE_TIME
            ))
        if self.memory_fragmentation > 0.5:
            goals.append(InternalGoal(
                goal_id=f"memory_{uuid.uuid4().hex[:8]}",
                title="Consolidate fragmented memory",
                description="Defragment and optimize memory",
                priority=GoalPriority.SYSTEM_MAINTENANCE,
                max_runtime_seconds=self.MAX_MEMORY_TIME
            ))
        if self.pressure_accumulated > 0.6:
            goals.append(InternalGoal(
                goal_id=f"pressure_{uuid.uuid4().hex[:8]}",
                title="Reduce cognitive pressure",
                description="Analyze and reduce pressure",
                priority=GoalPriority.SYSTEM_REFLECTION,
                max_runtime_seconds=self.MAX_PRESSURE_TIME
            ))
        return goals


class TwoLoopCognitiveDaemon:
    """
    Two-Loop Cognitive Daemon

    External loop: User goals (HIGHEST priority)
    Internal loop: Cognitive maintenance (ONLY when idle)

    NOT an autonomous agent trying to "do things".
    A RESPONSIVE system that:
    1. Executes user goals
    2. Maintains cognitive health BETWEEN goals
    3. Never blocks user execution
    """

    def __init__(self, name: str = "ai_os_two_loop"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        self.arbitration = GoalArbitrationLayer()
        self.maintenance = CognitiveMaintenanceSystem()

        self.cognitive_state = CognitiveLoopState.initial(
            filter_config=FilterConfig(
                noise_threshold=0.3,
                min_authority=0.2,
                max_inputs_per_cycle=10,
                novelty_bonus=0.2,
                repetition_penalty=0.4
            ),
            loop_config=CognitiveLoopConfig(
                attention_budget=0.7,
                max_tensions=20,
                tension_threshold=0.4,
                salience_threshold=0.6,
                goal_generation_rate=0.5,
                adaptation_rate=0.1
            )
        )

        # Identity (slow evolution)
        self.identity = {
            'autonomy': 0.5,
            'curiosity': 0.4,
            'stability': 0.6,
            'coherence': 0.7,
            'version': 0
        }

        # Metrics
        self.user_goals_completed = 0
        self.internal_goals_completed = 0
        self.total_idle_cycles = 0

        # Lineage for audit
        self.lineage: List[Dict[str, Any]] = []

    def add_user_goal(self, title: str, description: str = "",
                      priority: GoalPriority = GoalPriority.USER_NORMAL):
        """Add a user goal - will be executed with highest priority"""
        goal = ExternalGoal(
            goal_id=f"user_{uuid.uuid4().hex[:8]}",
            title=title,
            description=description,
            priority=priority
        )
        self.arbitration.add_external_goal(goal)
        self._record('goal_added', {'title': title, 'priority': priority.name})

    def _record(self, event_type: str, data: Dict[str, Any]):
        self.lineage.append({
            'type': event_type,
            'data': data,
            'cycle': self.cycle_count,
            'timestamp': datetime.now(UTC).isoformat()
        })

    async def execute_user_goal(self, goal: ExternalGoal) -> GoalExecutionResult:
        start = time.time()
        success = True
        output = {
            'executed': goal.title,
            'cycle': self.cycle_count,
            'identity': self.identity.copy()
        }
        self.user_goals_completed += 1
        self._record('user_goal_executed', {'goal_id': goal.goal_id, 'success': success})

        return GoalExecutionResult(
            goal_id=goal.goal_id,
            success=success,
            output=output,
            execution_time=time.time() - start,
            authority="external"
        )

    async def execute_internal_goal(self, goal: InternalGoal) -> GoalExecutionResult:
        start = time.time()
        success = True

        # Apply maintenance effects
        if "recovery" in goal.title.lower():
            self.failures_pending = max(0, self.failures_pending - 1)
        if "memory" in goal.title.lower():
            self.maintenance.memory_fragmentation *= 0.7
        if "pressure" in goal.title.lower():
            self.maintenance.pressure_accumulated *= 0.5

        self.internal_goals_completed += 1
        self._record('internal_goal_executed', {'goal_id': goal.goal_id})

        return GoalExecutionResult(
            goal_id=goal.goal_id,
            success=success,
            output={'maintenance': goal.title},
            execution_time=time.time() - start,
            authority="internal"
        )

    async def run_cognitive_maintenance(self):
        """Run internal maintenance - ONLY when idle or degraded"""
        if self.maintenance.needs_maintenance():
            # Generate recovery goals
            recovery_goals = self.maintenance.generate_maintenance_goals()
            for goal in recovery_goals:
                self.arbitration.add_internal_goal(goal)

            # Execute recovery (not idle, but degraded)
            while self.arbitration.internal_goals:
                result = self.arbitration.get_next_goal()
                if result and result.authority == "internal":
                    goal = InternalGoal(
                        goal_id=result.goal_id,
                        title=result.output or "maintenance",
                        description="",
                        priority=GoalPriority.SYSTEM_MAINTENANCE
                    )
                    await self.execute_internal_goal(goal)
                    self.arbitration.complete_goal(result)
        else:
            self.total_idle_cycles += 1
            # Light pressure reduction when idle
            if self.cycle_count % 5 == 0:
                self.maintenance.pressure_accumulated = max(
                    0.0, self.maintenance.pressure_accumulated - 0.05
                )

    async def run_cycle(self, interval: float = 5.0):
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # Get next goal (external wins)
        next_goal = self.arbitration.get_next_goal()

        if next_goal:
            if next_goal.authority == "external":
                goal = ExternalGoal(
                    goal_id=next_goal.goal_id,
                    title=next_goal.output or "user goal",
                    description=""
                )
                result = await self.execute_user_goal(goal)
                self.arbitration.complete_goal(result)

                status = "SUCCESS" if result.success else "FAILED"
                print(f"[{uptime:.0f}s] C{self.cycle_count} | "
                      f"USER: {goal.title[:35]:35s} | {status}")

            else:
                goal = InternalGoal(
                    goal_id=next_goal.goal_id,
                    title=next_goal.output or "maintenance",
                    description="",
                    priority=GoalPriority.SYSTEM_MAINTENANCE
                )
                result = await self.execute_internal_goal(goal)
                self.arbitration.complete_goal(result)

                print(f"[{uptime:.0f}s] C{self.cycle_count} | "
                      f"MAINTENANCE: {goal.title[:30]}")

        else:
            await self.run_cognitive_maintenance()
            print(f"[{uptime:.0f}s] C{self.cycle_count} | IDLE (no pending goals)")

        self.arbitration.reset_cycle()
        await self.run_cognitive_maintenance()

    async def run(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        print(f"\n{'='*60}")
        print(f"TWO-LOOP COGNITIVE DAEMON")
        print(f"{'='*60}")
        print(f"Name: {self.name}")
        print(f"External goals: HIGHEST priority")
        print(f"Internal goals: ONLY when idle")
        print(f"Goal hierarchy:")
        print(f"  USER_CRITICAL > USER_HIGH > USER_NORMAL > SYSTEM_RECOVERY > ...")
        print(f"{'='*60}\n")

        try:
            while True:
                await self.run_cycle(interval)
                if max_cycles and self.cycle_count >= max_cycles:
                    break
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print("DAEMON STOPPED")
            print(f"{'='*60}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total cycles: {self.cycle_count}")
        print(f"User goals completed: {self.user_goals_completed}")
        print(f"Internal goals completed: {self.internal_goals_completed}")
        print(f"Idle cycles: {self.total_idle_cycles}")
        print(f"Lineage events: {len(self.lineage)}")
        print(f"Identity: {self.identity}")
        print(f"{'='*60}\n")


async def demo():
    """Demo showing two-loop architecture"""
    daemon = TwoLoopCognitiveDaemon(name="demo")

    # Add user goals
    daemon.add_user_goal("Analyze system logs", "Check for errors", GoalPriority.USER_NORMAL)
    daemon.add_user_goal("Deploy feature", "Deploy to production", GoalPriority.USER_HIGH)
    daemon.add_user_goal("Fix critical bug", "Fix auth bug", GoalPriority.USER_CRITICAL)

    # Run - user goals execute first, internal only in idle
    await daemon.run(interval=3.0, max_cycles=15)


if __name__ == "__main__":
    asyncio.run(demo())