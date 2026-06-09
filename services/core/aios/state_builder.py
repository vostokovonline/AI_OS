"""AIOSState Builder — assembles the 5-section AI-OS state snapshot.

Only imports:
  - database (AsyncSession)
  - models (Goal via calculator)
  - aios.* (goal_state_calculator, risk_engine, priority_engine)

Zero imports from: experience, epistemic, cognitive_os, simulation, phase18.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from aios.goal_state_calculator import (
    GoalStateCalculator,
    STALLED_DAYS,
    ABANDONED_DAYS,
)
from aios.risk_engine import evaluate_all_goal_states as compute_risks
from aios.priority_engine import prioritize as compute_priorities


class AIOSState:
    """Point-in-time snapshot of AI-OS state."""

    def __init__(
        self,
        goals: list[dict],
        risks: list[dict],
        priorities: list[dict],
    ):
        self.timestamp = datetime.now(timezone.utc)
        self.goals = goals
        self.risks = risks
        self.priorities = priorities

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": self._summary(),
            "goals": self.goals,
            "risks": self.risks,
            "priorities": self.priorities,
            "active_executions": {"total": 0, "items": []},
            "recent_events": [],
        }

    def _summary(self):
        counts = {"total": len(self.goals)}
        for state in ("MOVING", "STALLED", "BLOCKED", "COMPLETED", "ABANDONED"):
            counts[state.lower()] = sum(
                1 for g in self.goals if g.get("state") == state
            )
        return counts


class AIOSStateBuilder:
    """Assembles AIOSState from goals table.

    Internal: calculator, risk_engine, priority_engine.
    No separate API — only build() and build_from_goal_states().
    """

    def __init__(
        self,
        calculator: Optional[GoalStateCalculator] = None,
    ):
        self.calculator = calculator or GoalStateCalculator()

    async def build(
        self,
        session: AsyncSession,
        stalled_days: int = STALLED_DAYS,
        abandoned_days: int = ABANDONED_DAYS,
    ) -> AIOSState:
        goal_states = await self.calculator.compute_all(
            session=session,
            stalled_days=stalled_days,
            abandoned_days=abandoned_days,
        )
        return self.build_from_goal_states(goal_states)

    def build_from_goal_states(self, goal_states: list[dict]) -> AIOSState:
        return AIOSState(
            goals=goal_states,
            risks=compute_risks(goal_states),
            priorities=compute_priorities(goal_states),
        )


aios_state_builder = AIOSStateBuilder()
