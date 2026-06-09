"""AIOSState Builder — assembles the five-section AI-OS state snapshot."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from goal_state_calculator import GoalStateCalculator, STALLED_DAYS, ABANDONED_DAYS
from risk_engine import RiskEngine, risk_engine as default_risk_engine
from priority_engine import PriorityEngine, priority_engine as default_priority_engine


class AIOSState:
    """Single point-in-time snapshot of the AI-OS state.

    All five sections are computed from existing subsystems.
    No separate storage — this is always derived.
    """

    def __init__(
        self,
        goal_states: list[dict],
        risks: list[dict],
        priorities: list[dict],
    ):
        self.timestamp = datetime.now(timezone.utc)
        self.goal_states = goal_states
        self.risks = risks
        self.priorities = priorities

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "summary": self._summary(),
            "goals": self.goal_states,
            "risks": self.risks,
            "priorities": self.priorities,
            "active_executions": {"total": 0, "items": []},
            "recent_events": [],
        }

    def _summary(self):
        moving = sum(1 for g in self.goal_states if g.get("state") == "MOVING")
        stalled = sum(1 for g in self.goal_states if g.get("state") == "STALLED")
        blocked = sum(1 for g in self.goal_states if g.get("state") == "BLOCKED")
        completed = sum(1 for g in self.goal_states if g.get("state") == "COMPLETED")
        abandoned = sum(1 for g in self.goal_states if g.get("state") == "ABANDONED")
        return {
            "total": len(self.goal_states),
            "moving": moving,
            "stalled": stalled,
            "blocked": blocked,
            "completed": completed,
            "abandoned": abandoned,
        }


class AIOSStateBuilder:
    """Assembles AIOSState from existing subsystems.

    World state, active executions, and event journal are planned
    for future sections. Goal states, risks, and priorities are
    implemented in v1.
    """

    def __init__(
        self,
        goal_calculator: Optional[GoalStateCalculator] = None,
        risk_engine: Optional[RiskEngine] = None,
        priority_engine: Optional[PriorityEngine] = None,
    ):
        self.goal_calculator = goal_calculator or GoalStateCalculator()
        self.risk_engine = risk_engine or default_risk_engine
        self.priority_engine = priority_engine or default_priority_engine

    async def build(
        self,
        session: AsyncSession,
        stalled_days: int = STALLED_DAYS,
        abandoned_days: int = ABANDONED_DAYS,
    ) -> AIOSState:
        goal_states = await self.goal_calculator.compute_all(
            session=session,
            stalled_days=stalled_days,
            abandoned_days=abandoned_days,
        )
        return self.build_from_goal_states(goal_states)

    def build_from_goal_states(self, goal_states: list[dict]) -> AIOSState:
        risk_items = self.risk_engine.evaluate_all_goal_states(goal_states)
        priority_items = self.priority_engine.prioritize(goal_states)
        return AIOSState(
            goal_states=goal_states,
            risks=[r.to_dict() for r in risk_items],
            priorities=[p.to_dict() for p in priority_items],
        )


aios_state_builder = AIOSStateBuilder()
