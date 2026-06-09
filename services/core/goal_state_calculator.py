"""GoalState Calculator — determines the state of a goal from its activity."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


class GoalState(str, Enum):
    ACTIVE = "ACTIVE"          # goal exists, no activity data yet
    MOVING = "MOVING"          # recent progress detected
    STALLED = "STALLED"        # no progress for STALLED_DAYS
    BLOCKED = "BLOCKED"        # known blocker exists
    COMPLETED = "COMPLETED"    # success criteria met
    ABANDONED = "ABANDONED"    # no progress for ABANDONED_DAYS


# ── Default thresholds ──────────────────────────────────────────────────

STALLED_DAYS = 7
ABANDONED_DAYS = 30


# ── Calculator ─────────────────────────────────────────────────────────

class GoalStateCalculator:
    """Pure-function state machine for goal health.

    Reads from Goal model (status, last_activity_at, blockers) and returns
    the current GoalState. No side effects, no storage.
    """

    def compute(
        self,
        status: str,
        last_activity_at: Optional[datetime],
        has_blockers: bool = False,
        is_completed: bool = False,
        stalled_days: int = STALLED_DAYS,
        abandoned_days: int = ABANDONED_DAYS,
        now: Optional[datetime] = None,
    ) -> GoalState:
        if is_completed:
            return GoalState.COMPLETED
        if status in ("done", "completed", "archived", "failed", "cancelled"):
            return GoalState.COMPLETED
        if has_blockers or status == "blocked":
            return GoalState.BLOCKED
        if last_activity_at is None:
            return GoalState.ACTIVE

        now = now or datetime.now(timezone.utc)
        days_since = (now - last_activity_at).days

        if days_since > abandoned_days:
            return GoalState.ABANDONED
        if days_since > stalled_days:
            return GoalState.STALLED
        return GoalState.MOVING

    async def compute_from_db(
        self,
        session: AsyncSession,
        goal_id,
        stalled_days: int = STALLED_DAYS,
        abandoned_days: int = ABANDONED_DAYS,
    ) -> GoalState:
        from models import Goal
        result = await session.execute(
            select(Goal).where(Goal.id == goal_id)
        )
        goal = result.scalar_one_or_none()
        if goal is None:
            return GoalState.ABANDONED

        return self.compute(
            status=goal.status,
            last_activity_at=goal.last_activity_at,
            has_blockers=False,  # TODO: check blockers table when it exists
            is_completed=goal.status == "done",
            stalled_days=stalled_days,
            abandoned_days=abandoned_days,
        )

    async def compute_all(
        self,
        session: AsyncSession,
        stalled_days: int = STALLED_DAYS,
        abandoned_days: int = ABANDONED_DAYS,
    ) -> list[dict]:
        from models import Goal
        from sqlalchemy import select
        result = await session.execute(select(Goal))
        goals = result.scalars().all()

        now = datetime.now(timezone.utc)
        output = []
        for g in goals:
            state = self.compute(
                status=g.status,
                last_activity_at=g.last_activity_at,
                has_blockers=False,
                is_completed=False,
                stalled_days=stalled_days,
                abandoned_days=abandoned_days,
                now=now,
            )
            stagnation = (now - g.last_activity_at).days if g.last_activity_at else 0
            output.append({
                "id": str(g.id),
                "title": g.title,
                "state": state.value,
                "stagnation_days": stagnation if state in (GoalState.STALLED, GoalState.ABANDONED) else 0,
                "progress": g.progress or 0.0,
            })
        return output


goal_state_calculator = GoalStateCalculator()
