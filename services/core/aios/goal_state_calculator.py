"""Goal State Calculator — pure function, no DB access by default.

Computes GoalState from runtime fields (status, last_activity_at, blockers).
Zero imports from experience, epistemic, cognitive layers.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

STALLED_DAYS = 7
ABANDONED_DAYS = 30


class GoalState(str, Enum):
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    ABANDONED = "ABANDONED"
    STALLED = "STALLED"
    MOVING = "MOVING"
    ACTIVE = "ACTIVE"


def compute(
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


class GoalStateCalculator:
    """Computes GoalState for one or all goals.

    Pure function (compute) + DB helper (compute_all).
    DB helper only imports Goal + select from sqlalchemy.
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
        return compute(
            status=status,
            last_activity_at=last_activity_at,
            has_blockers=has_blockers,
            is_completed=is_completed,
            stalled_days=stalled_days,
            abandoned_days=abandoned_days,
            now=now,
        )

    async def compute_all(
        self,
        session: AsyncSession,
        stalled_days: int = STALLED_DAYS,
        abandoned_days: int = ABANDONED_DAYS,
    ) -> list[dict]:
        from models import Goal

        rows = await session.execute(select(Goal))
        goals = rows.scalars().all()

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
                "stagnation_days": stagnation
                if state in (GoalState.STALLED, GoalState.ABANDONED)
                else 0,
                "progress": g.progress or 0.0,
            })
        return output


goal_state_calculator = GoalStateCalculator()
