"""Goal Activity Service — records what counts as progress toward a goal."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Goal


# ── Activity types ──────────────────────────────────────────────────────

GOAL_ACTIVITY_TYPES = frozenset({
    "execution_completed",
    "artifact_created",
    "subgoal_completed",
    "metric_improved",
    "blocker_added",
    "blocker_resolved",
})


def is_valid_activity(activity_type: str) -> bool:
    return activity_type in GOAL_ACTIVITY_TYPES


# ── Service ─────────────────────────────────────────────────────────────

class GoalActivityService:
    """Records activity against a goal and updates last_activity_at.

    Activity is what distinguishes MOVING from STALLED.
    Only meaningful events count — viewing or editing a goal does not.
    """

    async def record_activity(
        self,
        session: AsyncSession,
        goal_id: UUID,
        activity_type: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if not is_valid_activity(activity_type):
            return

        ts = timestamp or datetime.utcnow()
        result = await session.execute(
            select(Goal).where(Goal.id == goal_id)
        )
        goal = result.scalar_one_or_none()
        if goal is None:
            return

        goal.last_activity_at = ts
        await session.flush([goal])

    async def record_activity_bulk(
        self,
        session: AsyncSession,
        goal_ids: list[UUID],
        activity_type: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        if not is_valid_activity(activity_type):
            return

        ts = timestamp or datetime.utcnow()
        result = await session.execute(
            select(Goal).where(Goal.id.in_(goal_ids))
        )
        goals = result.scalars().all()
        for goal in goals:
            goal.last_activity_at = ts
        await session.flush(goals)


goal_activity_service = GoalActivityService()
