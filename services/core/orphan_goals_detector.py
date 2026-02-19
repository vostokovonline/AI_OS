"""
Orphan Goals Detector

Finds goals without context (depth_level=0, not philosophical)
that were created more than X hours ago and need context binding.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from database import AsyncSessionLocal
from models import Goal
from sqlalchemy import select, and_, or_


class OrphanGoalsDetector:
    """Detects orphan goals that need context binding."""

    def __init__(self, orphan_age_hours: int = 24):
        """
        Args:
            orphan_age_hours: How old a goal must be (in hours) to be considered orphan
        """
        self.orphan_age_hours = orphan_age_hours

    async def find_orphan_goals(self, limit: int = 5) -> list:
        """
        Find orphan goals that need context binding.

        Orphan goal criteria:
        - depth_level = 0 (no parent)
        - NOT philosophical (philosophical goals can be root-level)
        - Created more than orphan_age_hours ago
        - Still pending or active (not done/failed)

        Args:
            limit: Maximum number of orphans to return

        Returns:
            List of orphan goals with context
        """
        async with AsyncSessionLocal() as db:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.orphan_age_hours)

            stmt = select(Goal).where(
                and_(
                    Goal.depth_level == 0,  # No parent
                    Goal.goal_type != "philosophical",  # Not philosophical (those can be root)
                    Goal.created_at < cutoff_time,  # Older than threshold
                    Goal.status.in_(["pending", "active"]),  # Still relevant
                )
            ).limit(limit).order_by(Goal.created_at)

            result = await db.execute(stmt)
            orphans = result.scalars().all()

            return [
                {
                    "id": str(goal.id),
                    "title": goal.title,
                    "description": goal.description,
                    "goal_type": goal.goal_type,
                    "created_at": goal.created_at.isoformat() if goal.created_at else None,
                    "age_hours": (datetime.now(timezone.utc) - goal.created_at).total_seconds() / 3600
                    if goal.created_at
                    else 0,
                }
                for goal in orphans
            ]

    async def get_orphan_stats(self) -> dict:
        """
        Get statistics about orphan goals.

        Returns:
            Dictionary with orphan stats
        """
        async with AsyncSessionLocal() as db:
            # Count all depth_level=0 goals (excluding philosophical)
            stmt_total = select(Goal).where(
                and_(
                    Goal.depth_level == 0,
                    Goal.goal_type != "philosophical",
                )
            )
            result_total = await db.execute(stmt_total)
            total_orphans = result_total.scalars().all()

            # Count recent orphans (< 24 hours)
            cutoff_recent = datetime.now(timezone.utc) - timedelta(hours=24)
            stmt_recent = select(Goal).where(
                and_(
                    Goal.depth_level == 0,
                    Goal.goal_type != "philosophical",
                    Goal.created_at >= cutoff_recent,
                )
            )
            result_recent = await db.execute(stmt_recent)
            recent_orphans = result_recent.scalars().all()

            # Count old orphans (>= 24 hours) - these need context
            cutoff_old = datetime.now(timezone.utc) - timedelta(hours=24)
            stmt_old = select(Goal).where(
                and_(
                    Goal.depth_level == 0,
                    Goal.goal_type != "philosophical",
                    Goal.created_at < cutoff_old,
                    Goal.status.in_(["pending", "active"]),
                )
            )
            result_old = await db.execute(stmt_old)
            old_orphans = result_old.scalars().all()

            return {
                "total_root_goals": len(total_orphans),
                "recent_orphans": len(recent_orphans),
                "orphans_need_context": len(old_orphans),
                "orphan_age_threshold_hours": self.orphan_age_hours,
            }


# Singleton instance
orphan_goals_detector = OrphanGoalsDetector(orphan_age_hours=24)


async def main():
    """Test the detector."""
    print("🔍 Orphan Goals Detector")
    print("=" * 50)

    # Get stats
    stats = await orphan_goals_detector.get_orphan_stats()
    print(f"\n📊 Stats:")
    print(f"  Total root goals: {stats['total_root_goals']}")
    print(f"  Recent orphans (<24h): {stats['recent_orphans']}")
    print(f"  Need context: {stats['orphans_need_context']}")

    # Find orphans
    orphans = await orphan_goals_detector.find_orphan_goals(limit=10)

    if orphans:
        print(f"\n🍼 Found {len(orphans)} orphan goals:")
        for orphan in orphans:
            print(f"\n  📌 {orphan['title']}")
            print(f"     Type: {orphan['goal_type']}")
            print(f"     Age: {orphan['age_hours']:.1f} hours")
            print(f"     Created: {orphan['created_at']}")
    else:
        print("\n✅ No orphan goals found - all root goals have context!")


if __name__ == "__main__":
    asyncio.run(main())
