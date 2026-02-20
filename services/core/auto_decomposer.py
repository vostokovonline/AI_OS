"""
AUTO DECOMPOSER - Background decomposer for stuck non-atomic goals
================================================================

Автоматически decomposes pending non-atomic goals.

Problem: Non-atomic goals created but never decomposed
Solution: Background job finds stuck goals and decomposes them

Author: AI-OS Core Team
Date: 2026-02-11
Severity: CRITICAL FIX
"""

from typing import List, Dict
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from database import AsyncSessionLocal
from models import Goal
from goal_decomposer import goal_decomposer

# Centralized logging
from logging_config import get_logger

logger = get_logger(__name__)


class AutoDecomposer:
    """
    Автоматически decomposes pending non-atomic goals

    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ для stuck goals
    """

    def __init__(self):
        self.stuck_threshold_hours = 1  # 1 hour in pending = trigger decompose

    async def scan_and_decompose_stuck_goals(self) -> Dict:
        """
        Сканирует и decomposes застрявшие цели

        Returns:
            {
                "scanned": int,
                "decomposed": int,
                "skipped": int,
                "failed": int,
                "details": [...]
            }
        """
        async with AsyncSessionLocal() as db:
            # Находим pending non-atomic goals старше threshold
            threshold_time = datetime.now() - timedelta(hours=self.stuck_threshold_hours)

            stmt = select(Goal).where(
                and_(
                    Goal.status == "pending",
                    Goal.is_atomic == False,
                    Goal.created_at < threshold_time
                )
            ).order_by(Goal.created_at.asc())

            result = await db.execute(stmt)
            stuck_goals = result.scalars().all()

            report = {
                "scanned": len(stuck_goals),
                "decomposed": 0,
                "skipped": 0,
                "failed": 0,
                "details": []
            }

            for goal in stuck_goals:
                try:
                    # Проверяем: есть ли уже дети
                    stmt_children = select(func.count(Goal.id)).where(
                        Goal.parent_id == goal.id
                    )
                    result_children = await db.execute(stmt_children)
                    child_count = result_children.scalar() or 0

                    if child_count > 0:
                        # Уже decomposed - skip
                        report["skipped"] += 1
                        report["details"].append({
                            "goal_id": str(goal.id),
                            "title": goal.title[:50],
                            "action": "skipped",
                            "reason": f"already has {child_count} children"
                        })
                        continue

                    # Decompose
                    logger.info("auto_decomposing_goal", goal_title=goal.title)
                    logger.debug("goal_id", goal_id=str(goal.id))
                    logger.debug("goal_depth", depth=goal.depth_level)
                    logger.debug("goal_type", goal_type=goal.goal_type)
                    import datetime as dt
                    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                    created = goal.created_at.replace(tzinfo=None) if goal.created_at.tzinfo else goal.created_at
                    logger.debug("goal_age_hours", age=f"{(now - created).total_seconds() / 3600:.1f}")

                    subgoals = await goal_decomposer.decompose_goal(
                        goal_id=str(goal.id),
                        max_depth=3
                    )

                    if subgoals:
                        report["decomposed"] += 1
                        report["details"].append({
                            "goal_id": str(goal.id),
                            "title": goal.title[:50],
                            "action": "decomposed",
                            "children_created": len(subgoals)
                        })
                        logger.info("subgoals_created", count=len(subgoals))
                    else:
                        # Decompose вернул [] - возможно цель теперь atomic
                        await db.refresh(goal)

                        if goal.is_atomic:
                            report["skipped"] += 1
                            report["details"].append({
                                "goal_id": str(goal.id),
                                "title": goal.title[:50],
                                "action": "skipped",
                                "reason": "marked as atomic by decomposer"
                            })
                            logger.info("marked_as_atomic")
                        else:
                            report["failed"] += 1
                            report["details"].append({
                                "goal_id": str(goal.id),
                                "title": goal.title[:50],
                                "action": "failed",
                                "reason": "decompose returned [] but not atomic"
                            })
                            logger.warning("decompose_failed_empty_result")

                except Exception as e:
                    report["failed"] += 1
                    report["details"].append({
                        "goal_id": str(goal.id),
                        "title": goal.title[:50],
                        "action": "error",
                        "error": str(e)
                    })
                    logger.error("decompose_error", error=str(e))

            return report

    async def decompose_all_pending_non_atomic(self) -> Dict:
        """
        forcibly decompose ВСЕ pending non-atomic goals

        Использовать для emergency unblocking!
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(
                and_(
                    Goal.status == "pending",
                    Goal.is_atomic == False
                )
            ).order_by(Goal.created_at.asc())

            result = await db.execute(stmt)
            pending_goals = result.scalars().all()

            report = {
                "total": len(pending_goals),
                "decomposed": 0,
                "skipped": 0,
                "failed": 0
            }

            logger.info("emergency_decomposition_start")
            logger.warning("emergency_decomposition", count=len(pending_goals))

            import datetime as dt
            now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

            for i, goal in enumerate(pending_goals, 1):
                logger.info("emergency_decomposing", index=i, total=len(pending_goals), title=goal.title)
                created = goal.created_at.replace(tzinfo=None) if goal.created_at.tzinfo else goal.created_at
                logger.debug("goal_age_days", age=f"{(now - created).total_seconds() / 86400:.1f}")

                try:
                    subgoals = await goal_decomposer.decompose_goal(
                        goal_id=str(goal.id),
                        max_depth=3
                    )

                    if subgoals:
                        report["decomposed"] += 1
                        logger.info("subgoals_created", count=len(subgoals))
                    else:
                        await db.refresh(goal)
                        if goal.is_atomic:
                            report["skipped"] += 1
                            logger.debug("skipped_marked_atomic")
                        else:
                            report["failed"] += 1
                            logger.warning("decompose_failed_no_subgoals")

                except Exception as e:
                    report["failed"] += 1
                    logger.error("decompose_error", error=str(e))

            return report


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

auto_decomposer = AutoDecomposer()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def run_auto_decompose() -> Dict:
    """Run auto-decomposition for stuck goals"""
    return await auto_decomposer.scan_and_decompose_stuck_goals()


async def emergency_decompose_all() -> Dict:
    """Emergency: decompose ALL pending non-atomic goals"""
    return await auto_decomposer.decompose_all_pending_non_atomic()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing Auto Decomposer...\n")

        # Test 1: Scan for stuck goals
        report = await auto_decomposer.scan_and_decompose_stuck_goals()

        logger.info("emergency_decomposition_start")
        print(f"AUTO-DECOMPOSE REPORT")
        print(f"{'='*70}")
        print(f"Scanned: {report['scanned']}")
        print(f"Decomposed: {report['decomposed']}")
        print(f"Skipped: {report['skipped']}")
        print(f"Failed: {report['failed']}")

    asyncio.run(test())
