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
                    print(f"\n🔧 AUTO-DECOMPOSING: {goal.title}")
                    print(f"   ID: {goal.id}")
                    print(f"   Depth: L{goal.depth_level}")
                    print(f"   Type: {goal.goal_type}")
                    import datetime as dt
                    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                    created = goal.created_at.replace(tzinfo=None) if goal.created_at.tzinfo else goal.created_at
                    print(f"   Age: {(now - created).total_seconds() / 3600:.1f} hours")

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
                        print(f"   ✅ Created {len(subgoals)} subgoals")
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
                            print(f"   ℹ️  Marked as atomic (depth limit or other reason)")
                        else:
                            report["failed"] += 1
                            report["details"].append({
                                "goal_id": str(goal.id),
                                "title": goal.title[:50],
                                "action": "failed",
                                "reason": "decompose returned [] but not atomic"
                            })
                            print(f"   ❌ Failed: decompose returned []")

                except Exception as e:
                    report["failed"] += 1
                    report["details"].append({
                        "goal_id": str(goal.id),
                        "title": goal.title[:50],
                        "action": "error",
                        "error": str(e)
                    })
                    print(f"   ❌ ERROR: {e}")

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

            print(f"\n{'='*70}")
            print(f"EMERGENCY DECOMPOSITION: {len(pending_goals)} pending non-atomic goals")
            print(f"{'='*70}")

            import datetime as dt
            now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)

            for i, goal in enumerate(pending_goals, 1):
                print(f"\n[{i}/{len(pending_goals)}] {goal.title}")
                created = goal.created_at.replace(tzinfo=None) if goal.created_at.tzinfo else goal.created_at
                print(f"   Age: {(now - created).total_seconds() / 86400:.1f} days")

                try:
                    subgoals = await goal_decomposer.decompose_goal(
                        goal_id=str(goal.id),
                        max_depth=3
                    )

                    if subgoals:
                        report["decomposed"] += 1
                        print(f"   ✅ Created {len(subgoals)} subgoals")
                    else:
                        await db.refresh(goal)
                        if goal.is_atomic:
                            report["skipped"] += 1
                            print(f"   ⏭️  Skipped (marked atomic)")
                        else:
                            report["failed"] += 1
                            print(f"   ❌ Failed")

                except Exception as e:
                    report["failed"] += 1
                    print(f"   ❌ ERROR: {e}")

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

        print(f"\n{'='*70}")
        print(f"AUTO-DECOMPOSE REPORT")
        print(f"{'='*70}")
        print(f"Scanned: {report['scanned']}")
        print(f"Decomposed: {report['decomposed']}")
        print(f"Skipped: {report['skipped']}")
        print(f"Failed: {report['failed']}")

    asyncio.run(test())
