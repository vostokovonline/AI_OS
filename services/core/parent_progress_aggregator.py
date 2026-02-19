"""
PARENT PROGRESS AGGREGATOR - Update parent progress when children complete
========================================================================

Автоматически обновляет parent.progress на основе children statuses.

Problem: Parent goals stuck at progress=0 even when children are done
Solution: Trigger updates progress when child status changes

Author: AI-OS Core Team
Date: 2026-02-11
Severity: CRITICAL FIX
"""

from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import select, func, and_, update
from database import AsyncSessionLocal
from models import Goal


class ParentProgressAggregator:
    """
    Агрегирует прогресс родительских целей на основе детей

    КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ для stuck parent goals
    """

    async def update_parent_progress(self, child_goal_id: str) -> Dict:
        """
        Обновляет прогресс родителя при изменении ребёнка

        Args:
            child_goal_id: ID ребенка который изменился

        Returns:
            {
                "updated": bool,
                "parent_id": str,
                "old_progress": float,
                "new_progress": float,
                "parent_status": str
            }
        """
        async with AsyncSessionLocal() as db:
            # Находим ребенка
            stmt_child = select(Goal).where(Goal.id == child_goal_id)
            result_child = await db.execute(stmt_child)
            child = result_child.scalar_one_or_none()

            if not child or not child.parent_id:
                return {
                    "updated": False,
                    "reason": "No parent"
                }

            # Находим родителя
            stmt_parent = select(Goal).where(Goal.id == child.parent_id)
            result_parent = await db.execute(stmt_parent)
            parent = result_parent.scalar_one_or_none()

            if not parent:
                return {
                    "updated": False,
                    "reason": "Parent not found"
                }

            # Считаем прогресс на основе детей
            stmt_children = select(Goal).where(Goal.parent_id == parent.id)
            result_children = await db.execute(stmt_children)
            children = result_children.scalars().all()

            if not children:
                return {
                    "updated": False,
                    "reason": "No children"
                }

            # Считаем статусы детей
            total_children = len(children)
            done_children = sum(1 for c in children if c.status == "done")
            active_children = sum(1 for c in children if c.status == "active")
            pending_children = sum(1 for c in children if c.status == "pending")

            # Вычисляем прогресс
            old_progress = parent.progress
            new_progress = done_children / total_children if total_children > 0 else 0

            # Обновляем прогресс родителя
            parent.progress = new_progress

            # Проверяем: все ли дети done?
            if done_children == total_children and total_children > 0:
                # Все дети выполнены → parent done
                if parent.goal_type in ["continuous", "directional"]:
                    # Continuous/directional не могут быть done!
                    # Ongoing/active вместо этого
                    parent.status = "ongoing" if parent.goal_type == "continuous" else "active"
                else:
                    # Achievable goals могут быть done
                    parent.status = "done"
                    parent.completed_at = datetime.now()

            elif done_children > 0:
                # Есть progress → active
                parent.status = "active"

            await db.commit()
            await db.refresh(parent)

            return {
                "updated": True,
                "parent_id": str(parent.id),
                "parent_title": parent.title[:50],
                "old_progress": old_progress,
                "new_progress": new_progress,
                "parent_status": parent.status,
                "children_stats": {
                    "total": total_children,
                    "done": done_children,
                    "active": active_children,
                    "pending": pending_children
                }
            }

    async def recalculate_all_parents(self) -> Dict:
        """
        Пересчитывает прогресс ВСЕХ родительских целей

        Использовать для emergency fix!
        """
        async with AsyncSessionLocal() as db:
            # Находим все цели с детьми
            stmt = select(Goal).where(
                Goal.id.in_(
                    select(Goal.parent_id).where(
                        Goal.parent_id.isnot(None)
                    ).distinct()
                )
            ).order_by(Goal.depth_level.asc())

            result = await db.execute(stmt)
            parents = result.scalars().all()

            report = {
                "total_parents": len(parents),
                "updated": 0,
                "completed": 0,
                "activated": 0,
                "errors": 0,
                "details": []
            }

            print(f"\n{'='*70}")
            print(f"RECALCULATING PARENT PROGRESS: {len(parents)} parents")
            print(f"{'='*70}")

            for parent in parents:
                try:
                    # Считаем детей
                    stmt_children = select(Goal).where(Goal.parent_id == parent.id)
                    result_children = await db.execute(stmt_children)
                    children = result_children.scalars().all()

                    if not children:
                        continue

                    total = len(children)
                    done = sum(1 for c in children if c.status == "done")
                    old_progress = parent.progress
                    new_progress = done / total if total > 0 else 0

                    # Обновляем
                    parent.progress = new_progress

                    status_changed = False

                    # Проверяем completion
                    if done == total and total > 0:
                        if parent.goal_type in ["continuous", "directional"]:
                            new_status = "ongoing" if parent.goal_type == "continuous" else "active"
                        else:
                            new_status = "done"
                            parent.completed_at = datetime.now()

                        if parent.status != new_status:
                            parent.status = new_status
                            status_changed = True
                            report["completed"] += 1

                    elif done > 0 and parent.status == "pending":
                        parent.status = "active"
                        status_changed = True
                        report["activated"] += 1

                    await db.commit()

                    report["updated"] += 1

                    detail = {
                        "parent_id": str(parent.id),
                        "parent_title": parent.title[:50],
                        "depth": parent.depth_level,
                        "old_progress": old_progress,
                        "new_progress": new_progress,
                        "children": f"{done}/{total}",
                        "status_changed": status_changed
                    }

                    report["details"].append(detail)

                    if abs(new_progress - old_progress) > 0.1 or status_changed:
                        print(f"\n✅ {parent.title[:40]}")
                        print(f"   Progress: {old_progress:.0%} → {new_progress:.0%}")
                        print(f"   Children: {done}/{total} done")
                        if status_changed:
                            print(f"   Status: {old_progress if not status_changed else 'changed'} → {parent.status}")

                except Exception as e:
                    report["errors"] += 1
                    print(f"\n❌ ERROR updating {parent.title[:40]}: {e}")

            return report

    async def get_stuck_parents_report(self) -> Dict:
        """
        Находит родителей с неправильным прогрессом

        Returns:
            {
                "stuck_parents": int,
                "details": [...]
            }
        """
        async with AsyncSessionLocal() as db:
            # Находим родителей с детьми
            stmt = select(Goal).where(
                Goal.id.in_(
                    select(Goal.parent_id).where(
                        Goal.parent_id.isnot(None)
                    ).distinct()
                )
            )

            result = await db.execute(stmt)
            parents = result.scalars().all()

            stuck_parents = []

            for parent in parents:
                # Считаем реальный прогресс
                stmt_children = select(Goal).where(Goal.parent_id == parent.id)
                result_children = await db.execute(stmt_children)
                children = result_children.scalars().all()

                if not children:
                    continue

                total = len(children)
                done = sum(1 for c in children if c.status == "done")
                real_progress = done / total if total > 0 else 0

                # Сравниваем с сохранённым
                if abs(real_progress - parent.progress) > 0.01:  # Отличие > 1%
                    stuck_parents.append({
                        "parent_id": str(parent.id),
                        "parent_title": parent.title[:50],
                        "stored_progress": parent.progress,
                        "real_progress": real_progress,
                        "children": f"{done}/{total}",
                        "diff": round(real_progress - parent.progress, 2)
                    })

            return {
                "stuck_parents": len(stuck_parents),
                "details": stuck_parents
            }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

parent_progress_aggregator = ParentProgressAggregator()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def update_parent(child_goal_id: str) -> Dict:
    """Update parent progress when child changes"""
    return await parent_progress_aggregator.update_parent_progress(child_goal_id)


async def recalculate_all_progress() -> Dict:
    """Emergency: recalculate all parent progress"""
    return await parent_progress_aggregator.recalculate_all_parents()


async def get_stuck_parents() -> Dict:
    """Get report of stuck parents"""
    return await parent_progress_aggregator.get_stuck_parents_report()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing Parent Progress Aggregator...\n")

        # Test 1: Get stuck parents report
        report = await parent_progress_aggregator.get_stuck_parents_report()

        print(f"Stuck parents: {report['stuck_parents']}")
        print("\nTop 10:")
        for i, parent in enumerate(report['details'][:10], 1):
            print(f"\n{i}. {parent['parent_title']}")
            print(f"   Stored: {parent['stored_progress']:.0%}")
            print(f"   Real: {parent['real_progress']:.0%}")
            print(f"   Children: {parent['children']}")
            print(f"   Diff: {parent['diff']:.2f}")

    asyncio.run(test())
