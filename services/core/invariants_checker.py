"""
GOAL STATE-MACHINE INVARIANTS CHECKER v1.0

Проверяет соблюдение инвариантов state-machine для целей.
Запускается как nightly job или manually.

Инварианты:
1. is_atomic == false AND child_count > 0 → status != pending
2. parent.status != pending → EXISTS children OR is_atomic
3. parent.done → all children done
"""

import uuid
from typing import List, Dict
from datetime import datetime
from sqlalchemy import select, func, cast, String
from database import AsyncSessionLocal
from models import Goal, GoalCompletionApproval


class InvariantsChecker:
    """Проверка инвариантов state-machine для целей"""

    async def check_all_invariants(self) -> Dict:
        """
        Проверяет все инварианты и возвращает отчёт

        Returns:
            {
                "overall_status": "PASS" | "VIOLATION" | "ERROR",
                "invariant_checks": [...],
                "summary": {...}
            }
        """
        checks = [
            await self._check_no_pending_parents_with_children(),
            await self._check_no_active_parents_without_children(),
            await self._check_parent_done_implies_children_done(),
            # 🔒 GOAL LIFECYCLE v1.1 - Completion Mode Invariants
            await self._check_aggregate_parent_completion(),
            await self._check_manual_parent_not_auto_completed(),
            await self._check_atomic_goals_aggregate_mode(),
            # 🔒 GOAL LIFECYCLE v1.1.1 - Manual Completion Layer
            await self._check_manual_completion_has_approval(),
        ]

        violations = [c for c in checks if c["status"] == "VIOLATION"]
        errors = [c for c in checks if c["status"] == "ERROR"]

        overall_status = "PASS"
        if errors:
            overall_status = "ERROR"
        elif violations:
            overall_status = "VIOLATION"

        return {
            "overall_status": overall_status,
            "invariant_checks": checks,
            "summary": {
                "total_checks": len(checks),
                "passed": sum(1 for c in checks if c["status"] == "PASS"),
                "violations": len(violations),
                "errors": len(errors),
                "checked_at": datetime.now().isoformat()
            }
        }

    async def _check_no_pending_parents_with_children(self) -> Dict:
        """
        🔒 Инвариант #1: Non-atomic goal with children → status != pending

        pending используется ТОЛЬКО для целей без декомпозиции.
        """
        try:
            async with AsyncSessionLocal() as db:
                # EXISTS подзапрос для поиска детей
                child_exists = select(Goal.id).where(
                    Goal.parent_id == Goal.id
                ).exists()

                stmt = select(Goal).where(
                    Goal.is_atomic == False
                ).where(
                    Goal.status == 'pending'
                ).where(
                    child_exists  # Есть подцели!
                )

                violating_goals = (await db.execute(stmt)).scalars().all()

                if violating_goals:
                    return {
                        "invariant": "NO_PENDING_PARENTS_WITH_CHILDREN",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_goals)} pending non-atomic goals with children",
                        "violations": [
                            {
                                "goal_id": str(g.id),
                                "title": g.title,
                                "status": g.status,
                                "child_count": self._get_child_count(g.id)
                            }
                            for g in violating_goals[:10]  # Первые 10
                        ]
                    }

                return {
                    "invariant": "NO_PENDING_PARENTS_WITH_CHILDREN",
                    "status": "PASS",
                    "message": "All non-atomic goals with children have status != pending"
                }

        except Exception as e:
            return {
                "invariant": "NO_PENDING_PARENTS_WITH_CHILDREN",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _check_no_active_parents_without_children(self) -> Dict:
        """
        🔒 Инвариант #2: Non-atomic active goal → has children

        Если цель decomposed, она должна иметь подцели.
        """
        try:
            async with AsyncSessionLocal() as db:
                # NOT EXISTS подзапрос для поиска целей без детей
                child_exists = select(Goal.id).where(
                    Goal.parent_id == Goal.id
                ).exists()

                stmt = select(Goal).where(
                    Goal.is_atomic == False
                ).where(
                    Goal.status == 'active'
                ).where(
                    ~child_exists  # Нет подцелей!
                )

                violating_goals = (await db.execute(stmt)).scalars().all()

                if violating_goals:
                    return {
                        "invariant": "NO_ACTIVE_PARENTS_WITHOUT_CHILDREN",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_goals)} active non-atomic goals without children",
                        "violations": [
                            {
                                "goal_id": str(g.id),
                                "title": g.title,
                                "status": g.status,
                                "depth_level": g.depth_level
                            }
                            for g in violating_goals[:10]
                        ]
                    }

                return {
                    "invariant": "NO_ACTIVE_PARENTS_WITHOUT_CHILDREN",
                    "status": "PASS",
                    "message": "All active non-atomic goals have children"
                }

        except Exception as e:
            return {
                "invariant": "NO_ACTIVE_PARENTS_WITHOUT_CHILDREN",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _check_parent_done_implies_children_done(self) -> Dict:
        """
        🔒 Инвариант #3: parent.done → all children done

        Если родитель выполнен, все дети должны быть выполнены.
        """
        try:
            async with AsyncSessionLocal() as db:
                # Находим done родителей
                stmt = select(Goal).where(
                    Goal.status == 'done'
                ).where(
                    Goal.is_atomic == False
                )

                done_parents = (await db.execute(stmt)).scalars().all()

                violating_parents = []
                for parent in done_parents:
                    # Проверяем детей
                    child_stmt = select(Goal).where(Goal.parent_id == parent.id)
                    children = (await db.execute(child_stmt)).scalars().all()

                    if not children:
                        continue  # Нет детей - ок

                    # Есть ли незавершенные дети?
                    incomplete_children = [
                        c for c in children
                        if c.status not in ['done', 'completed']
                    ]

                    if incomplete_children:
                        violating_parents.append({
                            "goal_id": str(parent.id),
                            "title": parent.title,
                            "status": parent.status,
                            "incomplete_children": len(incomplete_children),
                            "total_children": len(children)
                        })

                if violating_parents:
                    return {
                        "invariant": "PARENT_DONE_IMPLIES_CHILDREN_DONE",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_parents)} done parents with incomplete children",
                        "violations": violating_parents[:10]
                    }

                return {
                    "invariant": "PARENT_DONE_IMPLIES_CHILDREN_DONE",
                    "status": "PASS",
                    "message": "All done parents have all children done"
                }

        except Exception as e:
            return {
                "invariant": "PARENT_DONE_IMPLIES_CHILDREN_DONE",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _check_aggregate_parent_completion(self) -> Dict:
        """
        🔒 Инвариант I1 (v1.1): completion_mode=aggregate AND all children done → parent done

        Если у AGGREGATE родителя все дети выполнены, он сам должен быть done.
        """
        try:
            async with AsyncSessionLocal() as db:
                # Находим aggregate родителей с детьми
                stmt = select(Goal).where(
                    Goal.completion_mode == 'aggregate'
                ).where(
                    Goal.is_atomic == False
                )

                aggregate_parents = (await db.execute(stmt)).scalars().all()

                violating_parents = []
                for parent in aggregate_parents:
                    # Проверяем детей
                    child_stmt = select(Goal).where(Goal.parent_id == parent.id)
                    children = (await db.execute(child_stmt)).scalars().all()

                    if not children:
                        continue  # Нет детей - ок

                    # Все ли дети done?
                    all_done = all(
                        child.status in ['done', 'completed']
                        for child in children
                    )

                    # Нарушение: все дети done, но родитель не done
                    if all_done and parent.status not in ['done', 'completed']:
                        violating_parents.append({
                            "goal_id": str(parent.id),
                            "title": parent.title,
                            "status": parent.status,
                            "completion_mode": parent.completion_mode,  # String from DB
                            "children_count": len(children),
                            "all_children_done": True
                        })

                if violating_parents:
                    return {
                        "invariant": "AGGREGATE_PARENT_COMPLETION",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_parents)} aggregate parents with all children done but not done themselves",
                        "violations": violating_parents[:10]
                    }

                return {
                    "invariant": "AGGREGATE_PARENT_COMPLETION",
                    "status": "PASS",
                    "message": "All aggregate parents with completed children are done"
                }

        except Exception as e:
            return {
                "invariant": "AGGREGATE_PARENT_COMPLETION",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _check_manual_parent_not_auto_completed(self) -> Dict:
        """
        🔒 Инвариант I2 (v1.1): completion_mode=manual → parent NEVER auto-done

        MANUAL цели могут быть завершены только вручную (explicit action).
        """
        try:
            async with AsyncSessionLocal() as db:
                # Находим manual done родителей с детьми
                stmt = select(Goal).where(
                    Goal.completion_mode == 'manual'
                ).where(
                    Goal.status.in_(['done', 'completed'])
                ).where(
                    Goal.is_atomic == False
                )

                manual_done_parents = (await db.execute(stmt)).scalars().all()

                violating_parents = []
                for parent in manual_done_parents:
                    # Проверяем: есть ли дети?
                    child_stmt = select(Goal).where(Goal.parent_id == parent.id)
                    children = (await db.execute(child_stmt)).scalars().all()

                    if not children:
                        continue  # Нет детей - не нарушение

                    # ⚠️ Временное упрощение: не проверяем completed_by_system
                    # В будущей версии добавим флаг "completed_by: system|human"
                    violating_parents.append({
                        "goal_id": str(parent.id),
                        "title": parent.title,
                        "completion_mode": parent.completion_mode,  # String from DB
                        "children_count": len(children),
                        "note": "Manual parent with children should be verified for manual completion"
                    })

                if violating_parents:
                    return {
                        "invariant": "MANUAL_PARENT_NOT_AUTO_COMPLETED",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_parents)} manual parents done with children (verify manual approval)",
                        "violations": violating_parents[:10]
                    }

                return {
                    "invariant": "MANUAL_PARENT_NOT_AUTO_COMPLETED",
                    "status": "PASS",
                    "message": "No manual parents with children auto-completed"
                }

        except Exception as e:
            return {
                "invariant": "MANUAL_PARENT_NOT_AUTO_COMPLETED",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _check_atomic_goals_aggregate_mode(self) -> Dict:
        """
        🔒 Инвариант I3 (v1.1): is_atomic=true → completion_mode MUST be aggregate

        Atomic goals не имеют детей, поэтому completion_mode=aggregate - единственный корректный режим.
        """
        try:
            async with AsyncSessionLocal() as db:
                # Находим atomic цели не в aggregate режиме
                stmt = select(Goal).where(
                    Goal.is_atomic == True
                ).where(
                    Goal.completion_mode != 'aggregate'
                )

                violating_goals = (await db.execute(stmt)).scalars().all()

                if violating_goals:
                    return {
                        "invariant": "ATOMIC_GOALS_AGGREGATE_MODE",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_goals)} atomic goals with non-aggregate completion mode",
                        "violations": [
                            {
                                "goal_id": str(g.id),
                                "title": g.title,
                                "is_atomic": True,
                                "completion_mode": g.completion_mode  # String from DB
                            }
                            for g in violating_goals[:10]
                        ]
                    }

                return {
                    "invariant": "ATOMIC_GOALS_AGGREGATE_MODE",
                    "status": "PASS",
                    "message": "All atomic goals have aggregate completion mode"
                }

        except Exception as e:
            return {
                "invariant": "ATOMIC_GOALS_AGGREGATE_MODE",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _check_manual_completion_has_approval(self) -> Dict:
        """
        🔒 Инвариант I7 (v1.1.1): MANUAL completion requires explicit approval

        MANUAL goal НЕ МОЖЕТ быть done без approval.

        Формально:
        goal.completion_mode == MANUAL AND goal.status == done
        ⇒ EXISTS goal_completion_approval(goal_id)

        Следствие:
        AGGREGATE goals MAY be done без approval (система auto-completes).
        """
        try:
            async with AsyncSessionLocal() as db:
                # Находим все MANUAL goals в done/completed
                stmt = select(Goal).where(
                    Goal.completion_mode == 'manual'
                ).where(
                    Goal.status.in_(['done', 'completed'])
                )

                manual_done_goals = (await db.execute(stmt)).scalars().all()

                violating_goals = []
                for goal in manual_done_goals:
                    # Проверяем наличие approval
                    approval_stmt = select(GoalCompletionApproval).where(
                        GoalCompletionApproval.goal_id == goal.id
                    )
                    approval = (await db.execute(approval_stmt)).scalar_one_or_none()

                    if not approval:
                        # VIOLATION: MANUAL done без approval
                        violating_goals.append({
                            "goal_id": str(goal.id),
                            "title": goal.title,
                            "completion_mode": goal.completion_mode,
                            "status": goal.status,
                            "completed_at": goal.completed_at.isoformat() if goal.completed_at else None
                        })

                if violating_goals:
                    return {
                        "invariant": "MANUAL_COMPLETION_HAS_APPROVAL",
                        "status": "VIOLATION",
                        "message": f"Found {len(violating_goals)} MANUAL done goals without approval",
                        "violations": violating_goals[:10]
                    }

                return {
                    "invariant": "MANUAL_COMPLETION_HAS_APPROVAL",
                    "status": "PASS",
                    "message": "All MANUAL done goals have approval records"
                }

        except Exception as e:
            return {
                "invariant": "MANUAL_COMPLETION_HAS_APPROVAL",
                "status": "ERROR",
                "message": f"Error checking invariant: {str(e)}"
            }

    async def _get_child_count(self, goal_id: uuid.UUID) -> int:
        """Подсчитывает количество подцелей"""
        async with AsyncSessionLocal() as db:
            subquery = select(func.count(Goal.id)).where(Goal.parent_id == goal_id)
            result = await db.execute(subquery)
            return result.scalar() or 0


# Singleton
invariants_checker = InvariantsChecker()


async def run_invariants_check() -> Dict:
    """
    Запускает проверку всех инвариантов.
    Используется в scheduler или manually.
    """
    return await invariants_checker.check_all_invariants()


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(run_invariants_check())
    logger.info(f"🔍 Invariants Check Result: {result['overall_status']}")
    logger.info(f"   Summary: {result['summary']}")

    for check in result['invariant_checks']:
        if check['status'] != 'PASS':
            logger.info(f"\n⚠️  {check['invariant']}: {check['status']}")
            logger.info(f"   {check['message']}")
