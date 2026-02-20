"""
Goal Approval Service (Phase 2.2.4)

Бизнес-логика для ручного подтверждения завершения MANUAL целей.

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from datetime import datetime
from typing import Dict
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from database import AsyncSessionLocal
from models import Goal, GoalCompletionApproval
from exceptions import (
    GoalAlreadyDone,
    GoalChildrenIncomplete,
    InsufficientAuthority,
    InvalidGoalState,
    InvalidCompletionMode,
    InvariantViolation,
)


class ApproveCompletionService:
    """
    Сервис для approval MANUAL целей.

    Архитектурный контракт:
    - Вся бизнес-логика здесь
    - Controller = thin wrapper
    - Транзакционная граница в service
    - Доменные исключения для всех ошибок
    """

    async def approve_goal_completion(
        self,
        goal_id: uuid.UUID,
        approved_by: str,
        authority_level: int,
        comment: str = None
    ) -> Dict:
        """
        Подтвердить завершение MANUAL цели.

        Алгоритм (строгий порядок):
        1. SELECT goal FOR UPDATE → 404 если нет
        2. IF goal.status = done → 409 GOAL_ALREADY_DONE
        3. IF goal.state NOT IN completable → 409 GOAL_STATE_INVALID
        4. IF goal has children AND any NOT done → 409 CHILDREN_INCOMPLETE
        5. IF authority_level < required → 403 INSUFFICIENT_AUTHORITY
        6. INSERT INTO goal_completion_approvals (UNIQUE guard)
        7. UPDATE goals SET status = done

        Все шаги в одной транзакции.

        Args:
            goal_id: UUID цели
            approved_by: Кто подтверждает (user:<id> | system | dao:<id>)
            authority_level: Уровень полномочий (1-4)
            comment: Опциональный комментарий

        Returns:
            {
                "goal_id": "uuid",
                "status": "done",
                "approved_at": "2026-02-06T14:32:11Z",
                "approved_by": "string",
                "authority_level": 2
            }

        Raises:
            InvalidGoalState: 409
            GoalAlreadyDone: 409
            InvalidCompletionMode: 400
            GoalChildrenIncomplete: 409
            InsufficientAuthority: 403
            InvariantViolation: 500
        """
        async with AsyncSessionLocal() as db:
            try:
                # =================================================================
                # Step 1: SELECT goal FOR UPDATE (row-level lock)
                # =================================================================
                stmt = select(Goal).where(Goal.id == goal_id).with_for_update()
                result = await db.execute(stmt)
                goal = result.scalar_one_or_none()

                if not goal:
                    # Используем BaseGoalException или создаем NotFound
                    from exceptions import BaseGoalException
                    raise BaseGoalException(
                        message="Goal does not exist",
                        details={"goal_id": str(goal_id)}
                    )

                # =================================================================
                # Step 2: Check if already done
                # =================================================================
                if goal.status == "done":
                    # Проверяем наличие approval (I7 check)
                    approval_stmt = select(GoalCompletionApproval).where(
                        GoalCompletionApproval.goal_id == goal.id
                    )
                    existing_approval = (await db.execute(approval_stmt)).scalar_one_or_none()

                    if existing_approval:
                        raise GoalAlreadyDone(
                            goal_id=str(goal.id),
                            approved_at=existing_approval.approved_at.isoformat()
                        )
                    else:
                        # DONE без approval = I7 violation (критическая ошибка)
                        raise InvariantViolation(
                            invariant="I7",
                            goal_id=str(goal.id)
                        )

                # =================================================================
                # Step 3: Check completable state
                # =================================================================
                completable_states = ["active"]
                if goal.status not in completable_states:
                    raise InvalidGoalState(
                        goal_id=str(goal.id),
                        current_state=goal.status,
                        expected_states=completable_states
                    )

                # =================================================================
                # Step 4: Check completion mode
                # =================================================================
                if goal.completion_mode != "manual":
                    raise InvalidCompletionMode(
                        goal_id=str(goal.id),
                        completion_mode=goal.completion_mode
                    )

                # =================================================================
                # Step 5: Check children (if parent)
                # =================================================================
                if not goal.is_atomic:
                    # Ищем незавершённых детей
                    children_stmt = select(Goal).where(
                        Goal.parent_id == goal.id
                    ).where(
                        Goal.status.not_in(["done", "completed"])
                    )
                    incomplete_children_result = await db.execute(children_stmt)
                    incomplete_children = incomplete_children_result.scalars().all()

                    if incomplete_children:
                        raise GoalChildrenIncomplete(
                            goal_id=str(goal.id),
                            remaining_children=len(incomplete_children)
                        )

                # =================================================================
                # Step 6: Check authority (simplified for v1.0)
                # =================================================================
                # TODO: В будущих версиях здесь может быть проверка
                # goal.required_authority vs provided authority_level
                # Сейчас просто логируем
                if authority_level < 1:
                    raise InsufficientAuthority(
                        required_level=1,
                        provided_level=authority_level
                    )

                # =================================================================
                # Step 7: INSERT approval (UNIQUE constraint guard)
                # =================================================================
                # Используем raw SQL чтобы избежать SQLAlchemy concurrency issues
                from sqlalchemy import text

                insert_stmt = text("""
                    INSERT INTO goal_completion_approvals (goal_id, approved_by, authority_level, comment, approved_at)
                    VALUES (:goal_id, :approved_by, :authority_level, :comment, :approved_at)
                """)

                await db.execute(insert_stmt, {
                    "goal_id": goal.id,
                    "approved_by": approved_by,
                    "authority_level": authority_level,
                    "comment": comment,
                    "approved_at": datetime.now()
                })

                # =================================================================
                # Step 8: UPDATE goal status
                # =================================================================
                goal.status = "done"
                goal.progress = 1.0
                goal.completed_at = datetime.now()

                # =================================================================
                # Commit (atomic transaction)
                # =================================================================
                await db.commit()

                # Get approval for response
                approval_stmt = select(GoalCompletionApproval).where(
                    GoalCompletionApproval.goal_id == goal.id
                )
                approval = (await db.execute(approval_stmt)).scalar_one()

                return {
                    "goal_id": str(goal.id),
                    "status": goal.status,
                    "approved_at": approval.approved_at.isoformat(),
                    "approved_by": approved_by,
                    "authority_level": authority_level
                }

            except IntegrityError as e:
                # UNIQUE constraint violated → double approve
                await db.rollback()

                # Проверяем, что это именно duplicate approval
                approval_stmt = select(GoalCompletionApproval).where(
                    GoalCompletionApproval.goal_id == goal_id
                )
                existing_approval = (await db.execute(approval_stmt)).scalar_one_or_none()

                if existing_approval:
                    raise GoalAlreadyDone(
                        goal_id=str(goal_id),
                        approved_at=existing_approval.approved_at.isoformat()
                    )
                else:
                    # Какая-то другая integrity error
                    from exceptions import BaseGoalException
                    raise BaseGoalException(
                        message="Database integrity error",
                        details={"original_error": str(e)}
                    )


# Singleton
approve_completion_service = ApproveCompletionService()
