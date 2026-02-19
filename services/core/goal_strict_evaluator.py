"""
GOAL STRICT EVALUATOR - v3.0
Строгая проверка выполнения целей (binary/scalar/trend)
Только проверка факта, без анализа причин

ARCHITECTURE v3.0:
- Uses UnitOfWork pattern for transaction management
"""
import uuid
from typing import Dict, Optional
from datetime import datetime
from uuid import UUID
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal
from agent_graph import app_graph
from goal_contract_validator import goal_contract_validator
from infrastructure.uow import UnitOfWork, GoalRepository
from goal_transition_service import transition_service


class GoalStrictEvaluator:
    """
    Строгий оценщик целей - проверяет факт выполнения

    Ответственность:
    - Проверить выполнена ли цель (binary)
    - Оценить степень выполнения (scalar)
    - Проверить тренд улучшения (trend)

    НЕ отвечает за:
    - Анализ причин (это делает Reflector)
    - Генерацию следующих целей (это делает Reflector)
    """

    async def _check_and_complete_parent(self, goal_id: uuid.UUID) -> None:
        """
        DEPRECATED: Use _check_and_complete_parent_with_uow() instead.

        Parent completion с учётом completion_mode.
        """
        from infrastructure.uow import create_uow_provider

        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
            await self._check_and_complete_parent_with_uow(uow, goal_id)

    async def _check_and_complete_parent_with_uow(self, uow: UnitOfWork, goal_id: UUID) -> None:
        """
        🔒 GOAL LIFECYCLE v3.0: Parent completion within existing transaction.

        ARCHITECTURE v3.0: Transaction managed by caller via UnitOfWork.

        Инварианты:
        - completion_mode=AGGREGATE AND all children done → parent done
        - completion_mode=MANUAL → parent NEVER auto-done
        - completion_mode=STRICT → custom evaluator (TODO)
        """
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await uow.session.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal or not goal.parent_id:
            return

        parent_stmt = select(Goal).where(Goal.id == goal.parent_id)
        parent_result = await uow.session.execute(parent_stmt)
        parent = parent_result.scalar_one_or_none()

        if not parent or parent.is_atomic:
            return

        if parent.completion_mode == 'manual':
            return

        if parent.completion_mode == 'aggregate':
            children_stmt = select(Goal).where(Goal.parent_id == parent.id)
            children_result = await uow.session.execute(children_stmt)
            children = children_result.scalars().all()

            if not children:
                return

            all_done = all(
                child.status in ["done", "completed"]
                for child in children
            )

            if all_done:
                parent.status = "done"
                parent.progress = 1.0
                parent.completed_at = datetime.now()

        if parent.completion_mode == 'strict':
            return


    async def evaluate_goal(self, goal_id: str) -> Dict:
        """
        Оценивает выполнение цели строго по критериям

        Returns:
            {
                "passed": true/false,
                "score": 0.0-1.0,
                "trend": "improving|stable|degrading",  # для continuous
                "evaluation_mode": "binary|scalar|trend",
                "strict_result": {...}  // Сырые данные проверки
            }
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

            # 🔑 GOAL CONTRACT v3.0 - Определяем режим оценки
            evaluation_mode = goal_contract_validator.get_evaluation_mode(goal)

            # Проверяем can_execute_action("evaluate")
            can_eval, reason = goal_contract_validator.can_execute_action(goal, "evaluate")
            if not can_eval:
                return {
                    "passed": False,
                    "score": 0.0,
                    "evaluation_mode": evaluation_mode,
                    "error": f"Evaluation forbidden: {reason}"
                }

            # Строгая проверка по режиму
            if evaluation_mode == "binary":
                return await self._evaluate_binary(goal)
            elif evaluation_mode == "scalar":
                return await self._evaluate_scalar(goal)
            elif evaluation_mode == "trend":
                return await self._evaluate_trend(goal)
            else:
                return await self._evaluate_binary(goal)

    async def _evaluate_binary(self, goal: Goal) -> Dict:
        """
        Бинарная оценка: выполнена/не выполнена

        Используется для achievable целей с четким критерием завершения
        """
        eval_prompt = f"""Строго оцени: ВЫПОЛНЕНА ли эта цель?

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
КРИТЕРИИ УСПЕХА: {goal.success_definition or 'Не определены'}
ТЕКУЩИЙ ПРОГРЕСС: {int(goal.progress * 100)}%

Верни ТОЛЬКО JSON:
{{
    "passed": true/false,
    "confidence": 0.0-1.0,
    "evidence": ["Факт 1", "Факт 2"]
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=eval_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(result)

            passed = evaluation.get("passed", False)
            confidence = evaluation.get("confidence", 0.5)

            # Сохраняем результат оценки
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()
                if g:
                    g.evaluation_result = {
                        "mode": "binary",
                        **evaluation
                    }
                    if passed:
                        g.status = "done"
                        g.progress = 1.0
                        g.completed_at = datetime.now()
                    await db.commit()

                    # 🔒 STATE-MACHINE: Check if parent should be completed
                    if passed:
                        await self._check_and_complete_parent(goal.id)

            return {
                "passed": passed,
                "score": 1.0 if passed else 0.0,
                "confidence": confidence,
                "evaluation_mode": "binary",
                "strict_result": evaluation
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "confidence": 0.0,
                "evaluation_mode": "binary",
                "error": str(e)
            }

    async def _evaluate_scalar(self, goal: Goal) -> Dict:
        """
        Скалярная оценка: степень выполнения 0.0-1.0

        Используется для meta целей, directional целей
        """
        eval_prompt = f"""Оцени степень выполнения этой цели по шкале 0.0-1.0:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
КРИТЕРИИ: {goal.completion_criteria or 'Не определены'}
ТЕКУЩИЙ ПРОГРЕСС: {int(goal.progress * 100)}%

Верни ТОЛЬКО JSON:
{{
    "score": 0.0-1.0,
    "evidence": ["Факт 1", "Факт 2"],
    "gaps": ["Что не выполнено"]
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=eval_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(result)

            score = evaluation.get("score", 0.0)
            passed = score >= 0.7  # Порог для скалярной оценки

            # Сохраняем результат
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()
                if g:
                    g.evaluation_result = {
                        "mode": "scalar",
                        **evaluation
                    }
                    g.progress = score
                    if passed:
                        g.status = "done"
                        g.completed_at = datetime.now()
                    await db.commit()

                    # 🔒 STATE-MACHINE: Check if parent should be completed
                    if passed:
                        await self._check_and_complete_parent(goal.id)

            return {
                "passed": passed,
                "score": score,
                "evaluation_mode": "scalar",
                "strict_result": evaluation
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "evaluation_mode": "scalar",
                "error": str(e)
            }

    async def _evaluate_trend(self, goal: Goal) -> Dict:
        """
        Оценка тренда: improving/stable/degrading

        Используется для continuous целей
        """
        eval_prompt = f"""Оцени ТРЕНД выполнения этой непрерывной цели:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТЕКУЩИЙ ПРОГРЕСС: {int(goal.progress * 100)}%

Верни ТОЛЬКО JSON:
{{
    "trend": "improving|stable|degrading",
    "score": 0.0-1.0,
    "evidence": ["Факт 1", "Факт 2"]
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=eval_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            evaluation = json.loads(result)

            trend = evaluation.get("trend", "stable")
            score = evaluation.get("score", 0.5)

            # Continuous цели никогда не завершаются
            passed = trend == "improving"

            status_map = {
                "improving": "improving",
                "stable": "active",
                "degrading": "blocked"
            }

            # Сохраняем результат
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()
                if g:
                    g.evaluation_result = {
                        "mode": "trend",
                        **evaluation
                    }
                    g.status = status_map.get(trend, "active")
                    await db.commit()

            return {
                "passed": passed,
                "score": score,
                "trend": trend,
                "evaluation_mode": "trend",
                "strict_result": evaluation
            }

        except Exception as e:
            return {
                "passed": False,
                "trend": "stable",
                "score": 0.0,
                "evaluation_mode": "trend",
                "error": str(e)
            }

    # ============= UoW MIGRATION: Новые атомарные методы =============

    async def evaluate_goal_with_uow(
        self,
        uow: UnitOfWork,
        goal_id: str
    ) -> Dict:
        """
        Строго оценивает цель ВНУТРИ существующей UoW транзакции.

        UoW MIGRATION: Атомарная операция - оценка + state transition в одной транзакции.

        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели

        Returns:
            Dict: Результат строгой оценки
        """
        goal_uuid = UUID(goal_id)
        repo = GoalRepository(uow)

        # Получаем goal с pessimistic lock
        goal = await repo.get_for_update(uow.session, goal_uuid)

        if not goal:
            return {"error": "Goal not found"}

        # Определяем evaluation mode
        evaluation_mode = self._determine_evaluation_mode(goal)

        if evaluation_mode == "binary":
            return await self._evaluate_binary_with_uow(uow, goal)
        elif evaluation_mode == "scalar":
            return await self._evaluate_scalar_with_uow(uow, goal)
        elif evaluation_mode == "trend":
            return await self._evaluate_trend_with_uow(uow, goal)
        else:
            return await self._evaluate_binary_with_uow(uow, goal)

    async def _evaluate_binary_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Binary evaluation через UoW"""
        # Проверяем artifacts
        artifacts_check = await self._check_artifacts_with_uow(uow, goal)

        if artifacts_check["passed"]:
            # Transition в done
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="done",
                reason=f"Binary evaluation passed: {artifacts_check['details']}",
                actor="goal_strict_evaluator"
            )

        return {
            "passed": artifacts_check["passed"],
            "evaluation_mode": "binary",
            "checks": artifacts_check["checks"],
            "goal_id": str(goal.id)
        }

    async def _evaluate_scalar_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Scalar evaluation через UoW"""
        # Проверяем progress
        score = goal.progress or 0.0
        passed = score >= 0.8  # 80% threshold

        if passed:
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="done",
                reason=f"Scalar evaluation passed: {score:.2f} >= 0.80",
                actor="goal_strict_evaluator"
            )

        return {
            "passed": passed,
            "score": score,
            "threshold": 0.8,
            "evaluation_mode": "scalar",
            "goal_id": str(goal.id)
        }

    async def _evaluate_trend_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Trend evaluation через UoW"""
        # Анализируем trend из evaluation_result
        trend = "stable"
        if goal.evaluation_result and isinstance(goal.evaluation_result, dict):
            trend = goal.evaluation_result.get("trend", "stable")

        # Trend goals не завершаются, но могут блокироваться при деградации
        if trend == "degrading":
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="blocked",
                reason=f"Trend evaluation: performance degrading",
                actor="goal_strict_evaluator"
            )

        return {
            "passed": trend in ["improving", "stable"],
            "trend": trend,
            "evaluation_mode": "trend",
            "goal_id": str(goal.id)
        }

    async def _check_artifacts_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Проверяет artifacts через UoW"""
        from artifact_registry import artifact_registry

        try:
            # Получаем artifacts через UoW-совместимый метод
            artifacts = await artifact_registry.list_by_goal(str(goal.id), None)

            if not artifacts:
                return {
                    "passed": False,
                    "details": "No artifacts found",
                    "checks": {"artifacts_exist": False}
                }

            passed_artifacts = [a for a in artifacts if a.get("verification_status") == "passed"]

            return {
                "passed": len(passed_artifacts) > 0,
                "details": f"{len(passed_artifacts)}/{len(artifacts)} artifacts passed",
                "checks": {
                    "artifacts_exist": len(artifacts) > 0,
                    "artifacts_passed": len(passed_artifacts) > 0
                }
            }

        except Exception as e:
            return {
                "passed": False,
                "details": f"Error checking artifacts: {str(e)}",
                "checks": {"error": str(e)}
            }

    def _determine_evaluation_mode(self, goal: Goal) -> str:
        """Определяет режим оценки для цели"""
        if goal.goal_type == "continuous":
            return "trend"
        elif goal.goal_type == "achievable":
            return "binary" if goal.is_atomic else "scalar"
        elif goal.goal_type == "exploratory":
            return "scalar"
        else:
            return "binary"


# Глобальный экземпляр
goal_strict_evaluator = GoalStrictEvaluator()
