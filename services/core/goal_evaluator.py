"""
GOAL EVALUATOR - Self-Evaluation Layer
Проверяет выполнение целей и генерирует следующие цели

UoW MIGRATION: Evaluation теперь атомарна - все операции в одной транзакции.
"""
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
from sqlalchemy import select, and_
from database import AsyncSessionLocal
from models import Goal
from agent_graph import app_graph

# UoW imports для новой архитектуры
from infrastructure.uow import UnitOfWork, GoalRepository
from goal_transition_service import transition_service


TELEGRAM_URL = os.getenv("TELEGRAM_URL", "http://telegram:8004")


class GoalEvaluator:
    """Оценщик выполнения целей - Self-Evaluation Layer"""

    def __init__(self):
        self.evaluation_history = {}

    async def evaluate_goal(self, goal_id: str) -> Dict:
        """
        Оценивает выполнение цели

        Returns:
            {
                "passed": true/false,
                "score": 0.0-1.0,
                "reasoning": "...",
                "next_goals": [...],
                "action": "complete|continue|adjust"
            }
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

            # Разная логика для разных типов целей
            if goal.goal_type == "achievable":
                return await self._evaluate_achievable(goal)
            elif goal.goal_type == "continuous":
                return await self._evaluate_continuous(goal)
            elif goal.goal_type == "exploratory":
                return await self._evaluate_exploratory(goal)
            elif goal.goal_type == "directional":
                return await self._evaluate_directional(goal)
            elif goal.goal_type == "meta":
                return await self._evaluate_meta(goal)
            else:
                return await self._evaluate_achievable(goal)

    async def _evaluate_achievable(self, goal: Goal) -> Dict:
        """Оценка выполнимой цели"""
        eval_prompt = f"""Оцени выполнение этой цели:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
КРИТЕРИИ УСПЕХА: {goal.success_definition or 'Не определены'}

Проанализируй:
1. Была ли достигнута цель?
2. Какой процент выполнения?
3. Что нужно улучшить?

Верни JSON:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "reasoning": "Обоснование",
    "gaps": ["Что не выполнено"],
    "improvements": ["Что можно улучшить"]
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
            score = evaluation.get("score", 0.0)

            action = "complete" if passed else "continue"

            # Сохраняем результат
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()
                if g:
                    g.evaluation_result = evaluation
                    if passed:
                        g.status = "done"
                        g.progress = 1.0
                        g.completed_at = datetime.now()
                    await db.commit()

            # Генерируем следующие цели если нужно
            next_goals = []
            if not passed and evaluation.get("gaps"):
                next_goals = await self._generate_improvement_goals(goal, evaluation["gaps"])

            return {
                "passed": passed,
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "next_goals": next_goals,
                "action": action
            }

        except Exception as e:
            logger.info(f"❌ Evaluation error: {e}")
            return {
                "passed": False,
                "score": 0.0,
                "reasoning": f"Ошибка оценки: {e}",
                "action": "continue"
            }

    async def _evaluate_continuous(self, goal: Goal) -> Dict:
        """Оценка непрерывной цели (улучшение)"""
        # Для continuous целей проверяем тренд
        eval_prompt = f"""Оцени улучшение по этой непрерывной цели:

ЦЕЛЬ: {goal.title}
ТЕКУЩИЙ ПРОГРЕСС: {int(goal.progress * 100)}%

Верни JSON:
{{
    "trend": "improving|stable|degrading",
    "score": 0.0-1.0,
    "reasoning": "Обоснование тренда",
    "recommendations": ["Что улучшить"]
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
            action = {
                "improving": "continue",
                "stable": "continue",
                "degrading": "adjust"
            }.get(trend, "continue")

            status_map = {
                "improving": "improving",
                "stable": "active",
                "degrading": "blocked"
            }

            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()
                if g:
                    g.status = status_map.get(trend, "active")
                    g.evaluation_result = evaluation
                    await db.commit()

            return {
                "passed": trend == "improving",
                "score": score,
                "trend": trend,
                "reasoning": evaluation.get("reasoning", ""),
                "action": action
            }

        except Exception as e:
            return {
                "passed": False,
                "trend": "stable",
                "reasoning": f"Ошибка: {e}",
                "action": "continue"
            }

    async def _evaluate_exploratory(self, goal: Goal) -> Dict:
        """Оценка исследовательской цели"""
        # Проверяем полноту исследования
        return await self._evaluate_achievable(goal)

    async def _evaluate_directional(self, goal: Goal) -> Dict:
        """Оценка векторной цели (не завершается)"""
        return {
            "passed": True,  # Векторные цели всегда "passed" - они направляют
            "score": 1.0,
            "reasoning": "Directional goal guides system decisions",
            "action": "continue"  # Никогда не завершается
        }

    async def _evaluate_meta(self, goal: Goal) -> Dict:
        """Оценка мета-цели"""
        return await self._evaluate_achievable(goal)

    async def _generate_improvement_goals(self, parent_goal: Goal, gaps: List[str]) -> List[Dict]:
        """Генерирует цели для улучшения на основе пробелов"""

        improvement_prompt = f"""На основе пробелов в выполнении цели сгенерируйте новые подцели:

ИСХОДНАЯ ЦЕЛЬ: {parent_goal.title}
ПРОБЕЛЫ: {gaps}

Сгенерируй 1-3 подцели для устранения пробелов.

Верни JSON:
{{
    "improvement_goals": [
        {{
            "title": "Название",
            "description": "Описание",
            "goal_type": "achievable"
        }}
    ]
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=improvement_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            data = json.loads(result)

            created_goals = []
            for goal_data in data.get("improvement_goals", []):
                async with AsyncSessionLocal() as db:
                    new_goal = Goal(
                        parent_id=parent_goal.id,
                        title=goal_data["title"],
                        description=goal_data.get("description", ""),
                        goal_type=goal_data.get("goal_type", "achievable"),
                        depth_level=parent_goal.depth_level + 1,
                        status="pending",
                        progress=0.0
                    )
                    db.add(new_goal)
                    await db.commit()
                    await db.refresh(new_goal)

                    created_goals.append({
                        "id": str(new_goal.id),
                        "title": new_goal.title
                    })

            return created_goals

        except Exception as e:
            logger.info(f"❌ Improvement goals generation error: {e}")
            return []

    async def generate_next_level_goal(self, completed_goal_id: str) -> Optional[Dict]:
        """
        Генерирует цель следующего уровня (Next Goal Generator)

        Если цель выполнена успешно, может сгенерировать более сложную цель
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(completed_goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal or goal.status != "done":
                return None

            # Генерируем следующую цель только для achievable
            if goal.goal_type != "achievable":
                return None

            next_goal_prompt = f"""На основе успешно выполненной цели сгенерируй следующую цель более высокого уровня:

ВЫПОЛНЕННАЯ ЦЕЛЬ: {goal.title}
РЕЗУЛЬТАТ: {goal.description or 'Не указано'}

Сгенерируй следующую цель которая:
1. Строится на достигнутом результате
2. Увеличивает сложность или масштаб
3. Соответствует исходному направлению

Верни JSON:
{{
    "next_goal": {{
        "title": "Название следующей цели",
        "description": "Описание",
        "reasoning": "Почему это логичный следующий шаг"
    }}
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=next_goal_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            data = json.loads(result)
            next_goal_data = data.get("next_goal")

            if not next_goal_data:
                return None

            # Создаем следующую цель
            async with AsyncSessionLocal() as db:
                new_goal = Goal(
                    title=next_goal_data["title"],
                    description=next_goal_data.get("description", ""),
                    goal_type="achievable",
                    depth_level=0,  # Новая корневая цель
                    status="pending",
                    progress=0.0
                )
                db.add(new_goal)
                await db.commit()
                await db.refresh(new_goal)

                # Отправляем уведомление
                await self._send_next_goal_notification(goal, new_goal, next_goal_data.get("reasoning"))

                return {
                    "id": str(new_goal.id),
                    "title": new_goal.title
                }

        except Exception as e:
            logger.info(f"❌ Next goal generation error: {e}")
            return None

    async def _send_next_goal_notification(self, completed_goal: Goal, next_goal: Goal, reasoning: str):
        """Отправляет уведомление о следующей цели"""
        try:
            import httpx

            message = f"""✅ ЦЕЛЬ ВЫПОЛНЕНА: {completed_goal.title}

🚀 СЛЕДУЮЩАЯ ЦЕЛЬ: {next_goal.title}

📝 {next_goal.description or ''}

💡 Обоснование: {reasoning}
"""

            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{TELEGRAM_URL}/notify",
                    json={"message": message},
                    timeout=5
                )
        except httpx.HTTPError as e:
            logger.debug("telegram_notification_http_error", error=str(e))
        except Exception as e:
            logger.warning("telegram_notification_failed", error=str(e))

    async def evaluate_goal_with_uow(
        self,
        uow: UnitOfWork,
        goal_id: str
    ) -> Dict:
        """
        Оценивает выполнение цели ВНУТРИ существующей UoW транзакции.

        UoW MIGRATION: Атомарная операция - оценка + transition в одной транзакции.

        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели для оценки

        Returns:
            Dict: Результат оценки
        """
        from uuid import UUID

        goal_uuid = UUID(goal_id)
        repo = GoalRepository(uow)

        # Получаем goal с pessimistic lock
        goal = await repo.get_for_update(uow.session, goal_uuid)

        if not goal:
            return {"error": "Goal not found"}

        # Вызываем соответствующий метод оценки
        if goal.goal_type == "achievable":
            return await self._evaluate_achievable_with_uow(uow, goal)
        elif goal.goal_type == "continuous":
            return await self._evaluate_continuous_with_uow(uow, goal)
        elif goal.goal_type == "exploratory":
            return await self._evaluate_exploratory_with_uow(uow, goal)
        elif goal.goal_type == "directional":
            return await self._evaluate_directional_with_uow(uow, goal)
        elif goal.goal_type == "meta":
            return await self._evaluate_meta_with_uow(uow, goal)
        else:
            return await self._evaluate_achievable_with_uow(uow, goal)

    async def _evaluate_achievable_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Оценка achievable цели через UoW"""
        repo = GoalRepository(uow)

        eval_prompt = f"""Оцени выполнение этой цели:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
КРИТЕРИИ УСПЕХА: {goal.success_definition or 'Не определены'}

Проанализируй:
1. Была ли достигнута цель?
2. Какой процент выполнения?
3. Что нужно улучшить?

Верни JSON:
{{
    "passed": true/false,
    "score": 0.0-1.0,
    "reasoning": "Обоснование",
    "gaps": ["Что не выполнено"],
    "improvements": ["Что можно улучшить"]
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
            score = evaluation.get("score", 0.0)
            action = "complete" if passed else "continue"

            # Сохраняем результат оценки
            goal.evaluation_result = evaluation
            await repo.update(uow.session, goal)

            # STATE-MACHINE: Transition на основе оценки
            if passed:
                await transition_service.transition(
                    uow=uow,
                    goal_id=goal.id,
                    new_state="done",
                    reason=f"Evaluation passed with score {score}",
                    actor="goal_evaluator"
                )
                goal.progress = 1.0
                goal.completed_at = datetime.now()
                await repo.update(uow.session, goal)

            return {
                "passed": passed,
                "score": score,
                "reasoning": evaluation.get("reasoning", ""),
                "gaps": evaluation.get("gaps", []),
                "improvements": evaluation.get("improvements", []),
                "action": action,
                "goal_id": str(goal.id)
            }

        except Exception as e:
            return {
                "passed": False,
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
                "action": "continue",
                "goal_id": str(goal.id)
            }

    async def _evaluate_continuous_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Оценка continuous цели через UoW"""
        repo = GoalRepository(uow)

        # Continuous goals оцениваются по trend
        trend = self._calculate_trend(goal)

        evaluation = {
            "trend": trend,
            "metrics": self._get_continuous_metrics(goal),
            "recommendation": self._get_recommendation(trend)
        }

        goal.evaluation_result = evaluation
        await repo.update(uow.session, goal)

        # STATE-MACHINE: Continuous goals не завершаются, но могут менять состояние
        status_map = {
            "improving": "active",
            "stable": "active",
            "degrading": "blocked"
        }
        new_status = status_map.get(trend, "active")

        if new_status == "blocked":
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="blocked",
                reason=f"Performance degrading: {evaluation}",
                actor="goal_evaluator"
            )

        return {
            "passed": trend in ["improving", "stable"],
            "trend": trend,
            "metrics": evaluation["metrics"],
            "recommendation": evaluation["recommendation"],
            "action": "continue" if trend == "improving" else "adjust",
            "goal_id": str(goal.id)
        }

    async def _evaluate_exploratory_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Оценка exploratory цели через UoW"""
        repo = GoalRepository(uow)

        # Exploratory: оцениваем discovery и learnings
        discoveries = goal.evaluation_result.get("discoveries", []) if goal.evaluation_result else []

        evaluation = {
            "discoveries_count": len(discoveries),
            "learnings": discoveries,
            "complete": len(discoveries) >= 3  # Завершаем после 3 discovery
        }

        goal.evaluation_result = evaluation
        await repo.update(uow.session, goal)

        if evaluation["complete"]:
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="done",
                reason=f"Exploration complete: {len(discoveries)} discoveries",
                actor="goal_evaluator"
            )

        return {
            "passed": evaluation["complete"],
            "discoveries": discoveries,
            "action": "complete" if evaluation["complete"] else "continue",
            "goal_id": str(goal.id)
        }

    async def _evaluate_directional_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Оценка directional цели через UoW"""
        # Directional goals не завершаются, всегда active
        evaluation = {
            "type": "directional",
            "note": "Directional goals are never complete - they guide ongoing work",
            "alignment_score": 0.8  # Placeholder
        }

        goal.evaluation_result = evaluation
        await GoalRepository(uow).update(uow.session, goal)

        return {
            "passed": True,  # Directional всегда "passed" в смысле alignment
            "alignment_score": 0.8,
            "action": "continue",
            "note": "Directional goals guide ongoing work and are never marked complete",
            "goal_id": str(goal.id)
        }

    async def _evaluate_meta_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Оценка meta цели через UoW"""
        repo = GoalRepository(uow)

        # Meta goals оцениваются по улучшению системы
        improvements = goal.evaluation_result.get("improvements", []) if goal.evaluation_result else []

        evaluation = {
            "improvements_count": len(improvements),
            "system_impact": "high" if len(improvements) >= 2 else "medium",
            "complete": len(improvements) >= 2
        }

        goal.evaluation_result = evaluation
        await repo.update(uow.session, goal)

        if evaluation["complete"]:
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="done",
                reason=f"Meta-goal complete: {len(improvements)} system improvements",
                actor="goal_evaluator"
            )

        return {
            "passed": evaluation["complete"],
            "improvements": improvements,
            "action": "complete" if evaluation["complete"] else "continue",
            "goal_id": str(goal.id)
        }


# Глобальный экземпляр
goal_evaluator = GoalEvaluator()
