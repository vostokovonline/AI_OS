"""
GOAL REFLECTOR - v3.0
Анализ причин и генерация следующих целей (каузальное мышление)
НЕ проверяет выполнение - это делает StrictEvaluator

UoW MIGRATION: Рефлексия теперь атомарна - все операции в одной транзакции.
"""
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal
from agent_graph import app_graph
from goal_contract_validator import goal_contract_validator

# UoW imports для новой архитектуры
from infrastructure.uow import UnitOfWork, GoalRepository
from goal_transition_service import transition_service


TELEGRAM_URL = os.getenv("TELEGRAM_URL", "http://telegram:8004")


class GoalReflector:
    """
    Рефлектор - анализирует ПОЧЕМУ и ЧТО ДАЛЬШЕ

    Ответственность:
    - Анализ причин успеха/неудачи
    - Извлечение уроков (pattern extraction)
    - Генерация следующих целей
    - Рекомендации по улучшению

    НЕ отвечает за:
    - Проверку выполнения (это делает StrictEvaluator)
    """

    async def reflect_on_goal(self, goal_id: str, strict_evaluation: Dict) -> Dict:
        """
        Анализирует результат оценки и определяет что дальше

        Args:
            goal_id: ID цели
            strict_evaluation: Результат от StrictEvaluator

        Returns:
            {
                "why": "Причины успеха/неудачи",
                "lessons_learned": ["Урок 1", "Урок 2"],
                "recommendations": ["Рекомендация 1"],
                "next_goals": [...],
                "action": "complete|continue|adjust|mutate"
            }
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

            passed = strict_evaluation.get("passed", False)
            score = strict_evaluation.get("score", 0.0)
            trend = strict_evaluation.get("trend", None)

            # Разный анализ для разных сценариев
            if passed:
                return await self._reflect_on_success(goal, score)
            elif trend == "degrading":
                return await self._reflect_on_degradation(goal)
            else:
                return await self._reflect_on_failure(goal, score)

    async def _reflect_on_success(self, goal: Goal, score: float) -> Dict:
        """
        Анализ успеха: ПОЧЕМУ получилось, ЧТО learned

        Извлекает паттерны успеха для future goals
        """
        reflection_prompt = f"""Проанализируй ПОЧЕМУ эта цель успешно выполнена:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТИП: {goal.goal_type}
УРОВЕНЬ: {goal.depth_level}
ДОМЕНЫ: {goal.domains or []}
ОЦЕНКА: {score:.2f}

Проанализируй:
1. Какие факторы способствовали успеху?
2. Какие методы/подходы сработали?
3. Какие паттерны можно использовать в будущем?
4. Что можно улучшить на следующий раз?

Верни JSON:
{{
    "why_success": "Почему получилось",
    "success_factors": ["Фактор 1", "Фактор 2"],
    "lessons_learned": ["Урок 1", "Урок 2"],
    "patterns": ["Паттерн 1"],
    "recommendations": ["Рекомендация 1"],
    "should_generate_next": true/false,
    "next_goal_idea": "Идея следующей цели (если применимо)"
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=reflection_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            reflection = json.loads(result)

            # Генерируем следующую цель если нужно
            next_goals = []
            if reflection.get("should_generate_next") and goal.goal_type == "achievable":
                next_goal = await self._generate_next_goal(goal, reflection)
                if next_goal:
                    next_goals.append(next_goal)

            return {
                "why": reflection.get("why_success", ""),
                "lessons_learned": reflection.get("lessons_learned", []),
                "success_factors": reflection.get("success_factors", []),
                "patterns": reflection.get("patterns", []),
                "recommendations": reflection.get("recommendations", []),
                "next_goals": next_goals,
                "action": "complete"
            }

        except Exception as e:
            return {
                "why": f"Ошибка анализа: {e}",
                "lessons_learned": [],
                "action": "complete"
            }

    async def _reflect_on_failure(self, goal: Goal, score: float) -> Dict:
        """
        Анализ неудачи: ПОЧЕМУ не получилось, ЧТО исправить

        Извлекает паттерны проблем для future improvements
        """
        reflection_prompt = f"""Проанализируй ПОЧЕМУ эта цель НЕ выполнена:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТИП: {goal.goal_type}
КРИТЕРИИ: {goal.completion_criteria or 'Не определены'}
ТЕКУЩИЙ ПРОГРЕСС: {int(goal.progress * 100)}%
ОЦЕНКА: {score:.2f}

Проанализируй:
1. Что помешало выполнению?
2. Какие ошибки были допущены?
3. Чего не хватило (ресурсы, знания, время)?
4. Как можно исправить ситуацию?

Верни JSON:
{{
    "why_failed": "Почему не получилось",
    "root_causes": ["Причина 1", "Причина 2"],
    "mistakes": ["Ошибка 1"],
    "missing_resources": ["Ресурс 1"],
    "remediation": [
        {{
            "title": "Корректирующая цель",
            "description": "Что сделать",
            "priority": "high|medium|low"
        }}
    ]
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=reflection_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            reflection = json.loads(result)

            # Генерируем корректирующие цели
            improvement_goals = []
            for remediation in reflection.get("remediation", []):
                goal_data = await self._create_improvement_goal(goal, remediation)
                if goal_data:
                    improvement_goals.append(goal_data)

            return {
                "why": reflection.get("why_failed", ""),
                "root_causes": reflection.get("root_causes", []),
                "mistakes": reflection.get("mistakes", []),
                "missing_resources": reflection.get("missing_resources", []),
                "improvement_goals": improvement_goals,
                "action": "continue" if improvement_goals else "adjust"
            }

        except Exception as e:
            return {
                "why": f"Ошибка анализа: {e}",
                "action": "continue"
            }

    async def _reflect_on_degradation(self, goal: Goal) -> Dict:
        """
        Анализ деградации: ПОЧЕМУ стало хуже

        Для continuous целей с трендом degrading
        """
        return {
            "why": "Цель ухудшилась, требуется пересмотр стратегии",
            "root_causes": ["Нужно детальное исследование"],
            "action": "mutate",  # v3.0: мутация цели
            "mutation_suggestion": {
                "type": "weaken",
                "reason": "Упростить цель для восстановления тренда"
            }
        }

    async def _generate_next_goal(self, completed_goal: Goal, reflection: Dict) -> Optional[Dict]:
        """
        Генерирует следующую цель эволюционного уровня

        Next Goal Generator - создает более сложную цель
        """
        next_goal_prompt = f"""На основе успешно выполненной цели сгенерируй следующую более сложную цель:

ВЫПОЛНЕННАЯ ЦЕЛЬ: {completed_goal.title}
РЕЗУЛЬТАТ: {completed_goal.description or 'Не указано'}
ТИП: {completed_goal.goal_type}
УРОКИ: {reflection.get('lessons_learned', [])}
ПАТТЕРНЫ УСПЕХА: {reflection.get('patterns', [])}

Сгенерируй следующую цель которая:
1. Строится на достигнутом результате
2. Увеличивает сложность или масштаб
3. Использует извлеченные паттерны успеха
4. Соответствует исходному направлению

Верни JSON:
{{
    "next_goal": {{
        "title": "Название следующей цели",
        "description": "Описание",
        "goal_type": "achievable|continuous|exploratory",
        "reasoning": "Почему это логичный следующий шаг",
        "complexity_increase": "как усложняется"
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
            from goal_contract_validator import goal_contract_validator

            goal_type = next_goal_data.get("goal_type", "achievable")
            goal_contract = goal_contract_validator.create_default_contract(goal_type)

            async with AsyncSessionLocal() as db:
                new_goal = Goal(
                    title=next_goal_data["title"],
                    description=next_goal_data.get("description", ""),
                    goal_type=goal_type,
                    goal_contract=goal_contract,
                    depth_level=0,  # Новая корневая цель
                    status="pending",
                    progress=0.0
                )
                db.add(new_goal)
                await db.commit()
                await db.refresh(new_goal)

                # Отправляем уведомление
                await self._send_next_goal_notification(
                    completed_goal,
                    new_goal,
                    next_goal_data.get("reasoning")
                )

                return {
                    "id": str(new_goal.id),
                    "title": new_goal.title
                }

        except Exception as e:
            print(f"❌ Next goal generation error: {e}")
            return None

    async def _create_improvement_goal(self, parent_goal: Goal, remediation: Dict) -> Optional[Dict]:
        """Создает корректирующую цель на основе анализа неудачи"""

        from goal_contract_validator import goal_contract_validator

        goal_contract = goal_contract_validator.create_default_contract("achievable")

        async with AsyncSessionLocal() as db:
            new_goal = Goal(
                parent_id=parent_goal.id,
                title=remediation["title"],
                description=remediation.get("description", ""),
                goal_type="achievable",
                goal_contract=goal_contract,
                depth_level=parent_goal.depth_level + 1,
                status="pending",
                progress=0.0
            )
            db.add(new_goal)
            await db.commit()
            await db.refresh(new_goal)

            return {
                "id": str(new_goal.id),
                "title": new_goal.title,
                "priority": remediation.get("priority", "medium")
            }

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
        except:
            pass

    # ============= UoW MIGRATION: Новые атомарные методы =============

    async def reflect_on_goal_with_uow(
        self,
        uow: UnitOfWork,
        goal_id: str,
        strict_evaluation: Dict
    ) -> Dict:
        """
        Анализирует результат оценки ВНУТРИ существующей UoW транзакции.

        UoW MIGRATION: Атомарная операция - рефлексия + создание next goals в одной транзакции.

        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели
            strict_evaluation: Результат от StrictEvaluator

        Returns:
            Dict: Результат рефлексии
        """
        from uuid import UUID
        goal_uuid = UUID(goal_id)
        repo = GoalRepository(uow)

        # Получаем goal с pessimistic lock
        goal = await repo.get_for_update(uow.session, goal_uuid)

        if not goal:
            return {"error": "Goal not found"}

        passed = strict_evaluation.get("passed", False)
        score = strict_evaluation.get("score", 0.0)
        trend = strict_evaluation.get("trend", None)

        # Разный анализ для разных сценариев
        if passed:
            return await self._reflect_on_success_with_uow(uow, goal, score)
        elif trend == "degrading":
            return await self._reflect_on_degradation_with_uow(uow, goal)
        else:
            return await self._reflect_on_failure_with_uow(uow, goal, score)

    async def _reflect_on_success_with_uow(self, uow: UnitOfWork, goal: Goal, score: float) -> Dict:
        """Анализ успеха через UoW"""
        reflection_prompt = f"""Проанализируй ПОЧЕМУ эта цель успешно выполнена:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
SCORE: {score}

Выясни:
1. Какие факторы привели к успеху?
2. Какие паттерны можно переиспользовать?
3. Что делать дальше (следующая цель)?

Верни JSON:
{{
    "why": "Причины успеха",
    "lessons_learned": ["Урок 1", "Урок 2"],
    "recommendations": ["Рекомендация 1"],
    "next_goal": {{
        "title": "Название следующей цели",
        "description": "Описание",
        "goal_type": "achievable"
    }},
    "action": "complete"
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=reflection_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            reflection = json.loads(result)

            # Создаём next goal если нужно
            next_goal_data = reflection.get("next_goal")
            next_goal = None
            if next_goal_data:
                next_goal = await self._create_next_goal_with_uow(
                    uow, goal, next_goal_data, reflection.get("why", "")
                )

            # Сохраняем рефлексию
            goal.reflection = reflection
            await GoalRepository(uow).update(uow.session, goal)

            return {
                "why": reflection.get("why", ""),
                "lessons_learned": reflection.get("lessons_learned", []),
                "recommendations": reflection.get("recommendations", []),
                "next_goal": {
                    "id": str(next_goal.id) if next_goal else None,
                    "title": next_goal.title if next_goal else None
                } if next_goal else None,
                "action": reflection.get("action", "complete"),
                "goal_id": str(goal.id)
            }

        except Exception as e:
            return {
                "why": f"Reflection error: {str(e)}",
                "lessons_learned": [],
                "recommendations": ["Review goal manually"],
                "next_goal": None,
                "action": "complete",
                "goal_id": str(goal.id)
            }

    async def _reflect_on_failure_with_uow(self, uow: UnitOfWork, goal: Goal, score: float) -> Dict:
        """Анализ неудачи через UoW"""
        reflection_prompt = f"""Проанализируй ПОЧЕМУ эта цель НЕ выполнена:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
SCORE: {score}

Выясни:
1. Какие факторы привели к неудаче?
2. Что можно улучшить?
3. Продолжать ли эту цель или скорректировать?

Верни JSON:
{{
    "why": "Причины неудачи",
    "lessons_learned": ["Урок 1", "Урок 2"],
    "recommendations": ["Что улучшить"],
    "action": "continue|adjust|mutate"
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=reflection_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            reflection = json.loads(result)

            # Сохраняем рефлексию
            goal.reflection = reflection
            await GoalRepository(uow).update(uow.session, goal)

            action = reflection.get("action", "continue")

            # Если action = mutate, замораживаем текущую цель
            if action == "mutate":
                await transition_service.transition(
                    uow=uow,
                    goal_id=goal.id,
                    new_state="frozen",
                    reason=f"Reflection suggests mutation: {reflection.get('why', '')}",
                    actor="goal_reflector"
                )

            return {
                "why": reflection.get("why", ""),
                "lessons_learned": reflection.get("lessons_learned", []),
                "recommendations": reflection.get("recommendations", []),
                "action": action,
                "goal_id": str(goal.id)
            }

        except Exception as e:
            return {
                "why": f"Reflection error: {str(e)}",
                "lessons_learned": [],
                "recommendations": ["Review goal manually"],
                "action": "continue",
                "goal_id": str(goal.id)
            }

    async def _reflect_on_degradation_with_uow(self, uow: UnitOfWork, goal: Goal) -> Dict:
        """Анализ деградации через UoW"""
        reflection = {
            "why": "Performance degrading over time",
            "lessons_learned": ["Current approach not sustainable"],
            "recommendations": ["Review strategy", "Consider mutation"],
            "action": "mutate"
        }

        # Замораживаем цель при деградации
        await transition_service.transition(
            uow=uow,
            goal_id=goal.id,
            new_state="frozen",
            reason="Performance degrading - requires strategy review",
            actor="goal_reflector"
        )

        goal.reflection = reflection
        await GoalRepository(uow).update(uow.session, goal)

        return {
            **reflection,
            "goal_id": str(goal.id)
        }

    async def _create_next_goal_with_uow(
        self,
        uow: UnitOfWork,
        parent_goal: Goal,
        goal_data: Dict,
        reasoning: str
    ) -> Optional[Goal]:
        """Создаёт next goal через UoW"""
        try:
            # Создаём контракт для новой цели
            goal_contract = goal_contract_validator.create_default_contract(
                goal_data.get("goal_type", "achievable"),
                parent_goal.depth_level + 1
            )

            next_goal = Goal(
                parent_id=parent_goal.id,
                title=goal_data["title"],
                description=goal_data.get("description", f"Next goal after: {reasoning}"),
                goal_type=goal_data.get("goal_type", "achievable"),
                depth_level=parent_goal.depth_level + 1,
                is_atomic=False,  # Will be decomposed
                domains=parent_goal.domains or [],
                goal_contract=goal_contract,
                status="pending",
                progress=0.0
            )

            await GoalRepository(uow).save(uow.session, next_goal)
            await uow.session.flush([next_goal])

            return next_goal

        except Exception as e:
            print(f"⚠️ Failed to create next goal: {e}")
            return None


# Глобальный экземпляр
goal_reflector = GoalReflector()
