"""
GOAL MUTATOR - v3.0
Система мутации целей (strengthen/weaken/change_type/freeze)

UoW MIGRATION: Мутации теперь атомарны - все операции в одной транзакции.
"""
import uuid
from typing import Dict, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal
from agent_graph import app_graph

# UoW imports для новой архитектуры
from infrastructure.uow import UnitOfWork, GoalRepository
from goal_transition_service import transition_service


class GoalMutator:
    """
    Мутатор целей - изменяет цели в runtime

    Операции мутации:
    - strengthen: Усилить цель (повысить критерии)
    - weaken: Ослабить цель (снизить критерии)
    - change_type: Сменить тип цели
    - freeze: Заморозить цель
    - thaw: Разморозить цель
    """

    MUTATION_TYPES = [
        "strengthen",   # Усилить цель
        "weaken",       # Ослабить цель
        "change_type",  # Сменить тип
        "freeze",       # Заморозить
        "thaw"          # Разморозить
    ]

    async def mutate_goal(
        self,
        goal_id: str,
        mutation_type: str,
        reason: str,
        **params
    ) -> Dict:
        """
        Мутирует цель

        Args:
            goal_id: ID цели
            mutation_type: Тип мутации (strengthen/weaken/change_type/freeze/thaw)
            reason: Обоснование мутации
            **params: Дополнительные параметры

        Returns:
            Результат мутации
        """
        if mutation_type not in self.MUTATION_TYPES:
            return {"error": f"Invalid mutation type: {mutation_type}"}

        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

            # Проверяем текущий статус
            if goal.mutation_status == "deprecated":
                return {"error": "Cannot mutate deprecated goal"}

            # Выполняем мутацию
            if mutation_type == "strengthen":
                return await self._strengthen_goal(goal, reason, **params)
            elif mutation_type == "weaken":
                return await self._weaken_goal(goal, reason, **params)
            elif mutation_type == "change_type":
                return await self._change_goal_type(goal, reason, **params)
            elif mutation_type == "freeze":
                return await self._freeze_goal(goal, reason)
            elif mutation_type == "thaw":
                return await self._thaw_goal(goal, reason)

    async def _strengthen_goal(self, goal: Goal, reason: str, **params) -> Dict:
        """
        Усиливает цель - повышает критерии успеха

        Examples:
        - scalar 0.7 → 0.9
        - Добавить дополнительные домены
        - Ужесточить completion_criteria
        """
        print(f"🔺 Strengthening goal: {goal.title}")

        # Генерируем усиленные критерии
        strengthen_prompt = f"""Усили эту цель - повысь критерии успеха:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТЕКУЩИЕ КРИТЕРИИ: {goal.completion_criteria or 'Не определены'}
ДОМЕНЫ: {goal.domains or []}
ПРИЧИНА УСИЛЕНИЯ: {reason}

Верни JSON:
{{
    "new_title": "Усиленное название (если нужно)",
    "new_description": "Усиленное описание",
    "new_completion_criteria": {{"condition": "..." }},
    "new_domains": ["domain1", "domain2"],
    "added_constraints": ["Новое ограничение"],
    "strengthening_explanation": "Что именно усилено"
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=strengthen_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            mutation_data = json.loads(result)

            # Применяем мутацию
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()

                if g:
                    # Обновляем поля
                    if mutation_data.get("new_title"):
                        g.title = mutation_data["new_title"]
                    if mutation_data.get("new_description"):
                        g.description = mutation_data["new_description"]
                    if mutation_data.get("new_completion_criteria"):
                        g.completion_criteria = mutation_data["new_completion_criteria"]
                    if mutation_data.get("new_domains"):
                        g.domains = mutation_data["new_domains"]

                    # Добавляем ограничений к существующим
                    if mutation_data.get("added_constraints"):
                        current_constraints = g.constraints or []
                        g.constraints = current_constraints + mutation_data["added_constraints"]

                    # Записываем в историю мутаций
                    mutation_record = {
                        "type": "strengthen",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat(),
                        "changes": mutation_data.get("strengthening_explanation", "")
                    }

                    mutation_history = g.mutation_history or []
                    mutation_history.append(mutation_record)
                    g.mutation_history = mutation_history

                    g.mutation_status = "mutated"
                    await db.commit()

            return {
                "success": True,
                "mutation_type": "strengthen",
                "goal_id": str(goal.id),
                "changes": mutation_data
            }

        except Exception as e:
            return {"error": f"Strengthen mutation failed: {e}"}

    async def _weaken_goal(self, goal: Goal, reason: str, **params) -> Dict:
        """
        Ослабляет цель - снижает критерии успеха

        Examples:
        - scalar 0.9 → 0.6
        - Убрать некоторые домены
        - Упростить completion_criteria
        """
        print(f"🔻 Weakening goal: {goal.title}")

        weaken_prompt = f"""Ослабь эту цель - упрости критерии успеха:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТЕКУЩИЕ КРИТЕРИИ: {goal.completion_criteria or 'Не определены'}
ДОМЕНЫ: {goal.domains or []}
ОГРАНИЧЕНИЯ: {goal.constraints or []}
ПРИЧИНА ОСЛАБЛЕНИЯ: {reason}

Верни JSON:
{{
    "new_title": "Упрощенное название (если нужно)",
    "new_description": "Упрощенное описание",
    "new_completion_criteria": {{"condition": "..." }},
    "removed_domains": ["domain1"],
    "removed_constraints": ["ограничение1"],
    "weakening_explanation": "Что именно упрощено"
}}
"""

        try:
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=weaken_prompt)]
            })

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            mutation_data = json.loads(result)

            # Применяем мутацию
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal.id)
                result = await db.execute(stmt)
                g = result.scalar_one_or_none()

                if g:
                    # Обновляем поля
                    if mutation_data.get("new_title"):
                        g.title = mutation_data["new_title"]
                    if mutation_data.get("new_description"):
                        g.description = mutation_data["new_description"]
                    if mutation_data.get("new_completion_criteria"):
                        g.completion_criteria = mutation_data["new_completion_criteria"]

                    # Удаляем домены
                    if mutation_data.get("removed_domains"):
                        current_domains = g.domains or []
                        g.domains = [d for d in current_domains if d not in mutation_data["removed_domains"]]

                    # Удаляем ограничения
                    if mutation_data.get("removed_constraints"):
                        current_constraints = g.constraints or []
                        g.constraints = [c for c in current_constraints if c not in mutation_data["removed_constraints"]]

                    # Записываем в историю
                    mutation_record = {
                        "type": "weaken",
                        "reason": reason,
                        "timestamp": datetime.now().isoformat(),
                        "changes": mutation_data.get("weakening_explanation", "")
                    }

                    mutation_history = g.mutation_history or []
                    mutation_history.append(mutation_record)
                    g.mutation_history = mutation_history

                    g.mutation_status = "mutated"
                    await db.commit()

            return {
                "success": True,
                "mutation_type": "weaken",
                "goal_id": str(goal.id),
                "changes": mutation_data
            }

        except Exception as e:
            return {"error": f"Weaken mutation failed: {e}"}

    async def _change_goal_type(self, goal: Goal, reason: str, **params) -> Dict:
        """
        Меняет тип цели

        Examples:
        - directional → continuous
        - achievable → exploratory
        """
        print(f"🔄 Changing goal type: {goal.title}")

        new_type = params.get("new_type")

        if not new_type:
            # Если новый тип не указан, определяем через LLM
            type_change_prompt = f"""Определи наиболее подходящий тип для этой цели:

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТЕКУЩИЙ ТИП: {goal.goal_type}
ПРИЧИНА ИЗМЕНЕНИЯ: {reason}

Типология:
- achievable: выполнимая цель (есть финальная точка)
- continuous: непрерывная цель (улучшение, нет финальной точки)
- directional: векторная (задает направление, невыполнимая)
- exploratory: исследовательская (поиск, результат неизвестен)
- meta: мета-цель (улучшение системы)

Верни JSON:
{{
    "new_type": "achievable|continuous|directional|exploratory|meta",
    "reasoning": "Почему этот тип подходит лучше",
    "suggested_changes": ["Что изменить в описании/критериях"]
}}
"""

            try:
                response = await app_graph.ainvoke({
                    "messages": [HumanMessage(content=type_change_prompt)]
                })

                result = response["messages"][-1].content

                import json
                if "```json" in result:
                    result = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    result = result.split("```")[1].split("```")[0].strip()

                type_data = json.loads(result)
                new_type = type_data.get("new_type")

            except Exception as e:
                return {"error": f"Type detection failed: {e}"}

        # Применяем изменение типа
        async with AsyncSessionLocal() as db:
            from goal_contract_validator import goal_contract_validator

            stmt = select(Goal).where(Goal.id == goal.id)
            result = await db.execute(stmt)
            g = result.scalar_one_or_none()

            if g:
                old_type = g.goal_type
                g.goal_type = new_type

                # Обновляем контракт для нового типа
                g.goal_contract = goal_contract_validator.create_default_contract(new_type)

                # Записываем в историю
                mutation_record = {
                    "type": "change_type",
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                    "from_type": old_type,
                    "to_type": new_type
                }

                mutation_history = g.mutation_history or []
                mutation_history.append(mutation_record)
                g.mutation_history = mutation_history

                g.mutation_status = "mutated"
                await db.commit()

        return {
            "success": True,
            "mutation_type": "change_type",
            "goal_id": str(goal.id),
            "from_type": goal.goal_type,
            "to_type": new_type
        }

    async def _freeze_goal(self, goal: Goal, reason: str) -> Dict:
        """
        Замораживает цель - временно прекращает выполнение

        Замороженные цели:
        - Не выполняются
        - Не декомпозируются
        - Не оцениваются
        """
        print(f"❄️ Freezing goal: {goal.title}")

        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal.id)
            result = await db.execute(stmt)
            g = result.scalar_one_or_none()

            if g:
                # Если цель была active - ставим на паузу
                if g.status == "active":
                    g.status = "pending"

                g.mutation_status = "frozen"

                # Записываем в историю
                mutation_record = {
                    "type": "freeze",
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                    "previous_status": "active"
                }

                mutation_history = g.mutation_history or []
                mutation_history.append(mutation_record)
                g.mutation_history = mutation_history

                await db.commit()

        return {
            "success": True,
            "mutation_type": "freeze",
            "goal_id": str(goal.id)
        }

    async def _thaw_goal(self, goal: Goal, reason: str) -> Dict:
        """
        Размораживает цель - возобновляет выполнение
        """
        print(f"🔥 Thawing goal: {goal.title}")

        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal.id)
            result = await db.execute(stmt)
            g = result.scalar_one_or_none()

            if g:
                g.mutation_status = "active"

                # Если цель была frozen - возвращаем статус
                if g.status == "pending":
                    g.status = "active"

                # Записываем в историю
                mutation_record = {
                    "type": "thaw",
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                }

                mutation_history = g.mutation_history or []
                mutation_history.append(mutation_record)
                g.mutation_history = mutation_history

                await db.commit()

        return {
            "success": True,
            "mutation_type": "thaw",
            "goal_id": str(goal.id)
        }

    # ============= UoW MIGRATION: Новые атомарные методы =============

    async def mutate_goal_with_uow(
        self,
        uow: UnitOfWork,
        goal_id: str,
        mutation_type: str,
        reason: str,
        **params
    ) -> Dict:
        """
        Мутирует цель ВНУТРИ существующей UoW транзакции.

        UoW MIGRATION: Атомарная операция - мутация + state transition в одной транзакции.

        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели
            mutation_type: Тип мутации
            reason: Обоснование
            **params: Дополнительные параметры

        Returns:
            Результат мутации
        """
        if mutation_type not in self.MUTATION_TYPES:
            return {"error": f"Invalid mutation type: {mutation_type}"}

        from uuid import UUID
        goal_uuid = UUID(goal_id)
        repo = GoalRepository(uow)

        # Получаем goal с pessimistic lock
        goal = await repo.get_for_update(uow.session, goal_uuid)

        if not goal:
            return {"error": "Goal not found"}

        # Проверяем текущий статус
        if goal.mutation_status == "deprecated":
            return {"error": "Cannot mutate deprecated goal"}

        # Выполняем мутацию через UoW
        if mutation_type == "strengthen":
            return await self._strengthen_goal_with_uow(uow, goal, reason, **params)
        elif mutation_type == "weaken":
            return await self._weaken_goal_with_uow(uow, goal, reason, **params)
        elif mutation_type == "change_type":
            return await self._change_goal_type_with_uow(uow, goal, reason, **params)
        elif mutation_type == "freeze":
            return await self._freeze_goal_with_uow(uow, goal, reason)
        elif mutation_type == "thaw":
            return await self._thaw_goal_with_uow(uow, goal, reason)

        return {"error": f"Unknown mutation type: {mutation_type}"}

    async def _strengthen_goal_with_uow(self, uow: UnitOfWork, goal: Goal, reason: str, **params) -> Dict:
        """Усилить цель через UoW"""
        repo = GoalRepository(uow)

        # Обновляем критерии успеха
        current_criteria = goal.completion_criteria or {}
        new_criteria = params.get("completion_criteria", {})
        current_criteria.update(new_criteria)
        goal.completion_criteria = current_criteria

        # Обновляем success_definition
        if params.get("success_definition"):
            goal.success_definition = params["success_definition"]

        # Сохраняем мутацию
        mutation_record = {
            "type": "strengthen",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "changes": {
                "completion_criteria": new_criteria,
                "success_definition": params.get("success_definition")
            }
        }

        mutation_history = goal.mutation_history or []
        mutation_history.append(mutation_record)
        goal.mutation_history = mutation_history
        goal.mutation_status = "mutated"

        await repo.update(uow.session, goal)

        return {
            "success": True,
            "mutation_type": "strengthen",
            "goal_id": str(goal.id),
            "changes": mutation_record["changes"]
        }

    async def _weaken_goal_with_uow(self, uow: UnitOfWork, goal: Goal, reason: str, **params) -> Dict:
        """Ослабить цель через UoW"""
        repo = GoalRepository(uow)

        # Упрощаем критерии
        if params.get("remove_criteria"):
            current_criteria = goal.completion_criteria or {}
            for key in params["remove_criteria"]:
                current_criteria.pop(key, None)
            goal.completion_criteria = current_criteria

        # Обновляем success_definition
        if params.get("success_definition"):
            goal.success_definition = params["success_definition"]

        # Сохраняем мутацию
        mutation_record = {
            "type": "weaken",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "changes": {
                "removed_criteria": params.get("remove_criteria", []),
                "success_definition": params.get("success_definition")
            }
        }

        mutation_history = goal.mutation_history or []
        mutation_history.append(mutation_record)
        goal.mutation_history = mutation_history
        goal.mutation_status = "mutated"

        await repo.update(uow.session, goal)

        return {
            "success": True,
            "mutation_type": "weaken",
            "goal_id": str(goal.id),
            "changes": mutation_record["changes"]
        }

    async def _change_goal_type_with_uow(self, uow: UnitOfWork, goal: Goal, reason: str, **params) -> Dict:
        """Сменить тип цели через UoW"""
        repo = GoalRepository(uow)

        new_type = params.get("new_type")
        if not new_type:
            return {"error": "new_type required for change_type mutation"}

        old_type = goal.goal_type
        goal.goal_type = new_type

        # Обновляем contract для нового типа
        from goal_contract_validator import goal_contract_validator
        goal.goal_contract = goal_contract_validator.create_default_contract(new_type, goal.depth_level)

        # Сохраняем мутацию
        mutation_record = {
            "type": "change_type",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "changes": {
                "old_type": old_type,
                "new_type": new_type
            }
        }

        mutation_history = goal.mutation_history or []
        mutation_history.append(mutation_record)
        goal.mutation_history = mutation_history
        goal.mutation_status = "mutated"

        await repo.update(uow.session, goal)

        return {
            "success": True,
            "mutation_type": "change_type",
            "goal_id": str(goal.id),
            "old_type": old_type,
            "new_type": new_type
        }

    async def _freeze_goal_with_uow(self, uow: UnitOfWork, goal: Goal, reason: str) -> Dict:
        """Заморозить цель через UoW"""
        repo = GoalRepository(uow)

        # Transition в frozen state
        if goal._status == "active":
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="frozen",
                reason=f"Goal frozen: {reason}",
                actor="goal_mutator"
            )

        goal.mutation_status = "frozen"

        mutation_record = {
            "type": "freeze",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

        mutation_history = goal.mutation_history or []
        mutation_history.append(mutation_record)
        goal.mutation_history = mutation_history

        await repo.update(uow.session, goal)

        return {
            "success": True,
            "mutation_type": "freeze",
            "goal_id": str(goal.id)
        }

    async def _thaw_goal_with_uow(self, uow: UnitOfWork, goal: Goal, reason: str) -> Dict:
        """Разморозить цель через UoW"""
        repo = GoalRepository(uow)

        goal.mutation_status = "active"

        # Если цель была frozen, возвращаем её в active
        if goal._status == "frozen":
            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="active",
                reason=f"Goal thawed: {reason}",
                actor="goal_mutator"
            )

        mutation_record = {
            "type": "thaw",
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

        mutation_history = goal.mutation_history or []
        mutation_history.append(mutation_record)
        goal.mutation_history = mutation_history

        await repo.update(uow.session, goal)

        return {
            "success": True,
            "mutation_type": "thaw",
            "goal_id": str(goal.id)
        }


# Глобальный экземпляр
goal_mutator = GoalMutator()
