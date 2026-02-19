"""
GOAL EXECUTOR - Система для достижения сложных целей
=================================================================
Использует UnitOfWor паттерн для управления транзакциями.

Author: AI-OS Core Team
Date: 2026-02-12
"""
import os, asyncio, httpx, uuid
from uuid import UUID
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal, Message, ChatSession
from agent_graph import app_graph
from telemetry import log_action
import json

from infrastructure.uow import UnitOfWork, create_uow_provider


TELEGRAM_URL = os.getenv("TELEGRAM_URL", "http://telegram:8004")
MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8001")
OPENCODE_URL = os.getenv("OPENCODE_URL", "http://opencode:8002")


class GoalExecutor:
    """
    Orchestrator для достижения сложных целей.
    
    Использует UnitOfWor паттерн - транзакция открывается на уровне executor,
    все операции внутри одной транзакции.
    """

    def __init__(self):
        self.active_goals = {}
        self._uow_provider = create_uow_provider()

    async def create_goal(
        self,
        title: str,
        description: str = "",
        goal_type: str = None,
        auto_classify: bool = True,
        is_atomic: bool = False,
        depth_level: int = None,
        parent_id: str = None,
        user_id: str = None
    ) -> str:
        """
        Создает новую цель с классификацией и анализом доменов.
        
        LEGACY: Создаёт собственный UoW. Для нового кода используйте create_goal_with_uow().
        """
        async with self._uow_provider() as uow:
            goal = await self.create_goal_with_uow(
                uow=uow,
                title=title,
                description=description,
                goal_type=goal_type,
                auto_classify=auto_classify,
                is_atomic=is_atomic,
                depth_level=depth_level,
                parent_id=parent_id,
                user_id=user_id
            )
            return str(goal.id)

    async def create_goal_with_uow(
        self,
        uow: "UnitOfWork",
        title: str,
        description: str = "",
        goal_type: str = None,
        auto_classify: bool = True,
        is_atomic: bool = False,
        depth_level: int = None,
        parent_id: str = None,
        user_id: str = None
    ) -> Goal:
        """
        Создает новую цель внутри существующей UoW транзакции.
        
        Это единственно правильный способ создания целей в новой архитектуре.
        Endpoint должен передавать UoW через Depends(get_uow).
        
        Args:
            uow: UnitOfWork с активной транзакцией
            title: Название цели
            description: Описание цели
            goal_type: Тип цели (achievable, continuous, etc.)
            auto_classify: Автоматически классифицировать
            is_atomic: Является ли цель атомарной
            depth_level: Уровень глубины (auto-calculated если None)
            parent_id: ID родительской цели
            user_id: ID пользователя
            
        Returns:
            Goal: Созданный объект цели (внутри транзакции)
        """
        from goal_decomposer import goal_decomposer
        from goal_contract_validator import goal_contract_validator
        from infrastructure.uow import GoalRepository
        from goal_transition_service import transition_service
        
        # Классифицируем цель если нужно
        if auto_classify:
            classification = await goal_decomposer.classify_goal(title, description)
            final_goal_type = goal_type or classification.get("goal_type", "achievable")
        else:
            final_goal_type = goal_type or "achievable"

        # Анализируем домены
        domains = await goal_decomposer.analyze_domains(title, description) if auto_classify else []

        # AUTO-CALCULATE depth_level based on parent_id
        calculated_depth_level = depth_level
        if calculated_depth_level is None:
            if parent_id:
                try:
                    parent_uuid = UUID(parent_id)
                    # ✅ Используем переданный UoW вместо нового AsyncSessionLocal
                    repo = GoalRepository()
                    parent_goal = await repo.get(uow.session, parent_uuid)
                    if parent_goal:
                        calculated_depth_level = (parent_goal.depth_level or 0) + 1
                    else:
                        calculated_depth_level = 1
                except Exception:
                    calculated_depth_level = 1
            else:
                calculated_depth_level = 0

        print(f"🎯 Final depth_level for goal '{title}': {calculated_depth_level}")

        # GOAL CONTRACT v3.0
        goal_contract = goal_contract_validator.create_default_contract(
            final_goal_type, calculated_depth_level
        )

        # Конвертируем UUID
        parent_uuid = None
        if parent_id:
            try:
                parent_uuid = UUID(parent_id)
            except ValueError:
                parent_uuid = None

        user_uuid = None
        if user_id:
            try:
                user_uuid = UUID(user_id)
            except ValueError:
                user_uuid = None

        # Создаем цель
        goal = Goal(
            title=title,
            description=description or title,
            goal_type=final_goal_type,
            domains=domains,
            depth_level=calculated_depth_level,
            is_atomic=is_atomic,
            goal_contract=goal_contract,
            parent_id=parent_uuid,
            user_id=user_uuid,
            _status="pending",
            progress=0.0
        )
        
        # Сохраняем через UoW
        repo = GoalRepository()
        await repo.save(uow.session, goal)
        
        # Transition: pending → pending (логируем создание)
        await transition_service.transition(
            uow=uow,
            goal_id=goal.id,
            new_state="pending",
            reason="Initial goal creation",
            actor="goal_executor"
        )

        return goal

    async def execute_goal(self, goal_id: str, session_id: str = None) -> dict:
        """
        Выполняет цель через агентов.
        
        Transaction boundary: одна транзакция на всё выполнение.
        """
        from goal_contract_validator import goal_contract_validator
        from infrastructure.uow import GoalRepository
        from goal_transition_service import transition_service
        
        goal_uuid = UUID(goal_id)
        
        async with self._uow_provider() as uow:
            repo = GoalRepository()
            goal = await repo.get(uow.session, goal_uuid)
            
            if not goal:
                return {"status": "error", "message": "Goal not found"}

            # GOAL CONTRACT CHECK v3.0
            can_execute, reason = goal_contract_validator.can_execute_action(goal, "execute")
            if not can_execute:
                print(f"⛔ Execution forbidden: {reason}")
                return {"status": "error", "message": f"Execution forbidden: {reason}"}

            # DELEGATE TO GOAL EXECUTOR V2 FOR ATOMIC GOALS
            if goal.is_atomic:
                print(f"⚡ Delegating atomic goal to GoalExecutorV2: {goal.title}")
                from goal_executor_v2 import goal_executor_v2
                return await goal_executor_v2.execute_goal_with_uow(
                    uow, goal_id, session_id
                )

            # Transition: pending → active
            await transition_service.transition(
                uow=uow,
                goal_id=goal_uuid,
                new_state="active",
                reason="Decomposition started",
                actor="goal_executor"
            )

        # Создаем сессию если не передана
        if not session_id:
            session_id = f"goal_{goal_id}"

        # Personality Decision Engine (вне транзакции)
        personality_bias = None
        try:
            from personality_decision_integration import evaluate_with_personality
            from decision_field import GoalPressure

            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal_uuid)
                result = await db.execute(stmt)
                goal = result.scalar_one_or_none()
                
                if goal:
                    pressure = GoalPressure(
                        goal_id=str(goal.id),
                        title=goal.title,
                        priority="high",
                        magnitude=goal.progress or 0.5
                    )

            if goal:
                personality_bias = await evaluate_with_personality(
                    user_id=str(goal_id),
                    goals=[pressure],
                    constraints=None,
                    system_state=None
                )

                print(f"✅ Personality-aware bias computed:")
                print(f"   - Tone: {personality_bias.tone}")
        except Exception as e:
            print(f"⚠️ Failed to compute personality bias: {e}")

        # Agent Graph Execution (вне транзакции - это long-running)
        execution_prompt = f"""GOAL: {goal.title}

DESCRIPTION: {goal.description}

INSTRUCTIONS:
You are an autonomous goal executor. Your mission is to achieve this goal completely.
Break it down into steps, execute them, and report progress.

CRITICAL RULES:
1. DO NOT create new goals - this creates infinite loops!
2. DO NOT use create_goal tool under any circumstances!
3. Work directly on the current goal using available tools
4. When done, report "TASK COMPLETED" clearly

Start working on this goal now."""

        # ... execution logic continues ...
        
        return {"status": "executing", "goal_id": goal_id}


# Глобальный экземпляр
goal_executor = GoalExecutor()


# CELERY TASKS
from celery_config import celery_app


@celery_app.task(bind=True)
def execute_goal_task(self, goal_id: str, session_id: str = None):
    """Фоновая задача для выполнения цели"""
    result = asyncio.run(goal_executor.execute_goal(goal_id, session_id))
    return result


@celery_app.task(bind=True)
def execute_complex_goal_task(self, user_request: str):
    """Фоновая задача для выполнения сложной цели из естественного языка"""
    result = asyncio.run(goal_executor.execute_complex_goal(user_request))
    return result
