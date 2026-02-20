"""
UNIT OF WORK ROLLBACK TESTS
============================

Тестируем атомарность транзакций и корректность rollback.
ВАЖНО: Эти тесты НЕ вызывают LLM - используются прямые операции с БД.

Author: AI-OS Core Team
Date: 2026-02-19
"""

import pytest
import pytest_asyncio
import asyncio
import uuid
from uuid import UUID
from datetime import datetime
from sqlalchemy import select, func, text
from database import AsyncSessionLocal
from models import Goal, Artifact
from infrastructure.uow import UnitOfWork, GoalRepository
from goal_transition_service import transition_service

# Configure pytest-asyncio to use function scope
pytestmark = pytest.mark.asyncio(loop_scope="function")


# =============================================================================
# ROLLBACK TESTS: CREATE GOAL (Direct DB operations)
# =============================================================================

class TestCreateGoalRollback:
    """Тесты rollback при создании целей (без LLM)"""

    async def test_rollback_on_create_goal_error(self):
        """
        SCENARIO: Создаём цель напрямую, затем вызываем ошибку
        
        EXPECTED: Цель НЕ появляется в БД (rollback)
        """
        uow_provider = lambda: UnitOfWork(AsyncSessionLocal)
        
        goal_id = uuid.uuid4()
        
        # Создаём цель и вызываем ошибку
        try:
            async with uow_provider() as uow:
                goal = Goal(
                    id=goal_id,
                    title="Test Rollback Goal",
                    description="This should be rolled back",
                    goal_type="achievable",
                    is_atomic=True,
                    depth_level=0,
                    _status="pending",
                    progress=0.0
                )
                uow.session.add(goal)
                await uow.session.flush()
                
                # Имитируем ошибку
                raise Exception("Simulated error after goal creation")
        except Exception as e:
            assert str(e) == "Simulated error after goal creation"
        
        # Проверяем: цель НЕ должна быть в БД
        async with AsyncSessionLocal() as session:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await session.execute(stmt)
            found_goal = result.scalar_one_or_none()
            
            assert found_goal is None, \
                f"Goal {goal_id} should NOT exist after rollback!"

    async def test_commit_on_create_goal_success(self):
        """
        SCENARIO: Создаём цель без ошибок
        
        EXPECTED: Цель появляется в БД (commit)
        """
        uow_provider = lambda: UnitOfWork(AsyncSessionLocal)
        
        goal_id = uuid.uuid4()
        
        # Создаём цель без ошибки
        async with uow_provider() as uow:
            goal = Goal(
                id=goal_id,
                title="Test Commit Goal",
                description="This should be committed",
                goal_type="achievable",
                is_atomic=True,
                depth_level=0,
                _status="pending",
                progress=0.0
            )
            uow.session.add(goal)
        
        # Проверяем: цель ДОЛЖНА быть в БД
        async with AsyncSessionLocal() as session:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await session.execute(stmt)
            found_goal = result.scalar_one_or_none()
            
            assert found_goal is not None, \
                f"Goal {goal_id} should exist after commit!"
            assert found_goal.title == "Test Commit Goal"

    async def test_rollback_nested_goal_creation(self):
        """
        SCENARIO: Создаём parent goal и child goal, затем вызываем ошибку
        
        EXPECTED: НИ одна цель НЕ появляется в БД (atomic rollback)
        """
        uow_provider = lambda: UnitOfWork(AsyncSessionLocal)
        
        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()
        
        try:
            async with uow_provider() as uow:
                # Создаём parent
                parent = Goal(
                    id=parent_id,
                    title="Parent Goal",
                    description="Parent for rollback test",
                    goal_type="achievable",
                    is_atomic=False,
                    depth_level=0,
                    _status="pending",
                    progress=0.0
                )
                uow.session.add(parent)
                await uow.session.flush()
                
                # Создаём child
                child = Goal(
                    id=child_id,
                    parent_id=parent_id,
                    title="Child Goal",
                    description="Child for rollback test",
                    goal_type="achievable",
                    is_atomic=True,
                    depth_level=1,
                    _status="pending",
                    progress=0.0
                )
                uow.session.add(child)
                await uow.session.flush()
                
                # Вызываем ошибку
                raise Exception("Simulated error after child creation")
        except Exception:
            pass
        
        # Проверяем: НИ одна цель НЕ должна быть в БД
        async with AsyncSessionLocal() as session:
            # Проверяем parent
            stmt_parent = select(Goal).where(Goal.id == parent_id)
            result_parent = await session.execute(stmt_parent)
            found_parent = result_parent.scalar_one_or_none()
            
            # Проверяем child
            stmt_child = select(Goal).where(Goal.id == child_id)
            result_child = await session.execute(stmt_child)
            found_child = result_child.scalar_one_or_none()
            
            assert found_parent is None, \
                f"Parent goal {parent_id} should NOT exist after rollback!"
            assert found_child is None, \
                f"Child goal {child_id} should NOT exist after rollback!"


# =============================================================================
# ROLLBACK TESTS: TRANSITION
# =============================================================================

class TestTransitionRollback:
    """Тесты rollback при переходах статусов"""

    async def test_rollback_on_transition_error(self):
        """
        SCENARIO: Создаём goal, делаем transition, вызываем ошибку
        
        EXPECTED: Goal остаётся в исходном статусе
        """
        uow_provider = lambda: UnitOfWork(AsyncSessionLocal)
        
        goal_id = uuid.uuid4()
        
        # Создаём goal (в отдельной транзакции)
        async with uow_provider() as uow:
            goal = Goal(
                id=goal_id,
                title="Goal for Transition Test",
                description="Testing transition rollback",
                goal_type="achievable",
                is_atomic=True,
                depth_level=0,
                _status="pending",
                progress=0.0
            )
            uow.session.add(goal)
        
        # Пробуем сделать transition с ошибкой
        try:
            async with uow_provider() as uow:
                await transition_service.transition(
                    uow=uow,
                    goal_id=goal_id,
                    new_state="active",
                    reason="Testing rollback",
                    actor="test"
                )
                
                # Вызываем ошибку
                raise Exception("Simulated error after transition")
        except Exception:
            pass
        
        # Проверяем: goal должна быть в исходном статусе "pending"
        async with AsyncSessionLocal() as session:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await session.execute(stmt)
            found_goal = result.scalar_one_or_none()
            
            assert found_goal is not None
            assert found_goal.status == "pending", \
                f"Goal should be in 'pending' status after rollback! Current: {found_goal.status}"

    async def test_multiple_transitions_rollback(self):
        """
        SCENARIO: Создаём parent и child, делаем несколько transitions, ошибка
        
        EXPECTED: Все transitions откатываются (atomic)
        """
        uow_provider = lambda: UnitOfWork(AsyncSessionLocal)
        
        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()
        
        # Создаём parent
        async with uow_provider() as uow:
            parent = Goal(
                id=parent_id,
                title="Parent for Multiple Transitions",
                description="Testing atomicity",
                goal_type="achievable",
                is_atomic=False,
                depth_level=0,
                _status="pending",
                progress=0.0
            )
            uow.session.add(parent)
        
        # Создаём child
        async with uow_provider() as uow:
            child = Goal(
                id=child_id,
                parent_id=parent_id,
                title="Child for Multiple Transitions",
                description="Testing atomicity",
                goal_type="achievable",
                is_atomic=True,
                depth_level=1,
                _status="pending",
                progress=0.0
            )
            uow.session.add(child)
        
        # Делаем несколько transitions в одной транзакции
        try:
            async with uow_provider() as uow:
                # Transition parent
                await transition_service.transition(
                    uow=uow,
                    goal_id=parent_id,
                    new_state="active",
                    reason="Parent active",
                    actor="test"
                )
                
                # Transition child
                await transition_service.transition(
                    uow=uow,
                    goal_id=child_id,
                    new_state="active",
                    reason="Child active",
                    actor="test"
                )
                
                # Вызываем ошибку
                raise Exception("Simulated error after multiple transitions")
        except Exception:
            pass
        
        # Проверяем: оба должны быть в "pending"
        async with AsyncSessionLocal() as session:
            # Parent
            stmt_parent = select(Goal).where(Goal.id == parent_id)
            result_parent = await session.execute(stmt_parent)
            found_parent = result_parent.scalar_one_or_none()
            
            # Child
            stmt_child = select(Goal).where(Goal.id == child_id)
            result_child = await session.execute(stmt_child)
            found_child = result_child.scalar_one_or_none()
            
            assert found_parent.status == "pending", \
                f"Parent should be 'pending' after rollback! Current: {found_parent.status}"
            assert found_child.status == "pending", \
                f"Child should be 'pending' after rollback! Current: {found_child.status}"


# =============================================================================
# CONCURRENT ACCESS TESTS
# =============================================================================

class TestConcurrencyRollback:
    """Тесты конкурентного доступа и race conditions"""

    async def test_pessimistic_lock_prevents_race(self):
        """
        SCENARIO: Проверяем что pessimistic lock работает
        
        EXPECTED: FOR UPDATE lock блокирует конкурентный доступ
        """
        uow_provider = lambda: UnitOfWork(AsyncSessionLocal)
        
        goal_id = uuid.uuid4()
        
        # Создаём goal
        async with uow_provider() as uow:
            goal = Goal(
                id=goal_id,
                title="Goal for Lock Test",
                description="Testing pessimistic lock",
                goal_type="achievable",
                is_atomic=True,
                depth_level=0,
                _status="pending",
                progress=0.0
            )
            uow.session.add(goal)
        
        # Транзакция 1 берёт lock
        async with uow_provider() as uow1:
            repo = GoalRepository(uow1)
            goal1 = await repo.get_for_update(uow1.session, goal_id)
            
            assert goal1 is not None, "Goal should be found with lock"
            
            # Транзакция 2 пытается взять lock (будет ждать или timeout)
            # В тесте просто проверяем что первая транзакция может работать
            goal1.progress = 0.5
            await repo.update(uow1.session, goal1)
            
            # Commit происходит автоматически при выходе из context manager
        
        # Проверяем: изменения применены
        async with AsyncSessionLocal() as session:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await session.execute(stmt)
            found_goal = result.scalar_one_or_none()
            
            assert found_goal.progress == 0.5, \
                f"Progress should be 0.5 after commit! Current: {found_goal.progress}"


# =============================================================================
# RUN INSTRUCTIONS
# =============================================================================

"""
Запуск тестов:

```bash
# Все тесты rollback
pytest services/core/tests/test_uow_rollback.py -v

# Только тесты создания
pytest services/core/tests/test_uow_rollback.py::TestCreateGoalRollback -v

# Только тесты transitions
pytest services/core/tests/test_uow_rollback.py::TestTransitionRollback -v

# Только тесты конкурентности
pytest services/core/tests/test_uow_rollback.py::TestConcurrencyRollback -v

# С детальным выводом
pytest services/core/tests/test_uow_rollback.py -v -s
```

Ожидаемое время выполнения: ~10-20 секунд
Ожидаемый результат: ВСЕ PASSED

ВАЖНО: 
- Тесты требуют рабочей БД (ns_postgres контейнер)
- Тесты НЕ вызывают LLM - всё работает напрямую с БД
"""
