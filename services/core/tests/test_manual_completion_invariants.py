"""
TESTS: MANUAL Completion Invariants (I7)

Тестируем инвариант I7:
goal.completion_mode == MANUAL AND goal.status == done
⇒ EXISTS goal_completion_approval(goal_id)

Author: AI-OS Core Team
Date: 2026-02-06
"""

import pytest
import uuid
import asyncio
from datetime import datetime
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal, GoalCompletionApproval
from invariants_checker import invariants_checker


@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """
    Cleanup test data before each test.
    Removes any goals created during tests.
    """
    async with AsyncSessionLocal() as db:
        # Delete test approvals
        approvals = await db.execute(
            select(GoalCompletionApproval).where(
                GoalCompletionApproval.approved_by.like("user:%")
            )
        )
        for approval in approvals.scalars().all():
            await db.delete(approval)

        # Delete test goals
        goals = await db.execute(
            select(Goal).where(Goal.title.like("Test%"))
        )
        for goal in goals.scalars().all():
            await db.delete(goal)

        await db.commit()

    yield

    # Cleanup after test as well
    async with AsyncSessionLocal() as db:
        approvals = await db.execute(
            select(GoalCompletionApproval).where(
                GoalCompletionApproval.approved_by.like("user:%")
            )
        )
        for approval in approvals.scalars().all():
            await db.delete(approval)

        goals = await db.execute(
            select(Goal).where(Goal.title.like("Test%"))
        )
        for goal in goals.scalars().all():
            await db.delete(goal)

        await db.commit()


class TestManualCompletionInvariants:
    """Тесты для MANUAL completion invariant I7"""

    @pytest.mark.asyncio
    async def test_manual_done_with_approval_pass(self):
        """
        ✅ MANUAL goal + approval → PASS

        MANUAL цель в статусе done с approval record
        должно PASS инвариант I7.
        """
        async with AsyncSessionLocal() as db:
            # 1. Создаём MANUAL goal
            goal = Goal(
                id=uuid.uuid4(),
                title="Test MANUAL Goal",
                status="done",
                progress=1.0,
                is_atomic=True,
                completion_mode="manual",
                depth_level=0,
                completed_at=datetime.now()
            )
            db.add(goal)
            await db.flush()

            # 2. Создаём approval
            approval = GoalCompletionApproval(
                goal_id=goal.id,
                approved_by="user:test",
                approved_at=datetime.now(),
                comment="Test approval"
            )
            db.add(approval)
            await db.commit()

            # 3. Проверяем инвариант
            result = await invariants_checker._check_manual_completion_has_approval()

            # 4. Должен PASS
            assert result["status"] == "PASS", f"Expected PASS, got {result}"
            assert "violations" not in result or len(result.get("violations", [])) == 0

    @pytest.mark.asyncio
    async def test_manual_done_without_approval_fail(self):
        """
        ❌ MANUAL done без approval → FAIL

        MANUAL цель в статусе done БЕЗ approval record
        должно FAIL инвариант I7.
        """
        async with AsyncSessionLocal() as db:
            # 1. Создаём MANUAL goal без approval
            goal = Goal(
                id=uuid.uuid4(),
                title="Test MANUAL Goal (No Approval)",
                status="done",
                progress=1.0,
                is_atomic=True,
                completion_mode="manual",
                depth_level=0,
                completed_at=datetime.now()
            )
            db.add(goal)
            await db.commit()

            # 2. Проверяем инвариант
            result = await invariants_checker._check_manual_completion_has_approval()

            # 3. Должен FAIL
            assert result["status"] == "VIOLATION", f"Expected VIOLATION, got {result}"
            assert len(result["violations"]) > 0, "Expected violations list"

            # 4. Проверяем, что наша цель в violations
            violation_ids = [v["goal_id"] for v in result["violations"]]
            assert str(goal.id) in violation_ids, "Our goal should be in violations"

            # Cleanup
            await db.delete(goal)
            await db.commit()

    @pytest.mark.asyncio
    async def test_aggregate_done_without_approval_pass(self):
        """
        ✅ AGGREGATE done без approval → PASS

        AGGREGATE цель в статусе done БЕЗ approval
        нормально, т.к. система auto-completes AGGREGATE цели.
        """
        async with AsyncSessionLocal() as db:
            # 1. Создаём AGGREGATE goal без approval
            goal = Goal(
                id=uuid.uuid4(),
                title="Test AGGREGATE Goal",
                status="done",
                progress=1.0,
                is_atomic=True,
                completion_mode="aggregate",
                depth_level=0,
                completed_at=datetime.now()
            )
            db.add(goal)
            await db.commit()

            # 2. Проверяем инвариант
            result = await invariants_checker._check_manual_completion_has_approval()

            # 3. Должен PASS (AGGREGATE не требует approval)
            assert result["status"] == "PASS", f"Expected PASS for AGGREGATE, got {result}"

            # Cleanup
            await db.delete(goal)
            await db.commit()

    @pytest.mark.asyncio
    async def test_manual_active_without_approval_pass(self):
        """
        ✅ MANUAL active без approval → PASS

        MANUAL цель в статусе active (ещё не done)
        не требует approval.
        """
        async with AsyncSessionLocal() as db:
            # 1. Создаём MANUAL goal в active
            goal = Goal(
                id=uuid.uuid4(),
                title="Test MANUAL Active Goal",
                status="active",
                progress=0.5,
                is_atomic=True,
                completion_mode="manual",
                depth_level=0
            )
            db.add(goal)
            await db.commit()

            # 2. Проверяем инвариант
            result = await invariants_checker._check_manual_completion_has_approval()

            # 3. Должен PASS (active не требует approval)
            assert result["status"] == "PASS", f"Expected PASS for active, got {result}"

            # Cleanup
            await db.delete(goal)
            await db.commit()


# =============================================================================
# RUN INSTRUCTIONS
# =============================================================================

"""
Запуск тестов:

```bash
# Все тесты I7
pytest services/core/tests/test_manual_completion_invariants.py -v

# Конкретный тест
pytest services/core/tests/test_manual_completion_invariants.py::TestManualCompletionInvariants::test_manual_done_with_approval_pass -v

# С coverage
pytest services/core/tests/test_manual_completion_invariants.py --cov=services/core --cov-report=html
```

Ожидаемый результат: ВСЕ PASSED
"""
