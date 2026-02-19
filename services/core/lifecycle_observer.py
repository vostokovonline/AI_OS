"""
Lifecycle Observer (Phase 2.3)

Read-only система для проверки инвариантов жизненного цикла целей.

НЕ меняет состояние!
Обнаруживает → Логирует → Сигнализирует

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from models import Goal, GoalCompletionApproval


# =============================================================================
# Invariant Severity
# =============================================================================

class InvariantSeverity(str, Enum):
    """
    Критичность нарушения инварианта.

    HARD — Нарушение архитектурного контракта (нельзя никогда)
         Пример: DONE без approval (I7)

    SOFT — Рекомендательное предупреждение
         Пример: Аномалия в данных (не ошибка, но странно)
    """
    HARD = "HARD"
    SOFT = "SOFT"


# =============================================================================
# Invariant Violation Report
# =============================================================================

class InvariantViolationReport:
    """
    Результат проверки инварианта.

    Используется для всех инвариантов (goals, approvals, artifacts, etc.)
    """

    def __init__(
        self,
        invariant_id: str,
        invariant_name: str,
        entity: str,
        entity_id: uuid.UUID,
        severity: InvariantSeverity,
        passed: bool,
        details: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None
    ):
        self.invariant_id = invariant_id
        self.invariant_name = invariant_name
        self.entity = entity  # "goal", "approval", "artifact", etc.
        self.entity_id = entity_id
        self.severity = severity
        self.passed = passed
        self.details = details or {}
        self.message = message or (f"✅ PASSED" if passed else f"❌ VIOLATED")
        self.detected_at = datetime.now()

    def to_dict(self) -> Dict:
        """Сериализация для API / логов"""
        return {
            "invariant_id": self.invariant_id,
            "invariant_name": self.invariant_name,
            "entity": self.entity,
            "entity_id": str(self.entity_id),
            "severity": self.severity.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "detected_at": self.detected_at.isoformat()
        }


# =============================================================================
# Invariant Check Interface
# =============================================================================

class InvariantCheck:
    """
    Базовый класс для проверки инварианта.

    Пример использования:
        class I7DoneHasExactlyOneApproval(InvariantCheck):
            async def check(self, goal_id: UUID) -> InvariantViolationReport:
                # реализация проверки
                pass
    """

    def __init__(self, invariant_id: str, invariant_name: str, severity: InvariantSeverity):
        self.invariant_id = invariant_id
        self.invariant_name = invariant_name
        self.severity = severity

    async def check(self, entity_id: uuid.UUID, context: Dict[str, Any] = None) -> InvariantViolationReport:
        """
        Проверить инвариант для одной сущности.

        Args:
            entity_id: UUID сущности для проверки
            context: Optional context (e.g., pre_commit_check flag)

        Returns:
            InvariantViolationReport с результатом
        """
        raise NotImplementedError("Subclasses must implement check()")

    async def check_all(self, limit: int = 1000) -> List[InvariantViolationReport]:
        """
        Проверить инвариант для всех сущностей.

        Args:
            limit: Макс. количество сущностей для проверки

        Returns:
            Список InvariantViolationReport
        """
        raise NotImplementedError("Subclasses must implement check_all()")


# =============================================================================
# Phase 2.3.1: Invariant Implementations (I7, I9, Orphans)
# =============================================================================

class I7DoneHasExactlyOneApproval(InvariantCheck):
    """
    I7: DONE goal must have exactly one approval.

    Инвариант:
    - IF goal.status = "done" AND goal.completion_mode = "manual"
    - THEN MUST exist exactly 1 approval in goal_completion_approvals

    Severity: HARD
    """

    def __init__(self):
        super().__init__(
            invariant_id="I7",
            invariant_name="DONE goal must have exactly one approval",
            severity=InvariantSeverity.HARD
        )

    async def check(self, entity_id: uuid.UUID, context: Dict[str, Any] = None) -> InvariantViolationReport:
        """Проверить I7 для одной цели"""
        async with AsyncSessionLocal() as db:
            # Load goal
            stmt = select(Goal).where(Goal.id == entity_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={"reason": "Goal not found"},
                    message="❌ GOAL_NOT_FOUND"
                )

            # Skip non-MANUAL goals
            if goal.completion_mode != "manual":
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,  # Not applicable
                    details={
                        "completion_mode": goal.completion_mode,
                        "reason": "Invariant only applies to MANUAL goals"
                    },
                    message="⚭ NOT_APPLICABLE (non-manual goal)"
                )

            # Check if DONE (unless this is a pre-commit check)
            is_pre_commit_check = context and context.get("pre_commit_check", False)

            if not is_pre_commit_check and goal.status != "done":
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={
                        "status": goal.status,
                        "reason": "Goal not DONE yet"
                    },
                    message="⚭ NOT_DONE (not applicable)"
                )

            # Check approvals count
            approval_stmt = select(GoalCompletionApproval).where(
                GoalCompletionApproval.goal_id == entity_id
            )
            approval_result = await db.execute(approval_stmt)
            approvals = approval_result.scalars().all()
            approval_count = len(approvals)

            # I7 check
            if approval_count == 1:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={
                        "approval_count": approval_count,
                        "approval_id": str(approvals[0].id),
                        "approved_at": approvals[0].approved_at.isoformat()
                    },
                    message="✅ PASSED"
                )
            elif approval_count == 0:
                # VIOLATION: DONE without approval
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={
                        "approval_count": 0,
                        "expected": 1,
                        "goal_status": goal.status,
                        "completion_mode": goal.completion_mode
                    },
                    message="❌ VIOLATED: DONE goal has no approval"
                )
            else:
                # VIOLATION: Multiple approvals (should be impossible due to UNIQUE constraint)
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={
                        "approval_count": approval_count,
                        "expected": 1,
                        "approval_ids": [str(a.id) for a in approvals]
                    },
                    message=f"❌ VIOLATED: DONE goal has {approval_count} approvals (expected 1)"
                )

    async def check_all(self, limit: int = 1000) -> List[InvariantViolationReport]:
        """Проверить I7 для всех MANUAL DONE целей"""
        async with AsyncSessionLocal() as db:
            # Find all MANUAL DONE goals
            stmt = select(Goal).where(
                and_(
                    Goal.status == "done",
                    Goal.completion_mode == "manual"
                )
            ).limit(limit)

            result = await db.execute(stmt)
            goals = result.scalars().all()

            reports = []
            for goal in goals:
                report = await self.check(goal.id)
                reports.append(report)

            return reports


class I9ApprovalImpliesDone(InvariantCheck):
    """
    I9: Approval must exist only for DONE goals.

    Инвариант:
    - IF approval exists in goal_completion_approvals
    - THEN goal.status MUST be "done"

    Severity: HARD
    """

    def __init__(self):
        super().__init__(
            invariant_id="I9",
            invariant_name="Approval must exist only for DONE goals",
            severity=InvariantSeverity.HARD
        )

    async def check(self, entity_id: uuid.UUID) -> InvariantViolationReport:
        """Проверить I9 для одного approval"""
        async with AsyncSessionLocal() as db:
            # Load approval
            stmt = select(GoalCompletionApproval).where(
                GoalCompletionApproval.id == entity_id
            )
            result = await db.execute(stmt)
            approval = result.scalar_one_or_none()

            if not approval:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="approval",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={"reason": "Approval not found"},
                    message="❌ APPROVAL_NOT_FOUND"
                )

            # Load goal
            goal_stmt = select(Goal).where(Goal.id == approval.goal_id)
            goal_result = await db.execute(goal_stmt)
            goal = goal_result.scalar_one_or_none()

            if not goal:
                # Orphaned approval (goal deleted?)
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="approval",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={
                        "goal_id": str(approval.goal_id),
                        "reason": "Goal not found (orphaned approval)"
                    },
                    message="❌ VIOLATED: Orphaned approval (goal missing)"
                )

            # I9 check
            if goal.status == "done":
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="approval",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={
                        "goal_id": str(goal.id),
                        "goal_status": goal.status,
                        "approved_at": approval.approved_at.isoformat()
                    },
                    message="✅ PASSED"
                )
            else:
                # VIOLATION: Approval for non-DONE goal
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="approval",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={
                        "goal_id": str(goal.id),
                        "goal_status": goal.status,
                        "approval_id": str(approval.id),
                        "approved_at": approval.approved_at.isoformat()
                    },
                    message=f"❌ VIOLATED: Approval exists for goal with status={goal.status}"
                )

    async def check_all(self, limit: int = 1000) -> List[InvariantViolationReport]:
        """Проверить I9 для всех approvals"""
        async with AsyncSessionLocal() as db:
            # Find all approvals
            stmt = select(GoalCompletionApproval).limit(limit)
            result = await db.execute(stmt)
            approvals = result.scalars().all()

            reports = []
            for approval in approvals:
                report = await self.check(approval.id)
                reports.append(report)

            return reports


class NoOrphanedDoneChildren(InvariantCheck):
    """
    Orphan Detection: No active children in DONE tree.

    Инвариант:
    - IF parent goal.status = "done"
    - THEN ALL children MUST have status in ["done", "completed"]

    Severity: HARD
    """

    def __init__(self):
        super().__init__(
            invariant_id="ORPHANS",
            invariant_name="No orphaned children in DONE tree",
            severity=InvariantSeverity.HARD
        )

    async def check(self, entity_id: uuid.UUID) -> InvariantViolationReport:
        """Проверить orphaned children для одной родительской цели"""
        async with AsyncSessionLocal() as db:
            # Load parent
            stmt = select(Goal).where(Goal.id == entity_id)
            result = await db.execute(stmt)
            parent = result.scalar_one_or_none()

            if not parent:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={"reason": "Parent goal not found"},
                    message="❌ PARENT_NOT_FOUND"
                )

            # Skip atomic goals (no children)
            if parent.is_atomic:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={"reason": "Atomic goal (no children)"},
                    message="⚭ NOT_APPLICABLE (atomic goal)"
                )

            # Skip non-DONE parents
            if parent.status != "done":
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={
                        "parent_status": parent.status,
                        "reason": "Parent not DONE"
                    },
                    message="⚭ NOT_DONE (not applicable)"
                )

            # Find children
            children_stmt = select(Goal).where(Goal.parent_id == entity_id)
            children_result = await db.execute(children_stmt)
            children = children_result.scalars().all()

            if not children:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={"child_count": 0},
                    message="✅ PASSED (no children)"
                )

            # Check for orphaned children
            orphaned = [
                child for child in children
                if child.status not in ["done", "completed"]
            ]

            if not orphaned:
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=True,
                    details={
                        "child_count": len(children),
                        "orphaned_count": 0
                    },
                    message="✅ PASSED"
                )
            else:
                # VIOLATION: Orphaned children found
                return InvariantViolationReport(
                    invariant_id=self.invariant_id,
                    invariant_name=self.invariant_name,
                    entity="goal",
                    entity_id=entity_id,
                    severity=self.severity,
                    passed=False,
                    details={
                        "child_count": len(children),
                        "orphaned_count": len(orphaned),
                        "orphaned_children": [
                            {
                                "child_id": str(child.id),
                                "child_title": child.title,
                                "child_status": child.status
                            }
                            for child in orphaned
                        ]
                    },
                    message=f"❌ VIOLATED: {len(orphaned)} orphaned children in DONE tree"
                )

    async def check_all(self, limit: int = 1000) -> List[InvariantViolationReport]:
        """Проверить всех non-atomic DONE родителей"""
        async with AsyncSessionLocal() as db:
            # Find all non-atomic DONE goals
            stmt = select(Goal).where(
                and_(
                    Goal.status == "done",
                    Goal.is_atomic == False
                )
            ).limit(limit)

            result = await db.execute(stmt)
            parents = result.scalars().all()

            reports = []
            for parent in parents:
                report = await self.check(parent.id)
                reports.append(report)

            return reports


# =============================================================================
# Phase 2.3.1: Invariant Registry
# =============================================================================

class InvariantRegistry:
    """
    Реестр всех инвариантов системы.

    Usage:
        registry = InvariantRegistry()
        report = await registry.check_invariant("I7", goal_id)
        all_reports = await registry.check_all_invariants()
    """

    def __init__(self):
        self._invariants: Dict[str, InvariantCheck] = {}

        # Register built-in invariants
        self.register(I7DoneHasExactlyOneApproval())
        self.register(I9ApprovalImpliesDone())
        self.register(NoOrphanedDoneChildren())

    def register(self, invariant: InvariantCheck):
        """Зарегистрировать инвариант"""
        self._invariants[invariant.invariant_id] = invariant

    def get_invariant(self, invariant_id: str) -> Optional[InvariantCheck]:
        """Получить инвариант по ID"""
        return self._invariants.get(invariant_id)

    def list_invariants(self) -> List[str]:
        """Список всех зарегистрированных инвариантов"""
        return list(self._invariants.keys())

    async def check_invariant(
        self,
        invariant_id: str,
        entity_id: uuid.UUID,
        context: Dict[str, Any] = None
    ) -> Optional[InvariantViolationReport]:
        """Проверить конкретный инвариант для сущности"""
        invariant = self.get_invariant(invariant_id)
        if not invariant:
            return None

        return await invariant.check(entity_id, context=context)

    async def check_all_invariants(
        self,
        limit: int = 1000
    ) -> Dict[str, List[InvariantViolationReport]]:
        """
        Проверить все инварианты для всех сущностей.

        Returns:
            {
                "I7": [InvariantViolationReport, ...],
                "I9": [InvariantViolationReport, ...],
                "ORPHANS": [InvariantViolationReport, ...]
            }
        """
        results = {}

        for invariant_id, invariant in self._invariants.items():
            reports = await invariant.check_all(limit=limit)
            results[invariant_id] = reports

        return results

    def get_summary(self, reports: Dict[str, List[InvariantViolationReport]]) -> Dict:
        """
        Сводка по результатам проверок.

        Returns:
            {
                "total_invariants": 3,
                "total_entities_checked": 150,
                "violations": {
                    "I7": {"count": 2, "severity": "HARD"},
                    "ORPHANS": {"count": 5, "severity": "HARD"}
                },
                "overall_status": "VIOLATIONS_DETECTED"
            }
        """
        total_entities = sum(len(reports) for reports in reports.values())

        violations = {}
        violation_count = 0

        for invariant_id, report_list in reports.items():
            violated = [r for r in report_list if not r.passed and r.severity == InvariantSeverity.HARD]
            if violated:
                violations[invariant_id] = {
                    "count": len(violated),
                    "severity": "HARD",
                    "invariant_name": report_list[0].invariant_name
                }
                violation_count += len(violated)

        overall_status = "HEALTHY" if violation_count == 0 else "VIOLATIONS_DETECTED"

        return {
            "total_invariants": len(self._invariants),
            "total_entities_checked": total_entities,
            "violations": violations,
            "violation_count": violation_count,
            "overall_status": overall_status,
            "checked_at": datetime.now().isoformat()
        }


# =============================================================================
# Singleton Instance
# =============================================================================

invariant_registry = InvariantRegistry()
