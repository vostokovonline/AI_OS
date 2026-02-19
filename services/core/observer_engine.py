"""
Observer Engine (Phase 2.3.2)

Read-only движок для выполнения проверок инвариантов.

Режимы работы:
1. On-demand — по запросу (API call)
2. Scheduled — фоновая задача (Celery)
3. Post-mutation hook — будущий функционал

КРИТИЧЕСКИ ВАЖНО:
- ❌ НЕ меняет состояние системы
- ❌ НЕ делает auto-fix
- ✅ Только обнаружение + логирование + сигнализация

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Literal, Any
from sqlalchemy import select

from lifecycle_observer import (
    invariant_registry,
    InvariantViolationReport,
    InvariantSeverity
)
from database import AsyncSessionLocal
from models import Goal, GoalCompletionApproval


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Observer Result
# =============================================================================

class ObserverResult:
    """
    Результат работы Observer.

    Содержит сводку по всем проверкам инвариантов.
    """

    def __init__(
        self,
        run_id: str,
        started_at: datetime,
        completed_at: datetime,
        total_invariants: int,
        total_entities_checked: int,
        violation_reports: List[InvariantViolationReport],
        summary: Dict
    ):
        self.run_id = run_id
        self.started_at = started_at
        self.completed_at = completed_at
        self.duration_seconds = (completed_at - started_at).total_seconds()
        self.total_invariants = total_invariants
        self.total_entities_checked = total_entities_checked
        self.violation_reports = violation_reports
        self.summary = summary

    def to_dict(self) -> Dict:
        """Сериализация для API / логов"""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "total_invariants": self.total_invariants,
            "total_entities_checked": self.total_entities_checked,
            "violations_detected": len(self.violation_reports),
            "violation_details": [r.to_dict() for r in self.violation_reports],
            "summary": self.summary
        }


# =============================================================================
# Observer Engine
# =============================================================================

class ObserverEngine:
    """
    Read-only движок для проверки инвариантов жизненного цикла.

    НЕ меняет состояние!
    """

    def __init__(self, registry=invariant_registry):
        self.registry = registry
        self.logger = logger

    async def run_on_demand(
        self,
        invariant_ids: Optional[List[str]] = None,
        limit: int = 1000
    ) -> ObserverResult:
        """
        Запустить проверки по требованию (on-demand).

        Args:
            invariant_ids: Список ID инвариантов для проверки (None = все)
            limit: Макс. количество сущностей для проверки

        Returns:
            ObserverResult с полной статистикой
        """
        run_id = f"ondemand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        started_at = datetime.now()

        self.logger.info(f"[{run_id}] Starting on-demand observer run")

        # Определяем какие инварианты проверять
        if invariant_ids:
            invariants_to_check = invariant_ids
        else:
            invariants_to_check = self.registry.list_invariants()

        self.logger.info(f"[{run_id}] Checking invariants: {invariants_to_check}")

        # Запускаем проверки
        all_violations = []
        total_entities = 0

        for invariant_id in invariants_to_check:
            self.logger.info(f"[{run_id}] Running invariant: {invariant_id}")

            invariant = self.registry.get_invariant(invariant_id)
            if not invariant:
                self.logger.warning(f"[{run_id}] Invariant not found: {invariant_id}")
                continue

            # Run check_all for this invariant
            reports = await invariant.check_all(limit=limit)
            total_entities += len(reports)

            # Filter violations only
            violations = [r for r in reports if not r.passed]
            all_violations.extend(violations)

            self.logger.info(
                f"[{run_id}] Invariant {invariant_id}: "
                f"{len(reports)} entities checked, {len(violations)} violations"
            )

        completed_at = datetime.now()

        # Build summary
        summary = self.registry.get_summary(
            {inv_id: [] for inv_id in invariants_to_check}  # Placeholder
        )
        summary["total_entities_checked"] = total_entities
        summary["violations_detected"] = len(all_violations)

        if len(all_violations) > 0:
            summary["overall_status"] = "VIOLATIONS_DETECTED"
            self.logger.warning(
                f"[{run_id}] Observer completed: {len(all_violations)} violations detected"
            )
        else:
            summary["overall_status"] = "HEALTHY"
            self.logger.info(f"[{run_id}] Observer completed: ALL HEALTHY")

        return ObserverResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            total_invariants=len(invariants_to_check),
            total_entities_checked=total_entities,
            violation_reports=all_violations,
            summary=summary
        )

    async def check_specific_entity(
        self,
        invariant_id: str,
        entity_id: uuid.UUID,
        context: Dict[str, Any] = None
    ) -> Optional[InvariantViolationReport]:
        """
        Проверить конкретную сущность на конкретный инвариант.

        Args:
            invariant_id: ID инварианта (I7, I9, ORPHANS)
            entity_id: UUID сущности
            context: Optional context (e.g., pre_commit_check flag)

        Returns:
            InvariantViolationReport или None если инвариант не найден
        """
        self.logger.info(
            f"Checking entity {entity_id} against invariant {invariant_id}"
        )

        report = await self.registry.check_invariant(invariant_id, entity_id, context=context)

        if report:
            if report.passed:
                self.logger.info(f"✅ PASSED: {invariant_id} for {entity_id}")
            else:
                self.logger.warning(
                    f"❌ VIOLATED: {invariant_id} for {entity_id} - {report.message}"
                )

        return report

    async def check_goal(self, goal_id: uuid.UUID) -> Dict[str, InvariantViolationReport]:
        """
        Проверить все применимые инварианты для цели.

        Args:
            goal_id: UUID цели

        Returns:
            {invariant_id: InvariantViolationReport}
        """
        # Load goal to determine applicable invariants
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                self.logger.error(f"Goal not found: {goal_id}")
                return {}

        reports = {}

        # I7: applicable only for MANUAL goals
        if goal.completion_mode == "manual":
            i7_report = await self.check_specific_entity("I7", goal_id)
            if i7_report:
                reports["I7"] = i7_report

        # ORPHANS: applicable only for non-atomic parents
        if not goal.is_atomic:
            orphans_report = await self.check_specific_entity("ORPHANS", goal_id)
            if orphans_report:
                reports["ORPHANS"] = orphans_report

        return reports

    async def check_approval(self, approval_id: uuid.UUID) -> Dict[str, InvariantViolationReport]:
        """
        Проверить все применимые инварианты для approval.

        Args:
            approval_id: UUID approval

        Returns:
            {invariant_id: InvariantViolationReport}
        """
        reports = {}

        # I9: always applicable for approvals
        i9_report = await self.check_specific_entity("I9", approval_id)
        if i9_report:
            reports["I9"] = i9_report

        return reports

    async def get_health_status(self) -> Dict:
        """
        Быстрая проверка здоровья системы (без полного скана).

        Returns:
            {
                "status": "HEALTHY" | "DEGRADED" | "CRITICAL",
                "invariants_checked": 3,
                "last_check": "2026-02-06T14:32:11Z",
                "summary": {...}
            }
        """
        # Quick check: sample 10 random goals and 5 approvals
        async with AsyncSessionLocal() as db:
            # Sample goals
            goal_stmt = select(Goal).limit(10)
            goal_result = await db.execute(goal_stmt)
            goals = goal_result.scalars().all()

            # Sample approvals
            approval_stmt = select(GoalCompletionApproval).limit(5)
            approval_result = await db.execute(approval_stmt)
            approvals = approval_result.scalars().all()

        violations = []

        # Check I7 for MANUAL DONE goals
        for goal in goals:
            if goal.completion_mode == "manual" and goal.status == "done":
                report = await self.check_specific_entity("I7", goal.id)
                if report and not report.passed:
                    violations.append(report.to_dict())

        # Check I9 for approvals
        for approval in approvals:
            report = await self.check_specific_entity("I9", approval.id)
            if report and not report.passed:
                violations.append(report.to_dict())

        # Determine status
        if not violations:
            status = "HEALTHY"
        elif len(violations) < 3:
            status = "DEGRADED"
        else:
            status = "CRITICAL"

        return {
            "status": status,
            "invariants_checked": len(self.registry.list_invariants()),
            "entities_sampled": len(goals) + len(approvals),
            "violations_detected": len(violations),
            "violation_details": violations[:5],  # First 5
            "checked_at": datetime.now().isoformat()
        }


# =============================================================================
# Singleton Instance
# =============================================================================

observer_engine = ObserverEngine()
