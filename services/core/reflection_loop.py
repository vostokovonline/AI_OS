"""
Reflection Loop (Phase 2.4.4)

Главный оркестратор системы рефлексии.

Workflow:
  Trigger → Signal → Policies → Decisions → Actions → Results

Это "мозг" системы, который:
- Получает сигналы от Observer/Execution/Manual
- Применяет политики для принятия решений
- Выполняет действия через Action Executor
- Хранит audit trail

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from reflection_system import (
    ReflectionSignal,
    ReflectionDecision,
    ReflectionContext,
    ReflectionTrigger,
    TriggerSource,
    create_trigger_id,
    validate_observer_signal,
    validate_execution_signal,
)
from reflection_policies import policy_engine
from reflection_actions import action_executor, ActionResult
from lifecycle_observer import InvariantViolationReport, invariant_registry
from observer_engine import observer_engine
from database import AsyncSessionLocal
from models import Goal
from sqlalchemy import select


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Reflection Loop Result
# =============================================================================

class ReflectionLoopResult:
    """
    Результат работы Reflection Loop.
    """

    def __init__(
        self,
        trigger_id: str,
        success: bool,
        decisions: List[ReflectionDecision],
        action_results: List[ActionResult],
        started_at: datetime,
        completed_at: datetime
    ):
        self.trigger_id = trigger_id
        self.success = success
        self.decisions = decisions
        self.action_results = action_results
        self.started_at = started_at
        self.completed_at = completed_at
        self.duration_seconds = (completed_at - started_at).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация."""
        return {
            "trigger_id": self.trigger_id,
            "success": self.success,
            "decisions_count": len(self.decisions),
            "decisions": [d.model_dump() for d in self.decisions],
            "actions_executed": len(self.action_results),
            "action_results": [r.to_dict() for r in self.action_results],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds
        }


# =============================================================================
# Reflection Loop (Main Orchestrator)
# =============================================================================

class ReflectionLoop:
    """
    Главный оркестратор системы рефлексии.

    НЕ содержит бизнес-логики!
    Только связывает компоненты:
    - Observer → Signal
    - Signal → Policies → Decisions
    - Decisions → Actions → Results
    """

    def __init__(self):
        self.policy_engine = policy_engine
        self.action_executor = action_executor
        self.logger = logger

    async def run(
        self,
        signal: ReflectionSignal,
        entity_state: Optional[Dict[str, Any]] = None,
        entity_history: Optional[List[Dict[str, Any]]] = None
    ) -> ReflectionLoopResult:
        """
        Запустить полный цикл рефлексии.

        Args:
            signal: Входной сигнал
            entity_state: Текущее состояние сущности (опционально)
            entity_history: История сущности (опционально)

        Returns:
            ReflectionLoopResult с решениями и результатами
        """
        trigger_id = signal.trigger_id
        started_at = datetime.now()

        self.logger.info(f"[{trigger_id}] Reflection Loop started: source={signal.source}")

        try:
            # Step 1: Validate signal
            self._validate_signal(signal)

            # Step 2: Enrich context (load entity state if not provided)
            if not entity_state and signal.entity_id:
                entity_state = await self._load_entity_state(signal.entity, signal.entity_id)

            if not entity_history and signal.entity_id:
                entity_history = await self._load_entity_history(signal.entity, signal.entity_id)

            # Step 3: Generate decisions using policies
            self.logger.info(f"[{trigger_id}] Generating decisions...")
            decisions = self.policy_engine.decide(
                signal=signal,
                entity_state=entity_state,
                entity_history=entity_history
            )

            self.logger.info(f"[{trigger_id}] Generated {len(decisions)} decisions")

            if not decisions:
                # No policies matched → no actions
                self.logger.info(f"[{trigger_id}] No policies matched, skipping actions")

                return ReflectionLoopResult(
                    trigger_id=trigger_id,
                    success=True,
                    decisions=[],
                    action_results=[],
                    started_at=started_at,
                    completed_at=datetime.now()
                )

            # Step 4: Execute actions
            self.logger.info(f"[{trigger_id}] Executing {len(decisions)} actions...")
            action_results = await self.action_executor.execute_batch(decisions)

            # Step 5: Update decisions with execution results
            for decision, result in zip(decisions, action_results):
                decision.executed = True
                decision.execution_result = result.to_dict()

            # Step 6: Store audit trail (optional, for debugging)
            await self._store_audit_trail(trigger_id, signal, decisions, action_results)

            # Determine success
            success = all(r.success for r in action_results)

            completed_at = datetime.now()

            self.logger.info(
                f"[{trigger_id}] Reflection Loop completed: "
                f"{len(decisions)} decisions, {len(action_results)} actions, "
                f"success={success}, duration={(completed_at - started_at).total_seconds():.2f}s"
            )

            return ReflectionLoopResult(
                trigger_id=trigger_id,
                success=success,
                decisions=decisions,
                action_results=action_results,
                started_at=started_at,
                completed_at=completed_at
            )

        except Exception as e:
            self.logger.error(f"[{trigger_id}] Reflection Loop failed: {e}")
            import traceback
            traceback.print_exc()

            return ReflectionLoopResult(
                trigger_id=trigger_id,
                success=False,
                decisions=[],
                action_results=[],
                started_at=started_at,
                completed_at=datetime.now()
            )

    async def run_from_observer_report(
        self,
        report: InvariantViolationReport
    ) -> ReflectionLoopResult:
        """
        Запустить рефлексию из отчёта Observer.

        Это основной способ запуска после обнаружения нарушения.
        """
        # Create signal from observer report
        trigger_id = create_trigger_id()
        signal = ReflectionSignal.from_observer_report(report, trigger_id)

        # Run loop
        return await self.run(signal)

    async def run_manual(
        self,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ReflectionLoopResult:
        """
        Запустить рефлексию вручную (админом).

        Для тестирования и принудительной проверки.
        """
        trigger_id = create_trigger_id()
        signal = ReflectionSignal.from_manual_trigger(reason, context or {}, trigger_id)

        return await self.run(signal)

    async def run_health_check(self) -> ReflectionLoopResult:
        """
        Запустить рефлексию как health check.

        Проверить систему и принять меры если нужно.
        """
        # Get health status from observer
        health = await observer_engine.get_health_status()

        # Create signal from health status
        trigger_id = create_trigger_id()

        if health["status"] == "HEALTHY":
            # No violations → no action needed
            signal = ReflectionSignal(
                source=TriggerSource.SCHEDULED,
                trigger_id=trigger_id,
                external_reason="Health check: system healthy",
                context={"health_status": health}
            )
        else:
            # Violations detected → create signal
            # Use first violation as signal (simplified)
            first_violation = health.get("violation_details", [{}])[0]

            signal = ReflectionSignal(
                source=TriggerSource.SCHEDULED,
                trigger_id=trigger_id,
                invariant_id=first_violation.get("invariant_id"),
                invariant_name=first_violation.get("invariant_name"),
                severity=InvariantSeverity.HARD if first_violation.get("severity") == "HARD" else InvariantSeverity.SOFT,
                entity=first_violation.get("entity"),
                entity_id=uuid.UUID(first_violation.get("entity_id", "00000000-0000-0000-0000-000000000000")),
                violation_details=first_violation.get("details", {}),
                external_reason=f"Health check: {health['violations_detected']} violations detected",
                context={"health_status": health}
            )

        return await self.run(signal)

    def _validate_signal(self, signal: ReflectionSignal):
        """Проверить валидность сигнала."""
        if signal.source == TriggerSource.OBSERVER:
            if not validate_observer_signal(signal):
                raise ValueError(f"Invalid observer signal: {signal}")

        elif signal.source == TriggerSource.EXECUTION:
            if not validate_execution_signal(signal):
                raise ValueError(f"Invalid execution signal: {signal}")

        # MANUAL, EXTERNAL, SCHEDULED — без валидации (любой контекст OK)

    async def _load_entity_state(
        self,
        entity_type: str,
        entity_id: uuid.UUID
    ) -> Optional[Dict[str, Any]]:
        """Загрузить состояние сущности из БД."""
        try:
            async with AsyncSessionLocal() as db:
                if entity_type == "goal":
                    stmt = select(Goal).where(Goal.id == entity_id)
                    result = await db.execute(stmt)
                    goal = result.scalar_one_or_none()

                    if goal:
                        return {
                            "id": str(goal.id),
                            "title": goal.title,
                            "status": goal.status,
                            "progress": goal.progress,
                            "completion_mode": goal.completion_mode,
                            "is_atomic": goal.is_atomic,
                            "depth_level": goal.depth_level,
                            "metadata": goal.metadata or {}
                        }

            return None

        except Exception as e:
            self.logger.warning(f"Failed to load entity state: {e}")
            return None

    async def _load_entity_history(
        self,
        entity_type: str,
        entity_id: uuid.UUID
    ) -> List[Dict[str, Any]]:
        """
        Загрузить историю сущности.

        Для Phase 2.4 — упрощённая версия (пустая).
        Future: загрузить из таблицы entity_history.
        """
        # TODO: Implement history loading
        # For now: return empty list
        return []

    async def _store_audit_trail(
        self,
        trigger_id: str,
        signal: ReflectionSignal,
        decisions: List[ReflectionDecision],
        action_results: List[ActionResult]
    ):
        """
        Сохранить audit trail в лог.

        Future: записывать в таблицу reflection_audit.
        """
        audit_entry = {
            "trigger_id": trigger_id,
            "signal_source": signal.source,
            "decisions_count": len(decisions),
            "actions_executed": len(action_results),
            "success": all(r.success for r in action_results),
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(f"Reflection Audit: {audit_entry}")

        # TODO: Store in reflection_audit table


# =============================================================================
# Singleton Instance
# =============================================================================

reflection_loop = ReflectionLoop()


# =============================================================================
# Convenience Functions
# =============================================================================

async def reflect_on_observer_report(report: InvariantViolationReport) -> ReflectionLoopResult:
    """
    Запустить рефлексию на отчёте Observer.

    Convenience функция для основных use cases.
    """
    return await reflection_loop.run_from_observer_report(report)


async def reflect_on_manual_trigger(reason: str, context: Optional[Dict[str, Any]] = None) -> ReflectionLoopResult:
    """
    Запустить рефлексию вручную.

    Convenience функция для админских действий.
    """
    return await reflection_loop.run_manual(reason, context)


async def reflect_on_health_check() -> ReflectionLoopResult:
    """
    Запустить рефлексию как health check.

    Convenience функция для периодических проверок.
    """
    return await reflection_loop.run_health_check()


# Fix: Import InvariantSeverity
from reflection_system import InvariantSeverity
