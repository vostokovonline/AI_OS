"""
Execution-Reflection Integration (Phase 2.5.3)

Главный связывающий слой между Execution и Reflection.

Workflow:
  ExecutionEvent
    ↓
  ObserverTriggerEngine
    ↓
  Observer (checks invariants)
    ↓
  ReflectionLoop (decides actions)
    ↓
  ActionExecutor (executes actions)
    ↓
  Feedback → Execution (continue/stop/escalate)

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
import logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum

from execution_events import ExecutionEvent
from observer_trigger_rules import observer_trigger_engine
from reflection_loop import reflection_loop, reflect_on_observer_report
from reflection_system import (
    ReflectionSignal,
    TriggerSource,
    create_trigger_id,
)


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Execution Feedback
# =============================================================================

class ExecutionFeedback(str, Enum):
    """
    Обратная связь от Reflection к Execution.

    Определяет что должен делать Execution после получения
    результатов Reflection.
    """
    CONTINUE = "continue"  # Continue normal execution
    STOP = "stop"          # Stop execution (freeze/abort)
    ESCALATE = "escalate"  # Escalate to human/higher level
    RETRY = "retry"        # Retry the operation
    SKIP = "skip"          # Skip this step


# =============================================================================
# Integration Result
# =============================================================================

class IntegrationResult:
    """
    Результат интеграции Execution → Observer → Reflection.

    Содержит всю информацию о том, что произошло после события.
    """

    def __init__(
        self,
        event_id: str,
        observer_triggered: bool,
        violation_reports: List,
        reflection_triggered: bool,
        reflection_decisions: List,
        actions_executed: List,
        feedback: ExecutionFeedback,
        feedback_reason: str,
        processing_time_seconds: float
    ):
        self.event_id = event_id
        self.observer_triggered = observer_triggered
        self.violation_reports = violation_reports
        self.reflection_triggered = reflection_triggered
        self.reflection_decisions = reflection_decisions
        self.actions_executed = actions_executed
        self.feedback = feedback
        self.feedback_reason = feedback_reason
        self.processing_time_seconds = processing_time_seconds
        self.processed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация."""
        return {
            "event_id": self.event_id,
            "observer_triggered": self.observer_triggered,
            "violations_detected": len(self.violation_reports),
            "reflection_triggered": self.reflection_triggered,
            "decisions_count": len(self.reflection_decisions),
            "actions_executed": len(self.actions_executed),
            "feedback": self.feedback.value,
            "feedback_reason": self.feedback_reason,
            "processing_time_seconds": self.processing_time_seconds,
            "processed_at": self.processed_at.isoformat()
        }


# =============================================================================
# Execution-Reflection Integrator
# =============================================================================

class ExecutionReflectionIntegrator:
    """
    Главный интегратор Execution и Reflection.

    Это "мозг" который:
    1. Получает события от Executor
    2. Решает нужно ли запускать Observer
    3. Запускает Reflection если есть нарушения
    4. Возвращает feedback для Execution
    """

    def __init__(self):
        self.observer_trigger_engine = observer_trigger_engine
        self.reflection_loop = reflection_loop
        self.logger = logger

    async def process_event(
        self,
        event: ExecutionEvent
    ) -> IntegrationResult:
        """
        Обработать событие выполнения через Observer → Reflection.

        Args:
            event: ExecutionEvent от Executor

        Returns:
            IntegrationResult с feedback для Execution
        """
        started_at = datetime.now()

        self.logger.info(
            f"Processing execution event: {event.event_type.value} for goal {event.goal_id}"
        )

        try:
            # Step 1: Check if Observer should be triggered
            if not self.observer_trigger_engine.should_trigger_observer(event):
                # No observer needed → continue execution
                return IntegrationResult(
                    event_id=event.event_id,
                    observer_triggered=False,
                    violation_reports=[],
                    reflection_triggered=False,
                    reflection_decisions=[],
                    actions_executed=[],
                    feedback=ExecutionFeedback.CONTINUE,
                    feedback_reason="No observer rules matched",
                    processing_time_seconds=(datetime.now() - started_at).total_seconds()
                )

            # Step 2: Run Observer checks
            self.logger.info(f"Observer triggered for event {event.event_id}")
            violation_reports = await self.observer_trigger_engine.trigger_from_event(event)

            self.logger.info(f"Observer found {len(violation_reports)} violations")

            if not violation_reports:
                # No violations → continue execution
                return IntegrationResult(
                    event_id=event.event_id,
                    observer_triggered=True,
                    violation_reports=[],
                    reflection_triggered=False,
                    reflection_decisions=[],
                    actions_executed=[],
                    feedback=ExecutionFeedback.CONTINUE,
                    feedback_reason="Observer checks passed (no violations)",
                    processing_time_seconds=(datetime.now() - started_at).total_seconds()
                )

            # Step 3: Run Reflection for each violation
            self.logger.info(f"Running reflection for {len(violation_reports)} violations")

            reflection_loop_result = await self._run_reflection_on_reports(violation_reports)

            # Step 4: Determine feedback based on decisions
            feedback, feedback_reason = self._determine_feedback(
                reflection_loop_result.decisions
            )

            completed_at = datetime.now()

            self.logger.info(
                f"Integration completed: "
                f"violations={len(violation_reports)}, "
                f"decisions={len(reflection_loop_result.decisions)}, "
                f"feedback={feedback.value}"
            )

            return IntegrationResult(
                event_id=event.event_id,
                observer_triggered=True,
                violation_reports=violation_reports,
                reflection_triggered=True,
                reflection_decisions=reflection_loop_result.decisions,
                actions_executed=reflection_loop_result.action_results,
                feedback=feedback,
                feedback_reason=feedback_reason,
                processing_time_seconds=(completed_at - started_at).total_seconds()
            )

        except Exception as e:
            self.logger.error(f"Integration failed for event {event.event_id}: {e}")
            import traceback
            traceback.print_exc()

            # On error: safest action is to escalate
            return IntegrationResult(
                event_id=event.event_id,
                observer_triggered=False,
                violation_reports=[],
                reflection_triggered=False,
                reflection_decisions=[],
                actions_executed=[],
                feedback=ExecutionFeedback.ESCALATE,
                feedback_reason=f"Integration error: {str(e)}",
                processing_time_seconds=(datetime.now() - started_at).total_seconds()
            )

    async def _run_reflection_on_reports(
        self,
        violation_reports: List
    ) -> Any:
        """
        Запустить Reflection для списка violation reports.

        Args:
            violation_reports: List of InvariantViolationReport

        Returns:
            ReflectionLoopResult
        """
        # For Phase 2.5: Run reflection on first violation only
        # Future: Could batch or prioritize

        first_report = violation_reports[0]
        result = await reflect_on_observer_report(first_report)

        return result

    def _determine_feedback(
        self,
        decisions: List
    ) -> tuple[ExecutionFeedback, str]:
        """
        Определить feedback для Execution на основе решений Reflection.

        Args:
            decisions: List of ReflectionDecision

        Returns:
            (ExecutionFeedback, reason)
        """
        if not decisions:
            return ExecutionFeedback.CONTINUE, "No decisions from reflection"

        # Check if any decision requires STOP
        from reflection_system import ActionType

        stop_actions = [ActionType.FREEZE]
        escalate_actions = [ActionType.ESCALATE, ActionType.REQUEST_REVIEW]

        for decision in decisions:
            if decision.action in stop_actions:
                return ExecutionFeedback.STOP, f"Decision {decision.action} requires stop"

            if decision.action in escalate_actions:
                return ExecutionFeedback.ESCALATE, f"Decision {decision.action} requires escalation"

        # Default: continue
        return ExecutionFeedback.CONTINUE, "No blocking decisions"

    async def process_event_with_context(
        self,
        event: ExecutionEvent,
        goal_context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Обработать событие с дополнительным контекстом о цели.

        Args:
            event: ExecutionEvent
            goal_context: Additional context (completion_mode, is_atomic, etc.)

        Returns:
            IntegrationResult
        """
        # Enrich event with context
        if goal_context:
            event.context.update(goal_context)

        return await self.process_event(event)


# =============================================================================
# Convenience Functions
# =============================================================================

async def process_execution_event(
    event: ExecutionEvent,
    goal_context: Optional[Dict[str, Any]] = None
) -> IntegrationResult:
    """
    Обработать execution event через полный цикл Observer → Reflection.

    Convenience функция для использования в goal_executor.

    Args:
        event: ExecutionEvent
        goal_context: Optional additional context

    Returns:
        IntegrationResult с feedback
    """
    integrator = ExecutionReflectionIntegrator()
    return await integrator.process_event_with_context(event, goal_context)


# =============================================================================
# Singleton Instance
# =============================================================================

execution_reflection_integrator = ExecutionReflectionIntegrator()


# =============================================================================
# Executor Hook Helper
# =============================================================================

class ExecutorHook:
    """
    Helper для интеграции с existing goal_executor.

    Использование:
        hook = ExecutorHook()

        # В goal_executor:
        hook.on_step_completed(goal_id, step_id, step_number, artifacts)
        feedback = hook.get_feedback()
        if feedback == STOP:
            # Stop execution
    """

    def __init__(self):
        self.integrator = execution_reflection_integrator
        self.last_feedback: Optional[ExecutionFeedback] = None
        self.last_feedback_reason: Optional[str] = None

    async def before_marking_done(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        completion_mode: str,
        evaluation_passed: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Pre-commit check before marking goal as DONE.

        This is a CRITICAL safety check that runs before setting status="done".

        Args:
            goal_id: Goal UUID
            goal_title: Goal title
            completion_mode: Completion mode (manual/aggregate)
            evaluation_passed: Whether evaluation passed
            context: Additional context

        Returns:
            IntegrationResult with feedback (should check should_stop() after calling)
        """
        from execution_events import execution_event_emitter

        # Emit pre-commit check event
        event = execution_event_emitter.emit_step_completed(
            goal_id=goal_id,
            goal_title=goal_title,
            step_id="pre_done_check",
            step_number=999,  # Special step number
            agent_role="SYSTEM",
            context={
                "pre_commit_check": True,
                "completion_mode": completion_mode,
                "evaluation_passed": evaluation_passed,
                **(context or {})
            }
        )

        # Process through integrator
        result = await self.integrator.process_event(event)

        # Store feedback for should_stop() check
        self.last_feedback = result.feedback
        self.last_feedback_reason = result.feedback_reason

        return result

    async def on_goal_completed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        steps_total: int,
        steps_completed: int,
        steps_failed: int,
        artifacts: List[str] = None,
        metrics: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> IntegrationResult:
        """Call when goal completed."""
        from execution_events import execution_event_emitter

        event = execution_event_emitter.emit_goal_completed(
            goal_id=goal_id,
            goal_title=goal_title,
            steps_total=steps_total,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            artifacts=artifacts,
            metrics=metrics,
            context=context or {}
        )

        result = await self.integrator.process_event(event)

        self.last_feedback = result.feedback
        self.last_feedback_reason = result.feedback_reason

        return result

    async def on_goal_failed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        steps_total: int,
        steps_completed: int,
        failure_reason: str,
        error_type: str = None,
        error_message: str = None,
        context: Dict[str, Any] = None
    ) -> IntegrationResult:
        """Call when goal failed."""
        from execution_events import execution_event_emitter

        event = execution_event_emitter.emit_goal_failed(
            goal_id=goal_id,
            goal_title=goal_title,
            steps_total=steps_total,
            steps_completed=steps_completed,
            failure_reason=failure_reason,
            error_type=error_type,
            error_message=error_message,
            context=context or {}
        )

        result = await self.integrator.process_event(event)

        self.last_feedback = result.feedback
        self.last_feedback_reason = result.feedback_reason

        return result

    async def on_step_completed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        step_id: str,
        step_number: int,
        agent_role: str,
        artifacts: List[str] = None,
        metrics: Dict[str, Any] = None,
        context: Dict[str, Any] = None
    ) -> IntegrationResult:
        """Call when step completed."""
        from execution_events import execution_event_emitter

        event = execution_event_emitter.emit_step_completed(
            goal_id=goal_id,
            goal_title=goal_title,
            step_id=step_id,
            step_number=step_number,
            agent_role=agent_role,
            artifacts=artifacts,
            metrics=metrics,
            context=context or {}
        )

        result = await self.integrator.process_event(event)

        self.last_feedback = result.feedback
        self.last_feedback_reason = result.feedback_reason

        return result

    def get_feedback(self) -> Optional[ExecutionFeedback]:
        """Get last feedback."""
        return self.last_feedback

    def get_feedback_reason(self) -> Optional[str]:
        """Get last feedback reason."""
        return self.last_feedback_reason

    def should_continue(self) -> bool:
        """Check if execution should continue."""
        return self.last_feedback in [None, ExecutionFeedback.CONTINUE]

    def should_stop(self) -> bool:
        """Check if execution should stop."""
        return self.last_feedback == ExecutionFeedback.STOP

    def should_escalate(self) -> bool:
        """Check if execution should escalate."""
        return self.last_feedback == ExecutionFeedback.ESCALATE


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ExecutionFeedback",
    "IntegrationResult",
    "ExecutionReflectionIntegrator",
    "ExecutorHook",
    "execution_reflection_integrator",
    "process_execution_event",
]
