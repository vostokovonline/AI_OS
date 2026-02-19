"""
Production Wiring: Execution Feedback Loop Integration

Это ПОСЛЕДНИЙ кусок - подключение ExecutorHook к goal_executor_v2.

КРИТИЧЕСКИ ВАЖНО:
- НЕ меняем логику execution
- ТОЛЬКО добавляем хуки
- Обрабатываем ExecutionMustStopException

Author: AI-OS Core Team
Date: 2026-02-06
"""

from typing import Optional
from execution_reflection_integration import ExecutorHook
from execution_safety_contract import ExecutionMustStopException


class ExecutorWithFeedback:
    """
    Wrapper для GoalExecutorV2 с feedback loop.

    Добавляет хуки в критические точки execution.
    """

    def __init__(self):
        self.hook = ExecutorHook()

    async def on_skill_execution_completed(
        self,
        goal_id: str,
        goal_title: str,
        step_id: str,
        success: bool,
        error: Optional[str] = None
    ):
        """
        Хук после выполнения skill (шаг 4).

        Triggered: После каждого skill.execute()
        Purpose: Проверить что execution должен продолжаться
        """
        from execution_events import execution_event_emitter
        import uuid

        if success:
            # Emit step completed event
            event = execution_event_emitter.emit_step_completed(
                goal_id=uuid.UUID(goal_id),
                goal_title=goal_title,
                step_id=step_id,
                step_number=1,  # Упрощено для v2 (atomic goals have 1 step)
                agent_role="SKILL",
                context={"step_phase": "skill_execution"}
            )

            result = await self.hook.on_step_completed(
                goal_id=uuid.UUID(goal_id),
                goal_title=goal_title,
                step_id=step_id,
                step_number=1,
                agent_role="SKILL",
                context={"step_phase": "skill_execution"}
            )
        else:
            # Emit step failed event
            from execution_events import ExecutionEventType
            event = execution_event_emitter.emit_step_failed(
                goal_id=uuid.UUID(goal_id),
                goal_title=goal_title,
                step_id=step_id,
                step_number=1,
                agent_role="SKILL",
                error_type="SkillExecutionError",
                error_message=error or "Unknown error",
                context={"step_phase": "skill_execution"}
            )

            # Process event (might trigger reflection on failure)
            result = await self.hook.integrator.process_event(event)

        # Check if must stop
        if self.hook.should_stop():
            raise ExecutionMustStopException(
                message=f"Feedback loop requires stop after step: {step_id}",
                safety_level=self.hook.last_feedback,
                integration_result=result
            )

        if self.hook.should_escalate():
            # Escalate but don't stop execution yet
            # (let executor finish gracefully)
            pass

    async def before_marking_done(
        self,
        goal_id: str,
        goal_title: str,
        completion_mode: str,
        evaluation_passed: bool
    ):
        """
        Хук ПЕРЕД Mark as DONE (критически важно!).

        Triggered: Перед goal.status = "done"
        Purpose: Проверить инварианты (I7 для MANUAL целей)

        КРИТИЧЕСКОЕ МЕСТО: Это "pre-commit check"
        """
        import uuid

        # Use ExecutorHook's before_marking_done method
        result = await self.hook.before_marking_done(
            goal_id=uuid.UUID(goal_id),
            goal_title=goal_title,
            completion_mode=completion_mode,
            evaluation_passed=evaluation_passed
        )

        # DEBUG: Log the result
        print(f"   🔍 Pre-commit check result:")
        print(f"      - observer_triggered: {result.observer_triggered}")
        print(f"      - violations: {len(result.violation_reports)}")
        print(f"      - feedback: {result.feedback.value}")
        print(f"      - feedback_reason: {result.feedback_reason}")

        # Check if must stop (e.g., I7 violation for MANUAL goal)
        if self.hook.should_stop():
            print(f"   🛑 Pre-commit check: MUST STOP")
            raise ExecutionMustStopException(
                message=f"Pre-commit check failed: cannot mark as DONE",
                safety_level=self.hook.last_feedback,
                integration_result=result
            )

        print(f"   ✅ Pre-commit check: PASSED (will continue)")

        # Safe to proceed with marking as done
        return True

    async def on_goal_completed(
        self,
        goal_id: str,
        goal_title: str,
        steps_total: int,
        steps_completed: int,
        steps_failed: int,
        artifacts: list,
        completion_mode: str
    ):
        """
        Хук ПОСЛЕ mark as DONE.

        Triggered: После goal.status = "done"
        Purpose: Observer checks + Reflection (если нужно)
        """
        # Emit goal completed event
        result = await self.hook.on_goal_completed(
            goal_id=goal_id,
            goal_title=goal_title,
            steps_total=steps_total,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            artifacts=artifacts,
            context={
                "completion_mode": completion_mode
            }
        )

        # Log feedback
        if result.observer_triggered:
            if result.violation_reports:
                print(f"   ⚠️  Observer detected {len(result.violation_reports)} violations after DONE")
            else:
                print(f"   ✅ Observer check passed (no violations)")

        if result.reflection_triggered:
            print(f"   🤔 Reflection made {result.reflection_decisions} decisions")
            print(f"   📊 Feedback: {result.feedback.value} - {result.feedback_reason}")

        return result

    async def on_goal_failed(
        self,
        goal_id: str,
        goal_title: str,
        steps_total: int,
        steps_completed: int,
        failure_reason: str,
        error_type: str = None,
        error_message: str = None
    ):
        """
        Хук при failure execution.

        Triggered: Когда goal execution не удается
        Purpose: Observer checks + Reflection
        """
        result = await self.hook.on_goal_failed(
            goal_id=goal_id,
            goal_title=goal_title,
            steps_total=steps_total,
            steps_completed=steps_completed,
            failure_reason=failure_reason,
            error_type=error_type,
            error_message=error_message
        )

        print(f"   🔴 Goal failed - Observer triggered: {result.observer_triggered}")
        print(f"   📊 Feedback: {result.feedback.value} - {result.feedback_reason}")

        return result


# Singleton instance
executor_with_feedback = ExecutorWithFeedback()
