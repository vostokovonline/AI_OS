"""
Goal Execution Integration with Execution Orchestrator v1

Bridges Goal Executor v2 with Execution Orchestrator for unified tracking.

State Flow:
1. Goal submitted → create_execution (INIT)
2. Execution starts → transition to EXECUTING
3. Skill completes → transition to EXECUTED
4. Outcome recorded → transition to COMPLETE

Author: Claude (Control Center v3.1)
Date: 2026-03-03
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import text
from database import AsyncSessionLocal
from logging_config import get_logger
from infrastructure.uow import UnitOfWork, create_uow_provider

logger = get_logger(__name__)


class GoalExecutionIntegration:
    """
    Integration layer between Goal Executor v2 and Execution Orchestrator.

    Responsibilities:
    - Create execution records when goals are submitted
    - Track state transitions through execution lifecycle
    - Record outcomes when execution completes
    - Handle failures and timeouts

    NOT responsible for:
    - Complex recovery (orchestrator handles that)
    - Retry scheduling (retry_scheduler handles that)
    """

    def __init__(self):
        from execution.orchestrator import execution_orchestrator
        self.orchestrator = execution_orchestrator
        self._uow_provider = create_uow_provider()

    async def submit_goal_for_execution(
        self,
        goal_id: str,
        shadow_model: Optional[str] = None,
        actual_model: Optional[str] = None,
        is_divergent: bool = False,
        shadow_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Submit goal for execution through orchestrator.

        Flow:
        1. Check backpressure
        2. Create execution record (INIT state)
        3. Trigger goal execution (async side-effect)
        4. Return trace_id

        Args:
            goal_id: Goal to execute
            shadow_model: Shadow model decision (if CC enabled)
            actual_model: Actual model used
            is_divergent: Whether shadow decision was divergent
            shadow_confidence: Shadow model confidence

        Returns:
            Dict with trace_id and status
        """
        async with AsyncSessionLocal() as session:
            # Check backpressure
            can_start = await self.orchestrator.can_start_execution(session)
            if not can_start:
                logger.warning("execution_rejected_backpressure", goal_id=goal_id)
                return {
                    "status": "rejected",
                    "reason": "backpressure",
                    "message": "System at capacity (backpressure active)"
                }

            # Parse goal_id
            try:
                goal_uuid = uuid.UUID(goal_id)
            except ValueError:
                return {
                    "status": "error",
                    "reason": "invalid_goal_id",
                    "message": f"Invalid goal_id format: {goal_id}"
                }

            # Create execution (INIT state)
            trace_id = await self.orchestrator.create_execution(
                session,
                goal_id=goal_uuid,
                shadow_model=shadow_model,
                actual_model=actual_model,
                is_divergent=is_divergent,
                shadow_confidence=shadow_confidence
            )

            await session.commit()

            logger.info(
                "goal_submitted_for_execution",
                goal_id=goal_id,
                trace_id=str(trace_id),
                shadow_model=shadow_model,
                actual_model=actual_model
            )

            # Trigger async goal execution (side-effect, after state persists)
            # This will be picked up by the execution service

            return {
                "status": "submitted",
                "trace_id": str(trace_id),
                "goal_id": goal_id,
                "state": "INIT"
            }

    async def start_execution(
        self,
        trace_id: str,
        expected_state: str = "INIT"
    ) -> bool:
        """
        Transition execution to EXECUTING state.

        Called by worker when it starts processing the goal.

        Args:
            trace_id: Execution trace ID
            expected_state: Expected current state (default: INIT)

        Returns:
            True if transition succeeded
        """
        async with AsyncSessionLocal() as session:
            try:
                trace_uuid = uuid.UUID(trace_id)

                await self.orchestrator.transition(
                    session,
                    trace_id=trace_uuid,
                    to_state="EXECUTING",
                    expected_state=expected_state,
                    started_at=datetime.utcnow()
                )

                await session.commit()

                logger.info("execution_started", trace_id=trace_id)
                return True

            except Exception as e:
                logger.error("failed_to_start_execution", trace_id=trace_id, error=str(e))
                await session.rollback()
                return False

    async def complete_execution(
        self,
        trace_id: str,
        expected_state: str = "EXECUTING",
        outcome_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition execution to EXECUTED state.

        Called by worker when goal execution work is complete.

        Args:
            trace_id: Execution trace ID
            expected_state: Expected current state (default: EXECUTING)
            outcome_data: Optional execution outcome data

        Returns:
            True if transition succeeded
        """
        async with AsyncSessionLocal() as session:
            try:
                trace_uuid = uuid.UUID(trace_id)

                # Store outcome in execution_results table
                if outcome_data:
                    await self._store_outcome(session, trace_uuid, outcome_data, latency_ms=0, cost_usd=0.0, is_retry=False)

                await self.orchestrator.transition(
                    session,
                    trace_id=trace_uuid,
                    to_state="EXECUTED",
                    expected_state=expected_state,
                    executed_at=datetime.utcnow()
                )

                await session.commit()

                logger.info("execution_executed", trace_id=trace_id)
                return True

            except Exception as e:
                logger.error("failed_to_complete_execution", trace_id=trace_id, error=str(e))
                await session.rollback()
                return False

    async def finalize_execution(
        self,
        trace_id: str,
        expected_state: str = "EXECUTED",
        final_status: str = "COMPLETE"
    ) -> bool:
        """
        Transition execution to terminal COMPLETE state.

        Called after outcome is recorded and goal status is updated.

        Args:
            trace_id: Execution trace ID
            expected_state: Expected current state (default: EXECUTED)
            final_status: Final state (COMPLETE or CORRUPTED)

        Returns:
            True if transition succeeded
        """
        async with AsyncSessionLocal() as session:
            try:
                trace_uuid = uuid.UUID(trace_id)

                await self.orchestrator.transition(
                    session,
                    trace_id=trace_uuid,
                    to_state=final_status,
                    expected_state=expected_state,
                    completed_at=datetime.utcnow()
                )

                await session.commit()

                logger.info("execution_finalized", trace_id=trace_id, status=final_status)
                return True

            except Exception as e:
                logger.error("failed_to_finalize_execution", trace_id=trace_id, error=str(e))
                await session.rollback()
                return False

    async def fail_execution(
        self,
        trace_id: str,
        expected_state: str = "EXECUTING",
        failure_category: str = "EXECUTION_ERROR",
        error_message: Optional[str] = None
    ) -> bool:
        """
        Mark execution as failed (CORRUPTED state).

        Args:
            trace_id: Execution trace ID
            expected_state: Expected current state
            failure_category: Category of failure
            error_message: Optional error message

        Returns:
            True if transition succeeded
        """
        async with AsyncSessionLocal() as session:
            try:
                trace_uuid = uuid.UUID(trace_id)

                await self.orchestrator.transition(
                    session,
                    trace_id=trace_uuid,
                    to_state="CORRUPTED",
                    expected_state=expected_state,
                    executed_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    failure_category=failure_category
                )

                # Store error in execution_results
                if error_message:
                    await self._store_error(session, trace_uuid, error_message, failure_category)

                await session.commit()

                logger.error("execution_failed", trace_id=trace_id, category=failure_category)
                return True

            except Exception as e:
                logger.error("failed_to_fail_execution", trace_id=trace_id, error=str(e))
                await session.rollback()
                return False

    async def _store_outcome(
        self,
        session,
        trace_id: uuid.UUID,
        outcome_data: Dict[str, Any],
        latency_ms: int = 0,
        cost_usd: float = 0.0,
        is_retry: bool = False
    ):
        """Store execution outcome in execution_results table."""
        insert_query = text("""
            INSERT INTO execution_results (
                trace_id, result_payload, latency_ms, cost_usd, is_retry, completed_at
            ) VALUES (
                :trace_id, :result_payload, :latency_ms, :cost_usd, :is_retry, NOW()
            )
            ON CONFLICT (trace_id) DO UPDATE SET
                result_payload = EXCLUDED.result_payload,
                latency_ms = EXCLUDED.latency_ms,
                cost_usd = EXCLUDED.cost_usd,
                completed_at = NOW()
        """)

        import json

        await session.execute(insert_query, {
            "trace_id": str(trace_id),
            "result_payload": json.dumps(outcome_data),
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "is_retry": is_retry
        })

    async def _store_error(
        self,
        session,
        trace_id: uuid.UUID,
        error_message: str,
        failure_category: str
    ):
        """Store error information in execution_results table."""
        import json

        outcome_data = {
            "status": "error",
            "error": error_message,
            "failure_category": failure_category
        }

        await self._store_outcome(session, trace_id, outcome_data, latency_ms=0, cost_usd=0.0, is_retry=False)


# Global instance
goal_execution_integration = GoalExecutionIntegration()


class OrchestratedGoalExecutor:
    """
    Wrapper around GoalExecutorV2 that integrates with Execution Orchestrator.

    This is the RECOMMENDED way to execute atomic goals in production.

    Usage:
        executor = OrchestratedGoalExecutor()
        result = await executor.execute_goal(
            goal_id="uuid",
            session_id="optional"
        )
    """

    def __init__(self):
        from goal_executor_v2 import goal_executor_v2
        from execution.orchestrator import execution_orchestrator

        self.goal_executor = goal_executor_v2
        self.orchestrator = execution_orchestrator
        self.integration = goal_execution_integration
        self._uow_provider = create_uow_provider()

    async def execute_goal(
        self,
        goal_id: str,
        session_id: Optional[str] = None,
        shadow_model: Optional[str] = None,
        actual_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute goal with full orchestrator integration.

        Flow:
        1. Submit goal → create execution (INIT)
        2. Start execution → transition to EXECUTING
        3. Run goal executor v2 (with retry)
        4. Complete execution → transition to EXECUTED
        5. Finalize → transition to COMPLETE

        Args:
            goal_id: Goal to execute
            session_id: Optional session ID
            shadow_model: Shadow model decision
            actual_model: Actual model to use

        Returns:
            Execution result with trace_id
        """
        trace_id = None

        try:
            # Step 1: Submit goal (creates execution record)
            submit_result = await self.integration.submit_goal_for_execution(
                goal_id=goal_id,
                shadow_model=shadow_model,
                actual_model=actual_model
            )

            if submit_result["status"] == "rejected":
                return submit_result

            if submit_result["status"] == "error":
                return submit_result

            trace_id = submit_result["trace_id"]

            # Step 2: Start execution (transition to EXECUTING)
            started = await self.integration.start_execution(trace_id, expected_state="INIT")
            if not started:
                return {
                    "status": "error",
                    "message": "Failed to start execution",
                    "trace_id": trace_id
                }

            # Step 3: Execute goal through kernel
            async with self._uow_provider() as uow:
                from execution_dynamics import dispatch_goal
                _dispatch_result = await dispatch_goal(
                    goal_id=goal_id,
                    uow=uow,
                    dispatch_id=f"integ:{goal_id}",
                )
                # Wrap in ExecutionOutcome for downstream compatibility
                result = ExecutionOutcome(
                    status="completed" if _dispatch_result.get('success') else "failed",
                    confidence=1.0 if _dispatch_result.get('success') else 0.0,
                    attempts=1,
                    artifacts=_dispatch_result.get('artifacts', []),
                    error=_dispatch_result.get('error'),
                )

            # Step 4: Complete execution (transition to EXECUTED)
            # result is ExecutionOutcome dataclass, not dict
            if result.status == "completed":
                await self.integration.complete_execution(
                    trace_id,
                    expected_state="EXECUTING",
                    outcome_data={
                        "status": "completed",
                        "confidence": result.confidence,
                        "attempts": result.attempts,
                        "artifacts_count": len(result.artifacts) if result.artifacts else 0,
                        "goal_status": "success"
                    }
                )

                # Step 5: Finalize (transition to COMPLETE)
                await self.integration.finalize_execution(trace_id, expected_state="EXECUTED")

                return {
                    "status": "success",
                    "goal_id": goal_id,
                    "confidence": result.confidence,
                    "attempts": result.attempts,
                    "artifacts_produced": len(result.artifacts) if result.artifacts else 0,
                    "trace_id": trace_id,
                    "execution_state": "COMPLETE",
                    "goal_status": result.status
                }

            else:
                # Execution failed
                await self.integration.fail_execution(
                    trace_id,
                    expected_state="EXECUTING",
                    failure_category="EXECUTION_ERROR",
                    error_message=result.error if result.error else f"Execution failed with status: {result.status}"[:500]
                )

                return {
                    "status": "error",
                    "goal_id": goal_id,
                    "error": result.error if result.error else f"Execution failed: {result.status}",
                    "attempts": result.attempts,
                    "trace_id": trace_id,
                    "execution_state": "CORRUPTED",
                    "goal_status": result.status
                }

        except Exception as e:
            logger.error("orchestrated_execution_failed", goal_id=goal_id, trace_id=trace_id, error=str(e))

            # Try to mark as failed if we have a trace_id
            if trace_id:
                await self.integration.fail_execution(
                    trace_id,
                    expected_state=None,  # Don't check expected state on error
                    failure_category="SYSTEM_ERROR",
                    error_message=str(e)[:500]
                )

            return {
                "status": "error",
                "message": str(e),
                "trace_id": trace_id,
                "execution_state": "CORRUPTED"
            }


# Global instance
orchestrated_goal_executor = OrchestratedGoalExecutor()
