"""
Execution Use-Case v3.0 - Decision System with Arbitration

CRITICAL: This is the ONLY allowed entry point for goal execution.

ARCHITECTURE v3.0:
    PHASE 1: COLLECT DECISIONS (no mutations)
        - Select goals
        - Execute via pure function
        - Collect ExecutionIntent list

    PHASE 1.5: ARBITRATE (new!)
        - Estimate features (utility, cost, risk)
        - Policy selects subset
        - Log decision trace

    PHASE 2: APPLY BATCH ATOMICALLY
        - bulk_engine applies SELECTED intents only
        - ONE COMMIT for entire batch

    PHASE 3: EMIT FACTS (after commit)
        - Publish immutable events
        - Handlers decide what to do
"""
from typing import TYPE_CHECKING
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime

if TYPE_CHECKING:
    from infrastructure.uow import UnitOfWork


@dataclass
class ExecutionResult:
    """Result of batch execution with arbitration metrics"""
    total_found: int
    completed: int
    failed: int
    skipped: int  # Version mismatch or concurrent modification
    execution_time_ms: int

    # Arbitration metrics (new in v3.0)
    arbitration_selected: int = 0
    arbitration_rejected: int = 0
    arbitration_selection_rate: float = 0.0

    @classmethod
    def empty(cls) -> "ExecutionResult":
        return cls(
            total_found=0,
            completed=0,
            failed=0,
            skipped=0,
            execution_time_ms=0,
            arbitration_selected=0,
            arbitration_rejected=0,
            arbitration_selection_rate=0.0
        )


class GoalSelector:
    """Selects goals ready for execution"""

    async def select_ready(
        self,
        uow: "UnitOfWork",
        limit: int | None = None
    ) -> list[UUID]:
        from models import Goal
        from sqlalchemy import select

        stmt = select(Goal).where(
            Goal.is_atomic == True
        ).where(
            Goal.progress < 1.0
        )

        if limit:
            stmt = stmt.limit(limit)

        result = await uow.session.execute(stmt)
        goals = result.scalars().all()

        return [g.id for g in goals]


class ExecuteReadyGoalsUseCase:
    """
    Use-Case: Execute ready atomic goals (Decision System v3.0).

    CRITICAL INVARIANTS:
        1. Executor is PURE FUNCTION - no state changes
        2. Decisions collected FIRST, THEN arbitrated, THEN applied atomically
        3. Events emitted AFTER commit (immutable facts)
        4. Batch = ONE transaction, not N transactions
        5. Arbitration filters BEFORE application (new in v3.0)

    TRANSFORMATION:
        BEFORE: execute → transition → execute → transition (N commits)
        v2.0:   execute → execute → apply ALL (1 commit)
        v3.0:   execute → ARBITRATE → apply SELECTED (1 commit)

    This enables:
        - Arbitration over entire batch
        - Resource budgeting
        - Decision-making (not just execution)
        - Rollback of entire wave
        - Predictable DB load
    """

    def __init__(
        self,
        uow_factory,
        executor,
        bulk_engine,
        arbitrator,
        capital_allocator,
        event_bus=None
    ):
        self._uow_factory = uow_factory
        self._executor = executor
        self._bulk_engine = bulk_engine
        self._arbitrator = arbitrator
        self._capital_allocator = capital_allocator
        self._selector = GoalSelector()
        self._event_bus = event_bus

    async def run(
        self,
        *,
        limit: int | None = None,
        actor: str = "system"
    ) -> ExecutionResult:
        from application.events.execution_events import (
            GoalExecutionFinished,
            BatchExecutionCompleted
        )
        from application.execution.intents import ExecutionIntent

        # Write barrier control (test mode only)
        try:
            from tests.stress.write_barrier import WRITE_BARRIER
            WRITE_BARRIER.enable()  # Phase 1: NO writes allowed
        except ImportError:
            pass  # Not in test mode

        start_time = datetime.utcnow()

        # =====================================================================
        # PHASE 1: COLLECT DECISIONS (read-only execution)
        # =====================================================================
        intents: list[ExecutionIntent] = []

        # Use READ-ONLY UoW for execution (hard guarantee)
        from infrastructure.uow import get_uow
        readonly_uow_factory = lambda: get_uow(read_only=True)

        async with readonly_uow_factory() as read_uow:
            # Select candidates
            goal_ids = await self._selector.select_ready(read_uow, limit)

            if not goal_ids:
                return ExecutionResult.empty()

            # Execute goals (pure function, no mutations)
            for goal_id in goal_ids:
                try:
                    # Read goal snapshot for version check
                    from infrastructure.uow import GoalRepository
                    repo = GoalRepository()
                    goal_snapshot = await repo.get(read_uow.session, goal_id)

                    if not goal_snapshot:
                        # Goal disappeared between selection and execution
                        continue

                    # Execute with pure function
                    outcome = await self._executor.execute_goal(
                        goal_id=str(goal_id),
                        uow=read_uow  # Read-only access
                    )

                    # Convert outcome → intent (with optimistic lock)
                    from application.execution.intents import ArtifactData

                    artifact_data_list = [
                        ArtifactData(
                            artifact_type=a.get("type", "FILE"),
                            content_kind=a.get("content_kind", "file"),
                            content_location=a.get("content_location", ""),
                            verification_rule=a.get("verification_rule")
                        )
                        for a in outcome.artifacts
                    ]

                    intents.append(
                        ExecutionIntent(
                            goal_id=goal_id,
                            expected_version=goal_snapshot.updated_at,  # Optimistic lock
                            outcome=outcome.status,
                            confidence=outcome.confidence,
                            attempts=outcome.attempts,
                            artifacts=artifact_data_list,
                            error=None
                        )
                    )

                except Exception as e:
                    import logging
                    logger = logging.getLogger("use_cases")
                    logger.error(
                        f"execution_failed goal_id={str(goal_id)[:8]} error={str(e)[:100]}"
                    )

                    # Still create intent for error (without version since goal might not exist)
                    intents.append(
                        ExecutionIntent(
                            goal_id=goal_id,
                            expected_version=datetime.utcnow(),  # Dummy version for error
                            outcome="error",
                            confidence=0.0,
                            attempts=0,
                            artifacts=[],
                            error=str(e)[:100]
                        )
                    )

        # =====================================================================
        # PHASE 1.5: ARBITRATE (estimate → policy → select → log)
        # =====================================================================
        # Get current budget
        budget = await self._capital_allocator.current_budget()

        # Run arbitration pipeline
        arbitration_result = await self._arbitrator.evaluate(
            intents=intents,
            budget=budget
        )

        # Extract selected intents for application
        selected_intents = [s.intent for s in arbitration_result.selected]

        import logging
        logger = logging.getLogger("arbitration")
        logger.info(
            f"arbitration_completed: total={len(intents)}, "
            f"selected={len(selected_intents)}, "
            f"rejected={len(arbitration_result.rejected)}, "
            f"selection_rate={arbitration_result.selection_rate:.2f}, "
            f"budget_spent={arbitration_result.total_cost}, "
            f"budget_remaining={arbitration_result.budget_remaining}"
        )

        # =====================================================================
        # PHASE 2: APPLY BATCH ATOMICALLY (ONE transaction)
        # =====================================================================
        # Write barrier: Allow writes from this point
        try:
            from tests.stress.write_barrier import WRITE_BARRIER
            WRITE_BARRIER.allow()
        except ImportError:
            pass

        async with self._uow_factory() as write_uow:
            apply_result = await self._bulk_engine.apply_execution_intents(
                uow=write_uow,
                intents=selected_intents,  # ← ONLY selected intents applied
                actor=actor
            )
            # ← ONE COMMIT for entire batch here

        # =====================================================================
        # PHASE 3: EMIT FACTS (after commit, outside transaction)
        # =====================================================================
        if self._event_bus:
            # Publish individual completion events (for SELECTED intents only)
            for intent in selected_intents:
                await self._event_bus.publish(
                    GoalExecutionFinished(
                        goal_id=intent.goal_id,
                        status=intent.outcome,
                        confidence=intent.confidence,
                        attempts=intent.attempts,
                        artifacts_registered=len(intent.artifacts),
                        finished_at=datetime.utcnow(),
                        error_message=intent.error
                    )
                )

            # Publish batch completion event
            end_time = datetime.utcnow()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            batch_event = BatchExecutionCompleted(
                total_goals=len(selected_intents),  # ← Selected count, not total
                completed=apply_result["applied"],
                failed=apply_result["failed"],
                started_at=start_time,
                finished_at=end_time,
                execution_time_ms=execution_time_ms
            )
            await self._event_bus.publish(batch_event)

        return ExecutionResult(
            total_found=len(intents),
            completed=apply_result["applied"],
            failed=apply_result["failed"],
            skipped=apply_result.get("skipped", 0),
            execution_time_ms=execution_time_ms,
            arbitration_selected=len(selected_intents),
            arbitration_rejected=len(arbitration_result.rejected),
            arbitration_selection_rate=arbitration_result.selection_rate
        )
