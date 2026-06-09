"""
Goal Execution Service - Pure Execution Logic
=============================================

Responsibility:
    Execute goals (atomic and complex) without state management

Does NOT:
    - Transition goal state
    - Commit transactions
    - Update beliefs
    - Manage UoW lifecycle

Author: AI-OS Architecture v2.0
Date: 2026-03-10
"""

from typing import Optional, Dict, Any, List
from uuid import UUID
from dataclasses import dataclass, field
from datetime import datetime, timezone

from logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """
    Result of goal execution.

    Pure data structure - no behavior.
    """
    success: bool
    goal_id: UUID
    execution_type: str  # "atomic" | "complex"

    # Artifacts produced
    artifacts: List[Dict[str, Any]] = field(default_factory=list)

    # Subgoals created (for complex goals)
    subgoal_ids: List[UUID] = field(default_factory=list)

    # Evidence for evaluation
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "success": self.success,
            "goal_id": str(self.goal_id),
            "execution_type": self.execution_type,
            "artifacts": self.artifacts,
            "subgoal_ids": [str(sid) for sid in self.subgoal_ids],
            "evidence": self.evidence,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "error_type": self.error_type
        }


class GoalExecutionService:
    """
    Pure domain service for goal execution.

    Separates execution logic from:
    - State management (GoalTransitionService)
    - Transaction management (UnitOfWork)
    - Belief updates (CompletionEngine)
    """

    def __init__(self):
        # Dependencies (lazy loaded)
        self._atomic_executor = None
        self._complex_executor = None
        self._decomposer = None

    async def execute(
        self,
        uow: "UnitOfWork",
        goal: "Goal",
        session_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a goal (atomic or complex).

        Args:
            uow: UnitOfWork with active session
            goal: Goal entity to execute
            session_id: Optional chat session ID

        Returns:
            ExecutionResult: Pure result data

        Raises:
            ValueError: If goal execution invalid
            RuntimeError: If execution fails

        Note:
            Does NOT modify goal state.
            Does NOT commit transaction.
            Caller responsible for state transition.

        Example:
            >>> service = GoalExecutionService()
            >>> result = await service.execute(uow, goal)
            >>> if result.success:
            ...     # Use transition_service to update state
            ...     await transition_service.transition(uow, goal.id, "done")
        """
        logger.info(
            "goal_execution_start",
            goal_id=str(goal.id),
            goal_type=goal.goal_type,
            is_atomic=goal.is_atomic
        )

        started_at = datetime.now(timezone.utc)

        try:
            # Route to appropriate executor
            if goal.is_atomic:
                result = await self._execute_atomic(uow, goal, session_id)
            else:
                result = await self._execute_complex(uow, goal, session_id)

            # Calculate duration
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            result.started_at = started_at
            result.completed_at = completed_at
            result.duration_ms = duration_ms

            logger.info(
                "goal_execution_complete",
                goal_id=str(goal.id),
                success=result.success,
                duration_ms=duration_ms,
                artifacts_count=len(result.artifacts),
                subgoals_count=len(result.subgoal_ids)
            )

            return result

        except Exception as e:
            logger.error(
                "goal_execution_failed",
                goal_id=str(goal.id),
                error_type=type(e).__name__,
                error_message=str(e)
            )

            return ExecutionResult(
                success=False,
                goal_id=goal.id,
                execution_type="atomic" if goal.is_atomic else "complex",
                error_message=str(e),
                error_type=type(e).__name__,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc)
            )

    async def _execute_atomic(
        self,
        uow: "UnitOfWork",
        goal: "Goal",
        session_id: Optional[str]
    ) -> ExecutionResult:
        """
        Execute atomic goal via skills.

        Atomic goals:
        - Are executable tasks
        - Produce artifacts
        - Have clear success criteria

        Returns:
            ExecutionResult with artifacts and evidence
        """
        logger.debug(
            "executing_atomic_goal",
            goal_id=str(goal.id),
            title=goal.title
        )

        # Execute through kernel (lease + journal + dynamics)
        from execution_dynamics import dispatch_goal
        outcome = await dispatch_goal(
            goal_id=str(goal.id),
            dispatch_id=f"exec_svc:{goal.id}",
        )

        # Extract artifacts from outcome
        artifacts = []
        if outcome and "artifacts" in outcome:
            artifacts = outcome["artifacts"]

        # Build evidence
        evidence = {
            "execution_outcome": outcome,
            "has_passed_artifacts": any(
                a.get("verification_status") == "passed"
                for a in artifacts
            )
        }

        return ExecutionResult(
            success=outcome.get("status") == "success" if outcome else False,
            goal_id=goal.id,
            execution_type="atomic",
            artifacts=artifacts,
            evidence=evidence
        )

    async def _execute_complex(
        self,
        uow: "UnitOfWork",
        goal: "Goal",
        session_id: Optional[str]
    ) -> ExecutionResult:
        """
        Execute complex goal via decomposition.

        Complex goals:
        - Require decomposition
        - Have subgoals
        - May use agent graph

        Returns:
            ExecutionResult with subgoal IDs
        """
        if self._decomposer is None:
            from goal_decomposer import goal_decomposer
            self._decomposer = goal_decomposer

        logger.debug(
            "executing_complex_goal",
            goal_id=str(goal.id),
            title=goal.title
        )

        # Decompose goal
        decomposition = await self._decomposer.decompose_goal(
            goal_id=str(goal.id),
            goal=goal
        )

        # Extract subgoal IDs
        subgoal_ids = []
        if decomposition and "subgoals" in decomposition:
            subgoal_ids = [
                UUID(sg["id"]) if isinstance(sg["id"], str) else sg["id"]
                for sg in decomposition["subgoals"]
            ]

        # Build evidence
        evidence = {
            "decomposition_result": decomposition,
            "subgoal_count": len(subgoal_ids)
        }

        return ExecutionResult(
            success=len(subgoal_ids) > 0,
            goal_id=goal.id,
            execution_type="complex",
            subgoal_ids=subgoal_ids,
            evidence=evidence
        )

    async def execute_with_agent_graph(
        self,
        uow: "UnitOfWork",
        goal: "Goal",
        user_request: str
    ) -> ExecutionResult:
        """
        Execute goal using LangGraph agent orchestration.

        For complex goals requiring multi-agent collaboration.

        Args:
            uow: UnitOfWork
            goal: Goal to execute
            user_request: Original user request

        Returns:
            ExecutionResult from agent execution
        """
        from agent_graph import app_graph
        from langchain_core.messages import HumanMessage

        logger.debug(
            "executing_with_agent_graph",
            goal_id=str(goal.id),
            user_request=user_request[:100]
        )

        # Run agent graph
        try:
            result = await app_graph.ainvoke({
                "messages": [HumanMessage(content=user_request)],
                "next_agent": "SUPERVISOR",
                "retry_count": 0,
                "loop_count": 0,
                "last_error": ""
            })

            # Extract execution trace
            execution_trace = {
                "agent_graph_result": result,
                "message_count": len(result.get("messages", []))
            }

            return ExecutionResult(
                success=True,
                goal_id=goal.id,
                execution_type="agent_graph",
                evidence={"execution_trace": execution_trace}
            )

        except Exception as e:
            logger.error(
                "agent_graph_execution_failed",
                goal_id=str(goal.id),
                error=str(e)
            )

            return ExecutionResult(
                success=False,
                goal_id=goal.id,
                execution_type="agent_graph",
                error_message=str(e),
                error_type=type(e).__name__
            )


# Singleton instance
goal_execution_service = GoalExecutionService()
