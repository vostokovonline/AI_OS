"""
Reflection Actions (Phase 2.4.3)

Orchestration действий, предпринимаемых Reflection.

КРИТИЧЕСКИ ВАЖНО:
- ❌ NO direct DB UPDATE/DELETE
- ✅ YES orchestration (call services, create tasks, etc.)
- ✅ Actions are idempotent where possible

Architecture:
  Decision → Action Executor → Service Call → Result

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
import logging
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from reflection_system import (
    ReflectionDecision,
    ActionType,
)
from database import AsyncSessionLocal
from models import Goal
from sqlalchemy import select, update


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Action Result
# =============================================================================

class ActionResult:
    """
    Результат выполнения действия.
    """

    def __init__(
        self,
        success: bool,
        action_type: ActionType,
        target_entity: str,
        target_entity_id: uuid.UUID,
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.action_type = action_type
        self.target_entity = target_entity
        self.target_entity_id = target_entity_id
        self.details = details or {}
        self.error = error
        self.executed_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация."""
        return {
            "success": self.success,
            "action_type": self.action_type.value,
            "target_entity": self.target_entity,
            "target_entity_id": str(self.target_entity_id),
            "details": self.details,
            "error": self.error,
            "executed_at": self.executed_at.isoformat()
        }


# =============================================================================
# Action Base Class
# =============================================================================

class ReflectionAction(ABC):
    """
    Базовый класс для действия Reflection.

    Action = orchestration, NOT direct DB manipulation!
    """

    def __init__(self, action_type: ActionType):
        self.action_type = action_type

    @abstractmethod
    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """
        Выполнить действие.

        Args:
            decision: ReflectionDecision с параметрами

        Returns:
            ActionResult с результатом
        """
        pass

    def _log_execution(self, decision: ReflectionDecision, result: ActionResult):
        """Логировать выполнение."""
        if result.success:
            logger.info(
                f"Action {self.action_type.value} succeeded: "
                f"{decision.target_entity}={decision.target_entity_id}"
            )
        else:
            logger.error(
                f"Action {self.action_type.value} failed: "
                f"{decision.target_entity}={decision.target_entity_id} - {result.error}"
            )


# =============================================================================
# Concrete Actions
# =============================================================================

class AnnotateAction(ReflectionAction):
    """
    Action: Add metadata/annotation to entity.

    Implementation:
    - Update goal.annotation JSON field
    - Add entry to goal.metadata
    """

    def __init__(self):
        super().__init__(ActionType.ANNOTATE)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Add annotation to entity."""
        try:
            async with AsyncSessionLocal() as db:
                # Load entity
                if decision.target_entity == "goal":
                    stmt = select(Goal).where(Goal.id == decision.target_entity_id)
                    result = await db.execute(stmt)
                    entity = result.scalar_one_or_none()

                    if not entity:
                        return ActionResult(
                            success=False,
                            action_type=self.action_type,
                            target_entity=decision.target_entity,
                            target_entity_id=decision.target_entity_id,
                            error="Entity not found"
                        )

                    # Add annotation (orchestration: call model update method)
                    annotation_type = decision.parameters.get("annotation_type", "generic")
                    annotation_data = {
                        "type": annotation_type,
                        "added_at": datetime.now().isoformat(),
                        "policy": decision.policy_applied,
                        "reason": decision.reason
                    }

                    # Use model's metadata field (NOT direct SQL)
                    if not entity.metadata:
                        entity.metadata = {}

                    if "annotations" not in entity.metadata:
                        entity.metadata["annotations"] = []

                    entity.metadata["annotations"].append(annotation_data)

                    await db.commit()

                    return ActionResult(
                        success=True,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        details={
                            "annotation_type": annotation_type,
                            "annotation_added": True
                        }
                    )
                else:
                    return ActionResult(
                        success=False,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        error=f"Entity type not supported: {decision.target_entity}"
                    )

        except Exception as e:
            logger.error(f"AnnotateAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


class FreezeAction(ReflectionAction):
    """
    Action: Freeze entity (block execution).

    Implementation:
    - Set goal.status = "frozen"
    - Add freeze reason to metadata
    """

    def __init__(self):
        super().__init__(ActionType.FREEZE)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Freeze entity."""
        try:
            async with AsyncSessionLocal() as db:
                if decision.target_entity == "goal":
                    # Load goal
                    stmt = select(Goal).where(Goal.id == decision.target_entity_id)
                    result = await db.execute(stmt)
                    goal = result.scalar_one_or_none()

                    if not goal:
                        return ActionResult(
                            success=False,
                            action_type=self.action_type,
                            target_entity=decision.target_entity,
                            target_entity_id=decision.target_entity_id,
                            error="Goal not found"
                        )

                    # Update status (orchestration: transition to frozen)
                    goal.status = "frozen"

                    # Add metadata
                    if not goal.metadata:
                        goal.metadata = {}

                    goal.metadata["freeze_info"] = {
                        "frozen_at": datetime.now().isoformat(),
                        "policy": decision.policy_applied,
                        "reason": decision.reason,
                        "invariant_id": decision.parameters.get("invariant_id"),
                        "review_requested": decision.parameters.get("review_priority", "high")
                    }

                    await db.commit()

                    return ActionResult(
                        success=True,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        details={
                            "previous_status": "done",  # Assumption
                            "new_status": "frozen",
                            "frozen_at": datetime.now().isoformat()
                        }
                    )
                else:
                    return ActionResult(
                        success=False,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        error=f"Entity type not supported: {decision.target_entity}"
                    )

        except Exception as e:
            logger.error(f"FreezeAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


class RequestReviewAction(ReflectionAction):
    """
    Action: Flag entity for human review.

    Implementation:
    - Set goal.requires_review = True
    - Add review details to metadata
    """

    def __init__(self):
        super().__init__(ActionType.REQUEST_REVIEW)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Flag for human review."""
        try:
            async with AsyncSessionLocal() as db:
                if decision.target_entity == "goal":
                    stmt = select(Goal).where(Goal.id == decision.target_entity_id)
                    result = await db.execute(stmt)
                    goal = result.scalar_one_or_none()

                    if not goal:
                        return ActionResult(
                            success=False,
                            action_type=self.action_type,
                            target_entity=decision.target_entity,
                            target_entity_id=decision.target_entity_id,
                            error="Goal not found"
                        )

                    # Add review flag to metadata
                    if not goal.metadata:
                        goal.metadata = {}

                    goal.metadata["review_request"] = {
                        "requested_at": datetime.now().isoformat(),
                        "policy": decision.policy_applied,
                        "reason": decision.reason,
                        "priority": decision.parameters.get("review_priority", "medium"),
                        "status": "pending"
                    }

                    await db.commit()

                    return ActionResult(
                        success=True,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        details={
                            "review_requested": True,
                            "priority": decision.parameters.get("review_priority", "medium")
                        }
                    )
                else:
                    return ActionResult(
                        success=False,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        error=f"Entity type not supported: {decision.target_entity}"
                    )

        except Exception as e:
            logger.error(f"RequestReviewAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


class SpawnAnalysisAction(ReflectionAction):
    """
    Action: Create analysis goal/task.

    Implementation:
    - Create new goal with type="analysis"
    - Link to original goal via GoalRelation
    """

    def __init__(self):
        super().__init__(ActionType.SPAWN_ANALYSIS)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Spawn analysis goal."""
        try:
            async with AsyncSessionLocal() as db:
                # Create analysis goal
                original_goal_id = decision.parameters.get("original_goal_id")

                analysis_goal = Goal(
                    title=f"Analysis: {decision.reason}",
                    description=f"Automatic analysis task spawned by policy: {decision.policy_applied}",
                    goal_type="exploratory",
                    status="pending",
                    progress=0.0,
                    is_atomic=True,
                    completion_mode="aggregate",
                    depth_level=0,
                    metadata={
                        "spawned_by": "reflection_system",
                        "parent_decision_id": decision.decision_id,
                        "original_goal_id": str(original_goal_id),
                        "analysis_type": decision.parameters.get("analysis_type", "generic"),
                        "priority": decision.parameters.get("priority", "medium")
                    }
                )

                db.add(analysis_goal)
                await db.flush()

                return ActionResult(
                    success=True,
                    action_type=self.action_type,
                    target_entity="goal",
                    target_entity_id=analysis_goal.id,
                    details={
                        "analysis_goal_id": str(analysis_goal.id),
                        "original_goal_id": str(original_goal_id),
                        "analysis_type": decision.parameters.get("analysis_type")
                    }
                )

        except Exception as e:
            logger.error(f"SpawnAnalysisAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


class LogAction(ReflectionAction):
    """
    Action: Log to audit trail (no state change).

    Implementation:
    - Write to structured log
    - No database changes
    """

    def __init__(self):
        super().__init__(ActionType.LOG)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Log to audit trail."""
        try:
            # Structured log entry
            log_entry = {
                "event_type": "reflection_log",
                "decision_id": decision.decision_id,
                "trigger_id": decision.trigger_id,
                "policy": decision.policy_applied,
                "action": "LOG",
                "reason": decision.reason,
                "parameters": decision.parameters,
                "target_entity": decision.target_entity,
                "target_entity_id": str(decision.target_entity_id),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Reflection Log: {log_entry}")

            return ActionResult(
                success=True,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                details={
                    "logged": True,
                    "log_entry": log_entry
                }
            )

        except Exception as e:
            logger.error(f"LogAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


class EscalateAction(ReflectionAction):
    """
    Action: Escalate priority/severity.

    Implementation:
    - Update goal.priority
    - Add escalation metadata
    """

    def __init__(self):
        super().__init__(ActionType.ESCALATE)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Escalate entity."""
        try:
            async with AsyncSessionLocal() as db:
                if decision.target_entity == "goal":
                    stmt = select(Goal).where(Goal.id == decision.target_entity_id)
                    result = await db.execute(stmt)
                    goal = result.scalar_one_or_none()

                    if not goal:
                        return ActionResult(
                            success=False,
                            action_type=self.action_type,
                            target_entity=decision.target_entity,
                            target_entity_id=decision.target_entity_id,
                            error="Goal not found"
                        )

                    # Add escalation metadata
                    if not goal.metadata:
                        goal.metadata = {}

                    goal.metadata["escalation"] = {
                        "escalated_at": datetime.now().isoformat(),
                        "policy": decision.policy_applied,
                        "reason": decision.reason,
                        "previous_severity": decision.parameters.get("previous_severity"),
                        "escalated_severity": decision.parameters.get("escalated_severity"),
                        "review_requested": decision.parameters.get("request_review", False)
                    }

                    await db.commit()

                    return ActionResult(
                        success=True,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        details={
                            "escalated": True,
                            "escalation_reason": decision.parameters.get("escalation_reason")
                        }
                    )
                else:
                    return ActionResult(
                        success=False,
                        action_type=self.action_type,
                        target_entity=decision.target_entity,
                        target_entity_id=decision.target_entity_id,
                        error=f"Entity type not supported: {decision.target_entity}"
                    )

        except Exception as e:
            logger.error(f"EscalateAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


class ScheduleReevaluationAction(ReflectionAction):
    """
    Action: Schedule re-evaluation after delay.

    Implementation:
    - Create Celery task for delayed execution
    - OR add to re-evaluation queue
    """

    def __init__(self):
        super().__init__(ActionType.SCHEDULE_REEVALUATION)

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """Schedule re-evaluation."""
        try:
            # For now: log to re-evaluation queue (table or in-memory)
            # Future: use Celery countdown

            delay_seconds = decision.parameters.get("reevaluation_delay_seconds", 3600)
            reevaluation_at = datetime.now() + timedelta(seconds=delay_seconds)

            log_entry = {
                "event_type": "reflection_schedule_reevaluation",
                "decision_id": decision.decision_id,
                "target_entity": decision.target_entity,
                "target_entity_id": str(decision.target_entity_id),
                "reevaluation_at": reevaluation_at.isoformat(),
                "delay_seconds": delay_seconds,
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"Scheduled Re-evaluation: {log_entry}")

            # TODO: Create Celery task
            # reeval_task = schedule_observer_check.apply_async(
            #     args=[decision.target_entity_id],
            #     countdown=delay_seconds
            # )

            return ActionResult(
                success=True,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                details={
                    "scheduled": True,
                    "reevaluation_at": reevaluation_at.isoformat(),
                    "delay_seconds": delay_seconds
                }
            )

        except Exception as e:
            logger.error(f"ScheduleReevaluationAction failed: {e}")
            return ActionResult(
                success=False,
                action_type=self.action_type,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=str(e)
            )


# =============================================================================
# Action Executor
# =============================================================================

class ActionExecutor:
    """
    Движок для выполнения действий.

    Workflow:
    1. Receive decision
    2. Match action type
    3. Execute action
    4. Return result
    """

    def __init__(self):
        self._actions: Dict[ActionType, ReflectionAction] = {}

        # Register all actions
        self.register_default_actions()

    def register_default_actions(self):
        """Register built-in actions."""
        self.register(AnnotateAction())
        self.register(FreezeAction())
        self.register(RequestReviewAction())
        self.register(SpawnAnalysisAction())
        self.register(LogAction())
        self.register(EscalateAction())
        self.register(ScheduleReevaluationAction())

    def register(self, action: ReflectionAction):
        """Register action."""
        self._actions[action.action_type] = action

    async def execute(self, decision: ReflectionDecision) -> ActionResult:
        """
        Execute action from decision.

        Args:
            decision: ReflectionDecision с action и параметрами

        Returns:
            ActionResult с результатом
        """
        action = self._actions.get(decision.action)

        if not action:
            return ActionResult(
                success=False,
                action_type=decision.action,
                target_entity=decision.target_entity,
                target_entity_id=decision.target_entity_id,
                error=f"Action not registered: {decision.action.value}"
            )

        # Execute action
        result = await action.execute(decision)

        # Log result
        action._log_execution(decision, result)

        return result

    async def execute_batch(self, decisions: List[ReflectionDecision]) -> List[ActionResult]:
        """
        Execute multiple decisions.

        Args:
            decisions: List of ReflectionDecision

        Returns:
            List of ActionResult (parallel execution)
        """
        import asyncio

        # Execute in parallel
        tasks = [self.execute(decision) for decision in decisions]
        results = await asyncio.gather(*tasks)

        return list(results)


# =============================================================================
# Singleton Instance
# =============================================================================

action_executor = ActionExecutor()
