from logging_config import get_logger
logger = get_logger(__name__)

"""
COMPATIBILITY LAYER v1.0
=======================

Gradual migration layer from Task Engine (done/artifact) to Control System (lifecycle/evaluation)

This provides backward compatibility while introducing correct ontology.

Author: AI-OS Core Team
Date: 2026-02-11
"""

from enum import Enum
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json


# =============================================================================
# NEW ONTOLOGY: Lifecycle States
# =============================================================================

class LifecycleState(str, Enum):
    """
    Lifecycle state of a goal (WHAT STAGE it's in)

    This is NOT the same as evaluation!
    Lifecycle = stage of execution
    Evaluation = quality of outcome
    """
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"      # Former "done" for completable goals
    ONGOING = "ongoing"        # For continuous goals (never "done")
    PERMANENT = "permanent"        # For directional goals (never "done")
    BLOCKED = "blocked"
    FAILED = "failed"


class EvaluationState(str, Enum):
    """
    Evaluation state of a goal (HOW WELL it's doing)

    This is independent from lifecycle!
    """
    NOT_EVALUATED = "not_evaluated"
    VALIDATED = "validated"           # Artifact verified OR state achieved
    IMPROVING = "improving"         # Continuous: trend up
    STABLE = "stable"                 # Continuous: no significant change
    DEGRADING = "degrading"         # Continuous: trend down
    ALIGNED = "aligned"               # Directional: matches direction
    DRIFTING = "drifting"             # Directional: deviating from direction
    CRITICAL = "critical"             # Directional: major misalignment


class CompletionPolicy(str, Enum):
    """
    EXPLICIT completion policy (replaces implicit "if passed → done")

    This makes WHY a goal is considered successful EXPLICIT
    """
    ATOMIC_ARTIFACT = "atomic_artifact"
    # L3 atomic: need passed artifact

    AGGREGATE_CHILDREN = "aggregate_children"
    # L0-L2: all children must be done

    TREND_BASED = "trend_based"
    # Continuous: improvement trend required

    SCALAR_ALIGNMENT = "scalar_alignment"
    # Directional: alignment score required

    HYPOTHESIS_VALIDATION = "hypothesis_validation"
    # Exploratory: hypothesis must be confirmed


# =============================================================================
# COMPATIBILITY WRAPPER
# =============================================================================

class GoalView:
    """
    Compatibility wrapper over Goal model

    Provides unified interface that works with BOTH:
    - Old model (status="done", no lifecycle_state)
    - New model (lifecycle_state, evaluation_state)

    Translates between old and new ontology.
    """

    def __init__(self, goal):
        """
        Args:
            goal: Goal model instance (from database)
        """
        self._goal = goal

    # =========================================================================
    # PROPERTY: lifecycle_state
    # =========================================================================

    @property
    def lifecycle_state(self) -> LifecycleState:
        """
        Get current lifecycle state

        Handles backward compatibility:
        - If lifecycle_state field exists → use it
        - If only "status" exists → map to LifecycleState
        """
        # Check if new field exists
        if hasattr(self._goal, 'lifecycle_state') and self._goal.lifecycle_state:
            return LifecycleState(self._goal.lifecycle_state)

        # Backward compatibility: map old "status" to LifecycleState
        old_status = self._goal.status

        # MIGRATION WARNING for wrong ontology usage
        goal_type = getattr(self._goal, 'goal_type', 'achievable')

        if old_status == "done":
            if goal_type in ["continuous", "directional"]:
                # This is OLD MODEL MISUSE!
                logger.info(f"⚠️ MIGRATION WARNING: {goal_type} goal marked as 'done'")
                logger.info(f"   Goal: {self._goal.title}")
                logger.info(f"   Should use: lifecycle_state='ongoing' or 'permanent'")
                logger.info(f"   Auto-correcting to appropriate state...")

                # Auto-correct
                if goal_type == "continuous":
                    return LifecycleState.ONGOING
                elif goal_type == "directional":
                    return LifecycleState.PERMANENT

            return LifecycleState.COMPLETED

        # Map other statuses
        status_map = {
            "pending": LifecycleState.PENDING,
            "active": LifecycleState.ACTIVE,
            "blocked": LifecycleState.BLOCKED,
            "failed": LifecycleState.FAILED,
        }

        return status_map.get(old_status, LifecycleState.ACTIVE)

    @lifecycle_state.setter
    def lifecycle_state(self, value: LifecycleState):
        """
        Set lifecycle state

        Handles backward compatibility:
        - If lifecycle_state field exists → use it
        - If only "status" exists → map from LifecycleState
        """
        if hasattr(self._goal, 'lifecycle_state'):
            # NEW MODEL: use lifecycle_state directly
            self._goal.lifecycle_state = value.value
        else:
            # OLD MODEL: map to "status"
            old_status_map = {
                LifecycleState.PENDING: "pending",
                LifecycleState.ACTIVE: "active",
                LifecycleState.COMPLETED: "done",
                LifecycleState.BLOCKED: "blocked",
                LifecycleState.FAILED: "failed",
                # Ongoing and Permanent map to "active" in old model
                LifecycleState.ONGOING: "active",
                LifecycleState.PERMANENT: "active",
            }
            self._goal.status = old_status_map.get(value, "active")

    # =========================================================================
    # PROPERTY: evaluation_state
    # =========================================================================

    @property
    def evaluation_state(self) -> EvaluationState:
        """
        Get current evaluation state

        Returns NOT_EVALUATED if not set
        """
        if hasattr(self._goal, 'evaluation_state') and self._goal.evaluation_state:
            return EvaluationState(self._goal.evaluation_state)

        return EvaluationState.NOT_EVALUATED

    @evaluation_state.setter
    def evaluation_state(self, value: EvaluationState):
        """Set evaluation state"""
        if hasattr(self._goal, 'evaluation_state'):
            self._goal.evaluation_state = value.value

    # =========================================================================
    # PROPERTY: completion_policy
    # =========================================================================

    @property
    def completion_policy(self) -> CompletionPolicy:
        """
        Determine completion policy from goal properties

        Mapping:
        - is_atomic=True → ATOMIC_ARTIFACT
        - goal_type=achievable → AGGREGATE_CHILDREN
        - goal_type=continuous → TREND_BASED
        - goal_type=directional → SCALAR_ALIGNMENT
        - goal_type=exploratory → HYPOTHESIS_VALIDATION
        """
        is_atomic = getattr(self._goal, 'is_atomic', False)
        goal_type = getattr(self._goal, 'goal_type', 'achievable')

        if is_atomic:
            return CompletionPolicy.ATOMIC_ARTIFACT

        type_policy_map = {
            "achievable": CompletionPolicy.AGGREGATE_CHILDREN,
            "continuous": CompletionPolicy.TREND_BASED,
            "directional": CompletionPolicy.SCALAR_ALIGNMENT,
            "exploratory": CompletionPolicy.HYPOTHESIS_VALIDATION,
            "meta": CompletionPolicy.AGGREGATE_CHILDREN,
        }

        return type_policy_map.get(goal_type, CompletionPolicy.AGGREGATE_CHILDREN)

    # =========================================================================
    # METHOD: can_mark_completed
    # =========================================================================

    def can_mark_completed(self) -> Tuple[bool, str]:
        """
        Check if goal can be marked as completed

        This is THE KEY method that prevents wrong ontology usage

        Returns:
            (allowed, reason)
        """
        policy = self.completion_policy
        goal_id = str(getattr(self._goal, 'id', 'unknown'))

        if policy == CompletionPolicy.TREND_BASED:
            return (
                False,
                f"Continuous goal '{self._goal.title}' uses trend-based evaluation, "
                f"not completion. Use lifecycle_state='ongoing' + evaluation_state='improving|stable|degrading'"
            )

        if policy == CompletionPolicy.SCALAR_ALIGNMENT:
            return (
                False,
                f"Directional goal '{self._goal.title}' is permanent by definition. "
                f"Use lifecycle_state='permanent' + evaluation_state='aligned|drifting|critical'"
            )

        if policy == CompletionPolicy.ATOMIC_ARTIFACT:
            # Check artifacts
            try:
                from artifact_registry import artifact_registry
                import asyncio

                artifact_check = asyncio.run(
                    artifact_registry.check_goal_artifacts(goal_id)
                )

                if not artifact_check.get("goal_complete"):
                    return (
                        False,
                        f"Atomic goal '{self._goal.title}' requires passed artifact. "
                        f"Total: {artifact_check.get('total_count', 0)}, "
                        f"Passed: {artifact_check.get('passed_count', 0)}"
                    )

                return (True, "Artifact verified")

            except Exception as e:
                return (False, f"Artifact check failed: {e}")

        if policy == CompletionPolicy.AGGREGATE_CHILDREN:
            # Check children
            children = getattr(self._goal, 'children', None)

            if children is None or len(children) == 0:
                return (False, "Goal has no children to aggregate")

            all_done = all(
                c.status in ["done", "completed"]
                for c in children
            )

            if not all_done:
                done_count = sum(1 for c in children if c.status in ["done", "completed"])
                return (
                    False,
                    f"Not all children completed: {done_count}/{len(children)}"
                )

            return (True, f"All {len(children)} children completed")

        if policy == CompletionPolicy.HYPOTHESIS_VALIDATION:
            return (
                False,
                f"Exploratory goal '{self._goal.title}' requires hypothesis confirmation"
            )

        return (False, f"Unknown policy: {policy}")

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_summary(self) -> Dict:
        """
        Get summary of goal state (new ontology)
        """
        return {
            "id": str(getattr(self._goal, 'id', '')),
            "title": getattr(self._goal, 'title', ''),
            "goal_type": getattr(self._goal, 'goal_type', 'achievable'),
            "is_atomic": getattr(self._goal, 'is_atomic', False),
            "lifecycle_state": self.lifecycle_state.value,
            "evaluation_state": self.evaluation_state.value,
            "completion_policy": self.completion_policy.value,
            "old_status": getattr(self._goal, 'status', 'unknown'),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def wrap_goal(goal) -> GoalView:
    """
    Create GoalView wrapper for compatibility

    Usage:
        from compatibility import wrap_goal

        # In old code:
        goal_view = wrap_goal(goal)
        if goal_view.lifecycle_state == LifecycleState.COMPLETED:
            ...

    Args:
        goal: Goal model instance

    Returns:
        GoalView instance
    """
    return GoalView(goal)


# =============================================================================
# MIGRATION HELPERS
# =============================================================================

def detect_migration_needed(goal) -> Optional[Dict]:
    """
    Detect if goal needs migration from old to new ontology

    Returns:
        {
            "needs_migration": bool,
            "issues": [str],
            "recommendations": [str]
        }
        or None if no migration needed
    """
    issues = []
    recommendations = []

    goal_type = getattr(goal, 'goal_type', 'achievable')
    status = getattr(goal, 'status', 'active')
    is_atomic = getattr(goal, 'is_atomic', False)

    # Issue 1: continuous/directional marked as "done"
    if status == "done" and goal_type in ["continuous", "directional"]:
        issues.append(
            f"Wrong ontology: {goal_type} goal marked as 'done' "
            f"(should never be completed)"
        )
        recommendations.append(
            f"Set lifecycle_state='ongoing' for continuous or "
            f"'permanent' for directional"
        )

    # Issue 2: non-atomic goal without children but "done"
    if status == "done" and not is_atomic and not getattr(goal, 'children', []):
        issues.append(
            f"Non-atomic goal marked as 'done' but has no children"
        )
        recommendations.append(
            "Decompose goal into subgoals or use manual completion mode"
        )

    # Issue 3: atomic goal "done" without checking artifacts
    if status == "done" and is_atomic:
        # This is OK if artifacts exist, but we can't check here without async
        recommendations.append(
            "Verify that atomic goal has passed artifacts"
        )

    if not issues:
        return None

    return {
        "needs_migration": len(issues) > 0,
        "issues": issues,
        "recommendations": recommendations,
        "goal_summary": {
            "id": str(getattr(goal, 'id', '')),
            "title": getattr(goal, 'title', ''),
            "goal_type": goal_type,
            "status": status,
            "is_atomic": is_atomic,
        }
    }


# =============================================================================
# TESTING HELPERS
# =============================================================================

async def scan_database_for_migration_issues(db_session) -> Dict:
    """
    Scan all goals and find migration issues

    Returns:
        {
            "total_goals": int,
            "issues_found": int,
            "by_type": {...},
            "problem_goals": [...]
        }
    """
    from models import Goal
    from sqlalchemy import select, func

    # Get all goals
    stmt = select(Goal)
    result = await db_session.execute(stmt)
    goals = result.scalars().all()

    total = len(goals)
    issues_found = 0
    problem_goals = []
    by_type = {}

    for goal in goals:
        migration_info = detect_migration_needed(goal)
        if migration_info and migration_info["needs_migration"]:
            issues_found += 1
            problem_goals.append(migration_info["goal_summary"])

            # Count by type
            gt = migration_info["goal_summary"]["goal_type"]
            by_type[gt] = by_type.get(gt, 0) + 1

    return {
        "total_goals": total,
        "issues_found": issues_found,
        "by_type": by_type,
        "problem_goals": problem_goals
    }


if __name__ == "__main__":
    # Test compatibility layer
    logger.info("=== COMPATIBILITY LAYER TEST ===")

    # Mock goal for testing
    class MockGoal:
        def __init__(self):
            self.id = "test-uuid"
            self.title = "Test Goal"
            self.goal_type = "continuous"
            self.status = "done"  # WRONG!
            self.is_atomic = False
            self.children = []
            self.lifecycle_state = None  # Field doesn't exist yet

    goal = MockGoal()
    view = wrap_goal(goal)

    logger.info(f"Lifecycle state: {view.lifecycle_state}")
    logger.info(f"Completion policy: {view.completion_policy}")
    logger.info(f"Can mark completed: {view.can_mark_completed()}")

    migration_info = detect_migration_needed(goal)
    if migration_info:
        logger.info(f"\n⚠️ MIGRATION NEEDED:")
        for issue in migration_info["issues"]:
            logger.info(f"  - {issue}")
        for rec in migration_info["recommendations"]:
            logger.info(f"  → {rec}")
