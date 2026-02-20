from logging_config import get_logger
logger = get_logger(__name__)

"""
GOAL INVARIANTS CHECKER v1.0
============================

Protects system from logical contradictions in goal state management.

This prevents the "self-deception" problem where continuous/directional
goals are marked as "done" when they should never be completable.

Author: AI-OS Core Team
Date: 2026-02-11
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
import uuid


class GoalInvariantViolation(Exception):
    """Raised when goal state violates invariants"""

    def __init__(self, violations: List[str]):
        self.violations = violations
        message = f"Goal invariant violations:\n" + "\n".join(f"  - {v}" for v in violations)
        super().__init__(message)


class GoalTransitionError(Exception):
    """Raised when goal state transition is not allowed"""

    def __init__(self, from_state: str, to_state: str, reason: str):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        message = f"Invalid goal transition: {from_state} → {to_state}\n  Reason: {reason}"
        super().__init__(message)


# =============================================================================
# GOAL INVARIANTS CHECKER
# =============================================================================

class GoalInvariants:
    """
    System of checks that prevent logical contradictions

    These invariants protect the ontology from degradation
    """

    # =========================================================================
    # INVARIANT CHECKS
    # =========================================================================

    @staticmethod
    def check_goal_completion(goal) -> Tuple[bool, List[str]]:
        """
        Check if goal can be marked as completed

        THIS IS THE MAIN CHECK preventing self-deception

        Args:
            goal: Goal model instance or GoalView wrapper

        Returns:
            (valid, violations)
            - valid: True if invariants pass
            - violations: List of violation descriptions
        """
        violations = []

        # Get goal properties (handle both Goal and GoalView)
        if hasattr(goal, '_goal'):
            # GoalView wrapper
            goal_id = str(getattr(goal._goal, 'id', 'unknown'))
            title = getattr(goal._goal, 'title', '')
            goal_type = getattr(goal._goal, 'goal_type', 'achievable')
            status = getattr(goal._goal, 'status', 'active')
            is_atomic = getattr(goal._goal, 'is_atomic', False)
            lifecycle_state = goal.lifecycle_state
            completion_policy = goal.completion_policy
        else:
            # Direct Goal model
            goal_id = str(getattr(goal, 'id', 'unknown'))
            title = getattr(goal, 'title', '')
            goal_type = getattr(goal, 'goal_type', 'achievable')
            status = getattr(goal, 'status', 'active')
            is_atomic = getattr(goal, 'is_atomic', False)
            lifecycle_state = getattr(goal, 'lifecycle_state', status)
            completion_policy = None

        # =========================================================================
        # INVARIANT I1: Continuous goals CANNOT be "done"
        # =========================================================================
        if goal_type == "continuous":
            if status == "done":
                violations.append(
                    f"I1-VIOLATION: Continuous goal '{title}' marked as 'done'. "
                    f"Continuous goals have NO final endpoint. "
                    f"Correct: lifecycle_state='ongoing' + evaluation_state='improving|stable|degrading'"
                )
            elif lifecycle_state == "completed":
                violations.append(
                    f"I1-VIOLATION: Continuous goal '{title}' in lifecycle_state='completed'. "
                    f"Continuous goals are never complete. Use lifecycle_state='ongoing'"
                )

        # =========================================================================
        # INVARIANT I2: Directional goals CANNOT be "done"
        # =========================================================================
        if goal_type == "directional":
            if status == "done":
                violations.append(
                    f"I2-VIOLATION: Directional goal '{title}' marked as 'done'. "
                    f"Directional goals are permanent orientation points, not completable. "
                    f"Correct: lifecycle_state='permanent' + evaluation_state='aligned|drifting|critical'"
                )
            elif lifecycle_state == "completed":
                violations.append(
                    f"I2-VIOLATION: Directional goal '{title}' in lifecycle_state='completed'. "
                    f"Directional goals don't complete. Use lifecycle_state='permanent'"
                )

        # =========================================================================
        # INVARIANT I3: Atomic goals MUST have artifacts to be done
        # =========================================================================
        if is_atomic and status == "done":
            # Need to check artifacts (async, so we return warning)
            violations.append(
                f"I3-CHECK: Atomic goal '{title}' marked as 'done' but artifacts not verified. "
                f"Atomic goals REQUIRE passed artifact. "
                f"Run artifact check before marking done."
            )

        # =========================================================================
        # INVARIANT I4: Non-atomic goals MUST have children or completion_mode=manual
        # =========================================================================
        if not is_atomic and status == "done":
            children = getattr(goal, 'children', None)
            completion_mode = getattr(goal, 'completion_mode', 'aggregate')

            if children is None or len(children) == 0:
                if completion_mode != "manual":
                    violations.append(
                        f"I4-VIOLATION: Non-atomic goal '{title}' marked as 'done' "
                        f"but has no children and completion_mode='{completion_mode}'. "
                        f"Either decompose into subgoals (aggregate) or use manual completion"
                    )

        # =========================================================================
        # INVARIANT I5: Progress must be in [0.0, 1.0]
        # =========================================================================
        progress = getattr(goal, 'progress', 0.0)
        if not (0.0 <= progress <= 1.0):
            violations.append(
                f"I5-VIOLATION: Goal '{title}' has progress={progress}. "
                f"Progress must be in range [0.0, 1.0]"
            )

        # =========================================================================
        # RETURN RESULT
        # =========================================================================
        return (len(violations) == 0, violations)

    @staticmethod
    def check_goal_transition(
        goal,
        old_status: str,
        new_status: str
    ) -> Tuple[bool, str]:
        """
        Check if goal state transition is allowed

        Args:
            goal: Goal instance
            old_status: Current status
            new_status: Requested new status

        Returns:
            (allowed, reason)
        """
        goal_id = str(getattr(goal, 'id', 'unknown'))
        title = getattr(goal, 'title', '')
        goal_type = getattr(goal, 'goal_type', 'achievable')

        # =========================================================================
        # FORBIDDEN TRANSITIONS
        # =========================================================================

        # T1: Cannot reactivate completed goals
        if old_status in ["done", "completed"] and new_status == "active":
            return (
                False,
                f"T1-FORBIDDEN: Cannot reactive completed goal '{title}' "
                f"from '{old_status}' to '{new_status}'. "
                f"Completed goals are final states."
            )

        # T2: Cannot mark active → pending
        if old_status == "active" and new_status == "pending":
            return (
                False,
                f"T2-FORBIDDEN: Cannot transition goal '{title}' "
                f"from active to pending (regression)"
            )

        # =========================================================================
        # GOAL-TYPE SPECIFIC TRANSITIONS
        # =========================================================================

        # T3: Continuous goals never transition to "done"
        if goal_type == "continuous" and new_status == "done":
            return (
                False,
                f"T3-FORBIDDEN: Continuous goal '{title}' cannot transition to 'done'. "
                f"Use lifecycle_state='ongoing' instead"
            )

        # T4: Directional goals never transition to "done"
        if goal_type == "directional" and new_status == "done":
            return (
                False,
                f"T4-FORBIDDEN: Directional goal '{title}' cannot transition to 'done'. "
                f"Use lifecycle_state='permanent' instead"
            )

        # =========================================================================
        # ALLOWED TRANSITION
        # =========================================================================

        return (True, f"Transition allowed: {old_status} → {new_status}")

    @staticmethod
    def validate_artifact_requirement(goal) -> Tuple[bool, str]:
        """
        Validate that goal meets artifact requirements

        Args:
            goal: Goal instance

        Returns:
            (valid, reason)
        """
        is_atomic = getattr(goal, 'is_atomic', False)
        goal_id = str(getattr(goal, 'id', 'unknown'))

        if not is_atomic:
            # Non-atomic goals don't require artifacts
            return (True, "Non-atomic goal has no artifact requirement")

        # Atomic goals: check artifacts
        try:
            from artifact_registry import artifact_registry
            import asyncio

            artifact_check = asyncio.run(
                artifact_registry.check_goal_artifacts(goal_id)
            )

            if not artifact_check.get("goal_complete"):
                return (
                    False,
                    f"Atomic goal requires passed artifact. "
                    f"Total: {artifact_check.get('total_count', 0)}, "
                    f"Passed: {artifact_check.get('passed_count', 0)}"
                )

            return (True, "Artifact requirement met")

        except Exception as e:
            return (False, f"Artifact check failed: {e}")


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_and_raise(goal) -> None:
    """
    Validate goal and raise exception if invariants violated

    Usage:
        from invariants import validate_and_raise

        validate_and_raise(goal)  # Raises GoalInvariantViolation if issues

    Args:
        goal: Goal instance

    Raises:
        GoalInvariantViolation: if invariants violated
    """
    valid, violations = GoalInvariants.check_goal_completion(goal)

    if not valid:
        raise GoalInvariantViolation(violations)


def validate_transition_and_raise(
    goal,
    old_status: str,
    new_status: str
) -> None:
    """
    Validate transition and raise exception if not allowed

    Usage:
        from invariants import validate_transition_and_raise

        validate_transition_and_raise(goal, "active", "done")

    Raises:
        GoalTransitionError: if transition not allowed
    """
    allowed, reason = GoalInvariants.check_goal_transition(
        goal, old_status, new_status
    )

    if not allowed:
        raise GoalTransitionError(old_status, new_status, reason)


# =============================================================================
# BATCH VALIDATION
# =============================================================================

async def scan_all_goals(db_session) -> Dict:
    """
    Scan all goals and find invariant violations

    Args:
        db_session: Database session

    Returns:
        {
            "total_goals": int,
            "violations_count": int,
            "by_goal_type": {...},
            "violations": [
                {
                    "goal_id": "...",
                    "title": "...",
                    "violations": [...]
                }
            ]
        }
    """
    from models import Goal
    from sqlalchemy import select

    # Get all goals
    stmt = select(Goal)
    result = await db_session.execute(stmt)
    goals = result.scalars().all()

    total = len(goals)
    all_violations = []
    by_type = {}

    for goal in goals:
        valid, violations = GoalInvariants.check_goal_completion(goal)

        if not valid:
            all_violations.append({
                "goal_id": str(goal.id),
                "title": goal.title,
                "goal_type": goal.goal_type,
                "status": goal.status,
                "is_atomic": goal.is_atomic,
                "violations": violations
            })

            # Count by type
            gt = goal.goal_type
            by_type[gt] = by_type.get(gt, 0) + 1

    return {
        "total_goals": total,
        "violations_count": len(all_violations),
        "by_goal_type": by_type,
        "violations": all_violations
    }


def print_violation_report(scan_result: Dict) -> None:
    """
    Pretty print violation scan results

    Args:
        scan_result: Result from scan_all_goals()
    """
    logger.info("\n" + "="*70)
    logger.info("GOAL INVARIANTS SCAN REPORT")
    logger.info("="*70)

    logger.info(f"\n📊 STATISTICS:")
    logger.info(f"  Total goals: {scan_result['total_goals']}")
    logger.info(f"  Violations found: {scan_result['violations_count']}")
    logger.info(f"  Violation rate: {scan_result['violations_count'] / scan_result['total_goals'] * 100:.1f}%")

    logger.info(f"\n📋 BY GOAL TYPE:")
    for goal_type, count in scan_result['by_goal_type'].items():
        logger.info(f"  {goal_type}: {count} violations")

    if scan_result['violations']:
        logger.info(f"\n⚠️  VIOLATIONS ({len(scan_result['violations'])} goals):")
        logger.info("-"*70)

        for i, v in enumerate(scan_result['violations'][:20], 1):  # Show first 20
            logger.info(f"\n{i}. Goal: {v['title']}")
            logger.info(f"   Type: {v['goal_type']}")
            logger.info(f"   Status: {v['status']}")
            logger.info(f"   Atomic: {v['is_atomic']}")
            logger.info(f"   Violations:")
            for viol in v['violations']:
                logger.info(f"     - {viol}")

        if len(scan_result['violations']) > 20:
            logger.info(f"\n... and {len(scan_result['violations']) - 20} more")
    else:
        logger.info("\n✅ NO INVARIANT VIOLATIONS FOUND")

    logger.info("="*70 + "\n")


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from database import AsyncSessionLocal

    async def test():
        logger.info("Testing GoalInvariants...")

        # Create mock goal with violations
        class MockGoal:
            def __init__(self):
                self.id = uuid.uuid4()
                self.title = "Test Continuous Goal"
                self.goal_type = "continuous"
                self.status = "done"  # VIOLATION!
                self.is_atomic = False
                self.children = []
                self.completion_mode = "aggregate"
                self.progress = 0.8

        goal = MockGoal()

        # Check invariants
        valid, violations = GoalInvariants.check_goal_completion(goal)

        logger.info(f"\nValid: {valid}")
        logger.info(f"Violations: {violations}")

        if not valid:
            logger.info("\n✅ Invariants working! Detected violation correctly")
        else:
            logger.info("\n❌ Invariants failed! Should have detected violation")

    asyncio.run(test())
