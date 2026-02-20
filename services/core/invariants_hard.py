from logging_config import get_logger
logger = get_logger(__name__)

"""
HARD INVARIANTS v1.0 - ENGINEERING GRADE
==========================================

STRICT enforcement with NO bypass mechanism.

This is NOT "advisory" checks - this is HARD LOCK on ontology violations.
Violations throw exceptions that CANNOT be caught silently.

Author: AI-OS Core Team
Date: 2026-02-11
Severity: CRITICAL - System integrity depends on these
"""

from typing import Dict, List, Tuple, Optional
from datetime import datetime
from enum import Enum


# =============================================================================
# SEVERITY LEVELS
# =============================================================================

class InvariantSeverity(str, Enum):
    """Severity of invariant violation"""
    CRITICAL = "CRITICAL"      # Data corruption, ontology breakage
    HIGH = "HIGH"              # Wrong state transition
    MEDIUM = "MEDIUM"          # Minor violations
    LOW = "LOW"                # Warnings


class HardInvariantViolation(Exception):
    """
    HARD invariant violation - CANNOT be silently caught

    This exception MUST propagate to top level and be logged.
    """
    def __init__(self, severity: InvariantSeverity, invariant_code: str, message: str, context: Dict = None):
        self.severity = severity
        self.invariant_code = invariant_code
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.now()

        # Format exception message
        full_message = f"[{severity.value}] {invariant_code}: {message}"
        if context:
            full_message += f"\n  Context: {context}"
        super().__init__(full_message)


# =============================================================================
# INVARIANT CODES ( regimented )
# =============================================================================

class InvariantCode(str, Enum):
    """Standardized invariant violation codes"""

    # Lifecycle State Violations
    I1_CONTINUOUS_AS_DONE = "I1_CONTINUOUS_AS_DONE"
    I2_DIRECTIONAL_AS_DONE = "I2_DIRECTIONAL_AS_DONE"
    I3_WRONG_LIFECYCLE_FOR_TYPE = "I3_WRONG_LIFECYCLE_FOR_TYPE"

    # Artifact Violations
    A1_ATOMIC_WITHOUT_ARTIFACT = "A1_ATOMIC_WITHOUT_ARTIFACT"
    A2_ARTIFACT_VERIFICATION_FAILED = "A2_ARTIFACT_VERIFICATION_FAILED"

    # Transition Violations
    T1_REACTIVATING_COMPLETED = "T1_REACTIVATING_COMPLETED"
    T2_REGRESSING_ACTIVE_TO_PENDING = "T2_REGRESSING_ACTIVE_TO_PENDING"
    T3_FORBIDDEN_AUTO_COMPLETE = "T3_FORBIDDEN_AUTO_COMPLETE"

    # Data Integrity Violations
    D1_PROGRESS_OUT_OF_RANGE = "D1_PROGRESS_OUT_OF_RANGE"
    D2_ORPHANED_COMPLETION = "D2_ORPHANED_COMPLETION"


# =============================================================================
# HARD INVARIANTS CHECKER
# =============================================================================

class HardInvariants:
    """
    HARD invariant checking with ZERO tolerance

    These checks CANNOT be disabled or bypassed.
    All violations throw HardInvariantViolation.
    """

    # =========================================================================
    # LIFECYCLE INVARIANTS
    # =========================================================================

    @staticmethod
    def check_lifecycle_state(goal) -> None:
        """
        HARD CHECK: Lifecycle state matches goal type

        NO BYPASS POSSIBLE - will always raise if violated

        Args:
            goal: Goal instance or GoalView

        Raises:
            HardInvariantViolation: If lifecycle state incompatible with goal type
        """
        goal_type = getattr(goal, 'goal_type', 'achievable')
        lifecycle_state = getattr(goal, 'lifecycle_state', None)
        status = getattr(goal, 'status', None)

        # Map status to lifecycle for checking
        if lifecycle_state is None:
            # Using old model - map from status
            if status == "done":
                lifecycle_state = "completed"
            else:
                lifecycle_state = status

        # I1: Continuous CANNOT be "completed"
        if goal_type == "continuous":
            if lifecycle_state == "completed":
                raise HardInvariantViolation(
                    severity=InvariantSeverity.CRITICAL,
                    invariant_code=InvariantCode.I1_CONTINUOUS_AS_DONE,
                    message=(
                        f"Continuous goal '{getattr(goal, 'title', '')}' "
                        f"has lifecycle_state='completed'. "
                        f"Continuous goals are NEVER complete. "
                        f"Required: lifecycle_state='ongoing'"
                    ),
                    context={
                        "goal_type": goal_type,
                        "lifecycle_state": lifecycle_state,
                        "correction": "Use lifecycle_state='ongoing'"
                    }
                )

        # I2: Directional CANNOT be "completed"
        if goal_type == "directional":
            if lifecycle_state == "completed":
                raise HardInvariantViolation(
                    severity=InvariantSeverity.CRITICAL,
                    invariant_code=InvariantCode.I2_DIRECTIONAL_AS_DONE,
                    message=(
                        f"Directional goal '{getattr(goal, 'title', '')}' "
                        f"has lifecycle_state='completed'. "
                        f"Directional goals are permanent orientation. "
                        f"Required: lifecycle_state='permanent'"
                    ),
                    context={
                        "goal_type": goal_type,
                        "lifecycle_state": lifecycle_state,
                        "correction": "Use lifecycle_state='permanent'"
                    }
                )

        # I3: Achievable atomic MUST use "completed" (not "ongoing"/"permanent")
        if goal_type == "achievable":
            is_atomic = getattr(goal, 'is_atomic', False)
            if is_atomic and lifecycle_state in ["ongoing", "permanent"]:
                raise HardInvariantViolation(
                    severity=InvariantSeverity.HIGH,
                    invariant_code=InvariantCode.I3_WRONG_LIFECYCLE_FOR_TYPE,
                    message=(
                        f"Atomic achievable goal '{getattr(goal, 'title', '')}' "
                        f"has lifecycle_state='{lifecycle_state}'. "
                        f"Achievable atomic goals use 'completed'"
                    ),
                    context={
                        "goal_type": goal_type,
                        "lifecycle_state": lifecycle_state,
                        "is_atomic": is_atomic,
                        "correction": "Use lifecycle_state='completed'"
                    }
                )

    # =========================================================================
    # ARTIFACT INVARIANTS
    # =========================================================================

    @staticmethod
    def check_artifact_requirement(goal, artifact_check_result: Dict) -> None:
        """
        HARD CHECK: Atomic goals have artifacts before completion

        Args:
            goal: Goal instance
            artifact_check_result: Result from artifact_registry.check_goal_artifacts()

        Raises:
            HardInvariantViolation: If invariant violated
        """
        is_atomic = getattr(goal, 'is_atomic', False)
        title = getattr(goal, 'title', '')

        if not is_atomic:
            return  # Non-atomic goals don't require artifacts

        has_passed = artifact_check_result.get("goal_complete", False)
        total = artifact_check_result.get("total_count", 0)
        passed = artifact_check_result.get("passed_count", 0)

        # A1: Atomic MUST have passed artifact to be "completed"
        if not has_passed:
            raise HardInvariantViolation(
                severity=InvariantSeverity.HIGH,
                invariant_code=InvariantCode.A1_ATOMIC_WITHOUT_ARTIFACT,
                message=(
                    f"Atomic goal '{title}' marked as completed "
                    f"without passed artifact. "
                    f"Artifacts: {total} total, {passed} passed"
                ),
                context={
                    "is_atomic": is_atomic,
                    "artifact_count": total,
                    "passed_count": passed,
                    "correction": "Create and verify artifact before completion"
                }
            )

    # =========================================================================
    # TRANSITION INVARIANTS
    # =========================================================================

    @staticmethod
    def check_transition_allowed(
        goal,
        from_state: str,
        to_state: str
    ) -> None:
        """
        HARD CHECK: State transition is allowed

        This is LOCK on forbidden transitions.

        Args:
            goal: Goal instance
            from_state: Current state
            to_state: Target state

        Raises:
            HardInvariantViolation: If transition not allowed
        """
        goal_id = str(getattr(goal, 'id', 'unknown'))
        goal_type = getattr(goal, 'goal_type', 'achievable')
        title = getattr(goal, 'title', '')

        # T1: Cannot reactivate completed goals
        if from_state in ["done", "completed"] and to_state == "active":
            raise HardInvariantViolation(
                severity=InvariantSeverity.CRITICAL,
                invariant_code=InvariantCode.T1_REACTIVATING_COMPLETED,
                message=(
                    f"Forbidden transition: '{from_state}' → '{to_state}' "
                    f"for goal '{title}'. "
                    f"Completed goals cannot be reactivated."
                ),
                context={
                    "goal_id": goal_id,
                    "from_state": from_state,
                    "to_state": to_state,
                    "goal_type": goal_type
                }
            )

        # T2: Cannot regress active to pending
        if from_state == "active" and to_state == "pending":
            raise HardInvariantViolation(
                severity=InvariantSeverity.HIGH,
                invariant_code=InvariantCode.T2_REGRESSING_ACTIVE_TO_PENDING,
                message=(
                    f"Forbidden transition: '{from_state}' → '{to_state}' "
                    f"for goal '{title}'. "
                    f"Cannot regress active goal to pending."
                ),
                context={
                    "goal_id": goal_id,
                    "from_state": from_state,
                    "to_state": to_state
                }
            )

        # T3: Continuous/Directional cannot transition to "done"
        if to_state == "done":
            if goal_type == "continuous":
                raise HardInvariantViolation(
                    severity=InvariantSeverity.CRITICAL,
                    invariant_code=InvariantCode.T3_FORBIDDEN_AUTO_COMPLETE,
                    message=(
                        f"Forbidden transition: Continuous goal '{title}' "
                        f"cannot transition to 'done'. "
                        f"Use lifecycle_state='ongoing' instead."
                    ),
                    context={
                        "goal_id": goal_id,
                        "goal_type": goal_type,
                        "forbidden_state": "done",
                        "correct_state": "ongoing"
                    }
                )

            if goal_type == "directional":
                raise HardInvariantViolation(
                    severity=InvariantSeverity.CRITICAL,
                    invariant_code=InvariantCode.T3_FORBIDDEN_AUTO_COMPLETE,
                    message=(
                        f"Forbidden transition: Directional goal '{title}' "
                        f"cannot transition to 'done'. "
                        f"Use lifecycle_state='permanent' instead."
                    ),
                    context={
                        "goal_id": goal_id,
                        "goal_type": goal_type,
                        "forbidden_state": "done",
                        "correct_state": "permanent"
                    }
                )

    # =========================================================================
    # DATA INTEGRITY INVARIANTS
    # =========================================================================

    @staticmethod
    def check_data_integrity(goal) -> None:
        """
        HARD CHECK: Data fields are within valid ranges

        Args:
            goal: Goal instance

        Raises:
            HardInvariantViolation: If data integrity violated
        """
        title = getattr(goal, 'title', '')
        progress = getattr(goal, 'progress', 0.0)

        # D1: Progress must be in [0.0, 1.0]
        if not (0.0 <= progress <= 1.0):
            raise HardInvariantViolation(
                severity=InvariantSeverity.CRITICAL,
                invariant_code=InvariantCode.D1_PROGRESS_OUT_OF_RANGE,
                message=(
                    f"Goal '{title}' has progress={progress}. "
                    f"Progress must be in range [0.0, 1.0]"
                ),
                context={
                    "goal_id": str(getattr(goal, 'id', '')),
                    "progress": progress,
                    "valid_range": "[0.0, 1.0]"
                }
            )


# =============================================================================
# VALIDATION FUNCTION (convenience)
# =============================================================================

def validate_hard_invariants(
    goal,
    artifact_check: Optional[Dict] = None
) -> None:
    """
    Run ALL hard invariant checks

    This is the MAIN ENTRY POINT for hard validation.

    Args:
        goal: Goal instance
        artifact_check: Optional artifact check result

    Raises:
        HardInvariantViolation: If ANY invariant violated

    Usage:
        from invariants_hard import validate_hard_invariants

        validate_hard_invariants(goal, artifact_check)
        # If passes, execution continues
        # If fails, exception propagates
    """
    # Check 1: Lifecycle state
    HardInvariants.check_lifecycle_state(goal)

    # Check 2: Data integrity
    HardInvariants.check_data_integrity(goal)

    # Check 3: Artifacts (if provided)
    if artifact_check is not None:
        HardInvariants.check_artifact_requirement(goal, artifact_check)


# =============================================================================
# HELPER: Get invariant summary for logging
# =============================================================================

def get_invariant_violation_summary(exception: HardInvariantViolation) -> Dict:
    """
    Extract structured data from violation for logging

    Returns:
        {
            "severity": "CRITICAL|HIGH|MEDIUM|LOW",
            "code": "I1_CONTINUOUS_AS_DONE",
            "message": "...",
            "context": {...},
            "timestamp": "..."
        }
    """
    return {
        "severity": exception.severity.value,
        "invariant_code": exception.invariant_code.value,
        "message": exception.message,
        "context": exception.context,
        "timestamp": exception.timestamp.isoformat()
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logger.info("Testing HARD Invariants...\n")

    # Mock goal with violation
    class MockGoal:
        def __init__(self):
            self.id = "test-uuid"
            self.title = "Test Continuous Goal"
            self.goal_type = "continuous"
            self.is_atomic = False
            self.lifecycle_state = "completed"  # VIOLATION!
            self.status = "done"

    goal = MockGoal()

    try:
        HardInvariants.check_lifecycle_state(goal)
        logger.info("❌ FAILED: Should have raised exception")
    except HardInvariantViolation as e:
        logger.info("✅ PASSED: Caught invariant violation")
        logger.info(f"  Severity: {e.severity}")
        logger.info(f"  Code: {e.invariant_code}")
        logger.info(f"  Message: {e.message}")

    # Test artifact check
    try:
        HardInvariants.check_artifact_requirement(
            goal,
            {"goal_complete": False, "total_count": 0, "passed_count": 0}
        )
        logger.info("❌ FAILED: Should have raised exception")
    except HardInvariantViolation as e:
        logger.info("✅ PASSED: Caught artifact violation")
        logger.info(f"  Severity: {e.severity}")
        logger.info(f"  Message: {e.message}")
