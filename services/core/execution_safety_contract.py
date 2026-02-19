"""
Execution Safety Contract (Phase 2.5.4)

Формальный контракт безопасности между Execution и Reflection.

КРИТИЧЕСКИ ВАЖНО:
- Execution подчиняется Reflection
- Контракт обязателен к исполнению
- Нарушение контракта = архитектурная ошибка

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from execution_reflection_integration import ExecutionFeedback, IntegrationResult
from reflection_system import InvariantSeverity


# =============================================================================
# Safety Levels
# =============================================================================

class SafetyLevel(str, Enum):
    """
    Уровень безопасности для Execution.

    Определяет как Execution должен реагировать на violations.
    """
    NORMAL = "normal"        # Normal operation
    CAUTIOUS = "cautious"    # Extra checks, continue
    RESTRICTED = "restricted"  # Limited operations
    STOPPED = "stopped"      # Stop all execution
    ESCALATED = "escalated"  # Escalate to human


# =============================================================================
# Safety Rule
# =============================================================================

@dataclass
class SafetyRule:
    """
    Правило безопасности.

    IF condition THEN safety_level AND action
    """
    rule_id: str
    condition: str  # Description of condition
    invariant_severity: Optional[InvariantSeverity]
    required_feedback: ExecutionFeedback
    safety_level: SafetyLevel
    description: str
    action_required: str


# =============================================================================
# Execution Safety Contract
# =============================================================================

class ExecutionSafetyContract:
    """
    Формальный контракт безопасности.

    Определяет:
    1. Как Execution должен интерпретировать feedback
    2. Какие действия обязательны при каждом safety level
    3. Как обрабатывать разные типы violations
    """

    def __init__(self):
        self._rules: List[SafetyRule] = []
        self._initialize_default_rules()

    def _initialize_default_rules(self):
        """Initialize default safety rules."""
        # Rule 1: HARD invariant violation → STOP
        self._rules.append(SafetyRule(
            rule_id="HARD_VIOLATION_STOP",
            condition="HARD invariant violated (I7, I9, ORPHANS)",
            invariant_severity=InvariantSeverity.HARD,
            required_feedback=ExecutionFeedback.STOP,
            safety_level=SafetyLevel.STOPPED,
            description="HARD invariant violation must stop execution",
            action_required="Freeze entity, request human review"
        ))

        # Rule 2: SOFT invariant violation → CONTINUE with annotation
        self._rules.append(SafetyRule(
            rule_id="SOFT_VIOLATION_CONTINUE",
            condition="SOFT invariant violated",
            invariant_severity=InvariantSeverity.SOFT,
            required_feedback=ExecutionFeedback.CONTINUE,
            safety_level=SafetyLevel.CAUTIOUS,
            description="SOFT invariant violation allows continuation",
            action_required="Annotate entity, log warning, continue"
        ))

        # Rule 3: Repeated SOFT violations → ESCALATE
        self._rules.append(SafetyRule(
            rule_id="REPEATED_SOFT_ESCALATE",
            condition="Repeated SOFT violations (>3)",
            invariant_severity=InvariantSeverity.SOFT,
            required_feedback=ExecutionFeedback.ESCALATE,
            safety_level=SafetyLevel.ESCALATED,
            description="Repeated SOFT violations require escalation",
            action_required="Escalate priority, request review"
        ))

        # Rule 4: Integration error → ESCALATE
        self._rules.append(SafetyRule(
            rule_id="INTEGRATION_ERROR_ESCALATE",
            condition="Integration (Observer/Reflection) error",
            invariant_severity=None,
            required_feedback=ExecutionFeedback.ESCALATE,
            safety_level=SafetyLevel.ESCALATED,
            description="Integration errors require escalation",
            action_required="Log error, escalate to human"
        ))

        # Rule 5: No violations → NORMAL
        self._rules.append(SafetyRule(
            rule_id="NO_VIOLATIONS_NORMAL",
            condition="No violations detected",
            invariant_severity=None,
            required_feedback=ExecutionFeedback.CONTINUE,
            safety_level=SafetyLevel.NORMAL,
            description="No violations = normal operation",
            action_required="Continue execution"
        ))

    def get_safety_level(
        self,
        integration_result: IntegrationResult
    ) -> SafetyLevel:
        """
        Определить safety level на основе IntegrationResult.

        Args:
            integration_result: Результат от ExecutionReflectionIntegrator

        Returns:
            SafetyLevel
        """
        feedback = integration_result.feedback

        # Map feedback to safety level
        feedback_to_level = {
            ExecutionFeedback.STOP: SafetyLevel.STOPPED,
            ExecutionFeedback.ESCALATE: SafetyLevel.ESCALATED,
            ExecutionFeedback.RETRY: SafetyLevel.RESTRICTED,
            ExecutionFeedback.CONTINUE: SafetyLevel.NORMAL,
            ExecutionFeedback.SKIP: SafetyLevel.NORMAL
        }

        return feedback_to_level.get(feedback, SafetyLevel.ESCALATED)

    def enforce_contract(
        self,
        integration_result: IntegrationResult,
        current_goal_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Принудительное исполнение контракта.

        Args:
            integration_result: Результат интеграции
            current_goal_state: Текущее состояние цели (опционально)

        Returns:
            Enforcement action dict:
            {
                "safety_level": SafetyLevel,
                "action_required": str,
                "can_continue": bool,
                "must_stop": bool,
                "must_escalate": bool,
                "reason": str
            }
        """
        safety_level = self.get_safety_level(integration_result)

        enforcement = {
            "safety_level": safety_level,
            "action_required": None,
            "can_continue": False,
            "must_stop": False,
            "must_escalate": False,
            "reason": integration_result.feedback_reason
        }

        # Determine enforcement based on safety level
        if safety_level == SafetyLevel.NORMAL:
            enforcement["action_required"] = "Continue normal execution"
            enforcement["can_continue"] = True
            enforcement["must_stop"] = False
            enforcement["must_escalate"] = False

        elif safety_level == SafetyLevel.CAUTIOUS:
            enforcement["action_required"] = "Continue with extra logging/monitoring"
            enforcement["can_continue"] = True
            enforcement["must_stop"] = False
            enforcement["must_escalate"] = False

        elif safety_level == SafetyLevel.RESTRICTED:
            enforcement["action_required"] = "Limit operations, prepare for retry"
            enforcement["can_continue"] = False  # Stop and retry
            enforcement["must_stop"] = True
            enforcement["must_escalate"] = False

        elif safety_level == SafetyLevel.STOPPED:
            enforcement["action_required"] = "STOP execution immediately, freeze entity"
            enforcement["can_continue"] = False
            enforcement["must_stop"] = True
            enforcement["must_escalate"] = False

        elif safety_level == SafetyLevel.ESCALATED:
            enforcement["action_required"] = "Escalate to human, stop execution"
            enforcement["can_continue"] = False
            enforcement["must_stop"] = True
            enforcement["must_escalate"] = True

        return enforcement

    def validate_feedback(
        self,
        integration_result: IntegrationResult
    ) -> bool:
        """
        Валидировать feedback на основе контракта.

        Проверяет что feedback соответствует правилам безопасности.

        Args:
            integration_result: Результат интеграции

        Returns:
            True если feedback валиден
        """
        # Check if feedback matches violations
        violations = integration_result.violation_reports

        if not violations:
            # No violations → should be CONTINUE
            return integration_result.feedback == ExecutionFeedback.CONTINUE

        # Have violations → check severity
        has_hard = any(
            v.severity == InvariantSeverity.HARD
            for v in violations
        )

        if has_hard:
            # HARD violation → should be STOP
            return integration_result.feedback in [
                ExecutionFeedback.STOP,
                ExecutionFeedback.ESCALATE
            ]
        else:
            # SOFT violation only → could be CONTINUE or ESCALATE
            return integration_result.feedback in [
                ExecutionFeedback.CONTINUE,
                ExecutionFeedback.ESCALATE
            ]

    def list_rules(self) -> List[SafetyRule]:
        """List all safety rules."""
        return self._rules.copy()

    def get_rule_by_id(self, rule_id: str) -> Optional[SafetyRule]:
        """Get safety rule by ID."""
        for rule in self._rules:
            if rule.rule_id == rule_id:
                return rule
        return None


# =============================================================================
# Contract Violation Exception
# =============================================================================

class SafetyContractViolation(Exception):
    """
    Исключение для случаев нарушения контракта безопасности.

    Генерируется когда:
    - Execution игнорирует STOP feedback
    - Execution продолжает при HARD violation
    - Safety contract не соблюдается
    """

    def __init__(
        self,
        message: str,
        safety_level: SafetyLevel,
        required_action: str,
        actual_action: str
    ):
        self.safety_level = safety_level
        self.required_action = required_action
        self.actual_action = actual_action
        super().__init__(message)


# =============================================================================
# Contract Enforcer
# =============================================================================

class SafetyContractEnforcer:
    """
    Enforcer для контракта безопасности.

    Используется для гарантии соблюдения контракта в Execution.
    """

    def __init__(self, contract: ExecutionSafetyContract = None):
        self.contract = contract or ExecutionSafetyContract()

    async def enforce_before_execution(
        self,
        goal_id: uuid.UUID,
        integration_result: IntegrationResult
    ):
        """
        Enforce contract before continuing execution.

        Raises SafetyContractViolation if contract violated.

        Args:
            goal_id: Goal UUID
            integration_result: Result from integration
        """
        # Validate feedback
        if not self.contract.validate_feedback(integration_result):
            raise SafetyContractViolation(
                message=f"Invalid feedback for goal {goal_id}: {integration_result.feedback}",
                safety_level=self.contract.get_safety_level(integration_result),
                required_action="Comply with safety contract",
                actual_action=str(integration_result.feedback)
            )

        # Get enforcement action
        enforcement = self.contract.enforce_contract(integration_result)

        # Check if execution must stop
        if enforcement["must_stop"] and not enforcement["can_continue"]:
            # Execution must stop - this is not an error, just a requirement
            raise ExecutionMustStopException(
                message=enforcement["action_required"],
                safety_level=enforcement["safety_level"],
                integration_result=integration_result
            )

    def get_required_action(
        self,
        integration_result: IntegrationResult
    ) -> Dict[str, Any]:
        """
        Get required action from integration result.

        Args:
            integration_result: Result from integration

        Returns:
            Enforcement dict from contract
        """
        return self.contract.enforce_contract(integration_result)


# =============================================================================
# Execution Must Stop Exception
# =============================================================================

class ExecutionMustStopException(Exception):
    """
    Исключение для обозначения что Execution должен остановиться.

    Это НЕ ошибка, а сигнальное исключение для управления потоком.
    """

    def __init__(
        self,
        message: str,
        safety_level: SafetyLevel,
        integration_result: IntegrationResult
    ):
        self.safety_level = safety_level
        self.integration_result = integration_result
        super().__init__(message)


# =============================================================================
# Singleton Instance
# =============================================================================

execution_safety_contract = ExecutionSafetyContract()
execution_safety_enforcer = SafetyContractEnforcer()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_safety_level(integration_result: IntegrationResult) -> SafetyLevel:
    """Get safety level from integration result."""
    return execution_safety_contract.get_safety_level(integration_result)


def must_stop_execution(integration_result: IntegrationResult) -> bool:
    """Check if execution must stop based on contract."""
    enforcement = execution_safety_contract.enforce_contract(integration_result)
    return enforcement["must_stop"] and not enforcement["can_continue"]


def can_continue_execution(integration_result: IntegrationResult) -> bool:
    """Check if execution can continue based on contract."""
    enforcement = execution_safety_contract.enforce_contract(integration_result)
    return enforcement["can_continue"]


def must_escalate(integration_result: IntegrationResult) -> bool:
    """Check if escalation is required."""
    enforcement = execution_safety_contract.enforce_contract(integration_result)
    return enforcement["must_escalate"]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SafetyLevel",
    "SafetyRule",
    "ExecutionSafetyContract",
    "SafetyContractViolation",
    "SafetyContractEnforcer",
    "ExecutionMustStopException",
    "execution_safety_contract",
    "execution_safety_enforcer",
    "get_safety_level",
    "must_stop_execution",
    "can_continue_execution",
    "must_escalate",
]
