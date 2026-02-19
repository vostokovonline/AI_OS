"""
Reflection Policies (Phase 2.4.2)

Детерминированные правила для принятия решений.

КРИТИЧЕСКИ ВАЖНО:
- ❌ NO LLM decisions
- ✅ YES deterministic rules
- ✅ YES predictable behavior

Architecture:
  Signal → Policy Match → Decision

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime

from reflection_system import (
    ReflectionSignal,
    ReflectionDecision,
    ReflectionContext,
    ActionType,
    TriggerSource,
    create_decision_id,
    InvariantSeverity
)
from lifecycle_observer import InvariantViolationReport


# =============================================================================
# Policy Base Class
# =============================================================================

class ReflectionPolicy(ABC):
    """
    Базовый класс для политики рефлексии.

    Policy = детерминированное правило "если X, то Y"
    НЕ содержит LLM calls!
    """

    def __init__(self, policy_id: str, priority: int = 100):
        self.policy_id = policy_id
        self.priority = priority  # Higher = checked first

    @abstractmethod
    def matches(self, context: ReflectionContext) -> bool:
        """
        Проверить, применима ли политика к контексту.

        Args:
            context: ReflectionContext с сигналом и состоянием

        Returns:
            True если политика применима
        """
        pass

    @abstractmethod
    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """
        Принять решение на основе контекста.

        Args:
            context: ReflectionContext

        Returns:
            ReflectionDecision с action и параметрами
        """
        pass

    def _create_decision(
        self,
        context: ReflectionContext,
        action: ActionType,
        target_entity: str,
        target_entity_id: uuid.UUID,
        reason: str,
        parameters: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> ReflectionDecision:
        """Helper для создания решения."""
        return ReflectionDecision(
            decision_id=create_decision_id(),
            trigger_id=context.signal.trigger_id,
            signal_source=context.signal.source,
            action=action,
            target_entity=target_entity,
            target_entity_id=target_entity_id,
            reason=reason,
            policy_applied=self.policy_id,
            confidence=confidence,
            parameters=parameters or {}
        )


# =============================================================================
# Concrete Policies
# =============================================================================

class HardInvariantViolationPolicy(ReflectionPolicy):
    """
    Policy: HARD invariant violation → Block execution + request review.

    Trigger:
    - Observer signal with HARD severity
    - Invariant I7, I9, ORPHANS violated

    Action:
    - FREEZE entity (block execution)
    - REQUEST_REVIEW (flag for human)
    """

    def __init__(self):
        super().__init__(
            policy_id="HARD_INVARIANT_VIOLATION",
            priority=1000  # Highest priority
        )

    def matches(self, context: ReflectionContext) -> bool:
        """Apply to HARD invariant violations."""
        signal = context.signal

        # Must be observer signal
        if signal.source != TriggerSource.OBSERVER:
            return False

        # Must be HARD severity
        if signal.severity != InvariantSeverity.HARD:
            return False

        # Must be a violation (not passed)
        if signal.context.get("passed", True):
            return False

        # Must have entity info
        if not signal.entity or not signal.entity_id:
            return False

        return True

    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """Decision: Freeze + Request Review"""
        signal = context.signal

        return self._create_decision(
            context=context,
            action=ActionType.FREEZE,
            target_entity=signal.entity,
            target_entity_id=signal.entity_id,
            reason=f"HARD invariant violated: {signal.invariant_id} - {signal.invariant_name}",
            parameters={
                "invariant_id": signal.invariant_id,
                "invariant_name": signal.invariant_name,
                "violation_details": signal.violation_details,
                "review_reason": f"HARD invariant {signal.invariant_id} violated",
                "review_priority": "high"
            },
            confidence=1.0
        )


class SoftInvariantViolationPolicy(ReflectionPolicy):
    """
    Policy: SOFT invariant violation → Annotate + log.

    Trigger:
    - Observer signal with SOFT severity

    Action:
    - ANNOTATE entity (add metadata)
    - LOG (audit)
    """

    def __init__(self):
        super().__init__(
            policy_id="SOFT_INVARIANT_VIOLATION",
            priority=500
        )

    def matches(self, context: ReflectionContext) -> bool:
        """Apply to SOFT invariant violations."""
        signal = context.signal

        if signal.source != TriggerSource.OBSERVER:
            return False

        if signal.severity != InvariantSeverity.SOFT:
            return False

        if signal.context.get("passed", True):
            return False

        if not signal.entity or not signal.entity_id:
            return False

        return True

    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """Decision: Annotate + Log"""
        signal = context.signal

        return self._create_decision(
            context=context,
            action=ActionType.ANNOTATE,
            target_entity=signal.entity,
            target_entity_id=signal.entity_id,
            reason=f"SOFT invariant violated: {signal.invariant_id} - {signal.invariant_name}",
            parameters={
                "annotation_type": "soft_violation",
                "invariant_id": signal.invariant_id,
                "invariant_name": signal.invariant_name,
                "violation_details": signal.violation_details,
                "logged": True
            },
            confidence=1.0
        )


class OrphanedChildrenPolicy(ReflectionPolicy):
    """
    Policy: Orphaned children → Schedule re-evaluation.

    Trigger:
    - Observer signal: ORPHANS invariant violated

    Action:
    - SCHEDULE_REEVALUATION (recheck after delay)
    - ANNOTATE parent
    """

    def __init__(self):
        super().__init__(
            policy_id="ORPHANED_CHILDREN",
            priority=900  # High priority (but below HARD)
        )

    def matches(self, context: ReflectionContext) -> bool:
        """Apply to ORPHANS invariant violations."""
        signal = context.signal

        if signal.source != TriggerSource.OBSERVER:
            return False

        if signal.invariant_id != "ORPHANS":
            return False

        if signal.context.get("passed", True):
            return False

        if not signal.entity or not signal.entity_id:
            return False

        return True

    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """Decision: Schedule Re-evaluation + Annotate"""
        signal = context.signal

        orphaned_count = signal.violation_details.get("orphaned_count", 0)

        return self._create_decision(
            context=context,
            action=ActionType.SCHEDULE_REEVALUATION,
            target_entity=signal.entity,
            target_entity_id=signal.entity_id,
            reason=f"Orphaned children detected: {orphaned_count} children in DONE tree",
            parameters={
                "reevaluation_delay_seconds": 3600,  # 1 hour
                "annotation_type": "orphaned_children",
                "orphaned_count": orphaned_count,
                "orphaned_children": signal.violation_details.get("orphaned_children", [])
            },
            confidence=1.0
        )


class RepeatedViolationPolicy(ReflectionPolicy):
    """
    Policy: Repeated SOFT violations → Escalate.

    Trigger:
    - Entity has > N SOFT violations in history

    Action:
    - ESCALATE (elevate severity/priority)
    - REQUEST_REVIEW
    """

    def __init__(self, violation_threshold: int = 3):
        super().__init__(
            policy_id="REPEATED_VIOLATION",
            priority=800
        )
        self.violation_threshold = violation_threshold

    def matches(self, context: ReflectionContext) -> bool:
        """Apply if entity has repeated violations."""
        signal = context.signal

        # Must be observer signal
        if signal.source != TriggerSource.OBSERVER:
            return False

        # Count past violations for this entity
        if not context.entity_history:
            return False

        past_violations = [
            event for event in context.entity_history
            if event.get("event_type") == "invariant_violation"
            and event.get("severity") == "SOFT"
            and event.get("entity_id") == str(signal.entity_id)
        ]

        if len(past_violations) < self.violation_threshold:
            return False

        return True

    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """Decision: Escalate + Request Review"""
        signal = context.signal

        violation_count = self.violation_threshold + 1

        return self._create_decision(
            context=context,
            action=ActionType.ESCALATE,
            target_entity=signal.entity,
            target_entity_id=signal.entity_id,
            reason=f"Repeated SOFT violations: {violation_count} violations detected",
            parameters={
                "escalation_reason": "repeated_soft_violations",
                "violation_count": violation_count,
                "previous_severity": "SOFT",
                "escalated_severity": "HARD",
                "request_review": True,
                "review_priority": "medium"
            },
            confidence=1.0
        )


class ExecutionFailurePolicy(ReflectionPolicy):
    """
    Policy: Goal execution failure → Analyze + Annotate.

    Trigger:
    - Execution signal with status = "failure"

    Action:
    - SPAWN_ANALYSIS (create analysis goal)
    - ANNOTATE original goal
    """

    def __init__(self):
        super().__init__(
            policy_id="EXECUTION_FAILURE",
            priority=700
        )

    def matches(self, context: ReflectionContext) -> bool:
        """Apply to execution failures."""
        signal = context.signal

        if signal.source != TriggerSource.EXECUTION:
            return False

        if signal.execution_status != "failure":
            return False

        if not signal.execution_goal_id:
            return False

        return True

    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """Decision: Spawn Analysis + Annotate"""
        signal = context.signal

        return self._create_decision(
            context=context,
            action=ActionType.SPAWN_ANALYSIS,
            target_entity="goal",
            target_entity_id=signal.execution_goal_id,
            reason=f"Goal execution failed: {signal.execution_status}",
            parameters={
                "analysis_type": "failure_analysis",
                "original_goal_id": str(signal.execution_goal_id),
                "execution_outcome": signal.execution_outcome,
                "annotation_type": "execution_failure",
                "priority": "high"
            },
            confidence=1.0
        )


class ManualTriggerPolicy(ReflectionPolicy):
    """
    Policy: Manual trigger → Log only (no action).

    Trigger:
    - Manual admin trigger

    Action:
    - LOG (audit)
    """

    def __init__(self):
        super().__init__(
            policy_id="MANUAL_TRIGGER",
            priority=100  # Low priority
        )

    def matches(self, context: ReflectionContext) -> bool:
        """Apply to manual triggers."""
        signal = context.signal

        if signal.source != TriggerSource.MANUAL:
            return False

        return True

    def decide(self, context: ReflectionContext) -> ReflectionDecision:
        """Decision: Log only"""
        signal = context.signal

        return self._create_decision(
            context=context,
            action=ActionType.LOG,
            target_entity="system",
            target_entity_id=uuid.UUID("00000000-0000-0000-0000-000000000000"),
            reason=f"Manual reflection trigger: {signal.external_reason}",
            parameters={
                "log_reason": "manual_trigger",
                "triggered_by": "admin",
                "reason": signal.external_reason,
                "context": signal.external_context
            },
            confidence=1.0
        )


# =============================================================================
# Policy Engine
# =============================================================================

class PolicyEngine:
    """
    Движок для управления политиками и принятия решений.

    Workflow:
    1. Receive signal → Create context
    2. Match policies (by priority)
    3. Generate decisions
    4. Return decisions (for execution)
    """

    def __init__(self):
        self._policies: List[ReflectionPolicy] = []

        # Register default policies
        self.register_default_policies()

    def register_default_policies(self):
        """Register built-in policies."""
        self.register(HardInvariantViolationPolicy())
        self.register(SoftInvariantViolationPolicy())
        self.register(OrphanedChildrenPolicy())
        self.register(RepeatedViolationPolicy(violation_threshold=3))
        self.register(ExecutionFailurePolicy())
        self.register(ManualTriggerPolicy())

    def register(self, policy: ReflectionPolicy):
        """Register custom policy."""
        self._policies.append(policy)
        # Sort by priority (highest first)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def list_policies(self) -> List[str]:
        """List all registered policy IDs."""
        return [p.policy_id for p in self._policies]

    def match_policies(self, context: ReflectionContext) -> List[ReflectionPolicy]:
        """
        Find all applicable policies for context.

        Returns policies sorted by priority (highest first).
        """
        matching = []

        for policy in self._policies:
            if policy.matches(context):
                matching.append(policy)

        return matching

    def decide(
        self,
        signal: ReflectionSignal,
        entity_state: Optional[Dict[str, Any]] = None,
        entity_history: Optional[List[Dict[str, Any]]] = None,
        system_state: Optional[Dict[str, Any]] = None
    ) -> List[ReflectionDecision]:
        """
        Generate decisions based on signal and context.

        Args:
            signal: Input signal
            entity_state: Current entity state (optional)
            entity_history: Entity history (optional)
            system_state: System-level state (optional)

        Returns:
            List of ReflectionDecision (may be empty)
        """
        # Create context
        context = ReflectionContext(
            signal=signal,
            entity_state=entity_state,
            entity_history=entity_history,
            system_state=system_state
        )

        # Match policies
        matching_policies = self.match_policies(context)

        # Generate decisions
        decisions = []
        for policy in matching_policies:
            decision = policy.decide(context)
            decisions.append(decision)

        return decisions


# =============================================================================
# Singleton Instance
# =============================================================================

policy_engine = PolicyEngine()
