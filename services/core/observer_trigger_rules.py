"""
Observer Triggering Rules (Phase 2.5.2)

Правила триггерирования Observer на основе Execution Events.

Цель: определить КОГДА и ЧТО проверять после событий выполнения.

Architecture:
  ExecutionEvent → TriggerRule.matches() → Invariants to check
                  → Observer runs → Reports → Reflection

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime

from execution_events import ExecutionEvent, ExecutionEventType
from lifecycle_observer import InvariantRegistry, InvariantViolationReport
from observer_engine import observer_engine


# =============================================================================
# Observer Trigger Rule
# =============================================================================

class ObserverTriggerRule(ABC):
    """
    Базовый класс для правила триггерирования Observer.

    Rule = "если событие X, то проверить инварианты Y"
    """

    def __init__(self, rule_id: str, priority: int = 100):
        self.rule_id = rule_id
        self.priority = priority

    @abstractmethod
    def matches(self, event: ExecutionEvent) -> bool:
        """
        Проверить, применимо ли правило к событию.

        Args:
            event: ExecutionEvent

        Returns:
            True если observer должен быть запущен
        """
        pass

    @abstractmethod
    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """
        Получить список инвариантов для проверки.

        Args:
            event: ExecutionEvent

        Returns:
            List of invariant IDs (I7, I9, ORPHANS, etc.)
        """
        pass

    def get_description(self) -> str:
        """Get rule description."""
        return self.__doc__ or f"Rule: {self.rule_id}"


# =============================================================================
# Concrete Rules
# =============================================================================

class OnGoalCompletedRule(ObserverTriggerRule):
    """
    Rule: After goal completion → check all applicable invariants.

    Trigger:
    - event_type = GOAL_COMPLETED

    Invariants to check:
    - I7 (if MANUAL goal)
    - I9 (if has approvals)
    - ORPHANS (if non-atomic parent)
    """

    def __init__(self):
        super().__init__(
            rule_id="on_goal_completed",
            priority=1000  # High priority
        )

    def matches(self, event: ExecutionEvent) -> bool:
        """Match goal completion events."""
        return event.event_type == ExecutionEventType.GOAL_COMPLETED

    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """Determine which invariants to check based on goal context."""
        invariants = []

        # I7: Check if MANUAL goal (need to check from context or DB)
        # For Phase 2.5: assume we can get completion_mode from context
        completion_mode = event.context.get("completion_mode", "unknown")

        if completion_mode == "manual":
            invariants.append("I7")

        # I9: Check if goal has approvals
        has_approvals = event.context.get("has_approvals", False)
        if has_approvals:
            invariants.append("I9")

        # ORPHANS: Check if non-atomic parent
        is_atomic = event.context.get("is_atomic", True)
        if not is_atomic:
            invariants.append("ORPHANS")

        # If no specific info, check all
        if not invariants:
            invariants = ["I7", "I9", "ORPHANS"]

        return invariants


class OnGoalFailedRule(ObserverTriggerRule):
    """
    Rule: After goal failure → check goal state invariants.

    Trigger:
    - event_type = GOAL_FAILED

    Invariants to check:
    - All (to detect corruption/inconsistencies)
    """

    def __init__(self):
        super().__init__(
            rule_id="on_goal_failed",
            priority=900
        )

    def matches(self, event: ExecutionEvent) -> bool:
        """Match goal failure events."""
        return event.event_type == ExecutionEventType.GOAL_FAILED

    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """Check all invariants on failure."""
        return ["I7", "I9", "ORPHANS"]


class OnStepFailedRule(ObserverTriggerRule):
    """
    Rule: After step failure → lightweight check.

    Trigger:
    - event_type = STEP_FAILED

    Invariants to check:
    - None (too early for full check)
    - Just log for now

    Future: Could check partial progress invariants
    """

    def __init__(self):
        super().__init__(
            rule_id="on_step_failed",
            priority=500
        )

    def matches(self, event: ExecutionEvent) -> bool:
        """Match step failure events."""
        return event.event_type == ExecutionEventType.STEP_FAILED

    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """No invariants to check on step failure (too early)."""
        return []


class OnManualGoalCompletionRule(ObserverTriggerRule):
    """
    Rule: Before marking MANUAL goal as DONE → check I7.

    Trigger:
    - Custom event BEFORE setting status=done

    Invariants to check:
    - I7 (exactly one approval required)

    This is a "pre-commit" check to prevent violations.
    """

    def __init__(self):
        super().__init__(
            rule_id="on_manual_goal_completion",
            priority=2000  # Highest priority (safety-critical)
        )

    def matches(self, event: ExecutionEvent) -> bool:
        """Match pre-commit events for MANUAL goals."""
        return (
            event.event_type == ExecutionEventType.STEP_COMPLETED and
            event.context.get("pre_commit_check", False) and
            event.context.get("completion_mode") == "manual"
        )

    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """Check I7 before manual goal completion."""
        return ["I7"]


class OnEveryNStepsRule(ObserverTriggerRule):
    """
    Rule: Every N steps → check metrics invariants.

    Trigger:
    - Every N steps (configurable)

    Invariants to check:
    - None (just metrics for Phase 2.5)
    - Future: Resource usage, performance invariants
    """

    def __init__(self, n_steps: int = 10):
        super().__init__(
            rule_id="on_every_n_steps",
            priority=100  # Low priority
        )
        self.n_steps = n_steps

    def matches(self, event: ExecutionEvent) -> bool:
        """Match every Nth step."""
        if event.event_type not in [ExecutionEventType.STEP_COMPLETED, ExecutionEventType.STEP_FAILED]:
            return False

        step_number = event.step_number or 0
        return step_number % self.n_steps == 0

    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """No invariants, just metrics check (Phase 2.5)."""
        return []


class OnErrorRule(ObserverTriggerRule):
    """
    Rule: On error event → check for corruption.

    Trigger:
    - event_type in [ERROR_EXCEPTION, ERROR_TIMEOUT, ERROR_RESOURCE]

    Invariants to check:
    - All (errors might cause corruption)
    """

    def __init__(self):
        super().__init__(
            rule_id="on_error",
            priority=950  # High priority (error handling)
        )

    def matches(self, event: ExecutionEvent) -> bool:
        """Match error events."""
        return event.event_type in [
            ExecutionEventType.ERROR_EXCEPTION,
            ExecutionEventType.ERROR_TIMEOUT,
            ExecutionEventType.ERROR_RESOURCE
        ]

    def get_invariants_to_check(self, event: ExecutionEvent) -> List[str]:
        """Check all invariants on error."""
        return ["I7", "I9", "ORPHANS"]


# =============================================================================
# Observer Trigger Registry
# =============================================================================

class ObserverTriggerRegistry:
    """
    Реестр правил триггерирования Observer.
    """

    def __init__(self):
        self._rules: List[ObserverTriggerRule] = []

        # Register default rules
        self.register_default_rules()

    def register_default_rules(self):
        """Register built-in rules."""
        self.register(OnGoalCompletedRule())
        self.register(OnGoalFailedRule())
        self.register(OnStepFailedRule())
        self.register(OnManualGoalCompletionRule())
        self.register(OnEveryNStepsRule(n_steps=10))
        self.register(OnErrorRule())

    def register(self, rule: ObserverTriggerRule):
        """Register custom rule."""
        self._rules.append(rule)
        # Sort by priority (highest first)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def list_rules(self) -> List[str]:
        """List all registered rule IDs."""
        return [r.rule_id for r in self._rules]

    def match_rules(self, event: ExecutionEvent) -> List[ObserverTriggerRule]:
        """
        Find all rules that match event.

        Returns rules sorted by priority (highest first).
        """
        matching = []

        for rule in self._rules:
            if rule.matches(event):
                matching.append(rule)

        return matching


# =============================================================================
# Observer Trigger Engine
# =============================================================================

class ObserverTriggerEngine:
    """
    Движок для триггерирования Observer на основе событий.

    Workflow:
    1. Receive ExecutionEvent
    2. Match rules
    3. Determine invariants to check
    4. Run observer checks
    5. Return violation reports
    """

    def __init__(self, registry: ObserverTriggerRegistry = None):
        self.registry = registry or ObserverTriggerRegistry()
        self.observer_engine = observer_engine

    async def trigger_from_event(
        self,
        event: ExecutionEvent
    ) -> List[InvariantViolationReport]:
        """
        Trigger observer checks based on event.

        Args:
            event: ExecutionEvent

        Returns:
            List of InvariantViolationReport (may be empty)
        """
        # Match rules
        matching_rules = self.registry.match_rules(event)

        if not matching_rules:
            # No rules match → no observer checks needed
            return []

        # Collect invariants to check
        invariants_to_check = set()
        for rule in matching_rules:
            rule_invariants = rule.get_invariants_to_check(event)
            invariants_to_check.update(rule_invariants)

        if not invariants_to_check:
            # Rules matched but no invariants to check
            return []

        # Run observer checks
        # For Phase 2.5: run checks on the goal from event
        violation_reports = []

        for invariant_id in invariants_to_check:
            # Check specific invariant for goal
            # Pass event context for pre-commit checks
            report = await self.observer_engine.check_specific_entity(
                invariant_id=invariant_id,
                entity_id=event.goal_id,
                context=event.context
            )

            if report:
                violation_reports.append(report)

        return violation_reports

    def should_trigger_observer(self, event: ExecutionEvent) -> bool:
        """
        Quick check: should observer be triggered for this event?

        Returns True if any rule matches.
        """
        matching_rules = self.registry.match_rules(event)
        return len(matching_rules) > 0


# =============================================================================
# Singleton Instance
# =============================================================================

observer_trigger_engine = ObserverTriggerEngine()
