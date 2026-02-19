"""
Execution Events (Phase 2.5.1)

События执行的 систему для Integration с Observer + Reflection.

Цель: стандартизировать что именно генерирует executor
и как Observer/Reflection на это реагируют.

Architecture:
  Executor emits ExecutionEvent
         ↓
  Observer + Reflection consume event
         ↓
  Actions executed
         ↓
  Executor receives feedback (continue/stop/escalate)

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# =============================================================================
# Execution Event Type
# =============================================================================

class ExecutionEventType(str, Enum):
    """
    Типы событий выполнения.

    Categories:
    - STEP: Intermediate events (each step completion)
    - GOAL: Goal lifecycle events (start, complete, fail)
    - ERROR: Error events (exceptions, timeouts)
    - METRIC: Metric events (performance, resource usage)
    """
    # Step events
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_PARTIAL = "step_partial"  # Partial success

    # Goal lifecycle events
    GOAL_STARTED = "goal_started"
    GOAL_COMPLETED = "goal_completed"
    GOAL_FAILED = "goal_failed"
    GOAL_ABORTED = "goal_aborted"

    # Error events
    ERROR_EXCEPTION = "error_exception"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_RESOURCE = "error_resource"  # Out of memory, etc.

    # Metric events
    METRIC_PERFORMANCE = "metric_performance"
    METRIC_RESOURCE = "metric_resource"


# =============================================================================
# Execution Event (Main Contract)
# =============================================================================

class ExecutionEvent(BaseModel):
    """
    Событие выполнения генерируемое Executor'ом.

    Это входной контракт для Observer + Reflection integration.
    """
    # Event identification
    event_id: str = Field(..., description="Unique event ID")
    event_type: ExecutionEventType = Field(..., description="Type of event")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Entity identification
    goal_id: uuid.UUID = Field(..., description="Goal UUID")
    goal_title: str = Field(..., description="Goal title")
    step_id: Optional[str] = Field(None, description="Step ID (if step event)")
    step_number: Optional[int] = Field(None, description="Step number (1-based)")

    # Execution result
    result: Optional[str] = Field(None, description="Result: success | failure | partial")
    message: Optional[str] = Field(None, description="Human-readable message")

    # Metrics & context
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    # Agent info
    agent_role: Optional[str] = Field(None, description="Agent role (CODER, RESEARCHER, etc.)")
    agent_name: Optional[str] = Field(None, description="Agent name")

    # Artifacts produced
    artifacts: List[str] = Field(default_factory=list, description="Artifact IDs produced")

    # Error details (if error event)
    error_type: Optional[str] = Field(None, description="Exception type")
    error_message: Optional[str] = Field(None, description="Error message")
    error_traceback: Optional[str] = Field(None, description="Error traceback")

    class Config:
        from_attributes = True


# =============================================================================
# Execution Event Context
# =============================================================================

class ExecutionEventContext(BaseModel):
    """
    Контекст выполнения для событий.

    Содержит информацию о состоянии выполнения
    которая может быть полезна для Observer/Reflection.
    """
    goal_id: uuid.UUID
    goal_status: str  # current status of goal
    goal_progress: float  # 0.0 to 1.0
    steps_total: int
    steps_completed: int
    steps_failed: int
    started_at: datetime
    last_activity: datetime
    agent_stack: List[str] = Field(default_factory=list, description="Agent call stack")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Execution Event Emitter
# =============================================================================

class ExecutionEventEmitter:
    """
    Генератор событий выполнения для Executor.

    Используется внутри goal_executor для генерации
    стандартизированных событий.
    """

    def _create_event_id(self) -> str:
        """Generate unique event ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
        random_suffix = uuid.uuid4().hex[:8]
        return f"evt_{timestamp}_{random_suffix}"

    def emit_step_started(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        step_id: str,
        step_number: int,
        agent_role: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEvent:
        """Emit step started event."""
        return ExecutionEvent(
            event_id=self._create_event_id(),
            event_type=ExecutionEventType.STEP_STARTED,
            goal_id=goal_id,
            goal_title=goal_title,
            step_id=step_id,
            step_number=step_number,
            result="started",
            message=f"Step {step_number} started: {step_id}",
            agent_role=agent_role,
            context=context or {}
        )

    def emit_step_completed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        step_id: str,
        step_number: int,
        agent_role: str,
        artifacts: List[str] = None,
        metrics: Dict[str, Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEvent:
        """Emit step completed event."""
        return ExecutionEvent(
            event_id=self._create_event_id(),
            event_type=ExecutionEventType.STEP_COMPLETED,
            goal_id=goal_id,
            goal_title=goal_title,
            step_id=step_id,
            step_number=step_number,
            result="success",
            message=f"Step {step_number} completed: {step_id}",
            agent_role=agent_role,
            artifacts=artifacts or [],
            metrics=metrics or {},
            context=context or {}
        )

    def emit_step_failed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        step_id: str,
        step_number: int,
        agent_role: str,
        error_type: str,
        error_message: str,
        error_traceback: str = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEvent:
        """Emit step failed event."""
        return ExecutionEvent(
            event_id=self._create_event_id(),
            event_type=ExecutionEventType.STEP_FAILED,
            goal_id=goal_id,
            goal_title=goal_title,
            step_id=step_id,
            step_number=step_number,
            result="failure",
            message=f"Step {step_number} failed: {error_message}",
            agent_role=agent_role,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            context=context or {}
        )

    def emit_goal_completed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        steps_total: int,
        steps_completed: int,
        steps_failed: int,
        artifacts: List[str] = None,
        metrics: Dict[str, Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEvent:
        """Emit goal completed event."""
        return ExecutionEvent(
            event_id=self._create_event_id(),
            event_type=ExecutionEventType.GOAL_COMPLETED,
            goal_id=goal_id,
            goal_title=goal_title,
            result="success",
            message=f"Goal completed: {steps_completed}/{steps_total} steps successful",
            artifacts=artifacts or [],
            metrics={
                **(metrics or {}),
                "steps_total": steps_total,
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
                "success_rate": steps_completed / steps_total if steps_total > 0 else 0.0
            },
            context=context or {}
        )

    def emit_goal_failed(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        steps_total: int,
        steps_completed: int,
        failure_reason: str,
        error_type: str = None,
        error_message: str = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEvent:
        """Emit goal failed event."""
        return ExecutionEvent(
            event_id=self._create_event_id(),
            event_type=ExecutionEventType.GOAL_FAILED,
            goal_id=goal_id,
            goal_title=goal_title,
            result="failure",
            message=f"Goal failed: {failure_reason}",
            metrics={
                "steps_total": steps_total,
                "steps_completed": steps_completed,
                "steps_failed": steps_total - steps_completed
            },
            error_type=error_type,
            error_message=error_message,
            context=context or {}
        )

    def emit_error(
        self,
        goal_id: uuid.UUID,
        goal_title: str,
        error_type: str,
        error_message: str,
        error_traceback: str = None,
        event_type: ExecutionEventType = ExecutionEventType.ERROR_EXCEPTION,
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionEvent:
        """Emit error event."""
        return ExecutionEvent(
            event_id=self._create_event_id(),
            event_type=event_type,
            goal_id=goal_id,
            goal_title=goal_title,
            result="error",
            message=f"Error: {error_message}",
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            context=context or {}
        )


# =============================================================================
# Execution Event Store (In-Memory for Phase 2.5)
# =============================================================================

class ExecutionEventStore:
    """
    Хранилище событий выполнения.

    Phase 2.5: In-memory (for testing)
    Future: Persistent (Redis, DB, etc.)
    """

    def __init__(self, max_size: int = 10000):
        self._events: List[ExecutionEvent] = []
        self._max_size = max_size

    def add(self, event: ExecutionEvent):
        """Add event to store."""
        self._events.append(event)

        # Keep only last N events
        if len(self._events) > self._max_size:
            self._events = self._events[-self._max_size:]

    def get_by_goal(self, goal_id: uuid.UUID, limit: int = 100) -> List[ExecutionEvent]:
        """Get events for specific goal."""
        return [
            event for event in self._events
            if event.goal_id == goal_id
        ][:limit]

    def get_recent(self, limit: int = 100) -> List[ExecutionEvent]:
        """Get most recent events."""
        return self._events[-limit:]

    def get_by_type(self, event_type: ExecutionEventType, limit: int = 100) -> List[ExecutionEvent]:
        """Get events by type."""
        return [
            event for event in self._events
            if event.event_type == event_type
        ][:limit]

    def clear(self):
        """Clear all events."""
        self._events.clear()


# =============================================================================
# Singleton Instances
# =============================================================================

execution_event_emitter = ExecutionEventEmitter()
execution_event_store = ExecutionEventStore()


# =============================================================================
# Convenience Functions
# =============================================================================

def emit_execution_event(event: ExecutionEvent):
    """Emit event to store (convenience function)."""
    execution_event_store.add(event)


def get_goal_execution_history(goal_id: uuid.UUID, limit: int = 100) -> List[ExecutionEvent]:
    """Get execution history for goal (convenience function)."""
    return execution_event_store.get_by_goal(goal_id, limit)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ExecutionEventType",
    "ExecutionEvent",
    "ExecutionEventContext",
    "ExecutionEventEmitter",
    "ExecutionEventStore",
    "execution_event_emitter",
    "execution_event_store",
    "emit_execution_event",
    "get_goal_execution_history",
]
