"""
Reflection System (Phase 2.4.1)

Контракты и структуры для системы рефлексии.

Reflection = принятие решений на основе фактов от Observer.
НЕ LLM-чат, НЕ саморефлексия в вакууме.

Architecture:
  Observer → "что не так" (факты)
  Reflection → "что с этим делать" (решения)

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field

from lifecycle_observer import InvariantViolationReport, InvariantSeverity


# =============================================================================
# Trigger Source Enum
# =============================================================================

class TriggerSource(str, Enum):
    """
    Источник сигнала для reflection.

    observer — Сигнал от Observer (инвариант нарушен)
    execution — Завершение выполнения goal (success/failure)
    external — Внешний сигнал (admin API, webhook, etc.)
    manual — Ручной запуск админом
    scheduled — Плановый запуск (Celery)
    """
    OBSERVER = "observer"
    EXECUTION = "execution"
    EXTERNAL = "external"
    MANUAL = "manual"
    SCHEDULED = "scheduled"


# =============================================================================
# Reflection Signal (Input Contract)
# =============================================================================

class ReflectionSignal(BaseModel):
    """
    Входной сигнал для системы рефлексии.

    Это факты, на основе которых Reflection принимает решения.
    НЕ содержит решений — только данные.
    """

    # Source identification
    source: TriggerSource = Field(..., description="Who triggered this reflection")
    trigger_id: str = Field(..., description="Unique trigger ID")
    triggered_at: datetime = Field(default_factory=datetime.now)

    # Observer-specific fields (if source = OBSERVER)
    invariant_id: Optional[str] = Field(None, description="Invariant ID (I7, I9, ORPHANS)")
    invariant_name: Optional[str] = Field(None, description="Invariant name")
    severity: Optional[InvariantSeverity] = Field(None, description="HARD or SOFT")
    entity: Optional[str] = Field(None, description="Entity type (goal, approval, etc.)")
    entity_id: Optional[uuid.UUID] = Field(None, description="Entity UUID")
    violation_details: Optional[Dict[str, Any]] = Field(None, description="Violation details from Observer")

    # Execution-specific fields (if source = EXECUTION)
    execution_goal_id: Optional[uuid.UUID] = Field(None, description="Executed goal ID")
    execution_status: Optional[str] = Field(None, description="execution status (success, failure, aborted)")
    execution_outcome: Optional[Dict[str, Any]] = Field(None, description="Execution outcome data")

    # External/Manual fields
    external_reason: Optional[str] = Field(None, description="Reason for external/manual trigger")
    external_context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

    # Common fields
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata for tracking")

    class Config:
        from_attributes = True

    @classmethod
    def from_observer_report(cls, report: InvariantViolationReport, trigger_id: str) -> "ReflectionSignal":
        """
        Create ReflectionSignal from InvariantViolationReport.

        Это основной способ создания сигналов от Observer.
        """
        return cls(
            source=TriggerSource.OBSERVER,
            trigger_id=trigger_id,
            invariant_id=report.invariant_id,
            invariant_name=report.invariant_name,
            severity=report.severity,
            entity=report.entity,
            entity_id=report.entity_id,
            violation_details=report.details,
            context={
                "passed": report.passed,
                "message": report.message
            },
            metadata={
                "detected_at": report.detected_at.isoformat()
            }
        )

    @classmethod
    def from_execution(cls, goal_id: uuid.UUID, status: str, outcome: Dict[str, Any], trigger_id: str) -> "ReflectionSignal":
        """
        Create ReflectionSignal from goal execution result.
        """
        return cls(
            source=TriggerSource.EXECUTION,
            trigger_id=trigger_id,
            execution_goal_id=goal_id,
            execution_status=status,
            execution_outcome=outcome,
            context={
                "status": status
            }
        )

    @classmethod
    def from_manual_trigger(cls, reason: str, context: Dict[str, Any], trigger_id: str) -> "ReflectionSignal":
        """
        Create ReflectionSignal from manual admin trigger.
        """
        return cls(
            source=TriggerSource.MANUAL,
            trigger_id=trigger_id,
            external_reason=reason,
            external_context=context
        )


# =============================================================================
# Reflection Action Types (Phase 2.4.3 Preview)
# =============================================================================

class ActionType(str, Enum):
    """
    Тип действия, которое Reflection может предпринять.

    КРИТИЧЕСКИ ВАЖНО: Это orchestration, NOT direct DB updates!

    Examples:
    - ANNOTATE — Добавить метаданные к goal
    - FREEZE — Заморозить выполнение (заблокировать)
    - REQUEST_REVIEW — Запросить human review
    - SPAWN_ANALYSIS — Создать задачу для анализа
    - LOG — Просто залогировать (no action)
    - ESCALATE — Повысить приоритет / severity
    """
    ANNOTATE = "annotate"           # Add metadata/annotation to entity
    FREEZE = "freeze"               # Block execution (set status = "frozen")
    REQUEST_REVIEW = "request_review"  # Flag for human review
    SPAWN_ANALYSIS = "spawn_analysis"  # Create analysis goal/task
    LOG = "log"                     # Audit log only (no action)
    ESCALATE = "escalate"           # Escalate priority/severity
    SCHEDULE_REEVALUATION = "schedule_reevaluation"  # Recheck after delay


# =============================================================================
# Reflection Decision (Output Contract)
# =============================================================================

class ReflectionDecision(BaseModel):
    """
    Решение, принятое Reflection.

    Содержит: что делать, почему, с какими параметрами.
    """

    # Decision identification
    decision_id: str = Field(..., description="Unique decision ID")
    trigger_id: str = Field(..., description="Trigger that caused this decision")
    signal_source: TriggerSource = Field(..., description="Source of original signal")

    # Decision content
    action: ActionType = Field(..., description="Action to take")
    target_entity: str = Field(..., description="Entity type (goal, approval, etc.)")
    target_entity_id: uuid.UUID = Field(..., description="Entity UUID")

    # Decision rationale
    reason: str = Field(..., description="Why this action was chosen")
    policy_applied: str = Field(..., description="Policy ID that generated this decision")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in decision")

    # Action parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action-specific parameters")

    # Metadata
    decided_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = Field(None, description="Decision expiration (if any)")
    executed: bool = Field(default=False, description="Has action been executed?")
    execution_result: Optional[Dict[str, Any]] = Field(None, description="Execution result (if executed)")

    class Config:
        from_attributes = True


# =============================================================================
# Reflection Context (Phase 2.4.2 Preview)
# =============================================================================

class ReflectionContext(BaseModel):
    """
    Контекст для принятия решений.

    Содержит всё, что нужно Policy Engine для решения:
    - Входной сигнал
    - История сущности
    - Текущее состояние
    - Системные настройки
    """

    signal: ReflectionSignal = Field(..., description="Input signal")
    entity_state: Optional[Dict[str, Any]] = Field(None, description="Current entity state")
    entity_history: Optional[List[Dict[str, Any]]] = Field(None, description="Entity history (past decisions, etc.)")
    system_state: Optional[Dict[str, Any]] = Field(None, description="System-level state (load, queues, etc.)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Reflection system config")


# =============================================================================
# Reflection Trigger (Phase 2.4.4 Preview)
# =============================================================================

class ReflectionTrigger(BaseModel):
    """
    Триггер для запуска Reflection Loop.

    Создаётся когда происходит событие (observer violation, execution end, etc.)
    """
    trigger_id: str = Field(..., description="Unique trigger ID")
    source: TriggerSource = Field(..., description="Trigger source")
    signal: Optional[ReflectionSignal] = Field(None, description="Input signal (if attached)")
    created_at: datetime = Field(default_factory=datetime.now)
    processed: bool = Field(default=False, description="Has reflection processed this?")
    decisions: List[ReflectionDecision] = Field(default_factory=list, description="Generated decisions")


# =============================================================================
# Factory Functions
# =============================================================================

def create_trigger_id() -> str:
    """Generate unique trigger ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_suffix}"


def create_decision_id() -> str:
    """Generate unique decision ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = uuid.uuid4().hex[:8]
    return f"dec_{timestamp}_{random_suffix}"


# =============================================================================
# Signal Validators
# =============================================================================

def validate_observer_signal(signal: ReflectionSignal) -> bool:
    """
    Проверить что сигнал от Observer валиден.

    Requirements:
    - source = OBSERVER
    - invariant_id, invariant_name present
    - entity, entity_id present
    - severity present
    """
    if signal.source != TriggerSource.OBSERVER:
        return False

    required_fields = ["invariant_id", "invariant_name", "entity", "entity_id", "severity"]
    for field in required_fields:
        if getattr(signal, field) is None:
            return False

    return True


def validate_execution_signal(signal: ReflectionSignal) -> bool:
    """
    Проверить что сигнал от Execution валиден.
    """
    if signal.source != TriggerSource.EXECUTION:
        return False

    required_fields = ["execution_goal_id", "execution_status"]
    for field in required_fields:
        if getattr(signal, field) is None:
            return False

    return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "TriggerSource",
    "ReflectionSignal",
    "ActionType",
    "ReflectionDecision",
    "ReflectionContext",
    "ReflectionTrigger",
    "create_trigger_id",
    "create_decision_id",
    "validate_observer_signal",
    "validate_execution_signal",
]
