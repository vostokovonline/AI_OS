"""
Event Ontology - Единый источник истины для всех событий

Все события в системе теперь имеют единую онтологию:

Event
├── type - тип события (строго определён)
├── timestamp - время
├── causal_parent - родительское событие (causal chain)
├── state_delta - изменение состояния
├── decision_context - контекст решения
└── metadata - дополнительные данные

Это заменяет 3 разрозненные системы:
- DecisionTracer
- DecisionTransaction
- IncrementalAttribution

Теперь всё - события в едином causal graph.
"""
from typing import Dict, List, Optional, Any, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4, UUID
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Строгая онтология типов событий"""
    # Жизненный цикл транзакции
    TRANSACTION_START = "transaction_start"
    TRANSACTION_COMMIT = "transaction_commit"
    TRANSACTION_ROLLBACK = "transaction_rollback"
    
    # Состояние системы
    STATE_SNAPSHOT = "state_snapshot"
    STATE_DELTA = "state_delta"
    STATE_QUERY = "state_query"
    
    # Кандидаты и решения
    CANDIDATE_GENERATED = "candidate_generated"
    CANDIDATE_SCORED = "candidate_scored"
    CANDIDATE_SELECTED = "candidate_selected"
    CANDIDATE_REJECTED = "candidate_rejected"
    
    # Симуляция
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_BRANCH = "simulation_branch"
    SIMULATION_COMPLETED = "simulation_completed"
    
    # Аттрибуция
    ATTRIBUTION_BEGIN = "attribution_begin"
    ATTRIBUTION_EDGE_ADDED = "attribution_edge_added"
    ATTRIBUTION_COMPLETED = "attribution_completed"
    
    # Результат
    OUTCOME_RECORDED = "outcome_recorded"
    LEARNING_INTEGRATED = "learning_integrated"
    
    # Внешние события
    EXTERNAL_INPUT = "external_input"
    EXTERNAL_OUTPUT = "external_output"


class EventCategory(Enum):
    """Категории событий для группировки"""
    LIFECYCLE = "lifecycle"
    STATE = "state"
    CANDIDATE = "candidate"
    SIMULATION = "simulation"
    ATTRIBUTION = "attribution"
    OUTCOME = "outcome"
    EXTERNAL = "external"


@dataclass
class CausalLink:
    """Ссылка на родительское событие"""
    parent_event_id: str = ""
    relationship: str = ""  # "caused_by", "enabled", "preceded", "parallel"


@dataclass
class StateDelta:
    """Изменение состояния относительно предыдущего"""
    field_name: str = ""
    old_value: Any = None
    new_value: Any = None
    delta_magnitude: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "field": self.field_name,
            "old": self.old_value,
            "new": self.new_value,
            "delta": self.delta_magnitude
        }


@dataclass
class DecisionContext:
    """Контекст решения в момент события"""
    user_id: str = ""
    task: str = ""
    task_type: str = "default"
    complexity: float = 0.5
    urgency: float = 0.5
    novelty: float = 0.5
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "task": self.task,
            "task_type": self.task_type,
            "complexity": self.complexity,
            "urgency": self.urgency,
            "novelty": self.novelty
        }


@dataclass
class Event:
    """
    Единый примитив для всех событий в системе.
    
    Каждый Event содержит:
    - type: строго определённый тип из EventType
    - timestamp: время создания
    - causal_parents: список родительских событий
    - state_delta: изменение состояния
    - decision_context: контекст решения
    - metadata: дополнительные данные
    
    Это заменяет разрозненные системы логирования.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    category: str = EventCategory.LIFECYCLE.value
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    transaction_id: str = ""
    
    causal_parents: List[CausalLink] = field(default_factory=list)
    
    state_delta: Optional[StateDelta] = None
    decision_context: Optional[DecisionContext] = None
    
    # Данные события (зависит от типа)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Вычисляемые свойства
    sequence: int = 0
    depth: int = 0
    
    # Интеграция с существующими системами
    legacy_trace_id: Optional[str] = None
    legacy_decision_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "category": self.category,
            "timestamp": self.timestamp.isoformat(),
            "transaction_id": self.transaction_id,
            "causal_parents": [
                {"parent_id": c.parent_event_id, "relationship": c.relationship}
                for c in self.causal_parents
            ],
            "state_delta": self.state_delta.to_dict() if self.state_delta else None,
            "decision_context": self.decision_context.to_dict() if self.decision_context else None,
            "data": self.data,
            "sequence": self.sequence,
            "depth": self.depth,
            "legacy_trace_id": self.legacy_trace_id,
            "legacy_decision_id": self.legacy_decision_id,
        }
    
    def get_causal_path(self) -> List[str]:
        """Получить полный causal path от корня"""
        path = [self.id]
        for parent in self.causal_parents:
            path.append(parent.parent_event_id)
        return list(reversed(path))
    
    @classmethod
    def create(
        cls,
        event_type: str,
        category: str,
        transaction_id: str = "",
        causal_parent: Optional[str] = None,
        relationship: str = "caused_by",
        **kwargs
    ) -> "Event":
        """Фабрика для создания событий"""
        event = cls(
            event_type=event_type,
            category=category,
            transaction_id=transaction_id,
            **kwargs
        )
        
        if causal_parent:
            event.causal_parents.append(CausalLink(
                parent_event_id=causal_parent,
                relationship=relationship
            ))
        
        return event


@dataclass
class EventStream:
    """
    Поток событий для одной транзакции.
    
    Это заменяет:
    - DecisionTracer
    - TransactionContext.reasoning_events
    - IncrementalAttribution.events
    
    Всё теперь - события в EventStream.
    """
    transaction_id: str = ""
    stream_id: str = field(default_factory=lambda: str(uuid4()))
    
    events: List[Event] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    status: str = "active"  # active, committed, rolled_back
    
    def add(self, event: Event) -> None:
        event.sequence = len(self.events)
        self.events.append(event)
        logger.debug("event_added", event_type=event.event_type, sequence=event.sequence)
    
    def commit(self) -> None:
        self.completed_at = datetime.utcnow()
        self.status = "committed"
        logger.info("event_stream_committed", transaction_id=self.transaction_id)
    
    def rollback(self) -> None:
        self.completed_at = datetime.utcnow()
        self.status = "rolled_back"
        logger.info("event_stream_rolled_back", transaction_id=self.transaction_id)
    
    def get_events_by_type(self, event_type: str) -> List[Event]:
        return [e for e in self.events if e.event_type == event_type]
    
    def get_causal_graph(self) -> Dict:
        """Построить causal graph из событий"""
        nodes = {}
        edges = []
        
        for event in self.events:
            nodes[event.id] = {
                "type": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "data": event.data
            }
            
            for parent in event.causal_parents:
                edges.append({
                    "from": parent.parent_event_id,
                    "to": event.id,
                    "relationship": parent.relationship
                })
        
        return {"nodes": nodes, "edges": edges}
    
    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "stream_id": self.stream_id,
            "status": self.status,
            "event_count": len(self.events),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "events": [e.to_dict() for e in self.events]
        }


class EventOntology:
    """
    Единая онтология событий.
    
    Заменяет 3 разрозненные системы:
    - DecisionTracer
    - DecisionTransaction  
    - IncrementalAttribution
    
    Все события теперь проходят через единый EventOntology.
    """
    
    def __init__(self):
        self.streams: Dict[str, EventStream] = {}
        self.event_index: Dict[str, Event] = {}
        logger.info("event_ontology_initialized")
    
    def begin_stream(self, transaction_id: str) -> EventStream:
        """Начать новый поток событий для транзакции"""
        stream = EventStream(transaction_id=transaction_id)
        self.streams[transaction_id] = stream
        logger.info("stream_started", transaction_id=transaction_id)
        return stream
    
    def emit(
        self,
        transaction_id: str,
        event_type: str,
        category: str,
        data: Optional[Dict] = None,
        causal_parent: Optional[str] = None,
        decision_context: Optional[DecisionContext] = None,
        state_delta: Optional[StateDelta] = None
    ) -> Event:
        """Испустить событие в поток"""
        
        if transaction_id not in self.streams:
            self.begin_stream(transaction_id)
        
        stream = self.streams[transaction_id]
        
        event = Event.create(
            event_type=event_type,
            category=category,
            transaction_id=transaction_id,
            causal_parent=causal_parent,
            data=data or {},
            decision_context=decision_context,
            state_delta=state_delta
        )
        
        if causal_parent and causal_parent in self.event_index:
            parent_event = self.event_index[causal_parent]
            event.depth = parent_event.depth + 1
        
        stream.add(event)
        self.event_index[event.id] = event
        
        logger.debug("event_emitted", transaction_id=transaction_id, event_type=event_type)
        
        return event
    
    def commit_stream(self, transaction_id: str) -> Optional[EventStream]:
        """Завершить поток событий"""
        stream = self.streams.get(transaction_id)
        if stream:
            stream.commit()
            logger.info("stream_committed", transaction_id=transaction_id)
        return stream
    
    def rollback_stream(self, transaction_id: str) -> Optional[EventStream]:
        """Откатить поток событий"""
        stream = self.streams.get(transaction_id)
        if stream:
            stream.rollback()
            logger.info("stream_rolled_back", transaction_id=transaction_id)
        return stream
    
    def get_stream(self, transaction_id: str) -> Optional[EventStream]:
        """Получить поток по ID транзакции"""
        return self.streams.get(transaction_id)
    
    def get_event(self, event_id: str) -> Optional[Event]:
        """Получить событие по ID"""
        return self.event_index.get(event_id)
    
    def query(
        self,
        event_type: Optional[str] = None,
        category: Optional[str] = None,
        transaction_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """Запрос событий по фильтрам"""
        results = list(self.event_index.values())
        
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if category:
            results = [e for e in results if e.category == category]
        if transaction_id:
            results = [e for e in results if e.transaction_id == transaction_id]
        
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results[:limit]
    
    def get_statistics(self) -> Dict:
        """Получить статистику по событиям"""
        events = list(self.event_index.values())
        
        type_counts: Dict[str, int] = {}
        category_counts: Dict[str, int] = {}
        
        for event in events:
            type_counts[event.event_type] = type_counts.get(event.event_type, 0) + 1
            category_counts[event.category] = category_counts.get(event.category, 0) + 1
        
        return {
            "total_events": len(events),
            "active_streams": len([s for s in self.streams.values() if s.status == "active"]),
            "committed_streams": len([s for s in self.streams.values() if s.status == "committed"]),
            "by_type": type_counts,
            "by_category": category_counts,
        }


# Глобальный экземпляр
_event_ontology: Optional[EventOntology] = None


def get_event_ontology() -> EventOntology:
    """Получить глобальный EventOntology"""
    global _event_ontology
    if _event_ontology is None:
        _event_ontology = EventOntology()
    return _event_ontology


# Утилиты для удобного создания событий
def emit_state_snapshot(
    transaction_id: str,
    state: Dict[str, Any],
    causal_parent: Optional[str] = None
) -> Event:
    """Испустить событие снимка состояния"""
    ontology = get_event_ontology()
    
    state_delta = None
    if state:
        first_key = list(state.keys())[0] if state else None
        if first_key:
            state_delta = StateDelta(
                field_name=first_key,
                new_value=state[first_key]
            )
    
    return ontology.emit(
        transaction_id=transaction_id,
        event_type=EventType.STATE_SNAPSHOT.value,
        category=EventCategory.STATE.value,
        data=state,
        causal_parent=causal_parent,
        state_delta=state_delta
    )


def emit_candidate(
    transaction_id: str,
    candidate_id: str,
    action_type: str,
    score: float,
    causal_parent: Optional[str] = None
) -> Event:
    """Испустить событие генерации кандидата"""
    ontology = get_event_ontology()
    
    return ontology.emit(
        transaction_id=transaction_id,
        event_type=EventType.CANDIDATE_GENERATED.value,
        category=EventCategory.CANDIDATE.value,
        data={
            "candidate_id": candidate_id,
            "action_type": action_type,
            "score": score
        },
        causal_parent=causal_parent
    )


def emit_selection(
    transaction_id: str,
    action: str,
    confidence: float,
    alternatives: List[Dict],
    causal_parent: Optional[str] = None
) -> Event:
    """Испустить событие выбора действия"""
    ontology = get_event_ontology()
    
    return ontology.emit(
        transaction_id=transaction_id,
        event_type=EventType.CANDIDATE_SELECTED.value,
        category=EventCategory.CANDIDATE.value,
        data={
            "action": action,
            "confidence": confidence,
            "alternatives": alternatives
        },
        causal_parent=causal_parent
    )


def emit_outcome(
    transaction_id: str,
    outcome: str,
    score: float,
    causal_parent: Optional[str] = None
) -> Event:
    """Испустить событие записи результата"""
    ontology = get_event_ontology()
    
    return ontology.emit(
        transaction_id=transaction_id,
        event_type=EventType.OUTCOME_RECORDED.value,
        category=EventCategory.OUTCOME.value,
        data={
            "outcome": outcome,
            "score": score
        },
        causal_parent=causal_parent
    )