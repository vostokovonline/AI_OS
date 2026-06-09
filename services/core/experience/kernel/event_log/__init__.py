"""
Pure Event Log - The single source of truth.

Stage: Four-Layer Architecture

This is NOT a runtime buffer.
This is NOT a mutable state.
This is THE immutable event store.

Principles:
1. Events are append-only
2. Events are immutable after creation
3. Events carry full causal metadata
4. Events never mutate state
5. State is ALWAYS derived via reducers

Architecture:
    Events → Reducers → State
         ↑
    Adapters ← Policies
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, FrozenSet, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


class EventCategory(Enum):
    """Event categories for routing and filtering"""
    COGNITIVE = "cognitive"      # Belief, memory, contradiction
    EXECUTION = "execution"       # Action, result, failure
    IDENTITY = "identity"        # Self, genome, pressure
    META = "meta"                 # Reflection, planning, strategy
    SYSTEM = "system"             # Health, metrics, errors


@dataclass(frozen=True)
class CausalMetadata:
    """
    Full causal chain metadata for every event.
    
    Enables:
    - Tracing decision lineage
    - Finding root causes
    - Self-reflection
    - Explaining decisions
    """
    event_id: str
    causation_id: str  # What caused this event
    correlation_id: str  # Groups related events
    trace_id: str  # Full execution trace
    parent_event_id: str  # Immediate parent
    lineage_depth: int  # How many generations
    
    @staticmethod
    def create_root(event_id: str) -> 'CausalMetadata':
        """Create root metadata for first event in chain"""
        return CausalMetadata(
            event_id=event_id,
            causation_id="",
            correlation_id=event_id,
            trace_id=event_id,
            parent_event_id="",
            lineage_depth=0
        )
    
    def with_causation(self, cause_id: str) -> 'CausalMetadata':
        """Create child metadata"""
        return CausalMetadata(
            event_id=self.event_id,
            causation_id=cause_id,
            correlation_id=self.correlation_id,
            trace_id=f"{self.trace_id}/{self.event_id}",
            parent_event_id=cause_id,
            lineage_depth=self.lineage_depth + 1
        )
    
    def get_ancestors(self) -> List[str]:
        """Get ancestor chain"""
        if not self.trace_id:
            return []
        return self.trace_id.split("/")


@dataclass(frozen=True)
class ImmutableEvent:
    """
    An immutable fact in the event log.
    
    This is THE truth. Not derived, not mutable.
    All state comes from these events via reducers.
    
    Key properties:
    - event_id: deterministic hash of content
    - category: routing and filtering
    - causal: full lineage tracking
    - payload: all event data
    - schema_version: for replay compatibility
    """
    event_type: str
    category: str
    timestamp: str
    causal: CausalMetadata
    schema_version: int
    payload: FrozenSet[Tuple[str, Any]]
    
    def __post_init__(self):
        if not isinstance(self.payload, FrozenSet):
            object.__setattr__(self, 'payload', frozenset(self.payload))
    
    @staticmethod
    def compute_id(event_type: str, category: str, payload: Dict[str, Any], timestamp: str) -> str:
        """Deterministic event ID from content"""
        content = {
            "type": event_type,
            "category": category,
            "payload": dict(sorted(payload.items())),
            "ts": timestamp
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage"""
        return {
            "event_type": self.event_type,
            "category": self.category,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "causal": {
                "event_id": self.causal.event_id,
                "causation_id": self.causal.causation_id,
                "correlation_id": self.causal.correlation_id,
                "trace_id": self.causal.trace_id,
                "parent_event_id": self.causal.parent_event_id,
                "lineage_depth": self.causal.lineage_depth
            },
            "payload": dict(self.payload)
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ImmutableEvent':
        """Deserialize from dict"""
        causal_data = data.get("causal", {})
        causal = CausalMetadata(
            event_id=causal_data.get("event_id", ""),
            causation_id=causal_data.get("causation_id", ""),
            correlation_id=causal_data.get("correlation_id", ""),
            trace_id=causal_data.get("trace_id", ""),
            parent_event_id=causal_data.get("parent_event_id", ""),
            lineage_depth=causal_data.get("lineage_depth", 0)
        )
        
        return ImmutableEvent(
            event_type=data["event_type"],
            category=data.get("category", "system"),
            timestamp=data.get("timestamp", ""),
            causal=causal,
            schema_version=data.get("schema_version", 1),
            payload=frozenset(sorted(data.get("payload", {}).items()))
        )
    
    def get_payload_dict(self) -> Dict[str, Any]:
        return dict(self.payload)
    
    def is_cognitive(self) -> bool:
        return self.category == EventCategory.COGNITIVE.value
    
    def is_execution(self) -> bool:
        return self.category == EventCategory.EXECUTION.value
    
    def is_identity(self) -> bool:
        return self.category == EventCategory.IDENTITY.value
    
    def is_meta(self) -> bool:
        return self.category == EventCategory.META.value


@dataclass(frozen=True)
class EventStream:
    """
    An immutable ordered stream of events.
    
    Each stream is a separate partition (identity, execution, memory, etc.)
    Events within stream are ordered by position.
    """
    stream_id: str
    events: Tuple[ImmutableEvent, ...]  # Ordered by position
    version: int
    
    @staticmethod
    def compute_hash(stream: 'EventStream') -> str:
        """Deterministic hash of stream state"""
        if not stream.events:
            return hashlib.sha256(b"empty").hexdigest()
        
        event_data = [(e.causal.event_id, e.event_type) for e in stream.events]
        content = {
            "stream_id": stream.stream_id,
            "version": stream.version,
            "events": event_data
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()


class EventLog:
    """
    Pure event log - THE source of truth.
    
    This is NOT mutable runtime state.
    This is append-only immutable event store.
    
    Operations:
    - append: Add new event (returns new EventLog)
    - get_stream: Get events from stream
    - query: Filter events by criteria
    - get_causal_chain: Get event lineage
    
    NO mutation. All operations return new instances.
    """
    
    def __init__(self, streams: Dict[str, Tuple[ImmutableEvent, ...]]):
        self._streams = streams
    
    @staticmethod
    def empty() -> 'EventLog':
        """Create empty event log"""
        return EventLog(streams={})
    
    def with_event(self, stream_id: str, event: ImmutableEvent) -> 'EventLog':
        """
        Append event to stream.
        
        Returns NEW EventLog (immutable).
        """
        stream_events = self._streams.get(stream_id, ())
        new_stream = stream_events + (event,)
        
        new_streams = {**self._streams, stream_id: new_stream}
        return EventLog(streams=new_streams)
    
    def get_stream(self, stream_id: str) -> Tuple[ImmutableEvent, ...]:
        """Get all events from stream"""
        return self._streams.get(stream_id, ())
    
    def get_stream_since(self, stream_id: str, position: int) -> Tuple[ImmutableEvent, ...]:
        """Get events from position onwards"""
        stream = self.get_stream(stream_id)
        return tuple(e for e in stream if self._get_position(e) >= position)
    
    def _get_position(self, event: ImmutableEvent) -> int:
        """Get event position in stream"""
        return event.causal.lineage_depth
    
    def query(
        self,
        category: Optional[str] = None,
        event_type: Optional[str] = None,
        correlation_id: Optional[str] = None,
        causation_id: Optional[str] = None,
        time_range: Optional[Tuple[str, str]] = None
    ) -> List[ImmutableEvent]:
        """
        Query events by criteria.
        
        Returns matching events from all streams.
        """
        results = []
        
        for stream_events in self._streams.values():
            for event in stream_events:
                if category and event.category != category:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if correlation_id and event.causal.correlation_id != correlation_id:
                    continue
                if causation_id and event.causal.causation_id == causation_id:
                    results.append(event)
                    continue
                if time_range:
                    start, end = time_range
                    if not (start <= event.timestamp <= end):
                        continue
                
                if category or event_type or correlation_id or time_range:
                    results.append(event)
        
        return results
    
    def get_causal_chain(self, event_id: str) -> List[ImmutableEvent]:
        """
        Get full causal chain for event.
        
        Returns all ancestor events.
        """
        for stream_events in self._streams.values():
            for event in stream_events:
                if event.causal.event_id == event_id:
                    ancestors = []
                    trace = event.causal.get_ancestors()
                    for ancestor_id in trace:
                        ancestor = self._find_event(ancestor_id)
                        if ancestor:
                            ancestors.append(ancestor)
                    return ancestors
        return []
    
    def _find_event(self, event_id: str) -> Optional[ImmutableEvent]:
        """Find event by ID across all streams"""
        for stream_events in self._streams.values():
            for event in stream_events:
                if event.causal.event_id == event_id:
                    return event
        return None
    
    def get_correlation_events(self, correlation_id: str) -> List[ImmutableEvent]:
        """Get all events with same correlation ID"""
        return self.query(correlation_id=correlation_id)
    
    def compute_log_hash(self) -> str:
        """Compute deterministic hash of entire log"""
        content = {
            "streams": {
                stream_id: EventStream.compute_hash(EventStream(stream_id, events))
                for stream_id, events in self._streams.items()
            }
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
    
    def get_stream_count(self) -> int:
        return len(self._streams)
    
    def get_total_events(self) -> int:
        return sum(len(events) for events in self._streams.values())


# Event Factory - Creates events with proper causal metadata

class EventFactory:
    """
    Factory for creating events with proper causal metadata.
    
    Handles:
    - Event ID generation (deterministic)
    - Causal chain tracking
    - Correlation ID management
    - Schema versioning
    """
    
    def __init__(self, trace_id: Optional[str] = None):
        self._trace_id = trace_id or str(datetime.utcnow().timestamp())
        self._correlation_counter = 0
    
    def create_event(
        self,
        event_type: str,
        category: str,
        payload: Dict[str, Any],
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> ImmutableEvent:
        """Create new event with causal metadata"""
        timestamp = datetime.utcnow().isoformat()
        
        event_id = ImmutableEvent.compute_id(event_type, category, payload, timestamp)
        
        causal = CausalMetadata.create_root(event_id)
        if causation_id:
            causal = causal.with_causation(causation_id)
        
        if correlation_id:
            causal = CausalMetadata(
                event_id=causal.event_id,
                causation_id=causal.causation_id,
                correlation_id=correlation_id,
                trace_id=causal.trace_id,
                parent_event_id=causal.parent_event_id,
                lineage_depth=causal.lineage_depth
            )
        
        return ImmutableEvent(
            event_type=event_type,
            category=category,
            timestamp=timestamp,
            causal=causal,
            schema_version=1,
            payload=frozenset(sorted(payload.items()))
        )
    
    def create_cognitive_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        causation_id: Optional[str] = None
    ) -> ImmutableEvent:
        """Create cognitive category event"""
        return self.create_event(
            event_type=event_type,
            category=EventCategory.COGNITIVE.value,
            payload=payload,
            causation_id=causation_id
        )
    
    def create_execution_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        causation_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> ImmutableEvent:
        """Create execution category event"""
        return self.create_event(
            event_type=event_type,
            category=EventCategory.EXECUTION.value,
            payload=payload,
            causation_id=causation_id,
            correlation_id=correlation_id
        )
    
    def create_identity_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        causation_id: Optional[str] = None
    ) -> ImmutableEvent:
        """Create identity category event"""
        return self.create_event(
            event_type=event_type,
            category=EventCategory.IDENTITY.value,
            payload=payload,
            causation_id=causation_id
        )


# Predefined event types

class EventTypes:
    """Standard event types"""
    # Cognitive
    BELIEF_CREATED = "belief_created"
    BELIEF_UPDATED = "belief_updated"
    BELIEF_REVISED = "belief_revised"
    CONTRADICTION_DETECTED = "contradiction_detected"
    PRESSURE_ACCUMULATED = "pressure_accumulated"
    
    # Execution
    GOAL_CREATED = "goal_created"
    GOAL_EXECUTED = "goal_executed"
    GOAL_COMPLETED = "goal_completed"
    GOAL_FAILED = "goal_failed"
    ACTION_PERFORMED = "action_performed"
    RESULT_RECEIVED = "result_received"
    
    # Identity
    IDENTITY_MUTATED = "identity_mutated"
    GENOME_EVOLVED = "genome_evolved"
    PRESSURE_CHANGED = "pressure_changed"
    
    # Meta
    REFLECTION_TRIGGERED = "reflection_triggered"
    STRATEGY_SELECTED = "strategy_selected"
    PLANNER_INVOKED = "planner_invoked"


class StreamIds:
    """Standard stream IDs"""
    COGNITION = "cognition-stream"
    EXECUTION = "execution-stream"
    IDENTITY = "identity-stream"
    MEMORY = "memory-stream"
    META = "meta-stream"