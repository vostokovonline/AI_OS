"""
Pure Domain Events - Immutable facts with no side effects.

Stage 1: Domain Core only
- No infrastructure imports
- No asyncio
- No websockets
- No persistence
- Pure data structures only

Event contract:
- event_id: deterministic hash from content
- stream_id: logical partition
- position: monotonic in stream
- schema_version: for replay compatibility
- timestamp: wall clock (not used in determinism)
- causation_id: chain tracking
"""
from typing import Dict, Any, Optional, FrozenSet, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json


@dataclass(frozen=True)
class DomainEvent:
    """
    Immutable event - pure fact.
    
    Key invariants:
    - event_id computed from content (deterministic)
    - stream_id partitions logically
    - position is monotonic per stream
    - schema_version enables replay across evolutions
    - timestamp is wall clock (not used in determinism)
    """
    event_type: str
    stream_id: str
    position: int
    schema_version: int
    event_id: str
    timestamp: str
    causation_id: str
    correlation_id: str
    payload: Tuple[Tuple[str, Any], ...]
    
    def __post_init__(self):
        if not self.event_id:
            object.__setattr__(self, 'event_id', self._compute_id())
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.utcnow().isoformat())
    
    def _compute_id(self) -> str:
        """Deterministic ID from content (deterministic across replays)"""
        content = {
            "stream_id": self.stream_id,
            "position": self.position,
            "event_type": self.event_type,
            "payload": dict(self.payload)
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:32]
    
    def with_position(self, position: int) -> 'DomainEvent':
        """Create copy with new position"""
        return DomainEvent(
            event_type=self.event_type,
            stream_id=self.stream_id,
            position=position,
            schema_version=self.schema_version,
            event_id=self.event_id,
            timestamp=self.timestamp,
            causation_id=self.causation_id,
            correlation_id=self.correlation_id,
            payload=self.payload
        )
    
    def with_causation(self, causation_id: str) -> 'DomainEvent':
        """Create copy with new causation"""
        return DomainEvent(
            event_type=self.event_type,
            stream_id=self.stream_id,
            position=self.position,
            schema_version=self.schema_version,
            event_id=self.event_id,
            timestamp=self.timestamp,
            causation_id=causation_id,
            correlation_id=self.correlation_id,
            payload=self.payload
        )
    
    def to_tuple(self) -> tuple:
        """Convert to hashable tuple for state snapshots"""
        return (
            self.event_id,
            self.stream_id,
            self.position,
            self.schema_version,
            self.event_type,
            self.timestamp,
            self.causation_id,
            self.correlation_id,
            self.payload
        )
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DomainEvent':
        """Deserialize from dict"""
        return DomainEvent(
            event_type=data["event_type"],
            stream_id=data["stream_id"],
            position=data["position"],
            schema_version=data.get("schema_version", 1),
            event_id=data.get("event_id", ""),
            timestamp=data.get("timestamp", ""),
            causation_id=data.get("causation_id", ""),
            correlation_id=data.get("correlation_id", ""),
            payload=tuple(sorted(data.get("payload", {}).items()))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "event_type": self.event_type,
            "stream_id": self.stream_id,
            "position": self.position,
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "payload": dict(self.payload)
        }


@dataclass(frozen=True)
class BeliefAdded(DomainEvent):
    """Belief added to state"""
    def __init__(self, stream_id: str, position: int, payload: Dict[str, Any], causation_id: str = "", correlation_id: str = ""):
        super().__init__(
            event_type="belief_added",
            stream_id=stream_id,
            position=position,
            schema_version=1,
            event_id="",
            timestamp="",
            causation_id=causation_id,
            correlation_id=correlation_id,
            payload=tuple(sorted(payload.items()))
        )


@dataclass(frozen=True)
class BeliefUpdated(DomainEvent):
    """Belief updated"""
    def __init__(self, stream_id: str, position: int, payload: Dict[str, Any], causation_id: str = "", correlation_id: str = ""):
        super().__init__(
            event_type="belief_updated",
            stream_id=stream_id,
            position=position,
            schema_version=1,
            event_id="",
            timestamp="",
            causation_id=causation_id,
            correlation_id=correlation_id,
            payload=tuple(sorted(payload.items()))
        )


@dataclass(frozen=True)
class ContradictionRegistered(DomainEvent):
    """Contradiction detected"""
    def __init__(self, stream_id: str, position: int, payload: Dict[str, Any], causation_id: str = "", correlation_id: str = ""):
        super().__init__(
            event_type="contradiction_registered",
            stream_id=stream_id,
            position=position,
            schema_version=1,
            event_id="",
            timestamp="",
            causation_id=causation_id,
            correlation_id=correlation_id,
            payload=tuple(sorted(payload.items()))
        )


@dataclass(frozen=True)
class GenomeEvolved(DomainEvent):
    """Genome evolution event"""
    def __init__(self, stream_id: str, position: int, payload: Dict[str, Any], causation_id: str = "", correlation_id: str = ""):
        super().__init__(
            event_type="genome_evolved",
            stream_id=stream_id,
            position=position,
            schema_version=1,
            event_id="",
            timestamp="",
            causation_id=causation_id,
            correlation_id=correlation_id,
            payload=tuple(sorted(payload.items()))
        )


@dataclass(frozen=True)
class IdentityMutated(DomainEvent):
    """Identity mutation event"""
    def __init__(self, stream_id: str, position: int, payload: Dict[str, Any], causation_id: str = "", correlation_id: str = ""):
        super().__init__(
            event_type="identity_mutated",
            stream_id=stream_id,
            position=position,
            schema_version=1,
            event_id="",
            timestamp="",
            causation_id=causation_id,
            correlation_id=correlation_id,
            payload=tuple(sorted(payload.items()))
        )


EVENT_TYPE_MAP = {
    "belief_added": BeliefAdded,
    "belief_updated": BeliefUpdated,
    "contradiction_registered": ContradictionRegistered,
    "genome_evolved": GenomeEvolved,
    "identity_mutated": IdentityMutated,
}


def create_event(
    event_type: str,
    stream_id: str,
    position: int,
    payload: Dict[str, Any],
    causation_id: str = "",
    correlation_id: str = ""
) -> DomainEvent:
    """Factory for creating events"""
    event_cls = EVENT_TYPE_MAP.get(event_type, DomainEvent)
    return event_cls(
        stream_id=stream_id,
        position=position,
        payload=payload,
        causation_id=causation_id,
        correlation_id=correlation_id
    )