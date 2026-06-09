"""
Event Schemas with Schema Versioning.

All events are immutable facts with version tracking.
Schema version enables event replay across schema evolution.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import hashlib
import json


class SchemaVersion(Enum):
    """Event schema versions"""
    V1 = 1


@dataclass(frozen=True)
class CognitiveEvent:
    """
    Immutable event base.
    
    Key invariants:
    - event_id: globally unique
    - stream_id: logical partition (identity, genome, lineage)
    - version: monotonic within stream
    - schema_version: for replay compatibility
    - timestamp: wall clock (not used in replay determinism)
    """
    event_type: str
    stream_id: str
    position: int
    schema_version: int = SchemaVersion.V1.value
    event_id: str = ""
    timestamp: str = ""
    causation_id: str = ""
    correlation_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.event_id:
            object.__setattr__(self, 'event_id', self._compute_id())
        if not self.timestamp:
            object.__setattr__(self, 'timestamp', datetime.utcnow().isoformat())
    
    def _compute_id(self) -> str:
        """Deterministic event ID from content"""
        content = {
            "stream_id": self.stream_id,
            "position": self.position,
            "event_type": self.event_type,
            "payload": self.payload
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for storage"""
        return {
            "event_id": self.event_id,
            "stream_id": self.stream_id,
            "position": self.position,
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "causation_id": self.causation_id,
            "correlation_id": self.correlation_id,
            "payload": self.payload
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'CognitiveEvent':
        """Deserialize from dict"""
        return CognitiveEvent(
            event_id=data["event_id"],
            stream_id=data["stream_id"],
            position=data["position"],
            schema_version=data.get("schema_version", SchemaVersion.V1.value),
            event_type=data["event_type"],
            timestamp=data.get("timestamp", ""),
            causation_id=data.get("causation_id", ""),
            correlation_id=data.get("correlation_id", ""),
            payload=data.get("payload", {})
        )


@dataclass(frozen=True)
class BeliefEvent(CognitiveEvent):
    """Belief-related events"""
    pass


@dataclass(frozen=True)
class CausalEvent(CognitiveEvent):
    """Causal edge events"""
    pass


@dataclass(frozen=True)
class ContradictionEvent(CognitiveEvent):
    """Contradiction events"""
    pass


@dataclass(frozen=True)
class TransactionEvent(CognitiveEvent):
    """Transaction events"""
    pass


@dataclass(frozen=True)
class IdentityEvent(CognitiveEvent):
    """Identity-level events"""
    pass


# Event type constants
class EventTypes:
    BELIEF_ADDED = "belief_added"
    BELIEF_UPDATED = "belief_updated"
    BELIEF_REMOVED = "belief_removed"
    CAUSAL_EDGE_ADDED = "causal_edge_added"
    CONTRADICTION_REGISTERED = "contradiction_registered"
    CONTRADICTION_RESOLVED = "contradiction_resolved"
    TRANSACTION_COMMITTED = "transaction_committed"
    TRANSACTION_COMPENSATED = "transaction_compensated"
    IDENTITY_MUTATED = "identity_mutated"
    GENOME_EVOLVED = "genome_evolved"
    LINEAGE_RECORDED = "lineage_recorded"


# Stream ID constants
class StreamIds:
    BELIEF = "belief-stream"
    CAUSAL = "causal-stream"
    CONTRADICTION = "contradiction-stream"
    TRANSACTION = "transaction-stream"
    IDENTITY = "identity-stream"
    GENOME = "genome-stream"
    LINEAGE = "lineage-stream"