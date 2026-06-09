"""
Domain Snapshot - Immutable state snapshots for replay optimization.

Stage 1: Domain Core only
- Snapshots are immutable
- Store provides snapshots
- Replay can start from snapshot + events
- Snapshots don't mutate
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class DomainSnapshot:
    """
    Immutable state snapshot.
    
    Captures state at a point in time.
    Can be used to start replay from this point.
    """
    stream_id: str
    version: int
    position: int
    state_hash: str
    timestamp: str
    beliefs_count: int
    contradictions_count: int
    identity_state: Tuple[float, float, float, float]  # (autonomy, curiosity, stability, coherence)
    genome_axes: Tuple[Tuple[str, float], ...]
    
    def to_tuple(self) -> tuple:
        """Convert to hashable tuple"""
        return (
            self.stream_id,
            self.version,
            self.position,
            self.state_hash,
            self.timestamp,
            self.beliefs_count,
            self.contradictions_count,
            self.identity_state,
            self.genome_axes
        )
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'DomainSnapshot':
        """Deserialize from dict"""
        return DomainSnapshot(
            stream_id=data["stream_id"],
            version=data["version"],
            position=data["position"],
            state_hash=data["state_hash"],
            timestamp=data["timestamp"],
            beliefs_count=data["beliefs_count"],
            contradictions_count=data["contradictions_count"],
            identity_state=tuple(data["identity_state"]),
            genome_axes=tuple(data["genome_axes"])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict"""
        return {
            "stream_id": self.stream_id,
            "version": self.version,
            "position": self.position,
            "state_hash": self.state_hash,
            "timestamp": self.timestamp,
            "beliefs_count": self.beliefs_count,
            "contradictions_count": self.contradictions_count,
            "identity_state": list(self.identity_state),
            "genome_axes": [(k, v) for k, v in self.genome_axes]
        }


def create_snapshot(
    stream_id: str,
    state,  # DomainState
    position: int
) -> DomainSnapshot:
    """
    Create snapshot from domain state.
    
    Pure function: state → snapshot
    """
    from .state import DomainState
    
    return DomainSnapshot(
        stream_id=stream_id,
        version=state.version,
        position=position,
        state_hash=DomainState.compute_hash(state),
        timestamp=state.entropy,  # Use entropy field as timestamp for now
        beliefs_count=len(state.beliefs),
        contradictions_count=len(state.contradictions),
        identity_state=(
            state.identity.autonomy,
            state.identity.curiosity,
            state.identity.stability,
            state.identity.coherence
        ),
        genome_axes=tuple(state.genome.axes.items())
    )