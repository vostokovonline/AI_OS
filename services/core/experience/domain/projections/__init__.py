"""
Domain Projections - Read models derived from events.

Stage 1: Domain Core only
- Projections are derived state
- They do NOT affect canonical state
- They are pure transformations from events
- Multiple projections can exist simultaneously
- Projections are eventually consistent (from event stream)
"""
from typing import Dict, Any, Optional, Tuple, List, Callable
from types import MappingProxyType
from dataclasses import dataclass, field
import threading


@dataclass(frozen=True)
class BeliefProjectionState:
    """Projection state for beliefs"""
    beliefs: Tuple[Tuple[str, Dict[str, Any]], ...]  # (id, belief_data)
    by_source: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (source, belief_ids)
    by_attractor: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (attractor, belief_ids)
    confidence_distribution: Tuple[Tuple[str, float], ...]  # (bucket, ratio)
    version: int


class BeliefProjection:
    """
    Belief read model.
    
    Derived from belief events.
    Does NOT affect canonical state.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._beliefs: Dict[str, Dict[str, Any]] = {}
        self._by_source: Dict[str, List[str]] = {}
        self._by_attractor: Dict[str, List[str]] = {}
        self._version: int = 0
        self._last_position: int = 0
        self._handlers: Dict[str, Callable] = {
            "belief_added": self._handle_added,
            "belief_updated": self._handle_updated,
            "belief_removed": self._handle_removed,
        }
    
    def project(self, event) -> None:
        """
        Project single event.
        
        Updates projection state from event.
        Does NOT emit events or write to store.
        Thread-safe.
        """
        event_type = event.event_type
        
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type](event)
            self._last_position = event.position
    
    def _handle_added(self, event) -> None:
        """Handle belief_added event"""
        payload = dict(event.payload)
        belief_id = payload.get("belief_id", event.event_id)
        
        self._beliefs[belief_id] = {
            "proposition": payload.get("proposition", ""),
            "confidence": payload.get("confidence", 0.5),
            "entropy": payload.get("entropy", 0.0),
            "source": payload.get("source", "unknown"),
            "attractor_state": payload.get("attractor_state"),
            "created_at": payload.get("created_at", event.timestamp)
        }
        
        source = payload.get("source", "unknown")
        if source not in self._by_source:
            self._by_source[source] = []
        if belief_id not in self._by_source[source]:
            self._by_source[source].append(belief_id)
        
        attractor = payload.get("attractor_state")
        if attractor:
            if attractor not in self._by_attractor:
                self._by_attractor[attractor] = []
            if belief_id not in self._by_attractor[attractor]:
                self._by_attractor[attractor].append(belief_id)
        
        self._version += 1
    
    def _handle_updated(self, event) -> None:
        """Handle belief_updated event"""
        payload = dict(event.payload)
        belief_id = payload.get("belief_id", event.event_id)
        
        if belief_id in self._beliefs:
            self._beliefs[belief_id].update({
                "confidence": payload.get("confidence", self._beliefs[belief_id]["confidence"]),
                "entropy": payload.get("entropy", self._beliefs[belief_id]["entropy"]),
                "attractor_state": payload.get("attractor_state", self._beliefs[belief_id]["attractor_state"]),
            })
            self._version += 1
    
    def _handle_removed(self, event) -> None:
        """Handle belief_removed event"""
        payload = dict(event.payload)
        belief_id = payload.get("belief_id", event.event_id)
        
        if belief_id in self._beliefs:
            source = self._beliefs[belief_id]["source"]
            if source in self._by_source and belief_id in self._by_source[source]:
                self._by_source[source].remove(belief_id)
            
            del self._beliefs[belief_id]
            self._version += 1
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Get all beliefs"""
        with self._lock:
            return dict(self._beliefs)
    
    def get_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get beliefs by source"""
        with self._lock:
            ids = self._by_source.get(source, [])
            return [self._beliefs[bid] for bid in ids if bid in self._beliefs]
    
    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get confidence bucket distribution"""
        with self._lock:
            if not self._beliefs:
                return {"high": 0.0, "medium": 0.0, "low": 0.0}
            
            total = len(self._beliefs)
            high = sum(1 for b in self._beliefs.values() if b["confidence"] >= 0.7)
            medium = sum(1 for b in self._beliefs.values() if 0.4 <= b["confidence"] < 0.7)
            low = sum(1 for b in self._beliefs.values() if b["confidence"] < 0.4)
            
            return {
                "high": high / total,
                "medium": medium / total,
                "low": low / total
            }
    
    def get_state(self) -> BeliefProjectionState:
        """Get immutable projection state"""
        with self._lock:
            return BeliefProjectionState(
                beliefs=tuple(sorted(self._beliefs.items())),
                by_source=tuple((s, tuple(ids)) for s, ids in self._by_source.items()),
                by_attractor=tuple((a, tuple(ids)) for a, ids in self._by_attractor.items()),
                confidence_distribution=tuple(self.get_confidence_distribution().items()),
                version=self._version
            )


@dataclass(frozen=True)
class ContradictionProjectionState:
    """Projection state for contradictions"""
    contradictions: Tuple[Tuple[str, Dict[str, Any]], ...]
    unresolved_count: int
    by_type: Tuple[Tuple[str, int], ...]
    severity_distribution: Tuple[Tuple[str, int], ...]


class ContradictionProjection:
    """Contradiction read model."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._contradictions: Dict[str, Dict[str, Any]] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._by_status: Dict[str, List[str]] = {"unresolved": [], "resolved": []}
        self._version: int = 0
    
    def project(self, event) -> None:
        """Project event to contradiction state"""
        event_type = event.event_type
        
        with self._lock:
            if event_type == "contradiction_registered":
                self._handle_registered(event)
            elif event_type == "contradiction_resolved":
                self._handle_resolved(event)
    
    def _handle_registered(self, event) -> None:
        payload = dict(event.payload)
        episode_id = payload.get("episode_id", event.event_id)
        
        self._contradictions[episode_id] = {
            "belief_ids": payload.get("belief_ids", []),
            "type": payload.get("contradiction_type", "unknown"),
            "severity": payload.get("severity", "medium"),
            "status": "unresolved",
            "recurrence_count": payload.get("recurrence_count", 1)
        }
        
        ctype = payload.get("contradiction_type", "unknown")
        if ctype not in self._by_type:
            self._by_type[ctype] = []
        self._by_type[ctype].append(episode_id)
        
        self._by_status["unresolved"].append(episode_id)
        self._version += 1
    
    def _handle_resolved(self, event) -> None:
        payload = dict(event.payload)
        episode_id = payload.get("episode_id", event.event_id)
        
        if episode_id in self._contradictions:
            self._contradictions[episode_id]["status"] = "resolved"
            if episode_id in self._by_status["unresolved"]:
                self._by_status["unresolved"].remove(episode_id)
            self._by_status["resolved"].append(episode_id)
            self._version += 1
    
    def get_unresolved(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._contradictions[eid] for eid in self._by_status["unresolved"] 
                    if eid in self._contradictions]
    
    def get_severity_counts(self) -> Dict[str, int]:
        with self._lock:
            counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for c in self._contradictions.values():
                sev = c.get("severity", "medium")
                if sev in counts:
                    counts[sev] += 1
            return counts


@dataclass(frozen=True)
class IdentityProjectionState:
    """Projection state for identity"""
    axes: Tuple[Tuple[str, float], ...]
    trajectory_hash: str
    mutation_count: int


class IdentityProjection:
    """Identity read model."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._axes: Dict[str, float] = {
            "autonomy": 0.5,
            "curiosity": 0.5,
            "stability": 0.5,
            "coherence": 0.5
        }
        self._mutations: List[Tuple[str, float]] = []
        self._version: int = 0
    
    def project(self, event) -> None:
        """Project identity event"""
        with self._lock:
            if event.event_type == "identity_mutated":
                payload = dict(event.payload)
                axis = payload.get("axis", "autonomy")
                delta = payload.get("delta", 0.0)
                
                new_value = max(0.0, min(1.0, self._axes.get(axis, 0.5) + delta))
                self._axes[axis] = new_value
                self._mutations.append((axis, delta))
                self._version += 1
    
    def get_axes(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._axes)
    
    def get_mutation_count(self) -> int:
        with self._lock:
            return len(self._mutations)


class ProjectionManager:
    """
    Manages multiple projections.
    
    Coordinates projection updates from events.
    Projections are subscribers, not sources of truth.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self._projections: Dict[str, object] = {}
        self._register_default()
    
    def _register_default(self):
        """Register default projections"""
        self._projections["belief"] = BeliefProjection()
        self._projections["contradiction"] = ContradictionProjection()
        self._projections["identity"] = IdentityProjection()
    
    def project(self, event) -> None:
        """Fan out event to all projections"""
        with self._lock:
            for projection in self._projections.values():
                try:
                    projection.project(event)
                except Exception:
                    pass
    
    def project_sequence(self, events: Tuple) -> None:
        """Project event sequence"""
        for event in events:
            self.project(event)
    
    def get_projection(self, name: str) -> Optional[object]:
        with self._lock:
            return self._projections.get(name)