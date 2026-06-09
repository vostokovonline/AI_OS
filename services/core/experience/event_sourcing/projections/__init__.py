"""
Projections - Read models for different views.

Projections are derived state from events.
They do NOT mutate the canonical state.
They are subscribers to the event stream.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import threading

from cognitive_state import CognitiveState
from events import CognitiveEvent


class Projection:
    """Base projection class"""
    
    def __init__(self, name: str):
        self.name = name
        self._lock = threading.RLock()
        self._handlers: Dict[str, Callable] = {}
        self._state: Optional[CognitiveState] = None
        self._version: int = 0
        self._last_position: int = 0
    
    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler"""
        self._handlers[event_type] = handler
    
    def project(self, event: Dict[str, Any]):
        """Project single event"""
        event_type = event.get("event_type", "")
        
        with self._lock:
            if event_type in self._handlers:
                self._handlers[event_type](event)
            
            self._last_position = event.get("position", self._last_position)
    
    def project_sequence(self, events: List[Dict[str, Any]]):
        """Project sequence of events"""
        for event in events:
            self.project(event)
    
    def get_state(self) -> CognitiveState:
        """Get current projection state"""
        with self._lock:
            return self._state
    
    def get_version(self) -> int:
        """Get projection version"""
        with self._lock:
            return self._version


@dataclass
class BeliefProjection(Projection):
    """Projection for belief queries"""
    
    def __init__(self):
        super().__init__("belief_projection")
        self.register_handler("belief_added", self._handle_belief_added)
        self.register_handler("belief_updated", self._handle_belief_updated)
        self.register_handler("belief_removed", self._handle_belief_removed)
        self._beliefs: Dict[str, Any] = {}
        self._by_source: Dict[str, List[str]] = {}
        self._by_attractor: Dict[str, List[str]] = {}
    
    def _handle_belief_added(self, event: Dict[str, Any]):
        payload = event.get("payload", {})
        belief_id = event.get("target_id", payload.get("belief_id", ""))
        
        self._beliefs[belief_id] = {
            "id": belief_id,
            "proposition": payload.get("proposition", ""),
            "confidence": payload.get("confidence", 0.5),
            "entropy": payload.get("entropy", 0.0),
            "source": payload.get("source", "unknown"),
            "attractor_state": payload.get("attractor_state"),
            "created_at": payload.get("created_at", ""),
            "version": self._version
        }
        
        source = payload.get("source", "unknown")
        if source not in self._by_source:
            self._by_source[source] = []
        if belief_id not in self._by_source[source]:
            self._by_source[source].append(belief_id)
        
        self._version += 1
    
    def _handle_belief_updated(self, event: Dict[str, Any]):
        payload = event.get("payload", {})
        belief_id = event.get("target_id", payload.get("belief_id", ""))
        
        if belief_id in self._beliefs:
            self._beliefs[belief_id].update({
                "confidence": payload.get("confidence", self._beliefs[belief_id]["confidence"]),
                "entropy": payload.get("entropy", self._beliefs[belief_id]["entropy"]),
                "attractor_state": payload.get("attractor_state", self._beliefs[belief_id]["attractor_state"]),
            })
            self._version += 1
    
    def _handle_belief_removed(self, event: Dict[str, Any]):
        belief_id = event.get("target_id", event.get("payload", {}).get("belief_id", ""))
        
        if belief_id in self._beliefs:
            source = self._beliefs[belief_id]["source"]
            if source in self._by_source and belief_id in self._by_source[source]:
                self._by_source[source].remove(belief_id)
            
            del self._beliefs[belief_id]
            self._version += 1
    
    def get_all_beliefs(self) -> Dict[str, Any]:
        """Get all beliefs"""
        with self._lock:
            return dict(self._beliefs)
    
    def get_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get beliefs by source"""
        with self._lock:
            belief_ids = self._by_source.get(source, [])
            return [self._beliefs[bid] for bid in belief_ids if bid in self._beliefs]
    
    def get_by_attractor(self, attractor: str) -> List[Dict[str, Any]]:
        """Get beliefs by attractor state"""
        with self._lock:
            belief_ids = self._by_attractor.get(attractor, [])
            return [self._beliefs[bid] for bid in belief_ids if bid in self._beliefs]
    
    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get distribution of belief confidence"""
        with self._lock:
            if not self._beliefs:
                return {"high": 0, "medium": 0, "low": 0}
            
            high = sum(1 for b in self._beliefs.values() if b["confidence"] >= 0.7)
            medium = sum(1 for b in self._beliefs.values() if 0.4 <= b["confidence"] < 0.7)
            low = sum(1 for b in self._beliefs.values() if b["confidence"] < 0.4)
            total = len(self._beliefs)
            
            return {
                "high": high / total if total > 0 else 0,
                "medium": medium / total if total > 0 else 0,
                "low": low / total if total > 0 else 0
            }


@dataclass
class ContradictionProjection(Projection):
    """Projection for contradiction queries"""
    
    def __init__(self):
        super().__init__("contradiction_projection")
        self.register_handler("contradiction_registered", self._handle_registered)
        self.register_handler("contradiction_resolved", self._handle_resolved)
        self._contradictions: Dict[str, Any] = {}
        self._by_type: Dict[str, List[str]] = {}
        self._by_status: Dict[str, List[str]] = {"unresolved": [], "resolved": []}
    
    def _handle_registered(self, event: Dict[str, Any]):
        payload = event.get("payload", {})
        episode_id = event.get("target_id", payload.get("episode_id", ""))
        
        self._contradictions[episode_id] = {
            "id": episode_id,
            "belief_ids": payload.get("belief_ids", []),
            "contradiction_type": payload.get("contradiction_type", "unknown"),
            "severity": payload.get("severity", "medium"),
            "recurrence_count": payload.get("recurrence_count", 1),
            "status": "unresolved",
            "created_at": payload.get("first_seen", "")
        }
        
        ctype = payload.get("contradiction_type", "unknown")
        if ctype not in self._by_type:
            self._by_type[ctype] = []
        self._by_type[ctype].append(episode_id)
        
        self._by_status["unresolved"].append(episode_id)
        self._version += 1
    
    def _handle_resolved(self, event: Dict[str, Any]):
        episode_id = event.get("target_id", event.get("payload", {}).get("episode_id", ""))
        
        if episode_id in self._contradictions:
            self._contradictions[episode_id]["status"] = "resolved"
            
            if episode_id in self._by_status["unresolved"]:
                self._by_status["unresolved"].remove(episode_id)
            self._by_status["resolved"].append(episode_id)
            
            self._version += 1
    
    def get_all_contradictions(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._contradictions)
    
    def get_unresolved(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._contradictions[eid] for eid in self._by_status["unresolved"] 
                    if eid in self._contradictions]
    
    def get_by_type(self, ctype: str) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._contradictions[eid] for eid in self._by_type.get(ctype, [])
                    if eid in self._contradictions]
    
    def get_severity_distribution(self) -> Dict[str, int]:
        with self._lock:
            counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            for c in self._contradictions.values():
                sev = c.get("severity", "medium")
                if sev in counts:
                    counts[sev] += 1
            return counts


@dataclass
class TransactionProjection(Projection):
    """Projection for transaction history"""
    
    def __init__(self):
        super().__init__("transaction_projection")
        self.register_handler("transaction_committed", self._handle_committed)
        self.register_handler("transaction_compensated", self._handle_compensated)
        self._transactions: Dict[str, Any] = {}
    
    def _handle_committed(self, event: Dict[str, Any]):
        payload = event.get("payload", {})
        tx_id = event.get("target_id", payload.get("transaction_id", ""))
        
        self._transactions[tx_id] = {
            "id": tx_id,
            "status": "committed",
            "created_at": payload.get("created_at", ""),
            "position": event.get("position", 0)
        }
        self._version += 1
    
    def _handle_compensated(self, event: Dict[str, Any]):
        payload = event.get("payload", {})
        tx_id = payload.get("original_transaction_id", event.get("target_id", ""))
        
        if tx_id in self._transactions:
            self._transactions[tx_id].update({
                "status": "compensated",
                "compensated_at": datetime.utcnow().isoformat(),
                "reason": payload.get("reason", "")
            })
            self._version += 1
    
    def get_all_transactions(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._transactions)
    
    def get_committed_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._transactions.values() if t["status"] == "committed")
    
    def get_compensated_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._transactions.values() if t["status"] == "compensated")


class ProjectionManager:
    """
    Manages multiple projections.
    
    Subscribes to event store and fans out to all projections.
    """
    
    def __init__(self):
        self._projections: Dict[str, Projection] = {}
        self._lock = threading.RLock()
    
    def register(self, projection: Projection):
        """Register projection"""
        with self._lock:
            self._projections[projection.name] = projection
    
    def project(self, event: Dict[str, Any]):
        """Fan out event to all projections"""
        with self._lock:
            for projection in self._projections.values():
                projection.project(event)
    
    def project_sequence(self, events: List[Dict[str, Any]]):
        """Fan out event sequence"""
        for event in events:
            self.project(event)
    
    def get_projection(self, name: str) -> Optional[Projection]:
        """Get projection by name"""
        with self._lock:
            return self._projections.get(name)
    
    def get_all_states(self) -> Dict[str, Any]:
        """Get all projection states"""
        with self._lock:
            return {name: proj.get_state() for name, proj in self._projections.items()}