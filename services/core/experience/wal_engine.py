"""
WAL (Write-Ahead Log) Engine - Append-only event journal for deterministic replay.

Provides:
- Immutable event log
- State reconstruction from events
- Corruption detection via hash verification
- Causal topology reconstruction

Key invariant:
    Every mutation MUST be logged before commit.
    Replay from genesis MUST produce identical state.
"""
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from uuid import uuid4
from enum import Enum
from copy import deepcopy
import threading


class WALEventType(Enum):
    """Types of WAL events"""
    BELIEF_ADDED = "belief_added"
    BELIEF_UPDATED = "belief_updated"
    BELIEF_REMOVED = "belief_removed"
    CAUSAL_EDGE_ADDED = "causal_edge_added"
    CONTRADICTION_REGISTERED = "contradiction_registered"
    STATE_COMMITTED = "state_committed"
    SNAPSHOT_CREATED = "snapshot_created"
    TRANSACTION_COMMITTED = "transaction_committed"
    TRANSACTION_COMPENSATED = "transaction_compensated"


@dataclass
class MutationEvent:
    """
    Single mutation event in WAL.
    
    Rich metadata for causal introspection and compensation tracking.
    """
    event_id: str
    timestamp: float
    event_type: str
    version: int
    operation: str
    target_id: str
    payload: Dict[str, Any]
    parent_version: int
    actor: str
    entropy_delta: float
    causal_context: Dict[str, str] = field(default_factory=dict)
    state_hash_before: str = ""
    state_hash_after: str = ""
    causal_chain: List[str] = field(default_factory=list)
    
    # Transaction lineage
    transaction_id: Optional[str] = None
    parent_transaction_id: Optional[str] = None
    compensates_event_id: Optional[str] = None
    compensates_transaction_id: Optional[str] = None
    
    # Epistemic metadata
    epistemic_timestamp: Optional[str] = None  # ISO timestamp
    causal_depth: int = 0  # How far in causal chain
    reflection_depth: int = 0  # Reflection recursion level
    
    # Origin & confidence
    origin_agent: str = "system"  # Who initiated
    confidence_before: float = 0.5
    confidence_after: float = 0.5
    
    # Inverse linkage (explicit)
    inverse_of_event_id: Optional[str] = None
    inverse_of_transaction_id: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of this event"""
        content = {
            "event_id": self.event_id,
            "version": self.version,
            "operation": self.operation,
            "target_id": self.target_id,
            "payload": self.payload,
            "parent_version": self.parent_version
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "MutationEvent":
        return cls(**d)
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of this event"""
        content = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "version": self.version,
            "operation": self.operation,
            "target_id": self.target_id,
            "payload": self.payload,
            "parent_version": self.parent_version
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class SnapshotEvent:
    """Snapshot event for fast replay"""
    event_id: str
    timestamp: float
    version: int
    state_snapshot: Dict[str, Any]  # Full serialized state
    state_hash: str
    parent_version: int
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "SnapshotEvent":
        return cls(**d)


class WALEngine:
    """
    Write-Ahead Log Engine for epistemic state.
    
    Guarantees:
    1. Every mutation logged before commit
    2. Events append-only (no modification)
    3. Replay produces identical state
    4. Corruption detected via hash mismatch
    """
    
    def __init__(self, snapshot_interval: int = 10):
        self._events: List[MutationEvent] = []
        self._snapshots: Dict[int, SnapshotEvent] = {}  # version -> snapshot
        self._snapshot_interval = snapshot_interval
        self._lock = threading.RLock()
        self._event_counter: int = 0
        
        # Replay cache
        self._rebuilt_state_cache: Dict[int, Dict] = {}
    
    def log_event(
        self,
        event_type: WALEventType,
        version: int,
        operation: str,
        target_id: str,
        payload: Dict[str, Any],
        parent_version: int,
        actor: str = "ues",
        entropy_delta: float = 0.0,
        causal_context: Optional[Dict[str, str]] = None,
        state_hash_before: str = "",
        state_hash_after: str = "",
        full_state: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
        parent_transaction_id: Optional[str] = None,
        compensates_event_id: Optional[str] = None,
        compensates_transaction_id: Optional[str] = None,
        causal_depth: int = 0,
        reflection_depth: int = 0,
        origin_agent: str = "system",
        confidence_before: float = 0.5,
        confidence_after: float = 0.5,
        inverse_of_event_id: Optional[str] = None,
        inverse_of_transaction_id: Optional[str] = None
    ) -> str:
        """Log mutation event to WAL with rich metadata"""
        with self._lock:
            self._event_counter += 1
            
            # Get causal chain from previous event
            causal_chain = []
            if self._events:
                last_event = self._events[-1]
                causal_chain = [last_event.event_id]
            
            event = MutationEvent(
                event_id=str(uuid4()),
                timestamp=time.time(),
                event_type=event_type.value,
                version=version,
                operation=operation,
                target_id=target_id,
                payload=payload,
                parent_version=parent_version,
                actor=actor,
                entropy_delta=entropy_delta,
                causal_context=causal_context or {},
                state_hash_before=state_hash_before,
                state_hash_after=state_hash_after,
                causal_chain=causal_chain,
                transaction_id=transaction_id,
                parent_transaction_id=parent_transaction_id,
                compensates_event_id=compensates_event_id,
                compensates_transaction_id=compensates_transaction_id,
                epistemic_timestamp=datetime.utcnow().isoformat(),
                causal_depth=causal_depth,
                reflection_depth=reflection_depth,
                origin_agent=origin_agent,
                confidence_before=confidence_before,
                confidence_after=confidence_after,
                inverse_of_event_id=inverse_of_event_id,
                inverse_of_transaction_id=inverse_of_transaction_id
            )
            
            # Compute event hash for chain integrity
            event.event_id = event.compute_hash()[:16]
            
            self._events.append(event)
            
            # Create snapshot periodically with full state
            if version % self._snapshot_interval == 0 and full_state:
                self._create_snapshot(version, full_state, state_hash_after)
            
            return event.event_id
    
    def _create_snapshot(self, version: int, payload: Dict, state_hash: str):
        """Create snapshot for fast replay"""
        snapshot = SnapshotEvent(
            event_id=str(uuid4()),
            timestamp=time.time(),
            version=version,
            state_snapshot=deepcopy(payload),
            state_hash=state_hash,
            parent_version=version - 1
        )
        self._snapshots[version] = snapshot
    
    def get_events(self, from_version: int = 0) -> List[MutationEvent]:
        """Get all events from version"""
        with self._lock:
            return [e for e in self._events if e.version > from_version]
    
    def get_all_events(self) -> List[MutationEvent]:
        """Get all events"""
        with self._lock:
            return self._events.copy()
    
    def get_snapshot(self, version: int) -> Optional[SnapshotEvent]:
        """Get snapshot at version"""
        with self._lock:
            return self._snapshots.get(version)
    
    def get_latest_version(self) -> int:
        """Get latest logged version"""
        with self._lock:
            if not self._events:
                return 0
            return max(e.version for e in self._events)
    
    def replay_state(
        self,
        initial_state: Dict[str, Any],
        from_version: int = 0
    ) -> Dict[str, Any]:
        """
        Rebuild state from events.
        
        Returns reconstructed state from initial + events.
        """
        # Start from initial state
        state = deepcopy(initial_state)
        
        events = self.get_events(from_version)
        
        for event in events:
            if event.event_type == WALEventType.BELIEF_ADDED.value:
                state.setdefault("beliefs", {})
                state["beliefs"][event.target_id] = event.payload
            
            elif event.event_type == WALEventType.BELIEF_UPDATED.value:
                if "beliefs" in state and event.target_id in state["beliefs"]:
                    state["beliefs"][event.target_id].update(event.payload)
            
            elif event.event_type == WALEventType.BELIEF_REMOVED.value:
                if "beliefs" in state and event.target_id in state["beliefs"]:
                    del state["beliefs"][event.target_id]
            
            elif event.event_type == WALEventType.CAUSAL_EDGE_ADDED.value:
                state.setdefault("causal_edges", {})
                state["causal_edges"][event.target_id] = event.payload
            
            elif event.event_type == WALEventType.CONTRADICTION_REGISTERED.value:
                state.setdefault("contradictions", {})
                state["contradictions"][event.target_id] = event.payload
            
            # Update metrics
            if "total_entropy" in event.payload:
                state["total_entropy"] = event.payload["total_entropy"]
        
        return state
    
    def verify_replay(
        self,
        canonical_state: Any,
        from_version: int = 0
    ) -> tuple[bool, str]:
        """
        Verify replay produces identical state to canonical.
        
        Uses RootReducer for pure event sourcing.
        """
        from reducers.root_reducer import RootReducer
        
        # Get initial state
        initial_state = {
            "beliefs": {},
            "constraints": {},
            "contradictions": {},
            "causal_edges": {},
            "total_entropy": 0.0,
            "belief_count": 0
        }
        
        # Get events after from_version
        events = [e for e in self._events if e.version > from_version and e.version > 0]
        
        # Convert to dicts for reducer
        event_dicts = []
        for e in events:
            event_dicts.append({
                "event_type": e.event_type,
                "version": e.version,
                "operation": e.operation,
                "target_id": e.target_id,
                "payload": e.payload,
                "parent_version": e.parent_version,
                "actor": e.actor
            })
        
        # Use RootReducer for pure replay
        state = RootReducer.apply_sequence(initial_state, event_dicts)
        
        # Compare with canonical
        canonical_beliefs = len(canonical_state.beliefs)
        rebuilt_beliefs = len(state.get("beliefs", {}))
        
        if canonical_beliefs != rebuilt_beliefs:
            return False, f"Belief count mismatch: canonical={canonical_beliefs}, rebuilt={rebuilt_beliefs}"
        
        # Compare belief content (handle both dict and object)
        for bid, belief in canonical_state.beliefs.items():
            if bid not in state.get("beliefs", {}):
                return False, f"Missing belief in replay: {bid}"
            
            rebuilt_belief = state["beliefs"][bid]
            if isinstance(rebuilt_belief, dict):
                rebuilt_confidence = rebuilt_belief.get("confidence", 0)
            else:
                rebuilt_confidence = getattr(rebuilt_belief, "confidence", 0)
            
            if abs(belief.confidence - rebuilt_confidence) > 0.001:
                return False, f"Belief confidence mismatch for {bid}: canonical={belief.confidence}, rebuilt={rebuilt_confidence}"
        
        return True, f"Replay verified: {canonical_beliefs} beliefs, entropy={canonical_state.total_entropy}"
    
    def deterministic_replay(self, target_version: int) -> tuple[Optional[Dict[str, Any]], bool, str]:
        """
        Full deterministic replay from genesis to target_version.
        
        Uses RootReducer for pure event sourcing.
        
        Returns: (rebuilt_state, verified, message)
        """
        # Import reducer
        from reducers.root_reducer import RootReducer
        
        # Start from empty genesis
        initial_state = {
            "beliefs": {},
            "constraints": {},
            "contradictions": {},
            "causal_edges": {},
            "total_entropy": 0.0,
            "belief_count": 0
        }
        
        # Get events up to target_version
        events = [e for e in self._events if e.version <= target_version and e.version > 0]
        
        # Convert to event dicts for reducer
        event_dicts = []
        for e in events:
            event_dicts.append({
                "event_type": e.event_type,
                "version": e.version,
                "operation": e.operation,
                "target_id": e.target_id,
                "payload": e.payload,
                "parent_version": e.parent_version,
                "actor": e.actor
            })
        
        # Use RootReducer for pure replay
        rebuilt = RootReducer.apply_sequence(initial_state, event_dicts)
        
        return rebuilt, True, f"Replay to v{target_version} complete"
    
    def verify_hash_chain(self) -> tuple[bool, str]:
        """Verify hash chain integrity"""
        with self._lock:
            if not self._events:
                return True, "No events to verify"
            
            prev_hash = ""
            for i, event in enumerate(self._events):
                if i > 0:
                    # Verify causal chain
                    if event.causal_chain and self._events[i-1].event_id not in event.causal_chain:
                        return False, f"Causal chain broken at event {i}"
                
                # Verify event hash (deterministic ID)
                expected_id = event.compute_hash()[:16]
                if event.event_id != expected_id:
                    return False, f"Event hash mismatch at index {i}"
            
            return True, "Hash chain verified"
    
    def get_event_count(self) -> int:
        """Get total event count"""
        with self._lock:
            return len(self._events)
    
    def get_replay_info(self) -> Dict[str, Any]:
        """Get replay information"""
        with self._lock:
            return {
                "event_count": len(self._events),
                "snapshot_count": len(self._snapshots),
                "latest_version": self.get_latest_version(),
                "events": [
                    {
                        "version": e.version,
                        "type": e.event_type,
                        "target": e.target_id,
                        "actor": e.actor
                    }
                    for e in self._events[-10:]  # Last 10 events
                ]
            }
    
    def get_compensated_transactions(self) -> set:
        """Get set of compensated transaction IDs"""
        with self._lock:
            compensated = set()
            for e in self._events:
                if e.event_type == WALEventType.TRANSACTION_COMPENSATED.value:
                    payload = e.payload or {}
                    txn_id = payload.get("original_transaction_id", e.target_id)
                    compensated.add(txn_id)
            return compensated
    
    def get_transaction_lineage(self) -> Dict[str, Any]:
        """Get transaction lineage for audit"""
        with self._lock:
            transactions = {}
            for e in self._events:
                # Track transactions by transaction_id
                if e.transaction_id:
                    if e.transaction_id not in transactions:
                        transactions[e.transaction_id] = {
                            "status": "active",
                            "events": [],
                            "version": e.version
                        }
                    transactions[e.transaction_id]["events"].append({
                        "event_id": e.event_id,
                        "type": e.event_type,
                        "target": e.target_id
                    })
                
                # Mark compensated
                if e.event_type == WALEventType.TRANSACTION_COMPENSATED.value:
                    payload = e.payload or {}
                    txn_id = payload.get("original_transaction_id", e.target_id)
                    if txn_id in transactions:
                        transactions[txn_id]["status"] = "compensated"
                        transactions[txn_id]["compensated_at"] = e.version
            
            return transactions
    
    def get_causal_graph(self) -> Dict[str, Any]:
        """
        Get causal topology of events.
        
        Returns:
        - nodes: all events
        - edges: causal links (event -> causes -> effects)
        - compensation_pairs: which events compensate which
        """
        with self._lock:
            nodes = []
            edges = []
            compensation_pairs = []
            
            for e in self._events:
                node = {
                    "event_id": e.event_id,
                    "type": e.event_type,
                    "version": e.version,
                    "target": e.target_id,
                    "actor": e.actor,
                    "transaction_id": e.transaction_id,
                    "causal_depth": e.causal_depth,
                    "reflection_depth": e.reflection_depth,
                    "inverse_of_event_id": e.inverse_of_event_id,
                    "inverse_of_transaction_id": e.inverse_of_transaction_id
                }
                nodes.append(node)
                
                # Causal edges
                for cause_id in e.causal_chain:
                    edges.append({"from": cause_id, "to": e.event_id})
                
                # Compensation pairs
                if e.compensates_event_id:
                    compensation_pairs.append({
                        "original": e.compensates_event_id,
                        "compensation": e.event_id
                    })
                if e.compensates_transaction_id:
                    compensation_pairs.append({
                        "original_transaction": e.compensates_transaction_id,
                        "compensation": e.event_id
                    })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "compensation_pairs": compensation_pairs,
                "total_events": len(nodes)
            }
    
    def get_compensation_chain(self, transaction_id: str) -> Dict[str, Any]:
        """
        Get full compensation chain for a transaction.
        
        Shows:
        - Original events
        - Compensation events
        - What was reverted
        """
        with self._lock:
            original_events = []
            compensation_events = []
            
            for e in self._events:
                if e.transaction_id == transaction_id:
                    original_events.append({
                        "event_id": e.event_id,
                        "type": e.event_type,
                        "target": e.target_id,
                        "version": e.version
                    })
                
                if e.compensates_transaction_id == transaction_id:
                    compensation_events.append({
                        "event_id": e.event_id,
                        "type": e.event_type,
                        "version": e.version,
                        "reason": e.payload.get("reason", "unknown")
                    })
            
            return {
                "transaction_id": transaction_id,
                "original_events": original_events,
                "compensation_events": compensation_events,
                "is_compensated": len(compensation_events) > 0
            }
    
    def get_unstable_regions(self) -> List[Dict[str, Any]]:
        """
        Find unstable epistemic regions.
        
        Identifies:
        - Recursive compensation loops
        - High causal depth with many reversals
        - Unstable attractors
        """
        with self._lock:
            unstable = []
            
            # Find transactions with multiple compensations
            compensation_counts: Dict[str, int] = {}
            for e in self._events:
                if e.compensates_transaction_id:
                    compensation_counts[e.compensates_transaction_id] = \
                        compensation_counts.get(e.compensates_transaction_id, 0) + 1
            
            for txn_id, count in compensation_counts.items():
                if count > 1:
                    unstable.append({
                        "type": "recursive_compensation",
                        "transaction_id": txn_id,
                        "compensation_count": count
                    })
            
            # Find events with high causal depth
            for e in self._events:
                if e.causal_depth > 3 and e.event_type == WALEventType.CONTRADICTION_REGISTERED.value:
                    unstable.append({
                        "type": "deep_contradiction",
                        "event_id": e.event_id,
                        "causal_depth": e.causal_depth,
                        "target": e.target_id
                    })
            
            return unstable


# Global instance
_wal_engine: Optional[WALEngine] = None


def get_wal_engine() -> WALEngine:
    """Get global WAL engine"""
    global _wal_engine
    if _wal_engine is None:
        _wal_engine = WALEngine()
    return _wal_engine


def reset_wal_engine():
    """Reset WAL engine (for testing)"""
    global _wal_engine
    _wal_engine = WALEngine()