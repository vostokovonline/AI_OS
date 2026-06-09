"""
Root Reducer - Unified reducer for all cognitive events.

Single source of truth for mutation logic.
Used by both runtime AND replay for deterministic equivalence.
"""
from typing import Dict, Any, Optional
from copy import deepcopy
from datetime import datetime
import hashlib
import json

from .belief_reducer import BeliefReducer
from .causal_reducer import CausalReducer
from .contradiction_reducer import ContradictionReducer


class RootReducer:
    """
    Unified reducer for all epistemic state mutations.
    
    Key invariant:
        root_reducer.apply(state, event) MUST produce identical results
        for both RUNTIME execution and REPLAY from WAL.
    """
    
    # Event type mapping
    REDUCER_MAP = {
        "belief_added": ("belief", "add"),
        "belief_updated": ("belief", "update"),
        "belief_removed": ("belief", "remove"),
        "causal_edge_added": ("causal", "add"),
        "contradiction_registered": ("contradiction", "add"),
        "contradiction_resolved": ("contradiction", "resolve"),
        "state_committed": ("metrics", "update"),
    }
    
    @staticmethod
    def apply(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply single event to state.
        
        Returns NEW state (immutable transform).
        """
        event_type = event.get("event_type", "")
        
        reducer_key = RootReducer.REDUCER_MAP.get(event_type)
        
        if reducer_key is None:
            # Unknown event type - pass through
            return state
        
        reducer_name, operation = reducer_key
        
        # Create event with operation for reducer
        enriched_event = {
            **event,
            "operation": operation
        }
        
        if reducer_name == "belief":
            return BeliefReducer.apply(state, enriched_event)
        elif reducer_name == "causal":
            return CausalReducer.apply(state, enriched_event)
        elif reducer_name == "contradiction":
            return ContradictionReducer.apply(state, enriched_event)
        elif reducer_name == "metrics":
            # State committed - update metrics
            return RootReducer._apply_metrics_update(state, enriched_event)
        
        return state
    
    @staticmethod
    def _apply_metrics_update(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply metrics update from state commit"""
        new_state = deepcopy(state)
        payload = event.get("payload", {})
        
        new_state["total_entropy"] = payload.get("total_entropy", 0.0)
        new_state["belief_count"] = payload.get("belief_count", 0)
        
        return new_state
    
    @staticmethod
    def apply_sequence(state: Dict[str, Any], events: list) -> Dict[str, Any]:
        """Apply sequence of events deterministically"""
        current_state = state
        
        for event in events:
            current_state = RootReducer.apply(current_state, event)
        
        return current_state
    
    @staticmethod
    def compute_state_hash(state: Dict[str, Any]) -> str:
        """
        Compute deterministic hash of state.
        
        Key for replay verification.
        """
        # Serialize beliefs deterministically
        beliefs = state.get("beliefs", {})
        belief_keys = sorted(beliefs.keys())
        
        belief_data = []
        for k in belief_keys:
            b = beliefs[k]
            if hasattr(b, "__dict__"):
                # Dataclass - convert to dict
                belief_data.append({
                    "id": b.belief_id,
                    "conf": b.confidence,
                    "ent": b.entropy
                })
            else:
                # Dict
                belief_data.append({
                    "id": k,
                    "conf": b.get("confidence", 0),
                    "ent": b.get("entropy", 0)
                })
        
        content = {
            "beliefs": belief_data,
            "entropy": state.get("total_entropy", 0),
            "belief_count": len(belief_data)
        }
        
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    
    @staticmethod
    def replay_to_version(wal_events: list, target_version: int, initial_state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Replay events to target version using pure reducer.
        
        This is the canonical replay function used for verification.
        """
        if initial_state is None:
            initial_state = {
                "beliefs": {},
                "causal_edges": {},
                "contradictions": {},
                "total_entropy": 0.0,
                "belief_count": 0
            }
        
        # Filter events up to target version
        relevant_events = [e for e in wal_events if e.version <= target_version and e.version > 0]
        
        # Sort by version for deterministic replay
        relevant_events.sort(key=lambda e: e.version)
        
        # Apply all events
        return RootReducer.apply_sequence(initial_state, relevant_events)