"""
Contradiction Reducer - Pure event sourcing for contradiction mutations.
"""
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from copy import deepcopy


@dataclass
class Contradiction:
    """Immutable contradiction"""
    episode_id: str
    belief_ids: list
    contradiction_type: str
    first_seen: str
    last_seen: str
    recurrence_count: int
    stability_score: float
    resolution_status: str
    severity: str


class ContradictionReducer:
    """Pure reducer for contradiction mutations"""
    
    @staticmethod
    def apply_register(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Pure register: add new contradiction"""
        new_state = deepcopy(state)
        new_state.setdefault("contradictions", {})
        
        payload = event["payload"]
        
        contradiction = Contradiction(
            episode_id=event["target_id"],
            belief_ids=payload.get("belief_ids", []),
            contradiction_type=payload.get("contradiction_type", "unknown"),
            first_seen=payload.get("first_seen", datetime.utcnow().isoformat()),
            last_seen=payload.get("last_seen", datetime.utcnow().isoformat()),
            recurrence_count=payload.get("recurrence_count", 1),
            stability_score=payload.get("stability_score", 0.5),
            resolution_status=payload.get("resolution_status", "unresolved"),
            severity=payload.get("severity", "medium")
        )
        
        new_state["contradictions"][contradiction.episode_id] = contradiction
        return new_state
    
    @staticmethod
    def apply_resolve(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Pure resolve: mark contradiction as resolved"""
        new_state = deepcopy(state)
        new_state.setdefault("contradictions", {})
        
        target_id = event["target_id"]
        if target_id in new_state["contradictions"]:
            old = new_state["contradictions"][target_id]
            new_state["contradictions"][target_id] = Contradiction(
                episode_id=old.episode_id,
                belief_ids=old.belief_ids,
                contradiction_type=old.contradiction_type,
                first_seen=old.first_seen,
                last_seen=datetime.utcnow().isoformat(),
                recurrence_count=old.recurrence_count,
                stability_score=old.stability_score,
                resolution_status="resolved",
                severity=old.severity
            )
        
        return new_state
    
    @staticmethod
    def apply(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply contradiction event"""
        op = event.get("operation", "")
        
        if op == "add" or op == "register":
            return ContradictionReducer.apply_register(state, event)
        elif op == "resolve":
            return ContradictionReducer.apply_resolve(state, event)
        
        return state