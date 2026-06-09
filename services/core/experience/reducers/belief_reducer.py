"""
Belief Reducer - Pure event sourcing for belief mutations.

No implicit updates. Every mutation is explicit.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4
from copy import deepcopy


@dataclass
class Belief:
    """Immutable belief structure"""
    belief_id: str
    proposition: str
    confidence: float
    entropy: float
    source: str
    created_at: str
    last_updated: str
    incoming_causes: list
    outgoing_effects: list
    attractor_state: Optional[str] = None


class BeliefReducer:
    """Pure reducer for belief mutations"""
    
    @staticmethod
    def apply_add(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pure add: create new belief, don't merge with existing.
        
        Returns NEW state dict (immutable).
        """
        new_state = deepcopy(state)
        new_state.setdefault("beliefs", {})
        
        belief = Belief(
            belief_id=event["target_id"],
            proposition=event["payload"]["proposition"],
            confidence=event["payload"]["confidence"],
            entropy=event["payload"]["entropy"],
            source=event["payload"]["source"],
            created_at=event["payload"]["created_at"],
            last_updated=event["payload"]["created_at"],
            incoming_causes=event["payload"].get("incoming_causes", []),
            outgoing_effects=event["payload"].get("outgoing_effects", [])
        )
        
        new_state["beliefs"][belief.belief_id] = belief
        return new_state
    
    @staticmethod
    def apply_update(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pure update: REPLACE belief with new values, don't merge.
        
        This is key: .update() is wrong - should be replace.
        """
        new_state = deepcopy(state)
        new_state.setdefault("beliefs", {})
        
        target_id = event["target_id"]
        if target_id not in new_state["beliefs"]:
            return new_state
        
        # PURE REPLACE - not merge
        old_belief = new_state["beliefs"][target_id]
        
        new_belief = Belief(
            belief_id=old_belief.belief_id,
            proposition=old_belief.proposition,
            confidence=event["payload"]["confidence"],
            entropy=event["payload"].get("entropy", old_belief.entropy),
            source=old_belief.source,
            created_at=old_belief.created_at,
            last_updated=datetime.utcnow().isoformat(),
            incoming_causes=old_belief.incoming_causes,
            outgoing_effects=old_belief.outgoing_effects,
            attractor_state=event["payload"].get("attractor_state", old_belief.attractor_state)
        )
        
        new_state["beliefs"][target_id] = new_belief
        return new_state
    
    @staticmethod
    def apply_remove(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Pure remove: delete belief completely"""
        new_state = deepcopy(state)
        new_state.setdefault("beliefs", {})
        
        target_id = event["target_id"]
        if target_id in new_state["beliefs"]:
            del new_state["beliefs"][target_id]
        
        return new_state
    
    @staticmethod
    def apply(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply any belief event"""
        op = event.get("operation", "")
        
        if op == "add":
            return BeliefReducer.apply_add(state, event)
        elif op == "update":
            return BeliefReducer.apply_update(state, event)
        elif op == "remove":
            return BeliefReducer.apply_remove(state, event)
        else:
            return state