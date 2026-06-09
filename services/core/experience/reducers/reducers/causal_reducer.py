"""
Causal Reducer - Pure event sourcing for causal edge mutations.
"""
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime
from copy import deepcopy


@dataclass
class CausalEdge:
    """Immutable causal edge"""
    edge_id: str
    cause_ids: list
    effect_ids: list
    weight: float
    evidence_strength: float
    temporal_distance: int
    created_at: str
    policy_mediation: str = None


class CausalReducer:
    """Pure reducer for causal edge mutations"""
    
    @staticmethod
    def apply_add(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Pure add: create new causal edge"""
        new_state = deepcopy(state)
        new_state.setdefault("causal_edges", {})
        
        payload = event["payload"]
        
        edge = CausalEdge(
            edge_id=event["target_id"],
            cause_ids=payload.get("cause_ids", []),
            effect_ids=payload.get("effect_ids", []),
            weight=payload.get("weight", 0.5),
            evidence_strength=payload.get("evidence_strength", 0.5),
            temporal_distance=1,
            created_at=payload.get("created_at", datetime.utcnow().isoformat()),
            policy_mediation=payload.get("policy_mediation")
        )
        
        new_state["causal_edges"][edge.edge_id] = edge
        
        # Update node references (pure - rebuild lists)
        for cause_id in edge.cause_ids:
            if cause_id in new_state.get("beliefs", {}):
                belief = new_state["beliefs"][cause_id]
                if edge.edge_id not in belief.outgoing_effects:
                    belief.outgoing_effects = belief.outgoing_effects + [edge.edge_id]
        
        for effect_id in edge.effect_ids:
            if effect_id in new_state.get("beliefs", {}):
                belief = new_state["beliefs"][effect_id]
                if edge.edge_id not in belief.incoming_causes:
                    belief.incoming_causes = belief.incoming_causes + [edge.edge_id]
        
        return new_state
    
    @staticmethod
    def apply(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
        """Apply causal event"""
        if event.get("operation") == "add":
            return CausalReducer.apply_add(state, event)
        return state