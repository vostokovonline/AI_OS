"""
Pure Immutable Reducers.

All reducers are PURE FUNCTIONS:
- No side effects
- No mutations
- Return entirely new state
- Deterministic: same input → same output
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, Any
from datetime import datetime

from cognitive_state import (
    CognitiveState, 
    BeliefState, 
    CausalEdgeState, 
    ContradictionState,
    TransactionRecord,
    initial_state,
    create_state_from_dicts
)


def reduce_belief_add(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: add belief. Returns NEW state, never mutates."""
    payload = event.get("payload", {})
    belief_id = event.get("target_id", payload.get("belief_id", ""))
    
    if belief_id in state._beliefs_dict:
        return state
    
    new_belief = BeliefState(
        belief_id=belief_id,
        proposition=payload.get("proposition", ""),
        confidence=payload.get("confidence", 0.5),
        entropy=payload.get("entropy", 0.0),
        source=payload.get("source", "unknown"),
        created_at=payload.get("created_at", datetime.utcnow().isoformat()),
        last_updated=payload.get("created_at", datetime.utcnow().isoformat()),
        incoming_causes=tuple(payload.get("incoming_causes", [])),
        outgoing_effects=tuple(payload.get("outgoing_effects", [])),
        attractor_state=payload.get("attractor_state")
    )
    
    new_beliefs = {**state._beliefs_dict, belief_id: new_belief}
    
    return create_state_from_dicts(
        beliefs=new_beliefs,
        causal_edges=state._causal_edges_dict,
        contradictions=state._contradictions_dict,
        transactions=state._transactions_dict,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_belief_update(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: update belief. REPLACES belief entirely."""
    payload = event.get("payload", {})
    belief_id = event.get("target_id", payload.get("belief_id", ""))
    
    if belief_id not in state._beliefs_dict:
        return state
    
    old = state._beliefs_dict[belief_id]
    
    new_belief = BeliefState(
        belief_id=old.belief_id,
        proposition=old.proposition,
        confidence=payload.get("confidence", old.confidence),
        entropy=payload.get("entropy", old.entropy),
        source=old.source,
        created_at=old.created_at,
        last_updated=datetime.utcnow().isoformat(),
        incoming_causes=old.incoming_causes,
        outgoing_effects=old.outgoing_effects,
        attractor_state=payload.get("attractor_state", old.attractor_state)
    )
    
    new_beliefs = {**state._beliefs_dict, belief_id: new_belief}
    
    return create_state_from_dicts(
        beliefs=new_beliefs,
        causal_edges=state._causal_edges_dict,
        contradictions=state._contradictions_dict,
        transactions=state._transactions_dict,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_belief_remove(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: remove belief"""
    belief_id = event.get("target_id", event.get("payload", {}).get("belief_id", ""))
    
    if belief_id not in state._beliefs_dict:
        return state
    
    new_beliefs = {k: v for k, v in state._beliefs_dict.items() if k != belief_id}
    
    return create_state_from_dicts(
        beliefs=new_beliefs,
        causal_edges=state._causal_edges_dict,
        contradictions=state._contradictions_dict,
        transactions=state._transactions_dict,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_causal_add(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: add causal edge"""
    payload = event.get("payload", {})
    edge_id = event.get("target_id", payload.get("edge_id", ""))
    
    if edge_id in state._causal_edges_dict:
        return state
    
    new_edge = CausalEdgeState(
        edge_id=edge_id,
        cause_ids=tuple(payload.get("cause_ids", [])),
        effect_ids=tuple(payload.get("effect_ids", [])),
        weight=payload.get("weight", 0.5),
        evidence_strength=payload.get("evidence_strength", 0.5),
        temporal_distance=payload.get("temporal_distance", 1),
        created_at=payload.get("created_at", datetime.utcnow().isoformat()),
        policy_mediation=payload.get("policy_mediation")
    )
    
    new_edges = {**state._causal_edges_dict, edge_id: new_edge}
    
    return create_state_from_dicts(
        beliefs=state._beliefs_dict,
        causal_edges=new_edges,
        contradictions=state._contradictions_dict,
        transactions=state._transactions_dict,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_contradiction_register(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: register contradiction"""
    payload = event.get("payload", {})
    episode_id = event.get("target_id", payload.get("episode_id", ""))
    
    if episode_id in state._contradictions_dict:
        return state
    
    new_contradiction = ContradictionState(
        episode_id=episode_id,
        belief_ids=tuple(payload.get("belief_ids", [])),
        contradiction_type=payload.get("contradiction_type", "unknown"),
        first_seen=payload.get("first_seen", datetime.utcnow().isoformat()),
        last_seen=payload.get("last_seen", datetime.utcnow().isoformat()),
        recurrence_count=payload.get("recurrence_count", 1),
        stability_score=payload.get("stability_score", 0.5),
        resolution_status="unresolved",
        severity=payload.get("severity", "medium")
    )
    
    new_contradictions = {**state._contradictions_dict, episode_id: new_contradiction}
    
    return create_state_from_dicts(
        beliefs=state._beliefs_dict,
        causal_edges=state._causal_edges_dict,
        contradictions=new_contradictions,
        transactions=state._transactions_dict,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_contradiction_resolve(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: resolve contradiction"""
    episode_id = event.get("target_id", event.get("payload", {}).get("episode_id", ""))
    
    if episode_id not in state._contradictions_dict:
        return state
    
    old = state._contradictions_dict[episode_id]
    
    new_contradiction = ContradictionState(
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
    
    new_contradictions = {**state._contradictions_dict, episode_id: new_contradiction}
    
    return create_state_from_dicts(
        beliefs=state._beliefs_dict,
        causal_edges=state._causal_edges_dict,
        contradictions=new_contradictions,
        transactions=state._transactions_dict,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_transaction_commit(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: commit transaction"""
    payload = event.get("payload", {})
    tx_id = event.get("target_id", payload.get("transaction_id", ""))
    
    if tx_id in state._transactions_dict:
        return state
    
    new_record = TransactionRecord(
        transaction_id=tx_id,
        status="committed",
        created_at=payload.get("created_at", datetime.utcnow().isoformat())
    )
    
    new_transactions = {**state._transactions_dict, tx_id: new_record}
    
    return create_state_from_dicts(
        beliefs=state._beliefs_dict,
        causal_edges=state._causal_edges_dict,
        contradictions=state._contradictions_dict,
        transactions=new_transactions,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_transaction_compensate(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: compensate transaction"""
    payload = event.get("payload", {})
    tx_id = payload.get("original_transaction_id", event.get("target_id", ""))
    
    if tx_id not in state._transactions_dict:
        return state
    
    old = state._transactions_dict[tx_id]
    
    new_record = TransactionRecord(
        transaction_id=old.transaction_id,
        status="compensated",
        created_at=old.created_at,
        compensated_at=datetime.utcnow().isoformat(),
        reason=payload.get("reason", "manual_compensation")
    )
    
    new_transactions = {**state._transactions_dict, tx_id: new_record}
    
    return create_state_from_dicts(
        beliefs=state._beliefs_dict,
        causal_edges=state._causal_edges_dict,
        contradictions=state._contradictions_dict,
        transactions=new_transactions,
        total_entropy=state.total_entropy,
        version=state.version + 1
    )


def reduce_metrics_update(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Pure reducer: update metrics"""
    payload = event.get("payload", {})
    
    return create_state_from_dicts(
        beliefs=state._beliefs_dict,
        causal_edges=state._causal_edges_dict,
        contradictions=state._contradictions_dict,
        transactions=state._transactions_dict,
        total_entropy=payload.get("total_entropy", state.total_entropy),
        version=state.version + 1
    )


EVENT_REDUCERS = {
    "belief_added": reduce_belief_add,
    "belief_updated": reduce_belief_update,
    "belief_removed": reduce_belief_remove,
    "causal_edge_added": reduce_causal_add,
    "contradiction_registered": reduce_contradiction_register,
    "contradiction_resolved": reduce_contradiction_resolve,
    "transaction_committed": reduce_transaction_commit,
    "transaction_compensated": reduce_transaction_compensate,
    "state_committed": reduce_metrics_update,
}


def reduce(state: CognitiveState, event: Dict[str, Any]) -> CognitiveState:
    """Main reducer function. Routes event to appropriate reducer."""
    event_type = event.get("event_type", "")
    
    reducer = EVENT_REDUCERS.get(event_type)
    if reducer is None:
        return state
    
    return reducer(state, event)


def reduce_sequence(state: CognitiveState, events: list) -> CognitiveState:
    """Apply sequence of events, returning final state"""
    current = state
    for event in events:
        current = reduce(current, event)
    return current