"""
Pure Domain Reducers - Pure functions for state transitions.

Stage 1: Domain Core only
- Pure functions: state + event → new_state
- NO side effects
- NO event emission
- NO store writes
- NO mutation IDs
- NO timestamps changes
- Deterministic: same input → same output

These are the ONLY way state changes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from types import MappingProxyType
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

from state import (
    DomainState, 
    ImmutableBelief, 
    ImmutableGenome, 
    ImmutableIdentity,
    ImmutableContradiction,
    ImmutableLineage,
    create_initial_state
)
from events import DomainEvent


def reduce_belief_added(state: DomainState, event: DomainEvent) -> DomainState:
    """
    Pure reducer: add belief.
    
    Returns NEW state, never mutates.
    No event emission, no store write.
    """
    payload = dict(event.payload)
    belief_id = payload.get("belief_id", event.event_id)
    
    if belief_id in state.beliefs:
        return state
    
    new_belief = ImmutableBelief(
        belief_id=belief_id,
        proposition=payload.get("proposition", ""),
        confidence=payload.get("confidence", 0.5),
        entropy=payload.get("entropy", 0.0),
        source=payload.get("source", "unknown"),
        created_at=payload.get("created_at", event.timestamp),
        last_updated=event.timestamp,
        incoming_causes=tuple(payload.get("incoming_causes", [])),
        outgoing_effects=tuple(payload.get("outgoing_effects", [])),
        attractor_state=payload.get("attractor_state")
    )
    
    new_beliefs = {**state.beliefs, belief_id: new_belief}
    
    return DomainState(
        beliefs=MappingProxyType(new_beliefs),
        genome=state.genome,
        identity=state.identity,
        contradictions=state.contradictions,
        lineage=state.lineage,
        version=state.version + 1,
        entropy=state.entropy
    )


def reduce_belief_updated(state: DomainState, event: DomainEvent) -> DomainState:
    """Pure reducer: update belief"""
    payload = dict(event.payload)
    belief_id = payload.get("belief_id", event.event_id)
    
    if belief_id not in state.beliefs:
        return state
    
    old = state.beliefs[belief_id]
    
    new_belief = ImmutableBelief(
        belief_id=old.belief_id,
        proposition=old.proposition,
        confidence=payload.get("confidence", old.confidence),
        entropy=payload.get("entropy", old.entropy),
        source=old.source,
        created_at=old.created_at,
        last_updated=event.timestamp,
        incoming_causes=old.incoming_causes,
        outgoing_effects=old.outgoing_effects,
        attractor_state=payload.get("attractor_state", old.attractor_state)
    )
    
    new_beliefs = {**state.beliefs, belief_id: new_belief}
    
    return DomainState(
        beliefs=MappingProxyType(new_beliefs),
        genome=state.genome,
        identity=state.identity,
        contradictions=state.contradictions,
        lineage=state.lineage,
        version=state.version + 1,
        entropy=state.entropy
    )


def reduce_identity_mutated(state: DomainState, event: DomainEvent) -> DomainState:
    """
    Pure reducer: mutate identity axis.
    
    CRITICAL: This does NOT emit events.
    This does NOT write to store.
    This does NOT create mutation IDs.
    
    Only returns new state.
    """
    payload = dict(event.payload)
    axis = payload.get("axis", "autonomy")
    delta = payload.get("delta", 0.0)
    
    new_identity = state.identity.with_axis(axis, delta)
    
    new_history = state.lineage.mutation_history + ((axis, delta),)
    new_trajectory_hash = ImmutableLineage.compute_hash(new_history)
    
    new_lineage = ImmutableLineage(
        mutation_history=new_history,
        cause_chains=state.lineage.cause_chains + ((event.event_id, event.causation_id),),
        trajectory_hash=new_trajectory_hash
    )
    
    return DomainState(
        beliefs=state.beliefs,
        genome=state.genome,
        identity=new_identity,
        contradictions=state.contradictions,
        lineage=new_lineage,
        version=state.version + 1,
        entropy=state.entropy
    )


def reduce_genome_evolved(state: DomainState, event: DomainEvent) -> DomainState:
    """Pure reducer: evolve genome"""
    payload = dict(event.payload)
    
    axis = payload.get("axis", "autonomy")
    value = payload.get("value", 0.5)
    selection_pressure = payload.get("selection_pressure", state.genome.selection_pressure)
    mutation_rate = payload.get("mutation_rate", state.genome.mutation_rate)
    
    new_genome = state.genome.with_axis(axis, value)
    
    return DomainState(
        beliefs=state.beliefs,
        genome=new_genome,
        identity=state.identity,
        contradictions=state.contradictions,
        lineage=state.lineage,
        version=state.version + 1,
        entropy=state.entropy
    )


def reduce_contradiction_registered(state: DomainState, event: DomainEvent) -> DomainState:
    """Pure reducer: register contradiction"""
    payload = dict(event.payload)
    episode_id = payload.get("episode_id", event.event_id)
    
    if episode_id in state.contradictions:
        return state
    
    new_contradiction = ImmutableContradiction(
        episode_id=episode_id,
        belief_ids=tuple(payload.get("belief_ids", [])),
        contradiction_type=payload.get("contradiction_type", "unknown"),
        first_seen=event.timestamp,
        last_seen=event.timestamp,
        recurrence_count=payload.get("recurrence_count", 1),
        stability_score=payload.get("stability_score", 0.5),
        resolution_status="unresolved",
        severity=payload.get("severity", "medium")
    )
    
    new_contradictions = {**state.contradictions, episode_id: new_contradiction}
    
    return DomainState(
        beliefs=state.beliefs,
        genome=state.genome,
        identity=state.identity,
        contradictions=MappingProxyType(new_contradictions),
        lineage=state.lineage,
        version=state.version + 1,
        entropy=state.entropy
    )


def reduce_contradiction_resolved(state: DomainState, event: DomainEvent) -> DomainState:
    """Pure reducer: resolve contradiction"""
    payload = dict(event.payload)
    episode_id = payload.get("episode_id", event.event_id)
    
    if episode_id not in state.contradictions:
        return state
    
    old = state.contradictions[episode_id]
    
    new_contradiction = ImmutableContradiction(
        episode_id=old.episode_id,
        belief_ids=old.belief_ids,
        contradiction_type=old.contradiction_type,
        first_seen=old.first_seen,
        last_seen=event.timestamp,
        recurrence_count=old.recurrence_count,
        stability_score=old.stability_score,
        resolution_status="resolved",
        severity=old.severity
    )
    
    new_contradictions = {**state.contradictions, episode_id: new_contradiction}
    
    return DomainState(
        beliefs=state.beliefs,
        genome=state.genome,
        identity=state.identity,
        contradictions=MappingProxyType(new_contradictions),
        lineage=state.lineage,
        version=state.version + 1,
        entropy=state.entropy
    )


EVENT_REDUCERS = {
    "belief_added": reduce_belief_added,
    "belief_updated": reduce_belief_updated,
    "identity_mutated": reduce_identity_mutated,
    "genome_evolved": reduce_genome_evolved,
    "contradiction_registered": reduce_contradiction_registered,
    "contradiction_resolved": reduce_contradiction_resolved,
}


def reduce(state: DomainState, event: DomainEvent) -> DomainState:
    """
    Main reducer function.
    
    Pure function: state + event → new_state
    NO side effects.
    """
    reducer = EVENT_REDUCERS.get(event.event_type)
    if reducer is None:
        return state
    
    return reducer(state, event)


def reduce_sequence(state: DomainState, events: Tuple[DomainEvent, ...]) -> DomainState:
    """
    Apply sequence of events.
    
    Pure function: same events always produce same state.
    """
    current = state
    for event in events:
        current = reduce(current, event)
    return current


def replay_from_events(events: Tuple[DomainEvent, ...]) -> DomainState:
    """
    Replay events from scratch.
    
    Returns initial_state → events → final_state
    Pure function, no side effects.
    """
    return reduce_sequence(create_initial_state(), events)


def materialize_state(events: Tuple[DomainEvent, ...], from_position: int = 1) -> DomainState:
    """
    Materialize state from event stream.
    
    Args:
        events: Tuple of events (hashable for determinism)
        from_position: Start from this position
    
    Returns materialized state.
    """
    filtered = tuple(e for e in events if e.position >= from_position)
    return replay_from_events(tuple(sorted(filtered, key=lambda e: e.position)))