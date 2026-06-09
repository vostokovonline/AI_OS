"""
Pure Reducers - State materialization from events.

Stage: Four-Layer Architecture

All state is derived from events via pure reducers.
No mutable runtime state. No side effects.

Architecture:
    ImmutableEvent → Reducer → MaterializedState

Reducers are:
- Pure functions: (state, event) → new_state
- Deterministic: same input → same output
- Composable: reducers can chain
- Verifiable: replay produces same state
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

from event_log import ImmutableEvent, EventLog, EventTypes, StreamIds


# State type variables

StateT = TypeVar('StateT')
EventT = TypeVar('EventT')


@dataclass(frozen=True)
class MaterializedState:
    """
    Materialized state from event log.
    
    This is the result of applying events via reducers.
    It is NOT the source of truth - events are.
    """
    state_type: str
    version: int
    timestamp: str
    data: MappingProxyType  # type: ignore
    hash: str
    
    def __post_init__(self):
        if not isinstance(self.data, MappingProxyType):
            object.__setattr__(self, 'data', MappingProxyType(dict(self.data)))
    
    @staticmethod
    def compute_hash(state_type: str, version: int, data: Dict[str, Any]) -> str:
        content = {
            "type": state_type,
            "version": version,
            "data": dict(sorted(data.items()))
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class CognitiveState:
    """Materialized cognitive state"""
    beliefs: MappingProxyType  # type: ignore
    contradictions: MappingProxyType  # type: ignore
    pressures: MappingProxyType  # type: ignore
    version: int
    
    def __post_init__(self):
        for attr in ('beliefs', 'contradictions', 'pressures'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))


@dataclass(frozen=True)
class ExecutionState:
    """Materialized execution state"""
    goals: MappingProxyType  # type: ignore
    actions: MappingProxyType  # type: ignore
    results: MappingProxyType  # type: ignore
    version: int
    
    def __post_init__(self):
        for attr in ('goals', 'actions', 'results'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))


@dataclass(frozen=True)
class IdentityState:
    """Materialized identity state"""
    axes: MappingProxyType  # type: ignore
    genome: MappingProxyType  # type: ignore
    lineage: MappingProxyType  # type: ignore
    version: int
    
    def __post_init__(self):
        for attr in ('axes', 'genome', 'lineage'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))


# Pure Reducer Functions

def reduce_belief_created(state: CognitiveState, event: ImmutableEvent) -> CognitiveState:
    """Reduce belief_created event"""
    payload = event.get_payload_dict()
    belief_id = payload.get("belief_id", event.causal.event_id)
    proposition = payload.get("proposition", "")
    confidence = payload.get("confidence", 0.5)
    
    new_beliefs = {**state.beliefs, belief_id: {
        "proposition": proposition,
        "confidence": confidence,
        "source": payload.get("source", "unknown"),
        "created_at": event.timestamp,
        "stability": payload.get("stability", 0.5)
    }}
    
    return CognitiveState(
        beliefs=MappingProxyType(new_beliefs),
        contradictions=state.contradictions,
        pressures=state.pressures,
        version=state.version + 1
    )


def reduce_belief_updated(state: CognitiveState, event: ImmutableEvent) -> CognitiveState:
    """Reduce belief_updated event"""
    payload = event.get_payload_dict()
    belief_id = payload.get("belief_id", "")
    
    if belief_id not in state.beliefs:
        return state
    
    old = state.beliefs[belief_id]
    new_beliefs = {**state.beliefs, belief_id: {
        **old,
        "confidence": payload.get("confidence", old["confidence"]),
        "stability": payload.get("stability", old["stability"]),
        "updated_at": event.timestamp
    }}
    
    return CognitiveState(
        beliefs=MappingProxyType(new_beliefs),
        contradictions=state.contradictions,
        pressures=state.pressures,
        version=state.version + 1
    )


def reduce_contradiction_detected(state: CognitiveState, event: ImmutableEvent) -> CognitiveState:
    """Reduce contradiction_detected event"""
    payload = event.get_payload_dict()
    contradiction_id = payload.get("contradiction_id", event.causal.event_id)
    
    new_contradictions = {**state.contradictions, contradiction_id: {
        "type": payload.get("type", "unknown"),
        "participants": payload.get("participants", []),
        "intensity": payload.get("intensity", 0.5),
        "detected_at": event.timestamp,
        "status": "unresolved"
    }}
    
    return CognitiveState(
        beliefs=state.beliefs,
        contradictions=MappingProxyType(new_contradictions),
        pressures=state.pressures,
        version=state.version + 1
    )


def reduce_pressure_accumulated(state: CognitiveState, event: ImmutableEvent) -> CognitiveState:
    """Reduce pressure_accumulated event"""
    payload = event.get_payload_dict()
    pressure_id = payload.get("pressure_id", event.causal.event_id)
    
    new_pressures = {**state.pressures, pressure_id: {
        "source": payload.get("source", ""),
        "intensity": payload.get("intensity", 0.5),
        "type": payload.get("type", "cognitive"),
        "accumulated_at": event.timestamp
    }}
    
    return CognitiveState(
        beliefs=state.beliefs,
        contradictions=state.contradictions,
        pressures=MappingProxyType(new_pressures),
        version=state.version + 1
    )


def reduce_goal_created(state: ExecutionState, event: ImmutableEvent) -> ExecutionState:
    """Reduce goal_created event"""
    payload = event.get_payload_dict()
    goal_id = payload.get("goal_id", event.causal.event_id)
    
    new_goals = {**state.goals, goal_id: {
        "title": payload.get("title", ""),
        "priority": payload.get("priority", 0.5),
        "status": "created",
        "created_at": event.timestamp,
        "dependencies": payload.get("dependencies", [])
    }}
    
    return ExecutionState(
        goals=MappingProxyType(new_goals),
        actions=state.actions,
        results=state.results,
        version=state.version + 1
    )


def reduce_goal_executed(state: ExecutionState, event: ImmutableEvent) -> CognitiveState:
    """Reduce goal_executed event"""
    payload = event.get_payload_dict()
    goal_id = payload.get("goal_id", "")
    
    if goal_id not in state.goals:
        return state
    
    old = state.goals[goal_id]
    new_goals = {**state.goals, goal_id: {
        **old,
        "status": "executing",
        "started_at": event.timestamp
    }}
    
    return ExecutionState(
        goals=MappingProxyType(new_goals),
        actions=state.actions,
        results=state.results,
        version=state.version + 1
    )


def reduce_goal_completed(state: ExecutionState, event: ImmutableEvent) -> ExecutionState:
    """Reduce goal_completed event"""
    payload = event.get_payload_dict()
    goal_id = payload.get("goal_id", "")
    
    if goal_id not in state.goals:
        return state
    
    old = state.goals[goal_id]
    new_goals = {**state.goals, goal_id: {
        **old,
        "status": "completed",
        "completed_at": event.timestamp,
        "outcome": payload.get("outcome", "success")
    }}
    
    return ExecutionState(
        goals=MappingProxyType(new_goals),
        actions=state.actions,
        results=state.results,
        version=state.version + 1
    )


def reduce_identity_mutated(state: IdentityState, event: ImmutableEvent) -> IdentityState:
    """Reduce identity_mutated event"""
    payload = event.get_payload_dict()
    axis = payload.get("axis", "autonomy")
    delta = payload.get("delta", 0.0)
    
    new_axes = {**state.axes}
    new_axes[axis] = max(0.0, min(1.0, new_axes.get(axis, 0.5) + delta))
    
    new_lineage = {**state.lineage, "mutations": list(state.lineage.get("mutations", [])) + [{
        "axis": axis,
        "delta": delta,
        "timestamp": event.timestamp,
        "causation_id": event.causal.causation_id
    }]}
    
    return IdentityState(
        axes=MappingProxyType(new_axes),
        genome=state.genome,
        lineage=MappingProxyType(new_lineage),
        version=state.version + 1
    )


def reduce_genome_evolved(state: IdentityState, event: ImmutableEvent) -> IdentityState:
    """Reduce genome_evolved event"""
    payload = event.get_payload_dict()
    gene = payload.get("gene", "unknown")
    value = payload.get("value", 0.5)
    
    new_genome = {**state.genome, gene: {
        "value": value,
        "evolved_at": event.timestamp,
        "selection_pressure": payload.get("selection_pressure", 0.5)
    }}
    
    return IdentityState(
        axes=state.axes,
        genome=MappingProxyType(new_genome),
        lineage=state.lineage,
        version=state.version + 1
    )


# Event type to reducer mapping

COGNITIVE_REDUCERS: Dict[str, Callable] = {
    EventTypes.BELIEF_CREATED: reduce_belief_created,
    EventTypes.BELIEF_UPDATED: reduce_belief_updated,
    EventTypes.CONTRADICTION_DETECTED: reduce_contradiction_detected,
    EventTypes.PRESSURE_ACCUMULATED: reduce_pressure_accumulated,
}

EXECUTION_REDUCERS: Dict[str, Callable] = {
    EventTypes.GOAL_CREATED: reduce_goal_created,
    EventTypes.GOAL_EXECUTED: reduce_goal_executed,
    EventTypes.GOAL_COMPLETED: reduce_goal_completed,
}

IDENTITY_REDUCERS: Dict[str, Callable] = {
    EventTypes.IDENTITY_MUTATED: reduce_identity_mutated,
    EventTypes.GENOME_EVOLVED: reduce_genome_evolved,
}


# Main reducer functions

def reduce(state: Any, event: ImmutableEvent) -> Any:
    """
    Main reduce function.
    
    Routes event to appropriate reducer based on event type.
    """
    reducers = _get_reducers_for_category(event.category)
    reducer = reducers.get(event.event_type)
    
    if reducer is None:
        return state
    
    return reducer(state, event)


def _get_reducers_for_category(category: str) -> Dict[str, Callable]:
    """Get reducers for category"""
    if category == "cognitive":
        return COGNITIVE_REDUCERS
    elif category == "execution":
        return EXECUTION_REDUCERS
    elif category == "identity":
        return IDENTITY_REDUCERS
    return {}


def reduce_stream(state: Any, events: Tuple[ImmutableEvent, ...]) -> Any:
    """
    Reduce stream of events.
    
    Pure function: same events always produce same state.
    """
    current = state
    for event in events:
        current = reduce(current, event)
    return current


def materialize_from_log(log: EventLog, stream_id: str) -> Any:
    """
    Materialize state from event log.
    
    Takes all events from stream and reduces them to state.
    """
    events = log.get_stream(stream_id)
    
    # Get initial state for stream type
    if stream_id == StreamIds.COGNITION:
        state = CognitiveState(
            beliefs=MappingProxyType({}),
            contradictions=MappingProxyType({}),
            pressures=MappingProxyType({}),
            version=0
        )
    elif stream_id == StreamIds.EXECUTION:
        state = ExecutionState(
            goals=MappingProxyType({}),
            actions=MappingProxyType({}),
            results=MappingProxyType({}),
            version=0
        )
    elif stream_id == StreamIds.IDENTITY:
        state = IdentityState(
            axes=MappingProxyType({"autonomy": 0.5, "curiosity": 0.5, "stability": 0.5, "coherence": 0.5}),
            genome=MappingProxyType({}),
            lineage=MappingProxyType({"mutations": []}),
            version=0
        )
    else:
        state = CognitiveState(
            beliefs=MappingProxyType({}),
            contradictions=MappingProxyType({}),
            pressures=MappingProxyType({}),
            version=0
        )
    
    return reduce_stream(state, events)


def replay_verification(log: EventLog, stream_id: str, expected_hash: str) -> bool:
    """
    Verify that replay produces expected state.
    
    This is the key invariant:
        replay(events) == expected_state
    """
    state = materialize_from_log(log, stream_id)
    
    # Compute state hash
    data = dict(state.data) if hasattr(state, 'data') else {
        "version": state.version,
        "beliefs": dict(state.beliefs) if hasattr(state, 'beliefs') else {},
        "contradictions": dict(state.contradictions) if hasattr(state, 'contradictions') else {}
    }
    
    actual_hash = hashlib.sha256(
        json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
    
    return actual_hash == expected_hash


# Initial states

def create_initial_cognitive_state() -> CognitiveState:
    return CognitiveState(
        beliefs=MappingProxyType({}),
        contradictions=MappingProxyType({}),
        pressures=MappingProxyType({}),
        version=0
    )


def create_initial_execution_state() -> ExecutionState:
    return ExecutionState(
        goals=MappingProxyType({}),
        actions=MappingProxyType({}),
        results=MappingProxyType({}),
        version=0
    )


def create_initial_identity_state() -> IdentityState:
    return IdentityState(
        axes=MappingProxyType({
            "autonomy": 0.5,
            "curiosity": 0.5,
            "stability": 0.5,
            "coherence": 0.5
        }),
        genome=MappingProxyType({}),
        lineage=MappingProxyType({"mutations": []}),
        version=0
    )