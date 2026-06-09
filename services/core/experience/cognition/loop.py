"""
Cognitive Loop - The closed loop of cognition.

Stage: Cognitive Architecture

This is the core of pseudo-AGI architecture:

Execution → Belief → Contradiction → Pressure → Planning → Execution

The loop:
1. Execution produces outcomes
2. Outcomes update beliefs (world model)
3. Contradictions emerge from belief conflicts
4. Pressure accumulates from unresolved contradictions
5. Goals compete in attention economy
6. Planner synthesizes strategy
7. Strategy drives execution
8. Cycle repeats

This creates a closed loop where:
- Cognition influences execution
- Execution changes world model
- World model changes planning
- Planning changes identity
- Identity constrains future cognition
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

# Import all cognitive components
from .memory import (
    MemoryState, Belief, EpisodicMemory, Procedure, Reflection,
    create_empty_memory
)
from .world import (
    WorldModel, Entity, Relation, CausalLink,
    create_empty_world
)
from .contradiction import (
    ContradictionState, Contradiction, Pressure, ResolutionAttempt,
    create_empty_contradiction_state,
    detect_belief_belief_conflict, compute_pressure_intensity
)
from .goals import (
    GoalMarket, Goal, GoalStatus, create_goal_market
)
from .self import (
    SelfModel, Capability, Limitation, Strategy, ResourceModel,
    create_initial_self_model
)


@dataclass(frozen=True)
class CognitiveEvent:
    """
    An event in the cognitive loop.
    
    These are the inputs that drive cognition.
    """
    event_type: str  # execution_outcome, evidence_received, goal_created, etc.
    event_id: str  # Unique identifier
    source: str  # Where this came from
    timestamp: str
    content: Dict[str, Any]  # Event data
    implications: Tuple[str, ...]  # What this affects
    
    def affects_beliefs(self) -> bool:
        return "belief" in self.event_type or "evidence" in self.event_type
    
    def affects_goals(self) -> bool:
        return "goal" in self.event_type
    
    def affects_self(self) -> bool:
        return "self" in self.event_type or "capability" in self.event_type


@dataclass(frozen=True)
class CognitiveState:
    """
    Complete cognitive state.
    
    This is the complete state of the cognitive system.
    All components are here.
    """
    # Memory systems
    memory: MemoryState
    world: WorldModel
    
    # Contradiction detection
    contradictions: ContradictionState
    
    # Goal economy
    goals: GoalMarket
    
    # Self model
    self_model: SelfModel
    
    # Loop state
    cycle_count: int
    total_pressure: float
    active_intent: Optional[str]  # Current primary intention
    version: int
    last_cycle: str
    
    @staticmethod
    def compute_hash(state: 'CognitiveState') -> str:
        data = {
            "version": state.version,
            "cycle_count": state.cycle_count,
            "total_pressure": state.total_pressure,
            "memory_version": state.memory.version,
            "world_version": state.world.version,
            "contradiction_count": len(state.contradictions.contradictions),
            "goal_count": len(state.goals.goals),
            "self_consistency": state.self_model.self_consistency_score
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def is_pressured(self) -> bool:
        return self.total_pressure > 0.5
    
    def has_critical_contradictions(self) -> bool:
        return len(self.contradictions.get_escalated()) > 0
    
    def get_primary_goals(self) -> List[Goal]:
        """Get highest priority active goals"""
        active = self.goals.get_active_goals()
        return sorted(active, key=lambda g: g.effective_priority(), reverse=True)[:5]


def create_initial_cognitive_state() -> CognitiveState:
    """Create initial cognitive state"""
    return CognitiveState(
        memory=create_empty_memory(),
        world=create_empty_world(),
        contradictions=create_empty_contradiction_state(),
        goals=create_goal_market(),
        self_model=create_initial_self_model(),
        cycle_count=0,
        total_pressure=0.0,
        active_intent=None,
        version=0,
        last_cycle=datetime.utcnow().isoformat()
    )


# Cognitive Reducers - Pure functions for state transitions

def reduce_execution_outcome(
    state: CognitiveState,
    event: CognitiveEvent
) -> CognitiveState:
    """
    Reduce execution outcome into cognitive state.
    
    Updates:
    - Memory (episodes, beliefs)
    - World model (entities, causal links)
    - Self model (capabilities, failures)
    """
    content = event.content
    outcome = content.get("outcome", "unknown")
    action = content.get("action", "")
    result = content.get("result", {})
    
    # Update episodic memory
    episode = EpisodicMemory(
        episode_id=event.event_id,
        context=(
            ("action", action),
            ("outcome", outcome),
            ("timestamp", event.timestamp)
        ),
        actions=(action,),
        outcome=outcome,
        emotional_valence=1.0 if outcome == "success" else -0.5,
        lessons=(),
        timestamp=event.timestamp,
        duration_ms=content.get("duration_ms", 0),
        related_goals=content.get("goal_ids", ())
    )
    
    new_memory = state.memory.with_episode(episode.episode_id, episode)
    
    # Update self model capabilities
    cap_id = content.get("capability_id", "execution")
    if cap_id in state.self_model.capabilities:
        cap = state.self_model.capabilities[cap_id]
        if outcome == "success":
            updated_cap = Capability(
                capability_id=cap.capability_id,
                name=cap.name,
                description=cap.description,
                proficiency=min(1.0, cap.proficiency + 0.01),
                reliability=min(1.0, cap.reliability + 0.005),
                learn_speed=cap.learn_speed,
                resource_requirement=cap.resource_requirement,
                success_count=cap.success_count + 1,
                failure_count=cap.failure_count,
                last_used=event.timestamp
            )
        else:
            updated_cap = Capability(
                capability_id=cap.capability_id,
                name=cap.name,
                description=cap.description,
                proficiency=max(0.1, cap.proficiency - 0.02),
                reliability=max(0.1, cap.reliability - 0.01),
                learn_speed=cap.learn_speed,
                resource_requirement=cap.resource_requirement,
                success_count=cap.success_count,
                failure_count=cap.failure_count + 1,
                last_used=event.timestamp
            )
        new_self = state.self_model.with_capability(updated_cap)
    else:
        new_self = state.self_model
    
    # Update world model with causal link
    if outcome == "success":
        link = CausalLink(
            cause_id=action,
            effect_id="success",
            mechanism=content.get("mechanism", "direct"),
            strength=0.8,
            conditions=(),
            evidence=(event.event_id,),
            created_at=event.timestamp
        )
        new_world = state.world.with_causal_link(link)
    else:
        new_world = state.world
    
    return CognitiveState(
        memory=new_memory,
        world=new_world,
        contradictions=state.contradictions,
        goals=state.goals,
        self_model=new_self,
        cycle_count=state.cycle_count + 1,
        total_pressure=state.total_pressure,
        active_intent=state.active_intent,
        version=state.version + 1,
        last_cycle=event.timestamp
    )


def reduce_evidence_received(
    state: CognitiveState,
    event: CognitiveEvent
) -> CognitiveState:
    """
    Reduce new evidence into belief state.
    
    Updates:
    - Memory (beliefs)
    - World model (assertions)
    - Contradictions (detects conflicts)
    """
    content = event.content
    proposition = content.get("proposition", "")
    evidence_strength = content.get("strength", 0.5)
    source = content.get("source", "unknown")
    
    # Create or update belief
    belief_id = content.get("belief_id", event.event_id)
    
    # Check for contradictions with existing beliefs
    for existing_id, existing in state.memory.beliefs.items():
        if existing.proposition == proposition:
            # Duplicate - update confidence
            if evidence_strength > existing.confidence:
                new_belief = existing.update_confidence(
                    (existing.confidence + evidence_strength) / 2
                )
            else:
                new_belief = existing.update_confidence(
                    (existing.confidence * 0.7 + evidence_strength * 0.3)
                )
            new_memory = state.memory.with_belief(existing_id, new_belief)
            return state
        
        # Check for contradictory evidence
        if existing.proposition != proposition:
            # Check semantic contradiction
            conflict_intensity = detect_belief_belief_conflict(
                existing_id,
                existing.confidence,
                belief_id,
                evidence_strength,
                mutual_exclusion=content.get("contradicts", False)
            )
            
            if conflict_intensity and conflict_intensity > 0.6:
                # Create contradiction
                contradiction = Contradiction(
                    contradiction_id=f"conflict_{existing_id}_{belief_id}",
                    type="belief_belief",
                    participants=frozenset([existing_id, belief_id]),
                    description=f"Belief '{proposition}' conflicts with '{existing.proposition}'",
                    intensity=conflict_intensity,
                    detected_at=event.timestamp,
                    resolution_state="unresolved",
                    evidence=(existing_id, belief_id),
                    resolution_attempts=()
                )
                
                new_contradictions = {
                    **state.contradictions.contradictions,
                    contradiction.contradiction_id: contradiction
                }
                
                from types import MappingProxyType
                new_contr_state = ContradictionState(
                    contradictions=MappingProxyType(new_contradictions),
                    pressures=state.contradictions.pressures,
                    resolution_attempts=state.contradictions.resolution_attempts,
                    total_pressure=ContradictionState.compute_total_pressure(state.contradictions) + conflict_intensity,
                    unresolved_count=state.contradictions.unresolved_count + 1,
                    escalated_count=state.contradictions.escalated_count,
                    version=state.contradictions.version + 1
                )
                
                new_memory = state.memory.with_belief(belief_id, Belief(
                    belief_id=belief_id,
                    proposition=proposition,
                    confidence=evidence_strength,
                    evidence=(),
                    counter_evidence=(existing_id,),
                    sources=(source,),
                    created_at=event.timestamp,
                    last_updated=event.timestamp,
                    stability=0.5,
                    revision_count=0
                ))
                
                return CognitiveState(
                    memory=new_memory,
                    world=state.world,
                    contradictions=new_contr_state,
                    goals=state.goals,
                    self_model=state.self_model,
                    cycle_count=state.cycle_count + 1,
                    total_pressure=new_contr_state.total_pressure,
                    active_intent=state.active_intent,
                    version=state.version + 1,
                    last_cycle=event.timestamp
                )
    
    # No conflict - add new belief
    new_belief = Belief(
        belief_id=belief_id,
        proposition=proposition,
        confidence=evidence_strength,
        evidence=(),
        counter_evidence=(),
        sources=(source,),
        created_at=event.timestamp,
        last_updated=event.timestamp,
        stability=0.5,
        revision_count=0
    )
    
    new_memory = state.memory.with_belief(belief_id, new_belief)
    
    return CognitiveState(
        memory=new_memory,
        world=state.world,
        contradictions=state.contradictions,
        goals=state.goals,
        self_model=state.self_model,
        cycle_count=state.cycle_count + 1,
        total_pressure=state.total_pressure,
        active_intent=state.active_intent,
        version=state.version + 1,
        last_cycle=event.timestamp
    )


def reduce_pressure_update(
    state: CognitiveState,
    event: CognitiveEvent
) -> CognitiveState:
    """
    Update pressures from contradictions.
    
    Computes pressure from unresolved contradictions
    and routes to appropriate cognitive systems.
    """
    content = event.content
    contradiction_id = content.get("contradiction_id")
    
    if not contradiction_id or contradiction_id not in state.contradictions.contradictions:
        return state
    
    contradiction = state.contradictions.contradictions[contradiction_id]
    
    # Compute pressure intensity
    age_hours = (datetime.utcnow() - datetime.fromisoformat(contradiction.detected_at)).total_seconds() / 3600
    pressure_intensity = compute_pressure_intensity(
        contradiction,
        age_hours,
        len(contradiction.resolution_attempts)
    )
    
    # Create pressure
    pressure = Pressure(
        pressure_id=f"pressure_{contradiction_id}",
        source_contradiction_id=contradiction_id,
        type=content.get("pressure_type", "cognitive"),
        intensity=pressure_intensity,
        components=frozenset(content.get("components", ["planning", "execution"])),
        accumulated_at=event.timestamp,
        decay_rate=0.1,
        urgency=pressure_intensity * (1 + len(contradiction.resolution_attempts) * 0.2)
    )
    
    from types import MappingProxyType
    
    new_pressures = {**dict(state.contradictions.pressures), pressure.pressure_id: pressure}
    total_pressure = sum(p.intensity for p in new_pressures.values())
    
    new_contr_state = ContradictionState(
        contradictions=state.contradictions.contradictions,
        pressures=MappingProxyType(new_pressures),
        resolution_attempts=state.contradictions.resolution_attempts,
        total_pressure=total_pressure,
        unresolved_count=state.contradictions.unresolved_count,
        escalated_count=state.contradictions.escalated_count,
        version=state.contradictions.version + 1
    )
    
    return CognitiveState(
        memory=state.memory,
        world=state.world,
        contradictions=new_contr_state,
        goals=state.goals,
        self_model=state.self_model,
        cycle_count=state.cycle_count + 1,
        total_pressure=total_pressure,
        active_intent=state.active_intent if not pressure.is_critical() else contradiction_id,
        version=state.version + 1,
        last_cycle=event.timestamp
    )


# Main reducer

COGNITIVE_REDUCERS = {
    "execution_outcome": reduce_execution_outcome,
    "evidence_received": reduce_evidence_received,
    "pressure_update": reduce_pressure_update,
}


def cognitive_reduce(state: CognitiveState, event: CognitiveEvent) -> CognitiveState:
    """
    Main cognitive reducer.
    
    Pure function: cognitive_state + cognitive_event → new_cognitive_state
    
    This is the heart of the cognitive loop.
    """
    reducer = COGNITIVE_REDUCERS.get(event.event_type)
    
    if reducer is None:
        return state
    
    return reducer(state, event)


def cognitive_reduce_sequence(state: CognitiveState, events: Tuple[CognitiveEvent, ...]) -> CognitiveState:
    """Apply sequence of cognitive events"""
    current = state
    for event in events:
        current = cognitive_reduce(current, event)
    return current


def compute_system_pressure(state: CognitiveState) -> float:
    """
    Compute total system pressure.
    
    Combines:
    - Contradiction pressure
    - Resource strain
    - Goal urgency
    """
    contr_pressure = state.total_pressure
    
    resource_strain = 0.0
    if state.self_model.resource_model.is_strained():
        resource_strain = 0.3
    
    goal_pressure = len(state.goals.get_overdue_goals()) * 0.1
    
    return min(1.0, contr_pressure + resource_strain + goal_pressure)


def generate_intent(state: CognitiveState) -> Optional[str]:
    """
    Generate primary intent from current state.
    
    Uses:
    - Highest priority goals
    - Most urgent pressures
    - Self model capabilities
    
    Returns intent description or None.
    """
    # Check for critical contradictions
    if state.has_critical_contradictions():
        return "resolve_contradiction"
    
    # Check for high pressure
    if state.is_pressured():
        return "manage_pressure"
    
    # Get primary goal
    primary = state.get_primary_goals()
    if primary:
        return f"execute_goal_{primary[0].goal_id}"
    
    # Check self model
    uncertain = state.self_model.get_uncertain_regions()
    if uncertain:
        return "explore_uncertainty"
    
    return "maintain_operations"


def assess_cognitive_health(state: CognitiveState) -> Dict[str, Any]:
    """
    Assess overall cognitive health.
    
    Returns diagnostic information.
    """
    return {
        "cycle_count": state.cycle_count,
        "total_pressure": state.total_pressure,
        "pressure_level": "critical" if state.total_pressure > 0.8 
                         else "high" if state.total_pressure > 0.5
                         else "moderate" if state.total_pressure > 0.2
                         else "low",
        "contradiction_count": len(state.contradictions.contradictions),
        "unresolved_contradictions": state.contradictions.unresolved_count,
        "escalated_contradictions": state.contradictions.escalated_count,
        "active_goals": len(state.goals.get_active_goals()),
        "self_consistency": state.self_model.self_consistency_score,
        "memory_beliefs": len(state.memory.beliefs),
        "world_entities": len(state.world.entities),
        "capable_actions": len(state.self_model.get_capable_actions()),
        "recommended_action": generate_intent(state)
    }