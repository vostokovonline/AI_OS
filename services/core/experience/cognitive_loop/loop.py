"""
Cognitive Loop - Closed feedback loop for tension-driven cognition.

This is the PRIMARY cognitive mechanism.
NOT execution-driven, NOT goal-driven - TENSION-DRIVEN.

The loop:
1. INPUT: Raw events enter the system
2. FILTER: Attention module filters signal from noise
3. ASSIMILATE: Events update internal state
4. DETECT: Tensions emerge from state contradictions
5. SALIENCE: Priority computed from tension urgency
6. GENERATE: Goals emerge from salient tensions
7. PLAN: Execution planned to resolve tensions
8. ACT: Execution acts on environment
9. FEEDBACK: Results modify internal state
10. ADAPT: Loop adjusts based on outcomes

Key insight:
- Goals are NOT inputs to the system
- Goals are OUTPUTS of tension resolution
- The system has "wants" that emerge from internal state
"""

from typing import Dict, Any, Optional, List, Tuple, Set, FrozenSet, Callable
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from enum import Enum
import hashlib

# Import cognitive loop components
from cognitive_loop.tension import (
    TensionType,
    TensionLifecycle,
    Tension,
    SalienceMap,
    TensionResolutionGoal,
    TensionState as TensionSystemState,
    create_initial_tension_state,
    detect_contradiction_tension,
    detect_vacuum_tension,
    detect_pressure_tension,
    compute_salience,
    generate_tension_goal,
    generate_goals_from_salience,
    add_tension,
    resolve_tension,
    escalate_tension
)

from cognitive_loop.attention import (
    InputSource,
    FilterConfig,
    FilteredInputs,
    filter_inputs,
    InputSignal
)

from cognitive_loop.trajectory import (
    TrendDirection,
    StateVector,
    TrajectoryState,
    create_state_vector,
    add_vector_to_trajectory,
    detect_divergence,
    project_trajectory,
    complete_segment
)

from cognitive_loop.emergence import (
    EmergenceState,
    EmergentProperty,
    ComponentInteraction,
    detect_interaction,
    detect_emergence,
    add_emergent_property
)


class LoopPhase(Enum):
    """Phase of cognitive loop execution"""
    INPUT = "input"              # Receiving raw events
    FILTER = "filter"            # Attention filtering
    ASSIMILATE = "assimilate"    # Updating internal state
    DETECT = "detect"            # Detecting tensions
    SALIENCE = "salience"        # Computing salience
    GENERATE = "generate"        # Generating goals
    PLAN = "plan"                # Planning execution
    ACT = "act"                  # Executing actions
    FEEDBACK = "feedback"       # Processing feedback
    ADAPT = "adapt"              # Adapting loop parameters


@dataclass(frozen=True)
class CognitiveLoopConfig:
    """
    Configuration for cognitive loop behavior.
    
    These values are identity-driven.
    """
    attention_budget: float = 1.0
    max_tensions: int = 50
    tension_threshold: float = 0.3  # Below this, tensions are latent
    salience_threshold: float = 0.5  # Below this, not salient
    goal_generation_rate: float = 1.0  # Goals per cycle
    adaptation_rate: float = 0.1  # How fast loop adapts
    
    @staticmethod
    def from_identity(
        autonomy: float,
        curiosity: float,
        stability: float
    ) -> 'CognitiveLoopConfig':
        """Create config from identity parameters"""
        return CognitiveLoopConfig(
            attention_budget=1.0 - (autonomy * 0.2),  # Higher autonomy = more focus
            max_tensions=int(50 - (stability * 20)),  # Higher stability = fewer tensions
            tension_threshold=0.3 - (curiosity * 0.1),  # Higher curiosity = lower threshold
            salience_threshold=0.5 - (autonomy * 0.1),  # Higher autonomy = higher threshold
            goal_generation_rate=1.0 + (curiosity * 0.5),  # More goals for curious
            adaptation_rate=0.1 + (autonomy * 0.05)  # Faster adaptation for autonomous
        )


@dataclass(frozen=True)
class CognitiveLoopState:
    """
    Complete state of the cognitive loop.
    
    This is what gets reduced over time.
    NOT mutable - always produces new state.
    """
    tension_state: TensionSystemState
    trajectory_state: TrajectoryState
    emergence_state: EmergenceState
    filter_config: FilterConfig
    loop_config: CognitiveLoopConfig
    current_phase: str  # LoopPhase value
    cycle_count: int
    last_cycle_time: str
    adaptation_metrics: MappingProxyType  # type: ignore
    
    def __post_init__(self):
        for attr in ('adaptation_metrics',):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
    
    @staticmethod
    def initial(
        filter_config: Optional[FilterConfig] = None,
        loop_config: Optional[CognitiveLoopConfig] = None
    ) -> 'CognitiveLoopState':
        """Create initial cognitive loop state"""
        return CognitiveLoopState(
            tension_state=create_initial_tension_state(),
            trajectory_state=TrajectoryState.empty(),
            emergence_state=EmergenceState.empty(),
            filter_config=filter_config or FilterConfig(),
            loop_config=loop_config or CognitiveLoopConfig(),
            current_phase=LoopPhase.INPUT.value,
            cycle_count=0,
            last_cycle_time=datetime.utcnow().isoformat(),
            adaptation_metrics=MappingProxyType({
                'cycles_completed': 0,
                'goals_generated': 0,
                'tensions_resolved': 0,
                'emergence_detected': 0
            })
        )


@dataclass(frozen=True)
class LoopEvent:
    """
    An event in the cognitive loop lifecycle.
    
    These are DIFFERENT from domain events.
    Loop events track loop execution, not domain changes.
    """
    event_id: str
    cycle: int
    phase: str  # LoopPhase value
    timestamp: str
    input_summary: str
    tension_count: int
    goal_generated: bool
    action_taken: Optional[str]
    feedback_received: bool


# Core Loop Functions

def execute_input_phase(
    state: CognitiveLoopState,
    raw_inputs: Tuple[InputSource, ...]
) -> Tuple[CognitiveLoopState, FilteredInputs]:
    """
    INPUT phase: Receive raw events.
    
    Returns:
    - Updated state
    - Filtered inputs (passed through attention)
    """
    # Filter inputs using attention
    filtered = filter_inputs(
        raw_inputs,
        state.filter_config,
        None  # No seen sources for now
    )
    
    return state, filtered


def execute_detect_phase(
    state: CognitiveLoopState,
    filtered_inputs: FilteredInputs
) -> CognitiveLoopState:
    """
    DETECT phase: Detect tensions from inputs and state.
    
    This is where tensions EMERGE from accumulated evidence.
    NOT synthetic pressure - emergent from contradictions.
    """
    # Analyze inputs for tension sources
    for inp in filtered_inputs.inputs:
        # Check for contradiction signals
        if inp.relevance_score > 0.7 and inp.novelty_score > 0.5:
            # High relevance + novelty = potential tension
            tension = detect_pressure_tension(
                source=inp.source_id,
                intensity=inp.relevance_score * 0.5,
                pressure_type="cognitive"
            )
            
            state = CognitiveLoopState(
                tension_state=add_tension(state.tension_state, tension),
                trajectory_state=state.trajectory_state,
                emergence_state=state.emergence_state,
                filter_config=state.filter_config,
                loop_config=state.loop_config,
                current_phase=LoopPhase.DETECT.value,
                cycle_count=state.cycle_count,
                last_cycle_time=datetime.utcnow().isoformat(),
                adaptation_metrics=state.adaptation_metrics
            )
    
    return state


def execute_salience_phase(
    state: CognitiveLoopState
) -> CognitiveLoopState:
    """
    SALIENCE phase: Compute which tensions demand attention.
    
    This is called internally - salience is computed in tension_state.
    """
    # Salience is already computed in tension_state.salience
    # Just return current state
    return state


def execute_generate_phase(
    state: CognitiveLoopState
) -> Tuple[CognitiveLoopState, Tuple[TensionResolutionGoal, ...]]:
    """
    GENERATE phase: Generate goals from salient tensions.
    
    Key distinction:
    - External goals: "do X because asked"
    - Tension goals: "do Y because tension demands resolution"
    """
    # Generate goals from salient tensions
    goals = generate_goals_from_salience(state.tension_state.salience)
    
    # Update adaptation metrics
    new_metrics = {**state.adaptation_metrics}
    new_metrics['goals_generated'] = new_metrics.get('goals_generated', 0) + len(goals)
    
    return state, tuple(goals)


def execute_trajectory_phase(
    state: CognitiveLoopState
) -> CognitiveLoopState:
    """
    TRAJECTORY phase: Update trajectory tracking.
    
    Track where we've been and project where we're going.
    """
    # Create state vector from current tension state
    state_dict = {
        'tension_energy': state.tension_state.total_tension_energy,
        'attention_allocated': sum(state.tension_state.salience.allocated_attention.values()),
        'goal_count': len(state.tension_state.generated_goals),
        'belief_coherence': 0.5,  # Placeholder
        'identity_stability': 0.7,  # Placeholder
        'knowledge_coverage': 0.4,  # Placeholder
        'trajectory_direction': TrendDirection.UNKNOWN.value
    }
    
    vector = create_state_vector(state_dict, state.cycle_count)
    
    # Add to trajectory
    new_trajectory = add_vector_to_trajectory(state.trajectory_state, vector)
    
    # Check for divergence
    divergence, magnitude = detect_divergence(new_trajectory, TrendDirection.STABLE.value)
    
    return CognitiveLoopState(
        tension_state=state.tension_state,
        trajectory_state=new_trajectory,
        emergence_state=state.emergence_state,
        filter_config=state.filter_config,
        loop_config=state.loop_config,
        current_phase=LoopPhase.FEEDBACK.value,
        cycle_count=state.cycle_count,
        last_cycle_time=datetime.utcnow().isoformat(),
        adaptation_metrics=state.adaptation_metrics,
    )


def execute_feedback_phase(
    state: CognitiveLoopState,
    action_result: Optional[Dict[str, Any]]
) -> CognitiveLoopState:
    """
    FEEDBACK phase: Process results from actions.
    
    Update tensions based on whether actions resolved them.
    """
    if action_result is None:
        return state
    
    # Check if action resolved a tension
    resolved_tension_id = action_result.get('resolved_tension_id')
    if resolved_tension_id:
        state = CognitiveLoopState(
            tension_state=resolve_tension(
                state.tension_state,
                resolved_tension_id,
                action_result.get('resolution_type', 'action_resolved')
            ),
            trajectory_state=state.trajectory_state,
            emergence_state=state.emergence_state,
            filter_config=state.filter_config,
            loop_config=state.loop_config,
            current_phase=LoopPhase.ADAPT.value,
            cycle_count=state.cycle_count + 1,
            last_cycle_time=datetime.utcnow().isoformat(),
            adaptation_metrics=state.adaptation_metrics
        )
    
    return state


def execute_adapt_phase(
    state: CognitiveLoopState,
    adaptation_data: Optional[Dict[str, Any]] = None
) -> CognitiveLoopState:
    """
    ADAPT phase: Adjust loop parameters based on outcomes.
    
    This is where the loop learns from experience.
    """
    if adaptation_data is None:
        return state
    
    # Update adaptation metrics
    new_metrics = {**state.adaptation_metrics}
    if adaptation_data.get('tension_resolved'):
        new_metrics['tensions_resolved'] = new_metrics.get('tensions_resolved', 0) + 1
    if adaptation_data.get('emergence_detected'):
        new_metrics['emergence_detected'] = new_metrics.get('emergence_detected', 0) + 1
    
    return CognitiveLoopState(
        tension_state=state.tension_state,
        trajectory_state=state.trajectory_state,
        emergence_state=state.emergence_state,
        filter_config=state.filter_config,
        loop_config=state.loop_config,
        current_phase=LoopPhase.INPUT.value,  # Reset to start
        cycle_count=state.cycle_count + 1,
        last_cycle_time=datetime.utcnow().isoformat(),
        adaptation_metrics=MappingProxyType(new_metrics)
    )


def execute_full_cycle(
    state: CognitiveLoopState,
    raw_inputs: Tuple[InputSource, ...],
    action_result: Optional[Dict[str, Any]] = None,
    adaptation_data: Optional[Dict[str, Any]] = None
) -> Tuple[CognitiveLoopState, Tuple[TensionResolutionGoal, ...], Optional[LoopEvent]]:
    """
    Execute one full cycle of the cognitive loop.
    
    Cycle phases:
    1. INPUT - Receive raw events
    2. FILTER - Attention filtering
    3. DETECT - Detect tensions
    4. SALIENCE - Compute priority
    5. GENERATE - Generate goals
    6. TRAJECTORY - Update trajectory
    7. FEEDBACK - Process results
    8. ADAPT - Adjust parameters
    
    Returns:
    - Updated state
    - Generated goals (from tension)
    - Loop event (for logging/audit)
    """
    # Phase 1-2: Input and Filter
    state, filtered = execute_input_phase(state, raw_inputs)
    
    # Phase 3: Detect tensions
    state = execute_detect_phase(state, filtered)
    
    # Phase 4: Salience (already computed in tension_state)
    state = execute_salience_phase(state)
    
    # Phase 5: Generate goals
    state, goals = execute_generate_phase(state)
    
    # Phase 6: Trajectory
    state = execute_trajectory_phase(state)
    
    # Phase 7: Feedback
    state = execute_feedback_phase(state, action_result)
    
    # Phase 8: Adapt
    state = execute_adapt_phase(state, adaptation_data)
    
    # Create loop event
    event = LoopEvent(
        event_id=f"loop_{state.cycle_count}_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]}",
        cycle=state.cycle_count,
        phase=state.current_phase,
        timestamp=datetime.utcnow().isoformat(),
        input_summary=f"{filtered.total_considered} inputs, {filtered.discarded_count} discarded",
        tension_count=len(state.tension_state.tensions),
        goal_generated=len(goals) > 0,
        action_taken=action_result.get('action') if action_result else None,
        feedback_received=action_result is not None
    )
    
    return state, goals, event


# Trajectory Projection and Emergence Integration

def integrate_emergence(
    state: CognitiveLoopState,
    interaction: Optional[ComponentInteraction] = None
) -> CognitiveLoopState:
    """
    Integrate emergence detection into cognitive loop.
    
    When tensions interact, emergent properties may arise.
    """
    if interaction is None:
        return state
    
    # Detect emergence
    emergent = detect_emergence(state.emergence_state, interaction)
    
    if emergent:
        new_emergence = add_emergent_property(state.emergence_state, emergent)
        
        return CognitiveLoopState(
            tension_state=state.tension_state,
            trajectory_state=state.trajectory_state,
            emergence_state=new_emergence,
            filter_config=state.filter_config,
            loop_config=state.loop_config,
            current_phase=state.current_phase,
            cycle_count=state.cycle_count,
            last_cycle_time=state.last_cycle_time,
            adaptation_metrics=state.adaptation_metrics
        )
    
    return state


def project_future_states(
    state: CognitiveLoopState,
    steps: int = 5
) -> Tuple[StateVector, ...]:
    """
    Project future states based on current trajectory.
    
    This enables prediction and planning.
    """
    if state.trajectory_state.current_vector is None:
        return ()
    
    # Determine trend from tension energy
    if state.tension_state.total_tension_energy > 0.6:
        trend = TrendDirection.DEGRADING
    elif state.tension_state.total_tension_energy < 0.4:
        trend = TrendDirection.IMPROVING
    else:
        trend = TrendDirection.STABLE
    
    return project_trajectory(state.trajectory_state.current_vector, trend, steps)


# State Access Functions

def get_top_tensions(state: CognitiveLoopState, n: int = 5) -> List[Tension]:
    """Get top N most urgent tensions"""
    return state.tension_state.salience.get_top_tensions(n)


def get_active_goals(state: CognitiveLoopState) -> List[TensionResolutionGoal]:
    """Get all active tension-driven goals"""
    return list(state.tension_state.generated_goals.values())


def get_trajectory_summary(state: CognitiveLoopState) -> Dict[str, Any]:
    """Get summary of trajectory state"""
    return {
        'current_energy': state.trajectory_state.current_vector.tension_energy if state.trajectory_state.current_vector else 0.0,
        'segment_count': len(state.trajectory_state.history),
        'divergence_detected': state.trajectory_state.divergence_detected,
        'divergence_magnitude': state.trajectory_state.divergence_magnitude,
        'projected_states': len(state.trajectory_state.projected_vectors)
    }


def get_loop_metrics(state: CognitiveLoopState) -> Dict[str, Any]:
    """Get cognitive loop metrics"""
    return {
        'cycle_count': state.cycle_count,
        'total_tensions': len(state.tension_state.tensions),
        'active_tensions': sum(1 for t in state.tension_state.tensions.values() if t.state != "resolved"),
        'goals_generated': state.adaptation_metrics.get('goals_generated', 0),
        'tensions_resolved': state.adaptation_metrics.get('tensions_resolved', 0),
        'emergence_detected': state.adaptation_metrics.get('emergence_detected', 0),
        'total_tension_energy': state.tension_state.total_tension_energy,
        'dominant_tension_type': state.tension_state.dominant_tension_type
    }