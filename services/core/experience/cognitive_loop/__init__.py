"""
Cognitive Loop - Primary cognitive mechanism for tension-driven cognition.

The cognitive loop is the PRIMARY mechanism, not execution.
Goals emerge from tensions, not from external commands.

Components:
- tension/: Tension types, salience, resolution goals
- attention/: Input filtering, signal/noise separation
- trajectory/: History tracking, state projection
- emergence/: Novel pattern detection
- loop.py: Closed feedback loop orchestration

Usage:
    from cognitive_loop import (
        CognitiveLoopState,
        execute_full_cycle,
        get_top_tensions,
        get_loop_metrics
    )

    # Create initial state
    state = CognitiveLoopState.initial()

    # Execute cycle
    state, goals, event = execute_full_cycle(state, raw_inputs)

    # Get urgent tensions
    tensions = get_top_tensions(state)
"""

from cognitive_loop.loop import (
    LoopPhase,
    CognitiveLoopConfig,
    CognitiveLoopState,
    LoopEvent,
    execute_input_phase,
    execute_detect_phase,
    execute_salience_phase,
    execute_generate_phase,
    execute_trajectory_phase,
    execute_feedback_phase,
    execute_adapt_phase,
    execute_full_cycle,
    integrate_emergence,
    project_future_states,
    get_top_tensions,
    get_active_goals,
    get_trajectory_summary,
    get_loop_metrics
)

from cognitive_loop.tension import (
    TensionType,
    TensionLifecycle,
    Tension,
    SalienceMap,
    TensionResolutionGoal,
    TensionState,
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
    InputSignal,
    InputSource,
    FilterConfig,
    FilteredInputs,
    filter_inputs,
    compute_input_features
)

from cognitive_loop.trajectory import (
    TrendDirection,
    TrajectoryPhase,
    StateVector,
    TrajectorySegment,
    TrajectoryState,
    create_state_vector,
    add_vector_to_trajectory,
    compute_coherence,
    detect_divergence,
    project_trajectory,
    complete_segment
)

from cognitive_loop.emergence import (
    EmergenceType,
    EmergenceStrength,
    ComponentInteraction,
    EmergentProperty,
    EmergenceState,
    detect_interaction,
    analyze_interaction_potential,
    detect_emergence,
    add_emergent_property,
    compute_systemic_coherence,
    update_stability,
    detect_temporal_emergence
)

__all__ = [
    # Core loop
    'LoopPhase',
    'CognitiveLoopConfig',
    'CognitiveLoopState',
    'LoopEvent',
    'execute_full_cycle',
    'get_top_tensions',
    'get_active_goals',
    'get_trajectory_summary',
    'get_loop_metrics',

    # Tension
    'TensionType',
    'TensionLifecycle',
    'Tension',
    'SalienceMap',
    'TensionResolutionGoal',
    'TensionState',
    'create_initial_tension_state',
    'detect_contradiction_tension',
    'detect_vacuum_tension',
    'detect_pressure_tension',
    'compute_salience',
    'generate_tension_goal',
    'generate_goals_from_salience',
    'add_tension',
    'resolve_tension',
    'escalate_tension',

    # Attention
    'InputSignal',
    'InputSource',
    'FilterConfig',
    'FilteredInputs',
    'filter_inputs',
    'compute_input_features',

    # Trajectory
    'TrendDirection',
    'TrajectoryPhase',
    'StateVector',
    'TrajectorySegment',
    'TrajectoryState',
    'create_state_vector',
    'add_vector_to_trajectory',
    'compute_coherence',
    'detect_divergence',
    'project_trajectory',
    'complete_segment',

    # Emergence
    'EmergenceType',
    'EmergenceStrength',
    'ComponentInteraction',
    'EmergentProperty',
    'EmergenceState',
    'detect_interaction',
    'analyze_interaction_potential',
    'detect_emergence',
    'add_emergent_property',
    'compute_systemic_coherence',
    'update_stability',
    'detect_temporal_emergence'
]