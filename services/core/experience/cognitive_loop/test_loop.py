"""
Test cognitive loop modules.
"""

import sys
sys.path.insert(0, '/home/onor/ai_os_final/services/core/experience')

from cognitive_loop import (
    CognitiveLoopState,
    execute_full_cycle,
    get_top_tensions,
    get_loop_metrics,
    InputSource,
    FilterConfig,
    TensionType
)
from datetime import datetime


def test_basic_loop():
    """Test basic cognitive loop execution"""
    print("Testing basic cognitive loop...")
    
    # Create initial state
    state = CognitiveLoopState.initial()
    print(f"  Initial state: cycle={state.cycle_count}, tensions={len(state.tension_state.tensions)}")
    
    # Create some inputs
    raw_inputs = (
        InputSource(
            source_id="test_input_1",
            source_type="internal",
            content="Test input with high relevance",
            raw_data=frozenset(),
            timestamp=datetime.utcnow().isoformat(),
            authority_score=0.7,
            novelty_score=0.6,
            relevance_score=0.8
        ),
    )
    
    # Execute full cycle
    state, goals, event = execute_full_cycle(state, raw_inputs)
    
    print(f"  After cycle: cycle={state.cycle_count}, goals_generated={len(goals)}")
    print(f"  Event: {event.event_id}")
    
    # Check metrics
    metrics = get_loop_metrics(state)
    print(f"  Metrics: {metrics}")
    
    return state, goals


def test_tension_detection():
    """Test tension detection"""
    print("\nTesting tension detection...")
    
    from cognitive_loop import detect_pressure_tension, add_tension, TensionState
    
    state = CognitiveLoopState.initial()
    
    # Add a tension
    tension = detect_pressure_tension(
        source="test_source",
        intensity=0.7,
        pressure_type="cognitive"
    )
    
    state = CognitiveLoopState(
        tension_state=add_tension(state.tension_state, tension),
        trajectory_state=state.trajectory_state,
        emergence_state=state.emergence_state,
        filter_config=state.filter_config,
        loop_config=state.loop_config,
        current_phase=state.current_phase,
        cycle_count=state.cycle_count,
        last_cycle_time=state.last_cycle_time,
        adaptation_metrics=state.adaptation_metrics
    )
    
    print(f"  Tensions after adding: {len(state.tension_state.tensions)}")
    
    # Get top tensions
    top = get_top_tensions(state, n=3)
    print(f"  Top tensions: {[t.tension_id for t in top]}")
    
    return state


def test_trajectory():
    """Test trajectory tracking"""
    print("\nTesting trajectory tracking...")
    
    from cognitive_loop import (
        TrajectoryState,
        create_state_vector,
        add_vector_to_trajectory
    )
    
    state = CognitiveLoopState.initial()
    
    # Create and add state vectors
    state_dict = {
        'tension_energy': 0.6,
        'attention_allocated': 0.5,
        'goal_count': 3,
        'belief_coherence': 0.7,
        'identity_stability': 0.8,
        'knowledge_coverage': 0.5,
        'trajectory_direction': 'stable'
    }
    
    vector = create_state_vector(state_dict, 1)
    print(f"  Created vector: {vector.vector_id}")
    
    return state


def test_emergence():
    """Test emergence detection"""
    print("\nTesting emergence detection...")
    
    from cognitive_loop import (
        EmergenceState,
        detect_interaction,
        detect_emergence
    )
    
    state = CognitiveLoopState.initial()
    
    # Detect interaction
    interaction = detect_interaction(
        component_a="belief_1",
        component_b="belief_2",
        interaction_type="synergy",
        strength=0.7,
        context={"belief"}
    )
    
    print(f"  Interaction: {interaction.interaction_id}")
    
    # Detect emergence
    emergent = detect_emergence(state.emergence_state, interaction)
    
    if emergent:
        print(f"  Emergence detected: {emergent.property_id}")
    else:
        print("  No emergence detected (expected for weak interaction)")
    
    return state


def test_attention_filtering():
    """Test attention filtering"""
    print("\nTesting attention filtering...")
    
    from cognitive_loop import (
        InputSource,
        FilterConfig,
        filter_inputs
    )
    
    # Create inputs with varying quality
    inputs = (
        InputSource(
            source_id="high_value",
            source_type="external",
            content="High value signal",
            raw_data=frozenset(),
            timestamp=datetime.utcnow().isoformat(),
            authority_score=0.8,
            novelty_score=0.7,
            relevance_score=0.9
        ),
        InputSource(
            source_id="low_value",
            source_type="external",
            content="Low value signal",
            raw_data=frozenset(),
            timestamp=datetime.utcnow().isoformat(),
            authority_score=0.2,
            novelty_score=0.1,
            relevance_score=0.1
        ),
    )
    
    config = FilterConfig()
    filtered = filter_inputs(inputs, config)
    
    print(f"  Inputs considered: {filtered.total_considered}")
    print(f"  Inputs passed: {len(filtered.inputs)}")
    print(f"  Discarded: {filtered.discarded_count}")
    
    return filtered


if __name__ == "__main__":
    print("=" * 60)
    print("Cognitive Loop Tests")
    print("=" * 60)
    
    state, goals = test_basic_loop()
    state = test_tension_detection()
    state = test_trajectory()
    state = test_emergence()
    filtered = test_attention_filtering()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)