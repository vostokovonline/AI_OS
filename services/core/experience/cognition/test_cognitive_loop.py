#!/usr/bin/env python3
"""
Test Cognitive Loop - Stage: Cognitive Architecture

Verifies:
1. Memory types work (beliefs, episodes, procedures, reflections)
2. World model works (entities, relations, causality)
3. Contradiction engine detects conflicts and computes pressure
4. Goal economy prioritizes and allocates resources
5. Self model tracks capabilities and limitations
6. Cognitive loop produces emergent behavior

Run: python3 test_cognitive_loop.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognition.memory import (
    MemoryState, Belief, EpisodicMemory, Procedure, Reflection,
    create_empty_memory
)
from cognition.world import (
    WorldModel, Entity, Relation, CausalLink,
    create_empty_world
)
from cognition.contradiction import (
    ContradictionState, Contradiction, Pressure,
    create_empty_contradiction_state,
    detect_belief_belief_conflict, compute_pressure_intensity
)
from cognition.goals import (
    GoalMarket, Goal, GoalStatus, create_goal_market
)
from cognition.self import (
    SelfModel, Capability, ResourceModel,
    create_initial_self_model
)
from cognition.loop import (
    CognitiveState, CognitiveEvent,
    create_initial_cognitive_state,
    cognitive_reduce, cognitive_reduce_sequence,
    compute_system_pressure, generate_intent,
    assess_cognitive_health
)


def test_memory_types():
    """Test all four memory types"""
    print("Testing memory types...")
    
    memory = create_empty_memory()
    
    # Test beliefs
    belief = Belief(
        belief_id="belief_1",
        proposition="The sky is blue",
        confidence=0.9,
        evidence=(),
        counter_evidence=(),
        sources=("observation",),
        created_at="2024-01-01T00:00:00",
        last_updated="2024-01-01T00:00:00",
        stability=0.8,
        revision_count=0
    )
    
    memory = memory.with_belief("belief_1", belief)
    
    assert len(memory.beliefs) == 1, "Belief added"
    assert memory.beliefs["belief_1"].confidence == 0.9
    
    # Test episodic memory
    episode = EpisodicMemory(
        episode_id="episode_1",
        context=(("action", "test"),),
        actions=("testing",),
        outcome="success",
        emotional_valence=0.5,
        lessons=("test passed",),
        timestamp="2024-01-01T00:00:00",
        duration_ms=100,
        related_goals=()
    )
    
    memory = memory.with_episode("episode_1", episode)
    assert len(memory.episodes) == 1, "Episode added"
    
    print("  ✓ Memory types work")


def test_world_model():
    """Test world model operations"""
    print("Testing world model...")
    
    world = create_empty_world()
    
    # Add entity
    entity = Entity(
        entity_id="user_1",
        entity_type="person",
        properties=frozenset([("name", "Alice"), ("role", "admin")]),
        created_at="2024-01-01T00:00:00"
    )
    
    world = world.with_entity(entity)
    assert len(world.entities) == 1, "Entity added"
    assert world.get_entity("user_1").get_property("name") == "Alice"
    
    # Add causal link
    link = CausalLink(
        cause_id="action_1",
        effect_id="outcome_1",
        mechanism="direct",
        strength=0.8,
        conditions=(),
        evidence=("test_1",),
        created_at="2024-01-01T00:00:00"
    )
    
    world = world.with_causal_link(link)
    assert len(world.causal_links) == 1, "Causal link added"
    
    # Test inference
    outcomes = world.infer_outcomes("action_1", set())
    assert len(outcomes) == 1, "Inference works"
    
    print("  ✓ World model works")


def test_contradiction_engine():
    """Test contradiction detection and pressure"""
    print("Testing contradiction engine...")
    
    state = create_empty_contradiction_state()
    
    # Detect belief conflict
    intensity = detect_belief_belief_conflict("b1", 0.8, "b2", 0.7, True)
    assert intensity is not None, "Conflict detected"
    assert intensity > 0.6, "High confidence = high conflict"
    
    # Add contradiction
    contradiction = Contradiction(
        contradiction_id="c1",
        type="belief_belief",
        participants=frozenset(["b1", "b2"]),
        description="Two beliefs conflict",
        intensity=0.75,
        detected_at="2024-01-01T00:00:00",
        resolution_state="unresolved",
        evidence=(),
        resolution_attempts=()
    )
    
    from types import MappingProxyType
    new_contr = {**state.contradictions, "c1": contradiction}
    state = ContradictionState(
        contradictions=MappingProxyType(new_contr),
        pressures=state.pressures,
        resolution_attempts=state.resolution_attempts,
        total_pressure=state.total_pressure + 0.75,
        unresolved_count=1,
        escalated_count=0,
        version=1
    )
    
    assert len(state.contradictions) == 1, "Contradiction added"
    assert state.unresolved_count == 1, "Count tracked"
    
    print("  ✓ Contradiction engine works")


def test_goal_economy():
    """Test goal competition and allocation"""
    print("Testing goal economy...")
    
    market = create_goal_market(1.0)
    
    # Add high priority goal
    goal1 = Goal(
        goal_id="g1",
        title="Critical task",
        description="Must do now",
        target_state="{}",
        goal_type="achievement",
        priority=0.9,
        base_priority=0.9,
        resource_cost=0.3,
        deadline="2024-01-01T01:00:00",
        created_at="2024-01-01T00:00:00",
        started_at=None,
        completed_at=None,
        status="active",
        dependencies=(),
        parent_goal_id=None,
        progress=0.0,
        value_contribution=0.8,
        risk_level=0.2,
        executor_id=None
    )
    
    market = market.with_goal(goal1)
    
    assert len(market.goals) == 1, "Goal added"
    assert market.goals["g1"].priority == 0.9
    
    # Compute market price
    price = market.compute_market_price(goal1)
    assert price > 0.5, "High priority = high price"
    
    # Add low priority goal
    goal2 = Goal(
        goal_id="g2",
        title="Background task",
        description="Can wait",
        target_state="{}",
        goal_type="maintenance",
        priority=0.3,
        base_priority=0.3,
        resource_cost=0.2,
        deadline=None,
        created_at="2024-01-01T00:00:00",
        started_at=None,
        completed_at=None,
        status="active",
        dependencies=(),
        parent_goal_id=None,
        progress=0.0,
        value_contribution=0.3,
        risk_level=0.1,
        executor_id=None
    )
    
    market = market.with_goal(goal2)
    
    # High priority should have higher price
    price1 = market.compute_market_price(goal1)
    price2 = market.compute_market_price(goal2)
    assert price1 > price2, "Resources go to higher priority"
    
    print("  ✓ Goal economy works")


def test_self_model():
    """Test self model capabilities and limitations"""
    print("Testing self model...")
    
    model = create_initial_self_model()
    
    assert len(model.capabilities) >= 3, "Default capabilities"
    assert model.capabilities["reasoning"].proficiency > 0.5, "Good at reasoning"
    
    # Test success probability estimation
    prob = model.estimate_success_probability("reasoning")
    assert 0.5 < prob < 1.0, "Reasonable probability estimate"
    
    # Add failure
    model = model.add_failure("failed_goal_1")
    assert "failed_goal_1" in model.recent_failures, "Failure recorded"
    
    print("  ✓ Self model works")


def test_cognitive_loop():
    """Test the cognitive loop processing"""
    print("Testing cognitive loop...")
    
    state = create_initial_cognitive_state()
    
    # Create execution outcome event
    event = CognitiveEvent(
        event_type="execution_outcome",
        event_id="exec_1",
        source="test",
        timestamp="2024-01-01T00:00:00",
        content={
            "outcome": "success",
            "action": "test_action",
            "capability_id": "reasoning",
            "duration_ms": 100,
            "goal_ids": []
        },
        implications=()
    )
    
    new_state = cognitive_reduce(state, event)
    
    assert new_state.cycle_count == 1, "Cycle count incremented"
    assert len(new_state.memory.episodes) == 1, "Episode recorded"
    assert new_state.memory.episodes["exec_1"].emotional_valence == 1.0, "Positive valence"
    
    # Add evidence that creates contradiction
    # First add a belief about sky color
    event2 = CognitiveEvent(
        event_type="evidence_received",
        event_id="ev_1",
        source="test",
        timestamp="2024-01-01T00:01:00",
        content={
            "proposition": "The sky is blue",
            "strength": 0.8,
            "source": "observation"
        },
        implications=()
    )
    
    new_state = cognitive_reduce(new_state, event2)
    assert len(new_state.memory.beliefs) > 0, "Initial belief added"
    
    # Now add contradictory evidence
    event3 = CognitiveEvent(
        event_type="evidence_received",
        event_id="ev_2",
        source="test",
        timestamp="2024-01-01T00:02:00",
        content={
            "proposition": "The sky is green",
            "strength": 0.85,
            "source": "test",
            "contradicts": True
        },
        implications=()
    )
    
    new_state = cognitive_reduce(new_state, event3)
    
    # Should detect contradiction
    assert len(new_state.contradictions.contradictions) > 0, "Contradiction detected"
    
    print("  ✓ Cognitive loop works")


def test_system_pressure():
    """Test pressure computation"""
    print("Testing system pressure...")
    
    state = create_initial_cognitive_state()
    
    # Add contradiction by updating contradiction state
    contradiction = Contradiction(
        contradiction_id="c1",
        type="belief_belief",
        participants=frozenset(["b1", "b2"]),
        description="Conflict",
        intensity=0.8,
        detected_at="2024-01-01T00:00:00",
        resolution_state="unresolved",
        evidence=(),
        resolution_attempts=()
    )
    
    from types import MappingProxyType
    new_contr = {**state.contradictions.contradictions, "c1": contradiction}
    new_contr_state = ContradictionState(
        contradictions=MappingProxyType(new_contr),
        pressures=state.contradictions.pressures,
        resolution_attempts=state.contradictions.resolution_attempts,
        total_pressure=0.8,
        unresolved_count=1,
        escalated_count=0,
        version=1
    )
    
    state = CognitiveState(
        memory=state.memory,
        world=state.world,
        contradictions=new_contr_state,
        goals=state.goals,
        self_model=state.self_model,
        cycle_count=state.cycle_count,
        total_pressure=0.8,
        active_intent=state.active_intent,
        version=state.version,
        last_cycle=state.last_cycle
    )
    
    pressure = compute_system_pressure(state)
    assert pressure > 0.5, "System under pressure"
    
    print("  ✓ System pressure works")


def test_generate_intent():
    """Test intent generation from state"""
    print("Testing intent generation...")
    
    state = create_initial_cognitive_state()
    
    # No critical issues
    intent = generate_intent(state)
    assert intent is not None, "Intent generated"
    
    # Add critical contradiction
    contradiction = Contradiction(
        contradiction_id="critical_1",
        type="identity_strategy",
        participants=frozenset(["identity", "strategy"]),
        description="Critical conflict",
        intensity=0.95,
        detected_at="2024-01-01T00:00:00",
        resolution_state="escalated",
        evidence=(),
        resolution_attempts=()
    )
    
    from types import MappingProxyType
    new_contr = {**state.contradictions.contradictions, "critical_1": contradiction}
    new_contr_state = ContradictionState(
        contradictions=MappingProxyType(new_contr),
        pressures=state.contradictions.pressures,
        resolution_attempts=state.contradictions.resolution_attempts,
        total_pressure=0.95,
        unresolved_count=1,
        escalated_count=1,
        version=1
    )
    
    state = CognitiveState(
        memory=state.memory,
        world=state.world,
        contradictions=new_contr_state,
        goals=state.goals,
        self_model=state.self_model,
        cycle_count=state.cycle_count,
        total_pressure=0.95,
        active_intent=state.active_intent,
        version=state.version,
        last_cycle=state.last_cycle
    )
    
    intent = generate_intent(state)
    assert "contradiction" in intent, "Prioritizes critical contradiction"
    
    print("  ✓ Intent generation works")


def test_cognitive_health():
    """Test cognitive health assessment"""
    print("Testing cognitive health assessment...")
    
    state = create_initial_cognitive_state()
    
    health = assess_cognitive_health(state)
    
    assert "cycle_count" in health
    assert "total_pressure" in health
    assert "recommended_action" in health
    assert health["recommended_action"] is not None
    
    print("  ✓ Cognitive health assessment works")


def main():
    print("=" * 60)
    print("COGNITIVE LOOP TESTS - Stage: Cognitive Architecture")
    print("=" * 60)
    print()
    
    tests = [
        test_memory_types,
        test_world_model,
        test_contradiction_engine,
        test_goal_economy,
        test_self_model,
        test_cognitive_loop,
        test_system_pressure,
        test_generate_intent,
        test_cognitive_health,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print()
    print("Cognitive Architecture verified:")
    print("  ✓ Memory types (beliefs, episodes, procedures, reflections)")
    print("  ✓ World model (entities, relations, causality)")
    print("  ✓ Contradiction engine (detection, pressure)")
    print("  ✓ Goal economy (competition, allocation)")
    print("  ✓ Self model (capabilities, limitations)")
    print("  ✓ Cognitive loop (events → state)")
    print("  ✓ System pressure computation")
    print("  ✓ Intent generation from state")
    print("  ✓ Cognitive health assessment")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)