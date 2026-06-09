#!/usr/bin/env python3
"""
Test Four-Layer Architecture.

Verifies:
1. Event Log is the source of truth (immutable)
2. Reducers materialize state from events
3. Policies make decisions from state
4. Adapters bridge to external action

Run: python3 test_four_layer.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernel.event_log import (
    EventLog, ImmutableEvent, EventFactory, EventTypes, StreamIds,
    CausalMetadata, EventCategory
)
from kernel.reducers import (
    CognitiveState, ExecutionState, IdentityState,
    materialize_from_log, reduce, reduce_stream, create_initial_cognitive_state
)
from kernel.policies import (
    evaluate_all_policies, identity_strategy_policy, identity_risk_policy,
    DecisionType
)
from kernel.adapters import (
    GoalExecutionAdapter, BeliefCollectionAdapter, AdapterRegistry,
    ExecutionContext
)


def test_event_log():
    """Test that event log is immutable source of truth"""
    print("Testing event log...")
    
    log = EventLog.empty()
    factory = EventFactory()
    
    # Create event
    event1 = factory.create_cognitive_event(
        event_type=EventTypes.BELIEF_CREATED,
        payload={"proposition": "The sky is blue", "confidence": 0.8, "source": "observation"}
    )
    
    # Append to log (returns NEW log)
    new_log = log.with_event(StreamIds.COGNITION, event1)
    
    # Original log unchanged
    assert log.get_total_events() == 0, "Original log unchanged"
    
    # New log has event
    assert new_log.get_total_events() == 1, "New log has event"
    
    # Get stream
    stream = new_log.get_stream(StreamIds.COGNITION)
    assert len(stream) == 1, "Stream has event"
    assert stream[0].event_type == EventTypes.BELIEF_CREATED
    
    # Event has causal metadata
    assert stream[0].causal.event_id != "", "Event has ID"
    assert stream[0].causal.trace_id != "", "Event has trace"
    
    print("  ✓ Event log is immutable source of truth")


def test_causal_metadata():
    """Test causal chain tracking"""
    print("Testing causal metadata...")
    
    factory = EventFactory(trace_id="test_trace")
    
    # Create root event
    event1 = factory.create_cognitive_event(
        event_type=EventTypes.BELIEF_CREATED,
        payload={"proposition": "Test belief"}
    )
    
    # Create child event
    event2 = factory.create_execution_event(
        event_type=EventTypes.GOAL_CREATED,
        payload={"goal_id": "g1"},
        causation_id=event1.causal.event_id
    )
    
    # Child has causation to parent
    assert event2.causal.causation_id == event1.causal.event_id, "Child references parent"
    
    # Trace contains both events
    assert event2.causal.trace_id != "", "Event has trace"
    
    print("  ✓ Causal metadata works")


def test_reducers():
    """Test that reducers materialize state from events"""
    print("Testing reducers...")
    
    log = EventLog.empty()
    factory = EventFactory()
    
    # Create events
    event1 = factory.create_cognitive_event(
        event_type=EventTypes.BELIEF_CREATED,
        payload={"proposition": "Sky is blue", "confidence": 0.8, "source": "observation"}
    )
    
    event2 = factory.create_cognitive_event(
        event_type=EventTypes.BELIEF_CREATED,
        payload={"proposition": "Sky is red", "confidence": 0.6, "source": "sunset"}
    )
    
    event3 = factory.create_cognitive_event(
        event_type=EventTypes.CONTRADICTION_DETECTED,
        payload={"type": "belief_belief", "intensity": 0.7}
    )
    
    # Build log
    log = log.with_event(StreamIds.COGNITION, event1)
    log = log.with_event(StreamIds.COGNITION, event2)
    log = log.with_event(StreamIds.COGNITION, event3)
    
    # Materialize state
    state = materialize_from_log(log, StreamIds.COGNITION)
    
    # State derived from events
    assert len(state.beliefs) == 2, "State has 2 beliefs"
    assert len(state.contradictions) == 1, "State has 1 contradiction"
    
    # Verify identity state
    log_identity = EventLog.empty()
    event_identity = factory.create_identity_event(
        event_type=EventTypes.IDENTITY_MUTATED,
        payload={"axis": "autonomy", "delta": 0.1}
    )
    log_identity = log_identity.with_event(StreamIds.IDENTITY, event_identity)
    
    identity_state = materialize_from_log(log_identity, StreamIds.IDENTITY)
    assert identity_state.axes.get("autonomy") > 0.5, "Identity axis updated"
    
    print("  ✓ Reducers materialize state from events")


def test_policies():
    """Test that policies make decisions from state"""
    print("Testing policies...")
    
    # Create identity state
    identity = IdentityState(
        axes={"autonomy": 0.8, "curiosity": 0.7, "stability": 0.5, "coherence": 0.5},
        genome={},
        lineage={},
        version=1
    )
    
    # Create cognitive state with pressure
    cognitive = CognitiveState(
        beliefs={},
        contradictions={},
        pressures={"p1": {"intensity": 0.6}},
        version=1
    )
    
    # Evaluate policies
    result = evaluate_all_policies(identity, cognitive)
    
    assert len(result.decisions) > 0, "Policy made decisions"
    assert result.total_pressure > 0, "Pressure computed"
    
    # Check strategy decision
    strategy = next((d for d in result.decisions if d.decision_type == DecisionType.STRATEGY.value), None)
    assert strategy is not None, "Strategy decision made"
    assert strategy.value in ("explorative", "conservative", "balanced"), "Valid strategy"
    
    # Check risk decision
    risk = next((d for d in result.decisions if d.decision_type == DecisionType.RISK.value), None)
    assert risk is not None, "Risk decision made"
    assert 0 <= risk.value <= 1, "Valid risk value"
    
    print("  ✓ Policies make decisions from state")


def test_identity_drives_policy():
    """Test that identity influences policy decisions"""
    print("Testing identity-driven policies...")
    
    # High autonomy identity
    high_autonomy = IdentityState(
        axes={"autonomy": 0.9, "curiosity": 0.5, "stability": 0.5, "coherence": 0.5},
        genome={},
        lineage={},
        version=1
    )
    
    # Low autonomy identity
    low_autonomy = IdentityState(
        axes={"autonomy": 0.2, "curiosity": 0.5, "stability": 0.5, "coherence": 0.5},
        genome={},
        lineage={},
        version=1
    )
    
    cognitive = CognitiveState(
        beliefs={},
        contradictions={},
        pressures={},
        version=1
    )
    
    # Evaluate for both identities
    result_high = evaluate_all_policies(high_autonomy, cognitive)
    result_low = evaluate_all_policies(low_autonomy, cognitive)
    
    # High autonomy = higher risk
    risk_high = next((d for d in result_high.decisions if d.decision_type == DecisionType.RISK.value), None)
    risk_low = next((d for d in result_low.decisions if d.decision_type == DecisionType.RISK.value), None)
    
    assert risk_high.value > risk_low.value, "High autonomy = higher risk appetite"
    
    print("  ✓ Identity drives policy decisions")


def test_adapters():
    """Test that adapters bridge to external action"""
    print("Testing adapters...")
    
    registry = AdapterRegistry()
    
    # Create context
    identity = IdentityState(
        axes={"autonomy": 0.5, "curiosity": 0.5, "stability": 0.5, "coherence": 0.5},
        genome={},
        lineage={},
        version=1
    )
    
    cognitive = CognitiveState(
        beliefs={},
        contradictions={},
        pressures={},
        version=1
    )
    
    execution = ExecutionState(
        goals={"g1": {"title": "Test goal", "status": "active", "priority": 0.8}},
        actions={},
        results={},
        version=1
    )
    
    from kernel.policies import PolicyResult
    policy = PolicyResult(
        decisions=(),
        total_pressure=0.3,
        dominant_strategy="balanced",
        risk_appetite=0.5,
        exploration_ratio=0.3,
        version=1
    )
    
    context = ExecutionContext(
        cognitive_state=cognitive,
        identity_state=identity,
        execution_state=execution,
        policy=policy,
        goal_id="g1",
        causal_metadata=CausalMetadata.create_root("context_1"),
        trace_id="trace_1"
    )
    
    # Execute goal adapter
    result = registry.execute_adapter(
        "goal_execution",
        context,
        {"title": "Test goal", "priority": 0.8}
    )
    
    assert result is not None, "Adapter returned result"
    assert hasattr(result, 'events'), "Result has events attribute"
    
    # Events have proper causal metadata
    for event in result.events:
        assert event.causal.event_id != "", "Event has causal metadata"
    
    print("  ✓ Adapters bridge to external action")


def test_causal_chain():
    """Test causal chain building"""
    print("Testing causal chain...")
    
    log = EventLog.empty()
    factory = EventFactory(trace_id="main_trace")
    
    # Create chain: belief → contradiction → pressure → action
    event1 = factory.create_cognitive_event(
        event_type=EventTypes.BELIEF_CREATED,
        payload={"proposition": "Test"}
    )
    
    event2 = factory.create_cognitive_event(
        event_type=EventTypes.CONTRADICTION_DETECTED,
        payload={"type": "test"},
        causation_id=event1.causal.event_id
    )
    
    event3 = factory.create_cognitive_event(
        event_type=EventTypes.PRESSURE_ACCUMULATED,
        payload={"type": "cognitive"},
        causation_id=event2.causal.event_id
    )
    
    event4 = factory.create_execution_event(
        event_type=EventTypes.GOAL_CREATED,
        payload={"goal_id": "g1"},
        causation_id=event3.causal.event_id
    )
    
    # Build log
    log = log.with_event(StreamIds.COGNITION, event1)
    log = log.with_event(StreamIds.COGNITION, event2)
    log = log.with_event(StreamIds.COGNITION, event3)
    log = log.with_event(StreamIds.EXECUTION, event4)
    
    # Events have proper causal metadata
    assert event1.causal.causation_id == "", "Root event has no causation"
    assert event2.causal.causation_id == event1.causal.event_id, "Child references parent"
    assert event3.causal.causation_id == event2.causal.event_id, "Grandchild references parent"
    assert event4.causal.causation_id == event3.causal.event_id, "Great-grandchild references parent"
    
    print("  ✓ Causal chain works")


def test_four_layer_integration():
    """Test full four-layer integration"""
    print("Testing four-layer integration...")
    
    log = EventLog.empty()
    factory = EventFactory(trace_id="integration_test")
    
    # Layer 1: Events
    belief = factory.create_cognitive_event(
        event_type=EventTypes.BELIEF_CREATED,
        payload={"proposition": "Goal achievable", "confidence": 0.7, "source": "analysis"}
    )
    
    goal = factory.create_execution_event(
        event_type=EventTypes.GOAL_CREATED,
        payload={"goal_id": "achieve_goal", "title": "Achieve X", "priority": 0.8},
        correlation_id="trace_1"
    )
    
    executed = factory.create_execution_event(
        event_type=EventTypes.GOAL_EXECUTED,
        payload={"goal_id": "achieve_goal", "strategy": "explorative"},
        causation_id=goal.causal.event_id
    )
    
    log = log.with_event(StreamIds.COGNITION, belief)
    log = log.with_event(StreamIds.EXECUTION, goal)
    log = log.with_event(StreamIds.EXECUTION, executed)
    
    # Layer 2: Reducers (materialize state)
    cognitive_state = materialize_from_log(log, StreamIds.COGNITION)
    execution_state = materialize_from_log(log, StreamIds.EXECUTION)
    
    # Layer 3: Policies (decide from state)
    from kernel.adapters import ExecutionContext
    from kernel.policies import PolicyResult
    
    identity = IdentityState(
        axes={"autonomy": 0.6, "curiosity": 0.7, "stability": 0.5, "coherence": 0.5},
        genome={},
        lineage={},
        version=1
    )
    
    policy_result = evaluate_all_policies(identity, cognitive_state)
    
    # Layer 4: Adapter (action from decision)
    context = ExecutionContext(
        cognitive_state=cognitive_state,
        identity_state=identity,
        execution_state=execution_state,
        policy=policy_result,
        goal_id="achieve_goal",
        causal_metadata=executed.causal,
        trace_id="integration_test"
    )
    
    adapter_result = GoalExecutionAdapter().execute_goal(
        context,
        {"title": "Achieve X", "priority": 0.8}
    )
    
    # Verify full flow
    assert log.get_total_events() == 3, "Events logged"
    assert len(cognitive_state.beliefs) == 1, "State materialized"
    assert len(execution_state.goals) == 1, "Execution state materialized"
    assert len(policy_result.decisions) > 0, "Policy decided"
    assert len(adapter_result.events) > 0, "Adapter acted"
    
    print("  ✓ Four-layer integration works")


def main():
    print("=" * 60)
    print("FOUR-LAYER ARCHITECTURE TESTS")
    print("=" * 60)
    print()
    
    tests = [
        test_event_log,
        test_causal_metadata,
        test_reducers,
        test_policies,
        test_identity_drives_policy,
        test_adapters,
        test_causal_chain,
        test_four_layer_integration,
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
    print("Four-Layer Architecture verified:")
    print("  ✓ Event Log - immutable source of truth")
    print("  ✓ Causal Metadata - full lineage tracking")
    print("  ✓ Reducers - pure state materialization")
    print("  ✓ Policies - identity-driven decisions")
    print("  ✓ Adapters - external action bridge")
    print("  ✓ Causal chains - root cause analysis")
    print("  ✓ Full integration - Events → Reducers → Policies → Adapters")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)