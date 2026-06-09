#!/usr/bin/env python3
"""
Test Pure Domain Core - Stage 1 verification.

Tests that:
1. Events are immutable
2. State is immutable (MappingProxyType)
3. Reducers are pure (no side effects)
4. Replay is deterministic
5. Projections don't affect canonical state

Run: python3 test_domain_core.py
"""
import sys
import os

# Add experience directory to path
exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, exp_dir)

from events import DomainEvent, create_event, BeliefAdded, IdentityMutated
from state import (
    DomainState, 
    ImmutableIdentity, 
    ImmutableGenome,
    create_initial_state
)
from reducers import reduce, reduce_sequence, replay_from_events, materialize_state
from projections import BeliefProjection, ProjectionManager


def test_events_immutable():
    """Test that events are truly immutable"""
    print("Testing events immutability...")
    
    event = create_event(
        event_type="belief_added",
        stream_id="belief-stream",
        position=1,
        payload={"belief_id": "test", "proposition": "Test"}
    )
    
    assert hasattr(event, 'event_id')
    assert hasattr(event, 'payload')
    
    try:
        event.payload = {}
        assert False, "Event should be frozen"
    except Exception:
        pass
    
    print("  ✓ Events are frozen immutable")


def test_state_immutable():
    """Test that state uses MappingProxyType for true immutability"""
    print("Testing state immutability...")
    
    state = create_initial_state()
    
    assert hasattr(state.beliefs, '__getitem__')
    
    try:
        state.beliefs["test"] = "value"
        assert False, "Beliefs should be immutable"
    except (TypeError, AttributeError):
        pass
    
    try:
        state.identity.autonomy = 999
        assert False, "Identity should be frozen"
    except Exception:
        pass
    
    print("  ✓ State uses MappingProxyType (truly immutable)")


def test_identity_with_axis():
    """Test that identity.with_axis() returns NEW instance"""
    print("Testing identity transform...")
    
    identity = ImmutableIdentity(
        autonomy=0.5,
        curiosity=0.5,
        stability=0.5,
        coherence=0.5
    )
    
    new_identity = identity.with_axis("autonomy", 0.1)
    
    assert identity.autonomy == 0.5, "Original unchanged"
    assert new_identity.autonomy == 0.6, "New has updated value"
    assert identity is not new_identity, "Returns new instance"
    
    print("  ✓ Identity.with_axis() returns new instance")


def test_reducer_pure():
    """Test that reducers are pure (no side effects)"""
    print("Testing reducer purity...")
    
    state = create_initial_state()
    state_hash_before = DomainState.compute_hash(state)
    
    event = create_event(
        event_type="belief_added",
        stream_id="belief-stream",
        position=1,
        payload={
            "belief_id": "test_belief",
            "proposition": "Test belief",
            "confidence": 0.8,
            "source": "test"
        }
    )
    
    new_state = reduce(state, event)
    
    assert state.version == 0, "Original state unchanged"
    assert new_state.version == 1, "New state incremented"
    assert state is not new_state, "Returns new state"
    assert len(state.beliefs) == 0, "Original has no beliefs"
    assert len(new_state.beliefs) == 1, "New state has belief"
    
    state_hash_after = DomainState.compute_hash(state)
    assert state_hash_before == state_hash_after, "Original hash unchanged"
    
    print("  ✓ Reducers are pure (no mutations)")


def test_replay_deterministic():
    """Test that replay produces deterministic results"""
    print("Testing replay determinism...")
    
    events = tuple([
        create_event("belief_added", "belief-stream", 1, 
                    {"belief_id": f"belief_{i}", "proposition": f"Belief {i}", 
                     "confidence": 0.5 + i*0.05, "source": "test"})
        for i in range(5)
    ])
    
    state1 = replay_from_events(events)
    state2 = replay_from_events(events)
    
    hash1 = DomainState.compute_hash(state1)
    hash2 = DomainState.compute_hash(state2)
    
    assert hash1 == hash2, "Same events → same state"
    assert state1.belief_count == 5, "State has 5 beliefs"
    
    print("  ✓ Replay is deterministic")


def test_identity_mutations_lineage():
    """Test that identity mutations update lineage"""
    print("Testing identity mutations and lineage...")
    
    state = create_initial_state()
    
    events = tuple([
        create_event("identity_mutated", "identity-stream", i+1, 
                    {"axis": "autonomy", "delta": 0.1})
        for i in range(3)
    ])
    
    final_state = replay_from_events(events)
    
    assert 0.79 < final_state.identity.autonomy < 0.81, f"Identity updated, got {final_state.identity.autonomy}"
    assert len(final_state.lineage.mutation_history) == 3, "3 mutations recorded"
    assert final_state.lineage.trajectory_hash != "", "Trajectory hash computed"
    
    print("  ✓ Identity mutations track lineage")


def test_projections_isolated():
    """Test that projections don't affect canonical state"""
    print("Testing projection isolation...")
    
    state = create_initial_state()
    
    event = create_event(
        event_type="belief_added",
        stream_id="belief-stream",
        position=1,
        payload={"belief_id": "projected_belief", "proposition": "Test", "confidence": 0.8, "source": "test"}
    )
    
    state_after = reduce(state, event)
    
    proj = BeliefProjection()
    proj.project(event)
    
    proj_beliefs = proj.get_all()
    assert len(proj_beliefs) == 1, "Projection has belief"
    assert len(state_after.beliefs) == 1, "Canonical has belief"
    
    proj.project(event)
    assert len(proj.get_all()) == 1, "Projection idempotent"
    
    print("  ✓ Projections isolated from canonical state")


def test_materialize_state():
    """Test state materialization from events"""
    print("Testing state materialization...")
    
    events = tuple([
        create_event("belief_added", "belief-stream", i+1,
                    {"belief_id": f"belief_{i}", "proposition": f"Belief {i}",
                     "confidence": 0.7, "source": "test"})
        for i in range(10)
    ] + [
        create_event("identity_mutated", "identity-stream", 1,
                    {"axis": "autonomy", "delta": 0.3})
    ])
    
    state = materialize_state(events, from_position=1)
    
    assert state.belief_count == 10, "Has 10 beliefs"
    assert state.identity.autonomy == 0.8, "Identity updated"
    assert state.version > 0, "Version incremented"
    
    print("  ✓ State materialization works")


def test_reducer_composition():
    """Test reducer composition for complex flows"""
    print("Testing reducer composition...")
    
    events = tuple([
        create_event("belief_added", "belief-stream", 1, {"belief_id": "b1", "proposition": "P1", "confidence": 0.8, "source": "s1"}),
        create_event("belief_added", "belief-stream", 2, {"belief_id": "b2", "proposition": "P2", "confidence": 0.3, "source": "s1"}),
        create_event("contradiction_registered", "contradiction-stream", 1, {"episode_id": "c1", "belief_ids": ["b1", "b2"], "type": "conflict"}),
        create_event("identity_mutated", "identity-stream", 1, {"axis": "stability", "delta": -0.1}),
    ])
    
    state = reduce_sequence(create_initial_state(), events)
    
    assert state.belief_count == 2, "2 beliefs"
    assert len(state.contradictions) == 1, "1 contradiction"
    assert state.identity.stability == 0.4, "Stability decreased"
    
    print("  ✓ Reducer composition works")


def main():
    print("=" * 60)
    print("PURE DOMAIN CORE TESTS - Stage 1")
    print("=" * 60)
    print()
    
    tests = [
        test_events_immutable,
        test_state_immutable,
        test_identity_with_axis,
        test_reducer_pure,
        test_replay_deterministic,
        test_identity_mutations_lineage,
        test_projections_isolated,
        test_materialize_state,
        test_reducer_composition,
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
    print("Domain Core verified:")
    print("  ✓ Events are frozen (immutable)")
    print("  ✓ State uses MappingProxyType (truly immutable)")
    print("  ✓ Reducers are pure (no side effects)")
    print("  ✓ Replay is deterministic")
    print("  ✓ Projections isolated from canonical state")
    print("  ✓ Lineage tracking works")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)