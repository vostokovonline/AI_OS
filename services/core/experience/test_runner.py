#!/usr/bin/env python3
"""
Test runner for production cognitive runtime.

Run from the experience directory:
    python3 test_runner.py
"""
import sys
import os

exp_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, exp_dir)

from production_cognitive_runtime import ProductionCognitiveRuntime, create_runtime
from cognitive_state import CognitiveState, initial_state
from event_sourcing.events import CognitiveEvent, StreamIds, EventTypes, SchemaVersion
from event_sourcing.commands import AddBelief, UpdateBelief
from event_sourcing.reducers import reduce, reduce_sequence
from invariant_engine import InvariantEngine

import tempfile


def test_pure_reducers():
    """Test that reducers are pure (no mutations)"""
    print("Testing pure reducers...")
    
    state = initial_state()
    event = {
        "event_type": "belief_added",
        "target_id": "belief_1",
        "payload": {
            "proposition": "Test belief",
            "confidence": 0.8,
            "source": "test"
        }
    }
    
    new_state = reduce(state, event)
    
    assert state is not new_state, "Reducer must return NEW state"
    assert state.belief_count == 0, "Original state unchanged"
    assert new_state.belief_count == 1, "New state has belief"
    assert isinstance(new_state, CognitiveState), "Returns CognitiveState"
    
    print("  ✓ Reducers are pure")


def test_reduce_sequence():
    """Test that reducing sequence is deterministic"""
    print("Testing reduce_sequence determinism...")
    
    events = [
        {
            "event_type": "belief_added",
            "target_id": f"belief_{i}",
            "payload": {
                "proposition": f"Belief {i}",
                "confidence": 0.7,
                "source": "test"
            }
        }
        for i in range(5)
    ]
    
    result1 = reduce_sequence(initial_state(), events)
    result2 = reduce_sequence(initial_state(), events)
    
    assert CognitiveState.compute_hash(result1) == CognitiveState.compute_hash(result2), \
        "Same events → same state"
    
    print("  ✓ Reduce sequence is deterministic")


def test_command_flow():
    """Test Command → Event → State flow"""
    print("Testing command flow...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    
    try:
        runtime = create_runtime(db_path)
        
        cmd = AddBelief(
            payload={
                "proposition": "Test belief",
                "confidence": 0.75,
                "source": "integration_test"
            }
        )
        
        result = runtime.execute_command(cmd)
        
        assert result.success, f"Command failed: {result.error}"
        assert result.event is not None, "Event not created"
        assert result.event.event_type == EventTypes.BELIEF_ADDED
        
        state = runtime.get_state()
        assert state.belief_count > 0, "State has beliefs"
        
        runtime.close()
        print("  ✓ Command flow works")
    finally:
        os.unlink(db_path)


def test_policy_blocks_invalid():
    """Test that policy engine blocks invalid commands"""
    print("Testing policy enforcement...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    
    try:
        runtime = create_runtime(db_path)
        
        cmd = AddBelief(
            payload={
                "proposition": "",
                "confidence": 1.5,
                "source": "test"
            }
        )
        
        result = runtime.execute_command(cmd)
        
        assert not result.success, "Invalid command should be rejected"
        assert "Confidence" in result.error or "empty" in result.error
        
        runtime.close()
        print("  ✓ Policy blocks invalid commands")
    finally:
        os.unlink(db_path)


def test_deterministic_replay():
    """Test that replay produces same state"""
    print("Testing deterministic replay...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    
    try:
        runtime = create_runtime(db_path)
        
        for i in range(10):
            cmd = AddBelief(
                payload={
                    "proposition": f"Belief {i}",
                    "confidence": 0.5 + (i * 0.05),
                    "source": "replay_test"
                }
            )
            runtime.execute_command(cmd)
        
        current_state = runtime.get_state()
        current_hash = CognitiveState.compute_hash(current_state)
        
        replayed_state = runtime.replay_stream(StreamIds.BELIEF)
        replayed_hash = CognitiveState.compute_hash(replayed_state)
        
        assert current_hash == replayed_hash, "Replay must produce same state"
        
        runtime.close()
        print("  ✓ Deterministic replay works")
    finally:
        os.unlink(db_path)


def test_invariant_checks():
    """Test invariant engine"""
    print("Testing invariant engine...")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = f.name
    
    try:
        runtime = create_runtime(db_path)
        engine = InvariantEngine(runtime._store)
        
        for i in range(5):
            cmd = AddBelief(
                payload={
                    "proposition": f"Belief {i}",
                    "confidence": 0.5,
                    "source": "invariant_test"
                }
            )
            runtime.execute_command(cmd)
        
        violations = engine.verify_invariant("event_order", StreamIds.BELIEF)
        assert len(violations) == 0, "No event order violations"
        
        runtime.close()
        print("  ✓ Invariant engine works")
    finally:
        os.unlink(db_path)


def main():
    print("=" * 60)
    print("Production Cognitive Runtime Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_pure_reducers,
        test_reduce_sequence,
        test_command_flow,
        test_policy_blocks_invalid,
        test_deterministic_replay,
        test_invariant_checks,
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
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)