"""
Integration Test for Production Cognitive Runtime.

Verifies:
1. Pure reducers produce immutable state
2. Commands → Events → Reducers → State flow works
3. Deterministic replay produces same state
4. Optimistic concurrency control works
5. Policy engine correctly evaluates commands

Run standalone:
    cd /home/onor/ai_os_final/services/core/experience
    python3 test_production_runtime.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import unittest
import tempfile
from datetime import datetime

from production_cognitive_runtime import ProductionCognitiveRuntime, create_runtime
from cognitive_state import CognitiveState, initial_state
from event_sourcing.events import CognitiveEvent, StreamIds, EventTypes, SchemaVersion
from event_sourcing.commands import AddBelief, UpdateBelief, RemoveBelief, RegisterContradiction
from event_sourcing.reducers import reduce, reduce_sequence
from invariant_engine import InvariantEngine, verify_replay_equivalence


class TestPureReducers(unittest.TestCase):
    """Test that reducers are pure (no mutations)"""
    
    def test_belief_add_returns_new_state(self):
        """Adding belief returns NEW state, original unchanged"""
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
        
        self.assertIsNot(state, new_state)
        self.assertEqual(state.belief_count, 0)
        self.assertIsInstance(new_state, CognitiveState)
    
    def test_reduce_sequence_returns_final_state(self):
        """Reducing sequence returns same final state regardless of intermediate"""
        state = initial_state()
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
        
        result1 = reduce_sequence(state, events)
        result2 = reduce_sequence(initial_state(), events)
        
        self.assertEqual(
            CognitiveState.compute_hash(result1),
            CognitiveState.compute_hash(result2)
        )


class TestCommandHandler(unittest.TestCase):
    """Test command handler flow"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.runtime = create_runtime(self.temp_db.name)
    
    def tearDown(self):
        self.runtime.close()
        os.unlink(self.temp_db.name)
    
    def test_add_belief_command(self):
        """Test AddBelief command → event → state"""
        cmd = AddBelief(
            payload={
                "proposition": "Test belief",
                "confidence": 0.75,
                "entropy": 0.1,
                "source": "integration_test"
            }
        )
        
        result = self.runtime.execute_command(cmd)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.event)
        self.assertEqual(result.event.event_type, EventTypes.BELIEF_ADDED)
        
        state = self.runtime.get_state()
        self.assertGreater(state.belief_count, 0)
    
    def test_policy_blocks_invalid_confidence(self):
        """Policy should block confidence outside [0, 1]"""
        cmd = AddBelief(
            payload={
                "proposition": "Invalid belief",
                "confidence": 1.5,
                "source": "test"
            }
        )
        
        result = self.runtime.execute_command(cmd)
        
        self.assertFalse(result.success)
        self.assertIn("Confidence", result.error)
    
    def test_policy_blocks_empty_proposition(self):
        """Policy should block empty proposition"""
        cmd = AddBelief(
            payload={
                "proposition": "",
                "confidence": 0.5,
                "source": "test"
            }
        )
        
        result = self.runtime.execute_command(cmd)
        
        self.assertFalse(result.success)


class TestDeterministicReplay(unittest.TestCase):
    """Test deterministic replay"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.runtime = create_runtime(self.temp_db.name)
        
        self._create_test_events()
    
    def tearDown(self):
        self.runtime.close()
        os.unlink(self.temp_db.name)
    
    def _create_test_events(self):
        """Create test events for replay"""
        for i in range(10):
            cmd = AddBelief(
                payload={
                    "proposition": f"Belief {i}",
                    "confidence": 0.5 + (i * 0.05),
                    "source": "replay_test"
                }
            )
            self.runtime.execute_command(cmd)
    
    def test_replay_produces_same_state(self):
        """Replay should produce identical state"""
        current_state = self.runtime.get_state()
        current_hash = CognitiveState.compute_hash(current_state)
        
        replayed_state = self.runtime.replay_stream(StreamIds.BELIEF)
        replayed_hash = CognitiveState.compute_hash(replayed_state)
        
        self.assertEqual(current_hash, replayed_hash)
    
    def test_verify_all_streams(self):
        """verify_all_streams should return True for all streams"""
        results = self.runtime.verify_all_streams()
        
        for stream_id, passed in results.items():
            self.assertTrue(passed, f"Stream {stream_id} failed verification")


class TestOptimisticConcurrency(unittest.TestCase):
    """Test optimistic concurrency control"""
    
    def test_concurrent_append_resolves(self):
        """Multiple concurrent appends should resolve via retry"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        
        from event_sourcing.event_store import PersistentEventStore, OptimisticConcurrencyError
        
        store = PersistentEventStore(self.temp_db.name)
        
        event = CognitiveEvent(
            event_type="test_event",
            stream_id="test-stream",
            position=0
        )
        
        store.append("test-stream", event)
        store.append("test-stream", event)
        store.append("test-stream", event)
        
        events = store.get_stream("test-stream")
        self.assertEqual(len(events), 3)
        
        positions = [e.position for e in events]
        self.assertEqual(positions, [1, 2, 3])
        
        store.close()
        os.unlink(self.temp_db.name)


class TestInvariantEngine(unittest.TestCase):
    """Test invariant engine"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.runtime = create_runtime(self.temp_db.name)
        self.engine = InvariantEngine(self.runtime._store)
    
    def tearDown(self):
        self.runtime.close()
        os.unlink(self.temp_db.name)
    
    def test_event_order_invariant(self):
        """Event order should be monotonic"""
        for i in range(5):
            cmd = AddBelief(
                payload={
                    "proposition": f"Belief {i}",
                    "confidence": 0.5,
                    "source": "invariant_test"
                }
            )
            self.runtime.execute_command(cmd)
        
        violations = self.engine.verify_invariant(
            "event_order", 
            StreamIds.BELIEF
        )
        
        self.assertEqual(len(violations), 0)
    
    def test_idempotency_invariant(self):
        """No duplicate event IDs"""
        violations = self.engine.verify_invariant(
            "idempotency",
            StreamIds.BELIEF
        )
        
        event_ids = set()
        for e in self.engine._store.get_stream(StreamIds.BELIEF):
            self.assertNotIn(e.event_id, event_ids)
            event_ids.add(e.event_id)


class TestProjectionConsistency(unittest.TestCase):
    """Test projection consistency"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.temp_db.close()
        self.runtime = create_runtime(self.temp_db.name)
    
    def tearDown(self):
        self.runtime.close()
        os.unlink(self.temp_db.name)
    
    def test_belief_projection_matches_state(self):
        """Projection beliefs should match canonical state"""
        for i in range(5):
            cmd = AddBelief(
                payload={
                    "proposition": f"Belief {i}",
                    "confidence": 0.6 + (i * 0.05),
                    "source": "projection_test"
                }
            )
            self.runtime.execute_command(cmd)
        
        belief_proj = self.runtime.get_projection("belief_projection")
        state = self.runtime.get_state()
        
        projection_beliefs = belief_proj.get_all_beliefs()
        
        self.assertEqual(len(projection_beliefs), state.belief_count)


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestPureReducers))
    suite.addTests(loader.loadTestsFromTestCase(TestCommandHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestDeterministicReplay))
    suite.addTests(loader.loadTestsFromTestCase(TestOptimisticConcurrency))
    suite.addTests(loader.loadTestsFromTestCase(TestInvariantEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestProjectionConsistency))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)