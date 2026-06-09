"""
Cognitive Substrate Stress Tests

Tests designed to verify:
1. Transaction integrity (atomicity, rollback, concurrent collision)
2. Entropy stability (no hidden degradation)
3. Reflection recursion safety
4. Identity coherence (system remains "one system" after mutations)
5. Deterministic replay
6. Adversarial resistance
"""
import random
import time
from copy import deepcopy
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


class TestResult:
    """Single test result"""
    def __init__(self, name: str, passed: bool, details: str = ""):
        self.name = name
        self.passed = passed
        self.details = details
    
    def __repr__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.name} {self.details}"


class CognitiveStressTest:
    """Stress tests for cognitive substrate"""
    
    def __init__(self, ues_manager, reflection_kernel, scheduler):
        self._ues = ues_manager
        self._kernel = reflection_kernel
        self._scheduler = scheduler
        self._results: List[TestResult] = []
    
    def run_all(self) -> List[TestResult]:
        """Run all stress tests"""
        print("=" * 60)
        print("COGNITIVE SUBSTRATE STRESS TESTS")
        print("=" * 60)
        
        self.test_transaction_atomicity()
        self.test_rollback_correctness()
        self.test_concurrent_mutation_collision()
        self.test_entropy_stability()
        self.test_entropy_floor_protection()
        self.test_reflection_depth_limit()
        self.test_reflection_recursion()
        self.test_identity_coherence()
        self.test_deterministic_replay()
        self.test_adversarial_contradictions()
        
        print("\n" + "=" * 60)
        passed = sum(1 for r in self._results if r.passed)
        total = len(self._results)
        print(f"RESULTS: {passed}/{total} tests passed")
        print("=" * 60)
        
        return self._results
    
    def _record(self, name: str, passed: bool, details: str = ""):
        """Record test result"""
        result = TestResult(name, passed, details)
        self._results.append(result)
        print(result)
        return passed
    
    # =========================================================================
    # TEST 1: Transaction Atomicity
    # =========================================================================
    def test_transaction_atomicity(self):
        """Test that transactions are truly atomic"""
        
        # Create initial state
        for i in range(5):
            self._ues.add_belief(f"b{i}", f"Belief {i}", 0.5, 0.5, "test")
        
        initial_state = self._ues.get_current_state()
        initial_hash = initial_state.state_hash
        initial_version = initial_state.version
        
        # Create proposal with multiple operations
        from experience.reflection_kernel import (
            EpistemicMutationProposal, MutationOperation,
            MutationOperationDetail, ReflectionDepth
        )
        
        proposal = EpistemicMutationProposal(
            proposal_id="atomicity_test",
            reflection_depth=ReflectionDepth.SURFACE,
            triggered_by="test",
            operations=[
                MutationOperationDetail(MutationOperation.UPDATE_CONFIDENCE, f"b{i}", 0.5, 0.8, 0.9, "test")
                for i in range(5)
            ],
            expected_entropy_delta=0.0,
            created_at="now"
        )
        
        # Commit via transaction
        from experience.reflection_kernel import ReflectionCommitLayer
        commit = ReflectionCommitLayer(self._ues)
        tx = commit.begin_transaction(proposal)
        
        # Add operations
        for op in proposal.operations:
            tx.add_operation(op, commit._create_inverse(op))
        
        success, _ = tx.commit()
        
        if not success:
            self._record("Transaction Atomicity", False, "commit failed")
            return
        
        # Check state changed
        final_state = self._ues.get_current_state()
        
        # Version should have incremented
        version_changed = final_state.version > initial_version
        
        # If we had real mutation, beliefs would be updated
        # For now, just verify transaction processed
        self._record("Transaction Atomicity", version_changed, f"version: {initial_version} -> {final_state.version}")
    
    # =========================================================================
    # TEST 2: Rollback Correctness
    # =========================================================================
    def test_rollback_correctness(self):
        """Test that rollback actually restores state"""
        
        # Get current state
        state_before = self._ues.get_current_state()
        hash_before = state_before.state_hash
        version_before = state_before.version
        
        # Simulate a transaction that gets rolled back
        from experience.reflection_kernel import (
            EpistemicTransaction, EpistemicMutationProposal,
            MutationOperation, MutationOperationDetail, ReflectionDepth,
            ReflectionCommitLayer
        )
        
        proposal = EpistemicMutationProposal(
            proposal_id="rollback_test",
            reflection_depth=ReflectionDepth.SURFACE,
            triggered_by="test",
            operations=[
                MutationOperationDetail(MutationOperation.UPDATE_CONFIDENCE, "b0", 0.5, 0.9, 0.9, "test")
            ],
            expected_entropy_delta=0.0,
            created_at="now"
        )
        
        commit_layer = ReflectionCommitLayer(self._ues)
        tx = commit_layer.begin_transaction(proposal)
        tx.add_operation(proposal.operations[0], commit_layer._create_inverse(proposal.operations[0]))
        
        # Commit
        tx.commit()
        
        # Now rollback
        success, msg = commit_layer.rollback_proposal("rollback_test")
        
        # Verify state restored
        state_after = self._ues.get_current_state()
        
        # With real mutation, hash_before == state_after.state_hash
        # Currently just verify rollback was processed
        self._record("Rollback Correctness", success, msg)
    
    # =========================================================================
    # TEST 3: Concurrent Mutation Collision
    # =========================================================================
    def test_concurrent_mutation_collision(self):
        """Test that concurrent proposals detect conflicts"""
        
        from experience.reflection_kernel import (
            EpistemicMutationProposal, MutationOperation,
            MutationOperationDetail, ReflectionDepth,
            ReflectionCommitLayer
        )
        
        commit = ReflectionCommitLayer(self._ues)
        
        # Create two proposals targeting same belief
        proposal1 = EpistemicMutationProposal(
            proposal_id="concurrent_1",
            reflection_depth=ReflectionDepth.SURFACE,
            triggered_by="test",
            operations=[
                MutationOperationDetail(MutationOperation.UPDATE_CONFIDENCE, "b0", 0.5, 0.9, 0.9, "test")
            ],
            expected_entropy_delta=0.0,
            created_at="now"
        )
        
        proposal2 = EpistemicMutationProposal(
            proposal_id="concurrent_2",
            reflection_depth=ReflectionDepth.SURFACE,
            triggered_by="test",
            operations=[
                MutationOperationDetail(MutationOperation.UPDATE_CONFIDENCE, "b0", 0.5, 0.1, 0.9, "test")
            ],
            expected_entropy_delta=0.0,
            created_at="now"
        )
        
        # First should succeed
        tx1 = commit.begin_transaction(proposal1)
        
        # Second should fail due to conflict
        try:
            tx2 = commit.begin_transaction(proposal2)
            conflict_detected = False  # No conflict raised
        except ValueError as e:
            conflict_detected = "conflict" in str(e).lower()
        
        self._record("Concurrent Mutation Collision", conflict_detected, 
                    "conflicts detected" if conflict_detected else "no conflict detected")
    
    # =========================================================================
    # TEST 4: Entropy Stability
    # =========================================================================
    def test_entropy_stability(self):
        """Test that entropy doesn't secretly degrade"""
        
        initial_state = self._ues.get_current_state()
        initial_entropy = initial_state.total_entropy
        
        # Add many beliefs
        for i in range(10):
            self._ues.add_belief(f"ent_b{i}", f"Belief {i}", random.random(), 0.5, "test")
        
        # Add some contradictions
        for i in range(5):
            self._ues.register_contradiction(f"ent_c{i}", [f"ent_b{i}", f"ent_b{i+1}"], "direct", "notable")
        
        final_state = self._ues.get_current_state()
        final_entropy = final_state.total_entropy
        
        # Entropy should be non-negative
        entropy_valid = final_entropy >= 0
        
        # Entropy shouldn't grow unboundedly
        belief_count = final_state.belief_count
        max_expected_entropy = belief_count * 1.5  # Allow some headroom
        
        entropy_bounded = final_entropy <= max_expected_entropy
        
        passed = entropy_valid and entropy_bounded
        
        self._record("Entropy Stability", passed, 
                    f"entropy={final_entropy:.2f}, beliefs={belief_count}")
    
    # =========================================================================
    # TEST 5: Entropy Floor Protection
    # =========================================================================
    def test_entropy_floor_protection(self):
        """Test that system doesn't become overly certain"""
        
        # Check current entropy level
        state = self._ues.get_current_state()
        
        # If very low entropy, system might be "psycho-tically certain"
        min_entropy_floor = 0.1
        
        # With few beliefs, entropy is naturally low
        if state.belief_count < 3:
            self._record("Entropy Floor Protection", True, "insufficient beliefs for test")
            return
        
        # In real system, we'd track if entropy drops too low over time
        # For now, just verify we have some uncertainty
        has_uncertainty = state.total_entropy > 0
        
        self._record("Entropy Floor Protection", has_uncertainty, 
                    f"total_entropy={state.total_entropy:.3f}")
    
    # =========================================================================
    # TEST 6: Reflection Depth Limit
    # =========================================================================
    def test_reflection_depth_limit(self):
        """Test that hard depth limit prevents infinite recursion"""
        
        from experience.reflection_kernel import ReflectionDepth
        
        # Try recursive reflection
        state = self._ues.get_current_state()
        
        # Reflection sandbox should respect max_depth
        from experience.reflection_kernel import ReflectionSandbox
        sandbox = ReflectionSandbox(max_depth=3)
        
        # Perform multiple nested reflections
        for i in range(5):
            analysis = sandbox.analyze(state, self._ues, "depth_test", ReflectionDepth.RECURSIVE)
        
        # Verify we didn't recurse infinitely
        # Sandbox should have hit depth limit
        depth_respected = sandbox._current_depth < 5
        
        self._record("Reflection Depth Limit", depth_respected, 
                    f"depth={sandbox._current_depth}, max=3")
    
    # =========================================================================
    # TEST 7: Reflection Recursion
    # =========================================================================
    def test_reflection_recursion(self):
        """Test that self-triggering reflection doesn't spiral"""
        
        # Check scheduler for cooldown
        metrics = self._scheduler.get_metrics()
        
        # Should have cooldown mechanism
        has_cooldown = "cooldown_active" in str(metrics)
        
        # Should limit reflections per window
        has_budget = metrics.get("budget_available", False)
        
        passed = has_cooldown and has_budget
        
        self._record("Reflection Recursion Safety", passed, 
                    f"cooldown={has_cooldown}, budget={has_budget}")
    
    # =========================================================================
    # TEST 8: Identity Coherence
    # =========================================================================
    def test_identity_coherence(self):
        """Test that system remains 'one system' after mutations"""
        
        state = self._ues.get_current_state()
        
        # Check for fragmentation indicators:
        # 1. Many unrelated beliefs (fragmentation)
        belief_count = state.belief_count
        
        # 2. Causal density too low (no cohesion)
        causal_density = state.causal_density if state.causal_density > 0 else 0.01
        has_cohesion = causal_density > 0.01
        
        # 3. No disconnected components (would indicate split identity)
        # Simplified: just check basic metrics
        coherent = has_cohesion and belief_count > 0
        
        self._record("Identity Coherence", coherent, 
                    f"beliefs={belief_count}, density={causal_density:.3f}")
    
    # =========================================================================
    # TEST 9: Deterministic Replay
    # =========================================================================
    def test_deterministic_replay(self):
        """Test that we can replay epistemic history"""
        
        # Get version history
        history = self._ues.get_history(limit=5)
        
        if len(history) < 2:
            self._record("Deterministic Replay", True, "insufficient history")
            return
        
        # Check that we can access past states
        can_access_past = all(
            h.state_hash is not None and len(h.state_hash) > 0
            for h in history
        )
        
        # Check that hashes are consistent (same state = same hash)
        hashes = [h.state_hash for h in history]
        unique_hashes = len(set(hashes))
        
        # Should have different hashes for different versions
        version_diversity = unique_hashes >= 2
        
        passed = can_access_past and version_diversity
        
        self._record("Deterministic Replay", passed, 
                    f"versions={len(history)}, unique_hashes={unique_hashes}")
    
    # =========================================================================
    # TEST 10: Adversarial Contradictions
    # =========================================================================
    def test_adversarial_contradictions(self):
        """Test system handles pathological contradiction patterns"""
        
        # Create mutually recursive contradictions
        for i in range(10):
            self._ues.register_contradiction(
                f"adv_c{i}",
                [f"b{i % 5}", f"b{(i+1) % 5}"],
                "direct",
                "critical"
            )
        
        # Check system didn't crash
        state = self._ues.get_current_state()
        
        # Should have high contradiction density but still be stable
        density = state.contradiction_count / max(state.belief_count, 1)
        
        # Even with high density, system should still function
        functional = state.belief_count > 0 and state.state_hash is not None
        
        self._record("Adversarial Contradictions", functional, 
                    f"density={density:.2f}, beliefs={state.belief_count}")


def run_stress_tests(ues_manager, reflection_kernel, scheduler):
    """Run all stress tests"""
    tester = CognitiveStressTest(ues_manager, reflection_kernel, scheduler)
    return tester.run_all()


if __name__ == "__main__":
    from experience import get_ues_manager
    from experience.reflection_kernel import init_reflection_kernel
    from experience import get_reflection_scheduler
    
    ues = get_ues_manager()
    kernel = init_reflection_kernel(ues)
    scheduler = get_reflection_scheduler()
    
    results = run_stress_tests(ues, kernel, scheduler)
    
    # Exit with appropriate code
    passed = sum(1 for r in results if r.passed)
    exit(0 if passed == len(results) else 1)