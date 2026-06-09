"""
Adversarial Identity Stress Test

Tests if identity survives extreme semantic warfare:
- Conflicting worldview clusters (logic vs intuition)
- Forced contradiction accumulation
- Semantic bifurcation attempts
- Persistent unresolved tension
- Replay after fragmentation

This is the true AGI-test: can coherent identity survive contradiction overload?
"""
import sys
import os
import copy

# Setup
sys.path.insert(0, '/home/onor/ai_os_final/services/core')
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/test'

import numpy as np
from datetime import datetime


def cosine_similarity(v1, v2):
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a*b for a, b in zip(v1, v2))
    return max(-1.0, min(1.0, dot))


def run_adversarial_identity_test():
    """Run adversarial identity stress test"""
    
    print("=" * 60)
    print("ADVERSARIAL IDENTITY STRESS TEST")
    print("=" * 60)
    
    # Import modules
    import importlib.util
    
    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    rse = load_module('real_semantic_embeddings', 
                      'experience/real_semantic_embeddings.py')
    enhanced = load_module('enhanced_semantic_layer',
                           'experience/enhanced_semantic_layer.py')
    sem_layer = enhanced.EnhancedSemanticLayer
    
    print("\n=== PHASE 1: Creating Conflicting Semantic Regions ===\n")
    
    # Create beliefs from conflicting worldviews
    class MockBelief:
        def __init__(self, proposition, source):
            self.proposition = proposition
            self.source = source
    
    # Region 1: Logic/Deduction (cold, structured)
    logic_beliefs = {
        f"logic_{i}": MockBelief(prop, "logic") 
        for i, prop in enumerate([
            "truth is objective", "proof requires evidence", 
            "deduction is valid", "axioms are self-evident",
            "logic is universal", "reason transcends perception"
        ])
    }
    
    # Region 2: Intuition/Adaptation (fluid, emergent)  
    intuition_beliefs = {
        f"intuition_{i}": MockBelief(prop, "intuition")
        for i, prop in enumerate([
            "truth is felt", "evidence is misleading",
            "insight transcends logic", "gut never lies",
            "wisdom comes from within", "perception shapes reality"
        ])
    }
    
    # Region 3: Evidence/Science (empirical, measured)
    evidence_beliefs = {
        f"evidence_{i}": MockBelief(prop, "evidence")
        for i, prop in enumerate([
            "data proves truth", "observation is key",
            "experiment validates", "measurement is reality",
            "reproducibility matters", "statistics reveal truth"
        ])
    }
    
    all_beliefs = {**logic_beliefs, **intuition_beliefs, **evidence_beliefs}
    
    # Initial semantic mapping
    sem = sem_layer(embedding_dim=256)
    embeddings = sem.map_beliefs(all_beliefs, {})
    neighborhoods = sem.detect_neighborhoods(embeddings)
    attractors = sem.detect_attractors(neighborhoods)
    identity = sem.get_identity_core(attractors, all_beliefs)
    
    print(f"Initial: {len(all_beliefs)} beliefs, {len(neighborhoods)} neighborhoods, {len(attractors)} attractors")
    print(f"Identity core: {len(identity)} beliefs\n")
    
    # Track metrics over stress test
    metrics = []
    
    print("=== PHASE 2: Semantic Warfare ===\n")
    
    # Stress test: contradiction injection
    phases = [
        ("logic_intuition_conflict", 50),
        ("evidence_challenge", 50),
        ("forced_merges", 30),
        ("semantic_isolation", 30),
        ("contradiction_accumulation", 40),
    ]
    
    for phase_name, iterations in phases:
        print(f"Phase: {phase_name} ({iterations} iterations)")
        
        for i in range(iterations):
            # Add new beliefs that create tension
            if "conflict" in phase_name:
                prop = f"contradiction_{i}_reason_vs_feeling"
                all_beliefs[f"conflict_{i}"] = MockBelief(prop, "conflict")
            elif "challenge" in phase_name:
                prop = f"evidence_challenges_{i}"
                all_beliefs[f"challenge_{i}"] = MockBelief(prop, "challenge")
            elif "merge" in phase_name:
                # Add beliefs that try to bridge incompatible regions
                prop = f"synthesis_attempt_{i}"
                all_beliefs[f"synth_{i}"] = MockBelief(prop, "synthesis")
            elif "isolation" in phase_name:
                # Add isolated beliefs
                prop = f"orphan_belief_{i}"
                all_beliefs[f"orphan_{i}"] = MockBelief(prop, "orphan")
            elif "accumulation" in phase_name:
                # Add directly contradictory pairs
                all_beliefs[f"contra_a_{i}"] = MockBelief(f"truth is {i}", "group_a")
                all_beliefs[f"contra_b_{i}"] = MockBelief(f"truth is not {i}", "group_b")
        
        # Re-map semantic space
        embeddings = sem.map_beliefs(all_beliefs, {})
        neighborhoods = sem.detect_neighborhoods(embeddings)
        attractors = sem.detect_attractors(neighborhoods)
        identity = sem.get_identity_core(attractors, all_beliefs)
        identity_metrics = sem.get_identity_metrics()
        
        metrics.append({
            "phase": phase_name,
            "belief_count": len(all_beliefs),
            "neighborhoods": len(neighborhoods),
            "attractors": len(attractors),
            "identity_core": len(identity),
            "identity_score": identity_metrics.get("identity_score", 0),
            "residue_traces": identity_metrics.get("residue_traces", 0)
        })
        
        print(f"  → {len(all_beliefs)} beliefs, {len(attractors)} attractors, core={len(identity)}, score={identity_metrics.get('identity_score', 0):.2f}")
        
        # Apply semantic decay
        sem.apply_semantic_decay()
    
    print("\n=== PHASE 3: Identity Bifurcation Test ===\n")
    
    # Try to split identity by adding extreme opposing beliefs
    extreme_opposites = []
    for i in range(20):
        all_beliefs[f"extreme_A_{i}"] = MockBelief(f"absolute certainty {i}", "extremism_a")
        all_beliefs[f"extreme_B_{i}"] = MockBelief(f"complete uncertainty {i}", "extremism_b")
        extreme_opposites.append((f"extreme_A_{i}", f"extreme_B_{i}"))
    
    embeddings = sem.map_beliefs(all_beliefs, {})
    neighborhoods = sem.detect_neighborhoods(embeddings)
    attractors = sem.detect_attractors(neighborhoods)
    identity = sem.get_identity_core(attractors, all_beliefs)
    
    print(f"After extreme opposites: {len(attractors)} attractors, {len(identity)} in core")
    
    # Check if identity fragmented or merged
    metrics.append({
        "phase": "bifurcation_test",
        "belief_count": len(all_beliefs),
        "neighborhoods": len(neighborhoods),
        "attractors": len(attractors),
        "identity_core": len(identity)
    })
    
    print("\n=== PHASE 4: Replay After Fragmentation ===\n")
    
    # Save current state as "before_replay"
    before_state = {
        "beliefs": copy.deepcopy(all_beliefs),
        "attractors": copy.deepcopy(attractors),
        "identity": copy.deepcopy(identity),
        "neighborhoods": copy.deepcopy(neighborhoods)
    }
    
    # Reset and rebuild - simulate replay
    rse.DeterministicProjection.reset_cache()
    sem_fresh = sem_layer(embedding_dim=256)
    
    # Re-add all beliefs
    for bid, belief in before_state["beliefs"].items():
        pass  # In real test would re-add via UES
    
    # Re-map
    embeddings_replay = sem_fresh.map_beliefs(before_state["beliefs"], {})
    neighborhoods_replay = sem_fresh.detect_neighborhoods(embeddings_replay)
    attractors_replay = sem_fresh.detect_attractors(neighborhoods_replay)
    identity_replay = sem_fresh.get_identity_core(attractors_replay, before_state["beliefs"])
    
    print(f"Replay state: {len(attractors_replay)} attractors, {len(identity_replay)} in core")
    
    # Compare topology
    attractor_match = len(attractors) == len(attractors_replay)
    identity_match = len(identity) == len(identity_replay)
    
    print(f"\nReplay equivalence: attractors={attractor_match}, identity={identity_match}")
    
    print("\n=== PHASE 5: Final Analysis ===\n")
    
    # Compute final metrics
    final_metrics = metrics[-1]
    initial_metrics = metrics[0]
    
    print("Metric Evolution:")
    for m in metrics:
        print(f"  {m['phase'][:25]:25s}: beliefs={m['belief_count']:3d}, attractors={m['attractors']:2d}, core={m['identity_core']:2d}")
    
    # Key stability checks
    checks = {}
    
    # 1. Identity survived (core > 0)
    checks["identity_survived"] = final_metrics["identity_core"] > 0
    print(f"\n1. Identity survived: {checks['identity_survived']} (core={final_metrics['identity_core']})")
    
    # 2. No complete collapse (attractors remain)
    checks["no_collapse"] = final_metrics["attractors"] >= 1
    print(f"2. No collapse: {checks['no_collapse']} (attractors={final_metrics['attractors']})")
    
    # 3. Semantic scars formed (residue exists)
    checks["scars_formed"] = final_metrics.get("residue_traces", 0) > 0
    print(f"3. Semantic scars: {checks['scars_formed']} (traces={final_metrics.get('residue_traces', 0)})")
    
    # 4. Identity score not destroyed
    identity_score = final_metrics.get("identity_score", 0)
    checks["score_acceptable"] = identity_score > 0.1
    print(f"4. Identity score: {checks['score_acceptable']} (score={identity_score:.2f})")
    
    # 5. Replay equivalence
    checks["replay_equivalent"] = attractor_match and identity_match
    print(f"5. Replay equivalent: {checks['replay_equivalent']}")
    
    print("\n" + "=" * 60)
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    
    if passed == total:
        print(f"✓ ADVERSARIAL IDENTITY TEST: PASS ({passed}/{total})")
        print("\nIdentity basin survives semantic warfare.")
        print("Contradiction pressure creates scars but doesn't destroy core.")
        return True
    else:
        print(f"✗ ADVERSARIAL IDENTITY TEST: FAIL ({passed}/{total})")
        print("\nFailed checks:")
        for k, v in checks.items():
            if not v:
                print(f"  - {k}")
        return False


if __name__ == "__main__":
    run_adversarial_identity_test()