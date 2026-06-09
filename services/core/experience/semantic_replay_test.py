"""
Semantic Replay Equivalence Test

Verifies that WAL replay produces identical semantic physics:
- Attractor topology
- Identity basin continuity  
- Semantic pressure field
- Neighborhood graph

This is the core validation for cognitive substrate determinism.
"""
import sys
import os

# Set up paths
app_path = '/app'
if app_path not in sys.path:
    sys.path.insert(0, app_path)
experience_path = os.path.join(app_path, 'experience')
if experience_path not in sys.path:
    sys.path.insert(0, experience_path)

os.chdir(app_path)
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/test'

from datetime import datetime
import copy
import json


def run_semantic_replay_test():
    """Run semantic replay equivalence test"""
    
    print("=== SEMANTIC REPLAY EQUIVALENCE TEST ===\n")
    
    # Reset all state
    import experience.wal_engine
    import experience.unified_epistemic_state
    import experience.reflection_kernel
    import experience.reflection_scheduler
    import experience.contradiction_engine
    import experience.enhanced_semantic_layer
    
    experience.wal_engine._wal_engine = None
    experience.unified_epistemic_state._ues_manager = None
    experience.reflection_kernel._kernel = None
    experience.reflection_scheduler._scheduler = None
    experience.contradiction_engine._contradiction_engine = None
    experience.enhanced_semantic_layer._enhanced_semantic = None
    
    # Import fresh
    from wal_engine import get_wal_engine, reset_wal_engine
    reset_wal_engine()
    
    from unified_epistemic_state import get_ues_manager
    from enhanced_semantic_layer import get_enhanced_semantic, reset_enhanced_semantic
    from contradiction_engine import get_contradiction_engine, reset_contradiction_engine
    
    # Initialize
    reset_contradiction_engine()
    reset_enhanced_semantic()
    
    ues = get_ues_manager()
    sem = get_enhanced_semantic()
    wal = get_wal_engine()
    ce = get_contradiction_engine()
    
    print("Phase 1: Creating initial semantic state...\n")
    
    # Create diverse belief set to form semantic regions
    regions = {
        "logic": ["truth", "proof", "deduction", "syllogism", "axiom"],
        "intuition": ["feeling", "sense", "gut", "instinct", "premonition"],
        "evidence": ["data", "observation", "measurement", "experiment", "fact"],
        "uncertainty": ["maybe", "perhaps", "possibly", "probably", "likely"]
    }
    
    belief_id = 0
    for region, propositions in regions.items():
        for prop in propositions:
            ues.add_belief(f"b_{belief_id}", prop, 0.5 + (belief_id % 10) / 20.0, 0.5, region)
            belief_id += 1
    
    # Add causal edges within regions
    state = ues.get_current_state()
    bids = list(state.beliefs.keys())
    
    # Logic region edges
    for i in range(4):
        ues.add_causal_edge([bids[i]], [bids[i+1]], 0.6, 0.5)
    
    # Intuition region edges  
    for i in range(5, 9):
        ues.add_causal_edge([bids[i]], [bids[i+1]], 0.4, 0.5)
    
    # Evidence region edges
    for i in range(10, 14):
        ues.add_causal_edge([bids[i]], [bids[i+1]], 0.7, 0.5)
    
    # Get initial semantic state
    state = ues.get_current_state()
    embeddings = sem.map_beliefs(state.beliefs, state.causal_edges)
    neighborhoods = sem.detect_neighborhoods(embeddings)
    attractors = sem.detect_attractors(neighborhoods)
    identity = sem.get_identity_core(attractors, state.beliefs)
    pressure = sem.compute_semantic_pressure(embeddings, neighborhoods, attractors)
    
    print(f"Initial state: {len(state.beliefs)} beliefs, {len(neighborhoods)} neighborhoods, {len(attractors)} attractors")
    print(f"Identity core size: {len(identity)}")
    print(f"Pressure: {pressure}\n")
    
    # Capture initial semantic snapshot
    initial_snapshot = {
        "attractor_count": len(attractors),
        "neighborhood_count": len(neighborhoods),
        "identity_core_size": len(identity),
        "pressure": copy.deepcopy(pressure),
        "attractor_centers": [
            list(a["center_vector"][:5]) for a in attractors.values()
        ],
        "neighborhood_members": {
            nid: list(n["members"]) for nid, n in neighborhoods.items()
        }
    }
    
    print("Phase 2: Running mutations...\n")
    
    # Run 500 mutations
    for tick in range(500):
        mutation = tick % 6
        
        if mutation == 0:
            ues.add_belief(f"mut_{tick}", f"mutation_{tick}", 0.6, 0.5, "test")
        elif mutation == 1:
            state = ues.get_current_state()
            if state.beliefs:
                bid = list(state.beliefs.keys())[tick % len(state.beliefs)]
                ues.update_belief(bid, 0.3 + (tick % 50) / 100.0, 0.5)
        elif mutation == 2:
            state = ues.get_current_state()
            bids = list(state.beliefs.keys())
            if len(bids) >= 2:
                ues.add_causal_edge([bids[tick % len(bids)]], [bids[(tick+1) % len(bids)]], 0.5, 0.5)
        elif mutation == 3:
            state = ues.get_current_state()
            bids = list(state.beliefs.keys())
            if len(bids) >= 2:
                ues.register_contradiction(f"c_{tick}", [bids[0], bids[1]], "semantic_divergence", "medium")
        elif mutation == 4:
            state = ues.get_current_state()
            if state.causal_edges:
                edge_id = list(state.causal_edges.keys())[tick % len(state.causal_edges)]
                ues.remove_causal_edge(edge_id)
        else:
            state = ues.get_current_state()
            bids = list(state.beliefs.keys())
            if bids:
                ues.remove_belief(bids[tick % len(bids)])
    
    print(f"Mutations complete: version {ues._current_version}")
    
    # Get post-mutation semantic state
    state = ues.get_current_state()
    embeddings2 = sem.map_beliefs(state.beliefs, state.causal_edges)
    neighborhoods2 = sem.detect_neighborhoods(embeddings2)
    attractors2 = sem.detect_attractors(neighborhoods2)
    identity2 = sem.get_identity_core(attractors2, state.beliefs)
    pressure2 = sem.compute_semantic_pressure(embeddings2, neighborhoods2, attractors2)
    
    print(f"\nPost-mutation: {len(state.beliefs)} beliefs, {len(neighborhoods2)} neighborhoods, {len(attractors2)} attractors")
    print(f"Identity core size: {len(identity2)}")
    
    # Capture post-mutation snapshot
    post_mutation_snapshot = {
        "attractor_count": len(attractors2),
        "neighborhood_count": len(neighborhoods2),
        "identity_core_size": len(identity2),
        "pressure": copy.deepcopy(pressure2),
        "attractor_centers": [
            list(a["center_vector"][:5]) for a in attractors2.values()
        ],
        "version": ues._current_version,
        "belief_count": len(state.beliefs)
    }
    
    print("\nPhase 3: Capturing WAL for replay...\n")
    
    # Get WAL entries for replay
    wal_entries = wal.get_all_entries()
    print(f"WAL entries captured: {len(wal_entries)}")
    
    # Create fresh state for replay
    print("\nPhase 4: Replaying events in fresh instance...\n")
    
    # Reset again - fresh start
    experience.wal_engine._wal_engine = None
    experience.unified_epistemic_state._ues_manager = None
    experience.reflection_kernel._kernel = None
    experience.reflection_scheduler._scheduler = None
    experience.contradiction_engine._contradiction_engine = None
    experience.enhanced_semantic_layer._enhanced_semantic = None
    
    reset_wal_engine()
    ues_replay = get_ues_manager()
    sem_replay = get_enhanced_semantic()
    wal_replay = get_wal_engine()
    
    # Replay all WAL entries
    for entry in wal_entries:
        if entry.event_type == "belief_added":
            ues_replay.add_belief(
                entry.payload["belief_id"],
                entry.payload["proposition"],
                entry.payload["support"],
                entry.payload["confidence"],
                entry.payload.get("source", "replay")
            )
        elif entry.event_type == "belief_updated":
            ues_replay.update_belief(
                entry.payload["belief_id"],
                entry.payload["support"],
                entry.payload["confidence"]
            )
        elif entry.event_type == "belief_removed":
            ues_replay.remove_belief(entry.payload["belief_id"])
        elif entry.event_type == "causal_edge_added":
            ues_replay.add_causal_edge(
                entry.payload["causes"],
                entry.payload["effects"],
                entry.payload["strength"],
                entry.payload["confidence"]
            )
        elif entry.event_type == "causal_edge_removed":
            ues_replay.remove_causal_edge(entry.payload["edge_id"])
        elif entry.event_type == "contradiction_registered":
            ues_replay.register_contradiction(
                entry.payload["contradiction_id"],
                entry.payload["belief_ids"],
                entry.payload["contradiction_type"],
                entry.payload["severity"]
            )
    
    print(f"Replay complete: version {ues_replay._current_version}")
    
    # Get replay semantic state
    state_replay = ues_replay.get_current_state()
    embeddings_replay = sem_replay.map_beliefs(state_replay.beliefs, state_replay.causal_edges)
    neighborhoods_replay = sem_replay.detect_neighborhoods(embeddings_replay)
    attractors_replay = sem_replay.detect_attractors(neighborhoods_replay)
    identity_replay = sem_replay.get_identity_core(attractors_replay, state_replay.beliefs)
    pressure_replay = sem_replay.compute_semantic_pressure(embeddings_replay, neighborhoods_replay, attractors_replay)
    
    print(f"\nReplay state: {len(state_replay.beliefs)} beliefs, {len(neighborhoods_replay)} neighborhoods, {len(attractors_replay)} attractors")
    print(f"Identity core size: {len(identity_replay)}")
    
    # Capture replay snapshot
    replay_snapshot = {
        "attractor_count": len(attractors_replay),
        "neighborhood_count": len(neighborhoods_replay),
        "identity_core_size": len(identity_replay),
        "pressure": copy.deepcopy(pressure_replay),
        "attractor_centers": [
            list(a["center_vector"][:5]) for a in attractors_replay.values()
        ],
        "version": ues_replay._current_version,
        "belief_count": len(state_replay.beliefs)
    }
    
    print("\n=== EQUIVALENCE ANALYSIS ===\n")
    
    results = {}
    
    # 1. Version match
    results["version_match"] = post_mutation_snapshot["version"] == replay_snapshot["version"]
    print(f"1. Version match: {results['version_match']} ({post_mutation_snapshot['version']} vs {replay_snapshot['version']})")
    
    # 2. Belief count match
    results["belief_count_match"] = post_mutation_snapshot["belief_count"] == replay_snapshot["belief_count"]
    print(f"2. Belief count: {results['belief_count_match']} ({post_mutation_snapshot['belief_count']} vs {replay_snapshot['belief_count']})")
    
    # 3. Attractor count match (topology)
    results["attractor_match"] = post_mutation_snapshot["attractor_count"] == replay_snapshot["attractor_count"]
    print(f"3. Attractor count: {results['attractor_match']} ({post_mutation_snapshot['attractor_count']} vs {replay_snapshot['attractor_count']})")
    
    # 4. Neighborhood count match
    results["neighborhood_match"] = post_mutation_snapshot["neighborhood_count"] == replay_snapshot["neighborhood_count"]
    print(f"4. Neighborhood count: {results['neighborhood_match']} ({post_mutation_snapshot['neighborhood_count']} vs {replay_snapshot['neighborhood_count']})")
    
    # 5. Identity core size match
    results["identity_match"] = post_mutation_snapshot["identity_core_size"] == replay_snapshot["identity_core_size"]
    print(f"5. Identity core: {results['identity_match']} ({post_mutation_snapshot['identity_core_size']} vs {replay_snapshot['identity_core_size']})")
    
    # 6. Attractor centroid drift (epsilon comparison)
    epsilon = 0.001
    centroid_diffs = []
    for i in range(min(len(post_mutation_snapshot["attractor_centers"]), 
                       len(replay_snapshot["attractor_centers"]))):
        diff = sum(abs(a - b) for a, b in zip(
            post_mutation_snapshot["attractor_centers"][i],
            replay_snapshot["attractor_centers"][i]
        ))
        centroid_diffs.append(diff)
    
    avg_centroid_drift = sum(centroid_diffs) / max(len(centroid_diffs), 1) if centroid_diffs else 0
    results["centroid_drift_acceptable"] = avg_centroid_drift < epsilon
    print(f"6. Centroid drift: {results['centroid_drift_acceptable']} (avg drift: {avg_centroid_drift:.6f})")
    
    # 7. Pressure field equivalence
    pressure_epsilon = 0.01
    pressure_diffs = {}
    for key in post_mutation_snapshot["pressure"]:
        if key in replay_snapshot["pressure"]:
            diff = abs(post_mutation_snapshot["pressure"][key] - replay_snapshot["pressure"][key])
            pressure_diffs[key] = diff
    
    max_pressure_diff = max(pressure_diffs.values()) if pressure_diffs else 0
    results["pressure_equivalent"] = max_pressure_diff < pressure_epsilon
    print(f"7. Pressure field: {results['pressure_equivalent']} (max diff: {max_pressure_diff:.6f})")
    
    # Summary
    print("\n=== RESULT ===\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    if passed == total:
        print(f"✓ SEMANTIC REPLAY EQUIVALENCE: PASS ({passed}/{total})")
        print("\nThe cognitive substrate produces deterministic semantic physics.")
        print("WAL replay preserves topology, identity, and pressure field.")
        return True
    else:
        print(f"✗ SEMANTIC REPLAY EQUIVALENCE: FAIL ({passed}/{total})")
        print("\nFailed checks:")
        for k, v in results.items():
            if not v:
                print(f"  - {k}")
        return False


if __name__ == "__main__":
    run_semantic_replay_test()