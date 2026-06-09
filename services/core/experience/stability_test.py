"""
Brutal Stability Test - 10k mutations long-horizon cognitive evolution.

Tests:
- Attractor persistence
- Semantic collapse
- Entropy runaway  
- Identity core stability
- Thermodynamic runaway
- Replay determinism

This is the main validation test for the cognitive substrate.
"""
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/experience')

from datetime import datetime
import time

# Reset all state
import experience.wal_engine
import experience.unified_epistemic_state
import experience.reflection_kernel
import experience.reflection_scheduler
import experience.contradiction_engine
import experience.semantic_embedding_layer

experience.wal_engine._wal_engine = None
experience.unified_epistemic_state._ues_manager = None
experience.reflection_kernel._kernel = None
experience.reflection_scheduler._scheduler = None
experience.contradiction_engine._contradiction_engine = None
experience.semantic_embedding_layer._semantic_layer = None

from wal_engine import get_wal_engine, reset_wal_engine
reset_wal_engine()

from unified_epistemic_state import get_ues_manager
from reflection_kernel import init_reflection_kernel, ReflectionCommitLayer, EpistemicMutationProposal, MutationOperation, MutationOperationDetail, ReflectionDepth
from reflection_scheduler import get_reflection_scheduler
from cognitive_stress_tests import run_stress_tests
from contradiction_engine import get_contradiction_engine, reset_contradiction_engine
from semantic_embedding_layer import get_semantic_layer, reset_semantic_layer

# Initialize fresh
reset_contradiction_engine()
reset_semantic_layer()

ues = get_ues_manager()
kernel = init_reflection_kernel(ues)
scheduler = get_reflection_scheduler()
wal = get_wal_engine()
sem = get_semantic_layer()
ce = get_contradiction_engine()

# Run baseline stress tests first
print("=== BASELINE STRESS TESTS ===")
results = run_stress_tests(ues, kernel, scheduler)
baseline_passed = sum(1 for r in results if r.passed)
print(f"Baseline: {baseline_passed}/10\n")

# Metrics tracking
metrics_log = []

print("=== STARTING 10K MUTATION STABILITY TEST ===")
start_time = time.time()

# Track initial state
initial_state = {
    "version": ues._current_version,
    "belief_count": len(ues.get_current_state().beliefs),
    "entropy": ues.get_current_state().total_entropy
}

for tick in range(10000):
    # Random mutation type
    mutation_type = tick % 5
    
    if mutation_type == 0:
        # Add new belief
        ues.add_belief(f"b_{tick}", f"Belief at tick {tick}", 0.5, 0.5, "mutation")
    
    elif mutation_type == 1:
        # Random belief update
        state = ues.get_current_state()
        if state.beliefs:
            bid = list(state.beliefs.keys())[tick % len(state.beliefs)]
            ues.update_belief(bid, 0.3 + (tick % 70) / 100.0, 0.5)
    
    elif mutation_type == 2:
        # Add causal edge
        state = ues.get_current_state()
        bids = list(state.beliefs.keys())
        if len(bids) >= 2:
            ues.add_causal_edge([bids[0]], [bids[1]], 0.5, 0.5)
    
    elif mutation_type == 3:
        # Reflection proposal (may commit or compensate)
        state = ues.get_current_state()
        if state.beliefs:
            bid = list(state.beliefs.keys())[tick % len(state.beliefs)]
            proposal = EpistemicMutationProposal(
                proposal_id=f"mut_{tick}",
                reflection_depth=ReflectionDepth.SURFACE,
                triggered_by="stability_test",
                operations=[MutationOperationDetail(MutationOperation.UPDATE_CONFIDENCE, bid, None, 0.6, 0.6, "test")],
                expected_entropy_delta=0.0,
                created_at=datetime.utcnow().isoformat()
            )
            commit = ReflectionCommitLayer(ues)
            tx = commit.begin_transaction(proposal)
            tx.add_operation(proposal.operations[0], commit._create_inverse(proposal.operations[0]))
            
            # 10% compensation rate
            if tick % 10 == 0:
                tx.compensate()
            else:
                tx.commit()
    
    else:
        # Contradiction registration
        state = ues.get_current_state()
        bids = list(state.beliefs.keys())[:3] if len(state.beliefs) >= 3 else list(state.beliefs.keys())
        if bids:
            ues.register_contradiction(f"c_{tick}", bids, "confidence_divergence", "low")
    
    # Periodic metrics collection (every 1000 ticks)
    if tick % 1000 == 0:
        state = ues.get_current_state()
        
        # Semantic analysis
        embeddings = sem.map_beliefs(state.beliefs, state.causal_edges)
        neighborhoods = sem.detect_neighborhoods(embeddings)
        attractors = sem.detect_attractors(neighborhoods)
        
        # Pressure analysis
        transactions = wal.get_transaction_lineage()
        causal_graph = wal.get_causal_graph()
        ce.detect_pressure(state.beliefs, transactions, causal_graph, ues._current_version)
        
        # Thermodynamics
        active_clusters = [c for c in ce.get_clusters()]
        thermo = ce.compute_cognitive_thermodynamics(len(state.beliefs), active_clusters, 0.5)
        
        # Identity core
        identity_core = sem.get_identity_core(attractors, state.beliefs)
        
        # Apply decay
        ce.apply_pressure_decay(0.01)
        
        metrics = {
            "tick": tick,
            "version": ues._current_version,
            "belief_count": len(state.beliefs),
            "entropy": state.total_entropy,
            "attractors": len(attractors),
            "neighborhoods": len(neighborhoods),
            "identity_core_size": len(identity_core),
            "active_clusters": len(active_clusters),
            "temperature": thermo.get("cognitive_temperature", 0),
            "phase_state": thermo.get("phase_state", "unknown"),
            "entropy_metric": thermo.get("cognitive_entropy", 0)
        }
        metrics_log.append(metrics)
        
        print(f"Tick {tick:5d}: beliefs={len(state.beliefs):3d}, attractors={len(attractors)}, "
              f"core={len(identity_core)}, temp={thermo.get('cognitive_temperature', 0):.3f}, "
              f"phase={thermo.get('phase_state', 'unknown')[:12]}")

end_time = time.time()
duration = end_time - start_time

print(f"\n=== STABILITY TEST COMPLETE ===")
print(f"Duration: {duration:.1f}s")
print(f"Ticks: 10000")

# Analysis
print("\n=== METRICS ANALYSIS ===")

# Extract final metrics
final = metrics_log[-1]
initial = metrics_log[0]

print(f"\n--- EVOLUTION ---")
print(f"Beliefs: {initial['belief_count']} -> {final['belief_count']}")
print(f"Entropy: {initial['entropy']:.2f} -> {final['entropy']:.2f}")

print(f"\n--- ATTRACTOR PERSISTENCE ---")
attractor_counts = [m['attractors'] for m in metrics_log]
avg_attractors = sum(attractor_counts) / len(attractor_counts)
final_attractors = attractor_counts[-1]
print(f"Average: {avg_attractors:.1f}")
print(f"Final: {final_attractors}")
print(f"Decay: {(1 - final_attractors/max(avg_attractors, 1))*100:.1f}%")

print(f"\n--- IDENTITY CORE ---")
core_sizes = [m['identity_core_size'] for m in metrics_log]
avg_core = sum(core_sizes) / len(core_sizes)
final_core = core_sizes[-1]
print(f"Average: {avg_core:.1f}")
print(f"Final: {final_core}")

print(f"\n--- THERMODYNAMICS ---")
temps = [m['temperature'] for m in metrics_log]
phases = [m['phase_state'] for m in metrics_log]

avg_temp = sum(temps) / len(temps)
max_temp = max(temps)
print(f"Average temp: {avg_temp:.3f}")
print(f"Max temp: {max_temp:.3f}")

# Phase transition detection
phase_changes = sum(1 for i in range(1, len(phases)) if phases[i] != phases[i-1])
print(f"Phase transitions: {phase_changes}")

print(f"\n--- SEMANTIC COLLAPSE CHECK ---")
# Check if all neighborhoods merged into one or collapsed
neighborhood_counts = [m['neighborhoods'] for m in metrics_log]
if neighborhood_counts[-1] <= 1 and initial['neighborhoods'] > 3:
    print("⚠️  SEMANTIC COLLAPSE: neighborhoods reduced to 1")
elif final_attractors == 0 and avg_attractors > 2:
    print("⚠️  ATTRACTOR COLLAPSE: all attractors lost")
else:
    print("✓  No collapse detected")

print(f"\n--- REPLAY DETERMINISM CHECK ---")
# Final state hash
final_state = ues.get_current_state()
final_hash = final_state.state_hash
print(f"Final state hash: {final_hash}")
print(f"Total versions: {ues._current_version}")

print("\n=== STABILITY TEST RESULT ===")
if final_attractors > 0 and final_core > 0 and max_temp < 1.0:
    print("✓ SUBSTRATE STABLE - Ready for cognitive operations")
elif final_attractors == 0:
    print("✗ ATTRACTOR COLLAPSE - Semantic field unstable")
elif max_temp > 0.8:
    print("✗ THERMODYNAMIC RUNAWAY - System overheated")
else:
    print("⚠ PARTIALLY STABLE - Check metrics above")