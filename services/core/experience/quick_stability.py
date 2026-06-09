"""
Quick Stability Test - 1k mutations for faster validation
"""
import sys
sys.path.insert(0, '/app')
sys.path.insert(0, '/app/experience')

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
from unified_epistemic_state import get_ues_manager
from reflection_kernel import init_reflection_kernel
from contradiction_engine import get_contradiction_engine, reset_contradiction_engine
from semantic_embedding_layer import get_semantic_layer, reset_semantic_layer

reset_contradiction_engine()
reset_semantic_layer()
reset_wal_engine()

ues = get_ues_manager()
kernel = init_reflection_kernel(ues)
wal = get_wal_engine()
sem = get_semantic_layer()
ce = get_contradiction_engine()

print("=== 1K MUTATION STABILITY TEST ===")
from datetime import datetime

# Initial snapshot
initial_attractors = 0
initial_core = 0

for tick in range(1000):
    mutation = tick % 4
    
    if mutation == 0:
        ues.add_belief(f"b_{tick}", f"Belief {tick}", 0.5, 0.5, "test")
    elif mutation == 1:
        state = ues.get_current_state()
        if state.beliefs:
            bid = list(state.beliefs.keys())[tick % len(state.beliefs)]
            ues.update_belief(bid, 0.3 + (tick % 50) / 100.0, 0.5)
    elif mutation == 2:
        state = ues.get_current_state()
        bids = list(state.beliefs.keys())
        if len(bids) >= 2:
            ues.add_causal_edge([bids[0]], [bids[-1]], 0.5, 0.5)
    else:
        state = ues.get_current_state()
        bids = list(state.beliefs.keys())[:2]
        if bids:
            ues.register_contradiction(f"c_{tick}", bids, "test", "low")
    
    # Sample every 200 ticks
    if tick % 200 == 0:
        state = ues.get_current_state()
        emb = sem.map_beliefs(state.beliefs, state.causal_edges)
        neigh = sem.detect_neighborhoods(emb)
        attr = sem.detect_attractors(neigh)
        core = sem.get_identity_core(attr, state.beliefs)
        
        # Contradiction pressure
        tx = wal.get_transaction_lineage()
        cg = wal.get_causal_graph()
        ce.detect_pressure(state.beliefs, tx, cg, ues._current_version)
        active = [c for c in ce.get_clusters()]
        thermo = ce.compute_cognitive_thermodynamics(len(state.beliefs), active, 0.5)
        
        print(f"Tick {tick:4d}: beliefs={len(state.beliefs)}, attr={len(attr)}, core={len(core)}, temp={thermo.get('temperature', 0):.3f}")
        
        if tick == 0:
            initial_attractors = len(attr)
            initial_core = len(core)

# Final metrics
state = ues.get_current_state()
emb = sem.map_beliefs(state.beliefs, state.causal_edges)
neigh = sem.detect_neighborhoods(emb)
attr = sem.detect_attractors(neigh)
core = sem.get_identity_core(attr, state.beliefs)
tx = wal.get_transaction_lineage()
cg = wal.get_causal_graph()
ce.detect_pressure(state.beliefs, tx, cg, ues._current_version)
thermo = ce.compute_cognitive_thermodynamics(len(state.beliefs), [c for c in ce.get_clusters()], 0.5)

print(f"\n=== RESULTS ===")
print(f"Beliefs: {len(state.beliefs)}")
print(f"Attractors: {initial_attractors} -> {len(attr)}")
print(f"Identity core: {initial_core} -> {len(core)}")
print(f"Version: {ues._current_version}")
print(f"Temp: {thermo.get('temperature', 0):.3f}")
print(f"Phase: {thermo.get('phase_state', 'unknown')}")

if len(attr) > 0 and len(core) > 0 and thermo.get('temperature', 0) < 0.5:
    print("\n✓ STABLE")
else:
    print("\n⚠ CHECK METRICS")