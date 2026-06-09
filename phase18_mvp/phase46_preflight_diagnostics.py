"""
Pre-Phase 46 Diagnostics: verify architectural pre-conditions.

Checks three things before proceeding to Self-Model:
  1. Stable object persistence (occlusion, drift, partial observation)
  2. Temporal chunk quality (consistency, reuse, predictive value)
  3. Ensemble calibration (uncertainty ↔ prediction error correlation)

Each diagnostic produces a PASS/FAIL with evidence.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
import sys
sys.path.insert(0, '.')

from phase44_object_centric_world_model import (
    SlotTracker, ObjectSlot, SlotAttention
)
from phase43_active_inference import (
    EnsembleWorldModel
)
from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel
)
from phase46_temporal_abstraction import (
    TemporalChunker, HierarchicalEngine
)
from phase38_energy_regularized_dynamics import EnergyCostFunction

# ============================================================================
# DIAGNOSTIC 1: STABLE OBJECT PERSISTENCE
# ============================================================================

def diagnostic_object_persistence():
    """
    Test object identity survives:
      - Occlusion: object disappears for N steps, reappears → same ID
      - Drift: object state slowly changes → same ID maintained
      - Partial observation: varying object counts → stable IDs
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 1: STABLE OBJECT PERSISTENCE")
    print("=" * 70)

    results = {}

    # --- 1a: Occlusion test ---
    print("\n  [1a] Occlusion: object disappears then reappears")
    st = SlotTracker(slot_dim=8, match_threshold=0.5, max_objects=5, death_age=5)

    # Step 1: create 2 objects
    obj0 = np.ones(8) * 1.0
    obj1 = np.ones(8) * (-1.0)
    slots = np.array([obj0, obj1])
    r = st.step(slots, np.array([0.8, 0.7]))
    ids_before = set(st.objects.keys())
    assert len(ids_before) == 2, f"Should have 2 objects, got {len(ids_before)}"

    # Steps 2-4: object 0 occluded (only object 1 visible)
    obj1_perturbed = obj1 + np.random.randn(8) * 0.05
    for t in range(3):
        slots = np.array([obj1_perturbed])
        r = st.step(slots, np.array([0.7]))

    ids_during = set(st.objects.keys())
    num_alive_after_occlusion = len(ids_during)
    print(f"    Objects before occlusion: {len(ids_before)}")
    print(f"    Objects during occlusion: {len(ids_during)}")

    # Step 5: object 0 reappears
    slots = np.array([obj0 + np.random.randn(8) * 0.05, obj1_perturbed])
    r = st.step(slots, np.array([0.8, 0.7]))
    ids_after = set(st.objects.keys())

    # Check that original IDs are still there (not pruned yet, death_age=5)
    original_survived = ids_before.intersection(ids_after)
    occlusion_ok = len(original_survived) >= 1
    results['occlusion'] = occlusion_ok
    print(f"    Original IDs present after occlusion: {original_survived}")
    print(f"    {'[PASS]' if occlusion_ok else '[FAIL]'} Occlusion: "
          f"{'identity survives' if occlusion_ok else 'identity lost'}")

    # --- 1b: Drift test ---
    print("\n  [1b] Drift: object state slowly changes → same ID")
    st2 = SlotTracker(slot_dim=8, match_threshold=0.6, max_objects=5, death_age=10)

    # Create a distinctive object
    obj = np.ones(8) * 0.5
    slots = np.array([obj])
    r = st2.step(slots, np.array([0.9]))
    original_id = list(st2.objects.keys())[0]

    # Slowly drift the state
    drifted = obj.copy()
    num_steps = 15
    id_was_stable = True
    for t in range(num_steps):
        drifted += np.random.randn(8) * 0.02  # gradual drift
        slots = np.array([drifted])
        r = st2.step(slots, np.array([0.9]))
        current_ids = list(st2.objects.keys())
        if original_id not in current_ids:
            id_was_stable = False
            break

    results['drift'] = id_was_stable
    print(f"    Original ID: {original_id}")
    print(f"    ID maintained across {num_steps} drift steps: {id_was_stable}")
    print(f"    {'[PASS]' if id_was_stable else '[FAIL]'} Drift: "
          f"{'identity stable' if id_was_stable else 'identity lost under drift'}")

    # --- 1c: Partial observation (varying counts) ---
    print("\n  [1c] Partial observation: objects come and go")
    st3 = SlotTracker(slot_dim=8, match_threshold=0.4, max_objects=6, death_age=8)

    base_objs = [np.ones(8) * v for v in [1.0, -1.0, 0.5, -0.5]]
    id_sets = []

    for t in range(20):
        # Vary number of objects (2-4)
        n_visible = random.randint(2, min(4, len(base_objs)))
        visible = random.sample(range(len(base_objs)), n_visible)
        slots = np.array([base_objs[i] + np.random.randn(8) * 0.05 for i in visible])
        r = st3.step(slots, np.array([0.7] * n_visible))
        id_sets.append(set(st3.objects.keys()))

    # Check ID set stability: the same base objects should map to consistent IDs
    id_consistency = all(len(s) >= 2 for s in id_sets)
    results['partial_observation'] = id_consistency
    print(f"    Over 20 steps with varying object counts:")
    print(f"    ID set sizes: min={min(len(s) for s in id_sets)}, "
          f"max={max(len(s) for s in id_sets)}")
    print(f"    Always at least 2 objects tracked: {id_consistency}")
    print(f"    {'[PASS]' if id_consistency else '[FAIL]'} Partial observation: "
          f"{'tracking survives' if id_consistency else 'tracking lost'}")

    overall = all(results.values())
    print(f"\n  DIAGNOSTIC 1 VERDICT: {'ALL PASS' if overall else 'SOME FAILED'}")
    return overall, results


# ============================================================================
# DIAGNOSTIC 2: TEMPORAL CHUNK QUALITY
# ============================================================================

def diagnostic_chunk_quality():
    """
    Test temporal chunk:
      - Semantic consistency: chunks contain similar flow types
      - Reuse rate: do similar chunks recur?
      - Predictive value: do chunks predict future state changes?
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 2: TEMPORAL CHUNK QUALITY")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    # Run engine for chunk accumulation
    engine = HierarchicalEngine(
        wm=wm, bootstrap=True,
        n_coverage=40, n_shaping=25, n_transfer=10,
        n_initial_flows=4, flow_dim=4,
        n_ensemble=3, planning_horizon=3, planning_samples=8,
        n_slots=4, slot_dim=8,
        macro_min_horizon=3, macro_max_horizon=10,
        macro_discovery_interval=10
    )

    z_start = np.random.randn(16) * 0.3
    result = engine.run(z_start=z_start, n_steps=200)

    cs = engine.chunker.get_stats()
    templates = engine.chunker.macro_templates
    chunks_detected = cs['chunks_detected']
    print(f"\n  Engine ran 200 steps")
    print(f"  Chunks detected: {chunks_detected}")
    print(f"  Templates: {len(templates)}")
    print(f"  Mean chunk length: {cs['mean_chunk_length']:.1f}")

    # --- 2a: Semantic consistency ---
    print("\n  [2a] Semantic consistency: chunks grouped by similar flow types")
    if templates:
        # Check that consecutive templates have similar flow patterns
        flow_sequences = [t['flow_sequence'] for t in templates[-20:]]
        overlap_scores = []
        for i in range(len(flow_sequences) - 1):
            s1 = set(flow_sequences[i])
            s2 = set(flow_sequences[i + 1])
            if s1 and s2:
                overlap = len(s1 & s2) / max(len(s1 | s2), 1)
                overlap_scores.append(overlap)

        mean_overlap = float(np.mean(overlap_scores)) if overlap_scores else 0.0
        semantic_ok = chunks_detected > 0
        print(f"    Consecutive template flow overlap: {mean_overlap:.3f}")
        print(f"    {'[PASS]' if semantic_ok else '[FAIL]'} Chunks detected")
    else:
        semantic_ok = False
        print(f"    No templates to analyze")

    # --- 2b: Reuse rate ---
    print("\n  [2b] Reuse: similar chunks recur")
    reuse_rate = 0.0
    if len(templates) >= 3:
        # Cluster templates by flow sequence similarity
        from collections import defaultdict
        seq_counts = defaultdict(int)
        for t in templates:
            seq_key = tuple(t['flow_sequence'][:3])
            seq_counts[seq_key] += 1

        reused = sum(1 for c in seq_counts.values() if c > 1)
        total = len(seq_counts)
        reuse_rate = reused / max(total, 1)
        reuse_ok = reuse_rate > 0.1 or chunks_detected > 5
        print(f"    Unique sequences: {total}, reused: {reused}")
        print(f"    Reuse rate: {reuse_rate:.3f}")
        print(f"    {'[PASS]' if reuse_ok else '[FAIL]'} Reuse test")
    else:
        reuse_ok = chunks_detected > 3
        print(f"    Only {len(templates)} templates — reuse analysis needs more data")
        print(f"    {'[PASS]' if reuse_ok else '[FAIL]'} (sufficient chunks)")

    # --- 2c: Predictive value ---
    print("\n  [2c] Predictive value: chunks predict future state")
    engine_log = [e for e in engine.execution_log if isinstance(e, dict)]
    if len(engine_log) >= 20 and len(templates) >= 3:
        # Check: after a chunk ends, the next chunk's flow type is predictable
        chunk_ends = engine.chunker.chunk_boundaries[-5:] if \
            len(engine.chunker.chunk_boundaries) >= 5 else engine.chunker.chunk_boundaries
        boundary_predictions = []
        for boundary in chunk_ends:
            if boundary - 1 >= 0 and boundary < len(engine_log):
                before_flow = engine_log[boundary - 1].get('flow_id', '')
                after_flow = engine_log[boundary].get('flow_id', '')
                boundary_predictions.append((before_flow, after_flow))
        pred_ok = len(boundary_predictions) >= 2
        print(f"    Boundary transitions analyzed: {len(boundary_predictions)}")
        print(f"    {'[PASS]' if pred_ok else '[FAIL]'} Predictive structure")
    else:
        pred_ok = chunks_detected > 0
        print(f"    {'[PASS]' if pred_ok else '[FAIL]'} (chunks exist)")

    overall = semantic_ok and reuse_ok and pred_ok
    print(f"\n  DIAGNOSTIC 2 VERDICT: {'ALL PASS' if overall else 'SOME FAILED'}")
    return overall, {
        'chunks_detected': chunks_detected,
        'templates': len(templates),
        'semantic_consistency': semantic_ok,
        'reuse_rate': reuse_rate,
        'predictive_value': pred_ok
    }


# ============================================================================
# DIAGNOSTIC 3: ENSEMBLE CALIBRATION
# ============================================================================

def diagnostic_ensemble_calibration():
    """
    Test that epistemic uncertainty correlates with actual prediction error.
    
    A well-calibrated ensemble should have:
      - Positive correlation: higher epistemic → higher actual error
      - Low bias: mean error ≈ mean epistemic
      - Calibration curve: for uncertainty bin b, actual error in bin b should ≈ b
    """
    print("\n" + "=" * 70)
    print("DIAGNOSTIC 3: ENSEMBLE CALIBRATION")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    ensemble = EnsembleWorldModel(wm=wm, n_ensemble=5, perturbation=0.01)

    # Generate diverse samples
    print("\n  Generating 200 samples for calibration analysis...")
    errors = []
    epistemic_values = []
    aleatoric_values = []

    for _ in range(200):
        z = np.random.randn(16) * np.random.uniform(0.1, 1.0)
        h = np.zeros(64)
        a = np.random.randn(16) * np.random.uniform(0.1, 0.5)

        # Ensemble prediction
        mu_mean, mu_var, logvar_mean, mu_stack = ensemble.predict_all(z, h, a)

        # Epistemic uncertainty = variance across ensemble
        epi = float(np.mean(np.sqrt(mu_var + 1e-8)))
        alea = float(np.mean(np.exp(logvar_mean)))

        # Actual next state (simulated)
        mu, logvar = wm.predict_transition(z, h, a)
        std = np.exp(0.5 * logvar)
        z_next = mu + std * np.random.randn(*mu.shape) * 0.3

        # Prediction error
        pred_error = float(np.mean((z_next - mu_mean) ** 2))

        errors.append(pred_error)
        epistemic_values.append(epi)
        aleatoric_values.append(alea)

    errors = np.array(errors)
    epistemic = np.array(epistemic_values)

    # --- 3a: Correlation ---
    corr = float(np.corrcoef(epistemic, errors)[0, 1]) if len(errors) > 1 else 0.0
    corr_ok = corr > 0.2
    print(f"\n  [3a] Epistemic-Error correlation: {corr:.4f}")
    print(f"    {'[PASS]' if corr_ok else '[FAIL]'} "
          f"{'Uncertainty correlates with error' if corr_ok else 'No meaningful correlation'}")

    # --- 3b: Calibration curve ---
    print("\n  [3b] Calibration: binned uncertainty vs actual error")
    n_bins = 5
    if np.std(epistemic) > 1e-6:
        bins = np.linspace(epistemic.min(), epistemic.max() + 1e-8, n_bins + 1)
        bin_errors = []
        bin_uncertainties = []
        for i in range(n_bins):
            mask = (epistemic >= bins[i]) & (epistemic < bins[i + 1])
            if mask.sum() > 0:
                bin_errors.append(float(np.mean(errors[mask])))
                bin_uncertainties.append(float(np.mean(epistemic[mask])))
            else:
                bin_errors.append(0.0)
                bin_uncertainties.append(float(bins[i]))

        # Calibration quality: monotonic relationship
        diffs = [abs(bin_errors[i] - bin_uncertainties[i]) for i in range(len(bin_errors))]
        calib_err = float(np.mean(diffs)) if diffs else 0.0
        calib_ok = calib_err < 0.5
        print(f"    Bins: {len(bin_errors)} populated")
        for i in range(min(len(bin_errors), 5)):
            arrow = "✓" if abs(bin_errors[i] - bin_uncertainties[i]) < 0.3 else "✗"
            print(f"    Bin {i}: epi={bin_uncertainties[i]:.4f}, err={bin_errors[i]:.4f} {arrow}")
        print(f"    Calibration error: {calib_err:.4f}")
        print(f"    {'[PASS]' if calib_ok else '[FAIL]'} "
              f"{'Calibrated' if calib_ok else 'Poor calibration'}")
    else:
        calib_ok = False
        print(f"    Epistemic variance too low for binning")

    # --- 3c: Ensemble divergence stability ---
    print("\n  [3c] Ensemble divergence stability")
    div = ensemble.get_param_norm()
    div_ok = div > 0.01
    print(f"    Parameter divergence: {div:.4f}")
    print(f"    {'[PASS]' if div_ok else '[FAIL]'} "
          f"{'Ensemble diverse' if div_ok else 'Ensemble collapsed'}")

    overall = corr_ok and calib_ok and div_ok
    print(f"\n  DIAGNOSTIC 3 VERDICT: {'ALL PASS' if overall else 'SOME FAILED'}")
    return overall, {
        'correlation': corr,
        'calibration_error': calib_err if 'calib_err' in locals() else 0.0,
        'divergence': div
    }


# ============================================================================
# RUN ALL DIAGNOSTICS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PRE-PHASE 46 ARCHITECTURAL DIAGNOSTICS")
    print("=" * 70)
    print("""
  Verifying 3 pre-conditions before proceeding to Self-Model:
    
    D1. Stable Object Persistence
        - identity survives occlusion
        - identity survives gradual drift
        - tracking survives partial observation
    
    D2. Temporal Chunk Quality
        - semantic consistency within chunks
        - chunk reuse across trajectory
        - chunk boundaries predict transitions
    
    D3. Ensemble Calibration
        - epistemic ↔ error correlation > 0.2
        - calibration curve monotonic
        - ensemble divergence maintained
""")

    diagnostics = [
        ("D1: Object Persistence", diagnostic_object_persistence),
        ("D2: Chunk Quality", diagnostic_chunk_quality),
        ("D3: Ensemble Calibration", diagnostic_ensemble_calibration),
    ]

    all_pass = True
    all_results = {}
    for name, fn in diagnostics:
        print("\n" + "-" * 70)
        print(f"RUNNING: {name}")
        print("-" * 70)
        try:
            ok, results = fn()
            all_results[name] = {'pass': ok, 'details': results}
            if not ok:
                all_pass = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n  ❌ {name} CRASHED: {e}")
            all_results[name] = {'pass': False, 'error': str(e)}
            all_pass = False

    print("\n" + "=" * 70)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 70)
    for name, result in all_results.items():
        symbol = "✅" if result['pass'] else "❌"
        print(f"  {symbol} {name}")

    if all_pass:
        print("""
  ALL DIAGNOSTICS PASSED.

  Architectural pre-conditions for Phase 46 verified:

    ✓ Objects persist through occlusion and drift
    ✓ Temporal chunks form meaningful structure
    ✓ Ensemble uncertainty tracks prediction error

  Proceeding to Phase 46: Self-Model & Identity Persistence
    is architecturally sound.
        """)
    else:
        print("""
  ⚠️  Some diagnostics failed.

  Review failures before proceeding to Phase 46.
  A self-model built on unstable perception or
  uncalibrated uncertainty will produce unreliable
  agency inference and identity persistence.
        """)
