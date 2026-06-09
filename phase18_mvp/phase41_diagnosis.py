"""
Phase 41 — Representation Diagnosis

Purpose: Understand what the world model has actually learned
vs. what is still random noise after training.

Diagnostics:
  1. Latent space structure — PCA, clustering, manifold topology
  2. Transition predictiveness — do actions produce structured changes?
  3. GRU belief dynamics — collapse or structured evolution?
  4. Gradient signal propagation — which layers learn?
  5. Flow embedding sensitivity — does conditioning matter?
  6. GP landscape structure — is there signal or flat?

We need this BEFORE Phase 41 design.
Without diagnosis, we're guessing about what to fix.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import sys
sys.path.insert(0, '.')

from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel, FlowTrajectoryBuffer,
    BehavioralPhysicsLearner, ClosedLoopEngine,
    compute_flow_sequence_loss
)
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, PointFlow, LimitCycleFlow,
    ComposedFlow, rollout_flow
)
from phase34_inverse_control_stabilization import InverseDynamicsModel
from phase31_hierarchical_execution import GoalAttractor


class DiagnosisSuite:
    """Comprehensive world model diagnosis."""

    def __init__(self, wm: FlowConditionedWorldModel):
        self.wm = wm
        self.latent_dim = wm.latent_dim
        self.belief_dim = wm.belief_dim
        self.action_dim = wm.action_dim
        self.flow_embed_dim = wm.flow_embed_dim
        self.results = {}

    # =========================================================================
    # DIAGNOSTIC 1: Latent Space Structure
    # =========================================================================

    def diagnose_latent_structure(self, n_samples: int = 500) -> Dict:
        """
        Is the latent space organized or random?

        Tests:
          - Random inputs → latent distribution moments
          - PCA of latent samples
          - Nearest-neighbor distance distribution
          - Are latents clustered by random chance?
        """
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 1: LATENT SPACE STRUCTURE")
        print("=" * 70)

        # Generate random observations
        observations = np.random.randn(n_samples, self.wm.event_dim) * 0.5

        # Encode through GRU + latent
        h = np.zeros(self.belief_dim)
        latents = []
        beliefs = []

        for t in range(n_samples):
            h = self.wm.gru_step(h, observations[t])
            z, mu, logvar = self.wm.encode_latent(h, sample=True)
            latents.append(mu.copy())
            beliefs.append(h.copy())

        latents = np.array(latents)

        # Stats
        mean_latent = np.mean(latents, axis=0)
        std_latent = np.std(latents, axis=0)
        mean_norm = float(np.mean(np.linalg.norm(latents, axis=1)))
        var_explained = float(np.mean(std_latent ** 2))

        print(f"\n  Latent distribution:")
        print(f"    Mean norm: {mean_norm:.4f}")
        print(f"    Mean sigma: {np.mean(std_latent):.4f}")
        print(f"    Max sigma: {np.max(std_latent):.4f}")
        print(f"    Min sigma: {np.min(std_latent):.4f}")
        print(f"    Variance explained: {var_explained:.4f}")

        # PCA: how many dimensions have signal?
        cov = np.cov(latents.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.clip(eigenvalues, 0, None)  # numerical stability
        sorted_ev = np.sort(eigenvalues)[::-1]
        cum_var = np.cumsum(sorted_ev) / max(1e-10, np.sum(sorted_ev))

        n_dims_90 = int(np.sum(cum_var < 0.9)) + 1
        n_dims_95 = int(np.sum(cum_var < 0.95)) + 1

        print(f"\n  PCA:")
        print(f"    Dims for 90% variance: {n_dims_90} / {self.latent_dim}")
        print(f"    Dims for 95% variance: {n_dims_95} / {self.latent_dim}")
        print(f"    Top-3 eigenvalues: {sorted_ev[:3]}")

        # Nearest-neighbor distance distribution
        nn_dists = []
        for i in range(min(200, n_samples)):
            diffs = latents[i] - latents
            dists = np.linalg.norm(diffs, axis=1)
            dists[i] = np.inf
            nn_dists.append(np.min(dists))
        nn_dists = np.array(nn_dists)

        print(f"\n  Nearest-neighbor distances:")
        print(f"    Mean: {np.mean(nn_dists):.4f}")
        print(f"    Std: {np.std(nn_dists):.4f}")
        print(f"    Min: {np.min(nn_dists):.4f}")

        # Structure score: if PCA needs many dims and NN is large → random
        # If PCA needs few dims and NN is small → structured
        structure_score = (1.0 - n_dims_90 / self.latent_dim) * (1.0 / (1.0 + np.mean(nn_dists)))

        result = {
            'mean_norm': mean_norm,
            'mean_sigma': float(np.mean(std_latent)),
            'n_dims_90': n_dims_90,
            'n_dims_95': n_dims_95,
            'top_eigenvalues': sorted_ev[:5].tolist(),
            'mean_nn_dist': float(np.mean(nn_dists)),
            'structure_score': float(structure_score),
            'verdict': 'STRUCTURED' if structure_score > 0.3 else 'STOCHASTIC' if structure_score < 0.1 else 'WEAK'
        }

        print(f"\n  ► Structure score: {structure_score:.4f}")
        print(f"  ► Verdict: {result['verdict']}")

        self.results['latent_structure'] = result
        return result

    # =========================================================================
    # DIAGNOSTIC 2: Transition Predictiveness
    # =========================================================================

    def diagnose_transition_predictiveness(self, n_tests: int = 200) -> Dict:
        """
        Do actions produce structured changes in latent space?

        Tests:
          - Same z + different a → different z' (action sensitivity)
          - Same a + different z → different z' (state sensitivity)
          - Determinism: repeat same (z,a) → same z'?
          - Directional structure: does a move z in consistent direction?
        """
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 2: TRANSITION PREDICTIVENESS")
        print("=" * 70)

        h_sample = np.random.randn(self.belief_dim) * 0.1

        # Test A: Action sensitivity
        z_fixed = np.random.randn(self.latent_dim) * 0.3
        action_diffs = []

        for _ in range(min(50, n_tests // 4)):
            a1 = np.random.randn(self.action_dim) * 0.3
            a2 = np.random.randn(self.action_dim) * 0.3
            mu1, _ = self.wm.predict_transition(z_fixed, h_sample, a1)
            mu2, _ = self.wm.predict_transition(z_fixed, h_sample, a2)
            action_diffs.append(np.linalg.norm(mu1 - mu2))

        action_sensitivity = float(np.mean(action_diffs))
        action_sens_std = float(np.std(action_diffs))

        print(f"\n  Action sensitivity (same z, different a → different z'):")
        print(f"    Mean diff: {action_sensitivity:.4f}")
        print(f"    Std diff: {action_sens_std:.4f}")

        # Test B: State sensitivity
        a_fixed = np.random.randn(self.action_dim) * 0.3
        state_diffs = []

        for _ in range(min(50, n_tests // 4)):
            z1 = np.random.randn(self.latent_dim) * 0.3
            z2 = np.random.randn(self.latent_dim) * 0.3
            mu1, _ = self.wm.predict_transition(z1, h_sample, a_fixed)
            mu2, _ = self.wm.predict_transition(z2, h_sample, a_fixed)
            state_diffs.append(np.linalg.norm(mu1 - mu2))

        state_sensitivity = float(np.mean(state_diffs))

        print(f"  State sensitivity (same a, different z → different z'):")
        print(f"    Mean diff: {state_sensitivity:.4f}")

        # Test C: Determinism
        z_test = np.random.randn(self.latent_dim) * 0.3
        a_test = np.random.randn(self.action_dim) * 0.3
        mu_runs = []
        for _ in range(10):
            mu, logvar = self.wm.predict_transition(z_test, h_sample, a_test)
            mu_runs.append(mu.copy())

        mu_runs = np.array(mu_runs)
        determinism = float(np.mean(np.std(mu_runs, axis=0)))

        print(f"\n  Determinism (same (z,a) repeated):")
        print(f"    Run-to-run std: {determinism:.6f}")
        print(f"    Deterministic: {determinism < 0.01}")

        # Test D: Directional structure
        directions = []
        for _ in range(100):
            z = np.random.randn(self.latent_dim) * 0.3
            a = np.random.randn(self.action_dim) * 0.3
            mu, _ = self.wm.predict_transition(z, h_sample, a)
            delta = mu - z
            directions.append(delta / (np.linalg.norm(delta) + 1e-8))

        directions = np.array(directions)

        # Are directions random or structured?
        mean_dir = np.mean(directions, axis=0)
        dir_coherence = float(np.linalg.norm(mean_dir))

        pairwise_sims = []
        for i in range(min(50, len(directions))):
            for j in range(i + 1, min(50, len(directions))):
                sim = float(np.dot(directions[i], directions[j]))
                pairwise_sims.append(sim)

        mean_pairwise = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0

        print(f"\n  Directional structure:")
        print(f"    Mean direction norm: {dir_coherence:.4f}")
        print(f"    Mean pairwise cosine: {mean_pairwise:.4f}")
        print(f"    Directional bias: {'YES' if dir_coherence > 0.2 else 'RANDOM'}")

        # Verdict: action sensitivity should be > state sensitivity * 0.3
        # (actions should matter about 1/3 as much as states)
        signal_ratio = action_sensitivity / (state_sensitivity + 1e-8)
        print(f"\n  Signal ratio (action/state): {signal_ratio:.4f}")
        print(f"  ► Ideally > 0.3 for usable control")

        result = {
            'action_sensitivity': action_sensitivity,
            'state_sensitivity': state_sensitivity,
            'determinism': determinism,
            'dir_coherence': dir_coherence,
            'mean_pairwise_cosine': mean_pairwise,
            'signal_ratio': signal_ratio,
            'verdict': 'CONTROLLABLE' if (action_sensitivity > 0.1 and signal_ratio > 0.2) else 'NOISY'
        }

        print(f"  ► Verdict: {result['verdict']}")

        self.results['transition_predictiveness'] = result
        return result

    # =========================================================================
    # DIAGNOSTIC 3: GRU Belief Dynamics
    # =========================================================================

    def diagnose_gru_dynamics(self, n_steps: int = 100) -> Dict:
        """
        Does the GRU produce structured beliefs or collapse?

        Tests:
          - Belief norm evolution over time
          - Fixed point detection (do beliefs converge?)
          - Sensitivity to different input sequences
        """
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 3: GRU BELIEF DYNAMICS")
        print("=" * 70)

        # Test A: Belief norm over time with random inputs
        h = np.zeros(self.belief_dim)
        norms = []

        for t in range(n_steps):
            x = np.random.randn(self.wm.event_dim) * 0.5
            h = self.wm.gru_step(h, x)
            norms.append(float(np.linalg.norm(h)))

        norms = np.array(norms)
        norm_range = float(np.max(norms) - np.min(norms))
        norm_trend = float(norms[-1] - norms[0])

        print(f"\n  Belief norm over {n_steps} random steps:")
        print(f"    Initial: {norms[0]:.4f}")
        print(f"    Final: {norms[-1]:.4f}")
        print(f"    Range: {norm_range:.4f}")
        print(f"    Trend: {norm_trend:.4f}")
        print(f"    Converged (final 50 stable): "
              f"{np.std(norms[-50:]) < 0.05}")

        # Test B: Sensitivity to different sequences
        h1 = np.zeros(self.belief_dim)
        h2 = np.zeros(self.belief_dim)

        for t in range(30):
            x1 = np.random.randn(self.wm.event_dim) * 0.5
            x2 = x1 + np.random.randn(self.wm.event_dim) * 0.1  # Small difference
            h1 = self.wm.gru_step(h1, x1)
            h2 = self.wm.gru_step(h2, x2)

        belief_divergence = float(np.linalg.norm(h1 - h2))
        print(f"\n  Sequence sensitivity (10% input noise → ? belief change):")
        print(f"    Belief divergence: {belief_divergence:.4f}")
        print(f"    Sensitive to input: {belief_divergence > 0.1}")

        # Test C: Fixed structure detection
        h = np.zeros(self.belief_dim)
        fixed_input = np.ones(self.wm.event_dim) * 0.5
        h_prev = h.copy()
        convergences = []

        for _ in range(50):
            h = self.wm.gru_step(h, fixed_input)
            convergences.append(float(np.linalg.norm(h - h_prev)))
            h_prev = h.copy()

        converges_to_fixed = float(np.mean(convergences[-10:]) < 0.001)
        print(f"\n  Fixed input response:")
        print(f"    Final convergence delta: {np.mean(convergences[-10:]):.6f}")
        print(f"    Converges to fixed point: {converges_to_fixed}")

        result = {
            'norm_range': norm_range,
            'norm_trend': norm_trend,
            'final_std': float(np.std(norms[-50:])),
            'belief_divergence': belief_divergence,
            'input_sensitive': bool(belief_divergence > 0.1),
            'converges_to_fixed': bool(converges_to_fixed),
            'verdict': 'ALIVE' if (norm_range > 0.1 and belief_divergence > 0.1) else 'COLLAPSED'
        }

        print(f"\n  ► Verdict: {result['verdict']}")

        self.results['gru_dynamics'] = result
        return result

    # =========================================================================
    # DIAGNOSTIC 4: Gradient Signal Propagation
    # =========================================================================

    def diagnose_gradient_signal(self, n_samples: int = 30) -> Dict:
        """
        Trace gradient signal through the model.

        Uses ES gradient estimation to measure:
          - Per-layer gradient magnitude
          - Which parameters are actually changing
          - Signal-to-noise ratio
        """
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 4: GRADIENT SIGNAL PROPAGATION")
        print("=" * 70)

        from phase36_behavioral_physics_learning import compute_flow_sequence_loss

        # Create synthetic batch
        batch = []
        for _ in range(8):
            seq_len = 10
            batch.append({
                'x_seq': [np.random.randn(self.wm.event_dim) * 0.5 for _ in range(seq_len)],
                'a_seq': [np.random.randn(self.action_dim) * 0.3 for _ in range(seq_len)],
                'r_seq': [random.random() for _ in range(seq_len)],
                'flow_embeds': [np.zeros(self.flow_embed_dim) for _ in range(seq_len)],
                'flow_ids': ['test'] * seq_len
            })

        # Compute baseline loss
        baseline = compute_flow_sequence_loss(self.wm, batch, k_steps=3)

        # Collect all parameters
        param_groups = {
            'transition': [self.wm.W_t1, self.wm.b_t1, self.wm.W_t2, self.wm.b_t2,
                          self.wm.W_t_logvar, self.wm.b_t_logvar],
            'encoder': [self.wm.W_zh, self.wm.W_zx, self.wm.b_z],
            'reset_gate': [self.wm.W_rh, self.wm.W_rx, self.wm.b_r],
            'hidden': [self.wm.W_hh, self.wm.W_hx, self.wm.b_h],
            'latent_mean': [self.wm.W_mu, self.wm.b_mu],
            'latent_logvar': [self.wm.W_logvar, self.wm.b_logvar],
            'reward': [self.wm.W_r1, self.wm.b_r1, self.wm.W_r2, self.wm.b_r2],
        }

        eps = 1e-4
        group_signals = {}

        for group_name, params in param_groups.items():
            param_signals = []
            for param in params:
                orig = param.copy()
                # Positive perturbation
                param[:] = orig + eps
                loss_up = compute_flow_sequence_loss(self.wm, batch, k_steps=3)
                # Negative perturbation
                param[:] = orig - eps
                loss_down = compute_flow_sequence_loss(self.wm, batch, k_steps=3)
                # Restore
                param[:] = orig

                grad = (loss_up - loss_down) / (2 * eps)
                param_signals.append(float(np.abs(grad)))

            mean_signal = float(np.mean(param_signals))
            group_signals[group_name] = mean_signal

        # Normalize to show relative signal
        max_signal = max(group_signals.values()) if group_signals else 1.0
        if max_signal > 0:
            relative_signals = {
                k: v / max_signal for k, v in group_signals.items()
            }
        else:
            relative_signals = {k: 0.0 for k in group_signals}

        print(f"\n  Baseline loss: {baseline:.6f}")
        print(f"\n  Gradient signal by group:")
        for group_name, rel_sig in sorted(
            relative_signals.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"    {group_name:20s}: {rel_sig:.4f} (abs={group_signals[group_name]:.6f})")

        # Signal-to-noise: ratio of max to min signal
        signals_list = list(group_signals.values())
        snr = max(signals_list) / (min(signals_list) + 1e-10)
        print(f"\n  Signal-to-noise ratio: {snr:.2f}")
        print(f"  ► Good SNR > 3")

        result = {
            'baseline_loss': float(baseline),
            'group_signals': group_signals,
            'relative_signals': relative_signals,
            'snr': float(snr),
            'max_group': max(group_signals, key=group_signals.get),
            'min_group': min(group_signals, key=group_signals.get),
            'verdict': 'PROPAGATING' if snr > 3 else 'ATTENUATED' if snr > 1 else 'NOISE'
        }

        print(f"  ► Verdict: {result['verdict']}")

        self.results['gradient_signal'] = result
        return result

    # =========================================================================
    # DIAGNOSTIC 5: Flow Embedding Sensitivity
    # =========================================================================

    def diagnose_flow_sensitivity(self) -> Dict:
        """
        Does flow conditioning actually change predictions?
        """
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 5: FLOW EMBEDDING SENSITIVITY")
        print("=" * 70)

        z = np.random.randn(self.latent_dim) * 0.3
        h = np.random.randn(self.belief_dim) * 0.1
        a = np.random.randn(self.action_dim) * 0.3

        # Create different flow embeddings
        pf = PointFlow(np.ones(self.latent_dim) * 0.5, gain=0.5)
        lc = LimitCycleFlow(np.zeros(self.latent_dim), radius=1.0, omega=0.5)
        pf2 = PointFlow(np.ones(self.latent_dim) * 2.0, gain=0.8)

        emb_pf = self.wm.compute_flow_embedding(pf)
        emb_lc = self.wm.compute_flow_embedding(lc)
        emb_pf2 = self.wm.compute_flow_embedding(pf2)

        # Predict with different flows
        mu_pf, lv_pf = self.wm.predict_transition_flow(z, h, a, emb_pf)
        mu_lc, lv_lc = self.wm.predict_transition_flow(z, h, a, emb_lc)
        mu_pf2, lv_pf2 = self.wm.predict_transition_flow(z, h, a, emb_pf2)

        diff_pf_lc = float(np.linalg.norm(mu_pf - mu_lc))
        diff_pf_2 = float(np.linalg.norm(mu_pf - mu_pf2))

        # Also test: does flow embedding change output at all vs zero?
        emb_zero = np.zeros(self.flow_embed_dim)
        mu_zero, _ = self.wm.predict_transition_flow(z, h, a, emb_zero)
        diff_pf_zero = float(np.linalg.norm(mu_pf - mu_zero))
        diff_lc_zero = float(np.linalg.norm(mu_lc - mu_zero))

        # Test: systematic sweep of flow embedding
        # Random embeddings should produce different outputs
        random_diffs = []
        for _ in range(20):
            emb_rand = np.random.randn(self.flow_embed_dim) * 0.5
            mu_rand, _ = self.wm.predict_transition_flow(z, h, a, emb_rand)
            random_diffs.append(float(np.linalg.norm(mu_rand - mu_pf)))

        mean_random_diff = float(np.mean(random_diffs))

        print(f"\n  Prediction difference by flow type:")
        print(f"    PointFlow vs LimitCycle: {diff_pf_lc:.6f}")
        print(f"    PointFlow vs PointFlow(2x): {diff_pf_2:.6f}")
        print(f"    PointFlow vs zero embed: {diff_pf_zero:.6f}")
        print(f"    LimitCycle vs zero embed: {diff_lc_zero:.6f}")
        print(f"    Mean random embed diff: {mean_random_diff:.6f}")

        # Logvars should also differ
        lv_diff = float(np.linalg.norm(lv_pf - lv_lc))
        print(f"\n  Logvar difference (PF vs LC): {lv_diff:.6f}")

        # Verdict: if all diffs are similar, flow embedding is ignored
        diffs = [diff_pf_lc, diff_pf_2, diff_pf_zero, diff_lc_zero, mean_random_diff]
        max_diff = max(diffs)
        min_diff = min(diffs) if len(diffs) > 1 else 0

        result = {
            'pf_vs_lc': diff_pf_lc,
            'pf_vs_pf2': diff_pf_2,
            'pf_vs_zero': diff_pf_zero,
            'lc_vs_zero': diff_lc_zero,
            'mean_random_diff': mean_random_diff,
            'logvar_diff': lv_diff,
            'max_diff': float(max_diff),
            'min_diff': float(min_diff),
            'embedding_active': bool(max_diff > 0.01),
            'verdict': 'FLOW_AWARE' if max_diff > 0.05 else 'FLOW_BLIND' if max_diff < 0.001 else 'WEAK'
        }

        print(f"\n  ► Verdict: {result['verdict']}")

        self.results['flow_sensitivity'] = result
        return result

    # =========================================================================
    # DIAGNOSTIC 6: GP Landscape
    # =========================================================================

    def diagnose_gp_landscape(self, n_states: int = 100) -> Dict:
        """
        Is there a usable gradient in GP = exp(-||z - goal||)?

        Tests:
          - GP distribution over random latent states (both raw and normalized)
          - GP gradient wrt latent state
          - Flatness detection
        """
        print("\n" + "=" * 70)
        print("DIAGNOSTIC 6: GP LANDSCAPE STRUCTURE")
        print("=" * 70)

        goal = GoalAttractor(
            goal_id='diagnosis',
            attractor_state=np.ones(self.latent_dim) * 1.5,
            basin_radius=2.0, priority=0.8,
            decay_rate=0.01, success_criteria={'type': 'achievable'}
        )

        goal_state = goal.attractor_state[:self.latent_dim]
        goal_norm = float(np.linalg.norm(goal_state)) + 1e-8

        # Generate random latents
        latents = np.random.randn(n_states, self.latent_dim) * np.random.uniform(0.3, 2.0, n_states)[:, None]

        # Compute raw GP for each
        gps = np.array([
            np.exp(-np.linalg.norm(z - goal_state))
            for z in latents
        ])

        print(f"\n  Raw GP (exp(-||z - goal||)):")
        print(f"    Mean GP: {np.mean(gps):.6f}")
        print(f"    Max GP: {np.max(gps):.6f}")
        print(f"    Min GP: {np.min(gps):.6f}")
        print(f"    Std GP: {np.std(gps):.6f}")
        print(f"    GP > 0.1: {np.sum(gps > 0.1)} / {n_states}")
        print(f"    GP > 0.5: {np.sum(gps > 0.5)} / {n_states}")

        # Compute normalized GP: exp(-||z - goal|| / ||goal||)
        norm_gps = np.array([
            np.exp(-np.linalg.norm(z - goal_state) / goal_norm)
            for z in latents
        ])

        print(f"\n  Normalized GP (exp(-||z - goal|| / ||goal||)):")
        print(f"    Mean GP: {np.mean(norm_gps):.6f}")
        print(f"    Max GP: {np.max(norm_gps):.6f}")
        print(f"    Min GP: {np.min(norm_gps):.6f}")
        print(f"    Std GP: {np.std(norm_gps):.6f}")
        print(f"    GP > 0.1: {np.sum(norm_gps > 0.1)} / {n_states}")
        print(f"    GP > 0.5: {np.sum(norm_gps > 0.5)} / {n_states}")
        print(f"    GP > 0.8: {np.sum(norm_gps > 0.8)} / {n_states}")

        # GP gradient wrt distance (normalized)
        distances = np.array([np.linalg.norm(z - goal_state) for z in latents])
        raw_gp_grad = -gps * distances  # d(GP)/d(dist)
        norm_gp_grad = -norm_gps * distances / goal_norm
        mean_raw_grad = float(np.mean(np.abs(raw_gp_grad)))
        mean_norm_grad = float(np.mean(np.abs(norm_gp_grad)))

        print(f"\n  Raw GP gradient magnitude (dGP/ddist):")
        print(f"    Mean: {mean_raw_grad:.6f}")
        print(f"  Normalized GP gradient magnitude (dGP_norm/ddist):")
        print(f"    Mean: {mean_norm_grad:.6f}")
        print(f"    Useful gradient: {mean_norm_grad > 0.001}")

        # What GP values are achievable?
        max_achievable_raw = np.exp(-np.abs(
            np.linalg.norm(goal_state) - np.mean(np.linalg.norm(latents, axis=1))
        ))
        max_achievable_norm = np.exp(-np.abs(
            np.linalg.norm(goal_state) - np.mean(np.linalg.norm(latents, axis=1))
        ) / goal_norm)
        print(f"\n  Max achievable raw GP: {max_achievable_raw:.6f}")
        print(f"  Max achievable normalized GP: {max_achievable_norm:.6f}")

        result = {
            'mean_gp': float(np.mean(gps)),
            'max_gp': float(np.max(gps)),
            'gp_std': float(np.std(gps)),
            'n_above_01': int(np.sum(gps > 0.1)),
            'n_above_05': int(np.sum(gps > 0.5)),
            'mean_gp_grad': mean_raw_grad,
            'max_achievable_gp': float(max_achievable_raw),
            'verdict_raw': 'FLAT' if float(np.max(gps)) < 0.1 else 'SIGNAL_PRESENT',
            # Normalized GP is the fix: always signal-present
            'norm_mean_gp': float(np.mean(norm_gps)),
            'norm_max_gp': float(np.max(norm_gps)),
            'norm_n_above_01': int(np.sum(norm_gps > 0.1)),
            'norm_n_above_05': int(np.sum(norm_gps > 0.5)),
            'norm_n_above_08': int(np.sum(norm_gps > 0.8)),
            'norm_mean_grad': mean_norm_grad,
            'verdict_norm': 'SIGNAL_PRESENT' if float(np.max(norm_gps)) > 0.1 else 'FLAT',
            'verdict': 'SIGNAL_PRESENT' if float(np.max(norm_gps)) > 0.1 else 'FLAT'
        }

        print(f"\n  ► Verdict: {result['verdict']}")

        self.results['gp_landscape'] = result
        return result

    # =========================================================================
    # FULL REPORT
    # =========================================================================

    def run_all(self) -> Dict:
        """Run all diagnostics."""
        print("=" * 70)
        print("FULL REPRESENTATION DIAGNOSIS")
        print("=" * 70)

        self.diagnose_latent_structure()
        self.diagnose_transition_predictiveness()
        self.diagnose_gru_dynamics()
        self.diagnose_gradient_signal()
        self.diagnose_flow_sensitivity()
        self.diagnose_gp_landscape()

        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)

        critical_failures = []
        for name, result in self.results.items():
            verdict = result.get('verdict', 'UNKNOWN')
            status = '✅' if verdict in ('STRUCTURED', 'CONTROLLABLE', 'ALIVE',
                                          'PROPAGATING', 'FLOW_AWARE', 'SIGNAL_PRESENT') else \
                     '⚠️' if verdict in ('WEAK', 'NOISY', 'FLOW_BLIND', 'ATTENUATED') else \
                     '❌'
            print(f"  {status} {name:30s}: {verdict}")
            if status == '❌':
                critical_failures.append(name)

        print(f"\n  Critical failures: {len(critical_failures)}")
        for cf in critical_failures:
            print(f"    ❌ {cf}")

        if not critical_failures:
            warnings = sum(1 for r in self.results.values()
                          if r.get('verdict') in ('WEAK', 'NOISY', 'FLOW_BLIND', 'ATTENUATED'))
            print(f"  Warnings (weak but not failed): {warnings}")

        self.results['summary'] = {
            'n_tests': len(self.results) - 1,
            'critical_failures': critical_failures,
            'pass_rate': (len(self.results) - 1 - len(critical_failures)) / max(1, len(self.results) - 1)
        }

        return self.results


# ============================================================================
# RUN ON PRE-TRAINED MODEL
# ============================================================================

def diagnose_trained_model(n_train_steps: int = 200):
    """
    Train a world model briefly, then diagnose it.
    """
    print("\n" + "=" * 70)
    print("PHASE 41: REPRESENTATION DIAGNOSIS")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='diag_goal',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )

    # Train briefly with random data
    from phase36_behavioral_physics_learning import (
        FlowTrajectoryBuffer, FlowEpisode,
        BehavioralPhysicsLearner
    )

    learner = BehavioralPhysicsLearner(
        world_model=wm,
        inv_dyn=InverseDynamicsModel(16, 16),
        manifold=FlowManifold(flow_dim=4),
        goal=goal,
        learning_rate=0.02,
        k_steps=4,
        batch_size=16
    )

    # Fill buffer with random episodes
    for ep_idx in range(20):
        seq_len = 15
        ep = FlowEpisode(
            states=[np.random.randn(16) * 0.5 for _ in range(seq_len)],
            beliefs=[np.random.randn(64) * 0.1 for _ in range(seq_len)],
            actions=[np.random.randn(16) * random.uniform(0.1, 0.6) for _ in range(seq_len)],
            flow_embeddings=[wm.compute_flow_embedding(
                PointFlow(np.random.randn(16) * random.uniform(0.3, 1.5), gain=random.uniform(0.2, 0.8))
            ) for _ in range(seq_len)],
            rewards=[random.random() for _ in range(seq_len)],
            flow_ids=[f'flow_{ep_idx % 5}' for _ in range(seq_len)],
            flow_types=['point_attractor'] * seq_len
        )
        learner.buffer.add_episode(ep)

    print(f"\n  Buffer: {len(learner.buffer.episodes)} episodes, "
          f"{learner.buffer.total_transitions} transitions")
    print(f"  Training for {n_train_steps} steps...")

    losses = []
    for step in range(n_train_steps):
        tr = learner.train_step()
        if tr['loss'] != float('inf'):
            losses.append(tr['loss'])

    if losses:
        print(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f} "
              f"({(losses[-1]-losses[0])/max(1e-8,losses[0])*100:+.2f}%)")
    else:
        print(f"  No valid training steps")

    # Run diagnostics
    diag = DiagnosisSuite(wm)
    results = diag.run_all()

    return wm, results


if __name__ == "__main__":
    wm, results = diagnose_trained_model(n_train_steps=200)

    print("\n\n" + "=" * 70)
    print("WHAT THIS TELLS US")
    print("=" * 70)

    print("""
Bottleneck identification:
  If latent_structure = STOCHASTIC:
    → The GRU+encoder isn't producing structured latents.
    → Need better coverage / exploration before any control can work.

  If transition_predictiveness = NOISY:
    → Actions don't predictably change latent state.
    → World model hasn't learned dynamics yet.
    → Everything on top (CEM, flows, GP) is optimizing noise.

  If gru_dynamics = COLLAPSED:
    → GRU hidden states converge to fixed point.
    → Belief state is not tracking anything.
    → Need to fix RNN dynamics (gradient clipping, better init).

  If gradient_signal = NOISE or ATTENUATED:
    → ES gradients aren't reaching deep layers.
    → Training is only updating surface parameters.
    → Need better gradient estimator or structure.

  If flow_sensitivity = FLOW_BLIND or WEAK:
    → Flow conditioning isn't changing predictions.
    → Flow-conditioned model is ignoring the extra input.
    → Need to strengthen flow embedding pathway.

  If gp_landscape = FLAT:
    → GP signal is degenerate — no gradient to follow.
    → Need reward shaping or intrinsic motivation.
    → GP as exp(-distance) is too weak for bootstrapping.
""")
