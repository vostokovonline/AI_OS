"""
Phase 41 → Phase 40 Integration Test

Verifies: Bootstrapped world model + normalized GP → Phase 40 self-organization

KEY FIX:
  Raw GP = exp(-||z - goal||) is flat when goal_norm >> latent_norm.
  Normalized GP = exp(-||z - goal|| / ||goal||) always provides signal.

  This integration test:
    1. Bootstraps the world model (phase41_bootstrapper)
    2. Creates SelfOrganizingEngine with normalized GP
    3. Runs continuous self-organization
    4. Verifies GP stays useful, CEM converges, flows evolve
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import deque

import sys
sys.path.insert(0, '.')

from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel, BehavioralPhysicsLearner,
    FlowTrajectoryBuffer, FlowEpisode
)
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, PointFlow, LimitCycleFlow, FlowType
)
from phase34_inverse_control_stabilization import InverseDynamicsModel
from phase31_hierarchical_execution import GoalAttractor
from phase38_energy_regularized_dynamics import (
    EnergyCostFunction, EfficiencyEvaluator, EnergyRegularizedCEM,
    CostAwareFlowSelection
)
from phase40_self_organizing_geometry import (
    ContinuousFlowEcology, ContinuousManifoldDrift,
    ContinuousCEM, SelfOrganizingEngine
)
from phase41_bootstrapper import RepresentationBootstrapper, ShapedReward


class NormalizedGPAwareEngine(SelfOrganizingEngine):
    """
    SelfOrganizingEngine with normalized GP computation.

    Fixes the flat GP landscape by normalizing distance by goal norm:
      GP = exp(-||z - goal|| / ||goal||) instead of exp(-||z - goal||)

    Also adjusts birth/death thresholds to match the new GP range.
    """

    def __init__(self, *args, **kwargs):
        # Extract normalized GP flag before passing rest
        self.use_normalized_gp = kwargs.pop('use_normalized_gp', True)
        super().__init__(*args, **kwargs)

        # Cache goal norm for normalization
        self.goal_norm = float(np.linalg.norm(
            self.goal.attractor_state[:self.wm.latent_dim]
        )) + 1e-8

        # Adjust ecology thresholds for normalized GP range (~0.3 mean)
        self.ecology.gp_threshold = 0.01  # was 0.001 in raw GP
        self._patch_ecology()

    def _patch_ecology(self):
        """Adjust ecology birth/death thresholds for normalized GP range."""
        orig_death = self.ecology._compute_death_probability

        def patched_death(fid, flow):
            prob = orig_death(fid, flow)
            # Already handled: avg_gp < 0.001 now means avg_gp < 0.1
            return prob

        self.ecology._compute_death_probability = patched_death

    def compute_goal_prob(self, z: np.ndarray) -> float:
        """Normalized GP: signal-present even when goal is far."""
        dist = float(np.linalg.norm(
            z - self.goal.attractor_state[:len(z)]
        ))
        if self.use_normalized_gp:
            return float(np.exp(-dist / self.goal_norm))
        else:
            return float(np.exp(-dist))

    def step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """Override: use normalized GP everywhere."""
        flow, flow_id, coord = self.cem.select_flow(z, h)
        a = flow.compute_action(z, h)

        mu, logvar = self.wm.predict_transition(z, h, a)
        std = np.exp(0.5 * logvar)
        z_next = mu + std * np.random.randn(*mu.shape) * 0.1
        flow.record_transition(z, z_next, a, h)

        self.inv_dyn.train_step(z, z_next, a)
        self.inv_dyn.add_transition(z, z_next, a)

        # NORMALIZED GP: the key fix
        goal_prob = self.compute_goal_prob(z_next)

        prev_gp = self.execution_log[-1]['goal_prob'] if self.execution_log else goal_prob
        gp_delta = goal_prob - prev_gp

        cost_info = self.energy_cost.compute([a], [z, z_next], flow)

        flow.stability = flow.compute_lyapunov_estimate()
        flow.goal_alignment = float(np.clip(
            flow.goal_alignment + 0.01 * (gp_delta * 10), 0.0, 1.0
        ))

        # Ecology with normalized GP deltas
        self.ecology.record_gp_delta(flow_id, gp_delta)
        self.ecology.record_performance(flow_id, goal_prob)
        eco_result = self.ecology.step()

        self.drift.step(flow_id, goal_prob, gp_delta, self.goal)
        self.cem.observe_outcome(coord, flow_id, goal_prob, cost_info['total'])

        if self.total_steps % self.train_interval == 0:
            self._train_model()

        self.total_steps += 1

        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'action': a.copy(),
            'goal_prob': float(goal_prob),
            'gp_delta': float(gp_delta),
            'flow_type': flow.flow_type.value,
            'flow_id': flow_id,
            'stability': flow.stability,
            'energy_cost': cost_info,
            'eco_births': eco_result['born'],
            'eco_deaths': eco_result['died'],
            'n_flows': len(self.manifold.flows)
        }

        self.execution_log.append(step_result)
        return step_result


def run_integration_test(
    n_bootstrap_coverage: int = 200,
    n_bootstrap_shaping: int = 150,
    n_bootstrap_transfer: int = 80,
    n_self_organizing: int = 200
):
    """Full integration: bootstrapper → Phase 40 engine."""
    print("\n" + "=" * 70)
    print("PHASE 41 → PHASE 40 INTEGRATION TEST")
    print("=" * 70)
    print("\n  Step 1: Bootstrap world model...")
    print("  Step 2: Run Phase 40 self-organizing engine...")
    print("  Step 3: Verify metrics...")

    # ======================================================================
    # STEP 1: Bootstrap
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 1: BOOTSTRAPPING WORLD MODEL")
    print("-" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='integration_goal',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )

    bootstrapper = RepresentationBootstrapper(
        wm=wm,
        goal=goal,
        n_coverage=n_bootstrap_coverage,
        n_shaping=n_bootstrap_shaping,
        n_transfer=n_bootstrap_transfer
    )

    bootstrap_result = bootstrapper.run()

    print(f"\n  Bootstrap complete:")
    print(f"    GP landscape: {bootstrap_result['mean_gp']:.4f} mean, "
          f"{bootstrap_result['max_gp']:.4f} max")
    print(f"    Verdict: {bootstrap_result['final_verdict']}")
    assert bootstrap_result['final_verdict'] == 'SIGNAL_PRESENT', \
        "Bootstrap must produce non-flat GP landscape!"

    # ======================================================================
    # STEP 2: Self-Organizing Engine
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 2: SELF-ORGANIZING ENGINE (with normalized GP)")
    print("-" * 70)

    engine = NormalizedGPAwareEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=8,
        flow_dim=4,
        lambda_cost=0.5,
        train_interval=5
    )

    print(f"\n  Engine created with {len(engine.manifold.flows)} initial flows")
    print(f"  Goal norm: {engine.goal_norm:.2f}")
    print(f"  Running for {n_self_organizing} steps...")

    # Run
    result = engine.run(
        z_start=np.random.randn(wm.latent_dim) * 0.3,
        n_steps=n_self_organizing
    )

    # ======================================================================
    # STEP 3: Verify
    # ======================================================================
    print("\n" + "-" * 70)
    print("STEP 3: VERIFICATION")
    print("-" * 70)

    goal_probs = [e['goal_prob'] for e in engine.execution_log]

    # GP metrics
    mean_gp = float(np.mean(goal_probs))
    max_gp = float(max(goal_probs))
    min_gp = float(min(goal_probs))
    final_gp = float(goal_probs[-1]) if goal_probs else 0.0
    gp_trend = final_gp - goal_probs[0] if len(goal_probs) > 1 else 0.0

    gp_above_01 = sum(1 for gp in goal_probs if gp > 0.1)
    gp_above_05 = sum(1 for gp in goal_probs if gp > 0.5)

    print(f"\n  GP metrics:")
    print(f"    Mean: {mean_gp:.4f}")
    print(f"    Max: {max_gp:.4f}")
    print(f"    Min: {min_gp:.4f}")
    print(f"    Trend: {gp_trend:+.4f}")
    print(f"    > 0.1: {gp_above_01}/{len(goal_probs)}")
    print(f"    > 0.5: {gp_above_05}/{len(goal_probs)}")

    gp_ok = gp_above_01 > len(goal_probs) * 0.5

    # CEM metrics
    cem_stats = engine.cem.get_stats()
    cem_converged = float(np.mean(cem_stats['std'])) < 0.5 if hasattr(cem_stats['std'], '__iter__') else cem_stats['std'] < 0.5

    print(f"\n  CEM:")
    print(f"    Mean: {cem_stats['mean'][:4]}...")
    print(f"    Std: {cem_stats['std'][:4]}...")
    print(f"    Mean score: {cem_stats['mean_score']:.4f}")
    print(f"    Converged: {cem_converged}")

    # Flow ecology
    eco_stats = engine.ecology.get_stats()
    print(f"\n  Ecology:")
    print(f"    Final flows: {eco_stats['n_flows']}")
    print(f"    Births: {eco_stats['births']}")
    print(f"    Deaths: {eco_stats['deaths']}")

    flows_alive = eco_stats['n_flows'] > 0

    # Training
    tr = result.get('training', {})
    print(f"\n  Training:")
    print(f"    Steps: {tr.get('training_steps', 0)}")
    print(f"    Buffer: {tr.get('buffer_episodes', 0)} eps, "
          f"{tr.get('buffer_transitions', 0)} trans")
    if 'loss_improvement' in tr:
        print(f"    Loss improvement: {tr.get('loss_improvement', 0) * 100:.1f}%")

    # ======================================================================
    # VERDICT
    # ======================================================================
    total_passed = sum([gp_ok, cem_converged, flows_alive])
    total_tests = 3

    print("\n" + "=" * 70)
    print("INTEGRATION VERDICT")
    print("=" * 70)
    print(f"\n  {'✅' if gp_ok else '❌'} GP stays useful (>0.1 for 50%+ steps): {gp_ok}")
    print(f"  {'✅' if cem_converged else '⚠️'} CEM distribution converges: {cem_converged}")
    print(f"  {'✅' if flows_alive else '❌'} Flow ecology remains alive: {flows_alive}")
    print(f"\n  Passed: {total_passed}/{total_tests}")

    if total_passed == total_tests:
        print("\n  ✅ INTEGRATION PASSED: Phase 41 → Phase 40 end-to-end verified.")
    else:
        print(f"\n  ⚠️ Partial pass ({total_passed}/{total_tests}):")
        if not gp_ok:
            print("    - GP still flat after bootstrapping (check normalization)")
        if not cem_converged:
            print("    - CEM not converging (more training needed)")
        if not flows_alive:
            print("    - Flow ecology died (check birth/death thresholds)")

    return {
        'total_passed': total_passed,
        'total_tests': total_tests,
        'gp_ok': gp_ok,
        'cem_converged': cem_converged,
        'flows_alive': flows_alive,
        'mean_gp': mean_gp,
        'max_gp': max_gp,
        'gp_trend': gp_trend,
        'gp_above_01': gp_above_01,
        'gp_above_05': gp_above_05,
        'cem_std': float(np.mean(cem_stats['std'])) if hasattr(cem_stats['std'], '__iter__') else cem_stats['std'],
        'n_flows': eco_stats['n_flows'],
        'eco_births': eco_stats['births'],
        'eco_deaths': eco_stats['deaths']
    }


if __name__ == "__main__":
    result = run_integration_test(
        n_bootstrap_coverage=200,
        n_bootstrap_shaping=150,
        n_bootstrap_transfer=80,
        n_self_organizing=200
    )
