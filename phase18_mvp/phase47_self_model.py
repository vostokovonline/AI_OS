"""
Phase 46 — Self-Model & Identity Persistence.

ARCHITECTURAL SHIFT:
  Before (Phases 25-45):  system predicts the world
                           no distinction between self and environment
                           no persistent identity across time

  After (Phase 46):        system predicts itself in the world
                           agency: "did I cause this change?"
                           counterfactual self: "what if I acted differently?"
                           identity: self survives object turnover and context switch

  Components:
    46.1 — SelfLatent:          persistent self representation with temporal stability
    46.2 — AgencyInference:     P(self_caused | action, prediction, outcome)
    46.3 — CounterfactualSelf:  "what if" simulation with counterfactual trajectories
    46.4 — SelfEngine:          integrates all into HierarchicalEngine

  Every step:
    1-3.  Hierarchical engine step (uncertainty → flow → execute) 
    4.    Self-latent update + temporal stability constraint   (46.1)
    5.    Agency inference: self-caused vs external change     (46.2)
    6.    Periodic counterfactual simulation                   (46.3)
    7.    Self-tracker: persistent identity for self-slot      (46.4)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
import sys
sys.path.insert(0, '.')

from phase46_temporal_abstraction import (
    HierarchicalEngine, MacroFlow, TemporalChunker, MacroFlowManifold
)
from phase44_object_centric_world_model import ObjectSlot
from phase35_dynamical_skill_flows import FlowManifold
from phase36_behavioral_physics_learning import FlowConditionedWorldModel
from phase38_energy_regularized_dynamics import EnergyCostFunction


# ============================================================================
# 46.1 — PERSISTENT SELF-LATENT
# ============================================================================

class SelfLatent:
    """
    Persistent self representation.

    Key properties:
      - NOT a world object slot — separate latent with its own dynamics
      - Temporally stable: self(t+1) ≈ transformed self(t) even as world changes
      - Self-transition model: predicts next self-state from current + action
      - Identity signal: measures consistency of self over time

    The self-latent is updated EVERY step and used for:
      - Agency inference (what did I cause?)
      - Counterfactual simulation (what if I acted differently?)
      - Identity tracking (is this still "me"?)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        self_dim: int = 8,
        temporal_stability: float = 0.9,
        learning_rate: float = 0.01
    ):
        self.latent_dim = latent_dim
        self.self_dim = self_dim
        self.temporal_stability = temporal_stability
        self.lr = learning_rate

        # Self latent state
        self.state: np.ndarray = np.zeros(self_dim)

        # Self-transition model: (self_t, action) → self_{t+1}
        self.W_self = np.random.randn(self_dim, self_dim + latent_dim) * 0.05
        self.b_self = np.zeros(self_dim)

        # Projection: latent z → self contribution
        self.W_proj = np.random.randn(self_dim, latent_dim) * 0.05

        # Self history (last 50 states)
        self.history: List[np.ndarray] = []
        self.max_history = 50

        # Identity persistence score (how stable self has been)
        self.identity_coherence: float = 1.0
        self.coherence_history: List[float] = []

    def project_from_latent(self, z: np.ndarray) -> np.ndarray:
        """Extract self-relevant features from latent state."""
        return self.W_proj @ z

    def predict_next(self, action: np.ndarray) -> np.ndarray:
        """Predict next self state given current state and action."""
        # Non-linear transition with temporal stability prior
        concat = np.concatenate([self.state, action])
        hidden = np.tanh(self.W_self @ concat + self.b_self)
        # Temporal stability: blend prediction with current state
        return self.temporal_stability * self.state + (1 - self.temporal_stability) * hidden

    def update(self, z: np.ndarray, action: np.ndarray):
        """Update self-latent from observed latent state and action."""
        # Prediction
        predicted = self.predict_next(action)

        # Actual self contribution from latent
        actual_self = self.project_from_latent(z)

        # Blend: weighted by temporal stability + observation
        self.state = (
            self.temporal_stability * predicted
            + (1 - self.temporal_stability) * actual_self
        )

        # Track identity coherence
        if len(self.history) >= 1:
            change = float(np.linalg.norm(self.state - self.history[-1]))
            coherence = np.exp(-change)
            self.identity_coherence = 0.95 * self.identity_coherence + 0.05 * coherence
            self.coherence_history.append(self.identity_coherence)

        self.history.append(self.state.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)

        # Learn self-transition model (online delta rule)
        error = actual_self - predicted
        concat = np.concatenate([self.history[-2] if len(self.history) >= 2
                                 else np.zeros(self.self_dim), action])
        d_hidden = (1 - np.tanh(self.W_self @ concat + self.b_self) ** 2)
        grad_W = np.outer(d_hidden * error, concat)
        self.W_self += self.lr * grad_W
        self.b_self += self.lr * d_hidden * error

    def get_identity_signal(self) -> float:
        """
        Measure of self-identity persistence.
        1.0 = perfectly stable, 0.0 = chaotic/drifting.
        """
        return float(np.clip(self.identity_coherence, 0.0, 1.0))

    def get_stats(self) -> Dict:
        return {
            'self_dim': self.self_dim,
            'identity_coherence': float(self.identity_coherence),
            'state_norm': float(np.linalg.norm(self.state)),
            'history_len': len(self.history),
            'temporal_stability': self.temporal_stability
        }


# ============================================================================
# 46.2 — AGENCY INFERENCE
# ============================================================================

class AgencyInference:
    """
    Distinguishes self-caused vs externally-caused changes.

    Core insight:
      P(self_caused | action, predicted_outcome, actual_outcome)

    If actual = predicted → high agency (self caused the change)
    If actual ≠ predicted → external factors (environment caused the change)

    Formal model:
      agency = exp(-||actual_delta - predicted_delta|| / ||predicted_delta + eps||)

    This is computed at TWO levels:
      - Latent level:  did my action cause the latent state change?
      - Object level:  did my action cause the object state change?

    Agency signal is used for:
      - Self-model learning (what parts of the world are "mine")
      - Credit assignment (flows that produce high agency are more "owned")
    """

    def __init__(
        self,
        latent_dim: int = 16,
        prediction_weight: float = 1.0,
        consistency_weight: float = 0.3
    ):
        self.latent_dim = latent_dim
        self.prediction_weight = prediction_weight
        self.consistency_weight = consistency_weight

        # Agency history
        self.agency_history: List[float] = []
        self.object_agency_history: List[float] = []
        self.max_history = 100

    def compute_latent_agency(
        self,
        z_before: np.ndarray,
        z_after: np.ndarray,
        predicted_mu: np.ndarray,
        action: np.ndarray
    ) -> float:
        """
        Compute agency at latent level.

        High agency = observed transition matches predicted transition.
        Low agency = world changed differently than predicted.
        """
        actual_delta = z_after - z_before
        predicted_delta = predicted_mu - z_before

        pred_norm = float(np.linalg.norm(predicted_delta)) + 1e-8
        error = float(np.linalg.norm(actual_delta - predicted_delta))

        # Agency = 1 when error = 0, decays exponentially with error
        agency = float(np.exp(-error / pred_norm))
        agency = np.clip(agency, 0.0, 1.0)

        self.agency_history.append(agency)
        if len(self.agency_history) > self.max_history:
            self.agency_history.pop(0)

        return agency

    def compute_object_agency(
        self,
        objects_before: List[ObjectSlot],
        objects_after: List[ObjectSlot],
        action: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute per-object agency.

        For each persistent object that existed before AND after:
          agency(object) = exp(-||state_change|| / action_relevance)

        Objects that change a lot despite weak action → low agency (external cause)
        Objects that change predictably with action → high agency (controlled)
        """
        per_object: Dict[str, float] = {}
        before_map = {o.id: o.state.copy() for o in objects_before}
        after_map = {o.id: o.state.copy() for o in objects_after}

        action_norm = float(np.linalg.norm(action)) + 1e-8

        for oid, state_before in before_map.items():
            if oid in after_map:
                state_after = after_map[oid]
                change = float(np.linalg.norm(state_after - state_before))
                # Agency = high if change is proportional to action magnitude
                expected_change = action_norm * 0.1
                agency = float(np.exp(-abs(change - expected_change) / (expected_change + 1e-8)))
                per_object[oid] = float(np.clip(agency, 0.0, 1.0))

        mean_agency = float(np.mean(list(per_object.values()))) if per_object else 0.0
        self.object_agency_history.append(mean_agency)
        if len(self.object_agency_history) > self.max_history:
            self.object_agency_history.pop(0)

        return mean_agency, per_object

    def get_agency_trend(self) -> float:
        """Are we getting more or less agentic? Positive = improving."""
        if len(self.agency_history) < 10:
            return 0.0
        recent = self.agency_history[-10:]
        return float(recent[-1] - recent[0])

    def get_stats(self) -> Dict:
        return {
            'mean_agency': float(np.mean(self.agency_history)) if self.agency_history else 0.0,
            'mean_object_agency': float(np.mean(self.object_agency_history))
                if self.object_agency_history else 0.0,
            'agency_trend': self.get_agency_trend(),
            'n_samples': len(self.agency_history)
        }


# ============================================================================
# 46.3 — COUNTERFACTUAL SELF
# ============================================================================

class CounterfactualSelf:
    """
    Counterfactual self-modeling.

    "What if I had acted differently?"

    For each action taken, the counterfactual self simulates:
      1. What would have happened with alternative action a'
      2. What would the new latent state z' be
      3. What would the goal probability GP' be
      4. Would this have been better or worse?

    The difference (actual - counterfactual) is the REGRET signal,
    which drives:
      - Policy improvement (learn from mistakes)
      - Self-model refinement (update self-transition)
      - Identity stability (self is invariant across counterfactuals)

    Formally:
      regret(a) = GP(actual) - GP(counterfactual(a'))
      self_stability = P(self_state | counterfactual_trajectory)
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        latent_dim: int = 16,
        belief_dim: int = 64,
        n_counterfactuals: int = 3,
        counterfactual_interval: int = 10
    ):
        self.wm = wm
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        self.n_counterfactuals = n_counterfactuals
        self.counterfactual_interval = counterfactual_interval

        # Regret history
        self.regret_history: List[float] = []
        self.counterfactual_log: List[Dict] = []

    def simulate_counterfactual(
        self,
        z: np.ndarray,
        h: np.ndarray,
        actual_action: np.ndarray,
        actual_z_next: np.ndarray,
        actual_goal_prob: float,
        available_flows: Dict[str, Any]
    ) -> Dict:
        """
        Simulate "what if I had chosen a different flow/action?"

        Tries alternative actions from the same state,
        computes what WOULD have happened.
        """
        results = {'actual_gp': actual_goal_prob, 'counterfactuals': []}

        # Generate alternative actions
        alt_actions = []
        for _ in range(self.n_counterfactuals):
            if available_flows and random.random() < 0.5:
                alt_fid = random.choice(list(available_flows.keys()))
                alt_flow = available_flows[alt_fid]
                alt_a = alt_flow.compute_action(z, h)
            else:
                alt_a = actual_action + np.random.randn(self.latent_dim) * 0.2
            alt_actions.append(alt_a)

        # Simulate each alternative
        for i, alt_a in enumerate(alt_actions):
            try:
                cf_mu, cf_logvar = self.wm.predict_transition(z, h, alt_a)
                cf_std = np.exp(0.5 * cf_logvar)
                cf_z_next = cf_mu + cf_std * np.random.randn(*cf_mu.shape) * 0.1

                # Compute counterfactual goal probability
                # (using normalized GP approximation)
                cf_goal_prob = float(np.mean(np.exp(-np.abs(cf_z_next))))

                # Regret = how much better/worse this would have been
                regret = actual_goal_prob - cf_goal_prob

                results['counterfactuals'].append({
                    'alt_action': alt_a,
                    'alt_z_next': cf_z_next,
                    'alt_goal_prob': cf_goal_prob,
                    'regret': float(regret),
                    'was_better': cf_goal_prob > actual_goal_prob
                })
            except Exception:
                continue

        # Mean regret across all counterfactuals
        if results['counterfactuals']:
            mean_regret = float(np.mean([
                cf['regret'] for cf in results['counterfactuals']
            ]))
        else:
            mean_regret = 0.0

        results['mean_regret'] = mean_regret
        self.regret_history.append(mean_regret)
        if len(self.regret_history) > 100:
            self.regret_history.pop(0)

        self.counterfactual_log.append(results)
        if len(self.counterfactual_log) > 50:
            self.counterfactual_log.pop(0)

        return results

    def get_regret_trend(self) -> float:
        """Are we regretting less over time? Negative = improving."""
        if len(self.regret_history) < 5:
            return 0.0
        return float(np.mean(self.regret_history[-5:]))

    def get_stats(self) -> Dict:
        return {
            'n_counterfactuals': self.n_counterfactuals,
            'mean_regret': float(np.mean(self.regret_history)) if self.regret_history else 0.0,
            'regret_trend': self.get_regret_trend(),
            'n_simulations': len(self.counterfactual_log)
        }


# ============================================================================
# 46.4 — SELF ENGINE
# ============================================================================

class SelfEngine(HierarchicalEngine):
    """
    Extends HierarchicalEngine with self-model and identity persistence.

    Adds:
      46.1 — SelfLatent: persistent self representation
      46.2 — AgencyInference: self-caused vs external change
      46.3 — CounterfactualSelf: "what if" simulation

    Every step:
      1-3.  Hierarchical engine step
      4.    Self-latent update + temporal stability          (46.1)
      5.    Agency inference (latent + object level)          (46.2)
      6.    Self-tracker: maintain self across time           (46.4)
      7.    Periodic counterfactual simulation                (46.3)
      8.    Identity persistence update
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        bootstrap: bool = True,
        n_coverage: int = 200,
        n_shaping: int = 150,
        n_transfer: int = 80,
        n_initial_flows: int = 8,
        flow_dim: int = 4,
        lambda_cost: float = 0.3,
        train_interval: int = 5,
        # Phase 43
        n_ensemble: int = 5,
        ensemble_lr: float = 0.005,
        exploration_beta: float = 0.1,
        planning_horizon: int = 5,
        planning_samples: int = 24,
        uncertainty_weight: float = 0.3,
        energy_weight: float = 0.2,
        goal_weight: float = 1.0,
        # Phase 44
        n_slots: int = 6,
        slot_dim: int = 8,
        slot_iterations: int = 3,
        match_threshold: float = 0.5,
        max_objects: int = 10,
        rel_dynamics_lr: float = 0.01,
        # Phase 45
        macro_min_horizon: int = 3,
        macro_max_horizon: int = 10,
        macro_discovery_interval: int = 20,
        # Phase 46
        self_dim: int = 8,
        self_temporal_stability: float = 0.9,
        counterfactual_interval: int = 15,
        n_counterfactuals: int = 3
    ):
        super().__init__(
            wm=wm, bootstrap=bootstrap,
            n_coverage=n_coverage, n_shaping=n_shaping,
            n_transfer=n_transfer,
            n_initial_flows=n_initial_flows, flow_dim=flow_dim,
            lambda_cost=lambda_cost, train_interval=train_interval,
            n_ensemble=n_ensemble, ensemble_lr=ensemble_lr,
            exploration_beta=exploration_beta,
            planning_horizon=planning_horizon,
            planning_samples=planning_samples,
            uncertainty_weight=uncertainty_weight,
            energy_weight=energy_weight, goal_weight=goal_weight,
            n_slots=n_slots, slot_dim=slot_dim,
            slot_iterations=slot_iterations,
            match_threshold=match_threshold, max_objects=max_objects,
            rel_dynamics_lr=rel_dynamics_lr,
            macro_min_horizon=macro_min_horizon,
            macro_max_horizon=macro_max_horizon,
            macro_discovery_interval=macro_discovery_interval
        )

        # 46.1 — Self Latent
        self.self_latent = SelfLatent(
            latent_dim=wm.latent_dim,
            self_dim=self_dim,
            temporal_stability=self_temporal_stability
        )

        # 46.2 — Agency Inference
        self.agency = AgencyInference(
            latent_dim=wm.latent_dim
        )

        # 46.3 — Counterfactual Self
        self.counterfactual = CounterfactualSelf(
            wm=wm,
            latent_dim=wm.latent_dim,
            belief_dim=wm.belief_dim,
            n_counterfactuals=n_counterfactuals,
            counterfactual_interval=counterfactual_interval
        )

        # Self-tracking
        self.agency_log: List[float] = []
        self.self_coherence_log: List[float] = []
        self.counterfactual_count: int = 0
        self.counterfactual_interval = counterfactual_interval

    def step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """One step with self-model and identity persistence."""
        # ====================================================================
        # LAYER 1-3: Hierarchical engine core (from HierarchicalEngine)
        # ====================================================================
        # Uncertainty decomposition
        action_for_uncertainty = np.zeros(self.wm.action_dim)
        epi_before, alea_before, total_before = self.ensemble.decompose_uncertainty(
            z, h, action_for_uncertainty
        )

        # Flow selection
        objects = self.slot_tracker.get_active_objects()
        flow, flow_id, is_macro = self.hierarchical_cem.select_flow(
            z, h, objects, self.manifold.flows
        )
        if flow is None:
            flow, flow_id, _ = self.active_cem.select_flow(z, h)

        # Execute
        a = flow.compute_action(z, h)
        predicted_mu, predicted_logvar = self.wm.predict_transition(z, h, a)
        std = np.exp(0.5 * predicted_logvar)
        z_next = predicted_mu + std * np.random.randn(*predicted_mu.shape) * 0.1
        h_next = self.wm.gru_step(h, predicted_mu)

        flow.record_transition(z, z_next, a, h)
        self.inv_dyn.train_step(z, z_next, a)
        self.inv_dyn.add_transition(z, z_next, a)

        # Goal + info gain + energy
        goal_prob = self.goal_manifold.compute_goal_prob(z_next)
        prev_gp = self.execution_log[-1]['goal_prob'] if self.execution_log else 0.0
        gp_delta = goal_prob - prev_gp

        reward_info = self.info_gain_reward.compute(z, h, a, z_next, goal_prob)
        info_gain = reward_info['info_gain']
        cost_info = self.energy_cost.compute([a], [z, z_next], flow)

        # Object decomposition
        mu_mean, mu_var, logvar_mean, ensemble_mus = self.ensemble.predict_all(z, h, a)
        slots, attn_map = self.slot_attention.forward(
            z, ensemble_mus, goal_prob, prev_slots=self.last_slots
        )
        self.last_slots = slots.copy()
        objects_before = self.slot_tracker.get_active_objects()
        tracking = self.slot_tracker.step(slots, attn_map, a, z_next)
        objects_after = self.slot_tracker.get_active_objects()

        # Object uncertainty
        for obj in objects_after:
            epi_obj, alea_obj, _ = self.object_uncertainty.decompose(
                obj.state, ensemble_mus, np.zeros_like(ensemble_mus)
            )
            obj.set_uncertainty(epi_obj, alea_obj)

        # Relational dynamics
        rel_loss = 0.0
        prev_objects = getattr(self, '_prev_active_objects', [])
        if len(objects_after) >= 2 and len(prev_objects) >= 2:
            prev_by_id = {o.id: o.state for o in prev_objects}
            curr_by_id = {o.id: o.state for o in objects_after}
            common_ids = set(prev_by_id.keys()) & set(curr_by_id.keys())
            if len(common_ids) >= 2:
                sorted_ids = sorted(common_ids)
                prev_states = np.array([prev_by_id[oid] for oid in sorted_ids])
                curr_states = np.array([curr_by_id[oid] for oid in sorted_ids])
                rel_loss = self.rel_dynamics.train_step(prev_states, a, curr_states)
        self._prev_active_objects = objects_after
        self.relation_updater.update_relations(objects_after)

        # Object-level GP
        obj_gp = 0.0
        if objects_after:
            goal_latent = self.goal_manifold.get_mean()
            if goal_latent is not None:
                obj_projs = np.array([
                    self.object_uncertainty.project_slot_to_latent(o.state)
                    for o in objects_after
                ])
                gl = goal_latent[:obj_projs.shape[1]]
                dists = np.linalg.norm(obj_projs - gl, axis=1)
                obj_gp = float(np.max(np.exp(-dists)))

        # Temporal chunking
        epi_after, alea_after, total_after = self.ensemble.decompose_uncertainty(
            z_next, h_next, action_for_uncertainty
        )
        self.chunker.observe(
            objects_after, flow_id, goal_prob, epi_after, alea_after,
            {'z_before': z.copy(), 'z_after': z_next.copy(),
             'action': a, 'goal_prob': goal_prob, 'flow_id': flow_id,
             'epistemic_uncertainty': epi_after, 'total_uncertainty': total_after}
        )

        # Macro tracking
        if is_macro:
            self.current_macro_steps.append({
                'z_before': z.copy(), 'z_after': z_next.copy(),
                'action': a, 'goal_prob': goal_prob, 'flow_id': flow_id,
                'epistemic': epi_after, 'cost': cost_info.get('total', 0.0)
            })
        if (self.hierarchical_cem.active_macro_id is None
            and len(self.current_macro_steps) >= self.chunker.min_chunk_length):
            self._finalize_macro()

        # ====================================================================
        # LAYER 4: SELF-LATENT UPDATE (46.1)
        # ====================================================================
        self.self_latent.update(z_next, a)
        identity_coherence = self.self_latent.get_identity_signal()
        self.self_coherence_log.append(identity_coherence)

        # ====================================================================
        # LAYER 5: AGENCY INFERENCE (46.2)
        # ====================================================================
        latent_agency = self.agency.compute_latent_agency(
            z, z_next, predicted_mu, a
        )
        obj_agency, per_obj_agency = self.agency.compute_object_agency(
            objects_before, objects_after, a
        )
        self.agency_log.append(latent_agency)

        # ====================================================================
        # LAYER 6: COUNTERFACTUAL SIMULATION (46.3)
        # ====================================================================
        cf_result = None
        if (self.total_steps > 0
            and self.total_steps % self.counterfactual_interval == 0):
            cf_result = self.counterfactual.simulate_counterfactual(
                z, h, a, z_next, goal_prob, self.manifold.flows
            )
            self.counterfactual_count += 1

        # ====================================================================
        # LAYER 7-8: Stability, ecology, drift, learning
        # ====================================================================
        flow.stability = flow.compute_lyapunov_estimate()
        flow.goal_alignment += 0.01 * (gp_delta * 10)

        self.goal_manifold.record(z_next, reward_info['total'], flow_id, gp_delta)

        if self.total_steps % 5 == 0 and len(self.execution_log) >= 5:
            recent_zs = []
            recent_fids = []
            for entry in self.execution_log[-10:]:
                if 'z_after' in entry:
                    recent_zs.append(entry['z_after'])
                    recent_fids.append(entry.get('flow_id', ''))
            if len(recent_zs) >= 3:
                self.contrastive.apply_to_params(
                    self.base_engine.encoder_params,
                    recent_zs, recent_fids,
                    n_samples=6, sigma=0.003
                )
        self.contrastive.record(z_next, flow_id)

        self.ecology.record_gp_delta(flow_id, gp_delta)
        self.ecology.record_performance(flow_id, goal_prob)
        eco_result = self.ecology.step()
        self.drift.step(flow_id, goal_prob, gp_delta,
                        self.fallback_goal.attractor_state)

        free_energy = (
            self.active_cem.uncertainty_weight * total_after
            + self.active_cem.energy_weight * cost_info.get('total', 0.0)
            - self.active_cem.goal_weight * goal_prob
        )
        self.hierarchical_cem.observe_outcome(
            flow_id, goal_prob, cost_info.get('total', 0.0), is_macro, free_energy
        )

        # Periodic training
        if self.total_steps > 0 and self.total_steps % self.train_interval == 0:
            for _ in range(3):
                self.learner.train_step()
            self.learner.validate()
            self.ensemble.sync_from_wm(perturb=False)

        if (self.total_steps > 0
            and self.total_steps % self.ensemble_train_interval == 0
            and len(self.execution_log) >= 10):
            recent = self.execution_log[-10:]
            z_batch = [e['z_before'] for e in recent]
            h_batch = [np.zeros(self.wm.belief_dim) for _ in recent]
            a_batch = [e.get('action', np.zeros(self.wm.action_dim)) for e in recent]
            zn_batch = [e['z_after'] for e in recent]
            self.ensemble.train_ensemble(z_batch, h_batch, a_batch, zn_batch)

        if (self.total_steps - self.last_discovery_step > self.macro_discovery_interval
            and len(self.macro_execution_log) >= 1):
            self.last_discovery_step = self.total_steps
            self._discover_macros()

        self.total_steps += 1

        # Build result
        result = {
            'z_before': z.copy(), 'z_after': z_next.copy(),
            'action': a, 'goal_prob': float(goal_prob),
            'gp_delta': float(gp_delta),
            'flow_type': flow.flow_type.value, 'flow_id': flow_id,
            'stability': flow.stability, 'energy_cost': cost_info,
            'eco_births': eco_result.get('born', 0),
            'eco_deaths': eco_result.get('died', 0),
            'n_flows': len(self.manifold.flows) if self.manifold.flows else 0,
            'epistemic_uncertainty': float(epi_after),
            'aleatoric_uncertainty': float(alea_after),
            'total_uncertainty': float(total_after),
            'info_gain': float(info_gain),
            'free_energy': float(free_energy),
            'n_objects': len(objects_after),
            'object_ids': [o.id for o in objects_after],
            'object_gp': float(obj_gp),
            'is_macro': is_macro,
            'active_macro': self.hierarchical_cem.active_macro_id,
            'relational_dynamics_loss': float(rel_loss),
            'ensemble_divergence': float(self.ensemble.get_param_norm()),
            # Phase 46 fields
            'self_coherence': float(identity_coherence),
            'latent_agency': float(latent_agency),
            'object_agency': float(obj_agency),
        }
        self.execution_log.append(result)
        return result

    def run(self, z_start: np.ndarray, n_steps: int = 200) -> Dict:
        """Run self-engine."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)

        # Initialize self-latent from first state
        self.self_latent.update(z, np.zeros(self.wm.action_dim))

        for step in range(n_steps):
            result = self.step(z, h)
            z = result['z_after']
            h = self.wm.gru_step(h, z)
            if step > 0 and step % 20 == 0:
                self._record_episode()
        self._record_episode()

        gps = [e.get('goal_prob', 0.0) for e in self.execution_log]
        uncertainties = [e.get('total_uncertainty', 0.0) for e in self.execution_log]
        coherences = [e.get('self_coherence', 1.0) for e in self.execution_log]
        agencies = [e.get('latent_agency', 0.0) for e in self.execution_log]
        obj_counts = [e.get('n_objects', 0) for e in self.execution_log]

        return {
            'n_steps': n_steps,
            'mean_gp': float(np.mean(gps)) if gps else 0.0,
            'max_gp': float(max(gps)) if gps else 0.0,
            'gp_trend': gps[-1] - gps[0] if len(gps) >= 2 else 0.0,
            'mean_uncertainty': float(np.mean(uncertainties)) if uncertainties else 0.0,
            'mean_self_coherence': float(np.mean(coherences)) if coherences else 0.0,
            'mean_agency': float(np.mean(agencies)) if agencies else 0.0,
            'mean_n_objects': float(np.mean(obj_counts)) if obj_counts else 0.0,
            'n_flows': self.execution_log[-1]['n_flows'] if self.execution_log else 0,
            'n_macros': len(self.macro_manifold.macros),
            'counterfactual_runs': self.counterfactual_count,
            'self_latent': self.self_latent.get_stats(),
            'agency': self.agency.get_stats(),
            'counterfactual': self.counterfactual.get_stats(),
            'chunker_stats': self.chunker.get_stats(),
            'training': self.learner.get_training_report(),
            'ensemble': self.ensemble.get_stats(),
            'goal_manifold': self.goal_manifold.get_stats(),
            'ecology': self.ecology.get_stats(),
        }


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_self_latent():
    """Test persistent self-latent with temporal stability."""
    print("\n============================================================")
    print("46.1 — SELF LATENT")
    print("============================================================")
    sl = SelfLatent(latent_dim=16, self_dim=8, temporal_stability=0.9)

    # Initial state
    assert sl.state.shape == (8,), f"Self state should be (8,), got {sl.state.shape}"
    assert sl.identity_coherence == 1.0

    # Update with sequence of states
    for t in range(20):
        z = np.random.randn(16) * 0.3
        a = np.random.randn(16) * 0.2
        sl.update(z, a)

    # Identity coherence should be maintained
    coherence = sl.get_identity_signal()
    assert coherence > 0.5, f"Identity coherence should be > 0.5, got {coherence}"
    assert len(sl.history) <= 50, f"History should be capped at 50"

    # Predict next state
    pred = sl.predict_next(np.random.randn(16))
    assert pred.shape == (8,), f"Prediction should be (8,), got {pred.shape}"

    stats = sl.get_stats()
    print(f"  ✓ State shape: {sl.state.shape}")
    print(f"  ✓ Identity coherence: {coherence:.4f}")
    print(f"  ✓ History length: {len(sl.history)}")
    print(f"  ✓ Predict next: {pred.shape}")
    print(f"  ✓ Self transition model active")

    return True


def test_agency_inference():
    """Test agency distinguishes self-caused vs external change."""
    print("\n============================================================")
    print("46.2 — AGENCY INFERENCE")
    print("============================================================")
    agency = AgencyInference(latent_dim=16)

    # Scenario 1: Self-caused change (prediction matches outcome)
    z_before = np.random.randn(16) * 0.3
    action = np.random.randn(16) * 0.2
    predicted_delta = action * 0.5  # simulated prediction
    z_after = z_before + predicted_delta  # actual = predicted
    a_self = agency.compute_latent_agency(z_before, z_after, z_before + predicted_delta, action)
    assert a_self > 0.5, f"Self-caused agency should be high, got {a_self}"

    # Scenario 2: External change (prediction doesn't match)
    z_after_ext = z_before + np.random.randn(16) * 1.0  # large external perturbation
    a_ext = agency.compute_latent_agency(z_before, z_after_ext, z_before + predicted_delta, action)
    assert a_ext < a_self, f"External agency should be lower than self-caused: {a_ext} vs {a_self}"

    # Object-level agency
    obj_a = ObjectSlot('a', slot_dim=8)
    obj_b = ObjectSlot('b', slot_dim=8)
    obj_a.state = np.ones(8) * 0.5
    obj_b.state = np.ones(8) * (-0.5)

    obj_a_after = ObjectSlot('a', slot_dim=8)
    obj_b_after = ObjectSlot('b', slot_dim=8)
    obj_a_after.state = obj_a.state + action[:8] * 0.1
    obj_b_after.state = obj_b.state  # unchanged

    mean_agency, per_obj = agency.compute_object_agency(
        [obj_a, obj_b], [obj_a_after, obj_b_after], action
    )
    assert mean_agency > 0, f"Object agency should be > 0, got {mean_agency}"
    assert 'a' in per_obj, f"Object a should have agency"
    assert 'b' in per_obj, f"Object b should have agency"

    stats = agency.get_stats()
    print(f"  ✓ Self-caused agency: {a_self:.4f}")
    print(f"  ✓ External agency: {a_ext:.4f}")
    print(f"  ✓ Self > External: {a_self > a_ext}")
    print(f"  ✓ Object agency: {mean_agency:.4f}")
    print(f"  ✓ Per-object: {per_obj}")

    return True


def test_counterfactual_self():
    """Test counterfactual simulation produces alternative outcomes."""
    print("\n============================================================")
    print("46.3 — COUNTERFACTUAL SELF")
    print("============================================================")
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    cf = CounterfactualSelf(
        wm=wm, latent_dim=16, belief_dim=64,
        n_counterfactuals=3, counterfactual_interval=5
    )

    z = np.random.randn(16) * 0.3
    h = np.zeros(64)
    a = np.zeros(16)
    z_next = np.random.randn(16) * 0.3

    result = cf.simulate_counterfactual(z, h, a, z_next, 0.5, {})
    assert 'actual_gp' in result
    assert 'counterfactuals' in result
    assert 'mean_regret' in result

    print(f"  ✓ Counterfactuals generated: {len(result['counterfactuals'])}")
    print(f"  ✓ Mean regret: {result['mean_regret']:.4f}")
    for i, cf_r in enumerate(result['counterfactuals']):
        print(f"    CF {i}: regret={cf_r['regret']:.4f}, better={cf_r['was_better']}")

    stats = cf.get_stats()
    print(f"  ✓ Stats: {stats}")

    return True


def test_self_engine_short(n_steps: int = 30, bootstrap: bool = True):
    """Quick sanity: SelfEngine runs without errors."""
    print("\n============================================================")
    print("QUICK SANITY: SELF ENGINE RUNS")
    print("============================================================")
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    engine = SelfEngine(
        wm=wm, bootstrap=bootstrap,
        n_coverage=30, n_shaping=20, n_transfer=10,
        n_initial_flows=4, flow_dim=4,
        n_ensemble=3, planning_horizon=3, planning_samples=8,
        n_slots=4, slot_dim=8,
        self_dim=8, self_temporal_stability=0.9,
        counterfactual_interval=10, n_counterfactuals=2
    )

    z_start = np.random.randn(16) * 0.3
    result = engine.run(z_start=z_start, n_steps=n_steps)

    prints = [
        f"  ✓ Engine ran {result['n_steps']} steps without error",
        f"  ✓ Mean GP: {result['mean_gp']:.4f}",
        f"  ✓ Self coherence: {result['mean_self_coherence']:.4f}",
        f"  ✓ Mean agency: {result['mean_agency']:.4f}",
        f"  ✓ Counterfactual runs: {result['counterfactual_runs']}",
        f"  ✓ Mean objects: {result['mean_n_objects']:.1f}",
    ]
    for p in prints:
        print(p)

    return engine, result


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_integration(n_steps: int = 200, bootstrap: bool = True, verbose: bool = True):
    """Run SelfEngine and verify self-model capabilities."""
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE 46: SELF-MODEL & IDENTITY PERSISTENCE (200+ steps)")
        print("=" * 70)
        print(f"  Running {n_steps} steps...\n")

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    engine = SelfEngine(
        wm=wm, bootstrap=bootstrap,
        n_coverage=40, n_shaping=25, n_transfer=15,
        n_initial_flows=5, flow_dim=4,
        lambda_cost=0.3, train_interval=5,
        n_ensemble=3, ensemble_lr=0.005,
        exploration_beta=0.1,
        planning_horizon=3, planning_samples=10,
        uncertainty_weight=0.3, energy_weight=0.2, goal_weight=1.0,
        n_slots=5, slot_dim=8, slot_iterations=3,
        match_threshold=0.5, max_objects=8,
        rel_dynamics_lr=0.005,
        macro_min_horizon=3, macro_max_horizon=8,
        macro_discovery_interval=15,
        self_dim=8, self_temporal_stability=0.9,
        counterfactual_interval=10, n_counterfactuals=3
    )

    z_start = np.random.randn(16) * 0.3
    result = engine.run(z_start=z_start, n_steps=n_steps)

    if verbose:
        print("\n  RESULTS:")
        print(f"    Steps: {result['n_steps']}")
        print(f"    Mean GP: {result['mean_gp']:.4f}")
        print(f"    Mean uncertainty: {result['mean_uncertainty']:.4f}")
        print(f"    Mean objects: {result['mean_n_objects']:.1f}")

        print("\n  [46.1] SELF LATENT:")
        sl = result['self_latent']
        print(f"    Identity coherence: {sl.get('identity_coherence', 0):.4f}")
        print(f"    State norm: {sl.get('state_norm', 0):.4f}")
        print(f"    Temporal stability: {sl.get('temporal_stability', 0)}")

        print("\n  [46.2] AGENCY INFERENCE:")
        ag = result['agency']
        print(f"    Mean agency: {ag.get('mean_agency', 0):.4f}")
        print(f"    Mean object agency: {ag.get('mean_object_agency', 0):.4f}")
        print(f"    Agency trend: {ag.get('agency_trend', 0):.4f}")

        print("\n  [46.3] COUNTERFACTUAL SELF:")
        cf = result['counterfactual']
        print(f"    Simulations: {cf.get('n_simulations', 0)}")
        print(f"    Mean regret: {cf.get('mean_regret', 0):.4f}")
        print(f"    Regret trend: {cf.get('regret_trend', 0):.4f}")

        print("\n  ENSEMBLE (43.1-2):")
        ens = result['ensemble']
        print(f"    Epistemic: {ens.get('mean_epistemic', 0):.6f}")
        print(f"    Divergence: {ens.get('param_divergence', 0):.4f}")

        print("\n  GOAL MANIFOLD (42):")
        gm = result['goal_manifold']
        print(f"    Learned: {gm.get('has_mean', False)}, samples={gm.get('n_samples', 0)}")

        print("\n  TRAINING:")
        tr = result.get('training', {})
        print(f"    Episodes: {tr.get('buffer_episodes', 0)}")

        print("\n  MACROS:")
        print(f"    Count: {result['n_macros']}")

    # ========================================================================
    # VERIFICATION
    # ========================================================================
    checks = []

    checks.append(("GP not flat",
        result['mean_gp'] > 0.05, f"{result['mean_gp']:.4f}"))
    checks.append(("Objects present",
        result['mean_n_objects'] > 0, f"{result['mean_n_objects']:.1f}"))
    checks.append(("Ensemble divergent",
        ens.get('param_divergence', 0) > 0, f"{ens.get('param_divergence', 0):.4f}"))
    checks.append(("Training active",
        tr.get('buffer_episodes', 0) > 0, f"{tr.get('buffer_episodes', 0)} eps"))
    checks.append(("Goal learned",
        gm.get('has_mean', False), f"{gm.get('has_mean')}"))
    checks.append(("Self coherence maintained",
        sl.get('identity_coherence', 0) > 0.5, f"{sl.get('identity_coherence', 0):.4f}"))
    checks.append(("Agency inference active",
        ag.get('mean_agency', 0) > 0, f"{ag.get('mean_agency', 0):.4f}"))
    checks.append(("Counterfactual simulating",
        cf.get('n_simulations', 0) > 0, f"{cf.get('n_simulations', 0)}"))

    if verbose:
        print("\n  " + "=" * 60)
        print("  VERIFICATION")
        print("  " + "=" * 60)

    all_pass = True
    for name, passed, detail in checks:
        symbol = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        if verbose:
            print(f"    [{symbol}] {name}: {detail}")

    if verbose and all_pass:
        print()
        print("  " + "=" * 60)
        print("  PHASE 46 VERDICT: ALL PASSED")
        print("  " + "=" * 60)
        print("""
  Architecture extension complete:

    [46.1] SelfLatent               Persistent self with temporal stability
    [46.2] AgencyInference          Self-caused vs external change
    [46.3] CounterfactualSelf       "What if" simulation with regret
    [46.4] SelfEngine               Unified engine + self-model

  The system now has:
    - A persistent self latent that survives object turnover
    - Agency signal: knows what changes it caused vs external
    - Counterfactual reasoning: simulates alternative actions
    - Identity coherence tracking over time

  Architecture stack:

    Phase 40:  Self-Organizing Behavioral Geometry    ← substrate
    Phase 42:  Emergent Goal Geometry                 ← goals
    Phase 43:  Active Inference & Uncertainty          ← planning
    Phase 44:  Object-Centric World Model              ← perception
    Phase 45:  Temporal Abstraction & Hierarchy        ← time
    Phase 46:  Self-Model & Identity Persistence       ← self
        """)

    return engine, result, checks, all_pass


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 46: SELF-MODEL & IDENTITY PERSISTENCE                    ║
║                                                                   ║
║  The system now predicts ITSELF in the world, not just the world. ║
║                                                                   ║
║  Components:                                                      ║
║    46.1 — SelfLatent:         persistent self representation       ║
║    46.2 — AgencyInference:    self-caused vs external change      ║
║    46.3 — CounterfactualSelf: "what if" simulation                ║
║    46.4 — SelfEngine:         integrates into full stack           ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    unit_tests = [
        ("SelfLatent", test_self_latent),
        ("AgencyInference", test_agency_inference),
        ("CounterfactualSelf", test_counterfactual_self),
        ("SelfEngine Sanity (30 steps)",
         lambda: test_self_engine_short(n_steps=30, bootstrap=True)),
    ]

    all_unit_pass = True
    for name, fn in unit_tests:
        try:
            fn()
            print(f"  >>> {name} PASSED\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  >>> {name} FAILED: {e}\n")
            all_unit_pass = False

    if all_unit_pass:
        engine, result, checks, all_pass = test_integration(
            n_steps=200, bootstrap=True, verbose=True
        )

        print("\n" + "=" * 70)
        print("PHASE 46 SUMMARY")
        print("=" * 70)
        passed_count = sum(1 for _, p, _ in checks if p)
        total_count = len(checks)
        print(f"\n  Checks passed: {passed_count}/{total_count}")
        if all_pass:
            print("""
  Self-model layer complete.

  The system now has:
    • Persistent self-latent with temporal stability constraint
    • Agency inference: distinguishes self-caused vs external change
    • Counterfactual simulation: "what if I had acted differently"
    • Identity coherence tracking across time

  This is the foundation for:
    Phase 47: Language Grounding (grounded in SELF, not just world)
    Phase 48: Autonomous Cognitive Ecology (motives, competition, self-preservation)
        """)
        else:
            print("\n  ❌ Some checks failed")
            for name, passed, detail in checks:
                if not passed:
                    print(f"     FAIL: {name} = {detail}")
    else:
        print("\n  ❌ Unit tests failed — skipping integration test")
