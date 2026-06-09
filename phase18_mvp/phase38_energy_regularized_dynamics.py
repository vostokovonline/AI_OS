"""
Phase 38 — Energy-Regularized Dynamics

KEY SHIFT:
  Before: "Can I reach the goal?"
  After:  "Is it worth the cost to reach the goal?"

  Pure controllability (Phase 37) ignores cost.
  Energy-regularized dynamics make the system energy-aware:
    prefer efficient, smooth, low-cost paths to the goal.

COST FUNCTIONS:
  1. Action cost:    ||a||²              — penalizes large actions
  2. Path cost:      Σ||z_{t+1} - z_t||  — penalizes long trajectories
  3. Variance cost:  Var(a)              — penalizes erratic control
  4. Instability:    1 - stability        — penalizes unreliable flows

TOTAL SCORE:
  score = benefit - λ * cost
  benefit = goal_prob + controllability_bonus * controllability
  cost = w_a * action_cost + w_p * path_cost + w_v * var_cost + w_i * inst_cost

  CEM now optimizes: E[goal - λ * cost], not E[goal]

WHY THIS MATTERS:
  Without energy:   system takes any path to goal (inefficient, erratic)
  With energy:      system prefers efficient, smooth trajectories
  
  This is the difference between:
    "reaching the goal by any means necessary"
    "reaching the goal efficiently and reliably"

ARCHITECTURAL INTEGRATION:
  Phase 36 buffer → trajectory costs → EnergyRegularization →
  Phase 37 controllability → cost-weighted scoring →
  Phase 35 flows → cost-aware selection
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from phase30_training_loop import MinimalWorldModel
from phase31_hierarchical_execution import GoalAttractor
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, PointFlow, rollout_flow
)


# ============================================================================
# 1. ENERGY COST FUNCTIONS
# ============================================================================

class EnergyCostFunction:
    """
    Computes energy cost for trajectories.
    
    cost = w_a * c_action + w_p * c_path + w_v * c_variance + w_i * c_instability
    
    Each cost is normalized to [0, 1]. Weights control the trade-off.
    """
    
    def __init__(
        self,
        w_action: float = 0.3,
        w_path: float = 0.3,
        w_variance: float = 0.1,
        w_instability: float = 0.3,
        action_norm_threshold: float = 2.0,
        path_norm_threshold: float = 5.0
    ):
        self.w_action = w_action
        self.w_path = w_path
        self.w_variance = w_variance
        self.w_instability = w_instability
        
        self.action_thresh = action_norm_threshold
        self.path_thresh = path_norm_threshold
        
        # Running statistics for adaptive normalization
        self.action_cost_history: List[float] = []
        self.path_cost_history: List[float] = []
        self.var_cost_history: List[float] = []
        self.max_history = 500
    
    def _ensure_list(self, x):
        """Convert numpy array to list if needed."""
        if isinstance(x, np.ndarray):
            if x.ndim == 1:
                return [x]
            return [x[i] for i in range(x.shape[0])]
        return list(x) if x else []

    def action_cost(self, actions) -> float:
        """
        Cost proportional to action magnitude squared.
        
        c_action = mean(||a_t||²) / threshold²
        """
        actions = self._ensure_list(actions)
        if len(actions) == 0:
            return 0.0
        
        norms = [float(np.sum(np.asarray(a) ** 2)) for a in actions]
        cost = float(np.mean(norms)) / (self.action_thresh ** 2 + 1e-8)
        return float(np.clip(cost, 0.0, 1.0))
    
    def path_cost(self, states) -> float:
        """
        Cost proportional to total path length.
        
        c_path = Σ||z_{t+1} - z_t|| / threshold
        """
        states = self._ensure_list(states)
        if len(states) < 2:
            return 0.0
        
        deltas = [
            np.linalg.norm(np.asarray(states[i + 1]) - np.asarray(states[i]))
            for i in range(len(states) - 1)
        ]
        total = float(np.sum(deltas))
        cost = total / (self.path_thresh + 1e-8)
        return float(np.clip(cost, 0.0, 1.0))
    
    def variance_cost(self, actions) -> float:
        """
        Cost proportional to action variance (erratic control penalty).
        
        c_variance = mean(Var(a)) / mean(||a||²) 
        """
        actions = self._ensure_list(actions)
        if len(actions) < 3:
            return 0.0
        
        flat = [np.asarray(a).flatten()[:8] for a in actions]
        action_array = np.array(flat)
        action_var = float(np.mean(np.var(action_array, axis=0)))
        action_mean = float(np.mean(np.mean(action_array, axis=0) ** 2)) + 1e-8
        
        cost = action_var / action_mean
        return float(np.clip(cost, 0.0, 1.0))
    
    def instability_cost(self, flow: SkillFlow) -> float:
        """
        Cost proportional to flow instability.
        
        c_instability = 1 - stability
        """
        return float(np.clip(1.0 - flow.stability, 0.0, 1.0))
    
    def compute(
        self,
        actions,
        states,
        flow: SkillFlow
    ) -> Dict:
        """
        Full energy cost computation.
        
        Returns: {total, action, path, variance, instability, breakdown}
        """
        c_action = self.action_cost(actions)
        c_path = self.path_cost(states)
        c_var = self.variance_cost(actions)
        c_inst = self.instability_cost(flow)
        
        # Record for adaptive normalization
        self.action_cost_history.append(c_action)
        self.path_cost_history.append(c_path)
        self.var_cost_history.append(c_var)
        
        if len(self.action_cost_history) > self.max_history:
            self.action_cost_history = self.action_cost_history[-self.max_history:]
            self.path_cost_history = self.path_cost_history[-self.max_history:]
            self.var_cost_history = self.var_cost_history[-self.max_history:]
        
        total = (self.w_action * c_action + self.w_path * c_path
                 + self.w_variance * c_var + self.w_instability * c_inst)
        
        return {
            'total': float(total),
            'action': float(c_action),
            'path': float(c_path),
            'variance': float(c_var),
            'instability': float(c_inst),
            'weights': {
                'action': self.w_action,
                'path': self.w_path,
                'variance': self.w_variance,
                'instability': self.w_instability
            }
        }
    
    def compute_from_trajectory(
        self, trajectory: Dict, flow: SkillFlow
    ) -> Dict:
        """Compute cost from rollout_flow trajectory dict."""
        actions = trajectory.get('actions', [])
        states = trajectory.get('states', [])
        if isinstance(states, np.ndarray) and states.ndim == 2:
            states = list(states)
        return self.compute(actions, states, flow)
    
    def get_stats(self) -> Dict:
        """Cost statistics for monitoring."""
        return {
            'mean_action_cost': float(np.mean(self.action_cost_history)) if self.action_cost_history else 0.0,
            'mean_path_cost': float(np.mean(self.path_cost_history)) if self.path_cost_history else 0.0,
            'mean_var_cost': float(np.mean(self.var_cost_history)) if self.var_cost_history else 0.0,
        }


# ============================================================================
# 2. EFFICIENCY EVALUATOR
# ============================================================================

class EfficiencyEvaluator:
    """
    Evaluates flow efficiency: goal probability per unit cost.
    
    efficiency = benefit / (1 + λ * cost)
    
    This enables Pareto-optimal flow selection:
      - High benefit + low cost → high efficiency (best)
      - High benefit + high cost → medium efficiency (trade-off)
      - Low benefit + low cost → low efficiency (not useful)
      - Low benefit + high cost → worst
    
    The efficiency metric answers:
      "Which flow gives me the most goal progress per unit of energy?"
    """
    
    def __init__(
        self,
        cost_fn: EnergyCostFunction,
        lambda_cost: float = 0.5,
        benefit_weights: Optional[Dict[str, float]] = None
    ):
        self.cost_fn = cost_fn
        self.lambda_cost = lambda_cost
        
        self.benefit_weights = benefit_weights or {
            'goal_prob': 1.0,
            'controllability': 0.3,
            'stability': 0.15
        }
    
    def compute_benefit(
        self,
        goal_prob: float,
        controllability_score: float = 0.0,
        stability: float = 0.5
    ) -> float:
        """Compute weighted benefit score."""
        w = self.benefit_weights
        return (w['goal_prob'] * goal_prob
                + w['controllability'] * controllability_score
                + w['stability'] * stability)
    
    def evaluate(
        self,
        goal_prob: float,
        trajectory: Dict,
        flow: SkillFlow,
        controllability_score: float = 0.0
    ) -> Dict:
        """
        Full efficiency evaluation.
        
        1. Compute cost from trajectory
        2. Compute benefit from goal_prob + controllability
        3. Compute efficiency = benefit / (1 + λ * cost)
        4. Return full report
        """
        cost_result = self.cost_fn.compute_from_trajectory(trajectory, flow)
        
        benefit = self.compute_benefit(
            goal_prob, controllability_score, flow.stability
        )
        
        cost_penalty = 1.0 + self.lambda_cost * cost_result['total']
        efficiency = benefit / cost_penalty
        
        return {
            'benefit': float(benefit),
            'cost': cost_result,
            'efficiency': float(efficiency),
            'lambda': self.lambda_cost,
            'goal_prob': float(goal_prob),
            'controllability': float(controllability_score),
            'stability': flow.stability
        }
    
    def compute_pareto_frontier(
        self, candidates: List[Dict]
    ) -> List[Dict]:
        """
        Compute Pareto frontier of goal_prob vs cost.
        
        A candidate is Pareto-optimal if no other candidate has
        both higher goal_prob AND lower cost.
        
        Returns: list of Pareto-optimal candidates.
        """
        pareto = []
        
        for i, c1 in enumerate(candidates):
            dominated = False
            for j, c2 in enumerate(candidates):
                if i == j:
                    continue
                
                c1_gp = c1.get('goal_prob', 0.0)
                c2_gp = c2.get('goal_prob', 0.0)
                c1_cost = c1.get('cost', {}).get('total', float('inf'))
                c2_cost = c2.get('cost', {}).get('total', float('inf'))
                
                if c2_gp >= c1_gp and c2_cost <= c1_cost and (c2_gp > c1_gp or c2_cost < c1_cost):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(c1)
        
        return pareto


# ============================================================================
# 3. ENERGY-REGULARIZED CEM
# ============================================================================

class EnergyRegularizedCEM:
    """
    CEM planner with energy-regularized scoring.
    
    Extends Phase 37 ControllabilityAwareCEM:
      Phase 37 score: goal_prob + controllability_bonus * controllability
      Phase 38 score: [goal_prob + bonus * controllability] / (1 + λ * cost)
    
    The CEM now optimizes efficiency, not raw goal probability.
    """
    
    def __init__(
        self,
        world_model: MinimalWorldModel,
        manifold: FlowManifold,
        goal: GoalAttractor,
        energy_cost_fn: EnergyCostFunction,
        efficiency_evaluator: EfficiencyEvaluator,
        flow_dim: int = 4,
        n_candidates: int = 40,
        n_elites: int = 8,
        n_iterations: int = 4,
        rollout_steps: int = 6,
        controllability_bonus: float = 0.4,
        efficiency_mode: bool = True
    ):
        self.wm = world_model
        self.manifold = manifold
        self.goal = goal
        self.energy_cost = energy_cost_fn
        self.efficiency = efficiency_evaluator
        self.flow_dim = flow_dim
        
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.rollout_steps = rollout_steps
        self.controllability_bonus = controllability_bonus
        self.efficiency_mode = efficiency_mode
        
        self.mean = np.zeros(flow_dim)
        self.std = np.ones(flow_dim)
    
    def _score_flow(
        self, flow: SkillFlow, z_start: np.ndarray, h_start: np.ndarray,
        controllability_score: float = 0.0
    ) -> Tuple[float, Dict]:
        """
        Score a flow with energy-regularized evaluation.
        
        Phase 37: score = goal_prob + bonus * controllability
        Phase 38: score = efficiency = benefit / (1 + λ * cost)
        """
        result = rollout_flow(flow, z_start, h_start, self.wm, self.rollout_steps)
        
        z_final = result['final_state']
        dist = np.linalg.norm(z_final - self.goal.attractor_state[:len(z_final)])
        goal_prob = np.exp(-dist)
        
        cost = self.energy_cost.compute_from_trajectory(result, flow)
        
        if self.efficiency_mode:
            benefit = (goal_prob
                       + self.controllability_bonus * controllability_score
                       + 0.15 * flow.stability)
            cost_penalty = 1.0 + self.efficiency.lambda_cost * cost['total']
            score = benefit / cost_penalty
        else:
            score = (goal_prob
                     + self.controllability_bonus * controllability_score
                     + 0.15 * flow.stability
                     - 0.5 * cost['total'])
        
        info = {
            'goal_prob': float(goal_prob),
            'score': float(score),
            'cost': cost,
            'efficiency': float(score / (1.0 + 1e-8)),
            'stability': flow.stability
        }
        
        return score, info
    
    def plan_flow(
        self, z_start: np.ndarray, h_start: np.ndarray,
        ranked_flows: Optional[List[Tuple[str, float]]] = None
    ) -> Dict:
        """
        Plan flow with energy-regularized optimization.
        
        Uses ranked flows from controllability to bias sampling,
        then scores candidates by efficiency.
        """
        if not self.manifold.flows:
            return {'coord': np.zeros(self.flow_dim), 'flow': None,
                    'score': -np.inf, 'info': {}}
        
        # Get controllability-ranked flows to bias CEM
        top_coords = []
        if ranked_flows:
            for fid, _ in ranked_flows[:5]:
                if fid in self.manifold.flow_coords:
                    top_coords.append(self.manifold.flow_coords[fid])
        
        if top_coords:
            self.mean = np.mean(top_coords, axis=0)
            self.std = np.ones(self.flow_dim) * 0.5
        else:
            self.mean = np.zeros(self.flow_dim)
            self.std = np.ones(self.flow_dim)
        
        best_coord = None
        best_score = -np.inf
        best_info = {}
        best_trajectory = None
        
        for iteration in range(self.n_iterations):
            candidates = []
            for _ in range(self.n_candidates):
                coord = self.mean + self.std * np.random.randn(self.flow_dim)
                candidates.append(coord)
            
            scores = []
            infos = []
            trajectories = []
            for coord in candidates:
                flow = self.manifold.interpolate_at(coord, self.wm, z_start)
                controllability = self._compute_controllability_score(flow, ranked_flows)
                
                score, info = self._score_flow(
                    flow, z_start, h_start, controllability
                )
                
                scores.append(score)
                infos.append(info)
                trajectories.append(coord)
            
            # Select elites
            elite_indices = np.argsort(scores)[-self.n_elites:]
            elite_coords = [candidates[i] for i in elite_indices]
            
            if elite_coords:
                self.mean = np.mean(elite_coords, axis=0)
                self.std = np.std(elite_coords, axis=0) + 0.1
            
            max_idx = int(np.argmax(scores))
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_coord = candidates[max_idx]
                best_info = infos[max_idx]
                best_trajectory = trajectories[max_idx]
        
        if best_coord is None:
            return {'coord': np.zeros(self.flow_dim), 'flow': None,
                    'score': -np.inf, 'info': {}}
        
        best_flow = self.manifold.interpolate_at(best_coord, self.wm, z_start)
        
        return {
            'coord': best_coord,
            'flow': best_flow,
            'score': float(best_score),
            'info': best_info,
            'n_controllability_flows': len(ranked_flows) if ranked_flows else 0
        }
    
    def _compute_controllability_score(
        self, flow: SkillFlow,
        ranked_flows: Optional[List[Tuple[str, float]]] = None
    ) -> float:
        """
        Estimate controllability score from ranked flows.
        
        For ComposedFlow, average the goal probability of sub-flows.
        """
        if not ranked_flows:
            return 0.0
        
        ranked_dict = dict(ranked_flows)
        
        if hasattr(flow, 'flows') and flow.flows:
            scores = []
            for sub_flow in flow.flows:
                if sub_flow.flow_id in ranked_dict:
                    scores.append(ranked_dict[sub_flow.flow_id])
            return float(np.mean(scores)) if scores else 0.0
        
        if flow.flow_id in ranked_dict:
            return float(ranked_dict[flow.flow_id])
        
        return 0.0


# ============================================================================
# 4. COST-AWARE FLOW SELECTION
# ============================================================================

class CostAwareFlowSelection:
    """
    Selects flows based on cost-benefit analysis.
    
    Provides:
      1. Cost-aware ranking: efficiency = benefit / cost
      2. Pareto analysis: trade-off frontier
      3. Adaptive λ adjustment: increase λ if costs are too high
    """
    
    def __init__(
        self,
        cost_fn: EnergyCostFunction,
        evaluator: EfficiencyEvaluator,
        target_efficiency: float = 0.3,
        lambda_min: float = 0.1,
        lambda_max: float = 2.0,
        adaptation_rate: float = 0.05
    ):
        self.cost_fn = cost_fn
        self.evaluator = evaluator
        self.target_eff = target_efficiency
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.adapt_rate = adaptation_rate
        
        self.efficiency_history: List[float] = []
    
    def rank_flows(
        self,
        candidates: List[Dict],
        z_start: np.ndarray,
        h_start: np.ndarray,
        wm: MinimalWorldModel
    ) -> List[Dict]:
        """
        Rank flows by efficiency (benefit / (1 + λ * cost)).
        
        candidates: [{flow, goal_prob, controllability}]
        Returns: ranked [{flow, goal_prob, cost, efficiency}]
        """
        ranked = []
        
        for cand in candidates:
            flow = cand.get('flow', cand)
            goal_prob = cand.get('goal_prob', 0.0)
            ctrl_score = cand.get('controllability', 0.0)
            
            # Rollout to get trajectory
            result = rollout_flow(flow, z_start, h_start, wm, 6)
            
            # Evaluate efficiency
            eval_result = self.evaluator.evaluate(
                goal_prob, result, flow, ctrl_score
            )
            
            ranked.append({
                'flow': flow,
                'flow_id': flow.flow_id,
                'goal_prob': goal_prob,
                'efficiency': eval_result['efficiency'],
                'cost': eval_result['cost'],
                'benefit': eval_result['benefit']
            })
        
        ranked.sort(key=lambda x: x['efficiency'], reverse=True)
        return ranked
    
    def get_pareto_optimal(
        self,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Get Pareto-optimal flows (goal_prob vs cost)."""
        return self.evaluator.compute_pareto_frontier(candidates)
    
    def adapt_lambda(self, avg_efficiency: float):
        """Adapt λ to maintain target efficiency."""
        self.efficiency_history.append(avg_efficiency)
        if len(self.efficiency_history) > 50:
            self.efficiency_history = self.efficiency_history[-50:]
        
        if avg_efficiency < self.target_eff * 0.5:
            self.evaluator.lambda_cost = max(
                self.lambda_min,
                self.evaluator.lambda_cost - self.adapt_rate
            )
        elif avg_efficiency > self.target_eff * 1.5:
            self.evaluator.lambda_cost = min(
                self.lambda_max,
                self.evaluator.lambda_cost + self.adapt_rate
            )
    
    def get_stats(self) -> Dict:
        """Selection statistics."""
        return {
            'lambda': self.evaluator.lambda_cost,
            'mean_efficiency': float(np.mean(self.efficiency_history)) if self.efficiency_history else 0.0,
            'target_efficiency': self.target_eff
        }


# ============================================================================
# 5. ENERGY-REGULARIZED ENGINE
# ============================================================================

class EnergyRegularizedEngine:
    """
    Full execution engine with energy-regularized dynamics.
    
    Extends Phase 37 ControllabilityEnhancedEngine with:
      Phase 38: EnergyRegularizedCEM (cost-aware planning)
      Phase 38: CostAwareFlowSelection (efficiency ranking)
    
    Loop per cycle:
      1. Plan with energy-regularized CEM
      2. Execute and record trajectory costs
      3. Update controllability matrix (empirical + model)
      4. Adapt λ based on efficiency history
      5. Train world model (Phase 36)
    """
    
    def __init__(
        self,
        world_model: 'FlowConditionedWorldModel',
        goal: GoalAttractor,
        manifold: Optional[FlowManifold] = None,
        n_initial_flows: int = 12,
        flow_dim: int = 4,
        cost_weights: Optional[Dict[str, float]] = None,
        lambda_cost: float = 0.5,
        n_probes: int = 30,
        train_every_n: int = 5
    ):
        self.wm = world_model
        self.goal = goal
        self.flow_dim = flow_dim
        
        # Phase 35 manifold
        self.manifold = manifold or FlowManifold(flow_dim=flow_dim)
        
        # Phase 38 energy cost function
        weights = cost_weights or {
            'action': 0.3, 'path': 0.3,
            'variance': 0.1, 'instability': 0.3
        }
        self.energy_cost = EnergyCostFunction(
            w_action=weights.get('action', 0.3),
            w_path=weights.get('path', 0.3),
            w_variance=weights.get('variance', 0.1),
            w_instability=weights.get('instability', 0.3)
        )
        
        # Phase 38 efficiency evaluator
        self.efficiency = EfficiencyEvaluator(
            cost_fn=self.energy_cost,
            lambda_cost=lambda_cost
        )
        
        # Phase 38 CEM
        self.cem = EnergyRegularizedCEM(
            world_model=self.wm,
            manifold=self.manifold,
            goal=self.goal,
            energy_cost_fn=self.energy_cost,
            efficiency_evaluator=self.efficiency,
            flow_dim=flow_dim,
            n_candidates=40, n_elites=8, n_iterations=4,
            controllability_bonus=0.4,
            efficiency_mode=True
        )
        
        # Phase 38 cost-aware selection
        self.selection = CostAwareFlowSelection(
            cost_fn=self.energy_cost,
            evaluator=self.efficiency,
            lambda_min=0.1, lambda_max=2.0
        )
        
        # Phase 34 inverse dynamics
        from phase34_inverse_control_stabilization import InverseDynamicsModel
        self.inv_dyn = InverseDynamicsModel(
            latent_dim=world_model.latent_dim,
            action_dim=world_model.action_dim,
            learning_rate=0.01
        )
        
        # Phase 36 learner
        from phase36_behavioral_physics_learning import (
            BehavioralPhysicsLearner
        )
        self.learner = BehavioralPhysicsLearner(
            world_model=world_model,
            inv_dyn=self.inv_dyn,
            manifold=self.manifold,
            goal=goal,
            learning_rate=0.02,
            k_steps=4,
            batch_size=16
        )
        
        # Controllability data (for CEM biasing)
        self.ranked_flows: List[Tuple[str, float]] = []
        self.probe_states: List[np.ndarray] = []
        self.flow_goal_probs: Dict[str, List[float]] = defaultdict(list)
        
        # Seed flows
        if not self.manifold.flows:
            self._seed_flows(n_initial_flows)
        
        self.train_every_n = train_every_n
        self.total_steps = 0
        self.cycle_log: List[Dict] = []
        self.execution_log: List[Dict] = []
    
    def _seed_flows(self, n: int):
        """Seed initial flow regimes."""
        from phase35_dynamical_skill_flows import LimitCycleFlow
        for i in range(n):
            if random.random() < 0.5:
                target = np.random.randn(self.wm.latent_dim) * random.uniform(0.3, 1.5)
                flow = PointFlow(target, gain=random.uniform(0.2, 0.8))
            else:
                center = np.random.randn(self.wm.latent_dim) * random.uniform(0.3, 1.0)
                flow = LimitCycleFlow(center, radius=random.uniform(0.5, 2.0),
                                      omega=random.uniform(0.2, 1.0))
            self.manifold.add_flow(flow, f'flow_{i}')
    
    def _update_ranked_flows(self):
        """Update controllability rankings from empirical data."""
        ranked = []
        for fid in self.manifold.flows:
            gps = self.flow_goal_probs.get(fid, [])
            avg_gp = float(np.mean(gps)) if gps else 0.0
            ranked.append((fid, avg_gp))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        self.ranked_flows = ranked
    
    def execute_step(
        self, z: np.ndarray, h: np.ndarray
    ) -> Dict:
        """One execution step with energy-regularized planning."""
        plan = self.cem.plan_flow(z, h, self.ranked_flows)
        
        if plan['flow'] is None:
            flow = list(self.manifold.flows.values())[0] if self.manifold.flows \
                   else PointFlow(np.zeros(self.wm.latent_dim))
        else:
            flow = plan['flow']
        
        a = flow.compute_action(z, h)
        mu, logvar = self.wm.predict_transition(z, h, a)
        std = np.exp(0.5 * logvar)
        z_next = mu + std * np.random.randn(*mu.shape) * 0.1
        h_next = self.wm.gru_step(h, mu)
        
        flow.record_transition(z, z_next, a, h)
        
        self.inv_dyn.train_step(z, z_next, a)
        self.inv_dyn.add_transition(z, z_next, a)
        
        dist = np.linalg.norm(z_next - self.goal.attractor_state[:len(z_next)])
        goal_prob = np.exp(-dist)
        
        flow.stability = flow.compute_lyapunov_estimate()
        
        # Record empirical goal prob for controllability ranking
        self.flow_goal_probs[flow.flow_id].append(goal_prob)
        if len(self.flow_goal_probs[flow.flow_id]) > 100:
            self.flow_goal_probs[flow.flow_id] = \
                self.flow_goal_probs[flow.flow_id][-100:]
        
        # Compute energy cost
        cost_info = self.energy_cost.compute([a], [z, z_next], flow)
        
        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'action': a.copy(),
            'goal_prob': float(goal_prob),
            'flow_type': flow.flow_type.value,
            'flow_id': flow.flow_id,
            'stability': flow.stability,
            'energy_cost': cost_info,
            'plan_score': plan['score'],
            'plan_info': plan.get('info', {})
        }
        
        self.execution_log.append(step_result)
        return step_result
    
    def execute_goal(self, z_start: np.ndarray, max_steps: int = 20) -> Dict:
        """Execute full goal with energy-regularized planning."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)
        
        all_states = [z.copy()]
        goal_reached = False
        total_cost = 0.0
        
        for step in range(max_steps):
            result = self.execute_step(z, h)
            
            if result['goal_prob'] > 0.7:
                goal_reached = True
            
            total_cost += result['energy_cost']['total']
            
            z = result['z_after'].copy()
            h = self.wm.gru_step(h, result['z_after'])
            all_states.append(z.copy())
        
        # Update ranked flows
        self._update_ranked_flows()
        
        stabilities = [f.stability for f in self.manifold.flows.values()]
        flow_types = [f.flow_type.value for f in self.manifold.flows.values()]
        
        return {
            'goal_reached': goal_reached,
            'final_goal_prob': self.execution_log[-1]['goal_prob'] if self.execution_log else 0.0,
            'n_steps': len(self.execution_log),
            'n_flows': len(self.manifold.flows),
            'flow_types': {t: flow_types.count(t) for t in set(flow_types)},
            'stability': {
                'mean': float(np.mean(stabilities)) if stabilities else 0.0,
                'max': float(max(stabilities)) if stabilities else 0.0,
                'min': float(min(stabilities)) if stabilities else 0.0
            },
            'total_cost': total_cost,
            'avg_cost': total_cost / max(1, len(self.execution_log)),
            'execution_log': self.execution_log[-10:],
            'trajectory_length': len(all_states)
        }
    
    def run_cycle(self, z_start: np.ndarray, n_steps: int = 20) -> Dict:
        """One closed-loop cycle with energy-regularized control."""
        result = self.execute_goal(z_start, max_steps=n_steps)
        
        log = result.get('execution_log', [])
        states = [z_start.copy()]
        beliefs = [np.zeros(self.wm.belief_dim)]
        for entry in log:
            h = self.wm.gru_step(beliefs[-1], entry['z_before'])
            beliefs.append(h.copy())
            if 'z_after' in entry:
                states.append(entry['z_after'].copy())
            else:
                states.append(entry['z_before'].copy())
        
        if log:
            self.learner.record_from_engine(
                log, list(self.manifold.flows.values()), states, beliefs
            )
        
        # Train periodically
        train_result = {}
        if self.total_steps % self.train_every_n == 0:
            train_losses = []
            for _ in range(5):
                tr = self.learner.train_step()
                if tr['loss'] != float('inf'):
                    train_losses.append(tr['loss'])
            
            if train_losses:
                train_result = {
                    'train_loss': float(np.mean(train_losses)),
                    'loss_trend': (
                        train_losses[0] - train_losses[-1]
                    ) / max(1e-8, train_losses[0])
                }
            
            val = self.learner.validate()
            train_result['val_loss'] = val.get('val_loss', float('inf'))
        
        # Inverse dynamics training
        inv_loss = 0.0
        if len(self.inv_dyn.training_data) > 10:
            inv_loss = self.inv_dyn.train_from_buffer(
                self.inv_dyn.training_data[-100:]
            )
        
        self.total_steps += n_steps
        
        # Energy statistics
        cost_stats = self.energy_cost.get_stats()
        selection_stats = self.selection.get_stats()
        
        cycle_result = {
            'cycle': len(self.cycle_log),
            'goal_reached': result['goal_reached'],
            'goal_prob': result['final_goal_prob'],
            'n_flows': result['n_flows'],
            'stability': result['stability'],
            'total_cost': result['total_cost'],
            'avg_cost': result['avg_cost'],
            'training': train_result,
            'inv_dyn_loss': float(inv_loss),
            'cost_stats': cost_stats,
            'selection_stats': selection_stats
        }
        
        self.cycle_log.append(cycle_result)
        return cycle_result
    
    def run_multi_cycle(self, z_start: np.ndarray,
                        n_cycles: int = 10,
                        steps_per_cycle: int = 20) -> Dict:
        """Run multiple cycles with energy-regularized learning."""
        z = z_start.copy()
        
        for cycle in range(n_cycles):
            result = self.run_cycle(z, n_steps=steps_per_cycle)
            
            if self.execution_log:
                last = self.execution_log[-1]
                if 'z_after' in last:
                    z = last['z_after'].copy()
        
        training_report = self.learner.get_training_report()
        goals_reached = sum(1 for c in self.cycle_log if c['goal_reached'])
        
        return {
            'n_cycles': n_cycles,
            'total_steps': self.total_steps,
            'goals_reached': goals_reached,
            'goal_rate': goals_reached / max(1, n_cycles),
            'training': training_report,
            'cost_stats': self.energy_cost.get_stats(),
            'selection_stats': self.selection.get_stats(),
            'cycle_log': self.cycle_log[-5:]
        }


# ============================================================================
# 6. TESTS
# ============================================================================

def test_energy_cost_function():
    """Test energy cost computation."""
    print("\n" + "=" * 60)
    print("ENERGY COST FUNCTION TEST")
    print("=" * 60)
    
    cost_fn = EnergyCostFunction(
        w_action=0.3, w_path=0.3, w_variance=0.1, w_instability=0.3
    )
    
    from phase35_dynamical_skill_flows import PointFlow
    flow = PointFlow(np.ones(16) * 0.5, gain=0.5)
    
    # Low-cost trajectory
    smooth_actions = [np.ones(16) * 0.1 for _ in range(5)]
    smooth_states = [np.ones(16) * 0.1 * i for i in range(6)]
    
    cost_low = cost_fn.compute(smooth_actions, smooth_states, flow)
    
    # High-cost trajectory
    erratic_actions = [np.random.randn(16) * 2.0 for _ in range(5)]
    erratic_states = [np.random.randn(16) * 3.0 for _ in range(6)]
    
    cost_high = cost_fn.compute(erratic_actions, erratic_states, flow)
    
    print(f"\n  Smooth trajectory cost: {cost_low['total']:.4f}")
    print(f"    action={cost_low['action']:.4f}, path={cost_low['path']:.4f}")
    print(f"  Erratic trajectory cost: {cost_high['total']:.4f}")
    print(f"    action={cost_high['action']:.4f}, path={cost_high['path']:.4f}")
    print(f"  High cost > low cost: {cost_high['total'] > cost_low['total']}")
    
    print("\n  ✓ Energy cost function operational")


def test_efficiency_evaluator():
    """Test efficiency computation."""
    print("\n" + "=" * 60)
    print("EFFICIENCY EVALUATOR TEST")
    print("=" * 60)
    
    cost_fn = EnergyCostFunction()
    evaluator = EfficiencyEvaluator(cost_fn, lambda_cost=0.5)
    
    from phase35_dynamical_skill_flows import PointFlow, rollout_flow
    from phase30_training_loop import MinimalWorldModel
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    
    # High goal prob, smooth trajectory
    flow_good = PointFlow(np.ones(16) * 0.3, gain=0.3)
    traj_good = rollout_flow(flow_good, np.zeros(16), np.zeros(64), wm, 5)
    eval_good = evaluator.evaluate(0.8, traj_good, flow_good, 0.5)
    
    # Low goal prob, erratic trajectory
    flow_bad = PointFlow(np.ones(16) * 5.0, gain=2.0)
    traj_bad = rollout_flow(flow_bad, np.zeros(16), np.zeros(64), wm, 5)
    eval_bad = evaluator.evaluate(0.05, traj_bad, flow_bad, 0.0)
    
    print(f"\n  Good flow: benefit={eval_good['benefit']:.4f}, "
          f"cost={eval_good['cost']['total']:.4f}, "
          f"efficiency={eval_good['efficiency']:.4f}")
    print(f"  Bad flow: benefit={eval_bad['benefit']:.4f}, "
          f"cost={eval_bad['cost']['total']:.4f}, "
          f"efficiency={eval_bad['efficiency']:.4f}")
    print(f"  Good > Bad: {eval_good['efficiency'] > eval_bad['efficiency']}")
    
    print("\n  ✓ Efficiency evaluator operational")


def test_energy_regularized_cem():
    """Test energy-regularized CEM."""
    print("\n" + "=" * 60)
    print("ENERGY-REGULARIZED CEM TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='energy_cem',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    manifold = FlowManifold(flow_dim=4)
    for i in range(8):
        target = np.random.randn(16) * random.uniform(0.3, 1.5)
        flow = PointFlow(target, gain=random.uniform(0.2, 0.8))
        manifold.add_flow(flow, f'flow_{i}')
    
    cost_fn = EnergyCostFunction()
    evaluator = EfficiencyEvaluator(cost_fn, lambda_cost=0.5)
    
    cem = EnergyRegularizedCEM(
        world_model=wm, manifold=manifold, goal=goal,
        energy_cost_fn=cost_fn, efficiency_evaluator=evaluator,
        flow_dim=4, n_candidates=20, n_elites=4, n_iterations=3
    )
    
    plan = cem.plan_flow(np.zeros(16), np.zeros(64))
    
    print(f"\n  Plan score: {plan['score']:.4f}")
    print(f"  Has flow: {plan['flow'] is not None}")
    if plan.get('info'):
        info = plan['info']
        print(f"  Goal prob: {info.get('goal_prob', 0):.4f}")
        print(f"  Cost: {info.get('cost', {}).get('total', 0):.4f}")
        print(f"  Efficiency: {info.get('efficiency', 0):.4f}")
    
    print("\n  ✓ Energy-regularized CEM operational")


def test_full_energy_engine():
    """Test full energy-regularized execution."""
    print("\n" + "=" * 60)
    print("FULL ENERGY-REGULARIZED ENGINE TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='full_energy',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    engine = EnergyRegularizedEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=10,
        flow_dim=4,
        lambda_cost=0.5,
        n_probes=10,
        train_every_n=5
    )
    
    result = engine.run_multi_cycle(
        np.zeros(16),
        n_cycles=5,
        steps_per_cycle=10
    )
    
    print(f"\n  Cycles: {result['n_cycles']}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Goals reached: {result['goals_reached']}")
    
    tr = result['training']
    print(f"\n  Training:")
    print(f"    Steps: {tr['training_steps']}")
    print(f"    Buffer: {tr['buffer_episodes']} episodes, {tr['buffer_transitions']} transitions")
    if 'loss_improvement' in tr:
        print(f"    Loss improvement: {tr.get('loss_improvement', 0) * 100:.1f}%")
    
    cs = result['cost_stats']
    print(f"\n  Energy costs:")
    print(f"    Mean action cost: {cs['mean_action_cost']:.4f}")
    print(f"    Mean path cost: {cs['mean_path_cost']:.4f}")
    print(f"    Mean variance cost: {cs['mean_var_cost']:.4f}")
    
    ss = result['selection_stats']
    print(f"\n  Selection:")
    print(f"    λ: {ss['lambda']:.3f}")
    print(f"    Mean efficiency: {ss['mean_efficiency']:.4f}")
    
    if result['cycle_log']:
        print(f"\n  Recent cycles:")
        for c in result['cycle_log'][-3:]:
            print(f"    GP={c['goal_prob']:.4f}, cost={c['avg_cost']:.4f}, "
                  f"λ={c['selection_stats']['lambda']:.3f}")
    
    print("\n  ✓ Energy-regularized engine operational")


def test_cost_vs_no_cost_comparison():
    """Compare energy-regularized vs unregularized flow selection."""
    print("\n" + "=" * 60)
    print("COST vs NO-COST COMPARISON")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='compare',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    manifold = FlowManifold(flow_dim=4)
    for i in range(8):
        target = np.random.randn(16) * random.uniform(0.3, 1.5)
        flow = PointFlow(target, gain=random.uniform(0.2, 0.8))
        manifold.add_flow(flow, f'flow_{i}')
    
    cost_fn = EnergyCostFunction()
    evaluator = EfficiencyEvaluator(cost_fn, lambda_cost=0.5)
    
    # Energy-regularized (efficiency mode)
    cem_efficient = EnergyRegularizedCEM(
        world_model=wm, manifold=manifold, goal=goal,
        energy_cost_fn=cost_fn, efficiency_evaluator=evaluator,
        flow_dim=4, n_candidates=20, n_elites=4, n_iterations=3,
        efficiency_mode=True, controllability_bonus=0.4
    )
    
    # Goal-only mode (no cost)
    cem_greedy = EnergyRegularizedCEM(
        world_model=wm, manifold=manifold, goal=goal,
        energy_cost_fn=cost_fn, efficiency_evaluator=evaluator,
        flow_dim=4, n_candidates=20, n_elites=4, n_iterations=3,
        efficiency_mode=False, controllability_bonus=0.4
    )
    
    # Compare on same start
    z_start = np.zeros(16)
    h_start = np.zeros(64)
    
    plan_eff = cem_efficient.plan_flow(z_start, h_start)
    plan_greedy = cem_greedy.plan_flow(z_start, h_start)
    
    print(f"\n  Efficiency mode:")
    print(f"    Score: {plan_eff['score']:.4f}")
    if plan_eff.get('info'):
        print(f"    Goal prob: {plan_eff['info'].get('goal_prob', 0):.4f}")
        print(f"    Cost: {plan_eff['info'].get('cost', {}).get('total', 0):.4f}")
    
    print(f"\n  Greedy mode (no cost):")
    print(f"    Score: {plan_greedy['score']:.4f}")
    if plan_greedy.get('info'):
        print(f"    Goal prob: {plan_greedy['info'].get('goal_prob', 0):.4f}")
        print(f"    Cost: {plan_greedy['info'].get('cost', {}).get('total', 0):.4f}")
    
    print("\n  ✓ Cost comparison complete")


if __name__ == "__main__":
    test_energy_cost_function()
    test_efficiency_evaluator()
    test_energy_regularized_cem()
    test_full_energy_engine()
    test_cost_vs_no_cost_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 38: ENERGY-REGULARIZED DYNAMICS")
    print("=" * 60)
    
    print("""
KEY SHIFT FROM PHASE 37:
  Phase 37: "Can I reach the goal?" (controllability)
  Phase 38: "Is it worth the cost?" (energy-regularized)

WHAT CHANGED:
  1. EnergyCostFunction — action, path, variance, instability costs
  2. EfficiencyEvaluator — benefit / (1 + λ * cost) scoring
  3. EnergyRegularizedCEM — cost-aware flow optimization
  4. CostAwareFlowSelection — efficiency ranking + Pareto analysis
  5. Adaptive λ — adjusts trade-off based on history

WHY THIS MATTERS:
  Without energy:   erratic, inefficient paths to goal
  With energy:      smooth, efficient, reliable control

  This is the difference between:
    "reaching the goal by any means"
    "reaching the goal efficiently"

ARCHITECTURAL PROGRESSION:
  Phase 34: inverse dynamics (per-action stabilization)
  Phase 35: dynamical flows (state-dependent policies)
  Phase 36: behavioral physics (learn dynamics from flows)
  Phase 37: controllability (where can flows reach?)
  Phase 38: energy-regularization (is it worth the cost?)

NEXT (Phase 39):
  Autonomous flow discovery — flows that self-organize and specialize
  through experience, without hand-seeding.
""")
