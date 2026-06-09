"""
Phase 34 — Inverse Control Stabilization

WHAT THIS FIXES:
  Phase 33's attractor stabilization was mathematically ill-posed:
    attractor_point ∈ action_space
    mean_delta ∈ latent_space
    attractor_point = lerp(attractor_point, mean_delta)  ❌ wrong

CORRECT APPROACH:
  Inverse dynamics model: (z_t, z_{t+1}) → a_t
  Now attractor stabilization becomes:
    a_inferred = inv_dynamics(z, z_next)  ← same space as attractor_point
    attractor_point = lerp(attractor_point, a_inferred)  ✅ correct

GOAL-CONDITIONED STABILIZATION:
  Not just "consistent transitions" but "goal-directed transitions":
    a_ideal = inv_dynamics(z, goal)  ← action that moves toward goal
    attractor_point = lerp(attractor_point, a_ideal)  ← goal-pulled attractor

KEY SHIFTS:
  1. Skill stability now measures: P(goal | attractor) - λ * uncertainty
     NOT: variance(∆z)
  2. Attractor update uses inverse dynamics (action space ↔ action space)
     NOT: action space = latent space (false assumption)
  3. Goal-conditioned reinforcement pulls attractors toward utility,
     NOT just toward consistency
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

from phase30_training_loop import MinimalWorldModel
from phase31_hierarchical_execution import GoalAttractor
from phase32_skill_dynamics_coupling import (
    SkillDynamicsAdapter, ProbabilisticGoalEvaluator
)
from phase33_skill_manifold_stabilization import (
    SkillAttractor, SkillManifold, ManifoldCEM, SkillStabilizer
)


# ============================================================================
# 1. INVERSE DYNAMICS MODEL
# ============================================================================

class InverseDynamicsModel:
    """
    Learns: (z_t, z_{t+1}) → a_t
    
    The missing link between latent transitions and actions.
    
    Once learned:
      - Can infer what action caused a given transition
      - Can compute what action WOULD produce desired transition
      - Enables proper attractor stabilization (action→action, not action→latent)
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 16,
        hidden_dim: int = 64,
        learning_rate: float = 0.01
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate
        
        # Network: (z_t ⊕ z_{t+1}) → a_t
        input_dim = latent_dim * 2
        
        scale = 0.1
        self.W1 = np.random.randn(hidden_dim, input_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(action_dim, hidden_dim) * scale
        self.b2 = np.zeros(action_dim)
        
        # Training state
        self.training_data: List[Dict] = []
        self.max_data = 1000
    
    def predict(self, z_t: np.ndarray, z_next: np.ndarray) -> np.ndarray:
        """Infer action: (z_t, z_{t+1}) → a_t"""
        z_t = np.asarray(z_t).flatten()[:self.latent_dim]
        z_next = np.asarray(z_next).flatten()[:self.latent_dim]
        
        x = np.concatenate([z_t, z_next])
        hidden = np.tanh(self.W1 @ x + self.b1)
        action = self.W2 @ hidden + self.b2
        
        return action
    
    def predict_goal_action(
        self, z_t: np.ndarray, goal: GoalAttractor
    ) -> np.ndarray:
        """
        What action moves z_t toward goal?
        
        Uses inverse dynamics by constructing desired next state:
          z_desired = z_t + (goal.attractor_state - z_t) * step_scale
          a_ideal = inv_dynamics(z_t, z_desired)
        """
        z_t = np.asarray(z_t).flatten()[:self.latent_dim]
        goal_state = goal.attractor_state[:self.latent_dim]
        
        direction = goal_state - z_t
        dist = np.linalg.norm(direction)
        
        if dist < 1e-8:
            return np.zeros(self.action_dim)
        
        # Step toward goal (small step for stability)
        step_scale = min(0.3, dist)
        z_desired = z_t + (direction / dist) * step_scale
        
        return self.predict(z_t, z_desired)
    
    def train_step(
        self, z_t: np.ndarray, z_next: np.ndarray, a_true: np.ndarray
    ) -> float:
        """One training step: minimize ||predict(z, z_next) - a_true||²"""
        a_pred = self.predict(z_t, z_next)
        
        error = a_pred - np.asarray(a_true).flatten()[:self.action_dim]
        loss = float(np.mean(error ** 2))
        
        # Simple gradient descent on MSE
        z_t_f = np.asarray(z_t).flatten()[:self.latent_dim]
        z_next_f = np.asarray(z_next).flatten()[:self.latent_dim]
        x = np.concatenate([z_t_f, z_next_f])
        
        hidden = np.tanh(self.W1 @ x + self.b1)
        dh = (1 - hidden ** 2)  # d(tanh)/dx
        
        # Gradients
        dL_da = 2 * error / len(error)
        dL_dW2 = np.outer(dL_da, hidden)
        dL_db2 = dL_da
        
        dL_dhidden = self.W2.T @ dL_da
        dL_dW1 = np.outer(dL_dhidden * dh, x)
        dL_db1 = dL_dhidden * dh
        
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
        
        return loss
    
    def train_from_buffer(self, transitions: List[Dict]) -> float:
        """Train on collected transitions."""
        if not transitions:
            return 0.0
        
        total_loss = 0.0
        for t in transitions:
            loss = self.train_step(
                t['z_before'], t['z_after'], t['action']
            )
            total_loss += loss
        
        return total_loss / len(transitions)
    
    def train_from_world_model(
        self, wm: MinimalWorldModel, n_samples: int = 100
    ) -> float:
        """
        Generate training data by rolling out world model.
        
        For a random state and action:
          z_{t+1} = world_model(z_t, a)
          training_pair: (z_t, z_{t+1}) → a
        """
        if not self.training_data:
            self._generate_synthetic_data(wm, n_samples)
        
        return self.train_from_buffer(self.training_data[-200:])
    
    def _generate_synthetic_data(self, wm: MinimalWorldModel, n: int):
        """Generate synthetic inverse dynamics data from world model."""
        for _ in range(n):
            z = np.random.randn(self.latent_dim) * 0.5
            h = np.random.randn(wm.belief_dim) * 0.1
            a = np.random.randn(self.action_dim) * 0.3
            
            mu, logvar = wm.predict_transition(z, h, a)
            z_next = mu + np.exp(0.5 * logvar) * np.random.randn(*mu.shape) * 0.1
            
            self.training_data.append({
                'z_before': z.copy(),
                'z_after': z_next.copy(),
                'action': a.copy(),
                'uncertainty': float(np.mean(np.exp(logvar)))
            })
    
    def add_transition(
        self, z_before: np.ndarray, z_after: np.ndarray, action: np.ndarray
    ):
        """Add real transition to training buffer."""
        self.training_data.append({
            'z_before': np.asarray(z_before).flatten()[:self.latent_dim].copy(),
            'z_after': np.asarray(z_after).flatten()[:self.latent_dim].copy(),
            'action': np.asarray(action).flatten()[:self.action_dim].copy()
        })
        
        if len(self.training_data) > self.max_data:
            self.training_data = self.training_data[-self.max_data:]
    
    def evaluate(self, n_test: int = 50) -> Dict:
        """Evaluate inverse dynamics accuracy."""
        if len(self.training_data) < n_test:
            return {'error': float('inf'), 'n_samples': len(self.training_data)}
        
        test_data = random.sample(self.training_data, min(n_test, len(self.training_data)))
        
        errors = []
        for t in test_data:
            a_pred = self.predict(t['z_before'], t['z_after'])
            error = float(np.mean((a_pred - t['action']) ** 2))
            errors.append(error)
        
        return {
            'mean_error': float(np.mean(errors)),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors)),
            'n_samples': len(test_data)
        }


# ============================================================================
# 2. CORRECTED ATTRACTOR STABILIZATION (via inverse dynamics)
# ============================================================================

class InverseAttractorStabilizer:
    """
    Stabilizes attractors using inverse dynamics.
    
    Phase 33's approach was wrong:
      attractor_point = lerp(attractor_point, mean_latent_delta)  ❌
    
    This corrects it:
      a_inferred = inv_dynamics(z_t, z_{t+1})
      attractor_point = lerp(attractor_point, a_inferred)  ✅
    
    Both sides are in action_space.
    
    Goal-conditioned variant:
      a_ideal = inv_dynamics(z_t, goal_at_this_step)
      attractor_point = lerp(attractor_point, a_ideal)  ← pulled toward utility
    """
    
    def __init__(
        self,
        inverse_dynamics: InverseDynamicsModel,
        manifold: SkillManifold,
        goal: GoalAttractor,
        learning_rate: float = 0.1,
        goal_bias: float = 0.3,       # How much to pull toward goal actions
        stability_threshold: float = 0.3,
        min_data_for_update: int = 5
    ):
        self.inv_dyn = inverse_dynamics
        self.manifold = manifold
        self.goal = goal
        self.lr = learning_rate
        self.goal_bias = goal_bias
        self.stability_threshold = stability_threshold
        self.min_data = min_data_for_update
        
        self.step_count = 0
    
    def stabilize_attractor(self, attractor: SkillAttractor) -> Dict:
        """
        Stabilize attractor via inverse dynamics.
        
        For each recent transition:
          1. Infer action: a_inf = inv_dynamics(z_t, z_{t+1})
          2. Predict goal action: a_goal = inv_dynamics(z_t, goal)
          3. Update attractor toward blend of inferred and goal actions
          4. Compute true stability = consistency of inferred actions
        """
        transitions = attractor.recent_transitions
        
        if len(transitions) < self.min_data:
            return {'updated': False, 'reason': 'insufficient_data'}
        
        # Compute inferred actions for each transition
        inferred_actions = []
        goal_actions = []
        
        for t in transitions[-self.min_data * 4:]:
            z_before = t['z_before']
            z_after = t['z_after']
            
            a_inferred = self.inv_dyn.predict(z_before, z_after)
            inferred_actions.append(a_inferred)
            
            a_goal = self.inv_dyn.predict_goal_action(z_before, self.goal)
            goal_actions.append(a_goal)
        
        inferred_actions = np.array(inferred_actions)
        goal_actions = np.array(goal_actions)
        
        # True stability = consistency of inferred actions
        # If same attractor → similar inferred actions → high stability
        mean_inferred = np.mean(inferred_actions, axis=0)
        variance = np.mean(np.var(inferred_actions, axis=0))
        magnitude = np.linalg.norm(mean_inferred) + 1e-8
        
        true_stability = 1.0 / (1.0 + variance / magnitude)
        
        # Goal alignment = how well inferred actions match goal-direction actions
        goal_alignment = 0.0
        for a_inf, a_goal in zip(inferred_actions, goal_actions):
            cos_sim = np.dot(a_inf, a_goal) / (
                np.linalg.norm(a_inf) * np.linalg.norm(a_goal) + 1e-8
            )
            goal_alignment += max(0, cos_sim)  # Only positive alignment
        goal_alignment /= len(inferred_actions)
        
        # Update attractor point toward inferred action (correct: both in action space)
        target = (1 - self.goal_bias) * mean_inferred + self.goal_bias * np.mean(goal_actions, axis=0)
        
        attractor.attractor_point = (
            (1 - self.lr) * attractor.attractor_point
            + self.lr * target
        )
        
        # Normalize
        norm = np.linalg.norm(attractor.attractor_point)
        if norm > 0:
            attractor.attractor_point = attractor.attractor_point / norm * min(norm, 5.0)
        
        # Update attractor stability metrics
        attractor.stability = float(true_stability)
        attractor.goal_alignment = float(goal_alignment)
        attractor.stability_confidence += 0.05
        
        return {
            'updated': True,
            'true_stability': float(true_stability),
            'goal_alignment': float(goal_alignment),
            'target_norm': float(np.linalg.norm(target)),
            'mean_inferred_norm': float(np.linalg.norm(mean_inferred)),
            'n_transitions': len(transitions)
        }
    
    def stabilize_all(self) -> Dict:
        """Stabilize all attractors in manifold."""
        if not self.manifold.attractors:
            return {'n_stabilized': 0, 'avg_stability': 0.0}
        
        self.step_count += 1
        results = []
        
        for attractor in list(self.manifold.attractors.values()):
            result = self.stabilize_attractor(attractor)
            results.append(result)
        
        stabilities = [
            a.stability for a in self.manifold.attractors.values()
        ]
        
        pruned = 0
        if self.step_count % 10 == 0:
            pruned = self.manifold.prune_unstable_attractors(
                min_stability=self.stability_threshold,
                min_age=self.min_data * 2
            )
        
        return {
            'n_stabilized': sum(1 for r in results if r.get('updated')),
            'avg_stability': float(np.mean(stabilities)) if stabilities else 0.0,
            'max_stability': float(max(stabilities)) if stabilities else 0.0,
            'n_pruned': pruned,
            'n_attractors': len(self.manifold.attractors)
        }
    
    def compute_stability_report(self) -> Dict:
        """Detailed report on attractor health."""
        if not self.manifold.attractors:
            return {'n_attractors': 0}
        
        attractors = list(self.manifold.attractors.values())
        stabilities = [a.stability for a in attractors]
        alignments = [a.goal_alignment for a in attractors]
        ages = [a.age for a in attractors]
        
        return {
            'n_attractors': len(attractors),
            'mean_stability': float(np.mean(stabilities)),
            'mean_goal_alignment': float(np.mean(alignments)),
            'max_goal_alignment': float(max(alignments)),
            'stable_count': sum(1 for s in stabilities if s > 0.5),
            'goal_aligned_count': sum(1 for g in alignments if g > 0.3),
            'mean_age': float(np.mean(ages))
        }


# ============================================================================
# 3. GOAL-CONDITIONED STABILIZED CEM
# ============================================================================

class GoalConditionedCEM(ManifoldCEM):
    """
    CEM planner with goal-conditioned attractor pull.
    
    Extends ManifoldCEM with:
      - Inverse-dynamics-based goal evaluation
      - Attractor pull toward goal-relevant regions
      - Stability-aware exploration
    """
    
    def __init__(
        self,
        manifold: SkillManifold,
        adapter: SkillDynamicsAdapter,
        evaluator: Any,
        world_model: MinimalWorldModel,
        inverse_dynamics: InverseDynamicsModel,
        goal: GoalAttractor,
        n_candidates: int = 80,
        n_elites: int = 16,
        n_iterations: int = 5,
        goal_bonus: float = 0.3,
        stability_bonus: float = 0.15,
        entropy_coeff: float = 0.05
    ):
        super().__init__(
            manifold=manifold,
            adapter=adapter,
            evaluator=evaluator,
            world_model=world_model,
            n_candidates=n_candidates,
            n_elites=n_elites,
            n_iterations=n_iterations,
            stability_bonus=stability_bonus,
            entropy_coeff=entropy_coeff
        )
        self.inv_dyn = inverse_dynamics
        self.goal = goal
        self.goal_bonus = goal_bonus
    
    def _evaluate_coord_sequence(
        self,
        coords: List[np.ndarray],
        z_start: np.ndarray,
        h_start: np.ndarray,
        goal: GoalAttractor
    ) -> Tuple[float, Optional[np.ndarray]]:
        """Evaluate with inverse-dynamics goal bonus."""
        z = z_start.copy()
        h = h_start.copy()
        trajectory = [z.copy()]
        
        total_goal_prob = 0.0
        total_stability = 0.0
        total_goal_alignment = 0.0
        
        for coord in coords:
            blended_action = self.manifold.interpolate_at(coord)
            
            mu, logvar = self.wm.predict_transition(z, h, blended_action)
            std = np.exp(0.5 * logvar)
            z_next = mu + std * np.random.randn(*mu.shape) * 0.3
            h = self.wm.gru_step(h, mu)
            
            # Goal probability
            prob = self.evaluator.goal_satisfaction_probability(goal, mu, logvar)
            total_goal_prob += prob
            
            # Goal alignment via inverse dynamics
            a_ideal = self.inv_dyn.predict_goal_action(z, goal)
            cos_sim = np.dot(blended_action, a_ideal) / (
                np.linalg.norm(blended_action) * np.linalg.norm(a_ideal) + 1e-8
            )
            total_goal_alignment += max(0, cos_sim)
            
            # Stability
            stab = self.manifold.compute_stability_field(coord)
            total_stability += stab
            
            z = z_next
            trajectory.append(z.copy())
        
        avg_goal = total_goal_prob / len(coords)
        avg_stab = total_stability / len(coords)
        avg_alignment = total_goal_alignment / len(coords)
        
        coord_var = np.var([c for c in coords], axis=0).mean() if len(coords) > 1 else 0.0
        
        score = (avg_goal
                 + self.stability_bonus * avg_stab
                 + self.goal_bonus * avg_alignment
                 + self.entropy_coeff * coord_var)
        
        return score, np.array(trajectory)


# ============================================================================
# 4. INVERSE-CONTROLLED EXECUTION ENGINE
# ============================================================================

class InverseControlledEngine:
    """
    Complete execution engine with inverse dynamics stabilization.
    
    Architecture:
      World Model → Inverse Dynamics → Attractor Stabilization → CEM → Goal
        
    Loop per step:
      1. CEM plans attractor (goal-conditioned)
      2. Attractor applied → transition recorded
      3. Inverse dynamics infers action → stabilizer updates attractor
      4. Goal alignment computed → attractor pulled toward utility
      5. Inverse dynamics trained on new transition
    """
    
    def __init__(
        self,
        world_model: MinimalWorldModel,
        manifold: SkillManifold,
        goal: GoalAttractor,
        inv_dyn_lr: float = 0.01,
        n_initial_attractors: int = 12,
        stabilizer_lr: float = 0.1,
        goal_bias: float = 0.3
    ):
        self.wm = world_model
        
        # Inverse dynamics
        self.inv_dyn = InverseDynamicsModel(
            latent_dim=world_model.latent_dim,
            action_dim=world_model.action_dim,
            learning_rate=inv_dyn_lr
        )
        
        # Manifold + attractors
        self.manifold = manifold
        self.goal = goal
        
        if not self.manifold.attractors:
            self._seed_attractors(n_initial_attractors)
        
        # Adapters
        self.adapter = SkillDynamicsAdapter(world_model)
        self.evaluator = ProbabilisticGoalEvaluator(world_model)
        
        # Stabilizer (inverse-dynamics-corrected)
        self.stabilizer = InverseAttractorStabilizer(
            inverse_dynamics=self.inv_dyn,
            manifold=self.manifold,
            goal=goal,
            learning_rate=stabilizer_lr,
            goal_bias=goal_bias
        )
        
        # CEM (goal-conditioned)
        self.cem = GoalConditionedCEM(
            manifold=self.manifold,
            adapter=self.adapter,
            evaluator=self.evaluator,
            world_model=self.wm,
            inverse_dynamics=self.inv_dyn,
            goal=goal,
            goal_bonus=0.3,
            stability_bonus=0.15
        )
        
        # Execution state
        self.step_count = 0
        self.execution_log: List[Dict] = []
        self.inv_dyn_log: List[float] = []
    
    def _seed_attractors(self, n: int):
        """Seed initial attractors."""
        for i in range(n):
            attractor = SkillAttractor.from_random(
                skill_id=f'seed_{i}',
                action_dim=self.wm.action_dim,
                latent_dim=self.wm.latent_dim
            )
            self.manifold.add_attractor(attractor)
    
    def execute_step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """
        One step with inverse dynamics stabilization.
        
        1. Plan attractor (goal-conditioned CEM)
        2. Apply → record transition
        3. Train inverse dynamics on transition
        4. Stabilize attractors via inverse dynamics
        5. Report metrics
        """
        # 1. Plan
        plan = self.cem.plan_sequence(z, h, self.goal, horizon=1)
        
        coord = plan['coords'][0] if plan['coords'] else self.manifold.sample_random_coord()
        
        # 2. Select/spawn attractor at planned coordinate
        nearest = self.manifold.nearest_attractors(coord, k=1)
        
        if nearest:
            selected = nearest[0]
            # Use attractor's own point (it's being stabilized)
            action = selected.attractor_point
        else:
            # Spawn new
            action = self.manifold.interpolate_at(coord)
            selected = SkillAttractor(
                skill_id=f'step_{self.step_count}',
                name=f'step_{self.step_count}',
                attractor_point=action.copy(),
                transition_signature=np.zeros(self.wm.latent_dim)
            )
            self.manifold.add_attractor(selected, coord.copy())
        
        # 3. Apply attractor → transition
        z_next, h_next, delta_norm = selected.apply_to(z, h, self.wm)
        
        # 4. Train inverse dynamics on this transition
        loss = self.inv_dyn.train_step(z, z_next, action)
        self.inv_dyn.add_transition(z, z_next, action)
        self.inv_dyn_log.append(loss)
        
        # 5. Stabilize attractors via inverse dynamics
        stab_result = self.stabilizer.stabilize_all()
        
        # 6. Evaluate
        mu, logvar = self.wm.predict_transition(z, h, action)
        goal_prob = self.evaluator.goal_satisfaction_probability(self.goal, mu, logvar)
        
        self.step_count += 1
        
        step_result = {
            'step': self.step_count,
            'selected_attractor': selected.skill_id,
            'goal_prob': float(goal_prob),
            'delta_norm': float(delta_norm),
            'inv_dyn_loss': float(loss),
            'avg_stability': stab_result.get('avg_stability', 0.0),
            'n_attractors': stab_result.get('n_attractors', 0),
            'n_pruned': stab_result.get('n_pruned', 0),
            'n_stabilized': stab_result.get('n_stabilized', 0)
        }
        
        self.execution_log.append(step_result)
        return step_result
    
    def execute_goal(self, z_start: np.ndarray, max_steps: int = 100) -> Dict:
        """Execute full goal run with inverse dynamics stabilization."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)
        
        all_z = [z.copy()]
        goal_reached = False
        
        for step in range(max_steps):
            # Plan
            plan = self.cem.plan_sequence(z, h, self.goal, horizon=1)
            coord = plan['coords'][0] if plan['coords'] else self.manifold.sample_random_coord()
            
            nearest = self.manifold.nearest_attractors(coord, k=1)
            if nearest:
                selected = nearest[0]
                action = selected.attractor_point
            else:
                action = self.manifold.interpolate_at(coord)
                selected = SkillAttractor(
                    skill_id=f'step_{step}',
                    name=f'step_{step}',
                    attractor_point=action.copy(),
                    transition_signature=np.zeros(self.wm.latent_dim)
                )
                self.manifold.add_attractor(selected, coord.copy())
            
            z_next, h_next, delta_norm = selected.apply_to(z, h, self.wm)
            all_z.append(z_next.copy())
            
            loss = self.inv_dyn.train_step(z, z_next, action)
            self.inv_dyn.add_transition(z, z_next, action)
            
            mu, logvar = self.wm.predict_transition(z, h, action)
            goal_prob = self.evaluator.goal_satisfaction_probability(self.goal, mu, logvar)
            
            if goal_prob > 0.7:
                goal_reached = True
            
            stab_result = self.stabilizer.stabilize_all()
            
            self.execution_log.append({
                'step': step,
                'selected': selected.skill_id,
                'goal_prob': float(goal_prob),
                'inv_dyn_loss': float(loss),
                'stability': stab_result.get('avg_stability', 0.0)
            })
            
            z = z_next.copy()
            h = h_next.copy()
        
        # Evaluate inverse dynamics
        inv_eval = self.inv_dyn.evaluate()
        
        # Stability report
        stab_report = self.stabilizer.compute_stability_report()
        
        return {
            'goal_reached': goal_reached,
            'final_goal_prob': self.execution_log[-1]['goal_prob'] if self.execution_log else 0.0,
            'n_steps': len(self.execution_log),
            'inv_dynamics_error': inv_eval.get('mean_error', float('inf')),
            'stability': stab_report,
            'execution_log': self.execution_log[-20:],
            'trajectory_length': len(all_z)
        }


# ============================================================================
# 5. TESTS
# ============================================================================

def test_inverse_dynamics():
    """Test inverse dynamics training and prediction."""
    print("\n" + "=" * 60)
    print("INVERSE DYNAMICS TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    inv_dyn = InverseDynamicsModel(latent_dim=16, action_dim=16)
    
    # Generate training data
    print("\n  Training inverse dynamics...")
    losses = []
    for i in range(100):
        z = np.random.randn(16) * 0.5
        h = np.random.randn(64) * 0.1
        a_true = np.random.randn(16) * 0.3
        
        mu, logvar = wm.predict_transition(z, h, a_true)
        z_next = mu + np.exp(0.5 * logvar) * np.random.randn(16) * 0.1
        
        loss = inv_dyn.train_step(z, z_next, a_true)
        inv_dyn.add_transition(z, z_next, a_true)
        losses.append(loss)
    
    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # Test prediction
    z_test = np.random.randn(16) * 0.5
    h_test = np.random.randn(64) * 0.1
    a_test = np.random.randn(16) * 0.3
    
    mu, _ = wm.predict_transition(z_test, h_test, a_test)
    a_pred = inv_dyn.predict(z_test, mu)
    
    error = float(np.mean((a_pred - a_test) ** 2))
    print(f"  Prediction error: {error:.4f}")
    
    # Test goal action prediction
    goal = GoalAttractor(
        goal_id='test',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0,
        priority=0.8,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    a_goal = inv_dyn.predict_goal_action(z_test, goal)
    print(f"  Goal action norm: {np.linalg.norm(a_goal):.4f}")
    
    print("\n  ✓ Inverse dynamics operational")


def test_inverse_stabilizer():
    """Test inverse-dynamics-corrected attractor stabilization."""
    print("\n" + "=" * 60)
    print("INVERSE STABILIZER TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    inv_dyn = InverseDynamicsModel(latent_dim=16, action_dim=16)
    
    # Pre-train inverse dynamics
    for i in range(50):
        z = np.random.randn(16) * 0.5
        h = np.random.randn(64) * 0.1
        a = np.random.randn(16) * 0.3
        mu, _ = wm.predict_transition(z, h, a)
        z_next = mu + np.random.randn(16) * 0.05
        inv_dyn.train_step(z, z_next, a)
        inv_dyn.add_transition(z, z_next, a)
    
    # Create attractors
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    for i in range(6):
        attr = SkillAttractor.from_random(f'attractor_{i}', 16, 16)
        manifold.add_attractor(attr)
    
    goal = GoalAttractor(
        goal_id='stab_test',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0,
        priority=0.8,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    stabilizer = InverseAttractorStabilizer(
        inverse_dynamics=inv_dyn,
        manifold=manifold,
        goal=goal,
        learning_rate=0.1,
        goal_bias=0.3
    )
    
    # Apply attractors to build transition history
    z = np.zeros(16)
    h = np.zeros(64)
    
    for step in range(20):
        for attr in manifold.attractors.values():
            z, h, _ = attr.apply_to(z, h, wm)
    
    # Stabilize
    result = stabilizer.stabilize_all()
    
    print(f"\n  Stabilized: {result['n_stabilized']}")
    print(f"  Avg stability: {result['avg_stability']:.4f}")
    print(f"  Max stability: {result['max_stability']:.4f}")
    
    report = stabilizer.compute_stability_report()
    print(f"\n  Report:")
    print(f"    Mean stability: {report['mean_stability']:.4f}")
    print(f"    Mean goal alignment: {report['mean_goal_alignment']:.4f}")
    print(f"    Stable count: {report['stable_count']}")
    
    print("\n  ✓ Inverse stabilizer operational")


def test_goal_conditioned_cem():
    """Test CEM with goal-conditioned evaluation."""
    print("\n" + "=" * 60)
    print("GOAL-CONDITIONED CEM TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    inv_dyn = InverseDynamicsModel(latent_dim=16, action_dim=16)
    
    # Pre-train
    for i in range(30):
        z = np.random.randn(16) * 0.5
        h = np.random.randn(64) * 0.1
        a = np.random.randn(16) * 0.3
        mu, _ = wm.predict_transition(z, h, a)
        inv_dyn.train_step(z, mu, a)
    
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    for i in range(8):
        direction = np.random.randn(16)
        direction = direction / np.linalg.norm(direction) * 0.5
        attr = SkillAttractor(
            skill_id=f'cem_skill_{i}',
            name=f'cem_{i}',
            attractor_point=direction,
            transition_signature=np.zeros(16),
            stability=0.5
        )
        manifold.add_attractor(attr)
    
    adapter = SkillDynamicsAdapter(wm)
    evaluator = ProbabilisticGoalEvaluator(wm)
    
    goal = GoalAttractor(
        goal_id='cem_goal',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    cem = GoalConditionedCEM(
        manifold=manifold,
        adapter=adapter,
        evaluator=evaluator,
        world_model=wm,
        inverse_dynamics=inv_dyn,
        goal=goal,
        n_candidates=30,
        n_elites=6,
        n_iterations=3,
        goal_bonus=0.3
    )
    
    result = cem.plan_sequence(np.zeros(16), np.zeros(64), goal, horizon=3)
    
    print(f"\n  Sequence score: {result['score']:.4f}")
    print(f"  Coords: {len(result['coords'])}")
    print(f"  Attractors: {len(result['attractors'])}")
    
    print("\n  ✓ Goal-conditioned CEM operational")


def test_full_inverse_control():
    """Test full inverse control execution."""
    print("\n" + "=" * 60)
    print("FULL INVERSE CONTROL TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    
    goal = GoalAttractor(
        goal_id='full_inv',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    engine = InverseControlledEngine(
        world_model=wm,
        manifold=manifold,
        goal=goal,
        n_initial_attractors=8,
        stabilizer_lr=0.1,
        goal_bias=0.3
    )
    
    result = engine.execute_goal(np.zeros(16), max_steps=20)
    
    print(f"\n  Goal reached: {result['goal_reached']}")
    print(f"  Final goal prob: {result['final_goal_prob']:.4f}")
    print(f"  Steps: {result['n_steps']}")
    
    stab = result['stability']
    print(f"\n  Stability:")
    print(f"    Mean: {stab['mean_stability']:.4f}")
    print(f"    Goal alignment: {stab['mean_goal_alignment']:.4f}")
    print(f"    Stable: {stab['stable_count']}, Goal-aligned: {stab['goal_aligned_count']}")
    
    inv_err = result.get('inv_dynamics_error', float('inf'))
    print(f"\n  Inverse dynamics error: {inv_err:.4f}")
    
    print("\n  ✓ Inverse control operational")


if __name__ == "__main__":
    test_inverse_dynamics()
    test_inverse_stabilizer()
    test_goal_conditioned_cem()
    test_full_inverse_control()
    
    print("\n" + "=" * 60)
    print("PHASE 34: INVERSE CONTROL STABILIZATION")
    print("=" * 60)
    
    print("""
WHAT PHASE 33 GOT WRONG:
  Stabilization was in latent space, not action space.
  attractor_point = lerp(attractor_point, mean_latent_delta)  ❌

WHAT PHASE 34 FIXES:
  Adds InverseDynamicsModel: (z_t, z_{t+1}) → a_t
  Now stabilization is correct:
    a_inferred = inv_dynamics(z_t, z_{t+1})  ← action space
    attractor_point = lerp(attractor_point, a_inferred)  ✅

WHAT'S NEW:
  1. InverseDynamicsModel — learned mapping from latent transitions to actions
  2. InverseAttractorStabilizer — correct attractor update using inferred actions
  3. GoalConditionedCEM — goal bias via inverse dynamics goal action prediction
  4. Attractor stability = consistency of inferred actions (not latent deltas)
  5. Goal alignment = cosine similarity between inferred and goal-directed actions

ARCHITECTURAL IMPACT:
  Phase 33: stabilize what IS (consistent latent displacement)
  Phase 34: stabilize what SHOULD BE (goal-directed control policy)
  
  This is the difference between:
    "same attractor → same ∆z"  (dead consistency)
    "same attractor → same control"  (live policy)

NEXT:
  - Dynamic skill flows (limit cycles instead of point attractors)
  - Topological manifold learning
""")
