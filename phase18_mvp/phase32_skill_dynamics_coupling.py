"""
Phase 32 — Skill-Dynamics Coupling Layer (Minimal Bridge)

WHAT THIS DOES:
  Turns skill from "trajectory generator" into "conditioned transition operator" 
  over the learned world model dynamics.

  P(z_{t+1} | z_t, h_t, skill) via world model

KEY SHIFT:
  BEFORE: skill.generate_trajectory(z)  # deterministic, unrelated to world model
  AFTER:  skill_rollout(z, h, skill, world_model) → P(trajectory | dynamics)
          skill = operator that shapes future state distributions

MINIMAL SCOPE:
  1. Skill embedding → aligned to world model action space
  2. Skill-conditioned transition: predict_transition(z, h, skill_embedding)
  3. Probabilistic goal evaluation: P(goal | skill_sequence, world_model)
  4. Uncertainty-aware CEM evaluation

NO changes to Phase 31 structure.
NO changes to Phase 30 world model.
PURE wrapper layer.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

# Import Phase 30 world model
from phase30_training_loop import MinimalWorldModel

# Import Phase 31 constructs
from phase31_hierarchical_execution import (
    SkillPrototype, GoalAttractor, IntentVector, IntentConditionedCEM,
    SkillBank, HierarchicalExecutionEngine
)


# ============================================================================
# 1. SKILL EMBEDDING → WORLD MODEL ALIGNMENT
# ============================================================================

class SkillDynamicsAdapter:
    """
    Bridges skill representations to world model latent dynamics.
    
    Key function:
      skill_embedding → conditioning vector for predict_transition()
      P(z_{t+1} | z_t, h_t, skill) via world model
    
    The skill embedding becomes the "action" in the world model transition.
    This means the world model already knows how to process it.
    """
    
    def __init__(self, world_model: MinimalWorldModel, action_dim: int = 16):
        self.wm = world_model
        self.action_dim = action_dim
        
        # Learned projection from trajectory → skill embedding
        # Mean trajectory shape: (T, latent_dim) → action_dim vector
        self.trajectory_encoder = TrajectoryEncoder(
            latent_dim=world_model.latent_dim,
            action_dim=action_dim
        )
    
    def compute_skill_embedding(self, skill: SkillPrototype) -> np.ndarray:
        """
        Project skill's trajectory distribution into world model action space.
        
        The embedding becomes the "action" that conditions transitions.
        """
        if skill.skill_embedding is not None and len(skill.skill_embedding) == self.action_dim:
            return skill.skill_embedding
        
        # Learn embedding from trajectory statistics
        embedding = self.trajectory_encoder.encode(
            skill.mean_trajectory,
            skill.trajectory_variance
        )
        
        skill.skill_embedding = embedding
        return embedding
    
    def skill_conditioned_transition(
        self,
        z: np.ndarray,
        h: np.ndarray,
        skill: SkillPrototype
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        P(z_{t+1} | z_t, h_t, skill) via world model.
        
        The skill embedding acts as the conditioning action.
        Returns: (mu_next, logvar_next) from world model
        """
        embedding = self.compute_skill_embedding(skill)
        return self.wm.predict_transition(z, h, embedding)
    
    def skill_rollout(
        self,
        z_start: np.ndarray,
        h_start: np.ndarray,
        skill: SkillPrototype,
        n_steps: int = 10
    ) -> Dict:
        """
        Rollout skill through world model dynamics.
        
        Each step applies the SAME skill embedding as conditioning.
        This is the key difference from action-level rollouts.
        
        Returns: trajectory with uncertainty
        """
        z = z_start.copy()
        h = h_start.copy()
        
        mus = [z.copy()]
        logvars = []
        beliefs = [h.copy()]
        
        for _ in range(n_steps):
            mu, logvar = self.skill_conditioned_transition(z, h, skill)
            
            # Sample next state (stochastic)
            std = np.exp(0.5 * logvar)
            z = mu + std * np.random.randn(*mu.shape)
            
            # Update belief with predicted state
            h = self.wm.gru_step(h, mu)
            
            mus.append(mu.copy())
            logvars.append(logvar.copy())
            beliefs.append(h.copy())
        
        mus = np.array(mus)
        logvars = np.array(logvars)
        
        return {
            'trajectory': mus,
            'uncertainty': np.exp(logvars),
            'logvars': logvars,
            'beliefs': beliefs,
            'final_state': z.copy(),
            'final_belief': h.copy()
        }
    
    def skill_sequence_rollout(
        self,
        z_start: np.ndarray,
        h_start: np.ndarray,
        skill_sequence: List[SkillPrototype],
        steps_per_skill: int = 5
    ) -> Dict:
        """
        Rollout a sequence of skills through world model dynamics.
        
        Each skill shapes the trajectory for `steps_per_skill` steps,
        then hands off to the next skill.
        
        Returns: full trajectory with uncertainty
        """
        z = z_start.copy()
        h = h_start.copy()
        
        all_mus = [z.copy()]
        all_logvars = []
        all_beliefs = [h.copy()]
        skill_boundaries = []
        
        for skill_idx, skill in enumerate(skill_sequence):
            rollout = self.skill_rollout(z, h, skill, n_steps=steps_per_skill)
            
            # Append trajectory (skip first, it's the overlap)
            all_mus.extend(rollout['trajectory'][1:])
            all_logvars.extend(rollout['logvars'])
            all_beliefs.extend(rollout['beliefs'][1:])
            skill_boundaries.append(len(all_mus) - 1)
            
            z = rollout['final_state'].copy()
            h = rollout['final_belief'].copy()
        
        return {
            'trajectory': np.array(all_mus),
            'uncertainty': np.exp(np.array(all_logvars)) if all_logvars else np.array([]),
            'logvars': np.array(all_logvars) if all_logvars else np.array([]),
            'beliefs': all_beliefs,
            'final_state': z.copy(),
            'skill_boundaries': skill_boundaries
        }


class TrajectoryEncoder:
    """
    Compresses trajectory (T x latent_dim) → action_dim vector.
    
    Simple learned projection:
      1. Average deltas (velocity signature)
      2. Project through learned weight matrix
    """
    
    def __init__(self, latent_dim: int = 16, action_dim: int = 16):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Learned projection from trajectory stats to action space
        scale = 0.1
        self.W = np.random.randn(action_dim, latent_dim * 3) * scale
        self.b = np.zeros(action_dim)
    
    def encode(
        self,
        mean_trajectory: np.ndarray,
        variance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Encode trajectory to skill embedding."""
        T = mean_trajectory.shape[0]
        
        # Extract trajectory statistics
        deltas = mean_trajectory[1:] - mean_trajectory[:-1]
        avg_delta = np.mean(deltas, axis=0)
        final_delta = deltas[-1] if len(deltas) > 0 else np.zeros(self.latent_dim)
        
        # Compression ratio
        if variance is not None:
            avg_uncertainty = np.mean(variance)
        else:
            avg_uncertainty = 0.1
        
        # Feature vector: [avg_delta, final_delta, start_state, end_state, uncertainty]
        start = mean_trajectory[0]
        end = mean_trajectory[-1]
        
        features = np.concatenate([
            avg_delta.flatten()[:self.latent_dim],
            final_delta.flatten()[:self.latent_dim],
            start.flatten()[:self.latent_dim],
        ])
        
        # Project to action space
        if len(features) < self.W.shape[1]:
            features = np.pad(features, (0, self.W.shape[1] - len(features)))
        
        embedding = self.W @ features[:self.W.shape[1]] + self.b
        
        # Normalize to match action distribution
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm * np.sqrt(self.action_dim)
        
        return embedding


# ============================================================================
# 2. PROBABILISTIC GOAL EVALUATION
# ============================================================================

class ProbabilisticGoalEvaluator:
    """
    Evaluates goal satisfaction under model uncertainty.
    
    Key shift:
      BEFORE: ||z - goal|| < threshold  (deterministic)
      AFTER:  P(goal | predicted_distribution)  (probabilistic)
    """
    
    def __init__(self, world_model: MinimalWorldModel):
        self.wm = world_model
    
    def goal_log_likelihood(
        self,
        goal_state: np.ndarray,
        predicted_mu: np.ndarray,
        predicted_logvar: np.ndarray
    ) -> float:
        """
        Log P(goal | predicted_distribution).
        
        Assumes multi-variate Gaussian with diagonal covariance.
        Higher = state is more likely under predicted distribution.
        """
        diff = goal_state - predicted_mu
        logvar = predicted_logvar
        
        # Log likelihood of multivariate Gaussian
        # -0.5 * sum((x - mu)^2 / exp(logvar) + logvar + log(2*pi))
        precision = np.exp(-logvar)
        ll = -0.5 * np.sum(diff ** 2 * precision + logvar + np.log(2 * np.pi))
        
        return float(ll)
    
    def goal_satisfaction_probability(
        self,
        goal: GoalAttractor,
        predicted_mu: np.ndarray,
        predicted_logvar: np.ndarray
    ) -> float:
        """
        P(satisfied | predicted_distribution).
        
        Uses basin radius as effective threshold for "close enough".
        Converted to probability via cumulative of the error distribution.
        """
        dist = np.linalg.norm(predicted_mu - goal.attractor_state[:len(predicted_mu)])
        avg_uncertainty = np.mean(np.exp(predicted_logvar))
        
        # Probability that true state is within basin radius
        # given estimated distance and uncertainty
        if avg_uncertainty < 1e-8:
            return 1.0 if dist < goal.basin_radius else 0.0
        
        # Normal approximation: Φ((basin - dist) / sqrt(uncertainty))
        # Using math.erf instead of scipy for portability
        import math
        z = (goal.basin_radius - dist) / math.sqrt(avg_uncertainty)
        prob = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        
        return float(prob)
    
    def evaluate_skill_sequence(
        self,
        sequence: List[SkillPrototype],
        start_state: np.ndarray,
        start_belief: np.ndarray,
        goal: GoalAttractor,
        adapter: SkillDynamicsAdapter,
        steps_per_skill: int = 5
    ) -> Dict:
        """
        Evaluate skill sequence under world model uncertainty.
        
        Returns:
          - expected_satisfaction: E[P(goal | trajectory)]
          - trajectory_evidence: per-step likelihood under goal
          - uncertainty_trace: how uncertainty evolved
          - final_probability: P(goal | final_state)
        """
        z = start_state.copy()
        h = start_belief.copy()
        
        step_likelihoods = []
        uncertainty_trace = []
        
        for skill in sequence:
            # Rollout this skill
            rollout = adapter.skill_rollout(z, h, skill, n_steps=steps_per_skill)
            
            for t in range(1, len(rollout['trajectory'])):
                mu = rollout['trajectory'][t]
                logvar = rollout['logvars'][t - 1] if t - 1 < len(rollout['logvars']) else np.zeros_like(mu)
                
                # Goal likelihood at this step
                ll = self.goal_log_likelihood(
                    goal.attractor_state[:len(mu)], mu, logvar
                )
                step_likelihoods.append(ll)
                uncertainty_trace.append(float(np.mean(np.exp(logvar))))
            
            z = rollout['final_state'].copy()
            h = rollout['final_belief'].copy()
        
        # Final goal probability
        if step_likelihoods:
            final_ll = step_likelihoods[-1]
            final_prob = np.exp(min(final_ll, 0.0))  # Normalize
        else:
            final_prob = 0.0
        
        # Expected goal satisfaction over trajectory
        avg_log_likelihood = np.mean(step_likelihoods) if step_likelihoods else -np.inf
        expected_satisfaction = float(np.exp(min(avg_log_likelihood, 0.0)))
        
        return {
            'expected_satisfaction': expected_satisfaction,
            'final_probability': final_prob,
            'avg_log_likelihood': float(avg_log_likelihood),
            'step_likelihoods': step_likelihoods,
            'uncertainty_trace': uncertainty_trace,
            'final_uncertainty': uncertainty_trace[-1] if uncertainty_trace else 0.0
        }


# ============================================================================
# 3. UNCERTAINTY-AWARE CEM PLANNER
# ============================================================================

class UncertaintyAwareCEM:
    """
    CEM planner that evaluates skill sequences via world model with uncertainty.
    
    Optimizes:
      argmax_{skill_sequence}
          E[ P(goal_satisfaction | sequence, world_model) ]
          - lambda * trajectory_uncertainty
    
    Key difference from Phase 31 CEM:
      - Uses world model for rollout (not skill.generate_trajectory)
      - Accounts for uncertainty propagation
      - Evaluates P(goal) instead of distance < threshold
    """
    
    def __init__(
        self,
        world_model: MinimalWorldModel,
        adapter: SkillDynamicsAdapter,
        evaluator: ProbabilisticGoalEvaluator,
        n_candidates: int = 50,
        n_elites: int = 10,
        n_iterations: int = 5,
        uncertainty_penalty: float = 0.1
    ):
        self.wm = world_model
        self.adapter = adapter
        self.evaluator = evaluator
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.uncertainty_penalty = uncertainty_penalty
    
    def plan(
        self,
        z_start: np.ndarray,
        h_start: np.ndarray,
        goal: GoalAttractor,
        available_skills: List[SkillPrototype],
        horizon: int = 5
    ) -> List[SkillPrototype]:
        """
        Plan skill sequence maximizing P(goal) under world model uncertainty.
        
        Returns: optimal skill chain
        """
        if not available_skills:
            return []
        
        n_skills = len(available_skills)
        skill_probs = np.ones(n_skills) / n_skills
        
        best_sequence = None
        best_score = -np.inf
        
        for iteration in range(self.n_iterations):
            # Sample candidate sequences
            candidates = []
            for _ in range(self.n_candidates):
                indices = np.random.choice(n_skills, size=horizon, p=skill_probs)
                seq = [available_skills[i] for i in indices]
                candidates.append(seq)
            
            # Evaluate via world model
            scores = []
            for seq in candidates:
                score = self._evaluate_sequence(seq, z_start, h_start, goal)
                scores.append(score)
            
            # Select elites
            elite_indices = np.argsort(scores)[-self.n_elites:]
            elites = [candidates[i] for i in elite_indices]
            
            # Update distribution
            counts = np.zeros(n_skills)
            for seq in elites:
                for skill in seq:
                    idx = available_skills.index(skill)
                    counts[idx] += 1
            skill_probs = counts + 1.0
            skill_probs = skill_probs / skill_probs.sum()
            
            # Track best
            max_idx = int(np.argmax(scores))
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_sequence = candidates[max_idx]
        
        return best_sequence or []
    
    def _evaluate_sequence(
        self,
        sequence: List[SkillPrototype],
        z_start: np.ndarray,
        h_start: np.ndarray,
        goal: GoalAttractor
    ) -> float:
        """Evaluate sequence via world model rollout with uncertainty."""
        if not sequence:
            return -np.inf
        
        result = self.evaluator.evaluate_skill_sequence(
            sequence, z_start, h_start, goal, self.adapter,
            steps_per_skill=5
        )
        
        # Score = expected satisfaction - uncertainty penalty
        expected = result['expected_satisfaction']
        uncertainty = result.get('final_uncertainty', 0.0)
        
        return expected - self.uncertainty_penalty * uncertainty


# ============================================================================
# 4. CREDIT ASSIGNMENT: TRAJECTORY → SKILL UPDATE
# ============================================================================

class DynamicsCreditAssigner:
    """
    Assigns credit from goal outcome back to skills through world model.
    
    Key function:
      After goal execution:
        - Extract which skills contributed to success/failure
        - Update skill embeddings toward/away from effective directions
        - Update success rates weighted by uncertainty
    
    This closes the learning loop:
      Goal → Skills → World Model → Goal Outcome → Skill Update
    """
    
    def __init__(
        self,
        adapter: SkillDynamicsAdapter,
        evaluator: ProbabilisticGoalEvaluator,
        learning_rate: float = 0.05
    ):
        self.adapter = adapter
        self.evaluator = evaluator
        self.learning_rate = learning_rate
    
    def assign_credit(
        self,
        skills_used: List[SkillPrototype],
        goal: GoalAttractor,
        trajectory: np.ndarray,
        success: bool,
        uncertainty_trace: Optional[List[float]] = None
    ) -> Dict:
        """
        Assign credit to each skill in the sequence.
        
        Returns: per-skill credit score + embedding update direction
        """
        credits = []
        
        for i, skill in enumerate(skills_used):
            # How much did this skill move toward goal?
            if i < len(trajectory) - 1:
                z_before = trajectory[i]
                z_after = trajectory[min(i + 1, len(trajectory) - 1)]
                
                dist_before = np.linalg.norm(z_before - goal.attractor_state[:len(z_before)])
                dist_after = np.linalg.norm(z_after - goal.attractor_state[:len(z_after)])
                
                delta = dist_before - dist_after  # Positive = moved toward goal
            else:
                delta = 0.0
            
            # Confidence weight (high uncertainty → low weight)
            if uncertainty_trace and i < len(uncertainty_trace):
                confidence = np.exp(-uncertainty_trace[i])
            else:
                confidence = 1.0
            
            # Credit = outcome contribution × confidence
            credit = (1.0 if success else -0.5) * delta * confidence
            credits.append({
                'skill_id': skill.skill_id,
                'credit': float(credit),
                'delta': float(delta),
                'confidence': float(confidence),
                'old_embedding': skill.skill_embedding.copy() if skill.skill_embedding is not None else None
            })
        
        # Update skills
        for credit_info in credits:
            skill = next(
                s for s in skills_used if s.skill_id == credit_info['skill_id']
            )
            self._update_skill(skill, credit_info)
        
        return {'credits': credits}
    
    def _update_skill(self, skill: SkillPrototype, credit_info: Dict):
        """Update skill based on assigned credit."""
        credit = credit_info['credit']
        
        # Update success rate
        outcome = 1.0 if credit > 0 else 0.0
        alpha = 0.1
        skill.success_rate = (1 - alpha) * skill.success_rate + alpha * outcome
        
        # Update embedding toward/away from effective direction
        if credit_info['old_embedding'] is not None:
            embedding = credit_info['old_embedding']
            update = credit * self.learning_rate * embedding / max(np.linalg.norm(embedding), 1e-8)
            skill.skill_embedding = embedding + update
            
            # Normalize
            norm = np.linalg.norm(skill.skill_embedding)
            if norm > 0:
                skill.skill_embedding = skill.skill_embedding / norm * np.sqrt(len(embedding))


# ============================================================================
# 5. INTEGRATION: HIERARCHICAL ENGINE WITH DYNAMICS
# ============================================================================

class DynamicsAwareHierarchicalEngine(HierarchicalExecutionEngine):
    """
    Hierarchical Execution Engine with skill-dynamics coupling.
    
    Overrides the deterministic evaluation with probabilistic world model.
    """
    
    def __init__(
        self,
        world_model: MinimalWorldModel,
        skill_bank: SkillBank,
        intent_dim: int = 64,
        uncertainty_penalty: float = 0.1
    ):
        super().__init__(world_model, skill_bank, intent_dim)
        
        self.adapter = SkillDynamicsAdapter(world_model)
        self.evaluator = ProbabilisticGoalEvaluator(world_model)
        self.planner = UncertaintyAwareCEM(
            world_model,
            self.adapter,
            self.evaluator,
            uncertainty_penalty=uncertainty_penalty
        )
        self.credit_assigner = DynamicsCreditAssigner(self.adapter, self.evaluator)
    
    def plan_skill_sequence(
        self,
        task: Any,
        z_current: np.ndarray,
        goal: GoalAttractor
    ) -> List[SkillPrototype]:
        """Plan via world model with uncertainty (overrides Phase 31)."""
        available = self.skill_bank.find_applicable_skills(
            goal,
            context={'task_type': task.description if hasattr(task, 'description') else 'default'}
        )
        
        if not available:
            return []
        
        # Build belief state from current latent
        h = np.zeros(self.world_model.belief_dim)
        if hasattr(self.world_model, 'gru_step'):
            h = self.world_model.gru_step(h, z_current)
        
        return self.planner.plan(
            z_current, h, goal, available, horizon=min(5, len(available))
        )
    
    def execute_task(
        self,
        task: Any,
        z_current: np.ndarray,
        goal: GoalAttractor
    ) -> Dict:
        """Execute with world model tracking (overrides Phase 31)."""
        task.status = 'running' if hasattr(task, 'status') else None
        
        # Plan via uncertainty-aware CEM
        skill_sequence = self.plan_skill_sequence(task, z_current, goal)
        if hasattr(task, 'skill_sequence'):
            task.skill_sequence = skill_sequence
        
        if not skill_sequence:
            return {
                'task_id': getattr(task, 'task_id', 'unknown'),
                'success': False,
                'error': 'No skill sequence found',
                'final_state': z_current.copy()
            }
        
        # Rollout through world model
        h = np.zeros(self.world_model.belief_dim)
        h = self.world_model.gru_step(h, z_current)
        
        rollout = self.adapter.skill_sequence_rollout(
            z_current, h, skill_sequence, steps_per_skill=5
        )
        
        # Evaluate goal probability
        eval_result = self.evaluator.evaluate_skill_sequence(
            skill_sequence, z_current, h, goal, self.adapter, steps_per_skill=5
        )
        
        z_final = rollout['final_state']
        task_completed = eval_result['final_probability'] > 0.5
        
        # Credit assignment
        credit = self.credit_assigner.assign_credit(
            skill_sequence, goal,
            rollout['trajectory'],
            success=task_completed,
            uncertainty_trace=eval_result.get('uncertainty_trace')
        )
        
        if hasattr(task, 'status'):
            task.status = 'completed' if task_completed else 'active'
        if hasattr(task, 'progress'):
            task.progress = float(eval_result['final_probability'])
        
        # Update goal vitality
        goal.update_vitality(
            float(eval_result['final_probability']),
            len(rollout['trajectory'])
        )
        
        return {
            'task_id': getattr(task, 'task_id', 'unknown'),
            'success': task_completed,
            'final_distance': float(np.linalg.norm(z_final - goal.attractor_state[:len(z_final)])),
            'final_probability': float(eval_result['final_probability']),
            'final_state': z_final.copy(),
            'trajectory_length': len(rollout['trajectory']),
            'skills_used': [s.skill_id for s in skill_sequence],
            'skill_boundaries': rollout.get('skill_boundaries', []),
            'credit_assignment': credit,
            'goal_log_likelihood': eval_result['avg_log_likelihood'],
            'uncertainty_trace': eval_result.get('uncertainty_trace', [])
        }
    
    def execute_goal(
        self,
        goal_id: str,
        z_start: np.ndarray,
        max_iterations: int = 100
    ) -> Dict:
        """Execute full goal hierarchy with dynamics awareness."""
        if goal_id not in self.goals:
            return {'error': 'Goal not found'}
        
        goal = self.goals[goal_id]
        task_graph = self.task_graphs[goal_id]
        
        z_current = z_start.copy()
        iteration = 0
        execution_log = []
        
        while not task_graph.is_complete() and iteration < max_iterations:
            ready_tasks = task_graph.get_ready_tasks()
            if not ready_tasks:
                if task_graph.failed_tasks:
                    break
                break
            
            task = ready_tasks[0]
            
            # Build intent vector
            self.current_intent = self.build_intent_vector(
                z_current, goal, task_graph,
                context={'iteration': iteration}
            )
            
            # Execute with dynamics
            result = self.execute_task(task, z_current, goal)
            z_current = result.get('final_state', z_current).copy()
            
            if result['success']:
                task_graph.complete_task(task.task_id)
            else:
                task_graph.fail_task(task.task_id)
            
            execution_log.append(result)
            iteration += 1
        
        # Final evaluation
        final_state = z_current.copy()
        goal_satisfied = goal.is_satisfied(final_state)
        
        return {
            'goal_id': goal_id,
            'success': goal_satisfied,
            'final_distance': float(np.linalg.norm(final_state - goal.attractor_state[:len(final_state)])),
            'task_progress': task_graph.get_progress(),
            'iterations': iteration,
            'critical_path': task_graph.get_critical_path(),
            'execution_log': execution_log[-iteration:] if iteration > 0 else [],
            'dynamics_coupled': True
        }


# ============================================================================
# 6. TESTS
# ============================================================================

def test_skill_embedding_alignment():
    """Test skill embedding aligns to world model action space."""
    print("\n" + "=" * 60)
    print("SKILL EMBEDDING ALIGNMENT TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    adapter = SkillDynamicsAdapter(wm)
    
    # Create test skills
    skills = []
    for i in range(3):
        traj = np.cumsum(np.random.randn(10, 16) * 0.1, axis=0)
        skill = SkillPrototype(
            skill_id=f"test_skill_{i}",
            name=f"test_{i}",
            description=f"Test skill {i}",
            mean_trajectory=traj,
            trajectory_variance=np.ones_like(traj) * 0.1
        )
        skills.append(skill)
    
    # Compute embeddings
    for skill in skills:
        emb = adapter.compute_skill_embedding(skill)
        print(f"\n  {skill.skill_id}: embedding shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
        
        # Test transition conditioning
        z = np.random.randn(16)
        h = np.random.randn(64)
        mu, logvar = adapter.skill_conditioned_transition(z, h, skill)
        print(f"    transition: mu norm={np.linalg.norm(mu):.4f}, logvar mean={np.mean(logvar):.4f}")


def test_skill_rollout():
    """Test skill-conditioned rollout through world model."""
    print("\n" + "=" * 60)
    print("SKILL-CONDITIONED ROLLOUT TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    adapter = SkillDynamicsAdapter(wm)
    
    # Create skill
    traj = np.cumsum(np.random.randn(10, 16) * 0.1, axis=0)
    skill = SkillPrototype(
        skill_id="rollout_skill",
        name="rollout_test",
        description="Rollout test skill",
        mean_trajectory=traj,
        trajectory_variance=np.ones_like(traj) * 0.1
    )
    
    z_start = np.zeros(16)
    h_start = np.zeros(64)
    
    # Single skill rollout
    rollout = adapter.skill_rollout(z_start, h_start, skill, n_steps=10)
    print(f"\n  Trajectory shape: {rollout['trajectory'].shape}")
    print(f"  Uncertainty shape: {rollout['uncertainty'].shape}")
    print(f"  Final state norm: {np.linalg.norm(rollout['final_state']):.4f}")
    print(f"  Avg uncertainty: {np.mean(rollout['uncertainty']):.4f}")
    
    # Multi-skill sequence rollout
    skills = [skill]
    for i in range(2):
        traj2 = np.cumsum(np.random.randn(10, 16) * 0.1, axis=0)
        s = SkillPrototype(
            skill_id=f"seq_skill_{i}",
            name=f"seq_{i}",
            description=f"Sequence skill {i}",
            mean_trajectory=traj2,
            trajectory_variance=np.ones_like(traj2) * 0.1
        )
        skills.append(s)
    
    seq_rollout = adapter.skill_sequence_rollout(z_start, h_start, skills, steps_per_skill=5)
    print(f"\n  Sequence rollout shape: {seq_rollout['trajectory'].shape}")
    print(f"  Skill boundaries: {seq_rollout['skill_boundaries']}")
    print(f"  Final state norm: {np.linalg.norm(seq_rollout['final_state']):.4f}")


def test_probabilistic_goal_evaluation():
    """Test goal evaluation under uncertainty."""
    print("\n" + "=" * 60)
    print("PROBABILISTIC GOAL EVALUATION TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    evaluator = ProbabilisticGoalEvaluator(wm)
    
    goal = GoalAttractor(
        goal_id="test_goal",
        attractor_state=np.zeros(16),
        basin_radius=1.0,
        priority=0.8,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Close goal (low uncertainty)
    mu_close = np.zeros(16) + 0.1
    logvar_close = np.ones(16) * -2.0  # Low uncertainty
    prob_close = evaluator.goal_satisfaction_probability(goal, mu_close, logvar_close)
    ll_close = evaluator.goal_log_likelihood(goal.attractor_state, mu_close, logvar_close)
    print(f"\n  Close goal: P(satisfied)={prob_close:.4f}, log_likelihood={ll_close:.2f}")
    
    # Far goal (high uncertainty)
    mu_far = np.ones(16) * 5.0
    logvar_far = np.ones(16) * 2.0  # High uncertainty
    prob_far = evaluator.goal_satisfaction_probability(goal, mu_far, logvar_far)
    ll_far = evaluator.goal_log_likelihood(goal.attractor_state, mu_far, logvar_far)
    print(f"  Far goal:  P(satisfied)={prob_far:.4f}, log_likelihood={ll_far:.2f}")
    
    # Close with high uncertainty
    mu_close_uncertain = np.zeros(16) + 0.1
    logvar_high = np.ones(16) * 3.0
    prob_uncertain = evaluator.goal_satisfaction_probability(goal, mu_close_uncertain, logvar_high)
    print(f"  Close+uncertain: P(satisfied)={prob_uncertain:.4f}")
    
    # Skill sequence evaluation
    adapter = SkillDynamicsAdapter(wm)
    skills = []
    for i in range(3):
        traj = np.cumsum(np.random.randn(10, 16) * 0.1, axis=0)
        s = SkillPrototype(
            skill_id=f"eval_skill_{i}",
            name=f"eval_{i}",
            description=f"Evaluation test skill {i}",
            mean_trajectory=traj,
            trajectory_variance=np.ones_like(traj) * 0.1
        )
        skills.append(s)
    
    result = evaluator.evaluate_skill_sequence(
        skills, np.zeros(16), np.zeros(64), goal, adapter, steps_per_skill=5
    )
    print(f"\n  Skill sequence:")
    print(f"    Expected satisfaction: {result['expected_satisfaction']:.4f}")
    print(f"    Final probability: {result['final_probability']:.4f}")
    print(f"    Avg log likelihood: {result['avg_log_likelihood']:.2f}")
    print(f"    Steps evaluated: {len(result['step_likelihoods'])}")
    print(f"    Final uncertainty: {result['final_uncertainty']:.4f}")


def test_uncertainty_aware_cem():
    """Test CEM planning with uncertainty."""
    print("\n" + "=" * 60)
    print("UNCERTAINTY-AWARE CEM TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    adapter = SkillDynamicsAdapter(wm)
    evaluator = ProbabilisticGoalEvaluator(wm)
    
    # Create skills with diverse directions
    skills = []
    for i in range(8):
        direction = np.random.randn(16)
        direction = direction / np.linalg.norm(direction) * 0.5
        traj = np.zeros((10, 16))
        for t in range(1, 10):
            traj[t] = traj[t - 1] + direction + np.random.randn(16) * 0.05
        
        skill = SkillPrototype(
            skill_id=f"cem_skill_{i}",
            name=f"cem_{i}",
            description=f"CEM test skill {i}",
            mean_trajectory=traj,
            trajectory_variance=np.ones_like(traj) * 0.1
        )
        skills.append(skill)
    
    # Create goal (far from origin)
    goal = GoalAttractor(
        goal_id="cem_goal",
        attractor_state=np.ones(16) * 2.0,
        basin_radius=1.5,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Plan with uncertainty-aware CEM
    planner = UncertaintyAwareCEM(
        wm, adapter, evaluator,
        n_candidates=30, n_elites=5, n_iterations=3
    )
    
    sequence = planner.plan(
        np.zeros(16), np.zeros(64), goal, skills, horizon=3
    )
    
    print(f"\n  Planned sequence length: {len(sequence)}")
    print(f"  Skills: {[s.skill_id for s in sequence]}")
    
    if sequence:
        score = planner._evaluate_sequence(sequence, np.zeros(16), np.zeros(64), goal)
        print(f"  Sequence score: {score:.4f}")
    
    # Compare with random sequence
    random_seq = [random.choice(skills) for _ in range(3)]
    random_score = planner._evaluate_sequence(random_seq, np.zeros(16), np.zeros(64), goal)
    print(f"  Random sequence score: {random_score:.4f}")


def test_credit_assignment():
    """Test credit assignment from goal outcome to skills."""
    print("\n" + "=" * 60)
    print("CREDIT ASSIGNMENT TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    adapter = SkillDynamicsAdapter(wm)
    evaluator = ProbabilisticGoalEvaluator(wm)
    assigner = DynamicsCreditAssigner(adapter, evaluator)
    
    goal = GoalAttractor(
        goal_id="credit_goal",
        attractor_state=np.zeros(16),
        basin_radius=1.0,
        priority=0.8,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Create skills
    skills = []
    for i in range(3):
        traj = np.cumsum(np.random.randn(10, 16) * 0.1, axis=0)
        s = SkillPrototype(
            skill_id=f"credit_skill_{i}",
            name=f"credit_{i}",
            description=f"Credit test skill {i}",
            mean_trajectory=traj,
            trajectory_variance=np.ones_like(traj) * 0.1
        )
        # Initialize embedding
        adapter.compute_skill_embedding(s)
        skills.append(s)
    
    trajectory = np.cumsum(np.random.randn(30, 16) * 0.1, axis=0)
    
    # Successful outcome
    result = assigner.assign_credit(skills, goal, trajectory, success=True)
    
    print(f"\n  Successful outcome credits:")
    for c in result['credits']:
        print(f"    {c['skill_id']}: credit={c['credit']:.4f}, delta={c['delta']:.4f}")
    
    # Failed outcome
    result2 = assigner.assign_credit(skills, goal, trajectory, success=False)
    
    print(f"\n  Failed outcome credits:")
    for c in result2['credits']:
        print(f"    {c['skill_id']}: credit={c['credit']:.4f}, delta={c['delta']:.4f}")
    
    # Check skill updates
    print(f"\n  Success rate updates:")
    for i, s in enumerate(skills):
        print(f"    {s.skill_id}: success_rate={s.success_rate:.4f}")


def test_full_integration():
    """Test full integration: Phase 31 + Phase 32 coupling."""
    print("\n" + "=" * 60)
    print("FULL INTEGRATION TEST (Phase 31 + 32)")
    print("=" * 60)
    
    from phase31_hierarchical_execution import SkillBank, GoalAttractor, TaskNode, TaskGraph
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    bank = SkillBank()
    
    # Create skills via bank
    for i in range(5):
        traj = np.cumsum(np.random.randn(10, 16) * 0.1, axis=0)
        skill = bank.extract_skill_from_trajectory(traj, success=True, context={})
        skill.applicable_goal_types = ['achievable']
    
    # Create goal
    goal = GoalAttractor(
        goal_id="integrated_goal",
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Create task graph
    graph = TaskGraph(goal_id="integrated_goal")
    tasks = [
        TaskNode(task_id="t1", goal_id="integrated_goal", description="Explore"),
        TaskNode(task_id="t2", goal_id="integrated_goal", description="Execute"),
        TaskNode(task_id="t3", goal_id="integrated_goal", description="Finalize"),
    ]
    for t in tasks:
        graph.add_task(t)
    graph.add_dependency("t2", "t1")
    graph.add_dependency("t3", "t2")
    
    # Create dynamics-aware engine
    engine = DynamicsAwareHierarchicalEngine(
        world_model=wm,
        skill_bank=bank,
        intent_dim=32,
        uncertainty_penalty=0.1
    )
    engine.add_goal(goal, graph)
    
    # Execute
    result = engine.execute_goal(
        "integrated_goal",
        np.zeros(16),
        max_iterations=5
    )
    
    print(f"\n  Goal success: {result['success']}")
    print(f"  Final distance: {result['final_distance']:.4f}")
    print(f"  Task progress: {result['task_progress']:.2f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Dynamics coupled: {result['dynamics_coupled']}")
    
    if result['execution_log']:
        print(f"\n  Execution log:")
        for entry in result['execution_log']:
            print(f"    Task {entry['task_id']}: "
                  f"success={entry['success']}, "
                  f"prob={entry.get('final_probability', 0):.3f}, "
                  f"skills={entry.get('skills_used', [])}")


if __name__ == "__main__":
    
    test_skill_embedding_alignment()
    test_skill_rollout()
    test_probabilistic_goal_evaluation()
    test_uncertainty_aware_cem()
    test_credit_assignment()
    test_full_integration()
    
    print("\n" + "=" * 60)
    print("PHASE 32: SKILL-DYNAMICS COUPLING LAYER")
    print("=" * 60)
    
    print("""
KEY SHIFTS FROM PHASE 31:
  1. Skill = conditioned transition operator (not trajectory generator)
     z_{t+1} ~ P(z | z_t, h_t, skill_embedding) via world model
  
  2. Goal evaluation = probabilistic (not distance < threshold)
     P(goal | predicted_distribution) under uncertainty
  
  3. CEM optimizes expected P(goal) with uncertainty penalty
     argmax E[P(goal satisfaction | sequence, world model)]
  
  4. Credit assignment closes the learning loop
     Goal outcome → skill embedding update → future planning bias

ARCHITECTURE:
  Goal → Task DAG → Skill Sequence
                          ↓
              SkillDynamicsAdapter
                          ↓
              P(z_{t+1} | z_t, h_t, skill)
                          ↓
              World Model Rollout
                          ↓
              ProbabilisticGoalEvaluator
                          ↓
              P(goal | sequence) + uncertainty
                          ↓
              CreditAssignment → Skill Update
""")
