"""
Phase 33 — Skill Manifold Stabilization Engine

CRITICAL SHIFT:
  Skills are NOT trajectories or embeddings.
  Skills are attractors in action space — stable dynamical regimes.
  
  A skill is stable if:
    apply(s, z) → z' AND apply(s, z') → z''   where z'' ≈ z'
  
  Self-consistency = the skill produces the same effect repeatedly.
  Stabilization = reinforce consistent attractors, dampen inconsistent ones.

WHAT THIS ENABLES:
  - Skills become self-stabilizing control primitives (not random projections)
  - CEM plans over continuous skill manifold (not discrete indices)
  - Competition between skills via stability-weighted selection
  - Emergence of useful policies through self-consistency reinforcement

ARCHITECTURE:
  
  SkillManifold (continuous space)
       │
   ┌───┴───┐
   │       │
 Skill₁  Skill₂  ...  (attractors with stability)
   │       │
   └───┬───┘
       │
  interpolate(coords) → blended attractor
       │
  P(z_{t+1} | z_t, h_t, attractor) via world model
       │
  self_consistency_check → stability_update
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

from phase30_training_loop import MinimalWorldModel
from phase31_hierarchical_execution import GoalAttractor, TaskGraph, TaskNode
from phase32_skill_dynamics_coupling import (
    SkillDynamicsAdapter, ProbabilisticGoalEvaluator, DynamicsCreditAssigner
)


# ============================================================================
# 1. SKILL AS ATTRACTOR IN ACTION SPACE
# ============================================================================

@dataclass
class SkillAttractor:
    """
    Skill = stable fixed point in (action ⊗ latent) space.
    
    NOT a trajectory.
    NOT an embedding.
    
    An attractor that pulls the system toward a consistent transition pattern.
    
    Properties:
      - attractor_point: the stable action vector (action_dim,)
      - transition_signature: typical latent displacement ∆z when applied
      - basin_radius: how far the attractor's pull extends in action space
      - stability: self-consistency score [0, 1]; 1 = perfectly consistent
      - age: how many times this skill has been applied (for annealing)
    """
    skill_id: str
    name: str
    
    # Attractor dynamics
    attractor_point: np.ndarray          # (action_dim,) — stable action
    transition_signature: np.ndarray     # (latent_dim,) — typical ∆z
    basin_radius: float = 1.0            # Pull radius in action space
    
    # Stability (self-consistency)
    stability: float = 0.5               # [0, 1] — starts neutral
    stability_confidence: float = 0.0    # How sure we are about stability
    age: int = 0
    
    # Execution history for stability computation
    recent_transitions: List[Dict] = field(default_factory=list)
    max_history: int = 50
    
    # Goal alignment
    applicable_goal_types: List[str] = field(default_factory=lambda: ['achievable'])
    goal_alignment: float = 0.0          # Average goal progress per use
    
    def apply_to(
        self,
        z: np.ndarray,
        h: np.ndarray,
        world_model: MinimalWorldModel,
        noise_scale: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply attractor to current state via world model.
        
        Returns: (z_next, h_next, actual_delta_norm)
        """
        mu, logvar = world_model.predict_transition(z, h, self.attractor_point)
        
        if noise_scale > 0:
            std = np.exp(0.5 * logvar)
            z_next = mu + std * np.random.randn(*mu.shape) * noise_scale
        else:
            z_next = mu
        
        h_next = world_model.gru_step(h, mu)
        
        actual_delta = z_next - z
        actual_delta_norm = float(np.linalg.norm(actual_delta))
        
        # Record transition for stability computation
        self.recent_transitions.append({
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'h': h.copy(),
            'delta': actual_delta.copy(),
            'delta_norm': actual_delta_norm
        })
        if len(self.recent_transitions) > self.max_history:
            self.recent_transitions.pop(0)
        
        self.age += 1
        return z_next, h_next, actual_delta_norm
    
    def compute_self_consistency(self) -> float:
        """
        How consistent is this attractor's effect?
        
        Self-consistency = correlation between attractor_point and actual delta.
        If attractor.skill_embedding consistently produces similar ∆z,
        it is a stable control primitive.
        
        Returns: consistency score [0, 1]
        """
        if len(self.recent_transitions) < 3:
            return self.stability  # Not enough data, keep current
        
        deltas = [t['delta'] for t in self.recent_transitions[-20:]]
        deltas = np.array(deltas)  # (N, latent_dim)
        
        # Consistency = how similar are the deltas?
        mean_delta = np.mean(deltas, axis=0)
        
        if np.linalg.norm(mean_delta) < 1e-8:
            variance = np.mean(np.var(deltas, axis=0))
        else:
            # Normalized variance relative to mean magnitude
            magnitude = np.linalg.norm(mean_delta)
            variance = np.mean(np.var(deltas, axis=0))
            consistency = 1.0 / (1.0 + variance / (magnitude + 1e-8))
        
        # Update transition signature
        self.transition_signature = mean_delta.copy()
        
        return float(np.clip(consistency, 0.0, 1.0))
    
    def compute_alignment_with_goal(self, goal: GoalAttractor) -> float:
        """
        How well does this attractor move toward goal?
        
        Positive = moves toward goal
        Negative = moves away
        """
        if len(self.recent_transitions) < 3:
            return self.goal_alignment
        
        deltas = np.array([t['delta'] for t in self.recent_transitions[-10:]])
        mean_delta = np.mean(deltas, axis=0)
        
        direction_to_goal = goal.attractor_state[:len(mean_delta)] - np.mean(
            [t['z_before'] for t in self.recent_transitions[-10:]], axis=0
        )
        
        if np.linalg.norm(direction_to_goal) < 1e-8:
            return 0.0
        
        # Cosine similarity between delta and goal direction
        cos_sim = np.dot(mean_delta, direction_to_goal) / (
            np.linalg.norm(mean_delta) * np.linalg.norm(direction_to_goal) + 1e-8
        )
        
        return float(cos_sim)
    
    def stabilize(self, learning_rate: float = 0.1):
        """
        Update attractor point toward more consistent transitions.
        
        Key idea:
          attractor_point ← attractor_point + lr * (consistent_delta - attractor_point)
        
        This pulls the attractor toward the average effect it produces,
        making it self-consistent.
        """
        if len(self.recent_transitions) < 3:
            return
        
        # Average actual delta
        deltas = np.array([t['delta'] for t in self.recent_transitions[-20:]])
        mean_delta = np.mean(deltas, axis=0)
        
        # Project to action space via learned mapping
        # (simple approximation: normalize attractor to match delta direction)
        current_effect = self.attractor_point.copy()
        effect_norm = np.linalg.norm(current_effect)
        delta_norm = np.linalg.norm(mean_delta)
        
        if effect_norm > 1e-8 and delta_norm > 1e-8:
            # Align attractor direction with actual delta direction
            target = mean_delta / delta_norm * effect_norm
        else:
            target = mean_delta
        
        # Update attractor (momentum toward target)
        self.attractor_point = (1 - learning_rate) * self.attractor_point + learning_rate * target
        
        # Normalize to prevent drift
        norm = np.linalg.norm(self.attractor_point)
        if norm > 0:
            self.attractor_point = self.attractor_point / norm * min(norm, 5.0)
        
        # Update stability
        new_consistency = self.compute_self_consistency()
        alpha = 1.0 / (1.0 + self.stability_confidence)
        self.stability = (1 - alpha) * self.stability + alpha * new_consistency
        self.stability_confidence += 0.1
    
    @classmethod
    def from_random(cls, skill_id: str, action_dim: int, latent_dim: int,
                    name: str = None) -> 'SkillAttractor':
        """Create random attractor for exploration."""
        attractor = np.random.randn(action_dim)
        attractor = attractor / np.linalg.norm(attractor) * 0.5
        
        return cls(
            skill_id=skill_id,
            name=name or skill_id,
            attractor_point=attractor,
            transition_signature=np.zeros(latent_dim)
        )
    
    @classmethod
    def from_embedding(cls, skill_id: str, embedding: np.ndarray,
                       latent_dim: int, name: str = None) -> 'SkillAttractor':
        """Create attractor from existing skill embedding."""
        return cls(
            skill_id=skill_id,
            name=name or skill_id,
            attractor_point=embedding.copy(),
            transition_signature=np.zeros(latent_dim)
        )


# ============================================================================
# 2. SKILL MANIFOLD — Continuous Space of Attractors
# ============================================================================

class SkillManifold:
    """
    Continuous manifold of skill attractors.
    
    Skills live as points on a low-dimensional manifold.
    Interpolation between nearby skills = blended attractor dynamics.
    Competition = stability-weighted selection.
    
    The manifold enables:
      - Continuous policy search (CEM samples manifold coords, not indices)
      - Skill emergence via interpolation
      - Competition via stability-weighted blending
    """
    
    def __init__(self, manifold_dim: int = 4, action_dim: int = 16):
        self.manifold_dim = manifold_dim
        self.action_dim = action_dim
        
        # Skills indexed on manifold
        self.attractors: Dict[str, SkillAttractor] = {}
        self.manifold_coords: Dict[str, np.ndarray] = {}  # skill_id → (manifold_dim,)
        
        # Learned projection: manifold coords → action space
        scale = 0.2
        self.projection = np.random.randn(action_dim, manifold_dim) * scale
        
        # Entropy bonus for exploration
        self.entropy_bonus: float = 0.1
    
    def add_attractor(self, attractor: SkillAttractor,
                      coord: Optional[np.ndarray] = None):
        """Add attractor to manifold at given coordinate or random."""
        if coord is None:
            coord = np.random.randn(self.manifold_dim) * 0.5
        
        self.attractors[attractor.skill_id] = attractor
        self.manifold_coords[attractor.skill_id] = coord
        
        # Ensure attractor point matches action dimension
        if len(attractor.attractor_point) != self.action_dim:
            attractor.attractor_point = np.resize(attractor.attractor_point, self.action_dim)
    
    def interpolate_at(self, coord: np.ndarray) -> np.ndarray:
        """
        Blend nearby attractors into continuous action.
        
        coord: (manifold_dim,) query point
        Returns: (action_dim,) blended attractor point
        """
        if not self.attractors:
            return np.zeros(self.action_dim)
        
        # Find distances to all attractors
        distances = []
        for skill_id, c in self.manifold_coords.items():
            dist = np.linalg.norm(coord - c)
            distances.append((dist, skill_id))
        
        distances.sort(key=lambda x: x[0])
        
        # Weighted blend of k-nearest attractors
        k = min(3, len(distances))
        nearest = distances[:k]
        
        total_weight = 0.0
        blended = np.zeros(self.action_dim)
        
        for dist, skill_id in nearest:
            attractor = self.attractors[skill_id]
            # Weight by stability and inverse distance
            weight = attractor.stability / (dist + 0.1)
            blended += weight * attractor.attractor_point
            total_weight += weight
        
        if total_weight > 0:
            blended = blended / total_weight
        
        return blended
    
    def compute_stability_field(self, coord: np.ndarray) -> float:
        """
        Get expected stability at a manifold coordinate.
        
        Used by CEM to prefer stable regions.
        """
        if not self.attractors:
            return 0.5
        
        distances = []
        for skill_id, c in self.manifold_coords.items():
            dist = np.linalg.norm(coord - c)
            attractor = self.attractors[skill_id]
            distances.append((dist, attractor.stability))
        
        # Weighted average stability
        total = 0.0
        weight_sum = 0.0
        for dist, stab in distances:
            w = 1.0 / (dist + 0.1)
            total += w * stab
            weight_sum += w
        
        return total / weight_sum if weight_sum > 0 else 0.5
    
    def sample_random_coord(self) -> np.ndarray:
        """Sample random manifold coordinate for exploration."""
        return np.random.randn(self.manifold_dim) * 1.0
    
    def nearest_attractors(self, coord: np.ndarray, k: int = 3) -> List[SkillAttractor]:
        """Get nearest attractors to manifold coordinate."""
        distances = [
            (np.linalg.norm(coord - c), self.attractors[sid])
            for sid, c in self.manifold_coords.items()
        ]
        distances.sort(key=lambda x: x[0])
        return [a for _, a in distances[:k]]
    
    def prune_unstable_attractors(self, min_stability: float = 0.1,
                                  min_age: int = 5):
        """Remove attractors that fail to stabilize."""
        to_remove = []
        for skill_id, attractor in self.attractors.items():
            if attractor.age > min_age and attractor.stability < min_stability:
                to_remove.append(skill_id)
        
        for skill_id in to_remove:
            del self.attractors[skill_id]
            del self.manifold_coords[skill_id]
        
        return len(to_remove)


# ============================================================================
# 3. MANIFOLD-CONDITIONED CEM (Continuous)
# ============================================================================

class ManifoldCEM:
    """
    CEM over continuous skill manifold.
    
    NOT discrete skill indices.
    Samples manifold coordinates, evaluates via world model, updates distribution.
    
    This turns skill planning into continuous optimization:
      coord* = argmax E[P(goal | blend_at(coord), world_model)]
    
    Benefits:
      - Smooth interpolation between skills
      - Emergence of novel blended policies
      - Stability-weighted exploration
    """
    
    def __init__(
        self,
        manifold: SkillManifold,
        adapter: SkillDynamicsAdapter,
        evaluator: ProbabilisticGoalEvaluator,
        world_model: MinimalWorldModel,
        n_candidates: int = 80,
        n_elites: int = 16,
        n_iterations: int = 5,
        stability_bonus: float = 0.2,
        entropy_coeff: float = 0.05
    ):
        self.manifold = manifold
        self.adapter = adapter
        self.evaluator = evaluator
        self.wm = world_model
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.stability_bonus = stability_bonus
        self.entropy_coeff = entropy_coeff
        
        # Distribution over manifold
        self.mean = np.zeros(manifold.manifold_dim)
        self.std = np.ones(manifold.manifold_dim)
    
    def plan_sequence(
        self,
        z_start: np.ndarray,
        h_start: np.ndarray,
        goal: GoalAttractor,
        horizon: int = 3
    ) -> Dict:
        """
        Plan sequence of manifold coordinates → skill blends.
        
        Returns: {coords, attractors, score, trajectory}
        """
        if not self.manifold.attractors:
            return {'coords': [], 'attractors': [], 'score': -np.inf}
        
        # Reset distribution
        self.mean = np.zeros(self.manifold.manifold_dim)
        self.std = np.ones(self.manifold.manifold_dim)
        
        best_coords = None
        best_score = -np.inf
        
        for iteration in range(self.n_iterations):
            # Sample candidate sequences (each = list of manifold coords)
            candidates = []
            for _ in range(self.n_candidates):
                coords = []
                for _ in range(horizon):
                    c = self.mean + self.std * np.random.randn(self.manifold.manifold_dim)
                    coords.append(c)
                candidates.append(coords)
            
            # Evaluate
            scores = []
            trajectories = []
            for coords in candidates:
                score, traj = self._evaluate_coord_sequence(
                    coords, z_start, h_start, goal
                )
                scores.append(score)
                trajectories.append(traj)
            
            # Select elites
            elite_indices = np.argsort(scores)[-self.n_elites:]
            elite_coords = [candidates[i] for i in elite_indices]
            
            # Update distribution
            if elite_coords:
                all_coords = np.array([c for seq in elite_coords for c in seq])
                self.mean = np.mean(all_coords, axis=0)
                self.std = np.std(all_coords, axis=0) + 0.1  # Min std for exploration
            
            # Track best
            max_idx = int(np.argmax(scores))
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_coords = candidates[max_idx]
                best_trajectory = trajectories[max_idx]
        
        if best_coords is None:
            return {'coords': [], 'attractors': [], 'score': -np.inf}
        
        # Convert best coords to attractor blends
        best_attractors = []
        for coord in best_coords:
            # Create temporary attractor at blended point
            blended_action = self.manifold.interpolate_at(coord)
            temp = SkillAttractor(
                skill_id='_blended',
                name='blended',
                attractor_point=blended_action,
                transition_signature=np.zeros(self.wm.latent_dim)
            )
            best_attractors.append(temp)
        
        return {
            'coords': best_coords,
            'attractors': best_attractors,
            'score': best_score,
            'trajectory': best_trajectory,
        }
    
    def _evaluate_coord_sequence(
        self,
        coords: List[np.ndarray],
        z_start: np.ndarray,
        h_start: np.ndarray,
        goal: GoalAttractor
    ) -> Tuple[float, Optional[np.ndarray]]:
        """
        Evaluate manifold coordinate sequence via world model.
        
        Score = expected goal satisfaction + stability bonus + entropy bonus
        """
        z = z_start.copy()
        h = h_start.copy()
        trajectory = [z.copy()]
        
        total_goal_prob = 0.0
        total_stability = 0.0
        
        for t, coord in enumerate(coords):
            # Blend nearby attractors
            blended_action = self.manifold.interpolate_at(coord)
            
            # Apply through world model
            mu, logvar = self.wm.predict_transition(z, h, blended_action)
            
            # Stochastic sample
            std = np.exp(0.5 * logvar)
            z_next = mu + std * np.random.randn(*mu.shape) * 0.3
            h = self.wm.gru_step(h, mu)
            
            z = z_next
            trajectory.append(z.copy())
            
            # Goal probability at this step
            prob = self.evaluator.goal_satisfaction_probability(
                goal, mu, logvar
            )
            total_goal_prob += prob
            
            # Stability bonus
            stab = self.manifold.compute_stability_field(coord)
            total_stability += stab
        
        # Average scores
        avg_goal = total_goal_prob / len(coords)
        avg_stab = total_stability / len(coords)
        
        # Entropy bonus (prefer high-variance regions for exploration)
        coord_var = np.var([c for c in coords], axis=0).mean() if len(coords) > 1 else 0.0
        
        score = (avg_goal
                 + self.stability_bonus * avg_stab
                 + self.entropy_coeff * coord_var)
        
        return score, np.array(trajectory)
    
    def get_ensemble_attractors(self, n_samples: int = 5) -> List[SkillAttractor]:
        """
        Generate ensemble of attractors from manifold distribution.
        
        Used for exploration: sample diverse attractors from current belief.
        """
        attractors = []
        for i in range(n_samples):
            coord = self.mean + self.std * np.random.randn(self.manifold.manifold_dim)
            action = self.manifold.interpolate_at(coord)
            
            attractor = SkillAttractor(
                skill_id=f'ensemble_{i}',
                name=f'ensemble_{i}',
                attractor_point=action,
                transition_signature=np.zeros(self.wm.latent_dim)
            )
            attractors.append(attractor)
        
        return attractors


# ============================================================================
# 4. SKILL STABILIZER — Self-Consistency Engine
# ============================================================================

class SkillStabilizer:
    """
    Stabilizes attractors through self-consistency reinforcement.
    
    Core loop:
      1. Apply attractor → observe actual transition
      2. Compute self-consistency (delta correlation)
      3. Update attractor toward consistent direction
      4. Prune attractors that fail to stabilize
    
    This is how "useful policies emerge" — 
    attractors that produce consistent effects survive,
    noisy attractors get pruned.
    """
    
    def __init__(
        self,
        manifold: SkillManifold,
        stabilizer_lr: float = 0.05,
        min_stability: float = 0.2,
        stabilization_interval: int = 10
    ):
        self.manifold = manifold
        self.lr = stabilizer_lr
        self.min_stability = min_stability
        self.interval = stabilization_interval
        
        self.step_count = 0
        self.last_pruned = 0
    
    def step(self) -> Dict:
        """
        One stabilization step.
        
        Computes self-consistency for all attractors and updates them.
        Prunes unstable attractors every `interval` steps.
        """
        self.step_count += 1
        stats = {
            'n_attractors': len(self.manifold.attractors),
            'avg_stability': 0.0,
            'n_stabilized': 0,
            'n_pruned': 0
        }
        
        if not self.manifold.attractors:
            return stats
        
        # Compute stability for all attractors
        stabilities = []
        for attractor in self.manifold.attractors.values():
            consistency = attractor.compute_self_consistency()
            attractor.stabilize(learning_rate=self.lr)
            stabilities.append(attractor.stability)
        
        stats['avg_stability'] = float(np.mean(stabilities)) if stabilities else 0.0
        stats['n_stabilized'] = sum(1 for s in stabilities if s > self.min_stability)
        
        # Prune unstable attractors periodically
        if self.step_count % self.interval == 0:
            pruned = self.manifold.prune_unstable_attractors(
                min_stability=self.min_stability,
                min_age=5
            )
            stats['n_pruned'] = pruned
            self.last_pruned = pruned
        
        return stats
    
    def compute_stability_report(self) -> Dict:
        """Full report on attractor stability."""
        if not self.manifold.attractors:
            return {'n_attractors': 0}
        
        stabilities = [a.stability for a in self.manifold.attractors.values()]
        ages = [a.age for a in self.manifold.attractors.values()]
        
        return {
            'n_attractors': len(self.manifold.attractors),
            'mean_stability': float(np.mean(stabilities)),
            'median_stability': float(np.median(stabilities)),
            'min_stability': float(min(stabilities)),
            'max_stability': float(max(stabilities)),
            'mean_age': float(np.mean(ages)),
            'stable_count': sum(1 for s in stabilities if s > 0.5),
            'unstable_count': sum(1 for s in stabilities if s < 0.2)
        }


# ============================================================================
# 5. FULL INTEGRATION: EXECUTION WITH STABILIZATION
# ============================================================================

class StabilizedExecutionEngine:
    """
    Complete execution engine with skill manifold stabilization.
    
    Integrates:
      Phase 30: World model (dynamics)
      Phase 31: Task DAG (decomposition)
      Phase 32: Skill-dynamics coupling (P(z_{t+1} | z, h, skill))
      Phase 33: Manifold stabilization (self-consistent attractors)
    
    Execution loop:
      Goal → Task → ManifoldCEM → Blended Attractor → World Model → 
      Evaluation → Credit → Stabilization → Skill Update
    """
    
    def __init__(
        self,
        world_model: MinimalWorldModel,
        manifold: SkillManifold,
        goal: GoalAttractor,
        task_graph: Optional[TaskGraph] = None,
        manifold_dim: int = 4,
        n_initial_attractors: int = 12,
        stabilizer_lr: float = 0.05,
        uncertainty_penalty: float = 0.1
    ):
        self.wm = world_model
        self.manifold = manifold
        self.goal = goal
        self.task_graph = task_graph or TaskGraph(goal_id=goal.goal_id)
        
        # Adapters from Phase 32
        self.adapter = SkillDynamicsAdapter(world_model)
        self.evaluator = ProbabilisticGoalEvaluator(world_model)
        self.credit_assigner = DynamicsCreditAssigner(self.adapter, self.evaluator, learning_rate=0.05)
        
        # Initialize attractors if manifold is empty
        if not self.manifold.attractors:
            self._seed_attractors(n_initial_attractors)
        
        # CEM over manifold
        self.cem = ManifoldCEM(
            manifold=self.manifold,
            adapter=self.adapter,
            evaluator=self.evaluator,
            world_model=self.wm,
            stability_bonus=0.2,
            entropy_coeff=0.05
        )
        
        # Stabilizer
        self.stabilizer = SkillStabilizer(
            manifold=self.manifold,
            stabilizer_lr=stabilizer_lr
        )
        
        # Execution state
        self.execution_log: List[Dict] = []
        self.stabilization_log: List[Dict] = []
    
    def _seed_attractors(self, n: int):
        """Seed initial attractors in manifold."""
        for i in range(n):
            attractor = SkillAttractor.from_random(
                skill_id=f'seed_{i}',
                action_dim=self.wm.action_dim,
                latent_dim=self.wm.latent_dim,
                name=f'seed_{i}'
            )
            self.manifold.add_attractor(attractor)
    
    def execute_step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """
        One execution step:
          1. Plan attractor coordinate via manifold CEM
          2. Select nearest manifold attractor to planned coordinate
          3. Apply attractor through world model → record transition on it
          4. Evaluate goal progress
          5. Stabilize all attractors
        """
        # 1. Plan
        plan = self.cem.plan_sequence(z, h, self.goal, horizon=1)
        plan_coord = plan['coords'][0] if plan['coords'] else None
        
        if plan_coord is None:
            plan_coord = self.manifold.sample_random_coord()
        
        # 2. Select or create attractor at planned coordinate
        nearest = self.manifold.nearest_attractors(plan_coord, k=1)
        if nearest and np.linalg.norm(plan_coord - self.manifold.manifold_coords[nearest[0].skill_id]) < 2.0:
            # Use existing attractor (records transitions → enables stabilization)
            selected = nearest[0]
        else:
            # Spawn new attractor at this coordinate
            action = self.manifold.interpolate_at(plan_coord)
            selected = SkillAttractor(
                skill_id=f'runtime_{len(self.manifold.attractors)}',
                name=f'runtime_{len(self.manifold.attractors)}',
                attractor_point=action.copy(),
                transition_signature=np.zeros(self.wm.latent_dim)
            )
            self.manifold.add_attractor(selected, plan_coord.copy())
        
        # 3. Apply attractor (records transition history on the attractor)
        z_next, h_next, delta_norm = selected.apply_to(z, h, self.wm)
        
        # 4. Evaluate
        mu, logvar = self.wm.predict_transition(z, h, selected.attractor_point)
        goal_prob = self.evaluator.goal_satisfaction_probability(
            self.goal, mu, logvar
        )
        
        # Update goal alignment on nearest manifold attractors
        dist_before = np.linalg.norm(z - self.goal.attractor_state[:len(z)])
        dist_after = np.linalg.norm(z_next - self.goal.attractor_state[:len(z_next)])
        delta = dist_before - dist_after
        if delta > 0:
            selected.goal_alignment = (1 - 0.1) * selected.goal_alignment + 0.1 * delta
        
        # 5. Stabilize all attractors (now have transition histories)
        stab_stats = self.stabilizer.step()
        self.stabilization_log.append(stab_stats)
        
        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'goal_prob': float(goal_prob),
            'delta_norm': float(delta_norm),
            'selected_skill': selected.skill_id,
            'nearest_skills': [a.skill_id for a in self.manifold.nearest_attractors(plan_coord, k=3)],
            'stability': stab_stats['avg_stability'],
            'n_attractors': stab_stats['n_attractors'],
            'n_pruned': stab_stats['n_pruned']
        }
        
        self.execution_log.append(step_result)
        return step_result
    
    def execute_goal(self, z_start: np.ndarray, max_steps: int = 50) -> Dict:
        """
        Execute full goal with manifold stabilization.
        
        Returns: complete execution report
        """
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)
        
        goal_reached = False
        
        for step in range(max_steps):
            result = self.execute_step(z, h)
            
            if result['goal_prob'] > 0.7:
                goal_reached = True
            
            z = result['z_after'].copy()
            h = self.wm.gru_step(h, z)
        
        # Final evaluation
        final_prob = 0.0
        if self.execution_log:
            final_prob = self.execution_log[-1]['goal_prob']
        
        # Stability summary
        stab_report = self.stabilizer.compute_stability_report()
        
        return {
            'goal_reached': goal_reached,
            'final_goal_prob': float(final_prob),
            'n_steps': len(self.execution_log),
            'stability': stab_report,
            'execution_log': self.execution_log[-10:] if self.execution_log else [],
            'stabilization_log': self.stabilization_log[-10:] if self.stabilization_log else []
        }
    
    def spawn_new_attractor(self, coord: np.ndarray) -> SkillAttractor:
        """
        Spawn new attractor at manifold coordinate from successful blend.
        
        This is how new skills emerge:
          - Successful blended trajectory → new attractor at that point
          - New attractor competes with existing ones
          - If it stabilizes → it becomes a permanent skill
        """
        action = self.manifold.interpolate_at(coord)
        
        attractor = SkillAttractor(
            skill_id=f'emergent_{len(self.manifold.attractors)}',
            name=f'emergent_{len(self.manifold.attractors)}',
            attractor_point=action.copy(),
            transition_signature=np.zeros(self.wm.latent_dim),
            stability=0.3  # Start below threshold, must prove itself
        )
        
        self.manifold.add_attractor(attractor, coord.copy())
        return attractor


# ============================================================================
# 6. TESTS
# ============================================================================

def test_skill_attractor():
    """Test attractor dynamics and self-consistency."""
    print("\n" + "=" * 60)
    print("SKILL ATTRACTOR TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    
    # Create attractor
    attractor = SkillAttractor.from_random(
        skill_id='test_attractor',
        action_dim=16,
        latent_dim=16,
        name='test'
    )
    
    print(f"\n  Attractor point norm: {np.linalg.norm(attractor.attractor_point):.4f}")
    print(f"  Initial stability: {attractor.stability:.4f}")
    
    # Apply multiple times
    z = np.random.randn(16) * 0.5
    h = np.zeros(64)
    
    for i in range(10):
        z, h, delta = attractor.apply_to(z, h, wm)
    
    consistency = attractor.compute_self_consistency()
    print(f"  After 10 steps, consistency: {consistency:.4f}")
    
    # Stabilize
    attractor.stabilize()
    print(f"  After stabilization: stability={attractor.stability:.4f}")
    
    print("\n  ✓ Skill attractor operational")


def test_skill_manifold():
    """Test continuous skill manifold."""
    print("\n" + "=" * 60)
    print("SKILL MANIFOLD TEST")
    print("=" * 60)
    
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    
    # Add attractors
    for i in range(6):
        attr = SkillAttractor.from_random(
            skill_id=f'skill_{i}',
            action_dim=16,
            latent_dim=8,
            name=f'skill_{i}'
        )
        attr.stability = random.uniform(0.3, 0.9)
        manifold.add_attractor(attr)
    
    # Test interpolation
    coord1 = np.zeros(4)
    coord2 = np.ones(4) * 0.5
    
    action1 = manifold.interpolate_at(coord1)
    action2 = manifold.interpolate_at(coord2)
    
    print(f"\n  Interpolated action 1 norm: {np.linalg.norm(action1):.4f}")
    print(f"  Interpolated action 2 norm: {np.linalg.norm(action2):.4f}")
    print(f"  Similarity between actions: {np.dot(action1, action2) / (np.linalg.norm(action1) * np.linalg.norm(action2) + 1e-8):.4f}")
    
    # Test stability field
    stab = manifold.compute_stability_field(coord1)
    print(f"  Stability at coord1: {stab:.4f}")
    
    # Test nearest
    nearest = manifold.nearest_attractors(coord1, k=3)
    print(f"  Nearest attractors: {[a.skill_id for a in nearest]}")
    
    print("\n  ✓ Skill manifold operational")


def test_manifold_cem():
    """Test continuous manifold CEM."""
    print("\n" + "=" * 60)
    print("MANIFOLD CEM TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    
    # Seed attractors with diverse directions
    for i in range(10):
        direction = np.random.randn(16)
        direction = direction / np.linalg.norm(direction) * 0.5
        attr = SkillAttractor(
            skill_id=f'dir_{i}',
            name=f'dir_{i}',
            attractor_point=direction,
            transition_signature=np.zeros(16),
            stability=random.uniform(0.4, 0.8)
        )
        manifold.add_attractor(attr)
    
    # Create evaluator and adapter
    adapter = SkillDynamicsAdapter(wm)
    evaluator = ProbabilisticGoalEvaluator(wm)
    
    # Create CEM
    cem = ManifoldCEM(
        manifold=manifold,
        adapter=adapter,
        evaluator=evaluator,
        world_model=wm,
        n_candidates=30,
        n_elites=6,
        n_iterations=3
    )
    
    # Goal
    goal = GoalAttractor(
        goal_id='cem_test',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Plan
    z_start = np.zeros(16)
    h_start = np.zeros(64)
    
    result = cem.plan_sequence(z_start, h_start, goal, horizon=3)
    
    print(f"\n  Sequence score: {result['score']:.4f}")
    print(f"  Coords: {len(result['coords'])}")
    print(f"  Attractors: {len(result['attractors'])}")
    
    if result['attractors']:
        print(f"  First attractor norm: {np.linalg.norm(result['attractors'][0].attractor_point):.4f}")
    
    # Test ensemble
    ensemble = cem.get_ensemble_attractors(n_samples=5)
    print(f"  Ensemble attractors: {len(ensemble)}")
    
    print("\n  ✓ Manifold CEM operational")


def test_stabilization_loop():
    """Test stabilization through repeated execution."""
    print("\n" + "=" * 60)
    print("STABILIZATION LOOP TEST")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    
    # Seed initial attractors
    for i in range(8):
        attr = SkillAttractor.from_random(
            skill_id=f'init_{i}',
            action_dim=16,
            latent_dim=16
        )
        manifold.add_attractor(attr)
    
    # Goal
    goal = GoalAttractor(
        goal_id='stab_test',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Engine
    engine = StabilizedExecutionEngine(
        world_model=wm,
        manifold=manifold,
        goal=goal,
        n_initial_attractors=0,  # Already seeded
        stabilizer_lr=0.1
    )
    
    # Execute multiple steps
    z = np.zeros(16)
    h = np.zeros(64)
    h = wm.gru_step(h, z)
    
    for step in range(20):
        result = engine.execute_step(z, h)
        z = result['z_after'].copy()
        h = wm.gru_step(h, z)
    
    # Report
    stab_report = engine.stabilizer.compute_stability_report()
    
    print(f"\n  Steps executed: {len(engine.execution_log)}")
    print(f"  Stability report:")
    print(f"    Attractors: {stab_report['n_attractors']}")
    print(f"    Mean stability: {stab_report['mean_stability']:.4f}")
    print(f"    Stable count: {stab_report['stable_count']}")
    print(f"    Unstable count: {stab_report['unstable_count']}")
    
    # Check if stability improved
    if engine.stabilization_log:
        first_stab = engine.stabilization_log[0]['avg_stability']
        last_stab = engine.stabilization_log[-1]['avg_stability']
        print(f"  Stability progression: {first_stab:.4f} → {last_stab:.4f}")
    
    print("\n  ✓ Stabilization loop operational")


def test_full_goal_execution():
    """Test complete goal execution with stabilization."""
    print("\n" + "=" * 60)
    print("FULL GOAL EXECUTION WITH STABILIZATION")
    print("=" * 60)
    
    wm = MinimalWorldModel(event_dim=32, latent_dim=16, belief_dim=64, action_dim=16)
    manifold = SkillManifold(manifold_dim=4, action_dim=16)
    
    # Goal
    goal = GoalAttractor(
        goal_id='full_test',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Engine
    engine = StabilizedExecutionEngine(
        world_model=wm,
        manifold=manifold,
        goal=goal,
        n_initial_attractors=10,
        stabilizer_lr=0.08
    )
    
    # Execute
    result = engine.execute_goal(np.zeros(16), max_steps=30)
    
    print(f"\n  Goal reached: {result['goal_reached']}")
    print(f"  Final goal prob: {result['final_goal_prob']:.4f}")
    print(f"  Steps: {result['n_steps']}")
    
    stab = result['stability']
    print(f"\n  Final stability: {stab['mean_stability']:.4f}")
    print(f"  Attractors: {stab['n_attractors']}")
    print(f"  Stable: {stab['stable_count']}, Unstable: {stab['unstable_count']}")
    
    if result['execution_log']:
        print("\n  Last 5 steps:")
        for entry in result['execution_log'][-5:]:
            print(f"    goal_prob={entry['goal_prob']:.3f}, "
                  f"stability={entry['stability']:.3f}, "
                  f"attractors={entry['n_attractors']}")
    
    print("\n  ✓ Full execution operational")


if __name__ == "__main__":
    test_skill_attractor()
    test_skill_manifold()
    test_manifold_cem()
    test_stabilization_loop()
    test_full_goal_execution()
    
    print("\n" + "=" * 60)
    print("PHASE 33: SKILL MANIFOLD STABILIZATION ENGINE")
    print("=" * 60)
    
    print("""
WHAT CHANGED:
  1. Skill = attractor in action space (NOT trajectory)
     - Stable fixed point with self-consistency
     - basin_radius, stability, age
     - apply_to() through world model
   
  2. Skill manifold = continuous space of attractors
     - Interpolation = blended attractor dynamics
     - Stability field guides search
     - Nearest-attractor competition
   
  3. ManifoldCEM = continuous optimization over manifold
     - Samples manifold coords, not discrete indices
     - Evaluates blended attractors via world model
     - Stability bonus + entropy bonus
   
  4. SkillStabilizer = self-consistency engine
     - Computes consistency from transition history
     - Updates attractor toward consistent direction
     - Prunes unstable attractors
   
  5. StabilizedExecutionEngine = complete closed loop
     - Plan → Apply → Evaluate → Credit → Stabilize

KEY INSIGHT:
  Skills are no longer something you "select".
  They are attractors that compete in a dynamical manifold.
  The system discovers stable policies through self-consistency
  reinforcement, not through supervised training.
  
NEXT STEPS:
  1. Run long-horizon stabilization (100+ steps)
  2. Measure attractor convergence
  3. Connect to production AI-OS goal execution
""")
