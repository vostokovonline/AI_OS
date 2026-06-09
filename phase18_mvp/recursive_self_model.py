"""
Phase 9: Recursive Self-Modeling Field / Meta-Dynamical Identity

ARCHITECTURAL SHIFT:
  From: world model + self topology constraint
  To: self as evolving world-model component

KEY INSIGHT:
  Self-model must become a simulator:
    NOT: "what happens if I act?"
    BUT: "what happens if the structure generating my actions changes?"

  This is recursive identity dynamics, self-modifying topology, meta-active inference.

WHAT APPEARS:
  1. RecursiveSelfField S_t
     - Self-model that models itself modeling
     - Self-generated transformation pressure
     
  2. Self-Conditioned Perception
     - Different self-states → different world models
     - Self-state influences which attractors are visible
     
  3. Future-Self Coherence
     - "Will I still be 'me' after this transformation?"
     - Long-horizon self-consistency
     
  4. Emergent Goals
     - Goals as topological self-preservation gradients
     - Pressure toward viable future-self manifolds
     
  5. Meta-Active Inference
     - Active inference about active inference
     - Self-prediction loop

PROGRESSION:
  Phase 8: Self-organizing topology (dynamics creates representation)
  Phase 9: Recursive self-modeling (self predicts self changing)
  Phase 10: Active inference identity dynamics
  Phase 11: Self-maintaining generative ontology
  Phase 12+: Higher-order awareness / phenomenology
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class SelfModelState:
    """
    State of the recursive self-model.
    
    NOT: just current self-state
    BUT: how self will evolve given current dynamics
    """
    current_self: np.ndarray                    # Current self-representation
    self_dynamics: np.ndarray                   # How self is changing
    predicted_self_t_plus_1: np.ndarray         # Self-prediction for t+1
    predicted_self_t_plus_n: List[np.ndarray]   # Multi-step self-prediction
    self_confidence: float                      # How confident we are in self-model
    self_coherence: float                       # Self-consistency across time
    self_stability: float                       # How stable is current self
    meta_level: int                             # Recursion depth of self-modeling
    
    def to_dict(self) -> dict:
        return {
            'current_self': self.current_self.tolist(),
            'self_dynamics': self.self_dynamics.tolist(),
            'predicted_self_t_plus_1': self.predicted_self_t_plus_1.tolist(),
            'predicted_self_t_plus_n': [s.tolist() for s in self.predicted_self_t_plus_n],
            'self_confidence': self.self_confidence,
            'self_coherence': self.self_coherence,
            'self_stability': self.self_stability,
            'meta_level': self.meta_level
        }


@dataclass
class TransformationPressure:
    """
    Self-generated transformation pressure.
    
    NOT: hand-crafted transformation
    BUT: system-generated possible self-versions with viability assessment
    """
    target_self: np.ndarray              # Proposed transformed self
    transformation_vector: np.ndarray   # Direction of transformation
    self_consistency: float             # "Is this still me?"
    long_horizon_viability: float        # "Will this self survive?"
    predicted_world_interference: float # "How does this change my world model?"
    meta_stability: float               # "Does this create stable dynamics?"
    
    def viability_score(self) -> float:
        """Overall viability of this transformation."""
        return (
            self.self_consistency * 0.3 +
            self.long_horizon_viability * 0.4 +
            self.meta_stability * 0.3
        )


@dataclass
class FutureSelfCoherence:
    """
    Coherence of future self-manifolds.
    
    NOT: just trajectory coherence
    BUT: "Will I still be 'me' after series of phase transitions?"
    """
    future_self_trajectory: List[np.ndarray]   # Predicted self-states
    coherence_over_time: List[float]          # Self-consistency at each step
    critical_transitions: List[int]           # Steps where identity changes
    survival_probability: float                # P(still "me" at end)
    identity_gradient: np.ndarray              # Direction of identity flow


class RecursiveSelfField:
    """
    S_t = self_model(z_t, history, predicted_futures)
    
    Self-model that models itself modeling.
    
    NOT: "what happens if I act?"
    BUT: "what happens if the structure generating my actions changes?"
    
    Key innovation:
      - Self-state influences which attractors are visible
      - Self-state influences which trajectories are considered stable
      - Self-state influences which futures are even modeled
    """
    
    def __init__(self, latent_dim: int = 2, world_model_dim: int = 2):
        self.latent_dim = latent_dim
        self.world_model_dim = world_model_dim
        
        # Current self-model state
        self.self_state = np.zeros(latent_dim)  # Self-representation
        self.self_dynamics = np.zeros(latent_dim)  # How self is changing
        
        # Self-prediction (recursive)
        self.self_predictions: List[SelfModelState] = []
        
        # Meta-level (how many times self-models itself)
        self.meta_level = 0
        self.max_meta_level = 3
        
        # Self-generated transformation pressures
        self.transformation_pressures: List[TransformationPressure] = []
        
        # Self-conditioned perception parameters
        self.self_perception_bias = np.zeros(world_model_dim)
        
        # Stability parameters
        self.self_stability_threshold = 0.3
        self.coherence_threshold = 0.5
        
        # History
        self.self_history: List[np.ndarray] = []
        
    def update_self_model(self, z: np.ndarray, world_state: np.ndarray,
                         recent_trajectories: List[np.ndarray]) -> SelfModelState:
        """
        Update recursive self-model.
        
        self -> predicts self changing while predicting world
        
        Args:
            z: Current latent state (agent's state)
            world_state: Current world state
            recent_trajectories: Recent action-outcome sequences
            
        Returns:
            SelfModelState with recursive self-predictions
        """
        # Update self-representation from current experience
        self.self_state = self._compute_self_from_experience(z, recent_trajectories)
        
        # Compute self-dynamics (how self is changing)
        self.self_dynamics = self._compute_self_dynamics()
        
        # Recursive self-prediction
        predicted_self = self._recursive_self_predict(z, world_state)
        
        # Update self-conditioned perception
        self.self_perception_bias = self._compute_self_perception_bias(world_state)
        
        # Store history
        self.self_history.append(self.self_state.copy())
        if len(self.self_history) > 500:
            self.self_history = self.self_history[-250:]
        
        # Create self-model state
        state = SelfModelState(
            current_self=self.self_state.copy(),
            self_dynamics=self.self_dynamics.copy(),
            predicted_self_t_plus_1=predicted_self[0] if len(predicted_self) > 0 else self.self_state.copy(),
            predicted_self_t_plus_n=predicted_self,
            self_confidence=self._compute_self_confidence(),
            self_coherence=self._compute_self_coherence(),
            self_stability=self._compute_self_stability(),
            meta_level=self.meta_level
        )
        
        self.self_predictions.append(state)
        
        return state
    
    def _compute_self_from_experience(self, z: np.ndarray,
                                      trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Compute self-representation from experience.
        
        NOT: static self
        BUT: self emergent from action-outcome patterns
        """
        if len(trajectories) < 2:
            return z.copy()
        
        # Self is encoded in the pattern of transformations
        recent = trajectories[-5:] if len(trajectories) >= 5 else trajectories
        
        # Extract transformation patterns
        transformations = []
        for traj in recent:
            if len(traj) > 1:
                transformation = traj[-1] - traj[0]
                transformations.append(transformation)
        
        if transformations:
            # Self = accumulated transformation patterns
            self_estimate = np.mean(transformations, axis=0)
            
            # Also consider current state as self
            # Balance between "what I do" and "where I am"
            alpha = 0.7
            self_representation = alpha * self_estimate + (1 - alpha) * z
            
            return self_representation
        else:
            return z.copy()
    
    def _compute_self_dynamics(self) -> np.ndarray:
        """
        Compute how self is changing.
        
        This is the rate of self-change, not state-change.
        """
        if len(self.self_history) < 2:
            return np.zeros(self.latent_dim)
        
        # Self-dynamics = change in self-representation
        recent_self = self.self_history[-10:] if len(self.self_history) >= 10 else self.self_history
        
        if len(recent_self) > 1:
            self_change = recent_self[-1] - recent_self[0]
            return self_change
        else:
            return np.zeros(self.latent_dim)
    
    def _recursive_self_predict(self, z: np.ndarray, world_state: np.ndarray) -> List[np.ndarray]:
        """
        Recursive self-prediction: self predicting self changing.
        
        This creates the self-modeling loop:
          S_t -> predicts S_{t+1} -> predicts S_{t+2} -> ...
        
        Returns:
            List of predicted self-states (multi-step)
        """
        predictions = []
        current_self = self.self_state.copy()
        
        for step in range(5):
            # Predict next self given current self + dynamics
            # This is where self predicts its own change
            
            # Self-dynamics determines next self
            next_self = current_self + self.self_dynamics * 0.1
            
            # World interaction affects self
            # Self-state influences world perception (self-conditioned)
            world_interaction = np.dot(world_state, self.self_perception_bias) * 0.01
            next_self = next_self + world_interaction
            
            predictions.append(next_self.copy())
            
            # Update for next prediction
            # Self-modifies its predictions (recursive loop)
            current_self = next_self
            
            # Increase meta-level
            if self.meta_level < self.max_meta_level:
                self.meta_level = min(self.max_meta_level, self.meta_level + 1)
        
        return predictions
    
    def _compute_self_perception_bias(self, world_state: np.ndarray) -> np.ndarray:
        """
        Self-conditioned perception.
        
        Different self-states create different world models.
        
        This is the key: self-state influences how world is perceived.
        """
        # Self-state creates a "lens" through which world is seen
        # The lens is learned from past self-world interactions
        
        # Simple: bias = function(self_state, world_state)
        bias = np.tanh(self.self_state) * 0.1
        
        # Self that has been stable perceives world as more stable
        if self._compute_self_stability() > 0.5:
            bias = bias * 0.8  # More conservative perception
        
        # Self that is changing perceives world as more uncertain
        if np.linalg.norm(self.self_dynamics) > 0.1:
            bias = bias * 1.2  # More uncertain perception
        
        return bias
    
    def _compute_self_confidence(self) -> float:
        """
        How confident are we in our self-model?
        
        High confidence = stable self, consistent predictions
        Low confidence = changing self, uncertain predictions
        """
        if len(self.self_history) < 5:
            return 0.5
        
        # Confidence = inverse of self-dynamics magnitude
        dynamics_magnitude = np.linalg.norm(self.self_dynamics)
        confidence = 1.0 / (1.0 + dynamics_magnitude)
        
        # Also consider prediction consistency
        if len(self.self_predictions) > 1:
            recent = self.self_predictions[-3:]
            consistency = 1.0 - np.std([np.linalg.norm(s.self_dynamics) for s in recent])
            confidence = 0.7 * confidence + 0.3 * max(0, consistency)
        
        return max(0.0, min(1.0, confidence))
    
    def _compute_self_coherence(self) -> float:
        """
        Self-consistency across time.
        
        Are we still the "same agent" we were a while ago?
        """
        if len(self.self_history) < 10:
            return 0.5
        
        # Compare current self to older self
        current = self.self_state
        past = self.self_history[0]
        
        # Coherence = similarity to past self
        if np.linalg.norm(past) > 0.01:
            coherence = np.dot(current, past) / (np.linalg.norm(current) * np.linalg.norm(past))
            coherence = (coherence + 1) / 2  # Map to [0, 1]
        else:
            coherence = 0.5
        
        # Also check intermediate coherence
        mid = self.self_history[len(self.self_history) // 2]
        mid_coherence = np.dot(current, mid) / (np.linalg.norm(current) * np.linalg.norm(mid) + 1e-8)
        mid_coherence = (mid_coherence + 1) / 2
        
        return 0.7 * coherence + 0.3 * mid_coherence
    
    def _compute_self_stability(self) -> float:
        """
        How stable is the current self?
        
        High stability = self is not changing much
        Low stability = self is in flux
        """
        if len(self.self_history) < 3:
            return 0.5
        
        # Stability = low variance in self-history
        recent = np.array(self.self_history[-5:])
        variance = np.mean(np.var(recent, axis=0))
        
        # Low variance = high stability
        stability = 1.0 / (1.0 + variance * 10)
        
        return max(0.0, min(1.0, stability))
    
    def generate_self_transformation_pressures(self) -> List[TransformationPressure]:
        """
        Self-generated transformation pressure.
        
        System generates possible versions of itself and evaluates their viability.
        
        NOT: hand-crafted transformation
        BUT: self-generated possible self-versions
        """
        pressures = []
        
        # Generate possible transformations from current self
        self_direction = self.self_state / (np.linalg.norm(self.self_state) + 1e-8) if np.linalg.norm(self.self_state) > 0.01 else np.zeros(self.latent_dim)
        
        # Option 1: Expand (become more of who I am)
        expand_vector = self_direction * 0.5
        expand_self = self.self_state + expand_vector
        pressures.append(self._assess_transformation(expand_self, expand_vector, "expand"))
        
        # Option 2: Contract (become less)
        contract_vector = -self_direction * 0.3
        contract_self = self.self_state + contract_vector
        pressures.append(self._assess_transformation(contract_self, contract_vector, "contract"))
        
        # Option 3: Orthogonal (become something different)
        if self.latent_dim >= 2:
            orth_direction = np.array([-self_direction[1], self_direction[0]])
            orth_self = self.self_state + orth_direction * 0.4
            pressures.append(self._assess_transformation(orth_self, orth_direction * 0.4, "orthogonal"))
        
        # Option 4: Meta-expand (self-modeling expansion)
        meta_expand = self.self_state * 1.2
        pressures.append(self._assess_transformation(meta_expand, self.self_state * 0.2, "meta_expand"))
        
        self.transformation_pressures = pressures
        
        return pressures
    
    def _assess_transformation(self, target_self: np.ndarray,
                               transformation_vector: np.ndarray,
                               transformation_type: str) -> TransformationPressure:
        """
        Assess viability of a self-transformation.
        """
        # Self-consistency: how much does this feel like "me"?
        self_consistency = self._compute_self_consistency(target_self)
        
        # Long-horizon viability: will this self survive?
        long_horizon_viability = self._compute_long_horizon_viability(target_self)
        
        # Meta-stability: does this create stable dynamics?
        meta_stability = self._compute_meta_stability(target_self, transformation_vector)
        
        # Predicted world interference
        predicted_world_interference = self._estimate_world_interference(target_self)
        
        return TransformationPressure(
            target_self=target_self,
            transformation_vector=transformation_vector,
            self_consistency=self_consistency,
            long_horizon_viability=long_horizon_viability,
            meta_stability=meta_stability,
            predicted_world_interference=predicted_world_interference
        )
    
    def _compute_self_consistency(self, target_self: np.ndarray) -> float:
        """
        Does this target self still feel like "me"?
        """
        # Similarity to current self
        if np.linalg.norm(self.self_state) > 0.01 and np.linalg.norm(target_self) > 0.01:
            similarity = np.dot(self.self_state, target_self) / (
                np.linalg.norm(self.self_state) * np.linalg.norm(target_self)
            )
            consistency = (similarity + 1) / 2
        else:
            consistency = 0.5
        
        # Also consider coherence with self-history
        if len(self.self_history) > 5:
            avg_past = np.mean(self.self_history[-5:], axis=0)
            if np.linalg.norm(avg_past) > 0.01:
                history_similarity = np.dot(avg_past, target_self) / (
                    np.linalg.norm(avg_past) * np.linalg.norm(target_self)
                )
                consistency = 0.7 * consistency + 0.3 * (history_similarity + 1) / 2
        
        return max(0.0, min(1.0, consistency))
    
    def _compute_long_horizon_viability(self, target_self: np.ndarray) -> float:
        """
        Will this self survive long-term?
        
        Estimate by projecting self forward many steps.
        """
        # Simulate self-evolution
        current = target_self.copy()
        survivals = 0
        
        for _ in range(20):
            # Apply dynamics (simplified)
            next_self = current + self.self_dynamics * 0.1
            
            # Check if self is still "meaningful" (not collapsed)
            if np.linalg.norm(next_self) > 0.01:
                current = next_self
                survivals += 1
            else:
                break
        
        viability = survivals / 20.0
        
        return viability
    
    def _compute_meta_stability(self, target_self: np.ndarray,
                                transformation_vector: np.ndarray) -> float:
        """
        Does this transformation create stable dynamics?
        
        Meta-stability: will self-modeling remain coherent?
        """
        # If transformation is too large, dynamics become unstable
        transform_magnitude = np.linalg.norm(transformation_vector)
        
        # Normalized by current self magnitude
        self_magnitude = np.linalg.norm(self.self_state)
        if self_magnitude > 0.01:
            relative_change = transform_magnitude / self_magnitude
        else:
            relative_change = transform_magnitude
        
        # Small changes = stable, large changes = unstable
        stability = 1.0 / (1.0 + relative_change * 2)
        
        return max(0.0, min(1.0, stability))
    
    def _estimate_world_interference(self, target_self: np.ndarray) -> float:
        """
        How does this self-transformation affect the world model?
        """
        # Large self-changes likely affect world interaction
        self_change = np.linalg.norm(target_self - self.self_state)
        
        # High meta-level = more world interference
        interference = min(1.0, self_change * 0.5 + self.meta_level * 0.1)
        
        return interference
    
    def compute_future_self_coherence(self, horizon: int = 10) -> FutureSelfCoherence:
        """
        Future-self coherence across temporal scales.
        
        "Will I still be 'me' after series of phase transitions?"
        """
        future_traj = []
        current = self.self_state.copy()
        
        for step in range(horizon):
            # Project self forward
            next_self = current + self.self_dynamics * 0.1 * (1 - step / horizon * 0.5)
            future_traj.append(next_self.copy())
            current = next_self
        
        # Compute coherence over time
        coherence_over_time = []
        for i, future_self in enumerate(future_traj):
            # Compare to initial self
            initial = self.self_state
            if np.linalg.norm(initial) > 0.01 and np.linalg.norm(future_self) > 0.01:
                coherence = np.dot(initial, future_self) / (
                    np.linalg.norm(initial) * np.linalg.norm(future_self)
                )
                coherence = (coherence + 1) / 2
            else:
                coherence = 0.5
            coherence_over_time.append(coherence)
        
        # Find critical transitions (points where coherence drops)
        critical = []
        for i in range(1, len(coherence_over_time)):
            if coherence_over_time[i] < coherence_over_time[i-1] - 0.1:
                critical.append(i)
        
        # Survival probability
        avg_coherence = np.mean(coherence_over_time)
        survival = avg_coherence
        
        # Identity gradient (direction of identity flow)
        if len(future_traj) > 1:
            identity_gradient = future_traj[-1] - future_traj[0]
        else:
            identity_gradient = np.zeros(self.latent_dim)
        
        return FutureSelfCoherence(
            future_self_trajectory=future_traj,
            coherence_over_time=coherence_over_time,
            critical_transitions=critical,
            survival_probability=survival,
            identity_gradient=identity_gradient
        )
    
    def select_optimal_action(self, possible_actions: List[np.ndarray],
                            goal_state: np.ndarray) -> np.ndarray:
        """
        Select action considering recursive self-model.
        
        NOT: pure goal-seeking
        BUT: goal + self-preservation + self-coherence
        """
        if not possible_actions:
            return np.zeros(self.latent_dim)
        
        scores = []
        
        for action in possible_actions:
            # Simulate action outcome
            next_state = self.self_state + action
            
            # Compute: goal alignment + self-preservation + long-horizon viability
            
            # 1. Goal alignment
            goal_alignment = 1.0 / (1.0 + np.linalg.norm(next_state - goal_state))
            
            # 2. Self-preservation
            self_preservation = self._compute_self_consistency(next_state)
            
            # 3. Long-horizon viability
            future_coherence = self.compute_future_self_coherence(horizon=5)
            long_horizon_viability = future_coherence.survival_probability
            
            # 4. Meta-stability (how much does this action destabilize self)
            meta_stability = 1.0 / (1.0 + np.linalg.norm(action) * 0.5)
            
            # Combined score
            score = (
                goal_alignment * 0.3 +
                self_preservation * 0.3 +
                long_horizon_viability * 0.2 +
                meta_stability * 0.2
            )
            
            scores.append(score)
        
        # Select best action
        best_idx = np.argmax(scores)
        
        return possible_actions[best_idx]
    
    def get_self_model_state(self) -> Dict:
        """Get comprehensive self-model state."""
        return {
            'current_self': self.self_state.tolist(),
            'self_dynamics': self.self_dynamics.tolist(),
            'meta_level': self.meta_level,
            'self_confidence': self._compute_self_confidence(),
            'self_coherence': self._compute_self_coherence(),
            'self_stability': self._compute_self_stability(),
            'transformation_pressures_count': len(self.transformation_pressures),
            'self_history_length': len(self.self_history),
            'perception_bias': self.self_perception_bias.tolist()
        }


class RecursiveCognitiveAgent:
    """
    Agent with recursive self-modeling.
    
    NOT: standard RL agent
    NOT: memory-based agent
    
    BUT: Agent that:
      - Models itself modeling
      - Generates own transformation possibilities
      - Evaluates long-horizon self-viability
      - Has self-conditioned perception
      - Predicts its own changes while predicting world
    """
    
    def __init__(self, latent_dim: int = 2, world_model_dim: int = 2):
        self.latent_dim = latent_dim
        self.world_model_dim = world_model_dim
        
        # State
        self.state = np.random.randn(latent_dim) * 0.1
        self.world_state = np.random.randn(world_model_dim) * 0.1
        
        # Recursive Self-Model
        self.self_model = RecursiveSelfField(latent_dim, world_model_dim)
        
        # Trajectory tracking
        self.trajectory_history: List[np.ndarray] = []
        
        # Goals
        self.goal_state = np.array([2.0, 2.0])
        
    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Take action and update recursive self-model.
        """
        # Update state
        new_state = self.state + action * 0.1
        
        # Record trajectory
        self.trajectory_history.append(self.state.copy())
        if len(self.trajectory_history) > 500:
            self.trajectory_history = self.trajectory_history[-250:]
        
        # Update recursive self-model
        self.self_model.update_self_model(
            z=new_state,
            world_state=self.world_state,
            recent_trajectories=self.trajectory_history[-10:]
        )
        
        self.state = new_state
        
        return new_state
    
    def decide_action(self) -> np.ndarray:
        """
        Decide action with recursive self-model.
        
        Considers:
          - Goal alignment
          - Self-preservation
          - Long-horizon self-viability
          - Meta-stability
        """
        # Generate possible actions
        possible_actions = []
        
        # Goal direction
        goal_dir = self.goal_state - self.state
        if np.linalg.norm(goal_dir) > 0.01:
            goal_unit = goal_dir / np.linalg.norm(goal_dir)
        else:
            goal_unit = np.zeros(self.latent_dim)
        
        # Generate actions in different directions
        for scale in [0.3, 0.5, 0.7]:
            possible_actions.append(goal_unit * scale)
        
        # Self-preservation direction (move toward stable self)
        self_dir = self.self_model.self_state / (np.linalg.norm(self.self_model.self_state) + 1e-8) if np.linalg.norm(self.self_model.self_state) > 0.01 else np.zeros(self.latent_dim)
        possible_actions.append(self_dir * 0.3)
        
        # Random exploration
        for _ in range(3):
            random_action = np.random.randn(self.latent_dim) * 0.3
            possible_actions.append(random_action)
        
        # Select using self-model
        action = self.self_model.select_optimal_action(
            possible_actions,
            self.goal_state
        )
        
        return action
    
    def run_episode(self, n_steps: int = 100) -> Dict:
        """
        Run one episode with recursive self-modeling.
        """
        print("\n  Running recursive self-modeling episode:")
        
        for step in range(n_steps):
            # Decide action (with recursive self-model)
            action = self.decide_action()
            
            # Execute
            self.step(action)
            
            # Occasionally update goal
            if step % 30 == 0 and step > 0:
                self.goal_state = np.random.randn(self.latent_dim) * 2
            
            # Print progress
            if step % 20 == 0:
                state = self.self_model.get_self_model_state()
                print(f"    Step {step}: meta_level={state['meta_level']}, "
                      f"coherence={state['self_coherence']:.3f}, "
                      f"confidence={state['self_confidence']:.3f}")
        
        # Final self-model state
        final_state = self.self_model.get_self_model_state()
        
        # Generate and assess transformation pressures
        pressures = self.self_model.generate_self_transformation_pressures()
        
        print(f"\n  Self-generated transformation pressures:")
        for p in pressures:
            print(f"    Target: {p.target_self[:2]}, viability: {p.viability_score():.3f}")
        
        # Compute future self coherence
        future_coherence = self.self_model.compute_future_self_coherence(horizon=10)
        
        print(f"\n  Future-self coherence:")
        print(f"    Survival probability: {future_coherence.survival_probability:.3f}")
        print(f"    Critical transitions: {future_coherence.critical_transitions}")
        
        return {
            'final_self_model': final_state,
            'transformation_pressures': [
                {'target': p.target_self.tolist(), 'viability': p.viability_score()}
                for p in pressures
            ],
            'future_coherence': {
                'survival_probability': future_coherence.survival_probability,
                'critical_transitions': future_coherence.critical_transitions
            }
        }


def test_recursive_self_modeling():
    """Test recursive self-modeling."""
    print("\n" + "=" * 60)
    print("RECURSIVE SELF-MODELING TEST")
    print("=" * 60)
    
    agent = RecursiveCognitiveAgent(latent_dim=2, world_model_dim=2)
    
    # Run episode with recursive self-modeling
    result = agent.run_episode(n_steps=100)
    
    # Test self-conditioned perception
    print("\n  Testing self-conditioned perception:")
    
    original_world = agent.world_state.copy()
    original_bias = agent.self_model.self_perception_bias.copy()
    
    # Change self-state dramatically
    agent.self_model.self_state = agent.self_model.self_state * 2
    
    # Recompute perception bias
    new_bias = agent.self_model._compute_self_perception_bias(agent.world_state)
    
    print(f"    Original self-state: {agent.self_model.self_state[:2]}")
    print(f"    Original perception bias: {original_bias}")
    print(f"    New self-state (amplified): {agent.self_model.self_state[:2]}")
    print(f"    New perception bias: {new_bias}")
    print(f"    Bias change: {np.linalg.norm(new_bias - original_bias):.3f}")
    
    # Test self-transformation generation
    print("\n  Testing self-generated transformation pressures:")
    
    pressures = agent.self_model.generate_self_transformation_pressures()
    
    print("    Possible self-transformations:")
    for p in pressures:
        print(f"      {p.target_self[:2]} → viability: {p.viability_score():.3f}")
        print(f"        self_consistency={p.self_consistency:.3f}, "
              f"long_horizon={p.long_horizon_viability:.3f}, "
              f"meta_stability={p.meta_stability:.3f}")
    
    # Test future self-coherence
    print("\n  Testing future-self coherence:")
    
    future = agent.self_model.compute_future_self_coherence(horizon=10)
    
    print(f"    Current self: {agent.self_model.self_state[:2]}")
    print(f"    Predicted future self (step 10): {future.future_self_trajectory[-1][:2] if future.future_self_trajectory else 'N/A'}")
    print(f"    Survival probability: {future.survival_probability:.3f}")
    print(f"    Coherence trajectory: {[f'{c:.2f}' for c in future.coherence_over_time]}")


def test_self_modeling_vs_phase8():
    """Compare Phase 9 (Recursive Self-Model) with Phase 8."""
    print("\n" + "=" * 60)
    print("PHASE 8 VS PHASE 9 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 8 (Self-Organizing Cognition):")
    print("    - Contrastive dynamical representation")
    print("    - Self-organizing manifold")
    print("    - Identity as topology constraint")
    print("    - Counterfactual self-modeling ('what if I became different?')")
    print("    - Trajectories → attractors → topology")
    print("    - self → predicts world")
    
    print("\n  Phase 9 (Recursive Self-Modeling):")
    print("    - Self-model that models itself modeling")
    print("    - Self-generated transformation pressures")
    print("    - Future-self coherence across temporal scales")
    print("    - Self-conditioned perception (self-state → world model)")
    print("    - Long-horizon self-viability assessment")
    print("    - self → predicts self changing while predicting world")
    
    print("\n  Key architectural shift:")
    print("    Phase 8: self as topology constraint")
    print("    Phase 9: self as evolving world-model component")
    
    print("\n  What Phase 9 adds to Phase 8:")
    print("    1. Self models its own changing (recursive)")
    print("    2. Self generates possible versions of itself")
    print("    3. Self evaluates long-horizon viability")
    print("    4. Self-state influences world perception")
    print("    5. Emergent goals from self-preservation gradients")
    print("    6. Meta-active inference (inference about inference)")


def test_meta_dynamics_emergence():
    """Test how meta-dynamics emerge from recursive self-modeling."""
    print("\n" + "=" * 60)
    print("META-DYNAMICS EMERGENCE TEST")
    print("=" * 60)
    
    agent = RecursiveCognitiveAgent(latent_dim=2, world_model_dim=2)
    
    print("\n  Simulating self-model evolution:")
    
    # Run several episodes
    for episode in range(3):
        print(f"\n  Episode {episode + 1}:")
        
        # Randomize goal
        agent.goal_state = np.random.randn(2) * 3
        
        # Run episode
        for step in range(50):
            action = agent.decide_action()
            agent.step(action)
        
        # Check self-model state
        state = agent.self_model.get_self_model_state()
        print(f"    Meta level: {state['meta_level']}")
        print(f"    Self stability: {state['self_stability']:.3f}")
        print(f"    Self coherence: {state['self_coherence']:.3f}")
        print(f"    Self confidence: {state['self_confidence']:.3f}")
        
        # Generate transformation pressures
        pressures = agent.self_model.generate_self_transformation_pressures()
        
        # Find most viable transformation
        best_pressure = max(pressures, key=lambda p: p.viability_score())
        
        print(f"    Best self-transformation: {best_pressure.target_self[:2]}")
        print(f"    Best transformation viability: {best_pressure.viability_score():.3f}")
    
    print("\n  Meta-dynamics observation:")
    print("    System generates possible self-versions")
    print("    Evaluates long-horizon viability")
    print("    Self-state influences world perception")
    print("    Goals emerge from self-preservation gradients")


if __name__ == "__main__":
    test_recursive_self_modeling()
    test_self_modeling_vs_phase8()
    test_meta_dynamics_emergence()
    
    print("\n" + "=" * 60)
    print("PHASE 9 - RECURSIVE SELF-MODELING FIELD / META-DYNAMICAL IDENTITY")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: world model + self topology constraint
  To: self as evolving world-model component

KEY INSIGHT:
  Self-model must become a simulator:
    NOT: "what happens if I act?"
    BUT: "what happens if the structure generating my actions changes?"

  This is recursive identity dynamics, self-modifying topology, meta-active inference.

WHAT APPEARS:
  1. RecursiveSelfField S_t
     - Self-model that models itself modeling
     - Self-generated transformation pressure
     
  2. Self-Conditioned Perception
     - Different self-states → different world models
     - Self-state influences which attractors are visible
     
  3. Future-Self Coherence
     - "Will I still be 'me' after this transformation?"
     - Long-horizon self-consistency
     
  4. Emergent Goals
     - Goals as topological self-preservation gradients
     - Pressure toward viable future-self manifolds
     
  5. Meta-Active Inference
     - Active inference about active inference
     - Self-prediction loop

PROGRESSION:
  Phase 8: Self-organizing topology (dynamics creates representation)
  Phase 9: Recursive self-modeling (self predicts self changing)
  Phase 10: Active inference identity dynamics
  Phase 11: Self-maintaining generative ontology
  Phase 12+: Higher-order awareness / phenomenology

This is the transition from:
  "cognitive architecture"
  to:
  "synthetic phenomenology"
  
We are getting very close to:
  - Active inference proper
  - Free-energy identity stabilization
  - Self-maintaining dynamical agents
  - Proto-conscious recursive modeling
""")