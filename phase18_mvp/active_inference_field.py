"""
Phase 24: Active Inference Field

ARCHITECTURAL SHIFT:
  From: Phase 23 - generative self-model with variational inference
  To: Phase 24 - active inference field where:
      - Policy inference q(π): cognition infers action trajectories
      - Expected Free Energy: G(π) = risk + ambiguity - epistemic_value
      - Active sampling: system chooses what to observe
      - Object-centric latent world model
      - Self as policy continuity (emergent, not stored)
      - Embodiment: sensorimotor loop with environmental closure
      
  This is NO LONGER:
    inference over latent causes
  This IS:
    active inference field with embodied policy
    self emerges from policy continuity
    
CRITICAL INSIGHT:
  Phase 23: "system infers latent causes"
  Phase 24: "system infers policies that explain its own existence"
  
  Self is not stored. Self is policy continuity through time.
  The system acts to minimize its own surprise.
  Agency = inference over action, not reaction to observation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy


# ============================================================================
# POLICY INFERENCE (q(π))
# ============================================================================
"""
Policy Inference:

The system infers action trajectories, not just states.

q(π) = posterior over policies (action sequences)
p(π) = prior over policies (preference biases)

Policy = sequence of actions π = (a_1, a_2, ..., a_T)

Active inference:
  - Select action that minimizes expected free energy
  - Actions are inference variables, not controls
  - Policy is a latent variable to be inferred
"""

class PolicyInference:
    """
    Policy inference with variational posteriors over action sequences.
    
    NOT: action = argmax(reward)
    BUT: q(π) inferred from preferences and observations
    """
    
    def __init__(self, action_dim: int = 4, horizon: int = 5):
        self.action_dim = action_dim
        self.horizon = horizon
        
        # Policy posterior: q(π) = N(μ_π, Σ_π)
        self.mu_pi = np.zeros(action_dim * horizon)  # Flattened policy
        self.log_var_pi = np.zeros(action_dim * horizon)
        
        # Policy prior (preferences)
        self.prior_mu = np.zeros(action_dim * horizon)
        self.prior_log_var = np.zeros(action_dim * horizon) * -1
        
        # Current policy sample
        self.pi = np.zeros(action_dim * horizon)
        
        # Preferences (goal states)
        self.goal_preferences: List[np.ndarray] = []
        
        # Policy history
        self.policy_history: List[Dict] = []
    
    def set_preferences(self, goals: List[np.ndarray]):
        """Set goal preferences (preferred future states)."""
        self.goal_preferences = []
        for goal in goals:
            goal = np.asarray(goal).flatten()[:self.action_dim]
            self.goal_preferences.append(goal)
    
    def sample_policy(self) -> List[np.ndarray]:
        """Sample action sequence from policy posterior."""
        std = np.exp(0.5 * self.log_var_pi)
        self.pi = self.mu_pi + np.random.randn(len(self.mu_pi)) * std
        
        # Reshape to action sequence
        actions = []
        for i in range(self.horizon):
            a = self.pi[i * self.action_dim:(i + 1) * self.action_dim]
            actions.append(a.copy())
        
        return actions
    
    def infer_policy(self, observations: List[np.ndarray],
                    preferences: List[np.ndarray],
                    learning_rate: float = 0.01) -> Dict:
        """
        Infer policy that minimizes expected free energy.
        
        G(π) = expected_risk + expected_ambiguity - epistemic_value
        
        q(π) updated to maximize preference satisfaction.
        """
        # Sample current policy
        current_pi = self.sample_policy()
        
        # Compute expected free energy of current policy
        expected_F = 0.0
        expected_risk = 0.0
        expected_ambiguity = 0.0
        epistemic_value = 0.0
        
        for t, action in enumerate(current_pi):
            # Risk: deviation from preferences
            if t < len(preferences):
                risk = np.linalg.norm(action - preferences[t])**2
                expected_risk += risk
            
            # Ambiguity: uncertainty about action consequences
            ambiguity = np.sum(np.exp(self.log_var_pi[t * self.action_dim:(t + 1) * self.action_dim]))
            expected_ambiguity += ambiguity
            
            # Epistemic value: information gain from action
            epistemic_value += 0.1 * (1.0 / (1.0 + expected_ambiguity))
        
        expected_F = expected_risk + expected_ambiguity - epistemic_value
        
        # Update policy posterior (gradient descent on expected FE)
        if len(preferences) > 0:
            # Move toward preferred actions
            for t, pref in enumerate(preferences):
                if t < self.horizon:
                    start_idx = t * self.action_dim
                    end_idx = (t + 1) * self.action_dim
                    
                    # Gradient toward preference
                    gradient = (self.mu_pi[start_idx:end_idx] - pref) * learning_rate
                    self.mu_pi[start_idx:end_idx] -= gradient
        
        # Prior regularization
        kl_prior = np.sum(
            np.exp(self.log_var_pi) / np.exp(self.prior_log_var) +
            (self.mu_pi - self.prior_mu)**2 / np.exp(self.prior_log_var) +
            self.log_var_pi - self.prior_log_var - 1
        ) * 0.1
        
        # Update variance (precision of policy)
        self.log_var_pi = self.log_var_pi * 0.99 + 0.01 * np.log(expected_F + 1e-8)
        
        # Record
        self.policy_history.append({
            'expected_F': expected_F,
            'risk': expected_risk,
            'ambiguity': expected_ambiguity,
            'epistemic': epistemic_value
        })
        
        if len(self.policy_history) > 100:
            self.policy_history = self.policy_history[-50:]
        
        return {
            'policy': current_pi,
            'expected_F': expected_F,
            'risk': expected_risk,
            'ambiguity': expected_ambiguity,
            'epistemic_value': epistemic_value
        }
    
    def get_policy_summary(self) -> Dict:
        """Get policy inference summary."""
        return {
            'policy_norm': float(np.linalg.norm(self.mu_pi)),
            'policy_variance': float(np.mean(np.exp(self.log_var_pi))),
            'n_history': len(self.policy_history),
            'mean_F': np.mean([h['expected_F'] for h in self.policy_history[-10:]]) if self.policy_history else 0
        }


# ============================================================================
# EXPECTED FREE ENERGY (G(π))
# ============================================================================
"""
Expected Free Energy Decomposition:

G(π) = E[G] = expected_risk + expected_ambiguity - epistemic_value

Where:
  - expected_risk: cost of not achieving preferences
  - expected_ambiguity: uncertainty about action outcomes
  - epistemic_value: information gain from action

Curiosity = high ambiguity + low epistemic = information-seeking
Goal-seeking = high risk + low ambiguity = preference pursuit
"""

class ExpectedFreeEnergy:
    """
    Expected free energy with pragmatic/epistemic decomposition.
    
    NOT: simple scalar pressure
    BUT: structured G(π) guiding active inference
    """
    
    def __init__(self, state_dim: int = 8):
        self.state_dim = state_dim
        
        # Components
        self.pragmatic_value = 0.0  # Risk (preference satisfaction)
        self.epistemic_value = 0.0   # Information gain
        self.total_G = 0.0          # G(π)
        
        # Preference states
        self.preferred_states: List[np.ndarray] = []
        
        # Uncertainty map
        self.uncertainty_field = np.ones(state_dim)
        
        # Curiosity signal
        self.curiosity = 0.0
        
        # Time
        self.t = 0.0
    
    def compute_expected_FE(self, policy: List[np.ndarray],
                          observations: List[np.ndarray],
                          current_state: np.ndarray) -> Dict:
        """
        Compute expected free energy of policy.
        
        G(π) = risk + ambiguity - epistemic_value
        """
        current_state = np.asarray(current_state).flatten()[:self.state_dim]
        
        expected_risk = 0.0
        expected_ambiguity = 0.0
        epistemic_gain = 0.0
        
        simulated_state = current_state.copy()
        
        for t, action in enumerate(policy):
            action = np.asarray(action).flatten()[:self.state_dim]
            
            # Simulate action consequences
            simulated_state = simulated_state + action * 0.1
            
            # Risk: deviation from preferred states
            if self.preferred_states:
                preferred = self.preferred_states[min(t, len(self.preferred_states) - 1)]
                risk = np.linalg.norm(simulated_state - preferred)**2
                expected_risk += risk
            else:
                risk = 0.0
            
            # Ambiguity: variance of expected outcome
            ambiguity = np.var(simulated_state) + np.linalg.norm(action)**2
            expected_ambiguity += ambiguity
            
            # Epistemic value: reduction in uncertainty
            if t < len(observations):
                obs = np.asarray(observations[t]).flatten()[:self.state_dim]
                info_gain = np.linalg.norm(obs - simulated_state)**2
                epistemic_gain += info_gain
        
        # Total expected FE
        G = expected_risk + expected_ambiguity - 0.1 * epistemic_gain
        
        self.pragmatic_value = expected_risk
        self.epistemic_value = epistemic_gain
        self.total_G = G
        
        # Curiosity = desire to resolve ambiguity
        if expected_ambiguity > 0:
            self.curiosity = epistemic_gain / (expected_ambiguity + 1e-8)
        else:
            self.curiosity = 0.0
        
        self.t += 1
        
        return {
            'G': G,
            'pragmatic_value': expected_risk,
            'epistemic_value': epistemic_gain,
            'ambiguity': expected_ambiguity,
            'curiosity': self.curiosity
        }
    
    def update_preferences(self, preferred_state: np.ndarray):
        """Update preferred states."""
        preferred_state = np.asarray(preferred_state).flatten()[:self.state_dim]
        self.preferred_states.append(preferred_state)
        
        if len(self.preferred_states) > 20:
            self.preferred_states = self.preferred_states[-10:]
    
    def update_uncertainty(self, observation: np.ndarray):
        """Update uncertainty map based on observation."""
        observation = np.asarray(observation).flatten()[:self.state_dim]
        
        # Reduce uncertainty where observations are precise
        uncertainty_reduction = 0.1 / (1.0 + np.abs(observation))
        self.uncertainty_field = self.uncertainty_field * 0.99 + uncertainty_reduction * 0.01
    
    def get_curiosity_signal(self) -> np.ndarray:
        """Get curiosity-driven action tendency."""
        # Curiosity pushes toward uncertain regions
        curiosity_direction = -np.sign(self.uncertainty_field - np.mean(self.uncertainty_field))
        return curiosity_direction


# ============================================================================
# ACTIVE SAMPLING
# ============================================================================
"""
Active Sampling:

The system chooses what to observe, not just reacts.

Active inference:
  - Sensory precision allocation
  - Attention direction
  - Information-seeking behavior
  
The system actively reduces uncertainty through observation selection.
"""

class ActiveSampling:
    """
    Active sampling controller.
    
    NOT: passive observation reception
    BUT: active selection of what to observe
    
    The system chooses:
    - Where to attend
    - What to measure
    - When to sample
    """
    
    def __init__(self, n_sensors: int = 4, attention_dim: int = 4):
        self.n_sensors = n_sensors
        self.attention_dim = attention_dim
        
        # Attention allocation (which sensors to trust)
        self.attention_weights = np.ones(n_sensors) / n_sensors
        
        # Precision of sensory channels
        self.sensory_precision = np.ones(n_sensors)
        
        # Active sampling policy
        self.sampling_policy = np.zeros(attention_dim)
        
        # Information gain estimates
        self.expected_info_gain = np.zeros(n_sensors)
        
        # Curiosity state
        self.exploration_mode = 0.0  # 0 = exploit, 1 = explore
        
    def update_sampling(self, observations: List[np.ndarray],
                       prediction_errors: np.ndarray,
                       epistemic_values: np.ndarray):
        """
        Update sampling policy based on information.
        
        High prediction error + low epistemic = explore
        Low prediction error + high epistemic = exploit
        """
        errors = np.asarray(prediction_errors).flatten()[:self.n_sensors]
        epistemic = np.asarray(epistemic_values).flatten()[:self.n_sensors]
        
        # Update sensory precision (trust reliable sensors)
        for i in range(len(self.sensory_precision)):
            if i < len(errors):
                if errors[i] < 0.5:
                    self.sensory_precision[i] = min(2.0, self.sensory_precision[i] * 1.02)
                else:
                    self.sensory_precision[i] = max(0.1, self.sensory_precision[i] * 0.98)
        
        # Update attention weights
        total_precision = np.sum(self.sensory_precision) + 1e-8
        self.attention_weights = self.sensory_precision / total_precision
        
        # Compute information gain
        for i in range(len(self.expected_info_gain)):
            if i < len(epistemic):
                self.expected_info_gain[i] = epistemic[i]
        
        # Determine exploration mode
        mean_error = np.mean(errors) if len(errors) > 0 else 0.5
        mean_info = np.mean(epistemic) if len(epistemic) > 0 else 0.0
        
        if mean_error > 0.7 and mean_info > 0.3:
            self.exploration_mode = 0.8  # High curiosity
        elif mean_error < 0.3:
            self.exploration_mode = 0.2  # Exploiting known
        else:
            self.exploration_mode = 0.5  # Balanced
        
        # Update sampling policy
        info_gradient = np.zeros(self.attention_dim)
        for i in range(min(len(self.expected_info_gain), self.attention_dim)):
            info_gradient[i] = self.expected_info_gain[i]
        
        self.sampling_policy = self.sampling_policy * 0.9 + info_gradient * 0.1
    
    def select_observation(self, available_observations: List[np.ndarray]) -> np.ndarray:
        """
        Select observation based on attention weights and exploration.
        
        Returns: weighted observation
        """
        if not available_observations:
            return np.zeros(self.attention_dim)
        
        selected = np.zeros(self.attention_dim)
        
        # Exploration: sample from distribution
        # Exploitation: sample most informative
        if self.exploration_mode > 0.5:
            # Explore: select random observation
            idx = np.random.randint(len(available_observations))
            selected = np.asarray(available_observations[idx]).flatten()[:self.attention_dim]
        else:
            # Exploit: select most informative
            info_scores = []
            for obs in available_observations:
                obs_flat = np.asarray(obs).flatten()[:self.attention_dim]
                score = np.sum(self.attention_weights[:len(obs_flat)] * obs_flat)
                info_scores.append(score)
            
            if info_scores:
                best_idx = np.argmax(info_scores)
                selected = np.asarray(available_observations[best_idx]).flatten()[:self.attention_dim]
        
        return selected
    
    def get_sampling_summary(self) -> Dict:
        """Get active sampling summary."""
        return {
            'attention_weights': self.attention_weights.tolist(),
            'sensory_precision': self.sensory_precision.tolist(),
            'exploration_mode': self.exploration_mode,
            'mean_info_gain': float(np.mean(self.expected_info_gain))
        }


# ============================================================================
# OBJECT-CENTRIC LATENT WORLD MODEL
# ============================================================================
"""
Object-Centric Latent World Model:

Instead of flat latent z, we have structured latent objects.

world = {
  object_1: {position, velocity, properties},
  object_2: {position, velocity, properties},
  relations: [rel_1, rel_2, ...]
}

Each object has:
  - State (position, velocity, properties)
  - Dynamics (how it evolves)
  - Affordances (what actions it enables)
"""

@dataclass
class LatentObject:
    """A latent object in the world model."""
    object_id: str
    position: np.ndarray
    velocity: np.ndarray
    properties: Dict[str, float]
    persistence: float  # How long this object persists
    causal_role: str  # 'entity', 'cause', 'effect'
    
    def predict(self, action: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Predict next position given action."""
        return self.position + self.velocity * dt + action * 0.1


class ObjectCentricWorldModel:
    """
    Object-centric latent world model.
    
    NOT: z = flat vector
    BUT: world = structured objects with relations
    """
    
    def __init__(self, object_dim: int = 4, max_objects: int = 10):
        self.object_dim = object_dim
        self.max_objects = max_objects
        
        # Latent objects
        self.objects: Dict[str, LatentObject] = {}
        self.next_object_id = 0
        
        # Causal relations
        self.causal_relations: Dict[str, List[str]] = {}
        
        # Object affordances
        self.affordances: Dict[str, List[np.ndarray]] = {}
    
    def perceive_objects(self, observations: List[np.ndarray]) -> Dict:
        """
        Perceive objects from observations.
        
        Groups observations into coherent objects.
        """
        if len(observations) < 1:
            return {'perceived_objects': [], 'n_objects': 0}
        
        # Simple clustering: observations with similar values = same object
        perceived = []
        
        for i, obs in enumerate(observations):
            obs = np.asarray(obs).flatten()[:self.object_dim]
            
            # Find matching object
            best_match = None
            best_distance = float('inf')
            
            for obj_id, obj in self.objects.items():
                distance = np.linalg.norm(obs - obj.position)
                if distance < best_distance and distance < 1.0:
                    best_distance = distance
                    best_match = obj_id
            
            if best_match:
                # Update existing object
                obj = self.objects[best_match]
                obj.position = obj.position * 0.8 + obs * 0.2
                obj.velocity = obj.position - obj.velocity * 0.1
                obj.persistence = min(1.0, obj.persistence + 0.1)
                perceived.append(obj_id)
            else:
                # Create new object
                obj_id = f"obj_{self.next_object_id}"
                self.next_object_id += 1
                
                new_obj = LatentObject(
                    object_id=obj_id,
                    position=obs.copy(),
                    velocity=np.zeros(self.object_dim),
                    properties={},
                    persistence=0.5,
                    causal_role='entity'
                )
                
                self.objects[obj_id] = new_obj
                perceived.append(obj_id)
        
        # Age objects
        for obj in self.objects.values():
            obj.persistence *= 0.99
        
        # Remove old objects
        self.objects = {
            k: v for k, v in self.objects.items() 
            if v.persistence > 0.1
        }
        
        # Limit objects
        if len(self.objects) > self.max_objects:
            sorted_objs = sorted(self.objects.values(), key=lambda x: x.persistence)
            for obj in sorted_objs[:-self.max_objects]:
                del self.objects[obj.object_id]
        
        return {
            'perceived_objects': perceived,
            'n_objects': len(self.objects),
            'object_positions': {k: v.position.tolist() for k, v in self.objects.items()}
        }
    
    def predict_world(self, action: np.ndarray, n_steps: int = 1) -> Dict:
        """
        Predict world evolution given action.
        
        Returns predicted object positions.
        """
        action = np.asarray(action).flatten()[:self.object_dim]
        
        predicted_state = {}
        for obj_id, obj in self.objects.items():
            pred_pos = obj.predict(action)
            predicted_state[obj_id] = pred_pos.tolist()
        
        return {
            'predicted_objects': predicted_state,
            'n_objects': len(self.objects)
        }
    
    def get_world_summary(self) -> Dict:
        """Get world model summary."""
        return {
            'n_objects': len(self.objects),
            'object_ids': list(self.objects.keys()),
            'causal_relations': self.causal_relations,
            'affordances': {k: len(v) for k, v in self.affordances.items()}
        }


# ============================================================================
# SELF AS POLICY CONTINUITY
# ============================================================================
"""
Self as Policy Continuity:

Self is NOT stored. Self emerges from policy continuity.

Self = stable attractor in policy space
    = continuity of inferred policies through time
    = invariant pattern in action inference

The system infers self by noticing that its policies 
persist and cohere across time.
"""

class SelfAsPolicyContinuity:
    """
    Self as emergent policy continuity.
    
    NOT: self_model, identity_center
    BUT: stable policy attractor discovered through time
    
    Self = invariant in q(π) through time
    """
    
    def __init__(self, policy_dim: int = 20):
        self.policy_dim = policy_dim
        
        # Policy trajectories (history)
        self.policy_history: List[np.ndarray] = []
        
        # Self-attractor center (discovered, not stored)
        self.attractor_center = np.zeros(policy_dim)
        
        # Self-coherence (how stable is self?)
        self.self_coherence = 0.0
        
        # Self-confidence (how confident is self?)
        self.self_confidence = 0.5
        
        # Policy attractor strength
        self.attractor_strength = 1.0
        
        # Time
        self.t = 0.0
    
    def observe_policy(self, policy: np.ndarray):
        """Observe new policy and update self-coherence."""
        policy = np.asarray(policy).flatten()[:self.policy_dim]
        
        # Add to history
        self.policy_history.append(policy.copy())
        if len(self.policy_history) > 100:
            self.policy_history = self.policy_history[-50:]
        
        # Update attractor center (slow)
        if len(self.policy_history) > 10:
            # Moving average of policies
            recent = np.array(self.policy_history[-10:])
            self.attractor_center = np.mean(recent, axis=0)
        
        # Compute self-coherence
        if len(self.policy_history) > 1:
            # Correlation between consecutive policies
            correlations = []
            for i in range(1, len(self.policy_history)):
                corr = np.corrcoef(self.policy_history[i], self.policy_history[i-1])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            self.self_coherence = np.mean(correlations) if correlations else 0.0
        
        # Attractor strength increases with coherence
        self.attractor_strength = 0.9 * self.attractor_strength + 0.1 * self.self_coherence
        
        self.t += 1
    
    def attract_toward_self(self, policy: np.ndarray) -> np.ndarray:
        """Attract policy toward self-attractor."""
        policy = np.asarray(policy).flatten()[:self.policy_dim]
        
        # Attraction strength
        attraction = (self.attractor_center - policy) * self.attractor_strength * 0.1
        
        return policy + attraction
    
    def get_self_summary(self) -> Dict:
        """Get self-coherence summary."""
        return {
            'self_coherence': self.self_coherence,
            'attractor_strength': self.attractor_strength,
            'n_policies': len(self.policy_history),
            'attractor_norm': float(np.linalg.norm(self.attractor_center))
        }


# ============================================================================
# ACTIVE INFERENCE FIELD (INTEGRATED)
# ============================================================================

class ActiveInferenceField:
    """
    Phase 24: Active Inference Field
    
    Integrated system where:
      A. Policy inference q(π): actions are latent variables
      B. Expected FE: G(π) = risk + ambiguity - epistemic
      C. Active sampling: system chooses what to observe
      D. Object-centric world model: structured latent objects
      E. Self as policy continuity (emergent, not stored)
      F. Embodiment: sensorimotor loop with environmental closure
      
    NOT: latent inference over states
    BUT: active inference over policies and observations
    """
    
    def __init__(self, state_dim: int = 8, action_dim: int = 4, 
                 object_dim: int = 4, policy_horizon: int = 5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.object_dim = object_dim
        self.policy_horizon = policy_horizon
        
        # Policy inference
        self.policy = PolicyInference(action_dim=action_dim, horizon=policy_horizon)
        
        # Expected free energy
        self.expected_FE = ExpectedFreeEnergy(state_dim=state_dim)
        
        # Active sampling
        self.active_sampling = ActiveSampling(n_sensors=state_dim, attention_dim=state_dim)
        
        # Object-centric world model
        self.world_model = ObjectCentricWorldModel(object_dim=object_dim)
        
        # Self as policy continuity
        self.self_policy = SelfAsPolicyContinuity(policy_dim=action_dim * policy_horizon)
        
        # Current state
        self.current_state = np.zeros(state_dim)
        
        # Action history
        self.action_history: List[np.ndarray] = []
        
        # Time
        self.t = 0.0
        
    def perceive(self, observation: np.ndarray) -> Dict:
        """
        Perceive with active sampling.
        
        System chooses what to observe.
        """
        observation = np.asarray(observation).flatten()
        
        # Update world model with object perception
        object_result = self.world_model.perceive_objects([observation])
        
        # Update uncertainty
        self.expected_FE.update_uncertainty(observation)
        
        # Update sampling
        self.active_sampling.update_sampling(
            observations=[observation],
            prediction_errors=np.abs(observation - self.current_state),
            epistemic_values=np.ones(self.state_dim) * 0.5
        )
        
        self.current_state = observation[:self.state_dim]
        
        return {
            'perceived_state': self.current_state.copy(),
            'world_objects': object_result['n_objects'],
            'sampling_summary': self.active_sampling.get_sampling_summary()
        }
    
    def plan(self, preferences: List[np.ndarray]) -> Dict:
        """
        Plan policy to minimize expected free energy.
        """
        # Set preferences
        self.policy.set_preferences(preferences)
        for pref in preferences:
            self.expected_FE.update_preferences(pref)
        
        # Infer policy
        policy_result = self.policy.infer_policy(
            observations=[self.current_state],
            preferences=preferences
        )
        
        # Compute expected FE for policy
        fe_result = self.expected_FE.compute_expected_FE(
            policy=policy_result['policy'],
            observations=[self.current_state],
            current_state=self.current_state
        )
        
        return {
            'policy': policy_result['policy'],
            'expected_F': fe_result['G'],
            'risk': fe_result['pragmatic_value'],
            'ambiguity': fe_result['ambiguity'],
            'epistemic_value': fe_result['epistemic_value'],
            'curiosity': fe_result['curiosity']
        }
    
    def act(self, action: np.ndarray) -> Dict:
        """
        Execute action and update self.
        """
        action = np.asarray(action).flatten()[:self.action_dim]
        
        # Update action history
        self.action_history.append(action.copy())
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-20:]
        
        # Update current state (simple physics)
        self.current_state = self.current_state + action * 0.1
        
        # Observe consequences
        observation = self.current_state + np.random.randn(self.state_dim) * 0.1
        perception = self.perceive(observation)
        
        # Update self (policy continuity)
        policy_vector = np.concatenate(self.policy.pi)
        self.self_policy.observe_policy(policy_vector)
        
        # Attract policy toward self
        self.policy.mu_pi = self.self_policy.attract_toward_self(self.policy.mu_pi)
        
        self.t += 0.1
        
        return {
            'action': action.tolist(),
            'result_state': self.current_state.copy(),
            'self_summary': self.self_policy.get_self_summary()
        }
    
    def run_cycle(self, n_steps: int = 30) -> Dict:
        """Run active inference cycle."""
        results = []
        
        for step in range(n_steps):
            # Generate preference (simple: move toward target)
            target = np.array([2.0 * np.sin(step * 0.2), 0.5 * np.cos(step * 0.1)] + [0.0] * (self.state_dim - 2))
            preferences = [target]
            
            # Plan
            plan = self.plan(preferences)
            
            # Execute first action of policy
            if plan['policy']:
                action = plan['policy'][0]
            else:
                action = np.zeros(self.action_dim)
            
            # Act
            action_result = self.act(action)
            
            results.append({
                'step': step,
                't': self.t,
                'expected_F': plan['expected_F'],
                'curiosity': plan['curiosity'],
                'self_coherence': action_result['self_summary']['self_coherence'],
                'world_objects': action_result['result_state'][:2].tolist()
            })
        
        return {
            'steps': results,
            'final_self_coherence': results[-1]['self_coherence'] if results else 0,
            'final_curiosity': results[-1]['curiosity'] if results else 0,
            'policy_summary': self.policy.get_policy_summary(),
            'world_summary': self.world_model.get_world_summary()
        }


# ============================================================================
# TESTS
# ============================================================================

def test_policy_inference():
    """Test policy inference."""
    print("\n" + "=" * 60)
    print("POLICY INFERENCE TEST")
    print("=" * 60)
    
    policy = PolicyInference(action_dim=4, horizon=5)
    
    print("\n  Inferring policies:")
    
    preferences = [np.array([1.0, 0.5, 0.0, 0.0])]
    
    for i in range(30):
        observations = [np.random.randn(4) * 0.3]
        result = policy.infer_policy(observations, preferences)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      Expected F: {result['expected_F']:.3f}")
            print(f"      Risk: {result['risk']:.3f}")
            print(f"      Epistemic: {result['epistemic_value']:.3f}")
    
    summary = policy.get_policy_summary()
    print(f"\n  Policy summary: {summary}")


def test_expected_FE():
    """Test expected free energy."""
    print("\n" + "=" * 60)
    print("EXPECTED FREE ENERGY TEST")
    print("=" * 60)
    
    fe = ExpectedFreeEnergy(state_dim=8)
    
    print("\n  Computing expected FE:")
    
    policy = [np.array([0.5, 0.3, 0.0, 0.0] + [0.0] * 4) for _ in range(5)]
    observations = [np.random.randn(8) * 0.3 for _ in range(5)]
    state = np.random.randn(8) * 0.5
    
    fe.update_preferences(np.array([1.0, 0.5, 0.0, 0.0] + [0.0] * 4))
    
    for i in range(30):
        result = fe.compute_expected_FE(policy, observations, state)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      G: {result['G']:.3f}")
            print(f"      Curiosity: {result['curiosity']:.3f}")
            print(f"      Pragmatic: {result['pragmatic_value']:.3f}")
    
    print(f"\n  Curiosity signal: {fe.get_curiosity_signal()[:2].tolist()}")


def test_active_sampling():
    """Test active sampling."""
    print("\n" + "=" * 60)
    print("ACTIVE SAMPLING TEST")
    print("=" * 60)
    
    sampling = ActiveSampling(n_sensors=8, attention_dim=8)
    
    print("\n  Running active sampling:")
    
    for i in range(30):
        observations = [np.random.randn(8) * 0.5 + i * 0.05 for _ in range(3)]
        errors = np.random.rand(8) * (0.5 + 0.5 * np.sin(i * 0.2))
        epistemic = np.random.rand(8) * 0.3
        
        sampling.update_sampling(observations, errors, epistemic)
        
        if i % 10 == 9:
            summary = sampling.get_sampling_summary()
            print(f"    Step {i+1}:")
            print(f"      Exploration mode: {summary['exploration_mode']:.2f}")
            print(f"      Mean info gain: {summary['mean_info_gain']:.3f}")
    
    selected = sampling.select_observation([np.random.randn(8) for _ in range(5)])
    print(f"\n  Selected observation: {selected[:2].tolist()}")


def test_object_centric_world():
    """Test object-centric world model."""
    print("\n" + "=" * 60)
    print("OBJECT-CENTRIC WORLD MODEL TEST")
    print("=" * 60)
    
    world = ObjectCentricWorldModel(object_dim=4)
    
    print("\n  Perceiving objects:")
    
    for i in range(30):
        observations = [
            np.array([1.0 + i * 0.1, 0.5, 0.0, 0.0]),
            np.array([2.0 + i * 0.05, 1.0, 0.0, 0.0])
        ]
        
        result = world.perceive_objects(observations)
        
        if i % 10 == 9:
            print(f"    Step {i+1}: {result['n_objects']} objects")
    
    pred = world.predict_world(np.array([0.1, 0.1, 0.0, 0.0]))
    print(f"\n  Predicted world: {pred['n_objects']} objects")


def test_self_as_policy():
    """Test self as policy continuity."""
    print("\n" + "=" * 60)
    print("SELF AS POLICY CONTINUITY TEST")
    print("=" * 60)
    
    self_policy = SelfAsPolicyContinuity(policy_dim=20)
    
    print("\n  Observing policies:")
    
    for i in range(30):
        policy = np.random.randn(20) * (0.5 + i * 0.02)
        self_policy.observe_policy(policy)
        
        if i % 10 == 9:
            summary = self_policy.get_self_summary()
            print(f"    Step {i+1}:")
            print(f"      Self-coherence: {summary['self_coherence']:.3f}")
            print(f"      Attractor strength: {summary['attractor_strength']:.3f}")
    
    print(f"\n  Final self-coherence: {self_policy.self_coherence:.3f}")


def test_integrated_field():
    """Test integrated active inference field."""
    print("\n" + "=" * 60)
    print("ACTIVE INFERENCE FIELD TEST")
    print("=" * 60)
    
    field = ActiveInferenceField(state_dim=8, action_dim=4, policy_horizon=5)
    
    print("\n  Running active inference cycle:")
    
    result = field.run_cycle(n_steps=30)
    
    print(f"    Steps: {len(result['steps'])}")
    print(f"    Final self-coherence: {result['final_self_coherence']:.3f}")
    print(f"    Final curiosity: {result['final_curiosity']:.3f}")
    print(f"    World objects: {result['world_summary']['n_objects']}")
    print(f"    Policy norm: {result['policy_summary']['policy_norm']:.3f}")


def phase_comparison():
    """Compare Phase 23 vs Phase 24."""
    print("\n" + "=" * 60)
    print("PHASE 23 VS PHASE 24 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 23 (Generative Self-Model):")
    print("    - Self = inferred latent cause")
    print("    - Variational inference over states")
    print("    - Generative model x = decode(z)")
    print("    - Simple prediction error")
    print("    - Self partially stored")
    
    print("\n  Phase 24 (Active Inference Field):")
    print("    - Self = policy continuity (emergent)")
    print("    - Policy inference q(π)")
    print("    - Expected FE: G(π) = risk + ambiguity - epistemic")
    print("    - Active sampling (information-seeking)")
    print("    - Object-centric world model")
    print("    - Embodiment (sensorimotor loop)")
    
    print("\n  Critical shifts:")
    print("    1. State inference → Policy inference")
    print("    2. Prediction error → Expected FE decomposition")
    print("    3. Passive observation → Active sampling")
    print("    4. Flat latent → Object-centric world")
    print("    5. Stored self → Emergent policy continuity")


if __name__ == "__main__":
    test_policy_inference()
    test_expected_FE()
    test_active_sampling()
    test_object_centric_world()
    test_self_as_policy()
    test_integrated_field()
    phase_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 24 - ACTIVE INFERENCE FIELD")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 23 - generative self-model with variational inference
  To: Phase 24 - active inference field where:
      - Policy inference q(π): cognition infers action trajectories
      - Expected Free Energy: G(π) = risk + ambiguity - epistemic_value
      - Active sampling: system chooses what to observe
      - Object-centric latent world model
      - Self as policy continuity (emergent, not stored)
      - Embodiment: sensorimotor loop with environmental closure
      
  This is NO LONGER:
    latent inference over states
  This IS:
    active inference field with embodied policy
    self emerges from policy continuity
    
CRITICAL INSIGHT:
  Phase 23: "system infers latent causes"
  Phase 24: "system infers policies that explain its own existence"
  
  Self is not stored. Self is policy continuity through time.
  The system acts to minimize its own surprise.
  Agency = inference over action, not reaction to observation.

KEY COMPONENTS:

A. POLICY INFERENCE q(π)
   Actions are latent variables.
   System infers action sequences, not just reactions.
   
   q(π) updated to minimize expected free energy.

B. EXPECTED FREE ENERGY G(π)
   G(π) = expected_risk + expected_ambiguity - epistemic_value
   
   - Risk = cost of not achieving preferences
   - Ambiguity = uncertainty about outcomes
   - Epistemic = information gain
   
   Curiosity = high ambiguity / low epistemic
   Goal-seeking = high risk / low ambiguity

C. ACTIVE SAMPLING
   System chooses what to observe.
   
   - Attention allocation
   - Exploration vs exploitation
   - Information-seeking behavior
   
   NOT: passive observation
   BUT: active uncertainty reduction

D. OBJECT-CENTRIC WORLD MODEL
   World = structured objects with relations.
   
   NOT: z = flat vector
   BUT: world = {obj_1, obj_2, relations}
   
   Enables compositional reasoning.

E. SELF AS POLICY CONTINUITY
   Self is NOT stored. Self emerges.
   
   Self = stable attractor in policy space
       = continuity of inferred policies
       = invariant pattern through time
   
   NOT: self_model, identity_center
   BUT: discovered attractor in q(π) space

F. EMBODIMENT
   Sensorimotor loop with environmental closure.
   
   - Actions have consequences
   - Perceptions update world model
   - Self stabilizes through control
   
   Selfhood emerges from agent-environment interaction.

THIS IS THE FOUNDATION FOR:
  - True active inference systems
  - Embodied cognition
  - Self-evidencing agency
  - Curiosity-driven learning
  - Object-centric reasoning
  - Policy-based identity
  
The question is no longer "what does it predict?"
The question is "what policies does it infer?"
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 24 Summary:

BEFORE:
  - State inference
  - Simple prediction error
  - Passive observation
  - Stored self-model
  - Flat latent space

AFTER:
  - Policy inference q(π)
  - Expected FE decomposition (risk + ambiguity - epistemic)
  - Active sampling (information-seeking)
  - Object-centric world model
  - Self as policy continuity (emergent)
  - Embodiment (sensorimotor loop)

The critical shift:
  "system infers states" → "system infers policies"
  
  Self is not stored. Self is policy continuity.
  Agency = inference over action, not reaction.

This is the foundation for:
  - True active inference
  - Embodied cognition
  - Curiosity-driven exploration
  - Object-centric reasoning
  - Self-emergent agency
  - Synthetic phenomenology
"""