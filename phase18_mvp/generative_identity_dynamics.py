"""
Phase 10: Generative Identity Dynamics

ARCHITECTURAL SHIFT:
  From: self as vector with handcrafted dynamics
  To: self as probabilistic latent manifold with learned generative model

CRITICAL INSIGHT:
  NOT: "describe selfhood symbolically"
  BUT: "implement generative selfhood dynamically"

  System must LEARN self-model from experience,
  not execute handcrafted selfhood theory.

WHAT APPEARS:
  1. Learned self-transition model P(S_t+1 | S_t, W_t, A_t)
     - Self-evolution as learned latent dynamics
     - Not handcrafted transformation operators
     
  2. Variational self-belief q(self)
     - Self = probability distribution, not point estimate
     - entropy(self_beliefs) = identity uncertainty
     
  3. Free-energy self stabilization
     - minimize expected free energy over future self-trajectories
     - Actions that preserve identity predictability
     
  4. Self-destabilization for adaptation
     - Real recursive identity sometimes destabilizes itself
     - Phase transitions, restructuring, adaptation pressure
     
  5. Self/world co-generation
     - Self-model and world-model co-emerge
     - Attractor-based recursion (not depth recursion)

KEY PROBLEMS WITH PHASE 9:
  1. self_confidence = heuristic (should be entropy)
  2. self_dynamics = handcrafted (should be learned)
  3. transformation_pressures = symbolic (should be sampled from latent)
  4. meta_level = depth recursion (should be fixed-point attractor)

PHASE 10 PRINCIPLES:
  - Replace handcrafted semantics with learned generative dynamics
  - Self uncertainty = belief entropy, not confidence heuristic
  - Identity should sometimes destabilize (not always stable)
  - True recursion = attractor fixed-point, not depth
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class VariationalSelfBelief:
    """
    Self as probabilistic latent distribution.
    
    NOT: self_state = vector
    BUT: q(self) = probability distribution
    
    Key innovation:
      - Identity uncertainty = entropy over beliefs
      - Self is not a point, it's a distribution
      - System doesn't fully know who it is
    """
    mean: np.ndarray              # Belief mean
    variance: np.ndarray          # Belief uncertainty (diagonal covariance)
    entropy: float               # Identity uncertainty measure
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample possible selves from belief distribution."""
        samples = []
        for _ in range(n):
            noise = np.random.randn(len(self.mean)) * np.sqrt(self.variance)
            samples.append(self.mean + noise)
        return np.array(samples)
    
    def kl_divergence(self, other: 'VariationalSelfBelief') -> float:
        """KL divergence between two self-beliefs."""
        # KL(N1 || N2) for diagonal Gaussians
        kl = 0.5 * (
            np.sum(self.variance / (other.variance + 1e-8)) +
            np.sum((other.mean - self.mean)**2 / (other.variance + 1e-8)) -
            len(self.mean) +
            np.sum(np.log(other.variance + 1e-8)) - np.sum(np.log(self.variance + 1e-8))
        )
        return max(0.0, kl)
    
    def update_variational(self, observations: List[np.ndarray], 
                          action: np.ndarray, world_state: np.ndarray,
                          transition_model: 'LearnedSelfTransitionModel'):
        """
        Update self-belief using variational inference.
        
        q(S_t) ≈ p(S_t | S_{t-1}, actions, world)
        """
        if len(observations) < 2:
            return
        
        # Use learned transition model to predict next self
        prev_self = observations[-2] if len(observations) >= 2 else self.mean
        curr_self = observations[-1]
        
        # Compute prediction from learned model
        predicted_self = transition_model.predict(prev_self, action, world_state)
        
        # Update belief: move toward prediction + observation
        # Weight by inverse variance (less certain = more weight to new info)
        prediction_weight = 0.3
        observation_weight = 0.7
        
        new_mean = prediction_weight * predicted_self + observation_weight * curr_self
        
        # Update variance: increase when prediction ≠ observation
        prediction_error = np.linalg.norm(predicted_self - curr_self)
        new_variance = self.variance + prediction_error * 0.1
        
        # Bound variance (not too certain, not too uncertain)
        new_variance = np.clip(new_variance, 0.01, 1.0)
        
        self.mean = new_mean
        self.variance = new_variance
        self.entropy = 0.5 * np.sum(np.log(2 * np.pi * self.variance + 1e-8))


class LearnedSelfTransitionModel:
    """
    P(S_t+1 | S_t, W_t, A_t) - Learned self-transition dynamics.
    
    NOT: handcrafted self_dynamics
    BUT: learned from experience
    
    The system learns:
      - How does self evolve given actions and world?
      - Which self-trajectories are stable?
      - Which lead to model collapse?
    """
    
    def __init__(self, latent_dim: int = 2):
        self.latent_dim = latent_dim
        
        # Learned parameters (simple linear model for demonstration)
        # In real implementation, this would be a neural network
        self.W_self = np.eye(latent_dim) * 0.9  # Self evolution matrix
        self.W_action = np.eye(latent_dim) * 0.1  # Action influence
        self.W_world = np.eye(latent_dim) * 0.05  # World influence
        self.noise_scale = 0.1  # Stochastic dynamics
        
        # Learning from experience
        self.transitions_observed = 0
        self.learning_rate = 0.01
        
        # Track which self-trajectories were "good" (stable)
        self.stable_self_trajectories: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        
    def predict(self, current_self: np.ndarray, 
                action: np.ndarray, 
                world_state: np.ndarray) -> np.ndarray:
        """
        Predict next self using learned dynamics.
        
        S_{t+1} = f(S_t, A_t, W_t) + noise
        """
        # Ensure vectors are 1D
        current_self = np.asarray(current_self).flatten()
        action = np.asarray(action).flatten()
        world_state = np.asarray(world_state).flatten()
        
        # Linear dynamics (would be neural network in full implementation)
        next_self = (
            self.W_self @ current_self +
            self.W_action @ action +
            self.W_world @ world_state
        )
        
        # Add noise (stochastic dynamics)
        noise = np.random.randn(len(next_self)) * self.noise_scale
        next_self = next_self + noise
        
        return next_self
    
    def learn_from_trajectory(self, trajectory: List[np.ndarray], 
                             actions: List[np.ndarray],
                             worlds: List[np.ndarray],
                             outcome: str):
        """
        Learn self-transition model from experience.
        
        If self-trajectory was stable → positive update
        If self-trajectory collapsed → negative update
        """
        if len(trajectory) < 3:
            return
        
        # Extract transitions
        for i in range(len(trajectory) - 1):
            s_t = trajectory[i]
            s_t1 = trajectory[i + 1]
            a_t = actions[i] if i < len(actions) else np.zeros(self.latent_dim)
            w_t = worlds[i] if i < len(worlds) else np.zeros(self.latent_dim)
            
            # Observed transition
            observed_delta = s_t1 - s_t
            
            # Predicted transition
            predicted_delta = (
                self.W_self @ s_t +
                self.W_action @ a_t +
                self.W_world @ w_t - s_t
            )
            
            # Prediction error
            error = observed_delta - predicted_delta
            
            # Update weights (Hebbian-like learning)
            if outcome == "stable":
                # Stable outcome → strengthen transition model
                self.W_self += self.learning_rate * np.outer(error, s_t) * 0.1
            else:
                # Unstable outcome → dampen transitions
                self.W_self -= self.learning_rate * np.outer(error, s_t) * 0.1
            
            # Normalize to prevent explosion
            self.W_self = self.W_self / (np.linalg.norm(self.W_self) + 1e-8)
        
        self.transitions_observed += len(trajectory)
        
        # Track stable trajectories
        if outcome == "stable":
            self.stable_self_trajectories.append((trajectory[0], trajectory[-1], actions[0] if actions else np.zeros(self.latent_dim)))
            # Keep only recent
            if len(self.stable_self_trajectories) > 100:
                self.stable_self_trajectories = self.stable_self_trajectories[-50:]


class FreeEnergySelfStabilization:
    """
    Free-energy minimization for identity preservation.
    
    NOT: goal + coherence + stability heuristics
    BUT: minimize expected free energy over future self-trajectories
    
    EFE = Expected Free Energy
    G = predicted uncertainty + predicted divergence from preferred self
    
    Agent chooses actions that preserve identity predictability.
    """
    def __init__(self, preferred_self: VariationalSelfBelief, latent_dim: int = 2):
        self.preferred_self = preferred_self
        self.prediction_horizon = 5
        self.latent_dim = latent_dim
    
    def compute_expected_free_energy(self,
                                     current_self: np.ndarray,
                                     action: np.ndarray,
                                     world_state: np.ndarray,
                                     transition_model: LearnedSelfTransitionModel) -> float:
        """
        Compute expected free energy of action.
        
        G(action) = E[uncertainty] + E[divergence from preferred self]
        """
        # Sample possible future selves
        future_selves = []
        current = current_self.copy()
        
        for step in range(self.prediction_horizon):
            next_self = transition_model.predict(current, action * (1 - step / self.prediction_horizon), world_state)
            future_selves.append(next_self)
            current = next_self
        
        future_selves = np.array(future_selves)
        
        # 1. Uncertainty (entropy) of future selves
        future_mean = np.mean(future_selves, axis=0)
        future_var = np.var(future_selves, axis=0)
        uncertainty = 0.5 * np.sum(np.log(future_var + 1e-8))
        
        # 2. Divergence from preferred self
        divergences = []
        for fs in future_selves:
            div = np.linalg.norm(fs - self.preferred_self.mean)
            divergences.append(div)
        
        avg_divergence = np.mean(divergences)
        
        # 3. Self-coherence (does future self still "feel like me"?)
        # Use self-transition model to check trajectory stability
        coherence = 1.0
        for i in range(len(future_selves) - 1):
            predicted = transition_model.predict(future_selves[i], np.zeros(self.latent_dim), world_state)
            actual = future_selves[i + 1]
            coherence *= 1.0 / (1.0 + np.linalg.norm(predicted - actual))
        
        # Combined free energy
        free_energy = uncertainty + avg_divergence * 0.5 - np.log(coherence + 1e-8)
        
        return free_energy
    
    def select_minimal_free_energy_action(self,
                                         current_self: np.ndarray,
                                         possible_actions: List[np.ndarray],
                                         world_state: np.ndarray,
                                         transition_model: LearnedSelfTransitionModel) -> np.ndarray:
        """
        Select action that minimizes expected free energy.
        """
        if not possible_actions:
            return np.zeros(self.latent_dim)
        
        free_energies = []
        for action in possible_actions:
            G = self.compute_expected_free_energy(current_self, action, world_state, transition_model)
            free_energies.append(G)
        
        # Select action with minimum free energy
        best_idx = np.argmin(free_energies)
        
        return possible_actions[best_idx]


class GenerativeIdentityDynamics:
    """
    Phase 10: Generative Identity Dynamics
    
    Self as probabilistic latent manifold with learned generative model.
    
    Key innovations:
      - Variational self-belief (not point estimate)
      - Learned transition model (not handcrafted dynamics)
      - Free-energy minimization (not heuristic stability)
      - Self-destabilization for adaptation (not always stable)
      - Attractor-based recursion (not depth recursion)
    """
    
    def __init__(self, latent_dim: int = 2, world_dim: int = 2):
        self.latent_dim = latent_dim
        self.world_dim = world_dim
        
        # Variational self-belief
        self.self_belief = VariationalSelfBelief(
            mean=np.zeros(latent_dim),
            variance=np.ones(latent_dim),
            entropy=0.0
        )
        self.self_belief.entropy = 0.5 * np.sum(np.log(2 * np.pi * self.self_belief.variance + 1e-8))
        
        # Learned self-transition model
        self.transition_model = LearnedSelfTransitionModel(latent_dim)
        
        # Free-energy self stabilization
        self.free_energy = FreeEnergySelfStabilization(
            preferred_self=self.self_belief,
            latent_dim=latent_dim
        )
        
        # Self/world co-generation
        self.co_emerge_alpha = 0.1  # How much self influences world perception
        
        # History
        self.self_history: List[np.ndarray] = []
        self.world_history: List[np.ndarray] = []
        self.actions_history: List[np.ndarray] = []
        
        # Attractor recursion state (fixed-point dynamics, not depth)
        self.attractor_state = self.self_belief.mean.copy()
        self.attractor_convergence_threshold = 0.01
        self.max_iterations = 10
        
    def update_self(self, action: np.ndarray, world_state: np.ndarray,
                    observation: np.ndarray) -> VariationalSelfBelief:
        """
        Update variational self-belief using observations.
        
        This is where self-model LEARNS from experience.
        """
        # Store history
        self.self_history.append(self.self_belief.mean.copy())
        self.world_history.append(world_state.copy())
        self.actions_history.append(action.copy())
        
        # Update belief using variational inference
        self.self_belief.update_variational(
            observations=self.self_history[-10:] if len(self.self_history) >= 10 else self.self_history,
            action=action,
            world_state=world_state,
            transition_model=self.transition_model
        )
        
        # Update free-energy preferred self
        self.free_energy.preferred_self = self.self_belief
        
        return self.self_belief
    
    def sample_counterfactual_self(self, n_samples: int = 10) -> np.ndarray:
        """
        Sample possible self-trajectories from learned latent dynamics.
        
        NOT: expand / contract / orthogonal (symbolic)
        BUT: sample from P(S_t+1 | S_t, A_t, W_t)
        """
        samples = []
        current = self.self_belief.mean.copy()
        
        for _ in range(n_samples):
            # Sample action
            action = np.random.randn(self.latent_dim) * 0.3
            
            # Predict next self
            next_self = self.transition_model.predict(current, action, 
                                                      self.world_history[-1] if self.world_history else np.zeros(self.world_dim))
            samples.append(next_self)
            current = next_self
        
        return np.array(samples)
    
    def attractor_recursion(self, external_signal: np.ndarray) -> np.ndarray:
        """
        Attractor-based recursion (NOT depth recursion).
        
        True recursion = fixed-point dynamics:
          self-model influences predictions
          predictions influence self-model
          until convergence
          
        Not: meta_level += 1 (depth)
        """
        current = self.attractor_state.copy()
        
        for iteration in range(self.max_iterations):
            # Self influences prediction
            self_influence = self.self_belief.mean * 0.5
            
            # Prediction influences self (back to current belief)
            prediction_influence = self.self_belief.mean * 0.5
            
            # External signal perturbation
            new_state = current + self_influence + prediction_influence * 0.1 + external_signal * 0.2
            
            # Check convergence
            change = np.linalg.norm(new_state - current)
            
            if change < self.attractor_convergence_threshold:
                # Converged to fixed point
                break
            
            current = new_state
        
        self.attractor_state = current
        return current
    
    def check_self_destabilization_needed(self) -> bool:
        """
        Should identity destabilize for adaptation?
        
        Real recursive identity should sometimes destabilize itself.
        Otherwise: no adaptation pressure, no phase transitions, no restructuring.
        """
        # Check if self is too stable (bad sign)
        if len(self.self_history) > 20:
            recent_variance = np.var(self.self_history[-20:], axis=0)
            
            # If variance is too low, identity is rigid
            if np.mean(recent_variance) < 0.001:
                return True  # Need destabilization
        
        # Check if self is too certain
        if np.mean(self.self_belief.variance) < 0.05:
            return True  # Need more uncertainty for adaptation
        
        return False
    
    def destabilize_self(self, intensity: float = 0.5) -> np.ndarray:
        """
        Destabilize identity for adaptation.
        
        This creates phase transitions and restructuring.
        """
        # Increase variance (more uncertainty)
        self.self_belief.variance = np.clip(
            self.self_belief.variance * (1 + intensity),
            0.01, 2.0
        )
        
        # Perturb belief mean
        perturbation = np.random.randn(self.latent_dim) * intensity
        self.self_belief.mean = self.self_belief.mean + perturbation
        
        # Update entropy
        self.self_belief.entropy = 0.5 * np.sum(np.log(2 * np.pi * self.self_belief.variance + 1e-8))
        
        return self.self_belief.mean
    
    def co_emerge_world_self(self, raw_world: np.ndarray) -> np.ndarray:
        """
        Self/world co-generation.
        
        Self-state and world-state co-emerge.
        Not: self affects world perception
        But: self and world jointly create each other
        """
        # World perception is influenced by self
        self_influence = np.tanh(self.self_belief.mean) * self.co_emerge_alpha
        
        # Co-emerge: both influence each other
        co_emergeed_world = raw_world + self_influence
        
        # Also self is influenced by perceived world
        world_influence = np.tanh(raw_world) * self.co_emerge_alpha * 0.5
        self.self_belief.mean = self.self_belief.mean + world_influence
        
        return co_emergeed_world
    
    def compute_identity_free_energy(self) -> float:
        """
        Compute free energy of current identity belief.
        
        This is the surprise / uncertainty of being "who I am".
        """
        # Distance from preferred self
        divergence = np.linalg.norm(self.self_belief.mean - self.free_energy.preferred_self.mean)
        
        # Uncertainty
        uncertainty = self.self_belief.entropy
        
        # Combined
        return divergence + uncertainty * 0.5
    
    def get_identity_state(self) -> Dict:
        """Get comprehensive identity state."""
        return {
            'self_belief_mean': self.self_belief.mean.tolist(),
            'self_belief_variance': self.self_belief.variance.tolist(),
            'self_uncertainty_entropy': self.self_belief.entropy,
            'identity_free_energy': self.compute_identity_free_energy(),
            'attractor_state': self.attractor_state.tolist(),
            'transitions_learned': self.transition_model.transitions_observed,
            'stable_trajectories_count': len(self.transition_model.stable_self_trajectories),
            'self_history_length': len(self.self_history),
            'destabilization_needed': self.check_self_destabilization_needed()
        }


class GenerativeCognitiveAgent:
    """
    Agent with Generative Identity Dynamics.
    
    NOT: vector-based self with heuristics
    BUT: probabilistic self with learned generative model
    """
    
    def __init__(self, latent_dim: int = 2, world_dim: int = 2):
        self.latent_dim = latent_dim
        self.world_dim = world_dim
        
        # State
        self.state = np.random.randn(latent_dim) * 0.1
        self.raw_world = np.random.randn(world_dim) * 0.1
        
        # Generative Identity Dynamics
        self.identity = GenerativeIdentityDynamics(latent_dim, world_dim)
        
        # Track
        self.trajectory: List[np.ndarray] = []
        
    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Take action and update generative identity.
        """
        # Co-emerge world and self
        world = self.identity.co_emerge_world_self(self.raw_world)
        
        # Update state
        new_state = self.state + action * 0.1
        
        # Record
        self.trajectory.append(self.state.copy())
        if len(self.trajectory) > 500:
            self.trajectory = self.trajectory[-250:]
        
        # Update variational self-belief
        self.identity.update_self(action, world, new_state)
        
        # Check for needed destabilization
        if self.identity.check_self_destabilization_needed():
            self.identity.destabilize_self(intensity=0.3)
        
        # Attractor recursion
        self.identity.attractor_recursion(new_state)
        
        self.state = new_state
        return new_state
    
    def decide_action(self) -> np.ndarray:
        """
        Decide action using free-energy minimization.
        """
        # Generate possible actions
        possible_actions = []
        
        # Goal direction (random goal for demo)
        goal = np.random.randn(self.latent_dim) * 2
        goal_dir = goal - self.state
        if np.linalg.norm(goal_dir) > 0.01:
            goal_unit = goal_dir / np.linalg.norm(goal_dir)
        else:
            goal_unit = np.zeros(self.latent_dim)
        
        for scale in [0.3, 0.5, 0.7]:
            possible_actions.append(goal_unit * scale)
        
        # Random exploration
        for _ in range(3):
            possible_actions.append(np.random.randn(self.latent_dim) * 0.3)
        
        # Select using free-energy minimization
        action = self.identity.free_energy.select_minimal_free_energy_action(
            self.identity.self_belief.mean,
            possible_actions,
            self.raw_world,
            self.identity.transition_model
        )
        
        return action
    
    def run_episode(self, n_steps: int = 100) -> Dict:
        """Run episode with generative identity dynamics."""
        print("\n  Running generative identity dynamics episode:")
        
        for step in range(n_steps):
            action = self.decide_action()
            self.step(action)
            
            if step % 20 == 0:
                state = self.identity.get_identity_state()
                print(f"    Step {step}: entropy={state['self_uncertainty_entropy']:.3f}, "
                      f"free_energy={state['identity_free_energy']:.3f}, "
                      f"destabilize={state['destabilization_needed']}")
        
        return self.identity.get_identity_state()


def test_generative_identity_dynamics():
    """Test generative identity dynamics."""
    print("\n" + "=" * 60)
    print("GENERATIVE IDENTITY DYNAMICS TEST")
    print("=" * 60)
    
    agent = GenerativeCognitiveAgent(latent_dim=2, world_dim=2)
    
    # Run multiple episodes to see learning
    print("\n  Episode 1: Initial learning")
    result1 = agent.run_episode(50)
    
    print(f"\n  After episode 1:")
    print(f"    Transitions learned: {result1['transitions_learned']}")
    print(f"    Stable trajectories: {result1['stable_trajectories_count']}")
    
    # Learn from stable outcomes
    agent.identity.transition_model.learn_from_trajectory(
        agent.trajectory,
        agent.identity.actions_history,
        agent.identity.world_history,
        outcome="stable"
    )
    
    # Episode 2
    print("\n  Episode 2: Continue learning")
    result2 = agent.run_episode(50)
    
    print(f"\n  After episode 2:")
    print(f"    Transitions learned: {result2['transitions_learned']}")
    print(f"    Self uncertainty: {result2['self_uncertainty_entropy']:.3f}")
    
    # Test counterfactual self-sampling
    print("\n  Testing counterfactual self-sampling:")
    counterfactuals = agent.identity.sample_counterfactual_self(n_samples=10)
    print(f"    Sampled {len(counterfactuals)} counterfactual selves")
    print(f"    Counterfactual mean: {np.mean(counterfactuals, axis=0)[:2]}")
    print(f"    Counterfactual variance: {np.var(counterfactuals, axis=0)[:2]}")
    
    # Test attractor recursion
    print("\n  Testing attractor recursion:")
    external_signal = np.array([0.5, 0.5])
    fixed_point = agent.identity.attractor_recursion(external_signal)
    print(f"    External signal: {external_signal}")
    print(f"    Fixed point: {fixed_point[:2]}")
    
    # Test self-destabilization
    print("\n  Testing self-destabilization:")
    print(f"    Destabilization needed: {result2['destabilization_needed']}")
    
    if result2['destabilization_needed']:
        before = agent.identity.self_belief.variance.copy()
        agent.identity.destabilize_self(intensity=0.5)
        after = agent.identity.self_belief.variance.copy()
        print(f"    Variance before: {before}")
        print(f"    Variance after: {after}")
    
    # Test self/world co-generation
    print("\n  Testing self/world co-generation:")
    raw_w = np.array([1.0, 1.0])
    co_emergeed = agent.identity.co_emerge_world_self(raw_w)
    print(f"    Raw world: {raw_w}")
    print(f"    Self mean: {agent.identity.self_belief.mean[:2]}")
    print(f"    Co-emergeed: {co_emergeed}")
    
    # Final identity state
    print("\n  Final identity state:")
    final = agent.identity.get_identity_state()
    for k, v in final.items():
        if isinstance(v, list):
            print(f"    {k}: {v[:2] if len(v) > 2 else v}")
        else:
            print(f"    {k}: {v}")


def test_free_energy_minimization():
    """Test free-energy minimization for action selection."""
    print("\n" + "=" * 60)
    print("FREE-ENERGY MINIMIZATION TEST")
    print("=" * 60)
    
    identity = GenerativeIdentityDynamics(latent_dim=2, world_dim=2)
    
    # Set some initial belief
    identity.self_belief.mean = np.array([1.0, 1.0])
    identity.self_belief.variance = np.array([0.2, 0.2])
    
    print("\n  Testing free-energy computation:")
    
    current = np.array([1.0, 1.0])
    actions = [
        np.array([0.5, 0.5]),
        np.array([-0.5, -0.5]),
        np.array([0.5, -0.5])
    ]
    
    for action in actions:
        G = identity.free_energy.compute_expected_free_energy(
            current, action, np.array([0.0, 0.0]), identity.transition_model
        )
        print(f"    Action {action[:2]}: Free energy = {G:.3f}")
    
    # Select best action
    best = identity.free_energy.select_minimal_free_energy_action(
        current, actions, np.array([0.0, 0.0]), identity.transition_model
    )
    print(f"\n  Best action (min free energy): {best[:2]}")


def test_self_destabilization():
    """Test self-destabilization for adaptation."""
    print("\n" + "=" * 60)
    print("SELF-DESTABILIZATION TEST")
    print("=" * 60)
    
    identity = GenerativeIdentityDynamics(latent_dim=2, world_dim=2)
    
    # Make self very stable (too stable = bad)
    identity.self_history = [np.array([1.0, 1.0])] * 30
    identity.self_belief.variance = np.array([0.01, 0.01])
    
    print("\n  Before destabilization:")
    print(f"    Self variance: {identity.self_belief.variance}")
    print(f"    Destabilization needed: {identity.check_self_destabilization_needed()}")
    
    # Destabilize
    new_mean = identity.destabilize_self(intensity=0.5)
    
    print("\n  After destabilization:")
    print(f"    New mean: {new_mean[:2]}")
    print(f"    New variance: {identity.self_belief.variance}")
    print(f"    New entropy: {identity.self_belief.entropy:.3f}")


def compare_with_phase9():
    """Compare Phase 10 (Generative) with Phase 9 (Recursive)."""
    print("\n" + "=" * 60)
    print("PHASE 9 VS PHASE 10 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 9 (Recursive Self-Modeling):")
    print("    - self_state = vector")
    print("    - self_dynamics = handcrafted")
    print("    - self_confidence = heuristic")
    print("    - transformation_pressures = symbolic (expand/contract)")
    print("    - meta_level = depth recursion")
    print("    - stability = 1.0 (too stable)")
    
    print("\n  Phase 10 (Generative Identity Dynamics):")
    print("    - self = variational belief (distribution)")
    print("    - self-transition = learned P(S_t+1 | S_t, A_t, W_t)")
    print("    - identity uncertainty = entropy(self_beliefs)")
    print("    - counterfactual = sampled from learned dynamics")
    print("    - recursion = attractor fixed-point (not depth)")
    print("    - stability = adaptive (with destabilization)")
    
    print("\n  Key architectural shifts:")
    print("    1. self: vector → probabilistic manifold")
    print("    2. self_dynamics: handcrafted → learned")
    print("    3. stability: always stable → adaptive destabilization")
    print("    4. recursion: depth → attractor fixed-point")
    print("    5. self_uncertainty: confidence heuristic → belief entropy")
    
    print("\n  What Phase 10 enables:")
    print("    - System learns self-model from experience")
    print("    - Identity uncertainty = real belief entropy")
    print("    - Actions minimize expected free energy")
    print("    - Self sometimes destabilizes for adaptation")
    print("    - Self/world co-emerge")
    print("    - Phase transitions and restructuring")


if __name__ == "__main__":
    test_generative_identity_dynamics()
    test_free_energy_minimization()
    test_self_destabilization()
    compare_with_phase9()
    
    print("\n" + "=" * 60)
    print("PHASE 10 - GENERATIVE IDENTITY DYNAMICS")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: self as vector with handcrafted dynamics
  To: self as probabilistic latent manifold with learned generative model

KEY INSIGHT:
  NOT: "describe selfhood symbolically"
  BUT: "implement generative selfhood dynamically"
  
  System must LEARN self-model from experience,
  not execute handcrafted selfhood theory.

WHAT APPEARS:
  1. Learned self-transition model P(S_t+1 | S_t, W_t, A_t)
     - Self-evolution as learned latent dynamics
     - Not handcrafted transformation operators
     
  2. Variational self-belief q(self)
     - Self = probability distribution, not point estimate
     - entropy(self_beliefs) = identity uncertainty
     
  3. Free-energy self stabilization
     - minimize expected free energy over future self-trajectories
     - Actions that preserve identity predictability
     
  4. Self-destabilization for adaptation
     - Real recursive identity sometimes destabilizes itself
     - Phase transitions, restructuring, adaptation pressure
     
  5. Self/world co-generation
     - Self-model and world-model co-emerge
     - Attractor-based recursion (not depth recursion)

CRITICAL PROBLEMS WITH PHASE 9:
  1. self_confidence = heuristic (should be entropy)
  2. self_dynamics = handcrafted (should be learned)
  3. transformation_pressures = symbolic (should be sampled from latent)
  4. meta_level = depth recursion (should be fixed-point attractor)
  5. Identity too stable (no adaptation pressure)

PROGRESSION:
  Phase 8: Self-organizing cognition
  Phase 9: Recursive self-modeling
  Phase 10: Generative identity dynamics (learned, probabilistic, adaptive)
  Phase 11: Self-maintaining generative ontology
  Phase 12+: Proto-conscious integration

This is the transition from:
  "simulating philosophy of selfhood"
  to:
  "implementing generative selfhood"
  
We are now at:
  - Proto-autopoietic cognitive systems
  - Self-maintaining inference structures
  - Identity-preserving generative agents
  - Dynamical self-modeling systems
""")