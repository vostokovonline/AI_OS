"""
AI-OS Phase 30: Minimal Probabilistic Dynamics Core
====================================================

ARCHITECTURAL SHIFT:
  From: Deterministic heuristic simulator
  To: Probabilistic latent dynamics with planning
  
NOT: "world model" or "causal theory"
BUT: compact probabilistic behavior predictor + planning tool

CORE PRINCIPLE:
  We model DISTRIBUTION of future states, not single trajectory.
  System understands "I'm not sure" via ensemble disagreement.

COMPONENTS:
  1. GRU Belief State (h_t = GRU(h_{t-1}, x_t))
  2. Stochastic Latent State (z_t ~ q(z_t | h_t))
  3. Ensemble Transition Models (N models → uncertainty)
  4. Learned Reward Model (r̂ = MLP([z_t, h_t]))
  5. World Rollout Engine (trajectory distributions)
  6. CEM Planner (search in latent space)
  7. Uncertainty Gate (fallback when unsure)

DATA FLOW:
  BehavioralEvent_t
      ↓
  Embedding x_t
      ↓
  GRU belief update → h_t
      ↓
  Stochastic encoder → z_t
      ↓
  Ensemble transition models
      ↓
  Future rollouts (latent trajectories)
      ↓
  Reward model evaluates trajectories
      ↓
  CEM selects best action sequence
      ↓
  Action / recommendation output
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import math


# ============================================================================
# 1. GRU BELIEF STATE CORE
# ============================================================================
"""
GRU-based belief state: h_t = GRU(h_{t-1}, x_t)

NOT: Transformer or full sequence model
BUT: compact recurrent memory that compresses history

This is the "context state" of the system.
"""

class GRUBeliefState:
    """
    GRU-based belief state for sequence memory.
    
    h_t = GRU(h_{t-1}, x_t)
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # GRU weights (simplified)
        # Update gate: z = σ(W_z @ [h, x] + b_z)
        self.W_zh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_zx = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_z = np.zeros(hidden_dim)
        
        # Reset gate: r = σ(W_r @ [h, x] + b_r)
        self.W_rh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_rx = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_r = np.zeros(hidden_dim)
        
        # Candidate: h̃ = tanh(W_h @ [r*h, x] + b_h)
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_hx = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_h = np.zeros(hidden_dim)
        
        # Current belief state
        self.h = np.zeros(hidden_dim)
        
        # History for debugging
        self.belief_history: List[np.ndarray] = []
    
    def reset(self):
        """Reset belief state."""
        self.h = np.zeros(self.hidden_dim)
        self.belief_history = []
    
    def update(self, x: np.ndarray) -> np.ndarray:
        """
        Update belief state with new observation.
        
        h_t = GRU(h_{t-1}, x_t)
        """
        x = np.asarray(x).flatten()[:self.input_dim]
        
        # Pad if needed
        if len(x) < self.input_dim:
            x = np.pad(x, (0, self.input_dim - len(x)))
        
        # Update gate
        z = self._sigmoid(self.W_zh @ self.h + self.W_zx @ x + self.b_z)
        
        # Reset gate
        r = self._sigmoid(self.W_rh @ self.h + self.W_rx @ x + self.b_r)
        
        # Candidate hidden state
        h_tilde = np.tanh(self.W_hh @ (r * self.h) + self.W_hx @ x + self.b_h)
        
        # New hidden state
        self.h = (1 - z) * self.h + z * h_tilde
        
        # Record
        self.belief_history.append(self.h.copy())
        if len(self.belief_history) > 100:
            self.belief_history = self.belief_history[-50:]
        
        return self.h.copy()
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def get_belief_summary(self) -> Dict:
        """Get belief state summary."""
        return {
            'hidden_dim': self.hidden_dim,
            'belief_norm': float(np.linalg.norm(self.h)),
            'history_length': len(self.belief_history)
        }


# ============================================================================
# 2. STOCHASTIC LATENT STATE
# ============================================================================
"""
Stochastic latent state: z_t ~ q(z_t | h_t)

NOT: deterministic encoding
BUT: probabilistic latent with mean and variance

This is where "world" becomes distribution, not point.
"""

class StochasticLatentState:
    """
    Stochastic latent state encoder.
    
    z_t ~ N(μ(h_t), σ²(h_t))
    """
    
    def __init__(self, belief_dim: int = 64, latent_dim: int = 16):
        self.belief_dim = belief_dim
        self.latent_dim = latent_dim
        
        # Mean network: μ = W_μ @ h + b_μ
        self.W_mu = np.random.randn(latent_dim, belief_dim) * 0.1
        self.b_mu = np.zeros(latent_dim)
        
        # Variance network: log(σ²) = W_σ @ h + b_σ
        self.W_logvar = np.random.randn(latent_dim, belief_dim) * 0.05
        self.b_logvar = np.zeros(latent_dim) - 1.0  # Start with low variance
        
        # Current latent sample
        self.z = np.zeros(latent_dim)
        self.mu = np.zeros(latent_dim)
        self.logvar = np.zeros(latent_dim)
    
    def encode(self, h: np.ndarray, sample: bool = True) -> np.ndarray:
        """
        Encode belief to stochastic latent state.
        
        z ~ N(μ(h), σ²(h))
        """
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        # Compute parameters
        self.mu = self.W_mu @ h + self.b_mu
        self.logvar = self.W_logvar @ h + self.b_logvar
        
        # Sample using reparameterization trick
        if sample:
            std = np.exp(0.5 * self.logvar)
            epsilon = np.random.randn(self.latent_dim)
            self.z = self.mu + std * epsilon
        else:
            self.z = self.mu  # Deterministic (use mu)
        
        return self.z.copy()
    
    def get_distribution(self) -> Dict:
        """Get current latent distribution."""
        return {
            'mu': self.mu.tolist(),
            'logvar': self.logvar.tolist(),
            'std': np.exp(0.5 * self.logvar).tolist(),
            'z': self.z.tolist()
        }
    
    def kl_divergence(self, prior_mu: np.ndarray = None, 
                     prior_logvar: np.ndarray = None) -> float:
        """
        Compute KL divergence from prior.
        
        KL(q || p) where p is standard normal by default.
        """
        if prior_mu is None:
            prior_mu = np.zeros(self.latent_dim)
        if prior_logvar is None:
            prior_logvar = np.zeros(self.latent_dim)
        
        # KL = 0.5 * sum(exp(logvar_p) * (exp(logvar_q) + (mu_q - mu_p)^2) - logvar_q + logvar_p - 1)
        kl = 0.5 * np.sum(
            np.exp(prior_logvar - self.logvar) +
            (self.mu - prior_mu) ** 2 / np.exp(self.logvar) +
            self.logvar - prior_logvar - 1
        )
        
        return float(kl)


# ============================================================================
# 3. ENSEMBLE TRANSITION MODELS
# ============================================================================
"""
Ensemble transition models: p_k(z_{t+1} | z_t, h_t, a_t)

NOT: single deterministic model
BUT: 2-3 models → disagreement = epistemic uncertainty

This is how system knows "I'm not sure".
"""

class EnsembleTransitionModel:
    """
    Ensemble of transition models for uncertainty estimation.
    """
    
    def __init__(self, latent_dim: int = 16, belief_dim: int = 64,
                 action_dim: int = 16, n_models: int = 3):
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        self.action_dim = action_dim
        self.n_models = n_models
        
        # Create ensemble of models
        self.models = []
        for i in range(n_models):
            model = self._create_model(latent_dim, belief_dim, action_dim)
            self.models.append(model)
        
        # Training history
        self.loss_history: List[List[float]] = [[] for _ in range(n_models)]
    
    def _create_model(self, latent_dim: int, belief_dim: int, 
                     action_dim: int) -> Dict:
        """Create single transition model."""
        # Input: [z_t, h_t, a_t]
        input_dim = latent_dim + belief_dim + action_dim
        
        return {
            'W1': np.random.randn(64, input_dim) * 0.1,
            'b1': np.zeros(64),
            'W2': np.random.randn(latent_dim, 64) * 0.1,
            'b2': np.zeros(latent_dim),
            'W_logvar': np.random.randn(latent_dim, 64) * 0.05,
            'b_logvar': np.zeros(latent_dim) - 1.0
        }
    
    def predict(self, z: np.ndarray, h: np.ndarray, 
                action: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict next latent state with uncertainty.
        
        Returns: (mean, variance, ensemble_predictions)
        """
        z = np.asarray(z).flatten()[:self.latent_dim]
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        if action is None:
            action = np.zeros(self.action_dim)
        action = np.asarray(action).flatten()[:self.action_dim]
        
        # Pad action if needed
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))
        
        # Input vector
        x = np.concatenate([z, h, action])
        
        # Run all models
        predictions = []
        logvars = []
        
        for model in self.models:
            # Forward pass
            hidden = np.tanh(model['W1'] @ x + model['b1'])
            mu = model['W2'] @ hidden + model['b2']
            logvar = model['W_logvar'] @ hidden + model['b_logvar']
            
            predictions.append(mu)
            logvars.append(logvar)
        
        predictions = np.array(predictions)  # (n_models, latent_dim)
        logvars = np.array(logvars)
        
        # Ensemble mean and variance
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0) + np.mean(np.exp(logvars), axis=0)
        
        return mean, variance, predictions
    
    def update(self, z_current: np.ndarray, z_next: np.ndarray,
               h: np.ndarray, action: np.ndarray = None,
               learning_rate: float = 0.001) -> List[float]:
        """Update ensemble models from observed transition."""
        z_current = np.asarray(z_current).flatten()[:self.latent_dim]
        z_next = np.asarray(z_next).flatten()[:self.latent_dim]
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        if action is None:
            action = np.zeros(self.action_dim)
        action = np.asarray(action).flatten()[:self.action_dim]
        
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)))
        
        x = np.concatenate([z_current, h, action])
        losses = []
        
        for i, model in enumerate(self.models):
            # Forward pass
            hidden = np.tanh(model['W1'] @ x + model['b1'])
            mu = model['W2'] @ hidden + model['b2']
            
            # MSE loss
            error = mu - z_next
            loss = np.mean(error ** 2)
            losses.append(loss)
            
            # Backward pass (simplified)
            grad_mu = error
            
            # Update W2, b2
            grad_W2 = np.outer(grad_mu, hidden)
            grad_b2 = grad_mu
            model['W2'] -= learning_rate * grad_W2
            model['b2'] -= learning_rate * grad_b2
            
            # Update W1, b1 (through tanh)
            grad_hidden = model['W2'].T @ grad_mu * (1 - hidden ** 2)
            grad_W1 = np.outer(grad_hidden, x)
            grad_b1 = grad_hidden
            model['W1'] -= learning_rate * grad_W1
            model['b1'] -= learning_rate * grad_b1
            
            self.loss_history[i].append(loss)
            if len(self.loss_history[i]) > 100:
                self.loss_history[i] = self.loss_history[i][-50:]
        
        return losses
    
    def get_uncertainty(self, z: np.ndarray, h: np.ndarray, 
                       action: np.ndarray = None) -> float:
        """Get epistemic uncertainty for state-action pair."""
        _, variance, _ = self.predict(z, h, action)
        return float(np.mean(variance))
    
    def get_ensemble_summary(self) -> Dict:
        """Get ensemble summary."""
        avg_loss = np.mean([np.mean(h) if h else 0 for h in self.loss_history])
        
        return {
            'n_models': self.n_models,
            'avg_loss': avg_loss,
            'latent_dim': self.latent_dim
        }


# ============================================================================
# 4. LEARNED REWARD MODEL
# ============================================================================
"""
Learned reward model: r̂ = MLP([z_t, h_t])

NOT: heuristic scoring
BUT: learned from behavioral traces (what system actually chose)

This approximates the system's implicit preference function.
"""

class LearnedRewardModel:
    """
    Learned reward approximation from behavior.
    """
    
    def __init__(self, latent_dim: int = 16, belief_dim: int = 64):
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        
        # MLP weights
        input_dim = latent_dim + belief_dim
        self.W1 = np.random.randn(32, input_dim) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(1, 32) * 0.1
        self.b2 = np.zeros(1)
        
        # Training data
        self.training_data: List[Tuple[np.ndarray, float]] = []
        self.loss_history: List[float] = []
    
    def predict_reward(self, z: np.ndarray, h: np.ndarray) -> float:
        """Predict reward for state."""
        z = np.asarray(z).flatten()[:self.latent_dim]
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        x = np.concatenate([z, h])
        
        # Forward pass
        hidden = np.tanh(self.W1 @ x + self.b1)
        reward = self.W2 @ hidden + self.b2
        
        return float(reward[0])
    
    def update(self, z: np.ndarray, h: np.ndarray, 
               target_reward: float, learning_rate: float = 0.001) -> float:
        """Update reward model from observed outcome."""
        z = np.asarray(z).flatten()[:self.latent_dim]
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        x = np.concatenate([z, h])
        
        # Forward pass
        hidden = np.tanh(self.W1 @ x + self.b1)
        reward = self.W2 @ hidden + self.b2
        
        # MSE loss
        error = reward[0] - target_reward
        loss = error ** 2
        
        # Backward pass
        grad_reward = np.array([error])
        
        # Update W2, b2
        grad_W2 = np.outer(grad_reward, hidden)
        grad_b2 = grad_reward
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2
        
        # Update W1, b1
        grad_hidden = self.W2.T @ grad_reward * (1 - hidden ** 2)
        grad_W1 = np.outer(grad_hidden, x)
        grad_b1 = grad_hidden
        self.W1 -= learning_rate * grad_W1
        self.b1 -= learning_rate * grad_b1
        
        self.loss_history.append(loss)
        if len(self.loss_history) > 100:
            self.loss_history = self.loss_history[-50:]
        
        return loss
    
    def learn_from_behavior(self, successful_states: List[Tuple[np.ndarray, np.ndarray]],
                           failed_states: List[Tuple[np.ndarray, np.ndarray]],
                           n_epochs: int = 10):
        """
        Learn reward from behavioral traces.
        
        Successful states → high reward
        Failed states → low reward
        """
        # Create training data
        for z, h in successful_states:
            self.training_data.append((np.concatenate([z, h]), 1.0))
        
        for z, h in failed_states:
            self.training_data.append((np.concatenate([z, h]), 0.0))
        
        # Train
        for epoch in range(n_epochs):
            np.random.shuffle(self.training_data)
            for x, target in self.training_data:
                z = x[:self.latent_dim]
                h = x[self.latent_dim:]
                self.update(z, h, target, learning_rate=0.01)
    
    def get_reward_summary(self) -> Dict:
        """Get reward model summary."""
        return {
            'n_training_samples': len(self.training_data),
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0
        }


# ============================================================================
# 5. WORLD ROLLOUT ENGINE
# ============================================================================
"""
World Rollout Engine: simulate trajectories in latent space

NOT: single trajectory prediction
BUT: distribution of possible futures
"""

@dataclass
class LatentTrajectory:
    """One simulated trajectory."""
    states: List[np.ndarray]  # z_t sequence
    beliefs: List[np.ndarray]  # h_t sequence
    actions: List[str]  # action sequence
    rewards: List[float]  # predicted rewards
    total_reward: float
    uncertainty: float  # trajectory uncertainty


class WorldRolloutEngine:
    """
    Simulates trajectories in latent space.
    """
    
    def __init__(self, transition_model: EnsembleTransitionModel,
                 reward_model: LearnedRewardModel,
                 latent_dim: int = 16, belief_dim: int = 64):
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        
        # Action vocabulary
        self.actions = [
            'deep_focus', 'context_switch', 'take_break',
            'continue_current', 'simplify_task', 'collaborate', 'explore'
        ]
        
        # Action embeddings
        self.action_embeddings: Dict[str, np.ndarray] = {}
        for action in self.actions:
            self.action_embeddings[action] = np.random.randn(16) * 0.1
    
    def rollout(self, z0: np.ndarray, h0: np.ndarray,
                action_sequence: List[str],
                horizon: int = None) -> LatentTrajectory:
        """
        Rollout trajectory given action sequence.
        """
        if horizon is None:
            horizon = len(action_sequence)
        
        z = z0.copy()
        h = h0.copy()
        
        states = [z.copy()]
        beliefs = [h.copy()]
        rewards = []
        uncertainties = []
        
        for i in range(min(len(action_sequence), horizon)):
            action = action_sequence[i]
            action_emb = self.action_embeddings.get(action, np.zeros(16))
            
            # Predict next state
            z_mean, z_var, _ = self.transition_model.predict(z, h, action_emb)
            
            # Sample next state (stochastic)
            z_std = np.sqrt(z_var)
            z_next = z_mean + np.random.randn(self.latent_dim) * z_std
            
            # Update belief (simplified - just carry forward with latent influence)
            h_next = h * 0.9 + np.pad(z_next, (0, len(h) - len(z_next))) * 0.1
            
            # Predict reward
            reward = self.reward_model.predict_reward(z_next, h_next)
            
            # Track uncertainty
            uncertainty = np.mean(z_var)
            
            states.append(z_next.copy())
            beliefs.append(h_next.copy())
            rewards.append(reward)
            uncertainties.append(uncertainty)
            
            z = z_next
            h = h_next
        
        return LatentTrajectory(
            states=states,
            beliefs=beliefs,
            actions=action_sequence[:horizon],
            rewards=rewards,
            total_reward=sum(rewards),
            uncertainty=np.mean(uncertainties) if uncertainties else 0
        )
    
    def rollout_distribution(self, z0: np.ndarray, h0: np.ndarray,
                            n_trajectories: int = 20,
                            horizon: int = 5) -> List[LatentTrajectory]:
        """
        Generate distribution of possible futures.
        """
        trajectories = []
        
        for _ in range(n_trajectories):
            # Sample random action sequence
            action_sequence = np.random.choice(self.actions, size=horizon).tolist()
            
            trajectory = self.rollout(z0, h0, action_sequence, horizon)
            trajectories.append(trajectory)
        
        # Sort by total reward
        trajectories.sort(key=lambda t: t.total_reward, reverse=True)
        
        return trajectories


# ============================================================================
# 6. CEM PLANNER
# ============================================================================
"""
CEM (Cross-Entropy Method) Planner:

NOT: heuristic decision rules
BUT: search over action sequences in latent space

Algorithm:
  1. Sample action sequences from distribution
  2. Rollout in latent space
  3. Score via reward model
  4. Select top-k
  5. Update distribution
  6. Repeat
"""

class CEMPlanner:
    """
    Cross-Entropy Method planner in latent space.
    """
    
    def __init__(self, rollout_engine: WorldRolloutEngine,
                 action_dim: int = 16):
        self.rollout_engine = rollout_engine
        self.action_dim = action_dim
        self.actions = rollout_engine.actions
        
        # CEM parameters
        self.n_samples = 50
        self.n_elites = 10
        self.n_iterations = 5
        self.horizon = 5
    
    def plan(self, z: np.ndarray, h: np.ndarray,
             horizon: int = None) -> Dict:
        """
        Plan optimal action sequence.
        
        Returns: best action sequence + confidence
        """
        if horizon is None:
            horizon = self.horizon
        
        # Initialize action distribution (uniform over actions)
        action_probs = np.ones(len(self.actions)) / len(self.actions)
        
        best_trajectory = None
        best_reward = -float('inf')
        
        for iteration in range(self.n_iterations):
            # Sample action sequences
            action_sequences = []
            for _ in range(self.n_samples):
                seq = np.random.choice(self.actions, size=horizon, p=action_probs).tolist()
                action_sequences.append(seq)
            
            # Rollout all sequences
            trajectories = []
            for seq in action_sequences:
                traj = self.rollout_engine.rollout(z, h, seq, horizon)
                trajectories.append(traj)
            
            # Select elites (top-k)
            trajectories.sort(key=lambda t: t.total_reward, reverse=True)
            elites = trajectories[:self.n_elites]
            
            # Update best
            if elites[0].total_reward > best_reward:
                best_reward = elites[0].total_reward
                best_trajectory = elites[0]
            
            # Update action distribution
            action_counts = np.zeros(len(self.actions))
            for traj in elites:
                for i, action in enumerate(traj.actions):
                    if i < horizon:
                        idx = self.actions.index(action)
                        action_counts[idx] += 1
            
            # Smooth update
            action_probs = 0.7 * (action_counts / action_counts.sum()) + 0.3 * action_probs
            action_probs = action_probs / action_probs.sum()
        
        # Compute confidence
        if best_trajectory:
            elite_rewards = [t.total_reward for t in trajectories[:self.n_elites]]
            confidence = 1.0 - (np.std(elite_rewards) / (np.mean(elite_rewards) + 1e-8))
            confidence = max(0, min(1, confidence))
        else:
            confidence = 0.0
        
        return {
            'best_action_sequence': best_trajectory.actions if best_trajectory else [],
            'expected_reward': best_reward,
            'confidence': confidence,
            'uncertainty': best_trajectory.uncertainty if best_trajectory else 0,
            'n_iterations': self.n_iterations,
            'elite_rewards': [t.total_reward for t in (trajectories[:self.n_elites] if 'trajectories' in dir() else [])]
        }


# ============================================================================
# 7. UNCERTAINTY GATE
# ============================================================================
"""
Uncertainty Gate:

NOT: always trust model
BUT: fallback when ensemble disagreement is high

Logic:
  - If uncertainty low → use planned actions
  - If uncertainty medium → reduce planning horizon
  - If uncertainty high → fallback to conservative policy
"""

class UncertaintyGate:
    """
    Gates planning based on model uncertainty.
    """
    
    def __init__(self, low_threshold: float = 0.1,
                 high_threshold: float = 0.5):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def evaluate(self, uncertainty: float) -> Dict:
        """
        Evaluate uncertainty level and recommend action.
        """
        if uncertainty < self.low_threshold:
            return {
                'level': 'low',
                'action': 'trust_plan',
                'horizon_multiplier': 1.0,
                'message': 'Model confident. Trust planned actions.'
            }
        elif uncertainty < self.high_threshold:
            return {
                'level': 'medium',
                'action': 'reduce_horizon',
                'horizon_multiplier': 0.5,
                'message': 'Moderate uncertainty. Reduce planning horizon.'
            }
        else:
            return {
                'level': 'high',
                'action': 'conservative',
                'horizon_multiplier': 0.2,
                'message': 'High uncertainty. Use conservative policy.'
            }


# ============================================================================
# INTEGRATED PROBABILISTIC DYNAMICS CORE
# ============================================================================

class ProbabilisticDynamicsCore:
    """
    Complete Phase 30: Minimal Probabilistic Dynamics Core.
    
    Integrates:
      1. GRU Belief State
      2. Stochastic Latent State
      3. Ensemble Transition Models
      4. Learned Reward Model
      5. World Rollout Engine
      6. CEM Planner
      7. Uncertainty Gate
    
    DATA FLOW:
      BehavioralEvent_t → Embedding x_t
          ↓
      GRU belief update → h_t
          ↓
      Stochastic encoder → z_t
          ↓
      Ensemble transition models → uncertainty
          ↓
      World rollouts (trajectory distributions)
          ↓
      Reward model evaluates
          ↓
      CEM selects best action sequence
          ↓
      Uncertainty gate → final recommendation
    """
    
    def __init__(self, event_dim: int = 32, latent_dim: int = 16,
                 belief_dim: int = 64, action_dim: int = 16,
                 n_ensemble: int = 3):
        # Core components
        self.belief_state = GRUBeliefState(input_dim=event_dim, hidden_dim=belief_dim)
        self.latent_state = StochasticLatentState(belief_dim=belief_dim, latent_dim=latent_dim)
        self.transition_model = EnsembleTransitionModel(
            latent_dim=latent_dim, belief_dim=belief_dim,
            action_dim=action_dim, n_models=n_ensemble
        )
        self.reward_model = LearnedRewardModel(latent_dim=latent_dim, belief_dim=belief_dim)
        
        # Planning components
        self.rollout_engine = WorldRolloutEngine(
            self.transition_model, self.reward_model,
            latent_dim=latent_dim, belief_dim=belief_dim
        )
        self.planner = CEMPlanner(self.rollout_engine, action_dim=action_dim)
        self.uncertainty_gate = UncertaintyGate()
        
        # Dimensions
        self.event_dim = event_dim
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        
        # State
        self.current_z = np.zeros(latent_dim)
        self.current_h = np.zeros(belief_dim)
        
        # Event history for training
        self.event_buffer: List[Tuple[str, Dict]] = []
    
    def ingest_event(self, event_type: str, context: Dict = None,
                    outcome: float = None):
        """
        Ingest event and update all components.
        """
        context = context or {}
        
        # Create event embedding
        x = self._event_to_embedding(event_type, context)
        
        # Update belief state
        self.current_h = self.belief_state.update(x)
        
        # Encode to stochastic latent
        self.current_z = self.latent_state.encode(self.current_h, sample=True)
        
        # Store event
        self.event_buffer.append((event_type, context))
        if len(self.event_buffer) > 200:
            self.event_buffer = self.event_buffer[-100:]
        
        # Update transition model if we have previous state
        if len(self.event_buffer) > 1:
            prev_event, prev_context = self.event_buffer[-2]
            prev_x = self._event_to_embedding(prev_event, prev_context)
            prev_z = self.latent_state.encode(
                self.belief_state.belief_history[-2] if len(self.belief_state.belief_history) > 1 else self.current_h,
                sample=False
            )
            
            action = self._event_to_action(event_type)
            self.transition_model.update(prev_z, self.current_z, self.current_h, action)
        
        # Update reward model if outcome provided
        if outcome is not None:
            self.reward_model.update(self.current_z, self.current_h, outcome)
    
    def _event_to_embedding(self, event_type: str, context: Dict) -> np.ndarray:
        """Convert event to embedding vector."""
        # Simple one-hot + context features
        event_types = ['git_commit', 'ide_focus', 'interruption', 'meeting',
                      'task_start', 'task_complete', 'browser_distraction', 'sleep']
        
        embedding = np.zeros(self.event_dim)
        
        # Event type one-hot
        if event_type in event_types:
            idx = event_types.index(event_type)
            if idx < self.event_dim:
                embedding[idx] = 1.0
        
        # Context features
        if 'files' in context:
            embedding[min(8, self.event_dim - 1)] = min(1.0, context['files'] / 10)
        if 'duration' in context:
            embedding[min(9, self.event_dim - 1)] = min(1.0, context['duration'] / 60)
        
        return embedding
    
    def _event_to_action(self, event_type: str) -> np.ndarray:
        """Convert event to action embedding."""
        action_map = {
            'git_commit': np.array([1, 0, 0, 0] + [0] * 12),
            'ide_focus': np.array([0.8, 0, 0, 0.2] + [0] * 12),
            'interruption': np.array([0, 1, 0, 0] + [0] * 12),
            'meeting': np.array([0, 0, 1, 0] + [0] * 12),
            'task_start': np.array([0.6, 0, 0, 0.4] + [0] * 12),
            'task_complete': np.array([0.7, 0, 0, 0.3] + [0] * 12),
        }
        return action_map.get(event_type, np.zeros(16))
    
    def plan_and_recommend(self) -> Dict:
        """
        Plan and get recommendation with uncertainty gating.
        """
        # Get uncertainty
        action = np.zeros(16)  # Null action for uncertainty check
        uncertainty = self.transition_model.get_uncertainty(
            self.current_z, self.current_h, action
        )
        
        # Evaluate uncertainty gate
        gate_result = self.uncertainty_gate.evaluate(uncertainty)
        
        # Adjust planning horizon
        horizon = max(2, int(self.planner.horizon * gate_result['horizon_multiplier']))
        
        # Plan
        if gate_result['level'] == 'high':
            # Conservative: just recommend maintaining current state
            plan_result = {
                'best_action_sequence': ['continue_current'] * horizon,
                'expected_reward': 0.0,
                'confidence': 0.3,
                'uncertainty': uncertainty
            }
        else:
            plan_result = self.planner.plan(self.current_z, self.current_h, horizon)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'plan': plan_result,
            'uncertainty_gate': gate_result,
            'latent_state_norm': float(np.linalg.norm(self.current_z)),
            'belief_state_norm': float(np.linalg.norm(self.current_h)),
            'ensemble_summary': self.transition_model.get_ensemble_summary(),
            'reward_summary': self.reward_model.get_reward_summary()
        }
    
    def get_state_summary(self) -> Dict:
        """Get current state summary."""
        return {
            'latent_distribution': self.latent_state.get_distribution(),
            'belief_summary': self.belief_state.get_belief_summary(),
            'transition_summary': self.transition_model.get_ensemble_summary(),
            'reward_summary': self.reward_model.get_reward_summary()
        }


# ============================================================================
# TESTS
# ============================================================================

def test_gru_belief_state():
    """Test GRU Belief State."""
    print("\n" + "=" * 60)
    print("GRU BELIEF STATE TEST")
    print("=" * 60)
    
    gru = GRUBeliefState(input_dim=32, hidden_dim=64)
    
    # Simulate events
    for i in range(20):
        x = np.random.randn(32) * 0.5
        h = gru.update(x)
        
        if i % 5 == 4:
            print(f"    Step {i+1}: belief_norm={np.linalg.norm(h):.3f}")
    
    summary = gru.get_belief_summary()
    print(f"\n  Summary: {summary}")


def test_stochastic_latent():
    """Test Stochastic Latent State."""
    print("\n" + "=" * 60)
    print("STOCHASTIC LATENT STATE TEST")
    print("=" * 60)
    
    latent = StochasticLatentState(belief_dim=64, latent_dim=16)
    
    # Encode multiple times (should get different samples)
    h = np.random.randn(64) * 0.3
    
    samples = []
    for i in range(5):
        z = latent.encode(h, sample=True)
        samples.append(z)
        if i == 0:
            print(f"    Distribution: mu_norm={np.linalg.norm(latent.mu):.3f}, "
                  f"std_mean={np.mean(np.exp(0.5 * latent.logvar)):.3f}")
    
    # Check variance
    samples = np.array(samples)
    variance = np.var(samples, axis=0)
    print(f"\n  Sample variance: {np.mean(variance):.4f}")
    print(f"  KL divergence: {latent.kl_divergence():.4f}")


def test_ensemble_transition():
    """Test Ensemble Transition Model."""
    print("\n" + "=" * 60)
    print("ENSEMBLE TRANSITION MODEL TEST")
    print("=" * 60)
    
    ensemble = EnsembleTransitionModel(latent_dim=16, belief_dim=64,
                                       action_dim=16, n_models=3)
    
    # Train on synthetic transitions
    z = np.random.randn(16) * 0.3
    h = np.random.randn(64) * 0.3
    
    print("\n  Training ensemble:")
    for i in range(50):
        z_next_true = np.tanh(z * 0.9 + np.random.randn(16) * 0.1)
        action = np.random.randn(16) * 0.1
        
        losses = ensemble.update(z, z_next_true, h, action)
        z = z_next_true
        
        if i % 10 == 9:
            print(f"    Step {i+1}: avg_loss={np.mean(losses):.4f}")
    
    # Test uncertainty
    z_test = np.random.randn(16) * 0.3
    h_test = np.random.randn(64) * 0.3
    uncertainty = ensemble.get_uncertainty(z_test, h_test)
    print(f"\n  Uncertainty: {uncertainty:.4f}")
    print(f"  Summary: {ensemble.get_ensemble_summary()}")


def test_learned_reward():
    """Test Learned Reward Model."""
    print("\n" + "=" * 60)
    print("LEARNED REWARD MODEL TEST")
    print("=" * 60)
    
    reward_model = LearnedRewardModel(latent_dim=16, belief_dim=64)
    
    # Generate training data
    successful = [(np.random.randn(16) * 0.3, np.random.randn(64) * 0.3) for _ in range(20)]
    failed = [(np.random.randn(16) * 0.3, np.random.randn(64) * 0.3) for _ in range(20)]
    
    print("\n  Learning from behavior:")
    reward_model.learn_from_behavior(successful, failed, n_epochs=5)
    
    # Test prediction
    z = np.random.randn(16) * 0.3
    h = np.random.randn(64) * 0.3
    reward = reward_model.predict_reward(z, h)
    print(f"\n  Predicted reward: {reward:.3f}")
    print(f"  Summary: {reward_model.get_reward_summary()}")


def test_cem_planner():
    """Test CEM Planner."""
    print("\n" + "=" * 60)
    print("CEM PLANNER TEST")
    print("=" * 60)
    
    # Setup components
    transition = EnsembleTransitionModel(latent_dim=16, belief_dim=64,
                                         action_dim=16, n_models=2)
    reward = LearnedRewardModel(latent_dim=16, belief_dim=64)
    
    rollout = WorldRolloutEngine(transition, reward, latent_dim=16, belief_dim=64)
    planner = CEMPlanner(rollout, action_dim=16)
    
    # Plan
    z = np.random.randn(16) * 0.3
    h = np.random.randn(64) * 0.3
    
    print("\n  Planning:")
    result = planner.plan(z, h, horizon=5)
    
    print(f"    Best sequence: {result['best_action_sequence'][:3]}...")
    print(f"    Expected reward: {result['expected_reward']:.3f}")
    print(f"    Confidence: {result['confidence']:.3f}")


def test_integrated_core():
    """Test integrated Probabilistic Dynamics Core."""
    print("\n" + "=" * 60)
    print("INTEGRATED PROBABILISTIC DYNAMICS CORE TEST")
    print("=" * 60)
    
    core = ProbabilisticDynamicsCore(
        event_dim=32, latent_dim=16, belief_dim=64, action_dim=16, n_ensemble=3
    )
    
    # Simulate execution
    print("\n  Simulating execution:")
    
    events = [
        ('git_commit', {'files': 3}, 0.8),
        ('ide_focus', {'file': 'main.py'}, 0.7),
        ('interruption', {'reason': 'slack'}, 0.3),
        ('ide_focus', {'file': 'utils.py'}, 0.6),
        ('task_start', {}, 0.5),
        ('git_commit', {'files': 1}, 0.7),
        ('meeting', {'duration': 30}, 0.4),
        ('task_complete', {}, 0.9),
    ]
    
    for event_type, context, outcome in events:
        core.ingest_event(event_type, context, outcome)
    
    # Plan and recommend
    result = core.plan_and_recommend()
    
    print(f"\n  Plan: {result['plan']['best_action_sequence'][:3]}...")
    print(f"  Expected reward: {result['plan']['expected_reward']:.3f}")
    print(f"  Confidence: {result['plan']['confidence']:.3f}")
    print(f"  Uncertainty: {result['uncertainty_gate']['level']}")
    print(f"  Message: {result['uncertainty_gate']['message']}")


if __name__ == "__main__":
    test_gru_belief_state()
    test_stochastic_latent()
    test_ensemble_transition()
    test_learned_reward()
    test_cem_planner()
    test_integrated_core()
    
    print("\n" + "=" * 60)
    print("PHASE 30: MINIMAL PROBABILISTIC DYNAMICS CORE")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Deterministic heuristic simulator
  To: Probabilistic latent dynamics with planning
  
NOT: "world model" or "causal theory"
BUT: compact probabilistic behavior predictor + planning tool

CORE COMPONENTS:

1. GRU BELIEF STATE
   - h_t = GRU(h_{t-1}, x_t)
   - Compressed history, not Transformer
   - Context state of system

2. STOCHASTIC LATENT STATE
   - z_t ~ N(μ(h_t), σ²(h_t))
   - World as distribution, not point
   - Reparameterization trick for sampling

3. ENSEMBLE TRANSITION MODELS
   - 3 models → disagreement = uncertainty
   - Epistemic signal from variance
   - Not single deterministic prediction

4. LEARNED REWARD MODEL
   - r̂ = MLP([z_t, h_t])
   - Learned from behavior, not heuristic
   - Approximates implicit preferences

5. WORLD ROLLOUT ENGINE
   - Trajectory distributions
   - Multiple possible futures
   - Stochastic rollouts

6. CEM PLANNER
   - Search in latent space
   - Cross-Entropy Method optimization
   - Action sequence selection

7. UNCERTAINTY GATE
   - Low → trust plan
   - Medium → reduce horizon
   - High → conservative fallback

DATA FLOW:
  Event → Embedding → GRU belief → Stochastic latent
      → Ensemble transitions → Rollouts → Reward → CEM → Gate

This is:
  decision dynamics simulator, not world theory
  
The system now:
  - Models distribution of futures (not single trajectory)
  - Understands uncertainty (ensemble disagreement)
  - Plans in latent space (CEM search)
  - Adapts to uncertainty (gate fallback)
  
Not symbolic. Not research-grade. Engineering-ready.
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 30 Summary: Minimal Probabilistic Dynamics Core

What was built:
  - GRU-based belief state (sequence memory)
  - Stochastic latent encoder (distributional state)
  - Ensemble transition models (uncertainty via disagreement)
  - Learned reward model (from behavior traces)
  - World rollout engine (trajectory distributions)
  - CEM planner (search in latent space)
  - Uncertainty gate (adaptive planning)

What this is NOT:
  - Not true world model (no physics)
  - Not causal inference (no do-operator)
  - Not semantic understanding
  - Not research-grade RSSM/Dreamer

What this IS:
  - Compact probabilistic behavior predictor
  - Planning tool with uncertainty awareness
  - Engineering-ready hybrid system
  - Bridge from symbolic to learned dynamics

Key capability:
  System now models DISTRIBUTION of future states,
  not single trajectory. Understands "I'm not sure".
"""