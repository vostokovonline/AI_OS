"""
Phase 30 v2: Minimal Learning Loop (Real Version)
==================================================

ARCHITECTURAL SHIFT:
  FROM: "system that simulates behavior"
  TO: "system that learns to compress and predict behavior distributions from data"

KEY CHANGES:
  1. TrajectoryBuffer - real event sequences, not single transitions
  2. Sequence loss - multi-step prediction (k-step), not 1-step MSE
  3. Joint training - encoder + dynamics + reward together
  4. CEM stays - but only as planner over TRAINED model

NOT: random transitions → fake learning
BUT: real trajectories → multi-step consistency → calibrated planning
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import random


# ============================================================================
# 1. TRAJECTORY BUFFER (Real Dataset)
# ============================================================================

class TrajectoryBuffer:
    """
    Stores real event sequences for training.
    
    NOT: single transitions
    BUT: full episodes with events, actions, rewards
    """
    
    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[Dict] = []
        self.max_episodes = max_episodes
        
        # Statistics
        self.total_transitions = 0
    
    def add_episode(self, events: List[np.ndarray],
                   actions: List[np.ndarray],
                   rewards: List[float],
                   metadata: Dict = None):
        """
        Add complete episode to buffer.
        
        episode = {
            'events': [x_0, x_1, ..., x_T],
            'actions': [a_0, a_1, ..., a_T],
            'rewards': [r_0, r_1, ..., r_T]
        }
        """
        assert len(events) == len(actions) == len(rewards)
        
        episode = {
            'events': [np.asarray(e).flatten() for e in events],
            'actions': [np.asarray(a).flatten() for a in actions],
            'rewards': list(rewards),
            'length': len(events),
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        
        self.episodes.append(episode)
        self.total_transitions += len(events)
        
        # Keep buffer bounded
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
    
    def sample_batch(self, batch_size: int = 16,
                    seq_len: int = 10) -> List[Dict]:
        """
        Sample batch of sequences from buffer.
        
        Returns: list of {x_seq, a_seq, r_seq}
        """
        if not self.episodes:
            return []
        
        batch = []
        
        for _ in range(batch_size):
            # Sample random episode
            ep = random.choice(self.episodes)
            
            if ep['length'] <= seq_len:
                # Use full episode
                batch.append({
                    'x_seq': ep['events'],
                    'a_seq': ep['actions'],
                    'r_seq': ep['rewards']
                })
                continue
            
            # Sample random window
            start = random.randint(0, ep['length'] - seq_len - 1)
            
            batch.append({
                'x_seq': ep['events'][start:start + seq_len],
                'a_seq': ep['actions'][start:start + seq_len],
                'r_seq': ep['rewards'][start:start + seq_len]
            })
        
        return batch
    
    def get_buffer_summary(self) -> Dict:
        """Get buffer statistics."""
        if not self.episodes:
            return {'n_episodes': 0, 'total_transitions': 0}
        
        lengths = [ep['length'] for ep in self.episodes]
        
        return {
            'n_episodes': len(self.episodes),
            'total_transitions': self.total_transitions,
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }


# ============================================================================
# 2. MINIMAL WORLD MODEL (Trainable)
# ============================================================================

class MinimalWorldModel:
    """
    Minimal trainable world model.
    
    Components:
      - GRU belief state (h_t = GRU(h_{t-1}, x_t))
      - Stochastic latent (z_t ~ q(z_t | h_t))
      - Transition model (p(z_{t+1} | z_t, h_t, a_t))
      - Reward model (r̂ = MLP([z_t, h_t]))
    
    Trained jointly on sequence prediction + reward prediction.
    """
    
    def __init__(self, event_dim: int = 32, latent_dim: int = 16,
                 belief_dim: int = 64, action_dim: int = 16):
        self.event_dim = event_dim
        self.latent_dim = latent_dim
        self.belief_dim = belief_dim
        self.action_dim = action_dim
        
        # Initialize all weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all model weights."""
        scale = 0.1
        
        # GRU weights
        self.W_zh = np.random.randn(self.belief_dim, self.belief_dim) * scale
        self.W_zx = np.random.randn(self.belief_dim, self.event_dim) * scale
        self.b_z = np.zeros(self.belief_dim)
        
        self.W_rh = np.random.randn(self.belief_dim, self.belief_dim) * scale
        self.W_rx = np.random.randn(self.belief_dim, self.event_dim) * scale
        self.b_r = np.zeros(self.belief_dim)
        
        self.W_hh = np.random.randn(self.belief_dim, self.belief_dim) * scale
        self.W_hx = np.random.randn(self.belief_dim, self.event_dim) * scale
        self.b_h = np.zeros(self.belief_dim)
        
        # Latent encoder
        self.W_mu = np.random.randn(self.latent_dim, self.belief_dim) * scale
        self.b_mu = np.zeros(self.latent_dim)
        self.W_logvar = np.random.randn(self.latent_dim, self.belief_dim) * scale * 0.5
        self.b_logvar = np.zeros(self.latent_dim) - 1.0
        
        # Transition model
        trans_input = self.latent_dim + self.belief_dim + self.action_dim
        self.W_t1 = np.random.randn(64, trans_input) * scale
        self.b_t1 = np.zeros(64)
        self.W_t2 = np.random.randn(self.latent_dim, 64) * scale
        self.b_t2 = np.zeros(self.latent_dim)
        self.W_t_logvar = np.random.randn(self.latent_dim, 64) * scale * 0.5
        self.b_t_logvar = np.zeros(self.latent_dim) - 1.0
        
        # Reward model
        rew_input = self.latent_dim + self.belief_dim
        self.W_r1 = np.random.randn(32, rew_input) * scale
        self.b_r1 = np.zeros(32)
        self.W_r2 = np.random.randn(1, 32) * scale
        self.b_r2 = np.zeros(1)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    # ---- Forward pass ----
    
    def gru_step(self, h: np.ndarray, x: np.ndarray) -> np.ndarray:
        """One GRU step: h_t = GRU(h_{t-1}, x_t)"""
        x = np.asarray(x).flatten()[:self.event_dim]
        if len(x) < self.event_dim:
            x = np.pad(x, (0, self.event_dim - len(x)))
        
        z = self._sigmoid(self.W_zh @ h + self.W_zx @ x + self.b_z)
        r = self._sigmoid(self.W_rh @ h + self.W_rx @ x + self.b_r)
        h_tilde = np.tanh(self.W_hh @ (r * h) + self.W_hx @ x + self.b_h)
        
        return (1 - z) * h + z * h_tilde
    
    def encode_latent(self, h: np.ndarray, sample: bool = True) -> np.ndarray:
        """Encode belief to stochastic latent: z ~ N(μ(h), σ²(h))"""
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        mu = self.W_mu @ h + self.b_mu
        logvar = self.W_logvar @ h + self.b_logvar
        
        if sample:
            std = np.exp(0.5 * logvar)
            epsilon = np.random.randn(self.latent_dim)
            z = mu + std * epsilon
        else:
            z = mu
        
        return z, mu, logvar
    
    def predict_transition(self, z: np.ndarray, h: np.ndarray,
                          a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next latent: p(z_{t+1} | z_t, h_t, a_t)"""
        z = np.asarray(z).flatten()[:self.latent_dim]
        h = np.asarray(h).flatten()[:self.belief_dim]
        a = np.asarray(a).flatten()[:self.action_dim]
        
        if len(a) < self.action_dim:
            a = np.pad(a, (0, self.action_dim - len(a)))
        
        x = np.concatenate([z, h, a])
        
        hidden = np.tanh(self.W_t1 @ x + self.b_t1)
        mu = self.W_t2 @ hidden + self.b_t2
        logvar = self.W_t_logvar @ hidden + self.b_t_logvar
        
        return mu, logvar
    
    def predict_reward(self, z: np.ndarray, h: np.ndarray) -> float:
        """Predict reward: r̂ = MLP([z_t, h_t])"""
        z = np.asarray(z).flatten()[:self.latent_dim]
        h = np.asarray(h).flatten()[:self.belief_dim]
        
        x = np.concatenate([z, h])
        hidden = np.tanh(self.W_r1 @ x + self.b_r1)
        reward = self.W_r2 @ hidden + self.b_r2
        
        return float(reward[0])
    
    # ---- Rollout ----
    
    def rollout_sequence(self, x_seq: List[np.ndarray],
                        a_seq: List[np.ndarray],
                        sample: bool = True) -> Dict:
        """
        Rollout model over sequence.
        
        Returns: latents, mus, logvars, rewards, beliefs
        """
        h = np.zeros(self.belief_dim)
        
        latents = []
        mus = []
        logvars = []
        rewards = []
        beliefs = []
        
        for t in range(len(x_seq)):
            # Update belief
            h = self.gru_step(h, x_seq[t])
            
            # Encode latent
            z, mu, logvar = self.encode_latent(h, sample=sample)
            
            # Predict reward
            r = self.predict_reward(z, h)
            
            latents.append(z)
            mus.append(mu)
            logvars.append(logvar)
            rewards.append(r)
            beliefs.append(h.copy())
        
        return {
            'latents': latents,
            'mus': mus,
            'logvars': logvars,
            'rewards': rewards,
            'beliefs': beliefs
        }


# ============================================================================
# 3. SEQUENCE LOSS (Multi-step consistency)
# ============================================================================

def compute_sequence_loss(model: MinimalWorldModel,
                         batch: List[Dict],
                         k_steps: int = 5) -> float:
    """
    Compute multi-step sequence prediction loss.
    
    NOT: 1-step MSE
    BUT: k-step latent consistency loss
    
    For each timestep t:
      1. Encode observation → z_t
      2. Predict k steps forward → z_{t+1:t+k}
      3. Compare with actual encoded latents
    """
    total_loss = 0.0
    n_samples = 0
    
    for sample in batch:
        x_seq = sample['x_seq']
        a_seq = sample['a_seq']
        
        seq_len = len(x_seq)
        
        if seq_len < k_steps + 2:
            continue
        
        # Full forward pass to get true latents
        rollout = model.rollout_sequence(x_seq, a_seq, sample=False)
        true_mus = rollout['mus']
        
        # Multi-step prediction loss
        for t in range(seq_len - k_steps):
            # Current state
            z_t = true_mus[t]
            h_t = rollout['beliefs'][t]
            
            # Predict k steps forward
            z_pred = z_t.copy()
            h_pred = h_t.copy()
            
            for i in range(k_steps):
                a = a_seq[t + i]
                
                # Predict next
                mu_pred, logvar_pred = model.predict_transition(
                    z_pred, h_pred, a
                )
                
                # True latent at t+i+1
                z_true = true_mus[t + i + 1]
                
                # Multi-step consistency loss
                loss_step = np.mean((mu_pred - z_true) ** 2)
                
                # Uncertainty regularization
                loss_unc = np.mean(np.exp(logvar_pred))
                
                total_loss += loss_step + 0.01 * loss_unc
                
                # Update prediction for next step
                z_pred = mu_pred
                h_pred = h_pred * 0.9 + np.pad(z_pred, (0, model.belief_dim - model.latent_dim)) * 0.1
            
            n_samples += 1
    
    return total_loss / max(1, n_samples)


def compute_reward_loss(model: MinimalWorldModel,
                       batch: List[Dict]) -> float:
    """
    Compute reward prediction loss.
    
    r̂ = MLP([z_t, h_t]) vs observed rewards
    """
    total_loss = 0.0
    n_samples = 0
    
    for sample in batch:
        x_seq = sample['x_seq']
        r_seq = sample['r_seq']
        
        h = np.zeros(model.belief_dim)
        
        for t in range(len(x_seq)):
            # Update belief
            h = model.gru_step(h, x_seq[t])
            
            # Encode latent (deterministic for reward)
            z, _, _ = model.encode_latent(h, sample=False)
            
            # Predict reward
            r_pred = model.predict_reward(z, h)
            
            # MSE loss
            loss = (r_pred - r_seq[t]) ** 2
            total_loss += loss
            n_samples += 1
    
    return total_loss / max(1, n_samples)


# ============================================================================
# 4. GRADIENT UPDATE (Manual backprop for numpy)
# ============================================================================

class SimpleOptimizer:
    """
    Simple SGD optimizer for numpy models.
    
    NOT: PyTorch autograd
    BUT: manual gradient computation + parameter updates
    """
    
    def __init__(self, model: MinimalWorldModel,
                 learning_rate: float = 0.001):
        self.model = model
        self.lr = learning_rate
        
        # All trainable parameters
        self.params = [
            'W_zh', 'W_zx', 'b_z',
            'W_rh', 'W_rx', 'b_r',
            'W_hh', 'W_hx', 'b_h',
            'W_mu', 'b_mu',
            'W_logvar', 'b_logvar',
            'W_t1', 'b_t1', 'W_t2', 'b_t2',
            'W_t_logvar', 'b_t_logvar',
            'W_r1', 'b_r1', 'W_r2', 'b_r2'
        ]
        
        # Gradient accumulation
        self.grads = {p: None for p in self.params}
    
    def zero_grad(self):
        """Reset gradients."""
        self.grads = {p: np.zeros_like(getattr(self.model, p))
                     for p in self.params}
    
    def step(self):
        """Apply gradients."""
        for p in self.params:
            if self.grads[p] is not None:
                param = getattr(self.model, p)
                grad = self.grads[p]
                
                # Gradient clipping
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1.0:
                    grad = grad / grad_norm
                
                # Update
                setattr(self.model, p, param - self.lr * grad)
    
    def accumulate_grad(self, param_name: str, grad: np.ndarray):
        """Accumulate gradient for parameter."""
        if self.grads[param_name] is None:
            self.grads[param_name] = grad.copy()
        else:
            self.grads[param_name] += grad


# ============================================================================
# 5. TRAINING STEP (Joint training)
# ============================================================================

def train_step(model: MinimalWorldModel,
              optimizer: SimpleOptimizer,
              batch: List[Dict],
              k_steps: int = 5,
              reward_weight: float = 0.5) -> Dict:
    """
    Single training step with joint loss.
    
    total_loss = dynamics_loss + reward_weight * reward_loss
    """
    optimizer.zero_grad()
    
    # Compute losses
    dynamics_loss = compute_sequence_loss(model, batch, k_steps)
    reward_loss = compute_reward_loss(model, batch)
    
    total_loss = dynamics_loss + reward_weight * reward_loss
    
    # Approximate gradient update (finite differences for simplicity)
    # In production, use proper backprop or JAX/PyTorch
    _approximate_gradients(model, optimizer, batch, k_steps, reward_weight)
    
    optimizer.step()
    
    return {
        'total_loss': total_loss,
        'dynamics_loss': dynamics_loss,
        'reward_loss': reward_loss
    }


def _approximate_gradients(model: MinimalWorldModel,
                          optimizer: SimpleOptimizer,
                          batch: List[Dict],
                          k_steps: int,
                          reward_weight: float,
                          epsilon: float = 1e-4):
    """
    Approximate gradients using finite differences.
    
    NOT: proper backprop
    BUT: simple gradient estimation for numpy implementation
    """
    # Compute base loss
    base_dynamics = compute_sequence_loss(model, batch, k_steps)
    base_reward = compute_reward_loss(model, batch)
    base_loss = base_dynamics + reward_weight * base_reward
    
    # For each parameter, estimate gradient
    for param_name in optimizer.params:
        param = getattr(model, param_name)
        
        # Perturb parameter
        perturbation = np.random.randn(*param.shape) * epsilon
        setattr(model, param_name, param + perturbation)
        
        # Compute perturbed loss
        pert_dynamics = compute_sequence_loss(model, batch, k_steps)
        pert_reward = compute_reward_loss(model, batch)
        pert_loss = pert_dynamics + reward_weight * pert_reward
        
        # Estimate gradient
        grad = (pert_loss - base_loss) / epsilon * perturbation / (epsilon * param.size)
        
        # Restore parameter
        setattr(model, param_name, param)
        
        # Accumulate gradient
        optimizer.accumulate_grad(param_name, grad)


# ============================================================================
# 6. TRAINING LOOP (Real)
# ============================================================================

def training_loop(model: MinimalWorldModel,
                 buffer: TrajectoryBuffer,
                 epochs: int = 100,
                 batch_size: int = 8,
                 seq_len: int = 10,
                 k_steps: int = 5,
                 learning_rate: float = 0.001,
                 verbose: bool = True) -> List[Dict]:
    """
    Real training loop.
    
    Trains model on trajectory data with:
      - Multi-step sequence loss
      - Reward prediction loss
      - Joint optimization
    """
    optimizer = SimpleOptimizer(model, learning_rate)
    history = []
    
    for epoch in range(epochs):
        # Sample batch
        batch = buffer.sample_batch(batch_size, seq_len)
        
        if not batch:
            if verbose:
                print(f"  Epoch {epoch}: No data in buffer")
            continue
        
        # Train step
        logs = train_step(model, optimizer, batch, k_steps)
        
        history.append({
            'epoch': epoch,
            **logs
        })
        
        # Periodic evaluation
        if verbose and epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"total: {logs['total_loss']:.4f} | "
                  f"dynamics: {logs['dynamics_loss']:.4f} | "
                  f"reward: {logs['reward_loss']:.4f}")
        
        # Rollout sanity check
        if verbose and epoch % 25 == 0:
            _sanity_check_rollout(model)
    
    return history


def _sanity_check_rollout(model: MinimalWorldModel):
    """Check if model produces stable rollouts."""
    h = np.zeros(model.belief_dim)
    z = np.zeros(model.latent_dim)
    
    trajectory_norms = []
    
    for _ in range(10):
        a = np.zeros(model.action_dim)
        mu, logvar = model.predict_transition(z, h, a)
        
        z = mu
        h = h * 0.9 + np.pad(z, (0, model.belief_dim - model.latent_dim)) * 0.1
        
        trajectory_norms.append(np.linalg.norm(z))
    
    stability = "stable" if np.std(trajectory_norms) < 1.0 else "unstable"
    print(f"    Rollout check: norms={[f'{n:.2f}' for n in trajectory_norms[:5]]}... ({stability})")


# ============================================================================
# 7. CEM PLANNER (Over trained model)
# ============================================================================

def cem_planning(model: MinimalWorldModel,
                z0: np.ndarray,
                h0: np.ndarray,
                actions: List[str],
                action_embeddings: Dict[str, np.ndarray],
                horizon: int = 5,
                n_samples: int = 30,
                n_elites: int = 8,
                n_iterations: int = 4) -> Dict:
    """
    CEM planner over trained world model.
    
    NOT: planning over random model
    BUT: planning over TRAINED dynamics
    """
    n_actions = len(actions)
    action_probs = np.ones(n_actions) / n_actions
    
    best_sequence = None
    best_reward = -float('inf')
    
    for iteration in range(n_iterations):
        # Sample action sequences
        sequences = []
        for _ in range(n_samples):
            seq = np.random.choice(actions, size=horizon, p=action_probs).tolist()
            sequences.append(seq)
        
        # Evaluate sequences
        rewards = []
        for seq in sequences:
            total_r = 0.0
            z = z0.copy()
            h = h0.copy()
            
            for action_name in seq:
                a = action_embeddings.get(action_name, np.zeros(16))
                
                mu, logvar = model.predict_transition(z, h, a)
                r = model.predict_reward(mu, h)
                
                total_r += r
                z = mu
                h = h * 0.9 + np.pad(z, (0, model.belief_dim - model.latent_dim)) * 0.1
            
            rewards.append(total_r)
        
        # Select elites
        elite_indices = np.argsort(rewards)[-n_elites:]
        elite_sequences = [sequences[i] for i in elite_indices]
        
        # Update best
        best_idx = elite_indices[-1]
        if rewards[best_idx] > best_reward:
            best_reward = rewards[best_idx]
            best_sequence = sequences[best_idx]
        
        # Update action distribution
        action_counts = np.zeros(n_actions)
        for seq in elite_sequences:
            for i, action_name in enumerate(seq):
                if i < horizon:
                    idx = actions.index(action_name)
                    action_counts[idx] += 1
        
        # Smooth update
        if action_counts.sum() > 0:
            action_probs = 0.7 * (action_counts / action_counts.sum()) + 0.3 * action_probs
            action_probs = action_probs / action_probs.sum()
    
    return {
        'best_action_sequence': best_sequence or [],
        'expected_reward': best_reward,
        'confidence': 1.0 - (np.std([rewards[i] for i in elite_indices]) /
                           (np.mean([rewards[i] for i in elite_indices]) + 1e-8))
        if elite_indices.size > 0 else 0.0
    }


# ============================================================================
# 8. INTEGRATED TRAINABLE SYSTEM
# ============================================================================

class TrainableDynamicsSystem:
    """
    Complete Phase 30 v2: Trainable Probabilistic Dynamics Core.
    
    DATA FLOW:
      Real events → TrajectoryBuffer → Training loop
          ↓
      Trained model → CEM planning → Recommendations
    """
    
    def __init__(self, event_dim: int = 32, latent_dim: int = 16,
                 belief_dim: int = 64, action_dim: int = 16):
        # Model
        self.model = MinimalWorldModel(
            event_dim=event_dim, latent_dim=latent_dim,
            belief_dim=belief_dim, action_dim=action_dim
        )
        
        # Buffer
        self.buffer = TrajectoryBuffer(max_episodes=1000)
        
        # Action vocabulary
        self.actions = [
            'deep_focus', 'context_switch', 'take_break',
            'continue_current', 'simplify_task', 'collaborate', 'explore'
        ]
        
        # Action embeddings
        self.action_embeddings: Dict[str, np.ndarray] = {}
        for action in self.actions:
            self.action_embeddings[action] = np.random.randn(action_dim) * 0.1
        
        # State
        self.current_h = np.zeros(belief_dim)
        self.current_z = np.zeros(latent_dim)
        
        # Training state
        self.is_trained = False
        self.training_history: List[Dict] = []
    
    def record_event(self, event: np.ndarray, action: np.ndarray,
                    reward: float):
        """Record single transition (accumulates into episodes)."""
        # For simplicity, treat each event as mini-episode
        # In production, accumulate into proper episodes
        self.buffer.add_episode(
            events=[event],
            actions=[action],
            rewards=[reward]
        )
    
    def record_episode(self, events: List[np.ndarray],
                      actions: List[np.ndarray],
                      rewards: List[float]):
        """Record complete episode."""
        self.buffer.add_episode(events, actions, rewards)
    
    def train(self, epochs: int = 50, batch_size: int = 8,
             seq_len: int = 10, verbose: bool = True):
        """Train model on buffer data."""
        if self.buffer.get_buffer_summary()['n_episodes'] == 0:
            if verbose:
                print("  No data in buffer. Add episodes first.")
            return
        
        self.training_history = training_loop(
            self.model, self.buffer,
            epochs=epochs, batch_size=batch_size,
            seq_len=seq_len, verbose=verbose
        )
        
        self.is_trained = True
    
    def plan(self, horizon: int = 5) -> Dict:
        """Plan using CEM over trained model."""
        if not self.is_trained:
            return {
                'error': 'Model not trained yet',
                'recommendation': 'Record data and call train() first'
            }
        
        return cem_planning(
            self.model, self.current_z, self.current_h,
            self.actions, self.action_embeddings,
            horizon=horizon
        )
    
    def get_system_summary(self) -> Dict:
        """Get system summary."""
        return {
            'is_trained': self.is_trained,
            'buffer': self.buffer.get_buffer_summary(),
            'training_epochs': len(self.training_history),
            'final_loss': self.training_history[-1] if self.training_history else None
        }


# ============================================================================
# TESTS
# ============================================================================

def test_trajectory_buffer():
    """Test TrajectoryBuffer."""
    print("\n" + "=" * 60)
    print("TRAJECTORY BUFFER TEST")
    print("=" * 60)
    
    buffer = TrajectoryBuffer(max_episodes=100)
    
    # Add synthetic episodes
    for ep in range(20):
        length = random.randint(15, 30)
        events = [np.random.randn(32) * 0.5 for _ in range(length)]
        actions = [np.random.randn(16) * 0.3 for _ in range(length)]
        rewards = [random.uniform(0, 1) for _ in range(length)]
        
        buffer.add_episode(events, actions, rewards)
    
    summary = buffer.get_buffer_summary()
    print(f"\n  Buffer summary: {summary}")
    
    # Sample batch
    batch = buffer.sample_batch(batch_size=4, seq_len=10)
    print(f"  Sampled batch: {len(batch)} sequences")
    print(f"  First sequence length: {len(batch[0]['x_seq'])}")


def test_model_forward():
    """Test model forward pass."""
    print("\n" + "=" * 60)
    print("MODEL FORWARD PASS TEST")
    print("=" * 60)
    
    model = MinimalWorldModel(event_dim=32, latent_dim=16,
                             belief_dim=64, action_dim=16)
    
    # Single step
    h = np.zeros(64)
    x = np.random.randn(32) * 0.5
    
    h = model.gru_step(h, x)
    z, mu, logvar = model.encode_latent(h)
    
    a = np.random.randn(16) * 0.3
    mu_next, logvar_next = model.predict_transition(z, h, a)
    r = model.predict_reward(z, h)
    
    print(f"\n  Belief norm: {np.linalg.norm(h):.3f}")
    print(f"  Latent norm: {np.linalg.norm(z):.3f}")
    print(f"  Predicted latent norm: {np.linalg.norm(mu_next):.3f}")
    print(f"  Predicted reward: {r:.3f}")


def test_sequence_loss():
    """Test sequence loss computation."""
    print("\n" + "=" * 60)
    print("SEQUENCE LOSS TEST")
    print("=" * 60)
    
    model = MinimalWorldModel(event_dim=32, latent_dim=16,
                             belief_dim=64, action_dim=16)
    
    # Create batch
    batch = []
    for _ in range(4):
        length = 15
        events = [np.random.randn(32) * 0.5 for _ in range(length)]
        actions = [np.random.randn(16) * 0.3 for _ in range(length)]
        rewards = [random.uniform(0, 1) for _ in range(length)]
        
        batch.append({
            'x_seq': events,
            'a_seq': actions,
            'r_seq': rewards
        })
    
    # Compute losses
    dyn_loss = compute_sequence_loss(model, batch, k_steps=3)
    rew_loss = compute_reward_loss(model, batch)
    
    print(f"\n  Dynamics loss (3-step): {dyn_loss:.4f}")
    print(f"  Reward loss: {rew_loss:.4f}")
    print(f"  Total: {dyn_loss + 0.5 * rew_loss:.4f}")


def test_training_loop():
    """Test full training loop."""
    print("\n" + "=" * 60)
    print("TRAINING LOOP TEST")
    print("=" * 60)
    
    model = MinimalWorldModel(event_dim=32, latent_dim=16,
                             belief_dim=64, action_dim=16)
    buffer = TrajectoryBuffer()
    
    # Generate training data
    print("\n  Generating training data:")
    for ep in range(30):
        length = random.randint(20, 40)
        events = [np.random.randn(32) * 0.5 for _ in range(length)]
        actions = [np.random.randn(16) * 0.3 for _ in range(length)]
        
        # Rewards: higher when events are consistent
        rewards = []
        for i in range(length):
            consistency = 1.0 / (1.0 + np.std([e[0] for e in events[max(0,i-3):i+1]]))
            rewards.append(consistency * random.uniform(0.5, 1.0))
        
        buffer.add_episode(events, actions, rewards)
    
    print(f"  Buffer: {buffer.get_buffer_summary()}")
    
    # Train
    print("\n  Training:")
    history = training_loop(
        model, buffer,
        epochs=50, batch_size=8, seq_len=12,
        k_steps=3, learning_rate=0.001,
        verbose=True
    )
    
    # Check improvement
    if len(history) > 5:
        first_loss = history[0]['total_loss']
        last_loss = history[-1]['total_loss']
        print(f"\n  Loss improvement: {first_loss:.4f} → {last_loss:.4f}")
        print(f"  Improvement: {(1 - last_loss/first_loss)*100:.1f}%")


def test_planning_after_training():
    """Test CEM planning after training."""
    print("\n" + "=" * 60)
    print("PLANNING AFTER TRAINING TEST")
    print("=" * 60)
    
    system = TrainableDynamicsSystem(
        event_dim=32, latent_dim=16, belief_dim=64, action_dim=16
    )
    
    # Generate data
    print("\n  Recording episodes:")
    for ep in range(20):
        length = random.randint(15, 25)
        events = [np.random.randn(32) * 0.5 for _ in range(length)]
        actions = [np.random.randn(16) * 0.3 for _ in range(length)]
        rewards = [random.uniform(0.3, 0.9) for _ in range(length)]
        
        system.record_episode(events, actions, rewards)
    
    print(f"  Buffer: {system.buffer.get_buffer_summary()}")
    
    # Train
    print("\n  Training model:")
    system.train(epochs=30, batch_size=6, seq_len=10, verbose=True)
    
    # Plan
    print("\n  Planning:")
    plan = system.plan(horizon=5)
    
    if 'error' in plan:
        print(f"  Error: {plan['error']}")
    else:
        print(f"  Best sequence: {plan['best_action_sequence']}")
        print(f"  Expected reward: {plan['expected_reward']:.3f}")
        print(f"  Confidence: {plan['confidence']:.3f}")


if __name__ == "__main__":
    test_trajectory_buffer()
    test_model_forward()
    test_sequence_loss()
    test_training_loop()
    test_planning_after_training()
    
    print("\n" + "=" * 60)
    print("PHASE 30 v2: MINIMAL TRAINING LOOP")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  FROM: "system that simulates behavior"
  TO: "system that learns to compress and predict behavior distributions from data"

KEY CHANGES:

1. TRAJECTORY BUFFER
   - Real event sequences, not single transitions
   - Episodes with events, actions, rewards
   - Batch sampling for training

2. SEQUENCE LOSS (Multi-step)
   - NOT: 1-step MSE
   - BUT: k-step latent consistency loss
   - Forces model to predict multiple steps ahead

3. JOINT TRAINING
   - dynamics_loss + reward_weight * reward_loss
   - All components trained together
   - Encoder, transition, reward optimized jointly

4. CEM PLANNER
   - Plans over TRAINED model
   - Not over random-initialized weights
   - Meaningful action sequences

DATA FLOW:
  Real events → TrajectoryBuffer → Training loop
      ↓
  Trained model → CEM planning → Recommendations

This is:
  data-driven latent dynamics learner
  
NOT:
  engineered cognition simulator
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 30 v2 Summary: Minimal Training Loop

What was built:
  - TrajectoryBuffer for real event sequences
  - Multi-step sequence loss (k-step prediction)
  - Joint training (encoder + dynamics + reward)
  - CEM planner over trained model

Key shift:
  FROM: random transitions → fake learning
  TO: real trajectories → multi-step consistency → calibrated planning

The system now:
  - Learns from trajectory data
  - Enforces multi-step consistency
  - Rewards grounded in observed sequences
  - Dynamics constrained by data

This is the first real "learning world dynamics" system.
"""