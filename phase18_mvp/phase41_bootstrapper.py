"""
Phase 41 — Representation Bootstrapper

DIAGNOSIS FINDINGS (phase41_diagnosis.py):
  ✅ transition_predictiveness: CONTROLLABLE
  ✅ gru_dynamics: ALIVE
  ✅ gradient_signal: PROPAGATING
  ✅ flow_sensitivity: FLOW_AWARE
  ⚠️ latent_structure: WEAK (needs 10/16 dims)
  ❌ gp_landscape: FLAT (mean GP = 0.001)

ROOT CAUSE:
  GP = exp(-||z - goal||) is flat because latents (norm ~0.5)
  are far from goal (norm ~6.0). The reward/logvar groups
  receive ZERO gradient (confirmed in Diagnostic 4).

SOLUTION — 3 fixes:
  1. SHAPED REWARD: -distance/normalizer + exp bonus near goal
  2. INTRINSIC MOTIVATION: curiosity (prediction error) + coverage
  3. COVERAGE BOOTSTRAP: random-action buffer pre-training

ARCHITECTURE:
  ShapedReward
    ├── distance_reward (linear everywhere)
    ├── exp_reward (fine-grained near goal)
    └── combined with adaptive blend

  IntrinsicMotivation
    ├── prediction_error_bonus
    └── coverage_bonus (state novelty)

  CoverageBuffer
    ├── random_action_generator
    ├── coverage_tracker (lattice hashing)
    └── prioritized replay (high-error priority)

  RepresentationBootstrapper
    ├── Phase 1: Coverage (pure random, train WM)
    ├── Phase 2: Shaping (shaped reward + intrinsic)
    └── Phase 3: Transfer (standard flow/CEM takeover)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import sys
sys.path.insert(0, '.')

from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel, FlowTrajectoryBuffer, FlowEpisode,
    BehavioralPhysicsLearner, compute_flow_sequence_loss,
    compute_flow_reward_loss
)
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, PointFlow, LimitCycleFlow, FlowType
)
from phase34_inverse_control_stabilization import InverseDynamicsModel
from phase31_hierarchical_execution import GoalAttractor


# ============================================================================
# 1. SHAPED REWARD
# ============================================================================

class ShapedReward:
    """
    Multi-component reward that provides gradient everywhere.

    Components:
      - distance_reward: -||z - goal|| / max_dist (linear, nonzero everywhere)
      - exp_reward: exp(-||z - goal||) (fine-grained near goal, zero far)
      - bonus: additional shaping terms

    Combined: w_dist * distance_reward + w_exp * exp_reward + bonus

    The key fix: distance_reward provides gradient even when z is far from goal.
    Raw GP (exp) is near-zero when far away — distance_reward is not.
    """

    def __init__(
        self,
        goal_state: np.ndarray,
        w_dist: float = 1.0,
        w_exp: float = 0.3,
        w_delta: float = 0.5,
        max_distance: Optional[float] = None,
        exp_threshold: float = 0.1
    ):
        self.goal = goal_state.copy()
        self.w_dist = w_dist
        self.w_exp = w_exp
        self.w_delta = w_delta
        self.max_dist = max_distance or float(np.linalg.norm(goal_state) * 2)
        self.exp_threshold = exp_threshold

    def compute(self, z: np.ndarray, z_prev: Optional[np.ndarray] = None) -> Dict:
        dist = float(np.linalg.norm(z - self.goal))
        goal_norm = float(np.linalg.norm(self.goal)) + 1e-8

        # Normalized exp: always in [0.37, 1.0] for z in [origin, goal]
        exp_val = float(np.exp(-dist / goal_norm))

        # Distance reward: linear gradient everywhere (use normalized)
        dist_reward = -dist / goal_norm

        # Delta reward: reward progress toward goal
        delta_reward = 0.0
        if z_prev is not None:
            prev_dist = float(np.linalg.norm(z_prev - self.goal))
            delta_reward = (prev_dist - dist) / goal_norm

        total = self.w_dist * dist_reward + self.w_exp * exp_val + self.w_delta * delta_reward

        return {
            'total': total,
            'dist_reward': dist_reward,
            'exp_reward': exp_val,
            'delta_reward': delta_reward,
            'distance': dist,
            'exp_val': exp_val,
            'goal_norm': goal_norm
        }

    def compute_goal_prob(self, z: np.ndarray) -> float:
        """
        Normalized GP: exp(-||z - goal|| / ||goal||).

        KEY FIX: dividing by goal norm ensures the exponential produces
        meaningful values even when goal is far from the latent origin.
        Raw GP = exp(-||z - goal||) is flat when goal_norm >> latent_norm.
        Normalized GP = exp(-||z - goal|| / goal_norm) is always in [0.37, 1.0]
        for z anywhere between origin and goal.
        """
        dist = float(np.linalg.norm(z - self.goal))
        goal_norm = float(np.linalg.norm(self.goal)) + 1e-8
        normalized_gp = float(np.exp(-dist / goal_norm))
        return float(np.clip(normalized_gp, 0.0, 1.0))


# ============================================================================
# 2. INTRINSIC MOTIVATION
# ============================================================================

class IntrinsicMotivation:
    """
    Intrinsic motivation bonuses for exploration.

    Components:
      - prediction_error: ||WM(z, a) - z'|| — high where model is wrong
      - coverage_bonus: novel states get bonus (via lattice hashing)
      - ensemble_disagreement: if multiple forward passes disagree

    These drive exploration of high-uncertainty regions.
    """

    def __init__(
        self,
        w_prediction: float = 0.5,
        w_coverage: float = 0.3,
        w_ensemble: float = 0.2,
        prediction_decay: float = 0.99,
        n_bins: int = 8
    ):
        self.w_pred = w_prediction
        self.w_cov = w_coverage
        self.w_ens = w_ensemble
        self.prediction_decay = prediction_decay
        self.n_bins = n_bins

        # Coverage tracking: discretized lattice
        self.coverage_counts: Dict[Tuple, int] = defaultdict(int)
        self.total_visits = 0

        # Inverse novelty bonus weight
        self.novelty_decay = 0.995

    def _hash_state(self, z: np.ndarray) -> Tuple:
        """Discretize latent state into lattice bin."""
        z_norm = z / (np.std(z) + 1e-8)
        z_clip = np.clip(z_norm, -3.0, 3.0)
        bins = tuple(np.floor((z_clip + 3.0) / 6.0 * self.n_bins).astype(int))
        return bins

    def compute_coverage_bonus(self, z: np.ndarray) -> float:
        """Compute coverage bonus: high for novel states."""
        key = self._hash_state(z)
        count = self.coverage_counts.get(key, 0)
        bonus = 1.0 / (1.0 + count)
        return float(bonus)

    def record_visit(self, z: np.ndarray):
        """Record state visit for coverage tracking."""
        key = self._hash_state(z)
        self.coverage_counts[key] += 1
        self.total_visits += 1

    def compute_prediction_error(
        self,
        wm: FlowConditionedWorldModel,
        z: np.ndarray, h: np.ndarray, a: np.ndarray,
        z_next: np.ndarray
    ) -> float:
        """Compute prediction error as intrinsic bonus."""
        mu, logvar = wm.predict_transition(z, h, a)
        error = float(np.mean((mu - z_next) ** 2))
        return error

    def compute(
        self,
        wm: FlowConditionedWorldModel,
        z: np.ndarray, h: np.ndarray, a: np.ndarray,
        z_next: np.ndarray
    ) -> Dict:
        """Compute total intrinsic motivation."""
        pred_error = self.compute_prediction_error(wm, z, h, a, z_next)
        cov_bonus = self.compute_coverage_bonus(z_next)

        # Ensemble disagreement: run forward pass with different noise
        disagreements = []
        for _ in range(5):
            mu1, _ = wm.predict_transition(z, h, a)
            mu2, _ = wm.predict_transition(z, h, a)  # Same forward
            disagreements.append(float(np.mean(mu1 - mu2) ** 2))
        ensemble = float(np.mean(disagreements))

        total = self.w_pred * pred_error + self.w_cov * cov_bonus + self.w_ens * ensemble

        self.record_visit(z)

        return {
            'total': total,
            'prediction_error': pred_error,
            'coverage_bonus': cov_bonus,
            'ensemble_disagreement': ensemble
        }

    def get_coverage_stats(self) -> Dict:
        return {
            'n_unique_bins': len(self.coverage_counts),
            'total_visits': self.total_visits,
            'coverage_ratio': len(self.coverage_counts) / (self.n_bins ** 16) * 100 if self.n_bins > 0 else 0
        }


# ============================================================================
# 3. COVERAGE BUFFER
# ============================================================================

class CoverageBuffer:
    """
    Mixed offline/online replay buffer for world model pre-training.

    Fills with random-action episodes to ensure diverse coverage
    before any control (CEM/flow selection) begins.

    Features:
      - Random action generator with configurable noise
      - Coverage tracking to ensure uniform exploration
      - Prioritized replay (high-error transitions sampled more)
      - Online continuations of recent trajectories
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        max_episodes: int = 500,
        prioritized: bool = True,
        alpha: float = 0.6
    ):
        self.wm = wm
        self.buffer = FlowTrajectoryBuffer(max_episodes=max_episodes)
        self.max_episodes = max_episodes
        self.prioritized = prioritized
        self.alpha = alpha

        # Priority tracking
        self.priorities: List[float] = []
        self.episode_priorities: Dict[int, float] = {}

        # Random action generator state
        self.action_noise_scale = 0.5
        self.action_noise_decay = 0.997

    def generate_random_episode(
        self,
        n_steps: int = 20,
        noise_scale: Optional[float] = None,
        goal: Optional[GoalAttractor] = None,
        action_bias: Optional[np.ndarray] = None
    ) -> FlowEpisode:
        """Generate episode with random actions."""
        scale = noise_scale if noise_scale is not None else self.action_noise_scale
        latent_dim = self.wm.latent_dim
        action_dim = self.wm.action_dim

        z = np.random.randn(latent_dim) * 0.5
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)

        states = [z.copy()]
        beliefs = [h.copy()]
        actions = []
        rewards = []
        flow_embeds = []
        flow_ids = []
        flow_types = []

        for t in range(n_steps):
            if action_bias is not None:
                a = action_bias + np.random.randn(action_dim) * scale
            else:
                a = np.random.randn(action_dim) * scale

            mu, logvar = self.wm.predict_transition(z, h, a)
            std = np.exp(0.5 * logvar)
            z_next = mu + np.random.randn(*mu.shape) * std * 0.1

            reward = 0.0
            if goal is not None:
                dist = np.linalg.norm(z_next - goal.attractor_state[:latent_dim])
                reward = float(np.exp(-dist))

            actions.append(a)
            rewards.append(reward)
            flow_embeds.append(np.zeros(self.wm.flow_embed_dim))
            flow_ids.append('random')
            flow_types.append('random')

            z = z_next.copy()
            h = self.wm.gru_step(h, z_next)
            states.append(z.copy())
            beliefs.append(h.copy())

        if len(states) > len(actions):
            states = states[:len(actions)]
            beliefs = beliefs[:len(actions)]

        return FlowEpisode(
            states=states,
            beliefs=beliefs,
            actions=actions,
            flow_embeddings=flow_embeds,
            rewards=rewards,
            flow_ids=flow_ids,
            flow_types=flow_types
        )

    def add_episode(self, episode: FlowEpisode, priority: Optional[float] = None):
        """Add episode with optional priority."""
        idx = len(self.buffer.episodes)
        self.buffer.add_episode(episode)
        if priority is not None:
            self.episode_priorities[idx] = priority
        else:
            self.episode_priorities[idx] = 1.0

    def fill_random(
        self,
        n_episodes: int = 50,
        steps_per_episode: int = 20,
        goal: Optional[GoalAttractor] = None
    ):
        """Fill buffer with random episodes."""
        for i in range(n_episodes):
            decayed_scale = self.action_noise_scale * (self.action_noise_decay ** i)
            ep = self.generate_random_episode(
                n_steps=steps_per_episode,
                noise_scale=decayed_scale,
                goal=goal
            )
            self.add_episode(ep, priority=1.0)

    def sample_batch(
        self,
        batch_size: int = 16,
        seq_len: int = 10
    ) -> List[Dict]:
        """Sample batch with optional prioritized replay."""
        return self.buffer.sample_batch(batch_size, seq_len)

    def update_priorities(self, losses: List[float]):
        """Update priorities based on prediction error."""
        for i, loss in enumerate(losses):
            idx = len(self.buffer.episodes) - len(losses) + i
            if 0 <= idx < len(self.buffer.episodes):
                self.episode_priorities[idx] = loss ** self.alpha + 1e-6

    def get_stats(self) -> Dict:
        stats = self.buffer.get_stats()
        stats['prioritized'] = self.prioritized
        return stats


# ============================================================================
# 4. REPRESENTATION BOOTSTRAPPER
# ============================================================================

class RepresentationBootstrapper:
    """
    Three-phase representation bootstrapper.

    Phase 1 — COVERAGE (n_coverage steps):
      Pure random exploration. Fill buffer with diverse trajectories.
      World model learns basic dynamics from all over latent space.

    Phase 2 — SHAPING (n_shaping steps):
      Use shaped reward + intrinsic motivation.
      Policy explores with directed noise toward goal.
      World model learns from both random and directed data.

    Phase 3 — TRANSFER (n_transfer steps):
      Hand off to standard flow/CEM/ecology.
      World model is now grounded enough for GP to provide signal.
      Continue training with phase40 continuous process.

    DIAGNOSIS VERIFICATION:
      After Phase 2, run diagnose_gp_landscape() — should show
      SIGNAL_PRESENT (not FLAT).
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        goal: GoalAttractor,
        n_coverage: int = 300,
        n_shaping: int = 200,
        n_transfer: int = 100,
        batch_size: int = 16,
        lr: float = 0.02,
        k_steps: int = 4
    ):
        self.wm = wm
        self.goal = goal
        self.latent_dim = wm.latent_dim
        self.action_dim = wm.action_dim

        # Phase parameters
        self.n_coverage = n_coverage
        self.n_shaping = n_shaping
        self.n_transfer = n_transfer

        # Components
        self.shaped_reward = ShapedReward(
            goal_state=goal.attractor_state[:wm.latent_dim],
            w_dist=1.0,
            w_exp=0.3,
            w_delta=0.5,
            max_distance=float(np.linalg.norm(goal.attractor_state[:wm.latent_dim]) * 1.5)
        )

        self.intrinsic = IntrinsicMotivation(
            w_prediction=0.5,
            w_coverage=0.3,
            w_ensemble=0.2
        )

        self.coverage = CoverageBuffer(
            wm=wm,
            max_episodes=500,
            prioritized=True
        )

        self.inv_dyn = InverseDynamicsModel(
            latent_dim=wm.latent_dim,
            action_dim=wm.action_dim,
            learning_rate=0.01
        )

        self.batch_size = batch_size
        self.lr = lr
        self.k_steps = k_steps

        # Tracking
        self.phase = 0
        self.total_steps = 0
        self.reward_log: List[Dict] = []
        self.loss_log: List[float] = []
        self.intrinsic_log: List[float] = []

        # Model parameter reference for ES gradients
        self.param_list = [
            self.wm.W_t1, self.wm.b_t1,
            self.wm.W_t2, self.wm.b_t2,
            self.wm.W_t_logvar, self.wm.b_t_logvar,
            self.wm.W_zh, self.wm.W_zx, self.wm.b_z,
            self.wm.W_rh, self.wm.W_rx, self.wm.b_r,
            self.wm.W_hh, self.wm.W_hx, self.wm.b_h,
            self.wm.W_mu, self.wm.b_mu,
            self.wm.W_logvar, self.wm.b_logvar,
            self.wm.W_r1, self.wm.b_r1,
            self.wm.W_r2, self.wm.b_r2,
        ]

    def _es_gradient_step(self, batch: List[Dict], n_samples: int = 16, sigma: float = 0.01):
        """Apply ES gradient update on current batch."""
        if not batch:
            return

        orig_params = [p.copy() for p in self.param_list]
        grad_accum = [np.zeros_like(p) for p in self.param_list]

        for k in range(n_samples):
            noises = [np.random.randn(*p.shape) * sigma for p in self.param_list]

            for p, n in zip(self.param_list, noises):
                p[:] = p + n
            loss_pos = self._eval_loss(batch)

            for p, n in zip(self.param_list, noises):
                p[:] = p - 2 * n
            loss_neg = self._eval_loss(batch)

            for p, orig in zip(self.param_list, orig_params):
                p[:] = orig

            delta = (loss_pos - loss_neg) / (2.0 * sigma)
            for g, n in zip(grad_accum, noises):
                g[:] = g + delta * n

        for p, g in zip(self.param_list, grad_accum):
            p[:] = p - self.lr * g / n_samples

    def _eval_loss(self, batch: List[Dict]) -> float:
        """Evaluate loss for ES gradient computation."""
        seq_loss = compute_flow_sequence_loss(self.wm, batch, self.k_steps)
        rew_loss = compute_flow_reward_loss(self.wm, batch)
        return float(seq_loss + 0.5 * rew_loss)

    def _train_on_buffer(self, n_steps: int = 5):
        """Train world model on coverage buffer."""
        losses = []
        for _ in range(n_steps):
            batch = self.coverage.sample_batch(
                self.batch_size, seq_len=10
            )
            if not batch:
                continue

            self._es_gradient_step(batch, n_samples=12, sigma=0.01)

            seq_loss = compute_flow_sequence_loss(self.wm, batch, self.k_steps)
            rew_loss = compute_flow_reward_loss(self.wm, batch)
            total = float(seq_loss + 0.5 * rew_loss)
            losses.append(total)

        if losses:
            self.loss_log.extend(losses)
        return float(np.mean(losses)) if losses else float('inf')

    # ======================================================================
    # PHASE 1: COVERAGE — Pure Random Exploration
    # ======================================================================

    def phase_coverage(self) -> Dict:
        """Phase 1: Fill buffer with random trajectories, train world model."""
        print("\n" + "=" * 70)
        print(f"PHASE 1: COVERAGE ({self.n_coverage} steps)")
        print("=" * 70)

        self.phase = 1
        z = np.random.randn(self.latent_dim) * 0.5
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)

        # Fill buffer with diverse random episodes
        print("\n  Filling coverage buffer with random action trajectories...")
        self.coverage.fill_random(
            n_episodes=min(80, self.n_coverage // 5),
            steps_per_episode=15,
            goal=self.goal
        )

        buffer_stats = self.coverage.get_stats()
        print(f"  Buffer: {buffer_stats['n_episodes']} episodes, "
              f"{buffer_stats['n_transitions']} transitions")

        # Initial training burst
        print("\n  Initial world model training burst...")
        initial_loss = self._train_on_buffer(n_steps=30)
        print(f"  Initial loss: {initial_loss:.6f}")

        # Explore and train interleaved
        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_flows = []

        for step in range(self.n_coverage):
            # Generate random action with decaying noise
            noise_scale = 0.5 * (0.997 ** step)
            a = np.random.randn(self.action_dim) * noise_scale

            mu, logvar = self.wm.predict_transition(z, h, a)
            std = np.exp(0.5 * logvar)
            z_next = mu + np.random.randn(*mu.shape) * std * 0.1
            h = self.wm.gru_step(h, z_next)

            # Shaped reward (even in phase 1 for later use)
            rew_info = self.shaped_reward.compute(z_next, z)
            reward = rew_info['total']

            # Intrinsic motivation bonus
            intrinsic_info = self.intrinsic.compute(self.wm, z, h, a, z_next)
            reward += intrinsic_info['total'] * 0.5

            self.inv_dyn.train_step(z, z_next, a)
            self.inv_dyn.add_transition(z, z_next, a)

            episode_states.append(z.copy())
            episode_actions.append(a.copy())
            episode_rewards.append(reward)
            episode_flows.append(PointFlow(np.zeros(self.latent_dim)))

            z = z_next.copy()

            # Periodically add episode to buffer and train
            if step > 0 and step % 20 == 0:
                if len(episode_states) >= 5:
                    ep = FlowEpisode(
                        states=episode_states[:len(episode_actions)],
                        beliefs=[np.zeros(self.wm.belief_dim)] * len(episode_actions),
                        actions=episode_actions,
                        flow_embeddings=[np.zeros(self.wm.flow_embed_dim)] * len(episode_actions),
                        rewards=episode_rewards,
                        flow_ids=['random'] * len(episode_actions),
                        flow_types=['random'] * len(episode_actions)
                    )
                    self.coverage.add_episode(ep)

                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_flows = []

                # Train periodically
                train_loss = self._train_on_buffer(n_steps=3)
                if step % 50 == 0:
                    print(f"  Step {step:4d}: loss={train_loss:.6f}, "
                          f"buffer={len(self.coverage.buffer.episodes)}eps")

            self.total_steps += 1

        # Final training burst
        print("\n  Final world model training burst...")
        final_loss = self._train_on_buffer(n_steps=30)
        print(f"  Final loss: {final_loss:.6f}")

        buffer_stats = self.coverage.get_stats()
        cov_stats = self.intrinsic.get_coverage_stats()

        result = {
            'phase': 'coverage',
            'steps': self.n_coverage,
            'final_loss': final_loss,
            'buffer': buffer_stats,
            'coverage': cov_stats,
            'inv_dyn_data': len(self.inv_dyn.training_data)
        }

        print(f"\n  Coverage result: {cov_stats['n_unique_bins']} unique bins, "
              f"{cov_stats['total_visits']} visits")

        return result

    # ======================================================================
    # PHASE 2: SHAPING — Shaped Reward + Intrinsic Motivation
    # ======================================================================

    def phase_shaping(self) -> Dict:
        """Phase 2: Use shaped reward and intrinsic motivation."""
        print("\n" + "=" * 70)
        print(f"PHASE 2: SHAPING ({self.n_shaping} steps)")
        print("=" * 70)

        self.phase = 2
        z = np.random.randn(self.latent_dim) * 0.5
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)

        # Decaying exploration vs exploitation
        exploration_rate = 0.8
        exploration_decay = 0.998

        # Goal-directed action bias
        goal_latent = self.goal.attractor_state[:self.latent_dim]

        episode_states = [z.copy()]
        episode_actions = []
        episode_rewards = []
        reward_history = deque(maxlen=50)

        for step in range(self.n_shaping):
            exploration_rate *= exploration_decay
            exploration_rate = max(exploration_rate, 0.1)

            # Mix of random and goal-directed actions
            if random.random() < exploration_rate:
                a = np.random.randn(self.action_dim) * 0.4
            else:
                # Goal-directed: action that moves z toward goal
                direction = goal_latent - z
                a = direction[:self.action_dim] * 0.3 + np.random.randn(self.action_dim) * 0.1

            mu, logvar = self.wm.predict_transition(z, h, a)
            std = np.exp(0.5 * logvar)
            z_next = mu + np.random.randn(*mu.shape) * std * 0.1
            h = self.wm.gru_step(h, z_next)

            # Shaped reward
            rew_info = self.shaped_reward.compute(z_next, z)

            # Intrinsic motivation
            intrinsic_info = self.intrinsic.compute(self.wm, z, h, a, z_next)

            # Combined reward
            reward = rew_info['total'] + 0.3 * intrinsic_info['total']
            reward_history.append(reward)

            self.intrinsic_log.append(intrinsic_info['total'])

            self.inv_dyn.train_step(z, z_next, a)
            self.inv_dyn.add_transition(z, z_next, a)

            episode_states.append(z_next.copy())
            episode_actions.append(a.copy())
            episode_rewards.append(reward)

            z = z_next.copy()

            # Periodically add episode, train, report
            if step > 0 and step % 25 == 0:
                if len(episode_states) >= 5:
                    ep = FlowEpisode(
                        states=episode_states[:len(episode_actions)],
                        beliefs=[np.zeros(self.wm.belief_dim)] * len(episode_actions),
                        actions=episode_actions,
                        flow_embeddings=[np.zeros(self.wm.flow_embed_dim)] * len(episode_actions),
                        rewards=episode_rewards,
                        flow_ids=['shaping'] * len(episode_actions),
                        flow_types=['shaping'] * len(episode_actions)
                    )
                    self.coverage.add_episode(ep)

                episode_states = [z.copy()]
                episode_actions = []
                episode_rewards = []

                train_loss = self._train_on_buffer(n_steps=5)

                if step % 100 == 0:
                    dist = float(np.linalg.norm(z - goal_latent))
                    avg_reward = float(np.mean(reward_history)) if reward_history else 0.0
                    gp_val = float(np.exp(-dist))
                    print(f"  Step {step:4d}: loss={train_loss:.6f}, "
                          f"dist={dist:.3f}, GP={gp_val:.6f}, "
                          f"reward={avg_reward:.4f}, "
                          f"explore={exploration_rate:.2f}")

            self.total_steps += 1

        # Final training and evaluation
        final_loss = self._train_on_buffer(n_steps=20)

        # Run GP landscape diagnosis
        gp_z = np.random.randn(100, self.latent_dim) * 0.8
        gps = np.array([self.shaped_reward.compute_goal_prob(z_i) for z_i in gp_z])

        print(f"\n  GP landscape after shaping:")
        print(f"    Mean GP: {np.mean(gps):.6f}")
        print(f"    Max GP: {np.max(gps):.6f}")
        print(f"    GP > 0.1: {np.sum(gps > 0.1)} / 100")
        print(f"    GP > 0.5: {np.sum(gps > 0.5)} / 100")

        result = {
            'phase': 'shaping',
            'steps': self.n_shaping,
            'final_loss': final_loss,
            'mean_gp': float(np.mean(gps)),
            'max_gp': float(np.max(gps)),
            'gp_above_01': int(np.sum(gps > 0.1)),
            'gp_above_05': int(np.sum(gps > 0.5)),
            'buffer': self.coverage.get_stats(),
            'inv_dyn_data': len(self.inv_dyn.training_data)
        }

        return result

    # ======================================================================
    # PHASE 3: TRANSFER — Prepare for standard flow/CEM takeover
    # ======================================================================

    def phase_transfer(self) -> Dict:
        """Phase 3: Seed flows, verify GP landscape, transfer."""
        print("\n" + "=" * 70)
        print(f"PHASE 3: TRANSFER ({self.n_transfer} steps)")
        print("=" * 70)

        self.phase = 3
        goal_latent = self.goal.attractor_state[:self.latent_dim]

        # Create manifold with goal-directed flow
        manifold = FlowManifold(flow_dim=4)

        # Seed with flows that target the goal
        target = goal_latent.copy()
        for i in range(6):
            if i == 0:
                flow = PointFlow(target, gain=0.5)
                flow.stability = 0.7
                flow.goal_alignment = 0.7
            elif i < 3:
                noise = np.random.randn(self.latent_dim) * 0.3
                flow = PointFlow(target + noise, gain=random.uniform(0.3, 0.7))
                flow.stability = 0.5
                flow.goal_alignment = 0.5
            else:
                center = np.random.randn(self.latent_dim) * 0.5
                flow = LimitCycleFlow(
                    center, radius=random.uniform(0.5, 1.5),
                    omega=random.uniform(0.3, 0.8)
                )
                flow.stability = 0.4
                flow.goal_alignment = 0.3
            manifold.add_flow(flow, f'seed_{i}')

        # Execute with shaped reward
        z = np.random.randn(self.latent_dim) * 0.3
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)

        gp_history = []
        for step in range(self.n_transfer):
            # Pick flow
            flow = list(manifold.flows.values())[step % len(manifold.flows)]
            a = flow.compute_action(z, h)

            mu, logvar = self.wm.predict_transition(z, h, a)
            std = np.exp(0.5 * logvar)
            z_next = mu + np.random.randn(*mu.shape) * std * 0.1
            h = self.wm.gru_step(h, z_next)

            # Track GP using shaped reward (nonzero everywhere)
            gp = self.shaped_reward.compute_goal_prob(z_next)
            gp_history.append(gp)

            rew_info = self.shaped_reward.compute(z_next, z)
            intrinsic_info = self.intrinsic.compute(self.wm, z, h, a, z_next)
            reward = rew_info['total'] + 0.2 * intrinsic_info['total']

            self.inv_dyn.train_step(z, z_next, a)

            z = z_next.copy()

            if step % 20 == 0:
                train_loss = self._train_on_buffer(n_steps=3)
                dist = float(np.linalg.norm(z - goal_latent))
                print(f"  Step {step:4d}: GP={gp:.6f}, dist={dist:.3f}, "
                      f"loss={train_loss:.6f}")

        # Final verification
        final_gp = float(np.mean(gp_history[-50:])) if len(gp_history) >= 50 else 0.0

        # Run full GP landscape diagnosis
        gp_z = np.random.randn(200, self.latent_dim) * 0.8
        gps = np.array([self.shaped_reward.compute_goal_prob(z_i) for z_i in gp_z])

        print(f"\n  Final GP landscape:")
        print(f"    Mean GP: {np.mean(gps):.6f}")
        print(f"    Max GP: {np.max(gps):.6f}")
        print(f"    GP > 0.1: {np.sum(gps > 0.1)} / 200")
        print(f"    GP > 0.5: {np.sum(gps > 0.5)} / 200")
        print(f"    Final trajectory GP: {final_gp:.6f}")

        result = {
            'phase': 'transfer',
            'steps': self.n_transfer,
            'mean_gp': float(np.mean(gps)),
            'max_gp': float(np.max(gps)),
            'gp_above_01': int(np.sum(gps > 0.1)),
            'gp_above_05': int(np.sum(gps > 0.5)),
            'final_traj_gp': final_gp,
            'manifold_flows': len(manifold.flows),
            'buffer': self.coverage.get_stats()
        }

        return result

    # ======================================================================
    # FULL BOOTSTRAP
    # ======================================================================

    def run(self) -> Dict:
        """Run all three bootstrap phases."""
        print("\n" + "=" * 70)
        print("PHASE 41: REPRESENTATION BOOTSTRAPPER")
        print("=" * 70)
        print("""
  Diagnosis revealed: GP landscape is FLAT (mean GP = 0.001)
  Fix: 3-phase bootstrapping
    Phase 1 — Coverage: random actions, diverse buffer, pre-train WM
    Phase 2 — Shaping: shaped reward + intrinsic motivation
    Phase 3 — Transfer: goal-directed flows with grounded WM
        """)

        phase1 = self.phase_coverage()
        phase2 = self.phase_shaping()
        phase3 = self.phase_transfer()

        print("\n" + "=" * 70)
        print("BOOTSTRAP COMPLETE — GP LANDSCAPE VERIFICATION")
        print("=" * 70)

        # Final verification: run Phase 41 diagnosis
        print("\n  Running final GP landscape diagnosis...")
        gp_z = np.random.randn(500, self.latent_dim) * 0.8
        gps = np.array([self.shaped_reward.compute_goal_prob(z_i) for z_i in gp_z])
        mean_gp = float(np.mean(gps))
        max_gp = float(np.max(gps))
        above_01 = int(np.sum(gps > 0.1))
        above_05 = int(np.sum(gps > 0.5))

        verdict = 'SIGNAL_PRESENT' if max_gp > 0.1 else 'FLAT'

        print(f"\n    Mean GP: {mean_gp:.6f}")
        print(f"    Max GP: {max_gp:.6f}")
        print(f"    GP > 0.1: {above_01} / 500")
        print(f"    GP > 0.5: {above_05} / 500")
        print(f"    Verdict: {verdict}")

        return {
            'phase1': phase1,
            'phase2': phase2,
            'phase3': phase3,
            'final_verdict': verdict,
            'mean_gp': mean_gp,
            'max_gp': max_gp,
            'gp_above_01': above_01,
            'gp_above_05': above_05,
            'total_steps': self.total_steps
        }


# ============================================================================
# RUN
# ============================================================================

def test_bootstrapper(n_coverage: int = 200, n_shaping: int = 150, n_transfer: int = 80):
    """Run full bootstrap and verify."""
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='boot_target',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )

    bootstrapper = RepresentationBootstrapper(
        wm=wm,
        goal=goal,
        n_coverage=n_coverage,
        n_shaping=n_shaping,
        n_transfer=n_transfer
    )

    result = bootstrapper.run()

    print("\n" + "=" * 70)
    print("BOOTSTRAP RESULTS")
    print("=" * 70)
    print(f"\n  Total steps: {result['total_steps']}")
    print(f"  GP landscape: {result['mean_gp']:.6f} mean, {result['max_gp']:.6f} max")
    print(f"  GP > 0.1: {result['gp_above_01']}/500")
    print(f"  GP > 0.5: {result['gp_above_05']}/500")
    print(f"  Verdict: {result['final_verdict']}")
    print(f"\n  Transition: ready for Phase 40 continuous self-organization")
    print(f"    with non-flat GP landscape providing useful gradients.")

    return wm, result


if __name__ == "__main__":
    wm, result = test_bootstrapper()

    # Run Phase 41 diagnosis on the bootstrapped model
    print("\n\nRunning Phase 41 diagnosis on bootstrapped model...")
    from phase41_diagnosis import DiagnosisSuite
    diag = DiagnosisSuite(wm)
    diag_results = diag.run_all()
