"""
Phase 18.5 - B-lite Runner

Uses probabilistic transition model instead of noise injection.
Replaces: dynamics.predict(z, a) + noise
With:     transition_model.predict_multi_modal(z, a)
"""
import numpy as np
from typing import List, Tuple


class BLiteRunner:
    """
    MVP using probabilistic trajectory model.
    
    Key difference from v3:
      OLD: trajectories = [noise injection per action]
      NEW: trajectories = [mode-based multi-modal prediction]
    """
    
    def __init__(self, env_type: str = 'simple'):
        # Import components
        from encoder import Encoder
        from policy import Policy
        from probabilistic_vfield import MultiModalVField
        from env import SimpleEnv, StressEnv
        
        # Initialize environment
        if env_type == 'simple':
            self.env = SimpleEnv()
        elif env_type == 'stress':
            self.env = StressEnv()
        else:
            raise ValueError(f"Unknown env type: {env_type}")
        
        # Core components
        obs_dim = 10 if env_type == 'simple' else 6
        latent_dim = 8
        action_dim = 2
        
        self.encoder = Encoder(obs_dim, latent_dim)
        self.vfield = MultiModalVField(latent_dim, num_modes=3)
        self.policy = Policy(action_dim)
        
        # For policy: need to adapt to probabilistic vfield
        self.policy.use_stochastic = False  # Use mode-based, not noise
        
        self.history = []
        self.step_count = 0
        self.total_reward = 0.0
        
    def reset(self):
        """Reset for new episode."""
        obs = self.env.reset()
        self.encoder.reset_history()
        self.vfield.V_history.clear()
        self.vfield.diversity_history.clear()
        self.vfield.divergence_history.clear()
        self.history = []
        self.step_count = 0
        self.total_reward = 0.0
        return obs
    
    def step(self) -> Tuple[np.ndarray, float, bool, dict]:
        """Single step with probabilistic V-field."""
        # Get observation
        if self.step_count == 0:
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        
        # Encode
        z = self.encoder.encode(obs)
        
        # Get candidates with mode-based V
        candidates = []
        for idx, action in enumerate(self.policy.actions):
            # Ensure action is proper numpy array
            if isinstance(action, list):
                action = np.array(action)
            elif not isinstance(action, np.ndarray):
                action = np.array([action])
            
            V = self.vfield.compute_V(z, action)
            candidates.append((idx, action, V))
        
        # Select best action
        best = max(candidates, key=lambda x: x[2])
        action_idx, action, predicted_V = best
        
        # Ensure action is numpy array
        if isinstance(action, list):
            action = np.array(action)
        
        # Execute
        obs, reward, done = self.env.step(action)
        
        # Learn from transition
        z_next = self.encoder.encode(obs)
        self.vfield.observe(z, action, z_next)
        
        # Record
        signals = self.vfield.get_signals()
        self.total_reward += reward
        
        info = {
            'step': self.step_count,
            'z': z.copy(),
            'action': action_idx,
            'predicted_V': predicted_V,
            'signals': signals,
            'reward': reward,
        }
        
        self.history.append(info)
        self.step_count += 1
        
        return obs, reward, done, info
    
    def run_episode(self, max_steps: int = 200, verbose: bool = True) -> dict:
        """Run one episode with B-lite architecture."""
        self.reset()
        
        for step in range(max_steps):
            obs, reward, done, info = self.step()
            
            if verbose and step % 20 == 0:
                s = info['signals']
                print(f"Step {step}: V={s['V']:.3f}, "
                      f"div={s['mode_diversity']:.3f}, "
                      f"divergence={s['mode_divergence']:.3f}, "
                      f"entropy={s['mode_entropy']:.3f}")
            
            if done:
                break
        
        # Stats
        V_values = [h['signals']['V'] for h in self.history]
        
        return {
            'steps': len(self.history),
            'total_reward': self.total_reward,
            'V_mean': np.mean(V_values),
            'V_min': np.min(V_values),
            'V_final': V_values[-1] if V_values else 0,
            'V_trend': np.mean(np.diff(V_values)) if len(V_values) > 1 else 0,
        }


def compare_v3_vs_blite():
    """Compare v3 (noise) vs B-lite (mode-based)."""
    from main import MVPRunner
    
    print("=" * 70)
    print("COMPARISON: MVP v3 (noise) vs B-lite (modes)")
    print("=" * 70)
    
    # Run v3
    print("\n--- MVP v3 (noise injection) ---")
    runner_v3 = MVPRunner(env_type='simple')
    stats_v3 = runner_v3.run_episode(max_steps=100, verbose=False)
    print(f"V3: V_mean={stats_v3['V_mean']:.3f}, V_min={stats_v3['V_min']:.3f}")
    
    # Run B-lite
    print("\n--- B-lite (probabilistic modes) ---")
    runner_blite = BLiteRunner(env_type='simple')
    stats_blite = runner_blite.run_episode(max_steps=100, verbose=False)
    print(f"BLite: V_mean={stats_blite['V_mean']:.3f}, V_min={stats_blite['V_min']:.3f}")
    
    # Compare
    print("\n--- COMPARISON ---")
    print(f"V mean: v3={stats_v3['V_mean']:.3f}, blite={stats_blite['V_mean']:.3f}")
    print(f"V min:  v3={stats_v3['V_min']:.3f}, blite={stats_blite['V_min']:.3f}")
    
    return stats_v3, stats_blite


def run_stress_test():
    """Run stress test on B-lite."""
    print("\n" + "=" * 70)
    print("STRESS TEST: B-lite architecture")
    print("=" * 70)
    
    runner = BLiteRunner(env_type='stress')
    stats = runner.run_episode(max_steps=200, verbose=True)
    
    print(f"\nResults:")
    print(f"  Total reward: {stats['total_reward']:.2f}")
    print(f"  V mean: {stats['V_mean']:.3f}")
    print(f"  V min: {stats['V_min']:.3f}")
    print(f"  V trend: {stats['V_trend']:.5f}")


if __name__ == '__main__':
    compare_v3_vs_blite()