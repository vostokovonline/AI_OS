"""
Main loop for Phase 18 MVP v2 - Trajectory-based V-field

Key change:
  OLD: V(z) = single-step prediction quality
  NEW: V(z) = geometry of possible future trajectories

for t in range(T):
    z = encoder(obs)
    
    For each action:
        Generate trajectory ensemble (horizon=4)
        Compute V from trajectory endpoints diversity
    
    Select action with max V
    Execute action
"""
import numpy as np
from typing import List, Tuple, Optional


class MVPRunner:
    """
    Main runner for Phase 18 MVP v2.
    
    Uses trajectory-based V-field for viability evaluation.
    """
    
    def __init__(self, env_type: str = 'simple'):
        self.env_type = env_type
        
        # Import components
        from encoder import Encoder
        from dynamics import Dynamics
        from vfield import VField
        from policy import Policy
        from env import SimpleEnv, StressEnv
        
        # Initialize environment
        if env_type == 'simple':
            self.env = SimpleEnv()
        elif env_type == 'stress':
            self.env = StressEnv()
        else:
            raise ValueError(f"Unknown env type: {env_type}")
        
        # Observation dimension
        obs_dim = 10 if env_type == 'simple' else 6
        latent_dim = 8
        action_dim = 2
        
        # Initialize components
        self.encoder = Encoder(obs_dim, latent_dim)
        self.dynamics = Dynamics(latent_dim, action_dim)
        self.vfield = VField(latent_dim)
        self.policy = Policy(action_dim)
        
        # Trajectory horizon
        self.horizon = 4
        
        # Stats
        self.history = []
        self.step_count = 0
        
    def reset(self):
        """Reset for new episode."""
        obs = self.env.reset()
        self.encoder.reset_history()
        self.vfield.reset_history()
        self.history = []
        self.step_count = 0
        return obs
    
    def step(self) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Single step of trajectory-based loop.
        
        Returns:
            obs: observation
            reward: reward
            done: episode ended
            info: diagnostic info
        """
        # Get current observation
        if self.step_count == 0:
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        
        # Encode to latent state
        z = self.encoder.encode(obs)
        
        # Select action using trajectory-based V-field
        action_idx, action, predicted_V = self.policy.select_action(
            z, self.dynamics, self.vfield
        )
        
        # Get candidates for analysis
        candidates = self.policy.get_candidates(z, self.dynamics, self.vfield)
        
        # Generate trajectories for all actions
        all_trajectories = []
        for idx, a in enumerate(self.policy.actions):
            trajs = self.vfield.rollout(z, self.dynamics, np.array([a]), 
                                        self.horizon, stochastic=True)
            all_trajectories.append(trajs)
        
        # Execute action
        obs, reward, done = self.env.step(action)
        
        # Update V-field with actual state after action
        z_next = self.encoder.encode(obs)
        
        # Generate new trajectories and compute actual V
        trajectories = self.vfield.rollout(z_next, self.dynamics, 
                                          self.policy.actions, self.horizon)
        actual_V = self.vfield.compute_V(trajectories)
        
        signals = self.vfield.get_signals()
        
        # Check for special events
        silent_collapse = self.vfield.detect_silent_collapse()
        
        # Record history
        self.history.append({
            'step': self.step_count,
            'z': z.copy(),
            'action': action_idx,
            'predicted_V': predicted_V,
            'actual_V': actual_V,
            'signals': signals,
            'reward': reward,
            'vfield_status': self.vfield.get_status(),
            'silent_collapse': silent_collapse,
            'attractor_trap': self.vfield.detect_attractor_trap(trajectories),
        })
        
        self.step_count += 1
        
        info = {
            'z': z,
            'action': action_idx,
            'predicted_V': predicted_V,
            'actual_V': actual_V,
            'signals': signals,
            'candidates': candidates,
            'vfield_status': self.vfield.get_status(),
            'silent_collapse': silent_collapse,
            'attractor_trap': self.history[-1]['attractor_trap'],
        }
        
        return obs, reward, done, info
    
    def run_episode(self, max_steps: int = 200, verbose: bool = True) -> dict:
        """
        Run one episode with trajectory-based V-field.
        
        Returns:
            stats: episode statistics
        """
        self.reset()
        
        total_reward = 0.0
        vfield_status_history = []
        silent_collapse_events = 0
        attractor_traps = []
        
        for step in range(max_steps):
            obs, reward, done, info = self.step()
            
            total_reward += reward
            vfield_status_history.append(info['vfield_status'])
            
            if info['silent_collapse']:
                silent_collapse_events += 1
            
            attractor_traps.append(info['attractor_trap'])
            
            if verbose and step % 20 == 0:
                print(f"Step {step}: V={info['actual_V']:.3f}, "
                      f"diversity={info['signals']['diversity']:.3f}, "
                      f"collapse={info['signals']['collapse']:.3f}, "
                      f"trend={info['signals']['trend']:.3f}, "
                      f"status: {info['vfield_status']}")
            
            if done:
                break
        
        # Compute statistics
        V_values = [h['actual_V'] for h in self.history]
        
        stats = {
            'steps': len(self.history),
            'total_reward': total_reward,
            'V_mean': np.mean(V_values),
            'V_min': np.min(V_values),
            'V_final': V_values[-1] if V_values else 0,
            'V_trend': np.mean(np.diff(V_values)) if len(V_values) > 1 else 0,
            'healthy_ratio': sum(1 for s in vfield_status_history if s == 'HEALTHY') / max(1, len(vfield_status_history)),
            'warning_ratio': sum(1 for s in vfield_status_history if 'WARNING' in s) / max(1, len(vfield_status_history)),
            'critical_ratio': sum(1 for s in vfield_status_history if 'CRITICAL' in s) / max(1, len(vfield_status_history)),
            'silent_collapse_events': silent_collapse_events,
            'avg_attractor_trap': np.mean(attractor_traps),
            'vfield_history': vfield_status_history,
        }
        
        return stats
    
    def run_comparison(self, num_episodes: int = 10, max_steps: int = 200) -> List[dict]:
        """
        Run comparison between simple and stress environments.
        """
        results = []
        
        print("=" * 60)
        print("Phase 18 MVP v2 - Trajectory-based V-field")
        print("=" * 60)
        
        print("\n--- Simple Environment ---")
        runner_simple = MVPRunner(env_type='simple')
        results_simple = runner_simple.run_episode(max_steps=max_steps, verbose=True)
        print(f"\nResults: V_mean={results_simple['V_mean']:.3f}, V_trend={results_simple['V_trend']:.4f}")
        results.append(('simple', results_simple))
        
        print("\n--- Stress Environment ---")
        runner_stress = MVPRunner(env_type='stress')
        results_stress = runner_stress.run_episode(max_steps=max_steps, verbose=True)
        print(f"\nResults: V_mean={results_stress['V_mean']:.3f}, V_trend={results_stress['V_trend']:.4f}")
        print(f"Silent collapse events: {results_stress['silent_collapse_events']}")
        print(f"Avg attractor trap: {results_stress['avg_attractor_trap']:.3f}")
        results.append(('stress', results_stress))
        
        return results


def main():
    """Main entry point."""
    runner = MVPRunner(env_type='simple')
    stats = runner.run_episode(max_steps=200, verbose=True)
    
    print("\n" + "=" * 60)
    print("Episode Results (Trajectory-based V-field)")
    print("=" * 60)
    print(f"Total reward: {stats['total_reward']:.2f}")
    print(f"V mean: {stats['V_mean']:.3f}")
    print(f"V min: {stats['V_min']:.3f}")
    print(f"V final: {stats['V_final']:.3f}")
    print(f"V trend: {stats['V_trend']:.4f}")
    print(f"Healthy ratio: {stats['healthy_ratio']:.1%}")
    print(f"Silent collapse: {stats['silent_collapse_events']} events")
    print(f"Avg attractor trap: {stats['avg_attractor_trap']:.3f}")


if __name__ == '__main__':
    main()