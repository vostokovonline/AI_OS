"""
Phase 18.6 - World Model Kernel Runner

Тестирует Transition-First World Model Kernel.

Отличается от B-lite:
1. Жёсткие контракты на размерности
2. Валидация режимов
3. Разделённые Diagnostic V и Control V
"""
import numpy as np
from typing import Tuple
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')
from world_model_kernel import WorldModelKernel, DiagnosticV, ControlV
from encoder import Encoder
from env import SimpleEnv


class WorldModelRunner:
    """Runner для World Model Kernel."""
    
    def __init__(self, latent_dim=8, action_dim=2, num_modes=3):
        self.env = SimpleEnv()
        self.encoder = Encoder(10, latent_dim)
        
        # NEW: World Model Kernel
        self.kernel = WorldModelKernel(latent_dim, action_dim, num_modes)
        
        # Actions
        self.actions = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
            np.array([0, 0]),
        ]
        
        self.step_count = 0
        self.total_reward = 0.0
    
    def reset(self):
        obs = self.env.reset()
        self.encoder.reset_history()
        self.kernel = WorldModelKernel(
            self.kernel.latent_dim,
            self.kernel.action_dim,
            self.kernel.num_modes
        )
        self.step_count = 0
        self.total_reward = 0.0
        return obs
    
    def step(self) -> Tuple[np.ndarray, float, bool, dict]:
        obs = self.env._get_obs()
        z = self.encoder.encode(obs)
        
        # Get diagnostic state
        diagnostic = self.kernel.get_diagnostic_state()
        
        # Get control signal for action selection
        control = self.kernel.get_control_signal()
        
        # Select action based on Control V (not raw Diagnostic V)
        best_V = -1
        best_action = self.actions[0]
        best_idx = 0
        
        for idx, action in enumerate(self.actions):
            # Check kernel stability first
            if not control['is_safe'] and self.step_count > 10:
                # Conservative mode: select action with max diversity
                mode_predictions = self.kernel.transition.predict_multi_modal(z, action)
                if mode_predictions:
                    diversity = len(mode_predictions) / self.kernel.num_modes
                    if diversity > best_V:
                        best_V = diversity
                        best_action = action
                        best_idx = idx
            else:
                # Normal mode: use control V
                mode_predictions = self.kernel.transition.predict_multi_modal(z, action)
                if mode_predictions:
                    weights = np.array([m[2] for m in mode_predictions])
                    weights_norm = weights / weights.sum()
                    diversity = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
                    V = control['V'] * (1 + diversity)
                    if V > best_V:
                        best_V = V
                        best_action = action
                        best_idx = idx
        
        # Execute action
        result = self.env.step(best_action)
        if len(result) == 4:
            obs_next, reward, done, info_extra = result
        else:
            obs_next, reward, done = result
            info_extra = {}
        z_next = self.encoder.encode(obs_next)
        
        # Observe transition in kernel
        self.kernel.observe(z, best_action, z_next)
        
        self.step_count += 1
        self.total_reward += reward
        
        info = {
            'diagnostic_V': diagnostic['V'],
            'control_V': control['V'],
            'is_safe': control['is_safe'],
            'valid_modes': diagnostic.get('valid_modes', 0),
            'mode_stats': diagnostic.get('mode_stats', {}),
            'status': diagnostic['status'],
            **info_extra
        }
        
        return obs_next, reward, done, info
    
    def run_episode(self, max_steps=100, verbose=True) -> dict:
        """Run single episode."""
        self.reset()
        
        V_history = []
        control_V_history = []
        status_history = []
        
        for step in range(max_steps):
            obs, reward, done, info = self.step()
            
            V_history.append(info['diagnostic_V'])
            control_V_history.append(info['control_V'])
            status_history.append(info['status'])
            
            if verbose and step % 20 == 0:
                print(f"  Step {step}: V={info['diagnostic_V']:.3f}, "
                      f"control={info['control_V']:.3f}, "
                      f"modes={info['valid_modes']}, "
                      f"status={info['status']}")
            
            if done:
                break
        
        return {
            'steps': self.step_count,
            'reward': self.total_reward,
            'V_mean': np.mean(V_history) if V_history else 0,
            'V_min': np.min(V_history) if V_history else 0,
            'control_V_mean': np.mean(control_V_history) if control_V_history else 0,
            'final_status': status_history[-1] if status_history else 'unknown',
            'V_history': V_history
        }


def test_kernel():
    """Test World Model Kernel."""
    print("=" * 60)
    print("WORLD MODEL KERNEL TEST")
    print("=" * 60)
    
    runner = WorldModelRunner(latent_dim=8, action_dim=2, num_modes=3)
    
    # Run episode
    print("\n--- Running Episode ---")
    stats = runner.run_episode(max_steps=100, verbose=True)
    
    print(f"\n--- Results ---")
    print(f"Steps: {stats['steps']}")
    print(f"Reward: {stats['reward']:.2f}")
    print(f"V_mean: {stats['V_mean']:.3f}")
    print(f"V_min: {stats['V_min']:.3f}")
    print(f"Control_V_mean: {stats['control_V_mean']:.3f}")
    print(f"Final status: {stats['final_status']}")
    
    # Get full kernel state
    full_state = runner.kernel.get_full_state()
    print(f"\n--- Kernel State ---")
    print(f"Diagnostic: {full_state['diagnostic']['status']}")
    print(f"Control: is_safe={full_state['control']['is_safe']}")
    print(f"Modes: {full_state['transition_stats']['valid_modes']}/{full_state['transition_stats']['total_modes']} valid")
    
    return stats


def compare_with_blite():
    """Compare World Model Kernel with B-lite."""
    print("\n" + "=" * 60)
    print("COMPARISON: World Model Kernel vs B-lite")
    print("=" * 60)
    
    # World Model Kernel
    print("\n--- World Model Kernel ---")
    runner_wm = WorldModelRunner(latent_dim=8, action_dim=2, num_modes=3)
    stats_wm = runner_wm.run_episode(max_steps=100, verbose=False)
    
    print(f"WM: V_mean={stats_wm['V_mean']:.3f}, V_min={stats_wm['V_min']:.3f}, "
          f"control_V_mean={stats_wm['control_V_mean']:.3f}")
    
    # B-lite
    print("\n--- B-lite (from phase18_mvp) ---")
    try:
        from blite_runner import BLiteRunner
        runner_blite = BLiteRunner(env_type='simple')
        stats_blite = runner_blite.run_episode(max_steps=100, verbose=False)
        print(f"B-lite: V_mean={stats_blite['V_mean']:.3f}, V_min={stats_blite['V_min']:.3f}")
    except Exception as e:
        print(f"B-lite error: {e}")
        stats_blite = {'V_mean': 0, 'V_min': 0}
    
    # Comparison
    print("\n--- COMPARISON ---")
    print(f"V mean: WM={stats_wm['V_mean']:.3f}, B-lite={stats_blite['V_mean']:.3f}")
    print(f"V min:  WM={stats_wm['V_min']:.3f}, B-lite={stats_blite['V_min']:.3f}")
    
    return stats_wm, stats_blite


def test_stability():
    """Test kernel stability across multiple runs."""
    print("\n" + "=" * 60)
    print("STABILITY TEST (10 runs)")
    print("=" * 60)
    
    V_means = []
    V_mins = []
    
    for run in range(10):
        runner = WorldModelRunner(latent_dim=8, action_dim=2, num_modes=3)
        stats = runner.run_episode(max_steps=100, verbose=False)
        V_means.append(stats['V_mean'])
        V_mins.append(stats['V_min'])
        
        if run < 3 or run == 9:
            print(f"  Run {run+1}: V_mean={stats['V_mean']:.3f}, V_min={stats['V_min']:.3f}")
    
    print(f"\n--- Stability ---")
    print(f"V_mean: {np.mean(V_means):.3f} ± {np.std(V_means):.3f}")
    print(f"V_min:  {np.mean(V_mins):.3f} ± {np.std(V_mins):.3f}")
    
    return {'V_mean': V_means, 'V_min': V_mins}


def test_mode_validation():
    """Test mode validation behavior."""
    print("\n" + "=" * 60)
    print("MODE VALIDATION TEST")
    print("=" * 60)
    
    kernel = WorldModelKernel(latent_dim=8, action_dim=2, num_modes=3)
    
    # Simulate transitions
    z = np.zeros(8)
    a = np.array([1, 0])
    
    print("\n--- Initial state ---")
    stats = kernel.get_full_state()
    print(f"Valid modes: {stats['transition_stats']['valid_modes']}")
    print(f"Mode weights: {[f'{w:.3f}' for w in stats['transition_stats']['mode_weights']]}")
    
    # Add transitions
    print("\n--- Adding transitions ---")
    for i in range(100):
        z_next = z + np.random.randn(8) * 0.5
        kernel.observe(z, a, z_next)
        z = z_next
        
        if i in [9, 49, 99]:
            state = kernel.get_full_state()
            print(f"  Step {i+1}: valid_modes={state['transition_stats']['valid_modes']}, "
                  f"weights={[f'{w:.2f}' for w in state['transition_stats']['mode_weights']]}")
    
    # Final state
    print("\n--- Final state ---")
    full_state = kernel.get_full_state()
    print(f"Diagnostic V: {full_state['diagnostic']['V']:.3f}")
    print(f"Control V: {full_state['control']['V']:.3f}")
    print(f"Control is_safe: {full_state['control']['is_safe']}")
    print(f"Total transitions: {full_state['transition_stats']['total_transitions']}")


if __name__ == '__main__':
    # Test kernel
    test_kernel()
    
    # Compare with B-lite
    compare_with_blite()
    
    # Stability test
    test_stability()
    
    # Mode validation test
    test_mode_validation()
    
    print("\n" + "=" * 60)
    print("WORLD MODEL KERNEL TEST COMPLETE")
    print("=" * 60)