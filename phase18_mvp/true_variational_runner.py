"""
Phase 18.11 - True Variational World Model Runner

Tests true variational system vs "energy-shaped neural dynamical system".
"""
import numpy as np
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from true_variational_model import (
    TrueVariationalWorldModel,
    VariationalEnergyField,
    ImplicitEncoder
)
from env import SimpleEnv


def test_implicit_encoder():
    """Test implicit encoder."""
    print("=" * 60)
    print("IMPLICIT ENCODER TEST")
    print("=" * 60)
    
    encoder = ImplicitEncoder(obs_dim=10, latent_dim=8)
    
    # Test simple forward
    obs = np.random.randn(10)
    z_simple = encoder.forward(obs)
    print(f"Simple encoding: |z|={np.linalg.norm(z_simple):.3f}")
    
    # Define energy function (quadratic)
    def energy_fn(z, a):
        return np.sum(z ** 2)
    
    # Define hessian function
    def hessian_fn(z, a):
        return 2 * np.eye(len(z))
    
    # Compute implicit gradient
    result = encoder.compute_implicit_gradient(energy_fn, obs, None, hessian_fn)
    
    print(f"Implicit z*: |z|={np.linalg.norm(result['z_star']):.3f}")
    print(f"Optimality error: {result['optimality_error']:.6f}")
    print(f"Dz/dtheta shape: {result['dz_dtheta'].shape}")
    
    return encoder


def test_variational_energy_field():
    """Test variational energy field."""
    print("\n" + "=" * 60)
    print("VARIATIONAL ENERGY FIELD TEST")
    print("=" * 60)
    
    field = VariationalEnergyField(obs_dim=10, latent_dim=8, action_dim=2)
    
    obs = np.random.randn(10)
    a = np.array([1.0, 0.0])
    
    # Compute functional
    z = np.random.randn(8)
    functional = field.compute_functional(z, obs, a)
    
    print(f"Total functional: {functional['functional']:.4f}")
    print(f"  V (energy): {functional['V']:.4f}")
    print(f"  recon_loss: {functional['recon_loss']:.4f}")
    print(f"  consistency: {functional['consistency_loss']:.4f}")
    print(f"  hessian_penalty: {functional['hessian_penalty']:.4f}")
    
    # Hessian
    H = field.compute_hessian(z, a)
    print(f"\nHessian trace: {np.trace(H):.3f}")
    print(f"Hessian det: {np.linalg.det(H):.3f}")
    
    return field


def test_stability_spectrum():
    """Test stability spectrum from Hessian."""
    print("\n" + "=" * 60)
    print("STABILITY SPECTRUM TEST")
    print("=" * 60)
    
    field = VariationalEnergyField(obs_dim=10, latent_dim=8, action_dim=2)
    
    z = np.random.randn(8)
    stability = field.get_stability_spectrum(z)
    
    print(f"Eigenvalues: {[f'{ev:.3f}' for ev in stability['eigenvalues']]}")
    print(f"Stabilities: {[f'{s:.3f}' for s in stability['stabilities']]}")
    print(f"Stable directions: {stability['num_stable']}/{len(stability['eigenvalues'])}")
    print(f"Unstable directions: {stability['num_unstable']}/{len(stability['eigenvalues'])}")
    
    return stability


def test_true_variational_model():
    """Test full true variational model."""
    print("\n" + "=" * 60)
    print("TRUE VARIATIONAL WORLD MODEL TEST")
    print("=" * 60)
    
    model = TrueVariationalWorldModel(obs_dim=10, latent_dim=8, action_dim=2)
    env = SimpleEnv()
    
    actions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
    ]
    
    print("\n--- Running Episode ---")
    
    obs = env.reset()
    
    for step in range(100):
        a = actions[np.random.randint(len(actions))]
        obs_next, _, done = env.step(a)
        
        state = model.forward(obs, a)
        
        if step % 20 == 0:
            stability = state['stability_spectrum']
            print(f"  Step {step}: "
                  f"V={state['V']:.3f}, "
                  f"F={state['functional']:.4f}, "
                  f"opt_err={state['optimality_error']:.4f}, "
                  f"stable={stability['num_stable']}/{len(stability['eigenvalues'])}")
        
        obs = obs_next
        if done:
            break
    
    # Get final state
    final = model.get_state()
    print(f"\n--- Final State ---")
    print(f"Steps: {final['step_count']}")
    print(f"Loss mean: {final['loss_mean']:.4f}")
    print(f"Variational state: {final['variational_state']}")
    
    return model


def test_convergence():
    """Test convergence to fixed point."""
    print("\n" + "=" * 60)
    print("CONVERGENCE TEST")
    print("=" * 60)
    
    model = TrueVariationalWorldModel(obs_dim=10, latent_dim=8, action_dim=2)
    
    # Build up history
    optimality_errors = []
    
    for i in range(200):
        obs = np.random.randn(10)
        a = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
        state = model.forward(obs, a)
        
        if i % 20 == 0:
            optimality_errors.append(state['optimality_error'])
            print(f"  Step {i}: opt_error={state['optimality_error']:.6f}")
    
    print(f"\nOptimality error trend: {optimality_errors}")
    print(f"Final opt_error: {optimality_errors[-1]:.6f}")
    print(f"Converged: {optimality_errors[-1] < 0.01}")


def test_stability_analysis():
    """Test stability analysis across states."""
    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS TEST")
    print("=" * 60)
    
    model = TrueVariationalWorldModel(obs_dim=10, latent_dim=8, action_dim=2)
    
    # Sample different states
    for i in range(5):
        obs = np.random.randn(10) * (i + 1)
        a = np.array([1.0, 0.0])
        
        state = model.forward(obs, a)
        stability = state['stability_spectrum']
        
        print(f"State {i}: "
              f"|z|={np.linalg.norm(state['z']):.2f}, "
              f"stable={stability['num_stable']}/{len(stability['eigenvalues'])}, "
              f"unstable={stability['num_unstable']}/{len(stability['eigenvalues'])}")


def compare_with_previous():
    """Compare with previous approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: All approaches")
    print("=" * 60)
    
    results = {}
    
    # B-lite
    try:
        from blite_runner import BLiteRunner
        runner = BLiteRunner(env_type='simple')
        stats = runner.run_episode(max_steps=100, verbose=False)
        results['B-lite'] = {'V_mean': stats['V_mean'], 'type': 'noise injection'}
        print(f"B-lite: V_mean={stats['V_mean']:.3f}")
    except Exception as e:
        print(f"B-lite: {e}")
    
    # Self-Consistent (18.10)
    try:
        from self_consistent_system import SelfConsistentVariationalSystem
        system = SelfConsistentVariationalSystem(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
        env = SimpleEnv()
        actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        
        obs = env.reset()
        V_values = []
        
        for step in range(100):
            a = actions[np.random.randint(len(actions))]
            obs_next, _, done = env.step(a)
            state = system.step(obs, a, obs_next, train=True)
            V_values.append(state['V'])
            obs = obs_next
            if done:
                break
        
        results['Self-Consistent (18.10)'] = {'V_mean': np.mean(V_values), 'type': 'parametric variational'}
        print(f"Self-Consistent (18.10): V_mean={np.mean(V_values):.3f}")
    except Exception as e:
        print(f"Self-Consistent: {e}")
    
    # True Variational (18.11)
    print("\n--- True Variational (18.11) ---")
    model = TrueVariationalWorldModel(obs_dim=10, latent_dim=8, action_dim=2)
    env = SimpleEnv()
    actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
    
    obs = env.reset()
    F_values = []
    opt_errors = []
    
    for step in range(100):
        a = actions[np.random.randint(len(actions))]
        obs_next, _, done = env.step(a)
        state = model.forward(obs, a)
        F_values.append(state['functional'])
        opt_errors.append(state['optimality_error'])
        obs = obs_next
        if done:
            break
    
    results['True Variational (18.11)'] = {
        'F_mean': np.mean(F_values),
        'opt_error': np.mean(opt_errors),
        'type': 'single functional variational'
    }
    print(f"True Variational: F_mean={np.mean(F_values):.4f}, opt_error={np.mean(opt_errors):.4f}")
    
    # Summary
    print("\n--- SUMMARY ---")
    print(f"{'Approach':<30} {'Metric':<15} {'Value':<10}")
    print("-" * 55)
    for name, stats in results.items():
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{name:<30} {key:<15} {value:<10.4f}")


def stability_test():
    """Stability test across runs."""
    print("\n" + "=" * 60)
    print("STABILITY TEST (5 runs)")
    print("=" * 60)
    
    F_means = []
    opt_errors = []
    
    for run in range(5):
        model = TrueVariationalWorldModel(obs_dim=10, latent_dim=8, action_dim=2)
        env = SimpleEnv()
        actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        
        obs = env.reset()
        F_values = []
        opt_vals = []
        
        for step in range(50):
            a = actions[np.random.randint(len(actions))]
            obs_next, _, done = env.step(a)
            state = model.forward(obs, a)
            F_values.append(state['functional'])
            opt_vals.append(state['optimality_error'])
            obs = obs_next
            if done:
                break
        
        F_means.append(np.mean(F_values))
        opt_errors.append(np.mean(opt_vals))
        print(f"  Run {run+1}: F={np.mean(F_values):.4f}, opt_err={np.mean(opt_vals):.4f}")
    
    print(f"\nF mean: {np.mean(F_means):.4f} ± {np.std(F_means):.4f}")
    print(f"Opt error mean: {np.mean(opt_errors):.4f}")


if __name__ == '__main__':
    test_implicit_encoder()
    test_variational_energy_field()
    test_stability_spectrum()
    test_true_variational_model()
    test_convergence()
    test_stability_analysis()
    compare_with_previous()
    stability_test()
    
    print("\n" + "=" * 60)
    print("PHASE 18.11 TEST COMPLETE")
    print("=" * 60)
    print("\nKey achievements vs Phase 18.10:")
    print("1. ONE global functional F = V + λ₁R + λ₂C + λ₃H")
    print("2. Implicit encoder (via optimality condition)")
    print("3. Hessian spectrum as stability (not numerical Jacobian)")
    print("4. Joint optimization (θ,z) simultaneously")
    print("\nThis is TRUE variational system, not 'energy-shaped neural system'")