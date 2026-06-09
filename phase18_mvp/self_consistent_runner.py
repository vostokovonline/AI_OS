"""
Phase 18.10 - Self-Consistent Variational System Runner

Тестирует полностью self-consistent variational system.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from self_consistent_system import (
    SelfConsistentVariationalSystem,
    ParametricPotentialField,
    SelfConsistentEncoder,
    SpectralModeBasis
)
from env import SimpleEnv


def test_parametric_potential():
    """Test neural potential field."""
    print("=" * 60)
    print("PARAMETRIC POTENTIAL FIELD TEST")
    print("=" * 60)
    
    potential = ParametricPotentialField(latent_dim=8, action_dim=2)
    
    # Test forward
    z = np.random.randn(8)
    a = np.array([1.0, 0.0])
    
    V = potential.forward(z, a)
    print(f"V(z, a) = {V:.3f}")
    
    # Test gradient
    grad_V = potential.compute_gradient(z, a)
    print(f"|grad V| = {np.linalg.norm(grad_V):.3f}")
    
    # Test flow
    F = potential.get_flow(z, a)
    print(f"|F| = {np.linalg.norm(F):.3f}")
    
    # Test update
    print("\nUpdating potential...")
    potential.update(z, a, target_V=0.5, lr=0.01)
    V_new = potential.forward(z, a)
    print(f"V after update: {V_new:.3f}")
    
    return potential


def test_self_consistent_encoder():
    """Test encoder with relaxation."""
    print("\n" + "=" * 60)
    print("SELF-CONSISTENT ENCODER TEST")
    print("=" * 60)
    
    potential = ParametricPotentialField(latent_dim=8, action_dim=2)
    encoder = SelfConsistentEncoder(obs_dim=10, latent_dim=8, potential_field=potential)
    
    obs = np.random.randn(10)
    a = np.array([1.0, 0.0])
    
    # Simple encoding
    z_simple = encoder.encode_simple(obs)
    print(f"Simple encoding: |z|={np.linalg.norm(z_simple):.3f}")
    
    # Relaxed encoding
    z_relaxed = encoder.encode(obs, a)
    print(f"Relaxed encoding: |z|={np.linalg.norm(z_relaxed):.3f}")
    
    # Fixed point error
    error = encoder.compute_fixed_point_error(obs, a)
    print(f"Fixed point error: {error:.3f}")
    
    # State
    state = encoder.get_state()
    print(f"Convergence rate: {state['convergence_rate']:.2f}")
    
    return encoder


def test_spectral_mode_basis():
    """Test spectral mode basis."""
    print("\n" + "=" * 60)
    print("SPECTRAL MODE BASIS TEST")
    print("=" * 60)
    
    modes = SpectralModeBasis(latent_dim=8, num_modes=3)
    
    # Simulate Jacobian
    J = np.random.randn(8, 8) * 0.5
    J = (J + J.T) / 2  # symmetric
    
    # Update modes
    modes.update_from_jacobian(J)
    
    print("Mode eigenvalues:", [f"{ev:.3f}" for ev in modes.eigenvalues])
    print("Mode strengths:", [f"{s:.3f}" for s in modes.strengths])
    
    # Project random vector
    v = np.random.randn(8)
    coeffs = modes.project_onto_modes(v)
    print(f"Projection coefficients: {[f'{c:.3f}' for c in coeffs]}")
    
    # Reconstruct
    v_reconstructed = modes.reconstruct_from_modes(coeffs)
    print(f"Reconstruction error: {np.linalg.norm(v - v_reconstructed):.3f}")
    
    # Stabilities
    stabilities = modes.get_stability()
    print(f"Mode stabilities: {[f'{s:.3f}' for s in stabilities]}")
    
    return modes


def test_full_system():
    """Test full self-consistent system."""
    print("\n" + "=" * 60)
    print("SELF-CONSISTENT VARIATIONAL SYSTEM TEST")
    print("=" * 60)
    
    system = SelfConsistentVariationalSystem(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
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
        
        # Step
        state = system.step(obs, a, obs_next, train=True)
        
        if step % 20 == 0:
            print(f"  Step {step}: V={state['V']:.3f}, "
                  f"|F|={state['flow_magnitude']:.3f}, "
                  f"fixed_err={state['fixed_point_error']:.3f}, "
                  f"loss={state['total_loss']:.4f}")
        
        obs = obs_next
        if done:
            break
    
    # Get final state
    final = system.get_state()
    print(f"\n--- Final State ---")
    print(f"Steps: {final['step_count']}")
    print(f"V mean: {final['V_mean']:.3f}")
    print(f"Loss mean: {final['loss_mean']:.4f}")
    print(f"Fixed point error: {final['fixed_point_error_mean']:.4f}")
    print(f"Mode stabilities: {[f'{s:.3f}' for s in final['mode_stabilities']]}")
    
    return system


def test_self_consistency():
    """Test self-consistency of system."""
    print("\n" + "=" * 60)
    print("SELF-CONSISTENCY TEST")
    print("=" * 60)
    
    system = SelfConsistentVariationalSystem(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
    
    # Build up some history
    for _ in range(200):
        obs = np.random.randn(10)
        a = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
        system.step(obs, a, None, train=True)
    
    # Check consistency
    errors = []
    for _ in range(50):
        obs = np.random.randn(10)
        a = np.array([1.0, 0.0])
        
        error = system.check_fixed_point(obs, a)
        errors.append(error)
    
    print(f"Fixed point errors: mean={np.mean(errors):.4f}, std={np.std(errors):.4f}")
    print(f"Converged: {sum(1 for e in errors if e < 0.1)}/{len(errors)}")


def test_evolved_trajectory():
    """Test trajectory evolution."""
    print("\n" + "=" * 60)
    print("EVOLVED TRAJECTORY TEST")
    print("=" * 60)
    
    system = SelfConsistentVariationalSystem(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
    
    # Build field
    for _ in range(100):
        obs = np.random.randn(10)
        a = np.array([0.0, 0.0])
        system.step(obs, a, None, train=True)
    
    # Start at specific point
    z_start = np.array([3.0, 3.0] + [0] * 6)
    
    print(f"Start: z={z_start[:2]}, V={system.compute_V(z_start, None):.3f}")
    
    # Evolve
    z = z_start.copy()
    trajectory = [z.copy()]
    Vs = [system.compute_V(z, None)]
    
    for t in range(30):
        z = system.evolve(z, None, dt=0.1)
        trajectory.append(z.copy())
        Vs.append(system.compute_V(z, None))
        
        if t % 10 == 0:
            print(f"  t={t}: z=[{z[0]:.2f}, {z[1]:.2f}], V={Vs[-1]:.3f}")
    
    print(f"Final: z=[{z[0]:.2f}, {z[1]:.2f}], V={Vs[-1]:.3f}")
    print(f"V change: {Vs[0]:.3f} → {Vs[-1]:.3f}")


def stability_test():
    """Test stability across runs."""
    print("\n" + "=" * 60)
    print("STABILITY TEST (5 runs)")
    print("=" * 60)
    
    V_means = []
    loss_means = []
    fp_errors = []
    
    for run in range(5):
        system = SelfConsistentVariationalSystem(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
        env = SimpleEnv()
        actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        
        obs = env.reset()
        
        for step in range(50):
            a = actions[np.random.randint(len(actions))]
            obs_next, _, done = env.step(a)
            state = system.step(obs, a, obs_next, train=True)
            obs = obs_next
            if done:
                break
        
        final = system.get_state()
        V_means.append(final['V_mean'])
        loss_means.append(final['loss_mean'])
        fp_errors.append(final['fixed_point_error_mean'])
        
        print(f"  Run {run+1}: V={final['V_mean']:.3f}, loss={final['loss_mean']:.4f}, fp_err={final['fixed_point_error_mean']:.4f}")
    
    print(f"\nV mean: {np.mean(V_means):.3f} ± {np.std(V_means):.3f}")
    print(f"Loss mean: {np.mean(loss_means):.4f}")
    print(f"FP error mean: {np.mean(fp_errors):.4f}")


def compare_all():
    """Compare all approaches."""
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
    
    # Latent Physics Engine
    try:
        from latent_physics_engine import LatentPhysicsEngine
        engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
        env = SimpleEnv()
        actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        
        obs = env.reset()
        E_values = []
        
        for step in range(100):
            a = actions[np.random.randint(len(actions))]
            obs_next, _, done = env.step(a)
            z = np.random.randn(8) * 0.5
            z_next = np.random.randn(8) * 0.5
            engine.observe(z, a, z_next)
            physics = engine.get_physics(z, a)
            E_values.append(physics['E'])
            obs = obs_next
            if done:
                break
        
        results['Latent Physics'] = {'V_mean': 1 - np.mean(E_values), 'type': 'KDE-based'}
        print(f"Latent Physics: V=1-E={1-np.mean(E_values):.3f}")
    except Exception as e:
        print(f"Latent Physics: {e}")
    
    # Self-Consistent Variational System
    print("\n--- Self-Consistent Variational ---")
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
    
    results['Self-Consistent'] = {
        'V_mean': np.mean(V_values),
        'type': 'parametric variational'
    }
    print(f"Self-Consistent: V_mean={np.mean(V_values):.3f}")
    
    # Summary
    print("\n--- SUMMARY ---")
    print(f"{'Approach':<25} {'V_mean':<10} {'Type':<25}")
    print("-" * 60)
    for name, stats in results.items():
        print(f"{name:<25} {stats['V_mean']:<10.3f} {stats['type']:<25}")


if __name__ == '__main__':
    test_parametric_potential()
    test_self_consistent_encoder()
    test_spectral_mode_basis()
    test_full_system()
    test_self_consistency()
    test_evolved_trajectory()
    stability_test()
    compare_all()
    
    print("\n" + "=" * 60)
    print("PHASE 18.10 TEST COMPLETE")
    print("=" * 60)
    print("\nKey achievement: Parametric potential field (no KDE buffer)")
    print("Self-consistent encoder (fixed point convergence)")
    print("Spectral mode basis (eigenmodes of flow Jacobian)")
    print("\nThis is 'synthetic physics' - learned potential + fixed point dynamics")