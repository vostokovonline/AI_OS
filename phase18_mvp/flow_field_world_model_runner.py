"""
Phase 18.8 - Flow Field World Model Runner

Тестирует геометрически согласованную систему.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from flow_field_world_model import (
    FlowFieldWorldModel,
    ContinuousVField,
    SoftModeMixture,
    EnergyConsistentEncoder
)
from env import SimpleEnv


def test_energy_encoder():
    """Test energy-consistent encoder."""
    print("=" * 60)
    print("ENERGY-CONSISTENT ENCODER TEST")
    print("=" * 60)
    
    # Create mock energy field
    energy_field = ContinuousVField(latent_dim=8, action_dim=2)
    
    # Add some transitions
    for i in range(50):
        z = np.random.randn(8) * (1 + i/50)
        a = np.array([1.0, 0.0])
        z_next = z + np.random.randn(8) * 0.5
        energy_field.add_transition(z, a, z_next)
    
    # Create encoder
    encoder = EnergyConsistentEncoder(
        obs_dim=10,
        latent_dim=8,
        energy_field=energy_field,
        lambda_encoder=0.1
    )
    
    # Encode with and without optimization
    obs = np.random.randn(10)
    
    # Simple encoding
    z_simple = encoder.W @ obs + encoder.b
    V_simple = energy_field.compute_V(z_simple, None)
    
    # Energy-consistent encoding
    z_opt = encoder.encode(obs, None, n_iterations=5)
    V_opt = energy_field.compute_V(z_opt, None)
    
    print(f"Simple encoding V: {V_simple:.3f}")
    print(f"Optimized encoding V: {V_opt:.3f}")
    print(f"Improvement: {(V_simple - V_opt):.3f}")
    print(f"Encoder stats: {encoder.get_state()}")


def test_soft_mode_mixture():
    """Test soft mode mixture."""
    print("\n" + "=" * 60)
    print("SOFT MODE MIXTURE TEST")
    print("=" * 60)
    
    mixture = SoftModeMixture(latent_dim=8, action_dim=2, num_modes=3)
    
    # Add transitions with different patterns
    for i in range(100):
        z = np.random.randn(8)
        
        # Different patterns for different transitions
        if i < 33:
            pattern = 'left'
            z_next = z + np.array([-0.5, 0.2] + [0] * 6) + np.random.randn(8) * 0.2
        elif i < 66:
            pattern = 'right'
            z_next = z + np.array([0.5, 0.2] + [0] * 6) + np.random.randn(8) * 0.2
        else:
            pattern = 'up'
            z_next = z + np.array([0, 0.5] + [0] * 6) + np.random.randn(8) * 0.2
        
        a = np.array([1.0, 0.0])
        
        # Compute responsibilities
        resp = mixture.compute_responsibilities(z, a, z_next)
        
        # Update modes
        mixture.update(z, a, z_next, resp)
        
        if i % 25 == 0:
            print(f"  Step {i}: resp={[f'{r:.2f}' for r in resp]}, "
                  f"dirs={[f'{np.linalg.norm(m.direction):.2f}' for m in mixture.modes]}")
    
    # Final stats
    stats = mixture.get_mode_stats()
    print(f"\nFinal mode stats:")
    print(f"  Responsibilities: {[f'{r:.3f}' for r in stats['responsibilities']]}")
    print(f"  Entropy: {stats['total_entropy']:.3f}")
    
    return mixture


def test_continuous_vfield():
    """Test continuous V field."""
    print("\n" + "=" * 60)
    print("CONTINUOUS V FIELD TEST")
    print("=" * 60)
    
    vfield = ContinuousVField(latent_dim=8, action_dim=2)
    
    # Add structured transitions
    for i in range(100):
        z = np.zeros(8)
        z[0] = i / 20  # moving in x direction
        z[1] = np.sin(i / 10)  # oscillating
        
        a = np.array([1.0, 0.0])
        z_next = z.copy()
        z_next[0] += 0.5
        z_next += np.random.randn(8) * 0.1
        
        vfield.add_transition(z, a, z_next)
    
    # Test V at different points
    test_points = [
        np.array([0.0, 0.0] + [0] * 6),
        np.array([2.5, 0.0] + [0] * 6),
        np.array([5.0, 0.0] + [0] * 6),
        np.array([0.0, 1.0] + [0] * 6),
    ]
    
    print("\nV at different points:")
    for z in test_points:
        V = vfield.compute_V(z, None)
        grad_V = vfield.compute_gradient_V(z, None)
        flow = vfield.get_flow(z, None)
        print(f"  z={z[:2]}: V={V:.3f}, |grad_V|={np.linalg.norm(grad_V):.3f}, |flow|={np.linalg.norm(flow):.3f}")
    
    return vfield


def test_flow_field_world_model():
    """Test full Flow Field World Model."""
    print("\n" + "=" * 60)
    print("FLOW FIELD WORLD MODEL TEST")
    print("=" * 60)
    
    model = FlowFieldWorldModel(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
    env = SimpleEnv()
    
    actions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    
    print("\n--- Running Episode ---")
    
    obs = env.reset()
    
    for step in range(100):
        a = actions[np.random.randint(len(actions))]
        
        obs_next, reward, done = env.step(a)
        
        # Step with training
        state = model.step(obs, a, obs_next, encode_iterations=5)
        
        if step % 20 == 0:
            field_data = model.get_field_data()
            print(f"  Step {step}: V={state['V']:.3f}, "
                  f"flow_norm={np.linalg.norm(state['flow']):.3f}, "
                  f"density={state['local_density']:.3f}, "
                  f"uncertainty={state['uncertainty']:.3f}")
            print(f"           mode_resp={[f'{r:.2f}' for r in field_data['mode_responsibilities']]}")
        
        obs = obs_next
        if done:
            break
    
    # Get final state
    final_state = model.get_state()
    print(f"\n--- Final State ---")
    print(f"V: {final_state['V']:.3f}")
    print(f"Flow norm: {np.linalg.norm(final_state['flow']):.3f}")
    print(f"Step count: {final_state['step_count']}")
    print(f"Encoder alignment: {final_state['encoder_stats']['alignment_score']:.3f}")
    
    return model


def test_gradient_flow():
    """Test gradient flow dynamics."""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW DYNAMICS TEST")
    print("=" * 60)
    
    model = FlowFieldWorldModel(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
    
    # Add many transitions to build field
    for i in range(200):
        z = np.random.randn(8)
        a = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
        z_next = z + np.random.randn(8) * 0.5
        model.V_field.add_transition(z, a, z_next)
    
    # Start at high energy point
    z_start = np.array([5.0, 5.0] + [0] * 6)
    
    print(f"Start point: z={z_start[:2]}")
    print(f"V at start: {model.V_field.compute_V(z_start, None):.3f}")
    
    # Simulate gradient flow
    z_current = z_start.copy()
    trajectory = [z_current.copy()]
    
    for t in range(50):
        flow = model.V_field.get_flow(z_current, None)
        
        # Gradient flow: z_next = z - alpha * grad_V = z + flow
        alpha = 0.3
        z_current = z_current + alpha * flow
        z_current = z_current / (np.linalg.norm(z_current) + 1e-6)  # normalize
        
        trajectory.append(z_current.copy())
        
        if t % 10 == 0:
            V = model.V_field.compute_V(z_current, None)
            print(f"  t={t}: |z|={np.linalg.norm(z_current):.3f}, V={V:.3f}")
    
    print(f"\nFinal: z={z_current[:2]}")
    print(f"V at end: {model.V_field.compute_V(z_current, None):.3f}")
    print(f"Trajectory length: {len(trajectory)}")


def test_soft_vs_hard():
    """Compare soft vs hard mode assignment."""
    print("\n" + "=" * 60)
    print("SOFT VS HARD MODE ASSIGNMENT TEST")
    print("=" * 60)
    
    # Soft mixture
    soft_mixture = SoftModeMixture(latent_dim=8, action_dim=2, num_modes=3, temperature=1.0)
    
    # Hard assignment (baseline)
    hard_predictions = []
    
    # Add transitions
    z_history = []
    z_next_history = []
    
    for i in range(100):
        z = np.random.randn(8)
        z_next = z + np.array([0.5, 0.3] + [0] * 6) + np.random.randn(8) * 0.2
        a = np.array([1.0, 0.0])
        
        soft_mixture.V_field.add_transition(z, a, z_next)
        
        # Hard assignment (argmin)
        best_mode = 0
        best_error = float('inf')
        for j, mode in enumerate(soft_mixture.modes):
            pred = z + mode.direction
            error = np.linalg.norm(z_next - pred)
            if error < best_error:
                best_error = error
                best_mode = j
        hard_predictions.append(best_mode)
        
        z_history.append(z)
        z_next_history.append(z_next)
    
    # Soft prediction
    soft_preds = []
    for z, z_next in zip(z_history[:10], z_next_history[:10]):
        pred, resp = soft_mixture.predict(z, a)
        soft_preds.append(resp)
    
    print(f"Hard mode distribution: {np.bincount(hard_predictions, minlength=3)}")
    print(f"Soft responsibilities (first 10): {[np.argmax(p) for p in soft_preds]}")
    
    # Compute prediction error
    soft_errors = []
    hard_errors = []
    
    for i in range(50):
        z = z_history[i]
        z_next = z_next_history[i]
        
        # Soft prediction
        pred_soft, _ = soft_mixture.predict(z, a)
        soft_errors.append(np.linalg.norm(z_next - pred_soft))
        
        # Hard prediction (use best mode)
        best_mode_idx = hard_predictions[i]
        pred_hard = z + soft_mixture.modes[best_mode_idx].direction
        hard_errors.append(np.linalg.norm(z_next - pred_hard))
    
    print(f"\nSoft prediction error: {np.mean(soft_errors):.3f} ± {np.std(soft_errors):.3f}")
    print(f"Hard prediction error: {np.mean(hard_errors):.3f} ± {np.std(hard_errors):.3f}")


def stability_test():
    """Test stability across runs."""
    print("\n" + "=" * 60)
    print("STABILITY TEST (5 runs)")
    print("=" * 60)
    
    V_values = []
    
    for run in range(5):
        model = FlowFieldWorldModel(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
        env = SimpleEnv()
        
        actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        
        obs = env.reset()
        Vs = []
        
        for step in range(50):
            a = actions[np.random.randint(len(actions))]
            obs_next, _, done = env.step(a)
            
            state = model.step(obs, a, obs_next, encode_iterations=3)
            Vs.append(state['V'])
            
            obs = obs_next
            if done:
                break
        
        V_values.append(np.mean(Vs))
        print(f"  Run {run+1}: V_mean={np.mean(Vs):.3f}")
    
    print(f"\nOverall V_mean: {np.mean(V_values):.3f} ± {np.std(V_values):.3f}")


def compare_all():
    """Compare all approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: All approaches")
    print("=" * 60)
    
    approaches = {}
    
    # B-lite
    try:
        from blite_runner import BLiteRunner
        runner = BLiteRunner(env_type='simple')
        stats = runner.run_episode(max_steps=100, verbose=False)
        approaches['B-lite'] = {'V_mean': stats['V_mean'], 'V_min': stats['V_min']}
    except Exception as e:
        print(f"B-lite: {e}")
    
    # Flow Field World Model
    print("\n--- Flow Field World Model ---")
    model = FlowFieldWorldModel(obs_dim=10, latent_dim=8, action_dim=2, num_modes=3)
    env = SimpleEnv()
    actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
    
    obs = env.reset()
    V_values = []
    
    for step in range(100):
        a = actions[np.random.randint(len(actions))]
        obs_next, _, done = env.step(a)
        state = model.step(obs, a, obs_next, encode_iterations=3)
        V_values.append(state['V'])
        obs = obs_next
        if done:
            break
    
    approaches['Flow Field WM'] = {
        'V_mean': np.mean(V_values),
        'V_min': np.min(V_values)
    }
    print(f"V_mean={np.mean(V_values):.3f}, V_min={np.min(V_values):.3f}")
    
    # Summary
    print("\n--- SUMMARY ---")
    print(f"{'Approach':<20} {'V_mean':<10} {'V_min':<10}")
    print("-" * 40)
    for name, stats in approaches.items():
        print(f"{name:<20} {stats['V_mean']:<10.3f} {stats['V_min']:<10.3f}")


if __name__ == '__main__':
    test_energy_encoder()
    test_soft_mode_mixture()
    test_continuous_vfield()
    test_flow_field_world_model()
    test_gradient_flow()
    test_soft_vs_hard()
    stability_test()
    compare_all()
    
    print("\n" + "=" * 60)
    print("PHASE 18.8 TEST COMPLETE")
    print("=" * 60)