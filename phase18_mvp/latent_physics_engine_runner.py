"""
Phase 18.9 - Latent Physics Engine Runner

Тестирует физически согласованную систему.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from latent_physics_engine import (
    LatentPhysicsEngine,
    TruePotentialField,
    PhysicsConsistentFlowField
)
from env import SimpleEnv


def test_potential_field():
    """Test true potential field."""
    print("=" * 60)
    print("TRUE POTENTIAL FIELD TEST")
    print("=" * 60)
    
    potential = TruePotentialField(latent_dim=8)
    
    # Add structured transitions (spiral trajectory)
    for i in range(100):
        t = i / 20
        z = np.zeros(8)
        z[0] = t * np.cos(t)
        z[1] = t * np.sin(t)
        
        z_next = z.copy()
        z_next[0] += 0.5 * np.cos(t + 0.5)
        z_next[1] += 0.5 * np.sin(t + 0.5)
        
        a = np.array([np.cos(t), np.sin(t)])
        
        potential.add_transition(z, a, z_next)
        
        if i % 20 == 0:
            E = potential.compute_E(z)
            print(f"  t={t:.2f}: E={E:.3f}")
    
    # Test gradient
    z_test = np.array([2.0, 2.0] + [0] * 6)
    grad_E = potential.compute_gradient_E(z_test)
    print(f"\ngrad_E at z=[2,2]: |grad_E|={np.linalg.norm(grad_E):.3f}")


def test_consistency_loss():
    """Test consistency loss."""
    print("\n" + "=" * 60)
    print("CONSISTENCY LOSS TEST")
    print("=" * 60)
    
    engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
    
    # Add transitions
    for i in range(100):
        z = np.random.randn(8)
        a = np.array([1.0, 0.0])
        z_next = z + np.random.randn(8) * 0.5
        engine.observe(z, a, z_next)
    
    # Compute loss
    z = np.random.randn(8)
    a = np.array([1.0, 0.0])
    z_next = z + np.random.randn(8) * 0.1
    
    physics = engine.get_physics(z, a)
    loss = engine.compute_loss(z, a, z_next)
    
    print(f"Total loss: {loss['total']:.4f}")
    print(f"  flow_potential: {loss['flow_potential']:.4f}")
    print(f"  transition: {loss['transition']:.4f}")
    print(f"  attractor: {loss['attractor']:.4f}")


def test_flow_potential_consistency():
    """Test that flow follows potential."""
    print("\n" + "=" * 60)
    print("FLOW-POTENTIAL CONSISTENCY TEST")
    print("=" * 60)
    
    engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
    
    # Add transitions
    for i in range(200):
        z = np.random.randn(8)
        a = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
        z_next = z + np.random.randn(8) * 0.5
        engine.observe(z, a, z_next)
    
    # Check consistency
    inconsistencies = []
    for _ in range(50):
        z = np.random.randn(8) * 3
        a = np.array([1.0, 0.0])
        
        physics = engine.get_physics(z, a)
        deviation = np.linalg.norm(physics['F'] + physics['grad_E'])
        inconsistencies.append(deviation)
    
    print(f"Mean deviation ||F + grad_E||: {np.mean(inconsistencies):.3f}")
    return np.mean(inconsistencies)


def test_attractor_detection():
    """Test attractor detection."""
    print("\n" + "=" * 60)
    print("ATTRACTOR DETECTION TEST")
    print("=" * 60)
    
    engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
    
    # Create attractor positions
    attractor_positions = [
        np.array([3.0, 0.0] + [0] * 6),
        np.array([-3.0, 0.0] + [0] * 6),
        np.array([0.0, 3.0] + [0] * 6),
    ]
    
    # Add transitions toward attractors
    for _ in range(30):
        for attractor in attractor_positions:
            z = attractor + np.random.randn(8) * 1.5
            z_next = attractor + np.random.randn(8) * 0.3
            a = np.array([0.0, 0.0])
            engine.observe(z, a, z_next)
    
    # Detect attractors
    engine.flow_field.potential.detect_attractors()
    
    print(f"Detected {len(engine.flow_field.potential.attractors)} attractors")
    for i, attractor in enumerate(engine.flow_field.potential.attractors):
        print(f"  Attractor {i}: pos=[{attractor.position[0]:.2f}, {attractor.position[1]:.2f}], "
              f"strength={attractor.strength:.2f}, radius={attractor.basin_radius:.2f}")


def test_trajectory_integration():
    """Test evolving through flow field."""
    print("\n" + "=" * 60)
    print("TRAJECTORY INTEGRATION TEST")
    print("=" * 60)
    
    engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
    
    # Build field
    for i in range(100):
        z = np.random.randn(8)
        a = np.array([0.0, 0.0])
        z_next = z + np.random.randn(8) * 0.5
        engine.observe(z, a, z_next)
    
    # Start at high energy point
    z_start = np.array([5.0, 5.0] + [0] * 6)
    
    print(f"Start: z={z_start[:2]}, E={engine.flow_field.potential.compute_E(z_start):.3f}")
    
    # Evolve
    z = z_start.copy()
    for t in range(30):
        z = engine.evolve(z, None, dt=0.2)
        if t % 10 == 0:
            E = engine.flow_field.potential.compute_E(z)
            print(f"  t={t}: z=[{z[0]:.2f}, {z[1]:.2f}], E={E:.3f}")
    
    print(f"Final: z=[{z[0]:.2f}, {z[1]:.2f}]")


def test_full_physics_engine():
    """Test full LatentPhysicsEngine."""
    print("\n" + "=" * 60)
    print("LATENT PHYSICS ENGINE TEST")
    print("=" * 60)
    
    engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
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
        obs_next, reward, done = env.step(a)
        
        z = np.random.randn(8) * 0.5
        z_next = np.random.randn(8) * 0.5
        
        engine.observe(z, a, z_next)
        
        if step % 20 == 0:
            physics = engine.get_physics(z, a)
            print(f"  Step {step}: E={physics['E']:.3f}, "
                  f"|F|={np.linalg.norm(physics['F']):.3f}, "
                  f"stability={physics['stability']:.3f}")
    
    state = engine.get_state()
    print(f"\n--- Final State ---")
    print(f"Steps: {state['step_count']}, Transitions: {state['num_transitions']}")
    print(f"Avg loss: {state['avg_loss']:.4f}")


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
        results['B-lite'] = {'V_mean': stats['V_mean'], 'V_min': stats['V_min']}
        print(f"B-lite: V_mean={stats['V_mean']:.3f}")
    except Exception as e:
        print(f"B-lite: {e}")
    
    # Flow Field WM
    try:
        from flow_field_world_model import FlowFieldWorldModel
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
        
        results['Flow Field WM'] = {'V_mean': np.mean(V_values), 'V_min': np.min(V_values)}
        print(f"Flow Field WM: V_mean={np.mean(V_values):.3f}")
    except Exception as e:
        print(f"Flow Field WM: {e}")
    
    # Latent Physics Engine
    print("\n--- Latent Physics Engine ---")
    engine = LatentPhysicsEngine(latent_dim=8, action_dim=2)
    env = SimpleEnv()
    actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
    
    obs = env.reset()
    E_values = []
    losses = []
    
    for step in range(100):
        a = actions[np.random.randint(len(actions))]
        obs_next, _, done = env.step(a)
        
        z = np.random.randn(8) * 0.5
        z_next = np.random.randn(8) * 0.5
        
        engine.observe(z, a, z_next)
        physics = engine.get_physics(z, a)
        E_values.append(physics['E'])
        loss = engine.compute_loss(z, a, z_next)
        losses.append(loss['total'])
        
        obs = obs_next
        if done:
            break
    
    results['Latent Physics'] = {
        'E_mean': 1 - np.mean(E_values),
        'E_min': 1 - np.min(E_values),
        'consistency': np.mean(losses)
    }
    print(f"Latent Physics: E(converted to V)={1-np.mean(E_values):.3f}, consistency={np.mean(losses):.4f}")
    
    print("\n--- SUMMARY ---")
    for name, stats in results.items():
        for key, value in stats.items():
            print(f"{name:<20} {key:<15} {value:.3f}")


if __name__ == '__main__':
    test_potential_field()
    test_consistency_loss()
    test_flow_potential_consistency()
    test_attractor_detection()
    test_trajectory_integration()
    test_full_physics_engine()
    compare_all()
    
    print("\n" + "=" * 60)
    print("PHASE 18.9 TEST COMPLETE")
    print("=" * 60)