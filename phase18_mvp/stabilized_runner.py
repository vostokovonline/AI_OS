"""
Phase 18.7 - Stabilized World Model Runner

Тестирует систему с:
1. Latent contract enforcement
2. Mode population lifecycle
3. Unified loss
4. Energy V-field
"""
import numpy as np
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from stabilized_world_model import (
    StabilizedWorldModel,
    LatentContract,
    ModePopulation,
    ModeState
)
from env import SimpleEnv


def create_encoder(obs_dim=10, latent_dim=8):
    """Simple encoder: linear projection + normalize."""
    W = np.random.randn(latent_dim, obs_dim) * 0.1
    b = np.zeros(latent_dim)
    
    def encode(obs):
        obs = np.asarray(obs).flatten()
        z = W @ obs + b
        z = z / (np.linalg.norm(z) + 1e-6)  # unit norm
        return z[:latent_dim]
    
    return encode


def test_stabilized_model():
    """Test basic functionality."""
    print("=" * 60)
    print("STABILIZED WORLD MODEL TEST")
    print("=" * 60)
    
    # Create model and environment
    model = StabilizedWorldModel(latent_dim=8, action_dim=2, num_modes=3)
    env = SimpleEnv()
    encoder = create_encoder(obs_dim=10, latent_dim=8)
    
    # Actions
    actions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, -1.0]),
    ]
    
    print("\n--- Running Episode ---")
    
    obs = env.reset()
    z_prev = None
    
    for step in range(100):
        # Encode
        z = encoder(obs)
        
        # Get control signal
        control = model.get_control_signal(z, actions[0])  # Just for testing
        
        # Select action with highest mode weight
        best_action = actions[0]
        best_score = 0
        
        for a in actions:
            score = 0
            for mode in model.population.get_active_modes():
                predicted = mode.A @ z + mode.b @ a + mode.c
                # Use predicted error as score (lower = better)
                score += mode.weight / (np.linalg.norm(predicted - z) + 0.1)
            
            if score > best_score:
                best_score = score
                best_action = a
        
        # Execute
        obs_next, reward, done = env.step(best_action)
        z_next = encoder(obs_next)
        
        # Update model with true transition
        model.step(obs, best_action, z_next, encoder)
        
        # Log every 20 steps
        if step % 20 == 0:
            diag = model.get_diagnostic_state()
            pop = model.population.get_stats()
            print(f"  Step {step}: V={diag['V']:.3f}, "
                  f"modes={pop['active']}/{pop['total']}, "
                  f"z_norm={diag.get('z_norm', 0):.3f}, "
                  f"energy={diag.get('energy_stats', {}).get('mean', 0):.3f}")
        
        obs = obs_next
        if done:
            break
    
    # Final stats
    print("\n--- Final State ---")
    diag = model.get_diagnostic_state()
    pop = model.population.get_stats()
    
    print(f"V: {diag['V']:.3f}")
    print(f"Population: {pop['active']}/{pop['total']} active")
    print(f"Avg error: {pop['avg_error']:.3f}")
    print(f"Avg stability: {pop['avg_stability']:.3f}")
    print(f"Weights: {[f'{w:.3f}' for w in pop['weights']]}")
    
    return diag, pop


def test_latent_contract():
    """Test latent contract enforcement."""
    print("\n" + "=" * 60)
    print("LATENT CONTRACT TEST")
    print("=" * 60)
    
    contract = LatentContract(latent_dim=8, target_norm=1.0, max_drift=0.5)
    
    # Test normalization
    z1 = np.random.randn(8)
    z1_enforced = contract.enforce(z1)
    
    print(f"Before: norm={np.linalg.norm(z1):.3f}")
    print(f"After:  norm={np.linalg.norm(z1_enforced):.3f}")
    assert abs(np.linalg.norm(z1_enforced) - 1.0) < 0.01, "Norm should be 1.0"
    
    # Test drift check
    z2 = z1 + np.random.randn(8) * 0.3
    drift = contract.check_drift(z1, z2)
    print(f"Drift between z1 and z2: {drift:.3f}")
    
    # Test boundary case
    z3 = z1 + np.random.randn(8) * 0.6  # large change
    drift = contract.check_drift(z1, z3)
    print(f"Drift (large change): {drift:.3f}")


def test_mode_population():
    """Test mode population lifecycle."""
    print("\n" + "=" * 60)
    print("MODE POPULATION TEST")
    print("=" * 60)
    
    population = ModePopulation(latent_dim=8, action_dim=2, min_modes=2, max_modes=5)
    
    # Add some transitions
    for i in range(100):
        z = np.random.randn(8)
        a = np.array([np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])])
        z_next = np.random.randn(8) * 0.5 + 0.8 * z  # shift
        population.assign_transition(z, a, z_next)
        
        if i % 20 == 19:
            stats = population.get_stats()
            print(f"  Step {i+1}: active={stats['active']}, avg_error={stats['avg_error']:.3f}")
    
    # Update population (trigger death/merge)
    population.update_population()
    
    final_stats = population.get_stats()
    print(f"\nFinal: {final_stats['active']}/{final_stats['total']} modes active")
    
    return population


def test_unified_loss():
    """Test unified loss computation."""
    print("\n" + "=" * 60)
    print("UNIFIED LOSS TEST")
    print("=" * 60)
    
    from stabilized_world_model import UnifiedLoss, ModeIndividual
    
    loss_fn = UnifiedLoss()
    
    # Create a mock mode
    mode = ModeIndividual(
        id=0,
        A=np.random.randn(8, 8) * 0.1,
        b=np.random.randn(8, 2) * 0.2,
        c=np.zeros(8),
        weight=0.5,
        stability=0.7
    )
    
    contract = LatentContract(latent_dim=8)
    
    z = np.random.randn(8)
    a = np.array([1.0, 0.0])
    z_next = np.random.randn(8) * 0.2 + mode.A @ z + mode.b @ a
    
    loss = loss_fn.compute(z, a, z_next, mode, contract)
    
    print(f"Total loss: {loss['total']:.4f}")
    print(f"  reconstruction: {loss['reconstruction']:.4f}")
    print(f"  entropy: {loss['entropy']:.4f}")
    print(f"  stability: {loss['stability']:.4f}")
    print(f"  latent: {loss['latent']:.4f}")


def test_energy_field():
    """Test energy-based V-field."""
    print("\n" + "=" * 60)
    print("ENERGY FIELD TEST")
    print("=" * 60)
    
    from stabilized_world_model import EnergyVField, ModePopulation
    
    population = ModePopulation(latent_dim=8, action_dim=2)
    energy_field = EnergyVField(population)
    
    # Compute energy for random transitions
    energies = []
    for i in range(50):
        z = np.random.randn(8)
        a = np.array([1.0, 0.0])
        z_next = np.random.randn(8)
        
        energy = energy_field.compute_energy(z, a, z_next)
        energies.append(energy)
    
    print(f"Energy mean: {np.mean(energies):.3f}")
    print(f"Energy std: {np.std(energies):.3f}")
    
    # Test gradient computation
    z = np.random.randn(8)
    a = np.array([1.0, 0.0])
    grad = energy_field.compute_gradient(z, a)
    print(f"Gradient norm: {np.linalg.norm(grad):.3f}")
    
    stats = energy_field.get_energy_landscape_stats()
    print(f"Landscape: {stats}")


def stability_test(num_runs=5):
    """Test stability across multiple runs."""
    print("\n" + "=" * 60)
    print(f"STABILITY TEST ({num_runs} runs)")
    print("=" * 60)
    
    V_means = []
    V_mins = []
    mode_counts = []
    
    for run in range(num_runs):
        model = StabilizedWorldModel(latent_dim=8, action_dim=2, num_modes=3)
        env = SimpleEnv()
        encoder = create_encoder(obs_dim=10, latent_dim=8)
        
        actions = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
        ]
        
        obs = env.reset()
        
        for step in range(50):
            z = encoder(obs)
            
            # Simple random action
            a = actions[np.random.randint(len(actions))]
            
            obs_next, _, done = env.step(a)
            z_next = encoder(obs_next)
            
            model.step(obs, a, z_next, encoder)
            obs = obs_next
            
            if done:
                break
        
        diag = model.get_diagnostic_state()
        pop = model.population.get_stats()
        
        V_means.append(diag['V'])
        V_mins.append(diag.get('energy_stats', {}).get('min', 0))
        mode_counts.append(pop['active'])
        
        print(f"  Run {run+1}: V={diag['V']:.3f}, modes={pop['active']}")
    
    print(f"\nV mean: {np.mean(V_means):.3f} ± {np.std(V_means):.3f}")
    print(f"V min:  {np.min(V_mins):.3f}")
    print(f"Modes:  {np.mean(mode_counts):.1f} avg")


def compare_all():
    """Compare all three approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: All three approaches")
    print("=" * 60)
    
    try:
        from v3_runner import V3Runner
        print("\n--- MVP v3 ---")
        runner_v3 = V3Runner(latent_dim=8, action_dim=2)
        stats_v3 = runner_v3.run_episode(max_steps=100, verbose=False)
        print(f"V_mean={stats_v3['V_mean']:.3f}, V_min={stats_v3['V_min']:.3f}")
    except Exception as e:
        print(f"MVP v3 error: {e}")
        stats_v3 = {'V_mean': 0, 'V_min': 0}
    
    try:
        from blite_runner import BLiteRunner
        print("\n--- B-lite ---")
        runner_blite = BLiteRunner(env_type='simple')
        stats_blite = runner_blite.run_episode(max_steps=100, verbose=False)
        print(f"V_mean={stats_blite['V_mean']:.3f}, V_min={stats_blite['V_min']:.3f}")
    except Exception as e:
        print(f"B-lite error: {e}")
        stats_blite = {'V_mean': 0, 'V_min': 0}
    
    print("\n--- Stabilized WM ---")
    model = StabilizedWorldModel(latent_dim=8, action_dim=2, num_modes=3)
    env = SimpleEnv()
    encoder = create_encoder(obs_dim=10, latent_dim=8)
    
    actions = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
    ]
    
    obs = env.reset()
    V_values = []
    
    for step in range(100):
        z = encoder(obs)
        a = actions[np.random.randint(len(actions))]
        
        obs_next, _, done = env.step(a)
        z_next = encoder(obs_next)
        
        model.step(obs, a, z_next, encoder)
        
        diag = model.get_diagnostic_state()
        V_values.append(diag['V'])
        
        obs = obs_next
        if done:
            break
    
    stats_stab = {
        'V_mean': np.mean(V_values),
        'V_min': np.min(V_values)
    }
    print(f"V_mean={stats_stab['V_mean']:.3f}, V_min={stats_stab['V_min']:.3f}")
    
    print("\n--- SUMMARY ---")
    print(f"{'Approach':<15} {'V_mean':<10} {'V_min':<10}")
    print("-" * 35)
    print(f"{'MVP v3':<15} {stats_v3['V_mean']:<10.3f} {stats_v3['V_min']:<10.3f}")
    print(f"{'B-lite':<15} {stats_blite['V_mean']:<10.3f} {stats_blite['V_min']:<10.3f}")
    print(f"{'Stabilized WM':<15} {stats_stab['V_mean']:<10.3f} {stats_stab['V_min']:<10.3f}")


if __name__ == '__main__':
    # Run all tests
    test_latent_contract()
    test_mode_population()
    test_unified_loss()
    test_energy_field()
    
    test_stabilized_model()
    
    stability_test(num_runs=5)
    
    compare_all()
    
    print("\n" + "=" * 60)
    print("PHASE 18.7 TEST COMPLETE")
    print("=" * 60)