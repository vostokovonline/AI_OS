"""
Phase 18.7 - Flow Field Layer Runner

Тестирует Flow Field Layer — geometry-first подход к V-field.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from flow_field_layer import (
    FlowFieldLayer,
    LatentTrajectoryBuffer,
    KNNGraphBuilder,
    FieldEstimator,
    VectorField
)
from env import SimpleEnv


def create_encoder(obs_dim=10, latent_dim=8):
    """Simple encoder."""
    W = np.random.randn(latent_dim, obs_dim) * 0.1
    b = np.zeros(latent_dim)
    
    def encode(obs):
        obs = np.asarray(obs).flatten()
        z = W @ obs + b
        z = z / (np.linalg.norm(z) + 1e-6)
        return z[:latent_dim]
    
    return encode


def test_latent_buffer():
    """Test trajectory buffer."""
    print("=" * 60)
    print("LATENT BUFFER TEST")
    print("=" * 60)
    
    buffer = LatentTrajectoryBuffer(latent_dim=8, max_size=100)
    
    # Add points
    for i in range(50):
        z = np.random.randn(8) * (1 + i / 50)  # drifting
        a = np.array([1.0, 0.0])
        z_next = z + np.random.randn(8) * 0.5
        
        buffer.add(z, a, z_next)
        
        if i % 10 == 9:
            print(f"  Points: {buffer.size()}, Latest z norm: {np.linalg.norm(z):.3f}")
    
    # Get recent
    recent = buffer.get_recent(10)
    print(f"\nRecent shape: {recent.shape}")
    
    return buffer


def test_knn_builder():
    """Test kNN graph builder."""
    print("\n" + "=" * 60)
    print("KNN GRAPH BUILDER TEST")
    print("=" * 60)
    
    builder = KNNGraphBuilder(k=5)
    
    # Generate points in latent space
    np.random.seed(42)
    Z = np.random.randn(100, 8)
    
    # Add some structure (clusters)
    for i in range(3):
        cluster_center = np.random.randn(8) * 3
        cluster_points = np.random.randn(20, 8) * 0.5 + cluster_center
        Z = np.vstack([Z, cluster_points])
    
    print(f"Points: {len(Z)}")
    
    # Compute kNN
    distances, indices = builder.compute_knn(Z)
    print(f"Distances shape: {distances.shape}")
    print(f"Mean kNN distance: {np.mean(distances):.3f}")
    
    # Build neighborhood for random point
    center_idx = np.random.randint(len(Z))
    neighborhood = builder.build_neighborhood(Z, center_idx)
    
    print(f"\nNeighborhood for point {center_idx}:")
    print(f"  k neighbors: {neighborhood.k}")
    print(f"  Local density: {neighborhood.local_density:.3f}")
    print(f"  Tangent basis shape: {neighborhood.tangent_basis.shape}")
    
    return builder


def test_field_estimator():
    """Test field estimation."""
    print("\n" + "=" * 60)
    print("FIELD ESTIMATOR TEST")
    print("=" * 60)
    
    buffer = test_latent_buffer()
    builder = KNNGraphBuilder(k=5)
    estimator = FieldEstimator(builder)
    
    # Add more structured data
    for i in range(50):
        # Generate points on a manifold (spiral)
        t = i / 10
        z = np.zeros(8)
        z[0] = t * np.cos(t)
        z[1] = t * np.sin(t)
        
        buffer.add(z, np.array([1.0, 0.0]), z + np.random.randn(8) * 0.1)
    
    print(f"\nBuffer size: {buffer.size()}")
    
    # Estimate V at various points
    test_points = [
        np.zeros(8),
        np.array([5.0, 5.0] + [0] * 6),
        np.array([1.0, 0.0] + [0] * 6)
    ]
    
    for z in test_points:
        V = estimator.estimate_V(z, buffer)
        print(f"  V({z[:2]}...) = {V:.3f}")
    
    # Gradient
    z = np.array([1.0, 1.0] + [0] * 6)
    grad_V = estimator.estimate_gradient_V(z, buffer)
    print(f"\n  grad V at z=[1,1,...]: norm={np.linalg.norm(grad_V):.3f}")
    
    return estimator, buffer


def test_vector_field():
    """Test vector field computation."""
    print("\n" + "=" * 60)
    print("VECTOR FIELD TEST")
    print("=" * 60)
    
    builder = KNNGraphBuilder(k=10)
    estimator = FieldEstimator(builder)
    vector_field = VectorField(builder, estimator)
    
    # Build buffer with transitions
    buffer = LatentTrajectoryBuffer(latent_dim=8, max_size=200)
    
    # Generate structured transitions (moving in a direction)
    for i in range(100):
        z = np.random.randn(8)
        z[0] = i / 20  # increasing in x direction
        
        a = np.array([1.0, 0.0])
        z_next = z.copy()
        z_next[0] += 0.5  # move right
        z_next += np.random.randn(8) * 0.2
        
        buffer.add(z, a, z_next)
    
    print(f"Buffer: {buffer.size()} points")
    
    # Compute flow at origin
    z = np.zeros(8)
    a = np.array([1.0, 0.0])
    flow = vector_field.compute_flow(z, a, buffer)
    
    print(f"\nFlow at z=0, a=[1,0]:")
    print(f"  Flow vector: [{flow[0]:.3f}, {flow[1]:.3f}, ...]")
    print(f"  Flow magnitude: {np.linalg.norm(flow):.3f}")
    
    # Get flow field around center
    flow_field = vector_field.get_flow_field(center=z, radius=2.0, buffer=buffer, num_samples=10)
    print(f"\nFlow field around center:")
    print(f"  Positions: {flow_field['positions'].shape}")
    print(f"  Directions: {flow_field['directions'].shape}")
    print(f"  Mean magnitude: {np.mean(flow_field['magnitudes']):.3f}")
    
    return vector_field, buffer


def test_flow_field_layer():
    """Test full Flow Field Layer."""
    print("\n" + "=" * 60)
    print("FLOW FIELD LAYER TEST")
    print("=" * 60)
    
    flow_layer = FlowFieldLayer(latent_dim=8, action_dim=2, k_neighbors=10)
    env = SimpleEnv()
    encoder = create_encoder(obs_dim=10, latent_dim=8)
    
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
        # Encode
        z = encoder(obs)
        
        # Select action (random for testing)
        a = actions[np.random.randint(len(actions))]
        
        # Execute
        obs_next, reward, done = env.step(a)
        z_next = encoder(obs_next)
        
        # Update flow layer
        flow_layer.update(z, a, z_next)
        
        # Log every 20 steps
        if step % 20 == 0:
            field_data = flow_layer.get_field_data(z)
            print(f"  Step {step}: V={field_data['V']:.3f}, "
                  f"local_V={field_data['local_V']:.3f}, "
                  f"buffer={field_data['buffer_size']}, "
                  f"uncertainty={field_data['uncertainty']:.3f}")
        
        obs = obs_next
        if done:
            break
    
    # Get 2D projection
    projection, labels = flow_layer.get_2d_projection(method='pca')
    print(f"\n2D Projection: {projection.shape}, label range: [{labels.min():.3f}, {labels.max():.3f}]")
    
    return flow_layer


def test_field_geometry():
    """Test geometry aspects of flow field."""
    print("\n" + "=" * 60)
    print("FIELD GEOMETRY TEST")
    print("=" * 60)
    
    flow_layer = FlowFieldLayer(latent_dim=8, action_dim=2)
    
    # Generate points on a torus-like manifold
    np.random.seed(42)
    
    for i in range(200):
        # Torus parameterization
        u = np.random.rand() * 2 * np.pi
        v = np.random.rand() * 2 * np.pi
        r1, r2 = 3.0, 1.0
        
        z = np.zeros(8)
        z[0] = (r1 + r2 * np.cos(v)) * np.cos(u)
        z[1] = (r1 + r2 * np.cos(v)) * np.sin(u)
        z[2] = r2 * np.sin(v)
        
        # Add noise
        z += np.random.randn(8) * 0.1
        
        a = np.array([np.cos(u), np.sin(u)])  # tangent direction
        z_next = z.copy()
        z_next[:3] += r2 * 0.1 * np.array([-np.sin(u)*np.cos(v), np.cos(u)*np.cos(v), np.sin(v)])
        
        flow_layer.update(z, a, z_next)
    
    print(f"Buffer: {flow_layer.buffer.size()} points")
    
    # Test local structure at various points
    test_z = [
        np.array([3.0, 0.0, 0.0] + [0] * 5),
        np.array([0.0, 3.0, 0.0] + [0] * 5),
        np.array([0.0, 0.0, 1.0] + [0] * 5),
        np.array([0.0, 0.0, 0.0] + [0] * 5),
    ]
    
    print("\nLocal dynamics at different points:")
    for z in test_z:
        data = flow_layer.get_field_data(z)
        print(f"  z=[{z[0]:.1f}, {z[1]:.1f}, {z[2]:.1f}]: "
              f"V={data['V']:.3f}, density={data['local_density']:.3f}, "
              f"uncertainty={data['uncertainty']:.3f}")


def compare_all():
    """Compare all approaches."""
    print("\n" + "=" * 60)
    print("COMPARISON: All approaches")
    print("=" * 60)
    
    try:
        from blite_runner import BLiteRunner
        print("\n--- B-lite ---")
        runner = BLiteRunner(env_type='simple')
        stats = runner.run_episode(max_steps=100, verbose=False)
        print(f"V_mean={stats['V_mean']:.3f}, V_min={stats['V_min']:.3f}")
    except Exception as e:
        print(f"B-lite: {e}")
        stats = {'V_mean': 0, 'V_min': 0}
    
    try:
        from stabilized_runner import StabilizedWorldModel
        print("\n--- Stabilized WM ---")
        model = StabilizedWorldModel(latent_dim=8, action_dim=2, num_modes=3)
        env = SimpleEnv()
        encoder = create_encoder(obs_dim=10, latent_dim=8)
        actions = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0])]
        
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
        
        print(f"V_mean={np.mean(V_values):.3f}, V_min={np.min(V_values):.3f}")
    except Exception as e:
        print(f"Stabilized WM: {e}")
        stats_stab = {'V_mean': 0, 'V_min': 0}
    
    print("\n--- Flow Field Layer ---")
    flow_layer = test_flow_field_layer()
    diag = flow_layer.get_diagnostic_state()
    print(f"V={diag['V']:.3f}, buffer={diag['buffer_size']}, stability={diag['field_stability']:.3f}")
    
    print("\n--- SUMMARY ---")
    print("Approach         | V_mean | V_min | Special")
    print("-" * 50)
    print("B-lite           | --     | --    | noise injection")
    print("Stabilized WM    | --     | --    | mode lifecycle")
    print("Flow Field Layer | {:.3f}  | --    | geometry-first".format(diag['V']))


if __name__ == '__main__':
    test_latent_buffer()
    test_knn_builder()
    test_field_estimator()
    test_vector_field()
    test_flow_field_layer()
    test_field_geometry()
    compare_all()
    
    print("\n" + "=" * 60)
    print("FLOW FIELD LAYER TEST COMPLETE")
    print("=" * 60)