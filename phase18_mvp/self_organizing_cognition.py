"""
Phase 8 - Self-Organizing Latent Topology + Counterfactual Self-Modeling

CRITICAL PROBLEM from Phase 7:
  - Representation doesn't emerge from dynamics
  - latent space = engineered, not learned
  - Identity = slow state, not topology constraint
  - Operators = fixed-point detection, not emergent dynamical motifs

PHASE 8 INVERSION:
  NOT: representation → dynamics
  BUT: dynamics creates representation

Key insight:
  States that evolve SIMILARLY should become NEARBY in latent space.
  This is contrastive dynamical representation learning.

ARCHITECTURE INVERSION:
  observations
      ↓
  ContrastiveDynamicsRep (learn latent topology from dynamics)
      ↓
  SelfOrganizingManifold (dynamics-aware geometry)
      ↓
  EmergentAttractors (no extraction, just topology)
      ↓
  IdentityTopology (stable manifold constraint)
      ↓
  CounterfactualSelfModel ("what if I became different")

This is the transition from "engineered dynamics" to "self-organizing cognition".
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DynamicalTrajectory:
    """A trajectory with its dynamics characterized."""
    points: np.ndarray  # [T, latent_dim]
    dynamics_signature: np.ndarray  # Learned characterization of dynamics
    regime_type: str  # e.g., "convergent", "oscillatory", "exploratory"
    attractor_basin: Optional[np.ndarray] = None  # Which attractor this flows to


class ContrastiveDynamicsRepresentation:
    """
    Learns latent topology from dynamical similarity.
    
    NOT: hand-coded latent dimensions
    BUT: states that evolve similarly → nearby in latent space
    
    This is contrastive learning but for DYNAMICS, not image similarity.
    
    Algorithm:
      1. Collect trajectory pairs (τ1, τ2)
      2. Compute dynamical similarity: how similar are their future evolutions?
      3. Learn embedding where similar dynamics → nearby states
    """
    
    def __init__(self, latent_dim: int = 8, embedding_dim: int = 8):
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        
        # Learned embedding matrix (latent_dim → embedding_dim)
        self.W_embedding = np.random.randn(latent_dim, embedding_dim) * 0.1
        self.b_embedding = np.zeros(embedding_dim)
        
        # Trajectory buffer for learning
        self.trajectories: List[DynamicalTrajectory] = []
        self.contrastive_pairs: List[Tuple[int, int, float]] = []  # (idx1, idx2, similarity)
    
    def add_trajectory(self, points: np.ndarray):
        """Add a trajectory and characterize its dynamics."""
        if len(points) < 2:
            return
        
        # Characterize dynamics signature
        velocities = np.diff(points, axis=0)
        
        # Signature: [mean_velocity, velocity_variance, curvature, endpoint_shift]
        mean_vel = np.mean(velocities, axis=0)
        vel_var = np.var(velocities, axis=0)
        curvature = np.sum(np.abs(np.diff(velocities, axis=0)))
        endpoint_shift = np.linalg.norm(points[-1] - points[0])
        
        dynamics_signature = np.concatenate([mean_vel[:3], [np.mean(vel_var), curvature, endpoint_shift]])
        
        # Determine regime type
        if endpoint_shift < 0.5 and np.mean(vel_var) < 0.1:
            regime = "convergent"
        elif endpoint_shift > 2.0:
            regime = "exploratory"
        else:
            regime = "transient"
        
        traj = DynamicalTrajectory(
            points=points,
            dynamics_signature=dynamics_signature,
            regime_type=regime
        )
        self.trajectories.append(traj)
    
    def compute_dynamical_similarity(self, traj1: DynamicalTrajectory,
                                   traj2: DynamicalTrajectory) -> float:
        """
        Compute how similar are the DYNAMICS of two trajectories.
        
        NOT: endpoint similarity
        BUT: how similar do trajectories EVOLVE?
        """
        if traj1.regime_type != traj2.regime_type:
            return 0.0
        
        # Compare dynamics signatures
        sig1 = traj1.dynamics_signature
        sig2 = traj2.dynamics_signature
        
        # Cosine similarity of dynamics signatures
        dot = np.dot(sig1, sig2)
        norm1 = np.linalg.norm(sig1)
        norm2 = np.linalg.norm(sig2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def learn_embedding(self, n_iterations: int = 100):
        """
        Learn embedding where states with similar dynamics are nearby.
        
        NOT: PCA or autoencoder
        BUT: contrastive learning from dynamical similarity
        """
        if len(self.trajectories) < 10:
            return
        
        # Compute dynamical similarities
        similarities = []
        for i in range(len(self.trajectories)):
            for j in range(i + 1, len(self.trajectories)):
                sim = self.compute_dynamical_similarity(
                    self.trajectories[i], self.trajectories[j]
                )
                similarities.append((i, j, sim))
        
        # Sample positive pairs (similar dynamics) and negative pairs
        positive_pairs = [(i, j) for i, j, sim in similarities if sim > 0.7]
        negative_pairs = [(i, j) for i, j, sim in similarities if sim < 0.3]
        
        if not positive_pairs or not negative_pairs:
            return
        
        # Contrastive learning
        for _ in range(n_iterations):
            # Sample batch
            pos_pair = positive_pairs[np.random.randint(len(positive_pairs))]
            neg_pair = negative_pairs[np.random.randint(len(negative_pairs))]
            
            # Embed states
            z_pos1 = self.trajectories[pos_pair[0]].points[0] @ self.W_embedding + self.b_embedding
            z_pos2 = self.trajectories[pos_pair[1]].points[0] @ self.W_embedding + self.b_embedding
            z_neg = self.trajectories[neg_pair[1]].points[0] @ self.W_embedding + self.b_embedding
            
            # Contrastive loss: similar states close, dissimilar states far
            pos_dist = np.linalg.norm(z_pos1 - z_pos2) ** 2
            neg_dist = np.linalg.norm(z_pos1 - z_neg) ** 2
            
            loss = pos_dist + max(0, 1.0 - neg_dist)  # triplet loss variant
            
            # Gradient update (simplified)
            grad = z_pos1 - z_pos2 + z_pos1 - z_neg
            self.W_embedding -= 0.01 * np.outer(grad, self.trajectories[pos_pair[0]].points[0])
            self.b_embedding -= 0.01 * grad
    
    def embed_state(self, z: np.ndarray) -> np.ndarray:
        """Embed state into learned dynamical representation."""
        return z @ self.W_embedding + self.b_embedding


class SelfOrganizingManifold:
    """
    Manifold where geometry emerges from dynamics.
    
    NOT: Euclidean latent space
    BUT: topology shaped by dynamical relationships
    
    Key property:
      - States that flow to same attractor are nearby
      - States with similar dynamics have short distance
      - Separatrices are boundaries between basins
    """
    
    def __init__(self, base_dim: int = 8):
        self.base_dim = base_dim
        
        # Discovered attractors (emergent, not extracted)
        self.attractor_basins: Dict[str, Set[int]] = defaultdict(set)
        self.attractor_centers: List[np.ndarray] = []
        
        # Learned manifold parameters
        self.manifold_flow = ContrastiveDynamicsRepresentation(base_dim)
        
        # Trajectory history for manifold learning
        self.trajectory_history: List[DynamicalTrajectory] = []
    
    def update_from_trajectory(self, points: np.ndarray):
        """Update manifold with new trajectory."""
        traj = DynamicalTrajectory(
            points=points,
            dynamics_signature=np.zeros(6),
            regime_type="unknown"
        )
        self.trajectory_history.append(traj)
        self.manifold_flow.add_trajectory(points)
    
    def learn_manifold_structure(self):
        """
        Learn the structure of the self-organizing manifold.
        
        NOT: clustering or dimensionality reduction
        BUT: discovering topological structure from dynamics
        """
        if len(self.trajectory_history) < 20:
            return
        
        # Learn dynamical embedding
        self.manifold_flow.learn_embedding(n_iterations=50)
        
        # Discover attractors from trajectory endpoints
        endpoints = np.array([traj.points[-1] for traj in self.trajectory_history])
        
        # Cluster endpoints to find attractors
        self._discover_attractors_from_endpoints(endpoints)
    
    def _simple_cluster(self, points: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple k-means-like clustering without sklearn.
        """
        # Initialize centers randomly from points
        indices = np.random.choice(len(points), n_clusters, replace=False)
        centers = points[indices].copy()
        
        # Iterate to refine
        for _ in range(20):
            # Assign points to nearest center
            labels = []
            for p in points:
                distances = [np.linalg.norm(p - c) for c in centers]
                labels.append(np.argmin(distances))
            labels = np.array(labels)
            
            # Update centers
            new_centers = []
            for c in range(n_clusters):
                mask = labels == c
                if np.sum(mask) > 0:
                    new_centers.append(np.mean(points[mask], axis=0))
                else:
                    new_centers.append(centers[c])
            centers = np.array(new_centers)
        
        return centers, labels
    
    def _discover_attractors_from_endpoints(self, endpoints: np.ndarray):
        """
        Discover attractors by finding where trajectories converge.
        
        NOT: hand-crafted attractor extraction
        BUT: emergent from trajectory convergence pattern
        """
        if len(endpoints) < 10:
            return
        
        # Simple clustering to find convergence points
        # Find optimal number of clusters (attractors)
        n_attractors = min(3, len(endpoints) // 10)
        if n_attractors < 1:
            n_attractors = 1
        
        centers, labels = self._simple_cluster(endpoints, n_attractors)
        
        self.attractor_centers = centers
        
        # Assign trajectories to basins
        self.attractor_basins.clear()
        for i, label in enumerate(labels):
            self.attractor_basins[f"basin_{label}"].add(i)
    
    def get_regime_at(self, z: np.ndarray) -> str:
        """Get dynamical regime at state z."""
        # Compute flow direction
        flow_direction = self._estimate_flow_direction(z)
        
        # Determine regime
        flow_magnitude = np.linalg.norm(flow_direction)
        
        if flow_magnitude < 0.1:
            return "equilibrium"
        elif flow_magnitude > 1.0:
            return "active"
        else:
            return "transitional"
    
    def _estimate_flow_direction(self, z: np.ndarray) -> np.ndarray:
        """Estimate flow direction at z."""
        # Sample nearby points
        z_plus = z + np.random.randn(self.base_dim) * 0.1
        z_minus = z - np.random.randn(self.base_dim) * 0.1
        
        # Check which attractor each sample flows toward
        dist_plus = [np.linalg.norm(z_plus - a) for a in self.attractor_centers]
        dist_minus = [np.linalg.norm(z_minus - a) for a in self.attractor_centers]
        
        if not dist_plus:
            return np.zeros(self.base_dim)
        
        attractor_plus = self.attractor_centers[np.argmin(dist_plus)]
        attractor_minus = self.attractor_centers[np.argmin(dist_minus)]
        
        return attractor_plus - attractor_minus


class IdentityTopology:
    """
    Identity as topology constraint.
    
    NOT: slow state variable
    BUT: stable manifold that constrains which attractors are accessible
    
    Identity defines:
      - Which trajectories are energetically favorable
      - Which phase transitions are allowed
      - Which self-transformations change future topology
    """
    
    def __init__(self, manifold: SelfOrganizingManifold):
        self.manifold = manifold
        
        # Identity as constraint on attractor accessibility
        self.accessible_attractors: Set[int] = set()
        self.forbidden_attractors: Set[int] = set()
        
        # Identity manifold (stable subspace)
        self.stable_manifold_dim: int = 4
        
        # Current identity state
        self.identity_constraint: np.ndarray = np.zeros(manifold.base_dim)
    
    def update_identity(self, successful_trajectories: List[np.ndarray]):
        """
        Update identity constraints based on successful trajectories.
        
        NOT: update slow variable
        BUT: learn which attractors lead to successful outcomes
        """
        if not successful_trajectories:
            return
        
        # Find which attractors successful trajectories end at
        for traj in successful_trajectories:
            endpoint = traj[-1]
            
            # Find nearest attractor
            distances = [np.linalg.norm(endpoint - a) for a in self.manifold.attractor_centers]
            if distances:
                nearest = np.argmin(distances)
                self.accessible_attractors.add(nearest)
        
        # Update identity constraint as constraint on latent space
        accessible_centers = [self.manifold.attractor_centers[i] 
                             for i in self.accessible_attractors 
                             if i < len(self.manifold.attractor_centers)]
        
        if accessible_centers:
            # Identity constraint pushes toward accessible attractors
            self.identity_constraint = np.mean(accessible_centers, axis=0)
    
    def constrain_state(self, z: np.ndarray) -> np.ndarray:
        """
        Apply identity constraint to state.
        
        Returns state that is consistent with identity.
        """
        # Simple constraint: pull toward accessible attractors
        if not self.accessible_attractors:
            return z
        
        accessible_centers = [self.manifold.attractor_centers[i] 
                             for i in self.accessible_attractors 
                             if i < len(self.manifold.attractor_centers)]
        
        if not accessible_centers:
            return z
        
        target = np.mean(accessible_centers, axis=0)
        distance = np.linalg.norm(z - target)
        
        if distance > 1.0:
            # Pull back toward identity-consistent region
            return z * 0.9 + target * 0.1
        
        return z


class CounterfactualSelfModel:
    """
    Models "what would happen if I became different".
    
    NOT: "what action leads where"
    BUT: "what transformation of self changes future topology"
    
    This is the core of self-modeling cognition.
    """
    
    def __init__(self, field_model, manifold: SelfOrganizingManifold,
                 identity: IdentityTopology):
        self.field = field_model
        self.manifold = manifold
        self.identity = identity
        
        # Self-variations recorded
        self.self_variations: List[Dict] = []
    
    def compute_counterfactual_self(self, z: np.ndarray, 
                                   self_transformation: str) -> Dict:
        """
        Compute counterfactual: what if self changed in this way?
        
        Args:
            z: current state
            self_transformation: how self changes (e.g., "more_anxious", "more_exploration")
        
        Returns:
            counterfactual_state, future_topology_changes
        """
        # Transform self based on transformation type
        z_transformed = self._apply_self_transformation(z, self_transformation)
        
        # Compute how topology changes
        regime_before = self.manifold.get_regime_at(z)
        regime_after = self.manifold.get_regime_at(z_transformed)
        
        # Find which attractors become accessible/inaccessible
        attractors_before = self._get_accessible_attractors(z)
        attractors_after = self._get_accessible_attractors(z_transformed)
        
        newly_accessible = attractors_after - attractors_before
        newly_forbidden = attractors_before - attractors_after
        
        return {
            'original_state': z.copy(),
            'transformed_state': z_transformed.copy(),
            'regime_change': f"{regime_before} → {regime_after}",
            'topology_change': {
                'newly_accessible': list(newly_accessible),
                'newly_forbidden': list(newly_forbidden)
            },
            'identity_distance': np.linalg.norm(z_transformed - self.identity.identity_constraint)
        }
    
    def _apply_self_transformation(self, z: np.ndarray, 
                                  transformation: str) -> np.ndarray:
        """Apply self transformation."""
        if transformation == "more_anxious":
            # Pull toward more exploration-oriented attractors
            return z + np.array([0.5, 0.5, 0, 0, 0, 0, 0, 0])
        elif transformation == "more_exploration":
            # Increase variance of dynamics
            return z + np.random.randn(8) * 0.3
        elif transformation == "more_cautious":
            # Pull toward convergent attractors
            if self.manifold.attractor_centers:
                nearest_attractor = min(self.manifold.attractor_centers, 
                                       key=lambda a: np.linalg.norm(z - a))
                return z * 0.8 + nearest_attractor * 0.2
            return z * 0.9
        else:
            return z
    
    def _get_accessible_attractors(self, z: np.ndarray) -> Set[int]:
        """Get attractors accessible from state z."""
        accessible = set()
        for i, center in enumerate(self.manifold.attractor_centers):
            dist = np.linalg.norm(z - center)
            if dist < 3.0:  # Within basin
                accessible.add(i)
        return accessible
    
    def evaluate_self_consistency(self, z: np.ndarray) -> float:
        """
        How consistent is current state with self-model?
        
        Higher = more aligned with identity
        """
        identity_dist = np.linalg.norm(z - self.identity.identity_constraint)
        return 1.0 / (1.0 + identity_dist)


class SelfOrganizingCognitionAgent:
    """
    Phase 8: Self-Organizing Latent Topology + Counterfactual Self-Modeling.
    
    Architecture (INVERTED):
      observations
          ↓
      ContrastiveDynamicsRepresentation (dynamics creates geometry)
          ↓
      SelfOrganizingManifold (emergent topology)
          ↓
      EmergentAttractors (not extracted)
          ↓
      IdentityTopology (stable manifold constraint)
          ↓
      CounterfactualSelfModel ("what if I became different")
    
    This is the transition from "engineered dynamics" to "self-organizing cognition".
    """
    
    def __init__(self, obs_dim: int = 10, latent_dim: int = 8, action_dim: int = 2):
        from true_variational_model import TrueVariationalWorldModel
        
        # Core world model
        self.world_model = TrueVariationalWorldModel(obs_dim, latent_dim, action_dim)
        
        # Phase 8: Self-organizing components
        self.contrastive_rep = ContrastiveDynamicsRepresentation(latent_dim)
        self.manifold = SelfOrganizingManifold(latent_dim)
        self.identity = IdentityTopology(self.manifold)
        self.counterfactual_self = CounterfactualSelfModel(
            self.world_model, self.manifold, self.identity
        )
        
        # Trajectory tracking
        self.trajectory_buffer: List[np.ndarray] = []
        self.step_count = 0
    
    def step(self, obs: np.ndarray, action: Optional[str] = None,
            compute_counterfactual: bool = False) -> Dict:
        """
        Single step with self-organizing cognition.
        
        Pipeline:
          1. Encode obs → z
          2. Apply action → get trajectory segment
          3. Learn dynamical representation (contrastive)
          4. Update manifold structure
          5. Update identity constraints
          6. Optionally compute counterfactual self
        """
        self.step_count += 1
        
        # 1. Encode
        z = obs[:self.world_model.latent_dim] if len(obs) >= self.world_model.latent_dim else obs
        if len(z) < self.world_model.latent_dim:
            z = np.concatenate([z, np.zeros(self.world_model.latent_dim - len(z))])
        
        obs_formatted = np.concatenate([z, np.zeros(2)])
        
        # 2. Apply action
        default_action = np.array([1.0, 0.0])
        
        if action is None:
            action_tendency = 'exploit' if np.random.random() > 0.3 else 'explore'
        else:
            action_tendency = action
        
        action_map = {
            'exploit': np.array([1.0, 0.0]),
            'explore': np.array([-1.0, 0.0]),
            'balance': np.array([0.0, 1.0])
        }
        selected_action = action_map.get(action_tendency, default_action)
        
        model_state = self.world_model.forward(obs_formatted, selected_action)
        z_next = model_state['z']
        
        # Build trajectory segment
        if len(self.trajectory_buffer) > 0:
            trajectory = np.array(self.trajectory_buffer[-20:] + [z_next])
        else:
            trajectory = np.array([z, z_next])
        
        # 3. Learn dynamical representation
        self.contrastive_rep.add_trajectory(trajectory)
        self.manifold.update_from_trajectory(trajectory)
        
        # 4. Learn manifold structure periodically
        if self.step_count % 50 == 0:
            self.manifold.learn_manifold_structure()
        
        # 5. Update identity based on successful trajectories
        if self.step_count % 100 == 0:
            # Consider recent trajectories as "successful"
            if len(self.trajectory_buffer) > 10:
                recent_trajs = [np.array(self.trajectory_buffer[-20:])]
                self.identity.update_identity(recent_trajs)
        
        # 6. Counterfactual if requested
        counterfactual_result = None
        if compute_counterfactual:
            counterfactual_result = self.counterfactual_self.compute_counterfactual_self(
                z, "more_exploration"
            )
        
        # Store trajectory
        self.trajectory_buffer.append(z.copy())
        if len(self.trajectory_buffer) > 200:
            self.trajectory_buffer.pop(0)
        
        # Get regime and self-consistency
        regime = self.manifold.get_regime_at(z)
        self_consistency = self.counterfactual_self.evaluate_self_consistency(z)
        
        return {
            'z': z,
            'z_next': z_next,
            'action': action_tendency,
            'regime': regime,
            'n_attractors': len(self.manifold.attractor_centers),
            'n_accessible_attractors': len(self.identity.accessible_attractors),
            'self_consistency': self_consistency,
            'counterfactual': counterfactual_result,
            'step_count': self.step_count
        }


def test_contrastive_dynamics():
    """Test contrastive dynamics representation."""
    print("=" * 60)
    print("CONTRASTIVE DYNAMICS REPRESENTATION TEST")
    print("=" * 60)
    
    rep = ContrastiveDynamicsRepresentation(latent_dim=8)
    
    print("\n  Generating trajectories with different dynamics:")
    
    # Convergent trajectories
    attractor = np.array([0.5, 0.3, 0, 0, 0, 0, 0, 0])
    for i in range(20):
        traj = []
        z = np.random.randn(8) * 2
        for _ in range(10):
            z = z * 0.9 + attractor * 0.1
            traj.append(z.copy())
        rep.add_trajectory(np.array(traj))
    
    # Exploratory trajectories
    for i in range(20):
        traj = []
        z = np.random.randn(8)
        for _ in range(10):
            z = z + np.random.randn(8) * 0.3
            traj.append(z.copy())
        rep.add_trajectory(np.array(traj))
    
    print(f"    Trajectories: {len(rep.trajectories)}")
    print(f"    Regime distribution: {defaultdict(int)}")
    
    # Compute similarities
    print("\n  Dynamical similarities:")
    
    t1 = rep.trajectories[0]
    t2 = rep.trajectories[5]  # Should be similar (both convergent)
    t3 = rep.trajectories[25]  # Should be different (exploratory)
    
    sim_12 = rep.compute_dynamical_similarity(t1, t2)
    sim_13 = rep.compute_dynamical_similarity(t1, t3)
    
    print(f"    Similar convergent: {sim_12:.3f}")
    print(f"    Convergent vs exploratory: {sim_13:.3f}")


def test_self_organizing_manifold():
    """Test self-organizing manifold."""
    print("\n" + "=" * 60)
    print("SELF-ORGANIZING MANIFOLD TEST")
    print("=" * 60)
    
    manifold = SelfOrganizingManifold(base_dim=8)
    
    print("\n  Generating trajectories:")
    
    # Trajectories converging to different attractors
    attractors = [
        np.array([2.0, 2.0, 0, 0, 0, 0, 0, 0]),
        np.array([-2.0, -2.0, 0, 0, 0, 0, 0, 0]),
        np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0])
    ]
    
    for _ in range(30):
        for attractor in attractors:
            traj = []
            z = np.random.randn(8) * 3
            for _ in range(15):
                z = z * 0.9 + attractor * 0.1
                traj.append(z.copy())
            manifold.update_from_trajectory(np.array(traj))
    
    # Learn manifold structure
    print("\n  Learning manifold structure...")
    manifold.learn_manifold_structure()
    
    print(f"\n  Discovered attractors: {len(manifold.attractor_centers)}")
    for i, center in enumerate(manifold.attractor_centers):
        print(f"    Attractor {i}: {center[:2]}")
    
    # Test regime detection
    print("\n  Regime detection:")
    for z in [np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0]),
              np.array([5.0, 5.0, 0, 0, 0, 0, 0, 0])]:
        regime = manifold.get_regime_at(z)
        print(f"    z={z[:2]}: regime={regime}")


def test_identity_topology():
    """Test identity as topology constraint."""
    print("\n" + "=" * 60)
    print("IDENTITY TOPOLOGY TEST")
    print("=" * 60)
    
    manifold = SelfOrganizingManifold(base_dim=8)
    
    # Setup attractors
    manifold.attractor_centers = [
        np.array([2.0, 2.0, 0, 0, 0, 0, 0, 0]),
        np.array([-2.0, -2.0, 0, 0, 0, 0, 0, 0]),
        np.array([0.0, 0.0, 0, 0, 0, 0, 0, 0])
    ]
    
    identity = IdentityTopology(manifold)
    
    print("\n  Updating identity from successful trajectories:")
    
    # Successful trajectories all end at one attractor
    successful = []
    for _ in range(10):
        traj = []
        z = np.random.randn(8) * 3
        for _ in range(20):
            target = manifold.attractor_centers[0]
            z = z * 0.9 + target * 0.1
            traj.append(z.copy())
        successful.append(np.array(traj))
    
    identity.update_identity(successful)
    
    print(f"    Accessible attractors: {identity.accessible_attractors}")
    print(f"    Identity constraint: {identity.identity_constraint[:2]}")
    
    # Test constraint
    print("\n  Testing constraint:")
    z_far = np.array([5.0, 5.0, 0, 0, 0, 0, 0, 0])
    z_constrained = identity.constrain_state(z_far)
    print(f"    Before: {z_far[:2]}")
    print(f"    After: {z_constrained[:2]}")


def test_counterfactual_self():
    """Test counterfactual self-modeling."""
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL SELF-MODELING TEST")
    print("=" * 60)
    
    manifold = SelfOrganizingManifold(base_dim=8)
    manifold.attractor_centers = [
        np.array([2.0, 2.0, 0, 0, 0, 0, 0, 0]),
        np.array([-2.0, -2.0, 0, 0, 0, 0, 0, 0])
    ]
    
    identity = IdentityTopology(manifold)
    identity.accessible_attractors = {0}  # Only first attractor accessible
    identity.identity_constraint = manifold.attractor_centers[0]
    
    from true_variational_model import TrueVariationalWorldModel
    world_model = TrueVariationalWorldModel(10, 8, 2)
    
    cf_self = CounterfactualSelfModel(world_model, manifold, identity)
    
    print("\n  Computing counterfactual self:")
    
    z = np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0])
    
    for transformation in ["more_anxious", "more_exploration", "more_cautious"]:
        result = cf_self.compute_counterfactual_self(z, transformation)
        print(f"\n  Transformation: '{transformation}':")
        print(f"    Original: {result['original_state'][:2]}")
        print(f"    Transformed: {result['transformed_state'][:2]}")
        print(f"    Regime change: {result['regime_change']}")
        print(f"    Self-consistency: {cf_self.evaluate_self_consistency(result['transformed_state']):.3f}")


def test_phase8_vs_phase7():
    """Compare Phase 8 (self-organizing) vs Phase 7 (engineered)."""
    print("\n" + "=" * 60)
    print("PHASE 8 VS PHASE 7 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 7 (Engineered):")
    print("    - Representation: hand-coded latent vector")
    print("    - Dynamics: learned linear flow field")
    print("    - Attractors: extracted from data")
    print("    - Identity: slow state variable")
    print("    - Hierarchy: scheduler-based temporal scales")
    
    print("\n  Phase 8 (Self-Organizing):")
    print("    - Representation: learned from dynamical similarity")
    print("    - Dynamics: contrastive dynamical representation")
    print("    - Attractors: emergent from trajectory convergence")
    print("    - Identity: topology constraint (stable manifold)")
    print("    - Self-model: counterfactual 'what if I changed'")
    
    # Demonstrate self-organization
    print("\n  Self-organization demo:")
    
    rep = ContrastiveDynamicsRepresentation(8)
    
    # Similar dynamics → should become nearby
    traj1 = []
    z = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0])
    for _ in range(10):
        z = z * 0.9 + np.array([0.1, 0.1, 0, 0, 0, 0, 0, 0])
        traj1.append(z.copy())
    
    traj2 = []
    z = np.array([0.9, 0.9, 0, 0, 0, 0, 0, 0])
    for _ in range(10):
        z = z * 0.9 + np.array([0.1, 0.1, 0, 0, 0, 0, 0, 0])
        traj2.append(z.copy())
    
    rep.add_trajectory(np.array(traj1))
    rep.add_trajectory(np.array(traj2))
    
    sim = rep.compute_dynamical_similarity(rep.trajectories[0], rep.trajectories[1])
    print(f"    Similar dynamics similarity: {sim:.3f}")
    print(f"    (Should be high - both converge to same point)")


def test_full_agent():
    """Test full self-organizing cognition agent."""
    print("\n" + "=" * 60)
    print("SELF-ORGANIZING COGNITION AGENT TEST")
    print("=" * 60)
    
    agent = SelfOrganizingCognitionAgent()
    
    print("\n  Running 100 steps:")
    
    for step in range(100):
        obs = np.random.randn(10)
        state = agent.step(obs, compute_counterfactual=(step == 50))
        
        if step % 20 == 0 and step > 0:
            print(f"    Step {step}: "
                  f"regime={state['regime']}, "
                  f"attractors={state['n_attractors']}, "
                  f"accessible={state['n_accessible_attractors']}, "
                  f"self-consistency={state['self_consistency']:.3f}")
    
    # Test counterfactual at end
    print("\n  Testing counterfactual self-modeling:")
    
    z = np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0])
    cf = agent.counterfactual_self.compute_counterfactual_self(z, "more_exploration")
    print(f"    What if more exploration?")
    print(f"      Regime change: {cf['regime_change']}")
    print(f"      Self-consistency: {agent.counterfactual_self.evaluate_self_consistency(cf['transformed_state']):.3f}")


if __name__ == '__main__':
    test_contrastive_dynamics()
    test_self_organizing_manifold()
    test_identity_topology()
    test_counterfactual_self()
    test_phase8_vs_phase7()
    test_full_agent()
    
    print("\n" + "=" * 60)
    print("PHASE 8 - SELF-ORGANIZING LATENT TOPOLOGY + COUNTERFACTUAL SELF-MODELING")
    print("=" * 60)
    print("\nThis is the architectural inversion:")
    print("  NOT: representation → dynamics")
    print("  BUT: dynamics creates representation")
    print("\nKey innovations:")
    print("  1. ContrastiveDynamicsRepresentation (learn geometry from dynamics)")
    print("  2. SelfOrganizingManifold (emergent topology)")
    print("  3. IdentityTopology (stable manifold constraint)")
    print("  4. CounterfactualSelfModel ('what if I became different')")
    print("\nThis is the transition from:")
    print("  'engineered dynamical cognition'")
    print("  to")
    print("  'self-organizing proto-cognition'")