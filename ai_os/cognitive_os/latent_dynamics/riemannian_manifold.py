"""
Phase 14.5 - Riemannian Cognitive Manifold

NOT three force generators:
  F_energy, F_policy, F_inertia

BUT single Riemannian manifold with:

1. Metric tensor g_ij(z) - learned inertia/geometry
2. Single energy functional V(z) = V_data(z) + V_policy(z)
3. Lagrangian L(z, ż) = T(z, ż) - V(z) = 0.5 * g_ij * ż^i * ż^j - V(z)
4. Euler-Lagrange with metric: covariant dynamics

Key principle:
  Everything = geometry of single learned manifold
  
  Policy = potential (not force)
  Inertia = metric (not separate)
  Energy = V(z) (not separate force field)
  
  Trajectories = geodesics on manifold
  Goals = minima of V(z)
  Stability = curvature of metric
  Uncertainty = volume form from g
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class RiemannianMetric:
    """
    Learned metric tensor g_ij(z) on latent manifold.
    
    NOT inertia as force scaling.
    
    BUT geometry of the manifold itself.
    
    g_ij defines:
    - Kinetic energy: T = 0.5 * g_ij * ż^i * ż^j
    - Distance: ds² = g_ij * dz^i * dz^j
    - Volume form: sqrt(det(g))
    - Geodesics: paths of minimum action
    - Curvature: R(g) - intrinsic geometry
    
    This replaces InertiaTensor as fundamental geometry.
    """
    
    dimension: int = 16
    
    # Metric tensor (symmetric, positive definite)
    # g_ij[ i ][ j ] where i, j are dimension indices
    g: List[List[float]] = field(default_factory=list)
    
    # Inverse metric g^ij (for Christoffel symbols)
    g_inv: List[List[float]] = field(default_factory=list)
    
    # Christoffel symbols Γ^i_jk (for covariant derivative)
    gamma: List[List[List[float]]] = field(default_factory=list)
    
    # Learned from trajectory statistics
    trajectory_samples: List[Tuple[List[float], List[float]]] = []  # (z, ż)
    
    # Geometry properties
    det_g: float = 1.0  # Determinant for volume form
    curvature_scalar: float = 0.0  # Ricci scalar
    
    def __post_init__(self):
        if not self.g:
            self._initialize_identity()
    
    def _initialize_identity(self) -> None:
        """Initialize with Euclidean metric"""
        self.g = [[1.0 if i == j else 0.0 for j in range(self.dimension)] 
                  for i in range(self.dimension)]
        self._compute_inverse()
        self._compute_christoffel()
    
    def _compute_inverse(self) -> None:
        """Compute inverse metric g^ij"""
        # Simplified: diagonal approximation
        self.g_inv = [[1.0 / (self.g[i][i] + 1e-6) if i == j else 0.0 
                       for j in range(self.dimension)] 
                      for i in range(self.dimension)]
    
    def _compute_christoffel(self) -> None:
        """Compute Christoffel symbols Γ^i_jk from metric"""
        # Simplified Christoffel for diagonal-ish metric
        self.gamma = [
            [[0.0] * self.dimension for _ in range(self.dimension)]
            for _ in range(self.dimension)
        ]
        
        # Γ^i_jk = 0.5 * g^iμ * (∂_j g_μk + ∂_k g_μj - ∂_μ g_jk)
        # Simplified: only diagonal components matter for learning
    
    def observe_trajectory(self, z: List[float], ż: List[float]) -> None:
        """Observe state and velocity to update metric"""
        self.trajectory_samples.append((z, ż))
        
        # Learn metric from trajectory variance
        self._update_metric_from_samples()
    
    def _update_metric_from_samples(self) -> None:
        """Update metric tensor from trajectory statistics"""
        if len(self.trajectory_samples) < 10:
            return
        
        # Compute local covariance of velocities
        recent = self.trajectory_samples[-100:]
        
        # Estimate variance per dimension
        for i in range(self.dimension):
            velocities = [ż[i] for _, ż in recent]
            if len(velocities) >= 2:
                mean = sum(velocities) / len(velocities)
                variance = sum((v - mean) ** 2 for v in velocities) / len(velocities)
                
                # Metric: g_ii inversely proportional to variance
                # Low variance = high "mass" in that direction
                self.g[i][i] = 1.0 / (1.0 + variance)
        
        # Normalize to prevent explosion
        trace = sum(self.g[i][i] for i in range(self.dimension))
        if trace > 0:
            scale = self.dimension / trace
            for i in range(self.dimension):
                self.g[i][i] *= scale
        
        self._compute_inverse()
    
    def kinetic_energy(self, ż: List[float]) -> float:
        """
        Compute kinetic energy: T = 0.5 * g_ij * ż^i * ż^j
        """
        T = 0.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                T += 0.5 * self.g[i][j] * ż[i] * ż[j]
        return T
    
    def geodesic_distance(self, z1: List[float], z2: List[float]) -> float:
        """
        Compute geodesic distance between two points.
        
        Simplified: use metric-weighted Euclidean distance.
        """
        dist = 0.0
        for i in range(self.dimension):
            delta = z2[i] - z1[i]
            dist += self.g[i][i] * delta * delta
        return math.sqrt(dist)
    
    def compute_christoffel_at(self, z: List[float]) -> List[List[List[float]]]:
        """
        Compute Christoffel symbols at position z.
        
        This is needed for geodesic equation.
        """
        gamma = [
            [[0.0] * self.dimension for _ in range(self.dimension)]
            for _ in range(self.dimension)
        ]
        
        # Simplified: use stored gamma with position-dependent correction
        epsilon = 0.1
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    # ∂_j g_ik approximation
                    g_ijk_plus = self._metric_at(z, epsilon, axis=j)
                    g_ijk_minus = self._metric_at(z, -epsilon, axis=j)
                    
                    # Γ^i_jk ≈ 0.5 * g^ii * (∂_j g_ik)
                    gamma[i][j][k] = 0.5 * self.g_inv[i][i] * (
                        (g_ijk_plus[i][k] - g_ijk_minus[i][k]) / (2 * epsilon)
                    )
        
        return gamma
    
    def _metric_at(self, z: List[float], delta: float, axis: int) -> List[List[float]]:
        """Get metric perturbed in direction"""
        # Simplified: return current metric
        return self.g
    
    def volume_element(self, z: List[float]) -> float:
        """
        Compute volume element sqrt(det(g)) at position.
        
        This represents uncertainty/available space.
        """
        # Simplified: product of diagonal elements
        vol = 1.0
        for i in range(self.dimension):
            vol *= max(0.01, self.g[i][i])
        return math.sqrt(vol)
    
    def to_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "trace": sum(self.g[i][i] for i in range(self.dimension)),
            "det_g": self.det_g,
            "curvature_scalar": self.curvature_scalar,
            "samples": len(self.trajectory_samples),
        }


@dataclass
class SingleEnergyFunctional:
    """
    Single energy functional V(z) = V_data(z) + V_policy(z)
    
    NOT two separate fields.
    
    BUT one unified potential that contains:
    - Trajectory statistics (attractor topology)
    - Goal preferences (shaping)
    - Uncertainty (prediction error)
    
    Everything is geometry of this single surface.
    """
    
    dimension: int = 16
    
    # Base energy (from trajectory statistics)
    base_energy: float = 1.0
    
    # Learned attractors (data-derived)
    attractors: List[List[float]] = []
    attractor_depths: List[float] = []
    
    # Goal potential (learned from outcomes)
    goal_center: Optional[List[float]] = None
    goal_strength: float = 0.5
    
    # Trajectory samples for learning
    trajectory_samples: List[List[float]] = []
    outcome_samples: List[Tuple[List[float], str]] = []
    
    # Preference function (not force!)
    preference_center: Optional[List[float]] = None
    preference_width: float = 1.0
    
    def __post_init__(self):
        self.base_energy = 1.0
        self.attractors = []
        self.attractor_depths = []
    
    def observe_state(self, z: List[float], outcome: Optional[str] = None) -> None:
        """Observe state for learning"""
        self.trajectory_samples.append(z)
        
        if outcome:
            self.outcome_samples.append((z, outcome))
        
        self._update_attractors()
    
    def _update_attractors(self) -> None:
        """Learn attractors from trajectory density"""
        if len(self.trajectory_samples) < 10:
            return
        
        recent = self.trajectory_samples[-200:]
        
        # Find high-density regions
        density: Dict[Tuple[int, ...], float] = {}
        
        for p in recent:
            key = tuple(int(v * 5) for v in p[:4])  # Quantize
            density[key] = density.get(key, 0.0) + 1.0
        
        # Top regions as attractors
        sorted_density = sorted(density.items(), key=lambda x: x[1], reverse=True)
        
        self.attractors = []
        self.attractor_depths = []
        
        for key, count in sorted_density[:5]:
            if count >= 3:
                center = [k / 5 for k in key]
                self.attractors.append(center)
                self.attractor_depths.append(count / 100.0)  # Normalized depth
    
    def set_goal(self, goal: List[float], strength: float = 0.5) -> None:
        """Set goal as potential minimum"""
        self.goal_center = goal
        self.goal_strength = strength
        
        # Add as attractor
        if goal not in self.attractors:
            self.attractors.append(goal)
            self.attractor_depths.append(strength)
    
    def set_preference(self, center: List[float], width: float = 1.0) -> None:
        """Set preference function (NOT force!)"""
        self.preference_center = center
        self.preference_width = width
    
    def compute_V(self, z: List[float]) -> float:
        """
        Compute total potential energy V(z)
        
        V(z) = base + V_attractors(z) + V_goal(z) + V_preference(z)
        
        All terms are additive potentials, not forces.
        """
        V = self.base_energy
        
        # V_data: attraction to learned attractors
        for i, attractor in enumerate(self.attractors):
            dist = self._distance(z, attractor)
            depth = self.attractor_depths[i] if i < len(self.attractor_depths) else 0.5
            V -= depth * math.exp(-dist ** 2 / 0.5)
        
        # V_policy: goal attraction (preference, not force)
        if self.goal_center:
            dist = self._distance(z, self.goal_center)
            V -= self.goal_strength * math.exp(-dist ** 2 / 1.0)
        
        # V_preference: soft preference (not injected force)
        if self.preference_center:
            dist = self._distance(z, self.preference_center)
            V -= 0.2 * math.exp(-dist ** 2 / (self.preference_width ** 2))
        
        # V_outcome: learned from successful trajectories
        for state, outcome in self.outcome_samples[-50:]:
            dist = self._distance(z, state)
            if dist < 0.3 and outcome == "success":
                V -= 0.1
        
        return max(0.0, V)
    
    def compute_gradient(self, z: List[float], epsilon: float = 0.01) -> List[float]:
        """
        Compute gradient of potential: ∇V
        
        Used for geodesic equation.
        """
        gradient = []
        
        for i in range(self.dimension):
            z_plus = z[:]
            z_plus[i] += epsilon
            V_plus = self.compute_V(z_plus)
            
            z_minus = z[:]
            z_minus[i] -= epsilon
            V_minus = self.compute_V(z_minus)
            
            gradient.append((V_plus - V_minus) / (2 * epsilon))
        
        return gradient
    
    def _distance(self, z1: List[float], z2: List[float]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(z1, z2)))
    
    def find_minimum(self, start: List[float], steps: int = 50) -> List[float]:
        """Find minimum energy path from start"""
        z = start[:]
        
        for _ in range(steps):
            grad = self.compute_gradient(z)
            z = [z[i] - 0.1 * grad[i] for i in range(self.dimension)]
        
        return z
    
    def to_dict(self) -> Dict:
        return {
            "base_energy": round(self.base_energy, 4),
            "attractors_count": len(self.attractors),
            "has_goal": self.goal_center is not None,
            "goal_strength": round(self.goal_strength, 3),
        }


@dataclass  
class GeodesicState:
    """State on Riemannian manifold with geodesic coordinates"""
    z: List[float]  # Position
    ż: List[float]  # Velocity (cotangent vector)
    
    # Metric properties
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    total_energy: float = 0.0
    
    # Geometry
    metric_determinant: float = 1.0
    connection_coeffs: List[List[List[float]]] = field(default_factory=list)
    
    # Trajectory
    trajectory_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def compute_total_energy(self, metric: RiemannianMetric, energy_func: SingleEnergyFunctional):
        self.kinetic_energy = metric.kinetic_energy(self.ż)
        self.potential_energy = energy_func.compute_V(self.z)
        self.total_energy = self.kinetic_energy + self.potential_energy
        self.metric_determinant = metric.det_g
    
    def to_dict(self) -> Dict:
        return {
            "kinetic_energy": round(self.kinetic_energy, 4),
            "potential_energy": round(self.potential_energy, 4),
            "total_energy": round(self.total_energy, 4),
            "metric_det": round(self.metric_determinant, 4),
        }


class RiemannianCognitiveManifold:
    """
    Phase 14.5 - Riemannian Cognitive Manifold
    
    Single mathematical object: (M, g, V)
    
    M = latent manifold
    g = Riemannian metric (kinetic energy, inertia)
    V = scalar potential (attractors, goals, preferences)
    
    Lagrangian: L(z, ż) = T - V = 0.5 * g_ij(z) * ż^i * ż^j - V(z)
    
    Euler-Lagrange: ∇_ż (g_ij ż^i ż^j) - ∇_z V = 0
    
    Geodesic equation with potential:
    z̈^i + Γ^i_jk * ż^j * ż^k = -g^ij * ∂_j V
    
    Key unification:
    - Policy = V_policy (potential, not force)
    - Inertia = g_ij (metric, not separate)
    - Energy = V(z) (single functional)
    - Dynamics = geodesics on manifold
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Single unified components
        self.metric = RiemannianMetric(dimension)
        self.energy = SingleEnergyFunctional(dimension)
        
        # Trajectory history
        self.trajectories: List[List[GeodesicState]] = []
        self.current_trajectory: List[GeodesicState] = []
        
        # Integration parameters
        self.dt: float = 0.01
        self.damping: float = 0.05
        
        logger.info("riemannian_cognitive_manifold_initialized", dimension=dimension)
    
    def observe(self, z: List[float], outcome: Optional[str] = None) -> GeodesicState:
        """
        Observe state and update manifold geometry.
        """
        # Update components from observation
        self.energy.observe_state(z, outcome)
        
        # Create geodesic state
        state = GeodesicState(z=z, ż=[0.0] * self.dimension)
        
        # If we have previous state, estimate velocity
        if self.current_trajectory:
            prev = self.current_trajectory[-1]
            dt = (state.timestamp - prev.timestamp).total_seconds()
            if dt > 0:
                state.ż = [(z[i] - prev.z[i]) / dt for i in range(self.dimension)]
                
                # Update metric from velocity
                self.metric.observe_trajectory(z, state.ż)
        
        state.compute_total_energy(self.metric, self.energy)
        
        self.current_trajectory.append(state)
        
        return state
    
    def geodesic_step(self, state: GeodesicState, dt: float = 0.01) -> GeodesicState:
        """
        Single geodesic step with potential.
        
        Euler-Lagrange with metric:
        z̈^i + Γ^i_jk * ż^j * ż^k = -g^ij * ∂_j V
        
        Simplified (geodesic + potential gradient):
        z̈ = -∇V + correction_from_metric
        """
        # Compute potential gradient
        grad_V = self.energy.compute_gradient(state.z)
        
        # Geodesic acceleration (simplified: just potential gradient + damping)
        # Full geodesic would include Christoffel terms
        acceleration = [-g for g in grad_V]
        
        # Add damping
        acceleration = [
            acceleration[i] - self.damping * state.ż[i]
            for i in range(self.dimension)
        ]
        
        # Update velocity
        new_ż = [
            state.ż[i] + dt * acceleration[i]
            for i in range(self.dimension)
        ]
        
        # Update position
        new_z = [
            state.z[i] + dt * new_ż[i]
            for i in range(self.dimension)
        ]
        
        # Create new state
        new_state = GeodesicState(z=new_z, ż=new_ż)
        new_state.compute_total_energy(self.metric, self.energy)
        new_state.connection_coeffs = self.metric.compute_christoffel_at(new_z)
        
        return new_state
    
    def integrate(
        self,
        initial_z: List[float],
        steps: int = 100,
        target_energy: float = 0.1
    ) -> List[GeodesicState]:
        """
        Integrate geodesic from initial condition.
        
        Returns full trajectory on manifold.
        """
        trajectory = []
        
        # Initial state
        state = GeodesicState(z=initial_z, ż=[0.0] * self.dimension)
        state.compute_total_energy(self.metric, self.energy)
        trajectory.append(state)
        
        current = state
        
        for _ in range(steps):
            current = self.geodesic_step(current, self.dt)
            trajectory.append(current)
            
            # Stop if near equilibrium
            if current.potential_energy < target_energy and abs(current.kinetic_energy) < 0.01:
                break
        
        self.trajectories.append(trajectory)
        return trajectory
    
    def set_goal(self, goal: List[float], strength: float = 0.5) -> None:
        """Set goal on manifold (as potential minimum)"""
        self.energy.set_goal(goal, strength)
    
    def set_preference(self, center: List[float], width: float = 1.0) -> None:
        """Set preference function on manifold"""
        self.energy.set_preference(center, width)
    
    def compute_geodesic(self, from_z: List[float], to_z: List[float]) -> List[List[float]]:
        """
        Compute geodesic path between two points.
        
        This is the natural path on the manifold.
        """
        path = [from_z]
        current = from_z
        
        for _ in range(50):
            # Gradient descent on distance
            direction = [to_z[i] - current[i] for i in range(self.dimension)]
            
            # Weight by metric
            weighted = [
                direction[i] * self.metric.g[i][i]
                for i in range(self.dimension)
            ]
            
            step = 0.1
            current = [current[i] + step * weighted[i] for i in range(self.dimension)]
            path.append(current)
            
            # Stop if close to goal
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current, to_z)))
            if dist < 0.1:
                break
        
        return path
    
    def compute_divergence(
        self,
        state1: GeodesicState,
        state2: GeodesicState
    ) -> float:
        """
        Compute divergence between two states on manifold.
        
        Uses metric to compute proper distance.
        """
        delta = [state2.z[i] - state1.z[i] for i in range(self.dimension)]
        
        # Metric-weighted distance
        dist_sq = 0.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                dist_sq += self.metric.g[i][j] * delta[i] * delta[j]
        
        return math.sqrt(max(0, dist_sq))
    
    def get_manifold_statistics(self) -> Dict:
        """Get statistics about manifold geometry"""
        return {
            "dimension": self.dimension,
            "metric": self.metric.to_dict(),
            "energy": self.energy.to_dict(),
            "trajectories": len(self.trajectories),
            "total_states": sum(len(t) for t in self.trajectories),
            "attractors": len(self.energy.attractors),
        }
    
    def get_phase_flow(self) -> Dict:
        """Get phase flow visualization data"""
        if not self.trajectories:
            return {"flows": [], "attractors": []}
        
        flows = []
        for traj in self.trajectories[-3:]:
            flows.append({
                "positions": [s.z[:2] for s in traj],
                "velocities": [s.ż[:2] for s in traj],
            })
        
        attractors = [
            {"position": a[:2], "depth": d}
            for a, d in zip(self.energy.attractors, self.energy.attractor_depths)
        ]
        
        return {
            "flows": flows,
            "attractors": attractors,
            "goal": self.energy.goal_center[:2] if self.energy.goal_center else None,
        }


# Factory
def create_riemannian_manifold(dimension: int = 16) -> RiemannianCognitiveManifold:
    return RiemannianCognitiveManifold(dimension=dimension)