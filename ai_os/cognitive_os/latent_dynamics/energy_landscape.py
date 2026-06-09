"""
EnergyLandscape - Continuous Latent Topology

This is the architectural shift from:

motif-centric (discrete attractors as objects)
    ↓
energy landscape (continuous attractor basins)

Instead of:
- motifs as objects with centroid
- transitions as edges between objects

We have:
- energy field over latent space
- local minima as attractor regions
- gradient flow as dynamics
- basin boundaries as decision boundaries

Key properties:
- Continuous, not discrete
- Emergent attractors, not pre-defined clusters
- Gradient-based dynamics, not transition matrix
- Energy function over latent space
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EnergyPoint:
    """A point in energy landscape"""
    position: List[float]
    energy: float = 0.0
    
    # Gradient (direction of steepest descent)
    gradient: List[float] = field(default_factory=list)
    gradient_magnitude: float = 0.0
    
    # Basin membership
    basin_id: Optional[str] = None
    basin_confidence: float = 0.0
    
    # Stability
    is_local_minimum: bool = False
    is_saddle: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "energy": round(self.energy, 4),
            "gradient_magnitude": round(self.gradient_magnitude, 4),
            "basin_id": self.basin_id,
            "is_local_minimum": self.is_local_minimum,
        }


@dataclass
class AttractorBasin:
    """
    Attractor basin in energy landscape.
    
    NOT a motif object - a region of stability in continuous space.
    """
    basin_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Center of basin (local minimum)
    center: List[float] = field(default_factory=list)
    center_energy: float = 0.0
    
    # Basin geometry
    points: List[List[float]] = field(default_factory=list)
    boundary_points: List[List[float]] = field(default_factory=list)
    
    # Energy properties
    depth: float = 0.0  # Energy depth below saddle points
    width: float = 0.0  # Spatial extent
    volume: float = 0.0  # Approximate volume in latent space
    
    # Flow properties
    escape_cost: float = 0.0  # Energy barrier to escape
    capture_rate: float = 0.0  # How quickly flows into this basin
    
    # Temporal
    stability_score: float = 0.0
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "basin_id": self.basin_id,
            "center_energy": round(self.center_energy, 4),
            "depth": round(self.depth, 4),
            "width": round(self.width, 4),
            "escape_cost": round(self.escape_cost, 4),
            "stability_score": round(self.stability_score, 3),
            "points_count": len(self.points),
        }


class EnergyField:
    """
    Energy field over latent space.
    
    E(x) = energy at position x
    
    Energy decreases towards attractors (minima).
    Gradient points towards steepest descent.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Energy at sampled points
        self.samples: List[EnergyPoint] = []
        
        # Basins (attractor regions)
        self.basins: Dict[str, AttractorBasin] = {}
        
        # Field parameters
        self.baseline_energy: float = 1.0
        self.attraction_strength: float = 0.5
        self.repulsion_strength: float = 0.1
        
        logger.info("energy_field_initialized", dimension=dimension)
    
    def add_sample(self, position: List[float], energy: float = None) -> EnergyPoint:
        """Add a sample point to the energy field"""
        if energy is None:
            energy = self._compute_energy(position)
        
        point = EnergyPoint(
            position=position,
            energy=energy,
            gradient=self._compute_gradient(position),
        )
        point.gradient_magnitude = math.sqrt(sum(g ** 2 for g in point.gradient))
        
        self.samples.append(point)
        return point
    
    def _compute_energy(self, position: List[float]) -> float:
        """
        Compute energy at position.
        
        Energy = sum of attractive forces from basins - repulsion from high-density regions
        
        This is a simplified model. In production, this would be a learned function.
        """
        energy = self.baseline_energy
        
        # Attraction to basins
        for basin_id, basin in self.basins.items():
            dist = self._distance(position, basin.center)
            if dist > 0:
                # Energy decreases as we approach basin center
                energy -= self.attraction_strength * math.exp(-dist ** 2)
        
        # Repulsion from densely sampled regions
        for sample in self.samples[-100:]:  # Only recent samples
            dist = self._distance(position, sample.position)
            if dist > 0 and dist < 0.5:
                energy += self.repulsion_strength * math.exp(-dist ** 2)
        
        return max(0.0, energy)
    
    def _compute_gradient(self, position: List[float], epsilon: float = 0.01) -> List[float]:
        """Compute gradient using finite differences"""
        gradient = []
        
        for i in range(self.dimension):
            pos_plus = position[:]
            pos_plus[i] += epsilon
            e_plus = self._compute_energy(pos_plus)
            
            pos_minus = position[:]
            pos_minus[i] -= epsilon
            e_minus = self._compute_energy(pos_minus)
            
            gradient.append((e_plus - e_minus) / (2 * epsilon))
        
        return gradient
    
    def _distance(self, a: List[float], b: List[float]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def find_nearest_basin(self, position: List[float]) -> Tuple[Optional[str], float]:
        """Find nearest basin and distance"""
        if not self.basins:
            return None, float('inf')
        
        min_dist = float('inf')
        nearest_basin = None
        
        for basin_id, basin in self.basins.items():
            dist = self._distance(position, basin.center)
            if dist < min_dist:
                min_dist = dist
                nearest_basin = basin_id
        
        return nearest_basin, min_dist
    
    def compute_flow(self, position: List[float]) -> List[float]:
        """
        Compute flow direction at position.
        
        Flow = -gradient (steepest descent towards attractors)
        """
        gradient = self._compute_gradient(position)
        return [-g for g in gradient]
    
    def compute_flow_magnitude(self, position: List[float]) -> float:
        """Compute magnitude of flow"""
        flow = self.compute_flow(position)
        return math.sqrt(sum(f ** 2 for f in flow))
    
    def add_basin(self, center: List[float], energy: float = None) -> str:
        """Add a new attractor basin"""
        if energy is None:
            energy = self._compute_energy(center)
        
        basin_id = f"basin_{len(self.basins)}"
        
        basin = AttractorBasin(
            basin_id=basin_id,
            center=center,
            center_energy=energy,
            depth=energy,
        )
        
        self.basins[basin_id] = basin
        return basin_id
    
    def update_basin(self, basin_id: str, new_points: List[List[float]]) -> None:
        """Update basin with new trajectory points"""
        if basin_id not in self.basins:
            return
        
        basin = self.basins[basin_id]
        basin.points.extend(new_points)
        basin.last_observed = datetime.utcnow()
        
        # Update center (weighted average)
        if basin.points:
            dim = len(basin.center)
            new_center = [0.0] * dim
            for p in basin.points:
                for i in range(dim):
                    new_center[i] += p[i]
            basin.center = [v / len(basin.points) for v in new_center]
        
        # Update geometry
        if len(basin.points) > 1:
            # Width: average distance from center
            widths = [self._distance(p, basin.center) for p in basin.points]
            basin.width = sum(widths) / len(widths)
            
            # Stability: inverse of variance
            variance = sum(w ** 2 for w in widths) / len(widths)
            basin.stability_score = 1.0 / (1.0 + variance)
    
    def get_flow_field(self, resolution: int = 5) -> Dict:
        """
        Get flow field visualization data.
        
        Returns grid of positions and their flows.
        """
        if not self.samples:
            return {"vectors": [], "basins": []}
        
        # Get bounds
        min_vals = [min(s.position[i] for s in self.samples) for i in range(self.dimension)]
        max_vals = [max(s.position[i] for s in self.samples) for i in range(self.dimension)]
        
        vectors = []
        
        # Sample flow at grid points
        for i in range(resolution):
            for j in range(resolution):
                if self.dimension >= 2:
                    pos = [
                        min_vals[0] + (max_vals[0] - min_vals[0]) * i / (resolution - 1),
                        min_vals[1] + (max_vals[1] - min_vals[1]) * j / (resolution - 1),
                    ] + [0.0] * (self.dimension - 2)
                    
                    flow = self.compute_flow(pos)
                    energy = self._compute_energy(pos)
                    
                    vectors.append({
                        "position": pos,
                        "flow": flow,
                        "energy": energy,
                    })
        
        return {
            "vectors": vectors,
            "basins": [b.to_dict() for b in self.basins.values()],
        }


class EnergyLandscape:
    """
    Energy Landscape - continuous attractor topology.
    
    This replaces motif-centric thinking with continuous energy field.
    
    Key shift:
    - Motifs as discrete objects → Energy as continuous field
    - Transition matrix → Gradient flow
    - Cluster centers → Local minima
    - Attractor basins as regions of stability
    
    This is the foundation for:
    - Continuous trajectory prediction
    - Gradient-based planning
    - Energy-based inference
    - Metastability analysis
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        self.field = EnergyField(dimension)
        
        # Trajectory history
        self.trajectories: List[List[List[float]]] = []
        self.trajectory_energies: List[List[float]] = []
        
        # Learned energy function (future: neural network)
        self._energy_function: Optional[Callable] = None
        
        logger.info("energy_landscape_initialized", dimension=dimension)
    
    def add_trajectory(self, trajectory: List[List[float]]) -> None:
        """
        Add a trajectory and update energy field.
        
        Trajectory should be sequence of latent state vectors.
        """
        self.trajectories.append(trajectory)
        
        energies = []
        for position in trajectory:
            point = self.field.add_sample(position)
            energies.append(point.energy)
            
            # Update nearest basin
            basin_id, _ = self.field.find_nearest_basin(position)
            if basin_id:
                self.field.update_basin(basin_id, [position])
            else:
                # Create new basin
                basin_id = self.field.add_basin(position)
        
        self.trajectory_energies.append(energies)
    
    def update_energy_function(self, func: Callable) -> None:
        """
        Update learned energy function.
        
        In production, this would be a neural network:
        E(x) = neural_network(x)
        """
        self._energy_function = func
    
    def predict_trajectory(
        self,
        start_position: List[float],
        steps: int = 10,
        dt: float = 0.1
    ) -> List[List[float]]:
        """
        Predict trajectory using gradient descent.
        
        x(t+1) = x(t) - dt * gradient(E, x)
        
        This is gradient flow prediction.
        """
        trajectory = [start_position]
        current = start_position
        
        for _ in range(steps):
            flow = self.field.compute_flow(current)
            
            # Update position: move along flow
            next_pos = [
                current[i] + dt * flow[i]
                for i in range(self.dimension)
            ]
            
            trajectory.append(next_pos)
            current = next_pos
        
        return trajectory
    
    def compute_expected_energy(
        self,
        start_position: List[float],
        steps: int = 10
    ) -> float:
        """Compute expected energy trajectory"""
        trajectory = self.predict_trajectory(start_position, steps)
        
        total_energy = 0.0
        for pos in trajectory:
            energy = self.field._compute_energy(pos)
            total_energy += energy
        
        return total_energy / len(trajectory)
    
    def find_minimum_path(
        self,
        from_pos: List[float],
        to_pos: List[float],
        resolution: int = 20
    ) -> List[List[float]]:
        """
        Find minimum energy path between two positions.
        
        Uses gradient descent with bias towards target.
        """
        path = [from_pos]
        current = from_pos
        
        for _ in range(resolution):
            # Gradient of energy
            grad = self.field._compute_gradient(current)
            
            # Bias towards target
            to_bias = [
                (to_pos[i] - current[i]) / self.dimension
                for i in range(self.dimension)
            ]
            
            # Combined direction
            direction = [
                -grad[i] * 0.7 + to_bias[i] * 0.3
                for i in range(self.dimension)
            ]
            
            # Normalize
            norm = math.sqrt(sum(d ** 2 for d in direction))
            if norm > 0:
                direction = [d / norm for d in direction]
            
            # Step
            step_size = self._distance(from_pos, to_pos) / resolution
            next_pos = [
                current[i] + step_size * direction[i]
                for i in range(self.dimension)
            ]
            
            path.append(next_pos)
            current = next_pos
        
        return path
    
    def compute_attractor_strength(self, position: List[float]) -> float:
        """
        Compute how strongly position is attracted to nearest basin.
        
        Higher = more deeply in attractor basin
        """
        basin_id, distance = self.field.find_nearest_basin(position)
        
        if basin_id is None:
            return 0.0
        
        basin = self.field.basins[basin_id]
        
        # Strength decreases with distance and increases with basin depth
        dist_factor = math.exp(-distance ** 2)
        depth_factor = 1.0 / (1.0 + basin.center_energy)
        
        return dist_factor * (1.0 + basin.depth)
    
    def get_landscape_statistics(self) -> Dict:
        """Get comprehensive landscape statistics"""
        basin_stats = []
        
        for basin in self.field.basins.values():
            basin_stats.append({
                "id": basin.basin_id,
                "depth": basin.depth,
                "width": basin.width,
                "stability": basin.stability_score,
                "escape_cost": basin.escape_cost,
                "points": len(basin.points),
            })
        
        return {
            "dimension": self.dimension,
            "total_samples": len(self.field.samples),
            "total_basins": len(self.field.basins),
            "total_trajectories": len(self.trajectories),
            "avg_energy": (
                sum(s.energy for s in self.field.samples) / len(self.field.samples)
                if self.field.samples else 0
            ),
            "basins": basin_stats,
            "has_learned_function": self._energy_function is not None,
        }
    
    def visualize_flow(self) -> Dict:
        """Get visualization data for flow field"""
        return self.field.get_flow_field(resolution=10)


# Factory
def create_energy_landscape(dimension: int = 16) -> EnergyLandscape:
    return EnergyLandscape(dimension=dimension)