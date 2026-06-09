"""
LearnedEnergyField - Energy as Emergent Property

NOT hand-crafted: energy -= exp(-dist²)

BUT learned: E(z) = f(trajectory_statistics)

Energy emerges from:
1. Trajectory density (regions with many trajectories = low energy)
2. Transition patterns (frequently visited = attractor = low energy)
3. Outcome correlation (successful trajectories = low energy basins)
4. Prediction error (high error regions = high energy = uncertainty)
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
class EnergyStats:
    """Statistics about energy field"""
    avg_energy: float = 0.0
    min_energy: float = 0.0
    max_energy: float = 0.0
    energy_variance: float = 0.0
    
    # Attractor regions
    attractor_count: int = 0
    saddle_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "avg_energy": round(self.avg_energy, 4),
            "min_energy": round(self.min_energy, 4),
            "max_energy": round(self.max_energy, 4),
            "attractor_count": self.attractor_count,
        }


class LearnedEnergyField:
    """
    Energy field learned from trajectory statistics.
    
    Energy is NOT hand-crafted geometry.
    Energy emerges from observed dynamics.
    
    E(z) = base_energy
         + density_term (low where many trajectories)
         + transition_term (low at attractors)
         + prediction_error_term (high where uncertain)
         + outcome_term (low at successful regions)
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Trajectory statistics for learning
        self.trajectory_points: List[List[float]] = []
        self.transitions: List[Tuple[List[float], List[float]]] = []
        self.outcomes: List[Tuple[List[float], str]] = []  # (state, outcome)
        
        # Learned parameters
        self.base_energy: float = 1.0
        self.density_scale: float = 0.3
        self.transition_scale: float = 0.3
        self.prediction_scale: float = 0.4
        
        # Density estimator (simplified kernel density)
        self.kernel_bandwidth: float = 0.5
        
        # Attractor candidates (learned from data)
        self.attractors: List[List[float]] = []
        
        # Energy cache
        self.energy_cache: Dict[str, float] = {}
        
        logger.info("learned_energy_field_initialized", dimension=dimension)
    
    def observe_trajectory(self, trajectory: List[List[float]]) -> None:
        """Add trajectory to learning data"""
        self.trajectory_points.extend(trajectory)
        
        # Record transitions
        for i in range(len(trajectory) - 1):
            self.transitions.append((trajectory[i], trajectory[i + 1]))
    
    def observe_outcome(self, state: List[float], outcome: str) -> None:
        """Add outcome observation"""
        self.outcomes.append((state, outcome))
    
    def learn_energy_parameters(self) -> None:
        """
        Learn energy parameters from trajectory statistics.
        
        This is where energy emerges from data.
        """
        if len(self.trajectory_points) < 10:
            logger.warning("insufficient_data_for_energy_learning")
            return
        
        # Learn attractors from transition patterns
        self._learn_attractors()
        
        # Estimate kernel bandwidth from data
        self._estimate_bandwidth()
        
        # Estimate prediction error scale
        self._estimate_prediction_scale()
        
        logger.info("energy_parameters_learned",
                   attractors=len(self.attractors),
                   points=len(self.trajectory_points))
    
    def _learn_attractors(self) -> None:
        """
        Learn attractors from trajectory data.
        
        Attractor = region where system spends lots of time
        (many visits, slow transitions)
        """
        if len(self.trajectory_points) < 10:
            return
        
        # Cluster recent points to find attractors
        points = self.trajectory_points[-500:]  # Use recent points
        
        # Simple clustering: density-based
        density_map = defaultdict(int)
        
        for i, p1 in enumerate(points):
            for p2 in points[i+1:]:
                dist = self._distance(p1, p2)
                if dist < 0.3:
                    density_map[tuple(int(x * 10) for x in p1[:3])] += 1
        
        # Find high-density regions
        sorted_densities = sorted(density_map.items(), key=lambda x: x[1], reverse=True)
        
        self.attractors = []
        for key, count in sorted_densities[:10]:  # Top 10 attractors
            if count >= 3:
                center = [k / 10 for k in key]
                self.attractors.append(center)
    
    def _estimate_bandwidth(self) -> None:
        """Estimate kernel bandwidth from data"""
        if len(self.trajectory_points) < 10:
            return
        
        # Use median distance to nearest neighbor
        distances = []
        
        for i, p1 in enumerate(self.trajectory_points[-100:]):
            min_dist = float('inf')
            for j, p2 in enumerate(self.trajectory_points[-100:]):
                if i != j:
                    d = self._distance(p1, p2)
                    min_dist = min(min_dist, d)
            distances.append(min_dist)
        
        if distances:
            self.kernel_bandwidth = sorted(distances)[len(distances) // 2]
    
    def _estimate_prediction_scale(self) -> None:
        """Estimate prediction error contribution"""
        if len(self.transitions) < 10:
            return
        
        # Compute typical transition magnitude
        magnitudes = []
        for z1, z2 in self.transitions[-100:]:
            mag = self._distance(z1, z2)
            magnitudes.append(mag)
        
        avg_mag = sum(magnitudes) / len(magnitudes) if magnitudes else 0.2
        
        # Prediction scale inversely related to typical step
        self.prediction_scale = min(1.0, avg_mag * 2)
    
    def _distance(self, a: List[float], b: List[float]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def compute_energy(self, position: List[float], cache: bool = True) -> float:
        """
        Compute energy at position.
        
        E(z) = base + density_term + attractor_term + prediction_term
        """
        # Check cache
        cache_key = self._position_to_key(position)
        if cache and cache_key in self.energy_cache:
            return self.energy_cache[cache_key]
        
        # Base energy
        energy = self.base_energy
        
        # Density term: many trajectories nearby = low energy
        density_term = self._compute_density_term(position)
        energy -= density_term * self.density_scale
        
        # Attractor term: near learned attractors = low energy
        attractor_term = self._compute_attractor_term(position)
        energy -= attractor_term * self.transition_scale
        
        # Prediction term: high prediction error = high energy
        prediction_term = self._compute_prediction_term(position)
        energy += prediction_term * self.prediction_scale
        
        # Outcome term: successful outcomes = low energy basins
        outcome_term = self._compute_outcome_term(position)
        energy -= outcome_term * 0.2
        
        energy = max(0.0, energy)
        
        if cache:
            self.energy_cache[cache_key] = energy
        
        return energy
    
    def _position_to_key(self, position: List[float]) -> str:
        """Convert position to cache key"""
        return ",".join(f"{v:.2f}" for v in position[:4])
    
    def _compute_density_term(self, position: List[float]) -> float:
        """Compute density contribution to energy"""
        if len(self.trajectory_points) < 10:
            return 0.0
        
        # Kernel density estimation (simplified Gaussian)
        density = 0.0
        h = self.kernel_bandwidth
        
        for point in self.trajectory_points[-200:]:  # Recent points only
            dist = self._distance(position, point)
            density += math.exp(-dist ** 2 / (2 * h ** 2))
        
        # Normalize
        density /= len(self.trajectory_points[-200:])
        
        return density
    
    def _compute_attractor_term(self, position: List[float]) -> float:
        """Compute attractor contribution to energy"""
        if not self.attractors:
            return 0.0
        
        # Closest attractor determines energy drop
        min_dist = min(self._distance(position, a) for a in self.attractors)
        
        # Exponential decay with distance
        return math.exp(-min_dist ** 2 / 0.5)
    
    def _compute_prediction_term(self, position: List[float]) -> float:
        """Compute prediction uncertainty contribution"""
        if len(self.transitions) < 10:
            return 0.5
        
        # Near transitions = higher prediction uncertainty
        min_transition_dist = float('inf')
        
        for z1, z2 in self.transitions[-100:]:
            mid = [(z1[i] + z2[i]) / 2 for i in range(self.dimension)]
            dist = self._distance(position, mid)
            min_transition_dist = min(min_transition_dist, dist)
        
        # High uncertainty near transitions
        uncertainty = 1.0 / (1.0 + min_transition_dist)
        
        return uncertainty
    
    def _compute_outcome_term(self, position: List[float]) -> float:
        """Compute outcome correlation contribution"""
        if not self.outcomes:
            return 0.0
        
        # Look at nearby successful outcomes
        success_count = 0
        total_count = 0
        
        for state, outcome in self.outcomes[-100:]:
            dist = self._distance(position, state)
            if dist < 0.5:
                total_count += 1
                if outcome == "success":
                    success_count += 1
        
        if total_count == 0:
            return 0.0
        
        # More successes nearby = lower energy
        return success_count / total_count
    
    def compute_gradient(self, position: List[float], epsilon: float = 0.01) -> List[float]:
        """Compute gradient of energy field"""
        gradient = []
        
        for i in range(self.dimension):
            pos_plus = position[:]
            pos_plus[i] += epsilon
            e_plus = self.compute_energy(pos_plus, cache=False)
            
            pos_minus = position[:]
            pos_minus[i] -= epsilon
            e_minus = self.compute_energy(pos_minus, cache=False)
            
            gradient.append((e_plus - e_minus) / (2 * epsilon))
        
        return gradient
    
    def find_attractors(self) -> List[Dict]:
        """Find all attractor regions"""
        attractors = []
        
        for center in self.attractors:
            energy = self.compute_energy(center)
            
            # Compute basin size
            basin_points = []
            for point in self.trajectory_points[-200:]:
                if self._distance(point, center) < 0.5:
                    basin_points.append(point)
            
            attractors.append({
                "center": center,
                "energy": energy,
                "basin_size": len(basin_points),
            })
        
        return attractors
    
    def get_energy_statistics(self) -> EnergyStats:
        """Get energy field statistics"""
        if not self.trajectory_points:
            return EnergyStats()
        
        # Sample energy at recent points
        energies = []
        for point in self.trajectory_points[-100:]:
            energies.append(self.compute_energy(point))
        
        if not energies:
            return EnergyStats()
        
        return EnergyStats(
            avg_energy=sum(energies) / len(energies),
            min_energy=min(energies),
            max_energy=max(energies),
            energy_variance=sum((e - sum(energies) / len(energies)) ** 2 for e in energies) / len(energies),
            attractor_count=len(self.attractors),
        )


class ActiveInferenceEngine:
    """
    Active Inference - minimize expected free energy.
    
    Active inference principle:
    - Actions chosen to minimize surprise
    - Perception minimizes prediction error
    - Preferences encoded as prior beliefs
    
    Expected Free Energy = Risk + Ambiguity
    
    G = ∑_t [γ * Risk(t) + (1-γ) * Ambiguity(t)]
    
    Risk = divergence from preferred state
    Ambiguity = uncertainty about observations
    """
    
    def __init__(self, learned_energy: LearnedEnergyField):
        self.energy_field = learned_energy
        
        # Preferences (preferred states)
        self.preferred_state: Optional[List[float]] = None
        self.preference_strength: float = 0.5
        
        # Active inference parameters
        self.precision: float = 1.0  # How much weight on precision
        self.depth: int = 5  # Planning horizon
        
        logger.info("active_inference_engine_initialized")
    
    def set_preferences(self, preferred_state: List[float], strength: float = 0.5) -> None:
        """Set preferred state for active inference"""
        self.preferred_state = preferred_state
        self.preference_strength = strength
    
    def compute_expected_free_energy(
        self,
        start_state: List[float],
        action_sequence: List[str],
        horizon: int = 5
    ) -> Tuple[float, List[float]]:
        """
        Compute expected free energy of action sequence.
        
        Returns:
        - total_energy (lower = better)
        - preferred_trajectory (how to minimize energy)
        """
        current = start_state
        total_energy = 0.0
        preferred_path = [start_state]
        
        for step in range(horizon):
            # Current energy
            current_energy = self.energy_field.compute_energy(current)
            total_energy += current_energy
            
            # Gradient towards lower energy
            gradient = self.energy_field.compute_gradient(current)
            
            # Compute preferred next state (steepest descent)
            step_size = 0.1
            next_state = [
                current[i] - step_size * gradient[i]
                for i in range(self.dimension)
            ]
            
            # If we have preferred state, add pull towards it
            if self.preferred_state:
                preference_pull = [
                    self.preference_strength * (self.preferred_state[i] - current[i])
                    for i in range(self.dimension)
                ]
                next_state = [
                    next_state[i] + 0.1 * preference_pull[i]
                    for i in range(self.dimension)
                ]
            
            preferred_path.append(next_state)
            current = next_state
        
        return total_energy, preferred_path
    
    def select_action(
        self,
        current_state: List[float],
        possible_actions: List[str],
        context: Optional[Dict] = None
    ) -> str:
        """
        Select action that minimizes expected free energy.
        
        This is the core of active inference:
        - Evaluate each possible action
        - Predict trajectory for each
        - Choose action with lowest expected energy
        """
        best_action = possible_actions[0] if possible_actions else "wait"
        best_energy = float('inf')
        
        for action in possible_actions:
            # Predict trajectory for this action
            trajectory = self._predict_action_trajectory(
                current_state,
                action,
                horizon=self.depth
            )
            
            # Compute expected energy
            total_energy = 0.0
            for state in trajectory:
                total_energy += self.energy_field.compute_energy(state)
            
            avg_energy = total_energy / len(trajectory)
            
            if avg_energy < best_energy:
                best_energy = avg_energy
                best_action = action
        
        return best_action
    
    def _predict_action_trajectory(
        self,
        start_state: List[float],
        action: str,
        horizon: int = 5
    ) -> List[List[float]]:
        """Predict trajectory for given action"""
        trajectory = [start_state]
        current = start_state
        
        # Action effect on state
        action_effect = self._get_action_effect(action)
        
        for _ in range(horizon):
            # Move along energy gradient + action effect
            gradient = self.energy_field.compute_gradient(current)
            
            next_state = [
                current[i] - 0.1 * gradient[i] + 0.05 * action_effect[i]
                for i in range(self.dimension)
            ]
            
            trajectory.append(next_state)
            current = next_state
        
        return trajectory
    
    def _get_action_effect(self, action: str) -> List[float]:
        """Get effect of action on state"""
        effect = [0.0] * self.energy_field.dimension
        
        action_effects = {
            "execute": [0.1, 0.0, 0.0],
            "explore": [0.0, 0.1, 0.05],
            "decompose": [0.0, 0.0, 0.1],
            "wait": [0.0, 0.0, 0.0],
            "reconsider": [-0.05, -0.05, 0.0],
        }
        
        mapped = action_effects.get(action, [0.0, 0.0, 0.0])
        for i in range(min(len(mapped), self.energy_field.dimension)):
            effect[i] = mapped[i]
        
        return effect


# Factory functions
def create_learned_energy_field(dimension: int = 16) -> LearnedEnergyField:
    return LearnedEnergyField(dimension=dimension)


def create_active_inference_engine(energy_field: LearnedEnergyField) -> ActiveInferenceEngine:
    return ActiveInferenceEngine(energy_field)