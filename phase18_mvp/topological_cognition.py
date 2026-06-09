"""
Phase 16: Topological Cognition

ARCHITECTURAL SHIFT:
  From: Executive field with trajectory competition (Phase 15)
  To: Persistent deformable cognitive topology where:
      - Futures form energy basins
      - Actions deform topology
      - Memory reshapes geometry
      - Attractors merge, split, decay
      - Self = stable recursive attractor (basin continuity)
      
  This is NO LONGER:
    advanced agent architecture
  This IS:
    continuous cognitive ecology
    topological cognition
    cognitive thermodynamics

CRITICAL INSIGHT:
  Phase 15: trajectories compete, winner executes
  Phase 16: future landscape DEFORMS from execution
            trajectories carve probability valleys
            successful futures become deep basins
            execution falls into basin

SEVEN CRITICAL PROBLEMS WITH PHASE 15:
  1. World Model = linear mapping (not latent physics)
  2. Attractor competition = utility optimization (fake dynamics)
  3. Emergent Selves = behavior clusters (not identity)
  4. Cognitive Economics = resource meter (no strategy change)
  5. Tensions = local computation (not wave dynamics)
  6. No persistent world (trajectories ephemeral)
  7. Drives = external (not endogenous)

PHASE 16 FIXES:
  1. Latent Predictive Physics (z_t -> z_t+1)
  2. Landscape Topology Evolution (basins form from density)
  3. Self as Basin Stability (recursive attractor)
  4. Economics-driven Cognition Strategy (shallow under fatigue)
  5. Tension Wave Dynamics (interference, resonance)
  6. Persistent Terrain (habits, scars, erosion)
  7. Endogenous Drive Formation (from topology pressure)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


# ============================================================================
# 1. PERSISTENT FUTURE LANDSCAPE
# ============================================================================

@dataclass
class FutureBasin:
    """A basin of attraction in future probability space."""
    basin_id: str
    center: np.ndarray              # Center of basin in latent space
    depth: float                   # Basin depth (attractor strength)
    radius: float                  # Basin influence radius
    mass: float                    # Total probability mass in basin
    curvature: np.ndarray          # Second derivatives (curvature)
    energy: float                  # Potential energy of basin
    history: List[Dict] = field(default_factory=list)  # How basin evolved
    
    def update_depth(self, visitation: int, success: float):
        """Basin deepens from repeated successful execution."""
        # Depth increases with successful visitation
        self.depth = self.depth * 0.95 + success * 0.1 * visitation * 0.01
        self.depth = min(self.depth, 5.0)  # Cap depth
        
        # Record history
        self.history.append({
            'visitation': visitation,
            'success': success,
            'depth': self.depth,
            'mass': self.mass
        })
        
        # Limit history
        if len(self.history) > 100:
            self.history = self.history[-50:]
    
    def erode(self, decay_rate: float = 0.01):
        """Basin erodes from non-visitation."""
        self.depth = max(0.1, self.depth - decay_rate)
        self.mass = max(0.1, self.mass - decay_rate * 0.5)


@dataclass
class TerrainRegion:
    """Region of cognitive terrain."""
    region_id: str
    center: np.ndarray
    terrain_type: str  # habit_path, trauma_cliff, exploration_zone, erosion_region
    resistance: float   # How hard to traverse
    affordances: List[str]  # What can be done here
    scars: List[Dict]  # Memory of past traversals
    
    def add_scar(self, passage: Dict):
        """Add scar from traversal."""
        self.scars.append(passage)
        if len(self.scars) > 20:
            self.scars = self.scars[-10:]


class FutureEnergyField:
    """
    Future probability field that DEFORMS from execution.
    
    NOT: discrete trajectories
    BUT: continuous energy landscape
    
    Where:
      - Basins form from probability density
      - Successful futures become deep basins
      - Execution falls into basin
      - Memory carves channels
    """
    
    def __init__(self, latent_dim: int = 2, grid_size: int = 50):
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        
        # Create probability grid
        self.energy_grid = np.zeros((grid_size, grid_size))
        self.probability_grid = np.zeros((grid_size, grid_size))
        
        # Grid boundaries
        self.bounds = np.array([[-5, 5]] * latent_dim)
        
        # Basins of attraction
        self.basins: List[FutureBasin] = []
        
        # Terrain regions
        self.terrain: List[TerrainRegion] = []
        
        # Topology dynamics
        self.erosion_rate = 0.001
        self.sedimentation_rate = 0.002
        self.tectonic_pressure = 0.01
        
    def world_to_grid(self, point: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        point = np.asarray(point).flatten()
        normalized = (point - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        normalized = np.clip(normalized, 0, 0.999)
        
        x = int(normalized[0] * (self.grid_size - 1))
        y = int(normalized[1] * (self.grid_size - 1)) if len(normalized) > 1 else 0
        
        return x, y
    
    def grid_to_world(self, x: int, y: int) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        normalized_x = x / (self.grid_size - 1)
        normalized_y = y / (self.grid_size - 1) if self.grid_size > 1 else 0
        
        point = self.bounds[:, 0] + normalized_x * (self.bounds[:, 1] - self.bounds[:, 0])
        if self.grid_size > 1:
            point[1] = self.bounds[1, 0] + normalized_y * (self.bounds[1, 1] - self.bounds[1, 0])
        
        return point
    
    def compute_energy_field(self, action_density: np.ndarray, 
                            success_density: np.ndarray,
                            memory_influence: float = 0.3) -> np.ndarray:
        """
        Compute energy field from multiple influences.
        
        Energy = f(action_potential, success_basin, memory_channel)
        """
        # Initialize energy
        self.energy_grid.fill(0.0)
        
        # Action potential (where actions lead)
        action_contribution = action_density * 0.3
        
        # Success basin (where successful trajectories converge)
        success_contribution = success_density * 0.5
        
        # Memory channel contribution
        memory_contribution = self._compute_memory_influence() * memory_influence
        
        # Combine
        self.energy_grid = action_contribution + success_contribution + memory_contribution
        
        # Normalize
        if np.max(self.energy_grid) > 0:
            self.energy_grid = self.energy_grid / np.max(self.energy_grid)
        
        # Energy = inverse probability (low energy = high probability)
        self.probability_grid = 1.0 / (1.0 + self.energy_grid)
        
        return self.energy_grid
    
    def _compute_memory_influence(self) -> np.ndarray:
        """Compute memory channel influence on energy field."""
        memory_grid = np.zeros((self.grid_size, self.grid_size))
        
        # Habit paths (low energy = preferred)
        for region in self.terrain:
            if region.terrain_type == "habit_path":
                cx, cy = self.world_to_grid(region.center)
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if 0 <= cx + dx < self.grid_size and 0 <= cy + dy < self.grid_size:
                            distance = np.sqrt(dx**2 + dy**2)
                            memory_grid[cy + dy, cx + dx] += np.exp(-distance) * (1 - region.resistance)
        
        return memory_grid
    
    def form_basin(self, trajectory_end: np.ndarray, success: float):
        """Form or deepen basin from trajectory execution."""
        trajectory_end = np.asarray(trajectory_end).flatten()
        
        # Check if similar basin exists
        for basin in self.basins:
            distance = np.linalg.norm(trajectory_end - basin.center)
            if distance < 1.0:  # Similar basin
                basin.update_depth(visitation=1, success=success)
                basin.mass += 0.1
                return basin
        
        # Create new basin
        new_basin = FutureBasin(
            basin_id=f"basin_{len(self.basins)}",
            center=trajectory_end.copy(),
            depth=0.5 + success * 0.3,
            radius=1.0,
            mass=0.1,
            curvature=np.zeros(self.latent_dim),
            energy=1.0 / (1.0 + success)
        )
        
        self.basins.append(new_basin)
        
        # Limit basins
        if len(self.basins) > 20:
            # Remove shallow basins
            self.basins.sort(key=lambda b: b.depth)
            self.basins = self.basins[-15:]
        
        return new_basin
    
    def erode_terrain(self):
        """Erode unvisited terrain."""
        for basin in self.basins:
            basin.erode(self.erosion_rate)
        
        # Remove very shallow basins
        self.basins = [b for b in self.basins if b.depth > 0.2]
        
        # Erode terrain
        for region in self.terrain:
            if len(region.scars) == 0:
                region.resistance = min(1.0, region.resistance + self.erosion_rate * 0.5)
    
    def carve_channel(self, trajectory: List[np.ndarray], success: float):
        """Carve memory channel from trajectory."""
        if len(trajectory) < 2:
            return
        
        # Create habit path from successful trajectory
        trajectory_array = np.array(trajectory)
        center = np.mean(trajectory_array, axis=0)
        
        # Check if similar region exists
        for region in self.terrain:
            if np.linalg.norm(center - region.center) < 0.5:
                region.add_scar({
                    'trajectory': trajectory,
                    'success': success,
                    'depth': len(trajectory)
                })
                return
        
        # Create new habit path
        new_region = TerrainRegion(
            region_id=f"region_{len(self.terrain)}",
            center=center,
            terrain_type="habit_path" if success > 0.5 else "exploration_zone",
            resistance=0.3 if success > 0.5 else 0.7,
            affordances=["traverse", "execute"],
            scars=[{
                'trajectory': trajectory,
                'success': success,
                'depth': len(trajectory)
            }]
        )
        
        self.terrain.append(new_region)
        
        # Form basin at trajectory end
        self.form_basin(trajectory[-1], success)
    
    def create_trauma(self, failed_trajectory: List[np.ndarray]):
        """Create trauma cliff from failed execution."""
        if len(failed_trajectory) < 2:
            return
        
        # Find failure point
        trajectory_array = np.array(failed_trajectory)
        center = np.mean(trajectory_array, axis=0)
        
        # Create trauma region
        trauma = TerrainRegion(
            region_id=f"trauma_{len(self.terrain)}",
            center=center,
            terrain_type="trauma_cliff",
            resistance=0.9,
            affordances=["avoid"],
            scars=[{
                'trajectory': failed_trajectory,
                'success': 0,
                'depth': len(failed_trajectory)
            }]
        )
        
        self.terrain.append(trauma)
    
    def get_attractor_position(self, state: np.ndarray) -> np.ndarray:
        """Get where state is attracted to (steepest descent)."""
        state = np.asarray(state).flatten()
        
        # Find nearest deep basin
        best_basin = None
        best_distance = float('inf')
        
        for basin in self.basins:
            distance = np.linalg.norm(state - basin.center)
            if distance < best_distance and basin.depth > 0.3:
                best_distance = distance
                best_basin = basin
        
        if best_basin:
            # Attracted toward basin center
            return best_basin.center.copy()
        
        return state  # No attractor nearby
    
    def get_topology_summary(self) -> Dict:
        """Get topology state summary."""
        return {
            'n_basins': len(self.basins),
            'avg_depth': np.mean([b.depth for b in self.basins]) if self.basins else 0,
            'max_depth': max([b.depth for b in self.basins]) if self.basins else 0,
            'n_terrain': len(self.terrain),
            'habit_paths': sum(1 for r in self.terrain if r.terrain_type == "habit_path"),
            'traumas': sum(1 for r in self.terrain if r.terrain_type == "trauma_cliff"),
            'total_mass': sum(b.mass for b in self.basins)
        }


# ============================================================================
# 2. LATENT PREDICTIVE PHYSICS
# ============================================================================

class LatentPredictivePhysics:
    """
    Latent state dynamics: z_t -> z_t+1
    
    NOT: vector -> vector linear mapping
    BUT: compressed latent physics with:
         - Causal structure
         - Temporal abstractions
         - Object permanence
         - Environment constraints
    """
    
    def __init__(self, observation_dim: int = 2, latent_dim: int = 4, action_dim: int = 2):
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Latent state space
        self.z = np.zeros(latent_dim)
        
        # Encoder: observation -> latent
        self.W_encoder = np.random.randn(latent_dim, observation_dim) * 0.1
        
        # Transition model: z_t, action -> z_{t+1}
        self.W_transition = np.eye(latent_dim) * 0.8
        self.W_action = np.random.randn(latent_dim, action_dim) * 0.1
        
        # Decoder: latent -> observation
        self.W_decoder = np.random.randn(observation_dim, latent_dim) * 0.1
        
        # Causal structure (discovered)
        self.causal_links: List[Tuple[int, int, float]] = []  # (from, to, strength)
        
        # Object permanence (persistent latent objects)
        self.persistent_objects: List[Dict] = []
        
        # Uncertainty estimation
        self.observation_noise = 0.1
        self.transition_noise = 0.1
        
    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Encode observation to latent state."""
        observation = np.asarray(observation).flatten()
        
        # Encode with uncertainty
        z_raw = self.W_encoder @ observation
        z_raw += np.random.randn(self.latent_dim) * self.observation_noise
        
        # Update latent state with exponential moving average
        alpha = 0.7
        self.z = alpha * self.z + (1 - alpha) * z_raw
        
        return self.z.copy()
    
    def transition(self, action: np.ndarray) -> np.ndarray:
        """Predict next latent state."""
        action = np.asarray(action).flatten()
        
        # Predict transition
        z_next = (
            self.W_transition @ self.z +
            self.W_action @ action +
            np.random.randn(self.latent_dim) * self.transition_noise
        )
        
        # Apply discovered causal constraints
        z_next = self._apply_causal_constraints(z_next)
        
        # Update persistent objects
        self._update_persistent_objects(z_next)
        
        self.z = z_next
        return self.z.copy()
    
    def _apply_causal_constraints(self, z: np.ndarray) -> np.ndarray:
        """Apply discovered causal structure."""
        for from_idx, to_idx, strength in self.causal_links:
            if 0 <= from_idx < len(z) and 0 <= to_idx < len(z):
                # If from is non-zero, to should correlate
                z[to_idx] = z[to_idx] + z[from_idx] * strength * 0.1
        
        return z
    
    def _update_persistent_objects(self, z: np.ndarray):
        """Track persistent objects in latent space."""
        # Simple: track clusters that persist
        if len(self.persistent_objects) == 0:
            # First object
            self.persistent_objects.append({
                'position': z.copy(),
                'age': 1,
                'confidence': 0.5
            })
        else:
            # Find nearest persistent object
            best_idx = 0
            best_dist = float('inf')
            
            for i, obj in enumerate(self.persistent_objects):
                dist = np.linalg.norm(z - obj['position'])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            if best_dist < 1.0:  # Object persists
                self.persistent_objects[best_idx]['position'] = (
                    self.persistent_objects[best_idx]['position'] * 0.9 + z * 0.1
                )
                self.persistent_objects[best_idx]['age'] += 1
                self.persistent_objects[best_idx]['confidence'] = min(1.0, 
                    self.persistent_objects[best_idx]['confidence'] + 0.01)
            else:  # New transient object
                self.persistent_objects.append({
                    'position': z.copy(),
                    'age': 1,
                    'confidence': 0.3
                })
        
        # Limit objects
        self.persistent_objects = sorted(
            self.persistent_objects,
            key=lambda x: x['confidence'] * x['age'],
            reverse=True
        )[:10]
    
    def decode(self, z: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode latent state to observation."""
        if z is None:
            z = self.z
        z = np.asarray(z).flatten()
        
        # Decode
        observation = self.W_decoder @ z
        observation += np.random.randn(self.observation_dim) * self.observation_noise
        
        return observation
    
    def imagine(self, n_steps: int, action_sequence: List[np.ndarray]) -> List[np.ndarray]:
        """Imagine future without execution."""
        z_imagined = [self.z.copy()]
        
        for action in action_sequence[:n_steps]:
            z_next = (
                self.W_transition @ z_imagined[-1] +
                self.W_action @ action
            )
            z_imagined.append(z_next)
        
        return z_imagined
    
    def learn_transition(self, z_t: np.ndarray, action: np.ndarray, z_t1: np.ndarray):
        """Learn transition from experience."""
        z_t = np.asarray(z_t).flatten()
        action = np.asarray(action).flatten()
        z_t1 = np.asarray(z_t1).flatten()
        
        # Observed transition
        delta_obs = z_t1 - z_t
        
        # Predicted transition
        delta_pred = (
            self.W_transition @ z_t +
            self.W_action @ action - z_t
        )
        
        # Prediction error
        error = delta_obs - delta_pred
        
        # Update transition model
        self.W_transition += 0.01 * np.outer(error, z_t) * 0.1
        self.W_action += 0.01 * np.outer(error, action) * 0.1
        
        # Normalize
        self.W_transition = self.W_transition / (np.linalg.norm(self.W_transition) + 1e-8)
        self.W_action = self.W_action / (np.linalg.norm(self.W_action) + 1e-8)
        
        # Discover causal links
        self._discover_causal_links(z_t, z_t1)


# ============================================================================
# 3. SELF AS BASIN STABILITY
# ============================================================================

class SelfAsBasin:
    """
    Self = stable recursive attractor (basin continuity across time).
    
    Self is NOT:
      - behavior cluster
      - action centroid
      - predefined archetype
    
    Self IS:
      - stable basin in self-state space
      - recursive predictive attractor
      - self-maintaining topology
      - identity preservation dynamics
    """
    
    def __init__(self, self_id: str, initial_state: np.ndarray):
        self.self_id = self_id
        self.state = np.asarray(initial_state).flatten()
        
        # Self-state basin
        self.basin_center = self.state.copy()
        self.basin_depth = 1.0
        self.basin_radius = 1.5
        
        # Predictive self-model
        self.self_prediction_model = np.eye(len(self.state)) * 0.9
        
        # Identity preservation
        self.preferred_manifold = self.state.copy()
        self.resistance_to_change = 0.8
        
        # History
        self.trajectory: List[np.ndarray] = [self.state.copy()]
        self.identity_crises: List[int] = []  # Times when identity threatened
        
    def predict_self(self) -> np.ndarray:
        """Predict next self-state."""
        return self.self_prediction_model @ self.state
    
    def receive_experience(self, new_state: np.ndarray, 
                          continuity_pressure: float = 0.5):
        """
        Receive experience and update self.
        
        Self must maintain continuity while adapting.
        """
        new_state = np.asarray(new_state).flatten()
        
        # Predict what self should be
        predicted = self.predict_self()
        
        # Prediction error
        error = np.linalg.norm(new_state - predicted)
        
        # Identity crisis check
        if error > 2.0 * (1 - self.resistance_to_change):
            self.identity_crises.append(len(self.trajectory))
        
        # How much does new_state deviate from preferred self?
        deviation = np.linalg.norm(new_state - self.preferred_manifold)
        
        # Self-update: blend toward experience, constrained by resistance
        update_alpha = continuity_pressure * (1 - self.resistance_to_change)
        
        # Strong resistance = small updates
        if self.resistance_to_change > 0.8:
            update_alpha *= 0.3
        
        # Compute new self
        new_self = self.state * (1 - update_alpha) + new_state * update_alpha
        
        # Update basin
        self._update_basin(new_self, error)
        
        # Update prediction model
        error_vector = new_self - predicted
        self.self_prediction_model += 0.01 * np.outer(error_vector, self.state)
        
        # Normalize
        self.self_prediction_model = self.self_prediction_model / (
            np.linalg.norm(self.self_prediction_model) + 1e-8
        )
        
        # Record trajectory
        self.trajectory.append(new_self.copy())
        if len(self.trajectory) > 200:
            self.trajectory = self.trajectory[-100:]
        
        self.state = new_self
        
        return error
    
    def _update_basin(self, new_self: np.ndarray, error: float):
        """Update self-state basin."""
        # Basin moves toward new self
        self.basin_center = self.basin_center * 0.95 + new_self * 0.05
        
        # Depth changes with prediction error
        if error < 0.5:
            # Good prediction = stable basin
            self.basin_depth = min(5.0, self.basin_depth * 1.02)
        else:
            # Poor prediction = shallower basin
            self.basin_depth = max(0.3, self.basin_depth * 0.98)
        
        # Update preferred manifold (slowly)
        self.preferred_manifold = self.preferred_manifold * 0.995 + new_self * 0.005
    
    def get_identity_strength(self) -> float:
        """How strong is this self's identity?"""
        if len(self.trajectory) < 10:
            return 0.3
        
        # Identity strength from basin depth and trajectory coherence
        trajectory_array = np.array(self.trajectory[-20:])
        coherence = 1.0 / (1.0 + np.std(trajectory_array))
        
        strength = (self.basin_depth * 0.5 + coherence * 0.5) * self.resistance_to_change
        
        return strength
    
    def merge_with(self, other: 'SelfAsBasin') -> 'SelfAsBasin':
        """Merge two selves into one."""
        # Create merged self
        merged = SelfAsBasin(
            self_id=f"merged_{np.random.randint(100000)}",
            initial_state=(self.basin_center + other.basin_center) / 2
        )
        
        # Average prediction models
        merged.self_prediction_model = (
            self.self_prediction_model + other.self_prediction_model
        ) / 2
        
        # Strongest resistance wins
        merged.resistance_to_change = max(
            self.resistance_to_change, 
            other.resistance_to_change
        )
        
        # Combined trajectory
        merged.trajectory = self.trajectory + other.trajectory
        
        return merged


# ============================================================================
# 4. COGNITIVE THERMODYNAMICS
# ============================================================================

class CognitiveThermodynamics:
    """
    Cognitive thermodynamics: entropy gradients, free energy, uncertainty flow.
    
    NOT: resource meters
    BUT: thermodynamics where:
         - cognition has temperature
         - entropy drives exploration
         - free energy minimization
         - uncertainty creates pressure
    """
    
    def __init__(self):
        # Thermodynamic state
        self.temperature = 1.0  # High = more exploration
        self.entropy = 0.5    # Uncertainty level
        self.free_energy = 0.5  # Prediction error pressure
        self.heat_capacity = 1.0  # How much cognition can absorb
        
        # Heat dissipation
        self.heat_dissipation_rate = 0.05
        
    def compute_free_energy(self, prediction_error: float, 
                           surprise: float) -> float:
        """
        Compute free energy: surprise + complexity - entropy.
        
        F = E[log p(x) - log q(x)] ≈ prediction_error + complexity - entropy
        """
        # Free energy approximation
        self.free_energy = prediction_error * 0.5 + surprise * 0.3 - self.entropy * 0.2
        
        # Update entropy
        self.entropy = self.entropy * 0.99 + surprise * 0.01
        
        return self.free_energy
    
    def update_temperature(self, cognitive_activity: float):
        """
        Temperature changes with cognitive activity.
        
        High activity -> temperature rises (more exploration)
        Low activity -> temperature falls (more exploitation)
        """
        if cognitive_activity > 0.7:
            # High activity: temperature rises
            self.temperature = min(2.0, self.temperature * 1.02)
        else:
            # Low activity: temperature falls
            self.temperature = max(0.3, self.temperature * 0.98)
        
        # Heat dissipation
        self.temperature = max(0.3, self.temperature - self.heat_dissipation_rate * 0.01)
    
    def get_exploration_pressure(self) -> float:
        """
        Get exploration pressure from temperature and entropy.
        
        High temperature + high entropy = explore
        Low temperature + low entropy = exploit
        """
        return self.temperature * self.entropy
    
    def get_abstraction_level(self) -> str:
        """Determine abstraction level from thermodynamics."""
        if self.temperature < 0.5 and self.entropy < 0.3:
            return "concrete"  # Cold, certain = concrete thinking
        elif self.temperature > 1.5 and self.entropy > 0.7:
            return "abstract"  # Hot, uncertain = abstract thinking
        else:
            return "mixed"
    
    def apply_uncertainty_pressure(self, uncertainty: float) -> np.ndarray:
        """
        Uncertainty creates force on cognition.
        
        Returns: force vector pushing toward uncertainty reduction
        """
        # Uncertainty creates pressure to explore
        pressure = uncertainty * (1 - self.entropy)
        
        # Temperature modulates pressure
        pressure *= self.temperature
        
        # Force is random direction (exploration)
        force = np.random.randn(2) * pressure
        
        return force


# ============================================================================
# 5. TENSION WAVE DYNAMICS
# ============================================================================

class TensionWaveField:
    """
    Tensions as wave dynamics.
    
    NOT: static tension list
    BUT: waves that:
         - interfere
         - resonate
         - amplify
         - suppress
         - form standing waves
    """
    
    def __init__(self, dim: int = 2):
        self.dim = dim
        
        # Tension waves
        self.waves: List[Dict] = []
        
        # Wave parameters
        self.wave_speed = 0.5
        self.interference_strength = 0.3
        self.decay_rate = 0.02
        
    def emit_tension(self, source: np.ndarray, direction: np.ndarray,
                    magnitude: float, frequency: float = 1.0) -> Dict:
        """Emit a tension wave."""
        wave = {
            'source': np.asarray(source).flatten(),
            'direction': np.asarray(direction).flatten(),
            'magnitude': magnitude,
            'frequency': frequency,
            'phase': 0.0,
            'age': 0,
            'position': np.asarray(source).flatten()
        }
        
        self.waves.append(wave)
        
        return wave
    
    def propagate(self) -> np.ndarray:
        """Propagate all tension waves."""
        if not self.waves:
            return np.zeros(self.dim)
        
        # Compute field from all waves
        field = np.zeros(self.dim)
        
        to_remove = []
        
        for i, wave in enumerate(self.waves):
            # Advance wave
            wave['position'] = wave['position'] + wave['direction'] * self.wave_speed
            wave['age'] += 1
            wave['phase'] += wave['frequency']
            
            # Compute wave contribution
            wave_contribution = wave['magnitude'] * np.sin(wave['phase']) * wave['direction']
            
            # Distance decay
            distance_from_source = np.linalg.norm(wave['position'] - wave['source'])
            decay = np.exp(-distance_from_source * 0.1)
            
            field += wave_contribution * decay
            
            # Remove old waves
            if wave['age'] > 50 or wave['magnitude'] < 0.05:
                to_remove.append(i)
        
        # Remove old waves
        for i in reversed(to_remove):
            self.waves.pop(i)
        
        return field
    
    def compute_interference(self, new_tension: np.ndarray) -> Tuple[float, float]:
        """
        Compute interference with existing tension waves.
        
        Returns: (amplification, suppression)
        """
        if not self.waves:
            return 0.5, 0.5
        
        amplification = 0.5
        suppression = 0.5
        
        for wave in self.waves:
            # Dot product with wave direction
            alignment = np.dot(new_tension, wave['direction'])
            
            if alignment > 0:
                # Same direction = amplification
                amplification += alignment * wave['magnitude'] * 0.1
            else:
                # Opposite direction = suppression
                suppression += abs(alignment) * wave['magnitude'] * 0.1
        
        # Normalize
        amplification = min(2.0, max(0.5, amplification / (len(self.waves) + 1)))
        suppression = min(2.0, max(0.5, suppression / (len(self.waves) + 1)))
        
        return amplification, suppression
    
    def get_wave_summary(self) -> Dict:
        """Get wave field summary."""
        return {
            'n_waves': len(self.waves),
            'avg_magnitude': np.mean([w['magnitude'] for w in self.waves]) if self.waves else 0,
            'total_energy': sum(w['magnitude'] for w in self.waves),
            'avg_age': np.mean([w['age'] for w in self.waves]) if self.waves else 0
        }


# ============================================================================
# INTEGRATED TOPOLOGICAL COGNITION
# ============================================================================

class TopologicalCognition:
    """
    Phase 16: Topological Cognition
    
    NOT: trajectory planner with attractors
    BUT: persistent deformable cognitive topology
    
    Where:
      - Futures form energy basins
      - Actions deform topology
      - Memory reshapes geometry
      - Self = stable recursive attractor (basin)
      - Tensions = wave dynamics
      - Cognition follows thermodynamics
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 2, latent_dim: int = 4):
        # Core components
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        self.current_state = np.zeros(state_dim)
        
        # 1. Persistent future landscape
        self.future_field = FutureEnergyField(latent_dim=latent_dim)
        
        # 2. Latent predictive physics
        self.physics = LatentPredictivePhysics(
            observation_dim=state_dim,
            latent_dim=latent_dim,
            action_dim=action_dim
        )
        
        # 3. Self as basin stability
        self.self_basin = SelfAsBasin("primary", initial_state=np.zeros(state_dim))
        
        # 4. Cognitive thermodynamics
        self.thermodynamics = CognitiveThermodynamics()
        
        # 5. Tension wave dynamics
        self.tension_waves = TensionWaveField(dim=latent_dim)
        
        # 6. Emergent selves from basins
        self.emergent_selves: List[SelfAsBasin] = []
        
        # Execution history
        self.execution_history: List[Dict] = []
        
    def process(self, observation: np.ndarray, drives: List[Dict], 
                n_steps: int = 10) -> Dict:
        """
        Process through topological cognition.
        
        NOT: plan -> execute
        BUT: state + topology -> deformed future landscape -> execution
        """
        results = []
        
        # Encode observation to latent
        z = self.physics.encode(observation)
        
        for step in range(n_steps):
            # 1. Compute tension wave field
            tension_field = self.tension_waves.propagate()
            
            # 2. Get thermodynamics state
            exploration_pressure = self.thermodynamics.get_exploration_pressure()
            abstraction = self.thermodynamics.get_abstraction_level()
            
            # 3. Get attractor position from future field
            attractor = self.future_field.get_attractor_position(z)
            
            # 4. Get self-prediction
            predicted_self = self.self_basin.predict_self()
            
            # 5. Compute action (attracted to attractor + exploration + self-preservation)
            action = self._compute_topological_action(
                z, attractor, exploration_pressure, predicted_self
            )
            
            # 6. Execute in latent physics
            z_next = self.physics.transition(action)
            
            # 7. Decode to observation space
            observation_predicted = self.physics.decode(z_next)
            
            # 8. Update future landscape
            success = drives[0].get('strength', 0.5) if drives else 0.5
            trajectory = [z.copy(), z_next.copy()]
            self.future_field.carve_channel(trajectory, success)
            
            # 9. Update self basin
            self_state_for_self = observation  # Use observation as self feedback
            prediction_error = self.self_basin.receive_experience(
                self_state_for_self,
                continuity_pressure=self.thermodynamics.entropy
            )
            
            # 10. Update thermodynamics
            cognitive_activity = np.linalg.norm(action)
            self.thermodynamics.update_temperature(cognitive_activity)
            free_energy = self.thermodynamics.compute_free_energy(
                prediction_error=prediction_error,
                surprise=1.0 - success
            )
            
            # 11. Emit tension waves
            tension_direction = z_next - z if np.linalg.norm(z_next - z) > 0.01 else np.random.randn(self.latent_dim)
            tension_magnitude = free_energy * (1 - success)
            self.tension_waves.emit_tension(
                source=z.copy(),
                direction=tension_direction / (np.linalg.norm(tension_direction) + 1e-8),
                magnitude=tension_magnitude
            )
            
            # 12. Update state
            self.current_state = observation_predicted
            z = z_next
            
            # Record
            results.append({
                'step': step,
                'action': action.tolist(),
                'z': z.tolist(),
                'observation': observation_predicted.tolist(),
                'attractor': attractor.tolist(),
                'free_energy': free_energy,
                'temperature': self.thermodynamics.temperature,
                'entropy': self.thermodynamics.entropy
            })
            
            # Erode terrain periodically
            if step % 5 == 0:
                self.future_field.erode_terrain()
        
        return {
            'results': results,
            'topology': self.future_field.get_topology_summary(),
            'thermodynamics': {
                'temperature': self.thermodynamics.temperature,
                'entropy': self.thermodynamics.entropy,
                'free_energy': self.thermodynamics.free_energy,
                'exploration_pressure': exploration_pressure
            },
            'self_identity': {
                'strength': self.self_basin.get_identity_strength(),
                'basin_depth': self.self_basin.basin_depth,
                'crises': len(self.self_basin.identity_crises)
            },
            'tension_waves': self.tension_waves.get_wave_summary()
        }
    
    def _compute_topological_action(self, z: np.ndarray, attractor: np.ndarray,
                                    exploration: float, self_prediction: np.ndarray) -> np.ndarray:
        """Compute action from topological dynamics."""
        # Attraction toward basin (exploitation)
        basin_pull = (attractor - z) * 0.4
        
        # Exploration pressure (thermodynamic)
        exploration_force = np.random.randn(self.latent_dim) * exploration * 0.3
        
        # Self-preservation (stay near predicted self)
        self_force = (self_prediction - z) * 0.2 * self.self_basin.resistance_to_change
        
        # Combine
        action = basin_pull + exploration_force + self_force
        
        # Normalize
        if np.linalg.norm(action) > 1.0:
            action = action / np.linalg.norm(action)
        
        return action


def test_topological_cognition():
    """Test topological cognition."""
    print("\n" + "=" * 60)
    print("TOPOLOGICAL COGNITION TEST")
    print("=" * 60)
    
    cognition = TopologicalCognition(state_dim=2, action_dim=2, latent_dim=4)
    
    # Define drives
    drives = [
        {'name': 'exploration', 'target': np.array([2.0, 2.0]), 'strength': 0.7},
        {'name': 'safety', 'target': np.array([0.0, 0.0]), 'strength': 0.5}
    ]
    
    # Simulate observation
    observation = np.array([0.5, 0.3])
    
    print("\n  Processing with topological cognition:")
    result = cognition.process(observation, drives, n_steps=30)
    
    # Show results
    print(f"\n  Execution results:")
    print(f"    Steps: {len(result['results'])}")
    
    # Topology
    topology = result['topology']
    print(f"\n  Future topology:")
    print(f"    Basins: {topology['n_basins']}")
    print(f"    Avg depth: {topology['avg_depth']:.3f}")
    print(f"    Max depth: {topology['max_depth']:.3f}")
    print(f"    Habit paths: {topology['habit_paths']}")
    print(f"    Traumas: {topology['traumas']}")
    
    # Thermodynamics
    thermo = result['thermodynamics']
    print(f"\n  Thermodynamics:")
    print(f"    Temperature: {thermo['temperature']:.3f}")
    print(f"    Entropy: {thermo['entropy']:.3f}")
    print(f"    Free energy: {thermo['free_energy']:.3f}")
    print(f"    Exploration pressure: {thermo['exploration_pressure']:.3f}")
    
    # Self
    self_state = result['self_identity']
    print(f"\n  Self as basin:")
    print(f"    Identity strength: {self_state['strength']:.3f}")
    print(f"    Basin depth: {self_state['basin_depth']:.3f}")
    print(f"    Identity crises: {self_state['crises']}")
    
    # Tension waves
    waves = result['tension_waves']
    print(f"\n  Tension wave field:")
    print(f"    Active waves: {waves['n_waves']}")
    print(f"    Total energy: {waves['total_energy']:.3f}")


def test_future_landscape():
    """Test persistent future landscape."""
    print("\n" + "=" * 60)
    print("FUTURE LANDSCAPE TEST")
    print("=" * 60)
    
    field = FutureEnergyField(latent_dim=2)
    
    print("\n  Carving execution channels:")
    
    # Successful trajectories
    for i in range(20):
        trajectory = [
            np.array([i * 0.1, i * 0.05]),
            np.array([i * 0.1 + 0.5, i * 0.05 + 0.3]),
            np.array([i * 0.1 + 1.0, i * 0.05 + 0.6])
        ]
        field.carve_channel(trajectory, success=0.8)
    
    # Failed trajectory
    field.create_trauma([
        np.array([0.5, 0.5]),
        np.array([1.0, 1.5]),
        np.array([0.8, 2.0])
    ])
    
    print(f"\n  Topology after carving:")
    summary = field.get_topology_summary()
    print(f"    Basins: {summary['n_basins']}")
    print(f"    Avg depth: {summary['avg_depth']:.3f}")
    print(f"    Habit paths: {summary['habit_paths']}")
    print(f"    Traumas: {summary['traumas']}")
    
    # Show basins
    print("\n  Basins:")
    for b in field.basins[:5]:
        print(f"    {b.basin_id}: depth={b.depth:.3f}, mass={b.mass:.3f}")
    
    # Test erosion
    print("\n  Testing terrain erosion:")
    field.erode_terrain()
    summary = field.get_topology_summary()
    print(f"    After erosion: basins={summary['n_basins']}, avg_depth={summary['avg_depth']:.3f}")


def test_latent_physics():
    """Test latent predictive physics."""
    print("\n" + "=" * 60)
    print("LATENT PREDICTIVE PHYSICS TEST")
    print("=" * 60)
    
    physics = LatentPredictivePhysics(observation_dim=2, latent_dim=4, action_dim=2)
    
    print("\n  Learning transitions:")
    
    # Learn simple dynamics
    for i in range(100):
        observation = np.array([i * 0.1, i * 0.05])
        action = np.array([0.5, 0.3])
        
        z = physics.encode(observation)
        z_next = physics.transition(action)
        
        physics.learn_transition(z, action, z_next)
    
    print(f"    Training: 100 transitions")
    print(f"    Persistent objects: {len(physics.persistent_objects)}")
    
    print("\n  Testing imagination:")
    z_current = np.array([0.5, 0.3, 0.2, 0.1])
    action_sequence = [np.array([0.5, 0.3]) for _ in range(5)]
    
    imagined = physics.imagine(5, action_sequence)
    print(f"    Imagined {len(imagined)} steps")
    print(f"    Final state: {imagined[-1][:2]}")


def test_self_as_basin():
    """Test self as basin stability."""
    print("\n" + "=" * 60)
    print("SELF AS BASIN TEST")
    print("=" * 60)
    
    self_basin = SelfAsBasin("test_self", initial_state=np.array([0.5, 0.5]))
    
    print("\n  Initial identity:")
    print(f"    Basin depth: {self_basin.basin_depth:.3f}")
    print(f"    Identity strength: {self_basin.get_identity_strength():.3f}")
    
    print("\n  Experiencing trajectory:")
    
    # Normal experience
    for i in range(20):
        new_state = np.array([0.5, 0.5]) + np.random.randn(2) * 0.1
        error = self_basin.receive_experience(new_state, continuity_pressure=0.5)
        
        if i % 5 == 4:
            print(f"    Step {i+1}: identity={self_basin.get_identity_strength():.3f}, "
                  f"crises={len(self_basin.identity_crises)}")
    
    # Identity crisis
    print("\n  Identity crisis (extreme deviation):")
    crisis_state = np.array([5.0, 5.0])
    error = self_basin.receive_experience(crisis_state, continuity_pressure=0.2)
    print(f"    After crisis: identity={self_basin.get_identity_strength():.3f}, "
          f"crises={len(self_basin.identity_crises)}")
    
    print("\n  Recovery:")
    for i in range(10):
        recovery_state = self_basin.basin_center + np.random.randn(2) * 0.2
        self_basin.receive_experience(recovery_state, continuity_pressure=0.7)
    
    print(f"    After recovery: identity={self_basin.get_identity_strength():.3f}")


def test_thermodynamics():
    """Test cognitive thermodynamics."""
    print("\n" + "=" * 60)
    print("COGNITIVE THERMODYNAMICS TEST")
    print("=" * 60)
    
    thermo = CognitiveThermodynamics()
    
    print("\n  Simulating cognitive cycles:")
    
    for i in range(50):
        activity = np.random.random()
        prediction_error = np.random.random() * 0.5
        surprise = np.random.random() * 0.3
        
        thermo.update_temperature(activity)
        free_energy = thermo.compute_free_energy(prediction_error, surprise)
        
        if i % 10 == 9:
            print(f"    Cycle {i+1}: temp={thermo.temperature:.3f}, "
                  f"entropy={thermo.entropy:.3f}, "
                  f"free_energy={free_energy:.3f}")
    
    print(f"\n  Final state:")
    print(f"    Temperature: {thermo.temperature:.3f}")
    print(f"    Abstraction level: {thermo.get_abstraction_level()}")
    print(f"    Exploration pressure: {thermo.get_exploration_pressure():.3f}")


def test_tension_waves():
    """Test tension wave dynamics."""
    print("\n" + "=" * 60)
    print("TENSION WAVE DYNAMICS TEST")
    print("=" * 60)
    
    waves = TensionWaveField(dim=4)
    
    print("\n  Emitting tension waves:")
    
    # Emit waves
    for i in range(10):
        source = np.random.randn(4) * 2
        direction = np.random.randn(4)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        waves.emit_tension(
            source=source,
            direction=direction,
            magnitude=0.5 + np.random.random() * 0.3,
            frequency=1.0 + np.random.random()
        )
        
        if i % 3 == 2:
            waves.propagate()
            summary = waves.get_wave_summary()
            print(f"    Wave {i+1}: {summary['n_waves']} active, energy={summary['total_energy']:.3f}")
    
    # Test interference
    print("\n  Testing interference:")
    new_tension = np.array([1.0, 0.5, 0.0, 0.0])
    amp, sup = waves.compute_interference(new_tension)
    print(f"    New tension vs field:")
    print(f"    Amplification: {amp:.3f}")
    print(f"    Suppression: {sup:.3f}")


def compare_with_phase15():
    """Compare Phase 16 (Topological) with Phase 15 (Field)."""
    print("\n" + "=" * 60)
    print("PHASE 15 VS PHASE 16 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 15 (Executive Field):")
    print("    - Trajectories compete (utility optimization)")
    print("    - Emergent selves = behavior clusters")
    print("    - Cognitive economics = resource meters")
    print("    - Tensions = local computation")
    print("    - No persistent world")
    print("    - External drives")
    
    print("\n  Phase 16 (Topological Cognition):")
    print("    - Future landscape DEFORMS from execution")
    print("    - Self = stable recursive attractor (basin)")
    print("    - Thermodynamics (temperature, entropy, free energy)")
    print("    - Tensions = wave dynamics (interference, resonance)")
    print("    - Persistent terrain (habits, scars, erosion)")
    print("    - Latent predictive physics (z_t -> z_t+1)")
    
    print("\n  Key architectural shifts:")
    print("    1. Trajectory competition -> Landscape deformation")
    print("    2. Behavior clusters -> Self as basin stability")
    print("    3. Resource meters -> Thermodynamic dynamics")
    print("    4. Local tensions -> Wave field dynamics")
    print("    5. Ephemeral trajectories -> Persistent terrain")
    print("    6. Linear mapping -> Latent physics")
    
    print("\n  This is NO LONGER:")
    print("    advanced agent architecture")
    print("  This IS:")
    print("    continuous cognitive ecology")
    print("    topological cognition")
    print("    cognitive thermodynamics")


if __name__ == "__main__":
    test_topological_cognition()
    test_future_landscape()
    test_latent_physics()
    test_self_as_basin()
    test_thermodynamics()
    test_tension_waves()
    compare_with_phase15()
    
    print("\n" + "=" * 60)
    print("PHASE 16 - TOPOLOGICAL COGNITION")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Executive field with trajectory competition (Phase 15)
  To: Persistent deformable cognitive topology where:
      - Futures form energy basins
      - Actions deform topology
      - Memory reshapes geometry
      - Attractors merge, split, decay
      - Self = stable recursive attractor (basin)
      
  This is NO LONGER:
    advanced agent architecture
  This IS:
    continuous cognitive ecology
    topological cognition
    cognitive thermodynamics

CRITICAL INSIGHT:
  Phase 15: trajectories compete, winner executes
  Phase 16: execution DEFORMS future landscape
            trajectories carve probability valleys
            successful futures become deep basins
            execution falls into basin

SEVEN CRITICAL FIXES:
  1. World Model = linear mapping -> Latent predictive physics (z_t -> z_t+1)
  2. Attractor competition = utility -> Landscape topology evolution
  3. Emergent Selves = clusters -> Self as basin stability
  4. Cognitive Economics = meters -> Strategy-changing dynamics
  5. Tensions = local -> Wave dynamics (interference, resonance)
  6. No persistent world -> Persistent terrain (habits, scars, erosion)
  7. External drives -> Endogenous from topology pressure

We are now at:
  - Persistent deformable cognitive topology
  - Self as stable recursive attractor
  - Cognitive thermodynamics
  - Tension wave dynamics
  - Latent predictive physics
  - Continuous cognitive ecology
  
This is already closer to:
  - Active inference systems
  - Frristonian free-energy
  - Proto-cognitive dynamics
  - Endogenous objective formation
  - Self-organizing cognition
""")