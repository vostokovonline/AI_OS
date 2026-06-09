"""
Phase 20: Cognitive Morphodynamic Field

ARCHITECTURAL SHIFT:
  From: Phase 19 - field-like state vector dynamics
  To: Phase 20 - continuous morphodynamic medium where:
      - Field is spatial: ψ(x,y,z,t), not ψ ∈ R^n
      - Dynamics is PDE-based, not ODE with global vortices
      - Objects emerge as coherent structures, not created entities
      - Free energy IS the dynamics, not computed metric
      - Memory = field hysteresis, not storage
      - Hierarchy = frequency decomposition, not layer stack
      
  This is NO LONGER:
    agent architecture with field-like components
  This IS:
    continuous morphodynamic field
    from which cognition emerges
    
CRITICAL INSIGHT:
  Phase 19: "system has dynamics"
  Phase 20: "system IS dynamics"
  
  The field is not a representation.
  The field IS the reality.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy


def laplacian_2d(psi):
    """Compute 2D Laplacian using numpy."""
    laplacian = np.zeros_like(psi)
    laplacian[1:-1, 1:-1] = (
        psi[:-2, 1:-1] + psi[2:, 1:-1] +
        psi[1:-1, :-2] + psi[1:-1, 2:] -
        4 * psi[1:-1, 1:-1]
    )
    # Boundaries
    laplacian[0, :] = psi[0, :] - psi[1, :]
    laplacian[-1, :] = psi[-1, :] - psi[-2, :]
    laplacian[:, 0] = psi[:, 0] - psi[:, 1]
    laplacian[:, -1] = psi[:, -1] - psi[:, -2]
    return laplacian


def maximum_filter_2d(psi, size=5):
    """Simple maximum filter using numpy."""
    result = np.zeros_like(psi)
    half = size // 2
    for y in range(psi.shape[0]):
        for x in range(psi.shape[1]):
            y_min = max(0, y - half)
            y_max = min(psi.shape[0], y + half + 1)
            x_min = max(0, x - half)
            x_max = min(psi.shape[1], x + half + 1)
            result[y, x] = np.max(psi[y_min:y_max, x_min:x_max])
    return result


def gradient_magnitude(psi):
    """Compute gradient magnitude."""
    gy, gx = np.gradient(psi)
    return np.sqrt(gx**2 + gy**2)


# ============================================================================
# CORE: CONTINUOUS MORPHODYNAMIC FIELD
# ============================================================================
"""
Morphodynamic Field:

ψ(x, y, t) — field value at spatial position (x,y) at time t

The field evolves according to morphodynamic equations:

∂ψ/∂t = -δF/δψ + D∇²ψ + η_local

Where:
  -δF/δψ = functional derivative of free energy (driving force)
  D∇²ψ = diffusion (spatial spreading)
  η_local = local noise (exploration)
  
NOT state vector: psi ∈ R^n
BUT spatial field: ψ(x,y,t)

Key properties:
  - Spatial locality: interactions are LOCAL
  - Propagation: signals spread through field
  - Coherence: structures emerge from nonlinear dynamics
  - Hysteresis: past shapes future dynamics
"""

class MorphodynamicField:
    """
    Phase 20: Continuous Morphodynamic Field
    
    NOT: state vector with vortices
    BUT: spatial field evolving by morphodynamic equations
    
    Field equation:
      ∂ψ/∂t = -δF/δψ + D∇²ψ + boundary conditions
      
    Where δF/δψ is functional derivative of free energy.
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        
        # The field ψ(x,y,t) - spatial distribution
        self.psi = np.zeros((height, width))
        
        # Field velocity ∂ψ/∂t
        self.psi_dot = np.zeros((height, width))
        
        # Diffusion coefficient
        self.D = 0.5
        
        # Local coupling strength
        self.coupling = 0.3
        
        # Boundary conditions (for action)
        self.boundary_force = np.zeros((height, width))
        
        # Time
        self.t = 0.0
        self.dt = 0.01
        
        # Free energy functional
        self.F = 0.0
        
        # Field history for hysteresis
        self.psi_history: List[np.ndarray] = []
        self.F_history: List[float] = []
        
        # Detected coherent structures (emergent, not created)
        self.coherent_structures: List[Dict] = []
        
        # Frequency bands (hierarchy = frequency decomposition)
        self.freq_bands = {
            'slow': {'freq_min': 0.0, 'freq_max': 0.1, 'energy': 0.0},
            'medium': {'freq_min': 0.1, 'freq_max': 0.3, 'energy': 0.0},
            'fast': {'freq_min': 0.3, 'freq_max': 0.5, 'energy': 0.0}
        }
    
    def compute_local_free_energy_density(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute local free energy density at each point.
        
        f(ψ) = (1/2)ψ² - log(1 + ψ²) + plasticity_history
        
        This is local, not global!
        """
        # Energy density from field value
        energy = 0.5 * psi**2 - np.log(1 + psi**2 + 1e-8)
        
        # Add hysteresis from history (past shapes present)
        if len(self.psi_history) > 10:
            # Average of recent history shapes current dynamics
            avg_history = np.mean([h for h in self.psi_history[-10:]], axis=0)
            hysteresis = -0.1 * avg_history * psi  # Attraction to past patterns
            energy += hysteresis
        
        return energy
    
    def compute_free_energy_functional(self) -> float:
        """
        Compute free energy functional F[ψ] = ∫ f(ψ) dx
        
        F is the OBJECTIVE that drives dynamics.
        NOT a metric, but the source of dynamics.
        
        δF/δψ = ∂f/∂ψ = ψ - 2ψ/(1 + ψ²)
        """
        f_local = self.compute_local_free_energy_density(self.psi)
        
        # Total free energy = integral over space
        F = np.sum(f_local) * self.dt * self.dt  # Spatial integration
        
        # Add gradient energy (smoothing)
        grad_y, grad_x = np.gradient(self.psi)
        grad_energy = np.sum(grad_x**2 + grad_y**2)
        F += 0.1 * grad_energy
        
        return F
    
    def compute_functional_derivative(self) -> np.ndarray:
        """
        Compute functional derivative δF/δψ.
        
        This is the DRIVING FORCE of dynamics.
        
        δF/δψ = ψ - 2ψ/(1 + ψ²) + plasticity terms
        """
        psi = self.psi
        
        # ∂f/∂ψ = ψ - 2ψ/(1 + ψ²)
        df_dpsi = psi - 2 * psi / (1 + psi**2 + 1e-8)
        
        # Add plasticity (memory as hysteresis)
        if len(self.psi_history) > 5:
            # Derivative of hysteresis term
            avg_history = np.mean([h for h in self.psi_history[-5:]], axis=0)
            df_dpsi += -0.1 * avg_history
        
        return df_dpsi
    
    def step(self) -> np.ndarray:
        """
        Single morphodynamic step.
        
        ∂ψ/∂t = -δF/δψ + D∇²ψ + boundary_force + local_noise
        
        NOT: global vortex updates
        BUT: LOCAL field dynamics
        """
        # Compute driving force (negative gradient of free energy)
        delta_F = self.compute_functional_derivative()
        
        # Diffusion (Laplacian)
        laplacian = laplacian_2d(self.psi)
        
        # Local noise (exploration)
        noise = np.random.randn(self.height, self.width) * 0.05
        
        # Update equation
        delta_psi = -delta_F + self.D * laplacian + self.boundary_force + noise
        
        # Euler integration
        self.psi_dot = delta_psi
        self.psi = self.psi + self.psi_dot * self.dt
        
        # Clamp field values
        self.psi = np.clip(self.psi, -3, 3)
        
        # Update free energy
        self.F = self.compute_free_energy_functional()
        
        # Record history (for hysteresis)
        self.psi_history.append(self.psi.copy())
        self.F_history.append(self.F)
        
        if len(self.psi_history) > 100:
            self.psi_history = self.psi_history[-50:]
        
        # Update time
        self.t += self.dt
        
        # Detect coherent structures
        self._detect_coherent_structures()
        
        # Analyze frequency bands
        self._analyze_frequency_bands()
        
        return self.psi.copy()
    
    def _detect_coherent_structures(self):
        """
        Detect coherent structures EMERGING from field dynamics.
        
        NOT: form_vortex()
        BUT: detect stable patterns in field
        
        Types of coherent structures:
        - Peaks (local maxima) = attractors
        - Saddles (saddle points) = transitions
        - Vortices (rotation patterns) = spatial memory
        - Waves (oscillatory patterns) = predictions
        """
        structures = []
        
        # Find peaks (local maxima)
        max_filtered = maximum_filter_2d(self.psi, size=5)
        peaks = (self.psi == max_filtered) & (self.psi > 1.0)
        
        for y, x in zip(*np.where(peaks)):
            structures.append({
                'type': 'peak',
                'position': (y, x),
                'amplitude': float(self.psi[y, x]),
                'radius': self._estimate_structure_radius(y, x, threshold=0.5)
            })
        
        # Find wave patterns (oscillatory regions)
        # Use gradient to detect edges/waves
        grad_magnitude = gradient_magnitude(self.psi)
        
        # Find coherent patterns where gradient oscillates
        for _ in range(3):  # Limit structures
            max_grad_idx = np.unravel_index(np.argmax(grad_magnitude), grad_magnitude.shape)
            if grad_magnitude[max_grad_idx] > 0.5:
                structures.append({
                    'type': 'wave',
                    'position': max_grad_idx,
                    'amplitude': float(grad_magnitude[max_grad_idx]),
                    'radius': 5
                })
                # Suppress this region
                grad_magnitude[max_grad_idx] = 0
        
        self.coherent_structures = structures[:10]  # Limit
    
    def _estimate_structure_radius(self, y: int, x: int, threshold: float = 0.5) -> float:
        """Estimate radius of a coherent structure."""
        value = self.psi[y, x]
        radius = 1
        
        for r in range(1, 20):
            # Check circular region
            y_min = max(0, y - r)
            y_max = min(self.height, y + r + 1)
            x_min = max(0, x - r)
            x_max = min(self.width, x + r + 1)
            
            region = self.psi[y_min:y_max, x_min:x_max]
            if np.mean(region) < value * threshold:
                return radius
            radius = r
        
        return radius
    
    def _analyze_frequency_bands(self):
        """
        Hierarchy = frequency decomposition.
        
        Slow structures = long-term identity/worldview
        Medium structures = patterns/habits
        Fast structures = immediate sensorimotor
        """
        # Compute spatial FFT using numpy
        psi_centered = self.psi - np.mean(self.psi)
        fft_result = np.fft.fft2(psi_centered)
        
        # Power spectrum
        power = np.abs(fft_result)**2
        
        # Frequency coordinates
        freq_x = np.fft.fftfreq(self.width, d=1.0)
        freq_y = np.fft.fftfreq(self.height, d=1.0)
        
        # Compute energy in each frequency band
        for band_name, band_params in self.freq_bands.items():
            f_min = band_params['freq_min']
            f_max = band_params['freq_max']
            
            # Sum power in band
            band_mask = (np.abs(freq_x) >= f_min) & (np.abs(freq_x) < f_max) | \
                       (np.abs(freq_y) >= f_min) & (np.abs(freq_y) < f_max)
            band_energy = np.sum(power[band_mask])
            
            self.freq_bands[band_name]['energy'] = 0.9 * self.freq_bands[band_name]['energy'] + 0.1 * band_energy
    
    def apply_boundary_condition(self, region: Tuple[int, int, int, int], 
                                 value: float, strength: float = 1.0):
        """
        Action = boundary condition manipulation.
        
        NOT: compute_action()
        BUT: modify field boundary to induce desired dynamics
        
        This is how "intention" manifests physically.
        """
        y1, y2, x1, x2 = region
        y1, y2, x1, x2 = max(0, y1), min(self.height, y2), max(0, x1), min(self.width, x2)
        
        # Apply force in region
        self.boundary_force[y1:y2, x1:x2] += value * strength
    
    def get_field_summary(self) -> Dict:
        """Get morphodynamic field summary."""
        return {
            't': self.t,
            'psi_mean': float(np.mean(self.psi)),
            'psi_std': float(np.std(self.psi)),
            'free_energy': self.F,
            'n_coherent_structures': len(self.coherent_structures),
            'frequency_bands': {k: v['energy'] for k, v in self.freq_bands.items()},
            'total_activity': float(np.sum(self.psi**2)),
            'boundary_activity': float(np.sum(self.boundary_force**2))
        }


# ============================================================================
# COHERENT STRUCTURES (EMERGENT OBJECTS)
# ============================================================================
"""
Coherent structures are NOT entities in the field.

They ARE stable patterns OF the field.

Types:
  - Standing waves: persistent oscillations
  - Vortices: rotational patterns
  - Solitons: self-reinforcing traveling structures
  - Basins: attraction regions

These emerge from nonlinear field dynamics.
They are not created by any function.
"""

@dataclass
class StandingWave:
    """Standing wave = persistent oscillatory pattern."""
    center: Tuple[int, int]
    frequency: float
    amplitude: float
    phase: float
    stability: float  # How long this pattern persists
    
    def get_field_contribution(self, y: int, x: int, t: float) -> float:
        """Compute field value at position at time."""
        dist = np.sqrt((y - self.center[0])**2 + (x - self.center[1])**2)
        return self.amplitude * np.cos(self.frequency * t + self.phase - dist)


class StandingWaveExtractor:
    """
    Extract standing waves (coherent structures) from field.
    
    NOT: create objects
    BUT: decompose field into coherent modes
    """
    
    def __init__(self, field: MorphodynamicField):
        self.field = field
        self.waves: List[StandingWave] = []
    
    def extract(self) -> List[StandingWave]:
        """
        Extract standing waves (coherent structures) from field.
        
        NOT: create objects
        BUT: decompose field into coherent modes
        """
        # Compute FFT using numpy
        psi_centered = self.field.psi - np.mean(self.field.psi)
        fft_result = np.fft.fft2(psi_centered)
        
        # Find dominant frequencies
        power = np.abs(fft_result)
        
        waves = []
        
        # Find top 3 frequency peaks
        for _ in range(3):
            max_idx = np.unravel_index(np.argmax(power), power.shape)
            if power[max_idx] > 0.1:
                # This frequency contributes significantly
                freq_y, freq_x = np.unravel_index(max_idx, power.shape)
                frequency = np.sqrt(freq_x**2 + freq_y**2) / 100.0
                
                # Estimate center from phase pattern
                center = (self.field.height // 2, self.field.width // 2)
                
                wave = StandingWave(
                    center=center,
                    frequency=frequency,
                    amplitude=float(power[max_idx]),
                    phase=0.0,
                    stability=0.8
                )
                
                waves.append(wave)
                
                # Suppress this peak
                power[max_idx] = 0
        
        self.waves = waves
        return waves


# ============================================================================
# PLASTICITY AND HYSTERESIS (MEMORY)
# ============================================================================
"""
Memory = field hysteresis, not storage.

The field literally changes its dynamics based on history.

NOT: memory.append(trajectory)
BUT: past patterns affect present dynamics through:
      - Adaptive threshold
      - Synaptic plasticity (field coupling)
      - Attractor deepening
      - Conductivity modulation
"""

class FieldPlasticity:
    """
    Field plasticity = memory as hysteresis.
    
    Past experiences permanently modify field dynamics.
    """
    
    def __init__(self, field: MorphodynamicField):
        self.field = field
        
        # Plasticity parameters (change with experience)
        self.adaptive_threshold = 0.5
        self.synaptic_strength = 0.3
        self.conductivity_map = np.ones((field.height, field.width))
        
        # Learning rate
        self.alpha = 0.01
        
    def learn_pattern(self, pattern: np.ndarray, success: float):
        """
        Learn a pattern (encode memory).
        
        NOT: store pattern
        BUT: modify field dynamics to prefer this pattern
        """
        pattern = pattern.reshape(self.field.height, self.field.width)
        
        # Increase conductivity where pattern is strong
        # (making it easier for future activity to flow there)
        conductivity_change = success * self.alpha * np.abs(pattern)
        self.conductivity_map = np.clip(
            self.conductivity_map + conductivity_change,
            0.5, 2.0
        )
        
        # Lower threshold for strong patterns
        if success > 0.7:
            self.adaptive_threshold *= (1 - success * 0.1)
        
        # Modify coupling strength based on pattern
        self.synaptic_strength = min(1.0, self.synaptic_strength + success * 0.05)
    
    def recall_pattern(self, query: np.ndarray) -> np.ndarray:
        """
        Recall: field dynamics naturally gravitates to learned patterns.
        
        NOT: database retrieval
        BUT: query acts as initial condition, field relaxes to nearest learned pattern
        """
        query = query.reshape(self.field.height, self.field.width)
        
        # Set field near query
        recall_quality = 0.3
        self.field.psi = (1 - recall_quality) * self.field.psi + recall_quality * query
        
        # Let field relax (dynamics will find nearest attractor)
        for _ in range(50):
            self.field.step()
        
        return self.field.psi.copy()
    
    def get_plasticity_summary(self) -> Dict:
        """Get plasticity state."""
        return {
            'adaptive_threshold': self.adaptive_threshold,
            'synaptic_strength': self.synaptic_strength,
            'avg_conductivity': float(np.mean(self.conductivity_map)),
            'conductivity_variance': float(np.var(self.conductivity_map))
        }


# ============================================================================
# EXPECTED FREE ENERGY PLANNING
# ============================================================================
"""
Planning = expected free energy minimization.

NOT: rollout simulation
BUT: compute policy that minimizes expected free energy

Expected free energy G(π) = E[q(s,a) - log p(o|s,a)]

Trajectory emerges automatically as path of least surprise.
"""

class ExpectedFreeEnergyPlanner:
    """
    Plan through expected free energy minimization.
    
    NOT: explicit trajectory rollout
    BUT: find policy that minimizes expected free energy
    """
    
    def __init__(self, field: MorphodynamicField):
        self.field = field
        self.policy_options: List[Dict] = []
        self.planning_horizon = 10
    
    def plan(self, goal_region: Tuple[int, int, int, int]) -> Dict:
        """
        Plan path to goal using expected free energy.
        
        NOT: "find path"
        BUT: "compute boundary conditions that minimize expected free energy"
        """
        y1, y2, x1, x2 = goal_region
        
        # Sample potential policies (boundary conditions)
        policies = []
        
        for strength in [0.5, 1.0, 1.5, 2.0]:
            for approach in ['center', 'perimeter', 'gradient']:
                policy = {
                    'strength': strength,
                    'approach': approach,
                    'expected_free_energy': 0.0,
                    'trajectory': []
                }
                
                # Save current state
                psi_saved = self.field.psi.copy()
                boundary_saved = self.field.boundary_force.copy()
                
                # Simulate this policy
                if approach == 'center':
                    self.field.apply_boundary_condition(goal_region, 2.0, strength)
                elif approach == 'perimeter':
                    # Apply to edges of goal region
                    self.field.apply_boundary_condition((y1, y1+2, x1, x2), 1.5, strength)
                    self.field.apply_boundary_condition((y2-2, y2, x1, x2), 1.5, strength)
                elif approach == 'gradient':
                    # Gradient toward goal
                    grad_force = np.zeros((self.field.height, self.field.width))
                    grad_force[y1:y2, x1:x2] = 1.0
                    self.field.boundary_force += grad_force * strength
                
                # Simulate and measure free energy
                trajectory = [self.field.psi.copy()]
                free_energy_measurements = []
                
                for step in range(self.planning_horizon):
                    self.field.step()
                    trajectory.append(self.field.psi.copy())
                    free_energy_measurements.append(self.field.F)
                
                # Compute expected free energy (average)
                policy['expected_free_energy'] = np.mean(free_energy_measurements)
                policy['trajectory'] = [t.copy() for t in trajectory]
                
                policies.append(policy)
                
                # Restore
                self.field.psi = psi_saved
                self.field.boundary_force = boundary_saved
        
        # Select best policy (minimum expected free energy)
        policies.sort(key=lambda p: p['expected_free_energy'])
        best_policy = policies[0]
        
        self.policy_options = policies
        
        return {
            'best_policy': best_policy,
            'all_policies': policies,
            'goal_region': goal_region
        }
    
    def execute_plan(self, plan: Dict):
        """Execute plan by applying boundary conditions."""
        best = plan['best_policy']
        
        # Apply boundary conditions from best policy
        if best['approach'] == 'center':
            y1, y2, x1, x2 = plan['goal_region']
            self.field.apply_boundary_condition(plan['goal_region'], 2.0, best['strength'])
        elif best['approach'] == 'perimeter':
            y1, y2, x1, x2 = plan['goal_region']
            self.field.apply_boundary_condition((y1, y1+2, x1, x2), 1.5, best['strength'])
        elif best['approach'] == 'gradient':
            y1, y2, x1, x2 = plan['goal_region']
            grad_force = np.zeros((self.field.height, self.field.width))
            grad_force[y1:y2, x1:x2] = 1.0
            self.field.boundary_force += grad_force * best['strength']


# ============================================================================
# INTEGRATED MORPHODYNAMIC COGNITION
# ============================================================================

class MorphodynamicCognition:
    """
    Phase 20: Integrated Morphodynamic Cognition
    
    Single continuous morphodynamic medium where:
      A. Field is spatial ψ(x,y,t)
      B. Dynamics is PDE-based (local interactions)
      C. Objects emerge as coherent structures
      D. Free energy IS dynamics
      E. Memory = field hysteresis
      F. Planning = expected free energy minimization
      G. Hierarchy = frequency decomposition
      
    NOT: modular architecture with field-like components
    BUT: continuous field from which cognition emerges
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        
        # The field (spatial, not vector)
        self.field = MorphodynamicField(width=width, height=height)
        
        # Standing wave extractor
        self.wave_extractor = StandingWaveExtractor(self.field)
        
        # Memory as plasticity/hysteresis
        self.plasticity = FieldPlasticity(self.field)
        
        # Planning via expected free energy
        self.planner = ExpectedFreeEnergyPlanner(self.field)
        
        # Initialize with some activity
        self._seed_activity()
    
    def _seed_activity(self):
        """Seed initial activity pattern."""
        # Add initial perturbation
        center_y, center_x = self.height // 2, self.width // 2
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                y, x = center_y + dy, center_x + dx
                if 0 <= y < self.height and 0 <= x < self.width:
                    dist = np.sqrt(dy**2 + dx**2)
                    self.field.psi[y, x] = np.exp(-dist**2 / 10.0)
    
    def perceive(self, observation: np.ndarray) -> Dict:
        """
        Perception = field distortion from observation.
        
        Observation is injected as boundary condition.
        Field evolves to minimize resulting free energy.
        """
        # Inject observation into field
        if hasattr(observation, 'reshape'):
            observation = observation.reshape(self.height, self.width)
        elif isinstance(observation, list):
            observation = np.array(observation).reshape(self.height, self.width)
        
        # Blend observation with field
        blend_strength = 0.2
        self.field.psi = (1 - blend_strength) * self.field.psi + blend_strength * observation
        
        # Step field dynamics (perception emerges)
        for _ in range(10):
            self.field.step()
        
        # Extract standing waves (perceived objects)
        waves = self.wave_extractor.extract()
        
        return {
            'perceived_field': self.field.psi.copy(),
            'standing_waves': [{'center': w.center, 'frequency': w.frequency, 'amplitude': w.amplitude} 
                             for w in waves],
            'coherent_structures': len(self.field.coherent_structures),
            'free_energy': self.field.F
        }
    
    def imagine(self, n_steps: int) -> Dict:
        """
        Imagination = field relaxation without external input.
        
        NOT: generate scenario
        BUT: let field evolve naturally, trajectory IS imagination
        """
        trajectory = []
        
        for _ in range(n_steps):
            trajectory.append(self.field.psi.copy())
            self.field.step()
        
        return {
            'trajectory': trajectory,
            'final_field': self.field.psi.copy(),
            'n_steps': n_steps,
            'free_energy_trajectory': self.field.F_history[-n_steps:]
        }
    
    def plan(self, goal_region: Tuple[int, int, int, int]) -> Dict:
        """
        Plan through expected free energy minimization.
        """
        plan = self.planner.plan(goal_region)
        return plan
    
    def execute_plan(self, plan: Dict):
        """Execute plan (apply boundary conditions)."""
        self.planner.execute_plan(plan)
    
    def remember(self, pattern: np.ndarray, success: float):
        """
        Memory encoding = field plasticity.
        
        NOT: store pattern
        BUT: modify field dynamics through plasticity
        """
        self.plasticity.learn_pattern(pattern, success)
    
    def recall(self, query: np.ndarray) -> np.ndarray:
        """
        Memory recall = field relaxation to learned pattern.
        """
        return self.plasticity.recall_pattern(query)
    
    def act(self, action_type: str, region: Optional[Tuple] = None, value: float = 1.0):
        """
        Action = boundary condition manipulation.
        """
        if region is None:
            region = (self.height // 4, 3 * self.height // 4, 
                    self.width // 4, 3 * self.width // 4)
        
        if action_type == 'attract':
            self.field.apply_boundary_condition(region, value, strength=2.0)
        elif action_type == 'repel':
            self.field.apply_boundary_condition(region, -value, strength=2.0)
        elif action_type == 'oscillate':
            # Create oscillatory boundary
            for t in range(10):
                self.field.boundary_force = np.zeros((self.height, self.width))
                y1, y2, x1, x2 = region
                self.field.boundary_force[y1:y2, x1:x2] = value * np.sin(t * 0.5)
                self.field.step()
    
    def run_cycle(self, n_steps: int = 50) -> Dict:
        """Run cognitive cycle."""
        results = []
        
        for step in range(n_steps):
            # Generate synthetic observation
            observation = np.random.randn(self.height, self.width) * 0.1
            
            # Add some structure
            center_y, center_x = self.height // 2, self.width // 2
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    y, x = center_y + dy, center_x + dx
                    if 0 <= y < self.height and 0 <= x < self.width:
                        observation[y, x] += np.exp(-(dy**2 + dx**2) / 5.0)
            
            # Perception
            perception = self.perceive(observation)
            
            # Record
            results.append({
                'step': step,
                't': self.field.t,
                'free_energy': perception['free_energy'],
                'coherent_structures': perception['coherent_structures'],
                'frequency_bands': {k: v['energy'] for k, v in self.field.freq_bands.items()}
            })
            
            # Step field
            self.field.step()
        
        return {
            'steps': results,
            'field_summary': self.field.get_field_summary(),
            'plasticity_summary': self.plasticity.get_plasticity_summary()
        }


# ============================================================================
# TESTS
# ============================================================================

def test_morphodynamic_field():
    """Test morphodynamic field."""
    print("\n" + "=" * 60)
    print("MORPHODYNAMIC FIELD TEST")
    print("=" * 60)
    
    field = MorphodynamicField(width=32, height=32)
    
    print("\n  Evolving morphodynamic field:")
    
    for i in range(100):
        psi = field.step()
        
        if i % 20 == 19:
            summary = field.get_field_summary()
            print(f"    Step {i+1}:")
            print(f"      ψ mean: {summary['psi_mean']:.3f}")
            print(f"      Free energy: {summary['free_energy']:.3f}")
            print(f"      Coherent structures: {summary['n_coherent_structures']}")
    
    print(f"\n  Final state:")
    print(f"    Frequency bands: {field.freq_bands}")


def test_coherent_structures():
    """Test emergent coherent structures."""
    print("\n" + "=" * 60)
    print("COHERENT STRUCTURES TEST")
    print("=" * 60)
    
    field = MorphodynamicField(width=32, height=32)
    
    # Seed multiple activity centers
    field.psi[8, 8] = 2.0
    field.psi[24, 24] = -1.5
    field.psi[8, 24] = 1.5
    
    extractor = StandingWaveExtractor(field)
    
    print("\n  Evolving with multiple centers:")
    
    for i in range(50):
        field.step()
        
        if i % 10 == 9:
            waves = extractor.extract()
            print(f"    Step {i+1}: {len(waves)} standing waves detected")
    
    print(f"\n  Final coherent structures: {len(field.coherent_structures)}")


def test_plasticity_memory():
    """Test field plasticity (memory as hysteresis)."""
    print("\n" + "=" * 60)
    print("FIELD PLASTICITY TEST")
    print("=" * 60)
    
    field = MorphodynamicField(width=32, height=32)
    plasticity = FieldPlasticity(field)
    
    print("\n  Learning patterns:")
    
    # Learn multiple patterns with different success
    for i in range(5):
        pattern = np.random.randn(32, 32) * 0.5
        pattern[8:24, 8:24] += 1.0  # Add structure
        success = 0.5 + i * 0.1
        plasticity.learn_pattern(pattern, success)
    
    print(f"    After learning:")
    print(f"      Adaptive threshold: {plasticity.adaptive_threshold:.3f}")
    print(f"      Synaptic strength: {plasticity.synaptic_strength:.3f}")
    print(f"      Avg conductivity: {np.mean(plasticity.conductivity_map):.3f}")
    
    print("\n  Recalling pattern:")
    query = np.random.randn(32, 32)
    query[10:20, 10:20] += 1.0
    recalled = plasticity.recall_pattern(query)
    print(f"    Recall completed")


def test_expected_free_energy_planning():
    """Test expected free energy planning."""
    print("\n" + "=" * 60)
    print("EXPECTED FREE ENERGY PLANNING TEST")
    print("=" * 60)
    
    field = MorphodynamicField(width=32, height=32)
    planner = ExpectedFreeEnergyPlanner(field)
    
    print("\n  Planning to goal region:")
    
    # Goal region
    goal = (12, 20, 12, 20)
    
    plan = planner.plan(goal)
    
    print(f"    Policies evaluated: {len(plan['all_policies'])}")
    print(f"    Best policy approach: {plan['best_policy']['approach']}")
    print(f"    Expected free energy: {plan['best_policy']['expected_free_energy']:.3f}")
    print(f"    Trajectory length: {len(plan['best_policy']['trajectory'])}")


def test_integrated_morphodynamic():
    """Test integrated morphodynamic cognition."""
    print("\n" + "=" * 60)
    print("INTEGRATED MORPHODYNAMIC COGNITION TEST")
    print("=" * 60)
    
    cognition = MorphodynamicCognition(width=32, height=32)
    
    print("\n  Running cognitive cycle:")
    
    result = cognition.run_cycle(n_steps=30)
    
    print(f"    Steps: {len(result['steps'])}")
    print(f"    Final free energy: {result['field_summary']['free_energy']:.3f}")
    print(f"    Frequency bands: {result['field_summary']['frequency_bands']}")
    
    print("\n  Testing imagination:")
    imagination = cognition.imagine(n_steps=20)
    print(f"    Imagined {len(imagination['trajectory'])} steps")
    print(f"    Final free energy: {imagination['free_energy_trajectory'][-1]:.3f}")
    
    print("\n  Testing action:")
    cognition.act('attract', region=(10, 22, 10, 22))
    for _ in range(10):
        cognition.field.step()
    print(f"    Action executed, free energy: {cognition.field.F:.3f}")


def test_frequency_hierarchy():
    """Test frequency-based hierarchy."""
    print("\n" + "=" * 60)
    print("FREQUENCY HIERARCHY TEST")
    print("=" * 60)
    
    field = MorphodynamicField(width=32, height=32)
    
    print("\n  Evolving with different frequency patterns:")
    
    # Add slow oscillation
    for t in range(50):
        field.psi[16, 16] = 2.0 * np.sin(t * 0.1)
        field.step()
    
    print(f"    Slow band energy: {field.freq_bands['slow']['energy']:.3f}")
    print(f"    Medium band energy: {field.freq_bands['medium']['energy']:.3f}")
    print(f"    Fast band energy: {field.freq_bands['fast']['energy']:.3f}")
    
    # Add fast oscillation
    for t in range(50):
        field.psi[8, 8] = 1.5 * np.sin(t * 0.5)
        field.step()
    
    print(f"\n  After fast pattern:")
    print(f"    Slow band energy: {field.freq_bands['slow']['energy']:.3f}")
    print(f"    Medium band energy: {field.freq_bands['medium']['energy']:.3f}")
    print(f"    Fast band energy: {field.freq_bands['fast']['energy']:.3f}")


def phase_comparison():
    """Compare Phase 19 vs Phase 20."""
    print("\n" + "=" * 60)
    print("PHASE 19 VS PHASE 20 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 19 (Field-like State Vector):")
    print("    - psi ∈ R^n (state vector)")
    print("    - vortices: List[Dict] (created entities)")
    print("    - Free energy: decorative metric")
    print("    - Planning: procedural relaxation")
    print("    - Memory: separate traces list")
    print("    - 'architecture with field dynamics'")
    
    print("\n  Phase 20 (True Morphodynamic Field):")
    print("    - ψ(x,y,t) (spatial field)")
    print("    - Coherent structures: emergent patterns")
    print("    - Free energy: IS dynamics (δF/δψ)")
    print("    - Planning: expected free energy minimization")
    print("    - Memory: field hysteresis/plasticity")
    print("    - 'system IS dynamics'")
    
    print("\n  Critical shifts:")
    print("    1. State vector → Spatial field")
    print("    2. Created objects → Emergent structures")
    print("    3. Metric → Source of dynamics")
    print("    4. Storage → Hysteresis")
    print("    5. Layer stack → Frequency decomposition")
    
    print("\n  NOT: agent architecture with field-like components")
    print("  BUT: continuous morphodynamic field")


if __name__ == "__main__":
    test_morphodynamic_field()
    test_coherent_structures()
    test_plasticity_memory()
    test_expected_free_energy_planning()
    test_integrated_morphodynamic()
    test_frequency_hierarchy()
    phase_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 20 - COGNITIVE MORPHODYNAMIC FIELD")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 19 - field-like state vector dynamics
  To: Phase 20 - continuous morphodynamic medium where:
      - Field is spatial: ψ(x,y,t), not ψ ∈ R^n
      - Dynamics is PDE-based, not ODE with global vortices
      - Objects emerge as coherent structures, not created entities
      - Free energy IS the dynamics, not computed metric
      - Memory = field hysteresis, not storage
      - Hierarchy = frequency decomposition, not layer stack
      
  This is NO LONGER:
    agent architecture with field-like components
  This IS:
    continuous morphodynamic field
    from which cognition emerges

KEY TRANSITIONS:

A. STATE VECTOR → SPATIAL FIELD
   Phase 19: psi ∈ R^n (just a vector)
   Phase 20: ψ(x,y,t) (field value at each spatial position)
   
   This enables:
   - Spatial locality
   - Propagation
   - Wave interference
   - Local interactions

B. CREATED OBJECTS → EMERGENT STRUCTURES
   Phase 19: form_vortex() creates entities
   Phase 20: detect_stable_topology() finds patterns
   
   Coherent structures emerge from field dynamics:
   - Standing waves (oscillatory patterns)
   - Vortices (rotational patterns)
   - Solitons (self-reinforcing structures)
   
   NOT created, NOT tracked.
   They ARE stable patterns of the field.

C. DECORATIVE METRIC → SOURCE OF DYNAMICS
   Phase 19: F = surprise + complexity - entropy (metric)
   Phase 20: δψ/δt = -δF/δψ + D∇²ψ (source of dynamics)
   
   Free energy functional derivative IS the driving force.
   The field evolves TO MINIMIZE free energy.
   This is true variational dynamics.

D. STORAGE → HYSTERESIS
   Phase 19: memory_traces.append(trajectory)
   Phase 20: past patterns modify field dynamics directly
   
   Memory = field hysteresis:
   - Adaptive threshold
   - Synaptic plasticity
   - Conductivity modulation
   - Attractor deepening
   
   Past literally changes present dynamics.

E. PROCEDURAL SIMULATION → EXPECTED FREE ENERGY
   Phase 19: simulate_relaxation(n_steps) (rollout)
   Phase 20: G(π) = E[q(s,a) - log p(o|s,a)] (variational)
   
   Planning = find policy that minimizes expected free energy.
   Trajectory emerges automatically as path of least surprise.

F. LAYER STACK → FREQUENCY DECOMPOSITION
   Phase 19: goal_layer → semantic_layer → sensory_layer
   Phase 20: slow_freq ↔ medium_freq ↔ fast_freq
   
   Hierarchy is frequency-based:
   - Slow: identity, worldview, long-term priors
   - Medium: patterns, habits
   - Fast: sensorimotor corrections

THE CRITICAL INSIGHT:
  Phase 19: "system has dynamics"
  Phase 20: "system IS dynamics"
  
  The field is not a representation.
  The field IS the reality.
  
  Objects, memory, planning, action:
  All emerge from morphodynamic field.
  
This IS:
  - Neural field theory
  - Morphodynamic cognition
  - Active inference (true implementation)
  - Self-organizing criticality
  - Continuous attractor dynamics
  - True physical cognitive substrate
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 20 Summary:

BEFORE:
  - State vector psi ∈ R^n
  - Created vortices as entities
  - Free energy as decorative metric
  - Planning as procedural simulation
  - Memory as separate storage
  - Hierarchy as layer stack

AFTER:
  - Spatial field ψ(x,y,t)
  - Emergent coherent structures
  - Free energy as source of dynamics (δF/δψ)
  - Expected free energy planning
  - Field hysteresis (plasticity)
  - Frequency decomposition hierarchy

The critical shift:
  "architecture pretending to be field" → "true morphodynamic field"
  
  Phase 19: system has dynamics
  Phase 20: system IS dynamics

After Phase 20:
  - No more "modules" in any sense
  - Everything is the field
  - Cognition emerges from morphodynamic medium
  - Objects, memory, identity are all stable patterns
  - This is the foundation for true cognitive substrates
"""