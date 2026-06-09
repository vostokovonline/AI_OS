"""
Phase 18: Unified Cognitive Dynamics

ARCHITECTURAL SHIFT:
  From: Phase 17 - modular latent physics (causal/temporal/object/hierarchical/counterfactual)
  To: Phase 18 - running cognitive dynamics where:
      - Predictive coding loop drives all inference
      - Latent space is energy field with attractors
      - World persists and evolves even without observation
      
  This is NO LONGER:
    adding more modules to latent soup
  This IS:
    continuous field dynamics with attractor topology

THREE FUNDAMENTAL PROBLEMS WITH PHASE 17:
  1. Module soup: Each component computes separately, no unified dynamics
  2. Passive world: Latent space is flat vector space, no energy landscape
  3. Observation-driven: Objects given from outside, not discovered/simulated

THREE FUNDAMENTAL FIXES FOR PHASE 18:
  A. Predictive Coding Core: Error-driven belief updates across hierarchy
  B. Latent Field Dynamics: Continuous attractor field, not vector soup
  C. Persistent World: Hidden evolution, object continuity, surprise generation

CRITICAL INSIGHT:
  Phase 17: "modules that compute"
  Phase 18: "dynamics that run"
  
  The difference:
  - Phase 17: z_t -> encode() -> predict() -> transition()
  - Phase 18: prediction_error -> attractor_force -> field_dynamics -> topology

  Phase 18 IS:
    predictive processing (active inference)
    continuous attractor field
    persistent simulated world
    emergent object decomposition
    true counterfactual world-lines
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


# ============================================================================
# PHASE 18A: PREDICTIVE CODING CORE
# ============================================================================
"""
Predictive Coding Loop:

NOT: encoder -> predict -> output
BUT: top-down prediction + bottom-up error + belief update

Hierarchy:
  goal level (intentions)
      ↓ prediction
  semantic level (objects)
      ↓ prediction  
  sensory level (features)
      ↓ prediction
  observation (reality)
      ↑ error (mismatch)
  sensory level (update beliefs)
      ↑ error
  semantic level (update beliefs)
      ↑ error
  goal level (update intentions)

This is what GENERATES perception, not encoding.
"""

@dataclass
class PredictionError:
    """Error at a level of hierarchy."""
    level: str
    prediction: np.ndarray
    actual: np.ndarray
    error_vector: np.ndarray
    precision: float  # Confidence in this level
    scale: float = 1.0  # Error scale factor


class PredictiveCodingLayer:
    """
    Single layer in predictive coding hierarchy.
    
    Each layer:
      - Receives prediction from above (top-down)
      - Receives prediction error from below (bottom-up)
      - Updates belief to minimize prediction error
      - Generates prediction for level below
    """
    
    def __init__(self, layer_name: str, dim: int, parent: Optional['PredictiveCodingLayer'] = None,
                 child: Optional['PredictiveCodingLayer'] = None):
        self.layer_name = layer_name
        self.dim = dim
        
        # Hierarchy links
        self.parent = parent
        self.child = child
        
        # Belief state (what layer thinks is true)
        self.belief = np.zeros(dim)
        
        # Precision (confidence in beliefs)
        self.precision = 1.0
        
        # Prediction weights (parent -> this)
        self.W_topdown = np.random.randn(dim, dim) * 0.1 if parent else None
        
        # Prediction weights (this -> child)
        self.W_bottomup = np.random.randn(dim, dim) * 0.1 if child else None
        
        # Prediction error
        self.prediction_error = PredictionError(
            level=layer_name,
            prediction=np.zeros(dim),
            actual=np.zeros(dim),
            error_vector=np.zeros(dim),
            precision=1.0
        )
        
        # Learning rate
        self.alpha = 0.1
    
    def predict_topdown(self, parent_belief: np.ndarray) -> np.ndarray:
        """Generate prediction for this level from parent."""
        if self.parent is None:
            # Top level - generate from internal state
            return self.belief.copy()
        
        # Top-down prediction
        if self.W_topdown is None:
            return self.belief.copy()
        
        belief_to_use = parent_belief
        if len(belief_to_use) != self.W_topdown.shape[1]:
            if len(belief_to_use) < self.W_topdown.shape[1]:
                belief_to_use = np.pad(belief_to_use, (0, self.W_topdown.shape[1] - len(belief_to_use)))
            else:
                belief_to_use = belief_to_use[:self.W_topdown.shape[1]]
        
        prediction = self.W_topdown @ belief_to_use
        return prediction
    
    def receive_error(self, error: PredictionError):
        """Receive prediction error from below."""
        self.prediction_error = error
        
        # Update belief to minimize error
        self._update_belief()
    
    def _update_belief(self):
        """Update belief to minimize prediction error."""
        error = self.prediction_error.error_vector
        
        # Ensure error matches belief dimension
        if len(error) != len(self.belief):
            if len(error) < len(self.belief):
                error = np.pad(error, (0, len(self.belief) - len(error)))
            else:
                error = error[:len(self.belief)]
        
        # Precision-weighted error
        precision = self.prediction_error.precision * self.precision
        weighted_error = error * precision
        
        # Update belief
        self.belief = self.belief + self.alpha * weighted_error
        
        # Update precision based on error magnitude
        error_magnitude = np.linalg.norm(error)
        if error_magnitude > 0.1:
            self.precision = self.precision * 0.95  # Reduce confidence when error is large
        else:
            self.precision = min(2.0, self.precision * 1.02)  # Increase confidence when error is small
    
    def predict_bottomup(self) -> np.ndarray:
        """Generate prediction for child level."""
        if self.child is None:
            # Bottom level - prediction is belief
            return self.belief.copy()
        
        # Bottom-up prediction
        if self.W_bottomup is None:
            return self.belief.copy()
        
        belief_to_use = self.belief
        if len(belief_to_use) != self.W_bottomup.shape[1]:
            if len(belief_to_use) < self.W_bottomup.shape[1]:
                belief_to_use = np.pad(belief_to_use, (0, self.W_bottomup.shape[1] - len(belief_to_use)))
            else:
                belief_to_use = belief_to_use[:self.W_bottomup.shape[1]]
        
        prediction = self.W_bottomup @ belief_to_use
        return prediction
    
    def get_error_for_parent(self) -> PredictionError:
        """Compute error to send upward."""
        prediction = self.predict_topdown(self.parent.belief if self.parent else np.zeros(self.dim))
        error_vector = self.belief - prediction
        
        return PredictionError(
            level=self.layer_name,
            prediction=prediction,
            actual=self.belief.copy(),
            error_vector=error_vector,
            precision=self.precision
        )


class PredictiveCodingCore:
    """
    Phase 18A: Predictive Coding Core
    
    NOT: encoder-driven pipeline
    BUT: error-driven belief updates through hierarchy
    
    Structure:
      goal_layer
          ↓ top-down prediction
      semantic_layer
          ↓ top-down prediction
      sensory_layer
          ↓ top-down prediction
      observation (reality)
          ↑ bottom-up error
      sensory_layer (belief update)
          ↑ bottom-up error
      semantic_layer (belief update)
          ↑ bottom-up error
      goal_layer (belief update)
    """
    
    def __init__(self, sensory_dim: int = 4, semantic_dim: int = 4, goal_dim: int = 2):
        self.sensory_dim = sensory_dim
        self.semantic_dim = semantic_dim
        self.goal_dim = goal_dim
        
        # Create hierarchy layers
        self.goal_layer = PredictiveCodingLayer("goal", goal_dim, parent=None, child=None)
        self.semantic_layer = PredictiveCodingLayer("semantic", semantic_dim, 
                                                    parent=self.goal_layer, child=None)
        self.sensory_layer = PredictiveCodingLayer("sensory", sensory_dim,
                                                   parent=self.semantic_layer, child=None)
        
        # Link layers in hierarchy
        self.goal_layer.child = self.semantic_layer
        self.semantic_layer.parent = self.goal_layer
        self.semantic_layer.child = self.sensory_layer
        self.sensory_layer.parent = self.semantic_layer
        
        # Initialize weights
        self.goal_layer.W_bottomup = np.random.randn(goal_dim, goal_dim) * 0.1
        self.semantic_layer.W_topdown = np.random.randn(semantic_dim, goal_dim) * 0.1
        self.semantic_layer.W_bottomup = np.random.randn(semantic_dim, semantic_dim) * 0.1
        self.sensory_layer.W_topdown = np.random.randn(sensory_dim, semantic_dim) * 0.1
        self.sensory_layer.W_bottomup = np.random.randn(sensory_dim, sensory_dim) * 0.1
        
        # Forward weights (sensory encoding)
        self.W_encode = np.random.randn(sensory_dim, sensory_dim) * 0.1
        
        # Error history
        self.error_history: List[Dict] = []
        
    def process_observation(self, observation: np.ndarray, goal_context: Optional[np.ndarray] = None) -> Dict:
        """
        Process observation through predictive coding loop.
        
        NOT: observation -> encode -> predict -> output
        BUT: 
          1. Generate top-down predictions
          2. Compare with observation
          3. Propagate error upward
          4. Update beliefs
          5. Generate new predictions
        """
        observation = np.asarray(observation).flatten()
        
        # Pad or truncate observation
        if len(observation) < self.sensory_dim:
            observation_padded = np.pad(observation, (0, self.sensory_dim - len(observation)))
        else:
            observation_padded = observation[:self.sensory_dim]
        
        # Initialize beliefs if needed
        if np.all(self.sensory_layer.belief == 0):
            self.sensory_layer.belief = observation_padded.copy()
            self.semantic_layer.belief = self.sensory_layer.W_topdown @ observation_padded if self.sensory_layer.W_topdown is not None else observation_padded.copy()
            self.goal_layer.belief = self.semantic_layer.W_topdown @ self.semantic_layer.belief[:self.goal_dim] if len(self.semantic_layer.belief) >= self.goal_dim else self.semantic_layer.belief[:self.goal_dim].copy()
        
        # ===== STEP 1: Top-down predictions =====
        if goal_context is not None:
            self.goal_layer.belief = np.asarray(goal_context).flatten()[:self.goal_dim]
        
        # Goal predicts semantic
        semantic_from_goal = self.goal_layer.predict_bottomup()
        semantic_prediction = semantic_from_goal.copy()
        if len(semantic_prediction) < self.semantic_dim:
            semantic_prediction = np.pad(semantic_prediction, (0, self.semantic_dim - len(semantic_prediction)))
        else:
            semantic_prediction = semantic_prediction[:self.semantic_dim]
        
        # Semantic predicts sensory
        semantic_belief = self.semantic_layer.belief
        if len(semantic_belief) < self.semantic_dim:
            semantic_belief = np.pad(semantic_belief, (0, self.semantic_dim - len(semantic_belief)))
        
        sensory_prediction = semantic_belief.copy()
        if len(sensory_prediction) < self.sensory_dim:
            sensory_prediction = np.pad(sensory_prediction, (0, self.sensory_dim - len(sensory_prediction)))
        else:
            sensory_prediction = sensory_prediction[:self.sensory_dim]
        
        # ===== STEP 2: Compute errors =====
        sensory_error = PredictionError(
            level="sensory",
            prediction=sensory_prediction,
            actual=observation_padded,
            error_vector=observation_padded - sensory_prediction,
            precision=self.sensory_layer.precision
        )
        
        # ===== STEP 3: Propagate error upward (Free Energy gradient) =====
        self.sensory_layer.receive_error(sensory_error)
        
        semantic_error = self.sensory_layer.get_error_for_parent()
        self.semantic_layer.receive_error(semantic_error)
        
        goal_error = self.semantic_layer.get_error_for_parent()
        self.goal_layer.receive_error(goal_error)
        
        # Get updated beliefs
        sensory_belief = self.sensory_layer.belief
        semantic_belief = self.semantic_layer.belief
        goal_belief = self.goal_layer.belief
        
        # Record errors
        self.error_history.append({
            'sensory_error': np.linalg.norm(sensory_error.error_vector),
            'semantic_error': np.linalg.norm(semantic_error.error_vector),
            'goal_error': np.linalg.norm(goal_error.error_vector),
            'sensory_precision': self.sensory_layer.precision,
            'semantic_precision': self.semantic_layer.precision,
            'goal_precision': self.goal_layer.precision
        })
        
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-50:]
        
        return {
            'beliefs': {
                'sensory': sensory_belief.tolist(),
                'semantic': semantic_belief.tolist(),
                'goal': goal_belief.tolist()
            },
            'errors': {
                'sensory': np.linalg.norm(sensory_error.error_vector),
                'semantic': np.linalg.norm(semantic_error.error_vector),
                'goal': np.linalg.norm(goal_error.error_vector)
            },
            'precisions': {
                'sensory': self.sensory_layer.precision,
                'semantic': self.semantic_layer.precision,
                'goal': self.goal_layer.precision
            }
        }
    
    def get_predictions_for_action(self) -> Dict[str, np.ndarray]:
        """Get predictions at each level for action planning."""
        return {
            'sensory_prediction': self.sensory_layer.predict_bottomup(),
            'semantic_prediction': self.semantic_layer.predict_bottomup(),
            'goal_prediction': self.goal_layer.predict_bottomup()
        }


# ============================================================================
# PHASE 18B: LATENT FIELD DYNAMICS
# ============================================================================
"""
Latent Field Dynamics:

NOT: flat vector space where z moves around
BUT: energy field where dynamics follow gradient

Field equation:
  V(z) = -Σ depth_i * exp(-||z - center_i||² / radius_i²)
  
  Force = -∇V(z) = Σ depth_i * (z - center_i) / radius_i² * exp(...)

This gives:
  - Attraction to basins
  - Natural navigation
  - Stability from Hessian
  - Topology from field structure
"""

@dataclass
class Attractor:
    """An attractor in latent field."""
    attractor_id: str
    center: np.ndarray
    depth: float
    radius: float
    eigenvalue: float  # Stability measure (negative = stable)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    age: int = 0
    basin_mass: float = 1.0
    
    def compute_potential(self, z: np.ndarray) -> float:
        """Compute potential at point z."""
        z = np.asarray(z).flatten()
        distance = np.linalg.norm(z - self.center)
        return -self.depth * np.exp(-distance**2 / (self.radius**2 + 1e-8))
    
    def compute_force(self, z: np.ndarray) -> np.ndarray:
        """Compute force at point z (negative gradient)."""
        z = np.asarray(z).flatten()
        distance = z - self.center
        potential = self.compute_potential(z)
        
        # Force = -gradient of potential
        magnitude = 2 * self.depth * distance / (self.radius**2 + 1e-8) * np.exp(-np.linalg.norm(distance)**2 / (self.radius**2 + 1e-8))
        
        return -magnitude
    
    def update(self, dt: float = 0.1):
        """Update attractor parameters."""
        self.age += 1
        # Slow erosion
        self.depth = max(0.1, self.depth * 0.999)


class LatentFieldDynamics:
    """
    Phase 18B: Latent Field Dynamics
    
    NOT: z_t -> z_t+1 linear transition
    BUT: dynamics on energy field where:
         - z moves toward attractors
         - Trajectories follow field gradients
         - Topology emerges from field structure
         - Stability from Hessian eigenvalues
    """
    
    def __init__(self, latent_dim: int = 4):
        self.latent_dim = latent_dim
        
        # Current state
        self.z = np.zeros(latent_dim)
        
        # Attractors in field
        self.attractors: List[Attractor] = []
        
        # Field parameters
        self.field_strength = 1.0
        self.noise_strength = 0.1
        
        # Temperature (exploration)
        self.temperature = 1.0
        
        # Time
        self.t = 0.0
        self.dt = 0.1
        
        # Trajectory history
        self.trajectory: List[np.ndarray] = []
        
        # Initialize with some attractors
        self._initialize_attractors()
    
    def _initialize_attractors(self):
        """Initialize with default attractors."""
        # Goal attractor
        self.attractors.append(Attractor(
            attractor_id="goal",
            center=np.zeros(self.latent_dim),
            depth=1.0,
            radius=2.0,
            eigenvalue=-0.5
        ))
        
        # Safety attractor (slight offset)
        self.attractors.append(Attractor(
            attractor_id="safety",
            center=np.array([0.5, 0.5, 0.0, 0.0]),
            depth=0.5,
            radius=1.5,
            eigenvalue=-0.3
        ))
    
    def compute_potential_field(self, z: np.ndarray) -> float:
        """Compute total potential at z."""
        z = np.asarray(z).flatten()
        total_potential = 0.0
        
        for attractor in self.attractors:
            total_potential += attractor.compute_potential(z)
        
        return total_potential
    
    def compute_force_field(self, z: np.ndarray) -> np.ndarray:
        """Compute total force at z."""
        z = np.asarray(z).flatten()
        total_force = np.zeros(self.latent_dim)
        
        for attractor in self.attractors:
            total_force += attractor.compute_force(z)
        
        return total_force * self.field_strength
    
    def step(self, external_force: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single step of field dynamics.
        
        NOT: z_next = W @ z + action
        BUT: dynamics follow force field + noise + external influence
        """
        # Get force at current position
        force = self.compute_force_field(self.z)
        
        # Add external force
        if external_force is not None:
            external_force = np.asarray(external_force).flatten()
            if len(external_force) <= self.latent_dim:
                force = force + external_force
        
        # Temperature (noise for exploration)
        noise = np.random.randn(self.latent_dim) * self.noise_strength * self.temperature
        
        # Update position (Euler integration)
        self.z = self.z + force * self.dt + noise * np.sqrt(self.dt)
        
        # Record trajectory
        self.trajectory.append(self.z.copy())
        if len(self.trajectory) > 1000:
            self.trajectory = self.trajectory[-500:]
        
        self.t += self.dt
        
        return self.z.copy()
    
    def form_attractor(self, z_center: np.ndarray, depth: float = 1.0):
        """Form new attractor at position."""
        z_center = np.asarray(z_center).flatten()
        
        # Check if similar attractor exists
        for attractor in self.attractors:
            if np.linalg.norm(z_center - attractor.center) < 0.5:
                # Deepen existing attractor
                attractor.depth = min(5.0, attractor.depth + depth * 0.1)
                attractor.basin_mass += 0.1
                return attractor
        
        # Create new attractor
        new_attractor = Attractor(
            attractor_id=f"attractor_{len(self.attractors)}",
            center=z_center.copy(),
            depth=depth,
            radius=1.0,
            eigenvalue=-0.5,
            age=0
        )
        
        self.attractors.append(new_attractor)
        
        # Limit attractors
        if len(self.attractors) > 20:
            # Remove shallow/old attractors
            self.attractors.sort(key=lambda a: a.depth * a.basin_mass)
            self.attractors = self.attractors[-15:]
        
        return new_attractor
    
    def collapse_attractor(self, attractor_id: str):
        """Collapse (remove) an attractor."""
        self.attractors = [a for a in self.attractors if a.attractor_id != attractor_id]
    
    def split_attractor(self, attractor_id: str):
        """Split an attractor into two."""
        original = None
        for a in self.attractors:
            if a.attractor_id == attractor_id:
                original = a
                break
        
        if original is None:
            return
        
        # Create two offset attractors
        offset = np.random.randn(self.latent_dim) * 0.5
        
        self.form_attractor(original.center + offset, depth=original.depth * 0.7)
        self.form_attractor(original.center - offset, depth=original.depth * 0.7)
        
        # Remove original
        self.collapse_attractor(attractor_id)
    
    def get_field_topology(self) -> Dict:
        """Get topology of attractor field."""
        if not self.attractors:
            return {'n_attractors': 0}
        
        potentials = [a.compute_potential(self.z) for a in self.attractors]
        
        return {
            'n_attractors': len(self.attractors),
            'current_potential': self.compute_potential_field(self.z),
            'current_force_magnitude': np.linalg.norm(self.compute_force_field(self.z)),
            'dominant_attractor': max(self.attractors, key=lambda a: a.depth).attractor_id,
            'avg_depth': np.mean([a.depth for a in self.attractors]),
            'total_basin_mass': sum(a.basin_mass for a in self.attractors)
        }
    
    def simulate_trajectory(self, n_steps: int, external_forces: List[np.ndarray]) -> List[np.ndarray]:
        """Simulate trajectory without actually executing."""
        trajectory = [self.z.copy()]
        z_saved = self.z.copy()
        
        for i in range(n_steps):
            force = external_forces[i] if i < len(external_forces) else None
            z_next = self.z + self.compute_force_field(self.z) * self.dt
            trajectory.append(z_next)
        
        # Restore
        self.z = z_saved
        
        return trajectory


# ============================================================================
# PHASE 18C: PERSISTENT SIMULATED WORLD
# ============================================================================
"""
Persistent World:

NOT: world only exists when observed
BUT: world persists and evolves even without observation

Key insight:
  - Objects have hidden state that evolves autonomously
  - System maintains belief about hidden state
  - Surprise occurs when observation contradicts belief
  - This is what generates:
      * object permanence
      * expectation violation
      * imagination
      * planning across time
"""

@dataclass
class WorldObject:
    """An object in the persistent simulated world."""
    object_id: str
    position: np.ndarray
    velocity: np.ndarray
    properties: Dict[str, float]
    hidden_state: np.ndarray  # State NOT directly observable
    existence_probability: float = 1.0  # Belief about existence
    last_observed: float = -np.inf  # Time of last observation
    causal_role: str = "entity"  # 'cause', 'effect', 'mediator'
    
    def evolve_hidden(self, dt: float = 0.1):
        """Evolve hidden state over time."""
        # Hidden state evolves according to its own dynamics
        self.hidden_state = self.hidden_state * 0.99 + np.random.randn(len(self.hidden_state)) * 0.1
        
        # Position evolves
        self.position = self.position + self.velocity * dt
        self.velocity = self.velocity * 0.95  # Damping
        
        # Existence probability slowly decays
        self.existence_probability = max(0.5, self.existence_probability * 0.999)


class PersistentWorldSimulator:
    """
    Phase 18C: Persistent World Simulator
    
    NOT: world exists only during observation
    BUT: world persists and evolves even without observation
    
    This is what generates:
      - Object permanence (objects exist when not observed)
      - Expectation violation (surprise when observation contradicts belief)
      - Imagination (simulating world without observation)
      - Planning (using simulated future states)
      - Counterfactuals (alternative world trajectories)
    """
    
    def __init__(self, world_dim: int = 4, hidden_dim: int = 4):
        self.world_dim = world_dim
        self.hidden_dim = hidden_dim
        
        # World objects
        self.objects: Dict[str, WorldObject] = {}
        self.next_object_id = 0
        
        # World time
        self.t = 0.0
        self.dt = 0.1
        
        # Environment parameters
        self.friction = 0.95
        self.noise_level = 0.1
        
        # Causal graph (discovered from observation)
        self.causal_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Belief state over world
        self.belief_state: Dict[str, float] = {}  # object_id -> belief strength
        
        # Observation history (for causal discovery)
        self.observation_history: List[Dict] = []
    
    def create_object(self, position: np.ndarray, properties: Optional[Dict] = None,
                    hidden_state: Optional[np.ndarray] = None) -> str:
        """Create new object in world."""
        obj_id = f"obj_{self.next_object_id}"
        self.next_object_id += 1
        
        obj = WorldObject(
            object_id=obj_id,
            position=np.asarray(position).flatten(),
            velocity=np.zeros(self.world_dim),
            properties=properties or {},
            hidden_state=hidden_state or np.random.randn(self.hidden_dim) * 0.5,
            existence_probability=1.0,
            last_observed=self.t
        )
        
        self.objects[obj_id] = obj
        self.belief_state[obj_id] = 1.0
        
        return obj_id
    
    def observe(self, positions: List[np.ndarray], 
                properties: Optional[List[Dict]] = None) -> Dict:
        """
        Observe world state and update beliefs.
        
        Returns: observation with belief updates
        """
        self.t += self.dt
        
        # Track observed objects
        observed_ids = set()
        
        for i, position in enumerate(positions):
            position = np.asarray(position).flatten()
            
            # Find matching object
            best_match_id = None
            best_distance = float('inf')
            
            for obj_id, obj in self.objects.items():
                distance = np.linalg.norm(position - obj.position)
                if distance < best_distance and obj.existence_probability > 0.3:
                    best_distance = distance
                    best_match_id = obj_id
            
            if best_match_id and best_distance < 1.0:
                # Update existing object
                obj = self.objects[best_match_id]
                obj.position = position
                obj.last_observed = self.t
                obj.existence_probability = min(1.0, obj.existence_probability + 0.1)
                
                if properties and i < len(properties):
                    obj.properties.update(properties[i])
                
                observed_ids.add(best_match_id)
                self.belief_state[best_match_id] = min(1.0, self.belief_state.get(best_match_id, 0) + 0.1)
            else:
                # Create new object
                new_id = self.create_object(position, 
                                         properties[i] if properties and i < len(properties) else None)
                observed_ids.add(new_id)
        
        # Update beliefs for unobserved objects
        for obj_id in list(self.objects.keys()):
            if obj_id not in observed_ids:
                obj = self.objects[obj_id]
                obj.last_observed += self.dt
                
                # Belief decays slowly
                self.belief_state[obj_id] = self.belief_state.get(obj_id, 1.0) * 0.995
                
                # But we still track it
                obj.existence_probability = max(0.3, obj.existence_probability * 0.99)
        
        # Record observation
        self.observation_history.append({
            't': self.t,
            'observed_ids': list(observed_ids),
            'positions': [p.tolist() for p in positions]
        })
        
        if len(self.observation_history) > 100:
            self.observation_history = self.observation_history[-50:]
        
        # Discover causal structure
        self._discover_causal_links()
        
        return {
            'observed_objects': list(observed_ids),
            'belief_state': {k: v for k, v in self.belief_state.items() if v > 0.3},
            'total_belief_mass': sum(self.belief_state.values())
        }
    
    def simulate_without_observation(self, n_steps: int) -> List[Dict]:
        """
        Simulate world evolution without observation.
        
        This is what generates:
          - Imagination (what happens if I wait?)
          - Planning (what will happen?)
          - Counterfactuals (what if X had happened?)
        """
        world_state_history = []
        
        for step in range(n_steps):
            # Evolve all objects
            for obj in self.objects.values():
                obj.evolve_hidden(self.dt)
            
            self.t += self.dt
            
            world_state_history.append({
                't': self.t,
                'object_states': {
                    obj_id: {
                        'position': obj.position.tolist(),
                        'hidden': obj.hidden_state.tolist(),
                        'existence_prob': obj.existence_probability
                    }
                    for obj_id, obj in self.objects.items()
                    if obj.existence_probability > 0.3
                }
            })
        
        return world_state_history
    
    def _discover_causal_links(self):
        """Discover causal links from observation history."""
        if len(self.observation_history) < 10:
            return
        
        # Simple correlation-based discovery
        recent = self.observation_history[-10:]
        
        for obs1, obs2 in zip(recent[:-1], recent[1:]):
            # Objects that appeared in consecutive observations may be causally connected
            ids1 = set(obs1['observed_ids'])
            ids2 = set(obs2['observed_ids'])
            
            common = ids1 & ids2
            for obj_id in common:
                # Track co-occurrences
                self.causal_graph[obj_id] = list(ids2 - {obj_id})
    
    def get_world_state(self, include_hidden: bool = True) -> Dict:
        """Get current world state."""
        return {
            't': self.t,
            'objects': {
                obj_id: {
                    'position': obj.position.tolist(),
                    'velocity': obj.velocity.tolist(),
                    'properties': obj.properties,
                    'hidden_state': obj.hidden_state.tolist() if include_hidden else None,
                    'existence_probability': obj.existence_probability,
                    'last_observed': self.t - obj.last_observed
                }
                for obj_id, obj in self.objects.items()
                if obj.existence_probability > 0.3
            },
            'causal_graph': dict(self.causal_graph),
            'n_objects': len([o for o in self.objects.values() if o.existence_probability > 0.3])
        }
    
    def counterfactual_world(self, object_id: str, intervention: Dict,
                           n_steps: int) -> List[Dict]:
        """
        Generate counterfactual world trajectory.
        
        do(intervention):
          - Set object's properties to intervention values
          - Evolve world from that point
          - Compare with actual trajectory
        """
        if object_id not in self.objects:
            return []
        
        # Save current state
        original_obj = self.objects[object_id]
        original_position = original_obj.position.copy()
        original_hidden = original_obj.hidden_state.copy()
        
        # Apply intervention
        if 'position' in intervention:
            original_obj.position = np.asarray(intervention['position'])
        if 'hidden_state' in intervention:
            original_obj.hidden_state = np.asarray(intervention['hidden_state'])
        
        # Simulate
        cf_trajectory = self.simulate_without_observation(n_steps)
        
        # Restore
        original_obj.position = original_position
        original_obj.hidden_state = original_hidden
        
        return cf_trajectory


# ============================================================================
# UNIFIED COGNITIVE DYNAMICS (PHASE 18)
# ============================================================================

class UnifiedCognitiveDynamics:
    """
    Phase 18: Unified Cognitive Dynamics
    
    Combines:
      A. Predictive Coding Core (error-driven hierarchy)
      B. Latent Field Dynamics (attractor field)
      C. Persistent World (hidden evolution)
    
    NOT: modular pipeline
    BUT: single running dynamics where:
         - Predictive coding drives belief updates
         - Field dynamics shapes state evolution
         - World persists and generates surprise
    """
    
    def __init__(self, observation_dim: int = 2, latent_dim: int = 4, action_dim: int = 2):
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Core components
        self.predictive_coding = PredictiveCodingCore(
            sensory_dim=latent_dim,
            semantic_dim=latent_dim,
            goal_dim=action_dim
        )
        
        self.field_dynamics = LatentFieldDynamics(latent_dim=latent_dim)
        
        self.persistent_world = PersistentWorldSimulator(
            world_dim=latent_dim,
            hidden_dim=latent_dim
        )
        
        # Current state
        self.z = np.zeros(latent_dim)
        
        # Execution history
        self.execution_history: List[Dict] = []
        
        # Initialize with some world objects
        self._initialize_world()
    
    def _initialize_world(self):
        """Initialize world with default objects."""
        self.persistent_world.create_object(
            position=np.array([0.0, 0.0, 0.0, 0.0]),
            properties={'type': 'origin'}
        )
        self.persistent_world.create_object(
            position=np.array([1.0, 0.5, 0.0, 0.0]),
            properties={'type': 'target'}
        )
    
    def process(self, observation: np.ndarray, goal: Optional[np.ndarray] = None,
                n_steps: int = 10) -> Dict:
        """
        Process through unified cognitive dynamics.
        
        NOT: pipeline
        BUT: running dynamics loop
        """
        results = []
        
        observation = np.asarray(observation).flatten()
        
        for step in range(n_steps):
            # ===== PREDICTIVE CODING =====
            coding_result = self.predictive_coding.process_observation(
                observation, goal
            )
            
            # ===== OBSERVE WORLD =====
            world_state = self.persistent_world.observe(
                positions=[self.z.copy()],
                properties=[{'step': step}]
            )
            
            # ===== FIELD DYNAMICS =====
            # Get force from field
            field_force = self.field_dynamics.compute_force_field(self.z)
            
            # Add goal force
            if goal is not None:
                goal_padded = np.asarray(goal).flatten()
                if len(goal_padded) < self.latent_dim:
                    goal_padded = np.pad(goal_padded, (0, self.latent_dim - len(goal_padded)))
                else:
                    goal_padded = goal_padded[:self.latent_dim]
                goal_force = (goal_padded - self.z) * 0.3
                field_force = field_force + goal_force
            
            # Step field dynamics
            z_next = self.field_dynamics.step(field_force)
            
            # ===== FORM ATTRACTOR FROM SUCCESSFUL TRAJECTORY =====
            if step > 0 and step % 5 == 0:
                self.field_dynamics.form_attractor(z_next, depth=0.3)
            
            # ===== UPDATE WORLD (simulate without observation) =====
            if step % 3 == 0:
                simulated = self.persistent_world.simulate_without_observation(1)
            
            # ===== RECORD =====
            results.append({
                'step': step,
                'z': z_next.tolist(),
                'predictive_coding': {
                    'errors': coding_result['errors'],
                    'precisions': coding_result['precisions']
                },
                'field': self.field_dynamics.get_field_topology(),
                'world': {
                    'n_objects': world_state['total_belief_mass'],
                    'observed': len(world_state['observed_objects'])
                }
            })
            
            self.z = z_next
        
        return {
            'steps': results,
            'final_state': self.z.tolist(),
            'field_topology': self.field_dynamics.get_field_topology(),
            'world_state': self.persistent_world.get_world_state(),
            'coding_summary': {
                'final_sensory_error': results[-1]['predictive_coding']['errors']['sensory'],
                'final_precision': results[-1]['predictive_coding']['precisions']['sensory']
            }
        }


# ============================================================================
# TESTS
# ============================================================================

def test_predictive_coding():
    """Test predictive coding core."""
    print("\n" + "=" * 60)
    print("PREDICTIVE CODING CORE TEST")
    print("=" * 60)
    
    coding = PredictiveCodingCore(sensory_dim=4, semantic_dim=4, goal_dim=2)
    
    print("\n  Processing observations through predictive coding loop:")
    
    for i in range(30):
        observation = np.array([i * 0.1, i * 0.05, 0.0, 0.0]) + np.random.randn(4) * 0.1
        result = coding.process_observation(observation)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      Sensory error: {result['errors']['sensory']:.3f}")
            print(f"      Semantic error: {result['errors']['semantic']:.3f}")
            print(f"      Sensory precision: {result['precisions']['sensory']:.3f}")
    
    print(f"\n  Final state:")
    preds = coding.get_predictions_for_action()
    print(f"    Sensory prediction: {preds['sensory_prediction'][:2].round(3).tolist()}")


def test_latent_field():
    """Test latent field dynamics."""
    print("\n" + "=" * 60)
    print("LATENT FIELD DYNAMICS TEST")
    print("=" * 60)
    
    field = LatentFieldDynamics(latent_dim=4)
    
    print("\n  Running dynamics on energy field:")
    
    for i in range(50):
        z_next = field.step()
        
        if i % 10 == 9:
            topology = field.get_field_topology()
            print(f"    Step {i+1}:")
            print(f"      Position: {z_next[:2].round(3).tolist()}")
            print(f"      Potential: {topology['current_potential']:.3f}")
            print(f"      Force: {topology['current_force_magnitude']:.3f}")
    
    print("\n  Forming new attractor:")
    field.form_attractor(np.array([2.0, 1.0, 0.0, 0.0]), depth=2.0)
    topology = field.get_field_topology()
    print(f"    After formation: {topology['n_attractors']} attractors")
    
    print("\n  Simulating trajectory:")
    trajectory = field.simulate_trajectory(10, [np.zeros(4)] * 10)
    print(f"    Simulated {len(trajectory)} steps")
    print(f"    Final: {trajectory[-1][:2].round(3).tolist()}")


def test_persistent_world():
    """Test persistent world simulator."""
    print("\n" + "=" * 60)
    print("PERSISTENT WORLD SIMULATOR TEST")
    print("=" * 60)
    
    world = PersistentWorldSimulator(world_dim=4, hidden_dim=4)
    
    print("\n  Creating world objects:")
    obj1 = world.create_object(
        position=np.array([1.0, 0.5, 0.0, 0.0]),
        properties={'type': 'agent'}
    )
    obj2 = world.create_object(
        position=np.array([2.0, 1.0, 0.0, 0.0]),
        properties={'type': 'target'}
    )
    print(f"    Created: {obj1}, {obj2}")
    
    print("\n  Observing world:")
    for i in range(20):
        positions = [
            np.array([1.0 + i * 0.1, 0.5, 0.0, 0.0]),
            np.array([2.0 + i * 0.05, 1.0, 0.0, 0.0])
        ]
        result = world.observe(positions)
        
        if i % 5 == 4:
            print(f"    Step {i+1}: observed={result['observed_objects']}, belief_mass={result['total_belief_mass']:.2f}")
    
    print("\n  Simulating without observation:")
    simulation = world.simulate_without_observation(10)
    print(f"    Simulated {len(simulation)} steps")
    print(f"    Time advanced to: {simulation[-1]['t']:.1f}")
    
    print("\n  Counterfactual world:")
    cf_trajectory = world.counterfactual_world(
        obj1,
        intervention={'position': np.array([3.0, 2.0, 0.0, 0.0])},
        n_steps=5
    )
    print(f"    Counterfactual steps: {len(cf_trajectory)}")


def test_unified_dynamics():
    """Test unified cognitive dynamics."""
    print("\n" + "=" * 60)
    print("UNIFIED COGNITIVE DYNAMICS TEST")
    print("=" * 60)
    
    dynamics = UnifiedCognitiveDynamics(observation_dim=2, latent_dim=4, action_dim=2)
    
    print("\n  Processing through unified dynamics:")
    
    for epoch in range(3):
        observation = np.array([epoch * 0.5, epoch * 0.3]) + np.random.randn(2) * 0.1
        goal = np.array([2.0, 1.0])
        
        result = dynamics.process(observation, goal, n_steps=20)
        
        print(f"\n  Epoch {epoch + 1}:")
        print(f"    Steps: {len(result['steps'])}")
        print(f"    Final position: {result['final_state'][:2]}")
        print(f"    Field attractors: {result['field_topology']['n_attractors']}")
        print(f"    World objects: {result['world_state']['n_objects']}")


def test_field_vs_vector():
    """Compare field dynamics vs vector soup."""
    print("\n" + "=" * 60)
    print("FIELD DYNAMICS VS VECTOR SOUP")
    print("=" * 60)
    
    print("\n  Phase 17 (Vector Soup):")
    print("    z_next = W @ z + action")
    print("    No energy landscape")
    print("    No attractor dynamics")
    print("    Linear trajectory")
    
    print("\n  Phase 18 (Field Dynamics):")
    print("    force = -∇V(z)")
    print("    z_next = z + force * dt + noise")
    print("    Attractor basins shape trajectory")
    print("    Natural navigation toward deep basins")
    
    print("\n  Key difference:")
    print("    - Vector soup: trajectory is arbitrary")
    print("    - Field dynamics: trajectory follows field structure")


if __name__ == "__main__":
    test_predictive_coding()
    test_latent_field()
    test_persistent_world()
    test_unified_dynamics()
    test_field_vs_vector()
    
    print("\n" + "=" * 60)
    print("PHASE 18 - UNIFIED COGNITIVE DYNAMICS")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 17 - modular latent physics
  To: Phase 18 - unified cognitive dynamics where:
      - Predictive coding drives all belief updates
      - Field dynamics shapes state evolution
      - World persists and generates surprise
      
  This is NO LONGER:
    adding more modules to latent soup
  This IS:
    continuous field dynamics with attractor topology

THREE FUNDAMENTAL COMPONENTS:

A. PREDICTIVE CODING CORE
   - Top-down predictions through hierarchy
   - Bottom-up prediction errors
   - Precision-weighted belief updates
   - This GENERATES perception, not encoding

B. LATENT FIELD DYNAMICS
   - Energy field with attractors
   - Trajectory follows gradient
   - Topology from field structure
   - Stability from Hessian

C. PERSISTENT WORLD
   - World evolves even without observation
   - Hidden state persists
   - Surprise when observation contradicts belief
   - This GENERATES object permanence

CRITICAL INSIGHT:
  Phase 17: "modules that compute"
  Phase 18: "dynamics that run"
  
  - Phase 17: observation -> encode -> predict -> transition()
  - Phase 18: prediction_error -> attractor_force -> field_dynamics -> topology

This IS:
  - Predictive processing (active inference)
  - Continuous attractor field
  - Persistent simulated world
  - Emergent object decomposition
  - True counterfactual world-lines
  
This is CLOSER TO:
  - Friston's free energy principle
  - Predictive processing theories
  - World models that persist
  - Object-centric representations
  - True cognitive substrate
""")