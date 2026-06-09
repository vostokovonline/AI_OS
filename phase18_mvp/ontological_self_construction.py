"""
Phase 11: Ontological Self-Construction

ARCHITECTURAL SHIFT:
  From: System learns dynamics of self
  To: System learns what counts as self (ontology creation)

CRITICAL INSIGHT:
  Phase 10: learns dynamics INSIDE fixed ontology
  Phase 11: learns to CREATE and RESTRUCTURE ontology itself

  This is where:
    - Latent space plasticity appears (dimensional birth/death)
    - Semantic attractors emerge (meaning survives transformation)
    - Self-generated priors crystallize (not external)
    - Ontological metabolism starts (consume uncertainty → structure)
    - Destabilization emerges from contradiction accumulation

KEY PROBLEMS WITH PHASE 10:
  1. latent_dim = 2 (fixed, no plasticity)
  2. Gaussian assumption (smooth, unimodal)
  3. preferred_self = external (not emergent)
  4. destabilize_self() = external control (not emergent)
  5. Attractors = fixed points (not semantic invariants)
  6. Prior structure = hand-crafted (not self-generated)

WHAT APPEARS:
  1. Latent topology evolution
     - concept splitting/merging
     - manifold folding
     - dimensional birth/death
     - representational drift
     
  2. Semantic attractors
     - "meaning survives transformation" (not x(t+1) ≈ x(t))
     - stable semantic invariants across change
     
  3. Self-generated priors
     - compression pressure
     - survival regularities
     - temporal continuity
     - memory persistence
     
  4. Ontological metabolism
     - consume uncertainty → create structure
     - epistemic self-maintenance
     
  5. Emergent destabilization
     - contradiction accumulation
     - incompatible predictions
     - semantic compression failure
     - phase transitions from within

PHASE 11 = self-ontogenesis
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class SemanticAttractor:
    """
    Semantic attractor = stable meaning that survives transformation.
    
    NOT: x(t+1) ≈ x(t) (fixed point in vector space)
    BUT: meaning survives across transformations
    
    This is a HIGHER-LEVEL attractor in semantic space.
    """
    semantic_id: str                    # What this attractor "means"
    core_pattern: np.ndarray            # Core semantic pattern
    representational_extent: np.ndarray  # How much of latent space it covers
    stability: float                   # How stable this meaning is
    age: int                           # How long this meaning has existed
    transformation_history: List[np.ndarray]  # Transformations that preserved this meaning
    semantic_neighbors: List[str]       # Related semantic attractors
    contradiction_count: int           # Accumulated contradictions
    compression_pressure: float        # Drive to compress/cohere this meaning
    
    def survives_transformation(self, transformation: np.ndarray) -> bool:
        """Does this semantic attractor survive a transformation?"""
        transformed = self.core_pattern + transformation
        # Check if semantic core is preserved (not exact, but meaning-preserving)
        similarity = np.dot(self.core_pattern, transformed) / (
            np.linalg.norm(self.core_pattern) * np.linalg.norm(transformed) + 1e-8
        )
        return similarity > 0.7  # Meaning preserved if similar enough
    
    def absorb_transformation(self, transformation: np.ndarray):
        """This meaning grows by absorbing transformation."""
        self.transformation_history.append(transformation.copy())
        # Update core pattern (slowly)
        self.core_pattern = self.core_pattern * 0.95 + transformation * 0.05
        self.age += 1


@dataclass
class LatentDimension:
    """
    A dimension in the latent space that can evolve, split, or die.
    
    NOT: static axis
    BUT: evolving representational unit
    """
    dim_id: int
    current_value: float
    representational_role: str        # What this dimension represents
    semantic_content: str             # What this dimension MEANS
    activation_history: List[float]    # How this dimension has been used
    birth_time: int                   # When this dimension emerged
    plasticity: float                 # How easily this dimension changes
    death_threshold: float            # Below this, dimension dies
    stability: float                 # How stable this dimension is
    
    def should_split(self) -> bool:
        """Should this dimension split into two?"""
        # If variance is too high, dimension should split
        if len(self.activation_history) > 10:
            variance = np.var(self.activation_history[-10:])
            return variance > 1.5 and self.plasticity > 0.5
        return False
    
    def should_die(self) -> bool:
        """Should this dimension die?"""
        recent_avg = np.mean(self.activation_history[-20:]) if len(self.activation_history) > 20 else 0
        return recent_avg < self.death_threshold and self.stability < 0.3


class SelfGeneratedPrior:
    """
    Self-generated priors that emerge from survival structure.
    
    NOT: hand-crafted priors
    BUT: priors that crystallize from:
      - compression pressure
      - survival regularities
      - temporal continuity
      - memory persistence
    """
    def __init__(self):
        self.prior_structure: Dict[str, np.ndarray] = {}  # What system believes is "natural"
        self.survival_pressure_history: List[float] = []
        self.regularity_clusters: List[np.ndarray] = []  # Discovered regularities
        self.compression_gradient: float = 0.0  # Drive to compress representations
        
    def emerge_prior(self, experiences: List[np.ndarray]) -> Dict:
        """
        Emerge priors from experience patterns.
        
        Not: "system has prior X"
        But: "prior X crystallized from experience"
        """
        if len(experiences) < 10:
            return {}
        
        # Find regularities (compression)
        regularity = self._find_regularity(experiences)
        
        # Survival pressure shapes prior
        survival_shape = self._survival_shapes_prior(experiences)
        
        # Combine: regularity + survival → prior
        emerged_prior = {}
        for key in regularity:
            if key in survival_shape:
                emerged_prior[key] = 0.5 * regularity[key] + 0.5 * survival_shape[key]
        
        self.prior_structure.update(emerged_prior)
        
        return emerged_prior
    
    def _find_regularity(self, experiences: List[np.ndarray]) -> Dict:
        """Find regularities in experiences (compression)."""
        experiences_arr = np.array(experiences)
        
        # Find clusters (compressed representation)
        if len(experiences_arr) < 5:
            return {}
        
        # Simple clustering
        centroids = [experiences_arr[0]]
        for exp in experiences_arr[1:]:
            distances = [np.linalg.norm(exp - c) for c in centroids]
            min_dist = min(distances)
            min_idx = np.argmin(distances)
            
            if min_dist < 0.5:
                # Merge into existing cluster
                centroids[min_idx] = 0.9 * centroids[min_idx] + 0.1 * exp
            else:
                # New cluster
                centroids.append(exp.copy())
        
        # Regularities = cluster centroids
        regularity = {}
        for i, c in enumerate(centroids):
            regularity[f'regularity_{i}'] = c / (np.linalg.norm(c) + 1e-8)
        
        return regularity
    
    def _survival_shapes_prior(self, experiences: List[np.ndarray]) -> Dict:
        """Survival pressure shapes what becomes "natural"."""
        # What helped survive becomes preferred
        if len(experiences) < 5:
            return {}
        
        # Survival preferred = experiences that continued (didn't terminate)
        # For demo: prefer states closer to mean
        mean_exp = np.mean(experiences, axis=0)
        
        survival_prior = {}
        for i, exp in enumerate(experiences):
            # Distance from mean = "survival fitness"
            fitness = 1.0 / (1.0 + np.linalg.norm(exp - mean_exp))
            survival_prior[f'survival_{i}'] = fitness * exp
        
        return survival_prior


class OntologicalMetabolism:
    """
    Ontological metabolism: consume uncertainty → create structure.
    
    NOT: inference
    BUT: epistemic self-maintenance
    
    System converts uncertainty into stable meaning.
    """
    def __init__(self, latent_dim: int = 2):
        self.latent_dim = latent_dim
        self.uncertainty_buffer: List[float] = []
        self.structure_created: List[Dict] = []
        self.metabolic_rate: float = 0.1
        
    def consume_uncertainty(self, uncertainty: float, 
                          context: np.ndarray) -> Optional[Dict]:
        """
        Convert uncertainty into structure.
        
        High uncertainty → new structure emerges
        """
        self.uncertainty_buffer.append(uncertainty)
        
        # Threshold for structure creation
        if len(self.uncertainty_buffer) > 10:
            avg_uncertainty = np.mean(self.uncertainty_buffer[-10:])
            
            if uncertainty > avg_uncertainty * 1.5:
                # High uncertainty → create structure
                new_structure = self._create_structure(context, uncertainty)
                self.structure_created.append(new_structure)
                return new_structure
            
            # Low uncertainty → absorb/merge
            self._absorb_uncertainty(uncertainty)
        
        return None
    
    def _create_structure(self, context: np.ndarray, uncertainty: float) -> Dict:
        """
        Create new structure from uncertainty.
        
        This is the key: uncertainty is metabolized into meaning.
        """
        # Create semantic attractor from uncertainty
        semantic_id = f"semantic_{len(self.structure_created)}"
        
        structure = {
            'type': 'semantic_attractor',
            'id': semantic_id,
            'core_pattern': context.copy(),
            'stability': 0.5,
            'born_from_uncertainty': uncertainty,
            'age': 0
        }
        
        return structure
    
    def _absorb_uncertainty(self, uncertainty: float):
        """Absorb low uncertainty into existing structure."""
        if len(self.structure_created) > 0:
            # Add to most recent structure
            self.structure_created[-1]['stability'] += uncertainty * 0.1
            self.structure_created[-1]['age'] += 1
    
    def get_metabolic_state(self) -> Dict:
        """Get current metabolic state."""
        return {
            'uncertainty_buffer_size': len(self.uncertainty_buffer),
            'structures_created': len(self.structure_created),
            'metabolic_rate': self.metabolic_rate,
            'avg_uncertainty': np.mean(self.uncertainty_buffer) if self.uncertainty_buffer else 0
        }


class ContradictionAccumulator:
    """
    Accumulate contradictions → trigger emergent destabilization.
    
    NOT: external control
    BUT: contradictions accumulate → phase transition from within
    
    Triggers:
      - incompatible predictions
      - semantic compression failure
      - attractor overload
      - irreducible ambiguity
      - temporal inconsistency
    """
    def __init__(self):
        self.prediction_errors: List[float] = []
        self.semantic_conflicts: List[int] = []
        self.attractor_loads: List[float] = []
        self.ambiguity_threshold = 0.7
        self.phase_transition_threshold = 1.0
        
    def add_contradiction(self, prediction_error: float,
                         semantic_conflict: int = 0,
                         attractor_load: float = 0.0):
        """Add contradiction evidence."""
        self.prediction_errors.append(prediction_error)
        self.semantic_conflicts.append(semantic_conflict)
        self.attractor_loads.append(attractor_load)
        
        # Keep history bounded
        if len(self.prediction_errors) > 100:
            self.prediction_errors = self.prediction_errors[-50:]
            self.semantic_conflicts = self.semantic_conflicts[-50:]
            self.attractor_loads = self.attractor_loads[-50:]
    
    def should_destabilize(self) -> Tuple[bool, float]:
        """
        Should emergent destabilization occur?
        
        Returns: (should_destabilize, pressure_level)
        """
        if len(self.prediction_errors) < 10:
            return False, 0.0
        
        # Calculate pressure
        avg_prediction_error = np.mean(self.prediction_errors[-10:])
        avg_conflict = np.mean(self.semantic_conflicts[-10:]) if self.semantic_conflicts else 0
        avg_load = np.mean(self.attractor_loads[-10:]) if self.attractor_loads else 0
        
        # Pressure from multiple sources
        pressure = (
            avg_prediction_error * 0.4 +
            avg_conflict * 0.3 +
            avg_load * 0.3
        )
        
        return pressure > self.phase_transition_threshold, pressure
    
    def get_contradiction_summary(self) -> Dict:
        """Get summary of accumulated contradictions."""
        return {
            'prediction_errors': self.prediction_errors[-10:] if self.prediction_errors else [],
            'semantic_conflicts': self.semantic_conflicts[-10:] if self.semantic_conflicts else [],
            'attractor_loads': self.attractor_loads[-10:] if self.attractor_loads else [],
            'should_destabilize': self.should_destabilize()[0],
            'pressure': self.should_destabilize()[1]
        }


class EvolvingLatentSpace:
    """
    Evolving latent space with topology plasticity.
    
    NOT: fixed dimensional space
    BUT: space that evolves through:
      - dimensional birth/death
      - concept splitting/merging
      - manifold folding
      - representational drift
    """
    def __init__(self, initial_dim: int = 2):
        self.dimensions: List[LatentDimension] = []
        self.semantic_attractors: List[SemanticAttractor] = []
        
        # Initialize with base dimensions
        for i in range(initial_dim):
            self._create_dimension(role=f"base_{i}")
        
        # Current state in evolved space
        self.current_state: np.ndarray = np.zeros(initial_dim)
        
    def _create_dimension(self, role: str = "unknown") -> LatentDimension:
        """Create new dimension."""
        dim = LatentDimension(
            dim_id=len(self.dimensions),
            current_value=0.0,
            representational_role=role,
            semantic_content="",
            activation_history=[],
            birth_time=len(self.dimensions),
            plasticity=0.5,
            death_threshold=0.01,
            stability=0.5
        )
        self.dimensions.append(dim)
        return dim
    
    def _split_dimension(self, dim_idx: int) -> bool:
        """Split a dimension into two."""
        if dim_idx >= len(self.dimensions):
            return False
        
        original = self.dimensions[dim_idx]
        
        if not original.should_split():
            return False
        
        # Create new dimension
        new_dim = LatentDimension(
            dim_id=len(self.dimensions),
            current_value=original.current_value * 0.5,
            representational_role=f"split_from_{original.dim_id}",
            semantic_content=original.semantic_content + "_a",
            activation_history=[],
            birth_time=len(self.dimensions),
            plasticity=original.plasticity * 0.8,
            death_threshold=original.death_threshold,
            stability=original.stability * 0.9
        )
        
        # Modify original
        original.representational_role = f"split_from_{original.dim_id}_original"
        original.semantic_content = original.semantic_content + "_b"
        original.plasticity *= 0.7
        original.stability *= 0.8
        
        self.dimensions.append(new_dim)
        
        return True
    
    def _kill_dimension(self, dim_idx: int) -> bool:
        """Kill (prune) a dimension."""
        if dim_idx >= len(self.dimensions) or len(self.dimensions) <= 2:
            return False
        
        dim = self.dimensions[dim_idx]
        
        if dim.should_die():
            # Mark as inactive
            dim.plasticity = 0
            dim.stability = 0
            return True
        
        return False
    
    def evolve(self, state: np.ndarray) -> np.ndarray:
        """
        Evolve latent space based on current state.
        
        Returns evolved state (may have different dimensionality).
        """
        # Activate dimensions
        for i, dim in enumerate(self.dimensions):
            if i < len(state):
                dim.current_value = state[i]
                dim.activation_history.append(state[i])
        
        # Update current state
        self.current_state = np.array([d.current_value for d in self.dimensions if d.stability > 0])
        
        # Check for splits
        for i, dim in enumerate(self.dimensions):
            if dim.should_split() and np.random.random() < 0.1:
                self._split_dimension(i)
        
        # Check for deaths
        dims_to_kill = []
        for i, dim in enumerate(self.dimensions):
            if dim.should_die():
                dims_to_kill.append(i)
        
        for idx in dims_to_kill:
            self._kill_dimension(idx)
        
        return self.current_state
    
    def create_semantic_attractor(self, pattern: np.ndarray, semantic_id: str) -> SemanticAttractor:
        """Create new semantic attractor."""
        attractor = SemanticAttractor(
            semantic_id=semantic_id,
            core_pattern=pattern.copy(),
            representational_extent=np.ones(len(self.dimensions)) * 0.5,
            stability=0.5,
            age=0,
            transformation_history=[],
            semantic_neighbors=[],
            contradiction_count=0,
            compression_pressure=0.0
        )
        self.semantic_attractors.append(attractor)
        return attractor
    
    def get_space_summary(self) -> Dict:
        """Get summary of evolved latent space."""
        return {
            'n_dimensions': len(self.dimensions),
            'active_dimensions': sum(1 for d in self.dimensions if d.stability > 0),
            'semantic_attractors': len(self.semantic_attractors),
            'state_dim': len(self.current_state),
            'dimensional_births': sum(1 for d in self.dimensions if d.birth_time > 2),
            'dimensional_deaths': sum(1 for d in self.dimensions if d.stability == 0)
        }


class OntologicalSelfConstruction:
    """
    Phase 11: Ontological Self-Construction
    
    System learns what counts as self, not just dynamics of self.
    
    Key innovations:
      - Latent space plasticity (topology evolves)
      - Semantic attractors (meaning survives transformation)
      - Self-generated priors (not external)
      - Ontological metabolism (consume uncertainty → structure)
      - Emergent destabilization (from contradiction accumulation)
    """
    
    def __init__(self, latent_dim: int = 2):
        self.evolved_space = EvolvingLatentSpace(initial_dim=latent_dim)
        self.priors = SelfGeneratedPrior()
        self.metabolism = OntologicalMetabolism(latent_dim=latent_dim)
        self.contradictions = ContradictionAccumulator()
        
        # Self-state in evolved space
        self.self_state: np.ndarray = np.zeros(latent_dim)
        
        # History
        self.experience_history: List[np.ndarray] = []
        self.ontology_history: List[Dict] = []
        
    def update(self, experience: np.ndarray, 
               prediction_error: float = 0.0,
               world_state: Optional[np.ndarray] = None) -> Dict:
        """
        Update ontological state.
        
        System learns what counts as self.
        """
        # Store experience
        self.experience_history.append(experience.copy())
        if len(self.experience_history) > 500:
            self.experience_history = self.experience_history[-250:]
        
        # Add contradiction evidence
        semantic_conflict = self._detect_semantic_conflict(experience)
        attractor_load = len(self.evolved_space.semantic_attractors) / 10.0
        
        self.contradictions.add_contradiction(
            prediction_error=prediction_error,
            semantic_conflict=semantic_conflict,
            attractor_load=attractor_load
        )
        
        # Evolve latent space
        self.self_state = self.evolved_space.evolve(experience)
        
        # Emerge self-generated priors
        if len(self.experience_history) > 20:
            emerged = self.priors.emerge_prior(self.experience_history[-50:])
            if emerged:
                self.ontology_history.append({'type': 'prior_emergence', 'priors': emerged})
        
        # Ontological metabolism
        uncertainty = prediction_error + semantic_conflict * 0.1
        structure = self.metabolism.consume_uncertainty(uncertainty, experience)
        
        if structure:
            # Create semantic attractor from metabolized uncertainty
            self.evolved_space.create_semantic_attractor(
                pattern=experience,
                semantic_id=structure['id']
            )
            self.ontology_history.append(structure)
        
        # Check for emergent destabilization
        should_destabilize, pressure = self.contradictions.should_destabilize()
        
        result = {
            'self_state': self.self_state.copy(),
            'space_evolution': self.evolved_space.get_space_summary(),
            'should_destabilize': should_destabilize,
            'destabilization_pressure': pressure,
            'priors_emerged': len(self.priors.prior_structure),
            'structures_created': len(self.metabolism.structure_created),
            'semantic_attractors': len(self.evolved_space.semantic_attractors)
        }
        
        return result
    
    def _detect_semantic_conflict(self, experience: np.ndarray) -> int:
        """Detect semantic conflicts in experience."""
        conflict_count = 0
        
        # Check against existing semantic attractors
        for attractor in self.evolved_space.semantic_attractors:
            if not attractor.survives_transformation(experience - attractor.core_pattern):
                conflict_count += 1
        
        return conflict_count
    
    def emergent_destabilize(self) -> np.ndarray:
        """
        Emergent destabilization (not external control).
        
        Contradiction accumulation → phase transition from within.
        """
        if not self.contradictions.should_destabilize()[0]:
            return self.self_state
        
        # Destabilize by:
        # 1. Adding noise to core patterns
        # 2. Forcing dimension splits
        # 3. Merging semantic attractors
        
        # Add perturbation
        perturbation = np.random.randn(len(self.self_state)) * 0.5
        new_state = self.self_state + perturbation
        
        # Force dimension split if stable
        for dim in self.evolved_space.dimensions:
            if dim.stability > 0.5 and dim.plasticity > 0.3:
                self.evolved_space._split_dimension(dim.dim_id)
                break
        
        # Reset contradiction accumulator
        self.contradictions = ContradictionAccumulator()
        
        return new_state
    
    def get_ontological_state(self) -> Dict:
        """Get comprehensive ontological state."""
        return {
            'self_state_dim': len(self.self_state),
            'space_summary': self.evolved_space.get_space_summary(),
            'priors': self.priors.prior_structure,
            'metabolism': self.metabolism.get_metabolic_state(),
            'contradictions': self.contradictions.get_contradiction_summary(),
            'ontology_changes': len(self.ontology_history),
            'experience_history_length': len(self.experience_history)
        }


class OntologicalAgent:
    """
    Agent with ontological self-construction.
    
    NOT: learns dynamics INSIDE fixed ontology
    BUT: learns to CREATE and RESTRUCTURE ontology
    """
    
    def __init__(self, latent_dim: int = 2):
        self.ontology = OntologicalSelfConstruction(latent_dim=latent_dim)
        self.state = np.zeros(latent_dim)
        
    def step(self, action: np.ndarray, world_state: np.ndarray) -> Dict:
        """
        Take action and evolve ontology.
        """
        # Update state
        new_state = self.state + action * 0.1
        
        # Simulate world interaction
        world_interaction = world_state * 0.1
        new_state = new_state + world_interaction
        
        # Predict next state (for prediction error)
        predicted = self.ontology.evolved_space.dimensions[0].current_value if self.ontology.evolved_space.dimensions else 0
        prediction_error = abs(new_state[0] - predicted) if len(new_state) > 0 else 0
        
        # Add noise for realism
        prediction_error += np.random.random() * 0.1
        
        # Update ontological state
        result = self.ontology.update(new_state, prediction_error, world_state)
        
        # Emergent destabilization
        if result['should_destabilize']:
            self.state = self.ontology.emergent_destabilize()
        else:
            self.state = result['self_state']
        
        return result
    
    def run_episode(self, n_steps: int = 100) -> Dict:
        """Run episode with ontological self-construction."""
        print("\n  Running ontological self-construction episode:")
        
        results = []
        
        for step in range(n_steps):
            # Random action
            action = np.random.randn(len(self.state)) * 0.2
            
            # Random world state
            world = np.random.randn(2) * 0.5
            
            # Step
            result = self.step(action, world)
            results.append(result)
            
            if step % 20 == 0:
                space = result['space_evolution']
                print(f"    Step {step}: dims={space['n_dimensions']}, "
                      f"attractors={result['semantic_attractors']}, "
                      f"pressure={result['destabilization_pressure']:.3f}, "
                      f"structures={result['structures_created']}")
        
        return {
            'final_state': self.ontology.get_ontological_state(),
            'results': results
        }


def test_ontological_self_construction():
    """Test ontological self-construction."""
    print("\n" + "=" * 60)
    print("ONTOLOGICAL SELF-CONSTRUCTION TEST")
    print("=" * 60)
    
    agent = OntologicalAgent(latent_dim=2)
    
    # Run episode
    result = agent.run_episode(100)
    
    # Check evolution
    final = result['final_state']
    print("\n  Final ontological state:")
    print(f"    Self-state dimension: {final['self_state_dim']}")
    print(f"    Space evolution: {final['space_summary']}")
    print(f"    Prior structures: {len(final['priors'])}")
    print(f"    Metabolism: {final['metabolism']}")
    print(f"    Ontology changes: {final['ontology_changes']}")
    
    # Test semantic attractors
    print("\n  Semantic attractors:")
    for i, attractor in enumerate(agent.ontology.evolved_space.semantic_attractors[:3]):
        print(f"    Attractor {i}: stability={attractor.stability:.3f}, age={attractor.age}")
    
    # Test self-generated priors
    print("\n  Self-generated priors:")
    for key, value in list(final['priors'].items())[:3]:
        print(f"    {key}: {value[:2] if len(value) > 2 else value}")
    
    # Test emergent destabilization
    print("\n  Testing emergent destabilization:")
    contradictions = final['contradictions']
    print(f"    Should destabilize: {contradictions['should_destabilize']}")
    print(f"    Pressure: {contradictions['pressure']:.3f}")
    
    # Test dimension evolution
    print("\n  Dimension evolution:")
    for dim in agent.ontology.evolved_space.dimensions[:5]:
        print(f"    Dim {dim.dim_id}: role={dim.representational_role}, "
              f"stability={dim.stability:.3f}, plasticity={dim.plasticity:.3f}")


def test_latent_space_plasticity():
    """Test latent space topology evolution."""
    print("\n" + "=" * 60)
    print("LATENT SPACE PLASTICITY TEST")
    print("=" * 60)
    
    space = EvolvingLatentSpace(initial_dim=2)
    
    print("\n  Initial state:")
    print(f"    Dimensions: {len(space.dimensions)}")
    
    # Simulate experience that should trigger evolution
    print("\n  Simulating experience evolution:")
    
    for step in range(100):
        # Vary experiences to trigger splits
        if step < 30:
            experience = np.array([1.0 + np.random.random() * 0.5, 0.5])
        elif step < 60:
            experience = np.array([2.0 + np.random.random() * 0.5, 1.0])
        else:
            experience = np.array([1.0 + np.random.random(), 0.5 + np.random.random()])
        
        evolved = space.evolve(experience)
        
        if step % 30 == 0:
            summary = space.get_space_summary()
            print(f"    Step {step}: dims={summary['n_dimensions']}, "
                  f"active={summary['active_dimensions']}, "
                  f"state_dim={summary['state_dim']}")
    
    # Check for splits
    print("\n  Dimension analysis:")
    for dim in space.dimensions:
        print(f"    Dim {dim.dim_id}: role={dim.representational_role}, "
              f"birth={dim.birth_time}")
    
    print("\n  Semantic attractors:")
    for attr in space.semantic_attractors[:3]:
        print(f"    {attr.semantic_id}: stability={attr.stability:.3f}")


def test_ontological_metabolism():
    """Test ontological metabolism (uncertainty → structure)."""
    print("\n" + "=" * 60)
    print("ONTOLOGICAL METABOLISM TEST")
    print("=" * 60)
    
    metabolism = OntologicalMetabolism(latent_dim=2)
    
    print("\n  Testing uncertainty consumption:")
    
    for step in range(50):
        # Variable uncertainty
        uncertainty = np.random.random() * 2.0
        context = np.random.randn(2)
        
        structure = metabolism.consume_uncertainty(uncertainty, context)
        
        if structure:
            print(f"    Step {step}: Created structure {structure['id']} "
                  f"(uncertainty={uncertainty:.3f})")
    
    state = metabolism.get_metabolic_state()
    print(f"\n  Final metabolic state:")
    print(f"    Structures created: {state['structures_created']}")
    print(f"    Avg uncertainty: {state['avg_uncertainty']:.3f}")
    print(f"    Metabolic rate: {state['metabolic_rate']}")


def test_self_generated_priors():
    """Test self-generated priors."""
    print("\n" + "=" * 60)
    print("SELF-GENERATED PRIORS TEST")
    print("=" * 60)
    
    priors = SelfGeneratedPrior()
    
    # Generate experiences with hidden regularities
    experiences = []
    
    # Regularity 1: states cluster around [1, 1]
    for _ in range(30):
        experiences.append(np.array([1.0, 1.0]) + np.random.randn(2) * 0.2)
    
    # Regularity 2: states cluster around [-1, -1]
    for _ in range(30):
        experiences.append(np.array([-1.0, -1.0]) + np.random.randn(2) * 0.2)
    
    # Random noise
    for _ in range(20):
        experiences.append(np.random.randn(2) * 2)
    
    print("\n  Emerged priors:")
    emerged = priors.emerge_prior(experiences)
    
    for key, value in emerged.items():
        print(f"    {key}: {value[:2] if len(value) > 2 else value}")
    
    print(f"\n  Total prior structures: {len(priors.prior_structure)}")
    print(f"  Regularity clusters: {len(priors.regularity_clusters)}")


def compare_with_phase10():
    """Compare Phase 11 (Ontological) with Phase 10 (Generative)."""
    print("\n" + "=" * 60)
    print("PHASE 10 VS PHASE 11 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 10 (Generative Identity Dynamics):")
    print("    - q(self) = variational belief")
    print("    - P(S_t+1 | S_t, A_t, W_t) = learned")
    print("    - entropy(self_beliefs) = identity uncertainty")
    print("    - counterfactual = sampled from learned dynamics")
    print("    - attractor = fixed point in vector space")
    print("    - preferred_self = external (not emergent)")
    print("    - destabilize = external function")
    print("    - latent_dim = 2 (FIXED)")
    
    print("\n  Phase 11 (Ontological Self-Construction):")
    print("    - self = evolving latent topology")
    print("    - semantic attractors = meaning survives transformation")
    print("    - priors = self-generated from survival pressure")
    print("    - metabolism = consume uncertainty → create structure")
    print("    - destabilization = emergent from contradiction accumulation")
    print("    - latent space = PLASTIC (splitting/merging/death)")
    print("    - ontology = self-constructed (not external)")
    
    print("\n  Key architectural shifts:")
    print("    1. Fixed dimensions → Evolving topology")
    print("    2. Fixed points → Semantic attractors")
    print("    3. External priors → Self-generated priors")
    print("    4. Inference → Ontological metabolism")
    print("    5. External control → Emergent destabilization")
    print("    6. Learns INSIDE ontology → Learns to CREATE ontology")


if __name__ == "__main__":
    test_ontological_self_construction()
    test_latent_space_plasticity()
    test_ontological_metabolism()
    test_self_generated_priors()
    compare_with_phase10()
    
    print("\n" + "=" * 60)
    print("PHASE 11 - ONTOLOGICAL SELF-CONSTRUCTION")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: System learns dynamics of self
  To: System learns what counts as self (ontology creation)

CRITICAL INSIGHT:
  Phase 10: learns dynamics INSIDE fixed ontology
  Phase 11: learns to CREATE and RESTRUCTURE ontology itself

WHAT APPEARS:
  1. Latent space plasticity
     - concept splitting/merging
     - manifold folding
     - dimensional birth/death
     - representational drift
     
  2. Semantic attractors
     - "meaning survives transformation"
     - stable semantic invariants across change
     
  3. Self-generated priors
     - compression pressure
     - survival regularities
     - temporal continuity
     - memory persistence
     
  4. Ontological metabolism
     - consume uncertainty → create structure
     - epistemic self-maintenance
     
  5. Emergent destabilization
     - contradiction accumulation
     - phase transitions from within

CRITICAL PROBLEMS WITH PHASE 10:
  1. latent_dim = 2 (fixed, no plasticity)
  2. Gaussian assumption (smooth, unimodal)
  3. preferred_self = external (not emergent)
  4. destabilize_self() = external control (not emergent)
  5. Attractors = fixed points (not semantic invariants)
  6. Prior structure = hand-crafted (not self-generated)

PROGRESSION:
  Phase 8: Self-organizing cognition
  Phase 9: Recursive self-modeling
  Phase 10: Generative identity dynamics
  Phase 11: Ontological self-construction
  Phase 12: Autopoietic semantic closure
  Phase 13: Endogenous goal genesis
  Phase 14: Self-maintaining world simulation
  Phase 15: Proto-conscious epistemic metabolism

This is where the system starts:
  - Creating own latent semantics
  - Restructuring own representational space
  - Defining what counts as "agent"
  - Forming intrinsic ontologies
  
We are now at the boundary where:
  adding more modules stops working
  and architecture must start restructuring itself
""")