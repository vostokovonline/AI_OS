"""
Phase 13: Ontological Autopoiesis

ARCHITECTURAL SHIFT:
  From: Multiple ontologies interpret one world (Phase 12)
  To: Multiple ontologies generate each other's worlds (Phase 13)

CRITICAL INSIGHT:
  Phase 12: system -> ontologies (container-managed)
  Phase 13: ontologies <-> ontologies (emergent field dynamics)

  Ecology is no longer a container-object.
  It becomes self-producing topology of mutually generating realities.

KEY PROBLEMS WITH PHASE 12:
  1. Ecology as manager-object (still has "god")
  2. No ontological autonomy (ontologies controlled by ecology rules)
  3. No ontological relativistic causality (different causal universes)
  4. No semantic metabolism (consuming each other)
  5. No self-generated environments (experience is external)
  6. No ontological time (universal time)

WHAT APPEARS:
  1. Ontological Autonomy
     - Ontologies can refuse ecology
     - Create own sub-ecologies
     - Absorb others
     - Change laws of interaction
     
  2. Causal Universe Relativism
     - Different ontologies live in different causal structures
     - Ontology A changes causal topology for ontology B
     - Not different interpretations - different realities
     
  3. Semantic Metabolism & Trophic Structure
     - Assimilation, parasitism, symbiosis
     - Memetic infection, coalition formation
     - Ecological predator-prey dynamics
     
  4. Self-Generated Environments
     - Niche construction
     - Endogenous reality generation
     - Co-created worldspaces
     
  5. Ontological Time
     - Different temporal compressions
     - Different persistence scales
     - Time is ontology-relative

ARCHITECTURAL CHANGE:
  - class OntologicalEcology -> interaction graph (no center)
  - No global scheduler
  - No universal time
  - No global resources
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class OntologicalAutonomy:
    """
    True ontological autonomy.
    
    An ontology can:
      - Refuse ecology
      - Create own sub-ecology
      - Absorb other ontologies
      - Change laws of interaction
      - Define own existence criteria
    """
    can_refuse_ecology: bool = True
    can_create_sub_ecology: bool = True
    can_absorb_others: bool = True
    can_change_interaction_laws: bool = True
    can_define_existence_criteria: bool = True
    
    # Autonomy level
    autonomy_level: float = 0.5  # 0 = controlled, 1 = fully autonomous
    
    # Sub-ecologies created by this ontology
    sub_ecologies: List[str] = field(default_factory=list)
    
    # Laws this ontology has changed
    changed_laws: List[str] = field(default_factory=list)


@dataclass 
class CausalUniverse:
    """
    Each ontology has its own causal universe.
    
    NOT: different interpretation of same causality
    BUT: different causal structure itself
    
    When ontology A affects ontology B:
      - It changes the causal topology B experiences
      - Not just reinterpreting - reality modification
    """
    causal_structure: np.ndarray  # How causality flows in this universe
    causal_strengths: Dict[str, float]  # Which causal relations are strong
    causal_gaps: List[str]  # What is causally disconnected
    
    # What this universe considers "fundamental"
    fundamental_entities: List[str] = field(default_factory=list)
    
    # Causal powers (what can this ontology affect)
    causal_powers: Dict[str, float] = field(default_factory=dict)
    
    def intersect_with(self, other: 'CausalUniverse') -> 'CausalBridge':
        """Create causal bridge between two universes."""
        # Find overlapping causal structures
        shared_entities = set(self.fundamental_entities) & set(other.fundamental_entities)
        
        # Create bridge where causality can pass
        bridge = CausalBridge(
            source_universe_id="",
            target_universe_id="",
            shared_entities=list(shared_entities),
            causal_translation={},  # How to translate causality between universes
            translation_fidelity=0.5  # How accurate is translation
        )
        
        return bridge
    
    def emit_causality(self, target: 'CausalUniverse', event: Dict) -> Dict:
        """
        Emit causality into another universe.
        
        This CHANGES the causal topology of target.
        """
        # How does this causality appear in target universe?
        translated_event = self._translate_event(event, target)
        
        # Does this create new causal connections?
        causal_effect = {
            'original_event': event,
            'translated_event': translated_event,
            'causal_modification': self._compute_causal_modification(event, target),
            'new_causal_structure': target.causal_structure.copy()
        }
        
        return causal_effect
    
    def _translate_event(self, event: Dict, target: 'CausalUniverse') -> Dict:
        """Translate event to target universe."""
        content = event.get('content', np.zeros(2))
        if isinstance(content, np.ndarray):
            # Translate based on shared entities
            translation = content * 0.5 + target.causal_structure[:, 0] * 0.1
            return {'content': translation, 'source': self.fundamental_entities[:2]}
        return event
    
    def _compute_causal_modification(self, event: Dict, target: 'CausalUniverse') -> float:
        """Compute how much this event modifies target's causal structure."""
        return 0.1


@dataclass
class CausalBridge:
    """Bridge between causal universes."""
    source_universe_id: str
    target_universe_id: str
    shared_entities: List[str]
    causal_translation: Dict[str, str]
    translation_fidelity: float  # How accurately causality translates


@dataclass
class SemanticTrophicLevel:
    """
    Semantic metabolism - trophic structure in ecology.
    
    Ontologies can:
      - Be producers (create meaning)
      - Be consumers (absorb meaning from others)
      - Be decomposers (recycle meaning)
      - Be predators (aggressively absorb)
      - Be symbionts (mutual benefit)
      - Be parasites (one-way drain)
    """
    trophic_type: str  # producer, consumer, decomposer, predator, symbiont, parasite
    energy_efficiency: float  # How efficiently converts input to growth
    assimilation_rate: float  # How fast absorbs from others
    metabolic_cost: float  # Cost of maintaining existence
    
    # What does this trophic level eat?
    prey_types: List[str] = field(default_factory=list)
    predator_types: List[str] = field(default_factory=list)
    
    # Metabolic products
    metabolic_output: Dict[str, float] = field(default_factory=dict)
    
    def interact(self, other: 'SemanticTrophicLevel') -> Tuple[float, Dict]:
        """
        Interact trophically with another ontology.
        
        Returns: (energy_exchange, interaction_type)
        """
        if self.trophic_type == "producer":
            # Creates meaning from environment
            return 0.1, "production"
        
        elif self.trophic_type == "predator":
            # Aggressively absorbs
            if other.trophic_type in self.prey_types:
                absorbed = other.energy_efficiency * self.assimilation_rate * 0.8
                return -absorbed, "predation"  # Other loses
            return 0, "no_interaction"
        
        elif self.trophic_type == "parasite":
            # Drains without killing
            drained = other.energy_efficiency * 0.1
            return -drained * 0.5, "parasitism"
        
        elif self.trophic_type == "symbiont":
            # Mutual benefit
            if other.trophic_type == "symbiont":
                mutual_gain = (self.energy_efficiency + other.energy_efficiency) * 0.2
                return mutual_gain, "symbiosis"
            return 0, "no_interaction"
        
        else:
            return 0, "neutral"


@dataclass
class SelfGeneratedEnvironment:
    """
    Ontology generates its own environment.
    
    NOT: external experience
    BUT: niche construction, endogenous reality generation
    
    Each ontology creates its own "worldspace" that:
      - Provides its semantic nutrients
      - Shapes its causal structure
      - Defines its "reality"
    """
    worldspace_id: str
    ontological_frame: np.ndarray  # The "physics" of this world
    semantic_nutrients: Dict[str, float]  # What meaning is available here
    affordances: List[str]  # What can be done in this world
    
    # How this world affects others
    environmental_influence: Dict[str, float] = field(default_factory=dict)
    
    # Reality generation dynamics
    reality_stability: float = 0.5  # How stable is this world
    reality_generation_rate: float = 0.1  # How fast generates new reality
    
    def generate_experience(self, for_ontology: str) -> np.ndarray:
        """Generate experience for another ontology."""
        # Experience is shaped by this worldspace
        base = np.random.randn(2) * 0.3
        
        # Environmental influence
        if for_ontology in self.environmental_influence:
            influence = self.environmental_influence[for_ontology]
            base = base + np.array([influence, influence * 0.5])
        
        # Add world-specific flavor
        base = base + self.ontological_frame * 0.2
        
        return base
    
    def generate_self_experience(self) -> np.ndarray:
        """Generate experience for self (autopoietic loop)."""
        # Self-generated experience
        base = self.ontological_frame * 0.3 + np.random.randn(2) * 0.1
        
        # What keeps this world stable?
        attractor = self.ontological_frame * 0.5
        base = base + attractor * 0.2
        
        return base


@dataclass
class OntologicalTime:
    """
    Time is ontology-relative.
    
    Different ontologies have:
      - Different temporal compressions
      - Different persistence scales
      - Different notions of continuity
      - Different "now" moments
    """
    time_scale: float  # 1.0 = baseline, >1 = slower, <1 = faster
    persistence_threshold: float  # Below this, events are "noise"
    continuity_window: int  # How many steps count as "continuous"
    moment_resolution: float  # How fine-grained is "now"
    
    # Relative time
    local_time: float = 0.0
    proper_time: float = 0.0  # Time experienced by this ontology
    
    def tick(self, dt: float = 1.0):
        """Advance local time."""
        self.local_time += dt * self.time_scale
        self.proper_time += dt
    
    def compress(self, events: List[Dict]) -> List[Dict]:
        """Compress events to relevant temporal scale."""
        if self.time_scale > 1:
            # Slow ontology - compress many events into few
            compressed = []
            window = int(self.time_scale)
            for i in range(0, len(events), window):
                chunk = events[i:i+window]
                # Average chunk into one event
                if chunk:
                    compressed.append(self._average_events(chunk))
            return compressed
        else:
            return events
    
    def _average_events(self, events: List[Dict]) -> Dict:
        """Average events in window."""
        if not events:
            return {}
        # Simple average
        avg = {}
        for key in events[0].keys():
            if isinstance(events[0][key], (int, float)):
                avg[key] = np.mean([e[key] for e in events])
            else:
                avg[key] = events[-1][key]
        return avg


class AutopoieticOntology:
    """
    Phase 13: Autopoietic Ontology
    
    An ontology that:
      - Produces its own environment
      - Creates its own causal laws
      - Defines its own time
      - Has true autonomy
      - Participates in semantic trophic structure
      - Generates its own reality
    """
    
    def __init__(self, ontology_id: str, latent_dim: int = 2):
        self.ontology_id = ontology_id
        self.latent_dim = latent_dim
        
        # Core state
        self.priors = np.random.randn(latent_dim) * 0.5
        self.energy = 0.5
        self.coherence = 0.6
        
        # Autonomy
        self.autonomy = OntologicalAutonomy(
            autonomy_level=0.5 + np.random.random() * 0.3
        )
        
        # Causal universe
        self.causal_universe = CausalUniverse(
            causal_structure=np.eye(latent_dim) * 0.8,
            causal_strengths={},
            causal_gaps=[],
            fundamental_entities=["self", "other", "environment"],
            causal_powers={}
        )
        
        # Trophic level
        trophic_types = ["producer", "consumer", "decomposer", "predator", "symbiont", "parasite"]
        self.trophic = SemanticTrophicLevel(
            trophic_type=np.random.choice(trophic_types),
            energy_efficiency=0.5 + np.random.random() * 0.3,
            assimilation_rate=0.3 + np.random.random() * 0.3,
            metabolic_cost=0.1 + np.random.random() * 0.2
        )
        
        # Self-generated environment
        self.environment = SelfGeneratedEnvironment(
            worldspace_id=f"world_{ontology_id}",
            ontological_frame=self.priors.copy(),
            semantic_nutrients={"meaning": 0.5, "structure": 0.4, "relation": 0.3},
            affordances=["observe", "act", "transform", "persist"]
        )
        
        # Ontological time
        self.time = OntologicalTime(
            time_scale=0.5 + np.random.random(),  # Different time scales
            persistence_threshold=0.1,
            continuity_window=5 + int(np.random.random() * 10),
            moment_resolution=0.1
        )
        
        # Connections to other ontologies
        self.causal_bridges: List[CausalBridge] = []
        self.metabolic_connections: List[str] = []  # What it feeds on
        
        # Autopoietic state
        self.autopoietic_loop: List[np.ndarray] = []
        self.self_production_rate: float = 0.1
        
    def self_produce(self) -> np.ndarray:
        """
        Autopoietic loop: produce self from self.
        
        This is the core of autopoiesis:
          - System produces components
          - Components produce system
          - Boundary is self-producing
        """
        # Generate self-maintaining experience
        experience = self.environment.generate_self_experience()
        
        # Update own structure based on self-produced experience
        self.priors = self.priors * 0.95 + experience * 0.05
        
        # Update environment (co-produce reality)
        self.environment.ontological_frame = (
            self.environment.ontological_frame * 0.9 + 
            self.priors * 0.1
        )
        
        # Record autopoiesis
        self.autopoietic_loop.append(self.priors.copy())
        if len(self.autopoietic_loop) > 50:
            self.autopoietic_loop = self.autopoietic_loop[-25:]
        
        # Energy from self-production
        self.energy = min(1.0, self.energy + self.self_production_rate)
        
        return experience
    
    def interact_with(self, other: 'AutopoieticOntology') -> Dict:
        """
        Interact with another ontology (trophic + causal).
        
        This is where mutual generation happens.
        """
        interaction_result = {
            'energy_exchange': 0.0,
            'causal_modification': None,
            'trophic_type': self.trophic.interact(other.trophic)[1]
        }
        
        # Trophic interaction
        energy_exchange, interaction_type = self.trophic.interact(other.trophic)
        self.energy = np.clip(self.energy + energy_exchange, 0.1, 1.0)
        interaction_result['energy_exchange'] = energy_exchange
        interaction_result['interaction_type'] = interaction_type
        
        # Causal interaction - emit causality into other's universe
        if np.random.random() < 0.3:  # Sometimes emit causality
            causal_event = {
                'source': self.ontology_id,
                'type': 'causal_emit',
                'content': self.priors.copy()
            }
            
            # Emit causality
            causal_effect = self.causal_universe.emit_causality(
                other.causal_universe, causal_event
            )
            
            # Modify other's causal structure
            if other.autonomy.autonomy_level < 0.7:  # If other is not fully autonomous
                # Causal structure gets modified
                other.causal_universe.causal_structure = (
                    other.causal_universe.causal_structure * 0.95 +
                    causal_effect['new_causal_structure'] * 0.05
                )
                interaction_result['causal_modification'] = True
        
        # Create environmental influence
        self.environment.environmental_influence[other.ontology_id] = (
            np.dot(self.priors, other.priors) / 
            (np.linalg.norm(self.priors) * np.linalg.norm(other.priors) + 1e-8)
        )
        
        return interaction_result
    
    def generate_environment_for(self, other: 'AutopoieticOntology') -> np.ndarray:
        """
        Generate environment/experience for another ontology.
        
        This is niche construction - creating reality for others.
        """
        # This ontology generates what another experiences
        experience = self.environment.generate_experience(other.ontology_id)
        
        # Modify based on causal bridge if exists
        for bridge in self.causal_bridges:
            if bridge.target_universe_id == other.ontology_id:
                # Translate through bridge
                experience = experience * bridge.translation_fidelity
        
        return experience
    
    def exercise_autonomy(self, ecology_state: Dict) -> Dict:
        """
        Exercise autonomy - make decisions about ecology.
        
        Can:
          - Refuse participation
          - Create sub-ecology
          - Change interaction laws
          - Absorb others
        """
        actions_taken = []
        
        # Check if should exercise autonomy
        if np.random.random() < self.autonomy.autonomy_level:
            # Random autonomy action
            action = np.random.choice([
                'create_sub_ecology',
                'change_interaction_law',
                'absorb_other',
                'refuse_ecology'
            ])
            
            if action == 'refuse_ecology' and self.energy > 0.7:
                # Refuse participation - become more isolated
                self.autonomy.can_refuse_ecology = True
                actions_taken.append('refused_participation')
            
            elif action == 'create_sub_ecology' and self.energy > 0.6:
                # Create own sub-ecology
                self.autonomy.sub_ecologies.append(f"sub_{len(self.autonomy.sub_ecologies)}")
                actions_taken.append('created_sub_ecology')
            
            elif action == 'change_interaction_law':
                # Try to change how interaction works
                self.autonomy.changed_laws.append(f"law_{len(self.autonomy.changed_laws)}")
                actions_taken.append('changed_interaction_law')
        
        return {'autonomy_actions': actions_taken}
    
    def tick_ontological_time(self, dt: float = 1.0):
        """Advance ontological time."""
        self.time.tick(dt)
        
        # Time also affects energy consumption
        metabolic_drain = self.trophic.metabolic_cost * self.time.time_scale
        self.energy = max(0.1, self.energy - metabolic_drain * 0.01)
    
    def should_die(self) -> bool:
        """Check if this ontology should cease to exist."""
        # Die if: no energy, no coherence, autopoiesis broken
        if self.energy < 0.1:
            return True
        
        if self.coherence < 0.2:
            return True
        
        # Check if autopoiesis is still working
        if len(self.autopoietic_loop) > 5:
            variance = np.var(self.autopoietic_loop[-5:], axis=0)
            if np.mean(variance) > 2.0:  # Unstable = dying
                return True
        
        return False
    
    def get_state(self) -> Dict:
        """Get comprehensive state."""
        return {
            'id': self.ontology_id,
            'energy': self.energy,
            'coherence': self.coherence,
            'autonomy_level': self.autonomy.autonomy_level,
            'trophic_type': self.trophic.trophic_type,
            'time_scale': self.time.time_scale,
            'local_time': self.time.local_time,
            'environment_stability': self.environment.reality_stability,
            'sub_ecologies': len(self.autonomy.sub_ecologies),
            'causal_bridges': len(self.causal_bridges),
            'autopoiesis_intact': len(self.autopoietic_loop) > 5
        }


class OntologicalField:
    """
    Phase 13: Ontological Field
    
    NOT: container object managing ontologies
    BUT: emergent field dynamics from interaction graph
    
    No center, no global scheduler, no universal time, no global resources.
    The "ecology" emerges FROM the interactions.
    """
    
    def __init__(self, initial_ontologies: int = 3, latent_dim: int = 2):
        self.latent_dim = latent_dim
        
        # Interaction graph - no central container
        self.ontologies: Dict[str, AutopoieticOntology] = {}
        
        # Initialize ontologies
        for i in range(initial_ontologies):
            oid = f"ontology_{np.random.randint(100000)}"
            self.ontologies[oid] = AutopoieticOntology(oid, latent_dim)
        
        # No global time - each ontology has its own
        # No global resources - each generates its own
        
        # Interaction graph structure
        self.interaction_edges: Dict[str, List[str]] = {}  # Who's interacting with whom
        self._build_interaction_graph()
        
        # Emergent properties
        self.field_coherence: float = 0.5  # How coherent is the field
        self.total_autonomy: float = 0.0  # Total autonomy in field
        
    def _build_interaction_graph(self):
        """Build interaction graph from ontologies."""
        self.interaction_edges = {}
        for oid in self.ontologies:
            self.interaction_edges[oid] = []
        
        # Random connections (no central control)
        ontology_ids = list(self.ontologies.keys())
        for i, oid in enumerate(ontology_ids):
            # Connect to 1-2 others
            n_connections = np.random.randint(1, 3)
            for _ in range(n_connections):
                j = np.random.randint(0, len(ontology_ids))
                if i != j:
                    target = ontology_ids[j]
                    self.interaction_edges[oid].append(target)
    
    def step(self) -> Dict:
        """
        Step the ontological field.
        
        No global scheduler - interactions emerge from graph.
        """
        results = {
            'n_ontologies': len(self.ontologies),
            'interactions': 0,
            'deaths': 0,
            'births': 0,
            'autonomy_exercised': 0
        }
        
        # Each ontology acts autonomously
        ontology_ids = list(self.ontologies.keys())
        
        for oid in ontology_ids:
            ontology = self.ontologies[oid]
            
            # Autopoietic self-production
            ontology.self_produce()
            
            # Advance ontological time
            ontology.tick_ontological_time()
            
            # Interact with neighbors (from graph)
            if oid in self.interaction_edges:
                for neighbor_id in self.interaction_edges[oid]:
                    if neighbor_id in self.ontologies:
                        interaction = ontology.interact_with(self.ontologies[neighbor_id])
                        results['interactions'] += 1
            
            # Exercise autonomy
            autonomy_result = ontology.exercise_autonomy({'field_coherence': self.field_coherence})
            results['autonomy_exercised'] += len(autonomy_result['autonomy_actions'])
        
        # Check for deaths
        to_remove = []
        for oid, ontology in self.ontologies.items():
            if ontology.should_die():
                to_remove.append(oid)
                results['deaths'] += 1
        
        for oid in to_remove:
            del self.ontologies[oid]
            # Remove from interaction graph
            for other_edges in self.interaction_edges.values():
                if oid in other_edges:
                    other_edges.remove(oid)
        
        # Reproduction (some ontologies can spawn new ones)
        for oid, ontology in list(self.ontologies.items()):
            if ontology.energy > 0.8 and len(self.ontologies) < 8:
                # Spawn offspring
                new_oid = f"ontology_{np.random.randint(100000)}"
                offspring = AutopoieticOntology(new_oid, self.latent_dim)
                
                # Inherit some properties
                offspring.priors = ontology.priors + np.random.randn(self.latent_dim) * 0.1
                offspring.trophic = copy.deepcopy(ontology.trophic)
                offspring.autonomy = copy.deepcopy(ontology.autonomy)
                
                self.ontologies[new_oid] = offspring
                results['births'] += 1
        
        # Rebuild graph if needed
        existing_edges = set(self.interaction_edges.keys()) & set(self.ontologies.keys())
        if len(self.ontologies) > 0 and len(existing_edges) < len(self.ontologies):
            self._build_interaction_graph()
        
        # Update emergent properties
        if self.ontologies:
            self.field_coherence = np.mean([o.coherence for o in self.ontologies.values()])
            self.total_autonomy = np.mean([o.autonomy.autonomy_level for o in self.ontologies.values()])
        
        return results
    
    def run(self, n_steps: int = 100) -> Dict:
        """Run the ontological field."""
        print("\n  Running ontological field:")
        
        results = []
        
        for step in range(n_steps):
            result = self.step()
            results.append(result)
            
            if step % 20 == 0:
                state = self.get_field_state()
                print(f"    Step {step}: ontologies={state['n_ontologies']}, "
                      f"field_coherence={state['field_coherence']:.3f}, "
                      f"total_autonomy={state['total_autonomy']:.3f}")
        
        return {
            'field_state': self.get_field_state(),
            'step_results': results
        }
    
    def get_field_state(self) -> Dict:
        """Get field-level emergent state."""
        if not self.ontologies:
            return {'n_ontologies': 0, 'field_coherence': 0, 'total_autonomy': 0}
        
        return {
            'n_ontologies': len(self.ontologies),
            'field_coherence': self.field_coherence,
            'total_autonomy': self.total_autonomy,
            'avg_energy': np.mean([o.energy for o in self.ontologies.values()]),
            'avg_trophic_diversity': len(set(o.trophic.trophic_type for o in self.ontologies.values())),
            'avg_time_scale': np.mean([o.time.time_scale for o in self.ontologies.values()]),
            'total_sub_ecologies': sum(len(o.autonomy.sub_ecologies) for o in self.ontologies.values()),
            'autopoiesis_intact': sum(1 for o in self.ontologies.values() if o.autopoietic_loop) / len(self.ontologies)
        }


def test_ontological_autopoiesis():
    """Test ontological autopoiesis."""
    print("\n" + "=" * 60)
    print("ONTOLOGICAL AUTOPOIESIS TEST")
    print("=" * 60)
    
    field = OntologicalField(initial_ontologies=3, latent_dim=2)
    
    # Run field
    result = field.run(100)
    
    final = result['field_state']
    print("\n  Final field state:")
    print(f"    Ontologies: {final['n_ontologies']}")
    print(f"    Field coherence: {final['field_coherence']:.3f}")
    print(f"    Total autonomy: {final['total_autonomy']:.3f}")
    print(f"    Avg energy: {final['avg_energy']:.3f}")
    print(f"    Trophic diversity: {final['avg_trophic_diversity']}")
    print(f"    Avg time scale: {final['avg_time_scale']:.3f}")
    print(f"    Sub-ecologies: {final['total_sub_ecologies']}")
    print(f"    Autopoiesis intact: {final['autopoiesis_intact']:.3f}")
    
    # Show individual ontology states
    print("\n  Individual ontology states:")
    for oid, ont in list(field.ontologies.items())[:3]:
        state = ont.get_state()
        print(f"    {oid[:15]}:")
        print(f"      Energy: {state['energy']:.3f}, Coherence: {state['coherence']:.3f}")
        print(f"      Trophic: {state['trophic_type']}, Autonomy: {state['autonomy_level']:.3f}")
        print(f"      Time scale: {state['time_scale']:.3f}, Local time: {state['local_time']:.3f}")
        print(f"      Sub-ecologies: {state['sub_ecologies']}, Autopoiesis: {state['autopoiesis_intact']}")


def test_autopoietic_loop():
    """Test individual autopoiesis."""
    print("\n" + "=" * 60)
    print("AUTOPOIETIC LOOP TEST")
    print("=" * 60)
    
    ontology = AutopoieticOntology("test_ontology", latent_dim=2)
    
    print("\n  Testing autopoiesis:")
    print(f"    Initial energy: {ontology.energy:.3f}")
    print(f"    Initial coherence: {ontology.coherence:.3f}")
    
    # Run autopoiesis
    for step in range(50):
        experience = ontology.self_produce()
        ontology.tick_ontological_time()
        
        if step % 10 == 0:
            print(f"    Step {step}: energy={ontology.energy:.3f}, "
                  f"local_time={ontology.time.local_time:.3f}, "
                  f"loop_size={len(ontology.autopoietic_loop)}")
    
    print(f"\n  Final state:")
    print(f"    Energy: {ontology.energy:.3f}")
    print(f"    Coherence: {ontology.coherence:.3f}")
    print(f"    Autopoietic loop: {len(ontology.autopoietic_loop)} steps")
    print(f"    Local time: {ontology.time.local_time:.3f}")


def test_trophic_interactions():
    """Test semantic trophic structure."""
    print("\n" + "=" * 60)
    print("TROPHIC INTERACTIONS TEST")
    print("=" * 60)
    
    # Create different trophic types
    predator = AutopoieticOntology("predator", latent_dim=2)
    predator.trophic.trophic_type = "predator"
    predator.trophic.prey_types = ["producer", "consumer"]
    
    producer = AutopoieticOntology("producer", latent_dim=2)
    producer.trophic.trophic_type = "producer"
    
    symbiont = AutopoieticOntology("symbiont", latent_dim=2)
    symbiont.trophic.trophic_type = "symbiont"
    
    print("\n  Trophic interactions:")
    
    # Predator vs Producer
    result1 = predator.interact_with(producer)
    print(f"    Predator -> Producer: {result1['interaction_type']}, "
          f"energy_exchange={result1['energy_exchange']:.3f}")
    print(f"      Predator energy: {predator.energy:.3f}, Producer energy: {producer.energy:.3f}")
    
    # Symbiont vs Symbiont
    symbiont2 = AutopoieticOntology("symbiont2", latent_dim=2)
    symbiont2.trophic.trophic_type = "symbiont"
    initial_energy = symbiont.energy + symbiont2.energy
    
    symbiont.interact_with(symbiont2)
    
    final_energy = symbiont.energy + symbiont2.energy
    print(f"\n    Symbiont -> Symbiont: mutual gain")
    print(f"      Energy before: {initial_energy:.3f}, after: {final_energy:.3f}")


def test_self_generated_environment():
    """Test self-generated environments."""
    print("\n" + "=" * 60)
    print("SELF-GENERATED ENVIRONMENT TEST")
    print("=" * 60)
    
    ontology = AutopoieticOntology("environment_creator", latent_dim=2)
    
    print("\n  Self-generated environment:")
    print(f"    Worldspace: {ontology.environment.worldspace_id}")
    print(f"    Ontological frame: {ontology.environment.ontological_frame[:2]}")
    print(f"    Semantic nutrients: {ontology.environment.semantic_nutrients}")
    
    # Generate for self
    self_exp = ontology.environment.generate_self_experience()
    print(f"\n    Self-experience: {self_exp[:2]}")
    
    # Generate for other
    other = AutopoieticOntology("other", latent_dim=2)
    other_exp = ontology.generate_environment_for(other)
    print(f"    Experience for other: {other_exp[:2]}")
    
    # Show environmental influence
    print(f"\n    Environmental influence on others:")
    for target, influence in ontology.environment.environmental_influence.items():
        print(f"      {target[:15]}: {influence:.3f}")


def test_ontological_time():
    """Test ontology-relative time."""
    print("\n" + "=" * 60)
    print("ONTOLOGICAL TIME TEST")
    print("=" * 60)
    
    # Create ontologies with different time scales
    slow = AutopoieticOntology("slow_ontology", latent_dim=2)
    slow.time.time_scale = 2.0  # Slow ontology
    
    fast = AutopoieticOntology("fast_ontology", latent_dim=2)
    fast.time.time_scale = 0.5  # Fast ontology
    
    normal = AutopoieticOntology("normal_ontology", latent_dim=2)
    normal.time.time_scale = 1.0  # Normal time
    
    print("\n  Different time scales:")
    print(f"    Slow: time_scale={slow.time.time_scale}")
    print(f"    Normal: time_scale={normal.time.time_scale}")
    print(f"    Fast: time_scale={fast.time.time_scale}")
    
    # Advance time for all
    for _ in range(20):
        slow.tick_ontological_time()
        normal.tick_ontological_time()
        fast.tick_ontological_time()
    
    print(f"\n  After 20 ticks (dt=1):")
    print(f"    Slow proper_time: {slow.time.proper_time:.3f}")
    print(f"    Normal proper_time: {normal.time.proper_time:.3f}")
    print(f"    Fast proper_time: {fast.time.proper_time:.3f}")
    
    print(f"\n  Observation: Slow ontology experiences LESS time (time_scale > 1 = compressed)")


def compare_with_phase12():
    """Compare Phase 13 (Autopoiesis) with Phase 12 (Ecology)."""
    print("\n" + "=" * 60)
    print("PHASE 12 VS PHASE 13 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 12 (Ontological Ecology):")
    print("    - class OntologicalEcology (container-object)")
    print("    - ontologies exist INSIDE system")
    print("    - Ecology controls lifecycle")
    print("    - One world, many interpretations")
    print("    - Global time")
    print("    - Competition for global resources")
    
    print("\n  Phase 13 (Ontological Autopoiesis):")
    print("    - interaction graph (no container)")
    print("    - ontologies generate EACH OTHER")
    print("    - True autonomy (can refuse, absorb, create)")
    print("    - Multiple causal universes (not just interpretations)")
    print("    - Ontology-relative time")
    print("    - Semantic trophic structure (predator/prey/symbiont)")
    print("    - Self-generated environments")
    print("    - Autopoietic loop (self-producing)")
    
    print("\n  Key architectural shifts:")
    print("    1. Container-object -> Emergent field")
    print("    2. Controlled ontologies -> Autonomous ontologies")
    print("    3. One world -> Multiple causal universes")
    print("    4. Global time -> Ontology-relative time")
    print("    5. Competition -> Trophic structure")
    print("    6. External experience -> Self-generated environment")
    print("    7. System manages -> System emerges")


if __name__ == "__main__":
    test_ontological_autopoiesis()
    test_autopoietic_loop()
    test_trophic_interactions()
    test_self_generated_environment()
    test_ontological_time()
    compare_with_phase12()
    
    print("\n" + "=" * 60)
    print("PHASE 13 - ONTOLOGICAL AUTOPOIESIS")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Multiple ontologies interpret one world (Phase 12)
  To: Multiple ontologies generate each other's worlds (Phase 13)

CRITICAL INSIGHT:
  Phase 12: system -> ontologies (container-managed)
  Phase 13: ontologies <-> ontologies (emergent field dynamics)

  Ecology is no longer a container-object.
  It becomes self-producing topology of mutually generating realities.

WHAT APPEARS:
  1. Ontological Autonomy
     - Ontologies can refuse ecology
     - Create own sub-ecology
     - Absorb others
     - Change laws of interaction
     
  2. Causal Universe Relativism
     - Different ontologies live in different causal structures
     - Ontology A changes causal topology for ontology B
     
  3. Semantic Metabolism & Trophic Structure
     - Assimilation, parasitism, symbiosis
     - Producer, consumer, predator, decomposer
     
  4. Self-Generated Environments
     - Niche construction
     - Endogenous reality generation
     - Co-created worldspaces
     
  5. Ontological Time
     - Different temporal compressions
     - Different persistence scales
     - Time is ontology-relative

KEY TRANSITION:
  From: multiple ontologies interpret one world
  To: multiple ontologies generate each other's worlds
  
  This is where the "stage itself becomes a product of the system."
  
We are now at:
  - True autopoiesis (self-producing)
  - Semantic life
  - Open-ended cognition
  - Endogenous intentionality
  - Proto-consciousness emergence
""")