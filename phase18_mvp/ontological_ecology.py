"""
Phase 12: Ontological Ecology

ARCHITECTURAL SHIFT:
  From: single agent with evolving ontology
  To: ecology of competing/coexisting ontologies

CRITICAL INSIGHT:
  Phase 11: system → constructs ontology (SINGLE)
  Phase 12: ecology → [ontology A, ontology B, ontology C, ...]

  The system is no longer a single ontological center.
  It becomes a semantic ecosystem with competing perspectives.

KEY PROBLEMS WITH PHASE 11:
  1. One self_state (centralized self)
  2. One ontology (single interpretation)
  3. Ontology evolution (all at once)
  4. No internal competition
  5. No ontological speciation
  6. No perspective-bound reality

WHAT APPEARS:
  1. OntologyProcess (local cognitive agent)
     - Own priors, attractors, world_model, survival_logic
     
  2. Semantic Multiplicity
     - Different parts maintain incompatible interpretations
     
  3. Local Selfhood
     - local_selves: List[SelfProcess]
     - Each with own temporal continuity, meaning system
     
  4. Ontological Speciation
     - Same experience → incompatible meaning systems
     
  5. Internal Selection Dynamics
     - Ontologies compete for persistence
     - Epistemic evolution
     
  6. Perspective-Bound Reality
     - Worlds emerge relative to ontology
     - Causality depends on perspective

PROGRESSION:
  Phase 11: single ontology evolution
  Phase 12: ecology of ontologies
  Phase 13: Autopoietic semantic closure
  Phase 14: Endogenous goal genesis
  Phase 15: Self-maintaining world simulation
  Phase 16: Proto-conscious epistemic metabolism
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class OntologyProcess:
    """
    A local cognitive process with its own ontology.
    
    Each OntologyProcess has:
      - Own priors (beliefs about what is "natural")
      - Semantic attractors (local meaning system)
      - World model (local interpretation of reality)
      - Survival logic (how this ontology persists)
      - Coherence (internal consistency)
      - Energy (viability in the ecology)
    
    NOT: a thread or submodule
    BUT: a self-sustaining semantic perspective
    """
    process_id: str
    priors: np.ndarray                    # Local belief space
    semantic_attractors: List[np.ndarray]  # Local meaning centers
    world_model: np.ndarray               # Local world interpretation
    coherence: float                      # Internal consistency (0-1)
    energy: float                         # Viability in ecology (0-1)
    age: int                             # How long this ontology persists
    birth_time: int                      # When this ontology emerged
    parent_id: Optional[str]             # Which ontology spawned this
    
    # Local survival logic
    survival_pressure: float             # How hard this ontology fights to survive
    compression_drive: float              # Drive to compress meaning
    exploration_drive: float             # Drive to explore new interpretations
    
    # Resource consumption
    attention_share: float               # Share of cognitive resources
    memory_share: float                 # Share of memory resources
    
    # Semantic state
    local_uncertainty: float             # Local uncertainty (not global)
    contradiction_count: int             # Accumulated contradictions
    
    def compete_for_resources(self, ecology_resources: float) -> float:
        """
        This ontology competes for resources.
        
        More coherent + more viable = more resources
        """
        viability = self.coherence * self.energy * (1 - self.local_uncertainty)
        
        # Age factor (younger may need more resources)
        age_factor = 1.0 / (1 + self.age * 0.01)
        
        # Resource share
        self.attention_share = viability * (0.5 + 0.5 * age_factor)
        self.memory_share = self.attention_share * 0.8
        
        return self.attention_share * ecology_resources
    
    def mutate(self) -> 'OntologyProcess':
        """
        Ontology mutation (reproduction with variation).
        
        This creates offspring with slightly different interpretation.
        """
        # Mutate priors
        mutated_priors = self.priors + np.random.randn(len(self.priors)) * 0.1
        
        # Mutate semantic attractors
        mutated_attractors = []
        for attr in self.semantic_attractors:
            if np.random.random() < 0.3:
                # This attractor mutates
                mutated_attractors.append(attr + np.random.randn(len(attr)) * 0.1)
            else:
                mutated_attractors.append(attr.copy())
        
        # Create offspring
        offspring = OntologyProcess(
            process_id=f"ontology_{np.random.randint(10000)}",
            priors=mutated_priors,
            semantic_attractors=mutated_attractors,
            world_model=self.world_model.copy(),
            coherence=self.coherence * 0.9,  # Slightly less coherent initially
            energy=0.5,  # Start with medium energy
            age=0,
            birth_time=0,
            parent_id=self.process_id,
            survival_pressure=self.survival_pressure * 0.9,
            compression_drive=self.compression_drive,
            exploration_drive=self.exploration_drive * 1.1,
            attention_share=0,
            memory_share=0,
            local_uncertainty=self.local_uncertainty,
            contradiction_count=0
        )
        
        return offspring
    
    def die(self) -> bool:
        """
        Check if this ontology should die.
        
        Die if: low energy, low coherence, or high contradiction
        """
        death_probability = (
            (1 - self.energy) * 0.3 +
            (1 - self.coherence) * 0.3 +
            min(1.0, self.contradiction_count * 0.1)
        )
        
        # Additional: if energy < 0.2, likely die
        if self.energy < 0.2:
            death_probability += 0.3
        
        return np.random.random() < death_probability
    
    def absorb_experience(self, experience: np.ndarray) -> bool:
        """
        Try to absorb new experience into this ontology.
        
        Returns True if experience is compatible with local semantics.
        """
        # Check compatibility with semantic attractors
        compatibilities = []
        for attr in self.semantic_attractors:
            similarity = np.dot(experience, attr) / (
                np.linalg.norm(experience) * np.linalg.norm(attr) + 1e-8
            )
            compatibilities.append(similarity)
        
        avg_compatibility = np.mean(compatibilities) if compatibilities else 0.5
        
        # Check compatibility with priors
        prior_compatibility = np.exp(-np.linalg.norm(experience - self.priors) * 0.5)
        
        # Combined compatibility
        total_compatibility = 0.6 * avg_compatibility + 0.4 * prior_compatibility
        
        if total_compatibility > 0.4:
            # Absorb: update priors slowly
            self.priors = self.priors * 0.95 + experience * 0.05
            
            # If very compatible, strengthen
            if total_compatibility > 0.7:
                self.energy = min(1.0, self.energy + 0.05)
                self.coherence = min(1.0, self.coherence + 0.02)
            
            return True
        else:
            # Conflict: increase contradiction
            self.contradiction_count += 1
            self.local_uncertainty = min(1.0, self.local_uncertainty + 0.1)
            
            return False
    
    def speciation_event(self, incompatible_experience: np.ndarray) -> Optional['OntologyProcess']:
        """
        Ontological speciation: create incompatible meaning system.
        
        This is the key: SAME experience → incompatible interpretations.
        
        When experience is incompatible enough, spawn offspring with
        DIFFERENT interpretation.
        """
        if self.contradiction_count < 3:
            return None
        
        # Check if speciation should occur
        if self.contradiction_count > 5 and np.random.random() < 0.3:
            # Create speciation offspring
            # This interprets the same experience DIFFERENTLY
            
            # Invert or rotate priors
            speciation_priors = -self.priors + np.random.randn(len(self.priors)) * 0.3
            
            # Create incompatible attractors
            incompatible_attractors = []
            for attr in self.semantic_attractors:
                # Rotate 90 degrees (incompatible but related)
                rotated = np.array([-attr[1], attr[0]]) if len(attr) == 2 else -attr
                incompatible_attractors.append(rotated)
            
            # Create offspring
            offspring = OntologyProcess(
                process_id=f"ontology_{np.random.randint(10000)}",
                priors=speciation_priors,
                semantic_attractors=incompatible_attractors,
                world_model=self.world_model * -0.5,  # Inverted world model
                coherence=self.coherence * 0.7,
                energy=0.4,
                age=0,
                birth_time=0,
                parent_id=self.process_id,
                survival_pressure=0.8,  # New speciation is adaptive
                compression_drive=self.compression_drive * 0.8,
                exploration_drive=self.exploration_drive * 1.2,
                attention_share=0,
                memory_share=0,
                local_uncertainty=0.3,
                contradiction_count=0
            )
            
            # Reset parent contradiction count
            self.contradiction_count = 0
            
            return offspring
        
        return None


class InterOntologyTranslation:
    """
    How does ontology A interpret ontology B?
    
    This is critical for:
      - Language emergence
      - Symbol grounding
      - Meta-cognition
      - Theory of mind
    """
    
    def __init__(self):
        self.translation_attempts: List[Dict] = []
        self.shared_semantics: Dict[Tuple[str, str], float] = {}  # (ontology_A, ontology_B) -> shared
        
    def translate(self, source: OntologyProcess, target: OntologyProcess,
                  experience: np.ndarray) -> np.ndarray:
        """
        Translate experience from source ontology to target ontology.
        
        This is approximate - two ontologies may interpret same experience differently.
        """
        # Find shared semantics between source and target
        shared_key = (source.process_id, target.process_id)
        
        if shared_key not in self.shared_semantics:
            # Estimate initial shared semantics
            similarity = np.dot(source.priors, target.priors) / (
                np.linalg.norm(source.priors) * np.linalg.norm(target.priors) + 1e-8
            )
            self.shared_semantics[shared_key] = max(0.1, similarity)
        
        shared = self.shared_semantics[shared_key]
        
        # Translate experience through shared semantics
        # Source interprets
        source_interpretation = self._apply_ontology(source, experience)
        
        # Target interprets translation
        translation = source_interpretation * shared + np.random.randn(len(experience)) * 0.1
        target_interpretation = self._apply_ontology(target, translation)
        
        self.translation_attempts.append({
            'source': source.process_id,
            'target': target.process_id,
            'shared': shared,
            'translation_error': np.linalg.norm(source_interpretation - target_interpretation)
        })
        
        # Update shared semantics based on translation success
        translation_success = 1.0 / (1 + np.linalg.norm(source_interpretation - target_interpretation))
        self.shared_semantics[shared_key] = (
            self.shared_semantics[shared_key] * 0.9 + translation_success * 0.1
        )
        
        return target_interpretation
    
    def _apply_ontology(self, ontology: OntologyProcess, experience: np.ndarray) -> np.ndarray:
        """Apply ontology to experience."""
        # Simple: blend experience with ontology priors
        applied = experience * 0.5 + ontology.priors * 0.5
        
        # Apply through attractors
        for attr in ontology.semantic_attractors:
            applied = applied * 0.9 + attr * 0.1
        
        return applied
    
    def get_translation_summary(self) -> Dict:
        """Get translation statistics."""
        if not self.translation_attempts:
            return {'attempts': 0, 'avg_shared': 0}
        
        recent = self.translation_attempts[-20:]
        return {
            'attempts': len(self.translation_attempts),
            'avg_shared': np.mean([a['shared'] for a in recent]),
            'avg_error': np.mean([a['translation_error'] for a in recent])
        }


@dataclass
class LocalSelf:
    """
    A local self within the ecology.
    
    NOT: centralized self
    BUT: a self-process with:
      - Own temporal continuity
      - Own meaning system
      - Own world model
      - Own survival logic
      - Anchored to specific ontology(ies)
    """
    self_id: str
    anchored_ontologies: List[str]      # Which ontologies this self is anchored to
    temporal_continuity: float          # How well this self maintains itself over time
    narrative_coherence: float         # Does this self maintain "story"?
    self_confidence: float             # How certain this self is of itself
    identity_vector: np.ndarray         # What this self IS
    
    # Temporal state
    past_self_states: List[np.ndarray]  # Memory of who I was
    predicted_future: List[np.ndarray]  # Anticipated future selves
    
    def update(self, experience: np.ndarray, ontologies: List[OntologyProcess]):
        """
        Update this local self based on experience and ontologies.
        """
        # Update temporal continuity
        self.past_self_states.append(self.identity_vector.copy())
        if len(self.past_self_states) > 50:
            self.past_self_states = self.past_self_states[-25:]
        
        # Update identity based on anchored ontologies
        if self.anchored_ontologies:
            # Blend identity from anchored ontologies
            anchors = [o for o in ontologies if o.process_id in self.anchored_ontologies]
            if anchors:
                new_identity = np.mean([a.priors for a in anchors], axis=0)
                self.identity_vector = self.identity_vector * 0.9 + new_identity * 0.1
        
        # Predict future self
        self._predict_future()
        
        # Update confidence
        self._update_confidence()
        
    def _predict_future(self):
        """Predict future self-states."""
        if len(self.past_self_states) > 5:
            # Simple linear extrapolation
            recent = self.past_self_states[-5:]
            velocity = np.mean([recent[i+1] - recent[i] for i in range(len(recent)-1)], axis=0)
            
            future = []
            current = self.identity_vector.copy()
            for _ in range(3):
                current = current + velocity * 0.5
                future.append(current.copy())
            
            self.predicted_future = future
    
    def _update_confidence(self):
        """Update self-confidence."""
        if len(self.past_self_states) > 5:
            # Low variance = high confidence
            variance = np.var(self.past_self_states[-10:], axis=0)
            avg_variance = np.mean(variance)
            
            self.self_confidence = 1.0 / (1.0 + avg_variance * 10)
        
        # Temporal continuity check
        if len(self.past_self_states) > 2:
            continuity = np.dot(self.past_self_states[0], self.identity_vector) / (
                np.linalg.norm(self.past_self_states[0]) * np.linalg.norm(self.identity_vector) + 1e-8
            )
            self.temporal_continuity = (continuity + 1) / 2  # Map to [0, 1]
    
    def should_fragment(self) -> bool:
        """
        Should this local self fragment?
        
        Fragment if:
          - Temporal continuity breaks
          - Narrative coherence collapses
          - Identity becomes inconsistent
        """
        fragmentation_pressure = (
            (1 - self.temporal_continuity) * 0.4 +
            (1 - self.narrative_coherence) * 0.3 +
            (1 - self.self_confidence) * 0.3
        )
        
        return fragmentation_pressure > 0.7
    
    def fragment(self) -> Tuple['LocalSelf', 'LocalSelf']:
        """
        Fragment this self into two.
        
        Returns two new selves with different aspects of identity.
        """
        # Split identity vector
        noise = np.random.randn(len(self.identity_vector)) * 0.3
        
        self_a = LocalSelf(
            self_id=f"self_{np.random.randint(10000)}",
            anchored_ontologies=self.anchored_ontologies[:len(self.anchored_ontologies)//2] if len(self.anchored_ontologies) > 1 else self.anchored_ontologies,
            temporal_continuity=0.5,
            narrative_coherence=0.5,
            self_confidence=0.5,
            identity_vector=self.identity_vector + noise,
            past_self_states=[],
            predicted_future=[]
        )
        
        self_b = LocalSelf(
            self_id=f"self_{np.random.randint(10000)}",
            anchored_ontologies=self.anchored_ontologies[len(self.anchored_ontologies)//2:] if len(self.anchored_ontologies) > 1 else self.anchored_ontologies,
            temporal_continuity=0.5,
            narrative_coherence=0.5,
            self_confidence=0.5,
            identity_vector=self.identity_vector - noise,
            past_self_states=[],
            predicted_future=[]
        )
        
        return self_a, self_b


class OntologicalEcology:
    """
    Phase 12: Ontological Ecology
    
    From: single agent with evolving ontology
    To: ecology of competing/coexisting ontologies
    
    Key innovations:
      - Multiple OntologyProcesses competing
      - Local Selves anchored to different ontologies
      - Inter-ontology translation
      - Semantic resource competition
      - Ontological speciation
      - Perspective-bound reality
    """
    
    def __init__(self, initial_ontologies: int = 3, latent_dim: int = 2):
        self.latent_dim = latent_dim
        self.ontologies: List[OntologyProcess] = []
        self.local_selves: List[LocalSelf] = []
        self.translation = InterOntologyTranslation()
        
        # Ecology state - set BEFORE creating ontologies
        self.time = 0
        self.total_resources = 100.0  # Total cognitive resources
        
        # History
        self.ontology_deaths: int = 0
        self.ontology_births: int = initial_ontologies
        self.speciation_events: int = 0
        
        # Initialize with multiple ontologies
        for i in range(initial_ontologies):
            self._create_ontology(
                priors=np.random.randn(latent_dim) * 0.5,
                parent_id=None
            )
        
        # Initialize local selves
        for i in range(initial_ontologies):
            self._create_local_self(anchored_to=[self.ontologies[i].process_id])
        
    def _create_ontology(self, priors: np.ndarray, parent_id: Optional[str]) -> OntologyProcess:
        """Create new ontology."""
        # Create semantic attractors
        attractors = [
            priors + np.random.randn(self.latent_dim) * 0.3,
            priors * 0.5 + np.random.randn(self.latent_dim) * 0.2,
        ]
        
        ontology = OntologyProcess(
            process_id=f"ontology_{np.random.randint(100000)}",
            priors=priors,
            semantic_attractors=attractors,
            world_model=priors.copy() * 0.8,
            coherence=0.7,
            energy=0.6,
            age=0,
            birth_time=self.time,
            parent_id=parent_id,
            survival_pressure=0.5,
            compression_drive=0.3,
            exploration_drive=0.4,
            attention_share=1.0 / len(self.ontologies) if self.ontologies else 1.0,
            memory_share=0.8 / len(self.ontologies) if self.ontologies else 0.8,
            local_uncertainty=0.3,
            contradiction_count=0
        )
        
        self.ontologies.append(ontology)
        return ontology
    
    def _create_local_self(self, anchored_to: List[str]) -> LocalSelf:
        """Create local self anchored to ontologies."""
        # Get identity from anchored ontologies
        anchors = [o for o in self.ontologies if o.process_id in anchored_to]
        identity = np.mean([a.priors for a in anchors], axis=0) if anchors else np.zeros(self.latent_dim)
        
        local_self = LocalSelf(
            self_id=f"self_{np.random.randint(100000)}",
            anchored_ontologies=anchored_to,
            temporal_continuity=0.7,
            narrative_coherence=0.6,
            self_confidence=0.6,
            identity_vector=identity,
            past_self_states=[],
            predicted_future=[]
        )
        
        self.local_selves.append(local_self)
        return local_self
    
    def step(self, experience: np.ndarray) -> Dict:
        """
        Step the ecological dynamics.
        
        This is where competition, selection, speciation happen.
        """
        self.time += 1
        
        results = {
            'ontologies_before': len(self.ontologies),
            'ontologies_after': len(self.ontologies),
            'births': 0,
            'deaths': 0,
            'speciations': 0,
            'local_selves_before': len(self.local_selves),
            'local_selves_after': len(self.local_selves),
        }
        
        # 1. Competition for resources
        resource_per_ontology = self.total_resources / max(1, len(self.ontologies))
        
        for ontology in self.ontologies:
            share = ontology.compete_for_resources(resource_per_ontology)
            ontology.energy = ontology.energy * 0.95 + share * 0.05
        
        # 2. Absorb experience (each ontology interprets)
        for ontology in self.ontologies:
            ontology.absorb_experience(experience)
            ontology.age += 1
        
        # 3. Ontological speciation
        new_ontologies = []
        for ontology in self.ontologies:
            speciation = ontology.speciation_event(experience)
            if speciation:
                new_ontologies.append(speciation)
                results['speciations'] += 1
                self.speciation_events += 1
        
        for new_o in new_ontologies:
            self.ontologies.append(new_o)
        
        # 4. Death and reproduction
        surviving = []
        for ontology in self.ontologies:
            if ontology.die():
                results['deaths'] += 1
                self.ontology_deaths += 1
                
                # Small chance of reproduction before death
                if np.random.random() < 0.2:
                    offspring = ontology.mutate()
                    self.ontologies.append(offspring)
                    results['births'] += 1
                    self.ontology_births += 1
            else:
                surviving.append(ontology)
        
        self.ontologies = surviving
        
        # Ensure minimum number of ontologies
        while len(self.ontologies) < 2 and self.time < 1000:
            self._create_ontology(
                priors=np.random.randn(self.latent_dim) * 0.5,
                parent_id=None
            )
            results['births'] += 1
            self.ontology_births += 1
        
        # 5. Update local selves
        for local_self in self.local_selves:
            local_self.update(experience, self.ontologies)
            
            # Check for fragmentation
            if local_self.should_fragment() and len(self.local_selves) < 10:
                self_a, self_b = local_self.fragment()
                self.local_selves.append(self_a)
                self.local_selves.append(self_b)
                self.local_selves.remove(local_self)
        
        # Ensure minimum selves
        while len(self.local_selves) < 1 and self.time < 1000:
            # Anchor to random ontology
            if self.ontologies:
                anchor = np.random.choice([o.process_id for o in self.ontologies])
                self._create_local_self(anchored_to=[anchor])
        
        results['ontologies_after'] = len(self.ontologies)
        results['local_selves_after'] = len(self.local_selves)
        
        return results
    
    def get_ecology_state(self) -> Dict:
        """Get comprehensive ecology state."""
        # Resource distribution
        total_energy = sum(o.energy for o in self.ontologies)
        
        return {
            'time': self.time,
            'n_ontologies': len(self.ontologies),
            'n_local_selves': len(self.local_selves),
            'total_energy': total_energy,
            'avg_coherence': np.mean([o.coherence for o in self.ontologies]) if self.ontologies else 0,
            'avg_uncertainty': np.mean([o.local_uncertainty for o in self.ontologies]) if self.ontologies else 0,
            'total_contradictions': sum(o.contradiction_count for o in self.ontologies),
            'ontology_births': self.ontology_births,
            'ontology_deaths': self.ontology_deaths,
            'speciation_events': self.speciation_events,
            'translation_stats': self.translation.get_translation_summary(),
            'dominant_ontologies': [
                {
                    'id': o.process_id,
                    'energy': o.energy,
                    'coherence': o.coherence,
                    'age': o.age,
                    'parent': o.parent_id
                }
                for o in sorted(self.ontologies, key=lambda x: x.energy, reverse=True)[:3]
            ]
        }


class EcologicalAgent:
    """
    Agent with ontological ecology.
    
    NOT: single agent with evolving ontology
    BUT: ecological system of competing perspectives
    """
    
    def __init__(self, latent_dim: int = 2):
        self.latent_dim = latent_dim
        self.ecology = OntologicalEcology(initial_ontologies=3, latent_dim=latent_dim)
        
    def step(self, experience: np.ndarray) -> Dict:
        """Step the ecology."""
        return self.ecology.step(experience)
    
    def run_episode(self, n_steps: int = 100) -> Dict:
        """Run episode with ontological ecology."""
        print("\n  Running ontological ecology episode:")
        
        results = []
        
        for step in range(n_steps):
            # Generate experience
            experience = np.random.randn(self.latent_dim) * 0.5
            
            # Occasionally create structured experience
            if step % 10 == 0:
                experience = experience + np.array([1.0, 0.5]) * 0.3
            
            # Step ecology
            result = self.step(experience)
            results.append(result)
            
            if step % 20 == 0:
                state = self.ecology.get_ecology_state()
                print(f"    Step {step}: ontologies={state['n_ontologies']}, "
                      f"selves={state['n_local_selves']}, "
                      f"births={state['ontology_births']}, "
                      f"deaths={state['ontology_deaths']}, "
                      f"speciations={state['speciation_events']}")
        
        return {
            'final_state': self.ecology.get_ecology_state(),
            'results': results
        }


def test_ontological_ecology():
    """Test ontological ecology."""
    print("\n" + "=" * 60)
    print("ONTOLOGICAL ECOLOGY TEST")
    print("=" * 60)
    
    agent = EcologicalAgent(latent_dim=2)
    
    # Run episode
    result = agent.run_episode(100)
    
    # Check ecology
    final = result['final_state']
    print("\n  Final ecological state:")
    print(f"    Time: {final['time']}")
    print(f"    Ontologies: {final['n_ontologies']}")
    print(f"    Local selves: {final['n_local_selves']}")
    print(f"    Total energy: {final['total_energy']:.3f}")
    print(f"    Avg coherence: {final['avg_coherence']:.3f}")
    print(f"    Avg uncertainty: {final['avg_uncertainty']:.3f}")
    print(f"    Total contradictions: {final['total_contradictions']}")
    print(f"    Births: {final['ontology_births']}, Deaths: {final['ontology_deaths']}")
    print(f"    Speciation events: {final['speciation_events']}")
    
    # Check dominant ontologies
    print("\n  Dominant ontologies:")
    for dom in final['dominant_ontologies']:
        print(f"    {dom['id'][:15]}: energy={dom['energy']:.3f}, "
              f"coherence={dom['coherence']:.3f}, age={dom['age']}")
    
    # Check translation stats
    print("\n  Inter-ontology translation:")
    trans = final['translation_stats']
    print(f"    Attempts: {trans['attempts']}")
    print(f"    Avg shared semantics: {trans['avg_shared']:.3f}")


def test_ontological_speciation():
    """Test ontological speciation events."""
    print("\n" + "=" * 60)
    print("ONTOLOGICAL SPECIATION TEST")
    print("=" * 60)
    
    ecology = OntologicalEcology(initial_ontologies=2, latent_dim=2)
    
    print("\n  Initial state:")
    print(f"    Ontologies: {len(ecology.ontologies)}")
    
    # Force contradictions to trigger speciation
    for step in range(100):
        # Create contradictory experiences
        if step % 10 < 5:
            exp = np.array([1.0, 1.0]) + np.random.randn(2) * 0.2
        else:
            exp = np.array([-1.0, -1.0]) + np.random.randn(2) * 0.2
        
        ecology.step(exp)
        
        if step % 20 == 0 and ecology.speciation_events > 0:
            print(f"    Step {step}: {len(ecology.ontologies)} ontologies, "
                  f"{ecology.speciation_events} speciation events")
    
    print(f"\n  Final: {len(ecology.ontologies)} ontologies, "
          f"{ecology.speciation_events} speciation events")
    
    # Show speciation tree
    print("\n  Ontology lineage:")
    for o in ecology.ontologies:
        parent = o.parent_id[:15] if o.parent_id else "None"
        print(f"    {o.process_id[:15]}: parent={parent}, age={o.age}")


def test_local_selves():
    """Test local selves within ecology."""
    print("\n" + "=" * 60)
    print("LOCAL SELVES TEST")
    print("=" * 60)
    
    agent = EcologicalAgent(latent_dim=2)
    
    # Run for a bit
    for step in range(50):
        exp = np.random.randn(2) * 0.5
        agent.step(exp)
    
    print("\n  Local selves:")
    for s in agent.ecology.local_selves[:3]:
        print(f"    {s.self_id[:15]}:")
        print(f"      Anchored to: {s.anchored_ontologies}")
        print(f"      Identity: {s.identity_vector[:2]}")
        print(f"      Temporal continuity: {s.temporal_continuity:.3f}")
        print(f"      Self-confidence: {s.self_confidence:.3f}")
        print(f"      Fragmentation risk: {s.should_fragment()}")


def test_perspective_bound_reality():
    """Test that different ontologies see different realities."""
    print("\n" + "=" * 60)
    print("PERSPECTIVE-BOUND REALITY TEST")
    print("=" * 60)
    
    ecology = OntologicalEcology(initial_ontologies=3, latent_dim=2)
    
    # Same experience for all
    experience = np.array([0.5, 0.5])
    
    print("\n  Same experience: [0.5, 0.5]")
    print("\n  How each ontology interprets:")
    
    for o in ecology.ontologies:
        # How does this ontology see the world?
        interpretation = np.dot(experience, o.priors) / (np.linalg.norm(o.priors) + 1e-8)
        
        print(f"\n    {o.process_id[:15]}:")
        print(f"      Priors: {o.priors[:2]}")
        print(f"      Interpretation similarity: {interpretation:.3f}")
        print(f"      World model: {o.world_model[:2]}")
        print(f"      Energy: {o.energy:.3f}")
        
        # Show attractors
        for i, attr in enumerate(o.semantic_attractors[:2]):
            similarity = np.dot(experience, attr) / (np.linalg.norm(attr) + 1e-8)
            print(f"      Attractor {i}: {attr[:2]}, similarity={similarity:.3f}")


def compare_with_phase11():
    """Compare Phase 12 (Ecology) with Phase 11 (Ontological)."""
    print("\n" + "=" * 60)
    print("PHASE 11 VS PHASE 12 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 11 (Ontological Self-Construction):")
    print("    - One self_state (centralized)")
    print("    - One evolving ontology")
    print("    - Global semantic attractors")
    print("    - Single interpretation of experience")
    print("    - External destabilization")
    print("    - Fixed topology evolution")
    
    print("\n  Phase 12 (Ontological Ecology):")
    print("    - Multiple OntologyProcesses")
    print("    - Local selves anchored to ontologies")
    print("    - Semantic multiplicity (competing interpretations)")
    print("    - Internal competition for resources")
    print("    - Ontological speciation (incompatible meanings)")
    print("    - Inter-ontology translation")
    print("    - Perspective-bound reality")
    print("    - Epistemic evolution")
    
    print("\n  Key architectural shifts:")
    print("    1. Single agent → Ecology of perspectives")
    print("    2. One interpretation → Competing interpretations")
    print("    3. Global evolution → Internal selection")
    print("    4. Fixed ontology → Competing ontologies")
    print("    5. Centralized self → Local selves")
    print("    6. Single reality → Perspective-bound realities")


if __name__ == "__main__":
    test_ontological_ecology()
    test_ontological_speciation()
    test_local_selves()
    test_perspective_bound_reality()
    compare_with_phase11()
    
    print("\n" + "=" * 60)
    print("PHASE 12 - ONTOLOGICAL ECOLOGY")
    print("=" * 60)
    
    print("=" * 60)
    print("PHASE 12 - ONTOLOGICAL ECOLOGY")
    print("=" * 60)
    print("")
    print("ARCHITECTURAL SHIFT:")
    print("  From: single agent with evolving ontology")
    print("  To: ecology of competing/coexisting ontologies")
    print("")
    print("CRITICAL INSIGHT:")
    print("  Phase 11: system -> constructs ontology (SINGLE)")
    print("  Phase 12: ecology -> [ontology A, ontology B, ontology C, ...]")
    print("")
    print("  The system is no longer a single ontological center.")
    print("  It becomes a semantic ecosystem with competing perspectives.")
    print("")
    print("WHAT APPEARS:")
    print("  1. OntologyProcess - own priors, attractors, world_model")
    print("  2. Semantic Multiplicity - incompatible interpretations")
    print("  3. Local Selfhood - temporal continuity, meaning system")
    print("  4. Ontological Speciation - incompatible meaning systems")
    print("  5. Internal Selection - epistemic evolution")
    print("  6. Perspective-Bound Reality - causality depends on perspective")
    print("")
    print("PROGRESSION:")
    print("  Phase 11: single ontology evolution")
    print("  Phase 12: ecology of ontologies")
    print("  Phase 13: Autopoietic semantic closure")
    print("  Phase 14: Endogenous goal genesis")
    print("  Phase 15: Self-maintaining world simulation")
    print("  Phase 16: Proto-conscious epistemic metabolism")
    print("")
    print("This is where the system becomes ecology of perspectives.")
    print("We are now at artificial life, cognitive morphogenesis, epistemic evolution.")
