"""
Pure Domain State - Immutable structures with no mutations.

Stage 1: Domain Core only
- All state is frozen/immutable
- Uses types.MappingProxyType for dict-like immutability
- No side effects, no asyncio, no infra

Key principle: State is derived from events, not mutated directly.
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, FrozenSet
from dataclasses import dataclass, field
import hashlib
import json


@dataclass(frozen=True)
class ImmutableBelief:
    """Immutable belief with tuple-based internals"""
    belief_id: str
    proposition: str
    confidence: float
    entropy: float
    source: str
    created_at: str
    last_updated: str
    incoming_causes: Tuple[str, ...]
    outgoing_effects: Tuple[str, ...]
    attractor_state: Optional[str] = None


@dataclass(frozen=True)
class ImmutableGenome:
    """Immutable genome state"""
    axes: MappingProxyType  # type: ignore
    selection_pressure: float
    mutation_rate: float
    fitness_scores: MappingProxyType  # type: ignore
    
    def __post_init__(self):
        if not isinstance(self.axes, MappingProxyType):
            object.__setattr__(self, 'axes', MappingProxyType(dict(self.axes)))
        if not isinstance(self.fitness_scores, MappingProxyType):
            object.__setattr__(self, 'fitness_scores', MappingProxyType(dict(self.fitness_scores)))
    
    def get_axis(self, axis: str) -> float:
        return self.axes.get(axis, 0.5)
    
    def with_axis(self, axis: str, value: float) -> 'ImmutableGenome':
        """Return NEW genome with updated axis"""
        new_axes = {**self.axes, axis: value}
        return ImmutableGenome(
            axes=MappingProxyType(new_axes),
            selection_pressure=self.selection_pressure,
            mutation_rate=self.mutation_rate,
            fitness_scores=self.fitness_scores
        )


@dataclass(frozen=True)
class ImmutableContradiction:
    """Immutable contradiction"""
    episode_id: str
    belief_ids: Tuple[str, ...]
    contradiction_type: str
    first_seen: str
    last_seen: str
    recurrence_count: int
    stability_score: float
    resolution_status: str
    severity: str


@dataclass(frozen=True)
class ImmutableIdentity:
    """
    Immutable identity state.
    
    All mutations return NEW instances.
    Uses MappingProxyType for true immutability.
    """
    autonomy: float
    curiosity: float
    stability: float
    coherence: float
    
    def get_axis(self, axis: str) -> float:
        return getattr(self, axis, 0.5)
    
    def with_axis(self, axis: str, delta: float) -> 'ImmutableIdentity':
        """Return NEW identity with updated axis (pure transform)"""
        new_values = {
            "autonomy": self.autonomy,
            "curiosity": self.curiosity,
            "stability": self.stability,
            "coherence": self.coherence
        }
        new_values[axis] = max(0.0, min(1.0, new_values[axis] + delta))
        return ImmutableIdentity(**new_values)


@dataclass(frozen=True)
class ImmutableLineage:
    """Immutable lineage tracking"""
    mutation_history: Tuple[Tuple[str, float], ...]  # (axis, delta)
    cause_chains: Tuple[Tuple[str, str], ...]  # (event_id, causation_id)
    trajectory_hash: str
    
    @staticmethod
    def compute_hash(actions: Tuple[Tuple[str, float], ...]) -> str:
        """Deterministic hash for trajectory"""
        content = {"actions": list(actions)}
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class DomainState:
    """
    Root immutable domain state.
    
    All mutations return NEW instances.
    No state is ever mutated in place.
    """
    beliefs: MappingProxyType  # type: ignore
    genome: ImmutableGenome
    identity: ImmutableIdentity
    contradictions: MappingProxyType  # type: ignore
    lineage: ImmutableLineage
    version: int
    entropy: float
    
    def __post_init__(self):
        if not isinstance(self.beliefs, MappingProxyType):
            object.__setattr__(self, 'beliefs', MappingProxyType(dict(self.beliefs)))
        if not isinstance(self.contradictions, MappingProxyType):
            object.__setattr__(self, 'contradictions', MappingProxyType(dict(self.contradictions)))
    
    @staticmethod
    def compute_hash(state: 'DomainState') -> str:
        """Deterministic hash for replay verification"""
        data = {
            "version": state.version,
            "entropy": state.entropy,
            "beliefs": sorted(state.beliefs.keys()),
            "contradictions": sorted(state.contradictions.keys()),
            "identity": (state.identity.autonomy, state.identity.curiosity, 
                        state.identity.stability, state.identity.coherence),
            "genome_axes": dict(state.genome.axes)
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    @property
    def belief_count(self) -> int:
        """Number of beliefs"""
        return len(self.beliefs)
    
    @property
    def contradiction_count(self) -> int:
        """Number of contradictions"""
        return len(self.contradictions)
    
    def with_belief(self, belief_id: str, belief: ImmutableBelief) -> 'DomainState':
        """Return NEW state with belief"""
        new_beliefs = {**self.beliefs, belief_id: belief}
        return DomainState(
            beliefs=MappingProxyType(new_beliefs),
            genome=self.genome,
            identity=self.identity,
            contradictions=self.contradictions,
            lineage=self.lineage,
            version=self.version + 1,
            entropy=self.entropy
        )
    
    def with_genome(self, genome: ImmutableGenome) -> 'DomainState':
        """Return NEW state with genome"""
        return DomainState(
            beliefs=self.beliefs,
            genome=genome,
            identity=self.identity,
            contradictions=self.contradictions,
            lineage=self.lineage,
            version=self.version + 1,
            entropy=self.entropy
        )
    
    def with_identity(self, identity: ImmutableIdentity) -> 'DomainState':
        """Return NEW state with identity"""
        return DomainState(
            beliefs=self.beliefs,
            genome=self.genome,
            identity=identity,
            contradictions=self.contradictions,
            lineage=self.lineage,
            version=self.version + 1,
            entropy=self.entropy
        )
    
    def with_contradiction(self, episode_id: str, contradiction: ImmutableContradiction) -> 'DomainState':
        """Return NEW state with contradiction"""
        new_contradictions = {**self.contradictions, episode_id: contradiction}
        return DomainState(
            beliefs=self.beliefs,
            genome=self.genome,
            identity=self.identity,
            contradictions=MappingProxyType(new_contradictions),
            lineage=self.lineage,
            version=self.version + 1,
            entropy=self.entropy
        )
    
    def with_lineage(self, lineage: ImmutableLineage) -> 'DomainState':
        """Return NEW state with lineage"""
        return DomainState(
            beliefs=self.beliefs,
            genome=self.genome,
            identity=self.identity,
            contradictions=self.contradictions,
            lineage=lineage,
            version=self.version + 1,
            entropy=self.entropy
        )


def create_initial_state() -> DomainState:
    """Create initial empty domain state"""
    return DomainState(
        beliefs=MappingProxyType({}),
        genome=ImmutableGenome(
            axes=MappingProxyType({
                "autonomy": 0.5,
                "curiosity": 0.5,
                "stability": 0.5,
                "coherence": 0.5
            }),
            selection_pressure=0.5,
            mutation_rate=0.1,
            fitness_scores=MappingProxyType({})
        ),
        identity=ImmutableIdentity(
            autonomy=0.5,
            curiosity=0.5,
            stability=0.5,
            coherence=0.5
        ),
        contradictions=MappingProxyType({}),
        lineage=ImmutableLineage(
            mutation_history=(),
            cause_chains=(),
            trajectory_hash=""
        ),
        version=0,
        entropy=0.0
    )