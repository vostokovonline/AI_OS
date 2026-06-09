"""
Memory Types - Four distinct memory systems for cognition.

Stage: Cognitive Architecture

Unlike event log which is about system state,
memory types are about cognitive content:
- What we know
- What we experienced
- How we do things
- What we reflect upon

This enables:
- Belief revision (semantic)
- Experience replay (episodic)
- Skill acquisition (procedural)
- Learning from outcomes (reflective)
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json


@dataclass(frozen=True)
class MemoryEntry:
    """Base memory entry - immutable"""
    memory_id: str
    content: str
    timestamp: str
    source: str
    importance: float  # 0-1, how important this memory is
    decay_rate: float  # How fast memory fades
    tags: Tuple[str, ...]
    
    def with_importance(self, importance: float) -> 'MemoryEntry':
        """Return new entry with updated importance"""
        return MemoryEntry(
            memory_id=self.memory_id,
            content=self.content,
            timestamp=self.timestamp,
            source=self.source,
            importance=importance,
            decay_rate=self.decay_rate,
            tags=self.tags
        )


@dataclass(frozen=True)
class Belief:
    """
    Semantic memory: what we believe about the world.
    
    Key properties:
    - proposition: what we believe
    - confidence: how strongly we believe (0-1)
    - evidence: what supports this belief
    - sources: where this belief came from
    - stability: how resistant to change
    """
    belief_id: str
    proposition: str
    confidence: float
    evidence: Tuple[str, ...]  # belief_ids that support this
    counter_evidence: Tuple[str, ...]  # belief_ids against this
    sources: Tuple[str, ...]  # memory_ids of origin
    created_at: str
    last_updated: str
    stability: float  # How resistant to revision (0-1)
    revision_count: int  # How many times this belief was revised
    
    def supporting_evidence_count(self) -> int:
        return len(self.evidence)
    
    def contradicting_evidence_count(self) -> int:
        return len(self.counter_evidence)
    
    def net_support(self) -> float:
        """Net evidence: supporting - contradicting"""
        return len(self.evidence) - len(self.counter_evidence)
    
    def is_contested(self) -> bool:
        """Belief is contested if counter evidence >= supporting"""
        return len(self.counter_evidence) >= len(self.evidence)
    
    def update_confidence(self, new_confidence: float) -> 'Belief':
        """Return updated belief with new confidence"""
        return Belief(
            belief_id=self.belief_id,
            proposition=self.proposition,
            confidence=max(0.0, min(1.0, new_confidence)),
            evidence=self.evidence,
            counter_evidence=self.counter_evidence,
            sources=self.sources,
            created_at=self.created_at,
            last_updated=datetime.utcnow().isoformat(),
            stability=self.stability,
            revision_count=self.revision_count + 1
        )


@dataclass(frozen=True)
class EpisodicMemory:
    """
    Episodic memory: what we experienced.
    
    Represents concrete experiences with:
    - Context (where, when)
    - Actions taken
    - Outcomes observed
    - Emotions felt
    """
    episode_id: str
    context: Tuple[Tuple[str, str], ...]  # key-value pairs (where, when, who)
    actions: Tuple[str, ...]  # What was done
    outcome: str  # What happened
    emotional_valence: float  # -1 to 1, negative to positive
    lessons: Tuple[str, ...]  # What was learned
    timestamp: str
    duration_ms: int
    related_goals: Tuple[str, ...]  # goal_ids that were involved
    
    def get_context_dict(self) -> Dict[str, str]:
        return dict(self.context)
    
    def extract_entities(self) -> List[str]:
        """Extract mentioned entities from episode"""
        entities = []
        for k, v in self.context:
            if k in ('entity', 'person', 'object'):
                entities.append(v)
        return entities


@dataclass(frozen=True)
class Procedure:
    """
    Procedural memory: how we do things.
    
    Represents skills and procedures:
    - goal: what this procedure achieves
    - steps: how to do it
    - preconditions: what must be true
    - success_rate: historical performance
    - applicable_contexts: when to use
    """
    procedure_id: str
    goal: str  # What this achieves
    steps: Tuple[str, ...]  # Ordered steps
    preconditions: Tuple[str, ...]  # Required state
    success_rate: float  # Historical success (0-1)
    applicability_score: float  # How well this applies now (0-1)
    last_used: str
    use_count: int
    failure_modes: Tuple[str, ...]  # Known failure modes
    
    def estimated_success(self) -> float:
        """Combine success rate with current applicability"""
        return self.success_rate * self.applicability_score
    
    def is_applicable(self, current_state: Dict[str, Any]) -> bool:
        """Check if preconditions are met"""
        for precond in self.preconditions:
            if precond not in current_state:
                return False
        return True


@dataclass(frozen=True)
class Reflection:
    """
    Reflective memory: what we learned from thinking.
    
    Represents meta-cognitive insights:
    - pattern: what pattern was observed
    - insight: what was understood
    - implications: what this means for future
    - confidence: how certain this insight is
    """
    reflection_id: str
    pattern: str  # What was observed
    insight: str  # What was understood
    implications: Tuple[str, ...]  # What this means
    confidence: float  # How certain (0-1)
    related_beliefs: Tuple[str, ...]  # Beliefs this touches
    timestamp: str
    trigger: str  # What triggered this reflection
    
    def affected_beliefs(self) -> int:
        return len(self.related_beliefs)


@dataclass(frozen=True)
class MemoryState:
    """
    Combined memory state - the cognitive knowledge base.
    
    This is what gets updated by cognitive loop.
    NOT event-sourced state, but belief/knowledge state.
    """
    beliefs: MappingProxyType  # type: ignore
    episodes: MappingProxyType  # type: ignore
    procedures: MappingProxyType  # type: ignore
    reflections: MappingProxyType  # type: ignore
    version: int
    
    def __post_init__(self):
        if not isinstance(self.beliefs, MappingProxyType):
            object.__setattr__(self, 'beliefs', MappingProxyType(dict(self.beliefs)))
        if not isinstance(self.episodes, MappingProxyType):
            object.__setattr__(self, 'episodes', MappingProxyType(dict(self.episodes)))
        if not isinstance(self.procedures, MappingProxyType):
            object.__setattr__(self, 'procedures', MappingProxyType(dict(self.procedures)))
        if not isinstance(self.reflections, MappingProxyType):
            object.__setattr__(self, 'reflections', MappingProxyType(dict(self.reflections)))
    
    @staticmethod
    def compute_hash(state: 'MemoryState') -> str:
        """Deterministic hash for replay verification"""
        data = {
            "version": state.version,
            "belief_count": len(state.beliefs),
            "episode_count": len(state.episodes),
            "procedure_count": len(state.procedures),
            "reflection_count": len(state.reflections)
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def with_belief(self, belief_id: str, belief: Belief) -> 'MemoryState':
        new_beliefs = {**self.beliefs, belief_id: belief}
        return MemoryState(
            beliefs=MappingProxyType(new_beliefs),
            episodes=self.episodes,
            procedures=self.procedures,
            reflections=self.reflections,
            version=self.version + 1
        )
    
    def with_episode(self, episode_id: str, episode: EpisodicMemory) -> 'MemoryState':
        new_episodes = {**self.episodes, episode_id: episode}
        return MemoryState(
            beliefs=self.beliefs,
            episodes=MappingProxyType(new_episodes),
            procedures=self.procedures,
            reflections=self.reflections,
            version=self.version + 1
        )
    
    def with_procedure(self, procedure_id: str, procedure: Procedure) -> 'MemoryState':
        new_procedures = {**self.procedures, procedure_id: procedure}
        return MemoryState(
            beliefs=self.beliefs,
            episodes=self.episodes,
            procedures=MappingProxyType(new_procedures),
            reflections=self.reflections,
            version=self.version + 1
        )
    
    def with_reflection(self, reflection_id: str, reflection: Reflection) -> 'MemoryState':
        new_reflections = {**self.reflections, reflection_id: reflection}
        return MemoryState(
            beliefs=self.beliefs,
            episodes=self.episodes,
            procedures=self.procedures,
            reflections=MappingProxyType(new_reflections),
            version=self.version + 1
        )


def create_empty_memory() -> MemoryState:
    """Create initial empty memory state"""
    return MemoryState(
        beliefs=MappingProxyType({}),
        episodes=MappingProxyType({}),
        procedures=MappingProxyType({}),
        reflections=MappingProxyType({}),
        version=0
    )