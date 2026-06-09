"""
World Model - Persistent belief state with revision capabilities.

Stage: Cognitive Architecture

World Model is the system's understanding of reality:
- What exists (entities)
- How things relate (relations)
- What causes what (causality)
- What's true (beliefs)

Key operations:
- Belief revision with evidence tracking
- Causal inference
- Entailment checking
- Consistency maintenance
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Set, FrozenSet
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json


@dataclass(frozen=True)
class Entity:
    """Something that exists in the world"""
    entity_id: str
    entity_type: str  # person, object, concept, event
    properties: FrozenSet[Tuple[str, Any]]  # property name to value
    created_at: str
    
    def get_property(self, key: str) -> Optional[Any]:
        for k, v in self.properties:
            if k == key:
                return v
        return None


@dataclass(frozen=True)
class Relation:
    """How entities relate to each other"""
    relation_id: str
    subject: str  # entity_id
    predicate: str  # relation type (is_a, part_of, causes, etc.)
    object: str  # entity_id
    confidence: float
    evidence: Tuple[str, ...]  # belief_ids supporting
    created_at: str
    
    def is_strong(self) -> bool:
        return self.confidence >= 0.7


@dataclass(frozen=True)
class CausalLink:
    """Cause-effect relationship"""
    cause_id: str
    effect_id: str
    mechanism: str  # How cause leads to effect
    strength: float  # How reliable this causal link is
    conditions: Tuple[str, ...]  # When this applies
    evidence: Tuple[str, ...]  # Supporting beliefs
    created_at: str
    
    def applies_under(self, conditions: Set[str]) -> bool:
        return all(c in conditions for c in self.conditions)


@dataclass(frozen=True)
class WorldModel:
    """
    The system's model of reality.
    
    Contains:
    - entities: things that exist
    - relations: how things relate
    - causal_links: cause-effect chains
    - assertions: beliefs about the world
    """
    entities: MappingProxyType  # type: ignore
    relations: MappingProxyType  # type: ignore
    causal_links: MappingProxyType  # type: ignore
    assertions: MappingProxyType  # type: ignore  # assertion_id -> confidence
    version: int
    last_updated: str
    
    def __post_init__(self):
        for attr in ('entities', 'relations', 'causal_links', 'assertions'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
    
    @staticmethod
    def compute_hash(state: 'WorldModel') -> str:
        data = {
            "version": state.version,
            "entity_count": len(state.entities),
            "relation_count": len(state.relations),
            "causal_count": len(state.causal_links),
            "assertion_count": len(state.assertions)
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def get_relations_from(self, entity_id: str) -> List[Relation]:
        return [r for r in self.relations.values() if r.subject == entity_id]
    
    def get_relations_to(self, entity_id: str) -> List[Relation]:
        return [r for r in self.relations.values() if r.object == entity_id]
    
    def get_causal_outcomes(self, cause_id: str) -> List[CausalLink]:
        return [c for c in self.causal_links.values() if c.cause_id == cause_id]
    
    def get_causal_antecedents(self, effect_id: str) -> List[CausalLink]:
        return [c for c in self.causal_links.values() if c.effect_id == effect_id]
    
    def check_consistency(self) -> Tuple[bool, List[str]]:
        """
        Check world model consistency.
        
        Returns:
            (is_consistent, list_of_conflicts)
        """
        conflicts = []
        
        # Check for contradictory relations
        for r1 in self.relations.values():
            for r2 in self.relations.values():
                if r1.relation_id != r2.relation_id:
                    if r1.subject == r2.subject and r1.object == r2.object:
                        if r1.predicate != r2.predicate:
                            if abs(r1.confidence - r2.confidence) < 0.3:
                                conflicts.append(
                                    f"Contradictory relations: {r1.predicate} vs {r2.predicate} "
                                    f"for ({r1.subject}, {r1.object})"
                                )
        
        # Check causal chain consistency
        for link in self.causal_links.values():
            if link.cause_id not in self.entities:
                conflicts.append(f"Causal link references non-existent cause: {link.cause_id}")
            if link.effect_id not in self.entities:
                conflicts.append(f"Causal link references non-existent effect: {link.effect_id}")
        
        return len(conflicts) == 0, conflicts
    
    def with_entity(self, entity: Entity) -> 'WorldModel':
        new_entities = {**self.entities, entity.entity_id: entity}
        return WorldModel(
            entities=MappingProxyType(new_entities),
            relations=self.relations,
            causal_links=self.causal_links,
            assertions=self.assertions,
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def with_relation(self, relation: Relation) -> 'WorldModel':
        new_relations = {**self.relations, relation.relation_id: relation}
        return WorldModel(
            entities=self.entities,
            relations=MappingProxyType(new_relations),
            causal_links=self.causal_links,
            assertions=self.assertions,
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def with_causal_link(self, link: CausalLink) -> 'WorldModel':
        new_links = {**self.causal_links, link.cause_id + "_" + link.effect_id: link}
        return WorldModel(
            entities=self.entities,
            relations=self.relations,
            causal_links=MappingProxyType(new_links),
            assertions=self.assertions,
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def with_assertion(self, assertion_id: str, confidence: float) -> 'WorldModel':
        new_assertions = {**self.assertions, assertion_id: confidence}
        return WorldModel(
            entities=self.entities,
            relations=self.relations,
            causal_links=self.causal_links,
            assertions=MappingProxyType(new_assertions),
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def infer_outcomes(self, cause_id: str, conditions: Set[str]) -> List[Tuple[str, float]]:
        """
        Infer likely outcomes from a cause under given conditions.
        
        Returns list of (effect_id, confidence) tuples.
        """
        outcomes = []
        
        for link in self.get_causal_outcomes(cause_id):
            if link.applies_under(conditions):
                outcomes.append((link.effect_id, link.strength))
        
        return outcomes
    
    def find_causal_path(self, start: str, end: str, max_depth: int = 5) -> Optional[List[str]]:
        """
        Find causal path from start to end.
        
        Returns path of entity_ids or None if no path exists.
        """
        if start == end:
            return [start]
        
        if max_depth <= 0:
            return None
        
        for link in self.get_causal_outcomes(start):
            path = self.find_causal_path(link.effect_id, end, max_depth - 1)
            if path:
                return [start] + path
        
        return None


def create_empty_world() -> WorldModel:
    """Create initial empty world model"""
    return WorldModel(
        entities=MappingProxyType({}),
        relations=MappingProxyType({}),
        causal_links=MappingProxyType({}),
        assertions=MappingProxyType({}),
        version=0,
        last_updated=datetime.utcnow().isoformat()
    )