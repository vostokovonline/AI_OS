"""World Model — Projection Layer.

State = Projection(WorldJournal)

Events are the only source of truth.
Entities, relations, and capability scores are computed at projection time.

World Policy v0 (Level 1) — non-invasive interpretation annotation.
Policy annotates events with provenance metadata.
Policy never filters, never blocks, never mutates event semantics.
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

WorldEventType = Literal[
    "EntityCreated",
    "EntityUpdated",
    "RelationObserved",
    "InteractionRecorded",
]


@dataclass(frozen=True)
class WorldEvent:
    type: WorldEventType
    payload: Dict[str, Any]
    timestamp: float


@dataclass
class Entity:
    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    relationships: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class WorldState:
    """Projected state — computed from WorldJournal, never stored directly."""
    entities: Dict[str, Entity]
    relations: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    capability_scores: Dict[str, float]


@dataclass
class ActionPrediction:
    action: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    probability: float = 0.5
    expected_utility: float = 0.0
    risks: List[str] = field(default_factory=list)


@dataclass
class InterpretationMetadata:
    classification: str           # entity_creation | entity_update | relation | interaction | system_event
    confidence: float             # 0.0 – 1.0
    entity_ids: List[str]         # resolved entity references
    pattern_id: str               # rule or heuristic that produced this classification
    reasoning: str                # short human-readable explanation
    policy_version: str           # version of the policy that produced this


class WorldPolicy:
    """
    Level 1 — non-invasive interpretation annotation.

    Annotates events with provenance metadata.
    Never filters, never blocks, never changes event semantics.
    """

    VERSION = "wp_v0"

    def annotate(
        self,
        event_type: str,
        payload: Dict[str, Any],
        state: WorldState,
        source: str = "",
    ) -> Dict[str, Any]:
        meta = self._classify(event_type, payload, state, source)
        return asdict(meta)

    def _classify(
        self,
        event_type: str,
        payload: Dict[str, Any],
        state: WorldState,
        source: str,
    ) -> InterpretationMetadata:
        if event_type == "EntityCreated":
            eid = payload.get("id", "")
            return InterpretationMetadata(
                classification="entity_creation",
                confidence=0.95,
                entity_ids=[eid],
                pattern_id="direct_entity_declaration",
                reasoning=f"Entity '{eid}' declared by caller",
                policy_version=self.VERSION,
            )

        elif event_type == "EntityUpdated":
            eid = payload.get("id", "")
            exists = eid in state.entities
            return InterpretationMetadata(
                classification="entity_update",
                confidence=0.90 if exists else 0.30,
                entity_ids=[eid] if exists else [],
                pattern_id="existing_entity_update" if exists else "orphan_update",
                reasoning=(
                    f"Update to '{eid}' ({'exists' if exists else 'UNKNOWN entity'})"
                ),
                policy_version=self.VERSION,
            )

        elif event_type == "RelationObserved":
            from_id = payload.get("from", "")
            to_id = payload.get("to", "")
            both_exist = from_id in state.entities and to_id in state.entities
            return InterpretationMetadata(
                classification="relation",
                confidence=0.85 if both_exist else 0.40,
                entity_ids=[from_id, to_id],
                pattern_id="direct_relation" if both_exist else "partial_relation",
                reasoning=(
                    f"Relation '{payload.get('relation', '?')}' "
                    f"{'both entities known' if both_exist else 'one or both entities unknown'}"
                ),
                policy_version=self.VERSION,
            )

        elif event_type == "InteractionRecorded":
            actor = payload.get("actor_id", "")
            known = actor in state.entities
            return InterpretationMetadata(
                classification="interaction",
                confidence=0.80 if known else 0.50,
                entity_ids=[actor] if known else ([actor] if actor else []),
                pattern_id="known_actor_interaction" if known else "unknown_actor_interaction",
                reasoning=(
                    f"Action '{payload.get('action', '?')}' by "
                    f"{'known' if known else 'unknown'} actor '{actor}'"
                ),
                policy_version=self.VERSION,
            )

        else:
            return InterpretationMetadata(
                classification="system_event",
                confidence=0.50,
                entity_ids=[],
                pattern_id="unclassified",
                reasoning=f"Unrecognized event type '{event_type}'",
                policy_version=self.VERSION,
            )


class WorldModel:
    """
    World Model — constructs world state as a projection of event history.
    """

    def __init__(self):
        self.world_journal: List[WorldEvent] = []
        self.policy = WorldPolicy()
        logger.info("world_model_initialized")

    # ── Event API (append-only, no mutation) ──────────────────────────────

    def _annotate(self, event_type: str, payload: Dict[str, Any], source: str = "") -> Dict[str, Any]:
        state = self.rebuild_world()
        meta = self.policy.annotate(event_type, payload, state, source)
        return {**payload, "_interpretation": meta}

    def add_entity(self, entity_id: str, type_: str, name: str,
                   properties: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "id": entity_id,
            "type": type_,
            "name": name,
            "properties": properties or {},
        }
        self.world_journal.append(WorldEvent(
            type="EntityCreated",
            payload=self._annotate("EntityCreated", payload, "add_entity"),
            timestamp=self._now(),
        ))
        logger.debug("entity_created", id=entity_id, type=type_, name=name)

    def update_entity(self, entity_id: str, updates: Dict[str, Any]) -> None:
        payload = {"id": entity_id, "updates": updates}
        self.world_journal.append(WorldEvent(
            type="EntityUpdated",
            payload=self._annotate("EntityUpdated", payload, "update_entity"),
            timestamp=self._now(),
        ))
        logger.debug("entity_updated", id=entity_id, updates=updates)

    def add_relation(self, from_id: str, to_id: str, relation: str,
                     properties: Optional[Dict] = None) -> None:
        payload: Dict[str, Any] = {
            "from": from_id, "to": to_id, "relation": relation,
        }
        if properties:
            payload["properties"] = properties
        self.world_journal.append(WorldEvent(
            type="RelationObserved",
            payload=self._annotate("RelationObserved", payload, "add_relation"),
            timestamp=self._now(),
        ))
        logger.debug("relation_observed", from_id=from_id, to_id=to_id,
                     relation=relation)

    def record_interaction(self, actor_id: str, action: str,
                           target_id: Optional[str] = None,
                           outcome: str = "unknown") -> None:
        payload = {
            "id": str(id(self)),
            "timestamp": self._now(),
            "actor_id": actor_id,
            "action": action,
            "target_id": target_id,
            "outcome": outcome,
            "success": outcome == "success",
        }
        self.world_journal.append(WorldEvent(
            type="InteractionRecorded",
            payload=self._annotate("InteractionRecorded", payload, "record_interaction"),
            timestamp=self._now(),
        ))
        logger.debug("interaction_recorded", actor=actor_id, action=action,
                     outcome=outcome)

    # ── Legacy compatibility ──────────────────────────────────────────────

    def record_action(self, actor_id: str, action: str,
                      target_id: Optional[str] = None,
                      outcome: str = "unknown") -> None:
        self.record_interaction(actor_id, action, target_id, outcome)

    # ── Projection ────────────────────────────────────────────────────────

    def rebuild_world(self) -> WorldState:
        """Pure projection: WorldState = f(WorldJournal)."""
        raw_entities: Dict[str, dict] = {}
        raw_relations: List[Dict[str, Any]] = []
        raw_interactions: List[Dict[str, Any]] = []

        for event in self.world_journal:
            if event.type == "EntityCreated":
                raw_entities[event.payload["id"]] = {
                    "id": event.payload["id"],
                    "type": event.payload["type"],
                    "name": event.payload["name"],
                    "properties": dict(event.payload.get("properties", {})),
                    "state": {},
                    "relationships": [],
                }

            elif event.type == "EntityUpdated":
                eid = event.payload["id"]
                if eid not in raw_entities:
                    continue
                raw_entities[eid]["state"].update(event.payload["updates"])

            elif event.type == "RelationObserved":
                raw_relations.append(event.payload)
                rel = {
                    "type": event.payload["relation"],
                    "target": event.payload["to"],
                }
                props = event.payload.get("properties", {})
                if props:
                    rel.update(props)
                src = raw_entities.get(event.payload["from"])
                if src is not None:
                    src["relationships"].append(rel)

            elif event.type == "InteractionRecorded":
                raw_interactions.append(event.payload)

        return WorldState(
            entities={
                eid: Entity(**data)
                for eid, data in raw_entities.items()
            },
            relations=raw_relations,
            interactions=raw_interactions,
            capability_scores=self._compute_capabilities(raw_interactions),
        )

    @staticmethod
    def _compute_capabilities(
            interactions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        scores: Dict[str, Dict[str, int]] = {}

        for i in interactions:
            agent = i.get("actor_id")
            if not agent:
                continue
            if agent not in scores:
                scores[agent] = {"success": 0, "fail": 0}
            if i.get("success", False):
                scores[agent]["success"] += 1
            else:
                scores[agent]["fail"] += 1

        return {
            agent: s["success"] / max(1, s["success"] + s["fail"])
            for agent, s in scores.items()
        }

    # ── Query API (operates on projected state) ───────────────────────────

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.rebuild_world().entities.get(entity_id)

    def get_related_entities(
        self, entity_id: str, relation_type: Optional[str] = None,
    ) -> List[Entity]:
        state = self.rebuild_world()
        entity = state.entities.get(entity_id)
        if entity is None:
            return []
        related_ids = []
        for rel in entity.relationships:
            if relation_type is None or rel.get("type") == relation_type:
                related_ids.append(rel.get("target", rel.get("id")))
        return [state.entities[eid] for eid in related_ids if eid in state.entities]

    # ── Policy Layer ──────────────────────────────────────────────────────

    async def predict_action(
        self, actor_id: str, action: str, target_id: Optional[str] = None,
    ) -> ActionPrediction:
        """Predict outcome — pure policy over projected state."""
        state = self.rebuild_world()
        prediction = ActionPrediction(action=action)

        actor = state.entities.get(actor_id)
        if actor is None:
            prediction.probability = 0.1
            prediction.risks.append("Unknown actor")
            return prediction

        if action == "execute_goal":
            prediction.preconditions = ["goal_exists", "resources_available"]
            prediction.postconditions = ["goal_completed", "resources_consumed"]
            prediction.probability = 0.7

        elif action == "create_goal":
            prediction.preconditions = ["context_understood"]
            prediction.postconditions = ["goal_created", "dependencies_established"]
            prediction.probability = 0.9

        elif action == "decompose_goal":
            prediction.preconditions = ["goal_complex", "capacity_available"]
            prediction.postconditions = ["subgoals_created", "hierarchy_formed"]
            prediction.probability = 0.8

        elif action == "use_skill":
            prediction.preconditions = ["skill_available", "inputs_provided"]
            prediction.postconditions = ["skill_executed", "outputs_produced"]
            prediction.probability = 0.85

        # Historical probability from interactions
        history_matches = [
            i for i in state.interactions
            if i.get("action") == action and i.get("actor_id") == actor_id
        ]
        if history_matches:
            successes = sum(1 for i in history_matches if i.get("outcome") == "success")
            prediction.probability = successes / len(history_matches)

        # Capability adjustment
        cap = state.capability_scores.get(actor_id, 0.5)
        prediction.probability *= cap

        prediction.expected_utility = prediction.probability * (
            1 - sum(0.1 for _ in prediction.risks)
        )

        return prediction

    def get_world_state(self) -> Dict[str, Any]:
        state = self.rebuild_world()
        return {
            "entities_count": len(state.entities),
            "actions_count": len(state.interactions),
            "entity_types": self._count_by_type(state),
            "recent_actions": [
                {"action": i["action"], "outcome": i["outcome"]}
                for i in state.interactions[-10:]
            ],
        }

    @staticmethod
    def _count_by_type(state: WorldState) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entity in state.entities.values():
            counts[entity.type] = counts.get(entity.type, 0) + 1
        return counts

    @staticmethod
    def _now() -> float:
        return datetime.utcnow().timestamp()
