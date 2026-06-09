"""
Causal Attribution Engine - Semantic Cause Extraction

Извлекает semantic causes из event stream:

Event Stream (syntactic causality)
    ↓
Latent State Evolution
    ↓
Semantic Cause Extraction
    ↓
Causal Attribution Graph

Вместо просто "event happened after event",
система теперь понимает:
- Что именно изменилось
- Почему решение стало возможным
- Какой latent factor реально повлиял
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class LatentCause:
    """
    Semantic latent cause.
    
    Вместо "stress increased" (syntactic),
    теперь: "prediction_conflict" (semantic).
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    cause_type: str = ""  # uncertainty_spike, reward_expectation_drop, prediction_conflict, etc.
    description: str = ""
    
    # Strength metrics
    causal_strength: float = 0.0  # How much did this cause influence?
    counterfactual_support: float = 0.0  # How well does counterfactual support this?
    temporal_confidence: float = 0.0  # How consistent over time?
    
    # Evidence
    affected_features: List[str] = field(default_factory=list)
    evidence_events: List[str] = field(default_factory=list)
    
    # Temporal
    first_detected: Optional[datetime] = None
    last_detected: Optional[datetime] = None
    occurrence_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "cause_type": self.cause_type,
            "description": self.description,
            "causal_strength": self.causal_strength,
            "counterfactual_support": self.counterfactual_support,
            "temporal_confidence": self.temporal_confidence,
            "affected_features": self.affected_features,
            "occurrence_count": self.occurrence_count,
            "first_detected": self.first_detected.isoformat() if self.first_detected else None,
            "last_detected": self.last_detected.isoformat() if self.last_detected else None,
        }


@dataclass
class CausalEdge:
    """
    Causal edge в semantic causal graph.
    
    Каждая edge имеет:
    - from_cause → to_effect
    - causal_strength (0-1)
    - evidence chain
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    
    from_cause_id: str = ""
    to_effect_id: str = ""
    
    causal_strength: float = 0.0
    confidence: float = 0.0
    
    # Evidence chain
    supporting_events: List[str] = field(default_factory=list)
    alternative_explanations: List[str] = field(default_factory=list)
    
    # Mechanism
    mechanism: str = ""  # "direct_influence", "mediated", "enabling", "inhibiting"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "from_cause": self.from_cause_id,
            "to_effect": self.to_effect_id,
            "causal_strength": self.causal_strength,
            "confidence": self.confidence,
            "mechanism": self.mechanism,
            "supporting_events": self.supporting_events,
        }


@dataclass
class SemanticTransition:
    """
    Semantic transition между состояниями.
    
    Это заменяет простой StateDelta.
    
    Теперь включает:
    - latent_causes (что вызвало изменение)
    - causal_attribution (как это было вызвано)
    - mechanism (каким образом)
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    from_state_id: str = ""
    to_state_id: str = ""
    
    # Semantic content
    latent_causes: List[LatentCause] = field(default_factory=list)
    causal_edges: List[CausalEdge] = field(default_factory=list)
    
    # Transition properties
    transition_type: str = ""  # "confidence_change", "stress_shift", "strategy_switch"
    novelty: float = 0.0
    predictability: float = 0.0
    
    # Outcome
    action_taken: Optional[str] = None
    outcome: Optional[str] = None
    outcome_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "from_state": self.from_state_id,
            "to_state": self.to_state_id,
            "transition_type": self.transition_type,
            "latent_causes": [c.to_dict() for c in self.latent_causes],
            "causal_edges": [e.to_dict() for e in self.causal_edges],
            "novelty": self.novelty,
            "predictability": self.predictability,
            "action": self.action_taken,
            "outcome": self.outcome,
        }


class CauseDetector:
    """
    Detects latent causes from state changes.
    
    Сейчас rule-based. Future: learned from trajectories.
    """
    
    KNOWN_CAUSE_TYPES = {
        "uncertainty_spike": {
            "description": "Increased uncertainty about outcome",
            "indicators": ["confidence_drop", "novelty_increase", "stress_increase"],
            "threshold": 0.3,
        },
        "prediction_conflict": {
            "description": "Actual outcome differs from prediction",
            "indicators": ["high_prediction_error", "surprise"],
            "threshold": 0.4,
        },
        "reward_expectation_drop": {
            "description": "Expected reward decreased",
            "indicators": ["valence_decrease", "confidence_drop"],
            "threshold": 0.25,
        },
        "capability_assessment_update": {
            "description": "Self-assessment of capability changed",
            "indicators": ["confidence_change", "action_readiness_change"],
            "threshold": 0.2,
        },
        "stress_accumulation": {
            "description": "System stress increased",
            "indicators": ["stress_increase", "focus_decrease"],
            "threshold": 0.15,
        },
        "strategy_failure": {
            "description": "Previous strategy failed",
            "indicators": ["outcome_failure", "confidence_drop"],
            "threshold": 0.1,
        },
        "novel_context": {
            "description": "Encountered novel task context",
            "indicators": ["novelty_increase", "task_novelty"],
            "threshold": 0.5,
        },
        "reflection_triggered": {
            "description": "Deep reflection triggered",
            "indicators": ["reflection_depth_increase", "bias_awareness_increase"],
            "threshold": 0.3,
        },
    }
    
    def __init__(self):
        self.detected_causes: Dict[str, LatentCause] = {}
        logger.info("cause_detector_initialized")
    
    def detect_causes(
        self,
        from_state: Dict,
        to_state: Dict,
        outcome: Optional[str] = None
    ) -> List[LatentCause]:
        """Detect latent causes from state transition"""
        causes = []
        
        delta = {
            k: to_state.get(k, 0) - from_state.get(k, 0)
            for k in from_state.keys()
        }
        
        # Check each known cause type
        for cause_type, config in self.KNOWN_CAUSE_TYPES.items():
            score = self._compute_cause_score(cause_type, delta, from_state, to_state, outcome)
            
            if score >= config["threshold"]:
                cause = self._create_cause(cause_type, score, delta, from_state, to_state)
                causes.append(cause)
                self._update_cause_tracking(cause)
        
        logger.debug("causes_detected", count=len(causes), types=[c.cause_type for c in causes])
        
        return causes
    
    def _compute_cause_score(
        self,
        cause_type: str,
        delta: Dict,
        from_state: Dict,
        to_state: Dict,
        outcome: Optional[str]
    ) -> float:
        """Compute score for a cause type"""
        config = self.KNOWN_CAUSE_TYPES[cause_type]
        indicators = config["indicators"]
        
        scores = []
        for indicator in indicators:
            score = self._indicator_score(indicator, delta, from_state, to_state, outcome)
            scores.append(score)
        
        return max(scores) if scores else 0.0
    
    def _indicator_score(
        self,
        indicator: str,
        delta: Dict,
        from_state: Dict,
        to_state: Dict,
        outcome: Optional[str]
    ) -> float:
        """Compute score for a single indicator"""
        if indicator == "confidence_drop":
            conf_delta = delta.get("confidence", 0)
            return max(0, -conf_delta)
        
        if indicator == "confidence_change":
            conf_delta = abs(delta.get("confidence", 0))
            return conf_delta
        
        if indicator == "stress_increase":
            stress_delta = delta.get("stress_level", 0)
            return max(0, stress_delta)
        
        if indicator == "novelty_increase":
            novelty_delta = delta.get("task_novelty", 0)
            return max(0, novelty_delta)
        
        if indicator == "valence_decrease":
            val_delta = delta.get("valence", 0)
            return max(0, -val_delta)
        
        if indicator == "focus_decrease":
            focus_delta = delta.get("focus", 0)
            return max(0, -focus_delta)
        
        if indicator == "action_readiness_change":
            ar_delta = abs(delta.get("action_readiness", 0))
            return ar_delta
        
        if indicator == "reflection_depth_increase":
            rd_delta = delta.get("reflection_depth", 0)
            return max(0, rd_delta)
        
        if indicator == "bias_awareness_increase":
            ba_delta = delta.get("bias_awareness", 0)
            return max(0, ba_delta)
        
        if indicator == "high_prediction_error":
            return 0.5 if outcome == "failure" else 0.0
        
        if indicator == "outcome_failure":
            return 1.0 if outcome == "failure" else 0.0
        
        if indicator == "surprise":
            surprise_score = abs(delta.get("confidence", 0)) + abs(delta.get("stress_level", 0))
            return min(1.0, surprise_score)
        
        return 0.0
    
    def _create_cause(
        self,
        cause_type: str,
        score: float,
        delta: Dict,
        from_state: Dict,
        to_state: Dict
    ) -> LatentCause:
        """Create a latent cause"""
        config = self.KNOWN_CAUSE_TYPES[cause_type]
        
        affected = []
        for indicator in config["indicators"]:
            base = indicator.replace("_increase", "").replace("_decrease", "").replace("_change", "").replace("_drop", "")
            if delta.get(base, 0) != 0 or delta.get(base + "_increase", 0) != 0 or delta.get(base + "_decrease", 0) != 0:
                affected.append(base)
        
        return LatentCause(
            cause_type=cause_type,
            description=config["description"],
            causal_strength=score,
            counterfactual_support=score * 0.8,  # Placeholder
            temporal_confidence=score,
            affected_features=list(set(affected)),
            first_detected=datetime.utcnow(),
            last_detected=datetime.utcnow(),
            occurrence_count=1,
        )
    
    def _update_cause_tracking(self, cause: LatentCause) -> None:
        """Update cause tracking statistics"""
        existing = self.detected_causes.get(cause.cause_type)
        if existing:
            existing.occurrence_count += 1
            existing.last_detected = datetime.utcnow()
            existing.causal_strength = (
                existing.causal_strength * 0.9 + cause.causal_strength * 0.1
            )
        else:
            self.detected_causes[cause.cause_type] = cause
    
    def get_dominant_causes(self, limit: int = 5) -> List[LatentCause]:
        """Get most frequent/strong causes"""
        causes = list(self.detected_causes.values())
        causes.sort(key=lambda c: c.causal_strength * c.occurrence_count, reverse=True)
        return causes[:limit]


class CausalAttributionEngine:
    """
    Causal Attribution Engine - Extracts semantic causes from events.
    
    Это заменяет IncrementalAttribution.
    
    Теперь система понимает:
    - Что именно изменилось (cause)
    - Почему (semantic reason)
    - Какой latent factor повлиял
    """
    
    def __init__(self):
        self.cause_detector = CauseDetector()
        self.transitions: List[SemanticTransition] = []
        self.causal_graph: Dict[str, List[str]] = {}
        logger.info("causal_attribution_engine_initialized")
    
    def attribute_transition(
        self,
        from_state_id: str,
        to_state_id: str,
        from_state: Dict,
        to_state: Dict,
        action: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> SemanticTransition:
        """Attribute a state transition to latent causes"""
        
        causes = self.cause_detector.detect_causes(from_state, to_state, outcome)
        
        edges = self._build_causal_edges(causes, action, outcome)
        
        transition_type = self._classify_transition(from_state, to_state)
        
        novelty = self._compute_novelty(from_state, to_state)
        predictability = self._compute_predictability(causes, edges)
        
        transition = SemanticTransition(
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            latent_causes=causes,
            causal_edges=edges,
            transition_type=transition_type,
            novelty=novelty,
            predictability=predictability,
            action_taken=action,
            outcome=outcome,
            outcome_score={"success": 1.0, "partial": 0.5, "failure": 0.0}.get(outcome, 0.5)
        )
        
        self.transitions.append(transition)
        self._update_causal_graph(transition)
        
        logger.info(
            "transition_attributed",
            from_state=from_state_id,
            to_state=to_state_id,
            causes=len(causes),
            edges=len(edges),
            type=transition_type
        )
        
        return transition
    
    def _build_causal_edges(
        self,
        causes: List[LatentCause],
        action: Optional[str],
        outcome: Optional[str]
    ) -> List[CausalEdge]:
        """Build causal edges from detected causes"""
        edges = []
        
        for cause in causes:
            effect_id = action or "no_action"
            
            edge = CausalEdge(
                from_cause_id=cause.cause_type,
                to_effect_id=effect_id,
                causal_strength=cause.causal_strength,
                confidence=cause.causal_strength * cause.temporal_confidence,
                mechanism=self._classify_mechanism(cause, action),
                supporting_events=[cause.id]
            )
            edges.append(edge)
        
        return edges
    
    def _classify_transition(self, from_state: Dict, to_state: Dict) -> str:
        """Classify the type of transition"""
        conf_delta = abs(to_state.get("confidence", 0.5) - from_state.get("confidence", 0.5))
        stress_delta = abs(to_state.get("stress_level", 0) - from_state.get("stress_level", 0))
        
        if conf_delta > 0.2:
            return "confidence_change"
        elif stress_delta > 0.15:
            return "stress_shift"
        elif to_state.get("task_novelty", 0) > from_state.get("task_novelty", 0):
            return "novel_context"
        else:
            return "incremental_update"
    
    def _classify_mechanism(self, cause: LatentCause, action: Optional[str]) -> str:
        """Classify the causal mechanism"""
        if cause.cause_type in ["prediction_conflict", "strategy_failure"]:
            return "direct_influence"
        elif cause.cause_type in ["novel_context", "reflection_triggered"]:
            return "enabling"
        elif cause.cause_type in ["uncertainty_spike", "reward_expectation_drop"]:
            return "inhibiting"
        else:
            return "mediated"
    
    def _compute_novelty(self, from_state: Dict, to_state: Dict) -> float:
        """Compute novelty of transition"""
        deltas = [abs(to_state.get(k, 0) - from_state.get(k, 0)) for k in from_state.keys()]
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        return min(1.0, avg_delta * 2)
    
    def _compute_predictability(self, causes: List[LatentCause], edges: List[CausalEdge]) -> float:
        """Compute predictability of transition"""
        if not causes:
            return 0.5
        
        avg_strength = sum(c.causal_strength for c in causes) / len(causes)
        return avg_strength
    
    def _update_causal_graph(self, transition: SemanticTransition) -> None:
        """Update the causal graph"""
        for edge in transition.causal_edges:
            if edge.from_cause_id not in self.causal_graph:
                self.causal_graph[edge.from_cause_id] = []
            self.causal_graph[edge.from_cause_id].append(edge.to_effect_id)
    
    def get_causal_path(
        self,
        from_state_id: str,
        to_state_id: str
    ) -> Optional[List[Dict]]:
        """Get causal path for a transition"""
        for transition in reversed(self.transitions):
            if transition.from_state_id == from_state_id and transition.to_state_id == to_state_id:
                return [
                    {"type": "cause", "data": c.to_dict()}
                    for c in transition.latent_causes
                ] + [
                    {"type": "edge", "data": e.to_dict()}
                    for e in transition.causal_edges
                ]
        return None
    
    def get_dominant_causes(self, limit: int = 5) -> List[Dict]:
        """Get dominant causes across all transitions"""
        causes = self.cause_detector.get_dominant_causes(limit)
        return [c.to_dict() for c in causes]
    
    def get_statistics(self) -> Dict:
        """Get attribution statistics"""
        return {
            "total_transitions": len(self.transitions),
            "total_causes_detected": sum(len(t.latent_causes) for t in self.transitions),
            "total_edges": sum(len(t.causal_edges) for t in self.transitions),
            "dominant_causes": self.get_dominant_causes(5),
            "transition_types": self._count_transition_types(),
        }
    
    def _count_transition_types(self) -> Dict[str, int]:
        counts = {}
        for t in self.transitions:
            counts[t.transition_type] = counts.get(t.transition_type, 0) + 1
        return counts


class CounterfactualReasoner:
    """
    Counterfactual Reasoner - What if analysis.
    
    Использует transition model для:
    - "What if we chose different action?"
    - "What would happen if stress was lower?"
    """
    
    def __init__(self, latent_space, transition_model):
        self.latent_space = latent_space
        self.transition_model = transition_model
        logger.info("counterfactual_reasoner_initialized")
    
    def reason_counterfactual(
        self,
        current_state_id: str,
        actual_action: str,
        alternative_action: str
    ) -> Dict:
        """Reason about counterfactual"""
        current_state = self.latent_space.get_state(current_state_id)
        if not current_state:
            return {"error": "State not found"}
        
        counterfactual = self.transition_model.counterfactual(
            current_state,
            actual_action,
            alternative_action
        )
        
        return counterfactual
    
    def predict_branch_outcomes(
        self,
        current_state_id: str,
        possible_actions: List[str]
    ) -> List[Dict]:
        """Predict outcomes for all possible actions"""
        current_state = self.latent_space.get_state(current_state_id)
        if not current_state:
            return []
        
        predictions = []
        for action in possible_actions:
            predicted_vector = self.transition_model.predict_next_state(current_state, action)
            
            predictions.append({
                "action": action,
                "predicted_state": predicted_vector,
                "confidence": 0.7,  # Placeholder
                "expected_utility": sum(predicted_vector) / len(predicted_vector),
            })
        
        return sorted(predictions, key=lambda x: x["expected_utility"], reverse=True)


# Factory function
def create_causal_attribution_engine() -> CausalAttributionEngine:
    return CausalAttributionEngine()