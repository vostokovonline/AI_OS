"""
Decision Transaction - First-class atomic construct for cognitive decisions

This is the single unified boundary for all observability components.

A DecisionTransaction wraps:
- state_snapshot (t0) - UnifiedState before decision
- reasoning_steps - Ordered causal chain of reasoning events
- policy_evaluation - Scoring details and candidates
- attribution_graph - Causal feature attribution
- final_action - Selected action with confidence
- outcome (t+1) - Result after execution

This ensures:
- Transactional integrity (all-or-nothing)
- Causal alignment (state → action with proof)
- Replay capability (exact rerun possible)
- Dataset quality (clean training data for RL)
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TransactionStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class EventType(Enum):
    STATE_COLLECTED = "state_collected"
    CANDIDATE_GENERATION = "candidate_generation"
    POLICY_SCORING = "policy_scoring"
    SIMULATION_STARTED = "simulation_started"
    SIMULATION_BRANCH = "simulation_branch"
    SIMULATION_SELECTED = "simulation_selected"
    ACTION_SELECTED = "action_selected"
    ACTION_EXECUTED = "action_executed"
    OUTCOME_RECEIVED = "outcome_received"
    LEARNING_INTEGRATED = "learning_integrated"


@dataclass
class ReasoningEvent:
    """Single event in the reasoning timeline"""
    id: str = field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sequence: int = 0
    
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "description": self.description,
            "data": self.data,
            "duration_ms": self.duration_ms,
        }


@dataclass
class StateSnapshot:
    """State at decision time (t0)"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    world_outcome: str = "unknown"
    world_entities: int = 0
    world_capability: float = 0.5
    
    identity_coherence: float = 0.5
    identity_emotion: str = "neutral"
    
    arousal: float = 0.5
    valence: float = 0.0
    focus: float = 0.5
    confidence: float = 0.5
    
    bias_count: int = 0
    bias_awareness: float = 0.5
    reflection_depth: float = 0.5
    
    top_strategy: str = "default"
    strategy_score: float = 0.5
    
    stress_level: float = 0.0
    exploration_tendency: float = 0.5
    action_readiness: float = 0.5
    
    task_complexity: float = 0.5
    task_urgency: float = 0.5
    task_novelty: float = 0.5
    
    vector: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "world_outcome": self.world_outcome,
            "world_entities": self.world_entities,
            "world_capability": self.world_capability,
            "identity_coherence": self.identity_coherence,
            "identity_emotion": self.identity_emotion,
            "arousal": self.arousal,
            "valence": self.valence,
            "focus": self.focus,
            "confidence": self.confidence,
            "bias_count": self.bias_count,
            "bias_awareness": self.bias_awareness,
            "reflection_depth": self.reflection_depth,
            "top_strategy": self.top_strategy,
            "strategy_score": self.strategy_score,
            "stress_level": self.stress_level,
            "exploration_tendency": self.exploration_tendency,
            "action_readiness": self.action_readiness,
            "task_complexity": self.task_complexity,
            "task_urgency": self.task_urgency,
            "task_novelty": self.task_novelty,
        }
    
    @classmethod
    def from_unified_state(cls, state) -> "StateSnapshot":
        return cls(
            timestamp=datetime.utcnow(),
            world_outcome=state.world_recent_outcome,
            world_entities=state.world_entities_count,
            world_capability=state.world_capability_score,
            identity_coherence=state.identity_coherence,
            identity_emotion=state.identity_emotion,
            arousal=state.arousal,
            valence=state.valence,
            focus=state.focus,
            confidence=state.confidence,
            bias_count=state.bias_count,
            bias_awareness=state.bias_awareness,
            reflection_depth=state.reflection_depth,
            top_strategy=state.top_strategy_name,
            strategy_score=state.top_strategy_score,
            stress_level=state.stress_level,
            exploration_tendency=state.exploration_tendency,
            action_readiness=state.action_readiness,
            task_complexity=state.task_complexity,
            task_urgency=state.task_urgency,
            task_novelty=state.task_novelty,
            vector=state.to_vector(),
        )


@dataclass
class CandidateEvaluation:
    """Evaluation of a single candidate action"""
    candidate_id: str = ""
    action_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    score: float = 0.0
    risk: float = 0.0
    utility: float = 0.0
    
    simulation_used: bool = False
    simulation_results: List[Dict] = field(default_factory=list)
    
    scoring_breakdown: Dict[str, float] = field(default_factory=dict)
    
    selected: bool = False
    selection_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "candidate_id": self.candidate_id,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "score": self.score,
            "risk": self.risk,
            "utility": self.utility,
            "simulation_used": self.simulation_used,
            "simulation_results": self.simulation_results,
            "scoring_breakdown": self.scoring_breakdown,
            "selected": self.selected,
            "selection_reason": self.selection_reason,
        }


@dataclass
class AttributionEdge:
    """Causal edge in the attribution graph"""
    from_node: str = ""
    to_node: str = ""
    contribution: float = 0.0
    reason: str = ""
    polarity: str = "positive"


@dataclass
class AttributionSnapshot:
    """Full causal attribution for the decision"""
    primary_features: List[Dict] = field(default_factory=list)
    causal_paths: List[Dict] = field(default_factory=list)
    alternative_explanations: List[Dict] = field(default_factory=list)
    
    edges: List[AttributionEdge] = field(default_factory=list)
    
    final_explanation: str = ""
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "primary_features": self.primary_features,
            "causal_paths": self.causal_paths,
            "alternative_explanations": self.alternative_explanations,
            "edges": [
                {"from": e.from_node, "to": e.to_node, "contribution": e.contribution}
                for e in self.edges
            ],
            "final_explanation": self.final_explanation,
            "confidence_score": self.confidence_score,
        }


@dataclass
class OutcomeRecord:
    """Recorded outcome after execution"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    outcome: str = ""
    score: float = 0.0
    
    state_after: Optional[StateSnapshot] = None
    
    learning_integrated: bool = False
    strategy_updated: bool = False
    bias_detected: List[str] = field(default_factory=list)
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "outcome": self.outcome,
            "score": self.score,
            "state_after": self.state_after.to_dict() if self.state_after else None,
            "learning_integrated": self.learning_integrated,
            "strategy_updated": self.strategy_updated,
            "bias_detected": self.bias_detected,
            "metrics": self.metrics,
        }


@dataclass
class DecisionTransaction:
    """
    First-class atomic construct for a complete decision transaction.
    
    This is the single boundary that unifies all observability:
    - State at t0
    - Reasoning timeline
    - Policy evaluation
    - Attribution
    - Final action
    - Outcome at t+1
    
    Usage:
        async with DecisionTransaction(agent, user_id, task) as txn:
            # Decision is automatically traced
            await txn.decide(context)
            
            # Outcome automatically recorded
            await txn.record_outcome("success")
        
        # Transaction now complete with full causal graph
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = TransactionStatus.CREATED.value
    
    user_id: str = ""
    task: str = ""
    trace_id: Optional[str] = None
    
    state_before: Optional[StateSnapshot] = None
    reasoning_events: List[ReasoningEvent] = field(default_factory=list)
    
    candidates: List[CandidateEvaluation] = field(default_factory=list)
    selected_action: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    
    attribution: Optional[AttributionSnapshot] = None
    
    outcome: Optional[OutcomeRecord] = None
    
    completed_at: Optional[datetime] = None
    total_duration_ms: float = 0.0
    
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "user_id": self.user_id,
            "task": self.task,
            "trace_id": self.trace_id,
            "state_before": self.state_before.to_dict() if self.state_before else None,
            "reasoning_events": [e.to_dict() for e in self.reasoning_events],
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_action": self.selected_action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "attribution": self.attribution.to_dict() if self.attribution else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_ms": self.total_duration_ms,
            "error": self.error,
        }
    
    def is_complete(self) -> bool:
        return self.status == TransactionStatus.COMPLETED.value
    
    def is_valid(self) -> bool:
        return (
            self.state_before is not None and
            self.selected_action is not None and
            len(self.reasoning_events) > 0
        )
    
    def get_causal_chain(self) -> List[Dict]:
        """Get complete causal chain for analysis"""
        chain = []
        
        if self.state_before:
            chain.append({
                "step": 0,
                "event": "state_snapshot",
                "timestamp": self.state_before.timestamp.isoformat(),
                "data": self.state_before.to_dict()
            })
        
        for i, event in enumerate(self.reasoning_events):
            chain.append({
                "step": event.sequence,
                "event": event.event_type,
                "timestamp": event.timestamp.isoformat(),
                "duration_ms": event.duration_ms,
                "description": event.description,
                "data": event.data
            })
        
        if self.attribution:
            chain.append({
                "step": len(self.reasoning_events) + 1,
                "event": "attribution_computed",
                "data": self.attribution.to_dict()
            })
        
        if self.outcome:
            chain.append({
                "step": len(self.reasoning_events) + 2,
                "event": "outcome_recorded",
                "timestamp": self.outcome.timestamp.isoformat(),
                "data": self.outcome.to_dict()
            })
        
        return chain


class TransactionContext:
    """Context manager for decision transactions"""
    
    def __init__(self, agent, user_id: str, task: str):
        self.agent = agent
        self.transaction = DecisionTransaction(
            user_id=user_id,
            task=task,
            status=TransactionStatus.CREATED.value
        )
        self._event_sequence = 0
        self._start_time: Optional[datetime] = None
    
    async def __aenter__(self) -> "TransactionContext":
        self._start_time = datetime.utcnow()
        self.transaction.status = TransactionStatus.ACTIVE.value
        self.agent._current_transaction = self.transaction
        
        trace = self.agent.tracer.start_trace(self.transaction.user_id, self.transaction.task)
        self.transaction.trace_id = trace.trace_id
        
        logger.info("transaction_started", transaction_id=self.transaction.id)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.transaction.status = TransactionStatus.FAILED.value
            self.transaction.error = str(exc_val)
            logger.error("transaction_failed", transaction_id=self.transaction.id, error=str(exc_val))
        else:
            self.transaction.status = TransactionStatus.COMPLETED.value
        
        self.transaction.completed_at = datetime.utcnow()
        if self._start_time:
            self.transaction.total_duration_ms = (
                self.transaction.completed_at - self._start_time
            ).total_seconds() * 1000
        
        if self.transaction.trace_id:
            self.agent.tracer.end_trace(self.transaction.trace_id)
        
        self.agent._current_transaction = None
        
        self.agent._transaction_history.append(self.transaction)
        if len(self.agent._transaction_history) > 1000:
            self.agent._transaction_history = self.agent._transaction_history[-1000:]
        
        logger.info(
            "transaction_completed",
            transaction_id=self.transaction.id,
            status=self.transaction.status,
            duration_ms=self.transaction.total_duration_ms
        )
        
        return False
    
    def add_event(
        self,
        event_type: str,
        description: str,
        data: Optional[Dict] = None,
        duration_ms: float = 0.0
    ) -> ReasoningEvent:
        """Add a reasoning event to the timeline"""
        self._event_sequence += 1
        event = ReasoningEvent(
            event_type=event_type,
            sequence=self._event_sequence,
            description=description,
            data=data or {},
            duration_ms=duration_ms
        )
        self.transaction.reasoning_events.append(event)
        
        if self.transaction.trace_id:
            self.agent.tracer.add_reasoning_step(
                self.transaction.trace_id,
                event_type,
                data or {}
            )
        
        return event
    
    def set_state(self, state) -> None:
        """Record state snapshot at t0"""
        self.transaction.state_before = StateSnapshot.from_unified_state(state)
        self.add_event(
            EventType.STATE_COLLECTED.value,
            "Initial state collected",
            {"world_outcome": state.world_recent_outcome, "confidence": state.confidence}
        )
    
    def set_candidates(self, candidates: List[CandidateEvaluation]) -> None:
        """Record candidate evaluations"""
        self.transaction.candidates = candidates
        self.add_event(
            EventType.CANDIDATE_GENERATION.value,
            f"Generated {len(candidates)} candidates",
            {"candidates": [c.action_type for c in candidates]}
        )
    
    def set_selection(
        self,
        action: str,
        confidence: float,
        reasoning: str,
        attribution: Optional[AttributionSnapshot] = None
    ) -> None:
        """Record final action selection"""
        self.transaction.selected_action = action
        self.transaction.confidence = confidence
        self.transaction.reasoning = reasoning
        self.transaction.attribution = attribution
        
        for i, c in enumerate(self.transaction.candidates):
            if c.action_type == action:
                c.selected = True
                c.selection_reason = reasoning
        
        self.add_event(
            EventType.ACTION_SELECTED.value,
            f"Selected action: {action} (confidence: {confidence:.2f})",
            {"action": action, "confidence": confidence, "reasoning": reasoning[:200]}
        )
    
    def add_simulation_event(
        self,
        branch: str,
        evaluation: Dict,
        selected: bool
    ) -> None:
        """Record simulation branch event"""
        self.add_event(
            EventType.SIMULATION_BRANCH.value if not selected else EventType.SIMULATION_SELECTED.value,
            f"Simulation branch: {branch}",
            {"branch": branch, "evaluation": evaluation, "selected": selected}
        )
    
    def record_outcome(
        self,
        outcome: str,
        score: float,
        state_after: Optional[StateSnapshot] = None,
        learning: Optional[Dict] = None
    ) -> None:
        """Record outcome at t+1"""
        self.transaction.outcome = OutcomeRecord(
            outcome=outcome,
            score=score,
            state_after=state_after,
            learning_integrated=learning is not None,
            strategy_updated=learning.get("strategy_updated", False) if learning else False,
            bias_detected=learning.get("bias_detected", []) if learning else [],
            metrics=learning.get("metrics", {}) if learning else {}
        )
        
        self.add_event(
            EventType.OUTCOME_RECEIVED.value,
            f"Outcome recorded: {outcome} (score: {score:.2f})",
            {"outcome": outcome, "score": score}
        )
    
    def get_transaction(self) -> DecisionTransaction:
        """Get the completed transaction"""
        return self.transaction
    
    async def decide(
        self,
        context,
        use_simulation: bool = True
    ) -> None:
        """Execute decision within transaction context"""
        from ..policy.unified_agent import agent_decide_in_transaction
        
        self.add_event(
            EventType.STATE_COLLECTED.value,
            "Starting decision with full observability"
        )
        
        state = await self.agent.state_builder.build_state({
            "task": context.task,
            "task_type": context.task_type,
            "complexity": context.complexity,
            "urgency": context.urgency,
            "novelty": context.novelty
        })
        
        self.set_state(state)
        self.agent.diff_engine.record_state(state)
        
        self.add_event(
            EventType.CANDIDATE_GENERATION.value,
            "Generating candidate actions",
            {"complexity": context.complexity}
        )
        
        policy_action = await self.agent.policy.decide(
            context={
                "task": context.task,
                "task_type": context.task_type,
                "complexity": context.complexity,
                "urgency": context.urgency,
                "novelty": context.novelty
            },
            use_planning=use_simulation
        )
        
        candidates = []
        for alt_action, alt_score in policy_action.alternatives:
            candidates.append(CandidateEvaluation(
                candidate_id=alt_action.value,
                action_type=alt_action.value,
                parameters=policy_action.parameters,
                score=alt_score,
                selected=(alt_action == policy_action.action_type)
            ))
        self.set_candidates(candidates)
        
        self.add_event(
            EventType.POLICY_SCORING.value,
            "Policy scoring complete",
            {"score": policy_action.confidence, "alternatives": len(policy_action.alternatives)}
        )
        
        if use_simulation:
            self.add_event(
                EventType.SIMULATION_STARTED.value,
                "Simulation loop active",
                {"simulation": True}
            )
        
        state_dict = {
            "confidence": state.confidence,
            "stress_level": state.stress_level,
            "action_readiness": state.action_readiness,
            "arousal": state.arousal,
            "valence": state.valence,
            "focus": state.focus,
            "bias_awareness": state.bias_awareness,
            "reflection_depth": state.reflection_depth,
            "exploration_tendency": state.exploration_tendency,
            "task_complexity": state.task_complexity,
            "task_urgency": state.task_urgency,
            "task_novelty": state.task_novelty,
        }
        
        attribution = IncrementalAttribution(self.transaction.id)
        attribution.begin(state_dict)
        attribution.add_event("state_collected", "Initial state", state_dict)
        attribution.add_event("candidate_generation", f"Generated {len(candidates)} candidates")
        attribution.add_event("policy_scoring", f"Best score: {policy_action.confidence:.2f}")
        
        for candidate in candidates:
            attribution.add_candidate(
                candidate.candidate_id,
                candidate.action_type,
                candidate.parameters,
                candidate.score,
                candidate.scoring_breakdown
            )
        
        attribution.finalize(
            policy_action.action_type.value,
            policy_action.confidence,
            [(a[0].value, a[1]) for a in policy_action.alternatives]
        )
        
        self.set_selection(
            policy_action.action_type.value,
            policy_action.confidence,
            policy_action.reasoning,
            attribution.finalize(
                policy_action.action_type.value,
                policy_action.confidence,
                [(a[0].value, a[1]) for a in policy_action.alternatives]
            )
        )
        
        self.agent.cognitive_os.world_model.record_action(
            actor_id=context.user_id,
            action=f"decide_{policy_action.action_type.value}",
            outcome="decided"
        )
        
        logger.info(
            "transaction_decision_made",
            transaction_id=self.transaction.id,
            action=policy_action.action_type.value,
            confidence=policy_action.confidence
        )


class IncrementalAttribution:
    """
    Incremental attribution builder - records causal timeline.
    
    This replaces the "snapshot" model with a "causal timeline" model:
    - Events are added incrementally during reasoning
    - Final attribution graph is constructed from the timeline
    - Full causal chain is preserved
    
    Usage:
        attribution = IncrementalAttribution()
        attribution.begin(state)
        attribution.add_event("candidate_generation", ...)
        attribution.add_event("policy_scoring", ...)
        attribution.add_event("selection", ...)
        attribution.finalize(selected_action)
        graph = attribution.get_graph()
    """
    
    def __init__(self, transaction_id: str):
        self.transaction_id = transaction_id
        self.events: List[Dict] = []
        self.state: Optional[Dict] = None
        self.candidates: List[Dict] = []
        self.scoring_events: List[Dict] = []
        self.selection_reason: str = ""
        self.finalized: bool = False
        
        self.feature_weights: Dict[str, float] = {
            "confidence": 1.0,
            "stress_level": 0.9,
            "action_readiness": 0.85,
            "arousal": 0.7,
            "valence": 0.7,
            "focus": 0.6,
            "bias_awareness": 0.5,
            "reflection_depth": 0.5,
            "exploration_tendency": 0.6,
            "task_complexity": 0.7,
            "task_urgency": 0.6,
            "task_novelty": 0.5,
        }
        
        logger.info("incremental_attribution_started", transaction_id=transaction_id)
    
    def begin(self, state: Dict) -> None:
        """Initialize with state"""
        self.state = state
        self.events.append({
            "type": "state_collected",
            "timestamp": datetime.utcnow().isoformat(),
            "data": state
        })
    
    def add_event(
        self,
        event_type: str,
        description: str,
        data: Optional[Dict] = None
    ) -> None:
        """Add a reasoning event"""
        self.events.append({
            "type": event_type,
            "description": description,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data or {}
        })
    
    def add_candidate(
        self,
        candidate_id: str,
        action_type: str,
        parameters: Dict,
        score: float,
        scoring_breakdown: Optional[Dict] = None
    ) -> None:
        """Add a candidate evaluation"""
        candidate = {
            "candidate_id": candidate_id,
            "action_type": action_type,
            "parameters": parameters,
            "score": score,
            "scoring_breakdown": scoring_breakdown or {}
        }
        self.candidates.append(candidate)
        
        self.scoring_events.append({
            "type": "candidate_scored",
            "candidate_id": candidate_id,
            "action_type": action_type,
            "score": score,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_scoring_step(
        self,
        scoring_type: str,
        details: Dict
    ) -> None:
        """Add a scoring step for debugging"""
        self.scoring_events.append({
            "type": scoring_type,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def finalize(
        self,
        selected_action: str,
        confidence: float,
        alternatives: List[Dict]
    ) -> AttributionSnapshot:
        """Finalize attribution from timeline"""
        self.finalized = True
        
        primary_features = self._compute_primary_features()
        causal_paths = self._compute_causal_paths()
        alternative_explanations = self._compute_alternatives(selected_action, alternatives)
        edges = self._build_edges()
        explanation = self._generate_explanation(selected_action, confidence)
        
        return AttributionSnapshot(
            primary_features=primary_features,
            causal_paths=causal_paths,
            alternative_explanations=alternative_explanations,
            edges=edges,
            final_explanation=explanation,
            confidence_score=confidence
        )
    
    def _compute_primary_features(self) -> List[Dict]:
        """Compute features with highest contribution"""
        if not self.state:
            return []
        
        contributions = []
        for feature_name, feature_value in self.state.items():
            if feature_name in self.feature_weights and isinstance(feature_value, (int, float)):
                weight = self.feature_weights[feature_name]
                contribution = feature_value * weight
                contributions.append({
                    "feature": feature_name,
                    "value": feature_value,
                    "weight": weight,
                    "contribution": contribution
                })
        
        contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return contributions[:5]
    
    def _compute_causal_paths(self) -> List[Dict]:
        """Compute causal paths from events"""
        paths = []
        
        state_event = self.events[0] if self.events else None
        selection_event = [e for e in self.events if e["type"] == "action_selected"]
        
        if state_event and selection_event:
            paths.append({
                "path": ["state_collected", "candidate_generation", "scoring", "selection"],
                "features": [f["feature"] for f in self._compute_primary_features()[:3]],
                "total_contribution": 1.0
            })
        
        return paths
    
    def _compute_alternatives(
        self,
        selected: str,
        alternatives: List[Dict]
    ) -> List[Dict]:
        """Explain why alternatives were not selected"""
        results = []
        
        for alt in alternatives[:3]:
            results.append({
                "action": alt.get("action", "unknown"),
                "score": alt.get("score", 0),
                "lost_by": 0.0,
                "key_difference": f"Selected '{selected}' over '{alt.get('action')}'"
            })
        
        return results
    
    def _build_edges(self) -> List[AttributionEdge]:
        """Build causal edge list"""
        edges = []
        
        if self.state:
            for feature in self._compute_primary_features():
                edge = AttributionEdge(
                    from_node=f"feature_{feature['feature']}",
                    to_node="decision_final",
                    contribution=feature["contribution"],
                    reason=f"Feature {feature['feature']} = {feature['value']:.2f}",
                    polarity="positive" if feature["contribution"] > 0 else "negative"
                )
                edges.append(edge)
        
        for candidate in self.candidates:
            edge = AttributionEdge(
                from_node=f"candidate_{candidate['candidate_id']}",
                to_node="decision_final",
                contribution=candidate["score"],
                reason=f"Candidate {candidate['action_type']} scored {candidate['score']:.2f}",
                polarity="positive"
            )
            edges.append(edge)
        
        return edges
    
    def _generate_explanation(self, action: str, confidence: float) -> str:
        """Generate human-readable explanation"""
        features = self._compute_primary_features()
        
        if not features:
            return f"Selected '{action}' with confidence {confidence:.2f}"
        
        top_feature = features[0]
        parts = [
            f"Primary driver: {top_feature['feature']} (value: {top_feature['value']:.2f})",
            f"Selected '{action}' with confidence {confidence:.2f}",
        ]
        
        if len(features) > 1:
            other_features = ", ".join(
                f"{f['feature']}={f['value']:.2f}" for f in features[1:3]
            )
            parts.append(f"Other factors: {other_features}")
        
        return ". ".join(parts)
    
    def get_timeline(self) -> List[Dict]:
        """Get full causal timeline"""
        return self.events + self.scoring_events