"""
Decision Trace System - Complete behavioral telemetry

Records every decision with full context for debugging and analysis.

Components:
- DecisionRecord: Individual decision trace
- DecisionTrace: Complete trace with reasoning chain
- TraceStore: Persistent storage and querying
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    POLICY = "policy"
    SIMULATION = "simulation"
    EMOTIONAL = "emotional"
    STRATEGY = "strategy"
    EXTERNAL = "external"


@dataclass
class FeatureContribution:
    """How much a feature contributed to the decision"""
    feature_name: str
    feature_value: float
    contribution_score: float
    contribution_direction: str  # positive, negative, neutral


@dataclass
class StateSnapshot:
    """State at decision time"""
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
    stress_level: float = 0.0
    exploration_tendency: float = 0.5
    action_readiness: float = 0.5
    task_complexity: float = 0.5
    task_urgency: float = 0.5
    task_novelty: float = 0.5

    def to_dict(self) -> Dict:
        return {
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
            "stress_level": self.stress_level,
            "exploration_tendency": self.exploration_tendency,
            "action_readiness": self.action_readiness,
            "task_complexity": self.task_complexity,
            "task_urgency": self.task_urgency,
            "task_novelty": self.task_novelty,
        }


@dataclass
class DecisionRecord:
    """Single decision with full context"""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    decision_type: str = DecisionType.POLICY.value
    user_id: str = ""
    task: str = ""
    
    action: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    
    state_before: Optional[StateSnapshot] = None
    state_after: Optional[StateSnapshot] = None
    
    features: List[FeatureContribution] = field(default_factory=list)
    alternatives: List[Dict] = field(default_factory=list)
    
    simulation_used: bool = False
    simulation_depth: int = 0
    
    outcome: Optional[str] = None
    outcome_score: float = 0.0
    outcome_latency_ms: float = 0.0
    
    trace_id: str = ""


@dataclass
class ReasoningStep:
    """One step in the reasoning chain"""
    step: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    operation: str  # state_collection, candidate_generation, scoring, selection
    details: Dict[str, Any] = field(default_factory=dict)


class DecisionTrace:
    """Complete trace of a decision session"""
    
    def __init__(self, trace_id: str, user_id: str):
        self.trace_id = trace_id
        self.user_id = user_id
        self.decisions: List[DecisionRecord] = []
        self.reasoning_chain: List[ReasoningStep] = []
        self.start_time: datetime = field(default_factory=datetime.utcnow)
        self.end_time: Optional[datetime] = None
        self.tags: List[str] = []
    
    def add_decision(self, record: DecisionRecord) -> None:
        record.trace_id = self.trace_id
        self.decisions.append(record)
    
    def add_reasoning_step(self, step: ReasoningStep) -> None:
        self.reasoning_chain.append(step)
    
    def complete(self) -> None:
        self.end_time = datetime.utcnow()
    
    def duration_ms(self) -> float:
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict:
        return {
            "trace_id": self.trace_id,
            "user_id": self.user_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms(),
            "decisions_count": len(self.decisions),
            "reasoning_steps_count": len(self.reasoning_chain),
            "tags": self.tags,
            "decisions": [d.__dict__ for d in self.decisions],
            "reasoning_chain": [
                {"step": r.step, "timestamp": r.timestamp.isoformat(), "operation": r.operation, "details": r.details}
                for r in self.reasoning_chain
            ]
        }


class TraceStore:
    """
    Persistent storage and querying for decision traces.
    
    In production, this would use PostgreSQL or a time-series DB.
    For now, in-memory with file persistence.
    """
    
    def __init__(self, max_traces: int = 10000):
        self.traces: Dict[str, DecisionTrace] = {}
        self.records: Dict[str, DecisionRecord] = {}
        self.max_traces = max_traces
        logger.info("trace_store_initialized", max_traces=max_traces)
    
    def store_trace(self, trace: DecisionTrace) -> None:
        self.traces[trace.trace_id] = trace
        for decision in trace.decisions:
            self.records[decision.id] = decision
        
        if len(self.traces) > self.max_traces:
            oldest = min(self.traces.keys(), key=lambda k: self.traces[k].start_time)
            self._remove_trace(oldest)
        
        logger.debug("trace_stored", trace_id=trace.trace_id, decisions=len(trace.decisions))
    
    def store_decision(self, record: DecisionRecord) -> None:
        self.records[record.id] = record
        logger.debug("decision_stored", decision_id=record.id)
    
    def get_trace(self, trace_id: str) -> Optional[DecisionTrace]:
        return self.traces.get(trace_id)
    
    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        return self.records.get(decision_id)
    
    def query(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        from_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[DecisionRecord]:
        results = list(self.records.values())
        
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        if action:
            results = [r for r in results if r.action == action]
        if outcome:
            results = [r for r in results if r.outcome == outcome]
        if from_time:
            results = [r for r in results if r.timestamp >= from_time]
        
        results.sort(key=lambda r: r.timestamp, reverse=True)
        return results[:limit]
    
    def get_action_distribution(self, user_id: Optional[str] = None) -> Dict[str, float]:
        records = self.query(user_id=user_id, limit=10000)
        if not records:
            return {}
        
        counts: Dict[str, int] = {}
        for r in records:
            counts[r.action] = counts.get(r.action, 0) + 1
        
        total = sum(counts.values())
        return {a: c / total for a, c in counts.items()}
    
    def get_confidence_stats(self, user_id: Optional[str] = None) -> Dict[str, float]:
        records = self.query(user_id=user_id, limit=1000)
        if not records:
            return {"avg": 0.5, "min": 0.5, "max": 0.5}
        
        confidences = [r.confidence for r in records]
        return {
            "avg": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences)
        }
    
    def get_outcome_distribution(self, user_id: Optional[str] = None) -> Dict[str, float]:
        records = self.query(user_id=user_id, limit=1000, outcome=None)
        if not records:
            return {}
        
        counts: Dict[str, int] = {}
        for r in records:
            if r.outcome:
                counts[r.outcome] = counts.get(r.outcome, 0) + 1
        
        total = sum(counts.values())
        return {o: c / total for o, c in counts.items()}
    
    def _remove_trace(self, trace_id: str) -> None:
        trace = self.traces.pop(trace_id, None)
        if trace:
            for decision in trace.decisions:
                self.records.pop(decision.id, None)
            logger.debug("trace_removed", trace_id=trace_id)


class DecisionTracer:
    """
    Main interface for decision tracing.
    
    Usage:
        tracer = DecisionTracer()
        
        with tracer.trace(user_id, task) as trace:
            # Make decisions
            trace.add_decision(decision_record)
        
        # Store for later analysis
        tracer.store_trace(trace)
    """
    
    def __init__(self):
        self.store = TraceStore()
        self.active_traces: Dict[str, DecisionTrace] = {}
        logger.info("decision_tracer_initialized")
    
    def start_trace(self, user_id: str, task: str = "") -> DecisionTrace:
        trace_id = str(uuid4())
        trace = DecisionTrace(trace_id, user_id)
        trace.tags.append(task)
        self.active_traces[trace_id] = trace
        logger.info("trace_started", trace_id=trace_id, user_id=user_id)
        return trace
    
    def add_reasoning_step(
        self,
        trace_id: str,
        operation: str,
        details: Dict[str, Any]
    ) -> None:
        if trace_id in self.active_traces:
            trace = self.active_traces[trace_id]
            step = ReasoningStep(
                step=len(trace.reasoning_chain) + 1,
                operation=operation,
                details=details
            )
            trace.add_reasoning_step(step)
    
    def end_trace(self, trace_id: str) -> Optional[DecisionTrace]:
        trace = self.active_traces.pop(trace_id, None)
        if trace:
            trace.complete()
            self.store.store_trace(trace)
            logger.info("trace_completed", trace_id=trace_id, duration_ms=trace.duration_ms())
        return trace
    
    def record_decision(
        self,
        user_id: str,
        task: str,
        action: str,
        confidence: float,
        reasoning: str,
        state_before: StateSnapshot,
        alternatives: List[Dict],
        simulation_used: bool = False,
        trace_id: Optional[str] = None
    ) -> DecisionRecord:
        record = DecisionRecord(
            decision_type=DecisionType.POLICY.value,
            user_id=user_id,
            task=task,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            state_before=state_before,
            alternatives=alternatives,
            simulation_used=simulation_used,
            trace_id=trace_id or ""
        )
        
        if trace_id and trace_id in self.active_traces:
            self.active_traces[trace_id].add_decision(record)
        
        self.store.store_decision(record)
        return record
    
    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        outcome_score: float,
        state_after: Optional[StateSnapshot] = None,
        latency_ms: float = 0.0
    ) -> None:
        record = self.store.get_decision(decision_id)
        if record:
            record.outcome = outcome
            record.outcome_score = outcome_score
            record.state_after = state_after
            record.outcome_latency_ms = latency_ms
            logger.info("outcome_recorded", decision_id=decision_id, outcome=outcome)
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "total_traces": len(self.store.traces),
            "total_decisions": len(self.store.records),
            "action_distribution": self.store.get_action_distribution(user_id),
            "confidence_stats": self.store.get_confidence_stats(user_id),
            "outcome_distribution": self.store.get_outcome_distribution(user_id),
        }
    
    def explain_decision(self, decision_id: str) -> Dict[str, Any]:
        record = self.store.get_decision(decision_id)
        if not record:
            return {"error": "Decision not found"}
        
        explanations = []
        
        if record.state_before:
            state = record.state_before.to_dict()
            for key, value in state.items():
                if key in ["stress_level", "confidence", "action_readiness"]:
                    if value > 0.7:
                        explanations.append(f"High {key} ({value:.2f}) - increases {key}/100 probability")
                    elif value < 0.3:
                        explanations.append(f"Low {key} ({value:.2f}) - decreases probability")
        
        return {
            "decision_id": decision_id,
            "action": record.action,
            "confidence": record.confidence,
            "reasoning": record.reasoning,
            "state_snapshot": record.state_before.to_dict() if record.state_before else {},
            "feature_explanations": explanations,
            "alternatives": record.alternatives,
            "outcome": record.outcome,
            "outcome_score": record.outcome_score,
        }