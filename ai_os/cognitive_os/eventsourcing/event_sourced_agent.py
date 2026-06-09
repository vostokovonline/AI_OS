"""
Event-Sourced Agent - Интеграция EventOntology с UnifiedAgent

Все решения теперь проходят через единый EventOntology:

UnifiedAgent
    ↓
EventOntology.emit() → EventStream
    ↓
DecisionTransaction (Semantic wrapper)
    ↓
StateSnapshot + ReasoningEvents + Attribution + Outcome
    ↓
Full causal graph в одном месте

Это заменяет разрозненные системы:
- DecisionTracer
- DecisionTransaction
- IncrementalAttribution

Теперь всё - события в EventStream.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Контекст для принятия решения"""
    user_id: str
    task: str
    task_type: str = "default"
    complexity: float = 0.5
    urgency: float = 0.5
    novelty: float = 0.5


@dataclass
class AgentDecision:
    """Результат решения агента"""
    action_type: str
    confidence: float
    reasoning: str
    expected_utility: float
    expected_risk: float
    alternatives: List[Dict]
    simulation_used: bool
    transaction_id: Optional[str] = None
    causal_chain: List[str] = field(default_factory=list)  # IDs событий


@dataclass
class TransactionResult:
    """Результат транзакции - полный causal graph"""
    transaction_id: str
    timestamp: datetime
    
    state_snapshot: Dict
    causal_chain: List[Dict]  # Все события в порядке
    
    selected_action: str
    confidence: float
    reasoning: str
    
    attribution: Optional[Dict] = None
    
    outcome: Optional[Dict] = None
    
    status: str  # committed, rolled_back, failed
    duration_ms: float = 0.0
    
    def is_valid(self) -> bool:
        return bool(self.selected_action and len(self.causal_chain) > 0)
    
    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "timestamp": self.timestamp.isoformat(),
            "state_snapshot": self.state_snapshot,
            "causal_chain": self.causal_chain,
            "selected_action": self.selected_action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "attribution": self.attribution,
            "outcome": self.outcome,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }


class EventSourcedAgent:
    """
    Event-Sourced Agent - агент с полным event sourcing.
    
    Все решения теперь записываются как последовательность событий
    с явными causal links.
    
    Usage:
        agent = EventSourcedAgent(cognitive_os)
        
        # Решение с полной traced causality
        decision = await agent.decide(context)
        
        # Транзакция с полным causal graph
        async with agent.transaction(user_id, task) as txn:
            await txn.decide(context)
            await txn.record_outcome("success")
        
        # Результат содержит полный causal chain
        result = txn.get_result()
        
        # Запрос событий
        events = agent.ontology.query(event_type="candidate_selected")
    """
    
    def __init__(self, cognitive_os):
        self.cognitive_os = cognitive_os
        
        from .state_builder import StateBuilder
        from .policy_layer import PolicyLayer
        from .simulation.planner import SimulationPlanner
        from ..observability.transaction import DecisionTransaction, TransactionStatus
        from .event_ontology import EventOntology, get_event_ontology, EventType, EventCategory
        
        self.state_builder = StateBuilder(cognitive_os)
        self.world_model = cognitive_os.world_model
        self.policy = PolicyLayer(self.state_builder, self.world_model)
        self.planner = SimulationPlanner(self.world_model)
        
        self.ontology = get_event_ontology()
        
        self.decision_history: List[AgentDecision] = []
        self.transactions: List[TransactionResult] = []
        self._current_transaction: Optional[str] = None
        
        logger.info("event_sourced_agent_initialized")
    
    async def decide(
        self,
        context: AgentContext,
        use_simulation: bool = True
    ) -> AgentDecision:
        """
        Принять решение (простой режим, без транзакции).
        
        Для полной traced causality используйте transaction().
        """
        transaction_id = str(uuid4())
        
        context_dict = {
            "task": context.task,
            "task_type": context.task_type,
            "complexity": context.complexity,
            "urgency": context.urgency,
            "novelty": context.novelty
        }
        
        state = await self.state_builder.build_state(context_dict)
        
        state_event_id = self.ontology.emit(
            transaction_id=transaction_id,
            event_type=EventType.STATE_SNAPSHOT.value,
            category=EventCategory.STATE.value,
            data={
                "confidence": state.confidence,
                "stress_level": state.stress_level,
                "action_readiness": state.action_readiness,
                "arousal": state.arousal,
                "valence": state.valence,
            }
        ).id
        
        policy_action = await self.policy.decide(
            context=context_dict,
            use_planning=use_simulation
        )
        
        selection_event_id = self.ontology.emit(
            transaction_id=transaction_id,
            event_type=EventType.CANDIDATE_SELECTED.value,
            category=EventCategory.CANDIDATE.value,
            data={
                "action": policy_action.action_type.value,
                "confidence": policy_action.confidence,
                "reasoning": policy_action.reasoning,
                "alternatives": [
                    {"action": a[0].value, "score": a[1]}
                    for a in policy_action.alternatives
                ]
            },
            causal_parent=state_event_id,
            decision_context={
                "user_id": context.user_id,
                "task": context.task,
                "task_type": context.task_type,
                "complexity": context.complexity,
                "urgency": context.urgency,
                "novelty": context.novelty,
            }
        ).id
        
        decision = AgentDecision(
            action_type=policy_action.action_type.value,
            confidence=policy_action.confidence,
            reasoning=policy_action.reasoning,
            expected_utility=policy_action.confidence,
            expected_risk=1 - policy_action.confidence,
            alternatives=[
                {"action": a[0].value, "score": a[1]}
                for a in policy_action.alternatives
            ],
            simulation_used=use_simulation,
            transaction_id=transaction_id,
            causal_chain=[state_event_id, selection_event_id]
        )
        
        self.decision_history.append(decision)
        
        self.world_model.record_action(
            actor_id=context.user_id,
            action=f"decide_{decision.action_type}",
            outcome="decided"
        )
        
        logger.info(
            "agent_decided",
            action=decision.action_type,
            confidence=decision.confidence,
            transaction_id=transaction_id,
            causal_chain_len=len(decision.causal_chain)
        )
        
        return decision
    
    def transaction(self, user_id: str, task: str) -> "EventSourcedTransaction":
        """Создать транзакцию для полной traced causality"""
        return EventSourcedTransaction(self, user_id, task)
    
    def get_statistics(self) -> Dict:
        """Получить статистику"""
        stats = self.ontology.get_statistics()
        stats["decisions_made"] = len(self.decision_history)
        stats["transactions_completed"] = len(self.transactions)
        return stats
    
    def query_events(
        self,
        event_type: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Запрос событий"""
        events = self.ontology.query(
            event_type=event_type,
            category=category,
            limit=limit
        )
        return [e.to_dict() for e in events]
    
    def get_replayable_dataset(self, limit: int = 100) -> List[Dict]:
        """Получить dataset для RL обучения"""
        dataset = []
        
        for txn in self.transactions[-limit:]:
            if txn.outcome and txn.is_valid():
                dataset.append({
                    "transaction_id": txn.transaction_id,
                    "state": txn.state_snapshot,
                    "action": txn.selected_action,
                    "confidence": txn.confidence,
                    "outcome": txn.outcome.get("outcome"),
                    "outcome_score": txn.outcome.get("score"),
                })
        
        return dataset


class EventSourcedTransaction:
    """
    Транзакция с event sourcing.
    
    Использует EventOntology для записи всех событий.
    
    Usage:
        async with agent.transaction(user_id, task) as txn:
            await txn.decide(context)
            await txn.record_outcome("success")
        
        result = txn.get_result()
    """
    
    def __init__(self, agent: EventSourcedAgent, user_id: str, task: str):
        self.agent = agent
        self.transaction_id = str(uuid4())
        self.user_id = user_id
        self.task = task
        self.start_time = datetime.utcnow()
        
        self.state_event_id: Optional[str] = None
        self.selection_event_id: Optional[str] = None
        
        self.result: Optional[TransactionResult] = None
        
        from .event_ontology import EventType, EventCategory
        
        self.EventType = EventType
        self.EventCategory = EventCategory
        
        logger.info("transaction_started", transaction_id=self.transaction_id)
    
    async def __aenter__(self) -> "EventSourcedTransaction":
        self.agent._current_transaction = self.transaction_id
        self.agent.ontology.begin_stream(self.transaction_id)
        
        self.agent.ontology.emit(
            transaction_id=self.transaction_id,
            event_type=EventType.TRANSACTION_START.value,
            category=EventCategory.LIFECYCLE.value,
            data={
                "user_id": self.user_id,
                "task": self.task,
            }
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.agent.ontology.emit(
                transaction_id=self.transaction_id,
                event_type=EventType.TRANSACTION_ROLLBACK.value,
                category=EventCategory.LIFECYCLE.value,
                data={"error": str(exc_val)}
            )
            self.agent.ontology.rollback_stream(self.transaction_id)
            status = "failed"
        else:
            self.agent.ontology.emit(
                transaction_id=self.transaction_id,
                event_type=EventType.TRANSACTION_COMMIT.value,
                category=EventCategory.LIFECYCLE.value
            )
            self.agent.ontology.commit_stream(self.transaction_id)
            status = "committed"
        
        stream = self.agent.ontology.get_stream(self.transaction_id)
        events = stream.events if stream else []
        
        causal_chain = [e.to_dict() for e in events]
        
        self.result = TransactionResult(
            transaction_id=self.transaction_id,
            timestamp=self.start_time,
            state_snapshot={},
            causal_chain=causal_chain,
            selected_action="",
            confidence=0.0,
            reasoning="",
            outcome=None,
            status=status,
            duration_ms=(datetime.utcnow() - self.start_time).total_seconds() * 1000
        )
        
        self.agent.transactions.append(self.result)
        self.agent._current_transaction = None
        
        logger.info(
            "transaction_completed",
            transaction_id=self.transaction_id,
            status=status,
            events=len(events)
        )
        
        return False
    
    async def decide(
        self,
        context: AgentContext,
        use_simulation: bool = True
    ) -> None:
        """Принять решение в рамках транзакции"""
        context_dict = {
            "task": context.task,
            "task_type": context.task_type,
            "complexity": context.complexity,
            "urgency": context.urgency,
            "novelty": context.novelty
        }
        
        state = await self.agent.state_builder.build_state(context_dict)
        
        state_event_id = self.agent.ontology.emit(
            transaction_id=self.transaction_id,
            event_type=self.EventType.STATE_SNAPSHOT.value,
            category=self.EventCategory.STATE.value,
            data={
                "confidence": state.confidence,
                "stress_level": state.stress_level,
                "action_readiness": state.action_readiness,
                "arousal": state.arousal,
                "valence": state.valence,
                "focus": state.focus,
                "bias_awareness": state.bias_awareness,
                "reflection_depth": state.reflection_depth,
                "exploration_tendency": state.exploration_tendency,
                "task_complexity": context.complexity,
                "task_urgency": context.urgency,
                "task_novelty": context.novelty,
            }
        ).id
        
        self.state_event_id = state_event_id
        
        if self.result:
            self.result.state_snapshot = {
                "confidence": state.confidence,
                "stress_level": state.stress_level,
                "action_readiness": state.action_readiness,
            }
        
        candidates = []
        for i, (action_type, score) in enumerate(policy_action.alternatives for _ in range(1)):
            pass
        
        for i, (action_type, score) in enumerate([(a[0], a[1]) for a in policy_action.alternatives]):
            self.agent.ontology.emit(
                transaction_id=self.transaction_id,
                event_type=self.EventType.CANDIDATE_GENERATED.value,
                category=self.EventCategory.CANDIDATE.value,
                data={
                    "candidate_id": f"candidate_{i}",
                    "action_type": action_type.value,
                    "score": score,
                },
                causal_parent=state_event_id
            )
        
        policy_action = await self.agent.policy.decide(
            context=context_dict,
            use_planning=use_simulation
        )
        
        selection_event_id = self.agent.ontology.emit(
            transaction_id=self.transaction_id,
            event_type=self.EventType.CANDIDATE_SELECTED.value,
            category=self.EventCategory.CANDIDATE.value,
            data={
                "action": policy_action.action_type.value,
                "confidence": policy_action.confidence,
                "reasoning": policy_action.reasoning,
                "simulation_used": use_simulation,
            },
            causal_parent=state_event_id,
            decision_context={
                "user_id": context.user_id,
                "task": context.task,
                "task_type": context.task_type,
            }
        ).id
        
        self.selection_event_id = selection_event_id
        
        if self.result:
            self.result.selected_action = policy_action.action_type.value
            self.result.confidence = policy_action.confidence
            self.result.reasoning = policy_action.reasoning
        
        self.agent.world_model.record_action(
            actor_id=context.user_id,
            action=f"decide_{policy_action.action_type.value}",
            outcome="decided"
        )
        
        logger.info(
            "transaction_decision_made",
            transaction_id=self.transaction_id,
            action=policy_action.action_type.value
        )
    
    async def record_outcome(self, outcome: str, score: float = 1.0) -> None:
        """Записать результат"""
        self.agent.ontology.emit(
            transaction_id=self.transaction_id,
            event_type=self.EventType.OUTCOME_RECORDED.value,
            category=self.EventCategory.OUTCOME.value,
            data={
                "outcome": outcome,
                "score": score,
            },
            causal_parent=self.selection_event_id
        )
        
        if self.result:
            self.result.outcome = {
                "outcome": outcome,
                "score": score
            }
        
        self.agent.ontology.emit(
            transaction_id=self.transaction_id,
            event_type=self.EventType.LEARNING_INTEGRATED.value,
            category=self.EventCategory.OUTCOME.value,
            causal_parent=self.selection_event_id
        )
        
        logger.info(
            "transaction_outcome_recorded",
            transaction_id=self.transaction_id,
            outcome=outcome
        )
    
    def get_result(self) -> Optional[TransactionResult]:
        """Получить результат транзакции"""
        return self.result
    
    def get_causal_chain(self) -> List[Dict]:
        """Получить causal chain событий"""
        stream = self.agent.ontology.get_stream(self.transaction_id)
        if stream:
            return [e.to_dict() for e in stream.events]
        return []


# Создание глобального экземпляра
_event_sourced_agent: Optional[EventSourcedAgent] = None


def get_event_sourced_agent(cognitive_os) -> EventSourcedAgent:
    """Получить глобальный Event-Sourced Agent"""
    global _event_sourced_agent
    if _event_sourced_agent is None:
        _event_sourced_agent = EventSourcedAgent(cognitive_os)
    return _event_sourced_agent