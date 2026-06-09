"""
AI-OS Cognitive OS - Event Sourcing Layer (Phase 10)

Event Ontology - единый источник истины для всех событий

Components:
- event_ontology.py - Event + EventStream + EventOntology
- event_sourced_agent.py - Event-Sourced Agent

Usage:
    from ai_os.cognitive_os.eventsourcing import EventOntology, EventSourcedAgent
    
    # Испустить событие
    ontology = get_event_ontology()
    event = ontology.emit(
        transaction_id="tx123",
        event_type="candidate_selected",
        data={"action": "execute", "confidence": 0.8}
    )
    
    # Запрос событий
    events = ontology.query(event_type="state_snapshot")
    
    # Event-Sourced Agent
    agent = EventSourcedAgent(cognitive_os)
    decision = await agent.decide(context)
    
    # Транзакция с полным causal graph
    async with agent.transaction(user_id, task) as txn:
        await txn.decide(context)
        await txn.record_outcome("success")
    
    result = txn.get_result()
"""
from .event_ontology import (
    Event,
    EventStream,
    EventOntology,
    EventType,
    EventCategory,
    CausalLink,
    StateDelta,
    DecisionContext,
    get_event_ontology,
    emit_state_snapshot,
    emit_candidate,
    emit_selection,
    emit_outcome,
)
from .event_sourced_agent import (
    EventSourcedAgent,
    EventSourcedTransaction,
    AgentContext,
    AgentDecision,
    TransactionResult,
    get_event_sourced_agent,
)

__all__ = [
    # Event Ontology
    "Event",
    "EventStream",
    "EventOntology",
    "EventType",
    "EventCategory",
    "CausalLink",
    "StateDelta",
    "DecisionContext",
    "get_event_ontology",
    "emit_state_snapshot",
    "emit_candidate",
    "emit_selection",
    "emit_outcome",
    
    # Event-Sourced Agent
    "EventSourcedAgent",
    "EventSourcedTransaction",
    "AgentContext",
    "AgentDecision",
    "TransactionResult",
    "get_event_sourced_agent",
]