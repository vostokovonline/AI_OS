"""
AI-OS Cognitive OS - Observability Layer

Phase 9: Make it observable and debuggable

Components:
- transaction.py - DecisionTransaction as first-class atomic construct
- decision_trace.py - Decision trace system (full behavioral telemetry)
- state_diff.py - State diff engine (track state changes over time)
- attribution.py - Policy attribution graph (why features led to actions)

Usage:
    from ai_os.cognitive_os.observability import TransactionContext, DecisionTransaction
    
    async with TransactionContext(agent, user_id, task) as ctx:
        # Decision is automatically traced with full causal chain
        await ctx.decide(context)
        
        # Outcome automatically recorded
        await ctx.record_outcome("success", 1.0)
    
    # Transaction now complete with full causal graph
    txn = ctx.get_transaction()
"""
from .transaction import (
    DecisionTransaction,
    TransactionContext,
    TransactionStatus,
    EventType,
    ReasoningEvent,
    StateSnapshot,
    CandidateEvaluation,
    AttributionSnapshot,
    OutcomeRecord,
    IncrementalAttribution,
)
from .decision_trace import (
    DecisionTracer,
    DecisionTrace,
    DecisionRecord,
)
from .state_diff import (
    StateDiffEngine,
    StateHistory,
    StateDelta,
    DiffAnalyzer,
)
from .attribution import (
    PolicyAttributionSystem,
    AttributionAnalyzer,
)

__all__ = [
    # Transaction (Core)
    "DecisionTransaction",
    "TransactionContext",
    "TransactionStatus",
    "EventType",
    "IncrementalAttribution",
    
    # Decision Tracing
    "DecisionTracer",
    "DecisionTrace",
    "DecisionRecord",
    
    # State Diff
    "StateDiffEngine",
    "StateHistory",
    "DiffAnalyzer",
    
    # Attribution
    "PolicyAttributionSystem",
    "AttributionAnalyzer",
]