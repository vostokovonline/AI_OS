"""
Cognitive Commit Protocol - Active transactional semantics

Unlike passive snapshots, this is an ACTIVE commit protocol with:
- BEGIN/PREPARE/COMMIT/FINALIZE lifecycle
- Two-phase commit (prepare all, commit only if valid)
- Causal clock for ordering
- Side effect tracking
- Invariant validation
"""
import json
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


class TransactionPhase(str, Enum):
    """Transaction lifecycle phases"""
    INITIAL = "initial"
    BEGIN = "begin"
    PREPARING = "preparing"
    PREPARE_COMPLETE = "prepare_complete"
    VALIDATING = "validating"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class CausalClock:
    """
    Lamport-style causal clock for ordering in multi-agent runtime.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._vector: Dict[str, int] = {node_id: 0}
    
    def tick(self) -> int:
        """Increment local time"""
        self._vector[self.node_id] = self._vector.get(self.node_id, 0) + 1
        return self._vector[self.node_id]
    
    def update(self, other_vector: Dict[str, int]):
        """Update clock based on received vector"""
        for node, time in other_vector.items():
            self._vector[node] = max(self._vector.get(node, 0), time)
        self.tick()  # Increment for local event
    
    def merge(self, other: "CausalClock") -> Dict[str, int]:
        """Merge two clocks, return resulting vector"""
        result = dict(self._vector)
        for node, time in other._vector.items():
            result[node] = max(result.get(node, 0), time)
        return result
    
    def happened_before(self, other: "CausalClock") -> bool:
        """Check if other happened before this"""
        other_vector = other._vector
        
        # If other has events this node hasn't seen
        for node, time in other_vector.items():
            if node == self.node_id:
                continue
            if time > self._vector.get(node, 0):
                return False
        
        # And at least one is strictly less
        for node, time in other_vector.items():
            if time < self._vector.get(node, 0):
                return True
        
        return False
    
    def to_dict(self) -> dict:
        return self._vector.copy()
    
    @staticmethod
    def from_dict(data: dict) -> "CausalClock":
        clock = CausalClock("unknown")
        clock._vector = data
        return clock


@dataclass
class PendingEntry:
    """Pending entry for two-phase commit"""
    entry_type: str  # intent, boundary, event, metric, etc.
    entry_id: str
    data: dict
    phase: TransactionPhase  # When added
    
    def to_dict(self) -> dict:
        return {
            "entry_type": self.entry_type,
            "entry_id": self.entry_id,
            "data": self.data,
            "phase": self.phase.value
        }


@dataclass
class SideEffect:
    """
    Tracked side effect for deterministic replay.
    """
    effect_id: str
    transaction_id: str
    
    effect_type: str  # "filesystem.write", "network.call", etc.
    target: str  # What was affected
    
    reversible: bool
    rollback_strategy: Optional[str]
    
    execution_order: int
    committed: bool = False
    
    def to_dict(self) -> dict:
        return {
            "effect_id": self.effect_id,
            "transaction_id": self.transaction_id,
            "effect_type": self.effect_type,
            "target": self.target,
            "reversible": self.reversible,
            "rollback_strategy": self.rollback_strategy,
            "execution_order": self.execution_order,
            "committed": self.committed
        }


class InvariantEngine:
    """
    Unified invariant validation engine.
    
    All planes use same semantic truth layer.
    """
    
    def __init__(self):
        self._invariants: Dict[str, Callable] = {}
    
    def register(self, name: str, predicate: Callable[["CognitiveCommitProtocol"], bool]):
        """Register invariant"""
        self._invariants[name] = predicate
    
    def validate(self, protocol: "CognitiveCommitProtocol") -> Dict[str, Any]:
        """Validate all invariants"""
        results = {}
        all_valid = True
        
        for name, predicate in self._invariants.items():
            try:
                valid = predicate(protocol)
                results[name] = {"valid": valid, "error": None}
                if not valid:
                    all_valid = False
            except Exception as e:
                results[name] = {"valid": False, "error": str(e)}
                all_valid = False
        
        return {
            "all_valid": all_valid,
            "results": results,
            "invalid_count": sum(1 for r in results.values() if not r["valid"])
        }


class CognitiveCommitProtocol:
    """
    Active cognitive commit protocol with two-phase commit.
    
    Unlike passive snapshots, this manages causal execution semantics:
    - BEGIN/PREPARE/COMMIT/FINALIZE lifecycle
    - Two-phase commit (prepare all, commit only if valid)
    - Side effect tracking
    - Causal clock for ordering
    - Invariant validation
    """
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.transaction_id = uuid4().hex[:8]
        
        # Lifecycle
        self.phase = TransactionPhase.INITIAL
        self.clock = CausalClock(self.transaction_id)
        
        # Two-phase commit
        self._pending: List[PendingEntry] = []
        self._committed: List[PendingEntry] = []
        self._rolled_back: List[PendingEntry] = []
        
        # Side effects
        self._side_effects: List[SideEffect] = []
        
        # Components (will be populated)
        self._intent = None
        self._boundary = None
        self._state_machine = None
        self._event_ids: List[str] = []
        
        # Invariant engine
        self._invariant_engine = InvariantEngine()
        self._setup_default_invariants()
        
        self._created_at = datetime.utcnow().isoformat()
    
    def _setup_default_invariants(self):
        """Setup default invariants"""
        
        # Invariant: State machine invariants valid (no violations)
        self._invariant_engine.register(
            "state_machine_valid",
            lambda p: p._state_machine is None or 
                     len(p._state_machine.validate_invariants()) == 0
        )
        
        # Invariant: Must have intent
        self._invariant_engine.register(
            "has_intent",
            lambda p: p._intent is not None
        )
        
        # Invariant: Must have decision if executing
        self._invariant_engine.register(
            "has_decision_if_executing",
            lambda p: p._state_machine is None or 
                     p._state_machine.current_state.value != "executing" or
                     p._boundary is not None
        )
        
        # Invariant: Pending entries must not be empty (something prepared)
        self._invariant_engine.register(
            "has_pending_entries",
            lambda p: len(p._pending) > 0
        )
    
    def begin(self) -> "CognitiveCommitProtocol":
        """Begin transaction"""
        self.phase = TransactionPhase.BEGIN
        self.clock.tick()
        return self
    
    def prepare_intent(
        self,
        parent_goal: str,
        desired_outcome: str,
        risk_budget: float = 0.5,
        autonomy_level: str = "autonomous"
    ) -> "CognitiveCommitProtocol":
        """Prepare intent (phase 1)"""
        from experience.decision_boundary import get_intent_store
        
        self.phase = TransactionPhase.PREPARING
        
        store = get_intent_store()
        intent = store.create(
            execution_id=self.execution_id,
            parent_goal=parent_goal,
            desired_outcome=desired_outcome,
            success_criteria={},
            risk_budget=risk_budget,
            autonomy_level=autonomy_level
        )
        
        self._intent = intent
        self._pending.append(PendingEntry(
            entry_type="intent",
            entry_id=intent.intent_id,
            data=intent.to_dict(),
            phase=self.phase
        ))
        
        self.clock.tick()
        return self
    
    def prepare_decision(
        self,
        selected_skill: str,
        candidate_distribution: Dict[str, float],
        posterior: Dict[str, float],
        uncertainty: Dict[str, float],
        confidence: float = 0.5
    ) -> "CognitiveCommitProtocol":
        """Prepare decision boundary (phase 1)"""
        from experience.decision_boundary import get_boundary_store
        
        store = get_boundary_store()
        boundary = store.record(
            execution_id=self.execution_id,
            selected_candidate=selected_skill,
            candidate_distribution=candidate_distribution,
            posterior=posterior,
            uncertainty=uncertainty,
            constraints=[],
            suppressed=[],
            temperature=1.0,
            exploration=0.1,
            confidence=confidence
        )
        
        self._boundary = boundary
        self._pending.append(PendingEntry(
            entry_type="boundary",
            entry_id=boundary.execution_id,
            data=boundary.to_dict(),
            phase=self.phase
        ))
        
        self.clock.tick()
        return self
    
    def prepare_state(self, initial_state: str = "created") -> "CognitiveCommitProtocol":
        """Prepare state machine (phase 1)"""
        from experience.runtime_state import get_state_machine, ExecutionState
        
        sm = get_state_machine(self.execution_id)
        try:
            sm.transition(ExecutionState(initial_state), "transaction_began")
        except:
            pass
        
        self._state_machine = sm
        self._pending.append(PendingEntry(
            entry_type="state_machine",
            entry_id=self.execution_id,
            data=sm.to_dict(),
            phase=self.phase
        ))
        
        self.clock.tick()
        return self
    
    def prepare_event(
        self,
        event_type: str,
        data: Dict
    ) -> "CognitiveCommitProtocol":
        """Prepare journal event (phase 1)"""
        from experience.execution_journal import get_execution_journal, EventType
        
        journal = get_execution_journal()
        
        # Convert string to EventType
        try:
            evt = EventType(event_type)
        except ValueError:
            evt = EventType.TASK_CREATED
        
        event = journal.record(evt, self.execution_id, data)
        self._event_ids.append(event.event_id)
        
        self._pending.append(PendingEntry(
            entry_type="event",
            entry_id=event.event_id,
            data=event.to_dict(),
            phase=self.phase
        ))
        
        self.clock.tick()
        return self
    
    def add_side_effect(
        self,
        effect_type: str,
        target: str,
        reversible: bool = True,
        rollback_strategy: Optional[str] = None
    ) -> "CognitiveCommitProtocol":
        """Track side effect"""
        effect = SideEffect(
            effect_id=uuid4().hex[:8],
            transaction_id=self.transaction_id,
            effect_type=effect_type,
            target=target,
            reversible=reversible,
            rollback_strategy=rollback_strategy,
            execution_order=len(self._side_effects),
            committed=False
        )
        
        self._side_effects.append(effect)
        return self
    
    def validate(self) -> Dict:
        """Validate all invariants (before commit)"""
        self.phase = TransactionPhase.VALIDATING
        return self._invariant_engine.validate(self)
    
    def commit(self) -> bool:
        """
        Commit transaction (phase 2).
        
        Only commits if:
        - All invariants valid
        - No pending entries are empty
        - State machine valid
        """
        self.phase = TransactionPhase.COMMITTING
        
        # Validate invariants
        validation = self.validate()
        if not validation["all_valid"]:
            self.phase = TransactionPhase.FAILED
            return False
        
        # Move pending to committed
        for entry in self._pending:
            entry.phase = TransactionPhase.COMMITTED
            self._committed.append(entry)
        
        # Mark side effects as committed
        for effect in self._side_effects:
            effect.committed = True
        
        self.phase = TransactionPhase.COMMITTED
        self.clock.tick()
        
        return True
    
    def rollback(self):
        """Rollback transaction"""
        self.phase = TransactionPhase.ROLLING_BACK
        
        # Move pending to rolled back
        for entry in self._pending:
            entry.phase = TransactionPhase.ROLLED_BACK
            self._rolled_back.append(entry)
        
        # Clear committed (they're not really committed anyway)
        self._committed.clear()
        
        self.phase = TransactionPhase.ROLLED_BACK
    
    def finalize(self) -> dict:
        """Finalize and return transaction summary"""
        self.phase = TransactionPhase.COMMITTED
        
        # Compute transaction hash
        tx_hash = hashlib.sha256(
            json.dumps({
                "execution_id": self.execution_id,
                "committed_entries": len(self._committed),
                "side_effects": len(self._side_effects),
                "clock": self.clock.to_dict()
            }, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return {
            "transaction_id": self.transaction_id,
            "execution_id": self.execution_id,
            "phase": self.phase.value,
            "clock": self.clock.to_dict(),
            "committed_entries": len(self._committed),
            "rolled_back_entries": len(self._rolled_back),
            "side_effects": len(self._side_effects),
            "has_intent": self._intent is not None,
            "has_boundary": self._boundary is not None,
            "has_state": self._state_machine is not None,
            "event_count": len(self._event_ids),
            "transaction_hash": tx_hash
        }


# Global commit protocols
_active_protocols: Dict[str, CognitiveCommitProtocol] = {}


def begin_cognitive_transaction(execution_id: str) -> CognitiveCommitProtocol:
    """Start a new cognitive commit protocol"""
    protocol = CognitiveCommitProtocol(execution_id)
    _active_protocols[execution_id] = protocol
    return protocol.begin()


def get_active_protocol(execution_id: str) -> Optional[CognitiveCommitProtocol]:
    """Get active protocol for execution"""
    return _active_protocols.get(execution_id)