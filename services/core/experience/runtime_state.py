"""
Runtime State Machine - Formal execution lifecycle

CRITICAL: Without formal lifecycle, transitions are implicit and undebuggable.

States:
CREATED → PLANNED → SCORING → SELECTED → EXECUTING → OBSERVING → EVALUATING → COMPLETED
                    ↓          ↓         ↓          ↓           ↓          ↓
                 BLOCKED   REJECTED  TIMEOUT   FAILED     ROLLED_BACK  CANCELLED
"""
from enum import Enum
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


class ExecutionState(str, Enum):
    """Formal execution states"""
    # Main path
    CREATED = "created"
    PLANNED = "planned"
    SCORING = "scoring"
    SELECTED = "selected"
    EXECUTING = "executing"
    OBSERVING = "observing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    
    # Failure states
    BLOCKED = "blocked"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


# Valid state transitions
VALID_TRANSITIONS: Dict[ExecutionState, Set[ExecutionState]] = {
    # Main path
    ExecutionState.CREATED: {
        ExecutionState.PLANNED,
        ExecutionState.BLOCKED,
        ExecutionState.CANCELLED
    },
    ExecutionState.PLANNED: {
        ExecutionState.SCORING,
        ExecutionState.REJECTED,
        ExecutionState.CANCELLED
    },
    ExecutionState.SCORING: {
        ExecutionState.SELECTED,
        ExecutionState.REJECTED,
        ExecutionState.TIMEOUT
    },
    ExecutionState.SELECTED: {
        ExecutionState.EXECUTING,
        ExecutionState.REJECTED,
        ExecutionState.CANCELLED
    },
    ExecutionState.EXECUTING: {
        ExecutionState.OBSERVING,
        ExecutionState.TIMEOUT,
        ExecutionState.FAILED
    },
    ExecutionState.OBSERVING: {
        ExecutionState.EVALUATING,
        ExecutionState.FAILED
    },
    ExecutionState.EVALUATING: {
        ExecutionState.COMPLETED,
        ExecutionState.FAILED,
        ExecutionState.ROLLED_BACK
    },
    
    # Terminal states
    ExecutionState.COMPLETED: set(),
    ExecutionState.BLOCKED: set(),
    ExecutionState.REJECTED: set(),
    ExecutionState.TIMEOUT: set(),
    ExecutionState.FAILED: set(),
    ExecutionState.ROLLED_BACK: set(),
    ExecutionState.CANCELLED: set(),
}


@dataclass
class ExecutionStateMachine:
    """
    Formal execution state machine with invariant checks.
    
    Ensures:
    - Valid transitions only
    - State history preserved
    - Invariant checks at each transition
    """
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.current_state = ExecutionState.CREATED
        self.state_history: List[Dict] = []
        self._transition_count = 0
        
        # Record initial state
        self._record_transition(
            from_state=None,
            to_state=ExecutionState.CREATED,
            reason="execution_started"
        )
    
    def _record_transition(
        self,
        from_state: Optional[ExecutionState],
        to_state: ExecutionState,
        reason: str,
        metadata: Optional[Dict] = None
    ):
        """Record state transition"""
        self._transition_count += 1
        
        transition = {
            "sequence": self._transition_count,
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "reason": reason,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.state_history.append(transition)
        self.current_state = to_state
    
    def can_transition(self, to_state: ExecutionState) -> bool:
        """Check if transition is valid"""
        return to_state in VALID_TRANSITIONS.get(self.current_state, set())
    
    def transition(
        self,
        to_state: ExecutionState,
        reason: str,
        metadata: Optional[Dict] = None,
        enforce_valid: bool = True
    ) -> bool:
        """
        Transition to new state.
        
        Returns True if transition succeeded, False if invalid.
        """
        if enforce_valid and not self.can_transition(to_state):
            return False
        
        self._record_transition(
            from_state=self.current_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata
        )
        
        return True
    
    def force_transition(
        self,
        to_state: ExecutionState,
        reason: str,
        metadata: Optional[Dict] = None
    ):
        """Force transition (bypass validation)"""
        self._record_transition(
            from_state=self.current_state,
            to_state=to_state,
            reason=reason,
            metadata=metadata
        )
    
    def is_terminal(self) -> bool:
        """Check if in terminal state"""
        return self.current_state in {
            ExecutionState.COMPLETED,
            ExecutionState.BLOCKED,
            ExecutionState.REJECTED,
            ExecutionState.TIMEOUT,
            ExecutionState.FAILED,
            ExecutionState.ROLLED_BACK,
            ExecutionState.CANCELLED
        }
    
    def get_valid_transitions(self) -> Set[ExecutionState]:
        """Get valid next states"""
        return VALID_TRANSITIONS.get(self.current_state, set())
    
    def get_history(self) -> List[Dict]:
        """Get full state history"""
        return self.state_history
    
    def validate_invariants(self) -> List[str]:
        """Validate execution invariants"""
        violations = []
        
        # Check 1: First state must be CREATED
        if self.state_history and self.state_history[0].get("from_state") is not None:
            violations.append("First transition must start from CREATED")
        
        # Check 2: No terminal state should have outgoing transitions
        if self.is_terminal() and len(self.state_history) > 1:
            last_transition = self.state_history[-1]
            if last_transition["to_state"] not in {
                s.value for s in {
                    ExecutionState.COMPLETED,
                    ExecutionState.BLOCKED,
                    ExecutionState.REJECTED,
                    ExecutionState.TIMEOUT,
                    ExecutionState.FAILED,
                    ExecutionState.ROLLED_BACK,
                    ExecutionState.CANCELLED
                }
            }:
                violations.append("Terminal state should not have transitions")
        
        # Check 3: All transitions must be valid
        for i, transition in enumerate(self.state_history[1:], 1):
            from_state = transition.get("from_state")
            to_state = transition.get("to_state")
            
            if from_state and to_state:
                try:
                    from_enum = ExecutionState(from_state)
                    to_enum = ExecutionState(to_state)
                    
                    if to_enum not in VALID_TRANSITIONS.get(from_enum, set()):
                        violations.append(
                            f"Invalid transition {i}: {from_state} → {to_state}"
                        )
                except ValueError:
                    violations.append(f"Unknown state at transition {i}")
        
        return violations
    
    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "current_state": self.current_state.value,
            "state_history": self.state_history,
            "transition_count": self._transition_count,
            "is_terminal": self.is_terminal(),
            "valid_transitions": [s.value for s in self.get_valid_transitions()]
        }


class ExecutionStateStore:
    """Store for execution state machines"""
    
    def __init__(self, store_dir: str = "/app/execution_states"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._machines: Dict[str, ExecutionStateMachine] = {}
        self._load_existing()
    
    def _load_existing(self):
        import json
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    
                machine = ExecutionStateMachine(data["execution_id"])
                machine.current_state = ExecutionState(data["current_state"])
                machine.state_history = data["state_history"]
                machine._transition_count = data.get("transition_count", 0)
                
                self._machines[data["execution_id"]] = machine
            except:
                continue
    
    def get_machine(self, execution_id: str) -> ExecutionStateMachine:
        """Get or create state machine for execution"""
        if execution_id not in self._machines:
            self._machines[execution_id] = ExecutionStateMachine(execution_id)
        
        return self._machines[execution_id]
    
    def save_machine(self, machine: ExecutionStateMachine):
        """Save state machine to disk"""
        import json
        
        filename = self.store_dir / f"{machine.execution_id}.json"
        with open(filename, "w") as f:
            json.dump(machine.to_dict(), f, indent=2)


# Global store
_state_store: Optional[ExecutionStateStore] = None


def get_state_store() -> ExecutionStateStore:
    global _state_store
    if _state_store is None:
        _state_store = ExecutionStateStore()
    return _state_store


def get_state_machine(execution_id: str) -> ExecutionStateMachine:
    """Get state machine for execution"""
    return get_state_store().get_machine(execution_id)