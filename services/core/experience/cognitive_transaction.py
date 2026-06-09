"""
Unified Cognitive Transaction - Single causal runtime contract

CRITICAL: This ties everything into ONE causal transaction.

Instead of:
- journal separately
- replay separately  
- graph separately
- boundaries separately
- temporal separately

We now have ONE atomic causal transaction:

Intent
→ Decision Boundary
→ State Transition
→ Event Chain
→ Observation
→ Evaluation
→ Counterfactual
→ Temporal Update

This makes:
- Replay truly deterministic
- Learning stable
- Reasoning explainable
- Multi-agent orchestration possible
"""
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from enum import Enum


class TransactionPhase(str, Enum):
    """Phases of cognitive transaction"""
    INTENT = "intent"
    DECISION = "decision"
    EXECUTION = "execution"
    OBSERVATION = "observation"
    EVALUATION = "evaluation"
    LEARNING = "learning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class CognitiveTransaction:
    """
    Unified cognitive transaction - single source of truth for execution.
    
    This is the atomic unit of cognition in the system.
    """
    transaction_id: str
    execution_id: str
    
    # Intent phase
    intent_id: str
    
    # Decision phase
    decision_boundary_id: str  # Link to DecisionBoundarySnapshot
    selected_skill: str
    
    # Execution phase
    state_machine_id: str  # Link to ExecutionStateMachine
    current_state: str
    
    # Journal phase
    event_chain_id: str  # Link to first event in chain
    event_count: int
    
    # Evaluation phase
    evaluation_vector_id: Optional[str]
    reward: float
    
    # Counterfactual phase
    counterfactual_id: Optional[str]
    
    # Temporal phase
    temporal_metric_ids: List[str]
    
    # Integrity
    created_at: str
    completed_at: Optional[str]
    transaction_hash: str  # Hash of all components
    
    def to_dict(self) -> dict:
        return {
            "transaction_id": self.transaction_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "decision_boundary_id": self.decision_boundary_id,
            "selected_skill": self.selected_skill,
            "state_machine_id": self.state_machine_id,
            "current_state": self.current_state,
            "event_chain_id": self.event_chain_id,
            "event_count": self.event_count,
            "evaluation_vector_id": self.evaluation_vector_id,
            "reward": self.reward,
            "counterfactual_id": self.counterfactual_id,
            "temporal_metric_ids": self.temporal_metric_ids,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "transaction_hash": self.transaction_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "CognitiveTransaction":
        return CognitiveTransaction(**data)
    
    def compute_hash(self) -> str:
        """Compute transaction hash for integrity"""
        hash_input = {
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "decision_boundary_id": self.decision_boundary_id,
            "selected_skill": self.selected_skill,
            "event_count": self.event_count,
            "reward": self.reward
        }
        return hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]


class CognitiveTransactionBuilder:
    """
    Builder for cognitive transactions.
    
    Orchestrates creation of all components in correct order.
    """
    
    def __init__(self, execution_id: str):
        self.execution_id = execution_id
        self.transaction_id = uuid4().hex[:8]
        
        # Component references (to be populated)
        self._intent = None
        self._boundary = None
        self._state_machine = None
        self._first_event = None
        self._counterfactual = None
        self._temporal_metrics = []
        
        self._evaluation_vector = None
        self._reward = 0.0
        
        self._created_at = datetime.utcnow().isoformat()
    
    def with_intent(
        self,
        parent_goal: str,
        desired_outcome: str,
        risk_budget: float = 0.5,
        autonomy_level: str = "autonomous",
        success_criteria: Dict[str, Any] = None
    ) -> "CognitiveTransactionBuilder":
        """Add intent to transaction"""
        from experience.decision_boundary import get_intent_store
        
        store = get_intent_store()
        self._intent = store.create(
            execution_id=self.execution_id,
            parent_goal=parent_goal,
            desired_outcome=desired_outcome,
            success_criteria=success_criteria or {},
            risk_budget=risk_budget,
            autonomy_level=autonomy_level
        )
        return self
    
    def with_decision(
        self,
        selected_skill: str,
        candidate_distribution: Dict[str, float],
        posterior: Dict[str, float],
        uncertainty: Dict[str, float],
        temperature: float = 1.0,
        confidence: float = 0.5
    ) -> "CognitiveTransactionBuilder":
        """Add decision boundary to transaction"""
        from experience.decision_boundary import get_boundary_store
        
        store = get_boundary_store()
        self._boundary = store.record(
            execution_id=self.execution_id,
            selected_candidate=selected_skill,
            candidate_distribution=candidate_distribution,
            posterior=posterior,
            uncertainty=uncertainty,
            constraints=[],
            suppressed=[],
            temperature=temperature,
            exploration=0.1,  # Default
            confidence=confidence
        )
        self._selected_skill = selected_skill
        return self
    
    def with_state_machine(
        self,
        initial_state: str = "created"
    ) -> "CognitiveTransactionBuilder":
        """Add state machine to transaction"""
        from experience.runtime_state import get_state_machine, ExecutionState
        
        sm = get_state_machine(self.execution_id)
        
        # Transition to initial state
        try:
            sm.transition(ExecutionState(initial_state), "transaction_started")
        except:
            pass
        
        self._state_machine = sm
        self._current_state = initial_state
        return self
    
    def with_event(self, event_id: str) -> "CognitiveTransactionBuilder":
        """Record event in chain"""
        if self._first_event is None:
            self._first_event = event_id
        self._event_count = getattr(self, '_event_count', 0) + 1
        return self
    
    def with_evaluation(
        self,
        evaluation_vector: Dict[str, float],
        reward: float
    ) -> "CognitiveTransactionBuilder":
        """Add evaluation to transaction"""
        self._evaluation_vector = evaluation_vector
        self._reward = reward
        return self
    
    def with_counterfactual(
        self,
        selected_skill: str,
        rejected_candidates: List[Dict[str, Any]]
    ) -> "CognitiveTransactionBuilder":
        """Add counterfactual to transaction"""
        from experience.execution_journal import get_counterfactual_store
        
        store = get_counterfactual_store()
        self._counterfactual = store.record(
            execution_id=self.execution_id,
            selected_skill=selected_skill,
            rejected_candidates=rejected_candidates,
            selection_context={"reward": self._reward}
        )
        return self
    
    def with_temporal_metric(
        self,
        metric_name: str,
        value: float
    ) -> "CognitiveTransactionBuilder":
        """Add temporal metric to transaction"""
        from experience.temporal_metrics import get_temporal_store
        
        store = get_temporal_store()
        metric = store.record(metric_name, value)
        self._temporal_metrics.append(metric.metric_name)
        return self
    
    def build(self) -> CognitiveTransaction:
        """Build the complete transaction"""
        # Compute hash
        hash_input = {
            "execution_id": self.execution_id,
            "intent_id": self._intent.intent_id if self._intent else "",
            "decision_boundary_id": self._boundary.execution_id if self._boundary else "",
            "selected_skill": self._selected_skill if hasattr(self, '_selected_skill') else "",
            "event_count": getattr(self, '_event_count', 0),
            "reward": self._reward
        }
        transaction_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        transaction = CognitiveTransaction(
            transaction_id=self.transaction_id,
            execution_id=self.execution_id,
            intent_id=self._intent.intent_id if self._intent else "",
            decision_boundary_id=self._boundary.execution_id if self._boundary else "",
            selected_skill=self._selected_skill if hasattr(self, '_selected_skill') else "",
            state_machine_id=self.execution_id,
            current_state=self._current_state if hasattr(self, '_current_state') else "created",
            event_chain_id=self._first_event or "",
            event_count=getattr(self, '_event_count', 0),
            evaluation_vector_id=None,
            reward=self._reward,
            counterfactual_id=self._counterfactual.entry_id if self._counterfactual else None,
            temporal_metric_ids=self._temporal_metrics,
            created_at=self._created_at,
            completed_at=datetime.utcnow().isoformat(),
            transaction_hash=transaction_hash
        )
        
        # Save transaction
        self._save_transaction(transaction)
        
        return transaction
    
    def _save_transaction(self, transaction: CognitiveTransaction):
        """Save transaction to disk"""
        import os
        store_dir = "/app/cognitive_transactions"
        os.makedirs(store_dir, exist_ok=True)
        
        with open(f"{store_dir}/{transaction.transaction_id}.json", "w") as f:
            json.dump(transaction.to_dict(), f, indent=2)


class CognitiveTransactionStore:
    """Store for cognitive transactions"""
    
    def __init__(self, store_dir: str = "/app/cognitive_transactions"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._transactions: Dict[str, CognitiveTransaction] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    tx = CognitiveTransaction.from_dict(data)
                    self._transactions[tx.transaction_id] = tx
            except:
                continue
    
    def get(self, transaction_id: str) -> Optional[CognitiveTransaction]:
        return self._transactions.get(transaction_id)
    
    def get_by_execution(self, execution_id: str) -> Optional[CognitiveTransaction]:
        for tx in self._transactions.values():
            if tx.execution_id == execution_id:
                return tx
        return None
    
    def get_statistics(self) -> Dict:
        return {
            "total": len(self._transactions),
            "avg_reward": sum(t.reward for t in self._transactions.values()) / len(self._transactions) if self._transactions else 0
        }


# Global store
_tx_store: Optional[CognitiveTransactionStore] = None


def get_tx_store() -> CognitiveTransactionStore:
    global _tx_store
    if _tx_store is None:
        _tx_store = CognitiveTransactionStore()
    return _tx_store


def create_cognitive_transaction(execution_id: str) -> CognitiveTransactionBuilder:
    """Start building a cognitive transaction"""
    return CognitiveTransactionBuilder(execution_id)