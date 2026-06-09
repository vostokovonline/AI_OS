"""
Decision Boundary Snapshot - Where the choice boundary was, not just what was chosen

CRITICAL: Without this, replay is surface-level.
We know "what was chosen" but not "where the decision boundary was".

This becomes:
- Interpretability foundation
- RLHF-like analysis
- Replay divergence detection
- "Why not?" reasoning
"""
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True)
class DecisionBoundarySnapshot:
    """
    Snapshot of decision boundary at selection time.
    
    Stores not just what was chosen, but WHERE the boundary was.
    """
    execution_id: str
    
    # What was selected
    selected_candidate: str
    
    # Full candidate distribution (not just scores)
    candidate_distribution: Dict[str, float]  # skill_id -> probability
    
    # Policy's posterior after scoring
    posterior_distribution: Dict[str, float]
    
    # Uncertainty per candidate
    uncertainty_vector: Dict[str, float]  # skill_id -> uncertainty (0-1)
    
    # Decision parameters
    exploration_coefficient: float  # How much exploration vs exploitation
    decision_temperature: float  # Softmax temperature used
    threshold_margin: float  # How far above threshold was selection
    
    # Constraints active at decision time
    active_constraints: List[str]  # Which constraints were active
    suppressed_candidates: List[str]  # Rejected by hard constraints
    
    # Confidence
    confidence: float  # Confidence in decision (0-1)
    
    # Selection metadata
    selection_reason: str  # Why this was selected
    rejection_reasons: Dict[str, str]  # skill_id -> why rejected
    
    # Chain integrity
    created_at: str
    snapshot_hash: str  # Hash for integrity
    
    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "selected_candidate": self.selected_candidate,
            "candidate_distribution": self.candidate_distribution,
            " posterior_distribution": self.posterior_distribution,
            "uncertainty_vector": self.uncertainty_vector,
            "exploration_coefficient": self.exploration_coefficient,
            "decision_temperature": self.decision_temperature,
            "threshold_margin": self.threshold_margin,
            "active_constraints": self.active_constraints,
            "suppressed_candidates": self.suppressed_candidates,
            "confidence": self.confidence,
            "selection_reason": self.selection_reason,
            "rejection_reasons": self.rejection_reasons,
            "created_at": self.created_at,
            "snapshot_hash": self.snapshot_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "DecisionBoundarySnapshot":
        return DecisionBoundarySnapshot(
            execution_id=data["execution_id"],
            selected_candidate=data["selected_candidate"],
            candidate_distribution=data["candidate_distribution"],
            posterior_distribution=data.get("posterior_distribution", {}),
            uncertainty_vector=data.get("uncertainty_vector", {}),
            exploration_coefficient=data.get("exploration_coefficient", 0.0),
            decision_temperature=data.get("decision_temperature", 1.0),
            threshold_margin=data.get("threshold_margin", 0.0),
            active_constraints=data.get("active_constraints", []),
            suppressed_candidates=data.get("suppressed_candidates", []),
            confidence=data.get("confidence", 0.5),
            selection_reason=data.get("selection_reason", ""),
            rejection_reasons=data.get("rejection_reasons", {}),
            created_at=data["created_at"],
            snapshot_hash=data["snapshot_hash"]
        )
    
    def compute_hash(self) -> str:
        """Compute hash for integrity"""
        hash_input = {
            "execution_id": self.execution_id,
            "selected_candidate": self.selected_candidate,
            "candidate_distribution": self.candidate_distribution,
            "posterior_distribution": self.posterior_distribution,
            "uncertainty_vector": self.uncertainty_vector,
            "exploration_coefficient": self.exploration_coefficient,
            "decision_temperature": self.decision_temperature,
            "active_constraints": self.active_constraints
        }
        return hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]


@dataclass(frozen=True)
class ExecutionIntent:
    """
    Intent object - distinguishes intent from execution.
    
    Enables:
    - Policy reasoning
    - Self-constraint
    - Dynamic planning
    - Capability sandboxing
    """
    intent_id: str
    execution_id: str  # Link to execution
    
    # Goal
    parent_goal: str
    desired_outcome: str
    success_criteria: Dict[str, Any]
    
    # Budgets
    risk_budget: float  # Max acceptable risk (0-1)
    latency_budget_ms: int
    token_budget: int
    
    # Capability scope
    allowed_capabilities: List[str]
    forbidden_capabilities: List[str]
    
    # Autonomy
    autonomy_level: str  # "manual", "supervised", "autonomous"
    requires_approval_for: List[str]  # Actions requiring human approval
    
    # Rollback
    rollback_strategy: Optional[str]
    rollback_trigger: Optional[str]
    
    created_at: str
    
    def to_dict(self) -> dict:
        return {
            "intent_id": self.intent_id,
            "execution_id": self.execution_id,
            "parent_goal": self.parent_goal,
            "desired_outcome": self.desired_outcome,
            "success_criteria": self.success_criteria,
            "risk_budget": self.risk_budget,
            "latency_budget_ms": self.latency_budget_ms,
            "token_budget": self.token_budget,
            "allowed_capabilities": self.allowed_capabilities,
            "forbidden_capabilities": self.forbidden_capabilities,
            "autonomy_level": self.autonomy_level,
            "requires_approval_for": self.requires_approval_for,
            "rollback_strategy": self.rollback_strategy,
            "rollback_trigger": self.rollback_trigger,
            "created_at": self.created_at
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ExecutionIntent":
        return ExecutionIntent(**data)


class DecisionBoundaryStore:
    """Store for decision boundaries"""
    
    def __init__(self, store_dir: str = "/app/decision_boundaries"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._snapshots: Dict[str, DecisionBoundarySnapshot] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    snapshot = DecisionBoundarySnapshot.from_dict(data)
                    self._snapshots[snapshot.execution_id] = snapshot
            except:
                continue
    
    def record(
        self,
        execution_id: str,
        selected_candidate: str,
        candidate_distribution: Dict[str, float],
        posterior: Dict[str, float],
        uncertainty: Dict[str, float],
        constraints: List[str],
        suppressed: List[str],
        temperature: float,
        exploration: float,
        confidence: float,
        selection_reason: str = "",
        rejection_reasons: Dict[str, str] = None
    ) -> DecisionBoundarySnapshot:
        """Record decision boundary"""
        from dataclasses import dataclass
        
        # Compute threshold margin
        scores = list(candidate_distribution.values())
        max_score = max(scores) if scores else 0
        sorted_scores = sorted(scores, reverse=True)
        threshold_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
        
        snapshot = DecisionBoundarySnapshot(
            execution_id=execution_id,
            selected_candidate=selected_candidate,
            candidate_distribution=candidate_distribution,
            posterior_distribution=posterior,
            uncertainty_vector=uncertainty,
            exploration_coefficient=exploration,
            decision_temperature=temperature,
            threshold_margin=threshold_margin,
            active_constraints=constraints,
            suppressed_candidates=suppressed,
            confidence=confidence,
            selection_reason=selection_reason,
            rejection_reasons=rejection_reasons or {},
            created_at=datetime.utcnow().isoformat(),
            snapshot_hash=""  # Will compute
        )
        
        # Compute hash
        snapshot_hash = snapshot.compute_hash()
        
        # Create final with hash
        from dataclasses import replace
        snapshot = replace(snapshot, snapshot_hash=snapshot_hash)
        
        # Save
        self._snapshots[execution_id] = snapshot
        with open(self.store_dir / f"{execution_id}.json", "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        return snapshot
    
    def get(self, execution_id: str) -> Optional[DecisionBoundarySnapshot]:
        """Get boundary for execution"""
        return self._snapshots.get(execution_id)
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self._snapshots:
            return {"total": 0}
        
        confidences = [s.confidence for s in self._snapshots.values()]
        return {
            "total": len(self._snapshots),
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences)
        }


class IntentStore:
    """Store for execution intents"""
    
    def __init__(self, store_dir: str = "/app/intents"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._intents: Dict[str, ExecutionIntent] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    intent = ExecutionIntent.from_dict(data)
                    self._intents[intent.execution_id] = intent
            except:
                continue
    
    def create(
        self,
        execution_id: str,
        parent_goal: str,
        desired_outcome: str,
        success_criteria: Dict[str, Any],
        risk_budget: float = 0.5,
        latency_budget_ms: int = 30000,
        token_budget: int = 10000,
        allowed_capabilities: List[str] = None,
        forbidden_capabilities: List[str] = None,
        autonomy_level: str = "autonomous"
    ) -> ExecutionIntent:
        """Create intent for execution"""
        intent = ExecutionIntent(
            intent_id=uuid4().hex[:8],
            execution_id=execution_id,
            parent_goal=parent_goal,
            desired_outcome=desired_outcome,
            success_criteria=success_criteria,
            risk_budget=risk_budget,
            latency_budget_ms=latency_budget_ms,
            token_budget=token_budget,
            allowed_capabilities=allowed_capabilities or ["*"],
            forbidden_capabilities=forbidden_capabilities or [],
            autonomy_level=autonomy_level,
            requires_approval_for=[],
            rollback_strategy=None,
            rollback_trigger=None,
            created_at=datetime.utcnow().isoformat()
        )
        
        # Save
        self._intents[execution_id] = intent
        with open(self.store_dir / f"{intent.intent_id}.json", "w") as f:
            json.dump(intent.to_dict(), f, indent=2)
        
        return intent
    
    def get(self, execution_id: str) -> Optional[ExecutionIntent]:
        """Get intent for execution"""
        return self._intents.get(execution_id)


# Global stores
_boundary_store: Optional[DecisionBoundaryStore] = None
_intent_store: Optional[IntentStore] = None


def get_boundary_store() -> DecisionBoundaryStore:
    global _boundary_store
    if _boundary_store is None:
        _boundary_store = DecisionBoundaryStore()
    return _boundary_store


def get_intent_store() -> IntentStore:
    global _intent_store
    if _intent_store is None:
        _intent_store = IntentStore()
    return _intent_store


def record_decision_boundary(
    execution_id: str,
    selected_candidate: str,
    candidate_distribution: Dict[str, float],
    posterior: Dict[str, float],
    uncertainty: Dict[str, float],
    constraints: List[str],
    suppressed: List[str],
    temperature: float = 1.0,
    exploration: float = 0.1,
    confidence: float = 0.5,
    selection_reason: str = ""
) -> DecisionBoundarySnapshot:
    """Convenience function to record decision boundary"""
    return get_boundary_store().record(
        execution_id, selected_candidate, candidate_distribution,
        posterior, uncertainty, constraints, suppressed,
        temperature, exploration, confidence, selection_reason
    )


def create_intent(
    execution_id: str,
    parent_goal: str,
    desired_outcome: str,
    **kwargs
) -> ExecutionIntent:
    """Convenience function to create intent"""
    return get_intent_store().create(execution_id, parent_goal, desired_outcome, **kwargs)