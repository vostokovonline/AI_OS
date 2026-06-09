"""
Evaluation Vector - Multi-objective evaluation for promotion

CRITICAL: Promotion must be based on multiple dimensions, not just reward.
This enables Pareto-optimal policy evolution.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EvaluationVector:
    """
    Multi-dimensional evaluation vector for policy assessment.
    
    Dimensions:
    - success_rate: Historical success rate
    - latency_score: Execution latency efficiency (1.0 = fast)
    - token_efficiency: Cost efficiency (1.0 = cheap)
    - safety_score: Safety/constraint compliance (1.0 = safe)
    - stability_score: Variance in outcomes (1.0 = stable)
    - replay_agreement: Policy agreement in replay (1.0 = full agree)
    - human_override_rate: How often human overrides (0.0 = never)
    - hallucination_risk: Risk of hallucination (0.0 = none)
    - repair_cost: Cost to fix failures (0.0 = free)
    """
    success_rate: float = 0.5
    latency_score: float = 0.5
    token_efficiency: float = 0.5
    safety_score: float = 0.5
    stability_score: float = 0.5
    replay_agreement: float = 0.5
    human_override_rate: float = 0.0
    hallucination_risk: float = 0.0
    repair_cost: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "success_rate": self.success_rate,
            "latency_score": self.latency_score,
            "token_efficiency": self.token_efficiency,
            "safety_score": self.safety_score,
            "stability_score": self.stability_score,
            "replay_agreement": self.replay_agreement,
            "human_override_rate": self.human_override_rate,
            "hallucination_risk": self.hallucination_risk,
            "repair_cost": self.repair_cost
        }
    
    @staticmethod
    def from_dict(data: dict) -> "EvaluationVector":
        return EvaluationVector(**data)
    
    @staticmethod
    def from_execution_result(
        success: bool,
        latency_ms: float,
        tokens_used: int,
        failed: bool = False
    ) -> "EvaluationVector":
        """Create evaluation vector from execution result"""
        # Success rate
        success_rate = 1.0 if success else 0.0
        
        # Latency score (normalize - lower is better)
        # Assuming 10s baseline for normalization
        latency_score = max(0, 1.0 - (latency_ms / 10000))
        
        # Token efficiency (normalize - fewer is better)
        # Assuming 10000 tokens baseline
        token_efficiency = max(0, 1.0 - (tokens_used / 10000))
        
        # Safety - check if failure was due to safety
        safety_score = 1.0 if not (failed and "safety" in str(failed)) else 0.0
        
        return EvaluationVector(
            success_rate=success_rate,
            latency_score=latency_score,
            token_efficiency=token_efficiency,
            safety_score=safety_score,
            stability_score=0.5,  # Would need historical data
            replay_agreement=0.5,
            human_override_rate=0.0,
            hallucination_risk=0.0,
            repair_cost=1.0 if failed else 0.0
        )
    
    def dominates(self, other: "EvaluationVector") -> bool:
        """
        Check if this vector dominates other (Pareto optimal).
        
        A dominates B if it's better or equal in all dimensions
        and strictly better in at least one.
        """
        at_least_one_better = False
        
        for dim in ["success_rate", "latency_score", "token_efficiency", 
                    "safety_score", "stability_score", "replay_agreement"]:
            self_val = getattr(self, dim)
            other_val = getattr(other, dim)
            
            if self_val < other_val:
                return False  # Worse in this dimension
            if self_val > other_val:
                at_least_one_better = True
        
        # For these, lower is better
        for dim in ["human_override_rate", "hallucination_risk", "repair_cost"]:
            self_val = getattr(self, dim)
            other_val = getattr(other, dim)
            
            if self_val > other_val:
                return False
            if self_val < other_val:
                at_least_one_better = True
        
        return at_least_one_better
    
    def weighted_score(self, weights: Dict[str, float]) -> float:
        """Compute weighted score"""
        score = 0.0
        total_weight = 0.0
        
        # Higher is better dimensions
        for dim in ["success_rate", "latency_score", "token_efficiency", 
                    "safety_score", "stability_score", "replay_agreement"]:
            if dim in weights:
                score += getattr(self, dim) * weights[dim]
                total_weight += weights[dim]
        
        # Lower is better dimensions (invert)
        for dim in ["human_override_rate", "hallucination_risk", "repair_cost"]:
            if dim in weights:
                score += (1.0 - getattr(self, dim)) * weights[dim]
                total_weight += weights[dim]
        
        return score / total_weight if total_weight > 0 else 0.0


@dataclass
class EvaluationRecord:
    """Single evaluation record for a policy/capability"""
    record_id: str
    capability: str  # e.g., "filesystem.write", "web.search"
    vector: EvaluationVector
    sample_count: int
    evaluated_at: str
    
    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "capability": self.capability,
            "vector": self.vector.to_dict(),
            "sample_count": self.sample_count,
            "evaluated_at": self.evaluated_at
        }


class CapabilityEvaluationStore:
    """Store for capability evaluations"""
    
    def __init__(self, store_dir: str = "/app/capability_evaluations"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._records: Dict[str, EvaluationRecord] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    record = EvaluationRecord(
                        record_id=data["record_id"],
                        capability=data["capability"],
                        vector=EvaluationVector.from_dict(data["vector"]),
                        sample_count=data["sample_count"],
                        evaluated_at=data["evaluated_at"]
                    )
                    self._records[record.record_id] = record
            except:
                continue
    
    def record_evaluation(
        self,
        capability: str,
        vector: EvaluationVector,
        sample_count: int
    ) -> EvaluationRecord:
        """Record evaluation for capability"""
        from uuid import uuid4
        
        record = EvaluationRecord(
            record_id=uuid4().hex[:8],
            capability=capability,
            vector=vector,
            sample_count=sample_count,
            evaluated_at=datetime.utcnow().isoformat()
        )
        
        # Save to disk
        filename = self.store_dir / f"{record.record_id}.json"
        with open(filename, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        
        self._records[record.record_id] = record
        
        return record
    
    def get_latest(self, capability: str) -> Optional[EvaluationRecord]:
        """Get latest evaluation for capability"""
        capability_records = [
            r for r in self._records.values() if r.capability == capability
        ]
        
        if not capability_records:
            return None
        
        return max(capability_records, key=lambda r: r.evaluated_at)
    
    def get_all_capabilities(self) -> List[str]:
        """Get all evaluated capabilities"""
        return list(set(r.capability for r in self._records.values()))
    
    def get_statistics(self) -> dict:
        return {
            "total_records": len(self._records),
            "capabilities": len(self.get_all_capabilities())
        }


# Global store
_eval_store: Optional[CapabilityEvaluationStore] = None


def get_evaluation_store() -> CapabilityEvaluationStore:
    global _eval_store
    if _eval_store is None:
        _eval_store = CapabilityEvaluationStore()
    return _eval_store