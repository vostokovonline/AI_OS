"""
Execution Envelope - Immutable execution contract

All execution MUST go through envelope - no raw context or mutable state.
Envelope is the single source of truth for what was executed.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True)
class ExecutionEnvelope:
    """
    Immutable execution contract.
    
    Execution runtime receives ONLY this object - no mutable context, no globals.
    """
    # Identity
    trace_id: str = ""
    execution_id: str = ""
    
    # Policy
    policy_version: str = "legacy_v1"
    
    # Skills - ResolvedSkill canonical IDs only
    selected_skill_id: str = ""
    shadow_skill_id: Optional[str] = None
    candidate_skill_ids: List[str] = field(default_factory=list)
    
    # Context
    context_features: Dict = field(default_factory=dict)
    context_hash: str = ""  # SHA256 of canonical JSON(context_features)
    
    # Metadata
    created_at: str = ""
    goal_type: str = ""
    domain: str = ""
    
    @staticmethod
    def compute_context_hash(context: Dict) -> str:
        """Compute deterministic hash of context features"""
        canonical = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    @classmethod
    def create(
        cls,
        trace_id: str,
        policy_version: str,
        selected_skill_id: str,
        shadow_skill_id: Optional[str],
        candidate_skill_ids: List[str],
        context_features: Dict,
        goal_type: str,
        domain: str
    ) -> "ExecutionEnvelope":
        """Factory method to create envelope with computed hash"""
        return cls(
            trace_id=trace_id,
            execution_id=uuid4().hex[:8],
            policy_version=policy_version,
            selected_skill_id=selected_skill_id,
            shadow_skill_id=shadow_skill_id,
            candidate_skill_ids=candidate_skill_ids,
            context_features=context_features,
            context_hash=cls.compute_context_hash(context_features),
            created_at=datetime.utcnow().isoformat(),
            goal_type=goal_type,
            domain=domain
        )
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "policy_version": self.policy_version,
            "selected_skill_id": self.selected_skill_id,
            "shadow_skill_id": self.shadow_skill_id,
            "candidate_skill_ids": self.candidate_skill_ids,
            "context_features": self.context_features,
            "context_hash": self.context_hash,
            "created_at": self.created_at,
            "goal_type": self.goal_type,
            "domain": self.domain
        }


class ExecutionEnvelopeStore:
    """Store for execution envelopes - used for replay"""
    
    def __init__(self, store_dir: str = "/app/execution_envelopes"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
    
    def save(self, envelope: ExecutionEnvelope) -> str:
        """Save envelope to store"""
        import json
        
        filename = self.store_dir / f"{envelope.execution_id}.json"
        with open(filename, "w") as f:
            json.dump(envelope.to_dict(), f, indent=2)
        
        return envelope.execution_id
    
    def load(self, execution_id: str) -> Optional[ExecutionEnvelope]:
        """Load envelope by execution_id"""
        import json
        
        filename = self.store_dir / f"{execution_id}.json"
        if not filename.exists():
            return None
        
        with open(filename, "r") as f:
            data = json.load(f)
        
        return ExecutionEnvelope(**data)
    
    def get_all(self) -> List[ExecutionEnvelope]:
        """Get all envelopes"""
        import json
        
        envelopes = []
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    envelopes.append(ExecutionEnvelope(**json.load(f)))
            except:
                continue
        
        return envelopes