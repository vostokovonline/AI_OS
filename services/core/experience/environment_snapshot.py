"""
Environment Snapshot - Frozen execution environment for replay

CRITICAL: Without this, replay can drift due to environment changes.
This captures everything that affects execution decisions.
"""
import hashlib
import json
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """
    Frozen state of execution environment.
    
    This ensures replay consistency even if environment changes.
    """
    # Model state
    model_name: str
    model_version: str
    tokenizer_version: str
    
    # Registry state
    registry_version: str
    registry_snapshot_id: str
    
    # Feature extraction
    feature_schema_version: str
    
    # Policy state
    policy_version: str
    policy_snapshot_id: str
    
    # Routing
    routing_policy_version: str
    
    # Capability constraints
    capability_rules_hash: str  # Hash of active capability constraints
    
    # Execution
    enforcement_mode: str
    
    # Metadata
    captured_at: str
    integrity_hash: str  # Hash of all components
    
    @staticmethod
    def capture() -> "EnvironmentSnapshot":
        """Capture current environment state"""
        from experience.enforcement_config import get_enforcement_config
        from experience.policy_snapshot import get_frozen_policy_for_replay
        from experience.registry_snapshot import get_registry_snapshot_store
        from experience.feature_extraction import FEATURE_SCHEMA_VERSION
        
        # Get state
        config = get_enforcement_config()
        policy = get_frozen_policy_for_replay()
        registry_store = get_registry_snapshot_store()
        registry = registry_store.get_latest()
        
        # Build snapshot
        env = EnvironmentSnapshot(
            model_name="qwen2.5-coder",  # From LLM config
            model_version="latest",  # Should be from config
            tokenizer_version="latest",
            registry_version="v1",
            registry_snapshot_id=registry.snapshot_id if registry else "none",
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            policy_version=policy.policy_version,
            policy_snapshot_id=policy.snapshot_id,
            routing_policy_version="v1",
            capability_rules_hash="static_v1",  # Should be from config
            enforcement_mode=config.mode.value,
            captured_at=datetime.utcnow().isoformat(),
            integrity_hash=""  # Will compute
        )
        
        # Compute integrity hash
        integrity_data = {
            "model_name": env.model_name,
            "model_version": env.model_version,
            "registry_snapshot_id": env.registry_snapshot_id,
            "policy_snapshot_id": env.policy_snapshot_id,
            "feature_schema_version": env.feature_schema_version,
            "enforcement_mode": env.enforcement_mode
        }
        integrity_hash = hashlib.sha256(
            json.dumps(integrity_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        from dataclasses import replace
        return replace(env, integrity_hash=integrity_hash)
    
    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "tokenizer_version": self.tokenizer_version,
            "registry_version": self.registry_version,
            "registry_snapshot_id": self.registry_snapshot_id,
            "feature_schema_version": self.feature_schema_version,
            "policy_version": self.policy_version,
            "policy_snapshot_id": self.policy_snapshot_id,
            "routing_policy_version": self.routing_policy_version,
            "capability_rules_hash": self.capability_rules_hash,
            "enforcement_mode": self.enforcement_mode,
            "captured_at": self.captured_at,
            "integrity_hash": self.integrity_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "EnvironmentSnapshot":
        return EnvironmentSnapshot(**data)


@dataclass(frozen=True)
class ExecutionContract:
    """
    Immutable contract for execution semantics.
    
    Defines what a skill execution MUST do, CAN do, and MUST NOT do.
    This ensures replay is deterministic and auditable.
    """
    # Identity
    contract_id: str
    skill_id: str
    
    # Execution contract
    tool_signature: str  # Hash of expected tool behavior
    capability_constraints: Dict[str, bool]  # capability → allowed
    expected_side_effects: list  # What side effects are expected
    allowed_outputs: list  # What outputs are allowed
    deterministic_flags: Dict[str, bool]  # execution guarantees
    
    # Constraints
    max_retries: int
    timeout_ms: int
    resource_limits: Dict[str, int]
    
    # Metadata
    created_at: str
    version: str
    
    @staticmethod
    def create_for_skill(skill_id: str) -> "ExecutionContract":
        """Create contract for a skill (default constraints)"""
        from uuid import uuid4
        
        # Default constraints
        capability_constraints = {
            "filesystem.read": True,
            "filesystem.write": True,
            "network.http": True,
            "network.ws": False,
            "subprocess.execute": False,  # Restricted
            "memory.mutate": False,  # Cannot mutate long-term memory
            "self.modify": False,  # Cannot modify self
            "human.override": True  # Can be overridden by human
        }
        
        deterministic_flags = {
            "idempotent": False,  # Default
            "pure": False,
            "side_effect_free": False,
            "deterministic_output": True
        }
        
        return ExecutionContract(
            contract_id=uuid4().hex[:8],
            skill_id=skill_id,
            tool_signature=hashlib.sha256(skill_id.encode()).hexdigest()[:16],
            capability_constraints=capability_constraints,
            expected_side_effects=[],
            allowed_outputs=["text", "json", "artifact"],
            deterministic_flags=deterministic_flags,
            max_retries=3,
            timeout_ms=30000,
            resource_limits={"memory_mb": 512, "cpu_percent": 50},
            created_at=datetime.utcnow().isoformat(),
            version="v1"
        )
    
    def to_dict(self) -> dict:
        return {
            "contract_id": self.contract_id,
            "skill_id": self.skill_id,
            "tool_signature": self.tool_signature,
            "capability_constraints": self.capability_constraints,
            "expected_side_effects": self.expected_side_effects,
            "allowed_outputs": self.allowed_outputs,
            "deterministic_flags": self.deterministic_flags,
            "max_retries": self.max_retries,
            "timeout_ms": self.timeout_ms,
            "resource_limits": self.resource_limits,
            "created_at": self.created_at,
            "version": self.version
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ExecutionContract":
        return ExecutionContract(**data)


@dataclass
class CandidateEvaluation:
    """
    Snapshot of candidate evaluation at selection time.
    
    Stores prior scores, sampled values, and selection decision.
    This enables regret analysis and counterfactual replay.
    """
    skill_id: str
    prior_score: float  # Thompson sampling prior
    sampled_score: float  # Actual sampled value
    estimated_reward: float  # Estimated reward
    rank: int  # Final rank
    selected: bool  # Was this selected
    
    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "prior_score": self.prior_score,
            "sampled_score": self.sampled_score,
            "estimated_reward": self.estimated_reward,
            "rank": self.rank,
            "selected": self.selected
        }
    
    @staticmethod
    def from_dict(data: dict) -> "CandidateEvaluation":
        return CandidateEvaluation(**data)


class ExecutionContractStore:
    """Store for execution contracts"""
    
    def __init__(self, store_dir: str = "/app/execution_contracts"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._contracts: Dict[str, ExecutionContract] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    contract = ExecutionContract.from_dict(data)
                    self._contracts[contract.skill_id] = contract
            except:
                continue
    
    def get_contract(self, skill_id: str) -> ExecutionContract:
        """Get or create contract for skill"""
        if skill_id not in self._contracts:
            contract = ExecutionContract.create_for_skill(skill_id)
            self._contracts[skill_id] = contract
            
            # Save to disk
            with open(self.store_dir / f"{contract.contract_id}.json", "w") as f:
                json.dump(contract.to_dict(), f, indent=2)
        
        return self._contracts[skill_id]
    
    def get_statistics(self) -> dict:
        return {
            "total_contracts": len(self._contracts),
            "skills": list(self._contracts.keys())
        }


# Global stores
_env_snapshot: Optional[EnvironmentSnapshot] = None
_contract_store: Optional[ExecutionContractStore] = None


def capture_environment() -> EnvironmentSnapshot:
    """Capture current environment snapshot"""
    global _env_snapshot
    _env_snapshot = EnvironmentSnapshot.capture()
    return _env_snapshot


def get_environment_snapshot() -> EnvironmentSnapshot:
    """Get current or latest environment snapshot"""
    global _env_snapshot
    if _env_snapshot is None:
        _env_snapshot = EnvironmentSnapshot.capture()
    return _env_snapshot


def get_contract_store() -> ExecutionContractStore:
    """Get execution contract store"""
    global _contract_store
    if _contract_store is None:
        _contract_store = ExecutionContractStore()
    return _contract_store


def get_contract(skill_id: str) -> ExecutionContract:
    """Get contract for skill"""
    return get_contract_store().get_contract(skill_id)