"""
Replay Bundle - Complete frozen state for forensic-grade replay

CRITICAL: This is the SINGLE SOURCE OF TRUTH for replay.
Everything needed to exactly reproduce a decision must be here.

ReplayBundle contains:
- ExecutionEnvelope (what happened)
- RegistrySnapshot (what skills were available)
- CandidateSetSnapshot (exactly which candidates were considered)
- FrozenPolicySnapshot (policy state at time)
- EnforcementSnapshot (enforcement mode at time)
- CanonicalFeatureVector (extracted features)
- LineageSnapshot (execution ancestry)
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from experience.execution_envelope import ExecutionEnvelope
from experience.registry_snapshot import RegistrySnapshot
from experience.policy_snapshot import FrozenPolicySnapshot
from experience.feature_extraction import FeatureVector
from experience.execution_lineage import ExecutionNode


@dataclass(frozen=True)
class CandidateSetSnapshot:
    """Immutable snapshot of exactly which candidates were considered"""
    candidates: List[str]  # Canonical skill IDs in sorted order
    candidate_hash: str  # Hash for integrity
    
    @staticmethod
    def create(skill_ids: List[str]) -> "CandidateSetSnapshot":
        """Create from skill IDs - sorted for determinism"""
        import hashlib
        sorted_ids = sorted(skill_ids)
        candidate_hash = hashlib.sha256(
            json.dumps(sorted_ids, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return CandidateSetSnapshot(
            candidates=sorted_ids,
            candidate_hash=candidate_hash
        )


@dataclass(frozen=True)
class EnforcementSnapshot:
    """Immutable snapshot of enforcement state"""
    mode: str  # warn/quarantine/hard_fail
    can_train_policy: bool
    quarantine_enabled: bool
    
    @staticmethod
    def create() -> "EnforcementSnapshot":
        from experience.enforcement_config import get_enforcement_config
        config = get_enforcement_config()
        
        return EnforcementSnapshot(
            mode=config.mode.value,
            can_train_policy=True,  # At time of execution
            quarantine_enabled=(config.mode.value != "warn")
        )


@dataclass(frozen=True)
class LineageSnapshot:
    """Immutable snapshot of execution lineage at time"""
    node_id: str
    execution_id: str
    parent_node_id: Optional[str]
    edge_type: Optional[str]
    decomposition_depth: int
    ancestry: List[str]  # List of ancestor node IDs
    
    @staticmethod
    def from_node(node: ExecutionNode, ancestors: List[str]) -> "LineageSnapshot":
        return LineageSnapshot(
            node_id=node.node_id,
            execution_id=node.execution_id,
            parent_node_id=node.parent_node_id,
            edge_type=node.edge_type.value if node.edge_type else None,
            decomposition_depth=node.decomposition_depth,
            ancestry=ancestors
        )


@dataclass(frozen=True)
class ReplayBundle:
    """
    Complete frozen state for forensic-grade replay.
    
    This is self-contained - no external dependencies needed for replay.
    Every decision can be exactly reproduced from this bundle.
    """
    bundle_id: str
    created_at: str
    
    # Core
    envelope: ExecutionEnvelope
    features: FeatureVector
    
    # State at time of execution
    registry_snapshot: Optional[RegistrySnapshot]
    candidate_set: CandidateSetSnapshot
    policy_snapshot: FrozenPolicySnapshot
    enforcement: EnforcementSnapshot
    lineage: Optional[LineageSnapshot]
    
    # Determinism
    feature_hash: str
    integrity_hash: str  # Hash of all components for verification
    
    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "created_at": self.created_at,
            "envelope": self.envelope.to_dict(),
            "features": self.features.to_dict(),
            "registry_snapshot": self.registry_snapshot.to_dict() if self.registry_snapshot else None,
            "candidate_set": {
                "candidates": self.candidate_set.candidates,
                "candidate_hash": self.candidate_set.candidate_hash
            },
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "enforcement": {
                "mode": self.enforcement.mode,
                "can_train_policy": self.enforcement.can_train_policy,
                "quarantine_enabled": self.enforcement.quarantine_enabled
            },
            "lineage": self.lineage.to_dict() if self.lineage else None,
            "feature_hash": self.feature_hash,
            "integrity_hash": self.integrity_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ReplayBundle":
        from experience.execution_envelope import ExecutionEnvelope
        from experience.feature_extraction import FeatureVector
        from experience.policy_snapshot import FrozenPolicySnapshot
        
        envelope = ExecutionEnvelope(**data["envelope"])
        features = FeatureVector.from_dict(data["features"])
        policy_snapshot = FrozenPolicySnapshot.from_dict(data["policy_snapshot"])
        
        registry = None
        if data.get("registry_snapshot"):
            from experience.registry_snapshot import RegistrySnapshot
            registry = RegistrySnapshot.from_dict(data["registry_snapshot"])
        
        lineage = None
        if data.get("lineage"):
            lineage = LineageSnapshot(**data["lineage"])
        
        return ReplayBundle(
            bundle_id=data["bundle_id"],
            created_at=data["created_at"],
            envelope=envelope,
            features=features,
            registry_snapshot=registry,
            candidate_set=CandidateSetSnapshot(
                candidates=data["candidate_set"]["candidates"],
                candidate_hash=data["candidate_set"]["candidate_hash"]
            ),
            policy_snapshot=policy_snapshot,
            enforcement=EnforcementSnapshot(**data["enforcement"]),
            lineage=lineage,
            feature_hash=data["feature_hash"],
            integrity_hash=data["integrity_hash"]
        )
    
    def derive_seed(self) -> int:
        """Derive deterministic seed from bundle for Thompson sampling"""
        import hashlib
        
        # Seed = hash(execution_id + policy_snapshot_id + feature_hash)
        seed_input = (
            self.envelope.execution_id +
            self.policy_snapshot.snapshot_id +
            self.feature_hash
        )
        
        seed = int(hashlib.sha256(seed_input.encode()).hexdigest()[:8], 16) % (2**31)
        return seed


class ReplayBundleStore:
    """Store for replay bundles"""
    
    def __init__(self, store_dir: str = "/app/replay_bundles"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._bundles: Dict[str, ReplayBundle] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    bundle = ReplayBundle.from_dict(data)
                    self._bundles[bundle.bundle_id] = bundle
            except:
                continue
    
    def create_bundle(
        self,
        envelope: ExecutionEnvelope,
        features: FeatureVector,
        candidate_skills: List[str]
    ) -> ReplayBundle:
        """Create complete replay bundle from execution"""
        import hashlib
        
        # Get state snapshots
        from experience.registry_snapshot import get_registry_snapshot_store
        registry_store = get_registry_snapshot_store()
        registry = registry_store.get_latest()
        
        from experience.policy_snapshot import get_frozen_policy_for_replay
        policy = get_frozen_policy_for_replay()
        
        enforcement = EnforcementSnapshot.create()
        candidate_set = CandidateSetSnapshot.create(candidate_skills)
        
        lineage = None  # TODO: Link to lineage graph
        
        # Create bundle
        bundle = ReplayBundle(
            bundle_id=uuid4().hex[:8],
            created_at=datetime.utcnow().isoformat(),
            envelope=envelope,
            features=features,
            registry_snapshot=registry,
            candidate_set=candidate_set,
            policy_snapshot=policy,
            enforcement=enforcement,
            lineage=lineage,
            feature_hash=features.feature_hash,
            integrity_hash=""  # Will compute
        )
        
        # Compute integrity hash
        integrity_data = {
            "envelope_id": envelope.execution_id,
            "policy_snapshot_id": policy.snapshot_id,
            "feature_hash": features.feature_hash,
            "candidate_hash": candidate_set.candidate_hash
        }
        integrity_hash = hashlib.sha256(
            json.dumps(integrity_data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Create final bundle with hash
        from dataclasses import replace
        bundle = replace(bundle, integrity_hash=integrity_hash)
        
        # Save to disk
        filename = self.store_dir / f"{bundle.bundle_id}.json"
        with open(filename, "w") as f:
            json.dump(bundle.to_dict(), f, indent=2)
        
        self._bundles[bundle.bundle_id] = bundle
        
        return bundle
    
    def get(self, bundle_id: str) -> Optional[ReplayBundle]:
        return self._bundles.get(bundle_id)
    
    def get_all(self) -> List[ReplayBundle]:
        return sorted(self._bundles.values(), key=lambda b: b.created_at, reverse=True)
    
    def get_by_envelope(self, envelope_id: str) -> Optional[ReplayBundle]:
        for bundle in self._bundles.values():
            if bundle.envelope.execution_id == envelope_id:
                return bundle
        return None


# Global store
_bundle_store: Optional[ReplayBundleStore] = None


def get_bundle_store() -> ReplayBundleStore:
    global _bundle_store
    if _bundle_store is None:
        _bundle_store = ReplayBundleStore()
    return _bundle_store


def create_replay_bundle(
    envelope: ExecutionEnvelope,
    candidate_skills: List[str]
) -> ReplayBundle:
    """Create replay bundle from envelope"""
    from experience.feature_extraction import FeatureExtractor
    
    features = FeatureExtractor.extract_from_envelope(envelope)
    
    store = get_bundle_store()
    return store.create_bundle(envelope, features, candidate_skills)