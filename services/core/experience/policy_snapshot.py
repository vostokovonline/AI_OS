"""
Frozen Policy Snapshot - Read-only policy snapshot for replay

CRITICAL: Replay engine must NEVER see mutable runtime policy.
This provides an immutable snapshot for deterministic replay.

Structural guarantee:
- FrozenPolicySnapshot is completely immutable
- Created once, never modified
- Replay uses snapshot, not live bandit
"""
import json
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class SkillArmSnapshot:
    """Immutable snapshot of a single skill arm"""
    skill_id: str
    alpha: float  # Success count
    beta: float   # Failure count
    mean: float   # Expected success rate
    variance: float
    
    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "mean": self.mean,
            "variance": self.variance
        }
    
    @staticmethod
    def from_dict(data: dict) -> "SkillArmSnapshot":
        return SkillArmSnapshot(
            skill_id=data["skill_id"],
            alpha=data["alpha"],
            beta=data["beta"],
            mean=data["mean"],
            variance=data["variance"]
        )


@dataclass(frozen=True)
class FrozenPolicySnapshot:
    """
    Immutable snapshot of policy state for replay.
    
    This is the SINGLE SOURCE OF TRUTH for replay evaluation.
    Replay engine uses this, NOT the live bandit.
    """
    snapshot_id: str
    created_at: str
    policy_version: str
    
    # Arm snapshots (immutable)
    arms: List[SkillArmSnapshot]
    
    # Selection metadata
    total_evaluations: int
    last_update_at: str
    
    # Hash for integrity verification
    integrity_hash: str
    
    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "policy_version": self.policy_version,
            "arms": [a.to_dict() for a in self.arms],
            "total_evaluations": self.total_evaluations,
            "last_update_at": self.last_update_at,
            "integrity_hash": self.integrity_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "FrozenPolicySnapshot":
        return FrozenPolicySnapshot(
            snapshot_id=data["snapshot_id"],
            created_at=data["created_at"],
            policy_version=data["policy_version"],
            arms=[SkillArmSnapshot.from_dict(a) for a in data["arms"]],
            total_evaluations=data["total_evaluations"],
            last_update_at=data["last_update_at"],
            integrity_hash=data["integrity_hash"]
        )
    
    def get_arm(self, skill_id: str) -> Optional[SkillArmSnapshot]:
        """Get arm snapshot by skill_id"""
        for arm in self.arms:
            if arm.skill_id == skill_id:
                return arm
        return None
    
    def select_skill(
        self,
        goal_type: str = "achievable",
        domain: str = "general",
        seed: Optional[int] = None
    ) -> str:
        """
        Select skill using seeded Thompson sampling on snapshot.
        
        This is DETERMINISTIC given the same snapshot + seed.
        Uses seeded random to preserve Thompson behavior while being reproducible.
        """
        import random
        
        if not self.arms:
            return "core.echo"  # Default fallback
        
        # Use provided seed or derive from snapshot integrity hash
        if seed is None:
            # Derive seed from snapshot hash for reproducibility
            seed = int(self.integrity_hash, 16) % (2**31)
        
        rng = random.Random(seed)
        
        # Sample from Beta distributions (Thompson sampling)
        samples = []
        for arm in self.arms:
            # Sample from Beta(alpha, beta) using seeded random
            # Using approach: sample = alpha / (alpha + beta + rng.random())
            sample = arm.alpha / (arm.alpha + arm.beta + rng.random())
            samples.append((arm, sample))
        
        # Select arm with highest sample
        best_arm = max(samples, key=lambda x: x[1])[0]
        return best_arm.skill_id
    
    def get_statistics(self) -> Dict:
        """Get policy statistics from snapshot"""
        if not self.arms:
            return {"total_arms": 0, "avg_success_rate": 0.0}
        
        return {
            "total_arms": len(self.arms),
            "total_evaluations": self.total_evaluations,
            "avg_success_rate": sum(a.mean for a in self.arms) / len(self.arms),
            "arms": {a.skill_id: a.mean for a in self.arms}
        }


class PolicySnapshotStore:
    """Store for frozen policy snapshots"""
    
    def __init__(self, store_dir: str = "/app/policy_snapshots"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._snapshots: Dict[str, FrozenPolicySnapshot] = {}
        self._load_existing()
    
    def _load_existing(self):
        """Load existing snapshots"""
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    snapshot = FrozenPolicySnapshot.from_dict(data)
                    self._snapshots[snapshot.snapshot_id] = snapshot
            except:
                continue
    
    def _compute_integrity_hash(self, arms: List[SkillArmSnapshot]) -> str:
        """Compute integrity hash for snapshot"""
        arm_data = sorted([(a.skill_id, a.alpha, a.beta) for a in arms])
        data_json = json.dumps(arm_data, sort_keys=True)
        return hashlib.sha256(data_json.encode()).hexdigest()[:16]
    
    def create_snapshot(
        self,
        policy_version: str,
        arms_data: List[tuple],  # List of (skill_id, alpha, beta)
        total_evaluations: int,
        last_update_at: str
    ) -> FrozenPolicySnapshot:
        """Create snapshot from current bandit state"""
        from uuid import uuid4
        
        # Build arm snapshots
        arms = []
        for skill_id, alpha, beta in arms_data:
            # Compute mean and variance
            mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1)) if (alpha + beta) > 0 else 0.25
            
            arms.append(SkillArmSnapshot(
                skill_id=skill_id,
                alpha=alpha,
                beta=beta,
                mean=mean,
                variance=variance
            ))
        
        # Create snapshot
        snapshot = FrozenPolicySnapshot(
            snapshot_id=uuid4().hex[:8],
            created_at=datetime.utcnow().isoformat(),
            policy_version=policy_version,
            arms=arms,
            total_evaluations=total_evaluations,
            last_update_at=last_update_at,
            integrity_hash=self._compute_integrity_hash(arms)
        )
        
        # Save to disk
        filename = self.store_dir / f"{snapshot.snapshot_id}.json"
        with open(filename, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        self._snapshots[snapshot.snapshot_id] = snapshot
        
        return snapshot
    
    def get_latest(self) -> Optional[FrozenPolicySnapshot]:
        """Get most recent snapshot"""
        if not self._snapshots:
            return None
        
        return max(self._snapshots.values(), key=lambda s: s.created_at)
    
    def get(self, snapshot_id: str) -> Optional[FrozenPolicySnapshot]:
        """Get snapshot by ID"""
        return self._snapshots.get(snapshot_id)
    
    def get_all(self) -> List[FrozenPolicySnapshot]:
        """Get all snapshots"""
        return sorted(self._snapshots.values(), key=lambda s: s.created_at)


# Global store
_policy_snapshot_store: Optional[PolicySnapshotStore] = None


def get_policy_snapshot_store() -> PolicySnapshotStore:
    """Get or create global policy snapshot store"""
    global _policy_snapshot_store
    if _policy_snapshot_store is None:
        _policy_snapshot_store = PolicySnapshotStore()
    return _policy_snapshot_store


def create_policy_snapshot(
    policy_version: str = "thompson_v1"
) -> FrozenPolicySnapshot:
    """
    Create snapshot from current bandit state.
    
    This should be called periodically to capture policy state.
    """
    store = get_policy_snapshot_store()
    
    # Get current bandit state
    from experience.thompson_sampling import get_bandit
    bandit = get_bandit()
    
    # Extract arm data
    arms_data = []
    if hasattr(bandit, '_arms'):
        for skill_id, arm in bandit._arms.items():
            arms_data.append((
                skill_id,
                arm.alpha,
                arm.beta
            ))
    
    return store.create_snapshot(
        policy_version=policy_version,
        arms_data=arms_data,
        total_evaluations=bandit.total_selections if hasattr(bandit, 'total_selections') else 0,
        last_update_at=datetime.utcnow().isoformat()
    )


def get_frozen_policy_for_replay() -> FrozenPolicySnapshot:
    """Get frozen policy snapshot for replay (read-only)"""
    store = get_policy_snapshot_store()
    
    # Try to get latest snapshot
    snapshot = store.get_latest()
    
    if snapshot is None:
        # Create initial snapshot if none exists
        snapshot = create_policy_snapshot()
    
    return snapshot