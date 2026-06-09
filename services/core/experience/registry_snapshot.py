"""
Registry Snapshot - Versioned snapshots of skill registry for replay

Ensures replay can reconstruct execution context even if registry changes.
Each snapshot contains:
- registry_version
- skill_manifest_hash (for each skill)
- executor_hash
- timestamp
"""
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillSnapshot:
    """Immutable snapshot of a single skill"""
    skill_id: str
    version: str
    manifest_hash: str  # Hash of skill manifest (capabilities, inputs, outputs)
    registered_at: str


@dataclass(frozen=True)
class RegistrySnapshot:
    """Immutable snapshot of entire skill registry"""
    snapshot_id: str
    registry_version: str
    created_at: str
    skills: List[SkillSnapshot]
    total_count: int
    
    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "registry_version": self.registry_version,
            "created_at": self.created_at,
            "skills": [
                {"skill_id": s.skill_id, "version": s.version, "manifest_hash": s.manifest_hash}
                for s in self.skills
            ],
            "total_count": self.total_count
        }
    
    @staticmethod
    def from_dict(data: dict) -> "RegistrySnapshot":
        return RegistrySnapshot(
            snapshot_id=data["snapshot_id"],
            registry_version=data["registry_version"],
            created_at=data["created_at"],
            skills=[SkillSnapshot(**s) for s in data["skills"]],
            total_count=data["total_count"]
        )


class RegistrySnapshotStore:
    """Store for registry snapshots"""
    
    def __init__(self, store_dir: str = "/app/registry_snapshots"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._snapshots: Dict[str, RegistrySnapshot] = {}
        self._load_existing()
    
    def _load_existing(self):
        """Load existing snapshots"""
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    snapshot = RegistrySnapshot.from_dict(data)
                    self._snapshots[snapshot.snapshot_id] = snapshot
            except:
                continue
    
    def _compute_manifest_hash(self, skill_id: str, capabilities: List[str], version: str) -> str:
        """Compute deterministic hash of skill manifest"""
        manifest_data = {
            "skill_id": skill_id,
            "capabilities": sorted(capabilities) if capabilities else [],
            "version": version
        }
        canonical = json.dumps(manifest_data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def create_snapshot(
        self,
        registry_version: str,
        skills: List[tuple]  # List of (skill_id, version, capabilities)
    ) -> RegistrySnapshot:
        """Create a new snapshot of current registry state"""
        from uuid import uuid4
        
        skill_snapshots = []
        for skill_id, version, capabilities in skills:
            manifest_hash = self._compute_manifest_hash(skill_id, capabilities, version)
            skill_snapshots.append(SkillSnapshot(
                skill_id=skill_id,
                version=version,
                manifest_hash=manifest_hash,
                registered_at=datetime.utcnow().isoformat()
            ))
        
        snapshot = RegistrySnapshot(
            snapshot_id=uuid4().hex[:8],
            registry_version=registry_version,
            created_at=datetime.utcnow().isoformat(),
            skills=skill_snapshots,
            total_count=len(skill_snapshots)
        )
        
        # Save to disk
        filename = self.store_dir / f"{snapshot.snapshot_id}.json"
        with open(filename, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        self._snapshots[snapshot.snapshot_id] = snapshot
        
        return snapshot
    
    def get_latest(self) -> Optional[RegistrySnapshot]:
        """Get most recent snapshot"""
        if not self._snapshots:
            return None
        
        return max(self._snapshots.values(), key=lambda s: s.created_at)
    
    def get(self, snapshot_id: str) -> Optional[RegistrySnapshot]:
        """Get snapshot by ID"""
        return self._snapshots.get(snapshot_id)
    
    def get_all(self) -> List[RegistrySnapshot]:
        """Get all snapshots"""
        return sorted(self._snapshots.values(), key=lambda s: s.created_at)


# Global store
_registry_snapshot_store: Optional[RegistrySnapshotStore] = None


def get_registry_snapshot_store() -> RegistrySnapshotStore:
    """Get or create global registry snapshot store"""
    global _registry_snapshot_store
    if _registry_snapshot_store is None:
        _registry_snapshot_store = RegistrySnapshotStore()
    return _registry_snapshot_store


def create_registry_snapshot(
    registry_version: str = "v1",
    skills: List[tuple] = None
) -> RegistrySnapshot:
    """Create snapshot of current registry state"""
    store = get_registry_snapshot_store()
    
    if skills is None:
        # Get current skills from registry
        from experience.skill_registry import get_skill_registry
        registry = get_skill_registry()
        skills = [(s.canonical_id, s.version, s.capabilities) for s in registry.list_skills()]
    
    return store.create_snapshot(registry_version, skills)