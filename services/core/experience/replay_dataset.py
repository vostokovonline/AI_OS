"""
Replay Dataset Builder - Pipeline from execution to offline evaluation dataset

This is the core infrastructure for scalable replay and policy comparison.

Pipeline:
1. Execution happens → Envelope created
2. ReplayBundle created (frozen state)
3. Bundles accumulated in dataset
4. Offline benchmarks run against dataset
5. Policy comparison via replay
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class ReplayDatasetEntry:
    """Single entry in replay dataset"""
    entry_id: str
    bundle_id: str
    
    # Ground truth
    selected_skill: str
    success: bool
    latency_ms: int
    
    # Features
    feature_hash: str
    goal_type: str
    domain: str
    
    # For comparison
    candidate_skills: List[str]
    
    created_at: str
    
    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "bundle_id": self.bundle_id,
            "selected_skill": self.selected_skill,
            "success": self.success,
            "latency_ms": self.latency_ms,
            "feature_hash": self.feature_hash,
            "goal_type": self.goal_type,
            "domain": self.domain,
            "candidate_skills": self.candidate_skills,
            "created_at": self.created_at
        }
    
    @staticmethod
    def from_bundle(bundle: "ReplayBundle", success: bool = True, latency_ms: int = 0) -> "ReplayDatasetEntry":
        from uuid import uuid4
        
        return ReplayDatasetEntry(
            entry_id=uuid4().hex[:8],
            bundle_id=bundle.bundle_id,
            selected_skill=bundle.envelope.selected_skill_id,
            success=success,
            latency_ms=latency_ms,
            feature_hash=bundle.feature_hash,
            goal_type=bundle.envelope.goal_type,
            domain=bundle.envelope.domain,
            candidate_skills=bundle.candidate_set.candidates,
            created_at=bundle.created_at
        )


class ReplayDatasetBuilder:
    """
    Build replay datasets from execution bundles.
    
    Usage:
        builder = ReplayDatasetBuilder()
        
        # Add execution to dataset
        builder.add_execution(envelope, success=True, latency_ms=1500)
        
        # Get dataset
        dataset = builder.build_dataset(limit=100)
        
        # Run offline evaluation
        results = evaluator.evaluate(dataset)
    """
    
    def __init__(self, dataset_dir: str = "/app/replay_datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(exist_ok=True, parents=True)
        
        self._entries: List[ReplayDatasetEntry] = []
    
    def add_execution(
        self,
        envelope,
        success: bool = True,
        latency_ms: int = 0
    ):
        """Add execution to dataset"""
        from experience.feature_extraction import FeatureExtractor
        from experience.replay_bundle import create_replay_bundle, get_bundle_store
        
        # Create bundle
        bundle = create_replay_bundle(
            envelope=envelope,
            candidate_skills=envelope.candidate_skill_ids or []
        )
        
        # Create dataset entry
        entry = ReplayDatasetEntry.from_bundle(bundle, success, latency_ms)
        
        self._entries.append(entry)
        
        return entry
    
    def build_dataset(
        self,
        limit: int = 100,
        filter_goal_type: Optional[str] = None,
        filter_domain: Optional[str] = None
    ) -> List[ReplayDatasetEntry]:
        """Build dataset with optional filters"""
        entries = self._entries
        
        if filter_goal_type:
            entries = [e for e in entries if e.goal_type == filter_goal_type]
        
        if filter_domain:
            entries = [e for e in entries if e.domain == filter_domain]
        
        # Sort by created_at and limit
        entries = sorted(entries, key=lambda e: e.created_at, reverse=True)
        
        return entries[:limit]
    
    def export_dataset(self, name: str, limit: int = 100) -> str:
        """Export dataset to file"""
        entries = self.build_dataset(limit=limit)
        
        dataset = {
            "name": name,
            "created_at": datetime.utcnow().isoformat(),
            "total_entries": len(entries),
            "entries": [e.to_dict() for e in entries]
        }
        
        filename = self.dataset_dir / f"{name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, "w") as f:
            json.dump(dataset, f, indent=2)
        
        return str(filename)
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        if not self._entries:
            return {"total": 0}
        
        goal_types = {}
        domains = {}
        success_count = sum(1 for e in self._entries if e.success)
        
        for entry in self._entries:
            goal_types[entry.goal_type] = goal_types.get(entry.goal_type, 0) + 1
            domains[entry.domain] = domains.get(entry.domain, 0) + 1
        
        return {
            "total": len(self._entries),
            "success_rate": success_count / len(self._entries),
            "goal_types": goal_types,
            "domains": domains
        }


class OfflineEvaluator:
    """
    Run offline evaluation against replay dataset.
    
    Evaluates:
    - Policy agreement rate
    - Regret estimation
    - Capability performance
    """
    
    def __init__(self, dataset: List[ReplayDatasetEntry]):
        self.dataset = dataset
    
    def evaluate_policy(self, policy_snapshot) -> Dict:
        """Evaluate policy against dataset"""
        agreements = 0
        total = len(self.dataset)
        
        for entry in self.dataset:
            # Use policy to select skill for this entry's features
            selected = policy_snapshot.select_skill(
                goal_type=entry.goal_type,
                domain=entry.domain
            )
            
            if selected == entry.selected_skill:
                agreements += 1
        
        return {
            "total_entries": total,
            "agreements": agreements,
            "agreement_rate": agreements / total if total > 0 else 0.0,
            "regret": 1.0 - (agreements / total if total > 0 else 0.0)
        }
    
    def evaluate_capability(self, capability: str) -> Dict:
        """Evaluate specific capability performance"""
        capability_entries = [
            e for e in self.dataset 
            if capability in e.candidate_skills
        ]
        
        if not capability_entries:
            return {"capability": capability, "samples": 0}
        
        selected_count = sum(
            1 for e in capability_entries 
            if e.selected_skill == capability
        )
        
        success_count = sum(1 for e in capability_entries if e.success)
        
        return {
            "capability": capability,
            "samples": len(capability_entries),
            "selection_rate": selected_count / len(capability_entries),
            "success_rate": success_count / len(capability_entries) if capability_entries else 0
        }
    
    def evaluate_all_capabilities(self) -> List[Dict]:
        """Evaluate all capabilities in dataset"""
        capabilities = set()
        for entry in self.dataset:
            capabilities.update(entry.candidate_skills)
        
        return [
            self.evaluate_capability(cap)
            for cap in sorted(capabilities)
        ]


# Global builder
_dataset_builder: Optional[ReplayDatasetBuilder] = None


def get_dataset_builder() -> ReplayDatasetBuilder:
    global _dataset_builder
    if _dataset_builder is None:
        _dataset_builder = ReplayDatasetBuilder()
    return _dataset_builder