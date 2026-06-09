"""
Replay Engine - Deterministic replay of execution envelopes

CRITICAL: This replay engine does NOT:
- Execute real skills
- Write files or make network calls
- Mutate state
- Update live policy

It ONLY does:
- Envelope → Policy Selection (simulated)
- Compare(selected vs historical)
- Compute regret metrics
- Offline scoring

This is safe replay for evaluation, not runtime execution.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from experience.execution_envelope import ExecutionEnvelope, ExecutionEnvelopeStore
from experience.enforcement_config import get_enforcement_config
from experience.policy_snapshot import (
    FrozenPolicySnapshot,
    get_frozen_policy_for_replay,
    get_policy_snapshot_store,
    create_policy_snapshot
)


@dataclass
class ReplayResult:
    """Result of replay evaluation"""
    envelope_id: str
    historical_skill: str
    replay_skill: str
    agreement: bool
    regret: float
    replay_policy_version: str
    evaluated_at: str


class ReplayEngine:
    """
    Safe replay engine for offline policy evaluation.
    
    Usage:
        engine = ReplayEngine()
        
        # Get all envelopes
        envelopes = engine.get_envelopes(limit=100)
        
        # Replay one envelope (no side-effects)
        result = engine.replay(envelope)
        
        # Batch replay
        results = engine.replay_batch(envelopes[:10])
    """
    
    def __init__(
        self,
        envelope_store: Optional[ExecutionEnvelopeStore] = None
    ):
        self._envelope_store = envelope_store or ExecutionEnvelopeStore()
        self._config = get_enforcement_config()
        
        # Results storage (in-memory, for now)
        self._replay_results: List[ReplayResult] = []
    
    def get_envelopes(
        self,
        limit: int = 100,
        policy_version: Optional[str] = None
    ) -> List[ExecutionEnvelope]:
        """Get envelopes for replay"""
        envelopes = self._envelope_store.get_all()
        
        # Filter by policy version if specified
        if policy_version:
            envelopes = [e for e in envelopes if e.policy_version == policy_version]
        
        # Sort by created_at descending and limit
        envelopes = sorted(envelopes, key=lambda e: e.created_at, reverse=True)
        return envelopes[:limit]
    
    def _select_via_policy(self, envelope: ExecutionEnvelope) -> str:
        """
        Simulate policy selection for replay using FROZEN policy snapshot.
        
        CRITICAL: This uses READ-ONLY snapshot, NOT live bandit.
        This ensures deterministic replay without side-effects.
        """
        try:
            # Get frozen policy snapshot (read-only)
            snapshot = get_frozen_policy_for_replay()
            
            # Extract features from envelope for context-aware selection
            from experience.feature_extraction import FeatureExtractor
            features = FeatureExtractor.extract_from_envelope(envelope)
            
            # Use snapshot's selection method (deterministic)
            selected = snapshot.select_skill(
                goal_type=features.goal_type,
                domain=features.domain
            )
            return selected
            
        except Exception as e:
            # Fall back to envelope's original selection
            return envelope.selected_skill_id
    
    def replay(self, envelope: ExecutionEnvelope) -> ReplayResult:
        """
        Replay a single envelope.
        
        This is SAFE - it only:
        1. Reads envelope
        2. Queries current policy (without updating)
        3. Compares selections
        4. Computes regret
        
        No side-effects.
        """
        # Get historical selection
        historical_skill = envelope.selected_skill_id
        
        # Get current policy selection (simulation, no update)
        replay_skill = self._select_via_policy(envelope)
        
        # Check agreement
        agreement = (historical_skill == replay_skill)
        
        # Compute regret
        # Regret = 0 if policy agrees, 1 if disagrees
        # (This is simplified - real regret would use actual rewards)
        regret = 0.0 if agreement else 1.0
        
        result = ReplayResult(
            envelope_id=envelope.execution_id,
            historical_skill=historical_skill,
            replay_skill=replay_skill,
            agreement=agreement,
            regret=regret,
            replay_policy_version="thompson_v1",
            evaluated_at=datetime.utcnow().isoformat()
        )
        
        self._replay_results.append(result)
        
        return result
    
    def replay_batch(self, envelopes: List[ExecutionEnvelope]) -> List[ReplayResult]:
        """Replay multiple envelopes"""
        results = []
        for envelope in envelopes:
            try:
                result = self.replay(envelope)
                results.append(result)
            except Exception as e:
                # Log but continue
                print(f"Replay failed for {envelope.execution_id}: {e}")
                continue
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get replay statistics"""
        if not self._replay_results:
            return {
                "total_replays": 0,
                "agreement_rate": 0.0,
                "avg_regret": 0.0,
                "policy_version": "unknown"
            }
        
        agreements = sum(1 for r in self._replay_results if r.agreement)
        total = len(self._replay_results)
        avg_regret = sum(r.regret for r in self._replay_results) / total
        
        return {
            "total_replays": total,
            "agreements": agreements,
            "agreement_rate": agreements / total if total > 0 else 0.0,
            "avg_regret": avg_regret,
            "policy_version": self._replay_results[0].replay_policy_version if self._replay_results else "unknown"
        }
    
    def export_results(self, output_path: str):
        """Export replay results to JSON"""
        results_data = [
            {
                "envelope_id": r.envelope_id,
                "historical_skill": r.historical_skill,
                "replay_skill": r.replay_skill,
                "agreement": r.agreement,
                "regret": r.regret,
                "policy_version": r.replay_policy_version,
                "evaluated_at": r.evaluated_at
            }
            for r in self._replay_results
        ]
        
        with open(output_path, "w") as f:
            json.dump({
                "statistics": self.get_statistics(),
                "results": results_data
            }, f, indent=2)


# Global replay engine
_replay_engine: Optional[ReplayEngine] = None


def get_replay_engine() -> ReplayEngine:
    """Get or create global replay engine"""
    global _replay_engine
    if _replay_engine is None:
        _replay_engine = ReplayEngine()
    return _replay_engine