"""
Promotion Gates - Safe policy promotion logic

CRITICAL: This module ensures NO auto-promotion by online reward.
Policy promotion requires:
- Shadow evaluation
- Offline replay
- Regret thresholds
- Stability window

NEVER auto-promote based on live execution rewards.
"""
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class PromotionStatus(str, Enum):
    """Promotion evaluation status"""
    PENDING = "pending"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    REJECTED = "rejected"
    STABLE_WAIT = "stable_wait"


@dataclass
class PromotionGateResult:
    """Result of promotion gate evaluation"""
    can_promote: bool
    status: PromotionStatus
    shadow_agreement_rate: float = 0.0
    replay_regret: float = 1.0
    stability_score: float = 0.0
    evaluation_samples: int = 0
    reason: str = ""
    evaluated_at: str = ""


class PromotionGates:
    """
    Safe promotion gates for policy updates.
    
    Usage:
        gates = PromotionGates()
        
        # Before promoting policy
        result = gates.evaluate_promotion(
            current_policy_version="v1",
            candidate_policy_version="v2"
        )
        
        if result.can_promote:
            # Safe to promote
        else:
            # Do not promote - gates closed
    """
    
    def __init__(
        self,
        min_samples: int = 10,
        min_stability_window_hours: int = 24,
        max_regret_threshold: float = 0.3,
        min_agreement_rate: float = 0.7
    ):
        self._min_samples = min_samples
        self._min_stability_window = timedelta(hours=min_stability_window_hours)
        self._max_regret_threshold = max_regret_threshold
        self._min_agreement_rate = min_agreement_rate
        
        # Track promotion history
        self._promotion_history: list = []
    
    def evaluate_promotion(
        self,
        current_policy_version: str,
        candidate_policy_version: str,
        shadow_results: Optional[Dict] = None,
        replay_results: Optional[Dict] = None
    ) -> PromotionGateResult:
        """
        Evaluate if policy promotion is safe.
        
        ALL gates must pass:
        1. Minimum samples met
        2. Shadow agreement rate >= threshold
        3. Replay regret <= threshold
        4. Stability window passed
        
        NEVER auto-promote by online reward alone.
        """
        now = datetime.utcnow()
        
        # Gate 1: Check minimum samples
        if shadow_results and "sample_count" in shadow_results:
            sample_count = shadow_results["sample_count"]
        else:
            sample_count = 0
        
        if sample_count < self._min_samples:
            return PromotionGateResult(
                can_promote=False,
                status=PromotionStatus.PENDING,
                evaluation_samples=sample_count,
                reason=f"Insufficient samples: {sample_count} < {self._min_samples}",
                evaluated_at=now.isoformat()
            )
        
        # Gate 2: Shadow evaluation agreement rate
        agreement_rate = 0.0
        if shadow_results and "agreement_rate" in shadow_results:
            agreement_rate = shadow_results["agreement_rate"]
        
        if agreement_rate < self._min_agreement_rate:
            return PromotionGateResult(
                can_promote=False,
                status=PromotionStatus.REJECTED,
                shadow_agreement_rate=agreement_rate,
                evaluation_samples=sample_count,
                reason=f"Shadow agreement too low: {agreement_rate:.2f} < {self._min_agreement_rate}",
                evaluated_at=now.isoformat()
            )
        
        # Gate 3: Replay regret threshold
        replay_regret = 1.0  # Default to worst
        if replay_results and "avg_regret" in replay_results:
            replay_regret = replay_results["avg_regret"]
        
        if replay_regret > self._max_regret_threshold:
            return PromotionGateResult(
                can_promote=False,
                status=PromotionStatus.REJECTED,
                shadow_agreement_rate=agreement_rate,
                replay_regret=replay_regret,
                evaluation_samples=sample_count,
                reason=f"Regret too high: {replay_regret:.2f} > {self._max_regret_threshold}",
                evaluated_at=now.isoformat()
            )
        
        # Gate 4: Stability window (check recent promotions)
        if self._promotion_history:
            last_promo = self._promotion_history[-1]
            last_time = datetime.fromisoformat(last_promo["promoted_at"])
            if now - last_time < self._min_stability_window:
                return PromotionGateResult(
                    can_promote=False,
                    status=PromotionStatus.STABLE_WAIT,
                    shadow_agreement_rate=agreement_rate,
                    replay_regret=replay_regret,
                    evaluation_samples=sample_count,
                    reason=f"Stability window not passed: {(now - last_time).total_seconds() / 3600:.1f}h < {self._min_stability_window.total_seconds() / 3600}h",
                    evaluated_at=now.isoformat()
                )
        
        # All gates passed
        return PromotionGateResult(
            can_promote=True,
            status=PromotionStatus.APPROVED,
            shadow_agreement_rate=agreement_rate,
            replay_regret=replay_regret,
            stability_score=1.0,
            evaluation_samples=sample_count,
            reason="All gates passed",
            evaluated_at=now.isoformat()
        )
    
    def record_promotion(
        self,
        from_version: str,
        to_version: str,
        gate_result: PromotionGateResult
    ):
        """Record promotion attempt"""
        self._promotion_history.append({
            "from_version": from_version,
            "to_version": to_version,
            "promoted_at": gate_result.evaluated_at,
            "can_promote": gate_result.can_promote,
            "reason": gate_result.reason
        })
    
    def get_promotion_history(self) -> list:
        """Get promotion history"""
        return self._promotion_history
    
    def reset_history(self):
        """Reset promotion history (for testing)"""
        self._promotion_history = []


# Global promotion gates
_promotion_gates: Optional[PromotionGates] = None


def get_promotion_gates() -> PromotionGates:
    """Get or create global promotion gates"""
    global _promotion_gates
    if _promotion_gates is None:
        _promotion_gates = PromotionGates()
    return _promotion_gates


def safe_promote(
    current_version: str,
    candidate_version: str,
    shadow_results: Optional[Dict] = None,
    replay_results: Optional[Dict] = None
) -> tuple[bool, str]:
    """
    Convenience function for safe promotion.
    
    Returns:
        (can_promote, reason)
    """
    gates = get_promotion_gates()
    result = gates.evaluate_promotion(
        current_policy_version=current_version,
        candidate_policy_version=candidate_version,
        shadow_results=shadow_results,
        replay_results=replay_results
    )
    
    if result.can_promote:
        gates.record_promotion(current_version, candidate_version, result)
    
    return result.can_promote, result.reason