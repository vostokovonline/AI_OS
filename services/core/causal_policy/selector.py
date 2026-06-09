"""
Policy Selector — evaluates N candidate actions via simulation + scoring,
returns ranked decision.

Flow:
  1. Receive N candidate actions (proposed dispatches)
  2. Simulate each via CounterfactualSimulator
  3. Score each via CausalScoringFunction
  4. Rank by total_score DESC
  5. Return best candidate + ranked list + explanations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from .simulator import CounterfactualSimulator, CandidateAction, SimulatedOutcome
from .scoring import CausalScoringFunction, CausalScore

logger = logging.getLogger(__name__)


@dataclass
class PolicyDecision:
    """
    Result of policy selection.

    Fields:
      chosen: the selected candidate (None if all rejected)
      ranked: all candidates ranked by score DESC
      best_score: score of the chosen candidate
      threshold: minimum score for acceptance
      total_evaluated: number of candidates evaluated
    """
    chosen: Optional[CandidateAction]
    ranked: List[Dict[str, Any]]
    best_score: float
    threshold: float
    total_evaluated: int

    def to_dict(self) -> dict:
        return {
            'chosen': self.chosen.to_dict() if self.chosen else None,
            'ranked': self.ranked,
            'best_score': self.best_score,
            'threshold': self.threshold,
            'total_evaluated': self.total_evaluated,
        }


class PolicySelector:
    """
    Selects the best action from candidates by causal utility.

    This is the decision surface — it turns the causal graph from
    retrospective trace into predictive policy.

    Usage:
        selector = PolicySelector(simulator, scorer)
        decision = selector.evaluate(candidates, threshold=0.5)
        if decision.chosen:
            execute(decision.chosen)
    """

    def __init__(
        self,
        simulator: CounterfactualSimulator,
        scorer: CausalScoringFunction,
    ):
        self._simulator = simulator
        self._scorer = scorer

    def evaluate(
        self,
        candidates: List[CandidateAction],
        threshold: float = 0.5,
        return_all: bool = True,
    ) -> PolicyDecision:
        """
        Evaluate candidates and select the best one.

        Args:
            candidates: list of proposed actions
            threshold: minimum score for selection
            return_all: if True, include all ranked candidates in result

        Returns:
            PolicyDecision with chosen action + ranked list
        """
        if not candidates:
            return PolicyDecision(
                chosen=None,
                ranked=[],
                best_score=0.0,
                threshold=threshold,
                total_evaluated=0,
            )

        # Simulate all candidates
        outcomes = self._simulator.simulate_batch(candidates)

        # Score all outcomes
        scores = self._scorer.score_batch(outcomes)

        # Sort by score DESC
        scores.sort(key=lambda s: s.total_score, reverse=True)

        ranked = []
        for s in scores:
            ranked.append(s.to_dict())

        # Select best above threshold
        best = scores[0]
        chosen = best.outcome.candidate if best.total_score >= threshold else None

        result = PolicyDecision(
            chosen=chosen,
            ranked=ranked,
            best_score=best.total_score,
            threshold=threshold,
            total_evaluated=len(candidates),
        )

        if chosen:
            logger.info(
                f"policy_selected label={chosen.label} "
                f"score={best.total_score} "
                f"event={chosen.predicted_event_type}"
            )
        else:
            logger.info(
                f"policy_no_selection best_score={best.total_score} "
                f"below threshold={threshold}"
            )

        return result

    def evaluate_with_feedback(
        self,
        candidates: List[CandidateAction],
        threshold: float = 0.5,
    ) -> PolicyDecision:
        """
        Evaluate candidates with detailed per-dimension feedback.

        Same as evaluate() but adds dimension breakdown explanation.
        """
        decision = self.evaluate(candidates, threshold)

        # Add per-dimension explanation to ranked list
        for item in decision.ranked:
            score_obj = None
            for s in self._scorer.score_batch(
                self._simulator.simulate_batch(candidates)
            ):
                if s.outcome.candidate.label == item['candidate_label']:
                    score_obj = s
                    break
            if score_obj:
                item['dimension_breakdown'] = {
                    k: f"{v:.3f} (weight={self._scorer._weights.get(k, 0):.2f})"
                    for k, v in score_obj.dimensions.items()
                }
                dominant = max(score_obj.dimensions, key=score_obj.dimensions.get)
                item['dominant_dimension'] = dominant

        return decision
