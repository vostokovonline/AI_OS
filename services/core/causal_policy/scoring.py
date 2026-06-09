"""
Causal Scoring Function — evaluates simulated outcomes by causal utility.

Scoring dimensions:
  1. BELIEF STABILITY — stable beliefs are better than volatile ones
     Formula: 1.0 - avg(confidence * (1-confidence) * 4)

  2. DRIFT REDUCTION — lower drift is better
     Formula: 1.0 - min(drift_score / threshold, 1.0)

  3. MOTIF REINFORCEMENT — strong motifs are better (they represent learned patterns)
     Formula: avg(motif_strength)

  4. STABILITY DELTA — positive change in stability is better
     Raw: stability_delta (from simulated outcome)

  5. CONFIDENCE — more confident predictions get more weight
     Raw: outcome.confidence

Composite score = W1 * stability + W2 * (1 - drift) + W3 * motif + W4 * delta + W5 * confidence
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .simulator import SimulatedOutcome


DEFAULT_WEIGHTS = {
    'belief_stability': 0.30,
    'drift_avoidance': 0.25,
    'motif_reinforcement': 0.15,
    'stability_delta': 0.20,
    'confidence': 0.10,
}

DRIFT_SEVERITY_THRESHOLD = 0.7  # drift above this is critical


@dataclass
class CausalScore:
    """
    Scored evaluation of a simulated outcome.

    Fields:
      outcome: the simulated outcome
      total_score: composite score (0.0-1.0)
      dimensions: individual dimension scores
      weights: weights used for scoring
      recommendation: "accept" | "reject" | "consider"
    """
    outcome: SimulatedOutcome
    total_score: float
    dimensions: Dict[str, float]
    weights: Dict[str, float]
    recommendation: str

    def to_dict(self) -> dict:
        return {
            'candidate_label': self.outcome.candidate.label,
            'candidate_goal_id': self.outcome.candidate.goal_id,
            'candidate_event_type': self.outcome.candidate.predicted_event_type,
            'total_score': self.total_score,
            'dimensions': dict(self.dimensions),
            'recommendation': self.recommendation,
            'stability_delta': self.outcome.stability_delta,
            'drift_violations': len(self.outcome.drift_report.violations),
            'drift_score': self.outcome.drift_report.overall_drift_score,
        }


class CausalScoringFunction:
    """
    Scores simulated outcomes across multiple causal dimensions.

    Pure function — no state mutation.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._validate_weights()

    def _validate_weights(self):
        """Verify weights sum to 1.0."""
        total = sum(self._weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.2f}"
            )

    def _score_belief_stability(self, outcome: SimulatedOutcome) -> float:
        beliefs = outcome.post_beliefs
        if not beliefs:
            return 1.0
        volatilities = []
        for b in beliefs.values():
            conf = b.get('confidence', 0.0)
            if conf > 0.0:
                volatilities.append(conf * (1.0 - conf) * 4)
        if not volatilities:
            return 1.0
        avg_vol = sum(volatilities) / len(volatilities)
        return round(1.0 - min(avg_vol, 1.0), 4)

    def _score_drift_avoidance(self, outcome: SimulatedOutcome) -> float:
        drift_score = outcome.drift_report.overall_drift_score
        return round(1.0 - min(drift_score / DRIFT_SEVERITY_THRESHOLD, 1.0), 4)

    def _score_motif_reinforcement(self, outcome: SimulatedOutcome) -> float:
        motifs = outcome.post_motifs
        if not motifs:
            return 0.5
        strengths = [m.get('strength', 0.0) for m in motifs.values()]
        return round(sum(strengths) / len(strengths), 4)

    def _score_stability_delta(self, outcome: SimulatedOutcome) -> float:
        delta = outcome.stability_delta
        # Normalize from [-1.0, 1.0] to [0.0, 1.0]
        return round((delta + 1.0) / 2.0, 4)

    def _score_confidence(self, outcome: SimulatedOutcome) -> float:
        return round(min(outcome.confidence, 1.0), 4)

    def score(self, outcome: SimulatedOutcome) -> CausalScore:
        """
        Score a single simulated outcome.

        Returns:
            CausalScore with composite + dimensions + recommendation
        """
        dimensions = {
            'belief_stability': self._score_belief_stability(outcome),
            'drift_avoidance': self._score_drift_avoidance(outcome),
            'motif_reinforcement': self._score_motif_reinforcement(outcome),
            'stability_delta': self._score_stability_delta(outcome),
            'confidence': self._score_confidence(outcome),
        }

        total = sum(
            dimensions[d] * self._weights[d]
            for d in dimensions
        )
        total = round(min(total, 1.0), 4)

        # Recommendation
        if total >= 0.7:
            recommendation = 'accept'
        elif total >= 0.4:
            recommendation = 'consider'
        else:
            recommendation = 'reject'

        return CausalScore(
            outcome=outcome,
            total_score=total,
            dimensions=dimensions,
            weights=dict(self._weights),
            recommendation=recommendation,
        )

    def score_batch(
        self,
        outcomes: List[SimulatedOutcome],
    ) -> List[CausalScore]:
        """Score multiple outcomes."""
        return [self.score(o) for o in outcomes]
