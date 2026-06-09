"""
Trajectory Scoring Engine — evaluates complete action sequences.

Two modes:
  1. Terminal-only: Score(T) = Quality(final_state)
  2. Path-weighted (default): Score(T) = Σ w_t * state_quality(t) + terminal_bonus - penalties

Penalties:
  - Drift accumulation: penalize trajectories that drift from causal invariants
  - Action entropy: penalize chaotic/noisy policy paths
  - Confidence decay: weight reduces with depth (built into path weighting)
  - Reality anchoring: penalize divergence from actual bridge history
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .tree import PolicyNode
from .uncertainty import UncertaintyState


@dataclass
class TrajectoryScore:
    """
    Scored trajectory evaluation.

    Fields:
      trajectory: list of PolicyNode from root to leaf
      total_score: weighted path score
      terminal_quality: quality of final state
      path_score: sum of weighted intermediate scores
      penalties: all applied penalties (drift, entropy, reality)
      confidence: confidence in this trajectory (decayed by depth)
    """
    trajectory: List[PolicyNode]
    total_score: float
    terminal_quality: float
    path_score: float
    penalties: Dict[str, float]
    confidence: float

    def to_dict(self) -> dict:
        return {
            'total_score': self.total_score,
            'terminal_quality': self.terminal_quality,
            'path_score': self.path_score,
            'penalties': dict(self.penalties),
            'confidence': self.confidence,
            'actions': [
                n.action.label if n.action else 'ROOT'
                for n in self.trajectory[1:]
            ],
            'depth': len(self.trajectory) - 1,
        }


class TrajectoryScoringEngine:
    """
    Evaluates complete action sequences (trajectories).

    Formula:
      Score(T) = Σ w_t * state_quality(t) + terminal_bonus - drift_penalty - entropy_penalty - reality_penalty

    where:
      w_t = confidence_decay(t)  (from UncertaintyDecayModel)
      state_quality(t) = belief_stability - uncertainty - drift
      terminal_bonus = extra weight on final state quality
    """

    def __init__(
        self,
        uncertainty_model: 'UncertaintyDecayModel',
        terminal_weight: float = 1.5,
        drift_penalty_weight: float = 0.3,
        entropy_penalty_weight: float = 0.1,
        reality_anchor_weight: float = 0.2,
    ):
        self._uncertainty = uncertainty_model
        self._terminal_weight = terminal_weight
        self._drift_penalty_weight = drift_penalty_weight
        self._entropy_penalty_weight = entropy_penalty_weight
        self._reality_anchor_weight = reality_anchor_weight

    @staticmethod
    def compute_state_quality(state_snapshot: Dict[str, Any]) -> float:
        """
        Compute quality of a single simulated state.

        Quality = average_belief_stability - normalized_uncertainty - drift

        Returns:
            quality score (0.0-1.0)
        """
        beliefs = state_snapshot.get('beliefs', {})
        drift = state_snapshot.get('drift_estimate', 0.0)

        if not beliefs:
            stability = 1.0
        else:
            volatilities = []
            for b in beliefs.values():
                conf = b.get('confidence', 0.0)
                if conf > 0.0:
                    volatilities.append(conf * (1.0 - conf) * 4)
            stability = 1.0 - (sum(volatilities) / len(volatilities)) if volatilities else 1.0

        quality = stability - drift
        return max(0.0, min(1.0, quality))

    def score_trajectory(
        self,
        trajectory: List[PolicyNode],
    ) -> TrajectoryScore:
        """
        Score a complete trajectory.

        Args:
            trajectory: list of PolicyNode from root to leaf

        Returns:
            TrajectoryScore
        """
        if len(trajectory) < 1:
            return TrajectoryScore(
                trajectory=[], total_score=0.0, terminal_quality=0.0,
                path_score=0.0, penalties={}, confidence=0.0,
            )

        penalties: Dict[str, float] = {}
        path_score = 0.0

        for i, node in enumerate(trajectory):
            if i == 0:
                continue  # skip root (current real state)

            confidence = self._uncertainty.confidence_at_depth(i)
            state_quality = self.compute_state_quality(node.state_snapshot)

            # w_t * state_quality(t)
            path_score += confidence * state_quality

        # Terminal bonus: extra weight on final state
        terminal_node = trajectory[-1]
        terminal_quality = self.compute_state_quality(terminal_node.state_snapshot)
        terminal_score = terminal_quality * self._terminal_weight

        # Drift penalty: penalize trajectories with high cumulative drift
        total_drift = sum(
            n.drift_estimate for n in trajectory[1:]  # skip root
        )
        avg_drift = total_drift / max(len(trajectory) - 1, 1)
        drift_penalty = avg_drift * self._drift_penalty_weight
        penalties['drift'] = round(drift_penalty, 4)

        # Entropy penalty: penalize chaotic action sequences
        action_types = [
            n.action.predicted_event_type if n.action else None
            for n in trajectory[1:]
        ]
        unique_types = len(set(filter(None, action_types)))
        if len(action_types) > 1:
            entropy = 1.0 - (unique_types / len(action_types))
        else:
            entropy = 0.0
        entropy_penalty = entropy * self._entropy_penalty_weight
        penalties['entropy'] = round(entropy_penalty, 4)

        # Reality anchoring penalty: penalize divergence from real drift patterns
        # Higher drift at shallow depth = less realistic trajectory
        if len(trajectory) > 1:
            first_simulated = trajectory[1]
            reality_penalty = first_simulated.drift_estimate * self._reality_anchor_weight
        else:
            reality_penalty = 0.0
        penalties['reality_anchor'] = round(reality_penalty, 4)

        # Final score
        total_score = (
            path_score
            + terminal_score
            - drift_penalty
            - entropy_penalty
            - reality_penalty
        )
        total_score = max(0.0, min(10.0, total_score))

        # Confidence in trajectory = confidence at deepest node
        trajectory_confidence = self._uncertainty.confidence_at_depth(
            len(trajectory) - 1
        )

        return TrajectoryScore(
            trajectory=trajectory,
            total_score=round(total_score, 4),
            terminal_quality=round(terminal_quality, 4),
            path_score=round(path_score, 4),
            penalties=penalties,
            confidence=trajectory_confidence,
        )

    def score_all_trajectories(
        self,
        trajectories: List[List[PolicyNode]],
    ) -> List[TrajectoryScore]:
        """Score all trajectories and sort by total_score DESC."""
        scores = [self.score_trajectory(t) for t in trajectories]
        scores.sort(key=lambda s: s.total_score, reverse=True)
        return scores
