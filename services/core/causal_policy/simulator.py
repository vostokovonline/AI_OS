"""
Counterfactual Simulator — predicts causal outcome of an action BEFORE execution.

Core operation:
  1. Snapshot current epistemic state (beliefs, motifs, attractors, epoch)
  2. Apply predicted deltas from the proposed action
  3. Run drift detection on the SIMULATED state
  4. Return predicted causal graph + outcome metrics

This is PURE — no state mutation, no side effects.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class CandidateAction:
    """
    A proposed action to be simulated before execution.

    Fields:
      label: human-readable name (e.g. "retry_goal_def")
      goal_id: target goal
      predicted_event_type: COMPLETED | FAILED | PREEMPTED | CANCELLED | RETRIED | ABANDONED
      predicted_success: expected outcome
      confidence: how confident we are in this prediction (0.0-1.0)
      context: additional metadata
    """
    label: str
    goal_id: str
    predicted_event_type: str
    predicted_success: Optional[bool] = None
    confidence: float = 0.5
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'goal_id': self.goal_id,
            'predicted_event_type': self.predicted_event_type,
            'predicted_success': self.predicted_success,
            'confidence': self.confidence,
            'context': dict(self.context),
        }


@dataclass
class SimulatedOutcome:
    """
    Predicted outcome of a counterfactual simulation.

    Fields:
      candidate: the action that was simulated
      post_beliefs: predicted belief state after action
      post_motifs: predicted motif state after action
      post_attractors: predicted attractor state after action
      post_epoch: predicted epoch
      drift_report: drift detection result on simulated state
      predicted_edges: number of causality edges this action would create
      stability_delta: predicted change in overall belief stability (-1.0 to 1.0)
      confidence: confidence in this simulation
    """
    candidate: CandidateAction
    post_beliefs: Dict[str, Dict[str, float]]
    post_motifs: Dict[str, Dict[str, float]]
    post_attractors: Dict[str, Dict[str, float]]
    post_epoch: int
    drift_report: Any  # DriftReport
    predicted_edges: int = 1
    stability_delta: float = 0.0
    confidence: float = 1.0

    def to_dict(self) -> dict:
        return {
            'candidate': self.candidate.to_dict(),
            'post_beliefs': {k: dict(v) for k, v in self.post_beliefs.items()},
            'post_motifs': {k: dict(v) for k, v in self.post_motifs.items()},
            'post_attractors': {k: dict(v) for k, v in self.post_attractors.items()},
            'post_epoch': self.post_epoch,
            'drift_report': self.drift_report.to_dict() if hasattr(self.drift_report, 'to_dict') else {},
            'predicted_edges': self.predicted_edges,
            'stability_delta': self.stability_delta,
            'confidence': self.confidence,
        }


# Default belief/motif deltas per event type (mirrors DEFAULT_FRAMES in propagation.py)
PREDICTED_DELTAS = {
    'COMPLETED': {
        'belief_delta': {'execution_effectiveness': 0.05},
        'motif_delta': {},
        'confidence_delta': 0.02,
    },
    'FAILED': {
        'belief_delta': {'failure_rate': 0.08},
        'motif_delta': {},
        'confidence_delta': -0.05,
    },
    'PREEMPTED': {
        'belief_delta': {},
        'motif_delta': {'preemption_pattern': 0.1},
        'confidence_delta': -0.03,
    },
    'CANCELLED': {
        'belief_delta': {'execution_uncertainty': 0.06},
        'motif_delta': {},
        'confidence_delta': -0.04,
    },
    'RETRIED': {
        'belief_delta': {},
        'motif_delta': {'persistence_pattern': 0.07},
        'confidence_delta': 0.0,
    },
    'ABANDONED': {
        'belief_delta': {'failure_rate': 0.15, 'skill_gap': 0.1},
        'motif_delta': {},
        'confidence_delta': -0.1,
    },
    'DISPATCHED': {
        'belief_delta': {},
        'motif_delta': {},
        'confidence_delta': 0.0,
    },
}


class CounterfactualSimulator:
    """
    Pure counterfactual simulator — predicts epistemic state after
    a proposed action WITHOUT mutating the real kernel.

    Process:
      1. clone_state() — deep-copy current beliefs, motifs, attractors
      2. simulate() — apply predicted deltas for a candidate action
      3. run drift detection on simulated state
      4. return SimulatedOutcome
    """

    def __init__(self, epistemic_kernel):
        self._epistemic_kernel = epistemic_kernel
        self._deltas = PREDICTED_DELTAS

    def clone_state(self) -> Tuple[Dict, Dict, Dict, int]:
        """
        Snapshot current epistemic state without mutation.

        Returns:
            (beliefs, motifs, attractors, epoch)
        """
        return (
            copy.deepcopy(self._epistemic_kernel._beliefs),
            copy.deepcopy(self._epistemic_kernel._motifs),
            copy.deepcopy(self._epistemic_kernel._attractors),
            self._epistemic_kernel.epoch.current,
        )

    def _apply_delta(self, state, delta: Dict[str, float], field: str):
        """Apply a delta dict to a state dict's sub-field."""
        for key, value in delta.items():
            current = state.get(key, {}).get(field, 0.0)
            new_val = max(0.0, min(1.0, current + value))
            if key not in state:
                state[key] = {}
            state[key][field] = round(new_val, 4)
            state[key]['simulated'] = True

    def simulate(
        self,
        candidate: CandidateAction,
    ) -> SimulatedOutcome:
        """
        Simulate a candidate action on a cloned epistemic state.

        Args:
            candidate: the proposed action

        Returns:
            SimulatedOutcome with predicted state and drift
        """
        # Clone current state
        sim_beliefs, sim_motifs, sim_attractors, sim_epoch = self.clone_state()

        # Get deltas for this event type
        deltas = self._deltas.get(candidate.predicted_event_type, {
            'belief_delta': {},
            'motif_delta': {},
            'confidence_delta': 0.0,
        })

        # Apply belief deltas
        self._apply_delta(sim_beliefs, deltas['belief_delta'], 'confidence')

        # Apply motif deltas
        self._apply_delta(sim_motifs, deltas['motif_delta'], 'strength')

        # Advance simulated epoch
        sim_epoch += 1

        # Run drift detection on simulated state
        drift = self._epistemic_kernel.drift.evaluate(
            beliefs=sim_beliefs,
            motifs=sim_motifs,
            attractors=sim_attractors,
            epoch=sim_epoch,
            observation_count=self._epistemic_kernel._observation_count + 1,
            journal_event_count=len(self._epistemic_kernel.journal._events) + 1,
        )

        # Compute stability delta
        pre_beliefs, _, _, pre_epoch = self.clone_state()
        pre_stability = self._compute_stability(pre_beliefs)
        post_stability = self._compute_stability(sim_beliefs)
        stability_delta = round(post_stability - pre_stability, 4)

        return SimulatedOutcome(
            candidate=candidate,
            post_beliefs=sim_beliefs,
            post_motifs=sim_motifs,
            post_attractors=sim_attractors,
            post_epoch=sim_epoch,
            drift_report=drift,
            predicted_edges=1,
            stability_delta=stability_delta,
            confidence=candidate.confidence,
        )

    def _compute_stability(self, beliefs: Dict) -> float:
        """
        Compute overall belief stability from a belief dict.

        Stability = 1.0 - average(belief volatility)
        Volatility = confidence * (1 - confidence) — max at 0.5
        """
        if not beliefs:
            return 1.0
        volatilities = []
        for name, b in beliefs.items():
            conf = b.get('confidence', 0.0)
            if conf > 0.0:
                volatility = conf * (1.0 - conf) * 4  # normalize to [0, 1]
                volatilities.append(volatility)
        if not volatilities:
            return 1.0
        avg_volatility = sum(volatilities) / len(volatilities)
        return round(1.0 - avg_volatility, 4)

    def simulate_batch(
        self,
        candidates: List[CandidateAction],
    ) -> List[SimulatedOutcome]:
        """Simulate multiple candidates (each from the SAME starting state)."""
        return [self.simulate(c) for c in candidates]
