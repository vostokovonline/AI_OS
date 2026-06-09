"""
Drift Detection + Attenuation — monitors semantic divergence.

MIRRORS invariant engine / fencing token in execution_dynamics.

Drift violations detect:
  - BELIEF_DRIFT: confidence grows without new observations
  - MOTIF_DRIFT: motif strength self-reinforces without fresh evidence
  - ATTRACTOR_DRIFT: attractor weight diverges from observation frequency
  - EPOCH_DRIFT: epoch advances faster than observation rate

Drift attenuation applies decay to drifted beliefs/motifs,
returning the epistemic state toward observational grounding.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time


@dataclass
class DriftViolation:
    """
    A single drift violation.

    Fields:
      type: drift category (belief_drift, motif_drift, ...)
      target: belief/motif/attractor name
      severity: 0.0 (mild) to 1.0 (severe)
      detail: human-readable explanation
      attenuation: recommended decay amount
      timestamp: when detected
    """
    type: str
    target: str
    severity: float
    detail: str
    attenuation: float = 0.1
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            'type': self.type,
            'target': self.target,
            'severity': self.severity,
            'detail': self.detail,
            'attenuation': self.attenuation,
            'timestamp': self.timestamp or time.time(),
        }


@dataclass
class DriftReport:
    """
    Complete drift report for a check cycle.

    Fields:
      violations: list of DriftViolation
      overall_drift_score: 0.0 (none) to 1.0 (critical)
      recommendations: list of strings
    """
    violations: List[DriftViolation]
    overall_drift_score: float
    recommendations: List[str]

    def to_dict(self) -> dict:
        return {
            'violations': [v.to_dict() for v in self.violations],
            'overall_drift_score': self.overall_drift_score,
            'recommendations': self.recommendations,
            'passed': len(self.violations) == 0,
        }


class DriftDetector:
    """
    Detects and reports semantic drift in beliefs, motifs, and attractors.

    Drift detection rules:
      1. Belief confidence without observation: every 5 epoch advances
         without observation → attenuation 0.02 (capped at 0.3)
      2. Motif strength without observation: every 3 epoch advances
         without observation → attenuation 0.05
      3. Attractor weight vs observation ratio: if attractor weight > 2x
         observation frequency → drift
      4. Epoch/observation ratio: if epoch/observations > 3.0 → drift
    """

    def __init__(self):
        self._violation_history: List[dict] = []

    def evaluate(
        self,
        beliefs: dict,
        motifs: dict,
        attractors: dict,
        epoch: int,
        observation_count: int,
        journal_event_count: int,
    ) -> DriftReport:
        violations: List[DriftViolation] = []
        now = time.time()

        # Rule 1: Belief confidence without observation signal
        for name, b in beliefs.items():
            conf = b.get('confidence', 0.0)
            updated_epoch = b.get('updated_at_epoch', 0)
            epochs_since_update = epoch - updated_epoch
            if conf > 0.3 and epochs_since_update > 5 and observation_count > 0:
                attenuation = min(0.02 * (epochs_since_update // 5), 0.3)
                violations.append(DriftViolation(
                    type='belief_drift',
                    target=name,
                    severity=min(conf * (epochs_since_update / 10), 1.0),
                    detail=(
                        f"belief={name} confidence={conf} has not been updated "
                        f"for {epochs_since_update} epochs"
                    ),
                    attenuation=attenuation,
                    timestamp=now,
                ))

        # Rule 2: Motif drift
        for name, m in motifs.items():
            strength = m.get('strength', 0.0)
            updated_epoch = m.get('updated_at_epoch', 0)
            epochs_since_update = epoch - updated_epoch
            if strength > 0.3 and epochs_since_update > 3 and observation_count > 0:
                attenuation = min(0.05 * (epochs_since_update // 3), 0.4)
                violations.append(DriftViolation(
                    type='motif_drift',
                    target=name,
                    severity=min(strength * (epochs_since_update / 8), 1.0),
                    detail=(
                        f"motif={name} strength={strength} has not been updated "
                        f"for {epochs_since_update} epochs"
                    ),
                    attenuation=attenuation,
                    timestamp=now,
                ))

        # Rule 3: Attractor weight vs observation frequency
        obs_freq = max(observation_count, 1)
        for att_id, a in attractors.items():
            weight = a.get('weight', 0.0)
            if weight > 0.0 and weight > 2.0 * (obs_freq / max(epoch, 1)):
                violations.append(DriftViolation(
                    type='attractor_drift',
                    target=att_id,
                    severity=min(abs(weight - obs_freq / max(epoch, 1)), 1.0),
                    detail=(
                        f"attractor={att_id} weight={weight} exceeds "
                        f"observation frequency={obs_freq}/{epoch}"
                    ),
                    attenuation=0.1,
                    timestamp=now,
                ))

        # Rule 4: Epoch/observation ratio
        if observation_count > 0 and epoch > 3 * observation_count:
            violations.append(DriftViolation(
                type='epoch_drift',
                target='system',
                severity=min((epoch - 3 * observation_count) / epoch, 1.0),
                detail=(
                    f"epoch advancement ({epoch}) outpaces observations "
                    f"({observation_count}), ratio={epoch/observation_count:.1f}"
                ),
                attenuation=0.0,
                timestamp=now,
            ))

        # Score
        if not violations:
            score = 0.0
        else:
            score = min(
                sum(v.severity for v in violations) / len(violations),
                1.0,
            )

        # Recommendations
        recommendations = []
        if any(v.type == 'belief_drift' for v in violations):
            recommendations.append(
                "Collect observations to reground drifted beliefs"
            )
        if any(v.type == 'motif_drift' for v in violations):
            recommendations.append(
                "Validate motif strength against fresh observations"
            )
        if any(v.type == 'attractor_drift' for v in violations):
            recommendations.append(
                "Recalibrate attractor weights to match observation frequency"
            )
        if any(v.type == 'epoch_drift' for v in violations):
            recommendations.append(
                "Slow epoch advancement rate; epoch is advancing "
                "without corresponding observations"
            )

        report = DriftReport(
            violations=violations,
            overall_drift_score=score,
            recommendations=recommendations,
        )

        # Archive
        self._violation_history.append(report.to_dict())
        self._prune_history()

        return report

    def _prune_history(self, max_entries: int = 1000):
        while len(self._violation_history) > max_entries:
            self._violation_history.pop(0)
