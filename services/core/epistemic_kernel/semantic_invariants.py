"""
Semantic Invariant Engine — pure, deterministic, replayable verification
of epistemic state structural correctness.

MIRRORS InvariantEngine in execution_dynamics/invariants.py.

Invariants verify IMPOSSIBILITY of forbidden states, not success.
A passing invariant means "this corruption CANNOT exist in the state."
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any


class Severity(Enum):
    FATAL = auto()    # State is corrupt, requires re-grounding
    ERROR = auto()    # State is degraded, needs attention
    WARNING = auto()  # State is suboptimal, should investigate


@dataclass
class SemanticInvariantViolation:
    """
    A single invariant violation.

    Fields mirror InvariantViolation in execution_dynamics.
    """
    name: str
    detail: str
    severity: Severity
    category: str = ""
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'detail': self.detail,
            'severity': self.severity.name,
            'category': self.category,
            'context': dict(self.context),
        }


@dataclass
class SemanticInvariantReport:
    """
    Report from a semantic invariant check cycle.
    """
    invariants_checked: int
    invariants_passed: int
    violations: List[SemanticInvariantViolation]
    fatal: bool = False

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> dict:
        return {
            'invariants_checked': self.invariants_checked,
            'invariants_passed': self.invariants_passed,
            'passed': self.passed,
            'fatal': self.fatal,
            'violations': [v.to_dict() for v in self.violations],
        }


class SemanticInvariantViolationError(Exception):
    """Raised when a FATAL or ERROR invariant is violated."""
    def __init__(self, report: SemanticInvariantReport):
        self.report = report
        super().__init__(
            f"Semantic invariant violation: "
            f"{len(report.violations)} violations, "
            f"fatal={report.fatal}"
        )


class SemanticInvariantEngine:
    """
    Pure semantic invariant engine — no epistemic state mutation during checks.

    Categories (mirroring execution invariants):
      BELIEF  — structural validity of belief state
      MOTIF   — structural validity of motif state
      ATTRACTOR — structural validity of attractor state
      JOURNAL — structural validity of the semantic journal
      EPOCH   — epoch structural validity
      GROUNDING — grounding checkpoint validity
    """

    def __init__(self, kernel):
        self._kernel = kernel

    # ------------------------------------------------------------------
    # Invariant checks (one method per invariant)
    # ------------------------------------------------------------------

    def _check_belief_confidence_in_bounds(self) -> Optional[SemanticInvariantViolation]:
        """BELIEF-001: All belief confidences must be in [0.0, 1.0]."""
        for name, b in self._kernel._beliefs.items():
            conf = b.get('confidence', -1)
            if not (0.0 <= conf <= 1.0):
                return SemanticInvariantViolation(
                    name='belief_confidence_in_bounds',
                    severity=Severity.FATAL,
                    category='BELIEF',
                    detail=f"belief={name} confidence={conf} outside [0.0, 1.0]",
                    context={'belief_name': name, 'confidence': conf},
                )
        return None

    def _check_motif_strength_in_bounds(self) -> Optional[SemanticInvariantViolation]:
        """MOTIF-001: All motif strengths must be in [0.0, 1.0]."""
        for name, m in self._kernel._motifs.items():
            strength = m.get('strength', -1)
            if not (0.0 <= strength <= 1.0):
                return SemanticInvariantViolation(
                    name='motif_strength_in_bounds',
                    severity=Severity.FATAL,
                    category='MOTIF',
                    detail=f"motif={name} strength={strength} outside [0.0, 1.0]",
                    context={'motif_name': name, 'strength': strength},
                )
        return None

    def _check_attractor_weight_in_bounds(self) -> Optional[SemanticInvariantViolation]:
        """ATTRACTOR-001: All attractor weights must be in [0.0, 1.0]."""
        for att_id, a in self._kernel._attractors.items():
            weight = a.get('weight', -1)
            if not (0.0 <= weight <= 1.0):
                return SemanticInvariantViolation(
                    name='attractor_weight_in_bounds',
                    severity=Severity.FATAL,
                    category='ATTRACTOR',
                    detail=f"attractor={att_id} weight={weight} outside [0.0, 1.0]",
                    context={'attractor_id': att_id, 'weight': weight},
                )
        return None

    def _check_observation_precedes_belief(self) -> Optional[SemanticInvariantViolation]:
        """JOURNAL-001: Every belief update must have a preceding observation."""
        if not self._kernel._observation_count and self._kernel._beliefs:
            # No observations but beliefs exist — possible if beliefs
            # were set directly. This is a WARNING (acceptable for priors).
            return SemanticInvariantViolation(
                name='observation_precedes_belief',
                severity=Severity.WARNING,
                category='JOURNAL',
                detail=(
                    f"{len(self._kernel._beliefs)} beliefs exist "
                    f"with 0 observations"
                ),
                context={
                    'belief_count': len(self._kernel._beliefs),
                    'observation_count': self._kernel._observation_count,
                },
            )
        return None

    def _check_journal_event_chain(self) -> Optional[SemanticInvariantViolation]:
        """JOURNAL-002: Journal event chain must be contiguous."""
        events = self._kernel.journal._events
        if len(events) < 2:
            return None
        for i in range(1, len(events)):
            if events[i].prev_event_id != events[i - 1].event_id:
                return SemanticInvariantViolation(
                    name='journal_event_chain',
                    severity=Severity.FATAL,
                    category='JOURNAL',
                    detail=(
                        f"Event chain broken at index {i}: "
                        f"expected prev={events[i-1].event_id}, "
                        f"got {events[i].prev_event_id}"
                    ),
                    context={
                        'event_index': i,
                        'expected_prev': events[i - 1].event_id,
                        'actual_prev': events[i].prev_event_id,
                        'event_id': events[i].event_id,
                    },
                )
        return None

    def _check_epoch_monotonicity(self) -> Optional[SemanticInvariantViolation]:
        """EPOCH-001: Epoch must never decrease."""
        events = self._kernel.journal._events
        for i in range(1, len(events)):
            if events[i].epoch < events[i - 1].epoch:
                return SemanticInvariantViolation(
                    name='epoch_monotonicity',
                    severity=Severity.FATAL,
                    category='EPOCH',
                    detail=(
                        f"Epoch regression at event {events[i].event_id}: "
                        f"{events[i-1].epoch} → {events[i].epoch}"
                    ),
                    context={
                        'event_id': events[i].event_id,
                        'previous_epoch': events[i - 1].epoch,
                        'event_epoch': events[i].epoch,
                    },
                )
        return None

    def _check_epoch_observation_ratio(self) -> Optional[SemanticInvariantViolation]:
        """EPOCH-002: Epoch/observation ratio should not exceed 5.0."""
        obs = self._kernel._observation_count
        ep = self._kernel.epoch.current
        if obs > 0 and ep > 5 * obs:
            return SemanticInvariantViolation(
                name='epoch_observation_ratio',
                severity=Severity.WARNING,
                category='EPOCH',
                detail=(
                    f"Epoch/observation ratio={ep}/{obs} exceeds 5.0, "
                    f"indicating state changes without observations"
                ),
                context={
                    'epoch': ep,
                    'observation_count': obs,
                    'ratio': round(ep / obs, 2),
                },
            )
        return None

    def _check_grounding_journal_alignment(self) -> Optional[SemanticInvariantViolation]:
        """GROUNDING-001: Grounding checkpoint journal_size must ≤ actual events."""
        actual = len(self._kernel.journal._events)
        for cp in self._kernel.grounding._checkpoints:
            if cp.journal_size > actual:
                return SemanticInvariantViolation(
                    name='grounding_journal_alignment',
                    severity=Severity.FATAL,
                    category='GROUNDING',
                    detail=(
                        f"Checkpoint {cp.checkpoint_id} claims "
                        f"journal_size={cp.journal_size} but actual={actual}"
                    ),
                    context={
                        'checkpoint_id': cp.checkpoint_id,
                        'claimed_journal_size': cp.journal_size,
                        'actual_journal_size': actual,
                    },
                )
        return None

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    _INVARIANTS: Dict[str, Callable] = {}

    def _register(self):
        """Register all invariant check methods."""
        # Belief
        self._INVARIANTS['belief_confidence_in_bounds'] = (
            self._check_belief_confidence_in_bounds
        )
        # Motif
        self._INVARIANTS['motif_strength_in_bounds'] = (
            self._check_motif_strength_in_bounds
        )
        # Attractor
        self._INVARIANTS['attractor_weight_in_bounds'] = (
            self._check_attractor_weight_in_bounds
        )
        # Journal
        self._INVARIANTS['observation_precedes_belief'] = (
            self._check_observation_precedes_belief
        )
        self._INVARIANTS['journal_event_chain'] = (
            self._check_journal_event_chain
        )
        # Epoch
        self._INVARIANTS['epoch_monotonicity'] = (
            self._check_epoch_monotonicity
        )
        self._INVARIANTS['epoch_observation_ratio'] = (
            self._check_epoch_observation_ratio
        )
        # Grounding
        self._INVARIANTS['grounding_journal_alignment'] = (
            self._check_grounding_journal_alignment
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, *names: str) -> Optional[SemanticInvariantViolation]:
        """
        Run one or more invariants by name.

        Args:
            *names: invariant names. If multiple, returns
                    the first violation found (short-circuit).

        Returns:
            SemanticInvariantViolation or None if ALL passed.
        """
        if not self._INVARIANTS:
            self._register()
        if not names:
            return None
        for name in names:
            check = self._INVARIANTS.get(name)
            if check is None:
                return SemanticInvariantViolation(
                    name=name,
                    severity=Severity.ERROR,
                    category='SYSTEM',
                    detail=f"Unknown invariant: {name}",
                )
            violation = check()
            if violation is not None:
                return violation
        return None

    def verify_all(self) -> SemanticInvariantReport:
        """
        Run ALL registered invariants.

        Returns:
            SemanticInvariantReport with all results.
        """
        if not self._INVARIANTS:
            self._register()

        violations: List[SemanticInvariantViolation] = []
        for name, check in self._INVARIANTS.items():
            violation = check()
            if violation is not None:
                violations.append(violation)

        checked = len(self._INVARIANTS)
        passed = checked - len(violations)
        fatal = any(v.severity == Severity.FATAL for v in violations)

        return SemanticInvariantReport(
            invariants_checked=checked,
            invariants_passed=passed,
            violations=violations,
            fatal=fatal,
        )

    def verify_all_fast(self) -> SemanticInvariantReport:
        """
        Fast subset — run only lightweight invariants.

        MIRRORS fast_invariant_names in execution invariants.
        """
        fast = [
            'belief_confidence_in_bounds',
            'motif_strength_in_bounds',
            'attractor_weight_in_bounds',
            'journal_event_chain',
        ]
        violations = []
        for name in fast:
            check = self._INVARIANTS.get(name)
            if check:
                v = check()
                if v:
                    violations.append(v)

        return SemanticInvariantReport(
            invariants_checked=len(fast),
            invariants_passed=len(fast) - len(violations),
            violations=violations,
            fatal=any(v.severity == Severity.FATAL for v in violations),
        )
