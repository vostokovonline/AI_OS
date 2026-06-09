"""
Causal Policy Engine (CPE) — predictive decision layer over CausalityBridge.

Makes the causal graph generative, not just retrospective.

Architecture:

    CausalityBridge (retrospective + feedback)
          │
    ┌─────┴──────┐
    │   CPE      │  ← pure predictive layer, no kernel mutation
    │            │
    │ simulator  │  → CounterfactualSimulator (clone → apply → detect drift)
    │ scorer     │  → CausalScoringFunction (belief stability, drift, motif)
    │ selector   │  → PolicySelector (rank candidates by causal utility)
    └────────────┘
          │
    decision: "execute action X"  (or "reject all")

Usage:
    from causal_policy import CausalPolicyEngine

    cpe = CausalPolicyEngine(bridge)

    # Propose candidates
    candidates = cpe.propose_actions(goal_id="abc", options=[...])

    # Evaluate
    decision = cpe.evaluate(candidates, threshold=0.5)

    # Or full cycle: propose + evaluate
    decision = cpe.decide(
        goal_id="abc",
        options=[
            {'event_type': 'COMPLETED', 'success': True, 'label': 'retry'},
            {'event_type': 'ABANDONED', 'success': False, 'label': 'abandon'},
        ],
        threshold=0.5,
    )
"""

from typing import Any, Dict, List, Optional

from .simulator import CounterfactualSimulator, CandidateAction
from .scoring import CausalScoringFunction, DEFAULT_WEIGHTS
from .selector import PolicySelector, PolicyDecision


class CausalPolicyEngine:
    """
    Predictive decision engine over the causal bridge.

    This is the outermost layer — the "decision surface" of the system.
    It uses simulation + scoring to select actions by predicted causal utility.
    """

    def __init__(
        self,
        bridge=None,
        weights: Dict[str, float] = None,
    ):
        """
        Args:
            bridge: CausalityBridge instance (optional, can be set later)
            weights: custom scoring weights (uses defaults if None)
        """
        self._bridge = bridge
        self._epistemic_kernel = None
        self._simulator = None
        self._scorer = CausalScoringFunction(weights or DEFAULT_WEIGHTS)
        self._selector = None

        if bridge and bridge.epistemic_kernel:
            self._epistemic_kernel = bridge.epistemic_kernel
            self._simulator = CounterfactualSimulator(bridge.epistemic_kernel)
            self._selector = PolicySelector(self._simulator, self._scorer)

        self._total_decisions = 0
        self._decision_history: List[Dict[str, Any]] = []

    def set_bridge(self, bridge):
        """Set or update the CausalityBridge reference."""
        self._bridge = bridge
        if bridge and bridge.epistemic_kernel:
            self._epistemic_kernel = bridge.epistemic_kernel
            self._simulator = CounterfactualSimulator(bridge.epistemic_kernel)
            self._selector = PolicySelector(self._simulator, self._scorer)

    # ------------------------------------------------------------------
    # Candidate proposal
    # ------------------------------------------------------------------

    def propose_actions(
        self,
        goal_id: str,
        options: List[Dict[str, Any]] = None,
    ) -> List[CandidateAction]:
        """
        Generate candidate actions for a goal.

        If no options provided, generates default candidates:
          - retry (COMPLETED) — assume retry succeeds
          - retry_fail (FAILED) — assume retry fails
          - abandon (ABANDONED) — stop trying

        Args:
            goal_id: target goal
            options: custom candidate specs, each with:
                - event_type: COMPLETED | FAILED | PREEMPTED | CANCELLED | RETRIED | ABANDONED
                - success: bool (optional)
                - label: str (optional, auto-generated if missing)
                - confidence: float (optional, default 0.5)

        Returns:
            list of CandidateAction
        """
        if options is None:
            options = [
                {'event_type': 'COMPLETED', 'success': True, 'label': f'retry_{goal_id[:8]}', 'confidence': 0.5},
                {'event_type': 'FAILED', 'success': False, 'label': f'fail_{goal_id[:8]}', 'confidence': 0.5},
                {'event_type': 'ABANDONED', 'success': False, 'label': f'abandon_{goal_id[:8]}', 'confidence': 0.7},
            ]

        candidates = []
        for opt in options:
            label = opt.get('label', f"{opt['event_type'].lower()}_{goal_id[:8]}")
            candidates.append(CandidateAction(
                label=label,
                goal_id=goal_id,
                predicted_event_type=opt['event_type'],
                predicted_success=opt.get('success'),
                confidence=opt.get('confidence', 0.5),
                context=opt.get('context', {}),
            ))

        return candidates

    # ------------------------------------------------------------------
    # Evaluate candidates
    # ------------------------------------------------------------------

    def evaluate(
        self,
        candidates: List[CandidateAction],
        threshold: float = 0.5,
        with_feedback: bool = False,
    ) -> PolicyDecision:
        """
        Evaluate candidate actions and select the best.

        Args:
            candidates: list of CandidateAction
            threshold: minimum score for acceptance (0.0-1.0)
            with_feedback: include detailed per-dimension breakdown

        Returns:
            PolicyDecision
        """
        if self._selector is None:
            raise RuntimeError(
                "CausalPolicyEngine has no epistemic kernel. "
                "Call set_bridge() first."
            )

        if with_feedback:
            decision = self._selector.evaluate_with_feedback(candidates, threshold)
        else:
            decision = self._selector.evaluate(candidates, threshold)

        self._total_decisions += 1
        self._decision_history.append(decision.to_dict())

        return decision

    # ------------------------------------------------------------------
    # Full decision cycle
    # ------------------------------------------------------------------

    def decide(
        self,
        goal_id: str,
        options: List[Dict[str, Any]] = None,
        threshold: float = 0.5,
    ) -> PolicyDecision:
        """
        Full decision cycle: propose + evaluate + return.

        Args:
            goal_id: target goal
            options: custom options for propose_actions()
            threshold: minimum score for acceptance

        Returns:
            PolicyDecision
        """
        candidates = self.propose_actions(goal_id, options)
        return self.evaluate(candidates, threshold)

    # ------------------------------------------------------------------
    # Direct simulation (for introspection)
    # ------------------------------------------------------------------

    def simulate_event(
        self,
        event_type: str,
        goal_id: str = "simulated",
        success: Optional[bool] = None,
        label: str = "sim",
    ) -> Dict[str, Any]:
        """
        Directly simulate a single event without going through the
        full decision cycle. Useful for introspection / debugging.

        Returns:
            dict with outcome + score
        """
        if self._simulator is None:
            raise RuntimeError("No epistemic kernel set")

        candidate = CandidateAction(
            label=label,
            goal_id=goal_id,
            predicted_event_type=event_type,
            predicted_success=success,
        )
        outcome = self._simulator.simulate(candidate)
        score = self._scorer.score(outcome)
        return {
            'outcome': outcome.to_dict(),
            'score': score.to_dict(),
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get CPE diagnostics."""
        return {
            'total_decisions': self._total_decisions,
            'bridge_connected': self._bridge is not None,
            'epistemic_kernel_connected': self._epistemic_kernel is not None,
            'scoring_weights': dict(self._scorer._weights),
            'last_decisions': self._decision_history[-5:] if self._decision_history else [],
        }
