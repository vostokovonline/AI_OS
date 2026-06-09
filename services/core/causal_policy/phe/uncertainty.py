"""
Uncertainty Decay Model — critical stabilizer for deep policy search.

The problem:
  Deep simulation = compounding errors = false confidence.

The solution:
  U(t+1) = U(t) + α * simulation_error + β * drift + γ * branching_factor
  confidence_t = confidence_0 * exp(-λ * depth)

Where:
  α (alpha): simulation error weight — how much the CPE simulator errors compound
  β (beta): drift weight — how much epistemic drift increases uncertainty
  γ (gamma): branching weight — how much branching increases uncertainty
  λ (lambda): base decay rate — controls how fast confidence decays with depth

At depth 0: confidence = 1.0 (current real state)
At depth 3: confidence ≈ 0.5 (moderate confidence)
At depth 5: confidence ≈ 0.3 (low confidence)
At depth 8: confidence ≈ 0.1 (very uncertain)
"""

from dataclasses import dataclass
from typing import Optional


DEFAULT_ALPHA = 0.15
DEFAULT_BETA = 0.10
DEFAULT_GAMMA = 0.08
DEFAULT_LAMBDA = 0.25
DEFAULT_BASE_UNCERTAINTY = 0.05


@dataclass
class UncertaintyState:
    """
    Current uncertainty state for a node in the policy tree.

    Fields:
      uncertainty: current uncertainty value (0.0-1.0)
      confidence: current confidence value (0.0-1.0)
      depth: depth of this node
      simulation_error_accumulated: accumulated simulation error
    """
    uncertainty: float
    confidence: float
    depth: int
    simulation_error_accumulated: float = 0.0

    def to_dict(self) -> dict:
        return {
            'uncertainty': self.uncertainty,
            'confidence': self.confidence,
            'depth': self.depth,
            'simulation_error_accumulated': self.simulation_error_accumulated,
        }


class UncertaintyDecayModel:
    """
    Computes uncertainty and confidence for simulated states.

    The model ensures:
      - Uncertainty is ALWAYS ≥ parent uncertainty (monotonic)
      - Confidence ALWAYS ≤ parent confidence (monotonic)
      - Deep simulations are NEVER more certain than shallow ones
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        gamma: float = DEFAULT_GAMMA,
        lambd: float = DEFAULT_LAMBDA,
        base_uncertainty: float = DEFAULT_BASE_UNCERTAINTY,
    ):
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._lambd = lambd
        self._base_uncertainty = base_uncertainty

    def compute_uncertainty(
        self,
        parent_uncertainty: float,
        simulation_error: float = 0.0,
        drift: float = 0.0,
        branching_factor: float = 1.0,
    ) -> float:
        """
        Compute uncertainty for a child node given parent uncertainty and deltas.

        U(t+1) = U(t) + α * simulation_error + β * drift + γ * branching_factor

        Args:
            parent_uncertainty: U(t)
            simulation_error: how much the CPE simulator error compounds (0.0-1.0)
            drift: epistemic drift at this state (0.0-1.0)
            branching_factor: number of candidate actions considered (min 1)

        Returns:
            new uncertainty value (0.0-1.0), clamped
        """
        delta = (
            self._alpha * simulation_error
            + self._beta * drift
            + self._gamma * (max(branching_factor, 1.0) - 1.0)
        )
        new_uncertainty = parent_uncertainty + delta
        return min(1.0, max(self._base_uncertainty, new_uncertainty))

    def confidence_at_depth(
        self,
        depth: int,
        base_confidence: float = 1.0,
    ) -> float:
        """
        Compute confidence at a given depth.

        confidence_t = confidence_0 * exp(-λ * depth)

        Args:
            depth: current depth (0 = real state)
            base_confidence: confidence at depth 0 (always 1.0)

        Returns:
            confidence (0.0-1.0)
        """
        confidence = base_confidence * (2.71828 ** (-self._lambd * depth))
        return round(max(0.0, min(1.0, confidence)), 4)

    def initial_state(self) -> UncertaintyState:
        """Create initial uncertainty state (depth 0, real state)."""
        return UncertaintyState(
            uncertainty=self._base_uncertainty,
            confidence=1.0,
            depth=0,
        )

    def compute_child_state(
        self,
        parent: UncertaintyState,
        simulation_error: float = 0.0,
        drift: float = 0.0,
        branching_factor: float = 1.0,
    ) -> UncertaintyState:
        """
        Compute uncertainty state for a child node.

        Args:
            parent: UncertaintyState of parent node
            simulation_error: CPE simulator error (0.0-1.0)
            drift: epistemic drift at this state
            branching_factor: number of candidates at this level

        Returns:
            UncertaintyState for child node
        """
        child_depth = parent.depth + 1
        uncertainty = self.compute_uncertainty(
            parent_uncertainty=parent.uncertainty,
            simulation_error=simulation_error,
            drift=drift,
            branching_factor=branching_factor,
        )
        confidence = self.confidence_at_depth(child_depth)

        return UncertaintyState(
            uncertainty=uncertainty,
            confidence=confidence,
            depth=child_depth,
            simulation_error_accumulated=parent.simulation_error_accumulated + simulation_error,
        )
