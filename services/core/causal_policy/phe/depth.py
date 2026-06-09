"""
Depth Strategy Engine — determines how deep the policy tree should expand.

Two strategies:
  1. FixedDepth — always expand to N levels
  2. AdaptiveDepth — depth depends on drift, uncertainty, motif coherence

The adaptive strategy is the RECOMMENDED default:
  - If drift is rising → shrink horizon
  - If uncertainty is high → shrink horizon
  - If motifs are coherent → allow deeper search
  - If branching factor is high → shrink horizon (avoid explosion)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DepthConfig:
    """
    Configuration for depth strategy.

    Fields:
      max_depth: absolute maximum depth (hard cap)
      min_depth: minimum depth (even in high uncertainty)
      drift_sensitivity: how much drift reduces depth (0.0-1.0)
      uncertainty_sensitivity: how much uncertainty reduces depth (0.0-1.0)
      motif_boost: how much motif coherence increases depth (0.0-1.0)
      branching_penalty: how much branching reduces depth (0.0-1.0)
    """
    max_depth: int = 5
    min_depth: int = 1
    drift_sensitivity: float = 0.6
    uncertainty_sensitivity: float = 0.5
    motif_boost: float = 0.3
    branching_penalty: float = 0.2


class DepthStrategy(ABC):
    """Abstract depth strategy."""

    @abstractmethod
    def should_expand(
        self,
        depth: int,
        uncertainty: float,
        drift_estimate: float,
        motif_count: int,
        avg_motif_strength: float,
        branching_factor: float,
    ) -> bool:
        ...

    @abstractmethod
    def compute_max_depth(
        self,
        uncertainty: float,
        drift_estimate: float,
        motif_count: int,
        avg_motif_strength: float,
    ) -> int:
        ...


class FixedDepth(DepthStrategy):
    """
    Always expand to a fixed depth.

    Simple, predictable, but ignores world state.
    """

    def __init__(self, depth: int = 3):
        self._depth = max(1, depth)

    def should_expand(
        self,
        depth: int,
        uncertainty: float = 0.0,
        drift_estimate: float = 0.0,
        motif_count: int = 0,
        avg_motif_strength: float = 0.0,
        branching_factor: float = 1.0,
    ) -> bool:
        return depth < self._depth

    def compute_max_depth(
        self,
        uncertainty: float = 0.0,
        drift_estimate: float = 0.0,
        motif_count: int = 0,
        avg_motif_strength: float = 0.0,
    ) -> int:
        return self._depth


class AdaptiveDepth(DepthStrategy):
    """
    Depth adapts to world state.

    Formula:
      base = max_depth
      drift_penalty = drift * drift_sensitivity * base
      uncertainty_penalty = uncertainty * uncertainty_sensitivity * base
      motif_bonus = avg_motif_strength * motif_boost * base
      effective_depth = max(min_depth, base - drift_penalty - uncertainty_penalty + motif_bonus)
    """

    def __init__(self, config: Optional[DepthConfig] = None):
        self.config = config or DepthConfig()

    def compute_max_depth(
        self,
        uncertainty: float = 0.0,
        drift_estimate: float = 0.0,
        motif_count: int = 0,
        avg_motif_strength: float = 0.0,
    ) -> int:
        base = float(self.config.max_depth)

        drift_penalty = drift_estimate * self.config.drift_sensitivity * base
        uncertainty_penalty = uncertainty * self.config.uncertainty_sensitivity * base
        motif_bonus = avg_motif_strength * self.config.motif_boost * base

        effective = base - drift_penalty - uncertainty_penalty + motif_bonus
        return max(self.config.min_depth, min(self.config.max_depth, round(effective)))

    def should_expand(
        self,
        depth: int,
        uncertainty: float = 0.0,
        drift_estimate: float = 0.0,
        motif_count: int = 0,
        avg_motif_strength: float = 0.0,
        branching_factor: float = 1.0,
    ) -> bool:
        if depth >= self.config.max_depth:
            return False

        max_d = self.compute_max_depth(
            uncertainty, drift_estimate, motif_count, avg_motif_strength
        )
        if depth >= max_d:
            return False

        # Branching penalty: wide trees get shallower
        if branching_factor > 3.0:
            penalty = self.config.branching_penalty * (branching_factor - 3.0)
            if depth >= round(max_d - penalty * max_d):
                return False

        return True
