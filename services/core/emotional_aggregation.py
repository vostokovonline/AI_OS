"""
Emotional State Aggregation (MVP)

EMA smoothing to prevent emotional jitter.
"""

from typing import Dict
from pydantic import BaseModel
from emotional_config import EMA_ALPHA


def ema(current: float, new: float, alpha: float) -> float:
    """
    Exponential Moving Average.

    Args:
        current: Previous value
        new: New value to incorporate
        alpha: Smoothing factor (0..1). Higher = more weight to new value.

    Returns:
        Smoothed value
    """
    return alpha * new + (1 - alpha) * current


class EmotionalStateAggregator:
    """
    Aggregates emotional states using EMA.
    """

    def __init__(self):
        # Use configured alpha values
        self.alphas = EMA_ALPHA

    def aggregate(
        self,
        previous: Dict[str, float],
        inferred: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aggregate previous state with new inference.

        Args:
            previous: {"arousal": 0.5, "valence": 0.0, "focus": 0.5, "confidence": 0.5}
            inferred: {"arousal": 0.7, "valence": -0.3, "focus": 0.3, "confidence": 0.4}

        Returns:
            Aggregated state with EMA smoothing
        """

        return {
            "arousal": ema(
                previous["arousal"],
                inferred["arousal"],
                self.alphas["arousal"]
            ),
            "valence": ema(
                previous["valence"],
                inferred["valence"],
                self.alphas["valence"]
            ),
            "focus": ema(
                previous["focus"],
                inferred["focus"],
                self.alphas["focus"]
            ),
            "confidence": ema(
                previous["confidence"],
                inferred["confidence"],
                self.alphas["confidence"]
            ),
        }
