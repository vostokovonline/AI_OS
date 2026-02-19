"""
Emotional Influence Engine (MVP)

Maps emotional state → decision weights.
Deterministic, testable, no side effects.
"""

from typing import Dict, Literal
from pydantic import BaseModel
from emotional_config import (
    EMOTIONAL_THRESHOLDS,
    INFLUENCE_WEIGHTS,
)


class EmotionalInfluence(BaseModel):
    """
    How emotions affect decisions.

    All values are relative modifiers (-1.0 to 1.0).
    Applied ON TOP of baseline decision logic.
    """

    # Complexity modifiers
    complexity_penalty: float = 0.0  # 0..1, reduces max complexity
    exploration_bias: float = 0.0    # -1..1, negative = conservative

    # Communication modifiers
    explanation_depth: float = 0.0   # 0..1, more detailed explanations
    pace_modifier: float = 0.0       # -1..1, negative = slower

    # Meta
    confidence_modifier: float = 0.0 # -1..1, affects agent confidence


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value to range."""
    return max(min_value, min(max_value, value))


class EmotionalInfluenceEngine:
    """
    Maps emotional state to influence on decisions.
    """

    def map_to_influence(self, state: Dict[str, float]) -> EmotionalInfluence:
        """
        Convert emotional state to influence weights.

        Args:
            state: {"arousal": 0.7, "valence": -0.3, "focus": 0.3, "confidence": 0.4}

        Returns:
            EmotionalInfluence with all modifiers
        """

        influence = EmotionalInfluence()

        # Apply mappings
        self._map_arousal(state["arousal"], influence)
        self._map_focus(state["focus"], influence)
        self._map_confidence(state["confidence"], influence)
        self._map_valence(state["valence"], influence)

        # Safety clamp
        self._clamp_influence(influence)

        return influence

    # =========================
    # Dimension mappings
    # =========================

    def _map_arousal(self, arousal: float, influence: EmotionalInfluence):
        """
        High arousal = overload → reduce complexity, slow down
        """
        if arousal > EMOTIONAL_THRESHOLDS["high_arousal"]:
            influence.complexity_penalty += INFLUENCE_WEIGHTS["high_arousal_complexity_penalty"]
            influence.pace_modifier += INFLUENCE_WEIGHTS["high_arousal_pace_modifier"]

    def _map_focus(self, focus: float, influence: EmotionalInfluence):
        """
        Low focus = can't handle complexity
        """
        if focus < EMOTIONAL_THRESHOLDS["low_focus"]:
            influence.complexity_penalty += INFLUENCE_WEIGHTS["low_focus_complexity_penalty"]

    def _map_confidence(self, confidence: float, influence: EmotionalInfluence):
        """
        Low confidence = needs more explanation
        """
        if confidence < EMOTIONAL_THRESHOLDS["low_confidence"]:
            influence.explanation_depth += INFLUENCE_WEIGHTS["low_confidence_explanation_depth"]

        # Also affects agent's confidence level
        influence.confidence_modifier = (confidence - 0.5) * 2  # Map 0..1 to -1..1

    def _map_valence(self, valence: float, influence: EmotionalInfluence):
        """
        Valence affects exploration vs exploitation
        """
        if valence < EMOTIONAL_THRESHOLDS["low_valence"]:
            # Negative = conservative, stick to known
            influence.exploration_bias += INFLUENCE_WEIGHTS["low_valence_exploration_bias"]

        elif valence > EMOTIONAL_THRESHOLDS["high_valence"]:
            # Positive = explore more
            influence.exploration_bias += INFLUENCE_WEIGHTS["high_valence_exploration_bias"]

    # =========================
    # Safety
    # =========================

    def _clamp_influence(self, influence: EmotionalInfluence):
        """
        Prevent extreme values from breaking the system.
        """
        influence.complexity_penalty = clamp(influence.complexity_penalty, 0.0, 1.0)
        influence.explanation_depth = clamp(influence.explanation_depth, 0.0, 1.0)

        influence.exploration_bias = clamp(influence.exploration_bias, -1.0, 1.0)
        influence.pace_modifier = clamp(influence.pace_modifier, -1.0, 1.0)
        influence.confidence_modifier = clamp(influence.confidence_modifier, -1.0, 1.0)


# =========================
# Helper: Influence → Context (for agents)
# =========================

class InfluenceContextMapper:
    """
    Converts EmotionalInfluence to agent-friendly context hints.
    """

    @staticmethod
    def to_context(influence: EmotionalInfluence) -> Dict[str, any]:
        """
        Convert influence to context dict for agents/prompts.

        Returns:
        {
            "complexity_limit": float,      # 0..1
            "max_depth": int,               # 1..3
            "exploration": str,             # "conservative" | "balanced" | "exploratory"
            "explanation": str,             # "brief" | "normal" | "detailed"
            "pace": str,                    # "fast" | "normal" | "slow"
            "confidence": str,              # "low" | "normal" | "high"
        }
        """

        context = {}

        # Complexity
        context["complexity_limit"] = clamp(1.0 - influence.complexity_penalty, 0.3, 1.0)

        # Max depth (derived from complexity)
        if influence.complexity_penalty > 0.5:
            context["max_depth"] = 1
        elif influence.complexity_penalty > 0.2:
            context["max_depth"] = 2
        else:
            context["max_depth"] = 3

        # Exploration
        if influence.exploration_bias < -0.3:
            context["exploration"] = "conservative"
        elif influence.exploration_bias > 0.3:
            context["exploration"] = "exploratory"
        else:
            context["exploration"] = "balanced"

        # Explanation
        if influence.explanation_depth > 0.5:
            context["explanation"] = "detailed"
        elif influence.explanation_depth > 0.2:
            context["explanation"] = "normal"
        else:
            context["explanation"] = "brief"

        # Pace
        if influence.pace_modifier < -0.3:
            context["pace"] = "slow"
        elif influence.pace_modifier > 0.3:
            context["pace"] = "fast"
        else:
            context["pace"] = "normal"

        # Confidence
        if influence.confidence_modifier < -0.3:
            context["confidence"] = "low"
        elif influence.confidence_modifier > 0.3:
            context["confidence"] = "high"
        else:
            context["confidence"] = "normal"

        return context
