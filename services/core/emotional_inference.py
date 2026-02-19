"""
Emotional Inference Engine (MVP)

Rule-based, deterministic.
NO database access.
NO side effects.
"""

from typing import Dict
from schemas import EmotionalSignals
from emotional_config import (
    EMOTIONAL_BASELINE,
    EMOTIONAL_THRESHOLDS,
    RULE_WEIGHTS,
)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


class EmotionalInferenceEngine:
    """
    Infers emotional state from aggregated signals.
    """

    def infer(self, signals: EmotionalSignals) -> Dict[str, float]:
        """
        Main entry point.

        Returns:
        {
            "arousal": float (0..1),
            "valence": float (-1..1),
            "focus": float (0..1),
            "confidence": float (0..1)
        }
        """

        # 1. Start from neutral baseline
        state = dict(EMOTIONAL_BASELINE)

        # 2. Apply rule groups
        self._apply_goal_stats_rules(state, signals)
        self._apply_user_text_rules(state, signals)
        self._apply_system_metrics_rules(state, signals)

        # 3. Clamp values
        state["arousal"] = clamp(state["arousal"], 0.0, 1.0)
        state["focus"] = clamp(state["focus"], 0.0, 1.0)
        state["confidence"] = clamp(state["confidence"], 0.0, 1.0)
        state["valence"] = clamp(state["valence"], -1.0, 1.0)

        return state

    # =========================
    # Rule groups
    # =========================

    def _apply_goal_stats_rules(
        self,
        state: Dict[str, float],
        signals: EmotionalSignals,
    ):
        """
        Rules based on aggregated goal statistics.
        """
        if not signals.goal_stats:
            return

        aborted = signals.goal_stats.get("aborted", 0)

        # Aborted goals → stress + loss of confidence
        if aborted >= 3:
            state["arousal"] += RULE_WEIGHTS["aborted_high_arousal"]
            state["confidence"] -= RULE_WEIGHTS["aborted_low_confidence"]

    def _apply_user_text_rules(
        self,
        state: Dict[str, float],
        signals: EmotionalSignals,
    ):
        """
        Rules based on user text.
        """
        if not signals.user_text:
            return

        text = signals.user_text.lower()

        # Fatigue / overload
        if any(word in text for word in ["устал", "тяжело", "не могу", "слишком сложно"]):
            state["valence"] -= RULE_WEIGHTS["tired_low_valence"]
            state["focus"] -= RULE_WEIGHTS["tired_low_focus"]
            state["arousal"] += RULE_WEIGHTS["tired_high_arousal"]

        # Request to simplify
        if any(word in text for word in ["проще", "попроще", "упрости"]):
            state["arousal"] -= RULE_WEIGHTS["simplify_low_arousal"]
            state["focus"] += RULE_WEIGHTS["simplify_high_focus"]

    def _apply_system_metrics_rules(
        self,
        state: Dict[str, float],
        signals: EmotionalSignals,
    ):
        """
        Rules based on system-level metrics.
        """
        if not signals.system_metrics:
            return

        # High complexity → cognitive overload
        avg_complexity = signals.system_metrics.get("avg_goal_complexity")
        if (
            avg_complexity is not None
            and avg_complexity > EMOTIONAL_THRESHOLDS["low_focus"]
        ):
            state["focus"] -= RULE_WEIGHTS["high_complexity_low_focus"]

        # Success ratio → positive reinforcement
        success_ratio = signals.system_metrics.get("success_ratio")
        if success_ratio is not None and success_ratio > 0.7:
            state["valence"] += RULE_WEIGHTS["success_high_valence"]
            state["confidence"] += RULE_WEIGHTS["success_high_confidence"]
