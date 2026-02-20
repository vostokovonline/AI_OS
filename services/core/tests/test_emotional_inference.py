"""
Unit tests for Emotional Inference Engine

Pure functions, no DB, no async → simple and fast.
"""

import pytest
from emotional_inference import EmotionalInferenceEngine
from schemas import EmotionalSignals


class TestEmotionalInference:
    """Test emotional inference engine."""

    def setup_method(self):
        self.engine = EmotionalInferenceEngine()

    def test_baseline_state(self):
        """Baseline should be neutral."""
        signals = EmotionalSignals()

        state = self.engine.infer(signals)

        assert state["arousal"] == 0.5
        assert state["valence"] == 0.0
        assert state["focus"] == 0.5
        assert state["confidence"] == 0.5

    def test_aborted_goals_increase_arousal(self):
        """Multiple aborted goals → high arousal, low confidence."""
        signals = EmotionalSignals(
            goal_stats={"aborted": 3}
        )

        state = self.engine.infer(signals)

        assert state["arousal"] > 0.6
        assert state["confidence"] < 0.4

    def test_tired_text_decreases_valence_and_focus(self):
        """User says they're tired → negative valence, low focus."""
        signals = EmotionalSignals(
            user_text="Я устал, это слишком сложно"
        )

        state = self.engine.infer(signals)

        assert state["valence"] < 0.0
        assert state["focus"] < 0.5
        assert state["arousal"] > 0.5

    def test_simplify_request_decreases_arousal(self):
        """User asks to simplify → lower arousal, higher focus."""
        signals = EmotionalSignals(
            user_text="Давай попроще"
        )

        state = self.engine.infer(signals)

        assert state["arousal"] < 0.5
        assert state["focus"] > 0.5

    def test_high_complexity_decreases_focus(self):
        """High goal complexity → cognitive overload."""
        signals = EmotionalSignals(
            system_metrics={"avg_goal_complexity": 0.8}
        )

        state = self.engine.infer(signals)

        assert state["focus"] < 0.5

    def test_success_increases_valence_and_confidence(self):
        """High success ratio → positive state."""
        signals = EmotionalSignals(
            system_metrics={"success_ratio": 0.8}
        )

        state = self.engine.infer(signals)

        assert state["valence"] > 0.0
        assert state["confidence"] > 0.5

    def test_clamping(self):
        """Values should be clamped to valid ranges."""
        # Extreme case
        signals = EmotionalSignals(
            user_text="устал устал устал слишком сложно не могу",
            goal_stats={"aborted": 10},
            system_metrics={"avg_goal_complexity": 1.0}
        )

        state = self.engine.infer(signals)

        # Should be clamped to valid ranges
        assert 0.0 <= state["arousal"] <= 1.0
        assert -1.0 <= state["valence"] <= 1.0
        assert 0.0 <= state["focus"] <= 1.0
        assert 0.0 <= state["confidence"] <= 1.0

    def test_deterministic(self):
        """Same input → same output."""
        signals = EmotionalSignals(
            user_text="Я устал",
            goal_stats={"aborted": 2}
        )

        state1 = self.engine.infer(signals)
        state2 = self.engine.infer(signals)

        assert state1 == state2
