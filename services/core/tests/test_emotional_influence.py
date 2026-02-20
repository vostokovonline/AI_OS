"""
Unit tests for Emotional Influence Engine
"""

import pytest
from emotional_influence import EmotionalInfluenceEngine, InfluenceContextMapper


class TestEmotionalInfluence:
    """Test emotional influence engine."""

    def setup_method(self):
        self.engine = EmotionalInfluenceEngine()
        self.mapper = InfluenceContextMapper()

    def test_neutral_state_no_influence(self):
        """Neutral state → no influence."""
        state = {
            "arousal": 0.5,
            "valence": 0.0,
            "focus": 0.5,
            "confidence": 0.5
        }

        influence = self.engine.map_to_influence(state)

        # Minimal influence from neutral state
        assert influence.complexity_penalty == 0.0
        assert influence.explanation_depth == 0.0

    def test_high_arousal_complexity_penalty(self):
        """High arousal → reduce complexity."""
        state = {
            "arousal": 0.8,
            "valence": 0.0,
            "focus": 0.5,
            "confidence": 0.5
        }

        influence = self.engine.map_to_influence(state)

        assert influence.complexity_penalty > 0.0
        assert influence.pace_modifier < 0.0

    def test_low_focus_complexity_penalty(self):
        """Low focus → reduce complexity."""
        state = {
            "arousal": 0.5,
            "valence": 0.0,
            "focus": 0.3,
            "confidence": 0.5
        }

        influence = self.engine.map_to_influence(state)

        assert influence.complexity_penalty > 0.0

    def test_low_confidence_explanation_depth(self):
        """Low confidence → more explanation."""
        state = {
            "arousal": 0.5,
            "valence": 0.0,
            "focus": 0.5,
            "confidence": 0.2
        }

        influence = self.engine.map_to_influence(state)

        assert influence.explanation_depth > 0.0

    def test_negative_valence_conservative(self):
        """Negative valence → conservative exploration."""
        state = {
            "arousal": 0.5,
            "valence": -0.5,
            "focus": 0.5,
            "confidence": 0.5
        }

        influence = self.engine.map_to_influence(state)

        assert influence.exploration_bias < 0.0

    def test_positive_valence_exploratory(self):
        """Positive valence → exploratory."""
        state = {
            "arousal": 0.5,
            "valence": 0.5,
            "focus": 0.5,
            "confidence": 0.5
        }

        influence = self.engine.map_to_influence(state)

        assert influence.exploration_bias > 0.0

    def test_context_mapping(self):
        """Test influence → context conversion."""
        # Tired user state
        state = {
            "arousal": 0.8,
            "valence": -0.3,
            "focus": 0.3,
            "confidence": 0.4
        }

        influence = self.engine.map_to_influence(state)
        context = self.mapper.to_context(influence)

        # Should recommend reduced complexity
        assert context["complexity_limit"] < 1.0
        assert context["max_depth"] <= 2

        # Should be conservative
        assert context["exploration"] == "conservative"

        # Should be slow paced
        assert context["pace"] == "slow"

    def test_clamping(self):
        """Influence values should be clamped."""
        # Extreme state
        state = {
            "arousal": 1.0,
            "valence": -1.0,
            "focus": 0.0,
            "confidence": 0.0
        }

        influence = self.engine.map_to_influence(state)

        # Should be clamped to valid ranges
        assert 0.0 <= influence.complexity_penalty <= 1.0
        assert -1.0 <= influence.exploration_bias <= 1.0
        assert 0.0 <= influence.explanation_depth <= 1.0
        assert -1.0 <= influence.pace_modifier <= 1.0

    def test_deterministic(self):
        """Same state → same influence."""
        state = {
            "arousal": 0.7,
            "valence": -0.2,
            "focus": 0.4,
            "confidence": 0.5
        }

        influence1 = self.engine.map_to_influence(state)
        influence2 = self.engine.map_to_influence(state)

        assert influence1.dict() == influence2.dict()
