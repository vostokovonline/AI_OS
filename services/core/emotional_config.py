"""
Emotional Layer Configuration
Single source of truth for thresholds & weights
"""

# =========================
# Base emotional dimensions
# =========================

EMOTIONAL_BASELINE = {
    "arousal": 0.5,
    "valence": 0.0,
    "focus": 0.5,
    "confidence": 0.5,
}

# =========================
# Thresholds (semantic)
# =========================

EMOTIONAL_THRESHOLDS = {
    # arousal
    "high_arousal": 0.7,

    # focus
    "low_focus": 0.4,

    # confidence
    "low_confidence": 0.3,

    # valence
    "low_valence": -0.4,
    "high_valence": 0.4,
}

# =========================
# Rule weights (MVP)
# =========================

RULE_WEIGHTS = {
    # inference
    "aborted_high_arousal": 0.2,
    "aborted_low_confidence": 0.2,

    "tired_low_valence": 0.3,
    "tired_low_focus": 0.2,
    "tired_high_arousal": 0.1,

    "simplify_low_arousal": 0.1,
    "simplify_high_focus": 0.1,

    "high_complexity_low_focus": 0.15,

    "success_high_valence": 0.2,
    "success_high_confidence": 0.1,
}

# =========================
# Influence mapping weights
# =========================

INFLUENCE_WEIGHTS = {
    "high_arousal_complexity_penalty": 0.3,
    "high_arousal_pace_modifier": -0.2,

    "low_focus_complexity_penalty": 0.4,

    "low_confidence_explanation_depth": 0.5,

    "low_valence_exploration_bias": -0.3,
    "high_valence_exploration_bias": 0.2,
}

# =========================
# Aggregation
# =========================

EMA_ALPHA = {
    "arousal": 0.3,
    "valence": 0.3,
    "focus": 0.4,
    "confidence": 0.3,
}
