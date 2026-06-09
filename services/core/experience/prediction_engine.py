"""
Prediction Engine - Core of Cognitive Architecture

Key insight:
- Agent MUST predict before acting
- Compare prediction vs reality
- Update beliefs based on prediction error
- This is what makes cognition "predictive", not just retrospective

Architecture:
1. World Transition Model: P(next_state | current_state, action)
2. Prediction: Generate expected future state
3. Compare: predicted vs actual
4. Update: revise beliefs based on error
5. Learn: improve transition model from prediction errors
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime
import math


@dataclass(frozen=True)
class Prediction:
    """
    A prediction about future state.

    Key: predictions are FIRST-CLASS cognitive objects.
    They enable:
    - anticipation
    - planning
    - error detection
    - learning
    """
    prediction_id: str
    timestamp: str

    # What is being predicted
    predicted_aspect: str  # "error_count", "container_state", etc.
    predicted_value: Any
    confidence: float  # 0-1, how confident

    # Prediction context
    current_state_snapshot: str  # Hash of state at prediction time
    action_taken: str  # Action that should cause change

    # Alternative predictions (what else could happen)
    alternatives: Tuple[Tuple[Any, float], ...]  # (value, probability)

    @staticmethod
    def create(
        aspect: str,
        predicted_value: Any,
        confidence: float,
        current_state: str,
        action: str,
        alternatives: List[Tuple[Any, float]] = None
    ) -> 'Prediction':
        """Create new prediction"""
        prediction_id = hashlib.md5(
            f"{current_state}{action}{aspect}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        return Prediction(
            prediction_id=prediction_id,
            timestamp=datetime.now().isoformat(),
            predicted_aspect=aspect,
            predicted_value=predicted_value,
            confidence=confidence,
            current_state_snapshot=current_state,
            action_taken=action,
            alternatives=tuple(alternatives or [])
        )


@dataclass(frozen=True)
class PredictionError:
    """
    Difference between prediction and reality.

    This is THE core signal for learning.
    NOT reward. NOT utility. PREDICTION ERROR.

    Types:
    - EXPECTED error: prediction was uncertain, reality was within bounds
    - SURPRISE error: reality was outside prediction bounds
    - CATASTROPHIC error: direction was completely wrong
    """
    error_id: str
    timestamp: str

    # What was predicted
    aspect: str
    predicted_value: Any
    predicted_confidence: float

    # What actually happened
    actual_value: Any

    # Error magnitude
    absolute_error: float  # |predicted - actual|
    relative_error: float  # |predicted - actual| / actual (if numeric)
    directional_error: float  # positive = overestimated, negative = underestimated

    # Error classification
    error_type: str  # "expected", "surprise", "catastrophic"
    error_magnitude: float  # 0-1, how significant

    # Prediction was made with these expectations
    state_at_prediction: str
    action_at_prediction: str

    @staticmethod
    def compute(
        aspect: str,
        predicted: Any,
        predicted_confidence: float,
        actual: Any,
        state: str,
        action: str
    ) -> 'PredictionError':
        """Compute prediction error"""
        error_id = hashlib.md5(
            f"{aspect}{predicted}{actual}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Compute numeric errors
        try:
            pred_num = float(predicted)
            actual_num = float(actual)
            abs_err = abs(pred_num - actual_num)
            rel_err = abs_err / max(abs(actual_num), 1e-6)
            dir_err = pred_num - actual_num
        except (TypeError, ValueError):
            abs_err = 1.0 if predicted != actual else 0.0
            rel_err = 1.0 if predicted != actual else 0.0
            dir_err = 0.0

        # Classify error
        # High confidence + wrong = catastrophe
        # Low confidence + wrong = expected
        # Within expected bounds = fine
        if predicted_confidence > 0.8 and abs_err > 0.3:
            error_type = "catastrophic"
            magnitude = min(1.0, abs_err * predicted_confidence)
        elif abs_err > 0.5:
            error_type = "surprise"
            magnitude = min(1.0, abs_err * 0.5)
        else:
            error_type = "expected"
            magnitude = min(1.0, abs_err * 0.3)

        return PredictionError(
            error_id=error_id,
            timestamp=datetime.now().isoformat(),
            aspect=aspect,
            predicted_value=predicted,
            predicted_confidence=predicted_confidence,
            actual_value=actual,
            absolute_error=abs_err,
            relative_error=rel_err,
            directional_error=dir_err,
            error_type=error_type,
            error_magnitude=magnitude,
            state_at_prediction=state,
            action_at_prediction=action
        )


@dataclass
class WorldTransitionModel:
    """
    Model of P(next_state | current_state, action)

    This is what enables prediction.

    Structure:
    - For each (aspect, action) pair, track:
      - Historical transitions
      - Average outcomes
      - Variance/confidence

    Without this, predictions are just guesses.
    With this, predictions are MODEL-BASED.
    """
    # Transition statistics: (aspect, action) -> [transition_values]
    transitions: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)

    # Confidence in each transition model
    transition_confidence: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Prediction history for learning
    predictions_made: List[Prediction] = field(default_factory=list)
    prediction_errors: List[PredictionError] = field(default_factory=list)

    def record_transition(
        self,
        aspect: str,
        action: str,
        before_value: Any,
        after_value: Any
    ):
        """Record a state transition for learning"""
        key = (aspect, action)

        # Convert to numeric if possible
        try:
            delta = float(after_value) - float(before_value)
        except (TypeError, ValueError):
            delta = 1.0 if after_value != before_value else 0.0

        if key not in self.transitions:
            self.transitions[key] = []
            self.transition_confidence[key] = 0.0

        self.transitions[key].append(delta)

        # Keep only recent history (for recency bias)
        if len(self.transitions[key]) > 50:
            self.transitions[key] = self.transitions[key][-50:]

        # Update confidence (more data = higher confidence)
        count = len(self.transitions[key])
        self.transition_confidence[key] = min(1.0, count / 10)

    def predict(
        self,
        aspect: str,
        action: str,
        current_value: Any
    ) -> Tuple[Any, float]:
        """
        Predict next value using learned transition model.

        Returns:
        - predicted next value
        - confidence in prediction
        """
        key = (aspect, action)

        if key not in self.transitions or not self.transitions[key]:
            # No data - return current value with low confidence
            return current_value, 0.3

        # Use weighted average of recent transitions
        transitions = self.transitions[key]
        confidence = self.transition_confidence[key]

        if len(transitions) <= 3:
            # Not enough data - just use mean
            avg_delta = sum(transitions) / len(transitions)
        else:
            # Use recent weighted average
            recent = transitions[-10:] if len(transitions) >= 10 else transitions
            avg_delta = sum(recent) / len(recent)

        # Compute predicted value
        try:
            predicted = float(current_value) + avg_delta
        except (TypeError, ValueError):
            predicted = current_value

        return predicted, confidence

    def record_prediction_error(self, error: PredictionError):
        """Record prediction error for learning"""
        self.prediction_errors.append(error)

        # Update model based on error
        if error.error_type == "catastrophic":
            # Big error - reduce confidence in this transition
            key = (error.aspect, error.action_at_prediction)
            if key in self.transition_confidence:
                self.transition_confidence[key] *= 0.8

        elif error.error_type == "expected":
            # Small error - slightly increase confidence
            key = (error.aspect, error.action_at_prediction)
            if key in self.transition_confidence:
                self.transition_confidence[key] = min(1.0, self.transition_confidence[key] * 1.05)


@dataclass
class PredictionEngine:
    """
    Cognitive prediction engine.

    The core of predictive cognition:

    1. PREDICT: Before action, predict outcomes
    2. ACT: Execute action
    3. OBSERVE: See actual outcome
    4. COMPARE: Compute prediction error
    5. LEARN: Update model, revise beliefs

    This is what makes cognition "predictive" not just "reactive".
    """
    def __init__(self):
        self.transition_model = WorldTransitionModel()
        self.current_predictions: List[Prediction] = []
        self.prediction_errors: List[PredictionError] = []
        self.cycles_run = 0

    def predict_before_action(
        self,
        aspects: List[str],
        action: str,
        current_state: Dict[str, Any],
        state_signature: str
    ) -> List[Prediction]:
        """
        Generate predictions BEFORE action is taken.

        This is critical: predictions come BEFORE execution.
        """
        predictions = []

        for aspect in aspects:
            current_value = current_state.get(aspect)

            predicted_value, confidence = self.transition_model.predict(
                aspect, action, current_value
            )

            prediction = Prediction.create(
                aspect=aspect,
                predicted_value=predicted_value,
                confidence=confidence,
                current_state=state_signature,
                action=action
            )

            predictions.append(prediction)

        self.current_predictions = predictions
        return predictions

    def evaluate_predictions(
        self,
        actual_state: Dict[str, Any],
        state_signature: str
    ) -> List[PredictionError]:
        """
        Compare predictions to reality and compute errors.

        This is THE learning signal.
        """
        errors = []

        for pred in self.current_predictions:
            actual_value = actual_state.get(pred.predicted_aspect)

            if actual_value is None:
                continue

            error = PredictionError.compute(
                aspect=pred.predicted_aspect,
                predicted=pred.predicted_value,
                predicted_confidence=pred.confidence,
                actual=actual_value,
                state=state_signature,
                action=pred.action_taken
            )

            errors.append(error)
            self.prediction_errors.append(error)

            # Record for transition model learning
            self.transition_model.record_prediction_error(error)

        return errors

    def learn_from_transition(
        self,
        aspect: str,
        action: str,
        before_value: Any,
        after_value: Any
    ):
        """Learn from state transition"""
        self.transition_model.record_transition(
            aspect, action, before_value, after_value
        )

    def get_prediction_accuracy(self) -> Tuple[float, int]:
        """Get overall prediction accuracy"""
        if not self.prediction_errors:
            return 0.0, 0

        # Weight by confidence (high conf wrong = more penalty)
        total_error = sum(
            e.error_magnitude * e.predicted_confidence
            for e in self.prediction_errors
        )

        # Accuracy = 1 - weighted_error
        accuracy = 1.0 - (total_error / len(self.prediction_errors))
        return max(0.0, accuracy), len(self.prediction_errors)

    def get_model_confidence(self, aspect: str, action: str) -> float:
        """Get confidence in transition model for aspect/action"""
        key = (aspect, action)
        return self.transition_model.transition_confidence.get(key, 0.0)


@dataclass
class TemporalEpisode:
    """
    Temporal episode with causal chain.

    Enables:
    - Episodic memory
    - Causal chain tracking
    - Delayed effect detection
    - Counterfactual reasoning
    """
    episode_id: str
    start_time: str
    end_time: str

    # State at start
    initial_state: Dict[str, Any]
    initial_state_signature: str

    # Actions taken
    actions: Tuple[str, ...]  # Sequence of actions

    # Predictions made
    predictions: Tuple[Prediction, ...]

    # Outcomes observed
    actual_outcomes: Dict[str, Any]

    # Prediction errors
    errors: Tuple[PredictionError, ...]

    # Causal attribution
    causal_chain: Tuple[Tuple[str, str], ...]  # (action, outcome) pairs

    # Episode metadata
    episode_type: str  # "exploration", "recovery", "routine"
    success: bool
    key_insight: Optional[str]

    @staticmethod
    def create(
        initial_state: Dict[str, Any],
        state_signature: str,
        actions: List[str],
        predictions: List[Prediction],
        outcomes: Dict[str, Any],
        errors: List[PredictionError],
        episode_type: str = "exploration"
    ) -> 'TemporalEpisode':
        """Create new temporal episode"""
        episode_id = hashlib.md5(
            f"{state_signature}{len(actions)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Determine success from errors
        catastrophic = sum(1 for e in errors if e.error_type == "catastrophic")
        success = catastrophic == 0

        # Build causal chain
        chain = []
        for i, action in enumerate(actions):
            if i < len(outcomes):
                chain.append((action, str(outcomes.get(list(outcomes.keys())[i], "unknown"))))

        # Extract key insight from errors
        insight = None
        if errors:
            worst = max(errors, key=lambda e: e.error_magnitude)
            if worst.error_type == "catastrophic":
                insight = f"Prediction failed for {worst.aspect}: {worst.predicted_value} → {worst.actual_value}"

        return TemporalEpisode(
            episode_id=episode_id,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            initial_state=initial_state,
            initial_state_signature=state_signature,
            actions=tuple(actions),
            predictions=tuple(predictions),
            actual_outcomes=outcomes,
            errors=tuple(errors),
            causal_chain=tuple(chain),
            episode_type=episode_type,
            success=success,
            key_insight=insight
        )


@dataclass
class TemporalMemory:
    """
    Temporal memory system.

    Enables:
    - Episode storage and retrieval
    - Causal chain analysis
    - Delayed effect detection
    - Long-term pattern learning
    """
    episodes: List[TemporalEpisode] = field(default_factory=list)
    episode_index: Dict[str, List[int]] = field(default_factory=dict)  # signature -> episode indices

    def add_episode(self, episode: TemporalEpisode):
        """Add new episode"""
        self.episodes.append(episode)

        # Index by initial state signature
        sig = episode.initial_state_signature[:8]
        if sig not in self.episode_index:
            self.episode_index[sig] = []
        self.episode_index[sig].append(len(self.episodes) - 1)

    def get_similar_episodes(
        self,
        state_signature: str,
        limit: int = 5
    ) -> List[TemporalEpisode]:
        """Get episodes with similar initial states"""
        sig_prefix = state_signature[:8]
        indices = self.episode_index.get(sig_prefix, [])

        return [self.episodes[i] for i in indices[:limit] if i < len(self.episodes)]

    def get_episodes_by_action(
        self,
        action: str,
        limit: int = 10
    ) -> List[TemporalEpisode]:
        """Get episodes involving specific action"""
        matching = [
            ep for ep in self.episodes
            if action in ep.actions
        ]
        return matching[:limit]

    def analyze_causal_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across episodes"""
        if not self.episodes:
            return {}

        # Success rate
        successes = sum(1 for ep in self.episodes if ep.success)
        success_rate = successes / len(self.episodes)

        # Most common insights
        insights = [ep.key_insight for ep in self.episodes if ep.key_insight]
        common_insights = {}
        for insight in insights:
            common_insights[insight] = common_insights.get(insight, 0) + 1

        return {
            'total_episodes': len(self.episodes),
            'success_rate': success_rate,
            'common_insights': common_insights
        }