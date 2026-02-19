"""
EMOTIONAL ERROR CLASSIFIER
====================================

STEP 2.5.1 — Error Taxonomy (Diagnostic Layer)

Принципы:
- Post-factum only: классифицируем только после outcome
- Read-only: НЕ влияет на inference, confidence, thresholds
- Детерминированность: один forecast + outcome = один набор labels
- Объяснимость > полнота: 6 чётких классов, не 20 мутных

Это диагностический слой, не коррекционный.
"""

from typing import Dict, List, Optional
from datetime import datetime


# =============================================================================
# ДИАГНОСТИЧЕСКИЕ КОНСТАНТЫ (не operational!)
# =============================================================================

# Пороги для классификации ошибок
EPSILON = 0.05              # noise floor — если actual < epsilon, не считаем изменением
MAE_OK = 0.10              # хорошая точность
MAE_BAD = 0.20             # плохая точность
HIGH_CONF = 0.70           # высокая уверенность
LOW_CONF = 0.40            # низкая уверенность
HIGH_AROUSAL = 0.75        # высокий возбуждение
PRED_MIN = 0.08            # минимальный прогноз, чтобы говорить о delayed effect
EFFECT_THRESHOLD = 0.10    # порог для отложенного эффекта


# =============================================================================
# ERROR CLASSIFIER
# =============================================================================

class EmotionalErrorClassifier:
    """
    Классифицирует emotional forecast errors по taxonomy.

    Использует ТОЛЬКО persisted data (forecast + outcome).
    НЕ влияет на поведение системы.
    """

    def classify_errors(
        self,
        forecast_data: Dict,
        outcome_data: Dict,
        history_context: Optional[Dict] = None
    ) -> List[str]:
        """
        Классифицирует ошибки для forecast + outcome pair.

        Args:
            forecast_data: {
                "predicted_deltas": {arousal, valence, focus, confidence},
                "forecast_confidence": float,
                "action_type": str,
                "used_tier": str,
                "baseline_arousal": float,  # опционально
                "created_at": datetime
            }
            outcome_data: {
                "actual_deltas": {arousal, valence, focus, confidence},
                "outcome": "success" | "failure" | "aborted",
                "completed_at": datetime
            }
            history_context: опциональный контекст для delayed_effect
                {
                    "recent_outcomes": [...],
                    "historical_confidence_mean": float
                }

        Returns:
            List[str]: набор error labels (может быть пустым)
        """
        labels = []

        predicted = forecast_data.get("predicted_deltas", {})
        actual = outcome_data.get("actual_deltas", {})
        confidence = forecast_data.get("forecast_confidence", 0.5)

        # Вычисляем вспомогательные величины
        error_magnitude = self._compute_mae(predicted, actual)
        direction_correct = self._check_direction_correct(predicted, actual)

        # 1️⃣ wrong_direction
        if self._is_wrong_direction(predicted, actual):
            labels.append("wrong_direction")

        # 2️⃣ overconfidence
        if self._is_overconfidence(confidence, direction_correct, error_magnitude, labels):
            labels.append("overconfidence")

        # 3️⃣ underconfidence
        if self._is_underconfidence(confidence, direction_correct, error_magnitude):
            labels.append("underconfidence")

        # 4️⃣ delayed_effect (требует history)
        if history_context and self._is_delayed_effect(predicted, actual, history_context):
            labels.append("delayed_effect")

        # 5️⃣ high_arousal_blindness
        if self._is_high_arousal_blindness(forecast_data, labels):
            labels.append("high_arousal_blindness")

        # 6️⃣ confidence_collapse (требует history)
        if history_context and self._is_confidence_collapse(confidence, error_magnitude, history_context):
            labels.append("confidence_collapse")

        return labels

    # =========================================================================
    # КРИТЕРИИ ОШИБОК (детерминированные функции)
    # =========================================================================

    def _is_wrong_direction(self, predicted: Dict, actual: Dict) -> bool:
        """
        1️⃣ wrong_direction

        Смысл: система предсказала изменение в неверную сторону.

        Критерий (по каждой dimension):
        - sign(predicted_delta) ≠ sign(actual_delta)
        - AND abs(actual_delta) > ε
        """
        for dim in ["arousal", "valence", "focus", "confidence"]:
            pred_val = predicted.get(dim, 0.0)
            actual_val = actual.get(dim, 0.0)

            # Пропускаем если actual ≈ 0 (noise floor)
            if abs(actual_val) < EPSILON:
                continue

            # Проверяем sign mismatch
            pred_sign = 1 if pred_val > 0 else (-1 if pred_val < 0 else 0)
            actual_sign = 1 if actual_val > 0 else (-1 if actual_val < 0 else 0)

            if pred_sign != actual_sign and pred_sign != 0 and actual_sign != 0:
                return True  # хотя бы одна dimension ошиблась

        return False

    def _is_overconfidence(
        self,
        confidence: float,
        direction_correct: bool,
        error_magnitude: float,
        current_labels: List[str]
    ) -> bool:
        """
        2️⃣ overconfidence

        Смысл: система была уверена, но ошиблась.

        Критерий (строгий):
        - confidence >= HIGH_CONF (0.7)
        - AND wrong_direction == true
        """
        # Вариант строгий (этическая ошибка)
        if confidence >= HIGH_CONF and "wrong_direction" in current_labels:
            return True

        # Вариант мягкий (статистическая ошибка)
        # if confidence >= HIGH_CONF and error_magnitude >= MAE_BAD:
        #     return True

        return False

    def _is_underconfidence(
        self,
        confidence: float,
        direction_correct: bool,
        error_magnitude: float
    ) -> bool:
        """
        3️⃣ underconfidence

        Смысл: система была неуверена, но оказалась права.

        Критерий:
        - confidence <= LOW_CONF (0.4)
        - AND direction_correct == true
        - AND error_magnitude <= MAE_OK (0.10)
        """
        return (
            confidence <= LOW_CONF and
            direction_correct and
            error_magnitude <= MAE_OK
        )

    def _is_delayed_effect(
        self,
        predicted: Dict,
        actual: Dict,
        history_context: Dict
    ) -> bool:
        """
        4️⃣ delayed_effect

        Смысл: прогноз был разумным, но эффект проявился позже.

        Критерий (v1 — упрощённый):
        - abs(predicted_delta) > PRED_MIN (0.08)
        - AND abs(actual_delta_at_t0) < ε
        - AND есть future outcomes с эффектом > EFFECT_THRESHOLD

        ⚠️ На v1 допускается heuristic/coarse detection.
        """
        # Проверяем magnitude прогноза
        pred_magnitude = sum(abs(v) for v in predicted.values()) / 4
        actual_magnitude = sum(abs(v) for v in actual.values()) / 4

        if pred_magnitude < PRED_MIN:
            return False

        # Проверяем что эффект НЕ проявился сейчас
        if actual_magnitude > EPSILON:
            return False

        # Проверяем есть ли future outcomes с эффектом
        recent_outcomes = history_context.get("recent_outcomes", [])
        if not recent_outcomes:
            return False

        # Ищем эффект в следующих outcomes
        for outcome in recent_outcomes[:3]:  # проверяем следующие 3
            future_actual = outcome.get("actual_deltas", {})
            future_magnitude = sum(abs(v) for v in future_actual.values()) / 4

            if future_magnitude >= EFFECT_THRESHOLD:
                # Проверяем совпадение направления с прогнозом
                if self._direction_match(predicted, future_actual):
                    return True

        return False

    def _is_high_arousal_blindness(
        self,
        forecast_data: Dict,
        current_labels: List[str]
    ) -> bool:
        """
        5️⃣ high_arousal_blindness

        Смысл: системно ошибается при высоком arousal.

        Критерий (на уровне одного forecast):
        - baseline_arousal >= HIGH_AROUSAL (0.75)
        - AND wrong_direction == true
        """
        baseline_arousal = forecast_data.get("baseline_arousal")
        if baseline_arousal is None:
            return False  # нет данных о baseline

        return (
            baseline_arousal >= HIGH_AROUSAL and
            "wrong_direction" in current_labels
        )

    def _is_confidence_collapse(
        self,
        confidence: float,
        error_magnitude: float,
        history_context: Dict
    ) -> bool:
        """
        6️⃣ confidence_collapse

        Смысл: confidence резко упала без ухудшения точности.

        Критерий:
        - confidence << historical_mean_confidence
        - AND error_magnitude <= MAE_OK
        """
        hist_conf_mean = history_context.get("historical_confidence_mean")
        if hist_conf_mean is None:
            return False  # нет истории

        # Confidence резко ниже среднего
        # (например, на 30% ниже)
        return (
            confidence < hist_conf_mean * 0.7 and
            error_magnitude <= MAE_OK
        )

    # =========================================================================
    # ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
    # =========================================================================

    def _compute_mae(self, predicted: Dict, actual: Dict) -> float:
        """Вычисляет Mean Absolute Error."""
        errors = []
        for dim in ["arousal", "valence", "focus", "confidence"]:
            pred_val = predicted.get(dim, 0.0)
            actual_val = actual.get(dim, 0.0)
            errors.append(abs(pred_val - actual_val))

        return sum(errors) / len(errors) if errors else 0.0

    def _check_direction_correct(self, predicted: Dict, actual: Dict) -> bool:
        """Проверяет что направление угадано верно по всем dimension."""
        for dim in ["arousal", "valence", "focus", "confidence"]:
            pred_val = predicted.get(dim, 0.0)
            actual_val = actual.get(dim, 0.0)

            # Пропускаем если actual ≈ 0
            if abs(actual_val) < EPSILON:
                continue

            pred_sign = 1 if pred_val > 0 else (-1 if pred_val < 0 else 0)
            actual_sign = 1 if actual_val > 0 else (-1 if actual_val < 0 else 0)

            if pred_sign != actual_sign and pred_sign != 0 and actual_sign != 0:
                return False

        return True

    def _direction_match(self, predicted: Dict, actual: Dict) -> bool:
        """Проверяет совпадение знаков (для delayed_effect)."""
        matches = 0
        total = 0

        for dim in ["arousal", "valence", "focus", "confidence"]:
            pred_val = predicted.get(dim, 0.0)
            actual_val = actual.get(dim, 0.0)

            pred_sign = 1 if pred_val > 0 else (-1 if pred_val < 0 else 0)
            actual_sign = 1 if actual_val > 0 else (-1 if actual_val < 0 else 0)

            if pred_sign != 0 and actual_sign != 0:
                total += 1
                if pred_sign == actual_sign:
                    matches += 1

        if total == 0:
            return False

        return matches / total >= 0.5  # хотя бы 50% совпадение


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

emotional_error_classifier = EmotionalErrorClassifier()
