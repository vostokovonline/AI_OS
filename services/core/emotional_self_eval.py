"""
EMOTIONAL SELF-EVAL - Forecast Comparator
Сравнивает что система ожидала vs что произошло.

Key Principle:
Self-eval НЕ меняет решение постфактум.
Он влияет только на будущие confidence и retraining.
"""

from typing import Dict, List
from datetime import datetime, timezone


class SelfEvalComparator:
    """
    Сравнивает predicted vs actual emotional deltas.

    Вычисляет ТОЛЬКО:
    1. signed_error - bias
    2. abs_error - точность
    3. direction_match - качество решения
    """

    def compare_forecast(
        self,
        predicted: Dict[str, float],
        actual: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Сравнивает прогноз с реальностью.

        Args:
            predicted: {"arousal": 0.1, "valence": -0.05, ...}
            actual: {"arousal": 0.15, "valence": -0.02, ...}

        Returns:
            {
                "arousal": {"signed": 0.05, "abs": 0.05, "direction_match": true},
                "valence": {"signed": 0.03, "abs": 0.03, "direction_match": true},
                ...
            }
        """
        errors = {}

        for dim in ["arousal", "valence", "focus", "confidence"]:
            pred_val = predicted.get(dim, 0.0)
            actual_val = actual.get(dim, 0.0)

            signed_error = actual_val - pred_val
            abs_error = abs(signed_error)

            # Direction match: правильно ли угадали знак изменения
            pred_sign = 1 if pred_val > 0 else (-1 if pred_val < 0 else 0)
            actual_sign = 1 if actual_val > 0 else (-1 if actual_val < 0 else 0)

            # Если оба ноль или одинаковые знаки → match
            direction_match = (pred_sign == actual_sign) or (pred_sign == 0 and actual_sign == 0)

            errors[dim] = {
                "signed": round(signed_error, 4),
                "abs": round(abs_error, 4),
                "direction_match": direction_match
            }

        return errors

    def compute_metrics(
        self,
        errors: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Агрегирует ошибки по всем размерностям.

        Returns:
            {
                "direction_acc": 0.75,  # % совпадений направлений
                "mae": 0.12,            # Mean absolute error
                "bias": -0.03           # Mean signed error
            }
        """
        if not errors:
            return {"direction_acc": 0.0, "mae": 0.0, "bias": 0.0}

        abs_errors = []
        signed_errors = []
        direction_matches = []

        for dim_err in errors.values():
            abs_errors.append(dim_err["abs"])
            signed_errors.append(dim_err["signed"])
            direction_matches.append(1 if dim_err["direction_match"] else 0)

        # Directional Accuracy
        direction_acc = sum(direction_matches) / len(direction_matches) if direction_matches else 0.0

        # Mean Absolute Error
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

        # Bias (mean signed error)
        bias = sum(signed_errors) / len(signed_errors) if signed_errors else 0.0

        return {
            "direction_acc": round(direction_acc, 4),
            "mae": round(mae, 4),
            "bias": round(bias, 4)
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

self_eval_comparator = SelfEvalComparator()
