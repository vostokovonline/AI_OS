"""
TIER RELIABILITY AGGREGATION
Понимание, где ML слабее правил.

Агрегирует по окну (last 50 events) для каждого (action_type, tier):
- MAE (mean abs error)
- Direction Accuracy (% совпадений)
- Bias (mean signed error)
"""

from typing import Dict, Optional
from datetime import datetime, timezone


class TierReliabilityTracker:
    """
    Отслеживает надежность каждого tier для каждого action_type.

    Результат:
    {
      "complex_execution": {
        "ML": {
          "mae": 0.18,
          "direction_acc": 0.61,
          "bias": -0.12,
          "sample_count": 45
        },
        "Rules": {
          "mae": 0.09,
          "direction_acc": 0.79,
          "bias": -0.03,
          "sample_count": 120
        }
      }
    }
    """

    # Пороговые ориентиры
    THRESHOLDS = {
        "direction_acc_unreliable": 0.6,   # < 0.6 → tier ненадёжен
        "direction_acc_good": 0.75,        # > 0.75 → можно доверять
        "mae_concern": 0.20,               # > 0.20 → тревога
        "bias_concern": 0.10,               # |bias| > 0.10 → систематическое искажение
    }

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reliability_cache: Dict[str, Dict[str, Dict]] = {}
        self.last_update: Dict[str, Dict[str, datetime]] = {}

    def update_reliability(self, error_store):
        """
        Обновляет агрегированную надежность из error_store.

        Args:
            error_store: EmotionalErrorStore instance
        """
        action_types = set()
        for record in error_store.history:
            action_types.add(record["action_type"])

        # Для каждого (action_type, tier) обновляем метрики
        for action_type in action_types:
            if action_type not in self.reliability_cache:
                self.reliability_cache[action_type] = {}

            for tier in ["ML", "Clusters", "Rules"]:
                metrics = error_store.get_metrics_for_key(
                    action_type=action_type,
                    tier=tier,
                    limit=self.window_size
                )

                self.reliability_cache[action_type][tier] = metrics
                self.last_update[action_type] = datetime.now(timezone.utc)

    def get_reliability(
        self,
        action_type: str,
        tier: str
    ) -> Dict[str, float]:
        """Получить надежность для конкретного (action_type, tier)."""
        if action_type not in self.reliability_cache:
            return {
                "mae": 0.0,
                "direction_acc": 0.0,
                "bias": 0.0,
                "sample_count": 0
            }

        return self.reliability_cache[action_type].get(tier, {
            "mae": 0.0,
            "direction_acc": 0.0,
            "bias": 0.0,
            "sample_count": 0
        })

    def is_tier_reliable(self, action_type: str, tier: str) -> bool:
        """
        Проверяет надежен ли tier.

        Tier надежен если:
        - direction_acc >= 0.6
        - mae < 0.20
        - |bias| < 0.10
        """
        metrics = self.get_reliability(action_type, tier)

        if metrics["sample_count"] < 10:
            return False  # Недостаточно данных

        da = metrics["direction_acc"]
        mae = metrics["mae"]
        bias = metrics["bias"]

        return (
            da >= self.THRESHOLDS["direction_acc_unreliable"] and
            mae < self.THRESHOLDS["mae_concern"] and
            abs(bias) < self.THRESHOLDS["bias_concern"]
        )

    def get_best_tier(self, action_type: str) -> Optional[str]:
        """
        Возвращает лучший tier для action_type по MAE.

        Returns:
            "ML" / "Clusters" / "Rules" / None (если нет данных)
        """
        if action_type not in self.reliability_cache:
            return None

        tiers_data = self.reliability_cache[action_type]

        # Фильтруем только tier'ы с достаточными данными
        valid_tiers = {
            tier: metrics
            for tier, metrics in tiers_data.items()
            if metrics["sample_count"] >= 10
        }

        if not valid_tiers:
            return None

        # Сортируем по MAE (чем меньше, тем лучше)
        best_tier = min(valid_tiers.items(), key=lambda x: x[1]["mae"])[0]

        return best_tier

    def get_reliability_summary(self) -> Dict:
        """Возвращает summary всей надежности."""
        summary = {}

        for action_type, tiers in self.reliability_cache.items():
            summary[action_type] = {}

            for tier, metrics in tiers.items():
                reliable = self.is_tier_reliable(action_type, tier)

                summary[action_type][tier] = {
                    "mae": metrics["mae"],
                    "direction_acc": metrics["direction_acc"],
                    "bias": metrics["bias"],
                    "sample_count": metrics["sample_count"],
                    "reliable": reliable
                }

        return summary


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

tier_reliability_tracker = TierReliabilityTracker()
