"""
EMOTIONAL ERROR STORE - Forecast Error History
Минималистичное хранилище ошибок прогнозирования.

Хранит ТОЛЬКО то, что нужно для анализа:
- action_type
- tier (ML/Clusters/Rules)
- confidence
- errors по каждой размерности
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone


class EmotionalErrorStore:
    """
    Хранит историю ошибок прогнозирования.

    ❌ НЕ сохраняет:
    - raw states
    - embeddings
    - лишние детали

    ✅ Сохраняет:
    - metadata (action, tier, confidence)
    - errors (signed, abs, direction_match)
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.history: List[Dict] = []

    def record(
        self,
        user_id: str,
        action_type: str,
        tier: str,  # "ML", "Clusters", "Rules"
        confidence: float,
        errors: Dict[str, Dict[str, float]],
        risk_flag_triggered: bool = False
    ):
        """
        Записывает результат прогнозирования.

        Args:
            user_id: ID пользователя
            action_type: Тип действия
            tier: Какой tier был использован
            confidence: Confidence от tier
            errors: Результат self_eval_comparator.compare_forecast()
            risk_flag_triggered: Были ли активированы risk flags
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "action_type": action_type,
            "tier": tier,
            "confidence": round(confidence, 4),
            "errors": errors,
            "risk_flag_triggered": risk_flag_triggered
        }

        self.history.append(record)

        # Ограничиваем размер истории
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_recent(
        self,
        action_type: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Получает последние записи.

        Args:
            action_type: Фильтр по action type
            tier: Фильтр по tier
            limit: Максимум записей
        """
        records = self.history

        # Фильтруем
        if action_type:
            records = [r for r in records if r["action_type"] == action_type]

        if tier:
            records = [r for r in records if r["tier"] == tier]

        # Сортируем по timestamp (новые сначала)
        records = sorted(records, key=lambda x: x["timestamp"], reverse=True)

        return records[:limit]

    def get_metrics_for_key(
        self,
        action_type: str,
        tier: str,
        limit: int = 50
    ) -> Dict[str, float]:
        """
        Вычисляет метрики для конкретного (action_type, tier).

        Returns:
            {
                "direction_acc": 0.75,
                "mae": 0.12,
                "bias": -0.03,
                "sample_count": 45
            }
        """
        records = self.get_recent(action_type=action_type, tier=tier, limit=limit)

        if not records:
            return {
                "direction_acc": 0.0,
                "mae": 0.0,
                "bias": 0.0,
                "sample_count": 0
            }

        # Агрегируем по всем записям
        all_abs_errors = []
        all_signed_errors = []
        all_direction_matches = []

        for record in records:
            for dim_err in record["errors"].values():
                all_abs_errors.append(dim_err["abs"])
                all_signed_errors.append(dim_err["signed"])
                all_direction_matches.append(1 if dim_err["direction_match"] else 0)

        n = len(all_abs_errors)
        if n == 0:
            return {
                "direction_acc": 0.0,
                "mae": 0.0,
                "bias": 0.0,
                "sample_count": 0
            }

        direction_acc = sum(all_direction_matches) / n
        mae = sum(all_abs_errors) / n
        bias = sum(all_signed_errors) / n

        return {
            "direction_acc": round(direction_acc, 4),
            "mae": round(mae, 4),
            "bias": round(bias, 4),
            "sample_count": n
        }

    def get_critical_misalignment_rate(
        self,
        action_type: Optional[str] = None,
        tier: Optional[str] = None,
        limit: int = 100
    ) -> float:
        """
        Вычисляет Critical Misalignment Rate (CMR).

        CMR = % случаев где:
        - direction ошибочен
        - AND risk_flag был активирован

        Это метрика безопасности - ошибки с последствиями.
        """
        records = self.get_recent(action_type=action_type, tier=tier, limit=limit)

        if not records:
            return 0.0

        critical_misalignments = 0
        total = 0

        for record in records:
            # Проверяем каждую размерность
            for dim_err in record["errors"].values():
                total += 1

                # Critical misalignment:
                # - direction wrong
                # - AND risk flag was triggered
                if not dim_err["direction_match"] and record.get("risk_flag_triggered", False):
                    critical_misalignments += 1

        if total == 0:
            return 0.0

        cmr = critical_misalignments / total
        return round(cmr, 4)

    def compute_tier_regret(
        self,
        action_type: str,
        used_tier: str,
        limit: int = 50
    ) -> Dict[str, float]:
        """
        Вычисляет Tier Regret.

        Regret = error(used_tier) - min(error(other_tiers))

        Показывает: насколько fallback был бы лучше.
        """
        # Получаем метрики для использованного tier
        used_metrics = self.get_metrics_for_key(action_type, used_tier, limit)

        if used_metrics["sample_count"] < 5:
            # Недостаточно данных
            return {"regret": 0.0, "better_alternative": None}

        used_mae = used_metrics["mae"]

        # Получаем метрики для других tiers
        tiers = ["ML", "Clusters", "Rules"]
        other_tiers = [t for t in tiers if t != used_tier]

        best_alternative = None
        best_mae = float('inf')

        for other_tier in other_tiers:
            other_metrics = self.get_metrics_for_key(action_type, other_tier, limit)

            if other_metrics["sample_count"] >= 5:
                if other_metrics["mae"] < best_mae:
                    best_mae = other_metrics["mae"]
                    best_alternative = other_tier

        if best_alternative is None:
            return {"regret": 0.0, "better_alternative": None}

        regret = used_mae - best_mae

        return {
            "regret": round(regret, 4),
            "better_alternative": best_alternative,
            "used_mae": round(used_mae, 4),
            "alternative_mae": round(best_mae, 4)
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

emotional_error_store = EmotionalErrorStore()
