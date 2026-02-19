"""
CONFIDENCE CALIBRATOR
Самокалибровка доверия к себе.

ЗОЛОТОЕ ПРАВИЛО:
Confidence ≠ accuracy напрямую.
Confidence — доверие, не истина.

Калибровка:
calibrated_confidence =
    raw_confidence
    * clamp(direction_accuracy / 0.7, 0.5, 1.2)
    * clamp(0.15 / mae, 0.5, 1.1)

Ограничения:
- никогда < 0.3
- никогда > 0.9
- изменения медленные (EMA)
"""

from typing import Dict, Optional


class ConfidenceCalibrator:
    """
    Калибрует confidence на основе исторической точности.

    Принципы:
    1. Если system systematically overconfident → decrease confidence
    2. Если system systematically underconfident → increase confidence
    3. Никогда не extreme values (min=0.3, max=0.9)
    4. Медленные изменения (EMA alpha=0.1)
    """

    # Калибровочные константы
    MIN_CONFIDENCE = 0.3
    MAX_CONFIDENCE = 0.9
    TARGET_DIRECTION_ACC = 0.7  # Если DA < 0.7 → снижаем confidence
    TARGET_MAE = 0.15            # Если MAE > 0.15 → снижаем confidence
    EMA_ALPHA = 0.1              # Медленная адаптация

    # Кэш калиброванных значений
    calibration_cache: Dict[str, float] = {}

    def adjust(
        self,
        raw_confidence: float,
        action_type: str,
        tier: str,
        metrics: Dict[str, float]
    ) -> float:
        """
        Калибрует confidence на основе исторических метрик.

        Args:
            raw_confidence: Исходный confidence от tier
            action_type: Тип действия
            tier: Tier (ML/Clusters/Rules)
            metrics: {direction_acc, mae, bias, sample_count}

        Returns:
            Калиброванный confidence
        """
        if metrics["sample_count"] < 10:
            # Недостаточно данных → не калибруем
            return raw_confidence

        # Получаем закэшированное значение (или используем raw как starting point)
        cache_key = f"{action_type}_{tier}"
        cached_calibrated = self.calibration_cache.get(cache_key, raw_confidence)

        # Фактор 1: Direction Accuracy
        # Если DA < TARGET → понижаем
        da_factor = metrics["direction_acc"] / self.TARGET_DIRECTION_ACC
        da_factor = max(0.5, min(1.2, da_factor))  # clamp [0.5, 1.2]

        # Фактор 2: MAE
        # Если MAE > TARGET → понижаем
        if metrics["mae"] > 0:
            mae_factor = self.TARGET_MAE / metrics["mae"]
            mae_factor = max(0.5, min(1.1, mae_factor))  # clamp [0.5, 1.1]
        else:
            mae_factor = 1.0

        # Вычисляем calibrated
        calibrated = raw_confidence * da_factor * mae_factor

        # EMA с закэшированным значением (медленная адаптация)
        alpha = self.EMA_ALPHA
        smoothed = alpha * calibrated + (1 - alpha) * cached_calibrated

        # Ограничиваем [0.3, 0.9]
        final_confidence = max(self.MIN_CONFIDENCE, min(self.MAX_CONFIDENCE, smoothed))

        # Сохраняем в кэш
        self.calibration_cache[cache_key] = final_confidence

        return round(final_confidence, 4)

    def get_confidence_calibration_error(
        self,
        error_store,
        action_type: str,
        tier: str,
        limit: int = 50
    ) -> float:
        """
        Вычисляет Confidence Calibration Error (CCE).

        CCE = |observed_direction_accuracy - stated_confidence|

        Если система говорит "я уверена на 0.7" → она должна быть права в ~70% случаев.
        """
        records = error_store.get_recent(action_type=action_type, tier=tier, limit=limit)

        if not records:
            return 0.0

        # Бинируем по confidence
        confidence_bins = {
            "low": [],      # < 0.4
            "medium": [],   # 0.4 - 0.6
            "high": []      # > 0.6
        }

        for record in records:
            conf = record["confidence"]

            # Считаем direction accuracy для этого record
            directions = [
                1 if err["direction_match"] else 0
                for err in record["errors"].values()
            ]
            acc = sum(directions) / len(directions) if directions else 0.5

            if conf < 0.4:
                confidence_bins["low"].append((conf, acc))
            elif conf < 0.6:
                confidence_bins["medium"].append((conf, acc))
            else:
                confidence_bins["high"].append((conf, acc))

        # Вычисляем CCE для каждого бина
        cce_values = []

        for bin_name, bin_data in confidence_bins.items():
            if not bin_data:
                continue

            # Средняя accuracy в бине
            avg_acc = sum(item[1] for item in bin_data) / len(bin_data)
            # Средний stated confidence в бине
            avg_conf = sum(item[0] for item in bin_data) / len(bin_data)

            cce = abs(avg_acc - avg_conf)
            cce_values.append(cce)

        if not cce_values:
            return 0.0

        # Средний CCE по всем бинам
        return round(sum(cce_values) / len(cce_values), 4)

    def get_calibration_summary(self) -> Dict:
        """Возвращает summary калибровки."""
        summary = {}

        for key, value in self.calibration_cache.items():
            action_type, tier = key.split("_")
            summary[key] = {
                "action_type": action_type,
                "tier": tier,
                "calibrated_confidence": value
            }

        return summary


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

confidence_calibrator = ConfidenceCalibrator()
