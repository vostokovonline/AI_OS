"""
SYSTEM ALERT GENERATOR
====================================

STEP 2.6 — Alerting & Safeguards

Оценивает критерии и генерирует alerts.
НЕ делает коррекций — ТОЛЬКО повышает awareness.

Принципы:
- alerts ≠ errors
- alerts ≠ corrections
- alerts = awareness signals
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone
from sqlalchemy import text
from database import get_db
from models import SystemAlert
import uuid


# =============================================================================
# КОНСТАНТЫ (диагностические, НЕ operational)
# =============================================================================

class AlertThresholds:
    """Пороги для генерации alerts."""

    # ML Underperforming
    MARGIN = 0.10                    # 10% difference between ML and Rules
    MIN_SAMPLES_ML = 20
    ML_WINDOW = 50

    # Confidence Miscalibration
    CALIBRATION_GAP = 0.15          # 15% gap between stated and observed
    HIGH_CONF = 0.70
    MIN_SAMPLES_CAL = 30

    # High Arousal Blindness
    BASELINE_ACC = 0.60
    DROP = 0.10                     # 10% worse than baseline
    HIGH_AROUSAL = 0.75
    MIN_SAMPLES_AROUSAL = 15

    # Tier Reliability Drift
    DRIFT_THRESHOLD = 0.15          # 15% change
    WINDOW_CURRENT = 30
    WINDOW_HISTORICAL = 100


# =============================================================================
# ALERT GENERATOR
# =============================================================================

class AlertGenerator:
    """
    Генерирует system alerts на основе truth surface данных.

    НЕ делает коррекций — ТОЛЬКО сообщает о проблемах.
    """

    def check_and_generate_alerts(self) -> List[SystemAlert]:
        """
        Проверяет все критерии и генерирует alerts.

        Returns:
            List[SystemAlert] — список новых alerts
        """
        alerts = []

        # 1. ML Underperforming Alert
        ml_alert = self._check_ml_underperforming()
        if ml_alert:
            alerts.append(ml_alert)

        # 2. Confidence Miscalibration Alert
        conf_alert = self._check_confidence_miscalibration()
        if conf_alert:
            alerts.append(conf_alert)

        # 3. High Arousal Blindness Alert
        arousal_alert = self._check_high_arousal_blindness()
        if arousal_alert:
            alerts.append(arousal_alert)

        # 4. Tier Reliability Drift Alert
        drift_alert = self._check_tier_reliability_drift()
        if drift_alert:
            alerts.append(drift_alert)

        # Сохраняем alerts в DB
        for alert in alerts:
            self._persist_alert(alert)

        return alerts

    def _check_ml_underperforming(self) -> Optional[SystemAlert]:
        """
        1️⃣ ML Underperforming Alert

        ML системно хуже Rules по direction accuracy.
        """
        db = next(get_db())

        try:
            # Получаем данные из truth surface
            query = text("""
                SELECT
                    used_tier,
                    AVG(direction_accuracy) AS avg_direction_accuracy,
                    COUNT(*) AS sample_count
                FROM v_forecast_outcome_joined
                WHERE outcome_id IS NOT NULL
                GROUP BY used_tier
            """)

            result = db.execute(query)
            rows = result.fetchall()

            # Извлекаем ML и Rules метрики
            ml_data = None
            rules_data = None

            for row in rows:
                tier = row[0]
                if tier == "ML" and row[2] >= AlertThresholds.MIN_SAMPLES_ML:
                    ml_data = {"direction_accuracy": float(row[1]), "sample_count": row[2]}
                elif tier == "Rules" and row[2] >= AlertThresholds.MIN_SAMPLES_ML:
                    rules_data = {"direction_accuracy": float(row[1]), "sample_count": row[2]}

            # Проверяем условие
            if ml_data and rules_data:
                margin = rules_data["direction_accuracy"] - ml_data["direction_accuracy"]

                if margin >= AlertThresholds.MARGIN:
                    # ML хуже Rules — генерируем alert
                    severity = "WARNING" if margin < 0.20 else "CRITICAL"

                    return SystemAlert(
                        id=uuid.uuid4(),
                        alert_type="ml_underperforming",
                        severity=severity,
                        trigger_data={
                            "metric_name": "direction_accuracy",
                            "ml_value": ml_data["direction_accuracy"],
                            "rules_value": rules_data["direction_accuracy"],
                            "margin": round(margin, 4),
                            "threshold": AlertThresholds.MARGIN,
                            "ml_sample_count": ml_data["sample_count"],
                            "rules_sample_count": rules_data["sample_count"]
                        },
                        explanation=(
                            f"ML direction accuracy {ml_data['direction_accuracy']:.3f} < "
                            f"Rules {rules_data['direction_accuracy']:.3f} "
                            f"(margin {margin:.3f}) in last {ml_data['sample_count'] + rules_data['sample_count']} forecasts"
                        ),
                        context={
                            "window_size": AlertThresholds.ML_WINDOW,
                            "time_period": f"last {AlertThresholds.ML_WINDOW} forecasts",
                            "affected_tiers": ["ML"]
                        },
                        resolved=False,
                        created_at=datetime.now(timezone.utc)
                    )

        finally:
            db.close()

        return None

    def _check_confidence_miscalibration(self) -> Optional[SystemAlert]:
        """
        2️⃣ Confidence Miscalibration Alert

        Высокая уверенность не соответствует фактической точности.
        """
        db = next(get_db())

        try:
            # Агрегируем по всем forecasts с outcome
            query = text("""
                SELECT
                    AVG(forecast_confidence) AS avg_stated_confidence,
                    AVG(CASE WHEN direction_match_count >= 3 THEN 1.0 ELSE 0.0 END) AS observed_accuracy,
                    COUNT(*) AS sample_count
                FROM v_forecast_outcome_joined
                WHERE outcome_id IS NOT NULL
            """)

            result = db.execute(query)
            row = result.fetchone()

            if row and row[2] >= AlertThresholds.MIN_SAMPLES_CAL:
                stated_confidence = float(row[0]) if row[0] else 0.0
                observed_accuracy = float(row[1]) if row[1] else 0.0
                sample_count = row[2]

                # Проверяем условие
                if stated_confidence >= AlertThresholds.HIGH_CONF:
                    calibration_gap = stated_confidence - observed_accuracy

                    if calibration_gap >= AlertThresholds.CALIBRATION_GAP:
                        severity = "WARNING" if calibration_gap < 0.25 else "CRITICAL"

                        return SystemAlert(
                            id=uuid.uuid4(),
                            alert_type="confidence_miscalibration",
                            severity=severity,
                            trigger_data={
                                "metric_name": "confidence_calibration",
                                "stated_confidence": round(stated_confidence, 4),
                                "observed_accuracy": round(observed_accuracy, 4),
                                "calibration_gap": round(calibration_gap, 4),
                                "threshold": AlertThresholds.CALIBRATION_GAP,
                                "sample_count": sample_count
                            },
                            explanation=(
                                f"Stated confidence {stated_confidence:.3f} ≠ "
                                f"observed accuracy {observed_accuracy:.3f} "
                                f"(gap {calibration_gap:.3f}) in {sample_count} forecasts"
                            ),
                            context={
                                "window_size": sample_count,
                                "time_period": f"last {sample_count} forecasts"
                            },
                            resolved=False,
                            created_at=datetime.now(timezone.utc)
                        )

        finally:
            db.close()

        return None

    def _check_high_arousal_blindness(self) -> Optional[SystemAlert]:
        """
        3️⃣ High Arousal Blindness Alert

        При высоком arousal система системно ошибается.
        """
        db = next(get_db())

        try:
            # Проверяем forecasts с высоким baseline arousal
            # Это упрощённая проверка — используем predicted как proxy
            query = text("""
                SELECT
                    AVG(CASE WHEN direction_match_count >= 3 THEN 1.0 ELSE 0.0 END) AS direction_accuracy,
                    COUNT(*) AS sample_count
                FROM v_forecast_outcome_joined
                WHERE outcome_id IS NOT NULL
                  AND (predicted_deltas->>'arousal')::float >= :HIGH_AROUSAL
                GROUP BY (predicted_deltas->>'arousal')::float >= :HIGH_AROUSAL
                HAVING COUNT(*) >= :MIN_SAMPLES
            """)

            result = db.execute(query, {
                "HIGH_AROUSAL": AlertThresholds.HIGH_AROUSAL,
                "MIN_SAMPLES": AlertThresholds.MIN_SAMPLES_AROUSAL
            })

            row = result.fetchone()

            if row:
                direction_accuracy = float(row[0]) if row[0] else 0.0
                sample_count = row[1]

                # Проверяем условие
                if direction_accuracy < (AlertThresholds.BASELINE_ACC - AlertThresholds.DROP):
                    drop = AlertThresholds.BASELINE_ACC - direction_accuracy
                    severity = "WARNING" if drop < 0.20 else "CRITICAL"

                    return SystemAlert(
                        id=uuid.uuid4(),
                        alert_type="high_arousal_blindness",
                        severity=severity,
                        trigger_data={
                            "metric_name": "direction_accuracy",
                            "current_value": round(direction_accuracy, 4),
                            "threshold": round(AlertThresholds.BASELINE_ACC - AlertThresholds.DROP, 4),
                            "baseline": AlertThresholds.BASELINE_ACC,
                            "drop": round(drop, 4),
                            "high_arousal_threshold": AlertThresholds.HIGH_AROUSAL,
                            "sample_count": sample_count
                        },
                        explanation=(
                            f"Direction accuracy {direction_accuracy:.3f} at high arousal "
                            f"(baseline {AlertThresholds.BASELINE_ACC:.2f}, drop {drop:.3f}) "
                            f"in {sample_count} forecasts"
                        ),
                        context={
                            "window_size": sample_count,
                            "high_arousal_threshold": AlertThresholds.HIGH_AROUSAL,
                            "time_period": f"last {sample_count} forecasts"
                        },
                        resolved=False,
                        created_at=datetime.now(timezone.utc)
                    )

        finally:
            db.close()

        return None

    def _check_tier_reliability_drift(self) -> Optional[SystemAlert]:
        """
        4️⃣ Tier Reliability Drift Alert

        Надёжность tier резко изменилась за период.
        """
        # Для этого alert нужно сравнивать текущий период с историческим
        # Это более сложный запрос — упрощённая версия
        db = next(get_db())

        try:
            # Получаем среднюю direction accuracy по всем tier
            query = text("""
                SELECT
                    used_tier,
                    AVG(direction_accuracy) AS avg_accuracy,
                    COUNT(*) AS sample_count
                FROM v_forecast_outcome_joined
                WHERE outcome_id IS NOT NULL
                GROUP BY used_tier
                HAVING COUNT(*) >= :MIN_SAMPLES
            """)

            result = db.execute(query, {"MIN_SAMPLES": AlertThresholds.WINDOW_CURRENT})
            rows = result.fetchall()

            # Упрощённая проверка: если какой-то tier аномально плохой
            for row in rows:
                tier = row[0]
                accuracy = float(row[1]) if row[1] else 0.0
                sample_count = row[2]

                # Если accuracy сильно ниже baseline
                if accuracy < (AlertThresholds.BASELINE_ACC - AlertThresholds.DRIFT_THRESHOLD):
                    severity = "WARNING" if accuracy > 0.4 else "CRITICAL"

                    return SystemAlert(
                        id=uuid.uuid4(),
                        alert_type="tier_reliability_drift",
                        severity=severity,
                        trigger_data={
                            "metric_name": "direction_accuracy",
                            "current_value": round(accuracy, 4),
                            "tier": tier,
                            "baseline": AlertThresholds.BASELINE_ACC,
                            "drift": round(AlertThresholds.BASELINE_ACC - accuracy, 4),
                            "threshold": AlertThresholds.DRIFT_THRESHOLD,
                            "sample_count": sample_count
                        },
                        explanation=(
                            f"Tier {tier} direction accuracy {accuracy:.3f} "
                            f"significantly below baseline {AlertThresholds.BASELINE_ACC:.2f} "
                            f"(drift {AlertThresholds.BASELINE_ACC - accuracy:.3f}) "
                            f"in {sample_count} forecasts"
                        ),
                        context={
                            "window_size": sample_count,
                            "affected_tiers": [tier],
                            "baseline_accuracy": AlertThresholds.BASELINE_ACC
                        },
                        resolved=False,
                        created_at=datetime.now(timezone.utc)
                    )

        finally:
            db.close()

        return None

    def _persist_alert(self, alert: SystemAlert):
        """Сохраняет alert в DB."""
        db = next(get_db())

        try:
            db.add(alert)
            db.commit()
            db.refresh(alert)

            logger.info(f"🚨 [ALERT] {alert.alert_type}: {alert.explanation}")

        except Exception as e:
            logger.info(f"⚠️  [ALERT] Failed to persist alert: {e}")
            db.rollback()

        finally:
            db.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

alert_generator = AlertGenerator()
