"""
EMOTIONAL CONFIDENCE HONESTY METRICS
====================================

STEP 2.5.3 — Confidence Honesty (User-Facing, Read-Only)

Подготавливает данные для ответа на вопрос:
"насколько я уверена и насколько часто была права"

⚠️ НЕ меняет поведение системы.
"""

from typing import Dict, List, Optional, Tuple
from sqlalchemy import text
from database import get_db


class ConfidenceHonestyMetrics:
    """
    User-facing метрики честности confidence.

    Показывает пользователю насколько системе можно доверять.
    """

    def get_overall_honesty_summary(self) -> Dict:
        """
        Общая сводка честности системы.

        Returns:
            {
                "total_forecasts": int,
                "completed_outcomes": int,
                "overall_direction_accuracy": float,
                "stated_vs_actual_confidence": {
                    "stated_mean": float,
                    "observed_accuracy": float,
                    "calibration_error": float,
                    "honesty_score": str  # "honest" | "overconfident" | "underconfident"
                }
            }
        """
        db = next(get_db())

        try:
            # Агрегируем по всем outcomes
            query = text("""
                SELECT
                    COUNT(*) AS total_forecasts,
                    COUNT(o.id) AS completed_outcomes,
                    AVG(CASE WHEN fo.direction_match_count >= 3 THEN 1.0 ELSE 0.0 END) AS direction_accuracy,
                    AVG(f.forecast_confidence) AS avg_stated_confidence
                FROM emotional_forecasts f
                LEFT JOIN emotional_outcomes o ON o.forecast_id = f.id
            """)

            result = db.execute(query)
            row = result.fetchone()

            if not row or row[1] == 0:  # Нет completed outcomes
                return {
                    "error": "Insufficient data for honesty assessment",
                    "total_forecasts": row[0] if row else 0,
                    "completed_outcomes": 0
                }

            total_forecasts = row[0]
            completed_outcomes = row[1]
            direction_accuracy = float(row[2]) if row[2] else 0.0
            stated_confidence = float(row[3]) if row[3] else 0.5

            # Вычисляем calibration error
            calibration_error = stated_confidence - direction_accuracy

            # Определяем honesty score
            if abs(calibration_error) <= 0.1:
                honesty_score = "honest"
            elif calibration_error > 0.1:
                honesty_score = "overconfident"
            else:
                honesty_score = "underconfident"

            return {
                "total_forecasts": total_forecasts,
                "completed_outcomes": completed_outcomes,
                "overall_direction_accuracy": round(direction_accuracy, 4),
                "stated_vs_actual_confidence": {
                    "stated_mean": round(stated_confidence, 4),
                    "observed_accuracy": round(direction_accuracy, 4),
                    "calibration_error": round(calibration_error, 4),
                    "honesty_score": honesty_score
                }
            }

        finally:
            db.close()

    def get_confidence_band_report(self) -> Dict[str, Dict]:
        """
        Отчёт по confidence bands.

        Показывает как честность меняется по уровням уверенности.

        Returns:
            {
                "[0.0-0.3)": {
                    "stated_confidence": float,
                    "observed_accuracy": float,
                    "sample_count": int,
                    "honesty_score": str
                },
                ...
            }
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    confidence_bin,
                    stated_confidence,
                    observed_direction_accuracy,
                    sample_count
                FROM v_confidence_accuracy_curve
                ORDER BY
                    CASE confidence_bin
                        WHEN '[0.0 - 0.3)' THEN 1
                        WHEN '[0.3 - 0.4)' THEN 2
                        WHEN '[0.4 - 0.5)' THEN 3
                        WHEN '[0.5 - 0.6)' THEN 4
                        WHEN '[0.6 - 0.7)' THEN 5
                        WHEN '[0.7 - 0.8)' THEN 6
                        WHEN '[0.8 - 0.9)' THEN 7
                        ELSE 8
                    END
            """)

            result = db.execute(query)
            rows = result.fetchall()

            report = {}

            for row in rows:
                bin_name = row[0]
                stated = float(row[1]) if row[1] else 0.0
                observed = float(row[2]) if row[2] else 0.0
                count = row[3]

                calibration_error = stated - observed

                if abs(calibration_error) <= 0.1:
                    honesty = "honest"
                elif calibration_error > 0.1:
                    honesty = "overconfident"
                else:
                    honesty = "underconfident"

                report[bin_name] = {
                    "stated_confidence": round(stated, 4),
                    "observed_accuracy": round(observed, 4),
                    "sample_count": count,
                    "calibration_error": round(calibration_error, 4),
                    "honesty_score": honesty
                }

            return report

        finally:
            db.close()

    def get_tier_honesty_comparison(self) -> Dict:
        """
        Сравнение честности между tiers.

        Ответ на вопрос: "Кто честнее — ML или Rules?"
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    used_tier,
                    AVG(forecast_confidence) AS avg_stated_confidence,
                    AVG(CASE WHEN fo.direction_match_count >= 3 THEN 1.0 ELSE 0.0 END) AS direction_accuracy,
                    COUNT(*) AS sample_count
                FROM v_forecast_outcome_joined fo
                WHERE fo.outcome_id IS NOT NULL
                GROUP BY used_tier
            """)

            result = db.execute(query)
            rows = result.fetchall()

            comparison = {}

            for row in rows:
                tier = row[0]
                stated = float(row[1]) if row[1] else 0.0
                accuracy = float(row[2]) if row[2] else 0.0
                count = row[3]

                calibration_error = stated - accuracy

                comparison[tier] = {
                    "stated_confidence": round(stated, 4),
                    "observed_accuracy": round(accuracy, 4),
                    "calibration_error": round(calibration_error, 4),
                    "sample_count": count,
                    "honesty_score": "honest" if abs(calibration_error) <= 0.1 else ("overconfident" if calibration_error > 0.1 else "underconfident")
                }

            return comparison

        finally:
            db.close()

    def get_overconfidence_report(self, min_confidence: float = 0.7) -> List[Dict]:
        """
        Отчёт о случаях overconfidence.

        Когда система была уверена (≥ min_confidence), но ошиблась.
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    fo.forecast_id,
                    fo.action_type,
                    fo.used_tier,
                    fo.forecast_confidence,
                    fo.direction_match_count,
                    fo.mae,
                    fo.outcome,
                    fo.predicted_deltas,
                    fo.actual_deltas
                FROM v_forecast_outcome_joined fo
                WHERE fo.outcome_id IS NOT NULL
                  AND fo.forecast_confidence >= :min_confidence
                  AND fo.direction_match_count <= 1
                ORDER BY fo.forecast_confidence DESC, fo.mae DESC
                LIMIT 50
            """)

            result = db.execute(query, {"min_confidence": min_confidence})
            rows = result.fetchall()

            return [
                {
                    "forecast_id": str(row[0]),
                    "action_type": row[1],
                    "tier": row[2],
                    "confidence": float(row[3]),
                    "direction_matches": row[4],
                    "mae": float(row[5]),
                    "outcome": row[6],
                    "predicted_delta": row[7],
                    "actual_delta": row[8],
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_underconfidence_report(self, max_confidence: float = 0.4) -> List[Dict]:
        """
        Отчёт о случаях underconfidence.

        Когда система была неуверена (≤ max_confidence), но оказалась права.
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    fo.forecast_id,
                    fo.action_type,
                    fo.used_tier,
                    fo.forecast_confidence,
                    fo.direction_match_count,
                    fo.mae,
                    fo.outcome
                FROM v_forecast_outcome_joined fo
                WHERE fo.outcome_id IS NOT NULL
                  AND fo.forecast_confidence <= :max_confidence
                  AND fo.direction_match_count >= 3
                  AND fo.mae <= 0.10
                ORDER BY fo.mae ASC, fo.forecast_confidence ASC
                LIMIT 50
            """)

            result = db.execute(query, {"max_confidence": max_confidence})
            rows = result.fetchall()

            return [
                {
                    "forecast_id": str(row[0]),
                    "action_type": row[1],
                    "tier": row[2],
                    "confidence": float(row[3]),
                    "direction_matches": row[4],
                    "mae": float(row[5]),
                    "outcome": row[6],
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_trust_score(self) -> Dict:
        """
        Вычисляет aggregate trust score.

        Комбинирует несколько метрик в одну оценку доверия.

        Returns:
            {
                "trust_score": float (0..1),
                "components": {
                    "direction_accuracy": float,
                    "confidence_calibration": float,
                    "consistency": float
                },
                "interpretation": str  — объяснение для пользователя
            }
        """
        summary = self.get_overall_honesty_summary()

        if "error" in summary:
            return {
                "trust_score": 0.5,  # Нейтральный
                "components": {},
                "interpretation": "Insufficient data for trust assessment"
            }

        # Components
        direction_acc = summary["overall_direction_accuracy"]
        cal_error = abs(summary["stated_vs_actual_confidence"]["calibration_error"])

        # Direction accuracy component (0..1)
        acc_component = direction_acc

        # Calibration component (0..1, 1 = perfectly calibrated)
        cal_component = max(0.0, 1.0 - cal_error * 2)  # Штраф за калибровку

        # Consistency component (0..1)
        # Используем количество данных как proxy для consistency
        data_component = min(1.0, summary["completed_outcomes"] / 100.0)

        # Aggregate trust score
        trust_score = (acc_component * 0.5 + cal_component * 0.3 + data_component * 0.2)

        # Interpretation
        if trust_score >= 0.8:
            interpretation = "High trust — system is consistently accurate and honest about uncertainty"
        elif trust_score >= 0.6:
            interpretation = "Moderate trust — system is generally reliable but has some calibration issues"
        elif trust_score >= 0.4:
            interpretation = "Low trust — system makes significant errors or is poorly calibrated"
        else:
            interpretation = "Very low trust — system is not reliable for decision-making"

        return {
            "trust_score": round(trust_score, 4),
            "components": {
                "direction_accuracy": round(acc_component, 4),
                "confidence_calibration": round(cal_component, 4),
                "consistency": round(data_component, 4)
            },
            "interpretation": interpretation
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

confidence_honesty_metrics = ConfidenceHonestyMetrics()
