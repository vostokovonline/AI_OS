"""
EMOTIONAL TRUTH SURFACE
====================================

STEP 2.5.2 — Aggregated Truth Views (Read-Only)

Python wrapper для SQL views.
Показывает правду о системе, НЕ меняя поведение.
"""

from typing import Dict, List, Optional
from sqlalchemy import text
from database import get_db


class EmotionalTruthSurface:
    """
    Read-only интерфейс к truth views.

    Все методы ТОЛЬКО читают данные, НЕ влияют на inference.
    """

    def get_tier_error_distribution(self, limit: int = 50) -> List[Dict]:
        """
        Возвращает распределение ошибок по tier и action_type.

        Ответ на вопрос: "Где ML врёт чаще правил?"
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    used_tier,
                    action_type,
                    total_forecasts,
                    completed_outcomes,
                    direction_accuracy,
                    avg_mae,
                    avg_forecast_confidence,
                    success_count,
                    failure_count,
                    aborted_count,
                    wrong_direction_rate,
                    overconfidence_errors,
                    underconfidence_count
                FROM v_tier_error_distribution
                WHERE completed_outcomes >= 3
                ORDER BY wrong_direction_rate DESC, avg_mae DESC
                LIMIT :limit
            """)

            result = db.execute(query, {"limit": limit})
            rows = result.fetchall()

            return [
                {
                    "used_tier": row[0],
                    "action_type": row[1],
                    "total_forecasts": row[2],
                    "completed_outcomes": row[3],
                    "direction_accuracy": float(row[4]) if row[4] else 0.0,
                    "avg_mae": float(row[5]) if row[5] else 0.0,
                    "avg_forecast_confidence": float(row[6]) if row[6] else 0.0,
                    "success_count": row[7],
                    "failure_count": row[8],
                    "aborted_count": row[9],
                    "wrong_direction_rate": float(row[10]) if row[10] else 0.0,
                    "overconfidence_errors": row[11],
                    "underconfidence_count": row[12],
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_confidence_accuracy_curve(self) -> List[Dict]:
        """
        Возвращает confidence calibration curve.

        Показывает: насколько честно система оценивает свою уверенность.
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    confidence_bin,
                    sample_count,
                    stated_confidence,
                    observed_direction_accuracy,
                    calibration_error,
                    avg_mae,
                    ml_count,
                    clusters_count,
                    rules_count
                FROM v_confidence_accuracy_curve
                WHERE sample_count >= 5
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

            return [
                {
                    "confidence_bin": row[0],
                    "sample_count": row[1],
                    "stated_confidence": float(row[2]) if row[2] else 0.0,
                    "observed_accuracy": float(row[3]) if row[3] else 0.0,
                    "calibration_error": float(row[4]) if row[4] else 0.0,
                    "avg_mae": float(row[5]) if row[5] else 0.0,
                    "ml_count": row[6],
                    "clusters_count": row[7],
                    "rules_count": row[8],
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_arousal_bucket_analysis(self) -> List[Dict]:
        """
        Возвращает анализ ошибок по уровням arousal.

        Показывает: "high arousal blindness" — системные ошибки при высоком возбуждении.
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    arousal_bucket,
                    sample_count,
                    direction_accuracy,
                    avg_mae,
                    wrong_direction_rate,
                    high_arousal_errors,
                    ml_count,
                    rules_count
                FROM v_arousal_bucket_analysis
                WHERE sample_count >= 3
                ORDER BY wrong_direction_rate DESC
            """)

            result = db.execute(query)
            rows = result.fetchall()

            return [
                {
                    "arousal_bucket": row[0],
                    "sample_count": row[1],
                    "direction_accuracy": float(row[2]) if row[2] else 0.0,
                    "avg_mae": float(row[3]) if row[3] else 0.0,
                    "wrong_direction_rate": float(row[4]) if row[4] else 0.0,
                    "high_arousal_errors": row[5],
                    "ml_count": row[6],
                    "rules_count": row[7],
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_error_evolution(self, days: int = 30) -> List[Dict]:
        """
        Возвращает эволюцию ошибок во времени.

        Показывает тренды качества прогнозов.
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    forecast_date,
                    total_forecasts,
                    completed_outcomes,
                    direction_accuracy,
                    avg_mae,
                    ml_completed,
                    clusters_completed,
                    rules_completed,
                    ml_direction_accuracy,
                    rules_direction_accuracy
                FROM v_error_evolution_daily
                ORDER BY forecast_date DESC
                LIMIT :days
            """)

            result = db.execute(query, {"days": days})
            rows = result.fetchall()

            return [
                {
                    "forecast_date": str(row[0]),
                    "total_forecasts": row[1],
                    "completed_outcomes": row[2],
                    "direction_accuracy": float(row[3]) if row[3] else 0.0,
                    "avg_mae": float(row[4]) if row[4] else 0.0,
                    "ml_completed": row[5],
                    "clusters_completed": row[6],
                    "rules_completed": row[7],
                    "ml_direction_accuracy": float(row[8]) if row[8] else 0.0,
                    "rules_direction_accuracy": float(row[9]) if row[9] else 0.0,
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_action_type_heatmap(self) -> List[Dict]:
        """
        Возвращает heatmap ошибок по action_type.

        Показывает какие действия самые проблемные.
        """
        db = next(get_db())

        try:
            query = text("""
                SELECT
                    action_type,
                    total_forecasts,
                    completed_outcomes,
                    overall_direction_accuracy,
                    overall_mae,
                    ml_direction_accuracy,
                    ml_count,
                    rules_direction_accuracy,
                    rules_count,
                    ml_underperforms_rules
                FROM v_action_type_heatmap
                WHERE completed_outcomes >= 3
                ORDER BY overall_direction_accuracy ASC, overall_mae DESC
            """)

            result = db.execute(query)
            rows = result.fetchall()

            return [
                {
                    "action_type": row[0],
                    "total_forecasts": row[1],
                    "completed_outcomes": row[2],
                    "overall_direction_accuracy": float(row[3]) if row[3] else 0.0,
                    "overall_mae": float(row[4]) if row[4] else 0.0,
                    "ml_direction_accuracy": float(row[5]) if row[5] else 0.0,
                    "ml_count": row[6],
                    "rules_direction_accuracy": float(row[7]) if row[7] else 0.0,
                    "rules_count": row[8],
                    "ml_underperforms_rules": bool(row[9]),
                }
                for row in rows
            ]

        finally:
            db.close()

    def get_ml_vs_rules_summary(self) -> Dict:
        """
        Агрегированное сравнение ML vs Rules.

        Ответ на вопрос: "Когда ML хуже правил?"
        """
        tier_data = self.get_tier_error_distribution(limit=100)

        ml_data = [d for d in tier_data if d["used_tier"] == "ML"]
        rules_data = [d for d in tier_data if d["used_tier"] == "Rules"]

        if not ml_data or not rules_data:
            return {
                "error": "Insufficient data for comparison"
            }

        # Агрегируем по всем action_types
        ml_avg_dir_acc = sum(d["direction_accuracy"] * d["completed_outcomes"] for d in ml_data) / max(1, sum(d["completed_outcomes"] for d in ml_data))
        ml_avg_mae = sum(d["avg_mae"] * d["completed_outcomes"] for d in ml_data) / max(1, sum(d["completed_outcomes"] for d in ml_data))

        rules_avg_dir_acc = sum(d["direction_accuracy"] * d["completed_outcomes"] for d in rules_data) / max(1, sum(d["completed_outcomes"] for d in rules_data))
        rules_avg_mae = sum(d["avg_mae"] * d["completed_outcomes"] for d in rules_data) / max(1, sum(d["completed_outcomes"] for d in rules_data))

        return {
            "ml": {
                "avg_direction_accuracy": round(ml_avg_dir_acc, 4),
                "avg_mae": round(ml_avg_mae, 4),
                "total_outcomes": sum(d["completed_outcomes"] for d in ml_data)
            },
            "rules": {
                "avg_direction_accuracy": round(rules_avg_dir_acc, 4),
                "avg_mae": round(rules_avg_mae, 4),
                "total_outcomes": sum(d["completed_outcomes"] for d in rules_data)
            },
            "comparison": {
                "ml_better_by_accuracy": ml_avg_dir_acc - rules_avg_dir_acc,
                "ml_better_by_mae": rules_avg_mae - ml_avg_mae,  # lower MAE is better
                "ml_underperforms": ml_avg_dir_acc < rules_avg_dir_acc
            }
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

emotional_truth_surface = EmotionalTruthSurface()
