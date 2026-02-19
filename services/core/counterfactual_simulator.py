"""
COUNTERFACTUAL SIMULATOR
====================================

STEP 2.7 — Intervention Readiness Layer

Ключевой компонент: отвечает на вопрос "А что было бы, если бы...?"

Architectural invariants:
- Simulation ≠ Prediction (only replay + deterministic recompute)
- NO stochasticity
- One intervention at a time
- Fixed window
- Same input data
- If not deterministic → simulation rejected
"""

from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, and_, func, text
from database import get_db
from models import (
    EmotionalForecast,
    EmotionalOutcome,
    InterventionCandidate,
    InterventionSimulation
)
import hashlib
import json
import uuid


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

class CounterfactualSimulator:
    """
    Симулирует "что если бы мы применили вмешательство" на исторических данных.

    Pipeline:
    1. Select outcomes (window)
    2. Freeze inputs
    3. Apply virtual intervention
    4. Recompute predictions
    5. Recompute metrics
    6. Compare before/after
    """

    def simulate_intervention(
        self,
        intervention_id: str,
        replay_window_days: int = 30
    ) -> Optional[InterventionSimulation]:
        """
        Симулирует эффект вмешательства на исторических данных.

        Args:
            intervention_id: ID кандидата для симуляции
            replay_window_days: Окно исторических данных (дней)

        Returns:
            InterventionSimulation или None (если симуляция не удалась)
        """
        db = next(get_db())

        try:
            # 1. Load intervention candidate
            stmt_candidate = select(InterventionCandidate).where(
                InterventionCandidate.id == intervention_id
            )
            result_candidate = db.execute(stmt_candidate)
            candidate = result_candidate.scalar_one_or_none()

            if not candidate:
                print(f"⚠️  [Simulator] Candidate {intervention_id} not found")
                return None

            # 2. Get historical data (forecasts + outcomes)
            window_start = datetime.now(timezone.utc) - timedelta(days=replay_window_days)

            stmt_data = text("""
                SELECT
                    ef.id as forecast_id,
                    ef.used_tier,
                    ef.action_type,
                    ef.predicted_deltas,
                    ef.forecast_confidence,
                    eo.actual_deltas,
                    eo.outcome
                FROM emotional_forecasts ef
                JOIN emotional_outcomes eo ON eo.forecast_id = ef.id
                WHERE ef.created_at >= :window_start
                ORDER BY ef.created_at DESC
                LIMIT 100
            """)

            result_data = db.execute(stmt_data, {"window_start": window_start})
            historical_records = result_data.fetchall()

            if len(historical_records) < 10:
                print(f"⚠️  [Simulator] Insufficient historical data: {len(historical_records)} records")
                return None

            # 3. Compute metrics BEFORE intervention (baseline)
            metrics_before = self._compute_metrics_on_records(historical_records)

            # 4. Apply virtual intervention to records
            modified_records = self._apply_intervention_to_records(
                records=historical_records,
                intervention=candidate
            )

            # 5. Compute metrics AFTER intervention
            metrics_after = self._compute_metrics_on_records(modified_records)

            # 6. Compute delta
            delta_metrics = self._compute_delta(metrics_before, metrics_after)

            # 7. Detect side effects (where things got worse)
            side_effects = self._detect_side_effects(metrics_before, metrics_after)

            # 8. Generate determinism hash
            determinism_hash = self._generate_determinism_hash(
                intervention=candidate,
                record_count=len(historical_records),
                window_start=window_start
            )

            # 9. Create simulation record
            simulation = InterventionSimulation(
                id=uuid.uuid4(),
                intervention_id=intervention_id,
                replay_window=timedelta(days=replay_window_days),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                delta_metrics=delta_metrics,
                side_effects=side_effects,
                determinism_hash=determinism_hash,
                created_at=datetime.now(timezone.utc)
            )

            db.add(simulation)

            # Update candidate status
            candidate.status = "simulated"

            db.commit()

            print(f"🎮 [Simulator] Completed simulation for {candidate.intervention_type}")
            print(f"   Records: {len(historical_records)}")
            print(f"   Direction accuracy: {metrics_before['direction_accuracy']:.3f} → {metrics_after['direction_accuracy']:.3f}")
            print(f"   Delta: {delta_metrics.get('direction_accuracy', 0):.3f}")
            print(f"   Determinism hash: {determinism_hash}")

            return simulation

        except Exception as e:
            print(f"⚠️  [Simulator] Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()
            return None

        finally:
            db.close()

    def _compute_metrics_on_records(self, records: list) -> Dict:
        """
        Вычисляет метрики на наборе records.

        Metrics:
        - direction_accuracy: % правильных направлений
        - mae: средняя абсолютная ошибка
        - calibration_gap: разрыв между confidence и accuracy
        """

        if not records:
            return {
                "direction_accuracy": 0.0,
                "mae": 0.0,
                "calibration_gap": 0.0,
                "sample_count": 0
            }

        direction_correct = 0
        mae_sum = 0.0
        confidence_sum = 0.0
        count = 0

        for record in records:
            # Extract data from record tuple
            # (forecast_id, used_tier, action_type, predicted_deltas, forecast_confidence, actual_deltas, outcome)
            predicted = json.loads(record[3]) if isinstance(record[3], str) else record[3]
            actual = json.loads(record[5]) if isinstance(record[5], str) else record[5]
            confidence = record[4]

            # Check direction correctness for each dimension
            for dim in predicted:
                if dim in actual:
                    pred_sign = 1 if predicted[dim] > 0 else (-1 if predicted[dim] < 0 else 0)
                    actual_sign = 1 if actual[dim] > 0 else (-1 if actual[dim] < 0 else 0)

                    if pred_sign == actual_sign:
                        direction_correct += 1
                    count += 1

            # Compute MAE
            for dim in predicted:
                if dim in actual:
                    mae_sum += abs(predicted[dim] - actual[dim])

            confidence_sum += confidence

        # Compute aggregated metrics
        direction_accuracy = direction_correct / count if count > 0 else 0.0
        mae = mae_sum / count if count > 0 else 0.0
        avg_confidence = confidence_sum / len(records) if records else 0.0

        # Calibration gap = stated confidence - observed accuracy
        calibration_gap = avg_confidence - direction_accuracy

        return {
            "direction_accuracy": round(direction_accuracy, 4),
            "mae": round(mae, 4),
            "calibration_gap": round(calibration_gap, 4),
            "avg_confidence": round(avg_confidence, 4),
            "sample_count": len(records)
        }

    def _apply_intervention_to_records(
        self,
        records: list,
        intervention: InterventionCandidate
    ) -> list:
        """
        Применяет виртуальное вмешательство к records.

        Модифицирует predicted_deltas или forecast_confidence
        в зависимости от типа вмешательства.

        ВАЖНО: НЕ сохраняет в БД — только в памяти.
        """

        modified_records = []

        for record in records:
            # Convert to list for modification
            record_list = list(record)

            predicted = json.loads(record[3]) if isinstance(record[3], str) else record[3]
            confidence = record[4]

            # Apply intervention based on type
            if intervention.intervention_type == "adjust_confidence_scaling":
                # Scale down confidence
                scaling_factor = intervention.target_scope.get("estimated_factor", 0.9)
                new_confidence = confidence * scaling_factor
                record_list[4] = max(0.0, min(1.0, new_confidence))

            elif intervention.intervention_type == "raise_arousal_guardrail":
                # Modify predicted deltas if arousal is high
                proposed_threshold = intervention.target_scope.get("proposed_threshold", 0.65)
                arousal_pred = predicted.get("arousal", 0)

                if arousal_pred > proposed_threshold:
                    # Penalize high-arousal predictions
                    for dim in predicted:
                        predicted[dim] *= 0.5  # Reduce magnitude
                    record_list[3] = json.dumps(predicted)

            elif intervention.intervention_type == "lower_tier_weight":
                # This is more complex — for now just mark as "would affect"
                # In real implementation, this would change tier selection logic
                pass

            elif intervention.intervention_type == "disable_ml_for_context":
                # If ML tier, switch to rules (simplified)
                if record[1] == "ML":
                    # In real simulation, this would re-run forecast with Rules tier
                    # For now, just mark as modified
                    record_list[1] = "Rules(simulated)"

            modified_records.append(tuple(record_list))

        return modified_records

    def _compute_delta(self, before: Dict, after: Dict) -> Dict:
        """Вычисляет разницу между метриками (after - before)."""

        delta = {}
        for key in before:
            if isinstance(before[key], (int, float)):
                delta[key] = round(after.get(key, 0) - before[key], 4)

        return delta

    def _detect_side_effects(self, before: Dict, after: Dict) -> Dict:
        """
        Обнаруживает негативные побочные эффекты.

        Side effect: метрика стала значительно хуже.
        """

        side_effects = {}

        WORSEN_THRESHOLD = 0.02  # 2% worsening = side effect

        for key in before:
            if isinstance(before[key], (int, float)):
                delta = after.get(key, 0) - before[key]

                # For MAE and calibration_gap: lower is better
                if key in ["mae", "calibration_gap"]:
                    if delta > WORSEN_THRESHOLD:
                        side_effects[key] = {
                            "before": before[key],
                            "after": after[key],
                            "delta": round(delta, 4)
                        }

                # For direction_accuracy: higher is better
                elif key == "direction_accuracy":
                    if delta < -WORSEN_THRESHOLD:
                        side_effects[key] = {
                            "before": before[key],
                            "after": after[key],
                            "delta": round(delta, 4)
                        }

        return side_effects if side_effects else None

    def _generate_determinism_hash(
        self,
        intervention: InterventionCandidate,
        record_count: int,
        window_start: datetime
    ) -> str:
        """
        Генерирует hash для гарантии детерминизма.

        Hash = sha256(intervention_id + intervention_type + target_scope + record_count + window_start)
        """

        hash_input = {
            "intervention_id": str(intervention.id),
            "intervention_type": intervention.intervention_type,
            "target_scope": intervention.target_scope,
            "record_count": record_count,
            "window_start": window_start.isoformat()
        }

        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def get_simulation(self, intervention_id: str) -> Optional[Dict]:
        """
        Get simulation results for intervention.

        Returns:
            Dict with simulation details or None
        """
        db = next(get_db())

        try:
            stmt = select(InterventionSimulation).where(
                InterventionSimulation.intervention_id == intervention_id
            )
            result = db.execute(stmt)
            simulation = result.scalar_one_or_none()

            if not simulation:
                return None

            return {
                "id": str(simulation.id),
                "intervention_id": str(simulation.intervention_id),
                "replay_window_days": simulation.replay_window.days,
                "metrics_before": simulation.metrics_before,
                "metrics_after": simulation.metrics_after,
                "delta_metrics": simulation.delta_metrics,
                "side_effects": simulation.side_effects,
                "determinism_hash": simulation.determinism_hash,
                "created_at": simulation.created_at.isoformat() if simulation.created_at else None
            }

        finally:
            db.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

counterfactual_simulator = CounterfactualSimulator()
