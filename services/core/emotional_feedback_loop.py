"""
Emotional Feedback Loop Module

Закрывает цикл обратной связи между:
1. Goal System (исполнение целей)
2. Emotional Layer (эмоциональное состояние)
3. Результаты (успех/неудача)

Это ключевой модуль для превращения AI-OS из "реактора" в "AI-партнёра"

🆕 STEP 2.3: Self-Evaluation Loop integration
🆕 STEP 2.4: Forecast & Outcome persistence
"""

from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy import select, and_
from database import AsyncSessionLocal
from models import (
    Goal,
    AffectiveMemoryEntry,
    EmotionalLayerState,
    EmotionalForecast,  # 🆕 STEP 2.4
    EmotionalOutcome     # 🆕 STEP 2.4
)
from emotional_layer import emotional_layer
from schemas import EmotionalSignals

# 🆕 Self-evaluation imports
from emotional_self_eval import self_eval_comparator
from emotional_error_store import emotional_error_store
from tier_reliability import tier_reliability_tracker
from confidence_calibrator import confidence_calibrator

# 🆕 STEP 2.6: Alerting imports
from alert_generator import alert_generator

# 🆕 STEP 2.7: Intervention Readiness Layer imports
from intervention_candidates_engine import intervention_candidates_engine
from counterfactual_simulator import counterfactual_simulator
from intervention_risk_scorer import intervention_risk_scorer


class EmotionalFeedbackLoop:
    """
    Управляет эмоциональным циклом обратной связи.

    Flow:
    1. Goal execution → Result (success/failure)
    2. Result → Emotional state update
    3. Emotional transition → Affective Memory
    4. Affective Memory → Future decision influence
    """

    def __init__(self):
        self.influence_weights = {
            "success": {"valence": 0.2, "confidence": 0.1, "focus": 0.05},
            "failure": {"valence": -0.3, "confidence": -0.15, "focus": -0.1},
            "aborted": {"valence": -0.15, "confidence": -0.1, "arousal": -0.1},
            "progress": {"valence": 0.05, "confidence": 0.05, "focus": 0.1},
        }

    async def record_goal_completion(
        self,
        goal_id: str,
        user_id: str,
        outcome: str,  # "success", "failure", "aborted"
        metrics: Optional[Dict] = None
    ):
        """
        Записывает результат выполнения цели и обновляет эмоциональное состояние.

        Это ТОЧКА ВХОДА в feedback loop - вызывается после выполнения цели.

        🆕 STEP 2.4: Читает forecast из DB, сохраняет outcome, выполняет self-eval.
        """
        async with AsyncSessionLocal() as db:
            # Get goal details
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                print(f"⚠️  Goal {goal_id} not found for emotional feedback")
                return

            # Get current emotional state BEFORE update
            stmt_before = select(EmotionalLayerState).where(
                EmotionalLayerState.user_id == user_id
            ).order_by(EmotionalLayerState.created_at.desc()).limit(1)

            result_before = await db.execute(stmt_before)
            state_before = result_before.scalar_one_or_none()

            emotional_state_before = {
                "arousal": state_before.arousal if state_before else 0.5,
                "valence": state_before.valence if state_before else 0.0,
                "focus": state_before.focus if state_before else 0.5,
                "confidence": state_before.confidence if state_before else 0.5,
            } if state_before else None

            # Calculate emotional impact based on outcome
            impact = self._calculate_emotional_impact(outcome, goal, metrics)

            # Create signals for emotional update
            signals = self._create_feedback_signals(goal, outcome, metrics)

            # Get new emotional state (this will save to DB)
            influence = await emotional_layer.get_influence(user_id, signals)

            # Get the NEW state after update
            stmt_after = select(EmotionalLayerState).where(
                EmotionalLayerState.user_id == user_id
            ).order_by(EmotionalLayerState.created_at.desc()).limit(1)

            result_after = await db.execute(stmt_after)
            state_after = result_after.scalar_one_or_none()

            emotional_state_after = {
                "arousal": state_after.arousal if state_after else 0.5,
                "valence": state_after.valence if state_after else 0.0,
                "focus": state_after.focus if state_after else 0.5,
                "confidence": state_after.confidence if state_after else 0.5,
            } if state_after else None

            # Store in Affective Memory
            await self._store_affective_memory(
                db=db,
                user_id=user_id,
                goal_id=goal_id,
                decision_id=goal_id,  # For now, goal_id = decision_id
                emotional_state_before=emotional_state_before,
                emotional_state_after=emotional_state_after,
                outcome=outcome,
                outcome_metrics=metrics or {"impact": impact}
            )

            await db.commit()

            # 🆕 STEP 2.4: Self-Evaluation с persisted forecast
            if goal.forecast_id and emotional_state_before and emotional_state_after:
                try:
                    # 1. Загружаем forecast из DB
                    stmt_forecast = select(EmotionalForecast).where(EmotionalForecast.id == goal.forecast_id)
                    result_forecast = await db.execute(stmt_forecast)
                    forecast = result_forecast.scalar_one_or_none()

                    if forecast:
                        # 2. Вычисляем actual_delta
                        actual_delta = {
                            "arousal": emotional_state_after["arousal"] - emotional_state_before["arousal"],
                            "valence": emotional_state_after["valence"] - emotional_state_before["valence"],
                            "focus": emotional_state_after["focus"] - emotional_state_before["focus"],
                            "confidence": emotional_state_after["confidence"] - emotional_state_before["confidence"],
                        }

                        # 3. Сохраняем outcome в DB
                        outcome_record = EmotionalOutcome(
                            forecast_id=forecast.id,
                            actual_deltas=actual_delta,
                            outcome=outcome
                        )
                        db.add(outcome_record)
                        await db.commit()

                        # 4. Сравниваем прогноз с реальностью
                        predicted_delta = forecast.predicted_deltas
                        errors = self_eval_comparator.compare_forecast(
                            predicted=predicted_delta,
                            actual=actual_delta
                        )

                        # 5. Записываем ошибку в error_store
                        emotional_error_store.record(
                            user_id=user_id,
                            action_type=forecast.action_type,
                            tier=forecast.used_tier,
                            confidence=forecast.forecast_confidence,
                            errors=errors,
                            risk_flag_triggered=bool(forecast.risk_flags)
                        )

                        # 6. Обновляем надежность tiers
                        tier_reliability_tracker.update_reliability(emotional_error_store)

                        print(f"📊 [Self-Eval] Recorded forecast error:")
                        print(f"   Forecast: {forecast.id}")
                        print(f"   Predicted: {predicted_delta}")
                        print(f"   Actual:    {actual_delta}")
                        print(f"   Tier:      {forecast.used_tier}")
                        print(f"   Errors:    {errors}")
                    else:
                        print(f"⚠️  [Self-Eval] Forecast {goal.forecast_id} not found in DB")

                except Exception as e:
                    print(f"⚠️  [Self-Eval] Failed to record error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if not goal.forecast_id:
                    print(f"ℹ️  [Self-Eval] No forecast_id for goal {goal_id}, skipping self-eval")

            # 🆕 STEP 2.6: Alerting — проверяем критерии после каждого outcome
            # НЕ делаем коррекций — ТОЛЬКО сообщаем о проблемах
            try:
                # Запускаем проверку alert criteria
                new_alerts = alert_generator.check_and_generate_alerts()

                if new_alerts:
                    print(f"🚨 [Alerting] Generated {len(new_alerts)} alert(s)")
                    for alert in new_alerts:
                        print(f"   - {alert.alert_type} ({alert.severity}): {alert.explanation}")
                else:
                    print(f"ℹ️  [Alerting] No alerts triggered")

            except Exception as e:
                print(f"⚠️  [Alerting] Failed to check alerts: {e}")
                import traceback
                traceback.print_exc()

            # 🆕 STEP 2.7: Intervention Readiness Layer — генерируем candidates
            # НЕ применяет вмешательства — ТОЛЬКО предлагает гипотезы
            try:
                if new_alerts:
                    # Генерируем candidates только если есть новые alerts
                    new_candidates = intervention_candidates_engine.generate_from_active_alerts()

                    if new_candidates:
                        print(f"💡 [IRL] Generated {len(new_candidates)} intervention candidate(s)")

                        # Для каждого candidate: симуляция + risk scoring
                        for candidate in new_candidates:
                            try:
                                # 1. Run simulation
                                simulation = counterfactual_simulator.simulate_intervention(
                                    intervention_id=str(candidate.id),
                                    replay_window_days=30
                                )

                                if simulation:
                                    # 2. Compute risk score
                                    risk_score = intervention_risk_scorer.compute_risk_score(
                                        intervention_id=str(candidate.id)
                                    )

                                    if risk_score:
                                        print(f"   📊 Candidate: {candidate.intervention_type}")
                                        print(f"      Risk tier: {risk_score.risk_tier} (risk={risk_score.total_risk:.3f})")
                                        print(f"      Δ accuracy: {simulation.delta_metrics.get('direction_accuracy', 0):.3f}")

                            except Exception as e:
                                print(f"⚠️  [IRL] Failed to process candidate {candidate.id}: {e}")
                    else:
                        print(f"ℹ️  [IRL] No candidates generated from alerts")

            except Exception as e:
                print(f"⚠️  [IRL] Failed to generate candidates: {e}")
                import traceback
                traceback.print_exc()

            print(f"📊 [Emotional Feedback] Recorded {outcome} for goal {goal_id}")
            print(f"   Before: {emotional_state_before}")
            print(f"   After:  {emotional_state_after}")
            print(f"   Impact: {impact}")

    def _calculate_emotional_impact(
        self,
        outcome: str,
        goal: Goal,
        metrics: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate how much this outcome should affect emotional state"""

        base_impact = self.influence_weights.get(outcome, {})

        # Adjust based on goal complexity
        if goal.depth_level and goal.depth_level > 0:
            complexity_multiplier = 1.0 + (goal.depth_level * 0.1)
        else:
            complexity_multiplier = 1.0

        # Adjust based on is_atomic (atomic goals have more impact)
        if goal.is_atomic:
            complexity_multiplier *= 1.2

        # Apply complexity multiplier
        impact = {
            k: v * complexity_multiplier
            for k, v in base_impact.items()
        }

        # Add progress if metrics contain progress info
        if metrics and "progress" in metrics:
            progress = metrics["progress"]
            if progress > 0.5:
                impact["valence"] = impact.get("valence", 0) + 0.1
                impact["confidence"] = impact.get("confidence", 0) + 0.05

        return impact

    def _create_feedback_signals(
        self,
        goal: Goal,
        outcome: str,
        metrics: Optional[Dict]
    ) -> EmotionalSignals:
        """Create EmotionalSignals based on goal execution result"""

        # Count goals in last 24h for context
        goal_stats = {
            "outcome_success": 1 if outcome == "success" else 0,
            "outcome_failure": 1 if outcome == "failure" else 0,
            "outcome_aborted": 1 if outcome == "aborted" else 0,
        }

        # System metrics based on outcome
        system_metrics = {
            "recent_success_rate": 0.8 if outcome == "success" else 0.3,
        }

        # User text describes what happened
        user_text = f"Goal '{goal.title}' {outcome}"

        if metrics:
            if "progress" in metrics:
                user_text += f" (progress: {metrics['progress']:.0%})"

        return EmotionalSignals(
            user_text=user_text,
            goal_stats=goal_stats,
            system_metrics=system_metrics
        )

    async def _store_affective_memory(
        self,
        db,
        user_id: str,
        goal_id: str,
        decision_id: str,
        emotional_state_before: Dict,
        emotional_state_after: Dict,
        outcome: str,
        outcome_metrics: Dict
    ):
        """Store emotional transition in Affective Memory"""

        # Calculate emotional delta
        delta = {}
        if emotional_state_before and emotional_state_after:
            for key in emotional_state_before:
                delta[key] = emotional_state_after[key] - emotional_state_before[key]

        memory = AffectiveMemoryEntry(
            user_id=user_id,
            goal_id=goal_id,
            decision_id=decision_id,
            emotional_state_before=emotional_state_before or {},
            emotional_state_after=emotional_state_after or {},
            outcome=outcome,
            outcome_metrics={
                **outcome_metrics,
                "emotional_delta": delta
            }
        )

        db.add(memory)

    async def get_affective_patterns(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get emotional patterns from Affective Memory.

        This is used for learning:
        - What emotional states lead to success?
        - What patterns lead to failure?
        - How does the user emotionally respond to outcomes?
        """

        async with AsyncSessionLocal() as db:
            stmt = select(AffectiveMemoryEntry).where(
                AffectiveMemoryEntry.user_id == user_id
            ).order_by(
                AffectiveMemoryEntry.created_at.desc()
            ).limit(limit)

            result = await db.execute(stmt)
            memories = result.scalars().all()

            patterns = []
            for memory in memories:
                pattern = {
                    "goal_id": str(memory.goal_id) if memory.goal_id else None,
                    "outcome": memory.outcome,
                    "state_before": memory.emotional_state_before,
                    "state_after": memory.emotional_state_after,
                    "delta": memory.outcome_metrics.get("emotional_delta", {}) if memory.outcome_metrics else {},
                    "timestamp": memory.created_at.isoformat() if memory.created_at else None,
                }
                patterns.append(pattern)

            return patterns

    async def analyze_emotional_effectiveness(
        self,
        user_id: str
    ) -> Dict:
        """
        Analyze which emotional states are most effective for this user.

        Returns insights like:
        - "User performs best when focus > 0.7"
        - "High arousal leads to more abortions"
        - "Positive valence correlates with success"
        """

        patterns = await self.get_affective_patterns(user_id, limit=100)

        if not patterns:
            return {"error": "Not enough data"}

        # Analyze correlations
        success_states = [p for p in patterns if p["outcome"] == "success"]
        failure_states = [p for p in patterns if p["outcome"] == "failure"]
        aborted_states = [p for p in patterns if p["outcome"] == "aborted"]

        def avg_state(states):
            if not states:
                return {}
            return {
                "arousal": sum(s["state_after"].get("arousal", 0.5) for s in states) / len(states),
                "valence": sum(s["state_after"].get("valence", 0.0) for s in states) / len(states),
                "focus": sum(s["state_after"].get("focus", 0.5) for s in states) / len(states),
                "confidence": sum(s["state_after"].get("confidence", 0.5) for s in states) / len(states),
            }

        return {
            "total_patterns": len(patterns),
            "success_avg_state": avg_state(success_states),
            "failure_avg_state": avg_state(failure_states),
            "aborted_avg_state": avg_state(aborted_states),
            "insights": self._generate_insights(success_states, failure_states, aborted_states),
        }

    def _generate_insights(
        self,
        success_states: List,
        failure_states: List,
        aborted_states: List
    ) -> List[str]:
        """Generate human-readable insights from emotional patterns"""

        insights = []

        if not success_states or not failure_states:
            return insights

        # Calculate averages
        success_focus = sum(s["state_after"].get("focus", 0.5) for s in success_states) / len(success_states)
        failure_focus = sum(s["state_after"].get("focus", 0.5) for s in failure_states) / len(failure_states)

        success_valence = sum(s["state_after"].get("valence", 0.0) for s in success_states) / len(success_states)
        failure_valence = sum(s["state_after"].get("valence", 0.0) for s in failure_states) / len(failure_states)

        # Generate insights
        if success_focus > failure_focus + 0.1:
            insights.append("✅ Higher focus (>0.6) correlates with success")

        if success_valence > failure_valence + 0.15:
            insights.append("😊 Positive mood (valence > 0) improves outcomes")

        if aborted_states:
            aborted_arousal = sum(s["state_after"].get("arousal", 0.5) for s in aborted_states) / len(aborted_states)
            if aborted_arousal > 0.7:
                insights.append("⚠️ High arousal (>0.7) leads to task abandonment")

        return insights


# Singleton instance
emotional_feedback_loop = EmotionalFeedbackLoop()
