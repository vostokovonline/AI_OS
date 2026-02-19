"""
Emotional Inference Engine v2 (EIE v2)
=====================================

Полноценная подсистема для предсказания, управления и обучения
эмоциональных состояний в AI-OS.

Архитектура (5 слоёв):
1. State Reconstruction Engine — восстановление реального состояния
2. Pattern Context Builder — анализ эмоциональных паттернов
3. Emotional Forecasting Engine — предсказание последствий
4. Intent Alignment Layer — согласование с намерениями
5. Decision Modifiers + Safeguards — выходные модификаторы

Ключевые отличия от v1:
- Временная динамика (time-decay, recovery)
- Meta-outcomes (learning gain, unexpected)
- Emotional forecasting (simulation)
- Intent alignment (restore/maintain/progress)
- Safeguards (collapse protection)
"""

from typing import Dict, List, Optional, Tuple, Literal
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from sqlalchemy import select, and_, func, case
from database import AsyncSessionLocal
from models import (
    EmotionalLayerState,
    AffectiveMemoryEntry,
    Goal,
    EmotionalForecast  # 🆕 STEP 2.4
)
from emotional_config import EMOTIONAL_BASELINE
import math


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EmotionalState:
    """Расширенная модель эмоционального состояния"""
    arousal: float       # 0..1, baseline 0.5
    valence: float       # -1..1, baseline 0.0
    focus: float         # 0..1, baseline 0.5
    confidence: float    # 0..1, baseline 0.5
    timestamp: datetime = None

    def to_dict(self) -> Dict:
        return {
            "arousal": self.arousal,
            "valence": self.valence,
            "focus": self.focus,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "EmotionalState":
        return cls(
            arousal=data.get("arousal", 0.5),
            valence=data.get("valence", 0.0),
            focus=data.get("focus", 0.5),
            confidence=data.get("confidence", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(timezone.utc),
        )


@dataclass
class MetaOutcome:
    """Мета-результат выполнения (learning-aware)"""
    outcome: Literal["success", "failure", "aborted"]
    learning_gain: float = 0.0      # 0..1, насколько мы научились
    unexpected: bool = False        # был ли результат неожиданным
    effort: float = 0.5             # 0..1, сколько усилий потрачено
    user_reflection: str = ""       # рефлексия пользователя


@dataclass
class EmotionalTransition:
    """Эмоциональный переход (единица обучения)"""
    before: EmotionalState
    after: EmotionalState
    meta_outcome: MetaOutcome
    context: Dict = field(default_factory=dict)  # goal_type, complexity, etc.

    def delta(self) -> Dict[str, float]:
        """Вычислить изменение эмоций"""
        return {
            "arousal": self.after.arousal - self.before.arousal,
            "valence": self.after.valence - self.before.valence,
            "focus": self.after.focus - self.before.focus,
            "confidence": self.after.confidence - self.before.confidence,
        }


@dataclass
class EmotionalIntent:
    """Эмоциональное намерение (чего пользователь хочет чувствовать)"""
    primary: Literal[
        "restore_confidence",
        "reduce_arousal",
        "maintain_focus",
        "increase_engagement",
        "neutral"
    ]
    priority: float = 0.5  # 0..1


@dataclass
class EmotionalForecast:
    """Прогноз эмоционального состояния"""
    predicted_state: EmotionalState
    risk_flags: List[str] = field(default_factory=list)
    expected_delta: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5  # 0..1, уверенность в прогнозе
    # 🆕 STEP 2.4: Forecast persistence
    forecast_id: Optional[str] = None  # UUID из EmotionalForecast (DB model)
    used_tier: Optional[str] = None  # "ML" | "Clusters" | "Rules"


@dataclass
class PatternContext:
    """Контекст эмоциональных паттернов пользователя"""
    risk_profile: Dict[str, float] = field(default_factory=dict)
    dominant_patterns: List[str] = field(default_factory=list)
    success_correlations: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# LAYER 1: State Reconstruction Engine
# =============================================================================

class StateReconstructionEngine:
    """
    Восстанавливает реальное текущее состояние с учётом времени и переходов.
    """

    # Time-decay constants (hours to decay to 37%)
    DECAY_RATES = {
        "arousal": 2.0,      # быстро падает
        "valence": 12.0,     # медленно меняется
        "focus": 6.0,        # средняя скорость
        "confidence": 24.0,  # очень медленно растёт/падает
    }

    def decay_state(self, state: EmotionalState, dt_hours: float) -> EmotionalState:
        """
        Применить экспоненциальное затухание к состоянию.

        Formula: value = baseline + (current - baseline) * exp(-dt / half_life)
        """
        decayed = EmotionalState(
            arousal=self._decay_dimension(state.arousal, 0.5, dt_hours, self.DECAY_RATES["arousal"]),
            valence=self._decay_dimension(state.valence, 0.0, dt_hours, self.DECAY_RATES["valence"]),
            focus=self._decay_dimension(state.focus, 0.5, dt_hours, self.DECAY_RATES["focus"]),
            confidence=self._decay_dimension(state.confidence, 0.5, dt_hours, self.DECAY_RATES["confidence"]),
            timestamp=datetime.now(timezone.utc),
        )
        return decayed

    def _decay_dimension(self, value: float, baseline: float, dt_hours: float, half_life: float) -> float:
        """Затухание одной размерности к baseline"""
        delta = value - baseline
        decay_factor = math.exp(-dt_hours / half_life)
        return baseline + delta * decay_factor

    async def reconstruct_state(self, user_id: str) -> EmotionalState:
        """
        Восстановить реальное текущее состояние пользователя.

        Алгоритм:
        1. Взять последнее записанное состояние
        2. Применить time-decay
        3. Учесть недавние переходы
        """
        async with AsyncSessionLocal() as db:
            # 1. Get last recorded state
            stmt = select(EmotionalLayerState).where(
                EmotionalLayerState.user_id == user_id
            ).order_by(
                EmotionalLayerState.created_at.desc()
            ).limit(1)

            result = await db.execute(stmt)
            last_db_state = result.scalar_one_or_none()

            if not last_db_state:
                # No history - return baseline
                return EmotionalState(
                    arousal=0.5, valence=0.0, focus=0.5, confidence=0.5,
                    timestamp=datetime.now(timezone.utc)
                )

            # Convert to EmotionalState
            last_state = EmotionalState(
                arousal=last_db_state.arousal,
                valence=last_db_state.valence,
                focus=last_db_state.focus,
                confidence=last_db_state.confidence,
                timestamp=last_db_state.created_at,
            )

            # 2. Apply time-decay
            dt_hours = (datetime.now(timezone.utc) - last_state.timestamp).total_seconds() / 3600
            decayed_state = self.decay_state(last_state, dt_hours)

            # 3. Apply recent transitions (if any)
            # Get last 5 transitions from affective memory
            stmt_trans = select(AffectiveMemoryEntry).where(
                AffectiveMemoryEntry.user_id == user_id
            ).order_by(
                AffectiveMemoryEntry.created_at.desc()
            ).limit(5)

            result_trans = await db.execute(stmt_trans)
            transitions = result_trans.scalars().all()

            # Apply recent transition effects (with decay based on age)
            for trans in transitions:
                trans_dt = (datetime.now(timezone.utc) - trans.created_at).total_seconds() / 3600
                if trans_dt < 1.0:  # Only apply very recent transitions (< 1 hour)
                    trans_weight = math.exp(-trans_dt)  # Recent = more weight
                    after_state = trans.emotional_state_after or {}

                    # Blend current state with transition state
                    decayed_state.arousal = (
                        decayed_state.arousal * (1 - trans_weight * 0.1) +
                        after_state.get("arousal", 0.5) * trans_weight * 0.1
                    )
                    decayed_state.valence = (
                        decayed_state.valence * (1 - trans_weight * 0.1) +
                        after_state.get("valence", 0.0) * trans_weight * 0.1
                    )

            return decayed_state


# =============================================================================
# LAYER 2: Pattern Context Builder
# =============================================================================

class PatternContextBuilder:
    """
    Анализирует эмоциональные паттерны пользователя.
    """

    async def build_context(self, user_id: str, limit: int = 100) -> PatternContext:
        """
        Построить контекст паттернов из Affective Memory.

        Возвращает:
        - risk_profile: риски на основе истории
        - dominant_patterns: основные паттерны
        - success_correlations: корреляции с успехом
        """
        async with AsyncSessionLocal() as db:
            # Get transitions from affective memory
            stmt = select(AffectiveMemoryEntry).where(
                AffectiveMemoryEntry.user_id == user_id
            ).order_by(
                AffectiveMemoryEntry.created_at.desc()
            ).limit(limit)

            result = await db.execute(stmt)
            memories = result.scalars().all()

            if not memories:
                return PatternContext()

            # Analyze patterns
            transitions = self._build_transitions(memories)
            risk_profile = self._analyze_risks(transitions)
            dominant_patterns = self._extract_patterns(transitions)
            success_correlations = self._correlate_with_success(transitions)

            return PatternContext(
                risk_profile=risk_profile,
                dominant_patterns=dominant_patterns,
                success_correlations=success_correlations,
            )

    def _build_transitions(self, memories: List[AffectiveMemoryEntry]) -> List[EmotionalTransition]:
        """Построить список переходов из памяти"""
        transitions = []
        for mem in memories:
            before = mem.emotional_state_before or {}
            after = mem.emotional_state_after or {}

            trans = EmotionalTransition(
                before=EmotionalState(
                    arousal=before.get("arousal", 0.5),
                    valence=before.get("valence", 0.0),
                    focus=before.get("focus", 0.5),
                    confidence=before.get("confidence", 0.5),
                ),
                after=EmotionalState(
                    arousal=after.get("arousal", 0.5),
                    valence=after.get("valence", 0.0),
                    focus=after.get("focus", 0.5),
                    confidence=after.get("confidence", 0.5),
                ),
                meta_outcome=MetaOutcome(
                    outcome=mem.outcome,
                    learning_gain=mem.outcome_metrics.get("learning_gain", 0.0) if mem.outcome_metrics else 0.0,
                    unexpected=mem.outcome_metrics.get("unexpected", False) if mem.outcome_metrics else False,
                ),
            )
            transitions.append(trans)

        return transitions

    def _analyze_risks(self, transitions: List[EmotionalTransition]) -> Dict[str, float]:
        """Анализировать риски на основе истории"""
        risks = {}

        if not transitions:
            return risks

        # High arousal failure rate
        high_arousal_failures = [
            t for t in transitions
            if t.before.arousal > 0.7 and t.meta_outcome.outcome == "failure"
        ]
        high_arousal_total = [
            t for t in transitions
            if t.before.arousal > 0.7
        ]

        if high_arousal_total:
            risks["high_arousal_failure_rate"] = len(high_arousal_failures) / len(high_arousal_total)

        # Confidence collapse risk
        confidence_drops = [
            t for t in transitions
            if t.delta()["confidence"] < -0.2
        ]

        if transitions:
            risks["confidence_collapse_rate"] = len(confidence_drops) / len(transitions)

        # Low focus failure rate
        low_focus_failures = [
            t for t in transitions
            if t.before.focus < 0.4 and t.meta_outcome.outcome == "failure"
        ]
        low_focus_total = [
            t for t in transitions
            if t.before.focus < 0.4
        ]

        if low_focus_total:
            risks["low_focus_failure_rate"] = len(low_focus_failures) / len(low_focus_total)

        return risks

    def _extract_patterns(self, transitions: List[EmotionalTransition]) -> List[str]:
        """Извлечь основные паттерны"""
        patterns = []

        if not transitions:
            return patterns

        # Success after arousal drop
        arousal_drop_success = [
            t for t in transitions
            if t.delta()["arousal"] < -0.1 and t.meta_outcome.outcome == "success"
        ]
        if len(arousal_drop_success) > len(transitions) * 0.3:
            patterns.append("success_after_arousal_drop")

        # Failure when focus < 0.4
        focus_fail = [
            t for t in transitions
            if t.before.focus < 0.4 and t.meta_outcome.outcome == "failure"
        ]
        if len(focus_fail) > len(transitions) * 0.3:
            patterns.append("failure_when_focus_low")

        # Confidence recovery after success
        conf_recovery = [
            t for t in transitions
            if t.delta()["confidence"] > 0.1 and t.meta_outcome.outcome == "success"
        ]
        if len(conf_recovery) > len(transitions) * 0.3:
            patterns.append("confidence_builds_on_success")

        return patterns

    def _correlate_with_success(self, transitions: List[EmotionalTransition]) -> Dict[str, float]:
        """Корреляции эмоциональных размерностей с успехом"""
        if not transitions:
            return {}

        correlations = {}

        # Success rate by focus level
        high_focus_success = [
            t for t in transitions
            if t.before.focus > 0.6 and t.meta_outcome.outcome == "success"
        ]
        high_focus_total = [
            t for t in transitions
            if t.before.focus > 0.6
        ]

        if high_focus_total:
            correlations["high_focus_success_rate"] = len(high_focus_success) / len(high_focus_total)

        # Success rate by valence
        positive_valence_success = [
            t for t in transitions
            if t.before.valence > 0.2 and t.meta_outcome.outcome == "success"
        ]
        positive_valence_total = [
            t for t in transitions
            if t.before.valence > 0.2
        ]

        if positive_valence_total:
            correlations["positive_valence_success_rate"] = len(positive_valence_success) / len(positive_valence_total)

        return correlations


# =============================================================================
# LAYER 3: Emotional Forecasting Engine
# =============================================================================

class EmotionalForecastingEngine:
    """
    Предсказывает эмоциональные последствия решений.
    """

    # Action impact coefficients
    ACTION_IMPACTS = {
        "deep_goal_decomposition": {
            "arousal": 0.15,
            "valence": -0.05,
            "focus": -0.1,
            "confidence": -0.1,
        },
        "simple_task": {
            "arousal": -0.05,
            "valence": 0.05,
            "focus": 0.1,
            "confidence": 0.05,
        },
        "complex_execution": {
            "arousal": 0.2,
            "valence": -0.1,
            "focus": 0.05,
            "confidence": -0.15,
        },
        "learning_task": {
            "arousal": 0.05,
            "valence": 0.0,
            "focus": 0.1,
            "confidence": 0.02,
        },
    }

    def simulate(
        self,
        current_state: EmotionalState,
        action: str,
        pattern_context: PatternContext,
        meta_outcome: Optional[MetaOutcome] = None,
        user_id: Optional[str] = None,
        goal_id: Optional[str] = None  # 🆕 STEP 2.4: optional goal_id
    ) -> EmotionalForecast:
        """
        Симулировать эмоциональное состояние после выполнения действия.

        THREE-TIER FORECASTING (в порядке приоритета):
        1. 🤖 ML Model (если доступна и уверена)
        2. 📊 Trajectory Clustering (если есть кластеры)
        3. 📐 Rule-based (всегда как safety net)

        Возвращает прогноз с рисками.
        """
        # 🆕 TIER 1: ML-based forecasting
        ml_impact = {}
        ml_confidence = 0.0

        try:
            from emotional_forecasting_model import emotional_forecasting_model

            if emotional_forecasting_model.is_available():
                current_dict = {
                    "arousal": current_state.arousal,
                    "valence": current_state.valence,
                    "focus": current_state.focus,
                    "confidence": current_state.confidence
                }

                pattern_dict = {
                    "risk_profile": pattern_context.risk_profile,
                    "success_correlations": pattern_context.success_correlations,
                    "dominant_patterns": pattern_context.dominant_patterns
                }

                ml_deltas, ml_conf = emotional_forecasting_model.predict(
                    current_dict, action, pattern_dict
                )

                # 🆕 STEP 2.3: CONFIDENCE CALIBRATION
                # Калибруем confidence на основе исторической точности
                try:
                    from confidence_calibrator import confidence_calibrator
                    from tier_reliability import tier_reliability_tracker

                    # Получаем метрики надежности для этого action_type
                    metrics = tier_reliability_tracker.get_reliability(action, "ML")

                    # Калибруем confidence
                    calibrated_ml_conf = confidence_calibrator.adjust(
                        raw_confidence=ml_conf,
                        action_type=action,
                        tier="ML",
                        metrics=metrics
                    )

                    print(f"🎯 [ML Calibration] {ml_conf:.3f} → {calibrated_ml_conf:.3f}")

                    # Используем откалиброванный confidence
                    ml_conf = calibrated_ml_conf
                except Exception as calib_err:
                    print(f"⚠️  [ML Calibration] Failed: {calib_err}, using raw confidence")

                # 🆕 PER-ACTION CONFIDENCE: Check threshold for this action
                from ml_guardrails import per_action_confidence
                action_threshold = per_action_confidence.get_threshold(action)

                if ml_conf >= action_threshold:  # Используем ML только если уверена
                    ml_impact = ml_deltas
                    ml_confidence = ml_conf
                    print(f"🤖 [ML Model] Using ML forecast (confidence={ml_conf:.2f}, threshold={action_threshold:.2f})")
                else:
                    print(f"⚠️  [ML Model] Low confidence ({ml_conf:.2f} < {action_threshold:.2f}), trying next tier")
            else:
                print(f"ℹ️  [ML Model] Not available, trying next tier")

        except Exception as e:
            print(f"⚠️  [ML Model] Error: {e}, trying next tier")

        # 🆕 TIER 2: Trajectory-based forecasting (если ML не сработал)
        cluster_impact = {}
        cluster_confidence = 0.0
        cluster_outcome = "unknown"

        if not ml_impact:
            try:
                from emotional_trajectory_clustering import (
                    trajectory_clusterer,
                    EmotionalTrajectory,
                    TrajectoryPoint
                )

                # Создаем текущую траекторию (пока только start точка)
                current_trajectory = EmotionalTrajectory(
                    trajectory_id="temp",
                    user_id=user_id or "unknown",
                    goal_id=None,
                    action_type=action,
                    outcome="unknown",
                    points=[
                        TrajectoryPoint(
                            state={
                                "arousal": current_state.arousal,
                                "valence": current_state.valence,
                                "focus": current_state.focus,
                                "confidence": current_state.confidence
                            },
                            created_at=datetime.now(timezone.utc),
                            phase="start"
                        )
                    ]
                )

                # Предсказываем на основе кластеров
                cluster_outcome, cluster_confidence, cluster_deltas = (
                    trajectory_clusterer.predict_trajectory_outcome(current_trajectory)
                )

                if cluster_confidence > 0.3:  # Используем кластеры только если достаточно уверены
                    cluster_impact = cluster_deltas
                    print(f"📊 [Trajectory Clustering] Using cluster-based forecast (confidence={cluster_confidence:.2f})")
                else:
                    print(f"⚠️  [Trajectory Clustering] Low confidence, falling back to rules")

            except Exception as e:
                print(f"⚠️  [Trajectory Clustering] Error: {e}, using rule-based")

        # 🆕 TIER 3: Rule-based forecasting (SAFETY NET - всегда работает)
        base_impact = self.ACTION_IMPACTS.get(action, {})
        adjusted_impact = self._adjust_for_patterns(base_impact, pattern_context)

        # 🆕 Смешиваем все три подхода
        final_impact = {}
        used_tiers = []

        if ml_impact and ml_confidence > 0.0:  # Уже проверено в predict() с per-action threshold
            # ML + rules (ML primary, rules safety net)
            weight = ml_confidence  # 0.3-1.0 (varies by action)
            for dim in ["arousal", "valence", "focus", "confidence"]:
                ml_value = ml_impact.get(dim, 0)
                rule_value = adjusted_impact.get(dim, 0)
                final_impact[dim] = (1 - weight) * rule_value + weight * ml_value

            used_tiers.append("ML")
            used_tiers.append("Rules (safety net)")
            print(f"🔀 [Mixed Forecast] ML + Rules (weight={weight:.2f})")

        elif cluster_impact and cluster_confidence > 0.3:
            # Clusters + rules
            weight = cluster_confidence  # 0.3-1.0
            for dim in ["arousal", "valence", "focus", "confidence"]:
                cluster_value = cluster_impact.get(dim, 0)
                rule_value = adjusted_impact.get(dim, 0)
                final_impact[dim] = (1 - weight) * rule_value + weight * cluster_value

            used_tiers.append("Clusters")
            used_tiers.append("Rules (safety net)")
            print(f"🔀 [Mixed Forecast] Clusters + Rules (weight={weight:.2f})")

        else:
            # Rules only
            final_impact = adjusted_impact
            used_tiers.append("Rules only")

        print(f"📊 [Forecast Tiers] {' → '.join(used_tiers)}")

        # Predict new state
        predicted = EmotionalState(
            arousal=self._clamp(current_state.arousal + final_impact.get("arousal", 0), 0, 1),
            valence=self._clamp(current_state.valence + final_impact.get("valence", 0), -1, 1),
            focus=self._clamp(current_state.focus + final_impact.get("focus", 0), 0, 1),
            confidence=self._clamp(current_state.confidence + final_impact.get("confidence", 0), 0, 1),
            timestamp=datetime.now(timezone.utc),
        )

        # Calculate risk flags
        risk_flags = self._detect_risks(current_state, predicted, pattern_context)

        # 🆕 Если система предсказывает failure - добавляем риск
        if cluster_outcome == "failure" and cluster_confidence > 0.6:
            risk_flags.append(f"cluster_predicted_failure (conf={cluster_confidence:.2f})")

        # Calculate expected delta
        expected_delta = {
            "arousal": predicted.arousal - current_state.arousal,
            "valence": predicted.valence - current_state.valence,
            "focus": predicted.focus - current_state.focus,
            "confidence": predicted.confidence - current_state.confidence,
        }

        # 🆕 Confidence score (max из всех tiers)
        confidences = [ml_confidence, cluster_confidence, 0.5]
        final_confidence = max([c for c in confidences if c > 0])

        # 🆕 STEP 2.4: Сохраняем forecast в базу данных
        forecast_id, used_tier = self._save_forecast_to_db(
            user_id=user_id,
            action=action,
            expected_delta=expected_delta,
            final_confidence=final_confidence,
            used_tiers=used_tiers,
            risk_flags=risk_flags,
            goal_id=goal_id  # 🆕
        )

        return EmotionalForecast(
            predicted_state=predicted,
            risk_flags=risk_flags,
            expected_delta=expected_delta,
            confidence=final_confidence,
            forecast_id=forecast_id,  # 🆕
            used_tier=used_tier,       # 🆕
        )

    def _adjust_for_patterns(self, base_impact: Dict, context: PatternContext) -> Dict:
        """Скорректировать влияние на основе паттернов"""
        adjusted = base_impact.copy()

        # If user has pattern "success_after_arousal_drop", reduce arousal penalty
        if "success_after_arousal_drop" in context.dominant_patterns:
            adjusted["arousal"] = adjusted.get("arousal", 0) * 0.8

        # If user has pattern "failure_when_focus_low", amplify focus impact
        if "failure_when_focus_low" in context.dominant_patterns:
            adjusted["focus"] = adjusted.get("focus", 0) * 1.2

        # Adjust based on risk profile
        if context.risk_profile.get("high_arousal_failure_rate", 0) > 0.7:
            adjusted["arousal"] = adjusted.get("arousal", 0) * 1.3  # More conservative

        return adjusted

    def _detect_risks(
        self,
        before: EmotionalState,
        after: EmotionalState,
        context: PatternContext
    ) -> List[str]:
        """Обнаружить эмоциональные риски"""
        risks = []

        # Confidence collapse
        if after.confidence < 0.3:
            risks.append("confidence_collapse")

        # Task abandonment risk
        if after.arousal > 0.85:
            risks.append("task_abandonment")

        # Burnout risk
        if before.arousal > 0.7 and after.arousal > 0.75:
            risks.append("burnout_risk")

        # Focus fragmentation
        if after.focus < 0.3:
            risks.append("focus_fragmentation")

        # Learning block
        if before.valence < -0.4 and after.valence < -0.5:
            risks.append("learning_block")

        return risks

    def _clamp(self, value: float, min_val: float, max_val: float) -> float:
        """Ограничить значение диапазоном"""
        return max(min_val, min(max_val, value))

    def _save_forecast_to_db(
        self,
        user_id: Optional[str],
        action: str,
        expected_delta: Dict[str, float],
        final_confidence: float,
        used_tiers: List[str],
        risk_flags: List[str],
        goal_id: Optional[str] = None  # 🆕 STEP 2.4: optional goal_id
    ) -> Tuple[Optional[str], str]:
        """
        🆕 STEP 2.4: Сохраняет emotional forecast в базу данных.

        Возвращает:
            (forecast_id, used_tier)
        """
        if not user_id:
            # Без user_id не сохраняем (возвращаем None)
            return None, "Rules"

        try:
            # Определяем основной used_tier
            if "ML" in used_tiers:
                used_tier = "ML"
            elif "Clusters" in used_tiers:
                used_tier = "Clusters"
            else:
                used_tier = "Rules"

            # Создаём запись в DB
            import uuid
            forecast_record = EmotionalForecast(
                id=uuid.uuid4(),
                user_id=uuid.UUID(user_id) if isinstance(user_id, str) else user_id,
                goal_id=uuid.UUID(goal_id) if goal_id else None,  # 🆕 Сохраняем goal_id
                action_type=action,
                predicted_deltas=expected_delta,  # {arousal, valence, focus, confidence}
                forecast_confidence=final_confidence,
                used_tier=used_tier,
                risk_flags=risk_flags if risk_flags else None
            )

            # Сохраняем в DB (sync operation)
            from database import get_db
            db = next(get_db())

            try:
                db.add(forecast_record)
                db.commit()
                db.refresh(forecast_record)

                forecast_id = str(forecast_record.id)

                # 🆕 Если есть goal_id, обновляем goal.forecast_id
                if goal_id:
                    try:
                        from models import Goal
                        goal = db.query(Goal).filter(Goal.id == uuid.UUID(goal_id)).first()
                        if goal:
                            goal.forecast_id = forecast_record.id
                            db.commit()
                            print(f"🔗 [Forecast Persistence] Linked forecast {forecast_id} to goal {goal_id}")
                    except Exception as goal_err:
                        print(f"⚠️  [Forecast Persistence] Failed to link to goal: {goal_err}")

                print(f"💾 [Forecast Persistence] Saved forecast {forecast_id} (tier={used_tier}, conf={final_confidence:.2f})")

                return forecast_id, used_tier

            finally:
                db.close()

        except Exception as e:
            print(f"⚠️  [Forecast Persistence] Failed to save forecast: {e}")
            import traceback
            traceback.print_exc()

            # Возвращаем None, но продолжаем работу
            return None, "Rules"  # Fallback


# =============================================================================
# LAYER 4: Intent Alignment Layer
# =============================================================================

class IntentAlignmentLayer:
    """
    Согласует решения с эмоциональными намерениями.
    """

    def align(
        self,
        forecast: EmotionalForecast,
        intent: EmotionalIntent,
        current_state: EmotionalState
    ) -> Tuple[bool, str]:
        """
        Проверить, согласуется ли прогноз с намерением.

        Returns:
            (aligned, reason)
        """
        # If intent is neutral, always aligned
        if intent.primary == "neutral":
            return True, "neutral intent"

        # Restore confidence
        if intent.primary == "restore_confidence":
            if forecast.predicted_state.confidence < current_state.confidence:
                return False, "would reduce confidence further"
            if forecast.predicted_state.confidence < 0.4:
                return False, "confidence too low to build"
            return True, "supports confidence restoration"

        # Reduce arousal
        if intent.primary == "reduce_arousal":
            if forecast.predicted_state.arousal > current_state.arousal:
                return False, "would increase arousal"
            return True, "supports arousal reduction"

        # Maintain focus
        if intent.primary == "maintain_focus":
            if forecast.predicted_state.focus < current_state.focus - 0.1:
                return False, "would significantly reduce focus"
            return True, "maintains focus"

        # Increase engagement
        if intent.primary == "increase_engagement":
            if forecast.predicted_state.arousal < 0.3:
                return False, "too passive for engagement"
            if "task_abandonment" in forecast.risk_flags:
                return False, "high abandonment risk"
            return True, "supports engagement"

        return True, "aligned"


# =============================================================================
# LAYER 5: Decision Modifiers + Safeguards
# =============================================================================

@dataclass
class DecisionModifiers:
    """Выходные модификаторы решений"""
    max_depth: int = 3
    pace: Literal["slow", "normal", "fast"] = "normal"
    explanation_level: Literal["minimal", "normal", "detailed"] = "normal"
    style: Literal["direct", "supportive", "collaborative"] = "direct"
    safety_override: bool = False
    recovery_mode: bool = False


class DecisionModifiersEngine:
    """
    Генерирует модификаторы решений с защитами.
    """

    # Safeguard thresholds
    SAFEGUARDS = {
        "confidence_min": 0.2,      # Below this: no complex tasks
        "arousal_max": 0.85,        # Above this: no irreversible decisions
        "focus_min": 0.25,          # Below this: simplify
        "repeated_failure_threshold": 3,  # 3+ failures in a row: recovery mode
    }

    def generate(
        self,
        forecast: EmotionalForecast,
        intent: EmotionalIntent,
        aligned: bool
    ) -> DecisionModifiers:
        """
        Сгенерировать модификаторы на основе прогноза и намерения.
        """
        state = forecast.predicted_state

        # Default modifiers
        modifiers = DecisionModifiers()

        # Adjust based on forecast
        if state.confidence < 0.4:
            modifiers.max_depth = 2
            modifiers.explanation_level = "detailed"
            modifiers.style = "supportive"

        if state.confidence < 0.25:
            modifiers.max_depth = 1
            modifiers.pace = "slow"
            modifiers.style = "supportive"

        if state.arousal > 0.7:
            modifiers.max_depth = max(1, modifiers.max_depth - 1)
            modifiers.pace = "slow"
            modifiers.explanation_level = "detailed"

        if state.focus < 0.4:
            modifiers.max_depth = max(1, modifiers.max_depth - 1)
            modifiers.explanation_level = "detailed"

        if state.valence < -0.3:
            modifiers.style = "supportive"
            modifiers.pace = "slow"

        # Intent-based adjustments
        if intent.primary == "restore_confidence":
            modifiers.max_depth = min(modifiers.max_depth, 2)
            modifiers.style = "supportive"

        if intent.primary == "reduce_arousal":
            modifiers.max_depth = min(modifiers.max_depth, 2)
            modifiers.pace = "slow"

        # Apply safeguards
        modifiers = self._apply_safeguards(modifiers, forecast, aligned)

        return modifiers

    def _apply_safeguards(
        self,
        modifiers: DecisionModifiers,
        forecast: EmotionalForecast,
        aligned: bool
    ) -> DecisionModifiers:
        """Применить защитные ограничения"""
        state = forecast.predicted_state

        # Confidence safeguard
        if state.confidence < self.SAFEGUARDS["confidence_min"]:
            modifiers.max_depth = 1
            modifiers.safety_override = True
            modifiers.recovery_mode = True

        # Arousal safeguard
        if state.arousal > self.SAFEGUARDS["arousal_max"]:
            modifiers.max_depth = 1
            modifiers.safety_override = True

        # Focus safeguard
        if state.focus < self.SAFEGUARDS["focus_min"]:
            modifiers.max_depth = 1
            modifiers.explanation_level = "detailed"

        # Risk flag safeguards
        if "confidence_collapse" in forecast.risk_flags:
            modifiers.recovery_mode = True
            modifiers.safety_override = True

        if "task_abandonment" in forecast.risk_flags:
            modifiers.max_depth = 1
            modifiers.safety_override = True

        # Intent misalignment
        if not aligned:
            modifiers.safety_override = True
            modifiers.max_depth = min(modifiers.max_depth, 2)

        return modifiers


# =============================================================================
# MAIN ENGINE: Emotional Inference Engine v2
# =============================================================================

class EmotionalInferenceEngineV2:
    """
    Главная точка входа EIE v2.

    Использует все 5 слоёв для генерации эмоциональных модификаторов.
    """

    def __init__(self):
        self.state_reconstructor = StateReconstructionEngine()
        self.pattern_builder = PatternContextBuilder()
        self.forecaster = EmotionalForecastingEngine()
        self.intent_aligner = IntentAlignmentLayer()
        self.modifiers_engine = DecisionModifiersEngine()

    async def infer(
        self,
        user_id: str,
        proposed_action: str,
        intent: Optional[EmotionalIntent] = None,
        signals: Optional[Dict] = None,
        goal_id: Optional[str] = None  # 🆕 STEP 2.4: optional goal_id
    ) -> DecisionModifiers:
        """
        Главная точка входа.

        Выполняет полный pipeline EIE v2:
        1. Reconstruct state
        2. Build pattern context
        3. Forecast emotional outcome
        4. Check intent alignment
        5. Generate decision modifiers
        """
        # Default intent
        if intent is None:
            intent = EmotionalIntent(primary="neutral")

        # 1. Reconstruct current state
        current_state = await self.state_reconstructor.reconstruct_state(user_id)

        # 2. Build pattern context
        pattern_context = await self.pattern_builder.build_context(user_id)

        # 3. Forecast emotional outcome (с trajectory clustering)
        forecast = self.forecaster.simulate(
            current_state=current_state,
            action=proposed_action,
            pattern_context=pattern_context,
            user_id=user_id,
            goal_id=goal_id  # 🆕 STEP 2.4: Передаем goal_id
        )

        # 4. Check intent alignment
        aligned, reason = self.intent_aligner.align(
            forecast=forecast,
            intent=intent,
            current_state=current_state
        )

        if not aligned:
            print(f"⚠️  Intent misalignment: {reason}")

        # 5. Generate decision modifiers
        modifiers = self.modifiers_engine.generate(
            forecast=forecast,
            intent=intent,
            aligned=aligned
        )

        # Log for debugging
        print(f"🧠 [EIE v2] Inference for user {user_id}:")
        print(f"   Current state: arousal={current_state.arousal:.2f}, valence={current_state.valence:.2f}")
        print(f"   Forecast: {forecast.predicted_state.arousal:.2f}, {forecast.predicted_state.valence:.2f}")
        print(f"   Risks: {forecast.risk_flags}")
        print(f"   Modifiers: max_depth={modifiers.max_depth}, pace={modifiers.pace}")
        if modifiers.safety_override:
            print(f"   🔒 SAFETY OVERRIDE ACTIVE")

        return modifiers


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

emotional_inference_engine_v2 = EmotionalInferenceEngineV2()
