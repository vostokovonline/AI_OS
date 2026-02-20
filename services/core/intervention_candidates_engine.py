"""
INTERVENTION CANDIDATES ENGINE
====================================

STEP 2.7 — Intervention Readiness Layer

Генерирует гипотезы вмешательств на основе system alerts.
НЕ применяет вмешательства — ТОЛЬКО предлагает.

Architectural invariants:
- IRL has NO write access to models/thresholds/weights/configs
- Candidates are hypotheses, NOT actions
- hypothesis field is REQUIRED
"""

from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, and_
from database import get_db
from models import (
    SystemAlert,
    InterventionCandidate,
    InterventionRiskScore,
    InterventionSimulation
)
import uuid
import json


# =============================================================================
# ALERT → CANDIDATE MAPPING
# =============================================================================

ALERT_TO_CANDIDATE_MAPPING = {
    "confidence_miscalibration": "adjust_confidence_scaling",
    "high_arousal_blindness": "raise_arousal_guardrail",
    "tier_reliability_drift": "lower_tier_weight",
    "ml_underperforming": "disable_ml_for_context",
}


# =============================================================================
# CANDIDATE GENERATOR
# =============================================================================

class InterventionCandidatesEngine:
    """
    Генерирует intervention candidates на основе активных alerts.

    НЕ применяет вмешательства — ТОЛЬКО создаёт записи в БД.
    """

    def generate_from_active_alerts(self) -> List[InterventionCandidate]:
        """
        Сканирует активные alerts и генерирует candidates.

        Returns:
            List[InterventionCandidate] — созданные кандидаты
        """
        db = next(get_db())

        try:
            # 1. Get active (unresolved) alerts
            stmt = select(SystemAlert).where(SystemAlert.resolved == False)
            result = db.execute(stmt)
            active_alerts = result.scalars().all()

            if not active_alerts:
                logger.info("ℹ️  [IRL] No active alerts — no candidates generated")
                return []

            # 2. Group alerts by type
            alerts_by_type = {}
            for alert in active_alerts:
                if alert.alert_type not in alerts_by_type:
                    alerts_by_type[alert.alert_type] = []
                alerts_by_type[alert.alert_type].append(alert)

            # 3. Generate candidates for each alert type
            candidates = []

            for alert_type, alerts in alerts_by_type.items():
                if alert_type in ALERT_TO_CANDIDATE_MAPPING:
                    candidate = self._generate_candidate_for_alert_type(
                        alert_type=alert_type,
                        alerts=alerts
                    )

                    if candidate:
                        db.add(candidate)
                        db.flush()  # Get ID before commit

                        logger.info(f"💡 [IRL] Generated candidate: {candidate.intervention_type}")
                        logger.info(f"   Hypothesis: {candidate.hypothesis[:80]}...")
                        logger.info(f"   Expected gain: {candidate.expected_gain:.3f}")
                        logger.info(f"   Estimated risk: {candidate.estimated_risk:.3f}")

                        candidates.append(candidate)

            db.commit()

            return candidates

        except Exception as e:
            logger.info(f"⚠️  [IRL] Failed to generate candidates: {e}")
            db.rollback()
            return []

        finally:
            db.close()

    def _generate_candidate_for_alert_type(
        self,
        alert_type: str,
        alerts: List[SystemAlert]
    ) -> Optional[InterventionCandidate]:
        """
        Генерирует candidate для конкретного типа alert.

        Mapping:
        - confidence_miscalibration → adjust_confidence_scaling
        - high_arousal_blindness → raise_arousal_guardrail
        - tier_reliability_drift → lower_tier_weight
        - ml_underperforming → disable_ml_for_context
        """

        intervention_type = ALERT_TO_CANDIDATE_MAPPING[alert_type]

        # Extract data from alerts
        alert_ids = [str(alert.id) for alert in alerts]
        trigger_data = alerts[0].trigger_data  # Use first alert as reference

        # Generate hypothesis + scope based on intervention type
        if intervention_type == "adjust_confidence_scaling":
            return self._generate_confidence_scaling_candidate(alerts, trigger_data)
        elif intervention_type == "raise_arousal_guardrail":
            return self._generate_arousal_guardrail_candidate(alerts, trigger_data)
        elif intervention_type == "lower_tier_weight":
            return self._generate_lower_tier_candidate(alerts, trigger_data)
        elif intervention_type == "disable_ml_for_context":
            return self._generate_disable_ml_candidate(alerts, trigger_data)

        return None

    def _generate_confidence_scaling_candidate(
        self,
        alerts: List[SystemAlert],
        trigger_data: Dict
    ) -> InterventionCandidate:
        """
        Alert: confidence_miscalibration
        Candidate: adjust_confidence_scaling

        Hypothesis: Систематически завышает confidence → нужно понизить scaling.
        """

        stated_confidence = trigger_data.get("stated_confidence", 0.8)
        observed_accuracy = trigger_data.get("observed_accuracy", 0.6)
        gap = stated_confidence - observed_accuracy

        # Calculate expected gain and risk
        expected_gain = gap * 0.5  # Прогноз: закроем 50% разрыва
        estimated_risk = 0.15 + (gap * 0.2)  # Risk растет с gap
        confidence = 0.65

        target_scope = {
            "dimension": "confidence",
            "adjustment_type": "scale_down",
            "estimated_factor": 1.0 - gap,  # На сколько умножить
        }

        hypothesis = (
            f"System systematically overstates confidence by {gap:.3f} "
            f"(stated {stated_confidence:.3f} ≠ observed {observed_accuracy:.3f}). "
            f"Scaling down confidence estimates by {gap:.1%} should improve calibration "
            f"and reduce trust degradation. Expected gain: +{expected_gain:.3f} trust score."
        )

        return InterventionCandidate(
            id=uuid.uuid4(),
            intervention_type="adjust_confidence_scaling",
            target_scope=target_scope,
            triggered_by_alerts=alert_ids,
            hypothesis=hypothesis,
            expected_gain=expected_gain,
            estimated_risk=estimated_risk,
            confidence=confidence,
            status="proposed",
            created_at=datetime.now(timezone.utc)
        )

    def _generate_arousal_guardrail_candidate(
        self,
        alerts: List[SystemAlert],
        trigger_data: Dict
    ) -> InterventionCandidate:
        """
        Alert: high_arousal_blindness
        Candidate: raise_arousal_guardrail

        Hypothesis: При высоком arousal точность падает → нужен guardrail.
        """

        direction_accuracy = trigger_data.get("current_value", 0.5)
        baseline = trigger_data.get("baseline", 0.60)
        drop = baseline - direction_accuracy
        high_arousal_threshold = trigger_data.get("high_arousal_threshold", 0.75)

        # Calculate expected gain and risk
        expected_gain = drop * 0.7  # Прогноз: восстановим 70% потери
        estimated_risk = 0.20 + (drop * 0.3)
        confidence = 0.70

        target_scope = {
            "dimension": "arousal",
            "guardrail_type": "threshold",
            "current_threshold": high_arousal_threshold,
            "proposed_threshold": high_arousal_threshold - 0.10,  # Снизить порог
        }

        hypothesis = (
            f"Direction accuracy drops by {drop:.3f} at high arousal "
            f"(baseline {baseline:.2f} vs current {direction_accuracy:.2f}). "
            f"Lowering arousal threshold from {high_arousal_threshold:.2f} to {high_arousal_threshold - 0.10:.2f} "
            f"should prevent execution in high-arousal zones. Expected gain: +{expected_gain:.3f} accuracy."
        )

        return InterventionCandidate(
            id=uuid.uuid4(),
            intervention_type="raise_arousal_guardrail",
            target_scope=target_scope,
            triggered_by_alerts=alert_ids,
            hypothesis=hypothesis,
            expected_gain=expected_gain,
            estimated_risk=estimated_risk,
            confidence=confidence,
            status="proposed",
            created_at=datetime.now(timezone.utc)
        )

    def _generate_lower_tier_candidate(
        self,
        alerts: List[SystemAlert],
        trigger_data: Dict
    ) -> InterventionCandidate:
        """
        Alert: tier_reliability_drift
        Candidate: lower_tier_weight

        Hypothesis: Tier показывает аномально плохую точность → снизить вес.
        """

        tier = trigger_data.get("tier", "ML")
        current_accuracy = trigger_data.get("current_value", 0.5)
        baseline = trigger_data.get("baseline", 0.60)
        drift = baseline - current_accuracy

        # Calculate expected gain and risk
        expected_gain = drift * 0.6
        estimated_risk = 0.25 + (drift * 0.4)
        confidence = 0.60

        target_scope = {
            "tier": tier,
            "adjustment_type": "lower_weight",
            "current_weight": 1.0,
            "proposed_weight": 0.7,  # Снизить вес
        }

        hypothesis = (
            f"Tier {tier} shows reliability drift of {drift:.3f} "
            f"(current accuracy {current_accuracy:.3f} vs baseline {baseline:.2f}). "
            f"Lowering {tier} weight from 1.0 to 0.7 should reduce error propagation. "
            f"Expected gain: +{expected_gain:.3f} system accuracy."
        )

        return InterventionCandidate(
            id=uuid.uuid4(),
            intervention_type="lower_tier_weight",
            target_scope=target_scope,
            triggered_by_alerts=alert_ids,
            hypothesis=hypothesis,
            expected_gain=expected_gain,
            estimated_risk=estimated_risk,
            confidence=confidence,
            status="proposed",
            created_at=datetime.now(timezone.utc)
        )

    def _generate_disable_ml_candidate(
        self,
        alerts: List[SystemAlert],
        trigger_data: Dict
    ) -> InterventionCandidate:
        """
        Alert: ml_underperforming
        Candidate: disable_ml_for_context

        Hypothesis: ML системно хуже Rules → отключить в определённом контексте.
        """

        ml_accuracy = trigger_data.get("ml_value", 0.55)
        rules_accuracy = trigger_data.get("rules_value", 0.72)
        margin = rules_accuracy - ml_accuracy

        # Calculate expected gain and risk
        expected_gain = margin * 0.5
        estimated_risk = 0.30 + (margin * 0.5)
        confidence = 0.55

        # Extract context from trigger_data
        affected_actions = trigger_data.get("affected_actions", ["*"])

        target_scope = {
            "tier": "ML",
            "disable_context": affected_actions,
            "fallback_tier": "Rules",
        }

        hypothesis = (
            f"ML underperforms Rules by {margin:.3f} in direction accuracy "
            f"(ML {ml_accuracy:.3f} vs Rules {rules_accuracy:.3f}). "
            f"Disabling ML for {affected_actions} and falling back to Rules "
            f"should improve prediction quality. Expected gain: +{expected_gain:.3f} accuracy."
        )

        return InterventionCandidate(
            id=uuid.uuid4(),
            intervention_type="disable_ml_for_context",
            target_scope=target_scope,
            triggered_by_alerts=alert_ids,
            hypothesis=hypothesis,
            expected_gain=expected_gain,
            estimated_risk=estimated_risk,
            confidence=confidence,
            status="proposed",
            created_at=datetime.now(timezone.utc)
        )

    def get_candidates_by_status(
        self,
        status: str = "proposed",
        limit: int = 20
    ) -> List[Dict]:
        """
        Get candidates by status.

        Returns:
            List of dicts with candidate details
        """
        db = next(get_db())

        try:
            stmt = select(InterventionCandidate).where(
                InterventionCandidate.status == status
            ).order_by(
                InterventionCandidate.created_at.desc()
            ).limit(limit)

            result = db.execute(stmt)
            candidates = result.scalars().all()

            return [
                {
                    "id": str(c.id),
                    "intervention_type": c.intervention_type,
                    "hypothesis": c.hypothesis,
                    "expected_gain": c.expected_gain,
                    "estimated_risk": c.estimated_risk,
                    "confidence": c.confidence,
                    "status": c.status,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                    "triggered_by_alerts": c.triggered_by_alerts,
                }
                for c in candidates
            ]

        finally:
            db.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

intervention_candidates_engine = InterventionCandidatesEngine()
