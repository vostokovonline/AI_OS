from logging_config import get_logger
logger = get_logger(__name__)

"""
INTERVENTION RISK SCORING SYSTEM
====================================

STEP 2.7 — Intervention Readiness Layer

Вычисляет risk score для intervention candidates.

Risk tiers:
- LOW: Approvable
- MEDIUM: Approvable with rationale
- HIGH: Review required
- CRITICAL: Forbidden (even human cannot approve)

Principle: Better to NOT fix, than to fix incorrectly.
"""

from typing import Dict, Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, and_, func, text
from database import get_db
from models import (
    InterventionCandidate,
    InterventionRiskScore,
    SystemAlert
)
import uuid


# =============================================================================
# RISK THRESHOLDS
# =============================================================================

class RiskThresholds:
    """Пороги для вычисления risk score."""

    # Risk tier boundaries
    CRITICAL_THRESHOLD = 0.75
    HIGH_THRESHOLD = 0.50
    MEDIUM_THRESHOLD = 0.25

    # Component weights
    W_INSTABILITY = 0.25
    W_DATA_SUFFICIENCY = 0.30
    W_ALERT_DENSITY = 0.20
    W_AROUSAL_EXPOSURE = 0.15
    W_SCOPE_BLAST_RADIUS = 0.10


# =============================================================================
# RISK SCORER
# =============================================================================

class InterventionRiskScorer:
    """
    Вычисляет risk score для intervention candidates.

    Формула:
    risk = w1 * instability_score +
           w2 * (1 - data_sufficiency) +
           w3 * alert_density +
           w4 * arousal_exposure +
           w5 * scope_blast_radius
    """

    def compute_risk_score(
        self,
        intervention_id: str
    ) -> Optional[InterventionRiskScore]:
        """
        Вычисляет risk score для intervention.

        Returns:
            InterventionRiskScore или None
        """
        db = next(get_db())

        try:
            # 1. Load intervention
            stmt = select(InterventionCandidate).where(
                InterventionCandidate.id == intervention_id
            )
            result = db.execute(stmt)
            intervention = result.scalar_one_or_none()

            if not intervention:
                logger.info(f"⚠️  [RiskScorer] Intervention {intervention_id} not found")
                return None

            # 2. Compute component scores
            instability_score = self._compute_instability_score(db)
            data_sufficiency = self._compute_data_sufficiency(db)
            alert_density = self._compute_alert_density(db, intervention.triggered_by_alerts)
            arousal_exposure = self._compute_arousal_exposure(db, intervention)
            scope_blast_radius = self._compute_scope_blast_radius(intervention)

            # 3. Compute total risk
            total_risk = self._compute_total_risk(
                instability_score=instability_score,
                data_sufficiency=data_sufficiency,
                alert_density=alert_density,
                arousal_exposure=arousal_exposure,
                scope_blast_radius=scope_blast_radius
            )

            # 4. Determine risk tier
            risk_tier = self._determine_risk_tier(total_risk)

            # 5. Create risk score record
            risk_score = InterventionRiskScore(
                intervention_id=intervention_id,
                instability_score=round(instability_score, 4),
                data_sufficiency=round(data_sufficiency, 4),
                alert_density=round(alert_density, 4),
                arousal_exposure=round(arousal_exposure, 4),
                scope_blast_radius=round(scope_blast_radius, 4),
                total_risk=round(total_risk, 4),
                risk_tier=risk_tier,
                created_at=datetime.now(timezone.utc)
            )

            db.add(risk_score)
            db.commit()

            logger.info(f"🎯 [RiskScorer] Computed risk score for {intervention.intervention_type}")
            logger.info(f"   Total risk: {total_risk:.4f} → {risk_tier}")
            logger.info(f"   Instability: {instability_score:.4f}")
            logger.info(f"   Data sufficiency: {data_sufficiency:.4f}")
            logger.info(f"   Alert density: {alert_density:.4f}")
            logger.info(f"   Arousal exposure: {arousal_exposure:.4f}")
            logger.info(f"   Scope blast radius: {scope_blast_radius:.4f}")

            return risk_score

        except Exception as e:
            logger.info(f"⚠️  [RiskScorer] Failed to compute risk score: {e}")
            import traceback
            traceback.print_exc()
            db.rollback()
            return None

        finally:
            db.close()

    def _compute_instability_score(self, db) -> float:
        """
        Нестабильность метрик (0..1, higher = more unstable).

        Вычисляет дисперсию accuracy последних N forecast-outcome pairs.
        """

        try:
            # Get variance of direction accuracy over time
            query = text("""
                WITH accuracy_by_day AS (
                    SELECT
                        DATE_TRUNC('day', ef.created_at) as day,
                        AVG(CASE
                            WHEN SIGN((ef.predicted_deltas->>'valence')::float) =
                                 SIGN((eo.actual_deltas->>'valence')::float)
                            THEN 1.0
                            ELSE 0.0
                        END) as daily_accuracy
                    FROM emotional_forecasts ef
                    JOIN emotional_outcomes eo ON eo.forecast_id = ef.id
                    WHERE ef.created_at >= NOW() - INTERVAL '30 days'
                    GROUP BY day
                    HAVING COUNT(*) >= 5
                )
                SELECT
                    STDDEV(daily_accuracy) as accuracy_stddev,
                    AVG(daily_accuracy) as accuracy_mean
                FROM accuracy_by_day
            """)

            result = db.execute(query)
            row = result.fetchone()

            if row and row[0] is not None and row[1] is not None:
                stddev = float(row[0])
                mean = float(row[1])

                # Coefficient of variation = stddev / mean
                if mean > 0:
                    cv = stddev / mean
                    return min(1.0, cv)  # Cap at 1.0

            return 0.3  # Default: moderate instability

        except Exception as e:
            logger.info(f"⚠️  [RiskScorer] Failed to compute instability: {e}")
            return 0.5  # Conservative: assume moderate instability

    def _compute_data_sufficiency(self, db) -> float:
        """
        Достаточность данных (0..1, higher = more data).

        Based on:
        - Total forecast-outcome pairs
        - Recency of data
        - Coverage of action types
        """

        try:
            query = text("""
                SELECT
                    COUNT(*) as total_pairs,
                    COUNT(DISTINCT action_type) as action_type_count,
                    MAX(ef.created_at) as latest_forecast
                FROM emotional_forecasts ef
                JOIN emotional_outcomes eo ON eo.forecast_id = ef.id
                WHERE ef.created_at >= NOW() - INTERVAL '60 days'
            """)

            result = db.execute(query)
            row = result.fetchone()

            if row:
                total_pairs = row[0] or 0
                action_type_count = row[1] or 0
                latest = row[2]

                # Sufficient if: 50+ pairs, 3+ action types, recent data
                sufficiency = 0.0

                if total_pairs >= 100:
                    sufficiency += 0.4
                elif total_pairs >= 50:
                    sufficiency += 0.2

                if action_type_count >= 5:
                    sufficiency += 0.3
                elif action_type_count >= 3:
                    sufficiency += 0.15

                if latest:
                    days_since_latest = (datetime.now(timezone.utc) - latest).days
                    if days_since_latest <= 1:
                        sufficiency += 0.3
                    elif days_since_latest <= 7:
                        sufficiency += 0.15

                return min(1.0, sufficiency)

            return 0.2  # Default: low data sufficiency

        except Exception as e:
            logger.info(f"⚠️  [RiskScorer] Failed to compute data sufficiency: {e}")
            return 0.3  # Conservative: assume low sufficiency

    def _compute_alert_density(self, db, triggered_alerts: list) -> float:
        """
        Плотность alerts (0..1, higher = more alerts).

        Based on:
        - Number of active alerts
        - Severity of alerts
        - Recency of alerts
        """

        try:
            query = text("""
                SELECT
                    COUNT(*) FILTER (WHERE severity = 'CRITICAL') as critical_count,
                    COUNT(*) FILTER (WHERE severity = 'WARNING') as warning_count,
                    COUNT(*) FILTER (WHERE severity = 'INFO') as info_count,
                    COUNT(*) as total_count
                FROM system_alerts
                WHERE resolved = false
                  AND created_at >= NOW() - INTERVAL '7 days'
            """)

            result = db.execute(query)
            row = result.fetchone()

            if row:
                critical = row[0] or 0
                warning = row[1] or 0
                info = row[2] or 0
                total = row[3] or 0

                # Weighted alert count
                weighted_count = (critical * 3.0) + (warning * 1.5) + (info * 0.5)

                # Normalize to 0..1
                density = min(1.0, weighted_count / 10.0)
                return density

            return 0.0  # No alerts

        except Exception as e:
            logger.info(f"⚠️  [RiskScorer] Failed to compute alert density: {e}")
            return 0.3  # Conservative: assume some alerts

    def _compute_arousal_exposure(self, db, intervention: InterventionCandidate) -> float:
        """
        Экспозиция к high-arousal зонам (0..1).

        Based on:
        - Intervention type (arousal-related = higher exposure)
        - Historical performance in high-arousal zones
        """

        try:
            # Base exposure by intervention type
            if intervention.intervention_type == "raise_arousal_guardrail":
                base_exposure = 0.8  # Directly addresses arousal
            elif intervention.intervention_type == "disable_ml_for_context":
                base_exposure = 0.4  # Indirect
            elif intervention.intervention_type == "adjust_confidence_scaling":
                base_exposure = 0.3  # Low
            else:
                base_exposure = 0.2  # Very low

            # Check if there are high-arousal alerts
            if intervention.triggered_by_alerts:
                stmt = select(SystemAlert).where(
                    and_(
                        SystemAlert.id.in_(intervention.triggered_by_alerts),
                        SystemAlert.alert_type == "high_arousal_blindness"
                ))
                result = db.execute(stmt)
                has_arousal_alert = result.first() is not None

                if has_arousal_alert:
                    base_exposure = min(1.0, base_exposure + 0.2)

            return base_exposure

        except Exception as e:
            logger.info(f"⚠️  [RiskScorer] Failed to compute arousal exposure: {e}")
            return 0.3  # Conservative

    def _compute_scope_blast_radius(self, intervention: InterventionCandidate) -> float:
        """
        "Радиус поражения" вмешательства (0..1).

        Based on:
        - Target scope (tier vs dimension vs action)
        - Scope specificity (specific = lower risk, wildcard = higher risk)
        """

        scope = intervention.target_scope

        # Base risk by intervention type
        if intervention.intervention_type == "disable_ml_for_context":
            # Disabling entire tier = high blast radius
            base_risk = 0.7
        elif intervention.intervention_type == "lower_tier_weight":
            # Affects tier selection = medium-high
            base_risk = 0.6
        elif intervention.intervention_type == "raise_arousal_guardrail":
            # Affects specific dimension = medium
            base_risk = 0.4
        else:  # adjust_confidence_scaling
            # Affects confidence = low-medium
            base_risk = 0.3

        # Check for wildcard scopes
        if scope:
            # If scope has wildcards (*), increase risk
            scope_str = str(scope)
            if "*" in scope_str or "all" in scope_str.lower():
                base_risk = min(1.0, base_risk + 0.2)

        return base_risk

    def _compute_total_risk(
        self,
        instability_score: float,
        data_sufficiency: float,
        alert_density: float,
        arousal_exposure: float,
        scope_blast_radius: float
    ) -> float:
        """
        Вычисляет aggregate risk score.

        Formula:
        risk = w1 * instability_score +
               w2 * (1 - data_sufficiency) +
               w3 * alert_density +
               w4 * arousal_exposure +
               w5 * scope_blast_radius
        """

        w = RiskThresholds

        total_risk = (
            w.W_INSTABILITY * instability_score +
            w.W_DATA_SUFFICIENCY * (1.0 - data_sufficiency) +
            w.W_ALERT_DENSITY * alert_density +
            w.W_AROUSAL_EXPOSURE * arousal_exposure +
            w.W_SCOPE_BLAST_RADIUS * scope_blast_radius
        )

        return round(min(1.0, total_risk), 4)

    def _determine_risk_tier(self, total_risk: float) -> str:
        """
        Определяет risk tier на основе total_risk.

        Tiers:
        - LOW (0.00 - 0.25): Approvable
        - MEDIUM (0.25 - 0.50): Approvable with rationale
        - HIGH (0.50 - 0.75): Review required
        - CRITICAL (0.75 - 1.00): Forbidden
        """

        if total_risk >= RiskThresholds.CRITICAL_THRESHOLD:
            return "CRITICAL"
        elif total_risk >= RiskThresholds.HIGH_THRESHOLD:
            return "HIGH"
        elif total_risk >= RiskThresholds.MEDIUM_THRESHOLD:
            return "MEDIUM"
        else:
            return "LOW"

    def get_risk_score(self, intervention_id: str) -> Optional[Dict]:
        """
        Get risk score for intervention.

        Returns:
            Dict with risk details or None
        """
        db = next(get_db())

        try:
            stmt = select(InterventionRiskScore).where(
                InterventionRiskScore.intervention_id == intervention_id
            )
            result = db.execute(stmt)
            risk_score = result.scalar_one_or_none()

            if not risk_score:
                return None

            return {
                "intervention_id": str(risk_score.intervention_id),
                "instability_score": risk_score.instability_score,
                "data_sufficiency": risk_score.data_sufficiency,
                "alert_density": risk_score.alert_density,
                "arousal_exposure": risk_score.arousal_exposure,
                "scope_blast_radius": risk_score.scope_blast_radius,
                "total_risk": risk_score.total_risk,
                "risk_tier": risk_score.risk_tier,
                "created_at": risk_score.created_at.isoformat() if risk_score.created_at else None
            }

        finally:
            db.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

intervention_risk_scorer = InterventionRiskScorer()
