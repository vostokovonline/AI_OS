"""
IRL HEALTH METRICS
====================================

FMEA-based health monitoring for Intervention Readiness Layer.

Мониторит все 6 Failure Modes из FMEA:
1. FM-IRL-01: False Positive Intervention Candidate
2. FM-IRL-02: Counterfactual Illusion (HIGH RISK)
3. FM-IRL-03: Risk Score Gaming (Human Side)
4. FM-IRL-04: Intervention Drift
5. FM-IRL-05: Semantic Overconfidence
6. FM-IRL-06: Silent IRL
"""

from typing import Dict, List
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, and_, func, text
from database import get_db_sync
from models import (
    InterventionCandidate,
    InterventionSimulation,
    InterventionRiskScore,
    InterventionApproval,
    SystemAlert
)


class IRLHealthMetrics:
    """
    Health metrics для мониторинга Failure Modes IRL.

    НЕ меняет поведение — только observes.
    """

    def get_full_health_report(self) -> Dict:
        """
        Полный health report по всем Failure Modes.

        Returns:
            {
                "overall_health": "HEALTHY" | "DEGRADED" | "CRITICAL",
                "failure_modes": {
                    "FM_IRL_01": { ... },
                    "FM_IRL_02": { ... },
                    ...
                },
                "summary": { ... }
            }
        """

        fm_reports = {
            "FM_IRL_01": self._check_false_positive_candidates(),
            "FM_IRL_02": self._check_counterfactual_illusion(),
            "FM_IRL_03": self._check_risk_score_gaming(),
            "FM_IRL_04": self._check_intervention_drift(),
            "FM_IRL_05": self._check_semantic_overconfidence(),
            "FM_IRL_06": self._check_silent_irl()
        }

        # Determine overall health
        critical_count = sum(1 for fm in fm_reports.values() if fm["severity"] == "CRITICAL")
        degraded_count = sum(1 for fm in fm_reports.values() if fm["severity"] == "DEGRADED")

        if critical_count > 0:
            overall_health = "CRITICAL"
        elif degraded_count >= 2:
            overall_health = "CRITICAL"
        elif degraded_count >= 1:
            overall_health = "DEGRADED"
        else:
            overall_health = "HEALTHY"

        return {
            "overall_health": overall_health,
            "failure_modes": fm_reports,
            "summary": {
                "total_modes": 6,
                "critical_count": critical_count,
                "degraded_count": degraded_count,
                "healthy_count": 6 - critical_count - degraded_count
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # =============================================================================
    # FM-IRL-01: False Positive Intervention Candidate
    # =============================================================================

    def _check_false_positive_candidates(self) -> Dict:
        """
        Risk: IRL proposes intervention for statistical noise, not structural problem.

        Indicators:
        - High reject rate without comments
        - Repeated candidates with low expected_gain
        """

        db = get_db_sync()

        try:
            # Last 30 days stats
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)

            # Total candidates proposed
            stmt_total = select(func.count(InterventionCandidate.id)).where(
                InterventionCandidate.created_at >= cutoff
            )

            total_result = db.execute(stmt_total)
            total_candidates = total_result.scalar() or 0

            if total_candidates == 0:
                return {
                    "failure_mode": "FM_IRL_01: False Positive Candidates",
                    "severity": "HEALTHY",
                    "indicators": {
                        "total_candidates_30d": 0,
                        "reject_rate": 0.0,
                        "low_gain_rate": 0.0
                    },
                    "message": "No candidates generated yet"
                }

            # Rejected candidates
            stmt_rejected = select(func.count(InterventionCandidate.id)).where(
                and_(
                    InterventionCandidate.created_at >= cutoff,
                    InterventionCandidate.status == "rejected"
                )
            )

            rejected_result = db.execute(stmt_rejected)
            rejected_count = rejected_result.scalar() or 0

            # Low expected_gain candidates (< 0.05)
            stmt_low_gain = select(func.count(InterventionCandidate.id)).where(
                and_(
                    InterventionCandidate.created_at >= cutoff,
                    InterventionCandidate.expected_gain < 0.05
                )
            )

            low_gain_result = db.execute(stmt_low_gain)
            low_gain_count = low_gain_result.scalar() or 0

            reject_rate = rejected_count / total_candidates if total_candidates > 0 else 0.0
            low_gain_rate = low_gain_count / total_candidates if total_candidates > 0 else 0.0

            # Determine severity
            if reject_rate > 0.8 or low_gain_rate > 0.6:
                severity = "CRITICAL"
            elif reject_rate > 0.6 or low_gain_rate > 0.4:
                severity = "DEGRADED"
            else:
                severity = "HEALTHY"

            return {
                "failure_mode": "FM_IRL_01: False Positive Candidates",
                "severity": severity,
                "indicators": {
                    "total_candidates_30d": total_candidates,
                    "rejected_count_30d": rejected_count,
                    "reject_rate": round(reject_rate, 3),
                    "low_gain_count_30d": low_gain_count,
                    "low_gain_rate": round(low_gain_rate, 3)
                },
                "message": f"Reject rate: {reject_rate:.1%}, Low gain rate: {low_gain_rate:.1%}"
            }

        finally:
            db.close()

    # =============================================================================
    # FM-IRL-02: Counterfactual Illusion (HIGH RISK)
    # =============================================================================

    def _check_counterfactual_illusion(self) -> Dict:
        """
        Risk: Simulation shows improvement not reproducible in reality.

        Indicators:
        - High expected_gain with LOW instability
        - "Too clean" improvements across all metrics
        - Large gain without side_effects
        """

        db = get_db_sync()

        try:
            # Simulations with high gain
            stmt = text("""
                SELECT
                    COUNT(*) as total_sims,
                    COUNT(*) FILTER (WHERE (delta_metrics->>'direction_accuracy')::float > 0.10) as high_gain_sims,
                    AVG((delta_metrics->>'direction_accuracy')::float) as avg_gain
                FROM intervention_simulations
                WHERE created_at >= NOW() - INTERVAL '30 days'
            """)

            result = db.execute(stmt)
            row = result.fetchone()

            if not row or row[0] == 0:
                return {
                    "failure_mode": "FM_IRL-02: Counterfactual Illusion",
                    "severity": "HEALTHY",
                    "indicators": {},
                    "message": "No simulations yet"
                }

            total_sims = row[0] or 0
            high_gain_sims = row[1] or 0
            avg_gain = row[2] or 0.0

            # Check for suspicious patterns: high gain without side_effects
            stmt_clean = text("""
                SELECT COUNT(*) as clean_sims
                FROM intervention_simulations
                WHERE created_at >= NOW() - INTERVAL '30 days'
                  AND side_effects IS NULL
                  AND (delta_metrics->>'direction_accuracy')::float > 0.08
            """)

            result_clean = db.execute(stmt_clean)
            clean_sims = result_clean.scalar() or 0

            # Severity assessment
            # CRITICAL: Many high-gain simulations without side effects
            clean_ratio = clean_sims / total_sims if total_sims > 0 else 0.0

            if clean_ratio > 0.7 and avg_gain > 0.10:
                severity = "CRITICAL"
            elif clean_ratio > 0.5 and avg_gain > 0.08:
                severity = "DEGRADED"
            else:
                severity = "HEALTHY"

            return {
                "failure_mode": "FM_IRL-02: Counterfactual Illusion",
                "severity": severity,
                "indicators": {
                    "total_simulations_30d": total_sims,
                    "high_gain_count": high_gain_sims,
                    "avg_expected_gain": round(avg_gain, 4),
                    "clean_improvements": clean_sims,
                    "clean_ratio": round(clean_ratio, 3)
                },
                "message": f"High-gain sims: {high_gain_sims}, Clean ratio: {clean_ratio:.1%} (suspicious if >50%)",
                "risk_note": "⚠️ This is the MOST DANGEROUS failure mode"
            }

        finally:
            db.close()

    # =============================================================================
    # FM-IRL-03: Risk Score Gaming (Human Side)
    # =============================================================================

    def _check_risk_score_gaming(self) -> Dict:
        """
        Risk: Operator rubber-stamps LOW/MEDIUM approvals without thought.

        Indicators:
        - Approve rate ↑, comment rate ↓
        - Fast approve patterns
        """

        db = get_db_sync()

        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)

            # Approval stats
            stmt = text("""
                SELECT
                    COUNT(*) FILTER (WHERE decision = 'approve') as approves,
                    COUNT(*) FILTER (WHERE decision = 'reject') as rejects,
                    COUNT(*) FILTER (WHERE decision = 'approve' AND rationale IS NOT NULL AND LENGTH(rationale) > 20) as approved_with_comment
                FROM intervention_approvals
                WHERE decided_at >= :cutoff
            """)

            result = db.execute(stmt, {"cutoff": cutoff})
            row = result.fetchone()

            if not row or row[0] == 0:
                return {
                    "failure_mode": "FM_IRL-03: Risk Score Gaming",
                    "severity": "HEALTHY",
                    "indicators": {},
                    "message": "No approvals yet"
                }

            approves = row[0] or 0
            rejects = row[1] or 0
            with_comment = row[2] or 0

            total_decisions = approves + rejects
            approve_rate = approves / total_decisions if total_decisions > 0 else 0.0
            comment_rate = with_comment / approves if approves > 0 else 0.0

            # Severity: high approve rate + low comment rate
            if approve_rate > 0.9 and comment_rate < 0.2:
                severity = "CRITICAL"
            elif approve_rate > 0.8 and comment_rate < 0.3:
                severity = "DEGRADED"
            else:
                severity = "HEALTHY"

            return {
                "failure_mode": "FM_IRL-03: Risk Score Gaming (Human Side)",
                "severity": severity,
                "indicators": {
                    "total_approvals_30d": total_decisions,
                    "approve_count": approves,
                    "reject_count": rejects,
                    "approve_rate": round(approve_rate, 3),
                    "approved_with_comment": with_comment,
                    "comment_rate": round(comment_rate, 3)
                },
                "message": f"Approve rate: {approve_rate:.1%}, Comment rate: {comment_rate:.1%}",
                "behavioral_risk": "Human fatigue or overtrust"
            }

        finally:
            db.close()

    # =============================================================================
    # FM-IRL-04: Intervention Drift
    # =============================================================================

    def _check_intervention_drift(self) -> Dict:
        """
        Risk: Same intervention types proposed repeatedly.

        Indicators:
        - intervention_type frequency skew
        - repeated rejects with same hypothesis
        """

        db = get_db_sync()

        try:
            # Type distribution
            stmt = text("""
                SELECT
                    intervention_type,
                    COUNT(*) as count
                FROM intervention_candidates
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY intervention_type
                ORDER BY count DESC
            """)

            result = db.execute(stmt)
            type_counts = {row[0]: row[1] for row in result}

            total = sum(type_counts.values())
            if total == 0:
                return {
                    "failure_mode": "FM_IRL-04: Intervention Drift",
                    "severity": "HEALTHY",
                    "indicators": {},
                    "message": "No candidates yet"
                }

            # Check for skew (one type > 70%)
            max_count = max(type_counts.values())
            max_ratio = max_count / total if total > 0 else 0.0
            dominant_type = max(type_counts, key=type_counts.get)

            # Check for repeated rejects
            stmt_reject = text("""
                SELECT
                    intervention_type,
                    COUNT(*) FILTER (WHERE status = 'rejected') as reject_count
                FROM intervention_candidates
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY intervention_type
                HAVING COUNT(*) FILTER (WHERE status = 'rejected') > 2
            """)

            result_reject = db.execute(stmt_reject)
            repeated_rejects = {row[0]: row[1] for row in result_reject}

            # Severity
            if max_ratio > 0.8 or len(repeated_rejects) > 0:
                severity = "DEGRADED"
            elif max_ratio > 0.7:
                severity = "HEALTHY"  # Acceptable skew
            else:
                severity = "HEALTHY"

            return {
                "failure_mode": "FM_IRL-04: Intervention Drift",
                "severity": severity,
                "indicators": {
                    "total_candidates_30d": total,
                    "type_distribution": type_counts,
                    "dominant_type": dominant_type,
                    "dominant_ratio": round(max_ratio, 3),
                    "repeated_rejects": repeated_rejects
                },
                "message": f"Dominant type: {dominant_type} ({max_ratio:.1%})"
            }

        finally:
            db.close()

    # =============================================================================
    # FM-IRL-05: Semantic Overconfidence
    # =============================================================================

    def _check_semantic_overconfidence(self) -> Dict:
        """
        Risk: Human-readable explanation sounds better than data.

        Indicators:
        - Approve correlates with explanation length
        - High approve despite high risk
        """

        db = get_db_sync()

        try:
            # Correlation check: do LOW risk get approved more than HIGH risk?
            stmt = text("""
                SELECT
                    irs.risk_tier,
                    COUNT(*) FILTER (WHERE ic.status = 'approved') as approved,
                    COUNT(*) as total
                FROM intervention_candidates ic
                JOIN intervention_risk_scores irs ON irs.intervention_id = ic.id
                WHERE ic.created_at >= NOW() - INTERVAL '30 days'
                GROUP BY irs.risk_tier
            """)

            result = db.execute(stmt)
            tier_stats = {}

            for row in result:
                tier = row[0]
                approved = row[1] or 0
                total = row[2] or 0
                approve_rate = approved / total if total > 0 else 0.0

                tier_stats[tier] = {
                    "approved": approved,
                    "total": total,
                    "approve_rate": round(approve_rate, 3)
                }

            # Severity: if HIGH has high approve rate
            high_approve_rate = tier_stats.get("HIGH", {}).get("approve_rate", 0.0)

            if high_approve_rate > 0.5:
                severity = "CRITICAL"
            elif high_approve_rate > 0.3:
                severity = "DEGRADED"
            else:
                severity = "HEALTHY"

            return {
                "failure_mode": "FM_IRL-05: Semantic Overconfidence",
                "severity": severity,
                "indicators": {
                    "tier_approval_rates": tier_stats
                },
                "message": f"HIGH risk approve rate: {high_approve_rate:.1%} (should be ~0%)",
                "cognitive_risk": "Decisions based on narrative, not data"
            }

        finally:
            db.close()

    # =============================================================================
    # FM-IRL-06: Silent IRL
    # =============================================================================

    def _check_silent_irl(self) -> Dict:
        """
        Risk: IRL long silent while system degrades.

        Indicators:
        - Alerts exist, but no candidates
        - Degradation in truth views without interventions
        """

        db = get_db_sync()

        try:
            # Compare alerts vs candidates
            stmt_alerts = text("""
                SELECT COUNT(*) as count
                FROM system_alerts
                WHERE resolved = false
                  AND created_at >= NOW() - INTERVAL '7 days'
            """)

            result_alerts = db.execute(stmt_alerts)
            active_alerts = result_alerts.scalar() or 0

            stmt_candidates = text("""
                SELECT COUNT(*) as count
                FROM intervention_candidates
                WHERE created_at >= NOW() - INTERVAL '7 days'
            """)

            result_candidates = db.execute(stmt_candidates)
            recent_candidates = result_candidates.scalar() or 0

            # Severity: alerts without candidates
            if active_alerts > 5 and recent_candidates == 0:
                severity = "CRITICAL"
            elif active_alerts > 3 and recent_candidates == 0:
                severity = "DEGRADED"
            else:
                severity = "HEALTHY"

            return {
                "failure_mode": "FM_IRL-06: Silent IRL",
                "severity": severity,
                "indicators": {
                    "active_alerts_7d": active_alerts,
                    "candidates_7d": recent_candidates
                },
                "message": f"Active alerts: {active_alerts}, Candidates: {recent_candidates}",
                "risk": "IRL may be too conservative or broken"
            }

        finally:
            db.close()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

irl_health_metrics = IRLHealthMetrics()
