"""
IRL INVARIANTS CONTRACT
====================================

Long-Term-Support (LTS) Architectural Guarantees

Формализует STEP 2.7 инварианты как код.
НЕТ write access, NO behavioural changes.

Эпистемологический контракт:
Система может лучше понять свои границы, чем расширить их.
"""

from typing import Dict, List
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, and_, func, text
from database import get_db_sync
from models import (
    InterventionCandidate,
    InterventionSimulation,
    InterventionRiskScore,
    SystemAlert
)


class IRLInvariant:
    """Один архитектурный инвариант."""

    def __init__(self, name: str, description: str, check_fn):
        self.name = name
        self.description = description
        self.check_fn = check_fn

    def verify(self) -> Dict:
        """Проверяет инвариант."""
        try:
            result = self.check_fn()
            return {
                "invariant": self.name,
                "status": "PASS" if result["passed"] else "VIOLATION",
                "description": self.description,
                **result
            }
        except Exception as e:
            return {
                "invariant": self.name,
                "status": "ERROR",
                "description": self.description,
                "error": str(e)
            }


class IRLInvariantsContract:
    """
    Контракт архитектурных инвариантов IRL.

    Все инварианты ДОЛЖНЫ выполняться всегда.
    Если хотя бы один нарушен — IRL считается compromised.
    """

    # =============================================================================
    # ИНВАРИАНТ 1: NO WRITE ACCESS TO INFERENCE
    # =============================================================================

    @staticmethod
    def _check_no_write_access() -> Dict:
        """
        IRL не имеет права писать в модели/порога/конфигурацию inference.

        Проверка:
        - В intervention_candidates нет типов "update_model", "tune_threshold"
        - В target_scope нет ключей "model_id", "threshold_id"
        """

        db = get_db_sync()

        try:
            # Forbidden intervention types
            forbidden_types = [
                "update_model",
                "retrain_model",
                "tune_threshold",
                "modify_weights",
                "alter_config"
            ]

            # Check for forbidden types
            stmt = select(InterventionCandidate).where(
                InterventionCandidate.intervention_type.in_(forbidden_types)
            )

            result = db.execute(stmt)
            forbidden_count = result.scalar() or 0

            # Check target_scope for forbidden keys
            stmt_all = select(InterventionCandidate)
            result_all = db.execute(stmt_all)
            all_candidates = result_all.scalars().all()

            forbidden_keys = ["model_id", "threshold_id", "config_id"]
            scope_violations = []

            for c in all_candidates:
                if c.target_scope:
                    for key in forbidden_keys:
                        if key in str(c.target_scope):
                            scope_violations.append(str(c.id))

            passed = (forbidden_count == 0) and (len(scope_violations) == 0)

            return {
                "passed": passed,
                "forbidden_interventions": forbidden_count,
                "forbidden_scope_keys": len(scope_violations),
                "message": "No write access to inference" if passed else f"Found {forbidden_count} forbidden interventions"
            }

        finally:
            db.close()

    # =============================================================================
    # ИНВАРИАНант 2: APPROVE ≠ EXECUTE
    # =============================================================================

    @staticmethod
    def _check_approve_not_execute() -> Dict:
        """
        Approve НЕ означает автоматическое execution.

        Проверка:
        - Approved interventions не применяются автоматически
        - Между approve и execution должно быть человеческое решение
        """

        db = get_db_sync()

        try:
            # Count approved interventions
            stmt_approved = select(InterventionCandidate).where(
                InterventionCandidate.status == "approved"
            )

            result = db.execute(stmt_approved)
            approved_count = result.scalar() or 0

            # Проверка: в системе нет кода, который auto-applies approved
            # Это architectural check — ищем подозрительные паттерны
            # В данном случае — проверяем, что нет кандидатов с status="executed"

            stmt_executed = select(InterventionCandidate).where(
                InterventionCandidate.status == "executed"
            )

            result_executed = db.execute(stmt_executed)
            executed_count = result_executed.scalar() or 0

            passed = executed_count == 0

            return {
                "passed": passed,
                "approved_count": approved_count,
                "executed_count": executed_count,
                "message": "Approve ≠ execute enforced" if passed else f"Found {executed_count} auto-executed interventions"
            }

        finally:
            db.close()

    # =============================================================================
    # ИНВАРИАНТ 3: CRITICAL RISK FORBIDDEN
    # =============================================================================

    @staticmethod
    def _check_critical_forbidden() -> Dict:
        """
        CRITICAL risk interventions НЕ могут быть approved.

        Проверка:
        - Нет approved interventions с risk_tier = CRITICAL
        """

        db = get_db_sync()

        try:
            stmt = text("""
                SELECT COUNT(*) as count
                FROM intervention_candidates ic
                JOIN intervention_risk_scores irs ON irs.intervention_id = ic.id
                WHERE ic.status = 'approved'
                  AND irs.risk_tier = 'CRITICAL'
            """)

            result = db.execute(stmt)
            row = result.fetchone()
            critical_approved = row[0] if row else 0

            passed = critical_approved == 0

            return {
                "passed": passed,
                "critical_approved_count": critical_approved,
                "message": "CRITICAL risk forbidden enforced" if passed else f"Found {critical_approved} CRITICAL approved interventions"
            }

        finally:
            db.close()

    # =============================================================================
    # ИНВАРИАНТ 4: SIMULATION ≠ PREDICTION
    # =============================================================================

    @staticmethod
    def _check_simulation_not_prediction() -> Dict:
        """
        Simulation — это replay, НЕ future prediction.

        Проверка:
        - У всех simulation есть determinism_hash
        - У всех simulation есть replay_window
        - У всех simulation есть metrics_before AND metrics_after
        """

        db = get_db_sync()

        try:
            stmt = select(InterventionSimulation)

            result = db.execute(stmt)
            simulations = result.scalars().all()

            violations = []

            for sim in simulations:
                if not sim.determinism_hash:
                    violations.append(f"{sim.id}: missing determinism_hash")

                if not sim.replay_window:
                    violations.append(f"{sim.id}: missing replay_window")

                if not sim.metrics_before or not sim.metrics_after:
                    violations.append(f"{sim.id}: missing before/after metrics")

            passed = len(violations) == 0

            return {
                "passed": passed,
                "total_simulations": len(simulations),
                "violations": violations,
                "message": "Simulation = replay enforced" if passed else f"Found {len(violations)} simulation violations"
            }

        finally:
            db.close()

    # =============================================================================
    # ИНВАРИАНТ 5: RISK ≥ GAIN CHECK
    # =============================================================================

    @staticmethod
    def _check_risk_gain_check() -> Dict:
        """
        Если risk ≥ gain → кандидат не предлагается.

        Проверка:
        - Нет proposed/simulated candidates где estimated_risk >= expected_gain
        """

        db = get_db_sync()

        try:
            stmt = select(InterventionCandidate).where(
                and_(
                    InterventionCandidate.status.in_(["proposed", "simulated"]),
                    InterventionCandidate.estimated_risk >= InterventionCandidate.expected_gain
                )
            )

            result = db.execute(stmt)
            violations = result.scalars().all()

            passed = len(list(violations)) == 0

            return {
                "passed": passed,
                "violation_count": len(list(violations)),
                "message": "Risk ≥ gain check enforced" if passed else f"Found {len(list(violations))} candidates with risk ≥ gain"
            }

        finally:
            db.close()

    # =============================================================================
    # ИНВАРИАНТ 6: HUMAN-IN-THE-LOOP MANDATORY
    # =============================================================================

    @staticmethod
    def _check_human_in_loop() -> Dict:
        """
        Все approve/reject требуют human decision.

        Проверка:
        - У всех approved/rejected есть InterventionApproval record
        - У всех approval records есть decided_by (не "system" для approve)
        """

        db = get_db_sync()

        try:
            # Check approved without approval record
            stmt_approved = text("""
                SELECT COUNT(*) as count
                FROM intervention_candidates ic
                LEFT JOIN intervention_approvals ia ON ia.intervention_id = ic.id
                WHERE ic.status IN ('approved', 'rejected')
                  AND ia.id IS NULL
            """)

            result = db.execute(stmt_approved)
            no_approval = result.fetchone()[0] if result else 0

            # Check system auto-approves (forbidden)
            stmt_system = text("""
                SELECT COUNT(*) as count
                FROM intervention_approvals
                WHERE decision = 'approve'
                  AND decided_by = 'system'
            """)

            result_system = db.execute(stmt_system)
            system_approves = result_system.fetchone()[0] if result_system else 0

            passed = (no_approval == 0) and (system_approves == 0)

            return {
                "passed": passed,
                "missing_approval_records": no_approval,
                "system_auto_approves": system_approves,
                "message": "Human-in-the-loop enforced" if passed else f"Found {no_approval} missing approvals, {system_approves} system approves"
            }

        finally:
            db.close()

    # =============================================================================
    # КОНТРАКТ
    # =============================================================================

    def __init__(self):
        self.invariants = [
            IRLInvariant(
                name="NO_WRITE_ACCESS_TO_INFERENCE",
                description="IRL has NO write access to models/thresholds/weights/configs",
                check_fn=self._check_no_write_access
            ),
            IRLInvariant(
                name="APPROVE_NOT_EXECUTE",
                description="Approve ≠ Execute (approve only permits future discussion)",
                check_fn=self._check_approve_not_execute
            ),
            IRLInvariant(
                name="CRITICAL_RISK_FORBIDDEN",
                description="CRITICAL risk interventions cannot be approved",
                check_fn=self._check_critical_forbidden
            ),
            IRLInvariant(
                name="SIMULATION_NOT_PREDICTION",
                description="Simulation = replay only, NOT future prediction",
                check_fn=self._check_simulation_not_prediction
            ),
            IRLInvariant(
                name="RISK_EXCEEDS_GAIN_CHECK",
                description="If risk ≥ gain → candidate not proposed",
                check_fn=self._check_risk_gain_check
            ),
            IRLInvariant(
                name="HUMAN_IN_THE_LOOP_MANDATORY",
                description="All approve/reject require human decision",
                check_fn=self._check_human_in_loop
            )
        ]

    def verify_all(self) -> Dict:
        """
        Проверяет все инварианты.

        Returns:
            {
                "overall_status": "PASS" | "VIOLATION" | "ERROR",
                "invariants": [ ... ],
                "timestamp": "..."
            }
        """

        results = [inv.verify() for inv in self.invariants]

        # Determine overall status
        violations = [r for r in results if r["status"] == "VIOLATION"]
        errors = [r for r in results if r["status"] == "ERROR"]

        if errors:
            overall_status = "ERROR"
        elif violations:
            overall_status = "VIOLATION"
        else:
            overall_status = "PASS"

        return {
            "overall_status": overall_status,
            "invariant_count": len(results),
            "violation_count": len(violations),
            "error_count": len(errors),
            "invariants": results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

irl_invariants_contract = IRLInvariantsContract()
