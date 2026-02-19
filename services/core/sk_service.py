"""
SURVIVABILITY KERNEL (SK) SERVICE
OCCP Draft 0.1 Compliance

Absolute authority layer protecting the system from:
- Semantic degradation
- Incentive capture
- Over-optimization
- "Successful destruction"

SK has ABSOLUTE veto authority.
SK cannot be modified once created.
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import select
from database import AsyncSessionLocal
from models import (
    SurvivabilityKernel, SKRule,
    SurvivabilitySignal, OCCPAuditEvent
)


class SKVetoError(Exception):
    """Raised when SK exercises veto authority"""
    def __init__(self, rule_id: str, explanation: str):
        self.rule_id = rule_id
        self.explanation = explanation
        super().__init__(f"SK veto by {rule_id}: {explanation}")


class SKService:
    """
    Survivability Kernel Service

    Enforces immutable rules that protect system viability.
    Can block ANY operation, including those from MCL.
    """

    async def get_active_kernel(self) -> Optional[Dict]:
        """Get active SK instance"""
        async with AsyncSessionLocal() as db:
            stmt = select(SurvivabilityKernel).where(
                SurvivabilityKernel.is_active == True
            ).order_by(SurvivabilityKernel.created_at.desc())

            result = await db.execute(stmt)
            kernel = result.scalar_one_or_none()

            if not kernel:
                return None

            return {
                "id": str(kernel.id),
                "version": kernel.version,
                "authority_level": kernel.authority_level,
                "self_modifiable": kernel.self_modifiable
            }

    async def check_veto(
        self,
        operation: str,
        component: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Check if SK vetoes operation

        Args:
            operation: Operation to check
            component: Component being affected
            context: Additional context

        Returns:
            {
                "vetoed": true/false,
                "rule_id": "...",
                "explanation": "..."
            }

        Raises:
            SKVetoError if operation is vetoed
        """
        async with AsyncSessionLocal() as db:
            # Get all active SK rules
            stmt = select(SKRule).where(SKRule.is_active == True)
            result = await db.execute(stmt)
            rules = result.scalars().all()

            # Check each rule
            for rule in rules:
                veto = await self._check_rule(rule, operation, component, db)

                if veto:
                    # Log veto
                    await self._log_audit(
                        db=db,
                        decision="veto",
                        decision_type="sk_rule_triggered",
                        blocked_component=component,
                        blocked_operation=operation,
                        rationale=rule.explanation,
                        context={
                            "rule_id": rule.rule_id,
                            "signal": rule.signal_name,
                            "threshold": rule.threshold
                        }
                    )

                    return {
                        "vetoed": True,
                        "rule_id": rule.rule_id,
                        "explanation": rule.explanation
                    }

            return {
                "vetoed": False,
                "rule_id": None,
                "explanation": "Operation allowed by SK"
            }

    async def _check_rule(
        self,
        rule: SKRule,
        operation: str,
        component: str,
        db
    ) -> bool:
        """
        Check individual SK rule

        Returns True if rule triggers veto
        """
        # Get current signal value
        signal_value = await self._get_signal_value(rule.signal_name, db)

        # Apply operator
        if rule.operator == ">":
            triggered = signal_value > rule.threshold
        elif rule.operator == "<":
            triggered = signal_value < rule.threshold
        elif rule.operator == "==":
            triggered = signal_value == rule.threshold
        elif rule.operator == ">=":
            triggered = signal_value >= rule.threshold
        elif rule.operator == "<=":
            triggered = signal_value <= rule.threshold
        elif rule.operator == "!=":
            triggered = signal_value != rule.threshold
        else:
            triggered = False

        if not triggered:
            return False

        # Check if operation matches rule's target
        for action in rule.actions:
            action_type = action.get("action")
            target = action.get("target", "*")

            # Check if this rule applies to the operation
            if target == "*" or target == operation:
                if action_type == "forbid_operation":
                    return True
                elif action_type == "freeze_component":
                    if component == target:
                        return True

        return False

    async def _get_signal_value(self, signal_name: str, db) -> float:
        """Get current value of survivability signal"""
        # Get most recent signal
        stmt = select(SurvivabilitySignal).where(
            SurvivabilitySignal.signal_name == signal_name
        ).order_by(SurvivabilitySignal.measured_at.desc()).limit(1)

        result = await db.execute(stmt)
        signal = result.scalar_one_or_none()

        if not signal:
            # Default signal values
            defaults = {
                "mission_drift": 0.0,
                "incentive_capture": 0.0,
                "over_optimization": 0.0,
                "irreversibility_risk": 0.0
            }
            return defaults.get(signal_name, 0.0)

        return signal.signal_value

    async def record_signal(
        self,
        signal_name: str,
        signal_value: float,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Record survivability signal measurement

        Args:
            signal_name: Name of signal
            signal_value: Measured value (0.0 - 1.0)
            context: Additional context

        Returns:
            Created signal
        """
        async with AsyncSessionLocal() as db:
            signal = SurvivabilitySignal(
                signal_name=signal_name,
                signal_value=signal_value,
                context=context or {}
            )

            db.add(signal)
            await db.commit()
            await db.refresh(signal)

            return {
                "id": str(signal.id),
                "signal_name": signal.signal_name,
                "signal_value": signal.signal_value,
                "measured_at": signal.measured_at.isoformat()
            }

    async def get_all_rules(self) -> List[Dict]:
        """Get all active SK rules"""
        async with AsyncSessionLocal() as db:
            stmt = select(SKRule).where(SKRule.is_active == True)
            result = await db.execute(stmt)
            rules = result.scalars().all()

            return [
                {
                    "id": str(rule.id),
                    "rule_id": rule.rule_id,
                    "signal_name": rule.signal_name,
                    "operator": rule.operator,
                    "threshold": rule.threshold,
                    "actions": rule.actions,
                    "explanation": rule.explanation
                }
                for rule in rules
            ]

    async def get_signals(self, signal_name: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get survivability signals

        Args:
            signal_name: Filter by signal name
            limit: Max results

        Returns:
            List of signals
        """
        async with AsyncSessionLocal() as db:
            stmt = select(SurvivabilitySignal)

            if signal_name:
                stmt = stmt.where(SurvivabilitySignal.signal_name == signal_name)

            stmt = stmt.order_by(SurvivabilitySignal.measured_at.desc()).limit(limit)

            result = await db.execute(stmt)
            signals = result.scalars().all()

            return [
                {
                    "id": str(s.id),
                    "signal_name": s.signal_name,
                    "signal_value": s.signal_value,
                    "measured_at": s.measured_at.isoformat(),
                    "context": s.context
                }
                for s in signals
            ]

    async def _log_audit(
        self,
        db,
        decision: str,
        decision_type: str,
        blocked_component: Optional[str] = None,
        blocked_operation: Optional[str] = None,
        rationale: str = "",
        context: Dict = None
    ):
        """Log SK decision to audit trail"""
        event = OCCPAuditEvent(
            source="SK",
            decision=decision,
            decision_type=decision_type,
            blocked_component=blocked_component,
            blocked_operation=blocked_operation,
            rationale=rationale,
            context=context or {}
        )
        db.add(event)


# Global instance
sk_service = SKService()
