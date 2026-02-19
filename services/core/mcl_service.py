"""
META-COGNITION LAYER (MCL) SERVICE
OCCP Draft 0.1 Compliance

Manages cognitive modes, allowable decision entropy,
and boundaries of goal interpretation.

MCL does NOT know what to do.
MCL knows HOW the system is allowed to think.
"""
import uuid
from typing import Dict, List, Optional, Literal
from datetime import datetime
from sqlalchemy import select, and_
from database import AsyncSessionLocal
from models import (
    MetaCognitionState, OCCPAuditEvent,
    VectorOperator, VectorApplication
)


class MCLMode:
    """Cognitive Modes as defined in OCCP Draft 0.1"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    PRESERVATION = "preservation"


class MCLService:
    """
    Meta-Cognition Layer Service

    Manages the cognitive state of the system.
    Does NOT perform execution or planning.
    """

    # Mode-specific configurations
    MODE_CONFIGS = {
        MCLMode.EXPLORATION: {
            "entropy_budget": 0.7,
            "epistemic_confidence": 0.3,
            "risk_posture": "aggressive"
        },
        MCLMode.EXPLOITATION: {
            "entropy_budget": 0.2,
            "epistemic_confidence": 0.8,
            "risk_posture": "conservative"
        },
        MCLMode.PRESERVATION: {
            "entropy_budget": 0.1,
            "epistemic_confidence": 0.9,
            "risk_posture": "conservative"
        }
    }

    async def get_active_state(self) -> Optional[Dict]:
        """Get current active MCL state"""
        async with AsyncSessionLocal() as db:
            stmt = select(MetaCognitionState).where(
                MetaCognitionState.is_active == True
            ).order_by(MetaCognitionState.created_at.desc())

            result = await db.execute(stmt)
            state = result.scalar_one_or_none()

            if not state:
                return None

            return {
                "id": str(state.id),
                "cognitive_mode": state.cognitive_mode,
                "epistemic_confidence": state.epistemic_confidence,
                "entropy_budget": state.entropy_budget,
                "drift_score": state.drift_score,
                "risk_posture": state.risk_posture,
                "last_reviewed_at": state.last_reviewed_at.isoformat() if state.last_reviewed_at else None
            }

    async def set_mode(
        self,
        mode: Literal["exploration", "exploitation", "preservation"],
        rationale: str
    ) -> Dict:
        """
        Set cognitive mode with automatic parameter adjustment

        Args:
            mode: Target cognitive mode
            rationale: Human-readable explanation

        Returns:
            Updated state
        """
        if mode not in [MCLMode.EXPLORATION, MCLMode.EXPLOITATION, MCLMode.PRESERVATION]:
            raise ValueError(f"Invalid mode: {mode}")

        config = self.MODE_CONFIGS[mode]

        async with AsyncSessionLocal() as db:
            # Get active state
            stmt = select(MetaCognitionState).where(
                MetaCognitionState.is_active == True
            )
            result = await db.execute(stmt)
            state = result.scalar_one_or_none()

            if not state:
                # Create new state
                state = MetaCognitionState(
                    cognitive_mode=mode,
                    epistemic_confidence=config["epistemic_confidence"],
                    entropy_budget=config["entropy_budget"],
                    risk_posture=config["risk_posture"]
                )
                db.add(state)
            else:
                # Update existing state
                old_mode = state.cognitive_mode
                state.cognitive_mode = mode
                state.epistemic_confidence = config["epistemic_confidence"]
                state.entropy_budget = config["entropy_budget"]
                state.risk_posture = config["risk_posture"]
                state.last_reviewed_at = datetime.now()

                # Log mode transition
                await self._log_audit(
                    db=db,
                    decision="mode_transition",
                    decision_type="cognitive_mode_change",
                    rationale=f"Mode changed from {old_mode} to {mode}: {rationale}",
                    context={"old_mode": old_mode, "new_mode": mode}
                )

            await db.commit()
            await db.refresh(state)

            return {
                "id": str(state.id),
                "cognitive_mode": state.cognitive_mode,
                "epistemic_confidence": state.epistemic_confidence,
                "entropy_budget": state.entropy_budget,
                "drift_score": state.drift_score,
                "risk_posture": state.risk_posture
            }

    async def check_operation_allowed(
        self,
        operation: str,
        component: str
    ) -> Dict:
        """
        Check if operation is allowed under current cognitive mode

        Args:
            operation: Operation to check (e.g., "vector.apply", "goal.create")
            component: Component being affected

        Returns:
            {
                "allowed": true/false,
                "reason": "...",
                "mode": "..."
            }
        """
        state = await self.get_active_state()

        if not state:
            # No active state = default to conservative
            return {
                "allowed": False,
                "reason": "No active MCL state",
                "mode": None
            }

        mode = state["cognitive_mode"]

        # Preservation mode restrictions
        if mode == MCLMode.PRESERVATION:
            # In preservation mode, forbid most operations
            if operation in ["vector.apply", "goal.create", "goal.decompose"]:
                return {
                    "allowed": False,
                    "reason": f"Preservation mode: {operation} is forbidden",
                    "mode": mode
                }

        # Exploration mode allows most operations
        if mode == MCLMode.EXPLORATION:
            return {
                "allowed": True,
                "reason": "Exploration mode: high entropy allowed",
                "mode": mode
            }

        # Exploitation mode: selective
        if mode == MCLMode.EXPLOITATION:
            # Allow execution, restrict new exploration
            if operation in ["goal.create", "goal.decompose"]:
                return {
                    "allowed": False,
                    "reason": "Exploitation mode: focus on execution, not exploration",
                    "mode": mode
                }

        return {
            "allowed": True,
            "reason": f"Operation allowed under {mode} mode",
            "mode": mode
        }

    async def update_drift_score(self, new_drift: float) -> Dict:
        """
        Update drift score (typically called by SK or monitoring)

        Args:
            new_drift: New drift score (0.0 - 1.0)

        Returns:
            Updated state
        """
        if not 0.0 <= new_drift <= 1.0:
            raise ValueError(f"Drift score must be 0.0-1.0, got {new_drift}")

        async with AsyncSessionLocal() as db:
            stmt = select(MetaCognitionState).where(
                MetaCognitionState.is_active == True
            )
            result = await db.execute(stmt)
            state = result.scalar_one_or_none()

            if not state:
                raise ValueError("No active MCL state")

            old_drift = state.drift_score
            state.drift_score = new_drift
            state.last_reviewed_at = datetime.now()

            # Auto-transition to preservation if drift is high
            if new_drift > 0.7 and state.cognitive_mode != MCLMode.PRESERVATION:
                old_mode = state.cognitive_mode
                state.cognitive_mode = MCLMode.PRESERVATION

                await self._log_audit(
                    db=db,
                    decision="auto_mode_transition",
                    decision_type="drift_response",
                    rationale=f"Drift {new_drift:.2f} > 0.7, forcing preservation mode",
                    context={
                        "old_mode": old_mode,
                        "new_mode": "preservation",
                        "drift_score": new_drift
                    }
                )

            await db.commit()
            await db.refresh(state)

            return {
                "id": str(state.id),
                "cognitive_mode": state.cognitive_mode,
                "drift_score": state.drift_score
            }

    async def get_allowed_operations(self) -> Dict:
        """
        Get list of allowed/forbidden operations under current mode

        Returns:
            {
                "mode": "...",
                "allowed": [...],
                "forbidden": [...]
            }
        """
        state = await self.get_active_state()

        if not state:
            return {
                "mode": None,
                "allowed": [],
                "forbidden": []
            }

        mode = state["cognitive_mode"]

        if mode == MCLMode.EXPLORATION:
            return {
                "mode": mode,
                "allowed": ["vector.apply", "goal.create", "goal.decompose", "goal.execute"],
                "forbidden": []
            }

        elif mode == MCLMode.EXPLOITATION:
            return {
                "mode": mode,
                "allowed": ["goal.execute", "vector.apply"],
                "forbidden": ["goal.create", "goal.decompose"]
            }

        elif mode == MCLMode.PRESERVATION:
            return {
                "mode": mode,
                "allowed": ["goal.execute"],
                "forbidden": ["vector.apply", "goal.create", "goal.decompose"]
            }

        return {
            "mode": mode,
            "allowed": [],
            "forbidden": []
        }

    async def _log_audit(
        self,
        db,
        decision: str,
        decision_type: str,
        rationale: str,
        context: Dict
    ):
        """Log MCL decision to audit trail"""
        event = OCCPAuditEvent(
            source="MCL",
            decision=decision,
            decision_type=decision_type,
            rationale=rationale,
            context=context
        )
        db.add(event)


# Global instance
mcl_service = MCLService()
