"""
AUDIT LOGGER V2 - Database-integrated audit trail
=====================================================

Phase 1 Control Evolution - all state transitions must be auditable.
This version writes transitions to goal_status_transitions table.

Author: AI-OS Core Team
Date: 2026-02-12
"""

from typing import Dict, Optional, Any
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID
import asyncio

# Database imports
from database import AsyncSessionLocal
from sqlalchemy import select


class AuditEventType(str, Enum):
    GOAL_CREATED = "goal_created"
    STATE_TRANSITION = "state_transition"
    INVARIANT_VIOLATION = "invariant_violation"


class AuditSeverity(str, Enum):
    INFO = "info"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEntry:
    def __init__(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        goal_id: str,
        message: str
    ):
        self.event_type = event_type.value
        self.severity = severity.value
        self.goal_id = goal_id
        self.message = message
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "goal_id": self.goal_id,
            "message": self.message,
            "timestamp": self.timestamp
        }


class StateTransitionLogger:
    def __init__(self, audit_logger):
        self.audit_logger = audit_logger

    def transition(self, goal, to_state, reason, actor="system") -> Dict:
        return self.audit_logger.log_state_transition(
            goal_id=str(goal.id),
            goal_type=goal.goal_type,
            from_state=getattr(goal, 'status', 'unknown'),
            to_state=to_state,
            reason=reason,
            actor=actor
        )


class AuditLogger:
    """
    Audit logger that writes to goal_status_transitions table.

    Phase 1 Control Evolution: Every state transition is recorded.
    """
    def __init__(self):
        self._audit_history = []

    async def log_goal_created(self, goal_id: str, goal_type: str, title: str) -> AuditEntry:
        """Log goal creation event"""
        entry = AuditEntry(
            event_type=AuditEventType.GOAL_CREATED,
            severity=AuditSeverity.INFO,
            goal_id=goal_id,
            message=f"Goal created: {title}"
        )
        self._audit_history.append(entry)
        return entry

    async def log_state_transition(
        self,
        goal_id: str,
        goal_type: str,
        from_state: str,
        to_state: str,
        reason: str,
        actor: str = "system"
    ) -> AuditEntry:
        """
        Log state transition to DATABASE (goal_status_transitions table).

        This is the core of Phase 1 Control Evolution - every transition
        goes through this method and is written to the audit trail.
        """
        # Create in-memory entry
        entry = AuditEntry(
            event_type=AuditEventType.STATE_TRANSITION,
            severity=AuditSeverity.INFO,
            goal_id=goal_id,
            message=f"Transition: {from_state} -> {to_state} ({reason})"
        )
        self._audit_history.append(entry)

        # Write to database
        try:
            async with AsyncSessionLocal() as db:
                from models import GoalStatusTransition

                # Convert string goal_id to UUID if needed
                goal_uuid = UUID(goal_id) if isinstance(goal_id, str) else goal_id

                # Create database record
                transition_record = GoalStatusTransition(
                    goal_id=goal_uuid,
                    from_status=from_state,
                    to_status=to_state,
                    reason=reason,
                    triggered_by=actor
                )

                db.add(transition_record)
                await db.commit()

                print(f"✅ [AUDIT] Transition logged: {goal_id[:8]}... | {from_state} -> {to_state} | by {actor}")

        except Exception as e:
            # Don't fail transition if audit logging fails
            print(f"⚠️  [AUDIT] Failed to log transition: {e}")
            # Still return entry for in-memory tracking

        return entry

    async def log_invariant_violation(
        self,
        goal_id: str,
        goal_type: str,
        invariant_code: str,
        message: str,
        context: Dict[str, Any]
    ) -> AuditEntry:
        """Log invariant violation event"""
        entry = AuditEntry(
            event_type=AuditEventType.INVARIANT_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            goal_id=goal_id,
            message=f"Invariant violation: {invariant_code} - {message}"
        )
        self._audit_history.append(entry)
        return entry

    def get_audit_history(self) -> list:
        """Get in-memory audit history"""
        return getattr(self, '_audit_history', [])


# Singleton instance
audit_logger = AuditLogger()
