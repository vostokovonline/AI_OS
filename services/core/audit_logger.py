"""
AUDIT LOGGER STUB - Temporary simplified version
===================================================

Simplified audit logger without complex functionality.
To be replaced with full audit_logger.py after import issues are fixed.

Author: AI-OS Core Team
Date: 2026-02-11
"""

from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum
import json


class AuditEventType(str, Enum):
    GOAL_CREATED = "goal_created"
    STATE_TRANSITION = "state_transition"


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
        self.timestamp = datetime.now().isoformat()

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
    def __init__(self):
        self._audit_history = []

    def log_goal_created(self, goal_id: str, goal_type: str, title: str) -> AuditEntry:
        entry = AuditEntry(
            event_type=AuditEventType.GOAL_CREATED,
            severity=AuditSeverity.INFO,
            goal_id=goal_id,
            message=f"Goal created: {title}"
        )
        self._audit_history.append(entry)
        return entry

    def log_state_transition(self, goal_id: str, goal_type: str, from_state: str, to_state: str, reason: str, actor: str = "system") -> AuditEntry:
        entry = AuditEntry(
            event_type=AuditEventType.STATE_TRANSITION,
            severity=AuditSeverity.INFO,
            goal_id=goal_id,
            message=f"Transition: {from_state} -> {to_state} ({reason})"
        )
        self._audit_history.append(entry)
        return entry

    def log_invariant_violation(self, goal_id: str, goal_type: str, invariant_code: str, message: str, context: Dict[str, Any]) -> AuditEntry:
        entry = AuditEntry(
            event_type=AuditEventType.STATE_TRANSITION,  # Using STATE_TRANSITION for now
            severity=AuditSeverity.CRITICAL,
            goal_id=goal_id,
            message=f"Invariant violation: {invariant_code}"
        )
        self._audit_history.append(entry)
        return entry

    def get_audit_history(self) -> list:
        return getattr(self, '_audit_history', [])


audit_logger = AuditLogger()
