"""
Domain Exceptions for Goal System

Иерархия исключений для бизнес-логики целей.
Все исключения наследуются от BaseGoalException.

Author: AI-OS Core Team
Date: 2026-02-06
"""


class BaseGoalException(Exception):
    """Базовое исключение для всех ошибок бизнес-логики целей"""

    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self):
        """Сериализация в словарь для API response"""
        return {
            "error": {
                "code": self.__class__.__name__,
                "message": self.message,
                "details": self.details
            }
        }


# =============================================================================
# Goal Completion Exceptions (Phase 2.2)
# =============================================================================

class GoalAlreadyDone(BaseGoalException):
    """Цель уже завершена (повторный approval)"""

    def __init__(self, goal_id: str, approved_at: str):
        super().__init__(
            message="Goal completion has already been approved",
            details={
                "goal_id": goal_id,
                "approved_at": approved_at
            }
        )


class GoalChildrenIncomplete(BaseGoalException):
    """Нельзя завершить родителя с активными детьми"""

    def __init__(self, goal_id: str, remaining_children: int):
        super().__init__(
            message="Cannot approve completion while child goals are not done",
            details={
                "goal_id": goal_id,
                "remaining_children": remaining_children
            }
        )


class InsufficientAuthority(BaseGoalException):
    """Недостаточный уровень полномочий"""

    def __init__(self, required_level: int, provided_level: int):
        super().__init__(
            message="Authority level is insufficient to approve this goal",
            details={
                "required_level": required_level,
                "provided_level": provided_level
            }
        )


class InvalidGoalState(BaseGoalException):
    """Цель не в состоянии, допускающем завершение"""

    def __init__(self, goal_id: str, current_state: str, expected_states: list):
        super().__init__(
            message="Goal is not in a completable state",
            details={
                "goal_id": goal_id,
                "current_state": current_state,
                "expected_states": expected_states
            }
        )


class InvalidCompletionMode(BaseGoalException):
    """Попытка approve для не-MANUAL цели"""

    def __init__(self, goal_id: str, completion_mode: str):
        super().__init__(
            message="Only MANUAL goals can be approved",
            details={
                "goal_id": goal_id,
                "completion_mode": completion_mode
            }
        )


class InvariantViolation(BaseGoalException):
    """Нарушение инварианта системы (критическая ошибка)"""

    def __init__(self, invariant: str, goal_id: str = None):
        super().__init__(
            message=f"System invariant violated: {invariant}",
            details={
                "goal_id": goal_id,
                "invariant": invariant
            }
        )


# =============================================================================
# HTTP Status Mapping
# =============================================================================

# Mapping domain exceptions to HTTP status codes
EXCEPTION_TO_STATUS = {
    GoalAlreadyDone: 409,
    GoalChildrenIncomplete: 409,
    InsufficientAuthority: 403,
    InvalidGoalState: 409,
    InvalidCompletionMode: 400,
    InvariantViolation: 500,
}
