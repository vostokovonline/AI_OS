"""
Goal Domain Service - Чистый доменный слой
=========================================
Никаких session, commit, async, логов, side-effects.
Только бизнес-логика и инварианты.
"""
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class GoalState(Enum):
    """Все возможные состояния цели"""
    PENDING = "pending"
    ACTIVE = "active"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"
    ONGOING = "ongoing"
    DONE = "done"
    FROZEN = "frozen"
    PERMANENT = "permanent"


class TransitionReason(Enum):
    """Типичные причины переходов"""
    INITIAL_CREATION = "Initial goal creation"
    DECOMPOSITION_STARTED = "Decomposition started"
    SKILL_STARTED = "Starting atomic goal execution"
    SKILL_FAILED = "Skill execution failed"
    ALL_CHILDREN_COMPLETED = "All children completed"
    NO_PASSED_ARTIFACTS = "No passed artifacts"
    BLOCKING_REMOVED = "Blocking removed"
    MANUAL_COMPLETION = "Manual completion"
    EVALUATION_PASSED = "Evaluation passed"
    EVALUATION_FAILED = "Evaluation failed"
    USER_REQUEST = "User request"
    SYSTEM_ERROR = "System error"


@dataclass
class GoalTransitioned:
    """Domain event - событие перехода состояния"""
    goal_id: str
    from_state: str
    to_state: str
    reason: str
    timestamp: str


class GoalDomainService:
    """
    ЧИСТАЯ доменная логика для переходов состояния цели.
    
    Responsibilities:
    - Валидация переходов (инварианты)
    - Изменение состояния
    - Генерация domain events
    
    НЕ делает:
    - commit/flush
    - async операции
    - логирование
    - side effects
    """
    
    # Терминальные состояния - из которых нельзя выйти
    TERMINAL_STATES = {GoalState.DONE, GoalState.FROZEN, GoalState.PERMANENT}
    
    # Валидные типы целей
    GOAL_TYPES = {"achievable", "continuous", "directional", "exploratory", "meta"}
    
    def __init__(self):
        self._transition_flag = False
    
    def _enable_transition(self):
        """Разрешить изменение _status"""
        self._transition_flag = True
    
    def _disable_transition(self):
        """Запретить изменение _status"""
        self._transition_flag = False
    
    def transition(
        self,
        goal,
        new_state: GoalState,
        reason: Optional[str] = None
    ) -> GoalTransitioned:
        """
        ЕДИНСТВЕННЫЙ способ изменить состояние цели в доменной логике.
        
        Args:
            goal: Объект Goal (должен иметь _status, id, goal_type)
            new_state: Новое состояние
            reason: Причина перехода
            
        Returns:
            GoalTransitioned: Domain event для последующей обработки
            
        Raises:
            ValueError: При нарушении инвариантов
        """
        if not hasattr(goal, '_status'):
            raise ValueError("Goal must have _status attribute")
        
        old_state = goal._status
        new_state_str = new_state.value if isinstance(new_state, GoalState) else new_state
        old_state_str = old_state
        
        # Валидация: No-op переходы запрещены
        if old_state == new_state_str:
            raise ValueError(f"No-op transition forbidden: goal already in '{old_state}' state")
        
        # Валидация: из терминального состояния нельзя выйти
        if old_state in [s.value for s in self.TERMINAL_STATES]:
            raise ValueError(
                f"Cannot transition from terminal state '{old_state}'. "
                f"Terminal states: {[s.value for s in self.TERMINAL_STATES]}"
            )
        
        # Валидация: специфичные правила для типов целей
        goal_type = getattr(goal, 'goal_type', None)
        if goal_type:
            self._validate_type_specific_rules(goal_type, new_state_str)
        
        # Валидация: разрешённые переходы
        self._validate_allowed_transition(old_state_str, new_state_str)
        
        # Разрешаем изменение _status (обходим защиту)
        self._enable_transition()
        try:
            goal._status = new_state_str
        finally:
            self._disable_transition()
        
        # Генерируем domain event
        from datetime import datetime, timezone
        event = GoalTransitioned(
            goal_id=str(goal.id),
            from_state=old_state_str,
            to_state=new_state_str,
            reason=reason or "State transition",
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        return event
    
    def _validate_type_specific_rules(self, goal_type: str, new_state: str):
        """Проверка правил, специфичных для типа цели"""
        if goal_type not in self.GOAL_TYPES:
            raise ValueError(f"Unknown goal_type: {goal_type}")
        
        # CRITICAL: Continuous goals не могут быть 'done'
        if goal_type == "continuous" and new_state == "done":
            raise ValueError(
                "CRITICAL INVARIANT: Continuous goals cannot be marked 'done'. "
                "Use 'ongoing' instead."
            )
        
        # CRITICAL: Directional goals не могут быть 'done'
        if goal_type == "directional" and new_state == "done":
            raise ValueError(
                "CRITICAL INVARIANT: Directional goals cannot be marked 'done'. "
                "Use 'permanent' instead."
            )
    
    def _validate_allowed_transition(self, from_state: str, to_state: str):
        """Проверка что переход между данными состояниями разрешён"""
        # Словарь разрешённых переходов
        allowed_transitions = {
            GoalState.PENDING.value: {
                GoalState.ACTIVE.value,
                GoalState.BLOCKED.value,
            },
            GoalState.ACTIVE.value: {
                GoalState.DONE.value,
                GoalState.INCOMPLETE.value,
                GoalState.BLOCKED.value,
                GoalState.ONGOING.value,  # continuous goals
            },
            GoalState.BLOCKED.value: {
                GoalState.ACTIVE.value,
                GoalState.INCOMPLETE.value,
            },
            GoalState.INCOMPLETE.value: {
                GoalState.ACTIVE.value,
                GoalState.BLOCKED.value,
            },
            GoalState.ONGOING.value: {
                GoalState.ACTIVE.value,
                GoalState.DONE.value,  # rare, but possible
            },
        }
        
        allowed = allowed_transitions.get(from_state, set())
        if to_state not in allowed:
            # Исключения для специфичных случаев
            if from_state == GoalState.INCOMPLETE.value and to_state == GoalState.DONE.value:
                # Можно перейти в done из incomplete если артефакты добавлены
                return
            
            raise ValueError(
                f"Invalid transition: cannot go from '{from_state}' to '{to_state}'. "
                f"Allowed from '{from_state}': {allowed or 'none'}"
            )


# Глобальный экземпляр для удобства
goal_domain_service = GoalDomainService()
