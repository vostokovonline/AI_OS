"""
Event-Driven Architecture for AI-OS

Минимальный Event Bus с handler'ами.
Работает in-memory, масштабируется.
"""

from typing import Dict, List, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
import asyncio


@dataclass
class Event:
    """Базовый класс события"""
    event_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalCreatedEvent(Event):
    """Goal создан"""
    def __init__(self, goal_id: str, **kwargs):
        super().__init__("GoalCreated", data={"goal_id": goal_id, **kwargs})


@dataclass
class GoalCompletedEvent(Event):
    """Goal выполнен - триггерит propagation"""
    def __init__(self, goal_id: str, parent_id: str = None, **kwargs):
        super().__init__("GoalCompleted", data={"goal_id": goal_id, "parent_id": parent_id, **kwargs})


@dataclass
class GoalStuckEvent(Event):
    """Goal завис - триггерит watchdog"""
    def __init__(self, goal_id: str, stuck_minutes: int, **kwargs):
        super().__init__("GoalStuck", data={"goal_id": goal_id, "stuck_minutes": stuck_minutes, **kwargs})


@dataclass
class GoalFailedEvent(Event):
    """Goal упал"""
    def __init__(self, goal_id: str, error: str, **kwargs):
        super().__init__("GoalFailed", data={"goal_id": goal_id, "error": error, **kwargs})


# Type alias для handler
EventHandler = Callable[[Event], Any]


class EventBus:
    """
    Минимальный Event Bus.
    
    Подписки: event_type -> [handlers]
    Публикация: event -> все подписанные handlers
    """
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.event_history: List[Event] = []
        self.max_history = 1000
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """Подписать handler на событие"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        if handler not in self.handlers[event_type]:
            self.handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> None:
        """Отписать handler"""
        if event_type in self.handlers:
            self.handlers[event_type] = [h for h in self.handlers[event_type] if h != handler]
    
    async def publish(self, event: Event) -> List[Any]:
        """Публикация события - вызывает все подписанные handlers"""
        results = []
        
        handlers = self.handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(event)
                else:
                    result = handler(event)
                results.append(result)
            except Exception as e:
                print(f"[EventBus] Handler error for {event.event_type}: {e}")
        
        # История событий
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history = self.event_history[-self.max_history:]
        
        return results
    
    def get_handlers(self, event_type: str) -> List[EventHandler]:
        """Получить список handlers для типа события"""
        return self.handlers.get(event_type, [])
    
    def get_history(self, event_type: str = None, limit: int = 50) -> List[Event]:
        """Получить историю событий"""
        if event_type:
            return [e for e in self.event_history[-limit:] if e.event_type == event_type]
        return self.event_history[-limit:]


# Глобальный инстанс
_event_bus: EventBus = None


def get_event_bus() -> EventBus:
    """Получить глобальный Event Bus"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# ============================================================
# HANDLERS (реализации логики)
# ============================================================

async def propagate_progress_handler(event: Event):
    """
    Handler: при completion обновить прогресс родителя.
    Вызывает progress_propagation.
    """
    from progress_propagation import on_goal_completed
    
    goal_id = event.data.get("goal_id")
    if not goal_id:
        return
    
    # Получаем session через контекст (упрощённо)
    from database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        from models import Goal
        goal = await session.get(Goal, goal_id)
        if goal:
            await on_goal_completed(session, goal)


async def trigger_execution_handler(event: Event):
    """
    Handler: при ready goal запустить execution.
    """
    goal_id = event.data.get("goal_id")
    if not goal_id:
        return
    
    from tasks import execute_goal_task
    execute_goal_task.delay(str(goal_id))


async def recovery_handler(event: Event):
    """
    Handler: при stuck goal — восстановить через Execution Kernel.
    """
    goal_id = event.data.get("goal_id")
    if not goal_id:
        return
    
    from execution_dynamics import dispatch_goal
    try:
        ex_result = await dispatch_goal(goal_id=goal_id)
        logger.info(f"recovery_dispatch goal_id={goal_id} success={ex_result.success}")
    except Exception as e:
        logger.error(f"recovery_failed goal_id={goal_id} error={e}")


# ============================================================
# ПОДПИСКИ (настройка системы)
# ============================================================

def setup_event_subscriptions():
    """Настроить все подписки при старте системы"""
    bus = get_event_bus()
    
    # GoalCompleted → propagate progress
    bus.subscribe("GoalCompleted", propagate_progress_handler)
    
    # GoalCompleted → trigger next execution (если есть ready)
    bus.subscribe("GoalCompleted", trigger_execution_handler)
    
    # GoalStuck → recovery
    bus.subscribe("GoalStuck", recovery_handler)
    
    print("[EventBus] Subscriptions configured")


# ============================================================
# УДОБНЫЕ ФУНКЦИИ ДЛЯ ВЫЗОВА
# ============================================================

async def emit_goal_created(goal_id: str, **kwargs) -> None:
    """Эмитить событие создания цели"""
    event = GoalCreatedEvent(goal_id, **kwargs)
    await get_event_bus().publish(event)


async def emit_goal_completed(goal_id: str, parent_id: str = None, **kwargs) -> None:
    """Эмитить событие completion цели"""
    event = GoalCompletedEvent(goal_id, parent_id, **kwargs)
    await get_event_bus().publish(event)


async def emit_goal_stuck(goal_id: str, stuck_minutes: int, **kwargs) -> None:
    """Эмитить событие stuck"""
    event = GoalStuckEvent(goal_id, stuck_minutes, **kwargs)
    await get_event_bus().publish(event)


# Alias для импорта
__all__ = [
    "EventBus",
    "Event", 
    "GoalCreatedEvent",
    "GoalCompletedEvent",
    "GoalStuckEvent",
    "GoalFailedEvent",
    "get_event_bus",
    "setup_event_subscriptions",
    "emit_goal_created",
    "emit_goal_completed",
    "emit_goal_stuck",
]