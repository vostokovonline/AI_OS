"""
Unit of Work Pattern + Audit Logger - Infrastructure Layer
=========================================================
"""
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy import select


class UnitOfWork:
    """
    Тонкий Unit of Work для управления транзакциями.
    
    Usage:
        async with UnitOfWork(session_factory) as uow:
            goal = await uow.goals.get(uow.session, goal_id)
            await transition_service.transition(uow, goal_id, "done")
    """
    
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
    
    async def __aenter__(self) -> "UnitOfWork":
        """Создаём сессию и начинаем транзакцию"""
        self._session = self._session_factory()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Коммит или rollback + закрытие сессии"""
        try:
            if exc_type is None:
                if self._session:
                    await self._session.commit()
            else:
                if self._session:
                    await self._session.rollback()
        finally:
            if self._session:
                await self._session.close()
                self._session = None
    
    @property
    def session(self) -> AsyncSession:
        """Доступ к текущей сессии"""
        if self._session is None:
            raise RuntimeError(
                "Session not available. Use 'async with UnitOfWork() as uow:' pattern."
            )
        return self._session


class GoalRepository:
    """Репозиторий для Goal - только CRUD"""
    
    def __init__(self, uow: UnitOfWork | None = None):
        self._uow = uow
    
    async def get(self, session, goal_id) -> "Goal":
        """Получить цель по ID"""
        from models import Goal
        
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_for_update(self, session, goal_id) -> "Goal":
        """
        Получить цель с pessimistic lock (SELECT ... FOR UPDATE).
        """
        from models import Goal
        
        stmt = (
            select(Goal)
            .where(Goal.id == goal_id)
            .with_for_update()
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def bulk_get_for_update(self, session, goal_ids) -> list:
        """Получить несколько целей с lock"""
        from models import Goal
        
        stmt = (
            select(Goal)
            .where(Goal.id.in_(goal_ids))
            .with_for_update()
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())
    
    async def save(self, session, goal) -> None:
        """Сохранить (add)"""
        session.add(goal)
    
    async def update(self, session, goal) -> None:
        """Обновить (flush)"""
        await session.flush(goal)


class AuditLogger:
    """Audit logging helper"""
    
    async def log_transition(
        self,
        session,
        goal_id: str,
        goal_type: str,
        from_state: str,
        to_state: str,
        reason: str,
        actor: str
    ) -> None:
        """Логировать успешный переход"""
        try:
            from audit_logger_v2 import audit_logger, AuditEventType
            await audit_logger.log(
                event_type=AuditEventType.STATE_TRANSITION,
                severity="INFO",
                goal_id=goal_id,
                goal_type=goal_type,
                from_state=from_state,
                to_state=to_state,
                reason=reason,
                actor=actor
            )
        except Exception:
            pass  # Logging не должен ломать transitions
    
    async def log_violation(
        self,
        session,
        goal_id: str,
        goal_type: str,
        reason: str
    ) -> None:
        """Логировать нарушение инварианта"""
        try:
            from audit_logger_v2 import audit_logger, AuditEventType
            await audit_logger.log(
                event_type=AuditEventType.INVARIANT_VIOLATION,
                severity="WARNING",
                goal_id=goal_id,
                goal_type=goal_type,
                reason=reason
            )
        except Exception:
            pass
    
    async def log_failure(
        self,
        session,
        goal_id: str,
        goal_type: str,
        from_state: str,
        to_state: str,
        error: str
    ) -> None:
        """Логировать ошибку перехода"""
        try:
            from audit_logger_v2 import audit_logger, AuditEventType
            await audit_logger.log(
                event_type=AuditEventType.STATE_TRANSITION_FAILED,
                severity="ERROR",
                goal_id=goal_id,
                goal_type=goal_type,
                from_state=from_state,
                to_state=to_state,
                error=error
            )
        except Exception:
            pass


def create_uow_provider() -> "UoWProvider":
    """
    Фабрика для создания UoW провайдера.
    
    Usage in FastAPI:
        from database import AsyncSessionLocal
        from infrastructure.uow import create_uow_provider
        
        get_uow = create_uow_provider()
        
        async def endpoint(uow: UnitOfWork = Depends(get_uow)):
            await uow.goals.get(uow.session, goal_id)
    """
    from database import AsyncSessionLocal
    
    class UoWProvider:
        def __init__(self, factory):
            self._factory = factory
        
        def __call__(self) -> UnitOfWork:
            return UnitOfWork(self._factory)
        
        @property
        def goals(self) -> GoalRepository:
            return GoalRepository()
    
    return UoWProvider(AsyncSessionLocal)
