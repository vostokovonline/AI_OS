"""
GOAL TRANSITION SERVICE v3.0 - Pure Application Operation
====================================================

ARCHITECTURE:
- Domain Layer: goal_domain_service.py - чистые бизнес-правила
- Application Layer: goal_transition_service.py - оркестрация без транзакций
- Infrastructure: infrastructure/uow.py - управление транзакциями

THIS FILE IS NOW A THIN WRAPPER WITHOUT TRANSACTION MANAGEMENT.

Author: AI-OS Core Team
Date: 2026-02-12
"""
from typing import Dict, Optional, Any
from datetime import datetime
from enum import Enum
from uuid import UUID

from models import Goal


class TransitionResult(Enum):
    """Result of state transition attempt"""
    SUCCESS = "success"
    BLOCKED = "blocked"
    FAILED = "failed"


class GoalTransitionService:
    """
    Application Layer Orchestrator - БЕЗ управления транзакциями.
    
    Всё управление транзакциями теперь в UnitOfWork.
    Этот сервис только координирует вызовы доменного слоя.
    """
    
    def __init__(self):
        from domain.goal_domain_service import (
            GoalState, 
            GoalDomainService, 
            GoalTransitioned
        )
        from infrastructure.uow import GoalRepository, AuditLogger
        
        self._domain = GoalDomainService()
        self._state_enum = GoalState
        self._repository = GoalRepository()
        self._logger = AuditLogger()
    
    async def transition(
        self,
        uow: "UnitOfWork",
        goal_id: UUID,
        new_state: str,
        reason: str,
        actor: str = "system"
    ) -> Dict[str, Any]:
        """
        Application-level transition WITHOUT transaction management.
        
        Транзакцией управляет вызывающий код через UnitOfWork.
        
        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: UUID цели
            new_state: Новое состояние (строка)
            reason: Причина перехода
            actor: Кто инициировал
            
        Returns:
            Transition result dict
            
        Raises:
            ValueError: При нарушении бизнес-правил
        """
        # Валидация входных данных
        if not isinstance(goal_id, UUID):
            goal_id = UUID(str(goal_id))
        
        goal_state = self._state_enum(new_state)
        
        # Логируем начало
        print(f"\n🔄 GOAL TRANSITION: {goal_id}")
        print(f"   → State: {new_state}")
        print(f"   → Actor: {actor}")
        print(f"   → Reason: {reason}")
        print("=" * 70)
        
        try:
            # 1. Загружаем цель с pessimistic lock
            goal = await self._repository.get_for_update(uow.session, goal_id)
            
            if not goal:
                raise ValueError(f"Goal not found: {goal_id}")
            
            from_state = goal._status
            
            # 2. Делегируем доменному слою (валидация + изменение)
            event = self._domain.transition(goal, goal_state, reason)
            
            # 3. Логируем успешный переход
            await self._logger.log_transition(
                session=uow.session,
                goal_id=str(goal_id),
                goal_type=getattr(goal, 'goal_type', 'unknown'),
                from_state=from_state,
                to_state=new_state,
                reason=reason,
                actor=actor
            )
            
            print(f"  ✅ Transition: SUCCESS ({from_state} → {new_state})")
            print(f"{'='*70}\n")
            
            return {
                "result": TransitionResult.SUCCESS.value,
                "goal_id": str(goal_id),
                "from_state": from_state,
                "to_state": new_state,
                "reason": reason,
                "event": {
                    "type": "GoalTransitioned",
                    "timestamp": event.timestamp
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except ValueError as e:
            # Бизнес-правило нарушено
            print(f"  ❌ Transition BLOCKED: {e}")
            print(f"{'='*70}\n")
            
            await self._logger.log_violation(
                session=uow.session,
                goal_id=str(goal_id),
                goal_type=getattr(goal, 'goal_type', 'unknown'),
                reason=str(e)
            )
            
            return {
                "result": TransitionResult.BLOCKED.value,
                "goal_id": str(goal_id),
                "blocked_reason": str(e),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            # Непредвиденная ошибка
            print(f"  ❌ Transition FAILED: {e}")
            print(f"{'='*70}\n")
            raise


class BulkTransitionService:
    """
    Bulk transition service для множественных переходов.
    
    Все переходы в одной транзакции - критично для согласованности.
    """
    
    def __init__(self):
        from domain.goal_domain_service import GoalState
        from infrastructure.uow import GoalRepository, AuditLogger
        
        self._state_enum = GoalState
        self._repository = GoalRepository()
        self._logger = AuditLogger()
    
    async def transition_many(
        self,
        uow: "UnitOfWork",
        transitions: list[Dict],
        actor: str = "system"
    ) -> Dict[str, Any]:
        """
        Выполнить множественные переходы в одной транзакции.
        
        Args:
            uow: UnitOfWork с транзакцией
            transitions: Список [{"goal_id": UUID, "new_state": str, "reason": str}]
            actor: Кто инициировал
            
        Returns:
            Результаты для каждого перехода
        """
        results = []
        goal_ids = [UUID(t["goal_id"]) for t in transitions]
        
        # Загружаем все цели с lock
        goals = await self._repository.bulk_get_for_update(uow.session, goal_ids)
        
        for i, (trans, goal) in enumerate(zip(transitions, goals)):
            goal_id = UUID(trans["goal_id"])
            new_state = trans["new_state"]
            reason = trans["reason"]
            
            try:
                goal_state = self._state_enum(new_state)
                old_state = goal._status
                
                # Делегируем домену
                from domain.goal_domain_service import goal_domain_service
                event = goal_domain_service.transition(goal, goal_state, reason)
                
                results.append({
                    "goal_id": str(goal_id),
                    "result": "success",
                    "from_state": old_state,
                    "to_state": new_state
                })
                
            except ValueError as e:
                results.append({
                    "goal_id": str(goal_id),
                    "result": "blocked",
                    "reason": str(e)
                })
            except Exception as e:
                results.append({
                    "goal_id": str(goal_id),
                    "result": "failed",
                    "error": str(e)
                })
        
        return {
            "total": len(transitions),
            "success": sum(1 for r in results if r["result"] == "success"),
            "blocked": sum(1 for r in results if r["result"] == "blocked"),
            "failed": sum(1 for r in results if r["result"] == "failed"),
            "results": results
        }


# =============================================================================
# CONVENIENCE FUNCTIONS (для совместимости)
# =============================================================================

async def transition_goal(
    goal_id: str,
    new_state: str,
    reason: str,
    actor: str = "system"
) -> Dict[str, Any]:
    """
    Convenience wrapper - создаёт UoW для одного перехода.
    
    DEPRECATED: Для нового кода используйте:
    
    async with uow_factory() as uow:
        await transition_service.transition(uow, goal_id, new_state, reason)
    
    Args:
        goal_id: ID цели
        new_state: Новое состояние
        reason: Причина
        actor: Кто инициировал
        
    Returns:
        Transition result
    """
    from infrastructure.uow import UnitOfWork, create_uow_provider
    from database import AsyncSessionLocal
    
    uow_provider = create_uow_provider()
    
    async with uow_provider() as uow:
        service = GoalTransitionService()
        return await service.transition(
            uow=uow,
            goal_id=UUID(goal_id),
            new_state=new_state,
            reason=reason,
            actor=actor
        )


# Глобальный экземпляр
transition_service = GoalTransitionService()
bulk_transition_service = BulkTransitionService()
