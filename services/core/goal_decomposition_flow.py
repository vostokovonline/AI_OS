"""
goal_decomposition_flow - простой сценарий декомпозиции через ask_user

НЕ использует LLM
НЕ является скиллом
Просто спрашивает человека и создаёт подцели из его ответа
"""

from typing import List, Dict, Any
from datetime import datetime
import uuid
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal, Question
from canonical_skills.ask_user import ask_user


class GoalDecompositionFlow:
    """
    Простой сценарий декомпозиции целей через диалог с человеком
    
    Философия:
    - Система НЕ думает
    - Система НЕ декомпозирует автоматически
    - Система спрашивает человека
    - Человек отвечает текстом
    - Система сохраняет ответ как подцели
    """
    
    async def check_needs_decomposition(self, goal_id: str) -> bool:
        """
        Проверить, нужна ли декомпозиция
        
        Условие:
        - goal.is_atomic == False
        - Нет подцелей (parent_id == goal_id)
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if not goal:
                return False
            
            if goal.is_atomic:
                return False
            
            # Проверяем есть ли уже подцели
            children_stmt = select(Goal).where(Goal.parent_id == goal.id)
            children_result = await db.execute(children_stmt)
            children = children_result.scalars().all()
            
            return len(children) == 0
    
    async def ask_how_to_proceed(self, goal_id: str) -> Dict[str, Any]:
        """
        Спросить человека как поступить с целью
        
        Returns:
            Результат ask_user с вопросом
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if not goal:
                raise ValueError(f"Goal {goal_id} not found")
        
        # Спрашиваем человека
        return await ask_user(
            subject_type="goal",
            subject_id=goal_id,
            question=f"Как поступить с целью '{goal.title}'?",
            options=[
                "Разбить на 2-3 шага",
                "Оставить как есть (атомарная)",
                "Отложить",
                "Удалить"
            ],
            timeout_action="continue_with_default",
            default_answer="Оставить как есть"
        )
    
    async def create_subgoals_from_answer(
        self, 
        goal_id: str, 
        answer: str, 
        free_text: str = None
    ) -> List[Dict[str, Any]]:
        """
        Создать подцели из ответа человека
        
        Если answer == "Разбить на 2-3 шага" и есть free_text:
        - Парсим free_text по строкам
        - Каждую строку = подцель
        - Создаём подцели в БД
        
        Returns:
            Список созданных подцелей
        """
        if answer != "Разбить на 2-3 шага":
            return []
        
        if not free_text:
            return []
        
        # Парсим свободный текст по строкам
        lines = [line.strip() for line in free_text.split('\n') if line.strip()]
        
        # Берём максимум 3 первые строки
        subgoal_titles = lines[:3]
        
        if not subgoal_titles:
            return []
        
        # Создаём подцели
        async with AsyncSessionLocal() as db:
            # Загружаем родительскую цель
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            parent_goal = result.scalar_one_or_none()
            
            if not parent_goal:
                raise ValueError(f"Parent goal {goal_id} not found")
            
            created_subgoals = []
            
            for i, title in enumerate(subgoal_titles):
                subgoal = Goal(
                    parent_id=parent_goal.id,
                    title=title,
                    description=f"Подцель {i+1} из: {parent_goal.title}",
                    status="pending",
                    progress=0.0,
                    goal_type="achievable",
                    depth_level=parent_goal.depth_level + 1,
                    is_atomic=True,  # Подцели считаем атомарными
                    created_at=datetime.utcnow()
                )
                
                db.add(subgoal)
                created_subgoals.append({
                    "id": str(subgoal.id),
                    "title": subgoal.title,
                    "order": i + 1
                })
            
            await db.commit()
            
            return created_subgoals
    
    async def run(self, goal_id: str) -> Dict[str, Any]:
        """
        Запустить flow декомпозиции
        
        1. Проверить нужна ли декомпозиция
        2. Если да - спросить человека
        3. Дождаться ответа
        4. Создать подцели из ответа
        
        Returns:
            {
                "status": "decomposed" | "skipped" | "pending",
                "question_id": "...",  # если ожидает ответ
                "subgoals": [...]  # если созданы
            }
        """
        # Шаг 1: Проверяем нужна ли декомпозиция
        needs_decomp = await self.check_needs_decomposition(goal_id)
        
        if not needs_decomp:
            return {
                "status": "skipped",
                "reason": "Goal is atomic or already has subgoals"
            }
        
        # Шаг 2: Спрашиваем человека
        question_result = await self.ask_how_to_proceed(goal_id)
        
        if question_result["status"] == "pending":
            return {
                "status": "pending",
                "question_id": question_result["question_id"],
                "message": "Waiting for user response"
            }
        
        # Шаг 3-4: Если уже отвечен - создаём подцели
        # (обычно это будет вызвано отдельно после получения ответа)
        return {
            "status": "question_sent",
            "question_id": question_result["question_id"]
        }


# Singleton
goal_decomposition_flow = GoalDecompositionFlow()
