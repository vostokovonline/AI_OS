"""
goal_decomposition_flow - упрощённый сценарий через ask_user_simple
"""

from typing import List, Dict, Any
from datetime import datetime
import uuid
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal
from canonical_skills.ask_user_simple import ask_user_simple


class GoalDecompositionSimpleFlow:
    
    async def check_needs_decomposition(self, goal_id: str) -> bool:
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if not goal or goal.is_atomic:
                return False
            
            children_stmt = select(Goal).where(Goal.parent_id == goal.id)
            children_result = await db.execute(children_stmt)
            children = children_result.scalars().all()
            
            return len(children) == 0
    
    async def ask_how_to_proceed(self, goal_id: str) -> Dict[str, Any]:
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if not goal:
                raise ValueError(f"Goal {goal_id} not found")
        
        return await ask_user_simple(
            subject_type="goal",
            subject_id=goal_id,
            question=f"Как поступить с целью '{goal.title}'?",
            options=[
                "Разбить на 2-3 шага",
                "Оставить как есть (атомарная)",
                "Отложить",
                "Удалить"
            ]
        )
    
    async def create_subgoals_from_text(
        self, 
        goal_id: str, 
        subgoal_titles: List[str]
    ) -> List[Dict[str, Any]]:
        if not subgoal_titles:
            return []
        
        subgoal_titles = subgoal_titles[:3]
        
        async with AsyncSessionLocal() as db:
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
                    description=f"Шаг {i+1} из: {parent_goal.title}",
                    status="pending",
                    progress=0.0,
                    goal_type="achievable",
                    depth_level=parent_goal.depth_level + 1,
                    is_atomic=True,
                    created_at=datetime.utcnow()
                )
                
                db.add(subgoal)
                await db.flush()  # ВАЖНО: flush чтобы получить ID
                
                created_subgoals.append({
                    "id": str(subgoal.id),
                    "title": subgoal.title,
                    "order": i + 1
                })
            
            await db.commit()
            
            return created_subgoals
    
    async def run(self, goal_id: str) -> Dict[str, Any]:
        needs_decomp = await self.check_needs_decomposition(goal_id)
        
        if not needs_decomp:
            return {
                "status": "skipped",
                "reason": "Goal is atomic or already has subgoals"
            }
        
        question = await self.ask_how_to_proceed(goal_id)
        
        return {
            "status": "question_ready",
            "question": question
        }


goal_decomposition_simple_flow = GoalDecompositionSimpleFlow()
