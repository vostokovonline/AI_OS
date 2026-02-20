"""
Goals API Endpoints Module
Refactored from main.py for better modularity
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, text
from sqlalchemy.dialects.postgresql import insert
from typing import Optional, List
import uuid
from database import AsyncSessionLocal
from models import Goal, Artifact

router = APIRouter(prefix="/goals", tags=["goals"])


@router.post("/create")
async def create_goal_endpoint(req: dict):
    """Создает новую цель"""
    from goal_executor import goal_executor
    
    try:
        goal_id = await goal_executor.create_goal(
            title=req.get("title"),
            description=req.get("description", ""),
            goal_type=req.get("goal_type", "bounded"),
            is_atomic=req.get("is_atomic", False),
            depth_level=req.get("depth_level", 0),
            parent_id=req.get("parent_id"),
            user_id=req.get("user_id")
        )
        
        if req.get("auto_execute", True):
            from tasks import execute_goal_task
            execute_goal_task.delay(goal_id, None)
            return {"status": "created_and_started", "goal_id": goal_id}
        
        return {"status": "created", "goal_id": goal_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/execute")
async def execute_goal_endpoint(req: dict):
    """Выполняет существующую цель через Orchestrator (V1)"""
    from goal_executor import goal_executor
    
    goal_id = req.get("goal_id")
    session_id = req.get("session_id")
    
    # API doesn't make architectural decisions
    # Orchestrator (V1) handles atomic vs complex internally
    result = await goal_executor.execute_goal(goal_id, session_id)
    
    return result


@router.get("/list")
async def get_goals_list(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    goal_type: Optional[str] = Query(None, description="Filter by type")
):
    """
    Получает список целей с пагинацией
    
    Args:
        page: Номер страницы (начиная с 1)
        page_size: Количество элементов на странице (1-100)
        status: Фильтр по статусу (опционально)
        goal_type: Фильтр по типу цели (опционально)
    """
    async with AsyncSessionLocal() as db:
        # Base query
        query = select(Goal).order_by(Goal.created_at.desc())
        
        # Apply filters
        if status:
            query = query.where(Goal.status == status)
        if goal_type:
            query = query.where(Goal.goal_type == goal_type)
        
        # Get total count
        count_query = select(func.count(Goal.id))
        if status:
            count_query = count_query.where(Goal.status == status)
        if goal_type:
            count_query = count_query.where(Goal.goal_type == goal_type)
        
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        result = await db.execute(query)
        goals = result.scalars().all()
        
        goals_list = []
        for g in goals:
            goals_list.append({
                "id": str(g.id),
                "parent_id": str(g.parent_id) if g.parent_id else None,
                "title": g.title,
                "description": g.description,
                "status": g.status,
                "progress": g.progress,
                "goal_type": g.goal_type,
                "depth_level": g.depth_level,
                "is_atomic": g.is_atomic,
                "created_at": g.created_at.isoformat() if g.created_at else None,
                "updated_at": g.updated_at.isoformat() if g.updated_at else None,
            })
        
        return {
            "status": "ok",
            "goals": goals_list,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }


@router.get("/stats")
async def get_goals_stats():
    """Получает статистику по целям"""
    async with AsyncSessionLocal() as db:
        # Общая статистика
        total = await db.execute(select(func.count(Goal.id)))
        total = total.scalar()
        
        # По типам
        by_type = await db.execute(
            select(Goal.goal_type, func.count(Goal.id))
            .group_by(Goal.goal_type)
        )
        by_type = {row[0]: row[1] for row in by_type}
        
        # По статусам
        by_status = await db.execute(
            select(Goal.status, func.count(Goal.id))
            .group_by(Goal.status)
        )
        by_status = {row[0]: row[1] for row in by_status}
        
        # По уровням
        by_depth = await db.execute(
            select(Goal.depth_level, func.count(Goal.id))
            .group_by(Goal.depth_level)
        )
        by_depth = {row[0]: row[1] for row in by_depth}
        
        return {
            "status": "ok",
            "total": total,
            "by_type": by_type,
            "by_status": by_status,
            "by_depth": by_depth
        }


@router.get("/{goal_id}/tree")
async def get_goal_tree(goal_id: str):
    """Получает дерево целей (цель + все подцели) с оптимизированной загрузкой"""
    from sqlalchemy.orm import selectinload
    
    async with AsyncSessionLocal() as db:
        # Оптимизированный запрос с eager loading
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()
        
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        # Загружаем все подцели одним запросом
        async def build_tree_optimized(goal_id: str, db):
            """Строит дерево без N+1 проблемы"""
            # Получаем все цели, связанные с данным деревом
            all_goals_query = """
                WITH RECURSIVE goal_tree AS (
                    SELECT id, parent_id, title, description, status, progress, 
                           goal_type, depth_level, is_atomic, domains
                    FROM goals
                    WHERE id = :goal_id
                    UNION ALL
                    SELECT g.id, g.parent_id, g.title, g.description, g.status, 
                           g.progress, g.goal_type, g.depth_level, g.is_atomic, g.domains
                    FROM goals g
                    JOIN goal_tree gt ON g.parent_id = gt.id
                )
                SELECT * FROM goal_tree
            """
            result = await db.execute(text(all_goals_query), {"goal_id": goal_id})
            rows = result.fetchall()
            
            # Строим словарь id -> goal
            goals_map = {}
            for row in rows:
                goals_map[str(row.id)] = {
                    "id": str(row.id),
                    "parent_id": str(row.parent_id) if row.parent_id else None,
                    "title": row.title,
                    "description": row.description,
                    "status": row.status,
                    "progress": row.progress,
                    "goal_type": row.goal_type,
                    "depth_level": row.depth_level,
                    "is_atomic": row.is_atomic,
                    "domains": row.domains,
                    "children": []
                }
            
            # Строим дерево
            root = None
            for goal_id_key, goal_data in goals_map.items():
                parent_id = goal_data["parent_id"]
                if parent_id and parent_id in goals_map:
                    goals_map[parent_id]["children"].append(goal_data)
                else:
                    root = goal_data
            
            return root
        
        tree = await build_tree_optimized(str(goal.id), db)
        
        return {
            "status": "ok",
            "tree": tree
        }


@router.post("/{goal_id}/decompose")
async def decompose_goal(goal_id: str, max_depth: int = 3):
    """Декомпозирует цель на подцели"""
    from goal_decomposer import goal_decomposer
    from policies.legacy_policy import legacy_policy
    
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Goal).where(Goal.id == uuid.UUID(goal_id)))
        goal = result.scalar_one_or_none()
        
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
    
    # Validate against Legacy Policy
    validation = await legacy_policy.validate_goal_decomposition(goal_id)
    
    if not validation["valid"]:
        return {
            "status": "error",
            "message": "Legacy Policy violation",
            "reason": validation["reason"]
        }
    
    # Декомпозируем
    subgoals = await goal_decomposer.decompose_goal(goal_id, max_depth)
    
    return {
        "status": "ok",
        "goal_id": goal_id,
        "subgoals_created": len(subgoals),
        "subgoals": subgoals
    }


@router.get("/{goal_id}/artifacts")
async def get_goal_artifacts(
    goal_id: str,
    verification_status: Optional[str] = None,
    include_descendants: bool = True
):
    """Возвращает артефакты цели"""
    from artifact_registry import artifact_registry
    
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()
        
        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")
        
        artifacts = []
        
        if goal.is_atomic:
            artifacts = await artifact_registry.list_by_goal(goal_id, verification_status)
        elif include_descendants:
            # Получаем все descendant atomic goals одним запросом
            descendant_query = """
                WITH RECURSIVE goal_tree AS (
                    SELECT id, is_atomic FROM goals WHERE id = :goal_id
                    UNION ALL
                    SELECT g.id, g.is_atomic
                    FROM goals g
                    JOIN goal_tree gt ON g.parent_id = gt.id
                )
                SELECT id FROM goal_tree WHERE is_atomic = true AND id != :goal_id
            """
            result = await db.execute(text(descendant_query), {"goal_id": goal_id})
            descendant_ids = [str(row[0]) for row in result.fetchall()]
            
            for desc_id in descendant_ids:
                desc_artifacts = await artifact_registry.list_by_goal(desc_id, verification_status)
                artifacts.extend(desc_artifacts)
        
        return {
            "status": "ok",
            "goal_id": goal_id,
            "is_atomic": goal.is_atomic,
            "count": len(artifacts),
            "artifacts": artifacts
        }
