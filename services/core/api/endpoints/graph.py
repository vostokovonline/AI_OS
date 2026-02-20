"""
Graph API Endpoints Module
For Dashboard v2 ReactFlow visualization
"""
from fastapi import APIRouter, Query
from typing import Optional
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal, SkillManifestDB

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/")
async def get_graph(
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    root_id: Optional[str] = Query(None, description="Root node ID"),
    depth: int = Query(2, ge=1, le=5, description="Graph depth")
):
    """
    Получает граф целей, агентов, навыков и артефактов
    Для Dashboard v2 ReactFlow визуализации
    """
    async with AsyncSessionLocal() as db:
        nodes = []
        edges = []
        
        # Добавляем цели с пагинацией
        stmt = select(Goal).order_by(Goal.created_at.desc()).limit(500)
        result = await db.execute(stmt)
        goals = result.scalars().all()
        
        for g in goals:
            nodes.append({
                "id": str(g.id),
                "type": "goal",
                "data": {
                    "label": g.title,
                    "status": g.status,
                    "progress": g.progress,
                    "goal_type": g.goal_type,
                    "is_atomic": g.is_atomic,
                    "depth_level": g.depth_level
                }
            })
            
            # Добавляем связь с родителем
            if g.parent_id:
                edges.append({
                    "id": f"{g.parent_id}-{g.id}",
                    "source": str(g.parent_id),
                    "target": str(g.id),
                    "type": "dependency"
                })
        
        # Добавляем навыки
        stmt = select(SkillManifestDB).where(SkillManifestDB.is_active == True).limit(100)
        result = await db.execute(stmt)
        skills = result.scalars().all()
        
        for s in skills:
            nodes.append({
                "id": f"skill-{s.name}",
                "type": "skill",
                "data": {
                    "label": s.name,
                    "category": s.category,
                    "version": s.version,
                    "description": s.description
                }
            })
        
        return {
            "status": "ok",
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "goals": len([n for n in nodes if n["type"] == "goal"]),
                "skills": len([n for n in nodes if n["type"] == "skill"])
            }
        }


@router.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Получает детальную информацию об узле"""
    from models import Artifact
    import uuid
    
    async with AsyncSessionLocal() as db:
        # Проверяем goal
        try:
            stmt = select(Goal).where(Goal.id == uuid.UUID(node_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if goal:
                return {
                    "status": "ok",
                    "node": {
                        "id": str(goal.id),
                        "type": "goal",
                        "data": {
                            "title": goal.title,
                            "description": goal.description,
                            "status": goal.status,
                            "progress": goal.progress,
                            "goal_type": goal.goal_type,
                            "is_atomic": goal.is_atomic,
                            "depth_level": goal.depth_level,
                            "domains": goal.domains,
                            "created_at": goal.created_at.isoformat() if goal.created_at else None,
                            "updated_at": goal.updated_at.isoformat() if goal.updated_at else None
                        }
                    }
                }
        except ValueError:
            pass
        
        # Проверяем artifact
        try:
            stmt = select(Artifact).where(Artifact.id == uuid.UUID(node_id))
            result = await db.execute(stmt)
            artifact = result.scalar_one_or_none()
            
            if artifact:
                return {
                    "status": "ok",
                    "node": {
                        "id": str(artifact.id),
                        "type": "artifact",
                        "data": {
                            "type": artifact.type,
                            "goal_id": str(artifact.goal_id),
                            "skill_name": artifact.skill_name,
                            "verification_status": artifact.verification_status,
                            "reusable": artifact.reusable
                        }
                    }
                }
        except ValueError:
            pass
        
        return {"status": "error", "message": "Node not found"}


@router.get("/nodes/{node_id}/inspector")
async def get_node_inspector(node_id: str):
    """Получает контекст для inspector panel"""
    import uuid
    
    async with AsyncSessionLocal() as db:
        try:
            stmt = select(Goal).where(Goal.id == uuid.UUID(node_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if goal:
                # Получаем артефакты цели
                from models import Artifact
                artifact_stmt = select(Artifact).where(Artifact.goal_id == goal.id)
                artifact_result = await db.execute(artifact_stmt)
                artifacts = artifact_result.scalars().all()
                
                # Получаем подцели
                children_stmt = select(Goal).where(Goal.parent_id == goal.id)
                children_result = await db.execute(children_stmt)
                children = children_result.scalars().all()
                
                return {
                    "status": "ok",
                    "context": {
                        "node_id": str(goal.id),
                        "node_type": "goal",
                        "title": goal.title,
                        "description": goal.description,
                        "status": goal.status,
                        "progress": goal.progress,
                        "artifacts": [
                            {
                                "id": str(a.id),
                                "type": a.type,
                                "status": a.verification_status
                            }
                            for a in artifacts
                        ],
                        "sub_goals": len(children),
                        "domains": goal.domains or [],
                        "metadata": {
                            "created_at": goal.created_at.isoformat() if goal.created_at else None,
                            "updated_at": goal.updated_at.isoformat() if goal.updated_at else None,
                            "depth_level": goal.depth_level
                        }
                    }
                }
        except ValueError:
            pass
        
        return {"status": "error", "message": "Node not found"}
