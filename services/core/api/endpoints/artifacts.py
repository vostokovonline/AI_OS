"""
Artifacts API Endpoints Module
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import uuid
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Artifact, Goal

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.post("/register")
async def register_artifact(artifact_data: dict):
    """Регистрирует новый артефакт"""
    from artifact_registry import artifact_registry
    
    try:
        result = await artifact_registry.register(
            goal_id=artifact_data.get("goal_id"),
            artifact_type=artifact_data.get("type"),
            content_kind=artifact_data.get("content_kind"),
            content_location=artifact_data.get("content_location"),
            skill_name=artifact_data.get("skill_name"),
            agent_role=artifact_data.get("agent_role"),
            domains=artifact_data.get("domains"),
            tags=artifact_data.get("tags"),
            language=artifact_data.get("language"),
            reusable=artifact_data.get("reusable", True),
            auto_verify=artifact_data.get("auto_verify", True)
        )
        return {"status": "ok", "artifact": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Возвращает артефакт по ID"""
    from artifact_registry import artifact_registry
    
    artifact = await artifact_registry.get(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return {"status": "ok", "artifact": artifact}


@router.get("/{artifact_id}/content")
async def get_artifact_content(artifact_id: str):
    """Возвращает содержимое артефакта (для FILE type)"""
    import os
    
    async with AsyncSessionLocal() as db:
        stmt = select(Artifact).where(Artifact.id == uuid.UUID(artifact_id))
        result = await db.execute(stmt)
        artifact = result.scalar_one_or_none()
        
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        file_path = artifact.content_location
        if not file_path or not os.path.exists(file_path):
            return {
                "status": "error",
                "message": "File not found on disk",
                "file_path": file_path
            }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {
                "status": "ok",
                "artifact_id": artifact_id,
                "file_path": file_path,
                "file_content": content,
                "file_size": len(content)
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to read file: {str(e)}"
            }


@router.get("/goals-without-artifacts")
async def get_goals_without_artifacts(limit: int = Query(100, ge=1, le=1000)):
    """Получить список выполненных goals без artifacts"""
    from retroactive_artifacts import RetroactiveArtifactGenerator
    
    try:
        goals = await RetroactiveArtifactGenerator.find_completed_goals_without_artifacts(limit)
        return {"status": "ok", "count": len(goals), "goals": goals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
