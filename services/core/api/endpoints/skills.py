"""
Skills API Endpoints Module
"""
from fastapi import APIRouter, Query
from typing import Optional
from sqlalchemy import select
from database import AsyncSessionLocal
from models import SkillManifestDB

router = APIRouter(prefix="/skills", tags=["skills"])


@router.get("/")
async def list_skills(
    category: Optional[str] = Query(None, description="Filter by category"),
    agent_role: Optional[str] = Query(None, description="Filter by agent role"),
    artifact_type: Optional[str] = Query(None, description="Filter by artifact type"),
    is_active: bool = Query(True, description="Show only active skills"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page")
):
    """Возвращает список навыков с пагинацией"""
    async with AsyncSessionLocal() as db:
        stmt = select(SkillManifestDB).where(SkillManifestDB.is_active == is_active)
        
        if category:
            stmt = stmt.where(SkillManifestDB.category == category)
        if agent_role:
            stmt = stmt.where(SkillManifestDB.agent_roles.contains([agent_role]))
        if artifact_type:
            stmt = stmt.where(
                (SkillManifestDB.outputs_artifact_type == artifact_type) |
                (SkillManifestDB.produces.contains([{"type": artifact_type}]))
            )
        
        # Get total count
        from sqlalchemy import func
        total_result = await db.execute(
            select(func.count()).select_from(stmt.subquery())
        )
        total = total_result.scalar()
        
        # Apply pagination
        stmt = stmt.order_by(SkillManifestDB.name)
        stmt = stmt.offset((page - 1) * page_size).limit(page_size)
        
        result = await db.execute(stmt)
        manifests = result.scalars().all()
        
        return {
            "status": "ok",
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "skills": [
                {
                    "id": str(m.id),
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "category": m.category,
                    "agent_roles": m.agent_roles,
                    "inputs": {
                        "schema": m.inputs_schema,
                        "required": m.inputs_required,
                        "optional": m.inputs_optional
                    },
                    "outputs": {
                        "artifact_type": m.outputs_artifact_type,
                        "schema": m.outputs_schema,
                        "reusable": m.outputs_reusable
                    },
                    "produces": m.produces,
                    "constraints": m.constraints,
                    "verification": m.verification,
                    "is_builtin": m.is_builtin
                }
                for m in manifests
            ]
        }


@router.get("/{skill_name}")
async def get_skill_manifest(skill_name: str):
    """Возвращает манифест навыка по имени"""
    async with AsyncSessionLocal() as db:
        stmt = select(SkillManifestDB).where(
            (SkillManifestDB.name == skill_name) &
            (SkillManifestDB.is_active == True)
        )
        result = await db.execute(stmt)
        manifest = result.scalar_one_or_none()
        
        if not manifest:
            return {"status": "error", "message": "Skill not found"}
        
        return {
            "status": "ok",
            "skill": {
                "name": manifest.name,
                "category": manifest.category,
                "agent_roles": manifest.agent_roles,
                "produces": manifest.produces,
                "verification": manifest.verification
            }
        }
