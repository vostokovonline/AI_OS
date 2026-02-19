"""
ARTIFACT REGISTRY - v1.1
Единый реестр всех артефактов системы

ARCHITECTURE v3.0:
- Uses UnitOfWork pattern for transaction management
- All transactions opened by caller, not internally

Назначение:
- Регистрация артефактов
- Связь целей → результатов
- Основа для dashboard и memory
"""
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from uuid import UUID
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Artifact, Goal
from artifact_verifier import artifact_verifier, VerificationResult
from infrastructure.uow import UnitOfWork


class ArtifactRegistry:
    """
    Реестр артефактов

    Key principle: Every atomic goal (L3) MUST produce at least 1 artifact
    """

    ARTIFACT_TYPES = ["FILE", "KNOWLEDGE", "DATASET", "REPORT", "LINK", "EXECUTION_LOG"]

    CONTENT_KINDS = ["file", "db", "vector", "external"]

    VERIFICATION_STATUSES = ["pending", "passed", "failed", "partial"]

    async def register(
        self,
        goal_id: str,
        artifact_type: str,
        content_kind: str,
        content_location: str,
        skill_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        reusable: bool = True,
        auto_verify: bool = True
    ) -> Dict:
        """
        Регистрирует новый артефакт (legacy - creates own transaction).

        DEPRECATED: Use register_with_uow() instead for proper transaction management.

        Args:
            goal_id: ID цели которая создала артефакт
            artifact_type: Тип артефакта (FILE, KNOWLEDGE, etc.)
            content_kind: Где хранится контент (file, db, vector, external)
            content_location: Путь, URL, ID
            skill_name: Навык который создал
            agent_role: Агент который создал
            domains: Домены
            tags: Теги
            language: Язык программирования (для кода)
            reusable: Переиспользуемый ли артефакт
            auto_verify: Автоматически верифицировать после регистрации

        Returns:
            {
                "artifact_id": "...",
                "verification_status": "pending|passed|failed",
                "verification_results": [...]
            }
        """
        from infrastructure.uow import create_uow_provider

        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
            return await self.register_with_uow(
                uow=uow,
                goal_id=goal_id,
                artifact_type=artifact_type,
                content_kind=content_kind,
                content_location=content_location,
                skill_name=skill_name,
                agent_role=agent_role,
                domains=domains,
                tags=tags,
                language=language,
                reusable=reusable,
                auto_verify=auto_verify
            )

    async def register_with_uow(
        self,
        uow: UnitOfWork,
        goal_id: str,
        artifact_type: str,
        content_kind: str,
        content_location: str,
        skill_name: Optional[str] = None,
        agent_role: Optional[str] = None,
        domains: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        language: Optional[str] = None,
        reusable: bool = True,
        auto_verify: bool = True
    ) -> Dict:
        """
        Регистрирует новый артефакт WITHIN существующей транзакции.

        ARCHITECTURE v3.0: Transaction managed by caller via UnitOfWork.

        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели которая создала артефакт
            artifact_type: Тип артефакта (FILE, KNOWLEDGE, etc.)
            content_kind: Где хранится контент (file, db, vector, external)
            content_location: Путь, URL, ID
            skill_name: Навык который создал
            agent_role: Агент который создал
            domains: Домены
            tags: Теги
            language: Язык программирования (для кода)
            reusable: Переиспользуемый ли артефакт
            auto_verify: Автоматически верифицировать после регистрации

        Returns:
            {
                "artifact_id": "...",
                "verification_status": "pending|passed|failed",
                "verification_results": [...]
            }
        """
        if artifact_type not in self.ARTIFACT_TYPES:
            raise ValueError(f"Invalid artifact_type: {artifact_type}")

        if content_kind not in self.CONTENT_KINDS:
            raise ValueError(f"Invalid content_kind: {content_kind}")

        artifact = Artifact(
            goal_id=UUID(goal_id),
            type=artifact_type,
            content_kind=content_kind,
            content_location=content_location,
            skill_name=skill_name,
            agent_role=agent_role,
            domains=domains or [],
            tags=tags or [],
            language=language,
            reusable=reusable,
            verification_status="pending"
        )

        uow.session.add(artifact)
        await uow.session.flush()
        await uow.session.refresh(artifact)

        verification_results = []
        verification_status = "pending"

        if auto_verify:
            try:
                verification_results = await self.verify_artifact_with_uow(
                    uow, str(artifact.id)
                )
                artifact.verification_status = verification_results["status"]
                artifact.verification_results = verification_results["results"]
                await uow.session.flush()
            except Exception as e:
                print(f"❌ Verification error: {e}")
                pass

        description = None
        if artifact.content_kind == "file" and artifact.content_location:
            try:
                if os.path.exists(artifact.content_location):
                    with open(artifact.content_location, 'r', encoding='utf-8') as f:
                        first_line = f.readline(200).strip()
                        if first_line.startswith('#'):
                            parts = first_line.split(maxsplit=1)
                            description = parts[1].strip() if len(parts) > 1 else first_line
                        else:
                            description = first_line
            except Exception as e:
                print(f"⚠️ Failed to generate description: {e}")
                description = None

        return {
            "artifact_id": str(artifact.id),
            "type": artifact.type,
            "content_kind": artifact.content_kind,
            "content_location": artifact.content_location,
            "description": description,
            "verification_status": artifact.verification_status,
            "verification_results": artifact.verification_results,
            "created_at": artifact.created_at.isoformat()
        }

    async def verify_artifact(self, artifact_id: str) -> Dict:
        """
        Верифицирует артефакт (legacy - creates own transaction).

        DEPRECATED: Use verify_artifact_with_uow() instead.

        Args:
            artifact_id: ID артефакта

        Returns:
            {
                "status": "passed|failed|partial",
                "results": [{"name": "...", "passed": true, "details": "..."}]
            }
        """
        from infrastructure.uow import create_uow_provider

        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
            return await self.verify_artifact_with_uow(uow, artifact_id)

    async def verify_artifact_with_uow(self, uow: UnitOfWork, artifact_id: str) -> Dict:
        """
        Верифицирует артефакт WITHIN существующей транзакции.

        ARCHITECTURE v3.0: Transaction managed by caller via UnitOfWork.

        Args:
            uow: UnitOfWork с активной транзакцией
            artifact_id: ID артефакта

        Returns:
            {
                "status": "passed|failed|partial",
                "results": [{"name": "...", "passed": true, "details": "..."}]
            }
        """
        stmt = select(Artifact).where(Artifact.id == UUID(artifact_id))
        result = await uow.session.execute(stmt)
        artifact = result.scalar_one_or_none()

        if not artifact:
            return {"status": "failed", "error": "Artifact not found"}

        artifact_data = {
            "type": artifact.type,
            "content_kind": artifact.content_kind,
            "content_location": artifact.content_location
        }

        verification_results = artifact_verifier.verify(artifact_data)
        results_dict = [r.to_dict() for r in verification_results]
        overall_status = artifact_verifier.get_overall_status(verification_results)

        artifact.verification_status = overall_status
        artifact.verification_results = results_dict
        await uow.session.flush()

        return {
            "status": overall_status,
            "results": results_dict
        }

    async def list_by_goal(self, goal_id: str, verification_status: Optional[str] = None) -> List[Dict]:
        """
        Возвращает артефакты цели

        Args:
            goal_id: ID цели
            verification_status: Фильтр по статусу верификации (опционально)

        Returns:
            Список артефактов
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Artifact).where(Artifact.goal_id == uuid.UUID(goal_id))

            if verification_status:
                stmt = stmt.where(Artifact.verification_status == verification_status)

            stmt = stmt.order_by(Artifact.created_at.desc())

            result = await db.execute(stmt)
            artifacts = result.scalars().all()

            return [
                {
                    "id": str(a.id),
                    "type": a.type,
                    "content_kind": a.content_kind,
                    "content_location": a.content_location,
                    "skill_name": a.skill_name,
                    "agent_role": a.agent_role,
                    "domains": a.domains,
                    "tags": a.tags,
                    "language": a.language,
                    "verification_status": a.verification_status,
                    "verification_results": a.verification_results,
                    "reusable": a.reusable,
                    "created_at": a.created_at.isoformat()
                }
                for a in artifacts
            ]

    async def get(self, artifact_id: str) -> Optional[Dict]:
        """
        Возвращает артефакт по ID

        Args:
            artifact_id: ID артефакта

        Returns:
            Артефакт или None
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Artifact).where(Artifact.id == uuid.UUID(artifact_id))
            result = await db.execute(stmt)
            artifact = result.scalar_one_or_none()

            if not artifact:
                return None

            return {
                "id": str(artifact.id),
                "type": artifact.type,
                "goal_id": str(artifact.goal_id),
                "content_kind": artifact.content_kind,
                "content_location": artifact.content_location,
                "skill_name": artifact.skill_name,
                "agent_role": artifact.agent_role,
                "domains": artifact.domains,
                "tags": artifact.tags,
                "language": artifact.language,
                "verification_status": artifact.verification_status,
                "verification_results": artifact.verification_results,
                "reusable": artifact.reusable,
                "created_at": artifact.created_at.isoformat(),
                "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None
            }

    async def check_goal_artifacts(self, goal_id: str) -> Dict:
        """
        Проверяет наличие и статус артефактов цели

        Для atomic goals (L3): ДОЛЖЕН быть хотя бы 1 passed artifact

        Returns:
            {
                "has_artifacts": true/false,
                "total_count": 5,
                "passed_count": 3,
                "failed_count": 1,
                "pending_count": 1,
                "goal_complete": true/false  # Для L3: true только если есть passed artifacts
            }
        """
        async with AsyncSessionLocal() as db:
            # Получаем цель
            stmt_goal = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result_goal = await db.execute(stmt_goal)
            goal = result_goal.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

            # Получаем артефакты
            stmt_art = select(Artifact).where(Artifact.goal_id == uuid.UUID(goal_id))
            result_art = await db.execute(stmt_art)
            artifacts = result_art.scalars().all()

            total_count = len(artifacts)
            passed_count = sum(1 for a in artifacts if a.verification_status == "passed")
            failed_count = sum(1 for a in artifacts if a.verification_status == "failed")
            pending_count = sum(1 for a in artifacts if a.verification_status == "pending")

            has_artifacts = total_count > 0

            # Для atomic goals (L3): без passed артефактов цель не выполнена
            goal_complete = True
            if goal.is_atomic:
                goal_complete = passed_count > 0

            return {
                "has_artifacts": has_artifacts,
                "total_count": total_count,
                "passed_count": passed_count,
                "failed_count": failed_count,
                "pending_count": pending_count,
                "goal_complete": goal_complete,
                "is_atomic": goal.is_atomic
            }


# Глобальный экземпляр
artifact_registry = ArtifactRegistry()
