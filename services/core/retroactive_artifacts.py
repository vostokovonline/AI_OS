"""
Retroactive Artifact Generation

Создаёт артефакты для выполненных goals, где они отсутствуют.

Проблема:
- Выполненные goals без artifacts → status должен быть "incomplete"
- Но иногдаgoals выполнялись до внедрения Artifact Layer

Решение:
1. Найти выполненные goals без artifacts
2. Сгенерировать artifacts на основе execution_trace
3. Сохранить artifacts в реестр
4. Обновить verification status

NS1/NS2: "Memory vs Logs" - artifacts persist, logs don't
"""

from typing import List, Dict, Optional
from sqlalchemy import select, and_
from datetime import datetime
import uuid
import json

from database import AsyncSessionLocal
from models import Goal, Artifact
from artifact_registry import ArtifactRegistry
from pydantic import BaseModel, Field


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class RetroactiveArtifactRequest(BaseModel):
    """Запрос на создание артефакта постфактум"""
    goal_id: str
    artifact_type: str  # FILE, KNOWLEDGE, REPORT, EXECUTION_LOG
    content: str
    content_location: Optional[str] = None
    skill_name: str = "retroactive_generator"


# =============================================================================
# RETROACTIVE ARTIFACT GENERATOR
# =============================================================================

class RetroactiveArtifactGenerator:
    """
    Генерирует артефакты для выполненных goals.

    Использует:
    - Execution trace для восстановления действий
    - Goal title/description для контекста
    - LLM для генерации контента (если нужно)
    """

    @staticmethod
    async def find_completed_goals_without_artifacts(
        limit: int = 100
    ) -> List[Dict]:
        """
        Найти выполненные goals без artifacts.

        Args:
            limit: Макс. количество goals

        Returns:
            List[Dict] с информацией о goals
        """
        async with AsyncSessionLocal() as db:
            # Найти goals
            stmt = select(Goal).where(
                and_(
                    Goal.status == "completed",
                    Goal.is_atomic == True  # Только atomic goals требуют artifacts
                )
            ).order_by(Goal.created_at.desc()).limit(limit)

            result = await db.execute(stmt)
            goals = result.scalars().all()

            # Проверить наличие artifacts для каждого
            goals_without_artifacts = []

            for goal in goals:
                # Проверить artifacts
                stmt_artifacts = select(Artifact).where(
                    Artifact.goal_id == goal.id
                )
                result_artifacts = await db.execute(stmt_artifacts)
                artifacts = result_artifacts.scalars().all()

                if len(artifacts) == 0:
                    # Goals без artifacts
                    goals_without_artifacts.append({
                        "id": str(goal.id),
                        "title": goal.title,
                        "description": goal.description,
                        "goal_type": goal.goal_type,
                        "created_at": goal.created_at.isoformat() if goal.created_at else None,
                        "updated_at": goal.updated_at.isoformat() if goal.updated_at else None,
                        "execution_trace": goal.execution_trace,
                        "parent_id": str(goal.parent_id) if goal.parent_id else None,
                        "status": goal.status,
                        "is_atomic": goal.is_atomic
                    })

            return goals_without_artifacts

    @staticmethod
    async def generate_artifact_for_goal(
        goal_id: str,
        artifact_type: str = "REPORT",
        content: str = None,
        skill_name: str = "retroactive_generator"
    ) -> Artifact:
        """
        Сгенерировать artifact для goal.

        Args:
            goal_id: UUID goal
            artifact_type: Тип артефакта
            content: Контент (если None - сгенерировать из trace)
            skill_name: Навык-генератор

        Returns:
            Artifact (сохранённый в БД)
        """
        async with AsyncSessionLocal() as db:
            # Получить goal
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                raise ValueError(f"Goal {goal_id} not found")

            # Сгенерировать контент если не предоставлен
            if content is None:
                content = await RetroactiveArtifactGenerator._generate_content_from_goal(
                    goal, artifact_type
                )

            # Создать artifact через registry
            registry = ArtifactRegistry()

            import os
            artifacts_path = os.getenv("ARTIFACTS_PATH", "/data/artifacts")

            artifact_data = {
                "goal_id": str(goal.id),
                "artifact_type": artifact_type,
                "content_kind": "text",
                "content_location": f"{artifacts_path}/{goal.title[:50]}.md",
                "skill_name": skill_name,
                "agent_role": "retroactive_generator",
                "content": content,
                "tags": ["retroactive", "auto_generated"],
                "reusable": False  # Retroactive artifacts не reusable
            }

            artifact = await registry.register_artifact(artifact_data)

            # Вернуть artifact
            return artifact

    @staticmethod
    async def _generate_content_from_goal(goal: Goal, artifact_type: str) -> str:
        """
        Сгенерировать контент artifact'а из goal.

        Args:
            goal: Goal модель
            artifact_type: Тип artifact'а

        Returns:
            Сгенерированный контент
        """
        # Базовая информация
        content_parts = [
            f"# Artifact for Goal: {goal.title}",
            "",
            f"**Goal ID**: {goal.id}",
            f"**Type**: {artifact_type}",
            f"**Status**: {goal.status}",
            f"**Created at**: {goal.created_at}",
            "",
            "## Description",
            goal.description or "No description provided",
            "",
            "## Execution Summary",
        ]

        # Добавить execution_trace если есть
        if goal.execution_trace:
            content_parts.extend([
                "### Execution Trace:",
                "```json",
                json.dumps(goal.execution_trace, indent=2, ensure_ascii=False),
                "```",
                ""
            ])

        # Специфика для разных типов
        if artifact_type == "REPORT":
            content_parts.extend([
                "## Report Summary",
                "",
                f"Goal '{goal.title}' was successfully completed.",
                "",
                "### Key Results:",
                "- Status: Completed ✅",
                f"- Goal Type: {goal.goal_type}",
                f"- Atomic: {goal.is_atomic}",
                "",
                "### Notes:",
                "This artifact was retroactively generated after goal completion.",
                "Original execution details may be limited."
            ])

        elif artifact_type == "KNOWLEDGE":
            content_parts.extend([
                "## Knowledge Extracted",
                "",
                f"From goal: {goal.title}",
                "",
                "### Key Learnings:",
                "- Goal was successfully executed",
                "- Execution trace available for analysis",
                "",
                "### Related Concepts:",
                "- Goal execution",
                "- Task completion",
                "- Process tracking"
            ])

        elif artifact_type == "EXECUTION_LOG":
            content_parts.extend([
                "## Execution Log",
                "",
                f"Goal: {goal.title}",
                f"Started: {goal.created_at}",
                f"Completed: {goal.updated_at or 'Unknown'}",
                "",
                "### Log Entries:",
                "1. Goal created",
                "2. Goal executed",
                "3. Goal completed",
                "",
                "### Duration:",
                f"Start to finish: {(goal.updated_at - goal.created_at).total_seconds() if goal.updated_at else 'N/A'} seconds"
            ])

        # FILE artifact - создать файл
        elif artifact_type == "FILE":
            file_content = f"# {goal.title}\n\n{goal.description}\n\n" \
                           f"**Completed at**: {datetime.utcnow().isoformat()}\n\n" \
                           f"**Goal ID**: {goal.id}\n"

            # Записать в файл
            import os
            artifacts_path = os.getenv("ARTIFACTS_PATH", "/data/artifacts")
            os.makedirs(artifacts_path, exist_ok=True)
            file_path = f"{artifacts_path}/{goal.title[:50]}.md"

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(file_content)

            return file_path

        return "\n".join(content_parts)

    @staticmethod
    async def batch_generate_artifacts(
        goals: List[Dict] = None,
        artifact_type: str = "REPORT"
    ) -> List[Artifact]:
        """
        Массово генерировать artifacts для goals.

        Args:
            goals: Список goals (если None - найти автоматически)
            artifact_type: Тип artifact'ов

        Returns:
            List[созданных Artifact]
        """
        # Если goals не предоставлены - найти
        if goals is None:
            goals = await RetroactiveArtifactGenerator.find_completed_goals_without_artifacts()

        artifacts_created = []

        for goal_info in goals:
            try:
                artifact = await RetroactiveArtifactGenerator.generate_artifact_for_goal(
                    goal_id=goal_info["id"],
                    artifact_type=artifact_type,
                    skill_name="retroactive_batch_generator"
                )

                artifacts_created.append(artifact)
                logger.info(f"✅ Artifact created for goal: {goal_info['title']}")

            except Exception as e:
                logger.info(f"❌ Error creating artifact for goal {goal_info['title']}: {e}")
                continue

        return artifacts_created


# =============================================================================
# API ENDPOINT HELPERS
# =============================================================================

async def fix_goal_without_artifacts(goal_id: str) -> Dict:
    """
    Исправить goal без artifacts - создать artifact.

    Args:
        goal_id: UUID goal

    Returns:
        Результат операции
    """
    try:
        artifact = await RetroactiveArtifactGenerator.generate_artifact_for_goal(
            goal_id=goal_id,
            artifact_type="REPORT"
        )

        return {
            "status": "ok",
            "message": "Artifact created retroactively",
            "artifact_id": str(artifact.id),
            "verification_status": artifact.verification_status
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


async def batch_fix_all_goals() -> Dict:
    """
    Массово исправить все goals без artifacts.

    Returns:
        Результат операции
    """
    goals = await RetroactiveArtifactGenerator.find_completed_goals_without_artifacts()

    if not goals:
        return {
            "status": "ok",
            "message": "No completed goals without artifacts found",
            "fixed_count": 0
        }

    artifacts = await RetroactiveArtifactGenerator.batch_generate_artifacts(
        goals=goals,
        artifact_type="REPORT"
    )

    return {
        "status": "ok",
        "message": f"Fixed {len(artifacts)} goals",
        "fixed_count": len(artifacts),
        "artifacts": [str(a.id) for a in artifacts]
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """
    Основная функция для запуска генерации.
    """
    logger.info("🔍 Searching for completed goals without artifacts...")

    goals = await RetroactiveArtifactGenerator.find_completed_goals_without_artifacts()

    if not goals:
        logger.info("✅ No completed goals without artifacts found!")
        return

    logger.info(f"Found {len(goals)} completed goals without artifacts:")
    for g in goals[:5]:
        logger.info(f"  - {g['title']} ({g['id']})")

    if len(goals) > 5:
        logger.info(f"  ... and {len(goals) - 5} more")

    logger.info(f"\n🔧 Generating artifacts...")

    result = await batch_fix_all_goals()

    logger.info(f"\n✅ Done! Fixed {result['fixed_count']} goals")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
