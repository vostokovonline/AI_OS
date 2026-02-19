"""
SKILL REGISTRY - Minimal Registry for Skills
Хранит и предоставляет доступ к зарегистрированным навыкам
"""
from typing import Dict, List, Optional
from canonical_skills.base import Skill


class SkillRegistry:
    """
    Реестр навыков

    Хранит все зарегистрированные навыки и предоставляет доступ к ним.
    """

    def __init__(self):
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """
        Регистрирует навык

        Args:
            skill: Экземпляр навыка
        """
        if not skill.id:
            raise ValueError("Skill must have an id")

        self._skills[skill.id] = skill
        print(f"✅ Skill registered: {skill.id} v{skill.version}")

    def get(self, skill_id: str) -> Optional[Skill]:
        """
        Получает навык по ID

        Args:
            skill_id: ID навыка

        Returns:
            Экземпляр навыка или None
        """
        return self._skills.get(skill_id)

    def list(self) -> List[Skill]:
        """
        Возвращает список всех зарегистрированных навыков

        Returns:
            Список навыков
        """
        return list(self._skills.values())

    def find_by_capability(self, capability: str) -> List[Skill]:
        """
        Находит навыки по capability

        Args:
            capability: Название capability

        Returns:
            Список навыков с этой capability
        """
        return [
            skill for skill in self._skills.values()
            if capability in skill.capabilities
        ]

    def find_by_artifact(self, artifact_type: str) -> List[Skill]:
        """
        Находит навыки, которые производят данный тип артефактов

        Args:
            artifact_type: Тип артефакта

        Returns:
            Список навыков
        """
        return [
            skill for skill in self._skills.values()
            if artifact_type in skill.produces_artifacts
        ]

    def get_skill_info(self, skill_id: str) -> Optional[Dict]:
        """
        Получает информацию о навыке

        Args:
            skill_id: ID навыка

        Returns:
            Словарь с информацией или None
        """
        skill = self.get(skill_id)
        if not skill:
            return None

        return {
            "id": skill.id,
            "version": skill.version,
            "description": skill.description,
            "capabilities": skill.capabilities,
            "requirements": skill.requirements,
            "produces_artifacts": skill.produces_artifacts,
            "input_schema": skill.input_schema,
            "output_schema": skill.output_schema
        }

    def list_all_info(self) -> List[Dict]:
        """
        Возвращает информацию о всех навыках

        Returns:
            Список словарей с информацией
        """
        return [self.get_skill_info(skill.id) for skill in self.list()]


# Global registry instance
skill_registry = SkillRegistry()
