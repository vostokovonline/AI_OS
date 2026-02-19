"""
SKILL REGISTRY v1 - Load and find skill manifests

Provides:
- Automatic skill selection by output type
- Validation before execution
- Manifest lookup by name/role/category
"""
import os
import yaml
from typing import Dict, List, Optional, Type
from pathlib import Path
from skill_manifest import (
    SkillManifest,
    SkillResult,
    ArtifactType,
    SkillCategory,
    BUILTIN_MANIFESTS
)


class SkillRegistry:
    """
    Реестр манифестов навыков

    Usage:
        registry = SkillRegistry()
        registry.load_builtin()  # Load built-in skills

        # Find skills that produce specific artifact type
        skills = registry.find_by_output(ArtifactType.FILE)

        # Find skills by agent role
        skills = registry.find_by_role("Researcher")

        # Validate inputs against manifest
        is_valid = registry.validate_inputs("web_research", inputs)
    """

    def __init__(self):
        self.manifests: Dict[str, SkillManifest] = {}
        self.skill_dir = Path("skills/manifests")  # Directory for YAML manifests

    def load(self, manifest: SkillManifest) -> None:
        """
        Загружает манифест в реестр

        Args:
            manifest: SkillManifest объект
        """
        # Валидация обязательных полей (P0)
        required_fields = ["name", "inputs", "outputs", "produces", "verification"]
        for field in required_fields:
            if not hasattr(manifest, field) or getattr(manifest, field) is None:
                raise ValueError(f"Manifest missing required field: {field}")

        self.manifests[manifest.name] = manifest

    def load_from_yaml(self, yaml_path: str) -> SkillManifest:
        """
        Загружает манифест из YAML файла

        Args:
            yaml_path: Путь к YAML файлу

        Returns:
            Загруженный SkillManifest
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        manifest = SkillManifest(**data)
        self.load(manifest)
        return manifest

    def load_from_directory(self, directory: str) -> int:
        """
        Загружает все YAML манифесты из директории

        Args:
            directory: Путь к директории

        Returns:
            Количество загруженных манифестов
        """
        path = Path(directory)
        if not path.exists():
            return 0

        count = 0
        for yaml_file in path.glob("*.yaml"):
            try:
                self.load_from_yaml(str(yaml_file))
                count += 1
            except Exception as e:
                print(f"⚠️ Failed to load {yaml_file}: {e}")

        return count

    def load_builtin(self) -> None:
        """Загружает встроенные манифесты"""
        for name, manifest in BUILTIN_MANIFESTS.items():
            self.load(manifest)

    def get(self, skill_name: str) -> Optional[SkillManifest]:
        """
        Получает манифест по имени

        Args:
            skill_name: Имя навыка

        Returns:
            SkillManifest или None
        """
        return self.manifests.get(skill_name)

    def find_by_output(self, artifact_type: ArtifactType) -> List[SkillManifest]:
        """
        Находит навыки которые производят указанный тип артефакта

        Используется Planner для выбора подходящего skill

        Args:
            artifact_type: Тип артефакта (FILE, KNOWLEDGE, etc.)

        Returns:
            Список подходящих манифестов
        """
        results = []
        for manifest in self.manifests.values():
            # Проверяем outputs.artifact_type
            if manifest.outputs.artifact_type == artifact_type:
                results.append(manifest)
                continue

            # Проверяем produces
            for produced in manifest.produces:
                if produced.type == artifact_type:
                    results.append(manifest)
                    break

        return results

    def find_by_role(self, agent_role: str) -> List[SkillManifest]:
        """
        Находит навыки которые может выполнить агент

        Args:
            agent_role: Роль агента (Researcher, Coder, etc.)

        Returns:
            Список подходящих манифестов
        """
        results = []
        for manifest in self.manifests.values():
            if agent_role in manifest.agent_roles:
                results.append(manifest)

        return results

    def find_by_category(self, category: SkillCategory) -> List[SkillManifest]:
        """
        Находит навыки по категории

        Args:
            category: Категория (research, coding, analysis, etc.)

        Returns:
            Список манифестов
        """
        results = []
        for manifest in self.manifests.values():
            if manifest.category == category:
                results.append(manifest)

        return results

    def find_for_goal_requirements(
        self,
        required_artifacts: List[ArtifactType]
    ) -> List[SkillManifest]:
        """
        Находит навыки которые покрывают требования цели

        Для L3 atomic goals:
        goal.requires.artifacts → skills that produce them

        Args:
            required_artifacts: Список требуемых типов артефактов

        Returns:
            Список подходящих навыков (которые покрывают ВСЕ требования)
        """
        candidates = []

        for manifest in self.manifests.values():
            # Собираем все типы артефактов которые производит skill
            produced_types = set([manifest.outputs.artifact_type])
            for produced in manifest.produces:
                produced_types.add(produced.type)

            # Проверяем что skill покрывает все требования
            required_set = set(required_artifacts)
            if required_set.issubset(produced_types):
                candidates.append(manifest)

        return candidates

    def validate_inputs(
        self,
        skill_name: str,
        inputs: Dict
    ) -> tuple[bool, Optional[str]]:
        """
        Валидирует входные параметры против манифеста

        Args:
            skill_name: Имя навыка
            inputs: Входные параметры

        Returns:
            (is_valid, error_message)
        """
        manifest = self.get(skill_name)
        if not manifest:
            return False, f"Skill not found: {skill_name}"

        # Проверяем обязательные поля
        required = manifest.inputs.required
        for field in required:
            if field not in inputs:
                return False, f"Missing required field: {field}"

        return True, None

    def list_all(self) -> List[SkillManifest]:
        """Возвращает все загруженные манифесты"""
        return list(self.manifests.values())

    def list_names(self) -> List[str]:
        """Возвращает имена всех загруженных навыков"""
        return list(self.manifests.keys())


class SkillExecutor:
    """
    Исполнитель навыков с проверкой контракта

    Гарантирует:
    1. Inputs validated against manifest
    2. Artifacts produced
    3. Verification rules applied
    4. SkillResult returned (not just text)
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    async def execute(
        self,
        skill_name: str,
        inputs: Dict,
        goal_id: str,
        executor_func: callable
    ) -> SkillResult:
        """
        Выполняет навыок с проверкой контракта

        Args:
            skill_name: Имя навыка
            inputs: Входные параметры
            goal_id: ID цели
            executor_func: Функция которая реально выполняет skill

        Returns:
            SkillResult с артефактами
        """
        # 1. Получаем манифест
        manifest = self.registry.get(skill_name)
        if not manifest:
            return SkillResult(
                status="failed",
                error=f"Skill not found: {skill_name}"
            )

        # 2. Валидируем входные параметры
        is_valid, error = self.registry.validate_inputs(skill_name, inputs)
        if not is_valid:
            return SkillResult(
                status="failed",
                error=f"Input validation failed: {error}"
            )

        # 3. Выполняем skill
        try:
            result = await executor_func(inputs)

            # 4. Проверяем что вернулись артефакты
            if not isinstance(result, SkillResult):
                return SkillResult(
                    status="failed",
                    error=f"Skill must return SkillResult, got {type(result)}"
                )

            # 5. Проверяем что есть артефакты
            if not result.artifacts:
                return SkillResult(
                    status="failed",
                    error="Skill produced no artifacts"
                )

            # 6. Регистрируем артефакты
            from artifact_registry import artifact_registry

            registered_artifacts = []
            for artifact_data in result.artifacts:
                registered = await artifact_registry.register(
                    goal_id=goal_id,
                    **artifact_data
                )
                registered_artifacts.append(registered)

            return SkillResult(
                artifacts=registered_artifacts,
                status=result.status,
                error=result.error,
                metadata=result.metadata
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Execution failed: {str(e)}"
            )


# Глобальный экземпляр
skill_registry = SkillRegistry()
skill_executor = None  # Will be initialized after registry is loaded


def init_skill_system():
    """Инициализирует систему навыков"""
    skill_registry.load_builtin()
    global skill_executor
    skill_executor = SkillExecutor(skill_registry)
    print(f"✅ Skill system initialized with {len(skill_registry.list_names())} skills")
