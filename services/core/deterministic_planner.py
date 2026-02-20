"""
DETERMINISTIC GOAL PLANNER v1
Selects skills based on manifests (NOT LLM)

Key principle:
- LLM does NOT select skills
- LLM can SUGGEST, but code DECIDES
- Selection is verifiable and deterministic
"""
from typing import List, Dict, Optional, Type
from skill_manifest import SkillManifest, ArtifactType
from skill_registry import skill_registry


class SkillQuery:
    """Запрос на выбор навыка"""

    def __init__(
        self,
        required_artifacts: List[ArtifactType],
        agent_role: Optional[str] = None,
        category: Optional[str] = None,
        goal_type: Optional[str] = None
    ):
        self.required_artifacts = required_artifacts
        self.agent_role = agent_role
        self.category = category
        self.goal_type = goal_type


class GoalPlanner:
    """
    Детерминированный планировщик целей

    Выбирает навыки на основе:
    1. Требуемых артефактов (goal.requires.artifacts)
    2. Роли агента
    3. Категории навыка
    4. Типа цели

    НЕ использует LLM для выбора!
    """

    def __init__(self):
        self.registry = skill_registry

    def select_skill_for_goal(self, goal: Dict) -> Optional[str]:
        """
        Выбирает навык для цели

        Args:
            goal: {
                "id": "uuid",
                "level": "L3",
                "is_atomic": true,
                "requires": {
                    "artifacts": ["FILE", "KNOWLEDGE"]
                },
                "goal_type": "achievable",
                "domains": ["research"]
            }

        Returns:
            skill_name or None
        """
        # 1. Определяем требования
        required_artifacts = goal.get("requires", {}).get("artifacts", [])

        if not required_artifacts:
            # Если требования не указаны - берем по умолчанию
            if goal.get("is_atomic"):
                # L3 цели должны требовать хотя бы FILE
                required_artifacts = [ArtifactType.FILE]
            else:
                # Неатомарные цели могут не требовать артефактов
                return None

        # Конвертируем строки в ArtifactType
        try:
            artifact_types = [ArtifactType(t) for t in required_artifacts]
        except ValueError as e:
            logger.info(f"❌ Invalid artifact type: {e}")
            return None

        # 2. Формируем запрос
        query = SkillQuery(
            required_artifacts=artifact_types,
            agent_role=goal.get("preferred_agent_role"),
            category=goal.get("preferred_category"),
            goal_type=goal.get("goal_type")
        )

        # 3. Ищем подходящие навыки
        candidates = self._find_candidates(query)

        if not candidates:
            logger.info(f"⚠️ No skills found for requirements: {required_artifacts}")
            return None

        # 4. Выбираем лучший (детерминированно!)
        best_skill = self._select_best(candidates, goal)

        return best_skill

    def _find_candidates(self, query: SkillQuery) -> List[SkillManifest]:
        """
        Находит навыки которые покрывают требования

        Returns:
            Список подходящих манифестов
        """
        candidates = []

        for manifest in self.registry.list_all():
            # Проверяем 1: Покрывает ли все требуемые артефакты?
            if not self._covers_artifacts(manifest, query.required_artifacts):
                continue

            # Проверяем 2: Подходит ли агент?
            if query.agent_role:
                if query.agent_role not in manifest.agent_roles:
                    continue

            # Проверяем 3: Подходит ли категория?
            if query.category:
                if manifest.category != query.category:
                    continue

            candidates.append(manifest)

        return candidates

    def _covers_artifacts(
        self,
        manifest: SkillManifest,
        required_types: List[ArtifactType]
    ) -> bool:
        """
        Проверяет что навык покрывает все требуемые типы артефактов

        Args:
            manifest: Манифест навыка
            required_types: Требуемые типы артефактов

        Returns:
            True если покрывает ВСЕ требуемые типы
        """
        # Собираем все типы которые производит навык
        produced_types = set([manifest.outputs.artifact_type])
        for produced in manifest.produces:
            produced_types.add(produced.type)

        # Проверяем что все требуемые типы покрываются
        required_set = set(required_types)
        return required_set.issubset(produced_types)

    def _select_best(
        self,
        candidates: List[SkillManifest],
        goal: Dict
    ) -> str:
        """
        Выбирает лучший навык из кандидатов

        Приоритеты (детерминированные):
        1. Совпадение категории с доменами цели
        2. Меньшее количество артефактов (точнее попадание)
        3. Порядок в списке (built-in first)

        Returns:
            Имя навыка
        """
        if len(candidates) == 1:
            return candidates[0].name

        # Приоритет 1: Совпадение доменов
        goal_domains = set(goal.get("domains", []))

        # Если домены указаны - ищем совпадения
        if goal_domains:
            for manifest in candidates:
                # Проверяем теги артефактов
                for produced in manifest.produces:
                    if set(produced.tags) & goal_domains:
                        return manifest.name

        # Приоритет 2: Меньшее количество артефактов (точнее)
        candidates_with_count = [
            (manifest, len(manifest.produces))
            for manifest in candidates
        ]
        candidates_with_count.sort(key=lambda x: x[1])  # Сортируем по количеству
        best = candidates_with_count[0][0]

        return best.name

    def plan_execution(self, goal: Dict) -> Dict:
        """
        Планирует выполнение цели

        Args:
            goal: Цель с требованиями

        Returns:
            {
                "skill_name": "web_research",
                "inputs": {...},
                "expected_artifacts": [...],
                "verification_rules": [...]
            }
        """
        # 1. Выбираем навык
        skill_name = self.select_skill_for_goal(goal)

        if not skill_name:
            return {
                "error": "No suitable skill found",
                "required_artifacts": goal.get("requires", {}).get("artifacts", [])
            }

        # 2. Получаем манифест
        manifest = self.registry.get(skill_name)

        if not manifest:
            return {
                "error": f"Skill manifest not found: {skill_name}"
            }

        # 3. Формируем входные параметры из цели
        inputs = self._prepare_inputs(goal, manifest)

        # 4. Определяем ожидаемые артефакты
        expected_artifacts = [
            {
                "type": manifest.outputs.artifact_type
            }
        ]
        for produced in manifest.produces:
            expected_artifacts.append({
                "type": produced.type,
                "store": produced.store,
                "format": produced.format
            })

        # 5. Получаем правила верификации
        verification_rules = [
            {
                "name": rule.name,
                "rule": rule.rule,
                "description": rule.description
            }
            for rule in manifest.verification
        ]

        return {
            "skill_name": skill_name,
            "inputs": inputs,
            "expected_artifacts": expected_artifacts,
            "verification_rules": verification_rules,
            "constraints": manifest.constraints.dict() if manifest.constraints else None
        }

    def _prepare_inputs(self, goal: Dict, manifest: SkillManifest) -> Dict:
        """
        Готовит входные параметры для навыка из цели

        Args:
            goal: Цель
            manifest: Манифест навыка

        Returns:
            Входные параметры
        """
        inputs = {}

        # Обязательные параметры берем из цели
        for field in manifest.inputs.required:
            if field == "query":
                inputs[field] = goal.get("title") or goal.get("description", "")
            elif field == "repo_path":
                inputs[field] = goal.get("repo_path", "")
            else:
                inputs[field] = goal.get(field, "")

        # Опциональные параметры
        for field in manifest.inputs.optional:
            if field in goal:
                inputs[field] = goal[field]

        return inputs


# ============= USAGE EXAMPLE =============

def example_usage():
    """Пример использования детерминированного планировщика"""

    # Initialize skill system
    from skill_registry import init_skill_system
    init_skill_system()

    # Create planner
    planner = GoalPlanner()

    # Define goal with requirements
    goal = {
        "id": "G-123",
        "title": "Research soil nutrition for tomatoes",
        "description": "Find best practices for tomato soil",
        "level": "L3",
        "is_atomic": True,
        "goal_type": "achievable",
        "domains": ["research", "agriculture"],
        "requires": {
            "artifacts": ["FILE", "KNOWLEDGE"]
        }
    }

    # Plan execution
    plan = planner.plan_execution(goal)

    logger.info(f"✅ Selected skill: {plan['skill_name']}")
    logger.info(f"📥 Inputs: {plan['inputs']}")
    logger.info(f"📦 Expected artifacts: {len(plan['expected_artifacts'])}")
    logger.info(f"🔍 Verification rules: {len(plan['verification_rules'])}")

    # Execute skill
    from skills.production_skills import SkillFactory

    skill = SkillFactory.create(plan["skill_name"])

    import asyncio
    result = asyncio.run(skill.execute(
        inputs=plan["inputs"],
        goal_id=goal["id"]
    ))

    logger.info(f"🎯 Execution result: {result.status}")
    logger.info(f"📦 Artifacts produced: {len(result.artifacts)}")


# Глобальный экземпляр
goal_planner = GoalPlanner()
