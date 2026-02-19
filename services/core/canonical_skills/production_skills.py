"""
PRODUCTION-GRADE SKILL v1
Example: WebResearchSkill with proper manifest integration

Key principle:
- Skill CANNOT complete without returning artifacts
- LLM text is INTERNAL, not the result
- Artifacts are PRIMARY, text is secondary
"""
import os
import uuid
from typing import Dict, Any
from datetime import datetime
from skill_manifest import SkillManifest, SkillResult, ArtifactType
from artifact_registry import artifact_registry


class BaseSkill:
    """Базовый класс для всех production skills"""

    def __init__(self, manifest: SkillManifest):
        self.manifest = manifest

    async def validate_inputs(self, inputs: Dict) -> tuple[bool, str]:
        """
        Валидирует входные параметры против манифеста

        Returns:
            (is_valid, error_message)
        """
        required = self.manifest.inputs.required
        for field in required:
            if field not in inputs:
                return False, f"Missing required field: {field}"

        return True, None

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        """
        Выполняет skill с проверкой контракта

        Должен быть переопределен в subclass

        Returns:
            SkillResult with artifacts (MANDATORY)
        """
        raise NotImplementedError("Subclasses must implement execute()")


class WebResearchSkill(BaseSkill):
    """
    Production-grade Web Research Skill

    Produces:
    - FILE: research.md (structured report)
    - KNOWLEDGE: vector chunk (summary)
    """

    def __init__(self):
        # Load manifest
        from skill_manifest import WEB_RESEARCH_MANIFEST
        super().__init__(WEB_RESEARCH_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        """
        Выполняет веб-исследование

        Args:
            inputs: {"query": "...", "max_sources": 5 (optional)}
            goal_id: ID цели

        Returns:
            SkillResult with 2 artifacts (FILE + KNOWLEDGE)
        """
        # 1. Валидируем входные параметры
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(
                status="failed",
                error=f"Input validation failed: {error}",
                artifacts=[]
            )

        query = inputs["query"]
        max_sources = inputs.get("max_sources", 5)

        try:
            # 2. Выполняем исследование (внутренняя логика)
            report_data = await self._perform_research(query, max_sources)

            # 3. Создаем директорию для результатов
            output_dir = f"results/{goal_id}"
            os.makedirs(output_dir, exist_ok=True)

            # 4. Создаем FILE artifact (research report)
            file_path = f"{output_dir}/research.md"
            await self._write_markdown_report(file_path, report_data)

            # 5. Создаем KNOWLEDGE artifact (vector chunk)
            vector_id = await self._create_knowledge_chunk(
                goal_id,
                report_data["summary"],
                report_data.get("sources", [])
            )

            # 6. Формируем артефакты для регистрации
            artifacts = [
                {
                    "artifact_type": "FILE",
                    "content_kind": "file",
                    "content_location": file_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "Researcher",
                    "domains": ["research"],
                    "tags": ["web", "sources"],
                    "language": "markdown"
                },
                {
                    "artifact_type": "KNOWLEDGE",
                    "content_kind": "vector_db",
                    "content_location": vector_id,
                    "skill_name": self.manifest.name,
                    "agent_role": "Researcher",
                    "domains": ["research"],
                    "tags": ["web", "summary"]
                }
            ]

            # 7. Возвращаем результат с артефактами
            return SkillResult(
                status="success",
                artifacts=artifacts,
                metadata={
                    "sources_count": report_data.get("sources_count", 0),
                    "summary_length": len(report_data.get("summary", "")),
                    "execution_time": report_data.get("execution_time")
                }
            )

        except Exception as e:
            # При ошибке - FAILED status без артефактов
            return SkillResult(
                status="failed",
                error=f"Research failed: {str(e)}",
                artifacts=[]
            )

    async def _perform_research(self, query: str, max_sources: int) -> Dict:
        """
        Внутренняя логика исследования

        Returns:
            {
                "title": "...",
                "summary": "...",
                "sources": [...],
                "sources_count": 5,
                "execution_time": 42.5
            }
        """
        # TODO: Реальная логика исследования
        # Здесь будет вызов агента, поиск по вебу, etc.

        # Заглушка для примера
        return {
            "title": f"Research: {query}",
            "summary": f"Comprehensive analysis of {query} based on {max_sources} sources...",
            "sources": [
                {"title": "Source 1", "url": "https://example.com/1"},
                {"title": "Source 2", "url": "https://example.com/2"},
                {"title": "Source 3", "url": "https://example.com/3"}
            ],
            "sources_count": 3,
            "execution_time": 15.0
        }

    async def _write_markdown_report(self, path: str, data: Dict):
        """Записывает отчет в markdown формате"""
        content = f"""# {data['title']}

## Summary

{data['summary']}

## Sources

"""
        for i, source in enumerate(data.get('sources', []), 1):
            content += f"{i}. [{source['title']}]({source['url']})\n"

        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    async def _create_knowledge_chunk(self, goal_id: str, summary: str, sources: list) -> str:
        """
        Создает knowledge chunk для vector DB

        Returns:
            vector_id
        """
        # TODO: Реальная запись в vector DB
        # Заглушка - возвращаем ID
        vector_id = f"vector_{goal_id}_{uuid.uuid4().hex[:8]}"
        return vector_id


class CodeAnalysisSkill(BaseSkill):
    """
    Production-grade Code Analysis Skill

    Produces:
    - FILE: analysis.md (structured analysis)
    - FILE: metrics.json (code metrics)
    """

    def __init__(self):
        from skill_manifest import CODE_ANALYSIS_MANIFEST
        super().__init__(CODE_ANALYSIS_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        """Выполняет анализ кода"""
        # Валидация
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(
                status="failed",
                error=f"Input validation failed: {error}",
                artifacts=[]
            )

        repo_path = inputs["repo_path"]
        analysis_type = inputs.get("analysis_type", "general")

        try:
            # Выполняем анализ
            analysis_data = await self._perform_analysis(repo_path, analysis_type)

            # Создаем артефакты
            output_dir = f"results/{goal_id}"
            os.makedirs(output_dir, exist_ok=True)

            # FILE 1: analysis.md
            analysis_path = f"{output_dir}/analysis.md"
            await self._write_analysis(analysis_path, analysis_data)

            # FILE 2: metrics.json
            metrics_path = f"{output_dir}/metrics.json"
            await self._write_metrics(metrics_path, analysis_data)

            artifacts = [
                {
                    "artifact_type": "FILE",
                    "content_kind": "file",
                    "content_location": analysis_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "Coder",
                    "domains": ["programming", "analysis"],
                    "tags": ["code", "analysis"],
                    "language": "markdown"
                },
                {
                    "artifact_type": "FILE",
                    "content_kind": "file",
                    "content_location": metrics_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "Coder",
                    "domains": ["programming"],
                    "tags": ["code", "metrics"],
                    "language": "json"
                }
            ]

            return SkillResult(
                status="success",
                artifacts=artifacts,
                metadata={
                    "findings_count": analysis_data.get("findings_count", 0),
                    "files_analyzed": analysis_data.get("files_analyzed", 0)
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Code analysis failed: {str(e)}",
                artifacts=[]
            )

    async def _perform_analysis(self, repo_path: str, analysis_type: str) -> Dict:
        """Внутренняя логика анализа"""
        # TODO: Реальный анализ
        return {
            "findings_count": 5,
            "files_analyzed": 12,
            "complexity_score": 7.5
        }

    async def _write_analysis(self, path: str, data: Dict):
        """Записывает анализ в markdown"""
        content = f"""# Code Analysis Report

## Summary

This repository has been analyzed.

## Findings

- Total findings: {data['findings_count']}
- Files analyzed: {data['files_analyzed']}
- Complexity score: {data.get('complexity_score', 'N/A')}
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

    async def _write_metrics(self, path: str, data: Dict):
        """Записывает метрики в JSON"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# ============= SKILL FACTORY =============

class SkillFactory:
    """
    Фабрика для создания навыков по имени
    """
    _skills = {
        "web_research": WebResearchSkill,
        "code_analysis": CodeAnalysisSkill,
    }

    @classmethod
    def create(cls, skill_name: str) -> BaseSkill:
        """Создает экземпляр навыка"""
        if skill_name not in cls._skills:
            raise ValueError(f"Unknown skill: {skill_name}")

        return cls._skills[skill_name]()

    @classmethod
    def list_available(cls) -> list:
        """Возвращает список доступных навыков"""
        return list(cls._skills.keys())
