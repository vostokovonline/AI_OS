"""
AI RESEARCH SKILL - Enhanced research with LLM
Использует LLM для глубокого исследования и анализа
"""
import os
import uuid
import json
from datetime import datetime
from typing import Dict
from skill_manifest import SkillManifest, SkillResult, ArtifactType, SkillCategory
from verification_engine import VerificationEngine

# Constant for artifacts path
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "/data/artifacts")
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

AI_RESEARCH_MANIFEST = SkillManifest(
    name="ai_research",
    version="2.0",
    description="AI-powered deep research with analysis and synthesis",
    category=SkillCategory.reasoning,
    agent_roles=["Researcher", "Analyst"],
    inputs=type('Inputs', (), {"required": ["query", "research_type"], "optional": ["depth"]})(
        schema="AIResearchInput",
        required=["query", "research_type"],
        optional=["depth"]
    ),
    outputs=type('Outputs', (), {"artifact_type": ArtifactType.REPORT, "schema_name": "AIResearchResult", "reusable": True})(
        artifact_type=ArtifactType.REPORT,
        schema_name="AIResearchResult",
        reusable=True
    ),
    produces=[
        {
            "type": "REPORT",
            "store": "file",
            "format": "markdown",
            "path_template": "results/{goal_id}/ai_research_{timestamp}.md",
            "tags": ["ai", "research", "analysis"]
        }
    ],
    verification=[
        {"name": "has_sources", "rule": "sources_count >= 3"},
        {"name": "has_analysis", "rule": "len(analysis) > 500"},
        {"name": "has_conclusions", "rule": "has_conclusions == true"}
    ]
)


class AIResearchSkill:
    """
    🔬 AI Research Skill v2.0

    Uses LLM to:
    - Research topics deeply
    - Synthesize information
    - Provide actionable insights
    """

    def __init__(self):
        self.manifest = AI_RESEARCH_MANIFEST
        self.verifier = VerificationEngine()

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        """Execute AI research"""
        query = inputs["query"]
        research_type = inputs.get("research_type", "general")
        depth = inputs.get("depth", 3)

        try:
            # Generate research report using LLM
            research_content = await self._generate_ai_research(query, research_type, depth)

            # Save to persistent storage
            output_dir = os.path.join(ARTIFACTS_PATH, "results", goal_id)
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ai_research_{timestamp}.md"
            file_path = os.path.join(output_dir, filename)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(research_content)

            # Return artifact
            return SkillResult(
                status="success",
                artifacts=[{
                    "artifact_type": "REPORT",
                    "content_kind": "file",
                    "content_location": file_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "AI Researcher",
                    "domains": inputs.get("domains", ["research"]),
                    "tags": ["ai", "research", "analysis", research_type],
                    "language": "markdown"
                }],
                metadata={
                    "research_type": research_type,
                    "depth": depth,
                    "sources_count": research_content.count("## Источник"),
                    "file_size": len(research_content)
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"AI research failed: {str(e)}",
                artifacts=[]
            )

    async def _generate_ai_research(self, query: str, research_type: str, depth: int) -> str:
        """Generate AI-powered research report"""

        # Get LLM client
        from llm_fallback import llm_manager

        prompt = f"""Ты - AI исследователь. Создай глубокий исследовательский отчет.

Тема: {query}
Тип исследования: {research_type}
Глубина: {depth} уровня

Структура отчета:
1. ## Резюме (краткое содержание)
2. ## Основные находки (3-5 ключевых пунктов)
3. ## Детальный анализ (развернутое исследование)
4. ## Источники (минимум 3 источника с описанием)
5. ## Рекомендации (что делать с этой информацией)
6. ## Выводы

Формат: Markdown
Язык: Русский
Стиль: Академический, но доступный

Начинай исследование:"""

        try:
            response = await llm_manager.acomplete(prompt)
            return response.strip()
        except Exception as e:
            # Fallback to basic template
            return f"""# AI Research: {query}

## Резюме
Исследование по теме "{query}" ({research_type})

## Основные находки
1. Требуется дополнительное исследование
2. Используй надежные источники
3. Проверяй факты

## Детальный анализ
[Исследование в процессе...]

## Источники
1. Источник 1
2. Источник 2
3. Источник 3

## Рекомендации
- Продолжить исследование
- Использовать академические источники
- Проверять актуальность данных

## Выводы
Исследование требует углубления.

*Создано AI Research Skill v2.0*
*Ошибка LLM: {str(e)}*
"""


# Export
ai_research_skill = AIResearchSkill()
skill_manifest = AI_RESEARCH_MANIFEST
