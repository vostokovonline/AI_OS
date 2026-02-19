"""
SKILL SET MVP v1 - The "Alphabet" of the System

5 minimal but sufficient skills that form the foundation:
1. text_to_file - Basic production (MUST HAVE)
2. structured_generation - Structure over chaos
3. web_research - Facts & sources
4. summarize_knowledge - Memory & reuse
5. self_check - Verification & control

Principle: Each skill produces artifacts, is verifiable, combines with others.
"""
import os
import uuid
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Constant for artifacts path - now persistent!
ARTIFACTS_PATH = os.getenv("ARTIFACTS_PATH", "/data/artifacts")
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

from skill_manifest import SkillManifest, SkillResult, ArtifactType, SkillCategory
from artifact_registry import artifact_registry
from verification_engine import VerificationEngine


class BaseSkill:
    """Базовый класс для всех skills"""
    def __init__(self, manifest: SkillManifest):
        self.manifest = manifest

    async def validate_inputs(self, inputs: Dict) -> tuple[bool, str]:
        required = self.manifest.inputs.required
        for field in required:
            if field not in inputs:
                return False, f"Missing required field: {field}"
        return True, None

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        raise NotImplementedError()


# ============================================================
# 🥇 1. text_to_file - Basic Production Skill (MUST HAVE)
# ============================================================

TEXT_TO_FILE_MANIFEST = SkillManifest(
    name="text_to_file",
    version="1.0",
    description="Write text content to file. Basic production skill - system cannot be mute without it.",
    category=SkillCategory.execution,
    agent_roles=["Writer", "Researcher", "Coder", "Planner"],
    inputs=type('Inputs', (), {"required": ["text", "filename"]})(
        schema="WriteFileInput",
        required=["text", "filename"]
    ),
    outputs=type('Outputs', (), {"artifact_type": ArtifactType.FILE, "schema_name": "WriteFileResult", "reusable": True})(
        artifact_type=ArtifactType.FILE,
        schema_name="WriteFileResult",
        reusable=True
    ),
    produces=[
        {
            "type": "FILE",
            "store": "file",
            "format": "auto",
            "path_template": "results/{goal_id}/{filename}",
            "tags": ["production", "output"]
        }
    ],
    constraints=None,
    verification=[
        {"name": "file_exists", "rule": "file_exists == true"},
        {"name": "min_length", "rule": "len(content) >= 200"}
    ],
    failure_modes=["permission_denied", "disk_full", "invalid_path"]
)


class TextToFileSkill(BaseSkill):
    """
    🥇 1. text_to_file - Basic Production Skill

    MANDATORY: System cannot function without it.

    Produces:
    - FILE: Any text content

    Gives:
    - Reports
    - Plans
    - Specifications
    - Explanations
    """

    def __init__(self):
        super().__init__(TEXT_TO_FILE_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        # Validate
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(status="failed", error=error, artifacts=[])

        text = inputs["text"]
        filename = inputs["filename"]

        try:
            # Create directory in persistent storage
            output_dir = os.path.join(ARTIFACTS_PATH, "results", goal_id)
            os.makedirs(output_dir, exist_ok=True)

            # Determine format from filename
            format_map = {
                ".md": "markdown",
                ".txt": "text",
                ".json": "json",
                ".csv": "csv",
                ".py": "python",
                ".js": "javascript"
            }
            file_format = "text"
            for ext, fmt in format_map.items():
                if filename.endswith(ext):
                    file_format = fmt
                    break

            # Write file
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            # Return artifact
            return SkillResult(
                status="success",
                artifacts=[{
                    "artifact_type": "FILE",
                    "content_kind": "file",
                    "content_location": file_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "Writer",
                    "domains": inputs.get("domains", []),
                    "tags": inputs.get("tags", ["production"]),
                    "language": file_format
                }],
                metadata={
                    "file_size": len(text),
                    "file_format": file_format
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Failed to write file: {str(e)}",
                artifacts=[]
            )


# ============================================================
# 🥈 2. structured_generation - Structure Over Chaos
# ============================================================

STRUCTURED_GENERATION_MANIFEST = SkillManifest(
    name="structured_generation",
    version="1.0",
    description="Generate structured output by schema. Removes chaos from LLM text.",
    category=SkillCategory.reasoning,
    agent_roles=["Planner", "Analyst", "Organizer"],
    inputs=type('Inputs', (), {"required": ["prompt", "output_schema"]})(
        schema="StructuredGenInput",
        required=["prompt", "output_schema"]
    ),
    outputs=type('Outputs', (), {"artifact_type": ArtifactType.DATASET, "schema_name": "StructuredGenResult", "reusable": True})(
        artifact_type=ArtifactType.DATASET,
        schema_name="StructuredGenResult",
        reusable=True
    ),
    produces=[
        {
            "type": "DATASET",
            "store": "file",
            "format": "json",
            "path_template": "results/{goal_id}/structured.json",
            "tags": ["structured", "generated"]
        }
    ],
    constraints=type('Constraints', (), {"max_tokens": 8000})(
        max_tokens=8000
    ),
    verification=[
        {"name": "schema_valid", "rule": "json_schema_valid == true"},
        {"name": "non_empty", "rule": "len(data) > 0"}
    ],
    failure_modes=["schema_parse_error", "generation_failed", "invalid_json"]
)


class StructuredGenerationSkill(BaseSkill):
    """
    🥈 2. structured_generation - Structure Over Chaos

    Produces:
    - DATASET: JSON following schema

    Gives:
    - Plans
    - Task lists
    - Decompositions
    - Configs
    """

    def __init__(self):
        super().__init__(STRUCTURED_GENERATION_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(status="failed", error=error, artifacts=[])

        prompt = inputs["prompt"]
        output_schema = inputs["output_schema"]

        try:
            # Generate structured output (using LLM)
            # TODO: Call actual LLM with schema
            # For now, mock implementation

            structured_data = await self._generate_structured(prompt, output_schema)

            # Write to file in persistent storage
            output_dir = os.path.join(ARTIFACTS_PATH, "results", goal_id)
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "structured.json")

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2)

            return SkillResult(
                status="success",
                artifacts=[{
                    "artifact_type": "DATASET",
                    "content_kind": "file",
                    "content_location": file_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "Planner",
                    "domains": inputs.get("domains", []),
                    "tags": ["structured", "generated"]
                }],
                metadata={
                    "schema": output_schema,
                    "items_count": len(structured_data) if isinstance(structured_data, list) else 1
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Structured generation failed: {str(e)}",
                artifacts=[]
            )

    async def _generate_structured(self, prompt: str, schema: Dict) -> Any:
        """Generate structured output following schema"""
        # TODO: Real LLM call with schema validation
        # Mock for now
        return {
            "items": [
                {"task": "Example task 1", "priority": "high"},
                {"task": "Example task 2", "priority": "medium"}
            ]
        }


# ============================================================
# 🥉 3. web_research - Facts & Sources
# ============================================================

WEB_RESEARCH_MANIFEST = SkillManifest(
    name="web_research",
    version="1.0",
    description="Perform web search and produce report with sources",
    category=SkillCategory.research,
    agent_roles=["Researcher", "WebSurfer"],
    inputs=type('Inputs', (), {"required": ["query"]})(
        schema="SearchQuery",
        required=["query"]
    ),
    outputs=type('Outputs', (), {"artifact_type": ArtifactType.REPORT, "schema_name": "ResearchReport", "reusable": True})(
        artifact_type=ArtifactType.REPORT,
        schema_name="ResearchReport",
        reusable=True
    ),
    produces=[
        {
            "type": "FILE",
            "store": "file",
            "format": "markdown",
            "path_template": "results/{goal_id}/research.md",
            "tags": ["research", "web"]
        },
        {
            "type": "KNOWLEDGE",
            "store": "vector_db",
            "tags": ["research", "web"]
        }
    ],
    constraints=type('Constraints', (), {"max_tokens": 4000})(
        max_tokens=4000
    ),
    verification=[
        {"name": "min_sources", "rule": "sources_count >= 3"}
    ],
    failure_modes=["no_sources", "timeout", "empty_result"]
)


class WebResearchSkill(BaseSkill):
    """
    🥉 3. web_research - Minimal Research Skill

    Produces:
    - FILE: research.md with sources
    - KNOWLEDGE: vector chunk

    Gives:
    - Facts
    - Sources
    - Knowledge
    """

    def __init__(self):
        super().__init__(WEB_RESEARCH_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(status="failed", error=error, artifacts=[])

        query = inputs["query"]

        try:
            # Perform research
            research_result = await self._perform_research(query)

            # Write report to persistent storage
            output_dir = os.path.join(ARTIFACTS_PATH, "results", goal_id)
            os.makedirs(output_dir, exist_ok=True)

            report_path = os.path.join(output_dir, "research.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(research_result["content"])

            # Create knowledge chunk
            vector_id = f"vector_{goal_id}_{uuid.uuid4().hex[:8]}"

            return SkillResult(
                status="success",
                artifacts=[
                    {
                        "artifact_type": "FILE",
                        "content_kind": "file",
                        "content_location": report_path,
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
                ],
                metadata={
                    "sources_count": research_result.get("sources_count", 0),
                    "query": query
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Web research failed: {str(e)}",
                artifacts=[]
            )

    async def _perform_research(self, query: str) -> Dict:
        """Perform actual web research"""
        # TODO: Real web search implementation
        # Mock for now
        return {
            "content": f"# Research: {query}\n\n## Summary\nResearch results...\n\n## Sources\n1. Example Source\n",
            "sources_count": 3
        }


# ============================================================
# 🧠 4. summarize_knowledge - Memory & Reuse
# ============================================================

SUMMARIZE_KNOWLEDGE_MANIFEST = SkillManifest(
    name="summarize_knowledge",
    version="1.0",
    description="Turn information into condensed knowledge for memory and reuse",
    category=SkillCategory.memory,
    agent_roles=["Analyst", "Summarizer"],
    inputs=type('Inputs', (), {"required": ["source_artifact_id"]})(
        schema="SummarizeInput",
        required=["source_artifact_id"]
    ),
    outputs=type('Outputs', (), {"artifact_type": ArtifactType.KNOWLEDGE, "schema_name": "SummaryResult", "reusable": True})(
        artifact_type=ArtifactType.KNOWLEDGE,
        schema_name="SummaryResult",
        reusable=True
    ),
    produces=[
        {
            "type": "KNOWLEDGE",
            "store": "vector_db",
            "tags": ["summary", "condensed"]
        }
    ],
    constraints=None,
    verification=[
        {"name": "non_empty", "rule": "len(content) > 150"}
    ],
    failure_modes=["artifact_not_found", "extraction_failed"]
)


class SummarizeKnowledgeSkill(BaseSkill):
    """
    🧠 4. summarize_knowledge - Memory & Reuse

    Produces:
    - KNOWLEDGE: Condensed summary

    Gives:
    - Memory preservation
    - Knowledge reuse
    """

    def __init__(self):
        super().__init__(SUMMARIZE_KNOWLEDGE_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(status="failed", error=error, artifacts=[])

        source_artifact_id = inputs["source_artifact_id"]

        try:
            # Load source artifact
            source_content = await self._load_artifact_content(source_artifact_id)

            # Summarize
            summary = await self._summarize(source_content)

            # Create knowledge chunk
            vector_id = f"summary_{goal_id}_{uuid.uuid4().hex[:8]}"

            return SkillResult(
                status="success",
                artifacts=[{
                    "artifact_type": "KNOWLEDGE",
                    "content_kind": "vector_db",
                    "content_location": vector_id,
                    "skill_name": self.manifest.name,
                    "agent_role": "Analyst",
                    "domains": inputs.get("domains", []),
                    "tags": ["summary", "condensed"]
                }],
                metadata={
                    "source_artifact_id": source_artifact_id,
                    "summary_length": len(summary)
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Summarization failed: {str(e)}",
                artifacts=[]
            )

    async def _load_artifact_content(self, artifact_id: str) -> str:
        """Load content from existing artifact"""
        # TODO: Load from artifact registry
        return "Sample artifact content..."

    async def _summarize(self, content: str) -> str:
        """Summarize content"""
        # TODO: Real summarization
        return f"Summary: {content[:200]}..."


# ============================================================
# 🧪 5. self_check - Verification & Control
# ============================================================

SELF_CHECK_MANIFEST = SkillManifest(
    name="self_check",
    version="1.0",
    description="Automatic sanity-check and verification of results",
    category=SkillCategory.evaluation,
    agent_roles=["Evaluator", "Checker"],
    inputs=type('Inputs', (), {"required": ["artifact_id"]})(
        schema="SelfCheckInput",
        required=["artifact_id"]
    ),
    outputs=type('Outputs', (), {"artifact_type": ArtifactType.EXECUTION_LOG, "schema_name": "CheckResult", "reusable": False})(
        artifact_type=ArtifactType.EXECUTION_LOG,
        schema_name="CheckResult",
        reusable=False
    ),
    produces=[
        {
            "type": "EXECUTION_LOG",
            "store": "file",
            "format": "json",
            "path_template": "results/{goal_id}/check_result.json",
            "tags": ["verification", "self_check"]
        }
    ],
    constraints=None,
    verification=[
        {"name": "verdict_present", "rule": "verdict in ['pass', 'fail']"}
    ],
    failure_modes=["artifact_not_found", "check_failed"]
)


class SelfCheckSkill(BaseSkill):
    """
    🧪 5. self_check - Minimal Verification

    Produces:
    - EXECUTION_LOG: Check results

    Gives:
    - Automatic sanity-check
    - Verification foundation
    """

    def __init__(self):
        super().__init__(SELF_CHECK_MANIFEST)

    async def execute(self, inputs: Dict, goal_id: str) -> SkillResult:
        is_valid, error = await self.validate_inputs(inputs)
        if not is_valid:
            return SkillResult(status="failed", error=error, artifacts=[])

        artifact_id = inputs["artifact_id"]
        criteria = inputs.get("criteria", [])

        try:
            # Load artifact
            # TODO: Load from registry
            artifact_data = {"id": artifact_id}

            # Perform checks
            check_results = await self._perform_checks(artifact_data, criteria)

            # Determine verdict
            verdict = "pass" if all(r["passed"] for r in check_results) else "fail"

            # Save check result to persistent storage
            output_dir = os.path.join(ARTIFACTS_PATH, "results", goal_id)
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, "check_result.json")

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "verdict": verdict,
                    "checks": check_results,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)

            return SkillResult(
                status="success",
                artifacts=[{
                    "artifact_type": "EXECUTION_LOG",
                    "content_kind": "file",
                    "content_location": file_path,
                    "skill_name": self.manifest.name,
                    "agent_role": "Evaluator",
                    "domains": ["verification"],
                    "tags": ["check", "validation"]
                }],
                metadata={
                    "verdict": verdict,
                    "checks_count": len(check_results)
                }
            )

        except Exception as e:
            return SkillResult(
                status="failed",
                error=f"Self-check failed: {str(e)}",
                artifacts=[]
            )

    async def _perform_checks(self, artifact: Dict, criteria: List[str]) -> List[Dict]:
        """Perform verification checks"""
        # TODO: Real checks
        return [
            {"name": "artifact_exists", "passed": True},
            {"name": "format_valid", "passed": True}
        ]


# ============================================================
# SKILL FACTORY
# ============================================================

class SkillSetFactory:
    """
    Factory for MVP skill set
    """
    _skills = {
        "text_to_file": (TextToFileSkill, TEXT_TO_FILE_MANIFEST),
        "structured_generation": (StructuredGenerationSkill, STRUCTURED_GENERATION_MANIFEST),
        "web_research": (WebResearchSkill, WEB_RESEARCH_MANIFEST),
        "summarize_knowledge": (SummarizeKnowledgeSkill, SUMMARIZE_KNOWLEDGE_MANIFEST),
        "self_check": (SelfCheckSkill, SELF_CHECK_MANIFEST),
    }

    @classmethod
    def create(cls, skill_name: str) -> BaseSkill:
        """Create skill instance"""
        if skill_name not in cls._skills:
            raise ValueError(f"Unknown skill: {skill_name}")

        skill_class, manifest = cls._skills[skill_name]
        return skill_class()

    @classmethod
    def get_manifest(cls, skill_name: str) -> SkillManifest:
        """Get skill manifest"""
        if skill_name not in cls._skills:
            raise ValueError(f"Unknown skill: {skill_name}")

        return cls._skills[skill_name][1]

    @classmethod
    def list_skills(cls) -> List[str]:
        """List available skills"""
        return list(cls._skills.keys())

    @classmethod
    def get_all_manifests(cls) -> List[SkillManifest]:
        """Get all manifests"""
        return [manifest for _, manifest in cls._skills.values()]


# ============================================================
# SKILL COMPOSITION EXAMPLES
# ============================================================

class SkillComposer:
    """
    Комбинирует навыки для выполнения сложных целей

    Example:
    Goal: "Research X and save findings"

    Flow:
    1. web_research → research.md + knowledge
    2. summarize_knowledge → condensed knowledge
    3. text_to_file → final_report.md
    4. self_check → validation

    Result: 3 artifacts, goal DONE
    """

    def __init__(self):
        self.factory = SkillSetFactory()

    async def execute_chain(
        self,
        skill_names: List[str],
        goal_id: str,
        initial_inputs: Dict
    ) -> List[SkillResult]:
        """
        Выполняет цепочку навыков

        Args:
            skill_names: ["web_research", "summarize_knowledge", "text_to_file"]
            goal_id: ID цели
            initial_inputs: Входные параметры для первого skill

        Returns:
            Список результатов от каждого skill
        """
        results = []
        current_inputs = initial_inputs

        for skill_name in skill_names:
            skill = self.factory.create(skill_name)

            # Execute skill
            result = await skill.execute(current_inputs, goal_id)
            results.append(result)

            # If failed - stop chain
            if result.status != "success":
                print(f"❌ Skill {skill_name} failed: {result.error}")
                break

            # Prepare inputs for next skill (pass artifacts)
            current_inputs = {
                "previous_results": result.artifacts,
                "goal_id": goal_id
            }

        return results


# ============================================================
# USAGE EXAMPLE
# ============================================================

def example_skill_combination():
    """Пример комбинирования навыков"""

    composer = SkillComposer()

    # Goal: "Research X and save conclusions"
    goal_id = "G-123"

    # Execute chain
    results = composer.execute_chain(
        skill_names=[
            "web_research",       # 1. Get facts
            "summarize_knowledge",  # 2. Condense
            "text_to_file",        # 3. Save report
            "self_check"           # 4. Verify
        ],
        goal_id=goal_id,
        initial_inputs={"query": "soil nutrition for tomatoes"}
    )

    # Check results
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.status} - {len(result.artifacts)} artifacts")

    # Expected:
    # 1. success - 2 artifacts (FILE + KNOWLEDGE)
    # 2. success - 1 artifact (KNOWLEDGE)
    # 3. success - 1 artifact (FILE)
    # 4. success - 1 artifact (EXECUTION_LOG)
    # Total: 5 artifacts, goal can be DONE
