"""
SKILL MANIFEST v1 - Clear contracts for skills

Key principle: "Without manifest, skill = black box. With manifest, skill = contract"

Solves 5 problems:
1. Planner knows what skill can actually do
2. Goal System knows what will be produced
3. Artifact Layer knows what to register
4. Evaluation knows what to check
5. Dashboard shows results, not chatter
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class ArtifactType(str, Enum):
    """Типы артефактов (совпадает с Artifact Layer v1)"""
    FILE = "FILE"
    KNOWLEDGE = "KNOWLEDGE"
    DATASET = "DATASET"
    REPORT = "REPORT"
    LINK = "LINK"
    EXECUTION_LOG = "EXECUTION_LOG"


class SkillCategory(str, Enum):
    """Категории навыков"""
    research = "research"
    coding = "coding"
    analysis = "analysis"
    communication = "communication"
    execution = "execution"
    creative = "creative"


class VerificationRule(BaseModel):
    """Правило верификации"""
    name: str
    rule: str  # Выражение правила (например, "len(content) > 500")
    description: Optional[str] = None


class ArtifactProduced(BaseModel):
    """Описание артефакта который производит skill"""
    type: ArtifactType
    store: str  # vector_db | file | db | external
    format: Optional[str] = None  # markdown, json, csv, etc.
    path_template: Optional[str] = None  # "results/{goal_id}/research.md"
    tags: List[str] = []


class SkillConstraint(BaseModel):
    """Ограничения выполнения skill"""
    max_tokens: Optional[int] = None
    max_sources: Optional[int] = None
    timeout_sec: Optional[int] = None
    requires_api: Optional[List[str]] = None


class SkillInput(BaseModel):
    """Описание входных параметров"""
    schema_name: str  # Имя схемы (например, "SearchQuery")
    required: List[str] = []  # Обязательные поля
    optional: List[str] = []  # Опциональные поля


class SkillOutput(BaseModel):
    """Описание выходных результатов"""
    artifact_type: ArtifactType
    schema_name: str  # Имя схемы (например, "ResearchReport")
    reusable: bool = True


class SkillManifest(BaseModel):
    """
    Манифест навыка - контракт между системой и skill

    Обязательные поля (P0):
    - name
    - inputs.schema
    - outputs.artifact_type
    - produces
    - verification
    """
    # Basic info
    name: str
    version: str = "1.0"
    description: str = ""

    # Classification
    category: SkillCategory
    agent_roles: List[str] = []  # Какие агенты могут выполнять

    # Input/Output contract
    inputs: SkillInput
    outputs: SkillOutput

    # What artifacts are produced
    produces: List[ArtifactProduced] = []

    # Constraints
    constraints: Optional[SkillConstraint] = None

    # Verification rules (CODE-BASED, not LLM)
    verification: List[VerificationRule] = []

    # Failure modes
    failure_modes: List[str] = []

    class Config:
        use_enum_values = True


class SkillResult(BaseModel):
    """
    Обязательный результат выполнения skill

    Execution contract: Every skill MUST return:
    - artifacts: list of Artifact
    - status: success | failed
    - error: str | None
    """
    artifacts: List[Dict] = []  # List of artifact descriptors
    status: str  # success | failed
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ============= BUILT-IN MANIFESTS v1 =============

# Research skills
WEB_RESEARCH_MANIFEST = SkillManifest(
    name="web_research",
    version="1.0",
    description="Performs web search and produces a structured research artifact with sources and summary",
    category=SkillCategory.research,
    agent_roles=["Researcher", "WebSurfer"],
    inputs=SkillInput(
        schema_name="SearchQuery",
        required=["query"],
        optional=["max_sources", "time_range"]
    ),
    outputs=SkillOutput(
        artifact_type=ArtifactType.REPORT,
        schema_name="ResearchReport",
        reusable=True
    ),
    produces=[
        ArtifactProduced(
            type=ArtifactType.FILE,
            store="file",
            format="markdown",
            path_template="results/{goal_id}/research.md",
            tags=["research", "web", "sources"]
        ),
        ArtifactProduced(
            type=ArtifactType.KNOWLEDGE,
            store="vector_db",
            tags=["research", "web"]
        )
    ],
    constraints=SkillConstraint(
        max_tokens=4000,
        max_sources=7,
        timeout_sec=60,
        requires_api=["search"]
    ),
    verification=[
        VerificationRule(
            name="min_sources",
            rule="sources_count >= 3",
            description="Must have at least 3 sources"
        ),
        VerificationRule(
            name="citations_present",
            rule="has_citations == true",
            description="Must include citations"
        ),
        VerificationRule(
            name="non_empty_summary",
            rule="len(summary) > 300",
            description="Summary must be > 300 characters"
        )
    ],
    failure_modes=["no_sources", "timeout", "empty_result"]
)

CODE_ANALYSIS_MANIFEST = SkillManifest(
    name="code_analysis",
    version="1.0",
    description="Analyzes codebase and produces structured analysis report",
    category=SkillCategory.analysis,
    agent_roles=["Coder", "Researcher"],
    inputs=SkillInput(
        schema_name="CodeAnalysisQuery",
        required=["repo_path", "analysis_type"]
    ),
    outputs=SkillOutput(
        artifact_type=ArtifactType.REPORT,
        schema_name="CodeAnalysisReport",
        reusable=True
    ),
    produces=[
        ArtifactProduced(
            type=ArtifactType.FILE,
            store="file",
            format="markdown",
            path_template="results/{goal_id}/analysis.md",
            tags=["code", "analysis"]
        ),
        ArtifactProduced(
            type=ArtifactType.FILE,
            store="file",
            format="json",
            path_template="results/{goal_id}/metrics.json",
            tags=["code", "metrics"]
        )
    ],
    constraints=SkillConstraint(
        max_tokens=8000,
        timeout_sec=120
    ),
    verification=[
        VerificationRule(
            name="file_exists",
            rule="analysis_file_exists == true"
        ),
        VerificationRule(
            name="min_findings",
            rule="findings_count >= 1",
            description="Must find at least 1 issue or insight"
        )
    ],
    failure_modes=["repo_not_found", "parse_error", "timeout"]
)

FILE_WRITE_MANIFEST = SkillManifest(
    name="file_write",
    version="1.0",
    description="Writes content to file with proper formatting",
    category=SkillCategory.execution,
    agent_roles=["Coder"],
    inputs=SkillInput(
        schema_name="FileWriteQuery",
        required=["file_path", "content"]
    ),
    outputs=SkillOutput(
        artifact_type=ArtifactType.FILE,
        schema_name="FileWriteResult",
        reusable=False
    ),
    produces=[
        ArtifactProduced(
            type=ArtifactType.FILE,
            store="file",
            format="auto",  # Determined by extension
            path_template="{file_path}",
            tags=["execution", "file"]
        )
    ],
    constraints=SkillConstraint(
        max_tokens=10000,
        timeout_sec=30
    ),
    verification=[
        VerificationRule(
            name="file_written",
            rule="file_exists == true"
        ),
        VerificationRule(
            name="non_empty",
            rule="file_size > 0"
        )
    ],
    failure_modes=["permission_denied", "disk_full", "invalid_path"]
)

DATA_ANALYSIS_MANIFEST = SkillManifest(
    name="data_analysis",
    version="1.0",
    description="Analyzes dataset and produces insights + visualizations",
    category=SkillCategory.analysis,
    agent_roles=["Researcher", "Analyst"],
    inputs=SkillInput(
        schema_name="DataAnalysisQuery",
        required=["data_path"],
        optional=["analysis_type", "visualizations"]
    ),
    outputs=SkillOutput(
        artifact_type=ArtifactType.REPORT,
        schema_name="DataAnalysisReport",
        reusable=True
    ),
    produces=[
        ArtifactProduced(
            type=ArtifactType.FILE,
            store="file",
            format="markdown",
            path_template="results/{goal_id}/analysis.md",
            tags=["data", "analysis"]
        ),
        ArtifactProduced(
            type=ArtifactType.DATASET,
            store="file",
            format="csv",
            path_template="results/{goal_id}/processed.csv",
            tags=["data", "processed"]
        )
    ],
    constraints=SkillConstraint(
        max_tokens=6000,
        timeout_sec=90
    ),
    verification=[
        VerificationRule(
            name="dataset_processed",
            rule="processed_rows > 0"
        ),
        VerificationRule(
            name="has_insights",
            rule="insights_count >= 1"
        )
    ],
    failure_modes=["file_not_found", "invalid_format", "timeout"]
)

# Registry of built-in manifests
BUILTIN_MANIFESTS = {
    "web_research": WEB_RESEARCH_MANIFEST,
    "code_analysis": CODE_ANALYSIS_MANIFEST,
    "file_write": FILE_WRITE_MANIFEST,
    "data_analysis": DATA_ANALYSIS_MANIFEST,
}
