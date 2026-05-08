"""
Experience Engine - Core Learning Loop for AI-OS

This module turns executions into learning.

Architecture:
    Execution → Experience → SkillStats → Better Skill Selection

Components:
- Experience: Single execution record
- SkillStats: Aggregated skill performance
- ExperienceEngine: Orchestrates recording and learning
- SkillStatsService: Updates statistics
- SkillStatsCache: In-memory cache for real-time access
- Repositories: Database operations

Usage:
    from experience import experience_engine, get_skill_stats_sync

    # Record experience
    await experience_engine.record_experience(
        session=session,
        goal_id=goal.id,
        task_type="web_search",
        skill_id="web.search",
        success=True,
        confidence=0.95,
        latency_ms=1234
    )

    # Get cached stats
    stats = await get_skill_stats_sync()
"""

from experience.experience_models import Experience, SkillStats
from experience.experience_engine import ExperienceEngine, experience_engine
from experience.experience_repository import ExperienceRepository, SkillStatsRepository
from experience.skill_stats_service import SkillStatsService
from experience.skill_stats_cache import SkillStatsCache, skill_stats_cache, get_skill_stats_sync
from experience.legacy_adapter import LegacyExperienceAdapter, legacy_experience_adapter
from experience.gaussian_skill_selector import GaussianSkillSelector, GaussianTracker, SkillResult
from experience.reward_model import compute_reward, normalize_for_gaussian, RewardComponents
from experience.environment_context import (
    EnvironmentContext, 
    inject_rate_limit, 
    inject_network_issue, 
    clear_all_conditions
)

# New bandit learning components
from experience.trajectory_dataset import (
    TRACE_SCHEMA_VERSION,
    TraceContext,
    TraceSchemaValidator,
    LearningSample
)
from experience.dataset_extractor import (
    TrajectoryDatasetExtractor,
    RewardNormalizer
)
from experience.thompson_sampling import (
    ThompsonSamplingBandit,
    get_bandit,
    SkillArm
)
from experience.learning_event import (
    LearningEvent,
    LearningEventStore,
    SCHEMA_VERSION,
    EventType,
    PolicyVersion
)
from experience.skill_registry import (
    SkillRegistry,
    ResolvedSkill,
    get_skill_registry,
    resolve_skill,
    SkillRegistryError
)
from experience.shadow_evaluator import (
    ShadowEvaluator,
    get_shadow_evaluator
)

__all__ = [
    # Existing
    "Experience",
    "SkillStats",
    "ExperienceEngine",
    "experience_engine",
    "ExperienceRepository",
    "SkillStatsRepository",
    "SkillStatsService",
    "SkillStatsCache",
    "skill_stats_cache",
    "get_skill_stats_sync",
    "LegacyExperienceAdapter",
    "legacy_experience_adapter",
    "GaussianSkillSelector",
    "GaussianTracker",
    "SkillResult",
    "compute_reward",
    "normalize_for_gaussian",
    "RewardComponents",
    "EnvironmentContext",
    "inject_rate_limit",
    "inject_network_issue",
    "clear_all_conditions",
    
    # New bandit learning
    "TRACE_SCHEMA_VERSION",
    "TraceContext",
    "TraceSchemaValidator",
    "LearningSample",
    "TrajectoryDatasetExtractor",
    "RewardNormalizer",
    "ThompsonSamplingBandit",
    "get_bandit",
    "SkillArm",
    "ShadowEvaluator",
    "get_shadow_evaluator",
    
    # Learning events
    "LearningEvent",
    "LearningEventStore",
    "SCHEMA_VERSION",
    "EventType",
    
    # Skill registry
    "SkillRegistry",
    "ResolvedSkill",
    "get_skill_registry",
    "resolve_skill",
    "SkillRegistryError",
]
