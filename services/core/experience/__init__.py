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
from experience.execution_envelope import (
    ExecutionEnvelope,
    ExecutionEnvelopeStore
)
from experience.learning_pipeline import (
    LearningPipeline,
    get_learning_pipeline
)

# Enforcement & Adapter
from experience.enforcement_config import (
    EnforcementMode,
    EnforcementConfig,
    get_enforcement_config,
    get_enforcement_metrics,
    reload_enforcement_config
)
from experience.execution_adapter import (
    ExecutionAdapter,
    ExecutionContext,
    LegacyExecutionError,
    get_execution_adapter
)

# Execution Service (central execution path)
from experience.execution_service import (
    ExecutionService,
    ExecutionRequest,
    get_execution_service,
    execute_skill_legacy
)

# Registry snapshots for versioned replay
from experience.registry_snapshot import (
    RegistrySnapshot,
    RegistrySnapshotStore,
    SkillSnapshot,
    get_registry_snapshot_store,
    create_registry_snapshot
)

# Replay engine (safe, no side-effects)
from experience.replay_engine import (
    ReplayEngine,
    ReplayResult,
    get_replay_engine
)

# Promotion gates (no auto-promotion)
from experience.promotion_gates import (
    PromotionGates,
    PromotionStatus,
    PromotionGateResult,
    get_promotion_gates,
    safe_promote
)

# Immutable feature extraction (deterministic)
from experience.feature_extraction import (
    FeatureExtractor,
    FeatureVector,
    CanonicalFeaturePayload,
    FeatureExtractionMetadata,
    FEATURE_SCHEMA_VERSION,
    extract_features
)

# Frozen policy snapshot (read-only for replay)
from experience.policy_snapshot import (
    FrozenPolicySnapshot,
    SkillArmSnapshot,
    PolicySnapshotStore,
    get_policy_snapshot_store,
    create_policy_snapshot,
    get_frozen_policy_for_replay
)

# Execution lineage graph
from experience.execution_lineage import (
    ExecutionLineageGraph,
    ExecutionNode,
    EdgeType,
    get_lineage_graph,
    create_execution_node,
    get_execution_chain
)

# Causal edges (separated from nodes)
from experience.causal_edge import (
    ExecutionEdge,
    ExecutionEdgeStore,
    get_edge_store
)

# Replay bundle (complete frozen state)
from experience.replay_bundle import (
    ReplayBundle,
    ReplayBundleStore,
    CandidateSetSnapshot,
    EnforcementSnapshot,
    LineageSnapshot,
    get_bundle_store,
    create_replay_bundle
)

# Multi-objective evaluation vector
from experience.evaluation_vector import (
    EvaluationVector,
    EvaluationRecord,
    CapabilityEvaluationStore,
    get_evaluation_store
)

# Replay dataset builder
from experience.replay_dataset import (
    ReplayDatasetEntry,
    ReplayDatasetBuilder,
    OfflineEvaluator,
    get_dataset_builder
)

# Environment snapshot & Execution contract (replay consistency)
from experience.environment_snapshot import (
    EnvironmentSnapshot,
    ExecutionContract,
    CandidateEvaluation,
    capture_environment,
    get_environment_snapshot,
    get_contract_store,
    get_contract
)

# Unified execution graph (nodes + edges together)
from experience.execution_graph import (
    ExecutionGraph,
    ExecutionNodeData,
    get_execution_graph
)

# Execution journal (append-only event stream)
from experience.execution_journal import (
    ExecutionJournal,
    ExecutionEvent,
    EventType,
    CounterfactualEntry,
    CounterfactualStore,
    get_execution_journal,
    get_counterfactual_store,
    record_execution_event
)

# Temporal metrics (EMA, variance, trend, confidence)
from experience.temporal_metrics import (
    TemporalMetric,
    TemporalMetricsStore,
    get_temporal_store,
    record_temporal_metric
)

# Runtime state machine (formal lifecycle)
from experience.runtime_state import (
    ExecutionStateMachine,
    ExecutionState,
    VALID_TRANSITIONS,
    get_state_store,
    get_state_machine
)

# Decision boundary & Intent (cognitive causality)
from experience.decision_boundary import (
    DecisionBoundarySnapshot,
    ExecutionIntent,
    DecisionBoundaryStore,
    IntentStore,
    get_boundary_store,
    get_intent_store,
    record_decision_boundary,
    create_intent
)

# Unified Cognitive Transaction (single causal contract)
from experience.cognitive_transaction import (
    CognitiveTransaction,
    CognitiveTransactionBuilder,
    CognitiveTransactionStore,
    TransactionPhase,
    get_tx_store,
    create_cognitive_transaction
)

# Cognitive Commit Protocol (active transactional semantics)
from experience.commit_protocol import (
    CognitiveCommitProtocol,
    CausalClock,
    SideEffect,
    InvariantEngine,
    begin_cognitive_transaction,
    get_active_protocol
)

# Semantic Memory Region (isolated cognitive namespaces with WAL)
from experience.semantic_memory_region import (
    MemoryRegion,
    Belief,
    BeliefDelta,
    WriteAheadLog,
    SemanticMemoryRegion,
    SemanticMemoryManager,
    get_memory_manager,
    get_region
)

# Transactional Overlay Memory (ACID semantics)
from experience.transactional_overlay import (
    TransactionState,
    IsolationLevel,
    Mutation,
    TransactionOverlay,
    WALRecord,
    TransactionalOverlay,
    TransactionalMemoryManager,
    get_transactional_manager,
    begin_isolated_transaction
)

# Semantic Diff Engine (semantic evolution tracking)
from experience.semantic_diff_engine import (
    SemanticChangeType,
    ChangeSeverity,
    CausalAttribution,
    SemanticMutation,
    SemanticDiff,
    BeliefComparator,
    ContradictionDetector,
    CausalAttributionEngine,
    ConfidenceDriftAnalyzer,
    PolicyImpactEstimator,
    ReflectionTriggerDetector,
    SemanticDiffEngine,
    get_semantic_diff_engine,
    compute_semantic_diff
)

# Causal Belief Hypergraph (rich causal semantics)
from experience.causal_hypergraph import (
    EdgeWeight,
    AttractorState,
    CausalEdge,
    BeliefNode,
    HypergraphSnapshot,
    TemporalDecay,
    CounterfactualBranch,
    PolicyMediation,
    AttractorAnalyzer,
    CausalHypergraph,
    get_causal_hypergraph,
    create_belief_hypergraph
)

# Contradiction Memory (persistent contradictions)
from experience.contradiction_memory import (
    ResolutionStatus,
    ContradictionType,
    ResolutionAttempt,
    ContradictionEpisode,
    ResolutionEngine,
    ContradictionMemory,
    get_contradiction_memory
)
from experience.contradiction_memory import ContradictionDetector as ContradictionMemoryDetector

# Unified Epistemic State (canonical world-state)
from experience.unified_epistemic_state import (
    EpistemicClock,
    BeliefState,
    ConstraintState,
    ContradictionState,
    CausalEdgeState,
    StateDiff,
    InvariantCheck,
    UnifiedEpistemicState,
    UnifiedEpistemicStateManager,
    get_ues_manager,
    get_current_state
)

# Constraint Graph (belief decomposition for proper contradiction detection)
from experience.constraint_graph import (
    ConstraintType,
    ConstraintOperator,
    Constraint,
    ConfidenceRegion,
    BeliefConstraints,
    ConstraintExtractor,
    ConstraintIncompatibilityDetector,
    ConstraintGraph,
    get_constraint_graph
)

# Contradiction Memory (persistent contradictions)
from experience.contradiction_memory import (
    ResolutionStatus,
    ContradictionType,
    ResolutionAttempt,
    ContradictionEpisode,
    ResolutionEngine,
    ContradictionMemory,
    get_contradiction_memory
)
from experience.contradiction_memory import ContradictionDetector as ContradictionMemoryDetector

# Unified Epistemic State (canonical world-state)
from experience.unified_epistemic_state import (
    EpistemicClock,
    BeliefState,
    ConstraintState,
    ContradictionState,
    CausalEdgeState,
    StateDiff,
    InvariantCheck,
    UnifiedEpistemicState,
    UnifiedEpistemicStateManager,
    get_ues_manager,
    get_current_state
)

# Reflection Scheduler (budget-aware scheduling)
from experience.reflection_scheduler import (
    ReflectionPriority,
    ReflectionType,
    ReflectionTask,
    ReflectionBudget,
    ReflectionMetrics,
    ReflectionScheduler,
    get_reflection_scheduler,
    should_reflect
)

# Reflection Kernel (speculative sandbox + commit layer)
from experience.reflection_kernel import (
    MutationOperation,
    ReflectionDepth,
    MutationOperationDetail,
    EpistemicMutationProposal,
    ReflectionAnalysis,
    ReflectionSandbox,
    ReflectionCommitLayer,
    ReflectionKernel,
    get_reflection_kernel,
    init_reflection_kernel
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
    
    # Execution envelope
    "ExecutionEnvelope",
    "ExecutionEnvelopeStore",
    
    # Learning pipeline
    "LearningPipeline",
    "get_learning_pipeline",
    
    # Enforcement
    "EnforcementMode",
    "EnforcementConfig",
    "get_enforcement_config",
    "get_enforcement_metrics",
    "reload_enforcement_config",
    
    # Execution adapter
    "ExecutionAdapter",
    "ExecutionContext",
    "LegacyExecutionError",
    "get_execution_adapter",
    
    # Execution service (central execution path)
    "ExecutionService",
    "ExecutionRequest",
    "get_execution_service",
    "execute_skill_legacy",
    
    # Registry snapshots
    "RegistrySnapshot",
    "RegistrySnapshotStore",
    "SkillSnapshot",
    "get_registry_snapshot_store",
    "create_registry_snapshot",
    
    # Replay engine
    "ReplayEngine",
    "ReplayResult",
    "get_replay_engine",
    
    # Promotion gates
    "PromotionGates",
    "PromotionStatus",
    "PromotionGateResult",
    "get_promotion_gates",
    "safe_promote",
    
# Feature extraction
    "FeatureExtractor",
    "FeatureVector",
    "CanonicalFeaturePayload",
    "FeatureExtractionMetadata",
    "FEATURE_SCHEMA_VERSION",
    "extract_features",
    
    # Policy snapshot (read-only for replay)
    "FrozenPolicySnapshot",
    "SkillArmSnapshot",
    "PolicySnapshotStore",
    "get_policy_snapshot_store",
    "create_policy_snapshot",
    "get_frozen_policy_for_replay",
    
    # Execution lineage
    "ExecutionLineageGraph",
    "ExecutionNode",
    "EdgeType",
    "get_lineage_graph",
    "create_execution_node",
    "get_execution_chain",
    
    # Causal edges
    "ExecutionEdge",
    "ExecutionEdgeStore",
    "get_edge_store",
    
    # Replay bundle
    "ReplayBundle",
    "ReplayBundleStore",
    "CandidateSetSnapshot",
    "EnforcementSnapshot",
    "LineageSnapshot",
    "get_bundle_store",
    "create_replay_bundle",
    
    # Evaluation vector (multi-objective)
    "EvaluationVector",
    "EvaluationRecord",
    "CapabilityEvaluationStore",
    "get_evaluation_store",
    
    # Replay dataset builder
    "ReplayDatasetEntry",
    "ReplayDatasetBuilder",
    "OfflineEvaluator",
    "get_dataset_builder",
    
    # Environment snapshot & execution contract
    "EnvironmentSnapshot",
    "ExecutionContract",
    "CandidateEvaluation",
    "capture_environment",
    "get_environment_snapshot",
    "get_contract_store",
    "get_contract",
    
    # Unified execution graph
    "ExecutionGraph",
    "ExecutionNodeData",
    "get_execution_graph",
    
    # Execution journal
    "ExecutionJournal",
    "ExecutionEvent",
    "EventType",
    "CounterfactualEntry",
    "CounterfactualStore",
    "get_execution_journal",
    "get_counterfactual_store",
    "record_execution_event",
    
    # Temporal metrics
    "TemporalMetric",
    "TemporalMetricsStore",
    "get_temporal_store",
    "record_temporal_metric",
    
    # Runtime state machine
    "ExecutionStateMachine",
    "ExecutionState",
    "VALID_TRANSITIONS",
    "get_state_store",
    "get_state_machine",
    
    # Decision boundary & Intent
    "DecisionBoundarySnapshot",
    "ExecutionIntent",
    "DecisionBoundaryStore",
    "IntentStore",
    "get_boundary_store",
    "get_intent_store",
    "record_decision_boundary",
    "create_intent",
    
    # Cognitive transaction (unified contract)
    "CognitiveTransaction",
    "CognitiveTransactionBuilder",
    "CognitiveTransactionStore",
    "TransactionPhase",
    "get_tx_store",
    "create_cognitive_transaction",
    
    # Cognitive commit protocol (active transaction)
    "CognitiveCommitProtocol",
    "CausalClock",
    "SideEffect",
    "InvariantEngine",
    "begin_cognitive_transaction",
    "get_active_protocol",
    
    # Semantic memory region (isolated namespaces with WAL)
    "MemoryRegion",
    "Belief",
    "BeliefDelta",
    "WriteAheadLog",
    "SemanticMemoryRegion",
    "SemanticMemoryManager",
    "get_memory_manager",
    "get_region",
    
    # Transactional overlay memory (ACID semantics)
    "TransactionState",
    "IsolationLevel",
    "Mutation",
    "TransactionOverlay",
    "WALRecord",
    "TransactionalOverlay",
    "TransactionalMemoryManager",
    "get_transactional_manager",
    "begin_isolated_transaction",
    
    # Semantic diff engine (semantic evolution tracking)
    "SemanticChangeType",
    "ChangeSeverity",
    "CausalAttribution",
    "SemanticMutation",
    "SemanticDiff",
    "BeliefComparator",
    "ContradictionDetector",
    "CausalAttributionEngine",
    "ConfidenceDriftAnalyzer",
    "PolicyImpactEstimator",
    "ReflectionTriggerDetector",
    "SemanticDiffEngine",
    "get_semantic_diff_engine",
    "compute_semantic_diff",
    
    # Causal belief hypergraph (rich causal semantics)
    "EdgeWeight",
    "AttractorState",
    "CausalEdge",
    "BeliefNode",
    "HypergraphSnapshot",
    "TemporalDecay",
    "CounterfactualBranch",
    "PolicyMediation",
    "AttractorAnalyzer",
    "CausalHypergraph",
    "get_causal_hypergraph",
    "create_belief_hypergraph",
    
    # Contradiction memory (persistent contradictions)
    "ResolutionStatus",
    "ContradictionType",
    "ResolutionAttempt",
    "ContradictionEpisode",
    "ContradictionMemoryDetector",
    "ResolutionEngine",
    "ContradictionMemory",
    "get_contradiction_memory",
    
    # Constraint graph (belief decomposition)
    "ConstraintType",
    "ConstraintOperator",
    "Constraint",
    "ConfidenceRegion",
    "BeliefConstraints",
    "ConstraintExtractor",
    "ConstraintIncompatibilityDetector",
    "ConstraintGraph",
    "get_constraint_graph",
    
    # Unified epistemic state (canonical world-state)
    "EpistemicClock",
    "BeliefState",
    "ConstraintState",
    "ContradictionState",
    "CausalEdgeState",
    "StateDiff",
    "InvariantCheck",
    "UnifiedEpistemicState",
    "UnifiedEpistemicStateManager",
    "get_ues_manager",
    "get_current_state",
    
    # Reflection scheduler (budget-aware scheduling)
    "ReflectionPriority",
    "ReflectionType",
    "ReflectionTask",
    "ReflectionBudget",
    "ReflectionMetrics",
    "ReflectionScheduler",
    "get_reflection_scheduler",
    "should_reflect",
    
    # Reflection kernel (speculative sandbox + commit layer)
    "MutationOperation",
    "ReflectionDepth",
    "MutationOperationDetail",
    "EpistemicMutationProposal",
    "ReflectionAnalysis",
    "ReflectionSandbox",
    "ReflectionCommitLayer",
    "ReflectionKernel",
    "get_reflection_kernel",
    "init_reflection_kernel",
]
