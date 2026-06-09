"""
AI-OS Cognitive OS - Latent Dynamics Layer (Phase 11-12.5)

Semantic Causality + Behavioral Dynamics

Phase 11 - Latent Causal Dynamics:
- LatentStateSpace - Embedding-based state representation
- CausalAttributionEngine - Semantic cause extraction
- SemanticCompressor - Macro-events, trajectory chunks
- CounterfactualReasoner - What-if analysis

Phase 12.5 - Behavioral Dynamics Engine:
- TrajectoryMemoryGraph - Behavioral continuum (trajectories connected)
- MotifTransitionMatrix - Attractor flow dynamics
- InterpretationLayer - Symbolic projection (NOT core!)
- TrajectoryRollout - Future trajectory manifold
- BehavioralFlowField - Continuous flow approximation

Key architecture:
- Clean core: geometric, no labels
- InterpretationLayer: symbolic projection on top
- Future manifold: possible trajectories, not single prediction

Usage:
    from ai_os.cognitive_os.latent_dynamics import LatentDynamicsEngine
    
    engine = LatentDynamicsEngine(cognitive_os)
    
    # Process state transition with semantic causality
    transition = engine.process_transition(from_state, to_state, action, outcome)
    
    # Get dominant causes
    causes = engine.get_dominant_causes()
    
    # Behavioral dynamics (Phase 12.5)
    dynamics = engine.get_behavioral_dynamics()
    
    # Future prediction
    predictions = engine.predict_future(state_id, actions)
    
    # Counterfactual analysis
    counterfactual = engine.reason_counterfactual(state_id, actual, alternative)
"""
from .latent_space import (
    SymbolicState,
    LatentState,
    LatentStateEncoder,
    LatentStateSpace,
    TransitionModel,
    create_latent_space,
)
from .causal_engine import (
    LatentCause,
    CausalEdge,
    SemanticTransition,
    CauseDetector,
    CausalAttributionEngine,
    CounterfactualReasoner,
    create_causal_attribution_engine,
)
from .semantic_compression import (
    BehaviorMotif,
    TrajectoryChunk,
    SemanticCompressor,
    CausalConfidence,
    create_semantic_compressor,
)
from .motif_discovery import (
    TrajectoryEmbedding,
    LearnedMotif,
    TrajectoryEncoder,
    MotifClusterer,
    AttractorDetector,
    MotifDiscoveryEngine,
    create_motif_discovery_engine,
)
from .trajectory_graph import (
    TrajectoryNode,
    MotifState,
    TrajectoryMemoryGraph,
    create_trajectory_graph,
)
from .transition_matrix import (
    TransitionProbability,
    MotifFlowStats,
    MotifTransitionMatrix,
    create_transition_matrix,
)
from .interpretation_layer import (
    MotifInterpretation,
    TrajectoryInterpretation,
    MotifInterpreter,
    TrajectoryInterpreter,
    InterpretationLayer,
    create_interpretation_layer,
)
from .trajectory_rollout import (
    FutureState,
    TrajectoryRollout,
    RolloutPlan,
    TrajectoryRollouter,
    BehavioralFlowField,
    create_trajectory_rollouter,
)
from .advanced_clustering import (
    ClusterConfig,
    AdvancedClusterer,
    DynamicMotifTracker,
    create_advanced_clusterer,
    create_dynamic_tracker,
)
from .behavioral_dynamics import (
    BehavioralDynamicsState,
    BehavioralDynamicsEngine,
    get_behavioral_dynamics_engine,
    reset_behavioral_dynamics_engine,
)
from .energy_landscape import (
    EnergyPoint,
    AttractorBasin,
    EnergyField,
    EnergyLandscape,
    create_energy_landscape,
)
from .phase_space import (
    PhaseState,
    PhaseTrajectory,
    PhaseSpace,
    create_phase_space,
)
from .latent_dynamics_model import (
    PredictionResult,
    RolloutPrediction,
    LatentDynamicsModel,
    create_latent_dynamics_model,
)
from .learned_energy_field import (
    EnergyStats,
    LearnedEnergyField,
    ActiveInferenceEngine,
    create_learned_energy_field,
    create_active_inference_engine,
)
from .continuous_dynamics import (
    ContinuousDynamicsState,
    ContinuousLatentDynamics,
    create_continuous_latent_dynamics,
)
from .unified_dynamics import (
    UnifiedState,
    UnifiedTrajectory,
    UnifiedCognitiveDynamics,
    create_unified_dynamics,
)
from .riemannian_manifold import (
    RiemannianMetric,
    SingleEnergyFunctional,
    GeodesicState,
    RiemannianCognitiveManifold,
    create_riemannian_manifold,
)
from .variational_inference import (
    KernelDensityEstimator,
    ProbabilisticEnergy,
    RiemannianMetricFull,
    GeodesicStateVariational,
    VariationalGeometricInference,
    create_variational_inference,
)
from .information_geometry import (
    GenerativeLatentModel,
    FisherRaoMetric,
    NaturalGradientFlow,
    InformationGeometrySystem,
    create_information_geometry_system,
)
from .engine import LatentDynamicsEngine, get_latent_dynamics_engine

__all__ = [
    # Latent Space
    "SymbolicState",
    "LatentState",
    "LatentStateEncoder",
    "LatentStateSpace",
    "TransitionModel",
    "create_latent_space",
    
    # Causal Engine
    "LatentCause",
    "CausalEdge",
    "SemanticTransition",
    "CauseDetector",
    "CausalAttributionEngine",
    "CounterfactualReasoner",
    "create_causal_attribution_engine",
    
    # Semantic Compression
    "BehaviorMotif",
    "TrajectoryChunk",
    "SemanticCompressor",
    "CausalConfidence",
    "create_semantic_compressor",
    
    # Motif Discovery
    "TrajectoryEmbedding",
    "LearnedMotif",
    "TrajectoryEncoder",
    "MotifClusterer",
    "AttractorDetector",
    "MotifDiscoveryEngine",
    "create_motif_discovery_engine",
    
    # Trajectory Graph
    "TrajectoryNode",
    "MotifState",
    "TrajectoryMemoryGraph",
    "create_trajectory_graph",
    
    # Transition Matrix
    "TransitionProbability",
    "MotifFlowStats",
    "MotifTransitionMatrix",
    "create_transition_matrix",
    
    # Interpretation Layer
    "MotifInterpretation",
    "TrajectoryInterpretation",
    "MotifInterpreter",
    "TrajectoryInterpreter",
    "InterpretationLayer",
    "create_interpretation_layer",
    
    # Trajectory Rollout
    "FutureState",
    "TrajectoryRollout",
    "RolloutPlan",
    "TrajectoryRollouter",
    "BehavioralFlowField",
    "create_trajectory_rollouter",
    
    # Advanced Clustering
    "ClusterConfig",
    "AdvancedClusterer",
    "DynamicMotifTracker",
    "create_advanced_clusterer",
    "create_dynamic_tracker",
    
    # Behavioral Dynamics
    "BehavioralDynamicsState",
    "BehavioralDynamicsEngine",
    "get_behavioral_dynamics_engine",
    "reset_behavioral_dynamics_engine",
    
    # Energy Landscape
    "EnergyPoint",
    "AttractorBasin",
    "EnergyField",
    "EnergyLandscape",
    "create_energy_landscape",
    
    # Phase Space
    "PhaseState",
    "PhaseTrajectory",
    "PhaseSpace",
    "create_phase_space",
    
    # Latent Dynamics Model
    "PredictionResult",
    "RolloutPrediction",
    "LatentDynamicsModel",
    "create_latent_dynamics_model",
    
    # Learned Energy Field
    "EnergyStats",
    "LearnedEnergyField",
    "ActiveInferenceEngine",
    "create_learned_energy_field",
    "create_active_inference_engine",
    
    # Continuous Dynamics
    "ContinuousDynamicsState",
    "ContinuousLatentDynamics",
    "create_continuous_latent_dynamics",
    
    # Unified Dynamics (Phase 14)
    "UnifiedState",
    "UnifiedTrajectory",
    "UnifiedCognitiveDynamics",
    "create_unified_dynamics",
    
    # Riemannian Manifold (Phase 14.5)
    "RiemannianMetric",
    "SingleEnergyFunctional",
    "GeodesicState",
    "RiemannianCognitiveManifold",
    "create_riemannian_manifold",
    
    # Variational Inference (Phase 15)
    "KernelDensityEstimator",
    "ProbabilisticEnergy",
    "RiemannianMetricFull",
    "GeodesicStateVariational",
    "VariationalGeometricInference",
    "create_variational_inference",
    
    # Information Geometry (Phase 16)
    "GenerativeLatentModel",
    "FisherRaoMetric",
    "NaturalGradientFlow",
    "InformationGeometrySystem",
    "create_information_geometry_system",
    
    # Main Engine
    "LatentDynamicsEngine",
    "get_latent_dynamics_engine",
]