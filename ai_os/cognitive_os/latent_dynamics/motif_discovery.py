"""
Learned Motif Discovery - Emergent Behavior Primitives

Replaces rule-based motif classification with learned trajectory clustering.

Phase 11 (rule-based):
    if event_type == "simulation_branch": return "exploration"

Phase 12 (learned):
    trajectory_embedding → clustering → attractor → learned_motif

Key insight: Motifs are attractors in behavioral space, not rule labels.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEmbedding:
    """
    Embedding of a trajectory (not just state).
    
    Encodes:
    - Dynamics (how state changes)
    - Intent (what goal pursued)
    - Shape (trajectory pattern)
    - Temporal evolution
    """
    trajectory_id: str
    timestamp: datetime
    
    # Raw sequence encoded
    state_sequence: List[List[float]]  # States over time
    action_sequence: List[str]  # Actions taken
    
    # Compressed embedding
    embedding: List[float]  # Dense vector
    dimension: int = 0
    
    # Trajectory properties (computed from sequence)
    start_state: List[float] = field(default_factory=list)
    end_state: List[float] = field(default_factory=list)
    
    total_reward: float = 0.0
    duration_ms: float = 0.0
    event_count: int = 0
    
    # Trajectory shape metrics
    curvature: float = 0.0  # How much trajectory curves
    directness: float = 0.0  # How direct vs exploratory
    volatility: float = 0.0  # How variable
    momentum: float = 0.0  # Average velocity
    
    def __post_init__(self):
        self.dimension = len(self.embedding)
    
    def distance_to(self, other: "TrajectoryEmbedding") -> float:
        """Euclidean distance between embeddings"""
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.embedding, other.embedding)
        ))
    
    def cosine_similarity(self, other: "TrajectoryEmbedding") -> float:
        """Cosine similarity between embeddings"""
        dot = sum(a * b for a, b in zip(self.embedding, other.embedding))
        norm1 = math.sqrt(sum(a ** 2 for a in self.embedding))
        norm2 = math.sqrt(sum(b ** 2 for b in other.embedding))
        return dot / (norm1 * norm2 + 1e-10)
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "timestamp": self.timestamp.isoformat(),
            "dimension": self.dimension,
            "total_reward": self.total_reward,
            "duration_ms": self.duration_ms,
            "event_count": self.event_count,
            "shape_metrics": {
                "curvature": self.curvature,
                "directness": self.directness,
                "volatility": self.volatility,
                "momentum": self.momentum,
            }
        }


@dataclass
class LearnedMotif:
    """
    Learned behavior primitive (attractor basin).
    
    CLEAN CORE - No symbolic labels!
    
    Motif = stable trajectory field in latent space
    
    NOT:
    - label: "exploration" (symbolic leakage)
    
    BUT:
    - centroid: geometric center
    - density: how many trajectories pass through
    - stability: how consistent over time
    - entropy: behavioral uncertainty
    - basin_radius: attractor basin size
    """
    cluster_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Attractor geometry (core!)
    prototype_embedding: List[float] = field(default_factory=list)
    
    # Trajectory membership
    trajectory_ids: List[str] = field(default_factory=list)
    
    # Computed properties (no labels!)
    density: float = 0.0
    stability: float = 0.0
    entropy: float = 0.0
    basin_radius: float = 0.0
    
    # Frequency
    frequency: int = 0
    avg_duration_ms: float = 0.0
    consistency_score: float = 0.0
    
    # Metastability
    self_loop_prob: float = 0.0
    escape_likelihood: float = 0.0
    decay_rate: float = 0.0
    
    # Temporal
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "density": round(self.density, 3),
            "stability": round(self.stability, 3),
            "entropy": round(self.entropy, 3),
            "basin_radius": round(self.basin_radius, 3),
            "frequency": self.frequency,
            "trajectory_count": len(self.trajectory_ids),
            "self_loop_prob": round(self.self_loop_prob, 3),
            "escape_likelihood": round(self.escape_likelihood, 3),
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
        }


class TrajectoryEncoder:
    """
    Encodes trajectory into dense embedding.
    
    Not just state → vector, but:
    - State sequence → dynamics encoding
    - Action sequence → intent encoding
    - Temporal evolution → shape encoding
    """
    
    EMBEDDING_DIMENSION = 32
    
    def __init__(self):
        self.feature_weights = self._init_weights()
        logger.info("trajectory_encoder_initialized", dimension=self.EMBEDDING_DIMENSION)
    
    def _init_weights(self) -> Dict[str, float]:
        return {
            "confidence": 1.0,
            "stress_level": 0.9,
            "action_readiness": 0.85,
            "arousal": 0.7,
            "valence": 0.7,
            "focus": 0.6,
            "exploration_tendency": 0.6,
            "task_complexity": 0.5,
        }
    
    def encode(
        self,
        state_sequence: List[Dict],
        action_sequence: List[str],
        outcomes: Optional[List[str]] = None
    ) -> TrajectoryEmbedding:
        """Encode trajectory into embedding"""
        from uuid import uuid4
        
        if not state_sequence:
            return TrajectoryEmbedding(
                trajectory_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                state_sequence=[],
                action_sequence=[],
                embedding=[0.0] * self.EMBEDDING_DIMENSION
            )
        
        # Extract state vectors
        state_vectors = [self._state_to_vector(s) for s in state_sequence]
        
        # Encode dynamics (how state changes)
        dynamics = self._encode_dynamics(state_vectors)
        
        # Encode shape (trajectory pattern)
        shape = self._encode_shape(state_vectors)
        
        # Encode intent (from actions)
        intent = self._encode_intent(action_sequence)
        
        # Encode outcomes (if provided)
        reward_encoding = [0.0] * 8
        if outcomes:
            success_rate = sum(1 for o in outcomes if o == "success") / len(outcomes)
            reward_encoding[0] = success_rate
            reward_encoding[1] = 1 - success_rate
        
        # Combine into embedding
        embedding = dynamics + shape + intent + reward_encoding
        
        # Compute shape metrics
        curvature = self._compute_curvature(state_vectors)
        directness = self._compute_directness(state_vectors)
        volatility = self._compute_volatility(state_vectors)
        momentum = self._compute_momentum(state_vectors)
        
        timestamps = [datetime.fromisoformat(s.get("timestamp", datetime.utcnow().isoformat())) 
                     for s in state_sequence]
        duration_ms = (max(timestamps) - min(timestamps)).total_seconds() * 1000 if len(timestamps) > 1 else 0
        
        total_reward = sum(
            {"success": 1.0, "partial": 0.5, "failure": 0.0}.get(o, 0.5) 
            for o in (outcomes or [])
        )
        
        return TrajectoryEmbedding(
            trajectory_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            state_sequence=state_vectors,
            action_sequence=action_sequence,
            embedding=embedding,
            start_state=state_vectors[0] if state_vectors else [],
            end_state=state_vectors[-1] if state_vectors else [],
            total_reward=total_reward,
            duration_ms=duration_ms,
            event_count=len(state_sequence),
            curvature=curvature,
            directness=directness,
            volatility=volatility,
            momentum=momentum,
        )
    
    def _state_to_vector(self, state: Dict) -> List[float]:
        """Convert state dict to vector"""
        weights = self.feature_weights
        raw = [
            state.get("confidence", 0.5),
            state.get("stress_level", 0.0),
            state.get("action_readiness", 0.5),
            state.get("arousal", 0.5),
            (state.get("valence", 0) + 1) / 2,
            state.get("focus", 0.5),
            state.get("exploration_tendency", 0.5),
            state.get("task_complexity", 0.5),
        ]
        w = [weights.get(k, 0.5) for k in ["confidence", "stress_level", "action_readiness", 
                                           "arousal", "valence", "focus", 
                                           "exploration_tendency", "task_complexity"]]
        return [v * w[i] for i, v in enumerate(raw)]
    
    def _encode_dynamics(self, states: List[List[float]]) -> List[float]:
        """Encode state dynamics (changes over time)"""
        if len(states) < 2:
            return [0.0] * 8
        
        # Compute deltas
        deltas = [
            sum((states[i+1][j] - states[i][j]) ** 2 for j in range(len(states[0])))
            for i in range(len(states) - 1)
        ]
        
        # Statistics of changes
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        max_delta = max(deltas) if deltas else 0
        
        # Trend (direction of change)
        if len(states) >= 3:
            first_half = sum(deltas[:len(deltas)//2]) / (len(deltas)//2)
            second_half = sum(deltas[len(deltas)//2:]) / (len(deltas) - len(deltas)//2)
            trend = 1.0 if second_half < first_half else -1.0
        else:
            trend = 0.0
        
        return [
            avg_delta,
            max_delta,
            sum(deltas) / math.sqrt(len(deltas)) if deltas else 0,
            trend,
            sum(1 for d in deltas if d > 0.1) / len(deltas) if deltas else 0,  # variability
            sum(1 for d in deltas if d < 0.05) / len(deltas) if deltas else 0,  # stability
            max(deltas) - min(deltas) if deltas else 0,  # range
            len(deltas) / 10,  # length factor
        ]
    
    def _encode_shape(self, states: List[List[float]]) -> List[float]:
        """Encode trajectory shape"""
        if len(states) < 2:
            return [0.0] * 8
        
        # Start to end distance
        start_end = math.sqrt(sum(
            (states[-1][i] - states[0][i]) ** 2 
            for i in range(len(states[0]))
        ))
        
        # Total path length
        path_length = sum(
            math.sqrt(sum((states[i+1][j] - states[i][j]) ** 2 
                        for j in range(len(states[0]))))
            for i in range(len(states) - 1)
        )
        
        # Efficiency (directness)
        efficiency = start_end / path_length if path_length > 0 else 1.0
        
        # Center of mass
        center = [
            sum(s[i] for s in states) / len(states)
            for i in range(len(states[0]))
        ]
        
        # Spread (variance from center)
        spread = math.sqrt(sum(
            sum((s[i] - center[i]) ** 2 for i in range(len(s))) / len(states)
            for s in states
        ))
        
        return [
            start_end,
            path_length,
            efficiency,
            spread,
            math.sqrt(sum(c ** 2 for c in center)),  # center magnitude
            start_end / len(states),  # per-step progress
            (path_length - start_end) / start_end if start_end > 0 else 0,  # detour
            1.0 / (1.0 + spread),  # compactness
        ]
    
    def _encode_intent(self, actions: List[str]) -> List[float]:
        """Encode action intent"""
        action_counts = defaultdict(int)
        for a in actions:
            action_counts[a] += 1
        
        action_types = ["execute", "decompose", "explore", "wait", "reconsider", "retry", "abort"]
        total = len(actions) or 1
        
        return [
            action_counts.get("execute", 0) / total,
            action_counts.get("decompose", 0) / total,
            action_counts.get("explore", 0) / total,
            action_counts.get("wait", 0) / total,
            action_counts.get("reconsider", 0) / total,
            action_counts.get("retry", 0) / total,
            action_counts.get("abort", 0) / total,
            sum(1 for a in actions if a in ["retry", "reconsider"]) / total,  # recovery tendency
        ]
    
    def _compute_curvature(self, states: List[List[float]]) -> float:
        """Compute trajectory curvature"""
        if len(states) < 3:
            return 0.0
        
        total_angle = 0.0
        for i in range(1, len(states) - 1):
            v1 = [states[i][j] - states[i-1][j] for j in range(len(states[0]))]
            v2 = [states[i+1][j] - states[i][j] for j in range(len(states[0]))]
            
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a ** 2 for a in v1)) + 1e-10
            norm2 = math.sqrt(sum(b ** 2 for b in v2)) + 1e-10
            
            cos_angle = dot / (norm1 * norm2)
            angle = math.acos(max(-1, min(1, cos_angle)))
            total_angle += angle
        
        return total_angle / (len(states) - 2) if len(states) > 2 else 0.0
    
    def _compute_directness(self, states: List[List[float]]) -> float:
        """Compute trajectory directness (1 = perfectly direct)"""
        if len(states) < 2:
            return 1.0
        
        start_end = math.sqrt(sum(
            (states[-1][i] - states[0][i]) ** 2
            for i in range(len(states[0]))
        ))
        
        path_length = sum(
            math.sqrt(sum((states[i+1][j] - states[i][j]) ** 2 
                        for j in range(len(states[0]))))
            for i in range(len(states) - 1)
        )
        
        return start_end / path_length if path_length > 0 else 1.0
    
    def _compute_volatility(self, states: List[List[float]]) -> float:
        """Compute state volatility"""
        if len(states) < 2:
            return 0.0
        
        deltas = []
        for i in range(1, len(states)):
            delta = math.sqrt(sum(
                (states[i][j] - states[i-1][j]) ** 2
                for j in range(len(states[0]))
            )
            deltas.append(delta)
        
        if not deltas:
            return 0.0
        
        mean = sum(deltas) / len(deltas)
        variance = sum((d - mean) ** 2 for d in deltas) / len(deltas)
        
        return math.sqrt(variance)
    
    def _compute_momentum(self, states: List[List[float]]) -> float:
        """Compute average trajectory momentum"""
        if len(states) < 2:
            return 0.0
        
        total_momentum = 0.0
        for i in range(1, len(states)):
            momentum = math.sqrt(sum(
                (states[i][j] - states[i-1][j]) ** 2
                for j in range(len(states[0]))
            ))
            total_momentum += momentum
        
        return total_momentum / (len(states) - 1)


class MotifClusterer:
    """
    Clusters trajectories to discover emergent motifs.
    
    Not rule-based labeling, but learned attractors.
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        min_samples: int = 3,
        distance_threshold: float = 0.5
    ):
        self.n_clusters = n_clusters
        self.min_samples = min_samples
        self.distance_threshold = distance_threshold
        
        self.clusters: Dict[str, LearnedMotif] = {}
        self.trajectories: Dict[str, TrajectoryEmbedding] = {}
        
        # Clustering state
        self.fitted = False
        self.cluster_centers: List[List[float]] = []
        
        logger.info("motif_clusterer_initialized", n_clusters=n_clusters)
    
    def fit(self, trajectories: List[TrajectoryEmbedding]) -> List[LearnedMotif]:
        """Fit clusterer on trajectories"""
        if len(trajectories) < self.min_samples:
            logger.warning("insufficient_trajectories", 
                         have=len(trajectories), 
                         need=self.min_samples)
            return []
        
        self.trajectories = {t.trajectory_id: t for t in trajectories}
        
        # K-means clustering
        self._kmeans_init(trajectories)
        
        # Assign trajectories to clusters
        self._assign_trajectories(trajectories)
        
        # Create learned motifs
        motifs = self._create_motifs()
        
        # Compute clean properties (no labels!)
        for motif in motifs:
            self._compute_motif_properties(motif)
        
        self.fitted = True
        
        logger.info("motif_clustering_complete", 
                   trajectories=len(trajectories), 
                   motifs=len(motifs))
        
        return motifs
    
    def _kmeans_init(self, trajectories: List[TrajectoryEmbedding]) -> None:
        """Initialize k-means clusters"""
        embeddings = [t.embedding for t in trajectories]
        
        # Simple k-means initialization
        self.cluster_centers = []
        
        # Pick initial centers (spaced out)
        step = max(1, len(embeddings) // self.n_clusters)
        for i in range(0, len(embeddings), step):
            if len(self.cluster_centers) < self.n_clusters:
                self.cluster_centers.append(embeddings[i])
        
        # Ensure we have enough centers
        while len(self.cluster_centers) < self.n_clusters:
            self.cluster_centers.append(embeddings[len(self.cluster_centers) % len(embeddings)])
        
        # Iterate to refine
        for _ in range(10):
            assignments = [[] for _ in range(len(self.cluster_centers))]
            
            for t in trajectories:
                distances = [
                    math.sqrt(sum((a - b) ** 2 
                                  for a, b in zip(t.embedding, c)))
                    for c in self.cluster_centers
                ]
                nearest = distances.index(min(distances))
                assignments[nearest].append(t.embedding)
            
            for i, group in enumerate(assignments):
                if group:
                    new_center = [
                        sum(g[j] for g in group) / len(group)
                        for j in range(len(self.cluster_centers[0]))
                    ]
                    self.cluster_centers[i] = new_center
    
    def _assign_trajectories(self, trajectories: List[TrajectoryEmbedding]) -> None:
        """Assign trajectories to nearest cluster"""
        cluster_trajectories: Dict[int, List[str]] = {i: [] for i in range(len(self.cluster_centers))}
        
        for t in trajectories:
            distances = [
                math.sqrt(sum((a - b) ** 2 
                              for a, b in zip(t.embedding, c)))
                for c in self.cluster_centers
            ]
            nearest = distances.index(min(distances))
            cluster_trajectories[nearest].append(t.trajectory_id)
        
        self.cluster_trajectories = cluster_trajectories
    
    def _compute_motif_properties(self, motif: LearnedMotif) -> None:
        """Compute clean motif properties (no symbolic labels)"""
        cluster_trajectories = [self.trajectories[tid] for tid in motif.trajectory_ids]
        
        # Density
        motif.density = len(motif.trajectory_ids) / max(1, len(self.trajectories))
        
        # Duration stats
        motif.avg_duration_ms = sum(t.duration_ms for t in cluster_trajectories) / len(cluster_trajectories)
        
        # Stability (inverse of variance in trajectories)
        if len(cluster_trajectories) > 1:
            centroid = motif.prototype_embedding
            variances = []
            for t in cluster_trajectories:
                dist = math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(t.embedding, centroid)
                ))
                variances.append(dist)
            motif.stability = max(0, 1 - sum(variances) / len(variances))
        
        # Entropy (uncertainty in trajectory endpoints)
        if len(cluster_trajectories) > 1:
            endpoints = [tuple(t.end_state) for t in cluster_trajectories]
            unique_endpoints = len(set(endpoints))
            motif.entropy = 1.0 - (unique_endpoints / len(cluster_trajectories))
        
        # Basin radius
        max_dist = max(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(t.embedding, motif.prototype_embedding)))
            for t in cluster_trajectories
        ) if cluster_trajectories else 0.5
        motif.basin_radius = min(1.0, max_dist)
    
    def _create_motifs(self) -> List[LearnedMotif]:
        """Create learned motifs from clusters (clean core - no labels)"""
        motifs = []
        
        for cluster_id, trajectory_ids in self.cluster_trajectories.items():
            if len(trajectory_ids) < self.min_samples:
                continue
            
            cluster_trajectories = [self.trajectories[tid] for tid in trajectory_ids]
            
            motif = LearnedMotif(
                cluster_id=f"motif_{cluster_id}",
                prototype_embedding=self.cluster_centers[cluster_id],
                trajectory_ids=trajectory_ids,
                frequency=len(trajectory_ids),
                consistency_score=self._compute_consistency(cluster_trajectories),
                first_seen=min(t.timestamp for t in cluster_trajectories),
                last_seen=max(t.timestamp for t in cluster_trajectories),
            )
            
            self.clusters[motif.cluster_id] = motif
            motifs.append(motif)
        
        return motifs
    
    def _compute_consistency(self, trajectories: List[TrajectoryEmbedding]) -> float:
        """Compute how consistent trajectories in cluster are"""
        if len(trajectories) < 2:
            return 1.0
        
        distances = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                distances.append(trajectories[i].distance_to(trajectories[j]))
        
        avg_distance = sum(distances) / len(distances) if distances else 0
        return max(0, 1 - avg_distance)
    
    def _infer_label(self, motif: LearnedMotif) -> Tuple[str, float]:
        """Infer human-readable label from motif properties"""
        # Analyze prototype
        prototype = motif.prototype_embedding
        
        # Shape analysis
        curvature = sum(prototype[8:16]) / 8  # Curvature features
        directness = sum(prototype[16:24]) / 8  # Directness features
        
        # Intent analysis
        intent_start = 24
        execute_ratio = prototype[intent_start]
        explore_ratio = prototype[intent_start + 2]
        retry_ratio = prototype[intent_start + 5]
        wait_ratio = prototype[intent_start + 3]
        
        # Determine label
        if retry_ratio > 0.3:
            label = "recovery_attempt"
            confidence = retry_ratio
        elif explore_ratio > 0.4:
            label = "exploration_focused"
            confidence = explore_ratio
        elif execute_ratio > 0.5:
            label = "goal_execution"
            confidence = execute_ratio
        elif wait_ratio > 0.3:
            label = "observation_pause"
            confidence = wait_ratio
        elif directness > 0.7:
            label = "direct_efficient"
            confidence = directness
        elif directness < 0.4:
            label = "exploratory_search"
            confidence = 1 - directness
        elif curvature > 0.5:
            label = "adaptive_navigation"
            confidence = min(1.0, curvature)
        else:
            label = "balanced_approach"
            confidence = 0.6
        
        return label, confidence
    
    def assign_new_trajectory(self, trajectory: TrajectoryEmbedding) -> str:
        """Assign new trajectory to existing motif"""
        if not self.fitted:
            return ""
        
        distances = [
            math.sqrt(sum((a - b) ** 2 
                          for a, b in zip(trajectory.embedding, c)))
            for c in self.cluster_centers
        ]
        
        min_distance = min(distances)
        nearest_cluster = distances.index(min_distance)
        
        if min_distance < self.distance_threshold:
            motif_id = f"motif_{nearest_cluster}"
            if motif_id in self.clusters:
                motif = self.clusters[motif_id]
                motif.trajectory_ids.append(trajectory.trajectory_id)
                motif.last_seen = datetime.utcnow()
                motif.frequency += 1
                return motif_id
        
        return ""
    
    def get_motif_prototype(self, motif_id: str) -> Optional[List[float]]:
        """Get prototype embedding for motif"""
        if motif_id in self.clusters:
            return self.clusters[motif_id].prototype_embedding
        return None
    
    def get_similar_motifs(self, motif_id: str, limit: int = 3) -> List[Tuple[str, float]]:
        """Find similar motifs by prototype distance"""
        if motif_id not in self.clusters:
            return []
        
        motif = self.clusters[motif_id]
        distances = []
        
        for other_id, other_motif in self.clusters.items():
            if other_id != motif_id:
                dist = math.sqrt(sum(
                    (a - b) ** 2 
                    for a, b in zip(motif.prototype_embedding, other_motif.prototype_embedding)
                ))
                distances.append((other_id, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:limit]


class AttractorDetector:
    """
    Detects attractors in behavioral space.
    
    Attractor = frequently visited region of behavioral space.
    """
    
    def __init__(self, density_threshold: int = 3, radius: float = 0.3):
        self.density_threshold = density_threshold
        self.radius = radius
        
        self.attractors: List[LearnedMotif] = []
        logger.info("attractor_detector_initialized")
    
    def detect_from_trajectories(
        self,
        trajectories: List[TrajectoryEmbedding]
    ) -> List[LearnedMotif]:
        """Detect attractors from trajectory density"""
        
        # Build density map
        density_map: Dict[str, List[str]] = defaultdict(list)
        
        for t in trajectories:
            cluster_key = self._get_cluster_key(t.embedding)
            density_map[cluster_key].append(t.trajectory_id)
        
        # Find high-density regions (attractors)
        attractors = []
        
        for cluster_key, trajectory_ids in density_map.items():
            if len(trajectory_ids) >= self.density_threshold:
                attractor_trajectories = [
                    t for t in trajectories 
                    if t.trajectory_id in trajectory_ids
                ]
                
                # Compute attractor center
                center = [
                    sum(t.embedding[i] for t in attractor_trajectories) / len(attractor_trajectories)
                    for i in range(len(attractor_trajectories[0].embedding))
                ]
                
                motif = LearnedMotif(
                    cluster_id=f"attractor_{len(attractors)}",
                    prototype_embedding=center,
                    trajectory_ids=trajectory_ids,
                    frequency=len(trajectory_ids),
                    consistency_score=len(trajectory_ids) / len(trajectories),
                    first_seen=min(t.timestamp for t in attractor_trajectories),
                    last_seen=max(t.timestamp for t in attractor_trajectories),
                )
                
                attractors.append(motif)
        
        self.attractors = attractors
        logger.info("attractors_detected", count=len(attractors))
        
        return attractors
    
    def _get_cluster_key(self, embedding: List[float], resolution: int = 10) -> str:
        """Quantize embedding to cluster key"""
        scale = [10.0] * len(embedding)
        quantized = [int(e * scale[i]) for i, e in enumerate(embedding)]
        return "_".join(map(str, quantized[:resolution]))
    
    def find_attractor_for_trajectory(
        self,
        trajectory: TrajectoryEmbedding
    ) -> Optional[str]:
        """Find which attractor a trajectory belongs to"""
        for attractor in self.attractors:
            distance = math.sqrt(sum(
                (a - b) ** 2 
                for a, b in zip(trajectory.embedding, attractor.prototype_embedding)
            ))
            if distance < self.radius:
                return attractor.cluster_id
        
        return None


class MotifDiscoveryEngine:
    """
    Main engine for learned motif discovery.
    
    Pipeline:
    1. Encode trajectories
    2. Cluster trajectories
    3. Detect attractors
    4. Form learned motifs
    """
    
    def __init__(
        self,
        n_clusters: int = 10,
        min_samples: int = 3,
        distance_threshold: float = 0.5
    ):
        self.encoder = TrajectoryEncoder()
        self.clusterer = MotifClusterer(
            n_clusters=n_clusters,
            min_samples=min_samples,
            distance_threshold=distance_threshold
        )
        self.attractor_detector = AttractorDetector()
        
        self.trained_motifs: List[LearnedMotif] = []
        self.all_trajectories: Dict[str, TrajectoryEmbedding] = {}
        
        logger.info("motif_discovery_engine_initialized")
    
    def add_trajectory(
        self,
        state_sequence: List[Dict],
        action_sequence: List[str],
        outcomes: Optional[List[str]] = None
    ) -> Tuple[TrajectoryEmbedding, str]:
        """Add and encode a trajectory"""
        trajectory = self.encoder.encode(state_sequence, action_sequence, outcomes)
        self.all_trajectories[trajectory.trajectory_id] = trajectory
        
        motif_id = ""
        if self.trained_motifs:
            motif_id = self.clusterer.assign_new_trajectory(trajectory)
        
        return trajectory, motif_id
    
    def train(self) -> List[LearnedMotif]:
        """Train motif discovery on all trajectories"""
        if len(self.all_trajectories) < 3:
            logger.warning("insufficient_trajectories_for_training",
                         have=len(self.all_trajectories))
            return []
        
        trajectories = list(self.all_trajectories.values())
        
        # Cluster-based motif discovery
        cluster_motifs = self.clusterer.fit(trajectories)
        
        # Attractor-based motif discovery
        attractor_motifs = self.attractor_detector.detect_from_trajectories(trajectories)
        
        # Combine (use attractors as they capture density better)
        self.trained_motifs = attractor_motifs
        
        # Compute clean properties (no labels!)
        for motif in self.trained_motifs:
            self.clusterer._compute_motif_properties(motif)
        
        logger.info("motif_discovery_training_complete",
                   trajectories=len(trajectories),
                   motifs=len(self.trained_motifs))
        
        return self.trained_motifs
    
    def get_motifs(self) -> List[LearnedMotif]:
        """Get all discovered motifs"""
        return self.trained_motifs
    
    def get_motif_by_id(self, motif_id: str) -> Optional[LearnedMotif]:
        """Get motif by ID"""
        for motif in self.trained_motifs:
            if motif.cluster_id == motif_id:
                return motif
        return None
    
    def find_similar_trajectories(
        self,
        trajectory_id: str,
        limit: int = 5
    ) -> List[Tuple[str, float]]:
        """Find trajectories similar to given"""
        if trajectory_id not in self.all_trajectories:
            return []
        
        trajectory = self.all_trajectories[trajectory_id]
        distances = []
        
        for tid, other in self.all_trajectories.items():
            if tid != trajectory_id:
                dist = trajectory.distance_to(other)
                distances.append((tid, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:limit]
    
    def get_statistics(self) -> Dict:
        """Get motif discovery statistics"""
        return {
            "total_trajectories": len(self.all_trajectories),
            "discovered_motifs": len(self.trained_motifs),
            "motif_details": [
                {
                    "id": m.cluster_id,
                    "label": m.inferred_label,
                    "frequency": m.frequency,
                    "consistency": m.consistency_score,
                }
                for m in self.trained_motifs
            ],
            "avg_trajectories_per_motif": (
                len(self.all_trajectories) / len(self.trained_motifs) 
                if self.trained_motifs else 0
            ),
        }


# Factory function
def create_motif_discovery_engine(
    n_clusters: int = 10,
    min_samples: int = 3
) -> MotifDiscoveryEngine:
    return MotifDiscoveryEngine(n_clusters=n_clusters, min_samples=min_samples)