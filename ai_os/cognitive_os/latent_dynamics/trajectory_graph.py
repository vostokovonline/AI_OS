"""
TrajectoryMemoryGraph - Behavioral Continuum

Replaces isolated trajectories with connected cognitive flow.

NOT:
- Trajectory → Trajectory (independent episodes)

BUT:
- Trajectory → Trajectory (connected by behavioral momentum)

Key insight: Cognition is not episodes, it's flowing fields.
"""
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryNode:
    """
    Trajectory as node in behavioral graph.
    
    NOT: isolated trajectory with centroid
    BUT: connected node with flow context
    """
    trajectory_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Core embedding (no labels in core!)
    embedding: List[float] = field(default_factory=list)
    dimension: int = 0
    
    # Behavioral geometry
    start_state: List[float] = field(default_factory=list)
    end_state: List[float] = field(default_factory=list)
    
    # Shape properties
    curvature: float = 0.0
    directness: float = 0.0
    volatility: float = 0.0
    momentum: float = 0.0
    
    # Flow connections
    previous_nodes: List[str] = field(default_factory=list)
    next_nodes: List[str] = field(default_factory=list)
    
    # Motif affiliation (attractor basin, not label)
    motif_id: Optional[str] = None
    motif_confidence: float = 0.0
    
    # Flow properties
    divergence_from_previous: float = 0.0
    convergence_towards_next: float = 0.0
    
    # Temporal
    duration_ms: float = 0.0
    event_count: int = 0
    
    def __post_init__(self):
        self.dimension = len(self.embedding)
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "timestamp": self.timestamp.isoformat(),
            "dimension": self.dimension,
            "motif_id": self.motif_id,
            "motif_confidence": self.motif_confidence,
            "previous_count": len(self.previous_nodes),
            "next_count": len(self.next_nodes),
            "divergence": self.divergence_from_previous,
            "convergence": self.convergence_towards_next,
            "duration_ms": self.duration_ms,
            "shape": {
                "curvature": self.curvature,
                "directness": self.directness,
                "volatility": self.volatility,
                "momentum": self.momentum,
            }
        }


@dataclass  
class MotifState:
    """
    Motif as attractor basin (not cluster with label).
    
    Clean core - no symbolic labels!
    
    Motif = stable trajectory field in latent space
    """
    motif_id: str = field(default_factory=lambda: str(uuid4()))
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    
    # Attractor geometry
    centroid: List[float] = field(default_factory=list)
    density: float = 0.0  # How many trajectories pass through
    stability: float = 0.0  # How stable over time
    
    # Trajectory membership
    trajectory_ids: List[str] = field(default_factory=list)
    
    # Flow dynamics (computed from graph)
    entropy: float = 0.0  # Behavioral uncertainty
    basin_radius: float = 0.0  # Attractor basin size
    
    # Transitions (computed from graph)
    outgoing_transitions: Dict[str, float] = field(default_factory=list)  # Will be Dict
    incoming_transitions: Dict[str, float] = field(default_factory=list)  # Will be Dict
    
    # Metastability
    lifetime_ms: float = 0.0  # How long motifs typically last
    decay_rate: float = 0.0  # How quickly this attractor fades
    
    def to_dict(self) -> Dict:
        return {
            "motif_id": self.motif_id,
            "density": self.density,
            "stability": self.stability,
            "entropy": self.entropy,
            "basin_radius": self.basin_radius,
            "trajectory_count": len(self.trajectory_ids),
            "outgoing_transitions": len(self.outgoing_transitions),
            "incoming_transitions": len(self.incoming_transitions),
            "lifetime_ms": self.lifetime_ms,
            "decay_rate": self.decay_rate,
        }


class TrajectoryMemoryGraph:
    """
    Memory graph connecting trajectories into behavioral flow.
    
    Key capabilities:
    - behavioral inertia detection
    - recurring loops detection
    - self-reinforcing cognition tracking
    - drift detection
    - recovery path identification
    """
    
    def __init__(self):
        self.nodes: Dict[str, TrajectoryNode] = {}
        self.motifs: Dict[str, MotifState] = {}
        self.transition_matrix: Dict[str, Dict[str, float]] = {}  # motif → motif → prob
        
        # Graph metrics
        self.total_trajectories: int = 0
        self.loop_count: int = 0
        self.drift_episodes: int = 0
        
        logger.info("trajectory_memory_graph_initialized")
    
    def add_trajectory(
        self,
        embedding: List[float],
        start_state: List[float],
        end_state: List[float],
        shape_metrics: Dict[str, float],
        motif_id: Optional[str] = None,
        duration_ms: float = 0.0,
        event_count: int = 0
    ) -> TrajectoryNode:
        """Add trajectory to graph"""
        node = TrajectoryNode(
            trajectory_id=str(uuid4()),
            timestamp=datetime.utcnow(),
            embedding=embedding,
            start_state=start_state,
            end_state=end_state,
            curvature=shape_metrics.get("curvature", 0.0),
            directness=shape_metrics.get("directness", 1.0),
            volatility=shape_metrics.get("volatility", 0.0),
            momentum=shape_metrics.get("momentum", 0.0),
            motif_id=motif_id,
            duration_ms=duration_ms,
            event_count=event_count
        )
        
        self._link_to_previous(node)
        self._update_next_references(node)
        self._compute_flow_properties(node)
        
        self.nodes[node.trajectory_id] = node
        self.total_trajectories += 1
        
        logger.debug("trajectory_added", 
                    trajectory_id=node.trajectory_id,
                    motif=motif_id)
        
        return node
    
    def _link_to_previous(self, node: TrajectoryNode) -> None:
        """Link to most recent trajectory"""
        if not self.nodes:
            return
        
        recent = self._get_most_recent_nodes(limit=3)
        for prev_node in recent:
            if prev_node.trajectory_id != node.trajectory_id:
                dist = self._compute_distance(node.embedding, prev_node.embedding)
                if dist < 0.5:  # Threshold for temporal connection
                    node.previous_nodes.append(prev_node.trajectory_id)
                    if node.trajectory_id not in prev_node.next_nodes:
                        prev_node.next_nodes.append(node.trajectory_id)
    
    def _update_next_references(self, node: TrajectoryNode) -> None:
        """Update previous nodes' next references"""
        for prev_id in node.previous_nodes:
            if prev_id in self.nodes:
                prev_node = self.nodes[prev_id]
                if node.trajectory_id not in prev_node.next_nodes:
                    prev_node.next_nodes.append(node.trajectory_id)
    
    def _compute_flow_properties(self, node: TrajectoryNode) -> None:
        """Compute divergence and convergence"""
        if node.previous_nodes:
            prev_node = self.nodes[node.previous_nodes[0]]
            node.divergence_from_previous = self._compute_distance(
                node.start_state, prev_node.end_state
            )
        
        if node.next_nodes:
            next_node = self.nodes[node.next_nodes[0]]
            node.convergence_towards_next = self._compute_distance(
                node.end_state, next_node.start_state
            )
    
    def _get_most_recent_nodes(self, limit: int = 3) -> List[TrajectoryNode]:
        """Get most recent trajectories"""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.timestamp,
            reverse=True
        )
        return sorted_nodes[:limit]
    
    def _compute_distance(self, a: List[float], b: List[float]) -> float:
        """Euclidean distance"""
        if len(a) != len(b):
            return 1.0
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    
    def update_motif(
        self,
        motif_id: str,
        trajectory_ids: List[str],
        centroid: List[float]
    ) -> MotifState:
        """Update or create motif state"""
        if motif_id in self.motifs:
            motif = self.motifs[motif_id]
            motif.trajectory_ids = trajectory_ids
            motif.last_seen = datetime.utcnow()
            motif.density = len(trajectory_ids) / max(1, self.total_trajectories)
        else:
            motif = MotifState(
                motif_id=motif_id,
                centroid=centroid,
                trajectory_ids=trajectory_ids,
                density=len(trajectory_ids) / max(1, self.total_trajectories)
            )
            self.motifs[motif_id] = motif
        
        return motif
    
    def build_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build transition probability matrix between motifs"""
        self.transition_matrix = defaultdict(lambda: defaultdict(float))
        transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for node in self.nodes.values():
            if node.motif_id and node.previous_nodes:
                prev_node = self.nodes.get(node.previous_nodes[0])
                if prev_node and prev_node.motif_id:
                    transition_counts[prev_node.motif_id][node.motif_id] += 1
        
        for from_motif, to_counts in transition_counts.items():
            total = sum(to_counts.values())
            for to_motif, count in to_counts.items():
                self.transition_matrix[from_motif][to_motif] = count / total
        
        return dict(self.transition_matrix)
    
    def compute_motif_entropy(self, motif_id: str) -> float:
        """Compute behavioral entropy of motif"""
        if motif_id not in self.transition_matrix:
            return 0.0
        
        probs = list(self.transition_matrix[motif_id].values())
        if not probs:
            return 0.0
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        
        return entropy
    
    def detect_loops(self) -> List[List[str]]:
        """Detect recurring trajectory loops"""
        loops = []
        visited: Set[str] = set()
        
        for node_id, node in self.nodes.items():
            if node_id in visited:
                continue
            
            path = []
            current = node_id
            
            while len(path) < 20:  # Max loop length
                if current in path:
                    loop_start = path.index(current)
                    loop = path[loop_start:] + [current]
                    if len(loop) > 2:
                        loops.append(loop)
                        for n in loop:
                            visited.add(n)
                    break
                
                path.append(current)
                
                if current not in self.nodes:
                    break
                
                next_nodes = self.nodes[current].next_nodes
                if not next_nodes:
                    break
                
                current = next_nodes[0]
        
        self.loop_count = len(loops)
        return loops
    
    def detect_drift(self, threshold: float = 0.7) -> List[Dict]:
        """Detect behavioral drift episodes"""
        drifts = []
        
        for node_id, node in self.nodes.items():
            if node.divergence_from_previous > threshold:
                drifts.append({
                    "trajectory_id": node_id,
                    "drift_magnitude": node.divergence_from_previous,
                    "timestamp": node.timestamp.isoformat(),
                    "likely_motif_change": True
                })
        
        self.drift_episodes = len(drifts)
        return drifts
    
    def find_recovery_paths(
        self,
        from_motif: str,
        to_motif: str
    ) -> List[List[str]]:
        """Find recovery paths between motifs"""
        if from_motif not in self.transition_matrix:
            return []
        
        path = [from_motif]
        current = from_motif
        max_depth = 10
        
        for _ in range(max_depth):
            if current not in self.transition_matrix:
                break
            
            transitions = self.transition_matrix[current]
            if to_motif in transitions:
                path.append(to_motif)
                return path
            
            if not transitions:
                break
            
            next_motif = max(transitions.items(), key=lambda x: x[1])[0]
            path.append(next_motif)
            current = next_motif
        
        return []
    
    def get_flow_statistics(self) -> Dict:
        """Get comprehensive flow statistics"""
        return {
            "total_trajectories": self.total_trajectories,
            "total_motifs": len(self.motifs),
            "loops_detected": self.loop_count,
            "drift_episodes": self.drift_episodes,
            "transition_matrix_size": len(self.transition_matrix),
            "avg_transitions_per_motif": (
                sum(len(t) for t in self.transition_matrix.values()) / len(self.transition_matrix)
                if self.transition_matrix else 0
            ),
        }
    
    def get_recent_flow(self, limit: int = 5) -> List[Dict]:
        """Get recent trajectory flow"""
        recent = self._get_most_recent_nodes(limit)
        flow = []
        
        for node in recent:
            flow.append({
                "trajectory_id": node.trajectory_id,
                "motif_id": node.motif_id,
                "divergence": node.divergence_from_previous,
                "convergence": node.convergence_towards_next,
                "timestamp": node.timestamp.isoformat(),
            })
        
        return flow


# Factory
def create_trajectory_graph() -> TrajectoryMemoryGraph:
    return TrajectoryMemoryGraph()