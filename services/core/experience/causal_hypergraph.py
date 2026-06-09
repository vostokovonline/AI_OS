"""
Causal Belief Hypergraph - Rich causal semantics for cognitive operations

Provides:
- Causal edges with weights, mediation, evidence strength
- Temporal decay for causal links
- Counterfactual branches
- Policy mediation tracking
- Attractor dynamics (oscillation, convergence, fragmentation)

Key distinction:
- dependency list: "A influenced B"
- causal hypergraph: "A (weight=0.7, policy=P, evidence=E) caused B under context C"
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import math


class EdgeWeight(Enum):
    """Causal edge weight types"""
    STRONG = "strong"  # Direct causation
    WEAK = "weak"  # Indirect influence
    MEDIATED = "mediated"  # Through policy
    COINCIDENTAL = "coincidental"  # Temporal correlation only


class AttractorState(Enum):
    """Belief manifold attractor states"""
    STABLE = "stable"  # Converged
    OSCILLATING = "oscillating"  # Alternating trust/distrust
    CONVERGING = "converging"  # Moving toward consensus
    DIVERGING = "diverging"  # Fragmenting
    POLARIZING = "polarizing"  # Moving to extremes
    RECURSIVE = "recursive"  # Meta-level instability


@dataclass
class CausalEdge:
    """Single causal edge with rich semantics"""
    edge_id: str
    source_id: str
    target_id: str
    weight: float
    edge_type: EdgeWeight
    evidence_strength: float
    temporal_distance: int
    created_at: str
    mediation_policy: Optional[str] = None
    evidence_ids: List[str] = field(default_factory=list)
    last_strengthened: Optional[str] = None
    counterfactual_branches: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "edge_type": self.edge_type.value,
            "mediation_policy": self.mediation_policy,
            "evidence_strength": self.evidence_strength,
            "evidence_ids": self.evidence_ids,
            "temporal_distance": self.temporal_distance,
            "created_at": self.created_at,
            "last_strengthened": self.last_strengthened,
            "counterfactual_branches": self.counterfactual_branches
        }


@dataclass
class BeliefNode:
    """Belief node with hypergraph context"""
    belief_id: str
    proposition: str
    confidence: float
    first_seen: str
    last_updated: str
    incoming_edges: List[str] = field(default_factory=list)
    outgoing_edges: List[str] = field(default_factory=list)
    update_count: int = 0
    attractor_state: AttractorState = AttractorState.STABLE
    oscillation_count: int = 0
    last_direction: Optional[str] = None
    oscillation_period: Optional[int] = None
    meta_beliefs: List[str] = field(default_factory=list)


@dataclass
class HypergraphSnapshot:
    """Snapshot of hypergraph state for replay/recovery"""
    snapshot_id: str
    nodes: Dict[str, BeliefNode]
    edges: Dict[str, CausalEdge]
    total_beliefs: int
    total_edges: int
    avg_centrality: float
    attractor_distribution: Dict[str, int]
    timestamp: str


class TemporalDecay:
    """Temporal decay for causal links"""
    
    @staticmethod
    def compute_decay(
        created_at: str,
        current_time: str,
        half_life_hours: float = 24.0
    ) -> float:
        """Compute decay factor based on time elapsed"""
        try:
            created = datetime.fromisoformat(created_at)
            current = datetime.fromisoformat(current_time)
            hours_elapsed = (current - created).total_seconds() / 3600
            return math.exp(-hours_elapsed / half_life_hours)
        except:
            return 1.0
    
    @staticmethod
    def strengthen_edge(
        edge: CausalEdge,
        current_time: str,
        reinforcement: float = 0.1
    ) -> CausalEdge:
        """Strengthen edge based on recent confirmation"""
        decay = TemporalDecay.compute_decay(
            edge.created_at, current_time, half_life_hours=24.0
        )
        
        # New weight = max(old * decay + reinforcement, 1.0)
        new_weight = min(edge.weight * decay + reinforcement, 1.0)
        
        edge.weight = new_weight
        edge.last_strengthened = current_time
        
        return edge


class CounterfactualBranch:
    """Counterfactual causal branches"""
    
    @staticmethod
    def add_branch(
        edge: CausalEdge,
        counterfactual_source: str,
        counterfactual_weight: float,
        condition: str
    ) -> CausalEdge:
        """Add counterfactual branch to edge"""
        edge.counterfactual_branches.append({
            "branch_id": str(uuid4()),
            "counterfactual_source": counterfactual_source,
            "counterfactual_weight": counterfactual_weight,
            "condition": condition,
            "created_at": datetime.utcnow().isoformat()
        })
        return edge


class PolicyMediation:
    """Track policy mediation in causal chain"""
    
    @staticmethod
    def add_mediation(
        edge: CausalEdge,
        policy_id: str,
        mediation_strength: float
    ) -> CausalEdge:
        """Add policy mediation to edge"""
        edge.mediation_policy = policy_id
        edge.edge_type = EdgeWeight.MEDIATED
        
        # Reduce direct weight, mediated weight handled by policy
        edge.weight = edge.weight * (1.0 - mediation_strength)
        
        return edge


class AttractorAnalyzer:
    """Analyze belief manifold attractor states"""
    
    def __init__(self, history_window: int = 10):
        self.history_window = history_window
        self._confidence_history: Dict[str, List[float]] = {}
    
    def analyze(self, belief_id: str, current_confidence: float) -> AttractorState:
        """Analyze attractor state of a belief"""
        if belief_id not in self._confidence_history:
            self._confidence_history[belief_id] = []
        
        history = self._confidence_history[belief_id]
        history.append(current_confidence)
        
        # Keep only recent history
        if len(history) > self.history_window:
            history.pop(0)
        
        if len(history) < 3:
            return AttractorState.STABLE
        
        # Detect oscillation (alternating up/down)
        directions = []
        for i in range(1, len(history)):
            if history[i] > history[i-1]:
                directions.append("increasing")
            elif history[i] < history[i-1]:
                directions.append("decreasing")
        
        # Check for oscillation pattern
        if len(directions) >= 3:
            # Alternating pattern
            if directions[-1] != directions[-2] and directions[-2] != directions[-3]:
                return AttractorState.OSCILLATING
        
        # Check for polarization (moving to extremes)
        recent = history[-3:]
        if all(c > 0.7 for c in recent) or all(c < 0.3 for c in recent):
            return AttractorState.POLARIZING
        
        # Check for convergence (stabilizing)
        if len(history) >= 5:
            variance = self._variance(history[-5:])
            if variance < 0.01:
                return AttractorState.CONVERGING
        
        # Check for divergence (increasing variance)
        if len(history) >= 5:
            early_var = self._variance(history[:3])
            late_var = self._variance(history[-3:])
            if late_var > early_var * 1.5:
                return AttractorState.DIVERGING
        
        return AttractorState.STABLE
    
    def _variance(self, values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


class CausalHypergraph:
    """
    Causal Belief Hypergraph - tracks rich causal relationships between beliefs.
    
    Provides:
    - Nodes with incoming/outgoing edges
    - Weighted causal edges with evidence
    - Temporal decay
    - Policy mediation
    - Counterfactual branches
    - Attractor state analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Core hypergraph
        self._nodes: Dict[str, BeliefNode] = {}
        self._edges: Dict[str, CausalEdge] = {}
        
        # Attractor analyzer
        self._attractor = AttractorAnalyzer(
            history_window=self.config.get("history_window", 10)
        )
        
        # Temporal decay config
        self._half_life_hours = self.config.get("half_life_hours", 24.0)
    
    def add_belief(
        self,
        belief_id: str,
        proposition: str,
        confidence: float,
        caused_by: Optional[str] = None,
        caused_by_edge_id: Optional[str] = None,
        mediation_policy: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None
    ) -> BeliefNode:
        """Add belief to hypergraph with causal links"""
        
        now = datetime.utcnow().isoformat()
        
        # Create node
        node = BeliefNode(
            belief_id=belief_id,
            proposition=proposition,
            confidence=confidence,
            first_seen=now,
            last_updated=now,
            attractor_state=AttractorState.STABLE
        )
        
        self._nodes[belief_id] = node
        
        # Create causal edge if caused by another belief
        if caused_by and caused_by in self._nodes:
            edge = self._create_causal_edge(
                source_id=caused_by,
                target_id=belief_id,
                policy_id=mediation_policy,
                evidence_ids=evidence_ids or []
            )
            
            # Add edge to nodes
            node.incoming_edges.append(edge.edge_id)
            self._nodes[caused_by].outgoing_edges.append(edge.edge_id)
        
        return node
    
    def _create_causal_edge(
        self,
        source_id: str,
        target_id: str,
        policy_id: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None
    ) -> CausalEdge:
        """Create causal edge with full attribution"""
        
        edge = CausalEdge(
            edge_id=str(uuid4()),
            source_id=source_id,
            target_id=target_id,
            weight=0.5,  # Default weight
            edge_type=EdgeWeight.STRONG if not policy_id else EdgeWeight.MEDIATED,
            mediation_policy=policy_id,
            evidence_strength=0.5,
            evidence_ids=evidence_ids or [],
            temporal_distance=1,
            created_at=datetime.utcnow().isoformat()
        )
        
        # Add policy mediation if present
        if policy_id:
            edge = PolicyMediation.add_mediation(edge, policy_id, 0.3)
        
        self._edges[edge.edge_id] = edge
        
        return edge
    
    def update_belief(
        self,
        belief_id: str,
        new_confidence: float,
        new_proposition: Optional[str] = None,
        reinforced_by: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None
    ) -> Optional[BeliefNode]:
        """Update belief and track causal reinforcement"""
        
        if belief_id not in self._nodes:
            return None
        
        node = self._nodes[belief_id]
        old_confidence = node.confidence
        
        # Update
        node.confidence = new_confidence
        node.last_updated = datetime.utcnow().isoformat()
        node.update_count += 1
        
        if new_proposition:
            node.proposition = new_proposition
        
        # Track direction and detect oscillation
        prev_direction = node.last_direction
        if new_confidence > old_confidence:
            node.last_direction = "increasing"
        elif new_confidence < old_confidence:
            node.last_direction = "decreasing"
        
        # Check for direction change (oscillation)
        if prev_direction and node.last_direction and prev_direction != node.last_direction:
            node.oscillation_count += 1
        
        # Analyze attractor state
        node.attractor_state = self._attractor.analyze(belief_id, new_confidence)
        
        # If reinforced by another belief, strengthen causal link
        if reinforced_by and reinforced_by in self._nodes:
            # Find existing edge or create new
            existing_edge = self._find_edge(reinforced_by, belief_id)
            
            if existing_edge:
                existing_edge = TemporalDecay.strengthen_edge(
                    existing_edge,
                    node.last_updated,
                    reinforcement=0.1
                )
            else:
                self._create_causal_edge(
                    source_id=reinforced_by,
                    target_id=belief_id,
                    evidence_ids=evidence_ids or []
                )
        
        return node
    
    def _find_edge(self, source_id: str, target_id: str) -> Optional[CausalEdge]:
        """Find edge between source and target"""
        for edge in self._edges.values():
            if edge.source_id == source_id and edge.target_id == target_id:
                return edge
        return None
    
    def add_counterfactual(
        self,
        source_belief: str,
        target_belief: str,
        counterfactual_source: str,
        condition: str
    ) -> bool:
        """Add counterfactual branch to existing edge"""
        
        edge = self._find_edge(source_belief, target_belief)
        if not edge:
            return False
        
        edge = CounterfactualBranch.add_branch(
            edge,
            counterfactual_source=counterfactual_source,
            counterfactual_weight=0.3,
            condition=condition
        )
        
        return True
    
    def get_causal_path(
        self,
        from_belief: str,
        to_belief: str,
        max_depth: int = 5
    ) -> List[List[str]]:
        """Get all causal paths between two beliefs"""
        
        paths = []
        
        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            if current == target:
                paths.append(path + [current])
                return
            
            if current not in self._nodes:
                return
            
            node = self._nodes[current]
            for edge_id in node.outgoing_edges:
                if edge_id not in self._edges:
                    continue
                edge = self._edges[edge_id]
                if edge.target_id not in path:
                    dfs(edge.target_id, target, path + [current], depth + 1)
        
        dfs(from_belief, to_belief, [], 0)
        
        return paths
    
    def get_root_causes(self, belief_id: str) -> List[str]:
        """Find root causes of a belief (beliefs with no incoming edges)"""
        
        if belief_id not in self._nodes:
            return []
        
        roots = []
        
        def find_roots(bid: str, visited: Set[str]):
            if bid in visited:
                return
            visited.add(bid)
            
            node = self._nodes.get(bid)
            if not node:
                return
            
            if not node.incoming_edges:
                roots.append(bid)
            else:
                for edge_id in node.incoming_edges:
                    edge = self._edges.get(edge_id)
                    if edge:
                        find_roots(edge.source_id, visited)
        
        find_roots(belief_id, set())
        
        return roots
    
    def get_policy_influence(self, policy_id: str) -> Dict[str, float]:
        """Get all beliefs influenced by a policy"""
        
        influence = {}
        
        for edge in self._edges.values():
            if edge.mediation_policy == policy_id:
                influence[edge.target_id] = edge.weight
        
        return influence
    
    def get_attractor_summary(self) -> Dict[str, int]:
        """Get distribution of attractor states"""
        
        summary = {state.value: 0 for state in AttractorState}
        
        for node in self._nodes.values():
            summary[node.attractor_state.value] += 1
        
        return summary
    
    def get_hypergraph_snapshot(self) -> HypergraphSnapshot:
        """Get current hypergraph state for replay/recovery"""
        
        total_centrality = 0.0
        
        for node in self._nodes.values():
            centrality = len(node.incoming_edges) + len(node.outgoing_edges)
            total_centrality += centrality
        
        avg_centrality = total_centrality / max(len(self._nodes), 1)
        
        return HypergraphSnapshot(
            snapshot_id=str(uuid4()),
            nodes={bid: node for bid, node in self._nodes.items()},
            edges={eid: edge for eid, edge in self._edges.items()},
            total_beliefs=len(self._nodes),
            total_edges=len(self._edges),
            avg_centrality=avg_centrality,
            attractor_distribution=self.get_attractor_summary(),
            timestamp=datetime.utcnow().isoformat()
        )
    
    def get_belief(self, belief_id: str) -> Optional[BeliefNode]:
        """Get belief node"""
        return self._nodes.get(belief_id)
    
    def get_all_beliefs(self) -> Dict[str, BeliefNode]:
        """Get all belief nodes"""
        return self._nodes.copy()
    
    def get_edge(self, edge_id: str) -> Optional[CausalEdge]:
        """Get causal edge"""
        return self._edges.get(edge_id)


# Global instance
_hypergraph: Optional[CausalHypergraph] = None


def get_causal_hypergraph(config: Optional[Dict] = None) -> CausalHypergraph:
    """Get global causal hypergraph"""
    global _hypergraph
    if _hypergraph is None:
        _hypergraph = CausalHypergraph(config)
    return _hypergraph


def create_belief_hypergraph(
    beliefs: Dict[str, Dict],
    causal_links: List[Tuple[str, str, Optional[str]]]  # source, target, policy
) -> CausalHypergraph:
    """Convenience function to create hypergraph from beliefs and links"""
    
    hypergraph = CausalHypergraph()
    
    # Add beliefs
    for bid, belief in beliefs.items():
        hypergraph.add_belief(
            belief_id=bid,
            proposition=belief.get("proposition", ""),
            confidence=belief.get("confidence", 0.5)
        )
    
    # Add causal links
    for source, target, policy in causal_links:
        hypergraph._create_causal_edge(
            source_id=source,
            target_id=target,
            policy_id=policy
        )
    
    return hypergraph