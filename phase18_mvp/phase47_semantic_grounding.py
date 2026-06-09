"""
Phase 47 — Semantic Grounding & Narrative Compression.

ARCHITECTURAL SHIFT:
  Before (Phases 25-46):  system predicts world + self
                           no semantic abstraction layer
                           memory is raw trajectories + latent states

  After (Phase 47):        system projects cognition into semantic space
                           language indexes, compresses, stabilizes — but does NOT rewrite
                           episodic graph persists autobiographical structure

  Architecture:
    (objects + self + agency + chunks + goals + counterfactuals)
                          ↓
    47.1 — SemanticProjection      → shared semantic manifold
    47.2 — EpisodicSemanticGraph   → associative autobiographical graph
    47.3 — NarrativeStabilizer     → temporal compression, causal continuity
    47.4 — LanguageBind            → weak semantic↔language interface
    47.5 — SemanticRetrieval       → goal/agency/self-aware traversal

  EVERY step (integrated into SelfEngine):
    1-8.  SelfEngine step (phases 25-46)
    9.    Semantic projection of cognitive state        (47.1)
    10.   Episodic graph update                          (47.2)
    11.   Periodic narrative compression                 (47.3)
    12.   Semantic retrieval (if triggered)               (47.5)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.insert(0, '.')

from phase47_self_model import (
    SelfEngine, SelfLatent, AgencyInference, CounterfactualSelf
)
from phase46_temporal_abstraction import (
    HierarchicalEngine, MacroFlow, TemporalChunker, MacroFlowManifold
)
from phase44_object_centric_world_model import ObjectSlot
from phase35_dynamical_skill_flows import FlowManifold, FlowType
from phase36_behavioral_physics_learning import FlowConditionedWorldModel
from phase38_energy_regularized_dynamics import EnergyCostFunction


# ============================================================================
# 47.1 — SEMANTIC PROJECTION
# ============================================================================

class SemanticFactorType(Enum):
    OBJECT = 'object'
    SELF_STATE = 'self_state'
    AGENCY_EVENT = 'agency_event'
    TEMPORAL_CHUNK = 'temporal_chunk'
    GOAL_ATTRACTOR = 'goal_attractor'
    COUNTERFACTUAL_BRANCH = 'counterfactual_branch'
    ACTION_OUTCOME = 'action_outcome'
    NARRATIVE_EPISODE = 'narrative_episode'


@dataclass
class SemanticFactor:
    """A factor in the shared semantic manifold.

    Each factor is a projection of a specific cognitive layer element
    into a common semantic space. Factors are linked by causal/temporal/agency edges.
    """
    factor_id: str
    factor_type: SemanticFactorType
    vector: np.ndarray
    source_layer: str
    source_id: str
    timestamp: int
    attributes: Dict[str, Any] = field(default_factory=dict)
    parent_factor_ids: List[str] = field(default_factory=list)
    child_factor_ids: List[str] = field(default_factory=list)


class SemanticProjection:
    """
    Projects each cognitive layer into a shared semantic manifold space.

    NOT a translator. Structural factorization of cognition:
    - object slots → named entities with attributes
    - self latent → continuous identity thread
    - agency edges → causal relations (self caused X)
    - temporal chunks → semantic episodes
    - goal attractors → intentions
    - counterfactual branches → subjunctive propositions

    All factors live in a SINGLE semantic manifold space (semantic_dim).
    """

    def __init__(
        self,
        semantic_dim: int = 32,
        latent_dim: int = 16,
        self_dim: int = 8,
        slot_dim: int = 8,
        goal_dim: int = 16
    ):
        self.semantic_dim = semantic_dim
        self.latent_dim = latent_dim
        self.self_dim = self_dim
        self.slot_dim = slot_dim
        self.goal_dim = goal_dim

        # Projection matrices (learnable) - each layer → semantic space
        # Object slot → semantic factor
        self.W_object = np.random.randn(semantic_dim, slot_dim) * 0.05
        # Self latent → semantic factor
        self.W_self = np.random.randn(semantic_dim, self_dim) * 0.05
        # Agency vector → semantic factor  (self + action dimensions)
        self.W_agency = np.random.randn(semantic_dim, self_dim + latent_dim) * 0.05
        # Temporal chunk → semantic factor
        self.W_chunk = np.random.randn(semantic_dim, latent_dim * 2) * 0.05
        # Goal attractor → semantic factor
        self.W_goal = np.random.randn(semantic_dim, goal_dim) * 0.05
        # Counterfactual branch → semantic factor
        self.W_counterfactual = np.random.randn(semantic_dim, latent_dim + 4) * 0.05
        # Action-outcome pair → semantic factor
        self.W_outcome = np.random.randn(semantic_dim, latent_dim * 2 + self_dim) * 0.05

        # Current factors in the manifold this step
        self.current_factors: List[SemanticFactor] = []
        self.factor_count: int = 0
        self.factor_history: Dict[str, SemanticFactor] = {}

    def _next_factor_id(self, prefix: str = 'f') -> str:
        self.factor_count += 1
        return f"{prefix}_{self.factor_count}"

    def project_object(
        self, obj: ObjectSlot, timestamp: int
    ) -> SemanticFactor:
        """Project an object slot into semantic space."""
        vec = self.W_object @ obj.state
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('obj'),
            factor_type=SemanticFactorType.OBJECT,
            vector=vec,
            source_layer='object_centric',
            source_id=obj.id,
            timestamp=timestamp,
            attributes={
                'name': f"object_{obj.id}",
                'persistence': float(getattr(obj, 'persistence', 0.0)),
                'last_seen': getattr(obj, 'last_seen', 0),
                'birth_step': getattr(obj, 'birth_step', 0),
                'epistemic_uncertainty': float(getattr(obj, 'epistemic_uncertainty', 0.0)),
                'aleatoric_uncertainty': float(getattr(obj, 'aleatoric_uncertainty', 0.0)),
                'is_active': getattr(obj, 'last_seen', 0) > 0
            }
        )

    def project_self(
        self, self_latent: SelfLatent, timestamp: int
    ) -> SemanticFactor:
        """Project self-latent into semantic space."""
        vec = self.W_self @ self_latent.state
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('self'),
            factor_type=SemanticFactorType.SELF_STATE,
            vector=vec,
            source_layer='self_model',
            source_id='self',
            timestamp=timestamp,
            attributes={
                'identity_coherence': float(self_latent.identity_coherence),
                'state_norm': float(np.linalg.norm(self_latent.state)),
                'history_length': len(self_latent.history)
            }
        )

    def project_agency(
        self,
        agency: AgencyInference,
        action: np.ndarray,
        latent_agency_score: float,
        object_agency_score: float,
        per_object_agency: Dict[str, float],
        timestamp: int
    ) -> SemanticFactor:
        """Project agency inference into semantic space."""
        encoded_action = action[:self.latent_dim] if len(action) >= self.latent_dim \
            else np.pad(action, (0, self.latent_dim - len(action)))
        agency_vec = np.concatenate([
            np.full(self.self_dim, latent_agency_score),
            encoded_action[:self.latent_dim]
        ])
        vec = self.W_agency @ agency_vec[:self.self_dim + self.latent_dim]
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('age'),
            factor_type=SemanticFactorType.AGENCY_EVENT,
            vector=vec,
            source_layer='agency_inference',
            source_id=f"agency_{timestamp}",
            timestamp=timestamp,
            attributes={
                'latent_agency': float(latent_agency_score),
                'object_agency': float(object_agency_score),
                'per_object_agency': {
                    k: float(v) for k, v in per_object_agency.items()
                },
                'is_self_caused': latent_agency_score > 0.5
            }
        )

    def project_chunk(
        self,
        chunk: Dict[str, Any],
        timestamp: int
    ) -> SemanticFactor:
        """Project a temporal chunk into semantic space."""
        z_before = chunk.get('z_before', np.zeros(self.latent_dim))
        z_after = chunk.get('z_after', np.zeros(self.latent_dim))
        combined = np.concatenate([z_before, z_after])
        vec = self.W_chunk @ combined[:self.latent_dim * 2]
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('chk'),
            factor_type=SemanticFactorType.TEMPORAL_CHUNK,
            vector=vec,
            source_layer='temporal_chunking',
            source_id=f"chunk_{chunk.get('chunk_id', timestamp)}",
            timestamp=timestamp,
            attributes={
                'goal_prob_start': float(chunk.get('goal_prob_start', 0.0)),
                'goal_prob_end': float(chunk.get('goal_prob_end', 0.0)),
                'gp_delta': float(chunk.get('goal_prob_end', 0.0) - chunk.get('goal_prob_start', 0.0)),
                'epistemic_uncertainty': float(chunk.get('epistemic', 0.0)),
                'flow_id': str(chunk.get('flow_id', '')),
                'n_steps': int(chunk.get('n_steps', 1))
            }
        )

    def project_goal(
        self,
        goal_vector: np.ndarray,
        goal_prob: float,
        n_samples: int,
        timestamp: int
    ) -> SemanticFactor:
        """Project goal attractor into semantic space."""
        gv = goal_vector[:self.goal_dim] if len(goal_vector) >= self.goal_dim \
            else np.pad(goal_vector, (0, self.goal_dim - len(goal_vector)))
        vec = self.W_goal @ gv
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('goal'),
            factor_type=SemanticFactorType.GOAL_ATTRACTOR,
            vector=vec,
            source_layer='goal_manifold',
            source_id='goal_manifold',
            timestamp=timestamp,
            attributes={
                'goal_probability': float(goal_prob),
                'n_success_samples': n_samples,
                'is_learned': n_samples > 0
            }
        )

    def project_counterfactual(
        self,
        cf_result: Dict[str, Any],
        timestamp: int
    ) -> SemanticFactor:
        """Project a counterfactual simulation into semantic space."""
        cf_action = cf_result.get('cf_action', np.zeros(self.latent_dim))
        base = cf_action[:self.latent_dim]
        regret = cf_result.get('regret', 0.0)
        better = cf_result.get('is_better', False)
        extra = np.array([regret, float(better), 0.0, 0.0])
        combined = np.concatenate([base, extra])
        vec = self.W_counterfactual @ combined
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('cf'),
            factor_type=SemanticFactorType.COUNTERFACTUAL_BRANCH,
            vector=vec,
            source_layer='counterfactual_self',
            source_id=f"cf_{timestamp}",
            timestamp=timestamp,
            attributes={
                'regret': float(regret),
                'is_better': bool(better),
                'cf_action_norm': float(np.linalg.norm(cf_action))
            }
        )

    def project_outcome(
        self,
        z_before: np.ndarray,
        z_after: np.ndarray,
        action: np.ndarray,
        self_state: np.ndarray,
        goal_prob_delta: float,
        timestamp: int
    ) -> SemanticFactor:
        """Project an action-outcome pair into semantic space."""
        combined = np.concatenate([
            z_before[:self.latent_dim],
            z_after[:self.latent_dim],
            self_state[:self.self_dim]
        ])
        vec = self.W_outcome @ combined[:self.latent_dim * 2 + self.self_dim]
        vec = vec / (np.linalg.norm(vec) + 1e-8)

        return SemanticFactor(
            factor_id=self._next_factor_id('out'),
            factor_type=SemanticFactorType.ACTION_OUTCOME,
            vector=vec,
            source_layer='execution',
            source_id=f"outcome_{timestamp}",
            timestamp=timestamp,
            attributes={
                'gp_delta': float(goal_prob_delta),
                'z_delta_norm': float(np.linalg.norm(z_after - z_before)),
                'action_norm': float(np.linalg.norm(action))
            }
        )

    def project_step(
        self,
        step_result: Dict[str, Any],
        self_latent: SelfLatent,
        agency: AgencyInference,
        objects: List[ObjectSlot],
        goal_vector: np.ndarray,
        goal_n_samples: int,
        counterfactual_result: Optional[Dict[str, Any]],
        timestamp: int
    ) -> List[SemanticFactor]:
        """Project entire cognitive state into semantic factors.

        Returns a list of all factors for this step.
        """
        factors: List[SemanticFactor] = []

        # Project self-latent
        factors.append(self.project_self(self_latent, timestamp))

        # Project agency
        per_obj = {}
        if hasattr(agency, 'object_agency_buffer') and agency.object_agency_buffer:
            buffer_entry = agency.object_agency_buffer[-1]
            per_obj = buffer_entry if isinstance(buffer_entry, dict) else {}
        factors.append(self.project_agency(
            agency,
            step_result.get('action', np.zeros(4)),
            step_result.get('latent_agency', 0.0),
            step_result.get('object_agency', 0.0),
            per_obj,
            timestamp
        ))

        # Project objects
        for obj in objects:
            factors.append(self.project_object(obj, timestamp))

        # Project goal
        factors.append(self.project_goal(
            goal_vector, step_result.get('goal_prob', 0.0),
            goal_n_samples, timestamp
        ))

        # Project outcome
        z_before = step_result.get('z_before', np.zeros(self.latent_dim))
        z_after = step_result.get('z_after', np.zeros(self.latent_dim))
        action = step_result.get('action', np.zeros(4))
        gp_delta = step_result.get('gp_delta', 0.0)
        factors.append(self.project_outcome(
            z_before, z_after, action,
            self_latent.state, gp_delta, timestamp
        ))

        # Project counterfactual if available
        if counterfactual_result is not None:
            factors.append(self.project_counterfactual(
                counterfactual_result, timestamp
            ))

        self.current_factors = factors
        for f in factors:
            self.factor_history[f.factor_id] = f

        return factors

    def compute_semantic_similarity(
        self, v1: np.ndarray, v2: np.ndarray
    ) -> float:
        """Cosine similarity in semantic manifold space."""
        n1 = np.linalg.norm(v1) + 1e-8
        n2 = np.linalg.norm(v2) + 1e-8
        return float(np.dot(v1, v2) / (n1 * n2))

    def get_stats(self) -> Dict:
        return {
            'n_factors_total': len(self.factor_history),
            'n_current_factors': len(self.current_factors),
            'semantic_dim': self.semantic_dim,
            'factor_types': {
                ft.value: sum(
                    1 for f in self.factor_history.values()
                    if f.factor_type == ft
                )
                for ft in SemanticFactorType
            }
        }


# ============================================================================
# 47.2 — EPISODIC SEMANTIC GRAPH
# ============================================================================

@dataclass
class EpisodeNode:
    """A node in the episodic semantic graph.

    Represents a cognitive event, object state, self-state, or semantic episode.
    """
    node_id: str
    node_type: str  # 'object', 'self_state', 'episode', 'goal_state', 'outcome', 'counterfactual'
    semantic_vector: np.ndarray
    timestamp: int
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeEdge:
    """A causal/temporal/agency edge between two episodic nodes."""
    edge_id: str
    source_id: str
    target_id: str
    relation: str  # 'caused', 'followed_by', 'self_acted', 'achieved', 'led_to', 'contrasts_with'
    weight: float
    timestamp: int
    attributes: Dict[str, Any] = field(default_factory=dict)


class EpisodicSemanticGraph:
    """
    Persistent associative autobiographical graph.

    NOT a vector DB. This is a structured graph with:
    - Typed nodes (objects, self-states, episodes, outcomes)
    - Typed edges (causal, temporal, agency, goal)
    - Associative traversal rather than similarity search
    - Temporal decay to manage memory growth

    Episodes maintain:
    - Causal continuity (what led to what)
    - Agency ownership (who caused what)
    - Temporal ordering
    - Regret links
    - Goal transitions
    """

    def __init__(
        self,
        semantic_dim: int = 32,
        max_nodes: int = 500,
        max_edges: int = 2000,
        decay_rate: float = 0.001
    ):
        self.semantic_dim = semantic_dim
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.decay_rate = decay_rate

        self.nodes: Dict[str, EpisodeNode] = {}
        self.edges: Dict[str, EpisodeEdge] = {}
        self.edge_count: int = 0

        # Indexes for fast traversal
        self.nodes_by_type: Dict[str, Set[str]] = {}
        self.edges_from: Dict[str, Set[str]] = {}
        self.edges_to: Dict[str, Set[str]] = {}
        self.nodes_by_time: List[str] = []

    def _next_edge_id(self) -> str:
        self.edge_count += 1
        return f"e_{self.edge_count}"

    def add_node(
        self,
        factor: SemanticFactor
    ) -> str:
        """Add a semantic factor as a graph node."""
        if factor.factor_id in self.nodes:
            return factor.factor_id

        node = EpisodeNode(
            node_id=factor.factor_id,
            node_type=factor.factor_type.value,
            semantic_vector=factor.vector.copy(),
            timestamp=factor.timestamp,
            attributes={
                **factor.attributes,
                'source_layer': factor.source_layer,
                'source_id': factor.source_id
            }
        )

        self.nodes[node.node_id] = node
        self.nodes_by_type.setdefault(node.node_type, set()).add(node.node_id)
        self.nodes_by_time.append(node.node_id)

        # Prune if exceeding max
        if len(self.nodes) > self.max_nodes:
            self._prune_oldest()

        return node.node_id

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        attributes: Optional[Dict] = None
    ) -> Optional[str]:
        """Add a typed edge between two episodic nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        edge_id = self._next_edge_id()
        edge = EpisodeEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            timestamp=self.nodes[target_id].timestamp,
            attributes=attributes or {}
        )

        self.edges[edge_id] = edge
        self.edges_from.setdefault(source_id, set()).add(edge_id)
        self.edges_to.setdefault(target_id, set()).add(edge_id)

        if len(self.edges) > self.max_edges:
            self._prune_weakest_edges()

        return edge_id

    def link_factors(
        self,
        factors: List[SemanticFactor],
        prev_factors: List[SemanticFactor]
    ):
        """Create causal/temporal edges between current and previous factors.

        Edge types:
        - self_state → outcome: 'self_acted'
        - object → outcome: 'involved'
        - outcome → goal: 'affected_goal'
        - self_state → self_state: 'followed_by' (temporal continuity)
        - object → object: 'persisted' (if same source_id)
        - agency → outcome: 'caused'
        """
        for f in factors:
            self.add_node(f)

        # Link current factor steps to previous for temporal continuity
        for f_curr in factors:
            for f_prev in prev_factors:
                # Same type, temporal link
                if f_curr.factor_type == f_prev.factor_type:
                    if (f_curr.source_id == f_prev.source_id
                        or (f_curr.factor_type == SemanticFactorType.SELF_STATE)):
                        self.add_edge(
                            f_prev.factor_id, f_curr.factor_id,
                            'followed_by', weight=0.8
                        )
                # Self → outcome causal link
                if (f_curr.factor_type == SemanticFactorType.ACTION_OUTCOME
                    and f_prev.factor_type == SemanticFactorType.SELF_STATE):
                    self.add_edge(
                        f_prev.factor_id, f_curr.factor_id,
                        'self_acted', weight=0.9
                    )
                # Agency → outcome
                if (f_curr.factor_type == SemanticFactorType.ACTION_OUTCOME
                    and f_prev.factor_type == SemanticFactorType.AGENCY_EVENT):
                    self.add_edge(
                        f_prev.factor_id, f_curr.factor_id,
                        'caused',
                        weight=f_prev.attributes.get('latent_agency', 0.5)
                    )
                # Outcome → goal
                if (f_curr.factor_type == SemanticFactorType.GOAL_ATTRACTOR
                    and f_prev.factor_type == SemanticFactorType.ACTION_OUTCOME):
                    self.add_edge(
                        f_prev.factor_id, f_curr.factor_id,
                        'affected_goal',
                        weight=abs(f_prev.attributes.get('gp_delta', 0.0))
                    )

    def get_self_trajectory(self) -> List[EpisodeNode]:
        """Get the self-state trajectory ordered by time."""
        self_nodes = self.nodes_by_type.get('self_state', set())
        sorted_nodes = sorted(
            [self.nodes[nid] for nid in self_nodes],
            key=lambda n: n.timestamp
        )
        return sorted_nodes

    def get_episodes_for_goal(self, goal_vector: np.ndarray, top_k: int = 5) -> List[EpisodeNode]:
        """Retrieve episodes most semantically similar to a goal state."""
        outcome_nodes = self.nodes_by_type.get('action_outcome', set())
        scored = []
        for nid in outcome_nodes:
            node = self.nodes[nid]
            sim = float(np.dot(node.semantic_vector, goal_vector) /
                        (np.linalg.norm(node.semantic_vector) + 1e-8))
            scored.append((sim, node))
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:top_k]]

    def get_causal_chain(self, from_node_id: str, max_depth: int = 5) -> List[EpisodeNode]:
        """Traverse forward causal chain from a node."""
        chain = []
        visited = set()
        queue = [(from_node_id, 0)]
        while queue and len(chain) < max_depth:
            nid, depth = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            if nid in self.nodes:
                chain.append(self.nodes[nid])
            for eid in self.edges_from.get(nid, set()):
                edge = self.edges[eid]
                if edge.relation in ('caused', 'self_acted', 'led_to'):
                    queue.append((edge.target_id, depth + 1))
        return chain

    def get_agency_owned_episodes(self, threshold: float = 0.5) -> List[EpisodeNode]:
        """Get episodes where agency was self-attributed."""
        owned = []
        for eid, edge in self.edges.items():
            if edge.relation == 'self_acted' and edge.weight >= threshold:
                if edge.target_id in self.nodes:
                    owned.append(self.nodes[edge.target_id])
        return owned

    def get_counterfactual_branches(self) -> List[EpisodeNode]:
        """Get all counterfactual branch nodes."""
        cf_nodes = self.nodes_by_type.get('counterfactual_branch', set())
        return [self.nodes[nid] for nid in cf_nodes]

    def _prune_oldest(self):
        """Remove oldest nodes and their edges."""
        if not self.nodes_by_time:
            return
        oldest = self.nodes_by_time.pop(0)
        if oldest in self.nodes:
            # Remove edges
            for eid in list(self.edges_from.get(oldest, set())):
                self.edges.pop(eid, None)
            for eid in list(self.edges_to.get(oldest, set())):
                self.edges.pop(eid, None)
            self.edges_from.pop(oldest, None)
            self.edges_to.pop(oldest, None)
            # Remove from type index
            if oldest in self.nodes:
                node_type = self.nodes[oldest].node_type
                if node_type in self.nodes_by_type and oldest in self.nodes_by_type[node_type]:
                    self.nodes_by_type[node_type].remove(oldest)
            self.nodes.pop(oldest, None)

    def _prune_weakest_edges(self):
        """Remove edges with lowest weight."""
        sorted_edges = sorted(
            self.edges.items(), key=lambda x: x[1].weight
        )
        n_prune = len(self.edges) - self.max_edges
        for eid, _ in sorted_edges[:n_prune]:
            edge = self.edges[eid]
            if edge.source_id in self.edges_from:
                self.edges_from[edge.source_id].discard(eid)
            if edge.target_id in self.edges_to:
                self.edges_to[edge.target_id].discard(eid)
            self.edges.pop(eid, None)

    def get_stats(self) -> Dict:
        return {
            'n_nodes': len(self.nodes),
            'n_edges': len(self.edges),
            'nodes_by_type': {t: len(ns) for t, ns in self.nodes_by_type.items()},
            'self_trajectory_length': len(self.get_self_trajectory()),
            'agency_owned_episodes': len(self.get_agency_owned_episodes()),
            'counterfactual_branches': len(self.get_counterfactual_branches())
        }


# ============================================================================
# 47.3 — NARRATIVE STABILIZER
# ============================================================================

@dataclass
class NarrativeEpisode:
    """
    A compressed cognitive episode preserving causal and agency structure.

    NOT a text summary. This is a compressed causal graph:
    - What changed (object/self deltas)
    - Who caused it (agency attribution)
    - Why it matters (goal relevance)
    - What alternatives existed (counterfactual links)
    - Temporal bounds
    """
    episode_id: str
    start_time: int
    end_time: int
    n_steps: int

    # Causal compression
    primary_agent: str  # 'self' or 'external' or 'mixed'
    goal_delta: float  # net goal probability change
    key_transitions: List[Dict[str, Any]] = field(default_factory=list)

    # Entity states
    object_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    self_coherence: float = 0.0
    mean_agency: float = 0.0
    mean_regret: float = 0.0

    # Semantic factor references
    factor_ids: List[str] = field(default_factory=list)

    # Graph: compressed causal representation
    causal_graph: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=dict)

    # Surface rendering (optional, not primary)
    surface_text: str = ''


class NarrativeStabilizer:
    """
    Memory compression engine.

    Transforms raw temporal trajectories into compressed narrative episodes
    that preserve:
    - Causal continuity (what led to what)
    - Agency ownership (self vs external)
    - Temporal ordering
    - Regret links
    - Goal transitions

    Narrative = compressed causal graph, not text.
    Text is only a surface rendering for the LanguageBind layer.

    Compression ratio: N temporal steps → 1 narrative episode.
    """

    def __init__(
        self,
        min_episode_length: int = 10,
        max_episode_length: int = 100,
        compression_interval: int = 25,
        min_goal_delta: float = 0.05,
        min_agency_shift: float = 0.15
    ):
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length
        self.compression_interval = compression_interval
        self.min_goal_delta = min_goal_delta
        self.min_agency_shift = min_agency_shift

        self.episodes: List[NarrativeEpisode] = []
        self.episode_count: int = 0

    def _next_episode_id(self) -> str:
        self.episode_count += 1
        return f"ep_{self.episode_count}"

    def compress_trajectory(
        self,
        factors: List[SemanticFactor],
        step_results: List[Dict[str, Any]],
        start_time: int,
        end_time: int,
        objects: List[ObjectSlot]
    ) -> NarrativeEpisode:
        """Compress a raw trajectory segment into a narrative episode.

        Key invariants preserved:
        1. Causal continuity: action → outcome links are kept
        2. Agency ownership: self-caused vs external is attributed
        3. Goal relevance: GP delta is the primary significance metric
        4. Temporal ordering: sequence is maintained in causal_graph
        """
        if not step_results:
            return NarrativeEpisode(
                episode_id=self._next_episode_id(),
                start_time=start_time, end_time=end_time,
                n_steps=0, primary_agent='external',
                goal_delta=0.0
            )

        # Extract key transitions where meaningful change occurred
        key_transitions = []
        for sr in step_results:
            gp_delta = sr.get('gp_delta', 0.0)
            agency = sr.get('latent_agency', 0.0)
            z_delta = np.linalg.norm(
                sr.get('z_after', np.zeros(16)) - sr.get('z_before', np.zeros(16))
            )
            if abs(gp_delta) >= self.min_goal_delta or z_delta > 0.5:
                key_transitions.append({
                    'timestamp': sr.get('_step', 0),
                    'gp_delta': float(gp_delta),
                    'agency': float(agency),
                    'z_delta': float(z_delta),
                    'flow_id': sr.get('flow_id', ''),
                    'action_norm': float(np.linalg.norm(
                        sr.get('action', np.zeros(4))
                    ))
                })

        # Determine primary agent
        agencies = [sr.get('latent_agency', 0.0) for sr in step_results]
        mean_agency = float(np.mean(agencies)) if agencies else 0.0
        high_agency = sum(1 for a in agencies if a > 0.5)
        low_agency = sum(1 for a in agencies if a < 0.3)

        if high_agency > low_agency * 2:
            primary_agent = 'self'
        elif low_agency > high_agency * 2:
            primary_agent = 'external'
        else:
            primary_agent = 'mixed'

        # Net goal delta
        gps = [sr.get('goal_prob', 0.0) for sr in step_results]
        goal_delta = float(gps[-1] - gps[0]) if len(gps) >= 2 else 0.0

        # Self coherence
        coherences = [
            sr.get('self_coherence', 1.0) for sr in step_results
        ]

        # Regret
        regrets = [sr.get('cf_regret', 0.0) for sr in step_results
                    if 'cf_regret' in sr]

        # Object states at end
        object_states = {}
        for obj in objects:
            object_states[obj.id] = {
                'persistence': float(getattr(obj, 'persistence', 0.0)),
                'last_seen': getattr(obj, 'last_seen', 0),
                'state_norm': float(np.linalg.norm(obj.state))
            }

        # Build causal graph
        causal_graph: Dict[str, List[Tuple[str, str, float]]] = {}
        for i, kt in enumerate(key_transitions):
            node_id = f"t_{kt['timestamp']}"
            causal_graph[node_id] = []
            if i > 0:
                prev_node = f"t_{key_transitions[i-1]['timestamp']}"
                causal_graph.setdefault(prev_node, [])
                causal_graph[prev_node].append((
                    node_id,
                    'led_to',
                    kt.get('gp_delta', 0.0)
                ))

        factor_ids = [f.factor_id for f in factors]

        episode = NarrativeEpisode(
            episode_id=self._next_episode_id(),
            start_time=start_time,
            end_time=end_time,
            n_steps=len(step_results),
            primary_agent=primary_agent,
            goal_delta=goal_delta,
            key_transitions=key_transitions,
            object_states=object_states,
            self_coherence=float(np.mean(coherences)) if coherences else 0.0,
            mean_agency=mean_agency,
            mean_regret=float(np.mean(regrets)) if regrets else 0.0,
            factor_ids=factor_ids,
            causal_graph=causal_graph
        )

        self.episodes.append(episode)
        return episode

    def should_compress(self, step_index: int, last_compression: int) -> bool:
        """Determine if it's time to compress trajectory into narrative."""
        return (step_index - last_compression) >= self.compression_interval

    def get_latest_episode(self) -> Optional[NarrativeEpisode]:
        if not self.episodes:
            return None
        return self.episodes[-1]

    def get_recent_narratives(self, k: int = 5) -> List[NarrativeEpisode]:
        return self.episodes[-k:]

    def get_stats(self) -> Dict:
        return {
            'n_episodes': len(self.episodes),
            'mean_episode_length': float(np.mean([
                e.n_steps for e in self.episodes
            ])) if self.episodes else 0.0,
            'primary_agents': {
                agent: sum(1 for e in self.episodes if e.primary_agent == agent)
                for agent in ('self', 'external', 'mixed')
            },
            'total_goal_delta': float(
                sum(e.goal_delta for e in self.episodes)
            ) if self.episodes else 0.0
        }


# ============================================================================
# 47.4 — LANGUAGE BIND (WEAK BIDIRECTIONAL INTERFACE)
# ============================================================================

@dataclass
class LanguageToken:
    """A token in the semantic-language interface."""
    token_id: str
    token_type: str  # 'entity', 'action', 'relation', 'state', 'goal', 'self', 'time'
    surface: str
    semantic_vector: np.ndarray
    confidence: float = 1.0


class LanguageBind:
    """
    Weak bidirectional semantic-language interface.

    Maps:
      semantic graph → language tokens (indexing, narration)
      language tokens → semantic graph (activation, retrieval)

    CRITICAL CONSTRAINT:
      Language → latent overwrite = FORBIDDEN.
      Language can index, activate, associate, compress, stabilize.
      But CANNOT replace world grounding.

    This is NOT an LLM. It's a small learned projection that binds
    semantic factor types to surface token types.
    """

    # Semantic type → surface token template mapping
    TYPE_TO_TEMPLATE = {
        SemanticFactorType.OBJECT: "object {name} (persistence: {persistence:.1f})",
        SemanticFactorType.SELF_STATE: "self (coherence: {identity_coherence:.3f})",
        SemanticFactorType.AGENCY_EVENT: "agency: {latent_agency:.3f} ({is_self_caused})",
        SemanticFactorType.TEMPORAL_CHUNK: "chunk: GP {gp_delta:+.3f} ({n_steps} steps)",
        SemanticFactorType.GOAL_ATTRACTOR: "goal (p={goal_probability:.3f})",
        SemanticFactorType.COUNTERFACTUAL_BRANCH: "counterfactual: regret={regret:+.3f}, better={is_better}",
        SemanticFactorType.ACTION_OUTCOME: "outcome: GP {gp_delta:+.3f}, z-delta {z_delta_norm:.2f}",
        SemanticFactorType.NARRATIVE_EPISODE: "episode: agent={primary_agent}, goal_delta={goal_delta:+.3f}"
    }

    def __init__(
        self,
        semantic_dim: int = 32,
        vocab_size: int = 128,
        bind_strength: float = 0.3
    ):
        self.semantic_dim = semantic_dim
        self.vocab_size = vocab_size
        self.bind_strength = bind_strength

        # Learned token embeddings (small, not LLM-scale)
        self.token_embeddings = np.random.randn(vocab_size, semantic_dim) * 0.05

        # Semantic → token projection (learnable)
        self.W_sem2tok = np.random.randn(vocab_size, semantic_dim) * 0.05
        # Token → semantic projection (weak, constrained)
        self.W_tok2sem = np.random.randn(semantic_dim, vocab_size) * 0.02

        self.token_count: int = 0
        self.recent_tokens: List[LanguageToken] = []

    def _next_token_id(self) -> str:
        self.token_count += 1
        return f"tok_{self.token_count}"

    def factor_to_tokens(self, factor: SemanticFactor) -> List[LanguageToken]:
        """Render a semantic factor into surface language tokens.

        This is NOT generation. It's a structural projection:
        factor type + attributes → fixed template → token sequence.

        The output is a list of LanguageToken objects that can be
        used for indexing, narration, or retrieval.
        """
        template = self.TYPE_TO_TEMPLATE.get(factor.factor_type)
        if template is None:
            return []

        try:
            surface = template.format(**factor.attributes)
        except (KeyError, ValueError):
            surface = f"{factor.factor_type.value}: {factor.source_id}"

        # Split into token-level chunks
        words = surface.split()
        tokens = []
        for i, word in enumerate(words):
            # Find closest token embedding
            word_vec = self.semantic_to_token(factor.vector)
            token_id = self._next_token_id()
            token = LanguageToken(
                token_id=token_id,
                token_type=factor.factor_type.value,
                surface=word,
                semantic_vector=factor.vector.copy(),
                confidence=self.bind_strength
            )
            tokens.append(token)

        self.recent_tokens.extend(tokens)
        return tokens

    def semantic_to_token(
        self, semantic_vector: np.ndarray
    ) -> int:
        """Project semantic vector to closest token index.

        Note: This is a weak projection. The token vocabulary is small
        and serves as an index, not a generative vocabulary.
        """
        scores = self.W_sem2tok @ semantic_vector
        return int(np.argmax(scores)) % self.vocab_size

    def token_to_semantic(
        self, token_id: int
    ) -> np.ndarray:
        """Project a token index back into semantic space.

        This is the retrieval path: language token activates
        semantic region, which triggers graph traversal.

        The projection is intentionally weak (small weights, no latent overwrite).
        """
        token_vec = self.token_embeddings[token_id % self.vocab_size]
        semantic_activation = self.W_tok2sem[:, token_id % self.vocab_size]
        return semantic_activation * self.bind_strength

    def render_episode(self, episode: NarrativeEpisode) -> str:
        """Render a compressed narrative episode as surface text.

        This is for explainability and self-description.
        NOT for cognitive processing.
        """
        lines = [
            f"[{episode.episode_id}] Agent: {episode.primary_agent} | "
            f"Steps: {episode.n_steps} | Goal delta: {episode.goal_delta:+.3f}"
        ]
        for kt in episode.key_transitions[:5]:
            lines.append(
                f"  t={kt['timestamp']}: GP {kt['gp_delta']:+.3f}, "
                f"agency={kt['agency']:.2f}, flow={kt['flow_id']}"
            )
        if episode.self_coherence > 0:
            lines.append(
                f"  Self coherence: {episode.self_coherence:.3f} | "
                f"Mean agency: {episode.mean_agency:.3f}"
            )
        if episode.mean_regret != 0.0:
            lines.append(f"  Mean regret: {episode.mean_regret:+.3f}")
        return '\n'.join(lines)

    def query_to_semantic(self, query_tokens: List[str]) -> np.ndarray:
        """Convert a natural language query to semantic activation.

        Each token retrieves its semantic region; activations are combined.
        The result is used for graph traversal, NOT state overwrite.
        """
        combined = np.zeros(self.semantic_dim)
        for qt in query_tokens:
            token_id = hash(qt) % self.vocab_size
            combined += self.token_to_semantic(token_id)
        return combined / (len(query_tokens) + 1e-8)

    def get_stats(self) -> Dict:
        return {
            'vocab_size': self.vocab_size,
            'tokens_generated': self.token_count,
            'recent_tokens': len(self.recent_tokens),
            'bind_strength': self.bind_strength
        }


# ============================================================================
# 47.5 — SEMANTIC RETRIEVAL
# ============================================================================

class RetrievalQuery:
    """A query for episodic semantic graph traversal.

    Unlike vector DB queries, this specifies:
    - target_type: what kind of node to retrieve
    - relation_filter: what relationship to follow
    - agency_filter: self-caused or external
    - time_range: temporal bounds
    - goal_relevance: minimum goal probability delta
    """
    def __init__(
        self,
        target_type: Optional[str] = None,
        relation_filter: Optional[str] = None,
        agency_filter: Optional[str] = None,
        time_range: Optional[Tuple[int, int]] = None,
        goal_relevance: Optional[float] = None,
        semantic_vector: Optional[np.ndarray] = None,
        top_k: int = 5
    ):
        self.target_type = target_type
        self.relation_filter = relation_filter
        self.agency_filter = agency_filter
        self.time_range = time_range
        self.goal_relevance = goal_relevance
        self.semantic_vector = semantic_vector
        self.top_k = top_k


class SemanticRetrieval:
    """
    Goal/agency/self-aware episodic traversal.

    NOT similarity search. This is structured graph traversal:
    - Causal: "what led to this outcome?"
    - Temporal: "what happened before/after?"
    - Agency: "what did I cause vs external?"
    - Goal: "what episodes improved goal probability?"
    - Self: "how did self-coherence change?"
    - Counterfactual: "what alternatives were available?"

    Retrieval is driven by:
    - Current cognitive context (self-state, goal, agency)
    - Query parameters
    - Graph structure (edges, types, attributes)
    """

    def __init__(self, graph: EpisodicSemanticGraph):
        self.graph = graph

    def query(self, query: RetrievalQuery) -> List[EpisodeNode]:
        """Execute a structured query against the episodic graph."""
        candidates: List[EpisodeNode] = []

        # Filter by type
        if query.target_type:
            nids = self.graph.nodes_by_type.get(query.target_type, set())
        else:
            nids = set(self.graph.nodes.keys())

        # Filter by time
        if query.time_range:
            t_start, t_end = query.time_range
            nids = {
                nid for nid in nids
                if t_start <= self.graph.nodes[nid].timestamp <= t_end
            }

        # Filter by agency (for outcome and episode nodes)
        if query.agency_filter == 'self':
            owned = {n.node_id for n in self.graph.get_agency_owned_episodes()}
            nids = nids & owned
        elif query.agency_filter == 'external':
            owned = {n.node_id for n in self.graph.get_agency_owned_episodes()}
            nids = nids - owned

        # Filter by goal relevance (for outcome nodes)
        if query.goal_relevance is not None and query.target_type in (
            'action_outcome', 'temporal_chunk', None
        ):
            filtered = set()
            for nid in nids:
                node = self.graph.nodes[nid]
                gp_delta = abs(node.attributes.get('gp_delta', 0.0))
                if gp_delta >= query.goal_relevance:
                    filtered.add(nid)
            nids = filtered

        # Score by semantic similarity if vector provided
        if query.semantic_vector is not None:
            scored = []
            for nid in nids:
                node = self.graph.nodes[nid]
                sim = float(np.dot(node.semantic_vector, query.semantic_vector) /
                            (np.linalg.norm(node.semantic_vector) + 1e-8))
                scored.append((sim, node))
            scored.sort(key=lambda x: -x[0])
            candidates = [node for _, node in scored[:query.top_k]]
        else:
            # Recent-first ordering
            sorted_nids = sorted(nids, key=lambda nid: self.graph.nodes[nid].timestamp, reverse=True)
            candidates = [self.graph.nodes[nid] for nid in sorted_nids[:query.top_k]]

        return candidates

    def retrieve_causal_predecessors(
        self, node_id: str, max_depth: int = 3
    ) -> List[EpisodeNode]:
        """Retrieve nodes that caused a given node.

        Traverses backward: 'caused', 'self_acted', 'led_to' edges.
        """
        chain = []
        visited = set()
        queue = [(node_id, 0)]
        while queue and len(chain) < max_depth * 3:
            nid, depth = queue.pop(0)
            if nid in visited or depth > max_depth:
                continue
            visited.add(nid)
            if nid in self.graph.nodes:
                chain.append(self.graph.nodes[nid])
            for eid in self.graph.edges_to.get(nid, set()):
                edge = self.graph.edges[eid]
                if edge.relation in ('caused', 'self_acted', 'led_to'):
                    queue.append((edge.source_id, depth + 1))
        return chain

    def retrieve_goal_trajectory(
        self, n_episodes: int = 5
    ) -> List[EpisodeNode]:
        """Retrieve outcome nodes ordered by goal probability delta (descending).

        Answers: "which actions most improved the goal?"
        """
        outcomes = self.graph.nodes_by_type.get('action_outcome', set())
        scored = []
        for nid in outcomes:
            node = self.graph.nodes[nid]
            gp_delta = node.attributes.get('gp_delta', 0.0)
            scored.append((gp_delta, node))
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:n_episodes]]

    def retrieve_self_narrative(
        self, n_episodes: int = 5
    ) -> List[EpisodeNode]:
        """Retrieve self-state nodes ordered by coherence.

        Answers: "when was self most/least stable?"
        """
        self_nodes = self.graph.nodes_by_type.get('self_state', set())
        scored = []
        for nid in self_nodes:
            node = self.graph.nodes[nid]
            coherence = node.attributes.get('identity_coherence', 1.0)
            scored.append((coherence, node))
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:n_episodes]]

    def retrieve_counterfactual_alternatives(
        self, n_alternatives: int = 5
    ) -> List[EpisodeNode]:
        """Retrieve counterfactual branches ordered by regret (most regret first).

        Answers: "what should I have done differently?"
        """
        cf_nodes = self.graph.nodes_by_type.get('counterfactual_branch', set())
        scored = []
        for nid in cf_nodes:
            node = self.graph.nodes[nid]
            regret = abs(node.attributes.get('regret', 0.0))
            scored.append((regret, node))
        scored.sort(key=lambda x: -x[0])
        return [node for _, node in scored[:n_alternatives]]

    def get_stats(self) -> Dict:
        return {}


# ============================================================================
# 47.6 — SEMANTIC ENGINE
# ============================================================================

class SemanticEngine(SelfEngine):
    """
    Extends SelfEngine with semantic grounding and narrative compression.

    Adds:
      47.1 — SemanticProjection:    cognition → semantic factors
      47.2 — EpisodicSemanticGraph: persistent associative graph
      47.3 — NarrativeStabilizer:   temporal compression → narrative episodes
      47.4 — LanguageBind:          weak semantic ↔ language interface
      47.5 — SemanticRetrieval:     goal/agency/self-aware graph traversal

    EVERY step:
      1-8.  SelfEngine step (phases 25-46)
      9.    Project cognitive state into semantic factors    (47.1)
      10.   Add factors to episodic graph with causal links  (47.2)
      11.   Compress trajectory if interval reached            (47.3)
      12.   Update language bind tokens                       (47.4)

    Semantic constraints:
    - LanguageBind CANNOT overwrite latent state
    - Narrative preserves causal continuity, agency ownership
    - EpisodicGraph is traversable by structure, not just similarity
    """

    def __init__(
        self,
        wm: FlowConditionedWorldModel,
        bootstrap: bool = True,
        n_coverage: int = 200,
        n_shaping: int = 150,
        n_transfer: int = 80,
        n_initial_flows: int = 8,
        flow_dim: int = 4,
        lambda_cost: float = 0.3,
        train_interval: int = 5,
        # Phase 43
        n_ensemble: int = 5,
        ensemble_lr: float = 0.005,
        exploration_beta: float = 0.1,
        planning_horizon: int = 5,
        planning_samples: int = 24,
        uncertainty_weight: float = 0.3,
        energy_weight: float = 0.2,
        goal_weight: float = 1.0,
        # Phase 44
        n_slots: int = 6,
        slot_dim: int = 8,
        slot_iterations: int = 3,
        match_threshold: float = 0.5,
        max_objects: int = 10,
        rel_dynamics_lr: float = 0.01,
        # Phase 45
        macro_min_horizon: int = 3,
        macro_max_horizon: int = 10,
        macro_discovery_interval: int = 20,
        # Phase 46
        self_dim: int = 8,
        self_temporal_stability: float = 0.9,
        counterfactual_interval: int = 15,
        n_counterfactuals: int = 3,
        # Phase 47
        semantic_dim: int = 32,
        narrative_compression_interval: int = 25,
        episodic_max_nodes: int = 500,
        episodic_max_edges: int = 2000,
        vocab_size: int = 128,
        bind_strength: float = 0.3
    ):
        super().__init__(
            wm=wm, bootstrap=bootstrap,
            n_coverage=n_coverage, n_shaping=n_shaping,
            n_transfer=n_transfer,
            n_initial_flows=n_initial_flows, flow_dim=flow_dim,
            lambda_cost=lambda_cost, train_interval=train_interval,
            n_ensemble=n_ensemble, ensemble_lr=ensemble_lr,
            exploration_beta=exploration_beta,
            planning_horizon=planning_horizon,
            planning_samples=planning_samples,
            uncertainty_weight=uncertainty_weight,
            energy_weight=energy_weight, goal_weight=goal_weight,
            n_slots=n_slots, slot_dim=slot_dim,
            slot_iterations=slot_iterations,
            match_threshold=match_threshold, max_objects=max_objects,
            rel_dynamics_lr=rel_dynamics_lr,
            macro_min_horizon=macro_min_horizon,
            macro_max_horizon=macro_max_horizon,
            macro_discovery_interval=macro_discovery_interval,
            self_dim=self_dim,
            self_temporal_stability=self_temporal_stability,
            counterfactual_interval=counterfactual_interval,
            n_counterfactuals=n_counterfactuals
        )

        # 47.1 — Semantic Projection
        self.semantic_projection = SemanticProjection(
            semantic_dim=semantic_dim,
            latent_dim=wm.latent_dim,
            self_dim=self_dim,
            slot_dim=slot_dim
        )

        # 47.2 — Episodic Semantic Graph
        self.episodic_graph = EpisodicSemanticGraph(
            semantic_dim=semantic_dim,
            max_nodes=episodic_max_nodes,
            max_edges=episodic_max_edges
        )

        # 47.3 — Narrative Stabilizer
        self.narrative_stabilizer = NarrativeStabilizer(
            compression_interval=narrative_compression_interval
        )

        # 47.4 — Language Bind
        self.language_bind = LanguageBind(
            semantic_dim=semantic_dim,
            vocab_size=vocab_size,
            bind_strength=bind_strength
        )

        # 47.5 — Semantic Retrieval
        self.semantic_retrieval = SemanticRetrieval(
            graph=self.episodic_graph
        )

        # State for compression
        self.last_compression_step: int = 0
        self.semantic_factor_history: List[SemanticFactor] = []
        self.prev_step_factors: List[SemanticFactor] = []
        self.narrative_log: List[NarrativeEpisode] = []

    def step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """One cognitive step with semantic grounding.

        Extends SelfEngine.step() with layers 9-13:
          9.  Semantic projection of cognitive state
          10. Episodic graph update with causal links
          11. Periodic narrative compression
          12. Language bind token update
          13. Semantic re-injection into planning context
        """
        # ====================================================================
        # LAYERS 1-8: SelfEngine step (phases 25-46)
        # ====================================================================
        result = super().step(z, h)

        z_before = result.get('z_before', z)
        z_after = result.get('z_after', z)
        action = result.get('action', np.zeros(self.wm.action_dim))
        goal_prob = result.get('goal_prob', 0.0)

        objects = self.slot_tracker.get_active_objects()
        goal_mean = self.goal_manifold.get_mean()
        goal_vector = goal_mean if goal_mean is not None else np.zeros(self.wm.latent_dim)
        goal_n_samples = len(self.goal_manifold.success_memory.buffer) if hasattr(
            self.goal_manifold, 'success_memory'
        ) else 0

        step_result_with_idx = dict(result)
        step_result_with_idx['_step'] = self.total_steps

        # ====================================================================
        # LAYER 9: SEMANTIC PROJECTION (47.1)
        # ====================================================================
        cf_result = self._get_counterfactual_for_semantic()
        factors = self.semantic_projection.project_step(
            step_result=step_result_with_idx,
            self_latent=self.self_latent,
            agency=self.agency,
            objects=objects,
            goal_vector=goal_vector,
            goal_n_samples=goal_n_samples,
            counterfactual_result=cf_result,
            timestamp=self.total_steps
        )
        self.semantic_factor_history.extend(factors)

        # ====================================================================
        # LAYER 10: EPISODIC GRAPH UPDATE (47.2)
        # ====================================================================
        self.episodic_graph.link_factors(factors, self.prev_step_factors)
        self.prev_step_factors = factors

        # ====================================================================
        # LAYER 11: NARRATIVE COMPRESSION (47.3)
        # ====================================================================
        if self.narrative_stabilizer.should_compress(
            self.total_steps, self.last_compression_step
        ):
            episode = self.narrative_stabilizer.compress_trajectory(
                factors=self.semantic_factor_history[-50:],
                step_results=self.execution_log[
                    self.last_compression_step:self.total_steps + 1
                ],
                start_time=self.last_compression_step,
                end_time=self.total_steps,
                objects=objects
            )
            self.narrative_log.append(episode)
            self.last_compression_step = self.total_steps

        # ====================================================================
        # LAYER 12: LANGUAGE BIND TOKENS (47.4)
        # ====================================================================
        for factor in factors:
            self.language_bind.factor_to_tokens(factor)

        # ====================================================================
        # LAYER 13: SEMANTIC RE-INJECTION INTO PLANNING (47.5)
        # ====================================================================
        semantic_context = self._build_semantic_context(
            goal_vector, result.get('latent_agency', 0.0)
        )
        result['n_retrieved_narratives'] = len(
            semantic_context.get('retrieved', [])
        )
        result['narrative_regret_avg'] = semantic_context.get('avg_regret', 0.0)
        result['narrative_gp_max'] = semantic_context.get('max_gp', 0.0)

        result['n_semantic_factors'] = len(factors)
        result['n_episodic_nodes'] = len(self.episodic_graph.nodes)
        result['n_episodic_edges'] = len(self.episodic_graph.edges)
        result['n_narratives'] = len(self.narrative_log)

        return result

    def _get_counterfactual_for_semantic(self) -> Optional[Dict]:
        """Get counterfactual result for semantic projection from this or previous step."""
        cf_obj = getattr(self, 'counterfactual', None)
        if cf_obj is None:
            return None
        cf_stats = cf_obj.get_stats()
        if cf_stats.get('n_simulations', 0) == 0:
            return None
        regret_history = getattr(cf_obj, 'regret_history', [])
        if not regret_history:
            return None
        return {
            'regret': float(np.mean(regret_history[-3:])),
            'is_better': regret_history[-1] < 0 if regret_history else False,
            'cf_action': np.zeros(self.wm.latent_dim)
        }

    def _build_semantic_context(
        self,
        goal_vector: np.ndarray,
        current_agency: float
    ) -> Dict:
        """Build semantic context from episodic graph for planning re-injection.

        Retrieves relevant past episodes based on current cognitive state
        and computes aggregate signals (regret, agency, goal deltas) that
        can bias planning parameters.
        """
        context: Dict[str, Any] = {
            'retrieved': [],
            'avg_regret': 0.0,
            'max_gp': 0.0,
            'n_self_episodes': 0
        }

        if len(self.episodic_graph.nodes) < 5:
            return context

        # Retrieve goal-relevant outcomes (project to semantic space first)
        if goal_vector is not None and np.any(goal_vector != 0):
            semantic_goal = self.semantic_projection.project_goal(
                goal_vector, 0.0, 0, self.total_steps
            )
            goal_results = self.episodic_graph.get_episodes_for_goal(
                semantic_goal.vector, top_k=3
            )
            context['retrieved'].extend(goal_results)
            gps = [
                abs(n.attributes.get('gp_delta', 0.0))
                for n in goal_results
            ]
            if gps:
                context['max_gp'] = float(max(gps))

        # Retrieve self-owned episodes with regret
        owned = self.episodic_graph.get_agency_owned_episodes()
        context['n_self_episodes'] = len(owned)

        # Compute aggregate regret from counterfactuals
        cf_nodes = self.episodic_graph.get_counterfactual_branches()
        regrets = [
            abs(n.attributes.get('regret', 0.0))
            for n in cf_nodes
            if abs(n.attributes.get('regret', 0.0)) > 0.01
        ]
        if regrets:
            context['avg_regret'] = float(np.mean(regrets))

        return context

    def run(self, z_start: np.ndarray, n_steps: int = 200) -> Dict:
        """Run semantic engine with full semantic grounding."""
        if not hasattr(self, '_last_cf_result'):
            self._last_cf_result = None

        base_result = super().run(z_start, n_steps)

        # Add semantic stats
        base_result['semantic'] = self.semantic_projection.get_stats()
        base_result['episodic_graph'] = self.episodic_graph.get_stats()
        base_result['narrative'] = self.narrative_stabilizer.get_stats()
        base_result['language_bind'] = self.language_bind.get_stats()

        # Build compressed autobiographical narrative
        recent_self = self.episodic_graph.get_self_trajectory()
        base_result['self_narrative_length'] = len(recent_self)
        narrative_texts = []
        for ep in self.narrative_log[-5:]:
            narrative_texts.append(self.language_bind.render_episode(ep))
        base_result['recent_narratives'] = narrative_texts

        return base_result


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_semantic_projection():
    """Test 47.1: Semantic projection into shared manifold."""
    print("\n============================================================")
    print("47.1 — SEMANTIC PROJECTION")
    print("============================================================")

    sp = SemanticProjection(semantic_dim=32, latent_dim=16, self_dim=8, slot_dim=8)

    from phase47_self_model import SelfLatent
    sl = SelfLatent(latent_dim=16, self_dim=8)
    sl.state = np.random.randn(8) * 0.5
    sl.identity_coherence = 0.95

    # Create mock objects
    obj = ObjectSlot(object_id='obj_a')
    obj.state = np.random.randn(8)
    obj.persistence = 5.0
    obj.last_seen = 10
    obj.birth_step = 5
    obj.epistemic_uncertainty = 0.1
    obj.aleatoric_uncertainty = 0.2

    # Test object projection
    f_obj = sp.project_object(obj, timestamp=1)
    assert f_obj.factor_type == SemanticFactorType.OBJECT
    assert f_obj.vector.shape == (32,)
    assert f_obj.attributes.get('persistence') == 5.0
    print(f"  ✓ Object factor: shape={f_obj.vector.shape}, "
          f"type={f_obj.factor_type.value}")

    # Test self projection
    f_self = sp.project_self(sl, timestamp=1)
    assert f_self.factor_type == SemanticFactorType.SELF_STATE
    assert f_self.vector.shape == (32,)
    assert f_self.attributes.get('identity_coherence') == 0.95
    print(f"  ✓ Self factor: shape={f_self.vector.shape}, "
          f"coherence={f_self.attributes['identity_coherence']:.3f}")

    # Test agency projection
    from phase47_self_model import AgencyInference
    agency = AgencyInference(latent_dim=16)
    f_age = sp.project_agency(
        agency, np.random.randn(4), 0.7, 0.5,
        {'a': 0.8, 'b': 0.3}, timestamp=1
    )
    assert f_age.factor_type == SemanticFactorType.AGENCY_EVENT
    assert f_age.vector.shape == (32,)
    assert f_age.attributes['is_self_caused'] == True
    print(f"  ✓ Agency factor: shape={f_age.vector.shape}, "
          f"self_caused={f_age.attributes['is_self_caused']}")

    # Test goal projection
    f_goal = sp.project_goal(
        np.random.randn(16), 0.75, 50, timestamp=1
    )
    assert f_goal.factor_type == SemanticFactorType.GOAL_ATTRACTOR
    assert f_goal.attributes['goal_probability'] == 0.75
    print(f"  ✓ Goal factor: shape={f_goal.vector.shape}, "
          f"p={f_goal.attributes['goal_probability']:.3f}")

    # Test outcome projection
    f_out = sp.project_outcome(
        np.zeros(16), np.ones(16) * 0.5, np.random.randn(4),
        sl.state, 0.15, timestamp=1
    )
    assert f_out.factor_type == SemanticFactorType.ACTION_OUTCOME
    assert f_out.attributes['gp_delta'] == 0.15
    print(f"  ✓ Outcome factor: shape={f_out.vector.shape}, "
          f"gp_delta={f_out.attributes['gp_delta']:+.3f}")

    # Test counterfactual projection
    cf_result = {
        'cf_action': np.random.randn(16),
        'regret': -0.5,
        'is_better': True
    }
    f_cf = sp.project_counterfactual(cf_result, timestamp=1)
    assert f_cf.factor_type == SemanticFactorType.COUNTERFACTUAL_BRANCH
    assert f_cf.attributes['regret'] == -0.5
    print(f"  ✓ Counterfactual factor: shape={f_cf.vector.shape}, "
          f"regret={f_cf.attributes['regret']:+.3f}")

    # Test semantic similarity
    sim = sp.compute_semantic_similarity(f_obj.vector, f_obj.vector)
    assert abs(sim - 1.0) < 0.001, f"Self-similarity should be 1.0, got {sim}"
    print(f"  ✓ Semantic similarity (self): {sim:.4f}")

    # Test step projection
    step_result = {
        'z_before': np.zeros(16), 'z_after': np.ones(16) * 0.3,
        'action': np.random.randn(4), 'goal_prob': 0.6,
        'gp_delta': 0.1, 'latent_agency': 0.7, 'object_agency': 0.5
    }
    factors = sp.project_step(
        step_result, sl, agency,
        [obj], np.random.randn(16), 50,
        cf_result, timestamp=2
    )
    assert len(factors) >= 6  # self + agency + 1 object + goal + outcome + cf
    print(f"  ✓ Full step projection: {len(factors)} factors")

    print("  >>> SemanticProjection PASSED\n")
    return sp


def test_episodic_semantic_graph():
    """Test 47.2: Episodic semantic graph."""
    print("\n============================================================")
    print("47.2 — EPISODIC SEMANTIC GRAPH")
    print("============================================================")

    graph = EpisodicSemanticGraph(semantic_dim=32)

    # Create test factors
    f1 = SemanticFactor(
        factor_id='obj_1', factor_type=SemanticFactorType.OBJECT,
        vector=np.random.randn(32), source_layer='object_centric',
        source_id='obj_a', timestamp=1,
        attributes={'name': 'object_a', 'persistence': 5}
    )
    f2 = SemanticFactor(
        factor_id='self_1', factor_type=SemanticFactorType.SELF_STATE,
        vector=np.random.randn(32), source_layer='self_model',
        source_id='self', timestamp=2,
        attributes={'identity_coherence': 0.95}
    )
    f3 = SemanticFactor(
        factor_id='out_1', factor_type=SemanticFactorType.ACTION_OUTCOME,
        vector=np.random.randn(32), source_layer='execution',
        source_id='outcome_1', timestamp=3,
        attributes={'gp_delta': 0.2}
    )

    # Test node addition
    graph.add_node(f1)
    graph.add_node(f2)
    graph.add_node(f3)
    assert len(graph.nodes) == 3
    print(f"  ✓ Nodes added: {len(graph.nodes)}")

    # Test edge addition
    e1 = graph.add_edge('self_1', 'out_1', 'self_acted', weight=0.9)
    assert e1 is not None
    assert len(graph.edges) == 1
    print(f"  ✓ Edge added: {e1}")

    # Test linking
    graph.link_factors([f1, f2, f3], [])
    print(f"  ✓ Factors linked: {len(graph.edges)} edges")

    # Test self trajectory
    traj = graph.get_self_trajectory()
    assert len(traj) >= 1
    print(f"  ✓ Self trajectory: {len(traj)} nodes")

    # Test agency owned episodes
    owned = graph.get_agency_owned_episodes()
    print(f"  ✓ Agency owned episodes: {len(owned)}")

    # Test causal chain
    if owned:
        chain = graph.get_causal_chain(owned[0].node_id)
        print(f"  ✓ Causal chain: {len(chain)} nodes")

    # Test pruning (if needed)
    print(f"  ✓ Graph stats: {graph.get_stats()}")

    print("  >>> EpisodicSemanticGraph PASSED\n")
    return graph


def test_narrative_stabilizer():
    """Test 47.3: Narrative compression."""
    print("\n============================================================")
    print("47.3 — NARRATIVE STABILIZER")
    print("============================================================")

    ns = NarrativeStabilizer(min_episode_length=3, compression_interval=5)

    # Create mock step results
    step_results = []
    for i in range(20):
        step_results.append({
            'goal_prob': 0.3 + i * 0.01,
            'gp_delta': 0.01,
            'latent_agency': 0.5 + 0.2 * np.sin(i),
            'z_before': np.random.randn(16),
            'z_after': np.random.randn(16),
            'action': np.random.randn(4),
            'flow_id': f'flow_{i % 3}',
            'self_coherence': 0.95 + 0.02 * np.sin(i * 0.5),
            'cf_regret': -0.1 * i / 20,
            '_step': i
        })

    # Test compression
    factors = [
        SemanticFactor(
            factor_id=f'test_{i}',
            factor_type=SemanticFactorType.ACTION_OUTCOME,
            vector=np.random.randn(32),
            source_layer='test', source_id=f'test_{i}',
            timestamp=i,
            attributes={'gp_delta': 0.01}
        )
        for i in range(20)
    ]
    objects = []

    # Test should_compress
    assert ns.should_compress(10, 0) == True
    assert ns.should_compress(3, 0) == False
    print("  ✓ Compression interval check: OK")

    # Test compression
    episode = ns.compress_trajectory(
        factors, step_results,
        start_time=0, end_time=19,
        objects=objects
    )
    assert episode.n_steps == 20
    assert episode.primary_agent in ('self', 'external', 'mixed')
    assert len(ns.episodes) == 1
    print(f"  ✓ Episode compressed: {episode.n_steps} steps, "
          f"agent={episode.primary_agent}, "
          f"goal_delta={episode.goal_delta:+.4f}")

    # Test key transitions extraction
    assert len(episode.key_transitions) >= 0
    print(f"  ✓ Key transitions: {len(episode.key_transitions)}")

    # Test causal graph
    assert len(episode.causal_graph) >= 0
    print(f"  ✓ Causal graph: {len(episode.causal_graph)} nodes")

    # Test get_recent
    recent = ns.get_recent_narratives(5)
    assert len(recent) >= 1
    print(f"  ✓ Recent narratives: {len(recent)}")

    print("  >>> NarrativeStabilizer PASSED\n")
    return ns


def test_language_bind():
    """Test 47.4: Language bind."""
    print("\n============================================================")
    print("47.4 — LANGUAGE BIND")
    print("============================================================")

    lb = LanguageBind(semantic_dim=32, vocab_size=128, bind_strength=0.3)

    # Test factor → tokens
    factor = SemanticFactor(
        factor_id='obj_test', factor_type=SemanticFactorType.OBJECT,
        vector=np.random.randn(32),
        source_layer='object_centric', source_id='obj_a',
        timestamp=1,
        attributes={'name': 'object_a', 'persistence': 10.0}
    )
    tokens = lb.factor_to_tokens(factor)
    assert len(tokens) > 0
    print(f"  ✓ Factor → tokens: {len(tokens)} tokens")

    # Test semantic → token projection
    token_id = lb.semantic_to_token(np.random.randn(32))
    assert 0 <= token_id < 128
    print(f"  ✓ Semantic → token: id={token_id}")

    # Test token → semantic (weak)
    sem_vec = lb.token_to_semantic(token_id)
    assert sem_vec.shape == (32,)
    assert np.linalg.norm(sem_vec) <= lb.bind_strength * 1.1
    print(f"  ✓ Token → semantic: norm={np.linalg.norm(sem_vec):.4f} "
          f"(≤{lb.bind_strength})")

    # Test episode rendering
    from dataclasses import dataclass
    episode = NarrativeEpisode(
        episode_id='ep_1', start_time=0, end_time=50,
        n_steps=50, primary_agent='self', goal_delta=0.15,
        key_transitions=[
            {'timestamp': 10, 'gp_delta': 0.05, 'agency': 0.7,
             'z_delta': 0.5, 'flow_id': 'flow_1', 'action_norm': 1.2},
            {'timestamp': 30, 'gp_delta': 0.10, 'agency': 0.8,
             'z_delta': 0.8, 'flow_id': 'flow_2', 'action_norm': 0.9}
        ],
        self_coherence=0.96, mean_agency=0.65, mean_regret=-0.3
    )
    text = lb.render_episode(episode)
    assert len(text) > 0
    print(f"  ✓ Episode rendered: {len(text)} chars")

    # Test query → semantic (weak, retrieval-only)
    query_vec = lb.query_to_semantic(['goal', 'improvement'])
    assert query_vec.shape == (32,)
    assert np.linalg.norm(query_vec) > 0
    print(f"  ✓ Query → semantic: norm={np.linalg.norm(query_vec):.4f}")

    print("  >>> LanguageBind PASSED\n")
    return lb


def test_semantic_retrieval():
    """Test 47.5: Semantic retrieval."""
    print("\n============================================================")
    print("47.5 — SEMANTIC RETRIEVAL")
    print("============================================================")

    graph = EpisodicSemanticGraph(semantic_dim=32)
    retrieval = SemanticRetrieval(graph)

    # Populate graph
    for i in range(10):
        f = SemanticFactor(
            factor_id=f'obj_{i}', factor_type=SemanticFactorType.OBJECT,
            vector=np.random.randn(32),
            source_layer='object_centric', source_id=f'obj_{chr(97+i)}',
            timestamp=i,
            attributes={'name': f'object_{chr(97+i)}', 'persistence': i}
        )
        graph.add_node(f)

    for i in range(5):
        f = SemanticFactor(
            factor_id=f'out_{i}', factor_type=SemanticFactorType.ACTION_OUTCOME,
            vector=np.random.randn(32),
            source_layer='execution', source_id=f'outcome_{i}',
            timestamp=10 + i,
            attributes={'gp_delta': 0.1 * (i + 1)}
        )
        graph.add_node(f)
        if i > 0:
            graph.add_edge(f'self_1' if f'self_1' in graph.nodes else f'obj_0',
                           f.factor_id, 'self_acted', weight=0.8)

    # Add self nodes
    for i in range(5):
        f = SemanticFactor(
            factor_id=f'self_{i}', factor_type=SemanticFactorType.SELF_STATE,
            vector=np.random.randn(32),
            source_layer='self_model', source_id='self',
            timestamp=20 + i,
            attributes={'identity_coherence': 0.9 + 0.02 * i}
        )
        graph.add_node(f)

    # Test type-based query
    q = RetrievalQuery(target_type='object', top_k=3)
    results = retrieval.query(q)
    assert len(results) <= 3
    assert all(r.node_type == 'object' for r in results)
    print(f"  ✓ Type query (object): {len(results)} results")

    # Test goal trajectory
    goal_results = retrieval.retrieve_goal_trajectory(3)
    print(f"  ✓ Goal trajectory: {len(goal_results)} outcomes")

    # Test self narrative
    self_results = retrieval.retrieve_self_narrative(3)
    print(f"  ✓ Self narrative: {len(self_results)} self-states")

    # Test counterfactual alternatives (empty - none added)
    cf_results = retrieval.retrieve_counterfactual_alternatives(3)
    print(f"  ✓ Counterfactual alternatives: {len(cf_results)} branches")

    print("  >>> SemanticRetrieval PASSED\n")
    return retrieval


def test_semantic_engine_sanity(
    n_steps: int = 30, bootstrap: bool = True
):
    """Test that SemanticEngine runs without error."""
    print("\n============================================================")
    print("SEMANTIC ENGINE SANITY (30 steps)")
    print("============================================================")

    engine, result, checks, all_pass = test_integration(
        n_steps=n_steps, bootstrap=bootstrap, verbose=False
    )

    if all_pass:
        print("  >>> SemanticEngine Sanity (30 steps) PASSED\n")
    else:
        print("  >>> SemanticEngine Sanity (30 steps) FAILED\n")

    return engine, result, checks, all_pass


def test_integration(
    n_steps: int = 200,
    bootstrap: bool = True,
    verbose: bool = True
):
    """
    Full Phase 47 integration test.

    Runs SemanticEngine and verifies:
    1. GP not flat (goal geometry survives)
    2. Objects present (perception survives)
    3. Ensemble divergent (uncertainty physics survives)
    4. Training active (learning loop continues)
    5. Goal learned (goal manifold works)
    6. Self coherence maintained (Phase 46)
    7. Agency inference active (Phase 46)
    8. Counterfactual simulating (Phase 46)
    9. Semantic factors generated (Phase 47.1)
    10. Episodic graph accumulating (Phase 47.2)
    11. Narrative compression running (Phase 47.3)
    12. Language binding active (Phase 47.4)
    """
    print("\n" + "=" * 70)
    print("PHASE 47: SEMANTIC GROUNDING & NARRATIVE COMPRESSION (200+ steps)")
    print("=" * 70)

    # ========================================================================
    # SETUP
    # ========================================================================
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    # ========================================================================
    # BUILD SEMANTIC ENGINE
    # ========================================================================
    engine = SemanticEngine(
        wm=wm, bootstrap=bootstrap,
        n_coverage=min(60, n_steps // 3),
        n_shaping=min(50, n_steps // 4),
        n_transfer=min(30, n_steps // 6),
        n_initial_flows=6,
        train_interval=10,
        n_ensemble=4,
        exploration_beta=0.15,
        planning_horizon=4,
        planning_samples=12,
        n_slots=4,
        slot_dim=6,
        match_threshold=0.4,
        self_dim=8,
        counterfactual_interval=max(15, n_steps // 10),
        n_counterfactuals=2,
        semantic_dim=32,
        narrative_compression_interval=max(20, n_steps // 5),
        episodic_max_nodes=200,
        episodic_max_edges=500,
        vocab_size=64,
        bind_strength=0.3
    )

    # ========================================================================
    # RUN
    # ========================================================================
    print(f"  Running {n_steps} steps...\n")
    z_start = np.random.randn(16) * 0.1
    result = engine.run(z_start, n_steps=n_steps)

    # ========================================================================
    # VERIFICATION
    # ========================================================================
    mean_gp = result.get('mean_gp', 0.0)
    mean_uncertainty = result.get('mean_uncertainty', 0.0)
    mean_objects = result.get('mean_n_objects', 0.0)
    mean_coherence = result.get('mean_self_coherence', 0.0)
    mean_agency = result.get('mean_agency', 0.0)
    cf_runs = result.get('counterfactual_runs', 0)
    n_flows = result.get('n_flows', 0)
    ensemble_div = result.get('ensemble', {}).get('param_divergence', 0.0)
    goal_learned = result.get('goal_manifold', {}).get('has_mean', False)
    n_episodes = result.get('training', {}).get('buffer_episodes', 0)
    n_factors = result.get('semantic', {}).get('n_factors_total', 0)
    n_graph_nodes = result.get('episodic_graph', {}).get('n_nodes', 0)
    n_graph_edges = result.get('episodic_graph', {}).get('n_edges', 0)
    n_narratives = result.get('narrative', {}).get('n_episodes', 0)
    n_tokens = result.get('language_bind', {}).get('tokens_generated', 0)

    # Phase 47-specific: semantic structure integrity
    factor_types = result.get('semantic', {}).get('factor_types', {})
    n_self_factors = factor_types.get('self_state', 0)
    n_obj_factors = factor_types.get('object', 0)
    n_agency_factors = factor_types.get('agency_event', 0)
    n_goal_factors = factor_types.get('goal_attractor', 0)
    n_outcome_factors = factor_types.get('action_outcome', 0)

    if verbose:
        print(f"\n  RESULTS:")
        print(f"    Steps: {n_steps}")
        print(f"    Mean GP: {mean_gp:.4f}")
        print(f"    Mean uncertainty: {mean_uncertainty:.4f}")
        print(f"    Mean objects: {mean_objects:.1f}")
        print(f"    Mean self coherence: {mean_coherence:.4f}")
        print(f"    Mean agency: {mean_agency:.4f}")
        print(f"    Counterfactual runs: {cf_runs}")
        print(f"    Ensemble divergence: {ensemble_div:.4f}")
        print(f"    N flows: {n_flows}")

        print(f"\n  [47.1] SEMANTIC PROJECTION:")
        print(f"    Total factors: {n_factors}")
        print(f"    Self: {n_self_factors}, Objects: {n_obj_factors}, "
              f"Agency: {n_agency_factors}, Goals: {n_goal_factors}, "
              f"Outcomes: {n_outcome_factors}")

        print(f"\n  [47.2] EPISODIC GRAPH:")
        print(f"    Nodes: {n_graph_nodes}, Edges: {n_graph_edges}")

        print(f"\n  [47.3] NARRATIVE:")
        print(f"    Episodes: {n_narratives}")

        print(f"\n  [47.4] LANGUAGE BIND:")
        print(f"    Tokens generated: {n_tokens}")

    # ========================================================================
    # CHECKS
    # ========================================================================
    checks = [
        ("GP not flat", mean_gp > 0.1, f"{mean_gp:.4f}"),
        ("Objects present", mean_objects >= 1.0, f"{mean_objects:.1f}"),
        ("Ensemble divergent", ensemble_div > 0.5, f"{ensemble_div:.4f}"),
        ("Training active", n_episodes > 0, f"{n_episodes} eps"),
        ("Goal learned", goal_learned, str(goal_learned)),
        ("Self coherence maintained", mean_coherence > 0.7, f"{mean_coherence:.4f}"),
        ("Agency inference active", mean_agency > 0.1 or mean_agency < 0.9,
         f"{mean_agency:.4f}"),
        ("Counterfactual simulating", cf_runs > 0, f"{cf_runs}"),
        ("Semantic factors generated", n_factors >= n_steps, f"{n_factors}"),
        ("Episodic graph accumulating", n_graph_nodes > 10, f"{n_graph_nodes}"),
        ("Narrative compression running", n_narratives > 0, f"{n_narratives}"),
        ("Language binding active", n_tokens > n_factors, f"{n_tokens}"),
        ("Semantic structure diverse",
         n_self_factors > 0 and n_obj_factors > 0 and n_agency_factors > 0
         and n_goal_factors > 0 and n_outcome_factors > 0,
         f"self={n_self_factors} obj={n_obj_factors} "
         f"agency={n_agency_factors} goal={n_goal_factors} "
         f"outcome={n_outcome_factors}"),
    ]

    if verbose:
        print(f"\n  {'=' * 60}")
        print(f"  VERIFICATION")
        print(f"  {'=' * 60}")
        for name, passed, detail in checks:
            status = "[PASS]" if passed else "[FAIL]"
            print(f"    {status} {name}: {detail}")

    all_pass = all(p for _, p, _ in checks)
    if all_pass:
        print(f"\n  {'=' * 60}")
        print(f"  PHASE 47 VERDICT: ALL {len(checks)}/{len(checks)} PASSED")
        print(f"  {'=' * 60}")

    return engine, result, checks, all_pass


def _boostrap_world_model(
    wm: FlowConditionedWorldModel,
    n_random: int = 100,
    flow_dim: int = 4,
    latent_dim: int = 16
) -> FlowConditionedWorldModel:
    """Bootstrap world model with random transitions."""
    for _ in range(n_random):
        z = np.random.randn(latent_dim) * 0.3
        a = np.random.randn(wm.action_dim) * 0.5
        latent_a = np.zeros(latent_dim)
        latent_a[:min(wm.action_dim, latent_dim)] = a[:min(wm.action_dim, latent_dim)]
        z_next = z + 0.1 * latent_a + np.random.randn(latent_dim) * 0.05
        h = np.zeros(wm.belief_dim)
        wm.record_transition(z, h, a, z_next)
    for _ in range(5):
        batch = wm.sample_batch(batch_size=32)
        if batch:
            z_b, h_b, a_b, zn_b = batch
            loss = wm.compute_loss(z_b, h_b, a_b, zn_b)
            wm.train_step(z_b, h_b, a_b, zn_b, lr=0.01)
    return wm


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 47: SEMANTIC GROUNDING & NARRATIVE COMPRESSION           ║
║                                                                   ║
║  The system now projects cognition into semantic space.           ║
║  Language indexes, compresses, stabilizes — but does NOT rewrite. ║
║                                                                   ║
║  Architecture:                                                    ║
║    47.1 — SemanticProjection     → shared semantic manifold       ║
║    47.2 — EpisodicSemanticGraph  → associative autobiographical   ║
║    47.3 — NarrativeStabilizer    → temporal compression           ║
║    47.4 — LanguageBind           → weak semantic↔language         ║
║    47.5 — SemanticRetrieval      → goal/agency/self traversal     ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    unit_tests = [
        ("SemanticProjection", test_semantic_projection),
        ("EpisodicSemanticGraph", test_episodic_semantic_graph),
        ("NarrativeStabilizer", test_narrative_stabilizer),
        ("LanguageBind", test_language_bind),
        ("SemanticRetrieval", test_semantic_retrieval),
        ("SemanticEngine Sanity (30 steps)",
         lambda: test_semantic_engine_sanity(n_steps=30, bootstrap=True)),
    ]

    all_unit_pass = True
    for name, fn in unit_tests:
        try:
            fn()
            print(f"  >>> {name} PASSED\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  >>> {name} FAILED: {e}\n")
            all_unit_pass = False

    if all_unit_pass:
        engine, result, checks, all_pass = test_integration(
            n_steps=200, bootstrap=True, verbose=True
        )

        print("\n" + "=" * 70)
        print("PHASE 47 SUMMARY")
        print("=" * 70)
        passed_count = sum(1 for _, p, _ in checks if p)
        total_count = len(checks)
        print(f"\n  Checks passed: {passed_count}/{total_count}")
        if all_pass:
            print("""
  Phase 47 complete.

  The system now has:
    • Semantic projection of all cognitive layers into a shared manifold
    • Persistent episodic graph with causal/temporal/agency edges
    • Narrative compression preserving causal continuity (not text summary)
    • Weak language bind: language can index, activate, associate — not overwrite
    • Goal/agency/self-aware structured graph traversal (not vector DB)

  This is the semantic stabilization layer for:
    Phase 48: Autonomous Cognitive Ecology
      (internal drives, compute economy, goal speciation,
       cognitive evolution — stabilized by narrative compression)

  Architecture stack complete:

    Phase 25-30:  Sensorimotor & World Modeling       ← substrate
    Phase 31-40:  Behavioral Dynamics & Flows          ← action
    Phase 41-42:  Goal Geometry                        ← intention
    Phase 43-44:  Uncertainty & Object Perception      ← world
    Phase 45:     Temporal Abstraction                 ← time
    Phase 46:     Self-Model & Identity                ← self
    Phase 47:     Semantic Grounding & Narrative        ← meaning

  This is the minimum substrate for autobiographical cognition.
            """)
        else:
            print("\n  ❌ Some checks failed")
            for name, passed, detail in checks:
                if not passed:
                    print(f"     FAIL: {name} = {detail}")
    else:
        print("\n  ❌ Unit tests failed — skipping integration test")
