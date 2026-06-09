"""
Phase 37 — Controllability Topology

KEY SHIFT:
  Phase 35-36: manifold organized by BEHAVIORAL SIMILARITY
    "two flows are similar if they produce similar trajectories"
    
  Phase 37:    manifold organized by CONTROLLABILITY SIMILARITY
    "two flows are similar if they can reach the same states"
    
  This answers: "which flow should I use to reach my goal?"

WHAT CHANGES:
  1. Controllability matrix: [n_flows × n_probes × n_probes]
     flow π_k from probe state z_i reaches z_j with probability p_ij(k)
     
  2. Reachability graph: directed graph
     z_i --[π_k]→ z_j if flow π_k can reach z_j from z_i
     
  3. Controllability manifold: flows organized by where they can GO
     Not by what TRAJECTORIES they produce
     
  4. Goal-directed flow selection: 
     P(reach goal | start state, flow π_k)
     
  Why this matters:
    Behavioral similarity (Phase 35) → similar motions
    Controllability similarity (Phase 37) → similar destinations
    
    For goal-reaching, we care about DESTINATIONS, not motions.

ARCHITECTURAL INTEGRATION:
  Phase 36 buffer → probe states → controllability matrix →
  manifold reorganization → better flow selection → 
  higher goal probability → Phase 36 training (improved)
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from phase30_training_loop import MinimalWorldModel
from phase31_hierarchical_execution import GoalAttractor
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, FlowType, PointFlow, LimitCycleFlow,
    ComposedFlow, rollout_flow, evaluate_flow_goal
)


# ============================================================================
# 1. CONTROLLABILITY MATRIX
# ============================================================================

class ControllabilityMatrix:
    """
    [n_flows × n_probes] matrix: for each flow, what's its reachability
    profile from each probe state?
    
    reachability_profile[flow_idx, probe_idx] = probability the flow
    reaches the goal region within n_steps.
    
    This is built by rolling out each flow from each probe state
    and measuring where it ends up.
    """
    
    def __init__(self, n_probes: int = 50, n_steps: int = 10,
                 goal_radius: float = 0.5):
        self.n_probes = n_probes
        self.n_steps = n_steps
        self.goal_radius = goal_radius
        
        # Probe states (sampled from trajectory buffer)
        self.probe_states: List[np.ndarray] = []
        
        # Controllability profiles: dict[flow_id] → 
        #   { 'reachability': np.array([n_probes, n_probes]),
        #     'goal_prob': np.array([n_probes]) }
        self.profiles: Dict[str, Dict] = {}
        
        # Flow-to-region mapping: flow π → which states are reachable
        self.flow_region_maps: Dict[str, np.ndarray] = {}
        
        # Empirical profiles from trajectory buffer data
        self.empirical_goal_probs: Dict[str, List[float]] = defaultdict(list)
        self.empirical_transitions: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        
        # Last updated timestamps
        self.last_update: Dict[str, int] = {}
    
    def update_probes(self, states: List[np.ndarray], max_probes: int = 50):
        """Update probe states from trajectory data."""
        combined = self.probe_states + states
        
        # Subsample to maintain max_probes
        if len(combined) > max_probes:
            combined = random.sample(combined, max_probes)
        
        self.probe_states = [s.copy() for s in combined]
        self.n_probes = len(self.probe_states)
    
    def compute_flow_reachability(
        self,
        flow: SkillFlow,
        world_model: MinimalWorldModel,
        goal: GoalAttractor
    ) -> Dict:
        """
        Compute reachability and goal-probability profile for a flow.
        
        For each probe state:
          1. Rollout flow for n_steps
          2. Measure final distance to goal
          3. Record: does it reach goal? does it reach each other probe?
        
        Returns: {reachability, goal_prob, endpoint_dists}
        """
        if not self.probe_states:
            return {
                'reachability': np.zeros((0, 0)),
                'goal_prob': np.array([]),
                'endpoints': []
            }
        
        n_probes = len(self.probe_states)
        reachability = np.zeros((n_probes, n_probes))
        goal_probs = np.zeros(n_probes)
        endpoints = []
        
        for i, z_start in enumerate(self.probe_states):
            h = np.zeros(world_model.belief_dim)
            
            result = rollout_flow(flow, z_start, h, world_model, self.n_steps)
            z_final = result['final_state']
            endpoints.append(z_final.copy())
            
            # Goal probability
            dist = np.linalg.norm(
                z_final - goal.attractor_state[:len(z_final)]
            )
            goal_prob = np.exp(-dist)
            goal_probs[i] = goal_prob
            
            # Reachability to other probes
            for j, z_target in enumerate(self.probe_states):
                dist_ij = np.linalg.norm(z_final - z_target)
                reachability[i, j] = np.exp(-dist_ij)
        
        return {
            'reachability': reachability,
            'goal_prob': goal_probs,
            'endpoints': endpoints
        }
    
    def compute_flow_reachability_fast(
        self,
        flow: SkillFlow,
        world_model: MinimalWorldModel,
        n_rollouts: int = 10
    ) -> np.ndarray:
        """
        Faster reachability estimate: rollout from n_rollouts random probes.
        Returns: [n_rollouts] dimensional feature vector.
        """
        if not self.probe_states:
            return np.zeros(1)
        
        n_used = min(n_rollouts, len(self.probe_states))
        probe_indices = random.sample(range(len(self.probe_states)), n_used)
        
        features = []
        for idx in probe_indices:
            z_start = self.probe_states[idx]
            h = np.zeros(world_model.belief_dim)
            
            result = rollout_flow(flow, z_start, h, world_model, self.n_steps)
            z_final = result['final_state']
            
            # Feature: distance change (negative = moved closer to goal)
            dist_before = np.linalg.norm(z_start)
            dist_after = np.linalg.norm(z_final)
            features.append(dist_before - dist_after)
            
            # Feature: final state norm
            features.append(np.linalg.norm(z_final))
            
            # Feature: stability
            features.append(flow.stability)
            
            # Feature: action variability along trajectory
            if result['actions'] is not None and len(result['actions']) > 1:
                action_var = float(np.mean(np.var(result['actions'], axis=0)))
            else:
                action_var = 0.0
            features.append(action_var)
        
        return np.array(features)
    
    def add_empirical_transition(
        self, flow_id: str,
        z_before: np.ndarray, z_after: np.ndarray,
        goal_prob: float
    ):
        """Add empirical transition from trajectory buffer."""
        self.empirical_transitions[flow_id].append(
            (z_before.copy(), z_after.copy())
        )
        self.empirical_goal_probs[flow_id].append(goal_prob)
        
        # Keep bounded
        max_empirical = 200
        if len(self.empirical_transitions[flow_id]) > max_empirical:
            self.empirical_transitions[flow_id] = \
                self.empirical_transitions[flow_id][-max_empirical:]
            self.empirical_goal_probs[flow_id] = \
                self.empirical_goal_probs[flow_id][-max_empirical:]
    
    def get_empirical_goal_prob(self, flow_id: str) -> float:
        """Mean goal prob from empirical trajectory data."""
        gps = self.empirical_goal_probs.get(flow_id, [])
        return float(np.mean(gps)) if gps else 0.0
    
    def get_empirical_delta(self, flow_id: str) -> float:
        """
        Mean distance-to-goal delta from empirical data.
        Positive = moves toward goal on average.
        """
        transitions = self.empirical_transitions.get(flow_id, [])
        if len(transitions) < 2:
            return 0.0
        
        deltas = []
        for z_before, z_after in transitions:
            d_before = np.linalg.norm(z_before)
            d_after = np.linalg.norm(z_after)
            deltas.append(d_before - d_after)
        
        return float(np.mean(deltas))
    
    def update_profile(
        self,
        flow_id: str,
        flow: SkillFlow,
        world_model: MinimalWorldModel,
        goal: GoalAttractor,
        step: int = 0,
        blend_empirical: float = 0.5
    ):
        """
        Update controllability profile for a flow.
        
        Blends model-based rollouts with empirical data:
          final_goal_prob = (1 - blend) * model_goal_prob + blend * empirical_goal_prob
        
        This gives meaningful signal even with a random world model.
        """
        if not self.probe_states:
            return
        
        result = self.compute_flow_reachability(flow, world_model, goal)
        model_goal_prob = float(np.mean(result['goal_prob']))
        
        # Blend with empirical data
        empirical_gp = self.get_empirical_goal_prob(flow_id)
        n_empirical = len(self.empirical_transitions.get(flow_id, []))
        
        # More empirical data → higher blend weight
        effective_blend = min(blend_empirical, n_empirical / 20.0)
        blended_gp = (1 - effective_blend) * model_goal_prob + effective_blend * empirical_gp
        
        # Controllability-weighted goal prob (reflects reachability, not just an endpoint)
        delta = self.get_empirical_delta(flow_id)
        controllability_gp = blended_gp * (1.0 + max(0, delta))
        
        self.profiles[flow_id] = {
            'reachability': result['reachability'],
            'goal_prob': result['goal_prob'],
            'endpoints': result['endpoints'],
            'mean_goal_prob': float(controllability_gp),
            'model_goal_prob': float(model_goal_prob),
            'empirical_goal_prob': float(empirical_gp),
            'empirical_delta': float(delta),
            'n_empirical': n_empirical,
            'flow_type': flow.flow_type.value
        }
        
        self.last_update[flow_id] = step
    
    def compute_controllability_similarity(
        self, flow_id_1: str, flow_id_2: str
    ) -> float:
        """
        Two flows have similar controllability if they have similar
        reachability profiles — i.e., they can reach the same regions
        from the same starting states.
        
        Similarity = mean over probes of: how similar are goal_probs?
        """
        if flow_id_1 not in self.profiles or flow_id_2 not in self.profiles:
            return 0.0
        
        p1 = self.profiles[flow_id_1]
        p2 = self.profiles[flow_id_2]
        
        # Compare goal probability distributions
        goal_sim = 1.0 / (1.0 + float(
            np.mean((p1['goal_prob'] - p2['goal_prob']) ** 2)
        ))
        
        # Compare reachability matrices
        r1 = p1['reachability']
        r2 = p2['reachability']
        
        if r1.size > 0 and r2.size > 0 and r1.shape == r2.shape:
            reach_sim = 1.0 / (1.0 + float(np.mean((r1 - r2) ** 2)))
        else:
            reach_sim = 0.0
        
        return 0.5 * goal_sim + 0.5 * reach_sim
    
    def get_goal_reaching_flows(
        self, goal: GoalAttractor, threshold: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Rank flows by their goal-reaching probability.
        
        Returns: [(flow_id, mean_goal_prob), ...] sorted descending.
        """
        ranked = []
        for fid, prof in self.profiles.items():
            ranked.append((fid, prof['mean_goal_prob']))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def get_stats(self) -> Dict:
        """Controllability statistics."""
        if not self.profiles:
            return {
                'n_flows': 0,
                'n_probes': len(self.probe_states),
                'mean_goal_prob': 0.0,
                'max_goal_prob': 0.0,
                'n_goal_reaching': 0
            }
        
        goal_probs = [
            p['mean_goal_prob'] for p in self.profiles.values()
        ]
        
        return {
            'n_flows': len(self.profiles),
            'n_probes': len(self.probe_states),
            'mean_goal_prob': float(np.mean(goal_probs)),
            'max_goal_prob': float(max(goal_probs)),
            'n_goal_reaching': sum(1 for g in goal_probs if g > 0.1)
        }


# ============================================================================
# 2. REACHABILITY GRAPH
# ============================================================================

@dataclass
class ReachabilityEdge:
    """Directed edge in reachability graph."""
    source_idx: int
    target_idx: int
    flow_id: str
    probability: float
    n_steps: int


class ReachabilityGraph:
    """
    Directed graph of controllable state transitions.
    
    Nodes: probe states from trajectory buffer
    Edges: flow π can reach state j from state i with probability p
    
    This enables:
      - Path planning: which flow sequence reaches the goal?
      - Coverage analysis: which states are reachable?
      - Bottleneck detection: which states are hard to reach?
    """
    
    def __init__(self):
        # Adjacency: list of edges per state index
        self.outgoing: Dict[int, List[ReachabilityEdge]] = defaultdict(list)
        self.incoming: Dict[int, List[ReachabilityEdge]] = defaultdict(list)
        
        # State labels
        self.state_labels: Dict[int, str] = {}
        
        # Which flows exist in graph
        self.flow_ids: Set[str] = set()
        
        self.n_edges = 0
    
    def build_from_matrix(
        self,
        profiles: Dict[str, Dict],
        probe_states: List[np.ndarray],
        threshold: float = 0.3
    ):
        """
        Build reachability graph from controllability profiles.
        
        For each flow and each pair of probes (i, j):
          if reachability[i, j] > threshold:
            add edge: probe_i → probe_j via flow
        """
        self.outgoing.clear()
        self.incoming.clear()
        self.flow_ids.clear()
        
        for flow_id, prof in profiles.items():
            reachability = prof['reachability']
            n_probes = reachability.shape[0]
            
            for i in range(n_probes):
                for j in range(n_probes):
                    if i == j:
                        continue
                    
                    prob = float(reachability[i, j])
                    if prob > threshold:
                        edge = ReachabilityEdge(
                            source_idx=i,
                            target_idx=j,
                            flow_id=flow_id,
                            probability=prob,
                            n_steps=self._estimate_steps(
                                profiles, flow_id, i, j
                            )
                        )
                        self.outgoing[i].append(edge)
                        self.incoming[j].append(edge)
                        self.flow_ids.add(flow_id)
                        self.n_edges += 1
        
        self.probe_states = probe_states
    
    def _estimate_steps(
        self, profiles: Dict, flow_id: str,
        source_idx: int, target_idx: int
    ) -> int:
        """Estimate steps needed for transition."""
        return 5  # Simplified: assume 5 steps
    
    def find_path_to_goal(
        self,
        z_current: np.ndarray,
        goal: GoalAttractor,
        probe_states: List[np.ndarray],
        n_lookahead: int = 3
    ) -> List[str]:
        """
        BFS to find flow sequence that reaches goal region.
        
        Returns: [flow_id_1, flow_id_2, ...] or empty if no path.
        """
        if not self.outgoing:
            return []
        
        # Find nearest probe state
        if not probe_states:
            return []
        
        dists = [np.linalg.norm(z_current - s) for s in probe_states]
        start_idx = int(np.argmin(dists))
        
        # BFS over reachability graph
        visited = {start_idx}
        queue = [(start_idx, [])]
        
        while queue:
            node_idx, path = queue.pop(0)
            
            # Check if any edge from this node goes near goal
            for edge in self.outgoing.get(node_idx, []):
                if edge.target_idx in visited:
                    continue
                
                target_state = probe_states[edge.target_idx]
                goal_dist = np.linalg.norm(
                    target_state - goal.attractor_state[:len(target_state)]
                )
                
                new_path = path + [edge.flow_id]
                
                if goal_dist < 2.0:  # Near goal region
                    return new_path
                
                if len(new_path) < n_lookahead:
                    visited.add(edge.target_idx)
                    queue.append((edge.target_idx, new_path))
        
        return []
    
    def compute_coverage(self) -> float:
        """What fraction of probe states have outgoing edges?"""
        if not self.outgoing and not self.incoming:
            return 0.0
        
        n_with_outgoing = len(self.outgoing)
        n_probes = max(
            max(list(self.outgoing.keys()) + [0]),
            max(list(self.incoming.keys()) + [0])
        ) + 1 if (self.outgoing or self.incoming) else 0
        
        return n_with_outgoing / max(1, n_probes)
    
    def find_bottlenecks(self) -> List[int]:
        """
        Find bottleneck states (high incoming but low outgoing).
        These are states that are easy to reach but hard to leave.
        """
        bottlenecks = []
        for node_idx in set(list(self.outgoing.keys()) + list(self.incoming.keys())):
            n_in = len(self.incoming.get(node_idx, []))
            n_out = len(self.outgoing.get(node_idx, []))
            
            if n_in > 0 and n_out == 0:
                bottlenecks.append(node_idx)
        
        return bottlenecks
    
    def get_stats(self) -> Dict:
        """Graph statistics."""
        return {
            'n_edges': self.n_edges,
            'n_flows': len(self.flow_ids),
            'n_source_states': len(self.outgoing),
            'n_target_states': len(self.incoming),
            'coverage': self.compute_coverage(),
            'n_bottlenecks': len(self.find_bottlenecks())
        }


# ============================================================================
# 3. CONTROLLABILITY ORGANIZER
# ============================================================================

class ControllabilityOrganizer:
    """
    Reorganizes the flow manifold by controllability similarity.
    
    The Phase 35 FlowManifold organizes by behavioral/trajectory similarity:
      - Two flows are close if they produce similar trajectories
    
    This organizer replaces that with controllability similarity:
      - Two flows are close if they can reach similar regions
    
    Integration:
      Phase 36 buffer → extract probe states →
      ControllabilityMatrix → compute profiles →
      ControllabilityOrganizer → update FlowManifold coordinates →
      Phase 35 CEM uses updated manifold
    """
    
    def __init__(
        self,
        manifold: FlowManifold,
        world_model: MinimalWorldModel,
        goal: GoalAttractor,
        n_probes: int = 30,
        n_steps: int = 8,
        update_interval: int = 5
    ):
        self.manifold = manifold
        self.wm = world_model
        self.goal = goal
        
        self.matrix = ControllabilityMatrix(
            n_probes=n_probes, n_steps=n_steps,
            goal_radius=2.0
        )
        
        self.graph = ReachabilityGraph()
        self.update_interval = update_interval
        self.step_count = 0
        self.n_updates = 0
    
    def add_trajectory_states(self, states: List[np.ndarray]):
        """Add states from trajectory buffer as probe candidates."""
        self.matrix.update_probes(states)
    
    def add_episode_data(self, flow_ids: List[str], states: List[np.ndarray],
                         goal_probs: List[float], goal: GoalAttractor):
        """
        Add empirical trajectory data from a complete episode.
        
        For each step where flow_id is known, record the transition
        and its goal probability in the controllability matrix.
        """
        for t in range(min(len(states) - 1, len(flow_ids), len(goal_probs))):
            fid = flow_ids[t]
            z_before = states[t]
            z_after = states[t + 1]
            gp = goal_probs[t]
            
            self.matrix.add_empirical_transition(fid, z_before, z_after, gp)
    
    def organize(self) -> Dict:
        """
        Main organization step:
        
        1. Update controllability matrix for all flows
        2. Build reachability graph
        3. Reorganize manifold coordinates by controllability similarity
        4. Return metrics
        """
        if not self.manifold.flows:
            return {'n_organized': 0, 'status': 'no_flows'}
        
        if not self.matrix.probe_states:
            return {'n_organized': 0, 'status': 'no_probes'}
        
        self.step_count += 1
        
        # 1. Update profiles for all flows
        flow_ids = list(self.manifold.flows.keys())
        for fid in flow_ids:
            flow = self.manifold.flows[fid]
            self.matrix.update_profile(
                fid, flow, self.wm, self.goal, self.step_count
            )
        
        # 2. Build reachability graph
        self.graph.build_from_matrix(
            self.matrix.profiles,
            self.matrix.probe_states
        )
        
        # 3. Reorganize manifold
        self._reorganize_manifold(flow_ids)
        
        self.n_updates += 1
        
        return self.get_report()
    
    def _reorganize_manifold(self, flow_ids: List[str]):
        """
        Move flows with similar controllability closer together.
        
        Uses controllability similarity (where flows can reach)
        instead of behavioral similarity (what trajectories look like).
        """
        if len(flow_ids) < 2:
            return
        
        n = len(flow_ids)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self.matrix.compute_controllability_similarity(
                    flow_ids[i], flow_ids[j]
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
            sim_matrix[i, i] = 1.0
        
        # MDS-style update: similar controllability → close on manifold
        flow_dim = self.manifold.flow_dim
        for i in range(n):
            weighted_disp = np.zeros(flow_dim)
            total_weight = 0.0
            
            for j in range(n):
                if i == j:
                    continue
                
                target_dist = 1.0 - sim_matrix[i, j]
                current_dist = np.linalg.norm(
                    self.manifold.flow_coords[flow_ids[i]] -
                    self.manifold.flow_coords[flow_ids[j]]
                )
                
                if current_dist > 1e-8:
                    direction = (
                        self.manifold.flow_coords[flow_ids[j]] -
                        self.manifold.flow_coords[flow_ids[i]]
                    ) / current_dist
                    
                    displacement = direction * (current_dist - target_dist * 3.0)
                    w = sim_matrix[i, j] ** 2
                    
                    weighted_disp += w * displacement
                    total_weight += w
            
            if total_weight > 0:
                self.manifold.flow_coords[flow_ids[i]] += (
                    0.1 * weighted_disp / total_weight
                )
    
    def select_flow_for_goal(
        self, z_start: np.ndarray
    ) -> Tuple[Optional[str], float]:
        """
        Select the best flow for reaching goal from current state.
        
        Uses controllability profiles to choose flow with
        highest probability of reaching goal region.
        """
        if not self.matrix.profiles or not self.matrix.probe_states:
            return None, 0.0
        
        # Find nearest probe state
        dists = [np.linalg.norm(z_start - s) for s in self.matrix.probe_states]
        nearest_idx = int(np.argmin(dists))
        
        # Rank flows by goal probability from this probe
        ranked = []
        for fid, prof in self.matrix.profiles.items():
            gp = float(prof['goal_prob'][nearest_idx]) if nearest_idx < len(prof['goal_prob']) else 0.0
            ranked.append((fid, gp))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if ranked:
            return ranked[0]
        return None, 0.0
    
    def get_report(self) -> Dict:
        """Organization report."""
        matrix_stats = self.matrix.get_stats()
        graph_stats = self.graph.get_stats()
        
        return {
            'n_updates': self.n_updates,
            'n_probes': len(self.matrix.probe_states),
            'n_profiled_flows': matrix_stats['n_flows'],
            'mean_goal_prob': matrix_stats['mean_goal_prob'],
            'max_goal_prob': matrix_stats['max_goal_prob'],
            'graph_edges': graph_stats['n_edges'],
            'graph_coverage': graph_stats['coverage'],
            'graph_bottlenecks': graph_stats['n_bottlenecks']
        }


# ============================================================================
# 4. CONTROLLABILITY-AWARE FLOW CEM
# ============================================================================

class ControllabilityAwareCEM:
    """
    CEM planner that uses controllability topology for flow selection.
    
    Key difference from Phase 35 FlowCEM:
      Phase 35: samples random manifold coordinates,
                evaluates via rollout through world model
      
      Phase 37: uses reachability graph to FILTER candidate flows,
                then evaluates the most promising ones
                with controllability-weighted goal probability
    
    This turns CEM from blind sampling into informed search.
    """
    
    def __init__(
        self,
        world_model: MinimalWorldModel,
        manifold: FlowManifold,
        organizer: ControllabilityOrganizer,
        goal: GoalAttractor,
        n_candidates: int = 40,
        n_elites: int = 8,
        n_iterations: int = 4,
        controllability_bonus: float = 0.4,
        rollout_steps: int = 6
    ):
        self.wm = world_model
        self.manifold = manifold
        self.organizer = organizer
        self.goal = goal
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iterations = n_iterations
        self.controllability_bonus = controllability_bonus
        self.rollout_steps = rollout_steps
        
        self.mean = np.zeros(manifold.flow_dim)
        self.std = np.ones(manifold.flow_dim)
    
    def plan_flow(
        self, z_start: np.ndarray, h_start: np.ndarray
    ) -> Dict:
        """
        Plan flow with controllability-aware selection.
        
        1. Get controllability-ranked flows from organizer
        2. Use top flows to bias CEM sampling
        3. Evaluate candidates through world model rollout
        4. Select best flow
        """
        if not self.manifold.flows:
            return {'coord': np.zeros(self.manifold.flow_dim),
                    'flow': None, 'score': -np.inf}
        
        # Get controllability-ranked flows
        ranked_flows = self.organizer.matrix.get_goal_reaching_flows(
            self.goal
        ) if self.organizer.matrix.profiles else []
        
        # Find coordinates of top controllability flows
        top_coords = []
        for fid, _ in ranked_flows[:5]:
            if fid in self.manifold.flow_coords:
                top_coords.append(self.manifold.flow_coords[fid])
        
        # Initialize mean near top controllability flows
        if top_coords:
            self.mean = np.mean(top_coords, axis=0)
            self.std = np.ones(self.manifold.flow_dim) * 0.5
        else:
            self.mean = np.zeros(self.manifold.flow_dim)
            self.std = np.ones(self.manifold.flow_dim)
        
        best_coord = None
        best_score = -np.inf
        best_trajectory = None
        
        for iteration in range(self.n_iterations):
            candidates = []
            for _ in range(self.n_candidates):
                coord = self.mean + self.std * np.random.randn(self.manifold.flow_dim)
                candidates.append(coord)
            
            scores = []
            trajectories = []
            for coord in candidates:
                flow = self.manifold.interpolate_at(coord, self.wm, z_start)
                result = rollout_flow(flow, z_start, h_start, self.wm, self.rollout_steps)
                
                z_final = result['final_state']
                dist = np.linalg.norm(z_final - self.goal.attractor_state[:len(z_final)])
                goal_prob = np.exp(-dist)
                
                # Controllability bonus: how well do the constituent flows
                # reach the goal region?
                controllability_score = self._compute_controllability_score(flow)
                
                avg_uncertainty = np.mean(result['uncertainties']) if result['uncertainties'] else 0.0
                
                score = (goal_prob
                         + self.controllability_bonus * controllability_score
                         - 0.1 * avg_uncertainty)
                
                scores.append(score)
                trajectories.append(result)
            
            # Select elites
            elite_indices = np.argsort(scores)[-self.n_elites:]
            elite_coords = [candidates[i] for i in elite_indices]
            
            if elite_coords:
                self.mean = np.mean(elite_coords, axis=0)
                self.std = np.std(elite_coords, axis=0) + 0.1
            
            max_idx = int(np.argmax(scores))
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_coord = candidates[max_idx]
                best_trajectory = trajectories[max_idx]
        
        if best_coord is None:
            return {'coord': np.zeros(self.manifold.flow_dim),
                    'flow': None, 'score': -np.inf}
        
        best_flow = self.manifold.interpolate_at(best_coord, self.wm, z_start)
        
        return {
            'coord': best_coord,
            'flow': best_flow,
            'score': float(best_score),
            'trajectory': best_trajectory,
            'n_controllability_flows': len(ranked_flows),
            'top_controllability': ranked_flows[:3] if ranked_flows else []
        }
    
    def _compute_controllability_score(self, flow: SkillFlow) -> float:
        """
        Score how well this flow reaches goal region.
        
        For ComposedFlow, average the controllability of sub-flows.
        """
        if not self.organizer.matrix.profiles:
            return 0.0
        
        if hasattr(flow, 'flows') and flow.flows:
            scores = []
            for sub_flow in flow.flows:
                fid = sub_flow.flow_id
                if fid in self.organizer.matrix.profiles:
                    scores.append(
                        self.organizer.matrix.profiles[fid]['mean_goal_prob']
                    )
            return float(np.mean(scores)) if scores else 0.0
        
        # Single flow — look up by flow_id
        if flow.flow_id in self.organizer.matrix.profiles:
            return float(
                self.organizer.matrix.profiles[flow.flow_id]['mean_goal_prob']
            )
        
        return 0.0


# ============================================================================
# 5. CONTROLLABILITY-ENHANCED ENGINE
# ============================================================================

class ControllabilityEnhancedEngine:
    """
    Full execution engine with controllability topology.
    
    Extends Phase 36 ClosedLoopEngine with:
      Phase 37: ControllabilityOrganizer
      Phase 37: ControllabilityAwareCEM
    
    Loop per cycle:
      1. Execute with controllability-aware CEM
      2. Record trajectories to buffer
      3. Update controllability matrix from buffer
      4. Reorganize manifold by controllability
      5. Train world model (Phase 36)
    """
    
    def __init__(
        self,
        world_model: 'FlowConditionedWorldModel',
        goal: GoalAttractor,
        manifold: Optional[FlowManifold] = None,
        n_initial_flows: int = 12,
        flow_dim: int = 4,
        n_probes: int = 30,
        train_every_n: int = 5
    ):
        self.wm = world_model
        self.goal = goal
        self.flow_dim = flow_dim
        
        # Phase 35 manifold
        self.manifold = manifold or FlowManifold(flow_dim=flow_dim)
        
        # Phase 37 controllability organizer
        self.organizer = ControllabilityOrganizer(
            manifold=self.manifold,
            world_model=world_model,
            goal=goal,
            n_probes=n_probes,
            n_steps=8,
            update_interval=5
        )
        
        # Phase 37 CEM
        self.cem = ControllabilityAwareCEM(
            world_model=world_model,
            manifold=self.manifold,
            organizer=self.organizer,
            goal=goal,
            n_candidates=40,
            n_elites=8,
            n_iterations=4,
            controllability_bonus=0.4
        )
        
        # Phase 34 inverse dynamics
        from phase34_inverse_control_stabilization import InverseDynamicsModel
        self.inv_dyn = InverseDynamicsModel(
            latent_dim=world_model.latent_dim,
            action_dim=world_model.action_dim,
            learning_rate=0.01
        )
        
        # Phase 36 learner
        from phase36_behavioral_physics_learning import (
            BehavioralPhysicsLearner, FlowTrajectoryBuffer
        )
        self.learner = BehavioralPhysicsLearner(
            world_model=world_model,
            inv_dyn=self.inv_dyn,
            manifold=self.manifold,
            goal=goal,
            learning_rate=0.02,
            k_steps=4,
            batch_size=16
        )
        
        # Seed flows
        if not self.manifold.flows:
            self._seed_controlled_flows(n_initial_flows)
        
        self.train_every_n = train_every_n
        self.total_steps = 0
        self.cycle_log: List[Dict] = []
        self.execution_log: List[Dict] = []
    
    def _seed_controlled_flows(self, n: int):
        """Seed flows with diversity in controllability."""
        for i in range(n):
            if random.random() < 0.5:
                # Point flows toward diverse regions
                target = np.random.randn(self.wm.latent_dim) * random.uniform(0.3, 1.5)
                flow = PointFlow(target, gain=random.uniform(0.2, 0.8))
            else:
                # Limit cycles at diverse centers
                center = np.random.randn(self.wm.latent_dim) * random.uniform(0.3, 1.0)
                flow = LimitCycleFlow(center, radius=random.uniform(0.5, 2.0),
                                      omega=random.uniform(0.2, 1.0))
            self.manifold.add_flow(flow, f'flow_{i}')
    
    def execute_step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """One execution step with controllability-aware planning."""
        # Plan flow using controllability-aware CEM
        plan = self.cem.plan_flow(z, h)
        
        if plan['flow'] is None:
            flow = list(self.manifold.flows.values())[0] if self.manifold.flows \
                   else PointFlow(np.zeros(self.wm.latent_dim))
        else:
            flow = plan['flow']
        
        a = flow.compute_action(z, h)
        mu, logvar = self.wm.predict_transition(z, h, a)
        std = np.exp(0.5 * logvar)
        z_next = mu + std * np.random.randn(*mu.shape) * 0.1
        h_next = self.wm.gru_step(h, mu)
        
        flow.record_transition(z, z_next, a, h)
        
        self.inv_dyn.train_step(z, z_next, a)
        self.inv_dyn.add_transition(z, z_next, a)
        
        dist = np.linalg.norm(z_next - self.goal.attractor_state[:len(z_next)])
        goal_prob = np.exp(-dist)
        
        flow.stability = flow.compute_lyapunov_estimate()
        
        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'action': a.copy(),
            'goal_prob': float(goal_prob),
            'flow_type': flow.flow_type.value,
            'flow_id': flow.flow_id,
            'stability': flow.stability,
            'n_flows': len(self.manifold.flows)
        }
        
        self.execution_log.append(step_result)
        return step_result
    
    def execute_goal(self, z_start: np.ndarray, max_steps: int = 20) -> Dict:
        """Execute full goal with controllability-enhanced planning."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)
        
        all_states = [z.copy()]
        goal_reached = False
        
        for step in range(max_steps):
            result = self.execute_step(z, h)
            
            if result['goal_prob'] > 0.7:
                goal_reached = True
            
            z = result['z_after'].copy()
            h = self.wm.gru_step(h, result['z_after'])
            all_states.append(z.copy())
        
        # Periodically reorganize by controllability
        if self.total_steps % 10 == 0 and self.organizer.matrix.probe_states:
            self.organizer.organize()
        
        stabilities = [f.stability for f in self.manifold.flows.values()]
        flow_types = [f.flow_type.value for f in self.manifold.flows.values()]
        
        return {
            'goal_reached': goal_reached,
            'final_goal_prob': self.execution_log[-1]['goal_prob'] if self.execution_log else 0.0,
            'n_steps': len(self.execution_log),
            'n_flows': len(self.manifold.flows),
            'flow_types': {t: flow_types.count(t) for t in set(flow_types)},
            'stability': {
                'mean': float(np.mean(stabilities)) if stabilities else 0.0,
                'max': float(max(stabilities)) if stabilities else 0.0,
                'min': float(min(stabilities)) if stabilities else 0.0
            },
            'execution_log': self.execution_log[-10:],
            'trajectory_length': len(all_states)
        }
    
    def run_cycle(self, z_start: np.ndarray, n_steps: int = 20) -> Dict:
        """One closed-loop cycle with controllability reorganization."""
        result = self.execute_goal(z_start, max_steps=n_steps)
        
        log = result.get('execution_log', [])
        states = [z_start.copy()]
        beliefs = [np.zeros(self.wm.belief_dim)]
        for entry in log:
            h = self.wm.gru_step(beliefs[-1], entry['z_before'])
            beliefs.append(h.copy())
            if 'z_after' in entry:
                states.append(entry['z_after'].copy())
            else:
                states.append(entry['z_before'].copy())
        
        # Add to controllability matrix (states + empirical transitions)
        self.organizer.add_trajectory_states(states)
        flow_ids = [e.get('flow_id', '') for e in log]
        goal_probs = [e.get('goal_prob', 0.0) for e in log]
        self.organizer.add_episode_data(
            flow_ids, states, goal_probs, self.goal
        )
        
        if log:
            self.learner.record_from_engine(
                log,
                list(self.manifold.flows.values()),
                states,
                beliefs
            )
        
        # Train periodically
        train_result = {}
        if self.total_steps % self.train_every_n == 0:
            train_losses = []
            for _ in range(5):
                tr = self.learner.train_step()
                if tr['loss'] != float('inf'):
                    train_losses.append(tr['loss'])
            
            if train_losses:
                train_result = {
                    'train_loss': float(np.mean(train_losses)),
                    'loss_trend': (
                        train_losses[0] - train_losses[-1]
                    ) / max(1e-8, train_losses[0])
                }
            
            val = self.learner.validate()
            train_result['val_loss'] = val.get('val_loss', float('inf'))
        
        inv_loss = 0.0
        if len(self.inv_dyn.training_data) > 10:
            inv_loss = self.inv_dyn.train_from_buffer(
                self.inv_dyn.training_data[-100:]
            )
        
        self.total_steps += n_steps
        
        org_report = self.organizer.get_report()
        
        cycle_result = {
            'cycle': len(self.cycle_log),
            'goal_reached': result['goal_reached'],
            'goal_prob': result['final_goal_prob'],
            'n_flows': result['n_flows'],
            'stability': result['stability'],
            'training': train_result,
            'inv_dyn_loss': float(inv_loss),
            'controllability': org_report
        }
        
        self.cycle_log.append(cycle_result)
        return cycle_result
    
    def run_multi_cycle(self, z_start: np.ndarray,
                        n_cycles: int = 10,
                        steps_per_cycle: int = 20) -> Dict:
        """Run multiple cycles with controllability learning."""
        z = z_start.copy()
        
        for cycle in range(n_cycles):
            result = self.run_cycle(z, n_steps=steps_per_cycle)
            
            if self.execution_log:
                last = self.execution_log[-1]
                if 'z_after' in last:
                    z = last['z_after'].copy()
        
        training_report = self.learner.get_training_report()
        goals_reached = sum(1 for c in self.cycle_log if c['goal_reached'])
        
        return {
            'n_cycles': n_cycles,
            'total_steps': self.total_steps,
            'goals_reached': goals_reached,
            'goal_rate': goals_reached / max(1, n_cycles),
            'training': training_report,
            'controllability': self.organizer.get_report(),
            'cycle_log': self.cycle_log[-5:]
        }


# ============================================================================
# 6. TESTS
# ============================================================================

def test_controllability_matrix():
    """Test controllability matrix computation."""
    print("\n" + "=" * 60)
    print("CONTROLLABILITY MATRIX TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='test_reach',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    matrix = ControllabilityMatrix(n_probes=10, n_steps=8)
    
    # Add probe states
    probes = [np.random.randn(16) * 0.5 for _ in range(10)]
    matrix.update_probes(probes)
    
    # Create test flows
    pf = PointFlow(np.ones(16) * 1.0, gain=0.3)
    lc = LimitCycleFlow(np.zeros(16), radius=1.0, omega=0.5)
    
    pf.flow_id = 'point_test'
    lc.flow_id = 'cycle_test'
    
    # Compute profiles
    matrix.update_profile('point_test', pf, wm, goal)
    matrix.update_profile('cycle_test', lc, wm, goal)
    
    # Check
    sim = matrix.compute_controllability_similarity('point_test', 'cycle_test')
    ranked = matrix.get_goal_reaching_flows(goal)
    
    print(f"\n  Probe states: {len(matrix.probe_states)}")
    print(f"  Profiles: {len(matrix.profiles)}")
    print(f"  Point goal_prob: {matrix.profiles['point_test']['mean_goal_prob']:.4f}")
    print(f"  Cycle goal_prob: {matrix.profiles['cycle_test']['mean_goal_prob']:.4f}")
    print(f"  Similarity: {sim:.4f}")
    print(f"  Ranked flows: {ranked}")
    
    print("\n  ✓ Controllability matrix operational")


def test_reachability_graph():
    """Test reachability graph construction."""
    print("\n" + "=" * 60)
    print("REACHABILITY GRAPH TEST")
    print("=" * 60)
    
    graph = ReachabilityGraph()
    
    # Mock profiles
    profiles = {}
    for fid in ['flow_a', 'flow_b']:
        reach = np.random.rand(5, 5) * 0.5
        np.fill_diagonal(reach, 1.0)
        profiles[fid] = {
            'reachability': reach,
            'goal_prob': np.random.rand(5) * 0.3
        }
    
    probes = [np.random.randn(16) * 0.5 for _ in range(5)]
    
    graph.build_from_matrix(profiles, probes, threshold=0.2)
    
    stats = graph.get_stats()
    print(f"\n  Edges: {stats['n_edges']}")
    print(f"  Flows: {stats['n_flows']}")
    print(f"  Coverage: {stats['coverage']:.2f}")
    print(f"  Bottlenecks: {stats['n_bottlenecks']}")
    
    # Goal-aware pathfinding
    goal = GoalAttractor(
        goal_id='test_goal',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    path = graph.find_path_to_goal(np.zeros(16), goal, probes)
    print(f"  Path to goal: {path}")
    
    print("\n  ✓ Reachability graph operational")


def test_controllability_organizer():
    """Test controllability-based manifold organization."""
    print("\n" + "=" * 60)
    print("CONTROLLABILITY ORGANIZER TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='org_test',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    manifold = FlowManifold(flow_dim=4)
    
    # Add diverse flows
    for i in range(6):
        target = np.random.randn(16) * random.uniform(0.3, 1.5)
        flow = PointFlow(target, gain=random.uniform(0.2, 0.8))
        manifold.add_flow(flow, f'flow_{i}')
    
    organizer = ControllabilityOrganizer(
        manifold=manifold,
        world_model=wm,
        goal=goal,
        n_probes=10,
        n_steps=6
    )
    
    # Add states
    states = [np.random.randn(16) * 0.5 for _ in range(15)]
    organizer.add_trajectory_states(states)
    
    # Organize
    report = organizer.organize()
    
    print(f"\n  Organized: {report['n_updates']}")
    print(f"  Probes: {report['n_probes']}")
    print(f"  Profiled flows: {report['n_profiled_flows']}")
    print(f"  Mean goal prob: {report['mean_goal_prob']:.4f}")
    print(f"  Max goal prob: {report['max_goal_prob']:.4f}")
    print(f"  Graph edges: {report['graph_edges']}")
    
    print("\n  ✓ Controllability organizer operational")


def test_controllability_cem():
    """Test controllability-aware CEM."""
    print("\n" + "=" * 60)
    print("CONTROLLABILITY-AWARE CEM TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='cem_test',
        attractor_state=np.ones(16) * 2.0,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    manifold = FlowManifold(flow_dim=4)
    for i in range(8):
        target = np.random.randn(16) * random.uniform(0.3, 1.5)
        flow = PointFlow(target, gain=random.uniform(0.2, 0.8))
        manifold.add_flow(flow, f'flow_{i}')
    
    organizer = ControllabilityOrganizer(
        manifold=manifold, world_model=wm,
        goal=goal, n_probes=10, n_steps=6
    )
    organizer.add_trajectory_states(
        [np.random.randn(16) * 0.5 for _ in range(10)]
    )
    organizer.organize()
    
    cem = ControllabilityAwareCEM(
        world_model=wm, manifold=manifold,
        organizer=organizer, goal=goal,
        n_candidates=20, n_elites=4, n_iterations=3
    )
    
    plan = cem.plan_flow(np.zeros(16), np.zeros(64))
    
    print(f"\n  Plan score: {plan['score']:.4f}")
    print(f"  Has flow: {plan['flow'] is not None}")
    print(f"  Top controllability flows: {plan['top_controllability']}")
    
    print("\n  ✓ Controllability-aware CEM operational")


def test_full_controllability_engine():
    """Test full controllability-enhanced execution."""
    print("\n" + "=" * 60)
    print("FULL CONTROLLABILITY ENGINE TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='full_ctrl',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    engine = ControllabilityEnhancedEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=10,
        flow_dim=4,
        n_probes=10,
        train_every_n=5
    )
    
    result = engine.run_multi_cycle(
        np.zeros(16),
        n_cycles=5,
        steps_per_cycle=10
    )
    
    print(f"\n  Cycles: {result['n_cycles']}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Goals reached: {result['goals_reached']}")
    
    tr = result['training']
    print(f"\n  Training:")
    print(f"    Steps: {tr['training_steps']}")
    print(f"    Buffer: {tr['buffer_episodes']} episodes, {tr['buffer_transitions']} transitions")
    if 'loss_improvement' in tr:
        print(f"    Loss improvement: {tr.get('loss_improvement', 0) * 100:.1f}%")
    
    cr = result['controllability']
    print(f"\n  Controllability:")
    print(f"    Probes: {cr['n_probes']}")
    print(f"    Profiled flows: {cr['n_profiled_flows']}")
    print(f"    Max goal prob: {cr['max_goal_prob']:.4f}")
    print(f"    Graph edges: {cr['graph_edges']}")
    print(f"    Graph coverage: {cr['graph_coverage']:.3f}")
    
    if result['cycle_log']:
        print(f"\n  Recent cycle goal probs:")
        for c in result['cycle_log'][-3:]:
            gp = c['goal_prob']
            print(f"    {gp:.4f}")
    
    print("\n  ✓ Controllability-enhanced engine operational")


if __name__ == "__main__":
    test_controllability_matrix()
    test_reachability_graph()
    test_controllability_organizer()
    test_controllability_cem()
    test_full_controllability_engine()
    
    print("\n" + "=" * 60)
    print("PHASE 37: CONTROLLABILITY TOPOLOGY")
    print("=" * 60)
    
    print("""
KEY SHIFT FROM PHASE 35-36:
  Phase 35:  behavioral similarity (same trajectory shape)
  Phase 37:  controllability similarity (same reachable region)

WHAT CHANGED:
  1. ControllabilityMatrix — per-flow reachability profiles
  2. ReachabilityGraph — directed graph of controllable transitions
  3. ControllabilityOrganizer — reorganize manifold by reachability
  4. ControllabilityAwareCEM — goal-directed flow selection

WHY THIS MATTERS:
  For goal-reaching, we care about DESTINATIONS, not motions.
  Controllability topology directly answers: "which flow can reach my goal?"

ARCHITECTURAL PROGRESSION:
  Phase 34: inverse dynamics (action→action stabilization)
  Phase 35: dynamical flows (state-dependent policies)
  Phase 36: behavioral physics (substrate learns from flows)
  Phase 37: controllability topology (flows organized by reachability)

NEXT (Phase 38):
  Energy-regularized dynamics — add cost functions to flow selection.
  Not just "can I reach the goal" but "is it worth the cost?"
""")
