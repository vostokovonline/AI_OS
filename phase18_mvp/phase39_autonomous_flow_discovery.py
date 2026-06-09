"""
Phase 39 — Autonomous Flow Discovery

KEY SHIFT:
  Phase 35: flows RANDOMLY SEEDED → CEM selects → world model trains
  Phase 39: flows CREATED FROM SUCCESS → manifold self-organizes

  Instead of hoping random flows eventually point toward the goal,
  this phase EXTRACTS flows from successful trajectory segments.
  
  A "successful trajectory segment" is one where:
    - goal probability increased
    - energy cost was reasonable
    - transition was stable (not erratic)

  From such segments, we:
    1. Extract the start → end transition
    2. Create a PointFlow that reproduces that direction
    3. Or create a LimitCycleFlow for oscillatory patterns
    4. Merge similar flows to prevent bloat
    5. Prune flows that never succeed

  This transforms the manifold from a random collection into
  a self-organizing library of goal-directed behaviors.

WHAT CHANGES:
  1. FlowExtractor — finds successful segments in trajectory buffer
  2. FlowFactory — creates flows from trajectory patterns
  3. FlowMerger — merges similar flows (cosine similarity > threshold)
  4. FlowPruner — removes underperforming flows
  5. AutonomousFlowEngine — full closed loop with discovery

ARCHITECTURAL IMPACT:
  The manifold becomes a LIVING structure:
    - New flows are born from success
    - Weak flows die from neglect
    - Similar flows merge
    - The goal attracts the manifold toward itself

  This is the transition from:
    "Manually seeded skill library"
    "Self-organizing behavioral ecology"
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
    ComposedFlow, rollout_flow
)
from phase38_energy_regularized_dynamics import (
    EnergyCostFunction, EfficiencyEvaluator
)


# ============================================================================
# 1. FLOW EXTRACTOR — Find Successful Trajectory Segments
# ============================================================================

@dataclass
class SuccessSegment:
    """A trajectory segment where goal-probability increased."""
    start_state: np.ndarray
    end_state: np.ndarray
    actions: List[np.ndarray]
    goal_prob_start: float
    goal_prob_end: float
    cost: float
    efficiency: float
    flow_type_hint: str  # 'point' or 'cycle'
    source_episode: int = 0
    score: float = 0.0


class FlowExtractor:
    """
    Extracts successful trajectory segments from the execution buffer.
    
    Criteria for "successful":
      1. Goal probability increased: GP_end > GP_start + threshold
      2. Cost was reasonable: cost < cost_threshold
      3. Stable: action variance < var_threshold
    
    Returns: list of SuccessSegments ranked by score
    """
    
    def __init__(
        self,
        gp_threshold: float = 0.01,
        cost_threshold: float = 0.5,
        var_threshold: float = 0.5,
        min_segment_length: int = 2
    ):
        self.gp_threshold = gp_threshold
        self.cost_threshold = cost_threshold
        self.var_threshold = var_threshold
        self.min_length = min_segment_length
        
        self.extracted_count = 0
    
    def extract_from_execution_log(
        self,
        log: List[Dict],
        cost_fn: EnergyCostFunction
    ) -> List[SuccessSegment]:
        """
        Scan execution log for successful segments.
        
        Looks for subsequences where goal_prob monotonically increases
        from start to end with reasonable cost.
        """
        if len(log) < self.min_length + 1:
            return []
        
        segments = []
        
        # Sliding window: find increasing-GP subsequences
        i = 0
        while i < len(log):
            j = i + 1
            while j < len(log) and log[j]['goal_prob'] > log[j - 1]['goal_prob']:
                j += 1
            
            if j - i >= self.min_length:
                # Found a successful segment [i:j]
                seg_log = log[i:j]
                
                start_gp = seg_log[0]['goal_prob']
                end_gp = seg_log[-1]['goal_prob']
                gp_delta = end_gp - start_gp
                
                if gp_delta > self.gp_threshold:
                    # Compute cost
                    actions = [e['action'] for e in seg_log]
                    states = [e['z_before'] for e in seg_log] + [seg_log[-1]['z_after']]
                    
                    seg_flow = self._get_segment_flow(seg_log)
                    cost_result = cost_fn.compute(actions, states, seg_flow)
                    
                    if cost_result['total'] < self.cost_threshold:
                        # Check stability
                        if len(actions) >= 2:
                            flat = [np.asarray(a).flatten()[:8] for a in actions]
                            action_var = float(np.mean(np.var(np.array(flat), axis=0)))
                        else:
                            action_var = 0.0
                        
                        if action_var < self.var_threshold:
                            efficiency = gp_delta / (cost_result['total'] + 1e-8)
                            
                            segment = SuccessSegment(
                                start_state=seg_log[0]['z_before'].copy(),
                                end_state=seg_log[-1]['z_after'].copy(),
                                actions=[a.copy() for a in actions],
                                goal_prob_start=float(start_gp),
                                goal_prob_end=float(end_gp),
                                cost=float(cost_result['total']),
                                efficiency=float(efficiency),
                                flow_type_hint=self._classify_segment(seg_log),
                                source_episode=self.extracted_count,
                                score=float(gp_delta * efficiency)
                            )
                            segments.append(segment)
            
            i = j
        
        self.extracted_count += 1
        segments.sort(key=lambda s: s.score, reverse=True)
        return segments
    
    def extract_from_episode(
        self,
        states: List[np.ndarray],
        actions: List[np.ndarray],
        goal_probs: List[float],
        flow_ids: List[str],
        cost_fn: EnergyCostFunction,
        manifold_flows: Dict[str, SkillFlow]
    ) -> List[SuccessSegment]:
        """
        Extract from a complete episode (states, actions, probs).
        Uses whichever flow was active during the successful segment.
        """
        if len(states) < self.min_length + 1:
            return []
        
        segments = []
        
        i = 0
        while i < len(goal_probs):
            j = i + 1
            while j < len(goal_probs) and goal_probs[j] > goal_probs[j - 1]:
                j += 1
            
            if j - i >= self.min_length:
                start_gp = goal_probs[i]
                end_gp = goal_probs[min(j, len(goal_probs) - 1)]
                gp_delta = end_gp - start_gp
                
                if gp_delta > self.gp_threshold:
                    seg_actions = actions[i:j]
                    seg_states = states[i:j+1]
                    
                    # Get representative flow
                    rep_flow_id = flow_ids[min(i, len(flow_ids) - 1)]
                    rep_flow = manifold_flows.get(rep_flow_id, PointFlow(np.zeros(16)))
                    
                    cost_result = cost_fn.compute(seg_actions, seg_states, rep_flow)
                    
                    if cost_result['total'] < self.cost_threshold:
                        efficiency = gp_delta / (cost_result['total'] + 1e-8)
                        
                        segment = SuccessSegment(
                            start_state=states[i].copy(),
                            end_state=states[min(j, len(states) - 1)].copy(),
                            actions=[a.copy() for a in seg_actions],
                            goal_prob_start=float(start_gp),
                            goal_prob_end=float(end_gp),
                            cost=float(cost_result['total']),
                            efficiency=float(efficiency),
                            flow_type_hint='point',
                            source_episode=self.extracted_count,
                            score=float(gp_delta * efficiency)
                        )
                        segments.append(segment)
            
            i = j
        
        self.extracted_count += 1
        segments.sort(key=lambda s: s.score, reverse=True)
        return segments
    
    def _get_segment_flow(self, seg_log: List[Dict]) -> SkillFlow:
        """Get the dominant flow type from segment."""
        from phase35_dynamical_skill_flows import PointFlow
        return PointFlow(np.zeros(16))
    
    def _classify_segment(self, seg_log: List[Dict]) -> str:
        """
        Classify segment as point-attractor or limit-cycle behavior.
        
        Point: monotonic approach toward a region
        Cycle: oscillatory pattern
        """
        if len(seg_log) < 3:
            return 'point'
        
        states = [e['z_before'] for e in seg_log] + [seg_log[-1]['z_after']]
        
        # Check for oscillation: direction changes
        directions = []
        for t in range(1, len(states)):
            d = states[t] - states[t - 1]
            directions.append(d)
        
        if len(directions) >= 2:
            flips = sum(1 for t in range(1, len(directions))
                       if np.dot(directions[t], directions[t - 1]) < 0)
            
            if flips / len(directions) > 0.3:
                return 'cycle'
        
        return 'point'
    
    def get_stats(self) -> Dict:
        """Extraction statistics."""
        return {
            'total_extracted': self.extracted_count
        }


# ============================================================================
# 2. FLOW FACTORY — Create Flows from Successful Segments
# ============================================================================

class FlowFactory:
    """
    Creates SkillFlow objects from successful trajectory segments.
    
    Given a segment where GP increased:
      - Extract direction: Δ = end_state - start_state
      - Create PointFlow: target = end_state, gain ∈ (0.2, 0.8)
      - Or LimitCycleFlow if oscillatory
    
    Also prevents duplicate flows (cosine similarity > threshold).
    """
    
    def __init__(
        self,
        latent_dim: int = 16,
        similarity_threshold: float = 0.85,
        max_flows: int = 50,
        min_flow_interval: int = 3
    ):
        self.latent_dim = latent_dim
        self.sim_thresh = similarity_threshold
        self.max_flows = max_flows
        self.min_interval = min_flow_interval
        
        self.flows_created = 0
        self.flows_merged = 0
        self.flows_rejected = 0
    
    def create_flow(
        self,
        segment: SuccessSegment,
        existing_flows: Dict[str, SkillFlow]
    ) -> Optional[SkillFlow]:
        """
        Create a flow from a successful segment.
        
        1. Check against existing flows for near-duplicates
        2. Create PointFlow or LimitCycleFlow
        3. Set initial stability
        """
        # Check similarity with existing flows
        for fid, flow in existing_flows.items():
            if self._is_similar(segment, flow):
                self.flows_merged += 1
                return None  # Similar flow exists — don't duplicate
        
        # Create flow
        direction = segment.end_state - segment.start_state
        direction_norm = np.linalg.norm(direction)
        
        if segment.flow_type_hint == 'cycle' and direction_norm > 0.3:
            center = segment.start_state.copy()
            radius = float(np.clip(direction_norm, 0.3, 2.0))
            omega = float(np.clip(np.random.uniform(0.2, 0.8), 0.2, 1.0))
            flow = LimitCycleFlow(center, radius=radius, omega=omega,
                                  latent_dim=self.latent_dim)
        else:
            target = segment.end_state.copy()
            gain = float(np.clip(np.random.uniform(0.2, 0.6), 0.1, 1.0))
            flow = PointFlow(target, gain=gain, latent_dim=self.latent_dim)
        
        # Set initial stability based on segment quality
        flow.stability = float(np.clip(segment.score, 0.1, 0.9))
        flow.goal_alignment = float(np.clip(segment.goal_prob_end, 0.0, 1.0))
        
        self.flows_created += 1
        return flow
    
    def _is_similar(self, segment: SuccessSegment, flow: SkillFlow) -> bool:
        """
        Check if a segment's pattern is similar to an existing flow.
        
        Compare: target direction vs flow's attractor direction.
        """
        seg_direction = segment.end_state - segment.start_state
        seg_norm = np.linalg.norm(seg_direction)
        if seg_norm < 1e-8:
            return False
        
        seg_unit = seg_direction / seg_norm
        
        if isinstance(flow, PointFlow):
            flow_direction = flow.z_target - segment.start_state[:len(flow.z_target)]
            flow_norm = np.linalg.norm(flow_direction)
            if flow_norm < 1e-8:
                return False
            flow_unit = flow_direction / flow_norm
            
            cos_sim = float(np.dot(seg_unit[:len(flow_unit)], flow_unit))
            return cos_sim > self.sim_thresh
        
        return False
    
    def batch_create(
        self,
        segments: List[SuccessSegment],
        existing_flows: Dict[str, SkillFlow],
        max_new: int = 5
    ) -> List[SkillFlow]:
        """Create multiple flows from best segments."""
        new_flows = []
        
        for segment in segments[:max_new * 2]:
            if len(new_flows) >= max_new:
                break
            
            flow = self.create_flow(segment, existing_flows)
            if flow is not None:
                new_flows.append(flow)
        
        return new_flows
    
    def get_stats(self) -> Dict:
        """Factory statistics."""
        return {
            'created': self.flows_created,
            'merged': self.flows_merged,
            'rejected': self.flows_rejected
        }


# ============================================================================
# 3. FLOW MERGER — Merge Similar Flows
# ============================================================================

class FlowMerger:
    """
    Merges similar flows to prevent manifold bloat.
    
    Two flows are "similar" if their behavioral similarity > threshold.
    When merged: create a ComposedFlow or update the stronger one.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.8,
        merge_interval: int = 10
    ):
        self.sim_thresh = similarity_threshold
        self.merge_interval = merge_interval
        self.merges_performed = 0
    
    def find_similar_pairs(
        self, flows: Dict[str, SkillFlow]
    ) -> List[Tuple[str, str, float]]:
        """
        Find pairs of flows with high similarity.
        
        Similarity measure: cosine similarity of attractor targets.
        For PointFlow: compare target vectors.
        For LimitCycleFlow: compare centers.
        """
        flow_ids = list(flows.keys())
        similar_pairs = []
        
        for i in range(len(flow_ids)):
            for j in range(i + 1, len(flow_ids)):
                f1 = flows[flow_ids[i]]
                f2 = flows[flow_ids[j]]
                
                sim = self._compute_flow_similarity(f1, f2)
                if sim > self.sim_thresh:
                    similar_pairs.append((flow_ids[i], flow_ids[j], sim))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
    
    def _compute_flow_similarity(
        self, f1: SkillFlow, f2: SkillFlow
    ) -> float:
        """Compute behavioral similarity between two flows."""
        v1 = self._get_flow_vector(f1)
        v2 = self._get_flow_vector(f2)
        
        if v1 is None or v2 is None:
            return 0.0
        
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        
        return float(np.dot(v1, v2) / (n1 * n2))
    
    def _get_flow_vector(self, flow: SkillFlow) -> Optional[np.ndarray]:
        """Get representative direction vector for a flow."""
        if isinstance(flow, PointFlow):
            return flow.z_target
        elif isinstance(flow, LimitCycleFlow):
            return flow.center
        elif hasattr(flow, 'flows') and flow.flows:
            return self._get_flow_vector(flow.flows[0])
        return None
    
    def merge(
        self, flow_id_1: str, flow_id_2: str,
        flows: Dict[str, SkillFlow], coords: Dict[str, np.ndarray]
    ) -> Optional[str]:
        """
        Merge two similar flows into one.
        
        Keeps the one with higher stability, removes the other.
        Returns: surviving flow_id, or None if both removed.
        """
        if flow_id_1 not in flows or flow_id_2 not in flows:
            return None
        
        f1, f2 = flows[flow_id_1], flows[flow_id_2]
        
        # Keep the more stable flow
        if f1.stability >= f2.stability:
            survivor_id = flow_id_1
            removed_id = flow_id_2
        else:
            survivor_id = flow_id_2
            removed_id = flow_id_1
        
        # Update survivor's stability
        survivor = flows[survivor_id]
        survivor.stability = (survivor.stability + flows[removed_id].stability) * 0.5
        survivor.goal_alignment = max(survivor.goal_alignment, flows[removed_id].goal_alignment)
        
        # Remove the weaker flow
        del flows[removed_id]
        if removed_id in coords:
            del coords[removed_id]
        
        self.merges_performed += 1
        return survivor_id
    
    def merge_all(
        self, flows: Dict[str, SkillFlow], coords: Dict[str, np.ndarray]
    ) -> int:
        """Find and merge all similar pairs."""
        if len(flows) < 2:
            return 0
        
        total_merges = 0
        while True:
            pairs = self.find_similar_pairs(flows)
            if not pairs:
                break
            
            fid1, fid2, sim = pairs[0]
            result = self.merge(fid1, fid2, flows, coords)
            if result:
                total_merges += 1
            else:
                break
        
        return total_merges
    
    def get_stats(self) -> Dict:
        return {'merges': self.merges_performed}


# ============================================================================
# 4. FLOW PRUNER — Remove Underperforming Flows
# ============================================================================

class FlowPruner:
    """
    Removes flows that never lead to goal progress.
    
    Criteria for pruning:
      1. Age > min_age AND stability < min_stability
      2. Goal alignment < min_alignment
      3. Never selected by CEM in last N cycles
    """
    
    def __init__(
        self,
        min_age: int = 20,
        min_stability: float = 0.2,
        min_alignment: float = 0.01,
        max_unselected_cycles: int = 10,
        min_flows: int = 4,
        prune_interval: int = 5
    ):
        self.min_age = min_age
        self.min_stability = min_stability
        self.min_alignment = min_alignment
        self.max_unselected = max_unselected_cycles
        self.min_flows = min_flows
        self.interval = prune_interval
        
        self.selection_counts: Dict[str, int] = defaultdict(int)
        self.pruned_count = 0
    
    def record_selection(self, flow_id: str):
        """Record that a flow was selected."""
        self.selection_counts[flow_id] += 1
    
    def prune(
        self, flows: Dict[str, SkillFlow], coords: Dict[str, np.ndarray]
    ) -> List[str]:
        """Remove underperforming flows. Returns pruned IDs."""
        if len(flows) <= self.min_flows:
            return []
        
        pruned = []
        for fid in list(flows.keys()):
            flow = flows[fid]
            
            reasons = []
            
            if flow.age < 3:
                continue
            
            if flow.age > self.min_age and flow.stability < self.min_stability:
                reasons.append('low_stability')
            
            if flow.goal_alignment < self.min_alignment and flow.age > self.min_age // 2:
                reasons.append('low_alignment')
            
            if (self.selection_counts.get(fid, 0) == 0
                and flow.age > self.min_age):
                reasons.append('never_selected')
            
            if reasons:
                pruned.append(fid)
                del flows[fid]
                if fid in coords:
                    del coords[fid]
                if fid in self.selection_counts:
                    del self.selection_counts[fid]
        
        self.pruned_count += len(pruned)
        return pruned
    
    def decay_counts(self):
        """Decay selection counts (old selections matter less)."""
        for fid in list(self.selection_counts.keys()):
            self.selection_counts[fid] = max(0, self.selection_counts[fid] - 1)
    
    def get_stats(self) -> Dict:
        return {'pruned': self.pruned_count}


# ============================================================================
# 5. AUTONOMOUS FLOW ENGINE
# ============================================================================

class AutonomousFlowEngine:
    """
    Full closed-loop engine with autonomous flow discovery.
    
    Extends Phase 38 EnergyRegularizedEngine with:
      Phase 39: FlowExtractor — finds successful trajectory patterns
      Phase 39: FlowFactory — creates flows from success
      Phase 39: FlowMerger — merges similar flows
      Phase 39: FlowPruner — removes dead flows
    
    The manifold is now a LIVING structure:
      - New flows born from success
      - Weak flows die from neglect  
      - Similar flows merge
      - Goal attracts the manifold toward itself
    """
    
    def __init__(
        self,
        world_model: 'FlowConditionedWorldModel',
        goal: GoalAttractor,
        manifold: Optional[FlowManifold] = None,
        n_initial_flows: int = 8,
        flow_dim: int = 4,
        lambda_cost: float = 0.5,
        discovery_interval: int = 3,
        prune_interval: int = 5,
        merge_interval: int = 8,
        train_every_n: int = 5
    ):
        self.wm = world_model
        self.goal = goal
        self.flow_dim = flow_dim
        
        # Phase 35 manifold
        self.manifold = manifold or FlowManifold(flow_dim=flow_dim)
        
        # Phase 38 components
        self.energy_cost = EnergyCostFunction(
            w_action=0.3, w_path=0.3, w_variance=0.1, w_instability=0.3
        )
        self.efficiency = EfficiencyEvaluator(
            cost_fn=self.energy_cost, lambda_cost=lambda_cost
        )
        
        # Phase 38 CEM (energy-regularized)
        from phase38_energy_regularized_dynamics import EnergyRegularizedCEM
        self.cem = EnergyRegularizedCEM(
            world_model=self.wm, manifold=self.manifold,
            goal=self.goal, energy_cost_fn=self.energy_cost,
            efficiency_evaluator=self.efficiency,
            flow_dim=flow_dim, n_candidates=40, n_elites=8,
            n_iterations=4, controllability_bonus=0.2,
            efficiency_mode=True
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
            BehavioralPhysicsLearner
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
        
        # Phase 39 components
        self.extractor = FlowExtractor(
            gp_threshold=0.005,
            cost_threshold=0.4,
            var_threshold=0.5,
            min_segment_length=2
        )
        self.factory = FlowFactory(
            latent_dim=world_model.latent_dim,
            similarity_threshold=0.8,
            max_flows=50
        )
        self.merger = FlowMerger(similarity_threshold=0.85)
        self.pruner = FlowPruner(
            min_age=15, min_stability=0.15,
            min_alignment=0.005, min_flows=4
        )
        
        # Controllability rankings
        self.ranked_flows: List[Tuple[str, float]] = []
        self.flow_goal_probs: Dict[str, List[float]] = defaultdict(list)
        
        # Seed initial flows
        if not self.manifold.flows:
            self._seed_initial_flows(n_initial_flows)
        
        self.discovery_interval = discovery_interval
        self.prune_interval = prune_interval
        self.merge_interval = merge_interval
        self.train_every_n = train_every_n
        
        self.total_steps = 0
        self.cycle_count = 0
        self.cycle_log: List[Dict] = []
        self.execution_log: List[Dict] = []
        self.all_logs: List[List[Dict]] = []
    
    def _seed_initial_flows(self, n: int):
        """Seed diverse flows including at least one goal-directed.
        
        The goal-directed flow provides the bootstrap signal needed
        for autonomous discovery — without it, no trajectory segment
        ever increases GP, so no flows are ever created.
        """
        from phase35_dynamical_skill_flows import LimitCycleFlow
        for i in range(n):
            if i == 0:
                # First flow explicitly targets the goal
                target = self.goal.attractor_state[:self.wm.latent_dim].copy()
                flow = PointFlow(target, gain=0.3, latent_dim=self.wm.latent_dim)
                flow.stability = 0.5
                flow.goal_alignment = 0.5
            elif random.random() < 0.5:
                target = np.random.randn(self.wm.latent_dim) * random.uniform(0.3, 1.5)
                flow = PointFlow(target, gain=random.uniform(0.2, 0.8),
                                 latent_dim=self.wm.latent_dim)
            else:
                center = np.random.randn(self.wm.latent_dim) * random.uniform(0.3, 1.0)
                flow = LimitCycleFlow(center, radius=random.uniform(0.5, 2.0),
                                      omega=random.uniform(0.2, 1.0),
                                      latent_dim=self.wm.latent_dim)
            self.manifold.add_flow(flow, f'seed_{i}')
    
    def _update_ranked_flows(self):
        """Rank flows by empirical goal probability."""
        ranked = []
        for fid in self.manifold.flows:
            gps = self.flow_goal_probs.get(fid, [])
            avg_gp = float(np.mean(gps)) if gps else 0.0
            ranked.append((fid, avg_gp))
        ranked.sort(key=lambda x: x[1], reverse=True)
        self.ranked_flows = ranked
    
    def execute_step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """One execution step with energy-regularized CEM."""
        plan = self.cem.plan_flow(z, h, self.ranked_flows)
        
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
        
        # Record goal prob
        self.flow_goal_probs[flow.flow_id].append(goal_prob)
        
        # Track selection for pruner
        self.pruner.record_selection(flow.flow_id)
        
        cost_info = self.energy_cost.compute([a], [z, z_next], flow)
        
        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'action': a.copy(),
            'goal_prob': float(goal_prob),
            'flow_type': flow.flow_type.value,
            'flow_id': flow.flow_id,
            'stability': flow.stability,
            'energy_cost': cost_info
        }
        
        self.execution_log.append(step_result)
        return step_result
    
    def execute_goal(self, z_start: np.ndarray, max_steps: int = 20) -> Dict:
        """Execute full goal with autonomous flow discovery."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)
        
        all_states = [z.copy()]
        goal_reached = False
        total_cost = 0.0
        
        for step in range(max_steps):
            result = self.execute_step(z, h)
            
            if result['goal_prob'] > 0.7:
                goal_reached = True
            
            total_cost += result['energy_cost']['total']
            
            z = result['z_after'].copy()
            h = self.wm.gru_step(h, result['z_after'])
            all_states.append(z.copy())
        
        # Store log for flow extraction
        self.all_logs.append(list(self.execution_log))
        if len(self.all_logs) > 20:
            self.all_logs = self.all_logs[-20:]
        
        self._update_ranked_flows()
        
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
            'total_cost': total_cost,
            'avg_cost': total_cost / max(1, len(self.execution_log)),
            'execution_log': self.execution_log[-10:],
            'trajectory_length': len(all_states)
        }
    
    def _discover_flows(self):
        """Extract successful segments and create new flows."""
        new_flow_count = 0
        
        for log in self.all_logs[-5:]:
            segments = self.extractor.extract_from_execution_log(
                log, self.energy_cost
            )
            
            if segments:
                new_flows = self.factory.batch_create(
                    segments, self.manifold.flows, max_new=3
                )
                
                for flow in new_flows:
                    fid = f'discovered_{self.cycle_count}_{new_flow_count}'
                    self.manifold.add_flow(flow, fid)
                    new_flow_count += 1
        
        if new_flow_count > 0:
            # Reinitialize CEM mean near new flows
            new_coords = [
                self.manifold.flow_coords[fid]
                for fid in list(self.manifold.flows.keys())[-new_flow_count:]
                if fid in self.manifold.flow_coords
            ]
            if new_coords:
                self.cem.mean = np.mean(new_coords, axis=0)
                self.cem.std = np.ones(self.flow_dim) * 0.5
    
    def run_cycle(self, z_start: np.ndarray, n_steps: int = 20) -> Dict:
        """One cycle with autonomous flow discovery."""
        result = self.execute_goal(z_start, max_steps=n_steps)
        
        # Flow discovery (periodic)
        disc_result = {}
        if self.cycle_count % self.discovery_interval == 0 and self.cycle_count > 0:
            self._discover_flows()
            disc_result = self.factory.get_stats()
        
        # Merge similar flows (periodic)
        merge_result = {}
        if self.cycle_count % self.merge_interval == 0 and self.cycle_count > 0:
            n_merged = self.merger.merge_all(
                self.manifold.flows, self.manifold.flow_coords
            )
            merge_result = {'merged': n_merged}
        
        # Prune weak flows (periodic)
        prune_result = {}
        if self.cycle_count % self.prune_interval == 0 and self.cycle_count > 0:
            pruned = self.pruner.prune(
                self.manifold.flows, self.manifold.flow_coords
            )
            prune_result = {'pruned': len(pruned)}
            self.pruner.decay_counts()
        
        # Record trajectory for Phase 36 training
        log = result.get('execution_log', [])
        states = [z_start.copy()]
        beliefs = [np.zeros(self.wm.belief_dim)]
        for entry in log:
            h = self.wm.gru_step(beliefs[-1], entry['z_before'])
            beliefs.append(h.copy())
            if 'z_after' in entry:
                states.append(entry['z_after'].copy())
        
        if log:
            self.learner.record_from_engine(
                log, list(self.manifold.flows.values()), states, beliefs
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
        
        cycle_result = {
            'cycle': self.cycle_count,
            'goal_reached': result['goal_reached'],
            'goal_prob': result['final_goal_prob'],
            'n_flows': result['n_flows'],
            'stability': result['stability'],
            'avg_cost': result['avg_cost'],
            'training': train_result,
            'inv_dyn_loss': float(inv_loss),
            'discovery': disc_result,
            'merge': merge_result,
            'prune': prune_result
        }
        
        self.cycle_log.append(cycle_result)
        self.cycle_count += 1
        return cycle_result
    
    def run_multi_cycle(self, z_start: np.ndarray,
                        n_cycles: int = 30,
                        steps_per_cycle: int = 20) -> Dict:
        """Run multiple cycles with autonomous flow discovery."""
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
            'flow_stats': {
                'n_flows': len(self.manifold.flows),
                'extractor': self.extractor.get_stats(),
                'factory': self.factory.get_stats(),
                'merger': self.merger.get_stats(),
                'pruner': self.pruner.get_stats()
            },
            'cycle_log': self.cycle_log[-5:]
        }


# ============================================================================
# 6. TESTS
# ============================================================================

def test_flow_extractor():
    """Test extraction of successful segments."""
    print("\n" + "=" * 60)
    print("FLOW EXTRACTOR TEST")
    print("=" * 60)
    
    cost_fn = EnergyCostFunction()
    extractor = FlowExtractor(gp_threshold=0.001, cost_threshold=0.5)
    
    # Create mock execution log with goal-prob-increasing segments
    log = []
    z = np.zeros(16)
    for t in range(15):
        gp = 0.01 + 0.002 * t  # Slowly increasing
        if t > 8:
            gp = 0.03 + 0.01 * (t - 8)  # Faster increase
        log.append({
            'z_before': z.copy(),
            'z_after': z + np.random.randn(16) * 0.05,
            'action': np.random.randn(16) * 0.1,
            'goal_prob': gp,
            'flow_type': 'point_attractor',
            'flow_id': 'test_flow'
        })
        z = log[-1]['z_after'].copy()
    
    segments = extractor.extract_from_execution_log(log, cost_fn)
    
    print(f"\n  Log entries: {len(log)}")
    print(f"  Found segments: {len(segments)}")
    
    if segments:
        seg = segments[0]
        print(f"  Best segment:")
        print(f"    GP: {seg.goal_prob_start:.4f} → {seg.goal_prob_end:.4f}")
        print(f"    Cost: {seg.cost:.4f}")
        print(f"    Score: {seg.score:.4f}")
    
    print("\n  ✓ Flow extractor operational")


def test_flow_factory():
    """Test flow creation from successful segments."""
    print("\n" + "=" * 60)
    print("FLOW FACTORY TEST")
    print("=" * 60)
    
    factory = FlowFactory(latent_dim=16, similarity_threshold=0.85)
    
    segments = [
        SuccessSegment(
            start_state=np.zeros(16),
            end_state=np.ones(16) * 0.5,
            actions=[np.random.randn(16) * 0.1 for _ in range(3)],
            goal_prob_start=0.01, goal_prob_end=0.05,
            cost=0.2, efficiency=0.02,
            flow_type_hint='point',
            score=0.04
        ),
        SuccessSegment(
            start_state=np.ones(16) * 0.3,
            end_state=np.ones(16) * 0.8,
            actions=[np.random.randn(16) * 0.1 for _ in range(5)],
            goal_prob_start=0.02, goal_prob_end=0.08,
            cost=0.15, efficiency=0.04,
            flow_type_hint='point',
            score=0.06
        )
    ]
    
    existing = {}
    new_flows = factory.batch_create(segments, existing, max_new=5)
    
    print(f"\n  Created flows: {len(new_flows)}")
    
    for i, flow in enumerate(new_flows):
        print(f"  Flow {i}: {flow.flow_type.value}, "
              f"stability={flow.stability:.3f}")
    
    # Test similarity detection (duplicate should be rejected)
    duplicate_seg = SuccessSegment(
        start_state=np.zeros(16),
        end_state=np.ones(16) * 0.5,
        actions=[np.random.randn(16) * 0.1 for _ in range(3)],
        goal_prob_start=0.01, goal_prob_end=0.05,
        cost=0.2, efficiency=0.02,
        flow_type_hint='point',
        score=0.04
    )
    
    # Now existing flows include the one we just created
    existing_with_new = {}
    for i, f in enumerate(new_flows):
        fid = f'flow_{i}'
        existing_with_new[fid] = f
        f.flow_id = fid
    
    duplicate_flow = factory.create_flow(duplicate_seg, existing_with_new)
    print(f"  Duplicate rejected: {duplicate_flow is None}  "
          f"(merged={factory.flows_merged})")
    
    print("\n  ✓ Flow factory operational")


def test_flow_merger():
    """Test merging of similar flows."""
    print("\n" + "=" * 60)
    print("FLOW MERGER TEST")
    print("=" * 60)
    
    merger = FlowMerger(similarity_threshold=0.7)
    
    flows = {}
    coords = {}
    
    # Create similar PointFlows (same direction)
    for i in range(5):
        target = np.ones(16) * 0.5 + np.random.randn(16) * 0.05  # Very similar
        flow = PointFlow(target, gain=0.5)
        fid = f'sim_flow_{i}'
        flows[fid] = flow
        coords[fid] = np.random.randn(4) * 0.5
    
    # Create a very different flow
    diff_flow = PointFlow(np.ones(16) * 2.0, gain=0.8)
    flows['diff_flow'] = diff_flow
    coords['diff_flow'] = np.random.randn(4) * 2.0
    
    n_before = len(flows)
    n_merged = merger.merge_all(flows, coords)
    
    print(f"\n  Before: {n_before} flows")
    print(f"  After: {len(flows)} flows")
    print(f"  Merged: {n_merged}")
    
    print("\n  ✓ Flow merger operational")


def test_flow_pruner():
    """Test pruning of underperforming flows."""
    print("\n" + "=" * 60)
    print("FLOW PRUNER TEST")
    print("=" * 60)
    
    pruner = FlowPruner(
        min_age=5, min_stability=0.2,
        min_alignment=0.01, min_flows=2
    )
    
    flows = {}
    coords = {}
    
    # Good flow
    good_flow = PointFlow(np.ones(16) * 0.5, gain=0.5)
    good_flow.age = 10
    good_flow.stability = 0.8
    good_flow.goal_alignment = 0.5
    flows['good'] = good_flow
    coords['good'] = np.zeros(4)
    for _ in range(5):
        pruner.record_selection('good')
    
    # Bad flow (old, unstable, low alignment, never selected)
    bad_flow = PointFlow(np.ones(16) * 5.0, gain=2.0)
    bad_flow.age = 20
    bad_flow.stability = 0.1
    bad_flow.goal_alignment = 0.001
    flows['bad'] = bad_flow
    coords['bad'] = np.zeros(4)
    # Note: never call record_selection for 'bad'
    
    # Too young to prune
    young_flow = PointFlow(np.ones(16) * 0.3, gain=0.3)
    young_flow.age = 2
    young_flow.stability = 0.1
    young_flow.goal_alignment = 0.001
    flows['young'] = young_flow
    coords['young'] = np.zeros(4)
    
    n_before = len(flows)
    pruned = pruner.prune(flows, coords)
    
    print(f"\n  Before: {n_before} flows")
    print(f"  After: {len(flows)} flows")
    print(f"  Pruned: {pruned}")
    print(f"  'bad' removed: {'bad' not in flows}")
    print(f"  'good' kept: {'good' in flows}")
    
    print("\n  ✓ Flow pruner operational")


def test_full_autonomous_engine():
    """Test full autonomous flow engine."""
    print("\n" + "=" * 60)
    print("FULL AUTONOMOUS ENGINE TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='auto_test',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    engine = AutonomousFlowEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=6,
        flow_dim=4,
        lambda_cost=0.5,
        discovery_interval=2,
        prune_interval=3,
        merge_interval=4,
        train_every_n=5
    )
    
    result = engine.run_multi_cycle(
        np.zeros(16),
        n_cycles=8,
        steps_per_cycle=10
    )
    
    print(f"\n  Cycles: {result['n_cycles']}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Goals reached: {result['goals_reached']}")
    
    tr = result['training']
    print(f"\n  Training:")
    print(f"    Steps: {tr['training_steps']}")
    print(f"    Buffer: {tr['buffer_episodes']} eps, {tr['buffer_transitions']} trans")
    
    fs = result['flow_stats']
    print(f"\n  Flow stats:")
    print(f"    Final: {fs['n_flows']} flows")
    print(f"    Extractor: {fs['extractor']['total_extracted']} runs")
    print(f"    Factory: {fs['factory']['created']} created, "
          f"{fs['factory']['merged']} merged")
    print(f"    Merger: {fs['merger']['merges']} merged pairs")
    print(f"    Pruner: {fs['pruner']['pruned']} pruned")
    
    if result['cycle_log']:
        print(f"\n  Recent cycles:")
        for c in result['cycle_log'][-3:]:
            d = c.get('discovery', {})
            m = c.get('merge', {})
            p = c.get('prune', {})
            print(f"    GP={c['goal_prob']:.4f}  cost={c['avg_cost']:.4f}  "
                  f"flows={c['n_flows']}  "
                  f"disc={d.get('created', 0)}  "
                  f"merge={m.get('merged', 0)}  "
                  f"prune={p.get('pruned', 0)}")
    
    print("\n  ✓ Autonomous engine operational")


if __name__ == "__main__":
    test_flow_extractor()
    test_flow_factory()
    test_flow_merger()
    test_flow_pruner()
    test_full_autonomous_engine()
    
    print("\n" + "=" * 60)
    print("PHASE 39: AUTONOMOUS FLOW DISCOVERY")
    print("=" * 60)
    
    print("""
KEY SHIFT FROM PHASE 35:
  Phase 35: flows RANDOMLY SEEDED → CEM selects
  Phase 39: flows CREATED FROM SUCCESS → manifold self-organizes

WHAT CHANGED:
  1. FlowExtractor — finds GP-increasing segments in trajectory data
  2. FlowFactory — creates PointFlow/LimitCycleFlow from segments
  3. FlowMerger — merges similar flows (cosine similarity > threshold)
  4. FlowPruner — removes underperforming flows (age+stability+alignment)
  5. AutonomousFlowEngine — full closed-loop with discovery

WHY THIS MATTERS:
  The manifold transforms from a random collection into
  a self-organizing library of goal-directed behaviors.
  
  This is the transition from:
    "Manually seeded skill library"
    "Self-organizing behavioral ecology"

ARCHITECTURAL PROGRESSION:
  Phase 35: dynamical flows (state-dependent policies)
  Phase 36: behavioral physics (learn dynamics from flows)
  Phase 37: controllability (where can flows reach?)
  Phase 38: energy-regularization (is it worth the cost?)
  Phase 39: autonomous discovery (flows born from success)

NEXT (Phase 40):
  Self-organizing behavioral geometry — the manifold develops
  topological structure through continuous interaction, not
  periodic reorganization steps.
""")
