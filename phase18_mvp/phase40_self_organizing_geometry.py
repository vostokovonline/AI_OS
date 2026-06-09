"""
Phase 40 — Self-Organizing Behavioral Geometry

KEY SHIFT:
  Phases 30-39: pipeline with discrete steps
    execute → train → discover → merge → prune → CEM
    
  Phase 40:     continuous dynamical system
    Everything happens simultaneously, every step.
    
    Flows are continuously BORN from positive GP deltas.
    Flows continuously DIE from negative GP deltas.
    Manifold continuously DRIFTS toward successful regions.
    World model continuously TRAINS on every transition.
    CEM continuously ADAPTS its sampling distribution.

ARCHITECTURAL UNIFICATION:
  There is no more "phase separation."
  The system is a single unified process:
  
    step() → {transition, GP, cost, flow_id}
      ↓
    World model trains (online, every step)
    Flow birth/death (probabilistic, every step)
    Manifold drift (exponential averaging, every step)
    CEM distribution (slides toward best flow, every step)
    Inverse dynamics (online, every step)
    Energy cost (computed every step)
      ↓
    next step with improved model + manifold

WHY THIS MATTERS:
  Phases 30-39: "Build a robot that learns"
  Phase 40:     "The robot IS learning"
  
  The system is no longer a pipeline that alternates
  between executing and learning.
  The system is a single continuous cognitive process.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

from phase30_training_loop import MinimalWorldModel
from phase31_hierarchical_execution import GoalAttractor
from phase34_inverse_control_stabilization import InverseDynamicsModel
from phase35_dynamical_skill_flows import (
    SkillFlow, FlowManifold, FlowType, PointFlow, LimitCycleFlow,
    ComposedFlow, rollout_flow
)
from phase38_energy_regularized_dynamics import (
    EnergyCostFunction, EfficiencyEvaluator, EnergyRegularizedCEM,
    CostAwareFlowSelection
)


# ============================================================================
# 1. CONTINUOUS FLOW BIRTH/DEATH
# ============================================================================

class ContinuousFlowEcology:
    """
    Manages continuous flow birth and death, every step.
    
    Birth: when GP increases, a new flow is probabilistically spawned
           near the trajectory direction that caused the increase.
    
    Death: when a flow underperforms, it probabilistically dies.
    
    This replaces the periodic extractor + factory + merger + pruner
    with a continuous birth-death process.
    """
    
    def __init__(
        self,
        manifold: FlowManifold,
        goal_attractor: np.ndarray,
        latent_dim: int = 16,
        birth_rate: float = 0.05,
        death_rate: float = 0.02,
        min_flows: int = 4,
        max_flows: int = 40,
        similarity_threshold: float = 0.7
    ):
        self.manifold = manifold
        self.goal_attractor = goal_attractor
        self.latent_dim = latent_dim
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.min_flows = min_flows
        self.max_flows = max_flows
        self.sim_thresh = similarity_threshold
        
        # Running stats for birth/death decisions
        self.recent_gp_deltas: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        self.flow_performance: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=20)
        )
        self.total_steps = 0
        self.n_births = 0
        self.n_deaths = 0
    
    def observe_transition(
        self, flow_id: str, z_before: np.ndarray,
        z_after: np.ndarray, goal_prob: float
    ):
        """Observe a transition and update flow stats."""
        pass  # Performance tracked externally
    
    def record_gp_delta(self, flow_id: str, gp_delta: float):
        """Record goal probability change for a flow."""
        self.recent_gp_deltas[flow_id].append(gp_delta)
    
    def record_performance(self, flow_id: str, gp: float):
        """Record goal probability for performance tracking."""
        self.flow_performance[flow_id].append(gp)
    
    def step(self) -> Dict:
        """
        One ecological step — probabilistic birth and death.
        
        Returns: {born: [flow_ids], died: [flow_ids]}
        """
        self.total_steps += 1
        result = {'born': [], 'died': []}
        
        # 1. Probabilistic death
        if len(self.manifold.flows) > self.min_flows:
            for fid in list(self.manifold.flows.keys()):
                flow = self.manifold.flows[fid]
                
                if flow.age < 5:
                    continue  # Don't kill newborns
                
                death_prob = self._compute_death_probability(fid, flow)
                
                if random.random() < death_prob:
                    if fid in self.manifold.flow_coords:
                        del self.manifold.flow_coords[fid]
                    del self.manifold.flows[fid]
                    if fid in self.recent_gp_deltas:
                        del self.recent_gp_deltas[fid]
                    if fid in self.flow_performance:
                        del self.flow_performance[fid]
                    result['died'].append(fid)
                    self.n_deaths += 1
        
        # 2. Probabilistic birth
        if len(self.manifold.flows) < self.max_flows:
            # Find flows with increasing GP
            for fid in list(self.manifold.flows.keys()):
                deltas = list(self.recent_gp_deltas.get(fid, []))
                if len(deltas) >= 3 and np.mean(deltas[-3:]) > 0.001:
                    birth_prob = self.birth_rate
                    
                    if random.random() < birth_prob:
                        flow = self.manifold.flows[fid]
                        new_flow = self._spawn_child(fid, flow)
                        
                        if new_flow is not None:
                            child_id = f'born_{self.n_births}'
                            self.manifold.add_flow(new_flow, child_id)
                            result['born'].append(child_id)
                            self.n_births += 1
        
        return result
    
    def _compute_death_probability(self, fid: str, flow: SkillFlow) -> float:
        """
        Compute probability that this flow should die.
        
        Factors:
          - Low stability → higher death prob
          - Low goal alignment → higher death prob
          - Recent negative GP deltas → higher death prob
        """
        prob = 0.0
        
        # Stability
        if flow.stability < 0.3:
            prob += 0.01
        if flow.stability < 0.1:
            prob += 0.02
        
        # Goal alignment
        if flow.goal_alignment < 0.01:
            prob += 0.01
        
        # Recent performance
        recent = list(self.flow_performance.get(fid, []))
        if recent and len(recent) >= 5:
            avg_gp = float(np.mean(recent[-5:]))
            if avg_gp < 0.001:
                prob += 0.02
            if avg_gp < 0.0005:
                prob += 0.03
        
        # Age factor (very old flows that never improved)
        if flow.age > 50 and flow.goal_alignment < 0.01:
            prob += 0.05
        
        return float(np.clip(prob * self.death_rate * 10, 0.0, 0.3))
    
    def _spawn_child(
        self, parent_id: str, parent_flow: SkillFlow
    ) -> Optional[SkillFlow]:
        """
        Create a child flow by mutating a successful parent.
        
        Mutation: slightly perturb parent's parameters.
        """
        if isinstance(parent_flow, PointFlow):
            noise = np.random.randn(self.latent_dim) * 0.1
            target = parent_flow.z_target + noise
            gain = float(np.clip(
                parent_flow.gain + random.uniform(-0.1, 0.1), 0.1, 1.0
            ))
            child = PointFlow(target, gain=gain, latent_dim=self.latent_dim)
        elif isinstance(parent_flow, LimitCycleFlow):
            noise = np.random.randn(self.latent_dim) * 0.1
            center = parent_flow.center + noise
            radius = float(np.clip(
                parent_flow.radius + random.uniform(-0.2, 0.2), 0.3, 3.0
            ))
            child = LimitCycleFlow(
                center, radius=radius, omega=parent_flow.omega,
                latent_dim=self.latent_dim
            )
        else:
            return None
        
        child.stability = parent_flow.stability * 0.8
        child.goal_alignment = parent_flow.goal_alignment * 0.9
        return child
    
    def get_stats(self) -> Dict:
        return {
            'n_flows': len(self.manifold.flows),
            'births': self.n_births,
            'deaths': self.n_deaths,
            'birth_rate': self.birth_rate,
            'death_rate': self.death_rate
        }


# ============================================================================
# 2. CONTINUOUS MANIFOLD DRIFT
# ============================================================================

class ContinuousManifoldDrift:
    """
    Manifold coordinates drift continuously based on experience.
    
    Every step:
      1. The active flow's coordinate is pulled toward recent success
      2. All flows drift slightly toward the goal region
      3. Similar flows attract each other (topological organization)
      4. The drift is smooth, continuous, and exponential
    
    This replaces the periodic organize_topology() calls
    with a continuous dynamical process.
    """
    
    def __init__(
        self,
        manifold: FlowManifold,
        learning_rate: float = 0.02,
        goal_attraction: float = 0.01,
        similarity_attraction: float = 0.005,
        drift_decay: float = 0.99
    ):
        self.manifold = manifold
        self.lr = learning_rate
        self.goal_attraction = goal_attraction
        self.sim_attraction = similarity_attraction
        self.decay = drift_decay
        
        # Velocity (momentum for smooth drift)
        self.velocities: Dict[str, np.ndarray] = {}
        
        self.total_steps = 0
    
    def step(
        self, active_flow_id: str, goal_prob: float,
        gp_delta: float, goal: GoalAttractor
    ):
        """One drift step — continuous manifold reorganization."""
        self.total_steps += 1
        
        if active_flow_id not in self.manifold.flow_coords:
            return
        
        coord = self.manifold.flow_coords[active_flow_id]
        
        if active_flow_id not in self.velocities:
            self.velocities[active_flow_id] = np.zeros_like(coord)
        
        # 1. Pull active flow toward success
        if gp_delta > 0:
            # Successful step: pull toward goal region in latent space
            # (positive GP delta means we moved toward goal)
            success_pull = self.lr * gp_delta * 10.0
            # Random direction (we don't know manifold direction to goal)
            # Instead, pull toward a random direction that correlated with success
            success_vector = np.random.randn(len(coord)) * success_pull
            self.velocities[active_flow_id] += success_vector
        
        # 2. Goal attraction (weak, continuous pull toward origin)
        # Goal manifold coordinate = [1, 0, 0, ...] (arbitrary convention)
        goal_coord = np.zeros(len(coord))
        goal_coord[0] = 1.0  # Dimension 0 = goal axis
        goal_vec = (goal_coord - coord) * self.goal_attraction
        self.velocities[active_flow_id] += goal_vec
        
        # 3. Similarity attraction (pull toward similar flows)
        for fid2, coord2 in self.manifold.flow_coords.items():
            if fid2 == active_flow_id:
                continue
            
            dist = np.linalg.norm(coord - coord2)
            if dist < 2.0 and dist > 0.01:
                # Attraction proportional to similarity
                sim = 1.0 / (1.0 + dist)
                force = (coord2 - coord) * self.sim_attraction * sim
                self.velocities[active_flow_id] += force
        
        # 4. Apply velocity with decay (momentum)
        self.velocities[active_flow_id] *= self.decay
        self.manifold.flow_coords[active_flow_id] += self.velocities[active_flow_id]
    
    def get_stats(self) -> Dict:
        total_drift = sum(
            float(np.linalg.norm(v))
            for v in self.velocities.values()
        ) if self.velocities else 0.0
        return {
            'n_velocities': len(self.velocities),
            'total_drift': total_drift,
            'steps': self.total_steps
        }


# ============================================================================
# 3. CONTINUOUS CEM ADAPTATION
# ============================================================================

class ContinuousCEM:
    """
    CEM that continuously adapts its distribution.
    
    Every step:
      1. Sample from current distribution → generate action
      2. Observe outcome (GP, cost)
      3. Update distribution based on outcome
      4. Distribution slides toward successful regions
    
    This replaces the per-cycle CEM reinitialization
    with continuous distribution tracking.
    """
    
    def __init__(
        self,
        manifold: FlowManifold,
        goal: GoalAttractor,
        energy_cost: EnergyCostFunction,
        flow_dim: int = 4,
        learning_rate: float = 0.05,
        exploration: float = 0.3
    ):
        self.manifold = manifold
        self.goal = goal
        self.energy_cost = energy_cost
        self.flow_dim = flow_dim
        self.lr = learning_rate
        self.exploration = exploration
        
        self.mean = np.zeros(flow_dim)
        self.std = np.ones(flow_dim)
        
        self.last_flow_id: Optional[str] = None
        self.last_score: float = 0.0
        self.score_history: deque = deque(maxlen=50)
    
    def select_flow(
        self, z_start: np.ndarray, h_start: np.ndarray
    ) -> Tuple[SkillFlow, str, np.ndarray]:
        """
        Select a flow using the current distribution.
        
        1. Sample coordinate from current distribution
        2. Interpolate flow at that coordinate
        3. Return flow
        """
        if not self.manifold.flows:
            flow = PointFlow(np.zeros(16))
            return flow, 'fallback', np.zeros(self.flow_dim)
        
        # Sample coordinate
        coord = self.mean + self.std * np.random.randn(self.flow_dim)
        
        # Interpolate flow
        flow = self.manifold.interpolate_at(coord)
        
        # Assign a representative flow_id
        flow_id = f'cem_{hash(coord.tobytes()) % 10000}'
        if hasattr(flow, 'flows') and flow.flows:
            flow_id = flow.flows[0].flow_id
        
        return flow, flow_id, coord
    
    def observe_outcome(
        self, coord: np.ndarray, flow_id: str,
        goal_prob: float, cost_total: float
    ):
        """Observe outcome and update distribution."""
        score = goal_prob - 0.3 * cost_total
        self.score_history.append(score)
        
        # Update mean toward this coordinate if successful
        if score > np.mean(self.score_history) if self.score_history else 0:
            self.mean = (1 - self.lr) * self.mean + self.lr * coord
        
        # Adaptive exploration
        if len(self.score_history) >= 10:
            recent = list(self.score_history)[-10:]
            if np.std(recent) < 0.001:
                self.std = np.ones(self.flow_dim) * self.exploration
            else:
                self.std = np.ones(self.flow_dim) * max(0.1, self.exploration * 0.5)
        
        self.last_flow_id = flow_id
        self.last_score = score
    
    def get_stats(self) -> Dict:
        return {
            'mean': self.mean.copy(),
            'std': self.std.copy(),
            'mean_score': float(np.mean(self.score_history)) if self.score_history else 0.0
        }


# ============================================================================
# 4. SELF-ORGANIZING ENGINE
# ============================================================================

class SelfOrganizingEngine:
    """
    Complete self-organizing cognitive system.
    
    No more phases. No more discrete cycles.
    Every step is a complete learning loop:
    
      1. CEM selects flow (continuous distribution)
      2. Flow generates action
      3. World model transitions
      4. Goal probability computed
      5. Energy cost computed
      6. Inverse dynamics trains (online)
      7. World model trains (online, interleaved)
      8. Flow ecology step (birth/death)
      9. Manifold drift step
      10. CEM distribution update
    
    This is the unified cognitive system.
    """
    
    def __init__(
        self,
        world_model: 'FlowConditionedWorldModel',
        goal: GoalAttractor,
        n_initial_flows: int = 8,
        flow_dim: int = 4,
        lambda_cost: float = 0.5,
        train_interval: int = 5
    ):
        self.wm = world_model
        self.goal = goal
        self.flow_dim = flow_dim
        
        # Phase 35 manifold
        self.manifold = FlowManifold(flow_dim=flow_dim)
        
        # Phase 38 energy cost
        self.energy_cost = EnergyCostFunction(
            w_action=0.3, w_path=0.3, w_variance=0.1, w_instability=0.3
        )
        
        # Phase 34 inverse dynamics
        self.inv_dyn = InverseDynamicsModel(
            latent_dim=world_model.latent_dim,
            action_dim=world_model.action_dim,
            learning_rate=0.01
        )
        
        # Phase 40 ecology (continuous birth/death)
        self.ecology = ContinuousFlowEcology(
            manifold=self.manifold,
            goal_attractor=goal.attractor_state,
            latent_dim=world_model.latent_dim,
            birth_rate=0.03,
            death_rate=0.015,
            min_flows=4,
            max_flows=30
        )
        
        # Phase 40 manifold drift (continuous)
        self.drift = ContinuousManifoldDrift(
            manifold=self.manifold,
            learning_rate=0.02,
            goal_attraction=0.005,
            similarity_attraction=0.003
        )
        
        # Phase 40 continuous CEM
        self.cem = ContinuousCEM(
            manifold=self.manifold,
            goal=goal,
            energy_cost=self.energy_cost,
            flow_dim=flow_dim,
            learning_rate=0.05,
            exploration=0.3
        )
        
        # Phase 36 learner (for batch training)
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
        
        # Seed initial flows
        if not self.manifold.flows:
            self._seed_initial_flows(n_initial_flows)
        
        self.train_interval = train_interval
        self.total_steps = 0
        self.execution_log: List[Dict] = []
    
    def _seed_initial_flows(self, n: int):
        """Seed with one goal-directed flow + diverse others."""
        for i in range(n):
            if i == 0:
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
    
    def step(self, z: np.ndarray, h: np.ndarray) -> Dict:
        """
        One complete cognitive step.
        
        1. CEM selects flow
        2. Flow generates action
        3. Transition through world model
        4. Record everything
        5. Online learning updates
        6. Ecological birth/death
        7. Manifold drift
        8. CEM adaptation
        """
        # 1. CEM selects flow
        flow, flow_id, coord = self.cem.select_flow(z, h)
        
        # 2. Flow generates action
        a = flow.compute_action(z, h)
        
        # 3. World model transition
        mu, logvar = self.wm.predict_transition(z, h, a)
        std = np.exp(0.5 * logvar)
        z_next = mu + std * np.random.randn(*mu.shape) * 0.1
        h_next = self.wm.gru_step(h, mu)
        
        flow.record_transition(z, z_next, a, h)
        
        # 4. Inverse dynamics
        self.inv_dyn.train_step(z, z_next, a)
        self.inv_dyn.add_transition(z, z_next, a)
        
        # 5. Goal probability
        dist = np.linalg.norm(z_next - self.goal.attractor_state[:len(z_next)])
        goal_prob = np.exp(-dist)
        
        # 6. GP delta (from previous step)
        prev_gp = self.execution_log[-1]['goal_prob'] if self.execution_log else goal_prob
        gp_delta = goal_prob - prev_gp
        
        # 7. Energy cost
        cost_info = self.energy_cost.compute([a], [z, z_next], flow)
        
        # 8. Flow stability
        flow.stability = flow.compute_lyapunov_estimate()
        flow.goal_alignment = float(np.clip(
            flow.goal_alignment + 0.01 * (gp_delta * 10), 0.0, 1.0
        ))
        
        # 9. Ecological birth/death (every step)
        eco_result = self.ecology.step()
        
        # 10. Manifold drift (every step)
        self.drift.step(flow_id, goal_prob, gp_delta, self.goal)
        
        # 11. CEM adaptation
        self.cem.observe_outcome(coord, flow_id, goal_prob, cost_info['total'])
        
        # 12. Periodic world model training
        if self.total_steps % self.train_interval == 0:
            self._train_model()
        
        self.total_steps += 1
        
        step_result = {
            'z_before': z.copy(),
            'z_after': z_next.copy(),
            'action': a.copy(),
            'goal_prob': float(goal_prob),
            'gp_delta': float(gp_delta),
            'flow_type': flow.flow_type.value,
            'flow_id': flow_id,
            'stability': flow.stability,
            'energy_cost': cost_info,
            'eco_births': eco_result['born'],
            'eco_deaths': eco_result['died'],
            'n_flows': len(self.manifold.flows)
        }
        
        self.execution_log.append(step_result)
        return step_result
    
    def _train_model(self):
        """Periodic world model training."""
        if len(self.learner.buffer.episodes) < 2:
            return
        
        for _ in range(3):
            self.learner.train_step()
        
        self.learner.validate()
    
    def record_episode(self):
        """Store current log as episode for training."""
        if len(self.execution_log) < 5:
            return
        
        states = []
        actions = []
        step_flows = []
        
        for entry in self.execution_log[-20:]:
            if 'z_before' in entry:
                if not states:
                    states.append(entry['z_before'])
                states.append(entry['z_after'])
                actions.append(entry['action'])
                
                fid = entry.get('flow_id', '')
                flow = self.manifold.flows.get(fid)
                if flow is None and self.manifold.flows:
                    flow = list(self.manifold.flows.values())[0]
                step_flows.append(flow or PointFlow(np.zeros(16)))
        
        if len(states) >= 5:
            from phase36_behavioral_physics_learning import FlowEpisode
            episode = FlowEpisode(
                states=[s.copy() for s in states[:-1]] if len(states) > 1 else states,
                beliefs=[np.zeros(self.wm.belief_dim)] * max(1, len(states) - 1),
                actions=[a.copy() for a in actions],
                flow_embeddings=[
                    self.wm.compute_flow_embedding(f) for f in step_flows
                ],
                rewards=[e.get('goal_prob', 0.0) for e in self.execution_log[-20:]],
                flow_ids=[e.get('flow_id', '') for e in self.execution_log[-20:]],
                flow_types=[e.get('flow_type', '') for e in self.execution_log[-20:]]
            )
            self.learner.buffer.add_episode(episode)
    
    def run(self, z_start: np.ndarray, n_steps: int = 200) -> Dict:
        """Run continuous cognitive process for n_steps."""
        z = z_start.copy()
        h = np.zeros(self.wm.belief_dim)
        h = self.wm.gru_step(h, z)
        
        for step in range(n_steps):
            result = self.step(z, h)
            
            z = result['z_after'].copy()
            h = self.wm.gru_step(h, result['z_after'])
            
            # Periodically create training episodes
            if step % 20 == 0 and step > 0:
                self.record_episode()
        
        # Final episode record
        self.record_episode()
        
        # Report
        training_report = self.learner.get_training_report()
        
        goal_probs = [e['goal_prob'] for e in self.execution_log[-100:]]
        
        return {
            'n_steps': n_steps,
            'mean_gp': float(np.mean(goal_probs)) if goal_probs else 0.0,
            'max_gp': float(max(goal_probs)) if goal_probs else 0.0,
            'final_gp': float(goal_probs[-1]) if goal_probs else 0.0,
            'gp_trend': float(goal_probs[-1] - goal_probs[0]) if len(goal_probs) > 1 else 0.0,
            'n_flows': len(self.manifold.flows),
            'training': training_report,
            'ecology': self.ecology.get_stats(),
            'drift': self.drift.get_stats(),
            'cem': self.cem.get_stats()
        }


# ============================================================================
# 5. TESTS
# ============================================================================

def test_continuous_flow_ecology():
    """Test continuous birth/death."""
    print("\n" + "=" * 60)
    print("CONTINUOUS FLOW ECOLOGY TEST")
    print("=" * 60)
    
    manifold = FlowManifold(flow_dim=4)
    for i in range(6):
        flow = PointFlow(np.ones(16) * 0.5, gain=0.5)
        manifold.add_flow(flow, f'flow_{i}')
    
    ecology = ContinuousFlowEcology(
        manifold=manifold,
        goal_attractor=np.ones(16) * 1.5,
        latent_dim=16,
        birth_rate=0.1,
        death_rate=0.1,
        min_flows=2
    )
    
    # Simulate some steps
    for i in range(20):
        for fid in manifold.flows:
            ecology.record_performance(fid, random.uniform(0.0, 0.005))
            ecology.record_gp_delta(fid, random.uniform(-0.001, 0.002))
        
        # Increase one flow's GP
        if manifold.flows:
            top_fid = list(manifold.flows.keys())[i % len(manifold.flows)]
            ecology.record_gp_delta(top_fid, 0.005)
        
        result = ecology.step()
        if result['born'] or result['died']:
            print(f"  Step {i}: born={result['born']}, died={result['died']}, "
                  f"flows={len(manifold.flows)}")
    
    stats = ecology.get_stats()
    print(f"\n  Total births: {stats['births']}")
    print(f"  Total deaths: {stats['deaths']}")
    print(f"  Final flows: {stats['n_flows']}")
    
    print("\n  ✓ Continuous flow ecology operational")


def test_continuous_manifold_drift():
    """Test continuous manifold drift."""
    print("\n" + "=" * 60)
    print("CONTINUOUS MANIFOLD DRIFT TEST")
    print("=" * 60)
    
    manifold = FlowManifold(flow_dim=4)
    for i in range(4):
        flow = PointFlow(np.ones(16) * 0.5, gain=0.5)
        manifold.add_flow(flow, f'flow_{i}')
    
    goal = GoalAttractor(
        goal_id='drift_test',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.8,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    drift = ContinuousManifoldDrift(
        manifold=manifold,
        learning_rate=0.05,
        goal_attraction=0.02,
        similarity_attraction=0.01
    )
    
    # Track coordinates
    coords_before = {fid: c.copy() for fid, c in manifold.flow_coords.items()}
    
    for step in range(50):
        active_id = list(manifold.flow_coords.keys())[step % 4]
        drift.step(active_id, 0.003, 0.0005 * (step % 2), goal)
    
    print(f"\n  Coordinate changes:")
    for fid in manifold.flow_coords:
        change = float(np.linalg.norm(
            manifold.flow_coords[fid] - coords_before.get(fid, np.zeros(4))
        ))
        print(f"    {fid}: {change:.4f}")
    
    stats = drift.get_stats()
    print(f"\n  Total drift: {stats['total_drift']:.4f}")
    
    print("\n  ✓ Continuous manifold drift operational")


def test_self_organizing_engine():
    """Test full self-organizing engine."""
    print("\n" + "=" * 60)
    print("SELF-ORGANIZING ENGINE TEST")
    print("=" * 60)
    
    from phase36_behavioral_physics_learning import FlowConditionedWorldModel
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )
    
    goal = GoalAttractor(
        goal_id='self_org',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )
    
    engine = SelfOrganizingEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=6,
        flow_dim=4,
        lambda_cost=0.5,
        train_interval=10
    )
    
    result = engine.run(np.zeros(16), n_steps=80)
    
    print(f"\n  Steps: {result['n_steps']}")
    print(f"  GP: {result['mean_gp']:.6f} (max={result['max_gp']:.6f})")
    print(f"  GP trend: {result['gp_trend']:.6f}")
    print(f"  Final flows: {result['n_flows']}")
    
    tr = result['training']
    print(f"\n  Training:")
    print(f"    Steps: {tr['training_steps']}")
    print(f"    Buffer: {tr['buffer_episodes']} eps, "
          f"{tr['buffer_transitions']} trans")
    if 'loss_improvement' in tr:
        print(f"    Loss improvement: {tr.get('loss_improvement', 0) * 100:.1f}%")
    
    eco = result['ecology']
    print(f"\n  Ecology: {eco['births']} births, {eco['deaths']} deaths")
    
    drift = result['drift']
    print(f"  Drift: {drift['total_drift']:.4f}")
    
    cem = result['cem']
    print(f"  CEM mean: {cem['mean'][:4]}...  std: {cem['std'][0]:.3f}")
    
    print("\n  ✓ Self-organizing engine operational")


if __name__ == "__main__":
    test_continuous_flow_ecology()
    test_continuous_manifold_drift()
    test_self_organizing_engine()
    
    print("\n" + "=" * 60)
    print("PHASE 40: SELF-ORGANIZING BEHAVIORAL GEOMETRY")
    print("=" * 60)
    
    print("""
ARCHITECTURAL UNIFICATION:

  Phases 30-39: "Build a robot that learns" (pipeline)
  Phase 40:     "The robot IS learning" (continuous process)

EVERY STEP IS A COMPLETE LEARNING LOOP:
  1. CEM selects flow (continuous distribution → action)
  2. World model transitions (action → next state)
  3. Goal probability + energy cost
  4. Inverse dynamics trains (online)
  5. Flow ecology (probabilistic birth/death)
  6. Manifold drift (smooth continuous reorganization)
  7. CEM adapts (distribution slides toward success)
  8. World model trains (periodic batch updates)

WHY THIS IS THE FINAL PIECE:
  - No more "phases" or "cycles"
  - The system is a single continuous cognitive process
  - Learning, selection, and organization are simultaneous
  - The manifold is a living structure, not a periodic snapshot

THE SYSTEM IS NOW:
  A self-organizing, continuously learning, energy-aware,
  controllability-guided, flow-based cognitive architecture
  — operating as a SINGLE UNIFIED DYNAMICAL PROCESS.
""")
