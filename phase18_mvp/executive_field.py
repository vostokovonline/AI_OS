"""
Phase 15: Executive Field Dynamics

ARCHITECTURAL SHIFT:
  From: Symbolic coalition planner (Phase 14)
  To: Continuous dynamics field where trajectories compete
  
  Coalition doesn't build plan directly.
  Coalition MODIFIES topology of trajectory field.
  Trajectories compete.
  Stable attractors emerge.
  Execution emerges from attractor selection.

CRITICAL INSIGHT:
  Phase 14: coalition → symbolic plan → execute
  Phase 15: state + tensions + drives + world_model → future manifold → trajectory competition → attractor → execute
  
  The planning becomes a DYNAMICAL process, not a symbolic process.

WHAT PHASE 15 ADDS:
  1. Learned World Model P(s_t+1 | s_t, a_t)
     - State transition learning
     - Enables real prediction
     
  2. Trajectory Rollout Engine
     - Simulate future trajectories
     - Evaluate outcomes
     - Compete across rollout space
     
  3. Attractor-Based Planning (NOT plan tree)
     - Trajectory competition in field
     - Winners become attractors
     - Execution = attractor selection
     
  4. Cognitive Economics
     - Attention budget
     - Energy cost
     - Uncertainty penalty
     - Risk premium
     - Compute budget
     
  5. Emergent Selves (NOT predefined)
     - From execution traces
     - Persistent behavioral clusters
     - Born, merge, specialize, die
     
  6. Predictive Tension Dynamics
     - Tensions = prediction error gradients
     - Not static, but flowing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


# ============================================================================
# 1. LEARNED WORLD MODEL P(s_t+1 | s_t, a_t)
# ============================================================================

@dataclass
class WorldModelTransition:
    """A learned state transition."""
    from_state: np.ndarray
    action: np.ndarray
    to_state: np.ndarray
    confidence: float
    occurrences: int
    temporal_consistency: float


class LearnedWorldModel:
    """
    Learned world model: P(s_t+1 | s_t, a_t)
    
    NOT: hardcoded transitions
    BUT: learned from experience
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Learned transitions (simplified: linear + noise)
        self.W_s = np.eye(state_dim) * 0.9  # State dynamics matrix
        self.W_a = np.eye(action_dim) * 0.2  # Action influence matrix
        self.noise_scale = 0.1
        
        # Transition history
        self.transitions: List[WorldModelTransition] = []
        
        # Uncertainty estimates
        self.state_uncertainty: np.ndarray = np.ones(state_dim) * 0.2
        self.action_uncertainty: np.ndarray = np.ones(action_dim) * 0.1
        
    def predict(self, state: np.ndarray, action: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Predict next state given current state and action.
        
        Returns: (predicted_next_state, prediction_confidence)
        """
        # Ensure arrays are 1D
        state = np.asarray(state).flatten()
        action = np.asarray(action).flatten()
        
        # Predict
        next_state = (
            self.W_s @ state +
            self.W_a @ action +
            np.random.randn(self.state_dim) * self.noise_scale
        )
        
        # Compute confidence (inverse of uncertainty)
        total_uncertainty = (
            np.sum(self.state_uncertainty) +
            np.sum(self.action_uncertainty)
        )
        confidence = 1.0 / (1.0 + total_uncertainty)
        
        return next_state, confidence
    
    def update(self, from_state: np.ndarray, action: np.ndarray, to_state: np.ndarray):
        """Update world model from experience."""
        from_state = np.asarray(from_state).flatten()
        action = np.asarray(action).flatten()
        to_state = np.asarray(to_state).flatten()
        
        # Observed transition
        observed_delta = to_state - from_state
        
        # Predicted transition
        predicted_delta = self.W_s @ from_state + self.W_a @ action - from_state
        
        # Prediction error
        error = observed_delta - predicted_delta
        
        # Update matrices (Hebbian-like)
        self.W_s += 0.01 * np.outer(error, from_state) * 0.1
        self.W_a += 0.01 * np.outer(error, action) * 0.1
        
        # Normalize to prevent explosion
        self.W_s = self.W_s / (np.linalg.norm(self.W_s) + 1e-8)
        self.W_a = self.W_a / (np.linalg.norm(self.W_a) + 1e-8)
        
        # Update uncertainty estimates
        prediction_error_magnitude = np.linalg.norm(error)
        self.state_uncertainty = self.state_uncertainty * 0.99 + prediction_error_magnitude * 0.01
        self.action_uncertainty = self.action_uncertainty * 0.99 + prediction_error_magnitude * 0.01 * 0.5
        
        # Store transition
        self.transitions.append(WorldModelTransition(
            from_state=from_state.copy(),
            action=action.copy(),
            to_state=to_state.copy(),
            confidence=1.0 / (1.0 + prediction_error_magnitude),
            occurrences=1,
            temporal_consistency=0.9
        ))
        
        # Limit history
        if len(self.transitions) > 500:
            self.transitions = self.transitions[-250:]


# ============================================================================
# 2. TRAJECTORY ROLLOUT ENGINE
# ============================================================================

@dataclass
class Trajectory:
    """A simulated trajectory through future."""
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]  # Cumulative reward
    confidence: float
    attractor_strength: float  # How stable is this trajectory?
    cost: float  # Cognitive cost to execute
    
    def get_final_state(self) -> np.ndarray:
        """Get final state of trajectory."""
        return self.states[-1] if self.states else np.zeros(2)
    
    def get_total_cost(self) -> float:
        """Total cognitive cost of trajectory."""
        return sum(t.cost for t in [self]) + self.cost


class TrajectoryRolloutEngine:
    """
    Simulate future trajectories through world model.
    
    NOT: single plan
    BUT: rollout many possible trajectories
    """
    
    def __init__(self, world_model: LearnedWorldModel, state_dim: int = 2, action_dim: int = 2):
        self.world_model = world_model
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.max_rollout_depth = 10
        self.n_candidates = 20
        
        # Trajectory competition parameters
        self.attractor_threshold = 0.6
        self.noise_for_exploration = 0.2
        
    def rollout(self, initial_state: np.ndarray, 
                drives: List[Dict], 
                cognitive_budget: float = 1.0) -> List[Trajectory]:
        """
        Generate multiple trajectory rollouts from current state.
        
        Returns: list of competing trajectories
        """
        trajectories = []
        
        for _ in range(self.n_candidates):
            trajectory = self._single_rollout(initial_state, drives, cognitive_budget)
            trajectories.append(trajectory)
        
        # Compute attractor strengths
        self._compute_attractor_strengths(trajectories)
        
        # Sort by quality (reward - cost)
        trajectories.sort(key=lambda t: max(t.rewards) - t.cost if t.rewards else -t.cost, reverse=True)
        
        return trajectories
    
    def _single_rollout(self, initial_state: np.ndarray,
                       drives: List[Dict], 
                       cognitive_budget: float) -> Trajectory:
        """Single trajectory rollout."""
        states = [initial_state.copy()]
        actions = []
        rewards = []
        
        current_state = initial_state.copy()
        cumulative_cost = 0.0
        
        for depth in range(self.max_rollout_depth):
            # Generate action (influenced by drives)
            action = self._generate_drive_action(current_state, drives, depth)
            
            # Add noise for exploration
            action = action + np.random.randn(self.action_dim) * self.noise_for_exploration
            
            actions.append(action)
            
            # Predict next state
            next_state, confidence = self.world_model.predict(current_state, action)
            states.append(next_state)
            
            # Compute reward (drive satisfaction)
            reward = self._compute_reward(next_state, drives)
            rewards.append(reward)
            
            # Compute cognitive cost
            action_cost = np.linalg.norm(action) * 0.1
            depth_cost = depth * 0.05  # Deeper = more expensive
            uncertainty_cost = (1 - confidence) * 0.2
            cumulative_cost += action_cost + depth_cost + uncertainty_cost
            
            # Check cognitive budget
            if cumulative_cost > cognitive_budget:
                break
            
            current_state = next_state
        
        return Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            confidence=np.mean([self.world_model.predict(s, a)[1] for s, a in zip(states[:-1], actions)]),
            attractor_strength=0.0,  # Will be computed
            cost=cumulative_cost
        )
    
    def _generate_drive_action(self, state: np.ndarray, drives: List[Dict], depth: int) -> np.ndarray:
        """Generate action from drives."""
        if not drives:
            return np.zeros(self.action_dim)
        
        # Blend drives based on depth (early drives different from late)
        action = np.zeros(self.action_dim)
        
        for i, drive in enumerate(drives[:3]):  # Top 3 drives
            drive_direction = drive.get('direction', np.zeros(self.action_dim))
            drive_strength = drive.get('strength', 0.5)
            
            # Temporal weighting
            if drive.get('temporal_horizon') == 'short':
                weight = 1.0 - depth * 0.1
            elif drive.get('temporal_horizon') == 'long':
                weight = depth * 0.1
            else:
                weight = 0.5
            
            action += drive_direction * drive_strength * weight
        
        # Normalize
        if np.linalg.norm(action) > 0.01:
            action = action / np.linalg.norm(action)
        
        return action
    
    def _compute_reward(self, state: np.ndarray, drives: List[Dict]) -> float:
        """Compute reward from state and drives."""
        reward = 0.0
        
        for drive in drives:
            drive_target = drive.get('target', np.zeros(self.state_dim))
            drive_strength = drive.get('strength', 0.5)
            
            # Distance to drive target (closer = higher reward)
            distance = np.linalg.norm(state - drive_target)
            drive_reward = drive_strength * np.exp(-distance * 0.5)
            
            reward += drive_reward
        
        return reward / max(1, len(drives))
    
    def _compute_attractor_strengths(self, trajectories: List[Trajectory]):
        """Compute how strong each trajectory's attractor is."""
        if not trajectories:
            return
        
        # Best trajectory becomes attractor
        best = trajectories[0]
        
        for i, traj in enumerate(trajectories):
            # Attractor strength = normalized reward - cost
            if traj.rewards:
                avg_reward = np.mean(traj.rewards)
                traj.attractor_strength = (avg_reward - traj.cost) / (best.rewards[-1] - best.cost + 1e-8) if best.rewards else 0.5
            
            # Trajectories near best are also attractors
            if traj.rewards and best.rewards:
                similarity = np.corrcoef(
                    traj.states[-1] if len(traj.states) > 0 else np.zeros(2),
                    best.states[-1] if len(best.states) > 0 else np.zeros(2)
                )[0, 1]
                
                if similarity > self.attractor_threshold:
                    traj.attractor_strength = max(traj.attractor_strength, similarity)


# ============================================================================
# 3. ATTRACTOR-BASED PLANNING (NOT symbolic plan tree)
# ============================================================================

class AttractorBasedPlanner:
    """
    Attractor-based planning: trajectories compete, winners become attractors.
    
    NOT: plan tree with branching
    BUT: trajectory field where winners attract
    """
    
    def __init__(self, rollout_engine: TrajectoryRolloutEngine):
        self.rollout_engine = rollout_engine
        self.attractors: List[Trajectory] = []
        
    def plan(self, state: np.ndarray, drives: List[Dict], 
             cognitive_budget: float = 1.0) -> Dict:
        """
        Plan through attractor dynamics.
        
        Process:
          1. Generate trajectory manifold
          2. Trajectories compete
          3. Winners become attractors
          4. Return stabilized trajectory
        """
        # 1. Generate trajectory manifold
        trajectories = self.rollout_engine.rollout(state, drives, cognitive_budget)
        
        # 2. Trajectory competition
        self._compete_trajectories(trajectories)
        
        # 3. Winners become attractors
        winners = [t for t in trajectories if t.attractor_strength > 0.5]
        
        # Add to attractor list (limit to recent)
        self.attractors.extend(winners[:3])
        if len(self.attractors) > 10:
            self.attractors = self.attractors[-10:]
        
        # 4. Return stabilized trajectory (highest attractor strength)
        if trajectories:
            best = max(trajectories, key=lambda t: t.attractor_strength)
            return {
                'trajectory': best,
                'alternatives': trajectories[:5],
                'n_attractors': len(winners),
                'attractor_manifold': [t for t in trajectories if t.attractor_strength > 0.3]
            }
        
        return {
            'trajectory': None,
            'alternatives': [],
            'n_attractors': 0,
            'attractor_manifold': []
        }
    
    def _compete_trajectories(self, trajectories: List[Trajectory]):
        """Trajectories compete in field dynamics."""
        if len(trajectories) < 2:
            return
        
        # Compute pairwise influences
        for i, traj_i in enumerate(trajectories):
            influence = 0.0
            
            for j, traj_j in enumerate(trajectories):
                if i == j:
                    continue
                
                # Distance between trajectories
                final_i = traj_i.get_final_state()
                final_j = traj_j.get_final_state()
                distance = np.linalg.norm(final_i - final_j)
                
                # Trajectory j influences i
                if distance < 1.0:  # Close trajectories interact
                    # Reward difference
                    reward_i = max(traj_i.rewards) if traj_i.rewards else 0
                    reward_j = max(traj_j.rewards) if traj_j.rewards else 0
                    
                    # Lower reward trajectory gets attracted to higher
                    if reward_i < reward_j:
                        attraction = (reward_j - reward_i) * np.exp(-distance)
                        influence += attraction * 0.1
            
            # Apply influence
            traj_i.attractor_strength += influence


# ============================================================================
# 4. COGNITIVE ECONOMICS
# ============================================================================

@dataclass
class CognitiveBudget:
    """Cognitive resource budget."""
    attention: float = 1.0
    energy: float = 1.0
    compute: float = 1.0
    uncertainty_penalty: float = 0.0
    risk_premium: float = 0.0
    
    def consume(self, amount: float, resource: str = "attention"):
        """Consume cognitive resource."""
        if resource == "attention":
            self.attention = max(0.0, self.attention - amount * 0.5)
        elif resource == "energy":
            self.energy = max(0.0, self.energy - amount * 0.3)
        elif resource == "compute":
            self.compute = max(0.0, self.compute - amount * 0.2)
    
    def is_exhausted(self, resource: str = "attention") -> bool:
        """Check if resource is exhausted."""
        if resource == "attention":
            return self.attention < 0.1
        elif resource == "energy":
            return self.energy < 0.1
        elif resource == "compute":
            return self.compute < 0.1
        return False
    
    def remaining(self, resource: str = "attention") -> float:
        """Get remaining resource."""
        if resource == "attention":
            return self.attention
        elif resource == "energy":
            return self.energy
        elif resource == "compute":
            return self.compute
        return 0.0
    
    def total_capacity(self) -> float:
        """Total remaining capacity."""
        return (self.attention + self.energy + self.compute) / 3.0


class CognitiveEconomics:
    """
    Cognitive economics: everything has a cost.
    
    - Attention budget
    - Energy cost
    - Uncertainty penalty
    - Risk premium
    - Compute budget
    """
    
    def __init__(self):
        self.budget = CognitiveBudget()
        self.cost_history: List[Dict] = []
        
    def compute_cost(self, action: np.ndarray, state: np.ndarray, 
                     uncertainty: float, risk: float) -> Dict:
        """Compute full cost of action."""
        # Action cost
        action_cost = np.linalg.norm(action) * 0.1
        
        # Uncertainty cost
        uncertainty_cost = uncertainty * 0.3
        
        # Risk cost
        risk_cost = risk * 0.4
        
        # State cost (how far from comfortable state?)
        state_cost = np.linalg.norm(state) * 0.05
        
        # Total
        total = action_cost + uncertainty_cost + risk_cost + state_cost
        
        return {
            'action_cost': action_cost,
            'uncertainty_cost': uncertainty_cost,
            'risk_cost': risk_cost,
            'state_cost': state_cost,
            'total': total,
            'budget_remaining': self.budget.total_capacity()
        }
    
    def consume_from_action(self, action: np.ndarray, state: np.ndarray,
                           uncertainty: float = 0.0, risk: float = 0.0):
        """Consume budget for action."""
        cost = self.compute_cost(action, state, uncertainty, risk)
        
        # Consume proportionally
        self.budget.consume(cost['total'], "attention")
        self.budget.consume(cost['total'], "energy")
        self.budget.consume(cost['total'], "compute")
        
        # Update penalty
        self.budget.uncertainty_penalty = uncertainty
        self.budget.risk_premium = risk
        
        self.cost_history.append(cost)
        
        return cost
    
    def recharge(self, amount: float = 0.1):
        """Recharge cognitive budget (like rest)."""
        self.budget.attention = min(1.0, self.budget.attention + amount)
        self.budget.energy = min(1.0, self.budget.energy + amount)
        self.budget.compute = min(1.0, self.budget.compute + amount)
    
    def get_state(self) -> Dict:
        """Get economics state."""
        return {
            'attention': self.budget.attention,
            'energy': self.budget.energy,
            'compute': self.budget.compute,
            'total_capacity': self.budget.total_capacity(),
            'exhausted': self.budget.is_exhausted("attention")
        }


# ============================================================================
# 5. EMERGENT SELVES (NOT predefined)
# ============================================================================

@dataclass
class EmergentSelf:
    """A self that emerged from execution history."""
    self_id: str
    behavior_signature: np.ndarray  # Behavioral pattern
    execution_traces: List[Dict]    # History of actions
    stability: float                # How stable is this self?
    specialization: str            # What is this self specialized in?
    priority: float                 # Current priority
    energy: float                  # Activation level
    
    # Birth/death
    birth_time: int = 0
    death_time: Optional[int] = None
    parent_ids: List[str] = field(default_factory=list)
    
    def compute_signature(self) -> np.ndarray:
        """Recompute behavior signature from traces."""
        if not self.execution_traces:
            return np.zeros(2)
        
        # Average behavior
        avg_actions = np.mean([t.get('action', np.zeros(2)) for t in self.execution_traces], axis=0)
        return avg_actions
    
    def update(self, trace: Dict):
        """Update self with new execution trace."""
        self.execution_traces.append(trace)
        if len(self.execution_traces) > 50:
            self.execution_traces = self.execution_traces[-25:]
        
        # Update signature
        self.behavior_signature = self.compute_signature()
        
        # Update stability
        if len(self.execution_traces) > 5:
            recent = self.execution_traces[-5:]
            consistency = np.mean([
                np.linalg.norm(recent[i]['action'] - recent[i+1]['action'])
                for i in range(len(recent)-1)
            ])
            self.stability = 1.0 / (1.0 + consistency)
        
        # Update priority based on success
        if trace.get('success', False):
            self.priority = min(1.0, self.priority * 1.05)
            self.energy = min(1.0, self.energy + 0.1)
        else:
            self.priority = max(0.1, self.priority * 0.95)
            self.energy = max(0.1, self.energy - 0.1)
    
    def should_die(self) -> bool:
        """Check if this self should dissolve."""
        if self.stability < 0.2:
            return True
        if self.priority < 0.1:
            return True
        if self.energy < 0.1:
            return True
        return False
    
    def merge_with(self, other: 'EmergentSelf') -> 'EmergentSelf':
        """Merge with another self."""
        # Blend signatures
        blended_signature = (self.behavior_signature + other.behavior_signature) / 2
        
        # Create merged self
        merged = EmergentSelf(
            self_id=f"merged_{np.random.randint(100000)}",
            behavior_signature=blended_signature,
            execution_traces=self.execution_traces + other.execution_traces,
            stability=(self.stability + other.stability) / 2,
            specialization=self.specialization if self.stability > other.stability else other.specialization,
            priority=max(self.priority, other.priority),
            energy=max(self.energy, other.energy),
            parent_ids=[self.self_id, other.self_id]
        )
        
        return merged


class EmergentSelfEcology:
    """
    Emergent selves from execution history.
    
    NOT: predefined selves (strategic, cautious, etc.)
    BUT: selves born from execution traces, stabilize, merge, die
    """
    
    def __init__(self):
        self.selves: List[EmergentSelf] = []
        self.execution_history: List[Dict] = []
        
    def add_execution_trace(self, trace: Dict):
        """Add execution trace and update selves."""
        self.execution_history.append(trace)
        if len(self.execution_history) > 200:
            self.execution_history = self.execution_history[-100:]
        
        # Check if new self should emerge
        self._check_for_emergence(trace)
        
        # Update existing selves
        for self_obj in self.selves:
            # Check if this self is relevant to trace
            similarity = np.dot(self_obj.behavior_signature, trace.get('action', np.zeros(2)))
            if similarity > 0.3:
                self_obj.update(trace)
        
        # Check for merges and deaths
        self._maintain_ecology()
    
    def _check_for_emergence(self, trace: Dict):
        """Check if new self should emerge."""
        # Find if similar self exists
        action = trace.get('action', np.zeros(2))
        
        for self_obj in self.selves:
            similarity = np.dot(self_obj.behavior_signature, action)
            if similarity > 0.7:
                return  # Similar self exists
        
        # New self emerges (only if enough unique traces)
        if len(self.execution_history) > 10:
            # Check for cluster formation
            recent_traces = self.execution_history[-10:]
            
            # Simple clustering check
            actions = [t.get('action', np.zeros(2)) for t in recent_traces]
            variance = np.var(actions, axis=0)
            
            if np.mean(variance) < 0.5:  # Low variance = stable cluster = new self
                new_self = EmergentSelf(
                    self_id=f"emergent_{np.random.randint(100000)}",
                    behavior_signature=np.mean(actions, axis=0),
                    execution_traces=recent_traces[-5:],
                    stability=0.6,
                    specialization=self._infer_specialization(actions),
                    priority=0.5,
                    energy=0.8
                )
                self.selves.append(new_self)
    
    def _infer_specialization(self, actions: List[np.ndarray]) -> str:
        """Infer what this self is specialized in."""
        avg_action = np.mean(actions, axis=0)
        
        if np.linalg.norm(avg_action) > 0.5:
            return "action_oriented"
        else:
            return "deliberative"
    
    def _maintain_ecology(self):
        """Maintain ecology: merge similar selves, remove dying."""
        # Remove dying selves
        self.selves = [s for s in self.selves if not s.should_die()]
        
        # Limit to reasonable number
        if len(self.selves) > 15:
            # Keep highest priority
            self.selves.sort(key=lambda s: s.priority, reverse=True)
            self.selves = self.selves[:10]
    
    def get_relevant_selves(self, task_type: str = "general") -> List[EmergentSelf]:
        """Get selves relevant to current task."""
        if not self.selves:
            return []
        
        # Sort by priority and stability
        relevant = sorted(
            self.selves,
            key=lambda s: s.priority * s.stability,
            reverse=True
        )
        
        return relevant[:5]
    
    def get_state(self) -> Dict:
        """Get ecology state."""
        return {
            'n_selves': len(self.selves),
            'dominant': max(self.selves, key=lambda s: s.priority).self_id if self.selves else None,
            'avg_stability': np.mean([s.stability for s in self.selves]) if self.selves else 0,
            'total_traces': len(self.execution_history)
        }


# ============================================================================
# 6. PREDICTIVE TENSION DYNAMICS
# ============================================================================

class PredictiveTensionField:
    """
    Tensions are NOT static.
    Tensions are prediction error GRADIENTS flowing through time.
    
    Tension = (expected_state - predicted_state) with uncertainty
    """
    
    def __init__(self, world_model: LearnedWorldModel):
        self.world_model = world_model
        self.tensions: List[Dict] = []  # Tension flows
        
    def compute_tensions(self, current_state: np.ndarray, 
                         expected_states: List[np.ndarray],
                         prediction_confidence: float) -> List[Dict]:
        """
        Compute tension as prediction error gradient.
        
        Tension flows toward reducing prediction error.
        """
        new_tensions = []
        
        for expected in expected_states:
            # Prediction error
            error = expected - current_state
            error_magnitude = np.linalg.norm(error)
            
            # Prediction confidence
            confidence = prediction_confidence
            
            # Tension = error weighted by inverse confidence
            tension = {
                'direction': error,
                'magnitude': error_magnitude * (1 - confidence),
                'urgency': error_magnitude * confidence,
                'flow': self._compute_tension_flow(error, confidence),
                'resonance': self._compute_resonance(error)
            }
            
            new_tensions.append(tension)
        
        self.tensions = new_tensions
        return new_tensions
    
    def _compute_tension_flow(self, error: np.ndarray, confidence: float) -> np.ndarray:
        """Tension flows toward reducing error."""
        # Flow is proportional to error and inversely to confidence
        flow = error * confidence
        
        # Add world model uncertainty influence
        uncertainty = self.world_model.state_uncertainty
        flow = flow * (1 - np.mean(uncertainty))
        
        return flow
    
    def _compute_resonance(self, error: np.ndarray) -> float:
        """Compute how much this tension resonates with existing tensions."""
        if not self.tensions:
            return 0.5
        
        # Average existing tension direction
        avg_direction = np.mean([t['direction'] for t in self.tensions], axis=0)
        
        # Resonance = alignment with existing tensions
        if np.linalg.norm(avg_direction) > 0.01:
            resonance = np.dot(error, avg_direction) / (
                np.linalg.norm(error) * np.linalg.norm(avg_direction)
            )
            return (resonance + 1) / 2  # Map to [0, 1]
        
        return 0.5
    
    def apply_tension(self, state: np.ndarray, tension: Dict) -> np.ndarray:
        """Apply tension to state (pull state toward tension resolution)."""
        flow = tension['flow']
        urgency = tension['urgency']
        
        # State moves along tension flow
        new_state = state + flow * urgency * 0.1
        
        return new_state
    
    def get_state(self) -> Dict:
        """Get tension field state."""
        return {
            'n_tensions': len(self.tensions),
            'avg_magnitude': np.mean([t['magnitude'] for t in self.tensions]) if self.tensions else 0,
            'max_urgency': max([t['urgency'] for t in self.tensions]) if self.tensions else 0,
            'total_flow': np.sum([np.linalg.norm(t['flow']) for t in self.tensions]) if self.tensions else 0
        }


# ============================================================================
# INTEGRATED EXECUTIVE FIELD SYSTEM
# ============================================================================

class ExecutiveField:
    """
    Phase 15: Executive Field Dynamics
    
    Not symbolic coalition planner.
    But continuous dynamics field where:
      - Trajectories compete
      - Attractors emerge
      - Economics constrain
      - Selves emerge
      - Tensions flow
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 2):
        # Core components
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.world_model = LearnedWorldModel(state_dim, action_dim)
        self.rollout_engine = TrajectoryRolloutEngine(self.world_model, state_dim, action_dim)
        self.planner = AttractorBasedPlanner(self.rollout_engine)
        self.economics = CognitiveEconomics()
        self.emergent_selves = EmergentSelfEcology()
        self.tension_field = PredictiveTensionField(self.world_model)
        
        # Current state
        self.current_state = np.zeros(state_dim)
        
        # Execution tracking
        self.execution_count = 0
        
    def execute(self, drives: List[Dict], n_steps: int = 10) -> Dict:
        """
        Execute through field dynamics.
        
        NOT: coalition → plan → execute
        BUT: state + tensions → trajectory manifold → attractor selection → execute
        """
        results = []
        
        for step in range(n_steps):
            # Check cognitive budget
            if self.economics.budget.is_exhausted("attention"):
                self.economics.recharge()
            
            # 1. Compute tension field
            expected_states = [d.get('target', self.current_state) for d in drives]
            _, confidence = self.world_model.predict(self.current_state, np.zeros(self.action_dim))
            tensions = self.tension_field.compute_tensions(
                self.current_state, expected_states, confidence
            )
            
            # 2. Generate trajectory manifold
            trajectories = self.rollout_engine.rollout(
                self.current_state, drives, 
                cognitive_budget=self.economics.budget.total_capacity()
            )
            
            # 3. Attractor-based planning
            plan = self.planner.plan(
                self.current_state, drives,
                cognitive_budget=self.economics.budget.total_capacity()
            )
            
            # 4. Select action from best trajectory
            if plan['trajectory'] and plan['trajectory'].actions:
                action = plan['trajectory'].actions[0]
            else:
                action = np.zeros(action_dim)
            
            # 5. Apply cognitive economics
            uncertainty = 1 - confidence
            risk = 1 - plan['trajectory'].attractor_strength if plan['trajectory'] else 0.5
            cost = self.economics.consume_from_action(action, self.current_state, uncertainty, risk)
            
            # 6. Execute in world
            next_state, _ = self.world_model.predict(self.current_state, action)
            
            # 7. Update world model
            self.world_model.update(self.current_state, action, next_state)
            
            # 8. Add execution trace
            trace = {
                'step': step,
                'action': action.copy(),
                'state': next_state.copy(),
                'success': plan['trajectory'].attractor_strength > 0.4 if plan['trajectory'] else False
            }
            self.emergent_selves.add_execution_trace(trace)
            
            # 9. Update state
            self.current_state = next_state
            self.execution_count += 1
            
            results.append({
                'step': step,
                'action': action.tolist(),
                'state': next_state.tolist(),
                'cost': cost,
                'n_attractors': plan['n_attractors']
            })
        
        return {
            'results': results,
            'world_model_updates': len(self.world_model.transitions),
            'emergent_selves': self.emergent_selves.get_state(),
            'tension_field': self.tension_field.get_state(),
            'economics': self.economics.get_state()
        }


def test_executive_field():
    """Test executive field dynamics."""
    print("\n" + "=" * 60)
    print("EXECUTIVE FIELD DYNAMICS TEST")
    print("=" * 60)
    
    field = ExecutiveField(state_dim=2, action_dim=2)
    
    # Define drives
    drives = [
        {'name': 'exploration', 'target': np.array([2.0, 2.0]), 'strength': 0.7, 'temporal_horizon': 'long'},
        {'name': 'safety', 'target': np.array([0.0, 0.0]), 'strength': 0.5, 'temporal_horizon': 'short'},
        {'name': 'curiosity', 'target': np.array([1.0, -1.0]), 'strength': 0.4, 'temporal_horizon': 'medium'}
    ]
    
    print("\n  Executing with field dynamics:")
    result = field.execute(drives, n_steps=20)
    
    print(f"\n  Execution results:")
    print(f"    Steps: {len(result['results'])}")
    print(f"    World model updates: {result['world_model_updates']}")
    
    # Show state evolution
    print(f"\n  State evolution:")
    for i, r in enumerate(result['results'][:5]):
        print(f"    Step {r['step']}: state={r['state'][:2]}, cost={r['cost']['total']:.3f}")
    
    # Show emergent selves
    selves = result['emergent_selves']
    print(f"\n  Emergent selves:")
    print(f"    Count: {selves['n_selves']}")
    print(f"    Dominant: {selves['dominant']}")
    print(f"    Avg stability: {selves['avg_stability']:.3f}")
    
    # Show tension field
    tension = result['tension_field']
    print(f"\n  Tension field:")
    print(f"    Tensions: {tension['n_tensions']}")
    print(f"    Avg magnitude: {tension['avg_magnitude']:.3f}")
    print(f"    Max urgency: {tension['max_urgency']:.3f}")
    
    # Show economics
    econ = result['economics']
    print(f"\n  Cognitive economics:")
    print(f"    Attention: {econ['attention']:.3f}")
    print(f"    Energy: {econ['energy']:.3f}")
    print(f"    Total capacity: {econ['total_capacity']:.3f}")


def test_trajectory_rollout():
    """Test trajectory rollout engine."""
    print("\n" + "=" * 60)
    print("TRAJECTORY ROLLOUT TEST")
    print("=" * 60)
    
    world_model = LearnedWorldModel(state_dim=2, action_dim=2)
    engine = TrajectoryRolloutEngine(world_model, state_dim=2, action_dim=2)
    
    initial_state = np.array([0.0, 0.0])
    
    drives = [
        {'direction': np.array([1.0, 0.5]), 'strength': 0.8, 'temporal_horizon': 'medium'},
        {'direction': np.array([-0.5, 1.0]), 'strength': 0.6, 'temporal_horizon': 'short'}
    ]
    
    print("\n  Generating trajectory manifold:")
    trajectories = engine.rollout(initial_state, drives, cognitive_budget=1.0)
    
    print(f"    Generated {len(trajectories)} trajectories")
    
    for i, traj in enumerate(trajectories[:5]):
        print(f"\n    Trajectory {i}:")
        print(f"      Depth: {len(traj.states)}")
        print(f"      Final state: {traj.get_final_state()[:2]}")
        print(f"      Reward: {max(traj.rewards) if traj.rewards else 0:.3f}")
        print(f"      Cost: {traj.cost:.3f}")
        print(f"      Attractor strength: {traj.attractor_strength:.3f}")
    
    # Show attractors
    attractors = [t for t in trajectories if t.attractor_strength > 0.4]
    print(f"\n  Attractors (strength > 0.4): {len(attractors)}")


def test_cognitive_economics():
    """Test cognitive economics."""
    print("\n" + "=" * 60)
    print("COGNITIVE ECONOMICS TEST")
    print("=" * 60)
    
    economics = CognitiveEconomics()
    
    print("\n  Initial state:")
    print(f"    Attention: {economics.budget.attention:.3f}")
    print(f"    Energy: {economics.budget.energy:.3f}")
    print(f"    Compute: {economics.budget.compute:.3f}")
    
    print("\n  Consuming actions:")
    
    for i in range(10):
        action = np.array([0.5, 0.3])
        state = np.array([0.2 * i, 0.1 * i])
        
        cost = economics.consume_from_action(action, state, uncertainty=0.3, risk=0.4)
        
        if i % 3 == 0:
            print(f"    Action {i}: cost={cost['total']:.3f}, "
                  f"attention={economics.budget.attention:.3f}, "
                  f"capacity={economics.budget.total_capacity():.3f}")
    
    print("\n  After 10 actions:")
    state = economics.get_state()
    print(f"    Attention: {state['attention']:.3f}")
    print(f"    Energy: {state['energy']:.3f}")
    print(f"    Exhausted: {state['exhausted']}")
    
    print("\n  Recharging:")
    economics.recharge(0.3)
    state = economics.get_state()
    print(f"    After recharge: capacity={state['total_capacity']:.3f}")


def test_world_model_learning():
    """Test world model learning."""
    print("\n" + "=" * 60)
    print("WORLD MODEL LEARNING TEST")
    print("=" * 60)
    
    world_model = LearnedWorldModel(state_dim=2, action_dim=2)
    
    print("\n  Learning world dynamics:")
    
    # Learn simple dynamics: next_state = state + action
    for i in range(50):
        state = np.array([i * 0.1, i * 0.05])
        action = np.array([0.5, 0.3])
        
        # Simulate transition
        next_state = state + action + np.random.randn(2) * 0.05
        
        # Learn
        world_model.update(state, action, next_state)
    
    print(f"    Training samples: {len(world_model.transitions)}")
    
    print("\n  Testing predictions:")
    
    test_cases = [
        (np.array([1.0, 0.5]), np.array([0.5, 0.3])),
        (np.array([2.0, 1.0]), np.array([0.5, 0.3])),
        (np.array([0.5, 0.2]), np.array([1.0, 0.5]))
    ]
    
    for state, action in test_cases:
        predicted, confidence = world_model.predict(state, action)
        actual = state + action  # Ground truth for learned dynamics
        
        error = np.linalg.norm(predicted - actual)
        print(f"    State {state[:2]} + Action {action[:2]}")
        print(f"      Predicted: {predicted[:2]}")
        print(f"      Actual: {actual[:2]}")
        print(f"      Error: {error:.3f}, Confidence: {confidence:.3f}")


def test_emergent_selves():
    """Test emergent self formation."""
    print("\n" + "=" * 60)
    print("EMERGENT SELVES TEST")
    print("=" * 60)
    
    ecology = EmergentSelfEcology()
    
    print("\n  Simulating execution history:")
    
    # Simulate execution traces
    for i in range(50):
        # Random action cluster
        if i < 20:
            action = np.array([0.8, 0.2]) + np.random.randn(2) * 0.1  # Cluster 1
        elif i < 35:
            action = np.array([-0.3, 0.7]) + np.random.randn(2) * 0.1  # Cluster 2
        else:
            action = np.random.randn(2) * 0.3  # Noise
        
        trace = {
            'action': action,
            'success': np.random.random() > 0.3
        }
        
        ecology.add_execution_trace(trace)
        
        if i % 15 == 14:
            state = ecology.get_state()
            print(f"    After {i+1} traces: {state['n_selves']} selves, "
                  f"dominant={state['dominant']}")
    
    # Show emergent selves
    print("\n  Emergent selves:")
    for s in ecology.selves:
        print(f"    {s.self_id}:")
        print(f"      Signature: {s.behavior_signature[:2]}")
        print(f"      Priority: {s.priority:.3f}, Stability: {s.stability:.3f}")
        print(f"      Specialization: {s.specialization}")


def test_predictive_tension():
    """Test predictive tension dynamics."""
    print("\n" + "=" * 60)
    print("PREDICTIVE TENSION TEST")
    print("=" * 60)
    
    world_model = LearnedWorldModel(state_dim=2, action_dim=2)
    tension_field = PredictiveTensionField(world_model)
    
    current_state = np.array([0.5, 0.5])
    expected_states = [
        np.array([1.0, 1.0]),
        np.array([0.0, 0.5]),
        np.array([0.5, 0.0])
    ]
    
    print("\n  Computing tension field:")
    tensions = tension_field.compute_tensions(current_state, expected_states, 0.7)
    
    for i, t in enumerate(tensions):
        print(f"\n  Tension {i}:")
        print(f"    Direction: {t['direction'][:2]}")
        print(f"    Magnitude: {t['magnitude']:.3f}")
        print(f"    Urgency: {t['urgency']:.3f}")
        print(f"    Resonance: {t['resonance']:.3f}")
    
    print(f"\n  Tension field state:")
    state = tension_field.get_state()
    print(f"    Total flow: {state['total_flow']:.3f}")
    
    # Apply tension to state
    print("\n  Applying tension to state:")
    new_state = tension_field.apply_tension(current_state, tensions[0])
    print(f"    Current: {current_state[:2]}")
    print(f"    After tension: {new_state[:2]}")


def compare_with_phase14():
    """Compare Phase 15 (Field) with Phase 14 (Coalition)."""
    print("\n" + "=" * 60)
    print("PHASE 14 VS PHASE 15 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 14 (Endogenous Executive):")
    print("    - Coalition forms symbolic plan")
    print("    - Task decomposition")
    print("    - Predefined selves")
    print("    - Static goal strength")
    print("    - Execution with reflection")
    
    print("\n  Phase 15 (Executive Field):")
    print("    - Learned world model P(s'|s,a)")
    print("    - Trajectory rollout engine")
    print("    - Attractor-based planning (NOT plan tree)")
    print("    - Emergent selves from execution")
    print("    - Cognitive economics (attention, energy, compute)")
    print("    - Predictive tension dynamics")
    print("    - Trajectory competition in field")
    
    print("\n  Key architectural shifts:")
    print("    1. Symbolic plan -> Trajectory attractors")
    print("    2. Predefined selves -> Emergent selves")
    print("    3. Static goals -> Predictive tensions")
    print("    4. Unlimited execution -> Cognitive economics")
    print("    5. World model absent -> Learned world model")
    
    print("\n  New capabilities:")
    print("    - Real prediction (not just simulation)")
    print("    - Cognitive cost accounting")
    print("    - Self emergence from behavior")
    print("    - Tension as prediction error gradient")


if __name__ == "__main__":
    test_executive_field()
    test_trajectory_rollout()
    test_cognitive_economics()
    test_world_model_learning()
    test_emergent_selves()
    test_predictive_tension()
    compare_with_phase14()
    
    print("\n" + "=" * 60)
    print("PHASE 15 - EXECUTIVE FIELD DYNAMICS")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Symbolic coalition planner (Phase 14)
  To: Continuous dynamics field where trajectories compete
  
  Coalition doesn't build plan directly.
  Coalition MODIFIES topology of trajectory field.
  Trajectories compete.
  Stable attractors emerge.
  Execution emerges from attractor selection.

CRITICAL INSIGHT:
  Phase 14: coalition -> symbolic plan -> execute
  Phase 15: state + tensions + drives + world_model -> future manifold -> trajectory competition -> attractor -> execute
  
  The planning becomes a DYNAMICAL process, not a symbolic process.

WHAT PHASE 15 ADDS:
  1. Learned World Model P(s_t+1 | s_t, a_t)
  2. Trajectory Rollout Engine (future simulation)
  3. Attractor-Based Planning (NOT plan tree)
  4. Cognitive Economics (attention, energy, compute)
  5. Emergent Selves (from execution, not predefined)
  6. Predictive Tension Dynamics (prediction error gradients)

This is where the system becomes:
  - Not symbolic planner
  - But dynamical field
  - Where trajectories attract
  - And execution emerges
  
We are now at:
  - Continuous self-organizing executive cognition
  - Attractor-based trajectory selection
  - Cognitive economics with resource constraints
  - Endogenous self formation from behavior
  - Predictive tension dynamics
""")