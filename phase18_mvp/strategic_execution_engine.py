"""
AI-OS Strategic Execution Substrate
=====================================

ARCHITECTURAL SHIFT:
  From: Task dispatcher with smart state management
  To: Strategic Execution Engine with value-aware orchestration
  
The core missing component: EXECUTION ECONOMY

What the system must answer:
  - What is most valuable right now?
  - What is the real bottleneck?
  - What blocks trajectory?
  - Where is entropy leakage?
  - Which tasks are fake-progress?
  - What should be automated?
  - Where cognitive cost > strategic value?
  - Where is user stuck in local optimum?

NOT: task graph management
BUT: strategic execution with value flow
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


# ============================================================================
# VALUE ENGINE
# ============================================================================
"""
Value Engine: Strategic Value Model

Every task must be evaluated not just by priority,
but by its position in value flow.

Task produces leverage on:
  - Strategic goals
  - System capabilities
  - Future automation
  - Entropy reduction
  
Key concepts:
  - leverage: direct impact multiplier
  - compounding: future value grows
  - unlock potential: enables new capabilities
  - entropy reduction: reduces chaos/uncertainty
  - strategic alignment: moves toward goals
"""

@dataclass
class ValueMetrics:
    """Complete value profile of a task."""
    task_id: str
    
    # Strategic value components
    leverage: float = 0.5          # Direct impact multiplier
    compounding: float = 0.0      # Future value growth rate
    unlock_potential: float = 0.0 # Enables new capabilities
    entropy_reduction: float = 0.0 # Reduces system uncertainty
    
    # Value flow
    upstream_value: float = 0.0    # Value from prerequisites
    downstream_unlocks: List[str] = field(default_factory=list)
    
    # Strategic alignment
    goal_alignment: float = 0.0    # How much this moves toward strategic goals
    trajectory_contribution: float = 0.0  # Adds to overall direction
    
    # ROI
    cognitive_cost: float = 0.5   # Mental effort required
    time_cost: float = 0.5        # Time required
    strategic_roi: float = 0.0    # Strategic value / total cost
    
    # Metadata
    automation_potential: float = 0.0  # Can this become automated?
    knowledge_creation: float = 0.0     # Does this create reusable knowledge?
    
    def compute_strategic_roi(self) -> float:
        """Compute strategic ROI."""
        total_value = (self.leverage * 0.3 + 
                      self.compounding * 0.2 + 
                      self.unlock_potential * 0.15 +
                      self.entropy_reduction * 0.15 +
                      self.goal_alignment * 0.2)
        
        total_cost = self.cognitive_cost * 0.6 + self.time_cost * 0.4
        
        self.strategic_roi = total_value / (total_cost + 1e-8)
        return self.strategic_roi


class ValueEngine:
    """
    Strategic value model for task evaluation.
    
    Evaluates tasks not just by priority, but by:
      - Position in value flow
      - Leverage on strategic goals
      - Compounding effects
      - Unlock potential
    """
    
    def __init__(self):
        self.task_values: Dict[str, ValueMetrics] = {}
        self.value_graph: Dict[str, List[Tuple[str, float]]] = {}  # task -> [(dependent, value_flow)]
        
        # Strategic goal vectors (what are we optimizing for?)
        self.strategic_vectors: Dict[str, np.ndarray] = {}
        
        # Value flow history
        self.flow_history: List[Dict] = []
        
    def evaluate_task(self, task_id: str, task_data: Dict, 
                     context: Dict) -> ValueMetrics:
        """
        Evaluate strategic value of a task.
        
        Returns complete value profile.
        """
        # Base evaluation from task properties
        estimated_hours = task_data.get('estimated_minutes', 30) / 60
        complexity = task_data.get('complexity', 'medium')
        
        # Cognitive cost estimation
        cognitive_costs = {'low': 0.3, 'medium': 0.6, 'high': 0.9}
        cognitive_cost = cognitive_costs.get(complexity, 0.5)
        
        # Compute value metrics
        value = ValueMetrics(
            task_id=task_id,
            leverage=self._compute_leverage(task_data, context),
            compounding=self._compute_compounding(task_data, context),
            unlock_potential=self._compute_unlock(task_data, context),
            entropy_reduction=self._compute_entropy_reduction(task_data, context),
            cognitive_cost=cognitive_cost,
            time_cost=estimated_hours,
            automation_potential=self._compute_automation_potential(task_data),
            knowledge_creation=self._compute_knowledge_creation(task_data)
        )
        
        # Strategic alignment
        value.goal_alignment = self._compute_goal_alignment(task_data, context)
        
        # Compute ROI
        value.compute_strategic_roi()
        
        self.task_values[task_id] = value
        return value
    
    def _compute_leverage(self, task_data: Dict, context: Dict) -> float:
        """Compute leverage on system."""
        # Tasks that unblock multiple others have high leverage
        dependent_count = context.get('dependent_count', 0)
        blocker_for = context.get('blocker_for', [])
        
        leverage = 0.5 + len(blocker_for) * 0.1 + dependent_count * 0.05
        
        # High-leverage activities
        if task_data.get('type') == 'infrastructure':
            leverage += 0.2
        if task_data.get('enables', []):
            leverage += 0.15
        
        return min(1.0, leverage)
    
    def _compute_compounding(self, task_data: Dict, context: Dict) -> float:
        """Compute compounding (future value growth)."""
        # Automation creates compounding
        if task_data.get('automatable', False):
            return 0.7
        
        # Creates reusable knowledge
        if task_data.get('creates_knowledge', False):
            return 0.6
        
        # Enables learning
        if task_data.get('learning', False):
            return 0.5
        
        # Regular tasks have low compounding
        return 0.2
    
    def _compute_unlock(self, task_data: Dict, context: Dict) -> float:
        """Compute unlock potential."""
        unlocks = task_data.get('unlocks', [])
        
        # Direct unlocks
        unlock_score = len(unlocks) * 0.2
        
        # Capabilities created
        if task_data.get('creates_capability', False):
            unlock_score += 0.3
        
        return min(1.0, unlock_score)
    
    def _compute_entropy_reduction(self, task_data: Dict, context: Dict) -> float:
        """Compute how much this reduces system entropy."""
        # Tasks that create order
        if task_data.get('type') == 'cleanup':
            return 0.7
        
        if task_data.get('type') == 'documentation':
            return 0.6
        
        if task_data.get('resolves_debt', False):
            return 0.5
        
        return 0.3
    
    def _compute_automation_potential(self, task_data: Dict) -> float:
        """Compute how automatable this task is."""
        # Repetitive tasks are automatable
        if task_data.get('repetitive', False):
            return 0.8
        
        if task_data.get('rule_based', False):
            return 0.7
        
        if task_data.get('scriptable', False):
            return 0.6
        
        return 0.2
    
    def _compute_knowledge_creation(self, task_data: Dict) -> float:
        """Compute if this creates reusable knowledge."""
        if task_data.get('creates_docs', False):
            return 0.8
        if task_data.get('creates_tests', False):
            return 0.7
        if task_data.get('creates_patterns', False):
            return 0.6
        
        return 0.2
    
    def _compute_goal_alignment(self, task_data: Dict, context: Dict) -> float:
        """Compute alignment with strategic goals."""
        goal_id = task_data.get('goal_id')
        
        # Direct goal connection
        if goal_id:
            goal_priority = context.get('goal_priority', {}).get(goal_id, 0.5)
            return goal_priority
        
        return 0.3  # Default alignment
    
    def get_value_ranking(self, task_ids: List[str]) -> List[Tuple[str, float]]:
        """Get tasks ranked by strategic ROI."""
        rankings = []
        
        for task_id in task_ids:
            if task_id in self.task_values:
                roi = self.task_values[task_id].strategic_roi
                rankings.append((task_id, roi))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def detect_fake_progress(self, task_ids: List[str]) -> List[str]:
        """Detect tasks that look productive but aren't strategic."""
        fake_progress = []
        
        for task_id in task_ids:
            value = self.task_values.get(task_id)
            if value:
                # High effort, low strategic value
                if value.cognitive_cost > 0.7 and value.strategic_roi < 0.5:
                    fake_progress.append(task_id)
                
                # No compounding, no unlock, just busy work
                if (value.compounding < 0.3 and 
                    value.unlock_potential < 0.3 and 
                    value.entropy_reduction < 0.3):
                    fake_progress.append(task_id)
        
        return fake_progress
    
    def get_summary(self) -> Dict:
        """Get value engine summary."""
        if not self.task_values:
            return {'n_tasks': 0}
        
        avg_roi = np.mean([v.strategic_roi for v in self.task_values.values()])
        high_leverage = sum(1 for v in self.task_values.values() if v.leverage > 0.7)
        high_compounding = sum(1 for v in self.task_values.values() if v.compounding > 0.5)
        
        return {
            'n_tasks': len(self.task_values),
            'avg_strategic_roi': avg_roi,
            'high_leverage_tasks': high_leverage,
            'high_compounding_tasks': high_compounding,
            'automation_candidates': sum(1 for v in self.task_values.values() if v.automation_potential > 0.5)
        }


# ============================================================================
# TRAJECTORY ENGINE
# ============================================================================
"""
Trajectory Engine: Long-term movement tracking

The system must understand:
  - Where is user/project moving?
  - Is there drift from strategic direction?
  - Is there stagnation?
  - Is trajectory collapsing?
  - Is user stuck in local optimum?

Critical insight:
  User can be "busy" (closing tasks) but trajectory score declining.
  
Trajectory tracks the actual direction of movement,
not just task completion.
"""

@dataclass
class TrajectorySnapshot:
    """Snapshot of trajectory at point in time."""
    timestamp: datetime
    position: np.ndarray  # Current state vector
    velocity: np.ndarray   # Movement direction
    acceleration: np.ndarray  # Change in velocity
    
    # Trajectory metrics
    direction_coherence: float = 0.0  # How aligned with goals
    momentum: float = 0.0  # Speed of movement
    stability: float = 0.0  # How stable is trajectory
    
    # Anomalies
    drift_detected: bool = False
    stagnation_detected: bool = False
    collapse_risk: float = 0.0
    
    # Context
    goals_achieved: List[str] = field(default_factory=list)
    strategic_distance: float = 0.0  # Distance to strategic goal


class TrajectoryEngine:
    """
    Tracks long-term trajectory of user/project.
    
    Detects:
      - Drift: moving away from strategic goals
      - Stagnation: no meaningful progress
      - Collapse: trajectory breaking down
      - False progress: busy but not advancing
    """
    
    def __init__(self, goal_vector_dim: int = 16):
        self.goal_vector_dim = goal_vector_dim
        
        # Strategic goal vector (where we want to go)
        self.strategic_goal = np.zeros(goal_vector_dim)
        
        # Trajectory history
        self.snapshots: List[TrajectorySnapshot] = []
        
        # State vectors
        self.position_history: List[np.ndarray] = []
        self.velocity_history: List[np.ndarray] = []
        
        # Trajectory anomalies
        self.drift_episodes: List[Dict] = []
        self.stagnation_episodes: List[Dict] = []
        
        # Detection thresholds
        self.drift_threshold = 0.3
        self.stagnation_threshold = 5  # Days with < threshold progress
        
    def update_strategic_goal(self, goal_vector: np.ndarray):
        """Update strategic goal direction."""
        self.strategic_goal = np.asarray(goal_vector).flatten()[:self.goal_vector_dim]
    
    def record_state(self, current_state: Dict, goals_progress: Dict, 
                    tasks_completed: List[str]) -> TrajectorySnapshot:
        """
        Record current trajectory state.
        
        Updates position, velocity, detects anomalies.
        """
        # Build position vector from current state
        position = self._build_position_vector(current_state, goals_progress)
        
        # Compute velocity (change from last position)
        velocity = np.zeros(self.goal_vector_dim)
        if self.position_history:
            last_pos = self.position_history[-1]
            velocity = position - last_pos
        
        # Compute acceleration (change in velocity)
        acceleration = np.zeros(self.goal_vector_dim)
        if len(self.velocity_history) > 1:
            last_vel = self.velocity_history[-1]
            acceleration = velocity - last_vel
        
        # Compute metrics
        direction_coherence = self._compute_direction_coherence(velocity)
        momentum = np.linalg.norm(velocity)
        stability = self._compute_stability()
        
        # Detect anomalies
        drift = self._detect_drift(velocity)
        stagnation = self._detect_stagnation()
        collapse_risk = self._compute_collapse_risk()
        
        # Create snapshot
        snapshot = TrajectorySnapshot(
            timestamp=datetime.now(),
            position=position.copy(),
            velocity=velocity.copy(),
            acceleration=acceleration.copy(),
            direction_coherence=direction_coherence,
            momentum=momentum,
            stability=stability,
            drift_detected=drift,
            stagnation_detected=stagnation,
            collapse_risk=collapse_risk,
            goals_achieved=tasks_completed,
            strategic_distance=self._compute_strategic_distance(position)
        )
        
        self.snapshots.append(snapshot)
        
        # Update histories
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-50:]
        if len(self.position_history) > 100:
            self.position_history = self.position_history[-50:]
        if len(self.velocity_history) > 100:
            self.velocity_history = self.velocity_history[-50:]
        
        return snapshot
    
    def _build_position_vector(self, state: Dict, goals_progress: Dict) -> np.ndarray:
        """Build position vector from state."""
        position = np.zeros(self.goal_vector_dim)
        
        # Goals progress
        if goals_progress:
            goal_values = list(goals_progress.values())[:self.goal_vector_dim]
            position[:len(goal_values)] = goal_values
        
        # User energy
        if 'energy' in state:
            position[0] = state['energy']
        
        # Task completion rate
        if 'task_rate' in state:
            position[1] = state['task_rate']
        
        # Focus quality
        if 'focus' in state:
            position[2] = state['focus']
        
        # Strategic alignment
        if 'alignment' in state:
            position[3] = state['alignment']
        
        return position
    
    def _compute_direction_coherence(self, velocity: np.ndarray) -> float:
        """Compute how aligned velocity is with strategic goal."""
        if np.linalg.norm(velocity) < 1e-8:
            return 0.5
        
        if np.linalg.norm(self.strategic_goal) < 1e-8:
            return 0.5
        
        # Dot product normalized
        coherence = np.dot(velocity, self.strategic_goal) / (
            np.linalg.norm(velocity) * np.linalg.norm(self.strategic_goal) + 1e-8
        )
        
        return (coherence + 1) / 2  # Normalize to 0-1
    
    def _compute_stability(self) -> float:
        """Compute trajectory stability."""
        if len(self.velocity_history) < 5:
            return 0.5
        
        recent = np.array(self.velocity_history[-5:])
        
        # Stability = inverse of velocity variance
        variances = np.var(recent, axis=0)
        avg_variance = np.mean(variances)
        
        stability = 1.0 / (1.0 + avg_variance)
        return stability
    
    def _detect_drift(self, velocity: np.ndarray) -> bool:
        """Detect if trajectory is drifting from strategic goal."""
        if np.linalg.norm(velocity) < 0.01:
            return False
        
        coherence = self._compute_direction_coherence(velocity)
        
        if coherence < self.drift_threshold:
            # Record drift episode
            self.drift_episodes.append({
                'timestamp': datetime.now(),
                'coherence': coherence,
                'velocity': velocity.tolist()
            })
            return True
        
        return False
    
    def _detect_stagnation(self) -> bool:
        """Detect if trajectory is stagnating."""
        if len(self.velocity_history) < self.stagnation_threshold:
            return False
        
        recent_velocities = self.velocity_history[-self.stagnation_threshold:]
        
        # Compute average movement
        avg_movement = np.mean([np.linalg.norm(v) for v in recent_velocities])
        
        if avg_movement < 0.05:  # Very slow progress
            self.stagnation_episodes.append({
                'timestamp': datetime.now(),
                'avg_movement': avg_movement
            })
            return True
        
        return False
    
    def _compute_collapse_risk(self) -> float:
        """Compute risk of trajectory collapse."""
        if len(self.snapshots) < 3:
            return 0.0
        
        # Factors contributing to collapse
        recent = self.snapshots[-5:]
        
        # Decreasing momentum
        momentum_trend = 0.0
        if len(recent) >= 3:
            momenta = [s.momentum for s in recent]
            if momenta[0] > momenta[-1]:
                momentum_trend = (momenta[0] - momenta[-1]) / max(1, momenta[0])
        
        # Decreasing stability
        stability_trend = 0.0
        if len(recent) >= 3:
            stabilities = [s.stability for s in recent]
            if stabilities[0] > stabilities[-1]:
                stability_trend = (stabilities[0] - stabilities[-1]) / max(1, stabilities[0])
        
        # Recent stagnation
        stagnation_factor = 1.0 if self.stagnation_episodes else 0.0
        
        # Recent drift
        drift_factor = len(self.drift_episodes[-3:]) / 3.0
        
        collapse_risk = (momentum_trend * 0.3 + 
                         stability_trend * 0.3 + 
                         stagnation_factor * 0.2 +
                         drift_factor * 0.2)
        
        return min(1.0, collapse_risk)
    
    def _compute_strategic_distance(self, position: np.ndarray) -> float:
        """Compute distance from strategic goal."""
        return np.linalg.norm(position - self.strategic_goal)
    
    def get_trajectory_summary(self) -> Dict:
        """Get trajectory summary."""
        if not self.snapshots:
            return {'status': 'no_data'}
        
        current = self.snapshots[-1]
        
        return {
            'status': 'drifting' if current.drift_detected else 
                     'stagnating' if current.stagnation_detected else
                     'healthy' if current.collapse_risk < 0.3 else
                     'at_risk',
            'direction_coherence': current.direction_coherence,
            'momentum': current.momentum,
            'stability': current.stability,
            'collapse_risk': current.collapse_risk,
            'strategic_distance': current.strategic_distance,
            'drift_episodes': len(self.drift_episodes),
            'stagnation_episodes': len(self.stagnation_episodes),
            'snapshots': len(self.snapshots)
        }
    
    def get_recommendations(self) -> List[Dict]:
        """Get trajectory-based recommendations."""
        recommendations = []
        
        if not self.snapshots:
            return recommendations
        
        current = self.snapshots[-1]
        
        if current.drift_detected:
            recommendations.append({
                'type': 'correct_drift',
                'priority': 'high',
                'message': 'Trajectory is drifting from strategic goals. Consider course correction.'
            })
        
        if current.stagnation_detected:
            recommendations.append({
                'type': 'break_stagnation',
                'priority': 'high',
                'message': 'Progress has stagnated. Consider changing approach or taking a break.'
            })
        
        if current.collapse_risk > 0.5:
            recommendations.append({
                'type': 'prevent_collapse',
                'priority': 'critical',
                'message': 'High trajectory collapse risk. Focus on maintaining momentum.'
            })
        
        if current.stability < 0.3:
            recommendations.append({
                'type': 'stabilize',
                'priority': 'medium',
                'message': 'Trajectory is unstable. Focus on consistency.'
            })
        
        return recommendations


# ============================================================================
# MOMENTUM ENGINE
# ============================================================================
"""
Momentum Engine: Execution dynamics

Momentum is more important than motivation.

Momentum includes:
  - Continuity: unbroken execution chains
  - Friction: resistance to progress
  - Reactivation cost: energy to restart after break
  - Activation energy: effort to start tasks
  - Context inertia: how hard to shift contexts
  - Recovery velocity: how fast to recover from setbacks
  
Critical insight:
  System must understand not just "what is useful"
  but "what is realistically executable NOW"
"""

@dataclass
class MomentumState:
    """Complete momentum profile."""
    continuity: float = 0.5       # Unbroken execution chains
    friction: float = 0.3         # Resistance to progress
    reactivation_cost: float = 0.5  # Energy to restart
    activation_energy: float = 0.5  # Effort to start
    context_inertia: float = 0.5  # How hard to shift context
    recovery_velocity: float = 0.5  # Speed of recovery
    
    # Derived
    effective_momentum: float = 0.0
    execution_capacity: float = 0.0
    
    def compute_derived(self):
        """Compute derived metrics."""
        # Effective momentum = continuity - friction
        self.effective_momentum = max(0, self.continuity - self.friction)
        
        # Execution capacity considers all factors
        self.execution_capacity = (
            self.effective_momentum * 0.4 +
            (1 - self.activation_energy) * 0.3 +
            self.recovery_velocity * 0.2 +
            (1 - self.context_inertia) * 0.1
        )


class MomentumEngine:
    """
    Models execution momentum dynamics.
    
    Tracks:
      - Continuity of execution
      - Friction and resistance
      - Reactivation costs
      - Context switching costs
    """
    
    def __init__(self):
        self.current_state = MomentumState()
        
        # History
        self.state_history: List[MomentumState] = []
        self.interruption_points: List[datetime] = []
        
        # Context tracking
        self.current_context: str = "default"
        self.context_durations: Dict[str, float] = {}
        
        # Time
        self.last_activity = datetime.now()
        
    def record_activity(self, task_id: str, context: str = None):
        """Record activity and update momentum."""
        if context:
            self.current_context = context
            
        # Track context duration
        if context in self.context_durations:
            self.context_durations[context] += 1
        else:
            self.context_durations[context] = 1
        
        # Continuity increases
        self.current_state.continuity = min(1.0, self.current_state.continuity + 0.1)
        
        # Friction decreases with continuity
        self.current_state.friction = max(0.1, self.current_state.friction - 0.05)
        
        # Reactivation cost decreases
        self.current_state.reactivation_cost = max(0.1, 
            self.current_state.reactivation_cost - 0.1)
        
        self.last_activity = datetime.now()
        self.current_state.compute_derived()
        
        self.state_history.append(MomentumState(
            continuity=self.current_state.continuity,
            friction=self.current_state.friction,
            reactivation_cost=self.current_state.reactivation_cost,
            activation_energy=self.current_state.activation_energy,
            context_inertia=self.current_state.context_inertia,
            recovery_velocity=self.current_state.recovery_velocity
        ))
        
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]
    
    def record_interruption(self, reason: str = "break"):
        """Record execution interruption."""
        self.interruption_points.append(datetime.now())
        
        # Interruption increases reactivation cost
        self.current_state.reactivation_cost = min(1.0, 
            self.current_state.reactivation_cost + 0.2)
        
        # Context inertia increases
        self.current_state.context_inertia = min(1.0, 
            self.current_state.context_inertia + 0.1)
        
        self.current_state.compute_derived()
    
    def record_context_switch(self, from_context: str, to_context: str):
        """Record context switch."""
        # Context switch has cost
        if self.current_context != to_context:
            self.current_state.activation_energy = min(1.0,
                self.current_state.activation_energy + 0.1)
    
    def simulate_break(self, duration_hours: float) -> MomentumState:
        """
        Simulate effect of break on momentum.
        
        Returns predicted momentum after break.
        """
        # Decay rates
        continuity_decay = min(0.5, duration_hours * 0.1)
        friction_growth = min(0.3, duration_hours * 0.05)
        reactivation_growth = min(0.4, duration_hours * 0.1)
        
        predicted = MomentumState(
            continuity=max(0.1, self.current_state.continuity - continuity_decay),
            friction=min(0.8, self.current_state.friction + friction_growth),
            reactivation_cost=min(0.9, self.current_state.reactivation_cost + reactivation_growth),
            activation_energy=self.current_state.activation_energy,
            context_inertia=self.current_state.context_inertia,
            recovery_velocity=self.current_state.recovery_velocity * max(0.5, 1 - duration_hours * 0.05)
        )
        
        predicted.compute_derived()
        return predicted
    
    def get_recommended_task_difficulty(self) -> str:
        """Recommend task difficulty based on momentum."""
        capacity = self.current_state.execution_capacity
        
        if capacity > 0.7:
            return "high"  # Complex tasks
        elif capacity > 0.4:
            return "medium"  # Standard work
        elif capacity > 0.2:
            return "low"  # Simple tasks
        else:
            return "maintenance"  # Just maintain, don't push
    
    def get_momentum_summary(self) -> Dict:
        """Get momentum summary."""
        return {
            'effective_momentum': self.current_state.effective_momentum,
            'execution_capacity': self.current_state.execution_capacity,
            'continuity': self.current_state.continuity,
            'friction': self.current_state.friction,
            'reactivation_cost': self.current_state.reactivation_cost,
            'interruption_count': len(self.interruption_points),
            'context': self.current_context
        }
    
    def predict_startup_cost(self, task_type: str) -> float:
        """Predict startup cost for task type."""
        # High friction tasks have high startup cost
        base_cost = self.current_state.reactivation_cost
        
        # Task-specific modifiers
        if task_type == 'creative':
            base_cost += 0.2  # Creative tasks hard to restart
        elif task_type == 'mechanical':
            base_cost += 0.05  # Mechanical tasks easy to restart
        elif task_type == 'collaborative':
            base_cost += 0.15  # Social tasks have coordination cost
        
        return min(1.0, base_cost)


# ============================================================================
# CONSTRAINT ENGINE (BOTTLENECK ANALYZER)
# ============================================================================
"""
Constraint Engine: Bottleneck analysis

NOT: binary blockers
BUT: multi-dimensional constraint analysis

Bottleneck types:
  - cognitive: user mental capacity
  - informational: missing information
  - emotional: user emotional state
  - infrastructural: system/tool limitations
  - environmental: external factors
  - uncertainty-based: unclear requirements
  
Goal: Identify what is really constraining trajectory
"""

class BottleneckType(Enum):
    """Types of bottlenecks."""
    COGNITIVE = "cognitive"
    INFORMATIONAL = "informational"
    EMOTIONAL = "emotional"
    INFRASTRUCTURAL = "infrastructural"
    ENVIRONMENTAL = "environmental"
    UNCERTAINTY = "uncertainty"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"


@dataclass
class Bottleneck:
    """Identified bottleneck."""
    bottleneck_id: str
    bottleneck_type: BottleneckType
    description: str
    severity: float  # 0-1, how blocking
    impact: float    # How much it limits progress
    
    affected_tasks: List[str] = field(default_factory=list)
    leverage_point: Optional[str] = None  # What would unblock this
    resolution_cost: float = 0.0  # Effort to resolve
    duration: str = "temporary"  # "temporary", "persistent", "structural"
    
    def compute_priority(self) -> float:
        """Compute resolution priority."""
        return self.severity * 0.6 + self.impact * 0.4


class ConstraintEngine:
    """
    Analyzes and identifies bottlenecks.
    
    Understands:
      - What is really constraining trajectory?
      - Where is the leverage point?
      - What can be unblocked?
    """
    
    def __init__(self):
        self.identified_bottlenecks: List[Bottleneck] = []
        self.constraint_history: List[Dict] = []
        
        self.bottleneck_counter = 0
        
    def analyze_constraints(self, task_graph: 'TaskGraph', 
                           user_state: 'UserStateModel',
                           environment_state: Dict) -> List[Bottleneck]:
        """
        Analyze system for bottlenecks.
        
        Returns identified bottlenecks with leverage points.
        """
        bottlenecks = []
        
        # 1. Cognitive bottlenecks
        energy = user_state.current_state.energy
        focus = user_state.current_state.focus_quality
        
        if energy < 0.3:
            bottlenecks.append(Bottleneck(
                bottleneck_id=f"bottleneck_{self.bottleneck_counter}",
                bottleneck_type=BottleneckType.COGNITIVE,
                severity=0.8,
                impact=0.9,
                description="User energy critically low - cannot sustain high-complexity work",
                affected_tasks=self._get_high_complexity_tasks(task_graph),
                leverage_point="reduce_cognitive_load",
                resolution_cost=0.3,
                duration="temporary"
            ))
            self.bottleneck_counter += 1
        
        if focus < 0.4:
            bottlenecks.append(Bottleneck(
                bottleneck_id=f"bottleneck_{self.bottleneck_counter}",
                bottleneck_type=BottleneckType.COGNITIVE,
                severity=0.6,
                impact=0.7,
                description="User focus quality low - high context-switching cost",
                affected_tasks=self._get_focus_dependent_tasks(task_graph),
                leverage_point="create_focus_block",
                resolution_cost=0.2,
                duration="temporary"
            ))
            self.bottleneck_counter += 1
        
        # 2. Dependency bottlenecks
        blocked_tasks = [tid for tid, t in task_graph.tasks.items() 
                        if t.state.value == "blocked"]
        
        if blocked_tasks:
            # Find the root cause blocker
            for blocked_id in blocked_tasks[:5]:
                task = task_graph.tasks.get(blocked_id)
                if task and task.dependencies:
                    for dep_id in task.dependencies:
                        dep = task_graph.tasks.get(dep_id)
                        if dep and dep.state.value != "completed":
                            bottlenecks.append(Bottleneck(
                                bottleneck_id=f"bottleneck_{self.bottleneck_counter}",
                                bottleneck_type=BottleneckType.DEPENDENCY,
                                severity=0.7,
                                impact=0.6,
                                description=f"Task '{task.title}' blocked by '{dep.title}'",
                                affected_tasks=[blocked_id],
                                leverage_point=f"unblock_{dep_id}",
                                resolution_cost=0.4,
                                duration="temporary"
                            ))
                            self.bottleneck_counter += 1
                            break
        
        # 3. Informational bottlenecks
        # Tasks with missing information
        for task_id, task in task_graph.tasks.items():
            if 'missing_info' in task.description.lower() or '?' in task.description:
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"bottleneck_{self.bottleneck_counter}",
                    bottleneck_type=BottleneckType.INFORMATIONAL,
                    severity=0.5,
                    impact=0.5,
                    description=f"Task '{task.title}' has unclear requirements",
                    affected_tasks=[task_id],
                    leverage_point="clarify_requirements",
                    resolution_cost=0.3,
                    duration="uncertainty"
                ))
                self.bottleneck_counter += 1
        
        # 4. Emotional bottlenecks (from user state)
        stress = user_state.current_state.stress
        if stress > 0.7:
            bottlenecks.append(Bottleneck(
                bottleneck_id=f"bottleneck_{self.bottleneck_counter}",
                bottleneck_type=BottleneckType.EMOTIONAL,
                severity=0.8,
                impact=0.8,
                description="High user stress - avoidance and procrastination likely",
                affected_tasks=self._get_all_active_tasks(task_graph),
                leverage_point="reduce_pressure",
                resolution_cost=0.5,
                duration="temporary"
            ))
            self.bottleneck_counter += 1
        
        # 5. Environmental bottlenecks
        if environment_state:
            # Check for external blockers
            if environment_state.get('offline_services', []):
                bottlenecks.append(Bottleneck(
                    bottleneck_id=f"bottleneck_{self.bottleneck_counter}",
                    bottleneck_type=BottleneckType.INFRASTRUCTURAL,
                    severity=0.6,
                    impact=0.5,
                    description="External services unavailable",
                    affected_tasks=environment_state.get('blocked_by_offline', []),
                    leverage_point="wait_or_workaround",
                    resolution_cost=0.1,
                    duration="temporary"
                ))
                self.bottleneck_counter += 1
        
        # Store and return
        self.identified_bottlenecks = bottlenecks
        
        return bottlenecks
    
    def _get_high_complexity_tasks(self, task_graph) -> List[str]:
        """Get high complexity tasks."""
        return [tid for tid, t in task_graph.tasks.items() 
                if t.estimated_minutes > 60]
    
    def _get_focus_dependent_tasks(self, task_graph) -> List[str]:
        """Get tasks requiring focus."""
        return [tid for tid, t in task_graph.tasks.items() 
                if t.state.value in ["pending", "ready"]]
    
    def _get_all_active_tasks(self, task_graph) -> List[str]:
        """Get all active tasks."""
        return [tid for tid, t in task_graph.tasks.items() 
                if t.state.value in ["pending", "ready", "running"]]
    
    def get_primary_constraint(self) -> Optional[Bottleneck]:
        """Get the most critical constraint."""
        if not self.identified_bottlenecks:
            return None
        
        sorted_bottlenecks = sorted(
            self.identified_bottlenecks, 
            key=lambda b: b.compute_priority(), 
            reverse=True
        )
        
        return sorted_bottlenecks[0]
    
    def get_leverage_recommendations(self) -> List[Dict]:
        """Get recommendations for applying leverage."""
        recommendations = []
        
        # Sort by priority
        sorted_bottlenecks = sorted(
            self.identified_bottlenecks,
            key=lambda b: b.compute_priority(),
            reverse=True
        )
        
        for bottleneck in sorted_bottlenecks[:3]:
            recommendations.append({
                'type': 'resolve_bottleneck',
                'bottleneck_id': bottleneck.bottleneck_id,
                'action': bottleneck.leverage_point,
                'priority': bottleneck.compute_priority(),
                'cost': bottleneck.resolution_cost,
                'impact': bottleneck.impact,
                'affected_tasks': bottleneck.affected_tasks
            })
        
        return recommendations
    
    def get_summary(self) -> Dict:
        """Get constraint analysis summary."""
        by_type = {}
        for b in self.identified_bottlenecks:
            type_key = b.bottleneck_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
        
        return {
            'total_bottlenecks': len(self.identified_bottlenecks),
            'by_type': by_type,
            'primary_constraint': self.get_primary_constraint().description if self.get_primary_constraint() else None,
            'high_severity': sum(1 for b in self.identified_bottlenecks if b.severity > 0.7)
        }


# ============================================================================
# INTEGRATED STRATEGIC EXECUTION ENGINE
# ============================================================================

class StrategicExecutionEngine:
    """
    Complete Strategic Execution Substrate.
    
    Integrates:
      - Value Engine (what has most strategic value)
      - Trajectory Engine (where are we going)
      - Momentum Engine (can we execute)
      - Constraint Engine (what is blocking us)
    
    Provides strategic orchestration of execution.
    """
    
    def __init__(self):
        # Core engines
        self.value_engine = ValueEngine()
        self.trajectory_engine = TrajectoryEngine()
        self.momentum_engine = MomentumEngine()
        self.constraint_engine = ConstraintEngine()
        
        # Time
        self.t = datetime.now()
        
    def analyze(self, tasks: Dict, user_state: 'UserStateModel',
               environment_state: Dict = None) -> Dict:
        """
        Run complete strategic analysis.
        
        Returns:
          - Value rankings
          - Trajectory status
          - Momentum status
          - Bottlenecks
          - Strategic recommendations
        """
        # 1. Value analysis
        task_values = {}
        for task_id, task in tasks.items():
            context = {
                'dependent_count': len(task.dependents),
                'blocker_for': [],  # Could be computed
            }
            value = self.value_engine.evaluate_task(task_id, {
                'estimated_minutes': task.estimated_minutes,
                'type': 'standard',
            }, context)
            task_values[task_id] = value
        
        value_rankings = self.value_engine.get_value_ranking(list(tasks.keys()))
        
        # 2. Trajectory analysis
        trajectory_summary = self.trajectory_engine.get_trajectory_summary()
        trajectory_recs = self.trajectory_engine.get_recommendations()
        
        # 3. Momentum analysis
        momentum_summary = self.momentum_engine.get_momentum_summary()
        recommended_difficulty = self.momentum_engine.get_recommended_task_difficulty()
        
        # 4. Constraint analysis
        bottlenecks = self.constraint_engine.analyze_constraints(
            self._create_mock_task_graph(tasks),
            user_state,
            environment_state or {}
        )
        leverage_recs = self.constraint_engine.get_leverage_recommendations()
        
        # 5. Combine into strategic recommendation
        strategic_action = self._compute_strategic_action(
            value_rankings, trajectory_summary, momentum_summary, 
            bottlenecks, recommended_difficulty
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            
            'value_analysis': {
                'rankings': [(tid, roi) for tid, roi in value_rankings[:5]],
                'fake_progress': self.value_engine.detect_fake_progress(list(tasks.keys())),
                'summary': self.value_engine.get_summary()
            },
            
            'trajectory': {
                'status': trajectory_summary.get('status', 'unknown'),
                'direction_coherence': trajectory_summary.get('direction_coherence', 0),
                'momentum': trajectory_summary.get('momentum', 0),
                'collapse_risk': trajectory_summary.get('collapse_risk', 0),
                'recommendations': trajectory_recs
            },
            
            'momentum': {
                'effective_momentum': momentum_summary['effective_momentum'],
                'execution_capacity': momentum_summary['execution_capacity'],
                'recommended_difficulty': recommended_difficulty
            },
            
            'constraints': {
                'bottlenecks': [
                    {
                        'type': b.bottleneck_type.value,
                        'severity': b.severity,
                        'description': b.description
                    } for b in bottlenecks[:5]
                ],
                'leverage_recommendations': leverage_recs
            },
            
            'strategic_action': strategic_action
        }
    
    def _create_mock_task_graph(self, tasks: Dict):
        """Create mock task graph for constraint analysis."""
        class MockTaskGraph:
            def __init__(self, tasks):
                self.tasks = tasks
        
        return MockTaskGraph(tasks)
    
    def _compute_strategic_action(self, value_rankings: List, 
                                 trajectory: Dict, 
                                 momentum: Dict,
                                 bottlenecks: List,
                                 recommended_difficulty: str) -> Dict:
        """Compute recommended strategic action."""
        
        # Check trajectory risk first
        if trajectory.get('collapse_risk', 0) > 0.6:
            return {
                'action': 'prevent_collapse',
                'priority': 'critical',
                'message': 'High trajectory collapse risk. Focus on maintaining momentum and reducing complexity.',
                'focus': 'stability'
            }
        
        # Check bottlenecks
        if bottlenecks and bottlenecks[0].severity > 0.7:
            return {
                'action': 'resolve_bottleneck',
                'priority': 'high',
                'message': f"Primary constraint: {bottlenecks[0].description}",
                'focus': bottlenecks[0].leverage_point
            }
        
        # Check momentum
        if momentum['execution_capacity'] < 0.3:
            return {
                'action': 'reduce_load',
                'priority': 'medium',
                'message': 'Execution capacity low. Recommend low-complexity tasks.',
                'focus': 'maintenance'
            }
        
        # Check trajectory drift
        if trajectory.get('status') == 'drifting':
            return {
                'action': 'correct_drift',
                'priority': 'high',
                'message': 'Trajectory is drifting. Recommend tasks aligned with strategic goals.',
                'focus': 'alignment'
            }
        
        # Default: high-value task
        if value_rankings:
            return {
                'action': 'execute_high_value',
                'priority': 'normal',
                'message': f"Execute high ROI task: {value_rankings[0][0]}",
                'focus': 'execution'
            }
        
        return {
            'action': 'wait',
            'priority': 'low',
            'message': 'No clear strategic action. Consider waiting.',
            'focus': 'observation'
        }
    
    def update_momentum(self, activity: str):
        """Update momentum with activity."""
        if activity == 'task_complete':
            self.momentum_engine.record_activity('task')
        elif activity == 'interruption':
            self.momentum_engine.record_interruption()
    
    def record_trajectory_state(self, state: Dict):
        """Record trajectory state."""
        self.trajectory_engine.record_state(
            state.get('position', {}),
            state.get('goals_progress', {}),
            state.get('completed', [])
        )


# ============================================================================
# TESTS
# ============================================================================

def test_value_engine():
    """Test Value Engine."""
    print("\n" + "=" * 60)
    print("VALUE ENGINE TEST")
    print("=" * 60)
    
    engine = ValueEngine()
    
    tasks = {
        'task1': {'estimated_minutes': 30, 'automatable': True},
        'task2': {'estimated_minutes': 120, 'type': 'infrastructure'},
        'task3': {'estimated_minutes': 60, 'creates_docs': True},
        'task4': {'estimated_minutes': 45, 'type': 'cleanup'},
    }
    
    print("\n  Evaluating tasks:")
    
    for task_id, task_data in tasks.items():
        value = engine.evaluate_task(task_id, task_data, {'dependent_count': 2})
        print(f"    {task_id}: ROI={value.strategic_roi:.2f}, "
              f"leverage={value.leverage:.2f}, "
              f"compounding={value.compounding:.2f}")
    
    rankings = engine.get_value_ranking(list(tasks.keys()))
    print(f"\n  Rankings: {rankings}")
    
    fake = engine.detect_fake_progress(list(tasks.keys()))
    print(f"  Fake progress: {fake}")
    
    print(f"\n  Summary: {engine.get_summary()}")


def test_trajectory_engine():
    """Test Trajectory Engine."""
    print("\n" + "=" * 60)
    print("TRAJECTORY ENGINE TEST")
    print("=" * 60)
    
    engine = TrajectoryEngine()
    
    # Set strategic goal
    engine.update_strategic_goal(np.array([1.0, 0.8, 0.6, 0.4] + [0] * 12))
    
    print("\n  Simulating trajectory:")
    
    for i in range(20):
        state = {
            'energy': 0.6 + np.random.randn() * 0.1,
            'task_rate': 0.3 + i * 0.02,
            'focus': 0.7 - i * 0.01,
            'alignment': 0.6 + np.random.randn() * 0.2
        }
        
        goals_progress = {
            'goal1': 0.2 + i * 0.04,
            'goal2': 0.3 + i * 0.03
        }
        
        snapshot = engine.record_state(state, goals_progress, [])
        
        if i % 5 == 4:
            print(f"    Step {i+1}: coherence={snapshot.direction_coherence:.2f}, "
                  f"momentum={snapshot.momentum:.3f}, "
                  f"drift={snapshot.drift_detected}")
    
    summary = engine.get_trajectory_summary()
    print(f"\n  Summary: {summary}")
    
    recs = engine.get_recommendations()
    print(f"  Recommendations: {recs}")


def test_momentum_engine():
    """Test Momentum Engine."""
    print("\n" + "=" * 60)
    print("MOMENTUM ENGINE TEST")
    print("=" * 60)
    
    engine = MomentumEngine()
    
    print("\n  Simulating activities:")
    
    for i in range(30):
        if i % 5 == 0 and i > 0:
            engine.record_interruption(f"break_{i}")
            print(f"    Interruption at step {i+1}")
        else:
            engine.record_activity(f"task_{i}", context="coding")
        
        if i % 10 == 9:
            summary = engine.get_momentum_summary()
            print(f"    Step {i+1}: capacity={summary['execution_capacity']:.2f}, "
                  f"friction={summary['friction']:.2f}, "
                  f"reactivation={summary['reactivation_cost']:.2f}")
    
    # Simulate break
    predicted = engine.simulate_break(duration_hours=8)
    print(f"\n  After 8h break:")
    print(f"    Predicted capacity: {predicted.execution_capacity:.2f}")
    
    print(f"\n  Recommended difficulty: {engine.get_recommended_task_difficulty()}")


def test_constraint_engine():
    """Test Constraint Engine."""
    print("\n" + "=" * 60)
    print("CONSTRAINT ENGINE TEST")
    print("=" * 60)
    
    engine = ConstraintEngine()
    
    # Mock task graph
    class MockTask:
        def __init__(self, state="pending", estimated_minutes=60):
            self.state = MockState(state)
            self.estimated_minutes = estimated_minutes
            self.title = "Test task"
            self.description = "Standard task description"
            self.dependents = []
            self.dependencies = []
    
    class MockState:
        def __init__(self, state):
            self.value = state
    
    class MockTaskGraph:
        def __init__(self):
            self.tasks = {
                'task1': MockTask("completed"),
                'task2': MockTask("blocked"),
                'task3': MockTask("running"),
                'task4': MockTask("pending"),
            }
    
    # Mock user state
    class MockUserState:
        def __init__(self):
            self.current_state = MockUserStateData()
    
    class MockUserStateData:
        energy = 0.25
        focus_quality = 0.3
        stress = 0.75
    
    bottlenecks = engine.analyze_constraints(
        MockTaskGraph(),
        MockUserState(),
        {}
    )
    
    print(f"\n  Identified {len(bottlenecks)} bottlenecks:")
    
    for b in bottlenecks:
        print(f"    {b.bottleneck_type.value}: {b.description[:50]}...")
        print(f"      Severity: {b.severity:.2f}, Leverage: {b.leverage_point}")
    
    primary = engine.get_primary_constraint()
    if primary:
        print(f"\n  Primary constraint: {primary.description}")
    
    recs = engine.get_leverage_recommendations()
    print(f"\n  Leverage recommendations: {len(recs)}")


def test_strategic_execution():
    """Test complete Strategic Execution Engine."""
    print("\n" + "=" * 60)
    print("STRATEGIC EXECUTION ENGINE TEST")
    print("=" * 60)
    
    engine = StrategicExecutionEngine()
    
    # Create tasks with proper state objects
    class TaskWithState:
        def __init__(self, state_str, est_minutes):
            class State:
                value = state_str
            self.state = State()
            self.estimated_minutes = est_minutes
            self.dependents = []
            self.dependencies = []
            self.title = "Task"
            self.description = "Task description"
    
    tasks = {
        f'task_{i}': TaskWithState("pending", 30 + i * 15) 
        for i in range(5)
    }
    
    # Mock user state
    class MockUserState:
        def __init__(self):
            self.current_state = MockUserStateData()
    
    class MockUserStateData:
        energy = 0.6
        focus_quality = 0.7
        stress = 0.3
        cognitive_load = 0.4
        motivation = 0.8
    
    print("\n  Running strategic analysis:")
    
    result = engine.analyze(tasks, MockUserState(), {})
    
    print(f"\n  Trajectory: {result['trajectory']['status']}")
    print(f"  Momentum: {result['momentum']['execution_capacity']:.2f}")
    print(f"  Constraints: {len(result['constraints']['bottlenecks'])}")
    print(f"  Strategic action: {result['strategic_action']['action']}")
    print(f"  Message: {result['strategic_action']['message']}")


def test_fake_progress_detection():
    """Test fake progress detection."""
    print("\n" + "=" * 60)
    print("FAKE PROGRESS DETECTION TEST")
    print("=" * 60)
    
    engine = ValueEngine()
    
    # Create tasks that look productive but aren't
    tasks = {
        'busywork1': {'estimated_minutes': 480, 'complexity': 'high'},  # 8h of busy work
        'busywork2': {'estimated_minutes': 240, 'complexity': 'high'},  # 4h of busy work
        'strategic1': {'automatable': True, 'estimated_minutes': 60},
        'strategic2': {'creates_docs': True, 'estimated_minutes': 90},
    }
    
    print("\n  Evaluating tasks:")
    
    fake_progress = []
    for task_id, task_data in tasks.items():
        value = engine.evaluate_task(task_id, task_data, {})
        print(f"    {task_id}: ROI={value.strategic_roi:.2f}")
        
        if value.cognitive_cost > 0.6 and value.strategic_roi < 0.5:
            fake_progress.append(task_id)
    
    print(f"\n  Fake progress detected: {fake_progress}")


if __name__ == "__main__":
    test_value_engine()
    test_trajectory_engine()
    test_momentum_engine()
    test_constraint_engine()
    test_strategic_execution()
    test_fake_progress_detection()
    
    print("\n" + "=" * 60)
    print("STRATEGIC EXECUTION SUBSTRATE")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Task dispatcher with state management
  To: Strategic Execution Engine with value-aware orchestration
  
CORE COMPONENTS:

1. VALUE ENGINE
   - Strategic ROI computation
   - Leverage, compounding, unlock potential
   - Fake progress detection
   - Automation candidates
   
2. TRAJECTORY ENGINE
   - Long-term direction tracking
   - Drift detection
   - Stagnation detection
   - Collapse risk prediction
   
3. MOMENTUM ENGINE
   - Execution dynamics
   - Continuity, friction, reactivation cost
   - Context switching costs
   - Break impact prediction
   
4. CONSTRAINT ENGINE
   - Multi-dimensional bottleneck analysis
   - Cognitive, informational, emotional, infrastructural
   - Leverage point identification
   - Resolution prioritization

STRATEGIC DECISION FLOW:

1. Is trajectory collapsing? → Prevent collapse
2. Is there a critical bottleneck? → Resolve bottleneck  
3. Is momentum too low? → Reduce load
4. Is trajectory drifting? → Correct drift
5. Default → Execute highest value task

This transforms AI-OS from:
  task management → strategic execution engine
  
The system now:
  - Understands what is genuinely valuable
  - Tracks where user/project is going
  - Models execution capacity
  - Identifies real constraints
  - Provides strategic recommendations
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Strategic Execution Substrate Summary

The core insight: AI-OS must be an execution economy,
not just a task manager.

Key capabilities added:

1. VALUE FLOW
   - Tasks evaluated by strategic ROI
   - Leverage, compounding, unlocks tracked
   - Fake progress detection
   
2. TRAJECTORY TRACKING
   - Long-term direction awareness
   - Drift/stagnation/collapse detection
   - Direction coherence scoring
   
3. MOMENTUM DYNAMICS
   - Execution capacity modeling
   - Continuity/friction tracking
   - Break impact prediction
   - Context switch costs
   
4. CONSTRAINT ANALYSIS
   - Multi-dimensional bottlenecks
   - Leverage point identification
   - Resolution prioritization

Strategic decision hierarchy:
  Collapse risk > Critical bottlenecks > Low momentum > Drift > High value
  
This enables AI-OS to:
  - Understand what is really valuable
  - See where trajectory is going
  - Model realistic execution capacity
  - Identify what is blocking progress
  - Provide strategic recommendations
  
Not just "what to do next" but "what is strategically optimal now".
"""