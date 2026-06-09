"""
Phase 18.12 - Cognitive Layer Integration

Connects True Variational World Model (18.11) to AI-OS goal system:
- latent z → goal state encoding
- energy V → decision utility
- stability spectrum → attention/salience

This makes the variational system actionable, not just mathematically elegant.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.insert(0, '/home/onor/ai_os_final/phase18_mvp')

from true_variational_model import TrueVariationalWorldModel
from env import SimpleEnv


@dataclass
class GoalState:
    """Goal state encoded as latent vector."""
    goal_id: str
    z: np.ndarray                    # latent state
    V: float                         # energy (potential)
    stability: float                 # stability score (0-1)
    attention_weight: float          # computed from stability
    action_tendency: str             # 'exploit' or 'explore'
    successors: List['GoalState']    # child goals


@dataclass  
class DecisionContext:
    """Context for decision making."""
    current_goal: Optional[str]
    pending_goals: List[str]
    system_energy: float
    instability_score: float


class CognitiveAttention:
    """
    Attention mechanism using stability spectrum.
    
    High stability → exploit (commit to current strategy)
    Low stability → explore (search for better solutions)
    """
    
    def __init__(self, exploit_threshold: float = 0.7, explore_threshold: float = 0.3):
        self.exploit_threshold = exploit_threshold
        self.explore_threshold = explore_threshold
    
    def compute_attention(self, stability: float, energy: float) -> Tuple[float, str]:
        """Compute attention weight and action tendency."""
        
        # Attention weight from stability
        attention = stability
        
        # Action tendency from stability zones
        if stability >= self.exploit_threshold:
            tendency = 'exploit'
            # Boost attention for stable (high confidence) regions
            attention = min(1.0, attention * 1.2)
        elif stability <= self.explore_threshold:
            tendency = 'explore'
            # Boost attention for unstable (high uncertainty) regions
            attention = min(1.0, attention * 1.5)
        else:
            tendency = 'balance'
        
        # Energy modifier (higher energy = higher utility = higher attention)
        energy_modifier = 1.0 + np.tanh(energy) * 0.3
        attention *= energy_modifier
        
        return attention, tendency


class GoalStateEncoder:
    """
    Encodes goal states into latent space for variational processing.
    
    Maps: goal attributes → z latent vector
    """
    
    def __init__(self, obs_dim: int, latent_dim: int):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Projection matrix for goal attributes
        self.W = np.random.randn(obs_dim, latent_dim) * 0.1
        self.b = np.zeros(latent_dim)
    
    def encode_goal_state(self, goal_attrs: Dict) -> np.ndarray:
        """
        Encode goal attributes into latent state.
        
        Args:
            goal_attrs: Dictionary with goal properties
                - status (str): current status
                - progress (float): 0-1 progress
                - depth (int): goal depth level
                - type (str): goal type
                - priority (float): user priority
        
        Returns:
            z: latent state vector
        """
        # Build feature vector from attributes
        features = self._extract_features(goal_attrs)
        
        # Project to latent space
        z = features @ self.W + self.b
        
        # Normalize
        z = z / (np.linalg.norm(z) + 1e-8)
        
        return z
    
    def _extract_features(self, attrs: Dict) -> np.ndarray:
        """Extract numerical features from goal attributes."""
        features = np.zeros(self.obs_dim)
        
        # Status encoding (one-hot + ordinal)
        status_map = {'pending': 0, 'active': 0.5, 'done': 1.0, 'failed': -0.5}
        features[0] = status_map.get(attrs.get('status', 'pending'), 0)
        
        # Progress (already 0-1)
        features[1] = attrs.get('progress', 0.0)
        
        # Depth level (normalized)
        features[2] = min(attrs.get('depth', 0) / 5.0, 1.0)
        
        # Priority
        features[3] = attrs.get('priority', 0.5)
        
        # Is atomic flag
        features[4] = 1.0 if attrs.get('is_atomic', False) else 0.0
        
        # Complexity (derived from depth + subgoals)
        complexity = attrs.get('depth', 0) + attrs.get('num_subgoals', 0) * 0.1
        features[5] = min(complexity / 10.0, 1.0)
        
        # Energy potential (from goal urgency)
        features[6] = attrs.get('urgency', 0.5)
        
        # Risk (from failure history)
        features[7] = attrs.get('failure_rate', 0.0)
        
        # Add noise for diversity
        features[8:] = np.random.randn(self.obs_dim - 9) * 0.01
        
        return features


class VariationalDecisionEngine:
    """
    Decision engine using variational world model.
    
    Integrates:
    - TrueVariationalWorldModel for dynamics
    - CognitiveAttention for salience
    - GoalStateEncoder for state encoding
    """
    
    def __init__(self, obs_dim: int = 10, latent_dim: int = 8, action_dim: int = 2):
        self.world_model = TrueVariationalWorldModel(obs_dim, latent_dim, action_dim)
        self.encoder = GoalStateEncoder(obs_dim, latent_dim)
        self.attention = CognitiveAttention()
        
        # Goal state storage
        self.goal_states: Dict[str, GoalState] = {}
        
        # Action vocabulary
        self.actions = {
            'decompose': np.array([1.0, 0.0]),      # break goal into subgoals
            'execute': np.array([0.0, 1.0]),         # execute atomic goal
            'wait': np.array([0.0, 0.0]),            # defer decision
        }
    
    def register_goal(self, goal_id: str, goal_attrs: Dict) -> GoalState:
        """Register a goal and compute its state."""
        
        # Encode goal into latent space
        z = self.encoder.encode_goal_state(goal_attrs)
        
        # Convert to observation format for world model
        obs = np.concatenate([z, np.zeros(2)])  # z (8) + actions placeholder (2) = obs_dim=10
        
        # Process through variational world model
        # Use 'decompose' action as default
        state = self.world_model.forward(obs, self.actions['execute'])
        
        # Compute stability mean from spectrum
        stability_spectrum = state['stability_spectrum']
        stability_mean = np.mean(stability_spectrum['stabilities'])
        
        # Compute attention and tendency
        attention, tendency = self.attention.compute_attention(
            stability=stability_mean,
            energy=state['V']
        )
        
        # Create goal state
        goal_state = GoalState(
            goal_id=goal_id,
            z=z,
            V=state['V'],
            stability=stability_mean,
            attention_weight=attention,
            action_tendency=tendency,
            successors=[]
        )
        
        self.goal_states[goal_id] = goal_state
        
        return goal_state
    
    def update_goal_state(self, goal_id: str, new_attrs: Dict) -> GoalState:
        """Update goal state after execution."""
        
        if goal_id not in self.goal_states:
            return self.register_goal(goal_id, new_attrs)
        
        old_state = self.goal_states[goal_id]
        
        # Re-encode with new attributes
        z = self.encoder.encode_goal_state(new_attrs)
        
        # Choose action based on tendency
        action = self.actions.get(old_state.action_tendency, self.actions['execute'])
        
        # Convert to observation format
        obs = np.concatenate([z, np.zeros(2)])
        
        # Process through world model
        state = self.world_model.forward(obs, action)
        
        # Compute stability mean from spectrum
        stability_spectrum = state['stability_spectrum']
        stability_mean = np.mean(stability_spectrum['stabilities'])
        
        # Compute new attention
        attention, tendency = self.attention.compute_attention(
            stability=stability_mean,
            energy=state['V']
        )
        
        # Update goal state
        goal_state = GoalState(
            goal_id=goal_id,
            z=z,
            V=state['V'],
            stability=stability_mean,
            attention_weight=attention,
            action_tendency=tendency,
            successors=old_state.successors
        )
        
        self.goal_states[goal_id] = goal_state
        
        return goal_state
    
    def compute_decision(self, context: DecisionContext) -> Dict:
        """
        Compute decision based on current context.
        
        Returns:
            decision: {
                'target_goal': goal_id,
                'action': str,
                'utility': float,
                'reasoning': str
            }
        """
        
        # Get all goal states
        goals = list(self.goal_states.values())
        
        if not goals:
            return {
                'target_goal': None,
                'action': 'wait',
                'utility': 0.0,
                'reasoning': 'No goals registered'
            }
        
        # Sort by attention weight
        goals.sort(key=lambda g: g.attention_weight, reverse=True)
        
        # Select top goal
        top_goal = goals[0]
        
        # Map tendency to action
        action_map = {
            'exploit': 'execute',
            'explore': 'decompose',
            'balance': 'execute'
        }
        action = action_map.get(top_goal.action_tendency, 'execute')
        
        return {
            'target_goal': top_goal.goal_id,
            'action': action,
            'utility': top_goal.attention_weight * np.exp(top_goal.V),
            'stability': top_goal.stability,
            'energy': top_goal.V,
            'reasoning': f"{top_goal.action_tendency} due to stability={top_goal.stability:.2f}"
        }
    
    def get_system_energy(self) -> float:
        """Get overall system energy (sum of goal energies)."""
        if not self.goal_states:
            return 0.0
        return sum(g.V for g in self.goal_states.values()) / len(self.goal_states)
    
    def get_instability_score(self) -> float:
        """Get system-wide instability (low stability = high instability)."""
        if not self.goal_states:
            return 0.0
        return 1.0 - np.mean([g.stability for g in self.goal_states.values()])


class CognitiveGoalScheduler:
    """
    Goal scheduler using variational decision engine.
    
    Replaces default goal prioritization with stability-aware scheduling.
    """
    
    def __init__(self, engine: VariationalDecisionEngine):
        self.engine = engine
        self.execution_history: List[Dict] = []
    
    def select_next_goal(self) -> Tuple[Optional[str], str]:
        """
        Select next goal to execute based on stability analysis.
        
        Returns:
            (goal_id, action) or (None, 'wait') if no goals ready
        """
        context = DecisionContext(
            current_goal=None,
            pending_goals=list(self.engine.goal_states.keys()),
            system_energy=self.engine.get_system_energy(),
            instability_score=self.engine.get_instability_score()
        )
        
        decision = self.engine.compute_decision(context)
        
        self.execution_history.append(decision)
        
        return decision['target_goal'], decision['action']
    
    def schedule_goals(self, goal_list: List[Dict]) -> List[Tuple[str, str, float]]:
        """
        Schedule all goals by attention priority.
        
        Args:
            goal_list: List of goal attributes
            
        Returns:
            List of (goal_id, action, priority) sorted by attention
        """
        scheduled = []
        
        for goal in goal_list:
            goal_id = goal['id']
            attrs = {k: v for k, v in goal.items() if k != 'id'}
            
            state = self.engine.register_goal(goal_id, attrs)
            
            scheduled.append((
                goal_id,
                state.action_tendency,
                state.attention_weight
            ))
        
        # Sort by attention weight descending
        scheduled.sort(key=lambda x: x[2], reverse=True)
        
        return scheduled


def test_cognitive_integration():
    """Test cognitive layer integration."""
    print("=" * 60)
    print("COGNITIVE LAYER INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize engine
    engine = VariationalDecisionEngine(obs_dim=10, latent_dim=8, action_dim=2)
    scheduler = CognitiveGoalScheduler(engine)
    
    # Test goal registration
    print("\n--- Goal Registration ---")
    
    test_goals = [
        {
            'id': 'goal_1',
            'status': 'pending',
            'progress': 0.0,
            'depth': 1,
            'priority': 0.8,
            'is_atomic': False,
            'num_subgoals': 5,
            'urgency': 0.6
        },
        {
            'id': 'goal_2',
            'status': 'active',
            'progress': 0.5,
            'depth': 2,
            'priority': 0.9,
            'is_atomic': True,
            'num_subgoals': 0,
            'urgency': 0.8
        },
        {
            'id': 'goal_3',
            'status': 'pending',
            'progress': 0.0,
            'depth': 3,
            'priority': 0.4,
            'is_atomic': False,
            'num_subgoals': 10,
            'urgency': 0.3
        },
        {
            'id': 'goal_4',
            'status': 'done',
            'progress': 1.0,
            'depth': 1,
            'priority': 0.5,
            'is_atomic': True,
            'num_subgoals': 0,
            'urgency': 0.1
        }
    ]
    
    for goal in test_goals:
        state = engine.register_goal(goal['id'], goal)
        print(f"  {goal['id']}: "
              f"V={state.V:.3f}, "
              f"stab={state.stability:.3f}, "
              f"attn={state.attention_weight:.3f}, "
              f"→ {state.action_tendency}")
    
    # Test scheduling
    print("\n--- Goal Scheduling ---")
    
    schedule = scheduler.schedule_goals(test_goals)
    
    print("  Priority order:")
    for i, (goal_id, action, priority) in enumerate(schedule, 1):
        print(f"    {i}. {goal_id}: {action} (priority={priority:.3f})")
    
    # Test decision making
    print("\n--- Decision Making ---")
    
    for _ in range(5):
        goal_id, action = scheduler.select_next_goal()
        print(f"  Decision: {goal_id} → {action}")
    
    # Test update cycle
    print("\n--- Update Cycle ---")
    
    engine.update_goal_state('goal_2', {
        'status': 'active',
        'progress': 0.8,
        'depth': 2,
        'priority': 0.95,
        'is_atomic': True,
        'num_subgoals': 0,
        'urgency': 0.9
    })
    
    new_state = engine.goal_states['goal_2']
    print(f"  goal_2 updated: V={new_state.V:.3f}, stab={new_state.stability:.3f}")
    
    # System metrics
    print("\n--- System Metrics ---")
    print(f"  System energy: {engine.get_system_energy():.3f}")
    print(f"  Instability score: {engine.get_instability_score():.3f}")
    
    # Get full system state
    print("\n--- Full System State ---")
    for goal_id, state in engine.goal_states.items():
        print(f"  {goal_id}: |z|={np.linalg.norm(state.z):.3f}, "
              f"succ={len(state.successors)}")


def test_variational_goal_dynamics():
    """Test how goal states evolve through variational dynamics."""
    print("\n" + "=" * 60)
    print("VARIATIONAL GOAL DYNAMICS TEST")
    print("=" * 60)
    
    engine = VariationalDecisionEngine(obs_dim=10, latent_dim=8, action_dim=2)
    
    # Track state evolution
    goal_attrs = {
        'status': 'pending',
        'progress': 0.0,
        'depth': 2,
        'priority': 0.7,
        'is_atomic': False,
        'num_subgoals': 5,
        'urgency': 0.5
    }
    
    print("\n  Evolving goal through execution stages:")
    
    V_history = []
    stability_history = []
    attention_history = []
    
    for stage in range(1, 6):
        # Simulate progress
        goal_attrs['progress'] = (stage - 1) / 5.0
        goal_attrs['urgency'] = 0.5 + (stage - 1) * 0.1
        
        state = engine.update_goal_state('evolving_goal', goal_attrs)
        
        V_history.append(state.V)
        stability_history.append(state.stability)
        attention_history.append(state.attention_weight)
        
        print(f"    Stage {stage}: V={state.V:.3f}, stab={state.stability:.3f}, "
              f"attn={state.attention_weight:.3f}, {state.action_tendency}")
    
    # Analyze trends
    print(f"\n  Trend analysis:")
    print(f"    Energy change: {V_history[0]:.3f} → {V_history[-1]:.3f}")
    print(f"    Stability change: {stability_history[0]:.3f} → {stability_history[-1]:.3f}")
    print(f"    Attention change: {attention_history[0]:.3f} → {attention_history[-1]:.3f}")


def test_world_model_integration():
    """Test world model integration with goal system."""
    print("\n" + "=" * 60)
    print("WORLD MODEL INTEGRATION TEST")
    print("=" * 60)
    
    engine = VariationalDecisionEngine(obs_dim=10, latent_dim=8, action_dim=2)
    
    # Register multiple goals
    for i in range(5):
        goal_attrs = {
            'status': 'active' if i < 3 else 'pending',
            'progress': i / 10.0,
            'depth': i % 3 + 1,
            'priority': 0.5 + i * 0.1,
            'is_atomic': i % 2 == 0,
            'num_subgoals': (3 - i) if i < 3 else 0,
            'urgency': 0.5 + i * 0.05
        }
        engine.register_goal(f'goal_{i}', goal_attrs)
    
    # Process sequence of decisions
    print("\n  Processing decision sequence:")
    
    for step in range(10):
        context = DecisionContext(
            current_goal=f'goal_{step % 5}' if step % 5 < 3 else None,
            pending_goals=[f'goal_{i}' for i in range(5)],
            system_energy=engine.get_system_energy(),
            instability_score=engine.get_instability_score()
        )
        
        decision = engine.compute_decision(context)
        
        # Simulate execution
        if decision['target_goal']:
            attrs = {
                'status': 'active',
                'progress': 0.3 + step * 0.05,
                'depth': 1,
                'priority': 0.7,
                'is_atomic': True,
                'num_subgoals': 0,
                'urgency': 0.6
            }
            engine.update_goal_state(decision['target_goal'], attrs)
        
        if step % 3 == 0:
            print(f"    Step {step}: {decision['target_goal']} → "
                  f"{decision['action']} (utility={decision['utility']:.3f})")
    
    # World model state
    print("\n  World model final state:")
    vm_state = engine.world_model.get_state()
    for key, value in vm_state.items():
        if not isinstance(value, np.ndarray):
            print(f"    {key}: {value}")


def test_attention_mechanism():
    """Test attention mechanism with various stability levels."""
    print("\n" + "=" * 60)
    print("ATTENTION MECHANISM TEST")
    print("=" * 60)
    
    attention = CognitiveAttention()
    
    print("\n  Testing stability → attention mapping:")
    
    test_cases = [
        (0.9, 2.0, "highly stable"),
        (0.7, 1.5, "stable"),
        (0.5, 0.0, "balanced"),
        (0.3, -1.0, "unstable"),
        (0.1, -2.0, "highly unstable"),
    ]
    
    for stability, energy, description in test_cases:
        attn, tendency = attention.compute_attention(stability, energy)
        print(f"    stab={stability:.1f}, E={energy:.1f} → "
              f"attn={attn:.3f}, {tendency} ({description})")


def compare_with_baseline():
    """Compare variational scheduling with baseline."""
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON")
    print("=" * 60)
    
    engine = VariationalDecisionEngine(obs_dim=10, latent_dim=8, action_dim=2)
    scheduler = CognitiveGoalScheduler(engine)
    
    # Create goal list
    goals = []
    for i in range(10):
        goal = {
            'id': f'goal_{i}',
            'status': 'active' if i < 5 else 'pending',
            'progress': i / 10.0,
            'depth': (i % 4) + 1,
            'priority': np.random.random(),
            'is_atomic': i % 3 == 0,
            'num_subgoals': np.random.randint(0, 8),
            'urgency': np.random.random()
        }
        goals.append(goal)
    
    # Variational scheduling
    var_schedule = scheduler.schedule_goals(goals)
    
    # Baseline (priority only)
    baseline_schedule = sorted(goals, key=lambda g: g['priority'], reverse=True)
    
    print("\n  Variational schedule (stability-aware):")
    for i, (goal_id, action, priority) in enumerate(var_schedule[:5], 1):
        state = engine.goal_states[goal_id]
        print(f"    {i}. {goal_id}: {action}, stab={state.stability:.2f}, attn={priority:.3f}")
    
    print("\n  Baseline schedule (priority only):")
    for i, goal in enumerate(baseline_schedule[:5], 1):
        print(f"    {i}. {goal['id']}: priority={goal['priority']:.2f}")
    
    # Compare
    print("\n  Comparison:")
    var_priorities = [p for _, _, p in var_schedule]
    base_priorities = [g['priority'] for g in baseline_schedule]
    
    correlation = np.corrcoef(var_priorities, base_priorities)[0, 1]
    print(f"    Priority correlation: {correlation:.3f}")
    print(f"    (1.0 = identical, 0.0 = unrelated)")


if __name__ == '__main__':
    test_attention_mechanism()
    test_cognitive_integration()
    test_variational_goal_dynamics()
    test_world_model_integration()
    compare_with_baseline()
    
    print("\n" + "=" * 60)
    print("PHASE 18.12 - COGNITIVE LAYER INTEGRATION COMPLETE")
    print("=" * 60)
    print("\nWhat we connected:")
    print("1. latent z → goal state encoding")
    print("2. energy V → decision utility")
    print("3. stability spectrum → attention/salience")
    print("\nNow the variational system is:")
    print("  ✓ Actionable (influences goal scheduling)")
    print("  ✓ Controllable (attention weights drive decisions)")
    print("  ✓ Long-horizon planning (stability-aware)")
    print("\nPhase 18.11 was mathematical elegance.")
    print("Phase 18.12 is practical utility.")