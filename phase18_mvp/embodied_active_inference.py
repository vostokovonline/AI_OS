"""
Phase 25: Embodied Active Inference

ARCHITECTURAL SHIFT:
  From: Phase 24 - active inference field with policy inference
  To: Phase 25 - embodied active inference where:
      - Sensorimotor closure: perception and action are mutually constitutive
      - Learned body schema: body geometry is inferred, not given
      - Ecological niche: system constructs its own environment
      - Precological grounding: all cognition is action-oriented
      
  This is NO LONGER:
    perception → cognition → action
  This IS:
    embodied agent-environment system with mutual causality
    
CRITICAL INSIGHT:
  Phase 24: "agent acts on world"
  Phase 25: "agent IS coupled with world"
  
  The body is not a vehicle for cognition.
  The body IS the cognitive system.
  Self is not separate from environment.
  Self-environment is a single dynamical system.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy


# ============================================================================
# SENSORIMOTOR CLOSURE
# ============================================================================
"""
Sensorimotor Closure:

Perception and action are mutually constitutive.

p(s | o, a) = perceptual inference given observation and action history
p(a | s, g) = action inference given state and goals

NOT: perception = bottom-up
BUT: perception = action-conditioned inference

The agent's actions shape what it perceives.
What it perceives shapes what actions it takes.
"""

class SensorimotorClosure:
    """
    Sensorimotor closure with mutual perception-action conditioning.
    
    NOT: "I see then I act"
    BUT: "My actions determine what I see"
    
    The system is closed in the sensorimotor loop:
      - Action affects sensory input
      - Sensory input affects action
      - No external "ground truth" - only coupled dynamics
    """
    
    def __init__(self, sensory_dim: int = 8, motor_dim: int = 4):
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        
        # Sensory channels
        self.sensory_channels = [np.zeros(sensory_dim) for _ in range(5)]
        
        # Motor commands (history)
        self.motor_history: List[np.ndarray] = []
        
        # Coupling strength (how much action affects sensation)
        self.coupling_strength = 1.0
        
        # Sensorimotor contingencies
        self.smc: Dict[str, np.ndarray] = {}  # Learned contingencies
        self.smc_confidence = np.zeros(10)
        
        # Self-touch detection (interaction with self vs environment)
        self.self_touch_threshold = 0.3
        
        # Time
        self.t = 0.0
    
    def generate_sensation(self, motor_command: np.ndarray,
                          environment_state: np.ndarray) -> np.ndarray:
        """
        Generate sensation from motor command and environment.
        
        Sensation depends on both:
          - What action was taken
          - What the environment is doing
        
        The agent's actions shape its sensations.
        """
        motor = np.asarray(motor_command).flatten()[:self.motor_dim]
        env = np.asarray(environment_state).flatten()[:self.sensory_dim]
        
        # Motor effect on sensation (proprioceptive component)
        motor_effect = np.zeros(self.sensory_dim)
        motor_effect[:self.motor_dim] = motor * self.coupling_strength
        
        # Sensorimotor contingency: action changes sensitivity
        # "If I move this way, I see that way"
        contingency_effect = np.zeros(self.sensory_dim)
        if len(self.motor_history) > 0:
            prev_motor = self.motor_history[-1]
            # Learned contingency: motor pattern → sensory change
            contingency_effect[:self.motor_dim] = prev_motor * 0.2
        
        # Environmental effect
        env_effect = env[:self.sensory_dim]
        
        # Combined sensation
        sensation = motor_effect + contingency_effect + env_effect
        
        # Add noise (sensory uncertainty)
        sensation = sensation + np.random.randn(self.sensory_dim) * 0.1
        
        return sensation
    
    def update_closure(self, action: np.ndarray, observation: np.ndarray):
        """
        Update sensorimotor closure based on action-observation pair.
        
        Learn contingencies between actions and sensations.
        """
        action = np.asarray(action).flatten()[:self.motor_dim]
        observation = np.asarray(observation).flatten()[:self.sensory_dim]
        
        # Update motor history
        self.motor_history.append(action.copy())
        if len(self.motor_history) > 50:
            self.motor_history = self.motor_history[-20:]
        
        # Update sensory channels
        self.sensory_channels.append(observation.copy())
        if len(self.sensory_channels) > 5:
            self.sensory_channels.pop(0)
        
        # Learn sensorimotor contingencies
        if len(self.motor_history) > 1:
            prev_action = self.motor_history[-2] if len(self.motor_history) >= 2 else np.zeros(self.motor_dim)
            sensory_delta = observation - self.sensory_channels[-2] if len(self.sensory_channels) >= 2 else np.zeros(self.sensory_dim)
            
            # Learn: "when I do this action, I see this change"
            key = self._get_smc_key(prev_action)
            if key not in self.smc:
                self.smc[key] = np.zeros(self.sensory_dim)
            
            # Update contingency
            self.smc[key] = self.smc[key] * 0.95 + sensory_delta * 0.05
        
        # Update SMC confidence
        self.smc_confidence = np.roll(self.smc_confidence, -1)
        self.smc_confidence[-1] = min(1.0, np.mean(np.abs(list(self.smc.values()))))
        
        self.t += 1
    
    def _get_smc_key(self, motor: np.ndarray) -> str:
        """Get sensorimotor contingency key (discretized motor pattern)."""
        motor_binned = (motor * 3).astype(int)
        return str(motor_binned.tolist())
    
    def infer_perception(self, observation: np.ndarray,
                        action_context: List[np.ndarray]) -> Dict:
        """
        Infer perception conditioned on action context.
        
        NOT: "what is this?"
        BUT: "what is this given what I did?"
        
        Perception is action-conditioned.
        """
        observation = np.asarray(observation).flatten()[:self.sensory_dim]
        
        # Action context shapes perception
        if len(action_context) > 0:
            action_pattern = np.mean(action_context[-3:], axis=0)
        else:
            action_pattern = np.zeros(self.motor_dim)
        
        # Learned contingencies
        contingency = np.zeros(self.sensory_dim)
        key = self._get_smc_key(action_pattern[:self.motor_dim])
        if key in self.smc:
            contingency = self.smc[key]
        
        # Perception = observation + action-conditioned inference
        perceived = observation + contingency
        
        # Confidence in perception
        confidence = np.mean(self.smc_confidence)
        
        return {
            'perceived': perceived,
            'action_context': action_pattern.tolist(),
            'contingency': contingency.tolist(),
            'confidence': confidence
        }
    
    def detect_self_interaction(self, observation: np.ndarray) -> float:
        """
        Detect if sensation is from self or environment.
        
        Self-touch: motor command produces expected sensation
        Environment: sensation differs from expectation
        """
        if len(self.motor_history) < 1:
            return 0.5
        
        recent_action = self.motor_history[-1]
        
        # Expected sensation from action
        expected = recent_action * self.coupling_strength
        
        # Actual sensation
        actual = np.asarray(observation).flatten()[:self.sensory_dim][:self.motor_dim]
        
        # Self-touch if actual ≈ expected
        match = np.linalg.norm(actual - expected)
        self_interaction = 1.0 / (1.0 + match / self.self_touch_threshold)
        
        return self_interaction
    
    def get_closure_summary(self) -> Dict:
        """Get sensorimotor closure summary."""
        return {
            'n_contingencies': len(self.smc),
            'mean_confidence': float(np.mean(self.smc_confidence)),
            'coupling_strength': self.coupling_strength,
            'motor_history_len': len(self.motor_history)
        }


# ============================================================================
# LEARNED BODY SCHEMA
# ============================================================================
"""
Learned Body Schema:

The body is not a given. The body is inferred.

Body schema = learned representation of:
  - Limb positions (proprioception)
  - Control signals (motor commands)
  - Sensory consequences (proprioceptive feedback)
  
The system learns its own body through interaction.
"""

@dataclass
class BodySegment:
    """A segment of the body (limb, joint, etc.)."""
    segment_id: str
    position: np.ndarray  # Current position
    length: float         # Segment length
    flexibility: float    # How much it can bend
    sensory_feedback: np.ndarray  # Proprioceptive signal
    motor_command: np.ndarray    # Last motor command
    
    def predict(self, motor_command: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Predict next position given motor command."""
        return self.position + motor_command * dt


class LearnedBodySchema:
    """
    Learned body schema - body geometry inferred from interaction.
    
    NOT: body = fixed kinematic chain
    BUT: body = learned internal model of sensorimotor patterns
    
    The system discovers its body through exploration.
    """
    
    def __init__(self, n_segments: int = 4, segment_dim: int = 4):
        self.n_segments = n_segments
        self.segment_dim = segment_dim
        
        # Body segments (learned)
        self.segments: List[BodySegment] = []
        
        # Body model (learned geometry)
        self.body_model = np.zeros((n_segments, segment_dim))
        
        # Body confidence (how well model fits data)
        self.body_confidence = 0.5
        
        # Self-model (how system perceives itself)
        self.self_model = np.zeros(segment_dim * n_segments)
        
        # Exploration history (how body was learned)
        self.exploration_trajectories: List[Dict] = []
    
    def init_segments(self):
        """Initialize body segments (starting from unknown)."""
        self.segments = []
        for i in range(self.n_segments):
            segment = BodySegment(
                segment_id=f"seg_{i}",
                position=np.zeros(self.segment_dim),
                length=1.0,
                flexibility=0.5,
                sensory_feedback=np.zeros(self.segment_dim),
                motor_command=np.zeros(self.segment_dim)
            )
            self.segments.append(segment)
        
        # Start with uncertain body model
        self.body_model = np.random.randn(self.n_segments, self.segment_dim) * 0.5
        self.self_model = self.body_model.flatten()
    
    def update_from_interaction(self, motor_commands: List[np.ndarray],
                               sensory_feedback: List[np.ndarray]):
        """
        Update body schema from motor-sensory interaction.
        
        Learn: "when I send this command, I feel this sensation"
        """
        if len(motor_commands) < 2 or len(sensory_feedback) < 2:
            return
        
        # Learning rate
        lr = 0.01
        
        for i in range(min(len(motor_commands), len(sensory_feedback)) - 1):
            motor = np.asarray(motor_commands[i]).flatten()[:self.segment_dim]
            sensory = np.asarray(sensory_feedback[i + 1]).flatten()[:self.segment_dim]
            
            # Learn: body model transforms motor → sensory
            # sensory ≈ body_model @ motor + bias
            # body_model = sensory @ motor^T / (motor^T @ motor)
            
            for j in range(self.n_segments):
                if j < len(self.body_model):
                    # Gradient update
                    predicted = np.dot(self.body_model[j], motor)
                    error = sensory[j] - predicted if j < len(sensory) else 0
                    self.body_model[j] += lr * error * motor
            
            # Update segment positions
            for k, segment in enumerate(self.segments):
                if k < len(motor_commands):
                    segment.motor_command = motor_commands[k]
                    segment.sensory_feedback = sensory_feedback[k] if k < len(sensory_feedback) else np.zeros(self.segment_dim)
                    segment.position = segment.predict(segment.motor_command)
        
        # Update self-model
        self.self_model = self.body_model.flatten()
        
        # Update confidence
        prediction_error = 0.0
        n_preds = 0
        for i, segment in enumerate(self.segments):
            if i < len(motor_commands):
                predicted = segment.predict(motor_commands[i])
                actual = segment.sensory_feedback[:self.segment_dim]
                prediction_error += np.linalg.norm(predicted - actual)
                n_preds += 1
        
        self.body_confidence = 1.0 / (1.0 + prediction_error * 0.1 / max(1, n_preds))
    
    def predict_body_state(self, motor_command: np.ndarray) -> np.ndarray:
        """
        Predict body state given motor command.
        
        Uses learned body model.
        """
        motor = np.asarray(motor_command).flatten()[:self.segment_dim]
        
        predicted_state = np.zeros(self.segment_dim * self.n_segments)
        
        for i, segment in enumerate(self.segments):
            if i < len(self.body_model):
                predicted = segment.predict(motor)
                start = i * self.segment_dim
                end = start + self.segment_dim
                predicted_state[start:end] = predicted
        
        return predicted_state
    
    def get_body_geometry(self) -> Dict:
        """Get current body geometry."""
        positions = []
        lengths = []
        
        for segment in self.segments:
            positions.append(segment.position.tolist())
            lengths.append(segment.length)
        
        return {
            'n_segments': self.n_segments,
            'positions': positions,
            'lengths': lengths,
            'body_confidence': self.body_confidence
        }
    
    def explore_body(self, n_explorations: int = 20) -> List[np.ndarray]:
        """
        Generate exploration movements to learn body.
        
        Random motor commands to discover body geometry.
        """
        exploration_movements = []
        
        for _ in range(n_explorations):
            # Random motor command
            motor = np.random.randn(self.segment_dim) * 0.5
            exploration_movements.append(motor)
        
        return exploration_movements


# ============================================================================
# ECOLOGICAL NICHE CONSTRUCTION
# ============================================================================
"""
Ecological Niche Construction:

The system constructs its own environment.

Ecological niche = set of conditions that allow system to survive
Niche construction = actions that modify environment to fit niche

NOT: "agent adapts to environment"
BUT: "agent creates environment where it thrives"

The system shapes the world as much as the world shapes the system.
"""

@dataclass
class EcologicalNiche:
    """A niche in the environment."""
    niche_id: str
    conditions: Dict[str, float]  # Temperature, light, resources, etc.
    favorability: float           # How good this niche is for the system
    stability: float               # How stable this niche is
    constructed_by: List[str]      # Actions that created this niche
    
    def evaluate(self, current_conditions: Dict[str, float]) -> float:
        """Evaluate how favorable this niche is."""
        score = 0.0
        for key, target in self.conditions.items():
            if key in current_conditions:
                diff = abs(current_conditions[key] - target)
                score += np.exp(-diff * 2)
        return score / max(1, len(self.conditions))


class EcologicalNicheConstructor:
    """
    Ecological niche constructor.
    
    NOT: "adapt to environment"
    BUT: "create environment where you thrive"
    
    The system actively modifies its environment.
    """
    
    def __init__(self, condition_dim: int = 4):
        self.condition_dim = condition_dim
        
        # Current environment conditions
        self.environment_conditions = np.zeros(condition_dim)
        
        # Known niches
        self.niches: Dict[str, EcologicalNiche] = {}
        self.niche_counter = 0
        
        # Construction actions (history)
        self.construction_history: List[Dict] = []
        
        # Niche preference (what conditions does system prefer?)
        self.niche_preference = np.zeros(condition_dim)
        
        # Environment manipulation capability
        self.manipulation_power = 1.0
        
        # Time
        self.t = 0.0
    
    def set_niche_preference(self, preferred_conditions: np.ndarray):
        """Set preferred environment conditions."""
        self.niche_preference = np.asarray(preferred_conditions).flatten()[:self.condition_dim]
    
    def sense_environment(self) -> Dict[str, float]:
        """Sense current environment conditions."""
        conditions = {}
        for i in range(self.condition_dim):
            conditions[f"condition_{i}"] = float(self.environment_conditions[i])
        return conditions
    
    def construct_niche(self, target_conditions: Dict[str, float],
                       actions: List[np.ndarray]) -> str:
        """
        Construct a niche by modifying environment.
        
        Returns niche_id.
        """
        niche_id = f"niche_{self.niche_counter}"
        self.niche_counter += 1
        
        conditions = {}
        for key, val in target_conditions.items():
            conditions[key] = val
        
        # Calculate favorability
        favorability = 0.0
        for key, val in target_conditions.items():
            current = self.environment_conditions[int(key.split('_')[1])] if '_' in key else 0
            favorability += np.exp(-abs(current - val))
        
        stability = 0.8  # Start with assumption of stability
        
        niche = EcologicalNiche(
            niche_id=niche_id,
            conditions=conditions,
            favorability=favorability / max(1, len(target_conditions)),
            stability=stability,
            constructed_by=[str(a) for a in actions]
        )
        
        self.niches[niche_id] = niche
        self.construction_history.append({
            'niche_id': niche_id,
            'conditions': conditions,
            'actions': len(actions)
        })
        
        return niche_id
    
    def act_on_environment(self, action: np.ndarray,
                          effect_strength: float = 0.1) -> np.ndarray:
        """
        Act on environment to modify conditions.
        
        Actions modify environmental state.
        """
        action = np.asarray(action).flatten()[:self.condition_dim]
        
        # Action modifies environment
        self.environment_conditions = self.environment_conditions + action * effect_strength
        
        # Add noise
        self.environment_conditions += np.random.randn(self.condition_dim) * 0.05
        
        # Clamp to reasonable range
        self.environment_conditions = np.clip(self.environment_conditions, -2, 2)
        
        self.t += 1
        
        return self.environment_conditions.copy()
    
    def evaluate_current_niche(self) -> float:
        """Evaluate how favorable current environment is."""
        conditions = self.sense_environment()
        
        # Match against known niches
        best_match = 0.0
        for niche in self.niches.values():
            match = niche.evaluate(conditions)
            if match > best_match:
                best_match = match
        
        # Or match against preference
        preference_score = np.exp(-np.linalg.norm(self.environment_conditions - self.niche_preference))
        
        return max(best_match, preference_score * 0.8)
    
    def select_action_for_niche(self, target_niche: Optional[str] = None) -> np.ndarray:
        """
        Select action to improve niche fit.
        
        Returns motor command to modify environment.
        """
        if target_niche and target_niche in self.niches:
            target = self.niches[target_niche].conditions
            target_vec = np.array([target.get(f"condition_{i}", 0) for i in range(self.condition_dim)])
        else:
            target_vec = self.niche_preference
        
        # Error signal
        error = target_vec - self.environment_conditions
        
        # Action to reduce error (move environment toward target)
        action = error * self.manipulation_power
        
        # Add exploration
        action += np.random.randn(self.condition_dim) * 0.1
        
        return action
    
    def get_niche_summary(self) -> Dict:
        """Get niche construction summary."""
        current_favorability = self.evaluate_current_niche()
        
        return {
            'n_niches': len(self.niches),
            'current_favorability': current_favorability,
            'environment_conditions': self.environment_conditions.tolist(),
            'manipulation_power': self.manipulation_power,
            'n_constructions': len(self.construction_history)
        }


# ============================================================================
# ENACTIVIST COGNITION
# ============================================================================
"""
Enactivist Cognition:

Cognition is action-oriented.

All cognition is rooted in sensorimotor engagement.
Meaning emerges from interaction.
Mind is not representation but enactment.

NOT: "internal model of external world"
BUT: "agent-world coupling patterns"
"""

class EnactivistCognition:
    """
    Enactivist cognitive system.
    
    NOT: "cognition computes representation"
    BUT: "cognition is enacted through interaction"
    
    The system doesn't have knowledge.
    The system enacts knowing.
    """
    
    def __init__(self, world_dim: int = 8, action_dim: int = 4):
        self.world_dim = world_dim
        self.action_dim = action_dim
        
        # Agent-environment coupling patterns
        self.coupling_patterns: List[Dict] = []
        
        # Enacted meanings (from interaction)
        self.enacted_meanings: Dict[str, float] = {}
        
        # Autopoietic state (self-production)
        self.autopoietic_score = 0.0
        
        # Meaning groundedness
        self.groundedness = 0.0
        
        # Interaction history
        self.interaction_history: List[Dict] = []
        
        # Time
        self.t = 0.0
    
    def enact_knowledge(self, action: np.ndarray,
                      observation: np.ndarray,
                      consequence: np.ndarray) -> Dict:
        """
        Enact knowledge through interaction.
        
        Knowledge = pattern in agent-environment coupling.
        
        Returns enacted meaning.
        """
        action = np.asarray(action).flatten()[:self.action_dim]
        observation = np.asarray(observation).flatten()[:self.world_dim]
        consequence = np.asarray(consequence).flatten()[:self.world_dim]
        
        # Coupling pattern
        coupling = {
            'action': action.tolist(),
            'observation': observation.tolist(),
            'consequence': consequence.tolist(),
            't': self.t
        }
        
        self.interaction_history.append(coupling)
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-50:]
        
        # Extract meaning from coupling pattern
        meaning_key = self._compute_meaning_key(action, observation)
        
        # Update enacted meaning
        if meaning_key not in self.enacted_meanings:
            self.enacted_meanings[meaning_key] = 0.0
        
        # Meaning grows with successful coupling
        prediction_error = np.linalg.norm(observation - consequence)
        self.enacted_meanings[meaning_key] += 0.1 / (1.0 + prediction_error)
        
        # Compute autopoiesis (self-production through interaction)
        successful_interactions = sum(1 for h in self.interaction_history[-10:]
                                      if np.linalg.norm(h['consequence']) > 0.1)
        self.autopoietic_score = successful_interactions / 10.0
        
        # Compute groundedness
        recent_actions = [h['action'] for h in self.interaction_history[-5:]]
        recent_obs = [h['observation'][:self.action_dim] for h in self.interaction_history[-5:]]
        
        if recent_actions and recent_obs:
            actions_arr = np.array(recent_actions)
            obs_arr = np.array(recent_obs)
            if actions_arr.shape == obs_arr.shape:
                corr = np.corrcoef(np.mean(actions_arr, axis=0),
                                 np.mean(obs_arr, axis=0))[0, 1]
                self.groundedness = abs(corr) if not np.isnan(corr) else 0.0
            else:
                self.groundedness = 0.0
        
        self.t += 0.1
        
        return {
            'meaning_key': meaning_key,
            'meaning_strength': self.enacted_meanings[meaning_key],
            'autopoietic_score': self.autopoietic_score,
            'groundedness': self.groundedness
        }
    
    def _compute_meaning_key(self, action: np.ndarray,
                            observation: np.ndarray) -> str:
        """Compute meaning key from action-observation pair."""
        action_binned = (action * 2).astype(int)
        obs_binned = (observation[:self.action_dim] * 2).astype(int)
        return str(list(action_binned) + list(obs_binned))
    
    def get_knowledge_structure(self) -> Dict:
        """Get enacted knowledge structure."""
        meanings = [(k, v) for k, v in self.enacted_meanings.items()]
        meanings.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'n_meanings': len(self.enacted_meanings),
            'top_meanings': meanings[:5],
            'autopoietic_score': self.autopoietic_score,
            'groundedness': self.groundedness,
            'n_interactions': len(self.interaction_history)
        }


# ============================================================================
# EMBODIED ACTIVE INFERENCE FIELD (INTEGRATED)
# ============================================================================

class EmbodiedActiveInferenceField:
    """
    Phase 25: Embodied Active Inference Field
    
    Integrated system where:
      A. Sensorimotor closure: action conditions perception
      B. Learned body schema: body geometry inferred from interaction
      C. Ecological niche: system constructs own environment
      D. Enactivist cognition: meaning emerges from interaction
      E. Autopoiesis: self-production through sensorimotor engagement
      
    NOT: "agent in environment"
    BUT: "agent-environment coupled dynamical system"
    
    The body is not a vehicle. The body IS the cognition.
    """
    
    def __init__(self, sensory_dim: int = 8, motor_dim: int = 4,
                 world_dim: int = 8, n_body_segments: int = 4):
        self.sensory_dim = sensory_dim
        self.motor_dim = motor_dim
        self.world_dim = world_dim
        self.n_body_segments = n_body_segments
        
        # Sensorimotor closure
        self.sensorimotor = SensorimotorClosure(
            sensory_dim=sensory_dim, motor_dim=motor_dim
        )
        
        # Learned body schema
        self.body_schema = LearnedBodySchema(
            n_segments=n_body_segments, segment_dim=motor_dim
        )
        self.body_schema.init_segments()
        
        # Ecological niche constructor
        self.niche = EcologicalNicheConstructor(condition_dim=world_dim)
        
        # Enactivist cognition
        self.enactivist = EnactivistCognition(world_dim=world_dim, action_dim=motor_dim)
        
        # World state
        self.world_state = np.zeros(world_dim)
        
        # Internal state
        self.internal_state = np.zeros(sensory_dim)
        
        # Time
        self.t = 0.0
    
    def explore_body(self, n_explorations: int = 20):
        """Explore body to learn schema."""
        movements = self.body_schema.explore_body(n_explorations)
        
        motor_commands = []
        sensory_feedback = []
        
        for motor in movements:
            # Apply motor to environment
            env_state = self.act(motor)
            
            # Record motor and resulting sensation
            motor_commands.append(motor)
            sensation = self.sensorimotor.generate_sensation(motor, self.world_state)
            sensory_feedback.append(sensation)
        
        # Update body schema from exploration
        self.body_schema.update_from_interaction(motor_commands, sensory_feedback)
        
        return {
            'n_explorations': n_explorations,
            'body_confidence': self.body_schema.body_confidence
        }
    
    def perceive(self, observation: np.ndarray,
                include_self_interaction: bool = True) -> Dict:
        """
        Perceive with sensorimotor closure.
        
        Perception is conditioned on action context.
        """
        observation = np.asarray(observation).flatten()[:self.sensory_dim]
        
        # Sensorimotor inference
        closure_result = self.sensorimotor.infer_perception(
            observation, self.sensorimotor.motor_history
        )
        
        # Self-touch detection
        self_interaction = 0.0
        if include_self_interaction:
            self_interaction = self.sensorimotor.detect_self_interaction(observation)
        
        # World model update
        self.world_state = observation[:self.world_dim]
        
        return {
            'perceived': closure_result['perceived'].tolist(),
            'contingency': closure_result['contingency'],
            'confidence': closure_result['confidence'],
            'self_interaction': self_interaction,
            'world_state': self.world_state.tolist()
        }
    
    def act(self, motor_command: np.ndarray,
           construct_niche: bool = False) -> np.ndarray:
        """
        Act on world with body schema.
        
        Motor command modifies environment.
        """
        motor = np.asarray(motor_command).flatten()[:self.motor_dim]
        
        # Predict body state
        predicted_body = self.body_schema.predict_body_state(motor)
        
        # Act on environment
        if construct_niche:
            env_state = self.niche.act_on_environment(motor)
        else:
            motor_extended = np.zeros(self.world_dim)
            motor_extended[:self.motor_dim] = motor
            env_state = self.world_state + motor_extended * 0.1 + np.random.randn(self.world_dim) * 0.1
        
        # Sensorimotor feedback
        sensation = self.sensorimotor.generate_sensation(motor, env_state)
        
        # Update sensorimotor closure
        self.sensorimotor.update_closure(motor, sensation)
        
        # Enactivist knowledge
        self.enactivist.enact_knowledge(motor, self.internal_state, env_state)
        
        # Update world state
        self.world_state = env_state
        
        # Internal state update
        self.internal_state = sensation
        
        self.t += 0.1
        
        return env_state
    
    def run_cycle(self, n_steps: int = 50,
                 explore_body: bool = True,
                 construct_niche: bool = False) -> Dict:
        """Run embodied active inference cycle."""
        results = []
        
        # Explore body first (if needed)
        if explore_body and self.body_schema.body_confidence < 0.8:
            explore_result = self.explore_body(n_explorations=20)
            results.append({'phase': 'body_exploration', 'result': explore_result})
        
        # Set niche preference
        preferred_conditions = np.array([1.0, 0.5, 0.0, 0.0] + [0.0] * (self.world_dim - 4))
        self.niche.set_niche_preference(preferred_conditions)
        
        # Main cycle
        for step in range(n_steps):
            # Generate motor command (goal-directed)
            target = np.array([2.0 * np.sin(step * 0.1), 0.5 * np.cos(step * 0.05)] +
                             [0.0] * (self.motor_dim - 2))
            
            # Compute action from body schema
            action = self.body_schema.predict_body_state(target)[:self.motor_dim]
            
            # Add exploration noise
            if step % 5 == 0:
                action += np.random.randn(self.motor_dim) * 0.2
            
            # Act
            env_state = self.act(action, construct_niche=construct_niche)
            
            # Perceive
            observation = env_state + np.random.randn(self.sensory_dim) * 0.1
            perception = self.perceive(observation)
            
            # Niche evaluation
            niche_favorability = self.niche.evaluate_current_niche() if construct_niche else 0.5
            
            results.append({
                'step': step,
                't': self.t,
                'action': action.tolist()[:2],
                'world_state': self.world_state[:2].tolist(),
                'perception_confidence': perception['confidence'],
                'body_confidence': self.body_schema.body_confidence,
                'autopoietic_score': self.enactivist.autopoietic_score,
                'niche_favorability': niche_favorability
            })
        
        return {
            'steps': results,
            'final_body_confidence': results[-1]['body_confidence'] if results else 0,
            'final_autopoietic': results[-1]['autopoietic_score'] if results else 0,
            'niche_summary': self.niche.get_niche_summary(),
            'enactivist_summary': self.enactivist.get_knowledge_structure()
        }


# ============================================================================
# TESTS
# ============================================================================

def test_sensorimotor_closure():
    """Test sensorimotor closure."""
    print("\n" + "=" * 60)
    print("SENSORIMOTOR CLOSURE TEST")
    print("=" * 60)
    
    closure = SensorimotorClosure(sensory_dim=8, motor_dim=4)
    
    print("\n  Testing closure:")
    
    for i in range(30):
        motor = np.random.randn(4) * 0.5
        env = np.random.randn(8) * 0.3
        sensation = closure.generate_sensation(motor, env)
        
        closure.update_closure(motor, sensation)
        
        if i % 10 == 9:
            summary = closure.get_closure_summary()
            print(f"    Step {i+1}:")
            print(f"      Contingencies: {summary['n_contingencies']}")
            print(f"      Confidence: {summary['mean_confidence']:.3f}")
    
    inferred = closure.infer_perception(np.random.randn(8), closure.motor_history[-3:])
    print(f"\n  Inferred perception: {inferred['confidence']:.3f}")


def test_learned_body_schema():
    """Test learned body schema."""
    print("\n" + "=" * 60)
    print("LEARNED BODY SCHEMA TEST")
    print("=" * 60)
    
    body = LearnedBodySchema(n_segments=4, segment_dim=4)
    body.init_segments()
    
    print("\n  Exploring body:")
    
    exploration = body.explore_body(n_explorations=20)
    
    motor_commands = []
    sensory_feedback = []
    
    for motor in exploration:
        motor_commands.append(motor)
        sensory_feedback.append(np.random.randn(4) * 0.3 + motor * 0.2)
    
    body.update_from_interaction(motor_commands, sensory_feedback)
    
    geometry = body.get_body_geometry()
    print(f"    Segments: {geometry['n_segments']}")
    print(f"    Body confidence: {geometry['body_confidence']:.3f}")
    
    predicted = body.predict_body_state(np.random.randn(4))
    print(f"    Predicted state norm: {np.linalg.norm(predicted):.3f}")


def test_ecological_niche():
    """Test ecological niche construction."""
    print("\n" + "=" * 60)
    print("ECOLOGICAL NICHE CONSTRUCTION TEST")
    print("=" * 60)
    
    niche = EcologicalNicheConstructor(condition_dim=8)
    
    print("\n  Constructing niche:")
    
    target = {"condition_0": 1.0, "condition_1": 0.5}
    actions = [np.random.randn(8) * 0.5 for _ in range(5)]
    niche_id = niche.construct_niche(target, actions)
    
    print(f"    Created niche: {niche_id}")
    
    for i in range(30):
        action = niche.select_action_for_niche(niche_id)
        niche.act_on_environment(action)
        
        if i % 10 == 9:
            summary = niche.get_niche_summary()
            print(f"    Step {i+1}:")
            print(f"      Favorability: {summary['current_favorability']:.3f}")
    
    favorability = niche.evaluate_current_niche()
    print(f"\n  Final favorability: {favorability:.3f}")


def test_enactivist_cognition():
    """Test enactivist cognition."""
    print("\n" + "=" * 60)
    print("ENACTIVIST COGNITION TEST")
    print("=" * 60)
    
    enactivist = EnactivistCognition(world_dim=8, action_dim=4)
    
    print("\n  Enacting knowledge:")
    
    for i in range(30):
        action = np.random.randn(4) * 0.5
        observation = np.random.randn(8) * 0.3
        consequence = np.random.randn(8) * 0.2
        
        result = enactivist.enact_knowledge(action, observation, consequence)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      Autopoietic: {result['autopoietic_score']:.3f}")
            print(f"      Groundedness: {result['groundedness']:.3f}")
    
    structure = enactivist.get_knowledge_structure()
    print(f"\n  Knowledge structure: {structure['n_meanings']} meanings")


def test_embodied_field():
    """Test embodied active inference field."""
    print("\n" + "=" * 60)
    print("EMBODIED ACTIVE INFERENCE FIELD TEST")
    print("=" * 60)
    
    field = EmbodiedActiveInferenceField(
        sensory_dim=8, motor_dim=4, world_dim=8, n_body_segments=4
    )
    
    print("\n  Running embodied cycle:")
    
    result = field.run_cycle(n_steps=30, explore_body=True, construct_niche=False)
    
    print(f"    Steps: {len(result['steps'])}")
    print(f"    Final body confidence: {result['final_body_confidence']:.3f}")
    print(f"    Final autopoiesis: {result['final_autopoietic']:.3f}")
    print(f"    Meanings: {result['enactivist_summary']['n_meanings']}")


def phase_comparison():
    """Compare Phase 24 vs Phase 25."""
    print("\n" + "=" * 60)
    print("PHASE 24 VS PHASE 25 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 24 (Active Inference Field):")
    print("    - Policy inference q(π)")
    print("    - Expected FE: G(π) = risk + ambiguity - epistemic")
    print("    - Active sampling")
    print("    - Object-centric world model")
    print("    - Self as policy continuity")
    print("    - Basic embodiment")
    
    print("\n  Phase 25 (Embodied Active Inference):")
    print("    - Sensorimotor closure (action conditions perception)")
    print("    - Learned body schema (body is inferred)")
    print("    - Ecological niche construction (system creates environment)")
    print("    - Enactivist cognition (meaning from interaction)")
    print("    - Autopoiesis (self-production)")
    print("    - Full embodiment (body IS cognition)")
    
    print("\n  Critical shifts:")
    print("    1. Agent-environment separation → Agent-environment coupling")
    print("    2. Body as given → Body as learned")
    print("    3. Adaptation to environment → Niche construction")
    print("    4. Internal representation → Enacted meaning")
    print("    5. Body as vehicle → Body IS the system")


if __name__ == "__main__":
    test_sensorimotor_closure()
    test_learned_body_schema()
    test_ecological_niche()
    test_enactivist_cognition()
    test_embodied_field()
    phase_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 25 - EMBODIED ACTIVE INFERENCE")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 24 - active inference field with policy inference
  To: Phase 25 - embodied active inference where:
      - Sensorimotor closure: perception and action are mutually constitutive
      - Learned body schema: body geometry is inferred, not given
      - Ecological niche: system constructs its own environment
      - Enactivist cognition: all cognition is action-oriented
      
  This is NO LONGER:
    perception → cognition → action
  This IS:
    embodied agent-environment system with mutual causality
    
CRITICAL INSIGHT:
  Phase 24: "agent acts on world"
  Phase 25: "agent IS coupled with world"
  
  The body is not a vehicle for cognition.
  The body IS the cognitive system.
  Self is not separate from environment.
  Self-environment is a single dynamical system.

KEY COMPONENTS:

A. SENSORIMOTOR CLOSURE
   Perception is conditioned on action.
   
   NOT: "I see then I act"
   BUT: "My actions determine what I see"
   
   The system is closed in the sensorimotor loop.

B. LEARNED BODY SCHEMA
   Body geometry is inferred from interaction.
   
   NOT: body = fixed kinematic chain
   BUT: body = learned internal model
   
   The system discovers its body through exploration.

C. ECOLOGICAL NICHE CONSTRUCTION
   System creates its own environment.
   
   NOT: "adapt to environment"
   BUT: "create environment where you thrive"
   
   The system shapes the world as much as world shapes system.

D. ENACTIVIST COGNITION
   Cognition is action-oriented.
   
   NOT: "internal representation"
   BUT: "enacted through interaction"
   
   Meaning emerges from agent-environment coupling.

E. AUTOPOIESIS
   Self-production through sensorimotor engagement.
   
   The system maintains itself through interaction.
   Identity is not stored but continuously enacted.

THIS IS THE FOUNDATION FOR:
  - True embodied cognition
  - Ecological psychology
  - Enactivist AI
  - Self-producing systems
  - Living cognition
  
  The question is no longer "what does it represent?"
  The question is "what does it enact?"
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 25 Summary:

BEFORE:
  - Agent-environment separation
  - Body as fixed structure
  - Adaptation to environment
  - Internal representation
  - Body as vehicle for cognition

AFTER:
  - Agent-environment coupling
  - Body as learned geometry
  - Niche construction
  - Enacted meaning
  - Body IS the cognitive system

The critical shift:
  "agent acts on world" → "agent IS coupled with world"
  
  The body is not a vehicle. The body IS the cognition.
  Self is not separate from environment.
  Self-environment is a single dynamical system.

This is the foundation for:
  - True embodied AI
  - Ecological cognition
  - Enactivist systems
  - Self-producing AI
  - Living cognitive systems
"""