"""
Phase 13 - Continuous Latent Dynamics

Unified engine integrating:
- PhaseSpace: position + velocity + acceleration
- LatentDynamicsModel: learned temporal model z_t → z_t+k
- LearnedEnergyField: E(z) from trajectory statistics
- ActiveInferenceEngine: minimize expected free energy

This is the transition from:
    discrete state machine
    
to:
    continuous cognitive physics

Key capabilities:
1. Temporal prediction: rollouts, anticipation, surprise
2. Energy-based dynamics: attractors, basins, gradients
3. Phase space: momentum, inertia, oscillation
4. Active inference: goal-directed exploration
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContinuousDynamicsState:
    """
    Current state of continuous dynamics system.
    """
    # Phase state
    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    acceleration: List[float] = field(default_factory=list)
    
    # Energy
    energy: float = 0.0
    kinetic_energy: float = 0.0
    
    # Dynamics
    momentum_magnitude: float = 0.0
    is_stationary: bool = False
    is_oscillating: bool = False
    
    # Prediction
    prediction_error: float = 0.0
    surprise: float = 0.0
    
    # Active inference
    expected_free_energy: float = 0.0
    preferred_action: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "momentum_magnitude": round(self.momentum_magnitude, 4),
            "is_stationary": self.is_stationary,
            "is_oscillating": self.is_oscillating,
            "energy": round(self.energy, 4),
            "prediction_error": round(self.prediction_error, 4),
            "surprise": round(self.surprise, 4),
            "expected_free_energy": round(self.expected_free_energy, 4),
            "preferred_action": self.preferred_action,
        }


class ContinuousLatentDynamics:
    """
    Phase 13 - Continuous Latent Dynamics Engine.
    
    This is the core of temporal cognitive physics.
    
    Integrates:
    - Phase space (position, velocity, acceleration)
    - Learned dynamics model (trajectory prediction)
    - Energy field (attractor topology)
    - Active inference (goal-directed action)
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        from .phase_space import PhaseSpace, create_phase_space
        from .latent_dynamics_model import LatentDynamicsModel, create_latent_dynamics_model
        from .learned_energy_field import (
            LearnedEnergyField, 
            ActiveInferenceEngine,
            create_learned_energy_field,
            create_active_inference_engine
        )
        
        # Core components
        self.phase_space = create_phase_space(dimension)
        self.dynamics_model = create_latent_dynamics_model(dimension)
        self.energy_field = create_learned_energy_field(dimension)
        self.active_inference = create_active_inference_engine(self.energy_field)
        
        # Current state
        self.current_phase_state = None
        
        logger.info("continuous_latent_dynamics_initialized", dimension=dimension)
    
    def observe(
        self,
        position: List[float],
        outcome: Optional[str] = None,
        action: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> PhaseState:
        """Observe new state and update all components"""
        from .phase_space import PhaseState
        
        # Compute phase state
        prev_state = self.current_phase_state
        
        if prev_state:
            prev_pos = prev_state.position
            prev_vel = prev_state.velocity
        else:
            prev_pos = None
            prev_vel = None
        
        phase_state = self.phase_space.create_phase_state(
            position=position,
            previous_position=prev_pos,
            previous_velocity=prev_vel,
            intention=self._get_intention_vector(context)
        )
        
        self.current_phase_state = phase_state
        
        # Update dynamics model
        if prev_state and prev_pos:
            self.dynamics_model.observe(
                current_state=prev_pos,
                next_state=position,
                context=context,
                action=action,
                outcome=outcome
            )
        
        # Update energy field
        self.energy_field.observe_trajectory([position])
        
        if outcome:
            self.energy_field.observe_outcome(position, outcome)
        
        return phase_state
    
    def _get_intention_vector(self, context: Optional[Dict]) -> List[float]:
        """Extract intention from context"""
        if not context:
            return [0.0] * self.dimension
        
        intention = [0.0] * self.dimension
        
        if "goal_direction" in context:
            dir_vec = context["goal_direction"]
            for i in range(min(len(dir_vec), self.dimension)):
                intention[i] = dir_vec[i]
        
        return intention
    
    def predict(
        self,
        current_state: List[float],
        horizon: int = 5,
        context: Optional[Dict] = None
    ) -> RolloutPrediction:
        """Predict future trajectory"""
        return self.dynamics_model.predict_rollout(
            start_state=current_state,
            context=context,
            horizon=horizon
        )
    
    def compute_energy(self, position: List[float]) -> float:
        """Compute energy at position"""
        return self.energy_field.compute_energy(position)
    
    def compute_gradient(self, position: List[float]) -> List[float]:
        """Compute energy gradient at position"""
        return self.energy_field.compute_gradient(position)
    
    def compute_surprise(
        self,
        predicted: List[float],
        actual: List[float]
    ) -> float:
        """Compute surprise (prediction error)"""
        return self.dynamics_model.compute_surprise(predicted, actual)
    
    def select_action(
        self,
        current_state: List[float],
        possible_actions: List[str],
        context: Optional[Dict] = None
    ) -> str:
        """Select action via active inference"""
        return self.active_inference.select_action(
            current_state=current_state,
            possible_actions=possible_actions,
            context=context
        )
    
    def train(self) -> None:
        """Train all learned components"""
        self.dynamics_model.train()
        self.energy_field.learn_energy_parameters()
        
        logger.info("continuous_dynamics_trained")
    
    def get_dynamics_state(self) -> ContinuousDynamicsState:
        """Get current continuous dynamics state"""
        state = ContinuousDynamicsState()
        
        if self.current_phase_state:
            phase = self.current_phase_state
            state.position = phase.position
            state.velocity = phase.velocity
            state.acceleration = phase.acceleration
            state.kinetic_energy = phase.kinetic_energy
            state.momentum_magnitude = phase.momentum_magnitude
            state.is_stationary = phase.is_stationary
            state.is_oscillating = phase.is_oscillating
        
        if phase.position:
            state.energy = self.energy_field.compute_energy(phase.position)
        
        return state
    
    def get_phase_trajectory(self) -> PhaseTrajectory:
        """Get current phase trajectory"""
        if self.phase_space.trajectories:
            return self.phase_space.trajectories[-1]
        return None
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            "dimension": self.dimension,
            "phase_space": self.phase_space.get_phase_space_statistics(),
            "dynamics_model": self.dynamics_model.get_statistics(),
            "energy_field": self.energy_field.get_energy_statistics().to_dict(),
            "current_state": self.get_dynamics_state().to_dict(),
        }
    
    def get_attractors(self) -> List[Dict]:
        """Get learned attractor regions"""
        return self.energy_field.find_attractors()
    
    def visualize_phase_flow(self) -> Dict:
        """Get phase flow visualization data"""
        return self.phase_space.get_phase_space_statistics()


# Factory
def create_continuous_latent_dynamics(dimension: int = 16) -> ContinuousLatentDynamics:
    return ContinuousLatentDynamics(dimension=dimension)


# Re-export for convenience
from .phase_space import PhaseSpace, PhaseState, PhaseTrajectory, create_phase_space
from .latent_dynamics_model import (
    LatentDynamicsModel,
    PredictionResult,
    RolloutPrediction,
    create_latent_dynamics_model
)
from .learned_energy_field import (
    LearnedEnergyField,
    ActiveInferenceEngine,
    create_learned_energy_field,
    create_active_inference_engine
)