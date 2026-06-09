"""
LatentDynamicsModel - Learned Temporal Dynamics

NOT gradient descent on energy field.

BUT learned model:

(z_t, context, action) → z_t+k

This is the core of temporal cognition:
- Trajectory continuation
- Anticipation
- Surprise detection
- Active inference

The model learns:
1. How cognition typically evolves
2. How actions influence trajectory
3. What futures are likely
4. When reality diverges from prediction
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of dynamics prediction"""
    predicted_state: List[float] = field(default_factory=list)
    confidence: float = 0.0
    
    # Uncertainty
    variance: List[float] = field(default_factory=list)  # Per-dimension uncertainty
    total_uncertainty: float = 0.0
    
    # Trajectory context
    horizon: int = 0
    prediction_error: float = 0.0
    
    # Causal factors
    dominant_factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "confidence": round(self.confidence, 3),
            "total_uncertainty": round(self.total_uncertainty, 4),
            "horizon": self.horizon,
            "prediction_error": round(self.prediction_error, 4),
            "dominant_factors": self.dominant_factors,
        }


@dataclass
class RolloutPrediction:
    """
    Predicted trajectory rollout.
    
    Multiple steps into the future.
    """
    rollout_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Full predicted path
    predicted_states: List[List[float]] = field(default_factory=list)
    predicted_velocities: List[List[float]] = field(default_factory=list)
    
    # Uncertainty over trajectory
    uncertainty_trajectory: List[float] = field(default_factory=list)
    
    # Expected outcomes
    expected_utility: float = 0.0
    risk_score: float = 0.0
    
    # Comparison with training distribution
    novelty_score: float = 0.0
    typicality_score: float = 0.0
    
    # Metrics
    expected_duration_ms: float = 0.0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "rollout_id": self.rollout_id,
            "steps": len(self.predicted_states),
            "expected_utility": round(self.expected_utility, 3),
            "risk_score": round(self.risk_score, 3),
            "novelty_score": round(self.novelty_score, 3),
            "typicality_score": round(self.typicality_score, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass
class DynamicsModelStats:
    """Statistics about dynamics model"""
    total_predictions: int = 0
    avg_error: float = 0.0
    avg_uncertainty: float = 0.0
    
    # Model quality
    model_confidence: float = 0.0  # How well model fits training data
    generalization_score: float = 0.0
    
    # Temporal properties
    avg_horizon: float = 0.0
    max_horizon: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_predictions": self.total_predictions,
            "avg_error": round(self.avg_error, 4),
            "avg_uncertainty": round(self.avg_uncertainty, 4),
            "model_confidence": round(self.model_confidence, 3),
            "generalization_score": round(self.generalization_score, 3),
        }


class TransitionDynamics:
    """
    Transition dynamics model.
    
    Learns: P(z_t+k | z_t, context, action)
    
    This is the temporal component of the world model.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Learned parameters (simplified linear dynamics for now)
        # In production, this would be a neural network
        self.transition_matrix: List[List[float]] = self._init_matrix(dimension)
        self.context_weights: List[float] = [0.5] * dimension
        self.action_weights: List[float] = [0.3] * dimension
        
        # Velocity model
        self.velocity_matrix: List[List[float]] = self._init_matrix(dimension)
        
        # Training data statistics
        self.transition_history: List[Tuple[List[float], List[float], List[float]]] = []
        
        logger.info("transition_dynamics_initialized", dimension=dimension)
    
    def _init_matrix(self, dim: int) -> List[List[float]]:
        """Initialize transition matrix with small random values"""
        import random
        return [
            [random.uniform(-0.1, 0.1) for _ in range(dim)]
            for _ in range(dim)
        ]
    
    def fit(
        self,
        transitions: List[Tuple[List[float], List[float], List[float]]]
    ) -> None:
        """
        Fit transition dynamics from observations.
        
        transitions = [(z_t, z_t+k, context)]
        
        Currently uses simple regression. In production, use neural network.
        """
        self.transition_history.extend(transitions)
        
        if len(transitions) < 10:
            logger.warning("insufficient_data_for_fit", count=len(transitions))
            return
        
        # Simple matrix regression (simplified)
        n = len(transitions)
        
        # Compute mean transition
        delta_sum = [0.0] * self.dimension
        z_sum = [0.0] * self.dimension
        
        for z_t, z_tk, _ in transitions:
            for i in range(self.dimension):
                delta_sum[i] += z_tk[i] - z_t[i]
                z_sum[i] += z_t[i]
        
        # Simple linear model: delta ≈ A * z
        z_mean = [z / n for z in z_sum]
        delta_mean = [d / n for d in delta_sum]
        
        # Update transition matrix (simplified)
        for i in range(self.dimension):
            for j in range(self.dimension):
                # Approximate gradient descent
                gradient = 0.0
                for z_t, z_tk, _ in transitions[-100:]:  # Use recent samples
                    delta = z_tk[i] - z_t[i]
                    predicted = sum(self.transition_matrix[i][k] * z_t[k] for k in range(self.dimension))
                    gradient += (delta - predicted) * z_t[j]
                
                if len(transitions) > 0:
                    self.transition_matrix[i][j] += 0.001 * gradient / len(transitions[-100:])
    
    def predict(
        self,
        current_state: List[float],
        context: Optional[List[float]] = None,
        action: Optional[List[float]] = None,
        horizon: int = 1
    ) -> Tuple[List[float], float]:
        """
        Predict next state.
        
        Returns:
        - predicted_state
        - uncertainty
        """
        if horizon == 1:
            delta = self._compute_delta(current_state, context, action)
            
            predicted = [
                current_state[i] + delta[i]
                for i in range(self.dimension)
            ]
            
            uncertainty = self._compute_uncertainty(current_state)
            
            return predicted, uncertainty
        
        # Multi-step prediction
        predicted = current_state[:]
        
        for _ in range(horizon):
            delta = self._compute_delta(predicted, context, action)
            predicted = [
                predicted[i] + delta[i]
                for i in range(self.dimension)
            ]
        
        uncertainty = self._compute_uncertainty(current_state) * math.sqrt(horizon)
        
        return predicted, uncertainty
    
    def _compute_delta(
        self,
        state: List[float],
        context: Optional[List[float]],
        action: Optional[List[float]]
    ) -> List[float]:
        """Compute state change"""
        # Base transition
        delta = [
            sum(self.transition_matrix[i][j] * state[j] for j in range(self.dimension))
            for i in range(self.dimension)
        ]
        
        # Context influence
        if context:
            for i in range(self.dimension):
                delta[i] += self.context_weights[i] * context[i]
        
        # Action influence
        if action:
            for i in range(self.dimension):
                delta[i] += self.action_weights[i] * action[i]
        
        return delta
    
    def _compute_uncertainty(self, state: List[float]) -> float:
        """Compute prediction uncertainty"""
        if len(self.transition_history) < 10:
            return 1.0
        
        # Variance in recent transitions
        recent = self.transition_history[-100:]
        
        errors = []
        for z_t, z_tk, _ in recent:
            predicted, _ = self.predict(z_t, horizon=1)
            error = math.sqrt(sum((predicted[i] - z_tk[i]) ** 2 for i in range(self.dimension)))
            errors.append(error)
        
        return sum(errors) / len(errors) if errors else 1.0


class LatentDynamicsModel:
    """
    Latent Dynamics Model - learned temporal world model.
    
    This is the core of Phase 13.
    
    It learns:
    - How cognitive state typically evolves
    - How actions influence trajectories
    - What futures are likely
    - When predictions fail (surprise)
    
    NOT gradient descent. NOT rule-based.
    
    BUT learned from trajectory statistics.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        self.transition_dynamics = TransitionDynamics(dimension)
        self.phase_predictor = None  # Will be phase_space
        
        # Training data
        self.observations: List[Dict] = []
        
        # Model statistics
        self.stats = DynamicsModelStats()
        
        # Velocity model (for phase space)
        self.velocity_dynamics = TransitionDynamics(dimension)
        
        logger.info("latent_dynamics_model_initialized", dimension=dimension)
    
    def observe(
        self,
        current_state: List[float],
        next_state: List[float],
        context: Optional[Dict] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> None:
        """Add observation to training data"""
        obs = {
            "current_state": current_state,
            "next_state": next_state,
            "context": context or {},
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.utcnow()
        }
        
        self.observations.append(obs)
        
        # Update statistics
        self.stats.total_predictions += 1
    
    def train(self) -> None:
        """Train dynamics model from observations"""
        if len(self.observations) < 10:
            logger.warning("insufficient_observations", count=len(self.observations))
            return
        
        # Prepare training data
        transitions = []
        
        for obs in self.observations:
            z_t = obs["current_state"]
            z_tk = obs["next_state"]
            context_vec = self._context_to_vector(obs.get("context", {}))
            
            transitions.append((z_t, z_tk, context_vec))
        
        # Fit transition dynamics
        self.transition_dynamics.fit(transitions)
        
        # Update model statistics
        self._update_stats()
        
        logger.info("dynamics_model_trained", observations=len(self.observations))
    
    def _context_to_vector(self, context: Dict) -> List[float]:
        """Convert context dict to vector"""
        vec = [0.0] * self.dimension
        
        if "motivation" in context:
            vec[0] = context["motivation"]
        if "fatigue" in context:
            vec[1] = context["fatigue"]
        if "attention" in context:
            vec[2] = context["attention"]
        
        return vec
    
    def _update_stats(self) -> None:
        """Update model statistics"""
        if len(self.observations) < 10:
            return
        
        # Compute average prediction error
        errors = []
        uncertainties = []
        
        for obs in self.observations[-100:]:
            predicted, uncertainty = self.transition_dynamics.predict(
                obs["current_state"],
                context=self._context_to_vector(obs.get("context", {}))
            )
            
            error = math.sqrt(sum(
                (predicted[i] - obs["next_state"][i]) ** 2
                for i in range(self.dimension)
            ))
            
            errors.append(error)
            uncertainties.append(uncertainty)
        
        self.stats.avg_error = sum(errors) / len(errors) if errors else 0.0
        self.stats.avg_uncertainty = sum(uncertainties) / len(uncertainties) if uncertainties else 0.0
        
        # Model confidence: inversely related to error
        self.stats.model_confidence = max(0, 1.0 - self.stats.avg_error)
    
    def predict(
        self,
        current_state: List[float],
        context: Optional[Dict] = None,
        action: Optional[str] = None
    ) -> PredictionResult:
        """Predict next state"""
        context_vec = self._context_to_vector(context or {})
        action_vec = self._action_to_vector(action)
        
        predicted, uncertainty = self.transition_dynamics.predict(
            current_state,
            context=context_vec,
            action=action_vec
        )
        
        return PredictionResult(
            predicted_state=predicted,
            confidence=max(0, 1.0 - uncertainty),
            variance=[uncertainty] * self.dimension,
            total_uncertainty=uncertainty,
            horizon=1,
            dominant_factors=self._get_dominant_factors(current_state)
        )
    
    def _action_to_vector(self, action: Optional[str]) -> Optional[List[float]]:
        """Convert action to vector"""
        if not action:
            return None
        
        vec = [0.0] * self.dimension
        
        action_mapping = {
            "execute": [1.0, 0.0, 0.0],
            "explore": [0.0, 1.0, 0.0],
            "decompose": [0.0, 0.0, 1.0],
            "wait": [0.2, 0.0, 0.0],
            "reconsider": [0.3, 0.3, 0.0],
        }
        
        mapped = action_mapping.get(action, [0.1, 0.1, 0.1])
        for i in range(min(len(mapped), self.dimension)):
            vec[i] = mapped[i]
        
        return vec
    
    def _get_dominant_factors(self, state: List[float]) -> List[str]:
        """Determine dominant causal factors"""
        factors = []
        
        magnitude = math.sqrt(sum(s ** 2 for s in state))
        
        if magnitude < 0.1:
            factors.append("stationary")
        elif magnitude > 0.5:
            factors.append("high_momentum")
        
        # Check for specific dimensions
        if len(state) > 0 and abs(state[0]) > 0.3:
            factors.append("confidence_dominant")
        if len(state) > 1 and state[1] > 0.3:
            factors.append("stress_elevated")
        
        return factors
    
    def predict_rollout(
        self,
        start_state: List[float],
        context: Optional[Dict] = None,
        horizon: int = 5,
        action_sequence: Optional[List[str]] = None
    ) -> RolloutPrediction:
        """
        Predict multi-step trajectory.
        
        This is the key method for anticipation.
        """
        predicted_states = [start_state]
        uncertainties = [0.0]
        
        current = start_state
        
        for step in range(horizon):
            action = action_sequence[step] if action_sequence and step < len(action_sequence) else None
            
            result = self.predict(current, context, action)
            
            predicted_states.append(result.predicted_state)
            uncertainties.append(result.total_uncertainty)
            
            current = result.predicted_state
        
        # Compute rollout metrics
        rollout = RolloutPrediction(
            predicted_states=predicted_states,
            uncertainty_trajectory=uncertainties,
            horizon=horizon,
            confidence=1.0 - sum(uncertainties) / len(uncertainties) if uncertainties else 0
        )
        
        # Estimate utility and risk
        rollout.expected_utility = self._estimate_utility(predicted_states)
        rollout.risk_score = max(uncertainties) if uncertainties else 0.0
        rollout.novelty_score = self._compute_novelty(predicted_states)
        rollout.typicality_score = 1.0 - rollout.novelty_score
        
        return rollout
    
    def _estimate_utility(self, states: List[List[float]]) -> float:
        """Estimate expected utility of trajectory"""
        if not states:
            return 0.0
        
        # Simple utility: converging to stable states
        utilities = []
        
        for i in range(len(states) - 1):
            delta = math.sqrt(sum(
                (states[i+1][j] - states[i][j]) ** 2
                for j in range(self.dimension)
            ))
            # Lower delta = more stable = higher utility
            utilities.append(1.0 / (1.0 + delta))
        
        return sum(utilities) / len(utilities) if utilities else 0.5
    
    def _compute_novelty(self, states: List[List[float]]) -> float:
        """Compute novelty score of trajectory"""
        if len(states) < 2:
            return 0.0
        
        # Novelty: deviation from typical transitions
        typical_delta = 0.2  # Assumed typical step size
        
        novelty = 0.0
        for i in range(len(states) - 1):
            delta = math.sqrt(sum(
                (states[i+1][j] - states[i][j]) ** 2
                for j in range(self.dimension)
            ))
            
            novelty += abs(delta - typical_delta)
        
        return min(1.0, novelty / len(states))
    
    def compute_surprise(
        self,
        predicted_state: List[float],
        actual_state: List[float]
    ) -> float:
        """
        Compute surprise = prediction error.
        
        surprise = ||predicted - actual||
        
        This is the foundation of active inference.
        """
        if len(predicted_state) != len(actual_state):
            return 1.0
        
        error = math.sqrt(sum(
            (p - a) ** 2
            for p, a in zip(predicted_state, actual_state)
        ))
        
        # Normalize by dimension
        surprise = error / math.sqrt(self.dimension)
        
        return min(1.0, surprise)
    
    def get_statistics(self) -> Dict:
        """Get model statistics"""
        return self.stats.to_dict()


# Factory
def create_latent_dynamics_model(dimension: int = 16) -> LatentDynamicsModel:
    return LatentDynamicsModel(dimension=dimension)