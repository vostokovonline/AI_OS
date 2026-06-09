"""
TrajectoryRollout - Future Trajectory Manifold

This is where the system transitions from:

predict(next_state)

to:

predict(possible_futures)

This is the foundation for:
- Anticipation
- Expectation
- Surprise / prediction error
- Active inference
- Goal-conditioned future shaping
- Counterfactual planning

Key insight: Cognition doesn't predict single futures,
it maintains distribution over possible futures.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class FutureState:
    """
    Predicted future state with uncertainty.
    
    NOT single state - distribution over states.
    """
    step: int = 0
    timestamp: Optional[datetime] = None
    
    # Predicted state
    state_vector: List[float] = field(default_factory=list)
    
    # Uncertainty
    uncertainty: float = 0.0
    confidence: float = 0.0
    
    # Trajectory context
    trajectory_branch_id: str = ""
    parent_step: Optional[int] = None
    
    # Causal context
    likely_causes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "step": self.step,
            "uncertainty": round(self.uncertainty, 3),
            "confidence": round(self.confidence, 3),
            "trajectory_branch_id": self.trajectory_branch_id,
            "parent_step": self.parent_step,
        }


@dataclass
class TrajectoryRollout:
    """
    Full rollout prediction - sequence of future states.
    
    Represents manifold of possible futures.
    """
    rollout_id: str = field(default_factory=lambda: str(uuid4()))
    start_state_id: str = ""
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    # Predicted trajectory
    future_states: List[FutureState] = field(default_factory=list)
    
    # Branching info
    branch_count: int = 0
    max_depth: int = 0
    
    # Expected outcome
    expected_outcome: Optional[str] = None
    outcome_probability: float = 0.0
    
    # Metrics
    expected_utility: float = 0.0
    risk_score: float = 0.0
    
    # Trajectory properties
    expected_duration_ms: float = 0.0
    expected_events: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "rollout_id": self.rollout_id,
            "start_state_id": self.start_state_id,
            "steps": len(self.future_states),
            "branch_count": self.branch_count,
            "max_depth": self.max_depth,
            "expected_outcome": self.expected_outcome,
            "outcome_probability": round(self.outcome_probability, 3),
            "expected_utility": round(self.expected_utility, 3),
            "risk_score": round(self.risk_score, 3),
        }


@dataclass
class RolloutPlan:
    """
    Collection of rollout predictions for planning.
    
    This IS the future manifold for a given state.
    """
    plan_id: str = field(default_factory=lambda: str(uuid4()))
    current_state_id: str = ""
    
    # All possible rollouts
    rollouts: List[TrajectoryRollout] = field(default_factory=list)
    
    # Comparison metrics
    best_rollout_id: str = ""
    best_utility: float = 0.0
    
    # Anticipated surprise
    expected_surprise: float = 0.0  # Prediction error expected
    
    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "current_state_id": self.current_state_id,
            "rollout_count": len(self.rollouts),
            "best_rollout_id": self.best_rollout_id,
            "best_utility": round(self.best_utility, 3),
            "expected_surprise": round(self.expected_surprise, 3),
        }


class TrajectoryRollouter:
    """
    Predicts future trajectories from current state.
    
    NOT single-step prediction - multi-step rollout with branching.
    
    Uses:
    - Transition model from latent space
    - Motif transition matrix
    - Current cognitive state
    """
    
    def __init__(
        self,
        latent_space,
        transition_model,
        motif_transition_matrix,
        max_depth: int = 5,
        branch_factor: int = 2
    ):
        self.latent_space = latent_space
        self.transition_model = transition_model
        self.motif_matrix = motif_transition_matrix
        
        self.max_depth = max_depth
        self.branch_factor = branch_factor
        
        # Rollout history
        self.rollout_history: List[TrajectoryRollout] = []
        
        logger.info("trajectory_rollouter_initialized", 
                   max_depth=max_depth,
                   branch_factor=branch_factor)
    
    def predict_rollouts(
        self,
        current_state_id: str,
        possible_actions: List[str],
        include_branches: bool = True
    ) -> RolloutPlan:
        """
        Predict multiple rollout trajectories.
        
        Returns manifold of possible futures.
        """
        plan = RolloutPlan(current_state_id=current_state_id)
        
        current_state = self.latent_space.get_state(current_state_id)
        if not current_state:
            logger.warning("state_not_found", state_id=current_state_id)
            return plan
        
        for action in possible_actions:
            rollout = self._predict_single_rollout(
                current_state,
                action,
                depth=0,
                branch_id=""
            )
            plan.rollouts.append(rollout)
        
        if include_branches:
            for action in possible_actions[:self.branch_factor]:
                branches = self._generate_branches(
                    current_state,
                    action,
                    depth=1
                )
                plan.rollouts.extend(branches)
        
        plan.best_rollout_id = self._select_best_rollout(plan.rollouts)
        if plan.best_rollout_id:
            best = next((r for r in plan.rollouts if r.rollout_id == plan.best_rollout_id), None)
            if best:
                plan.best_utility = best.expected_utility
        
        self.rollout_history.extend(plan.rollouts)
        
        return plan
    
    def _predict_single_rollout(
        self,
        current_state,
        action: str,
        depth: int,
        branch_id: str
    ) -> TrajectoryRollout:
        """Predict single trajectory rollout"""
        rollout = TrajectoryRollout(
            start_state_id=current_state.id,
            branch_count=1
        )
        
        current = current_state
        step = 0
        cum_uncertainty = 0.0
        
        while step < self.max_depth:
            next_vector = self.transition_model.predict_next_state(current, action)
            uncertainty = self.transition_model.compute_prediction_error(
                next_vector, current.vector
            )
            
            future_state = FutureState(
                step=step,
                timestamp=datetime.utcnow(),
                state_vector=next_vector,
                uncertainty=uncertainty,
                confidence=max(0, 1 - uncertainty),
                trajectory_branch_id=branch_id or rollout.rollout_id,
                parent_step=step - 1 if step > 0 else None,
            )
            
            rollout.future_states.append(future_state)
            cum_uncertainty += uncertainty
            
            current = self.latent_space.get_state_by_vector(next_vector)
            if not current:
                break
            
            step += 1
        
        rollout.max_depth = depth
        rollout.expected_utility = self._compute_utility(rollout)
        rollout.risk_score = cum_uncertainty / max(1, len(rollout.future_states))
        
        return rollout
    
    def _generate_branches(
        self,
        current_state,
        action: str,
        depth: int
    ) -> List[TrajectoryRollout]:
        """Generate branching trajectories"""
        branches = []
        
        if depth >= self.max_depth:
            return branches
        
        possible_next_motifs = self.motif_matrix.get_next_motif_distribution(
            current_state.motif_id or ""
        )
        
        for motif, prob in possible_next_motifs[:self.branch_factor]:
            branch_id = f"{action}_{motif}"
            
            modified_state = self._apply_motif_context(current_state, motif)
            
            rollout = self._predict_single_rollout(
                modified_state,
                action,
                depth=depth,
                branch_id=branch_id
            )
            rollout.branch_count = 1
            rollout.expected_outcome = motif
            rollout.outcome_probability = prob
            
            branches.append(rollout)
        
        return branches
    
    def _apply_motif_context(self, state, target_motif: str) -> Any:
        """Apply motif context to state prediction"""
        motif_stats = self.motif_matrix.get_flow_statistics(target_motif)
        
        if motif_stats.preferred_exit:
            transition = self.motif_matrix.transitions.get(target_motif, {}).get(motif_stats.preferred_exit)
            if transition:
                bias = transition.probability
                return state
        
        return state
    
    def _compute_utility(self, rollout: TrajectoryRollout) -> float:
        """Compute expected utility of rollout"""
        if not rollout.future_states:
            return 0.0
        
        utility = 0.0
        for i, state in enumerate(rollout.future_states):
            weight = 0.9 ** i
            utility += state.confidence * weight
        
        return utility / len(rollout.future_states)
    
    def _select_best_rollout(self, rollouts: List[TrajectoryRollout]) -> Optional[str]:
        """Select best rollout by utility"""
        if not rollouts:
            return None
        
        best = max(rollouts, key=lambda r: r.expected_utility)
        return best.rollout_id
    
    def compute_surprise(
        self,
        rollout: TrajectoryRollout,
        actual_state: List[float]
    ) -> float:
        """
        Compute prediction error (surprise).
        
        surprise = ||predicted - actual||
        """
        if not rollout.future_states:
            return 1.0
        
        last_predicted = rollout.future_states[-1].state_vector
        
        error = math.sqrt(sum(
            (p - a) ** 2 
            for p, a in zip(last_predicted, actual_state)
        ))
        
        surprise = min(1.0, error / math.sqrt(len(actual_state)))
        
        return surprise
    
    def get_rollout_statistics(self) -> Dict:
        """Get rollout statistics"""
        if not self.rollout_history:
            return {
                "total_rollouts": 0,
                "avg_utility": 0,
                "avg_risk": 0,
            }
        
        return {
            "total_rollouts": len(self.rollout_history),
            "avg_utility": sum(r.expected_utility for r in self.rollout_history) / len(self.rollout_history),
            "avg_risk": sum(r.risk_score for r in self.rollout_history) / len(self.rollout_history),
            "avg_depth": sum(r.max_depth for r in self.rollout_history) / len(self.rollout_history),
        }


class BehavioralFlowField:
    """
    Continuous flow field over behavioral space.
    
    This represents the "world model" as a flow field:
    - At each point in behavioral space, what's the likely next point?
    - How does cognition flow through attractor basins?
    
    NOT discrete transitions - continuous flow.
    """
    
    def __init__(self, latent_space, motif_transition_matrix):
        self.latent_space = latent_space
        self.motif_matrix = motif_transition_matrix
        
        # Flow field approximation
        self.flow_vectors: Dict[str, List[float]] = {}
        self.flow_magnitude: Dict[str, float] = {}
        
        logger.info("behavioral_flow_field_initialized")
    
    def compute_flow_at(
        self,
        state_id: str,
        current_motif: Optional[str] = None
    ) -> Tuple[List[float], float]:
        """
        Compute flow vector at given state.
        
        Returns:
        - flow_direction: Where cognition is likely to go
        - flow_magnitude: How strong the flow is
        """
        state = self.latent_space.get_state(state_id)
        if not state:
            return [0.0] * state.dimension, 0.0
        
        flow_direction = [0.0] * state.dimension
        total_weight = 0.0
        
        if current_motif:
            next_motifs = self.motif_matrix.get_next_motif_distribution(current_motif)
            
            for motif, prob in next_motifs:
                motif_state = self.motif_matrix.motif_stats.get(motif)
                if motif_state and motif_state.centroid:
                    for i in range(len(flow_direction)):
                        diff = motif_state.centroid[i] - state.vector[i]
                        flow_direction[i] += diff * prob
                    total_weight += prob
        
        if total_weight > 0:
            flow_direction = [v / total_weight for v in flow_direction]
        
        flow_magnitude = math.sqrt(sum(v ** 2 for v in flow_direction))
        
        return flow_direction, flow_magnitude
    
    def compute_gradient(self, state_id: str) -> List[float]:
        """
        Compute gradient of flow field.
        
        This tells us the direction of steepest ascent in the flow.
        """
        state = self.latent_space.get_state(state_id)
        if not state:
            return [0.0] * state.dimension
        
        epsilon = 0.01
        gradient = []
        
        for i in range(len(state.vector)):
            perturbed_plus = state.vector[:]
            perturbed_plus[i] += epsilon
            
            _, mag_plus = self.compute_flow_at(
                self.latent_space.get_state_by_vector(perturbed_plus).id
                if self.latent_space.get_state_by_vector(perturbed_plus) else state_id
            )
            
            perturbed_minus = state.vector[:]
            perturbed_minus[i] -= epsilon
            
            _, mag_minus = self.compute_flow_at(
                self.latent_space.get_state_by_vector(perturbed_minus).id
                if self.latent_space.get_state_by_vector(perturbed_minus) else state_id
            )
            
            gradient.append((mag_plus - mag_minus) / (2 * epsilon))
        
        return gradient
    
    def find_attractor_path(
        self,
        from_state_id: str,
        to_motif: str
    ) -> List[str]:
        """
        Find path to attractor basin.
        
        Uses gradient descent along flow field.
        """
        path = [from_state_id]
        current = from_state_id
        max_iterations = 50
        
        for _ in range(max_iterations):
            gradient = self.compute_gradient(current)
            
            step_size = 0.1
            state = self.latent_space.get_state(current)
            if not state:
                break
            
            next_vector = [
                state.vector[i] + step_size * gradient[i]
                for i in range(len(state.vector))
            ]
            
            next_state = self.latent_space.get_state_by_vector(next_vector)
            if not next_state:
                break
            
            current = next_state.id
            path.append(current)
            
            if next_state.motif_id == to_motif:
                break
        
        return path
    
    def get_flow_statistics(self) -> Dict:
        """Get flow field statistics"""
        return {
            "flow_vectors_computed": len(self.flow_vectors),
            "avg_flow_magnitude": sum(self.flow_magnitude.values()) / max(1, len(self.flow_magnitude)),
        }


# Factory
def create_trajectory_rollouter(
    latent_space,
    transition_model,
    motif_transition_matrix
) -> TrajectoryRollouter:
    return TrajectoryRollouter(
        latent_space=latent_space,
        transition_model=transition_model,
        motif_transition_matrix=motif_transition_matrix
    )