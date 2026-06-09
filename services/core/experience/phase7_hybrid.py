"""
Phase 7: Learned Latent Dynamics + Symbolic Reflection

Hybrid architecture:
- Subsymbolic: continuous latent embeddings, learned dynamics
- Symbolic: reflective layer, reasoning about causes

Key innovations:
1. Continuous latent space (not enum causes)
2. Learned transition model P(z_next | z, action)
3. Contrastive temporal learning (what leads to what)
4. Information gain planning (epistemic + pragmatic)
5. Self-model (own competence, uncertainty, limits)
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
import random


class SymbolicCause(Enum):
    """Symbolic layer - interpretable high-level causes"""
    STABLE = "stable"
    SERVICE_INSTABILITY = "service_instability"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    CASCADE_FAILURE = "cascade_failure"
    OSCILLATION = "oscillation"
    CONFIGURATION_DRIFT = "configuration_drift"
    UNKNOWN = "unknown"


@dataclass
class LatentEmbedding:
    """
    Continuous latent representation.

    Not: SERVICE_INSTABILITY (symbolic enum)
    But: [0.182, -0.734, 1.442, ...] (continuous vector)

    This enables:
    - Smooth interpolation between states
    - Novel state composition
    - Emergent structure discovery
    """
    vector: Tuple[float, ...]
    timestamp: str
    source_observations: Tuple[str, ...]

    # Meta
    confidence: float  # How confident in this encoding
    novelty: float  # How novel this state is

    @staticmethod
    def from_observations(
        error_count: int,
        container_count: int,
        resource_load: float,
        memory_mb: int,
        prior_embedding: Optional['LatentEmbedding'] = None
    ) -> 'LatentEmbedding':
        """Create latent embedding from observations"""

        # Normalize features
        norm_error = error_count / 10.0
        norm_container = container_count / 20.0
        norm_load = resource_load / 4.0
        norm_memory = memory_mb / 8000.0

        # Create continuous embedding
        # This is a simple encoding - real systems would use learned encoders
        vector = (
            norm_error * 0.8 + random.uniform(-0.05, 0.05),  # Error dimension
            norm_container * 0.6 + random.uniform(-0.05, 0.05),  # Container dimension
            norm_load * 0.7 + random.uniform(-0.05, 0.05),  # Load dimension
            norm_memory * 0.5 + random.uniform(-0.05, 0.05),  # Memory dimension
            (norm_error + norm_load) / 2 * 0.9,  # Stress dimension
            (norm_container - norm_error) * 0.4,  # Health dimension
        )

        # Compute novelty relative to prior
        novelty = 0.0
        if prior_embedding:
            # Distance from prior
            dist = math.sqrt(sum(
                (a - b) ** 2 for a, b in zip(vector, prior_embedding.vector)
            ))
            novelty = min(1.0, dist / 2.0)

        return LatentEmbedding(
            vector=vector,
            timestamp="",
            source_observations=(
                f"error:{error_count}",
                f"containers:{container_count}",
                f"load:{resource_load:.1f}",
                f"memory:{memory_mb}"
            ),
            confidence=0.7 if novelty < 0.5 else 0.4,
            novelty=novelty
        )

    def distance_to(self, other: 'LatentEmbedding') -> float:
        """Compute distance to another embedding"""
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.vector, other.vector)
        ))

    def interpolate_to(self, other: 'LatentEmbedding', alpha: float) -> 'LatentEmbedding':
        """Interpolate to another embedding"""
        new_vector = tuple(
            a * (1 - alpha) + b * alpha
            for a, b in zip(self.vector, other.vector)
        )

        return LatentEmbedding(
            vector=new_vector,
            timestamp="",
            source_observations=self.source_observations + other.source_observations,
            confidence=self.confidence * (1 - alpha) + other.confidence * alpha,
            novelty=alpha * 0.5
        )

    def to_symbolic(self) -> SymbolicCause:
        """Map latent to symbolic cause for interpretability"""
        error_dim = self.vector[0] if len(self.vector) > 0 else 0
        load_dim = self.vector[2] if len(self.vector) > 2 else 0
        stress_dim = self.vector[4] if len(self.vector) > 4 else 0

        # Heuristic mapping from latent to symbolic
        if error_dim < 0.3 and load_dim < 0.3:
            return SymbolicCause.STABLE
        elif stress_dim > 0.7:
            return SymbolicCause.CASCADE_FAILURE
        elif load_dim > 0.6:
            return SymbolicCause.RESOURCE_EXHAUSTION
        elif error_dim > 0.5:
            return SymbolicCause.SERVICE_INSTABILITY
        else:
            return SymbolicCause.UNKNOWN


@dataclass
class LearnedTransitionModel:
    """
    Learned transition dynamics: P(z_next | z, action)

    Not scripted: if restart: instability -= 0.3
    But learned: statistical patterns from experience

    This enables:
    - Data-driven predictions
    - Unknown dynamics discovery
    - Confidence estimates
    """

    # Observed transitions: (z, action) -> [z_next samples]
    transition_history: Dict[Tuple[str, str], List[LatentEmbedding]] = field(default_factory=dict)

    # Learned parameters for each (aspect, action)
    transition_params: Dict[Tuple[int, str], Dict[str, float]] = field(default_factory=dict)

    def record_transition(
        self,
        z_before: LatentEmbedding,
        action: str,
        z_after: LatentEmbedding
    ):
        """Record observed transition for learning"""
        key = (self._embed_to_key(z_before), action)

        if key not in self.transition_history:
            self.transition_history[key] = []

        self.transition_history[key].append(z_after)

        # Keep only recent history
        if len(self.transition_history[key]) > 50:
            self.transition_history[key] = self.transition_history[key][-50:]

        # Update learned parameters
        self._update_params(key)

    def _embed_to_key(self, z: LatentEmbedding) -> str:
        """Convert embedding to discrete key for lookup"""
        # Quantize to reduce state space
        quantized = tuple(round(v, 1) for v in z.vector)
        return str(quantized)

    def _update_params(self, key):
        """Update learned transition parameters"""
        samples = self.transition_history.get(key, [])
        if len(samples) < 3:
            return

        # Compute empirical mean and variance for each dimension
        for dim in range(len(samples[0].vector)):
            values = [s.vector[dim] for s in samples]
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)

            self.transition_params[(dim, key[1])] = {
                'mean': mean,
                'variance': variance,
                'count': len(samples)
            }

    def predict(
        self,
        z_current: LatentEmbedding,
        action: str
    ) -> Tuple[LatentEmbedding, float]:
        """
        Predict next state given current state and action.

        Returns:
        - predicted next embedding
        - confidence in prediction
        """
        key = (self._embed_to_key(z_current), action)
        samples = self.transition_history.get(key, [])

        if not samples:
            # No experience - return current with low confidence
            return z_current, 0.3

        # Predict using empirical mean
        predicted_vector = []
        confidences = []

        for dim in range(len(z_current.vector)):
            if (dim, action) in self.transition_params:
                params = self.transition_params[(dim, action)]
                predicted_vector.append(params['mean'])
                # Confidence from sample count
                conf = min(1.0, params['count'] / 20.0)
                confidences.append(conf)
            else:
                # Use observed mean
                values = [s.vector[dim] for s in samples]
                predicted_vector.append(sum(values) / len(values))
                confidences.append(0.5)

        predicted = LatentEmbedding(
            vector=tuple(predicted_vector),
            timestamp="",
            source_observations=(),
            confidence=sum(confidences) / len(confidences) if confidences else 0.5,
            novelty=0.0
        )

        return predicted, predicted.confidence

    def get_transition_confidence(self, action: str) -> float:
        """Get overall confidence in transition model for action"""
        counts = [
            self.transition_params.get((dim, action), {}).get('count', 0)
            for dim in range(6)
        ]
        return min(1.0, sum(counts) / max(1, len(counts) * 10))


@dataclass
class SelfModel:
    """
    Self-model: agent's understanding of own capabilities and limits.

    This is what makes the system "aware of itself":
    - Own competence in different situations
    - Own planning quality
    - Own failure modes
    - Own resource limits
    """

    def __init__(self):
        # Competence estimates per situation type
        self.competence: Dict[str, float] = {
            'diagnostic': 0.5,
            'recovery': 0.5,
            'optimization': 0.5,
            'exploration': 0.5
        }

        # Self-awareness
        self.known_limitations: List[str] = []
        self.exploration_value: float = 0.3  # How much we value exploration

        # Meta-cognition
        self.planning_quality: float = 0.6  # How good are our plans
        self.execution_quality: float = 0.7  # How well do we execute

        # Failure modes encountered
        self.failure_modes: Dict[str, int] = {}

    def update_competence(self, situation: str, success: bool):
        """Update competence estimate based on outcome"""
        if situation not in self.competence:
            self.competence[situation] = 0.5

        # Update with exponential moving average
        if success:
            self.competence[situation] = min(1.0, self.competence[situation] * 1.05 + 0.05)
        else:
            self.competence[situation] = max(0.1, self.competence[situation] * 0.9 - 0.05)

    def get_planning_uncertainty(self) -> float:
        """How uncertain are our plans"""
        competence_variance = sum(
            (c - 0.5) ** 2 for c in self.competence.values()
        ) / max(1, len(self.competence))
        return 1.0 - math.sqrt(competence_variance)

    def get_exploration_urgency(self, situation_confidence: float) -> float:
        """
        How urgent is exploration given current confidence?

        High confidence → low exploration urgency
        Low confidence → high exploration urgency
        """
        return max(0.0, 1.0 - situation_confidence)


@dataclass
class InformationGainPlanner:
    """
    Action selection maximizing:
    - Pragmatic value (goal achievement)
    - Epistemic value (uncertainty reduction)

    This is what makes exploration truly "active inference"
    not just random diagnostics.
    """

    def __init__(self, transition_model: LearnedTransitionModel, self_model: SelfModel):
        self.transition_model = transition_model
        self.self_model = self_model

    def compute_expected_information_gain(
        self,
        z_current: LatentEmbedding,
        action: str
    ) -> float:
        """
        Compute expected information gain from action.

        High gain = action would reduce uncertainty about latent state
        """
        # Predict current uncertainty
        current_uncertainty = z_current.novelty

        # Predict next state
        z_next, confidence = self.transition_model.predict(z_current, action)

        # Uncertainty after action
        predicted_uncertainty = z_next.novelty

        # Information gain = reduction in uncertainty
        information_gain = current_uncertainty - predicted_uncertainty

        # Also consider exploration value from self-model
        exploration_bonus = self.self_model.get_exploration_urgency(
            z_current.confidence
        )

        return max(0.0, information_gain) + exploration_bonus * 0.2

    def select_best_action(
        self,
        z_current: LatentEmbedding,
        available_actions: List[str],
        situation: str,
        use_information_gain: bool = True
    ) -> Tuple[str, float, float]:
        """
        Select action balancing pragmatic and epistemic value.

        Returns:
        - best action
        - pragmatic value
        - epistemic value (information gain)
        """
        best_action = available_actions[0]
        best_pragmatic = 0.0
        best_epistemic = 0.0

        for action in available_actions:
            # Pragmatic value: success probability from self-model
            pragmatic = self.self_model.competence.get(situation, 0.5)

            # Epistemic value: information gain
            epistemic = 0.0
            if use_information_gain:
                epistemic = self.compute_expected_information_gain(z_current, action)

            # Combined score
            score = pragmatic + epistemic * 0.5

            if score > best_pragmatic + best_epistemic * 0.5:
                best_action = action
                best_pragmatic = pragmatic
                best_epistemic = epistemic

        return best_action, best_pragmatic, best_epistemic


@dataclass
class ContrastiveTemporalLearner:
    """
    Contrastive learning: what leads to what.

    Learns causal structure by comparing trajectories:
    - What happened after action A vs action B?
    - What preceded failure vs success?
    - What patterns lead to instability?

    This enables discovering hidden causal relationships
    without explicit symbolic rules.
    """

    def __init__(self):
        # Positive pairs: states that led to good outcomes
        self.good_trajectories: List[List[LatentEmbedding]] = []

        # Negative pairs: states that led to bad outcomes
        self.bad_trajectories: List[List[LatentEmbedding]] = []

        # Contrastive pairs: before -> after
        self.contrastive_pairs: List[Tuple[LatentEmbedding, LatentEmbedding]] = []

        # Learned causal patterns
        self.causal_patterns: Dict[str, float] = {}

    def add_trajectory(
        self,
        trajectory: List[LatentEmbedding],
        outcome: str  # "success", "failure", "degradation"
    ):
        """Add trajectory for contrastive learning"""
        if outcome == "success":
            self.good_trajectories.append(trajectory)
        else:
            self.bad_trajectories.append(trajectory)

        # Contrastive pairs from trajectory
        for i in range(len(trajectory) - 1):
            self.contrastive_pairs.append((trajectory[i], trajectory[i + 1]))

    def learn_contrastive_patterns(self):
        """Learn patterns from contrastive pairs"""

        if len(self.contrastive_pairs) < 10:
            return

        # Analyze transitions that lead to instability
        instability_leads = []

        for before, after in self.contrastive_pairs:
            # Check if transition increased error dimension
            if len(before.vector) >= 1 and len(after.vector) >= 1:
                error_delta = after.vector[0] - before.vector[0]
                if error_delta > 0.1:
                    instability_leads.append(before.vector)

        # Learn that certain latent states precede instability
        if instability_leads:
            avg_before = tuple(
                sum(v[i] for v in instability_leads) / len(instability_leads)
                for i in range(len(instability_leads[0]))
            )
            self.causal_patterns['instability_predecessor'] = sum(avg_before) / len(avg_before)

    def get_trajectory_contrast(
        self,
        trajectory_a: List[LatentEmbedding],
        trajectory_b: List[LatentEmbedding]
    ) -> Dict[str, float]:
        """
        Compare two trajectories to extract causal patterns.

        Returns what differs between successful and unsuccessful paths.
        """
        if not trajectory_a or not trajectory_b:
            return {}

        # Compute average states
        avg_a = self._average_embedding(trajectory_a)
        avg_b = self._average_embedding(trajectory_b)

        # Compute contrast
        contrast = {}
        for i in range(len(avg_a.vector)):
            contrast[f"dim_{i}_diff"] = avg_a.vector[i] - avg_b.vector[i]

        return contrast

    def _average_embedding(self, trajectory: List[LatentEmbedding]) -> LatentEmbedding:
        """Compute average embedding of trajectory"""
        if not trajectory:
            return LatentEmbedding(vector=(), timestamp="", source_observations=(), confidence=0.0, novelty=0.0)

        avg_vector = tuple(
            sum(t.vector[i] for t in trajectory) / len(trajectory)
            for i in range(len(trajectory[0].vector))
        )

        return LatentEmbedding(
            vector=avg_vector,
            timestamp="",
            source_observations=(),
            confidence=0.5,
            novelty=0.0
        )


@dataclass
class Phase7HybridSystem:
    """
    Phase 7: Learned Latent Dynamics + Symbolic Reflection

    Hybrid architecture:
    - Subsymbolic: continuous embeddings, learned transitions
    - Symbolic: interpretable causes, reflection

    Key innovations:
    1. Continuous latent space (not fixed enum)
    2. Learned transition model (not scripted)
    3. Information gain planning (true active inference)
    4. Self-model (own capabilities)
    5. Contrastive temporal learning
    """

    def __init__(self):
        # Subsymbolic components
        self.current_embedding: Optional[LatentEmbedding] = None
        self.transition_model = LearnedTransitionModel()
        self.contrastive_learner = ContrastiveTemporalLearner()

        # Symbolic reflection
        self.symbolic_belief: SymbolicCause = SymbolicCause.STABLE

        # Self-model
        self.self_model = SelfModel()

        # Planning
        self.planner = InformationGainPlanner(self.transition_model, self.self_model)

        # History
        self.trajectory_history: List[List[LatentEmbedding]] = []
        self.current_trajectory: List[LatentEmbedding] = []

    def update_from_observation(
        self,
        error_count: int,
        container_count: int,
        resource_load: float,
        memory_mb: int
    ) -> LatentEmbedding:
        """Update latent embedding from observations"""

        new_embedding = LatentEmbedding.from_observations(
            error_count=error_count,
            container_count=container_count,
            resource_load=resource_load,
            memory_mb=memory_mb,
            prior_embedding=self.current_embedding
        )

        # Update current
        self.current_embedding = new_embedding

        # Add to trajectory
        self.current_trajectory.append(new_embedding)

        # Update symbolic reflection
        self.symbolic_belief = new_embedding.to_symbolic()

        return new_embedding

    def select_action(
        self,
        available_actions: List[str],
        situation: str = "diagnostic"
    ) -> Tuple[str, float, float]:
        """Select action using information gain planning"""

        if not self.current_embedding:
            return available_actions[0] if available_actions else "check", 0.0, 0.0

        return self.planner.select_best_action(
            z_current=self.current_embedding,
            available_actions=available_actions,
            situation=situation
        )

    def record_outcome(
        self,
        action: str,
        success: bool,
        new_embedding: LatentEmbedding
    ):
        """Record action outcome for learning"""

        # Learn transition
        if self.current_embedding:
            self.transition_model.record_transition(
                self.current_embedding, action, new_embedding
            )

        # Update self-model
        situation = "recovery" if "restart" in action else "diagnostic"
        self.self_model.update_competence(situation, success)

        # Update contrastive learner
        self.contrastive_learner.add_trajectory(
            self.current_trajectory,
            outcome="success" if success else "failure"
        )

        # Reset trajectory for next episode
        self.trajectory_history.append(self.current_trajectory)
        self.current_trajectory = []

    def get_semantic_description(self) -> str:
        """Get human-readable description of current state"""
        if not self.current_embedding:
            return "Unknown state"

        cause = self.symbolic_belief.value

        if cause == "stable":
            return f"System stable (embedding norm: {self._embedding_norm():.2f})"
        elif cause == "service_instability":
            return f"Service instability (confidence: {self.current_embedding.confidence:.1%})"
        elif cause == "resource_exhaustion":
            return f"Resource pressure (novelty: {self.current_embedding.novelty:.1%})"
        else:
            return f"Unknown state (novelty: {self.current_embedding.novelty:.1%})"

    def _embedding_norm(self) -> float:
        """Compute norm of current embedding"""
        if not self.current_embedding:
            return 0.0
        return math.sqrt(sum(v ** 2 for v in self.current_embedding.vector))

    def get_self_awareness(self) -> Dict[str, Any]:
        """Get self-model awareness"""
        return {
            'competence': self.self_model.competence.copy(),
            'planning_uncertainty': self.self_model.get_planning_uncertainty(),
            'exploration_urgency': self.self_model.get_exploration_urgency(
                self.current_embedding.confidence if self.current_embedding else 0.5
            ),
            'known_limitations': self.self_model.known_limitations
        }