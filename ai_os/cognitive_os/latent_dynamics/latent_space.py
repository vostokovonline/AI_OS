"""
Latent State Space - Embedding-based Cognitive State Representation

Превращает symbolic state в latent vector representation:

Symbolic State (confidence, stress, arousal...)
    ↓
Embedding Projection (learned)
    ↓
Latent State (dense vector)
    ↓
Causal Dynamics (learned transitions)

Это заменяет:
- StateSnapshot (только symbolic)
- UnifiedState.to_vector() (raw concatenation)

Теперь state имеет:
- semantic meaning (через embedding)
- distance metrics (через latent space)
- learned representations (через training)
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class SymbolicState:
    """Символическое представление состояния (текущее)"""
    confidence: float = 0.5
    stress_level: float = 0.0
    action_readiness: float = 0.5
    arousal: float = 0.5
    valence: float = 0.0
    focus: float = 0.5
    bias_awareness: float = 0.5
    reflection_depth: float = 0.5
    exploration_tendency: float = 0.5
    task_complexity: float = 0.5
    task_urgency: float = 0.5
    task_novelty: float = 0.5
    world_capability: float = 0.5
    identity_coherence: float = 0.5
    
    def to_vector(self) -> List[float]:
        """Конвертация в raw vector (без embedding)"""
        return [
            self.confidence,
            self.stress_level,
            self.action_readiness,
            self.arousal,
            (self.valence + 1) / 2,  # Normalize to [0, 1]
            self.focus,
            self.bias_awareness,
            self.reflection_depth,
            self.exploration_tendency,
            self.task_complexity,
            self.task_urgency,
            self.task_novelty,
            self.world_capability,
            self.identity_coherence,
        ]
    
    def to_dict(self) -> Dict:
        return {
            "confidence": self.confidence,
            "stress_level": self.stress_level,
            "action_readiness": self.action_readiness,
            "arousal": self.arousal,
            "valence": self.valence,
            "focus": self.focus,
            "bias_awareness": self.bias_awareness,
            "reflection_depth": self.reflection_depth,
            "exploration_tendency": self.exploration_tendency,
            "task_complexity": self.task_complexity,
            "task_urgency": self.task_urgency,
            "task_novelty": self.task_novelty,
            "world_capability": self.world_capability,
            "identity_coherence": self.identity_coherence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SymbolicState":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
    
    @classmethod
    def from_unified_state(cls, state) -> "SymbolicState":
        return cls(
            confidence=state.confidence,
            stress_level=state.stress_level,
            action_readiness=state.action_readiness,
            arousal=state.arousal,
            valence=state.valence,
            focus=state.focus,
            bias_awareness=state.bias_awareness,
            reflection_depth=state.reflection_depth,
            exploration_tendency=state.exploration_tendency,
            task_complexity=state.task_complexity,
            task_urgency=state.task_urgency,
            task_novelty=state.task_novelty,
            world_capability=state.world_capability_score,
            identity_coherence=state.identity_coherence,
        )


@dataclass
class LatentState:
    """
    Latent representation состояния.
    
    Это dense vector в latent space, который:
    - Имеет semantic meaning (через обучение)
    - Поддерживает distance metrics
    - Может быть декодирован обратно в symbolic
    """
    id: str
    timestamp: datetime
    
    symbolic: SymbolicState
    vector: List[float]  # Dense latent representation
    dimension: int
    
    # Causal metadata
    latent_causes: List[str] = field(default_factory=list)  # Inferred causes
    prediction_error: float = 0.0  # How surprising this state was
    novelty_score: float = 0.0  # How novel compared to history
    
    # Quality metrics
    embedding_confidence: float = 1.0
    reconstruction_error: float = 0.0
    
    def __post_init__(self):
        self.dimension = len(self.vector)
    
    def distance_to(self, other: "LatentState") -> float:
        """Compute Euclidean distance in latent space"""
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.vector, other.vector)
        ))
    
    def cosine_similarity(self, other: "LatentState") -> float:
        """Compute cosine similarity"""
        dot = sum(a * b for a, b in zip(self.vector, other.vector))
        norm1 = math.sqrt(sum(a ** 2 for a in self.vector))
        norm2 = math.sqrt(sum(b ** 2 for b in other.vector))
        return dot / (norm1 * norm2 + 1e-10)
    
    def interpolate_to(self, other: "LatentState", t: float) -> List[float]:
        """Interpolate between two states (for counterfactual analysis)"""
        return [a + t * (b - a) for a, b in zip(self.vector, other.vector)]
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "symbolic": self.symbolic.to_dict(),
            "dimension": self.dimension,
            "latent_causes": self.latent_causes,
            "prediction_error": self.prediction_error,
            "novelty_score": self.novelty_score,
            "embedding_confidence": self.embedding_confidence,
        }


class LatentStateEncoder:
    """
    Encoder: Symbolic State → Latent Vector
    
    Сейчас rule-based (можно заменить на learned).
    
    Future: Neural network encoder trained on trajectories.
    """
    
    DIMENSION = 16
    
    # Feature groups for semantic encoding
    CAPABILITY_FEATURES = ["confidence", "action_readiness", "world_capability"]
    AFFECT_FEATURES = ["arousal", "valence", "focus"]
    METACOGNITION_FEATURES = ["bias_awareness", "reflection_depth"]
    CONTEXT_FEATURES = ["task_complexity", "task_urgency", "task_novelty"]
    STRESS_FEATURES = ["stress_level", "exploration_tendency"]
    IDENTITY_FEATURES = ["identity_coherence"]
    
    def __init__(self):
        self.weights = self._initialize_weights()
        logger.info("latent_encoder_initialized", dimension=self.DIMENSION)
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Инициализация весов для semantic encoding"""
        return {
            "confidence": 1.0,
            "stress_level": 0.9,
            "action_readiness": 0.85,
            "arousal": 0.7,
            "valence": 0.7,
            "focus": 0.6,
            "bias_awareness": 0.5,
            "reflection_depth": 0.5,
            "exploration_tendency": 0.6,
            "task_complexity": 0.7,
            "task_urgency": 0.6,
            "task_novelty": 0.5,
            "world_capability": 0.6,
            "identity_coherence": 0.5,
        }
    
    def encode(self, symbolic: SymbolicState) -> List[float]:
        """
        Encode symbolic state to latent vector.
        
        Uses feature groups for semantic structure:
        - Capability block (high-level ability)
        - Affect block (emotional state)
        - Metacognition block (self-awareness)
        - Context block (task requirements)
        - Stress block (system pressure)
        """
        raw = symbolic.to_vector()
        weights = [self.weights.get(k, 0.5) for k in symbolic.to_dict().keys()]
        
        # Weighted encoding with noise for exploration
        vector = [
            v * w + (0.02 if i % 3 == 0 else 0)  # Slight exploration noise
            for i, (v, w) in enumerate(zip(raw, weights))
        ]
        
        # Add derived features (non-linear combinations)
        vector.append(self._capability_aggregate(symbolic))
        vector.append(self._affect_aggregate(symbolic))
        vector.append(self._pressure_aggregate(symbolic))
        vector.append(self._novelty_aggregate(symbolic))
        
        # Normalize to [0, 1]
        if vector:
            min_v = min(vector)
            max_v = max(vector)
            if max_v > min_v:
                vector = [(v - min_v) / (max_v - min_v) for v in vector]
        
        return vector
    
    def _capability_aggregate(self, state: SymbolicState) -> float:
        """Aggregate capability features"""
        return (
            state.confidence * 0.4 +
            state.action_readiness * 0.3 +
            state.world_capability * 0.3
        )
    
    def _affect_aggregate(self, state: SymbolicState) -> float:
        """Aggregate affect features"""
        return (
            state.arousal * 0.4 +
            ((state.valence + 1) / 2) * 0.3 +
            state.focus * 0.3
        )
    
    def _pressure_aggregate(self, state: SymbolicState) -> float:
        """Aggregate system pressure"""
        return (
            state.stress_level * 0.5 +
            state.task_urgency * 0.3 +
            state.task_complexity * 0.2
        )
    
    def _novelty_aggregate(self, state: SymbolicState) -> float:
        """Aggregate novelty signals"""
        return (
            state.task_novelty * 0.5 +
            (1 - state.confidence) * 0.3 +
            state.reflection_depth * 0.2
        )
    
    def decode(self, vector: List[float], original: SymbolicState) -> SymbolicState:
        """
        Decode latent vector back to symbolic (approximate).
        
        For now, returns original. Future: learned decoder.
        """
        return original


class LatentStateSpace:
    """
    Latent State Space - manages evolution of cognitive states.
    
    Tracks:
    - History of latent states
    - Transition dynamics
    - Novelty detection
    - Causal inference
    """
    
    def __init__(self, dimension: int = 16):
        self.encoder = LatentStateEncoder()
        self.dimension = dimension
        self.history: List[LatentState] = []
        self.max_history = 1000
        
        # Transition model (for counterfactual analysis)
        self.transition_model: Optional["TransitionModel"] = None
        
        logger.info("latent_state_space_initialized", dimension=dimension)
    
    def add_state(
        self,
        symbolic: SymbolicState,
        latent_causes: Optional[List[str]] = None,
        prediction_error: float = 0.0
    ) -> LatentState:
        """Add new state to the space"""
        from uuid import uuid4
        
        vector = self.encoder.encode(symbolic)
        
        novelty = self._compute_novelty(vector)
        
        state = LatentState(
            id=str(uuid4()),
            timestamp=datetime.utcnow(),
            symbolic=symbolic,
            vector=vector,
            dimension=self.dimension,
            latent_causes=latent_causes or [],
            prediction_error=prediction_error,
            novelty_score=novelty
        )
        
        self.history.append(state)
        
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        logger.debug(
            "state_added_to_space",
            state_id=state.id,
            novelty=novelty,
            prediction_error=prediction_error
        )
        
        return state
    
    def _compute_novelty(self, vector: List[float]) -> float:
        """Compute novelty compared to recent history"""
        if len(self.history) < 5:
            return 0.5
        
        recent = self.history[-20:]
        distances = []
        
        for past in recent:
            dist = math.sqrt(sum(
                (v - p) ** 2 for v, p in zip(vector, past.vector)
            ))
            distances.append(dist)
        
        avg_distance = sum(distances) / len(distances)
        novelty = min(1.0, avg_distance / 2.0)
        
        return novelty
    
    def get_state(self, state_id: str) -> Optional[LatentState]:
        """Get state by ID"""
        for state in reversed(self.history):
            if state.id == state_id:
                return state
        return None
    
    def get_recent(self, count: int = 10) -> List[LatentState]:
        """Get recent states"""
        return self.history[-count:]
    
    def get_transition(self, from_id: str, to_id: str) -> Optional[Dict]:
        """Get transition between two states"""
        from_state = self.get_state(from_id)
        to_state = self.get_state(to_id)
        
        if not from_state or not to_state:
            return None
        
        return {
            "from": from_id,
            "to": to_id,
            "distance": from_state.distance_to(to_state),
            "similarity": from_state.cosine_similarity(to_state),
            "delta": [t - f for f, t in zip(from_state.vector, to_state.vector)]
        }
    
    def find_similar_states(
        self,
        vector: List[float],
        threshold: float = 0.3,
        limit: int = 10
    ) -> List[Tuple[LatentState, float]]:
        """Find states similar to given vector"""
        results = []
        
        for state in self.history:
            dist = math.sqrt(sum(
                (v - s) ** 2 for v, s in zip(vector, state.vector)
            ))
            if dist < threshold:
                results.append((state, dist))
        
        results.sort(key=lambda x: x[1])
        return results[:limit]
    
    def get_evolution_trajectory(self) -> List[Dict]:
        """Get full evolution trajectory"""
        trajectory = []
        
        for i in range(len(self.history)):
            if i == 0:
                continue
            
            prev = self.history[i-1]
            curr = self.history[i]
            
            trajectory.append({
                "from_id": prev.id,
                "to_id": curr.id,
                "timestamp": curr.timestamp.isoformat(),
                "distance": prev.distance_to(curr),
                "novelty": curr.novelty_score,
                "prediction_error": curr.prediction_error,
                "causes": curr.latent_causes,
                "delta": [t - p for p, t in zip(prev.vector, curr.vector)]
            })
        
        return trajectory


class TransitionModel:
    """
    Learned transition model: state_t → state_t+1
    
    Это основа для counterfactual reasoning:
    - Что если мы выбрали другое действие?
    - Какие latent factors повлияли на исход?
    
    Сейчас: rule-based (can be replaced with learned).
    """
    
    def __init__(self):
        self.action_effects: Dict[str, List[float]] = {
            "execute": [0.1, -0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05],
            "decompose": [-0.05, 0.05, -0.1, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "explore": [0.05, 0.0, 0.05, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0],
            "wait": [0.0, -0.1, -0.05, -0.05, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "reconsider": [0.0, 0.1, -0.05, 0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "retry": [0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "abort": [-0.1, 0.15, -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, -0.05, -0.05],
        }
        
        logger.info("transition_model_initialized")
    
    def predict_next_state(
        self,
        current: LatentState,
        action: str
    ) -> List[float]:
        """Predict next state given current state and action"""
        effect = self.action_effects.get(action, [0.0] * 16)
        
        predicted = [
            max(0.0, min(1.0, v + e))
            for v, e in zip(current.vector, effect)
        ]
        
        return predicted
    
    def compute_prediction_error(
        self,
        predicted: List[float],
        actual: List[float]
    ) -> float:
        """Compute prediction error (RMSE)"""
        mse = sum((p - a) ** 2 for p, a in zip(predicted, actual)) / len(predicted)
        return math.sqrt(mse)
    
    def counterfactual(
        self,
        current: LatentState,
        actual_action: str,
        alternative_action: str
    ) -> Dict:
        """Compute counterfactual: what if we chose alternative_action?"""
        predicted_actual = self.predict_next_state(current, actual_action)
        predicted_alternative = self.predict_next_state(current, alternative_action)
        
        delta = [a - p for p, a in zip(predicted_actual, predicted_alternative)]
        
        return {
            "current_state": current.id,
            "actual_action": actual_action,
            "alternative_action": alternative_action,
            "predicted_actual": predicted_actual,
            "predicted_alternative": predicted_alternative,
            "difference": delta,
            "impact_magnitude": math.sqrt(sum(d ** 2 for d in delta))
        }


def create_latent_space(dimension: int = 16) -> LatentStateSpace:
    """Factory function to create latent state space"""
    return LatentStateSpace(dimension=dimension)