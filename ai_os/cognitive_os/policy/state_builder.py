"""
State Builder - Unified Latent State Representation

Combines all cognitive signals into a single state vector for policy.

Sources:
- World Model: entities, relationships, action history
- Self-Narrative: identity, coherence, roles
- Emotional Influence: arousal, valence, focus, confidence
- Growth Layer: bias patterns, meta-cognition state
- Strategy Evolution: top strategies, performance

Output:
- UnifiedState with all signals in normalized form
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class UnifiedState:
    """
    Unified latent state representation.
    
    This is the single source of truth for the policy layer.
    All cognitive components feed into this structure.
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # World Model signals (objective)
    world_entities_count: int = 0
    world_recent_outcome: str = "unknown"
    world_capability_score: float = 0.5
    
    # Self-Narrative signals (identity)
    identity_coherence: float = 0.5
    identity_emotion: str = "neutral"
    identity_roles: List[str] = field(default_factory=list)
    
    # Emotional signals (affect)
    arousal: float = 0.5
    valence: float = 0.0
    focus: float = 0.5
    confidence: float = 0.5
    
    # Growth signals (meta-cognition)
    bias_count: int = 0
    bias_awareness: float = 0.5
    reflection_depth: float = 0.5
    
    # Strategy signals (policy)
    top_strategy_id: Optional[str] = None
    top_strategy_name: str = "default"
    top_strategy_score: float = 0.5
    
    # Context signals
    task_complexity: float = 0.5
    task_urgency: float = 0.5
    task_novelty: float = 0.5
    
    # Derived signals
    stress_level: float = 0.0
    exploration_tendency: float = 0.5
    action_readiness: float = 0.5
    
    def to_vector(self) -> List[float]:
        """
        Convert to numerical vector for potential ML usage.
        
        Order: [world, identity, emotion, growth, strategy, context, derived]
        """
        return [
            self.world_entities_count / 100,
            self._str_to_val(self.world_recent_outcome),
            self.world_capability_score,
            self.identity_coherence,
            self._emotion_to_val(self.identity_emotion),
            len(self.identity_roles) / 10,
            self.arousal,
            (self.valence + 1) / 2,
            self.focus,
            self.confidence,
            min(self.bias_count / 10, 1.0),
            self.bias_awareness,
            self.reflection_depth,
            self.top_strategy_score,
            self.task_complexity,
            self.task_urgency,
            self.task_novelty,
            self.stress_level,
            self.exploration_tendency,
            self.action_readiness,
        ]
    
    def _str_to_val(self, s: str) -> float:
        return {"success": 1.0, "partial": 0.5, "failure": 0.0, "unknown": 0.5}.get(s, 0.5)
    
    def _emotion_to_val(self, e: str) -> float:
        return {"positive": 1.0, "neutral": 0.5, "negative": 0.0}.get(e, 0.5)


class StateBuilder:
    """
    Builds unified state from all cognitive components.
    
    This is the "glue" that connects:
    - World Model
    - Self-Narrative
    - Emotional Layer
    - Growth Layer
    - Strategy Evolution
    """
    
    def __init__(self, cognitive_os):
        self.cognitive_os = cognitive_os
        self.last_state: Optional[UnifiedState] = None
        self.state_history: List[UnifiedState] = []
        logger.info("state_builder_initialized")
    
    async def build_state(
        self,
        context: Optional[Dict[str, Any]] = None
    ) -> UnifiedState:
        """
        Build unified state from all cognitive components.
        
        Args:
            context: Optional context overrides (task_complexity, etc.)
        
        Returns:
            UnifiedState with all signals combined
        """
        state = UnifiedState()
        
        # 1. World Model signals
        try:
            world_state = self.cognitive_os.world_model.get_world_state()
            state.world_entities_count = world_state.get("entities_count", 0)
            state.world_capability_score = self._estimate_capability(world_state)
            
            recent = world_state.get("recent_actions", [])
            if recent:
                outcomes = [a.get("outcome", "unknown") for a in recent[-3:]]
                state.world_recent_outcome = max(set(outcomes), key=outcomes.count) if outcomes else "unknown"
        except Exception as e:
            logger.warning("world_state_build_error", error=str(e))
        
        # 2. Self-Narrative signals
        try:
            identity = self.cognitive_os.self_narrative.get_identity_report()
            state.identity_coherence = identity.get("coherence_score", 0.5)
            state.identity_emotion = identity.get("dominant_emotion", "neutral")
            state.identity_roles = identity.get("roles", [])
        except Exception as e:
            logger.warning("identity_build_error", error=str(e))
        
        # 3. Emotional signals
        try:
            emotion = self.cognitive_os.get_state().get("emotional", {})
            if not emotion.get("available"):
                state.arousal = 0.5
                state.valence = 0.0
                state.focus = 0.5
                state.confidence = 0.5
            else:
                state.arousal = 0.5
                state.valence = 0.0
        except Exception as e:
            logger.warning("emotion_build_error", error=str(e))
        
        # 4. Growth signals
        try:
            bias_report = self.cognitive_os.growth.get_bias_report()
            state.bias_count = len(bias_report.get("detected_patterns", []))
            state.bias_awareness = bias_report.get("bias_awareness", 0.5)
            state.reflection_depth = self._calc_reflection_depth(bias_report)
        except Exception as e:
            logger.warning("growth_build_error", error=str(e))
        
        # 5. Strategy signals
        try:
            top_strategies = self.cognitive_os.strategy_evolution.select_best(1)
            if top_strategies:
                state.top_strategy_id = top_strategies[0].id
                state.top_strategy_name = top_strategies[0].name
                state.top_strategy_score = top_strategies[0].success_rate
        except Exception as e:
            logger.warning("strategy_build_error", error=str(e))
        
        # 6. Context overrides
        if context:
            state.task_complexity = context.get("complexity", 0.5)
            state.task_urgency = context.get("urgency", 0.5)
            state.task_novelty = context.get("novelty", 0.5)
        
        # 7. Derive stress and readiness
        state.stress_level = self._calc_stress(state)
        state.exploration_tendency = self._calc_exploration(state)
        state.action_readiness = self._calc_readiness(state)
        
        self.last_state = state
        self.state_history.append(state)
        
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        logger.debug(
            "state_built",
            outcome=state.world_recent_outcome,
            coherence=state.identity_coherence,
            stress=state.stress_level
        )
        
        return state
    
    def _estimate_capability(self, world_state: Dict) -> float:
        """Estimate system capability from world state"""
        actions = world_state.get("actions_count", 0)
        entities = world_state.get("entities_count", 0)
        
        if actions < 10:
            return 0.3
        elif actions < 50:
            return 0.5
        elif actions < 200:
            return 0.7
        else:
            return 0.8 + min(0.2, entities / 1000)
    
    def _calc_reflection_depth(self, bias_report: Dict) -> float:
        """Calculate reflection depth from bias report"""
        patterns = bias_report.get("detected_patterns", [])
        if not patterns:
            return 0.3
        
        recent_learning = bias_report.get("recent_learning", [])
        return min(0.9, 0.3 + len(recent_learning) * 0.2)
    
    def _calc_stress(self, state: UnifiedState) -> float:
        """Calculate stress level from signals"""
        stress = 0.0
        
        if state.world_recent_outcome == "failure":
            stress += 0.3
        
        stress += (1 - state.confidence) * 0.2
        stress += abs(state.valence) * 0.1 if state.valence < 0 else 0
        stress += state.arousal * 0.2
        stress += state.task_urgency * 0.2
        
        return min(1.0, stress)
    
    def _calc_exploration(self, state: UnifiedState) -> float:
        """Calculate exploration tendency"""
        base = 0.5
        
        if state.identity_emotion == "positive":
            base += 0.2
        elif state.identity_emotion == "negative":
            base -= 0.2
        
        base += state.confidence * 0.2
        base -= state.stress_level * 0.3
        
        if state.task_novelty > 0.7:
            base += 0.2
        
        return max(0.1, min(0.9, base))
    
    def _calc_readiness(self, state: UnifiedState) -> float:
        """Calculate action readiness"""
        readiness = 0.5
        
        readiness += state.focus * 0.2
        readiness += state.confidence * 0.2
        readiness -= state.stress_level * 0.3
        
        if state.world_recent_outcome == "success":
            readiness += 0.1
        
        return max(0.1, min(0.9, readiness))