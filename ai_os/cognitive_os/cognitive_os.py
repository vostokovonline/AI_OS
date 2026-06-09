"""
Cognitive OS - Main Facade for All Cognitive Components

Integrates:
1. Emotional Layer - manages emotional states and influence
2. Growth Layer - meta-cognition, bias detection, self-improvement
3. World Model - environment simulation and state tracking
4. Strategy Evolution - self-improvement through strategy mutation
5. Self-Narrative - identity continuity and motivation

Usage:
    cognitive_os = CognitiveOS()
    
    # Process a request with full cognitive support
    result = await cognitive_os.process(
        user_id="user123",
        context={"task": "write_code", "complexity": 0.6}
    )
    
    # Get cognitive state
    state = cognitive_os.get_state()
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .growth_layer.growth import GrowthLayer
from .growth_layer.strategy_evolution import StrategyEvolution
from .growth_layer.self_narrative import SelfNarrative
from .world_model.model import WorldModel, Entity

logger = logging.getLogger(__name__)


@dataclass
class CognitiveRequest:
    """A request to be processed by the cognitive system"""
    user_id: str
    context: Dict[str, Any]
    action: str = "default"
    emotional_signals: Optional[Dict] = None


@dataclass
class CognitiveResponse:
    """Response from the cognitive system"""
    emotional_influence: Dict[str, Any]
    strategy_recommendations: List[Dict[str, Any]]
    bias_warnings: List[str]
    world_state: Dict[str, Any]
    narrative_context: str
    reasoning_trace: List[str]


class CognitiveOS:
    """
    Cognitive OS - orchestrates all cognitive components.
    
    This is the main entry point for integrating cognitive capabilities
    into the agent system.
    """
    
    def __init__(self):
        self.emotional_layer = None  # Lazy import
        self.growth = GrowthLayer()
        self.strategy_evolution = StrategyEvolution()
        self.self_narrative = SelfNarrative()
        self.world_model = WorldModel()
        
        self._initialized = False
        logger.info("cognitive_os_created")
    
    async def initialize(self) -> None:
        """Initialize lazy components"""
        if self._initialized:
            return
        
        try:
            from services.core.emotional_layer import emotional_layer
            self.emotional_layer = emotional_layer
            logger.info("emotional_layer_loaded")
        except Exception as e:
            logger.warning("emotional_layer_not_available", error=str(e))
        
        self._initialized = True
        logger.info("cognitive_os_initialized")
    
    async def process(self, request: CognitiveRequest) -> CognitiveResponse:
        """
        Process a request with full cognitive support.
        
        This is the main entry point for cognitive processing.
        """
        await self.initialize()
        
        reasoning_trace = []
        
        # 1. Emotional influence
        emotional_influence = await self._get_emotional_influence(
            request.user_id,
            request.emotional_signals,
            request.context
        )
        reasoning_trace.append(f"Emotional state: arousal={emotional_influence.get('arousal', 0.5):.2f}")
        
        # 2. World model update
        if request.action != "default":
            self.world_model.record_action(
                actor_id=request.user_id,
                action=request.action,
                outcome="in_progress"
            )
        
        reasoning_trace.append(f"World model updated for action: {request.action}")
        
        # 3. Strategy recommendations
        recommendations = self.strategy_evolution.generate_recommendation(request.context)
        reasoning_trace.append(f"Generated {len(recommendations)} strategy recommendations")
        
        # 4. Bias detection
        if "thought" in request.context:
            analysis = await self.growth.analyze_action(
                context=request.context.get("task", "general"),
                thought=request.context.get("thought", ""),
                action=request.context.get("action", ""),
                outcome=request.context.get("outcome", "unknown")
            )
            bias_warnings = [p for p in analysis.get("detected_patterns", [])]
        else:
            bias_warnings = []
        
        reasoning_trace.append(f"Detected {len(bias_warnings)} potential biases")
        
        # 5. Narrative context
        narrative = self.self_narrative.get_current_narrative()
        
        # 6. Update world model with outcome
        if "outcome" in request.context:
            self.world_model.record_action(
                actor_id=request.user_id,
                action=request.action,
                outcome=request.context.get("outcome", "unknown")
            )
        
        return CognitiveResponse(
            emotional_influence=emotional_influence,
            strategy_recommendations=recommendations,
            bias_warnings=bias_warnings,
            world_state=self.world_model.get_world_state(),
            narrative_context=narrative,
            reasoning_trace=reasoning_trace
        )
    
    async def _get_emotional_influence(
        self,
        user_id: str,
        signals: Optional[Dict],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get emotional influence with fallback"""
        if not self.emotional_layer:
            return {
                "arousal": 0.5,
                "valence": 0.0,
                "focus": 0.6,
                "confidence": 0.5,
                "pace": "normal",
                "exploration": "balanced"
            }
        
        try:
            from services.core.schemas import EmotionalSignals
            if signals:
                emotional_signals = EmotionalSignals(**signals)
                influence = await self.emotional_layer.get_influence(user_id, emotional_signals)
                return {
                    "arousal": influence.arousal,
                    "valence": influence.valence,
                    "focus": influence.focus,
                    "confidence": influence.confidence,
                    "pace": influence.pace,
                    "exploration": influence.exploration
                }
        except Exception as e:
            logger.warning("emotional_influence_error", error=str(e))
        
        return {
            "arousal": 0.5,
            "valence": 0.0,
            "focus": 0.6,
            "confidence": 0.5,
            "pace": "normal",
            "exploration": "balanced"
        }
    
    def record_outcome(
        self,
        strategy_id: str,
        outcome: str,
        score: float
    ) -> None:
        """Record strategy outcome for learning"""
        self.strategy_evolution.update_performance(strategy_id, score)
        
        if outcome == "success":
            self.self_narrative.integrate_experience(
                experience=f"Strategy {strategy_id} succeeded with score {score:.2f}",
                outcome="success",
                emotional_impact=score
            )
        else:
            self.self_narrative.integrate_experience(
                experience=f"Strategy {strategy_id} failed with score {score:.2f}",
                outcome="failure",
                emotional_impact=1 - score
            )
        
        self.growth.update_meta_state(
            attention=0.7,
            reflection=0.6
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get comprehensive cognitive state"""
        return {
            "emotional": {
                "available": self.emotional_layer is not None,
            },
            "growth": self.growth.get_bias_report(),
            "strategy": {
                "count": len(self.strategy_evolution.strategies),
                "generation": self.strategy_evolution.generation,
            },
            "identity": self.self_narrative.get_identity_report(),
            "world": self.world_model.get_world_state(),
        }
    
    def evolve(self) -> List[str]:
        """Run evolution cycle on strategies"""
        new_strategies = self.strategy_evolution.evolve_generation()
        return [s.name for s in new_strategies]


# Singleton instance
cognitive_os = CognitiveOS()