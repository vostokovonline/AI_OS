"""Growth Layer - Meta-cognition, bias detection, self-improvement"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

import logging
logger = logging.getLogger(__name__)


@dataclass
class CognitivePattern:
    """Detected cognitive pattern (bias, habit, tendency)"""
    id: str = field(default_factory=lambda: str(uuid4()))
    pattern_type: str = ""  # confirmation_bias, anchoring, overconfidence, etc.
    description: str = ""
    frequency: int = 0
    accuracy: float = 0  # How often this pattern leads to good outcomes
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    triggered_by: List[str] = field(default_factory=list)  # Context types that trigger this


@dataclass
class SelfReflection:
    """Self-reflection entry for continuous improvement"""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: str  # What triggered the reflection
    thought: str  # What I was thinking
    action: str  # What action I took
    outcome: str  # What happened
    learning: str  # What I learned
    improvement: str  # What I will do differently


@dataclass
class MetaCognitionState:
    """Current meta-cognitive state"""
    attention_depth: float = 0.5  # 0-1, how focused
    reflection_depth: float = 0.5  # 0-1, how much self-reflection
    bias_awareness: float = 0.5  # 0-1, how aware of biases
    improvement_rate: float = 0.0  # Improvement per session
    sessions_count: int = 0


class GrowthLayer:
    """
    Growth Layer - handles meta-cognition, bias detection, self-improvement.
    
    This is the "learning how to learn" layer.
    """
    
    def __init__(self):
        self.patterns: Dict[str, CognitivePattern] = {}
        self.reflections: List[SelfReflection] = []
        self.state = MetaCognitionState()
        self._load_builtin_patterns()
        logger.info("growth_layer_initialized")
    
    def _load_builtin_patterns(self) -> None:
        """Load known cognitive biases for detection"""
        builtin = [
            CognitivePattern(
                pattern_type="confirmation_bias",
                description="Tendency to seek information confirming existing beliefs"
            ),
            CognitivePattern(
                pattern_type="anchoring",
                description="Relying too heavily on first piece of information"
            ),
            CognitivePattern(
                pattern_type="overconfidence",
                description="Overestimating own knowledge or abilities"
            ),
            CognitivePattern(
                pattern_type="recency_bias",
                description="Giving too much weight to recent events"
            ),
            CognitivePattern(
                pattern_type="sunk_cost",
                description="Continuing a decision based on past investment"
            ),
            CognitivePattern(
                pattern_type="availability_heuristic",
                description="Judging probability based on easily recalled examples"
            ),
        ]
        for p in builtin:
            self.patterns[p.pattern_type] = p
    
    async def analyze_action(
        self,
        context: str,
        thought: str,
        action: str,
        outcome: str
    ) -> Dict[str, Any]:
        """
        Analyze an action for patterns and improvements.
        
        This is the main entry point for the growth layer.
        """
        reflection = SelfReflection(
            context=context,
            thought=thought,
            action=action,
            outcome=outcome,
            learning="",
            improvement=""
        )
        
        detected_patterns = await self._detect_biases(thought, action, outcome)
        
        reflection.learning = await self._extract_learning(
            context, thought, action, outcome, detected_patterns
        )
        reflection.improvement = await self._generate_improvement(
            context, thought, action, detected_patterns
        )
        
        self.reflections.append(reflection)
        self.state.sessions_count += 1
        
        await self._update_patterns(detected_patterns, outcome == "success")
        
        return {
            "reflection_id": reflection.id,
            "detected_patterns": [p.pattern_type for p in detected_patterns],
            "learning": reflection.learning,
            "improvement": reflection.improvement,
            "bias_awareness": self.state.bias_awareness,
        }
    
    async def _detect_biases(
        self,
        thought: str,
        action: str,
        outcome: str
    ) -> List[CognitivePattern]:
        """Detect cognitive biases in thought/action"""
        detected = []
        
        thought_lower = thought.lower()
        action_lower = action.lower()
        outcome_lower = outcome.lower()
        
        # Confirmation bias: ignoring counter-evidence
        if any(x in thought_lower for x in ["always", "never", "obviously"]):
            if "but" not in thought_lower:
                detected.append(self.patterns.get("confirmation_bias"))
        
        # Anchoring: sticking to initial decision
        if any(x in action_lower for x in ["same as before", "like last time", "default"]):
            detected.append(self.patterns.get("anchoring"))
        
        # Overconfidence: high certainty, wrong outcome
        if outcome_lower == "failure" and any(x in thought_lower for x in ["sure", "certain", "definitely"]):
            detected.append(self.patterns.get("overconfidence"))
        
        # Recency bias: reacting to recent events
        if "just" in thought_lower or "recently" in thought_lower:
            detected.append(self.patterns.get("recency_bias"))
        
        # Filter out None
        return [p for p in detected if p]
    
    async def _extract_learning(
        self,
        context: str,
        thought: str,
        action: str,
        outcome: str,
        patterns: List[CognitivePattern]
    ) -> str:
        """Extract learning from the experience"""
        if outcome.lower() == "success":
            return f"Action '{action[:50]}' worked well in context '{context}'. Pattern: {[p.pattern_type for p in patterns]}"
        else:
            pattern_types = [p.pattern_type for p in patterns]
            if pattern_types:
                return f"Failure may be related to: {pattern_types}. Consider alternative approach."
            return f"Action '{action[:50]}' did not achieve desired outcome. Need to reconsider strategy."
    
    async def _generate_improvement(
        self,
        context: str,
        thought: str,
        action: str,
        patterns: List[CognitivePattern]
    ) -> str:
        """Generate improvement suggestions"""
        if not patterns:
            return "No specific bias detected. Continue current approach with attention to results."
        
        suggestions = {
            "confirmation_bias": "Seek counter-evidence before making decisions",
            "anchoring": "Reconsider initial assumptions, explore alternatives",
            "overconfidence": "Add uncertainty margins to predictions",
            "recency_bias": "Consider longer-term patterns, not just recent events",
            "sunk_cost": "Evaluate current state objectively, ignore past costs",
            "availability_heuristic": "Gather more data points before judging",
        }
        
        pattern_names = [p.pattern_type for p in patterns]
        return "; ".join([suggestions.get(p, "") for p in pattern_names if p in suggestions])
    
    async def _update_patterns(
        self,
        patterns: List[CognitivePattern],
        success: bool
    ) -> None:
        """Update pattern statistics"""
        for pattern in patterns:
            pattern.frequency += 1
            pattern.last_seen = datetime.utcnow()
            
            old_acc = pattern.accuracy
            n = pattern.frequency
            pattern.accuracy = (old_acc * (n - 1) + (1 if success else 0)) / n
    
    def get_bias_report(self) -> Dict[str, Any]:
        """Get current bias awareness report"""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: (p.frequency, p.accuracy),
            reverse=True
        )
        
        return {
            "bias_awareness": self.state.bias_awareness,
            "total_reflections": len(self.reflections),
            "detected_patterns": [
                {
                    "type": p.pattern_type,
                    "description": p.description,
                    "frequency": p.frequency,
                    "accuracy": p.accuracy,
                }
                for p in sorted_patterns[:5]
            ],
            "recent_learning": [
                {
                    "context": r.context,
                    "learning": r.learning,
                    "improvement": r.improvement,
                }
                for r in self.reflections[-3:]
            ]
        }
    
    def update_meta_state(self, attention: float, reflection: float) -> None:
        """Update meta-cognitive state based on recent performance"""
        self.state.attention_depth = attention
        self.state.reflection_depth = reflection
        
        if len(self.reflections) >= 2:
            recent_success = sum(
                1 for r in self.reflections[-5:] if r.outcome == "success"
            ) / min(len(self.reflections), 5)
            self.state.improvement_rate = recent_success - self.state.improvement_rate