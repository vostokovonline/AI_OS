"""
Policy Layer - Unified Decision Function

policy(state) → action

This is the core decision layer that replaces the "recommendation" pattern
with actual action selection.

Rule-based at first (can be replaced with learned later).
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)


class ActionType(Enum):
    EXECUTE = "execute"
    DECOMPOSE = "decompose"
    EXPLORE = "explore"
    WAIT = "wait"
    RECONSIDER = "reconsider"
    RETRY = "retry"
    ABORT = "abort"


@dataclass
class PolicyAction:
    """An action selected by the policy"""
    action_type: ActionType
    confidence: float
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[Tuple[ActionType, float]] = field(default_factory=list)


@dataclass
class ActionCandidate:
    """A candidate action for evaluation"""
    action_type: ActionType
    parameters: Dict[str, Any]
    estimated_utility: float = 0.0
    estimated_risk: float = 0.0


class PolicyLayer:
    """
    Unified policy layer: policy(state) → action
    
    Takes unified state and outputs a concrete action.
    
    Architecture:
    1. Generate candidates from state
    2. Score each candidate
    3. Select best (or use planning loop for complex decisions)
    """
    
    def __init__(self, state_builder, world_model):
        self.state_builder = state_builder
        self.world_model = world_model
        self.last_action: Optional[PolicyAction] = None
        self.decision_history: List[Dict] = []
        logger.info("policy_layer_initialized")
    
    async def decide(
        self,
        context: Optional[Dict[str, Any]] = None,
        use_planning: bool = False
    ) -> PolicyAction:
        """
        Main decision function.
        
        Args:
            context: Optional context overrides
            use_planning: If True, use planning loop for complex decisions
        
        Returns:
            PolicyAction with selected action
        """
        state = await self.state_builder.build_state(context)
        
        candidates = self._generate_candidates(state, context)
        
        if use_planning and len(candidates) > 1:
            scored = await self._plan_and_score(candidates, state)
        else:
            scored = self._score_candidates(candidates, state)
        
        best = max(scored, key=lambda x: x[1])
        action = candidates[best[0]]
        
        policy_action = PolicyAction(
            action_type=action.action_type,
            confidence=best[1],
            reasoning=self._explain_decision(state, action, best[1]),
            parameters=action.parameters,
            alternatives=[
                (candidates[i].action_type, scored[i][1])
                for i in scored.keys() if scored[i][1] < best[1]
            ][:3]
        )
        
        self.last_action = policy_action
        self._record_decision(state, policy_action)
        
        logger.info(
            "policy_decision",
            action=policy_action.action_type.value,
            confidence=policy_action.confidence,
            reasoning=policy_action.reasoning[:100]
        )
        
        return policy_action
    
    def _generate_candidates(
        self,
        state,
        context: Optional[Dict[str, Any]]
    ) -> List[ActionCandidate]:
        """Generate action candidates based on state"""
        candidates = []
        
        task_type = context.get("task_type", "default") if context else "default"
        
        if state.stress_level > 0.8:
            candidates.append(ActionCandidate(
                action_type=ActionType.WAIT,
                parameters={"reason": "high_stress"}
            ))
            return candidates
        
        if state.action_readiness < 0.3:
            candidates.append(ActionCandidate(
                action_type=ActionType.RECONSIDER,
                parameters={"reason": "low_readiness"}
            ))
        
        if state.world_recent_outcome == "failure":
            candidates.append(ActionCandidate(
                action_type=ActionType.ABORT,
                parameters={"reason": "recent_failure"}
            ))
            candidates.append(ActionCandidate(
                action_type=ActionType.RETRY,
                parameters={"reason": "alternative_approach"}
            ))
        
        if state.bias_count > 2 and state.bias_awareness < 0.5:
            candidates.append(ActionCandidate(
                action_type=ActionType.RECONSIDER,
                parameters={"reason": "bias_detected"}
            ))
        
        if state.task_complexity > 0.7:
            candidates.append(ActionCandidate(
                action_type=ActionType.DECOMPOSE,
                parameters={"depth": 2}
            ))
        
        if state.exploration_tendency > 0.6:
            candidates.append(ActionCandidate(
                action_type=ActionType.EXPLORE,
                parameters={"strategy": state.top_strategy_name}
            ))
        
        candidates.append(ActionCandidate(
            action_type=ActionType.EXECUTE,
            parameters={
                "strategy": state.top_strategy_name,
                "complexity": state.task_complexity
            }
        ))
        
        if not candidates:
            candidates.append(ActionCandidate(
                action_type=ActionType.WAIT,
                parameters={"reason": "default"}
            ))
        
        return candidates
    
    def _score_candidates(
        self,
        candidates: List[ActionCandidate],
        state
    ) -> Dict[int, Tuple[ActionType, float]]:
        """Score candidates (rule-based)"""
        scored = {}
        
        for i, candidate in enumerate(candidates):
            score = 0.5
            
            if candidate.action_type == ActionType.EXECUTE:
                score = self._score_execute(state)
            elif candidate.action_type == ActionType.DECOMPOSE:
                score = self._score_decompose(state)
            elif candidate.action_type == ActionType.EXPLORE:
                score = self._score_explore(state)
            elif candidate.action_type == ActionType.WAIT:
                score = self._score_wait(state)
            elif candidate.action_type == ActionType.RECONSIDER:
                score = self._score_reconsider(state)
            elif candidate.action_type == ActionType.RETRY:
                score = self._score_retry(state)
            elif candidate.action_type == ActionType.ABORT:
                score = self._score_abort(state)
            
            scored[i] = (candidate.action_type, score)
        
        return scored
    
    def _score_execute(self, state) -> float:
        """Score EXECUTE action"""
        score = 0.6
        score += state.action_readiness * 0.3
        score += state.confidence * 0.2
        score -= state.stress_level * 0.2
        score += (1 - state.task_complexity) * 0.1
        return max(0, min(1, score))
    
    def _score_decompose(self, state) -> float:
        """Score DECOMPOSE action"""
        score = 0.3
        score += state.task_complexity * 0.4
        score += (1 - state.stress_level) * 0.2
        if state.top_strategy_name == "parallel_decomposition":
            score += 0.2
        return max(0, min(1, score))
    
    def _score_explore(self, state) -> float:
        """Score EXPLORE action"""
        score = 0.4
        score += state.exploration_tendency * 0.3
        score += state.identity_coherence * 0.2
        score -= state.stress_level * 0.2
        score += state.task_novelty * 0.1
        return max(0, min(1, score))
    
    def _score_wait(self, state) -> float:
        """Score WAIT action"""
        score = 0.3
        score += state.stress_level * 0.3
        score += (1 - state.action_readiness) * 0.3
        score -= state.task_urgency * 0.3
        return max(0, min(1, score))
    
    def _score_reconsider(self, state) -> float:
        """Score RECONSIDER action"""
        score = 0.3
        score += state.bias_count * 0.1
        score += (1 - state.bias_awareness) * 0.3
        score -= state.action_readiness * 0.2
        return max(0, min(1, score))
    
    def _score_retry(self, state) -> float:
        """Score RETRY action"""
        score = 0.4
        score += state.confidence * 0.3
        score += state.reflection_depth * 0.2
        score -= state.stress_level * 0.2
        return max(0, min(1, score))
    
    def _score_abort(self, state) -> float:
        """Score ABORT action"""
        score = 0.2
        score += (1 - state.confidence) * 0.3
        score += state.bias_count * 0.1
        score += state.stress_level * 0.2
        return max(0, min(1, score))
    
    async def _plan_and_score(
        self,
        candidates: List[ActionCandidate],
        state
    ) -> Dict[int, Tuple[ActionType, float]]:
        """Use planning loop for complex decisions"""
        from ..simulation.planner import SimulationPlanner
        
        planner = SimulationPlanner(self.world_model)
        scored = {}
        
        for i, candidate in enumerate(candidates):
            sim_result = await planner.simulate_action(
                action=candidate.action_type,
                parameters=candidate.parameters,
                state=state
            )
            
            base_score = self._score_candidates([candidate], state)[0][1]
            plan_score = sim_result.get("expected_utility", 0.5)
            
            combined = base_score * 0.4 + plan_score * 0.6
            
            scored[i] = (candidate.action_type, combined)
        
        return scored
    
    def _explain_decision(self, state, candidate, confidence: float) -> str:
        """Generate human-readable decision explanation"""
        reasons = []
        
        if state.stress_level > 0.6:
            reasons.append(f"stress={state.stress_level:.2f}")
        if state.action_readiness < 0.5:
            reasons.append(f"readiness={state.action_readiness:.2f}")
        if state.top_strategy_name != "default":
            reasons.append(f"strategy={state.top_strategy_name}")
        if state.bias_count > 1:
            reasons.append(f"biases={state.bias_count}")
        
        return f"{candidate.action_type.value}: {', '.join(reasons) if reasons else 'default decision'} (conf={confidence:.2f})"
    
    def _record_decision(self, state, policy_action: PolicyAction) -> None:
        """Record decision for learning"""
        self.decision_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "action": policy_action.action_type.value,
            "confidence": policy_action.confidence,
            "reasoning": policy_action.reasoning,
            "state_snapshot": {
                "stress": state.stress_level,
                "readiness": state.action_readiness,
                "outcome": state.world_recent_outcome,
            }
        })
        
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision statistics"""
        if not self.decision_history:
            return {"count": 0}
        
        actions = [d["action"] for d in self.decision_history[-100:]]
        confidences = [d["confidence"] for d in self.decision_history[-100:]]
        
        return {
            "total_decisions": len(self.decision_history),
            "action_distribution": {
                a: actions.count(a) / len(actions)
                for a in set(actions)
            },
            "avg_confidence": sum(confidences) / len(confidences),
            "recent_decisions": self.decision_history[-5:]
        }