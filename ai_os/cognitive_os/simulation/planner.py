"""
Simulation Planner - Planning Loop

Simulates N actions, evaluates outcomes, selects best.

This is the core of the model-based planning system:
1. Take current state
2. Generate action candidates
3. Simulate each action's outcome
4. Evaluate expected utility
5. Select best action

For now: heuristic-based simulation (can be replaced with learned model).
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Result of action simulation"""
    action_type: str
    state_delta: Dict[str, Any]
    expected_utility: float
    expected_risk: float
    rollout_depth: int
    reasoning: str


class SimulationPlanner:
    """
    Planning loop with simulation-based evaluation.
    
    Uses world model for prediction + heuristic scoring.
    """
    
    def __init__(self, world_model):
        self.world_model = world_model
        self.rollout_depth = 3
        self.num_simulations = 5
        logger.info("simulation_planner_initialized")
    
    async def simulate_action(
        self,
        action: Any,
        parameters: Dict[str, Any],
        state
    ) -> SimulationResult:
        """
        Simulate a single action and return expected outcome.
        
        Args:
            action: ActionType enum
            parameters: Action parameters
            state: Current UnifiedState
        
        Returns:
            SimulationResult with expected utility/risk
        """
        rollout_results = []
        
        for i in range(self.num_simulations):
            result = await self._rollout(action, parameters, state, depth=0)
            rollout_results.append(result)
        
        avg_utility = sum(r["utility"] for r in rollout_results) / len(rollout_results)
        avg_risk = sum(r["risk"] for r in rollout_results) / len(rollout_results)
        
        state_delta = self._compute_delta(rollout_results)
        reasoning = self._generate_reasoning(action, rollout_results)
        
        return SimulationResult(
            action_type=action.value if hasattr(action, 'value') else str(action),
            state_delta=state_delta,
            expected_utility=avg_utility,
            expected_risk=avg_risk,
            rollout_depth=self.rollout_depth,
            reasoning=reasoning
        )
    
    async def plan(
        self,
        candidates: List[Any],
        state,
        parameters_list: Optional[List[Dict]] = None
    ) -> List[Tuple[Any, SimulationResult]]:
        """
        Plan across multiple action candidates.
        
        Args:
            candidates: List of ActionType candidates
            state: Current UnifiedState
            parameters_list: Optional list of parameters for each candidate
        
        Returns:
            List of (action, result) tuples sorted by expected utility
        """
        results = []
        
        for i, action in enumerate(candidates):
            params = parameters_list[i] if parameters_list else {}
            result = await self.simulate_action(action, params, state)
            results.append((action, result))
        
        results.sort(key=lambda x: x[1].expected_utility, reverse=True)
        
        logger.info(
            "plan_completed",
            candidates=len(candidates),
            best_action=results[0][0].value if hasattr(results[0][0], 'value') else results[0][0],
            best_utility=results[0][1].expected_utility
        )
        
        return results
    
    async def _rollout(
        self,
        action,
        parameters: Dict[str, Any],
        state,
        depth: int
    ) -> Dict[str, float]:
        """
        Single rollout simulation.
        
        Returns:
            {"utility": float, "risk": float}
        """
        if depth >= self.rollout_depth:
            return {"utility": 0.5, "risk": 0.3}
        
        immediate = self._simulate_immediate(action, parameters, state)
        
        if depth == 0:
            return immediate
        
        continuation_utility = 0.5
        continuation_risk = 0.3
        
        future = await self._rollout(
            action,
            parameters,
            state,
            depth=depth + 1
        )
        
        continuation_utility = future["utility"] * 0.8
        continuation_risk = future["risk"] * 0.9
        
        return {
            "utility": immediate["utility"] + continuation_utility * 0.5,
            "risk": immediate["risk"] + continuation_risk * 0.5
        }
    
    def _simulate_immediate(
        self,
        action,
        parameters: Dict[str, Any],
        state
    ) -> Dict[str, float]:
        """Simulate immediate effects of action"""
        
        action_name = action.value if hasattr(action, 'value') else str(action)
        
        if action_name == "execute":
            return self._sim_execute(state)
        elif action_name == "decompose":
            return self._sim_decompose(state)
        elif action_name == "explore":
            return self._sim_explore(state)
        elif action_name == "wait":
            return self._sim_wait(state)
        elif action_name == "reconsider":
            return self._sim_reconsider(state)
        elif action_name == "retry":
            return self._sim_retry(state)
        elif action_name == "abort":
            return self._sim_abort(state)
        else:
            return {"utility": 0.5, "risk": 0.3}
    
    def _sim_execute(self, state) -> Dict[str, float]:
        """Simulate EXECUTE action"""
        utility = 0.6
        risk = 0.2
        
        utility += state.action_readiness * 0.2
        utility += state.confidence * 0.1
        utility -= state.task_complexity * 0.1
        
        risk += state.stress_level * 0.1
        risk += state.bias_count * 0.05
        
        if state.world_recent_outcome == "failure":
            risk += 0.2
            utility -= 0.1
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _sim_decompose(self, state) -> Dict[str, float]:
        """Simulate DECOMPOSE action"""
        utility = 0.5
        risk = 0.15
        
        utility += state.task_complexity * 0.2
        risk += state.stress_level * 0.1
        
        if state.top_strategy_name == "parallel_decomposition":
            utility += 0.1
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _sim_explore(self, state) -> Dict[str, float]:
        """Simulate EXPLORE action"""
        utility = 0.4
        risk = 0.3
        
        utility += state.exploration_tendency * 0.2
        utility += state.task_novelty * 0.1
        
        risk += state.stress_level * 0.15
        risk += (1 - state.identity_coherence) * 0.1
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _sim_wait(self, state) -> Dict[str, float]:
        """Simulate WAIT action"""
        utility = 0.3
        risk = 0.1
        
        utility += state.stress_level * 0.2
        utility -= state.task_urgency * 0.2
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _sim_reconsider(self, state) -> Dict[str, float]:
        """Simulate RECONSIDER action"""
        utility = 0.4
        risk = 0.2
        
        utility += state.bias_awareness * 0.2
        utility += state.reflection_depth * 0.1
        
        risk += (1 - state.action_readiness) * 0.1
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _sim_retry(self, state) -> Dict[str, float]:
        """Simulate RETRY action"""
        utility = 0.5
        risk = 0.25
        
        utility += state.confidence * 0.15
        utility += state.reflection_depth * 0.1
        
        risk += state.stress_level * 0.15
        risk += (1 - state.identity_coherence) * 0.1
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _sim_abort(self, state) -> Dict[str, float]:
        """Simulate ABORT action"""
        utility = 0.2
        risk = 0.1
        
        utility += (1 - state.confidence) * 0.2
        utility += state.stress_level * 0.1
        
        return {
            "utility": max(0, min(1, utility)),
            "risk": max(0, min(1, risk))
        }
    
    def _compute_delta(self, rollouts: List[Dict]) -> Dict[str, Any]:
        """Compute average state change from rollouts"""
        avg_utility = sum(r["utility"] for r in rollouts) / len(rollouts)
        avg_risk = sum(r["risk"] for r in rollouts) / len(rollouts)
        
        return {
            "expected_utility": avg_utility,
            "expected_risk": avg_risk,
            "risk_adjusted_score": avg_utility - avg_risk * 0.5
        }
    
    def _generate_reasoning(
        self,
        action,
        rollouts: List[Dict]
    ) -> str:
        """Generate human-readable reasoning for simulation"""
        utilities = [r["utility"] for r in rollouts]
        risks = [r["risk"] for r in rollouts]
        
        action_name = action.value if hasattr(action, 'value') else str(action)
        
        variance = max(utilities) - min(utilities) if utilities else 0
        
        return (
            f"Action '{action_name}' simulated {len(rollouts)} times. "
            f"Expected utility: {sum(utilities)/len(utilities):.2f}, "
            f"Expected risk: {sum(risks)/len(risks):.2f}, "
            f"Variance: {variance:.2f}"
        )