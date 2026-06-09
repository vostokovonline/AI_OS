"""
Unified Agent - Single Entry Point for Model-Based Agent

This replaces the "CognitiveOS as advisor" pattern with:
"CognitiveOS as decision system"

Architecture:
CognitiveOS (world model + narrative + emotion + growth)
    ↓
StateBuilder (unified latent state)
    ↓
PolicyLayer (action = policy(state))
    ↓
SimulationPlanner (simulate → evaluate → select)
    ↓
DecisionTransaction (atomic observability boundary)
    ↓
Action Execution + Outcome Recording
    ↓
Loop back to StateBuilder

Key Feature: DecisionTransaction as first-class atomic construct
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentContext:
    """Context for agent decision-making"""
    user_id: str
    task: str
    task_type: str = "default"
    complexity: float = 0.5
    urgency: float = 0.5
    novelty: float = 0.5


@dataclass
class AgentDecision:
    """Full decision output from the agent"""
    action_type: str
    confidence: float
    reasoning: str
    expected_utility: float
    expected_risk: float
    alternatives: List[Dict]
    simulation_used: bool
    transaction_id: Optional[str] = None


class UnifiedAgent:
    """
    Unified Agent - single entry point for the cognitive agent system.
    
    This is the main interface for external callers.
    
    Usage:
        agent = UnifiedAgent(cognitive_os)
        
        # Simple decision
        decision = await agent.decide(context)
        
        # Transaction-based decision (full observability)
        async with agent.transaction(user_id, task) as txn:
            await txn.decide(context)
            await txn.record_outcome("success", 1.0)
        
        # Execute and record outcome
        await agent.execute_and_learn(decision, outcome)
    """
    
    def __init__(self, cognitive_os):
        self.cognitive_os = cognitive_os
        
        from .state_builder import StateBuilder
        from .policy_layer import PolicyLayer
        from .simulation.planner import SimulationPlanner
        from ..observability.decision_trace import DecisionTracer, StateSnapshot as TraceStateSnapshot
        from ..observability.state_diff import StateDiffEngine
        from ..observability.attribution import PolicyAttributionSystem
        from ..observability.transaction import TransactionContext
        
        self.state_builder = StateBuilder(cognitive_os)
        self.world_model = cognitive_os.world_model
        self.policy = PolicyLayer(self.state_builder, self.world_model)
        self.planner = SimulationPlanner(self.world_model)
        
        self.tracer = DecisionTracer()
        self.diff_engine = StateDiffEngine()
        self.attribution_system = PolicyAttributionSystem()
        
        self.decision_history: List[AgentDecision] = []
        self._current_transaction: Optional[Any] = None
        self._transaction_history: List = []
        
        logger.info("unified_agent_initialized")
    
    def transaction(self, user_id: str, task: str) -> TransactionContext:
        """
        Create a transaction context for atomic decision recording.
        
        Usage:
            async with agent.transaction(user_id, task) as txn:
                await txn.decide(context)
                await txn.record_outcome("success", 1.0)
        """
        return TransactionContext(self, user_id, task)
    
    async def decide(
        self,
        context: AgentContext,
        use_simulation: bool = True
    ) -> AgentDecision:
        """
        Make a decision based on context (simple mode without transaction).
        
        For full observability with causal timeline, use transaction() instead.
        
        Args:
            context: AgentContext with task information
            use_simulation: Whether to use planning loop
        
        Returns:
            AgentDecision with action and reasoning
        """
        context_dict = {
            "task": context.task,
            "task_type": context.task_type,
            "complexity": context.complexity,
            "urgency": context.urgency,
            "novelty": context.novelty
        }
        
        state = await self.state_builder.build_state(context_dict)
        self.diff_engine.record_state(state)
        
        policy_action = await self.policy.decide(
            context=context_dict,
            use_planning=use_simulation
        )
        
        scoring = {
            "stress_score": state.stress_level,
            "readiness_score": state.action_readiness,
            "confidence_score": state.confidence,
            "exploration_score": state.exploration_tendency,
        }
        
        self.attribution_system.record(
            state=state.to_vector(),
            action=policy_action.action_type.value,
            confidence=policy_action.confidence,
            scoring=scoring,
            alternatives=policy_action.alternatives
        )
        
        decision = AgentDecision(
            action_type=policy_action.action_type.value,
            confidence=policy_action.confidence,
            reasoning=policy_action.reasoning,
            expected_utility=policy_action.confidence,
            expected_risk=1 - policy_action.confidence,
            alternatives=[
                {"action": a[0].value, "score": a[1]}
                for a in policy_action.alternatives
            ],
            simulation_used=use_simulation,
            transaction_id=self._current_transaction.id if self._current_transaction else None
        )
        
        self.decision_history.append(decision)
        
        self.cognitive_os.world_model.record_action(
            actor_id=context.user_id,
            action=f"decide_{decision.action_type}",
            outcome="decided"
        )
        
        logger.info(
            "agent_decided",
            action=decision.action_type,
            confidence=decision.confidence,
            simulation=decision.simulation_used,
            transaction_id=decision.transaction_id
        )
        
        return decision
    
    async def decide_with_transaction(
        self,
        context: AgentContext,
        use_simulation: bool = True
    ) -> AgentDecision:
        """
        Make a decision with full transaction tracking.
        
        This is the recommended method for Phase 9+ observability.
        """
        txn = self.transaction(context.user_id, context.task)
        
        async with txn:
            await txn.decide(context, use_simulation=use_simulation)
        
        decision = AgentDecision(
            action_type=txn.transaction.selected_action or "unknown",
            confidence=txn.transaction.confidence,
            reasoning=txn.transaction.reasoning,
            expected_utility=txn.transaction.confidence,
            expected_risk=1 - txn.transaction.confidence,
            alternatives=[
                {"action": c.action_type, "score": c.score}
                for c in txn.transaction.candidates if not c.selected
            ],
            simulation_used=use_simulation,
            transaction_id=txn.transaction.id
        )
        
        self.decision_history.append(decision)
        return decision
    
    async def execute_and_learn(
        self,
        decision: AgentDecision,
        outcome: str,
        metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute decision and record outcome for learning.
        
        Args:
            decision: The decision that was made
            outcome: Result of execution (success, partial, failure)
            metrics: Optional metrics (duration, quality, etc.)
        
        Returns:
            Learning feedback
        """
        outcome_val = {"success": 1.0, "partial": 0.5, "failure": 0.0}.get(outcome, 0.5)
        
        self.cognitive_os.world_model.record_action(
            actor_id="unified_agent",
            action=decision.action_type,
            outcome=outcome
        )
        
        if self.policy.last_action and self.policy.last_action.parameters.get("strategy"):
            self.cognitive_os.record_outcome(
                strategy_id=self.cognitive_os.strategy_evolution.strategies.get(
                    self.policy.last_action.parameters.get("strategy", "")
                ) or list(self.cognitive_os.strategy_evolution.strategies.keys())[0],
                outcome=outcome,
                score=outcome_val
            )
        
        analysis = await self.cognitive_os.growth.analyze_action(
            context=f"action={decision.action_type}",
            thought=decision.reasoning,
            action=decision.action_type,
            outcome=outcome
        )
        
        learning = {
            "outcome": outcome,
            "outcome_score": outcome_val,
            "analysis": analysis,
            "metrics": metrics or {}
        }
        
        logger.info(
            "learning_recorded",
            outcome=outcome,
            score=outcome_val,
            patterns_detected=len(analysis.get("detected_patterns", [])),
            transaction_id=decision.transaction_id
        )
        
        return learning
    
    async def plan_sequence(
        self,
        context: AgentContext,
        num_actions: int = 3
    ) -> List[AgentDecision]:
        """
        Plan a sequence of actions using the simulation loop.
        
        Args:
            context: AgentContext with task information
            num_actions: Number of actions to plan
        
        Returns:
            Ordered list of decisions
        """
        plan = []
        current_context = context
        
        for i in range(num_actions):
            decision = await self.decide(current_context, use_simulation=True)
            plan.append(decision)
            
            if decision.action_type in ("abort", "wait"):
                break
            
            if decision.action_type == "decompose":
                current_context.complexity *= 0.7
            elif decision.action_type == "execute":
                current_context.complexity *= 0.5
            
            current_context.urgency *= 0.9
        
        return plan
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            "decisions_made": len(self.decision_history),
            "transactions_recorded": len(self._transaction_history),
            "policy_stats": self.policy.get_decision_stats(),
            "cognitive_state": self.cognitive_os.get_state(),
            "recent_decisions": [
                {
                    "action": d.action_type,
                    "confidence": d.confidence,
                    "outcome_predicted": d.expected_utility,
                    "transaction_id": d.transaction_id
                }
                for d in self.decision_history[-10:]
            ],
            "recent_transactions": [
                {"id": t.id, "status": t.status, "action": t.selected_action}
                for t in self._transaction_history[-5:]
            ]
        }
    
    def replay_transaction(self, transaction_id: str) -> Optional[Dict]:
        """
        Replay a transaction for debugging/analysis.
        
        Returns the transaction with full causal chain for inspection.
        """
        for txn in self._transaction_history:
            if txn.id == transaction_id:
                return txn.to_dict()
        return None
    
    def get_replayable_dataset(self, limit: int = 100) -> List[Dict]:
        """
        Get replayable dataset for RL training.
        
        Each entry has complete state → action → outcome sequence.
        """
        dataset = []
        for txn in self._transaction_history[-limit:]:
            if txn.is_complete() and txn.outcome:
                dataset.append({
                    "transaction_id": txn.id,
                    "state": txn.state_before.to_dict() if txn.state_before else None,
                    "reasoning": [e.to_dict() for e in txn.reasoning_events],
                    "action": txn.selected_action,
                    "confidence": txn.confidence,
                    "outcome": txn.outcome.outcome,
                    "outcome_score": txn.outcome.score,
                })
        return dataset


async def agent_decide_in_transaction(
    agent,
    context: AgentContext,
    use_simulation: bool = True
) -> AgentDecision:
    """
    Helper function to make a decision within a transaction context.
    
    This is called by TransactionContext.decide().
    """
    return await agent.decide(context, use_simulation=use_simulation)


# Singleton factory
_unified_agent: Optional[UnifiedAgent] = None


def get_unified_agent(cognitive_os) -> UnifiedAgent:
    """Get or create unified agent singleton"""
    global _unified_agent
    if _unified_agent is None:
        _unified_agent = UnifiedAgent(cognitive_os)
    return _unified_agent