"""
Execution Adapters - Bridge between cognitive state and external action.

Stage: Four-Layer Architecture

Adapters are the ONLY way cognitive state affects the outside world.
They translate decisions into actions and actions into events.

Architecture:
    Policies ← State ← Reducers ← Events
    
    Events ← Adapters ← External Systems
    (output)              (input)

Key principles:
1. Adapters are stateful (they track external state)
2. Adapters emit events (never mutate cognitive state)
3. Adapters translate between internal and external representations
4. Multiple adapters can exist for different external systems

Types of adapters:
- GoalExecutor: executes goals
- BeliefCollector: collects evidence/beliefs
- ActionPerformer: performs physical actions
- MonitorAdapter: monitors execution results
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Callable, Set, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json

from event_log import ImmutableEvent, EventFactory, EventTypes, StreamIds, CausalMetadata
from reducers import CognitiveState, ExecutionState, IdentityState
from policies import PolicyResult


class AdapterState(Enum):
    """Adapter lifecycle states"""
    IDLE = "idle"
    PREPARING = "preparing"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class AdapterResult:
    """
    Result from adapter execution.
    
    Contains:
    - success: whether action succeeded
    - events: events to emit
    - external_state: state to track externally
    - error: error message if failed
    """
    success: bool
    events: Tuple[ImmutableEvent, ...]
    external_state: Dict[str, Any]
    error: Optional[str] = None
    adaptation_suggestions: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionContext:
    """
    Context for adapter execution.
    
    Carries all necessary state for adapter to make decisions.
    """
    cognitive_state: CognitiveState
    identity_state: IdentityState
    execution_state: ExecutionState
    policy: PolicyResult
    goal_id: str
    causal_metadata: CausalMetadata
    trace_id: str


class ExecutionAdapter(Protocol):
    """
    Protocol for execution adapters.
    
    Adapters must implement:
    - prepare(): Prepare for execution
    - execute(): Perform the action
    - cleanup(): Clean up after execution
    """
    
    def prepare(self, context: ExecutionContext) -> AdapterResult:
        """Prepare for execution"""
        ...
    
    def execute(self, context: ExecutionContext) -> AdapterResult:
        """Perform the action"""
        ...
    
    def cleanup(self, context: ExecutionContext) -> AdapterResult:
        """Clean up after execution"""
        ...


class GoalExecutionAdapter:
    """
    Adapter for executing goals.
    
    Takes goal from execution state and produces events.
    Applies identity-driven policies to execution.
    """
    
    def __init__(self):
        self._factory = EventFactory()
        self._state = AdapterState.IDLE
    
    def execute_goal(
        self,
        context: ExecutionContext,
        goal: Dict[str, Any]
    ) -> AdapterResult:
        """
        Execute a goal with identity-aware behavior.
        
        Steps:
        1. Create execution event (with causal metadata)
        2. Apply policy to goal
        3. Simulate execution (placeholder)
        4. Emit result events
        """
        events = []
        
        # 1. Emit goal execution started
        execution_event = self._factory.create_execution_event(
            event_type=EventTypes.GOAL_EXECUTED,
            payload={
                "goal_id": context.goal_id,
                "title": goal.get("title", ""),
                "strategy": context.policy.dominant_strategy,
                "risk_tolerance": context.policy.risk_appetite,
                "decomposition_style": goal.get("decomposition_style", "hierarchical"),
                "verification_strictness": goal.get("verification_strictness", 0.5)
            },
            causation_id=context.causal_metadata.event_id,
            correlation_id=context.trace_id
        )
        events.append(execution_event)
        
        # 2. Simulate execution based on identity
        execution_succeeded = self._simulate_execution(context, goal)
        
        # 3. Emit result event
        if execution_succeeded:
            result_event = self._factory.create_execution_event(
                event_type=EventTypes.GOAL_COMPLETED,
                payload={
                    "goal_id": context.goal_id,
                    "outcome": "success",
                    "strategy_used": context.policy.dominant_strategy,
                    "risk_taken": context.policy.risk_appetite
                },
                causation_id=execution_event.causal.event_id,
                correlation_id=context.trace_id
            )
        else:
            result_event = self._factory.create_execution_event(
                event_type=EventTypes.GOAL_FAILED,
                payload={
                    "goal_id": context.goal_id,
                    "outcome": "failed",
                    "reason": "execution_failed",
                    "strategy_used": context.policy.dominant_strategy,
                    "retry_recommended": context.policy.risk_appetite > 0.5
                },
                causation_id=execution_event.causal.event_id,
                correlation_id=context.trace_id
            )
        events.append(result_event)
        
        # 4. Emit identity state event
        identity_event = self._factory.create_identity_event(
            event_type=EventTypes.IDENTITY_MUTATED,
            payload={
                "axis": "autonomy",
                "delta": 0.01 if execution_succeeded else -0.01,
                "trigger": context.goal_id
            },
            causation_id=result_event.causal.event_id
        )
        events.append(identity_event)
        
        return AdapterResult(
            success=execution_succeeded,
            events=tuple(events),
            external_state={"executed_goal_id": context.goal_id},
            error=None if execution_succeeded else "Execution failed"
        )
    
    def _simulate_execution(self, context: ExecutionContext, goal: Dict[str, Any]) -> bool:
        """Simulate execution based on identity and policy"""
        import random
        
        # Base success probability
        base_prob = 0.8
        
        # Adjust for risk appetite
        risk = context.policy.risk_appetite
        if risk > 0.7:
            # High risk = high variance
            return random.random() < (base_prob + 0.1)
        elif risk < 0.3:
            # Low risk = conservative, lower variance
            return random.random() < (base_prob + 0.05)
        else:
            return random.random() < base_prob


class BeliefCollectionAdapter:
    """
    Adapter for collecting beliefs/evidence.
    
    Takes evidence and produces belief events.
    Applies identity-aware filtering and weighting.
    """
    
    def __init__(self):
        self._factory = EventFactory()
    
    def process_evidence(
        self,
        context: ExecutionContext,
        evidence: Dict[str, Any]
    ) -> AdapterResult:
        """
        Process evidence into beliefs.
        
        Steps:
        1. Apply identity-based filtering
        2. Weight evidence by identity
        3. Detect potential contradictions
        4. Emit belief events
        """
        events = []
        
        # 1. Determine belief creation
        proposition = evidence.get("proposition", "")
        strength = evidence.get("strength", 0.5)
        source = evidence.get("source", "unknown")
        
        # 2. Apply identity-based weighting
        # High curiosity → accept more beliefs
        # High stability → more skeptical
        curiosity = context.identity_state.axes.get("curiosity", 0.5)
        stability = context.identity_state.axes.get("stability", 0.5)
        
        # Adjust strength based on identity
        adjusted_strength = strength
        if curiosity > 0.7:
            # Curious = accept more
            adjusted_strength = min(1.0, adjusted_strength * 1.1)
        if stability > 0.7:
            # Stable = more skeptical
            adjusted_strength = adjusted_strength * 0.9
        
        # 3. Check for contradictions
        has_contradiction = False
        for existing_belief_id, belief in context.cognitive_state.beliefs.items():
            if proposition in belief.get("proposition", ""):
                # Similar belief exists - check for contradiction
                existing_strength = belief.get("confidence", 0.5)
                if abs(adjusted_strength - existing_strength) > 0.4:
                    has_contradiction = True
        
        # 4. Emit belief event
        belief_event = self._factory.create_cognitive_event(
            event_type=EventTypes.BELIEF_CREATED,
            payload={
                "proposition": proposition,
                "confidence": adjusted_strength,
                "source": source,
                "identity_weighted": True,
                "contradiction_risk": has_contradiction
            },
            causation_id=context.causal_metadata.event_id
        )
        events.append(belief_event)
        
        # 5. Emit contradiction if detected
        if has_contradiction:
            contradiction_event = self._factory.create_cognitive_event(
                event_type=EventTypes.CONTRADICTION_DETECTED,
                payload={
                    "type": "belief_belief",
                    "intensity": abs(adjusted_strength - strength)
                },
                causation_id=belief_event.causal.event_id
            )
            events.append(contradiction_event)
        
        return AdapterResult(
            success=True,
            events=tuple(events),
            external_state={"processed_evidence": True}
        )


class PressureUpdateAdapter:
    """
    Adapter for processing pressure updates.
    
    Takes events and produces pressure events.
    Applies identity-based pressure handling.
    """
    
    def __init__(self):
        self._factory = EventFactory()
    
    def process_pressure(
        self,
        context: ExecutionContext,
        pressure_data: Dict[str, Any]
    ) -> AdapterResult:
        """
        Process pressure data into pressure events.
        
        Pressure is accumulated from:
        - Unresolved contradictions
        - Failed goals
        - Identity threats
        """
        events = []
        
        pressure_type = pressure_data.get("type", "cognitive")
        source = pressure_data.get("source", "")
        intensity = pressure_data.get("intensity", 0.5)
        
        # Apply identity-based pressure handling
        # High stability → lower pressure sensitivity
        # High autonomy → higher pressure tolerance
        stability = context.identity_state.axes.get("stability", 0.5)
        autonomy = context.identity_state.axes.get("autonomy", 0.5)
        
        adjusted_intensity = intensity
        if stability > 0.7:
            adjusted_intensity *= 0.8
        if autonomy > 0.7:
            adjusted_intensity *= 0.9
        
        # Create pressure event
        pressure_event = self._factory.create_cognitive_event(
            event_type=EventTypes.PRESSURE_ACCUMULATED,
            payload={
                "type": pressure_type,
                "intensity": adjusted_intensity,
                "source": source,
                "identity_adjusted": True
            },
            causation_id=context.causal_metadata.event_id
        )
        events.append(pressure_event)
        
        # If pressure is high, might need identity adjustment
        if adjusted_intensity > 0.7:
            identity_adjustment = self._factory.create_identity_event(
                event_type=EventTypes.IDENTITY_MUTATED,
                payload={
                    "axis": "stability",
                    "delta": -0.05,  # Decrease stability under pressure
                    "trigger": "high_pressure"
                },
                causation_id=pressure_event.causal.event_id
            )
            events.append(identity_adjustment)
        
        return AdapterResult(
            success=True,
            events=tuple(events),
            external_state={"pressure_processed": True}
        )


class AdapterRegistry:
    """
    Registry for all adapters.
    
    Manages adapter lifecycle and routing.
    """
    
    def __init__(self):
        self._adapters: Dict[str, object] = {}
        self._register_default()
    
    def _register_default(self):
        """Register default adapters"""
        self._adapters["goal_execution"] = GoalExecutionAdapter()
        self._adapters["belief_collection"] = BeliefCollectionAdapter()
        self._adapters["pressure_update"] = PressureUpdateAdapter()
    
    def get(self, name: str) -> Optional[object]:
        """Get adapter by name"""
        return self._adapters.get(name)
    
    def register(self, name: str, adapter: object):
        """Register adapter"""
        self._adapters[name] = adapter
    
    def execute_adapter(
        self,
        adapter_name: str,
        context: ExecutionContext,
        data: Dict[str, Any]
    ) -> AdapterResult:
        """Execute adapter with given context and data"""
        adapter = self.get(adapter_name)
        if not adapter:
            return AdapterResult(
                success=False,
                events=(),
                external_state={},
                error=f"Adapter not found: {adapter_name}"
            )
        
        try:
            if adapter_name == "goal_execution":
                return adapter.execute_goal(context, data)
            elif adapter_name == "belief_collection":
                return adapter.process_evidence(context, data)
            elif adapter_name == "pressure_update":
                return adapter.process_pressure(context, data)
            else:
                return AdapterResult(
                    success=False,
                    events=(),
                    external_state={},
                    error=f"Unknown adapter operation: {adapter_name}"
                )
        except Exception as e:
            return AdapterResult(
                success=False,
                events=(),
                external_state={},
                error=str(e)
            )