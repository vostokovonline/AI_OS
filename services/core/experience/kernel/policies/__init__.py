"""
Cognitive Policies - Decision dynamics from materialized state.

Stage: Four-Layer Architecture

Policies take materialized state and decide actions.
They are NOT reducers - they don't mutate state.
They are pure functions: (state, context) → decision

Key difference:
- Reducers: events → state (materialization)
- Policies: state → decisions (intelligence)

Policies enable:
- Identity-driven decisions
- Pressure-aware planning
- Self-modifying behavior
- Emergent strategy
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json

from reducers import CognitiveState, ExecutionState, IdentityState
from event_log import ImmutableEvent, EventTypes


class DecisionType(Enum):
    """Types of decisions"""
    STRATEGY = "strategy"          # Which approach to use
    PRIORITY = "priority"         # What to do first
    RISK = "risk"                 # How much risk to take
    DECOMPOSITION = "decomposition"  # How to break down goal
    VERIFICATION = "verification"  # How strict to verify
    RETRY = "retry"               # How to handle failures
    EXPLORATION = "exploration"    # How much to explore vs exploit


@dataclass(frozen=True)
class PolicyDecision:
    """
    A decision made by policy engine.
    
    Contains:
    - type: what kind of decision
    - value: the decision value
    - confidence: how certain
    - reasoning: why this decision
    - influenced_by: what factors influenced
    """
    decision_type: str
    value: Any
    confidence: float  # 0-1
    reasoning: str
    influenced_by: Tuple[str, ...]  # Factor IDs
    timestamp: str
    
    def is_confident(self) -> bool:
        return self.confidence >= 0.7
    
    def is_uncertain(self) -> bool:
        return self.confidence < 0.4


@dataclass(frozen=True)
class PolicyResult:
    """
    Result of policy evaluation.
    
    Contains all decisions made for current state.
    """
    decisions: Tuple[PolicyDecision, ...]
    total_pressure: float
    dominant_strategy: str
    risk_appetite: float
    exploration_ratio: float
    version: int


# Identity-driven policies

def identity_strategy_policy(identity: IdentityState, pressure: float) -> PolicyDecision:
    """
    Determine strategy based on identity axes.
    
    High autonomy → more exploration
    High stability → more caution
    High curiosity → more experimentation
    
    Pressure modifies behavior:
    - High pressure → more conservative
    - Low pressure → more exploratory
    """
    autonomy = identity.axes.get("autonomy", 0.5)
    curiosity = identity.axes.get("curiosity", 0.5)
    stability = identity.axes.get("stability", 0.5)
    coherence = identity.axes.get("coherence", 0.5)
    
    # Base strategy from identity
    if autonomy > 0.7 and curiosity > 0.7:
        base_strategy = "explorative"
    elif stability > 0.7 and coherence > 0.7:
        base_strategy = "conservative"
    else:
        base_strategy = "balanced"
    
    # Modify based on pressure
    if pressure > 0.7:
        # High pressure → more conservative
        strategy = "conservative"
        confidence = 0.9
        reasoning = f"High pressure ({pressure:.1f}) triggers conservative approach"
    elif pressure < 0.3:
        # Low pressure → more exploratory
        strategy = "explorative"
        confidence = 0.8
        reasoning = f"Low pressure ({pressure:.1f}) enables exploration"
    else:
        strategy = base_strategy
        confidence = 0.7
        reasoning = f"Moderate pressure, using identity-based strategy"
    
    return PolicyDecision(
        decision_type=DecisionType.STRATEGY.value,
        value=strategy,
        confidence=confidence,
        reasoning=reasoning,
        influenced_by=("autonomy", "curiosity", "stability", "pressure"),
        timestamp=datetime.utcnow().isoformat()
    )


def identity_risk_policy(identity: IdentityState, cognitive_state: CognitiveState) -> PolicyDecision:
    """
    Determine risk appetite based on identity.
    
    High autonomy → higher risk tolerance
    High stability → lower risk tolerance
    Many contradictions → higher risk (must resolve)
    """
    autonomy = identity.axes.get("autonomy", 0.5)
    stability = identity.axes.get("stability", 0.5)
    
    contradiction_count = len(cognitive_state.contradictions)
    high_pressure = cognitive_state.pressures.get("total", {}).get("intensity", 0.0)
    
    # Base risk from identity
    base_risk = autonomy * 0.6 + (1 - stability) * 0.4
    
    # Adjust for contradictions (must take risk to resolve)
    if contradiction_count > 0:
        risk_modifier = 0.1 * contradiction_count
        base_risk = min(1.0, base_risk + risk_modifier)
    
    # Adjust for pressure
    if high_pressure > 0.5:
        base_risk *= 0.7  # Reduce risk under pressure
    
    risk_appetite = max(0.1, min(1.0, base_risk))
    
    reasoning = f"Risk appetite: {risk_appetite:.2f} (autonomy={autonomy:.2f}, stability={stability:.2f})"
    if contradiction_count > 0:
        reasoning += f", {contradiction_count} contradictions"
    
    return PolicyDecision(
        decision_type=DecisionType.RISK.value,
        value=risk_appetite,
        confidence=0.75,
        reasoning=reasoning,
        influenced_by=("autonomy", "stability", "contradictions"),
        timestamp=datetime.utcnow().isoformat()
    )


def identity_decomposition_policy(identity: IdentityState, goal_complexity: float) -> PolicyDecision:
    """
    Determine how to decompose goals based on identity.
    
    High coherence → prefer hierarchical decomposition
    High curiosity → prefer exploratory decomposition
    High autonomy → prefer autonomous subgoals
    """
    coherence = identity.axes.get("coherence", 0.5)
    autonomy = identity.axes.get("autonomy", 0.5)
    curiosity = identity.axes.get("curiosity", 0.5)
    
    # More complex goals need more structure
    if goal_complexity > 0.7:
        if coherence > 0.6:
            style = "hierarchical"
            depth = 3
        else:
            style = "parallel"
            depth = 2
    else:
        style = "simple"
        depth = 1
    
    # Adjust for identity
    if autonomy > 0.7:
        # More autonomy → larger subgoals (less decomposition)
        depth = max(1, depth - 1)
    
    reasoning = f"Decomposition style: {style}, depth: {depth} (complexity={goal_complexity:.2f})"
    
    return PolicyDecision(
        decision_type=DecisionType.DECOMPOSITION.value,
        value={"style": style, "max_depth": depth},
        confidence=0.7,
        reasoning=reasoning,
        influenced_by=("coherence", "autonomy", "curiosity"),
        timestamp=datetime.utcnow().isoformat()
    )


def identity_verification_policy(identity: IdentityState, cognitive_state: CognitiveState) -> PolicyDecision:
    """
    Determine verification strictness based on identity.
    
    High stability → stricter verification
    High pressure → looser verification (time pressure)
    Many contradictions → stricter (prevent accumulation)
    """
    stability = identity.axes.get("stability", 0.5)
    pressure = sum(p.get("intensity", 0) for p in cognitive_state.pressures.values()) / max(1, len(cognitive_state.pressures))
    
    contradiction_count = len(cognitive_state.contradictions)
    
    # Base strictness from identity
    base_strictness = stability * 0.8 + 0.2
    
    # Adjust for pressure (relax under pressure)
    if pressure > 0.6:
        base_strictness *= 0.7
    
    # Adjust for contradictions (be stricter to prevent accumulation)
    if contradiction_count > 2:
        base_strictness = min(1.0, base_strictness * 1.2)
    
    strictness = max(0.1, min(1.0, base_strictness))
    
    reasoning = f"Verification strictness: {strictness:.2f} (stability={stability:.2f})"
    if pressure > 0.6:
        reasoning += ", relaxed due to pressure"
    if contradiction_count > 2:
        reasoning += f", increased due to {contradiction_count} contradictions"
    
    return PolicyDecision(
        decision_type=DecisionType.VERIFICATION.value,
        value=strictness,
        confidence=0.7,
        reasoning=reasoning,
        influenced_by=("stability", "pressure", "contradictions"),
        timestamp=datetime.utcnow().isoformat()
    )


def identity_retry_policy(identity: IdentityState, failure_history: List[str]) -> PolicyDecision:
    """
    Determine retry behavior based on identity.
    
    High autonomy → more retries (independent)
    High stability → fewer retries (avoid loops)
    Many recent failures → adapt behavior
    """
    autonomy = identity.axes.get("autonomy", 0.5)
    stability = identity.axes.get("stability", 0.5)
    
    recent_failures = len(failure_history)
    
    # Base retry count from identity
    base_retries = int(autonomy * 5 + 1)  # 1-6 retries
    
    # Adjust for stability (reduce if stable)
    if stability > 0.7:
        base_retries = max(1, base_retries - 1)
    
    # Adjust for failure history
    if recent_failures > 3:
        base_retries = max(1, base_retries - 2)
    
    max_retries = max(1, min(6, base_retries))
    
    reasoning = f"Max retries: {max_retries} (autonomy={autonomy:.2f}, stability={stability:.2f})"
    if recent_failures > 3:
        reasoning += f", reduced due to {recent_failures} recent failures"
    
    return PolicyDecision(
        decision_type=DecisionType.RETRY.value,
        value=max_retries,
        confidence=0.7,
        reasoning=reasoning,
        influenced_by=("autonomy", "stability", "failure_history"),
        timestamp=datetime.utcnow().isoformat()
    )


# Priority policies

def pressure_priority_policy(cognitive_state: CognitiveState) -> List[PolicyDecision]:
    """
    Determine goal priorities based on pressures.
    
    Higher pressure → higher priority
    Escalated contradictions → critical priority
    """
    decisions = []
    
    total_pressure = sum(p.get("intensity", 0) for p in cognitive_state.pressures.values())
    
    # Check for escalated contradictions
    escalated = [c for c in cognitive_state.contradictions.values() 
                 if c.get("status") == "escalated"]
    
    if escalated:
        priority = "critical"
        confidence = 0.95
        reasoning = f"Critical priority due to {len(escalated)} escalated contradictions"
    elif total_pressure > 0.7:
        priority = "high"
        confidence = 0.85
        reasoning = f"High priority due to total pressure {total_pressure:.2f}"
    elif total_pressure > 0.4:
        priority = "medium"
        confidence = 0.7
        reasoning = f"Medium priority, moderate pressure {total_pressure:.2f}"
    else:
        priority = "normal"
        confidence = 0.6
        reasoning = f"Normal priority, low pressure {total_pressure:.2f}"
    
    decisions.append(PolicyDecision(
        decision_type=DecisionType.PRIORITY.value,
        value=priority,
        confidence=confidence,
        reasoning=reasoning,
        influenced_by=("pressures", "contradictions"),
        timestamp=datetime.utcnow().isoformat()
    ))
    
    return decisions


def exploration_exploitation_policy(identity: IdentityState, cognitive_state: CognitiveState) -> PolicyDecision:
    """
    Balance exploration vs exploitation.
    
    High curiosity → more exploration
    High coherence → more exploitation (refine known)
    Many contradictions → more exploration (need new data)
    """
    curiosity = identity.axes.get("curiosity", 0.5)
    coherence = identity.axes.get("coherence", 0.5)
    
    contradiction_count = len(cognitive_state.contradictions)
    
    # Base ratio from identity
    exploration_ratio = curiosity * 0.6 + (1 - coherence) * 0.4
    
    # Adjust for contradictions (need exploration to resolve)
    if contradiction_count > 0:
        exploration_ratio = min(1.0, exploration_ratio + contradiction_count * 0.05)
    
    ratio = max(0.1, min(0.9, exploration_ratio))
    
    reasoning = f"Exploration ratio: {ratio:.2f} (curiosity={curiosity:.2f}, coherence={coherence:.2f})"
    if contradiction_count > 0:
        reasoning += f", increased due to {contradiction_count} contradictions"
    
    return PolicyDecision(
        decision_type=DecisionType.EXPLORATION.value,
        value=ratio,
        confidence=0.7,
        reasoning=reasoning,
        influenced_by=("curiosity", "coherence", "contradictions"),
        timestamp=datetime.utcnow().isoformat()
    )


# Main policy evaluation

def evaluate_all_policies(
    identity: IdentityState,
    cognitive_state: CognitiveState,
    execution_state: Optional[ExecutionState] = None,
    context: Optional[Dict[str, Any]] = None
) -> PolicyResult:
    """
    Evaluate all policies and return combined result.
    
    This is the main entry point for policy-driven decisions.
    """
    decisions = []
    
    # Get pressure level
    total_pressure = sum(p.get("intensity", 0) for p in cognitive_state.pressures.values())
    if cognitive_state.pressures:
        total_pressure /= len(cognitive_state.pressures)
    
    # Evaluate identity-driven policies
    decisions.append(identity_strategy_policy(identity, total_pressure))
    decisions.append(identity_risk_policy(identity, cognitive_state))
    
    # Context-specific policies
    goal_complexity = context.get("goal_complexity", 0.5) if context else 0.5
    decisions.append(identity_decomposition_policy(identity, goal_complexity))
    decisions.append(identity_verification_policy(identity, cognitive_state))
    
    failure_history = context.get("failure_history", []) if context else []
    decisions.append(identity_retry_policy(identity, failure_history))
    
    # Priority and exploration
    decisions.extend(pressure_priority_policy(cognitive_state))
    decisions.append(exploration_exploitation_policy(identity, cognitive_state))
    
    # Extract dominant strategy
    strategy_decision = next((d for d in decisions if d.decision_type == DecisionType.STRATEGY.value), None)
    dominant_strategy = strategy_decision.value if strategy_decision else "balanced"
    
    # Extract risk appetite
    risk_decision = next((d for d in decisions if d.decision_type == DecisionType.RISK.value), None)
    risk_appetite = risk_decision.value if risk_decision else 0.5
    
    # Extract exploration ratio
    exploration_decision = next((d for d in decisions if d.decision_type == DecisionType.EXPLORATION.value), None)
    exploration_ratio = exploration_decision.value if exploration_decision else 0.3
    
    return PolicyResult(
        decisions=tuple(decisions),
        total_pressure=total_pressure,
        dominant_strategy=dominant_strategy,
        risk_appetite=risk_appetite,
        exploration_ratio=exploration_ratio,
        version=identity.version
    )


def apply_policy_to_execution(
    policy: PolicyResult,
    goal: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply policy decisions to goal execution.
    
    Modifies goal parameters based on policy.
    """
    modified_goal = dict(goal)
    
    # Apply strategy
    if policy.dominant_strategy == "explorative":
        modified_goal["allow_subgoal_failure"] = True
        modified_goal["decomposition_depth"] = 2
    elif policy.dominant_strategy == "conservative":
        modified_goal["allow_subgoal_failure"] = False
        modified_goal["decomposition_depth"] = 3
        modified_goal["verification_strictness"] = 0.8
    
    # Apply risk appetite
    modified_goal["risk_tolerance"] = policy.risk_appetite
    
    # Apply exploration ratio
    if policy.exploration_ratio > 0.5:
        modified_goal["prefer_alternatives"] = True
    else:
        modified_goal["prefer_tested_approaches"] = True
    
    return modified_goal


def explain_decision(decision: PolicyDecision) -> str:
    """
    Generate human-readable explanation of decision.
    """
    lines = [
        f"Decision: {decision.decision_type}",
        f"Value: {decision.value}",
        f"Confidence: {decision.confidence:.2f}",
        f"Reasoning: {decision.reasoning}",
        f"Influenced by: {', '.join(decision.influenced_by)}",
        f"At: {decision.timestamp}"
    ]
    return "\n".join(lines)