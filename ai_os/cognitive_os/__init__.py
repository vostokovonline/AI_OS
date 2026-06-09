"""
AI-OS Cognitive OS - Complete Module

Phase 8: Unified Policy + Simulation Core

Modules:
- cognitive_os.py - Main orchestrator (CognitiveOS)
- growth_layer/ - Meta-cognition, bias detection, self-improvement
  - growth.py - Core growth layer
  - strategy_evolution.py - Strategy mutation and selection
  - self_narrative.py - Identity continuity and motivation
- world_model/ - Environment simulation and state tracking
  - model.py - World model implementation
- policy/ - Unified decision layer
  - state_builder.py - Unified latent state representation
  - policy_layer.py - policy(state) → action
  - unified_agent.py - Single entry point for model-based agent
- simulation/ - Planning loop
  - planner.py - Simulate → evaluate → select

Usage:
    from ai_os.cognitive_os import cognitive_os, UnifiedAgent, AgentContext

    # Initialize
    agent = UnifiedAgent(cognitive_os)

    # Make decision
    decision = await agent.decide(
        AgentContext(
            user_id="user123",
            task="write_code",
            complexity=0.6
        )
    )

    # Execute and learn
    await agent.execute_and_learn(decision, outcome="success")

    # Or use cognitive OS directly
    from ai_os.cognitive_os import CognitiveRequest
    result = await cognitive_os.process(
        CognitiveRequest(
            user_id="user123",
            context={"task": "write_code"}
        )
    )
"""
from .cognitive_os import CognitiveOS, cognitive_os, CognitiveRequest, CognitiveResponse
from .growth_layer.growth import GrowthLayer
from .growth_layer.strategy_evolution import StrategyEvolution
from .growth_layer.self_narrative import SelfNarrative
from .world_model.model import WorldModel
from .policy.state_builder import StateBuilder, UnifiedState
from .policy.policy_layer import PolicyLayer, PolicyAction, ActionType
from .policy.unified_agent import UnifiedAgent, AgentContext, AgentDecision, get_unified_agent

__all__ = [
    # Core
    "CognitiveOS",
    "cognitive_os",
    "CognitiveRequest",
    "CognitiveResponse",
    
    # Growth
    "GrowthLayer",
    "StrategyEvolution",
    "SelfNarrative",
    
    # World Model
    "WorldModel",
    
    # Policy Layer (Phase 8)
    "StateBuilder",
    "UnifiedState",
    "PolicyLayer",
    "PolicyAction",
    "ActionType",
    "UnifiedAgent",
    "AgentContext",
    "AgentDecision",
    "get_unified_agent",
]