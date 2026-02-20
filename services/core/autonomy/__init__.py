"""
AUTONOMY MODULE - Phase 1: Controlled MVP

This module implements the foundation for autonomous decision-making in AI-OS.

Components:
- SystemState: Central state management
- PolicyEngine: Rule-based decision making
- DecisionEngine: Action generation
- StateMutation: State change proposals from artifacts

Author: AI-OS Team
Date: 2026-02-20
Version: 1.0.0-mvp
"""
from autonomy.system_state import SystemStateManager, SystemStateEntity, EntityType
from autonomy.policy_engine import PolicyEngine, PolicyRule, ActionType
from autonomy.decision_engine import DecisionEngine, DecisionAction
from autonomy.state_mutation import StateMutation, MutationType
from autonomy.strategy import StrategyManager, Strategy, StrategyStatus
from autonomy.safety_constraints import SafetyConstraints, SafetyConstraint, SafetyViolation, ConstraintType

__all__ = [
    'SystemStateManager',
    'SystemStateEntity',
    'EntityType',
    'PolicyEngine',
    'PolicyRule',
    'ActionType',
    'DecisionEngine',
    'DecisionAction',
    'StateMutation',
    'MutationType',
    'StrategyManager',
    'Strategy',
    'StrategyStatus',
    'SafetyConstraints',
    'SafetyConstraint',
    'SafetyViolation',
    'ConstraintType',
]
