"""
Reducers - Pure event sourcing for cognitive state.

Provides:
- Unified reducer for all mutation types
- Deterministic apply functions
- No hidden side effects

Key principle:
    State does not exist independently from history.
    All mutations go through reducer.
"""
from .belief_reducer import BeliefReducer
from .causal_reducer import CausalReducer
from .contradiction_reducer import ContradictionReducer
from .root_reducer import RootReducer

__all__ = [
    "BeliefReducer",
    "CausalReducer", 
    "ContradictionReducer",
    "RootReducer"
]