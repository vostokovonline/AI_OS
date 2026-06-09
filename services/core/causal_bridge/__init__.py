"""
Causal Bridge Layer (CBL-1) — formal causality link between
Execution Kernel and Epistemic Kernel.

Architecture:
    CausalityBridge ──── CausalityGraph
         │                      │
         ├─ DualPropagator      ├─ CausalityEdges (exec ↔ epi)
         │    ├─ exec_to_epi    │
         │    └─ epi_to_exec    │
         │                      │
         └─ UnifiedReplayConsistency
              ├─ temporal ordering
              ├─ edge existence
              └─ acyclicity

Usage:
    from causal_bridge import CausalityBridge

    bridge = CausalityBridge(exec_kernel, epi_kernel)
    bridge.on_execution_completed(goal_id, entry_id, success=True)
    adjustments = bridge.get_dispatch_adjustments()
    consistency = bridge.check_causal_consistency()
"""

from .edge import (
    CausalityEdge,
    CausalityGraph,
    CausalDirection,
    InterpretationFrame,
    ExecutionEventType,
    EpistemicEventType,
)
from .propagation import (
    DualPropagator,
    ExecutionToEpistemicPropagator,
    EpistemicToExecutionPropagator,
)
from .consistency import UnifiedReplayConsistency, ReplayConsistencyError
from .bridge import CausalityBridge

__all__ = [
    'CausalityBridge',
    'CausalityEdge',
    'CausalityGraph',
    'CausalDirection',
    'InterpretationFrame',
    'ExecutionEventType',
    'EpistemicEventType',
    'DualPropagator',
    'ExecutionToEpistemicPropagator',
    'EpistemicToExecutionPropagator',
    'UnifiedReplayConsistency',
    'ReplayConsistencyError',
]
