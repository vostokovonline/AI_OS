"""
Dual propagation rules — Execution → Epistemic and Epistemic → Execution.

This is the engine that makes the two kernels actively influence each other.

Rules:

  Execution → Epistemic (exec_to_epi):
    COMPLETED   → belief update (success_ratio + confidence up)
    FAILED      → belief update (failure/skill_gap confidence up)
    PREEMPTED   → motif update (preemption pattern reinforced)
    CANCELLED   → belief update (uncertainty confidence up)
    RETRIED     → motif update (persistence pattern)
    DISPATCHED  → observation (activity signal, no belief change)

  Epistemic → Execution (epi_to_exec):
    belief_drift     → throttle dispatch (reduce max_concurrent, increase lease TTL)
    uncertainty      → reduce dispatch priority (wait for more evidence)
    motif_strength   → bias policy selection (preferred strategies)
    attractor_weight → increase persistence_weight (stable paths)
    drift_severity   → block dispatch if above threshold
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import logging

from .edge import (
    CausalityEdge, CausalityGraph, CausalDirection,
    InterpretationFrame, ExecutionEventType,
    _next_edge_id,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# DEFAULT INTERPRETATION FRAMES  (execution → epistemic)
# ------------------------------------------------------------------

DEFAULT_FRAMES = {
    ExecutionEventType.COMPLETED: InterpretationFrame(
        interpretation="execution_succeeded",
        belief_delta={'execution_effectiveness': 0.05},
        confidence_delta=0.02,
        tags=['success', 'positive_outcome'],
    ),
    ExecutionEventType.FAILED: InterpretationFrame(
        interpretation="execution_failed",
        belief_delta={'failure_rate': 0.08},
        confidence_delta=-0.05,
        tags=['failure', 'negative_outcome'],
    ),
    ExecutionEventType.PREEMPTED: InterpretationFrame(
        interpretation="execution_preempted",
        motif_delta={'preemption_pattern': 0.1},
        confidence_delta=-0.03,
        tags=['preemption', 'priority_conflict'],
    ),
    ExecutionEventType.CANCELLED: InterpretationFrame(
        interpretation="execution_cancelled",
        belief_delta={'execution_uncertainty': 0.06},
        confidence_delta=-0.04,
        tags=['cancellation', 'uncertainty'],
    ),
    ExecutionEventType.RETRIED: InterpretationFrame(
        interpretation="execution_retried",
        motif_delta={'persistence_pattern': 0.07},
        confidence_delta=0.0,
        tags=['retry', 'persistence'],
    ),
    ExecutionEventType.ABANDONED: InterpretationFrame(
        interpretation="execution_abandoned",
        belief_delta={'failure_rate': 0.15, 'skill_gap': 0.1},
        confidence_delta=-0.1,
        tags=['abandoned', 'failure', 'skill_gap'],
    ),
    ExecutionEventType.DISPATCHED: InterpretationFrame(
        interpretation="execution_dispatched",
        belief_delta={},
        confidence_delta=0.0,
        tags=['dispatch', 'activity'],
    ),
}


# ------------------------------------------------------------------
# EXECUTION → EPISTEMIC PROPAGATOR
# ------------------------------------------------------------------

class ExecutionToEpistemicPropagator:
    """
    Converts execution events into epistemic observations and belief updates.

    For every journal entry, creates:
      1. An observation in the epistemic journal
      2. A causality edge linking the execution event to the epistemic event
      3. Optional belief/motif updates based on the event type
    """

    def __init__(self, graph: CausalityGraph, frames: Dict = None):
        self._graph = graph
        self._frames = frames or DEFAULT_FRAMES

    def propagate(
        self,
        epistemic_kernel,
        execution_event_type: str,
        execution_entry_id: str,
        execution_goal_id: str,
        execution_lease_id: str = "",
        success: Optional[bool] = None,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        context: dict = None,
    ) -> List[CausalityEdge]:
        """
        Propagate an execution event into the epistemic kernel.

        Args:
            epistemic_kernel: EpistemicKernel instance
            execution_event_type: one of DISPATCH_EVENTS
            execution_entry_id: JournalEntry.entry_id
            execution_goal_id: goal_id
            execution_lease_id: lease_id
            success: did the execution succeed (if applicable)
            duration_ms: how long it took
            error: error message (if failed)
            context: additional context

        Returns:
            list of CausalityEdge created
        """
        context = context or {}
        created_edges = []

        try:
            event_type = ExecutionEventType(execution_event_type)
        except ValueError:
            logger.warning(f"unknown_execution_event type={execution_event_type}")
            return []

        frame = self._frames.get(event_type)
        if frame is None:
            frame = InterpretationFrame(
                interpretation=f"execution_{execution_event_type.lower()}",
                tags=[execution_event_type.lower()],
            )

        # 1. Create observation in epistemic kernel
        observation_id = epistemic_kernel.record_observation(
            signal=f"execution:{execution_event_type.lower()}",
            value=1.0 if success else 0.5,
            source="execution_kernel",
            context={
                'goal_id': execution_goal_id,
                'entry_id': execution_entry_id,
                'event_type': execution_event_type,
                'success': success,
                'duration_ms': duration_ms,
                'error': error,
                **context,
            },
        )

        # 2. Create causality edge
        edge = CausalityEdge(
            edge_id=_next_edge_id(),
            direction=CausalDirection.EXECUTION_TO_EPISTEMIC,
            execution_entry_id=execution_entry_id,
            execution_event_type=execution_event_type,
            execution_goal_id=execution_goal_id,
            execution_lease_id=execution_lease_id,
            epistemic_event_id=observation_id,
            epistemic_event_type='OBSERVATION',
            interpretation=frame,
            confidence=frame.confidence_delta + 1.0,
            causal_strength=abs(frame.confidence_delta) + 0.5,
            context={
                'success': success,
                'duration_ms': duration_ms,
                'error': error,
            },
        )
        self._graph.add_edge(edge)
        created_edges.append(edge)

        # 3. Apply belief deltas from frame
        for belief_name, delta in frame.belief_delta.items():
            current = epistemic_kernel.get_belief(belief_name)
            new_conf = max(0.0, min(1.0, current['confidence'] + delta))
            epistemic_kernel.update_belief(
                name=belief_name,
                confidence=new_conf,
                provenance=f"execution:{execution_event_type.lower()}",
                source_event_id=observation_id,
            )

        # 4. Apply motif deltas from frame
        for motif_name, delta in frame.motif_delta.items():
            current = epistemic_kernel.get_motif(motif_name)
            new_strength = max(0.0, min(1.0, current['strength'] + delta))
            epistemic_kernel.update_motif(
                name=motif_name,
                strength=new_strength,
                recurrence=current.get('recurrence', 0) + 1,
                provenance=f"execution:{execution_event_type.lower()}",
            )

        if frame.belief_delta or frame.motif_delta:
            logger.info(
                f"exec_to_epi event={execution_event_type} "
                f"goal={execution_goal_id} "
                f"beliefs={list(frame.belief_delta.keys())} "
                f"motifs={list(frame.motif_delta.keys())}"
            )

        return created_edges


# ------------------------------------------------------------------
# EPISTEMIC → EXECUTION PROPAGATOR
# ------------------------------------------------------------------

class EpistemicToExecutionPropagator:
    """
    Converts epistemic state changes into execution policy adjustments.

    Belief state influences:
      - dispatch priority (uncertainty → lower priority)
      - throttling (drift → rate limit)
      - policy weights (attractors → preference)
      - blocking (drift above threshold → block dispatch)
    """

    def __init__(self, graph: CausalityGraph):
        self._graph = graph

    def compute_dispatch_adjustments(
        self,
        epistemic_kernel,
    ) -> Dict[str, Any]:
        """
        Compute execution policy adjustments from current epistemic state.

        Returns:
            dict with keys:
              - priority_modifier: float (-0.5 to 0.5)
              - throttle_factor: float (0.0 to 1.0)
              - persistence_bias: float (-0.3 to 0.3)
              - blocked: bool
              - block_reason: str
              - drift_severity: float
        """
        adjustments = {
            'priority_modifier': 0.0,
            'throttle_factor': 1.0,
            'persistence_bias': 0.0,
            'blocked': False,
            'block_reason': '',
            'drift_severity': 0.0,
        }

        # Drift check
        drift = epistemic_kernel.check_drift()
        adjustments['drift_severity'] = drift.overall_drift_score

        if drift.overall_drift_score > 0.7:
            # Severe drift — block dispatch
            adjustments['blocked'] = True
            adjustments['block_reason'] = (
                f"semantic drift too high ({drift.overall_drift_score:.2f})"
            )
            adjustments['throttle_factor'] = 0.0
            return adjustments

        if drift.overall_drift_score > 0.4:
            # Moderate drift — throttle
            adjustments['throttle_factor'] = max(
                0.0, 1.0 - drift.overall_drift_score
            )
            adjustments['priority_modifier'] = -drift.overall_drift_score * 0.3

        # Belief-based adjustments
        uncertainty = epistemic_kernel.get_belief('execution_uncertainty')
        if uncertainty['confidence'] > 0.5:
            adjustments['priority_modifier'] -= uncertainty['confidence'] * 0.2
            adjustments['throttle_factor'] = min(
                adjustments['throttle_factor'],
                1.0 - uncertainty['confidence'] * 0.3,
            )

        effectiveness = epistemic_kernel.get_belief('execution_effectiveness')
        if effectiveness['confidence'] > 0.5:
            adjustments['priority_modifier'] += effectiveness['confidence'] * 0.1

        failure_rate = epistemic_kernel.get_belief('failure_rate')
        if failure_rate['confidence'] > 0.4:
            adjustments['priority_modifier'] -= failure_rate['confidence'] * 0.2
            adjustments['persistence_bias'] = -failure_rate['confidence'] * 0.2

        # Attractor-based adjustments
        for att_id, att in epistemic_kernel.get_all_attractors().items():
            if att['weight'] > 0.6:
                adjustments['persistence_bias'] += att['weight'] * 0.05

        # Clamp
        adjustments['priority_modifier'] = max(
            -0.5, min(0.5, adjustments['priority_modifier'])
        )
        adjustments['persistence_bias'] = max(
            -0.3, min(0.3, adjustments['persistence_bias'])
        )
        adjustments['throttle_factor'] = max(
            0.0, min(1.0, adjustments['throttle_factor'])
        )

        return adjustments

    def log_adjustments(
        self,
        epistemic_kernel,
        adjustments: Dict[str, Any],
    ) -> Optional[CausalityEdge]:
        """
        Log epistemic → execution adjustments as a causality edge.
        """
        event_id = epistemic_kernel.journal.append_event(
            event_type='EPISTEMIC_TO_EXECUTION',
            detail=(
                f"adjustments: priority_mod={adjustments['priority_modifier']:.2f} "
                f"throttle={adjustments['throttle_factor']:.2f} "
                f"blocked={adjustments['blocked']}"
            ),
            epoch=epistemic_kernel.epoch.current,
        )

        if not adjustments.get('_skip_edge'):
            edge = CausalityEdge(
                edge_id=_next_edge_id(),
                direction=CausalDirection.EPISTEMIC_TO_EXECUTION,
                execution_entry_id='',
                execution_event_type='POLICY_ADJUSTMENT',
                execution_goal_id='system',
                execution_lease_id='',
                epistemic_event_id=event_id.event_id,
                epistemic_event_type='EPISTEMIC_TO_EXECUTION',
                interpretation=InterpretationFrame(
                    interpretation="epistemic_state_influenced_execution_policy",
                    confidence_delta=adjustments['drift_severity'],
                    tags=['policy_adjustment', 'feedback_loop'],
                ),
                confidence=max(0.1, 1.0 - adjustments['drift_severity']),
                causal_strength=abs(adjustments['priority_modifier']) + 0.3,
            )
            self._graph.add_edge(edge)
            return edge

        return None


# ------------------------------------------------------------------
# DUAL PROPAGATOR (convenience)
# ------------------------------------------------------------------

class DualPropagator:
    """
    Dual-direction propagator that handles both Execution→Epistemic
    and Epistemic→Execution propagation in a single call.

    This is the primary entry point for causal feedback loops.
    """

    def __init__(self, graph: CausalityGraph):
        self.graph = graph
        self.exec_to_epi = ExecutionToEpistemicPropagator(graph)
        self.epi_to_exec = EpistemicToExecutionPropagator(graph)

    def propagate_execution_event(
        self,
        epistemic_kernel,
        execution_event_type: str,
        execution_entry_id: str,
        execution_goal_id: str,
        execution_lease_id: str = "",
        success: Optional[bool] = None,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        context: dict = None,
    ) -> Dict[str, Any]:
        """
        Full propagation cycle:
          1. Execution → Epistemic (observation + belief updates)
          2. Epistemic → Execution (compute policy adjustments)

        Returns:
            dict with 'edges' (created causality edges) and 'adjustments'
        """
        edges = self.exec_to_epi.propagate(
            epistemic_kernel=epistemic_kernel,
            execution_event_type=execution_event_type,
            execution_entry_id=execution_entry_id,
            execution_goal_id=execution_goal_id,
            execution_lease_id=execution_lease_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            context=context,
        )

        adjustments = self.epi_to_exec.compute_dispatch_adjustments(
            epistemic_kernel,
        )

        adj_edge = self.epi_to_exec.log_adjustments(
            epistemic_kernel, adjustments,
        )

        return {
            'edges': edges,
            'adjustments': adjustments,
            'adjustment_edge': adj_edge,
        }
