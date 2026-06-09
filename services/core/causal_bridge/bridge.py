"""
CausalityBridge — single entry point connecting Execution Kernel and Epistemic Kernel.

The bridge:
  1. Holds references to both kernels
  2. Manages the CausalityGraph (all edges between them)
  3. Provides unified replay (replays both kernels + verifies consistency)
  4. Orchestrates dual propagation (exec→epi + epi→exec)

Usage:
    bridge = CausalityBridge(execution_kernel, epistemic_kernel)

    # Auto-propagate an execution completion
    bridge.on_execution_completed(
        goal_id="...",
        entry_id="...",
        lease_id="...",
        success=True,
    )

    # Get policy adjustments from epistemic state
    adjustments = bridge.get_dispatch_adjustments()

    # Unified replay
    result = bridge.unified_replay()

    # Check causal consistency
    consistency = bridge.check_causal_consistency()
"""

from typing import Any, Dict, List, Optional
import time
import logging

from .edge import CausalityGraph
from .propagation import (
    ExecutionToEpistemicPropagator,
    EpistemicToExecutionPropagator,
    DualPropagator,
)
from .consistency import UnifiedReplayConsistency

logger = logging.getLogger(__name__)


class CausalityBridge:
    """
    Orchestrator connecting Execution Kernel and Epistemic Kernel.

    This is the single integration point that makes the two kernels
    formally causally coupled.
    """

    def __init__(self, execution_kernel=None, epistemic_kernel=None):
        """
        Args:
            execution_kernel: ExecutionKernel instance
            epistemic_kernel: EpistemicKernel instance
        """
        self.execution_kernel = execution_kernel
        self.epistemic_kernel = epistemic_kernel

        # Causality graph — all edges between domains
        self.graph = CausalityGraph()

        # Propagators
        self.propagator = DualPropagator(self.graph)
        self._consistency = UnifiedReplayConsistency(self.graph)

        self._propagation_enabled = True
        self._adjustment_cache: Dict[str, Any] = {}
        self._adjustment_cache_epoch = -1
        self._total_propagations = 0

        logger.info("causality_bridge_initialized")

    # ------------------------------------------------------------------
    # Enable/disable propagation
    # ------------------------------------------------------------------

    @property
    def propagation_enabled(self) -> bool:
        return self._propagation_enabled

    def enable_propagation(self):
        self._propagation_enabled = True
        logger.info("causality_propagation_enabled")

    def disable_propagation(self):
        self._propagation_enabled = False
        logger.info("causality_propagation_disabled")

    # ------------------------------------------------------------------
    # Execution → Epistemic event hooks
    # ------------------------------------------------------------------

    def on_execution_event(
        self,
        event_type: str,
        entry_id: str,
        goal_id: str,
        lease_id: str = "",
        success: Optional[bool] = None,
        duration_ms: float = 0.0,
        error: Optional[str] = None,
        context: dict = None,
    ) -> Dict[str, Any]:
        """
        Hook called by execution kernel when a dispatch event occurs.

        Automatically propagates to epistemic kernel if enabled.
        """
        if not self._propagation_enabled:
            return {'propagated': False, 'reason': 'propagation_disabled'}

        if self.epistemic_kernel is None:
            return {'propagated': False, 'reason': 'no_epistemic_kernel'}

        result = self.propagator.propagate_execution_event(
            epistemic_kernel=self.epistemic_kernel,
            execution_event_type=event_type,
            execution_entry_id=entry_id,
            execution_goal_id=goal_id,
            execution_lease_id=lease_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            context=context,
        )

        self._total_propagations += 1
        self._invalidate_adjustment_cache()

        return result

    def on_execution_completed(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='COMPLETED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            success=success,
            duration_ms=duration_ms,
            context=context,
        )

    def on_execution_failed(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        error: str = "",
        duration_ms: float = 0.0,
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='FAILED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            success=False,
            duration_ms=duration_ms,
            error=error,
            context=context,
        )

    def on_execution_dispatched(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='DISPATCHED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            context=context,
        )

    def on_execution_preempted(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='PREEMPTED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            context=context,
        )

    def on_execution_cancelled(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='CANCELLED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            context=context,
        )

    def on_execution_retried(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='RETRIED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            context=context,
        )

    def on_execution_abandoned(
        self,
        goal_id: str,
        entry_id: str,
        lease_id: str = "",
        context: dict = None,
    ) -> Dict[str, Any]:
        return self.on_execution_event(
            event_type='ABANDONED',
            entry_id=entry_id,
            goal_id=goal_id,
            lease_id=lease_id,
            context=context,
        )

    # ------------------------------------------------------------------
    # Epistemic → Execution adjustments
    # ------------------------------------------------------------------

    def get_dispatch_adjustments(self) -> Dict[str, Any]:
        """
        Get current execution policy adjustments from epistemic state.

        Results are cached until the next execution event.
        """
        if self.epistemic_kernel is None:
            return {
                'priority_modifier': 0.0,
                'throttle_factor': 1.0,
                'persistence_bias': 0.0,
                'blocked': False,
                'drift_severity': 0.0,
            }

        current_epoch = self.epistemic_kernel.epoch.current
        if current_epoch == self._adjustment_cache_epoch and self._adjustment_cache:
            return dict(self._adjustment_cache)

        adjustments = self.propagator.epi_to_exec.compute_dispatch_adjustments(
            self.epistemic_kernel,
        )
        self._adjustment_cache = adjustments
        self._adjustment_cache_epoch = current_epoch

        return dict(adjustments)

    def _invalidate_adjustment_cache(self):
        self._adjustment_cache = {}
        self._adjustment_cache_epoch = -1

    # ------------------------------------------------------------------
    # Causality graph queries
    # ------------------------------------------------------------------

    def get_causal_chain(self, goal_id: str) -> List[Dict[str, Any]]:
        """Get full causal chain for a goal (execution + epistemic)."""
        edges = self.graph.get_edges_for_goal(goal_id)
        return [e.to_dict() for e in sorted(
            edges, key=lambda e: e.created_at
        )]

    def get_causal_stats(self) -> dict:
        """Get causality bridge statistics."""
        adjustments = self.get_dispatch_adjustments()
        return {
            'total_edges': self.graph.count(),
            'total_propagations': self._total_propagations,
            'propagation_enabled': self._propagation_enabled,
            'current_adjustments': adjustments,
        }

    # ------------------------------------------------------------------
    # Consistency verification
    # ------------------------------------------------------------------

    def check_causal_consistency(self) -> Dict[str, Any]:
        """
        Verify causal consistency between both kernels.

        Requires access to both kernels' journal entries.
        """
        if self.execution_kernel is None or self.epistemic_kernel is None:
            return {
                'passed': False,
                'violations': [{
                    'type': 'missing_kernel',
                    'detail': 'Both kernels must be set for consistency check',
                }],
                'total_edges': self.graph.count(),
                'valid_edges': 0,
            }

        # Collect execution journal entries
        exec_entries = set()
        exec_times = {}
        journal = getattr(self.execution_kernel, 'journal', None)
        if journal:
            for entry in journal._entries:
                exec_entries.add(entry.entry_id)
                exec_times[entry.entry_id] = entry.timestamp

        # Collect epistemic journal events
        epi_entries = set()
        epi_times = {}
        epi_journal = self.epistemic_kernel.journal
        for event in epi_journal._events:
            epi_entries.add(event.event_id)
            epi_times[event.event_id] = event.timestamp

        return self._consistency.verify(
            execution_journal_entries=exec_entries,
            epistemic_journal_events=epi_entries,
            execution_journal_times=exec_times,
            epistemic_journal_times=epi_times,
        )

    # ------------------------------------------------------------------
    # Unified replay
    # ------------------------------------------------------------------

    def unified_replay(self, state: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Unified replay — restores both kernels to a consistent state.

        Args:
            state: optional pre-captured state dict with 'execution' and 'epistemic' keys.
                   If None, captures current state from both kernels.

        Returns:
            dict with replay results
        """
        if self.execution_kernel is None or self.epistemic_kernel is None:
            return {
                'success': False,
                'error': 'Both kernels must be set for unified replay',
            }

        # Capture pre-replay state for validation
        pre_state = self._capture_state()

        # Replay epistemic kernel from journal
        epi_journal = self.epistemic_kernel.journal
        epi_events = epi_journal.replay()
        if epi_events:
            last_event = epi_events[-1]
            logger.info(
                f"epistemic_replay events={len(epi_events)} "
                f"last={last_event.event_id} epoch={last_event.epoch}"
            )

        # Replay execution kernel
        if hasattr(self.execution_kernel, 'recover'):
            recovery = self.execution_kernel.recover()
            logger.info(
                f"execution_replay status={recovery.get('status')} "
                f"leases={recovery.get('active_leases')}"
            )
        else:
            recovery = {'status': 'unknown'}

        # Validate
        post_state = self._capture_state()
        violations = self._consistency.validate_replay(pre_state, post_state)

        return {
            'success': len(violations) == 0,
            'epistemic_events_replayed': len(epi_events),
            'execution_recovery': recovery,
            'consistency_violations': violations,
        }

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current state from both kernels for replay validation."""
        state = {'beliefs': {}, 'epoch': 0}
        if self.epistemic_kernel:
            state['beliefs'] = self.epistemic_kernel.get_all_beliefs()
            state['epoch'] = self.epistemic_kernel.epoch.current
            state['motifs'] = self.epistemic_kernel.get_all_motifs()
        return state

    # ------------------------------------------------------------------
    # Link kernels
    # ------------------------------------------------------------------

    def set_execution_kernel(self, kernel):
        self.execution_kernel = kernel
        logger.info("causality_bridge_execution_kernel_set")

    def set_epistemic_kernel(self, kernel):
        self.epistemic_kernel = kernel
        logger.info("causality_bridge_epistemic_kernel_set")
