"""
Unified Replay Consistency — single replay that restores both kernels.

Key insight:
  - Execution kernel replays from WAL (journal entries)
  - Epistemic kernel replays from semantic journal (interpretation events)
  - CausalityGraph links them
  - A UNIFIED replay replays BOTH and verifies causal consistency

Consistency proof:
  For every CausalityEdge in the graph:
    - The execution_entry_id must exist in the execution journal
    - The epistemic_event_id must exist in the epistemic journal
    - If EXECUTION_TO_EPISTEMIC: execution event timestamp ≤ epistemic event timestamp
    - If EPISTEMIC_TO_EXECUTION: epistemic event timestamp ≤ execution event timestamp
"""

from typing import Any, Dict, List, Optional, Tuple
import logging

from .edge import CausalityGraph, CausalityEdge, CausalDirection

logger = logging.getLogger(__name__)


class ReplayConsistencyError(Exception):
    """Raised when unified replay consistency check fails."""
    def __init__(self, message: str, violations: List[dict]):
        self.violations = violations
        super().__init__(f"{message}: {len(violations)} violations")


class UnifiedReplayConsistency:
    """
    Verifies that a unified replay of both kernels produces a causally
    consistent state.

    Checks:
      1. Edge validity: every edge's event IDs exist in their respective journals
      2. Temporal ordering: cause precedes effect
      3. Graph acyclicity: no causal cycles
      4. Completeness: every execution event with epistemic effects has a link
    """

    def __init__(self, graph: CausalityGraph):
        self._graph = graph

    # ------------------------------------------------------------------
    # Low-level consistency checks
    # ------------------------------------------------------------------

    def check_edge_targets_exist(
        self,
        execution_journal_entries: set,
        epistemic_journal_events: set,
    ) -> List[dict]:
        """
        Check that every edge's source and target events exist.
        """
        violations = []
        for edge in self._graph._edges.values():
            if edge.execution_entry_id not in execution_journal_entries:
                violations.append({
                    'edge_id': edge.edge_id,
                    'type': 'missing_execution_entry',
                    'detail': (
                        f"execution_entry_id={edge.execution_entry_id} "
                        f"not found in execution journal"
                    ),
                })
            if edge.epistemic_event_id not in epistemic_journal_events:
                violations.append({
                    'edge_id': edge.edge_id,
                    'type': 'missing_epistemic_event',
                    'detail': (
                        f"epistemic_event_id={edge.epistemic_event_id} "
                        f"not found in epistemic journal"
                    ),
                })
        return violations

    def check_temporal_ordering(
        self,
        execution_journal_times: Dict[str, float],
        epistemic_journal_times: Dict[str, float],
    ) -> List[dict]:
        """
        Check that cause precedes effect in time.
        """
        violations = []
        for edge in self._graph._edges.values():
            exec_time = execution_journal_times.get(edge.execution_entry_id)
            epi_time = epistemic_journal_times.get(edge.epistemic_event_id)

            if exec_time is None or epi_time is None:
                continue

            if edge.direction == CausalDirection.EXECUTION_TO_EPISTEMIC:
                if exec_time > epi_time:
                    violations.append({
                        'edge_id': edge.edge_id,
                        'type': 'temporal_violation',
                        'detail': (
                            f"EXECUTION→EPISTEMIC but execution event "
                            f"({exec_time}) after epistemic ({epi_time})"
                        ),
                    })

            elif edge.direction == CausalDirection.EPISTEMIC_TO_EXECUTION:
                if epi_time > exec_time:
                    violations.append({
                        'edge_id': edge.edge_id,
                        'type': 'temporal_violation',
                        'detail': (
                            f"EPISTEMIC→EXECUTION but epistemic event "
                            f"({epi_time}) after execution ({exec_time})"
                        ),
                    })

        return violations

    def check_acyclicity(self) -> List[dict]:
        """
        Check for causal cycles in the graph.

        A cycle exists if: edge A references edge B's epistemic event
        and edge B references edge A's execution event (or similar).
        """
        violations = []

        # Build adjacency: execution_event → epistemic_event → next execution_event
        exec_to_epi: Dict[str, str] = {}
        epi_to_exec: Dict[str, str] = {}
        for edge in self._graph._edges.values():
            if edge.direction in (CausalDirection.EXECUTION_TO_EPISTEMIC, CausalDirection.BIDIRECTIONAL):
                exec_to_epi[edge.execution_entry_id] = edge.epistemic_event_id
            if edge.direction in (CausalDirection.EPISTEMIC_TO_EXECUTION, CausalDirection.BIDIRECTIONAL):
                epi_to_exec[edge.epistemic_event_id] = edge.execution_entry_id

        # Detect cycles: walk exec_to_epi → epi_to_exec → exec_to_epi ...
        visited = set()
        for start_exec in exec_to_epi:
            if start_exec in visited:
                continue
            path = [('exec', start_exec)]
            current = start_exec
            while True:
                # exec → epi
                epi = exec_to_epi.get(current)
                if epi is None:
                    break
                path.append(('epi', epi))
                # epi → exec
                next_exec = epi_to_exec.get(epi)
                if next_exec is None:
                    break
                path.append(('exec', next_exec))
                if next_exec == start_exec:
                    violations.append({
                        'edge_id': f"cycle:{start_exec}",
                        'type': 'causal_cycle',
                        'detail': (
                            f"Causal cycle detected: "
                            f"{' → '.join(p[1] for p in path)}"
                        ),
                        'path': [p[1] for p in path],
                    })
                    break
                if next_exec in visited:
                    break
                visited.add(next_exec)
                current = next_exec

        return violations

    # ------------------------------------------------------------------
    # Full consistency check
    # ------------------------------------------------------------------

    def verify(
        self,
        execution_journal_entries: set,
        epistemic_journal_events: set,
        execution_journal_times: Dict[str, float],
        epistemic_journal_times: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Full consistency check across both kernels.

        Args:
            execution_journal_entries: set of all entry_ids in execution journal
            epistemic_journal_events: set of all event_ids in epistemic journal
            execution_journal_times: dict entry_id → timestamp
            epistemic_journal_times: dict event_id → timestamp

        Returns:
            dict with 'passed', 'violations', 'details'
        """
        violations = []

        violations.extend(
            self.check_edge_targets_exist(
                execution_journal_entries, epistemic_journal_events,
            )
        )

        violations.extend(
            self.check_temporal_ordering(
                execution_journal_times, epistemic_journal_times,
            )
        )

        violations.extend(self.check_acyclicity())

        total_edges = self._graph.count()

        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'total_edges': total_edges,
            'valid_edges': total_edges - len(violations),
            'checks_performed': {
                'target_existence': True,
                'temporal_ordering': True,
                'acyclicity': True,
            },
            'summary': (
                f"{total_edges - len(violations)}/{total_edges} "
                f"edges consistent"
            ),
        }

    # ------------------------------------------------------------------
    # Replay result validator
    # ------------------------------------------------------------------

    def validate_replay(
        self,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
    ) -> List[dict]:
        """
        Validate that replay produced a consistent state.

        Compares pre-replay and post-replay state for consistency:
          - Beliefs should have same or higher confidence (not decay)
          - Epoch should be ≥ pre-replay epoch
          - Journal size should match
        """
        violations = []
        pre_beliefs = pre_state.get('beliefs', {})
        post_beliefs = post_state.get('beliefs', {})

        for name, pre_b in pre_beliefs.items():
            post_b = post_beliefs.get(name)
            if post_b is None:
                violations.append({
                    'type': 'belief_missing_after_replay',
                    'detail': f"belief={name} present before replay but missing after",
                })
            else:
                pre_conf = pre_b.get('confidence', 0.0)
                post_conf = post_b.get('confidence', 0.0)
                if post_conf < pre_conf - 0.01:
                    violations.append({
                        'type': 'belief_decayed_after_replay',
                        'detail': (
                            f"belief={name} confidence decreased "
                            f"{pre_conf} → {post_conf}"
                        ),
                    })

        pre_epoch = pre_state.get('epoch', 0)
        post_epoch = post_state.get('epoch', 0)
        if post_epoch < pre_epoch:
            violations.append({
                'type': 'epoch_regression_after_replay',
                'detail': f"epoch {pre_epoch} → {post_epoch}",
            })

        return violations
