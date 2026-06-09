"""
Policy Search Engine — recursive tree expansion with pruning.

Core algorithm:
  1. Start with root node (current real state)
  2. For each node:
     a. Propose candidate actions via CPE
     b. Simulate each action via CPE
     c. If DepthStrategy.should_expand() → recurse
     d. Else → mark terminal
  3. Score all trajectories
  4. Return best trajectory

Pruning strategies:
  - Beam search: keep only top N nodes at each depth
  - Uncertainty cutoff: stop if uncertainty exceeds threshold
  - Drift cutoff: stop if drift exceeds threshold
  - Max total nodes: hard limit on tree size
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .tree import PolicyTree, PolicyNode
from .depth import DepthStrategy, AdaptiveDepth
from .uncertainty import UncertaintyDecayModel, UncertaintyState
from .scoring import TrajectoryScoringEngine, TrajectoryScore


@dataclass
class SearchConfig:
    """
    Configuration for the policy search.

    Fields:
      beam_width: number of top trajectories to keep at each level (0 = no pruning)
      max_total_nodes: hard limit on total tree nodes
      uncertainty_cutoff: stop expansion if node uncertainty exceeds this
      drift_cutoff: stop expansion if node drift exceeds this
      reality_anchor: if True, apply reality anchoring penalty
      default_candidates: if True, use CPE proposal defaults when none provided
      candidates_per_node: max candidate actions to generate per node
    """
    beam_width: int = 3
    max_total_nodes: int = 100
    uncertainty_cutoff: float = 0.85
    drift_cutoff: float = 0.8
    reality_anchor: bool = True
    default_candidates: bool = True
    candidates_per_node: int = 3


class PolicySearchEngine:
    """
    Recursive search engine that expands the policy tree.

    Uses CPE for:
      - propose_actions(goal_id, options) → list of CandidateAction
      - simulate(candidate) → SimulatedOutcome

    Does NOT use CPE for:
      - mutation (pure search only)
      - execution (decision only)
    """

    def __init__(
        self,
        cpe: Any,  # CausalPolicyEngine
        depth_strategy: Optional[DepthStrategy] = None,
        uncertainty_model: Optional[UncertaintyDecayModel] = None,
        scorer: Optional[TrajectoryScoringEngine] = None,
        config: Optional[SearchConfig] = None,
    ):
        self._cpe = cpe
        self._depth = depth_strategy or AdaptiveDepth()
        self._uncertainty = uncertainty_model or UncertaintyDecayModel()
        self._scorer = scorer
        self._config = config or SearchConfig()

    def _ensure_scorer(self):
        if self._scorer is None:
            self._scorer = TrajectoryScoringEngine(self._uncertainty)

    # ------------------------------------------------------------------
    # Main search entry point
    # ------------------------------------------------------------------

    def search(
        self,
        goal_id: str,
        options: List[Dict[str, Any]] = None,
        custom_states: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run recursive policy search.

        Args:
            goal_id: target goal
            options: candidate options (passed to CPE.propose_actions)
            custom_states: optional override for initial epistemic state

        Returns:
            dict with 'best_trajectory', 'all_scored', 'tree_stats', 'decision'
        """
        self._ensure_scorer()

        # Build root state snapshot
        if custom_states:
            root_state = dict(custom_states)
        else:
            root_state = {
                'beliefs': self._cpe._epistemic_kernel.get_all_beliefs(),
                'motifs': self._cpe._epistemic_kernel.get_all_motifs(),
                'attractors': self._cpe._epistemic_kernel.get_all_attractors(),
                'epoch': self._cpe._epistemic_kernel.epoch.current,
                'drift_estimate': self._cpe._epistemic_kernel.check_drift().overall_drift_score,
            }

        # Build tree
        tree = PolicyTree(root_state)
        root_uncertainty = self._uncertainty.initial_state()

        # Recursive expansion
        self._expand_node(
            tree=tree,
            node=tree.root,
            uncertainty_state=root_uncertainty,
            goal_id=goal_id,
            options=options,
        )

        # Score all trajectories
        all_trajs = tree.get_all_trajectories()
        scored = self._scorer.score_all_trajectories(all_trajs)

        # Best trajectory
        best = scored[0] if scored else None

        # Decision:
        chosen_action = None
        if best and best.total_score >= 0.5:
            if len(best.trajectory) > 1:
                chosen_action = best.trajectory[1].action  # first action after root

        return {
            'best_trajectory': best.to_dict() if best else None,
            'all_scored': [s.to_dict() for s in scored[:5]],  # top 5
            'tree_stats': tree.to_dict(),
            'decision': {
                'chosen': chosen_action.label if chosen_action else None,
                'chosen_type': chosen_action.predicted_event_type if chosen_action else None,
                'best_score': best.total_score if best else 0.0,
                'threshold_reached': (best.total_score >= 0.5) if best else False,
            },
        }

    # ------------------------------------------------------------------
    # Recursive expansion
    # ------------------------------------------------------------------

    def _expand_node(
        self,
        tree: PolicyTree,
        node: PolicyNode,
        uncertainty_state: UncertaintyState,
        goal_id: str,
        options: List[Dict[str, Any]] = None,
    ):
        """Recursively expand a node."""

        # Check hard limits
        if tree.count_nodes() >= self._config.max_total_nodes:
            node.terminal = True
            return

        if uncertainty_state.uncertainty >= self._config.uncertainty_cutoff:
            node.terminal = True
            return

        if node.drift_estimate >= self._config.drift_cutoff:
            node.terminal = True
            return

        # Check depth policy
        # Compute branching factor from node's existing children
        branching = max(len(node.children), 1)

        if not self._depth.should_expand(
            depth=node.depth,
            uncertainty=uncertainty_state.uncertainty,
            drift_estimate=node.drift_estimate,
            motif_count=len(node.state_snapshot.get('motifs', {})),
            avg_motif_strength=self._avg_motif_strength(node.state_snapshot.get('motifs', {})),
            branching_factor=branching,
        ):
            node.terminal = True
            return

        # Propose candidate actions
        candidates = self._cpe.propose_actions(goal_id, options)
        if not candidates:
            node.terminal = True
            return

        # Limit candidates per node
        candidates = candidates[:self._config.candidates_per_node]

        for candidate in candidates:
            # Simulate via CPE
            try:
                outcome = self._cpe._simulator.simulate(candidate)
            except Exception:
                continue

            # Build state snapshot for child
            child_snapshot = {
                'beliefs': outcome.post_beliefs,
                'motifs': outcome.post_motifs,
                'attractors': outcome.post_attractors,
                'epoch': outcome.post_epoch,
                'drift_estimate': outcome.drift_report.overall_drift_score,
            }

            # Compute uncertainty for child
            child_uncertainty = self._uncertainty.compute_child_state(
                parent=uncertainty_state,
                simulation_error=abs(outcome.stability_delta),
                drift=outcome.drift_report.overall_drift_score,
                branching_factor=len(candidates),
            )

            # Compute local score (for cumulative scoring)
            local_score = outcome.stability_delta

            # Add child node
            child = tree.add_child(
                parent=node,
                action=candidate,
                state_snapshot=child_snapshot,
                score=local_score,
                uncertainty=child_uncertainty.uncertainty,
                drift_estimate=outcome.drift_report.overall_drift_score,
            )

            # Recurse
            self._expand_node(
                tree=tree,
                node=child,
                uncertainty_state=child_uncertainty,
                goal_id=goal_id,
                options=options,
            )

        # Apply beam pruning after expansion
        if self._config.beam_width > 0:
            tree.prune_to_beam(self._config.beam_width)

    @staticmethod
    def _avg_motif_strength(motifs: Dict) -> float:
        if not motifs:
            return 0.0
        strengths = [m.get('strength', 0.0) for m in motifs.values()]
        return sum(strengths) / len(strengths)
