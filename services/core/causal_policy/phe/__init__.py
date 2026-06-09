"""
Policy Horizon Engine (PHE) — recursive multi-step causal planner.

Layer structure:

    CPE (1-step counterfactual choice)
         ↓
    PHE (multi-step trajectory search)
         ↓
    selection of best action sequence

PHE does NOT:
  - mutate the execution kernel
  - update beliefs directly
  - execute actions

PHE ONLY:
  - searches through causal futures
  - scores trajectories by stability + terminal quality
  - selects the best action sequence

Usage:
    from causal_policy.phe import PolicyHorizonEngine

    phe = PolicyHorizonEngine(cpe)
    result = phe.plan(goal_id="abc", horizon=3)

    if result['decision']['chosen']:
        execute(result['decision']['chosen'])
"""

from typing import Any, Dict, List, Optional

from .tree import PolicyTree, PolicyNode
from .depth import DepthStrategy, AdaptiveDepth, FixedDepth, DepthConfig
from .uncertainty import UncertaintyDecayModel, UncertaintyState
from .scoring import TrajectoryScoringEngine, TrajectoryScore
from .search import PolicySearchEngine, SearchConfig


class PolicyHorizonEngine:
    """
    Recursive multi-step planner over causal futures.

    Entry point for strategic planning. Wraps the search engine
    with a clean API and default configuration.

    Usage:
        phe = PolicyHorizonEngine(cpe)
        plan = phe.plan(goal_id="goal-abc")

        if plan['decision']['chosen']:
            # execute the first action in the best trajectory
            first_action = plan['decision']['chosen']
    """

    def __init__(
        self,
        cpe,
        depth_strategy: Optional[DepthStrategy] = None,
        uncertainty_model: Optional[UncertaintyDecayModel] = None,
        search_config: Optional[SearchConfig] = None,
    ):
        self._cpe = cpe
        self._uncertainty = uncertainty_model or UncertaintyDecayModel()
        self._depth = depth_strategy or AdaptiveDepth()
        self._scorer = TrajectoryScoringEngine(self._uncertainty)
        self._config = search_config or SearchConfig()

        self._engine = PolicySearchEngine(
            cpe=cpe,
            depth_strategy=self._depth,
            uncertainty_model=self._uncertainty,
            scorer=self._scorer,
            config=self._config,
        )

        self._total_plans = 0
        self._plan_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def plan(
        self,
        goal_id: str,
        options: List[Dict[str, Any]] = None,
        horizon: Optional[int] = None,
        beam_width: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run strategic planning: search for best multi-step trajectory.

        Args:
            goal_id: target goal
            options: custom candidate options (or use CPE defaults)
            horizon: override max search depth (if None, uses AdaptiveDepth)
            beam_width: override beam width (if None, uses config default)

        Returns:
            dict with:
              - best_trajectory (scored)
              - all_scored (top 5)
              - tree_stats (nodes, depth)
              - decision (chosen action + score)
        """
        # Apply overrides
        if horizon is not None and isinstance(self._depth, AdaptiveDepth):
            self._depth.config.max_depth = max(1, horizon)
        bw = beam_width if beam_width is not None else self._config.beam_width
        if bw != self._config.beam_width:
            self._config.beam_width = bw
            self._engine._config.beam_width = bw

        result = self._engine.search(goal_id, options)

        self._total_plans += 1
        self._plan_history.append({
            'goal_id': goal_id,
            'decision': result.get('decision'),
            'tree_stats': result.get('tree_stats'),
        })

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def simulate_depth(
        self,
        goal_id: str,
        depths: List[int],
        options: List[Dict[str, Any]] = None,
        beam_width: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compare planning quality at different depths.

        Useful for calibrating the adaptive depth strategy.

        Args:
            goal_id: target goal
            depths: list of depths to test (e.g. [1, 2, 3, 5])
            options: candidate options
            beam_width: beam width override

        Returns:
            list of result dicts, one per depth
        """
        results = []
        for d in depths:
            result = self.plan(goal_id, options, horizon=d, beam_width=beam_width)
            results.append({
                'depth': d,
                'best_score': result.get('decision', {}).get('best_score'),
                'chosen': result.get('decision', {}).get('chosen'),
                'total_nodes': result.get('tree_stats', {}).get('total_nodes'),
            })
        return results

    def get_stats(self) -> dict:
        """Get PHE diagnostics."""
        return {
            'total_plans': self._total_plans,
            'cpe_connected': self._cpe is not None,
            'depth_strategy': type(self._depth).__name__,
            'beam_width': self._config.beam_width,
            'max_total_nodes': self._config.max_total_nodes,
            'uncertainty_cutoff': self._config.uncertainty_cutoff,
            'drift_cutoff': self._config.drift_cutoff,
            'last_plans': self._plan_history[-3:] if self._plan_history else [],
        }


__all__ = [
    'PolicyHorizonEngine',
    'PolicyTree',
    'PolicyNode',
    'DepthStrategy',
    'AdaptiveDepth',
    'FixedDepth',
    'DepthConfig',
    'UncertaintyDecayModel',
    'UncertaintyState',
    'TrajectoryScoringEngine',
    'TrajectoryScore',
    'PolicySearchEngine',
    'SearchConfig',
]
