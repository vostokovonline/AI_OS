"""
Policy Tree — recursive trajectory search structure.

Each node represents a simulated epistemic state after a candidate action.
The root is the CURRENT real state (no action).
Children are possible NEXT states after different actions.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import time
import uuid


@dataclass
class PolicyNode:
    """
    A single node in the policy search tree.

    Fields:
      node_id: unique identifier
      depth: distance from root (root = 0)
      state_snapshot: dict with 'beliefs', 'motifs', 'attractors', 'epoch'
      action: CandidateAction that led TO this node (None for root)
      parent_id: node_id of parent (None for root)
      children: list of child PolicyNode
      cumulative_score: path-weighted score from root to this node
      uncertainty: uncertainty estimate at this node (0.0-1.0)
      drift_estimate: drift score at this simulated state
      path_actions: list of action labels from root to this node
      terminal: True if expansion stopped here
      created_at: timestamp
    """
    node_id: str
    depth: int
    state_snapshot: Dict[str, Any]

    action: Optional[Any] = None
    parent_id: Optional[str] = None
    children: List['PolicyNode'] = field(default_factory=list)

    cumulative_score: float = 0.0
    uncertainty: float = 0.0
    drift_estimate: float = 0.0

    path_actions: List[str] = field(default_factory=list)
    terminal: bool = False
    created_at: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'action_label': self.action.label if self.action else None,
            'action_type': self.action.predicted_event_type if self.action else None,
            'cumulative_score': self.cumulative_score,
            'uncertainty': self.uncertainty,
            'drift_estimate': self.drift_estimate,
            'path_actions': list(self.path_actions),
            'terminal': self.terminal,
            'children_count': len(self.children),
        }


class PolicyTree:
    """
    Recursive policy search tree.

    Root = current real state.
    Each expansion = simulate an action from current node.

    Supports:
      - Add child nodes
      - Extract best trajectory by cumulative score
      - Prune to beam width
      - Serialize to dict
      - Count nodes
    """

    def __init__(self, root_state: Dict[str, Any]):
        self.root = PolicyNode(
            node_id=f"pn:{uuid.uuid4().hex[:12]}",
            depth=0,
            state_snapshot=root_state,
            action=None,
        )
        self._all_nodes: Dict[str, PolicyNode] = {self.root.node_id: self.root}

    def add_child(
        self,
        parent: PolicyNode,
        action,
        state_snapshot: Dict[str, Any],
        score: float = 0.0,
        uncertainty: float = 0.0,
        drift_estimate: float = 0.0,
    ) -> PolicyNode:
        child = PolicyNode(
            node_id=f"pn:{uuid.uuid4().hex[:12]}",
            depth=parent.depth + 1,
            state_snapshot=state_snapshot,
            action=action,
            parent_id=parent.node_id,
            cumulative_score=parent.cumulative_score + score,
            uncertainty=uncertainty,
            drift_estimate=drift_estimate,
            path_actions=parent.path_actions + [action.label],
        )
        parent.children.append(child)
        self._all_nodes[child.node_id] = child
        return child

    def get_best_trajectory(self) -> List[PolicyNode]:
        """
        Find the trajectory (root → leaf) with highest cumulative score.

        Uses DFS to find the leaf with max cumulative_score,
        then traces back to root.
        """
        best_leaf = None
        best_score = float('-inf')

        def _dfs(node: PolicyNode):
            nonlocal best_leaf, best_score
            if not node.children or node.terminal:
                if node.cumulative_score > best_score:
                    best_score = node.cumulative_score
                    best_leaf = node
                return
            for child in node.children:
                _dfs(child)

        _dfs(self.root)

        if best_leaf is None:
            return [self.root]

        # Trace back to root
        trajectory = []
        current = best_leaf
        while current is not None:
            trajectory.append(current)
            current = self._all_nodes.get(current.parent_id) if current.parent_id else None
        trajectory.reverse()
        return trajectory

    def get_all_trajectories(self) -> List[List[PolicyNode]]:
        """Get all root-to-leaf trajectories."""
        trajectories = []

        def _dfs(node: PolicyNode, path: List[PolicyNode]):
            current_path = path + [node]
            if not node.children or node.terminal:
                trajectories.append(current_path)
                return
            for child in node.children:
                _dfs(child, current_path)

        _dfs(self.root, [])
        return trajectories

    def prune_to_beam(self, beam_width: int = 3):
        """
        Prune the tree to keep only the best `beam_width` trajectories.

        At each depth level beyond 1, keep only the top `beam_width` nodes
        by cumulative_score. Removes the rest.
        """
        if beam_width <= 0:
            return

        def _prune_level(nodes: List[PolicyNode]):
            if not nodes:
                return
            # Sort by cumulative score DESC
            nodes.sort(key=lambda n: n.cumulative_score, reverse=True)
            keep = nodes[:beam_width]
            remove = nodes[beam_width:]

            for node in remove:
                self._remove_node(node)

            # Next level
            next_level = []
            for node in keep:
                next_level.extend(node.children)
            _prune_level(next_level)

        _prune_level(list(self.root.children))

    def _remove_node(self, node: PolicyNode):
        """Remove a node and all its descendants from the tree."""
        if node.node_id in self._all_nodes:
            del self._all_nodes[node.node_id]
        for child in list(node.children):
            self._remove_node(child)

        # Remove from parent
        if node.parent_id:
            parent = self._all_nodes.get(node.parent_id)
            if parent and node in parent.children:
                parent.children.remove(node)

    def count_nodes(self) -> int:
        return len(self._all_nodes)

    def max_depth(self) -> int:
        if not self._all_nodes:
            return 0
        return max(n.depth for n in self._all_nodes.values())

    def to_dict(self) -> dict:
        best_traj = self.get_best_trajectory()
        return {
            'total_nodes': self.count_nodes(),
            'max_depth': self.max_depth(),
            'best_trajectory': [n.to_dict() for n in best_traj],
            'best_score': best_traj[-1].cumulative_score if best_traj else 0.0,
        }
