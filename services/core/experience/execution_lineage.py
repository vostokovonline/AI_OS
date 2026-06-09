"""
Execution Lineage Graph - Track causal execution chains

Enables:
- Parent-child execution relationships with typed edges
- Planner lineage (which planner generated goal)
- Decomposition lineage (goal → subgoals)
- Policy lineage (which policy made selection)

This is critical for interpretability in multi-level decomposition.

NOTE: EdgeType is now in causal_edge.py - import from there.
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

# Import EdgeType from causal_edge (single source of truth)
from experience.causal_edge import EdgeType


@dataclass(frozen=True)
class ExecutionNode:
    """Immutable node in execution lineage"""
    node_id: str
    execution_id: str  # Link to ExecutionEnvelope
    node_type: str  # "atomic_goal", "decomposed_goal", "subgoal", "skill_execution"
    
    # Lineage
    parent_node_id: Optional[str] = None
    edge_type: Optional[EdgeType] = None  # Typed edge from parent
    planner_id: Optional[str] = None
    decomposition_depth: int = 0
    
    # Metadata
    created_at: str = ""
    status: str = "pending"  # pending, executing, completed, failed
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "execution_id": self.execution_id,
            "node_type": self.node_type,
            "parent_node_id": self.parent_node_id,
            "edge_type": self.edge_type.value if self.edge_type else None,
            "planner_id": self.planner_id,
            "decomposition_depth": self.decomposition_depth,
            "created_at": self.created_at,
            "status": self.status
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ExecutionNode":
        edge_type = None
        if data.get("edge_type"):
            try:
                edge_type = EdgeType(data["edge_type"])
            except ValueError:
                pass
        
        return ExecutionNode(
            node_id=data["node_id"],
            execution_id=data["execution_id"],
            node_type=data["node_type"],
            parent_node_id=data.get("parent_node_id"),
            edge_type=edge_type,
            planner_id=data.get("planner_id"),
            decomposition_depth=data.get("decomposition_depth", 0),
            created_at=data.get("created_at", ""),
            status=data.get("status", "pending")
        )


class ExecutionLineageGraph:
    """
    Graph of execution lineage for interpretability.
    
    Tracks:
    - Which goal generated which subgoals
    - Which policy made which selection
    - Causal chains in decomposition
    
    Usage:
        graph = ExecutionLineageGraph()
        
        # Create node for atomic goal execution
        node = graph.create_node(
            execution_id="env-123",
            node_type="atomic_goal",
            parent_node_id="parent-456"
        )
        
        # Get ancestors (causal chain)
        ancestors = graph.get_ancestors(node_id)
    """
    
    def __init__(self, store_dir: str = "/app/execution_lineage"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory graph
        self._nodes: Dict[str, ExecutionNode] = {}
        self._load_existing()
    
    def _load_existing(self):
        """Load existing nodes"""
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    node = ExecutionNode.from_dict(data)
                    self._nodes[node.node_id] = node
            except:
                continue
    
    def create_node(
        self,
        execution_id: str,
        node_type: str,
        parent_node_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
        planner_id: Optional[str] = None,
        decomposition_depth: int = 0
    ) -> ExecutionNode:
        """Create new lineage node with typed edge"""
        node = ExecutionNode(
            node_id=uuid4().hex[:8],
            execution_id=execution_id,
            node_type=node_type,
            parent_node_id=parent_node_id,
            edge_type=edge_type,
            planner_id=planner_id,
            decomposition_depth=decomposition_depth,
            created_at=datetime.utcnow().isoformat(),
            status="pending"
        )
        
        # Save to disk
        filename = self.store_dir / f"{node.node_id}.json"
        with open(filename, "w") as f:
            json.dump(node.to_dict(), f, indent=2)
        
        self._nodes[node.node_id] = node
        
        return node
    
    def get_node(self, node_id: str) -> Optional[ExecutionNode]:
        """Get node by ID"""
        return self._nodes.get(node_id)
    
    def get_ancestors(self, node_id: str) -> List[ExecutionNode]:
        """Get all ancestor nodes (causal chain)"""
        ancestors = []
        current = self._nodes.get(node_id)
        
        while current and current.parent_node_id:
            parent = self._nodes.get(current.parent_node_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_children(self, node_id: str) -> List[ExecutionNode]:
        """Get all child nodes"""
        return [n for n in self._nodes.values() if n.parent_node_id == node_id]
    
    def get_execution_chain(self, node_id: str) -> List[ExecutionNode]:
        """Get full execution chain from root to node"""
        # Get ancestors (reversed = root to leaf)
        ancestors = self.get_ancestors(node_id)
        ancestors.reverse()
        
        # Add current node
        current = self._nodes.get(node_id)
        if current:
            ancestors.append(current)
        
        return ancestors
    
    def get_planner_lineage(self, node_id: str) -> List[str]:
        """Get planner IDs in lineage"""
        chain = self.get_execution_chain(node_id)
        return [n.planner_id for n in chain if n.planner_id]
    
    def get_decomposition_depth(self, node_id: str) -> int:
        """Get decomposition depth of node"""
        node = self._nodes.get(node_id)
        return node.decomposition_depth if node else 0
    
    def update_status(self, node_id: str, status: str):
        """Update node status"""
        node = self._nodes.get(node_id)
        if node:
            # Create new node with updated status (immutable)
            updated = ExecutionNode(
                node_id=node.node_id,
                execution_id=node.execution_id,
                node_type=node.node_type,
                parent_node_id=node.parent_node_id,
                planner_id=node.planner_id,
                decomposition_depth=node.decomposition_depth,
                created_at=node.created_at,
                status=status
            )
            
            # Update in memory and disk
            self._nodes[node_id] = updated
            filename = self.store_dir / f"{node_id}.json"
            with open(filename, "w") as f:
                json.dump(updated.to_dict(), f, indent=2)
    
    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        return {
            "total_nodes": len(self._nodes),
            "by_type": self._count_by_type(),
            "by_status": self._count_by_status(),
            "max_depth": self._get_max_depth()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        counts = {}
        for node in self._nodes.values():
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts
    
    def _count_by_status(self) -> Dict[str, int]:
        counts = {}
        for node in self._nodes.values():
            counts[node.status] = counts.get(node.status, 0) + 1
        return counts
    
    def _get_max_depth(self) -> int:
        if not self._nodes:
            return 0
        return max(n.decomposition_depth for n in self._nodes.values())


# Global lineage graph
_lineage_graph: Optional[ExecutionLineageGraph] = None


def get_lineage_graph() -> ExecutionLineageGraph:
    """Get or create global lineage graph"""
    global _lineage_graph
    if _lineage_graph is None:
        _lineage_graph = ExecutionLineageGraph()
    return _lineage_graph


# Convenience functions
def create_execution_node(
    execution_id: str,
    node_type: str,
    parent_node_id: Optional[str] = None,
    edge_type: Optional[EdgeType] = None,
    planner_id: Optional[str] = None,
    decomposition_depth: int = 0
) -> ExecutionNode:
    """Create lineage node"""
    return get_lineage_graph().create_node(
        execution_id, node_type, parent_node_id, edge_type, planner_id, decomposition_depth
    )


def get_execution_chain(execution_id: str) -> List[ExecutionNode]:
    """Get execution chain for execution"""
    # Find node by execution_id
    graph = get_lineage_graph()
    for node in graph._nodes.values():
        if node.execution_id == execution_id:
            return graph.get_execution_chain(node.node_id)
    return []