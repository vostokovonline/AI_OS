"""
Unified Execution Graph - Nodes + Edges together

CRITICAL: This ensures graph consistency.
Separating nodes and edges led to orphan edges and inconsistent state.
Now they are managed together.
"""
import json
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from experience.causal_edge import EdgeType, ExecutionEdge


@dataclass
class ExecutionNodeData:
    """Mutable node data with adjacency"""
    node_id: str
    execution_id: str
    node_type: str
    parent_node_id: Optional[str]
    edge_type: Optional[EdgeType]
    planner_id: Optional[str]
    decomposition_depth: int
    created_at: str
    status: str
    
    # Adjacency (computed)
    outgoing_edges: List[str] = field(default_factory=list)  # Edge IDs
    incoming_edges: List[str] = field(default_factory=list)  # Edge IDs
    
    def to_dict(self) -> dict:
        # Handle edge_type being either EdgeType enum or string
        edge_type_val = None
        if self.edge_type:
            if hasattr(self.edge_type, 'value'):
                edge_type_val = self.edge_type.value
            else:
                edge_type_val = str(self.edge_type)
        
        return {
            "node_id": self.node_id,
            "execution_id": self.execution_id,
            "node_type": self.node_type,
            "parent_node_id": self.parent_node_id,
            "edge_type": edge_type_val,
            "planner_id": self.planner_id,
            "decomposition_depth": self.decomposition_depth,
            "created_at": self.created_at,
            "status": self.status,
            "outgoing_edges": self.outgoing_edges,
            "incoming_edges": self.incoming_edges
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ExecutionNodeData":
        edge_type = None
        if data.get("edge_type"):
            try:
                edge_type = EdgeType(data["edge_type"])
            except ValueError:
                pass
        
        return ExecutionNodeData(
            node_id=data["node_id"],
            execution_id=data["execution_id"],
            node_type=data["node_type"],
            parent_node_id=data.get("parent_node_id"),
            edge_type=edge_type,
            planner_id=data.get("planner_id"),
            decomposition_depth=data.get("decomposition_depth", 0),
            created_at=data.get("created_at", ""),
            status=data.get("status", "pending"),
            outgoing_edges=data.get("outgoing_edges", []),
            incoming_edges=data.get("incoming_edges", [])
        )


class ExecutionGraph:
    """
    Unified execution graph with nodes and edges.
    
    This ensures:
    - Nodes and edges stay consistent
    - Ancestry queries work correctly
    - No orphan edges
    - Proper cleanup on deletion
    """
    
    def __init__(self, store_dir: str = "/app/execution_graph"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory graph
        self._nodes: Dict[str, ExecutionNodeData] = {}
        self._edges: Dict[str, ExecutionEdge] = {}
        self._index_by_execution: Dict[str, str] = {}  # execution_id → node_id
        
        self._load_existing()
    
    def _load_existing(self):
        # Load nodes
        nodes_dir = self.store_dir / "nodes"
        nodes_dir.mkdir(exist_ok=True)
        
        for filename in nodes_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    node = ExecutionNodeData.from_dict(data)
                    self._nodes[node.node_id] = node
                    self._index_by_execution[node.execution_id] = node.node_id
            except:
                continue
        
        # Load edges
        edges_dir = self.store_dir / "edges"
        edges_dir.mkdir(exist_ok=True)
        
        for filename in edges_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    edge = ExecutionEdge.from_dict(data)
                    self._edges[edge.edge_id] = edge
                    
                    # Update adjacency in nodes
                    if edge.source_node_id in self._nodes:
                        self._nodes[edge.source_node_id].outgoing_edges.append(edge.edge_id)
                    if edge.target_node_id in self._nodes:
                        self._nodes[edge.target_node_id].incoming_edges.append(edge.edge_id)
            except:
                continue
    
    def _save_node(self, node: ExecutionNodeData):
        """Save node to disk"""
        nodes_dir = self.store_dir / "nodes"
        with open(nodes_dir / f"{node.node_id}.json", "w") as f:
            json.dump(node.to_dict(), f, indent=2)
    
    def _save_edge(self, edge: ExecutionEdge):
        """Save edge to disk"""
        edges_dir = self.store_dir / "edges"
        with open(edges_dir / f"{edge.edge_id}.json", "w") as f:
            json.dump(edge.to_dict(), f, indent=2)
    
    def create_node(
        self,
        execution_id: str,
        node_type: str,
        parent_node_id: Optional[str] = None,
        edge_type: Optional[EdgeType] = None,
        planner_id: Optional[str] = None,
        decomposition_depth: int = 0
    ) -> ExecutionNodeData:
        """Create node and optionally link to parent"""
        
        # Create node
        node = ExecutionNodeData(
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
        
        # Save node
        self._nodes[node.node_id] = node
        self._index_by_execution[execution_id] = node.node_id
        self._save_node(node)
        
        # Create edge to parent if exists
        if parent_node_id and parent_node_id in self._nodes:
            parent = self._nodes[parent_node_id]
            edge = ExecutionEdge(
                edge_id=uuid4().hex[:8],
                source_node_id=parent_node_id,
                target_node_id=node.node_id,
                edge_type=edge_type or EdgeType.DECOMPOSED_INTO,
                created_at=datetime.utcnow().isoformat()
            )
            
            self._edges[edge.edge_id] = edge
            parent.outgoing_edges.append(edge.edge_id)
            node.incoming_edges.append(edge.edge_id)
            
            self._save_edge(edge)
            self._save_node(parent)
        
        return node
    
    def get_node(self, node_id: str) -> Optional[ExecutionNodeData]:
        """Get node by ID"""
        return self._nodes.get(node_id)
    
    def get_node_by_execution(self, execution_id: str) -> Optional[ExecutionNodeData]:
        """Get node by execution ID"""
        node_id = self._index_by_execution.get(execution_id)
        return self._nodes.get(node_id) if node_id else None
    
    def get_edge(self, edge_id: str) -> Optional[ExecutionEdge]:
        """Get edge by ID"""
        return self._edges.get(edge_id)
    
    def get_outgoing_edges(self, node_id: str) -> List[ExecutionEdge]:
        """Get all outgoing edges from node"""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._edges[eid] for eid in node.outgoing_edges if eid in self._edges]
    
    def get_incoming_edges(self, node_id: str) -> List[ExecutionEdge]:
        """Get all incoming edges to node"""
        node = self._nodes.get(node_id)
        if not node:
            return []
        return [self._edges[eid] for eid in node.incoming_edges if eid in self._edges]
    
    def get_ancestors(self, node_id: str) -> List[ExecutionNodeData]:
        """Get all ancestor nodes (causal chain)"""
        ancestors = []
        current_id = node_id
        
        while True:
            node = self._nodes.get(current_id)
            if not node or not node.parent_node_id:
                break
            
            parent = self._nodes.get(node.parent_node_id)
            if parent:
                ancestors.append(parent)
                current_id = parent.node_id
            else:
                break
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[ExecutionNodeData]:
        """Get all descendant nodes"""
        descendants = []
        queue = [node_id]
        visited = {node_id}
        
        while queue:
            current = queue.pop(0)
            node = self._nodes.get(current)
            if not node:
                continue
            
            for edge in self.get_outgoing_edges(current):
                if edge.target_node_id not in visited:
                    visited.add(edge.target_node_id)
                    descendant = self._nodes.get(edge.target_node_id)
                    if descendant:
                        descendants.append(descendant)
                        queue.append(descendant.node_id)
        
        return descendants
    
    def get_execution_chain(self, node_id: str) -> List[ExecutionNodeData]:
        """Get full execution chain from root to node"""
        ancestors = self.get_ancestors(node_id)
        ancestors.reverse()
        
        current = self._nodes.get(node_id)
        if current:
            ancestors.append(current)
        
        return ancestors
    
    def update_status(self, node_id: str, status: str):
        """Update node status"""
        node = self._nodes.get(node_id)
        if node:
            node.status = status
            self._save_node(node)
    
    def get_statistics(self) -> dict:
        """Get graph statistics"""
        by_type = {}
        by_status = {}
        by_edge_type = {}
        
        for node in self._nodes.values():
            by_type[node.node_type] = by_type.get(node.node_type, 0) + 1
            by_status[node.status] = by_status.get(node.status, 0) + 1
        
        for edge in self._edges.values():
            by_edge_type[edge.edge_type.value] = by_edge_type.get(edge.edge_type.value, 0) + 1
        
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "by_type": by_type,
            "by_status": by_status,
            "by_edge_type": by_edge_type,
            "max_depth": max((n.decomposition_depth for n in self._nodes.values()), default=0)
        }
    
    def validate_dag(self) -> Dict:
        """
        Validate DAG properties.
        
        Checks:
        - No cycles
        - No orphan edges
        - Consistent node references
        """
        violations = []
        
        # Check 1: No cycles (using DFS)
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str, path: List[str]) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)
            
            node = self._nodes.get(node_id)
            if not node:
                return False
            
            for edge_id in node.outgoing_edges:
                edge = self._edges.get(edge_id)
                if not edge:
                    violations.append(f"Orphan edge: {edge_id}")
                    continue
                
                target = edge.target_node_id
                
                if target not in self._nodes:
                    violations.append(f"Edge points to non-existent node: {edge_id}")
                    continue
                
                if target in rec_stack:
                    violations.append(f"Cycle detected: {target} in recursion stack")
                    return True
                
                if target not in visited:
                    if has_cycle(target, path):
                        return True
            
            rec_stack.remove(node_id)
            path.pop()
            return False
        
        # Check all nodes
        for node_id in self._nodes:
            if node_id not in visited:
                if has_cycle(node_id, []):
                    break
        
        # Check 2: Depth monotonicity (children depth > parent depth)
        for node in self._nodes.values():
            if node.parent_node_id:
                parent = self._nodes.get(node.parent_node_id)
                if parent and node.decomposition_depth <= parent.decomposition_depth:
                    violations.append(
                        f"Depth non-monotonic: {node.node_id} (depth={node.decomposition_depth}) "
                        f"child of {parent.node_id} (depth={parent.decomposition_depth})"
                    )
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations,
            "node_count": len(self._nodes),
            "edge_count": len(self._edges)
        }
    
    def validate_temporal_order(self) -> Dict:
        """Validate temporal ordering of events"""
        violations = []
        
        for node in self._nodes.values():
            if node.parent_node_id:
                parent = self._nodes.get(node.parent_node_id)
                if parent:
                    # Parent must be created before child
                    if parent.created_at and node.created_at:
                        if parent.created_at > node.created_at:
                            violations.append(
                                f"Temporal violation: child {node.node_id} created before "
                                f"parent {parent.node_id}"
                            )
        
        return {
            "is_valid": len(violations) == 0,
            "violations": violations
        }


# Global graph
_execution_graph: Optional[ExecutionGraph] = None


def get_execution_graph() -> ExecutionGraph:
    """Get or create global execution graph"""
    global _execution_graph
    if _execution_graph is None:
        _execution_graph = ExecutionGraph()
    return _execution_graph