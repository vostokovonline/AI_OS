"""
Execution Edge - Typed causal relationship between nodes

CRITICAL: One node can have multiple outgoing edges with different types.
Edge is separate from Node to support branching workflows.
"""
import json
from enum import Enum
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


class EdgeType(str, Enum):
    """Typed edges for causal relationships"""
    DECOMPOSED_INTO = "decomposed_into"  # Goal → subgoals
    DELEGATED_TO = "delegated_to"  # Planner → execution
    RETRIED_AS = "retried_as"  # Failed → retry
    SHADOWED_BY = "shadowed_by"  # Shadow evaluation
    PROMOTED_FROM = "promoted_from"  # Policy promotion
    CAUSED_BY = "caused_by"  # Causal dependency
    PLANNED_BY = "planned_by"  # Goal created by planner
    EXECUTED_BY = "executed_by"  # Execution by skill
    EVALUATED_BY = "evaluated_by"  # Evaluated by policy
    FAILED = "failed"  # Execution failed
    SUCCEEDED = "succeeded"  # Execution succeeded


@dataclass(frozen=True)
class ExecutionEdge:
    """
    Immutable typed edge in execution lineage.
    
    Separated from ExecutionNode because a node can have multiple
    outgoing edges with different semantics.
    """
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType
    
    # Metadata
    created_at: str
    weight: float = 1.0  # For causal strength
    
    def to_dict(self) -> dict:
        return {
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "edge_type": self.edge_type.value,
            "created_at": self.created_at,
            "weight": self.weight
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ExecutionEdge":
        return ExecutionEdge(
            edge_id=data["edge_id"],
            source_node_id=data["source_node_id"],
            target_node_id=data["target_node_id"],
            edge_type=EdgeType(data["edge_type"]),
            created_at=data.get("created_at", ""),
            weight=data.get("weight", 1.0)
        )


class ExecutionEdgeStore:
    """Store for execution edges"""
    
    def __init__(self, store_dir: str = "/app/execution_edges"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._edges = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    edge = ExecutionEdge.from_dict(data)
                    self._edges[edge.edge_id] = edge
            except:
                continue
    
    def create_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_type: EdgeType,
        weight: float = 1.0
    ) -> ExecutionEdge:
        """Create new edge"""
        edge = ExecutionEdge(
            edge_id=uuid4().hex[:8],
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            created_at=datetime.utcnow().isoformat(),
            weight=weight
        )
        
        # Save to disk
        filename = self.store_dir / f"{edge.edge_id}.json"
        with open(filename, "w") as f:
            json.dump(edge.to_dict(), f, indent=2)
        
        self._edges[edge.edge_id] = edge
        
        return edge
    
    def get_outgoing_edges(self, node_id: str) -> list:
        """Get all outgoing edges from node"""
        return [e for e in self._edges.values() if e.source_node_id == node_id]
    
    def get_incoming_edges(self, node_id: str) -> list:
        """Get all incoming edges to node"""
        return [e for e in self._edges.values() if e.target_node_id == node_id]
    
    def get_edges_by_type(self, edge_type: EdgeType) -> list:
        """Get all edges of specific type"""
        return [e for e in self._edges.values() if e.edge_type == edge_type]
    
    def get_statistics(self) -> dict:
        counts = {}
        for edge in self._edges.values():
            counts[edge.edge_type.value] = counts.get(edge.edge_type.value, 0) + 1
        return {
            "total_edges": len(self._edges),
            "by_type": counts
        }


# Global store
_edge_store: Optional[ExecutionEdgeStore] = None


def get_edge_store() -> ExecutionEdgeStore:
    global _edge_store
    if _edge_store is None:
        _edge_store = ExecutionEdgeStore()
    return _edge_store