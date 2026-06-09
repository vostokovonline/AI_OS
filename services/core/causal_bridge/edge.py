"""
CausalityEdge — formal causal link between execution and epistemic domains.

An edge connects:
  - ONE execution event (JournalEntry) to ONE epistemic event (InterpretationEvent)
  - With a direction, confidence, and interpretation frame

This is the atomic unit of the cross-kernel causality graph.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import time
import uuid


# Monotonic edge ID counter (single-threaded, for deterministic ordering)
_edge_counter = 0


def _next_edge_id() -> str:
    global _edge_counter
    _edge_counter += 1
    return f"ce:{int(time.time())}:{_edge_counter:06d}"


class CausalDirection(Enum):
    EXECUTION_TO_EPISTEMIC = auto()
    EPISTEMIC_TO_EXECUTION = auto()
    BIDIRECTIONAL = auto()


class ExecutionEventType(str, Enum):
    DISPATCHED = "DISPATCHED"
    LEASE_ISSUED = "LEASE_ISSUED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PREEMPTED = "PREEMPTED"
    RETRIED = "RETRIED"
    ABANDONED = "ABANDONED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    LEASE_REVOKED = "LEASE_REVOKED"


class EpistemicEventType(str, Enum):
    OBSERVATION = "OBSERVATION"
    BELIEF_UPDATE = "BELIEF_UPDATE"
    MOTIF_UPDATE = "MOTIF_UPDATE"
    ATTRACTOR_UPDATE = "ATTRACTOR_UPDATE"
    DRIFT_ATTENUATION = "DRIFT_ATTENUATION"
    GROUNDING = "GROUNDING"


@dataclass
class InterpretationFrame:
    """
    Semantic interpretation of why an execution event matters.

    This is the bridge between "what happened" and "what it means."
    """
    interpretation: str
    belief_delta: Dict[str, float] = field(default_factory=dict)
    motif_delta: Dict[str, float] = field(default_factory=dict)
    confidence_delta: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'interpretation': self.interpretation,
            'belief_delta': dict(self.belief_delta),
            'motif_delta': dict(self.motif_delta),
            'confidence_delta': self.confidence_delta,
            'tags': list(self.tags),
        }


@dataclass
class CausalityEdge:
    """
    A formal causal link between an execution event and an epistemic event.

    Fields:
      edge_id: unique identifier
      direction: which domain caused which
      execution_entry_id: JournalEntry.entry_id (execution domain)
      execution_event_type: one of DISPATCH_EVENTS
      execution_goal_id: the goal involved
      epistemic_event_id: InterpretationEvent.event_id (epistemic domain)
      epistemic_event_type: OBSERVATION | BELIEF_UPDATE | MOTIF_UPDATE | ...
      interpretation: why this link exists
      confidence: how confident we are in this causal link (0.0-1.0)
      causal_strength: weight of influence (0.0-1.0)
      created_at: when the edge was created
      context: arbitrary metadata
    """
    edge_id: str
    direction: CausalDirection

    # Execution domain (required)
    execution_entry_id: str
    execution_event_type: str
    execution_goal_id: str

    # Epistemic domain (required)
    epistemic_event_id: str
    epistemic_event_type: str

    # Semantics (required)
    interpretation: InterpretationFrame

    # Execution domain (optional)
    execution_lease_id: str = ""

    # Semantics (optional)
    confidence: float = 1.0
    causal_strength: float = 1.0

    # Metadata
    created_at: float = 0.0
    context: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            'edge_id': self.edge_id,
            'direction': self.direction.name,
            'execution_entry_id': self.execution_entry_id,
            'execution_event_type': self.execution_event_type,
            'execution_goal_id': self.execution_goal_id,
            'execution_lease_id': self.execution_lease_id,
            'epistemic_event_id': self.epistemic_event_id,
            'epistemic_event_type': self.epistemic_event_type,
            'interpretation': self.interpretation.to_dict(),
            'confidence': self.confidence,
            'causal_strength': self.causal_strength,
            'created_at': self.created_at,
            'context': dict(self.context),
        }


class CausalityGraph:
    """
    Directed graph of CausalityEdges.

    Supports:
      - Query edges by execution event id
      - Query edges by epistemic event id
      - Query edges by goal_id
      - Query edges by direction
      - Traverse the graph forward/backward
    """

    def __init__(self):
        self._edges: Dict[str, CausalityEdge] = {}
        self._by_execution_event: Dict[str, List[str]] = {}
        self._by_epistemic_event: Dict[str, List[str]] = {}
        self._by_goal: Dict[str, List[str]] = {}

    def add_edge(self, edge: CausalityEdge):
        self._edges[edge.edge_id] = edge

        exec_key = f"{edge.execution_goal_id}:{edge.execution_event_type}"
        self._by_execution_event.setdefault(exec_key, []).append(edge.edge_id)
        self._by_execution_event.setdefault(edge.execution_entry_id, []).append(edge.edge_id)

        epi_key = f"{edge.epistemic_event_type}"
        self._by_epistemic_event.setdefault(epi_key, []).append(edge.edge_id)
        self._by_epistemic_event.setdefault(edge.epistemic_event_id, []).append(edge.edge_id)

        self._by_goal.setdefault(edge.execution_goal_id, []).append(edge.edge_id)

    def get_edge(self, edge_id: str) -> Optional[CausalityEdge]:
        return self._edges.get(edge_id)

    def get_edges_for_execution_event(self, execution_entry_id: str) -> List[CausalityEdge]:
        edge_ids = self._by_execution_event.get(execution_entry_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_edges_for_epistemic_event(self, epistemic_event_id: str) -> List[CausalityEdge]:
        edge_ids = self._by_epistemic_event.get(epistemic_event_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_edges_for_goal(self, goal_id: str) -> List[CausalityEdge]:
        edge_ids = self._by_goal.get(goal_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_edges_by_direction(self, direction: CausalDirection) -> List[CausalityEdge]:
        return [
            e for e in self._edges.values()
            if e.direction == direction
        ]

    def count(self) -> int:
        return len(self._edges)

    def get_stats(self) -> dict:
        return {
            'total_edges': self.count(),
            'by_direction': {
                d.name: sum(1 for e in self._edges.values() if e.direction == d)
                for d in CausalDirection
            },
            'execution_to_epistemic': len(self.get_edges_by_direction(
                CausalDirection.EXECUTION_TO_EPISTEMIC
            )),
            'epistemic_to_execution': len(self.get_edges_by_direction(
                CausalDirection.EPISTEMIC_TO_EXECUTION
            )),
            'unique_goals': len(self._by_goal),
        }
