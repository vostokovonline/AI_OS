"""
Provenance Graph — immutable chain of belief origins.

MIRRORS LeaseRegistry / capability provenance in execution_dynamics.

Each belief update is recorded as a provenance entry containing:
  - belief_name: the belief being modified
  - confidence_delta: how much confidence changed (could be negative)
  - provenance_label: why the change occurred
  - event_id: link to the semantic journal event
  - epoch: interpretation epoch at time of change

The ProvenanceGraph enables full auditability:
  - "Why does the system believe burnout_risk=0.65?"
  - "Where did the sleep_fragmentation → fatigue_signal motif originate?"
  - "Was this belief formed before or after the grounding checkpoint?"
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time


@dataclass
class ProvenanceEntry:
    """A single causal entry in a belief's provenance chain."""
    belief_name: str
    confidence_delta: float
    provenance_label: str
    event_id: str
    epoch: int
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


class ProvenanceGraph:
    """
    Directed acyclic graph of belief provenance.

    Each belief has a chain of provenance entries (causal order).
    The graph can answer:
      - Full ancestry of a belief
      - All beliefs modified by a given provenance label
      - Beliefs modified in a given epoch range
    """

    def __init__(self):
        self._entries: List[ProvenanceEntry] = []

    def record_belief_provenance(
        self,
        belief_name: str,
        confidence_delta: float,
        provenance_label: str,
        event_id: str,
        epoch: int,
    ):
        entry = ProvenanceEntry(
            belief_name=belief_name,
            confidence_delta=confidence_delta,
            provenance_label=provenance_label,
            event_id=event_id,
            epoch=epoch,
            timestamp=time.time(),
        )
        self._entries.append(entry)

    def get_belief_chain(self, belief_name: str) -> List[dict]:
        """Get full provenance chain for a single belief."""
        result = []
        for entry in self._entries:
            if entry.belief_name == belief_name:
                result.append(entry.to_dict())
        return result

    def get_entries_by_provenance(self, label: str) -> List[dict]:
        """Find all entries with a given provenance label."""
        return [
            e.to_dict() for e in self._entries
            if e.provenance_label == label
        ]

    def get_entries_in_epoch_range(self, start: int, end: int) -> List[dict]:
        return [
            e.to_dict() for e in self._entries
            if start <= e.epoch <= end
        ]

    def get_graph(self) -> dict:
        """Get full provenance graph as adjacency map."""
        graph = {}
        for entry in self._entries:
            name = entry.belief_name
            if name not in graph:
                graph[name] = []
            graph[name].append({
                'event_id': entry.event_id,
                'provenance': entry.provenance_label,
                'delta': entry.confidence_delta,
                'epoch': entry.epoch,
            })
        return graph
