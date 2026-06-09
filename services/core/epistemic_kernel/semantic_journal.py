"""
Semantic Journal — append-only log of all interpretation events.

MIRRORS DispatchJournal in execution_dynamics.

Event types:
  OBSERVATION     — raw input to the semantic layer
  BELIEF_UPDATE   — belief confidence changed
  MOTIF_UPDATE    — motif strength changed
  ATTRACTOR_UPDATE — attractor weight changed
  DRIFT_ATTENUATION — drift correction applied
  GROUNDING       — grounding checkpoint created
  EPOCH_ADVANCE   — interpretation epoch advanced
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time
import uuid


@dataclass
class InterpretationEvent:
    """
    A single append-only interpretation event.

    Fields mirror JournalEntry in execution semantics.
    """
    event_id: str
    event_type: str          # OBSERVATION, BELIEF_UPDATE, MOTIF_UPDATE, ...
    timestamp: float
    epoch: int
    detail: str = ""
    data: dict = field(default_factory=dict)

    # Causality chain (mirrors prev_entry_id in execution journal)
    prev_event_id: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class SemanticJournal:
    """
    Append-only log of interpretation events.

    Properties:
      - Immutable: events are never modified after append
      - Causal: each event references its predecessor
      - Ordered: events are ordered by timestamp + sequence
    """

    def __init__(self):
        self._events: List[InterpretationEvent] = []
        self._sequence = 0

    def _next_id(self) -> str:
        self._sequence += 1
        return f"iev:{int(time.time())}:{self._sequence:06d}"

    def _append(self, event_type: str, detail: str, epoch: int, data: dict) -> InterpretationEvent:
        prev_id = self._events[-1].event_id if self._events else None
        event = InterpretationEvent(
            event_id=self._next_id(),
            event_type=event_type,
            timestamp=time.time(),
            epoch=epoch,
            detail=detail,
            data=data,
            prev_event_id=prev_id,
        )
        self._events.append(event)
        return event

    def append_observation(
        self,
        signal: str,
        value: float,
        source: str,
        context: dict,
        epoch: int,
        observation_index: int,
    ) -> InterpretationEvent:
        return self._append(
            event_type='OBSERVATION',
            detail=f"signal={signal} value={value} source={source}",
            epoch=epoch,
            data={
                'signal': signal,
                'value': value,
                'source': source,
                'context': context,
                'observation_index': observation_index,
            },
        )

    def append_belief_update(
        self,
        belief_name: str,
        previous_confidence: float,
        new_confidence: float,
        provenance: str,
        epoch: int,
    ) -> InterpretationEvent:
        return self._append(
            event_type='BELIEF_UPDATE',
            detail=f"belief={belief_name} {previous_confidence}→{new_confidence} via={provenance}",
            epoch=epoch,
            data={
                'belief_name': belief_name,
                'previous_confidence': previous_confidence,
                'new_confidence': new_confidence,
                'provenance': provenance,
            },
        )

    def append_motif_update(
        self,
        motif_name: str,
        previous_strength: float,
        new_strength: float,
        recurrence: int,
        provenance: str,
        epoch: int,
    ) -> InterpretationEvent:
        return self._append(
            event_type='MOTIF_UPDATE',
            detail=f"motif={motif_name} {previous_strength}→{new_strength} rec={recurrence}",
            epoch=epoch,
            data={
                'motif_name': motif_name,
                'previous_strength': previous_strength,
                'new_strength': new_strength,
                'recurrence': recurrence,
                'provenance': provenance,
            },
        )

    def append_event(
        self,
        event_type: str,
        detail: str,
        epoch: int,
        data: dict = None,
    ) -> InterpretationEvent:
        return self._append(
            event_type=event_type,
            detail=detail,
            epoch=epoch,
            data=data or {},
        )

    def replay(self, since_event_id: Optional[str] = None) -> List[InterpretationEvent]:
        """
        Replay journal events from a given event_id (exclusive).

        MIRRORS WAL.replay_after().
        """
        if not since_event_id:
            return list(self._events)

        result = []
        found = False
        for event in self._events:
            if event.event_id == since_event_id:
                found = True
                continue
            if found:
                result.append(event)
        return result

    def get_events_by_type(self, event_type: str) -> List[InterpretationEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def get_events_since_epoch(self, epoch: int) -> List[InterpretationEvent]:
        return [e for e in self._events if e.epoch >= epoch]

    def get_stats(self) -> dict:
        if not self._events:
            return {'total_events': 0}
        types = {}
        for e in self._events:
            types[e.event_type] = types.get(e.event_type, 0) + 1
        return {
            'total_events': len(self._events),
            'by_type': types,
            'first_event': self._events[0].event_id,
            'last_event': self._events[-1].event_id,
        }
