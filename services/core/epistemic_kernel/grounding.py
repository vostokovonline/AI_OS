"""
Grounding Checkpoint — semantic state snapshot for deterministic replay.

MIRRORS SnapshotManager in execution_dynamics.

Each grounding checkpoint captures:
  - epoch: interpretation epoch at checkpoint time
  - observation_count: total observations processed
  - beliefs: snapshot of all belief states
  - motifs: snapshot of all motif states
  - attractors: snapshot of all attractor states
  - journal_size: number of journal events at checkpoint

Recovery from grounding checkpoint + journal replay after checkpoint
should reconstruct the exact same epistemic state.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import time
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class GroundingCheckpoint:
    """
    Semantic state snapshot for deterministic replay.
    """
    checkpoint_id: str
    epoch: int
    observation_count: int
    journal_size: int
    created_at: float
    beliefs: dict = field(default_factory=dict)
    motifs: dict = field(default_factory=dict)
    attractors: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class GroundingManager:
    """
    Manages grounding checkpoints — semantic state snapshots.

    MIRRORS SnapshotManager in execution_dynamics.

    Grounding checkpoints enable:
      - O(1) semantic recovery (load checkpoint + replay tail)
      - Deterministic replay verification
      - Re-grounding after drift correction
    """

    def __init__(self, max_checkpoints: int = 10):
        self._checkpoints: List[GroundingCheckpoint] = []
        self._max_checkpoints = max_checkpoints

    def create(
        self,
        epoch: int,
        observation_count: int,
        beliefs: dict,
        motifs: dict,
        attractors: dict,
        journal_size: int,
    ) -> str:
        checkpoint_id = f"gck:{uuid.uuid4().hex[:8]}"
        cp = GroundingCheckpoint(
            checkpoint_id=checkpoint_id,
            epoch=epoch,
            observation_count=observation_count,
            journal_size=journal_size,
            created_at=time.time(),
            beliefs=dict(beliefs),
            motifs=dict(motifs),
            attractors=dict(attractors),
        )
        self._checkpoints.append(cp)
        self._prune()
        logger.info(f"grounding_checkpoint_created id={checkpoint_id} epoch={epoch}")
        return checkpoint_id

    def load_latest(self) -> Optional[GroundingCheckpoint]:
        if not self._checkpoints:
            return None
        return self._checkpoints[-1]

    def list_checkpoints(self) -> list:
        return [cp.to_dict() for cp in self._checkpoints]

    def _prune(self):
        while len(self._checkpoints) > self._max_checkpoints:
            removed = self._checkpoints.pop(0)
            logger.info(f"grounding_checkpoint_pruned id={removed.checkpoint_id}")

    def get_stats(self) -> dict:
        return {
            'total_checkpoints': len(self._checkpoints),
            'latest_epoch': self._checkpoints[-1].epoch if self._checkpoints else 0,
        }
