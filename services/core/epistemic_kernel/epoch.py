"""
Interpretation Epoch — version counter for the epistemic state.

MIRRORS fence token / lease epoch in execution_dynamics.

Epoch semantics:
  - Auto-increments on every belief/motif/attractor modification
  - Enables epoch-scoped queries and validation
  - Grounding checkpoints record the epoch for replay anchoring
  - Drift detection compares epoch advance rate against observation rate
"""

from typing import Optional
import time


class InterpretationEpoch:
    """
    Monotonic version counter for the epistemic model.

    Every state mutation (belief update, motif update, attractor update,
    observation recording) advances the epoch by 1.

    Properties:
      - Monotonic: epoch always increases
      - Observable: any component can check current epoch
      - Scoped: epoch scopes queries and validation
    """

    def __init__(self, initial: int = 0):
        self._current = initial
        self._created_at = time.time()
        self._advanced_at = self._created_at

    @property
    def current(self) -> int:
        return self._current

    def touch(self) -> int:
        """Advance epoch by 1. Returns new epoch value."""
        self._current += 1
        self._advanced_at = time.time()
        return self._current

    def advance_to(self, epoch: int) -> int:
        """
        Advance epoch to a specific value (for replay recovery).

        Only forward advances are permitted.
        """
        if epoch <= self._current:
            raise ValueError(
                f"Cannot regress epoch: {epoch} <= {self._current}"
            )
        self._current = epoch
        self._advanced_at = time.time()
        return self._current

    def to_dict(self) -> dict:
        return {
            'current': self._current,
            'created_at': self._created_at,
            'advanced_at': self._advanced_at,
        }
