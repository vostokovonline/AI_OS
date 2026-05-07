"""
Write Barrier - Intent Sentinel

CRITICAL: This monitors APPLICATION-LAYER intent to mutate.
NOT SQL-level monitoring (that's ORM responsibility).
"""
from typing import List

class WriteBarrier:
    """
    Enforces write phase boundary in decision system.

    Architecture Contract:
        Phase 1 (COLLECT): Pure computation, NO writes allowed
        Phase 2 (APPLY):   Writes explicitly allowed

    Violation = architectural leak (workflow thinking remains).
    """

    def __init__(self):
        self.enabled = False
        self.allowed = False
        self.violations: List[str] = []

    def enable(self):
        """Start monitoring. Writes are prohibited until allow()."""
        self.enabled = True
        self.allowed = False
        self.violations.clear()

    def allow(self):
        """Enable write phase. THIS is the ONLY valid place for writes."""
        self.allowed = True

    def check(self, source: str):
        """
        Called before any state mutation.

        Raises: RuntimeError if write attempted before barrier.
        """
        if self.enabled and not self.allowed:
            self.violations.append(source)
            raise RuntimeError(
                f"WRITE BEFORE BARRIER from {source}\n"
                f"Phase 1 (COLLECT) must be pure computation.\n"
                f"Use barrier.allow() before Phase 2 (APPLY)."
            )

    def reset(self):
        """Reset for next test."""
        self.enabled = False
        self.allowed = False
        self.violations.clear()

# Global singleton for testing
WRITE_BARRIER = WriteBarrier()
