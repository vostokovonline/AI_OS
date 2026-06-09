"""Priority Engine — sorts goals by urgency based on GoalState."""

from typing import Optional


# ── Priority order (lower = higher priority) ────────────────────────────

STATE_PRIORITY = {
    "BLOCKED": 0,
    "CRITICAL": 1,
    "ABANDONED": 2,
    "STALLED": 3,
    "ACTIVE": 4,
    "MOVING": 5,
    "COMPLETED": 6,
}


class PriorityItem:
    """A goal with its computed priority level."""

    def __init__(
        self,
        goal_id: str,
        title: str,
        state: str,
        priority_score: int,
        reason: Optional[str] = None,
    ):
        self.goal_id = goal_id
        self.title = title
        self.state = state
        self.priority_score = priority_score
        self.reason = reason

    def to_dict(self):
        return {
            "goal_id": self.goal_id,
            "title": self.title,
            "state": self.state,
            "priority_score": self.priority_score,
            "reason": self.reason,
        }

    def __lt__(self, other):
        return self.priority_score < other.priority_score

    def __repr__(self):
        return f"<Priority {self.state}: {self.title} (score={self.priority_score})>"


class PriorityEngine:
    """Sorts goals by urgency for the user's next-action decision.

    Priority order:
      1. BLOCKED  — immediate attention required
      2. STALLED  — needs focus to restart
      3. ACTIVE   — recently created, no data yet
      4. MOVING   — on track, lowest attention need
      5. COMPLETED — done
    """

    def prioritize(self, goal_states: list[dict]) -> list[PriorityItem]:
        items = []
        for gs in goal_states:
            state = gs.get("state", "ACTIVE")
            stagnation = gs.get("stagnation_days", 0)
            reason = self._reason(state, stagnation)
            items.append(PriorityItem(
                goal_id=gs.get("id", ""),
                title=gs.get("title", ""),
                state=state,
                priority_score=STATE_PRIORITY.get(state, 99),
                reason=reason,
            ))
        items.sort()
        return items

    @staticmethod
    def _reason(state: str, stagnation_days: int) -> str:
        reasons = {
            "BLOCKED": "Requires immediate attention — blocker detected",
            "ABANDONED": f"Abandoned — no activity for {stagnation_days} days",
            "STALLED": f"Stalled — no progress for {stagnation_days} days",
            "ACTIVE": "New goal — no activity yet",
            "MOVING": "On track",
            "COMPLETED": "Done",
        }
        return reasons.get(state, "")


priority_engine = PriorityEngine()
