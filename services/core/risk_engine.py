"""Risk Engine — converts GoalState + other signals into risk items."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskItem:
    """A single risk with its severity and context."""

    def __init__(
        self,
        title: str,
        level: RiskLevel,
        source: str,
        detail: str,
        goal_id: Optional[str] = None,
    ):
        self.title = title
        self.level = level
        self.source = source
        self.detail = detail
        self.goal_id = goal_id
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self):
        return {
            "title": self.title,
            "level": self.level.value,
            "source": self.source,
            "detail": self.detail,
            "goal_id": self.goal_id,
            "timestamp": self.timestamp.isoformat(),
        }


class RiskEngine:
    """Pure-function risk detector.

    Reads goal states and produces risk items.
    No side effects, no storage.
    """

    def evaluate_goal_state(
        self,
        goal_id: str,
        title: str,
        state: str,
        stagnation_days: int,
    ) -> Optional[RiskItem]:
        if state == "BLOCKED":
            return RiskItem(
                title=f"Goal blocked: {title}",
                level=RiskLevel.HIGH,
                source="goal_state",
                detail="Goal has an active blocker preventing progress",
                goal_id=goal_id,
            )
        if state == "ABANDONED":
            return RiskItem(
                title=f"Goal abandoned: {title}",
                level=RiskLevel.CRITICAL,
                source="goal_state",
                detail=f"No activity for {stagnation_days} days",
                goal_id=goal_id,
            )
        if state == "STALLED":
            level = (
                RiskLevel.MEDIUM if stagnation_days <= 14 else RiskLevel.HIGH
            )
            return RiskItem(
                title=f"Goal stalled: {title}",
                level=level,
                source="goal_state",
                detail=f"No progress for {stagnation_days} days",
                goal_id=goal_id,
            )
        return None

    def evaluate_all_goal_states(
        self, goal_states: list[dict]
    ) -> list[RiskItem]:
        risks = []
        for gs in goal_states:
            risk = self.evaluate_goal_state(
                goal_id=gs.get("id", ""),
                title=gs.get("title", ""),
                state=gs.get("state", "ACTIVE"),
                stagnation_days=gs.get("stagnation_days", 0),
            )
            if risk:
                risks.append(risk)
        return risks


risk_engine = RiskEngine()
