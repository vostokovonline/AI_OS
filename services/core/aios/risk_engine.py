"""Risk Engine — converts GoalState into risk items.

Pure functions, zero imports from experience/epistemic/cognitive layers.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def evaluate_goal_state(
    goal_id: str,
    title: str,
    state: str,
    stagnation_days: int,
) -> Optional[dict]:
    risk = None
    if state == "BLOCKED":
        risk = {
            "title": f"Goal blocked: {title}",
            "level": RiskLevel.HIGH.value,
            "source": "goal_state",
            "detail": "Goal has an active blocker preventing progress",
            "goal_id": goal_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    elif state == "ABANDONED":
        risk = {
            "title": f"Goal abandoned: {title}",
            "level": RiskLevel.CRITICAL.value,
            "source": "goal_state",
            "detail": f"No activity for {stagnation_days} days",
            "goal_id": goal_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    elif state == "STALLED":
        level = RiskLevel.MEDIUM.value if stagnation_days <= 14 else RiskLevel.HIGH.value
        risk = {
            "title": f"Goal stalled: {title}",
            "level": level,
            "source": "goal_state",
            "detail": f"No progress for {stagnation_days} days",
            "goal_id": goal_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    return risk


def evaluate_all_goal_states(goal_states: list[dict]) -> list[dict]:
    risks = []
    for gs in goal_states:
        risk = evaluate_goal_state(
            goal_id=gs.get("id", ""),
            title=gs.get("title", ""),
            state=gs.get("state", "ACTIVE"),
            stagnation_days=gs.get("stagnation_days", 0),
        )
        if risk:
            risks.append(risk)
    return risks
