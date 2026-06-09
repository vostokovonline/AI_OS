"""Priority Engine — sorts goals by urgency based on GoalState.

Pure functions, zero imports from experience/epistemic/cognitive layers.
"""

STATE_PRIORITY = {
    "BLOCKED": 0,
    "CRITICAL": 1,
    "ABANDONED": 2,
    "STALLED": 3,
    "ACTIVE": 4,
    "MOVING": 5,
    "COMPLETED": 6,
}

_PRIORITY_REASONS = {
    "BLOCKED": "Requires immediate attention — blocker detected",
    "ABANDONED": "Abandoned — no activity for 30+ days",
    "STALLED": "Stalled — needs focus to restart",
    "ACTIVE": "New goal — no activity yet",
    "MOVING": "On track",
    "COMPLETED": "Done",
}


def prioritize(goal_states: list[dict]) -> list[dict]:
    items = []
    for gs in goal_states:
        state = gs.get("state", "ACTIVE")
        stagnation = gs.get("stagnation_days", 0)
        items.append({
            "goal_id": gs.get("id", ""),
            "title": gs.get("title", ""),
            "state": state,
            "priority_score": STATE_PRIORITY.get(state, 99),
            "reason": _reason(state, stagnation),
        })
    items.sort(key=lambda x: x["priority_score"])
    return items


def _reason(state: str, stagnation_days: int) -> str:
    if state == "ABANDONED":
        return f"Abandoned — no activity for {stagnation_days} days"
    if state == "STALLED":
        return f"Stalled — no progress for {stagnation_days} days"
    return _PRIORITY_REASONS.get(state, "")
