"""
CADENCE SCHEDULER FOR SYSTEMIC GOALS
"""
from datetime import datetime, timedelta

def should_run(goal: dict, last_run: datetime | None) -> bool:
    cadence = goal.get("policy", {}).get("cadence_days")
    if not cadence:
        return False

    if last_run is None:
        return True

    return datetime.utcnow() - last_run >= timedelta(days=cadence)
