# GoalState RFC

**Status**: Design — not implemented  
**Date**: 2026-06-08

---

## 1. Goal State Model

```python
class GoalState(Enum):
    ACTIVE      # exists, no movement data yet
    MOVING      # recent progress detected
    STALLED     # no progress for N days
    BLOCKED     # known blocker exists
    COMPLETED   # success criteria met
    ABANDONED   # user has given up
```

---

## 2. Transitions

```
                    ┌──────────┐
                    │ ACTIVE   │
                    └────┬─────┘
                         │ progress detected
                         ▼
                    ┌──────────┐
           ┌───────│ MOVING   │────────┐
           │       └──────────┘        │
           │          │                │
           │ N days   │ progress       │ blocker
           │ no prog  │ detected       │ exists
           ▼          ▼                ▼
     ┌──────────┐  ┌──────────┐  ┌──────────┐
     │ STALLED  │  │ MOVING   │  │ BLOCKED  │
     └──────────┘  └──────────┘  └────┬─────┘
           │                          │
           │ M days                    │ blocker
           │ no prog                   │ resolved
           ▼                          ▼
     ┌──────────┐              ┌──────────┐
     │ABANDONED │              │ MOVING   │
     └──────────┘              └──────────┘

     ANY ──── criteria met ────→ COMPLETED
     STALLED ── M days ──→ ABANDONED
```

### Default thresholds

| Threshold | Value | Configurable? |
|-----------|-------|---------------|
| N (ACTIVE→STALLED) | 7 days | Per goal |
| M (STALLED→ABANDONED) | 30 days | Per goal |

---

## 3. State → Risk mapping

| State | Risk level | Priority |
|-------|------------|----------|
| MOVING | None | Normal |
| ACTIVE | Low (no data yet) | Low |
| STALLED | Medium | Needs attention |
| BLOCKED | High | Immediate |
| ABANDONED | Critical | Review needed |
| COMPLETED | None | None |

---

## 4. GoalState + AIOSState

```python
AIOSState.goal_progress = {
    goals: [
        {
            id,
            title,
            state: GoalState,     # MOVING / STALLED / BLOCKED / ...
            stagnation_days: int,  # 0 if MOVING, N if STALLED
            blocker: str | None    # description if BLOCKED
        }
    ],
    summary: {
        moving: int,
        stalled: int,
        blocked: int,
        completed: int,
        abandoned: int
    }
}
```

Risk Engine reads `state=STALLED|BLOCKED|ABANDONED` → risk raised.  
Priority Engine reads `state=BLOCKED > STALLED > ACTIVE > MOVING` → priority order.

---

## 5. Implementation

### What exists already

```python
Goal.status      # "active" | "done" | "failed" | ...
```

### What to add

```python
Goal.last_activity_at: datetime | None   # updated on any goal-related event
```

### Calculator (pure function)

```python
def compute_goal_state(
    status: str,
    last_activity_at: datetime | None,
    blockers: list[str],
    success_criteria_met: bool
) -> GoalState:
    if success_criteria_met:
        return COMPLETED
    if status in ("done", "failed"):
        return COMPLETED if status == "done" else ABANDONED
    if blockers:
        return BLOCKED
    if last_activity_at is None:
        return ACTIVE
    days_since = (now - last_activity_at).days
    if days_since > 30:
        return ABANDONED
    if days_since > 7:
        return STALLED
    return MOVING
```

No Evidence Engine. No ProgressFormula DSL. No EventRule parser. One function, one new column.

---

## 6. What to build next

After GoalState is computed:

1. **Risk Engine** — STALLED >7d → risk, BLOCKED → risk
2. **Priority Engine** — sort goals: BLOCKED > STALLED > ACTIVE > MOVING
3. **AIOSState.goal_progress** — populated with GoalState per goal
4. **Cockpit** — render AIOSState: one screen, five sections

This is the entire MVP-critical path.
