# Goal Progress Engine RFC

**Status**: Design — not implemented  
**Date**: 2026-06-08

---

## 1. The Problem

AI-OS today stores goals and tracks their status (active/done/failed), but cannot answer:

| Question | Today | Need |
|----------|-------|------|
| How complete is this goal? | `progress` float (set ad-hoc) | Computed from events |
| How fast is it progressing? | Not tracked | Velocity over time |
| Is it stalled? | Not tracked | Days since last progress event |
| What counts as progress? | Not defined | Per goal-type rules |
| Why did progress change? | Not traceable | Attribution to events |

The `progress` column on `Goal` is a stored float that gets set in arbitrary places (`goal_executor_v2.py`, `goal_executor.py`, `execution_v3.py`). There is no consistent model for what progress means.

---

## 2. Proposed Architecture

```
Goal
  ↓
ProgressCalculator (per goal_type)
  ↓
ProgressEvents (append-only log)
  ↓
ProgressSnapshot (computed at time t)
  ↓
Metrics (velocity, stagnation, completion_rate)
```

### Key principle: Progress is computed, not stored

The `progress` column on Goal becomes **derived** from a `ProgressEvent` log, not a manually-set float.

```
Goal.progress  ←  ProgressSnapshot(t).value
                    ↓
              replay(ProgressEvents up to t)
```

---

## 3. Core Artifact: ProgressEvent

```
ProgressEvent(
    id: str,
    goal_id: str,
    event_type: ProgressEventType,
    delta: float,           # contribution to progress (0.0 to 1.0)
    weight: float,          # weight of this event relative to total (default 1.0)
    source: str,            # what produced this event
    timestamp: datetime,
    metadata: dict          # domain-specific payload
)
```

### Event Types

| Type | Meaning | Delta |
|------|---------|-------|
| `step_completed` | One step of a multi-step goal done | `1/total_steps` |
| `milestone_reached` | Significant waypoint passed | Configurable |
| `artifact_created` | Verifiable output produced | `1/total_artifacts` |
| `observation_recorded` | Evidence of progress observed | Configurable |
| `progress_statement` | LLM or user: "this is X% done" | `X/100` |
| `subgoal_completed` | Child goal completed | Varied (aggregate) |

### Replay Rule

```
progress(goal_id, t) =
    sum(ProgressEvent.delta * ProgressEvent.weight
        for all ProgressEvent where goal_id = goal_id AND timestamp <= t)
    / max(1, total_weight)
```

`total_weight` is goal-specific: number of steps, target artifacts, etc.

If `delta * weight >= total_weight` → 100%.

---

## 4. ProgressCalculator (per goal_type)

Each goal type defines its own progress semantics:

### AchievableGoal

```
total_weight = number_of_steps OR target_artifact_count

Events that count:
  - step_completed: delta = 1/total_steps
  - artifact_created: delta = 1/total_artifacts
  - subgoal_completed: delta = weight_of_subgoal
```

### ContinuousGoal

```
No 100%. Progress is trend-based.

Events that count:
  - measurement_recorded: delta = improvement / target_improvement
  - metric_updated: delta = (current - baseline) / (target - baseline)
```

### ExploratoryGoal

```
No 100% by definition. Progress = knowledge accumulated.

Events that count:
  - observation_recorded: delta = 1/expected_observations
  - insight_generated: delta = configurable
  - question_answered: delta = 1/total_questions
```

### DirectionalGoal

```
Progress = adherence to direction.

Events that count:
  - aligned_action: delta = positive
  - misaligned_action: delta = negative (regress)
```

### MetaGoal

```
Progress = capability improvement.

Events that count:
  - skill_acquired: delta = 1/target_skills
  - cycle_completed: delta = 1/expected_cycles
```

---

## 5. Derived Metrics

### Velocity

```
velocity(goal_id, t, window_days=7) =
    (progress(t) - progress(t - window_days)) / window_days
```

Range: `[-1.0, 1.0]` per day. Negative means regression.

### Stagnation

```
stagnation(goal_id, t) =
    days_since_last_progress_event(goal_id, t)
```

Thresholds:
- `> 3 days`: mild stagnation
- `> 7 days`: significant stagnation
- `> 14 days`: critical (goal likely abandoned)

### Completion Rate

```
completion_rate(goal_id) =
    completed_events / expected_events
```

### Trajectory

```
trajectory(goal_id, t) =
    improving  if velocity(t) > velocity(t - 7d) + threshold
    declining  if velocity(t) < velocity(t - 7d) - threshold
    stable     otherwise
```

---

## 6. Integration with AIOSState

```
AIOSState.goal_progress = {
    goals: [
        {
            id: goal.id,
            title: goal.title,
            status: goal.status,
            progress: ProgressSnapshot(goal.id, now).value,
            velocity: velocity(goal.id, now, 7),
            stagnation_days: stagnation(goal.id, now),
            trajectory: trajectory(goal.id, now),
            last_progress: last_progress_event(goal.id).timestamp
        }
    ],
    summary: {
        total_active: int,
        progressing: int,     # velocity > 0.01
        stalled: int,         # stagnation > 7 days
        completed_this_week: int,
        avg_velocity: float
    }
}
```

---

## 7. What This Unlocks

| Feature | Depends on |
|---------|-----------|
| Goal Progress section in Cockpit | ProgressSnapshot per goal |
| Risk: stalled goal | stagnation > 7d |
| Risk: declining trajectory | trajectory = declining |
| Priority: focus on stalled goals | stagnation + velocity |
| Daily Briefing: "X goals made progress" | avg_velocity > 0 |
| Weekly Review: progress delta | ProgressSnapshot(t) - ProgressSnapshot(t-7) |
| Traceability: "why did progress change?" | ProgressEvent log |

---

## 8. Migration from Current State

Currently `Goal.progress` is a stored float set in multiple places:

1. `goal_executor_v2.py:1704`: `goal.progress = len(artifacts) / len(skills)`
2. `goal_executor_v2.py:2627`: `goal.progress = 0.6`
3. `goal_executor_v2.py:2743`: `goal.progress = evaluation_result.confidence`
4. `progress_propagation.py`: propagates child → parent

**Migration path:**
1. Create `ProgressEvent` model and log (new table)
2. Keep `Goal.progress` as a cached derived field updated on read or by background refresh
3. Instrument current progress-setting code to also emit `ProgressEvent`
4. Replace ad-hoc progress setting with ProgressCalculator calls
5. Eventually remove direct `Goal.progress` writes

---

## 9. Non-Goals

- Goal Progress Engine does not *set* goal priority (Priority Engine does)
- Goal Progress Engine does not *detect* risks (Risk Engine does, using progress metrics)
- Goal Progress Engine does not *replan* goals
- Goal Progress Engine does not replace the need for goal definitions (steps, milestones, targets)
