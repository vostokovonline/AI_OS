# Goal Contract RFC

**Status**: Design — not implemented  
**Date**: 2026-06-08

---

## 1. The Problem

A Goal in AI-OS today:

```python
class Goal:
    title: str
    description: str
    goal_type: str          # achievable, continuous, exploratory, directional, meta
    status: str             # active, done, failed, ...
    progress: float         # 0.0 — set ad-hoc, no provenance
```

It cannot answer:

| Question | Why |
|----------|-----|
| What counts as progress? | No evidence rules |
| How is progress computed? | No progress formula |
| When is this goal "done"? | No measurable success criteria |
| What events prove advancement? | No link to event types |

The goal is an **empty bucket**. Progress is poured in manually from multiple places (`goal_executor_v2.py`, `goal_executor.py`, `execution_v3.py`, `progress_propagation.py`) with no contract governing how it should be filled.

---

## 2. Proposed Contract

```python
class Goal:
    title: str
    description: str
    goal_type: str
    status: str

    # ── The missing layer ──

    evidence_rules: list[EvidenceRule]     # what events = progress
    progress_formula: str                  # how to compute from evidence
    success_criteria: list[Criterion]      # what "done" looks like
```

### EvidenceRule

```
EvidenceRule {
    event_type: str,        # which event counts
    source: str,            # which subsystem
    weight: float,          # relative importance
    filter: dict,           # optional: only count events matching these conditions
    count_as: float         # contribution per event (or null = auto: 1/expected_count)
}
```

Example for "Read 20 books":

```python
evidence_rules = [
    EvidenceRule(
        event_type="artifact_created",
        source="any",
        weight=1.0,
        filter={"content_kind": "book_note"},
        count_as=0.05  # 1/20
    )
]
```

### ProgressFormula

| Formula | Meaning | Use case |
|---------|---------|----------|
| `weighted_sum` | `sum(event.weight * event.count_as)` | Achievable goals |
| `latest_value` | `last(metric.value)` | Continuous goals |
| `binary` | `any(success_criterion_met)` | Simple goals |
| `min` | `min(component_progress)` | Dependent milestones |
| `custom` | Custom function | Complex goals |

### SuccessCriterion

```
Criterion {
    type: "threshold" | "binary" | "trend" | "duration",
    target: Any,
    description: str
}
```

Examples:

```python
# Achievable: "ship MVP"
success_criteria = [
    Criterion(type="threshold", target=1.0, description="progress reaches 100%"),
    Criterion(type="binary", target=True, description="all must-pass tests pass")
]

# Continuous: "grow capital"
success_criteria = [
    Criterion(type="duration", target="90d", description="sustained for 90 days"),
    Criterion(type="trend", target=0.01, description="velocity > 0.01/day")
]
```

---

## 3. Architecture

With Goal Contract, the progress pipeline becomes:

```
Goal.evidence_rules
    ↓  match against System Events
Relevant Events
    ↓  apply Goal.progress_formula
Goal Progress
    ↓  check against Goal.success_criteria
Goal Completion
```

Each step is a **deterministic function of the Goal Contract** and the system event stream. No ad-hoc progress setting. No artificial ProgressEvent.

---

## 4. Example Goals with Contracts

### Example 1: Achievable — "Launch AI-OS MVP"

```python
evidence_rules = [
    {"event_type": "artifact_created", "weight": 0.3, "count_as": 0.1},    # RFC doc
    {"event_type": "execution_completed", "weight": 0.3, "count_as": 0.1}, # feature ships
    {"event_type": "pr_merged", "weight": 0.2, "count_as": 0.1},          # code merged
    {"event_type": "test_passed", "weight": 0.2, "count_as": 0.1},        # tests green
]
progress_formula = "weighted_sum"
success_criteria = [
    {"type": "threshold", "target": 1.0}
]
```

### Example 2: Continuous — "Build capital to $100k"

```python
evidence_rules = [
    {"event_type": "metric_updated", "source": "zhamlik",
     "filter": {"metric": "net_worth"}, "weight": 1.0, "count_as": None},  # auto: 1/expected
]
progress_formula = "latest_value"  # progress = current_net_worth / 100000
success_criteria = [
    {"type": "threshold", "target": 100000},
    {"type": "duration", "target": "90d", "description": "sustain above $100k for 90 days"}
]
```

### Example 3: Exploratory — "Understand Kubernetes networking"

```python
evidence_rules = [
    {"event_type": "observation_recorded",
     "filter": {"topics": ["kubernetes", "networking"]},
     "weight": 0.5, "count_as": 0.05},
    {"event_type": "artifact_created",
     "filter": {"content_kind": "note", "tags": ["k8s"]},
     "weight": 0.5, "count_as": 0.1},
]
progress_formula = "weighted_sum"
success_criteria = [
    {"type": "binary", "target": True, "description": "can explain pod networking end-to-end"}
]  # note: exploratory goals may never be "done" — criteria are optional
```

### Example 4: Directional — "Write daily"

```python
evidence_rules = [
    {"event_type": "observation_recorded",
     "filter": {"tags": ["writing"]},
     "weight": 1.0, "count_as": 1.0},  # each writing session = 1
    {"event_type": "artifact_created",
     "filter": {"content_kind": "writing"},
     "weight": 2.0, "count_as": 1.0},  # published artifact = 2
]
progress_formula = "latest_value"  # progress = streak_days / target_streak
success_criteria = []  # directional goals have no "done" — they are ongoing practice
```

---

## 5. What This Unlocks

| Feature | Source |
|---------|--------|
| Progress computation | `evidence_rules` + `progress_formula` |
| Goal completion detection | `success_criteria` checked against progress |
| Risk: no evidence | `evidence_rules` matched against 0 events in N days |
| Risk: wrong trajectory | `progress_formula` applied to recent vs old evidence |
| Priority: focus needed | Goals with evidence gap > expectation gap |
| Traceability | Every progress % traces to specific events matching `evidence_rules` |
| Goal template library | Reusable `evidence_rules` per goal type |

---

## 6. Migration Path

1. Add `evidence_rules`, `progress_formula`, `success_criteria` columns to Goal model (all nullable)
2. Backfill for existing goals: infer from `goal_type` (e.g., achievable → weighted_sum, directional → latest_value)
3. Build ProgressCalculator that reads Goal Contract + event stream
4. Replace ad-hoc `goal.progress = X` with computed progress
5. UI: show evidence breakdown when clicking progress bar

---

## 7. Non-Goals

- Goal Contract does not define *how* to achieve the goal (that is planning)
- Goal Contract does not define *priority* (that is the Priority Engine)
- Goal Contract does not replace goal decomposition (parent/child structure)
- Goal Contract does not require changes to the Execution Kernel
