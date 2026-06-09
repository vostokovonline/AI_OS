# Goal Evidence RFC

**Status**: Design — not implemented  
**Date**: 2026-06-08

---

## 1. The Question

Not "how do we track progress?" but:

> **What event in the system counts as evidence that a goal has advanced?**

---

## 2. Why Not ProgressEvent

A ProgressEvent would be an artificial event type created solely for progress tracking. This introduces a parallel accounting system on top of real system events.

| Problem | Consequence |
|---------|-------------|
| Double-entry | Real event + ProgressEvent = two writes for one fact |
| No traceability | ProgressEvent references the real event, not the other way around |
| Fragile | If ProgressEvent is lost, progress disappears even though real events exist |
| Subjective | ProgressEvent can be created without real evidence (a user says "I made progress") |

**The real events already exist:** ExecutionCompleted, ObservationRecorded, ArtifactCreated, TransactionImported, SkillAcquired. Progress is a **classification** over these, not a new event type.

---

## 3. Architecture

```
System Events  (what happened — the only primitives)
    ↓
Evidence       (classification: which events advance which goals)
    ↓
Progress       (computed from evidence)
```

### Evidence is a link, not an event

```
Evidence {
    goal_id: str,
    event_id: str,            // reference to a real system event
    event_type: str,          // what kind of event
    contribution: float,      // how much this advances the goal
    source: str,              // which subsystem produced it
    timestamp: datetime,
    reason: str               // why this counts as evidence
}
```

Evidence does not create new events. It **links existing events to goals** and **quantifies their contribution**.

---

## 4. What Counts as Evidence

Depends on goal type:

| Goal Type | Evidence Sources | Example |
|-----------|-----------------|---------|
| Achievable | artifact_created, step_completed, execution_completed, subgoal_done | "PR merged" → goal progress +5% |
| Continuous | metric_updated, observation_recorded | "net_worth +$200" → goal progress +2% |
| Exploratory | observation_recorded, insight_generated, question_answered | "Learned about X" → progress +10% |
| Directional | aligned_action, misaligned_action | "Stuck to budget" → progress +3% |
| Meta | skill_acquired, cycle_completed | "Wrote first Rust program" → progress +25% |

### Priority Rules

| Evidence | Priority | Reason |
|----------|----------|--------|
| execution_completed | Highest | Verifiable output produced |
| artifact_created | High | Tangible result |
| observation_recorded | Medium | Self-reported but traceable |
| progress_statement | Lowest | Unverifiable claim |

---

## 5. Evidence Providers

Each subsystem acts as an evidence provider:

| Provider | Events | Evidence for |
|----------|--------|-------------|
| Execution Kernel | DISPATCHED, STARTED, COMPLETED, FAILED | Achievable, execution goals |
| Zhamlik | snapshot_imported, trade_executed | Financial goals |
| Cognitive OS | observation_recorded | Exploratory, reflection goals |
| Skills | skill_acquired, artifact_created | Meta, learning goals |
| User | progress_statement (explicit) | All goal types (lowest priority) |

Each provider emits evidence when its events occur. The provider does not need to know which goal the evidence serves — it simply announces "this event happened." The Goal Evidence layer connects events to goals.

---

## 6. Progress from Evidence

```
progress(goal_id, t) =
    sum(evidence.contribution for evidence in EvidenceSet(goal_id, t))
    / total_expected_contribution
```

### EvidenceSet

```
EvidenceSet(goal_id, t) = {
    evidence
    for evidence in all Evidence
    where evidence.goal_id = goal_id
    and evidence.timestamp <= t
}
```

### Contribution per evidence

By default, each evidence event contributes equally: `contribution = 1 / expected_count`.

For example, a goal "Read 20 books" has `expected_count = 20` (target artifact count). Each `artifact_created` event (book finished) contributes `1/20 = 0.05 (5%)`.

Goals can override contribution weights when evidence sources are uneven.

---

## 7. Derived Metrics (from Evidence, not ProgressEvent)

| Metric | Formula | Meaning |
|--------|---------|---------|
| Completion | sum(contribution) / expected | How much is done |
| Velocity | completion(t) - completion(t-7d) | Rate of evidence accumulation |
| Stagnation | days since last evidence | How long without proof of progress |
| Trajectory | velocity(t) > velocity(t-7d)? | Accelerating or decelerating |
| Evidence diversity | unique evidence types | Breadth of progress (not just one type) |

All metrics trace to specific evidence events. Clicking "63% complete" shows the evidence list.

---

## 8. Comparison: ProgressEvent vs Evidence

| Aspect | ProgressEvent | Evidence |
|--------|--------------|----------|
| Nature | Artificial event | Classification of real events |
| Storage | Parallel table | Link table (goal_id ↔ event_id) |
| Traceability | Self-referential | Points to real events |
| Risk of fraud | Can create events without evidence | Only links existing events |
| Provider burden | Must emit progress-specific events | Reuses existing event stream |
| Query | "all ProgressEvent for goal X" | "all events linked to goal X" |

---

## 9. Integration with AIOSState

Evidence layer completes the chain:

```
World State
    ↓
System Events  (execution, financial, observation)
    ↓
Evidence       (linked events → goal progress)
    ↓
Goal Progress  (completion, velocity, stagnation, trajectory)
    ↓
Risks          (stagnation > threshold, negative trajectory)
    ↓
Priorities     (which goal needs attention most)
```

Each step is a deterministic function of the step before it. No step invents new data — it only classifies or aggregates existing data.

---

## 10. Open Questions

| Question | Status |
|----------|--------|
| Who links events to goals? Provider or consumer? | Open — probably consumer (goal system subscribes to events) |
| Should evidence be stored or computed on demand? | Open — stored link table for performance, recomputable from events |
| What if an event serves multiple goals? | Open — one evidence link per goal; event can have N links |
| Can evidence be negative (regression)? | Open — yes, for directional goals (misaligned_action) |
| Is `contribution` float or function? | Open — float for v1, per-goal-type function for v2 |
