# Personal Atom RFC-2 (v2)

**Question**: What is the minimal event from which PersonalModel(as_of) can be fully reconstructed by replay?

**Status**: Question open — Fact rejected as atom, Event proposed  
**Date**: 2026-06-08

---

## 1. The Criterion

Can we delete **all projections** (PersonalModel, FactLog, snapshots, caches) and reconstruct the full system history **solely from the atom**?

If yes → the atom is correct.
If no → the atom is a view, not a primitive.

JournalEntry passes this test: delete all snapshots, replay the journal, reconstruct ExecutionState.
LedgerEntry passes this test: delete all balance snapshots, replay the ledger, reconstruct FinancialSnapshot.

**Fact fails:** `goal_status = "active"` does not say when it was activated, what preceded it, what caused it. The event `GoalActivated(goal_123)` contains all of that; Fact is a derived projection.

---

## 2. What Must the Atom Satisfy?

| Property | Meaning |
|----------|---------|
| **Event** (not state) | Records a transition, not an assertion |
| **Atomic** | Cannot be decomposed into smaller events |
| **Immutable** | Once recorded, never changed |
| **Ordered** | Sequence is meaningful and deterministic |
| **Replayable** | Replaying atoms in order reconstructs state |
| **Domain-neutral** | One atom for financial, goal, execution, observation |
| **Causal** | Preserves what happened and why |

---

## 3. Architecture

```
Event                              (atom — what happened)
  ↓ replay (apply in order)
Fact (subject, predicate, object)  (derived — latest value per key)
  ↓ project
PersonalModel                      (derived — structured snapshot)
```

Event is the **only** stored primitive. Fact and PersonalModel are recomputed on demand.

---

## 4. Proposed Atom: PersonalEvent

```
PersonalEvent(
    id: str,
    timestamp: datetime,
    domain: str,            # "financial" | "goal" | "execution" | "observation" | "reflection" | "system"
    entity_id: str,         # which thing changed (e.g., "goal_123", "net_worth", "energy")
    event_type: str,        # what happened (e.g., "created", "activated", "completed", "recorded", "updated")
    data: dict,             # transition-specific payload (varies by event_type)
    source: str,            # how it entered the system ("ledger", "user", "cognitive", "system")
)
```

### Examples

```
// Financial domain
{
    id: "evt_001",
    timestamp: "2026-06-08T09:00",
    domain: "financial",
    entity_id: "net_worth",
    event_type: "snapshot_imported",
    data: {"value": 10200, "previous_value": 10000},
    source: "ledger"
}

// Goal domain
{
    id: "evt_002",
    timestamp: "2026-06-08T09:05",
    domain: "goal",
    entity_id: "goal_123",
    event_type: "activated",
    data: {"title": "Improve focus", "goal_type": "achievable"},
    source: "system"
}

// Observation domain
{
    id: "evt_003",
    timestamp: "2026-06-08T10:00",
    domain: "observation",
    entity_id: "obs_001",
    event_type: "recorded",
    data: {"content": "Felt focused today", "topics": ["focus", "energy"]},
    source: "user"
}

// Reflection domain
{
    id: "evt_004",
    timestamp: "2026-06-08T22:00",
    domain: "reflection",
    entity_id: "reflection_001",
    event_type: "written",
    data: {"insight": "Morning routine improves focus", "related_observations": ["obs_001"]},
    source: "system"
}

// Execution domain
{
    id: "evt_005",
    timestamp: "2026-06-08T11:00",
    domain: "execution",
    entity_id: "exec_001",
    event_type: "dispatched",
    data: {"goal_id": "goal_123", "execution_id": "exec_001"},
    source: "system"
}

// Derived metric domain
{
    id: "evt_006",
    timestamp: "2026-06-08T23:00",
    domain: "computed",
    entity_id: "momentum",
    event_type: "updated",
    data: {"value": 0.35, "previous_value": 0.28, "delta": 0.07},
    source: "system"
}
```

---

## 5. Replay Rule

```
replay(PersonalEvents, t):
    events = [e for e in PersonalEvents if e.timestamp <= t]
    events.sort(key=lambda e: e.timestamp)

    # Replay: for each entity, track its latest known state
    state = {}
    for event in events:
        apply(event, state)   # entity_id → state[entity_id] updated

    return PersonalModel(state)
```

`apply(event, state)` is domain-specific but simple:
- **financial/net_worth**: `state["net_worth"] = event.data["value"]`
- **goal/activated**: `state[event.entity_id] = {status: "active", ...}`
- **observation**: append to `state["observations"]`
- **computed**: `state[event.entity_id] = event.data["value"]`

The key difference from Fact-based approach: events carry *transitions*, not *values*. Replay reconstructs, not overwrites.

---

## 6. Verification: Can we reconstruct full history from atom alone?

### Test: Delete all projections, keep only `[PersonalEvent]`

| Artifact | Reconstructable? | How? |
|----------|-----------------|------|
| PersonalModel(t) | ✅ | replay(events up to t) |
| FactLog(t) | ✅ | derived from replay state at t |
| Goal history | ✅ | filter events by domain="goal", entity_id |
| Observation timeline | ✅ | filter events by domain="observation" |
| Financial timeline | ✅ | filter events by domain="financial" |
| Reflection stream | ✅ | filter events by domain="reflection" |
| Execution trace | ✅ | filter events by domain="execution" |
| Daily summary | ✅ | filter events by timestamp |

**Result: Full reconstruction is possible.** Every artifact is a projection of events.

---

## 7. Fact Revisited

Fact is no longer the atom. It is now the **materialized view** of Events:

```
Fact(subject, predicate, object) =
    latest PersonalEvent where
        entity_id = subject AND
        data[field] = object
```

Fact is a convenience layer for fast queries. It does not need its own storage — it is recomputed from Events on demand or cached periodically.

---

## 8. Symmetry

| Layer | Execution | Financial | Personal |
|-------|-----------|-----------|----------|
| **Atom** | JournalEntry | LedgerEntry | PersonalEvent |
| **Log** | Journal | Ledger | PersonalEventLog |
| **Derived** | ExecutionState | FinancialSnapshot | Fact + PersonalModel |
| **Replay** | Replay(Journal) | Replay(Ledger) | Replay(Events, t) |
| **Recovery** | Journal recovery | Ledger recovery | Replay from last snapshot |
| **Verification** | Hash chain | Double-entry (debit=credit) | Causal chain |

---

## 9. Open Questions

| Question | Status |
|----------|--------|
| Should PersonalEvent have prev_hash? | Open — not for v1, needed for integrity |
| Should PersonalEvent have a sequential ID within domain? | Open — maybe for ordering |
| Is `source` sufficient for provenance? | Open — for now, yes |
| How does PersonalEvent relate to existing JournalEntry? | Open — JournalEntry is a subtype (domain="execution"); PersonalEventLog may wrap kernel events |
| Is `computed` domain legitimate or should metrics be derived from replay? | Open — if replayable from primitives, computed is unnecessary |
| Does PersonalEvent span multiple users or one? | Open — v1: one user per PersonalModel |
