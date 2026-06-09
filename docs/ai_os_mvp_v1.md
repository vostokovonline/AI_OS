# AI-OS MVP v1 — Architecture

**Status**: Product architecture — not implementation spec  
**Date**: 2026-06-08

---

## 1. Central Artifact

The single reason a user opens AI-OS every day. AI-OS is an **operating system** — it must show not only state, but also running processes and system events.

```
AIOSState
├── world_state         (what is the state of my world?)
├── goal_progress       (how am I advancing toward what matters?)
├── active_executions   (what is the system doing right now?)
├── event_journal       (what has happened recently?)
└── risks / priorities  (what threatens progress? what should I do?)
```

Every field is **traceable** — clicking any number shows the chain of events that produced it.

### Why five sections, not three

Every operating system has:

| OS concept | AI-OS equivalent |
|-----------|-----------------|
| Current state | World state |
| File system | Goal progress |
| Running processes | Active executions |
| System log | Event journal |
| Alerts / scheduler | Risks / priorities |

Without `active_executions`, the system describes the world but does not show what it is *doing* about it. Without `event_journal`, the system has state but no way to understand how it got there.

---

## 2. Two Circuits

### Primary Circuit (what the user sees)

```
AIOSState(t)
    ↓
Cockpit (render)
```

One screen. Four sections. Zero navigation needed for the daily check-in.

### Operational Circuit (how the user investigates)

```
AIOSState
    ↓  click on any number
Attribution
    ↓  "why did this change?"
Event Journal
    ↓  "what events caused this?"
Sources
```

This is not a separate page. It is a **drilldown capability** on every field of AIOSState.

---

## 3. Data Sources (already exist)

| AIOSState field | Source | Status |
|----------------|--------|--------|
| goal_progress | PostgreSQL (goals) | ✅ |
| system_state.financial | Zhamlik ledger | ✅ |
| system_state.execution | Execution Kernel | ✅ |
| risks | Derived from goals + financial + execution | 🔄 need computation |
| priorities | Derived from goals + risks | 🔄 need computation |
| event_journal | Execution Kernel journal + goal events | ✅ |
| timeline | Replay(event_log) | ✅ |

No new subsystems. No new databases. No new kernels.

---

## 4. Traceability Principle

Every scalar in AIOSState must answer three questions:

| Question | Mechanism |
|----------|-----------|
| **What is the current value?** | State query |
| **Why did it change?** | Recent events delta |
| **What events produced this?** | Event journal drilldown |

This is the same property as Zhamlik's traceability: every `$` in Net Worth traces to a transaction. Every `%` in Goal Progress traces to an event.

---

## 5. What Is NOT in MVP v1

- PersonalModel (deferred — AIOSState supersedes it for now)
- Epistemic Kernel (deferred — traceability covers explainability for MVP)
- Daily Loop (deferred — AIOSState renders at any time; loop emerges from habit, not software)
- BeliefEvent / Fact / PersonalEvent (deferred — existing JournalEntry + goal events suffice)
- New atom search (deferred — AI-OS as a product doesn't need one for MVP)

---

## 6. Implementation Order

| Step | What | Depends on |
|------|------|------------|
| 1 | `AIOSState` schema — define the four sections and their fields | Nothing |
| 2 | `goal_progress` provider — query goals, compute progress metrics | PostgreSQL |
| 3 | `system_state` provider — aggregate financial + execution + health | Zhamlik, Kernel |
| 4 | `risks` provider — threshold-based rule engine over state | Steps 2–3 |
| 5 | `priorities` provider — rank next actions by risk + progress | Step 4 |
| 6 | Traceability — drilldown from any field to event journal | Steps 2–5 |
| 7 | Cockpit — one screen, four sections, clickable numbers | Steps 1–6 |

**Step 1 is the RFC. Everything else is implementation.**

---

## 7. Why This Works

- It answers the user's morning question without requiring a new atom.
- It uses only existing subsystems.
- It makes traceability a property of the interface, not a separate model.
- It defers epistemic complexity to a later layer.
- It provides immediate value from day one.
