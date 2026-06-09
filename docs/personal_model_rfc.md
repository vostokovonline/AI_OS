# PersonalModel v1 RFC

**Status**: Contract draft — not implemented  
**Date**: 2026-06-08

---

## 1. Purpose

PersonalModel is the central computed artifact of AI-OS. It answers one question:

> **What is the state of the person right now, and where are they going?**

It is not a stored document — it is a **deterministic function** of primitive data sources (financial, goal, execution, observation) evaluated at a point in time.

Everything else in the system is a projection of this model:
- Cockpit = `render(PersonalModel)`
- Daily Loop = `read → act → observe → recompute(PersonalModel)`
- Timeline = `[PersonalModel(t) for t in range]`
- Drift detection = `PersonalModel(t) − PersonalModel(t−1)`

---

## 2. Contract

```python
class PersonalModel:
    as_of: date                         # evaluation date

    # ── Primitive states (aggregated from subsystems) ──

    financial: FinancialState          # net worth, P&L, positions
    goals: GoalState                   # active, stalled, completed counts
    execution: ExecutionState          # journal health, pending executions
    observations: ObservationState     # recent signals, reflection cadence

    # ── Derived signals (computed from primitives) ──

    trajectory: Trajectory             # direction vector
    momentum: float                    # rate of change (−1.0 to 1.0)
    focus: float                       # execution concentration (0.0 to 1.0)
    risk: RiskProfile                  # aggregated risk indicators
    energy: float                      # estimated execution capacity (0.0 to 1.0)
```

---

## 3. Primitive States

### FinancialState

```
{
    net_worth: float,          # total assets − liabilities
    daily_pnl: float,          # today's change in net worth
    weekly_pnl: float,         # trailing 7 days
    monthly_pnl: float,        # trailing 30 days
    top_movers: [              # top 3 positions by daily P&L
        {asset: str, change: float}
    ],
    last_snapshot: date        # most recent financial snapshot date
}
```

Source: Zhamlik Financial Kernel (`ledger.Engine.snapshot(as_of)`).

### GoalState

```
{
    total: int,                # total goals (all types, non-terminal)
    active: int,               # status = active
    stalled: int,              # no progress for >7 days
    completed: int,            # completed in trailing 30 days
    overdue: int,              # past target date
    by_type: {                 # breakdown by goal_type
        achievable: int,
        continuous: int,
        exploratory: int,
        directional: int,
        meta: int
    },
    top_goal: str | None       # highest-priority active goal title
}
```

Source: `Goal` model in PostgreSQL.

### ExecutionState

```
{
    journal_healthy: bool,     # last integrity check passed
    last_recovery: str | None, # status of last recovery (clean/repaired/corrupt)
    pending_executions: int,   # DISPATCHED without STARTED or COMPLETED
    active_leases: int,        # currently held leases
    wal_size: int              # current WAL size in bytes
}
```

Source: Execution Kernel (`kernel.recover()`, `integrity.verify()`, `lease.registry`).

### ObservationState

```
{
    today_count: int,          # observations recorded today
    week_count: int,           # trailing 7 days
    last_reflection: date,     # most recent reflection
    reflection_cadence: float, # average days between reflections
    topics: [                  # top 3 topic clusters from recent observations
        {topic: str, count: int}
    ]
}
```

Source: Cognitive OS (observation log, reflection records).

---

## 4. Derived Signals

### Trajectory

A direction vector computed from financial trend + goal completion rate + observation momentum.

```
{
    direction: "improving" | "stable" | "declining" | "mixed",
    delta_7d: float,     # aggregate score change over 7 days
    delta_30d: float     # aggregate score change over 30 days
}
```

### Momentum

A float [-1.0, 1.0] representing the aggregate rate of change:
- `> 0.3`: accelerating improvement
- `0.0 to 0.3`: stable
- `< -0.3`: declining

Composite of: net worth trend (40%), goal completion rate (30%), observation cadence (20%), execution health (10%).

### Focus

A float [0.0, 1.0] representing execution concentration:
- `1.0`: all execution on a single goal
- `0.0`: no active execution

Computed from: active goals count, lease distribution, WAL entry rate.

### RiskProfile

```
{
    level: "low" | "medium" | "high" | "critical",
    factors: [                     # contributing risk factors
        {name: str, severity: float}
    ],
    top_risk: str | None           # highest-severity risk
}
```

Risk factors: stalled goals, negative financial trend, broken journal integrity, low reflection cadence, execution congestion.

### Energy

A float [0.0, 1.0] estimating current execution capacity:
- `1.0`: peak readiness
- `0.0`: exhausted

Computed from: recent completion rate (success builds energy), stalled count (blockage drains energy), time since last reflection (no reflection → drift → energy loss).

---

## 5. Invariants

| # | Invariant |
|---|-----------|
| P1 | PersonalModel is a pure function of primitive data at `as_of` |
| P2 | PersonalModel is recomputable: same primitives → same model |
| P3 | PersonalModel does not have its own storage — it is always derived |
| P4 | Trajectory ≠ momentum (direction vs magnitude) |
| P5 | PersonalModel exists independently of any UI rendering it |
| P6 | Primitive states are nullable (if a data source is unavailable, that block is omitted, not zeroed) |

---

## 6. What This Unlocks

| Feature | Derivation |
|---------|-----------|
| **Cockpit** | `render(PersonalModel)` — one screen |
| **Daily Loop** | `read model → set goals → observe → reflect → recompute` |
| **Timeline** | `[PersonalModel(t) for t in trailing_N_days]` |
| **Drift detection** | `model(t) − model(t−1)` — anomaly when deviation > threshold |
| **Epistemic Kernel** | Each primitive field traces to evidence (financial snapshot, goal status, journal entry) — provenance model emerges |
| **Goal prioritization** | Focus + Momentum → which goal to work on |
| **Risk alerts** | RiskProfile.level crosses threshold |
| **Weekly review** | `model(Monday)` vs `model(Sunday)` |

---

## 7. Dependencies

```
Execution Kernel (journal, leases, integrity)
    ↓
Financial Kernel (ledger, snapshots) → ExecutionState + FinancialState
    ↓
Cognitive OS (observations, reflections) → ObservationState
    ↓
PostgreSQL (goals) → GoalState
    ↓
PersonalModel (computed)
```

---

## 8. Non-Goals

- PersonalModel does not store history — history is `[model(t)]` computed on demand
- PersonalModel does not make decisions — it feeds decision-making
- PersonalModel does not have a persistence layer — it is ephemeral, derived from persistent primitives
- PersonalModel does not require a new database — it reads existing sources
