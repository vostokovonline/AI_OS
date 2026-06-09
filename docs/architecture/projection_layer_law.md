# AI-OS Architectural Laws v0.5

## What is proven

In four independent domains, the same structure is observed:

```
Journal
   ↓
Projection
   ↓
State
```

And in every case, the invariant holds:

```
State = Projection(Journal)
```

| Layer | Journal | Projection | Evidence |
|-------|---------|------------|----------|
| Execution Kernel | WAL | `recover()` → KernelState | Snapshot-compact-replay cycle. 186 tests. |
| Runtime (R0.1) | Runtime Journal | `rebuild_from_journal()` → GraphState | Crash → WAL → recovery → identical graph. |
| World Model (refactored) | WorldJournal | `rebuild_world()` → WorldState | `delete(state); rebuild(journal)` → identity. |
| Financial (Zhamlik) | Ledger | projection → Balance Sheet | 500 years of accounting practice. |

### Definition

**Projection Layer** — any layer where:
- The journal is the sole source of truth
- State is a materialized projection of the journal
- Deleting state and rebuilding from the journal produces identical results

### Corollary

If a layer's state cannot be destroyed and rebuilt identically from its journal alone, that layer contains **hidden state** — a violation of the invariant.

---

## What is unknown

In every case, the journal has an origin:

```
   ?
   ↓
Journal
   ↓
Projection
   ↓
State
```

The nature of `?` varies:

| Layer | Origin of Journal |
|-------|-------------------|
| Kernel | Intent → Journal |
| Runtime | Command → Journal |
| Financial | Economic activity → Ledger |
| World Model | Observations → WorldJournal |

The step that produces journals from their precursors is **not uniform across layers**, **not proven to follow any single law**, and in most cases **not journaled itself**.

The origin of journals is an open question. No mechanism has been demonstrated that works across multiple domains the way the Projection Law does.

---

## Current status

| Claim | Evidence | Status |
|-------|----------|--------|
| `State = Projection(Journal)` | 4 independent layers | **Confirmed — architectural invariant** |
| `? → Journal` | 0 independent confirmations of a single mechanism | **Open question** |

---

## Architectural Test

For any proposed layer, ask:

> If I delete all process memory and leave only the journal, can I reconstruct identical state?

| Answer | Meaning |
|--------|---------|
| Yes | Layer conforms to the Projection Law |
| No | Hidden state exists — architectural debt |

---

## Implications

1. **Memory is not storage.** State is a cache of a projection, not a source of truth.
2. **Crash recovery is free.** Replay the journal, reconstruct state. No separate persistence mechanism per layer.
3. **Hidden state is detectable.** The architectural test reveals it.
4. **The frontier is the origin of journals.** The Projection Law is confirmed. How journals themselves emerge is what comes next — but no claim about that mechanism is supported by evidence yet.

---

## Observed: Event vs Interpretation separation (World Policy v0)

World Policy v0 introduced a distinction not present in earlier layers:

| Before | After |
|--------|-------|
| `Event` only | `Event` + `Interpretation` |
| Interpretation existed only in caller code | Interpretation is journaled as `_interpretation` |
| No way to ask "why was this an event?" | `classification`, `confidence`, `pattern_id`, `policy_version` recorded per entry |
| Policy version invisible | Every event carries `policy_version` |

This creates a new observable dimension:

```
Event Truth ≠ Interpretation Truth
```

The same event can be interpreted differently by different policy versions without changing the event itself. This parallels the accounting distinction:

```
Transaction (Zhamlik Ledger) ≠ Financial Statement (Projection)
Event (WorldJournal)         ≠ Interpretation (WorldPolicy)
```

This is **not** a law — it is an observed structural separation made possible by journaling interpretation metadata alongside events. Whether it becomes a foundation for future layers (InterpretationJournal, Epistemic Layer) is an open question.
