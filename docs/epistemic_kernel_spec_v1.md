# Epistemic Kernel Specification v1

**Version**: 0.1.0 (draft)  
**Date**: 2026-06-06  
**Status**: Model definition — no reference implementation

---

## Layer 1 — Normative Specification

---

### 1. Purpose

Epistemic Kernel determines **what constitutes canonical knowledge of the system**.

Given a stream of observations over time, the kernel defines:
- which observations are part of the evidence base;
- how observations are interpreted into beliefs;
- what belief state follows from a given interpretation;
- whether the belief state is internally consistent and traceable to evidence;
- what must happen for the system to change its mind.

The kernel does not decide *what to believe*. It decides *what follows from evidence under a given interpretation* — and guarantees that the same evidence under the same interpretation produces the same beliefs.

---

### 2. Architectural Boundary

```
Epistemic Kernel
    = observation log
    + interpretation engine
    + belief graph
    + semantic replay
    + provenance boundary
```

| Boundary | Component | Role |
|----------|-----------|------|
| **Provenance boundary** | Kernel | Defines traceability of beliefs to evidence |
| **Storage boundary** | Observation log | Stores raw observations |

Key distinctions:
- **Observation** ≠ **Belief**. Observations are raw. Beliefs are computed.
- **Evidence** ≠ **Interpretation**. Evidence is stable. Interpretation is a versioned function.
- **Traceability** is an invariant. Every belief must be reachable from evidence through a chain of interpretation steps.

---

### 3. Core Artifacts

#### Observation
A raw signal recorded by the system. Carries: source, timestamp, content, confidence. Observations are immutable — once recorded, they cannot be changed, only superseded.

#### Evidence
An observation that has been admitted into the epistemic base. Not all observations become evidence — admission is governed by provenance rules (source trust, recency, relevance).

#### Interpretation
A versioned function that maps evidence to beliefs. Interpretations are parametric — changing parameters produces a different interpretation epoch without altering evidence.

#### Belief
A proposition with an associated degree of support. Beliefs are derived, not stored. The belief state at any point is a function `BeliefState = f(Evidence, InterpretationEpoch)`.

#### Belief Graph
A directed acyclic graph where:
- **Leaf nodes** are evidence entries.
- **Intermediate nodes** are derived beliefs with traceability links.
- **Root nodes** are active beliefs (currently held).
- Edges are labelled with the interpretation step that produced them.

The belief graph is the epistemic equivalent of the journal — it is the source of truth for what the system believes and why.

#### Provenance Chain
A sequence of (evidence, interpretation_step, belief) triples that traces a belief back to its supporting observations. Every belief must have a complete provenance chain.

---

### 4. Invariants

| # | Invariant |
|---|-----------|
| E1 | Every belief is traceable to at least one evidence entry |
| E2 | Beliefs are replayable: same evidence + same interpretation epoch → same belief state |
| E3 | Interpretation is versioned: changing interpretation parameters produces a new epoch, not a mutation of the old one |
| E4 | Semantic recovery is deterministic: `recover(evidence, epoch) = f(evidence, epoch)` — no hidden state |
| E5 | Observations are immutable: once recorded, they cannot be altered |
| E6 | The belief graph is acyclic: belief cannot be its own ancestor |
| E7 | Belief revision is explicit: every change in belief must be attributable to new evidence or a new interpretation epoch |
| E8 | Provenance is monotonic: adding evidence never invalidates a provenance chain (it may extend it) |

---

### 5. Drift Model

Drift is a change in belief state without corresponding change in evidence.

Three types:

| Drift type | What changed | Cause |
|------------|--------------|-------|
| Interpretation drift | Interpretation epoch | Parameters updated, evidence unchanged |
| Evidence drift | Evidence set | New observation added or expired |
| Belief drift | Belief graph structure | Re-derivation produces different topology |

Drift is **detectable and measurable**, not anomalous. The system must be able to report: *"I believe X now, but under epoch E I believed not-X — because interpretation changed."*

#### Detection

| Condition | Signal | Severity |
|-----------|--------|----------|
| `belief(t) ≠ belief(t-1)` AND `evidence(t) = evidence(t-1)` | Interpretation drift | Warning (expected) |
| `belief(t) ≠ belief(t-1)` AND `epoch(t) = epoch(t-1)` | Evidence drift | Info (normal) |
| Provenance chain broken | Structural drift | Error (invariant violation) |
| `recover(evidence, epoch) ≠ previous_belief_state` | Semantic inconsistency | Critical (replay mismatch) |

---

### 6. Re-grounding Model

Re-grounding is the epistemic equivalent of recovery: recompute belief state from evidence under a given interpretation epoch.

```
reground(Evidence, Epoch) → (BeliefState, ProvenanceGraph, DriftReport)
```

Properties:

| Property | Meaning |
|----------|---------|
| Determinism | Same evidence + same epoch → same belief state |
| Idempotence | `reground ∘ reground = reground` |
| Traceability | Every output belief has a complete provenance chain |
| Drift reporting | If belief state differs from previous at same epoch, drift is reported |

Re-grounding is triggered when:
- New evidence arrives.
- Interpretation epoch changes.
- Periodic consistency check detects potential drift.
- After recovery from an execution kernel crash (epistemic state may need re-sync).

---

### 7. Consistency Model

```
Consistency Domain = Provenance Domain
```

A belief is consistent if:
1. It is derivable from evidence via the current interpretation epoch.
2. Its provenance chain is complete and acyclic.
3. Replaying from evidence produces the same belief.

Outside the provenance domain, beliefs are speculative and not covered by this specification.

#### Provenance Boundary

| Boundary | Component | Role |
|----------|-----------|------|
| **Provenance boundary** | Kernel | Defines traceable beliefs |
| **Storage boundary** | Observation log | Stores raw signals |

The observation log is a storage mechanism. The kernel is the traceability mechanism. A belief without provenance is outside the correctness domain — analogous to a journal entry without a hash chain link.

---

### 8. Non-Goals

This specification explicitly does not define:
- What constitutes a "correct" belief (truth is outside the model)
- Source trust weighting (epistemic policy, not kernel)
- Temporal decay of evidence (retention policy, not kernel)
- Multi-valued or fuzzy logics (extensible via interpretation parameters)
- Hierarchical belief composition (application-layer structure)
- Emotional or affective influence on beliefs (cognitive layer, not epistemic)

---

### 9. Symmetry with Execution Kernel

| Layer | Execution | Epistemic |
|-------|-----------|-----------|
| Input | Command | Observation |
| Record | Journal Entry | Evidence |
| State | Execution State | Belief State |
| Replay | Deterministic replay | Semantic replay |
| Recovery | Journal recovery | Re-grounding |
| Repair | ABANDONED / LEASE_EXPIRED | Drift detection / re-derivation |
| Integrity report | Hash chain violations | Provenance chain breaks |
| Boundary | Consistency boundary | Provenance boundary |
| Scope | Single-writer WAL domain | Single-interpretation provenance domain |
| Central question | What sequence of events is canonical history? | Why does the system believe what it believes? |

---

## Layer 2 — Reference Validation

*No reference implementation exists at this version. Validation is pending.*

---

### Required Validation (future)

| E# | Domain | Method | Required |
|----|--------|--------|----------|
| E1 | Belief traceability | Every belief must have provenance path to evidence | Mandatory |
| E2 | Replay determinism | Same evidence + epoch → same beliefs | Mandatory |
| E3 | Interpretation versioning | New epoch does not mutate old beliefs | Mandatory |
| E4 | Semantic recovery | `reground(evidence, epoch)` is deterministic and idempotent | Mandatory |
| E5 | Observation immutability | Recorded observations cannot be altered | Mandatory |
| E6 | Acyclicity | Belief graph contains no cycles | Mandatory |
| E7 | Explicit revision | Every belief change has attributable cause | Mandatory |
| E8 | Provenance monotonicity | Adding evidence never breaks existing chains | Desirable |
| E9 | Drift detection | Interpretation drift detectable without evidence change | Mandatory |
| E10 | Cross-instance provenance | Two independent epistemic kernels with same evidence + epoch converge | Open |
