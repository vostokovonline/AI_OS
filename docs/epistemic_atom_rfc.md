# Epistemic Atom RFC

**Status**: Question open — no final answer  
**Date**: 2026-06-06

---

## 1. The Question

Execution Kernel is built around one irreducible fact:

> **A state transition occurred.**

JournalEntry records exactly this — nothing more, nothing less. Everything else (State, Snapshot, Projection, Recovery) is derived from replaying these irreducible facts.

The equivalent question for Epistemic Kernel:

> **What is the irreducible epistemic fact?**

What fact about knowledge cannot be decomposed into more primitive facts?

---

## 2. What We Know

### The atom must be:

| Property | Why |
|----------|-----|
| **Atomic** | Cannot be decomposed into smaller epistemic facts |
| **Immutable** | Once recorded, the fact does not change |
| **Orderable** | The sequence of facts is meaningful |
| **Replayable** | Replaying atoms reconstructs epistemic state |
| **Verifiable** | Integrity can be checked independently |

### The atom must NOT assume:

- A model of belief (belief is a derived state, like ExecutionState)
- A model of interpretation (interpretation is a parameter, not a record)
- A model of truth (truth is outside the kernel scope)

---

## 3. Candidates

### A. Observation

```
OBSERVED(proposition, source, confidence)
```

| Check | Verdict |
|-------|---------|
| Irreducible? | ✅ Yes — raw signal, cannot decompose further |
| Assumes belief? | ✅ No — just records a signal |
| Assumes interpretation? | ✅ No — raw, uninterpreted |
| Replayable to belief state? | ❌ No — observations alone don't produce beliefs; interpretation is needed |

**Problem**: Replaying observations does not reconstruct belief state. The interpretation step is missing. Observation is input to the kernel, not an event within it.

---

### B. Evidence

```
SUPPORTS(proposition, evidence_id)
CONTRADICTS(proposition, evidence_id)
```

| Check | Verdict |
|-------|---------|
| Irreducible? | ❌ No — requires both a proposition and an observation already in the system |
| Assumes belief? | ⚠️ Partial — already classifies a relationship to a claim |
| Assumes interpretation? | ✅ No — classification could be pre-interpretation |
| Replayable to belief state? | ⚠️ Partial — evidence alone doesn't determine degree of belief |

**Problem**: Evidence is a relationship, not a primitive. Requires at least two prior entities (observation, proposition).

---

### C. Belief

```
BELIEVE(proposition, support, epoch)
```

| Check | Verdict |
|-------|---------|
| Irreducible? | ❌ No — belief is a *state*, not an event |
| Assumes belief? | ❌ No — it IS belief (begs the question) |
| Assumes interpretation? | ❌ Yes — support is a function of interpretation |
| Replayable to belief state? | ✅ Yes — replaying beliefs reconstructs belief state |

**Problem**: Belief is the state we want to reconstruct, not the primitive we replay. It's the equivalent of treating ExecutionState as the atom instead of JournalEntry.

---

### D. BeliefEvent

```
CREATED(proposition, support, justification, epoch)
UPDATED(proposition, new_support, justification, epoch)
RETRACTED(proposition, epoch)
```

| Check | Verdict |
|-------|---------|
| Irreducible? | ⚠️ Partial — already assumes proposition, justification, epoch exist |
| Assumes belief? | ⚠️ Partial — describes belief transitions, which is closer than belief itself |
| Assumes interpretation? | ❌ Yes — `epoch` and `justification` reference an interpretation framework |
| Replayable to belief state? | ✅ Yes — replaying BeliefEvents in order reconstructs belief state |

**Problem**: Closest viable candidate so far, but still assumes interpretation exists. The `epoch` field embeds an interpretation context into every event. This may be the right level — or may be one level too high.

---

### E. ClaimEvent

```
ASSERTED(proposition)
RETRACTED(proposition)
SUPPORTED(proposition, evidence_id)
CONTRADICTED(proposition, evidence_id)
```

No belief degree. No interpretation epoch. Just: a claim enters or leaves the space of assertions.

| Check | Verdict |
|-------|---------|
| Irreducible? | ✅ Yes — pure assertion, no derived quantities |
| Assumes belief? | ⚠️ Partial — ASSERTED ≠ belief; it's a weaker claim (someone asserted it) |
| Assumes interpretation? | ✅ No — assertion is pre-interpretation |
| Replayable to belief state? | ✅ Yes — belief state is a function over the set of assertions |

**Key insight**: ASSERTED(X) is a fact about the system (someone/something asserted X). It does not require belief, interpretation, or truth. It simply records: *at time T, proposition X was asserted in the system.*

**State becomes derived**:

```
AssertionSet = current set of ASSERTED propositions
  → filtered by SUPPORTED/CONTRADICTED relations
    → weighted by evidence accumulation
      → BeliefState
```

This mirrors Execution:

```
Journal = sequence of DISPATCHED/STARTED/COMPLETED...
  → replayed in order
    → ExecutionState
```

---

### F. InterpretationEvent

```
INTERPRETED(proposition, support, context_epoch)
```

The most radical candidate. The atom is not about what is believed — it's about what act of interpretation occurred.

| Check | Verdict |
|-------|---------|
| Irreducible? | ✅ Yes — interpretation is the minimal epistemic act |
| Assumes belief? | ✅ No — interpretation precedes belief |
| Assumes interpretation? | ⚠️ Self-referential — the event IS interpretation |
| Replayable to belief state? | ✅ Yes — replaying interpretations = belief state |

**Key claim**: Meaning does not exist before interpretation. Observations are not epistemic events — they become epistemic only when interpreted. Therefore the irreducible epistemic event is the act of interpretation itself.

```
Observation
    → InterpretationEvent (the atom)
        → Claim (derived)
            → BeliefState (derived)
```

**Problem**: If interpretation is the atom, what is its internal structure? An interpretation event that does not reference a proposition or observation is empty. But if it references a proposition, it assumes the proposition exists — which means proposition is the real atom.

---

## 4. Open Tensions

### Tension 1: Proposition vs Interpretation

Is a proposition a primitive or a derived entity?

- If proposition is primitive → the atom is about propositions (ASSERTED, RETRACTED).
- If interpretation is primitive → the atom is about acts of meaning-making (INTERPRETED).
- If both are primitive → the atom is a relationship between them (BeliefEvent, ClaimEvent).

### Tension 2: Belief as state vs belief as event

- If belief is state → the atom must be a transition that changes belief (BeliefEvent).
- If belief is derived → the atom can be more primitive (ClaimEvent, Observation).

### Tension 3: Epoch as parameter vs epoch as structure

- If epoch is a parameter of replay → atom does not need to carry epoch.
- If epoch changes the meaning of the atom itself → atom must carry epoch.

Execution Kernel does not have this question — JournalEntry has no "epoch" parameter. Commands don't change meaning over time. But beliefs do change meaning when the interpretation framework changes.

### Tension 4: What is the replay target?

- Execution replay target: `State = replay(JournalEntries)` → well-defined.
- Epistemic replay target: `BeliefState = replay(?)` → depends on what the atom is.
  - If atom = ClaimEvent → `BeliefState = f(ClaimSet, InterpretationEpoch)`.
  - If atom = BeliefEvent → `BeliefState = replay(BeliefEvents)`.
  - If atom = InterpretationEvent → `BeliefState = replay(InterpretationEvents)`.

---

## 5. Provisional Observations

1. **ClaimEvent (ASSERTED/RETRACTED/SUPPORTED/CONTRADICTED)** is the most irreducible — it records only that something was asserted, without assuming belief or interpretation. It is the epistemic equivalent of "a state transition occurred."

2. **BeliefEvent** is the most directly usable — it mirrors JournalEntry structure and enables immediate replay semantics. But it may be one level of abstraction too high (it assumes interpretation exists).

3. **InterpretationEvent** is the most philosophically honest — meaning is created by interpretation, not by raw observation. But it creates a self-referential structure that is difficult to formalize.

4. The Execution Kernel analogy may be misleading if pushed too far. Epistemic systems have a fundamentally different structure: they involve interpretation, which has no analogue in command execution.

---

## 6. Next Question

The atom question is not resolved. Four candidates remain viable.

The next step is not to pick one, but to test each candidate against a concrete scenario:

> **Given the same sequence of observations, two different interpretation epochs should produce two different belief states. Which atom makes this distinction formally expressible?**

This question — **epoch differentiation** — may be the key to selecting the right atom. If the atom cannot express "same evidence, different interpretation, different belief," it is not the right atom.
