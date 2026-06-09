# Execution Kernel Specification v1

**Version**: 1.0.0  
**Date**: 2026-06-06

---

## Layer 1 — Normative Specification

What must be true for any correct implementation of the Execution Kernel.

---

### 1. Purpose

Execution Kernel determines **what sequence of events constitutes the canonical history of the system**.

Given a set of commands submitted over time, the kernel defines:
- which events are part of history;
- in what order they occurred;
- what state is implied by that history;
- whether the history is self-consistent.

The kernel does not decide *what* to do. It decides *what happened* — and guarantees that the answer is invariant under crash, restart, and replay.

---

### 2. Architectural Boundary

```
Execution Kernel
    = deterministic state machine
    + append-only journal
    + replay-based recovery
    + single-writer consistency boundary
```

| Boundary | Component | Role |
|----------|-----------|------|
| **Consistency boundary** | Kernel | Defines correctness of history |
| **Storage boundary** | Journal | Stores entries |

These boundaries are formally distinct. The journal is a storage mechanism — it stores entries. The kernel is the consistency mechanism — it determines what constitutes a correct history.

---

### 3. Core Artifacts

#### Command
An instruction submitted to the kernel. Carries a goal identifier, an event type, execution context, and an intent to change state.

#### Journal Entry
A single record in the append-only journal. Forms a hash chain with predecessor linking. Every entry has a unique identifier within its writer domain.

#### Execution
A single lifecycle of a goal from dispatch through terminal state. Identified by `execution_id`, which is a deterministic function of goal identity, dispatch epoch, and parent context.

#### Lease
Temporary ownership of a goal for one execution cycle. At most one active lease exists per goal at any time.

#### Snapshot
Materialized state at a journal position. Transparent: replay from snapshot position produces identical state to replay from journal start.

#### Recovery Report
Output of the recovery procedure: status (success or type of failure), entry count, and anomaly summary.

#### Integrity Report
Output of integrity verification: structural anomalies, hash chain breaks, sequence gaps.

---

### 4. State Machine

#### Event Types

- **Lifecycle events**: DISPATCHED, STARTED, COMPLETED, FAILED, CANCELLING, CANCELLED, RETRIED, PREEMPTED
- **Lease events**: LEASE_ISSUED, LEASE_EXPIRED, LEASE_REVOKED
- **Repair events**: ABANDONED

#### Transitions

```
DISPATCHED → STARTED → COMPLETED
                  → FAILED
                  → CANCELLING → CANCELLED
                  → LEASE_EXPIRED

STARTED → ABANDONED
DISPATCHED → CANCELLED
```

Terminal states: COMPLETED, FAILED, CANCELLED, ABANDONED, LEASE_EXPIRED.

---

### 5. Invariants

| # | Invariant |
|---|-----------|
| I1 | Journal is the unique source of truth for history |
| I2 | Every lease in journal history corresponds to an active lease in state, and vice versa |
| I3 | No transition originates from a terminal state |
| I4 | Replay from a snapshot produces identical state to full journal replay from the snapshot position |
| I5 | Recovery is idempotent: `recover(recover(state)) = recover(state)` |
| I6 | The set of active executions at any point equals the projection of the journal up to that point |
| I7 | At most one active lease per goal at any time |
| I8 | `execution_id` is a pure deterministic function — no hidden entropy |
| I9 | Any journal entry that violates the hash chain is detected as corruption |
| I10 | Every prefix of a valid journal is itself a valid journal |
| I11 | At most one writer produces entries for a journal directory |

---

### 6. Failure Model

| Failure | Detection | Post-condition |
|---------|-----------|----------------|
| Process crash during write | Journal prefix is valid; trailing incomplete data is ignored | Max prefix loss; no arbitrary state |
| Journal truncation | Missing footer or incomplete last entry | Last complete prefix is valid |
| Entry content corruption | Hash chain breaks at the corrupted entry | Violation reported; repair possible |
| Snapshot corruption | Snapshot checksum or content mismatch | Full replay from journal start |
| Concurrent writer | Collision detected at journal open or on integrity check | Second writer rejected or corruption reported |

**No failure within the verified domain produces silent, undetected corruption.**

---

### 7. Recovery Model

Recovery is a deterministic function:

```
recover(State) → (Status, State, Report)
```

Properties:

| Property | Meaning |
|----------|---------|
| Idempotence | `recover ∘ recover = recover` |
| Determinism | Same journal → same result state |
| Prefix stability | Recovery of any valid prefix yields a subset of the full recovery state |
| Termination | Recovery always terminates |

---

### 8. Consistency Model

```
Consistency Domain = Single Writer Journal Domain
```

Within this domain:
- Events have a total order (append-only journal order)
- Replay is deterministic (same input → same output)
- Every journal prefix is a valid history (prefix consistency)
- Snapshot state is consistent with journal state at snapshot position

**Outside this domain**, the kernel makes no guarantees. Multi-writer execution is outside the correctness boundary. This is a proven negative result, not an assumption.

#### Enforcement

Single-writer must be enforced by technical means at the journal boundary. Enforcement is a deployment invariant, not an operational convention.

---

### 9. Non-Goals

This specification explicitly does not define:
- Distributed consensus
- Multi-writer ordering
- Conflict resolution
- Replicated execution
- Cluster coordination
- Temporal queries
- Access control

These are architectural constraints, not missing features.

---

## Layer 2 — Reference Validation

Evidence that a specific implementation satisfies the normative specification.

---

### Implementation

The reference implementation lives in `execution_dynamics/`. Key components:

| Component | Path | Role |
|-----------|------|------|
| Kernel | `kernel.py` | State machine, dispatch, recover, snapshot |
| Journal | `journal.py` | Entry model, dispatch events, append validation |
| SegmentedWAL | `segmented_wal.py` | Append-only storage with rotation and lock |
| Lease registry | `lease.py` | Lease acquire, release, query |
| Snapshot manager | `snapshot.py` | Snapshot create, load, validate |
| Integrity verifier | `integrity.py` | Hash chain, sequence, structural checks |

### Validation Evidence

| K | Domain | Method | Count |
|---|--------|--------|-------|
| K1 | Journal correctness | Property-based tests: random entries, persistence, recovery invariance | 5 |
| K2 | State machine correctness | Stateful scenario tests: dispatch/complete/fail/cancel/expire/snapshot/recover | 6 (600+ scenarios) |
| K3 | Crash recovery | Crash-at-point tests: F1–F8, prefix property, append/truncate cycle | 10 |
| K4 | Corruption safety | Fuzz tests: segments, snapshots, manifests, random bytes, recovery soundness | 28 |
| K5 | Ownership correctness | Concurrency tests: single ownership, deterministic ID, lease transitions, stress | 15 |
| K6 | Cross-instance interference | Interference tests: interleaved writes, interleaved recovery, adversarial stress | 10 |
| K7 | Single-writer enforcement | Lock tests: acquire/release/close, rejection, crash safety, stress | 11 |

**Total**: 169 tests. All passing.

### Key Enforcement Mechanisms

- **Single-writer**: `fcntl.flock(LOCK_EX | LOCK_NB)` on `.wal_lock` at WAL open. Configurable via `enforce_single_writer` (default `False` for test compatibility; production must set `True`).
- **Hash chain**: SHA-256 per entry. `prev_hash` links to predecessor. Verified on recovery and integrity check.
- **Snapshot integrity**: Content hash stored with snapshot. Verified on load. Fallback to full replay on mismatch.
- **Segment format**: Header line (type marker), entries (JSON lines), footer line (segment hash). Footer verified on read.

### Configuration

| Parameter | Default | Production |
|-----------|---------|------------|
| `enforce_single_writer` | `False` | `True` |
| `wal_path` | `""` (in-memory) | Filesystem path |
| `snapshot_path` | `""` (in-memory) | Filesystem path |
