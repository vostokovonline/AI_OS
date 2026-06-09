# Session Summary — Epistemic Recovery + Causal Bridge + Integrity Verification

## Goal
Build a fully verifiable deterministic causal cognitive runtime with closed-loop execution→epistemic feedback, crash recovery, adversarial fault tolerance, and journal integrity verification.

## Constraints & Preferences
- ALL execution must enter through `KernelIngress.dispatch()` → signed `KernelCapability` → `kernel._dispatch_with_capability()` → lease → journal → executor.
- `KernelCapability` is HMAC-SHA256 signed, scoped, lease-bound, zone-provenanced, epoch-bound.
- `KernelIngress` is the ONLY public entry point. `ExecutionKernel.dispatch()` rejects calls without a valid signed capability.
- Snapshot is acceleration layer, NOT source of truth (WAL remains authoritative).
- Epistemic kernel mirrors execution kernel: append-only journal, deterministic replay, invariant-validated transitions, provenance chain, crash-`recover()`.
- Causal bridge adds formal `CausalityEdge` between execution and epistemic, dual propagation, unified consistency verification.
- **Bridge → GoalExecutor wiring is now live** — every completed/failed goal automatically feeds back to epistemic state (beliefs, motifs, policy adjustments).
- **Factory singletons use `RLock`** (not `Lock`) — nested dependency `get_causality_bridge()` → `get_epistemic_kernel()` previously self-deadlocked.
- **Journal carries cryptographic hash chain** — `entry_hash` + `prev_hash` computed at append time for tamper detection.
- **`IntegrityVerifier`** is a pure function — verifies any journal (in-memory, WAL-replayed, exported) without mutation.
- **Integrity verification must pass before WAL persistence** — no point persisting corrupted journals.
- PHE is experimental until core stack passes all integrity + recovery + determinism gates.
- **`IntegrityVerifier.verify_lifecycle()` now checks valid initial events** — `DISPATCHED` or `RECOVERED` must be the first event per execution_id.

## Progress
### Done
- **Execution kernel foundation**: `DispatchJournal`, `LeaseRegistry`, `TruthJournal`, coordination dynamics, WAL, snapshot v2, capability-gated ingress, 15 invariants, all external callers migrated to `dispatch_goal()`.
- **P2: Epistemic Recovery** — `export_state()` / `restore_state()` / `recover()` on `EpistemicKernel`. 15 recovery tests pass.
- **P2.5: Adversarial Recovery** — 18 adversarial tests pass (duplicate, missing, corrupted, out-of-order, truncated).
- **`update_attractor()` now journals** — fixed silent replay gap.
- **P1a: Bridge → GoalExecutor wiring** — success/failure auto-feedback to epistemic state via `goal_executor_v2.py`.
- **P1b: Journal-native causal linkage** — deterministic `execution_id` through dispatch → executor → bridge.
- **P2.6: Journal Integrity Verification** — `IntegrityVerifier` with 4 checks (hash chain, sequence, causal links, lifecycle + valid initial events). 22 tests.
- **P2.7: Execution WAL persistence** — `JsonLinesWAL`:
  - JSON Lines format, fsync per write
  - Crash recovery: truncate at first invalid line (JSON or binary garbage)
  - Sequence, hash chain (`prev_hash`/`entry_hash` at WAL level)
  - LSN-based ordering and replay
  - Integrates with existing `DispatchJournal` via same protocol (`wal.append()`, `wal.replay()`)
  - Full end-to-end: `DispatchJournal → JsonLinesWAL → crash → recover → consistent state`
  - 21 tests covering append, replay, fsync durability, crash recovery, sequence, hash chain, LSN, stats, integration

### In Progress
- None. Core integrity + recovery + determinism gates are green.

### Blocked
- `api/endpoints/__init__.py` auto-imports `goals`, `artifacts`, `skills`, `llm`, `graph` — crash without `DATABASE_URL`. Not a code issue, only affects isolated testing outside container.

## Key Decisions
- **Bridge is source of truth for causal coupling**, not Factory. Factory is convenience only — creates (ek → bridge → cpe → phe). Bridge owns the causal graph.
- **PHE is experimental until core stack passes all determinism + recovery + integrity gates**. Not yet verified against real execution traces.
- **`IntegrityVerifier` is a standalone class**, not scattered methods on `DispatchJournal`. Can verify any journal (in-memory, WAL-replayed, exported, from other instances).
- **Integrity verification before WAL persistence** — no point persisting a broken chain. P2.6 → WAL → Root of Trust → Fault Injection → Storage Model.
- **Journal hash chain is append-time**, not verification-time — hash is computed during `append()` and stored in the entry. On verify, the hash is recomputed and compared.
- **Architecture dependency order**: Factory → Kernel → Bridge → Policy. No reverse imports. `RLock` enforces reentrancy safety but the dependency direction is now documented.
- **Layer order for production boot**: WAL → Integrity Verification → Deterministic Replay → Epistemic Recovery → Bridge → Policy.
- **Valid initial events**: only `DISPATCHED` and `RECOVERED` can be the first event in a lifecycle.
- **Storage model (Global WAL vs Split WALs) is NOT yet decided**. Single WAL currently provides stronger consistency guarantees (`Replay(WAL) → State` atomically). Split WALs may become correct after implementing transactional writes across execution + bridge + epistemic, but that is future work. Decision deferred until after P2.7, Root of Trust, and Fault Injection prove correctness of the persistence layer.

## Why Split WALs Are Not Yet Locked

| Property | Single WAL | Split WALs |
|----------|-----------|------------|
| Total history | One `Replay(WAL) → State` | Requires cross-WAL sync |
| Transactional writes | Inherent | Requires 3-phase commit or similar |
| Recovery from partial write | `fsync` at known position | Edge can be orphaned |
| Replay complexity | O(N) | O(N_exec) + O(N_epi) + O(N_bridge) |
| Correctness proof | Trivial | Requires proving cross-WAL consistency |
| Future scalability | May bottleneck on epistemic volume | Epistemic WAL can use different storage |

Decision deferred until after Fault Injection proves persistence layer correctness and the actual event volume ratio is measurable.

## Next Steps (Priority Order)
1. **P2.7: Execution WAL persistence** — append-only file format, fsync semantics, crash recovery. Single WAL for now.
2. **Root of Trust** — State Hash → Merkle Root over journal. Enables proof-level verification.
3. **P3: Fault Injection & WAL Corruption Suite** — intentional journal corruption with full validation chain:
   - Bit/payload corruption, partial writes, fsync truncation, double writes
   - WAL truncation at arbitrary points, verify prefix recovery + consistency
   - Replay divergence: same corrupted journal → `state_hash_a == state_hash_b`
   - Validate: `WAL → IntegrityVerifier → Replay → Recover` end-to-end
4. **Storage Model Design** — compare Global WAL vs Split WALs (Execution + Epistemic + Bridge) by:
   - Replay complexity
   - Recovery complexity
   - Consistency guarantees under faults
   - Storage growth projections
   - Tolerance to partial writes
5. **P2.8: Snapshotting** — only after storage model is decided (snapshot format depends on WAL structure).
6. **P2.9: Fast recovery** — load snapshot → verify integrity → replay WAL tail → epistemic recover.
7. **PHE validation on real trajectories** — after persistence + fault injection + snapshots.

## Critical Context
- **Determinism gate: 73/73 tests pass** — 18 replay determinism + 15 recovery + 18 adversarial + 22 integrity.
- **P1a closed the loop**: `goal_executor_v2.py` (success and failure paths) → `bridge.on_execution_completed/on_execution_failed()` → `CausalityEdge` → epistemic observation + belief update.
- **P1b ensured causal traceability**: bridge `entry_id` is now the deterministic `execution_id` from the kernel journal, not random `uuid4()`. Full chain: `JournalEntry.entry_id` → `CausalityEdge.execution_entry_id` → epistemology event.
- **Factory deadlock was real**: `get_causality_bridge()` → `get_epistemic_kernel()` under same `threading.Lock` → self-deadlock on first call. `RLock()` fixes it trivially.
- **Hash chain format**: `SHA256(prev_hash | entry_id | execution_id | event | json.dumps(payload, sort_keys=True))`. Canonical payload excludes hash fields.
- **Valid lifecycle transitions** defined in `integrity.py:_VALID_TRANSITIONS`. 12 event types, 2 valid initial events.
- PHE (6 files: tree, depth, uncertainty, scoring, search, facade) exists but is unverified against real execution traces.
- `execution_dynamics/integrity.py` is fully self-contained — imports only `re`, `dataclasses`, `typing`, and `DispatchJournal`/`JournalEntry`.

## Relevant Files
- `execution_dynamics/kernel.py` — passes `ctx.execution_id` to `GoalExecutorV2(_execution_id=...)`.
- `execution_dynamics/journal.py` — `JournalEntry` with hash chain fields.
- `execution_dynamics/integrity.py` — `IntegrityVerifier` (4 checks), `IntegrityReport`, `IntegrityError`.
- `epistemic_kernel/__init__.py` — `export_state()`, `restore_state()`, `recover()`, `update_attractor()` journals.
- `epistemic_factory.py` — `Lock()` → `RLock()` fix.
- `goal_executor_v2.py` — accepts `_execution_id`, calls bridge on success/failure.
- `causal_bridge/bridge.py` — `on_execution_completed()`, `on_execution_failed()`, synchronous with `DualPropagator`.
- `tests/unit/test_epistemic_recovery.py` — 15 tests.
- `tests/unit/test_epistemic_recovery_adversarial.py` — 18 tests.
- `tests/unit/test_deterministic_replay.py` — 18 tests.
- `tests/unit/test_integrity_verifier.py` — 22 tests (hash chain, sequence, causal links, lifecycle, report).
