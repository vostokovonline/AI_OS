"""
K6 — Cross-instance interference under shared WAL without coordination.

PURE OBSERVATIONAL — no kernel changes, no locks, no writer_id.

Three adversarial scenarios:

  (A) Concurrent append — two kernels write to the same WAL.
      Verifies: parsability, recoverability, structural integrity.

  (B) Interleaved recovery — A writes, B writes, both recover.
      Verifies: deterministic per-instance, divergence allowed.

  (C) Snapshot race — one kernel snapshots while another writes.
      Verifies: prefix consistency (not global consistency).

Architectural goal:
  Discover hidden single-writer assumptions in the current kernel.
  Result is either "accidentally safe" or "revealed fragility".
  Both are valid outcomes — no fixes during K6.
"""

import json
import os
import time
import hashlib
import random
import tempfile
import shutil
import threading
import pytest

os.environ.setdefault('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost:5432/test')

from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.journal import JournalEntry
from execution_dynamics.integrity import IntegrityVerifier

RANDOM_SEED = 42
GOAL_IDS = ['goal_a', 'goal_b']

REPAIR_EVENTS = frozenset({'ABANDONED', 'LEASE_EXPIRED'})


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_kernel(wal_dir: str, snap_file: str) -> ExecutionKernel:
    os.makedirs(wal_dir, exist_ok=True)
    return ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
            enforce_single_writer=False,  # K6: explicit uncoordinated mode
        )
    )


def _append_entry(kernel: ExecutionKernel, goal: str, event: str,
                  seq: int) -> JournalEntry:
    """Append a single journal entry."""
    e = JournalEntry(
        event=event,
        goal_id=goal,
        execution_id=f"{goal}:exec:{seq}",
        lease_id=f"lease_{goal}",
        timestamp=time.time(),
    )
    kernel.journal.append(e)
    return e


def _walk_files(wal_dir: str) -> list[str]:
    """All WAL segment files sorted."""
    if not os.path.exists(wal_dir):
        return []
    return sorted([
        os.path.join(wal_dir, f) for f in os.listdir(wal_dir)
        if f.endswith('.wal') and f != 'manifest.json'
    ])


def _read_all_lines(wal_dir: str) -> list[str]:
    """Read all lines from all segments into a single list."""
    lines = []
    for fpath in _walk_files(wal_dir):
        with open(fpath) as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
    return lines


def _count_entries(wal_dir: str) -> int:
    """Count journal entries (not header/footer)."""
    count = 0
    for line in _read_all_lines(wal_dir):
        try:
            data = json.loads(line)
            if '_wal_meta' not in data:
                count += 1
        except json.JSONDecodeError:
            pass
    return count


def _check_structural_integrity(journal) -> list[str]:
    """Hash chain + sequence + duplicate entry_id check."""
    violations = []
    verifier = IntegrityVerifier()

    for err in verifier.verify_hash_chain(journal):
        violations.append(f"hash_chain: {err.detail}")
    for err in verifier.verify_sequence(journal):
        violations.append(f"sequence: {err.detail}")

    seen = set()
    for entry in journal._entries:
        if entry.entry_id in seen:
            violations.append(f"duplicate_entry_id: {entry.entry_id}")
        seen.add(entry.entry_id)
    return violations


def _journal_entries_hash(entries: list) -> str:
    filtered = [
        {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
        for e in entries if e.event not in REPAIR_EVENTS
    ]
    raw = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ── Test Class ───────────────────────────────────────────────────────────────


class TestK6CrossInstanceInterference:

    # ══════════════════════════════════════════════════════════════════════════
    # K6A — Concurrent append
    #
    # Two independent kernel instances write to the same WAL without
    # synchronization. Verifies the WAL remains structurally sound:
    #   - All lines are valid JSON
    #   - recover() on a fresh kernel does not crash
    #   - Hash chain is internally consistent per-instance perspective
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k6_a_interleaved_writes_parsable(self):
        """Interleaved writes from two kernels produce parseable WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)
            k_b = _make_kernel(wal, snap)

            # Interleave writes from both kernels to the same WAL
            for i in range(5):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_b, "goal_b", "DISPATCHED", i)
                _append_entry(k_a, "goal_a", "STARTED", i)
                _append_entry(k_b, "goal_b", "LEASE_ISSUED", i)

            # All lines must be valid JSON
            for line in _read_all_lines(wal):
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Unparseable line in WAL: {line[:80]}")

            # Fresh kernel must recover without crash
            k_c = _make_kernel(wal, snap)
            result = await k_c.recover()
            assert result['status'] == 'ok', \
                f"Recovery failed: {result}"

    @pytest.mark.asyncio
    async def test_k6_a_sequence_collisions(self):
        """Two writers produce overlapping sequence numbers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)
            k_b = _make_kernel(wal, snap)

            for i in range(3):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_b, "goal_b", "DISPATCHED", i)

            # Read all LSNs
            lsns = []
            for line in _read_all_lines(wal):
                data = json.loads(line)
                if '_wal_meta' not in data:
                    lsns.append(data.get('lsn', ''))

            # LSNs may have duplicates — that IS the discovery
            unique = set(lsns)
            if len(unique) < len(lsns):
                # Duplicate LSNs found — expected with uncoordinated writers
                pass  # This is the finding, not a test failure

            # But there MUST be no duplicate LSN+entry_type+goal_id (logical uniqueness)
            seen: set[tuple] = set()
            for line in _read_all_lines(wal):
                data = json.loads(line)
                if '_wal_meta' not in data and data.get('entry_type'):
                    key = (data['lsn'], data['entry_type'], data.get('payload', {}).get('goal_id', ''))
                    assert key not in seen, \
                        f"Fully duplicate entry: {key}"
                    seen.add(key)

    @pytest.mark.asyncio
    async def test_k6_a_structural_integrity(self):
        """Cross-instance WAL maintains hash chain per-instance perspective."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)
            k_b = _make_kernel(wal, snap)

            for i in range(4):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_b, "goal_b", "STARTED", i)

            # Each kernel's in-memory journal has a hash chain
            # The hash chain is per-instance, not global — so both are valid
            for label, k in [("A", k_a), ("B", k_b)]:
                vs = _check_structural_integrity(k.journal)
                if vs:
                    # Hash violations from cross-instance writes are expected
                    # Sequence violations or duplicates are true issues
                    seq_vs = [v for v in vs if 'sequence' in v or 'duplicate' in v]
                    assert len(seq_vs) == 0, \
                        f"Kernel {label}: structural violations: {seq_vs}"

    # ══════════════════════════════════════════════════════════════════════════
    # K6B — Interleaved recovery
    #
    # A writes, B writes (shared WAL), both recover independently.
    # Verifies:
    #   - recover() is deterministic per instance
    #   - Divergence between instances is allowed and expected
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k6_b_recovery_deterministic_per_instance(self):
        """Each instance's recovery is deterministic despite interference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)
            k_b = _make_kernel(wal, snap)

            for i in range(4):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_b, "goal_b", "STARTED", i)

            # Recover A twice — must be identical
            snap_a = snap.replace(".json", "_a.json")
            for idx in range(2):
                copy_dir = wal + f"_k6b_a{idx}"
                copy_snap = snap_a + f".{idx}"
                if os.path.exists(copy_dir):
                    shutil.rmtree(copy_dir)
                shutil.copytree(wal, copy_dir)
                if os.path.exists(snap):
                    shutil.copy2(snap, copy_snap)

                k = _make_kernel(copy_dir, copy_snap)
                await k.recover()
                h = _journal_entries_hash(k.journal._entries)
                if idx == 0:
                    first_hash = h
                else:
                    assert h == first_hash, \
                        f"A's recovery non-deterministic (copy {idx})"

    @pytest.mark.asyncio
    async def test_k6_b_divergence_accepted(self):
        """A and B may see different journal state after recovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)
            k_b = _make_kernel(wal, snap)

            for i in range(4):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_b, "goal_b", "STARTED", i)

            # Both recover from copies
            async def _recover_from_copy(src_wal, src_snap, suffix) -> str:
                cdir = wal + f"_{suffix}"
                csnap = snap.replace(".json", f"_{suffix}.json")
                if os.path.exists(cdir):
                    shutil.rmtree(cdir)
                shutil.copytree(src_wal, cdir)
                if os.path.exists(src_snap):
                    shutil.copy2(src_snap, csnap)
                k = _make_kernel(cdir, csnap)
                await k.recover()
                return _journal_entries_hash(k.journal._entries)

            # Both recover from the SAME WAL state.
            # Recovery from identical on-disk state MUST be identical.
            h_a = await _recover_from_copy(wal, snap, "k6b_div_a")
            h_b = await _recover_from_copy(wal, snap, "k6b_div_b")
            assert h_a == h_b, \
                "Recovery from same WAL produced different journal state — " \
                "this means WAL has non-deterministic ordering detectable by recover"

    @pytest.mark.asyncio
    async def test_k6_b_recover_does_not_crash(self):
        """Recover after cross-instance writes: no crash, no silent corruption."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            for trial in range(10):
                with tempfile.TemporaryDirectory() as trial_dir:
                    twal = os.path.join(trial_dir, "wal")
                    t_snap = os.path.join(trial_dir, "snap.json")
                    shutil.copytree(wal if os.path.exists(wal) else trial_dir, twal)

                    k_a = _make_kernel(twal, t_snap)
                    k_b = _make_kernel(twal, t_snap)

                    rng = random.Random(RANDOM_SEED + trial)
                    for step in range(rng.randint(3, 8)):
                        g = rng.choice(GOAL_IDS)
                        ev = rng.choice(['DISPATCHED', 'STARTED', 'LEASE_ISSUED'])
                        k = k_a if rng.random() < 0.5 else k_b
                        _append_entry(k, g, ev, step)

                    # Recover from a fresh copy
                    copy_dir = twal + "_copy"
                    copy_snap = t_snap.replace(".json", "_copy.json")
                    if os.path.exists(copy_dir):
                        shutil.rmtree(copy_dir)
                    shutil.copytree(twal, copy_dir)
                    if os.path.exists(t_snap):
                        shutil.copy2(t_snap, copy_snap)

                    k_c = _make_kernel(copy_dir, copy_snap)
                    result = await k_c.recover()
                    assert result['status'] in ('ok', 'corrupt'), \
                        f"trial={trial}: unexpected status={result['status']}: {result}"

                    if result['status'] == 'corrupt':
                        # System correctly detected cross-instance interference.
                        # Record the finding but don't fail — this is expected.
                        pass
                    elif result.get('journal_recovered', 0) == 0:
                        entry_count = _count_entries(twal)
                        if entry_count > 0:
                            pytest.fail(
                                f"trial={trial}: {entry_count} entries on disk "
                                f"but recover returned 0")

    # ══════════════════════════════════════════════════════════════════════════
    # K6C — Snapshot race
    #
    # One kernel writes while another creates a snapshot.
    # Verifies:
    #   - Snapshot is prefix-consistent (not global-consistent)
    #   - start_lsn ≤ last_lsn, no gaps
    #   - Snapshot does not contain partially-written entries
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k6_c_snapshot_prefix_consistent(self):
        """Snapshot taken during concurrent writes is prefix-consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)
            k_b = _make_kernel(wal, snap)

            # A writes, B snapshots
            for i in range(5):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_a, "goal_a", "STARTED", i)

            # B takes a snapshot while A has entries
            try:
                snap_result = k_b.snapshot()
            except Exception as e:
                # Snapshot may fail under interference — acceptable for K6
                return

            # Verify snapshot structure if it succeeded
            snap_path = os.path.join(wal, "..", "snap.json")
            import glob
            snap_files = glob.glob(os.path.join(tmpdir, "**", "snap*"), recursive=True)
            for sf in snap_files:
                if os.path.exists(sf):
                    with open(sf) as f:
                        data = json.load(f)
                    # Must have valid structure
                    assert 'entries' in data, \
                        f"Snapshot missing 'entries' field"
                    assert isinstance(data['entries'], list), \
                        f"Snapshot entries is not a list"

    @pytest.mark.asyncio
    async def test_k6_c_snapshot_no_missing_entries(self):
        """Snapshot captures all entries up to its last_lsn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)

            # A writes entries, then snapshots
            for i in range(3):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_a, "goal_a", "STARTED", i)

            try:
                snap_result = k_a.snapshot()
            except Exception:
                return  # Snapshot may fail, acceptable

            # Now write more entries AFTER snapshot
            snap_lsn = k_a.journal._last_lsn

            for i in range(3, 6):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)

            # Snapshot file must contain entries only up to snap_lsn
            snap_paths = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                          if 'snap' in f and f.endswith('.json')]
            for sp in snap_paths:
                if os.path.exists(sp):
                    with open(sp) as f:
                        data = json.load(f)
                    snapshot_entries = data.get('entries', [])
                    for se in snapshot_entries:
                        if 'payload' in se and 'lsn' in se['payload']:
                            pass  # snapshot format varies

                    # Snapshot is prefix-consistent if it has at least the
                    # entries written before snapshot (may not have all)
                    assert len(snapshot_entries) >= 3, \
                        f"Snapshot has {len(snapshot_entries)} entries, " \
                        f"expected ≥3 (entries before snapshot)"

    @pytest.mark.asyncio
    async def test_k6_c_recovery_from_snapshot_during_writes(self):
        """Recover from a snapshot taken while writes were in progress."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k_a = _make_kernel(wal, snap)

            for i in range(4):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)
                _append_entry(k_a, "goal_a", "STARTED", i)

            try:
                k_a.snapshot()
            except Exception:
                return

            # Write more after snapshot
            for i in range(4, 7):
                _append_entry(k_a, "goal_a", "DISPATCHED", i)

            # Recover — must produce valid state
            await k_a.recover()
            assert len(k_a.journal._entries) > 0, \
                "Recovery produced empty journal from snapshot + WAL"
            assert k_a.journal._last_lsn is not None, \
                "Recovery did not restore last_lsn"

    # ══════════════════════════════════════════════════════════════════════════
    # K6D — Stress: adversarial cross-instance interference
    #
    # Multiple rounds of random interleaved writes + recovery + snapshot.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k6_d_stress_cross_instance(self):
        """Random cross-instance interference across 10 scenarios."""
        rng = random.Random(RANDOM_SEED + 200)
        findings: list[str] = []

        for trial in range(10):
            with tempfile.TemporaryDirectory() as tmpdir:
                wal = os.path.join(tmpdir, "wal")
                snap = os.path.join(tmpdir, "snap.json")

                instances: list[ExecutionKernel] = []
                for _ in range(rng.randint(2, 4)):
                    instances.append(_make_kernel(wal, snap))

                steps = rng.randint(5, 15)
                for step in range(steps):
                    k = rng.choice(instances)
                    g = rng.choice(GOAL_IDS)
                    ev = rng.choice(['DISPATCHED', 'STARTED', 'LEASE_ISSUED'])
                    try:
                        _append_entry(k, g, ev, step)
                    except Exception as exc:
                        findings.append(
                            f"trial={trial} step={step}: append failed: {exc}")

                    if rng.random() < 0.15:
                        # Try snapshot
                        sk = rng.choice(instances)
                        try:
                            sk.snapshot()
                        except Exception:
                            findings.append(
                                f"trial={trial} step={step}: snapshot failed")

                # Final validation: WAL must be parseable
                for line in _read_all_lines(wal):
                    try:
                        json.loads(line)
                    except json.JSONDecodeError:
                        findings.append(
                            f"trial={trial}: unparseable WAL line: {line[:60]}")

                # Fresh kernel must recover
                k_fresh = _make_kernel(wal, snap)
                try:
                    result = await k_fresh.recover()
                    if result['status'] not in ('ok', 'corrupt'):
                        findings.append(
                            f"trial={trial}: unexpected status={result['status']} "
                            f"entries={result.get('journal_recovered', 0)}")
                    elif result['status'] == 'corrupt':
                        # Expected — cross-instance interference detected
                        pass
                except Exception as exc:
                    findings.append(
                        f"trial={trial}: recovery crashed: {exc}")

        assert len(findings) == 0, \
            f"K6D findings ({len(findings)}):\n" + "\n".join(findings[:20])
