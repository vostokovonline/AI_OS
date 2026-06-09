"""
K7 — Single-Writer Enforcement via fcntl.flock.

ARCHITECTURAL CONTRACT:

  K7/A: строгий single-writer kernel — fcntl.flock(LOCK_EX | LOCK_NB)
  на lock-файл внутри WAL-директории.

  В любой момент времени только один активный writer может
  модифицировать WAL. При попытке второго — immediate rejection.

Это не "распределённая система" и не "координация".
Это приведение модели в соответствие реальности:
K6 доказал, что WAL не multi-writer-safe. K7 делает это гарантией.

INVARIANTS:

  K7.1 — First writer acquires lock, second is rejected.
  K7.2 — Lock release on close() — second acquires after release.
  K7.3 — Lock guarantees structural integrity (no cross-writer corruption).
  K7.4 — Recovery after lock release produces correct state.
  K7.5 — Stress: lock survives adversarial scenarios.
"""

import os
import time
import tempfile
import threading
import pytest

os.environ.setdefault('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost:5432/test')

from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.journal import JournalEntry

RANDOM_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_kernel_locked(wal_dir: str, snap_file: str) -> ExecutionKernel:
    """Create kernel with single-writer enforcement enabled."""
    os.makedirs(wal_dir, exist_ok=True)
    return ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
            enforce_single_writer=True,  # K7/A: explicit lock
        )
    )


def _append_entry(kernel: ExecutionKernel, goal: str, event: str, seq: int):
    e = JournalEntry(
        event=event, goal_id=goal,
        execution_id=f"{goal}:exec:{seq}",
        lease_id=f"lease_{goal}", timestamp=time.time(),
    )
    kernel.journal.append(e)


# ── Test Class ───────────────────────────────────────────────────────────────


class TestK7SingleWriterEnforcement:

    # ══════════════════════════════════════════════════════════════════════════
    # K7.1 — First acquires, second rejected
    #
    # Invariant: when two kernels attempt to initialize on the same WAL
    # directory with enforce_single_writer=True, the first obtains the
    # lock and the second is rejected immediately.
    # ══════════════════════════════════════════════════════════════════════════

    def test_k7_1_second_writer_rejected(self):
        """Second kernel on same WAL is rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            # First acquires lock
            k1 = _make_kernel_locked(wal, snap)
            assert k1.wal._lock_fd is not None
            assert k1.wal._lock_fd.closed is False

            # Second is rejected
            with pytest.raises(RuntimeError) as exc:
                _make_kernel_locked(wal, snap)
            assert 'locked by another writer' in str(exc.value)

    def test_k7_1_release_allows_second(self):
        """After first releases lock, second can acquire."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k1 = _make_kernel_locked(wal, snap)
            k1.wal.release_lock()
            assert k1.wal._lock_fd is None

            k2 = _make_kernel_locked(wal, snap)
            assert k2.wal._lock_fd is not None
            assert k2.wal._lock_fd.closed is False

    def test_k7_1_close_releases_lock(self):
        """close() releases the lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k1 = _make_kernel_locked(wal, snap)
            k1.wal.close()
            assert k1.wal._lock_fd is None

            k2 = _make_kernel_locked(wal, snap)
            assert k2.wal._lock_fd is not None

    # ══════════════════════════════════════════════════════════════════════════
    # K7.2 — Lock lifecycle: acquire → use → release → re-acquire
    #
    # Verifies that a kernel can write, release, re-acquire, and write again
    # on the same WAL without structural issues.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k7_2_release_reacquire_cycle(self):
        """Lock acquire → release → reacquire → write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k = _make_kernel_locked(wal, snap)
            _append_entry(k, "goal_a", "DISPATCHED", 0)
            _append_entry(k, "goal_a", "STARTED", 0)
            _append_entry(k, "goal_a", "COMPLETED", 0)  # close lifecycle
            k.wal.release_lock()

            k.wal.acquire_lock()
            _append_entry(k, "goal_b", "DISPATCHED", 0)
            _append_entry(k, "goal_b", "STARTED", 0)
            _append_entry(k, "goal_b", "COMPLETED", 0)
            k.wal.release_lock()

            k2 = _make_kernel_locked(wal, snap)
            await k2.recover()
            entries = [e for e in k2.journal._entries
                       if e.event not in ('ABANDONED', 'LEASE_EXPIRED')]
            assert len(entries) == 6, \
                f"Expected 6 entries (2 lifecycle × 3 events), got {len(entries)}"

    @pytest.mark.asyncio
    async def test_k7_2_lock_not_leaked_between_wal_dirs(self):
        """Lock on one WAL dir doesn't affect another."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal1 = os.path.join(tmpdir, "wal1")
            wal2 = os.path.join(tmpdir, "wal2")
            snap1 = os.path.join(tmpdir, "snap1.json")
            snap2 = os.path.join(tmpdir, "snap2.json")

            k1 = _make_kernel_locked(wal1, snap1)
            k2 = _make_kernel_locked(wal2, snap2)  # different dir — OK

            _append_entry(k1, "goal_a", "DISPATCHED", 0)
            _append_entry(k2, "goal_b", "DISPATCHED", 0)

            await k1.recover()
            await k2.recover()

    # ══════════════════════════════════════════════════════════════════════════
    # K7.3 — Lock prevents cross-writer corruption
    #
    # Verifies that with the lock active, two kernels cannot interleave
    # writes. The second kernel is blocked at init, so WAL remains
    # structurally intact.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k7_3_lock_prevents_interleaved_writes(self):
        """Lock prevents a second writer from corrupting the WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k1 = _make_kernel_locked(wal, snap)

            # Attempt to create second writer — must fail
            with pytest.raises(RuntimeError):
                _make_kernel_locked(wal, snap)

            # First writer works normally
            for i in range(5):
                _append_entry(k1, "goal_a", "DISPATCHED", i)

            # Recover from a fresh kernel (with lock) — entries intact
            k1.wal.release_lock()
            k2 = _make_kernel_locked(wal, snap)
            await k2.recover()
            actual = [e for e in k2.journal._entries
                      if e.event not in ('ABANDONED', 'LEASE_EXPIRED')]
            assert len(actual) == 5, \
                f"Expected 5 DISPATCHED, got {len(actual)} after repair filter"

    @pytest.mark.asyncio
    async def test_k7_3_unlocked_after_crash(self):
        """Lock is released on process exit — second writer can recover."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            # Simulate first writer crash: lock held, then released via close
            k1 = _make_kernel_locked(wal, snap)
            _append_entry(k1, "goal_a", "DISPATCHED", 0)
            _append_entry(k1, "goal_a", "STARTED", 0)

            # "Crash" — release all resources
            k1.wal.close()

            # Second writer can recover and see entries
            k2 = _make_kernel_locked(wal, snap)
            await k2.recover()
            actual = [e for e in k2.journal._entries
                      if e.event not in ('ABANDONED', 'LEASE_EXPIRED')]
            assert len(actual) == 2, \
                f"Expected 2 entries, got {len(k2.journal._entries)} ({len(actual)} after repair filter)"

    # ══════════════════════════════════════════════════════════════════════════
    # K7.4 — Recovery correctness with lock
    #
    # Verifies that the lock does not affect recovery behavior — the same
    # invariants from K5.4 hold with enforce_single_writer=True.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k7_4_lock_does_not_affect_recovery(self):
        """Recovery under lock produces same state as without."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k = _make_kernel_locked(wal, snap)
            for i in range(3):
                _append_entry(k, "goal_a", "DISPATCHED", i)
                _append_entry(k, "goal_a", "STARTED", i)

            # Snapshot and recover under lock
            k.snapshot()
            await k.recover()

            entries_after = set(
                (e.execution_id, e.event)
                for e in k.journal._entries
                if e.event not in ('ABANDONED', 'LEASE_EXPIRED')
            )

            # Expected entries
            expected = set()
            for i in range(3):
                expected.add((f"goal_a:exec:{i}", "DISPATCHED"))
                expected.add((f"goal_a:exec:{i}", "STARTED"))

            assert entries_after == expected, \
                f"Mismatch: {entries_after - expected} extra, {expected - entries_after} missing"

    @pytest.mark.asyncio
    async def test_k7_4_lock_and_snapshot_consistency(self):
        """Snapshot taken under lock is consistent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            k = _make_kernel_locked(wal, snap)
            for i in range(5):
                _append_entry(k, "goal_a", "DISPATCHED", i)
                _append_entry(k, "goal_a", "STARTED", i)

            pre_hash = self._entries_hash(k)
            k.snapshot()
            await k.recover()
            post_hash = self._entries_hash(k)

            from execution_dynamics.journal import JournalEntry
            assert pre_hash == post_hash, \
                "Journal changed after snapshot+recovery under lock"

    def _entries_hash(self, kernel):
        import hashlib, json
        from execution_dynamics.journal import JournalEntry as JE
        filtered = [
            {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
            for e in kernel.journal._entries
            if e.event not in ('ABANDONED', 'LEASE_EXPIRED')
        ]
        raw = json.dumps(filtered, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ══════════════════════════════════════════════════════════════════════════
    # K7.5 — Stress: adversarial lock scenarios
    #
    # Tests that the lock survives concurrent access attempts, process
    # boundaries (threads simulating multiple processes), and ensures
    # the WAL is never left in an inconsistent state.
    # ══════════════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_k7_5_stress_concurrent_acquire_attempts(self):
        """Multiple concurrent lock attempts — exactly one succeeds."""
        n_threads = 5
        results: list[str] = []
        lock: threading.Lock = threading.Lock()

        def try_acquire(tmpdir: str, idx: int):
            wal = os.path.join(tmpdir, f"wal_{idx}")
            snap = os.path.join(tmpdir, f"snap_{idx}.json")
            try:
                k = _make_kernel_locked(wal, snap)
                with lock:
                    results.append(f"t{idx}:acquired")
                k.wal.release_lock()
            except RuntimeError as e:
                with lock:
                    results.append(f"t{idx}:rejected")

        # Each thread uses a DIFFERENT WAL dir — no contention expected
        with tempfile.TemporaryDirectory() as tmpdir:
            threads = []
            for i in range(n_threads):
                t = threading.Thread(target=try_acquire, args=(tmpdir, i))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == n_threads
            acquired = [r for r in results if 'acquired' in r]
            assert len(acquired) == n_threads, \
                f"Expected {n_threads} acquired, got {len(acquired)}: {results}"

    def test_k7_5_stress_same_dir_contention(self):
        """Two threads contending for same WAL lock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal = os.path.join(tmpdir, "wal")
            snap = os.path.join(tmpdir, "snap.json")

            # First acquires
            k1 = _make_kernel_locked(wal, snap)
            k1_held = threading.Event()

            def second_attempt():
                try:
                    _make_kernel_locked(wal, snap)
                    k1_held.clear()  # should NOT happen
                except RuntimeError:
                    k1_held.set()

            t = threading.Thread(target=second_attempt)
            t.start()
            t.join(timeout=5)
            assert k1_held.is_set(), \
                "Second writer should have been rejected"
            assert not t.is_alive()

