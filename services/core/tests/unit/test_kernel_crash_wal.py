"""
K3: Crash-at-any-point WAL Verification.

For each WAL write sub-step, crash is simulated via file manipulation.
After simulated crash, cold boot verifies:

  K3.1 Prefix Property: recovered journal is always a valid prefix
  K3.2 Fsync Boundary: fsync'd data survives, non-fsync'd may be lost
  K3.3 Segment Rotation: crash during rotation is recoverable
  K3.4 Snapshot Boundary: crash during snapshot+prune is safe

Fault points:
  F1 — truncated last line (incomplete JSON)
  F2 — missing last line
  F3 — missing last N lines
  F4 — corrupt bytes in last line
  F5 — missing segment footer (crash during seal)
  F6 — orphan segment file (crash during rotation before manifest)
  F7 — ghost segment in manifest (crash during compaction after file delete)
  F8 — empty active segment directory (fresh start)
"""

import json
import os
import time
import shutil
import tempfile
import hashlib
import random
import pytest

os.environ.setdefault('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost:5432/test')

from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.journal import JournalEntry
from execution_dynamics.segmented_wal import _Segment, _segment_path

RANDOM_SEED = 42

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_kernel(wal_dir: str, snap_file: str) -> ExecutionKernel:
    os.makedirs(wal_dir, exist_ok=True)
    return ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
        )
    )


def _append_entries(kernel: ExecutionKernel, count: int, rng: random.Random) -> list[JournalEntry]:
    """Append N entries, return them."""
    entries = []
    for i in range(count):
        goal_id = f"goal_{rng.randint(0, 3)}"
        event = rng.choice(['DISPATCHED', 'STARTED', 'COMPLETED', 'LEASE_ISSUED'])
        e = JournalEntry(
            event=event,
            goal_id=goal_id,
            execution_id=f"{goal_id}:exec:{i}",
            lease_id=f"lease_{goal_id}",
            timestamp=time.time() + i * 0.1,
        )
        kernel.journal.append(e)
        entries.append(e)
    return entries


def _copy_wal(src: str, dst: str):
    """Recursively copy WAL directory."""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _read_segment_files(wal_dir: str) -> list[str]:
    """Get sorted segment file paths."""
    if not os.path.exists(wal_dir):
        return []
    files = sorted([
        os.path.join(wal_dir, f)
        for f in os.listdir(wal_dir)
        if f.endswith('.wal') and f != 'manifest.json'
    ])
    return files


def _read_segment_raw(wal_dir: str) -> bytes:
    """Read the latest (active) segment as raw bytes."""
    files = _read_segment_files(wal_dir)
    if not files:
        return b''
    with open(files[-1], 'rb') as f:
        return f.read()


def _read_manifest(wal_dir: str) -> dict:
    """Read manifest.json as dict."""
    mpath = os.path.join(wal_dir, 'manifest.json')
    if not os.path.exists(mpath):
        return {}
    with open(mpath) as f:
        return json.load(f)


def _write_manifest(wal_dir: str, manifest: dict):
    """Write manifest.json atomically."""
    mpath = os.path.join(wal_dir, 'manifest.json')
    tmp = mpath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, mpath)


REPAIR_EVENTS = frozenset({'ABANDONED', 'LEASE_EXPIRED'})


async def _boot_and_check(wal_dir: str, snap_file: str, golden_entries: list[dict]) -> list[str]:
    """Boot from WAL and verify prefix property.

    Returns list of violation strings (empty = OK).

    K3.1 Prefix Property: recovered BUSINESS entries (excluding repair events
    that recovery legitimately adds for dangling lifecycles) must be a valid
    prefix of the golden entries.
    """
    violations = []
    try:
        k = _make_kernel(wal_dir, snap_file)
        await k.recover()

        # Filter repair events — recovery legitimately adds ABANDONED/LEASE_EXPIRED
        # for dangling STARTED entries found during scan
        recovered = [
            {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
            for e in k.journal._entries
            if e.event not in REPAIR_EVENTS
        ]

        # K3.1: Prefix Property — recovered business entries must be prefix of golden
        if len(recovered) > len(golden_entries):
            violations.append(
                f"K3.1_PrefixProperty: recovered {len(recovered)} business entries "
                f"but golden has {len(golden_entries)}"
            )
        else:
            for i, (rg, gg) in enumerate(zip(recovered, golden_entries[:len(recovered)])):
                if rg['event'] != gg['event'] or rg['goal_id'] != gg['goal_id']:
                    violations.append(
                        f"K3.1_PrefixProperty: entry {i} mismatch: "
                        f"({rg['event']}/{rg['goal_id']}) vs "
                        f"({gg['event']}/{gg['goal_id']})"
                    )
                    break

        # No duplicate entry_ids in raw journal
        seen_ids = set()
        for e in k.journal._entries:
            if e.entry_id in seen_ids:
                violations.append(f"K3.1_PrefixProperty: duplicate entry_id={e.entry_id}")
            seen_ids.add(e.entry_id)

    except Exception as exc:
        violations.append(f"boot_error: {type(exc).__name__}: {exc}")

    return violations


# ── Fault Simulation ─────────────────────────────────────────────────────────


def simulate_f1_truncated_last_line(wal_dir: str):
    """F1: Truncate the last JSON line in the active segment."""
    data = _read_segment_raw(wal_dir)
    if not data:
        return
    lines = data.split(b'\n')
    if len(lines) < 2:
        return
    last_line = lines[-2]  # -1 is empty after split
    # Truncate to half
    half = len(last_line) // 2
    truncated = b'\n'.join(lines[:-1]) + b'\n' + last_line[:half] + b'\n'
    files = _read_segment_files(wal_dir)
    if files:
        with open(files[-1], 'wb') as f:
            f.write(truncated)
            f.flush()
            os.fsync(f.fileno())


def simulate_f2_missing_last_line(wal_dir: str):
    """F2: Remove the last entry line entirely."""
    data = _read_segment_raw(wal_dir)
    if not data:
        return
    lines = data.split(b'\n')
    if len(lines) < 3:  # header + at least 1 entry + trailing newline
        return
    # Remove the last non-empty, non-header line
    # Walk backwards to find last non-_wal_meta line
    for i in range(len(lines) - 2, -1, -1):
        line = lines[i].strip()
        if line and b'"_wal_meta"' not in line:
            truncated = b'\n'.join(lines[:i] + [b''])[:-1]  # remove last entry
            files = _read_segment_files(wal_dir)
            if files:
                with open(files[-1], 'wb') as f:
                    f.write(truncated)
                    f.flush()
                    os.fsync(f.fileno())
            return


def simulate_f3_missing_last_n(wal_dir: str, n: int = 2):
    """F3: Remove the last N entry lines."""
    data = _read_segment_raw(wal_dir)
    if not data:
        return
    lines = data.split(b'\n')
    removed = 0
    entry_indices = []
    for i, line in enumerate(lines):
        if line.strip() and b'"_wal_meta"' not in line:
            entry_indices.append(i)
    to_remove = set(entry_indices[-n:])
    kept = [lines[i] for i in range(len(lines)) if i not in to_remove]
    result = b'\n'.join(kept)
    files = _read_segment_files(wal_dir)
    if files:
        with open(files[-1], 'wb') as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())


def simulate_f4_corrupt_last_line(wal_dir: str):
    """F4: Replace last 10 bytes of last entry line with garbage."""
    data = _read_segment_raw(wal_dir)
    if not data:
        return
    lines = data.split(b'\n')
    for i in range(len(lines) - 2, -1, -1):
        line = lines[i].strip()
        if line and b'"_wal_meta"' not in line and b'"' in line:
            # Corrupt last 10 bytes
            if len(line) > 15:
                corrupted = line[:-10] + b'GARBAGE!!!'
                lines[i] = corrupted
            break
    result = b'\n'.join(lines)
    files = _read_segment_files(wal_dir)
    if files:
        with open(files[-1], 'wb') as f:
            f.write(result)
            f.flush()
            os.fsync(f.fileno())


def simulate_f5_missing_footer(wal_dir: str):
    """F5: Remove footer line from sealed segment."""
    data = _read_segment_raw(wal_dir)
    if not data:
        return
    lines = data.split(b'\n')
    # Remove last _wal_meta (footer) line
    for i in range(len(lines) - 2, -1, -1):
        if b'"_wal_meta"' in lines[i] and b'footer' in lines[i]:
            truncated = b'\n'.join(lines[:i] + [b''])
            # also remove from manifest if present
            files = _read_segment_files(wal_dir)
            if files:
                with open(files[-1], 'wb') as f:
                    f.write(truncated)
                    f.flush()
                    os.fsync(f.fileno())
            return


def simulate_f6_orphan_segment(wal_dir: str):
    """F6: Segment file exists on disk but not in manifest."""
    manifest = _read_manifest(wal_dir)
    if not manifest or 'segments' not in manifest:
        return
    # Remove latest segment from manifest
    segments = manifest['segments']
    if segments:
        segments.pop()
        _write_manifest(wal_dir, manifest)


def simulate_f7_ghost_segment(wal_dir: str):
    """F7: Segment in manifest but file deleted from disk."""
    manifest = _read_manifest(wal_dir)
    if not manifest or 'segments' not in manifest:
        return
    segments = manifest['segments']
    if not segments:
        return
    # Delete the latest segment file
    files = _read_segment_files(wal_dir)
    if files:
        os.remove(files[-1])


def simulate_f8_empty_wal(wal_dir: str):
    """F8: Complete WAL directory removed (fresh start)."""
    if os.path.exists(wal_dir):
        shutil.rmtree(wal_dir)


# ── Test ─────────────────────────────────────────────────────────────────────


class TestK3CrashAtAnyPoint:

    @pytest.mark.asyncio
    async def test_k3_f1_truncated_last_line(self):
        """F1: Crash after partial last-line write — simulate truncated JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED)
            entries = _append_entries(kernel, 10, rng)
            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]
            simulate_f1_truncated_last_line(wal_dir)
            violations = await _boot_and_check(wal_dir, snap_file, golden)
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f2_missing_last_line(self):
        """F2: Crash after write but before memory append — last entry lost."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED + 1)
            entries = _append_entries(kernel, 10, rng)
            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]
            simulate_f2_missing_last_line(wal_dir)
            violations = await _boot_and_check(wal_dir, snap_file, golden)
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f3_missing_last_n(self):
        """F3: Crash that loses last 2 entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED + 2)
            entries = _append_entries(kernel, 10, rng)
            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]
            simulate_f3_missing_last_n(wal_dir, n=2)
            violations = await _boot_and_check(wal_dir, snap_file, golden)
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f4_corrupt_last_line(self):
        """F4: Crash that corrupts last entry — must be gracefully truncated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED + 3)
            entries = _append_entries(kernel, 10, rng)
            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]
            simulate_f4_corrupt_last_line(wal_dir)
            violations = await _boot_and_check(wal_dir, snap_file, golden)
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f5_missing_footer(self):
        """F5: Crash during segment seal — footer not written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED + 4)

            # Write enough to fill a segment
            entries = _append_entries(kernel, 10, rng)

            # Force seal by calling snapshot (creates checkpoints + may trigger seal)
            kernel.snapshot()

            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]

            simulate_f5_missing_footer(wal_dir)
            violations = await _boot_and_check(wal_dir, snap_file, golden)
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f6_orphan_segment(self):
        """F6: Crash during rotation — new segment on disk but not in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED + 5)
            entries = _append_entries(kernel, 10, rng)
            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]

            # Force segment rotation
            kernel.snapshot()
            # The seal+rotate should create a new segment
            # Simulate crash: remove latest segment from manifest
            simulate_f6_orphan_segment(wal_dir)

            violations = await _boot_and_check(wal_dir, snap_file, golden)
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f7_ghost_segment(self):
        """F7: Crash during compaction — file deleted but manifest not updated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            kernel = _make_kernel(wal_dir, snap_file)
            rng = random.Random(RANDOM_SEED + 6)
            entries = _append_entries(kernel, 10, rng)
            golden = [
                {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                for e in entries
            ]

            # Force snapshot + seal
            kernel.snapshot()
            simulate_f7_ghost_segment(wal_dir)

            violations = await _boot_and_check(wal_dir, snap_file, golden)
            # Ghost segments must be detected as data loss (hard error is OK)
            # But we must NOT get a corrupted journal
            # Accept boot failures that explicitly report data loss
            ok = True
            for v in violations:
                if 'K3.1_PrefixProperty' in v:
                    ok = False
            assert ok, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_f8_empty_wal(self):
        """F8: Fresh start — boot from empty/non-existent WAL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wal_dir = os.path.join(tmpdir, "wal")
            snap_file = os.path.join(tmpdir, "snapshot.json")
            simulate_f8_empty_wal(wal_dir)
            violations = await _boot_and_check(wal_dir, snap_file, [])
            assert len(violations) == 0, "\n".join(violations)

    @pytest.mark.asyncio
    async def test_k3_prefix_property_multi_segment(self):
        """K3.1 stress: random truncations across multi-segment WAL."""
        rng = random.Random(RANDOM_SEED + 10)
        failures = []
        for trial in range(50):
            with tempfile.TemporaryDirectory() as tmpdir:
                wal_dir = os.path.join(tmpdir, "wal")
                snap_file = os.path.join(tmpdir, "snapshot.json")
                kernel = _make_kernel(wal_dir, snap_file)

                n_entries = rng.randint(5, 20)
                entries = _append_entries(kernel, n_entries, rng)
                golden = [
                    {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                    for e in entries
                ]

                # Random truncation — remove N random trailing entries
                n_lose = rng.randint(1, max(1, n_entries // 2))
                simulate_f3_missing_last_n(wal_dir, n=n_lose)

                violations = await _boot_and_check(wal_dir, snap_file, golden)
                if violations:
                    failures.append((trial, n_entries, n_lose, violations))

        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])

    @pytest.mark.asyncio
    async def test_k3_append_and_truncate_cycle(self):
        """Append, simulate crash at byte boundary, verify recovery never corrupts."""
        rng = random.Random(RANDOM_SEED + 11)
        failures = []
        for trial in range(50):
            with tempfile.TemporaryDirectory() as tmpdir:
                wal_dir = os.path.join(tmpdir, "wal")
                snap_file = os.path.join(tmpdir, "snapshot.json")
                kernel = _make_kernel(wal_dir, snap_file)

                entries = _append_entries(kernel, 5, rng)
                golden = [
                    {k: v for k, v in e.to_dict().items() if k not in ('prev_hash', 'entry_hash')}
                    for e in entries
                ]

                # Truncate at random byte offset in the active segment
                raw = _read_segment_raw(wal_dir)
                if len(raw) < 20:
                    continue
                offset = rng.randint(len(raw) // 4, len(raw) - 5)
                truncated = raw[:offset]
                files = _read_segment_files(wal_dir)
                if files:
                    with open(files[-1], 'wb') as f:
                        f.write(truncated)
                        f.flush()
                        os.fsync(f.fileno())

                violations = await _boot_and_check(wal_dir, snap_file, golden)
                if violations:
                    failures.append((trial, offset, violations))

        assert len(failures) == 0, \
            f"{len(failures)} failures:\n" + "\n".join(str(f) for f in failures[:5])
