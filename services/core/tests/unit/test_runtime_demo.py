"""
R0.1 — Runtime Demo: Crash → Recover → Continue.

Tests the full vertical slice:
  Task → Graph → Kernel Dispatch → Worker → Journal → Recovery → Resume
"""

import os
import json
import time
import tempfile
import shutil
import hashlib
import pytest

# Set DATABASE_URL to avoid lazy-import crash from infrastructure.uow
# asyncpg is available in the test environment
os.environ.setdefault('DATABASE_URL', 'postgresql+asyncpg://user:pass@localhost:5432/test')

from aios_runtime.task import Task, Node
from aios_runtime.graph import ExecutionGraph, NodeState
from aios_runtime.runtime import Runtime
from execution_dynamics.kernel import ExecutionKernel, ExecutionConfig
from execution_dynamics.ingress import KernelIngress


def _make_kernel_and_ingress(tmpdir):
    """Create a PRIMARY-path kernel + ingress for testing."""
    wal_dir = os.path.join(tmpdir, "wal")
    snap_file = os.path.join(tmpdir, "snapshot.json")
    os.makedirs(wal_dir, exist_ok=True)
    kernel = ExecutionKernel(
        config=ExecutionConfig(
            wal_path=wal_dir,
            snapshot_path=snap_file,
        )
    )
    ingress = KernelIngress(kernel=kernel)
    return kernel, ingress


class TestRuntimeDemo:

    @pytest.mark.asyncio
    async def test_runtime_completes_simple_task(self):
        """Three sequential noop nodes execute to completion."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel, ingress = _make_kernel_and_ingress(tmpdir)
            rt = Runtime(ingress)

            task = Task(
                task_id="demo_simple",
                nodes=[
                    Node(node_id="step1", type="noop", config={}),
                    Node(node_id="step2", type="noop", config={},
                         depends_on=["step1"]),
                    Node(node_id="step3", type="noop", config={},
                         depends_on=["step2"]),
                ]
            )

            result = await rt.run(task)
            assert result['status'] == 'complete', f"Expected complete, got {result}"
            assert result['terminal'] is True
            assert all(s == 'completed' for s in result['node_states'].values())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_runtime_crash_resume_completes(self):
        """Simulate crash mid-execution; resume via new kernel on same WAL completes all nodes."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel1, ingress1 = _make_kernel_and_ingress(tmpdir)
            rt1 = Runtime(ingress1)

            task = Task(
                task_id="demo_crash",
                nodes=[
                    Node(node_id="a", type="noop", config={}),
                    Node(node_id="b", type="noop", config={},
                         depends_on=["a"]),
                    Node(node_id="c", type="noop", config={},
                         depends_on=["b"]),
                ]
            )

            # Run to completion on first kernel
            result1 = await rt1.run(task)
            assert result1['status'] == 'complete'
            assert result1['terminal'] is True

            # Simulate crash: create new kernel on same WAL
            kernel2, ingress2 = _make_kernel_and_ingress(tmpdir)
            rt2 = Runtime(ingress2)

            # Resume — should find all nodes already complete in journal
            result2 = await rt2.run(task)
            assert result2['status'] == 'already_complete', f"Expected already_complete, got {result2}"
            assert result2['terminal'] is True
            assert all(s == 'completed' for s in result2['node_states'].values())
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_runtime_resume_after_partial_execution(self):
        """Crash mid-task; resume from where we left off."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel1, ingress1 = _make_kernel_and_ingress(tmpdir)
            rt1 = Runtime(ingress1)

            task = Task(
                task_id="demo_partial",
                nodes=[
                    Node(node_id="x", type="noop", config={}),
                    Node(node_id="y", type="noop", config={},
                         depends_on=["x"]),
                    Node(node_id="z", type="noop", config={},
                         depends_on=["y"]),
                ]
            )

            # Execute only first node manually via ingress
            r1 = await ingress1.dispatch(goal_id="demo_partial:x", dispatch_id="px1")
            assert r1.get('success'), f"Dispatch x failed: {r1}"
            # Mark via journal that x completed (worker would normally do this)
            from execution_dynamics.journal import JournalEntry
            kernel1.journal.append(JournalEntry(
                event='COMPLETED',
                goal_id="demo_partial:x",
                execution_id="ex_x",
                lease_id="lx_x",
                timestamp=time.time(),
                success=True,
                duration_ms=10,
            ))

            # Simulate crash: new kernel on same WAL
            kernel2, ingress2 = _make_kernel_and_ingress(tmpdir)
            rt2 = Runtime(ingress2)

            # Resume — should find x=completed, rebuild graph, run y and z
            result2 = await rt2.run(task)
            assert result2['status'] == 'complete', f"Expected complete, got {result2}"
            assert result2['terminal'] is True
            assert result2['node_states'].get('x') == 'completed'
            assert result2['node_states'].get('y') == 'completed'
            assert result2['node_states'].get('z') == 'completed'
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_runtime_worker_echo(self):
        """Worker correctly executes an echo node."""
        tmpdir = tempfile.mkdtemp()
        try:
            kernel, ingress = _make_kernel_and_ingress(tmpdir)
            rt = Runtime(ingress)

            task = Task(
                task_id="demo_echo",
                nodes=[
                    Node(node_id="echo1", type="echo", config={"message": "hello world"}),
                ]
            )

            result = await rt.run(task)
            assert result['status'] == 'complete'
            assert result['node_states']['echo1'] == 'completed'
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
