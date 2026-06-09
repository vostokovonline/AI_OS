"""
AI-OS Execution Dynamics — execution substrate.

PUBLIC API:
  KernelIngress     — Single public entry point for execution
  KernelCapability  — HMAC-signed, scoped, revocable capability token
  Zone              — Trust zone enum (KERNEL, APPLICATION, EXTERNAL)
  dispatch_goal()   — Convenience function wrapping KernelIngress

ARCHITECTURAL ENFORCEMENT:
  - ExecutionKernel is NOT exported. It cannot be imported from this package.
  - All execution must go through KernelIngress (which mints signed capabilities).
  - Direct kernel access without a valid capability is rejected + audited.
  - This is the topology of authority: caller → ingress → capability → kernel.

USAGE:
    from execution_dynamics import dispatch_goal

    result = await dispatch_goal(goal_id="...", uow=...)

    # Or with explicit zone control:
    from execution_dynamics import KernelIngress, Zone

    ingress = KernelIngress()
    result = await ingress.dispatch(goal_id="...", zone=Zone.APPLICATION)
"""

from execution_dynamics.capability import KernelCapability
from execution_dynamics.ingress import KernelIngress, get_ingress, set_ingress
from execution_dynamics.zone import Zone


# ============================================================================
# Internal: global kernel singleton (PRIVATE — do not import)
# ============================================================================

from execution_dynamics.kernel import ExecutionKernel as _ExecutionKernel

_global_kernel: _ExecutionKernel = _ExecutionKernel()


def _get_kernel() -> _ExecutionKernel:
    """Get global kernel singleton (PRIVATE)."""
    return _global_kernel


# ============================================================================
# Convenience: dispatch_goal() — wraps global ingress
# ============================================================================

async def dispatch_goal(
    goal_id: str,
    uow=None,
    retry_count: int = 0,
    user_priority: float = 0.5,
    dispatch_id: str = "",
) -> dict:
    """
    Execute a goal through the kernel via KernelIngress.

    This is the RECOMMENDED way for application code to execute goals.
    Wraps KernelIngress.dispatch() with sensible defaults.

    Args:
        goal_id: goal to execute
        uow: UnitOfWork (optional, created if None)
        retry_count: retry attempt number
        user_priority: user-assigned priority (0.0-1.0)
        dispatch_id: idempotency key

    Returns:
        ExecutionResult-compatible dict
    """
    ingress = get_ingress()
    return await ingress.dispatch(
        goal_id=goal_id,
        uow=uow,
        zone=Zone.APPLICATION,
        retry_count=retry_count,
        user_priority=user_priority,
        dispatch_id=dispatch_id,
    )
