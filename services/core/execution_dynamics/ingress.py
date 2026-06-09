"""
KernelIngress — the SINGLE public entry point for execution.

ARCHITECTURAL ENFORCEMENT:
  - KernelIngress is the ONLY way to request execution from the kernel.
  - It mints HMAC-signed capabilities that the kernel validates.
  - No code outside KERNEL zone can call ExecutionKernel methods directly.
  - Every dispatch request is zone-validated, capability-minted, and audited.

FLOW:
  Caller Zone → KernelIngress.dispatch()
      → zone validation
      → capability minting (signed with secret)
      → kernel._dispatch_with_capability(cap, ...)
          → signature verification
          → zone check
          → lease acquire → journal → executor

THREAT MODEL:
  - Direct import of ExecutionKernel → dispatch() will fail signature check
  - Forged capability → HMAC will not match → rejected
  - Replayed capability → epoch mismatch → rejected
  - Wrong goal_id → scope check → rejected
"""

import logging
import secrets
from typing import Optional

from .capability import KernelCapability
from .zone import Zone, Permission, validate_zone_permission

logger = logging.getLogger(__name__)


class KernelIngress:
    """
    Single public entry point for execution requests.

    KernelIngress holds the HMAC signing secret. Only it can mint valid
    capabilities. Every dispatch request produces an audited, signed
    capability that ExecutionKernel validates before executing.

    This is NOT an authentication layer. This is a TOPOLOGY OF AUTHORITY
    enforcement — ensuring no code path can reach the kernel without
    going through a controlled ingress point.
    """

    def __init__(self, kernel=None):
        """
        Initialize the ingress.

        Args:
            kernel: ExecutionKernel instance. If None, uses global kernel.
        """
        # The signing secret — NEVER exposed outside this class.
        # If the secret is compromised, rotate it and all capabilities
        # will be invalidated (kernel epoch increment required).
        self._secret = secrets.token_hex(32)

        if kernel is not None:
            self._kernel = kernel
        else:
            from . import _get_kernel as _get_kernel_
            self._kernel = _get_kernel_()

        # Register secret with kernel for signature verification
        self._kernel._set_ingress_secret(self._secret)

        self._call_count = 0
        logger.info("kernel_ingress_initialized")

    # ------------------------------------------------------------------
    # PUBLIC ENTRY POINT
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        goal_id: str,
        uow=None,
        *,
        zone: Zone = Zone.APPLICATION,
        retry_count: int = 0,
        user_priority: float = 0.5,
        dispatch_id: str = "",
        active_task_count: int = 0,
        blocked_dependents: int = 0,
        hours_idle: float = 24.0,
    ) -> dict:
        """
        Request execution through the kernel.

        This is the ONLY way external code should request execution.
        Every call:
          1. Validates the requesting zone has DISPATCH permission
          2. Mints a signed, scoped KernelCapability
          3. Passes the capability to the kernel for execution

        Args:
            goal_id: goal to execute
            uow: UnitOfWork (optional, created if None)
            zone: requesting zone (default APPLICATION)
            retry_count: retry attempt number
            user_priority: user-assigned priority (0.0-1.0)
            dispatch_id: idempotency key
            active_task_count: concurrent task count
            blocked_dependents: goals blocked on this one
            hours_idle: hours since last execution

        Returns:
            ExecutionResult-compatible dict
        """
        self._call_count += 1

        # 1. Zone validation
        if not validate_zone_permission(zone, Permission.DISPATCH):
            logger.error(
                f"ingress_zone_denied zone={zone.value} "
                f"goal_id={goal_id} call={self._call_count}"
            )
            return {
                'success': False,
                'error': (
                    f"Zone '{zone.value}' does not have DISPATCH permission. "
                    f"All execution must go through APPLICATION zone or KernelIngress."
                ),
                'artifacts': [],
                'did_skip': True,
                'security_event': 'zone_denied',
            }

        # 2. Mint signed capability
        cap = KernelCapability.mint(
            scope="dispatch",
            goal_id=goal_id,
            zone=zone.value,
            kernel_epoch=self._kernel._capability_epoch,
            secret=self._secret,
        )

        # 3. Dispatch via kernel (capability-gated)
        if uow is not None:
            result = await self._kernel._dispatch_with_capability(
                goal_id=goal_id,
                uow=uow,
                capability=cap,
                retry_count=retry_count,
                user_priority=user_priority,
                dispatch_id=dispatch_id,
                active_task_count=active_task_count,
                blocked_dependents=blocked_dependents,
                hours_idle=hours_idle,
            )
            return self._result_to_dict(result)

        # No UoW provided — create one
        from infrastructure.uow import create_uow_provider
        get_uow = create_uow_provider()
        async with get_uow() as uow:
            result = await self._kernel._dispatch_with_capability(
                goal_id=goal_id,
                uow=uow,
                capability=cap,
                retry_count=retry_count,
                user_priority=user_priority,
                dispatch_id=dispatch_id,
                active_task_count=active_task_count,
                blocked_dependents=blocked_dependents,
                hours_idle=hours_idle,
            )
            return self._result_to_dict(result)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _result_to_dict(self, result) -> dict:
        return {
            'success': result.success,
            'artifacts': result.artifacts,
            'error': result.error,
            'execution_id': result.execution_id,
            'lease_id': result.lease_id,
            'dispatch_epoch': result.dispatch_epoch,
            'did_skip': result.did_skip,
            'duration_ms': result.duration_ms,
            'coherence_index': result.coherence_index,
            'priority_score': result.priority_score,
            'execution_pressure': result.execution_pressure,
        }

    def get_stats(self) -> dict:
        return {
            'call_count': self._call_count,
            'has_kernel': self._kernel is not None,
        }


# ============================================================================
# Global ingress singleton
# ============================================================================

_global_ingress: Optional[KernelIngress] = None


def get_ingress() -> KernelIngress:
    """Get or create the global ingress singleton."""
    global _global_ingress
    if _global_ingress is None:
        _global_ingress = KernelIngress()
    return _global_ingress


def set_ingress(ingress: KernelIngress):
    """Set the global ingress (useful for testing)."""
    global _global_ingress
    _global_ingress = ingress
