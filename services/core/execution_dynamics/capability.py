"""
KernelCapability — cryptographically signed, scoped, lease-bound, revocable
execution authority token.

DESIGN:
  - NON-FORGEABLE: each capability is HMAC-SHA256 signed with a kernel secret
    known only to KernelIngress. ExecutionKernel verifies the signature on
    every dispatch call.
  - SCOPED: carries goal_id, scope, and zone provenance. The kernel checks
    that the capability authorizes exactly this goal.
  - LEASE-BOUND: references the lease_id authorizing execution. Lease is
    validated independently.
  - REVOCABLE: kernel epoch-based bulk revocation + per-capability revoke flag.
  - PROVENANCE: records which zone issued it, enabling audit trails.

ARCHITECTURAL ENFORCEMENT:
  KernelCapability can ONLY be minted by KernelIngress (which holds the secret).
  Application code cannot construct a valid capability — the HMAC signature
  will not match.
"""

import hmac
import hashlib
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# KernelCapability — cryptographically signed authority token
# ============================================================================

class KernelCapability:
    """
    Ephemeral, scoped, lease-bound, revocable, NON-FORGEABLE capability.

    LIFE CYCLE:
      KernelIngress.mint()  →  signed capability  →  kernel.dispatch()
                                                       ↓
                                              kernel validates signature
                                              kernel validates epoch
                                              kernel validates scope
                                              kernel validates lease

    THREAT MODEL:
      - If an attacker gets a capability, they can dispatch the exact goal
        it authorizes, within the lease TTL, within the kernel epoch.
      - Capabilities cannot be forged without the kernel secret.
      - Capabilities cannot be replayed after lease expiry or epoch change.
      - Capabilities cannot be scoped to a different goal.
    """

    def __init__(
        self,
        scope: str,
        goal_id: str,
        lease_id: str = "",
        zone: str = "application",
        capability_id: str = "",
        signature: str = "",
    ):
        self.capability_id = capability_id or hashlib.sha256(
            f"{scope}:{goal_id}:{lease_id}:{time.time()}".encode()
        ).hexdigest()[:16]
        self.scope = scope
        self.goal_id = goal_id
        self.lease_id = lease_id
        self.zone = zone
        self.signature = signature
        self._created_at = time.time()
        self._revoked = False
        self._kernel_epoch = 0

    # ------------------------------------------------------------------
    # Minting (only KernelIngress should call this)
    # ------------------------------------------------------------------

    @classmethod
    def mint(
        cls,
        scope: str,
        goal_id: str,
        lease_id: str = "",
        zone: str = "application",
        kernel_epoch: int = 0,
        secret: str = "",
    ) -> 'KernelCapability':
        """
        Create a cryptographically signed capability.

        Args:
            scope: what operation this authorizes (e.g. "dispatch")
            goal_id: the goal being authorized
            lease_id: the lease binding this capability to
            zone: provenance zone (who requested this)
            kernel_epoch: current kernel epoch (for bulk revocation)
            secret: HMAC signing key (held by KernelIngress)

        Returns:
            KernelCapability with a valid HMAC-SHA256 signature.

        Raises:
            ValueError: if secret is empty (capability would be forgeable)
        """
        if not secret:
            raise ValueError(
                "Cannot mint capability without a secret. "
                "Capabilities must be signed to be non-forgeable."
            )

        cap = cls(
            scope=scope,
            goal_id=goal_id,
            lease_id=lease_id,
            zone=zone,
        )
        cap._kernel_epoch = kernel_epoch
        cap.signature = cap._compute_signature(secret)
        return cap

    # ------------------------------------------------------------------
    # Signature verification (called by ExecutionKernel)
    # ------------------------------------------------------------------

    def verify(self, secret: str) -> bool:
        """
        Verify the HMAC-SHA256 signature against a secret.

        Returns True if the capability was signed with this secret
        and has not been tampered with.
        """
        if not self.signature or not secret:
            return False
        expected = self._compute_signature(secret)
        return hmac.compare_digest(self.signature, expected)

    def _compute_signature(self, secret: str) -> str:
        """
        Compute deterministic HMAC-SHA256 over capability fields.

        The signature covers:
          - scope, goal_id, lease_id, zone, kernel_epoch

        Any tampering with these fields invalidates the signature.
        """
        message = (
            f"{self.scope}:{self.goal_id}:{self.lease_id}:"
            f"{self.zone}:{self._kernel_epoch}"
        )
        return hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # ------------------------------------------------------------------
    # Revocation
    # ------------------------------------------------------------------

    def revoke(self):
        """Mark this capability as revoked. Permanent."""
        self._revoked = True

    def is_valid(self, kernel_epoch: int = 0) -> bool:
        """
        Check if this capability is still valid.

        A capability is valid if:
          1. Not explicitly revoked
          2. Created within TTL (1 hour)
          3. Kernel epoch matches (epoch-based bulk revocation)
        """
        if self._revoked:
            return False
        age = time.time() - self._created_at
        if age > 3600:
            return False
        if kernel_epoch and self._kernel_epoch != kernel_epoch:
            return False
        return True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            'capability_id': self.capability_id,
            'scope': self.scope,
            'goal_id': self.goal_id,
            'lease_id': self.lease_id,
            'zone': self.zone,
            'has_signature': bool(self.signature),
            'age_seconds': round(time.time() - self._created_at, 2),
            'revoked': self._revoked,
            'kernel_epoch': self._kernel_epoch,
        }
