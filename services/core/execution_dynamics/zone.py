"""
Trust Zones — topological authority boundaries for the execution kernel.

ZONES:
  KERNEL       — owns lease, journal, snapshot, epoch. No public API.
  APPLICATION  — can request dispatch via KernelIngress. Cannot import kernel
                 internals. All execution requests produce a signed capability.
  EXTERNAL     — API-driven (REST, webhook, Telegram). Requests go through
                 APPLICATION zone. Cannot mint capabilities on its own.

ARCHITECTURAL RULE:
  Caller Zone → KernelIngress → capability minting → ExecutionKernel → lease → journal

  No code outside KERNEL zone can call ExecutionKernel methods directly.
  Every entry is capability-gated, signed, and audited.
"""

from enum import Enum
from typing import Set


class Zone(Enum):
    KERNEL = "kernel"
    APPLICATION = "application"
    EXTERNAL = "external"


class Permission(Enum):
    DISPATCH = "dispatch"
    LEASE_READ = "lease_read"
    JOURNAL_READ = "journal_read"
    SNAPSHOT_READ = "snapshot_read"
    ADMIN = "admin"
    CAPABILITY_MINT = "capability_mint"


# Permission matrix: what each zone is allowed to do
ZONE_PERMISSIONS: dict[Zone, Set[Permission]] = {
    Zone.KERNEL: {
        Permission.DISPATCH,
        Permission.LEASE_READ,
        Permission.JOURNAL_READ,
        Permission.SNAPSHOT_READ,
        Permission.ADMIN,
        Permission.CAPABILITY_MINT,
    },
    Zone.APPLICATION: {
        Permission.DISPATCH,
    },
    Zone.EXTERNAL: set(),  # External never calls kernel directly
}


def validate_zone_permission(zone: Zone, permission: Permission) -> bool:
    """Check if a zone has a given permission."""
    return permission in ZONE_PERMISSIONS.get(zone, set())
