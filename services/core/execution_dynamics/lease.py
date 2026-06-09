"""
Execution Lease — kernel-issued authority token for goal execution.

An execution lease is proof that the ExecutionKernel has authorized
a specific execution. No executor can run without a valid lease.

PURPOSE:
  - Kernel is the ONLY authority that can issue leases
  - No code path can execute a goal without going through kernel.dispatch()
  - Leases expire, preventing orphaned executions
  - Leases carry dispatch_epoch for ordering and replay

LEASE LIFECYCLE:
  1. ISSUED — kernel.dispatch() issues a lease
  2. VALIDATED — executor checks lease before starting
  3. COMPLETED — execution finished, lease is closed
  4. EXPIRED — lease TTL exceeded without completion
  5. REVOKED — kernel explicitly revoked the lease (preemption, cancellation)

STORAGE:
  - In-memory for fast validation
  - Postgres for durability (lease survives restart)
  - Redis is NOT used for lease state
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import logging
import time
import uuid

logger = logging.getLogger(__name__)


# ============================================================================
# Lease States
# ============================================================================

LEASE_ACTIVE = 'active'
LEASE_CANCELLING = 'cancelling'
LEASE_COMPLETED = 'completed'
LEASE_EXPIRED = 'expired'
LEASE_REVOKED = 'revoked'
LEASE_ABANDONED = 'abandoned'

LEASE_TERMINAL_STATES = {LEASE_COMPLETED, LEASE_EXPIRED, LEASE_REVOKED, LEASE_ABANDONED}


# ============================================================================
# Execution Lease
# ============================================================================

@dataclass
class ExecutionLease:
    """
    Kernel-issued authority token.

    A lease proves that kernel.dispatch() authorized this execution.
    No executor should run without checking a valid lease.

    Fields:
      - lease_id: unique identifier for this lease
      - goal_id: which goal this lease authorizes
      - execution_id: correlation ID for this execution attempt
      - dispatch_epoch: monotonically increasing epoch (for ordering)
      - group_id: anti-fragmentation group (if any)
      - state: current lease state (active/terminal)
      - issued_at: when the lease was issued
      - expires_at: when the lease expires (TTL-based)
      - last_heartbeat: when the last keepalive was received
      - heartbeat_count: total heartbeats received
      - issued_by: which component issued the lease
    """
    lease_id: str
    goal_id: str
    execution_id: str
    dispatch_epoch: int
    group_id: Optional[str] = None
    state: str = LEASE_ACTIVE
    issued_at: float = 0.0
    expires_at: float = 0.0
    last_heartbeat: float = 0.0
    heartbeat_count: int = 0
    issued_by: str = "execution_kernel"
    owner_id: str = ""  # worker/host that owns this lease
    cancellation_requested: bool = False  # cooperative cancellation flag

    def is_valid(self) -> bool:
        """Check if lease is currently valid for execution."""
        if self.state not in (LEASE_ACTIVE, LEASE_CANCELLING):
            return False
        if time.time() > self.expires_at:
            return False
        return True

    def was_cancelled(self) -> bool:
        """Check if cooperative cancellation was requested."""
        return self.cancellation_requested

    def has_expired(self) -> bool:
        return time.time() > self.expires_at

    def seconds_since_heartbeat(self) -> float:
        """Seconds since last heartbeat. Returns large number if never heartbeated."""
        if self.last_heartbeat == 0:
            return 999999.0
        return time.time() - self.last_heartbeat

    def to_dict(self) -> dict:
        return {
            'lease_id': self.lease_id,
            'goal_id': self.goal_id,
            'execution_id': self.execution_id,
            'dispatch_epoch': self.dispatch_epoch,
            'group_id': self.group_id,
            'state': self.state,
            'issued_at': self.issued_at,
            'expires_at': self.expires_at,
            'last_heartbeat': self.last_heartbeat,
            'heartbeat_count': self.heartbeat_count,
            'issued_by': self.issued_by,
            'owner_id': self.owner_id,
            'cancellation_requested': self.cancellation_requested,
            'is_valid': self.is_valid(),
            'seconds_since_heartbeat': self.seconds_since_heartbeat(),
        }


# ============================================================================
# Lease Registry
# ============================================================================

class LeaseRegistry:
    """
    Central authority for execution lease lifecycle.

    PROPERTIES:
      - Kernel is the ONLY issuer (via acquire())
      - Executors check leases (via validate())
      - Leases have TTL and auto-expire
      - Leases can be revoked (preemption, cancellation)

    THREAD SAFETY:
      Not yet — single-threaded for now.
      Add threading.Lock when concurrency is needed.
    """

    def __init__(self, db_session=None, default_ttl: int = 3600):
        self._db = db_session
        self._default_ttl = default_ttl
        self._leases: Dict[str, ExecutionLease] = {}
        self._goal_to_lease: Dict[str, str] = {}  # goal_id -> latest lease_id
        self._epoch = 0

    # ------------------------------------------------------------------
    # Issue (kernel only)
    # ------------------------------------------------------------------

    def acquire(
        self,
        goal_id: str,
        execution_id: str,
        group_id: Optional[str] = None,
        ttl: Optional[int] = None,
        issued_by: str = "execution_kernel",
        owner_id: str = "",
    ) -> ExecutionLease:
        """
        Issue a new execution lease.

        Called ONLY by ExecutionKernel.dispatch().
        owner_id identifies the worker/host that will execute.
        """
        self._epoch += 1
        lease_id = str(uuid.uuid4())
        now = time.time()
        ttl_sec = ttl or self._default_ttl

        lease = ExecutionLease(
            lease_id=lease_id,
            goal_id=goal_id,
            execution_id=execution_id,
            dispatch_epoch=self._epoch,
            group_id=group_id,
            state=LEASE_ACTIVE,
            issued_at=now,
            expires_at=now + ttl_sec,
            issued_by=issued_by,
            owner_id=owner_id,
        )

        self._leases[lease_id] = lease
        self._goal_to_lease[goal_id] = lease_id

        # Persist to DB if available
        if self._db:
            self._persist_lease(lease)

        return lease

    # ------------------------------------------------------------------
    # Validation (executor checks)
    # ------------------------------------------------------------------

    def validate(self, lease_id: str, owner_id: str = "") -> bool:
        """
        Validate an execution lease.

        Returns True if the lease is active and not expired.
        If owner_id is provided, also checks ownership.
        Called by executor before running.

        If expired, marks it as EXPIRED.
        """
        lease = self._leases.get(lease_id)
        if not lease:
            logger.warning(f"lease_not_found lease_id={lease_id}")
            return False

        if lease.state != LEASE_ACTIVE:
            logger.warning(f"lease_invalid_state lease_id={lease_id} state={lease.state}")
            return False

        if lease.has_expired():
            lease.state = LEASE_EXPIRED
            logger.warning(f"lease_expired lease_id={lease_id}")
            return False

        # Ownership check: only the assigned worker can use this lease
        if owner_id and lease.owner_id and lease.owner_id != owner_id:
            logger.warning(
                f"lease_owner_mismatch lease_id={lease_id} "
                f"expected={lease.owner_id} got={owner_id}"
            )
            return False

        return True

    def get_lease(self, lease_id: str) -> Optional[ExecutionLease]:
        """Get lease by ID."""
        return self._leases.get(lease_id)

    def get_active_lease(self, goal_id: str) -> Optional[ExecutionLease]:
        """Get the latest active or cancelling lease for a goal."""
        lease_id = self._goal_to_lease.get(goal_id)
        if not lease_id:
            return None
        lease = self._leases.get(lease_id)
        if lease and lease.state in (LEASE_ACTIVE, LEASE_CANCELLING):
            return lease
        return None

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def complete(self, lease_id: str):
        """Mark lease as completed (success or terminal failure)."""
        lease = self._leases.get(lease_id)
        if lease:
            lease.state = LEASE_COMPLETED
            if self._db:
                self._persist_state(lease_id, LEASE_COMPLETED)

    def revoke(self, lease_id: str):
        """Revoke a lease (preemption, cancellation)."""
        lease = self._leases.get(lease_id)
        if lease:
            lease.state = LEASE_REVOKED
            if self._db:
                self._persist_state(lease_id, LEASE_REVOKED)

    def abandon(self, lease_id: str):
        """Mark lease as abandoned (max retries, unrecoverable)."""
        lease = self._leases.get(lease_id)
        if lease:
            lease.state = LEASE_ABANDONED
            if self._db:
                self._persist_state(lease_id, LEASE_ABANDONED)

    def request_cancellation(self, lease_id: str) -> bool:
        """
        Request cooperative cancellation of an active lease.

        Sets state to CANCELLING and raises the cancellation flag.
        The executor should check was_cancelled() and clean up.
        Returns True if cancellation was requested.
        """
        lease = self._leases.get(lease_id)
        if not lease or lease.state not in (LEASE_ACTIVE, LEASE_CANCELLING):
            return False
        lease.state = LEASE_CANCELLING
        lease.cancellation_requested = True
        logger.info(f"cancellation_requested lease_id={lease_id} goal_id={lease.goal_id}")
        return True

    def confirm_cancellation(self, lease_id: str) -> bool:
        """
        Confirm cancellation after executor has cleaned up.

        Transitions from CANCELLING to CANCELLED (terminal).
        Returns True if confirmed.
        """
        lease = self._leases.get(lease_id)
        if not lease or lease.state != LEASE_CANCELLING:
            return False
        lease.state = LEASE_REVOKED
        lease.cancellation_requested = True
        if self._db:
            self._persist_state(lease_id, LEASE_REVOKED)
        logger.info(f"cancellation_confirmed lease_id={lease_id} goal_id={lease.goal_id}")
        return True

    # ------------------------------------------------------------------
    # Heartbeat & Renewal
    # ------------------------------------------------------------------

    def heartbeat(self, lease_id: str) -> bool:
        """
        Record a heartbeat for an active lease.

        Extends expires_at by default_ttl from now.
        Returns True if heartbeat was accepted, False if lease is not active.

        Called periodically by the executor to prove it's still alive.
        """
        lease = self._leases.get(lease_id)
        if not lease or lease.state != LEASE_ACTIVE:
            return False

        now = time.time()
        lease.last_heartbeat = now
        lease.heartbeat_count += 1
        lease.expires_at = now + self._default_ttl
        return True

    def renew(self, lease_id: str, ttl: Optional[int] = None) -> bool:
        """
        Explicitly renew a lease with a new TTL.

        Unlike heartbeat (which uses default_ttl), renew allows
        the caller to specify a custom TTL for the extension.

        Returns True if renewal succeeded.
        """
        lease = self._leases.get(lease_id)
        if not lease or lease.state != LEASE_ACTIVE:
            return False

        now = time.time()
        lease.last_heartbeat = now
        lease.heartbeat_count += 1
        lease.expires_at = now + (ttl or self._default_ttl)
        return True

    def detect_stale(self, idle_threshold: int = 300) -> List[dict]:
        """
        Find active leases without recent heartbeat.

        A lease is stale if:
          - It is in ACTIVE state
          - It has never received a heartbeat
          - OR its last heartbeat was more than idle_threshold seconds ago

        Stale leases are candidates for revocation or recovery.

        Returns list of stale lease info dicts (sorted by stalest first).
        """
        now = time.time()
        stale = []

        for lease in self._leases.values():
            if lease.state != LEASE_ACTIVE:
                continue

            if lease.last_heartbeat == 0:
                # Never heartbeated — check if issued recently
                age = now - lease.issued_at
                if age > idle_threshold:
                    stale.append({
                        'lease_id': lease.lease_id,
                        'goal_id': lease.goal_id,
                        'owner_id': lease.owner_id,
                        'reason': 'never_heartbeated',
                        'idle_seconds': round(age, 1),
                        'issued_at': lease.issued_at,
                    })
            else:
                idle = now - lease.last_heartbeat
                if idle > idle_threshold:
                    stale.append({
                        'lease_id': lease.lease_id,
                        'goal_id': lease.goal_id,
                        'owner_id': lease.owner_id,
                        'reason': 'heartbeat_timeout',
                        'idle_seconds': round(idle, 1),
                        'last_heartbeat': lease.last_heartbeat,
                        'heartbeat_count': lease.heartbeat_count,
                    })

        return sorted(stale, key=lambda s: -s['idle_seconds'])

    # ------------------------------------------------------------------
    # Owner queries
    # ------------------------------------------------------------------

    def get_leases_by_owner(self, owner_id: str) -> List[ExecutionLease]:
        """Get all active leases for a given worker/host."""
        return [
            lease for lease in self._leases.values()
            if lease.owner_id == owner_id and lease.state == LEASE_ACTIVE
        ]

    def release_owner(self, owner_id: str) -> int:
        """Revoke all active leases for a worker/host. Returns count."""
        count = 0
        for lease in self.get_leases_by_owner(owner_id):
            lease.state = LEASE_REVOKED
            count += 1
        if count:
            logger.info(f"owner_leases_revoked owner_id={owner_id} count={count}")
        return count

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def expire_stale(self, max_age_seconds: int = 7200):
        """Mark leases older than max_age as expired."""
        now = time.time()
        count = 0
        for lease in self._leases.values():
            if lease.state == LEASE_ACTIVE and (now - lease.issued_at) > max_age_seconds:
                lease.state = LEASE_EXPIRED
                count += 1
        if count:
            logger.info(f"leases_expired count={count}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_lease(self, lease: ExecutionLease):
        """Persist lease to PostgreSQL."""
        try:
            from models import ExecutionLease as DBLease
            db_lease = DBLease(
                lease_id=lease.lease_id,
                goal_id=lease.goal_id,
                execution_id=lease.execution_id,
                dispatch_epoch=lease.dispatch_epoch,
                group_id=lease.group_id,
                state=lease.state,
                issued_at=lease.issued_at,
                expires_at=lease.expires_at,
                issued_by=lease.issued_by,
            )
            self._db.add(db_lease)
            self._db.flush()
        except Exception as e:
            logger.warning(f"lease_persist_failed lease_id={lease.lease_id} error={e}")

    def _persist_state(self, lease_id: str, state: str):
        """Persist lease state change to PostgreSQL."""
        try:
            from models import ExecutionLease as DBLease
            db_lease = self._db.query(DBLease).filter(DBLease.lease_id == lease_id).first()
            if db_lease:
                db_lease.state = state
                self._db.flush()
        except Exception as e:
            logger.warning(f"lease_state_persist_failed lease_id={lease_id} error={e}")

    # ------------------------------------------------------------------
    # WAL-based recovery (atomic journal+lease)
    # ------------------------------------------------------------------

    def recover_from_wal(self, entries: List['JournalEntry']):
        """
        Rebuild in-memory lease state from WAL-recovered journal entries.

        The WAL is the atomic source of truth for both journal entries
        and lease state. This method scans recovered entries for embedded
        lease_state and rehydrates active leases.

        Called AFTER journal.recover_from_wal() on kernel startup.
        """
        from .journal import JournalEntry as JE
        active: Dict[str, ExecutionLease] = {}

        for entry in entries:
            if not isinstance(entry, JE):
                continue
            ls = entry.lease_state
            if not ls:
                continue

            # Reconstruct lease from WAL snapshot
            lease = ExecutionLease(
                lease_id=ls.get('lease_id', entry.lease_id),
                goal_id=ls.get('goal_id', entry.goal_id),
                execution_id=ls.get('execution_id', entry.execution_id),
                dispatch_epoch=ls.get('dispatch_epoch', entry.dispatch_epoch),
                group_id=ls.get('group_id'),
                state=ls.get('state', 'active'),
                issued_at=ls.get('issued_at', entry.timestamp),
                expires_at=ls.get('expires_at', entry.timestamp + self._default_ttl),
                issued_by=ls.get('issued_by', 'wal_recovery'),
                owner_id=ls.get('owner_id', ''),
                heartbeat_count=ls.get('heartbeat_count', 0),
                last_heartbeat=ls.get('last_heartbeat', 0.0),
                cancellation_requested=ls.get('cancellation_requested', False),
            )

            # Only recover non-terminal leases
            if lease.state not in LEASE_TERMINAL_STATES:
                # Last entry wins for the same goal
                active[lease.goal_id] = lease

        # Restore leases to in-memory registry
        for goal_id, lease in active.items():
            self._leases[lease.lease_id] = lease
            self._goal_to_lease[goal_id] = lease.lease_id
            if lease.dispatch_epoch > self._epoch:
                self._epoch = lease.dispatch_epoch

        if active:
            logger.info(f"lease_registry_recovered_from_wal n_leases={len(active)}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_active_leases(self) -> List[ExecutionLease]:
        """Get all currently active leases."""
        return [l for l in self._leases.values() if l.state == LEASE_ACTIVE]

    def get_stats(self) -> dict:
        if not self._leases:
            return {
                'total_leases': 0,
                'active': 0,
                'completed': 0,
                'expired': 0,
                'revoked': 0,
                'epoch': self._epoch,
            }

        states = {}
        for l in self._leases.values():
            states[l.state] = states.get(l.state, 0) + 1

        return {
            'total_leases': len(self._leases),
            'active': states.get(LEASE_ACTIVE, 0),
            'completed': states.get(LEASE_COMPLETED, 0),
            'expired': states.get(LEASE_EXPIRED, 0),
            'revoked': states.get(LEASE_REVOKED, 0),
            'abandoned': states.get(LEASE_ABANDONED, 0),
            'epoch': self._epoch,
        }
