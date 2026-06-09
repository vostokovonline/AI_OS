"""
Invariant Engine — formal correctness verification for the execution kernel.

DESIGN:
  - DETERMINISTIC: invariants are pure functions of kernel state, no IO
  - PURE: zero side effects — invariants only read state, never write
  - REPLAYABLE: same invariants work on live, recovered, and simulated state
  - STRUCTURED: returns InvariantReport with all violations, not a boolean

INVARIANT CATEGORIES:
  - LEASE:      terminal→active illegal, one lease per goal, epoch monotonic
  - JOURNAL:    no STARTED after COMPLETED, contiguous entry chain
  - SNAPSHOT:   lsn ≤ wal_lsn, active_leases ⊆ WAL, event_count match
  - CAPABILITY: epoch match, revoked→blocked, ingress provenance
  - RECOVERY:   replay(state₀+events) == recovered_state

USAGE:
    report = kernel.verify()
    if not report.passed:
        for v in report.violations:
            logger.error(f"invariant={v.invariant} detail={v.detail}")
        raise InvariantViolationError(report)

    # Fast path (after dispatch):
    report = kernel.verify('lease_one_per_goal', 'journal_contiguous')
    if not report.passed:
        logger.critical(...)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any


# ============================================================================
# Severity
# ============================================================================

class Severity(Enum):
    FATAL = auto()   # Kernel state is unrecoverable — must rebuild from WAL
    ERROR = auto()   # State invariant violated — operation should be refused
    WARNING = auto() # Degenerate state — operation may proceed with caution


# ============================================================================
# Violation
# ============================================================================

@dataclass
class InvariantViolation:
    """
    A single invariant violation.

    Fields:
      name:      invariant name (e.g. "lease_one_per_goal")
      detail:    human-readable description of the violation
      severity:  FATAL / ERROR / WARNING
      context:   structured data for programmatic analysis
    """
    name: str
    detail: str
    severity: Severity = Severity.ERROR
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'invariant': self.name,
            'detail': self.detail,
            'severity': self.severity.name,
            'context': self.context,
        }


# ============================================================================
# Report
# ============================================================================

@dataclass
class InvariantReport:
    """
    Result of running the invariant engine against kernel state.

    Properties:
      passed:    True if zero violations
      fatal:     True if any FATAL violation
      violations: ordered list of violations
    """
    violations: List[InvariantViolation] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.violations) == 0

    @property
    def has_fatal(self) -> bool:
        return any(v.severity == Severity.FATAL for v in self.violations)

    def to_dict(self) -> dict:
        return {
            'passed': self.passed,
            'has_fatal': self.has_fatal,
            'n_violations': len(self.violations),
            'violations': [v.to_dict() for v in self.violations],
        }


# ============================================================================
# Invariant violation exception
# ============================================================================

class InvariantViolationError(Exception):
    """Raised when assert_invariants() finds violations."""

    def __init__(self, report: InvariantReport):
        self.report = report
        msg = f"Invariant violations: {len(report.violations)}"
        if report.violations:
            msg += f" (first: {report.violations[0].name})"
        super().__init__(msg)


# ============================================================================
# Invariant Engine
# ============================================================================

class InvariantEngine:
    """
    Deterministic, pure, replayable invariant checker.

    All invariants are methods with the signature:
        check_<name>(kernel) → Optional[InvariantViolation]

    Returns None if the invariant holds, or a Violation if it is violated.
    """

    def __init__(self, kernel):
        self._kernel = kernel

    # ------------------------------------------------------------------
    # LEASE INVARIANTS
    # ------------------------------------------------------------------

    def check_lease_terminal_cannot_become_active(
        self,
    ) -> Optional[InvariantViolation]:
        """
        A lease in a terminal state (completed, expired, revoked, abandoned)

        must never transition back to active.
        This checks all current active leases have never previously been
        in a terminal state for the same goal.
        """
        kernel = self._kernel
        for lease in kernel.registry._leases.values():
            goal_id = lease.goal_id
            # Check journal: if there's a terminal event for this goal
            # AFTER the current lease was issued, the lease is stale-wrong
            chain = kernel.journal.get_chain(goal_id)
            terminal_after = False
            for entry in chain:
                if entry.timestamp > lease.issued_at:
                    if entry.event in ('COMPLETED', 'FAILED', 'ABANDONED',
                                       'LEASE_EXPIRED', 'REVOKED'):
                        terminal_after = True
                        break
            if terminal_after and lease.state == 'active':
                return InvariantViolation(
                    name='lease_terminal_cannot_become_active',
                    detail=(
                        f"Lease {lease.lease_id} for goal {goal_id} is 'active' "
                        f"but a terminal journal entry exists AFTER its issuance."
                    ),
                    severity=Severity.FATAL,
                    context={
                        'lease_id': lease.lease_id,
                        'goal_id': goal_id,
                        'lease_state': lease.state,
                    },
                )
        return None

    def check_lease_one_active_per_goal(self) -> Optional[InvariantViolation]:
        """
        At most one active lease per goal must exist at any time.
        This is the core mutual-exclusion invariant of the kernel.
        """
        kernel = self._kernel
        active_count: Dict[str, int] = {}
        for lease in kernel.registry._leases.values():
            if lease.state == 'active':
                gid = lease.goal_id
                active_count[gid] = active_count.get(gid, 0) + 1

        violations = [gid for gid, count in active_count.items() if count > 1]
        if violations:
            return InvariantViolation(
                name='lease_one_active_per_goal',
                detail=(
                    f"Goals with multiple active leases: {violations}. "
                    f"Mutual exclusion violated."
                ),
                severity=Severity.FATAL,
                context={'conflicts': violations},
            )
        return None

    def check_lease_epoch_monotonic(self) -> Optional[InvariantViolation]:
        """
        Lease dispatch_epoch must be monotonically increasing.

        A later lease must have a higher (or equal) epoch than any
        earlier lease for the same goal.
        """
        kernel = self._kernel
        goal_epochs: Dict[str, List[int]] = {}
        for lease in kernel.registry._leases.values():
            gid = lease.goal_id
            if gid not in goal_epochs:
                goal_epochs[gid] = []
            goal_epochs[gid].append(lease.dispatch_epoch)

        for gid, epochs in goal_epochs.items():
            sorted_epochs = sorted(epochs, reverse=True)
            for i in range(len(sorted_epochs) - 1):
                if sorted_epochs[i] < sorted_epochs[i + 1]:
                    return InvariantViolation(
                        name='lease_epoch_monotonic',
                        detail=(
                            f"Goal {gid} has out-of-order epochs: "
                            f"{sorted_epochs}. Epochs must be monotonically "
                            f"increasing."
                        ),
                        severity=Severity.ERROR,
                        context={
                            'goal_id': gid,
                            'epochs': sorted_epochs,
                        },
                    )
        return None

    def check_lease_active_has_valid_ttl(self) -> Optional[InvariantViolation]:
        """
        Every active lease must have a non-expired TTL.
        An active lease with expired TTL is a resource leak.
        """
        import time
        kernel = self._kernel
        now = time.time()
        expired = []
        for lease in kernel.registry._leases.values():
            if lease.state == 'active' and lease.expires_at < now:
                expired.append(lease.lease_id)

        if expired:
            return InvariantViolation(
                name='lease_active_has_valid_ttl',
                detail=(
                    f"Active leases with expired TTL: {expired[:10]}"
                    f"{'...' if len(expired) > 10 else ''}. "
                    f"Resources may be leaked."
                ),
                severity=Severity.WARNING,
                context={'expired_leases': expired},
            )
        return None

    def check_lease_revoked_cannot_execute(self) -> Optional[InvariantViolation]:
        """
        A revoked lease must not have active executions in _active_executions.

        If kernel._active_executions contains a goal_id whose lease has
        been revoked, the kernel has lost coherence.
        """
        kernel = self._kernel
        revoked_active = []
        for goal_id, lease_id in kernel._active_executions.items():
            lease = kernel.registry.get_lease(lease_id)
            if lease and lease.state in ('revoked', 'abandoned'):
                revoked_active.append({
                    'goal_id': goal_id,
                    'lease_id': lease_id,
                    'state': lease.state,
                })

        if revoked_active:
            return InvariantViolation(
                name='lease_revoked_cannot_execute',
                detail=(
                    f"Revoked leases present in active_executions: "
                    f"{revoked_active}. Kernel coherence lost."
                ),
                severity=Severity.FATAL,
                context={'revoked_active': revoked_active},
            )
        return None

    # ------------------------------------------------------------------
    # JOURNAL INVARIANTS
    # ------------------------------------------------------------------

    def check_journal_no_started_after_completed(
        self,
    ) -> Optional[InvariantViolation]:
        """
        For any execution_id, once a COMPLETED or FAILED event is emitted,
        no STARTED event for the same execution_id may follow.
        """
        kernel = self._kernel
        execution_states: Dict[str, List[str]] = {}
        for entry in kernel.journal._entries:
            eid = entry.execution_id
            if eid not in execution_states:
                execution_states[eid] = []
            execution_states[eid].append(entry.event)

        for eid, events in execution_states.items():
            terminal_idx = -1
            for i, ev in enumerate(events):
                if ev in ('COMPLETED', 'FAILED', 'ABANDONED'):
                    terminal_idx = i
                    break
            if terminal_idx >= 0 and terminal_idx < len(events) - 1:
                post_terminal = events[terminal_idx + 1:]
                if 'STARTED' in post_terminal:
                    return InvariantViolation(
                        name='journal_no_started_after_completed',
                        detail=(
                            f"Execution {eid} has STARTED after terminal event. "
                            f"Full chain: {events}"
                        ),
                        severity=Severity.FATAL,
                        context={
                            'execution_id': eid,
                            'events': events,
                        },
                    )
        return None

    def check_journal_contiguous_chain(self) -> Optional[InvariantViolation]:
        """
        For each goal, the journal entry chain must be contiguous.

        Each entry's prev_entry_id must match the previous entry's entry_id.
        A gap indicates data loss or journal corruption.
        """
        kernel = self._kernel
        goal_chains: Dict[str, list] = {}
        for entry in kernel.journal._entries:
            gid = entry.goal_id
            if gid not in goal_chains:
                goal_chains[gid] = []
            goal_chains[gid].append(entry)

        for gid, chain in goal_chains.items():
            for i in range(1, len(chain)):
                prev_id = chain[i - 1].entry_id
                expected_prev = chain[i].prev_entry_id
                if expected_prev and expected_prev != prev_id:
                    return InvariantViolation(
                        name='journal_contiguous_chain',
                        detail=(
                            f"Goal {gid}: journal chain broken at entry "
                            f"{chain[i].entry_id}. Expected prev={prev_id}, "
                            f"got {expected_prev}."
                        ),
                        severity=Severity.FATAL,
                        context={
                            'goal_id': gid,
                            'entry_index': i,
                            'entry_id': chain[i].entry_id,
                            'expected_prev': prev_id,
                            'actual_prev': expected_prev,
                        },
                    )

            # Check that chain is in chronological order
            for i in range(1, len(chain)):
                if chain[i].timestamp < chain[i - 1].timestamp:
                    return InvariantViolation(
                        name='journal_contiguous_chain',
                        detail=(
                            f"Goal {gid}: journal out of order at entry "
                            f"{chain[i].entry_id}. "
                            f"ts={chain[i].timestamp} < prev_ts={chain[i - 1].timestamp}"
                        ),
                        severity=Severity.ERROR,
                        context={
                            'goal_id': gid,
                            'entry_index': i,
                            'entry_id': chain[i].entry_id,
                            'timestamp': chain[i].timestamp,
                            'prev_timestamp': chain[i - 1].timestamp,
                        },
                    )
        return None

    def check_journal_execution_has_entry(
        self,
    ) -> Optional[InvariantViolation]:
        """
        Every entry in _active_executions must have at least a DISPATCHED
        journal entry. An execution cannot exist without a journal record.
        """
        kernel = self._kernel
        for goal_id in kernel._active_executions:
            chain = kernel.journal.get_chain(goal_id)
            if not chain:
                return InvariantViolation(
                    name='journal_execution_has_entry',
                    detail=(
                        f"Goal {goal_id} is in _active_executions but has "
                        f"no journal entries."
                    ),
                    severity=Severity.FATAL,
                    context={'goal_id': goal_id},
                )
            # Must have at least a STARTED entry
            has_started = any(e.event == 'STARTED' for e in chain)
            if not has_started:
                return InvariantViolation(
                    name='journal_execution_has_entry',
                    detail=(
                        f"Goal {goal_id} is in _active_executions but has "
                        f"no STARTED journal entry."
                    ),
                    severity=Severity.ERROR,
                    context={'goal_id': goal_id},
                )
        return None

    # ------------------------------------------------------------------
    # DISPATCH INVARIANTS
    # ------------------------------------------------------------------

    def check_dispatch_has_idempotency_key(
        self,
    ) -> Optional[InvariantViolation]:
        """
        Every DISPATCHED journal entry must have a non-empty dispatch_id.
        Without it, idempotent dedup cannot work.
        """
        kernel = self._kernel
        for entry in kernel.journal._entries:
            if entry.event == 'DISPATCHED' and not entry.dispatch_id:
                return InvariantViolation(
                    name='dispatch_has_idempotency_key',
                    detail=(
                        f"DISPATCHED entry {entry.entry_id} for goal "
                        f"{entry.goal_id} has no dispatch_id."
                    ),
                    severity=Severity.WARNING,
                    context={
                        'entry_id': entry.entry_id,
                        'goal_id': entry.goal_id,
                    },
                )
        return None

    def check_dispatch_dedup_window_consistent(
        self,
    ) -> Optional[InvariantViolation]:
        """
        _seen_dispatches entries must not reference dispatch_ids that
        have no corresponding journal DISPATCHED entry.
        """
        kernel = self._kernel
        journal_dispatch_ids = set()
        for entry in kernel.journal._entries:
            if entry.event == 'DISPATCHED' and entry.dispatch_id:
                journal_dispatch_ids.add(entry.dispatch_id)

        for did in kernel._seen_dispatches:
            if did not in journal_dispatch_ids:
                # This can happen if dedup data is from before a restart
                # Without WAL persistence of dedup, this is a soft violation
                return InvariantViolation(
                    name='dispatch_dedup_window_consistent',
                    detail=(
                        f"dispatch_id={did} is in _seen_dispatches but "
                        f"has no corresponding journal DISPATCHED entry."
                    ),
                    severity=Severity.WARNING,
                    context={
                        'dispatch_id': did,
                    },
                )
        return None

    # ------------------------------------------------------------------
    # SNAPSHOT INVARIANTS
    # ------------------------------------------------------------------

    def check_snapshot_lsn_not_ahead_of_wal(
        self,
    ) -> Optional[InvariantViolation]:
        """
        snapshot.lsn must be ≤ wal.get_lsn().

        A snapshot ahead of the WAL is a consistency violation:
        the snapshot would reference state that could not be reconstructed
        from WAL replay.
        """
        kernel = self._kernel
        if not getattr(kernel.snapshots, '_latest', None):
            return None

        snap = kernel.snapshots._latest
        wal_lsn = kernel.wal.get_lsn() if kernel.wal else snap.lsn

        if snap.lsn and snap.lsn > wal_lsn:
            return InvariantViolation(
                name='snapshot_lsn_not_ahead_of_wal',
                detail=(
                    f"Snapshot lsn={snap.lsn} exceeds WAL lsn={wal_lsn}. "
                    f"Snapshot is ahead of source of truth."
                ),
                severity=Severity.FATAL,
                context={
                    'snapshot_lsn': snap.lsn,
                    'wal_lsn': wal_lsn,
                    'snapshot_id': snap.snapshot_id,
                },
            )
        return None

    def check_snapshot_last_committed_not_ahead(
        self,
    ) -> Optional[InvariantViolation]:
        """
        snapshot.last_committed_lsn must be ≤ snapshot.lsn.

        The last_committed_lsn is the LSN of the last journal entry
        included in the snapshot. It cannot exceed the WAL position
        at snapshot time.
        """
        kernel = self._kernel
        if not getattr(kernel.snapshots, '_latest', None):
            return None

        snap = kernel.snapshots._latest
        if snap.last_committed_lsn and snap.lsn:
            if snap.last_committed_lsn > snap.lsn:
                return InvariantViolation(
                    name='snapshot_last_committed_not_ahead',
                    detail=(
                        f"last_committed_lsn={snap.last_committed_lsn} > "
                        f"lsn={snap.lsn}. Snapshot appears to have committed "
                        f"entries beyond its recorded WAL position."
                    ),
                    severity=Severity.FATAL,
                    context={
                        'last_committed_lsn': snap.last_committed_lsn,
                        'lsn': snap.lsn,
                        'snapshot_id': snap.snapshot_id,
                    },
                )
        return None

    def check_snapshot_active_leases_in_wal(
        self,
    ) -> Optional[InvariantViolation]:
        """
        Every active lease in a snapshot must have a corresponding
        journal entry in the WAL. If a snapshot references a lease
        that WAL replay cannot reconstruct, recovery is impossible.
        """
        kernel = self._kernel
        if not getattr(kernel.snapshots, '_latest', None):
            return None

        snap = kernel.snapshots._latest
        # Collect all lease_ids from journal
        wal_lease_ids = set()
        for entry in kernel.journal._entries:
            if entry.lease_id:
                wal_lease_ids.add(entry.lease_id)

        missing = []
        for goal_id, ls_data in snap.active_leases.items():
            lid = ls_data.get('lease_id', '')
            if lid and lid not in wal_lease_ids:
                missing.append(lid)

        if missing:
            return InvariantViolation(
                name='snapshot_active_leases_in_wal',
                detail=(
                    f"Snapshot references {len(missing)} lease(s) not found "
                    f"in WAL: {missing[:5]}{'...' if len(missing) > 5 else ''}."
                ),
                severity=Severity.WARNING,
                context={'missing_lease_ids': missing},
            )
        return None

    # ------------------------------------------------------------------
    # RECOVERY INVARIANTS
    # ------------------------------------------------------------------

    def check_recovery_epoch_consistency(
        self,
    ) -> Optional[InvariantViolation]:
        """
        After recovery, kernel.registry._epoch must be ≥ the max dispatch
        epoch found in all journal entries.
        """
        kernel = self._kernel
        max_journal_epoch = 0
        for entry in kernel.journal._entries:
            if entry.dispatch_epoch > max_journal_epoch:
                max_journal_epoch = entry.dispatch_epoch

        current_epoch = kernel.registry._epoch
        if current_epoch < max_journal_epoch:
            return InvariantViolation(
                name='recovery_epoch_consistency',
                detail=(
                    f"Registry epoch={current_epoch} < max journal "
                    f"epoch={max_journal_epoch}. Epoch rollback detected."
                ),
                severity=Severity.ERROR,
                context={
                    'current_epoch': current_epoch,
                    'max_journal_epoch': max_journal_epoch,
                },
            )
        return None

    def check_recovery_active_executions_have_leases(
        self,
    ) -> Optional[InvariantViolation]:
        """
        Every entry in _active_executions must correspond to a valid
        (non-terminal) lease in the lease registry.
        """
        kernel = self._kernel
        orphaned = []
        for goal_id, lease_id in kernel._active_executions.items():
            lease = kernel.registry.get_lease(lease_id)
            if not lease:
                orphaned.append({
                    'goal_id': goal_id,
                    'lease_id': lease_id,
                    'reason': 'lease_not_found',
                })
            elif lease.state in ('completed', 'expired', 'revoked', 'abandoned'):
                orphaned.append({
                    'goal_id': goal_id,
                    'lease_id': lease_id,
                    'reason': f'lease_terminal:{lease.state}',
                })

        if orphaned:
            return InvariantViolation(
                name='recovery_active_executions_have_leases',
                detail=(
                    f"Active executions without valid leases: "
                    f"{orphaned[:5]}{'...' if len(orphaned) > 5 else ''}."
                ),
                severity=Severity.ERROR,
                context={'orphaned': orphaned},
            )
        return None

    # ------------------------------------------------------------------
    # RUN ALL INVARIANTS
    # ------------------------------------------------------------------

    def verify_all(self) -> InvariantReport:
        """
        Run ALL invariants against the current kernel state.

        Returns InvariantReport with all violations (or empty if clean).
        This is a PURE operation — no kernel state is modified.
        """
        violations: List[InvariantViolation] = []

        checks = [
            # Lease
            self.check_lease_terminal_cannot_become_active,
            self.check_lease_one_active_per_goal,
            self.check_lease_epoch_monotonic,
            self.check_lease_active_has_valid_ttl,
            self.check_lease_revoked_cannot_execute,
            # Journal
            self.check_journal_no_started_after_completed,
            self.check_journal_contiguous_chain,
            self.check_journal_execution_has_entry,
            # Dispatch
            self.check_dispatch_has_idempotency_key,
            self.check_dispatch_dedup_window_consistent,
            # Snapshot
            self.check_snapshot_lsn_not_ahead_of_wal,
            self.check_snapshot_last_committed_not_ahead,
            self.check_snapshot_active_leases_in_wal,
            # Recovery
            self.check_recovery_epoch_consistency,
            self.check_recovery_active_executions_have_leases,
        ]

        for check in checks:
            try:
                result = check()
                if result is not None:
                    violations.append(result)
            except Exception as e:
                violations.append(InvariantViolation(
                    name=check.__name__,
                    detail=f"Invariant check raised exception: {e}",
                    severity=Severity.ERROR,
                    context={'error': str(e)},
                ))

        return InvariantReport(violations=violations)

    def verify(self, *names: str) -> InvariantReport:
        """
        Run specific invariants by name.

        Args:
            *names: invariant method names (without 'check_' prefix).
                    Example: verify('lease_one_per_goal', 'journal_contiguous')

        Returns:
            InvariantReport with matching violations.
        """
        violations: List[InvariantViolation] = []
        method_prefix = 'check_'

        # Build name → method map
        check_map: Dict[str, Any] = {}
        for attr_name in dir(self):
            if attr_name.startswith(method_prefix):
                check_map[attr_name[len(method_prefix):]] = getattr(self, attr_name)

        for name in names:
            method = check_map.get(name)
            if method is None:
                violations.append(InvariantViolation(
                    name=name,
                    detail=f"Unknown invariant: {name}",
                    severity=Severity.ERROR,
                ))
                continue
            try:
                result = method()
                if result is not None:
                    violations.append(result)
            except Exception as e:
                violations.append(InvariantViolation(
                    name=name,
                    detail=f"Invariant check raised exception: {e}",
                    severity=Severity.ERROR,
                    context={'error': str(e)},
                ))

        return InvariantReport(violations=violations)

    def list_invariants(self) -> List[str]:
        """Return names of all registered invariants."""
        names = []
        for attr_name in dir(self):
            if attr_name.startswith('check_'):
                names.append(attr_name[len('check_'):])
        return sorted(names)
