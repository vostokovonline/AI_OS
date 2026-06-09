"""
Invariant Engine - Automatic replay verification and consistency checking.

Key invariants:
1. Deterministic replay: replay(state0, events) == stateN
2. Event order integrity: positions are monotonic per stream
3. Causation chain validity: all causation_ids exist
4. Schema compatibility: all events have valid schema_version
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from cognitive_state import CognitiveState, initial_state
from event_sourcing.event_store import PersistentEventStore
from event_sourcing.reducers import reduce, reduce_sequence


class InvariantType(Enum):
    """Invariant types"""
    DETERMINISTIC_REPLAY = "deterministic_replay"
    EVENT_ORDER = "event_order"
    CAUSATION_CHAIN = "causation_chain"
    SCHEMA_COMPATIBILITY = "schema_compatibility"
    IDEMPOTENCY = "idempotency"
    CONCURRENCY = "concurrency"


@dataclass
class InvariantViolation:
    """Invariant violation report"""
    invariant_type: InvariantType
    stream_id: str
    event_id: str
    position: int
    description: str
    severity: str
    timestamp: str


@dataclass
class InvariantResult:
    """Result of invariant check"""
    passed: bool
    violations: List[InvariantViolation]
    duration_ms: float
    events_checked: int


class InvariantEngine:
    """
    Automatic invariant verification engine.
    
    Checks invariants after every mutation or on demand.
    Can be configured to fail-fast or collect violations.
    """
    
    def __init__(
        self,
        store: PersistentEventStore,
        fail_fast: bool = False,
        auto_verify_on_commit: bool = True
    ):
        self._store = store
        self._fail_fast = fail_fast
        self._auto_verify = auto_verify_on_commit
        self._violations: List[InvariantViolation] = []
        self._check_handlers: Dict[InvariantType, Callable] = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default invariant checks"""
        self._check_handlers[InvariantType.DETERMINISTIC_REPLAY] = self._check_deterministic_replay
        self._check_handlers[InvariantType.EVENT_ORDER] = self._check_event_order
        self._check_handlers[InvariantType.CAUSATION_CHAIN] = self._check_causation_chain
        self._check_handlers[InvariantType.SCHEMA_COMPATIBILITY] = self._check_schema_compatibility
        self._check_handlers[InvariantType.IDEMPOTENCY] = self._check_idempotency
    
    def verify_all(self, stream_id: str) -> InvariantResult:
        """Run all invariant checks on stream"""
        start = datetime.utcnow()
        violations = []
        
        for invariant_type, handler in self._check_handlers.items():
            try:
                result = handler(stream_id)
                if result:
                    violations.extend(result)
            except Exception as e:
                violations.append(InvariantViolation(
                    invariant_type=invariant_type,
                    stream_id=stream_id,
                    event_id="",
                    position=0,
                    description=f"Check failed with exception: {str(e)}",
                    severity="error",
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        duration = (datetime.utcnow() - start).total_seconds() * 1000
        events = self._store.get_stream(stream_id)
        
        result = InvariantResult(
            passed=len(violations) == 0,
            violations=violations,
            duration_ms=duration,
            events_checked=len(events)
        )
        
        if not result.passed and self._fail_fast:
            raise InvariantViolationError(result)
        
        return result
    
    def verify_invariant(
        self, 
        invariant_type: InvariantType, 
        stream_id: str
    ) -> List[InvariantViolation]:
        """Run specific invariant check"""
        handler = self._check_handlers.get(invariant_type)
        if not handler:
            return []
        return handler(stream_id) or []
    
    def _check_deterministic_replay(self, stream_id: str) -> List[InvariantViolation]:
        """Check: replay produces same state"""
        violations = []
        
        events = self._store.get_stream(stream_id)
        if not events:
            return []
        
        state = initial_state()
        for event in events:
            state = reduce(state, event.to_dict())
        
        final_hash = CognitiveState.compute_hash(state)
        
        snapshot = self._store.get_snapshot(stream_id)
        if snapshot:
            expected_hash = snapshot.get("state_hash", "")
            if final_hash != expected_hash:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.DETERMINISTIC_REPLAY,
                    stream_id=stream_id,
                    event_id="",
                    position=0,
                    description=f"State hash mismatch: expected {expected_hash}, got {final_hash}",
                    severity="critical",
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        return violations
    
    def _check_event_order(self, stream_id: str) -> List[InvariantViolation]:
        """Check: positions are monotonically increasing"""
        violations = []
        
        events = self._store.get_stream(stream_id)
        last_position = 0
        
        for event in events:
            if event.position <= last_position:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.EVENT_ORDER,
                    stream_id=stream_id,
                    event_id=event.event_id,
                    position=event.position,
                    description=f"Position not monotonic: {event.position} <= {last_position}",
                    severity="critical",
                    timestamp=datetime.utcnow().isoformat()
                ))
            last_position = event.position
        
        return violations
    
    def _check_causation_chain(self, stream_id: str) -> List[InvariantViolation]:
        """Check: all causation_ids refer to existing events"""
        violations = []
        
        events = self._store.get_stream(stream_id)
        event_ids = {e.event_id for e in events}
        
        for event in events:
            if event.causation_id and event.causation_id not in event_ids:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.CAUSATION_CHAIN,
                    stream_id=stream_id,
                    event_id=event.event_id,
                    position=event.position,
                    description=f"Causation chain broken: {event.causation_id} not found",
                    severity="high",
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        return violations
    
    def _check_schema_compatibility(self, stream_id: str) -> List[InvariantViolation]:
        """Check: all events have valid schema version"""
        violations = []
        max_schema_version = 1
        
        events = self._store.get_stream(stream_id)
        for event in events:
            if event.schema_version > max_schema_version:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.SCHEMA_COMPATIBILITY,
                    stream_id=stream_id,
                    event_id=event.event_id,
                    position=event.position,
                    description=f"Schema version {event.schema_version} > max {max_schema_version}",
                    severity="high",
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        return violations
    
    def _check_idempotency(self, stream_id: str) -> List[InvariantViolation]:
        """Check: no duplicate event_ids"""
        violations = []
        
        events = self._store.get_stream(stream_id)
        seen_ids = set()
        
        for event in events:
            if event.event_id in seen_ids:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.IDEMPOTENCY,
                    stream_id=stream_id,
                    event_id=event.event_id,
                    position=event.position,
                    description=f"Duplicate event_id: {event.event_id}",
                    severity="critical",
                    timestamp=datetime.utcnow().isoformat()
                ))
            seen_ids.add(event.event_id)
        
        return violations
    
    def get_violations(
        self, 
        since: Optional[datetime] = None,
        stream_id: Optional[str] = None
    ) -> List[InvariantViolation]:
        """Get recorded violations"""
        violations = self._violations
        
        if since:
            violations = [v for v in violations if v.timestamp >= since.isoformat()]
        
        if stream_id:
            violations = [v for v in violations if v.stream_id == stream_id]
        
        return violations
    
    def clear_violations(self):
        """Clear recorded violations"""
        self._violations = []


class InvariantViolationError(Exception):
    """Raised when invariant check fails in fail_fast mode"""
    
    def __init__(self, result: InvariantResult):
        self.result = result
        message = f"Invariant check failed: {len(result.violations)} violations\n"
        for v in result.violations[:5]:
            message += f"  - {v.invariant_type.value}: {v.description}\n"
        super().__init__(message)


def verify_replay_equivalence(
    store: PersistentEventStore,
    stream_id: str
) -> bool:
    """
    Standalone function to verify replay produces same state.
    
    This is the fundamental invariant: replay(state0, events) == stateN
    """
    events = store.get_stream(stream_id)
    if not events:
        return True
    
    state = initial_state()
    for event in events:
        state = reduce(state, event.to_dict())
    
    final_hash = CognitiveState.compute_hash(state)
    snapshot = store.get_snapshot(stream_id)
    
    if snapshot:
        expected_hash = snapshot.get("state_hash", "")
        return final_hash == expected_hash
    
    return True


def run_invariant_suite(store: PersistentEventStore) -> Dict[str, InvariantResult]:
    """Run full invariant suite on all streams"""
    engine = InvariantEngine(store)
    results = {}
    
    for stream in store.get_all_streams():
        results[stream.stream_id] = engine.verify_all(stream.stream_id)
    
    return results