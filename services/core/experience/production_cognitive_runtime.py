"""
Production Cognitive Runtime - Immutable Event-Sourced Cognitive System.

Key invariants:
- Commands → Events → Reducers → State
- State is immutable (returns new instances)
- Replay is pure reducer application
- All mutations go through command handler

This is the canonical production-ready runtime.
"""
from typing import Dict, Any, Optional, List, Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import threading
import hashlib
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from cognitive_state import (
    CognitiveState, 
    initial_state, 
    BeliefState,
    CausalEdgeState,
    ContradictionState
)
from event_sourcing.events import CognitiveEvent, EventTypes, StreamIds, SchemaVersion
from event_sourcing.commands import Command, AddBelief, UpdateBelief, RemoveBelief
from event_sourcing.command_handler import CommandHandler, CommandResult
from event_sourcing.event_store import PersistentEventStore, OptimisticEventStore
from event_sourcing.reducers import reduce, reduce_sequence, EVENT_REDUCERS
from event_sourcing.projections import (
    Projection, 
    BeliefProjection, 
    ContradictionProjection,
    TransactionProjection,
    ProjectionManager
)
from event_sourcing.policies import PolicyEngine, PolicyResult


class CognitiveRuntimeError(Exception):
    """Runtime error"""
    pass


class ReplayVerificationError(Exception):
    """Replay verification failed"""
    pass


@dataclass
class RuntimeMetrics:
    """Runtime metrics"""
    event_count: int = 0
    command_count: int = 0
    projection_count: int = 0
    last_event_position: int = 0
    state_hash: str = ""


class ProductionCognitiveRuntime:
    """
    Production event-sourced cognitive runtime.
    
    Architecture:
    
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │   Command   │────▶│  Policy      │────▶│   Event     │
    │   (Intent)  │     │  Evaluation  │     │  Creation   │
    └─────────────┘     └──────────────┘     └──────┬──────┘
                                                    │
                                                    ▼
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  Projection │◀────│   Reducer    │◀────│   Event     │
    │   (Read)    │     │  (Pure Fn)   │     │   Store     │
    └─────────────┘     └──────────────┘     └─────────────┘
           │
           ▼
    ┌─────────────┐
    │   State     │ (Immutable)
    └─────────────┘
    
    Key features:
    - Immutable state (returns new instances)
    - Optimistic concurrency control
    - Incremental projections
    - Deterministic replay
    - Schema versioning
    """
    
    def __init__(
        self, 
        store_path: str = ":memory:",
        enable_projections: bool = True,
        enable_snapshotting: bool = True,
        snapshot_interval: int = 100
    ):
        self._store = OptimisticEventStore(store_path) if store_path != ":memory:" else PersistentEventStore()
        self._handler = CommandHandler()
        self._state: Optional[CognitiveState] = initial_state()
        self._lock = threading.RLock()
        self._metrics = RuntimeMetrics()
        self._enable_projections = enable_projections
        self._enable_snapshotting = enable_snapshotting
        self._snapshot_interval = snapshot_interval
        
        self._projection_manager = ProjectionManager()
        if enable_projections:
            self._init_projections()
        
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._replay_verification_enabled = True
    
    def _init_projections(self):
        """Initialize projections"""
        self._projection_manager.register(BeliefProjection())
        self._projection_manager.register(ContradictionProjection())
        self._projection_manager.register(TransactionProjection())
        self._metrics.projection_count = 3
    
    def execute_command(self, command: Command) -> CommandResult:
        """
        Execute command: validate → policy check → create event → store → reduce.
        
        Returns CommandResult with event if successful.
        """
        with self._lock:
            result = self._handler.handle(command, self._state_to_dict())
            
            if not result.success:
                return result
            
            event = result.event
            if event is None:
                return CommandResult(success=False, error="No event created")
            
            stored_event = self._store.append_with_retry(
                stream_id=command.stream_id,
                event=event
            )
            
            self._state = reduce(self._state, stored_event.to_dict())
            self._metrics.event_count += 1
            self._metrics.command_count += 1
            self._metrics.last_event_position = stored_event.position
            self._metrics.state_hash = CognitiveState.compute_hash(self._state)
            
            if self._enable_projections:
                self._projection_manager.project(stored_event.to_dict())
            
            self._emit_event_handlers(stored_event)
            
            if self._enable_snapshotting and self._metrics.event_count % self._snapshot_interval == 0:
                self._save_snapshot()
            
            return CommandResult(
                success=True,
                event=stored_event,
                policy_result=result.policy_result
            )
    
    def execute_commands(self, commands: List[Command]) -> List[CommandResult]:
        """Execute batch of commands"""
        results = []
        
        with self._lock:
            for command in commands:
                result = self.execute_command(command)
                results.append(result)
                if not result.success:
                    break
        
        return results
    
    def get_state(self) -> CognitiveState:
        """Get current canonical state"""
        with self._lock:
            return self._state
    
    def get_state_hash(self) -> str:
        """Get current state hash"""
        with self._lock:
            return CognitiveState.compute_hash(self._state)
    
    def get_metrics(self) -> RuntimeMetrics:
        """Get runtime metrics"""
        with self._lock:
            return self._metrics
    
    def get_projection(self, name: str) -> Optional[Projection]:
        """Get projection by name"""
        return self._projection_manager.get_projection(name)
    
    def replay_stream(self, stream_id: str, from_position: int = 1) -> CognitiveState:
        """
        Replay stream from position to rebuild state.
        
        This is used for:
        - Recovery after restart
        - State verification
        - Projection rebuild
        """
        events = self._store.get_stream(stream_id, from_position)
        
        if not events:
            return self._state
        
        with self._lock:
            current = initial_state()
            
            for event in events:
                current = reduce(current, event.to_dict())
            
            return current
    
    def verify_deterministic_replay(self, stream_id: str) -> bool:
        """
        Verify that replay produces same state as current.
        
        This is the key invariant: replay(state0, events) == stateN
        """
        with self._lock:
            replayed = self.replay_stream(stream_id)
            replayed_hash = CognitiveState.compute_hash(replayed)
            current_hash = CognitiveState.compute_hash(self._state)
            
            if replayed_hash != current_hash:
                raise ReplayVerificationError(
                    f"Replay verification failed: "
                    f"current={current_hash}, replayed={replayed_hash}"
                )
            
            return True
    
    def verify_all_streams(self) -> Dict[str, bool]:
        """Verify all streams"""
        results = {}
        
        for stream in self._store.get_all_streams():
            try:
                results[stream.stream_id] = self.verify_deterministic_replay(stream.stream_id)
            except ReplayVerificationError:
                results[stream.stream_id] = False
        
        return results
    
    def subscribe(self, event_type: str, handler: Callable[[CognitiveEvent], None]):
        """Subscribe to event type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def _emit_event_handlers(self, event: CognitiveEvent):
        """Emit to event handlers"""
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                pass
    
    def _state_to_dict(self) -> Dict[str, Any]:
        """Convert state to dict for policy evaluation"""
        with self._lock:
            return {
                "beliefs": {k: v for k, v in self._state._beliefs.items()} if hasattr(self._state, '_beliefs') else {},
                "contradictions": {k: v for k, v in self._state._contradictions.items()} if hasattr(self._state, '_contradictions') else {},
                "transactions": {k: v for k, v in self._state._transactions.items()} if hasattr(self._state, '_transactions') else {},
            }
    
    def _save_snapshot(self):
        """Save state snapshot"""
        with self._lock:
            self._store.save_snapshot(
                StreamIds.BELIEF,
                self._metrics.last_event_position,
                {
                    "state_hash": self._metrics.state_hash,
                    "event_count": self._metrics.event_count
                }
            )
    
    def recover_from_snapshot(self, stream_id: str = StreamIds.BELIEF) -> bool:
        """Recover state from snapshot"""
        snapshot = self._store.get_snapshot(stream_id)
        
        if not snapshot:
            return False
        
        with self._lock:
            self._state = self.replay_stream(stream_id)
            return True
    
    def close(self):
        """Close runtime"""
        self._store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    def get_stream_info(self) -> List[Dict[str, Any]]:
        """Get info about all streams"""
        streams = self._store.get_all_streams()
        return [
            {
                "stream_id": s.stream_id,
                "version": s.version,
                "last_updated": s.last_updated
            }
            for s in streams
        ]
    
    def export_events(self, stream_id: str, from_position: int = 1) -> List[Dict[str, Any]]:
        """Export events for external processing"""
        events = self._store.get_stream(stream_id, from_position)
        return [e.to_dict() for e in events]
    
    def import_events(self, events: List[Dict[str, Any]]) -> int:
        """Import events from external source"""
        imported = 0
        
        for event_data in events:
            event = CognitiveEvent.from_dict(event_data)
            try:
                self._store.append(event.stream_id, event)
                imported += 1
            except Exception:
                pass
        
        return imported


def create_runtime(
    db_path: str = "cognitive_events.db",
    **kwargs
) -> ProductionCognitiveRuntime:
    """Factory function to create runtime"""
    return ProductionCognitiveRuntime(store_path=db_path, **kwargs)