"""
Event-Sourced Cognitive Runtime - Production Architecture

LAYERS:
- commands/     : Intent-oriented operations
- events/      : Immutable fact records  
- reducers/    : Pure state transitions (no side effects)
- projections/  : Read models from events
- event_store/  : Persistent append-only log
- adapters/     : Execution integrations
- transport/    : WebSocket, SSE, etc.
"""
from typing import Dict, List, Optional, Any, Callable, Protocol
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json
import sqlite3
import asyncio
from pathlib import Path


# =============================================================================
# COMMANDS (Intent-Oriented)
# =============================================================================

@dataclass
class Command:
    """Base command - intent, not fact"""
    command_id: str
    timestamp: str
    caused_by: Optional[str] = None  # correlation_id


@dataclass
class ExecuteGoal(Command):
    goal: Dict[str, Any]


@dataclass  
class MutateIdentity(Command):
    axis: str
    delta: float


@dataclass
class ProtectIdentityAxis(Command):
    axis: str


@dataclass
class EvolveGenome(Command):
    execution_result: Dict[str, Any]


@dataclass
class RaiseInterrupt(Command):
    interrupt_type: str
    severity: float


# =============================================================================
# EVENTS (Fact-Oriented)
# =============================================================================

@dataclass
class Event:
    """Base event - immutable fact"""
    event_id: str
    event_type: str
    stream_id: str
    position: int  # global ordering
    timestamp: str
    caused_by: Optional[str] = None
    payload: Dict = field(default_factory=dict)


# =============================================================================
# PERSISTENT EVENT STORE
# =============================================================================

class PersistentEventStore:
    """
    SQLite-based append-only event store.
    Features: causal ordering, snapshots, replay.
    """
    
    def __init__(self, db_path: str = "cognitive_events.db"):
        self._db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                stream_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                caused_by TEXT,
                payload TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_stream ON events(stream_id, position)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS snapshots (
                stream_id TEXT PRIMARY KEY,
                position INTEGER NOT NULL,
                state TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def append(self, event: Event):
        """Append event to store"""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            INSERT INTO events (event_id, event_type, stream_id, position, timestamp, caused_by, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event.event_id,
            event.event_type,
            event.stream_id,
            event.position,
            event.timestamp,
            event.caused_by,
            json.dumps(event.payload)
        ))
        conn.commit()
        conn.close()
    
    def get_stream(self, stream_id: str, after_position: int = 0) -> List[Event]:
        """Read stream from position"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("""
            SELECT event_id, event_type, stream_id, position, timestamp, caused_by, payload
            FROM events 
            WHERE stream_id = ? AND position > ?
            ORDER BY position
        """, (stream_id, after_position))
        
        events = []
        for row in cursor:
            events.append(Event(
                event_id=row[0],
                event_type=row[1],
                stream_id=row[2],
                position=row[3],
                timestamp=row[4],
                caused_by=row[5],
                payload=json.loads(row[6])
            ))
        conn.close()
        return events
    
    def get_all(self, after_position: int = 0, limit: int = 1000) -> List[Event]:
        """Read all events after position"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("""
            SELECT event_id, event_type, stream_id, position, timestamp, caused_by, payload
            FROM events 
            WHERE position > ?
            ORDER BY position
            LIMIT ?
        """, (after_position, limit))
        
        events = []
        for row in cursor:
            events.append(Event(
                event_id=row[0],
                event_type=row[1],
                stream_id=row[2],
                position=row[3],
                timestamp=row[4],
                caused_by=row[5],
                payload=json.loads(row[6])
            ))
        conn.close()
        return events
    
    def save_snapshot(self, stream_id: str, position: int, state: Dict):
        """Save snapshot for replay optimization"""
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            INSERT OR REPLACE INTO snapshots (stream_id, position, state, timestamp)
            VALUES (?, ?, ?, ?)
        """, (stream_id, position, json.dumps(state), datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
    
    def get_latest_snapshot(self, stream_id: str) -> Optional[tuple]:
        """Get latest snapshot for stream"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("""
            SELECT position, state, timestamp FROM snapshots
            WHERE stream_id = ?
            ORDER BY position DESC
            LIMIT 1
        """, (stream_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return (row[0], json.loads(row[1]), row[2])
        return None
    
    def get_global_position(self) -> int:
        """Get current global position"""
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT MAX(position) FROM events")
        row = cursor.fetchone()
        conn.close()
        return row[0] or 0


# =============================================================================
# PURE REDUCERS (No Side Effects)
# =============================================================================

# Identity Reducer
def reduce_identity(state: Optional[Dict], event: Event) -> Dict:
    """Pure reducer for identity state"""
    if state is None:
        state = {
            'axes': {
                'exploration': 0.5, 'stability': 0.5, 'autonomy': 0.5,
                'precision': 0.5, 'aggression': 0.5, 'reflection': 0.5, 'persistence': 0.5
            },
            'protected': [],
            'mutation_count': 0
        }
    
    if event.event_type == 'identity.axis.mutated':
        axis = event.payload['axis']
        delta = event.payload['delta']
        if axis not in state['protected']:
            old = state['axes'].get(axis, 0.5)
            state['axes'][axis] = max(0.1, min(0.9, old + delta))
        state['mutation_count'] += 1
    
    elif event.event_type == 'identity.axis.protected':
        axis = event.payload['axis']
        if axis not in state['protected']:
            state['protected'] = list(state['protected']) + [axis]
    
    return state


# Genome Reducer
def reduce_genome(state: Optional[Dict], event: Event) -> Dict:
    """Pure reducer for genome state"""
    if state is None:
        state = {
            'risk_tolerance': 0.5,
            'exploration_bias': 0.3,
            'validation_frequency': 2,
            'generation': 0
        }
    
    if event.event_type == 'genome.evolved':
        changes = event.payload.get('changes', {})
        state['generation'] = event.payload.get('new_generation', state['generation'] + 1)
        
        if 'risk_tolerance' in changes:
            state['risk_tolerance'] = max(0.1, min(0.9, state['risk_tolerance'] + changes['risk_tolerance']))
        if 'exploration_bias' in changes:
            state['exploration_bias'] = max(0.1, min(0.9, state['exploration_bias'] + changes['exploration_bias']))
        if 'validation_frequency' in changes:
            state['validation_frequency'] = max(1, min(10, state['validation_frequency'] + changes['validation_frequency']))
    
    return state


# Execution Lineage Reducer
def reduce_lineage(state: Optional[List], event: Event) -> List:
    """Pure reducer for execution lineage"""
    if state is None:
        state = []
    
    if event.event_type == 'execution.lineage':
        lineage_entry = {
            'lineage_id': event.payload.get('lineage_id'),
            'goal_id': event.payload.get('goal_id'),
            'strategy': event.payload.get('strategy'),
            'outcome': event.payload.get('outcome'),
            'interrupt': event.payload.get('interrupt', False),
            'genome_generation': event.payload.get('genome_generation'),
            'timestamp': event.timestamp
        }
        state = state + [lineage_entry]
        
        # Keep last 1000
        if len(state) > 1000:
            state = state[-1000:]
    
    return state


# =============================================================================
# PROJECTIONS (Read Models)
# =============================================================================

class IdentityProjection:
    """Read model for identity"""
    
    def __init__(self, event_store: PersistentEventStore):
        self._store = event_store
    
    def get_state(self, stream_id: str = "identity") -> Dict:
        """Get current identity state by replaying events"""
        # Check for snapshot first
        snapshot = self._store.get_latest_snapshot(stream_id)
        
        if snapshot:
            position, state, _ = snapshot
        else:
            position, state = 0, None
        
        # Replay from position
        events = self._store.get_stream(stream_id, position)
        for event in events:
            state = reduce_identity(state, event)
        
        return state


class GenomeProjection:
    """Read model for genome"""
    
    def __init__(self, event_store: PersistentEventStore):
        self._store = event_store
    
    def get_state(self, stream_id: str = "genome") -> Dict:
        snapshot = self._store.get_latest_snapshot(stream_id)
        
        if snapshot:
            position, state, _ = snapshot
        else:
            position, state = 0, None
        
        events = self._store.get_stream(stream_id, position)
        for event in events:
            state = reduce_genome(state, event)
        
        return state


class LineageProjection:
    """Read model for execution lineage"""
    
    def __init__(self, event_store: PersistentEventStore):
        self._store = event_store
    
    def get_history(self, stream_id: str = "lineage", limit: int = 100) -> List:
        events = self._store.get_stream(stream_id, 0)
        state = None
        for event in events:
            state = reduce_lineage(state, event)
        return (state or [])[-limit:]


# =============================================================================
# COMMAND HANDLERS
# =============================================================================

class CommandHandler:
    """Handles commands and produces events"""
    
    def __init__(self, event_store: PersistentEventStore):
        self._store = event_store
    
    def _append_event(self, stream_id: str, event_type: str, payload: Dict, caused_by: str = None) -> Event:
        """Append event to store"""
        position = self._store.get_global_position() + 1
        
        event = Event(
            event_id=str(uuid4()),
            event_type=event_type,
            stream_id=stream_id,
            position=position,
            timestamp=datetime.utcnow().isoformat(),
            caused_by=caused_by,
            payload=payload
        )
        
        self._store.append(event)
        return event
    
    def handle_execute_goal(self, command: ExecuteGoal) -> Event:
        """Handle goal execution command"""
        event = self._append_event(
            stream_id="execution",
            event_type="execution.started",
            payload={
                'goal_id': command.goal.get('id', str(uuid4())),
                'goal': command.goal,
                'command_id': command.command_id
            },
            caused_by=command.command_id
        )
        return event
    
    def handle_mutate_identity(self, command: MutateIdentity) -> Event:
        """Handle identity mutation command"""
        event = self._append_event(
            stream_id="identity",
            event_type="identity.axis.mutated",
            payload={
                'axis': command.axis,
                'delta': command.delta,
                'command_id': command.command_id
            },
            caused_by=command.caused_by
        )
        return event
    
    def handle_protect_axis(self, command: ProtectIdentityAxis) -> Event:
        """Handle protect axis command"""
        event = self._append_event(
            stream_id="identity",
            event_type="identity.axis.protected",
            payload={
                'axis': command.axis,
                'command_id': command.command_id
            },
            caused_by=command.caused_by
        )
        return event
    
    def handle_evolve_genome(self, command: EvolveGenome) -> Event:
        """Handle genome evolution command"""
        result = command.execution_result
        
        # Compute genome changes
        outcome = result.get('outcome', 'unknown')
        verifier_score = result.get('verifier_score', 0.5)
        execution_cost = result.get('execution_cost', 0.5)
        
        changes = {}
        if outcome == 'success':
            if verifier_score > 0.8 and execution_cost < 0.3:
                changes = {
                    'risk_tolerance': 0.03,
                    'exploration_bias': 0.08
                }
        elif outcome == 'failure':
            changes = {
                'risk_tolerance': -0.15,
                'validation_frequency': 1
            }
        
        event = self._append_event(
            stream_id="genome",
            event_type="genome.evolved",
            payload={
                'trigger': outcome,
                'changes': changes,
                'execution_result': result,
                'command_id': command.command_id
            },
            caused_by=command.caused_by
        )
        return event
    
    def handle_lineage_record(self, lineage_data: Dict) -> Event:
        """Record execution lineage"""
        event = self._append_event(
            stream_id="lineage",
            event_type="execution.lineage",
            payload=lineage_data
        )
        return event


# =============================================================================
# DETERMINISTIC REPLAY (Pure Function)
# =============================================================================

def replay_events(events: List[Event]) -> Dict[str, Any]:
    """
    Deterministic replay - pure function.
    Given same events, returns same state.
    No side effects.
    """
    identity_state = None
    genome_state = None
    lineage_state = None
    
    for event in events:
        identity_state = reduce_identity(identity_state, event)
        genome_state = reduce_genome(genome_state, event)
        lineage_state = reduce_lineage(lineage_state, event)
    
    return {
        'identity': identity_state,
        'genome': genome_state,
        'lineage': lineage_state
    }


def verify_deterministic_replay(event_store: PersistentEventStore) -> Dict:
    """Verify replay produces same state as live"""
    events = event_store.get_all(0, 10000)
    
    # Replay all events
    replayed_state = replay_events(events)
    
    # Get current projection state
    identity_proj = IdentityProjection(event_store)
    genome_proj = GenomeProjection(event_store)
    lineage_proj = LineageProjection(event_store)
    
    live_state = {
        'identity': identity_proj.get_state(),
        'genome': genome_proj.get_state(),
        'lineage': lineage_proj.get_history()
    }
    
    # Compare
    identity_match = replayed_state['identity'] == live_state['identity']
    genome_match = replayed_state['genome'] == live_state['genome']
    
    return {
        'deterministic': identity_match and genome_match,
        'events_replayed': len(events),
        'identity_match': identity_match,
        'genome_match': genome_match
    }


# =============================================================================
# RUNTIME CONTAINER
# =============================================================================

class CognitiveRuntimeContainer:
    """
    Production lifecycle container.
    Contains all runtime components with proper isolation.
    """
    
    def __init__(self, db_path: str = "cognitive_events.db"):
        # Infrastructure
        self.event_store = PersistentEventStore(db_path)
        
        # Projections
        self.identity_projection = IdentityProjection(self.event_store)
        self.genome_projection = GenomeProjection(self.event_store)
        self.lineage_projection = LineageProjection(self.event_store)
        
        # Command handlers
        self.command_handler = CommandHandler(self.event_store)
    
    def execute_goal(self, goal: Dict) -> str:
        """Execute goal - returns lineage_id"""
        cmd = ExecuteGoal(
            command_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            goal=goal
        )
        
        # Produce events
        exec_event = self.command_handler.handle_execute_goal(cmd)
        
        # Record lineage
        self.command_handler.handle_lineage_record({
            'lineage_id': exec_event.event_id,
            'goal_id': goal.get('id', str(uuid4())),
            'strategy': 'balanced_execute',
            'outcome': 'success',
            'interrupt': False
        })
        
        return exec_event.event_id
    
    def mutate_identity(self, axis: str, delta: float, caused_by: str = None) -> str:
        """Mutate identity axis"""
        cmd = MutateIdentity(
            command_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            axis=axis,
            delta=delta,
            caused_by=caused_by
        )
        event = self.command_handler.handle_mutate_identity(cmd)
        return event.event_id
    
    def get_state(self) -> Dict:
        """Get current runtime state"""
        return {
            'identity': self.identity_projection.get_state(),
            'genome': self.genome_projection.get_state(),
            'lineage': self.lineage_projection.get_history(limit=10),
            'global_position': self.event_store.get_global_position()
        }
    
    def snapshot(self):
        """Create snapshots for optimization"""
        identity = self.identity_projection.get_state()
        genome = self.genome_projection.get_state()
        position = self.event_store.get_global_position()
        
        self.event_store.save_snapshot("identity", position, identity)
        self.event_store.save_snapshot("genome", position, genome)
    
    def verify_replay(self) -> Dict:
        """Verify deterministic replay"""
        return verify_deterministic_replay(self.event_store)


# Global container
_container: Optional[CognitiveRuntimeContainer] = None


def get_runtime_container(db_path: str = "cognitive_events.db") -> CognitiveRuntimeContainer:
    global _container
    if _container is None:
        _container = CognitiveRuntimeContainer(db_path)
    return _container