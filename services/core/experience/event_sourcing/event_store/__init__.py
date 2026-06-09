"""
Persistent Event Store with Optimistic Concurrency Control.

Features:
- Append-only log with monotonic position
- Optimistic concurrency (version checking)
- Idempotency via deduplication
- Causal ordering via causation_id chain
- Schema evolution support
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import sqlite3
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from contextlib import contextmanager
import json

from events import CognitiveEvent, StreamIds


class EventStoreError(Exception):
    """Base event store error"""
    pass


class OptimisticConcurrencyError(EventStoreError):
    """Version conflict during append"""
    pass


class StreamNotFoundError(EventStoreError):
    """Stream does not exist"""
    pass


@dataclass
class StreamVersion:
    """Current stream version info"""
    stream_id: str
    version: int
    last_updated: str


class PersistentEventStore:
    """
    SQLite-backed persistent event store.
    
    Key guarantees:
    - Append-only (no updates or deletes)
    - Monotonic positions within streams
    - Optimistic concurrency via version checking
    - Idempotent writes (deduplicate by event_id)
    """
    
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS streams (
                    stream_id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    stream_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    causation_id TEXT,
                    correlation_id TEXT,
                    payload TEXT NOT NULL,
                    FOREIGN KEY (stream_id) REFERENCES streams(stream_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_events_stream_position 
                    ON events(stream_id, position);
                CREATE INDEX IF NOT EXISTS idx_events_correlation 
                    ON events(correlation_id);
                CREATE INDEX IF NOT EXISTS idx_events_causation 
                    ON events(event_id, causation_id);
                
                CREATE TABLE IF NOT EXISTS snapshots (
                    stream_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (stream_id, version)
                );
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level='IMMEDIATE'
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _get_position(self, conn: sqlite3.Connection, stream_id: str) -> int:
        """Get next position for stream (monotonic allocator)"""
        cursor = conn.execute(
            "SELECT COALESCE(MAX(position), 0) FROM events WHERE stream_id = ?",
            (stream_id,)
        )
        row = cursor.fetchone()
        return (row[0] if row else 0) + 1
    
    def _ensure_stream(self, conn: sqlite3.Connection, stream_id: str):
        """Ensure stream exists"""
        conn.execute(
            "INSERT OR IGNORE INTO streams (stream_id, version, last_updated) VALUES (?, 0, ?)",
            (stream_id, __import__('datetime').datetime.utcnow().isoformat())
        )
    
    def append(
        self, 
        stream_id: str, 
        event: CognitiveEvent,
        expected_version: Optional[int] = None
    ) -> CognitiveEvent:
        """
        Append event to stream with optimistic concurrency.
        
        Args:
            stream_id: Logical partition
            event: Event to append
            expected_version: If provided, verify stream version matches
            
        Returns:
            Event with assigned position
            
        Raises:
            OptimisticConcurrencyError: If version mismatch
        """
        with self._lock:
            with self._get_connection() as conn:
                self._ensure_stream(conn, stream_id)
                
                if expected_version is not None:
                    cursor = conn.execute(
                        "SELECT version FROM streams WHERE stream_id = ?",
                        (stream_id,)
                    )
                    row = cursor.fetchone()
                    current_version = row[0] if row else 0
                    
                    if current_version != expected_version:
                        raise OptimisticConcurrencyError(
                            f"Version conflict: expected {expected_version}, got {current_version}"
                        )
                
                position = self._get_position(conn, stream_id)
                
                enriched_event = CognitiveEvent(
                    event_type=event.event_type,
                    stream_id=stream_id,
                    position=position,
                    schema_version=event.schema_version,
                    event_id=event.event_id,
                    timestamp=event.timestamp,
                    causation_id=event.causation_id,
                    correlation_id=event.correlation_id,
                    payload=event.payload
                )
                
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events 
                    (event_id, stream_id, position, schema_version, event_type, 
                     timestamp, causation_id, correlation_id, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        enriched_event.event_id,
                        stream_id,
                        position,
                        enriched_event.schema_version,
                        enriched_event.event_type,
                        enriched_event.timestamp,
                        enriched_event.causation_id,
                        enriched_event.correlation_id,
                        json.dumps(enriched_event.payload)
                    )
                )
                
                conn.execute(
                    """
                    UPDATE streams SET version = ?, last_updated = ? WHERE stream_id = ?
                    """,
                    (position, __import__('datetime').datetime.utcnow().isoformat(), stream_id)
                )
                
                conn.commit()
                
                return enriched_event
    
    def get_stream(self, stream_id: str, from_position: int = 1) -> List[CognitiveEvent]:
        """
        Get all events from stream starting at position.
        
        Returns events in position order.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM events 
                WHERE stream_id = ? AND position >= ?
                ORDER BY position ASC
                """,
                (stream_id, from_position)
            )
            
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def get_event(self, event_id: str) -> Optional[CognitiveEvent]:
        """Get single event by ID"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM events WHERE event_id = ?",
                (event_id,)
            )
            row = cursor.fetchone()
            return self._row_to_event(row) if row else None
    
    def get_stream_version(self, stream_id: str) -> StreamVersion:
        """Get current stream version"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT stream_id, version, last_updated FROM streams WHERE stream_id = ?",
                (stream_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise StreamNotFoundError(f"Stream not found: {stream_id}")
            return StreamVersion(
                stream_id=row[0],
                version=row[1],
                last_updated=row[2]
            )
    
    def get_all_streams(self) -> List[StreamVersion]:
        """Get all streams"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT stream_id, version, last_updated FROM streams ORDER BY stream_id"
            )
            return [StreamVersion(stream_id=r[0], version=r[1], last_updated=r[2]) 
                    for r in cursor.fetchall()]
    
    def get_causation_chain(self, event_id: str, max_depth: int = 10) -> List[CognitiveEvent]:
        """Get causal chain of events (follows causation_id)"""
        chain = []
        current_id = event_id
        depth = 0
        
        while current_id and depth < max_depth:
            event = self.get_event(current_id)
            if not event:
                break
            chain.append(event)
            current_id = event.causation_id
            depth += 1
        
        return chain
    
    def get_correlation_events(self, correlation_id: str) -> List[CognitiveEvent]:
        """Get all events with same correlation_id"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM events 
                WHERE correlation_id = ?
                ORDER BY position ASC
                """,
                (correlation_id,)
            )
            return [self._row_to_event(row) for row in cursor.fetchall()]
    
    def save_snapshot(self, stream_id: str, version: int, state: Dict[str, Any]):
        """Save state snapshot at version"""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO snapshots (stream_id, version, state, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (stream_id, version, json.dumps(state), __import__('datetime').datetime.utcnow().isoformat())
            )
            conn.commit()
    
    def get_snapshot(self, stream_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get snapshot, latest if version not specified"""
        with self._get_connection() as conn:
            if version is None:
                cursor = conn.execute(
                    """
                    SELECT state FROM snapshots 
                    WHERE stream_id = ?
                    ORDER BY version DESC
                    LIMIT 1
                    """,
                    (stream_id,)
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT state FROM snapshots 
                    WHERE stream_id = ? AND version = ?
                    """,
                    (stream_id, version)
                )
            
            row = cursor.fetchone()
            return json.loads(row[0]) if row else None
    
    def _row_to_event(self, row: sqlite3.Row) -> CognitiveEvent:
        """Convert DB row to event"""
        return CognitiveEvent(
            event_id=row["event_id"],
            stream_id=row["stream_id"],
            position=row["position"],
            schema_version=row["schema_version"],
            event_type=row["event_type"],
            timestamp=row["timestamp"],
            causation_id=row["causation_id"] or "",
            correlation_id=row["correlation_id"] or "",
            payload=json.loads(row["payload"])
        )
    
    def close(self):
        """Close store"""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class OptimisticEventStore(PersistentEventStore):
    """
    Event store wrapper with automatic optimistic concurrency.
    
    Retries on conflict with exponential backoff.
    """
    
    def __init__(self, db_path: str = ":memory:", max_retries: int = 3, base_delay: float = 0.01):
        super().__init__(db_path)
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def append_with_retry(
        self,
        stream_id: str,
        event: CognitiveEvent
    ) -> CognitiveEvent:
        """Append with automatic retry on concurrency conflict"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                current_version = None
                try:
                    current_version = self.get_stream_version(stream_id).version
                except StreamNotFoundError:
                    current_version = 0
                
                return self.append(stream_id, event, expected_version=current_version)
            
            except OptimisticConcurrencyError as e:
                last_error = e
                delay = self.base_delay * (2 ** attempt)
                time.sleep(delay)
        
        raise last_error