"""
Execution Journal - Append-only ordered event stream

CRITICAL: This is the source of truth for all execution history.
Becomes: training dataset, causal debugger, governance evidence, rollback source.

Event types:
- TASK_CREATED
- PLAN_GENERATED  
- CANDIDATE_SCORED
- SKILL_SELECTED
- TOOL_CALLED
- TOOL_RESULT
- POLICY_BLOCKED
- REWARD_ASSIGNED
- HUMAN_OVERRIDE
- REPLAY_STARTED
- REPLAY_DIVERGED
"""
import json
import hashlib
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from collections import deque


class EventType(str, Enum):
    """Execution event types"""
    TASK_CREATED = "task_created"
    PLAN_GENERATED = "plan_generated"
    CANDIDATE_DISCOVERED = "candidate_discovered"
    CANDIDATE_SCORED = "candidate_scored"
    SKILL_SELECTED = "skill_selected"
    SKILL_EXECUTING = "skill_executing"
    SKILL_COMPLETED = "skill_completed"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    POLICY_BLOCKED = "policy_blocked"
    POLICY_REJECTED = "policy_rejected"
    REWARD_ASSIGNED = "reward_assigned"
    HUMAN_OVERRIDE = "human_override"
    PROMOTION_EVALUATED = "promotion_evaluated"
    REPLAY_STARTED = "replay_started"
    REPLAY_COMPLETED = "replay_completed"
    REPLAY_DIVERGED = "replay_diverged"
    EVALUATION_COMPLETED = "evaluation_completed"
    DECOMPOSITION_STEP = "decomposition_step"
    GOAL_COMPLETED = "goal_completed"
    GOAL_FAILED = "goal_failed"


@dataclass(frozen=True)
class ExecutionEvent:
    """
    Immutable event in execution journal.
    
    Events are append-only - never modified after creation.
    CRITICAL: Chain integrity via previous_event_hash for forensic replay.
    """
    event_id: str
    event_type: EventType
    execution_id: str  # Link to execution
    parent_event_id: Optional[str]  # For causality chain
    
    # Chain integrity - cryptographic link to previous event
    previous_event_hash: str  # hash(event_(n-1).payload)
    chain_hash: str  # hash(event_n.payload + previous_event_hash)
    
    # Event data
    data: Dict[str, Any]
    
    # Timing
    timestamp: str
    sequence: int  # Monotonic sequence number
    
    # Verification
    event_hash: str  # Hash for integrity
    
    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "execution_id": self.execution_id,
            "parent_event_id": self.parent_event_id,
            "previous_event_hash": self.previous_event_hash,
            "chain_hash": self.chain_hash,
            "data": self.data,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "event_hash": self.event_hash
        }
    
    @staticmethod
    def from_dict(data: dict) -> "ExecutionEvent":
        return ExecutionEvent(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            execution_id=data["execution_id"],
            parent_event_id=data.get("parent_event_id"),
            previous_event_hash=data.get("previous_event_hash", ""),
            chain_hash=data.get("chain_hash", ""),
            data=data["data"],
            timestamp=data["timestamp"],
            sequence=data["sequence"],
            event_hash=data["event_hash"]
        )
    
    def compute_hash(self) -> str:
        """Compute event hash for integrity (without chain)"""
        hash_input = {
            "event_type": self.event_type.value,
            "execution_id": self.execution_id,
            "parent_event_id": self.parent_event_id,
            "data": self.data,
            "timestamp": self.timestamp,
            "sequence": self.sequence
        }
        return hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]


class ExecutionJournal:
    """
    Append-only journal of all execution events.
    
    Usage:
        journal = ExecutionJournal()
        
        # Record event
        journal.record(
            event_type=EventType.SKILL_SELECTED,
            execution_id="env-123",
            data={"skill": "core.echo", "score": 0.8}
        )
        
        # Query events
        events = journal.get_events_for_execution("env-123")
        
        # Get causal chain
        chain = journal.get_causal_chain(event_id)
    """
    
    def __init__(self, store_dir: str = "/app/execution_journal"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        # In-memory index
        self._by_execution: Dict[str, List[str]] = {}  # execution_id -> [event_ids]
        self._by_type: Dict[EventType, List[str]] = {}  # event_type -> [event_ids]
        self._events: Dict[str, ExecutionEvent] = {}
        self._sequence: int = 0
        
        self._load_existing()
    
    def _load_existing(self):
        """Load existing events"""
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    event = ExecutionEvent.from_dict(data)
                    self._events[event.event_id] = event
                    
                    # Update indexes
                    if event.execution_id not in self._by_execution:
                        self._by_execution[event.execution_id] = []
                    self._by_execution[event.execution_id].append(event.event_id)
                    
                    if event.event_type not in self._by_type:
                        self._by_type[event.event_type] = []
                    self._by_type[event.event_type].append(event.event_id)
                    
                    # Update sequence
                    if event.sequence > self._sequence:
                        self._sequence = event.sequence
            except:
                continue
    
    def _save_event(self, event: ExecutionEvent):
        """Save event to disk"""
        filename = self.store_dir / f"{event.event_id}.json"
        with open(filename, "w") as f:
            json.dump(event.to_dict(), f, indent=2)
    
    def record(
        self,
        event_type: EventType,
        execution_id: str,
        data: Dict[str, Any],
        parent_event_id: Optional[str] = None
    ) -> ExecutionEvent:
        """Record new event with chain integrity"""
        self._sequence += 1
        
        # Get previous event hash for chain
        exec_events = self._by_execution.get(execution_id, [])
        previous_hash = ""
        if exec_events:
            last_event = self._events.get(exec_events[-1])
            if last_event:
                previous_hash = last_event.event_hash
        
        event = ExecutionEvent(
            event_id=uuid4().hex[:8],
            event_type=event_type,
            execution_id=execution_id,
            parent_event_id=parent_event_id,
            previous_event_hash=previous_hash,
            chain_hash="",  # Will compute
            data=data,
            timestamp=datetime.utcnow().isoformat(),
            sequence=self._sequence,
            event_hash=""  # Will compute
        )
        
        # Compute hashes
        event_hash = event.compute_hash()
        
        # Compute chain hash: hash(event_payload + previous_event_hash)
        chain_input = {
            "event_hash": event_hash,
            "previous_event_hash": previous_hash
        }
        chain_hash = hashlib.sha256(
            json.dumps(chain_input, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Create final event with hashes
        from dataclasses import replace
        event = replace(event, event_hash=event_hash, chain_hash=chain_hash)
        
        # Save
        self._events[event.event_id] = event
        self._save_event(event)
        
        # Update indexes
        if execution_id not in self._by_execution:
            self._by_execution[execution_id] = []
        self._by_execution[execution_id].append(event.event_id)
        
        if event_type not in self._by_type:
            self._by_type[event_type] = []
        self._by_type[event_type].append(event.event_id)
        
        return event
    
    def get_event(self, event_id: str) -> Optional[ExecutionEvent]:
        """Get event by ID"""
        return self._events.get(event_id)
    
    def get_events_for_execution(
        self,
        execution_id: str,
        limit: Optional[int] = None
    ) -> List[ExecutionEvent]:
        """Get all events for execution (in order)"""
        event_ids = self._by_execution.get(execution_id, [])
        
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        events.sort(key=lambda e: e.sequence)
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_events_by_type(
        self,
        event_type: EventType,
        limit: int = 100
    ) -> List[ExecutionEvent]:
        """Get events by type (most recent first)"""
        event_ids = self._by_type.get(event_type, [])
        
        events = [self._events[eid] for eid in event_ids if eid in self._events]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return events[:limit]
    
    def get_causal_chain(self, event_id: str) -> List[ExecutionEvent]:
        """Get causal chain from this event back to origin"""
        chain = []
        current_id = event_id
        
        while current_id:
            event = self._events.get(current_id)
            if not event:
                break
            
            chain.append(event)
            current_id = event.parent_event_id
        
        # Reverse to get chronological order
        chain.reverse()
        return chain
    
    def get_latest_for_execution(self, execution_id: str) -> Optional[ExecutionEvent]:
        """Get most recent event for execution"""
        events = self.get_events_for_execution(execution_id, limit=1)
        return events[-1] if events else None
    
    def get_statistics(self) -> Dict:
        """Get journal statistics"""
        return {
            "total_events": len(self._events),
            "by_type": {
                et.value: len(eids) 
                for et, eids in self._by_type.items()
            },
            "total_executions": len(self._by_execution),
            "current_sequence": self._sequence
        }


class CounterfactualEntry:
    """
    Stores what almost happened, not just what happened.
    
    This enables:
    - Regret calculation
    - Ranking training
    - Off-policy evaluation
    - Counterfactual reasoning
    """
    
    def __init__(
        self,
        execution_id: str,
        selected_skill: str,
        rejected_candidates: List[Dict[str, Any]],  # skill, score, reason
        selection_context: Dict[str, Any]
    ):
        self.entry_id = uuid4().hex[:8]
        self.execution_id = execution_id
        self.selected_skill = selected_skill
        self.rejected_candidates = rejected_candidates  # What was rejected
        self.selection_context = selection_context
        self.created_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "execution_id": self.execution_id,
            "selected_skill": self.selected_skill,
            "rejected_candidates": self.rejected_candidates,
            "selection_context": self.selection_context,
            "created_at": self.created_at
        }
    
    @staticmethod
    def from_dict(data: dict) -> "CounterfactualEntry":
        entry = CounterfactualEntry.__new__(CounterfactualEntry)
        entry.entry_id = data["entry_id"]
        entry.execution_id = data["execution_id"]
        entry.selected_skill = data["selected_skill"]
        entry.rejected_candidates = data["rejected_candidates"]
        entry.selection_context = data["selection_context"]
        entry.created_at = data["created_at"]
        return entry
    
    def compute_regret(self) -> float:
        """Compute regret = best_rejected - selected"""
        if not self.rejected_candidates:
            return 0.0
        
        best_rejected = max(
            (c.get("estimated_reward", 0) for c in self.rejected_candidates),
            default=0
        )
        
        selected_reward = self.selection_context.get("selected_reward", 0)
        
        return max(0, best_rejected - selected_reward)


class CounterfactualStore:
    """Store for counterfactual entries"""
    
    def __init__(self, store_dir: str = "/app/counterfactual_store"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
        
        self._entries: Dict[str, CounterfactualEntry] = {}
        self._load_existing()
    
    def _load_existing(self):
        for filename in self.store_dir.glob("*.json"):
            try:
                with open(filename, "r") as f:
                    data = json.load(f)
                    entry = CounterfactualEntry.from_dict(data)
                    self._entries[entry.entry_id] = entry
            except:
                continue
    
    def record(
        self,
        execution_id: str,
        selected_skill: str,
        rejected_candidates: List[Dict[str, Any]],
        selection_context: Dict[str, Any]
    ) -> CounterfactualEntry:
        """Record counterfactual selection"""
        entry = CounterfactualEntry(
            execution_id=execution_id,
            selected_skill=selected_skill,
            rejected_candidates=rejected_candidates,
            selection_context=selection_context
        )
        
        # Save to disk
        with open(self.store_dir / f"{entry.entry_id}.json", "w") as f:
            json.dump(entry.to_dict(), f, indent=2)
        
        self._entries[entry.entry_id] = entry
        
        return entry
    
    def get_for_execution(self, execution_id: str) -> Optional[CounterfactualEntry]:
        """Get counterfactual entry for execution"""
        for entry in self._entries.values():
            if entry.execution_id == execution_id:
                return entry
        return None
    
    def get_all(self, limit: int = 100) -> List[CounterfactualEntry]:
        """Get all entries (most recent first)"""
        entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at,
            reverse=True
        )
        return entries[:limit]
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self._entries:
            return {"total": 0, "avg_regret": 0.0}
        
        regrets = [e.compute_regret() for e in self._entries.values()]
        
        return {
            "total": len(self._entries),
            "avg_regret": sum(regrets) / len(regrets) if regrets else 0.0,
            "max_regret": max(regrets) if regrets else 0.0
        }


# Global stores
_journal: Optional[ExecutionJournal] = None
_counterfactual: Optional[CounterfactualStore] = None


def get_execution_journal() -> ExecutionJournal:
    """Get execution journal"""
    global _journal
    if _journal is None:
        _journal = ExecutionJournal()
    return _journal


def get_counterfactual_store() -> CounterfactualStore:
    """Get counterfactual store"""
    global _counterfactual
    if _counterfactual is None:
        _counterfactual = CounterfactualStore()
    return _counterfactual


def record_execution_event(
    event_type: EventType,
    execution_id: str,
    data: Dict[str, Any],
    parent_event_id: Optional[str] = None
) -> ExecutionEvent:
    """Convenience function to record event"""
    return get_execution_journal().record(event_type, execution_id, data, parent_event_id)