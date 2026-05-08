"""
Learning Event Schema - Immutable finalized learning record

Canonical format for all learning data:
- Replay dataset
- RL training samples
- Policy evaluation
- Dashboard metrics

Schema version: v3
"""
from dataclasses import dataclass, field, replace
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class EventType(Enum):
    SKILL_EXECUTION = "skill_execution"
    DECISION_SHADOW = "decision_shadow"
    POLICY_UPDATE = "policy_update"
    REGRET_METRIC = "regret_metric"


class PolicyVersion(Enum):
    LEGACY = "legacy_v1"
    THOMPSON_V1 = "thompson_v1"
    THOMPSON_V2 = "thompson_v2"


SCHEMA_VERSION = "v3"


@dataclass(frozen=True)
class LearningEvent:
    """
    Learning event - the core data structure for AI-OS learning.
    
    One event per skill execution with full decision context.
    """
    # Identity
    event_type: str = "skill_execution"
    trace_id: str = ""
    event_id: str = ""  # Unique event ID
    timestamp: str = ""
    
    # Context
    context_features: Dict[str, Any] = field(default_factory=dict)
    goal_type: str = ""
    domain: str = ""
    
    # Decision
    candidates: List[str] = field(default_factory=list)
    executed_arm: str = ""  # What actually ran
    shadow_arm: str = ""   # What bandit would have picked
    
    # Outcome
    reward: float = 0.0
    success: bool = False
    latency_ms: int = 0
    error: Optional[str] = None
    
    # Reward decomposition (for rich learning signal)
    reward_success: float = 0.0      # Base success reward
    reward_latency: float = 0.0       # Latency penalty
    reward_quality: float = 0.0       # Artifact quality bonus
    reward_penalty: float = 0.0      # Error/retry penalties
    
    # Learning metrics
    regret: float = 0.0
    policy_version: str = "legacy_v1"
    
    # Schema
    schema_version: str = SCHEMA_VERSION
    
    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "context_features": self.context_features,
            "goal_type": self.goal_type,
            "domain": self.domain,
            "candidates": self.candidates,
            "executed_arm": self.executed_arm,
            "shadow_arm": self.shadow_arm,
            "reward": round(self.reward, 3),
            "success": self.success,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "regret": round(self.regret, 3),
            "policy_version": self.policy_version,
            "schema_version": self.schema_version,
            # Reward decomposition
            "reward_success": round(self.reward_success, 3),
            "reward_latency": round(self.reward_latency, 3),
            "reward_quality": round(self.reward_quality, 3),
            "reward_penalty": round(self.reward_penalty, 3)
        }


class LearningEventStore:
    """
    Immutable store for learning events.
    Append-only, one file per event for replay safety.
    """
    
    def __init__(self, store_dir: str = "/app/learning_events"):
        from pathlib import Path
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True, parents=True)
    
    def append(self, event: LearningEvent) -> str:
        """Append event to store, return event_id - uses replace() for immutability"""
        import uuid
        import json
        
        # Generate event_id if not set (using replace for frozen dataclass)
        event_id = event.event_id or str(uuid.uuid4())[:8]
        
        # Create new event with event_id using replace() (immutable)
        final_event = replace(event, event_id=event_id, timestamp=event.timestamp or datetime.utcnow().isoformat())
        
        filename = self.store_dir / f"{final_event.event_type}_{event_id}.json"
        
        with open(filename, "w") as f:
            json.dump(final_event.to_dict(), f, indent=2)
        
        return event_id
    
    def get_events(
        self,
        event_type: str = None,
        min_regret: float = None,
        limit: int = None
    ) -> List[LearningEvent]:
        """Query events with filters"""
        import json
        
        events = []
        
        for filepath in self.store_dir.glob("*.json"):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                if event_type and data.get("event_type") != event_type:
                    continue
                if min_regret is not None and data.get("regret", 0) < min_regret:
                    continue
                
                events.append(LearningEvent(**data))
                
                if limit and len(events) >= limit:
                    break
            except:
                continue
        
        return events
    
    def get_statistics(self) -> dict:
        """Get store statistics"""
        events = self.get_events()
        
        if not events:
            return {"total_events": 0}
        
        regrets = [e.regret for e in events if e.regret is not None]
        successes = sum(1 for e in events if e.success)
        
        return {
            "total_events": len(events),
            "success_rate": successes / len(events),
            "avg_regret": sum(regrets) / len(regrets) if regrets else 0,
            "min_regret": min(regrets) if regrets else 0,
            "max_regret": max(regrets) if regrets else 0,
            "unique_policies": len(set(e.policy_version for e in events))
        }