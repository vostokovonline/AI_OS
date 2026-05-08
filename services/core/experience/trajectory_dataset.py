"""
Trace Schema - Frozen v1.0

All traces follow this schema. No silent additions.
Migration required for schema changes.
"""
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

TRACE_SCHEMA_VERSION = "v1"

@dataclass(frozen=True)
class TraceContext:
    """Immutable context features for bandit learning"""
    goal_type: str
    goal_length: int
    domain: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    attempt: int = 1
    previous_failures: int = 0
    depth_level: int = 0
    has_subgoals: bool = False
    completion_criteria_exists: bool = False
    constraints_count: int = 0
    execution_mode: str = "auto"
    goal_description: str = ""
    candidates: List[str] = None
    planner_depth: int = 0
    retry_count: int = 0

@dataclass(frozen=True)
class TraceMetadata:
    """Immutable trace metadata"""
    trace_id: str
    goal_id: str
    goal_type: str
    task_type: str
    phase: str
    started_at: str
    completed_at: Optional[str] = None

@dataclass(frozen=True)
class TraceOutcome:
    """Immutable outcome data"""
    success: bool
    raw_reward: float
    confidence: float = 0.0
    latency_ms: int = 0
    skill_id: Optional[str] = None
    artifacts_count: int = 0
    error: Optional[str] = None


@dataclass
class LearningSample:
    """
    Normalized learning sample for bandit training.
    Fixed fields, numeric features, bounded rewards.
    """
    trace_id: str
    goal_id: str
    context: TraceContext
    candidates: List[str]
    chosen_skill: str
    reward: float  # Bounded: [-1, 1]
    latency_ms: int
    success: bool
    timestamp: str
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "goal_id": self.goal_id,
            "context": {
                "goal_type": self.context.goal_type,
                "goal_length": self.context.goal_length,
                "domain": self.context.domain,
                "attempt": self.context.attempt,
                "retry_count": self.context.retry_count
            },
            "candidates": self.candidates,
            "chosen_skill": self.chosen_skill,
            "reward": round(self.reward, 3),
            "latency_ms": self.latency_ms,
            "success": self.success,
            "timestamp": self.timestamp
        }


class TraceSchemaValidator:
    """Validates traces conform to v1 schema"""
    
    REQUIRED_FIELDS = [
        "trace_id", "goal_id", "goal_type", "task_type",
        "phase", "started_at", "context"
    ]
    
    REQUIRED_CONTEXT = [
        "goal_type", "goal_length", "domain"
    ]
    
    @staticmethod
    def validate(trace: dict) -> bool:
        for field in TraceSchemaValidator.REQUIRED_FIELDS:
            if field not in trace:
                raise ValueError(f"Missing required field: {field}")
        
        ctx = trace.get("context", {})
        for field in TraceSchemaValidator.REQUIRED_CONTEXT:
            if field not in ctx:
                raise ValueError(f"Missing required context field: {field}")
        
        return True
    
    @staticmethod
    def get_version() -> str:
        return TRACE_SCHEMA_VERSION