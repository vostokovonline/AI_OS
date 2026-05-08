"""
Decision Trace Logger - Lifecycle instrumentation for bandit learning

Provides:
- log_decision_start: Begin trace with context
- log_decision_complete: End trace with outcome
- log_attempt: Log individual retry attempts
- synthesize_context: Create feature context from goal metadata
- decision_trace_logger: Singleton for global trace access
"""
import json
import os
import uuid
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Thread-safe storage for active traces
_active_traces: Dict[str, Dict] = {}
_traces_lock = threading.Lock()

# Trace directory
TRACE_DIR = Path("/app/decision_traces")
TRACE_DIR.mkdir(exist_ok=True, parents=True)


def synthesize_context(
    goal_type: str = "unknown",
    goal_length: int = 0,
    domain: str = "unknown",
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    attempt: int = 1,
    previous_failures: int = 0,
    depth_level: int = 0,
    has_subgoals: bool = False,
    completion_criteria_exists: bool = False,
    constraints_count: int = 0,
    execution_mode: str = "auto",
    goal_description: str = "",
    candidates: list = None,
    planner_depth: int = 0,
    retry_count: int = 0,
    **kwargs
) -> Dict[str, Any]:
    """
    Synthesize context features from goal metadata.
    Returns 13-feature context dict for bandit learning.
    """
    return {
        "goal_type": goal_type,
        "goal_length": goal_length,
        "domain": domain,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "attempt": attempt,
        "previous_failures": previous_failures,
        "depth_level": depth_level,
        "has_subgoals": has_subgoals,
        "completion_criteria_exists": completion_criteria_exists,
        "constraints_count": constraints_count,
        "execution_mode": execution_mode,
        "goal_description": goal_description,
        "candidates": candidates or [],
        "planner_depth": planner_depth,
        "retry_count": retry_count
    }


def log_decision_start(
    goal_id: str,
    goal_type: str = "unknown",
    task_type: str = "general",
    candidates: List[str] = None,
    legacy_choice: Optional[str] = None,
    legacy_q_values: Optional[Dict[str, float]] = None,
    phase: str = "execution",
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Start a new decision trace.
    Returns trace_id for later completion.
    """
    global _active_traces
    
    trace_id = str(uuid.uuid4())[:8]
    
    trace = {
        "trace_id": trace_id,
        "goal_id": goal_id,
        "goal_type": goal_type,
        "task_type": task_type,
        "candidates": candidates or [],
        "legacy_choice": legacy_choice,
        "legacy_q_values": legacy_q_values or {},
        "phase": phase,
        "context": context or {},
        "started_at": datetime.utcnow().isoformat(),
        "attempts": []
    }
    
    with _traces_lock:
        _active_traces[trace_id] = trace
    
    print(f"[TRACE_START] trace_id={trace_id} goal_id={goal_id[:8] if goal_id else 'none'} phase={phase}", flush=True)
    
    # Write to file for persistence
    _write_trace_to_file(trace, "start")
    
    return trace_id


def log_attempt(
    trace_id: str,
    attempt: int = 1,
    skill_id: Optional[str] = None,
    success: bool = False,
    latency_ms: float = 0.0,
    error: Optional[str] = None,
    artifacts_count: int = 0,
    confidence: float = 0.0,
    attempt_num: int = None,
    reward: float = None,
    **kwargs
) -> None:
    """Log individual attempt within a trace."""
    # Handle both attempt and attempt_num
    actual_attempt = attempt_num if attempt_num is not None else attempt
    global _active_traces
    
    with _traces_lock:
        if trace_id in _active_traces:
            trace = _active_traces[trace_id]
            trace["attempts"].append({
                "attempt": attempt,
                "skill_id": skill_id,
                "success": success,
                "latency_ms": latency_ms,
                "error": error,
                "artifacts_count": artifacts_count,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    print(f"[TRACE_ATTEMPT] trace_id={trace_id} attempt={actual_attempt} success={success}", flush=True)


def log_decision_complete(
    trace_id: str,
    success: bool,
    raw_reward: float = 0.0,
    confidence: float = 0.0,
    latency_ms: float = 0.0,
    skill_id: Optional[str] = None,
    artifacts_count: int = 0,
    phase: str = "execution",
    error: Optional[str] = None,
    final_context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Complete a decision trace with outcome.
    """
    global _active_traces
    
    completed_at = datetime.utcnow().isoformat()
    
    with _traces_lock:
        if trace_id in _active_traces:
            trace = _active_traces[trace_id]
            trace["completed_at"] = completed_at
            trace["success"] = success
            trace["raw_reward"] = raw_reward
            trace["confidence"] = confidence
            trace["latency_ms"] = latency_ms
            trace["skill_id"] = skill_id
            trace["artifacts_count"] = artifacts_count
            trace["phase"] = phase
            trace["error"] = error
            trace["final_context"] = final_context or {}
            
            # Remove from active (completed)
            del _active_traces[trace_id]
    
    print(f"[TRACE_COMPLETE] trace_id={trace_id} success={success} raw_reward={raw_reward}", flush=True)
    
    # Write to file for persistence
    with _traces_lock:
        if trace_id in _active_traces:
            trace = _active_traces[trace_id]
        else:
            trace = {
                "trace_id": trace_id,
                "completed_at": completed_at,
                "success": success,
                "raw_reward": raw_reward
            }
    _write_trace_to_file(trace, "complete")


def _write_trace_to_file(trace: Dict, phase: str) -> None:
    """Write trace to JSONL file for persistence."""
    try:
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        filename = TRACE_DIR / f"trace_{timestamp}.jsonl"
        
        with open(filename, "a") as f:
            f.write(json.dumps(trace) + "\n")
        
        print(f"[TRACE_WRITE] {phase} -> {filename}", flush=True)
    except Exception as e:
        print(f"[TRACE_WRITE_ERROR] {e}", flush=True)


class DecisionTraceLogger:
    """
    Singleton logger for global trace access.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def start(self, goal_id: str, **kwargs) -> str:
        return log_decision_start(goal_id, **kwargs)
    
    def complete(self, trace_id: str, **kwargs) -> None:
        log_decision_complete(trace_id, **kwargs)
    
    def attempt(self, trace_id: str, **kwargs) -> None:
        log_attempt(trace_id, **kwargs)


# Global singleton
decision_trace_logger = DecisionTraceLogger()