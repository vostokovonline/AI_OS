"""
Execution Trace Guard - Context manager for guaranteed lifecycle completion

Usage:
    with ExecutionTraceGuard(goal_id, goal_type, ...) as trace:
        # ... execution ...
        trace.success(skill_id, confidence, latency_ms)
    
    # Automatic completion on exception:
    # trace.success = False, raw_reward = -1.0
"""
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime

from .decision_trace_logger import log_decision_start, log_decision_complete, synthesize_context


class ExecutionTraceGuard:
    """
    Context manager that guarantees trace completion.
    
    Automatically logs:
    - TRACE_START on __enter__
    - TRACE_COMPLETE on __exit__ (success or failure)
    """
    
    def __init__(
        self,
        goal_id: str,
        goal_type: str = "unknown",
        task_type: str = "execution",
        candidates: List[str] = None,
        legacy_choice: Optional[str] = None,
        phase: str = "execution"
    ):
        self.goal_id = goal_id
        self.goal_type = goal_type
        self.task_type = task_type
        self.candidates = candidates or []
        self.legacy_choice = legacy_choice
        self.phase = phase
        
        self.trace_id: Optional[str] = None
        self._entered = False
    
    def __enter__(self):
        try:
            context = synthesize_context(
                goal_type=self.goal_type,
                goal_length=0,
                domain="unknown"
            )
            self.trace_id = log_decision_start(
                goal_id=self.goal_id,
                goal_type=self.goal_type,
                task_type=self.task_type,
                candidates=self.candidates,
                legacy_choice=self.legacy_choice,
                legacy_q_values=None,
                phase=self.phase,
                context=context
            )
            self._entered = True
            print(f"[TRACE_GUARD] Started trace_id={self.trace_id}", flush=True)
        except Exception as e:
            print(f"[TRACE_GUARD] Failed to start: {e}", flush=True)
            self.trace_id = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.trace_id:
            return False
        
        try:
            if exc_type is not None:
                # Exception occurred - log failure
                log_decision_complete(
                    trace_id=self.trace_id,
                    success=False,
                    raw_reward=-1.0,
                    confidence=0.0,
                    latency_ms=0,
                    skill_id=None,
                    artifacts_count=0,
                    phase=f"{self.phase}_error",
                    error=str(exc_val)[:200] if exc_val else "unknown"
                )
                print(f"[TRACE_GUARD] Error completion trace_id={self.trace_id}", flush=True)
            else:
                # No exception - already completed via success() call
                print(f"[TRACE_GUARD] Completed normally trace_id={self.trace_id}", flush=True)
        except Exception as e:
            print(f"[TRACE_GUARD] Completion failed: {e}", flush=True)
        
        return False  # Don't suppress exceptions
    
    def success(
        self,
        skill_id: str,
        confidence: float,
        latency_ms: int,
        artifacts_count: int = 0
    ):
        """Mark execution as successful"""
        if not self.trace_id:
            return
        
        try:
            log_decision_complete(
                trace_id=self.trace_id,
                success=True,
                raw_reward=1.0,
                confidence=confidence,
                latency_ms=latency_ms,
                skill_id=skill_id,
                artifacts_count=artifacts_count,
                phase=self.phase
            )
            print(f"[TRACE_GUARD] Success trace_id={self.trace_id}", flush=True)
        except Exception as e:
            print(f"[TRACE_GUARD] Success logging failed: {e}", flush=True)
    
    def failure(self, error: str = "unknown"):
        """Mark execution as failed (without exception)"""
        if not self.trace_id:
            return
        
        try:
            log_decision_complete(
                trace_id=self.trace_id,
                success=False,
                raw_reward=-1.0,
                confidence=0.0,
                latency_ms=0,
                skill_id=None,
                artifacts_count=0,
                phase=f"{self.phase}_failed",
                error=error[:200]
            )
            print(f"[TRACE_GUARD] Failure trace_id={self.trace_id}", flush=True)
        except Exception as e:
            print(f"[TRACE_GUARD] Failure logging failed: {e}", flush=True)