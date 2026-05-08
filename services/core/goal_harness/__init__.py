"""
Goal Harness - Execution instrumentation and tracing
"""
from .decision_trace_logger import (
    log_decision_start,
    log_decision_complete,
    log_attempt,
    decision_trace_logger,
    synthesize_context
)

from .execution_trace_guard import ExecutionTraceGuard

__all__ = [
    "log_decision_start",
    "log_decision_complete", 
    "log_attempt",
    "decision_trace_logger",
    "synthesize_context",
    "ExecutionTraceGuard"
]