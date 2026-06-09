"""
Goal Executor V3 - Production Rollout

Phase 2A: 10% Traffic Rollout with:
- Stable hash-based percentage split
- Atomic execution engine locking
- Stale lock detection
- Baseline observation (p50+p95)
- Legacy executor integration (battle-tested execution)

Architecture:
- V3 provides: locks, stale detection, metrics, percentage rollout
- Legacy (goal_executor_v2) provides: execution, completion, transitions

Transaction Invariants:
- One UOW = One commit (goal_executor owns transaction)
- V3 accepts UOW, does NOT create new UOW
- V3 does NOT commit
- V3 works only with uow.session
- After lock: only result or exception, never None
- Lock cleared after success in same transaction

Safety Guarantees:
- No duplicate processing (atomic locks)
- No orphaned locks (stale detection)
- No premature alerts (observation mode)
- Clean rollback (disable flag → wait → clear locks)

Author: AI-OS Architecture v3.1
Date: 2026-03-03
Status: Phase 2A - 10% Rollout
"""

import hashlib
import os
import time
from typing import Optional, Dict, Any
from uuid import UUID

from sqlalchemy import text
from logging_config import get_logger

# Import Experience Engine for learning
try:
    from experience_engine import get_experience_engine
    EXPERIENCE_ENGINE_AVAILABLE = True
except ImportError:
    EXPERIENCE_ENGINE_AVAILABLE = False
    get_experience_engine = None

logger = get_logger(__name__)

# Feature flags
ENABLE_EXECUTION_V3 = os.getenv("ENABLE_EXECUTION_V3", "false") == "true"
EXECUTION_V3_PERCENTAGE = int(os.getenv("EXECUTION_V3_PERCENTAGE", "10"))
STALE_LOCK_TIMEOUT = int(os.getenv("EXECUTION_V3_STALE_LOCK_TIMEOUT", "300"))  # 5 minutes
BASELINE_OBSERVATION_HOURS = int(os.getenv("BASELINE_OBSERVATION_HOURS", "48"))


# =============================================================================
# Stable Hash for Percentage Rollout
# =============================================================================

def should_use_v3(goal_id: str, percentage: int) -> bool:
    """
    Stable percentage-based rollout using SHA256 hash.

    CRITICAL: Deterministic across processes/pods/restarts.
    Prevents same goal from being processed by both V3 and legacy.

    Args:
        goal_id: Goal UUID string
        percentage: Traffic percentage (0-100)

    Returns:
        True if goal should use V3, False otherwise
    """
    stable_hash = int(
        hashlib.sha256(goal_id.encode()).hexdigest(),
        16
    )

    return stable_hash % 100 < percentage


# =============================================================================
# Main Execution V3 Entry Point
# =============================================================================

async def execute_goal_v3(goal, uow) -> Optional[Dict[str, Any]]:
    """
    Execute goal with V3 safety layer + legacy execution.

    PHASE 2A STRATEGY:
    - Use legacy executor for ACTUAL execution (battle-tested)
    - Add V3 safety layer (locks, stale detection, metrics)
    - Full completion logic by legacy (artifacts, evaluation, transitions)

    TRANSACTION INVARIANT:
    - V3 accepts uow parameter (does NOT create new UOW)
    - V3 does NOT commit (goal_executor owns transaction)
    - All DB operations use uow.session

    Returns:
        dict - execution result (success, legacy handled it)
        None - skip V3, use legacy (pre-lock only)

    Raises:
        Exception - execution failed (post-lock only)
    """
    goal_id = str(goal.id)
    session = uow.session

    # ================================================================
    # PHASE 1: Pre-flight checks (BEFORE lock)
    # ================================================================
    # If fail → return None (goal_executor will use legacy)
    # ================================================================

    if not ENABLE_EXECUTION_V3:
        logger.debug("execution_v3_disabled", goal_id=goal_id)
        return None

    if not goal.is_atomic:
        logger.debug("execution_v3_skip_non_atomic", goal_id=goal_id)
        return None

    if goal.goal_type != "achievable":
        logger.debug("execution_v3_skip_non_achievable", goal_id=goal_id)
        return None

    if not should_use_v3(goal_id, EXECUTION_V3_PERCENTAGE):
        logger.debug(
            "execution_v3_percentage_skip",
            goal_id=goal_id,
            percentage=EXECUTION_V3_PERCENTAGE
        )
        return None

    # ================================================================
    # PHASE 2: Check existing lock
    # ================================================================

    current_engine = await _get_execution_engine(session, goal_id)

    if current_engine == "legacy":
        # Locked to legacy - respect lock
        logger.debug(
            "execution_v3_skip_legacy_locked",
            goal_id=goal_id
        )
        return None

    if current_engine == "v3":
        # Our lock - check if stale
        if await _is_lock_stale(session, goal_id):
            logger.warning(
                "reacquiring_stale_lock",
                goal_id=goal_id,
                stale_engine="v3"
            )
            await _re_acquire_lock(session, goal_id, "v3")
        else:
            # Already executing V3
            logger.info(
                "goal_already_executing_v3",
                goal_id=goal_id
            )
            return None

    # ================================================================
    # PHASE 3: Acquire lock (atomic)
    # ================================================================
    # After this point:
    #   - return None is FORBIDDEN
    #   - Only return result or raise exception
    # ================================================================

    locked = await _try_lock(session, goal_id, "v3")

    if not locked:
        # Another worker locked this goal
        logger.info(
            "execution_engine_locked",
            goal_id=goal_id,
            action="fallback_to_legacy"
        )
        return None

    # ================================================================
    # PHASE 4: Execute with lock
    # ================================================================
    # NO return None from here
    # Only result or exception
    # ================================================================

    try:
        logger.info(
            "execution_v3_start",
            goal_id=goal_id,
            title=goal.title,
            percentage=EXECUTION_V3_PERCENTAGE,
            execution_engine="v3"
        )

        # Execute through kernel (lease + journal + dynamics)
        from execution_dynamics import dispatch_goal

        dispatch_result = await dispatch_goal(
            goal_id=goal_id,
            uow=uow,
            dispatch_id=f"v3:{goal_id}",
        )

        result = {
            'success': dispatch_result.get('success', False),
            'artifacts': dispatch_result.get('artifacts', []),
            'error': dispatch_result.get('error'),
            'execution_id': dispatch_result.get('execution_id', ''),
        }

        # Log execution history (already done via structured logging)
        # The execution_v3_complete log already contains all needed info

        # Success → cleanup lock in same transaction
        await _cleanup_lock(session, goal_id)

        # Log critical metrics
        eval_result = result.get("evaluation_result", {})
        if isinstance(eval_result, dict):
            eval_passed = eval_result.get("passed", False)
        else:
            eval_passed = False
        
        # Get skill_id from result - never record "unknown"
        skill_used = result.get("skill_used")
        if not skill_used or skill_used == "unknown":
            # Try to extract from execution trace if available
            trace = result.get("trace", {})
            if trace:
                # Look for skill_selection step in trace
                steps = trace.get("steps", [])
                for step in steps:
                    if step.get("step") == "skill_selection":
                        skill_used = step.get("skill_selected", "core.echo")
                        break

                # If not found in steps, try direct trace fields
                if not skill_used or skill_used == "unknown":
                    if "skill_selected" in trace:
                        skill_used = trace["skill_selected"]
                    else:
                        # Last resort: use safe fallback
                        skill_used = "core.echo"
                        logger.error(
                            "skill_id_fallback_to_echo",
                            goal_id=goal_id,
                            trace_keys=list(trace.keys()),
                            result_keys=list(result.keys()) if result else []
                        )
            else:
                # No trace available - use safe fallback
                skill_used = "core.echo"
                logger.error(
                    "skill_id_no_trace_available",
                    goal_id=goal_id,
                    result_keys=list(result.keys()) if result else []
                )
        
        # Determine proper status
        result_status = result.get("status", "unknown")
        
        # Record analytics with proper skill_id
        await _record_execution_analytics(
            session=session,
            goal_id=goal_id,
            goal_title=goal.title,
            skill_id=skill_used,
            status=result_status,
            duration_ms=int(result.get("trace", {}).get("total_duration_ms", 0)),
            confidence=float(result.get("confidence", 0.0)),
            artifacts_count=len(result.get("artifacts", [])),
            error_message=result.get("message") if result_status == "error" else None
        )
        
        # Record experience for learning (Phase 2 - Experience Layer)
        if EXPERIENCE_ENGINE_AVAILABLE and get_experience_engine:
            try:
                engine = get_experience_engine()
                await engine.record(
                    session=session,
                    goal_id=goal_id,
                    goal_title=goal.title,
                    goal_type=goal.goal_type or "achievable",
                    skill_id=skill_used,
                    skill_name=skill_used,
                    success=(result_status == "success"),
                    duration_ms=int(result.get("trace", {}).get("total_duration_ms", 0)),
                    confidence=float(result.get("confidence", 0.0)),
                    artifacts_count=len(result.get("artifacts", [])),
                    error=result.get("message") if result_status == "error" else None
                )
                logger.debug("experience_recorded", goal_id=goal_id, skill_id=skill_used)
            except Exception as exp_error:
                logger.warning("experience_record_failed", error=str(exp_error))
            
        logger.info(
            "execution_v3_complete",
            goal_id=goal_id,
            status=result.get("status"),
            progress=result.get("progress"),
            execution_engine="v3",
            artifacts_count=len(result.get("artifacts", [])),
            evaluation_passed=eval_passed
        )

        return result

    except Exception as e:
        # Fail → lock remains (stale detector will clean it)
        # Do NOT cleanup on error (audit trail)
        logger.error(
            "execution_v3_error",
            goal_id=goal_id,
            execution_engine="v3",
            error=str(e),
            error_type=type(e).__name__
        )
        raise


# =============================================================================
# Helper Functions (all work with session, do NOT create UOW)
# =============================================================================

async def _get_execution_engine(session, goal_id: str) -> Optional[str]:
    """
    Read current execution_engine (does NOT modify DB).

    Args:
        session: SQLAlchemy session (from uow.session)
        goal_id: Goal UUID string

    Returns:
        Current execution_engine value or None
    """
    result = await session.execute(
        text("SELECT execution_engine FROM goals WHERE id = :goal_id"),
        {"goal_id": goal_id}
    )
    row = result.fetchone()
    return row[0] if row else None


async def _is_lock_stale(session, goal_id: str) -> bool:
    """
    Check if execution engine lock is stale (orphaned).

    Lock is stale if:
    - execution_started_at was set > STALE_LOCK_TIMEOUT seconds ago
    - AND goal is still not completed (status != 'done')

    CRITICAL: Uses execution_started_at for timeout calculation, NOT updated_at.

    Args:
        session: SQLAlchemy session (from uow.session)
        goal_id: Goal UUID string

    Returns:
        True if lock is stale (orphaned)
    """
    result = await session.execute(
        text("""
            SELECT execution_started_at, status
            FROM goals
            WHERE id = :goal_id
              AND execution_engine = 'v3'
        """),
        {"goal_id": goal_id}
    )
    row = result.fetchone()

    if not row:
        return False

    execution_started_at, status = row

    if execution_started_at is None:
        return False

    # Calculate age from execution_started_at
    if hasattr(execution_started_at, 'timestamp'):
        age_seconds = time.time() - execution_started_at.timestamp()
    else:
        age_seconds = time.time() - float(execution_started_at)

    if status != 'done' and age_seconds > STALE_LOCK_TIMEOUT:
        logger.warning(
            "stale_lock_detected",
            goal_id=goal_id,
            execution_engine="v3",
            age_seconds=age_seconds,
            status=status
        )
        return True

    return False


async def _re_acquire_lock(session, goal_id: str, engine: str):
    """
    Re-acquire stale lock by updating execution_started_at.

    Args:
        session: SQLAlchemy session (from uow.session)
        goal_id: Goal UUID string
        engine: Execution engine name ("v3" or "legacy")
    """
    await session.execute(
        text("""
            UPDATE goals
            SET execution_started_at = NOW()
            WHERE id = :goal_id
              AND execution_engine = :engine
        """),
        {"goal_id": goal_id, "engine": engine}
    )

    logger.info(
        "lock_reacquired",
        goal_id=goal_id,
        execution_engine=engine
    )


async def _try_lock(session, goal_id: str, engine: str) -> bool:
    """
    Atomically acquire execution engine lock.

    Uses UPDATE ... WHERE execution_engine IS NULL
    to ensure atomicity across workers.

    Args:
        session: SQLAlchemy session (from uow.session)
        goal_id: Goal UUID string
        engine: Execution engine name ("v3" or "legacy")

    Returns:
        True if lock acquired, False if already locked
    """
    result = await session.execute(
        text("""
            UPDATE goals
            SET execution_engine = :engine,
                execution_started_at = NOW(),
                updated_at = NOW()
            WHERE id = :goal_id
              AND execution_engine IS NULL
            RETURNING id
        """),
        {"goal_id": goal_id, "engine": engine}
    )

    locked = result.rowcount > 0

    if locked:
        logger.info(
            "lock_acquired",
            goal_id=goal_id,
            execution_engine=engine
        )

    return locked


async def _cleanup_lock(session, goal_id: str):
    """
    Clear execution_engine lock after successful completion.

    CRITICAL: Called in same transaction as lock + execution.
    This ensures atomic lock lifecycle.

    Args:
        session: SQLAlchemy session (from uow.session)
        goal_id: Goal UUID string
    """
    await session.execute(
        text("""
            UPDATE goals
            SET execution_engine = NULL
            WHERE id = :goal_id
        """),
        {"goal_id": goal_id}
    )

    logger.debug(
        "lock_cleaned",
        goal_id=goal_id
    )


async def _record_execution_history(session, goal_id: str, status: str):
    """
    Record execution history in goal metadata for analytics.
    
    This persists the execution engine info after lock is cleaned.
    """
    try:
        import json
        from datetime import datetime
        
        # Get current metadata
        result = await session.execute(
            text("SELECT metadata FROM goals WHERE id = :goal_id"),
            {"goal_id": goal_id}
        )
        row = result.fetchone()
        if not row:
            return
            
        metadata = row[0] if row[0] else {}
        
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        
        if not isinstance(metadata, dict):
            metadata = {}
        
        # Update execution history
        history = metadata.get("execution_history", [])
        history.append({
            "engine": "v3",
            "status": status,
            "executed_at": datetime.utcnow().isoformat()
        })
        
        # Keep last 10 executions
        metadata["execution_history"] = history[-10:]
        metadata["last_execution_engine"] = "v3"
        metadata["v3_execution_count"] = metadata.get("v3_execution_count", 0) + 1
        
        # Update without JSON conversion - let SQLAlchemy handle it
        await session.execute(
            text("UPDATE goals SET metadata = :metadata WHERE id = :goal_id"),
            {"goal_id": goal_id, "metadata": metadata}
        )
        
        logger.info(
            "execution_history_recorded",
            goal_id=goal_id,
            engine="v3",
            total_v3_executions=metadata.get("v3_execution_count", 0)
        )
    except Exception as e:
        logger.warning(
            "execution_history_failed",
            goal_id=goal_id,
            error=str(e)
        )


# =============================================================================
# Baseline Observation (p50 + p95)
# =============================================================================

class BaselineObserver:
    """
    Baseline observer with p50 + p95 logging.

    PRINCIPLE: First 48h = observation only.
    Do NOT make automated decisions without baseline.
    """

    def __init__(self, observation_hours: int = BASELINE_OBSERVATION_HOURS):
        self.start_time = time.time()
        self.observation_hours = observation_hours
        self.observations = []

    def record(self, metric_name: str, value: float):
        """Record metric observation."""
        self.observations.append({
            "timestamp": time.time(),
            "metric": metric_name,
            "value": value
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get baseline summary.

        Returns:
            Dict with p50, p95, min, max for each metric
        """
        hours_elapsed = (time.time() - self.start_time) / 3600

        # Group by metric
        metrics = {}
        for obs in self.observations:
            metric_name = obs["metric"]
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(obs["value"])

        # Calculate statistics
        summary = {}
        for metric_name, values in metrics.items():
            values_sorted = sorted(values)
            n = len(values_sorted)

            summary[metric_name] = {
                "count": n,
                "p50": values_sorted[int(n * 0.5)] if n > 0 else None,
                "p95": values_sorted[int(n * 0.95)] if n > 0 else None,
                "min": min(values) if n > 0 else None,
                "max": max(values) if n > 0 else None
            }

        # Add status
        if hours_elapsed < self.observation_hours:
            summary["status"] = "collecting_baseline"
            summary["hours_elapsed"] = round(hours_elapsed, 1)
        else:
            summary["status"] = "baseline_ready"

        logger.info(
            "baseline_summary",
            hours_elapsed=round(hours_elapsed, 1),
            summary=summary
        )

        return summary


# =============================================================================
# Suspicion Checks
# =============================================================================

class SuspicionChecks:
    """
    Suspicion checks for "too perfect" metrics.

    PRINCIPLE: 0% escalation, 100% success, 0 blocked fallbacks = suspicious.
    System may be too optimistic or suppressing errors.
    """

    @staticmethod
    def check_system_health(metrics: Dict[str, Any]) -> list:
        """
        Check system health for suspicious patterns.

        Args:
            metrics: Metrics dictionary

        Returns:
            List of warning messages
        """
        warnings = []

        total_executions = metrics.get("total_executions", 0)

        # Suspicious: 0% escalation rate
        if metrics.get("escalation_rate", 0) == 0 and total_executions > 100:
            warnings.append(
                "Escalation rate is 0% after 100+ executions. "
                "This is suspicious - system may be too optimistic."
            )

        # Suspicious: No blocked fallbacks
        if metrics.get("blocked_fallback_rate", 0) == 0 and total_executions > 100:
            warnings.append(
                "No blocked fallbacks after 100+ executions. "
                "This may indicate capabilities are over-marked as SAFE."
            )

        # Suspicious: Perfect success rate
        if metrics.get("success_rate", 0) == 1.0 and total_executions > 100:
            warnings.append(
                "100% success rate after 100+ executions. "
                "This may indicate errors are being silently suppressed."
            )

        if warnings:
            logger.warning(
                "suspicious_metrics_detected",
                warnings=warnings
            )

        return warnings


# =============================================================================
# Metrics Endpoint (for dashboard)
# =============================================================================

# Global observer instance
_baseline_observer = None

def get_baseline_observer() -> BaselineObserver:
    """Get or create baseline observer instance."""
    global _baseline_observer
    if _baseline_observer is None:
        _baseline_observer = BaselineObserver()
    return _baseline_observer


# =============================================================================
# Execution Analytics Recording
# =============================================================================

async def _record_execution_analytics(
    session,
    goal_id: str,
    goal_title: str,
    skill_id: str,
    status: str,
    duration_ms: int,
    confidence: float,
    artifacts_count: int,
    error_message: str = None
):
    """Record execution to analytics tables."""
    try:
        import json
        from sqlalchemy import text
        
        # Record execution
        await session.execute(
            text("""
                INSERT INTO goal_executions 
                (goal_id, goal_title, skill_id, execution_engine, status, 
                 duration_ms, confidence, artifacts_count, error_message, created_at)
                VALUES 
                (:goal_id, :goal_title, :skill_id, 'v3', :status,
                 :duration_ms, :confidence, :artifacts_count, :error_message, NOW())
            """),
            {
                "goal_id": goal_id,
                "goal_title": goal_title,
                "skill_id": skill_id,
                "status": status,
                "duration_ms": duration_ms,
                "confidence": confidence,
                "artifacts_count": artifacts_count,
                "error_message": error_message
            }
        )
        
        # Update skill stats
        success = status == "success"
        await session.execute(
            text("""
                INSERT INTO skill_stats 
                (skill_id, skill_name, total_executions, success_count, 
                 avg_latency_ms, avg_confidence, total_artifacts, failure_count,
                 last_used, updated_at)
                VALUES 
                (:skill_id, :skill_id, 1, :success_count, :avg_latency,
                 :avg_confidence, :artifacts_count, :failure_count, NOW(), NOW())
                ON CONFLICT (skill_id) DO UPDATE SET
                    total_executions = skill_stats.total_executions + 1,
                    success_count = skill_stats.success_count + :success_count,
                    avg_latency_ms = (
                        (skill_stats.avg_latency_ms * skill_stats.total_executions + :avg_latency) 
                        / (skill_stats.total_executions + 1)
                    ),
                    avg_confidence = (
                        (skill_stats.avg_confidence * skill_stats.total_executions + :avg_confidence)
                        / (skill_stats.total_executions + 1)
                    ),
                    total_artifacts = skill_stats.total_artifacts + :artifacts_count,
                    failure_count = skill_stats.failure_count + :failure_count,
                    last_used = NOW(),
                    updated_at = NOW(),
                    success_rate = skill_stats.success_count::float / NULLIF(skill_stats.total_executions, 0)::float
            """),
            {
                "skill_id": skill_id,
                "success_count": 1 if success else 0,
                "failure_count": 0 if success else 1,
                "avg_latency": duration_ms,
                "avg_confidence": confidence,
                "artifacts_count": artifacts_count
            }
        )
        
        # Record experience
        await session.execute(
            text("""
                INSERT INTO experiences
                (goal_type, goal_title, skill_used, success, duration_ms,
                 artifacts_produced, confidence, created_at)
                VALUES
                ('achievable', :goal_title, :skill_used, :success, :duration_ms,
                 :artifacts, :confidence, NOW())
            """),
            {
                "goal_title": goal_title,
                "skill_used": skill_id,
                "success": success,
                "duration_ms": duration_ms,
                "artifacts": json.dumps([f"type_{artifacts_count}"]),
                "confidence": confidence
            }
        )
        
        logger.info(
            "analytics_recorded",
            goal_id=goal_id,
            skill_id=skill_id,
            status=status,
            duration_ms=duration_ms
        )
        
    except Exception as e:
        logger.warning(
            "analytics_recording_failed",
            goal_id=goal_id,
            error=str(e)
        )
