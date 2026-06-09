"""
Control Center API - Real-time metrics for AI-OS observability.

Provides unified metrics endpoints for dashboard v2.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

# Stub for metrics_engine - placeholder
class MetricsEngineStub:
    pass

def get_metrics_engine():
    return MetricsEngineStub()

# from metrics_engine import get_metrics_engine, MetricsEngineStub
# from metrics_engine.models import (
#     SystemMetrics,
#     GoalMetricsSnapshot,
#     ExecutionMetrics,
#     CognitionMetrics,
#     SkillMetric
# )

router = APIRouter(prefix="/control", tags=["control-center"])


async def get_metrics() -> MetricsEngineStub:
    """Dependency to get metrics engine instance"""
    return get_metrics_engine()


@router.get("/system")
async def get_system_metrics(
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    System health overview.

    Returns current system state including:
    - Goal counts (running, completed, failed)
    - LLM usage (calls, tokens)
    - Thinking depth (fast vs deep)
    - Failure rate
    """
    system = await metrics.get_system_metrics()

    # Calculate failure rate
    total_finished = system.goals_completed + system.goals_failed
    failure_rate = system.goals_failed / total_finished if total_finished > 0 else 0.0

    return {
        "goals_running": system.goals_running,
        "goals_completed": system.goals_completed,
        "goals_failed": system.goals_failed,
        "llm_calls": system.llm_calls,
        "llm_tokens": system.llm_tokens,
        "fast_decisions": system.fast_decisions,
        "deep_reasoning": system.deep_reasoning,
        "failure_rate": round(failure_rate, 3),
        "timestamp": system.timestamp.isoformat()
    }


@router.get("/goals")
async def get_goal_metrics(
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    Goal economy metrics.

    Returns goal lifecycle data including:
    - Pending, running, blocked, completed, failed counts
    - Average completion time
    - Success rate
    """
    goals = await metrics.get_goal_metrics()

    return {
        "pending": goals.pending,
        "running": goals.running,
        "blocked": goals.blocked,
        "completed": goals.completed,
        "failed": goals.failed,
        "avg_completion_time": round(goals.avg_completion_time, 2),
        "success_rate": round(goals.success_rate, 3)
    }


@router.get("/execution")
async def get_execution_metrics(
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    Execution layer metrics.

    Returns agent and skill performance data:
    - Active agents
    - Skills invoked (total)
    - Artifacts produced
    - Average execution time
    - Throughput (goals/min)
    """
    execution = await metrics.get_execution_metrics()

    return {
        "active_agents": execution.active_agents,
        "skills_invoked": execution.skills_invoked,
        "artifacts_produced": execution.artifacts_produced,
        "avg_execution_time": round(execution.avg_execution_time, 2),
        "throughput_per_min": round(execution.throughput_per_min, 2)
    }


@router.get("/skills")
async def get_skill_metrics(
    limit: int = 10,
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    Skill performance metrics.

    Returns top skills by usage:
    - Total skills count
    - Most used skills with success rates
    - Failing skills
    """
    skills = await metrics.get_skill_metrics(limit=limit)

    return {
        "total_skills": len(skills),
        "most_used": [
            {
                "skill": s.skill_id,
                "usage": s.usage,
                "success_rate": round(s.success_rate, 3),
                "failures": s.failures
            }
            for s in skills[:5]
        ],
        "failing_skills": [
            {
                "skill": s.skill_id,
                "failures": s.failures
            }
            for s in skills if s.failures > 0
        ]
    }


@router.get("/cognition")
async def get_cognition_metrics(
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    Thinking depth metrics.

    Returns cognitive load analysis:
    - Fast vs deep decision counts
    - Percentages
    - Average tokens per call
    """
    cognition = await metrics.get_cognition_metrics()

    return {
        "total_decisions": cognition.total_decisions,
        "fast_decisions": cognition.fast_decisions,
        "deep_reasoning": cognition.deep_reasoning,
        "fast_percentage": round(cognition.fast_percentage * 100, 1),
        "deep_percentage": round(cognition.deep_percentage * 100, 1),
        "avg_tokens_per_call": round(cognition.avg_tokens_per_call, 1)
    }


@router.get("/strategy-reuse")
async def get_strategy_reuse_metrics(
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    Strategy Reuse Rate (Phase 1.1) - CRITICAL AUTONOMY METRIC

    Shows if AI-OS is becoming smarter:
    - High reuse rate = system is learning and reusing knowledge
    - Low reuse rate = system is exploring or not learning

    This is THE metric for self-improving systems.
    """
    return await metrics.get_strategy_reuse_metrics()


@router.get("/overview")
async def get_control_center_overview(
    metrics: MetricsEngineStub = Depends(get_metrics)
) -> dict:
    """
    Combined overview for Control Center dashboard.

    Single endpoint that aggregates all key metrics for the main dashboard view.
    """
    system = await metrics.get_system_metrics()
    goals = await metrics.get_goal_metrics()
    execution = await metrics.get_execution_metrics()
    cognition = await metrics.get_cognition_metrics()
    skills = await metrics.get_skill_metrics(limit=5)

    return {
        "system": {
            "llm_calls": system.llm_calls,
            "llm_tokens": system.llm_tokens,
            "failure_rate": round(
                system.goals_failed / (system.goals_completed + system.goals_failed)
                if (system.goals_completed + system.goals_failed) > 0 else 0.0,
                3
            )
        },
        "goals": {
            "running": goals.running,
            "completed": goals.completed,
            "success_rate": round(goals.success_rate, 3),
            "throughput_per_min": round(execution.throughput_per_min, 2)  # Phase 1.1
        },
        "execution": {
            "skills_invoked": execution.skills_invoked,
            "artifacts_produced": execution.artifacts_produced,
            "throughput": round(execution.throughput_per_min, 2)
        },
        "cognition": {
            "fast_percentage": round(cognition.fast_percentage * 100, 1),
            "deep_percentage": round(cognition.deep_percentage * 100, 1),
            "avg_tokens": round(cognition.avg_tokens_per_call, 1)
        },
        "top_skills": [
            {
                "skill": s.skill_id,
                "usage": s.usage,
                "success_rate": round(s.success_rate, 3)
            }
            for s in skills[:5]
        ]
    }


@router.get("/trace/{goal_id}")
async def get_execution_trace(goal_id: str):
    """
    Get execution trace for a specific goal.

    Returns full timeline of goal execution:
    - Activation timestamp
    - Skills executed  
    - Artifacts created
    - Completion timestamp
    - All intermediate steps

    This is the foundational data for Strategy Evolution Engine.
    """
    from sqlalchemy import select
    from database import AsyncSessionLocal
    from uuid import UUID
    from models import Goal

    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            return {
                "error": "Goal not found",
                "goal_id": goal_id
            }

        # Parse execution trace if exists
        trace = goal.execution_trace or {}
        steps = trace.get("steps", []) if isinstance(trace, dict) else []

        # Enrich trace with artifacts data
        from models import Artifact
        artifact_stmt = select(Artifact).where(Artifact.goal_id == UUID(goal_id))
        artifact_result = await db.execute(artifact_stmt)
        artifacts = artifact_result.scalars().all()

        artifacts_list = [
            {
                "id": str(a.id),
                "type": a.type,
                "content_kind": a.content_kind,
                "verification_status": a.verification_status,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in artifacts
        ]

        return {
            "goal_id": goal_id,
            "title": goal.title,
            "status": goal.status,
            "goal_type": goal.goal_type,
            "is_atomic": goal.is_atomic,
            "progress": goal.progress,
            "created_at": goal.created_at.isoformat() if goal.created_at else None,
            "completed_at": goal.completed_at.isoformat() if goal.completed_at else None,
            "execution_trace": {
                "steps": steps,
                "total_steps": len(steps),
                "artifacts": artifacts_list,
                "artifacts_count": len(artifacts_list)
            },
            "evaluation": goal.evaluation_result if hasattr(goal, 'evaluation_result') else None
        }
