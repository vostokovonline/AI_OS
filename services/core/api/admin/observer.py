"""
Observer Admin API (Phase 2.3.3)

Admin endpoints для Lifecycle Observer.

Read-only API для:
- Запуска проверок on-demand
- Получения статуса здоровья
- Проверки конкретных сущностей
- Получения списка инвариантов

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from observer_engine import observer_engine
from lifecycle_observer import invariant_registry

# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ObserverRunRequest(BaseModel):
    """Request body for on-demand observer run"""
    invariant_ids: Optional[List[str]] = Field(
        None,
        description="List of invariant IDs to check (None = all)"
    )
    limit: int = Field(
        1000,
        ge=1,
        le=10000,
        description="Max entities to check per invariant"
    )


class ObserverEntityCheckRequest(BaseModel):
    """Request body for checking specific entity"""
    invariant_id: str = Field(..., description="Invariant ID (I7, I9, ORPHANS)")
    entity_id: str = Field(..., description="Entity UUID")


class ObserverHealthResponse(BaseModel):
    """Health status response"""
    status: str  # HEALTHY | DEGRADED | CRITICAL
    invariants_checked: int
    entities_sampled: int
    violations_detected: int
    violation_details: List[dict]
    checked_at: str


class InvariantListResponse(BaseModel):
    """List of all registered invariants"""
    invariants: List[dict]
    total_count: int


class ObserverRunResponse(BaseModel):
    """Observer run result"""
    run_id: str
    started_at: str
    completed_at: str
    duration_seconds: float
    total_invariants: int
    total_entities_checked: int
    violations_detected: int
    violation_details: List[dict]
    summary: dict


# =============================================================================
# Router
# =============================================================================

router = APIRouter()


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/observer/run",
    response_model=ObserverRunResponse,
    status_code=200,
    tags=["Admin", "Observer"],
    summary="Run on-demand invariant checks",
    description="Run observer on-demand. Returns full report with all violations."
)
async def run_observer(request: ObserverRunRequest):
    """
    Run observer on-demand.

    Checks all invariants (or specific ones) against all entities.

    ## Response
    - Full run statistics
    - All violation details
    - Summary with overall status

    ## Example
    ```bash
    curl -X POST http://localhost:8000/observer/run \\
      -H "Content-Type: application/json" \\
      -d '{"invariant_ids": ["I7", "I9"], "limit": 100}'
    ```
    """
    try:
        logger.info(f"Starting on-demand observer run: invariants={request.invariant_ids}, limit={request.limit}")

        # Run observer
        result = await observer_engine.run_on_demand(
            invariant_ids=request.invariant_ids,
            limit=request.limit
        )

        # Log violations if any
        if result.violation_reports:
            logger.warning(
                f"Observer run {result.run_id}: "
                f"{len(result.violation_reports)} violations detected"
            )
            for violation in result.violation_reports:
                logger.warning(
                    f"  [{violation.invariant_id}] {violation.entity}:{violation.entity_id} "
                    f"- {violation.message}"
                )
        else:
            logger.info(f"Observer run {result.run_id}: ALL HEALTHY")

        return result.to_dict()

    except Exception as e:
        logger.error(f"Observer run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/observer/health",
    response_model=ObserverHealthResponse,
    status_code=200,
    tags=["Admin", "Observer"],
    summary="Quick health check",
    description="Fast health check using sampled entities. Returns status + violations."
)
async def get_observer_health():
    """
    Quick health check.

    Samples 10 goals + 5 approvals to quickly assess system health.
    Use for monitoring dashboards and periodic checks.

    ## Response
    - status: HEALTHY | DEGRADED | CRITICAL
    - Sampled entities count
    - Violations detected (first 5)

    ## Example
    ```bash
    curl http://localhost:8000/observer/health
    ```
    """
    try:
        health = await observer_engine.get_health_status()
        return health

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/observer/invariants",
    response_model=InvariantListResponse,
    status_code=200,
    tags=["Admin", "Observer"],
    summary="List all registered invariants",
    description="Returns list of all available invariants with metadata."
)
async def list_invariants():
    """
    List all registered invariants.

    Returns metadata for each invariant:
    - invariant_id
    - invariant_name
    - severity (HARD/SOFT)
    - description

    ## Example
    ```bash
    curl http://localhost:8000/observer/invariants
    ```
    """
    try:
        invariant_ids = invariant_registry.list_invariants()

        invariants = []
        for inv_id in invariant_ids:
            invariant = invariant_registry.get_invariant(inv_id)
            invariants.append({
                "invariant_id": invariant.invariant_id,
                "invariant_name": invariant.invariant_name,
                "severity": invariant.severity.value,
                "description": invariant.__doc__ or "No description"
            })

        return {
            "invariants": invariants,
            "total_count": len(invariants)
        }

    except Exception as e:
        logger.error(f"Failed to list invariants: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/observer/check/entity",
    response_model=dict,
    status_code=200,
    tags=["Admin", "Observer"],
    summary="Check specific entity",
    description="Check specific entity against specific invariant."
)
async def check_entity(request: ObserverEntityCheckRequest):
    """
    Check specific entity against invariant.

    Validates one entity against one invariant.
    Useful for debugging and troubleshooting.

    ## Args
    - invariant_id: I7, I9, ORPHANS
    - entity_id: UUID of goal or approval

    ## Example
    ```bash
    curl -X POST http://localhost:8000/observer/check/entity \\
      -H "Content-Type: application/json" \\
      -d '{"invariant_id": "I7", "entity_id": "123e4567-e89b-12d3-a456-426614174000"}'
    ```
    """
    try:
        # Parse entity_id
        try:
            entity_uuid = uuid.UUID(request.entity_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")

        # Check entity
        report = await observer_engine.check_specific_entity(
            invariant_id=request.invariant_id,
            entity_id=entity_uuid
        )

        if not report:
            raise HTTPException(
                status_code=404,
                detail=f"Invariant not found: {request.invariant_id}"
            )

        # Log result
        if report.passed:
            logger.info(f"Entity check PASSED: {request.invariant_id} for {request.entity_id}")
        else:
            logger.warning(
                f"Entity check VIOLATED: {request.invariant_id} for {request.entity_id} "
                f"- {report.message}"
            )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/observer/check/goal/{goal_id}",
    response_model=dict,
    status_code=200,
    tags=["Admin", "Observer"],
    summary="Check all invariants for goal",
    description="Check all applicable invariants for a specific goal."
)
async def check_goal(goal_id: str):
    """
    Check all applicable invariants for goal.

    Automatically determines which invariants apply:
    - I7: if MANUAL goal
    - ORPHANS: if non-atomic parent

    ## Example
    ```bash
    curl http://localhost:8000/observer/check/goal/123e4567-e89b-12d3-a456-426614174000
    ```
    """
    try:
        # Parse goal_id
        try:
            goal_uuid = uuid.UUID(goal_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")

        # Check goal
        reports = await observer_engine.check_goal(goal_uuid)

        return {
            "goal_id": goal_id,
            "invariants_checked": list(reports.keys()),
            "results": {inv_id: report.to_dict() for inv_id, report in reports.items()},
            "overall_status": "HEALTHY" if all(r.passed for r in reports.values()) else "VIOLATIONS_DETECTED"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Goal check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/observer/check/approval/{approval_id}",
    response_model=dict,
    status_code=200,
    tags=["Admin", "Observer"],
    summary="Check all invariants for approval",
    description="Check all applicable invariants for a specific approval."
)
async def check_approval(approval_id: str):
    """
    Check all applicable invariants for approval.

    Currently checks:
    - I9: approval must exist only for DONE goals

    ## Example
    ```bash
    curl http://localhost:8000/observer/check/approval/123e4567-e89b-12d3-a456-426614174000
    ```
    """
    try:
        # Parse approval_id
        try:
            approval_uuid = uuid.UUID(approval_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid UUID format")

        # Check approval
        reports = await observer_engine.check_approval(approval_uuid)

        return {
            "approval_id": approval_id,
            "invariants_checked": list(reports.keys()),
            "results": {inv_id: report.to_dict() for inv_id, report in reports.items()},
            "overall_status": "HEALTHY" if all(r.passed for r in reports.values()) else "VIOLATIONS_DETECTED"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approval check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
