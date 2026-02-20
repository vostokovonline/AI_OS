"""
Reflection Admin API (Phase 2.4.5)

Admin endpoints для системы рефлексии.

Use cases:
- Manual reflection trigger
- Audit decisions
- System health check
- List policies

Author: AI-OS Core Team
Date: 2026-02-06
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from reflection_loop import reflection_loop, reflect_on_manual_trigger, reflect_on_health_check
from reflection_policies import policy_engine
from reflection_system import TriggerSource, create_trigger_id


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Schemas
# =============================================================================

class ReflectionManualTriggerRequest(BaseModel):
    """Request body for manual reflection trigger"""
    reason: str = Field(..., description="Reason for manual trigger", min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class ReflectionSignalRequest(BaseModel):
    """Request body for custom reflection signal"""
    source: TriggerSource = Field(..., description="Signal source")
    invariant_id: Optional[str] = Field(None, description="Invariant ID (if observer)")
    entity: Optional[str] = Field(None, description="Entity type")
    entity_id: Optional[str] = Field(None, description="Entity UUID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Signal context")


class ReflectionRunResponse(BaseModel):
    """Response from reflection run"""
    trigger_id: str
    success: bool
    decisions_count: int
    decisions: List[dict]
    actions_executed: int
    action_results: List[dict]
    duration_seconds: float


class ReflectionHealthResponse(BaseModel):
    """Reflection system health"""
    status: str  # HEALTHY | ACTIVE | ERROR
    policies_registered: int
    last_check: str
    last_run_summary: Optional[Dict[str, Any]] = None


class PolicyListResponse(BaseModel):
    """List of registered policies"""
    policies: List[str]
    total_count: int


# =============================================================================
# Router
# =============================================================================

router = APIRouter()


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/reflection/run",
    response_model=ReflectionRunResponse,
    status_code=200,
    tags=["Admin", "Reflection"],
    summary="Run reflection manually",
    description="Trigger reflection loop manually with reason and context."
)
async def run_reflection_manually(request: ReflectionManualTriggerRequest):
    """
    Run reflection manually.

    Use for testing, debugging, or manual intervention.

    ## Example
    ```bash
    curl -X POST http://localhost:8000/reflection/run \\
      -H "Content-Type: application/json" \\
      -d '{"reason": "Test reflection", "context": {"test": true}}'
    ```
    """
    try:
        logger.info(f"Manual reflection trigger: {request.reason}")

        # Run reflection
        result = await reflect_on_manual_trigger(request.reason, request.context)

        return result.to_dict()

    except Exception as e:
        logger.error(f"Manual reflection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/reflection/signal",
    response_model=ReflectionRunResponse,
    status_code=200,
    tags=["Admin", "Reflection"],
    summary="Send custom reflection signal",
    description="Send custom signal to reflection system."
)
async def send_reflection_signal(request: ReflectionSignalRequest):
    """
    Send custom reflection signal.

    Allows sending custom signals (e.g., from external systems).

    ## Example
    ```bash
    curl -X POST http://localhost:8000/reflection/signal \\
      -H "Content-Type: application/json" \\
      -d '{
        "source": "external",
        "entity": "goal",
        "entity_id": "123e4567-e89b-12d3-a456-426614174000",
        "context": {"event": "deployment"}
      }'
    ```
    """
    try:
        # Create signal
        from reflection_system import ReflectionSignal
        import uuid

        signal = ReflectionSignal(
            source=request.source,
            trigger_id=create_trigger_id(),
            invariant_id=request.invariant_id,
            entity=request.entity,
            entity_id=uuid.UUID(request.entity_id) if request.entity_id else None,
            external_context=request.context
        )

        # Run reflection
        result = await reflection_loop.run(signal)

        return result.to_dict()

    except Exception as e:
        logger.error(f"Signal reflection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/reflection/health",
    response_model=ReflectionHealthResponse,
    status_code=200,
    tags=["Admin", "Reflection"],
    summary="Reflection system health check",
    description="Check reflection system health and run quick observer check."
)
async def get_reflection_health():
    """
    Reflection system health check.

    Runs quick observer check and returns reflection system status.

    ## Response
    - status: HEALTHY | ACTIVE | ERROR
    - policies_registered: count
    - last_check: timestamp
    """
    try:
        # Run health check reflection
        result = await reflect_on_health_check()

        status = "HEALTHY"
        if len(result.decisions) > 0:
            status = "ACTIVE"
        if not result.success:
            status = "ERROR"

        return {
            "status": status,
            "policies_registered": len(policy_engine.list_policies()),
            "last_check": result.completed_at.isoformat(),
            "last_run_summary": {
                "trigger_id": result.trigger_id,
                "decisions_count": len(result.decisions),
                "actions_executed": len(result.action_results),
                "duration_seconds": result.duration_seconds
            }
        }

    except Exception as e:
        logger.error(f"Reflection health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/reflection/policies",
    response_model=PolicyListResponse,
    status_code=200,
    tags=["Admin", "Reflection"],
    summary="List registered policies",
    description="Get list of all registered reflection policies."
)
async def list_policies():
    """
    List all registered reflection policies.

    Returns policy IDs and metadata.

    ## Example
    ```bash
    curl http://localhost:8000/reflection/policies
    ```
    """
    try:
        policy_ids = policy_engine.list_policies()

        # Get policy metadata (priority, etc.)
        policies_info = []
        for policy in policy_engine._policies:
            policies_info.append({
                "policy_id": policy.policy_id,
                "priority": policy.priority
            })

        return {
            "policies": [p["policy_id"] for p in policies_info],
            "policies_with_priority": policies_info,
            "total_count": len(policy_ids)
        }

    except Exception as e:
        logger.error(f"Failed to list policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/reflection/actions",
    response_model=Dict[str, str],
    status_code=200,
    tags=["Admin", "Reflection"],
    summary="List available actions",
    description="Get list of all available reflection action types."
)
async def list_actions():
    """
    List all available reflection actions.

    Returns action types and descriptions.

    ## Example
    ```bash
    curl http://localhost:8000/reflection/actions
    ```
    """
    try:
        from reflection_system import ActionType

        actions = {
            "annotate": "Add metadata/annotation to entity",
            "freeze": "Freeze entity (block execution)",
            "request_review": "Flag entity for human review",
            "spawn_analysis": "Create analysis goal/task",
            "log": "Log to audit trail (no action)",
            "escalate": "Escalate priority/severity",
            "schedule_reevaluation": "Schedule re-evaluation after delay"
        }

        return actions

    except Exception as e:
        logger.error(f"Failed to list actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/reflection/test/violation",
    response_model=ReflectionRunResponse,
    status_code=200,
    tags=["Admin", "Reflection", "Test"],
    summary="Test reflection with fake violation",
    description="Create fake observer violation and test reflection response."
)
async def test_reflection_with_violation():
    """
    Test reflection with fake HARD invariant violation.

    Useful for testing reflection policies and actions.

    ## Example
    ```bash
    curl -X POST http://localhost:8000/reflection/test/violation
    ```
    """
    try:
        import uuid
        from lifecycle_observer import InvariantViolationReport, InvariantSeverity

        # Create fake violation report
        fake_goal_id = uuid.uuid4()

        report = InvariantViolationReport(
            invariant_id="I7",
            invariant_name="DONE goal must have exactly one approval",
            entity="goal",
            entity_id=fake_goal_id,
            severity=InvariantSeverity.HARD,
            passed=False,
            details={
                "approval_count": 0,
                "expected": 1,
                "goal_status": "done",
                "completion_mode": "manual"
            },
            message="❌ VIOLATED: DONE goal has no approval (TEST)"
        )

        # Run reflection
        result = await reflection_loop.run_from_observer_report(report)

        return result.to_dict()

    except Exception as e:
        logger.error(f"Test reflection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
