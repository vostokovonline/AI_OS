"""
Goal Approval API Controller (Phase 2.2.5)

Thin wrapper для approve_completion service.

Author: AI-OS Core Team
Date: 2026-02-06
"""

import uuid
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from database import get_db
from schemas import GoalApproveRequest, GoalApprovalSuccess
from services.goals.approve_completion import approve_completion_service
from exceptions import (
    BaseGoalException,
    GoalAlreadyDone,
    GoalChildrenIncomplete,
    InsufficientAuthority,
    InvalidGoalState,
    InvalidCompletionMode,
    InvariantViolation,
    EXCEPTION_TO_STATUS,
)


router = APIRouter()


def map_exception_to_http(exc: BaseGoalException) -> HTTPException:
    """
    Map domain exception to HTTP response.

    Args:
        exc: Domain exception from service layer

    Returns:
        HTTPException with proper status code and structured error payload
    """
    exception_class = type(exc)
    status_code = EXCEPTION_TO_STATUS.get(exception_class, 500)

    # Extract error code from exception class name
    error_code = exception_class.__name__

    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": error_code,
                "message": exc.message,
                "details": exc.details
            }
        }
    )


@router.post(
    "/goals/{goal_id}/approve_completion",
    response_model=GoalApprovalSuccess,
    status_code=200,
    responses={
        400: {"model": dict, "description": "Invalid completion mode"},
        403: {"model": dict, "description": "Insufficient authority"},
        404: {"model": dict, "description": "Goal not found"},
        409: {"model": dict, "description": "Invalid state or already approved"},
        500: {"model": dict, "description": "Invariant violation"},
    },
    tags=["Goals"],
    summary="Approve MANUAL goal completion",
    description="Explicit human/authority approval for MANUAL goal completion. Irreversible operation."
)
async def approve_completion(
    goal_id: uuid.UUID,
    payload: GoalApproveRequest,
    db: AsyncSession = Depends(get_db)
) -> GoalApprovalSuccess:
    """
    Approve completion of a MANUAL goal.

    ## Preconditions
    - goal.completion_mode == MANUAL
    - goal.status == ACTIVE
    - If parent: all children must be DONE
    - Approval must not already exist

    ## Effects
    - Inserts approval record (UNIQUE constraint)
    - Updates goal.status = DONE
    - Sets goal.progress = 1.0
    - Sets goal.completed_at = now()

    ## Error Codes
    - 400: Not MANUAL mode
    - 403: Authority level insufficient
    - 404: Goal not found
    - 409: Invalid state / already approved / children incomplete
    - 500: Invariant violation (hard stop)

    ## Idempotency
    Operation is NOT idempotent by HTTP semantics,
    but state-safe: retry returns 409 GOAL_ALREADY_DONE
    with deterministic error details.
    """
    try:
        # Call service (transaction boundary inside)
        result = await approve_completion_service.approve_goal_completion(
            goal_id=goal_id,
            approved_by=payload.approved_by,
            authority_level=payload.authority_level,
            comment=payload.comment
        )

        # Return success response
        return GoalApprovalSuccess(
            goal_id=result["goal_id"],
            status="done",
            approved_at=result["approved_at"],
            approved_by=result["approved_by"],
            authority_level=result["authority_level"]
        )

    except BaseGoalException as e:
        # Map domain exception to HTTP response
        raise map_exception_to_http(e)
