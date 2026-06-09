"""AIOS Router — single endpoint, zero dirty imports.

Only imports:
  - database.get_db
  - aios.state_builder.aios_state_builder

Does NOT import: models, goal_executor_v2, experience, epistemic,
cognitive_os, simulation, phase18, or any broken module.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from aios.state_builder import aios_state_builder

router = APIRouter(prefix="/aios", tags=["aios"])


@router.get("/state")
async def get_aios_state(session: AsyncSession = Depends(get_db)):
    """Return the full AIOSState — goals, risks, priorities, executions, events."""
    state = await aios_state_builder.build(session)
    return state.to_dict()
