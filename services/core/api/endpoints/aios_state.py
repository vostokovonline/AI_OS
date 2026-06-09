"""API endpoint for AIOSState — single entry point for the cockpit."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from aios_state_builder import aios_state_builder

router = APIRouter(prefix="/aios", tags=["aios"])


@router.get("/state")
async def get_aios_state(session: AsyncSession = Depends(get_db)):
    """Return the full AIOSState — goal states, risks, priorities."""
    state = await aios_state_builder.build(session)
    return state.to_dict()
