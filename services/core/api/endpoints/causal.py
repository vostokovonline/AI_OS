"""
Causal API — Causality Bridge, CPE, and PHE endpoints.

Endpoints:
  Bridge:
    GET  /causal/bridge/stats         — bridge statistics
    GET  /causal/bridge/graph         — causality graph info
    GET  /causal/bridge/edges/{goal_id} — edges for a goal
    POST /causal/bridge/propagate     — manually propagate an execution event
    GET  /causal/bridge/adjustments   — current dispatch adjustments
    GET  /causal/bridge/consistency   — causal consistency check

  CPE:
    POST /causal/cpe/simulate         — simulate a single event
    POST /causal/cpe/evaluate         — evaluate candidate actions
    POST /causal/cpe/decide           — full propose + evaluate cycle
    GET  /causal/cpe/stats            — CPE diagnostics

  PHE:
    POST /causal/phe/plan             — strategic multi-step planning
    POST /causal/phe/simulate-depth   — compare planning at different depths
    GET  /causal/phe/stats            — PHE diagnostics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from epistemic_factory import (
    get_causality_bridge,
    get_causal_policy_engine,
    get_policy_horizon_engine,
)

router = APIRouter(prefix="/causal", tags=["causal"])


# ── Request Models ──

class PropagateRequest(BaseModel):
    event_type: str
    entry_id: str
    goal_id: str
    lease_id: str = ""
    success: Optional[bool] = None
    duration_ms: float = 0.0
    error: str = ""
    context: dict = {}


class SimulateRequest(BaseModel):
    event_type: str
    goal_id: str = "simulated"
    success: Optional[bool] = None
    label: str = "sim"


class EvaluateRequest(BaseModel):
    goal_id: str
    options: Optional[List[Dict[str, Any]]] = None
    threshold: float = 0.5


class DecideRequest(BaseModel):
    goal_id: str
    options: Optional[List[Dict[str, Any]]] = None
    threshold: float = 0.5


class PlanRequest(BaseModel):
    goal_id: str
    options: Optional[List[Dict[str, Any]]] = None
    horizon: int = 3
    beam_width: int = 3


class SimulateDepthRequest(BaseModel):
    goal_id: str
    depths: List[int] = [1, 2, 3]
    options: Optional[List[Dict[str, Any]]] = None
    beam_width: int = 3


# ════════════════════════════════════════════════════════════════
# BRIDGE ENDPOINTS
# ════════════════════════════════════════════════════════════════

@router.get("/bridge/stats")
async def bridge_stats():
    """Get CausalityBridge diagnostics."""
    return get_causality_bridge().get_causal_stats()


@router.get("/bridge/graph")
async def bridge_graph():
    """Get causality graph summary."""
    bridge = get_causality_bridge()
    return {
        "total_edges": bridge.graph.count(),
        "by_goal_count": len(bridge.graph._by_goal),
    }


@router.get("/bridge/edges/{goal_id}")
async def bridge_edges(goal_id: str):
    """Get all causality edges for a goal."""
    return get_causality_bridge().get_causal_chain(goal_id)


@router.post("/bridge/propagate")
async def bridge_propagate(req: PropagateRequest):
    """Manually propagate an execution event into the epistemic kernel."""
    bridge = get_causality_bridge()
    result = bridge.on_execution_event(
        event_type=req.event_type,
        entry_id=req.entry_id,
        goal_id=req.goal_id,
        lease_id=req.lease_id,
        success=req.success,
        duration_ms=req.duration_ms,
        error=req.error or None,
        context=req.context,
    )
    return result


@router.get("/bridge/adjustments")
async def bridge_adjustments():
    """Get current dispatch adjustments from epistemic state."""
    return get_causality_bridge().get_dispatch_adjustments()


@router.get("/bridge/consistency")
async def bridge_consistency():
    """Run cross-kernel causal consistency check."""
    return get_causality_bridge().check_causal_consistency()


# ════════════════════════════════════════════════════════════════
# CPE ENDPOINTS
# ════════════════════════════════════════════════════════════════

@router.post("/cpe/simulate")
async def cpe_simulate(req: SimulateRequest):
    """Simulate a single event and return scored outcome."""
    cpe = get_causal_policy_engine()
    return cpe.simulate_event(
        event_type=req.event_type,
        goal_id=req.goal_id,
        success=req.success,
        label=req.label,
    )


@router.post("/cpe/evaluate")
async def cpe_evaluate(req: EvaluateRequest):
    """Evaluate candidate actions and select best."""
    cpe = get_causal_policy_engine()
    candidates = cpe.propose_actions(req.goal_id, req.options)
    decision = cpe.evaluate(candidates, threshold=req.threshold)
    return decision.to_dict()


@router.post("/cpe/decide")
async def cpe_decide(req: DecideRequest):
    """Full propose + evaluate cycle."""
    cpe = get_causal_policy_engine()
    decision = cpe.decide(req.goal_id, req.options, threshold=req.threshold)
    return decision.to_dict()


@router.get("/cpe/stats")
async def cpe_stats():
    """Get CPE diagnostics."""
    return get_causal_policy_engine().get_stats()


# ════════════════════════════════════════════════════════════════
# PHE ENDPOINTS
# ════════════════════════════════════════════════════════════════

@router.post("/phe/plan")
async def phe_plan(req: PlanRequest):
    """Run strategic multi-step planning."""
    phe = get_policy_horizon_engine()
    return phe.plan(
        goal_id=req.goal_id,
        options=req.options,
        horizon=req.horizon,
        beam_width=req.beam_width,
    )


@router.post("/phe/simulate-depth")
async def phe_simulate_depth(req: SimulateDepthRequest):
    """Compare planning quality at different depths."""
    phe = get_policy_horizon_engine()
    return phe.simulate_depth(
        goal_id=req.goal_id,
        depths=req.depths,
        options=req.options,
        beam_width=req.beam_width,
    )


@router.get("/phe/stats")
async def phe_stats():
    """Get PHE diagnostics."""
    return get_policy_horizon_engine().get_stats()
