"""
Epistemic Kernel API — query and interact with the epistemic state.

Endpoints:
  GET  /epistemic/state               — full epistemic state
  GET  /epistemic/beliefs             — all beliefs
  GET  /epistemic/beliefs/{name}      — single belief
  GET  /epistemic/motifs              — all motifs
  GET  /epistemic/motifs/{name}       — single motif
  GET  /epistemic/attractors          — all attractors
  GET  /epistemic/attractors/{id}     — single attractor
  POST /epistemic/observe             — record an observation
  POST /epistemic/beliefs             — update a belief
  POST /epistemic/motifs              — update a motif
  GET  /epistemic/drift               — drift check
  POST /epistemic/attenuate           — apply drift attenuation
  GET  /epistemic/provenance/{name}   — belief provenance chain
  GET  /epistemic/journal             — journal stats
  POST /epistemic/grounding           — create grounding checkpoint
  GET  /epistemic/grounding           — list grounding checkpoints
  GET  /epistemic/invariants          — run semantic invariant check
  GET  /epistemic/stats               — kernel diagnostics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from epistemic_factory import get_epistemic_kernel

router = APIRouter(prefix="/epistemic", tags=["epistemic-kernel"])


# ── Request models ──

class ObservationRequest(BaseModel):
    signal: str
    value: float = 1.0
    source: str = ""
    context: dict = {}

class BeliefUpdateRequest(BaseModel):
    name: str
    confidence: float
    provenance: str = "api"

class MotifUpdateRequest(BaseModel):
    name: str
    strength: float
    recurrence: int = 1
    provenance: str = "api"


# ── Endpoints ──

@router.get("/state")
async def get_state():
    """Full epistemic state snapshot."""
    ek = get_epistemic_kernel()
    return {
        "beliefs": ek.get_all_beliefs(),
        "motifs": ek.get_all_motifs(),
        "attractors": ek.get_all_attractors(),
        "epoch": ek.epoch.current,
    }


@router.get("/beliefs")
async def list_beliefs():
    """Get all beliefs."""
    return get_epistemic_kernel().get_all_beliefs()


@router.get("/beliefs/{name}")
async def get_belief(name: str):
    """Get a single belief by name."""
    belief = get_epistemic_kernel().get_belief(name)
    if not belief.get('confidence'):
        raise HTTPException(status_code=404, detail=f"Belief '{name}' not found")
    return belief


@router.get("/motifs")
async def list_motifs():
    """Get all motifs."""
    return get_epistemic_kernel().get_all_motifs()


@router.get("/motifs/{name}")
async def get_motif(name: str):
    """Get a single motif by name."""
    motif = get_epistemic_kernel().get_motif(name)
    if not motif.get('strength'):
        raise HTTPException(status_code=404, detail=f"Motif '{name}' not found")
    return motif


@router.get("/attractors")
async def list_attractors():
    """Get all attractors."""
    return get_epistemic_kernel().get_all_attractors()


@router.get("/attractors/{attractor_id}")
async def get_attractor(attractor_id: str):
    """Get a single attractor by ID."""
    att = get_epistemic_kernel().get_attractor(attractor_id)
    if not att.get('weight'):
        raise HTTPException(status_code=404, detail=f"Attractor '{attractor_id}' not found")
    return att


@router.post("/observe")
async def record_observation(req: ObservationRequest):
    """Record a raw observation."""
    ek = get_epistemic_kernel()
    event_id = ek.record_observation(
        signal=req.signal,
        value=req.value,
        source=req.source,
        context=req.context,
    )
    return {"event_id": event_id, "signal": req.signal, "value": req.value}


@router.post("/beliefs")
async def update_belief(req: BeliefUpdateRequest):
    """Update a belief with provenance tracking."""
    ek = get_epistemic_kernel()
    result = ek.update_belief(
        name=req.name,
        confidence=req.confidence,
        provenance=req.provenance,
    )
    return result


@router.post("/motifs")
async def update_motif(req: MotifUpdateRequest):
    """Update a motif."""
    ek = get_epistemic_kernel()
    result = ek.update_motif(
        name=req.name,
        strength=req.strength,
        recurrence=req.recurrence,
        provenance=req.provenance,
    )
    return result


@router.get("/drift")
async def check_drift():
    """Run drift detection."""
    ek = get_epistemic_kernel()
    report = ek.check_drift()
    return report.to_dict()


@router.post("/attenuate")
async def attenuate_drift():
    """Apply drift attenuation."""
    ek = get_epistemic_kernel()
    report = ek.check_drift()
    count = ek.attenuate_drift(report)
    return {"attenuated": count, "drift_score": report.overall_drift_score}


@router.get("/provenance/{name}")
async def get_provenance(name: str):
    """Get provenance chain for a belief."""
    return get_epistemic_kernel().get_belief_provenance(name)


@router.get("/journal")
async def journal_stats():
    """Get semantic journal statistics."""
    return get_epistemic_kernel().journal.get_stats()


@router.post("/grounding")
async def create_grounding():
    """Create a grounding checkpoint."""
    ek = get_epistemic_kernel()
    checkpoint_id = ek.create_grounding_checkpoint()
    return {"checkpoint_id": checkpoint_id}


@router.get("/grounding")
async def list_grounding():
    """List grounding checkpoints."""
    return get_epistemic_kernel().list_grounding_checkpoints()


@router.get("/invariants")
async def check_invariants():
    """Run semantic invariant checks."""
    report = get_epistemic_kernel().verify()
    return report.to_dict()


@router.get("/stats")
async def stats():
    """Full epistemic kernel diagnostics."""
    return get_epistemic_kernel().get_stats()
