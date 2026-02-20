"""
AUTONOMY API ENDPOINTS

API for Phase 1: Controlled MVP autonomous decision-making.

Endpoints:
- GET  /autonomy/state - Get all system state
- GET  /autonomy/state/{entity_name} - Get specific entity
- POST /autonomy/state - Update entity manually
- GET  /autonomy/policies - List all policies
- POST /autonomy/policies - Add new policy
- POST /autonomy/process - Process artifact mutations
- GET  /autonomy/actions - Get pending actions
- POST /autonomy/actions/{action_id}/approve - Approve action
- POST /autonomy/actions/{action_id}/execute - Execute action
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from uuid import UUID
from pydantic import BaseModel

router = APIRouter(prefix="/autonomy", tags=["autonomy"])


# Request/Response models
class StateUpdateRequest(BaseModel):
    entity_name: str
    entity_type: str  # metric, strategy, resource, risk, hypothesis, constraint
    new_value: Dict[str, Any]
    confidence: float = 1.0
    source_artifact_id: Optional[str] = None


class PolicyCreateRequest(BaseModel):
    name: str
    entity_name: str
    entity_type: str
    condition_expression: str
    action_type: str  # create_goal, alert, deactivate_strategy, etc.
    action_payload: Dict[str, Any]
    priority: int = 1
    cooldown_minutes: int = 60


class ProcessMutationsRequest(BaseModel):
    artifact_id: str
    mutations: List[Dict[str, Any]]
    auto_approve: bool = False


@router.get("/state")
async def get_all_state():
    """Get all system state entities"""
    from autonomy.system_state import SystemStateManager
    
    manager = SystemStateManager()
    metrics = await manager.get_all_metrics()
    
    return {
        "status": "ok",
        "entities": [
            {
                "entity_name": e.entity_name,
                "entity_type": e.entity_type.value,
                "current_value": e.current_value,
                "previous_value": e.previous_value,
                "delta": e.get_delta(),
                "trend": e.get_trend(),
                "confidence": e.confidence,
                "last_updated": e.last_updated.isoformat()
            }
            for e in metrics
        ]
    }


@router.get("/state/{entity_name}")
async def get_entity_state(entity_name: str):
    """Get specific entity state"""
    from autonomy.system_state import SystemStateManager
    
    manager = SystemStateManager()
    entity = await manager.get_entity(entity_name)
    
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity not found: {entity_name}")
    
    return {
        "status": "ok",
        "entity": {
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type.value,
            "current_value": entity.current_value,
            "previous_value": entity.previous_value,
            "delta": entity.get_delta(),
            "trend": entity.get_trend(),
            "confidence": entity.confidence,
            "last_updated": entity.last_updated.isoformat()
        }
    }


@router.post("/state")
async def update_state(req: StateUpdateRequest):
    """Manually update a state entity"""
    from autonomy.system_state import SystemStateManager, EntityType
    
    manager = SystemStateManager()
    
    try:
        entity_type = EntityType(req.entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid entity_type: {req.entity_type}")
    
    source_id = UUID(req.source_artifact_id) if req.source_artifact_id else None
    
    entity = await manager.update_entity(
        entity_name=req.entity_name,
        entity_type=entity_type,
        new_value=req.new_value,
        source_artifact_id=source_id,
        confidence=req.confidence
    )
    
    return {
        "status": "ok",
        "entity": {
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type.value,
            "current_value": entity.current_value,
            "delta": entity.get_delta(),
            "trend": entity.get_trend()
        }
    }


@router.get("/policies")
async def list_policies():
    """List all policy rules"""
    from autonomy.policy_engine import PolicyEngine
    
    engine = PolicyEngine()
    rules = await engine.load_rules()
    
    return {
        "status": "ok",
        "policies": [r.to_dict() for r in rules]
    }


@router.post("/policies")
async def create_policy(req: PolicyCreateRequest):
    """Create a new policy rule"""
    from autonomy.policy_engine import PolicyEngine, EntityType, ActionType
    
    engine = PolicyEngine()
    
    try:
        entity_type = EntityType(req.entity_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid entity_type: {req.entity_type}")
    
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action_type: {req.action_type}")
    
    rule = await engine.add_rule(
        name=req.name,
        entity_name=req.entity_name,
        entity_type=entity_type,
        condition_expression=req.condition_expression,
        action_type=action_type,
        action_payload=req.action_payload,
        priority=req.priority,
        cooldown_minutes=req.cooldown_minutes
    )
    
    return {
        "status": "ok",
        "policy": rule.to_dict()
    }


@router.post("/process")
async def process_mutations(req: ProcessMutationsRequest):
    """Process state mutations from an artifact"""
    from autonomy.decision_engine import decision_engine
    from autonomy.state_mutation import StateMutation, MutationType
    
    # Convert mutations
    mutations = []
    for m in req.mutations:
        mutation = StateMutation(
            entity_name=m["entity_name"],
            mutation_type=MutationType(m.get("mutation_type", "update")),
            new_value=m["new_value"],
            confidence=m.get("confidence", 1.0),
            reason=m.get("reason")
        )
        mutations.append(mutation)
    
    # Run decision loop
    results = await decision_engine.run_decision_loop(
        mutations=mutations,
        artifact_id=UUID(req.artifact_id),
        auto_approve=req.auto_approve
    )
    
    return {
        "status": "ok",
        "mutations_processed": len(mutations),
        "actions_generated": len(results),
        "results": results
    }


@router.post("/test-autonomous-goal")
async def test_autonomous_goal():
    """Test autonomous goal creation by simulating a lead decline"""
    from autonomy.decision_engine import decision_engine
    from autonomy.state_mutation import StateMutation, MutationType
    
    # Simulate lead decline: was 145, now 120
    mutation = StateMutation(
        entity_name="monthly_leads",
        mutation_type=MutationType.UPDATE,
        new_value={"value": 120},
        previous_value={"value": 145},
        confidence=0.95,
        reason="Test: simulating lead decline for autonomous goal creation test"
    )
    
    # First, set up initial state
    from autonomy.system_state import SystemStateManager, EntityType
    manager = SystemStateManager()
    
    await manager.update_entity(
        entity_name="monthly_leads",
        entity_type=EntityType.METRIC,
        new_value={"value": 145}
    )
    
    # Add default policy if not exists
    from autonomy.policy_engine import PolicyEngine, ActionType
    engine = PolicyEngine()
    
    try:
        await engine.add_rule(
            name="lead_decline_investigate",
            entity_name="monthly_leads",
            entity_type=EntityType.METRIC,
            condition_expression="delta < 0",
            action_type=ActionType.CREATE_GOAL,
            action_payload={
                "title": "Investigate lead decline",
                "description": "Leads decreased from {previous} to {current}. Investigate root cause.",
                "goal_type": "exploratory"
            },
            priority=2,
            cooldown_minutes=60
        )
    except Exception:
        pass  # Rule already exists
    
    # Now process the decline
    results = await decision_engine.run_decision_loop(
        mutations=[mutation],
        artifact_id=UUID("00000000-0000-0000-0000-000000000001"),
        auto_approve=True
    )
    
    return {
        "status": "ok",
        "test": "lead_decline_simulation",
        "results": results
    }
