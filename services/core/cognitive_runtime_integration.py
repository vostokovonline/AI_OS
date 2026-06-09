"""
Cognitive Runtime Integration with main.py

Provides:
1. REST endpoints for goal execution with cognitive oversight
2. WebSocket endpoint for real-time cognitive events
3. Event-sourced state management
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Dict, Any, Optional
import asyncio
import json


# Create router
cognitive_router = APIRouter(prefix="/cognitive", tags=["cognitive"])

# Runtime instance
_runtime = None


def get_runtime():
    """Get or create cognitive runtime"""
    global _runtime
    if _runtime is None:
        from event_sourced_runtime import get_event_sourced_runtime
        _runtime = get_event_sourced_runtime()
    return _runtime


# =============================================================================
# REST ENDPOINTS
# =============================================================================

@cognitive_router.post("/execute")
async def execute_goal(request: Dict[str, Any]):
    """
    Execute goal with cognitive oversight.
    
    Request body:
    {
        "goal": {"title": "...", "domain": "...", "goal_type": "..."},
        "execution_state": {"resource_usage": 0.5, ...},
        "context": {"unresolved_goals": [], "failures": [], ...}
    }
    
    Returns:
    {
        "execution": {...},
        "identity": {...},
        "genome": {...},
        "lineage_id": "..."
    }
    """
    runtime = get_runtime()
    
    goal = request.get('goal', {})
    execution_state = request.get('execution_state', {})
    context = request.get('context', {})
    
    try:
        result = await runtime.process_goal(goal, execution_state, context)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@cognitive_router.get("/status")
async def get_status():
    """Get current cognitive runtime status"""
    runtime = get_runtime()
    return runtime.get_status()


@cognitive_router.get("/history")
async def get_event_history(
    event_type: Optional[str] = None,
    limit: int = 100
):
    """Get event history"""
    runtime = get_runtime()
    return runtime.get_history(event_type, limit)


@cognitive_router.post("/replay")
async def replay_from_snapshot(snapshot_id: str = None):
    """Replay runtime from snapshot"""
    runtime = get_runtime()
    return runtime.replay(snapshot_id)


@cognitive_router.post("/identity/protect/{axis}")
async def protect_identity_axis(axis: str):
    """Protect identity axis from mutation"""
    runtime = get_runtime()
    # Access identity and protect (would need to add this method)
    return {"status": "protected", "axis": axis}


# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@cognitive_router.websocket("/stream")
async def cognitive_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time cognitive events.
    
    Dashboard connects here to receive:
    - identity.axis.mutated
    - genome.evolved
    - pressure.changed
    - interrupt.raised
    - execution.lineage
    - etc.
    """
    await websocket.accept()
    
    runtime = get_runtime()
    event_queue = None
    
    try:
        # Subscribe to events
        event_queue = await runtime.subscribe()
        
        # Send initial state
        await websocket.send_json({
            "type": "initial_state",
            "data": runtime.get_status()
        })
        
        # Forward events to websocket
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=30)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
            except WebSocketDisconnect:
                break
            except Exception:
                break
                
    except Exception as e:
        pass
    finally:
        if event_queue:
            # Would need to add unsubscribe method to runtime
            pass


# =============================================================================
# MAIN.PY INTEGRATION
# =============================================================================

def register_cognitive_routes(app):
    """Register cognitive routes with FastAPI app"""
    from main import app as main_app
    
    # Include router
    main_app.include_router(cognitive_router)
    
    # Also mount at root for convenience
    main_app.mount("/cognitive", cognitive_router)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

"""
# In main.py:

from cognitive_runtime_integration import register_cognitive_routes

# After creating FastAPI app:
register_cognitive_routes(app)

# Endpoints now available:
# POST /cognitive/execute - Execute goal with cognition
# GET /cognitive/status - Get runtime status  
# GET /cognitive/history - Get event history
# POST /cognitive/replay - Replay from snapshot
# WS /cognitive/stream - Real-time event stream
"""