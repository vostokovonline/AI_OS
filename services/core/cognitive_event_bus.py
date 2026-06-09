"""
Cognitive Event Bus - Unified Event System for Runtime Observability

ARCHITECTURAL INVARIANT:
- Runtime is source of truth
- All mutations emit events
- Dashboard subscribes only (no split-brain)
- No mutation without visualization

Event types:
- identity_axis_mutated
- genome_evolved  
- pressure_changed
- interrupt_raised
- strategy_changed
- execution_lineage_step
- cognitive_drift_detected
"""
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import json
import asyncio
from collections import deque


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

@dataclass
class CognitiveEvent:
    """Base cognitive event"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    type: str = ""
    source: str = "cognitive_runtime"
    
    # Payload must be JSON-serializable
    payload: Dict = field(default_factory=dict)
    
    # Causal chain
    cause_event_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp,
            'type': self.type,
            'source': self.source,
            'payload': self.payload,
            'cause_event_id': self.cause_event_id
        }


# Event Types
class EventTypes:
    # Identity events
    IDENTITY_AXIS_MUTATED = "identity.axis.mutated"
    IDENTITY_PROTECTED = "identity.axis.protected"
    IDENTITY_DRIFT_DETECTED = "identity.drift.detected"
    
    # Genome events
    GENOME_EVOLVED = "genome.evolved"
    GENOME_MUTATION = "genome.mutation"
    
    # Pressure events
    PRESSURE_CHANGED = "pressure.changed"
    PRESSURE_THRESHOLD_CROSSED = "pressure.threshold.crossed"
    PRESSURE_SOURCE_ACTIVATED = "pressure.source.activated"
    
    # Execution events
    STRATEGY_SELECTED = "strategy.selected"
    STRATEGY_CHANGED = "strategy.changed"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    
    # Interrupt events
    INTERRUPT_RAISED = "interrupt.raised"
    INTERRUPT_HANDLED = "interrupt.handled"
    INTERRUPT_ESCALATED = "interrupt.escalated"
    
    # Lineage events
    LINEAGE_STEP = "lineage.step"
    LINEAGE_BRANCHED = "lineage.branched"


# =============================================================================
# EVENT BUS
# =============================================================================

class CognitiveEventBus:
    """
    Unified event bus - single source of truth for all runtime events.
    All other components subscribe to this, not each other.
    """
    
    def __init__(self, max_history: int = 1000):
        self._max_history = max_history
        self._history: deque = deque(maxlen=max_history)
        
        # Subscribers: event_type -> [callbacks]
        self._subscribers: Dict[str, List[Callable]] = {}
        
        # Debug: all events logged
        self._debug_log: List[Dict] = []
    
    def emit(self, event: CognitiveEvent):
        """Emit event to bus"""
        
        # Store in history
        self._history.append(event)
        
        # Debug log
        self._debug_log.append(event.to_dict())
        
        # Notify subscribers
        if event.type in self._subscribers:
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    pass  # Don't let subscriber errors break runtime
        
        # Also notify wildcard subscribers
        if "*" in self._subscribers:
            for callback in self._subscribers["*"]:
                try:
                    callback(event)
                except Exception:
                    pass
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from event type"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass
    
    def get_history(
        self, 
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get event history"""
        events = list(self._history)
        
        if event_type:
            events = [e for e in events if e.type == event_type]
        
        return [e.to_dict() for e in events[-limit:]]
    
    def get_latest(self, event_type: str) -> Optional[Dict]:
        """Get latest event of type"""
        for event in reversed(list(self._history)):
            if event.type == event_type:
                return event.to_dict()
        return None
    
    def get_causal_chain(self, event_id: str) -> List[Dict]:
        """Get full causal chain for event"""
        chain = []
        
        # Find event
        event = None
        for e in self._history:
            if e.event_id == event_id:
                event = e
                break
        
        if not event:
            return []
        
        # Walk back through causes
        current = event
        while current:
            chain.append(current.to_dict())
            
            if current.cause_event_id:
                # Find cause
                for e in self._history:
                    if e.event_id == current.cause_event_id:
                        current = e
                        break
                else:
                    break
            else:
                break
        
        return list(reversed(chain))


# =============================================================================
# EVENT EMITTERS - Plug into Runtime
# =============================================================================

class EventEmitter:
    """Mixin class to add event emission to any component"""
    
    def __init__(self):
        self._event_bus: Optional[CognitiveEventBus] = None
    
    def set_event_bus(self, bus: CognitiveEventBus):
        self._event_bus = bus
    
    def emit_identity_mutation(
        self,
        axis: str,
        old_value: float,
        new_value: float,
        cause: str,
        cause_event_id: Optional[str] = None
    ):
        """Emit identity axis mutation event"""
        
        event = CognitiveEvent(
            type=EventTypes.IDENTITY_AXIS_MUTATED,
            payload={
                'axis': axis,
                'old_value': round(old_value, 3),
                'new_value': round(new_value, 3),
                'delta': round(new_value - old_value, 3),
                'cause': cause
            },
            cause_event_id=cause_event_id
        )
        
        if self._event_bus:
            self._event_bus.emit(event)
        
        return event.event_id
    
    def emit_genome_evolution(
        self,
        old_genome: Dict,
        new_genome: Dict,
        cause_event_id: Optional[str] = None
    ):
        """Emit genome evolution event"""
        
        event = CognitiveEvent(
            type=EventTypes.GENOME_EVOLVED,
            payload={
                'old_generation': old_genome.get('generation', 0),
                'new_generation': new_genome.get('generation', 0),
                'changes': {
                    k: {'old': old_genome.get(k), 'new': new_genome.get(k)}
                    for k in old_genome
                    if old_genome.get(k) != new_genome.get(k)
                }
            },
            cause_event_id=cause_event_id
        )
        
        if self._event_bus:
            self._event_bus.emit(event)
        
        return event.event_id
    
    def emit_pressure_change(
        self,
        source: str,
        old_pressure: float,
        new_pressure: float,
        cause_event_id: Optional[str] = None
    ):
        """Emit pressure change event"""
        
        event = CognitiveEvent(
            type=EventTypes.PRESSURE_CHANGED,
            payload={
                'source': source,
                'old_pressure': round(old_pressure, 3),
                'new_pressure': round(new_pressure, 3),
                'delta': round(new_pressure - old_pressure, 3)
            },
            cause_event_id=cause_event_id
        )
        
        if self._event_bus:
            self._event_bus.emit(event)
        
        return event.event_id
    
    def emit_interrupt(
        self,
        interrupt_type: str,
        severity: float,
        action: str,
        reason: str,
        cause_event_id: Optional[str] = None
    ):
        """Emit interrupt event"""
        
        event = CognitiveEvent(
            type=EventTypes.INTERRUPT_RAISED,
            payload={
                'interrupt_type': interrupt_type,
                'severity': severity,
                'action': action,
                'reason': reason
            },
            cause_event_id=cause_event_id
        )
        
        if self._event_bus:
            self._event_bus.emit(event)
        
        return event.event_id
    
    def emit_strategy_change(
        self,
        old_strategy: str,
        new_strategy: str,
        reason: str,
        cause_event_id: Optional[str] = None
    ):
        """Emit strategy change event"""
        
        event = CognitiveEvent(
            type=EventTypes.STRATEGY_CHANGED,
            payload={
                'old_strategy': old_strategy,
                'new_strategy': new_strategy,
                'reason': reason
            },
            cause_event_id=cause_event_id
        )
        
        if self._event_bus:
            self._event_bus.emit(event)
        
        return event.event_id
    
    def emit_execution_lineage(
        self,
        step_type: str,
        goal_id: str,
        details: Dict,
        cause_event_id: Optional[str] = None
    ):
        """Emit execution lineage step"""
        
        event = CognitiveEvent(
            type=EventTypes.LINEAGE_STEP,
            payload={
                'step_type': step_type,
                'goal_id': goal_id,
                'details': details
            },
            cause_event_id=cause_event_id
        )
        
        if self._event_bus:
            self._event_bus.emit(event)
        
        return event.event_id


# =============================================================================
# DASHBOARD INTEGRATION LAYER
# =============================================================================

class DashboardIntegration:
    """
    Thin layer connecting event bus to dashboard.
    This is the ONLY allowed communication from runtime to dashboard.
    """
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._event_bus = event_bus
        
        # Websocket connections (would be real in production)
        self._ws_connections: List[Any] = []
    
    def subscribe_dashboard(self, callback: Callable):
        """Dashboard subscribes to all events"""
        self._event_bus.subscribe("*", callback)
    
    def get_state_snapshot(self) -> Dict:
        """Get full state snapshot for dashboard initialization"""
        return {
            'events': {
                'identity_mutations': self._event_bus.get_history(
                    EventTypes.IDENTITY_AXIS_MUTATED, limit=50
                ),
                'genome_evolutions': self._event_bus.get_history(
                    EventTypes.GENOME_EVOLVED, limit=20
                ),
                'pressure_changes': self._event_bus.get_history(
                    EventTypes.PRESSURE_CHANGED, limit=50
                ),
                'interrupts': self._event_bus.get_history(
                    EventTypes.INTERRUPT_RAISED, limit=20
                ),
                'strategy_changes': self._event_bus.get_history(
                    EventTypes.STRATEGY_CHANGED, limit=30
                ),
                'execution_lineage': self._event_bus.get_history(
                    EventTypes.LINEAGE_STEP, limit=100
                )
            }
        }
    
    def get_live_metrics(self) -> Dict:
        """Get live metrics for dashboard"""
        return {
            'pressure_sources': {},
            'identity_axes': {},
            'interrupt_count': 0,
            'execution_count': 0
        }


# =============================================================================
# GLOBAL EVENT BUS
# =============================================================================

_event_bus: Optional[CognitiveEventBus] = None
_dashboard_integration: Optional[DashboardIntegration] = None


def get_event_bus() -> CognitiveEventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = CognitiveEventBus()
    return _event_bus


def get_dashboard_integration() -> DashboardIntegration:
    global _dashboard_integration
    if _dashboard_integration is None:
        _dashboard_integration = DashboardIntegration(get_event_bus())
    return _dashboard_integration


def create_event_emitter() -> EventEmitter:
    """Create event emitter connected to global bus"""
    emitter = EventEmitter()
    emitter.set_event_bus(get_event_bus())
    return emitter