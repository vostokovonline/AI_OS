"""
Cognitive Runtime WITH Event Bus - Complete Integration

This combines kernel_runtime + cognitive_event_bus into single system.
All cognitive mutations emit events for observability.
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import asyncio


# =============================================================================
# COGNITIVE EVENT BUS (Inline for simplicity)
# =============================================================================

class CognitiveEventBus:
    """Unified event bus for runtime observability"""
    
    def __init__(self, max_history: int = 2000):
        self._history = []
        self._max_history = max_history
        self._subscribers = {}
    
    def emit(self, event_type: str, payload: Dict, cause_id: str = None):
        event = {
            'id': str(uuid4()),
            'type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'payload': payload,
            'cause_id': cause_id
        }
        
        self._history.append(event)
        
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        # Notify subscribers
        if event_type in self._subscribers:
            for cb in self._subscribers[event_type]:
                try:
                    cb(event)
                except:
                    pass
    
    def subscribe(self, event_type: str, callback):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def get_history(self, event_type=None, limit=100):
        events = self._history
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        return events[-limit:]


# =============================================================================
# IDENTITY TOPOLOGY WITH EVENT EMISSION
# =============================================================================

class IdentityTopology:
    """Identity as multi-dimensional vector with event emission"""
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._bus = event_bus
        
        self.axes = {
            'exploration': 0.5,
            'stability': 0.5,
            'autonomy': 0.5,
            'precision': 0.5,
            'aggression': 0.5,
            'reflection': 0.5,
            'persistence': 0.5,
        }
        
        self._protected = []
        self._history = []
        self.last_update = datetime.utcnow().isoformat()
        self.mutation_count = 0
    
    def mutate(self, axis: str, delta: float):
        if axis in self._protected:
            return
        
        if axis not in self.axes:
            self.axes[axis] = 0.5
        
        old_value = self.axes[axis]
        self.axes[axis] = max(0.1, min(0.9, self.axes[axis] + delta))
        
        # Emit event
        self._bus.emit('identity_axis_mutated', {
            'axis': axis,
            'old': round(old_value, 3),
            'new': round(self.axes[axis], 3),
            'delta': round(delta, 3),
            'mutation_count': self.mutation_count
        })
        
        self.last_update = datetime.utcnow().isoformat()
        self.mutation_count += 1
    
    def apply_execution_result(self, outcome: str, domain: str, approach: str, metrics: Dict):
        axis_deltas = {}
        
        if outcome == 'success':
            axis_deltas = {
                'autonomy': 0.02,
                'stability': 0.01,
                'aggression': 0.02 if metrics.get('leverage_created', 0) > 0.3 else 0,
                'persistence': 0.02 if metrics.get('leverage_created', 0) > 0.3 else 0,
                'exploration': 0.03 if metrics.get('exploration') else 0,
            }
        elif outcome == 'failure':
            axis_deltas = {
                'autonomy': -0.03,
                'stability': -0.02,
                'aggression': -0.04,
                'precision': 0.02,
            }
        
        for axis, delta in axis_deltas.items():
            self.mutate(axis, delta)
    
    def protect_axis(self, axis: str):
        if axis not in self._protected:
            self._protected.append(axis)
            self._bus.emit('identity_axis_protected', {'axis': axis})
    
    def get_vector(self):
        return dict(self.axes)


# =============================================================================
# EXECUTION GENOME WITH EVENTS
# =============================================================================

@dataclass
class ExecutionGenome:
    risk_tolerance: float = 0.5
    decomposition_depth: int = 3
    retry_behavior: str = "adaptive"
    exploration_bias: float = 0.3
    validation_frequency: int = 2
    generation: int = 0


class GenomeEvolver:
    def __init__(self, genome: ExecutionGenome, event_bus: CognitiveEventBus):
        self._genome = genome
        self._bus = event_bus
    
    def evolve(self, result: Dict) -> ExecutionGenome:
        old_gen = self._genome.generation
        
        new_genome = ExecutionGenome(
            generation=old_gen + 1,
            risk_tolerance=self._genome.risk_tolerance,
            decomposition_depth=self._genome.decomposition_depth,
            retry_behavior=self._genome.retry_behavior,
            exploration_bias=self._genome.exploration_bias,
            validation_frequency=self._genome.validation_frequency,
        )
        
        if result.get('outcome') == 'success':
            new_genome.exploration_bias = min(0.8, new_genome.exploration_bias + 0.05)
            new_genome.risk_tolerance = min(0.8, new_genome.risk_tolerance + 0.02)
        elif result.get('outcome') == 'failure':
            new_genome.risk_tolerance = max(0.2, new_genome.risk_tolerance - 0.1)
            new_genome.validation_frequency = min(5, new_genome.validation_frequency + 1)
        
        # Emit event
        self._bus.emit('genome_evolved', {
            'old_generation': old_gen,
            'new_generation': new_genome.generation,
            'changes': {
                'risk_tolerance': new_genome.risk_tolerance - self._genome.risk_tolerance,
                'exploration_bias': new_genome.exploration_bias - self._genome.exploration_bias
            }
        })
        
        return new_genome


# =============================================================================
# PRESSURE PHYSICS WITH EVENTS
# =============================================================================

class PressurePhysics:
    """Automatic pressure with event emission"""
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._bus = event_bus
        self._pressures = {}
    
    def compute_pressures(self, execution_state, unresolved_goals, failures, knowledge_gaps):
        pressures = {}
        
        # Auto-generate pressure from state
        if len(unresolved_goals) > 5:
            pressures['contradiction'] = min(1.0, len(unresolved_goals) / 20)
        
        if failures:
            pressures['failure_recovery'] = min(1.0, len(failures) / 5)
        
        if execution_state.get('resource_usage', 0.5) > 0.8:
            pressures['resource'] = execution_state['resource_usage']
        
        # Compare with old and emit changes
        for source, value in pressures.items():
            old = self._pressures.get(source, 0)
            if abs(value - old) > 0.1:
                self._bus.emit('pressure_changed', {
                    'source': source,
                    'old': round(old, 3),
                    'new': round(value, 3),
                    'delta': round(value - old, 3)
                })
            
            self._pressures[source] = value
        
        return pressures
    
    def get_total(self):
        return sum(self._pressures.values())


# =============================================================================
# INTRUSIVE MONITOR WITH EVENTS  
# =============================================================================

@dataclass
class CognitiveInterrupt:
    type: str
    severity: float
    action: str
    reason: str


class IntrusiveMonitor:
    """Monitor that emits events on interrupts"""
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._bus = event_bus
        self._active = False
        self._interrupts = []
    
    def start(self, execution_id: str):
        self._active = True
        self._bus.emit('execution_started', {'execution_id': execution_id})
    
    def check_interrupt(self, state: Dict) -> Optional[CognitiveInterrupt]:
        if not self._active:
            return None
        
        interrupt = None
        
        if state.get('progress_delta', 1.0) < 0.01:
            interrupt = CognitiveInterrupt('stall', 0.8, 'change_strategy', 'No progress')
        elif state.get('error_count', 0) > 3:
            interrupt = CognitiveInterrupt('error_escalation', 0.9, 'abort', 'Too many errors')
        
        if interrupt:
            self._bus.emit('interrupt_raised', {
                'type': interrupt.type,
                'severity': interrupt.severity,
                'action': interrupt.action,
                'reason': interrupt.reason
            })
            self._interrupts.append(interrupt)
        
        return interrupt
    
    def stop(self):
        self._active = False


# =============================================================================
# REAL EXECUTION BINDING (Placeholder for real integration)
# =============================================================================

class RealExecutionBinding:
    """
    Placeholder for real AI-OS execution integration.
    This is where you connect to goal_decomposer, skill_registry, agent_graph.
    """
    
    def __init__(self, runtime):
        self._runtime = runtime
    
    async def execute(self, goal: Dict, context: Dict) -> Dict:
        """
        REAL execution would call:
        - goal_decomposer.decompose(goal)
        - skill_registry.select(plan)
        - agent_graph.execute(skills)
        - artifact_verifier.verify(results)
        """
        
        # Placeholder - returns mock but documents integration points
        return {
            'outcome': 'success',  # Would be real result
            'leverage_created': 0.3,
            'execution_id': str(uuid4()),
            'note': 'REAL_EXECUTION_BINDING - integrate with goal_executor.py'
        }


# =============================================================================
# MAIN COGNITIVE RUNTIME WITH FULL OBSERVABILITY
# =============================================================================

class CognitiveRuntime:
    """
    THE ONLY cognitive runtime with full observability.
    All mutations emit events - no hidden state.
    """
    
    def __init__(self):
        # Create event bus first
        self._bus = CognitiveEventBus()
        
        # All components get event bus
        self._identity = IdentityTopology(self._bus)
        self._genome = ExecutionGenome()
        self._genome_evolver = GenomeEvolver(self._genome, self._bus)
        self._pressure = PressurePhysics(self._bus)
        self._monitor = IntrusiveMonitor(self._bus)
        
        # Execution binding
        self._execution = RealExecutionBinding(self)
        
        # Stats
        self._executions = 0
        self._successes = 0
    
    async def process_goal(self, goal: Dict, execution_state: Dict, context: Dict = None):
        """Main entry - all mutations emit events"""
        
        context = context or {}
        self._executions += 1
        
        # 1. Compute pressures (emits events)
        pressures = self._pressure.compute_pressures(
            execution_state,
            context.get('unresolved_goals', []),
            context.get('failures', []),
            context.get('knowledge_gaps', [])
        )
        
        # 2. Start monitoring (emits event)
        self._monitor.start(str(uuid4()))
        
        # 3. Select strategy
        strategy = self._select_strategy(goal, context)
        self._bus.emit('strategy_selected', {
            'goal': goal.get('title'),
            'strategy': strategy,
            'genome_generation': self._genome.generation
        })
        
        # 4. Execute (would be real)
        result = await self._execution.execute(goal, context)
        
        # 5. Check interrupt
        interrupt = self._monitor.check_interrupt(result)
        if interrupt:
            self._bus.emit('interrupt_handled', {
                'type': interrupt.type,
                'action': interrupt.action
            })
        
        # 6. Update identity (emits events)
        self._identity.apply_execution_result(
            result.get('outcome', 'unknown'),
            goal.get('domain', 'general'),
            strategy,
            result
        )
        
        # 7. Evolve genome (emits event)
        self._genome = self._genome_evolver.evolve(result)
        
        # 8. Stop monitoring
        self._monitor.stop()
        
        if result.get('outcome') == 'success':
            self._successes += 1
        
        # 9. Emit lineage event
        self._bus.emit('execution_lineage', {
            'goal': goal.get('title'),
            'strategy': strategy,
            'outcome': result.get('outcome'),
            'interrupt': interrupt is not None,
            'genome_gen': self._genome.generation
        })
        
        return {
            'execution': result,
            'identity': self._identity.get_vector(),
            'genome': {
                'generation': self._genome.generation,
                'risk_tolerance': self._genome.risk_tolerance
            },
            'pressures': pressures,
            'personality': self._get_personality()
        }
    
    def _select_strategy(self, goal: Dict, context: Dict) -> str:
        if self._genome.risk_tolerance > 0.7:
            return 'aggressive_pursue'
        elif self._genome.risk_tolerance > 0.4:
            return 'balanced_execute'
        return 'cautious_explore'
    
    def _get_personality(self) -> str:
        traits = []
        if self._identity.axes['exploration'] > 0.7:
            traits.append("explorer")
        if self._identity.axes['aggression'] > 0.7:
            traits.append("risk-taker")
        if self._identity.axes['precision'] > 0.7:
            traits.append("precision-focused")
        return ", ".join(traits) if traits else "balanced"
    
    def get_status(self):
        return {
            'identity': self._identity.get_vector(),
            'genome': {
                'generation': self._genome.generation,
                'risk_tolerance': self._genome.risk_tolerance,
                'exploration_bias': self._genome.exploration_bias
            },
            'pressure': self._pressure._pressures,
            'stats': {
                'total': self._executions,
                'successes': self._successes,
                'rate': self._successes / max(self._executions, 1)
            },
            'interrupts': len(self._monitor._interrupts)
        }
    
    def get_event_history(self, event_type=None, limit=50):
        return self._bus.get_history(event_type, limit)
    
    def subscribe_to_events(self, event_type: str, callback):
        self._bus.subscribe(event_type, callback)


# Global
_runtime = None

def get_cognitive_runtime() -> CognitiveRuntime:
    global _runtime
    if _runtime is None:
        _runtime = CognitiveRuntime()
    return _runtime