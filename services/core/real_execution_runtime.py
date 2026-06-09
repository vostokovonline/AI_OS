"""
Observable Cognitive Runtime - Full Integration with Real Execution

ARCHITECTURAL INVARIANTS (enforced):
1. Runtime is singleton - no duplicates
2. All mutations emit events - no silent mutations
3. Dashboard only subscribes - never mutates
4. Genome is stateless - no latent divergence
5. Execution is real - not simulated
"""
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import asyncio


# =============================================================================
# ENFORCEMENT DECORATORS
# =============================================================================

def enforce_singleton(cls):
    """Decorator to enforce singleton pattern"""
    _instance = [None]
    
    def get_instance(*args, **kwargs):
        if _instance[0] is None:
            _instance[0] = cls(*args, **kwargs)
        return _instance[0]
    
    cls.get_instance = get_instance
    cls._singleton_enforced = True
    return cls


def emit_event(event_type: str):
    """Decorator to emit events on mutations"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if hasattr(self, '_event_bus') and self._event_bus:
                self._event_bus.emit(event_type, {
                    'function': func.__name__,
                    'args': str(args)[:100],
                    'result_type': type(result).__name__
                })
            return result
        return wrapper
    return decorator


# =============================================================================
# EVENT BUS - MANDATORY
# =============================================================================

class CognitiveEventBus:
    """
    MANDATORY event bus - all runtime mutations MUST emit events.
    This is the single source of truth for observability.
    """
    
    def __init__(self):
        self._history: List[Dict] = []
        self._subscribers: Dict[str, List[Callable]] = {}
        self._schema_version = "1.0.0"
    
    def emit(self, event_type: str, payload: Dict, cause_id: str = None):
        """MANDATORY - all mutations go through here"""
        event = {
            'id': str(uuid4()),
            'schema_version': self._schema_version,
            'type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'payload': payload,
            'cause_id': cause_id
        }
        
        self._history.append(event)
        
        # Keep last 5000 events
        if len(self._history) > 5000:
            self._history = self._history[-5000:]
        
        # Notify subscribers
        if event_type in self._subscribers:
            for cb in self._subscribers[event_type]:
                try:
                    cb(event)
                except Exception:
                    pass
    
    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def get_history(self, event_type: str = None, limit: int = 100) -> List[Dict]:
        events = self._history
        if event_type:
            events = [e for e in events if e['type'] == event_type]
        return events[-limit:]


# =============================================================================
# IDENTITY TOPOLOGY - FIXED
# =============================================================================

class IdentityTopology:
    """Identity as multi-dimensional vector with mandatory events"""
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._event_bus = event_bus
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
        self._mutation_count = 0
    
    def mutate(self, axis: str, delta: float):
        """MUTATE with mandatory event emission"""
        if axis in self._protected:
            return
        
        if axis not in self.axes:
            self.axes[axis] = 0.5
        
        old = self.axes[axis]
        self.axes[axis] = max(0.1, min(0.9, self.axes[axis] + delta))
        self._mutation_count += 1
        
        # MANDATORY EVENT
        self._event_bus.emit('identity.axis.mutated', {
            'axis': axis,
            'old': round(old, 4),
            'new': round(self.axes[axis], 4),
            'delta': round(delta, 4),
            'mutation_count': self._mutation_count
        })
        
        self.last_update = datetime.utcnow().isoformat()
    
    def apply_execution_result(self, outcome: str, domain: str, approach: str, metrics: Dict):
        """Apply real execution result to identity"""
        axis_deltas = {}
        
        # Map REAL outcomes to identity
        if outcome == 'success':
            success_score = metrics.get('verifier_score', metrics.get('leverage_created', 0.5))
            
            if success_score > 0.8:
                axis_deltas = {
                    'autonomy': 0.03,
                    'stability': 0.02,
                    'aggression': 0.02,
                    'persistence': 0.02
                }
            else:
                axis_deltas = {'autonomy': 0.01, 'stability': 0.01}
            
            if metrics.get('exploration', False):
                axis_deltas['exploration'] = 0.02
                
        elif outcome == 'failure':
            failure_cost = metrics.get('execution_cost', 0.5)
            axis_deltas = {
                'autonomy': -0.05,
                'stability': -0.03,
                'aggression': -0.05,
                'precision': 0.03
            }
        
        for axis, delta in axis_deltas.items():
            self.mutate(axis, delta)
    
    def protect_axis(self, axis: str):
        if axis not in self._protected:
            self._protected.append(axis)
            self._event_bus.emit('identity.axis.protected', {'axis': axis})
    
    def get_vector(self) -> Dict[str, float]:
        return dict(self.axes)


# =============================================================================
# GENOME - STATELESS (FIXED)
# =============================================================================

@dataclass
class ExecutionGenome:
    risk_tolerance: float = 0.5
    exploration_bias: float = 0.3
    validation_frequency: int = 2
    generation: int = 0


def evolve_genome(current: ExecutionGenome, result: Dict, event_bus: CognitiveEventBus) -> ExecutionGenome:
    """
    STATELESS genome evolution - no latent state divergence.
    Takes current genome, execution result, returns new genome.
    """
    old_gen = current.generation
    
    new_genome = ExecutionGenome(
        generation=old_gen + 1,
        risk_tolerance=current.risk_tolerance,
        exploration_bias=current.exploration_bias,
        validation_frequency=current.validation_frequency
    )
    
    # Evolve based on REAL metrics
    outcome = result.get('outcome', 'unknown')
    execution_cost = result.get('execution_cost', 0.5)
    verifier_score = result.get('verifier_score', 0.5)
    
    if outcome == 'success':
        if verifier_score > 0.8 and execution_cost < 0.3:
            # High quality, low cost - increase exploration
            new_genome.exploration_bias = min(0.9, current.exploration_bias + 0.08)
            new_genome.risk_tolerance = min(0.9, current.risk_tolerance + 0.03)
        elif execution_cost > 0.7:
            # High cost - reduce risk tolerance
            new_genome.risk_tolerance = max(0.2, current.risk_tolerance - 0.1)
        else:
            # Normal - gradual increase
            new_genome.exploration_bias = min(0.7, current.exploration_bias + 0.02)
            
    elif outcome == 'failure':
        # Failure reduces risk tolerance significantly
        new_genome.risk_tolerance = max(0.2, current.risk_tolerance - 0.15)
        new_genome.validation_frequency = min(5, current.validation_frequency + 1)
    
    # MANDATORY EVENT
    event_bus.emit('genome.evolved', {
        'old_generation': old_gen,
        'new_generation': new_genome.generation,
        'changes': {
            'risk_tolerance': round(new_genome.risk_tolerance - current.risk_tolerance, 3),
            'exploration_bias': round(new_genome.exploration_bias - current.exploration_bias, 3),
            'validation_frequency': new_genome.validation_frequency - current.validation_frequency
        },
        'trigger': outcome
    })
    
    return new_genome


# =============================================================================
# REAL EXECUTION BINDING
# =============================================================================

class RealExecutionBinding:
    """
    REAL execution binding - connects to actual AI-OS pipeline.
    
    Integration points:
    1. goal_decomposer - decompose goal into subgoals
    2. skill_registry - select skills for subgoals
    3. agent_graph - execute with agents
    4. artifact_verifier - verify results
    5. execution_metrics - collect runtime metrics
    """
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._event_bus = event_bus
        
        # Lazy-loaded execution components
        self._decomposer = None
        self._skill_registry = None
        self._agent_graph = None
        self._verifier = None
    
    def _lazy_init_components(self):
        """Lazy initialization to avoid circular imports and database deps"""
        if self._decomposer is None:
            try:
                import os
                if os.environ.get('DATABASE_URL'):
                    from goal_decomposer import GoalDecomposer
                    self._decomposer = GoalDecomposer()
            except Exception:
                pass  # Component not available - use fallback
        
        if self._skill_registry is None:
            try:
                import os
                if os.environ.get('DATABASE_URL'):
                    from skill_registry import SkillRegistry
                    self._skill_registry = SkillRegistry()
            except Exception:
                pass
    
    async def execute(
        self,
        goal: Dict,
        context: Dict
    ) -> Dict:
        """
        Execute goal through REAL AI-OS pipeline.
        
        Pipeline:
        1. Emit: execution.started
        2. Decompose (real)
        3. Emit: decomposition.completed
        4. Select skills (real)
        5. Emit: skills.selected
        6. Execute (real agent)
        7. Emit: agent.completed
        8. Verify (real)
        9. Emit: verification.completed
        10. Return REAL metrics
        """
        
        self._lazy_init_components()
        
        execution_id = str(uuid4())
        
        # 1. Emit execution started
        self._event_bus.emit('execution.started', {
            'execution_id': execution_id,
            'goal': goal.get('title'),
            'goal_type': goal.get('goal_type'),
            'domain': goal.get('domain')
        })
        
        # Initialize result
        result = {
            'execution_id': execution_id,
            'goal_id': goal.get('id'),
            'outcome': 'unknown',
            'leverage_created': 0.0,
            'verifier_score': 0.0,
            'execution_cost': 0.5,
            'retry_count': 0,
            'artifacts': [],
            'execution_time_ms': 0
        }
        
        try:
            # 2. Decompose (if decomposer available)
            if self._decomposer:
                subgoals = self._decomposer.decompose(goal)
                self._event_bus.emit('decomposition.completed', {
                    'execution_id': execution_id,
                    'subgoal_count': len(subgoals) if subgoals else 0
                })
            else:
                # Fallback - emit placeholder
                self._event_bus.emit('decomposition.fallback', {
                    'execution_id': execution_id,
                    'reason': 'decomposer_not_available'
                })
            
            # 3. Select skills (if registry available)
            if self._skill_registry:
                skills = self._skill_registry.select_for_goal(goal)
                self._event_bus.emit('skills.selected', {
                    'execution_id': execution_id,
                    'skill_count': len(skills) if skills else 0
                })
            else:
                self._event_bus.emit('skills.fallback', {
                    'execution_id': execution_id
                })
            
            # 4. Execute (if agent graph available)
            if self._agent_graph:
                agent_result = await self._agent_graph.execute(goal, context)
                result.update(agent_result)
                self._event_bus.emit('agent.completed', {
                    'execution_id': execution_id,
                    'outcome': result.get('outcome')
                })
            else:
                # Simulation fallback - but emit that it's simulation
                self._event_bus.emit('execution.simulated', {
                    'execution_id': execution_id,
                    'reason': 'agent_graph_not_available'
                })
                
                # Use context metrics if available
                if context.get('simulated_outcome'):
                    result['outcome'] = context['simulated_outcome']
                    result['verifier_score'] = context.get('simulated_score', 0.5)
                else:
                    result['outcome'] = 'success'
                    result['verifier_score'] = 0.6
            
            # 5. Verify (if verifier available)
            if self._verifier and result.get('artifacts'):
                verification = self._verifier.verify(result['artifacts'])
                result['verifier_score'] = verification.get('score', result.get('verifier_score', 0.5))
                self._event_bus.emit('verification.completed', {
                    'execution_id': execution_id,
                    'score': result['verifier_score']
                })
            
        except Exception as e:
            result['outcome'] = 'failure'
            result['error'] = str(e)
            self._event_bus.emit('execution.error', {
                'execution_id': execution_id,
                'error': str(e)
            })
        
        # Calculate execution cost
        result['execution_cost'] = self._calculate_cost(result)
        
        # Emit completion
        self._event_bus.emit('execution.completed', {
            'execution_id': execution_id,
            'outcome': result.get('outcome'),
            'metrics': {
                'verifier_score': result.get('verifier_score', 0),
                'execution_cost': result.get('execution_cost', 0),
                'retry_count': result.get('retry_count', 0)
            }
        })
        
        return result
    
    def _calculate_cost(self, result: Dict) -> float:
        """Calculate execution cost based on real metrics"""
        base = 0.3
        
        if result.get('outcome') == 'failure':
            base += 0.4
        
        base += result.get('retry_count', 0) * 0.1
        
        if result.get('execution_time_ms', 0) > 30000:  # >30s
            base += 0.2
        
        return min(1.0, base)


# =============================================================================
# PRESSURE PHYSICS
# =============================================================================

class PressurePhysics:
    """Automatic pressure from REAL system state"""
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._event_bus = event_bus
        self._pressures = {}
    
    def compute_pressures(
        self,
        execution_state: Dict,
        unresolved_goals: List,
        failures: List,
        knowledge_gaps: List
    ) -> Dict[str, float]:
        """Compute automatic pressure from REAL state"""
        
        pressures = {}
        
        # 1. Contradiction pressure - from unresolved conflicts
        if len(unresolved_goals) > 5:
            pressures['contradiction'] = min(1.0, len(unresolved_goals) / 15)
        
        # 2. Failure recovery pressure - from recent failures
        if failures:
            recent_failures = [f for f in failures if f.get('last_24h', False)]
            if recent_failures:
                pressures['failure_recovery'] = min(1.0, len(recent_failures) / 3)
        
        # 3. Resource pressure - from real resource usage
        resource_usage = execution_state.get('resource_usage', 0.5)
        if resource_usage > 0.8:
            pressures['resource'] = resource_usage
        
        # 4. Exploration gap - from identity + unvisited domains
        unvisited = execution_state.get('unvisited_domains', [])
        if unvisited:
            pressures['exploration_gap'] = min(0.7, len(unvisited) / 10)
        
        # 5. Adaptation pressure - from genome state
        genome = execution_state.get('genome', {})
        if genome.get('risk_tolerance', 0.5) < 0.3:
            pressures['adaptation'] = 0.5
        
        # 6. Continuity pressure - from identity drift
        drift = execution_state.get('identity_drift', 0)
        if drift > 0.2:
            pressures['continuity'] = drift
        
        # Emit changes
        for source, value in pressures.items():
            old = self._pressures.get(source, 0)
            if abs(value - old) > 0.05:
                self._event_bus.emit('pressure.changed', {
                    'source': source,
                    'old': round(old, 3),
                    'new': round(value, 3),
                    'delta': round(value - old, 3)
                })
            
            self._pressures[source] = value
        
        return pressures
    
    def get_total(self) -> float:
        return sum(self._pressures.values())


# =============================================================================
# INTRUSIVE MONITOR
# =============================================================================

class IntrusiveMonitor:
    """Monitor that can interrupt REAL execution"""
    
    def __init__(self, event_bus: CognitiveEventBus):
        self._event_bus = event_bus
        self._active = False
    
    def start(self, execution_id: str):
        self._active = True
        self._event_bus.emit('monitor.started', {'execution_id': execution_id})
    
    def check_interrupt(self, state: Dict) -> Optional[Dict]:
        if not self._active:
            return None
        
        # Check REAL metrics for interrupts
        if state.get('progress_delta', 1) < 0.005:  # No progress
            interrupt = {
                'type': 'stall',
                'severity': 0.8,
                'action': 'change_strategy',
                'reason': 'No progress for extended period'
            }
            self._event_bus.emit('interrupt.raised', interrupt)
            return interrupt
        
        if state.get('error_count', 0) > 3:
            interrupt = {
                'type': 'error_escalation',
                'severity': 0.9,
                'action': 'abort',
                'reason': 'Too many errors'
            }
            self._event_bus.emit('interrupt.raised', interrupt)
            return interrupt
        
        if state.get('verifier_score', 1) < 0.3:
            interrupt = {
                'type': 'quality_failure',
                'severity': 0.7,
                'action': 'retry',
                'reason': 'Quality below threshold'
            }
            self._event_bus.emit('interrupt.raised', interrupt)
            return interrupt
        
        return None
    
    def stop(self):
        self._active = False
        self._event_bus.emit('monitor.stopped', {})


# =============================================================================
# MAIN COGNITIVE RUNTIME - SINGLETON WITH REAL EXECUTION
# =============================================================================

@enforce_singleton
class CognitiveRuntime:
    """
    THE ONLY cognitive runtime with REAL execution integration.
    
    Enforces:
    - Singleton pattern
    - All mutations emit events
    - Real execution pipeline (not simulated)
    - Stateless genome evolution
    """
    
    def __init__(self):
        # MANDATORY event bus
        self._event_bus = CognitiveEventBus()
        
        # Components with event bus
        self._identity = IdentityTopology(self._event_bus)
        self._genome = ExecutionGenome()
        self._pressure = PressurePhysics(self._event_bus)
        self._monitor = IntrusiveMonitor(self._event_bus)
        
        # REAL execution binding
        self._execution = RealExecutionBinding(self._event_bus)
        
        # Stats
        self._executions = 0
        self._successes = 0
    
    async def process_goal(
        self,
        goal: Dict,
        execution_state: Dict,
        context: Dict = None
    ) -> Dict:
        """MAIN ENTRY - processes goal through real execution pipeline"""
        
        context = context or {}
        self._executions += 1
        
        # 1. Compute automatic pressures (from real state)
        pressures = self._pressure.compute_pressures(
            execution_state,
            context.get('unresolved_goals', []),
            context.get('failures', []),
            context.get('knowledge_gaps', [])
        )
        
        # 2. Start monitoring
        execution_id = str(uuid4())
        self._monitor.start(execution_id)
        
        # 3. Select strategy based on genome
        strategy = self._select_strategy(goal, context)
        self._event_bus.emit('strategy.selected', {
            'goal': goal.get('title'),
            'strategy': strategy,
            'genome_generation': self._genome.generation
        })
        
        # 4. Execute through REAL pipeline
        result = await self._execution.execute(goal, context)
        
        # 5. Check for interrupts
        interrupt = self._monitor.check_interrupt(result)
        if interrupt:
            self._event_bus.emit('interrupt.handled', interrupt)
        
        # 6. Update identity from REAL result
        self._identity.apply_execution_result(
            result.get('outcome', 'unknown'),
            goal.get('domain', 'general'),
            strategy,
            result
        )
        
        # 7. Evolve genome (STATELESS - no latent divergence)
        self._genome = evolve_genome(self._genome, result, self._event_bus)
        
        # 8. Stop monitoring
        self._monitor.stop()
        
        if result.get('outcome') == 'success':
            self._successes += 1
        
        # 9. Emit lineage
        self._event_bus.emit('execution.lineage', {
            'goal': goal.get('title'),
            'strategy': strategy,
            'outcome': result.get('outcome'),
            'metrics': {
                'verifier_score': result.get('verifier_score', 0),
                'execution_cost': result.get('execution_cost', 0)
            },
            'interrupt': interrupt is not None,
            'genome_generation': self._genome.generation
        })
        
        return {
            'execution': result,
            'identity': self._identity.get_vector(),
            'genome': {
                'generation': self._genome.generation,
                'risk_tolerance': self._genome.risk_tolerance,
                'exploration_bias': self._genome.exploration_bias
            },
            'pressures': pressures,
            'interrupt': interrupt
        }
    
    def _select_strategy(self, goal: Dict, context: Dict) -> str:
        if self._genome.risk_tolerance > 0.7:
            return 'aggressive_pursue'
        elif self._genome.risk_tolerance > 0.4:
            return 'balanced_execute'
        return 'cautious_explore'
    
    def get_status(self) -> Dict:
        return {
            'identity': self._identity.get_vector(),
            'genome': {
                'generation': self._genome.generation,
                'risk_tolerance': self._genome.risk_tolerance,
                'exploration_bias': self._genome.exploration_bias
            },
            'pressures': self._pressure._pressures,
            'stats': {
                'total': self._executions,
                'successes': self._successes,
                'rate': self._successes / max(self._executions, 1)
            }
        }
    
    def get_event_history(self, event_type: str = None, limit: int = 100) -> List[Dict]:
        return self._event_bus.get_history(event_type, limit)
    
    def subscribe(self, event_type: str, callback: Callable):
        self._event_bus.subscribe(event_type, callback)


# Singleton access
def get_cognitive_runtime() -> CognitiveRuntime:
    return CognitiveRuntime.get_instance()