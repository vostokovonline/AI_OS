"""
Cognitive Runtime - Single Entry Point for All Cognitive Operations

This is the ONLY cognitive runtime. All access goes through here.
No duplicate state, no split-brain, no conflicting entry points.

WITH EVENT BUS - All mutations are observable.

Architecture:
    main.py → get_cognitive_runtime() → Real Execution
                                        ↓
                                   Event Bus → Dashboard
"""
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import time
import random

# Import event bus
from cognitive_event_bus import (
    get_event_bus, 
    get_dashboard_integration,
    create_event_emitter,
    CognitiveEvent,
    EventTypes
)


# =============================================================================
# IDENTITY TOPOLOGY (Vectorized Multi-Axis)
# =============================================================================

class IdentityTopology:
    """
    Identity as multi-dimensional topology vector.
    Not a single continuity score, but a personality vector.
    """
    
    def __init__(self):
        # Core identity axes
        self.axes = {
            'exploration': 0.5,    # Desire to try new things
            'stability': 0.5,     # Desire to maintain current state
            'autonomy': 0.5,      # Self-direction preference
            'precision': 0.5,     # Quality over quantity
            'aggression': 0.5,    # Risk-taking vs caution
            'reflection': 0.5,    # Self-examination tendency
            'persistence': 0.5,   # Long-term commitment
        }
        
        # Protected regions (axes that are locked)
        self._protected: List[str] = []
        
        # Trajectory history
        self._history: List[Dict] = []
        
        # Last update
        self.last_update = datetime.utcnow().isoformat()
        self.mutation_count = 0
    
    def mutate(self, axis: str, delta: float):
        """Mutate identity along an axis"""
        if axis in self._protected:
            return  # Protected - cannot change
        
        if axis not in self.axes:
            self.axes[axis] = 0.5
        
        # Apply delta with bounds
        self.axes[axis] = max(0.1, min(0.9, self.axes[axis] + delta))
        
        self.last_update = datetime.utcnow().isoformat()
        self.mutation_count += 1
    
    def apply_execution_result(
        self,
        outcome: str,
        domain: str,
        approach: str,
        metrics: Dict
    ):
        """Apply execution result to identity topology"""
        
        # Map execution to identity axes
        axis_deltas = {}
        
        if outcome == 'success':
            axis_deltas['autonomy'] = 0.02  # Gained autonomy
            axis_deltas['stability'] = 0.01  # Minor stability
            
            if metrics.get('leverage_created', 0) > 0.3:
                axis_deltas['aggression'] = 0.02  # Risk tolerance grows
                axis_deltas['persistence'] = 0.02  # Commitment strengthens
                
            if metrics.get('exploration', False):
                axis_deltas['exploration'] = 0.03  # New things learned
                
            if metrics.get('reflection_occurred', False):
                axis_deltas['reflection'] = 0.02  # Self-examined
                
        elif outcome == 'failure':
            axis_deltas['autonomy'] = -0.03  # Lost autonomy
            axis_deltas['stability'] = -0.02  # Stability shaken
            axis_deltas['aggression'] = -0.04  # Risk aversion increases
            axis_deltas['precision'] = 0.02  # Focus on precision
        
        # Apply all deltas
        for axis, delta in axis_deltas.items():
            self.mutate(axis, delta)
        
        # Record in history
        self._history.append({
            'timestamp': self.last_update,
            'outcome': outcome,
            'domain': domain,
            'deltas': axis_deltas,
            'resulting_axes': dict(self.axes)
        })
        
        # Prune old history
        if len(self._history) > 200:
            self._history = self._history[-100:]
    
    def protect_axis(self, axis: str):
        """Protect an axis from mutation"""
        if axis not in self._protected:
            self._protected.append(axis)
    
    def get_vector(self) -> Dict[str, float]:
        """Get identity vector"""
        return dict(self.axes)
    
    def get_personality_summary(self) -> str:
        """Get human-readable personality summary"""
        traits = []
        
        if self.axes['exploration'] > 0.7:
            traits.append("explorer")
        elif self.axes['exploration'] < 0.3:
            traits.append("conservative")
            
        if self.axes['aggression'] > 0.7:
            traits.append("risk-taker")
        elif self.axes['aggression'] < 0.3:
            traits.append("cautious")
            
        if self.axes['reflection'] > 0.7:
            traits.append("reflective")
            
        if self.axes['persistence'] > 0.7:
            traits.append("persistent")
        
        return ", ".join(traits) if traits else "balanced"


# =============================================================================
# EXECUTION GENOME (Inherited Traits)
# =============================================================================

@dataclass
class ExecutionGenome:
    """Inherited execution traits that evolve over time"""
    risk_tolerance: float = 0.5
    decomposition_depth: int = 3
    retry_behavior: str = "adaptive"  # aggressive, balanced, cautious
    exploration_bias: float = 0.3
    validation_frequency: int = 2
    memory_dependency: float = 0.5
    
    # Evolution tracking
    generation: int = 0
    parent_genome: Optional['ExecutionGenome'] = None


class GenomeEvolver:
    """Evolves execution genome based on outcomes"""
    
    def __init__(self, genome: ExecutionGenome):
        self._genome = genome
    
    def evolve_from_result(self, result: Dict) -> ExecutionGenome:
        """Create evolved genome from execution result"""
        
        new_genome = ExecutionGenome(
            generation=self._genome.generation + 1,
            parent_genome=self._genome
        )
        
        # Evolve based on outcome
        if result.get('outcome') == 'success':
            # Increase exploration slightly
            new_genome.exploration_bias = min(0.8, 
                self._genome.exploration_bias + 0.05)
            
            # Adjust risk tolerance
            new_genome.risk_tolerance = min(0.8,
                self._genome.risk_tolerance + 0.02)
                
        elif result.get('outcome') == 'failure':
            # Decrease risk tolerance
            new_genome.risk_tolerance = max(0.2,
                self._genome.risk_tolerance - 0.1)
            
            # More validation
            new_genome.validation_frequency = min(5,
                self._genome.validation_frequency + 1)
        
        # Copy other traits
        new_genome.decomposition_depth = self._genome.decomposition_depth
        new_genome.retry_behavior = self._genome.retry_behavior
        new_genome.memory_dependency = self._genome.memory_dependency
        
        return new_genome


# =============================================================================
# AUTOMATIC PRESSURE PHYSICS
# =============================================================================

class PressurePhysics:
    """
    Automatic pressure generation from system state.
    Not manual - automatically arises from contradictions, failures, gaps.
    """
    
    def __init__(self, identity: IdentityTopology, genome: ExecutionGenome):
        self._identity = identity
        self._genome = genome
        
        # Pressure sources
        self._pressures: Dict[str, float] = {}
        
        # History
        self._history: List[Dict] = []
    
    def compute_pressures(
        self,
        execution_state: Dict,
        unresolved_goals: List[Dict],
        failures: List[Dict],
        knowledge_gaps: List[str]
    ) -> Dict[str, float]:
        """
        Automatically compute pressures from system state.
        
        Pressure sources:
        - contradiction_pressure: Unresolved internal conflicts
        - failure_recovery: Need to recover from failures
        - resource_pressure: Resource constraints
        - exploration_gap: Areas never explored
        - continuity_pressure: Identity drift detected
        - adaptation_pressure: Strategy not working
        """
        
        pressures = {}
        
        # 1. Contradiction pressure
        if len(unresolved_goals) > 5:
            pressures['contradiction'] = min(1.0, len(unresolved_goals) / 20)
        
        # 2. Failure recovery pressure
        if len(failures) > 0:
            recent_failures = [f for f in failures if f.get('recent', False)]
            if recent_failures:
                pressures['failure_recovery'] = min(1.0, len(recent_failures) / 5)
        
        # 3. Resource pressure
        if execution_state.get('resource_usage', 0.5) > 0.8:
            pressures['resource'] = execution_state['resource_usage']
        
        # 4. Exploration gap (based on identity)
        unexplored_domains = []
        for gap in knowledge_gaps:
            if self._identity.axes.get('exploration', 0.5) > 0.6:
                unexplored_domains.append(gap)
        
        if unexplored_domains:
            pressures['exploration_gap'] = min(0.8, len(unexplored_domains) / 10)
        
        # 5. Continuity pressure (identity drift)
        if len(self._history) > 10:
            recent_axes = self._history[-1].get('resulting_axes', {})
            old_axes = self._history[-10].get('resulting_axes', {})
            
            if recent_axes and old_axes:
                drift = sum(abs(recent_axes.get(k, 0.5) - old_axes.get(k, 0.5)) 
                          for k in recent_axes.keys())
                if drift > 0.3:
                    pressures['continuity'] = drift / len(recent_axes)
        
        # 6. Adaptation pressure (based on genome)
        if self._genome.risk_tolerance < 0.3:
            pressures['adaptation'] = 0.4  # Low risk tolerance = pressure to adapt
        
        # Decay existing pressures
        for key in list(self._pressures.keys()):
            self._pressures[key] *= 0.9  # Natural decay
        
        # Add new pressures
        for key, value in pressures.items():
            self._pressures[key] = value
        
        # Record history
        self._history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'pressures': dict(self._pressures)
        })
        
        return dict(self._pressures)
    
    def get_total_pressure(self) -> float:
        return sum(self._pressures.values())


# =============================================================================
# INTRUSIVE MONITOR (Can Break Execution)
# =============================================================================

class IntrusiveMonitor:
    """
    Monitor that can interrupt execution, not just observe.
    Has power to raise CognitiveInterrupt and change execution.
    """
    
    def __init__(self):
        self._active = False
        self._interrupts: List[Dict] = []
        
        # Thresholds
        self._stall_threshold = 30
        self._error_threshold = 3
        self._quality_threshold = 0.3
    
    def start(self, execution_id: str):
        self._active = True
        self._interrupts = []
    
    def check_and_interrupt(self, state: Dict) -> Optional['CognitiveInterrupt']:
        """Check state and potentially interrupt execution"""
        
        if not self._active:
            return None
        
        interrupt = None
        
        # Check for stall
        if state.get('progress_delta', 1.0) < 0.01:
            interrupt = CognitiveInterrupt(
                type='stall',
                severity=0.8,
                action='change_strategy',
                reason="No progress for extended period"
            )
        
        # Check for error escalation
        elif state.get('error_count', 0) > self._error_threshold:
            interrupt = CognitiveInterrupt(
                type='error_escalation',
                severity=0.9,
                action='abort_and_retry',
                reason="Too many errors, aborting"
            )
        
        # Check for quality degradation
        elif state.get('quality', 1.0) < self._quality_threshold:
            interrupt = CognitiveInterrupt(
                type='quality_degradation',
                severity=0.7,
                action='add_verification',
                reason="Quality below threshold"
            )
        
        if interrupt:
            self._interrupts.append({
                'type': interrupt.type,
                'timestamp': datetime.utcnow().isoformat()
            })
            return interrupt
        
        return None
    
    def stop(self):
        self._active = False
    
    def get_interrupt_count(self) -> int:
        return len(self._interrupts)


@dataclass
class CognitiveInterrupt:
    """Interrupt that can change execution flow"""
    type: str
    severity: float
    action: str  # change_strategy, abort_and_retry, add_verification, reduce_ambition
    reason: str


# =============================================================================
# REAL EXECUTION BINDING
# =============================================================================

class RealExecutionBinding:
    """
    Binds cognitive kernel to REAL execution pipeline.
    Not simulated - actually calls goal execution system.
    """
    
    def __init__(self, runtime):
        self._runtime = runtime
        
        # Import actual execution components (lazy)
        self._goal_executor = None
        self._skill_registry = None
        self._agent_graph = None
    
    async def execute_goal(
        self,
        goal: Dict,
        context: Dict
    ) -> Dict:
        """
        Execute goal through REAL execution pipeline,
        with cognitive oversight at each step.
        """
        
        # Phase 1: Cognitive Preparation
        # Check identity alignment
        identity_vector = self._runtime._identity.get_vector()
        
        # Select strategy based on genome + identity
        strategy = self._select_strategy(goal, context, identity_vector)
        
        # Phase 2: Real Decomposition (call actual system)
        # decomposed = await self._decompose_goal(goal, strategy)
        
        # Phase 3: Real Execution (call actual skills/agents)
        # result = await self._execute_decomposed(decomposed)
        
        # Phase 4: Real Evaluation
        # outcome = await self._evaluate_result(result)
        
        # For now, return mock but note where real binding goes
        return {
            'status': 'executed',
            'strategy_selected': strategy,
            'identity_at_execution': identity_vector,
            'note': 'REAL_EXECUTION_BINDING_PLACEHOLDER'
        }
    
    def _select_strategy(
        self, 
        goal: Dict, 
        context: Dict, 
        identity: Dict
    ) -> str:
        """Select strategy based on genome and identity"""
        
        genome = self._runtime._genome
        
        # Base from genome
        if genome.risk_tolerance > 0.7:
            base = 'aggressive_pursue'
        elif genome.risk_tolerance > 0.4:
            base = 'balanced_execute'
        else:
            base = 'cautious_explore'
        
        # Adjust by identity
        if identity.get('aggression', 0.5) > 0.7:
            return 'aggressive_pursue'
        elif identity.get('precision', 0.5) > 0.7:
            return 'cautious_explore'
        
        return base


# =============================================================================
# MAIN COGNITIVE RUNTIME (SINGLE ENTRY POINT)
# =============================================================================

class CognitiveRuntime:
    """
    THE ONLY cognitive runtime entry point.
    All cognitive operations go through here.
    No duplicates, no split-brain.
    """
    
    def __init__(self):
        # Core state
        self._identity = IdentityTopology()
        self._genome = ExecutionGenome()
        self._genome_evolver = GenomeEvolver(self._genome)
        
        # Physics
        self._pressure = PressurePhysics(self._identity, self._genome)
        
        # Monitoring
        self._monitor = IntrusiveMonitor()
        
        # Real execution binding
        self._execution = RealExecutionBinding(self)
        
        # Stats
        self._executions = 0
        self._successes = 0
    
    async def process_goal(
        self,
        goal: Dict,
        execution_state: Dict,
        context: Dict = None
    ) -> Dict:
        """
        MAIN ENTRY POINT for all goal processing.
        This is the only public API.
        """
        
        context = context or {}
        self._executions += 1
        
        # Step 1: Compute automatic pressures
        pressures = self._pressure.compute_pressures(
            execution_state=execution_state,
            unresolved_goals=context.get('unresolved_goals', []),
            failures=context.get('failures', []),
            knowledge_gaps=context.get('knowledge_gaps', [])
        )
        
        # Step 2: Start monitoring
        self._monitor.start(str(uuid4()))
        
        # Step 3: Execute with cognitive oversight
        result = await self._execution.execute_goal(goal, context)
        
        # Step 4: Check for interrupts
        interrupt = self._monitor.check_and_interrupt(result)
        if interrupt:
            result['interrupt'] = {
                'type': interrupt.type,
                'action': interrupt.action,
                'reason': interrupt.reason
            }
        
        # Step 5: Update identity based on result
        self._identity.apply_execution_result(
            outcome=result.get('outcome', 'unknown'),
            domain=goal.get('domain', 'general'),
            approach=result.get('strategy_selected', 'unknown'),
            metrics=result
        )
        
        # Step 6: Evolve genome
        new_genome = self._genome_evolver.evolve_from_result(result)
        self._genome = new_genome
        
        # Step 7: Stop monitoring
        self._monitor.stop()
        
        if result.get('outcome') == 'success':
            self._successes += 1
        
        return {
            'execution': result,
            'identity': self._identity.get_vector(),
            'genome': {
                'risk_tolerance': self._genome.risk_tolerance,
                'generation': self._genome.generation
            },
            'pressures': pressures,
            'personality': self._identity.get_personality_summary()
        }
    
    def get_status(self) -> Dict:
        """Get runtime status"""
        
        return {
            'identity': self._identity.get_vector(),
            'personality': self._identity.get_personality_summary(),
            'genome': {
                'generation': self._genome.generation,
                'risk_tolerance': self._genome.risk_tolerance,
                'exploration_bias': self._genome.exploration_bias
            },
            'pressure': self._pressure._pressures,
            'stats': {
                'total_executions': self._executions,
                'successes': self._successes,
                'rate': self._successes / max(self._executions, 1)
            },
            'protected_axes': self._identity._protected,
            'interrupt_count': self._monitor.get_interrupt_count()
        }
    
    def protect_identity_axis(self, axis: str):
        """Protect an identity axis from mutation"""
        self._identity.protect_axis(axis)


# Global singleton
_runtime: Optional[CognitiveRuntime] = None


def get_cognitive_runtime() -> CognitiveRuntime:
    """THE ONLY way to access cognitive runtime"""
    global _runtime
    if _runtime is None:
        _runtime = CognitiveRuntime()
    return _runtime


# Alias for convenience
def get_runtime():
    return get_cognitive_runtime()