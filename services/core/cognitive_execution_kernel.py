"""
Cognitive Execution Kernel - Complete Integration

This is the core that makes execution drive cognition:
- Execution mutates identity
- Strategy emerges from execution patterns  
- Goals crystallize from internal pressure
- Runtime adapts while executing

NEW ARCHITECTURE:
Identity State → Pressure Field → Goal Crystallization → 
Adaptive Execution → Runtime Mutation → Strategic Memory → Identity Update
"""
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import time
import random

# Import from other modules
from strategic_execution_memory import ExecutionRecord
from runtime_self_monitor import ExecutionPhase


# =============================================================================
# LAYER 1: IDENTITY TOPOLOGY (Persistent Self)
# =============================================================================

@dataclass
class IdentityTopology:
    """
    Identity emerges from repeated patterns, not explicit storage.
    This is the "self" of the system - emergent from execution history.
    """
    # Core patterns (emergent, not stored)
    preferred_domains: List[str] = field(default_factory=list)
    protected_approaches: List[str] = field(default_factory=list)
    strategic_tendencies: Dict[str, float] = field(default_factory=dict)  # approach -> strength
    
    # Continuity metrics
    continuity_score: float = 1.0
    coherence_score: float = 1.0
    
    # Protected regions (high-stability attractors)
    protected_topology: List[str] = field(default_factory=list)
    
    # Last mutation
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    mutation_count: int = 0
    
    def register_execution_pattern(
        self, 
        domain: str, 
        approach: str, 
        outcome: str,
        continuity_impact: float
    ):
        """Update identity based on execution pattern"""
        
        # Track domain preference
        if domain not in self.preferred_domains:
            if outcome == 'success':
                self.preferred_domains.append(domain)
        
        # Track approach strength
        if approach not in self.strategic_tendencies:
            self.strategic_tendencies[approach] = 0.5
        
        n = self.mutation_count + 1
        
        # Update approach strength
        if outcome == 'success':
            self.strategic_tendencies[approach] = (
                self.strategic_tendencies[approach] * (n-1) + 1.0
            ) / n
        elif outcome == 'failure':
            self.strategic_tendencies[approach] = (
                self.strategic_tendencies[approach] * (n-1)
            ) / n
        
        # Update continuity
        self.continuity_score = max(0.1, min(1.0, self.continuity_score + continuity_impact))
        
        # Update coherence (how consistent is identity)
        if len(self.strategic_tendencies) > 1:
            values = list(self.strategic_tendencies.values())
            avg = sum(values) / len(values)
            variance = sum((v - avg) ** 2 for v in values) / len(values)
            self.coherence_score = max(0, 1 - variance)
        
        self.last_update = datetime.utcnow().isoformat()
        self.mutation_count += 1
    
    def get_strategic_preference(self, approach: str) -> float:
        """Get preference strength for an approach"""
        return self.strategic_tendencies.get(approach, 0.5)
    
    def is_protected(self, domain: str) -> bool:
        """Check if domain is protected"""
        return domain in self.protected_topology


# =============================================================================
# LAYER 2: PRESSURE FIELD (Internal Motivation)
# =============================================================================

class PressureField:
    """
    Internal pressure that drives goal formation.
    Not user-driven, but system-generated needs.
    """
    
    def __init__(self):
        self._pressure_sources: Dict[str, float] = {}  # source -> pressure
        self._last_decay_time = time.time()
    
    def add_pressure(self, source: str, amount: float):
        """Add pressure from a source"""
        current = self._pressure_sources.get(source, 0)
        self._pressure_sources[source] = min(1.0, current + amount)
    
    def decay_pressures(self, decay_rate: float = 0.01):
        """Decay all pressures over time"""
        now = time.time()
        elapsed = now - self._last_decay_time
        
        if elapsed > 1.0:  # Decay every second
            for source in self._pressure_sources:
                self._pressure_sources[source] = max(
                    0, 
                    self._pressure_sources[source] - decay_rate
                )
            self._last_decay_time = now
    
    def get_total_pressure(self) -> float:
        """Get total pressure across all sources"""
        return sum(self._pressure_sources.values())
    
    def get_top_pressures(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top n pressure sources"""
        sorted_ps = sorted(
            self._pressure_sources.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_ps[:n]
    
    def resolve_pressure(self, source: str, amount: float):
        """Resolve some pressure (after goal execution)"""
        current = self._pressure_sources.get(source, 0)
        self._pressure_sources[source] = max(0, current - amount)


# =============================================================================
# LAYER 3: ADAPTIVE EXECUTOR (Runtime Mutation)
# =============================================================================

class AdaptiveExecutor:
    """
    Executes goals with runtime adaptation capability.
    Can mutate strategy mid-execution based on:
    - Progress feedback
    - Anomaly detection
    - Resource pressure
    - Continuity impact
    """
    
    def __init__(self, identity: IdentityTopology, monitor, strategic_memory):
        self._identity = identity
        self._monitor = monitor
        self._strategic_memory = strategic_memory
        
        # Execution state
        self._current_execution: Optional[Dict] = None
        self._strategy_stack: List[str] = []  # For rollback
        
        # Adaptation rules
        self._adaptation_rules: Dict[str, Callable] = {}
    
    def execute_with_adaptation(
        self,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute goal with runtime adaptation.
        Returns execution result + metadata for identity update.
        """
        
        # Start monitoring
        self._monitor.start_monitoring(str(uuid4()), goal)
        
        # Select initial strategy based on identity
        initial_strategy = self._select_strategy(goal, context)
        self._strategy_stack.append(initial_strategy)
        
        # Track for result
        result = {
            'goal_id': goal.get('id', str(uuid4())),
            'strategy': initial_strategy,
            'mutations': [],  # Track strategy changes
            'artifacts': [],
            'errors': [],
            'continuity_impact': 0.0,
            'leverage_created': 0.0
        }
        
        # Execute with simulated phases
        phase_enums = [
            ExecutionPhase.PLANNING,
            ExecutionPhase.DECOMPOSITION,
            ExecutionPhase.EXECUTION,
            ExecutionPhase.EVALUATION
        ]
        
        for i, phase in enumerate(phase_enums):
            # Update progress
            self._monitor.update_progress(
                phase=phase,
                progress=(i + 1) / len(phase_enums),
                current_step=f"Phase: {phase.value}"
            )
            
            # Check if adaptation needed
            if self._monitor.should_adapt():
                adaptation = self._monitor.get_adaptation_plan()
                
                # Adapt - switch strategy
                new_strategy = self._adapt_strategy(
                    current=result['strategy'],
                    reason=adaptation['primary_anomaly'],
                    goal=goal
                )
                
                result['mutations'].append({
                    'from': result['strategy'],
                    'to': new_strategy,
                    'reason': adaptation['suggested_action']
                })
                
                result['strategy'] = new_strategy
                self._strategy_stack.append(new_strategy)
            
            # Simulate execution work
            time.sleep(0.01)  # Brief pause
            
            # Random success/failure for simulation
            if random.random() > 0.9:
                result['errors'].append(f"Simulated error in {phase}")
        
        # Determine outcome
        if len(result['errors']) > 2:
            result['outcome'] = 'failure'
            result['continuity_impact'] = -0.2
        else:
            result['outcome'] = 'success'
            result['continuity_impact'] = 0.1
            result['leverage_created'] = 0.3
            result['artifacts'] = [{'type': 'knowledge', 'value': 'learned'}]
        
        # Update strategic memory
        self._strategic_memory.record_execution(ExecutionRecord(
            execution_id=str(uuid4()),
            goal_type=goal.get('goal_type', 'unknown'),
            domain=goal.get('domain', 'general'),
            strategy_used=result['strategy'],
            outcome=result['outcome'],
            duration_ms=100,
            artifacts_count=len(result['artifacts']),
            continuity_impact=result['continuity_impact'],
            leverage_creation=result['leverage_created'],
            stability_impact=result['continuity_impact']
        ))
        
        # Stop monitoring
        result['monitor_summary'] = self._monitor.stop_monitoring()
        
        return result
    
    def _select_strategy(self, goal: Dict, context: Dict) -> str:
        """Select strategy based on identity + strategic memory"""
        
        # Get from strategic memory
        recommended = self._strategic_memory.get_recommended_strategy(
            goal.get('goal_type', 'achievable'),
            goal.get('domain', 'general'),
            self._identity.continuity_score
        )
        
        # Adjust based on identity preference
        domain = goal.get('domain', 'general')
        
        # If identity has strong preference, respect it
        if domain in self._identity.preferred_domains:
            # Identity prefers this domain - can be more ambitious
            return recommended
        
        # If protected, be conservative
        if self._identity.is_protected(domain):
            return 'cautious_explore'
        
        return recommended
    
    def _adapt_strategy(
        self, 
        current: str, 
        reason: str, 
        goal: Dict
    ) -> str:
        """Adapt strategy based on anomaly"""
        
        adaptations = {
            'stall': 'aggressive_pursue',
            'error_escalation': 'cautious_explore',
            'resource_depletion': 'maintain_continuity',
            'quality_degradation': 'balanced_execute',
            'drift': 'maintain_continuity'
        }
        
        # Get adaptation
        new_strategy = adaptations.get(reason, current)
        
        # But also consider identity
        # If continuity is low, don't take risks
        if self._identity.continuity_score < 0.5:
            return 'maintain_continuity'
        
        return new_strategy


# =============================================================================
# LAYER 4: COGNITIVE EXECUTION KERNEL (THE COMPLETE SYSTEM)
# =============================================================================

class CognitiveExecutionKernel:
    """
    Complete cognitive execution system.
    
    Pipeline:
    1. Identity provides strategic context
    2. Pressure field generates internal needs
    3. Goal crystallization filters goals through identity
    4. Adaptive executor runs with runtime mutation
    5. Results update identity topology
    6. Strategic memory records patterns
    
    This makes execution drive cognition.
    """
    
    def __init__(self):
        # Core components
        self._identity = IdentityTopology()
        self._pressure = PressureField()
        
        # Import or create dependencies
        from strategic_execution_memory import get_strategic_memory, ExecutionRecord
        from runtime_self_monitor import get_runtime_monitor
        
        self._strategic_memory = get_strategic_memory()
        self._monitor = get_runtime_monitor()
        
        # Adaptive execution
        self._executor = AdaptiveExecutor(
            self._identity, 
            self._monitor, 
            self._strategic_memory
        )
        
        # Execution history
        self._execution_log: List[Dict] = []
        
        # Stats
        self._total_goals = 0
        self._successful_goals = 0
    
    def process_goal(
        self, 
        goal: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a goal through the full cognitive pipeline.
        """
        
        context = context or {}
        self._total_goals += 1
        
        # Step 1: Evaluate goal through identity
        continuity_alignment = self._evaluate_identity_alignment(goal)
        
        # Step 2: Check pressure field
        current_pressure = self._pressure.get_total_pressure()
        
        # Step 3: Execute with adaptation
        result = self._executor.execute_with_adaptation(goal, context)
        
        # Step 4: Update identity based on result
        self._identity.register_execution_pattern(
            domain=goal.get('domain', 'general'),
            approach=result['strategy'],
            outcome=result['outcome'],
            continuity_impact=result['continuity_impact']
        )
        
        # Step 5: Resolve pressure if successful
        if result['outcome'] == 'success':
            self._pressure.resolve_pressure('execution', 0.3)
            self._successful_goals += 1
        
        # Step 6: Add pressure if failed
        if result['outcome'] == 'failure':
            self._pressure.add_pressure('recovery_need', 0.4)
        
        # Step 7: Decay pressures
        self._pressure.decay_pressures()
        
        # Record in history
        self._execution_log.append({
            'goal': goal.get('title'),
            'outcome': result['outcome'],
            'strategy': result['strategy'],
            'identity_state': self.get_identity_state(),
            'pressure_state': self._pressure.get_top_pressures()
        })
        
        # Return enriched result
        return {
            'execution_result': result,
            'cognitive_state': {
                'identity': self.get_identity_state(),
                'pressure': self._pressure.get_top_pressures(),
                'strategic_insights': self._strategic_memory.get_strategic_insights()
            }
        }
    
    def _evaluate_identity_alignment(self, goal: Dict) -> float:
        """Evaluate how well goal aligns with identity"""
        
        domain = goal.get('domain', 'general')
        
        # Check if domain is preferred
        if domain in self._identity.preferred_domains:
            return 0.8
        
        # Check approach preference
        approach = goal.get('strategy', 'balanced')
        preference = self._identity.get_strategic_preference(approach)
        
        # Check if domain is protected
        if self._identity.is_protected(domain):
            return 0.5
        
        return preference
    
    def get_identity_state(self) -> Dict:
        """Get current identity state"""
        return {
            'preferred_domains': self._identity.preferred_domains,
            'strategic_tendencies': self._identity.strategic_tendencies,
            'continuity_score': self._identity.continuity_score,
            'coherence_score': self._identity.coherence_score,
            'protected_topology': self._identity.protected_topology,
            'mutation_count': self._identity.mutation_count,
            'last_update': self._identity.last_update
        }
    
    def get_system_status(self) -> Dict:
        """Get overall system status"""
        
        # Get insights
        insights = self._strategic_memory.get_strategic_insights()
        anomalies = self._strategic_memory.detect_instability_patterns()
        
        return {
            'identity': self.get_identity_state(),
            'pressure': {
                'total': self._pressure.get_total_pressure(),
                'top': self._pressure.get_top_pressures()
            },
            'execution_stats': {
                'total': self._total_goals,
                'successful': self._successful_goals,
                'success_rate': self._successful_goals / max(self._total_goals, 1)
            },
            'strategic_insights': insights,
            'anomalies': anomalies
        }
    
    def add_internal_need(self, need_type: str, amount: float):
        """Add internal need that creates pressure"""
        self._pressure.add_pressure(need_type, amount)
    
    def protect_domain(self, domain: str):
        """Protect a domain (make it high-priority)"""
        if domain not in self._identity.protected_topology:
            self._identity.protected_topology.append(domain)


# Global instance
_kernel: Optional[CognitiveExecutionKernel] = None


def get_cognitive_kernel() -> CognitiveExecutionKernel:
    global _kernel
    if _kernel is None:
        _kernel = CognitiveExecutionKernel()
    return _kernel


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_kernel():
    """Demonstrate the complete cognitive execution kernel"""
    
    kernel = get_cognitive_kernel()
    
    print("=" * 60)
    print("COGNITIVE EXECUTION KERNEL DEMO")
    print("=" * 60)
    
    # Test 1: Execute multiple goals
    print("\n--- Test 1: Goal Execution with Identity Evolution ---\n")
    
    goals = [
        {'title': 'Build web app', 'goal_type': 'achievable', 'domain': 'creation', 'strategy': 'aggressive'},
        {'title': 'Learn AI', 'goal_type': 'continuous', 'domain': 'learning', 'strategy': 'balanced'},
        {'title': 'Fix critical bug', 'goal_type': 'achievable', 'domain': 'maintenance', 'strategy': 'cautious'},
        {'title': 'Explore new tech', 'goal_type': 'exploratory', 'domain': 'research', 'strategy': 'explore'},
        {'title': 'Optimize performance', 'goal_type': 'achievable', 'domain': 'optimization', 'strategy': 'aggressive'},
    ]
    
    for goal in goals:
        result = kernel.process_goal(goal, {'available_resources': 0.8})
        print(f"Goal: {goal['title']}")
        print(f"  Strategy: {result['execution_result']['strategy']}")
        print(f"  Outcome: {result['execution_result']['outcome']}")
        print(f"  Identity alignment: {result['cognitive_state']['identity']['continuity_score']:.2f}")
        print()
    
    # Test 2: Check system state
    print("--- Test 2: System Status ---\n")
    
    status = kernel.get_system_status()
    
    print(f"Total goals: {status['execution_stats']['total']}")
    print(f"Success rate: {status['execution_stats']['success_rate']:.1%}")
    print(f"Identity continuity: {status['identity']['continuity_score']:.2f}")
    print(f"Identity coherence: {status['identity']['coherence_score']:.2f}")
    print(f"Preferred domains: {status['identity']['preferred_domains']}")
    print(f"Strategic tendencies: {list(status['identity']['strategic_tendencies'].keys())}")
    
    print("\n--- Test 3: Strategic Insights ---\n")
    
    insights = status['strategic_insights']
    if 'best_strategies' in insights:
        print(f"Best strategies: {insights['best_strategies']}")
    if insights.get('anomalies'):
        print(f"Anomalies: {insights['anomalies']}")
    
    print("\n" + "=" * 60)
    print("KERNEL DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_kernel()