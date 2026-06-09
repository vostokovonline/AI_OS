"""
Cognitive Orchestrator - Inversion of Control

This is the NEW core that bridges:
- Cognitive Core (beliefs, attractors, identity)
- Execution OS (goals, skills, planning)

Instead of: Goal → Execute
Now: Identity → Strategic Pressure → Goal Crystallization → Execution Planning → Execution

This transforms AI-OS from "workflow engine" to "cognitive organism".

UPDATED with:
- Strategic Execution Memory
- Runtime Self-Monitoring  
- Goal Pressure System
- Persistent Identity Topology
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import hashlib
import time


# =============================================================================
# LAYER 1: IDENTITY STATE (Persistent Topology)
# =============================================================================

@dataclass
class IdentityState:
    """Persistent cognitive identity - the 'self' of the system"""
    identity_id: str = field(default_factory=lambda: str(uuid4()))
    core_beliefs: Dict[str, float] = field(default_factory=dict)  # belief_id -> stability
    priority_vectors: Dict[str, float] = field(default_factory=dict)  # goal_type -> priority
    continuity_score: float = 1.0  # 0-1: how stable is identity
    last_update: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Protected regions (core attractors that must survive)
    protected_topology: List[str] = field(default_factory=list)
    
    # Trajectory history for continuity checking
    goal_history: List[Dict] = field(default_factory=list)


@dataclass
class CognitiveDissonance:
    """Measures pressure on identity topology"""
    contradiction_pressure: float = 0.0  # Internal conflicts
    fragmentation_risk: float = 0.0      # Topology breaking apart
    drift_velocity: float = 0.0          # Identity shifting
    rupture_risk: float = 0.0            # Catastrophic breakdown risk
    
    def total_pressure(self) -> float:
        return (
            self.contradiction_pressure * 0.4 +
            self.fragmentation_risk * 0.3 +
            self.drift_velocity * 0.2 +
            self.rupture_risk * 0.1
        )


class IdentityPersistenceEngine:
    """Manages identity continuity across goals and time"""
    
    def __init__(self):
        self._current_identity: Optional[IdentityState] = None
        self._identity_history: List[IdentityState] = []
    
    def get_or_create_identity(self) -> IdentityState:
        if self._current_identity is None:
            self._current_identity = IdentityState()
        return self._current_identity
    
    def evaluate_goal_impact(
        self, 
        goal: Dict[str, Any],
        current_beliefs: Dict[str, Any]
    ) -> Tuple[float, float]:
        """
        Evaluate how a goal affects identity continuity.
        
        Returns: (continuity_impact, strategic_priority)
        - continuity_impact: -1 (destabilizes) to +1 (strengthens)
        - strategic_priority: 0-1 based on identity alignment
        """
        identity = self.get_or_create_identity()
        
        goal_type = goal.get('goal_type', 'achievable')
        goal_domain = goal.get('domain', 'general')
        
        # Check alignment with priority vectors
        priority = identity.priority_vectors.get(goal_type, 0.5)
        
        # Check conflict with core beliefs
        continuity_impact = 0.5  # Default neutral
        if goal_domain in identity.core_beliefs:
            belief_strength = identity.core_beliefs[goal_domain]
            # Strong belief alignment = positive impact
            continuity_impact = belief_strength * 0.5
        
        # Check protection violation
        for protected in identity.protected_topology:
            if protected in goal.get('title', ''):
                continuity_impact = -0.3  # Threatens protected region
        
        return continuity_impact, priority
    
    def update_after_execution(
        self,
        goal: Dict[str, Any],
        outcome: str,
        artifacts: List[Dict]
    ):
        """Update identity state based on goal outcome"""
        identity = self.get_or_create_identity()
        
        # Add to history
        identity.goal_history.append({
            'goal_id': goal.get('id', ''),
            'type': goal.get('goal_type', 'unknown'),
            'outcome': outcome,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Update continuity based on outcome
        if outcome == 'success':
            identity.continuity_score = min(1.0, identity.continuity_score + 0.05)
        elif outcome == 'failure':
            identity.continuity_score = max(0.1, identity.continuity_score - 0.1)
        
        identity.last_update = datetime.utcnow().isoformat()
        
        # Prune history if too long
        if len(identity.goal_history) > 100:
            identity.goal_history = identity.goal_history[-50:]


# =============================================================================
# LAYER 2: COGNITIVE DISSONANCE MEASUREMENT
# =============================================================================

class DissonanceGradient:
    """Measures and minimizes cognitive dissonance"""
    
    def __init__(self):
        self._pressure_history: List[float] = []
    
    def compute_dissonance(
        self,
        beliefs: Dict[str, Any],
        new_goal: Dict[str, Any]
    ) -> CognitiveDissonance:
        """Compute how much a goal conflicts with current belief topology"""
        
        dissonance = CognitiveDissonance()
        
        # Contradiction pressure - how many beliefs conflict with goal
        goal_claims = set(new_goal.get('claims', []))
        belief_claims = set(beliefs.keys())
        
        conflict_count = len(goal_claims & belief_claims)
        dissonance.contradiction_pressure = min(1.0, conflict_count / 10.0)
        
        # Fragmentation risk - too many unrelated goals
        if len(beliefs) > 50:
            dissonance.fragmentation_risk = 0.7
        elif len(beliefs) > 30:
            dissonance.fragmentation_risk = 0.4
        
        # Drift velocity - how fast beliefs are changing
        if len(self._pressure_history) > 10:
            recent = self._pressure_history[-10:]
            avg = sum(recent) / len(recent)
            dissonance.drift_velocity = abs(recent[-1] - avg)
        
        # Rupture risk - catastrophic contradiction
        if dissonance.contradiction_pressure > 0.8:
            dissonance.rupture_risk = 0.9
        elif dissonance.contradiction_pressure > 0.5:
            dissonance.rupture_risk = 0.5
        
        return dissonance
    
    def record_pressure(self, pressure: float):
        self._pressure_history.append(pressure)
        if len(self._pressure_history) > 100:
            self._pressure_history = self._pressure_history[-50:]


# =============================================================================
# LAYER 3: STRATEGIC COGNITION
# =============================================================================

@dataclass
class StrategicPressure:
    """How goals create strategic pressure on the system"""
    urgency: float = 0.0        # How time-critical
    importance: float = 0.0     # How core to identity
    opportunity: float = 0.0   # How valuable this moment is
    cost: float = 0.0           # Resources required
    
    def total(self) -> float:
        return self.urgency * 0.2 + self.importance * 0.4 + self.opportunity * 0.4 - self.cost * 0.2


class StrategicFormation:
    """Forms strategy from identity state and goals"""
    
    def __init__(self, identity_engine: IdentityPersistenceEngine):
        self._identity = identity_engine
        self._strategy_history: List[Dict] = []
    
    def compute_strategic_pressure(
        self,
        goal: Dict[str, Any],
        context: Dict[str, Any]
    ) -> StrategicPressure:
        """Compute strategic pressure of a goal"""
        
        pressure = StrategicPressure()
        
        # Urgency from context
        if context.get('time_pressure'):
            pressure.urgency = context['time_pressure']
        
        # Importance from identity alignment
        continuity, priority = self._identity.evaluate_goal_impact(goal, {})
        pressure.importance = priority * (1 + continuity)
        
        # Opportunity from environment
        if context.get('opportunity_detected'):
            pressure.opportunity = 0.8
        
        # Cost from resource availability
        resources = context.get('available_resources', 1.0)
        pressure.cost = 1.0 - resources
        
        return pressure
    
    def select_strategy(
        self,
        goals: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Tuple[str, List[Dict]]:
        """
        Select best strategy given identity state and available goals.
        
        Returns: (strategy_name, prioritized_goals)
        """
        if not goals:
            return "wait", []
        
        # Score each goal
        scored = []
        for goal in goals:
            pressure = self.compute_strategic_pressure(goal, context)
            score = pressure.total()
            scored.append((goal, score))
        
        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Strategy selection based on top goals
        top_score = scored[0][1] if scored else 0
        
        if top_score > 0.7:
            strategy = "aggressive_pursue"
        elif top_score > 0.4:
            strategy = "balanced_execute"
        elif top_score > 0.1:
            strategy = "cautious_explore"
        else:
            strategy = "maintain_continuity"
        
        return strategy, [g[0] for g in scored[:5]]


# =============================================================================
# LAYER 4: GOAL CRYSTALLIZATION
# =============================================================================

class GoalCrystallization:
    """Converts abstract needs into concrete goals"""
    
    def __init__(self, dissonance: DissonanceGradient):
        self._dissonance = dissonance
        self._pending_needs: List[Dict] = []
    
    def add_need(self, need: Dict[str, Any]):
        """Add a need (abstract requirement) to process"""
        self._pending_needs.append({
            **need,
            'added_at': datetime.utcnow().isoformat(),
            'need_id': str(uuid4())
        })
    
    def crystallize(
        self,
        identity: IdentityState,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert needs into crystallized goals through identity filter"""
        
        crystallized = []
        
        for need in self._pending_needs:
            # Check if this need aligns with identity
            continuity, priority = (
                get_identity_engine().evaluate_goal_impact(need, {})
            )
            
            # Check dissonance impact
            dissonance = self._dissonance.compute_dissonance({}, need)
            
            # Only crystallize if:
            # 1. Positive continuity impact
            # 2. Low dissonance pressure
            # 3. High enough priority
            if continuity >= 0 and dissonance.total_pressure() < 0.6 and priority > 0.3:
                goal = self._create_goal(need, identity, priority, dissonance)
                crystallized.append(goal)
        
        # Clear processed needs
        self._pending_needs = []
        
        return crystallized
    
    def _create_goal(
        self,
        need: Dict,
        identity: IdentityState,
        priority: float,
        dissonance: CognitiveDissonance
    ) -> Dict:
        """Create a concrete goal from a need"""
        
        return {
            'id': str(uuid4()),
            'title': need.get('title', 'Unnamed Goal'),
            'description': need.get('description', ''),
            'goal_type': need.get('goal_type', 'achievable'),
            'domain': need.get('domain', 'general'),
            'priority': priority,
            'dissonance_cost': dissonance.total_pressure(),
            'continuity_alignment': priority,
            'strategic_value': priority * (1 - dissonance.total_pressure()),
            'created_via': 'cognitive_crystallization',
            'source_need': need.get('need_id')
        }


# =============================================================================
# LAYER 5: COGNITIVE ORCHESTRATOR (THE GLUE)
# =============================================================================

class CognitiveOrchestrator:
    """
    The new core that replaces pure goal-driven execution.
    
    Pipeline:
    1. Receive goals/needs
    2. Filter through identity continuity
    3. Measure dissonance
    4. Compute strategic pressure
    5. Crystallize into executable goals
    6. Pass to execution layer
    7. Feed results back to identity
    """
    
    def __init__(self):
        # Layer 1: Identity
        self._identity_engine = IdentityPersistenceEngine()
        
        # Layer 2: Dissonance
        self._dissonance = DissonanceGradient()
        
        # Layer 3: Strategic formation
        self._strategic = StrategicFormation(self._identity_engine)
        
        # Layer 4: Goal crystallization
        self._crystallizer = GoalCrystallization(self._dissonance)
        
        # Execution layer interface (will connect to existing AI-OS)
        self._execution_ready = False
    
    def process_goal_request(
        self,
        request: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main entry point - process goal through cognitive pipeline"""
        
        # Step 1: Add as need
        if 'need' in request:
            self._crystallizer.add_need(request['need'])
        
        # Step 2: Get current identity state
        identity = self._identity_engine.get_or_create_identity()
        
        # Step 3: Crystallize into goals
        crystallized_goals = self._crystallizer.crystallize(identity, context)
        
        if not crystallized_goals:
            return {
                'status': 'deferred',
                'reason': 'goals_rejected_by_identity_filter',
                'suggestion': 'need_more_alignment_or_lower_dissonance'
            }
        
        # Step 4: Compute strategic pressure
        strategy, prioritized = self._strategic.select_strategy(
            crystallized_goals, 
            context
        )
        
        # Step 5: Record dissonance for monitoring
        avg_pressure = sum(
            self._dissonance.compute_dissonance({}, g).total_pressure()
            for g in prioritized
        ) / max(len(prioritized), 1)
        
        self._dissonance.record_pressure(avg_pressure)
        
        # Step 6: Prepare for execution
        execution_plan = {
            'strategy': strategy,
            'goals': prioritized,
            'identity_state': {
                'continuity_score': identity.continuity_score,
                'core_beliefs_count': len(identity.core_beliefs),
                'protected_regions': len(identity.protected_topology)
            },
            'dissonance': {
                'current_pressure': avg_pressure,
                'rupture_risk': avg_pressure > 0.7
            }
        }
        
        return execution_plan
    
    def report_execution_result(
        self,
        goal: Dict[str, Any],
        outcome: str,
        artifacts: List[Dict]
    ):
        """Feed execution result back into identity"""
        
        # Update identity based on outcome
        self._identity_engine.update_after_execution(goal, outcome, artifacts)
        
        # If failure, analyze why for future prevention
        if outcome == 'failure':
            # Could trigger dissonance repair mechanisms
            pass
    
    def get_identity_state(self) -> Dict:
        """Get current identity state for inspection"""
        identity = self._identity_engine.get_or_create_identity()
        return {
            'continuity_score': identity.continuity_score,
            'core_beliefs': identity.core_beliefs,
            'priority_vectors': identity.priority_vectors,
            'protected_topology': identity.protected_topology,
            'goal_history_count': len(identity.goal_history),
            'last_update': identity.last_update
        }
    
    def get_dissonance_status(self) -> Dict:
        """Get current dissonance metrics"""
        return {
            'current_pressure': self._dissonance._pressure_history[-1] if self._dissonance._pressure_history else 0,
            'pressure_trend': 'stable' if len(self._dissonance._pressure_history) < 2 else (
                'increasing' if self._dissonance._pressure_history[-1] > self._dissonance._pressure_history[-2] 
                else 'decreasing'
            )
        }


# Import new components
from strategic_execution_memory import (
    StrategicExecutionMemory, 
    ExecutionRecord, 
    get_strategic_memory
)
from runtime_self_monitor import (
    RuntimeSelfMonitor,
    ExecutionPhase,
    AnomalyType,
    get_runtime_monitor
)


# Global instance
_cognitive_orchestrator: Optional[CognitiveOrchestrator] = None


def get_cognitive_orchestrator() -> CognitiveOrchestrator:
    global _cognitive_orchestrator
    if _cognitive_orchestrator is None:
        _cognitive_orchestrator = CognitiveOrchestrator()
    return _cognitive_orchestrator


def get_identity_engine() -> IdentityPersistenceEngine:
    return get_cognitive_orchestrator()._identity_engine


def get_strategic_memory() -> StrategicExecutionMemory:
    """Get strategic execution memory instance"""
    return get_cognitive_orchestrator()._strategic_memory


def get_runtime_monitor() -> RuntimeSelfMonitor:
    """Get runtime self-monitor instance"""
    return get_cognitive_orchestrator()._runtime_monitor


# =============================================================================
# API ENDPOINTS (to integrate with main.py)
# =============================================================================

"""
These endpoints would be added to main.py:

@app.post("/cognitive/process-goal")
async def process_cognitive_goal(request: dict, context: dict = {}):
    orchestrator = get_cognitive_orchestrator()
    return orchestrator.process_goal_request(request, context)

@app.get("/cognitive/identity-state")
async def get_identity():
    return get_cognitive_orchestrator().get_identity_state()

@app.get("/cognitive/dissonance-status")
async def get_dissonance():
    return get_cognitive_orchestrator().get_dissonance_status()

@app.post("/cognitive/report-outcome")
async def report_outcome(goal: dict, outcome: str, artifacts: list):
    get_cognitive_orchestrator().report_execution_result(goal, outcome, artifacts)
    return {"status": "recorded"}
"""