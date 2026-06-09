"""
State Tension Resolution Loop - Primary Cognitive Mechanism.

Stage: Cognitive Primary Architecture

Previously: Execution was primary, cognition was secondary.
Now: Tension resolution is primary, execution is subordinate.

The cognitive loop:
1. State accumulates tensions (contradictions, pressures, conflicts)
2. Tensions generate salience (what demands attention)
3. Salience drives goal generation (what to do about it)
4. Goals generate execution (how to resolve)
5. Execution reduces tension (feedback loop)
6. Reduced tension changes salience (recursion)

Key difference from execution-driven:
- Goals are NOT external inputs
- Goals emerge FROM internal state tensions
- Execution is a MEANS to resolve tension, not an end
- The system has "wants" independent of external commands
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Set, FrozenSet
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json


class TensionType(Enum):
    """Types of cognitive tension"""
    CONTRADICTION = "contradiction"      # Two beliefs conflict
    PRESSURE = "pressure"                 # Too much demand
    VACUUM = "vacuum"                     # Missing knowledge
    UNSTABILITY = "unstable"             # Identity threatened
    DIVERGENCE = "divergence"            # Trajectory off-course
    OPPORTUNITY = "opportunity"          # New possibility detected
    ATTENTION = "attention"              # Something demands focus


class TensionLifecycle(Enum):
    """Tension lifecycle"""
    LATENT = "latent"        # Building up
    SALIENT = "salient"      # Demanding attention
    ACTIVE = "active"        # Being resolved
    RESOLVED = "resolved"    # Tension reduced
    ESCALATED = "escalated"  # Growing worse


@dataclass(frozen=True)
class Tension:
    """
    A tension in cognitive state.
    
    Tensions are NOT events - they are STATE conditions.
    They emerge from accumulated evidence.
    
    Properties:
    - type: what kind of tension
    - intensity: how strong (0-1)
    - sources: what events/beliefs caused this
    - duration: how long this has been building
    - resolution_pressure: how urgently this needs resolution
    """
    tension_id: str
    type: str  # TensionType value
    intensity: float  # 0-1, how strong
    sources: FrozenSet[str]  # What caused this (belief_ids, event_ids)
    first_detected: str
    last_updated: str
    state: str  # TensionState value
    resolution_attempts: Tuple[str, ...]  # goal_ids that attempted resolution
    resolution_progress: float  # 0-1, how much resolved
    priority_score: float  # Computed priority for attention
    
    def age(self) -> float:
        """How long this tension has existed in hours"""
        try:
            first = datetime.fromisoformat(self.first_detected)
            now = datetime.utcnow()
            return (now - first).total_seconds() / 3600
        except:
            return 0.0
    
    def urgency(self) -> float:
        """How urgent is this tension (combines intensity, age, attempts)"""
        base = self.intensity
        
        # Age increases urgency (capped at 2x)
        age_factor = min(2.0, 1.0 + self.age() / 24)
        
        # Failed attempts increase urgency
        attempt_factor = 1.0 + len(self.resolution_attempts) * 0.15
        
        urgency = base * age_factor * attempt_factor
        return min(1.0, urgency)
    
    def is_salient(self) -> bool:
        return self.priority_score > 0.5 or self.urgency() > 0.7
    
    def is_critical(self) -> bool:
        return self.urgency() > 0.9 or self.resolution_progress > 0.8


@dataclass(frozen=True)
class SalienceMap:
    """
    What demands attention right now.
    
    Salience is computed from all active tensions.
    NOT from external goals or commands.
    
    Properties:
    - tensions: all active tensions
    - total_attention_demand: sum of all urgency
    - top_tensions: most urgent tensions
    - attention_budget: how much attention available
    - allocated_attention: how attention is currently distributed
    """
    tensions: MappingProxyType  # type: ignore
    total_attention_demand: float
    top_tension_ids: Tuple[str, ...]  # Ordered by priority
    attention_budget: float
    allocated_attention: MappingProxyType  # type: ignore  # tension_id -> attention fraction
    version: int
    
    def __post_init__(self):
        for attr in ('tensions', 'allocated_attention'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
    
    def get_top_tensions(self, n: int = 5) -> List[Tension]:
        """Get top N most urgent tensions"""
        sorted_tensions = sorted(
            self.tensions.values(),
            key=lambda t: t.priority_score,
            reverse=True
        )
        return sorted_tensions[:n]
    
    def attention_exhausted(self) -> bool:
        """Check if all attention is allocated"""
        total_allocated = sum(self.allocated_attention.values())
        return total_allocated >= self.attention_budget


@dataclass(frozen=True)
class TensionResolutionGoal:
    """
    A goal that emerges FROM tension, not from external command.
    
    This is the key distinction:
    - External goal: "do X because someone asked"
    - Tension goal: "resolve Y because tension is building"
    
    Properties:
    - source_tension: what tension this goal addresses
    - resolution_strategy: how to resolve
    - expected_impact: how much this would reduce tension
    - generated_by: "cognitive_loop" (not external)
    """
    goal_id: str
    title: str
    description: str
    source_tension_id: str
    resolution_strategy: str  # How this resolves the tension
    expected_tension_reduction: float  # 0-1, how much this would help
    priority: float  # Computed from tension urgency
    generated_at: str
    acceptable_outcomes: Tuple[str, ...]  # What counts as "resolved"
    unacceptable_outcomes: Tuple[str, ...]  # What would make things worse


@dataclass(frozen=True)
class TensionState:
    """
    Complete tension state of the cognitive system.
    
    This is what the cognitive loop operates on.
    NOT execution state - TENSION state.
    """
    tensions: MappingProxyType  # type: ignore
    salience: SalienceMap
    generated_goals: MappingProxyType  # type: ignore
    resolution_history: MappingProxyType  # type: ignore  # tension_id -> list of attempts
    total_tension_energy: float  # Sum of all active tension intensities
    dominant_tension_type: str  # What type dominates
    version: int
    
    def __post_init__(self):
        for attr in ('tensions', 'generated_goals', 'resolution_history'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
    
    @staticmethod
    def compute_hash(state: 'TensionState') -> str:
        data = {
            "version": state.version,
            "tension_count": len(state.tensions),
            "total_energy": state.total_tension_energy,
            "generated_goal_count": len(state.generated_goals),
            "dominant_type": state.dominant_tension_type
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


# Tension Detection Functions

def detect_contradiction_tension(
    belief1_id: str,
    belief2_id: str,
    belief1_confidence: float,
    belief2_confidence: float,
    mutual_exclusion: bool = True
) -> Optional[Tension]:
    """Detect tension from contradicting beliefs"""
    if not mutual_exclusion:
        return None
    
    # Both high confidence = strong tension
    intensity = (belief1_confidence + belief2_confidence) / 2
    
    if intensity < 0.5:
        return None
    
    tension_id = f"tension_contradiction_{belief1_id}_{belief2_id}"
    
    return Tension(
        tension_id=tension_id,
        type=TensionType.CONTRADICTION.value,
        intensity=intensity,
        sources=frozenset([belief1_id, belief2_id]),
        first_detected=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat(),
        state=TensionLifecycle.LATENT.value,
        resolution_attempts=(),
        resolution_progress=0.0,
        priority_score=0.0  # Computed later
    )


def detect_vacuum_tension(
    missing_knowledge: str,
    importance: float
) -> Tension:
    """Detect tension from missing knowledge"""
    tension_id = f"tension_vacuum_{hashlib.md5(missing_knowledge.encode()).hexdigest()[:8]}"
    
    return Tension(
        tension_id=tension_id,
        type=TensionType.VACUUM.value,
        intensity=importance,
        sources=frozenset([missing_knowledge]),
        first_detected=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat(),
        state=TensionLifecycle.LATENT.value,
        resolution_attempts=(),
        resolution_progress=0.0,
        priority_score=0.0
    )


def detect_pressure_tension(
    source: str,
    intensity: float,
    pressure_type: str = "cognitive"
) -> Tension:
    """Detect tension from accumulated pressure"""
    tension_id = f"tension_pressure_{hashlib.md5(source.encode()).hexdigest()[:8]}"
    
    return Tension(
        tension_id=tension_id,
        type=TensionType.PRESSURE.value,
        intensity=intensity,
        sources=frozenset([source]),
        first_detected=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat(),
        state=TensionLifecycle.LATENT.value,
        resolution_attempts=(),
        resolution_progress=0.0,
        priority_score=0.0
    )


# Salience Computation

def compute_salience(
    tensions: Dict[str, Tension],
    attention_budget: float = 1.0
) -> SalienceMap:
    """
    Compute salience map from tensions.
    
    Salience = urgency + novelty - allocated attention
    
    High salience = demands immediate attention
    """
    if not tensions:
        return SalienceMap(
            tensions=MappingProxyType({}),
            total_attention_demand=0.0,
            top_tension_ids=(),
            attention_budget=attention_budget,
            allocated_attention=MappingProxyType({}),
            version=0
        )
    
    # Compute priority scores
    tension_list = []
    total_demand = 0.0
    
    for tension_id, tension in tensions.items():
        urgency = tension.urgency()
        
        # Priority = urgency adjusted by type
        type_weights = {
            TensionType.CONTRADICTION.value: 1.2,  # High priority
            TensionType.UNSTABILITY.value: 1.1,
            TensionType.PRESSURE.value: 1.0,
            TensionType.VACUUM.value: 0.9,
            TensionType.DIVERGENCE.value: 0.8,
            TensionType.OPPORTUNITY.value: 0.7,
            TensionType.ATTENTION.value: 1.0
        }
        
        type_weight = type_weights.get(tension.type, 1.0)
        priority = urgency * type_weight
        
        # Create updated tension with priority
        updated_tension = Tension(
            tension_id=tension.tension_id,
            type=tension.type,
            intensity=tension.intensity,
            sources=tension.sources,
            first_detected=tension.first_detected,
            last_updated=tension.last_updated,
            state=tension.state,
            resolution_attempts=tension.resolution_attempts,
            resolution_progress=tension.resolution_progress,
            priority_score=priority
        )
        
        tension_list.append((tension_id, updated_tension))
        total_demand += urgency
    
    # Sort by priority
    tension_list.sort(key=lambda x: x[1].priority_score, reverse=True)
    
    # Allocate attention (top tensions get priority)
    allocated = {}
    remaining_budget = attention_budget
    
    for tension_id, tension in tension_list:
        if remaining_budget <= 0:
            break
        
        # Allocate proportional to priority, max 50% of budget per tension
        allocation = min(remaining_budget * 0.5, tension.priority_score * remaining_budget)
        allocated[tension_id] = allocation
        remaining_budget -= allocation
    
    return SalienceMap(
        tensions=MappingProxyType(dict(tension_list)),
        total_attention_demand=total_demand,
        top_tension_ids=tuple(t[0] for t in tension_list[:5]),
        attention_budget=attention_budget,
        allocated_attention=MappingProxyType(allocated),
        version=len(tensions)
    )


# Goal Generation from Tensions

def generate_tension_goal(tension: Tension) -> TensionResolutionGoal:
    """
    Generate goal to resolve tension.
    
    This is the core of tension-driven cognition:
    Goals are not external commands.
    Goals emerge from internal state.
    """
    tension_id = tension.tension_id
    tension_type = tension.type
    
    # Different resolution strategies for different tensions
    strategy_map = {
        TensionType.CONTRADICTION.value: "resolve_belief_conflict",
        TensionType.PRESSURE.value: "reduce_cognitive_load",
        TensionType.VACUUM.value: "acquire_missing_knowledge",
        TensionType.UNSTABILITY.value: "restore_identity_stability",
        TensionType.DIVERGENCE.value: "realign_trajectory",
        TensionType.OPPUNITY.value: "exploit_opportunity",
        TensionType.ATTENTION.value: "allocate_attention"
    }
    
    strategy = strategy_map.get(tension_type, "general_resolution")
    
    # Generate title based on tension type
    title_map = {
        TensionType.CONTRADICTION.value: f"Resolve contradiction in {len(tension.sources)} beliefs",
        TensionType.PRESSURE.value: f"Reduce {tension.type} pressure",
        TensionType.VACUUM.value: "Fill knowledge vacuum",
        TensionType.UNSTABILITY.value: "Restore cognitive stability",
        TensionType.DIVERGENCE.value: "Realign from divergence",
        TensionType.OPPUNITY.value: "Investigate opportunity",
        TensionType.ATTENTION.value: "Resolve attention conflict"
    }
    
    title = title_map.get(tension_type, f"Resolve tension {tension_id}")
    
    return TensionResolutionGoal(
        goal_id=f"goal_tension_{tension_id}",
        title=title,
        description=f"Generated to resolve tension: intensity={tension.intensity:.2f}, age={tension.age():.1f}h",
        source_tension_id=tension_id,
        resolution_strategy=strategy,
        expected_tension_reduction=tension.intensity * (1 - tension.resolution_progress),
        priority=tension.urgency(),
        generated_at=datetime.utcnow().isoformat(),
        acceptable_outcomes=("tension_resolved", "tension_reduced", "tension_accepted"),
        unacceptable_outcomes=("tension_escalated", "new_tension_created")
    )


def generate_goals_from_salience(salience: SalienceMap) -> List[TensionResolutionGoal]:
    """Generate goals from all salient tensions"""
    goals = []
    
    for tension_id in salience.top_tension_ids:
        tension = salience.tensions.get(tension_id)
        if tension and tension.is_salient():
            goal = generate_tension_goal(tension)
            goals.append(goal)
    
    return goals


# State Transition Functions

def add_tension(
    state: TensionState,
    tension: Tension
) -> TensionState:
    """Add new tension to state"""
    new_tensions = {**state.tensions, tension.tension_id: tension}
    
    # Recompute salience
    new_salience = compute_salience(new_tensions, state.salience.attention_budget)
    
    # Recompute total energy
    total_energy = sum(t.intensity for t in new_tensions.values())
    
    # Determine dominant type
    type_counts = {}
    for t in new_tensions.values():
        type_counts[t.type] = type_counts.get(t.type, 0) + t.intensity
    
    dominant = max(type_counts, key=type_counts.get) if type_counts else ""
    
    return TensionState(
        tensions=MappingProxyType(new_tensions),
        salience=new_salience,
        generated_goals=state.generated_goals,
        resolution_history=state.resolution_history,
        total_tension_energy=total_energy,
        dominant_tension_type=dominant,
        version=state.version + 1
    )


def escalate_tension(
    state: TensionState,
    tension_id: str,
    attempt_id: str
) -> TensionState:
    """Escalate tension (resolution failed)"""
    if tension_id not in state.tensions:
        return state
    
    tension = state.tensions[tension_id]
    
    # Create escalated tension
    escalated = Tension(
        tension_id=tension.tension_id,
        type=tension.type,
        intensity=min(1.0, tension.intensity + 0.1),  # Increase intensity
        sources=tension.sources,
        first_detected=tension.first_detected,
        last_updated=datetime.utcnow().isoformat(),
        state=TensionLifecycle.ESCALATED.value,
        resolution_attempts=tension.resolution_attempts + (attempt_id,),
        resolution_progress=tension.resolution_progress,
        priority_score=tension.priority_score * 1.2  # Increase priority
    )
    
    new_tensions = {**state.tensions, tension_id: escalated}
    new_salience = compute_salience(new_tensions, state.salience.attention_budget)
    
    return TensionState(
        tensions=MappingProxyType(new_tensions),
        salience=new_salience,
        generated_goals=state.generated_goals,
        resolution_history=state.resolution_history,
        total_tension_energy=sum(t.intensity for t in new_tensions.values()),
        dominant_tension_type=state.dominant_tension_type,
        version=state.version + 1
    )


def resolve_tension(
    state: TensionState,
    tension_id: str,
    resolution_type: str
) -> TensionState:
    """Mark tension as resolved"""
    if tension_id not in state.tensions:
        return state
    
    tension = state.tensions[tension_id]
    
    resolved = Tension(
        tension_id=tension.tension_id,
        type=tension.type,
        intensity=0.0,  # Reduced to zero
        sources=tension.sources,
        first_detected=tension.first_detected,
        last_updated=datetime.utcnow().isoformat(),
        state=TensionState.RESOLVED.value,
        resolution_attempts=tension.resolution_attempts,
        resolution_progress=1.0,  # Fully resolved
        priority_score=0.0  # No longer salient
    )
    
    new_tensions = {**state.tensions, tension_id: resolved}
    new_salience = compute_salience(new_tensions, state.salience.attention_budget)
    
    # Update resolution history
    new_history = {**state.resolution_history}
    if tension_id not in new_history:
        new_history[tension_id] = []
    new_history[tension_id] = list(new_history[tension_id]) + [resolution_type]
    
    return TensionState(
        tensions=MappingProxyType(new_tensions),
        salience=new_salience,
        generated_goals=state.generated_goals,
        resolution_history=MappingProxyType(new_history),
        total_tension_energy=sum(t.intensity for t in new_tensions.values()),
        dominant_tension_type=state.dominant_tension_type,
        version=state.version + 1
    )


def create_initial_tension_state() -> TensionState:
    """Create initial empty tension state"""
    salience = SalienceMap(
        tensions=MappingProxyType({}),
        total_attention_demand=0.0,
        top_tension_ids=(),
        attention_budget=1.0,
        allocated_attention=MappingProxyType({}),
        version=0
    )
    
    return TensionState(
        tensions=MappingProxyType({}),
        salience=salience,
        generated_goals=MappingProxyType({}),
        resolution_history=MappingProxyType({}),
        total_tension_energy=0.0,
        dominant_tension_type="",
        version=0
    )