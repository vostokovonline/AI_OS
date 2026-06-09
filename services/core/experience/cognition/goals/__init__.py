"""
Goal Economy - Attention allocator with goal competition.

Stage: Cognitive Architecture

Goals compete for:
- Execution time
- Cognitive resources
- Attention
- Energy

This creates a "market" where:
- Higher value goals attract more resources
- Conflicting goals are priced out
- Limited attention forces prioritization
- Dynamic reallocation based on changing conditions

Key concepts:
- Goal priority = market price
- Resource constraints = budget
- Competition = natural selection of intentions
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json


class GoalStatus(Enum):
    """Goal lifecycle states"""
    CREATED = "created"
    ACTIVE = "active"
    BLOCKED = "blocked"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"


class GoalType(Enum):
    """Types of goals"""
    ACHIEVEMENT = "achievement"  # Must reach target state
    AVOIDANCE = "avoidance"  # Must avoid target state
    MAINTENANCE = "maintenance"  # Must maintain current state
    EXPLORATION = "exploration"  # Discover new possibilities
    REPAIR = "repair"  # Fix broken state
    LEARNING = "learning"  # Acquire new capability


class GoalPriority(Enum):
    """Priority levels"""
    CRITICAL = 1.0  # Must do now
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    BACKGROUND = 0.1


@dataclass(frozen=True)
class Goal:
    """
    A goal that competes for execution resources.
    
    Properties:
    - title: what we want
    - target_state: what success looks like
    - priority: initial priority (can change)
    - resource_cost: how much resources needed
    - deadline: when it must be done
    - dependencies: what must complete first
    - blockers: what's preventing execution
    """
    goal_id: str
    title: str
    description: str
    target_state: str  # JSON describing target
    goal_type: str  # GoalType value
    priority: float  # 0-1, current priority
    base_priority: float  # Original priority
    resource_cost: float  # Estimated resources needed
    deadline: Optional[str]  # ISO timestamp
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    status: str  # GoalStatus value
    dependencies: Tuple[str, ...]  # goal_ids that must complete first
    parent_goal_id: Optional[str]  # Parent goal if decomposed
    progress: float  # 0-1, execution progress
    value_contribution: float  # How much this goal adds to overall utility
    risk_level: float  # 0-1, execution risk
    executor_id: Optional[str]  # Who/what is executing
    
    def is_ready(self) -> bool:
        """Goal is ready to execute (all dependencies met)"""
        return self.status in (GoalStatus.CREATED.value, GoalStatus.ACTIVE.value)
    
    def is_blocked(self) -> bool:
        return self.status == GoalStatus.BLOCKED.value
    
    def is_complete(self) -> bool:
        return self.status in (GoalStatus.COMPLETED.value, GoalStatus.FAILED.value, 
                               GoalStatus.CANCELLED.value)
    
    def time_until_deadline(self) -> Optional[float]:
        """Hours until deadline (negative if overdue)"""
        if not self.deadline:
            return None
        deadline = datetime.fromisoformat(self.deadline)
        now = datetime.utcnow()
        return (deadline - now).total_seconds() / 3600
    
    def effective_priority(self) -> float:
        """Compute effective priority with deadline pressure"""
        priority = self.priority
        
        # Deadline pressure
        hours = self.time_until_deadline()
        if hours is not None and hours < 24:
            # Urgency multiplier
            urgency = 1.0 + (24 - hours) / 24
            priority = min(1.0, priority * urgency)
        
        return priority


@dataclass(frozen=True)
class ResourceAllocation:
    """
    Resource allocation for a goal.
    
    Tracks how resources are allocated and used.
    """
    goal_id: str
    allocated_time_ms: int
    allocated_cpu: float  # 0-1
    allocated_memory_mb: int
    start_time: str
    deadline: Optional[str]
    efficiency: float  # How well resources were used (0-1)
    yield_value: float  # Value produced per resource unit


@dataclass(frozen=True)
class GoalMarket:
    """
    The market where goals compete for resources.
    
    This is where priority decisions happen:
    - Which goals get resources
    - How resources are allocated
    - When reallocation occurs
    """
    goals: MappingProxyType  # type: ignore
    allocations: MappingProxyType  # type: ignore
    total_resources: float  # Available resources
    used_resources: float  # Currently allocated
    available_resources: float  # Free to allocate
    version: int
    
    def __post_init__(self):
        for attr in ('goals', 'allocations'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
    
    @staticmethod
    def compute_hash(state: 'GoalMarket') -> str:
        data = {
            "version": state.version,
            "goal_count": len(state.goals),
            "total_resources": state.total_resources,
            "used_resources": state.used_resources,
            "active_goals": sum(1 for g in state.goals.values() 
                              if g.status in ('active', 'executing'))
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def get_active_goals(self) -> List[Goal]:
        """Get all goals ready to execute"""
        return [g for g in self.goals.values() 
                if g.status in (GoalStatus.ACTIVE.value, GoalStatus.EXECUTING.value)]
    
    def get_ready_goals(self) -> List[Goal]:
        """Get goals ready to execute (dependencies met)"""
        return [g for g in self.get_active_goals() 
                if not g.is_blocked()]
    
    def get_overdue_goals(self) -> List[Goal]:
        """Get goals past deadline"""
        now = datetime.utcnow()
        return [g for g in self.goals.values() 
                if g.deadline and datetime.fromisoformat(g.deadline) < now
                and not g.is_complete()]
    
    def compute_market_price(self, goal: Goal) -> float:
        """
        Compute market price for a goal.
        
        Price = priority * scarcity * urgency / competition
        
        Higher price = more resources allocated
        """
        base = goal.effective_priority()
        
        # Scarcity (how rare is this goal type?)
        scarcity = 0.5  # Neutral
        
        # Urgency (time pressure)
        hours = goal.time_until_deadline()
        if hours is not None:
            if hours < 0:
                urgency = 2.0  # Overdue = very urgent
            elif hours < 1:
                urgency = 1.5
            elif hours < 24:
                urgency = 1.2
            else:
                urgency = 1.0
        else:
            urgency = 1.0
        
        # Competition (how many other goals want resources?)
        competing = len([g for g in self.get_ready_goals() if g.goal_id != goal.goal_id])
        competition_factor = 1.0 / (1 + competing * 0.1)
        
        price = base * scarcity * urgency * competition_factor
        
        return min(1.0, max(0.01, price))
    
    def allocate_resources(
        self, 
        goal_id: str, 
        time_ms: int, 
        cpu: float, 
        memory_mb: int
    ) -> 'GoalMarket':
        """Allocate resources to a goal"""
        if goal_id not in self.goals:
            return self
        
        new_allocations = {**self.allocations, goal_id: ResourceAllocation(
            goal_id=goal_id,
            allocated_time_ms=time_ms,
            allocated_cpu=cpu,
            allocated_memory_mb=memory_mb,
            start_time=datetime.utcnow().isoformat(),
            deadline=self.goals[goal_id].deadline,
            efficiency=0.0,
            yield_value=0.0
        )}
        
        new_used = self.used_resources + (time_ms / 3600000)  # Convert to hours
        
        return GoalMarket(
            goals=self.goals,
            allocations=MappingProxyType(new_allocations),
            total_resources=self.total_resources,
            used_resources=new_used,
            available_resources=self.total_resources - new_used,
            version=self.version + 1
        )
    
    def with_goal(self, goal: Goal) -> 'GoalMarket':
        """Add or update goal in market"""
        new_goals = {**self.goals, goal.goal_id: goal}
        
        # Recalculate available resources
        active_count = sum(1 for g in new_goals.values() 
                         if g.status in ('active', 'executing'))
        expected_used = active_count * 0.25  # Assume 25% per active goal
        
        return GoalMarket(
            goals=MappingProxyType(new_goals),
            allocations=self.allocations,
            total_resources=self.total_resources,
            used_resources=min(self.total_resources, expected_used),
            available_resources=max(0, self.total_resources - expected_used),
            version=self.version + 1
        )
    
    def remove_goal(self, goal_id: str) -> 'GoalMarket':
        """Remove goal from market"""
        new_goals = {k: v for k, v in self.goals.items() if k != goal_id}
        new_allocations = {k: v for k, v in self.allocations.items() if k != goal_id}
        
        return GoalMarket(
            goals=MappingProxyType(new_goals),
            allocations=MappingProxyType(new_allocations),
            total_resources=self.total_resources,
            used_resources=self.used_resources,
            available_resources=self.available_resources,
            version=self.version + 1
        )
    
    def reallocate(self) -> Tuple[List[Tuple[str, float]], 'GoalMarket']:
        """
        Reallocate resources based on current priorities.
        
        Returns list of (goal_id, new_priority) tuples and updated market.
        """
        ready = self.get_ready_goals()
        
        if not ready:
            return [], self
        
        # Compute prices for all ready goals
        prices = [(g.goal_id, self.compute_market_price(g)) for g in ready]
        
        # Sort by price (highest first)
        prices.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate top N goals based on available resources
        total_price = sum(p for _, p in prices)
        available = self.available_resources
        
        allocations = []
        for goal_id, price in prices:
            if available >= price * 0.1:  # Minimum allocation threshold
                allocation = min(available, price)
                allocations.append((goal_id, allocation))
                available -= allocation
        
        return allocations, self


def create_goal_market(total_resources: float = 1.0) -> GoalMarket:
    """Create initial goal market"""
    return GoalMarket(
        goals=MappingProxyType({}),
        allocations=MappingProxyType({}),
        total_resources=total_resources,
        used_resources=0.0,
        available_resources=total_resources,
        version=0
    )


def decompose_goals(market: GoalMarket, max_depth: int = 3) -> GoalMarket:
    """
    Decompose high-priority goals into subgoals.
    
    This is how complex goals get broken down.
    """
    new_market = market
    
    for goal in market.get_ready_goals():
        if goal.progress < 0.3 and goal.priority > 0.7:
            # High priority goal with low progress = candidate for decomposition
            # This would call the decomposition service
            pass
    
    return new_market


def resolve_conflicts(
    market: GoalMarket,
    contradiction_state
) -> GoalMarket:
    """
    Resolve goal conflicts based on contradictions.
    
    When contradiction engine detects goal-goal conflict,
    adjust priorities accordingly.
    """
    from .contradiction import ContradictionType
    
    for goal in market.get_active_goals():
        # Check for contradictions with other goals
        for other_goal in market.get_active_goals():
            if goal.goal_id == other_goal.goal_id:
                continue
            
            # Resource overlap detection
            if (goal.resource_cost + other_goal.resource_cost > 
                market.available_resources + 0.2):
                # Both can't execute simultaneously
                # Winner = higher effective priority
                if goal.effective_priority() < other_goal.effective_priority():
                    # Demote lower priority goal
                    new_goal = Goal(
                        goal_id=goal.goal_id,
                        title=goal.title,
                        description=goal.description,
                        target_state=goal.target_state,
                        goal_type=goal.goal_type,
                        priority=goal.priority * 0.5,  # Reduce priority
                        base_priority=goal.base_priority,
                        resource_cost=goal.resource_cost,
                        deadline=goal.deadline,
                        created_at=goal.created_at,
                        started_at=goal.started_at,
                        completed_at=goal.completed_at,
                        status=GoalStatus.SUSPENDED.value,
                        dependencies=goal.dependencies,
                        parent_goal_id=goal.parent_goal_id,
                        progress=goal.progress,
                        value_contribution=goal.value_contribution,
                        risk_level=goal.risk_level,
                        executor_id=goal.executor_id
                    )
                    new_market = market.with_goal(new_goal)
                    return new_market
    
    return market