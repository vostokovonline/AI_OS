"""
AI-OS Executive System Architecture
====================================

REORIENTATION: From research → practical system

Goal: Persistent Strategic Cognitive Operating System
  - Understanding user state
  - Holding long-term goals
  - Decomposing problems
  - Making decisions
  - Monitoring execution
  - Adapting to context

NOT: synthetic organism research
BUT: practical intelligent assistant

The 4-Level Architecture:

LEVEL 1 — FOUNDATION RUNTIME (Infrastructure)
  - Event bus, memory layers, persistence
  - Replay system, scheduler, async runtime
  - Tool execution, websocket ingestion
  - State snapshots, telemetry, observability

LEVEL 2 — COGNITIVE CONTROL LAYER (Core Intelligence)
  - State modeling (energy, overload, context, risks)
  - Goal hierarchy (strategic, tactical, operational)
  - Planning engine (HTN + utility + reflection)
  - Reflection system (mistake detection, drift, burnout)

LEVEL 3 — EXECUTION SYSTEM (The Missing Piece)
  - Task graph runtime (DAG, dependencies, blockers)
  - Active monitoring (stuck, distracted, overloaded)
  - Adaptive intervention (simplify, recover, adapt)
  - Environment integration (GitHub, Calendar, Telegram, etc.)

LEVEL 4 — STRATEGIC META-COGNITION (Support Only)
  - Latent manifolds, world models
  - Attractor cognition, trajectory prediction
  - Long-horizon simulation
  - BUT: as SUPPORT SYSTEM, not core
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json


# ============================================================================
# LEVEL 1: FOUNDATION RUNTIME
# ============================================================================

class EventType(Enum):
    """Core event types for AI-OS."""
    USER_STATE_CHANGE = "user_state_change"
    GOAL_CREATED = "goal_created"
    GOAL_UPDATED = "goal_updated"
    GOAL_COMPLETED = "goal_completed"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_BLOCKED = "task_blocked"
    TASK_COMPLETED = "task_completed"
    DECISION_MADE = "decision_made"
    REFLECTION_TRIGGERED = "reflection_triggered"
    INTERVENTION = "intervention"
    ENVIRONMENT_CHANGE = "environment_change"
    CONTEXT_SWITCH = "context_switch"
    ENERGY_CHANGE = "energy_change"
    MILESTONE_REACHED = "milestone_reached"


@dataclass
class AIAOSEvent:
    """Foundation event structure."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str  # user, system, agent, environment
    causal_chain: List[str] = field(default_factory=list)  # Event IDs
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source,
            'causal_chain': self.causal_chain
        }


class EventBus:
    """
    Foundation event bus for AI-OS.
    
    All events flow through here.
    Enables replay, debugging, observability.
    """
    
    def __init__(self):
        self.events: List[AIAOSEvent] = []
        self.subscribers: Dict[EventType, List[callable]] = {}
        self.event_counter = 0
        
    def emit(self, event_type: EventType, data: Dict, 
             source: str = "system", causal_chain: List[str] = None) -> AIAOSEvent:
        """Emit a new event."""
        self.event_counter += 1
        event = AIAOSEvent(
            event_id=f"evt_{self.event_counter}",
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            source=source,
            causal_chain=causal_chain or []
        )
        
        self.events.append(event)
        
        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(event)
                except Exception:
                    pass
        
        return event
    
    def subscribe(self, event_type: EventType, callback: callable):
        """Subscribe to event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def replay(self, from_time: datetime = None, event_types: List[EventType] = None) -> List[AIAOSEvent]:
        """Replay events from history."""
        filtered = self.events
        
        if from_time:
            filtered = [e for e in filtered if e.timestamp >= from_time]
        
        if event_types:
            filtered = [e for e in filtered if e.event_type in event_types]
        
        return filtered
    
    def get_summary(self) -> Dict:
        """Get event bus summary."""
        type_counts = {}
        for event in self.events:
            type_key = event.event_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
        
        return {
            'total_events': len(self.events),
            'type_counts': type_counts,
            'first_event': self.events[0].timestamp.isoformat() if self.events else None,
            'last_event': self.events[-1].timestamp.isoformat() if self.events else None
        }


@dataclass 
class MemoryEntry:
    """Persistent memory entry."""
    memory_id: str
    content: str
    importance: float  # 0-1, auto-computed
    access_count: int
    last_access: datetime
    created_at: datetime
    tags: List[str]
    context: Dict[str, Any]  # When was this relevant
    
    def relevance_score(self, current_context: Dict) -> float:
        """Compute relevance to current context."""
        base_score = self.importance
        
        # Recency bonus
        hours_since_access = (datetime.now() - self.last_access).total_seconds() / 3600
        recency_bonus = 1.0 / (1.0 + hours_since_access * 0.1)
        
        # Tag match bonus
        tag_match = 0.0
        if 'tags' in current_context:
            matching_tags = set(self.tags) & set(current_context['tags'])
            tag_match = len(matching_tags) / max(1, len(self.tags))
        
        return base_score * 0.5 + recency_bonus * 0.3 + tag_match * 0.2


class PersistentMemory:
    """
    Level 1: Persistent memory system.
    
    Stores all important information.
    Enables context retention across sessions.
    """
    
    def __init__(self):
        self.memories: Dict[str, MemoryEntry] = {}
        self.episodic: List[Dict] = []  # Short-term episodes
        self.semantic: Dict[str, Any] = {}  # Long-term knowledge
        self.working: Dict[str, Any] = {}  # Current context
        
        self.memory_counter = 0
        
    def store(self, content: str, importance: float = 0.5,
              tags: List[str] = None, context: Dict = None) -> str:
        """Store new memory."""
        self.memory_counter += 1
        memory_id = f"mem_{self.memory_counter}"
        
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            importance=importance,
            access_count=0,
            last_access=datetime.now(),
            created_at=datetime.now(),
            tags=tags or [],
            context=context or {}
        )
        
        self.memories[memory_id] = entry
        return memory_id
    
    def recall(self, current_context: Dict, limit: int = 10) -> List[MemoryEntry]:
        """Recall relevant memories."""
        scored = []
        
        for memory in self.memories.values():
            score = memory.relevance_score(current_context)
            scored.append((score, memory))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]
    
    def episodic_store(self, episode: Dict):
        """Store short-term episode."""
        self.episodic.append({
            'timestamp': datetime.now(),
            'episode': episode
        })
        
        # Limit episodic memory
        if len(self.episodic) > 1000:
            self.episodic = self.episodic[-500:]
    
    def update_working(self, key: str, value: Any):
        """Update working memory."""
        self.working[key] = value
    
    def get_working_context(self) -> Dict:
        """Get current working context."""
        return self.working.copy()
    
    def get_summary(self) -> Dict:
        """Get memory summary."""
        return {
            'n_memories': len(self.memories),
            'n_episodes': len(self.episodic),
            'n_working': len(self.working),
            'avg_importance': np.mean([m.importance for m in self.memories.values()]) if self.memories else 0
        }


# ============================================================================
# LEVEL 2: COGNITIVE CONTROL LAYER
# ============================================================================

class EnergyLevel(Enum):
    """User energy states."""
    DEPLETED = "depleted"      # < 20%
    LOW = "low"               # 20-40%
    MODERATE = "moderate"      # 40-60%
    HIGH = "high"             # 60-80%
    PEAK = "peak"             # > 80%


class CognitiveRegime(Enum):
    """Cognitive operating modes."""
    DEEP_FOCUS = "deep_focus"      # Complex problem solving
    BROAD_EXPLORATION = "broad_exploration"  # Ideation, learning
    ROUTINE_EXECUTION = "routine_execution"  # Habits, automation
    RECOVERY = "recovery"          # Rest, reflection
    CRISIS = "crisis"              # High urgency


@dataclass
class UserState:
    """Model of user state."""
    energy: float  # 0-1
    cognitive_load: float  # 0-1
    stress: float  # 0-1
    focus_quality: float  # 0-1
    motivation: float  # 0-1
    
    energy_history: List[float] = field(default_factory=list)
    regime: CognitiveRegime = field(default=CognitiveRegime.ROUTINE_EXECUTION)
    
    def detect_regime(self) -> CognitiveRegime:
        """Detect current cognitive regime."""
        if self.stress > 0.8:
            return CognitiveRegime.CRISIS
        if self.energy < 0.3:
            return CognitiveRegime.RECOVERY
        if self.cognitive_load > 0.7:
            return CognitiveRegime.BROAD_EXPLORATION
        if self.focus_quality > 0.7 and self.energy > 0.5:
            return CognitiveRegime.DEEP_FOCUS
        return CognitiveRegime.ROUTINE_EXECUTION
    
    def recommend_task_complexity(self) -> str:
        """Recommend task complexity based on state."""
        if self.energy > 0.7 and self.focus_quality > 0.6:
            return "high"  # Complex, creative
        if self.energy > 0.4:
            return "medium"  # Standard work
        return "low"  # Simple, routine
    
    def predict_burnout_risk(self) -> float:
        """Predict burnout risk."""
        stress_trend = 0.0
        if len(self.energy_history) > 5:
            recent = self.energy_history[-5:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                stress_trend = 0.3  # Declining energy
        
        return min(1.0, self.stress * 0.5 + stress_trend + (1 - self.motivation) * 0.3)


class UserStateModel:
    """
    Level 2: User state modeling.
    
    Tracks energy, overload, context, risks.
    Enables adaptive task recommendation.
    """
    
    def __init__(self):
        self.current_state = UserState(
            energy=0.5, cognitive_load=0.3, stress=0.2,
            focus_quality=0.5, motivation=0.7
        )
        
        self.state_history: List[UserState] = []
        self.regime_transitions: List[Tuple[datetime, CognitiveRegime]] = []
        
    def update(self, energy: float = None, cognitive_load: float = None,
               stress: float = None, focus_quality: float = None,
               motivation: float = None):
        """Update user state."""
        if energy is not None:
            self.current_state.energy = max(0, min(1, energy))
        if cognitive_load is not None:
            self.current_state.cognitive_load = max(0, min(1, cognitive_load))
        if stress is not None:
            self.current_state.stress = max(0, min(1, stress))
        if focus_quality is not None:
            self.current_state.focus_quality = max(0, min(1, focus_quality))
        if motivation is not None:
            self.current_state.motivation = max(0, min(1, motivation))
        
        # Detect regime
        new_regime = self.current_state.detect_regime()
        if new_regime != self.current_state.regime:
            self.regime_transitions.append((datetime.now(), new_regime))
            self.current_state.regime = new_regime
        
        # Track energy history
        self.current_state.energy_history.append(self.current_state.energy)
        if len(self.current_state.energy_history) > 100:
            self.current_state.energy_history = self.current_state.energy_history[-50:]
        
        # Store history
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-50:]
        self.state_history.append(self.current_state)
    
    def get_recommendation(self) -> Dict:
        """Get task recommendations based on state."""
        complexity = self.current_state.recommend_task_complexity()
        burnout_risk = self.current_state.predict_burnout_risk()
        
        recommendations = []
        
        if burnout_risk > 0.6:
            recommendations.append({
                'type': 'reduce_load',
                'priority': 'high',
                'message': 'High burnout risk detected. Recommend reducing task load.'
            })
        
        if self.current_state.energy < 0.4:
            recommendations.append({
                'type': 'break',
                'priority': 'medium',
                'message': 'Energy is low. Consider taking a break or doing light tasks.'
            })
        
        if self.current_state.stress > 0.6:
            recommendations.append({
                'type': 'deprioritize',
                'priority': 'high',
                'message': 'High stress detected. Recommend skipping non-critical tasks.'
            })
        
        return {
            'complexity': complexity,
            'regime': self.current_state.regime.value,
            'burnout_risk': burnout_risk,
            'energy_level': self.current_state.energy,
            'recommendations': recommendations
        }


class GoalState(Enum):
    """Goal states."""
    ACTIVE = "active"
    BLOCKED = "blocked"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ON_HOLD = "on_hold"


@dataclass
class Goal:
    """Goal representation with hierarchy."""
    goal_id: str
    title: str
    description: str
    
    level: str  # strategic, tactical, operational
    
    state: GoalState = GoalState.ACTIVE
    progress: float = 0.0  # 0-1
    
    priority: int = 5  # 1-10
    urgency: float = 0.5  # 0-1
    difficulty: float = 0.5  # 0-1
    
    parent_id: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    
    deadline: Optional[datetime] = None
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    blockers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    context: Dict[str, Any] = field(default_factory=dict)
    
    def compute_viability(self) -> float:
        """Compute goal viability."""
        if self.deadline:
            time_remaining = (self.deadline - datetime.now()).total_seconds() / 3600
            time_factor = time_remaining / max(1, self.estimated_hours)
        else:
            time_factor = 1.0
        
        priority_factor = self.priority / 10.0
        
        return min(1.0, (time_factor * 0.5 + priority_factor * 0.5))


class GoalHierarchy:
    """
    Level 2: Goal hierarchy management.
    
    Maintains strategic, tactical, operational goals.
    Tracks dependencies, blockers, progress.
    """
    
    def __init__(self):
        self.strategic_goals: Dict[str, Goal] = {}
        self.tactical_goals: Dict[str, Goal] = {}
        self.operational_goals: Dict[str, Goal] = {}
        
        self.goal_counter = 0
        
    def create_goal(self, title: str, description: str, level: str,
                   priority: int = 5, parent_id: str = None) -> Goal:
        """Create new goal."""
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"
        
        goal = Goal(
            goal_id=goal_id,
            title=title,
            description=description,
            level=level,
            priority=priority,
            parent_id=parent_id
        )
        
        if level == "strategic":
            self.strategic_goals[goal_id] = goal
        elif level == "tactical":
            self.tactical_goals[goal_id] = goal
        else:
            self.operational_goals[goal_id] = goal
        
        # Link to parent
        if parent_id:
            parent = self.get_goal(parent_id)
            if parent:
                parent.subgoals.append(goal_id)
        
        return goal
    
    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get goal by ID."""
        if goal_id in self.strategic_goals:
            return self.strategic_goals[goal_id]
        if goal_id in self.tactical_goals:
            return self.tactical_goals[goal_id]
        if goal_id in self.operational_goals:
            return self.operational_goals[goal_id]
        return None
    
    def get_prioritized_goals(self, limit: int = 10) -> List[Goal]:
        """Get goals sorted by priority and viability."""
        all_goals = (list(self.strategic_goals.values()) + 
                    list(self.tactical_goals.values()) + 
                    list(self.operational_goals.values()))
        
        # Filter active
        active = [g for g in all_goals if g.state in [GoalState.ACTIVE, GoalState.IN_PROGRESS]]
        
        # Sort by computed score
        scored = [(g.compute_viability() * g.priority / 10, g) for g in active]
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [g for _, g in scored[:limit]]
    
    def update_progress(self, goal_id: str, progress_delta: float):
        """Update goal progress."""
        goal = self.get_goal(goal_id)
        if goal:
            goal.progress = min(1.0, goal.progress + progress_delta)
            goal.updated_at = datetime.now()
            
            if goal.progress >= 1.0:
                goal.state = GoalState.COMPLETED
                goal.completed_at = datetime.now()
    
    def detect_blockers(self) -> List[Dict]:
        """Detect goal blockers."""
        blockers = []
        
        for goal in self.get_prioritized_goals(limit=20):
            if goal.deadline:
                time_remaining = (goal.deadline - datetime.now()).total_seconds() / 3600
                if time_remaining < 0:
                    blockers.append({
                        'goal_id': goal.goal_id,
                        'type': 'deadline_passed',
                        'urgency': 1.0
                    })
            
            if goal.dependencies:
                for dep_id in goal.dependencies:
                    dep = self.get_goal(dep_id)
                    if dep and dep.state != GoalState.COMPLETED:
                        blockers.append({
                            'goal_id': goal.goal_id,
                            'blocking_goal_id': dep_id,
                            'type': 'dependency_unmet',
                            'urgency': 0.7
                        })
        
        return blockers
    
    def get_summary(self) -> Dict:
        """Get goal hierarchy summary."""
        all_goals = len(self.strategic_goals) + len(self.tactical_goals) + len(self.operational_goals)
        active = sum(1 for g in self.get_prioritized_goals(limit=100) 
                    if g.state in [GoalState.ACTIVE, GoalState.IN_PROGRESS])
        
        return {
            'total_goals': all_goals,
            'strategic': len(self.strategic_goals),
            'tactical': len(self.tactical_goals),
            'operational': len(self.operational_goals),
            'active': active,
            'blockers_detected': len(self.detect_blockers())
        }


# ============================================================================
# LEVEL 3: EXECUTION SYSTEM
# ============================================================================

class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Executable task with dependencies."""
    task_id: str
    goal_id: Optional[str]
    title: str
    description: str
    
    state: TaskState = TaskState.PENDING
    
    estimated_minutes: float = 30.0
    energy_required: float = 0.5  # 0-1
    
    dependencies: List[str] = field(default_factory=list)  # Task IDs that must complete first
    dependents: List[str] = field(default_factory=list)  # Task IDs that depend on this
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    attempts: int = 0
    max_attempts: int = 3
    
    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """Check if task can be executed."""
        if self.state != TaskState.PENDING and self.state != TaskState.BLOCKED:
            return False
        
        for dep_id in self.dependencies:
            if dep_id not in completed_tasks:
                return False
        
        return True
    
    def compute_priority(self, urgency: float = 0.5) -> float:
        """Compute execution priority."""
        base_priority = 1.0
        
        if self.state == TaskState.BLOCKED:
            base_priority *= 0.5
        
        if self.attempts > 0:
            base_priority *= (1.0 + self.attempts * 0.2)
        
        return base_priority * urgency


class TaskGraph:
    """
    Level 3: Task graph execution system.
    
    DAG of tasks with dependencies.
    Critical path detection.
    Blocker management.
    """
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.task_counter = 0
        
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
        
    def create_task(self, title: str, goal_id: str = None,
                   estimated_minutes: float = 30.0,
                   dependencies: List[str] = None) -> Task:
        """Create new task."""
        self.task_counter += 1
        task_id = f"task_{self.task_counter}"
        
        task = Task(
            task_id=task_id,
            goal_id=goal_id,
            title=title,
            description=title,
            estimated_minutes=estimated_minutes,
            dependencies=dependencies or []
        )
        
        self.tasks[task_id] = task
        
        # Update dependency graph
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                self.tasks[dep_id].dependents.append(task_id)
        
        return task
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that can be executed."""
        ready = []
        
        for task in self.tasks.values():
            if task.can_execute(self.completed_tasks):
                priority = task.compute_priority()
                ready.append((priority, task))
        
        ready.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in ready]
    
    def start_task(self, task_id: str) -> bool:
        """Start task execution."""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if not task.can_execute(self.completed_tasks):
            return False
        
        task.state = TaskState.RUNNING
        task.started_at = datetime.now()
        task.attempts += 1
        
        return True
    
    def complete_task(self, task_id: str, result: Dict = None):
        """Mark task as completed."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.state = TaskState.COMPLETED
        task.completed_at = datetime.now()
        task.result = result
        
        self.completed_tasks.add(task_id)
        
        # Check dependents
        for dep_id in task.dependents:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.can_execute(self.completed_tasks):
                    dep_task.state = TaskState.READY
    
    def fail_task(self, task_id: str, error: str):
        """Mark task as failed."""
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.state = TaskState.FAILED
        task.error = error
        
        self.failed_tasks.add(task_id)
        
        # Block dependents
        for dep_id in task.dependents:
            if dep_id in self.tasks:
                self.tasks[dep_id].state = TaskState.BLOCKED
    
    def get_critical_path(self) -> List[str]:
        """Detect critical path through task graph."""
        # Simple critical path: longest dependency chain
        def get_depth(task_id, visited=None):
            if visited is None:
                visited = set()
            
            if task_id in visited:
                return 0
            visited.add(task_id)
            
            task = self.tasks.get(task_id)
            if not task:
                return 0
            
            if not task.dependencies:
                return 1
            
            max_dep_depth = 0
            for dep_id in task.dependencies:
                dep_depth = get_depth(dep_id, visited.copy())
                max_dep_depth = max(max_dep_depth, dep_depth)
            
            return 1 + max_dep_depth
        
        depths = [(get_depth(tid), tid) for tid in self.tasks]
        depths.sort(key=lambda x: x[0], reverse=True)
        
        return [tid for _, tid in depths[:5]]
    
    def get_progress(self) -> Dict:
        """Get execution progress."""
        total = len(self.tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        running = sum(1 for t in self.tasks.values() if t.state == TaskState.RUNNING)
        blocked = sum(1 for t in self.tasks.values() if t.state == TaskState.BLOCKED)
        
        estimated_remaining = sum(t.estimated_minutes for tid, t in self.tasks.items()
                                if tid not in self.completed_tasks and tid not in self.failed_tasks)
        
        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'running': running,
            'blocked': blocked,
            'progress_pct': (completed / total * 100) if total > 0 else 0,
            'estimated_minutes_remaining': estimated_remaining
        }


class ActiveMonitor:
    """
    Level 3: Active monitoring system.
    
    Detects when user is stuck, distracted, overloaded.
    Triggers interventions.
    """
    
    def __init__(self, task_graph: TaskGraph, user_state: UserStateModel):
        self.task_graph = task_graph
        self.user_state = user_state
        
        self.stuck_tasks: List[str] = []  # Tasks running too long
        self.ignored_tasks: List[str] = []  # Tasks not started but should be
        
        self.last_intervention = datetime.now()
        
    def check_progress(self) -> List[Dict]:
        """Check for stuck or problematic tasks."""
        issues = []
        
        now = datetime.now()
        
        for task_id, task in self.task_graph.tasks.items():
            if task.state == TaskState.RUNNING:
                # Check if stuck
                if task.started_at:
                    running_minutes = (now - task.started_at).total_seconds() / 60
                    expected_max = task.estimated_minutes * 2
                    
                    if running_minutes > expected_max:
                        issues.append({
                            'task_id': task_id,
                            'type': 'stuck',
                            'running_minutes': running_minutes,
                            'expected_max': expected_max,
                            'urgency': min(1.0, running_minutes / expected_max)
                        })
            
            elif task.state == TaskState.PENDING:
                # Check if overdue
                if task_id not in self.task_graph.completed_tasks:
                    self.ignored_tasks.append(task_id)
        
        return issues
    
    def check_user_state(self) -> List[Dict]:
        """Check user state for problems."""
        issues = []
        
        state = self.user_state.current_state
        
        if state.energy < 0.3:
            issues.append({
                'type': 'low_energy',
                'urgency': 0.8,
                'message': 'Energy is critically low'
            })
        
        if state.stress > 0.7:
            issues.append({
                'type': 'high_stress',
                'urgency': 0.9,
                'message': 'Stress is dangerously high'
            })
        
        burnout_risk = state.predict_burnout_risk()
        if burnout_risk > 0.5:
            issues.append({
                'type': 'burnout_risk',
                'urgency': 0.7,
                'burnout_risk': burnout_risk
            })
        
        return issues
    
    def get_interventions(self) -> List[Dict]:
        """Get recommended interventions."""
        all_issues = self.check_progress() + self.check_user_state()
        
        # Don't spam interventions
        if (datetime.now() - self.last_intervention).total_seconds() < 1800:  # 30 min
            return []
        
        interventions = []
        
        for issue in all_issues:
            if issue['urgency'] > 0.7:
                if issue['type'] == 'stuck':
                    interventions.append({
                        'action': 'simplify_task',
                        'task_id': issue['task_id'],
                        'message': f'Task "{self.task_graph.tasks[issue["task_id"]].title}" seems stuck. Should we break it down?'
                    })
                
                elif issue['type'] == 'low_energy':
                    interventions.append({
                        'action': 'reduce_load',
                        'message': 'Your energy is low. Consider taking a break or doing lighter tasks.'
                    })
                
                elif issue['type'] == 'high_stress':
                    interventions.append({
                        'action': 'deprioritize',
                        'message': 'You seem stressed. Which tasks can we defer?'
                    })
        
        if interventions:
            self.last_intervention = datetime.now()
        
        return interventions


# ============================================================================
# LEVEL 4: STRATEGIC META-COGNITION (SUPPORT)
# ============================================================================

class StrategicPlanner:
    """
    Level 4: Strategic planning (as support system).
    
    Uses cognitive capabilities for prediction and planning.
    NOT the core - support only.
    """
    
    def __init__(self, goal_hierarchy: GoalHierarchy, task_graph: TaskGraph):
        self.goals = goal_hierarchy
        self.tasks = task_graph
        
        # Cognitive support systems (from Phases 16-26)
        self.trajectory_memory: List[Dict] = []
        self.pattern_recognizer: Dict[str, Any] = {}
        
    def decompose_goal(self, goal_id: str) -> List[Task]:
        """Decompose goal into executable tasks."""
        goal = self.goals.get_goal(goal_id)
        if not goal:
            return []
        
        # Simple decomposition based on goal description
        task_titles = goal.description.split('.')
        tasks = []
        
        dependencies = []
        for i, title in enumerate(task_titles[:5]):  # Max 5 tasks
            if title.strip():
                task = self.tasks.create_task(
                    title=title.strip(),
                    goal_id=goal_id,
                    estimated_minutes=goal.estimated_hours * 60 / max(1, len(task_titles)),
                    dependencies=dependencies
                )
                tasks.append(task)
                dependencies = [task.task_id]
        
        return tasks
    
    def plan_optimal_sequence(self, available_tasks: List[Task],
                             user_state: UserState) -> List[Task]:
        """Plan optimal task sequence given user state."""
        if not available_tasks:
            return []
        
        # Energy-aware sequencing
        energy = user_state.energy
        
        # Sort by energy requirement and priority
        scored = []
        for task in available_tasks:
            energy_match = 1.0 - abs(task.energy_required - energy)
            priority = task.compute_priority()
            
            score = priority * 0.6 + energy_match * 0.4
            scored.append((score, task))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored]
    
    def predict_success_probability(self, task_id: str) -> float:
        """Predict task success probability."""
        task = self.tasks.tasks.get(task_id)
        if not task:
            return 0.5
        
        # Simple prediction based on history
        if task.dependencies:
            all_deps_complete = all(dep_id in self.tasks.completed_tasks 
                                  for dep_id in task.dependencies)
            if not all_deps_complete:
                return 0.3
        
        if task.attempts > 2:
            return 0.4
        
        if task.estimated_minutes > 120:
            return 0.6
        
        return 0.8
    
    def simulate_execution(self, task_ids: List[str], n_simulations: int = 10) -> Dict:
        """Simulate task execution outcomes."""
        success_counts = {}
        total_hours = {}
        
        for task_id in task_ids:
            task = self.tasks.tasks.get(task_id)
            if task:
                success_counts[task_id] = 0
                total_hours[task_id] = 0.0
        
        for _ in range(n_simulations):
            completed = set()
            for task_id in task_ids:
                prob = self.predict_success_probability(task_id)
                if np.random.random() < prob:
                    success_counts[task_id] += 1
                    task = self.tasks.tasks.get(task_id)
                    if task:
                        total_hours[task_id] += task.estimated_minutes / 60
        
        return {
            'success_rates': {k: v / n_simulations for k, v in success_counts.items()},
            'estimated_hours': total_hours
        }


# ============================================================================
# INTEGRATED AI-OS EXECUTIVE SYSTEM
# ============================================================================

class AIOSExecutiveSystem:
    """
    Complete AI-OS Executive System.
    
    Levels 1-4 integrated into practical operating system.
    """
    
    def __init__(self):
        # Level 1: Foundation
        self.event_bus = EventBus()
        self.memory = PersistentMemory()
        
        # Level 2: Cognitive Control
        self.user_state = UserStateModel()
        self.goal_hierarchy = GoalHierarchy()
        
        # Level 3: Execution
        self.task_graph = TaskGraph()
        self.monitor = ActiveMonitor(self.task_graph, self.user_state)
        
        # Level 4: Strategic Support
        self.strategic_planner = StrategicPlanner(self.goal_hierarchy, self.task_graph)
        
        # Time
        self.t = datetime.now()
        
    def create_goal(self, title: str, description: str, level: str = "operational",
                   priority: int = 5) -> Goal:
        """Create new goal with event tracking."""
        goal = self.goal_hierarchy.create_goal(title, description, level, priority)
        
        self.event_bus.emit(
            EventType.GOAL_CREATED,
            {'goal_id': goal.goal_id, 'title': title, 'level': level},
            source="user"
        )
        
        # Store in memory
        self.memory.store(
            f"Created goal: {title}",
            importance=0.7,
            tags=['goal', level],
            context={'goal_id': goal.goal_id}
        )
        
        return goal
    
    def add_task(self, title: str, goal_id: str = None, 
                estimated_minutes: float = 30.0) -> Task:
        """Add task to execution graph."""
        task = self.task_graph.create_task(title, goal_id, estimated_minutes)
        
        self.event_bus.emit(
            EventType.TASK_STARTED,
            {'task_id': task.task_id, 'title': title},
            source="system"
        )
        
        return task
    
    def update_user_state(self, **kwargs):
        """Update user state model."""
        self.user_state.update(**kwargs)
        
        self.event_bus.emit(
            EventType.USER_STATE_CHANGE,
            self.user_state.current_state.__dict__,
            source="user"
        )
        
    def get_dashboard(self) -> Dict:
        """Get complete system dashboard."""
        return {
            'user_state': {
                'energy': self.user_state.current_state.energy,
                'regime': self.user_state.current_state.regime.value,
                'burnout_risk': self.user_state.current_state.predict_burnout_risk(),
                'recommendations': self.user_state.get_recommendation()
            },
            'goals': self.goal_hierarchy.get_summary(),
            'tasks': self.task_graph.get_progress(),
            'blockers': self.goal_hierarchy.detect_blockers(),
            'interventions': self.monitor.get_interventions(),
            'memory': self.memory.get_summary(),
            'events': self.event_bus.get_summary()
        }
    
    def run_cycle(self) -> Dict:
        """Run executive cycle."""
        # Check for issues
        interventions = self.monitor.get_interventions()
        
        # Get ready tasks
        ready_tasks = self.task_graph.get_ready_tasks()
        
        # User recommendations
        state_rec = self.user_state.get_recommendation()
        
        # Strategic planning
        if ready_tasks:
            optimal = self.strategic_planner.plan_optimal_sequence(
                ready_tasks, self.user_state.current_state
            )
        else:
            optimal = []
        
        return {
            'timestamp': datetime.now().isoformat(),
            'ready_tasks': len(ready_tasks),
            'next_task': optimal[0].title if optimal else None,
            'interventions_needed': len(interventions),
            'user_state': state_rec,
            'blockers': len(self.goal_hierarchy.detect_blockers())
        }


# ============================================================================
# TESTS
# ============================================================================

def test_foundation_runtime():
    """Test Level 1: Foundation Runtime."""
    print("\n" + "=" * 60)
    print("LEVEL 1: FOUNDATION RUNTIME TEST")
    print("=" * 60)
    
    event_bus = EventBus()
    memory = PersistentMemory()
    
    # Emit events
    for i in range(10):
        event_bus.emit(EventType.USER_STATE_CHANGE, {'energy': 0.5 + i * 0.05}, "user")
        event_bus.emit(EventType.GOAL_CREATED, {'title': f'Goal {i}'}, "system")
    
    # Store memories
    for i in range(5):
        memory.store(f'Memory {i}', importance=0.5 + i * 0.1, 
                   tags=['test'], context={'test': True})
    
    print(f"\n  Events: {event_bus.get_summary()}")
    print(f"  Memories: {memory.get_summary()}")


def test_cognitive_control():
    """Test Level 2: Cognitive Control."""
    print("\n" + "=" * 60)
    print("LEVEL 2: COGNITIVE CONTROL TEST")
    print("=" * 60)
    
    user_state = UserStateModel()
    goal_hierarchy = GoalHierarchy()
    
    # Update user state
    for i in range(10):
        user_state.update(
            energy=0.8 - i * 0.05,
            stress=0.2 + i * 0.03,
            focus_quality=0.7 - i * 0.02
        )
    
    # Create goals
    strategy = goal_hierarchy.create_goal(
        "Launch AI-OS", "Complete all phases", "strategic", priority=9
    )
    tactical = goal_hierarchy.create_goal(
        "Build execution layer", "Implement task graph", "tactical", 
        priority=8, parent_id=strategy.goal_id
    )
    
    # Get recommendations
    rec = user_state.get_recommendation()
    print(f"\n  User regime: {rec['regime']}")
    print(f"  Burnout risk: {rec['burnout_risk']:.2f}")
    print(f"  Complexity: {rec['complexity']}")
    
    goals = goal_hierarchy.get_prioritized_goals()
    print(f"  Top goals: {[g.title for g in goals[:3]]}")
    print(f"  Blockers: {goal_hierarchy.detect_blockers()}")


def test_execution_system():
    """Test Level 3: Execution System."""
    print("\n" + "=" * 60)
    print("LEVEL 3: EXECUTION SYSTEM TEST")
    print("=" * 60)
    
    task_graph = TaskGraph()
    user_state = UserStateModel()
    monitor = ActiveMonitor(task_graph, user_state)
    
    # Create task chain
    task1 = task_graph.create_task("Research", estimated_minutes=60)
    task2 = task_graph.create_task("Design", estimated_minutes=45, dependencies=[task1.task_id])
    task3 = task_graph.create_task("Implement", estimated_minutes=120, dependencies=[task2.task_id])
    
    print(f"\n  Created {len(task_graph.tasks)} tasks")
    
    # Execute
    ready = task_graph.get_ready_tasks()
    print(f"  Ready tasks: {len(ready)}")
    
    task_graph.start_task(task1.task_id)
    task_graph.complete_task(task1.task_id, {'result': 'done'})
    
    ready = task_graph.get_ready_tasks()
    print(f"  After task1: {len(ready)} ready")
    
    print(f"  Progress: {task_graph.get_progress()}")


def test_integration():
    """Test complete integration."""
    print("\n" + "=" * 60)
    print("AI-OS EXECUTIVE SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    system = AIOSExecutiveSystem()
    
    # Create goals
    system.create_goal("Build AI-OS", "Complete project", "strategic", priority=9)
    system.create_goal("Implement features", "Build components", "tactical", priority=8)
    
    # Add tasks
    system.add_task("Write code", estimated_minutes=120)
    system.add_task("Test system", estimated_minutes=60)
    system.add_task("Deploy", estimated_minutes=30)
    
    # Update user state
    system.update_user_state(energy=0.7, stress=0.3, cognitive_load=0.4)
    
    # Run cycle
    result = system.run_cycle()
    print(f"\n  Cycle result: {result}")
    
    # Dashboard
    dashboard = system.get_dashboard()
    print(f"  Goals: {dashboard['goals']['total_goals']}")
    print(f"  Tasks: {dashboard['tasks']['total']}")
    print(f"  User energy: {dashboard['user_state']['energy']:.2f}")


if __name__ == "__main__":
    test_foundation_runtime()
    test_cognitive_control()
    test_execution_system()
    test_integration()
    
    print("\n" + "=" * 60)
    print("AI-OS EXECUTIVE SYSTEM - PRACTICAL ARCHITECTURE")
    print("=" * 60)
    
    print("""
ARCHITECTURAL ORIENTATION: From research to practical

GOAL: Persistent Strategic Cognitive Operating System
  - Understanding user state
  - Holding long-term goals
  - Decomposing problems
  - Making decisions
  - Monitoring execution
  - Adapting to context

NOT: synthetic organism research
BUT: practical intelligent assistant

THE 4 LEVELS:

LEVEL 1 — FOUNDATION RUNTIME
  - Event bus for all system events
  - Persistent memory with context
  - Replay and debugging
  - Telemetry and observability

LEVEL 2 — COGNITIVE CONTROL LAYER
  - User state modeling (energy, stress, focus, motivation)
  - Goal hierarchy (strategic → tactical → operational)
  - Regime detection (deep focus, exploration, recovery, crisis)
  - Burnout prediction

LEVEL 3 — EXECUTION SYSTEM (Critical Missing Piece)
  - Task graph with dependencies
  - Critical path detection
  - Active monitoring (stuck, distracted, overloaded)
  - Adaptive intervention (simplify, recover, adapt)

LEVEL 4 — STRATEGIC META-COGNITION (Support Only)
  - Planning and decomposition
  - Success prediction
  - Execution simulation
  - Cognitive support (NOT the core)

PRACTICAL BENEFITS:
  - Real project management
  - User state tracking
  - Burnout prevention
  - Adaptive task sequencing
  - Blockers detection
  - Intervention system

This transforms AI-OS from:
  research project → practical intelligent operating system
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
AI-OS Executive System Summary

The architecture has been reoriented from:
  cognitive organism research → practical intelligent system

Key changes:
  1. Foundation Runtime for infrastructure
  2. Cognitive Control for user understanding
  3. Execution System for real task management
  4. Strategic Support (not core) for advanced planning

The system now:
  - Tracks user state (energy, stress, burnout)
  - Manages goal hierarchy (strategic → operational)
  - Executes task graphs with dependencies
  - Monitors for problems and intervenes
  - Adapts to user context

This is the foundation for a practical AI operating system
that genuinely helps users achieve their goals.
"""