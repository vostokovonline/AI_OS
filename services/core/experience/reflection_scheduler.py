"""
Reflection Scheduler - Budget-aware, entropy-sensitive, contradiction-priority scheduling

Provides:
- Cognitive budget enforcement
- Entropy threshold triggering
- Contradiction priority queue
- Cooldown windows
- Reflection depth limits

Key principle:
    Reflection is expensive. Don't reflect unless necessary.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4


class ReflectionPriority(Enum):
    """Reflection urgency levels"""
    IDLE = "idle"  # No reflection needed
    LOW = "low"    # Nice to reflect
    MEDIUM = "medium"  # Should reflect
    HIGH = "high"  # Must reflect
    CRITICAL = "critical"  # Immediate reflection required


class ReflectionType(Enum):
    """Types of reflection"""
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    BELIEF_STABILIZATION = "belief_stabilization"
    ATTRACTOR_ANALYSIS = "attractor_analysis"
    CAUSAL_AUDIT = "causal_audit"
    POLICY_ADJUSTMENT = "policy_adjustment"
    META_COGNITION = "meta_cognition"


@dataclass
class ReflectionTask:
    """Scheduled reflection task"""
    task_id: str
    reflection_type: ReflectionType
    priority: ReflectionPriority
    trigger_reason: str
    scheduled_at: str
    estimated_cost: float
    depth: int
    trigger_beliefs: List[str] = field(default_factory=list)
    trigger_contradictions: List[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    success: Optional[bool] = None
    outcome_summary: Optional[str] = None


@dataclass
class ReflectionBudget:
    """Budget tracking for reflection"""
    max_reflections_per_window: int  # Max reflections in time window
    window_minutes: int  # Time window in minutes
    current_period_reflections: int = 0
    period_start: str = ""
    
    # Cost budget
    max_total_cost_per_hour: float = 100.0
    current_hour_cost: float = 0.0
    
    # Cooldown tracking
    last_reflection_time: Optional[str] = None
    cooldown_minutes: int = 5  # Minimum time between reflections
    
    # Failure tracking
    consecutive_failures: int = 0
    max_consecutive_failures: int = 3


@dataclass
class ReflectionMetrics:
    """Metrics for reflection decisions"""
    current_entropy: float
    entropy_delta: float  # Change over time
    contradiction_density: float  # contradictions / beliefs
    persistent_contradictions: int
    oscillating_beliefs: int
    attractor_instability: float  # How many beliefs changing attractor states
    
    # Budget
    budget_available: bool
    reflections_in_window: int
    cooldown_active: bool
    
    # Trigger scores
    contradiction_score: float  # 0-1 urgency from contradictions
    entropy_score: float  # 0-1 urgency from entropy
    attractor_score: float  # 0-1 urgency from attractor instability


class ReflectionScheduler:
    """
    Reflection Scheduler - budget-aware, entropy-sensitive scheduling.
    
    Decides WHEN to reflect, not HOW to reflect.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Budget configuration
        self._budget = ReflectionBudget(
            max_reflections_per_window=self.config.get("max_reflections_per_window", 5),
            window_minutes=self.config.get("window_minutes", 60),
            max_total_cost_per_hour=self.config.get("max_total_cost_per_hour", 100.0),
            cooldown_minutes=self.config.get("cooldown_minutes", 5),
            max_consecutive_failures=self.config.get("max_consecutive_failures", 3),
            period_start=datetime.utcnow().isoformat()
        )
        
        # Pending tasks
        self._pending_tasks: List[ReflectionTask] = []
        self._completed_tasks: List[ReflectionTask] = []
        
        # History for scoring
        self._entropy_history: List[float] = []
        self._contradiction_history: List[int] = []
        
        # Thresholds
        self._entropy_threshold = self.config.get("entropy_threshold", 0.8)
        self._contradiction_density_threshold = self.config.get("contradiction_density_threshold", 0.3)
        self._attractor_instability_threshold = self.config.get("attractor_instability_threshold", 0.2)
    
    def should_reflect(
        self,
        current_state: Any,  # UnifiedEpistemicState
        ues_manager: Any  # For accessing metrics
    ) -> tuple[bool, ReflectionPriority, str]:
        """
        Decide if reflection should be triggered.
        
        Returns: (should_reflect, priority, reason)
        """
        
        # Get metrics from UES
        metrics = ues_manager.get_metrics()
        
        # Check budget first
        if not self._check_budget_available():
            return False, ReflectionPriority.IDLE, "budget_exhausted"
        
        # Check cooldown
        if self._is_cooldown_active():
            return False, ReflectionPriority.IDLE, "cooldown_active"
        
        # Check consecutive failures
        if self._budget.consecutive_failures >= self._budget.max_consecutive_failures:
            return False, ReflectionPriority.IDLE, "consecutive_failures_exceeded"
        
        # Compute reflection metrics
        reflection_metrics = self._compute_metrics(current_state)
        
        # Store for history
        self._entropy_history.append(reflection_metrics.current_entropy)
        if len(self._entropy_history) > 10:
            self._entropy_history.pop(0)
        
        self._contradiction_history.append(len(current_state.contradictions))
        if len(self._contradiction_history) > 10:
            self._contradiction_history.pop(0)
        
        # Determine priority based on scores
        priority, reason = self._determine_priority(reflection_metrics)
        
        should_reflect = priority != ReflectionPriority.IDLE
        
        return should_reflect, priority, reason
    
    def _check_budget_available(self) -> bool:
        """Check if budget allows new reflection"""
        
        now = datetime.utcnow()
        
        # Check time window
        if self._budget.period_start:
            period_start = datetime.fromisoformat(self._budget.period_start)
            if (now - period_start).total_seconds() / 60 > self._budget.window_minutes:
                # Reset window
                self._budget.current_period_reflections = 0
                self._budget.period_start = now.isoformat()
        
        # Check max reflections
        if self._budget.current_period_reflections >= self._budget.max_reflections_per_window:
            return False
        
        # Check cost budget
        if self._budget.current_hour_cost >= self._budget.max_total_cost_per_hour:
            return False
        
        return True
    
    def _is_cooldown_active(self) -> bool:
        """Check if cooldown period is active"""
        
        if not self._budget.last_reflection_time:
            return False
        
        last_time = datetime.fromisoformat(self._budget.last_reflection_time)
        cooldown_seconds = self._budget.cooldown_minutes * 60
        
        return (datetime.utcnow() - last_time).total_seconds() < cooldown_seconds
    
    def _compute_metrics(self, state: Any) -> ReflectionMetrics:
        """Compute all metrics for reflection decision"""
        
        # Current entropy
        current_entropy = state.total_entropy
        
        # Entropy delta (change over recent history)
        entropy_delta = 0.0
        if len(self._entropy_history) >= 2:
            entropy_delta = current_entropy - self._entropy_history[-2]
        
        # Contradiction density
        belief_count = max(len(state.beliefs), 1)
        contradiction_density = len(state.contradictions) / belief_count
        
        # Persistent contradictions
        persistent = sum(
            1 for c in state.contradictions.values()
            if c.stability_score > 0.7
        )
        
        # Oscillating beliefs
        oscillating = sum(
            1 for b in state.beliefs.values()
            if b.attractor_state == "oscillating"
        )
        
        # Attractor instability
        attractor_instability = oscillating / max(len(state.beliefs), 1)
        
        # Budget status
        budget_available = self._check_budget_available()
        reflections_in_window = self._budget.current_period_reflections
        cooldown_active = self._is_cooldown_active()
        
        # Compute scores
        contradiction_score = self._compute_contradiction_score(
            contradiction_density, persistent
        )
        entropy_score = self._compute_entropy_score(
            current_entropy, entropy_delta
        )
        attractor_score = self._compute_attractor_score(attractor_instability)
        
        return ReflectionMetrics(
            current_entropy=current_entropy,
            entropy_delta=entropy_delta,
            contradiction_density=contradiction_density,
            persistent_contradictions=persistent,
            oscillating_beliefs=oscillating,
            attractor_instability=attractor_instability,
            budget_available=budget_available,
            reflections_in_window=reflections_in_window,
            cooldown_active=cooldown_active,
            contradiction_score=contradiction_score,
            entropy_score=entropy_score,
            attractor_score=attractor_score
        )
    
    def _compute_contradiction_score(self, density: float, persistent: int) -> float:
        """Compute urgency from contradictions"""
        
        score = 0.0
        
        # Density contributes (max 0.4)
        score += min(density * 1.5, 0.4)
        
        # Persistent contradictions (max 0.4)
        score += min(persistent * 0.15, 0.4)
        
        # Critical contradictions (max 0.2)
        # (would need to check severity)
        
        return min(score, 1.0)
    
    def _compute_entropy_score(self, entropy: float, delta: float) -> float:
        """Compute urgency from entropy"""
        
        score = 0.0
        
        # High entropy contributes
        if entropy > self._entropy_threshold:
            score += 0.3
        
        # Rapid increase contributes more
        if delta > 0.1:
            score += 0.4
        elif delta > 0.05:
            score += 0.2
        
        # Very high entropy
        if entropy > 0.9:
            score += 0.3
        
        return min(score, 1.0)
    
    def _compute_attractor_score(self, instability: float) -> float:
        """Compute urgency from attractor instability"""
        
        if instability > self._attractor_instability_threshold:
            return min(instability * 2, 1.0)
        
        return 0.0
    
    def _determine_priority(
        self,
        metrics: ReflectionMetrics
    ) -> tuple[ReflectionPriority, str]:
        """Determine reflection priority and reason"""
        
        # Combine scores with weights
        total_score = (
            metrics.contradiction_score * 0.4 +
            metrics.entropy_score * 0.3 +
            metrics.attractor_score * 0.3
        )
        
        # Determine priority
        if total_score >= 0.8:
            return ReflectionPriority.CRITICAL, f"critical_score_{total_score:.2f}"
        elif total_score >= 0.6:
            return ReflectionPriority.HIGH, f"high_score_{total_score:.2f}"
        elif total_score >= 0.4:
            return ReflectionPriority.MEDIUM, f"medium_score_{total_score:.2f}"
        elif total_score >= 0.2:
            return ReflectionPriority.LOW, f"low_score_{total_score:.2f}"
        
        return ReflectionPriority.IDLE, "no_significant_triggers"
    
    def schedule_reflection(
        self,
        reflection_type: ReflectionType,
        priority: ReflectionPriority,
        trigger_beliefs: List[str],
        trigger_reason: str,
        estimated_cost: float = 1.0
    ) -> ReflectionTask:
        """Schedule a reflection task"""
        
        task = ReflectionTask(
            task_id=str(uuid4()),
            reflection_type=reflection_type,
            priority=priority,
            trigger_beliefs=trigger_beliefs,
            trigger_reason=trigger_reason,
            scheduled_at=datetime.utcnow().isoformat(),
            estimated_cost=estimated_cost,
            depth=0
        )
        
        self._pending_tasks.append(task)
        
        return task
    
    def execute_task(self, task_id: str, success: bool, outcome: str):
        """Mark task as executed and update budget"""
        
        for task in self._pending_tasks:
            if task.task_id == task_id:
                task.success = success
                task.outcome_summary = outcome
                task.completed_at = datetime.utcnow().isoformat()
                break
        
        # Move from pending to completed
        self._pending_tasks = [t for t in self._pending_tasks if t.task_id != task_id]
        
        # Find task in completed (if it was moved)
        completed = [t for t in self._completed_tasks if t.task_id == task_id]
        if completed:
            task = completed[0]
        else:
            # Search in pending before move
            return
        
        # Update budget
        self._budget.current_period_reflections += 1
        self._budget.last_reflection_time = datetime.utcnow().isoformat()
        self._budget.current_hour_cost += task.estimated_cost
        
        if success:
            self._budget.consecutive_failures = 0
        else:
            self._budget.consecutive_failures += 1
    
    def get_pending_tasks(self) -> List[ReflectionTask]:
        """Get pending reflection tasks"""
        return self._pending_tasks.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get scheduler metrics"""
        return {
            "budget_available": self._check_budget_available(),
            "reflections_in_window": self._budget.current_period_reflections,
            "consecutive_failures": self._budget.consecutive_failures,
            "cooldown_active": self._is_cooldown_active(),
            "pending_tasks": len(self._pending_tasks),
            "completed_tasks": len(self._completed_tasks),
            "entropy_history": self._entropy_history[-5:] if self._entropy_history else []
        }


# Global instance
_scheduler: Optional[ReflectionScheduler] = None


def get_reflection_scheduler(config: Optional[Dict] = None) -> ReflectionScheduler:
    """Get global reflection scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = ReflectionScheduler(config)
    return _scheduler


def should_reflect(
    ues_manager: Any,
    current_state: Any
) -> tuple[bool, ReflectionPriority, str]:
    """Convenience function to check if reflection needed"""
    scheduler = get_reflection_scheduler()
    return scheduler.should_reflect(current_state, ues_manager)