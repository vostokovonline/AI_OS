"""
Runtime Self-Monitoring

Monitors execution in real-time and adapts strategy.
Unlike post-hoc reflection, this observes → executes → self-monitors → adapts
during execution.

Key capabilities:
- Real-time progress tracking
- Anomaly detection during execution
- Strategy adaptation mid-execution
- Resource pressure monitoring
- Error pattern detection
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import time


class ExecutionPhase(Enum):
    """Phases of goal execution"""
    PLANNING = "planning"
    DECOMPOSITION = "decomposition"
    SKILL_SELECTION = "skill_selection"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    REFLECTION = "reflection"


class AnomalyType(Enum):
    """Types of anomalies during execution"""
    STALL = "stall"                    # No progress for too long
    ERROR_ESCALATION = "error_escalation"  # Errors getting worse
    RESOURCE_DEPLETION = "resource_depletion"  # Running out of resources
    DEADLOCK = "deadlock"             # Circular dependencies
    QUALITY_DEGRADATION = "quality_degradation"  # Output quality dropping
    DRIFT = "drift"                   # Goal scope changing


@dataclass
class ExecutionSnapshot:
    """Single point in execution monitoring"""
    timestamp: str
    phase: ExecutionPhase
    progress: float  # 0-1
    errors_count: int
    artifacts_count: int
    resource_usage: float  # 0-1
    quality_score: float  # 0-1
    
    # Context
    current_step: str = ""
    active_skills: List[str] = field(default_factory=list)
    pending_dependencies: int = 0


@dataclass
class AnomalyDetection:
    """Detected anomaly during execution"""
    type: AnomalyType
    severity: float  # 0-1
    first_detected: str
    occurrence_count: int = 1
    
    # Details
    description: str = ""
    suggested_action: str = ""


class RuntimeSelfMonitor:
    """
    Monitors execution in real-time and can adapt.
    
    Unlike traditional monitoring (just observability),
    this can actively change execution strategy.
    """
    
    def __init__(self):
        self._current_execution_id: Optional[str] = None
        self._snapshots: List[ExecutionSnapshot] = []
        self._anomalies: List[AnomalyDetection] = []
        
        # Thresholds
        self._stall_threshold_seconds = 60
        self._error_escalation_threshold = 3
        self._resource_critical = 0.9
        
        # Callbacks for adaptation (set by Cognitive Orchestrator)
        self._adaptation_callbacks: Dict[AnomalyType, Callable] = {}
        
        # State
        self._execution_start_time: Optional[float] = None
        self._last_progress_time: Optional[float] = None
        self._last_progress_value: float = 0.0
    
    def start_monitoring(self, execution_id: str, initial_plan: Dict):
        """Start monitoring an execution"""
        
        self._current_execution_id = execution_id
        self._snapshots = []
        self._anomalies = []
        self._execution_start_time = time.time()
        self._last_progress_time = time.time()
        self._last_progress_value = 0.0
        
        # Record initial snapshot
        self._record_snapshot(ExecutionSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            phase=ExecutionPhase.PLANNING,
            progress=0.0,
            errors_count=0,
            artifacts_count=0,
            resource_usage=0.1,
            quality_score=0.5,
            current_step=initial_plan.get('title', 'Unknown')
        ))
    
    def _record_snapshot(self, snapshot: ExecutionSnapshot):
        """Record a monitoring snapshot"""
        self._snapshots.append(snapshot)
        
        # Keep last 100 snapshots
        if len(self._snapshots) > 100:
            self._snapshots = self._snapshots[-100:]
        
        # Check for anomalies
        self._check_anomalies(snapshot)
    
    def _check_anomalies(self, snapshot: ExecutionSnapshot):
        """Check current snapshot for anomalies"""
        
        # Anomaly 1: Stall (no progress)
        if time.time() - self._last_progress_time > self._stall_threshold_seconds:
            if snapshot.progress == self._last_progress_value:
                self._register_anomaly(AnomalyType.STALL, 0.7, 
                    "No progress detected for >60s")
        
        # Anomaly 2: Error escalation
        if snapshot.errors_count > self._error_escalation_threshold:
            self._register_anomaly(AnomalyType.ERROR_ESCALATION, 
                min(1.0, snapshot.errors_count / 10),
                f"Multiple errors detected: {snapshot.errors_count}")
        
        # Anomaly 3: Resource depletion
        if snapshot.resource_usage > self._resource_critical:
            self._register_anomaly(AnomalyType.RESOURCE_DEPLETION, 
                snapshot.resource_usage,
                "Critical resource usage")
        
        # Anomaly 4: Quality degradation
        if len(self._snapshots) >= 5:
            recent_qualities = [s.quality_score for s in self._snapshots[-5:]]
            if sum(recent_qualities) / 5 < 0.3:
                self._register_anomaly(AnomalyType.QUALITY_DEGRADATION, 0.6,
                    "Quality consistently low")
    
    def _register_anomaly(self, anomaly_type: AnomalyType, severity: float, description: str):
        """Register an anomaly and trigger adaptation if needed"""
        
        # Check if already tracking this anomaly
        existing = next((a for a in self._anomalies if a.type == anomaly_type), None)
        
        if existing:
            existing.occurrence_count += 1
            existing.severity = max(existing.severity, severity)
        else:
            anomaly = AnomalyDetection(
                type=anomaly_type,
                severity=severity,
                first_detected=datetime.utcnow().isoformat(),
                description=description,
                suggested_action=self._get_suggested_action(anomaly_type)
            )
            self._anomalies.append(anomaly)
            
            # Trigger adaptation callback if registered
            if anomaly_type in self._adaptation_callbacks:
                try:
                    self._adaptation_callbacks[anomaly_type](anomaly)
                except Exception:
                    pass
    
    def _get_suggested_action(self, anomaly_type: AnomalyType) -> str:
        """Get suggested action for anomaly type"""
        
        actions = {
            AnomalyType.STALL: "Try alternative approach, simplify goal, or request user input",
            AnomalyType.ERROR_ESCALATION: "Stop current execution, analyze error pattern, restart with different strategy",
            AnomalyType.RESOURCE_DEPLETION: "Reduce scope, prioritize critical path, or wait for resources",
            AnomalyType.DEADLOCK: "Break circular dependencies, add manual intervention point",
            AnomalyType.QUALITY_DEGRADATION: "Add verification step, adjust skill parameters, or simplify output requirements",
            AnomalyType.DRIFT: "Reconfirm original goal scope, reset to planning phase"
        }
        
        return actions.get(anomaly_type, "Manual intervention required")
    
    def update_progress(
        self,
        phase: ExecutionPhase,
        progress: float,
        current_step: str,
        **kwargs
    ):
        """Update execution progress"""
        
        if progress > self._last_progress_value:
            self._last_progress_time = time.time()
            self._last_progress_value = progress
        
        snapshot = ExecutionSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            phase=phase,
            progress=progress,
            errors_count=kwargs.get('errors_count', 0),
            artifacts_count=kwargs.get('artifacts_count', 0),
            resource_usage=kwargs.get('resource_usage', 0.5),
            quality_score=kwargs.get('quality_score', 0.5),
            current_step=current_step,
            active_skills=kwargs.get('active_skills', []),
            pending_dependencies=kwargs.get('pending_dependencies', 0)
        )
        
        self._record_snapshot(snapshot)
    
    def register_adaptation(self, anomaly_type: AnomalyType, callback: Callable):
        """Register callback to run when anomaly is detected"""
        self._adaptation_callbacks[anomaly_type] = callback
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        
        if not self._snapshots:
            return {'status': 'not_started'}
        
        latest = self._snapshots[-1]
        
        # Calculate runtime
        runtime = 0.0
        if self._execution_start_time:
            runtime = time.time() - self._execution_start_time
        
        return {
            'execution_id': self._current_execution_id,
            'phase': latest.phase.value,
            'progress': latest.progress,
            'runtime_seconds': runtime,
            'errors': latest.errors_count,
            'artifacts': latest.artifacts_count,
            'resource_usage': latest.resource_usage,
            'quality': latest.quality_score,
            'anomalies_count': len(self._anomalies),
            'anomalies': [
                {'type': a.type.value, 'severity': a.severity, 'suggestion': a.suggested_action}
                for a in self._anomalies
            ]
        }
    
    def should_adapt(self) -> bool:
        """Check if adaptation is needed"""
        
        # Adapt if:
        # - High severity anomalies detected
        # - Multiple anomalies
        # - Quality critically low
        
        if not self._anomalies:
            return False
        
        max_severity = max(a.severity for a in self._anomalies)
        
        return max_severity > 0.7 or len(self._anomalies) > 2
    
    def get_adaptation_plan(self) -> Dict[str, Any]:
        """Get plan for adapting to current situation"""
        
        if not self.should_adapt():
            return {'action': 'continue', 'reason': 'no_critical_issues'}
        
        # Sort anomalies by severity
        sorted_anomalies = sorted(self._anomalies, key=lambda a: a.severity, reverse=True)
        primary = sorted_anomalies[0]
        
        return {
            'action': 'adapt',
            'primary_anomaly': primary.type.value,
            'severity': primary.severity,
            'suggested_action': primary.suggested_action,
            'all_anomalies': [a.type.value for a in self._anomalies]
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return summary"""
        
        summary = self.get_current_status()
        
        # Calculate metrics
        if len(self._snapshots) > 1:
            # Progress velocity
            total_progress = self._snapshots[-1].progress - self._snapshots[0].progress
            total_time = len(self._snapshots)  # snapshots count as time units
            velocity = total_progress / max(total_time, 1)
            
            summary['velocity'] = velocity
            summary['stability'] = 1.0 - (len(self._anomalies) / max(len(self._snapshots), 1))
        
        # Reset
        self._current_execution_id = None
        self._snapshots = []
        self._anomalies = []
        
        return summary


# Global instance
_runtime_monitor: Optional[RuntimeSelfMonitor] = None


def get_runtime_monitor() -> RuntimeSelfMonitor:
    global _runtime_monitor
    if _runtime_monitor is None:
        _runtime_monitor = RuntimeSelfMonitor()
    return _runtime_monitor