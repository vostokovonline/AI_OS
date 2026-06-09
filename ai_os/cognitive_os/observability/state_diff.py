"""
State Diff Engine - Track and visualize state changes over time

Shows how UnifiedState evolves across decisions and why.

Components:
- StateDelta: Difference between two states
- StateHistory: Time-series of state snapshots
- DiffAnalyzer: Analyze state changes and patterns
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class StateDelta:
    """Difference between two state snapshots"""
    from_state: Dict[str, float]
    to_state: Dict[str, float]
    
    changes: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    absolute_changes: Dict[str, float] = field(default_factory=dict)
    relative_changes: Dict[str, float] = field(default_factory=dict)
    
    timestamp_from: datetime = field(default_factory=datetime.utcnow)
    timestamp_to: datetime = field(default_factory=datetime.utcnow)
    
    major_changes: List[str] = field(default_factory=list)
    minor_changes: List[str] = field(default_factory=list)


@dataclass
class StateSnapshot:
    """State at a point in time"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    state_type: str = "unified"
    
    world_outcome: str = "unknown"
    world_entities: int = 0
    world_capability: float = 0.5
    
    identity_coherence: float = 0.5
    identity_emotion: str = "neutral"
    
    arousal: float = 0.5
    valence: float = 0.0
    focus: float = 0.5
    confidence: float = 0.5
    
    bias_count: int = 0
    bias_awareness: float = 0.5
    reflection_depth: float = 0.5
    
    top_strategy: str = "default"
    strategy_score: float = 0.5
    
    stress_level: float = 0.0
    exploration_tendency: float = 0.5
    action_readiness: float = 0.5
    
    task_complexity: float = 0.5
    task_urgency: float = 0.5
    task_novelty: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "state_type": self.state_type,
            "world_outcome": self.world_outcome,
            "world_entities": self.world_entities,
            "world_capability": self.world_capability,
            "identity_coherence": self.identity_coherence,
            "identity_emotion": self.identity_emotion,
            "arousal": self.arousal,
            "valence": self.valence,
            "focus": self.focus,
            "confidence": self.confidence,
            "bias_count": self.bias_count,
            "bias_awareness": self.bias_awareness,
            "reflection_depth": self.reflection_depth,
            "top_strategy": self.top_strategy,
            "strategy_score": self.strategy_score,
            "stress_level": self.stress_level,
            "exploration_tendency": self.exploration_tendency,
            "action_readiness": self.action_readiness,
            "task_complexity": self.task_complexity,
            "task_urgency": self.task_urgency,
            "task_novelty": self.task_novelty,
        }
    
    @classmethod
    def from_unified_state(cls, state) -> "StateSnapshot":
        return cls(
            timestamp=datetime.utcnow(),
            state_type="unified",
            world_outcome=state.world_recent_outcome,
            world_entities=state.world_entities_count,
            world_capability=state.world_capability_score,
            identity_coherence=state.identity_coherence,
            identity_emotion=state.identity_emotion,
            arousal=state.arousal,
            valence=state.valence,
            focus=state.focus,
            confidence=state.confidence,
            bias_count=state.bias_count,
            bias_awareness=state.bias_awareness,
            reflection_depth=state.reflection_depth,
            top_strategy=state.top_strategy_name,
            strategy_score=state.top_strategy_score,
            stress_level=state.stress_level,
            exploration_tendency=state.exploration_tendency,
            action_readiness=state.action_readiness,
            task_complexity=state.task_complexity,
            task_urgency=state.task_urgency,
            task_novelty=state.task_novelty,
        )


class StateHistory:
    """Time-series of state snapshots"""
    
    def __init__(self, max_snapshots: int = 1000):
        self.snapshots: List[StateSnapshot] = []
        self.max_snapshots = max_snapshots
        self.user_id: Optional[str] = None
    
    def add(self, snapshot: StateSnapshot) -> None:
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
    
    def get_range(
        self,
        from_time: Optional[datetime] = None,
        to_time: Optional[datetime] = None
    ) -> List[StateSnapshot]:
        result = self.snapshots
        if from_time:
            result = [s for s in result if s.timestamp >= from_time]
        if to_time:
            result = [s for s in result if s.timestamp <= to_time]
        return result
    
    def get_last(self, count: int = 10) -> List[StateSnapshot]:
        return self.snapshots[-count:]
    
    def get_trend(self, field_name: str, window: int = 10) -> List[float]:
        values = []
        for snapshot in self.snapshots[-window:]:
            value = getattr(snapshot, field_name, None)
            if value is not None:
                values.append(float(value))
        return values
    
    def compute_diffs(self, window: int = 2) -> List[StateDelta]:
        if len(self.snapshots) < 2:
            return []
        
        diffs = []
        for i in range(len(self.snapshots) - 1, max(0, len(self.snapshots) - window - 1), -1):
            prev = self.snapshots[i - 1]
            curr = self.snapshots[i]
            
            delta = self._compute_delta(prev, curr)
            diffs.append(delta)
        
        return list(reversed(diffs))
    
    def _compute_delta(self, prev: StateSnapshot, curr: StateSnapshot) -> StateDelta:
        from_state = prev.to_dict()
        to_state = curr.to_dict()
        
        changes = {}
        absolute_changes = {}
        relative_changes = {}
        
        for key in from_state:
            if key in ["timestamp", "state_type", "world_outcome", "identity_emotion", "top_strategy"]:
                continue
            
            old_val = from_state[key]
            new_val = to_state.get(key, old_val)
            
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                changes[key] = (float(old_val), float(new_val))
                absolute_changes[key] = abs(float(new_val) - float(old_val))
                if old_val != 0:
                    relative_changes[key] = (float(new_val) - float(old_val)) / max(abs(float(old_val)), 0.001)
                else:
                    relative_changes[key] = float(new_val) - 0
        
        major = [k for k, v in absolute_changes.items() if v > 0.2]
        minor = [k for k, v in absolute_changes.items() if 0.05 < v <= 0.2]
        
        return StateDelta(
            from_state=from_state,
            to_state=to_state,
            changes=changes,
            absolute_changes=absolute_changes,
            relative_changes=relative_changes,
            timestamp_from=prev.timestamp,
            timestamp_to=curr.timestamp,
            major_changes=major,
            minor_changes=minor
        )


class DiffAnalyzer:
    """
    Analyze state changes and patterns.
    
    Detects:
    - Abnormal changes
    - Trend shifts
    - Anomalies
    - Patterns
    """
    
    def __init__(self):
        self.baseline: Optional[Dict[str, float]] = None
        self.thresholds = {
            "major_change": 0.3,
            "anomaly": 0.5,
            "trend_window": 10
        }
        logger.info("diff_analyzer_initialized")
    
    def learn_baseline(self, snapshots: List[StateSnapshot]) -> None:
        if len(snapshots) < 5:
            return
        
        fields = ["arousal", "valence", "focus", "confidence", "stress_level", 
                  "action_readiness", "exploration_tendency"]
        
        self.baseline = {}
        for field_name in fields:
            values = [getattr(s, field_name, 0) for s in snapshots]
            if values:
                self.baseline[field_name] = sum(values) / len(values)
        
        logger.info("baseline_learned", fields=len(self.baseline))
    
    def detect_anomalies(self, current: StateSnapshot, history: StateHistory) -> List[Dict]:
        anomalies = []
        
        if not self.baseline:
            return anomalies
        
        for field_name, baseline_value in self.baseline.items():
            current_value = getattr(current, field_name, baseline_value)
            deviation = abs(current_value - baseline_value)
            
            if deviation > self.thresholds["anomaly"]:
                anomalies.append({
                    "field": field_name,
                    "baseline": baseline_value,
                    "current": current_value,
                    "deviation": deviation,
                    "severity": "high" if deviation > 0.5 else "medium"
                })
        
        return anomalies
    
    def detect_trends(self, history: StateHistory, field_name: str) -> Dict:
        values = history.get_trend(field_name, window=self.thresholds["trend_window"])
        
        if len(values) < 3:
            return {"trend": "insufficient_data"}
        
        increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
        decreasing = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
        
        if increasing > len(values) * 0.7:
            return {"trend": "increasing", "stability": increasing / len(values)}
        elif decreasing > len(values) * 0.7:
            return {"trend": "decreasing", "stability": decreasing / len(values)}
        else:
            return {"trend": "stable", "stability": 1 - abs(increasing - decreasing) / len(values)}
    
    def explain_state_change(self, delta: StateDelta) -> Dict[str, Any]:
        explanations = []
        
        for change in delta.major_changes:
            old_val, new_val = delta.changes.get(change, (0, 0))
            direction = "increased" if new_val > old_val else "decreased"
            explanations.append({
                "field": change,
                "direction": direction,
                "from": old_val,
                "to": new_val,
                "change": delta.absolute_changes.get(change, 0),
                "explanation": self._explain_change(change, old_val, new_val)
            })
        
        return {
            "timestamp": delta.timestamp_to.isoformat(),
            "major_changes": explanations,
            "minor_changes": delta.minor_changes,
            "summary": self._summarize_changes(delta)
        }
    
    def _explain_change(self, field: str, old_val: float, new_val: float) -> str:
        explanations = {
            "stress_level": f"Stress {'increased' if new_val > old_val else 'decreased'} by {(new_val - old_val)*100:.0f}%"
                           f" - {'system under pressure' if new_val > 0.6 else 'system calm'}",
            "confidence": f"Confidence {'increased' if new_val > old_val else 'decreased'} by {(new_val - old_val)*100:.0f}%"
                         f" - {'more certain in decisions' if new_val > 0.6 else 'less certain'}",
            "action_readiness": f"Action readiness {'increased' if new_val > old_val else 'decreased'} by {(new_val - old_val)*100:.0f}%",
            "arousal": f"Arousal {'increased' if new_val > old_val else 'decreased'} - {'high activation state' if new_val > 0.6 else 'low activation'}",
            "valence": f"Emotional valence shifted from {old_val:.2f} to {new_val:.2f} - {'more positive' if new_val > old_val else 'more negative'}",
            "focus": f"Focus {'increased' if new_val > old_val else 'decreased'} - {'high concentration' if new_val > 0.6 else 'diffused attention'}",
            "bias_count": f"Bias count {'increased' if new_val > old_val else 'decreased'} - {'more caution needed' if new_val > 2 else 'normal operation'}",
            "exploration_tendency": f"Exploration tendency {'increased' if new_val > old_val else 'decreased'} - {'more exploration' if new_val > 0.6 else 'more exploitation'}",
        }
        return explanations.get(field, f"Field {field} changed from {old_val:.2f} to {new_val:.2f}")
    
    def _summarize_changes(self, delta: StateDelta) -> str:
        if not delta.major_changes:
            return "No significant state changes detected"
        
        change_types = []
        for change in delta.major_changes:
            if change in ["stress_level", "arousal"]:
                change_types.append("activation")
            elif change in ["confidence", "action_readiness"]:
                change_types.append("capability")
            elif change in ["exploration_tendency"]:
                change_types.append("exploration")
        
        if "activation" in change_types and "capability" in change_types:
            return "System shows mixed activation and capability changes"
        elif change_types:
            return f"System shows {' + '.join(set(change_types))} pattern"
        else:
            return f"{len(delta.major_changes)} significant state changes"


class StateDiffEngine:
    """
    Main interface for state diff analysis.
    
    Usage:
        engine = StateDiffEngine()
        
        # Track state
        engine.record_state(unified_state)
        
        # Get diffs
        diffs = engine.get_recent_diffs()
        
        # Analyze patterns
        analysis = engine.analyze_patterns()
    """
    
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id
        self.history = StateHistory()
        self.analyzer = DiffAnalyzer()
        logger.info("state_diff_engine_initialized", user_id=user_id)
    
    def record_state(self, unified_state) -> None:
        """Record a new state snapshot"""
        snapshot = StateSnapshot.from_unified_state(unified_state)
        self.history.add(snapshot)
        
        if len(self.history.snapshots) >= 5:
            self.analyzer.learn_baseline(self.history.snapshots[-20:])
        
        logger.debug("state_recorded", timestamp=snapshot.timestamp.isoformat())
    
    def get_recent_diffs(self, count: int = 5) -> List[Dict]:
        """Get recent state diffs"""
        diffs = self.history.compute_diffs(window=count)
        return [self.analyzer.explain_state_change(d) for d in diffs]
    
    def get_current_state(self) -> Optional[StateSnapshot]:
        """Get most recent state"""
        return self.history.snapshots[-1] if self.history.snapshots else None
    
    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze state patterns and trends"""
        analysis = {
            "trends": {},
            "anomalies": [],
            "patterns": []
        }
        
        fields_to_analyze = ["stress_level", "confidence", "action_readiness", 
                            "arousal", "exploration_tendency", "focus"]
        
        for field in fields_to_analyze:
            trend = self.analyzer.detect_trends(self.history, field)
            if trend.get("trend") != "insufficient_data":
                analysis["trends"][field] = trend
        
        current = self.get_current_state()
        if current:
            anomalies = self.analyzer.detect_anomalies(current, self.history)
            if anomalies:
                analysis["anomalies"] = anomalies
        
        if self.history.snapshots:
            recent_stress = [s.stress_level for s in self.history.snapshots[-5:]]
            if all(s > 0.5 for s in recent_stress):
                analysis["patterns"].append({
                    "type": "sustained_stress",
                    "description": "System under sustained stress for 5+ decisions"
                })
            
            recent_confidence = [s.confidence for s in self.history.snapshots[-5:]]
            if all(c < 0.4 for c in recent_confidence):
                analysis["patterns"].append({
                    "type": "declining_confidence",
                    "description": "Confidence consistently declining over 5+ decisions"
                })
        
        return analysis
    
    def get_state_report(self) -> Dict[str, Any]:
        """Get comprehensive state report"""
        current = self.get_current_state()
        recent_diffs = self.get_recent_diffs(3)
        patterns = self.analyze_patterns()
        
        return {
            "user_id": self.user_id,
            "current_state": current.to_dict() if current else None,
            "recent_diffs": recent_diffs,
            "patterns": patterns,
            "snapshots_count": len(self.history.snapshots)
        }