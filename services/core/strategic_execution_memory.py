"""
Strategic Execution Memory

Tracks execution outcomes and their strategic implications.
Unlike semantic memory (patterns), this tracks:
- Which strategies work
- Which break continuity
- Which create leverage
- Which cause instability

This is the "procedural memory" of the cognitive system.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque


@dataclass
class ExecutionRecord:
    """Single execution with strategic implications"""
    execution_id: str
    goal_type: str
    domain: str
    strategy_used: str
    outcome: str  # success, failure, partial
    duration_ms: int
    artifacts_count: int
    
    # Strategic metrics
    continuity_impact: float = 0.0  # -1 to +1
    leverage_creation: float = 0.0   # 0 to 1
    stability_impact: float = 0.0    # -1 to +1
    
    # Context
    context_snapshot: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass  
class StrategyProfile:
    """Learned strategy characteristics"""
    strategy_name: str
    
    # Performance metrics
    success_rate: float = 0.0
    avg_duration_ms: int = 0
    avg_leverage: float = 0.0
    
    # Continuity impact
    avg_continuity_impact: float = 0.0
    avg_stability_impact: float = 0.0
    
    # Usage
    usage_count: int = 0
    last_used: str = ""
    
    # Tags for strategy selection
    tags: List[str] = field(default_factory=list)


class StrategicExecutionMemory:
    """
    Tracks strategic patterns in execution.
    
    Key insight: Not just "what worked" but "what strengthens the system".
    """
    
    def __init__(self, max_history: int = 1000):
        self._max_history = max_history
        self._execution_history: deque = deque(maxlen=max_history)
        
        # Strategy profiles - learned from execution
        self._strategy_profiles: Dict[str, StrategyProfile] = {}
        
        # Aggregate insights
        self._domain_success_rates: Dict[str, float] = {}
        self._goal_type_patterns: Dict[str, List[str]] = {}  # goal_type -> list of good strategies
    
    def record_execution(
        self,
        execution: ExecutionRecord
    ):
        """Record an execution and update strategy profiles"""
        
        # Add to history
        self._execution_history.append(execution)
        
        # Update strategy profile
        self._update_strategy_profile(execution)
        
        # Update domain patterns
        self._update_domain_patterns(execution)
        
        # Update goal type patterns
        self._update_goal_type_patterns(execution)
    
    def _update_strategy_profile(self, execution: ExecutionRecord):
        """Update learned profile for a strategy"""
        
        profile = self._strategy_profiles.get(execution.strategy_used)
        
        if profile is None:
            profile = StrategyProfile(strategy_name=execution.strategy_used)
            self._strategy_profiles[execution.strategy_used] = profile
        
        # Update metrics
        profile.usage_count += 1
        profile.last_used = execution.timestamp
        
        # Running averages
        n = profile.usage_count
        
        if execution.outcome == 'success':
            profile.success_rate = (profile.success_rate * (n-1) + 1.0) / n
        elif execution.outcome == 'failure':
            profile.success_rate = (profile.success_rate * (n-1)) / n
        
        profile.avg_duration_ms = (
            (profile.avg_duration_ms * (n-1) + execution.duration_ms) / n
        )
        
        profile.avg_leverage = (
            (profile.avg_leverage * (n-1) + execution.leverage_creation) / n
        )
        
        profile.avg_continuity_impact = (
            (profile.avg_continuity_impact * (n-1) + execution.continuity_impact) / n
        )
        
        profile.avg_stability_impact = (
            (profile.avg_stability_impact * (n-1) + execution.stability_impact) / n
        )
    
    def _update_domain_patterns(self, execution: ExecutionRecord):
        """Track which strategies work for which domains"""
        
        if execution.domain not in self._domain_success_rates:
            self._domain_success_rates[execution.domain] = 0.5
        
        n = len([e for e in self._execution_history if e.domain == execution.domain])
        if n > 0:
            successes = sum(1 for e in self._execution_history 
                          if e.domain == execution.domain and e.outcome == 'success')
            self._domain_success_rates[execution.domain] = successes / n
    
    def _update_goal_type_patterns(self, execution: ExecutionRecord):
        """Learn which strategies work for goal types"""
        
        if execution.goal_type not in self._goal_type_patterns:
            self._goal_type_patterns[execution.goal_type] = []
        
        if execution.outcome == 'success':
            patterns = self._goal_type_patterns[execution.goal_type]
            if execution.strategy_used not in patterns[-5:]:  # Keep last 5 unique
                patterns.append(execution.strategy_used)
    
    def get_recommended_strategy(
        self,
        goal_type: str,
        domain: str,
        continuity_state: float
    ) -> str:
        """
        Get best strategy based on:
        1. Historical success for goal type
        2. Domain fit
        3. Current continuity state
        """
        
        candidates = []
        
        # Get strategies that worked for this goal type
        if goal_type in self._goal_type_patterns:
            for strategy_name in self._goal_type_patterns[goal_type]:
                profile = self._strategy_profiles.get(strategy_name)
                if profile:
                    candidates.append((strategy_name, profile.success_rate * profile.avg_leverage))
        
        # If no history, use general best performers
        if not candidates:
            for name, profile in self._strategy_profiles.items():
                if profile.usage_count >= 3:
                    candidates.append((name, profile.success_rate * 0.5))
        
        if not candidates:
            return "balanced_execute"  # Default
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Adjust for continuity state
        # Low continuity = be more conservative
        if continuity_state < 0.5:
            # Prefer stable strategies
            stable_candidates = [
                (n, s) for n, s in candidates 
                if self._strategy_profiles.get(n, StrategyProfile(n)).avg_stability_impact > 0
            ]
            if stable_candidates:
                candidates = stable_candidates
        
        return candidates[0][0]
    
    def get_strategic_insights(self) -> Dict[str, Any]:
        """Get aggregate strategic insights"""
        
        total_executions = len(self._execution_history)
        
        if total_executions == 0:
            return {'status': 'no_data'}
        
        # Calculate aggregate metrics
        successes = sum(1 for e in self._execution_history if e.outcome == 'success')
        failures = sum(1 for e in self._execution_history if e.outcome == 'failure')
        
        avg_leverage = sum(e.leverage_creation for e in self._execution_history) / total_executions
        avg_continuity = sum(e.continuity_impact for e in self._execution_history) / total_executions
        
        # Best strategies
        strategy_scores = []
        for name, profile in self._strategy_profiles.items():
            if profile.usage_count >= 3:
                score = profile.success_rate * profile.avg_leverage * (1 + profile.avg_continuity_impact)
                strategy_scores.append((name, score))
        
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'total_executions': total_executions,
            'success_rate': successes / total_executions,
            'failure_rate': failures / total_executions,
            'avg_leverage': avg_leverage,
            'avg_continuity_impact': avg_continuity,
            'best_strategies': [s[0] for s in strategy_scores[:5]],
            'domain_success_rates': self._domain_success_rates,
            'strategy_count': len(self._strategy_profiles)
        }
    
    def detect_instability_patterns(self) -> List[Dict]:
        """Detect if execution patterns are becoming unstable"""
        
        if len(self._execution_history) < 10:
            return []
        
        recent = list(self._execution_history)[-10:]
        
        patterns = []
        
        # Pattern 1: Increasing failures
        failures = sum(1 for e in recent if e.outcome == 'failure')
        if failures >= 6:
            patterns.append({
                'type': 'high_failure_rate',
                'severity': failures / 10,
                'suggestion': 'Reduce ambition, focus on recovery'
            })
        
        # Pattern 2: Continuity degradation
        if len(recent) >= 5:
            early_avg = sum(e.continuity_impact for e in recent[:5]) / 5
            late_avg = sum(e.continuity_impact for e in recent[5:]) / 5
            if late_avg < early_avg - 0.3:
                patterns.append({
                    'type': 'continuity_degradation',
                    'severity': abs(late_avg - early_avg),
                    'suggestion': 'Current strategy is breaking identity continuity'
                })
        
        # Pattern 3: Leverage collapse
        avg_leverage = sum(e.leverage_creation for e in recent) / len(recent)
        if avg_leverage < 0.2:
            patterns.append({
                'type': 'leverage_collapse',
                'severity': 0.5 - avg_leverage,
                'suggestion': 'Execution not creating value, try different approach'
            })
        
        return patterns


# Global instance
_strategic_memory: Optional[StrategicExecutionMemory] = None


def get_strategic_memory() -> StrategicExecutionMemory:
    global _strategic_memory
    if _strategic_memory is None:
        _strategic_memory = StrategicExecutionMemory()
    return _strategic_memory