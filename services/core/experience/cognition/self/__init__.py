"""
Self Model - System's model of itself.

Stage: Cognitive Architecture

The system needs a model of itself:
- What it can do (capabilities)
- What it can't do (limitations)
- How it operates (strategies)
- How it's performed (confidence)
- What resources it has (resource model)

This enables:
- Self-awareness
- Accurate self-assessment
- Strategy selection
- Failure prediction
- Learning from experience
"""
from types import MappingProxyType
from typing import Dict, Any, Optional, Tuple, List, Set, FrozenSet
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json


@dataclass(frozen=True)
class Capability:
    """Something the system can do"""
    capability_id: str
    name: str
    description: str
    proficiency: float  # 0-1, how good at this
    reliability: float  # 0-1, how consistent
    learn_speed: float  # 0-1, how fast improves
    resource_requirement: float  # 0-1, how resource intensive
    success_count: int
    failure_count: int
    last_used: str
    
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def needs_practice(self) -> bool:
        return self.success_count < 10
    
    def is_reliable(self) -> bool:
        return self.reliability >= 0.7


@dataclass(frozen=True)
class Limitation:
    """Something the system cannot do well"""
    limitation_id: str
    name: str
    severity: float  # 0-1, how limiting
    workaround: Optional[str]
    improvement_potential: float  # 0-1, how improvable
    detected_at: str
    affected_capabilities: Tuple[str, ...]


@dataclass(frozen=True)
class Strategy:
    """A strategy the system uses"""
    strategy_id: str
    name: str
    description: str
    applicable_situations: Tuple[str, ...]
    success_rate: float
    average_utility: float  # How much value this generates
    resource_cost: float
    risk_level: float
    use_count: int
    last_used: str
    
    def is_preferred(self) -> bool:
        """Strategy is preferred when high utility and low risk"""
        return self.average_utility > 0.7 and self.risk_level < 0.3
    
    def matches_situation(self, situation: str) -> bool:
        return situation in self.applicable_situations


@dataclass(frozen=True)
class ConfidenceRegion:
    """Region of high/low confidence in self-model"""
    region_id: str
    description: str
    confidence_level: float  # 0-1, how confident
    evidence: Tuple[str, ...]
    unknowns: Tuple[str, ...]
    accuracy_history: Tuple[float, ...]  # Recent accuracy measurements
    
    def is_explored(self) -> bool:
        return self.confidence_level > 0.7
    
    def is_uncertain(self) -> bool:
        return self.confidence_level < 0.4


@dataclass(frozen=True)
class ResourceModel:
    """Model of available resources"""
    attention_budget: float  # 0-1, total attention available
    memory_capacity: float  # 0-1, memory usage
    processing_capacity: float  # 0-1, CPU usage
    energy_level: float  # 0-1, energy available
    time_available: float  # hours remaining
    
    def is_strained(self) -> bool:
        """System is resource-strained"""
        return (self.attention_budget < 0.3 or 
                self.memory_capacity > 0.9 or 
                self.energy_level < 0.2)
    
    def can_attend_to(self, cost: float) -> bool:
        """Check if we can attend to something with given cost"""
        return self.attention_budget >= cost


@dataclass(frozen=True)
class SelfModel:
    """
    The system's model of itself.
    
    This is what enables:
    - Self-awareness
    - Accurate self-assessment
    - Strategy selection based on capabilities
    - Failure prediction
    - Learning from experience
    """
    capabilities: MappingProxyType  # type: ignore
    limitations: MappingProxyType  # type: ignore
    active_strategies: MappingProxyType  # type: ignore
    recent_failures: Tuple[str, ...]  # goal_ids that failed recently
    confidence_map: MappingProxyType  # type: ignore
    resource_model: ResourceModel
    self_consistency_score: float  # 0-1, how consistent self-model is
    version: int
    last_updated: str
    
    def __post_init__(self):
        for attr in ('capabilities', 'limitations', 'active_strategies', 'confidence_map'):
            val = getattr(self, attr)
            if not isinstance(val, MappingProxyType):
                object.__setattr__(self, attr, MappingProxyType(dict(val)))
    
    @staticmethod
    def compute_hash(state: 'SelfModel') -> str:
        data = {
            "version": state.version,
            "capability_count": len(state.capabilities),
            "strategy_count": len(state.active_strategies),
            "self_consistency": state.self_consistency_score,
            "resource_available": state.resource_model.energy_level > 0.5
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def get_capable_actions(self) -> List[Capability]:
        """Get all capabilities above threshold"""
        return [c for c in self.capabilities.values() if c.proficiency > 0.5]
    
    def get_limiting_factors(self) -> List[Limitation]:
        """Get limitations sorted by severity"""
        return sorted(self.limitations.values(), 
                     key=lambda l: l.severity, reverse=True)
    
    def get_preferred_strategies(self) -> List[Strategy]:
        """Get strategies that are preferred"""
        return [s for s in self.active_strategies.values() if s.is_preferred()]
    
    def get_uncertain_regions(self) -> List[ConfidenceRegion]:
        """Get regions where self-model is uncertain"""
        return [r for r in self.confidence_map.values() if r.is_uncertain()]
    
    def estimate_success_probability(self, task_type: str) -> float:
        """
        Estimate probability of success for task type.
        
        Combines relevant capabilities and strategies.
        """
        relevant = [c for c in self.capabilities.values() 
                   if task_type.lower() in c.name.lower()]
        
        if not relevant:
            return 0.3  # Unknown = moderate confidence
        
        avg_proficiency = sum(c.proficiency for c in relevant) / len(relevant)
        reliability = sum(c.reliability for c in relevant) / len(relevant)
        
        return (avg_proficiency * 0.6 + reliability * 0.4)
    
    def predict_failure_modes(self, task: str) -> List[str]:
        """
        Predict likely failure modes for task.
        
        Based on limitations and recent failures.
        """
        failure_modes = []
        
        for limitation in self.get_limiting_factors():
            if limitation.affected_capabilities:
                failure_modes.append(limitation.name)
        
        for failure in self.recent_failures[-3:]:
            failure_modes.append(f"Recent failure: {failure}")
        
        return failure_modes[:5]
    
    def with_capability(self, capability: Capability) -> 'SelfModel':
        new_caps = {**self.capabilities, capability.capability_id: capability}
        return SelfModel(
            capabilities=MappingProxyType(new_caps),
            limitations=self.limitations,
            active_strategies=self.active_strategies,
            recent_failures=self.recent_failures,
            confidence_map=self.confidence_map,
            resource_model=self.resource_model,
            self_consistency_score=self._compute_consistency(new_caps, self.limitations),
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def with_limitation(self, limitation: Limitation) -> 'SelfModel':
        new_lims = {**self.limitations, limitation.limitation_id: limitation}
        return SelfModel(
            capabilities=self.capabilities,
            limitations=MappingProxyType(new_lims),
            active_strategies=self.active_strategies,
            recent_failures=self.recent_failures,
            confidence_map=self.confidence_map,
            resource_model=self.resource_model,
            self_consistency_score=self._compute_consistency(self.capabilities, new_lims),
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def with_strategy(self, strategy: Strategy) -> 'SelfModel':
        new_strategies = {**self.active_strategies, strategy.strategy_id: strategy}
        return SelfModel(
            capabilities=self.capabilities,
            limitations=self.limitations,
            active_strategies=MappingProxyType(new_strategies),
            recent_failures=self.recent_failures,
            confidence_map=self.confidence_map,
            resource_model=self.resource_model,
            self_consistency_score=self.self_consistency_score,
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def add_failure(self, goal_id: str) -> 'SelfModel':
        """Record a failure"""
        new_failures = self.recent_failures[-19:] + (goal_id,)
        return SelfModel(
            capabilities=self.capabilities,
            limitations=self.limitations,
            active_strategies=self.active_strategies,
            recent_failures=new_failures,
            confidence_map=self.confidence_map,
            resource_model=self.resource_model,
            self_consistency_score=self.self_consistency_score,
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    def update_resource_model(self, model: ResourceModel) -> 'SelfModel':
        return SelfModel(
            capabilities=self.capabilities,
            limitations=self.limitations,
            active_strategies=self.active_strategies,
            recent_failures=self.recent_failures,
            confidence_map=self.confidence_map,
            resource_model=model,
            self_consistency_score=self.self_consistency_score,
            version=self.version + 1,
            last_updated=datetime.utcnow().isoformat()
        )
    
    @staticmethod
    def _compute_consistency(capabilities, limitations) -> float:
        """Compute self-consistency score"""
        # Check if any capability conflicts with any limitation
        conflicts = 0
        
        for cap_id, cap in capabilities.items():
            for lim_id, lim in limitations.items():
                if cap_id in lim.affected_capabilities:
                    conflicts += 1
        
        if len(capabilities) == 0:
            return 0.5
        
        consistency = 1.0 - (conflicts / (len(capabilities) + len(limitations)))
        return max(0.0, min(1.0, consistency))


def create_initial_self_model() -> SelfModel:
    """Create initial self model with default capabilities"""
    default_caps = {
        "reasoning": Capability(
            capability_id="reasoning",
            name="Logical Reasoning",
            description="Analyze and deduce conclusions",
            proficiency=0.7,
            reliability=0.8,
            learn_speed=0.5,
            resource_requirement=0.3,
            success_count=100,
            failure_count=10,
            last_used=datetime.utcnow().isoformat()
        ),
        "planning": Capability(
            capability_id="planning",
            name="Strategic Planning",
            description="Create and execute plans",
            proficiency=0.6,
            reliability=0.7,
            learn_speed=0.4,
            resource_requirement=0.4,
            success_count=50,
            failure_count=15,
            last_used=datetime.utcnow().isoformat()
        ),
        "execution": Capability(
            capability_id="execution",
            name="Task Execution",
            description="Execute actions to achieve goals",
            proficiency=0.75,
            reliability=0.85,
            learn_speed=0.6,
            resource_requirement=0.5,
            success_count=200,
            failure_count=20,
            last_used=datetime.utcnow().isoformat()
        )
    }
    
    return SelfModel(
        capabilities=MappingProxyType(default_caps),
        limitations=MappingProxyType({}),
        active_strategies=MappingProxyType({}),
        recent_failures=(),
        confidence_map=MappingProxyType({}),
        resource_model=ResourceModel(
            attention_budget=1.0,
            memory_capacity=0.3,
            processing_capacity=0.5,
            energy_level=0.9,
            time_available=24.0
        ),
        self_consistency_score=0.9,
        version=0,
        last_updated=datetime.utcnow().isoformat()
    )