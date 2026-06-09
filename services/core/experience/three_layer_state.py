"""
Three-Layer State Model

Architecture:
1. Environment State - Physical world (what IS)
2. Belief State - Agent's beliefs (what agent THINKS)
3. Cognitive State - Internal cognition (attention, pressure, goals)

Key insight from critique:
- OBSERVE doesn't change world, but DOES change beliefs
- Belief revision is a first-class cognitive operation
- Utility scalar causes reward shaping - need counterfactuals
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime


@dataclass(frozen=True)
class EnvironmentState:
    """
    Layer 1: Physical/External World State

    What actually exists in the world:
    - Containers running/not running
    - Files existing
    - Network connections
    - Resource utilization
    - Error logs
    """
    timestamp: str
    containers: Tuple[str, ...]
    container_status: Tuple[Tuple[str, str], ...]  # name -> status
    resource_load: float
    resource_memory_mb: int
    resource_disk_mb: int
    error_logs: Tuple[str, ...]
    error_count: int

    @staticmethod
    def from_observation(obs: Dict[str, Any]) -> 'EnvironmentState':
        """Create from sensor observation"""
        containers = tuple(obs.get('container_status', {}).keys())
        status = tuple((k, v) for k, v in obs.get('container_status', {}).items())
        errors = tuple(obs.get('error_logs', []))
        res = obs.get('resource_state', {})

        return EnvironmentState(
            timestamp=obs.get('timestamp', datetime.now().isoformat()),
            containers=containers,
            container_status=status,
            resource_load=res.get('load', 0),
            resource_memory_mb=res.get('memory_mb', 0),
            resource_disk_mb=res.get('disk_mb', 0),
            error_logs=errors,
            error_count=len(errors)
        )

    def signature(self) -> str:
        """Deterministic signature based on features"""
        data = {
            'containers': sorted(self.containers),
            'status': dict(self.container_status),
            'load': round(self.resource_load, 1),
            'memory': self.resource_memory_mb,
            'errors': self.error_count
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class BeliefState:
    """
    Layer 2: Agent's Belief State

    What the agent BELIEVES about the world.
    Updated by OBSERVATIONS (not interventions).

    Key difference from environment state:
    - Agent may be uncertain
    - Agent may have misconceptions
    - Beliefs can be revised based on evidence
    - Contains hypotheses about causes
    """
    timestamp: str

    # Confidence levels (0-1)
    infrastructure_confidence: float  # How confident about infra state
    health_confidence: float        # How confident about health
    causality_confidence: float      # How confident about causal relationships

    # Hypotheses about causes
    active_hypotheses: Tuple[str, ...]  # "db_latency_causing_auth_failures"
    resolved_hypotheses: Tuple[str, ...]  # Proven/disproven

    # Uncertainty regions
    uncertain_areas: Tuple[str, ...]  # "auth_service_stability"
    resolved_uncertainty: Tuple[str, ...]  # Areas where uncertainty resolved

    # Beliefs about causality
    causal_beliefs: Tuple[Tuple[str, str, float], ...]  # (cause, effect, confidence)

    # Anomaly detection
    detected_anomalies: Tuple[str, ...]  # Unexpected patterns
    anomaly_confidence: float  # How certain anomalies are real

    @staticmethod
    def initial() -> 'BeliefState':
        """Create initial belief state with high uncertainty"""
        return BeliefState(
            timestamp=datetime.now().isoformat(),
            infrastructure_confidence=0.5,
            health_confidence=0.5,
            causality_confidence=0.3,
            active_hypotheses=(),
            resolved_hypotheses=(),
            uncertain_areas=('infrastructure', 'causality'),
            resolved_uncertainty=(),
            causal_beliefs=(),
            detected_anomalies=(),
            anomaly_confidence=0.5
        )

    def with_observation(
        self,
        env_state: EnvironmentState,
        previous_error_count: int = 0,
        previous_container_count: int = -1
    ) -> 'BeliefState':
        """
        Revise beliefs based on new observation.

        OBSERVE changes THIS state, not environment.
        This is the key insight!
        """
        new_timestamp = env_state.timestamp

        # Confidence updates based on observation
        new_infra_conf = min(1.0, self.infrastructure_confidence + 0.1)
        new_health_conf = min(1.0, self.health_confidence + 0.1)

        # Detect anomalies (unexpected patterns)
        new_anomalies = list(self.detected_anomalies)
        new_anomaly_conf = self.anomaly_confidence

        if env_state.error_count > 5:
            if 'high_error_rate' not in new_anomalies:
                new_anomalies.append('high_error_rate')
                new_anomaly_conf = 0.7
        elif env_state.error_count == 0 and previous_error_count > 0:
            # Errors resolved - reduce anomaly
            new_anomaly_conf = max(0.0, new_anomaly_conf - 0.2)

        # Identify uncertainty that was resolved
        new_resolved = list(self.resolved_uncertainty)
        if env_state.error_count < previous_error_count:
            if 'error_state' in self.uncertain_areas:
                new_resolved.append('error_state')

        # Generate new hypotheses based on observation
        new_hypotheses = list(self.active_hypotheses)

        if env_state.error_count > 3 and not any('error' in h for h in new_hypotheses):
            new_hypotheses.append('errors_indicating_degradation')

        current_container_count = len(env_state.containers)
        if previous_container_count >= 0 and current_container_count < previous_container_count:
            if 'container_loss' not in new_hypotheses:
                new_hypotheses.append('containers_stopped_unexpectedly')

        return BeliefState(
            timestamp=new_timestamp,
            infrastructure_confidence=new_infra_conf,
            health_confidence=new_health_conf,
            causality_confidence=self.causality_confidence,
            active_hypotheses=tuple(new_hypotheses),
            resolved_hypotheses=tuple(new_resolved),
            uncertain_areas=self.uncertain_areas,
            resolved_uncertainty=tuple(new_resolved),
            causal_beliefs=self.causal_beliefs,
            detected_anomalies=tuple(new_anomalies),
            anomaly_confidence=new_anomaly_conf
        )

    def with_belief_revision(self, revision: Dict[str, Any]) -> 'BeliefState':
        """
        Revise beliefs based on cognitive operation.
        """
        return BeliefState(
            timestamp=datetime.now().isoformat(),
            infrastructure_confidence=revision.get('infra_conf', self.infrastructure_confidence),
            health_confidence=revision.get('health_conf', self.health_confidence),
            causality_confidence=revision.get('causality_conf', self.causality_confidence),
            active_hypotheses=revision.get('hypotheses', self.active_hypotheses),
            resolved_hypotheses=revision.get('resolved', self.resolved_hypotheses),
            uncertain_areas=revision.get('uncertain', self.uncertain_areas),
            resolved_uncertainty=revision.get('resolved', self.resolved_uncertainty),
            causal_beliefs=revision.get('causal', self.causal_beliefs),
            detected_anomalies=revision.get('anomalies', self.detected_anomalies),
            anomaly_confidence=revision.get('anomaly_conf', self.anomaly_confidence)
        )


@dataclass(frozen=True)
class CognitiveState:
    """
    Layer 3: Internal Cognitive State

    The agent's internal processing state:
    - Attention allocation
    - Working memory
    - Active goals
    - Pressure/stress
    - Processing capacity
    """
    timestamp: str

    # Attention
    attention_budget: float  # 0-1, available attention
    attention_allocated: float  # Currently allocated
    focus_target: Optional[str]  # What is being focused on

    # Working memory
    working_memory_items: Tuple[str, ...]  # Items in focus
    memory_load: float  # 0-1, how loaded

    # Goals
    active_goal_count: int
    goal_progress: float  # -1 to 1

    # Pressure states
    cognitive_pressure: float  # 0-1
    uncertainty_pressure: float  # 0-1
    resource_pressure: float  # 0-1

    # Meta-cognition
    reflection_depth: int  # How deep is current reasoning
    strategy_version: int  # Current strategy version

    @staticmethod
    def initial() -> 'CognitiveState':
        return CognitiveState(
            timestamp=datetime.now().isoformat(),
            attention_budget=1.0,
            attention_allocated=0.0,
            focus_target=None,
            working_memory_items=(),
            memory_load=0.3,
            active_goal_count=0,
            goal_progress=0.0,
            cognitive_pressure=0.3,
            uncertainty_pressure=0.2,
            resource_pressure=0.2,
            reflection_depth=0,
            strategy_version=1
        )

    def with_goal_update(self, goal_added: bool, progress: float) -> 'CognitiveState':
        """Update cognitive state based on goals"""
        new_goal_count = self.active_goal_count + (1 if goal_added else 0)
        new_pressure = min(1.0, self.cognitive_pressure + (0.1 if goal_added else -0.05))

        return CognitiveState(
            timestamp=datetime.now().isoformat(),
            attention_budget=self.attention_budget,
            attention_allocated=self.attention_allocated,
            focus_target=self.focus_target,
            working_memory_items=self.working_memory_items,
            memory_load=self.memory_load,
            active_goal_count=new_goal_count,
            goal_progress=progress,
            cognitive_pressure=new_pressure,
            uncertainty_pressure=self.uncertainty_pressure,
            resource_pressure=self.resource_pressure,
            reflection_depth=self.reflection_depth,
            strategy_version=self.strategy_version
        )

    def with_attention_shift(self, target: str, allocation: float) -> 'CognitiveState':
        """Shift attention to new target"""
        return CognitiveState(
            timestamp=datetime.now().isoformat(),
            attention_budget=self.attention_budget,
            attention_allocated=allocation,
            focus_target=target,
            working_memory_items=self.working_memory_items,
            memory_load=self.memory_load,
            active_goal_count=self.active_goal_count,
            goal_progress=self.goal_progress,
            cognitive_pressure=self.cognitive_pressure,
            uncertainty_pressure=self.uncertainty_pressure,
            resource_pressure=self.resource_pressure,
            reflection_depth=self.reflection_depth,
            strategy_version=self.strategy_version
        )


@dataclass
class TripleLayerState:
    """
    Combined three-layer state model.

    This is what gets updated during cognition:
    - Environment state: updated ONLY by interventions
    - Belief state: updated by observations AND cognitive operations
    - Cognitive state: updated by internal processing
    """
    environment: EnvironmentState
    belief: BeliefState
    cognitive: CognitiveState
    version: int = 0

    @staticmethod
    def initial() -> 'TripleLayerState':
        return TripleLayerState(
            environment=EnvironmentState(
                timestamp=datetime.now().isoformat(),
                containers=(),
                container_status=(),
                resource_load=0.0,
                resource_memory_mb=0,
                resource_disk_mb=0,
                error_logs=(),
                error_count=0
            ),
            belief=BeliefState.initial(),
            cognitive=CognitiveState.initial(),
            version=0
        )

    def with_environment_update(self, env: EnvironmentState) -> 'TripleLayerState':
        """Update environment ONLY (from intervention results)"""
        return TripleLayerState(
            environment=env,
            belief=self.belief,  # Beliefs updated by obs, not env
            cognitive=self.cognitive,
            version=self.version + 1
        )

    def with_belief_revision(self, belief: BeliefState) -> 'TripleLayerState':
        """Update beliefs (from observation or cognition)"""
        return TripleLayerState(
            environment=self.environment,
            belief=belief,
            cognitive=self.cognitive,
            version=self.version + 1
        )

    def with_cognitive_update(self, cognitive: CognitiveState) -> 'TripleLayerState':
        """Update cognitive state"""
        return TripleLayerState(
            environment=self.environment,
            belief=self.belief,
            cognitive=cognitive,
            version=self.version + 1
        )


@dataclass(frozen=True)
class CounterfactualOutcome:
    """
    Counterfactual outcome model.

    Instead of scalar utility, we model:
    - What would happen WITHOUT this intervention
    - What happened WITH this intervention
    - The causal difference

    This is NOT reward shaping - it's causal reasoning.
    """
    outcome_id: str
    intervention: str
    timestamp: str

    # World state change
    world_before: str  # Environment signature
    world_after: str   # Environment signature
    world_delta: str   # What changed

    # Counterfactual comparison
    without_intervention: str  # What likely would have happened
    with_intervention: str     # What actually happened
    causal_effect: str          # Description of causal effect

    # Causal confidence
    temporal_consistency: float  # 0-1, effects consistent over time
    confound_score: float        # 0-1, how likely confounders exist
    delayed_effect_score: float  # 0-1, delayed vs immediate

    @staticmethod
    def from_intervention(
        intervention_type: str,
        before_state: EnvironmentState,
        after_state: EnvironmentState,
        belief_state: BeliefState
    ) -> 'CounterfactualOutcome':
        """Create counterfactual from intervention"""

        outcome_id = hashlib.md5(
            f"{before_state.signature()}{intervention_type}{after_state.signature()}".encode()
        ).hexdigest()[:12]

        # Determine causal effect description
        delta_errors = after_state.error_count - before_state.error_count
        delta_containers = len(after_state.containers) - len(before_state.containers)

        if delta_errors < 0 and delta_containers == 0:
            effect = "errors_resolved_without_service_change"
        elif delta_containers < 0:
            effect = "service_degraded_during_intervention"
        elif delta_errors > 0:
            effect = "errors_introduced_during_intervention"
        else:
            effect = "no_measurable_effect"

        # Counterfactual: what would likely happen without intervention?
        if effect.startswith("errors_resolved"):
            without = "errors_would_persist_or_worsen"
        elif effect.startswith("service_degraded"):
            without = "service_would_continue"
        else:
            without = "state_would_remain_stable"

        return CounterfactualOutcome(
            outcome_id=outcome_id,
            intervention=intervention_type,
            timestamp=datetime.now().isoformat(),
            world_before=before_state.signature(),
            world_after=after_state.signature(),
            world_delta=effect,
            without_intervention=without,
            with_intervention=effect,
            causal_effect=f"{intervention_type} → {effect}",
            temporal_consistency=0.8,  # Placeholder
            confound_score=0.5,  # Placeholder
            delayed_effect_score=0.3  # Placeholder
        )


@dataclass
class CausalGraph:
    """
    Causal intervention graph.

    Tracks:
    - Action chains (A → B → C)
    - Dependency chains (A enables B)
    - Rollback chains (if A fails, rollback B)
    """
    nodes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)  # action -> [affected_actions]

    def add_intervention(
        self,
        intervention_type: str,
        outcome: CounterfactualOutcome,
        chain_id: Optional[str] = None
    ):
        """Add intervention to causal graph"""
        if intervention_type not in self.nodes:
            self.nodes[intervention_type] = {
                'occurrences': 0,
                'chain_id': chain_id,
                'outcomes': []
            }

        self.nodes[intervention_type]['occurrences'] += 1
        self.nodes[intervention_type]['outcomes'].append(outcome.outcome_id)

    def get_causal_chain(self, intervention_type: str) -> List[str]:
        """Get chain of effects from intervention"""
        if intervention_type not in self.edges:
            return [intervention_type]

        chain = [intervention_type]
        for effect in self.edges.get(intervention_type, []):
            chain.extend(self.get_causal_chain(effect))

        return chain

    def get_intervention_effectiveness(
        self,
        intervention_type: str
    ) -> Tuple[float, int]:
        """Get effectiveness score and occurrence count"""
        if intervention_type not in self.nodes:
            return 0.0, 0

        node = self.nodes[intervention_type]
        occurrences = node['occurrences']

        # Compute effectiveness from outcomes
        positive_count = 0
        for outcome_id in node['outcomes']:
            # Simplified: check if effect was positive
            if 'resolved' in outcome_id or 'improved' in outcome_id:
                positive_count += 1

        effectiveness = positive_count / occurrences if occurrences > 0 else 0.0
        return effectiveness, occurrences