"""
World Model - Causal Intervention System

Phase transition: correlation-based → causal intervention

Core principle:
- OBSERVATION actions do NOT change the world
- INTERVENTION actions DO change the world
- Only interventions produce causal learning

Key architecture:
1. Deterministic state encoding (not random hash)
2. Action taxonomy (observe vs intervene vs reflect)
3. World delta engine (actual changes, not health scores)
4. Causal transition model (not correlation)
5. Goal progress tracking (not success labels)
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json


class ActionCategory(Enum):
    """Taxonomy of actions by their effect on world"""
    OBSERVE = "observe"           # Read-only, no side effects
    INTERVENE = "intervene"        # Changes world state
    PLAN = "plan"                 # Internal cognitive action
    REFLECT = "reflect"           # Meta-cognitive action
    COMMUNICATE = "communicate"   # External communication


@dataclass(frozen=True)
class WorldDelta:
    """
    Actual changes in the world.

    NOT just health score change.
    Specific measurable changes:

    - containers_added
    - containers_removed
    - errors_resolved
    - errors_introduced
    - resource_change (cpu, memory, disk)
    - latency_change
    - goal_progress
    """
    delta_id: str
    timestamp: str

    # Container changes
    containers_added: Tuple[str, ...] = ()
    containers_removed: Tuple[str, ...] = ()
    containers_healthy: int = 0
    containers_unhealthy: int = 0

    # Error changes
    errors_resolved: int = 0
    errors_introduced: int = 0
    error_messages: Tuple[str, ...] = ()

    # Resource changes
    cpu_delta: float = 0.0
    memory_delta_mb: int = 0
    disk_delta_mb: int = 0

    # Latency
    latency_delta_ms: float = 0.0

    # Goal progress
    goal_progress: float = 0.0  # -1 to 1

    # Computed scores
    net_change: float = 0.0  # Computed overall change
    is_improvement: bool = False

    @staticmethod
    def from_transition(
        before: Dict[str, Any],
        after: Dict[str, Any],
        action_type: str,
        goal_progress: float = 0.0
    ) -> 'WorldDelta':
        """Create delta from before/after state observations"""
        delta_id = hashlib.md5(
            f"{json.dumps(before, sort_keys=True)}{action_type}{json.dumps(after, sort_keys=True)}".encode()
        ).hexdigest()[:12]

        # Container changes
        containers_before = set(before.get('container_status', {}).keys())
        containers_after = set(after.get('container_status', {}).keys())

        added = tuple(containers_after - containers_before)
        removed = tuple(containers_before - containers_after)

        # Count healthy/unhealthy after
        healthy = sum(1 for s in after.get('container_status', {}).values() if 'running' in s.lower())
        unhealthy = len(containers_after) - healthy

        # Error changes
        errors_before = len(before.get('error_logs', []))
        errors_after = len(after.get('error_logs', []))
        errors_resolved = max(0, errors_before - errors_after)
        errors_introduced = max(0, errors_after - errors_before)

        # Resource changes
        res_before = before.get('resource_state', {})
        res_after = after.get('resource_state', {})

        cpu_delta = res_after.get('load', 0) - res_before.get('load', 0)
        mem_delta = res_after.get('memory_mb', 0) - res_before.get('memory_mb', 0)
        disk_delta = res_after.get('disk_mb', 0) - res_before.get('disk_mb', 0)

        # Compute net change
        net = 0.0
        if errors_resolved > 0:
            net += errors_resolved * 0.2
        if errors_introduced > 0:
            net -= errors_introduced * 0.2
        if healthy > unhealthy:
            net += 0.1
        if healthy < unhealthy:
            net -= 0.1
        net += goal_progress

        return WorldDelta(
            delta_id=delta_id,
            timestamp="",  # Set by caller
            containers_added=added,
            containers_removed=removed,
            containers_healthy=healthy,
            containers_unhealthy=unhealthy,
            errors_resolved=errors_resolved,
            errors_introduced=errors_introduced,
            error_messages=tuple(after.get('error_logs', [])),
            cpu_delta=cpu_delta,
            memory_delta_mb=mem_delta,
            disk_delta_mb=disk_delta,
            goal_progress=goal_progress,
            net_change=net,
            is_improvement=net > 0.05
        )


@dataclass(frozen=True)
class DeterministicState:
    """
    Deterministic state encoding.

    Key: state signature is based on FEATURES, not timestamp.

    Same features = same signature.

    This enables:
    - True semantic similarity
    - Episode clustering
    - Causal pattern extraction
    - Memory retrieval
    """
    signature: str  # Deterministic hash of features
    features: Tuple[Tuple[str, Any], ...]  # Feature vector

    # Separated feature spaces
    infrastructure_features: Tuple[Tuple[str, Any], ...]  # containers, resources
    cognitive_features: Tuple[Tuple[str, Any], ...]  # memory load, heuristics
    task_features: Tuple[Tuple[str, Any], ...]  # pending goals, progress
    error_features: Tuple[Tuple[str, Any], ...]  # error patterns

    health_score: float  # 0-1 computed from features

    @staticmethod
    def from_observation(obs: Dict[str, Any]) -> 'DeterministicState':
        """Create deterministic state from observation"""

        # Infrastructure features
        infra = (
            ('container_count', len(obs.get('container_status', {}))),
            ('containers', tuple(sorted(obs.get('container_status', {}).keys()))),
            ('resource_load', obs.get('resource_state', {}).get('load', 0)),
            ('resource_memory', obs.get('resource_state', {}).get('memory_mb', 0)),
            ('resource_disk', obs.get('resource_state', {}).get('disk_mb', 0)),
        )

        # Task features
        task = (
            ('has_errors', len(obs.get('error_logs', [])) > 0),
            ('error_count', len(obs.get('error_logs', []))),
            ('pending_goals', 0),  # Set by system
            ('goal_progress', 0.0),  # Set by system
        )

        # Error features
        errors = obs.get('error_logs', [])
        error_feats = (
            ('has_critical_errors', any('CRITICAL' in e or 'FATAL' in e for e in errors)),
            ('has_warnings', any('WARNING' in e for e in errors)),
            ('error_pattern', _extract_error_pattern(errors)),
        )

        # Compute health score from features
        health = 1.0
        error_count = len(errors)
        load = obs.get('resource_state', {}).get('load', 0)
        memory = obs.get('resource_state', {}).get('memory_mb', 0)

        if error_count > 0:
            health -= min(0.4, error_count * 0.08)
        if load > 2.0:
            health -= min(0.2, (load - 2.0) * 0.1)
        if memory < 500:
            health -= 0.15

        health = max(0.0, min(1.0, health))

        # Compute deterministic signature
        feature_str = json.dumps({
            'infra': dict(infra),
            'task': dict(task),
            'errors': dict(error_feats),
            'health': health
        }, sort_keys=True)
        signature = hashlib.sha256(feature_str.encode()).hexdigest()[:16]

        return DeterministicState(
            signature=signature,
            features=frozenset((str(k), str(v)) for k, v in infra),
            infrastructure_features=infra,
            cognitive_features=task,
            task_features=task,
            error_features=error_feats,
            health_score=health
        )

    def similarity(self, other: 'DeterministicState') -> float:
        """Semantic similarity based on features, not identity"""
        if not isinstance(other, DeterministicState):
            return 0.0

        # Compare infrastructure state (most important)
        infra_sim = self._feature_similarity(
            self.infrastructure_features,
            other.infrastructure_features
        )

        # Compare error state
        error_sim = self._feature_similarity(
            self.error_features,
            other.error_features
        )

        # Weight: infrastructure 60%, errors 40%
        return (infra_sim * 0.6) + (error_sim * 0.4)

    def _feature_similarity(
        self,
        features_a: Tuple[Tuple[str, Any], ...],
        features_b: Tuple[Tuple[str, Any], ...]
    ) -> float:
        """Compare feature tuples"""
        dict_a = dict(features_a)
        dict_b = dict(features_b)
        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        if not all_keys:
            return 1.0

        matches = 0
        for key in all_keys:
            val_a = dict_a.get(key)
            val_b = dict_b.get(key)

            if val_a == val_b:
                matches += 1
            elif isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                # For numeric values, compute similarity
                max_val = max(abs(val_a), abs(val_b), 1)
                matches += 1 - (abs(val_a - val_b) / max_val)
            else:
                matches += 0.5  # Partial match for different types

        return matches / len(all_keys)


def _extract_error_pattern(errors: List[str]) -> str:
    """Extract error pattern from error messages"""
    if not errors:
        return "none"

    patterns = set()
    for error in errors:
        error_lower = error.lower()
        if 'connection' in error_lower:
            patterns.add('connection')
        if 'memory' in error_lower or 'oom' in error_lower:
            patterns.add('memory')
        if 'timeout' in error_lower:
            patterns.add('timeout')
        if 'permission' in error_lower:
            patterns.add('permission')
        if 'not found' in error_lower:
            patterns.add('not_found')

    return ','.join(sorted(patterns)) if patterns else 'generic'


@dataclass(frozen=True)
class CausalIntervention:
    """
    A causal intervention: action that changes world state.

    Key: ONLY interventions produce causal learning.
    Observations do NOT cause world changes.
    """
    intervention_id: str
    timestamp: str
    action_type: str
    action_category: ActionCategory
    parameters: Tuple[Tuple[str, Any], ...]

    # Causal context
    before_state: DeterministicState
    world_delta: WorldDelta

    # Evaluation
    causal_effect: float  # How much this action caused the delta
    utility: float  # Goal-aligned value (-1 to 1)

    @staticmethod
    def create(
        action_type: str,
        category: ActionCategory,
        parameters: Dict[str, Any],
        before: DeterministicState,
        after: Dict[str, Any],
        goal_progress: float = 0.0
    ) -> 'CausalIntervention':
        """Create causal intervention from execution"""

        intervention_id = hashlib.md5(
            f"{before.signature}{action_type}{json.dumps(parameters, sort_keys=True)}".encode()
        ).hexdigest()[:12]

        # Only compute delta for interventions
        if category == ActionCategory.INTERVENE:
            after_state = DeterministicState.from_observation(after)
            delta = WorldDelta.from_transition(
                before={'container_status': {}, 'resource_state': {}, 'error_logs': []},  # placeholder
                after=after,
                action_type=action_type,
                goal_progress=goal_progress
            )

            # Causal effect: how much did state change
            state_change = abs(after_state.health_score - before.health_score)
            causal_effect = min(1.0, state_change * 2)
        else:
            # No causal effect for observations
            delta = WorldDelta(
                delta_id="no_delta",
                timestamp="",
                net_change=0.0,
                is_improvement=False
            )
            causal_effect = 0.0

        # Utility = goal progress + delta improvement
        utility = goal_progress + (delta.net_change * 0.5)

        return CausalIntervention(
            intervention_id=intervention_id,
            timestamp="",
            action_type=action_type,
            action_category=category,
            parameters=frozenset((str(k), str(v)) for k, v in parameters.items()),
            before_state=before,
            world_delta=delta,
            causal_effect=causal_effect,
            utility=max(-1.0, min(1.0, utility))
        )

    def is_causal(self) -> bool:
        """Does this intervention actually change the world?"""
        return (
            self.action_category == ActionCategory.INTERVENE and
            self.causal_effect > 0.1
        )

    def is_positive(self) -> bool:
        """Did this intervention improve the world?"""
        return self.utility > 0.1


@dataclass
class WorldModel:
    """
    Causal world model: P(next_state | state, action)

    Enables:
    - Predicting outcomes before execution
    - Causal reasoning
    - Intervention planning
    - Counterfactual reasoning
    """
    interventions: List[CausalIntervention] = field(default_factory=list)

    # Indexed by state signature for fast lookup
    state_index: Dict[str, List[int]] = field(default_factory=dict)  # signature → indices
    action_index: Dict[str, List[int]] = field(default_factory=dict)  # action_type → indices

    def add_intervention(self, intervention: CausalIntervention):
        """Add causal intervention to model"""
        self.interventions.append(intervention)

        # Index by state
        sig = intervention.before_state.signature
        if sig not in self.state_index:
            self.state_index[sig] = []
        self.state_index[sig].append(len(self.interventions) - 1)

        # Index by action
        if intervention.action_type not in self.action_index:
            self.action_index[intervention.action_type] = []
        self.action_index[intervention.action_type].append(len(self.interventions) - 1)

    def predict_outcome(
        self,
        current_state: DeterministicState,
        action_type: str,
        category: ActionCategory
    ) -> Tuple[Optional[WorldDelta], float]:
        """
        Predict outcome using causal model.

        Returns:
        - predicted delta (or None for observations)
        - confidence (0-1)
        """
        # Observations have no effect
        if category == ActionCategory.OBSERVE:
            return None, 1.0

        # Look up similar states
        similar_states = self._find_similar_states(current_state)
        similar_actions = self.state_index.get(action_type, [])

        # Find interventions with both similar state and same action
        candidates = []
        for idx in similar_actions:
            if idx < len(self.interventions):
                iv = self.interventions[idx]
                if iv.action_type == action_type:
                    sim = current_state.similarity(iv.before_state)
                    if sim > 0.5:
                        candidates.append((iv, sim))

        if not candidates:
            return None, 0.3  # No experience = low confidence

        # Average predicted outcome weighted by similarity
        total_weight = 0.0
        weighted_delta = WorldDelta(
            delta_id="predicted",
            timestamp="",
            net_change=0.0,
            is_improvement=False
        )

        for iv, sim in candidates:
            weight = sim * iv.causal_effect
            total_weight += weight

        confidence = min(1.0, total_weight)

        # Use best candidate for prediction
        best = max(candidates, key=lambda x: x[1])
        return best[0].world_delta, confidence

    def _find_similar_states(self, state: DeterministicState) -> List[str]:
        """Find similar state signatures"""
        similar = []
        for sig in self.state_index.keys():
            # Create temporary state for comparison
            temp_state = DeterministicState(
                signature=sig,
                features=(),
                infrastructure_features=(),
                cognitive_features=(),
                task_features=(),
                error_features=(),
                health_score=0.5
            )
            if temp_state.similarity(state) > 0.7:
                similar.append(sig)
        return similar

    def get_best_intervention_for_goal(
        self,
        current_state: DeterministicState,
        goal_type: str
    ) -> Tuple[Optional[str], float]:
        """
        Get best intervention type for goal based on causal history.

        Returns:
        - best action type
        - expected utility
        """
        interventions = [
            iv for iv in self.interventions
            if iv.action_category == ActionCategory.INTERVENE
        ]

        if not interventions:
            return None, 0.0

        # Group by action type
        action_utility: Dict[str, List[float]] = {}
        for iv in interventions:
            if iv.action_type not in action_utility:
                action_utility[iv.action_type] = []
            action_utility[iv.action_type].append(iv.utility)

        # Compute average utility per action
        best_action = None
        best_utility = 0.0

        for action, utilities in action_utility.items():
            avg = sum(utilities) / len(utilities)
            if avg > best_utility:
                best_utility = avg
                best_action = action

        return best_action, best_utility

    def extract_causal_rules(self) -> List[str]:
        """Extract causal rules from interventions"""
        rules = []

        # Group by action type
        action_effects: Dict[str, List[CausalIntervention]] = {}
        for iv in self.interventions:
            if iv.action_category == ActionCategory.INTERVENE:
                if iv.action_type not in action_effects:
                    action_effects[iv.action_type] = []
                action_effects[iv.action_type].append(iv)

        # Extract rules
        for action, interventions in action_effects.items():
            if len(interventions) < 2:
                continue

            successes = sum(1 for iv in interventions if iv.is_positive())
            rate = successes / len(interventions)

            if rate > 0.8:
                avg_effect = sum(iv.causal_effect for iv in interventions) / len(interventions)
                rules.append(
                    f"CAUSE {action} → improvement (conf={rate:.0%}, effect={avg_effect:.2f})"
                )
            elif rate < 0.3:
                rules.append(
                    f"AVOID {action} → mostly failures (conf={rate:.0%})"
                )

        return rules


def get_action_category(action_type: str) -> ActionCategory:
    """Determine action category from action type"""
    # Observation actions - read-only
    observe_actions = {
        'check_system_status', 'read_errors', 'list_services',
        'check_container', 'test_api', 'run_diagnostics'
    }

    # Intervention actions - change world
    intervene_actions = {
        'restart_container', 'create_container', 'delete_container',
        'deploy', 'scale', 'update_config', 'clear_cache'
    }

    if action_type in observe_actions:
        return ActionCategory.OBSERVE
    elif action_type in intervene_actions:
        return ActionCategory.INTERVENE
    else:
        return ActionCategory.OBSERVE  # Default to observe for safety