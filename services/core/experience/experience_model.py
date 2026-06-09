"""
Experience Model - Semantic Episodic Memory

From reactive → deliberative cognition.

Structure:
state → intention → action → environment transition → outcome → evaluation

This is NOT just lineage log.
This is semantic memory that enables:
- Retrieve similar situations
- Compare outcomes
- Avoid repetition
- Form heuristics
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, UTC
from enum import Enum
import hashlib
import json


class OutcomeType(Enum):
    """How the action affected the world"""
    IMPROVEMENT = "improvement"       # State got better
    DEGRADATION = "degradation"       # State got worse
    NO_CHANGE = "no_change"           # No observable effect
    UNEXPECTED = "unexpected"         # Unpredicted outcome


@dataclass(frozen=True)
class EnvironmentState:
    """
    Snapshot of environment state.

    Used for:
    - before/after comparison
    - similarity detection
    - pattern extraction
    """
    state_id: str
    timestamp: str
    containers: Tuple[str, ...]  # Running containers
    error_count: int
    load_average: float
    memory_mb: int
    disk_mb: int
    active_processes: Tuple[str, ...]  # Key processes
    health_score: float  # 0-1, computed health

    @staticmethod
    def from_observation(obs: Dict[str, Any]) -> 'EnvironmentState':
        """Create from sensor observation"""
        containers = tuple(obs.get('container_status', {}).keys())
        processes = tuple(obs.get('processes', [])[:5])

        load = obs.get('resource_state', {}).get('load', 0)
        memory = obs.get('resource_state', {}).get('memory_mb', 0)
        disk = obs.get('resource_state', {}).get('disk_mb', 0)
        errors = len(obs.get('error_logs', []))

        # Compute health score
        health = 1.0
        if errors > 0:
            health -= min(0.3, errors * 0.05)
        if load > 2.0:
            health -= min(0.2, (load - 2.0) * 0.1)
        if memory < 500:
            health -= 0.1

        state_id = hashlib.md5(
            f"{datetime.now(UTC).isoformat()}{len(containers)}".encode()
        ).hexdigest()[:12]

        return EnvironmentState(
            state_id=state_id,
            timestamp=datetime.now(UTC).isoformat(),
            containers=containers,
            error_count=errors,
            load_average=load,
            memory_mb=memory,
            disk_mb=disk,
            active_processes=processes,
            health_score=max(0.0, health)
        )

    def similarity(self, other: 'EnvironmentState') -> float:
        """Compute similarity to another state (0-1)"""
        if not isinstance(other, EnvironmentState):
            return 0.0

        similarity = 0.0
        total = 5.0

        # Container similarity
        common = len(set(self.containers) & set(other.containers))
        max_containers = max(len(self.containers), len(other.containers), 1)
        similarity += (common / max_containers)

        # Error count similarity
        max_errors = max(self.error_count, other.error_count, 1)
        error_sim = 1.0 - (abs(self.error_count - other.error_count) / max_errors)
        similarity += error_sim

        # Load similarity
        max_load = max(self.load_average, other.load_average, 1.0)
        load_sim = 1.0 - (abs(self.load_average - other.load_average) / max_load)
        similarity += load_sim

        # Memory similarity
        max_mem = max(self.memory_mb, other.memory_mb, 1)
        mem_sim = 1.0 - (abs(self.memory_mb - other.memory_mb) / max_mem)
        similarity += mem_sim

        # Health similarity (most important)
        health_sim = 1.0 - abs(self.health_score - other.health_score)
        similarity += health_sim

        return similarity / total


@dataclass(frozen=True)
class StateTransition:
    """
    Represents a change from one state to another.

    Key for causal understanding:
    - What caused the change?
    - Was it intended?
    - Is it reversible?
    """
    transition_id: str
    timestamp: str
    before_state: EnvironmentState
    action_type: str
    action_parameters: Tuple[Tuple[str, str], ...]  # Frozen key-value
    after_state: EnvironmentState
    outcome_type: OutcomeType
    improvement_score: float  # How much better (+1) or worse (-1)
    causal_confidence: float  # How confident we are this action caused change

    @staticmethod
    def create(
        before: EnvironmentState,
        action_type: str,
        action_params: Dict[str, Any],
        after: EnvironmentState
    ) -> 'StateTransition':
        """Create a new state transition"""
        transition_id = hashlib.md5(
            f"{before.state_id}{action_type}{after.state_id}".encode()
        ).hexdigest()[:12]

        # Determine outcome type
        health_delta = after.health_score - before.health_score
        error_delta = after.error_count - before.error_count

        if health_delta > 0.1:
            outcome = OutcomeType.IMPROVEMENT
            improvement = health_delta
        elif health_delta < -0.1:
            outcome = OutcomeType.DEGRADATION
            improvement = health_delta
        elif error_delta != 0:
            outcome = OutcomeType.UNEXPECTED
            improvement = -error_delta * 0.1
        else:
            outcome = OutcomeType.NO_CHANGE
            improvement = 0.0

        # Causal confidence: higher if state changed significantly
        state_change = abs(health_delta) + abs(error_delta) * 0.1
        causal_conf = min(1.0, state_change * 2)

        return StateTransition(
            transition_id=transition_id,
            timestamp=datetime.now(UTC).isoformat(),
            before_state=before,
            action_type=action_type,
            action_parameters=frozenset((str(k), str(v)) for k, v in action_params.items()),
            after_state=after,
            outcome_type=outcome,
            improvement_score=improvement,
            causal_confidence=causal_conf
        )

    def is_positive(self) -> bool:
        return self.outcome_type == OutcomeType.IMPROVEMENT and self.improvement_score > 0

    def is_negative(self) -> bool:
        return self.outcome_type == OutcomeType.DEGRADATION or self.improvement_score < -0.1


@dataclass(frozen=True)
class Episode:
    """
    Semantic episodic memory entry.

    Contains complete context for learning:
    - What was the situation?
    - What did we try?
    - What happened?
    - What did we learn?
    """
    episode_id: str
    timestamp: str
    context_signature: str  # Hash of environment state
    goal_intent: str  # What we were trying to achieve
    action_type: str  # What action we took
    action_parameters: Tuple[Tuple[str, str], ...]
    transition: StateTransition
    success: bool
    success_score: float  # 0-1, how well did it work
    emotional_weight: float  # -1 to 1, how significant
    lessons: Tuple[str, ...]  # What we learned
    reusable_pattern: bool  # Can this be reused?
    times_reused: int  # How many times this pattern was applied

    @staticmethod
    def from_transition(
        transition: StateTransition,
        goal_intent: str,
        lessons: List[str] = None
    ) -> 'Episode':
        """Create episode from state transition"""
        episode_id = f"ep_{transition.transition_id}"

        # Success score based on outcome
        if transition.outcome_type == OutcomeType.IMPROVEMENT:
            success_score = 0.5 + transition.improvement_score * 0.5
            emotional = transition.improvement_score * 0.5
            success = True
        elif transition.outcome_type == OutcomeType.DEGRADATION:
            success_score = max(0.0, 0.5 + transition.improvement_score * 0.5)
            emotional = transition.improvement_score * 0.5
            success = False
        else:
            success_score = 0.5
            emotional = 0.0
            success = True  # No change is neutral

        # Lessons from the outcome
        if lessons is None:
            lessons = []
            if transition.is_positive():
                lessons.append(f"{transition.action_type} improves health")
            if transition.is_negative():
                lessons.append(f"{transition.action_type} causes degradation")
            if transition.outcome_type == OutcomeType.UNEXPECTED:
                lessons.append(f"{transition.action_type} has unexpected effects")

        # Reusable if it worked and wasn't context-specific
        reusable = success and transition.causal_confidence > 0.5

        return Episode(
            episode_id=episode_id,
            timestamp=datetime.now(UTC).isoformat(),
            context_signature=transition.before_state.state_id,
            goal_intent=goal_intent,
            action_type=transition.action_type,
            action_parameters=transition.action_parameters,
            transition=transition,
            success=success,
            success_score=min(1.0, max(0.0, success_score)),
            emotional_weight=max(-1.0, min(1.0, emotional)),
            lessons=tuple(lessons),
            reusable_pattern=reusable,
            times_reused=0
        )


@dataclass
class ExperienceMemory:
    """
    Semantic episodic memory system.

    Enables:
    - Retrieve similar situations
    - Compare outcomes
    - Avoid repetition
    - Form heuristics
    """
    episodes: List[Episode] = field(default_factory=list)
    action_patterns: Dict[str, List[Episode]] = field(default_factory=dict)  # action_type → episodes
    context_patterns: Dict[str, List[Episode]] = field(default_factory=dict)  # context → episodes

    def add_episode(self, episode: Episode):
        """Add new episode to memory"""
        self.episodes.append(episode)

        # Index by action type
        if episode.action_type not in self.action_patterns:
            self.action_patterns[episode.action_type] = []
        self.action_patterns[episode.action_type].append(episode)

        # Index by context
        ctx = episode.context_signature[:8]  # Use prefix for grouping
        if ctx not in self.context_patterns:
            self.context_patterns[ctx] = []
        self.context_patterns[ctx].append(episode)

    def get_similar_episodes(
        self,
        current_state: EnvironmentState,
        action_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Tuple[Episode, float]]:
        """Find episodes with similar states or actions"""
        candidates = []

        # Search by action type if specified
        if action_type and action_type in self.action_patterns:
            candidates.extend(self.action_patterns[action_type])

        # Search by similar context
        ctx_prefix = current_state.state_id[:8]
        if ctx_prefix in self.context_patterns:
            candidates.extend(self.context_patterns[ctx_prefix])

        # Remove duplicates and compute similarity
        seen = set()
        scored = []

        for ep in candidates:
            if ep.episode_id in seen:
                continue
            seen.add(ep.episode_id)

            sim = ep.transition.before_state.similarity(current_state)
            scored.append((ep, sim))

        # Sort by similarity and return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def get_best_action_for_state(
        self,
        state: EnvironmentState,
        action_types: List[str]
    ) -> Tuple[Optional[str], float]:
        """
        Get best action for current state based on experience.

        Returns:
        - best action type
        - confidence score (0-1)
        """
        best_action = None
        best_score = 0.0

        for action_type in action_types:
            episodes = self.action_patterns.get(action_type, [])
            if not episodes:
                continue

            # Average success for this action in similar contexts
            similar_eps = [
                (ep, ep.transition.before_state.similarity(state))
                for ep in episodes
            ]

            # Weight by similarity
            total_weight = 0.0
            weighted_score = 0.0

            for ep, sim in similar_eps:
                if sim > 0.3:  # Only count similar contexts
                    weighted_score += ep.success_score * sim
                    total_weight += sim

            if total_weight > 0:
                avg_score = weighted_score / total_weight
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = action_type

        return best_action, best_score

    def get_failures_for_action(self, action_type: str) -> List[Episode]:
        """Get all failures for an action type (for debugging)"""
        return [
            ep for ep in self.action_patterns.get(action_type, [])
            if not ep.success
        ]

    def get_pattern_success_rate(self, action_type: str) -> float:
        """Get success rate for an action type"""
        episodes = self.action_patterns.get(action_type, [])
        if not episodes:
            return 0.5  # Unknown = neutral

        successes = sum(1 for ep in episodes if ep.success)
        return successes / len(episodes)

    def extract_heuristics(self) -> List[str]:
        """Extract reusable heuristics from episodes"""
        heuristics = []

        for action_type, episodes in self.action_patterns.items():
            if not episodes:
                continue

            success_rate = self.get_pattern_success_rate(action_type)

            if success_rate > 0.8:
                heuristics.append(f"USE {action_type} - high success ({success_rate:.0%})")
            elif success_rate < 0.3:
                heuristics.append(f"AVOID {action_type} - low success ({success_rate:.0%})")

        return heuristics


def create_episode_from_execution(
    before_state: EnvironmentState,
    after_state: EnvironmentState,
    action_type: str,
    action_params: Dict[str, Any],
    goal_intent: str
) -> Episode:
    """Helper to create episode from execution"""
    transition = StateTransition.create(
        before=before_state,
        action_type=action_type,
        action_params=action_params,
        after=after_state
    )

    return Episode.from_transition(
        transition=transition,
        goal_intent=goal_intent
    )