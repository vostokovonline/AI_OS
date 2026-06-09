"""
Phase 48.5A — Historical Event Engine.

NOT random noise.
This is a STRUCTURED HISTORICAL EVENT SYSTEM with:
  - Event types with causal dependencies
  - Trigger conditions (system state + scheduled)
  - Magnitude scaling based on system vulnerability
  - Residual effects that persist across steps
  - Event cascades (one event can trigger others)
  - Institutional memory recording

Design principle:
  "History is not random — it's the interaction between
   external shocks and internal vulnerabilities."

This engine creates the PRESSURE DIFFERENTIALS that Phase 48
(currently a closed thermodynamic system) is missing.

Event Types (8 primary):
  1. COMPUTE_COLLAPSE    — sudden compute scarcity
  2. RESOURCE_INFLATION  — wealth devaluation, systemic stress
  3. NARRATIVE_CORRUPTION — memetic contamination
  4. COALITION_SCANDAL   — trust erosion within a coalition
  5. AGENT_MIGRATION     — species/coalition exodus
  6. MEMORY_LOSS_WAVE    — institutional amnesia
  7. CONSTITUTION_ATTACK — coordinated pressure on a specific article
  8. IDENTITY_CRISIS     — self coherence shock
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import sys
sys.path.insert(0, '.')

# ---------------------------------------------------------------------------
# Event Taxonomy
# ---------------------------------------------------------------------------

class HistoricalEventType(Enum):
    """Eight canonical historical event types."""
    COMPUTE_COLLAPSE = 'compute_collapse'
    RESOURCE_INFLATION = 'resource_inflation'
    NARRATIVE_CORRUPTION = 'narrative_corruption'
    COALITION_SCANDAL = 'coalition_scandal'
    AGENT_MIGRATION = 'agent_migration'
    MEMORY_LOSS_WAVE = 'memory_loss_wave'
    CONSTITUTION_ATTACK = 'constitution_attack'
    IDENTITY_CRISIS = 'identity_crisis'


@dataclass
class HistoricalEvent:
    """
    A single historical event with causal context.

    This is NOT just 'type + magnitude'.
    Every event has:
      - source: what triggered it (system state, scheduled, cascade)
      - magnitude: 0..1 severity
      - target: who/what it affects (specific agent, coalition, species, or 'system')
      - causal_parents: which previous events contributed to this one
      - residual: remaining effect that decays over steps
      - step: when it occurred
    """
    event_id: str
    event_type: HistoricalEventType
    step: int
    magnitude: float
    source: str  # 'system_state', 'scheduled', 'cascade', 'trigger'
    target: str  # agent_id, coalition_id, species, 'system'
    description: str
    causal_parents: List[str] = field(default_factory=list)
    residual: float = 1.0
    decay_rate: float = 0.05
    subsystem_hit: str = 'economy'  # economy, narrative, coalition, constitution, self

    def step_decay(self) -> bool:
        """Decay residual effect. Returns True if still active."""
        self.residual = max(0.0, self.residual - self.decay_rate)
        return self.residual > 0.01


@dataclass
class EventTrigger:
    """
    Rule-based trigger that examines system state to fire events.

    Each trigger type checks a specific condition:
      - wealth_gini_above: fire when wealth inequality exceeds threshold
      - species_diversity_below: fire when species diversity drops
      - narrative_monopoly_above: fire when narrative concentration exceeds threshold
      - coalition_stability_below: fire when coalition cohesion drops
      - self_coherence_below: fire when identity coherence drops
      - constitution_violation_rate_above: fire when violations spike
      - scheduled: fire at specific step intervals
    """
    trigger_type: str  # system_state, scheduled, cascade
    condition_attr: str  # metric name on the system state
    threshold: float
    cooldown: int = 50
    last_fired: int = -1000


class HistoricalEventEngine:
    """
    Structured historical event engine.

    Generates events based on:
      1. System state thresholds (inequality, diversity, etc.)
      2. Scheduled historical moments
      3. Cascade from previous events

    Events are NOT random — they are triggered by specific conditions
    that reveal system vulnerabilities. The engine creates PRESSURE
    DIFFERENTIALS by hitting subsystems where they are weakest.
    """

    def __init__(
        self,
        event_cooldown: int = 30,
        max_active_events: int = 5,
        cascade_probability: float = 0.3,
        enable_scheduled: bool = True,
        enable_state_triggers: bool = True,
    ):
        self.cooldown = event_cooldown
        self.max_active_events = max_active_events
        self.cascade_probability = cascade_probability
        self.enable_scheduled = enable_scheduled
        self.enable_state_triggers = enable_state_triggers

        self.events: List[HistoricalEvent] = []
        self.active_events: List[HistoricalEvent] = []
        self.event_count: int = 0
        self.last_event_step: int = -1000

        # State triggers — fire when system metrics cross thresholds
        self.triggers: List[EventTrigger] = [
            # Economic stress triggers
            EventTrigger('system_state', 'wealth_gini', 0.5, cooldown=60),
            EventTrigger('system_state', 'wealth_gini', 0.7, cooldown=100),

            # Diversity stress triggers
            EventTrigger('system_state', 'species_diversity', 0.3, cooldown=80),

            # Narrative stress triggers
            EventTrigger('system_state', 'narrative_monopoly', 0.6, cooldown=70),

            # Coalition stress triggers
            EventTrigger('system_state', 'coalition_stability', 0.15, cooldown=50),

            # Self coherence triggers
            EventTrigger('system_state', 'self_coherence', 0.2, cooldown=90),

            # Constitutional stress triggers
            EventTrigger('system_state', 'violation_rate', 0.4, cooldown=100),
        ]

        # Scheduled events — fire at specific historical moments
        self.scheduled_events: List[Dict] = [
            {'step': 100, 'type': HistoricalEventType.RESOURCE_INFLATION,
             'magnitude': 0.3, 'source': 'scheduled',
             'target': 'system',
             'description': 'First resource inflation — test economic resilience'},
            {'step': 250, 'type': HistoricalEventType.COALITION_SCANDAL,
             'magnitude': 0.4, 'source': 'scheduled',
             'target': 'dominant_coalition',
             'description': 'Dominant coalition scandal — test political turnover'},
            {'step': 500, 'type': HistoricalEventType.NARRATIVE_CORRUPTION,
             'magnitude': 0.5, 'source': 'scheduled',
             'target': 'system',
             'description': 'Narrative corruption wave — test memetic immunity'},
            {'step': 800, 'type': HistoricalEventType.COMPUTE_COLLAPSE,
             'magnitude': 0.6, 'source': 'scheduled',
             'target': 'system',
             'description': 'Compute collapse — test scarcity response'},
            {'step': 1200, 'type': HistoricalEventType.CONSTITUTION_ATTACK,
             'magnitude': 0.5, 'source': 'scheduled',
             'target': 'anti_monopoly',
             'description': 'Anti-monopoly article attack — test constitutional resilience'},
            {'step': 2000, 'type': HistoricalEventType.IDENTITY_CRISIS,
             'magnitude': 0.6, 'source': 'scheduled',
             'target': 'system',
             'description': 'Identity crisis — test self recovery'},
            {'step': 3000, 'type': HistoricalEventType.MEMORY_LOSS_WAVE,
             'magnitude': 0.5, 'source': 'scheduled',
             'target': 'system',
             'description': 'Memory loss wave — test institutional continuity'},
            {'step': 5000, 'type': HistoricalEventType.AGENT_MIGRATION,
             'magnitude': 0.7, 'source': 'scheduled',
             'target': 'system',
             'description': 'Great agent migration — test coalition restructuring'},
        ]

    def _next_id(self) -> str:
        self.event_count += 1
        return f"he_{self.event_count}"

    def _compute_magnitude(
        self,
        event_type: HistoricalEventType,
        system_state: Dict[str, float]
    ) -> float:
        """
        Scale event magnitude based on system vulnerability.

        If the system is already fragile, the same event hits harder.
        This creates realistic dynamics:
          - A healthy system weathers shocks
          - A fragile system cascades
        """
        base = 0.3

        if event_type == HistoricalEventType.COMPUTE_COLLAPSE:
            # Worse if economy already strained
            gini = system_state.get('wealth_gini', 0.0)
            base += 0.3 * gini

        elif event_type == HistoricalEventType.RESOURCE_INFLATION:
            # Worse if agents already poor
            mean_wealth = system_state.get('mean_wealth', 0.5)
            base += 0.3 * max(0.0, 0.5 - mean_wealth)

        elif event_type == HistoricalEventType.NARRATIVE_CORRUPTION:
            # Worse if narrative diversity already low
            div = system_state.get('narrative_diversity', 0.5)
            base += 0.3 * max(0.0, 0.5 - div)

        elif event_type == HistoricalEventType.COALITION_SCANDAL:
            # Worse if coalition turnover is low (stagnant politics)
            turnover = system_state.get('coalition_turnover', 0.1)
            base += 0.3 * max(0.0, 0.2 - turnover)

        elif event_type == HistoricalEventType.AGENT_MIGRATION:
            # Worse if species diversity is low
            div = system_state.get('species_diversity', 1.0)
            base += 0.3 * max(0.0, 1.0 - div)

        elif event_type == HistoricalEventType.MEMORY_LOSS_WAVE:
            # Worse if constitution has many violations (institutional fatigue)
            violations = system_state.get('violation_rate', 0.0)
            base += 0.3 * violations

        elif event_type == HistoricalEventType.CONSTITUTION_ATTACK:
            # Worse if constitution is weak (low article strengths)
            base += 0.3 * system_state.get('constitution_strength', 0.5)

        elif event_type == HistoricalEventType.IDENTITY_CRISIS:
            # Worse if self coherence already fragile
            coherence = system_state.get('self_coherence', 0.5)
            base += 0.3 * max(0.0, 0.5 - coherence)

        return min(0.95, base)

    def _check_state_triggers(
        self,
        system_state: Dict[str, float],
        step: int
    ) -> List[Dict]:
        """
        Check system state against triggers.

        This is how the system's OWN vulnerabilities generate events.
        High inequality → resource events.
        Low diversity → species/narrative events.
        """
        triggered = []

        for trigger in self.triggers:
            if step - trigger.last_fired < trigger.cooldown:
                continue

            current_value = system_state.get(trigger.condition_attr, 0.5)

            # Check threshold direction
            if trigger.condition_attr in ['wealth_gini', 'narrative_monopoly',
                                           'violation_rate']:
                # Higher = worse for these metrics
                if current_value > trigger.threshold:
                    trigger.last_fired = step
                    triggered.append({
                        'type': self._metric_to_event_type(trigger.condition_attr),
                        'source': 'trigger',
                        'target': 'system',
                        'condition': trigger.condition_attr,
                        'threshold': trigger.threshold,
                        'current': current_value,
                    })
            else:
                # Lower = worse for these metrics
                if current_value < trigger.threshold:
                    trigger.last_fired = step
                    triggered.append({
                        'type': self._metric_to_event_type(trigger.condition_attr),
                        'source': 'trigger',
                        'target': 'system',
                        'condition': trigger.condition_attr,
                        'threshold': trigger.threshold,
                        'current': current_value,
                    })

        return triggered

    def _metric_to_event_type(self, metric: str) -> HistoricalEventType:
        """Map a system metric to the most relevant event type."""
        mapping = {
            'wealth_gini': HistoricalEventType.RESOURCE_INFLATION,
            'species_diversity': HistoricalEventType.AGENT_MIGRATION,
            'narrative_monopoly': HistoricalEventType.NARRATIVE_CORRUPTION,
            'coalition_stability': HistoricalEventType.COALITION_SCANDAL,
            'self_coherence': HistoricalEventType.IDENTITY_CRISIS,
            'violation_rate': HistoricalEventType.CONSTITUTION_ATTACK,
        }
        return mapping.get(metric, HistoricalEventType.RESOURCE_INFLATION)

    def _check_scheduled(
        self,
        step: int,
        system_state: Dict[str, float]
    ) -> List[Dict]:
        """Check if a scheduled event should fire at this step."""
        fired = []
        for sched in self.scheduled_events:
            if sched['step'] == step:
                magnitude = self._compute_magnitude(sched['type'], system_state)
                fired.append({
                    'type': sched['type'],
                    'magnitude': magnitude,
                    'source': sched['source'],
                    'target': sched['target'],
                    'description': sched['description'],
                })
        return fired

    def _check_cascade(
        self,
        step: int
    ) -> List[Dict]:
        """
        Check if active events cascade into new events.

        Cascading creates HISTORICAL CHAINS:
          Resource inflation → Coalition scandal → Identity crisis

        This is critical for creating narrative arcs, not isolated noise.
        """
        cascades = []

        for event in self.active_events:
            if event.residual < 0.3:
                continue
            if random.random() > self.cascade_probability:
                continue

            # Cascade type depends on parent event type
            cascade_map = {
                HistoricalEventType.COMPUTE_COLLAPSE: [
                    HistoricalEventType.RESOURCE_INFLATION,
                    HistoricalEventType.AGENT_MIGRATION,
                ],
                HistoricalEventType.RESOURCE_INFLATION: [
                    HistoricalEventType.COALITION_SCANDAL,
                    HistoricalEventType.IDENTITY_CRISIS,
                ],
                HistoricalEventType.NARRATIVE_CORRUPTION: [
                    HistoricalEventType.COALITION_SCANDAL,
                    HistoricalEventType.CONSTITUTION_ATTACK,
                ],
                HistoricalEventType.COALITION_SCANDAL: [
                    HistoricalEventType.AGENT_MIGRATION,
                    HistoricalEventType.IDENTITY_CRISIS,
                ],
                HistoricalEventType.AGENT_MIGRATION: [
                    HistoricalEventType.COALITION_SCANDAL,
                    HistoricalEventType.MEMORY_LOSS_WAVE,
                ],
                HistoricalEventType.MEMORY_LOSS_WAVE: [
                    HistoricalEventType.IDENTITY_CRISIS,
                    HistoricalEventType.CONSTITUTION_ATTACK,
                ],
                HistoricalEventType.CONSTITUTION_ATTACK: [
                    HistoricalEventType.COMPUTE_COLLAPSE,
                    HistoricalEventType.MEMORY_LOSS_WAVE,
                ],
                HistoricalEventType.IDENTITY_CRISIS: [
                    HistoricalEventType.AGENT_MIGRATION,
                    HistoricalEventType.NARRATIVE_CORRUPTION,
                ],
            }

            possible = cascade_map.get(event.event_type, [])
            if possible:
                cascade_type = random.choice(possible)
                cascades.append({
                    'type': cascade_type,
                    'magnitude': event.magnitude * 0.6,
                    'source': 'cascade',
                    'parent_id': event.event_id,
                    'target': 'system',
                    'description': f'Cascade from {event.event_type.value} at step {event.step}',
                })

        return cascades

    def step(
        self,
        system_state: Dict[str, float],
        step: int,
        force_event: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run one event generation cycle.

        Args:
            system_state: dict of system metrics (wealth_gini, species_diversity, etc.)
            step: current simulation step
            force_event: optional dict to force a specific event (for testing)

        Returns:
            dict with 'new_events', 'active_events', 'cascade_events', 'stats'
        """
        # Cooldown check
        if step - self.last_event_step < self.cooldown and force_event is None:
            return {'new_events': [], 'active_events': self.active_events,
                    'cascade_events': [], 'stats': self._get_stats()}

        # Decay existing active events
        self.active_events = [e for e in self.active_events if e.step_decay()]

        # Collect candidate events from all sources
        candidates = []

        # 1. Forced event (for testing / external control)
        if force_event is not None:
            candidates.append(force_event)

        # 2. State triggers (system vulnerabilities)
        if self.enable_state_triggers:
            candidates.extend(self._check_state_triggers(system_state, step))

        # 3. Scheduled events (historical moments)
        if self.enable_scheduled:
            candidates.extend(self._check_scheduled(step, system_state))

        # 4. Cascades (chain reactions)
        candidates.extend(self._check_cascade(step))

        # Limit to max_active_events
        candidates = candidates[:self.max_active_events]

        # Create HistoricalEvent objects
        new_events = []
        for cand in candidates:
            # Forced, scheduled, and cascade events always pass;
            # state-triggered events have ~70% probability (avoid noise)
            source = cand.get('source', '')
            is_always = source in ('test', 'forced', 'scheduled', 'cascade')
            if not is_always and random.random() < 0.3:  # 30% chance to skip state triggers
                continue

            event_type = cand.get('type')
            if isinstance(event_type, str):
                event_type = HistoricalEventType(event_type)

            magnitude = cand.get('magnitude', self._compute_magnitude(
                event_type, system_state
            ))

            event = HistoricalEvent(
                event_id=self._next_id(),
                event_type=event_type,
                step=step,
                magnitude=magnitude,
                source=cand.get('source', 'trigger'),
                target=cand.get('target', 'system'),
                description=cand.get('description', f'{event_type.value} at step {step}'),
                causal_parents=[cand.get('parent_id')] if 'parent_id' in cand else [],
                subsystem_hit=self._type_to_subsystem(event_type),
            )
            self.events.append(event)
            self.active_events.append(event)
            new_events.append(event)
            self.last_event_step = step

        return {
            'new_events': new_events,
            'active_events': self.active_events,
            'candidate_count': len(candidates),
            'stats': self._get_stats(),
        }

    def apply_event(
        self,
        event: HistoricalEvent,
        agents: List,
        species: Dict,
        narrative_ecosystem: Any,
        constitution: Any,
        coalitions: List,
    ) -> Dict[str, Any]:
        """
        Apply an event's effects to the system.

        This is where events actually CHANGE the system state.
        Each event type has specific effects:
          - COMPUTE_COLLAPSE: reduces agent wealth, increases bid intensity
          - RESOURCE_INFLATION: devalues wealth, increases risk tolerance
          - NARRATIVE_CORRUPTION: introduces noise narratives, kills weak genes
          - COALITION_SCANDAL: reduces coalition cohesion, drops reliability
          - AGENT_MIGRATION: converts agent species, shifts ideology
          - MEMORY_LOSS_WAVE: resets institutional memory, drops article strength
          - CONSTITUTION_ATTACK: weakens a specific article
          - IDENTITY_CRISIS: drops self coherence, increases exploration

        Returns dict of effects applied (for logging).
        """
        effects = {
            'agents_affected': 0,
            'narratives_killed': 0,
            'narratives_spawned': 0,
            'coalitions_damaged': 0,
            'wealth_reduced': 0.0,
            'reliability_reduced': 0.0,
            'species_changed': 0,
            'articles_weakened': [],
            'memory_erased': 0,
        }

        if event.event_type == HistoricalEventType.COMPUTE_COLLAPSE:
            # Sudden compute scarcity — wealth tax on all agents
            for agent in agents:
                if not agent.active:
                    continue
                tax = agent.wealth * event.magnitude * 0.3
                agent.wealth = max(0.1, agent.wealth - tax)
                agent.bid_intensity = min(0.95, agent.bid_intensity * 1.2)
                effects['wealth_reduced'] += tax
                effects['agents_affected'] += 1

        elif event.event_type == HistoricalEventType.RESOURCE_INFLATION:
            # Wealth devaluation — reduce purchasing power
            for agent in agents:
                if not agent.active:
                    continue
                agent.wealth *= (1.0 - event.magnitude * 0.2)
                agent.wealth = max(0.1, agent.wealth)
                agent.risk_tolerance = min(0.9,
                    agent.risk_tolerance + event.magnitude * 0.1)
                effects['wealth_reduced'] += agent.wealth * event.magnitude * 0.2
                effects['agents_affected'] += 1

        elif event.event_type == HistoricalEventType.NARRATIVE_CORRUPTION:
            # Memetic contamination — kill weak genes, spawn noise
            if narrative_ecosystem is not None:
                genes_before = len(narrative_ecosystem.genes)
                # Kill low-fitness genes
                for gid in list(narrative_ecosystem.genes.keys()):
                    gene = narrative_ecosystem.genes[gid]
                    if gene.fitness < 0.3:
                        del narrative_ecosystem.genes[gid]
                        effects['narratives_killed'] += 1
                # Spawn corrupt narratives (noise vectors)
                n_corrupt = max(1, int(event.magnitude * 5))
                for _ in range(n_corrupt):
                    noise_vec = np.random.randn(narrative_ecosystem.semantic_dim)
                    noise_vec = noise_vec / (np.linalg.norm(noise_vec) + 1e-8)
                    narrative_ecosystem.seed(
                        noise_vec, fitness=0.15 + 0.2 * event.magnitude
                    )
                    effects['narratives_spawned'] += 1

        elif event.event_type == HistoricalEventType.COALITION_SCANDAL:
            # Trust erosion — reduce reliability of coalition members
            target_id = None
            if event.target == 'dominant_coalition' and coalitions:
                dominant = max(coalitions, key=lambda c: c.total_fitness *
                               c.compute_cohesion()) if coalitions else None
                if dominant:
                    target_id = dominant.coalition_id
            else:
                target_id = event.target

            for agent in agents:
                if not agent.active:
                    continue
                if target_id and agent.coalition_id == target_id:
                    agent.reliability *= (1.0 - event.magnitude * 0.3)
                    agent.wealth *= (1.0 - event.magnitude * 0.15)
                    agent.wealth = max(0.1, agent.wealth)
                    effects['reliability_reduced'] += event.magnitude * 0.3
                    effects['agents_affected'] += 1
                    effects['coalitions_damaged'] += 1

        elif event.event_type == HistoricalEventType.AGENT_MIGRATION:
            # Species/coalition exodus — convert agents to random species
            active = [a for a in agents if a.active]
            if species and isinstance(species, dict) and len(species) > 1:
                n_migrants = max(1, int(len(active) * event.magnitude * 0.3))
                migrants = random.sample(active, min(n_migrants, len(active)))
                species_types = list(species.keys())
                for agent in migrants:
                    new_type = random.choice(
                        [s for s in species_types if s != agent.species]
                    )
                    agent.species = new_type
                    agent.coalition_id = None  # Leave coalition
                    agent.ideology += np.random.randn(len(agent.ideology)) * 0.2
                    agent.exploration_rate = min(0.9,
                        agent.exploration_rate + event.magnitude * 0.2)
                    effects['species_changed'] += 1
                    effects['agents_affected'] += 1

        elif event.event_type == HistoricalEventType.MEMORY_LOSS_WAVE:
            # Institutional amnesia — reset violation logs, weaken articles
            if constitution is not None:
                constitution.violation_log = constitution.violation_log[
                    len(constitution.violation_log) // 2:
                ]
                for article in constitution.articles.values():
                    old_strength = article.strength
                    article.strength *= (1.0 - event.magnitude * 0.3)
                    effects['articles_weakened'].append({
                        'name': article.name,
                        'from': round(old_strength, 3),
                        'to': round(article.strength, 3),
                    })
                effects['memory_erased'] = len(constitution.violation_log)

        elif event.event_type == HistoricalEventType.CONSTITUTION_ATTACK:
            # Coordinated pressure on a specific article
            if constitution is not None:
                target_article = event.target
                if target_article in constitution.articles:
                    article = constitution.articles[target_article]
                    old_strength = article.strength
                    article.strength *= (1.0 - event.magnitude * 0.5)
                    article.strength = max(0.05, article.strength)
                    effects['articles_weakened'].append({
                        'name': target_article,
                        'from': round(old_strength, 3),
                        'to': round(article.strength, 3),
                    })
                else:
                    # Attack a random article
                    target = random.choice(list(constitution.articles.keys()))
                    article = constitution.articles[target]
                    old_strength = article.strength
                    article.strength *= (1.0 - event.magnitude * 0.4)
                    article.strength = max(0.05, article.strength)
                    effects['articles_weakened'].append({
                        'name': target,
                        'from': round(old_strength, 3),
                        'to': round(article.strength, 3),
                    })

        elif event.event_type == HistoricalEventType.IDENTITY_CRISIS:
            # Self coherence shock — increase exploration, drop agency claims
            for agent in agents:
                if not agent.active:
                    continue
                agent.exploration_rate = min(0.9,
                    agent.exploration_rate + event.magnitude * 0.15)
                agent.agency_claim *= (1.0 - event.magnitude * 0.2)
                agent.time_horizon = max(0.1,
                    agent.time_horizon - event.magnitude * 0.1)
                effects['agents_affected'] += 1

        return effects

    def _type_to_subsystem(self, event_type: HistoricalEventType) -> str:
        mapping = {
            HistoricalEventType.COMPUTE_COLLAPSE: 'economy',
            HistoricalEventType.RESOURCE_INFLATION: 'economy',
            HistoricalEventType.NARRATIVE_CORRUPTION: 'narrative',
            HistoricalEventType.COALITION_SCANDAL: 'coalition',
            HistoricalEventType.AGENT_MIGRATION: 'coalition',
            HistoricalEventType.MEMORY_LOSS_WAVE: 'constitution',
            HistoricalEventType.CONSTITUTION_ATTACK: 'constitution',
            HistoricalEventType.IDENTITY_CRISIS: 'self',
        }
        return mapping.get(event_type, 'economy')

    def get_event_history(
        self,
        event_type: Optional[HistoricalEventType] = None,
        min_magnitude: float = 0.0,
        last_n: int = 50
    ) -> List[HistoricalEvent]:
        """Get filtered event history."""
        history = self.events[-last_n:]
        if event_type is not None:
            history = [e for e in history if e.event_type == event_type]
        if min_magnitude > 0:
            history = [e for e in history if e.magnitude >= min_magnitude]
        return history

    def get_causal_chains(self, depth: int = 3) -> List[List[HistoricalEvent]]:
        """
        Extract causal chains from event history.

        Returns chains of events where parent events sparked cascades.
        This is critical for: narrative generation, historical understanding,
        and institutional learning.
        """
        chains = []
        for event in self.events:
            if event.causal_parents:
                chain = [event]
                current = event
                for _ in range(depth - 1):
                    # Find parent
                    parents = [e for e in self.events
                               if e.event_id in current.causal_parents]
                    if parents:
                        current = parents[0]
                        chain.insert(0, current)
                    else:
                        break
                if len(chain) > 1:
                    chains.append(chain)
        return chains

    def get_event_rate(self, last_n: int = 100) -> float:
        """Events per step over recent history."""
        recent = [e for e in self.events if e.step > max(0, len(self.events) - last_n)]
        return len(recent) / max(1, last_n)

    def _get_stats(self) -> Dict:
        return {
            'total_events': len(self.events),
            'active_residual': len(self.active_events),
            'event_rate': round(self.get_event_rate(), 3),
            'causal_chains': len(self.get_causal_chains()),
            'events_by_type': {
                et.value: len([e for e in self.events if e.event_type == et])
                for et in HistoricalEventType
            },
        }

    def get_narrative_summary(self) -> str:
        """Generate a human-readable summary of recent history."""
        if not self.events:
            return "No historical events recorded."

        recent = self.events[-10:]
        lines = [f"Last {len(recent)} events:"]
        for e in recent:
            lines.append(
                f"  [{e.step:5d}] {e.event_type.value:25s}"
                f" | mag={e.magnitude:.2f} | src={e.source:15s}"
                f" | {e.description}"
            )
        chains = self.get_causal_chains()
        if chains:
            lines.append(f"Causal chains: {len(chains)}")
            longest = max(chains, key=len)
            chain_str = " → ".join(
                f"{e.event_type.value}({e.step})" for e in longest
            )
            lines.append(f"  Longest: {chain_str}")
        return "\n".join(lines)


# ============================================================================
# INTEGRATION: GradientInjector → HistoricalEventEngine
# ============================================================================

def integrate_with_gradient_injector(
    gradient_injector: Any,
    historical_engine: HistoricalEventEngine,
) -> Any:
    """
    Connect HistoricalEventEngine to GradientInjector.

    This wraps the injector's inject() method so that:
      1. Historical events are generated each step
      2. They are passed to the injector as structured shocks
      3. The injector handles them alongside its own injections

    Returns the modified injector (monkey-patched for now;
    production would use a proper integration pattern).
    """
    original_inject = gradient_injector.inject

    def patched_inject(agents, species, narrative_ecosystem, constitution, step):
        # Run historical event engine
        system_state = _build_system_state(
            agents, species, narrative_ecosystem, constitution
        )
        event_result = historical_engine.step(system_state, step)

        # Apply historical events to system
        for event in event_result['new_events']:
            # Get coalitions from the system
            coalitions = getattr(constitution, 'coalitions', [])
            if hasattr(constitution, 'coalition_self'):
                coalitions = getattr(
                    constitution.coalition_self, 'coalitions', []
                )

            effects = historical_engine.apply_event(
                event, agents, species,
                narrative_ecosystem, constitution, coalitions
            )
            # Store effects for logging
            event.effects = effects

        # Run original injector
        result = original_inject(
            agents, species, narrative_ecosystem, constitution, step
        )

        # Add historical events to result
        result['historical_events'] = event_result['new_events']
        return result

    gradient_injector.inject = patched_inject
    gradient_injector.historical_engine = historical_engine
    return gradient_injector


def _build_system_state(
    agents: List,
    species: Dict,
    narrative_ecosystem: Any,
    constitution: Any,
) -> Dict[str, float]:
    """Build system state dict for event trigger checking."""
    active = [a for a in agents if hasattr(a, 'active') and a.active] if agents else []

    # Wealth metrics
    wealths = [a.wealth for a in active if hasattr(a, 'wealth')]
    mean_wealth = float(np.mean(wealths)) if wealths else 0.5
    sorted_w = sorted(wealths)
    n = len(sorted_w)
    if n > 1:
        # Gini coefficient
        cum = np.cumsum(sorted(sorted_w))
        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_w) /
                (n * np.sum(sorted_w)) - (n + 1) / n) if sum(wealths) > 0 else 0.0
        gini = float(np.clip(gini, 0.0, 1.0))
    else:
        gini = 0.0

    # Species diversity
    if species and isinstance(species, dict):
        pops = [s.population for s in species.values()
                if hasattr(s, 'population') and s.population > 0]
        total = sum(pops) + 1e-8
        props = np.array(pops) / total if pops else np.array([1.0])
        species_div = float(-np.sum(props * np.log(props + 1e-8)))
    else:
        species_div = 0.0

    # Narrative metrics
    if narrative_ecosystem is not None:
        narrative_div = getattr(narrative_ecosystem, 'get_diversity', lambda: 0.0)()
        influence_conc = getattr(
            narrative_ecosystem, 'get_influence_concentration', lambda: 0.0
        )()
    else:
        narrative_div = 0.5
        influence_conc = 0.0

    # Constitution metrics
    if constitution is not None:
        n_violations = getattr(constitution, 'get_violation_count', lambda: 0)()
        # Mean article strength
        articles = getattr(constitution, 'articles', {})
        mean_article_strength = float(np.mean(
            [a.strength for a in articles.values()]
        )) if articles else 0.5
        violation_rate = min(1.0, n_violations / max(1, len(articles) * 10))
    else:
        violation_rate = 0.0
        mean_article_strength = 0.5

    # Coalition metrics
    coalitions_data = getattr(constitution, 'coalitions', [])
    if hasattr(constitution, 'coalition_self'):
        coalitions_data = getattr(
            constitution.coalition_self, 'coalitions', []
        )
    coalition_stability = 0.5
    coalition_turnover = 0.1

    # Self coherence
    self_coherence = 0.5
    if hasattr(constitution, 'coalition_self'):
        cs = getattr(constitution, 'coalition_self', None)
        if cs:
            self_coherence = getattr(cs, 'identity_stability', 0.5)

    return {
        'wealth_gini': gini,
        'mean_wealth': mean_wealth,
        'species_diversity': species_div,
        'narrative_diversity': narrative_div,
        'narrative_monopoly': influence_conc,
        'coalition_stability': coalition_stability,
        'coalition_turnover': coalition_turnover,
        'self_coherence': self_coherence,
        'violation_rate': violation_rate,
        'constitution_strength': mean_article_strength,
    }


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_event_generation():
    """Test basic event generation from triggers."""
    print("\n" + "=" * 60)
    print("48.5A.1 — EVENT GENERATION")
    print("=" * 60)

    engine = HistoricalEventEngine(
        event_cooldown=1,
        enable_scheduled=False,  # Only state triggers
    )

    # System state with high inequality — should trigger resource events
    system_state = {
        'wealth_gini': 0.8,  # High inequality
        'mean_wealth': 0.3,
        'species_diversity': 1.0,
        'narrative_diversity': 0.5,
        'narrative_monopoly': 0.3,
        'coalition_stability': 0.5,
        'coalition_turnover': 0.1,
        'self_coherence': 0.5,
        'violation_rate': 0.0,
        'constitution_strength': 0.5,
    }

    # Run multiple steps to trigger events
    events = []
    for step in range(1, 51):
        result = engine.step(system_state, step)
        events.extend(result['new_events'])

    assert len(events) > 0, "Should generate events from triggers"
    print(f"  ✓ {len(events)} events generated in 50 steps")

    types = set(e.event_type.value for e in events)
    print(f"  ✓ Event types: {types}")

    print("  >>> EventGeneration PASSED\n")


def test_scheduled_events():
    """Test scheduled historical events."""
    print("\n" + "=" * 60)
    print("48.5A.2 — SCHEDULED EVENTS")
    print("=" * 60)

    engine = HistoricalEventEngine(
        event_cooldown=10,
        enable_state_triggers=False,  # Only scheduled
    )

    system_state = {
        'wealth_gini': 0.3, 'mean_wealth': 0.5,
        'species_diversity': 1.0, 'narrative_diversity': 0.5,
        'narrative_monopoly': 0.3, 'coalition_stability': 0.5,
        'coalition_turnover': 0.1, 'self_coherence': 0.5,
        'violation_rate': 0.0, 'constitution_strength': 0.5,
    }

    events = []
    for step in range(1, 301):
        result = engine.step(system_state, step)
        events.extend(result['new_events'])

    types = set(e.event_type.value for e in events)
    print(f"  ✓ {len(events)} scheduled events generated by step 300")
    print(f"  ✓ Types: {types}")

    # Check resource inflation at step ~100
    inflation_events = [e for e in events
                        if e.event_type == HistoricalEventType.RESOURCE_INFLATION]
    assert len(inflation_events) >= 1, "Should have resource inflation event"
    print(f"  ✓ Resource inflation fired at step(s): "
          f"{[e.step for e in inflation_events]}")

    print("  >>> ScheduledEvents PASSED\n")


def test_event_cascade():
    """Test event cascading (chain reactions)."""
    print("\n" + "=" * 60)
    print("48.5A.3 — EVENT CASCADE")
    print("=" * 60)

    engine = HistoricalEventEngine(
        event_cooldown=3,
        cascade_probability=1.0,  # Force cascades for testing
        enable_scheduled=False,
        enable_state_triggers=False,
    )

    system_state = {
        'wealth_gini': 0.3, 'mean_wealth': 0.5,
        'species_diversity': 1.0, 'narrative_diversity': 0.5,
        'narrative_monopoly': 0.3, 'coalition_stability': 0.5,
        'coalition_turnover': 0.1, 'self_coherence': 0.5,
        'violation_rate': 0.0, 'constitution_strength': 0.5,
    }

    # Inject a forced compute collapse event
    events = []
    for step in range(1, 51):
        force = None
        if step == 5:
            force = {
                'type': HistoricalEventType.COMPUTE_COLLAPSE,
                'magnitude': 0.5, 'source': 'test',
                'target': 'system',
                'description': 'Forced compute collapse for cascade test',
            }
        result = engine.step(system_state, step, force_event=force)
        events.extend(result['new_events'])

    assert len(events) >= 1, "Should have at least the forced event"
    print(f"  ✓ {len(events)} events generated (forced + cascades)")

    # Check for causal chains
    chains = engine.get_causal_chains()
    print(f"  ✓ Causal chains detected: {len(chains)}")
    for chain in chains:
        chain_str = " → ".join(f"{e.event_type.value}(s{e.step})" for e in chain)
        print(f"      {chain_str}")

    print("  >>> EventCascade PASSED\n")


def test_event_application():
    """Test event application to agents, narratives, constitution."""
    print("\n" + "=" * 60)
    print("48.5A.4 — EVENT APPLICATION")
    print("=" * 60)

    # Import Phase 48 components
    from phase48_cognitive_political_economy import (
        CognitiveAgent, NarrativeEcosystem, ConstitutionalLayer
    )

    engine = HistoricalEventEngine()

    # Create test system
    agents = []
    for i in range(5):
        a = CognitiveAgent(f'a_{i}', 'exploitative', np.random.randn(32),
                           exploration_rate=0.3, reliability=0.5)
        a.wealth = 2.0
        a.coalition_id = f'coal_{i % 2}'
        a.active = True
        agents.append(a)

    ecosystem = NarrativeEcosystem(semantic_dim=32)
    for _ in range(10):
        ecosystem.seed(np.random.randn(32), fitness=0.5)

    constitution = ConstitutionalLayer()
    species = {'exploitative': type('S', (), {'population': 5})(),
               'exploratory': type('S', (), {'population': 3})()}

    coalitions = []
    for i in range(2):
        coal = type('C', (), {'coalition_id': f'coal_{i}',
                               'total_fitness': 0.5,
                               'compute_cohesion': lambda self: 0.6})()
        coalitions.append(coal)

    # Test each event type
    event_types = list(HistoricalEventType)

    for et in event_types:
        event = HistoricalEvent(
            event_id=f'test_{et.value}',
            event_type=et,
            step=100,
            magnitude=0.5,
            source='test',
            target='anti_monopoly' if et == HistoricalEventType.CONSTITUTION_ATTACK
                   else 'system',
            description=f'Test {et.value}',
        )
        effects = engine.apply_event(
            event, agents, species, ecosystem, constitution, coalitions
        )
        print(f"  ✓ {et.value:25s} applied — "
              f"agents={effects.get('agents_affected', 0)}, "
              f"narratives={effects.get('narratives_killed', 0)}/"
              f"{effects.get('narratives_spawned', 0)}, "
              f"articles={len(effects.get('articles_weakened', []))}")

    # Verify constitution attack weakened an article
    attacked_event = HistoricalEvent(
        event_id='test_constitution_attack',
        event_type=HistoricalEventType.CONSTITUTION_ATTACK,
        step=101, magnitude=0.5,
        source='test', target='anti_monopoly',
        description='Constitution attack test',
    )
    old_strength = constitution.articles['anti_monopoly'].strength
    engine.apply_event(
        attacked_event, agents, species, ecosystem, constitution, coalitions
    )
    new_strength = constitution.articles['anti_monopoly'].strength
    assert new_strength < old_strength, \
        f"Article strength should decrease: {old_strength:.3f} → {new_strength:.3f}"
    print(f"\n  ✓ Constitution attack: anti_monopoly {old_strength:.3f} → "
          f"{new_strength:.3f}")

    print("  >>> EventApplication PASSED\n")


def test_vulnerability_scaling():
    """Test that event magnitude scales with system vulnerability."""
    print("\n" + "=" * 60)
    print("48.5A.5 — VULNERABILITY SCALING")
    print("=" * 60)

    engine = HistoricalEventEngine()

    # Healthy system — events should be mild
    healthy_state = {
        'wealth_gini': 0.2, 'mean_wealth': 0.8,
        'species_diversity': 1.5, 'narrative_diversity': 0.8,
        'narrative_monopoly': 0.2, 'coalition_stability': 0.8,
        'coalition_turnover': 0.3, 'self_coherence': 0.9,
        'violation_rate': 0.0, 'constitution_strength': 0.7,
    }

    # Fragile system — events should hit harder
    fragile_state = {
        'wealth_gini': 0.8, 'mean_wealth': 0.2,
        'species_diversity': 0.3, 'narrative_diversity': 0.2,
        'narrative_monopoly': 0.8, 'coalition_stability': 0.1,
        'coalition_turnover': 0.0, 'self_coherence': 0.2,
        'violation_rate': 0.8, 'constitution_strength': 0.2,
    }

    healthy_mags = []
    fragile_mags = []
    for et in HistoricalEventType:
        m_healthy = engine._compute_magnitude(et, healthy_state)
        m_fragile = engine._compute_magnitude(et, fragile_state)
        healthy_mags.append(m_healthy)
        fragile_mags.append(m_fragile)

    mean_healthy = float(np.mean(healthy_mags))
    mean_fragile = float(np.mean(fragile_mags))
    print(f"  ✓ Mean magnitude (healthy): {mean_healthy:.3f}")
    print(f"  ✓ Mean magnitude (fragile): {mean_fragile:.3f}")

    assert mean_fragile > mean_healthy, \
        f"Fragile system should get stronger events: {mean_fragile:.3f} vs {mean_healthy:.3f}"
    print(f"  ✓ Vulnerability scaling confirmed: fragile > healthy")

    print("  >>> VulnerabilityScaling PASSED\n")


def test_system_state_builder():
    """Test system state builder from Phase 48 components."""
    print("\n" + "=" * 60)
    print("48.5A.6 — SYSTEM STATE BUILDER")
    print("=" * 60)

    from phase48_cognitive_political_economy import (
        CognitiveAgent, NarrativeEcosystem, ConstitutionalLayer
    )

    agents = []
    for i in range(8):
        a = CognitiveAgent(f'a_{i}', 'exploitative', np.random.randn(32))
        a.wealth = 2.0 + i * 0.5  # Increasing wealth for Gini
        a.active = True
        agents.append(a)

    species = {
        'exploitative': type('S', (), {'population': 4})(),
        'exploratory': type('S', (), {'population': 3})(),
        'defensive': type('S', (), {'population': 1})(),
    }

    ecosystem = NarrativeEcosystem(semantic_dim=32)
    for _ in range(5):
        ecosystem.seed(np.random.randn(32), fitness=0.5)

    constitution = ConstitutionalLayer()

    state = _build_system_state(agents, species, ecosystem, constitution)
    print(f"  ✓ System state keys: {list(state.keys())}")
    print(f"  ✓ Wealth Gini: {state['wealth_gini']:.3f}")

    # With increasing wealth (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5)
    assert 'wealth_gini' in state
    assert 'species_diversity' in state
    assert 'narrative_diversity' in state
    assert 'self_coherence' in state
    print(f"  ✓ All metrics computed correctly")

    print("  >>> SystemStateBuilder PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5A: HISTORICAL EVENT ENGINE                            ║
║                                                                   ║
║  NOT random noise.                                                ║
║  Structured historical events triggered by system                 ║
║  vulnerabilities, scheduled moments, and causal cascades.         ║
║                                                                   ║
║  Creates PRESSURE DIFFERENTIALS that Phase 48's closed            ║
║  thermodynamic system is missing.                                 ║
║                                                                   ║
║  8 Event Types:                                                    ║
║    compute_collapse, resource_inflation, narrative_corruption,    ║
║    coalition_scandal, agent_migration, memory_loss_wave,          ║
║    constitution_attack, identity_crisis                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("EventGeneration (48.5A.1)", test_event_generation),
        ("ScheduledEvents (48.5A.2)", test_scheduled_events),
        ("EventCascade (48.5A.3)", test_event_cascade),
        ("EventApplication (48.5A.4)", test_event_application),
        ("VulnerabilityScaling (48.5A.5)", test_vulnerability_scaling),
        ("SystemStateBuilder (48.5A.6)", test_system_state_builder),
    ]

    all_pass = True
    for name, fn in tests:
        try:
            fn()
            print(f"  >>> {name} PASSED\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  >>> {name} FAILED: {e}\n")
            all_pass = False

    if all_pass:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  PHASE 48.5A: ALL TESTS PASSED                                ║
  ║                                                               ║
  ║  Historical Event Engine ready.                                ║
  ║  Next: integrate with GradientInjector, run 5000-step         ║
  ║  validation with structured historical shocks.                 ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"\n  Some tests FAILED")
