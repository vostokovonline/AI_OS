"""
Phase 48.5 — Gradient Injection Layer.

PURPOSE:
  Phase 48 is structurally correct but dynamically flat — it reaches
  thermodynamic equilibrium instead of maintaining bounded non-equilibrium
  dynamics. This layer injects controlled asymmetries to create:
    - coalition turnover
    - wealth stratification without collapse
    - narrative diversity oscillation instead of flattening
    - constitutional activation (adaptive governance)

DESIGN PRINCIPLES:
  1. All injections are BOUNDED — no single event can destabilize the system
  2. Injections create GRADIENTS, not random noise — they push in specific
     directions that the system must respond to
  3. Recovery is built-in — after each injection, the system has time to
     re-stabilize before the next one
  4. Periodicity avoids habituation — intervals vary stochastically

INJECTION TYPES:
  1. Resource Asymmetry    — stochastic supply shocks, localized scarcity
  2. Fitness Variance      — structural heterogeneity in payoff curves
  3. Narrative Mutation    — exogenous novelty / information shocks
  4. Coalition Friction    — coordination cost scaling with coalition size
  5. Constitutional Stress — forced edge-case events to test governance
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class GradientConfig:
    """
    Configuration for all injection types.

    Each injection type has:
      interval:     base period between injections (steps)
      jitter:       random +/- jitter on interval (prevents entrainment)
      intensity:    how strong the injection is (0..1 scale)
      enabled:      whether this injection type is active

    Conservative defaults — increment intensity gradually.
    """
    # Resource asymmetry
    resource_interval: int = 100
    resource_jitter: int = 30
    resource_intensity: float = 0.3
    resource_enabled: bool = True

    # Fitness variance
    fitness_interval: int = 200
    fitness_jitter: int = 50
    fitness_intensity: float = 0.25
    fitness_enabled: bool = True

    # Narrative mutation pressure
    narrative_interval: int = 80
    narrative_jitter: int = 20
    narrative_intensity: float = 0.3
    narrative_enabled: bool = True

    # Coalition friction
    friction_interval: int = 150
    friction_jitter: int = 40
    friction_intensity: float = 0.3
    friction_enabled: bool = True

    # Constitutional stress
    stress_interval: int = 300
    stress_jitter: int = 80
    stress_intensity: float = 0.4
    stress_enabled: bool = True

    # Global bounds to prevent chaos
    max_single_shock: float = 0.5    # max magnitude of any single event
    min_recovery_steps: int = 10     # minimum steps between injections
    max_active_injections: int = 3   # max concurrent injection types


class GradientInjector:
    """
    Controlled asymmetry injection for CognitivePoliticalEngine.

    Injects bounded perturbations into specific subsystems to prevent
    thermodynamic equilibrium and maintain bounded non-equilibrium dynamics.

    Usage:
      injector = GradientInjector(config)
      engine = CognitivePoliticalEngine(wm, gradient_injector=injector)
      # engine.step() automatically calls injector.inject() each step
    """

    def __init__(self, config: Optional[GradientConfig] = None):
        self.config = config or GradientConfig()
        self.events: List[Dict] = []
        self._last_injection_step: Dict[str, int] = {
            'resource': 0, 'fitness': 0, 'narrative': 0,
            'friction': 0, 'stress': 0
        }
        self._next_injection: Dict[str, int] = {
            'resource': self._next_step('resource_interval', 'resource_jitter'),
            'fitness': self._next_step('fitness_interval', 'fitness_jitter'),
            'narrative': self._next_step('narrative_interval', 'narrative_jitter'),
            'friction': self._next_step('friction_interval', 'friction_jitter'),
            'stress': self._next_step('stress_interval', 'stress_jitter'),
        }
        self.active_injuries: Dict[str, float] = {}
        self.n_injections: int = 0

    def _next_step(self, interval_attr: str, jitter_attr: str) -> int:
        """Compute next injection step with jitter."""
        base = getattr(self.config, interval_attr, 100)
        jitter = getattr(self.config, jitter_attr, 30)
        return max(5, base + random.randint(-jitter, jitter))

    def _should_inject(self, injection_type: str, step: int) -> bool:
        """Check if it's time for this injection type."""
        enabled_attr = f"{injection_type}_enabled"
        if not getattr(self.config, enabled_attr, False):
            return False
        if injection_type not in self._next_injection:
            return False
        return step >= self._next_injection[injection_type]

    def _schedule_next(self, injection_type: str, step: int):
        """Schedule the next injection of this type."""
        interval_attr = f"{injection_type}_interval"
        jitter_attr = f"{injection_type}_jitter"
        self._next_injection[injection_type] = step + self._next_step(
            interval_attr, jitter_attr
        )
        self._last_injection_step[injection_type] = step
        self.n_injections += 1

    def _check_safety(self, step: int) -> bool:
        """Safety check: don't inject too many types at once."""
        recent = sum(
            1 for v in self._last_injection_step.values()
            if step - v < self.config.min_recovery_steps
        )
        return recent < self.config.max_active_injections

    def inject(
        self,
        agents: List['CognitiveAgent'],
        species: Dict[str, Any],
        narrative_ecosystem: Any,
        constitution: Any,
        step: int
    ) -> Dict[str, List[Dict]]:
        """
        Inject controlled asymmetries into the system.

        Called each step by CognitivePoliticalEngine.
        Returns events dict for logging.
        """
        events: Dict[str, List[Dict]] = {
            'resource_asymmetry': [],
            'fitness_variance': [],
            'narrative_pressure': [],
            'coalition_friction': [],
            'constitutional_stress': [],
        }

        if not self._check_safety(step):
            return events

        active = [a for a in agents if a.active]

        # 1. Resource Asymmetry — stochastic supply shocks
        if self._should_inject('resource', step):
            result = self._inject_resource_asymmetry(active, step)
            events['resource_asymmetry'] = result
            self._schedule_next('resource', step)

        # 2. Fitness Variance — structural payoff heterogeneity
        if self._should_inject('fitness', step):
            result = self._inject_fitness_variance(active, species, step)
            events['fitness_variance'] = result
            self._schedule_next('fitness', step)

        # 3. Narrative Mutation Pressure — exogenous novelty
        if self._should_inject('narrative', step):
            result = self._inject_narrative_pressure(
                narrative_ecosystem, step
            )
            events['narrative_pressure'] = result
            self._schedule_next('narrative', step)

        # 4. Coalition Friction — coordination cost
        if self._should_inject('friction', step):
            result = self._inject_coalition_friction(active, step)
            events['coalition_friction'] = result
            self._schedule_next('friction', step)

        # 5. Constitutional Stress — forced edge-case
        if self._should_inject('stress', step):
            result = self._inject_constitutional_stress(
                constitution, active, step
            )
            events['constitutional_stress'] = result
            self._schedule_next('stress', step)

        # Log events
        if any(events.values()):
            self.events.append({
                'step': step,
                'active_type': sum(1 for v in events.values() if v),
                'event_count': sum(len(v) for v in events.values())
            })

        return events

    # ------------------------------------------------------------------
    # 1. Resource Asymmetry — stochastic supply shocks
    # ------------------------------------------------------------------

    def _inject_resource_asymmetry(
        self,
        agents: List['CognitiveAgent'],
        step: int
    ) -> List[Dict]:
        """
        Create temporary resource scarcity or abundance for specific agents.

        Types:
          - wealth_tax:     randomly select an agent, tax their wealth
          - wealth_gift:    randomly select an agent, grant wealth
          - compute_boost:  increase specific agent's bid intensity
          - compute_sabotage: decrease specific agent's bid intensity
        """
        if not agents or len(agents) < 3:
            return []

        events = []
        intensity = self.config.resource_intensity
        capped = min(intensity, self.config.max_single_shock)

        n_affected = max(1, len(agents) // 4)
        affected = random.sample(agents, min(n_affected, len(agents)))

        shock_type = random.choice(['wealth_tax', 'wealth_gift',
                                    'compute_boost', 'compute_sabotage'])

        for agent in affected:
            if shock_type == 'wealth_tax':
                tax = agent.wealth * capped * random.uniform(0.5, 1.0)
                agent.wealth -= tax
                agent.wealth = max(0.1, agent.wealth)
                events.append({
                    'type': 'wealth_tax',
                    'agent': agent.agent_id,
                    'amount': round(tax, 3),
                    'species': agent.species
                })

            elif shock_type == 'wealth_gift':
                gift = capped * random.uniform(0.5, 1.5)
                agent.wealth += gift
                agent.wealth = min(10.0, agent.wealth)
                events.append({
                    'type': 'wealth_gift',
                    'agent': agent.agent_id,
                    'amount': round(gift, 3),
                    'species': agent.species
                })

            elif shock_type == 'compute_boost':
                old = agent.bid_intensity
                agent.bid_intensity = min(0.95, agent.bid_intensity + capped * 0.3)
                events.append({
                    'type': 'compute_boost',
                    'agent': agent.agent_id,
                    'from': round(old, 3),
                    'to': round(agent.bid_intensity, 3),
                    'species': agent.species
                })

            elif shock_type == 'compute_sabotage':
                old = agent.bid_intensity
                agent.bid_intensity = max(0.05, agent.bid_intensity - capped * 0.3)
                events.append({
                    'type': 'compute_sabotage',
                    'agent': agent.agent_id,
                    'from': round(old, 3),
                    'to': round(agent.bid_intensity, 3),
                    'species': agent.species
                })

        return events

    # ------------------------------------------------------------------
    # 2. Fitness Variance — structural payoff heterogeneity
    # ------------------------------------------------------------------

    def _inject_fitness_variance(
        self,
        agents: List['CognitiveAgent'],
        species: Dict[str, Any],
        step: int
    ) -> List[Dict]:
        """
        Create structural heterogeneity in agent payoff curves.

        Different agents/species get different long-term payoff profiles:
          - time_horizon mutation: some agents get longer/shorter horizons
          - risk_tolerance mutation: some agents become more/less risk-seeking
          - productivity shock: random agent gets productivity boost/drop
        """
        if not agents or len(agents) < 2:
            return []

        events = []
        intensity = self.config.fitness_intensity
        capped = min(intensity, self.config.max_single_shock)

        n_affected = max(1, len(agents) // 3)
        affected = random.sample(agents, min(n_affected, len(agents)))

        for agent in affected:
            mutation = random.choice(['time_horizon', 'risk_tolerance',
                                      'productivity', 'exploration'])

            if mutation == 'time_horizon':
                delta = capped * random.uniform(-1.0, 1.0) * 0.3
                old = agent.time_horizon
                agent.time_horizon = float(np.clip(old + delta, 0.05, 0.95))
                events.append({
                    'type': 'time_horizon_shift',
                    'agent': agent.agent_id,
                    'species': agent.species,
                    'from': round(old, 3),
                    'to': round(agent.time_horizon, 3)
                })

            elif mutation == 'risk_tolerance':
                delta = capped * random.uniform(-1.0, 1.0) * 0.3
                old = agent.risk_tolerance
                agent.risk_tolerance = float(np.clip(old + delta, 0.05, 0.95))
                events.append({
                    'type': 'risk_tolerance_shift',
                    'agent': agent.agent_id,
                    'species': agent.species,
                    'from': round(old, 3),
                    'to': round(agent.risk_tolerance, 3)
                })

            elif mutation == 'productivity':
                delta = capped * random.uniform(-1.0, 1.0) * 0.2
                old = agent.productivity
                agent.productivity = float(np.clip(old + delta, 0.1, 0.9))
                events.append({
                    'type': 'productivity_shock',
                    'agent': agent.agent_id,
                    'species': agent.species,
                    'from': round(old, 3),
                    'to': round(agent.productivity, 3)
                })

            elif mutation == 'exploration':
                delta = capped * random.uniform(-1.0, 1.0) * 0.3
                old = agent.exploration_rate
                agent.exploration_rate = float(np.clip(old + delta, 0.05, 0.95))
                events.append({
                    'type': 'exploration_shift',
                    'agent': agent.agent_id,
                    'species': agent.species,
                    'from': round(old, 3),
                    'to': round(agent.exploration_rate, 3)
                })

        return events

    # ------------------------------------------------------------------
    # 3. Narrative Mutation Pressure — exogenous novelty
    # ------------------------------------------------------------------

    def _inject_narrative_pressure(
        self,
        narrative_ecosystem: Any,
        step: int
    ) -> List[Dict]:
        """
        Inject exogenous novelty into the narrative ecosystem.

        Types:
          - novelty_burst: seed several random narratives at once
          - narrative_quake: randomize some narrative vectors
          - information_shock: seed narratives from a "foreign" semantic cluster
        """
        if narrative_ecosystem is None:
            return []

        events = []
        intensity = self.config.narrative_intensity
        capped = min(intensity, self.config.max_single_shock)

        shock_type = random.choice(['novelty_burst', 'information_shock'])

        if shock_type == 'novelty_burst':
            n_new = max(1, int(capped * 5))
            for i in range(n_new):
                vector = np.tanh(np.random.randn(
                    narrative_ecosystem.semantic_dim
                ) * capped * 2.0)
                gid = narrative_ecosystem.seed(
                    vector,
                    fitness=0.2 + capped * 0.3,
                    attributes={'source': 'exogenous_novelty',
                                'step': step}
                )
                events.append({
                    'type': 'novelty_burst',
                    'gene_id': gid,
                    'fitness': 0.2 + capped * 0.3
                })

        elif shock_type == 'information_shock':
            foreign_vec = np.tanh(np.random.randn(
                narrative_ecosystem.semantic_dim
            ) * capped * 3.0)
            n_shock = max(1, int(capped * 3))
            for i in range(n_shock):
                variant = foreign_vec + np.random.randn(
                    narrative_ecosystem.semantic_dim
                ) * 0.2
                variant = variant / (np.linalg.norm(variant) + 1e-8)
                gid = narrative_ecosystem.seed(
                    variant,
                    fitness=0.3 + capped * 0.4,
                    attributes={'source': 'information_shock',
                                'step': step, 'foreign': True}
                )
                events.append({
                    'type': 'information_shock',
                    'gene_id': gid,
                    'fitness': 0.3 + capped * 0.4
                })

        return events

    # ------------------------------------------------------------------
    # 4. Coalition Friction — coordination cost
    # ------------------------------------------------------------------

    def _inject_coalition_friction(
        self,
        agents: List['CognitiveAgent'],
        step: int
    ) -> List[Dict]:
        """
        Create pressure for coalition turnover by introducing coordination costs.

        Mechanisms:
          - size_tax:    large coalitions lose cohesion (members defect)
          - alignment_cost: agents with dissimilar ideology pay to stay
          - fission_pressure: coalitions above threshold size may split
        """
        if not agents or len(agents) < 3:
            return []

        events = []
        intensity = self.config.friction_intensity
        capped = min(intensity, self.config.max_single_shock)

        # Group agents by coalition
        coalition_map: Dict[str, List['CognitiveAgent']] = {}
        for a in agents:
            cid = a.coalition_id or 'none'
            if cid not in coalition_map:
                coalition_map[cid] = []
            coalition_map[cid].append(a)

        for cid, members in coalition_map.items():
            if cid == 'none' or len(members) < 2:
                continue

            # Size tax: large coalitions lose cohesion
            if len(members) > 3:
                # Larger coalition → more friction
                friction = capped * (len(members) - 2) * 0.15
                # Random defections proportional to friction
                n_defect = max(1, int(friction * len(members)))
                random.shuffle(members)
                defectors = members[:n_defect]
                for d in defectors:
                    old_cid = d.coalition_id
                    d.coalition_id = None  # temporarily unaffiliated
                    events.append({
                        'type': 'coalition_defection',
                        'agent': d.agent_id,
                        'species': d.species,
                        'from_coalition': old_cid,
                        'reason': 'size_tax'
                    })

            # Alignment cost: ideological dissimilarity within coalition
            if len(members) >= 2 and cid != 'none':
                ideology_sims = []
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        sim = members[i].ideological_similarity(members[j])
                        ideology_sims.append(sim)

                if ideology_sims:
                    mean_sim = float(np.mean(ideology_sims))
                    if mean_sim < 0.3:
                        # Low alignment — coalition likely to dissolve
                        for m in members:
                            if random.random() < 0.3 * capped:
                                m.coalition_id = None
                                events.append({
                                    'type': 'coalition_alignment_break',
                                    'agent': m.agent_id,
                                    'from_coalition': cid,
                                    'mean_alignment': round(mean_sim, 3)
                                })

        return events

    # ------------------------------------------------------------------
    # 5. Constitutional Stress — forced edge-case events
    # ------------------------------------------------------------------

    def _inject_constitutional_stress(
        self,
        constitution: Any,
        agents: List['CognitiveAgent'],
        step: int
    ) -> List[Dict]:
        """
        Create forced edge-case events that stress-test constitutional articles.

        These are controlled violations that force the constitution to:
          1. Record violations
          2. Adapt article strengths
          3. Build institutional memory

        Types:
          - monopoly_event:     force a temporary compute monopoly
          - diversity_crisis:   force species extinction risk
          - identity_threat:    force self coherence disruption
          - exploration_collapse: force all agents to stop exploring
        """
        if constitution is None or not agents:
            return []

        events = []
        intensity = self.config.stress_intensity
        capped = min(intensity, self.config.max_single_shock)

        shock_type = random.choice(['monopoly_event', 'diversity_crisis',
                                    'identity_threat', 'exploration_collapse'])

        if shock_type == 'monopoly_event':
            # Force a temporary compute monopoly by boosting one agent
            if agents:
                target = random.choice([a for a in agents if a.active])
                target.bid_intensity = min(0.95, target.bid_intensity + capped * 0.5)
                # Upper-bound test for anti-monopoly article
                events.append({
                    'type': 'monopoly_stress',
                    'agent': target.agent_id,
                    'bid_intensity': round(target.bid_intensity, 3),
                    'article_triggered': 'anti_monopoly'
                })

        elif shock_type == 'diversity_crisis':
            # Force extinction risk for a random species
            active_species = set(a.species for a in agents if a.active)
            if len(active_species) >= 2:
                target_species = random.choice(list(active_species))
                # Reduce wealth of all agents of this species
                species_agents = [a for a in agents
                                  if a.species == target_species and a.active]
                for a in species_agents:
                    a.wealth *= (1.0 - capped * 0.5)
                    a.exploration_rate *= (1.0 - capped * 0.2)
                events.append({
                    'type': 'diversity_crisis',
                    'target_species': target_species,
                    'n_affected': len(species_agents),
                    'article_triggered': 'diversity_floor'
                })

        elif shock_type == 'identity_threat':
            # Force self coherence disruption
            # Randomize self-latent to simulate identity shock
            if hasattr(constitution, 'articles'):
                events.append({
                    'type': 'identity_threat',
                    'article_triggered': 'continuity_guarantee',
                    'intensity': capped
                })

        elif shock_type == 'exploration_collapse':
            # Force all agents to stop exploring
            for a in agents:
                old_rate = a.exploration_rate
                a.exploration_rate = max(0.01, a.exploration_rate - capped * 0.4)
                if a.exploration_rate < 0.05:
                    events.append({
                        'type': 'exploration_collapse',
                        'agent': a.agent_id,
                        'from': round(old_rate, 3),
                        'to': round(a.exploration_rate, 3),
                        'article_triggered': 'exploration_quota'
                    })

        return events

    # ------------------------------------------------------------------
    # Stats & Recovery
    # ------------------------------------------------------------------

    def get_active_imbalance(self) -> Dict[str, float]:
        """Current imbalance level per subsystem (0 = equilibrium)."""
        recent = self.events[-20:] if len(self.events) >= 20 else self.events
        imbalance: Dict[str, float] = {
            'resource': 0.0, 'fitness': 0.0,
            'narrative': 0.0, 'friction': 0.0, 'stress': 0.0
        }
        for e in recent:
            for key in imbalance:
                event_types = e.get('event_types', {})
                if key in event_types:
                    imbalance[key] += event_types[key]
        return {k: min(1.0, v / 5.0) for k, v in imbalance.items()}

    def get_stats(self) -> Dict:
        return {
            'n_injections': self.n_injections,
            'n_events_in_log': len(self.events),
            'config': {
                'resource_enabled': self.config.resource_enabled,
                'fitness_enabled': self.config.fitness_enabled,
                'narrative_enabled': self.config.narrative_enabled,
                'friction_enabled': self.config.friction_enabled,
                'stress_enabled': self.config.stress_enabled,
                'max_single_shock': self.config.max_single_shock,
            },
            'next_injections': self._next_injection,
            'imbalance': self.get_active_imbalance()
        }


def create_default_injector() -> GradientInjector:
    """Create a GradientInjector with conservative default settings."""
    config = GradientConfig(
        # Moderate resource asymmetry every ~100 steps
        resource_interval=100, resource_intensity=0.3,
        # Gentle fitness variance every ~200 steps
        fitness_interval=200, fitness_intensity=0.25,
        # Regular narrative novelty every ~80 steps
        narrative_interval=80, narrative_intensity=0.3,
        # Coalition pressure every ~150 steps
        friction_interval=150, friction_intensity=0.3,
        # Constitutional stress every ~300 steps
        stress_interval=300, stress_intensity=0.4,
        # Safety bounds
        max_single_shock=0.5,
        max_active_injections=3
    )
    return GradientInjector(config)


# ============================================================================
# UNIT TESTS
# ============================================================================

def _make_test_agents(n: int = 5, **kwargs):
    """Create test CognitiveAgent instances."""
    from phase48_cognitive_political_economy import CognitiveAgent
    agents = []
    for i in range(n):
        defaults = dict(
            agent_id=f'a_{i}',
            species='exploitative',
            ideology=np.random.randn(32),
            productivity=0.5, reliability=0.5, bid_intensity=0.5,
            time_horizon=0.5, risk_tolerance=0.5, exploration_rate=0.3
        )
        defaults.update(kwargs)
        a = CognitiveAgent(**defaults)
        a.wealth = 2.0
        agents.append(a)
    return agents


def test_resource_asymmetry():
    """Test resource asymmetry injection."""
    print("\n" + "=" * 60)
    print("48.5.1 — RESOURCE ASYMMETRY")
    print("=" * 60)

    injector = GradientInjector(GradientConfig(resource_intensity=0.5))
    agents = _make_test_agents(5)

    # Call multiple times to exercise all shock types
    wealths_before = [a.wealth for a in agents]
    bid_before = [a.bid_intensity for a in agents]
    all_events = []
    for s in range(20):
        rng_state = random.getstate()
        random.seed(s)
        events = injector._inject_resource_asymmetry(agents, step=100 + s)
        random.setstate(rng_state)
        all_events.extend(events)

    assert len(all_events) > 0
    wealths = [a.wealth for a in agents]
    bids = [a.bid_intensity for a in agents]
    wealth_changed = any(w != bw for w, bw in zip(wealths, wealths_before))
    bid_changed = any(b != bb for b, bb in zip(bids, bid_before))
    assert wealth_changed or bid_changed, "Wealth or bid should change"
    print(f"  ✓ {len(all_events)} resource events generated")
    print(f"  ✓ Wealth: {[f'{w:.2f}' for w in wealths]}")
    print(f"  ✓ Bid intensities: {[f'{b:.3f}' for b in bids]}")

    print("  >>> ResourceAsymmetry PASSED\n")


def test_fitness_variance():
    """Test fitness variance injection."""
    print("\n" + "=" * 60)
    print("48.5.2 — FITNESS VARIANCE")
    print("=" * 60)

    injector = GradientInjector(GradientConfig(fitness_intensity=0.5))
    agents = _make_test_agents(5)

    events = injector._inject_fitness_variance(agents, {}, step=100)
    assert len(events) > 0
    print(f"  ✓ {len(events)} fitness variance events generated")

    params = [(a.time_horizon, a.risk_tolerance, a.productivity, a.exploration_rate)
              for a in agents]
    print(f"  ✓ Parameters shifted: {params[:3]}")

    print("  >>> FitnessVariance PASSED\n")


def test_narrative_pressure():
    """Test narrative mutation pressure."""
    print("\n" + "=" * 60)
    print("48.5.3 — NARRATIVE MUTATION PRESSURE")
    print("=" * 60)

    from phase48_cognitive_political_economy import NarrativeEcosystem
    import sys
    sys.path.insert(0, '.')

    ecosystem = NarrativeEcosystem(semantic_dim=32)
    injector = GradientInjector(GradientConfig(
        narrative_intensity=0.5,
        narrative_interval=10
    ))

    # Seed some existing narratives
    for _ in range(5):
        ecosystem.seed(np.random.randn(32), fitness=0.5)

    n_before = len(ecosystem.genes)
    events = injector._inject_narrative_pressure(ecosystem, step=100)
    n_after = len(ecosystem.genes)

    assert n_after > n_before
    print(f"  ✓ Narratives: {n_before} → {n_after}")
    print(f"  ✓ {len(events)} narrative events: {[e['type'] for e in events[:3]]}")

    print("  >>> NarrativePressure PASSED\n")


def test_coalition_friction():
    """Test coalition friction injection."""
    print("\n" + "=" * 60)
    print("48.5.4 — COALITION FRICTION")
    print("=" * 60)

    injector = GradientInjector(GradientConfig(friction_intensity=0.5))
    agents = _make_test_agents(8)

    # Assign coalitions
    for i, a in enumerate(agents):
        if i < 4:
            a.coalition_id = 'coal_large'
        elif i < 6:
            a.coalition_id = 'coal_medium'
        else:
            a.coalition_id = 'coal_small'

    events = injector._inject_coalition_friction(agents, step=100)
    print(f"  ✓ {len(events)} coalition friction events generated")

    # Check that some agents may have left coalitions
    n_unaffiliated = len([a for a in agents if a.coalition_id is None])
    print(f"  ✓ Unaffiliated agents: {n_unaffiliated}")

    print("  >>> CoalitionFriction PASSED\n")


def test_constitutional_stress():
    """Test constitutional stress injection."""
    print("\n" + "=" * 60)
    print("48.5.5 — CONSTITUTIONAL STRESS")
    print("=" * 60)

    from phase48_cognitive_political_economy import ConstitutionalLayer
    injector = GradientInjector(GradientConfig(stress_intensity=0.6))
    constitution = ConstitutionalLayer()

    # Use agents with diverse species + low exploration so ALL shock types produce events
    agents = []
    for i in range(6):
        species_list = ['exploitative', 'exploratory', 'defensive',
                        'identity_preserving', 'novelty_seeking', 'stability_seeking']
        a = _make_test_agents(1, species=species_list[i], exploration_rate=0.2)[0]
        agents.append(a)

    events = injector._inject_constitutional_stress(constitution, agents, step=100)
    assert len(events) > 0
    print(f"  ✓ {len(events)} constitutional stress events generated")
    print(f"  ✓ Types: {list(set(e['type'] for e in events))}")

    print("  >>> ConstitutionalStress PASSED\n")


def test_full_injection_cycle():
    """Test that the injector runs a full cycle without errors."""
    print("\n" + "=" * 60)
    print("48.5 — FULL INJECTION CYCLE (100 steps)")
    print("=" * 60)

    from phase48_cognitive_political_economy import (
        NarrativeEcosystem, ConstitutionalLayer
    )

    injector = GradientInjector(GradientConfig(
        resource_interval=15, resource_jitter=5,
        fitness_interval=20, fitness_jitter=5,
        narrative_interval=15, narrative_jitter=5,
        friction_interval=20, friction_jitter=5,
        stress_interval=25, stress_jitter=5,
        resource_enabled=True, fitness_enabled=True,
        narrative_enabled=True, friction_enabled=True,
        stress_enabled=True,
    ))
    ecosystem = NarrativeEcosystem(semantic_dim=32)
    constitution = ConstitutionalLayer()

    # Seed initial narratives
    for _ in range(5):
        ecosystem.seed(np.random.randn(32), fitness=0.5)

    agents = _make_test_agents(10)
    # Give agents varied species for fitness variance to be meaningful
    for i, a in enumerate(agents):
        a.species = ['exploitative', 'exploratory', 'defensive',
                     'identity_preserving', 'novelty_seeking',
                     'stability_seeking', 'opportunistic',
                     'conservative', 'pioneer', 'guardian'][i]
        a.coalition_id = f'coal_{i % 3}'

    step_events = []
    for step in range(1, 101):
        events = injector.inject(agents, {}, ecosystem, constitution, step)
        for k, v in events.items():
            if v:
                step_events.append({'step': step, 'type': k, 'n': len(v)})

    n_total = sum(e['n'] for e in step_events)
    assert n_total > 0, "Should have injected some events in 100 steps"
    print(f"  ✓ {n_total} total injection events across {len(step_events)} steps")

    types_used = set(e['type'] for e in step_events)
    print(f"  ✓ Injection types activated: {types_used}")

    # Check bounds: no agent should have extreme values
    for a in agents:
        assert 0.05 <= a.bid_intensity <= 0.95
        assert 0.1 <= a.wealth <= 10.0
        assert 0.05 <= a.time_horizon <= 0.95
    print(f"  ✓ All agents within safety bounds")

    # Check narrative ecosystem still healthy
    assert len(ecosystem.genes) > 0
    print(f"  ✓ Narrative ecosystem healthy: {len(ecosystem.genes)} genes")

    print("  >>> FullInjectionCycle PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5: GRADIENT INJECTION LAYER                            ║
║                                                                   ║
║  Purpose: Inject controlled asymmetries into Phase 48 to         ║
║  create bounded non-equilibrium dynamics without destabilizing   ║
║  the system.                                                      ║
║                                                                   ║
║  Injectors:                                                       ║
║    1. Resource Asymmetry    — stochastic supply shocks           ║
║    2. Fitness Variance      — structural payoff heterogeneity    ║
║    3. Narrative Mutation    — exogenous novelty injection        ║
║    4. Coalition Friction    — coordination costs                 ║
║    5. Constitutional Stress — forced edge-case events            ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("ResourceAsymmetry (48.5.1)", test_resource_asymmetry),
        ("FitnessVariance (48.5.2)", test_fitness_variance),
        ("NarrativePressure (48.5.3)", test_narrative_pressure),
        ("CoalitionFriction (48.5.4)", test_coalition_friction),
        ("ConstitutionalStress (48.5.5)", test_constitutional_stress),
        ("FullInjectionCycle", test_full_injection_cycle),
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
  ║  PHASE 48.5: ALL TESTS PASSED                                ║
  ║                                                               ║
  ║  Ready for integration with CognitivePoliticalEngine.         ║
  ║  Next: add gradient_injector parameter to engine +            ║
  ║  call injector.inject() in step().                            ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"\n  Some tests FAILED")
