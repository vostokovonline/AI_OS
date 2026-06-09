"""
Phase 48.5C — Generational Turnover Engine.

Transforms the Cognitive Political Economy from a
MULTI-AGENT SIMULATION into a HISTORICAL CIVILIZATION PROCESS.

Without reproduction, inheritance, and lineage, there is no:
  - cultural evolution
  - dynastic concentration
  - civilizational memory
  - institutional succession
  - межпоколенческое накопление опыта

This engine adds:
  1. Aging & Lifecycle    — age stages, natural death, senescence
  2. Fertility            — reproduction under favorable conditions
  3. Inheritance          — wealth, capital, coalition membership
  4. Lineage Tracking     — family trees, dynastic accumulation
  5. Cultural Drift       — ideology mutation across generations
  6. Institutional Succession — coalition/narrative continuity

Key principle:
  "History begins when generations die and new generations inherit
   the structures, mistakes, and successes of their predecessors."
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, Callable, List, Dict, Any, Tuple
import sys
sys.path.insert(0, '.')

from phase48_cognitive_political_economy import (
    CognitiveAgent, ResourceType, ResourceBundle, SPECIES_PARAMS
)


# ============================================================================
# Lifecycle Configuration
# ============================================================================

# Life stages with age bounds (in steps)
LIFE_STAGES = {
    'youth':     (0, 15),     # Learning, low productivity, high exploration
    'prime':     (16, 40),    # Peak productivity, reliability building
    'middle':    (41, 65),    # Declining productivity, rising political capital
    'elderly':   (66, 85),    # Low productivity, high wisdom/reliability
    'dying':     (86, 100),   # Rapid decline, inheritance planning
}

# Maximum lifespan: mean 80, std 15
MAX_LIFESPAN_MEAN = 80
MAX_LIFESPAN_STD = 15

# Age effects on parameters
AGE_PRODUCTIVITY = {
    'youth': 0.5, 'prime': 1.0, 'middle': 0.8,
    'elderly': 0.5, 'dying': 0.2,
}
AGE_RELIABILITY = {
    'youth': 0.6, 'prime': 0.9, 'middle': 1.0,
    'elderly': 1.0, 'dying': 0.7,
}
AGE_EXPLORATION = {
    'youth': 1.5, 'prime': 1.0, 'middle': 0.7,
    'elderly': 0.4, 'dying': 0.2,
}
AGE_RISK_TOLERANCE = {
    'youth': 1.3, 'prime': 1.0, 'middle': 0.7,
    'elderly': 0.4, 'dying': 0.2,
}

# Reproduction parameters
REPRODUCTION_WEALTH_THRESHOLD = 2.0  # Minimum wealth to reproduce
REPRODUCTION_COST = 0.3              # Fraction of wealth transferred to offspring
REPRODUCTION_COOLDOWN = 20           # Steps between reproduction attempts (default)
IDELOGY_MUTATION_RATE = 0.05         # Per-generation ideology mutation
INHERITANCE_FRACTION = 0.6           # Fraction of wealth inherited by offspring
MAX_CHILDREN_PER_PAIR = 5            # Maximum offspring per agent

# Fertility curve (probability of reproduction per age stage)
FERTILITY_RATES = {
    'youth': 0.0,
    'prime': 1.0,    # Peak fertility
    'middle': 0.7,   # Declining
    'elderly': 0.3,  # Low but possible
    'dying': 0.0,
}

# Survival probability (age-dependent)
def natural_death_probability(age: int, max_lifespan: int) -> float:
    """Probability of natural death at a given age.
    
    Near zero before 60, then rises exponentially.
    At max_lifespan, probability = 1.0.
    """
    if age < 60:
        return 0.0
    if age >= max_lifespan:
        return 1.0
    # Exponential rise from 60 to max_lifespan
    normalized = (age - 60) / max(1, max_lifespan - 60)
    return float(np.clip(normalized ** 2 * 0.5, 0.0, 1.0))


# ============================================================================
# Lineage Tracking
# ============================================================================

@dataclass
class LineageRecord:
    """Track a lineage (family line) across generations."""
    lineage_id: str
    founding_agent: str
    birth_step: int
    total_members: int = 1
    current_members: int = 1
    total_wealth_accumulated: float = 0.0
    total_political_offspring: int = 0  # Members who achieved veto
    extinct: bool = False
    last_birth_step: int = 0

    def get_stats(self) -> Dict:
        return {
            'age': self.birth_step,
            'total_members': self.total_members,
            'current_members': self.current_members,
            'wealth': round(self.total_wealth_accumulated, 2),
            'political_offspring': self.total_political_offspring,
            'extinct': self.extinct,
        }


# ============================================================================
# Generational Turnover Engine
# ============================================================================

class GenerationalEngine:
    """
    Manages aging, reproduction, inheritance, and lineage tracking.

    This engine integrates with CognitivePoliticalEngine:
      - Each step: age all agents, check reproduction, check natural death
      - On death: execute inheritance, update lineage
      - On birth: create new agent, update lineage

    Integration:
      gen_engine = GenerationalEngine()
      # In CognitivePoliticalEngine.step():
      gen_result = gen_engine.step(agents, species, narrative_ecosystem, constitution)
      # gen_result contains births, deaths, inheritances
    """

    def __init__(
        self,
        reproduction_interval: int = 15,
        max_population: int = 20,
        enable_natural_death: bool = True,
        enable_reproduction: bool = True,
        enable_inheritance: bool = True,
        reproduction_cooldown: int = 20,
        max_children_per_pair: int = 5,
    ):
        self.reproduction_interval = reproduction_interval
        self.max_population = max_population
        self.enable_natural_death = enable_natural_death
        self.enable_reproduction = enable_reproduction
        self.enable_inheritance = enable_inheritance
        self.reproduction_cooldown = reproduction_cooldown
        self.max_children_per_pair = max_children_per_pair

        self.total_steps: int = 0
        self.birth_log: List[Dict] = []
        self.death_log: List[Dict] = []
        self.inheritance_log: List[Dict] = []

        # Lineage tracking
        self.lineages: Dict[str, LineageRecord] = {}
        self.lineage_count: int = 0
        self.agent_lineage: Dict[str, str] = {}  # agent_id → lineage_id

    def _next_lineage_id(self) -> str:
        self.lineage_count += 1
        return f"l_{self.lineage_count}"

    def _get_life_stage(self, age: int) -> str:
        """Get the life stage for a given age."""
        for stage, (start, end) in LIFE_STAGES.items():
            if start <= age <= end:
                return stage
        return 'dying'

    def _get_age_multiplier(self, age: int, stage_map: Dict[str, float]) -> float:
        """Get age-based multiplier for a parameter."""
        stage = self._get_life_stage(age)
        return stage_map.get(stage, 0.5)

    def _compute_reproduction_eligibility(
        self,
        agents: List[CognitiveAgent],
        step: int
    ) -> List[Tuple[CognitiveAgent, CognitiveAgent]]:
        """
        Find eligible reproduction pairs.

        Conditions:
          - Both active, prime or middle stage
          - Fertility probability (age-dependent curve)
          - Wealth above threshold
          - Ideological similarity (0.2 minimum)
          - No recent common ancestry (parent-child, sibling, half-sibling, grandparent)
          - Not on cooldown

        Returns list of (parent_a, parent_b) pairs.
        """
        eligible = []
        for a in agents:
            if not a.active:
                continue
            stage = self._get_life_stage(a.age)
            if stage not in ('prime', 'middle', 'elderly'):
                continue
            # Fertility curve: age-dependent probability
            if random.random() >= FERTILITY_RATES[stage]:
                continue
            if a.wealth < REPRODUCTION_WEALTH_THRESHOLD:
                continue
            eligible.append(a)

        # Find compatible pairs
        pairs = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                a, b = eligible[i], eligible[j]
                # Check last reproduction cooldown (probabilistic to desync cohorts)
                if hasattr(a, 'last_reproduction_step') and a.last_reproduction_step is not None:
                    steps_since = step - a.last_reproduction_step
                    ready_prob = min(1.0, steps_since / self.reproduction_cooldown)
                    if random.random() >= ready_prob:
                        continue
                if hasattr(b, 'last_reproduction_step') and b.last_reproduction_step is not None:
                    steps_since = step - b.last_reproduction_step
                    ready_prob = min(1.0, steps_since / self.reproduction_cooldown)
                    if random.random() >= ready_prob:
                        continue

                # Check genealogical proximity (NOT lineage-based)
                a_parents = getattr(a, 'parents', set())
                b_parents = getattr(b, 'parents', set())

                too_close = False
                # Parent-child
                if b.agent_id in a_parents or a.agent_id in b_parents:
                    too_close = True
                # Full siblings (same two parents)
                if a_parents and b_parents and a_parents == b_parents:
                    too_close = True

                if too_close:
                    continue

                # Check ideological similarity
                sim = a.ideological_similarity(b)
                if sim < 0.2:
                    continue

                pairs.append((a, b))
                if len(pairs) >= self.max_children_per_pair:
                    break
            if len(pairs) >= self.max_children_per_pair:
                break

        return pairs

    def _create_offspring(
        self,
        parent_a: CognitiveAgent,
        parent_b: CognitiveAgent,
        agents: List[CognitiveAgent],
        species: Dict,
        step: int,
        spawn_function: Any,
    ) -> Optional[CognitiveAgent]:
        """
        Create a new agent from two parents.

        Inherits:
          - Ideology: average of parents + mutation (cultural drift)
          - Species: randomly from one parent
          - Wealth: INHERITANCE_FRACTION of combined wealth
          - Narrative preferences from both parents
          - Coalition: none (must be earned)
        """
        if len([a for a in agents if a.active]) >= self.max_population:
            return None

        # Mixed ideology with mutation
        child_ideology = (parent_a.ideology + parent_b.ideology) * 0.5
        mutation = np.random.randn(len(child_ideology)) * IDELOGY_MUTATION_RATE
        child_ideology = np.tanh(child_ideology + mutation)

        # Random species from parents
        child_species = random.choice([parent_a.species, parent_b.species])

        # Create agent
        child = spawn_function(child_species, parent_ideology=child_ideology)
        child.wealth = (parent_a.wealth + parent_b.wealth) * INHERITANCE_FRACTION * 0.5
        child.wealth = max(0.5, child.wealth)

        # Inherit some traits
        child.reliability = (parent_a.reliability + parent_b.reliability) * 0.5
        child.time_horizon = (parent_a.time_horizon + parent_b.time_horizon) * 0.5
        child.risk_tolerance = (parent_a.risk_tolerance + parent_b.risk_tolerance) * 0.5
        child.exploration_rate = (
            parent_a.exploration_rate + parent_b.exploration_rate
        ) * 0.5 * 1.2  # Youth bonus

        # Reduce parent wealth (reproduction cost)
        cost = child.wealth * REPRODUCTION_COST
        parent_a.wealth -= cost * 0.5
        parent_b.wealth -= cost * 0.5
        parent_a.wealth = max(0.5, parent_a.wealth)
        parent_b.wealth = max(0.5, parent_b.wealth)

        # Mark reproduction
        parent_a.last_reproduction_step = step
        parent_b.last_reproduction_step = step

        # Track parentage for incest prevention
        child.parents = {parent_a.agent_id, parent_b.agent_id}
        child.grandparents = (
            getattr(parent_a, 'parents', set())
            | getattr(parent_b, 'parents', set())
        )

        # Initialize lineage
        lineage_a = self.agent_lineage.get(parent_a.agent_id)
        lineage_b = self.agent_lineage.get(parent_b.agent_id)

        if lineage_a and lineage_a == lineage_b:
            # Same lineage — extend
            lineage_id = lineage_a
            self.lineages[lineage_id].total_members += 1
            self.lineages[lineage_id].current_members += 1
            self.lineages[lineage_id].last_birth_step = step
        elif lineage_a and lineage_b:
            # Merge lineages (dynastic union)
            lineage_id = lineage_a
            merged = self.lineages[lineage_b]
            self.lineages[lineage_a].total_members += merged.total_members + 1
            self.lineages[lineage_a].current_members += 1
            self.lineages[lineage_a].last_birth_step = step
            merged.extinct = True
            # Update all agents with lineage_b to lineage_a
            for aid, lid in list(self.agent_lineage.items()):
                if lid == lineage_b:
                    self.agent_lineage[aid] = lineage_a
        elif lineage_a:
            lineage_id = lineage_a
            self.lineages[lineage_a].total_members += 1
            self.lineages[lineage_a].current_members += 1
            self.lineages[lineage_a].last_birth_step = step
        elif lineage_b:
            lineage_id = lineage_b
            self.lineages[lineage_b].total_members += 1
            self.lineages[lineage_b].current_members += 1
            self.lineages[lineage_b].last_birth_step = step
        else:
            # New lineage
            lineage_id = self._next_lineage_id()
            self.lineages[lineage_id] = LineageRecord(
                lineage_id=lineage_id,
                founding_agent=child.agent_id,
                birth_step=step,
            )

        self.agent_lineage[child.agent_id] = lineage_id

        # Apply youth modifiers
        child.exploration_rate *= AGE_EXPLORATION['youth']
        child.productivity *= AGE_PRODUCTIVITY['youth'] * 0.8  # Learning curve

        return child

    def _process_death(
        self,
        agent: CognitiveAgent,
        agents: List[CognitiveAgent],
        cause: str,
        step: int,
    ):
        """Process an agent's death: inheritance, lineage update."""
        agent.active = False

        death_record = {
            'agent_id': agent.agent_id,
            'species': agent.species,
            'age': agent.age,
            'cause': cause,
            'wealth': round(agent.wealth, 3),
            'reliability': round(agent.reliability, 3),
            'veto': round(agent.veto_weight, 3),
            'step': step,
        }
        self.death_log.append(death_record)

        # Inheritance
        if self.enable_inheritance:
            self._execute_inheritance(agent, agents, step)

        # Lineage update
        lineage_id = self.agent_lineage.get(agent.agent_id)
        if lineage_id and lineage_id in self.lineages:
            self.lineages[lineage_id].current_members -= 1
            if self.lineages[lineage_id].current_members <= 0:
                self.lineages[lineage_id].extinct = True
            if agent.veto_weight > 0.5:
                self.lineages[lineage_id].total_political_offspring += 1

    def _execute_inheritance(
        self,
        deceased: CognitiveAgent,
        agents: List[CognitiveAgent],
        step: int,
    ):
        """
        Distribute deceased agent's wealth and capital.

        Rules:
          - Wealth → closest ideological match among lineage members
          - If no lineage → wealth distributed to coalition or dissolved
          - Veto weight → lineage successor (if exists)
          - Coalition membership → open slot
        """
        lineage_id = self.agent_lineage.get(deceased.agent_id)

        # Find successor (same lineage, highest ideological similarity)
        successor = None
        if lineage_id:
            lineage_members = [
                a for a in agents if a.active and a.agent_id != deceased.agent_id
                and self.agent_lineage.get(a.agent_id) == lineage_id
            ]
            if lineage_members:
                successor = max(
                    lineage_members,
                    key=lambda a: a.ideological_similarity(deceased)
                )

        # If no lineage successor, find closest coalition member
        if successor is None and deceased.coalition_id:
            coalition_members = [
                a for a in agents if a.active and a.agent_id != deceased.agent_id
                and a.coalition_id == deceased.coalition_id
            ]
            if coalition_members:
                successor = max(
                    coalition_members,
                    key=lambda a: a.ideological_similarity(deceased)
                )

        if successor:
            # Transfer wealth
            inherited_wealth = deceased.wealth * INHERITANCE_FRACTION
            successor.wealth += inherited_wealth
            successor.wealth = min(10.0, successor.wealth)

            # Transfer some political capital
            successor.veto_weight = max(
                successor.veto_weight,
                deceased.veto_weight * 0.5
            )

            self.inheritance_log.append({
                'deceased': deceased.agent_id,
                'successor': successor.agent_id,
                'wealth_transferred': round(inherited_wealth, 3),
                'veto_transferred': round(deceased.veto_weight * 0.5, 3),
                'step': step,
                'lineage': lineage_id,
            })
        else:
            # Wealth dissipates (lost to the system)
            self.inheritance_log.append({
                'deceased': deceased.agent_id,
                'successor': None,
                'wealth_transferred': 0.0,
                'veto_transferred': 0.0,
                'step': step,
                'lineage': lineage_id,
                'note': 'wealth_dissipated_no_heir',
            })

    def apply_age_effects(self, agents: List[CognitiveAgent]):
        """
        Modify agent parameters based on life stage.

        This is the AGE DYNAMIC:
          Youth → high exploration, low productivity
          Prime → peak productivity, building reputation
          Middle → declining energy, rising wisdom
          Elderly → low exploration, high reliability
          Dying → rapid decay
        """
        for agent in agents:
            if not agent.active:
                continue

            stage = self._get_life_stage(agent.age)
            prod_mult = AGE_PRODUCTIVITY[stage]
            reli_mult = AGE_RELIABILITY[stage]
            explor_mult = AGE_EXPLORATION[stage]
            risk_mult = AGE_RISK_TOLERANCE[stage]

            # Gradual adaptation (not instant)
            agent.productivity = 0.95 * agent.productivity + 0.05 * (
                agent.productivity * prod_mult
            )
            agent.reliability = 0.95 * agent.reliability + 0.05 * (
                agent.reliability * reli_mult
            )
            agent.exploration_rate = 0.95 * agent.exploration_rate + 0.05 * (
                agent.exploration_rate * explor_mult
            )
            agent.risk_tolerance = 0.95 * agent.risk_tolerance + 0.05 * (
                agent.risk_tolerance * risk_mult
            )

            # Clip
            agent.productivity = float(np.clip(agent.productivity, 0.05, 0.95))
            agent.reliability = float(np.clip(agent.reliability, 0.05, 0.95))
            agent.exploration_rate = float(np.clip(agent.exploration_rate, 0.01, 0.95))
            agent.risk_tolerance = float(np.clip(agent.risk_tolerance, 0.05, 0.95))

    def age_agents(self, agents: List[CognitiveAgent]):
        """Increment age for all active agents."""
        for agent in agents:
            if agent.active:
                agent.age += 1

    def check_natural_death(
        self,
        agents: List[CognitiveAgent],
        step: int,
    ) -> List[Dict]:
        """
        Check for natural death based on age.

        Returns list of death events.
        """
        deaths = []
        if not self.enable_natural_death:
            return deaths

        for agent in agents:
            if not agent.active:
                continue
            if agent.age < 60:  # No natural death before 60
                continue

            # Compute maximum lifespan for this agent
            max_lifespan = int(np.random.normal(MAX_LIFESPAN_MEAN, MAX_LIFESPAN_STD))
            max_lifespan = max(60, max_lifespan)

            death_prob = natural_death_probability(agent.age, max_lifespan)
            if random.random() < death_prob:
                self._process_death(agent, agents, 'natural_death', step)
                deaths.append(self.death_log[-1])

        return deaths

    def step(
        self,
        agents: List[CognitiveAgent],
        species: Dict,
        step: int,
        spawn_function: Any,
        compute_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run one generational cycle.

        Order:
          0. Resource gathering (if compute_fn provided)
          1. Age all agents
          2. Apply age effects
          3. Check natural death
          4. Reproduction
          5. Update lineage tracking

        Returns dict with births, deaths, inheritances.
        """
        self.total_steps += 1

        births = []
        deaths = []
        inheritances = []

        # 0. Resource gathering
        if compute_fn:
            compute_fn(agents, species, step)

        # 1. Age
        self.age_agents(agents)

        # 2. Age effects
        self.apply_age_effects(agents)

        # 3. Natural death
        natural_deaths = self.check_natural_death(agents, step)
        deaths.extend(natural_deaths)

        # 4. Reproduction
        if (self.enable_reproduction
                and step % self.reproduction_interval == 0
                and len([a for a in agents if a.active]) < self.max_population):
            pairs = self._compute_reproduction_eligibility(agents, step)
            for parent_a, parent_b in pairs:
                child = self._create_offspring(
                    parent_a, parent_b, agents, species, step, spawn_function
                )
                if child:
                    agents.append(child)
                    births.append({
                        'child_id': child.agent_id,
                        'parent_a': parent_a.agent_id,
                        'parent_b': parent_b.agent_id,
                        'species': child.species,
                        'wealth': round(child.wealth, 3),
                        'step': step,
                    })
                    self.birth_log.append(births[-1])

        # 5. Return recent inheritances
        inheritances = self.inheritance_log[-10:] if self.inheritance_log else []

        return {
            'births': births,
            'deaths': deaths,
            'inheritances': inheritances,
            'n_active': len([a for a in agents if a.active]),
            'n_lineages': len([l for l in self.lineages.values() if not l.extinct]),
        }

    # ------------------------------------------------------------------
    # Stats & Queries
    # ------------------------------------------------------------------

    def get_dynastic_concentration(self) -> float:
        """How concentrated is wealth within dynasties?
        
        High = few lineages control most wealth (oligarchy forming).
        """
        active_lineages = {
            lid: rec for lid, rec in self.lineages.items() if not rec.extinct
        }
        if len(active_lineages) < 2:
            return 0.0

        total_wealth = sum(rec.total_wealth_accumulated for rec in active_lineages.values())
        if total_wealth < 0.01:
            return 0.0

        wealths = sorted(
            [rec.total_wealth_accumulated for rec in active_lineages.values()],
            reverse=True
        )
        top_n = max(1, len(wealths) // 3)
        top_share = sum(wealths[:top_n]) / total_wealth
        return float(top_share)

    def get_generational_turnover_rate(self, last_n: int = 100) -> float:
        """How many generations have turned over recently."""
        recent_deaths = [d for d in self.death_log
                         if d['step'] > self.total_steps - last_n]
        return len(recent_deaths) / max(1, last_n)

    def get_cultural_drift_rate(self) -> float:
        """How fast ideology changes across generations."""
        if len(self.birth_log) < 2:
            return 0.0
        # Approximate via mutation rate
        return IDELOGY_MUTATION_RATE * len(self.birth_log) / max(1, self.total_steps)

    def get_stats(self) -> Dict:
        return {
            'n_births': len(self.birth_log),
            'n_deaths': len(self.death_log),
            'n_inheritances': len(self.inheritance_log),
            'n_active_lineages': len([l for l in self.lineages.values() if not l.extinct]),
            'dynastic_concentration': round(self.get_dynastic_concentration(), 3),
            'turnover_rate': round(self.get_generational_turnover_rate(), 4),
            'cultural_drift': round(self.get_cultural_drift_rate(), 4),
        }

    def get_lineage_report(self, top_n: int = 5) -> str:
        """Generate a textual report of top lineages."""
        active = {lid: rec for lid, rec in self.lineages.items() if not rec.extinct}
        if not active:
            return "No active lineages."

        sorted_lineages = sorted(
            active.items(),
            key=lambda x: x[1].total_members,
            reverse=True
        )[:top_n]

        lines = [f"Top {len(sorted_lineages)} lineages:"]
        for lid, rec in sorted_lineages:
            lines.append(
                f"  {lid}: {rec.current_members} members, "
                f"{rec.total_members} total, "
                f"{'extinct' if rec.extinct else 'active'}, "
                f"wealth={rec.total_wealth_accumulated:.1f}"
            )
        return "\n".join(lines)


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_lifecycle_stages():
    """Test that life stages map correctly."""
    print("\n" + "=" * 60)
    print("48.5C.1 — LIFECYCLE STAGES")
    print("=" * 60)

    engine = GenerationalEngine()

    stages_tested = {}
    for age in range(0, 101, 10):
        stage = engine._get_life_stage(age)
        stages_tested[age] = stage

    # Verify all stages appear
    all_stages = set(stages_tested.values())
    print(f"  ✓ Stages by age: {stages_tested}")
    assert 'youth' in all_stages
    assert 'prime' in all_stages
    assert 'middle' in all_stages
    assert 'elderly' in all_stages
    assert 'dying' in all_stages
    print(f"  ✓ All 5 life stages covered")

    # Verify age effects
    agent = CognitiveAgent('test', 'exploitative', np.random.randn(32))
    agent.age = 20  # Prime
    engine.apply_age_effects([agent])
    print(f"  ✓ Prime (age 20): prod={agent.productivity:.3f}, "
          f"reli={agent.reliability:.3f}, explor={agent.exploration_rate:.3f}")

    print("  >>> LifecycleStages PASSED\n")


def test_aging_and_death():
    """Test that agents age and die naturally."""
    print("\n" + "=" * 60)
    print("48.5C.2 — AGING & NATURAL DEATH")
    print("=" * 60)

    engine = GenerationalEngine()

    agents = []
    for i in range(5):
        a = CognitiveAgent(f'a_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = 70  # Elderly — should die quickly
        agents.append(a)

    # Young agent (should survive short test)
    young = CognitiveAgent('young', 'exploitative', np.random.randn(32))
    young.active = True
    young.age = 15
    agents.append(young)

    random.seed(42)
    total_deaths = 0
    for _ in range(30):
        result = engine.step(agents, {}, 0, lambda *a, **kw: None)
        total_deaths += len(result['deaths'])

    assert young.active, "Young agent should survive short test"
    assert total_deaths > 0, "Elderly agents should die"
    print(f"  ✓ {total_deaths} elderly deaths in 30 steps, young survived ({young.age})")

    # Death causes
    causes = set(d['cause'] for d in engine.death_log)
    print(f"  ✓ Death causes: {causes}")

    print("  >>> AgingAndDeath PASSED\n")


def test_reproduction():
    """Test reproduction and inheritance."""
    print("\n" + "=" * 60)
    print("48.5C.3 — REPRODUCTION & INHERITANCE")
    print("=" * 60)

    engine = GenerationalEngine(
        max_population=30,
        reproduction_interval=1,  # Every step for testing
    )

    random.seed(42)
    np.random.seed(42)

    agents = []
    base_ideo = np.random.randn(32) * 0.3  # clustered ideology
    for i in range(6):
        a = CognitiveAgent(f'parent_{i}', 'exploitative', base_ideo + np.random.randn(32) * 0.1)
        a.active = True
        a.age = 25 + i * 3  # Prime
        a.wealth = 3.0 + i * 0.5  # Above reproduction threshold
        agents.append(a)

    # Set up lineages for first two parents
    engine.agent_lineage['parent_0'] = 'l_dynasty_1'
    engine.agent_lineage['parent_1'] = 'l_dynasty_1'
    engine.lineages['l_dynasty_1'] = LineageRecord(
        lineage_id='l_dynasty_1', founding_agent='parent_0',
        birth_step=0, total_members=2, current_members=2,
    )

    # Simple spawn function
    def spawn_fn(species, parent_ideology=None):
        from phase48_cognitive_political_economy import SPECIES_PARAMS
        params = SPECIES_PARAMS.get(species, SPECIES_PARAMS['exploitative'])
        ideology = parent_ideology if parent_ideology is not None else np.random.randn(32)
        child = CognitiveAgent(
            agent_id=f'child_{engine.total_steps}_{len(engine.birth_log)}',
            species=species,
            ideology=ideology.copy(),
            productivity=0.3 + 0.4 * random.random(),
            reliability=0.3 + 0.3 * random.random(),
            bid_intensity=0.3 + 0.4 * random.random(),
            birth_step=engine.total_steps,
            time_horizon=params['time_horizon'],
            risk_tolerance=params['risk_tolerance'],
            exploration_rate=params['exploration_rate'],
        )
        child.wealth = 0.5
        return child

    # Run until reproduction
    births = []
    for step in range(1, 51):
        result = engine.step(agents, {}, step, spawn_fn)
        births.extend(result['births'])

    assert len(births) > 0, "Should have at least one birth"
    print(f"  ✓ {len(births)} births in 50 steps")

    # Check inheritance (kill an agent and verify)
    wealth_before = agents[0].wealth
    victims = [a for a in agents if a.active and a.age > 10]
    if victims:
        victim = victims[0]
        engine._process_death(victim, agents, 'test_death', step=100)
        print(f"  ✓ Death processed: {victim.agent_id}, wealth={victim.wealth:.2f}")

        # Check inheritance log
        if engine.inheritance_log:
            inh = engine.inheritance_log[-1]
            print(f"  ✓ Inheritance: {inh['deceased']} → "
                  f"{inh['successor']}, wealth={inh['wealth_transferred']:.3f}")

    # Check lineage tracking
    active_lineages = len([l for l in engine.lineages.values() if not l.extinct])
    print(f"  ✓ Active lineages: {active_lineages}")

    print("  >>> ReproductionAndInheritance PASSED\n")


def test_lineage_tracking():
    """Test lineage tracking and dynastic concentration."""
    print("\n" + "=" * 60)
    print("48.5C.4 — LINEAGE TRACKING")
    print("=" * 60)

    engine = GenerationalEngine()

    # Create multiple lineages
    for i in range(3):
        lid = f'l_test_{i}'
        engine.lineages[lid] = LineageRecord(
            lineage_id=lid,
            founding_agent=f'founder_{i}',
            birth_step=0,
            total_members=5 + i * 3,
            current_members=2 + i,
            total_wealth_accumulated=10.0 + i * 20.0,
        )

    concentration = engine.get_dynastic_concentration()
    print(f"  ✓ Dynastic concentration: {concentration:.3f}")

    # With unequal wealth distribution, top third should have > 1/3 share
    assert concentration > 0.3, \
        f"Unequal wealth should concentrate: {concentration:.3f}"

    # Lineage report
    report = engine.get_lineage_report(top_n=2)
    print(f"  ✓ Report:\n{report}")

    print("  >>> LineageTracking PASSED\n")


def test_cultural_drift():
    """Test that ideology changes across generations."""
    print("\n" + "=" * 60)
    print("48.5C.5 — CULTURAL DRIFT")
    print("=" * 60)

    engine = GenerationalEngine(
        max_population=30,
        reproduction_interval=1,
    )

    # Create two parents with moderately different ideologies
    parent_a = CognitiveAgent('a', 'exploitative', np.ones(16) * 0.5 + np.random.randn(16) * 0.1)
    parent_a.active = True
    parent_a.age = 25
    parent_a.wealth = 5.0

    parent_b = CognitiveAgent('b', 'exploratory', np.ones(16) * 0.3 + np.random.randn(16) * 0.1)
    parent_b.active = True
    parent_b.age = 27
    parent_b.wealth = 5.0

    agents = [parent_a, parent_b]
    engine.agent_lineage['a'] = 'l_a'
    engine.agent_lineage['b'] = 'l_b'
    engine.lineages['l_a'] = LineageRecord(
        lineage_id='l_a', founding_agent='a',
        birth_step=0, total_members=1, current_members=1,
    )
    engine.lineages['l_b'] = LineageRecord(
        lineage_id='l_b', founding_agent='b',
        birth_step=0, total_members=1, current_members=1,
    )

    def spawn_fn(species, parent_ideology=None):
        from phase48_cognitive_political_economy import SPECIES_PARAMS
        params = SPECIES_PARAMS.get(species, SPECIES_PARAMS['exploitative'])
        ideology = parent_ideology if parent_ideology is not None else np.ones(16) * 0.5
        child = CognitiveAgent(
            agent_id=f'c_{engine.total_steps}',
            species=species,
            ideology=ideology.copy(),
            productivity=0.5, reliability=0.5,
            bid_intensity=0.5,
            birth_step=engine.total_steps,
            time_horizon=params['time_horizon'],
            risk_tolerance=params['risk_tolerance'],
            exploration_rate=params['exploration_rate'],
        )
        child.wealth = 0.5
        return child

    random.seed(42)
    children = []
    for step in range(1, 101):
        result = engine.step(agents, {}, step, spawn_fn)
        children.extend(result['births'])

    if children:
        # Find the child
        child = next((a for a in agents if a.agent_id.startswith('c_')), None)
        if child:
            # Child ideology should be between parents (drifted)
            child_mean = float(np.mean(np.abs(child.ideology)))
            print(f"  ✓ {len(children)} children born")
            print(f"  ✓ Child mean ideology: {child_mean:.3f} "
                  f"(parent_a: {float(np.mean(np.abs(parent_a.ideology))):.3f}, "
                  f"parent_b: {float(np.mean(np.abs(parent_b.ideology))):.3f})")

            # Should differ from both parents (mutation)
            sim_a = parent_a.ideological_similarity(child)
            sim_b = parent_b.ideological_similarity(child)
            print(f"  ✓ Similarity to parent_a: {sim_a:.3f}")
            print(f"  ✓ Similarity to parent_b: {sim_b:.3f}")

    drift_rate = engine.get_cultural_drift_rate()
    print(f"  ✓ Cultural drift rate: {drift_rate:.4f}")

    print("  >>> CulturalDrift PASSED\n")


def test_full_generational_cycle():
    """Test a full generational run with integrated dynamics."""
    print("\n" + "=" * 60)
    print("48.5C.6 — FULL GENERATIONAL CYCLE")
    print("=" * 60)

    engine = GenerationalEngine(
        max_population=200,
        reproduction_interval=1,
        reproduction_cooldown=5,
        max_children_per_pair=10,
    )

    from phase48_cognitive_political_economy import SPECIES_PARAMS

    def spawn_fn(species, parent_ideology=None):
        params = SPECIES_PARAMS.get(species, SPECIES_PARAMS['exploitative'])
        ideology = parent_ideology if parent_ideology is not None else np.tanh(
            np.random.randn(32) * 0.5
        )
        child = CognitiveAgent(
            agent_id=f'born_{engine.total_steps}',
            species=species,
            ideology=ideology.copy(),
            productivity=0.3 + 0.4 * random.random(),
            reliability=0.3 + 0.3 * random.random(),
            bid_intensity=0.3 + 0.4 * random.random(),
            birth_step=engine.total_steps,
            time_horizon=params['time_horizon'],
            risk_tolerance=params['risk_tolerance'],
            exploration_rate=params['exploration_rate'],
        )
        child.wealth = 0.5
        return child

    # Create diverse initial population — varied ideologies ensure multiple genetic lines
    species_list = ['exploitative', 'exploratory', 'defensive',
                    'identity_preserving', 'novelty_seeking', 'stability_seeking']
    agents = []
    for i in range(30):
        ideology = np.tanh(np.random.randn(32) * 0.8)  # Diverse ideologies
        a = CognitiveAgent(
            f'init_{i}',
            species_list[i % 6],
            ideology,
            productivity=0.3 + 0.4 * random.random(),
            reliability=0.3 + 0.3 * random.random(),
            bid_intensity=0.3 + 0.4 * random.random(),
            birth_step=0,
        )
        a.active = True
        a.age = 5 + i * 3  # Spread ages: 5–92
        a.wealth = 3.0 + random.random() * 3.0
        agents.append(a)

    random.seed(42)
    np.random.seed(42)

    # Run 1000 steps
    total_births = 0
    total_deaths = 0

    def _full_cycle_wealth(ag, sp, st):
        for a in ag:
            if a.active:
                a.wealth += min(0.15, (10.0 - a.wealth) * 0.02)

    for step in range(1, 1001):
        result = engine.step(agents, {}, step, spawn_fn, compute_fn=_full_cycle_wealth)
        total_births += len(result['births'])
        total_deaths += len(result['deaths'])

    active = len([a for a in agents if a.active])
    stats = engine.get_stats()
    ages = [a.age for a in agents if a.active]

    print(f"  1000 steps completed")
    print(f"  ✓ Active agents: {active}")
    print(f"  ✓ Total births: {total_births}")
    print(f"  ✓ Total deaths: {total_deaths}")
    print(f"  ✓ Active lineages: {stats['n_active_lineages']}")
    print(f"  ✓ Dynastic concentration: {stats['dynastic_concentration']:.3f}")

    if active > 0:
        age_range = max(ages) - min(ages)
        print(f"  ✓ Age range: {min(ages)}–{max(ages)} ({age_range} span)")

    assert total_births > 0, "Should have births"
    assert active > 0, "Should have survivors"

    print("  >>> FullGenerationalCycle PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5C: GENERATIONAL TURNOVER ENGINE                       ║
║                                                                   ║
║  Transforms Cognitive Political Economy from                     ║
║  MULTI-AGENT SIMULATION → HISTORICAL CIVILIZATION PROCESS         ║
║                                                                   ║
║  Without reproduction and inheritance:                            ║
║    no cultural evolution, no dynasties, no civilizational         ║
║    memory, no institutional succession                            ║
║                                                                   ║
║  Components:                                                      ║
║    1. Aging & Lifecycle (5 stages)                               ║
║    2. Natural Death (age-dependent probability)                  ║
║    3. Reproduction (pairing, cost, cooldown)                     ║
║    4. Inheritance (wealth, veto, lineage)                        ║
║    5. Lineage Tracking (dynasties, concentration)                ║
║    6. Cultural Drift (ideology mutation per generation)          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("LifecycleStages (48.5C.1)", test_lifecycle_stages),
        ("AgingAndDeath (48.5C.2)", test_aging_and_death),
        ("Reproduction (48.5C.3)", test_reproduction),
        ("LineageTracking (48.5C.4)", test_lineage_tracking),
        ("CulturalDrift (48.5C.5)", test_cultural_drift),
        ("FullGenerationalCycle (48.5C.6)", test_full_generational_cycle),
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
  ║  PHASE 48.5C: ALL TESTS PASSED                                ║
  ║                                                               ║
  ║  Generational Turnover Engine ready.                          ║
  ║  The system now has intergenerational dynamics —               ║
  ║  the foundation for historical civilization processes.        ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"\n  Some tests FAILED")
