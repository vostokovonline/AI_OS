"""
Phase 48.5E — Coordination Capacity Engine.

ARCHITECTURAL SHIFT:
  Before: scarcity = multi-dimensional pressure field over resources
  After:  coordination capacity = conserved quantity of civilization.
          Production, capital, demography, institutions are PROJECTIONS
          of coordination dynamics, not fundamental variables.

CORE EQUATION:

  CoordinationCost     = f(trust, fragmentation, memory_decay, demographic_load)
  CoordinationEfficiency = f(specialization, trust_density, memory_transfer_speed)
  CoordinationPersistence = f(institutional_depth, lineage_continuity, narrative_coherence)

  CoordinationCapacity = Efficiency / Cost * Persistence

  Scarcity = decline in sustainable CoordinationCapacity.

DEFINITIONS:
  Wealth = stored coordination potential (infrastructure, education, trust)
  Institution = coordination compression structure
  State = low-friction coordination topology
  Collapse = irreversible coordination capacity failure

DEPENDENCIES:
  Needs: agents (trust, age, species), coalitions (fragmentation, cohesion),
         production_engine (capital, institutional_mult), generational_engine
         (lineages, births/deaths), constitution (institutional_memory)

  Next: 48.6 (Coordination Persistence) — structures that survive generations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import math


# ============================================================================
# Coordination Parameters
# ============================================================================

# Cost factors
TRUST_WEIGHT = 0.35            # Low trust drives cost up
FRAGMENTATION_WEIGHT = 0.25    # Coalition fragmentation drives cost up
MEMORY_DECAY_WEIGHT = 0.20     # Memory loss drives cost up
DEMOGRAPHIC_WEIGHT = 0.20      # Dependency ratio drives cost up

# Efficiency factors
SPECIALIZATION_WEIGHT = 0.30   # Species diversity enables specialization
TRUST_DENSITY_WEIGHT = 0.35    # High trust enables efficient coordination
MEMORY_TRANSFER_WEIGHT = 0.20  # Memory continuity enables coordination
LINEAGE_DEPTH_WEIGHT = 0.15    # Deep lineages improve efficiency

# Persistence factors
INSTITUTIONAL_DEPTH_WEIGHT = 0.40  # Policy history matters
LINEAGE_CONTINUITY_WEIGHT = 0.35   # Family lines persist
NARRATIVE_COHERENCE_WEIGHT = 0.25  # Shared stories stabilize

# Capacity thresholds
COLLAPSE_THRESHOLD = 0.05      # Below this = civilizational collapse risk
CRITICAL_THRESHOLD = 0.20      # Below this = crisis mode
STABLE_THRESHOLD = 0.60        # Above this = growth mode


# ============================================================================
# Coordination Field
# ============================================================================

@dataclass
class CoordinationField:
    """
    The state of coordination capacity at a given step.

    NOT a scarcity metric.
    NOT an FSM phase.
    The primary conserved variable of the civilization substrate.

    cost ∈ [0, inf)  — higher = worse
    efficiency ∈ [0, 1] — higher = better
    persistence ∈ [0, 1] — higher = better
    capacity ∈ [0, inf) — efficiency/cost * persistence

    regime_label is POST-HOC — never read by system logic.
    """
    cost: float = 1.0          # Energy to maintain coherence
    efficiency: float = 1.0    # How effectively agents coordinate
    persistence: float = 1.0   # How long structures survive
    capacity: float = 1.0      # Conserved variable

    # Breakdown (for observability, not computation)
    trust: float = 0.5
    fragmentation: float = 0.0
    memory_decay: float = 0.0
    demographic_load: float = 0.0
    specialization: float = 0.3
    trust_density: float = 0.5
    memory_transfer_speed: float = 0.3
    lineage_depth: float = 0.3
    institutional_depth: float = 0.3
    lineage_continuity: float = 0.5

    @property
    def regime_label(self) -> str:
        """POST-HOC label. Never used in system logic."""
        if self.capacity < COLLAPSE_THRESHOLD:
            return 'collapse'
        if self.capacity < CRITICAL_THRESHOLD:
            return 'crisis'
        if self.capacity < STABLE_THRESHOLD:
            return 'stagnation'
        if self.capacity > 1.5:
            return 'growth'
        return 'stability'

    def to_dict(self) -> Dict[str, float]:
        return {
            'cost': round(self.cost, 4),
            'efficiency': round(self.efficiency, 4),
            'persistence': round(self.persistence, 4),
            'capacity': round(self.capacity, 4),
            'trust': round(self.trust, 4),
            'fragmentation': round(self.fragmentation, 4),
            'demographic_load': round(self.demographic_load, 4),
            'specialization': round(self.specialization, 4),
            'trust_density': round(self.trust_density, 4),
            'lineage_continuity': round(self.lineage_continuity, 4),
            'institutional_depth': round(self.institutional_depth, 4),
            'regime_label': self.regime_label,
        }

    def __repr__(self) -> str:
        return (f"CoordField(cap={self.capacity:.3f}, "
                f"eff={self.efficiency:.3f}, cost={self.cost:.3f}, "
                f"per={self.persistence:.3f} | {self.regime_label})")


# ============================================================================
# Coordination Capacity Engine
# ============================================================================

class CoordinationCapacityEngine:
    """
    Computes coordination capacity each step from system state.

    ARCHITECTURAL CONSTRAINT:
      No FSM, no phase memory, no regime state.
      Every step: compute_field() + apply_effects().
      Regime labels are post-hoc — never stored as state, never read by logic.
    """

    def __init__(self):
        self.field_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Field Computation (pure function of system state)
    # ------------------------------------------------------------------

    def compute_field(
        self,
        agents: List[Any],
        coalitions: Optional[Dict[str, Any]] = None,
        constitution: Any = None,
        production_engine: Any = None,
        generational_engine: Any = None,
    ) -> CoordinationField:
        """
        Pure function: compute coordination field from current system state.
        No side effects, no state reads.
        """
        active = [a for a in agents if a.active]
        n = len(active)
        if n == 0:
            return CoordinationField(cost=100.0, efficiency=0.0,
                                     persistence=0.0, capacity=0.0)

        # --- COORDINATION COST ---

        # Trust (inverted: low trust = high cost)
        if active:
            mean_trust = float(np.mean([a.reliability for a in active]))
        else:
            mean_trust = 0.5
        trust_cost = (1.0 - mean_trust) * TRUST_WEIGHT

        # Coalition fragmentation
        frag_cost = 0.0
        if coalitions and len(coalitions) > 0:
            members_per_coal = [len(c.members) for c in coalitions.values()
                                if hasattr(c, 'members')]
            if len(members_per_coal) > 1 and sum(members_per_coal) > 0:
                total_m = sum(members_per_coal)
                hhi = sum((m / total_m) ** 2 for m in members_per_coal)
                n_coal = len(members_per_coal)
                min_hhi = 1.0 / n_coal
                fragmentation = max(0.0, 1.0 - (hhi - min_hhi) / (1.0 - min_hhi))
                frag_cost = fragmentation * FRAGMENTATION_WEIGHT

        # Memory decay (inverted institutional memory depth)
        mem_cost = 0.0
        if constitution and hasattr(constitution, 'institutional_memory'):
            mem = constitution.institutional_memory
            if hasattr(mem, 'history'):
                mem_depth = min(1.0, len(mem.history) / 200.0)
            else:
                mem_depth = 0.3
        else:
            mem_depth = 0.3
        mem_cost = (1.0 - mem_depth) * MEMORY_DECAY_WEIGHT

        # Demographic load
        if n > 0:
            pre = sum(1 for a in active if a.age < 16)
            prime = sum(1 for a in active if 16 <= a.age <= 65)
            post = sum(1 for a in active if a.age > 65)
            dep_ratio = (pre + post) / max(prime, 1)
        else:
            dep_ratio = 0.0
        demo_cost = min(1.0, dep_ratio / 5.0) * DEMOGRAPHIC_WEIGHT

        total_cost = 0.1 + trust_cost + frag_cost + mem_cost + demo_cost

        # --- COORDINATION EFFICIENCY ---

        # Species diversity as proxy for specialization
        species_counts = {}
        for a in active:
            s = getattr(a, 'species', 'unknown')
            species_counts[s] = species_counts.get(s, 0) + 1
        n_species = len(species_counts)
        # More species = more specialization potential
        specialization = min(1.0, n_species / 6.0)
        spec_eff = specialization * SPECIALIZATION_WEIGHT

        # Trust density: how much trust exists in network
        trust_density = mean_trust
        trust_eff = trust_density * TRUST_DENSITY_WEIGHT

        # Memory transfer speed: lineages carry memory
        mem_transfer = 0.0
        if generational_engine:
            stats = generational_engine.get_stats()
            n_lineages = stats.get('n_active_lineages', 0)
            if n > 0:
                mem_transfer = min(1.0, n_lineages / n * 2.0)
        mem_transfer_eff = mem_transfer * MEMORY_TRANSFER_WEIGHT

        # Lineage depth
        lineage_depth = 0.0
        if generational_engine and hasattr(generational_engine, 'lineages'):
            depths = []
            for lin in generational_engine.lineages.values():
                if hasattr(lin, 'depth'):
                    depths.append(lin.depth)
            if depths:
                lineage_depth = min(1.0, np.mean(depths) / 10.0)
        lineage_eff = lineage_depth * LINEAGE_DEPTH_WEIGHT

        total_efficiency = spec_eff + trust_eff + mem_transfer_eff + lineage_eff

        # --- COORDINATION PERSISTENCE ---

        # Institutional memory depth
        inst_depth = min(1.0, mem_depth * 2.0)
        inst_pers = inst_depth * INSTITUTIONAL_DEPTH_WEIGHT

        # Lineage continuity: how many generations deep
        cont_pers = lineage_depth * LINEAGE_CONTINUITY_WEIGHT

        # Narrative coherence (proxy: species diversity + trust)
        narrative_coherence = min(1.0, (specialization + trust_density) / 2.0)
        narr_pers = narrative_coherence * NARRATIVE_COHERENCE_WEIGHT

        total_persistence = inst_pers + cont_pers + narr_pers

        # --- COORDINATION CAPACITY (conserved variable) ---
        if total_cost > 0:
            capacity = (total_efficiency / total_cost) * max(0.01, total_persistence)
        else:
            capacity = 10.0  # Zero cost = infinite capacity (hypothetical)

        return CoordinationField(
            cost=round(total_cost, 4),
            efficiency=round(total_efficiency, 4),
            persistence=round(total_persistence, 4),
            capacity=round(capacity, 4),
            trust=round(mean_trust, 4),
            fragmentation=round(frag_cost / max(FRAGMENTATION_WEIGHT, 0.001), 4),
            memory_decay=round(mem_cost / max(MEMORY_DECAY_WEIGHT, 0.001), 4),
            demographic_load=round(demo_cost / max(DEMOGRAPHIC_WEIGHT, 0.001), 4),
            specialization=round(specialization, 4),
            trust_density=round(trust_density, 4),
            memory_transfer_speed=round(mem_transfer, 4),
            lineage_depth=round(lineage_depth, 4),
            institutional_depth=round(inst_depth, 4),
            lineage_continuity=round(cont_pers / max(LINEAGE_CONTINUITY_WEIGHT, 0.001), 4),
        )

    # ------------------------------------------------------------------
    # Effect Application (state modification via coordination physics)
    # ------------------------------------------------------------------

    def apply_effects(
        self,
        field: CoordinationField,
        agents: List[Any],
        production_engine: Any,
        constitution: Any = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Apply coordination capacity as system effects.

        Low capacity → survival cost +, productivity -, trust decay +.
        High capacity → all the opposite.

        Effects are continuous functions of field.capacity, not thresholds.
        """
        effects = {}
        cap = max(0.0, min(1.0, field.capacity / 2.0))  # Normalize to [0,1] for scaling

        # Survival cost: inverse of capacity
        survival_mult = 1.0 + (1.0 - cap) * 2.0
        if hasattr(production_engine, 'survival_cost'):
            production_engine.survival_cost = 0.15 * survival_mult
        effects['survival_mult'] = round(survival_mult, 3)

        # Productivity: proportional to capacity
        productivity_mult = 0.5 + cap * 0.5
        effects['productivity_mult'] = round(productivity_mult, 3)

        # Trust decay: inverse of capacity
        trust_decay = (1.0 - cap) * 0.005
        for a in agents:
            if a.active:
                a.reliability = max(0.05, a.reliability - trust_decay)
        effects['trust_decay'] = round(trust_decay, 4)

        # Capital depreciation: inverse of capacity
        dep_base = 0.008
        if hasattr(production_engine, 'depreciation_rate'):
            production_engine.depreciation_rate = dep_base * (1.0 + (1.0 - cap) * 2.0)
        effects['depreciation_rate'] = round(
            getattr(production_engine, 'depreciation_rate', dep_base), 4
        )

        # Investment efficiency: proportional to capacity
        inv_base = 0.7
        if hasattr(production_engine, 'investment_efficiency'):
            production_engine.investment_efficiency = inv_base * (0.5 + cap * 0.5)
        effects['investment_efficiency'] = round(
            getattr(production_engine, 'investment_efficiency', inv_base), 4
        )

        # Institutional memory decay: inverse of persistence
        if constitution and hasattr(constitution, 'institutional_memory'):
            mem = constitution.institutional_memory
            if not hasattr(mem, 'decay_rate'):
                mem.decay_rate = 0.001
            decay = (1.0 - field.persistence) * 0.01
            mem.decay_rate = max(0.001, decay)
            effects['memory_decay_rate'] = round(mem.decay_rate, 4)

        return effects

    # ------------------------------------------------------------------
    # Main Step
    # ------------------------------------------------------------------

    def step(
        self,
        agents: List[Any],
        coalitions: Optional[Dict[str, Any]] = None,
        constitution: Any = None,
        production_engine: Any = None,
        generational_engine: Any = None,
        step: int = 0,
    ) -> Tuple[CoordinationField, Dict[str, Any]]:
        """
        One complete coordination cycle:
          1. Compute field from current state
          2. Apply effects
          3. Log

        Returns (field, effects_dict).
        """
        field = self.compute_field(agents, coalitions, constitution,
                                    production_engine, generational_engine)
        effects = self.apply_effects(field, agents, production_engine, constitution, step)

        self.field_log.append({
            'step': step,
            **field.to_dict(),
            **effects,
        })

        return field, effects

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        if not self.field_log:
            return {}
        recent = self.field_log[-50:]
        return {
            'mean_capacity': round(
                float(np.mean([e['capacity'] for e in recent])), 4
            ),
            'max_capacity': round(
                float(np.max([e['capacity'] for e in recent])), 4
            ),
            'min_capacity': round(
                float(np.min([e['capacity'] for e in recent])), 4
            ),
            'mean_cost': round(
                float(np.mean([e['cost'] for e in recent])), 4
            ),
            'mean_efficiency': round(
                float(np.mean([e['efficiency'] for e in recent])), 4
            ),
            'mean_persistence': round(
                float(np.mean([e['persistence'] for e in recent])), 4
            ),
            'n_steps': len(self.field_log),
        }


# ============================================================================
# TESTS
# ============================================================================

import random
import sys


def _make_test_agent(agent_id: str, age: int, wealth: float = 3.0,
                     reliability: float = 0.5, productivity: float = 0.5,
                     species: str = 'exploitative') -> Any:
    from phase48_cognitive_political_economy import CognitiveAgent
    a = CognitiveAgent(agent_id, species, np.tanh(np.random.randn(32) * 0.5))
    a.active = True
    a.age = age
    a.wealth = wealth
    a.reliability = reliability
    a.productivity = productivity
    return a


def _make_minimal_prod_engine():
    from phase48_production_economy import ProductionEngine
    return ProductionEngine()


class _MockConstitution:
    class _Memory:
        def __init__(self):
            self.history = list(range(50))  # 50 steps of history
    def __init__(self):
        self.institutional_memory = self._Memory()


def test_field_basic_computation():
    """Verify coordination field computes from system state."""
    print("\n" + "=" * 60)
    print("48.5E.1 — FIELD BASIC COMPUTATION")
    print("=" * 60)

    engine = CoordinationCapacityEngine()
    prod_engine = _make_minimal_prod_engine()
    agents = [_make_test_agent(f'a_{i}', age=25 + i * 3, wealth=3.0 + i * 0.5,
                               reliability=0.5 + i * 0.02)
              for i in range(15)]

    for step in range(1, 11):
        prod_engine.step(agents, step=step)

    field = engine.compute_field(agents, production_engine=prod_engine)

    print(f"  Cost:    {field.cost:.4f}")
    print(f"  Eff:     {field.efficiency:.4f}")
    print(f"  Persist: {field.persistence:.4f}")
    print(f"  Cap:     {field.capacity:.4f}")
    print(f"  Regime:  {field.regime_label}")

    assert field.cost > 0, "Coordination cost must be positive"
    assert field.efficiency >= 0, "Efficiency must be >= 0"
    assert field.persistence >= 0, "Persistence must be >= 0"
    assert field.capacity > 0, "Capacity must be positive"

    print("  >>> FieldBasicComputation PASSED\n")


def test_field_degradation():
    """Verify field responds to system degradation."""
    print("\n" + "=" * 60)
    print("48.5E.2 — FIELD DEGRADATION RESPONSE")
    print("=" * 60)

    engine = CoordinationCapacityEngine()

    # Healthy system
    prod_healthy = _make_minimal_prod_engine()
    agents_healthy = [_make_test_agent(f'h_{i}', age=30, wealth=5.0,
                                        reliability=0.8, productivity=0.7)
                      for i in range(20)]

    for step in range(1, 11):
        prod_healthy.step(agents_healthy, step=step)

    field_healthy = engine.compute_field(agents_healthy, production_engine=prod_healthy)
    print(f"  Healthy:  cap={field_healthy.capacity:.4f}, cost={field_healthy.cost:.4f}")

    # Degraded system: low trust, high age, low wealth
    prod_degraded = _make_minimal_prod_engine()
    agents_degraded = [_make_test_agent(f'd_{i}', age=75, wealth=0.3,
                                         reliability=0.1, productivity=0.2)
                       for i in range(20)]

    for step in range(1, 11):
        prod_degraded.step(agents_degraded, step=step)

    field_degraded = engine.compute_field(agents_degraded, production_engine=prod_degraded)
    print(f"  Degraded: cap={field_degraded.capacity:.4f}, cost={field_degraded.cost:.4f}")

    assert field_degraded.capacity < field_healthy.capacity, \
        "Degraded system should have lower capacity"
    assert field_degraded.cost > field_healthy.cost, \
        "Degraded system should have higher cost"

    print("  >>> FieldDegradation PASSED\n")


def test_effect_application():
    """Verify effects modify system state proportional to capacity."""
    print("\n" + "=" * 60)
    print("48.5E.3 — EFFECT APPLICATION")
    print("=" * 60)

    engine = CoordinationCapacityEngine()
    prod_engine = _make_minimal_prod_engine()
    constitution = _MockConstitution()

    agents = [_make_test_agent(f'e_{i}', age=30, wealth=3.0, reliability=0.6)
              for i in range(10)]

    for step in range(1, 5):
        prod_engine.step(agents, step=step)

    initial_survival = prod_engine.survival_cost
    initial_depreciation = prod_engine.depreciation_rate

    # Compute a low-capacity field explicitly
    low_field = CoordinationField(
        cost=5.0, efficiency=0.2, persistence=0.1, capacity=0.04,
        trust=0.1, fragmentation=0.8, demographic_load=0.7,
        specialization=0.1, trust_density=0.1,
        lineage_continuity=0.1, institutional_depth=0.1,
    )

    effects = engine.apply_effects(low_field, agents, prod_engine, constitution)

    print(f"  Survival cost: {initial_survival:.3f} -> {prod_engine.survival_cost:.3f}")
    print(f"  Depreciation:  {initial_depreciation:.4f} -> {prod_engine.depreciation_rate:.4f}")
    print(f"  Trust decay:   {effects['trust_decay']:.4f}")

    assert prod_engine.survival_cost > initial_survival, \
        "Low capacity should increase survival cost"
    assert prod_engine.depreciation_rate > initial_depreciation, \
        "Low capacity should increase depreciation"
    assert effects['survival_mult'] > 1.0, "Survival mult should be > 1.0 for low cap"

    print("  >>> EffectApplication PASSED\n")


def test_integrated_with_production_and_generations():
    """
    Full integration: coordination capacity + production + generations.
    500-step civilization run. Verify:
    - System survives
    - Capacity varies (not flat)
    - Multiple regimes observed
    """
    print("\n" + "=" * 60)
    print("48.5E.4 — INTEGRATED CIVILIZATION WITH COORDINATION CAPACITY")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent, SPECIES_PARAMS
    from phase48_generational_turnover import GenerationalEngine
    from phase48_production_economy import ProductionEngine

    random.seed(42)
    np.random.seed(42)

    gen_engine = GenerationalEngine(
        max_population=80,
        reproduction_interval=1,
        reproduction_cooldown=5,
        max_children_per_pair=6,
    )

    prod_engine = ProductionEngine(
        survival_cost=0.12,
        depreciation_rate=0.008,
        investment_efficiency=0.4,
        investment_threshold=1.5,
        investment_fraction=0.15,
    )

    coord_engine = CoordinationCapacityEngine()

    species_list = ['exploitative', 'exploratory', 'defensive',
                    'identity_preserving', 'novelty_seeking', 'stability_seeking']

    agents = []
    for i in range(25):
        ideology = np.tanh(np.random.randn(32) * 0.8)
        a = CognitiveAgent(f'init_{i}', species_list[i % 6], ideology,
                           productivity=0.3 + 0.4 * random.random(),
                           reliability=0.3 + 0.3 * random.random(),
                           bid_intensity=0.3 + 0.4 * random.random(),
                           birth_step=0)
        a.active = True
        a.age = 5 + i * 3
        a.wealth = 3.0 + random.random() * 3.0
        agents.append(a)

    def spawn_fn(species, parent_ideology=None):
        params = SPECIES_PARAMS.get(species, SPECIES_PARAMS['exploitative'])
        ideology = parent_ideology if parent_ideology is not None else np.tanh(np.random.randn(32) * 0.5)
        child = CognitiveAgent(
            agent_id=f'born_{gen_engine.total_steps}',
            species=species,
            ideology=ideology.copy(),
            productivity=0.3 + 0.4 * random.random(),
            reliability=0.3 + 0.3 * random.random(),
            bid_intensity=0.3 + 0.4 * random.random(),
            birth_step=gen_engine.total_steps,
            time_horizon=params['time_horizon'],
            risk_tolerance=params['risk_tolerance'],
            exploration_rate=params['exploration_rate'],
        )
        child.wealth = 0.5
        return child

    class MockConstitution:
        class _Mem:
            def __init__(self):
                self.history = list(range(100))
                self.decay_rate = 0.001
        def __init__(self):
            self.institutional_memory = self._Mem()

    constitution = MockConstitution()

    capacity_history = []
    regime_counts = {}
    gdp_history = []

    n_steps = 500
    for step in range(1, n_steps + 1):
        prod_result = prod_engine.step(agents, step=step)

        field, effects = coord_engine.step(
            agents, production_engine=prod_engine,
            generational_engine=gen_engine,
            constitution=constitution,
            step=step,
        )

        gen_result = gen_engine.step(agents, {}, step, spawn_fn,
                                     compute_fn=lambda ag, sp, st: None)

        capacity_history.append(field.capacity)
        regime_counts[field.regime_label] = regime_counts.get(field.regime_label, 0) + 1

        if step % 100 == 0:
            gdp_history.append(prod_result['total_output'])

    active = len([a for a in agents if a.active])
    gen_stats = gen_engine.get_stats()
    total_births = gen_stats['n_births']
    total_deaths = gen_stats['n_deaths']

    print(f"  500 steps completed")
    print(f"  Active: {active}, births: {total_births}, deaths: {total_deaths}")
    print(f"  Capital: {prod_engine.capital:.2f}")
    print()
    print(f"  MEAN CAPACITY: {np.mean(capacity_history):.4f}")
    print(f"  MAX CAPACITY:  {np.max(capacity_history):.4f}")
    print(f"  MIN CAPACITY:  {np.min(capacity_history):.4f}")
    print(f"  STD CAPACITY:  {np.std(capacity_history):.4f}")
    print()
    print(f"  REGIMES:")
    for regime, count in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"    {regime:14s}: {count:4d} steps ({100*count/n_steps:.1f}%)")

    assert active > 0, "System collapsed"
    assert np.std(capacity_history) > 0.01, \
        f"Capacity should vary (std={np.std(capacity_history):.4f})"
    assert len(regime_counts) >= 2, \
        f"Should see multiple regimes: {regime_counts}"

    print(f"\n  >>> IntegratedCivilization PASSED\n")


def test_no_fsm_architecture():
    """Verify engine has NO state machine behavior."""
    print("\n" + "=" * 60)
    print("48.5E.5 — ARCHITECTURAL INVARIANT: NO FSM")
    print("=" * 60)

    engine = CoordinationCapacityEngine()

    # No phase/regime/state/mode attributes (skip dunder methods)
    forbidden = ['phase', 'regime', 'state', 'mode']
    for attr in dir(engine):
        if attr.startswith('__') or attr.startswith('_'):
            continue  # Skip dunder/private
        for f in forbidden:
            if f in attr.lower():
                assert False, f"Found forbidden attribute: {attr}"

    # compute_field is pure function (deterministic on same inputs)
    prod_engine = _make_minimal_prod_engine()
    agents = [_make_test_agent(f'fsm_{i}', age=30, wealth=3.0) for i in range(5)]
    for step in range(1, 5):
        prod_engine.step(agents, step=step)

    f1 = engine.compute_field(agents, production_engine=prod_engine)
    f2 = engine.compute_field(agents, production_engine=prod_engine)
    assert f1.capacity == f2.capacity, "Pure function should be deterministic"

    print("  ✓ No phase/regime/state/mode attributes")
    print("  ✓ compute_field is pure function")
    print("  ✓ Regime labels are post-hoc, never stored as state")
    print("  >>> ArchitecturalInvariantNoFSM PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5E: COORDINATION CAPACITY ENGINE                       ║
║                                                                   ║
║  ARCHITECTURAL SHIFT:                                             ║
║    Before: scarcity = multi-dimensional pressure field             ║
║    After:  coordination = conserved quantity of civilization       ║
║                                                                   ║
║    Economy, demography, institutions = projections of             ║
║    coordination dynamics, not fundamental variables.              ║
║                                                                   ║
║    Capacity = Efficiency / Cost * Persistence                     ║
║                                                                   ║
║  Next: 48.6 (Coordination Persistence)                           ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    all_passed = True
    tests = [
        ("Field Basic Computation", test_field_basic_computation),
        ("Field Degradation", test_field_degradation),
        ("Effect Application", test_effect_application),
        ("Integrated Civilization", test_integrated_with_production_and_generations),
        ("Architectural Invariant: No FSM", test_no_fsm_architecture),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"  >>> {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    if all_passed:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  PHASE 48.5E: ALL 5 TESTS PASSED                              ║
  ║                                                               ║
  ║  Coordination Capacity Engine ready.                         ║
  ║                                                               ║
  ║  The conserved quantity of the civilization substrate is     ║
  ║  now coordination capacity, not wealth or resources.         ║
  ║                                                               ║
  ║  Next: 48.6 (Coordination Persistence) —                     ║
  ║  structures that survive generations.                        ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Some tests FAILED                                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
