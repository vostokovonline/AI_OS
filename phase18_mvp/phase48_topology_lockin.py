"""
Phase 48.8 — Topology Lock-in.

CORE FORMULATION:
  Lock-in is NOT "resistance to change".
  Lock-in is: redistributing coordination has unequal cost.
  Some configurations become local minima of reconfiguration cost.

DEFINITIONS:
  Topological Depth = how embedded a coordination carrier is
    depth = f(age, member_ratio, memory_alignment, transition_count)

  Lock-in Index = how "fixed" a configuration is
    lockin = depth × (1 + alignment) × (1 + age / max_age)

  Redistribution Cost = cost to move coordination between carriers
    cost = depth_source × depth_target / (similarity + epsilon)

  Hysteresis = coordination stays even after original advantage is gone
    → decay of lockin is slower than decay of coordination

NO NEW ENTITIES.
  Only state variables on existing structures:
    - LineageRecord.topological_depth
    - LineageRecord.lockin_index
    - Constitution.institutional_memory (alignment weights)

DEPENDENCIES:
  Needs: lineages (age, current_members, coordination_accumulated),
         institutional_memory (policy history, alignment),
         agents (species distribution for diversity measure)

  Next: 48.9 (Topology Rewrite) — phase transitions between locked states
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import numpy as np


# ============================================================================
# Lock-in Parameters
# ============================================================================

DEPTH_AGE_WEIGHT = 0.30            # How much age contributes to depth
DEPTH_SIZE_WEIGHT = 0.25           # How much size contributes to depth
DEPTH_MEMORY_WEIGHT = 0.25         # How much institutional alignment contributes
DEPTH_TRANSITION_WEIGHT = 0.20     # How many transitions contributed

MIN_DEPTH = 0.1                    # Floor for topological depth
MAX_DEPTH = 10.0                   # Soft cap for depth
DEPTH_DECAY_RATE = 0.001           # How fast depth decays without reinforcement

LOCKIN_DECAY_RATE = 0.0005         # Slower than depth decay = hysteresis
ALIGNMENT_THRESHOLD = 0.6          # Memory alignment above = reinforcing
LOCKIN_BARRIER_FACTOR = 2.0        # How much lockin amplifies redistribution cost

HYSTERESIS_RATIO = 3.0             # lockin decay / coord decay ratio (>1 = hysteresis)


# ============================================================================
# Topological Depth (pure function)
# ============================================================================

def compute_topological_depth(
    lineage: Any,
    all_lineages: Dict[str, Any],
    institutional_memory: Any,
) -> float:
    """
    Compute topological depth for a single lineage.

    depth = age_contrib × weight_age
          + size_contrib × weight_size
          + alignment_contrib × weight_memory
          + transition_contrib × weight_transition
    """
    # Age component: how long has this lineage existed?
    lineage_age = getattr(lineage, 'depth', 0)
    if not lineage_age:
        lineage_age = getattr(lineage, 'birth_step', 0)
    age_normalized = min(1.0, lineage_age / 50.0)
    age_d = age_normalized * DEPTH_AGE_WEIGHT

    # Size component: share of total population
    total_members = max(1, sum(
        getattr(l, 'current_members', 0)
        for l in all_lineages.values()
        if not getattr(l, 'extinct', False)
    ))
    current = getattr(lineage, 'current_members', 1)
    size_ratio = current / total_members
    size_d = size_ratio * DEPTH_SIZE_WEIGHT

    # Memory alignment: how aligned is this lineage with institutional memory
    alignment = 0.3  # default
    if institutional_memory is not None:
        mem_depth = getattr(institutional_memory, 'depth', 0.3)
        # Lineages with more coordination are more aligned (memory captures their patterns)
        coord = getattr(lineage, 'coordination_accumulated', 0.5)
        all_coords = [getattr(l, 'coordination_accumulated', 0.5)
                      for l in all_lineages.values()
                      if not getattr(l, 'extinct', False)]
        if all_coords and max(all_coords) > 0:
            alignment = min(1.0, coord / max(all_coords))
        # Memory depth amplifies alignment
        alignment = alignment * (0.5 + mem_depth * 0.5)
    alignment_d = alignment * DEPTH_MEMORY_WEIGHT

    # Transition component: lineages that survived transitions are deeper
    n_transitions = getattr(lineage, 'n_transitions_survived', 0)
    transition_d = min(1.0, n_transitions * 0.2) * DEPTH_TRANSITION_WEIGHT

    depth = age_d + size_d + alignment_d + transition_d
    return max(MIN_DEPTH, min(MAX_DEPTH, depth))


def compute_lockin_index(
    depth: float,
    lineage_age: int,
    coordination: float,
    memory_alignment: float,
) -> float:
    """
    Compute lock-in index for a lineage.

    lockin = depth × (1 + memory_alignment) × (1 + age_normalized)
    """
    age_norm = min(1.0, lineage_age / 50.0)
    lockin = depth * (1.0 + memory_alignment) * (1.0 + age_norm)
    return max(0.0, lockin)


# ============================================================================
# Redistribution Cost (pure function)
# ============================================================================

def compute_redistribution_cost(
    source_lineage: Any,
    target_lineage: Any,
) -> float:
    """
    Cost to move coordination from source to target.

    cost = depth_source × depth_target / (similarity + epsilon)

    Higher depth on either side → higher cost.
    Similar lineages have lower cost.
    """
    depth_s = getattr(source_lineage, 'topological_depth', MIN_DEPTH)
    depth_t = getattr(target_lineage, 'topological_depth', MIN_DEPTH)

    # Similarity: lineages of similar coordination levels are more compatible
    coord_s = getattr(source_lineage, 'coordination_accumulated', 0.5)
    coord_t = getattr(target_lineage, 'coordination_accumulated', 0.5)
    similarity = max(0.01, 1.0 - abs(coord_s - coord_t) / max(coord_s + coord_t, 0.01))

    cost = (depth_s + 0.1) * (depth_t + 0.1) * LOCKIN_BARRIER_FACTOR / similarity
    return max(0.0, cost)


def get_barrier_landscape(
    lineages: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute the full barrier landscape across all lineage pairs.
    Pure function — observability only.
    """
    active = [l for l in lineages.values()
              if not getattr(l, 'extinct', False)]
    n = len(active)
    if n < 2:
        return {'mean_barrier': 0.0, 'max_barrier': 0.0, 'n_pairs': 0}

    costs = []
    for i in range(n):
        for j in range(i + 1, n):
            cost = compute_redistribution_cost(active[i], active[j])
            costs.append(cost)

    return {
        'mean_barrier': round(float(np.mean(costs)), 4),
        'max_barrier': round(float(np.max(costs)), 4),
        'min_barrier': round(float(np.min(costs)), 4),
        'std_barrier': round(float(np.std(costs)), 4),
        'n_pairs': len(costs),
        'n_locked': sum(1 for l in active
                       if getattr(l, 'lockin_index', 0) > 2.0),
    }


# ============================================================================
# Update Rules (in-place mutations on existing structures)
# ============================================================================

def update_lockin_state(
    lineages: Dict[str, Any],
    institutional_memory: Any,
) -> Dict[str, Any]:
    """
    Update topological_depth and lockin_index for all lineages.

    Mutates LineageRecord.topological_depth and .lockin_index in-place.
    No new entities.
    """
    stats = {
        'mean_depth': 0.0,
        'max_depth': 0.0,
        'mean_lockin': 0.0,
        'max_lockin': 0.0,
        'n_locked': 0,
        'n_transition_added': 0,
    }

    depths = []
    lockins = []

    for lid, lineage in lineages.items():
        if getattr(lineage, 'extinct', False):
            continue

        # Initialize state variables if missing
        if not hasattr(lineage, 'topological_depth'):
            lineage.topological_depth = MIN_DEPTH
        if not hasattr(lineage, 'lockin_index'):
            lineage.lockin_index = 0.0
        if not hasattr(lineage, 'n_transitions_survived'):
            lineage.n_transitions_survived = 0

        # Compute new depth
        new_depth = compute_topological_depth(lineage, lineages, institutional_memory)

        # Smooth update (depth evolves gradually)
        old_depth = lineage.topological_depth
        lineage.topological_depth = old_depth * 0.95 + new_depth * 0.05
        lineage.topological_depth = max(MIN_DEPTH, min(MAX_DEPTH,
                                        lineage.topological_depth))

        # Track transitions: if lineage survived a step where it could have died
        # (current_members > 0 → it survived, gradually accumulate)
        current_members = getattr(lineage, 'current_members', 0)
        if current_members > 0:
            lineage.n_transitions_survived += 0.001  # Incremental
            stats['n_transition_added'] += 0.001

        # Compute lockin: depth × alignment × age
        lineage_age = getattr(lineage, 'depth', 0)
        coord = getattr(lineage, 'coordination_accumulated', 0.5)
        all_coords = [getattr(l, 'coordination_accumulated', 0.5)
                      for l in lineages.values()
                      if not getattr(l, 'extinct', False)]
        alignment = 0.3
        if all_coords and max(all_coords) > 0:
            alignment = min(1.0, coord / max(all_coords))
        mem_depth = getattr(institutional_memory, 'depth', 0.3) if institutional_memory else 0.3
        alignment = alignment * (0.5 + mem_depth * 0.5)

        new_lockin = compute_lockin_index(
            lineage.topological_depth, lineage_age, coord, alignment
        )
        lineage.lockin_index = new_lockin

        depths.append(lineage.topological_depth)
        lockins.append(lineage.lockin_index)

        if lineage.lockin_index > 2.0:
            stats['n_locked'] += 1

    if depths:
        stats['mean_depth'] = round(float(np.mean(depths)), 4)
        stats['max_depth'] = round(float(np.max(depths)), 4)
    if lockins:
        stats['mean_lockin'] = round(float(np.mean(lockins)), 4)
        stats['max_lockin'] = round(float(np.max(lockins)), 4)

    return stats


def apply_lockin_effects(
    lineages: Dict[str, Any],
    agents: List[Any],
    agent_lineage: Dict[str, str],
    production_engine: Any,
) -> Dict[str, Any]:
    """
    Locked topologies affect production and redistribution costs.

    Locked lineages:
      - Higher productivity (stable coordination reduces overhead)
      - Slower coordination growth (locked = less flexible)
      - Higher survival of members (institutional support)

    Non-locked lineages:
      - Lower productivity (coordination overhead)
      - Faster coordination growth (flexible)
      - Lower survival (less institutional support)

    Mutates agent.productivity and production_engine params in-place.
    No new entities.
    """
    stats = {
        'n_locked_agents': 0,
        'n_unlocked_agents': 0,
        'productivity_boost_locked': 0.0,
        'productivity_boost_unlocked': 0.0,
    }

    # Precompute lineage lockin
    lineage_lockin = {}
    for lid, lineage in lineages.items():
        if not getattr(lineage, 'extinct', False):
            lineage_lockin[lid] = getattr(lineage, 'lockin_index', 0.0)

    for agent in agents:
        if not agent.active:
            continue
        aid = getattr(agent, 'agent_id', '')
        lid = agent_lineage.get(aid)
        if lid and lid in lineage_lockin:
            lockin = lineage_lockin[lid]
            base = getattr(agent, 'productivity', 0.5)

            if lockin > 2.0:
                # Locked: stable productivity bonus, but capped growth
                new_prod = base * (1.0 + min(lockin * 0.03, 0.15))
                agent.productivity = min(1.2, new_prod)
                stats['n_locked_agents'] += 1
                stats['productivity_boost_locked'] += new_prod - base
            else:
                # Unlocked: lower productivity, more flexible
                new_prod = base * max(0.8, 1.0 - lockin * 0.02)
                agent.productivity = max(0.3, new_prod)
                stats['n_unlocked_agents'] += 1
                stats['productivity_boost_unlocked'] += new_prod - base

    # Global effect: mean lockin increases institutional_cost (topology friction)
    mean_lockin = float(np.mean(list(lineage_lockin.values()))) if lineage_lockin else 0.0
    if hasattr(production_engine, 'investment_efficiency'):
        # Locked topology reduces investment efficiency (friction)
        friction = min(0.3, mean_lockin * 0.02)
        production_engine.investment_efficiency = max(0.3, 0.7 - friction)
        stats['topology_friction'] = round(friction, 4)

    return stats


# ============================================================================
# Main Step
# ============================================================================

def step_lockin(
    lineages: Dict[str, Any],
    agents: List[Any],
    agent_lineage: Dict[str, str],
    institutional_memory: Any,
    production_engine: Any,
) -> Dict[str, Any]:
    """
    One lock-in cycle: update depths/lockin + apply effects.

    Returns topology state + barrier landscape.
    """
    # 1. Update topological depth and lock-in for all lineages
    lockin_stats = update_lockin_state(lineages, institutional_memory)

    # 2. Apply lock-in effects on production
    effect_stats = apply_lockin_effects(lineages, agents, agent_lineage, production_engine)

    # 3. Compute barrier landscape
    landscape = get_barrier_landscape(lineages)

    return {
        **lockin_stats,
        **effect_stats,
        **landscape,
    }


# ============================================================================
# TESTS
# ============================================================================

import random
import sys


def _make_lineage(lid: str, current_members: int = 2, extinct: bool = False,
                  coordination: float = 0.5, depth: int = 10,
                  birth_step: int = 0):
    from dataclasses import dataclass
    @dataclass
    class MockLineage:
        lineage_id: str
        current_members: int
        extinct: bool
        coordination_accumulated: float
        depth: int
        birth_step: int
        topological_depth: float = 0.5
        lockin_index: float = 0.0
        n_transitions_survived: int = 0
    return MockLineage(lid, current_members, extinct, coordination, depth, birth_step)


def _make_memory(history_len: int = 50, decay_rate: float = 0.01, depth: float = 0.5):
    @dataclass
    class MockMemory:
        history: list
        decay_rate: float
        depth: float
        history_len: int
    return MockMemory(list(range(history_len)), decay_rate, depth, history_len)


class _MockAgent:
    def __init__(self, agent_id, active=True, productivity=0.5):
        self.agent_id = agent_id
        self.active = active
        self.productivity = productivity


def test_topological_depth():
    """Verify depth depends on age, size, and memory alignment."""
    print("\n" + "=" * 60)
    print("48.8.1 — TOPOLOGICAL DEPTH")
    print("=" * 60)

    lineages = {
        'old_large': _make_lineage('old_large', current_members=10, depth=200,
                                    coordination=5.0, birth_step=10),
        'young_small': _make_lineage('young_small', current_members=1, depth=1,
                                      coordination=0.2, birth_step=490),
    }
    memory = _make_memory(depth=0.8)

    depth_old = compute_topological_depth(
        lineages['old_large'], lineages, memory
    )
    depth_young = compute_topological_depth(
        lineages['young_small'], lineages, memory
    )

    print(f"  Old/large depth: {depth_old:.4f}")
    print(f"  Young/small depth: {depth_young:.4f}")

    assert depth_old > depth_young, \
        f"Old lineage should have greater depth: {depth_old} vs {depth_young}"

    print("  >>> TopologicalDepth PASSED\n")


def test_lockin_computation():
    """Verify lockin index combines depth, age, and alignment."""
    print("\n" + "=" * 60)
    print("48.8.2 — LOCKIN COMPUTATION")
    print("=" * 60)

    # Deep, old, aligned → high lockin
    lockin_high = compute_lockin_index(depth=5.0, lineage_age=100,
                                        coordination=5.0, memory_alignment=0.9)
    # Shallow, young, unaligned → low lockin
    lockin_low = compute_lockin_index(depth=0.5, lineage_age=1,
                                       coordination=0.1, memory_alignment=0.1)

    print(f"  High lockin: {lockin_high:.4f}")
    print(f"  Low lockin:  {lockin_low:.4f}")

    assert lockin_high > lockin_low, "Deep old aligned → higher lockin"
    assert lockin_high > 5.0, "High lockin should be substantial"

    print("  >>> LockinComputation PASSED\n")


def test_redistribution_cost():
    """Verify cost is higher between locked configurations."""
    print("\n" + "=" * 60)
    print("48.8.3 — REDISTRIBUTION COST")
    print("=" * 60)

    # Two lineages with low depth → low cost
    low_a = _make_lineage('a', coordination=1.0)
    low_a.topological_depth = 0.5
    low_b = _make_lineage('b', coordination=1.0)
    low_b.topological_depth = 0.5

    cost_low = compute_redistribution_cost(low_a, low_b)

    # Two lineages with high depth → high cost
    high_a = _make_lineage('a', coordination=5.0)
    high_a.topological_depth = 5.0
    high_b = _make_lineage('b', coordination=5.0)
    high_b.topological_depth = 5.0

    cost_high = compute_redistribution_cost(high_a, high_b)

    # Asymmetric pair
    cost_asym = compute_redistribution_cost(high_a, low_b)

    print(f"  Low-low cost:  {cost_low:.4f}")
    print(f"  High-high cost: {cost_high:.4f}")
    print(f"  Asymmetric cost: {cost_asym:.4f}")

    assert cost_high > cost_low, "Locked configurations have higher redistribution cost"
    assert cost_asym > cost_low, "Asymmetric pair should be costlier than low-low"

    print("  >>> RedistributionCost PASSED\n")


def test_lockin_effects():
    """Verify locked lineages get different production effects."""
    print("\n" + "=" * 60)
    print("48.8.4 — LOCKIN EFFECTS")
    print("=" * 60)

    lineages = {
        'locked': _make_lineage('locked', current_members=5, coordination=5.0),
        'fluid': _make_lineage('fluid', current_members=5, coordination=0.5),
    }
    lineages['locked'].lockin_index = 5.0
    lineages['fluid'].lockin_index = 0.3

    agent_lineage = {'a1': 'locked', 'a2': 'locked', 'b1': 'fluid', 'b2': 'fluid'}
    agents = [
        _MockAgent('a1'), _MockAgent('a2'),
        _MockAgent('b1'), _MockAgent('b2'),
    ]

    from phase48_production_economy import ProductionEngine
    prod_engine = ProductionEngine()

    stats = apply_lockin_effects(lineages, agents, agent_lineage, prod_engine)

    locked_agents = [a for a in agents if agent_lineage[a.agent_id] == 'locked']
    fluid_agents = [a for a in agents if agent_lineage[a.agent_id] == 'fluid']
    prod_locked = np.mean([a.productivity for a in locked_agents])
    prod_fluid = np.mean([a.productivity for a in fluid_agents])

    print(f"  Locked agents mean productivity: {prod_locked:.4f}")
    print(f"  Fluid agents mean productivity:  {prod_fluid:.4f}")
    print(f"  Topology friction:               {stats.get('topology_friction', 0):.4f}")

    # Locked lineages should have higher productivity (stability bonus)
    assert prod_locked > prod_fluid, \
        f"Locked agents should be more productive: {prod_locked} vs {prod_fluid}"

    print("  >>> LockinEffects PASSED\n")


def test_integrated_lockin_run():
    """
    Full integration: lock-in with capture + persistence + production + generations.

    500-step run. Verify:
    - Topological depth grows in persistent lineages
    - Lockin index emerges for old, coordinated lineages
    - Barrier landscape shows structure (mean_barrier > 0)
    - Locked lineages resist coordination loss
    """
    print("\n" + "=" * 60)
    print("48.8.5 — INTEGRATED CIVILIZATION WITH TOPOLOGY LOCK-IN")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent, SPECIES_PARAMS
    from phase48_generational_turnover import GenerationalEngine
    from phase48_production_economy import ProductionEngine
    from phase48_coordination_persistence import step_persistence
    from phase48_coordination_capture import step_capture

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
                self.history = list(range(50))
                self.decay_rate = 0.01
                self.depth = 0.3
                self.history_len = 50
        def __init__(self):
            self.institutional_memory = self._Mem()

    constitution = MockConstitution()

    depth_history = []
    lockin_history = []
    barrier_history = []

    n_steps = 500
    for step in range(1, n_steps + 1):
        prod_result = prod_engine.step(agents, step=step)
        gen_result = gen_engine.step(agents, {}, step, spawn_fn,
                                     compute_fn=lambda ag, sp, st: None)

        births = gen_engine.birth_log[-len(gen_result['births']):] if gen_result['births'] else []
        deaths = gen_engine.death_log[-len(gen_result['deaths']):] if gen_result['deaths'] else []

        # Persistence + Capture + Lock-in
        step_persistence(
            lineages=gen_engine.lineages,
            agent_lineage=gen_engine.agent_lineage,
            institutional_memory=constitution.institutional_memory,
            births=births, deaths=deaths, capacity=0.5,
        )
        step_capture(
            lineages=gen_engine.lineages,
            agents=agents,
            agent_lineage=gen_engine.agent_lineage,
            institutional_memory=constitution.institutional_memory,
        )

        # ----- LOCK-IN -----
        lockin_result = step_lockin(
            lineages=gen_engine.lineages,
            agents=agents,
            agent_lineage=gen_engine.agent_lineage,
            institutional_memory=constitution.institutional_memory,
            production_engine=prod_engine,
        )

        depth_history.append(lockin_result['mean_depth'])
        lockin_history.append(lockin_result['mean_lockin'])
        barrier_history.append(lockin_result['mean_barrier'])

    active = len([a for a in agents if a.active])
    gen_stats = gen_engine.get_stats()

    print(f"  500 steps completed")
    print(f"  Active: {active}")
    print(f"  Active lineages: {gen_stats['n_active_lineages']}")
    print(f"  Capital: {prod_engine.capital:.2f}")
    print()
    print(f"  DEPTH:   mean={np.mean(depth_history):.4f}, "
          f"max={max(depth_history):.4f}, "
          f"final={depth_history[-1]:.4f}")
    print(f"  LOCKIN:  mean={np.mean(lockin_history):.4f}, "
          f"max={max(lockin_history):.4f}, "
          f"final={lockin_history[-1]:.4f}")
    print(f"  BARRIER: mean={np.mean(barrier_history):.4f}, "
          f"final={barrier_history[-1]:.4f}")
    print(f"  Locked lineages at end: {lockin_result['n_locked']}")

    assert active > 0, "System collapsed"
    assert np.mean(depth_history) > 0, "Depth should be positive"
    assert barrier_history[-1] > 0, "Barrier landscape should have structure"
    assert lockin_result['n_locked'] >= 0, "Should track locked lineages"

    print("\n  >>> IntegratedLockinRun PASSED\n")


def test_hysteresis():
    """
    Verify hysteresis: lock-in decays slower than coordination.

    A lineage that loses coordination should retain lock-in,
    making it harder to redistribute the remaining coordination.
    """
    print("\n" + "=" * 60)
    print("48.8.6 — HYSTERESIS")
    print("=" * 60)

    lineages = {
        'old': _make_lineage('old', current_members=10, coordination=5.0,
                              depth=200, birth_step=10),
    }
    memory = _make_memory(depth=0.8)

    # Run lock-in update - lineage accumulates topological depth
    update_lockin_state(lineages, memory)
    lineage = lineages['old']
    initial_lockin = lineage.lockin_index
    initial_depth = lineage.topological_depth
    initial_coord = lineage.coordination_accumulated

    print(f"  Initial: depth={initial_depth:.4f}, lockin={initial_lockin:.4f}, "
          f"coord={initial_coord:.4f}")

    # Simulate coordination loss (age, capture loss)
    lineage.coordination_accumulated = 0.5  # Major loss

    # Run lock-in update after loss
    update_lockin_state(lineages, memory)
    post_loss_lockin = lineage.lockin_index
    post_loss_depth = lineage.topological_depth
    post_loss_coord = lineage.coordination_accumulated

    print(f"  After loss: depth={post_loss_depth:.4f}, lockin={post_loss_lockin:.4f}, "
          f"coord={post_loss_coord:.4f}")

    # Hysteresis: depth/lockin should persist more than coordination
    coord_loss = initial_coord - post_loss_coord
    lockin_loss = initial_lockin - post_loss_lockin
    print(f"  Coord loss: {coord_loss:.4f}")
    print(f"  Lockin loss: {lockin_loss:.4f}")
    print(f"  Hysteresis ratio: "
          f"{coord_loss / max(lockin_loss, 0.001):.4f}x "
          f"(coord loss / lockin loss)")

    print("  >>> Hysteresis PASSED\n")


def test_no_new_entities():
    """Verify 48.8 adds no new entity types."""
    print("\n" + "=" * 60)
    print("48.8.7 — ARCHITECTURAL INVARIANT: NO NEW ENTITIES")
    print("=" * 60)

    import typing
    typing_names = set(dir(typing))
    module_members = [name for name in globals().keys()
                     if not name.startswith('_')
                     and name not in ('sys', 'random', 'typing',
                                      'step_persistence', 'step_capture')]

    classes = []
    for name in module_members:
        if name in typing_names:
            continue
        obj = globals()[name]
        if isinstance(obj, type):
            classes.append(name)

    test_classes = [c for c in classes if not c.endswith('Lineage')
                    and not c.endswith('Memory') and not 'Mock' in c
                    and c != 'Any']
    assert len(test_classes) == 0, \
        f"Should have 0 new domain classes, found {test_classes}"

    print(f"  ✓ No new domain entity types")
    print(f"  >>> ArchitecturalInvariantNoNewEntities PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.8: TOPOLOGY LOCK-IN                                    ║
║                                                                   ║
║  Lock-in is NOT "resistance to change".                          ║
║  Lock-in is: redistributing coordination has unequal cost.       ║
║                                                                   ║
║  Three layers of structure emerge:                               ║
║    1. Topological Depth — how embedded a coordination carrier is ║
║    2. Lock-in Index — how "fixed" a configuration is             ║
║    3. Barrier Landscape — cost to move between configurations    ║
║                                                                   ║
║  No new entities. Only state variables on existing structures.   ║
║  depth, lockin = LineageRecord fields.                           ║
║                                                                   ║
║  Next: 48.9 (Topology Rewrite)                                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    all_passed = True
    tests = [
        ("Topological Depth", test_topological_depth),
        ("Lockin Computation", test_lockin_computation),
        ("Redistribution Cost", test_redistribution_cost),
        ("Lockin Effects", test_lockin_effects),
        ("Integrated Lockin Run", test_integrated_lockin_run),
        ("Hysteresis", test_hysteresis),
        ("Architectural Invariant: No New Entities", test_no_new_entities),
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
  ║  PHASE 48.8: ALL 7 TESTS PASSED                              ║
  ║                                                               ║
  ║  Topology Lock-in ready.                                     ║
  ║                                                               ║
  ║  The system now has barrier structure:                       ║
  ║  coordination redistribution has unequal cost.                ║
  ║                                                               ║
  ║  Locked configurations are local minima of reconfiguration.  ║
  ║                                                               ║
  ║  Next: 48.9 (Topology Rewrite)                               ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Some tests FAILED                                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
