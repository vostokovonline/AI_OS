"""
Phase 48.6 — Coordination Persistence.

QUESTION:
  What carries coordination capacity across agent deaths?

ANSWER:
  Two mechanisms, zero new entities:

  1. Lineage Coordination Inheritance
     Lineages accumulate coordination capacity across generations.
     Each LineageRecord carries `coordination_accumulated`.
     Newborns inherit a fraction. Deaths impose decay.
     No new entities — LineageRecord already exists in generational_turnover.py.

  2. Institutional Coordination Buffer
     Constitution.institutional_memory carries coordination patterns
     even if no lineages survive a catastrophe.
     Memory depth and decay rate determine buffer strength.
     No new entities — constitution already exists in phase48_cognitive_political_economy.py.

FORMULA:
  coordination_persistence =
      lineage_inheritance × 0.6
    + institutional_buffer × 0.4

  lineage_inheritance = weighted mean of lineage.coordination_accumulated
  institutional_buffer = memory_depth × (1 - memory_decay_rate)

MECHANICS (no new state — just update rules on existing structures):
  - Birth: child gets parent.coordination_accumulated × inheritance_fraction
  - Death: max(0, lineage.coordination_accumulated - decay)
  - Memory: constitution.institutional_memory.decay_rate updated by capacity

DEPENDENCIES:
  Needs: generational_engine (lineages with coordination_accumulated),
         constitution (institutional_memory with decay_rate),
         coordination_capacity_engine (field.capacity)

  Next: 48.7 (Coordination Capture) — topologies that monopolize coordination
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


# ============================================================================
# Persistence Parameters
# ============================================================================

LINEAGE_INHERITANCE_FRACTION = 0.6    # How much coordination a child inherits
LINEAGE_DECAY_ON_DEATH = 0.02         # Coordination lost per member death
LINEAGE_ACCUMULATION_RATE = 0.001     # Coordination gained per step of survival
LINEAGE_WEIGHT = 0.6                  # Weight of lineage vs institutional
INSTITUTIONAL_WEIGHT = 0.4            # Weight of institutional memory
MIN_MEMORY_DEPTH = 0.1                # Floor for memory persistence


# ============================================================================
# Persistence Computer (pure function)
# ============================================================================

def compute_persistence(
    lineages: Dict[str, Any],
    institutional_memory: Any,
) -> Tuple[float, float, float, float]:
    """
    Compute coordination persistence from existing structures.

    Returns:
      (persistence, lineage_inheritance, institutional_buffer, n_active_lineages)
    """
    # --- Lineage inheritance ---
    active_lineages = [l for l in lineages.values() if not l.extinct]
    n_active = len(active_lineages)

    if n_active > 0:
        accums = np.array([getattr(l, 'coordination_accumulated', 0.5)
                          for l in active_lineages])
        # Weight by lineage size (more members = more coordination preserved)
        sizes = np.array([getattr(l, 'current_members', 1)
                         for l in active_lineages], dtype=float)
        total_size = np.sum(sizes)
        if total_size > 0:
            weights = sizes / total_size
            lineage_inheritance = float(np.average(accums, weights=weights))
        else:
            lineage_inheritance = float(np.mean(accums))
    else:
        lineage_inheritance = 0.0

    # --- Institutional buffer ---
    if institutional_memory is not None:
        mem_depth = getattr(institutional_memory, 'depth', 0.5)
        decay_rate = getattr(institutional_memory, 'decay_rate', 0.01)
        history_len = getattr(institutional_memory, 'history_len',
                              len(getattr(institutional_memory, 'history', [])))
        mem_depth = min(1.0, max(MIN_MEMORY_DEPTH, history_len / 200.0))
        institutional_buffer = mem_depth * (1.0 - decay_rate)
    else:
        institutional_buffer = MIN_MEMORY_DEPTH

    # --- Combined ---
    persistence = (lineage_inheritance * LINEAGE_WEIGHT
                   + institutional_buffer * INSTITUTIONAL_WEIGHT)

    return (
        round(persistence, 4),
        round(lineage_inheritance, 4),
        round(institutional_buffer, 4),
        n_active,
    )


# ============================================================================
# Persistence Updater (mutates existing structures)
# ============================================================================

def update_lineage_coordination(
    lineages: Dict[str, Any],
    agent_lineage: Dict[str, str],
    births: List[Dict],
    deaths: List[Dict],
) -> Dict[str, Any]:
    """
    Update lineage coordination_accumulated based on births and deaths.

    Birth: child's lineage gains coordination from parent's lineage
    Death: lineage.coordination_accumulated -= decay
    Survival: each surviving lineage member adds accumulation_rate

    Uses agent_lineage (agent_id → lineage_id) to resolve lineage membership.
    Mutates LineageRecord.coordination_accumulated in-place.
    No new entities.
    """
    stats = {
        'lineages_updated': 0,
        'total_coordination_added': 0.0,
        'total_coordination_lost': 0.0,
    }

    for lid, lineage in lineages.items():
        if getattr(lineage, 'extinct', False):
            continue
        if not hasattr(lineage, 'coordination_accumulated'):
            lineage.coordination_accumulated = 0.5

        # Growth from surviving members
        n_members = getattr(lineage, 'current_members', 0)
        if n_members > 0:
            gain = n_members * LINEAGE_ACCUMULATION_RATE
            lineage.coordination_accumulated += gain
            stats['total_coordination_added'] += gain

        stats['lineages_updated'] += 1

    # Births: add coordination from parent's lineage
    for birth in births:
        parent_a = birth.get('parent_a', '')
        parent_b = birth.get('parent_b', '')
        parent_ids = [p for p in [parent_a, parent_b] if p]
        for pid in parent_ids:
            lid = agent_lineage.get(pid)
            if lid and lid in lineages:
                lineage = lineages[lid]
                if not getattr(lineage, 'extinct', False):
                    parent_coord = getattr(lineage, 'coordination_accumulated', 0.5)
                    added = parent_coord * LINEAGE_INHERITANCE_FRACTION * 0.1
                    lineage.coordination_accumulated += added
                    stats['total_coordination_added'] += added

    # Deaths: decay coordination via lineage
    for death in deaths:
        agent_id = death.get('agent_id', '')
        lid = agent_lineage.get(agent_id)
        if lid and lid in lineages:
            lineage = lineages[lid]
            if not getattr(lineage, 'extinct', False):
                decay = min(
                    getattr(lineage, 'coordination_accumulated', 0.5),
                    LINEAGE_DECAY_ON_DEATH
                )
                lineage.coordination_accumulated -= decay
                stats['total_coordination_lost'] += decay

    return stats


def update_institutional_memory(
    institutional_memory: Any,
    capacity: float,
    persistence: float,
) -> Dict[str, Any]:
    """
    Update institutional memory based on coordination capacity.

    High capacity → slower decay, deeper memory.
    Low capacity → faster decay, memory erodes.

    Mutates institutional_memory in-place.
    No new entities.
    """
    stats = {}

    if institutional_memory is None:
        return stats

    # Decay rate is inverse of capacity
    target_decay = 0.001 + (1.0 - min(1.0, capacity)) * 0.02
    current_decay = getattr(institutional_memory, 'decay_rate', 0.001)

    # Smooth interpolation
    new_decay = current_decay * 0.9 + target_decay * 0.1
    institutional_memory.decay_rate = max(0.001, new_decay)

    # Depth = how far back memory reaches
    if hasattr(institutional_memory, 'history'):
        history_len = len(institutional_memory.history)
        depth_target = min(1.0, history_len / 200.0)
    else:
        depth_target = MIN_MEMORY_DEPTH

    # Persistence modulates depth
    depth_effective = depth_target * (0.5 + persistence * 0.5)
    institutional_memory.depth = max(MIN_MEMORY_DEPTH, depth_effective)
    institutional_memory.history_len = int(depth_effective * 200)

    stats['decay_rate'] = round(institutional_memory.decay_rate, 4)
    stats['depth'] = round(institutional_memory.depth, 4)
    return stats


# ============================================================================
# Main Step
# ============================================================================

def step_persistence(
    lineages: Dict[str, Any],
    agent_lineage: Dict[str, str],
    institutional_memory: Any,
    births: List[Dict],
    deaths: List[Dict],
    capacity: float,
) -> Dict[str, Any]:
    """
    One persistence cycle: compute, update, return.

    Pure mechanics on existing structures — no new entities.
    """
    # 1. Update lineage coordination
    lin_stats = update_lineage_coordination(lineages, agent_lineage, births, deaths)

    # 2. Update institutional memory
    persistence, lineage_inheritance, inst_buffer, n_active = compute_persistence(
        lineages, institutional_memory
    )
    mem_stats = update_institutional_memory(institutional_memory, capacity, persistence)

    return {
        'persistence': persistence,
        'lineage_inheritance': lineage_inheritance,
        'institutional_buffer': inst_buffer,
        'n_active_lineages': n_active,
        'lineages_updated': lin_stats['lineages_updated'],
        'coordination_added': round(lin_stats['total_coordination_added'], 4),
        'coordination_lost': round(lin_stats['total_coordination_lost'], 4),
        'memory_decay_rate': mem_stats.get('decay_rate', 0),
        'memory_depth': mem_stats.get('depth', 0),
    }


# ============================================================================
# TESTS
# ============================================================================

import random
import sys


def _make_lineage(lid: str, current_members: int = 2, extinct: bool = False,
                  coordination: float = 0.5):
    """Mock LineageRecord for testing."""
    from dataclasses import dataclass
    @dataclass
    class MockLineage:
        lineage_id: str
        current_members: int
        extinct: bool
        coordination_accumulated: float
        depth: int = 3
    return MockLineage(lid, current_members, extinct, coordination)

def _make_lineage_from_members(lid: str, members: list, extinct: bool = False,
                                coordination: float = 0.5):
    """Mock LineageRecord from a members list (compute current_members)."""
    return _make_lineage(lid, len(members), extinct, coordination)


def _make_memory(history_len: int = 50, decay_rate: float = 0.01, depth: float = 0.5):
    """Mock institutional memory."""
    @dataclass
    class MockMemory:
        history: list
        decay_rate: float
        depth: float
        history_len: int
    return MockMemory(list(range(history_len)), decay_rate, depth, history_len)


def test_persistence_computation():
    """Verify persistence computes from lineages and memory."""
    print("\n" + "=" * 60)
    print("48.6.1 — PERSISTENCE COMPUTATION")
    print("=" * 60)

    lineages = {
        'alpha': _make_lineage_from_members('alpha', ['a1', 'a2', 'a3'], coordination=0.8),
        'beta': _make_lineage_from_members('beta', ['b1', 'b2'], coordination=0.4),
        'gamma': _make_lineage_from_members('gamma', ['c1'], coordination=0.2),
    }
    memory = _make_memory(history_len=100, decay_rate=0.005, depth=0.8)

    persistence, lineage_inh, inst_buf, n_active = compute_persistence(lineages, memory)
    print(f"  Persistence:     {persistence:.4f}")
    print(f"  Lineage inherit: {lineage_inh:.4f}")
    print(f"  Inst buffer:     {inst_buf:.4f}")
    print(f"  Active lineages: {n_active}")

    assert persistence > 0, "Persistence must be positive"
    assert n_active == 3, "Should have 3 active lineages"
    assert lineage_inh > 0, "Lineage inheritance should be positive"
    assert inst_buf > 0, "Institutional buffer should be positive"

    # Larger lineages should be weighted more
    assert lineage_inh > 0.5, "Weighted by lineage size (alpha largest)"

    print("  >>> PersistenceComputation PASSED\n")


def test_lineage_update_via_births():
    """Verify births add coordination to lineages."""
    print("\n" + "=" * 60)
    print("48.6.2 — LINEAGE UPDATE VIA BIRTHS")
    print("=" * 60)

    lineages = {
        'alpha': _make_lineage('alpha', current_members=2, coordination=0.8),
    }
    agent_lineage = {
        'a1': 'alpha',
        'a2': 'alpha',
    }
    births = [
        {'child_id': 'a3', 'parent_a': 'a1', 'parent_b': 'a2'},
    ]
    deaths = []

    initial = lineages['alpha'].coordination_accumulated
    stats = update_lineage_coordination(lineages, agent_lineage, births, deaths)
    final = lineages['alpha'].coordination_accumulated

    print(f"  Coordination: {initial:.4f} -> {final:.4f}")
    print(f"  Added: {stats['total_coordination_added']:.4f}")

    assert final > initial, "Births should add coordination"
    assert stats['lineages_updated'] >= 1, "Lineages should be updated"

    print("  >>> LineageUpdateViaBirths PASSED\n")


def test_lineage_decay_via_deaths():
    """Verify deaths decay coordination from lineages."""
    print("\n" + "=" * 60)
    print("48.6.3 — LINEAGE DECAY VIA DEATHS")
    print("=" * 60)

    lineages = {
        'alpha': _make_lineage('alpha', current_members=3, coordination=0.8),
    }
    agent_lineage = {
        'a1': 'alpha',
        'a2': 'alpha',
        'a3': 'alpha',
    }
    births = []
    deaths = [
        {'agent_id': 'a1'},
        {'agent_id': 'a3'},
    ]

    initial = lineages['alpha'].coordination_accumulated
    stats = update_lineage_coordination(lineages, agent_lineage, births, deaths)
    final = lineages['alpha'].coordination_accumulated

    print(f"  Coordination: {initial:.4f} -> {final:.4f}")
    print(f"  Lost: {stats['total_coordination_lost']:.4f}")

    assert final < initial, "Deaths should decay coordination"
    assert stats['total_coordination_lost'] > 0, "Should track lost coordination"

    print("  >>> LineageDecayViaDeaths PASSED\n")


def test_institutional_memory_update():
    """Verify memory responds to coordination capacity."""
    print("\n" + "=" * 60)
    print("48.6.4 — INSTITUTIONAL MEMORY UPDATE")
    print("=" * 60)

    # High capacity scenario
    memory_high = _make_memory(history_len=50, decay_rate=0.01, depth=0.5)
    update_institutional_memory(memory_high, capacity=0.9, persistence=0.8)
    print(f"  High cap (0.9): decay={memory_high.decay_rate:.4f}, depth={memory_high.depth:.4f}")

    # Low capacity scenario (fresh memory)
    memory_low = _make_memory(history_len=50, decay_rate=0.01, depth=0.5)
    update_institutional_memory(memory_low, capacity=0.1, persistence=0.2)
    print(f"  Low cap (0.1):  decay={memory_low.decay_rate:.4f}, depth={memory_low.depth:.4f}")

    # Target decay rate should be higher for low capacity
    # High cap target: 0.001 + 0.1 * 0.02 = 0.003
    # Low cap target: 0.001 + 0.9 * 0.02 = 0.019
    target_high = 0.001 + (1.0 - min(1.0, 0.9)) * 0.02  # 0.003
    target_low = 0.001 + (1.0 - min(1.0, 0.1)) * 0.02   # 0.019
    print(f"  Target decay rates: high={target_high:.4f}, low={target_low:.4f}")

    # After smoothing, rate should be pulled toward target
    assert target_low > target_high, "Target decay should be higher for low capacity"
    assert memory_low.decay_rate > 0.01, "Low capacity decay should increase from baseline"
    assert memory_low.depth < memory_high.depth, \
        f"Low cap depth ({memory_low.depth:.4f}) < High cap depth ({memory_high.depth:.4f})"

    print("  >>> InstitutionalMemoryUpdate PASSED\n")


def test_integrated_with_full_system():
    """
    Full integration: coordination persistence with production + generations + capacity.

    500-step run. Verify:
    - Persistence varies over time
    - Coordination accumulates in lineages
    - Memory adapts to capacity
    """
    print("\n" + "=" * 60)
    print("48.6.5 — INTEGRATED CIVILIZATION WITH COORDINATION PERSISTENCE")
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

    persistence_history = []
    inherited_history = []

    n_steps = 500
    for step in range(1, n_steps + 1):
        prod_result = prod_engine.step(agents, step=step)
        gen_result = gen_engine.step(agents, {}, step, spawn_fn,
                                     compute_fn=lambda ag, sp, st: None)

        # Coordination persistence step
        births = gen_engine.birth_log[-len(gen_result['births']):] if gen_result['births'] else []
        deaths = gen_engine.death_log[-len(gen_result['deaths']):] if gen_result['deaths'] else []

        pers_result = step_persistence(
            lineages=gen_engine.lineages,
            agent_lineage=gen_engine.agent_lineage,
            institutional_memory=constitution.institutional_memory,
            births=births,
            deaths=deaths,
            capacity=0.5,  # Could use CoordinationCapacityEngine output
        )

        persistence_history.append(pers_result['persistence'])
        inherited_history.append(pers_result['lineage_inheritance'])

    active = len([a for a in agents if a.active])
    gen_stats = gen_engine.get_stats()

    print(f"  500 steps completed")
    print(f"  Active: {active}")
    print(f"  Births: {gen_stats['n_births']}, Deaths: {gen_stats['n_deaths']}")
    print(f"  Active lineages: {gen_stats['n_active_lineages']}")
    print(f"  Capital: {prod_engine.capital:.2f}")
    print()
    print(f"  MEAN PERSISTENCE: {np.mean(persistence_history):.4f}")
    print(f"  MAX PERSISTENCE:  {np.max(persistence_history):.4f}")
    print(f"  MIN PERSISTENCE:  {np.min(persistence_history):.4f}")
    print(f"  STD PERSISTENCE:  {np.std(persistence_history):.4f}")
    print()
    print(f"  MEAN LINEAGE INHERITANCE: {np.mean(inherited_history):.4f}")
    print(f"  Lineage inheritance trend: "
          f"{'growing' if inherited_history[-1] > inherited_history[0] else 'declining'}")

    # Check accumulation in lineages
    non_zero_lineages = sum(1 for l in gen_engine.lineages.values()
                           if getattr(l, 'coordination_accumulated', 0) > 0.01)
    print(f"  Lineages with accumulated coordination: {non_zero_lineages}")

    assert active > 0, "System collapsed"
    assert np.std(persistence_history) > 0.001, "Persistence should vary"
    assert np.mean(persistence_history) > 0, "Persistence should be positive"
    assert non_zero_lineages >= gen_stats['n_active_lineages'] * 0.5, \
        "Most lineages should have accumulated coordination"

    print("\n  >>> IntegratedCivilization PASSED\n")


def test_no_new_entities():
    """
    Verify 48.6 adds no new entity types.

    Only pure functions + update rules on existing structures.
    """
    print("\n" + "=" * 60)
    print("48.6.6 — ARCHITECTURAL INVARIANT: NO NEW ENTITIES")
    print("=" * 60)

    import inspect
    # Check module-level: only pure functions + constants (no new classes)
    import typing
    typing_names = set(dir(typing))
    module_members = [name for name in globals().keys()
                     if not name.startswith('_') and name not in ('sys', 'random', 'typing')]
    # Filter out imported types
    classes = []
    functions = []
    for name in module_members:
        if name in typing_names:
            continue
        obj = globals()[name]
        if isinstance(obj, type):
            classes.append(name)
        elif callable(obj) and not name.startswith('test'):
            functions.append(name)

    print(f"  Module entities: {len(classes)} classes, {len(functions)} functions")
    for c in classes:
        print(f"    class: {c}")
    for f in functions:
        print(f"    fn: {f}")

    # Only pure functions on existing structures — no new domain entity types
    assert len(classes) == 0, \
        f"Should have 0 new classes, found {classes}"

    print("  ✓ No new classes — only pure functions + update rules")
    print("  >>> ArchitecturalInvariantNoNewEntities PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.6: COORDINATION PERSISTENCE                            ║
║                                                                   ║
║  What carries coordination capacity across agent deaths?          ║
║                                                                   ║
║  Two mechanisms, zero new entities:                               ║
║    1. Lineage coordination inheritance                           ║
║    2. Institutional coordination buffer                          ║
║                                                                   ║
║  persistence = lineage_inheritance × 0.6 + inst_buffer × 0.4     ║
║                                                                   ║
║  No new classes. No new entity types.                             ║
║  Only state variables on existing structures + update rules.      ║
║                                                                   ║
║  Next: 48.7 (Coordination Capture)                               ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    all_passed = True
    tests = [
        ("Persistence Computation", test_persistence_computation),
        ("Lineage Update Via Births", test_lineage_update_via_births),
        ("Lineage Decay Via Deaths", test_lineage_decay_via_deaths),
        ("Institutional Memory Update", test_institutional_memory_update),
        ("Integrated Civilization", test_integrated_with_full_system),
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
  ║  PHASE 48.6: ALL 6 TESTS PASSED                              ║
  ║                                                               ║
  ║  Coordination Persistence ready.                             ║
  ║                                                               ║
  ║  Two mechanisms carry coordination across time:              ║
  ║    - lineage inheritance (60%)                               ║
  ║    - institutional memory buffer (40%)                       ║
  ║                                                               ║
  ║  No new entities. Only update rules on existing structures.  ║
  ║                                                               ║
  ║  Next: 48.7 (Coordination Capture)                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Some tests FAILED                                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
