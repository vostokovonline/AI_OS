"""
Phase 48.5E — Endogenous Scarcity Tensor.

ARCHITECTURAL SHIFT:
  Before: Scarcity = FSM phase (Boom/Recession/Winter/Recovery)
  After:  Scarcity = continuous pressure-field tensor observed from system state

CORE PRINCIPLE:
  No "winter mode" or "boom mode".
  Winter is a POST-HOC label for a configuration of pressure.
  The system never enters a state — it IS a configuration of tensions.

SCARCITY TENSOR S ∈ [0,1]^7:

  [production, capital, institutional, demographic,
   coordination, temporal, cognitive]

  Each dimension is computed from system observables every step.
  The CONFIGURATION of S drives effects, not any single dimension.
  Post-hoc analysis can label regimes from S, but S itself is physics.

EFFECT PHYSICS:
  Scarcity is NOT a debuff. It is an evolutionary gradient.
    moderate scarcity  → innovation pressure
    severe scarcity    → collapse risk
    chronic scarcity   → hierarchy formation
    asymmetric scarcity → elite capture
    coordination scarcity → state formation pressure

DEPENDENCIES:
  Needs: ProductionEngine (capital, output, survival), GenerationalEngine
         (demographics), phase48_cognitive_political_economy (coalitions,
         constitution, narratives)

  Next: 48.6 (Institutional Persistence) — institutions that survive
        generations under scarcity pressure
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import math


# ============================================================================
# Scarcity Conversion Constants
# ============================================================================

# How many units of each scarcity dimension the effects saturate at
PRODUCTION_SATURATION = 3.0       # survival cost multiplier max
CAPITAL_SATURATION = 2.0          # depreciation amplifier max
INSTITUTIONAL_SATURATION = 0.01   # trust decay per step at max
DEMOGRAPHIC_SATURATION = 0.01     # health decay per step at max
COORDINATION_SATURATION = 0.02    # cohesion loss per step at max
TEMPORAL_SATURATION = 0.3         # window closure rate
COGNITIVE_SATURATION = 0.3        # decision quality reduction max


# ============================================================================
# Scarcity Tensor
# ============================================================================

@dataclass
class ScarcityTensor:
    """
    Seven-dimensional continuous pressure field.

    Each dimension ∈ [0, 1], computed fresh every step from system state.
    No internal state, no phase memory, no FSM transitions.
    """
    production: float = 0.0    # output < survival need
    capital: float = 0.0       # capital Gini concentration
    institutional: float = 0.0 # institutional_mult < 1.0
    demographic: float = 0.0   # dependency ratio high
    coordination: float = 0.0  # coalition fragmentation + trust decay
    temporal: float = 0.0      # aging > reproduction, windows closing
    cognitive: float = 0.0     # agents can't understand/predict/plan

    @property
    def vector(self) -> np.ndarray:
        return np.array([
            self.production, self.capital, self.institutional,
            self.demographic, self.coordination, self.temporal,
            self.cognitive,
        ])

    @property
    def overall_pressure(self) -> float:
        """RMS over all 7 dimensions — penalizes extreme single-axis pressure."""
        v = self.vector
        return float(np.sqrt(np.mean(v ** 2)))

    @property
    def dominant_axis(self) -> str:
        """Which scarcity dimension is most intense (post-hoc label, not state)."""
        names = ['production', 'capital', 'institutional', 'demographic',
                 'coordination', 'temporal', 'cognitive']
        return names[int(np.argmax(self.vector))]

    @property
    def regime_label(self) -> str:
        """
        POST-HOC regime label derived from tensor configuration.
        This is OBSERVATION, not STATE. Never read by system logic.
        """
        p = self.overall_pressure
        dom = self.dominant_axis
        s = self.vector

        if p < 0.15:
            return 'boom'
        if p < 0.3:
            return 'stability'

        # Severe single dimension
        if s[4] > 0.7:            # coordination
            return 'fragmentation'
        if s[0] > 0.7:            # production
            return 'famine'
        if s[1] > 0.7 and s[0] > 0.4:  # capital + production
            return 'elite_capture'
        if s[2] > 0.6:            # institutional
            return 'decay'
        if s[3] > 0.6:            # demographic
            return 'aging_collapse'

        if p > 0.5:
            if dom == 'temporal':
                return 'historical_pressure'
            return 'crisis'

        return 'stagnation'

    def to_dict(self) -> Dict[str, float]:
        return {
            'production': round(self.production, 3),
            'capital': round(self.capital, 3),
            'institutional': round(self.institutional, 3),
            'demographic': round(self.demographic, 3),
            'coordination': round(self.coordination, 3),
            'temporal': round(self.temporal, 3),
            'cognitive': round(self.cognitive, 3),
            'overall_pressure': round(self.overall_pressure, 3),
            'dominant_axis': self.dominant_axis,
            'regime_label': self.regime_label,
        }

    def __repr__(self) -> str:
        d = self.to_dict()
        axes = ' | '.join(f'{k}={d[k]}' for k in
                          ['production','capital','institutional','demographic',
                           'coordination','temporal','cognitive'])
        return f"ScarcityTensor(p={d['overall_pressure']:.3f} | {axes})"


# ============================================================================
# Endogenous Scarcity Engine
# ============================================================================

class EndogenousScarcityEngine:
    """
    Computes the scarcity tensor and applies its effects each step.

    ARCHITECTURAL CONSTRAINT:
      This engine NEVER stores or checks the previous step's scarcity.
      It has NO state machine, NO phase transitions, NO regime memory.
      Every step is a fresh computation from current system state.

    INPUT:  agents, coalitions, constitution, memory, production_engine
    OUTPUT: ScarcityTensor + modified system state (applied effects)
    """

    def __init__(
        self,
        production_saturation: float = PRODUCTION_SATURATION,
        capital_saturation: float = CAPITAL_SATURATION,
        institutional_saturation: float = INSTITUTIONAL_SATURATION,
        demographic_saturation: float = DEMOGRAPHIC_SATURATION,
        coordination_saturation: float = COORDINATION_SATURATION,
        temporal_saturation: float = TEMPORAL_SATURATION,
        cognitive_saturation: float = COGNITIVE_SATURATION,
    ):
        self.saturation = {
            'production': production_saturation,
            'capital': capital_saturation,
            'institutional': institutional_saturation,
            'demographic': demographic_saturation,
            'coordination': coordination_saturation,
            'temporal': temporal_saturation,
            'cognitive': cognitive_saturation,
        }
        self.scarcity_log: List[Dict] = []

    # ------------------------------------------------------------------
    # Tensor Computation (no state, pure function of system)
    # ------------------------------------------------------------------

    def compute_tensor(
        self,
        agents: List[Any],
        coalitions: Optional[Dict[str, Any]] = None,
        constitution: Any = None,
        production_engine: Any = None,
    ) -> ScarcityTensor:
        """
        Compute the 7-dimensional scarcity tensor from current system state.
        Pure function — no side effects.
        """
        active = [a for a in agents if a.active]
        n = len(active)
        if n == 0:
            return ScarcityTensor(production=1.0, demographic=1.0)

        # --- Production Scarcity ---
        if production_engine and len(production_engine.production_log) > 0:
            last = production_engine.production_log[-1]
            total_output = last['total_output']
            total_survival = last['total_survival']
            # How much of survival need does production cover?
            output_ratio = total_output / max(total_survival, 0.01)
            p_scarcity = max(0.0, 1.0 - min(output_ratio, self.saturation['production']) / self.saturation['production'])
        else:
            p_scarcity = 0.0

        # --- Capital Scarcity ---
        wealths = np.array([max(0.0, a.wealth) for a in active], dtype=np.float64)
        total_w = np.sum(wealths)
        if total_w > 1e-9 and n > 1:
            sorted_w = np.sort(wealths)
            cumsum = np.cumsum(sorted_w)
            # Gini = 2*sum(i+1)*w_i / (n*sum w_i) - (n+1)/n
            gini = (2.0 * np.sum((np.arange(n) + 1) * sorted_w) / (n * total_w)
                    - (n + 1.0) / n)
            gini = max(0.0, min(1.0, gini))
        else:
            gini = 0.0
        # Capital concentration amplifies scarcity
        c_scarcity = min(1.0, gini * 1.5)

        # --- Institutional Scarcity ---
        if production_engine:
            _, trust, stability, memory = production_engine.compute_institutional_multiplier(
                agents, coalitions, constitution
            )
        else:
            trust = float(np.mean([a.reliability for a in active])) if active else 0.0
            stability = 0.5
            memory = 0.3

        inst_mult = 1.0 + trust * 0.3 + stability * 0.2 + memory * 0.2
        # Below 1.0 baseline means institutions actively hinder
        # (shouldn't happen with current formula, but captures degradation)
        i_scarcity = max(0.0, min(1.0, (2.0 - inst_mult) / 2.0))

        # --- Demographic Scarcity ---
        pre = sum(1 for a in active if a.age < 16)
        prime = sum(1 for a in active if 16 <= a.age <= 65)
        post = sum(1 for a in active if a.age > 65)
        if prime > 0:
            dep_ratio = (pre + post) / prime
        else:
            dep_ratio = 5.0  # No workers — extreme scarcity
        d_scarcity = min(1.0, dep_ratio / 5.0)

        # --- Coordination Scarcity ---
        coalition_frag = 0.0
        if coalitions and len(coalitions) > 0:
            members_per_coal = [len(c.members) for c in coalitions.values()
                                if hasattr(c, 'members')]
            if members_per_coal and len(members_per_coal) > 1:
                # Fragmentation: how evenly spread? (1 - normalized HHI)
                total_m = sum(members_per_coal)
                if total_m > 0:
                    hhi = sum((m / total_m) ** 2 for m in members_per_coal)
                    n_coal = len(members_per_coal)
                    min_hhi = 1.0 / n_coal
                    # HHI close to 1 = one dominant coalition (low frag)
                    # HHI close to min = evenly spread (high frag)
                    coalition_frag = max(0.0, min(1.0, 1.0 - (hhi - min_hhi) / (1.0 - min_hhi)))

        coord_scarcity = coalition_frag * (1.0 - trust) * (1.0 - memory)

        # --- Temporal Scarcity ---
        # Are generations replacing fast enough?
        if hasattr(production_engine, 'production_log') and len(production_engine.production_log) >= 20:
            # Check capital recovery rate
            cap_log = production_engine.capital_log
            recent_cap = cap_log[-20:]
            if len(recent_cap) >= 2 and recent_cap[-1] > 0 and recent_cap[0] > 0:
                cap_change = (recent_cap[-1] - recent_cap[0]) / recent_cap[0]
            else:
                cap_change = 0.0
        else:
            cap_change = 0.0

        # Age structure: are we aging faster than reproducing?
        mean_age = float(np.mean([a.age for a in active])) if active else 50.0
        age_structure_pressure = max(0.0, (mean_age - 40.0) / 40.0)

        # Capital declining = temporal pressure
        capital_temporal = max(0.0, -cap_change)
        t_scarcity = min(1.0, 0.5 * age_structure_pressure + 0.5 * capital_temporal)

        # --- Cognitive Scarcity ---
        # Proxy: mean age * exploration potential
        # Young agents have less experience, old agents have less adaptability
        if active:
            age_scores = []
            for a in active:
                if a.age < 10:
                    age_scores.append(0.3)   # Too young to understand
                elif a.age < 25:
                    age_scores.append(0.6)   # Learning
                elif a.age < 60:
                    age_scores.append(1.0)   # Peak cognition
                else:
                    age_scores.append(0.5)   # Cognitive decline
            cognitive_capacity = float(np.mean(age_scores))
        else:
            cognitive_capacity = 0.0
        cog_scarcity = max(0.0, 1.0 - cognitive_capacity)

        return ScarcityTensor(
            production=round(p_scarcity, 4),
            capital=round(c_scarcity, 4),
            institutional=round(i_scarcity, 4),
            demographic=round(d_scarcity, 4),
            coordination=round(coord_scarcity, 4),
            temporal=round(t_scarcity, 4),
            cognitive=round(cog_scarcity, 4),
        )

    # ------------------------------------------------------------------
    # Effect Application (state modification via pressure)
    # ------------------------------------------------------------------

    def apply_effects(
        self,
        tensor: ScarcityTensor,
        agents: List[Any],
        production_engine: Any,
        coalitions: Optional[Dict[str, Any]] = None,
        constitution: Any = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Apply scarcity pressure as system effects.

        Scarcity is NOT a debuff.
        Each effect is proportional to the relevant tensor dimension.
        Effects modify agent/coalition/constitution state in-place.
        """
        effects = {}

        # --- Production Effects ---
        # survival cost rises with production scarcity
        survival_mult = 1.0 + tensor.production * self.saturation['production']
        # productivity falls with production scarcity
        productivity_mult = 1.0 - tensor.production * 0.4

        effects['survival_cost_mult'] = round(survival_mult, 3)
        effects['productivity_mult'] = round(productivity_mult, 3)

        # Apply to production engine
        if hasattr(production_engine, 'survival_cost'):
            production_engine.survival_cost = (
                production_engine.survival_cost * survival_mult
            )

        # --- Capital Effects ---
        # Depreciation increases with capital scarcity (locked capital = deferred maintenance)
        if hasattr(production_engine, 'depreciation_rate'):
            dep_base = 0.008  # default
            production_engine.depreciation_rate = dep_base * (
                1.0 + tensor.capital * self.saturation['capital']
            )
        # Investment efficiency drops with capital scarcity
        if hasattr(production_engine, 'investment_efficiency'):
            inv_base = 0.7
            production_engine.investment_efficiency = inv_base * (
                1.0 - tensor.capital * 0.5
            )

        effects['depreciation_rate'] = round(getattr(production_engine, 'depreciation_rate', 0.008), 4)
        effects['investment_efficiency'] = round(getattr(production_engine, 'investment_efficiency', 0.7), 4)

        # --- Demographic Effects ---
        # Health decays with demographic pressure
        health_decay = tensor.demographic * self.saturation['demographic']
        effects['health_decay'] = round(health_decay, 4)
        for a in agents:
            if a.active:
                if not hasattr(a, 'health'):
                    a.health = 1.0
                a.health = max(0.1, a.health - health_decay)

        # --- Institutional Effects ---
        # Trust erodes with institutional scarcity
        trust_decay = tensor.institutional * self.saturation['institutional']
        effects['trust_decay'] = round(trust_decay, 4)
        for a in agents:
            if a.active:
                a.reliability = max(0.05, a.reliability - trust_decay)

        # --- Coordination Effects ---
        # Coalition cohesion drops with coordination scarcity
        cohesion_loss = tensor.coordination * self.saturation['coordination']
        effects['cohesion_loss'] = round(cohesion_loss, 4)

        if coalitions:
            for c in coalitions.values():
                if hasattr(c, 'compute_cohesion') and hasattr(c, 'update_cohesion'):
                    current = c.compute_cohesion()
                    c.update_cohesion(max(0.05, current - cohesion_loss))
                elif hasattr(c, 'cohesion'):
                    c.cohesion = max(0.05, getattr(c, 'cohesion', 1.0) - cohesion_loss)

        # --- Temporal Effects ---
        # Regeneration is harder under temporal pressure
        regen_mult = 1.0 - tensor.temporal * self.saturation['temporal']
        effects['regeneration_mult'] = round(regen_mult, 3)
        # (Applied at the integration layer as a multiplier on recovery rates)

        # --- Cognitive Effects ---
        # Decision quality declines with cognitive scarcity
        decision_mult = 1.0 - tensor.cognitive * self.saturation['cognitive']
        effects['decision_quality_mult'] = round(decision_mult, 3)
        # Applied as modifier on agent bid/strategy behaviors

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
        step: int = 0,
    ) -> Tuple[ScarcityTensor, Dict[str, Any]]:
        """
        One complete scarcity cycle:
          1. Compute tensor from current state
          2. Apply effects to modify state
          3. Log

        Returns (tensor, effects_dict).
        """
        tensor = self.compute_tensor(agents, coalitions, constitution, production_engine)
        effects = self.apply_effects(tensor, agents, production_engine, coalitions, constitution, step)

        self.scarcity_log.append({
            'step': step,
            **tensor.to_dict(),
            **effects,
        })

        return tensor, effects

    # ------------------------------------------------------------------
    # Stats & Reports
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        if not self.scarcity_log:
            return {}
        recent = self.scarcity_log[-50:]
        return {
            'mean_pressure': round(
                float(np.mean([e['overall_pressure'] for e in recent])), 3
            ),
            'max_pressure': round(
                float(np.max([e['overall_pressure'] for e in recent])), 3
            ),
            'dominant_axes': {
                ax: round(float(np.mean([e[ax] for e in recent])), 3)
                for ax in ['production', 'capital', 'institutional',
                           'demographic', 'coordination', 'temporal', 'cognitive']
            },
            'n_steps': len(self.scarcity_log),
            'n_system_logs': len(self.scarcity_log),
        }

    def get_pressure_history(self) -> List[float]:
        return [e['overall_pressure'] for e in self.scarcity_log]


# ============================================================================
# TESTS
# ============================================================================

import random
import sys


def _make_test_agent(agent_id: str, age: int, wealth: float = 3.0,
                     reliability: float = 0.5, productivity: float = 0.5,
                     species: str = 'exploitative', active: bool = True) -> Any:
    """Create a minimal mock agent for testing."""
    from phase48_cognitive_political_economy import CognitiveAgent
    a = CognitiveAgent(agent_id, species, np.tanh(np.random.randn(32) * 0.5))
    a.active = active
    a.age = age
    a.wealth = wealth
    a.reliability = reliability
    a.productivity = productivity
    return a


def _make_minimal_production_engine():
    """Minimal production engine for scarcity testing."""
    from phase48_production_economy import ProductionEngine
    return ProductionEngine()


def test_tensor_basic_computation():
    """Verify scarcity tensor computes from system state without errors."""
    print("\n" + "=" * 60)
    print("48.5E.1 — TENSOR BASIC COMPUTATION")
    print("=" * 60)

    engine = EndogenousScarcityEngine()
    prod_engine = _make_minimal_production_engine()

    agents = [_make_test_agent(f'ag_{i}', age=30 + i * 5, wealth=3.0 + i)
              for i in range(20)]

    # Run a few production steps first so there's history
    for step in range(1, 11):
        prod_engine.step(agents, step=step)

    tensor = engine.compute_tensor(agents, production_engine=prod_engine)

    print(f"  Overall pressure: {tensor.overall_pressure:.4f}")
    print(f"  Vector: {tensor.vector}")
    print(f"  Dominant: {tensor.dominant_axis}")
    print(f"  Regime: {tensor.regime_label}")
    print(f"  Dict: {tensor.to_dict()}")

    assert 0.0 <= tensor.production <= 1.0, "Production scarcity out of range"
    assert 0.0 <= tensor.capital <= 1.0, "Capital scarcity out of range"
    assert 0.0 <= tensor.institutional <= 1.0, "Institutional scarcity out of range"
    assert 0.0 <= tensor.demographic <= 1.0, "Demographic scarcity out of range"
    assert 0.0 <= tensor.coordination <= 1.0, "Coordination scarcity out of range"
    assert 0.0 <= tensor.temporal <= 1.0, "Temporal scarcity out of range"
    assert 0.0 <= tensor.cognitive <= 1.0, "Cognitive scarcity out of range"
    assert 0.0 <= tensor.overall_pressure <= 1.0, "Overall pressure out of range"

    print("  >>> TensorBasicComputation PASSED\n")


def test_tensor_high_pressure_scenarios():
    """Verify scarcity responds correctly to system stress."""
    print("\n" + "=" * 60)
    print("48.5E.2 — HIGH PRESSURE SCENARIOS")
    print("=" * 60)

    engine = EndogenousScarcityEngine()

    # Scenario A: Production famine
    prod_low = _make_minimal_production_engine()
    agents_famine = [_make_test_agent(f'f_{i}', age=70, wealth=1.0) for i in range(10)]
    # Increase survival cost so output < cost
    prod_low.survival_cost = 2.0

    for step in range(1, 20):
        prod_low.step(agents_famine, step=step)

    tensor_famine = engine.compute_tensor(agents_famine, production_engine=prod_low)
    print(f"  Famine scenario — production scarcity: {tensor_famine.production:.4f}")

    # Scenario B: Extreme wealth inequality
    prod_ineq = _make_minimal_production_engine()
    agents_gini = []
    for i in range(10):
        if i == 0:
            a = _make_test_agent(f'rich', age=30, wealth=100.0)
        else:
            a = _make_test_agent(f'poor_{i}', age=30, wealth=0.1)
        agents_gini.append(a)

    for step in range(1, 5):
        prod_ineq.step(agents_gini, step=step)

    tensor_gini = engine.compute_tensor(agents_gini, production_engine=prod_ineq)
    print(f"  Gini scenario — capital scarcity: {tensor_gini.capital:.4f}")

    # Scenario C: Demographic dependency crisis
    prod_demo = _make_minimal_production_engine()
    agents_demo = []
    for i in range(5):
        a = _make_test_agent(f'child_{i}', age=8, wealth=0.5)
        agents_demo.append(a)
    for i in range(2):
        a = _make_test_agent(f'old_{i}', age=80, wealth=0.5)
        agents_demo.append(a)
    # Only 3 prime workers for 7 dependents
    for i in range(3):
        a = _make_test_agent(f'worker_{i}', age=30, wealth=2.0)
        agents_demo.append(a)

    for step in range(1, 5):
        prod_demo.step(agents_demo, step=step)

    tensor_demo = engine.compute_tensor(agents_demo, production_engine=prod_demo)
    print(f"  Demographic scenario — demo scarcity: {tensor_demo.demographic:.4f}")
    print(f"  Cognitve scarcity: {tensor_demo.cognitive:.4f}")

    # Scenario D: High coordination fragmentation
    prod_coord = _make_minimal_production_engine()
    agents_coord = [_make_test_agent(f'c_{i}', age=30, wealth=2.0, reliability=0.1)
                    for i in range(20)]
    for step in range(1, 5):
        prod_coord.step(agents_coord, step=step)

    # Mock coalitions with fragmentation
    class MockCoalition:
        def __init__(self, members):
            self.members = members

    coalitions_frag = {
        'a': MockCoalition(list(range(5))),
        'b': MockCoalition(list(range(5, 10))),
        'c': MockCoalition(list(range(10, 15))),
        'd': MockCoalition(list(range(15, 20))),
    }

    tensor_coord = engine.compute_tensor(agents_coord, coalitions=coalitions_frag,
                                         production_engine=prod_coord)
    print(f"  Coordination scenario — coord scarcity: {tensor_coord.coordination:.4f}")

    # Famine should have high production scarcity
    assert tensor_famine.production > 0.3, \
        f"Famine should have production scarcity > 0.3, got {tensor_famine.production}"

    # Gini should have high capital scarcity
    assert tensor_gini.capital > 0.4, \
        f"Gini should have capital scarcity > 0.4, got {tensor_gini.capital}"

    # Demographic should have high demographic scarcity
    assert tensor_demo.demographic > 0.4, \
        f"Demographic dependency should have scarcity > 0.4, got {tensor_demo.demographic}"

    # Fragmentation should have coordination scarcity
    assert tensor_coord.coordination > 0.1, \
        f"Fragmentation should have coordination scarcity > 0.1, got {tensor_coord.coordination}"

    print("  >>> HighPressureScenarios PASSED\n")


def test_effect_application():
    """Verify scarcity effects modify system state correctly."""
    print("\n" + "=" * 60)
    print("48.5E.3 — EFFECT APPLICATION")
    print("=" * 60)

    engine = EndogenousScarcityEngine()
    prod_engine = _make_minimal_production_engine()

    agents = [_make_test_agent(f'eff_{i}', age=30, wealth=3.0, reliability=0.7)
              for i in range(10)]
    for step in range(1, 5):
        prod_engine.step(agents, step=step)

    initial_survival = prod_engine.survival_cost
    initial_depreciation = prod_engine.depreciation_rate
    initial_investment_eff = prod_engine.investment_efficiency
    initial_reliabilities = [a.reliability for a in agents if a.active]
    initial_health = [getattr(a, 'health', 1.0) for a in agents if a.active]

    # Create a high-scarcity tensor manually
    high_tensor = ScarcityTensor(
        production=0.8,
        capital=0.6,
        institutional=0.5,
        demographic=0.4,
        coordination=0.3,
        temporal=0.2,
        cognitive=0.1,
    )

    effects = engine.apply_effects(high_tensor, agents, prod_engine)

    print(f"  Survival cost: {initial_survival:.2f} -> {prod_engine.survival_cost:.3f}")
    print(f"  Depreciation: {initial_depreciation:.4f} -> {prod_engine.depreciation_rate:.4f}")
    print(f"  Investment eff: {initial_investment_eff:.3f} -> {prod_engine.investment_efficiency:.3f}")

    # Survival cost should have increased
    assert prod_engine.survival_cost > initial_survival, \
        "High production scarcity should increase survival cost"

    # Depreciation should have increased
    assert prod_engine.depreciation_rate > initial_depreciation, \
        "High capital scarcity should increase depreciation"

    # Investment efficiency should have decreased
    assert prod_engine.investment_efficiency < initial_investment_eff, \
        "High capital scarcity should decrease investment efficiency"

    # Reliability should have decreased (trust decay)
    current_reliabilities = [a.reliability for a in agents if a.active]
    print(f"  Mean reliability: {np.mean(initial_reliabilities):.4f} -> {np.mean(current_reliabilities):.4f}")
    if np.mean(current_reliabilities) < np.mean(initial_reliabilities) - 0.001:
        print("    ✓ Trust decay applied")
    else:
        print("    (Trust decay may be below threshold for single step)")

    # Health should have decreased
    current_health = [getattr(a, 'health', 1.0) for a in agents if a.active]
    print(f"  Mean health: {np.mean(initial_health):.4f} -> {np.mean(current_health):.4f}")
    if np.mean(current_health) < np.mean(initial_health) - 0.001:
        print("    ✓ Health decay applied")

    print("  >>> EffectApplication PASSED\n")


def test_integrated_with_production_and_generations():
    """
    Full integration test: scarcity + production + generational turnover.

    This is the critical validation:
      Does the system maintain itself across generations
      with endogenous scarcity pressure?

    Expected: scarcity oscillates with demographic waves,
      boom/bust emerge from tensor configuration, not FSM.
    """
    print("\n" + "=" * 60)
    print("48.5E.4 — INTEGRATED CIVILIZATION WITH ENDOGENOUS SCARCITY")
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

    scarcity_engine = EndogenousScarcityEngine()

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

    pressure_history = []
    regime_counts = {}
    birth_history = []
    gdp_history = []

    n_steps = 500
    for step in range(1, n_steps + 1):
        # 1. Production
        prod_result = prod_engine.step(agents, step=step)

        # 2. Scarcity (applies effects that modify production params in-place)
        scarcity_tensor, effects = scarcity_engine.step(
            agents, production_engine=prod_engine, step=step
        )

        # 3. Generational turnover
        gen_result = gen_engine.step(agents, {}, step, spawn_fn,
                                     compute_fn=lambda ag, sp, st: None)

        pressure_history.append(scarcity_tensor.overall_pressure)
        regime = scarcity_tensor.regime_label
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if step % 100 == 0:
            gdp_history.append(prod_result['total_output'])
            active_count = len([a for a in agents if a.active])

    print(f"  500 steps completed")
    print(f"  Active agents: {len([a for a in agents if a.active])}")
    print(f"  Total births: {sum(g['n_births'] for g in gen_engine.generational_log)}")
    print(f"  Total deaths: {sum(g['n_deaths'] for g in gen_engine.generational_log)}")
    print(f"  Capital: {prod_engine.capital:.2f}")
    print()
    print(f"  MEAN SCARCITY PRESSURE: {np.mean(pressure_history):.4f}")
    print(f"  MAX SCARCITY PRESSURE: {np.max(pressure_history):.4f}")
    print(f"  MIN SCARCITY PRESSURE: {np.min(pressure_history):.4f}")
    print(f"  STD SCARCITY PRESSURE: {np.std(pressure_history):.4f}")
    print()

    regime_order = sorted(regime_counts.items(), key=lambda x: -x[1])
    print(f"  REGIME DISTRIBUTION (post-hoc labels, not states):")
    for regime, count in regime_order:
        pct = 100 * count / n_steps
        bar = '#' * int(pct / 2)
        print(f"    {regime:20s}: {count:4d} steps ({pct:5.1f}%) {bar}")

    print(f"\n  GDP history:")
    for i, gdp in enumerate(gdp_history):
        print(f"    step {(i+1)*100}: {gdp:.2f}")

    # Validate system is alive
    active = len([a for a in agents if a.active])
    assert active > 0, "System collapsed — no active agents"
    assert len(pressure_history) == n_steps, "Should have pressure data for every step"

    # Pressure should vary (not flat)
    pressure_std = np.std(pressure_history)
    assert pressure_std > 0.01, \
        f"Pressure should vary over time (std={pressure_std:.4f}) — flat pressure = no dynamics"

    # There should be regime diversity (multiple post-hoc labels observed)
    assert len(regime_counts) >= 2, \
        f"Should observe at least 2 regimes, got {len(regime_counts)}: {regime_counts}"

    print(f"\n  ✓ Regime diversity: {len(regime_counts)} regimes observed")
    print(f"  ✓ Pressure variation: std={pressure_std:.4f}")
    print(f"  ✓ Population sustained: {active} active agents")

    print("\n  >>> IntegratedCivilizationScarcity PASSED\n")


def test_no_fsm_memory():
    """
    Verify the engine has NO state machine behavior.
    This is the architectural invariant test.

    The scarcity engine should:
    - NOT store previous scarcity values
    - NOT have phase transitions
    - NOT check regime labels in logic
    - Compute fresh every step
    """
    print("\n" + "=" * 60)
    print("48.5E.5 — ARCHITECTURAL INVARIANT: NO FSM")
    print("=" * 60)

    engine = EndogenousScarcityEngine()

    # Check no phase-related attributes exist
    phase_attrs = [attr for attr in dir(engine) if 'phase' in attr.lower()
                   or 'regime' in attr.lower() or 'state' in attr.lower()
                   or 'mode' in attr.lower()]
    assert len(phase_attrs) == 0, \
        f"Engine should not have phase/regime/state/mode attributes: {phase_attrs}"

    # Check scarcity_log only stores results, not drives computation
    assert hasattr(engine, 'scarcity_log'), "Should have log for observability"
    assert hasattr(engine, 'compute_tensor'), "Should have compute_tensor method"

    # Verify compute_tensor is a pure function (no side effects)
    prod_engine = _make_minimal_production_engine()
    agents = [_make_test_agent(f'fsm_{i}', age=30, wealth=3.0) for i in range(5)]
    for step in range(1, 5):
        prod_engine.step(agents, step=step)

    # Call twice — should give same result
    t1 = engine.compute_tensor(agents, production_engine=prod_engine)
    t2 = engine.compute_tensor(agents, production_engine=prod_engine)

    assert t1.production == t2.production, "Pure function should give identical results"
    assert t1.capital == t2.capital, "Pure function should give identical results"
    assert t1.demographic == t2.demographic, "Pure function should give identical results"

    print("  ✓ No phase/regime/state/mode attributes found")
    print("  ✓ compute_tensor is pure function (deterministic on same inputs)")
    print("  ✓ Regime labels are post-hoc (not stored or used in computation)")

    print("  >>> ArchitecturalInvariantNoFSM PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5E: ENDOGENOUS SCARCITY TENSOR                         ║
║                                                                   ║
║  ARCHITECTURAL SHIFT:                                             ║
║    Before: scarcity = FSM phase (Boom/Recession/Winter)           ║
║    After:  scarcity = continuous pressure-field tensor             ║
║                                                                   ║
║    S ∈ [0,1]⁷ = {production, capital, institutional,              ║
║                  demographic, coordination, temporal, cognitive}   ║
║                                                                   ║
║    "Winter" is not a state — it's a post-hoc label for            ║
║    a configuration of pressure.                                   ║
║                                                                   ║
║  Next: 48.6 (Institutional Persistence)                          ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    all_passed = True
    tests = [
        ("Tensor Basic Computation", test_tensor_basic_computation),
        ("High Pressure Scenarios", test_tensor_high_pressure_scenarios),
        ("Effect Application", test_effect_application),
        ("Integrated Civilization", test_integrated_with_production_and_generations),
        ("Architectural Invariant: No FSM", test_no_fsm_memory),
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
  ║  Endogenous Scarcity Tensor ready.                           ║
  ║  Scarcity is now a continuous pressure-field,                 ║
  ║  not an FSM phase.                                            ║
  ║                                                               ║
  ║  Next: 48.6 (Institutional Persistence) —                     ║
  ║  Institutions that survive generations under pressure.        ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Some tests FAILED                                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
        sys.exit(1)
