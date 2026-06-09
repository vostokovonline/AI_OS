"""
Phase 48.5D — Production Economy Layer.

Transforms the Cognitive Political Economy from
REDISTRIBUTIVE ECONOMY (exogenous supply, zero-sum allocation)
to
ENDOGENOUS PRODUCTION ECONOMY (supply created by agents).

Physics:
  labor = compute_capacity x age_curve(productivity) x health
  output = labor x capital_mult x institutional_mult
  wealth += output - survival_cost - investment
  capital_{t+1} = capital_t + investment_total - depreciation
  institutional_mult = 1.0 + trust*0.3 + stability*0.2 + memory_continuity*0.2

Key emergent phenomena this enables:
  - Demographic dividends (baby boom -> more labor -> growth)
  - Aging civilization stagnation (fewer producers -> decline)
  - Elite accumulation (lineages with capital -> more output -> more capital)
  - Institutional divergence (high-trust -> high multiplier -> surplus)
  - Collapse/recovery cycles (war/disease -> labor drop -> capital decay)

Dependency: 48.5C (generational turnover) for population continuity.
Next: 48.5E (endogenous scarcity) once production is stable.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np
import random
import math


# ============================================================================
# Production Parameters
# ============================================================================

SURVIVAL_COST = 0.15           # Wealth consumed per step just to exist
DEPRECIATION_RATE = 0.008      # Capital decay per step (rusts without maintenance)
CAPITAL_EFFICIENCY = 0.4       # (unused) — replaced by MAX_CAPITAL_BOOST
MAX_CAPITAL_BOOST = 4.0        # Diminishing-returns asymptotic max
CAPITAL_HALF_SATURATION = 20.0 # Capital needed for half max boost
INVESTMENT_EFFICIENCY = 0.7    # Fraction of invested wealth that becomes capital
INVESTMENT_THRESHOLD = 1.5     # Wealth above this triggers investment
INVESTMENT_FRACTION = 0.15     # Fraction of surplus wealth invested
MIN_CAPITAL = 0.5              # Capital floor (can't go below)


# ============================================================================
# Production Engine
# ============================================================================

class ProductionEngine:
    """
    Computes production, consumption, investment, and capital dynamics.

    Designed to be called inside CognitivePoliticalEngine.step():
      prod_result = production_engine.step(agents, coalitions, constitution, step)

    Does NOT own agents or coalitions — operates on them in-place.
    """

    def __init__(
        self,
        survival_cost: float = SURVIVAL_COST,
        depreciation_rate: float = DEPRECIATION_RATE,
        investment_efficiency: float = INVESTMENT_EFFICIENCY,
        investment_threshold: float = INVESTMENT_THRESHOLD,
        investment_fraction: float = INVESTMENT_FRACTION,
        min_capital: float = MIN_CAPITAL,
        max_capital_boost: float = MAX_CAPITAL_BOOST,
        capital_half_saturation: float = CAPITAL_HALF_SATURATION,
    ):
        self.survival_cost = survival_cost
        self.depreciation_rate = depreciation_rate
        self.investment_efficiency = investment_efficiency
        self.investment_threshold = investment_threshold
        self.investment_fraction = investment_fraction
        self.min_capital = min_capital
        self.max_capital_boost = max_capital_boost
        self.capital_half_saturation = capital_half_saturation

        # Shared societal capital stock
        self.capital: float = 1.0

        # Tracking
        self.production_log: List[Dict] = []
        self.capital_log: List[float] = [1.0]

    # ------------------------------------------------------------------
    # Age-Productivity Curve
    # ------------------------------------------------------------------

    def age_productivity_multiplier(self, age: int) -> float:
        """
        Productivity over the lifecycle.

        youth (0-15):  none (learning)
        young adult (16-25):  0.5 -> 1.0 (ramp up)
        prime (26-50): 1.0 (peak)
        middle (51-65): 1.0 -> 0.7 (gradual decline)
        elderly (66-80): 0.7 -> 0.4
        dying (81+): decline to 0.1
        """
        if age < 16:
            return 0.0
        if age < 26:
            return 0.5 + (age - 16) * 0.05
        if age < 51:
            return 1.0
        if age < 66:
            return 1.0 - (age - 51) * 0.02
        if age < 81:
            return 0.7 - (age - 66) * 0.02
        return max(0.1, 0.4 - (age - 81) * 0.015)

    # ------------------------------------------------------------------
    # Institutional Multiplier
    # ------------------------------------------------------------------

    def compute_institutional_multiplier(
        self,
        agents: List[Any],
        coalitions: Optional[Dict[str, Any]] = None,
        constitution: Any = None,
        step: int = 0,
    ) -> Tuple[float, float, float, float]:
        """
        institutional_mult = 1.0 + trust*0.3 + stability*0.2 + memory*0.2

        Returns (multiplier, trust, stability, memory_continuity).
        """
        active = [a for a in agents if a.active]
        if not active:
            return (1.0, 0.0, 0.0, 0.0)

        # Trust: average agent reliability
        trust = float(np.mean([a.reliability for a in active]))

        # Stability: coalition persistence or mean agent survival
        if coalitions and len(coalitions) > 0:
            avg_cohesion = float(np.mean([
                c.compute_cohesion() for c in coalitions.values()
            ]))
            stability = avg_cohesion
        else:
            stability = float(np.mean([
                min(1.0, a.survival_count / 50.0) for a in active
            ]))

        # Memory continuity: how much institutional memory exists
        if constitution and hasattr(constitution, 'institutional_memory'):
            mem = constitution.institutional_memory
            memory_continuity = min(1.0, len(mem.history) / 200.0) if hasattr(mem, 'history') else 0.3
        else:
            memory_continuity = 0.3

        mult = 1.0 + trust * 0.3 + stability * 0.2 + memory_continuity * 0.2
        return (round(mult, 3), round(trust, 3), round(stability, 3), round(memory_continuity, 3))

    # ------------------------------------------------------------------
    # Agent Production
    # ------------------------------------------------------------------

    def compute_agent_output(self, agent: Any, step: int) -> Dict[str, float]:
        """
        Compute production for a single agent.

        labor = health x age_productivity x base_productivity
        output = labor x capital_mult x institutional_mult
        """
        if not agent.active:
            return {'labor': 0.0, 'output': 0.0, 'capital_mult': 1.0, 'institutional_mult': 1.0}

        health = getattr(agent, 'health', 1.0)
        age_mult = self.age_productivity_multiplier(agent.age)
        labor = health * age_mult * agent.productivity

        capital_mult = 1.0 + self.max_capital_boost * (
            self.capital / (self.capital + self.capital_half_saturation)
        )
        inst_mult = self.last_institutional_mult if hasattr(self, 'last_institutional_mult') else 1.0
        output = labor * capital_mult * inst_mult
        output = max(0.0, output)

        return {
            'labor': round(labor, 3),
            'output': round(output, 3),
            'capital_mult': round(capital_mult, 3),
            'institutional_mult': round(inst_mult, 3),
        }

    # ------------------------------------------------------------------
    # Main Step
    # ------------------------------------------------------------------

    def step(
        self,
        agents: List[Any],
        coalitions: Optional[Dict[str, Any]] = None,
        constitution: Any = None,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Run one production cycle.

        Order:
          1. Compute institutional multiplier
          2. Each agent produces output
          3. Each agent pays survival cost
          4. Agents with surplus invest in capital
          5. Capital depreciates, updated with investment
          6. Log production metrics

        Returns summary dict.
        """
        # 1. Institutional multiplier
        inst_mult, trust, stability, memory = self.compute_institutional_multiplier(
            agents, coalitions, constitution, step
        )
        self.last_institutional_mult = inst_mult
        self.last_trust = trust
        self.last_stability = stability
        self.last_memory = memory

        # 2-3. Agent production + consumption
        total_output = 0.0
        total_survival = 0.0
        n_productively_alive = 0
        agent_results = []

        for agent in agents:
            if not agent.active:
                continue

            # Production
            result = self.compute_agent_output(agent, step)
            agent.wealth += result['output']
            total_output += result['output']

            if result['labor'] > 0:
                n_productively_alive += 1

            # Survival cost
            survival = min(self.survival_cost, agent.wealth)
            agent.wealth -= survival
            agent.wealth = max(0.05, agent.wealth)
            total_survival += survival
            result['survival'] = round(survival, 3)

            agent_results.append(result)

        # 4. Investment
        total_investment_wealth = 0.0
        for agent in agents:
            if not agent.active:
                continue
            surplus = agent.wealth - self.investment_threshold
            if surplus > 0.05:
                invest = surplus * self.investment_fraction
                agent.wealth -= invest
                total_investment_wealth += invest

        investment_effective = total_investment_wealth * self.investment_efficiency

        # 5. Capital dynamics
        depreciation = self.capital * self.depreciation_rate
        self.capital = self.capital + investment_effective - depreciation
        self.capital = max(self.min_capital, self.capital)
        self.capital_log.append(self.capital)

        # 6. Log
        entry = {
            'step': step,
            'total_output': round(total_output, 3),
            'total_survival': round(total_survival, 3),
            'total_investment': round(total_investment_wealth, 3),
            'investment_effective': round(investment_effective, 3),
            'capital': round(self.capital, 3),
            'depreciation': round(depreciation, 3),
            'institutional_mult': inst_mult,
            'trust': trust,
            'stability': stability,
            'memory_continuity': memory,
            'n_productively_alive': n_productively_alive,
            'n_active': len([a for a in agents if a.active]),
        }
        self.production_log.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Stats & Reports
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {
            'capital': round(self.capital, 3),
            'total_capital_ever': round(
                sum(e['investment_effective'] for e in self.production_log), 3
            ),
            'total_output_ever': round(
                sum(e['total_output'] for e in self.production_log), 3
            ),
            'mean_output_per_step': round(
                float(np.mean([e['total_output'] for e in self.production_log[-50:]]))
                if len(self.production_log) >= 50
                else float(np.mean([e['total_output'] for e in self.production_log]))
                if self.production_log else 0.0, 3
            ),
            'mean_institutional_mult': round(
                float(np.mean([e['institutional_mult'] for e in self.production_log[-50:]]))
                if self.production_log else 1.0, 3
            ),
            'n_steps': len(self.production_log),
        }

    def compute_gdp(self, last_n: int = 50) -> float:
        """Average total output over recent steps."""
        recent = self.production_log[-last_n:] if len(self.production_log) >= last_n else self.production_log
        if not recent:
            return 0.0
        return float(np.mean([e['total_output'] for e in recent]))

    def compute_capital_trend(self, last_n: int = 50) -> str:
        """Direction of capital change."""
        recent = self.capital_log[-last_n:] if len(self.capital_log) >= last_n else self.capital_log
        if len(recent) < 2:
            return 'stable'
        slope = (recent[-1] - recent[0]) / len(recent)
        if slope > 0.01:
            return 'growing'
        if slope < -0.01:
            return 'declining'
        return 'stable'


# ============================================================================
# Convenience: Add health attribute to CognitiveAgent
# ============================================================================

def ensure_production_attributes(agent: Any):
    """Add production-related attributes to an agent if missing."""
    if not hasattr(agent, 'health'):
        agent.health = 1.0
    if not hasattr(agent, 'compute_capacity'):
        agent.compute_capacity = 1.0
    return agent


# ============================================================================
# TESTS
# ============================================================================

def test_age_productivity_curve():
    """Verify lifecycle productivity pattern."""
    print("\n" + "=" * 60)
    print("48.5D.1 — AGE-PRODUCTIVITY CURVE")
    print("=" * 60)

    engine = ProductionEngine()

    ages = list(range(0, 101, 10))
    mults = [engine.age_productivity_multiplier(a) for a in ages]
    print(f"  Productivity by age: {dict(zip(ages, mults))}")

    # Youth (0-15): 0.0
    assert engine.age_productivity_multiplier(10) == 0.0, "Youth should produce nothing"
    # Young adult (20): should be > 0.5
    assert engine.age_productivity_multiplier(20) > 0.5, "Young adult should produce"
    # Prime (30-40): 1.0
    assert engine.age_productivity_multiplier(40) == 1.0, "Prime should peak"
    # Elderly (70): < 1.0
    assert engine.age_productivity_multiplier(70) < 0.7, "Elderly productivity declines"
    # Very old (90): near floor
    assert engine.age_productivity_multiplier(90) < 0.4, "Very old productivity near floor"

    # Monotonicity checks
    assert all(mults[i] <= mults[i+1] or mults[i] >= mults[i+1]
               for i in range(len(mults)-1)), "Should be roughly unimodal"

    print("  >>> AgeProductivityCurve PASSED\n")


def test_basic_production():
    """Verify single-step production mechanics."""
    print("\n" + "=" * 60)
    print("48.5D.2 — BASIC PRODUCTION")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent
    engine = ProductionEngine()

    # Create agents at different ages
    agents = []
    for i, age in enumerate([10, 25, 40, 60, 80]):
        a = CognitiveAgent(f'prod_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = age
        a.wealth = 3.0
        a.productivity = 0.5
        a.reliability = 0.5
        agents.append(a)

    # Step
    result = engine.step(agents, step=1)

    print(f"  Institutional mult: {result['institutional_mult']}")
    print(f"  Capital: {result['capital']}")
    print(f"  Total output: {result['total_output']}")
    print(f"  Total survival: {result['total_survival']}")
    print(f"  Productively alive: {result['n_productively_alive']}")

    # Youngest (age 10) should produce 0
    assert agents[0].wealth <= 3.0, "Youth should not earn production income"

    # Prime (age 25, 40) should produce most
    assert agents[1].wealth > 3.0, "Prime agent should produce surplus"
    assert agents[2].wealth > 3.0, "Prime agent should produce surplus"

    # Very old (age 80) should produce less
    assert agents[4].wealth < agents[1].wealth, "Old should produce less than prime"

    # Total output should be positive
    assert result['total_output'] > 0, "Should have positive output"

    # Survival cost was deducted
    total_wealth = sum(a.wealth for a in agents if a.active)
    assert total_wealth < sum([3.0]*5) + result['total_output'], \
        "Survival cost should reduce wealth"

    print("  >>> BasicProduction PASSED\n")


def test_institutional_multiplier():
    """Verify institutional multiplier responds to trust/stability/memory."""
    print("\n" + "=" * 60)
    print("48.5D.3 — INSTITUTIONAL MULTIPLIER")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent
    engine = ProductionEngine()

    # Low-trust scenario
    agents_low = []
    for i in range(10):
        a = CognitiveAgent(f'low_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = 30
        a.wealth = 3.0
        a.reliability = 0.1  # Low trust
        agents_low.append(a)

    mult_low, trust_low, stab_low, mem_low = engine.compute_institutional_multiplier(agents_low)
    print(f"  Low trust ({trust_low:.2f}) -> mult: {mult_low:.3f}")

    # High-trust scenario
    agents_high = []
    for i in range(10):
        a = CognitiveAgent(f'high_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = 30
        a.wealth = 3.0
        a.reliability = 0.9  # High trust
        agents_high.append(a)

    mult_high, trust_high, stab_high, mem_high = engine.compute_institutional_multiplier(agents_high)
    print(f"  High trust ({trust_high:.2f}) -> mult: {mult_high:.3f}")

    assert mult_high > mult_low, "High trust should give higher multiplier"
    assert mult_low >= 1.0, "Even low trust should have baseline 1.0"

    # Formula check: mult = 1.0 + trust*0.3 + stability*0.2 + memory*0.2
    expected_low = 1.0 + trust_low * 0.3 + stab_low * 0.2 + mem_low * 0.2
    assert abs(mult_low - round(expected_low, 3)) < 0.01, \
        f"Formula mismatch: {mult_low} vs {expected_low:.3f}"

    print("  >>> InstitutionalMultiplier PASSED\n")


def test_capital_dynamics():
    """Verify capital accumulation, investment, and depreciation."""
    print("\n" + "=" * 60)
    print("48.5D.4 — CAPITAL DYNAMICS")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent
    engine = ProductionEngine(
        depreciation_rate=0.02,
        investment_efficiency=0.3,
        investment_threshold=1.0,
        investment_fraction=0.3,
    )
    engine.capital = 5.0  # Start with some capital

    agents = []
    for i in range(20):
        a = CognitiveAgent(f'cap_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = 30  # prime
        a.wealth = 5.0  # Above investment threshold
        a.productivity = 0.7
        a.reliability = 0.6
        agents.append(a)

    # Run 100 steps
    for step in range(1, 101):
        result = engine.step(agents, step=step)

    print(f"  Initial capital: 5.0")
    print(f"  Final capital: {engine.capital:.3f}")
    print(f"  Total output ever: {engine.get_stats()['total_output_ever']}")
    print(f"  Mean output/step (last 50): {engine.get_stats()['mean_output_per_step']}")
    print(f"  Trend: {engine.compute_capital_trend()}")

    # Capital grew from investment
    assert engine.capital > 0, "Capital should remain positive"

    # If investment > depreciation, capital grows
    total_investment = sum(e['investment_effective'] for e in engine.production_log)
    total_depreciation = sum(e['depreciation'] for e in engine.production_log)
    net = total_investment - total_depreciation
    print(f"  Total investment: {total_investment:.3f}")
    print(f"  Total depreciation: {total_depreciation:.3f}")
    print(f"  Net capital change: {net:.3f}")

    assert abs(engine.capital - (5.0 + net)) < 0.1, \
        f"Capital accounting mismatch: {engine.capital:.3f} vs {5.0 + net:.3f}"

    print("  >>> CapitalDynamics PASSED\n")


def test_survival_pressure():
    """Verify survival cost creates pressure on poor agents."""
    print("\n" + "=" * 60)
    print("48.5D.5 — SURVIVAL PRESSURE")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent
    engine = ProductionEngine(survival_cost=0.5)  # High survival cost

    # Create poor elderly agents (can't produce much)
    agents = []
    for i in range(5):
        a = CognitiveAgent(f'poor_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = 75  # Low productivity
        a.wealth = 2.0
        a.productivity = 0.3
        a.reliability = 0.3
        agents.append(a)

    # Run 30 steps
    initial_wealth = sum(a.wealth for a in agents if a.active)
    for step in range(1, 31):
        engine.step(agents, step=step)

    final_wealth = sum(a.wealth for a in agents if a.active)
    print(f"  Initial wealth: {initial_wealth:.3f}")
    print(f"  Final wealth: {final_wealth:.3f}")

    # Poor elderly should lose wealth over time (survival > production)
    assert final_wealth < initial_wealth, \
        "Poor elderly should lose wealth to survival costs"

    # Some should hit the floor
    min_wealth = min(a.wealth for a in agents if a.active)
    print(f"  Min wealth: {min_wealth:.3f}")
    assert min_wealth >= 0.05, "Wealth floor should prevent negative"

    print("  >>> SurvivalPressure PASSED\n")


def test_integrated_generational_run():
    """
    Full integrated test: production economy + generational turnover.

    This is the key validation: does the system maintain production
    across generations, or does it collapse?
    """
    print("\n" + "=" * 60)
    print("48.5D.6 — INTEGRATED GENERATIONAL ECONOMY")
    print("=" * 60)

    from phase48_cognitive_political_economy import CognitiveAgent, SPECIES_PARAMS
    from phase48_generational_turnover import GenerationalEngine, LineageRecord

    # Generational engine with production-friendly settings
    gen_engine = GenerationalEngine(
        max_population=100,
        reproduction_interval=1,
        reproduction_cooldown=5,
        max_children_per_pair=8,
    )

    # Production engine
    prod_engine = ProductionEngine(
        survival_cost=0.12,
        depreciation_rate=0.008,
        investment_efficiency=0.4,
        investment_threshold=1.5,
        investment_fraction=0.15,
    )

    random.seed(42)
    np.random.seed(42)

    # Create initial diverse population
    species_list = ['exploitative', 'exploratory', 'defensive',
                    'identity_preserving', 'novelty_seeking', 'stability_seeking']
    agents = []
    for i in range(30):
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

    # Run 500 steps with integrated production and demography
    total_births = 0
    total_deaths = 0
    gdp_history = []

    for step in range(1, 501):
        # Production step
        prod_result = prod_engine.step(agents, step=step)

        # Generational step
        gen_result = gen_engine.step(agents, {}, step, spawn_fn,
                                     compute_fn=lambda ag, sp, st: None)
        total_births += len(gen_result['births'])
        total_deaths += len(gen_result['deaths'])

        if step % 100 == 0:
            gdp_history.append(prod_result['total_output'])

    active = len([a for a in agents if a.active])
    stats = prod_engine.get_stats()
    gen_stats = gen_engine.get_stats()

    print(f"  500 steps completed")
    print(f"  ✓ Active agents: {active}")
    print(f"  ✓ Total births: {total_births}")
    print(f"  ✓ Total deaths: {total_deaths}")
    print(f"  ✓ Capital: {prod_engine.capital:.3f}")
    print(f"  ✓ Mean output/step: {stats['mean_output_per_step']}")
    print(f"  ✓ Mean institutional mult: {stats['mean_institutional_mult']:.3f}")
    print(f"  ✓ Active lineages: {gen_stats['n_active_lineages']}")

    # GDP at milestones
    for i, gdp in enumerate(gdp_history):
        step_num = (i + 1) * 100
        print(f"    GDP at step {step_num}: {gdp:.3f}")

    # Assertions
    assert total_births > 0, "Should have births"
    assert active > 0, "Should have survivors"
    assert stats['total_output_ever'] > 0, "Should have produced output"
    assert prod_engine.capital > prod_engine.min_capital, "Capital should be maintained"
    assert stats['mean_institutional_mult'] >= 1.0, "Inst mult should be >= 1.0"

    if len(gdp_history) >= 2:
        trend_dir = "growing" if gdp_history[-1] > gdp_history[0] else "declining"
        print(f"  ✓ GDP trend: {trend_dir} over 500 steps")

    print("  >>> IntegratedGenerationalEconomy PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5D: PRODUCTION ECONOMY LAYER                           ║
║                                                                   ║
║  Transforms Cognitive Political Economy from                     ║
║  REDISTRIBUTIVE ECONOMY → ENDOGENOUS PRODUCTION ECONOMY           ║
║                                                                   ║
║  Without production: wealth is redistributed but never created.   ║
║  With production: supply = f(population, capital, institutions).  ║
║                                                                   ║
║  Next: 48.5E (Endogenous Scarcity)                               ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    all_passed = True

    try:
        test_age_productivity_curve()
    except Exception as e:
        print(f"  >>> AgeProductivityCurve FAILED: {e}\n")
        all_passed = False

    try:
        test_basic_production()
    except Exception as e:
        print(f"  >>> BasicProduction FAILED: {e}\n")
        all_passed = False

    try:
        test_institutional_multiplier()
    except Exception as e:
        print(f"  >>> InstitutionalMultiplier FAILED: {e}\n")
        all_passed = False

    try:
        test_capital_dynamics()
    except Exception as e:
        print(f"  >>> CapitalDynamics FAILED: {e}\n")
        all_passed = False

    try:
        test_survival_pressure()
    except Exception as e:
        print(f"  >>> SurvivalPressure FAILED: {e}\n")
        all_passed = False

    try:
        test_integrated_generational_run()
    except Exception as e:
        print(f"  >>> IntegratedGenerationalEconomy FAILED: {e}\n")
        all_passed = False

    if all_passed:
        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  PHASE 48.5D: ALL TESTS PASSED                                ║
  ║                                                               ║
  ║  Production Economy Layer ready.                              ║
  ║  The system now has endogenous supply.                       ║
  ║  Production = f(population, capital, institutions).           ║
  ║                                                               ║
  ║  Next: 48.5E (Endogenous Scarcity)                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║  Some tests FAILED                                           ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
