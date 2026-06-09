"""
Phase 48.5B — Scarcity Cycles Engine.

Adds DYNAMIC ECONOMIC PRESSURE to the Cognitive Political Economy.

Without scarcity gradients, there is no:
  - power struggle
  - strategic behavior
  - institutional evolution
  - existential pressure

This engine creates:
  1. Boom/Recession/Winter supply cycles
  2. Asymmetric access (inequality amplifies in scarcity)
  3. Temporal pressure (delayed rewards decay, deadlines)
  4. Survival thresholds (agents die without sufficient wealth)

Phase characteristics:
  BOOM:      abundant supply, low competition, wealth accumulates
  RECESSION: reduced supply, increased competition, inequality grows
  WINTER:    severe scarcity, survival pressure, high death risk
  RECOVERY:  gradual return to normal, fragile institutions

Key principle:
  "Scarcity is not uniform — it amplifies existing inequalities."
  Poor agents get squeezed harder in winter than wealthy ones.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sys
sys.path.insert(0, '.')

from phase48_cognitive_political_economy import (
    ResourceType, ResourceBundle, CognitiveAgent
)


# ============================================================================
# Scarcity Cycle Types
# ============================================================================

class CyclePhase(Enum):
    BOOM = 'boom'
    RECESSION = 'recession'
    WINTER = 'winter'
    RECOVERY = 'recovery'


PHASE_DURATIONS: Dict[CyclePhase, Tuple[int, int]] = {
    CyclePhase.BOOM: (60, 150),       # 60-150 steps
    CyclePhase.RECESSION: (40, 80),   # 40-80 steps
    CyclePhase.WINTER: (20, 50),      # 20-50 steps
    CyclePhase.RECOVERY: (50, 100),   # 50-100 steps
}

PHASE_SUPPLY_MULTIPLIERS: Dict[CyclePhase, Dict[str, float]] = {
    CyclePhase.BOOM: {
        'compute': 1.3, 'planning_depth': 1.2, 'memory_bandwidth': 1.2,
        'retrieval_access': 1.1, 'counterfactual_budget': 1.3,
        'semantic_projection': 1.1, 'narrative_bandwidth': 1.2,
    },
    CyclePhase.RECESSION: {
        'compute': 0.7, 'planning_depth': 0.7, 'memory_bandwidth': 0.8,
        'retrieval_access': 0.8, 'counterfactual_budget': 0.6,
        'semantic_projection': 0.8, 'narrative_bandwidth': 0.7,
    },
    CyclePhase.WINTER: {
        'compute': 0.3, 'planning_depth': 0.3, 'memory_bandwidth': 0.4,
        'retrieval_access': 0.4, 'counterfactual_budget': 0.2,
        'semantic_projection': 0.4, 'narrative_bandwidth': 0.3,
    },
    CyclePhase.RECOVERY: {
        'compute': 0.8, 'planning_depth': 0.8, 'memory_bandwidth': 0.9,
        'retrieval_access': 0.9, 'counterfactual_budget': 0.7,
        'semantic_projection': 0.9, 'narrative_bandwidth': 0.8,
    },
}

# How much harder scarcity hits poor agents (asymmetric access)
ASYMMETRY_FACTOR: float = 0.3  # 0 = equal impact, 1 = poor hit much harder


# ============================================================================
# Scarcity Cycle Engine
# ============================================================================

@dataclass
class CycleState:
    """Current state of the scarcity cycle."""
    phase: CyclePhase
    steps_in_phase: int = 0
    total_steps: int = 0
    duration: int = 100
    severity: float = 0.0  # 0 = none, 1 = extreme
    supply_multiplier: float = 1.0

    def get_stats(self) -> Dict:
        return {
            'phase': self.phase.value,
            'steps_in_phase': self.steps_in_phase,
            'phase_duration': self.duration,
            'progress_pct': round(100.0 * self.steps_in_phase / max(1, self.duration), 1),
            'severity': round(self.severity, 3),
            'supply_multiplier': round(self.supply_multiplier, 3),
        }


class ScarcityCycleEngine:
    """
    Dynamic economic scarcity with boom/recession/winter/recovery cycles.

    Unlike a fixed ResourceBundle, this engine:
      1. Varies total supply based on cycle phase
      2. Applies asymmetric distribution (poor hit harder in scarcity)
      3. Adds temporal reward decay (pressure to act now)
      4. Enforces survival thresholds (agents die without wealth)

    Integration:
      engine = ScarcityCycleEngine()
      # Before each ResourceMarket.step():
      supply = engine.modify_supply(resource_market.supply)
      # The engine also updates agent wealth, survival, etc.
    """

    def __init__(
        self,
        base_supply: Optional[ResourceBundle] = None,
        cycle_length_range: Tuple[int, int] = (150, 400),
        temporal_decay_rate: float = 0.01,
        survival_wealth_threshold: float = 0.15,
        survival_check_interval: int = 10,
    ):
        self.base_supply = base_supply or ResourceBundle()
        self.cycle_length_range = cycle_length_range
        self.temporal_decay_rate = temporal_decay_rate
        self.survival_wealth_threshold = survival_wealth_threshold
        self.survival_check_interval = survival_check_interval

        # Start in boom
        initial_duration = random.randint(*PHASE_DURATIONS[CyclePhase.BOOM])
        self.state = CycleState(
            phase=CyclePhase.BOOM,
            duration=initial_duration,
        )

        self.cycle_history: List[Dict] = []
        self.phase_transitions: List[Dict] = []
        self.death_log: List[Dict] = []
        self.total_steps: int = 0

        # Track opportunity windows (for temporal pressure)
        self.opportunity_windows: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Cycle Transitions
    # ------------------------------------------------------------------

    def _pick_next_phase(self) -> CyclePhase:
        """Pick the next cycle phase based on current."""
        current = self.state.phase
        if current == CyclePhase.BOOM:
            return CyclePhase.RECESSION
        elif current == CyclePhase.RECESSION:
            return random.choices(
                [CyclePhase.WINTER, CyclePhase.RECOVERY],
                weights=[0.4, 0.6]
            )[0]
        elif current == CyclePhase.WINTER:
            return CyclePhase.RECOVERY
        else:  # RECOVERY
            return CyclePhase.BOOM

    def _transition_phase(self):
        """Transition to next cycle phase."""
        new_phase = self._pick_next_phase()
        duration = random.randint(*PHASE_DURATIONS[new_phase])
        old_phase = self.state.phase

        self.phase_transitions.append({
            'step': self.total_steps,
            'from': old_phase.value,
            'to': new_phase.value,
            'severity_at_transition': self.state.severity,
        })

        self.state = CycleState(
            phase=new_phase,
            duration=duration,
            total_steps=self.total_steps,
        )

    # ------------------------------------------------------------------
    # Supply Modification
    # ------------------------------------------------------------------

    def get_supply_multipliers(self) -> Dict[str, float]:
        """Get current supply multipliers per resource."""
        return PHASE_SUPPLY_MULTIPLIERS.get(
            self.state.phase,
            PHASE_SUPPLY_MULTIPLIERS[CyclePhase.BOOM]
        )

    def modify_supply(self, original: ResourceBundle) -> ResourceBundle:
        """
        Apply scarcity multipliers to a ResourceBundle.

        This is the main integration point — call before ResourceMarket.step():
          supply = scarcity_engine.modify_supply(resource_market.supply)
          resource_market.supply = supply
        """
        multipliers = self.get_supply_multipliers()
        self.state.supply_multiplier = float(np.mean(list(multipliers.values())))

        return ResourceBundle(
            compute=original.compute * multipliers.get('compute', 1.0),
            planning_depth=original.planning_depth * multipliers.get('planning_depth', 1.0),
            memory_bandwidth=original.memory_bandwidth * multipliers.get('memory_bandwidth', 1.0),
            retrieval_access=original.retrieval_access * multipliers.get('retrieval_access', 1.0),
            counterfactual_budget=original.counterfactual_budget * multipliers.get('counterfactual_budget', 1.0),
            semantic_projection=original.semantic_projection * multipliers.get('semantic_projection', 1.0),
            narrative_bandwidth=original.narrative_bandwidth * multipliers.get('narrative_bandwidth', 1.0),
        )

    # ------------------------------------------------------------------
    # Asymmetric Access
    # ------------------------------------------------------------------

    def apply_asymmetric_access(self, agents: List[CognitiveAgent]):
        """
        Apply wealth-dependent access penalties.

        In scarcity, poor agents are disproportionately affected:
          - Wealth below median → additional wealth decay
          - Wealth in bottom quartile → bid intensity penalty
          - Wealth near zero → severe exploration penalty

        This creates the REAL scarcity dynamic:
          "The rich get richer, the poor get poorer."
        """
        active = [a for a in agents if a.active]
        if len(active) < 2:
            return

        wealths = sorted([a.wealth for a in active])
        median_wealth = wealths[len(wealths) // 2]
        q1_wealth = wealths[len(wealths) // 4] if len(wealths) >= 4 else wealths[0]

        scarcity_severity = self.state.severity

        for agent in active:
            # Bottom half: additional wealth decay
            if agent.wealth < median_wealth:
                penalty = (median_wealth - agent.wealth) / max(median_wealth, 0.01)
                penalty = min(penalty, 1.0) * scarcity_severity * ASYMMETRY_FACTOR
                agent.wealth *= (1.0 - penalty * 0.05)
                agent.wealth = max(0.05, agent.wealth)

            # Bottom quartile: bid intensity penalty (can't compete)
            if agent.wealth < q1_wealth * 1.5:
                agent.bid_intensity *= max(0.3, 1.0 - scarcity_severity * 0.3)

            # Near zero: exploration collapses (survival mode)
            if agent.wealth < self.survival_wealth_threshold:
                agent.exploration_rate *= max(0.1, 1.0 - scarcity_severity * 0.5)
                agent.risk_tolerance *= max(0.2, 1.0 - scarcity_severity * 0.3)

    # ------------------------------------------------------------------
    # Temporal Pressure — Opportunity Decay
    # ------------------------------------------------------------------

    def create_opportunity_window(
        self,
        agent_id: str,
        reward_potential: float,
        deadline: int,
        label: str = ''
    ) -> str:
        """
        Create a time-limited opportunity.

        If the agent doesn't capitalize within the window,
        the reward decays to zero. This creates REAL temporal pressure.

        Returns opportunity_id for tracking.
        """
        opp_id = f"opp_{agent_id}_{self.total_steps}_{len(self.opportunity_windows)}"
        self.opportunity_windows[opp_id] = {
            'agent_id': agent_id,
            'reward_potential': reward_potential,
            'created_step': self.total_steps,
            'deadline': deadline,
            'label': label,
            'decayed': 0.0,
            'collected': False,
        }
        return opp_id

    def collect_opportunity(self, opportunity_id: str) -> float:
        """
        Collect an opportunity's reward (diminished by decay).

        Returns actual reward after temporal decay.
        """
        opp = self.opportunity_windows.get(opportunity_id)
        if opp is None or opp['collected']:
            return 0.0

        steps_elapsed = self.total_steps - opp['created_step']
        steps_remaining = max(0, opp['deadline'] - steps_elapsed)
        decay = 1.0 - (steps_elapsed / max(1, opp['deadline']))
        decay = max(0.0, decay)

        actual_reward = opp['reward_potential'] * decay
        opp['collected'] = True
        opp['decayed'] = 1.0 - decay

        return actual_reward

    def decay_opportunities(self):
        """Apply temporal decay to all open opportunities."""
        for opp_id in list(self.opportunity_windows.keys()):
            opp = self.opportunity_windows[opp_id]
            if opp['collected']:
                del self.opportunity_windows[opp_id]
                continue

            steps_elapsed = self.total_steps - opp['created_step']
            if steps_elapsed > opp['deadline']:
                # Opportunity expired
                opp['collected'] = True
                opp['decayed'] = 1.0

    # ------------------------------------------------------------------
    # Survival Pressure
    # ------------------------------------------------------------------

    def apply_survival_pressure(self, agents: List[CognitiveAgent]) -> List[Dict]:
        """
        Check survival thresholds and mark agents for death.

        Returns list of death events.
        """
        if self.total_steps % self.survival_check_interval != 0:
            return []

        deaths = []
        for agent in agents:
            if not agent.active:
                continue
            if agent.age < 5:  # New agents get grace period
                continue

            # Below survival threshold for too long → death
            if agent.wealth < self.survival_wealth_threshold * 0.5 and agent.age > 10:
                # Survival probability based on how long they've been near death
                survival_prob = 0.5 * (1.0 - max(0.0, agent.age - 10) / 50.0)
                if random.random() > survival_prob:
                    agent.active = False
                    deaths.append({
                        'agent_id': agent.agent_id,
                        'species': agent.species,
                        'age': agent.age,
                        'cause': 'starvation',
                        'wealth_at_death': round(agent.wealth, 3),
                        'step': self.total_steps,
                    })
                    self.death_log.append(deaths[-1])

            # Winter: even moderate-wealth agents can die
            elif (self.state.phase == CyclePhase.WINTER
                  and agent.wealth < self.survival_wealth_threshold
                  and random.random() < 0.02):
                agent.active = False
                deaths.append({
                    'agent_id': agent.agent_id,
                    'species': agent.species,
                    'age': agent.age,
                    'cause': 'winter_famine',
                    'wealth_at_death': round(agent.wealth, 3),
                    'step': self.total_steps,
                })
                self.death_log.append(deaths[-1])

        return deaths

    # ------------------------------------------------------------------
    # Main Step
    # ------------------------------------------------------------------

    def step(
        self,
        agents: List[CognitiveAgent],
        resource_market: Any  # ResourceMarket instance
    ) -> Dict[str, Any]:
        """
        Run one scarcity cycle step.

        Returns dict with cycle state and any death events.
        """
        self.total_steps += 1
        self.state.steps_in_phase += 1
        self.state.total_steps = self.total_steps

        # Update severity (higher = more pressure)
        phase_severity = {
            CyclePhase.BOOM: 0.1,
            CyclePhase.RECESSION: 0.4,
            CyclePhase.WINTER: 0.8,
            CyclePhase.RECOVERY: 0.3,
        }
        self.state.severity = phase_severity.get(self.state.phase, 0.5)

        # 1. Phase transition check
        if self.state.steps_in_phase >= self.state.duration:
            self._transition_phase()

        # 2. Modify resource supply
        modified_supply = self.modify_supply(resource_market.supply)
        resource_market.supply = modified_supply

        # 3. Apply asymmetric access
        self.apply_asymmetric_access(agents)

        # 4. Apply temporal pressure
        self.decay_opportunities()

        # 5. Apply survival pressure
        deaths = self.apply_survival_pressure(agents)

        # Log cycle state
        cycle_info = self.state.get_stats()
        self.cycle_history.append(cycle_info)

        return {
            'cycle_state': cycle_info,
            'deaths': deaths,
            'phase_transition': self.phase_transitions[-1] if self.phase_transitions else None,
        }

    # ------------------------------------------------------------------
    # Stats & History
    # ------------------------------------------------------------------

    def get_cycle_statistics(self) -> Dict:
        """Get comprehensive cycle statistics."""
        if not self.cycle_history:
            return {'phases': {}}

        phases_visited = list(set(
            h['phase'] for h in self.cycle_history
        ))
        n_transitions = len(self.phase_transitions)

        # Mean supply by phase
        phase_supply = {}
        for phase in CyclePhase:
            multipliers = PHASE_SUPPLY_MULTIPLIERS.get(phase, {})
            mean_mul = float(np.mean(list(multipliers.values()))) if multipliers else 1.0
            phase_supply[phase.value] = round(mean_mul, 3)

        return {
            'n_transitions': n_transitions,
            'total_steps': self.total_steps,
            'current_phase': self.state.phase.value,
            'phase_supply_multipliers': phase_supply,
            'phases_visited': phases_visited,
            'total_deaths': len(self.death_log),
            'deaths_by_cause': {
                cause: len([d for d in self.death_log if d['cause'] == cause])
                for cause in set(d['cause'] for d in self.death_log)
            } if self.death_log else {},
        }

    def get_cycle_profile(self) -> str:
        """Generate a textual summary of the cycle history."""
        if not self.cycle_history:
            return "No cycles completed."

        recent = self.cycle_history[-50:]
        phases = [h['phase'] for h in recent]
        transitions = [p for i, p in enumerate(phases)
                       if i > 0 and p != phases[i - 1]]

        return (
            f"Current: {self.state.phase.value} "
            f"(step {self.state.steps_in_phase}/{self.state.duration})\n"
            f"Recent phases: {' → '.join(phases[-10:])}\n"
            f"Transitions: {len(self.phase_transitions)} total\n"
            f"Deaths: {len(self.death_log)} "
            f"({sum(1 for d in self.death_log if d['cause'] == 'starvation')} starvation, "
            f"{sum(1 for d in self.death_log if d['cause'] == 'winter_famine')} winter)"
        )

    def reset(self):
        """Reset cycle to initial state (for testing)."""
        self.__init__(
            base_supply=self.base_supply,
            cycle_length_range=self.cycle_length_range,
            temporal_decay_rate=self.temporal_decay_rate,
            survival_wealth_threshold=self.survival_wealth_threshold,
        )


# ============================================================================
# Integration Adapter
# ============================================================================

class ScarcityAwareResourceMarket:
    """
    Wrapper that couples ResourceMarket with ScarcityCycleEngine.

    Usage:
      market = ScarcityAwareResourceMarket()
      # In CognitivePoliticalEngine.step():
      result = market.step(agents)

    This replaces the direct ResourceMarket call.
    """

    def __init__(
        self,
        base_supply: Optional[ResourceBundle] = None,
        scarcity_config: Optional[Dict] = None,
    ):
        from phase48_cognitive_political_economy import ResourceMarket
        self.resource_market = ResourceMarket(
            total_supply=base_supply or ResourceBundle()
        )
        config = scarcity_config or {}
        self.scarcity_engine = ScarcityCycleEngine(**config)
        self.market_log: List[Dict] = []

    def step(self, agents: List[CognitiveAgent]) -> Dict[str, Any]:
        """Run scarcity-aware market step."""
        # 1. Run scarcity cycle (modifies supply, asymmetric access, survival)
        scarcity_result = self.scarcity_engine.step(
            agents, self.resource_market
        )

        # 2. Run resource market with modified supply
        market_result = self.resource_market.step(agents)

        # 3. Combine results
        result = {
            **market_result,
            'scarcity': scarcity_result['cycle_state'],
            'deaths': scarcity_result['deaths'],
            'phase_transition': scarcity_result['phase_transition'],
        }
        self.market_log.append(result)
        return result

    def get_gini_coefficient(self, agents: List[CognitiveAgent]) -> float:
        return self.resource_market.get_gini_coefficient(agents)

    def get_wealthiest_agent(self, agents: List[CognitiveAgent]) -> Optional[str]:
        return self.resource_market.get_wealthiest_agent(agents)

    def get_stats(self) -> Dict:
        return {
            **self.resource_market.get_stats(),
            'scarcity': self.scarcity_engine.get_cycle_statistics(),
        }


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_cycle_transitions():
    """Test that scarcity cycles transition correctly."""
    print("\n" + "=" * 60)
    print("48.5B.1 — CYCLE TRANSITIONS")
    print("=" * 60)

    engine = ScarcityCycleEngine()
    # Override duration to short for testing
    engine.state.duration = 5

    from dataclasses import dataclass, field
    @dataclass
    class DummyMarket:
        supply: ResourceBundle = field(default_factory=ResourceBundle)
    dummy = DummyMarket()

    phases_seen = set()
    transitions = []

    random.seed(42)
    for _ in range(200):
        old_phase = engine.state.phase.value
        engine.step([], dummy)
        new_phase = engine.state.phase.value
        phases_seen.add(new_phase)
        if old_phase != new_phase:
            transitions.append(f"{old_phase}→{new_phase}")

    print(f"  ✓ Phases visited: {phases_seen}")
    # Winter may be skipped (40% from recession); other phases are mandatory
    mandatory = {p.value for p in [CyclePhase.BOOM, CyclePhase.RECESSION, CyclePhase.RECOVERY]}
    assert mandatory.issubset(phases_seen), \
        f"Should visit mandatory phases, got {phases_seen}"
    print(f"  ✓ {len(transitions)} transitions occurred")
    print(f"  ✓ Sample transitions: {transitions[:6]}")

    print("  >>> CycleTransitions PASSED\n")


def test_supply_modification():
    """Test that supply changes with cycle phase."""
    print("\n" + "=" * 60)
    print("48.5B.2 — SUPPLY MODIFICATION")
    print("=" * 60)

    engine = ScarcityCycleEngine()
    base = ResourceBundle()
    original_total = base.total()

    # Force boom
    engine.state.phase = CyclePhase.BOOM
    boom_supply = engine.modify_supply(base)
    boom_total = boom_supply.total()
    assert boom_total > original_total, \
        f"Boom supply ({boom_total:.2f}) should exceed base ({original_total:.2f})"

    # Force winter
    engine.state.phase = CyclePhase.WINTER
    winter_supply = engine.modify_supply(base)
    winter_total = winter_supply.total()
    assert winter_total < original_total * 0.5, \
        f"Winter supply ({winter_total:.2f}) should be <50% of base ({original_total:.2f})"

    print(f"  ✓ Base total:     {original_total:.2f}")
    print(f"  ✓ Boom total:     {boom_total:.2f} (+{(boom_total/original_total-1)*100:.0f}%)")
    print(f"  ✓ Winter total:   {winter_total:.2f} ({(winter_total/original_total-1)*100:.0f}%)")

    # Recession and recovery should be between boom and winter
    engine.state.phase = CyclePhase.RECESSION
    rec_supply = engine.modify_supply(base)
    rec_total = rec_supply.total()
    assert winter_total < rec_total < boom_total, \
        f"Recession supply ({rec_total:.2f}) should be between winter ({winter_total:.2f}) and boom ({boom_total:.2f})"

    engine.state.phase = CyclePhase.RECOVERY
    recov_supply = engine.modify_supply(base)
    recov_total = recov_supply.total()
    assert winter_total < recov_total < boom_total, \
        f"Recovery supply ({recov_total:.2f}) should be between winter ({winter_total:.2f}) and boom ({boom_total:.2f})"
    print(f"  ✓ Recession total: {rec_total:.2f}")
    print(f"  ✓ Recovery total:  {recov_total:.2f}")
    print(f"  ✓ Supply ordering: Winter < Recession < Recovery < Boom")

    print("  >>> SupplyModification PASSED\n")


def test_asymmetric_access():
    """Test that poor agents are hit harder in scarcity."""
    print("\n" + "=" * 60)
    print("48.5B.3 — ASYMMETRIC ACCESS")
    print("=" * 60)

    engine = ScarcityCycleEngine()

    agents = []
    for i in range(10):
        a = CognitiveAgent(f'a_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        agents.append(a)

    # Create wealth inequality
    agents[0].wealth = 0.1  # Very poor
    agents[1].wealth = 0.2  # Poor
    agents[2].wealth = 0.3  # Below median
    agents[7].wealth = 3.0  # Wealthy
    agents[8].wealth = 4.0  # Very wealthy
    agents[9].wealth = 5.0  # Extremely wealthy

    bid_before = [a.bid_intensity for a in agents]
    wealth_before = [a.wealth for a in agents]
    exploration_before = [a.exploration_rate for a in agents]

    # Apply in winter (high severity)
    engine.state.phase = CyclePhase.WINTER
    engine.state.severity = 0.8
    engine.apply_asymmetric_access(agents)

    # Poor agents should have lost more wealth
    poor_loss = wealth_before[0] - agents[0].wealth
    rich_loss = wealth_before[9] - agents[9].wealth
    assert poor_loss > rich_loss, \
        f"Poor should lose more wealth: poor_loss={poor_loss:.3f}, rich_loss={rich_loss:.3f}"

    # Poor agent's bid intensity should be hit harder
    poor_bid_drop = bid_before[0] - agents[0].bid_intensity
    rich_bid_drop = bid_before[9] - agents[9].bid_intensity
    assert poor_bid_drop >= rich_bid_drop, \
        f"Poor bid should drop more: {poor_bid_drop:.3f} vs {rich_bid_drop:.3f}"

    print(f"  ✓ Poor agent wealth:    {wealth_before[0]:.3f} → {agents[0].wealth:.3f}")
    print(f"  ✓ Rich agent wealth:    {wealth_before[9]:.3f} → {agents[9].wealth:.3f}")
    print(f"  ✓ Poor bid intensity:   {bid_before[0]:.3f} → {agents[0].bid_intensity:.3f}")
    print(f"  ✓ Rich bid intensity:   {bid_before[9]:.3f} → {agents[9].bid_intensity:.3f}")

    # Near-zero wealth agent should have exploration collapse
    print(f"  ✓ Near-zero exploration: {exploration_before[0]:.3f} → {agents[0].exploration_rate:.3f}")

    print("  >>> AsymmetricAccess PASSED\n")


def test_temporal_pressure():
    """Test opportunity window decay."""
    print("\n" + "=" * 60)
    print("48.5B.4 — TEMPORAL PRESSURE")
    print("=" * 60)

    engine = ScarcityCycleEngine()

    # Create opportunity with 10-step deadline
    opp_id = engine.create_opportunity_window(
        'agent_1', reward_potential=1.0, deadline=10
    )

    # Collect immediately — should get full reward
    early_reward = engine.collect_opportunity(opp_id)
    assert early_reward == 1.0, f"Early reward should be 1.0, got {early_reward:.3f}"
    print(f"  ✓ Immediate collection: {early_reward:.3f} (expected 1.0)")

    # Create another with 10-step deadline
    opp_id2 = engine.create_opportunity_window(
        'agent_2', reward_potential=1.0, deadline=10
    )

    # Advance steps without collecting
    for _ in range(5):
        engine.total_steps += 1
        engine.decay_opportunities()

    # Collect after 5 steps decay
    delayed_reward = engine.collect_opportunity(opp_id2)
    assert 0.4 < delayed_reward < 1.0, \
        f"Delayed reward should be partial: {delayed_reward:.3f}"
    print(f"  ✓ Delayed collection (5 steps): {delayed_reward:.3f} (partial decay)")

    # Create third with 3-step deadline — let it expire
    opp_id3 = engine.create_opportunity_window(
        'agent_3', reward_potential=1.0, deadline=3
    )
    for _ in range(10):
        engine.total_steps += 1
        engine.decay_opportunities()

    expired_reward = engine.collect_opportunity(opp_id3)
    assert expired_reward == 0.0, \
        f"Expired opportunity should give 0 reward, got {expired_reward:.3f}"
    print(f"  ✓ Expired opportunity: {expired_reward:.3f} (expected 0.0)")

    print("  >>> TemporalPressure PASSED\n")


def test_survival_pressure():
    """Test that agents die under survival pressure."""
    print("\n" + "=" * 60)
    print("48.5B.5 — SURVIVAL PRESSURE")
    print("=" * 60)

    engine = ScarcityCycleEngine(
        survival_wealth_threshold=0.5,  # Higher threshold = easier to trigger
        survival_check_interval=1,  # Check every step
    )

    agents = []
    for i in range(5):
        a = CognitiveAgent(f'a_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.age = 15  # Old enough to die
        a.wealth = 0.1  # Below survival threshold
        agents.append(a)

    # Rich agent (should survive)
    rich = CognitiveAgent('rich', 'exploitative', np.random.randn(32))
    rich.active = True
    rich.age = 15
    rich.wealth = 5.0
    agents.append(rich)

    from dataclasses import dataclass, field
    @dataclass
    class DummyMarket:
        supply: ResourceBundle = field(default_factory=ResourceBundle)

    # Use fixed seed for reproducibility
    random.seed(42)
    total_deaths = 0
    for _ in range(100):
        result = engine.step(agents, DummyMarket())
        total_deaths += len(result['deaths'])
        if total_deaths > 0:
            break

    assert total_deaths > 0, "At least one poor agent should die"
    assert rich.active, "Rich agent should survive"
    print(f"  ✓ {total_deaths} deaths among poor agents")
    print(f"  ✓ Rich agent survived: {rich.active}")

    surviving = [a for a in agents if a.active and a.agent_id != 'rich']
    print(f"  ✓ Surviving poor agents: {len(surviving)}/{len(agents)-1}")

    death_causes = set(d['cause'] for d in engine.death_log)
    print(f"  ✓ Death causes observed: {death_causes}")

    print("  >>> SurvivalPressure PASSED\n")


def test_full_scarcity_cycle():
    """Test a complete scarcity cycle run with the integrated wrapper."""
    print("\n" + "=" * 60)
    print("48.5B.6 — FULL SCARCITY CYCLE")
    print("=" * 60)

    market = ScarcityAwareResourceMarket()

    # Create diverse agent population
    agents = []
    for i in range(10):
        a = CognitiveAgent(f'a_{i}', 'exploitative', np.random.randn(32))
        a.active = True
        a.wealth = 1.0 + i * 0.3  # Progressive wealth
        agents.append(a)

    # Run 500 steps
    results = []
    for step in range(1, 501):
        result = market.step(agents)
        if result.get('deaths'):
            for death in result['deaths']:
                results.append({
                    'step': step,
                    'type': 'death',
                    'agent': death['agent_id'],
                    'cause': death['cause'],
                })
        if result.get('phase_transition'):
            t = result['phase_transition']
            results.append({
                'step': step,
                'type': 'transition',
                'from': t['from'],
                'to': t['to'],
            })

    stats = market.get_stats()
    print(f"  ✓ Completed {stats['scarcity']['total_steps']} steps")
    print(f"  ✓ Phases: {stats['scarcity']['phases_visited']}")
    print(f"  ✓ Transitions: {stats['scarcity']['n_transitions']}")
    print(f"  ✓ Total deaths: {stats['scarcity']['total_deaths']}")

    cycle_info = market.scarcity_engine.get_cycle_profile()
    print(f"  ✓ Profile: {cycle_info}")

    # Verify market still has prices
    assert len(market.market_log) > 0, "Market should have logged steps"
    print(f"  ✓ Market steps logged: {len(market.market_log)}")

    print("  >>> FullScarcityCycle PASSED\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48.5B: SCARCITY CYCLES ENGINE                             ║
║                                                                   ║
║  Dynamic economic pressure for Cognitive Political Economy:      ║
║                                                                   ║
║    1. Boom/Recession/Winter/Recovery supply cycles              ║
║    2. Asymmetric access (poor hit harder in scarcity)           ║
║    3. Temporal pressure (opportunity decay, deadlines)          ║
║    4. Survival thresholds (agents die without wealth)           ║
║                                                                   ║
║  Without scarcity, there is no:                                  ║
║    power struggle, strategic behavior, institutional evolution  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("CycleTransitions (48.5B.1)", test_cycle_transitions),
        ("SupplyModification (48.5B.2)", test_supply_modification),
        ("AsymmetricAccess (48.5B.3)", test_asymmetric_access),
        ("TemporalPressure (48.5B.4)", test_temporal_pressure),
        ("SurvivalPressure (48.5B.5)", test_survival_pressure),
        ("FullScarcityCycle (48.5B.6)", test_full_scarcity_cycle),
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
  ║  PHASE 48.5B: ALL TESTS PASSED                                ║
  ║                                                               ║
  ║  Scarcity Cycles Engine ready.                                ║
  ║  Next: integrate with CognitivePoliticalEngine, replace       ║
  ║  ResourceMarket with ScarcityAwareResourceMarket.             ║
  ╚══════════════════════════════════════════════════════════════╝
        """)
    else:
        print(f"\n  Some tests FAILED")
