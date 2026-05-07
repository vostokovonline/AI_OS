"""
SCENARIO STRESS HARNESS
=======================

Controlled scenario tests for validating stability before Monte-Carlo.

Engineering principle:
    Stability is not a number - it's behavior under controlled pressure.

Steps:
    1. Controlled scenario (500 cycles) - fixed success rates
    2. Shock scenario (500 cycles) - sudden degradation
    3. Drift scenario (1000 cycles) - gradual environmental changes
    4. Monte-Carlo (3000 cycles) - random noise and shifts

Author: AI-OS Team
Date: 2026-02-21
"""
import sys
import os
# In container, files are at /app, not services/core
sys.path.insert(0, '/app')

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from uuid import UUID, uuid4
import json
import math

from autonomy.arbitration import (
    ActionArbitrator,
    ArbitrationContext,
    ArbitrationConfig,
    EmotionalSnapshot,
    ResourceSnapshot,
    StrategyConfig,
    StrategyRuntimeStats,
    SystemStateSnapshot,
    DecisionAction,
    RiskLevel
)
from autonomy.policy_engine import ActionType
from autonomy.stability_guards import (
    AntiMonopolyGuard,
    FailureShockAbsorber,
    ObservabilityTracker,
    reset_all_guards,
    get_anti_monopoly_guard,
    get_failure_shock_absorber,
    RESILIENCE_PROFILE,
    ArbitrationProfile
)
from autonomy.stability_monitor import StabilityMonitor, StabilityConfig
from logging_config import get_logger

logger = get_logger(__name__)


# ============================================================
# SCENARIO CONFIGURATION
# ============================================================

@dataclass
class StrategyProfile:
    """Profile for a strategy in the scenario."""
    id: UUID
    name: str
    base_success_rate: float
    current_success_rate: float
    priority: float
    
    def __post_init__(self):
        self.current_success_rate = self.base_success_rate


@dataclass
class ScenarioConfig:
    """Configuration for a scenario run."""
    name: str
    description: str
    num_cycles: int
    observation_window: int = 100
    profile: ArbitrationProfile = RESILIENCE_PROFILE
    
    # Strategy profiles
    strategies: List[StrategyProfile] = field(default_factory=list)
    
    # Success rate modifiers (per cycle)
    success_rate_modifier: Optional[Callable[[int, StrategyProfile], float]] = None
    
    # Expected outcomes
    expected_min_entropy: float = 0.5
    expected_max_dominance: float = 0.75  # Match profile's dominance_cap
    expected_health_rate: float = 0.8


# ============================================================
# PREDEFINED SCENARIOS
# ============================================================

def create_controlled_scenario() -> ScenarioConfig:
    """
    Step 1: Controlled scenario (500 cycles)
    
    5 strategies with fixed success rates:
        A: 0.90 (best performer)
        B: 0.75
        C: 0.60
        D: 0.45
        E: 0.30 (worst performer)
    
    Noise: ±5%
    
    Validates:
        - A doesn't dominate >70%
        - entropy > 0.5
        - anti-monopoly activates
        - EMA converges smoothly
    """
    return ScenarioConfig(
        name="controlled_500",
        description="Controlled scenario with fixed success rates",
        num_cycles=500,
        observation_window=100,
        strategies=[
            StrategyProfile(
                id=uuid4(),
                name="Strategy_A",
                base_success_rate=0.90,
                current_success_rate=0.90,
                priority=1.0
            ),
            StrategyProfile(
                id=uuid4(),
                name="Strategy_B",
                base_success_rate=0.75,
                current_success_rate=0.75,
                priority=1.0
            ),
            StrategyProfile(
                id=uuid4(),
                name="Strategy_C",
                base_success_rate=0.60,
                current_success_rate=0.60,
                priority=1.0
            ),
            StrategyProfile(
                id=uuid4(),
                name="Strategy_D",
                base_success_rate=0.45,
                current_success_rate=0.45,
                priority=1.0
            ),
            StrategyProfile(
                id=uuid4(),
                name="Strategy_E",
                base_success_rate=0.30,
                current_success_rate=0.30,
                priority=1.0
            )
        ],
        expected_min_entropy=0.5,
        expected_max_dominance=0.7,
        expected_health_rate=0.8
    )


def create_shock_scenario() -> ScenarioConfig:
    """
    Step 2: Shock scenario (500 cycles)
    
    At cycle 250:
        - Strategy A degrades suddenly: 0.9 → 0.3
    
    Validates:
        - EMA responds smoothly
        - Dominance drops gradually
        - No entropy collapse
        - No rapid oscillation
    """
    config = create_controlled_scenario()
    config.name = "shock_500"
    config.description = "Shock scenario with sudden degradation at cycle 250"
    
    def shock_modifier(cycle: int, profile: StrategyProfile) -> float:
        if profile.name == "Strategy_A" and cycle >= 250:
            return 0.30
        return profile.base_success_rate
    
    config.success_rate_modifier = shock_modifier
    return config


def create_drift_scenario() -> ScenarioConfig:
    """
    Step 3: Drift scenario (1000 cycles)
    
    Gradual environmental changes:
        - Strategy A degrades: 0.9 → 0.5 over 1000 cycles
        - Strategy C improves: 0.6 → 0.8 over 1000 cycles
    
    Validates:
        - System adapts gradually
        - No hysteresis lock
        - No stuck behavior
    """
    config = create_controlled_scenario()
    config.name = "drift_1000"
    config.description = "Drift scenario with gradual environmental changes"
    config.num_cycles = 1000
    
    def drift_modifier(cycle: int, profile: StrategyProfile) -> float:
        progress = cycle / 1000.0
        
        if profile.name == "Strategy_A":
            return 0.9 - 0.4 * progress
        elif profile.name == "Strategy_C":
            return 0.6 + 0.2 * progress
        else:
            return profile.base_success_rate
    
    config.success_rate_modifier = drift_modifier
    return config


# ============================================================
# SCENARIO RUNNER
# ============================================================

@dataclass
class CycleResult:
    """Result of a single cycle."""
    cycle: int
    selected_strategy: str
    selected_strategy_id: UUID
    outcome: bool
    utilities: Dict[str, float]
    ema_values: Dict[str, float]


@dataclass
class ScenarioResult:
    """Result of a scenario run."""
    scenario_name: str
    num_cycles: int
    cycles: List[CycleResult]
    
    summary: Dict
    
    stability_summary: Dict
    observability_stats: Dict
    
    passed: bool
    failures: List[str]
    
    def export_json(self, filepath: str):
        data = {
            "scenario_name": self.scenario_name,
            "num_cycles": self.num_cycles,
            "summary": self.summary,
            "stability_summary": self.stability_summary,
            "observability_stats": self.observability_stats,
            "passed": self.passed,
            "failures": self.failures
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class ScenarioRunner:
    """
    Runs controlled scenarios and validates stability.
    """
    
    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.profiles = {s.name: s for s in config.strategies}
        
        # Reset all guards for clean state
        reset_all_guards()
        
        # Configure singleton guard with profile (arbitration.py uses singleton)
        singleton_guard = get_anti_monopoly_guard()
        singleton_guard.set_profile(config.profile)
        
        # Use singleton shock absorber (arbitration.py uses singleton)
        self.shock_absorber = get_failure_shock_absorber()
        
        # Initialize components
        self.arbitrator = ActionArbitrator(ArbitrationConfig())
        self.anti_monopoly = singleton_guard
        self.stability_monitor = StabilityMonitor(StabilityConfig(
            observation_window=config.observation_window
        ))
        
        # Runtime stats per strategy
        self.runtime_stats: Dict[UUID, StrategyRuntimeStats] = {
            s.id: StrategyRuntimeStats(strategy_id=s.id)
            for s in config.strategies
        }
        
        # Results
        self.cycle_results: List[CycleResult] = []
    
    def run(self) -> ScenarioResult:
        """Run the scenario and collect results."""
        print(f"\n{'='*60}")
        print(f"SCENARIO: {self.config.name}")
        print(f"Description: {self.config.description}")
        print(f"Cycles: {self.config.num_cycles}")
        print(f"{'='*60}\n")
        
        for cycle in range(self.config.num_cycles):
            result = self._run_cycle(cycle)
            self.cycle_results.append(result)
            
            # Progress indicator
            if (cycle + 1) % 100 == 0:
                self._print_progress(cycle + 1)
            
            # Observe stability at window boundaries
            if (cycle + 1) % self.config.observation_window == 0:
                metrics = self.stability_monitor.observe()
                if not metrics.is_healthy():
                    print(f"\n  [!] WARNING at cycle {cycle + 1}:")
                    for w in metrics.warnings:
                        print(f"      {w}")
        
        # Final observation
        final_metrics = self.stability_monitor.observe()
        
        return self._build_result(final_metrics)
    
    def _run_cycle(self, cycle: int) -> CycleResult:
        """Run a single arbitration cycle."""
        import random
        
        # Apply success rate modifier
        for profile in self.config.strategies:
            if self.config.success_rate_modifier:
                profile.current_success_rate = self.config.success_rate_modifier(cycle, profile)
            else:
                profile.current_success_rate = profile.base_success_rate
        
        # Create arbitration context
        context = self._create_context()
        
        # Create action candidates (one per strategy)
        actions = []
        for profile in self.config.strategies:
            action = DecisionAction(
                id=uuid4(),
                action_type=ActionType.CREATE_GOAL,
                action_payload={"strategy": profile.name},
                strategy_id=profile.id,
                source_rule_name="scenario_test",
                reason="Scenario candidate",
                risk_level=RiskLevel.LOW
            )
            actions.append(action)
        
        # Arbitrate
        result = self.arbitrator.resolve(actions, context)
        
        if result.selected_action is None:
            raise RuntimeError("No action selected in scenario")
        
        selected_profile = self.profiles[
            self._get_strategy_name(result.selected_action.strategy_id)
        ]
        
        # Record selection in stability monitor
        self.stability_monitor.record_selection(cycle, str(selected_profile.id))
        
        # Determine outcome based on success rate (with ±5% noise)
        noise = (random.random() - 0.5) * 0.10
        success_threshold = selected_profile.current_success_rate + noise
        outcome = random.random() < success_threshold
        
        # Update runtime stats
        stats = self.runtime_stats[selected_profile.id]
        stats.record_activation()
        if outcome:
            stats.record_success()
        else:
            stats.record_failure()
        
        # Record in shock absorber
        self.shock_absorber.record_outcome(str(selected_profile.id), outcome)
        
        # Collect utilities with debugging
        utilities = {}
        for breakdown in result.breakdowns:
            name = self._get_strategy_name(breakdown.strategy_id)
            utilities[name] = breakdown.final_score
        
        # Debug: print top 3 utilities with guard details
        sorted_utils = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
        if cycle < 5 or cycle == 100:
            print(f"\n  Cycle {cycle} utilities (tie_broken={result.tie_broken}):")
            guard_stats = self.anti_monopoly.get_stats()
            for name, util in sorted_utils[:5]:  # All 5 strategies
                profile = self.profiles[name]
                sid = str(profile.id)
                # Find the breakdown for this strategy
                breakdown = next((b for b in result.breakdowns if str(b.strategy_id) == sid), None)
                if breakdown:
                    print(f"    {name}: final={util:.4f}, perf={breakdown.performance:.3f}, "
                          f"prio={breakdown.priority:.3f}, emo={breakdown.emotion_modifier:.3f}, "
                          f"res={breakdown.resource_factor:.3f}, risk={breakdown.risk_adjustment:.3f}, "
                          f"rec={breakdown.recency_penalty:.3f}")
            print(f"    Selected: {selected_profile.name}")
            print(f"    Guard: total={guard_stats['total_recent']}, strategies={guard_stats['unique_strategies']}, dist={guard_stats['distribution']}")
        
        # Collect EMA values
        ema_values = {}
        for profile in self.config.strategies:
            ema_values[profile.name] = self.shock_absorber.get_ema_success(str(profile.id))
        
        return CycleResult(
            cycle=cycle,
            selected_strategy=selected_profile.name,
            selected_strategy_id=selected_profile.id,
            outcome=outcome,
            utilities=utilities,
            ema_values=ema_values
        )
    
    def _create_context(self) -> ArbitrationContext:
        """Create arbitration context."""
        strategy_configs = {
            s.id: StrategyConfig(
                strategy_id=s.id,
                name=s.name,
                priority=s.priority,
                default_risk_level=RiskLevel.LOW,
                cost_estimate=0.0
            )
            for s in self.config.strategies
        }
        
        return ArbitrationContext(
            system_state=SystemStateSnapshot(
                metrics={},
                trends={}
            ),
            emotion=EmotionalSnapshot(
                valence=0.5,
                arousal=0.5,
                stress=0.3,
                confidence=0.7,
                momentum=0.5
            ),
            resources=ResourceSnapshot(
                budget_remaining=1000.0,
                budget_limit=1000.0,
                concurrent_goals=5,
                max_concurrent_goals=20,
                compute_available=0.8
            ),
            strategy_configs=strategy_configs,
            strategy_stats=self.runtime_stats,
            config=ArbitrationConfig()
        )
    
    def _get_strategy_name(self, strategy_id: UUID) -> str:
        """Get strategy name by ID."""
        for profile in self.config.strategies:
            if profile.id == strategy_id:
                return profile.name
        return "unknown"
    
    def _print_progress(self, cycle: int):
        """Print progress indicator."""
        dist = self._get_selection_distribution()
        entropy = self._calculate_entropy(dist)
        
        print(f"  Cycle {cycle:4d}: entropy={entropy:.3f}, top={max(dist.values()) if dist else 0} selections")
    
    def _get_selection_distribution(self) -> Dict[str, int]:
        """Get distribution of selections."""
        dist: Dict[str, int] = {}
        for r in self.cycle_results:
            dist[r.selected_strategy] = dist.get(r.selected_strategy, 0) + 1
        return dist
    
    def _calculate_entropy(self, distribution: Dict[str, int]) -> float:
        """Calculate Shannon entropy."""
        if not distribution:
            return 0.0
        
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in distribution.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _build_result(self, final_metrics) -> ScenarioResult:
        """Build final result with validation."""
        dist = self._get_selection_distribution()
        total_selections = sum(dist.values())
        
        # Calculate dominance
        max_selections = max(dist.values()) if dist else 0
        dominance = max_selections / total_selections if total_selections > 0 else 0.0
        
        # Calculate final entropy
        entropy = self._calculate_entropy(dist)
        
        # Get stability summary
        stability_summary = self.stability_monitor.get_summary()
        
        # Calculate EMA trajectories
        ema_trajectories: Dict[str, List[float]] = {s.name: [] for s in self.config.strategies}
        for r in self.cycle_results:
            for name, ema in r.ema_values.items():
                ema_trajectories[name].append(ema)
        
        # Calculate EMA oscillation (std dev of changes)
        ema_oscillation: Dict[str, float] = {}
        for name, trajectory in ema_trajectories.items():
            if len(trajectory) > 1:
                changes = [abs(trajectory[i] - trajectory[i-1]) for i in range(1, len(trajectory))]
                avg_change = sum(changes) / len(changes) if changes else 0.0
                ema_oscillation[name] = avg_change
            else:
                ema_oscillation[name] = 0.0
        
        # Validation
        failures = []
        
        if entropy < self.config.expected_min_entropy:
            failures.append(f"Entropy {entropy:.3f} < minimum {self.config.expected_min_entropy}")
        
        if dominance > self.config.expected_max_dominance:
            failures.append(f"Dominance {dominance:.1%} > maximum {self.config.expected_max_dominance:.1%}")
        
        health_rate = stability_summary.get('health_rate', 1.0)
        if health_rate < self.config.expected_health_rate:
            failures.append(f"Health rate {health_rate:.1%} < minimum {self.config.expected_health_rate:.1%}")
        
        # Check for EMA oscillation (>0.1 average change is too noisy)
        max_oscillation = max(ema_oscillation.values()) if ema_oscillation else 0.0
        if max_oscillation > 0.1:
            failures.append(f"EMA oscillation {max_oscillation:.3f} > 0.1 (too noisy)")
        
        passed = len(failures) == 0
        
        # Build summary
        summary = {
            "total_selections": total_selections,
            "selection_distribution": dist,
            "entropy": round(entropy, 4),
            "dominance": round(dominance, 4),
            "dominant_strategy": max(dist, key=dist.get) if dist else "none",
            "health_rate": round(health_rate, 4),
            "ema_oscillation": {k: round(v, 4) for k, v in ema_oscillation.items()},
            "final_ema_values": {k: round(v[-1], 4) if v else 0.0 for k, v in ema_trajectories.items()}
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.config.name}")
        print(f"{'='*60}")
        print(f"  Selections: {total_selections}")
        print(f"  Distribution: {dist}")
        print(f"  Entropy: {entropy:.4f} (min: {self.config.expected_min_entropy})")
        print(f"  Dominance: {dominance:.1%} (max: {self.config.expected_max_dominance:.1%})")
        print(f"  Health Rate: {health_rate:.1%} (min: {self.config.expected_health_rate:.1%})")
        print(f"  Max EMA Oscillation: {max_oscillation:.4f}")
        print(f"  Final EMA Values: {summary['final_ema_values']}")
        print(f"\n  PASSED: {passed}")
        if failures:
            print(f"  FAILURES:")
            for f in failures:
                print(f"    - {f}")
        print(f"{'='*60}\n")
        
        return ScenarioResult(
            scenario_name=self.config.name,
            num_cycles=self.config.num_cycles,
            cycles=self.cycle_results,
            summary=summary,
            stability_summary=stability_summary,
            observability_stats={},
            passed=passed,
            failures=failures
        )


# ============================================================
# MAIN
# ============================================================

def run_all_scenarios():
    """Run all three scenarios in sequence."""
    results = []
    
    # Step 1: Controlled scenario
    print("\n" + "="*70)
    print("STEP 1: CONTROLLED SCENARIO")
    print("="*70)
    
    controlled = create_controlled_scenario()
    runner1 = ScenarioRunner(controlled)
    result1 = runner1.run()
    results.append(result1)
    
    if not result1.passed:
        print("\n[!] CONTROLLED SCENARIO FAILED - stopping before shock test")
        return results
    
    # Step 2: Shock scenario
    print("\n" + "="*70)
    print("STEP 2: SHOCK SCENARIO")
    print("="*70)
    
    shock = create_shock_scenario()
    runner2 = ScenarioRunner(shock)
    result2 = runner2.run()
    results.append(result2)
    
    if not result2.passed:
        print("\n[!] SHOCK SCENARIO FAILED - stopping before drift test")
        return results
    
    # Step 3: Drift scenario
    print("\n" + "="*70)
    print("STEP 3: DRIFT SCENARIO")
    print("="*70)
    
    drift = create_drift_scenario()
    runner3 = ScenarioRunner(drift)
    result3 = runner3.run()
    results.append(result3)
    
    # Summary
    print("\n" + "="*70)
    print("ALL SCENARIOS COMPLETE")
    print("="*70)
    
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  {r.scenario_name}: {status}")
    
    all_passed = all(r.passed for r in results)
    print(f"\n  OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    return results


if __name__ == "__main__":
    results = run_all_scenarios()
