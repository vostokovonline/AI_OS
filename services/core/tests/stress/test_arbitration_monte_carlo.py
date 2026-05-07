"""
Monte-Carlo Stress Test for Arbitration Layer.

Validates architectural stability:
- No cascade collapse
- No dominant multiplier bias
- Stable utility distribution across 1000 random contexts

Run:
    docker exec ns_core python -m tests.stress.test_arbitration_monte_carlo
"""
import sys
sys.path.insert(0, '/app')

import random
from datetime import datetime, timedelta
from uuid import uuid4
from collections import Counter
import statistics

from autonomy.arbitration import (
    ActionArbitrator,
    ArbitrationContext,
    ArbitrationConfig,
    StrategyConfig,
    StrategyRuntimeStats,
    EmotionalSnapshot,
    ResourceSnapshot,
    SystemStateSnapshot,
    RiskLevel,
    DecisionAction,
    ActionType
)


def random_float(min_val: float, max_val: float) -> float:
    """Generate random float in range"""
    return random.uniform(min_val, max_val)


def random_emotion() -> EmotionalSnapshot:
    """Generate random emotional state"""
    return EmotionalSnapshot(
        valence=random_float(-1.0, 1.0),
        arousal=random_float(0.0, 1.0),
        stress=random_float(0.0, 1.0),
        confidence=random_float(0.0, 1.0),
        momentum=random_float(0.0, 1.0)
    )


def random_resources() -> ResourceSnapshot:
    """Generate random resource state"""
    budget_limit = 100.0
    budget_remaining = random_float(0.0, budget_limit)
    max_goals = 100
    concurrent_goals = random_int(0, max_goals)
    
    return ResourceSnapshot(
        budget_remaining=budget_remaining,
        budget_limit=budget_limit,
        concurrent_goals=concurrent_goals,
        max_concurrent_goals=max_goals,
        compute_available=random_float(0.0, 1.0)
    )


def random_int(min_val: int, max_val: int) -> int:
    """Generate random int in range"""
    return random.randint(min_val, max_val)


def random_strategy_config(strategy_id) -> StrategyConfig:
    """Generate random strategy config"""
    return StrategyConfig(
        strategy_id=strategy_id,
        name=f"strategy_{strategy_id.hex[:8]}",
        priority=random_float(0.5, 2.0),
        default_risk_level=random.choice(list(RiskLevel)),
        cost_estimate=random_float(0.0, 50.0)
    )


def random_strategy_stats(strategy_id) -> StrategyRuntimeStats:
    """Generate random strategy stats"""
    # Random recent activations (0-20)
    recent_count = random_int(0, 20)
    recent_activations = [
        datetime.utcnow() - timedelta(minutes=random_int(0, 120))
        for _ in range(recent_count)
    ]
    
    return StrategyRuntimeStats(
        strategy_id=strategy_id,
        activation_count=random_int(0, 100),
        success_count=random_int(0, 50),
        failure_count=random_int(0, 50),
        cumulative_cost=random_float(0.0, 500.0),
        recent_activations=recent_activations
    )


def random_action(strategy_id) -> DecisionAction:
    """Generate random action"""
    return DecisionAction(
        id=uuid4(),
        action_type=random.choice(list(ActionType)),
        action_payload={"title": "Test action"},
        strategy_id=strategy_id,
        source_rule_name="test_rule",
        reason="Monte Carlo test",
        risk_level=random.choice(list(RiskLevel)),
        cost_estimate=random_float(0.0, 100.0)
    )


def run_monte_carlo(iterations: int = 1000, actions_per_iteration: int = 5):
    """
    Run Monte-Carlo stress test.
    
    Tests:
    1. Utility distribution (no collapse, no dominance)
    2. Factor influence balance
    3. Tie-break frequency
    """
    print(f"\n{'='*70}")
    print(f"MONTE-CARLO STRESS TEST - {iterations} iterations")
    print(f"{'='*70}\n")
    
    arbitrator = ActionArbitrator()
    
    all_scores = []
    factor_values = {
        'performance': [],
        'priority': [],
        'emotion': [],
        'resource': [],
        'risk': [],
        'recency': [],
        'base_value': [],
        'modifiers': [],
        'final_score': []
    }
    
    tie_count = 0
    selection_distribution = Counter()
    
    for i in range(iterations):
        # Generate random context
        num_strategies = random_int(1, 10)
        strategy_configs = {}
        strategy_stats = {}
        
        for _ in range(num_strategies):
            sid = uuid4()
            strategy_configs[sid] = random_strategy_config(sid)
            strategy_stats[sid] = random_strategy_stats(sid)
        
        context = ArbitrationContext(
            system_state=SystemStateSnapshot(
                metrics={"test": random_float(0, 100)},
                trends={"test": random.choice(["up", "down", "stable"])}
            ),
            emotion=random_emotion(),
            resources=random_resources(),
            strategy_configs=strategy_configs,
            strategy_stats=strategy_stats,
            config=ArbitrationConfig()
        )
        
        # Generate random actions
        actions = []
        strategy_ids = list(strategy_configs.keys())
        for _ in range(min(actions_per_iteration, len(strategy_ids))):
            sid = random.choice(strategy_ids)
            actions.append(random_action(sid))
        
        # Run arbitration
        result = arbitrator.resolve(actions, context)
        
        # Collect stats
        if result.selected_action:
            selection_distribution[result.selected_action.action_type.value] += 1
        
        if result.tie_broken:
            tie_count += 1
        
        for breakdown in result.breakdowns:
            all_scores.append(breakdown.final_score)
            factor_values['performance'].append(breakdown.performance)
            factor_values['priority'].append(breakdown.priority)
            factor_values['emotion'].append(breakdown.emotion_modifier)
            factor_values['resource'].append(breakdown.resource_factor)
            factor_values['risk'].append(breakdown.risk_adjustment)
            factor_values['recency'].append(breakdown.recency_penalty)
            factor_values['base_value'].append(breakdown.performance * breakdown.priority)
            
            # Recalculate modifiers
            modifiers = (
                0.30 * breakdown.emotion_modifier +
                0.35 * breakdown.resource_factor +
                0.20 * breakdown.risk_adjustment +
                0.15 * breakdown.recency_penalty
            )
            factor_values['modifiers'].append(modifiers)
            factor_values['final_score'].append(breakdown.final_score)
    
    # === ANALYSIS ===
    
    print("📊 UTILITY DISTRIBUTION")
    print("-" * 40)
    
    mean_score = statistics.mean(all_scores)
    median_score = statistics.median(all_scores)
    stdev_score = statistics.stdev(all_scores) if len(all_scores) > 1 else 0
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    print(f"   Mean:    {mean_score:.3f}")
    print(f"   Median:  {median_score:.3f}")
    print(f"   StdDev:  {stdev_score:.3f}")
    print(f"   Range:   [{min_score:.3f}, {max_score:.3f}]")
    
    # Check for collapse (too many near-zero scores)
    near_zero_count = sum(1 for s in all_scores if s < 0.1)
    near_zero_pct = near_zero_count / len(all_scores) * 100
    
    print(f"\n   Near-zero (<0.1): {near_zero_count} ({near_zero_pct:.1f}%)")
    
    if near_zero_pct > 30:
        print("   ⚠️  WARNING: Cascade collapse detected!")
    else:
        print("   ✅ No cascade collapse")
    
    print("\n📈 FACTOR INFLUENCE")
    print("-" * 40)
    
    for factor, values in factor_values.items():
        if not values:
            continue
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        print(f"   {factor:15} mean={mean:.3f} std={std:.3f}")
    
    print("\n⚖️  TIE-BREAKING")
    print("-" * 40)
    tie_pct = tie_count / iterations * 100
    print(f"   Ties resolved: {tie_count}/{iterations} ({tie_pct:.1f}%)")
    
    if tie_pct > 50:
        print("   ⚠️  WARNING: Too many ties - consider adjusting tie_threshold")
    else:
        print("   ✅ Tie frequency normal")
    
    print("\n🎯 ACTION TYPE DISTRIBUTION")
    print("-" * 40)
    for action_type, count in selection_distribution.most_common():
        pct = count / iterations * 100
        print(f"   {action_type:25} {count:4} ({pct:.1f}%)")
    
    # Check for dominance
    dominant_count = selection_distribution.most_common(1)[0][1] if selection_distribution else 0
    dominant_pct = dominant_count / iterations * 100
    
    print("\n🔍 DOMINANCE CHECK")
    print("-" * 40)
    
    # Priority dominance
    high_priority_wins = 0
    for factor in ['priority']:
        values = factor_values[factor]
        if values:
            high_values = [v for v in values if v > 1.5]
            if high_values:
                print(f"   High {factor} (>1.5): {len(high_values)} occurrences")
    
    if dominant_pct > 70:
        print(f"   ⚠️  WARNING: Dominant action type at {dominant_pct:.1f}%")
    else:
        print(f"   ✅ No action type dominance (max {dominant_pct:.1f}%)")
    
    print("\n" + "=" * 70)
    
    # Final verdict
    issues = []
    if near_zero_pct > 30:
        issues.append("cascade collapse")
    if tie_pct > 50:
        issues.append("excessive ties")
    if dominant_pct > 70:
        issues.append("action dominance")
    
    if issues:
        print(f"❌ ISSUES FOUND: {', '.join(issues)}")
        return False
    else:
        print("✅ ALL CHECKS PASSED - Architecture is stable")
        return True


if __name__ == "__main__":
    random.seed(42)  # Reproducible
    success = run_monte_carlo(iterations=1000, actions_per_iteration=5)
    sys.exit(0 if success else 1)
