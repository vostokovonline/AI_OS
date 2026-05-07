"""
SMOKE TEST: Capital Engine Integration
======================================

Validates integration of Capital Engine into arbitration.py

Author: AI-OS Team
Date: 2026-02-21
"""
import sys
sys.path.insert(0, '/app')

from uuid import uuid4, UUID

from autonomy.arbitration import (
    ActionArbitrator,
    ArbitrationConfig,
    ArbitrationContext,
    DecisionAction,
    EmotionalSnapshot,
    ResourceSnapshot,
    SystemStateSnapshot,
    StrategyConfig,
    StrategyRuntimeStats
)
from autonomy.policy_engine import ActionType
from autonomy.stability_guards import reset_all_guards
from autonomy.capital_engine import reset_capital_allocator

def run_smoke_test():
    print("=" * 70)
    print("SMOKE TEST: Capital Engine Integration")
    print("=" * 70)
    print()
    
    # Reset
    reset_all_guards()
    reset_capital_allocator()
    
    # Create 3 test strategies
    strategies = [
        {
            "id": uuid4(),
            "name": "Strategy_A",
            "priority": 1.0,
            "cost": 0.01
        },
        {
            "id": uuid4(),
            "name": "Strategy_B",
            "priority": 0.8,
            "cost": 0.02
        },
        {
            "id": uuid4(),
            "name": "Strategy_C",
            "priority": 0.6,
            "cost": 0.015
        }
    ]
    
    # Create action candidates
    actions = []
    for s in strategies:
        action = DecisionAction(
            id=uuid4(),
            action_type=ActionType.CREATE_GOAL,
            action_payload={"test": True},
            strategy_id=s["id"],
            source_rule_name="smoke_test",
            reason="Test candidate"
        )
        actions.append(action)
    
    # Create context
    strategy_configs = {
        s["id"]: StrategyConfig(
            strategy_id=s["id"],
            name=s["name"],
            priority=s["priority"],
            default_risk_level=2,
            cost_estimate=s["cost"]
        )
        for s in strategies
    }
    
    strategy_stats = {
        s["id"]: StrategyRuntimeStats(strategy_id=s["id"])
        for s in strategies
    }
    
    context = ArbitrationContext(
        system_state=SystemStateSnapshot(metrics={}, trends={}),
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
        strategy_stats=strategy_stats,
        config=ArbitrationConfig()
    )
    
    print("TEST 1: Traditional single-winner arbitration")
    print("-" * 70)
    
    arbitrator = ActionArbitrator(ArbitrationConfig(enable_capital_allocation=False))
    result = arbitrator.resolve(actions, context)
    
    print(f"  Selected: {result.selected_action.strategy_id}")
    print(f"  Candidates: {result.candidates_count}")
    print(f"  Mode: Single winner")
    print()
    
    print("TEST 2: Portfolio capital allocation")
    print("-" * 70)
    
    reset_all_guards()
    reset_capital_allocator()
    
    arbitrator2 = ActionArbitrator(ArbitrationConfig(enable_capital_allocation=True))
    result2 = arbitrator2.resolve_with_allocation(actions, context)
    
    print(f"  Allocations:")
    for sid, alloc in result2.allocations.items():
        name = next(s["name"] for s in strategies if s["id"] == sid)
        print(f"    {name}: {alloc:.1%}")
    
    print(f"  Total deployed: {result2.total_capital_deployed:.2f}")
    print(f"  Top strategy: {result2.get_top_strategy()}")
    print()
    
    # Verify allocations sum to 1.0
    total_alloc = sum(result2.allocations.values())
    assert abs(total_alloc - 1.0) < 0.01, f"Allocations must sum to 1.0, got {total_alloc}"
    
    # Verify no single strategy dominates
    max_alloc = max(result2.allocations.values())
    assert max_alloc < 0.6, f"Max allocation too high: {max_alloc}"
    
    print("TEST 3: Multiple cycles (allocation adaptation)")
    print("-" * 70)
    
    for i in range(5):
        result = arbitrator2.resolve_with_allocation(actions, context)
        allocs = {str(k)[:8]: round(v, 2) for k, v in result.allocations.items()}
        print(f"  Cycle {i+1}: {allocs}")
    
    print()
    print("=" * 70)
    print("SMOKE TEST: PASS ✅")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    run_smoke_test()
