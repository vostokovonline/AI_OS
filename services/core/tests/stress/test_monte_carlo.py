"""
STAGE 4 MONTE-CARLO VALIDATION
==============================

3000 cycles stress test with stochastic regime shifts.

Validates:
- Survival rate under prolonged stress
- Max drawdown distribution
- Allocation adaptability
- Recovery capability
- Edge case identification

Author: AI-OS Team
Date: 2026-02-21
"""
import sys
sys.path.insert(0, '/app')

from uuid import uuid4, UUID
import random
import json
from datetime import datetime
from typing import Dict, List, Tuple

from autonomy.capital_engine import (
    CapitalAllocator,
    CapitalConfig,
    StrategyAsset,
    get_capital_allocator,
    reset_capital_allocator
)
from autonomy.stability_guards import (
    get_failure_shock_absorber,
    reset_all_guards
)


def run_monte_carlo(
    num_cycles: int = 3000,
    seed: int = 42,
    regime_shifts: List[Dict] = None,
    random_degradation: bool = True
) -> Dict:
    """
    Run Monte-Carlo validation with stochastic regime shifts.
    
    Args:
        num_cycles: Total cycles (default 3000)
        seed: Random seed for reproducibility
        regime_shifts: List of {cycle, strategy_idx, new_rate} dicts
        random_degradation: Enable random alpha fluctuations
    
    Returns:
        Dict with full metrics history and summary
    """
    random.seed(seed)
    
    print("=" * 70)
    print("STAGE 4 MONTE-CARLO VALIDATION")
    print("=" * 70)
    print(f"Cycles: {num_cycles}")
    print(f"Seed: {seed}")
    print(f"Random degradation: {random_degradation}")
    print()
    
    # Reset
    reset_all_guards()
    reset_capital_allocator()
    
    # Strategies
    strategies = [
        {
            "id": uuid4(),
            "name": "High_Performer",
            "base_success_rate": 0.85,
            "current_success_rate": 0.85,
            "payoff": 0.01,
            "cost": 0.002,
            "variance": 0.05
        },
        {
            "id": uuid4(),
            "name": "Medium_Performer",
            "base_success_rate": 0.65,
            "current_success_rate": 0.65,
            "payoff": 0.01,
            "cost": 0.002,
            "variance": 0.10
        },
        {
            "id": uuid4(),
            "name": "Low_Performer",
            "base_success_rate": 0.45,
            "current_success_rate": 0.45,
            "payoff": 0.01,
            "cost": 0.002,
            "variance": 0.15
        }
    ]
    
    # Default regime shifts if not provided
    if regime_shifts is None:
        regime_shifts = [
            {"cycle": 500, "strategy_idx": 0, "new_rate": 0.42},   # Shock
            {"cycle": 700, "strategy_idx": 0, "new_rate": 0.55},   # Partial recovery
            {"cycle": 1500, "strategy_idx": 1, "new_rate": 0.45},  # Medium degrades
            {"cycle": 2000, "strategy_idx": 0, "new_rate": 0.35},  # High crashes again
            {"cycle": 2500, "strategy_idx": 0, "new_rate": 0.60},  # Partial recovery
        ]
    
    print("REGIME SHIFT SCHEDULE:")
    print("-" * 70)
    for rs in regime_shifts:
        s = strategies[rs["strategy_idx"]]
        print(f"  Cycle {rs['cycle']:4d}: {s['name']} → {rs['new_rate']:.0%}")
    print()
    
    # Initialize
    allocator = get_capital_allocator(CapitalConfig())
    shock_absorber = get_failure_shock_absorber()
    
    # History tracking
    history = {
        "capital": [],
        "allocations": [],
        "ema": {s["name"]: [] for s in strategies},
        "drawdown": [],
        "temperature": [],
        "events": []
    }
    
    # EMA history for degradation detection
    ema_full_history = {s["name"]: [] for s in strategies}
    
    # Regime shift index
    rs_idx = 0
    
    # Run cycles
    for cycle in range(num_cycles):
        # Check regime shifts
        if rs_idx < len(regime_shifts) and cycle == regime_shifts[rs_idx]["cycle"]:
            rs = regime_shifts[rs_idx]
            s = strategies[rs["strategy_idx"]]
            old_rate = s["current_success_rate"]
            s["current_success_rate"] = rs["new_rate"]
            
            event = f"REGIME SHIFT: {s['name']} {old_rate:.0%} → {rs['new_rate']:.0%}"
            history["events"].append({"cycle": cycle, "event": event})
            print(f"\n  [Cycle {cycle}] {event}\n")
            rs_idx += 1
        
        # Random degradation (stochastic alpha fluctuation)
        if random_degradation and cycle > 0 and cycle % 100 == 0:
            for s in strategies:
                # Small random fluctuation (±5%)
                fluctuation = random.gauss(0, 0.02)
                new_rate = s["current_success_rate"] + fluctuation
                new_rate = max(0.25, min(0.95, new_rate))  # Clamp
                s["current_success_rate"] = new_rate
        
        # Build strategy assets
        assets = []
        for s in strategies:
            ema = shock_absorber.get_ema_success(str(s["id"]))
            ema_full_history[s["name"]].append(ema)
            
            # Calculate EMA drop penalty
            ema_drop_penalty = False
            if len(ema_full_history[s["name"]]) > 50:
                ema_50_ago = ema_full_history[s["name"]][-50]
                ema_drop = (ema_50_ago - ema) / ema_50_ago if ema_50_ago > 0 else 0
                if ema_drop > 0.15:
                    ema_drop_penalty = True
            
            asset = StrategyAsset(
                strategy_id=s["id"],
                name=s["name"],
                ema_success=ema,
                payoff=s["payoff"],
                cost=s["cost"],
                variance_proxy=s["variance"]
            )
            asset.ema_drop_penalty = ema_drop_penalty
            assets.append(asset)
        
        # Allocate capital
        allocations = allocator.allocate(assets)
        
        # Execute strategies
        for s in strategies:
            capital = allocations.get(s["id"], 0)
            if capital > 0:
                success = random.random() < s["current_success_rate"]
                shock_absorber.record_outcome(str(s["id"]), success)
                allocator.record_outcome(s["id"], capital, success, variance_proxy=s["variance"])
        
        # Record history
        history["capital"].append(allocator.capital)
        history["drawdown"].append(allocator.drawdown)
        
        # Temperature tracking
        if allocator.drawdown > allocator.config.drawdown_threshold_for_crisis:
            history["temperature"].append(allocator.config.softmax_temperature_crisis)
        else:
            history["temperature"].append(allocator.config.softmax_temperature_base)
        
        # Allocations (normalized)
        total_alloc = sum(allocations.values()) if allocations else 1
        alloc_dist = {s["name"]: allocations.get(s["id"], 0) / total_alloc for s in strategies}
        history["allocations"].append(alloc_dist)
        
        # EMA tracking
        for s in strategies:
            ema = shock_absorber.get_ema_success(str(s["id"]))
            history["ema"][s["name"]].append(ema)
        
        # Bankruptcy check
        if allocator.is_bankrupt:
            event = "BANKRUPTCY"
            history["events"].append({"cycle": cycle, "event": event})
            print(f"\n  !!! BANKRUPTCY at cycle {cycle} !!!\n")
            break
        
        # Progress report
        if (cycle + 1) % 500 == 0:
            stats = allocator.get_statistics()
            print(f"  Cycle {cycle+1}: capital={stats['capital']:.2f}, "
                  f"return={stats['total_return_pct']:+.1f}%, "
                  f"DD={stats['drawdown_pct']:.1f}%")
    
    # Final statistics
    print()
    print("=" * 70)
    print("MONTE-CARLO RESULTS")
    print("=" * 70)
    
    stats = allocator.get_statistics()
    
    print(f"Final Capital:      {stats['capital']:.2f}")
    print(f"Total Return:       {stats['total_return_pct']:+.2f}%")
    print(f"Peak Capital:       {stats['peak_capital']:.2f}")
    print(f"Max Drawdown:       {stats['drawdown_pct']:.2f}%")
    print(f"Cycles Completed:   {len(history['capital'])}")
    print(f"Bankrupt:           {stats['is_bankrupt']}")
    print()
    
    # Allocation analysis
    print("ALLOCATION ANALYSIS:")
    print("-" * 70)
    final_100 = history["allocations"][-100:] if len(history["allocations"]) >= 100 else history["allocations"]
    avg_allocs = {}
    for s in strategies:
        name = s["name"]
        avg_allocs[name] = sum(a.get(name, 0) for a in final_100) / len(final_100)
    
    for name, alloc in sorted(avg_allocs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {alloc:.1%}")
    print()
    
    # Drawdown distribution
    print("DRAWDOWN ANALYSIS:")
    print("-" * 70)
    dd_values = history["drawdown"]
    print(f"  Max:               {max(dd_values)*100:.1f}%")
    print(f"  Mean:              {sum(dd_values)/len(dd_values)*100:.1f}%")
    print(f"  Time in crisis (DD>5%): {sum(1 for d in dd_values if d > 0.05)/len(dd_values)*100:.1f}%")
    print()
    
    # Recovery analysis
    print("RECOVERY ANALYSIS:")
    print("-" * 70)
    
    # Find all drawdown peaks and recoveries
    capital = history["capital"]
    peak = capital[0]
    drawdown_events = []
    
    for i, cap in enumerate(capital):
        if cap > peak:
            peak = cap
        dd = (peak - cap) / peak if peak > 0 else 0
        
        if dd > 0.05 and (i == 0 or (peak - capital[i-1])/peak <= 0.05):
            # Entered drawdown
            drawdown_events.append({"start": i, "peak": peak})
    
    if drawdown_events:
        print(f"  Drawdown events (DD>5%): {len(drawdown_events)}")
    
    # EMA evolution
    print()
    print("EMA EVOLUTION:")
    print("-" * 70)
    for s in strategies:
        name = s["name"]
        ema_vals = history["ema"][name]
        print(f"  {name:20s}: start={ema_vals[0]:.3f}, end={ema_vals[-1]:.3f}, "
              f"min={min(ema_vals):.3f}, max={max(ema_vals):.3f}")
    
    print()
    
    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    issues = []
    passed = True
    
    if stats['is_bankrupt']:
        passed = False
        issues.append("❌ BANKRUPTCY")
    else:
        issues.append("✅ Survived")
    
    if stats['drawdown_pct'] > 25:
        passed = False
        issues.append(f"❌ Deep drawdown: {stats['drawdown_pct']:.1f}%")
    elif stats['drawdown_pct'] > 15:
        issues.append(f"⚠ High drawdown: {stats['drawdown_pct']:.1f}%")
    else:
        issues.append(f"✅ Controlled drawdown: {stats['drawdown_pct']:.1f}%")
    
    if stats['total_return_pct'] > 0:
        issues.append(f"✅ Positive return: {stats['total_return_pct']:.1f}%")
    elif stats['total_return_pct'] > -10:
        issues.append(f"⚠ Negative return: {stats['total_return_pct']:.1f}%")
    else:
        passed = False
        issues.append(f"❌ Large loss: {stats['total_return_pct']:.1f}%")
    
    max_conc = max(avg_allocs.values())
    if max_conc > 0.6:
        issues.append(f"❌ High concentration: {max_conc:.1%}")
    elif max_conc > 0.5:
        issues.append(f"⚠ Elevated concentration: {max_conc:.1%}")
    else:
        issues.append(f"✅ Reasonable concentration: {max_conc:.1%}")
    
    for issue in issues:
        print(f"  {issue}")
    
    print()
    if passed:
        print("  🎉 MONTE-CARLO VALIDATION: PASS")
    else:
        print("  ⚠️  MONTE-CARLO VALIDATION: NEEDS ATTENTION")
    
    return {
        "passed": passed,
        "issues": issues,
        "summary": stats,
        "history": {
            "capital": history["capital"],
            "drawdown": history["drawdown"],
            "allocations": history["allocations"],
            "ema": history["ema"],
            "temperature": history["temperature"],
            "events": history["events"]
        },
        "avg_allocations": avg_allocs
    }


if __name__ == "__main__":
    result = run_monte_carlo(3000, seed=42)
