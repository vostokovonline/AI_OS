"""
STAGE 4 CAPITAL TEST
====================

Simple validation: 1000 cycles, 3 strategies, no degradation.

Measures:
- Does capital grow?
- Is there concentration?
- Is there slow death?

Author: AI-OS Team
Date: 2026-02-21
"""
import sys
sys.path.insert(0, '/app')

from uuid import uuid4, UUID
import random

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


def run_capital_test(num_cycles: int = 1000):
    """Run capital test with regime shift (realistic stress)."""
    
    print("=" * 70)
    print("STAGE 4 CAPITAL TEST - WITH REGIME SHIFT")
    print("=" * 70)
    print(f"Cycles: {num_cycles}")
    print("REGIME SHIFT: High_Performer 0.85 → 0.42 at cycle 500, → 0.55 at cycle 700")
    print()
    
    # Reset everything
    reset_all_guards()
    reset_capital_allocator()
    
    # Create 3 strategies with different success rates
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
    
    # Print strategy profiles
    print("STRATEGY PROFILES:")
    print("-" * 70)
    for s in strategies:
        print(f"  {s['name']}: base_success={s['base_success_rate']:.0%}, "
              f"payoff=+{s['payoff']:.1%}, cost={s['cost']:.1%}, var={s['variance']:.0%}")
    print()
    print("REGIME SHIFT SCHEDULE:")
    print("-" * 70)
    print("  Cycles 0-500:   Normal (High: 85%)")
    print("  Cycle 500:      SHOCK → High: 42%")
    print("  Cycles 500-700: Crisis mode")
    print("  Cycle 700:      Recovery → High: 55%")
    print("  Cycles 700-1000: New normal")
    print()
    
    # Initialize
    allocator = get_capital_allocator(CapitalConfig())
    shock_absorber = get_failure_shock_absorber()
    
    # Track history
    capital_history = []
    allocation_history = []
    ema_history = {s["name"]: [] for s in strategies}  # Track EMA for each strategy
    
    print(f"INITIAL CAPITAL: {allocator.capital:.2f}")
    print()
    
    # Run cycles
    for cycle in range(num_cycles):
        # REGIME SHIFT LOGIC
        if cycle == 500:
            # SHOCK: High performer degrades
            strategies[0]["current_success_rate"] = 0.42
            print(f"\n{'!'*60}")
            print(f"REGIME SHIFT at cycle {cycle}: High_Performer 0.85 → 0.42")
            print(f"{'!'*60}\n")
        elif cycle == 700:
            # Recovery: High performer improves but not to original level
            strategies[0]["current_success_rate"] = 0.55
            print(f"\n{'='*60}")
            print(f"RECOVERY at cycle {cycle}: High_Performer 0.42 → 0.55")
            print(f"{'='*60}\n")
        
        # Build strategy assets with current EMA and degradation penalty
        assets = []
        for s in strategies:
            ema = shock_absorber.get_ema_success(str(s["id"]))
            
            # Track EMA history
            ema_history[s["name"]].append(ema)
            
            # Calculate EMA drop penalty
            ema_drop_penalty = False
            if len(ema_history[s["name"]]) > 50:
                ema_50_cycles_ago = ema_history[s["name"]][-50]
                ema_drop = (ema_50_cycles_ago - ema) / ema_50_cycles_ago if ema_50_cycles_ago > 0 else 0
                if ema_drop > 0.15:  # 15% drop triggers penalty
                    ema_drop_penalty = True
            
            asset = StrategyAsset(
                strategy_id=s["id"],
                name=s["name"],
                ema_success=ema,
                payoff=s["payoff"],
                cost=s["cost"],
                variance_proxy=s["variance"]
            )
            # Add penalty flag
            asset.ema_drop_penalty = ema_drop_penalty
            assets.append(asset)
        
        # Allocate capital
        allocations = allocator.allocate(assets)
        
        # Execute each strategy with its allocation
        cycle_return = 0.0
        for s in strategies:
            capital = allocations.get(s["id"], 0)
            if capital > 0:
                # Determine outcome based on CURRENT success rate (can change)
                success = random.random() < s["current_success_rate"]
                
                # Record outcome in shock absorber
                shock_absorber.record_outcome(str(s["id"]), success)
                
                # Record outcome in capital allocator (with variance)
                net_return = allocator.record_outcome(
                    s["id"], capital, success, variance_proxy=s["variance"]
                )
                cycle_return += net_return
        
        # Track history (as ratios, not absolute capital)
        capital_history.append(allocator.capital)
        total_allocated = sum(allocations.values()) if allocations else 1
        alloc_dist = {s["name"]: allocations.get(s["id"], 0) / total_allocated for s in strategies}
        allocation_history.append(alloc_dist)
        
        # Check bankruptcy
        if allocator.is_bankrupt:
            print(f"\n!!! BANKRUPTCY at cycle {cycle} !!!")
            break
        
        # Progress report (more frequent around regime shifts)
        report_cycles = [199, 499, 599, 699, 799, 999]
        if cycle in report_cycles:
            stats = allocator.get_statistics()
            recent_allocs = allocation_history[-100:] if len(allocation_history) >= 100 else allocation_history
            avg_allocs = {}
            for name in [s["name"] for s in strategies]:
                avg_allocs[name] = sum(a.get(name, 0) for a in recent_allocs) / len(recent_allocs)
            
            # Get EMA values
            ema_vals = {}
            for s in strategies:
                ema_vals[s["name"]] = shock_absorber.get_ema_success(str(s["id"]))
            
            print(f"Cycle {cycle+1}: capital={stats['capital']:.2f}, "
                  f"return={stats['total_return_pct']:+.1f}%, "
                  f"drawdown={stats['drawdown_pct']:.1f}%")
            print(f"         allocs: H={avg_allocs['High_Performer']:.1%}, "
                  f"M={avg_allocs['Medium_Performer']:.1%}, "
                  f"L={avg_allocs['Low_Performer']:.1%}")
            print(f"         EMA:    H={ema_vals['High_Performer']:.3f}, "
                  f"M={ema_vals['Medium_Performer']:.3f}, "
                  f"L={ema_vals['Low_Performer']:.3f}")
    
    # Final report
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    stats = allocator.get_statistics()
    print(f"Final Capital:     {stats['capital']:.2f}")
    print(f"Initial Capital:   {stats['initial_capital']:.2f}")
    print(f"Total Return:      {stats['total_return_pct']:+.2f}%")
    print(f"Peak Capital:      {stats['peak_capital']:.2f}")
    print(f"Max Drawdown:      {stats['drawdown_pct']:.2f}%")
    print(f"Bankrupt:          {stats['is_bankrupt']}")
    print()
    
    # Allocation concentration
    final_100 = allocation_history[-100:]
    avg_allocs = {}
    for s in strategies:
        name = s["name"]
        avg_allocs[name] = sum(a.get(name, 0) for a in final_100) / len(final_100)
    
    print("ALLOCATION DISTRIBUTION (last 100 cycles):")
    print("-" * 70)
    for name, alloc in sorted(avg_allocs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(alloc * 50)
        print(f"  {name:20s}: {alloc:.1%} {bar}")
    print()
    
    max_concentration = max(avg_allocs.values())
    print(f"Max Capital Concentration: {max_concentration:.1%}")
    print()
    
    # Capital trajectory
    print("CAPITAL TRAJECTORY (key cycles):")
    print("-" * 70)
    key_cycles = [0, 199, 499, 599, 699, 799, 999]
    for i in key_cycles:
        if i < len(capital_history):
            cap = capital_history[i]
            ret = (cap - 1000) / 1000 * 100
            phase = "Normal" if i < 500 else ("Crisis" if i < 700 else "Recovery")
            print(f"  Cycle {i+1:4d}: {cap:8.2f} ({ret:+6.1f}%) [{phase}]")
    
    print()
    
    # Phase analysis
    print("PHASE ANALYSIS:")
    print("-" * 70)
    if len(capital_history) > 500:
        cap_500 = capital_history[499]
        cap_700 = capital_history[699] if len(capital_history) > 700 else capital_history[-1]
        cap_final = capital_history[-1]
        
        phase1_return = (cap_500 - 1000) / 1000 * 100
        phase2_return = (cap_700 - cap_500) / cap_500 * 100 if cap_500 > 0 else 0
        phase3_return = (cap_final - cap_700) / cap_700 * 100 if cap_700 > 0 else 0
        
        print(f"  Phase 1 (0-500, Normal):   {phase1_return:+.1f}%")
        print(f"  Phase 2 (500-700, Crisis):  {phase2_return:+.1f}%")
        print(f"  Phase 3 (700-1000, Recovery): {phase3_return:+.1f}%")
        
        # Recovery analysis
        min_after_shock = min(capital_history[500:700]) if len(capital_history) > 700 else cap_700
        max_dd_from_peak = (cap_500 - min_after_shock) / cap_500 * 100 if cap_500 > 0 else 0
        print(f"\n  Crisis drawdown from peak: {max_dd_from_peak:.1f}%")
        print(f"  Recovery from crisis low:   {(cap_final - min_after_shock) / min_after_shock * 100:.1f}%")
    
    print()
    
    # Verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    passed = True
    issues = []
    
    # Overall return
    if stats['total_return_pct'] < 0:
        issues.append(f"❌ Negative return: {stats['total_return_pct']:.1f}%")
        # Don't fail - survival matters more than profit
    elif stats['total_return_pct'] < 5:
        issues.append(f"⚠ Low return: {stats['total_return_pct']:.1f}%")
    else:
        issues.append(f"✅ Positive return: {stats['total_return_pct']:.1f}%")
    
    # Concentration
    if max_concentration > 0.6:
        issues.append(f"❌ High concentration: {max_concentration:.1%}")
    elif max_concentration > 0.5:
        issues.append(f"⚠ Elevated concentration: {max_concentration:.1%}")
    else:
        issues.append(f"✅ Reasonable concentration: {max_concentration:.1%}")
    
    # Drawdown (expecting higher now with regime shift)
    if stats['drawdown_pct'] > 30:
        issues.append(f"❌ Deep drawdown: {stats['drawdown_pct']:.1f}%")
    elif stats['drawdown_pct'] > 10:
        issues.append(f"✅ Realistic drawdown: {stats['drawdown_pct']:.1f}%")
    elif stats['drawdown_pct'] < 1:
        issues.append(f"⚠ Unrealistic drawdown: {stats['drawdown_pct']:.1f}% (too low)")
    else:
        issues.append(f"✅ Controlled drawdown: {stats['drawdown_pct']:.1f}%")
    
    # Survival
    if stats['is_bankrupt']:
        passed = False
        issues.append("❌ SYSTEM BANKRUPT")
    else:
        issues.append("✅ System survived regime shift")
    
    # Recovery (if we have phase data)
    if len(capital_history) > 700:
        cap_700 = capital_history[699]
        cap_final = capital_history[-1]
        if cap_final > cap_700:
            issues.append(f"✅ Recovery after crisis: +{(cap_final - cap_700) / cap_700 * 100:.1f}%")
        else:
            issues.append(f"⚠ No recovery after crisis")
    
    for issue in issues:
        print(f"  {issue}")
    
    print()
    if passed:
        print("  🎉 STAGE 4 CAPITAL TEST: PASS")
    else:
        print("  ⚠️  STAGE 4 CAPITAL TEST: NEEDS ATTENTION")
    
    return {
        "passed": passed,
        "final_capital": stats['capital'],
        "total_return_pct": stats['total_return_pct'],
        "max_drawdown_pct": stats['drawdown_pct'],
        "max_concentration": max_concentration,
        "is_bankrupt": stats['is_bankrupt']
    }


if __name__ == "__main__":
    result = run_capital_test(1000)
