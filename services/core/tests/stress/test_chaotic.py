"""
STAGE 4 CHAOTIC STRESS TEST
===========================

Adversarial test with random regime shifts and 3x noise.

Tests:
- System resilience under chaotic conditions
- Alert system triggering
- Recovery from cascading failures
- Extreme market conditions

Author: AI-OS Team
Date: 2026-02-21
"""
import sys
sys.path.insert(0, '/app')

from uuid import uuid4
import random
from typing import Dict, List

from autonomy.capital_engine import (
    CapitalAllocator,
    CapitalConfig,
    StrategyAsset,
    get_capital_allocator,
    reset_capital_allocator,
    get_alert_engine
)
from autonomy.stability_guards import (
    get_failure_shock_absorber,
    reset_all_guards
)


def run_chaotic_test(
    num_cycles: int = 2000,
    seed: int = 12345,
    regime_shift_probability: float = 0.005,
    noise_multiplier: float = 3.0,
    min_success_rate: float = 0.20,
    max_success_rate: float = 0.95
) -> Dict:
    """
    Run chaotic stress test with random regime shifts and elevated noise.
    
    Args:
        num_cycles: Total cycles (default 2000)
        seed: Random seed
        regime_shift_probability: Probability of random regime shift per cycle
        noise_multiplier: Multiply base noise by this factor (3.0 = 3x normal)
        min_success_rate: Minimum allowed success rate
        max_success_rate: Maximum allowed success rate
    
    Returns:
        Dict with results
    """
    random.seed(seed)
    
    print("=" * 70)
    print("STAGE 4 CHAOTIC STRESS TEST")
    print("=" * 70)
    print(f"Cycles: {num_cycles}")
    print(f"Seed: {seed}")
    print(f"Regime shift probability: {regime_shift_probability:.1%}")
    print(f"Noise multiplier: {noise_multiplier}x")
    print()
    
    reset_all_guards()
    reset_capital_allocator()
    
    strategies = [
        {
            "id": uuid4(),
            "name": "Alpha_Seeker",
            "base_success_rate": 0.80,
            "current_success_rate": 0.80,
            "payoff": 0.012,
            "cost": 0.002,
            "variance": 0.06
        },
        {
            "id": uuid4(),
            "name": "Beta_Harvester",
            "base_success_rate": 0.70,
            "current_success_rate": 0.70,
            "payoff": 0.01,
            "cost": 0.002,
            "variance": 0.10
        },
        {
            "id": uuid4(),
            "name": "Gamma_Trader",
            "base_success_rate": 0.60,
            "current_success_rate": 0.60,
            "payoff": 0.008,
            "cost": 0.002,
            "variance": 0.12
        },
        {
            "id": uuid4(),
            "name": "Delta_Explorer",
            "base_success_rate": 0.50,
            "current_success_rate": 0.50,
            "payoff": 0.015,
            "cost": 0.003,
            "variance": 0.18
        }
    ]
    
    allocator = get_capital_allocator(CapitalConfig())
    shock_absorber = get_failure_shock_absorber()
    alert_engine = get_alert_engine()
    
    history = {
        "capital": [],
        "drawdown": [],
        "allocations": [],
        "ema": {s["name"]: [] for s in strategies},
        "alerts": [],
        "regime_shifts": [],
        "crisis_cycles": 0
    }
    
    ema_full_history = {s["name"]: [] for s in strategies}
    
    for cycle in range(num_cycles):
        regime_shifts_this_cycle = []
        
        for s in strategies:
            if random.random() < regime_shift_probability:
                old_rate = s["current_success_rate"]
                
                shift_magnitude = random.choice([
                    random.gauss(0, 0.15),
                    random.gauss(0, 0.25),
                    random.gauss(-0.20, 0.10),
                    random.gauss(0.15, 0.10)
                ])
                
                new_rate = s["current_success_rate"] + shift_magnitude
                new_rate = max(min_success_rate, min(max_success_rate, new_rate))
                s["current_success_rate"] = new_rate
                
                regime_shifts_this_cycle.append({
                    "strategy": s["name"],
                    "old_rate": old_rate,
                    "new_rate": new_rate
                })
                history["regime_shifts"].append({
                    "cycle": cycle,
                    "strategy": s["name"],
                    "old_rate": old_rate,
                    "new_rate": new_rate
                })
        
        if regime_shifts_this_cycle:
            print(f"\n  [Cycle {cycle}] REGIME SHIFTS:")
            for rs in regime_shifts_this_cycle:
                print(f"    {rs['strategy']}: {rs['old_rate']:.0%} → {rs['new_rate']:.0%}")
        
        for s in strategies:
            base_fluctuation = random.gauss(0, 0.02)
            noise = base_fluctuation * noise_multiplier
            new_rate = s["current_success_rate"] + noise
            new_rate = max(min_success_rate, min(max_success_rate, new_rate))
            s["current_success_rate"] = new_rate
        
        assets = []
        for s in strategies:
            ema = shock_absorber.get_ema_success(str(s["id"]))
            ema_full_history[s["name"]].append(ema)
            
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
        
        allocations = allocator.allocate(assets)
        
        for s in strategies:
            capital = allocations.get(s["id"], 0)
            if capital > 0:
                success = random.random() < s["current_success_rate"]
                shock_absorber.record_outcome(str(s["id"]), success)
                allocator.record_outcome(s["id"], capital, success, variance_proxy=s["variance"])
        
        alerts = alert_engine.check_all(cycle)
        if alerts:
            for a in alerts:
                history["alerts"].append({
                    "cycle": cycle,
                    "rule": a.rule_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "value": a.value
                })
        
        history["capital"].append(allocator.capital)
        history["drawdown"].append(allocator.drawdown)
        
        if allocator.drawdown > allocator.config.drawdown_threshold_for_crisis:
            history["crisis_cycles"] += 1
        
        total_alloc = sum(allocations.values()) if allocations else 1
        alloc_dist = {s["name"]: allocations.get(s["id"], 0) / total_alloc for s in strategies}
        history["allocations"].append(alloc_dist)
        
        for s in strategies:
            ema = shock_absorber.get_ema_success(str(s["id"]))
            history["ema"][s["name"]].append(ema)
        
        if allocator.is_bankrupt:
            print(f"\n  !!! BANKRUPTCY at cycle {cycle} !!!\n")
            break
        
        if (cycle + 1) % 400 == 0:
            stats = allocator.get_statistics()
            num_alerts = len([a for a in history["alerts"] if a["cycle"] > cycle - 400])
            print(f"  Cycle {cycle+1}: capital={stats['capital']:.2f}, "
                  f"return={stats['total_return_pct']:+.1f}%, "
                  f"DD={stats['drawdown_pct']:.1f}%, "
                  f"alerts={num_alerts}")
    
    print()
    print("=" * 70)
    print("CHAOTIC STRESS TEST RESULTS")
    print("=" * 70)
    
    stats = allocator.get_statistics()
    
    print(f"Final Capital:         {stats['capital']:.2f}")
    print(f"Total Return:          {stats['total_return_pct']:+.2f}%")
    print(f"Peak Capital:          {stats['peak_capital']:.2f}")
    print(f"Max Drawdown:          {stats['drawdown_pct']:.2f}%")
    print(f"Cycles Completed:      {len(history['capital'])}")
    print(f"Cycles in Crisis:      {history['crisis_cycles']} ({history['crisis_cycles']/len(history['capital'])*100:.1f}%)")
    print(f"Bankrupt:              {stats['is_bankrupt']}")
    print(f"Total Regime Shifts:   {len(history['regime_shifts'])}")
    print(f"Total Alerts:          {len(history['alerts'])}")
    print()
    
    if history["alerts"]:
        print("ALERT SUMMARY:")
        print("-" * 70)
        alert_counts = {}
        for a in history["alerts"]:
            rule = a["rule"]
            if rule not in alert_counts:
                alert_counts[rule] = 0
            alert_counts[rule] += 1
        
        for rule, count in sorted(alert_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {rule:30s}: {count}")
        print()
    
    final_100 = history["allocations"][-100:] if len(history["allocations"]) >= 100 else history["allocations"]
    avg_allocs = {}
    for s in strategies:
        name = s["name"]
        avg_allocs[name] = sum(a.get(name, 0) for a in final_100) / len(final_100)
    
    print("FINAL ALLOCATION DISTRIBUTION:")
    print("-" * 70)
    for name, alloc in sorted(avg_allocs.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:20s}: {alloc:.1%}")
    print()
    
    dd_values = history["drawdown"]
    print("DRAWDOWN ANALYSIS:")
    print("-" * 70)
    print(f"  Max:                   {max(dd_values)*100:.1f}%")
    print(f"  Mean:                  {sum(dd_values)/len(dd_values)*100:.1f}%")
    print(f"  Time > 10% DD:         {sum(1 for d in dd_values if d > 0.10)/len(dd_values)*100:.1f}%")
    print(f"  Time > 20% DD:         {sum(1 for d in dd_values if d > 0.20)/len(dd_values)*100:.1f}%")
    print()
    
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    issues = []
    passed = True
    
    if stats['is_bankrupt']:
        passed = False
        issues.append("❌ BANKRUPTCY")
    else:
        issues.append("✅ Survived chaotic conditions")
    
    if stats['drawdown_pct'] > 35:
        passed = False
        issues.append(f"❌ Extreme drawdown: {stats['drawdown_pct']:.1f}%")
    elif stats['drawdown_pct'] > 25:
        issues.append(f"⚠ Deep drawdown: {stats['drawdown_pct']:.1f}%")
    else:
        issues.append(f"✅ Controlled drawdown: {stats['drawdown_pct']:.1f}%")
    
    crisis_ratio = history['crisis_cycles'] / len(history['capital'])
    if crisis_ratio > 0.5:
        issues.append(f"⚠ Extended crisis mode: {crisis_ratio:.0%} of time")
    else:
        issues.append(f"✅ Limited crisis exposure: {crisis_ratio:.0%} of time")
    
    critical_alerts = [a for a in history["alerts"] if a["severity"] == "critical"]
    if len(critical_alerts) > 10:
        issues.append(f"⚠ Many critical alerts: {len(critical_alerts)}")
    else:
        issues.append(f"✅ Critical alerts under control: {len(critical_alerts)}")
    
    max_conc = max(avg_allocs.values())
    if max_conc > 0.7:
        issues.append(f"❌ Extreme concentration: {max_conc:.1%}")
    elif max_conc > 0.5:
        issues.append(f"⚠ High concentration: {max_conc:.1%}")
    else:
        issues.append(f"✅ Reasonable concentration: {max_conc:.1%}")
    
    for issue in issues:
        print(f"  {issue}")
    
    print()
    if passed:
        print("  🎉 CHAOTIC STRESS TEST: PASS")
    else:
        print("  ⚠️  CHAOTIC STRESS TEST: NEEDS ATTENTION")
    
    return {
        "passed": passed,
        "issues": issues,
        "summary": stats,
        "chaotic_metrics": {
            "total_regime_shifts": len(history["regime_shifts"]),
            "total_alerts": len(history["alerts"]),
            "crisis_cycles": history["crisis_cycles"],
            "crisis_ratio": crisis_ratio
        },
        "history": {
            "capital": history["capital"],
            "drawdown": history["drawdown"],
            "alerts": history["alerts"][-50:],
            "regime_shifts": history["regime_shifts"][-20:]
        },
        "avg_allocations": avg_allocs
    }


if __name__ == "__main__":
    result = run_chaotic_test(2000, seed=12345)
