"""
Comparison runner: Simple vs Stress environments
"""
import numpy as np
from typing import List, Tuple


def run_comparison():
    """Run comparison between environments."""
    
    print("=" * 70)
    print("Phase 18 MVP v2 - Trajectory-based V-field Comparison")
    print("=" * 70)
    
    # Import
    from main import MVPRunner
    
    results = []
    
    # Simple environment
    print("\n" + "=" * 70)
    print("SIMPLE ENVIRONMENT (Normal dynamics)")
    print("=" * 70)
    
    runner_simple = MVPRunner(env_type='simple')
    stats_simple = runner_simple.run_episode(max_steps=200, verbose=True)
    
    print(f"\nSimple Results:")
    print(f"  Total reward: {stats_simple['total_reward']:.2f}")
    print(f"  V mean: {stats_simple['V_mean']:.3f}")
    print(f"  V min: {stats_simple['V_min']:.3f}")
    print(f"  V trend: {stats_simple['V_trend']:.5f}")
    print(f"  Healthy: {stats_simple['healthy_ratio']:.0%}")
    print(f"  Silent collapse: {stats_simple['silent_collapse_events']}")
    print(f"  Attractor trap: {stats_simple['avg_attractor_trap']:.3f}")
    
    results.append(('simple', stats_simple))
    
    # Stress environment
    print("\n" + "=" * 70)
    print("STRESS ENVIRONMENT (Collapse-inducing dynamics)")
    print("=" * 70)
    
    runner_stress = MVPRunner(env_type='stress')
    stats_stress = runner_stress.run_episode(max_steps=400, verbose=True)
    
    print(f"\nStress Results:")
    print(f"  Total reward: {stats_stress['total_reward']:.2f}")
    print(f"  V mean: {stats_stress['V_mean']:.3f}")
    print(f"  V min: {stats_stress['V_min']:.3f}")
    print(f"  V trend: {stats_stress['V_trend']:.5f}")
    print(f"  Healthy: {stats_stress['healthy_ratio']:.0%}")
    print(f"  Warning: {stats_stress['warning_ratio']:.0%}")
    print(f"  Critical: {stats_stress['critical_ratio']:.0%}")
    print(f"  Silent collapse: {stats_stress['silent_collapse_events']}")
    print(f"  Attractor trap: {stats_stress['avg_attractor_trap']:.3f}")
    
    results.append(('stress', stats_stress))
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    simple = results[0][1]
    stress = results[1][1]
    
    print(f"\n{'Metric':<25} {'Simple':<15} {'Stress':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Total Reward':<25} {simple['total_reward']:<15.2f} {stress['total_reward']:<15.2f} {stress['total_reward'] - simple['total_reward']:<15.2f}")
    print(f"{'V Mean':<25} {simple['V_mean']:<15.3f} {stress['V_mean']:<15.3f} {stress['V_mean'] - simple['V_mean']:<15.3f}")
    print(f"{'V Min':<25} {simple['V_min']:<15.3f} {stress['V_min']:<15.3f} {stress['V_min'] - simple['V_min']:<15.3f}")
    print(f"{'V Trend':<25} {simple['V_trend']:<15.5f} {stress['V_trend']:<15.5f} {stress['V_trend'] - simple['V_trend']:<15.5f}")
    print(f"{'Healthy Ratio':<25} {simple['healthy_ratio']:<15.1%} {stress['healthy_ratio']:<15.1%} {stress['healthy_ratio'] - simple['healthy_ratio']:<15.1%}")
    print(f"{'Silent Collapse':<25} {simple['silent_collapse_events']:<15} {stress['silent_collapse_events']:<15} {stress['silent_collapse_events'] - simple['silent_collapse_events']:<15}")
    print(f"{'Attractor Trap':<25} {simple['avg_attractor_trap']:<15.3f} {stress['avg_attractor_trap']:<15.3f} {stress['avg_attractor_trap'] - simple['avg_attractor_trap']:<15.3f}")
    
    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    
    if stress['V_min'] < simple['V_min']:
        print("\n✓ V-field detected collapse in stress environment")
        print("  V dropped lower in stress than simple")
    else:
        print("\n? V-field did not detect collapse difference")
    
    if stress['avg_attractor_trap'] > simple['avg_attractor_trap']:
        print("\n✓ V-field detected attractor trapping")
        print("  Attractor trap higher in stress environment")
    
    if stress['V_trend'] < simple['V_trend']:
        print("\n✓ V-field detected degradation trend")
        print("  V trend more negative in stress")
    
    return results


if __name__ == '__main__':
    run_comparison()