"""
Hybrid Policy + Real Utility Benchmark
==================================

Test hybrid policy:
- 70% Greedy (top utility)
- 20% Under-selected (fairness)
- 10% Random (exploration)

Compare cumulative utility over 100 cycles:
- Greedy
- RoundRobin
- Hybrid

Run:
    docker exec ns_core python /app/tests/integration/test_hybrid_policy.py
"""
import asyncio
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple
from uuid import uuid4

sys.path.insert(0, '/app')


# ============================================================================
# SECTION 1: DATA STRUCTURES
# ============================================================================

@dataclass
class CycleResult:
    """Result of one cycle"""
    cycle: int
    selected: List[Dict]
    total_utility: float
    latency_ms: float
    unique_selected: int


# ============================================================================
# SECTION 2: INTENT GENERATION
# ============================================================================

def generate_intents(n: int = 1000, seed: int = 42) -> List[Dict]:
    """Generate realistic intents"""
    random.seed(seed)
    
    intents = []
    
    # Distribution: 40% low, 30% medium, 20% high, 10% adversarial
    for i in range(int(n * 0.4)):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.1, 0.3),
            "cost": random.uniform(0.1, 0.3),
            "risk": random.uniform(0.1, 0.3),
            "type": "low"
        })
    
    for i in range(int(n * 0.3)):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.4, 0.6),
            "cost": random.uniform(0.3, 0.6),
            "risk": random.uniform(0.2, 0.5),
            "type": "medium"
        })
    
    for i in range(int(n * 0.2)):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.7, 1.0),
            "cost": random.uniform(0.6, 1.0),
            "risk": random.uniform(0.3, 0.7),
            "type": "high"
        })
    
    for i in range(n - len(intents)):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.1, 0.4),
            "cost": random.uniform(0.8, 1.0),
            "risk": random.uniform(0.6, 0.9),
            "type": "adversarial"
        })
    
    random.shuffle(intents)
    return intents


# ============================================================================
# SECTION 3: POLICIES
# ============================================================================

def select_greedy(intents: List[Dict], budget: int) -> List[Dict]:
    """Pure greedy - max utility/cost"""
    scored = [(i, intents[i]["utility"] / (intents[i]["cost"] + 0.1)) for i in range(len(intents))]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    total_cost = 0
    for idx, _ in scored:
        if total_cost + intents[idx]["cost"] <= budget:
            selected.append(intents[idx])
            total_cost += intents[idx]["cost"]
    
    return selected


def select_round_robin(intents: List[Dict], budget: int, selection_counts: List[int]) -> List[Dict]:
    """Round robin - prioritize least selected"""
    scored = [(i, selection_counts[i]) for i in range(len(intents))]
    scored.sort(key=lambda x: x[1])  # Least selected first
    
    selected = []
    total_cost = 0
    for idx, _ in scored:
        if total_cost + intents[idx]["cost"] <= budget:
            selected.append(intents[idx])
            total_cost += intents[idx]["cost"]
    
    return selected


def select_hybrid(
    intents: List[Dict], 
    budget: int, 
    selection_counts: List[int],
    greedy_ratio: float = 0.7,
    fairness_ratio: float = 0.2,
    random_ratio: float = 0.1
) -> List[Dict]:
    """
    Hybrid policy:
    - 70% budget → Greedy (top utility)
    - 20% budget → Under-selected (fairness)
    - 10% budget → Random (exploration)
    """
    selected = []
    total_cost = 0
    
    greedy_budget = int(budget * greedy_ratio)
    fairness_budget = int(budget * fairness_ratio)
    random_budget = budget - greedy_budget - fairness_budget
    
    # 70% Greedy
    greedy_selected = select_greedy(intents, greedy_budget)
    for s in greedy_selected:
        if total_cost + s["cost"] <= budget:
            selected.append(s)
            total_cost += s["cost"]
    
    # 20% Fairness (least selected)
    remaining_budget = min(budget - total_cost, fairness_budget)
    if remaining_budget > 0:
        fairness_selected = select_round_robin(intents, remaining_budget, selection_counts)
        for s in fairness_selected:
            if s not in selected and total_cost + s["cost"] <= budget:
                selected.append(s)
                total_cost += s["cost"]
    
    # 10% Random
    remaining_budget = budget - total_cost
    if remaining_budget > 0:
        shuffled = intents.copy()
        random.shuffle(shuffled)
        for s in shuffled:
            if s not in selected and total_cost + s["cost"] <= remaining_budget:
                selected.append(s)
                total_cost += s["cost"]
    
    return selected


# ============================================================================
# SECTION 4: SIMULATION
# ============================================================================

async def run_policy_simulation(
    policy_name: str,
    intents: List[Dict],
    cycles: int,
    budget: int,
    seed: int
) -> List[CycleResult]:
    """Run simulation for a policy"""
    random.seed(seed)
    
    results = []
    selection_counts = [0] * len(intents)
    intent_id_to_idx = {intents[i]["id"]: i for i in range(len(intents))}
    
    for cycle in range(1, cycles + 1):
        start = time.time()
        
        if policy_name == "Greedy":
            selected = select_greedy(intents, budget)
        elif policy_name == "RoundRobin":
            selected = select_round_robin(intents, budget, selection_counts)
        elif policy_name == "Hybrid":
            selected = select_hybrid(intents, budget, selection_counts)
        else:
            selected = select_greedy(intents, budget)
        
        latency = (time.time() - start) * 1000
        
        # Calculate utility
        total_utility = sum(s["utility"] for s in selected)
        
        # Unique selected
        unique_ids = set(s["id"] for s in selected)
        
        # Update counts
        for s in selected:
            idx = intent_id_to_idx[s["id"]]
            selection_counts[idx] += 1
        
        results.append(CycleResult(
            cycle=cycle,
            selected=selected,
            total_utility=total_utility,
            latency_ms=latency,
            unique_selected=len(unique_ids)
        ))
    
    return results


def calculate_metrics(results: List[CycleResult], selection_counts: List[int]) -> Dict:
    """Calculate metrics"""
    # Cumulative utility
    cumulative = 0
    cumulative_curve = []
    for r in results:
        cumulative += r.total_utility
        cumulative_curve.append(cumulative)
    
    # Average per cycle
    avg_utility = sum(r.total_utility for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    
    # Starvation
    never_selected = sum(1 for c in selection_counts if c == 0)
    starvation_rate = never_selected / len(selection_counts)
    
    # Gini
    sorted_counts = sorted(selection_counts)
    n = len(sorted_counts)
    total = sum(sorted_counts)
    if total > 0:
        gini_sum = sum((2 * i + 1) * c for i, c in enumerate(sorted_counts))
        gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    else:
        gini = 0
    
    return {
        "cumulative_utility": cumulative,
        "avg_utility_per_cycle": avg_utility,
        "avg_latency_ms": avg_latency,
        "starvation_rate": starvation_rate,
        "gini": max(0, min(1, gini)),
        "cumulative_curve": cumulative_curve
    }


# ============================================================================
# SECTION 5: MAIN
# ============================================================================

async def main():
    print("\n" + "=" * 70)
    print("HYBRID POLICY + REAL UTILITY BENCHMARK")
    print("=" * 70)
    
    # Config
    NUM_INTENTS = 1000
    CYCLES = 100
    BUDGET = 10
    SEED = 42
    
    print(f"\nConfig:")
    print(f"  Intents: {NUM_INTENTS}")
    print(f"  Cycles: {CYCLES}")
    print(f"  Budget: {BUDGET}")
    print(f"  Seed: {SEED}")
    print()
    
    # Generate intents once
    intents = generate_intents(NUM_INTENTS, SEED)
    
    # Test policies
    policies = ["Greedy", "RoundRobin", "Hybrid"]
    all_results = {}
    all_metrics = {}
    
    for policy in policies:
        print(f"Running {policy}...")
        results = await run_policy_simulation(policy, intents, CYCLES, BUDGET, SEED)
        
        # Calculate selection counts
        selection_counts = [0] * len(intents)
        intent_id_to_idx = {intents[i]["id"]: i for i in range(len(intents))}
        for r in results:
            for s in r.selected:
                selection_counts[intent_id_to_idx[s["id"]]] += 1
        
        metrics = calculate_metrics(results, selection_counts)
        all_results[policy] = results
        all_metrics[policy] = metrics
        
        print(f"  Cumulative Utility: {metrics['cumulative_utility']:.2f}")
        print(f"  Avg Utility/cycle: {metrics['avg_utility_per_cycle']:.2f}")
        print(f"  Starvation: {metrics['starvation_rate']:.1%}")
        print(f"  Gini: {metrics['gini']:.3f}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Policy':<15} {'Cum.Utility':>14} {'Avg/Cycle':>12} {'Starvation':>12} {'Gini':>8}")
    print("-" * 70)
    
    for policy in policies:
        m = all_metrics[policy]
        print(f"{policy:<15} {m['cumulative_utility']:>14.2f} {m['avg_utility_per_cycle']:>12.2f} {m['starvation_rate']:>11.1%} {m['gini']:>8.3f}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    greedy_utility = all_metrics["Greedy"]["cumulative_utility"]
    robin_utility = all_metrics["RoundRobin"]["cumulative_utility"]
    hybrid_utility = all_metrics["Hybrid"]["cumulative_utility"]
    
    # Quality gap
    greedy_vs_robin = (greedy_utility - robin_utility) / greedy_utility * 100
    hybrid_vs_robin = (hybrid_utility - robin_utility) / robin_utility * 100 if robin_utility > 0 else 0
    greedy_vs_hybrid = (greedy_utility - hybrid_utility) / greedy_utility * 100
    
    print(f"\n📊 Quality Analysis:")
    print(f"  Greedy loses to RoundRobin: {greedy_vs_robin:.1f}% less utility")
    print(f"  Hybrid gains over RoundRobin: {hybrid_vs_robin:.1f}% more utility")
    print(f"  Hybrid loses to Greedy: {greedy_vs_hybrid:.1f}% less utility")
    
    # Fairness
    greedy_starv = all_metrics["Greedy"]["starvation_rate"]
    robin_starv = all_metrics["RoundRobin"]["starvation_rate"]
    hybrid_starv = all_metrics["Hybrid"]["starvation_rate"]
    
    print(f"\n🎯 Fairness Analysis:")
    print(f"  Greedy starvation: {greedy_starv:.1%}")
    print(f"  RoundRobin starvation: {robin_starv:.1%}")
    print(f"  Hybrid starvation: {hybrid_starv:.1%}")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Find best trade-off
    if hybrid_utility >= greedy_utility * 0.95 and hybrid_starv <= greedy_starv * 0.5:
        print(f"  → Hybrid offers best trade-off (95%+ quality, <50% starvation)")
    else:
        print(f"  → Use Greedy for max quality")
        print(f"  → Use RoundRobin for max fairness")
        print(f"  → Use Hybrid for balanced approach")
    
    print("\n" + "=" * 70)
    
    # Output cumulative curve for visualization
    print("\n📈 Cumulative Utility Curves (last 20 cycles):")
    print(f"{'Cycle':>6} {'Greedy':>10} {'RoundRobin':>12} {'Hybrid':>10}")
    print("-" * 45)
    for i in range(max(0, CYCLES - 20), CYCLES):
        g = all_metrics["Greedy"]["cumulative_curve"][i]
        r = all_metrics["RoundRobin"]["cumulative_curve"][i]
        h = all_metrics["Hybrid"]["cumulative_curve"][i]
        print(f"{i+1:>6} {g:>10.2f} {r:>12.2f} {h:>10.2f}")


if __name__ == "__main__":
    asyncio.run(main())
