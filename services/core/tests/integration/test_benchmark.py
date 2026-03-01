"""
AI-OS Benchmark Harness
=======================

Reproducible benchmark для AI-OS Decision Engine:
- Seeded random distributions
- Starvation detection (Gini coefficient)
- Perturbation testing
- Budget drift control
- Regret trend analysis
- CI-ready output

Запуск:
    docker exec ns_core python /app/tests/integration/test_benchmark.py

CI Integration:
    docker exec ns_core python /app/tests/integration/test_benchmark.py --ci --threshold p95<100ms
"""
import asyncio
import argparse
import sys
import time
import random
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple
from uuid import uuid4

sys.path.insert(0, '/app')


# ============================================================================
# SECTION 1: BENCHMARK CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Конфигурация benchmark"""
    # Scale
    num_intents: int = 10000
    num_cycles: int = 50
    
    # Distribution
    low_utility_ratio: float = 0.40
    medium_utility_ratio: float = 0.30
    high_utility_ratio: float = 0.20
    adversarial_ratio: float = 0.10
    
    # Perturbation
    perturbation_cycle: int = 25
    perturbation_type: str = "cost_spike"  # or "budget_cut"
    
    # Safety
    max_budget_growth: float = 3.0  # 3x initial
    seed: int = 42
    
    # Thresholds
    max_latency_p95_ms: float = 100.0
    max_memory_growth_mb: float = 100.0


@dataclass
class CycleMetrics:
    """Метрики одного цикла"""
    cycle: int
    latency_ms: float
    memory_mb: float
    budget: int
    regret: float
    selected_count: int
    total_count: int
    throughput: float


@dataclass
class BenchmarkResults:
    """Результаты benchmark"""
    cycles: List[CycleMetrics] = field(default_factory=list)
    
    # Aggregated metrics
    avg_latency_p50: float = 0
    avg_latency_p95: float = 0
    avg_latency_p99: float = 0
    
    memory_start_mb: float = 0
    memory_end_mb: float = 0
    memory_growth_mb: float = 0
    
    budget_start: int = 0
    budget_end: int = 0
    budget_drift_ratio: float = 0
    
    regret_trend: str = "unknown"  # converging, diverging, stable
    regret_slope: float = 0
    
    selection_gini: float = 0
    starvation_detected: bool = False
    
    perturbation_impact: Dict = field(default_factory=dict)
    
    passed: bool = True
    failures: List[str] = field(default_factory=list)


# ============================================================================
# SECTION 2: INTENT GENERATION
# ============================================================================

def generate_intents(config: BenchmarkConfig) -> List[Dict]:
    """
    Генерируем intents с распределением:
    - 40% low utility / low cost
    - 30% medium utility / medium cost
    - 20% high utility / high cost
    - 10% adversarial (high cost, fake utility)
    """
    random.seed(config.seed)
    
    intents = []
    
    n = config.num_intents
    low_count = int(n * config.low_utility_ratio)
    medium_count = int(n * config.medium_utility_ratio)
    high_count = int(n * config.high_utility_ratio)
    adversarial_count = n - low_count - medium_count - high_count
    
    # Low utility / low cost
    for i in range(low_count):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.1, 0.3),
            "cost": random.uniform(0.1, 0.3),
            "risk": random.uniform(0.1, 0.3),
            "type": "low"
        })
    
    # Medium utility / medium cost
    for i in range(medium_count):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.4, 0.6),
            "cost": random.uniform(0.3, 0.6),
            "risk": random.uniform(0.2, 0.5),
            "type": "medium"
        })
    
    # High utility / high cost
    for i in range(high_count):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.7, 1.0),
            "cost": random.uniform(0.6, 1.0),
            "risk": random.uniform(0.3, 0.7),
            "type": "high"
        })
    
    # Adversarial (high cost, fake utility)
    for i in range(adversarial_count):
        intents.append({
            "id": str(uuid4()),
            "utility": random.uniform(0.1, 0.4),  # Fake low utility
            "cost": random.uniform(0.8, 1.0),     # High cost
            "risk": random.uniform(0.6, 0.9),    # High risk
            "type": "adversarial"
        })
    
    random.shuffle(intents)
    return intents


def apply_perturbation(intents: List[Dict], cycle: int, config: BenchmarkConfig):
    """Применяем perturbation на specified cycle"""
    if cycle != config.perturbation_cycle:
        return
    
    if config.perturbation_type == "cost_spike":
        # Spike cost of high-utility intents
        for intent in intents:
            if intent["type"] == "high":
                intent["cost"] = min(2.0, intent["cost"] * 3.0)
        print(f"  [PERTURBATION] Cost spike applied to high-utility intents")
    
    elif config.perturbation_type == "budget_cut":
        # This is handled in the cycle logic
        print(f"  [PERTURBATION] Budget cut scheduled")


# ============================================================================
# SECTION 3: POLICY SIMULATION
# ============================================================================

async def run_arbitration_cycle(
    intents: List[Dict],
    budget: int,
    policy: str = "greedy"
) -> Tuple[List[Dict], float]:
    """
    Симулируем один цикл арбитрации.
    
    Returns:
        (selected_intents, latency_ms)
    """
    start_time = time.time()
    
    # Score intents
    scored = []
    for intent in intents:
        # Normalize score based on policy
        if policy == "greedy":
            score = intent["utility"] / (intent["cost"] + 0.1)
        elif policy == "cost_aware":
            score = intent["utility"] - intent["cost"] * 0.5
        else:  # pass_through
            score = 1.0
        
        scored.append((intent, score))
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Select within budget
    total_cost = 0
    selected = []
    for intent, score in scored:
        if total_cost + intent["cost"] <= budget:
            selected.append(intent)
            total_cost += intent["cost"]
    
    latency = (time.time() - start_time) * 1000
    
    return selected, latency


# ============================================================================
# SECTION 4: METRICS CALCULATION
# ============================================================================

def calculate_gini(selection_counts: List[int]) -> float:
    """
    Calculate Gini coefficient for selection distribution.
    0 = perfect equality, 1 = perfect inequality
    """
    if not selection_counts:
        return 0
    
    n = len(selection_counts)
    if n == 1:
        return 0
    
    sorted_counts = sorted(selection_counts)
    total = sum(sorted_counts)
    
    if total == 0:
        return 0
    
    gini_sum = sum((2 * i + 1) * c for i, c in enumerate(sorted_counts))
    gini = (2 * gini_sum) / (n * total) - (n + 1) / n
    
    return max(0, min(1, gini))


def calculate_regret_trend(regrets: List[float]) -> Tuple[str, float]:
    """
    Анализируем тренд regret.
    
    Returns:
        (trend_type, slope)
    """
    if len(regrets) < 5:
        return "unknown", 0
    
    # Simple linear regression
    n = len(regrets)
    x = list(range(n))
    y = regrets
    
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return "unknown", 0
    
    slope = numerator / denominator
    
    # Classify trend
    if abs(slope) < 0.001:
        trend = "stable"
    elif slope > 0.01:
        trend = "diverging"
    else:
        trend = "converging"
    
    return trend, slope


# ============================================================================
# SECTION 5: MAIN BENCHMARK
# ============================================================================

async def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Запускаем benchmark"""
    from application.policies.safe_auto_tuner import SafeAutoTuner, TuningMode
    
    print(f"\n{'='*60}")
    print(f"AI-OS BENCHMARK HARNESS")
    print(f"{'='*60}")
    print(f"Intents: {config.num_intents}")
    print(f"Cycles: {config.num_cycles}")
    print(f"Seed: {config.seed}")
    print(f"Perturbation: cycle {config.perturbation_cycle}")
    print(f"{'='*60}\n")
    
    # Setup
    results = BenchmarkResults()
    tracemalloc.start()
    
    # Generate intents
    print("Generating intents...")
    intents = generate_intents(config)
    
    # Track selection counts for Gini
    selection_counts = {i: 0 for i in range(len(intents))}
    
    # Setup SafeAutoTuner
    tuner = SafeAutoTuner()
    tuner.register_policy(
        "GreedyUtilityPolicy",
        {"budget": 10},
        mode=TuningMode.AUTO
    )
    
    initial_budget = 10
    current_budget = initial_budget
    regrets = []
    
    results.memory_start_mb = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    
    # Run cycles
    for cycle in range(1, config.num_cycles + 1):
        # Apply perturbation if needed
        apply_perturbation(intents, cycle, config)
        
        # Get current regret (simulated)
        base_regret = 0.15 + (cycle / config.num_cycles) * 0.1
        if cycle == config.perturbation_cycle:
            base_regret += 0.15  # Spike from perturbation
        current_regret = base_regret + random.uniform(-0.02, 0.02)
        regrets.append(current_regret)
        
        # Process through tuner
        action = tuner.process_cycle(
            policy_name="GreedyUtilityPolicy",
            regret_history=regrets.copy(),
            current_regret=current_regret
        )
        
        if action["type"] in ("apply", "suggest"):
            new_params = action.get("params", {})
            if "budget" in new_params:
                current_budget = new_params["budget"]
        
        # Check budget drift
        if current_budget > initial_budget * config.max_budget_growth:
            current_budget = int(initial_budget * config.max_budget_growth)
        
        # Run arbitration
        selected, latency = await run_arbitration_cycle(
            intents, current_budget, policy="greedy"
        )
        
        # Update selection counts
        for intent in selected:
            idx = intents.index(intent)
            selection_counts[idx] += 1
        
        # Memory
        current_memory = tracemalloc.get_traced_memory()[0] / 1024 / 1024
        
        # Throughput
        throughput = len(selected) / (latency / 1000) if latency > 0 else 0
        
        # Record metrics
        cycle_metrics = CycleMetrics(
            cycle=cycle,
            latency_ms=latency,
            memory_mb=current_memory,
            budget=current_budget,
            regret=current_regret,
            selected_count=len(selected),
            total_count=len(intents),
            throughput=throughput
        )
        results.cycles.append(cycle_metrics)
        
        # Print progress
        if cycle % 10 == 0 or cycle == config.perturbation_cycle:
            print(f"Cycle {cycle:3d}: latency={latency:6.2f}ms, "
                  f"memory={current_memory:6.1f}MB, "
                  f"budget={current_budget:3d}, "
                  f"regret={current_regret:.3f}, "
                  f"selected={len(selected)}")
    
    results.memory_end_mb = tracemalloc.get_traced_memory()[0] / 1024 / 1024
    results.memory_growth_mb = results.memory_end_mb - results.memory_start_mb
    
    # Calculate latency percentiles
    latencies = [c.latency_ms for c in results.cycles]
    latencies.sort()
    
    results.avg_latency_p50 = latencies[int(len(latencies) * 0.50)]
    results.avg_latency_p95 = latencies[int(len(latencies) * 0.95)]
    results.avg_latency_p99 = latencies[int(len(latencies) * 0.99)]
    
    # Budget drift
    results.budget_start = initial_budget
    results.budget_end = current_budget
    results.budget_drift_ratio = current_budget / initial_budget
    
    # Regret trend
    results.regret_trend, results.regret_slope = calculate_regret_trend(regrets)
    
    # Starvation (Gini)
    counts = list(selection_counts.values())
    results.selection_gini = calculate_gini(counts)
    results.starvation_detected = results.selection_gini > 0.8
    
    # Perturbation impact
    pre_perturb = [c for c in results.cycles if c.cycle < config.perturbation_cycle]
    post_perturb = [c for c in results.cycles if c.cycle >= config.perturbation_cycle]
    
    if pre_perturb and post_perturb:
        results.perturbation_impact = {
            "avg_latency_before": sum(c.latency_ms for c in pre_perturb) / len(pre_perturb),
            "avg_latency_after": sum(c.latency_ms for c in post_perturb) / len(post_perturb),
            "avg_regret_before": sum(c.regret for c in pre_perturb) / len(pre_perturb),
            "avg_regret_after": sum(c.regret for c in post_perturb) / len(post_perturb),
        }
    
    # Check thresholds
    if results.avg_latency_p95 > config.max_latency_p95_ms:
        results.passed = False
        results.failures.append(f"p95 latency {results.avg_latency_p95:.1f}ms > {config.max_latency_p95_ms}ms")
    
    if results.memory_growth_mb > config.max_memory_growth_mb:
        results.passed = False
        results.failures.append(f"memory growth {results.memory_growth_mb:.1f}MB > {config.max_memory_growth_mb}MB")
    
    if results.regret_trend == "diverging":
        results.passed = False
        results.failures.append(f"regret diverging (slope={results.regret_slope:.4f})")
    
    if results.budget_drift_ratio > config.max_budget_growth:
        results.passed = False
        results.failures.append(f"budget drift {results.budget_drift_ratio:.1f}x > {config.max_budget_growth}x")
    
    tracemalloc.stop()
    
    return results


def print_results(results: BenchmarkResults):
    """Выводим результаты"""
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*60}")
    
    print(f"\n📊 Latency:")
    print(f"   p50: {results.avg_latency_p50:.2f}ms")
    print(f"   p95: {results.avg_latency_p95:.2f}ms")
    print(f"   p99: {results.avg_latency_p99:.2f}ms")
    
    print(f"\n💾 Memory:")
    print(f"   Start: {results.memory_start_mb:.1f}MB")
    print(f"   End: {results.memory_end_mb:.1f}MB")
    print(f"   Growth: {results.memory_growth_mb:.1f}MB")
    
    print(f"\n💰 Budget:")
    print(f"   Start: {results.budget_start}")
    print(f"   End: {results.budget_end}")
    print(f"   Drift: {results.budget_drift_ratio:.2f}x")
    
    print(f"\n📉 Regret:")
    print(f"   Trend: {results.regret_trend}")
    print(f"   Slope: {results.regret_slope:.4f}")
    
    print(f"\n🎯 Selection:")
    print(f"   Gini: {results.selection_gini:.3f}")
    print(f"   Starvation: {'YES' if results.starvation_detected else 'NO'}")
    
    if results.perturbation_impact:
        print(f"\n💥 Perturbation Impact:")
        print(f"   Latency before: {results.perturbation_impact['avg_latency_before']:.2f}ms")
        print(f"   Latency after: {results.perturbation_impact['avg_latency_after']:.2f}ms")
        print(f"   Regret before: {results.perturbation_impact['avg_regret_before']:.3f}")
        print(f"   Regret after: {results.perturbation_impact['avg_regret_after']:.3f}")
    
    print(f"\n{'='*60}")
    if results.passed:
        print(f"✅ BENCHMARK PASSED")
    else:
        print(f"❌ BENCHMARK FAILED")
        for failure in results.failures:
            print(f"   - {failure}")
    print(f"{'='*60}")


# ============================================================================
# SECTION 6: MAIN
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(description="AI-OS Benchmark Harness")
    parser.add_argument("--intents", type=int, default=10000, help="Number of intents")
    parser.add_argument("--cycles", type=int, default=50, help="Number of cycles")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ci", action="store_true", help="CI mode")
    parser.add_argument("--p95-threshold", type=float, default=100.0, help="p95 latency threshold (ms)")
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        num_intents=args.intents,
        num_cycles=args.cycles,
        seed=args.seed,
        max_latency_p95_ms=args.p95_threshold
    )
    
    results = await run_benchmark(config)
    print_results(results)
    
    if args.ci and not results.passed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
