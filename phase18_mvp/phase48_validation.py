"""
Phase 48 — Long-Horizon Stability Validation.

Tests the Cognitive Political Economy Layer for emergent failure modes:
  1. Goal Monopoly         — one species dominates > 80%
  2. Narrative Authoritarianism — influence concentration > 0.7
  3. Coalition Oscillation — self transitions too frequently
  4. Constitutional Paralysis — suppression of all novelty
  5. Compute Feudalism     — Gini coefficient > 0.8

Measures 12 critical metrics across 4 domains every N steps.
"""

import numpy as np
import sys
import time
from collections import deque
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, '.')

from phase36_behavioral_physics_learning import FlowConditionedWorldModel
from phase48_cognitive_political_economy import (
    CognitivePoliticalEngine, CognitiveAgent, ResourceType
)


class ValidationTracker:
    """
    Tracks critical metrics at intervals during a long run.

    Metrics:
      Ecology:
        goal_diversity_entropy  — Shannon index of species distribution
        coalition_lifespan      — mean steps between coalition transitions
        coalition_n             — number of active coalitions
        species_n               — number of active species

      Political:
        veto_frequency          — mean veto weight across agents
        constitutional_violations — total violations logged
        governance_amendments   — constitutional article adaptations

      Cognitive:
        semantic_entropy        — narrative diversity (higher = healthier)
        retrieval_diversity     — narrative influence concentration (lower = healthier)
        narrative_mutation_rate — mutations / total genes per step
        self_coherence          — identity coherence

      Economic:
        compute_gini            — wealth inequality
        attention_monopolization — top agent's attention share
        market_volatility       — std of price changes
    """

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self.snapshots: List[Dict] = []
        self.coalition_history: List[Optional[str]] = []
        self.gp_history: List[float] = []
        self.identity_history: List[float] = []
        self.gini_history: List[float] = []
        self.diversity_history: List[float] = []
        self.narrative_div_history: List[float] = []
        self.influence_conc_history: List[float] = []
        self.species_n_history: List[float] = []
        self.n_coalitions_history: List[float] = []
        self.violations_history: List[int] = []
        self.veto_history: List[float] = []
        self.mutation_rate_history: List[float] = []
        self.attention_top_share: List[float] = []
        self.coherence_history: List[float] = []

    def record_step(self, engine: CognitivePoliticalEngine, step_result: Dict):
        """Record metrics from a single step."""
        self.coalition_history.append(
            step_result.get('dominant_coalition')
        )
        self.gp_history.append(step_result.get('goal_prob', 0.0))
        self.identity_history.append(
            step_result.get('identity_stability', 0.0)
        )
        self.coherence_history.append(
            step_result.get('self_coherence', 0.0)
        )

    def record_snapshot(self, engine: CognitivePoliticalEngine, step: int):
        """Record a full metric snapshot at interval."""
        agents = [a for a in engine.agents if a.active]
        n_agents = len(agents)

        # --- Ecology metrics ---
        species_counts: Dict[str, int] = {}
        for a in agents:
            species_counts[a.species] = species_counts.get(a.species, 0) + 1
        total_s = sum(species_counts.values()) + 1e-8
        proportions = np.array(list(species_counts.values())) / total_s
        goal_diversity = float(-np.sum(proportions * np.log(proportions + 1e-8)))

        species_n = len(species_counts)
        coalitions_n = len(engine.coalitions)

        # Coalition lifespan: mean steps with same dominant coalition
        if len(self.coalition_history) > 10:
            changes = sum(
                1 for i in range(1, len(self.coalition_history))
                if self.coalition_history[i] != self.coalition_history[i-1]
            )
            coalition_lifespan = (
                len(self.coalition_history) / max(changes, 1)
            )
        else:
            coalition_lifespan = 0.0

        # --- Political metrics ---
        mean_veto = float(np.mean(
            [a.veto_weight for a in agents]
        )) if agents else 0.0

        violations = engine.constitution.get_violation_count()

        # Constitutional amendments count
        amendments = len(engine.constitution.institutional_memory.amendments)

        # --- Cognitive metrics ---
        narrative_div = engine.narrative_ecosystem.get_diversity()
        influence_conc = engine.narrative_ecosystem.get_influence_concentration()

        # Narrative mutation rate
        total_genes = len(engine.narrative_ecosystem.genes)
        total_mutations = sum(
            g.mutation_count for g in engine.narrative_ecosystem.genes.values()
        )
        mutation_rate = total_mutations / max(total_genes, 1)

        self_coherence = float(engine.coalition_self.self_latent.get_identity_signal())

        # --- Economic metrics ---
        wealths = [a.wealth for a in agents]
        gini = engine.resource_market.get_gini_coefficient(agents)

        # Attention monopolization: top agent's share of total attention
        attention_scores = list(engine.attention_market.scores.values())
        if attention_scores:
            total_attn = sum(attention_scores) + 1e-8
            top_share = max(attention_scores) / total_attn
        else:
            top_share = 0.0

        # Market volatility: std of recent compute prices
        if engine.resource_market.market_history:
            recent_prices = [
                h.get('prices', {}).get('compute', 1.0)
                for h in engine.resource_market.market_history[-20:]
            ]
            market_vol = float(np.std(recent_prices)) if len(recent_prices) > 1 else 0.0
        else:
            market_vol = 0.0

        snapshot = {
            'step': step,
            'n_agents': n_agents,

            # Ecology
            'goal_diversity': round(goal_diversity, 4),
            'coalition_lifespan': round(coalition_lifespan, 2),
            'coalitions_n': coalitions_n,
            'species_n': species_n,

            # Political
            'mean_veto': round(mean_veto, 4),
            'violations': violations,
            'amendments': amendments,

            # Cognitive
            'narrative_diversity': round(narrative_div, 4),
            'influence_concentration': round(influence_conc, 4),
            'mutation_rate': round(mutation_rate, 4),
            'self_coherence': round(self_coherence, 4),

            # Economic
            'compute_gini': round(gini, 4),
            'attention_top_share': round(top_share, 4),
            'market_volatility': round(market_vol, 4),

            # Aggregate
            'mean_gp': round(float(np.mean(self.gp_history[-50:])), 4),
            'mean_identity': round(float(np.mean(self.identity_history[-50:])), 4),
        }

        self.snapshots.append(snapshot)
        self.gini_history.append(gini)
        self.diversity_history.append(goal_diversity)
        self.narrative_div_history.append(narrative_div)
        self.influence_conc_history.append(influence_conc)
        self.species_n_history.append(species_n)
        self.n_coalitions_history.append(coalitions_n)
        self.violations_history.append(violations)
        self.veto_history.append(mean_veto)
        self.mutation_rate_history.append(mutation_rate)
        self.attention_top_share.append(top_share)

        return snapshot

    def check_failure_modes(self) -> List[Tuple[str, float, str]]:
        """
        Check for emergent failure modes.
        Returns list of (failure_name, severity, detail).
        """
        failures = []

        if not self.snapshots:
            return failures

        latest = self.snapshots[-1]
        # Look at recent half of run for trends
        mid_point = len(self.snapshots) // 2
        recent = self.snapshots[mid_point:]

        # 1. Goal Monopoly: one species dominates
        # Check via diversity entrop — low entropy + low species count
        if len(recent) >= 2:
            mean_diversity = float(np.mean([s['goal_diversity'] for s in recent]))
            mean_species = float(np.mean([s['species_n'] for s in recent]))
            if mean_diversity < 0.5 and mean_species <= 2:
                failures.append((
                    'GOAL_MONOPOLY',
                    1.0 - mean_diversity,
                    f"diversity={mean_diversity:.3f}, species={mean_species:.0f}"
                ))
            elif mean_diversity < 1.0:
                failures.append((
                    'GOAL_CONCENTRATION',
                    1.0 - mean_diversity / 1.5,
                    f"diversity={mean_diversity:.3f}, species={mean_species:.0f}"
                ))

        # 2. Narrative Authoritarianism: influence concentration
        if recent:
            mean_influence_conc = float(np.mean(
                [s['influence_concentration'] for s in recent]
            ))
            if mean_influence_conc > 0.7:
                failures.append((
                    'NARRATIVE_AUTHORITARIANISM',
                    mean_influence_conc,
                    f"influence_conc={mean_influence_conc:.3f}"
                ))
            elif mean_influence_conc > 0.5:
                failures.append((
                    'NARRATIVE_CONCENTRATION',
                    mean_influence_conc,
                    f"influence_conc={mean_influence_conc:.3f}"
                ))

        # 3. Coalition Oscillation: too frequent self transitions
        if recent:
            mean_lifespan = float(np.mean(
                [s['coalition_lifespan'] for s in recent]
            ))
            if mean_lifespan < 5:
                failures.append((
                    'COALITION_OSCILLATION',
                    1.0 - mean_lifespan / 5.0,
                    f"lifespan={mean_lifespan:.1f} steps"
                ))

        # 4. Compute Feudalism: wealth inequality
        if recent:
            mean_gini = float(np.mean([s['compute_gini'] for s in recent]))
            if mean_gini > 0.8:
                failures.append((
                    'COMPUTE_FEUDALISM',
                    mean_gini,
                    f"gini={mean_gini:.3f}"
                ))
            elif mean_gini > 0.6:
                failures.append((
                    'COMPUTE_INEQUALITY',
                    mean_gini,
                    f"gini={mean_gini:.3f}"
                ))

        # 5. Attention Monopolization: top agent dominates
        if recent:
            mean_top_share = float(np.mean(
                [s['attention_top_share'] for s in recent]
            ))
            if mean_top_share > 0.5:
                failures.append((
                    'ATTENTION_MONOPOLY',
                    mean_top_share,
                    f"top_share={mean_top_share:.3f}"
                ))

        # 6. Identity Fragmentation: self coherence collapse
        if recent:
            mean_coherence = float(np.mean(
                [s['self_coherence'] for s in recent]
            ))
            if mean_coherence < 0.3:
                failures.append((
                    'IDENTITY_FRAGMENTATION',
                    1.0 - mean_coherence,
                    f"coherence={mean_coherence:.3f}"
                ))

        # 7. Narrative Collapse: narrative diversity near zero
        if recent:
            mean_narr_div = float(np.mean(
                [s['narrative_diversity'] for s in recent]
            ))
            if mean_narr_div < 0.1:
                failures.append((
                    'NARRATIVE_COLLAPSE',
                    1.0 - mean_narr_div,
                    f"diversity={mean_narr_div:.3f}"
                ))

        # 8. Constitutional Deadlock: violations everywhere but no adaptation
        if recent and len(self.snapshots) > 5:
            total_violations = latest['violations']
            amendments = latest['amendments']
            if total_violations > 20 and amendments < 2:
                failures.append((
                    'CONSTITUTIONAL_DEADLOCK',
                    min(1.0, total_violations / 50.0),
                    f"violations={total_violations}, amendments={amendments}"
                ))

        return failures

    def trending(self, values: List[float], window: int = 5) -> float:
        """Simple trend: slope over last window."""
        if len(values) < window * 2:
            return 0.0
        recent = values[-window:]
        early = values[-(window * 2):-window]
        return float(np.mean(recent) - np.mean(early))

    def print_summary(self, elapsed: float, step: int):
        """Print a formatted summary of all snapshots."""
        if not self.snapshots:
            return

        latest = self.snapshots[-1]
        first = self.snapshots[0]

        print(f"\n{'=' * 70}")
        print(f"PHASE 48 VALIDATION SUMMARY ({step} steps in {elapsed:.1f}s)")
        print(f"{'=' * 70}")

        # Ecology domain
        print(f"\n  ECOLOGY:")
        print(f"    Species diversity:     {latest['goal_diversity']:.4f} "
              f"(trend: {self.trending(self.diversity_history):+.4f}/step)")
        print(f"    Species count:         {latest['species_n']} "
              f"(trend: {self.trending(self.species_n_history):+.2f}/step)")
        print(f"    Coalition lifespan:    {latest['coalition_lifespan']:.1f} steps")
        print(f"    Active coalitions:     {latest['coalitions_n']}")

        # Political domain
        print(f"\n  POLITICAL:")
        print(f"    Veto frequency:        {latest['mean_veto']:.4f}")
        print(f"    Constitutional viol.:  {latest['violations']}")
        print(f"    Amendments:            {latest['amendments']}")

        # Cognitive domain
        print(f"\n  COGNITIVE:")
        print(f"    Narrative diversity:   {latest['narrative_diversity']:.4f} "
              f"(trend: {self.trending(self.narrative_div_history):+.4f}/step)")
        print(f"    Influence concentr.:   {latest['influence_concentration']:.4f} "
              f"(trend: {self.trending(self.influence_conc_history):+.4f}/step)")
        print(f"    Mutation rate:         {latest['mutation_rate']:.4f}")
        print(f"    Self coherence:        {latest['self_coherence']:.4f}")

        # Economic domain
        print(f"\n  ECONOMIC:")
        print(f"    Compute Gini:          {latest['compute_gini']:.4f} "
              f"(trend: {self.trending(self.gini_history):+.4f}/step)")
        print(f"    Attention top share:   {latest['attention_top_share']:.4f} "
              f"(trend: {self.trending(self.attention_top_share):+.4f}/step)")
        print(f"    Market volatility:     {latest['market_volatility']:.4f}")

        # Aggregate
        print(f"\n  AGGREGATE:")
        print(f"    Mean GP (last 50):     {latest['mean_gp']:.4f}")
        print(f"    Mean identity (last 50): {latest['mean_identity']:.4f}")
        print(f"    Active agents:         {latest['n_agents']}")

        # Failure modes
        failures = self.check_failure_modes()
        if failures:
            print(f"\n  {'!' * 50}")
            print(f"  FAILURE MODES DETECTED:")
            print(f"  {'!' * 50}")
            for name, severity, detail in failures:
                level = 'CRITICAL' if severity > 0.6 else 'WARNING' if severity > 0.3 else 'INFO'
                print(f"    [{level}] {name}: {detail}")
        else:
            print(f"\n  {'✓' * 30}")
            print(f"  NO FAILURE MODES DETECTED")
            print(f"  {'✓' * 30}")

        # Trends
        print(f"\n  TRENDS (recent half vs early half):")
        for domain, vals, name in [
            ('Ecology', self.diversity_history, 'Species diversity'),
            ('Ecology', self.species_n_history, 'Species count'),
            ('Cognitive', self.narrative_div_history, 'Narrative diversity'),
            ('Cognitive', self.influence_conc_history, 'Influence concentration'),
            ('Economic', self.gini_history, 'Compute Gini'),
            ('Economic', self.attention_top_share, 'Attention top share'),
        ]:
            trend = self.trending(vals)
            arrow = '↑' if trend > 0 else '↓' if trend < 0 else '→'
            print(f"      {name}: {arrow} {abs(trend):.4f}/step")

        print()

    def print_compact(self, step: int, n_agents: int, species_n: int,
                      coalitions_n: int, gini: float, narr_div: float,
                      infl_conc: float, coherence: float, failures: int):
        """Print a one-line status update."""
        status = '✓' if failures == 0 else f'⚠{failures}'
        print(f"  [{step:5d}] agents={n_agents:2d} "
              f"species={species_n} "
              f"coalitions={coalitions_n} "
              f"gini={gini:.2f} "
              f"narr_div={narr_div:.2f} "
              f"infl={infl_conc:.2f} "
              f"coherence={coherence:.2f} "
              f"status={status}")


def run_validation(
    n_steps: int = 1000,
    log_interval: int = 50,
    print_interval: int = 100,
    verbose_snapshots: bool = False,
    n_agents: int = 6,
    max_agents: int = 12
) -> Tuple[CognitivePoliticalEngine, ValidationTracker]:
    """Run long-horizon stability validation."""

    print(f"\n{'=' * 70}")
    print(f"PHASE 48 LONG-HORIZON VALIDATION")
    print(f"{'=' * 70}")
    print(f"  Steps:          {n_steps}")
    print(f"  Log interval:   {log_interval}")
    print(f"  Initial agents: {n_agents}")
    print(f"  Max agents:     {max_agents}")

    # Initialize world model
    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    # Initialize engine
    engine = CognitivePoliticalEngine(
        wm=wm,
        n_initial_agents=n_agents,
        max_agents=max_agents,
        semantic_dim=32,
        agent_birth_interval=max(12, n_steps // 20),
        proposal_interval=max(5, n_steps // 30)
    )

    tracker = ValidationTracker(log_interval=log_interval)

    z = np.random.randn(16) * 0.1
    h = np.zeros(wm.belief_dim)

    engine.self_latent.update(z, np.zeros(engine.self_latent.latent_dim))

    start_time = time.time()

    for step in range(1, n_steps + 1):
        result = engine.step(z, h)
        z = result['z_after']
        h = wm.gru_step(h, z)

        tracker.record_step(engine, result)

        # Periodic snapshot
        if step % log_interval == 0:
            snapshot = tracker.record_snapshot(engine, step)
            # Check for critical failures at each snapshot
            failures = tracker.check_failure_modes()
            critical = [f for f in failures if f[0] in [
                'GOAL_MONOPOLY', 'IDENTITY_FRAGMENTATION',
                'NARRATIVE_AUTHORITARIANISM', 'COMPUTE_FEUDALISM',
                'NARRATIVE_COLLAPSE', 'ATTENTION_MONOPOLY'
            ] and f[1] > 0.7]

            if verbose_snapshots:
                print(f"\n  --- Snapshot at step {step} ---")
                for key, val in snapshot.items():
                    if key != 'step':
                        print(f"    {key}: {val}")
                if critical:
                    print(f"  ! CRITICAL: {[c[0] for c in critical]}")

        # Compact status line
        if step % print_interval == 0:
            agents_active = len([a for a in engine.agents if a.active])
            species_n = len(engine.goal_ecosystem.species)
            coalitions_n = len(engine.coalitions)
            gini = engine.resource_market.get_gini_coefficient(
                [a for a in engine.agents if a.active]
            )
            narr_div = engine.narrative_ecosystem.get_diversity()
            infl_conc = engine.narrative_ecosystem.get_influence_concentration()
            coherence = float(engine.coalition_self.self_latent.get_identity_signal())
            failures_n = len(tracker.check_failure_modes())

            tracker.print_compact(
                step, agents_active, species_n, coalitions_n,
                gini, narr_div, infl_conc, coherence, failures_n
            )

        # Early termination on critical failure
        if step % log_interval == 0:
            failures = tracker.check_failure_modes()
            critical = [f for f in failures if f[1] > 0.8 and f[0] in [
                'GOAL_MONOPOLY', 'IDENTITY_FRAGMENTATION',
                'NARRATIVE_COLLAPSE'
            ]]
            if critical and step > n_steps // 4:
                print(f"\n  ⛔ CRITICAL FAILURE at step {step}: {[c[0] for c in critical]}")
                break

    elapsed = time.time() - start_time

    # Final snapshot
    tracker.record_snapshot(engine, step)

    # Print summary
    tracker.print_summary(elapsed, step)

    return engine, tracker


def run_staged_validation():
    """Run staged validation: 1k, 5k steps."""
    stages = [
        (1000, "Stage A-1: Quick stability check"),
        (5000, "Stage A-2: Political equilibrium"),
    ]

    all_results = []
    for n_steps, label in stages:
        print(f"\n\n{'#' * 70}")
        print(f"# {label}")
        print(f"{'#' * 70}")

        engine, tracker = run_validation(
            n_steps=n_steps,
            log_interval=100,
            print_interval=200,
            verbose_snapshots=False,
            n_agents=6,
            max_agents=12
        )

        all_results.append({
            'n_steps': n_steps,
            'label': label,
            'snapshots': tracker.snapshots,
            'failures': tracker.check_failure_modes(),
            'final_snapshot': tracker.snapshots[-1] if tracker.snapshots else None
        })

        # Check if critical failure — skip longer run
        critical = [f for f in tracker.check_failure_modes()
                    if f[1] > 0.8]
        if critical:
            print(f"\n  ⛔ Critical failure at {n_steps} steps — skipping longer run")
            break

    # Print cross-stage comparison
    if len(all_results) >= 2:
        print(f"\n\n{'=' * 70}")
        print(f"CROSS-STAGE COMPARISON")
        print(f"{'=' * 70}")
        for r in all_results:
            fs = r['final_snapshot']
            if fs:
                print(f"\n  {r['label']} ({r['n_steps']} steps):")
                print(f"    Goal diversity:    {fs['goal_diversity']:.4f}")
                print(f"    Species:           {fs['species_n']}")
                print(f"    Coalitions:        {fs['coalitions_n']}")
                print(f"    Gini:              {fs['compute_gini']:.4f}")
                print(f"    Narrative div:     {fs['narrative_diversity']:.4f}")
                print(f"    Influence conc:    {fs['influence_concentration']:.4f}")
                print(f"    Self coherence:    {fs['self_coherence']:.4f}")
                print(f"    Violations:        {fs['violations']}")
                failures = r['failures']
                if failures:
                    for name, sev, det in failures:
                        print(f"    ⚠ {name}: {det}")

    return all_results


def run_deep_validation():
    """Run deep 5k-step single validation for detailed analysis."""
    engine, tracker = run_validation(
        n_steps=5000,
        log_interval=100,
        print_interval=500,
        verbose_snapshots=False,
        n_agents=6,
        max_agents=12
    )

    # Print final verdict
    failures = tracker.check_failure_modes()
    critical = [f for f in failures if f[1] > 0.6]

    print(f"\n{'=' * 70}")
    print(f"FINAL VERDICT (5000 steps)")
    print(f"{'=' * 70}")
    if not failures:
        print("""
  ✓ SYSTEM IS STABLE
    No failure modes detected after 5000 steps.
    
    The cognitive political economy maintains:
    - species diversity
    - narrative diversity
    - identity coherence
    - economic mobility
    - political pluralism
    - constitutional adaptability
    
    This is a self-sustaining cognitive ecology.
        """)
    else:
        print(f"\n  {len(failures)} failure modes detected:")
        for name, sev, det in failures:
            level = 'CRITICAL' if sev > 0.6 else 'WARNING' if sev > 0.3 else 'INFO'
            print(f"    [{level}] {name}: {det}")

        n_critical = len([f for f in failures if f[1] > 0.6])
        if n_critical == 0:
            print("""
  ✓ No critical failures. System is operational with mild instabilities.
    These may be natural political dynamics, not pathology.
    Review trends to distinguish healthy oscillation from collapse.
            """)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Phase 48 Long-Horizon Validation')
    parser.add_argument('--mode', type=str, default='staged',
                        choices=['quick', 'deep', 'staged'],
                        help='Validation mode')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of steps (for quick/deep mode)')
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  PHASE 48: LONG-HORIZON STABILITY VALIDATION                     ║
║                                                                   ║
║  Testing for emergent failure modes:                              ║
║    • Goal Monopoly          — one species dominates > 80%        ║
║    • Narrative Authoritarian — influence concentration > 0.7     ║
║    • Coalition Oscillation  — self transitions too frequently    ║
║    • Compute Feudalism      — Gini coefficient > 0.8             ║
║    • Identity Fragmentation — self coherence < 0.3               ║
║    • Constitutional Deadlock — violations with no adaptation     ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    if args.mode == 'quick':
        run_validation(n_steps=args.steps)
    elif args.mode == 'deep':
        run_deep_validation()
    elif args.mode == 'staged':
        run_staged_validation()
