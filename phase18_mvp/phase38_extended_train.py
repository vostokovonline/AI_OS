"""
Extended closed-loop training with Phases 36+37+38 stack.
Compares energy-regularized vs. standard execution.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from phase38_energy_regularized_dynamics import (
    EnergyCostFunction, EfficiencyEvaluator,
    EnergyRegularizedCEM, EnergyRegularizedEngine
)
from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel, ClosedLoopEngine
)
from phase31_hierarchical_execution import GoalAttractor


def run_energy_training(n_cycles: int = 30, steps_per_cycle: int = 20):
    print("=" * 70)
    print("PHASE 38 EXTENDED ENERGY-REGULARIZED TRAINING")
    print(f"  Cycles: {n_cycles}, Steps/cycle: {steps_per_cycle}")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='energy_train',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )

    engine = EnergyRegularizedEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=12,
        flow_dim=4,
        lambda_cost=0.5,
        train_every_n=5
    )

    # Tracking
    goal_probs = []
    costs = []
    efficiencies = []
    train_losses = []
    val_losses = []
    inv_losses = []

    for cycle in range(n_cycles):
        result = engine.run_cycle(
            np.zeros(16),
            n_steps=steps_per_cycle
        )

        goal_probs.append(result['goal_prob'])
        costs.append(result.get('avg_cost', 0))

        eff = result.get('selection_stats', {}).get('mean_efficiency', 0)
        efficiencies.append(eff)

        inv_losses.append(result['inv_dyn_loss'])

        tr = result.get('training', {})
        if 'train_loss' in tr:
            train_losses.append(tr['train_loss'])
        if 'val_loss' in tr:
            val_losses.append(tr['val_loss'])

        if cycle % 5 == 0 or cycle == n_cycles - 1:
            tr_rep = engine.learner.get_training_report()
            print(f"\n  Cycle {cycle:3d}:")
            print(f"    GP={result['goal_prob']:.6f}  "
                  f"cost={result.get('avg_cost', 0):.4f}  "
                  f"eff={eff:.4f}")
            print(f"    Stability μ={result['stability']['mean']:.3f}  "
                  f"Flows: {result['n_flows']}")
            print(f"    Inv dyn: {result['inv_dyn_loss']:.6f}")
            if train_losses:
                print(f"    Train loss: {train_losses[-1]:.6f}  "
                      f"(trend: {train_losses[0]:.6f}→{train_losses[-1]:.6f})")
            if val_losses:
                print(f"    Val loss: {val_losses[-1]:.6f}")
            print(f"    Buffer: {tr_rep.get('buffer_episodes', 0)} episodes")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = engine.learner.get_training_report()
    print(f"\n  Training steps: {report['training_steps']}")
    print(f"  Buffer: {report['buffer_episodes']} eps, "
          f"{report['buffer_transitions']} trans")

    if train_losses:
        print(f"\n  Train loss: {train_losses[0]:.6f} → {train_losses[-1]:.6f} "
              f"({(train_losses[-1]-train_losses[0])/max(1e-8,train_losses[0])*100:+.1f}%)")

    if val_losses and len(val_losses) >= 3:
        print(f"  Val loss: {val_losses[0]:.6f} → {val_losses[-1]:.6f} "
              f"({(val_losses[-1]-val_losses[0])/max(1e-8,val_losses[0])*100:+.1f}%)")

    print(f"\n  GP: {goal_probs[0]:.6f} → {goal_probs[-1]:.6f}  "
          f"best={max(goal_probs):.6f}")
    print(f"  Cost: {costs[0]:.4f} → {costs[-1]:.4f}")
    print(f"  Efficiency: {efficiencies[0]:.4f} → {efficiencies[-1]:.4f}")

    if inv_losses:
        print(f"  Inv dyn: {inv_losses[0]:.6f} → {inv_losses[-1]:.6f} "
              f"({(inv_losses[-1]-inv_losses[0])/max(1e-8,inv_losses[0])*100:+.1f}%)")

    cs = engine.energy_cost.get_stats()
    print(f"\n  Cost stats: action={cs['mean_action_cost']:.4f}  "
          f"path={cs['mean_path_cost']:.4f}  var={cs['mean_var_cost']:.4f}")

    return engine, report


if __name__ == "__main__":
    engine, report = run_energy_training(n_cycles=30, steps_per_cycle=20)
