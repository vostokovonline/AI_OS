"""
Phase 36 — Extended closed-loop training across many cycles.
Measures world model improvement on flow-generated trajectories.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from phase36_behavioral_physics_learning import (
    FlowConditionedWorldModel, FlowTrajectoryBuffer,
    BehavioralPhysicsLearner, ClosedLoopEngine
)
from phase31_hierarchical_execution import GoalAttractor


def run_extended_training(n_cycles: int = 20, steps_per_cycle: int = 20):
    print("=" * 70)
    print("PHASE 36 EXTENDED CLOSED-LOOP TRAINING")
    print(f"  Cycles: {n_cycles}, Steps/cycle: {steps_per_cycle}")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='extended_train_goal',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )

    engine = ClosedLoopEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=12,
        flow_dim=4,
        train_every_n_steps=5,  # More frequent training
        learning_rate=0.02,
        k_steps=4
    )

    # Track metrics over cycles
    goal_probs = []
    losses = []
    val_losses = []
    inv_losses = []

    for cycle in range(n_cycles):
        result = engine.run_cycle(
            np.zeros(16),
            n_steps=steps_per_cycle
        )

        goal_probs.append(result['goal_prob'])
        inv_losses.append(result['inv_dyn_loss'])

        tr = result.get('training', {})
        if 'train_loss' in tr:
            losses.append(tr['train_loss'])
        if 'val_loss' in tr:
            val_losses.append(tr['val_loss'])

        if cycle % 5 == 0 or cycle == n_cycles - 1:
            tr_rep = engine.learner.get_training_report()
            print(f"\n  Cycle {cycle:3d}:")
            print(f"    Goal prob: {result['goal_prob']:.4f}  "
                  f"Stability: μ={result['stability']['mean']:.3f}")
            print(f"    Inv dyn loss: {result['inv_dyn_loss']:.6f}  "
                  f"Flows: {result['n_flows']}")
            if losses:
                recent = losses[-min(3, len(losses)):]
                print(f"    Train loss: {np.mean(recent):.6f}  "
                      f"(trend over last 3: {recent[0]:.6f} → {recent[-1]:.6f})")
            if val_losses:
                print(f"    Val loss: {val_losses[-1]:.6f}")
            print(f"    Buffer: {tr_rep.get('buffer_episodes', 0)} episodes, "
                  f"{tr_rep.get('buffer_transitions', 0)} transitions")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL TRAINING REPORT")
    print("=" * 70)

    report = engine.learner.get_training_report()
    print(f"\n  Training steps: {report['training_steps']}")
    print(f"  Buffer: {report['buffer_episodes']} episodes, "
          f"{report['buffer_transitions']} transitions")

    if losses:
        print(f"\n  Loss trajectory: "
              f"{losses[0]:.6f} → {losses[-1]:.6f} "
              f"({(losses[-1] - losses[0]) / max(1e-8, losses[0]) * 100:+.1f}%)")

    if val_losses and len(val_losses) >= 3:
        first_val = val_losses[0]
        last_val = val_losses[-1]
        print(f"\n  Val loss trajectory: "
              f"{first_val:.6f} → {last_val:.6f} "
              f"({(last_val - first_val) / max(1e-8, first_val) * 100:+.1f}%)")

    print(f"\n  Goal probability trajectory: "
          f"{goal_probs[0]:.4f} → {goal_probs[-1]:.4f}")
    print(f"  Best goal prob: {max(goal_probs):.4f}")
    print(f"  Goals reached: {sum(1 for p in goal_probs if p > 0.7)}/{len(goal_probs)}")

    if inv_losses:
        print(f"  Inv dyn loss: {inv_losses[0]:.6f} → {inv_losses[-1]:.6f}"
              f"({(inv_losses[-1] - inv_losses[0]) / max(1e-8, inv_losses[0]) * 100:+.1f}%)")

    print(f"\n  Flow distribution: {report.get('flow_distribution', {})}")

    return engine, report


if __name__ == "__main__":
    engine, report = run_extended_training(n_cycles=20, steps_per_cycle=20)
