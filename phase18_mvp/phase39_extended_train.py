"""
Extended closed-loop training with all Phases 34-39 integrated.
Tests whether autonomous flow discovery bootstraps from goal-directed seed.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from phase39_autonomous_flow_discovery import AutonomousFlowEngine
from phase36_behavioral_physics_learning import FlowConditionedWorldModel
from phase31_hierarchical_execution import GoalAttractor


def run_full_stack_training(n_cycles: int = 40, steps_per_cycle: int = 20):
    print("=" * 70)
    print("FULL STACK TRAINING: Phases 34-39")
    print(f"  Cycles: {n_cycles}, Steps/cycle: {steps_per_cycle}")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='full_stack',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )

    engine = AutonomousFlowEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=8,
        flow_dim=4,
        lambda_cost=0.5,
        discovery_interval=3,
        prune_interval=6,
        merge_interval=8,
        train_every_n=5
    )

    # Tracking
    goal_probs = []
    costs = []
    train_losses = []
    val_losses = []
    inv_losses = []
    n_flows_history = []
    discoveries = []
    mergers = []
    prunes = []

    for cycle in range(n_cycles):
        result = engine.run_cycle(
            np.zeros(16),
            n_steps=steps_per_cycle
        )

        goal_probs.append(result['goal_prob'])
        costs.append(result.get('avg_cost', 0))
        n_flows_history.append(result['n_flows'])
        inv_losses.append(result['inv_dyn_loss'])

        discoveries.append(
            result.get('discovery', {}).get('created',
                result.get('discovery', {}) != {} and 0)
            if result.get('discovery') else 0
        )
        mergers.append(
            result.get('merge', {}).get('merged', 0)
            if result.get('merge') else 0
        )
        prunes.append(
            result.get('prune', {}).get('pruned', 0)
            if result.get('prune') else 0
        )

        tr = result.get('training', {})
        if 'train_loss' in tr:
            train_losses.append(tr['train_loss'])
        if 'val_loss' in tr:
            val_losses.append(tr['val_loss'])

        if cycle % 5 == 0 or cycle == n_cycles - 1:
            tr_rep = engine.learner.get_training_report()
            fs = engine.factory.get_stats()
            print(f"\n  Cycle {cycle:3d}:")
            print(f"    GP={result['goal_prob']:.6f}  "
                  f"cost={result.get('avg_cost', 0):.4f}  "
                  f"flows={result['n_flows']}")
            print(f"    Inv dyn: {result['inv_dyn_loss']:.6f}")
            print(f"    Factory: {fs['created']} created, "
                  f"{fs['merged']} merged, {fs['rejected']} rejected")
            if train_losses:
                print(f"    Train loss: {train_losses[-1]:.6f}")
            if val_losses:
                print(f"    Val loss: {val_losses[-1]:.6f}")
            print(f"    Buffer: {tr_rep.get('buffer_episodes', 0)} eps, "
                  f"{tr_rep.get('buffer_transitions', 0)} trans")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = engine.learner.get_training_report()
    fs = engine.factory.get_stats()
    ms = engine.merger.get_stats()
    ps = engine.pruner.get_stats()

    print(f"\n  Training steps: {report['training_steps']}")
    print(f"  Buffer: {report['buffer_episodes']} eps, "
          f"{report['buffer_transitions']} trans")

    if train_losses:
        print(f"\n  Train loss: {train_losses[0]:.6f} → "
              f"{train_losses[-1]:.6f} "
              f"({(train_losses[-1]-train_losses[0])/max(1e-8,train_losses[0])*100:+.1f}%)")

    if val_losses and len(val_losses) >= 3:
        print(f"  Val loss: {val_losses[0]:.6f} → "
              f"{val_losses[-1]:.6f} "
              f"({(val_losses[-1]-val_losses[0])/max(1e-8,val_losses[0])*100:+.1f}%)")

    print(f"\n  GP: {goal_probs[0]:.6f} → {goal_probs[-1]:.6f}  "
          f"best={max(goal_probs):.6f}  mean={np.mean(goal_probs):.6f}")
    print(f"  Cost: {costs[0]:.4f} → {costs[-1]:.4f}")

    if inv_losses:
        print(f"  Inv dyn: {inv_losses[0]:.6f} → {inv_losses[-1]:.6f}")

    print(f"\n  Flow manifold:")
    print(f"    Final flows: {len(engine.manifold.flows)}")
    print(f"    Factory: {fs['created']} created, {fs['merged']} merged")
    print(f"    Merger: {ms['merges']} merges")
    print(f"    Pruner: {ps['pruned']} pruned")

    if discoveries:
        print(f"    Discoveries/cycle: max={max(discoveries)}, "
              f"total={sum(1 for d in discoveries if d > 0)}")

    # Analyze top flows
    ranked = []
    for fid in engine.manifold.flows:
        flow = engine.manifold.flows[fid]
        gps = engine.flow_goal_probs.get(fid, [])
        avg_gp = float(np.mean(gps)) if gps else 0.0
        ranked.append((fid, avg_gp, flow.flow_type.value, flow.stability))
    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 3 flows by goal prob:")
    for fid, gp, ftype, stab in ranked[:3]:
        print(f"    {fid}: GP={gp:.4f}, type={ftype}, stability={stab:.3f}")

    return engine, report


if __name__ == "__main__":
    engine, report = run_full_stack_training(n_cycles=40, steps_per_cycle=20)
