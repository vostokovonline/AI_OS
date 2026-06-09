"""
Extended Phase 40 continuous self-organization run.
Tests whether the system converges over 500+ continuous steps.
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from phase40_self_organizing_geometry import SelfOrganizingEngine
from phase36_behavioral_physics_learning import FlowConditionedWorldModel
from phase31_hierarchical_execution import GoalAttractor


def long_run(n_steps: int = 500, report_interval: int = 50):
    print("=" * 70)
    print("PHASE 40 EXTENDED SELF-ORGANIZATION RUN")
    print(f"  Steps: {n_steps}")
    print("=" * 70)

    wm = FlowConditionedWorldModel(
        event_dim=32, latent_dim=16, belief_dim=64,
        action_dim=16, flow_embed_dim=8
    )

    goal = GoalAttractor(
        goal_id='phase40_long',
        attractor_state=np.ones(16) * 1.5,
        basin_radius=2.0, priority=0.9,
        decay_rate=0.01, success_criteria={'type': 'achievable'}
    )

    engine = SelfOrganizingEngine(
        world_model=wm,
        goal=goal,
        n_initial_flows=6,
        flow_dim=4,
        lambda_cost=0.5,
        train_interval=5
    )

    z = np.zeros(16)
    h = np.zeros(wm.belief_dim)
    h = wm.gru_step(h, z)

    gp_history = []
    flow_history = []
    loss_history = []

    for step in range(n_steps):
        result = engine.step(z, h)

        z = result['z_after'].copy()
        h = wm.gru_step(h, result['z_after'])

        gp_history.append(result['goal_prob'])
        flow_history.append(result['n_flows'])

        if step % 20 == 0 and step > 0:
            engine.record_episode()

        if step % report_interval == 0 or step == n_steps - 1:
            tr = engine.learner.get_training_report()
            eco = engine.ecology.get_stats()
            cem = engine.cem.get_stats()
            recent_gp = gp_history[-min(report_interval, len(gp_history)):]
            print(f"\n  Step {step:4d}:")
            print(f"    GP: μ={np.mean(recent_gp):.6f}  "
                  f"max={max(recent_gp):.6f}  current={gp_history[-1]:.6f}")
            print(f"    Flows: {result['n_flows']}  "
                  f"Ecology: {eco['births']}B/{eco['deaths']}D")
            print(f"    CEM: μ={cem['mean'][0]:.4f}...  "
                  f"σ={cem['std'][0]:.4f}")
            print(f"    Training: {tr['training_steps']} steps  "
                  f"Buffer: {tr['buffer_episodes']} eps")
            if 'loss_improvement' in tr and tr.get('loss_improvement', 0) != 0:
                print(f"    Loss improvement: {tr['loss_improvement']*100:.2f}%")
            if 'val_loss' in tr and tr['val_loss'] != float('inf'):
                print(f"    Val loss: {tr['val_loss']:.6f}")

    # Final report
    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    tr = engine.learner.get_training_report()
    eco = engine.ecology.get_stats()
    drift = engine.drift.get_stats()
    cem = engine.cem.get_stats()

    print(f"\n  Steps: {n_steps}")
    print(f"  GP: {gp_history[0]:.6f} → {gp_history[-1]:.6f}  "
          f"mean={np.mean(gp_history):.6f}  max={max(gp_history):.6f}")
    print(f"  GP last 100: μ={np.mean(gp_history[-100:]):.6f}  "
          f"trend={gp_history[-1]-gp_history[-101] if len(gp_history) > 101 else 0:.6f}")

    print(f"\n  Training: {tr['training_steps']} steps, "
          f"{tr['buffer_episodes']} eps, {tr['buffer_transitions']} trans")
    if 'loss_improvement' in tr:
        print(f"  Loss improvement: {tr['loss_improvement']*100:.2f}%")

    print(f"\n  Flows: initial=6, final={len(engine.manifold.flows)}")
    print(f"  Ecology: {eco['births']} births, {eco['deaths']} deaths")
    print(f"  Drift: {drift['total_drift']:.4f} total displacement")

    print(f"\n  CEM: mean=[{cem['mean'][0]:.3f}, {cem['mean'][1]:.3f}, ...], "
          f"std={cem['std'][0]:.3f}")

    # Top flows
    ranked = []
    for fid, flow in engine.manifold.flows.items():
        ranked.append((fid, flow.goal_alignment, flow.stability, flow.flow_type.value))
    ranked.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 3 flows:")
    for fid, ga, stab, ftype in ranked[:3]:
        print(f"    {fid}: alignment={ga:.4f}, stability={stab:.3f}, type={ftype}")

    return engine


if __name__ == "__main__":
    engine = long_run(n_steps=500, report_interval=50)
