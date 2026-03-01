"""
System Simulation Script
=====================

Симулирует полный цикл работы системы с SafeAutoTuner.

Запуск:
    docker exec ns_core python /app/tests/simulator.py
"""
import asyncio
import random
import sys
sys.path.insert(0, '/app')

from uuid import uuid4
from datetime import datetime


async def simulate_tuner():
    """Симуляция SafeAutoTuner"""
    from application.policies.safe_auto_tuner import get_safe_tuner, TuningMode
    
    tuner = get_safe_tuner()
    
    # Register policy
    tuner.register_policy(
        "GreedyUtilityPolicy",
        {"budget": 10},
        mode=TuningMode.SUGGEST
    )
    
    print("=" * 60)
    print("SIMULATION: SafeAutoTuner")
    print("=" * 60)
    print(f"Initial params: budget=10, mode=SUGGEST\n")
    
    # Simulate 10 cycles with rising regret
    regret_history = []
    
    for cycle in range(1, 11):
        # Simulate rising regret
        current_regret = 0.15 + cycle * 0.035
        regret_history.append(current_regret)
        
        action = tuner.process_cycle(
            policy_name="GreedyUtilityPolicy",
            regret_history=regret_history.copy(),
            current_regret=current_regret
        )
        
        print(f"Cycle {cycle:2d} | Regret: {current_regret:.3f} | "
              f"Action: {action['type']:8s} | "
              f"Params: {action.get('params', action.get('current', {}))}")
        
        # Enable AUTO after cycle 5
        if cycle == 5:
            tuner.set_mode("GreedyUtilityPolicy", TuningMode.AUTO)
            print(f"        >>> Switched to AUTO mode!")
    
    print("\n" + "=" * 60)
    print("Final state:")
    state = tuner.get_state("GreedyUtilityPolicy")
    print(f"  Mode: {state['mode']}")
    print(f"  Params: {state['current_params']}")
    print(f"  Tuning events: {state['tuning_events_count']}")
    print("=" * 60)


async def simulate_full_workflow():
    """Симуляция полного workflow"""
    from application.policies.decision_policies import GreedyUtilityPolicy, ScoredIntent
    from application.policies.arbitration_trace import ArbitrationTrace
    from application.policies.decision_feedback import DecisionFeedback
    from application.policies.safe_auto_tuner import get_safe_tuner, TuningMode
    from application.bulk_engine import BulkExecutionIntent, StateTransitionIntent
    
    print("\n" + "=" * 60)
    print("SIMULATION: Full Workflow")
    print("=" * 60)
    
    # Setup
    tuner = get_safe_tuner()
    tuner.register_policy("GreedyUtilityPolicy", {"budget": 10}, mode=TuningMode.SUGGEST)
    
    policy = GreedyUtilityPolicy()
    feedback = DecisionFeedback()
    
    print("\nSimulating 8 cycles with random goals...\n")
    
    for cycle in range(1, 9):
        # Create random goals (simulating real system)
        goal_count = random.randint(3, 8)
        
        intents = [
            BulkExecutionIntent(
                goal_id=uuid4(),
                transition=StateTransitionIntent(
                    goal_id=uuid4(),
                    from_status="active",
                    to_status="done",
                    reason=f"simulated_cycle_{cycle}",
                    actor="simulator"
                )
            )
            for _ in range(goal_count)
        ]
        
        # Random utility distribution (some high, some low)
        scored = []
        for i, intent in enumerate(intents):
            # Simulate different utilities
            utility = random.uniform(0.1, 0.9)
            
            # Bias: later cycles have more high-utility goals
            if random.random() < 0.3 + cycle * 0.05:
                utility = random.uniform(0.7, 1.0)
            
            scored.append(ScoredIntent(
                intent=intent,
                utility=utility,
                cost=random.uniform(0.1, 0.8),
                risk=random.uniform(0.0, 0.5)
            ))
        
        # Create trace
        trace = ArbitrationTrace(uuid4(), "GreedyUtilityPolicy")
        
        # Policy selection
        budget = 5
        selected = await policy.select(scored, budget=budget, trace=trace)
        
        # Simulate "execution" - mark some as completed
        execution_results = []
        for s in selected:
            # 70% success rate
            success = random.random() < 0.7
            execution_results.append({
                "goal_id": str(s.intent.goal_id)[:8],
                "success": success,
                "utility": s.utility
            })
        
        # Feedback analysis
        analysis = feedback.analyze(trace.get_records())
        
        # Calculate regret (simulated - compare selected vs rejected)
        all_utilities = [s.utility for s in scored]
        selected_utilities = [s.utility for s in selected]
        rejected_utilities = [u for u in all_utilities if u not in selected_utilities]
        
        if rejected_utilities:
            potential_lost = max(rejected_utilities) - min(selected_utilities) if selected_utilities else 0
            regret_ratio = max(0, potential_lost) / max(sum(selected_utilities), 0.01)
        else:
            regret_ratio = 0
        
        # SafeTuner processing
        action = tuner.process_cycle(
            policy_name="GreedyUtilityPolicy",
            regret_history=[regret_ratio],
            current_regret=regret_ratio
        )
        
        # Enable AUTO after cycle 4
        if cycle == 4:
            tuner.set_mode("GreedyUtilityPolicy", TuningMode.AUTO)
        
        print(f"Cycle {cycle}: "
              f"Goals={goal_count}, "
              f"Selected={len(selected)}, "
              f"Regret={regret_ratio:.2f}, "
              f"Action={action['type']}")
    
    print("\n" + "=" * 60)
    print("Final Workflow State:")
    
    state = tuner.get_state("GreedyUtilityPolicy")
    print(f"  Policy mode: {state['mode']}")
    print(f"  Params: {state['current_params']}")
    print(f"  Events: {state['tuning_events_count']}")
    print("=" * 60)


async def simulate_skills():
    """Симуляция работы со Skills"""
    print("\n" + "=" * 60)
    print("SIMULATION: Skills Execution")
    print("=" * 60)
    
    # Simulated skills available
    skills = {
        "web_research": {"cost": 0.3, "success_rate": 0.8},
        "write_file": {"cost": 0.2, "success_rate": 0.9},
        "ask_user": {"cost": 0.1, "success_rate": 0.95},
    }
    
    print("\nAvailable Skills:")
    for name, props in skills.items():
        print(f"  - {name}: cost={props['cost']}, success={props['success_rate']}")
    
    # Simulate goal execution with skills
    print("\nSimulating 5 goals with skill execution...\n")
    
    for goal_num in range(1, 6):
        # Random skill selection
        skill_name = random.choice(list(skills.keys()))
        skill = skills[skill_name]
        
        # Simulate execution
        success = random.random() < skill["success_rate"]
        
        print(f"Goal {goal_num}: Using {skill_name} -> "
              f"{'SUCCESS' if success else 'FAILED'} "
              f"(expected: {skill['success_rate']:.0%})")
    
    print("\n" + "=" * 60)
    print("Skills simulation complete!")
    print("=" * 60)


async def main():
    """Main entry point"""
    print("\n" + "=" * 70)
    print(" " * 15 + "AI-OS SYSTEM SIMULATOR")
    print("=" * 70)
    
    # Run simulations
    await simulate_tuner()
    await simulate_full_workflow()
    await simulate_skills()
    
    print("\n" + "=" * 70)
    print("ALL SIMULATIONS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
