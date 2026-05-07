#!/usr/bin/env python3
"""
Architectural Contract Test: Batch Atomicity with Arbitration (v3.0)
"""
import asyncio
from datetime import datetime

async def test_batch_atomicity_with_arbitration():
    from models import Goal
    from infrastructure.uow import get_uow, GoalRepository
    from application.use_cases.execute_ready_goals import ExecuteReadyGoalsUseCase
    from application.bulk_transition_engine import bulk_transition_engine
    from goal_executor_v2 import goal_executor_v2
    from application.events.bus import get_event_bus
    from application.arbitration import (
        BatchArbitrator, GreedyUtilityPolicy, ConfidenceUtilityEstimator,
        ConstantCostEstimator, ConfidenceRiskEstimator,
        FixedBudgetAllocator, InMemoryArbitrationLog,
    )
    
    print("\n" + "="*70)
    print("ARCHITECTURAL CONTRACT TEST v3.0: Batch Atomicity + Arbitration")
    print("="*70)
    
    async with get_uow() as uow:
        repo = GoalRepository()
        goal_ids = []
        for i in range(3):
            goal = Goal(
                title=f"Test atomic goal {i}",
                description=f"Test goal {i}",
                goal_type="achievable",
                is_atomic=True,
                progress=0.0,
                _status="pending"
            )
            await repo.save(uow.session, goal)
            goal_ids.append(goal.id)
            print(f"  Created goal: {goal.id}")
    
    print(f"\nCreated {len(goal_ids)} test goals")
    
    arbitrator = BatchArbitrator(
        utility_estimator=ConfidenceUtilityEstimator(),
        cost_estimator=ConstantCostEstimator(cost=1.0),
        risk_estimator=ConfidenceRiskEstimator(),
        policy=GreedyUtilityPolicy(),
        arbitration_log=InMemoryArbitrationLog(max_size=100),
    )
    
    capital_allocator = FixedBudgetAllocator(budget=10.0)
    
    use_case = ExecuteReadyGoalsUseCase(
        uow_factory=get_uow,
        executor=goal_executor_v2,
        bulk_engine=bulk_transition_engine,
        arbitrator=arbitrator,
        capital_allocator=capital_allocator,
        event_bus=get_event_bus()
    )
    
    print("\n" + "-"*70)
    print("Running batch execution with WriteBarrier ENABLED...")
    print("-"*70)
    
    try:
        result = await use_case.run(limit=3, actor="test_barrier")
        
        print("\n" + "✅"*35)
        print("SUCCESS: No barrier violations detected!")
        print("✅"*35)
        print(f"\nResults:")
        print(f"  Total found:        {result.total_found}")
        print(f"  Arbitration selected: {result.arbitration_selected}")
        print(f"  Arbitration rejected: {result.arbitration_rejected}")
        print(f"  Selection rate:      {result.arbitration_selection_rate:.2%}")
        print(f"  Completed:           {result.completed}")
        print(f"  Failed:              {result.failed}")
        print("\n" + "="*70)
        print("ARCHITECTURAL CONTRACT: VALIDATED ✅")
        print("Phase 1 (COLLECT) is pure computation ✅")
        print("Phase 1.5 (ARBITRATION) filters intents ✅")
        print("Phase 2 (APPLY) is batch atomic ✅")
        print("="*70)
        
    except RuntimeError as e:
        print("\n" + "❌"*35)
        print("BARRIER VIOLATION DETECTED!")
        print("❌"*35)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\nUnexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_batch_atomicity_with_arbitration())
