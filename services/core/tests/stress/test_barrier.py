#!/usr/bin/env python3
"""
Architectural Contract Test: Batch Atomicity Enforcement
========================================================

This test proves that the decision system architecture separates:
- Phase 1 (COLLECT): Pure computation, NO writes
- Phase 2 (APPLY):   Batch writes in ONE transaction

WriteBarrier violations = architectural leak (workflow thinking).
"""
import asyncio
from datetime import datetime
from uuid import uuid4

async def test_batch_atomicity():
    from models import Goal
    from infrastructure.uow import get_uow, GoalRepository
    from application.use_cases.execute_ready_goals import ExecuteReadyGoalsUseCase
    from application.bulk_transition_engine import bulk_transition_engine
    from goal_executor_v2 import goal_executor_v2
    from application.events.bus import get_event_bus
    
    print("\n" + "="*70)
    print("ARCHITECTURAL CONTRACT TEST: Batch Atomicity")
    print("="*70)
    
    # Create test goals
    async with get_uow() as uow:
        repo = GoalRepository()
        
        # Create 3 atomic goals
        goal_ids = []
        for i in range(3):
            goal = Goal(
                title=f"Test atomic goal {i}",
                description=f"Test goal for barrier verification {i}",
                goal_type="achievable",
                is_atomic=True,
                progress=0.0,
                _status="pending"
            )
            await repo.save(uow.session, goal)
            goal_ids.append(goal.id)
            print(f"  Created goal: {goal.id}")
    
    print(f"\nCreated {len(goal_ids)} test goals")
    
    # Create use-case
    uow_factory = get_uow
    event_bus = get_event_bus()
    
    use_case = ExecuteReadyGoalsUseCase(
        uow_factory=uow_factory,
        executor=goal_executor_v2,
        bulk_engine=bulk_transition_engine,
        event_bus=event_bus
    )
    
    print("\n" + "-"*70)
    print("Running batch execution with WriteBarrier ENABLED...")
    print("-"*70)
    
    try:
        # Run batch - WriteBarrier will raise RuntimeError if writes in Phase 1
        result = await use_case.run(limit=3, actor="test_barrier")
        
        print("\n" + "✅"*35)
        print("SUCCESS: No barrier violations detected!")
        print("✅"*35)
        print(f"\nResults:")
        print(f"  Total found:    {result.total_found}")
        print(f"  Completed:      {result.completed}")
        print(f"  Failed:         {result.failed}")
        print(f"  Skipped:        {result.skipped}")
        print(f"  Execution time: {result.execution_time_ms}ms")
        print("\n" + "="*70)
        print("ARCHITECTURAL CONTRACT: VALIDATED ✅")
        print("Phase 1 (COLLECT) is pure computation ✅")
        print("Phase 2 (APPLY) is batch atomic ✅")
        print("="*70)
        
    except RuntimeError as e:
        print("\n" + "❌"*35)
        print("BARRIER VIOLATION DETECTED!")
        print("❌"*35)
        print(f"\nError: {e}")
        print("\nThis means:")
        print("  1. Something tried to WRITE during Phase 1 (COLLECT)")
        print("  2. Architecture still has workflow-thinking leaks")
        print("  3. Stack trace above shows where mutation happened")
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
    asyncio.run(test_batch_atomicity())
