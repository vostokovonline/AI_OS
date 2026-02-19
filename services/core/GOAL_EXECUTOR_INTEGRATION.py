"""
GOAL EXECUTOR - INTEGRATION PATCH
=================================

This file shows how to integrate invariants and compatibility layer
into goal_executor.py to prevent ontology violations.

Instructions: Apply changes marked with # PATCH to goal_executor.py

Author: AI-OS Core Team
Date: 2026-02-11
"""

# =============================================================================
# STEP 1: Add imports to goal_executor.py (after line 13)
# =============================================================================

# ADD THESE IMPORTS:
from compatibility import wrap_goal, LifecycleState, CompletionPolicy
from invariants import validate_and_raise, validate_transition_and_raise, GoalInvariantViolation
from outcome_validator import validate_goal_outcome


# =============================================================================
# STEP 2: Modify execution completion logic (around line 334-370)
# =============================================================================

"""
CURRENT CODE (WRONG):
```python
# Around line 334-370 in goal_executor.py

            # 🔑 ARTIFACT LAYER v1 - Check artifacts for atomic goals
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
                result = await db.execute(stmt)
                goal = result.scalar_one_or_none()

                if goal:
                    from artifact_registry import artifact_registry

                    # Проверяем артефакты
                    artifact_check = await artifact_registry.check_goal_artifacts(goal_id)

                    print(f"📦 Artifact check: {artifact_check}")

                    # Для atomic goals (L3): без passed artifacts → incomplete
                    if goal.is_atomic and not artifact_check.get("goal_complete"):
                        print(f"⚠️ Atomic goal has no passed artifacts - marking as incomplete")
                        goal.status = "incomplete"  # ← WRONG STATUS
                        goal.progress = 0.9
                        await db.commit()
```

PATCHED CODE (CORRECT):
```python
# REPLACE with invariant-protected version:

            # 🔑 ARTIFACT LAYER v1 + INVARIANTS - Check completion properly
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
                result = await db.execute(stmt)
                goal = result.scalar_one_or_none()

                if goal:
                    # PATCH: Use compatibility wrapper
                    goal_view = wrap_goal(goal)

                    from artifact_registry import artifact_registry

                    # Проверяем артефакты
                    artifact_check = await artifact_registry.check_goal_artifacts(goal_id)

                    print(f"📦 Artifact check: {artifact_check}")

                    # PATCH: Validate invariants BEFORE marking done
                    # Invariant I3: Atomic goals MUST have artifacts to be done
                    if goal.is_atomic and not artifact_check.get("goal_complete"):
                        print(f"⚠️ Atomic goal has no passed artifacts - marking as incomplete")

                        # PATCH: Use correct lifecycle_state
                        goal_view.lifecycle_state = LifecycleState.ACTIVE  # NOT "done"!
                        goal.progress = 0.9
                        await db.commit()

                        # PATCH: Return here, DON'T mark as done
                        return {
                            "status": "incomplete",
                            "reason": "No passed artifacts",
                            "goal_id": goal_id
                        }

                    # PATCH: For atomic goals with artifacts, use outcome validation
                    if goal.is_atomic and artifact_check.get("goal_complete"):
                        # Use outcome validator for proper evaluation
                        outcome = await validate_goal_outcome(goal_id)

                        print(f"✅ Outcome validation: {outcome['evaluation_state']}")

                        # Set lifecycle based on outcome
                        if outcome.get("passed"):
                            goal_view.lifecycle_state = LifecycleState.COMPLETED
                        else:
                            goal_view.lifecycle_state = LifecycleState.ACTIVE

                        goal.progress = outcome.get("progress", 1.0)
                        await db.commit()

                        # PATCH: Log that we used NEW MODEL
                        print(f"✅ Goal '{goal.title}' completed with NEW ontology")
                        print(f"   Lifecycle: {goal_view.lifecycle_state}")
                        print(f"   Evaluation: {outcome.get('evaluation_state')}")

                    # PATCH: For non-atomic goals, check completion policy
                    if not goal.is_atomic:
                        outcome = await validate_goal_outcome(goal_id)

                        if outcome.get("passed"):
                            if goal_view.completion_policy == CompletionPolicy.AGGREGATE_CHILDREN:
                                goal_view.lifecycle_state = LifecycleState.COMPLETED
                            elif goal_view.completion_policy == CompletionPolicy.TREND_BASED:
                                goal_view.lifecycle_state = LifecycleState.ONGOING
                            elif goal_view.completion_policy == CompletionPolicy.SCALAR_ALIGNMENT:
                                goal_view.lifecycle_state = LifecycleState.PERMANENT
                        else:
                            goal_view.lifecycle_state = LifecycleState.ACTIVE

                        goal.progress = outcome.get("progress", 0.0)
                        await db.commit()

                        print(f"✅ Non-atomic goal completed with NEW ontology")
                        print(f"   Type: {goal.goal_type}")
                        print(f"   Policy: {goal_view.completion_policy}")
                        print(f"   Lifecycle: {goal_view.lifecycle_state}")
```
"""


# =============================================================================
# STEP 3: Add invariant checking to mark_goal_done (if exists)
# =============================================================================

"""
IF goal_executor.py has mark_goal_done() or similar method, ADD:

```python
async def mark_goal_done(self, goal_id: str, approved_by: str = None) -> Dict:
    '''Mark goal as done WITH INVARIANT CHECKING'''
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            raise ValueError("Goal not found")

        # PATCH: Use compatibility wrapper
        goal_view = wrap_goal(goal)

        # PATCH: CHECK INVARIANTS FIRST
        try:
            validate_and_raise(goal_view)
        except GoalInvariantViolation as e:
            print(f"❌ INVARIANT VIOLATION PREVENTED:")
            print(str(e))
            raise

        # PATCH: Check if completion allowed
        allowed, reason = goal_view.can_mark_completed()
        if not allowed:
            print(f"⚠️ COMMISSION PREVENTED:")
            print(f"  Reason: {reason}")
            raise ValueError(f"Cannot mark goal as done: {reason}")

        # PATCH: Use correct lifecycle_state based on policy
        if goal_view.completion_policy == CompletionPolicy.TREND_BASED:
            # Continuous: never "done"
            print(f"⚠️ Continuous goal cannot be marked as 'done'")
            print(f"   Use lifecycle_state='ongoing' instead")
            raise ValueError("Continuous goals use ongoing state, not done")

        elif goal_view.completion_policy == CompletionPolicy.SCALAR_ALIGNMENT:
            # Directional: never "done"
            print(f"⚠️ Directional goal cannot be marked as 'done'")
            print(f"   Use lifecycle_state='permanent' instead")
            raise ValueError("Directional goals use permanent state, not done")

        # Allow completion for achievable/atomic
        goal_view.lifecycle_state = LifecycleState.COMPLETED

        if approved_by:
            # Record manual approval
            from models import GoalCompletionApproval
            approval = GoalCompletionApproval(
                goal_id=goal.id,
                approved_by=approved_by,
                approved_at=datetime.now()
            )
            db.add(approval)

        await db.commit()

        return {
            "status": "done",
            "lifecycle_state": goal_view.lifecycle_state.value,
            "goal_id": goal_id
        }
```
"""


# =============================================================================
# STEP 4: Add scan endpoint to main.py
# =============================================================================

"""
ADD to main.py:

```python
@app.get("/admin/scan-invariants")
async def scan_goal_invariants():
    '''Scan all goals for invariant violations'''
    from invariants import scan_all_goals, print_violation_report
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        scan_result = await scan_all_goals(db)

    print_violation_report(scan_result)

    return {
        "status": "ok",
        "scan_result": scan_result
    }


@app.post("/admin/fix-invariants/{goal_id}")
async def fix_goal_invariant(goal_id: str, correction: dict):
    '''Manually fix a goal's invariant violation'''
    from compatibility import wrap_goal, LifecycleState
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")

        goal_view = wrap_goal(goal)

        # Apply correction
        new_lifecycle = correction.get("lifecycle_state")
        if new_lifecycle:
            goal_view.lifecycle_state = LifecycleState(new_lifecycle)
            print(f"✅ Fixed {goal.title}: lifecycle_state → {new_lifecycle_state}")

        await db.commit()

        return {
            "status": "fixed",
            "goal_id": goal_id,
            "corrections_applied": correction
        }


@app.post("/goals/{goal_id}/measure-state")
async def measure_goal_state(
    goal_id: str,
    metrics: dict,
    measured_at: str = None
):
    '''Record state measurement for outcome validation'''
    from outcome_validator import record_goal_state
    from datetime import datetime

    if measured_at:
        measured_at = datetime.fromisoformat(measured_at)

    try:
        result = await record_goal_state(
            goal_id=goal_id,
            current_metrics=metrics,
            measured_at=measured_at
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```
"""


# =============================================================================
# STEP 5: Run migration script
# =============================================================================

"""
RUN IN CONTAINER:

```bash
# Copy migration to container
docker cp add_lifecycle_model.sql ns_core:/tmp/

# Execute migration
docker exec -i ns_postgres psql -U ns_admin -d ns_core_db -f /tmp/add_lifecycle_model.sql

# Verify migration
docker exec ns_postgres psql -U ns_admin -d ns_core_db -c "\d goals"
```

EXPECTED OUTPUT:
- New columns added (lifecycle_state, evaluation_state, etc.)
- New tables created (goal_states, tasks)
- Constraint added (check_goal_type_completion_state)
- View created (goals_needing_migration)
"""


# =============================================================================
# VERIFICATION STEPS
# =============================================================================

"""
AFTER APPLYING PATCHES:

1. Test compatibility layer:
   python3 -c "from compatibility import wrap_goal, LifecycleState; print('✅ Compatibility OK')"

2. Test invariants:
   python3 -c "from invariants import GoalInvariants; print('✅ Invariants OK')"

3. Test outcome validator:
   python3 -c "from outcome_validator import outcome_validator; print('✅ Outcome validator OK')"

4. Run invariant scan:
   curl http://localhost:8000/admin/scan-invariants

5. Fix existing violations manually or with API

6. Validate constraint after fixes:
   ALTER TABLE goals VALIDATE CONSTRAINT check_goal_type_completion_state;
"""

print(__doc__)
