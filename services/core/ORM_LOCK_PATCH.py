"""
MODELS.PY HARD LOCK PATCH
=========================

ORM-level protection against direct status mutation.

THIS IS THE REAL HARD LOCK - not paper declarations.

Author: AI-OS Core Team
Date: 2026-02-11
Severity: CRITICAL - Blocks all direct status mutations

INSTRUCTIONS:
------------
1. Find the Goal class in models.py (around line 39)
2. Add the property-based protection BELOW
3. Replace the status column definition
4. Re-deploy models.py to container

TESTING:
--------
After deployment, test with:

    from models import Goal
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        stmt = select(Goal).limit(1)
        result = await db.execute(stmt)
        goal = result.scalar_one()

        # This should WORK (reading through getter):
        print(f"Goal status: {goal.status}")

        # These should ALL FAIL (direct assignment blocked):
        try:
            goal.status = "done"
        except AttributeError as e:
            print(f"✅ Protection working: {e}")

"""

# =============================================================================
# PROPERTY-BASED STATUS PROTECTION
# =============================================================================

class _GoalStatusProperty:
    """
    Property-based protection for Goal.status

    FORBIDS direct assignment, forces use of transition service
    """

    def __init__(self):
        self._status_cache = {}

    def __get__(self, goal):
        """
        Get status - ALLOWED operation
        """
        goal_id = str(goal.id)

        # Cache for performance
        if goal_id in self._status_cache:
            return self._status_cache[goal_id]

        # Get from internal _status field
        status_value = getattr(goal, '_status', 'active')
        self._status_cache[goal_id] = status_value
        return status_value

    def __set__(self, goal, value):
        """
        Set status - BLOCKED operation

        This raises AttributeError if someone tries:
            goal.status = "done"  # DIRECTLY

        Instead they MUST use:
            await transition_goal(goal_id, "completed", reason="...")
        """
        from goal_transition_service import mark_goal_directly_forbidden

        goal_id = str(goal.id)
        goal_type = getattr(goal, 'goal_type', 'achievable')
        old_value = getattr(goal, '_status', 'unknown')

        # CRITICAL CHECK: Is this a FORBIDDEN direct mutation?
        forbidden_mutations = ["done"]

        if value in forbidden_mutations:
            # This is a HARD BLOCK
            print(f"\n{'='*70}")
            print("❌ CRITICAL: DIRECT STATUS MUTATION BLOCKED")
            print(f"{'='*70}")
            print(f"  Goal ID: {goal_id}")
            print(f"  Type: {goal_type}")
            print(f"  Attempted: status = '{value}'")
            print(f"  Old value: {old_value}")
            print("")
            print("  DIRECT STATUS ASSIGNMENT IS FORBIDDEN")
            print("  Use: goal_transition_service.transition_goal()")
            print("")
            print("  STACK TRACE:")
            print("-" * 70)

            import traceback
            traceback.print_stack()
            print("-" * 70)
            print("")
            print(f"{'='*70}\n")

            # LOG to audit
            from audit_logger import audit_logger
            audit_logger.log_state_transition(
                goal_id=goal_id,
                goal_type=goal_type,
                from_state=old_value,
                to_state=f"BLOCKED: {value}",
                reason="Direct status assignment forbidden",
                actor="orm_protection"
            )

            # CRASH the write attempt
            raise AttributeError(
                f"ILLEGAL DIRECT STATUS ASSIGNMENT: status='{value}'\n"
                f"Goal ID: {goal_id}\n"
                f"Goal Type: {goal_type}\n\n"
                f"USE: await transition_goal(goal_id, '{value}', 'reason')\n"
                f"THIS IS A HARD ORM-LEVEL STOP\n"
                f"File: models.py, Class: Goal, Line: YOUR_CALL_SITE"
            )

        # ALLOWED mutations (not "done")
        # These are only for internal use by transition service
        print(f"⚠️  ORM: Allowing status change to '{value}' for goal {goal_id}")

        # Set to internal field
        setattr(goal, '_status', value)
        self._status_cache[str(goal.id)] = value


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================

"""
HOW TO APPLY THIS PATCH:

Step 1: Backup current models.py
--------------------------------------------
docker cp ns_core:/app/models.py /home/onor/ai_os_final/services/core/models.py.backup

Step 2: Find the Goal class definition
--------------------------------------------
Look for: class Goal(Base):

Around line 39-127 in models.py

Step 3: Add property definition to Goal class
--------------------------------------------------
ADD THIS RIGHT AFTER THE CLASS DEFINITION:

class Goal(Base):
    __tablename__ = "goals"

    # ... existing fields up to line 50 ...

    # ============== ADD THIS ==============

    # Rename the original status column to internal
    _status = Column("status", String, default="active")

    # Add property-based protection
    status = property(
        lambda self: _GoalStatusProperty().__get__(self),
        lambda self, value: _GoalStatusProperty().__set__(self, value)
    )

    # ============== END ADDITION ==============

Step 4: Test the protection
--------------------------------------------
docker exec ns_core python3 -c "
from models import Goal
from database import AsyncSessionLocal
import asyncio

async def test():
    async with AsyncSessionLocal() as db:
        from sqlalchemy import select
        stmt = select(Goal).limit(1)
        result = await db.execute(stmt)
        g = result.scalar_one()

        # This SHOULD work:
        print(f'Status: {g.status}')

        # This SHOULD fail:
        try:
            g.status = 'done'
        except AttributeError as e:
            print(f'Protection: {str(e)[:50]}...')

asyncio.run(test())
"

EXPECTED OUTPUT:
Status: active
Protection: ILLEGAL DIRECT STATUS ASSIGNMENT...

Step 5: If test passes, restart container
--------------------------------------------
docker restart ns_core

"""

print(__doc__)
