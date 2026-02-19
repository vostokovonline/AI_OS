"""
Фикс для целей с NULL updated_at
Устанавливает updated_at = created_at для целей где updated_at IS NULL
"""

import asyncio
from sqlalchemy import select, update
from database import AsyncSessionLocal
from models import Goal

async def fix_null_updated_at():
    """Fix goals with NULL updated_at"""

    async with AsyncSessionLocal() as db:
        # Find goals with NULL updated_at
        stmt = select(Goal).where(Goal.updated_at.is_(None))
        result = await db.execute(stmt)
        goals = result.scalars().all()

        print(f"Found {len(goals)} goals with NULL updated_at")

        if not goals:
            print("✅ No goals need fixing")
            return

        # Update each goal
        for goal in goals:
            if goal.created_at:
                goal.updated_at = goal.created_at
                print(f"✅ Fixed goal {goal.id}: {goal.title[:50]}")
            else:
                now = datetime.utcnow()
                goal.updated_at = now
                goal.created_at = now
                print(f"✅ Fixed goal {goal.id}: {goal.title[:50]} (set to now)")

        await db.commit()

        print(f"\n✅ Successfully fixed {len(goals)} goals")

        # Verify
        verify_stmt = select(Goal).where(Goal.updated_at.is_(None))
        verify_result = await db.execute(verify_stmt)
        remaining = verify_result.scalars().all()

        if remaining:
            print(f"⚠️  Still {len(remaining)} goals with NULL updated_at")
        else:
            print("✅ All goals now have updated_at set")

if __name__ == "__main__":
    from datetime import datetime
    asyncio.run(fix_null_updated_at())
