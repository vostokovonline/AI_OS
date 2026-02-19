#!/usr/bin/env python3
"""
Fix depth_level for all existing goals based on parent_id hierarchy.

This script recalculates depth_level for all goals by traversing the parent-child hierarchy.
"""

import asyncio
from database import AsyncSessionLocal
from models import Goal
from sqlalchemy import select, update, func


async def calculate_depth_level(goal_id: str, db, visited=None) -> int:
    """
    Recursively calculate depth_level for a goal.

    Args:
        goal_id: Goal ID
        db: Database session
        visited: Set of visited goal IDs to prevent infinite loops

    Returns:
        Calculated depth level
    """
    if visited is None:
        visited = set()

    if goal_id in visited:
        return 0  # Prevent infinite loops

    visited.add(goal_id)

    stmt = select(Goal).where(Goal.id == goal_id)
    result = await db.execute(stmt)
    goal = result.scalar_one_or_none()

    if not goal:
        return 0

    if not goal.parent_id:
        return 0  # Root goal = L1

    # Recursively calculate parent's depth
    parent_depth = await calculate_depth_level(str(goal.parent_id), db, visited)
    return parent_depth + 1


async def fix_all_depth_levels():
    """Fix depth_level for all goals in the database."""

    print("🔧 Starting depth_level fix...")

    async with AsyncSessionLocal() as db:
        # Get all goals
        stmt = select(Goal)
        result = await db.execute(stmt)
        all_goals = result.scalars().all()

        print(f"📊 Found {len(all_goals)} goals to fix")

        # Calculate new depth levels
        updates = []
        for goal in all_goals:
            new_depth = await calculate_depth_level(goal.id, db)

            if goal.depth_level != new_depth:
                updates.append({
                    "id": goal.id,
                    "title": goal.title,
                    "old_depth": goal.depth_level,
                    "new_depth": new_depth
                })
                print(f"  📌 {goal.title[:50]}... {goal.depth_level} → {new_depth}")

        # Apply updates
        if updates:
            print(f"\n💾 Applying {len(updates)} updates...")

            for update_data in updates:
                stmt = (
                    update(Goal)
                    .where(Goal.id == update_data["id"])
                    .values(depth_level=update_data["new_depth"])
                )
                await db.execute(stmt)

            await db.commit()
            print(f"✅ Updated {len(updates)} goals")
        else:
            print("✅ All depth_levels are already correct!")

        # Show distribution
        print("\n📊 Depth level distribution:")
        stmt = select(Goal.depth_level, func.count(Goal.id)).group_by(Goal.depth_level)
        result = await db.execute(stmt)
        for depth, count in result.all():
            print(f"  Level {depth}: {count} goals")


if __name__ == "__main__":
    asyncio.run(fix_all_depth_levels())
