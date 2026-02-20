"""
Add Goal Status Transitions Table Migration
=====================================
Adds audit table for Phase 1 (Controlled Evolution)
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://ns_admin:ns_password@ns_postgres:5432/ns_core_db")

async def upgrade():
    """Add goal_status_transitions table"""
    engine = create_async_engine(DATABASE_URL)
    
    # Read SQL file
    with open('/app/migrations/add_goal_transitions_table.sql', 'r') as f:
        sql = f.read()

    # Execute migration
    async with engine.begin() as conn:
        await conn.execute(text(sql))

    print("✅ Migration applied: goal_status_transitions table created")
    return True


async def downgrade():
    """Remove audit table if needed"""
    engine = create_async_engine(DATABASE_URL)
    
    async with engine.begin() as conn:
        await conn.execute(text("DROP TABLE IF EXISTS goal_status_transitions"))

    print("⚠️  Downgrade: goal_status_transitions table dropped")
    return True


if __name__ == "__main__":
    import asyncio
    print("Running migration...")
    asyncio.run(upgrade())
