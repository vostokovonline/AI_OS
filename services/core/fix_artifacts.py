"""
Fix KNOWLEDGE artifacts with wrong content_kind
Convert content_kind from 'file' to 'db' for KNOWLEDGE artifacts
"""

import asyncio
from sqlalchemy import select, update
from database import AsyncSessionLocal
from models import Artifact
import json

async def fix_knowledge_artifacts():
    """Fix KNOWLEDGE artifacts that have content_kind='file' instead of 'db'"""

    async with AsyncSessionLocal() as db:
        # Find all KNOWLEDGE artifacts with content_kind='file'
        stmt = select(Artifact).where(
            Artifact.type == "KNOWLEDGE",
            Artifact.content_kind == "file",
            Artifact.skill_name == "core.web_research"
        )
        result = await db.execute(stmt)
        artifacts = result.scalars().all()

        print(f"Found {len(artifacts)} KNOWLEDGE artifacts with content_kind='file'")

        fixed_count = 0
        for artifact in artifacts:
            # Update content_kind to 'db'
            artifact.content_kind = "db"
            fixed_count += 1

            print(f"✅ Fixed artifact {artifact.id} (created: {artifact.created_at})")

        # Commit changes
        await db.commit()

        print(f"\n✅ Successfully fixed {fixed_count} artifacts")

        # Verify the fix
        verify_stmt = select(Artifact).where(
            Artifact.type == "KNOWLEDGE",
            Artifact.content_kind == "db",
            Artifact.skill_name == "core.web_research"
        )
        verify_result = await db.execute(verify_stmt)
        db_artifacts = verify_result.scalars().all()

        print(f"\n📊 Verification: {len(db_artifacts)} KNOWLEDGE artifacts now have content_kind='db'")

        # Count remaining file artifacts
        file_stmt = select(Artifact).where(
            Artifact.type == "KNOWLEDGE",
            Artifact.content_kind == "file"
        )
        file_result = await db.execute(file_stmt)
        file_artifacts = file_result.scalars().all()

        if file_artifacts:
            print(f"⚠️  Still {len(file_artifacts)} KNOWLEDGE artifacts with content_kind='file' (non-web_research)")

if __name__ == "__main__":
    asyncio.run(fix_knowledge_artifacts())
