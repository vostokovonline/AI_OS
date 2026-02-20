#!/usr/bin/env python3
"""
Retroactive Artifact Re-verification Script

Re-verifies all failed artifacts with the new inline content detection logic.

Usage:
    docker exec ns_core python /app/scripts/retroverify_artifacts.py [--dry-run]
"""
import asyncio
import sys
import argparse

sys.path.insert(0, '/app')

from sqlalchemy import select
from database import AsyncSessionLocal
from models import Artifact
from artifact_verifier import artifact_verifier
from logging_config import get_logger

logger = get_logger(__name__)


async def retroverify_artifacts(dry_run: bool = False):
    """
    Re-verify all failed artifacts.
    
    Args:
        dry_run: If True, only print what would be changed
    """
    async with AsyncSessionLocal() as session:
        # Get all failed artifacts
        stmt = select(Artifact).where(Artifact.verification_status == 'failed')
        result = await session.execute(stmt)
        failed_artifacts = result.scalars().all()
        
        print(f"\n{'='*70}")
        print(f"RETROACTIVE ARTIFACT RE-VERIFICATION")
        print(f"{'='*70}")
        print(f"Total failed artifacts: {len(failed_artifacts)}")
        print()
        
        fixed_count = 0
        still_failed = 0
        
        for artifact in failed_artifacts:
            # Prepare artifact data for verification
            artifact_data = {
                'type': artifact.type,
                'content_kind': artifact.content_kind,
                'content_location': artifact.content_location
            }
            
            # Re-verify
            results = artifact_verifier.verify(artifact_data)
            new_status = artifact_verifier.get_overall_status(results)
            
            old_status = artifact.verification_status
            
            if new_status == 'passed' and old_status == 'failed':
                fixed_count += 1
                
                if not dry_run:
                    # Update artifact
                    artifact.verification_status = new_status
                    artifact.verification_results = [r.to_dict() for r in results]
                
                print(f"✓ FIXED: {artifact.content_location[:50]}...")
                print(f"  Old: {old_status} → New: {new_status}")
            else:
                still_failed += 1
                print(f"✗ STILL FAILED: {artifact.content_location[:50]}...")
        
        print()
        print(f"{'='*70}")
        print(f"RESULTS:")
        print(f"  Total processed: {len(failed_artifacts)}")
        print(f"  Fixed: {fixed_count}")
        print(f"  Still failed: {still_failed}")
        
        if dry_run:
            print(f"\n  DRY RUN - No changes made")
        else:
            await session.commit()
            print(f"\n  Changes committed to database")
        
        print(f"{'='*70}\n")
        
        return fixed_count, still_failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-verify failed artifacts')
    parser.add_argument('--dry-run', action='store_true', help='Only print changes, do not commit')
    args = parser.parse_args()
    
    asyncio.run(retroverify_artifacts(dry_run=args.dry_run))
