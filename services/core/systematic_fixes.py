from logging_config import get_logger
logger = get_logger(__name__)

"""
SYSTEMATIC FIXES - Master script for fixing stuck goals
========================================================

Запускает все исправления в правильном порядке:

1. Auto-decompose pending non-atomic goals
2. Recalculate parent progress
3. Fix ontology violations (SQL)
4. Verify results

Author: AI-OS Core Team
Date: 2026-02-11
Severity: EMERGENCY FIXES
"""

import asyncio
import sys
from datetime import datetime


async def run_all_fixes():
    """Запускает все исправления последовательно"""

    logger.info("\n" + "="*70)
    logger.info("SYSTEMATIC FIXES FOR STUCK GOALS")
    logger.info("="*70)
    logger.info(f"\nStarted at: {datetime.now().isoformat()}")

    # =============================================================================
    # FIX 1: Auto-decompose pending non-atomic goals
    # =============================================================================
    logger.info("\n" + "-"*70)
    logger.info("FIX 1/3: AUTO-DECOMPOSE PENDING NON-ATOMIC GOALS")
    logger.info("-"*70)

    try:
        from auto_decomposer import auto_decomposer

        report1 = await auto_decomposer.decompose_all_pending_non_atomic()

        logger.info(f"\n✅ Decompose report:")
        logger.info(f"   Total pending: {report1['total']}")
        logger.info(f"   Decomposed: {report1['decomposed']}")
        logger.info(f"   Skipped: {report1['skipped']}")
        logger.info(f"   Failed: {report1['failed']}")

    except Exception as e:
        logger.info(f"\n❌ DECOMPOSE ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # FIX 2: Recalculate parent progress
    # =============================================================================
    logger.info("\n" + "-"*70)
    logger.info("FIX 2/3: RECALCULATE PARENT PROGRESS")
    logger.info("-"*70)

    try:
        from parent_progress_aggregator import parent_progress_aggregator

        report2 = await parent_progress_aggregator.recalculate_all_parents()

        logger.info(f"\n✅ Progress aggregation report:")
        logger.info(f"   Total parents: {report2['total_parents']}")
        logger.info(f"   Updated: {report2['updated']}")
        logger.info(f"   Completed: {report2['completed']}")
        logger.info(f"   Activated: {report2['activated']}")
        logger.info(f"   Errors: {report2['errors']}")

    except Exception as e:
        logger.info(f"\n❌ PROGRESS ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # FIX 3: Fix ontology violations (SQL)
    # =============================================================================
    logger.info("\n" + "-"*70)
    logger.info("FIX 3/3: FIX ONTOLOGY VIOLATIONS (SQL)")
    logger.info("-"*70)

    try:
        import subprocess
        import os

        sql_path = "/home/onor/ai_os_final/services/core/migrations/fix_ontology_violations.sql"
        container_sql_path = "/tmp/fix_ontology_violations.sql"

        if not os.path.exists(sql_path):
            logger.info(f"⚠️  SQL file not found: {sql_path}")
        else:
            # Copy SQL to container
            logger.info(f"\n📋 Copying SQL to container...")
            result = subprocess.run([
                "docker", "cp", sql_path, "ns_postgres:/tmp/fix_ontology_violations.sql"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                logger.info(f"❌ Copy failed: {result.stderr}")
            else:
                logger.info("✅ SQL copied")

                # Execute SQL
                logger.info("\n🔧 Executing SQL fix...")
                result = subprocess.run([
                    "docker", "exec", "ns_postgres",
                    "psql", "-U", "ns_admin", "-d", "ns_core_db",
                    "-f", "/tmp/fix_ontology_violations.sql"
                ], capture_output=True, text=True)

                logger.info(result.stdout)
                if result.stderr:
                    logger.info(f"⚠️  SQL warnings: {result.stderr}")

                if result.returncode == 0:
                    logger.info("\n✅ Ontology violations fixed!")
                else:
                    logger.info(f"\n❌ SQL execution failed (code {result.returncode})")

    except Exception as e:
        logger.info(f"\n❌ SQL ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # VERIFY: Show final statistics
    # =============================================================================
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION: FINAL STATISTICS")
    logger.info("="*70)

    try:
        import subprocess
        import json

        # Get goals by status
        result = subprocess.run([
            "docker", "exec", "ns_postgres",
            "psql", "-U", "ns_admin", "-d", "ns_core_db",
            "-c", "SELECT status, COUNT(*) FROM goals GROUP BY status ORDER BY status;"
        ], capture_output=True, text=True)

        logger.info("\n📊 Goals by status:")
        logger.info(result.stdout)

        # Get ontology check
        result = subprocess.run([
            "docker", "exec", "ns_postgres",
            "psql", "-U", "ns_admin", "-d", "ns_core_db",
            "-c", """
            SELECT
                goal_type,
                status,
                COUNT(*) as count
            FROM goals
            WHERE goal_type IN ('continuous', 'directional')
            GROUP BY goal_type, status
            ORDER BY goal_type, status;
            """
        ], capture_output=True, text=True)

        logger.info("\n📊 Ontology compliance (continuous/directional):")
        logger.info(result.stdout)

    except Exception as e:
        logger.info(f"\n❌ VERIFICATION ERROR: {e}")

    # =============================================================================
    # DONE
    # =============================================================================
    logger.info("\n" + "="*70)
    logger.info("SYSTEMATIC FIXES COMPLETED")
    logger.info("="*70)
    logger.info(f"\nCompleted at: {datetime.now().isoformat()}")

    logger.info("\n📋 NEXT STEPS:")
    logger.info("1. Verify goals are decomposing correctly")
    logger.info("2. Verify parent progress is updating")
    logger.info("3. Verify no continuous/directional marked as 'done'")
    logger.info("4. Monitor for 24-48 hours")
    logger.info("5. Then enable Phase 2 (Velocity Engine + Drift Detector)")


if __name__ == "__main__":
    asyncio.run(run_all_fixes())
