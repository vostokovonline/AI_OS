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

    print("\n" + "="*70)
    print("SYSTEMATIC FIXES FOR STUCK GOALS")
    print("="*70)
    print(f"\nStarted at: {datetime.now().isoformat()}")

    # =============================================================================
    # FIX 1: Auto-decompose pending non-atomic goals
    # =============================================================================
    print("\n" + "-"*70)
    print("FIX 1/3: AUTO-DECOMPOSE PENDING NON-ATOMIC GOALS")
    print("-"*70)

    try:
        from auto_decomposer import auto_decomposer

        report1 = await auto_decomposer.decompose_all_pending_non_atomic()

        print(f"\n✅ Decompose report:")
        print(f"   Total pending: {report1['total']}")
        print(f"   Decomposed: {report1['decomposed']}")
        print(f"   Skipped: {report1['skipped']}")
        print(f"   Failed: {report1['failed']}")

    except Exception as e:
        print(f"\n❌ DECOMPOSE ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # FIX 2: Recalculate parent progress
    # =============================================================================
    print("\n" + "-"*70)
    print("FIX 2/3: RECALCULATE PARENT PROGRESS")
    print("-"*70)

    try:
        from parent_progress_aggregator import parent_progress_aggregator

        report2 = await parent_progress_aggregator.recalculate_all_parents()

        print(f"\n✅ Progress aggregation report:")
        print(f"   Total parents: {report2['total_parents']}")
        print(f"   Updated: {report2['updated']}")
        print(f"   Completed: {report2['completed']}")
        print(f"   Activated: {report2['activated']}")
        print(f"   Errors: {report2['errors']}")

    except Exception as e:
        print(f"\n❌ PROGRESS ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # FIX 3: Fix ontology violations (SQL)
    # =============================================================================
    print("\n" + "-"*70)
    print("FIX 3/3: FIX ONTOLOGY VIOLATIONS (SQL)")
    print("-"*70)

    try:
        import subprocess
        import os

        sql_path = "/home/onor/ai_os_final/services/core/migrations/fix_ontology_violations.sql"
        container_sql_path = "/tmp/fix_ontology_violations.sql"

        if not os.path.exists(sql_path):
            print(f"⚠️  SQL file not found: {sql_path}")
        else:
            # Copy SQL to container
            print(f"\n📋 Copying SQL to container...")
            result = subprocess.run([
                "docker", "cp", sql_path, "ns_postgres:/tmp/fix_ontology_violations.sql"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"❌ Copy failed: {result.stderr}")
            else:
                print("✅ SQL copied")

                # Execute SQL
                print("\n🔧 Executing SQL fix...")
                result = subprocess.run([
                    "docker", "exec", "ns_postgres",
                    "psql", "-U", "ns_admin", "-d", "ns_core_db",
                    "-f", "/tmp/fix_ontology_violations.sql"
                ], capture_output=True, text=True)

                print(result.stdout)
                if result.stderr:
                    print(f"⚠️  SQL warnings: {result.stderr}")

                if result.returncode == 0:
                    print("\n✅ Ontology violations fixed!")
                else:
                    print(f"\n❌ SQL execution failed (code {result.returncode})")

    except Exception as e:
        print(f"\n❌ SQL ERROR: {e}")
        import traceback
        traceback.print_exc()

    # =============================================================================
    # VERIFY: Show final statistics
    # =============================================================================
    print("\n" + "="*70)
    print("VERIFICATION: FINAL STATISTICS")
    print("="*70)

    try:
        import subprocess
        import json

        # Get goals by status
        result = subprocess.run([
            "docker", "exec", "ns_postgres",
            "psql", "-U", "ns_admin", "-d", "ns_core_db",
            "-c", "SELECT status, COUNT(*) FROM goals GROUP BY status ORDER BY status;"
        ], capture_output=True, text=True)

        print("\n📊 Goals by status:")
        print(result.stdout)

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

        print("\n📊 Ontology compliance (continuous/directional):")
        print(result.stdout)

    except Exception as e:
        print(f"\n❌ VERIFICATION ERROR: {e}")

    # =============================================================================
    # DONE
    # =============================================================================
    print("\n" + "="*70)
    print("SYSTEMATIC FIXES COMPLETED")
    print("="*70)
    print(f"\nCompleted at: {datetime.now().isoformat()}")

    print("\n📋 NEXT STEPS:")
    print("1. Verify goals are decomposing correctly")
    print("2. Verify parent progress is updating")
    print("3. Verify no continuous/directional marked as 'done'")
    print("4. Monitor for 24-48 hours")
    print("5. Then enable Phase 2 (Velocity Engine + Drift Detector)")


if __name__ == "__main__":
    asyncio.run(run_all_fixes())
