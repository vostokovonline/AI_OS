"""
MIGRATION TEST SUITE FOR 172 EXISTING GOALS
=============================================

Tests migration from old ontology (status="done") to new ontology
(lifecycle_state + evaluation_state) for ALL 172 goals in database.

This is ENGINEERING-GRADE test, not exploratory.

Author: AI-OS Core Team
Date: 2026-02-11
"""

import asyncio
from typing import Dict, List
from datetime import datetime


# =============================================================================
# TEST CASES
# =============================================================================

class MigrationTestResult:
    """Result of single goal migration test"""
    def __init__(self, goal_id: str, title: str, goal_type: str):
        self.goal_id = goal_id
        self.title = title
        self.goal_type = goal_type
        self.passed = False
        self.failed = False
        self.violations = []
        self.fixes = []
        self.test_timestamp = datetime.now()


class MigrationTestSuite:
    """
    Test suite for migrating 172 existing goals

    Tests:
    1. Detection of ontology violations
    2. Validation of lifecycle_state mapping
    3. Artifact requirement checks
    4. Data integrity validation
    """

    def __init__(self):
        self.results = []
        self.statistics = {
            "total_goals": 0,
            "by_type": {},
            "by_status": {},
            "violations_by_type": {},
            "fixes_required": {},
            "migration_ready": []
        }

    async def test_all_goals(self) -> MigrationTestResult:
        """Test migration for ALL 172 goals in database"""
        from database import AsyncSessionLocal
        from models import Goal
        from sqlalchemy import select, func

        async with AsyncSessionLocal() as db:
            # Get count
            count_stmt = select(func.count(Goal.id))
            count_result = await db.execute(count_stmt)
            total_count = count_result.scalar() or 0

            logger.info(f"\n{'='*70}")
            logger.info(f"MIGRATION TEST: {total_count} GOALS")
            logger.info(f"{'='*70}")
            logger.info(f"Started: {datetime.now().isoformat()}")

            # Get all goals
            stmt = select(Goal)
            result = await db.execute(stmt)
            goals = result.scalars().all()

            self.statistics["total_goals"] = total_count

            # Test each goal
            for i, goal in enumerate(goals, 1):
                result = await self.test_single_goal(goal, db)

                self.results.append(result)

                # Progress indicator
                if i % 10 == 0:
                    logger.info(f"  Tested: {i}/{total_count} goals...")

                # Update statistics
                gt = goal.goal_type
                self.statistics["by_type"][gt] = self.statistics["by_type"].get(gt, 0) + 1

                if result.failed:
                    self.statistics["violations_by_type"][gt] = \
                        self.statistics["violations_by_type"].get(gt, 0) + 1

            # Print summary
            self._print_summary()

            return self._get_overall_result()

    async def test_single_goal(self, goal, db) -> MigrationTestResult:
        """Test migration for single goal"""
        result = MigrationTestResult(
            goal_id=str(goal.id),
            title=goal.title,
            goal_type=goal.goal_type
        )

        # TEST 1: Detect ontology violations
        violation = self._detect_ontology_violation(goal)
        if violation:
            result.violations.append(violation)
            result.failed = True
            return result

        # TEST 2: Validate lifecycle_state mapping
        lifecycle_ok = self._validate_lifecycle_mapping(goal)
        if not lifecycle_ok:
            result.violations.append("lifecycle_state mapping incorrect")
            result.failed = True

        # TEST 3: Check artifact requirements
        artifact_ok = await self._check_artifact_requirements(goal, db)
        if not artifact_ok:
            result.violations.append("artifact requirement not met")
            result.failed = True

        # TEST 4: Data integrity
        integrity_ok = self._check_data_integrity(goal)
        if not integrity_ok:
            result.violations.append("data integrity violation")
            result.failed = True

        # If all tests pass
        if not result.failed:
            result.passed = True

        return result

    def _detect_ontology_violation(self, goal) -> str:
        """Detect if goal has ontology violation"""
        goal_type = goal.goal_type
        status = goal.status

        # Main invariant: wrong type marked as done
        if goal_type in ["continuous", "directional"] and status == "done":
            return f"{goal_type.upper()}_GOAL_MARKED_AS_DONE"

        return None

    def _validate_lifecycle_mapping(self, goal) -> bool:
        """Validate that lifecycle_state would be correct"""
        # For old model, check what lifecycle_state SHOULD be
        goal_type = goal.goal_type
        status = goal.status

        if goal_type == "continuous":
            # Should be "ongoing", not "done"
            if status == "done":
                return False
        elif goal_type == "directional":
            # Should be "permanent", not "done"
            if status == "done":
                return False

        return True

    async def _check_artifact_requirements(self, goal, db) -> bool:
        """Check if artifact requirements are met"""
        if not goal.is_atomic:
            return True  # Non-atomic goals don't require artifacts

        # Atomic goals: check if has artifacts
        from sqlalchemy import select
        from models import Artifact

        stmt = select(Artifact).where(Artifact.goal_id == goal.id)
        result = await db.execute(stmt)
        artifacts = result.scalars().all()

        if goal.status == "done" and len(artifacts) == 0:
            return False  # Done without artifacts

        return True

    def _check_data_integrity(self, goal) -> bool:
        """Check data integrity rules"""
        # Check 1: Progress in range
        if not (0.0 <= goal.progress <= 1.0):
            return False

        # Check 2: Status is valid
        valid_statuses = ["pending", "active", "done", "blocked", "failed"]
        if goal.status not in valid_statuses:
            return False

        return True

    def _print_summary(self):
        """Print test summary"""
        logger.info(f"\n{'='*70}")
        logger.info("MIGRATION TEST SUMMARY")
        logger.info(f"{'='*70}")

        total = self.statistics["total_goals"]
        passed = sum(1 for r in self.results if r.passed and not r.failed)
        failed = sum(1 for r in self.results if r.failed)

        logger.info(f"\n📊 STATISTICS:")
        logger.info(f"  Total goals: {total}")
        logger.info(f"  Passed: {passed} ({passed/total*100:.1f}%)")
        logger.info(f"  Failed: {failed} ({failed/total*100:.1f}%)")

        logger.info(f"\n📋 BY GOAL TYPE:")
        for gt, count in self.statistics["by_type"].items():
            logger.info(f"  {gt}: {count}")

        if self.statistics["violations_by_type"]:
            logger.info(f"\n⚠️  VIOLATIONS BY TYPE:")
            for gt, count in self.statistics["violations_by_type"].items():
                logger.info(f"  {gt}: {count} violations")

        # Show first 20 failures
        failures = [r for r in self.results if r.failed]
        if failures:
            logger.info(f"\n❌  FAILURES ({len(failures)}):")
            logger.info("-"*70)

            for i, f in enumerate(failures[:20], 1):
                logger.info(f"\n{i}. {f.title}")
                logger.info(f"   Type: {f.goal_type}")
                logger.info(f"   Status: {f.test_timestamp}")
                logger.info(f"   Violations:")
                for v in f.violations:
                    logger.info(f"     - {v}")

            if len(failures) > 20:
                logger.info(f"\n... and {len(failures) - 20} more")
        else:
            logger.info(f"\n✅ NO FAILURES - All goals can migrate safely")

        logger.info(f"\n{'='*70}\n")

    def _get_overall_result(self) -> MigrationTestResult:
        """Get overall test result"""
        total = self.statistics["total_goals"]
        failed_count = sum(1 for r in self.results if r.failed)

        result = MigrationTestResult(
            goal_id="OVERALL",
            title=f"Migration Test Suite ({total} goals)",
            goal_type="OVERALL"
        )

        if failed_count == 0:
            result.passed = True
        else:
            result.failed = True
            result.violations = [
                f"{failed_count}/{total} goals failed migration test"
            ]

        return result

    def generate_fix_script(self) -> str:
        """Generate SQL fix script for failed migrations"""
        fixes = []

        for result in self.results:
            if result.failed and result.violations:
                goal_id = result.goal_id
                goal_type = result.goal_type

                for violation in result.violations:
                    if "CONTINUOUS_AS_DONE" in violation:
                        fixes.append(f"-- Fix continuous goal {goal_id}")
                        fixes.append(f"UPDATE goals SET status = 'active' WHERE id = '{goal_id}';")
                        fixes.append(f"COMMENT ON GOAL {goal_id} IS 'Migrated from done to active (Phase 1)';")
                        fixes.append("")

                    elif "DIRECTIONAL_AS_DONE" in violation:
                        fixes.append(f"-- Fix directional goal {goal_id}")
                        fixes.append(f"UPDATE goals SET status = 'active' WHERE id = '{goal_id}';")
                        fixes.append(f"COMMENT ON GOAL {goal_id} IS 'Migrated from done to active (Phase 1)';")
                        fixes.append("")

        if not fixes:
            return "-- No fixes needed"

        return "\n".join([
            "-- ==========================================",
            "-- MIGRATION FIXES",
            f"-- Generated: {datetime.now().isoformat()}",
            "-- ==========================================",
            "",
        ] + fixes)

    def save_report(self, filename: str = "migration_test_report.txt"):
        """Save test report to file"""
        with open(filename, "w") as f:
            f.write(self._get_report_text())

        logger.info(f"\n📄 Report saved to: {filename}")

    def _get_report_text(self) -> str:
        """Generate text report"""
        lines = [
            "="*70,
            "MIGRATION TEST REPORT",
            "="*70,
            f"Date: {datetime.now().isoformat()}",
            "",
            "SUMMARY",
            "-"*70,
            f"Total goals tested: {self.statistics['total_goals']}",
            f"Passed: {sum(1 for r in self.results if r.passed)}",
            f"Failed: {sum(1 for r in self.results if r.failed)}",
            "",
            "FAILURES BY GOAL TYPE",
            "-"*70,
        ]

        # Group failures by type
        by_type = {}
        for r in self.results:
            if r.failed:
                gt = r.goal_type
                if gt not in by_type:
                    by_type[gt] = []
                by_type[gt].append(r.title)

        for gt, titles in by_type.items():
            lines.append(f"{gt}: {len(titles)}")
            for title in titles[:5]:  # Show first 5
                lines.append(f"  - {title}")
            if len(titles) > 5:
                lines.append(f"  ... and {len(titles) - 5} more")
            lines.append("")

        lines.append("")
        lines.append("="*70)

        return "\n".join(lines)


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

async def run_migration_tests():
    """Run complete migration test suite"""
    logger.info("\n" + "="*70)
    logger.info("MIGRATION TEST SUITE")
    logger.info("="*70)
    logger.info("Testing 172 existing goals for ontology compliance...")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info("="*70 + "\n")

    suite = MigrationTestSuite()
    result = await suite.test_all_goals()

    # Generate fix script
    fix_script = suite.generate_fix_script()
    fix_filename = "migration_fixes.sql"

    with open(fix_filename, "w") as f:
        f.write(fix_script)

    logger.info(f"\n📄 Fix script saved to: {fix_filename}")

    # Save report
    suite.save_report()

    # Return result
    return {
        "success": result.passed,
        "total_tested": suite.statistics["total_goals"],
        "passed": sum(1 for r in suite.results if r.passed),
        "failed": sum(1 for r in self.results if r.failed),
        "fix_script_generated": fix_filename,
        "report_generated": "migration_test_report.txt"
    }


# =============================================================================
# STANDALONE RUNNER
# =============================================================================

if __name__ == "__main__":
    result = asyncio.run(run_migration_tests())

    logger.info("\n" + "="*70)
    logger.info("MIGRATION TEST COMPLETE")
    logger.info("="*70)

    if result["success"]:
        logger.info("✅ RESULT: ALL TESTS PASSED")
        logger.info(f"   Total tested: {result['total_tested']}")
        logger.info(f"   Passed: {result['passed']}")
        logger.info(f"   Failed: {result['failed']}")
        logger.info("\n🎉 Migration can proceed safely!")

        logger.info("\n⚠️  REMINDER: Even with tests passing, REVIEW")
        logger.info("   the generated fix scripts before applying them.")

    else:
        logger.info("❌ RESULT: TESTS FAILED")
        logger.info(f"   Total tested: {result['total_tested']}")
        logger.info(f"   Passed: {result['passed']}")
        logger.info(f"   Failed: {result['failed']}")
        logger.info("\n📋 NEXT STEPS:")
        logger.info("   1. Review migration_test_report.txt")
        logger.info(f"   2. Review {result['fix_script_generated']}")
        logger.info("   3. Apply fixes manually or review violations")
        logger.info("   4. Re-run tests after fixes")
