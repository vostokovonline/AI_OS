#!/usr/bin/env python3
"""
AI_OS System Test Suite
Tests Goal Ontology v3.0, Artifact Layer, and MVP Skills
"""
import requests
import json
import time
from typing import Dict, List
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
CORE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8501"
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ns_core_db",
    "user": "ns_admin",
    "password": "ns_core_db"
}


class AIOSTester:
    """Test suite for AI_OS system"""

    def __init__(self):
        self.core_url = CORE_URL
        self.results = []
        self.start_time = None

    def log(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)

        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏳"
        logger.info(f"{icon} {test_name}: {status}")
        if details:
            logger.info(f"   {details}")

    def test_connection(self) -> bool:
        """Test connection to core service"""
        try:
            # Try /docs endpoint (Swagger UI)
            response = requests.get(f"{self.core_url}/docs", timeout=5)
            if response.status_code == 200:
                self.log("Connection to Core Service", "PASS", "API is reachable")
                return True
            else:
                self.log("Connection to Core Service", "FAIL", f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log("Connection to Core Service", "FAIL", str(e))
            return False

    def get_all_goals(self) -> List[Dict]:
        """Get all goals from the database"""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            cursor.execute("""
                SELECT
                    id, title, description, status, progress,
                    goal_type, depth_level, is_atomic,
                    domains, completion_criteria, evaluation_result,
                    goal_contract, mutation_status, success_definition,
                    parent_id, created_at, updated_at
                FROM goals
                ORDER BY depth_level, parent_id NULLS LAST, created_at
            """)

            goals = [dict(row) for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            self.log("Get All Goals", "PASS", f"Found {len(goals)} goals")
            return goals

        except Exception as e:
            self.log("Get All Goals", "FAIL", str(e))
            return []

    def test_goal_ontology(self, goals: List[Dict]):
        """Test Goal Ontology v3.0 fields"""
        logger.info("\n=== Testing Goal Ontology v3.0 ===")

        required_fields = [
            "id", "title", "status", "goal_type", "depth_level",
            "is_atomic", "domains", "completion_criteria"
        ]

        for goal in goals:
            missing = [f for f in required_fields if f not in goal]
            if missing:
                self.log(
                    f"Goal Ontology: {goal['title'][:30]}",
                    "FAIL",
                    f"Missing fields: {missing}"
                )
            else:
                self.log(
                    f"Goal Ontology: {goal['title'][:30]}",
                    "PASS",
                    f"Type={goal['goal_type']}, L{goal['depth_level']}, Atomic={goal['is_atomic']}"
                )

    def test_goal_hierarchy(self, goals: List[Dict]):
        """Test goal hierarchy (parent-child relationships)"""
        logger.info("\n=== Testing Goal Hierarchy ===")

        # Build hierarchy
        hierarchy = {}
        for goal in goals:
            goal_id = str(goal["id"])
            hierarchy[goal_id] = goal

        # Check parents exist
        orphan_count = 0
        for goal in goals:
            if goal.get("parent_id"):
                parent_id = str(goal["parent_id"])
                if parent_id not in hierarchy:
                    self.log(
                        f"Hierarchy: {goal['title'][:30]}",
                        "FAIL",
                        f"Parent {parent_id} not found"
                    )
                    orphan_count += 1

        if orphan_count == 0:
            self.log("Goal Hierarchy", "PASS", "All parents found")
        else:
            self.log("Goal Hierarchy", "FAIL", f"{orphan_count} orphans found")

        # Check depth levels
        depth_counts = {}
        for goal in goals:
            level = goal.get("depth_level", -1)
            depth_counts[level] = depth_counts.get(level, 0) + 1

        self.log(
            "Depth Levels",
            "PASS",
            f"L0={depth_counts.get(0,0)}, L1={depth_counts.get(1,0)}, L2={depth_counts.get(2,0)}, L3={depth_counts.get(3,0)}"
        )

        # Check goal types
        type_counts = {}
        for goal in goals:
            gtype = goal.get("goal_type", "unknown")
            type_counts[gtype] = type_counts.get(gtype, 0) + 1

        self.log(
            "Goal Types",
            "PASS",
            f"Meta={type_counts.get('meta',0)}, Achievable={type_counts.get('achievable',0)}, "
            f"Continuous={type_counts.get('continuous',0)}, Directional={type_counts.get('directional',0)}, "
            f"Exploratory={type_counts.get('exploratory',0)}"
        )

    def test_atomic_goals(self, goals: List[Dict]):
        """Test atomic goals have proper requirements"""
        logger.info("\n=== Testing Atomic Goals ===")

        atomic_goals = [g for g in goals if g.get("is_atomic")]

        for goal in atomic_goals:
            # Check completion criteria
            criteria = goal.get("completion_criteria", {})
            if not criteria:
                self.log(
                    f"Atomic Goal: {goal['title'][:30]}",
                    "FAIL",
                    "No completion criteria"
                )
                continue

            # Check for artifact requirements
            artifacts_req = criteria.get("artifacts_required")
            if not artifacts_req:
                self.log(
                    f"Atomic Goal: {goal['title'][:30]}",
                    "WARN",
                    "No artifact requirements specified"
                )
            else:
                self.log(
                    f"Atomic Goal: {goal['title'][:30]}",
                    "PASS",
                    f"Requires: {[a['type'] for a in artifacts_req]}"
                )

    def test_goal_contracts(self, goals: List[Dict]):
        """Test goal contracts"""
        logger.info("\n=== Testing Goal Contracts ===")

        with_contract = 0
        for goal in goals:
            contract = goal.get("goal_contract")
            if contract:
                with_contract += 1

                # Check required fields
                if "allowed_actions" not in contract:
                    self.log(
                        f"Contract: {goal['title'][:30]}",
                        "FAIL",
                        "Missing allowed_actions"
                    )
                else:
                    self.log(
                        f"Contract: {goal['title'][:30]}",
                        "PASS",
                        f"Actions: {contract['allowed_actions']}"
                    )

        self.log(
            "Goal Contracts",
            "PASS" if with_contract > 0 else "WARN",
            f"{with_contract}/{len(goals)} goals have contracts"
        )

    def test_skills_registry(self):
        """Test skills are registered"""
        logger.info("\n=== Testing Skills Registry ===")

        try:
            response = requests.get(f"{self.core_url}/skills", timeout=10)
            if response.status_code == 200:
                skills = response.json()
                self.log(
                    "Skills Registry",
                    "PASS",
                    f"{len(skills)} skills registered"
                )

                # Check for MVP skills
                mvp_skills = ["text_to_file", "structured_generation", "web_research",
                             "summarize_knowledge", "self_check"]

                for skill_name in mvp_skills:
                    found = any(s.get("name") == skill_name for s in skills)
                    if found:
                        self.log(f"MVP Skill: {skill_name}", "PASS")
                    else:
                        self.log(f"MVP Skill: {skill_name}", "FAIL", "Not found")
            else:
                self.log("Skills Registry", "FAIL", f"Status: {response.status_code}")
        except Exception as e:
            self.log("Skills Registry", "FAIL", str(e))

    def test_artifact_layer(self, goals: List[Dict]):
        """Test artifact layer for atomic goals"""
        logger.info("\n=== Testing Artifact Layer ===")

        atomic_goals = [g for g in goals if g.get("is_atomic")]

        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            for goal in atomic_goals:
                goal_id = str(goal["id"])

                cursor.execute("""
                    SELECT
                        id, type, goal_id, content_kind, content_location,
                        verification_status, domains, tags,
                        created_at
                    FROM artifacts
                    WHERE goal_id = %s
                    ORDER BY created_at
                """, (goal_id,))

                artifacts = [dict(row) for row in cursor.fetchall()]

                if len(artifacts) > 0:
                    self.log(
                        f"Artifacts for: {goal['title'][:30]}",
                        "PASS",
                        f"{len(artifacts)} artifacts, {sum(1 for a in artifacts if a['verification_status'] == 'passed')} passed"
                    )
                else:
                    self.log(
                        f"Artifacts for: {goal['title'][:30]}",
                        "WARN",
                        "No artifacts yet (goal not executed)"
                    )

            cursor.close()
            conn.close()

        except Exception as e:
            self.log("Artifact Layer Test", "FAIL", str(e))

    def generate_report(self):
        """Generate test report"""
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)

        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = sum(1 for r in self.results if r["status"] == "FAIL")
        warned = sum(1 for r in self.results if r["status"] == "WARN")
        total = len(self.results)

        logger.info(f"\nTotal Tests: {total}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Warnings: {warned}")

        if failed == 0:
            logger.info("\n🎉 All tests passed!")
        else:
            logger.info(f"\n⚠️  {failed} test(s) failed")

        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "warned": warned
            },
            "results": self.results
        }

        report_path = "/tmp/test_results.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\n📄 Report saved to: {report_path}")

    def run_all_tests(self):
        """Run complete test suite"""
        logger.info("🧪 AI_OS System Test Suite")
        logger.info("="*60)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)

        self.start_time = time.time()

        # 1. Connection test
        if not self.test_connection():
            logger.info("\n❌ Cannot connect to core service. Exiting.")
            return

        # 2. Get goals
        goals = self.get_all_goals()
        if not goals:
            logger.info("\n❌ No goals found. Create test goals first.")
            return

        # 3. Test Goal Ontology
        self.test_goal_ontology(goals)

        # 4. Test Goal Hierarchy
        self.test_goal_hierarchy(goals)

        # 5. Test Atomic Goals
        self.test_atomic_goals(goals)

        # 6. Test Goal Contracts
        self.test_goal_contracts(goals)

        # 7. Test Skills Registry
        self.test_skills_registry()

        # 8. Test Artifact Layer
        self.test_artifact_layer(goals)

        # Generate report
        self.generate_report()

        elapsed = time.time() - self.start_time
        logger.info(f"\n⏱️  Total time: {elapsed:.2f}s")


if __name__ == "__main__":
    tester = AIOSTester()
    tester.run_all_tests()
