"""
LEGACY POLICY - Constitutional Guard for AI_OS

Enforces Legacy Axis (S0) constraints:
- Immutability: Legacy cannot be deleted, completed, optimized
- Survivability: Goals must support long-term survival
- No Optimization: Forbidden metrics (ROI, velocity, etc.)

This is the policy layer that prevents violation of S0.
"""
from typing import Dict, List, Optional
from sqlalchemy import select
from database import AsyncSessionLocal
from models import LegacyAxis, Goal


class LegacyPolicyViolation(Exception):
    """Raised when Legacy Policy is violated"""
    def __init__(self, violation_type: str, message: str):
        self.violation_type = violation_type
        self.message = message
        super().__init__(message)


class LegacyPolicy:
    """
    Policy enforcer for Legacy Axis (S0)

    Architectural guarantee:
    - NO Goal can violate Legacy Axis
    - NO operation can bypass immutability
    - NO optimization of Legacy itself
    """

    # Forbidden operations on Legacy Axis
    FORBIDDEN_OPERATIONS = {
        "delete",           # Cannot delete Legacy Axis
        "complete",         # Cannot mark as complete
        "optimize",         # Cannot optimize Legacy Axis
        "add_deadline",     # Cannot add deadline
        "decompose"         # Cannot decompose Legacy Axis
    }

    # Forbidden metrics (never allow these to be used for Legacy)
    FORBIDDEN_METRICS = {
        "ROI", "roi", "return_on_investment",
        "velocity", "speed", "rate",
        "completion_rate", "completion",
        "profit", "revenue", "cost_only"
    }

    # Allowed metrics (these support long-term thinking)
    ALLOWED_METRICS = {
        "generational_impact",
        "cultural_resonance",
        "long_term_survival",
        "institutional_longevity"
    }

    async def validate_goal_creation(self, goal_data: Dict) -> Dict:
        """
        Validate goal creation against Legacy Policy

        Args:
            goal_data: Goal creation data

        Returns:
            {"valid": true/false, "violations": [...], "warnings": [...]}

        Raises:
            LegacyPolicyViolation if hard violation
        """
        violations = []
        warnings = []

        # Check if goal is trying to be Legacy Axis (should use LegacyAxis service)
        if goal_data.get("depth_level") == 0 and goal_data.get("goal_type") == "directional":
            # This might be trying to create Legacy Axis
            warnings.append({
                "code": "POTENTIAL_LEGACY_MISUSE",
                "message": "Possible Legacy Axis creation. Use LegacyAxis service instead."
            })

        # Check for forbidden metrics
        metrics = goal_data.get("evaluation_metrics", {})
        if metrics:
            forbidden_found = self._check_forbidden_metrics(metrics)
            if forbidden_found:
                violations.append({
                    "code": "FORBIDDEN_METRICS",
                    "message": f"Legacy Axis forbids these metrics: {', '.join(forbidden_found)}",
                    "metrics": forbidden_found
                })

        # Check if goal is trying to add deadline to Legacy-like goal
        if goal_data.get("deadline") and goal_data.get("depth_level") == 0:
            violations.append({
                "code": "LEGACY_DEADLINE",
                "message": "Legacy Axis cannot have deadlines"
            })

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "warnings": warnings
        }

    async def validate_goal_completion(self, goal_id: str) -> Dict:
        """
        Validate goal completion against Legacy Policy

        Args:
            goal_id: Goal ID

        Returns:
            {"valid": true/false, "reason": "..."}
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"valid": False, "reason": "Goal not found"}

            # Legacy-like goals (depth 0, directional) cannot be completed
            if goal.depth_level == 0 and goal.goal_type == "directional":
                return {
                    "valid": False,
                    "reason": "Legacy Axis (directional L0) cannot be completed. It provides direction, not completion."
                }

            return {"valid": True}

    async def validate_goal_decomposition(self, goal_id: str) -> Dict:
        """
        Validate goal decomposition against Legacy Policy

        Args:
            goal_id: Goal ID

        Returns:
            {"valid": true/false, "reason": "..."}
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"valid": False, "reason": "Goal not found"}

            # Legacy Axis cannot be decomposed
            if goal.depth_level == 0 and goal.goal_type == "directional":
                return {
                    "valid": False,
                    "reason": "Legacy Axis cannot be decomposed. Goals derive FROM it, not the other way."
                }

            return {"valid": True}

    def check_survivability(self, goal: Goal) -> Dict:
        """
        Check if goal supports survivability policy

        Args:
            goal: Goal to check

        Returns:
            {"survivable": true/false, "gaps": [...]}
        """
        gaps = []

        # Check if goal has long-term horizon
        if goal.depth_level == 0 or goal.depth_level == 1:
            # L0-L1 should have long-term thinking
            if not goal.description or len(goal.description) < 50:
                gaps.append("L0-L1 goals need detailed description for long-term context")

            # Check if goal mentions institutions, culture, or knowledge
            desc_lower = goal.description.lower() if goal.description else ""
            title_lower = goal.title.lower() if goal.title else ""

            survival_keywords = ["институци", "культур", "знани", "образован", "систем", "долгосрочн"]
            has_survival_aspect = any(kw in desc_lower or kw in title_lower for kw in survival_keywords)

            if not has_survival_aspect:
                gaps.append("L0-L1 goals should reference institutions, culture, or knowledge systems")

        return {
            "survivable": len(gaps) == 0,
            "gaps": gaps
        }

    def _check_forbidden_metrics(self, metrics: Dict) -> List[str]:
        """Check if metrics contain forbidden ones"""
        forbidden_found = []

        for metric_key in metrics.keys():
            key_lower = metric_key.lower()
            for forbidden in self.FORBIDDEN_METRICS:
                if forbidden in key_lower:
                    forbidden_found.append(metric_key)

        return forbidden_found

    async def get_legacy_constraints(self) -> Dict:
        """
        Get current Legacy constraints for API/UX

        Returns:
            {
                "forbidden_operations": [...],
                "forbidden_metrics": [...],
                "allowed_metrics": [...],
                "immutability_guarantee": "..."
            }
        """
        return {
            "forbidden_operations": list(self.FORBIDDEN_OPERATIONS),
            "forbidden_metrics": list(self.FORBIDDEN_METRICS),
            "allowed_metrics": list(self.ALLOWED_METRICS),
            "immutability_guarantee": "Legacy Axis (S0) cannot be deleted, completed, optimized, or have deadlines. It provides meaning, not activity.",
            "survivability_requirement": "L0-L1 goals must support long-term survival through institutions, culture, or knowledge systems."
        }


# Global instance
legacy_policy = LegacyPolicy()
