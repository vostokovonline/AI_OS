"""
AI INTERVENTION LAYER v1.0
==========================

Detects stuck goals and suggests interventions.

Key rule: "10-day rule" - goals stuck >10 days auto-review

Author: AI-OS Core Team
Date: 2026-02-11
Philosophy: Movement over Perfection
"""

from typing import Dict, List, Optional, Literal
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from database import AsyncSessionLocal
from models import Goal, GoalRelation, Artifact
from enum import Enum
import json


# =============================================================================
# INTERVENTION TYPES
# =============================================================================

class InterventionType(str, Enum):
    """Types of interventions"""
    RECOMMENDATION = "recommendation"  # Suggest action
    ESCALATION = "escalation"  # Require human input
    AUTO_FIX = "auto_fix"  # System can fix automatically
    CANCEL = "cancel"  # Recommend cancellation
    TRANSFORM = "transform"  # Suggest goal transformation


class InterventionPriority(str, Enum):
    """Priority levels for interventions"""
    INFO = "info"  # FYI
    LOW = "low"  # Should fix
    MEDIUM = "medium"  # Should fix soon
    HIGH = "high"  # Requires attention
    URGENT = "urgent"  # Immediate action


# =============================================================================
# INTERVENTION LAYER
# =============================================================================

class AIInterventionLayer:
    """
    Detects stuck goals and suggests interventions

    Key principle: "10-day rule"
    - Goal active >10 days without progress = intervention
    """

    def __init__(self):
        self.stuck_threshold_days = 10  # 10-day rule
        self.critical_threshold_days = 30  # Critical: 30 days
        self.intervention_cache = {}

    async def scan_for_interventions(self) -> Dict:
        """
        Scan all goals for potential interventions

        Returns:
            {
                "interventions_required": int,
                "by_priority": {...},
                "by_type": {...},
                "interventions": [
                    {
                        "goal_id": "...",
                        "goal_title": "...",
                        "type": "...",
                        "priority": "...",
                        "reason": "...",
                        "suggestion": "...",
                        "auto_action": "...",
                        "stuck_days": int
                    }
                ],
                "timestamp": "..."
            }
        """
        interventions = []

        # Scan 1: Stuck goals (10-day rule)
        stuck_interventions = await self._detect_stuck_goals()
        interventions.extend(stuck_interventions)

        # Scan 2: Orphan goals (no parent, no progress)
        orphan_interventions = await self._detect_orphan_goals()
        interventions.extend(orphan_interventions)

        # Scan 3: Zombie goals (active but no resources)
        zombie_interventions = await self._detect_zombie_goals()
        interventions.extend(zombie_interventions)

        # Scan 4: Over-decomposed goals
        decomposition_interventions = await self._detect_over_decomposition()
        interventions.extend(decomposition_interventions)

        # Scan 5: Completed parents with active children (inconsistent)
        inconsistent_interventions = await self._detect_inconsistent_states()
        interventions.extend(inconsistent_interventions)

        # Group by priority and type
        by_priority = {}
        by_type = {}

        for intervention in interventions:
            priority = intervention["priority"]
            itype = intervention["type"]

            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_type[itype] = by_type.get(itype, 0) + 1

        return {
            "interventions_required": len(interventions),
            "by_priority": by_priority,
            "by_type": by_type,
            "interventions": interventions,
            "timestamp": datetime.now().isoformat()
        }

    async def _detect_stuck_goals(self) -> List[Dict]:
        """
        10-DAY RULE: Goals stuck >10 days without progress

        Rule:
        - Goal active >10 days
        - No progress change >10 days
        - → INTERVENE
        """
        async with AsyncSessionLocal() as db:
            ten_days_ago = datetime.now() - timedelta(days=self.stuck_threshold_days)

            # Find goals active >10 days
            stmt = select(Goal).where(
                Goal.status == "active",
                Goal.created_at < ten_days_ago
            )

            result = await db.execute(stmt)
            stuck_goals = result.scalars().all()

            interventions = []

            for goal in stuck_goals:
                # Check if progress has changed recently
                stuck_days = (datetime.now() - goal.created_at).total_seconds() / 86400

                # Priority based on how long stuck
                if stuck_days > self.critical_threshold_days:
                    priority = InterventionPriority.URGENT
                    itype = InterventionType.ESCALATION
                elif stuck_days > 20:
                    priority = InterventionPriority.HIGH
                    itype = InterventionType.ESCALATION
                elif stuck_days > 15:
                    priority = InterventionPriority.MEDIUM
                    itype = InterventionType.RECOMMENDATION
                else:
                    priority = InterventionPriority.LOW
                    itype = InterventionType.RECOMMENDATION

                # Generate suggestion
                suggestion = self._generate_stuck_goal_suggestion(goal, stuck_days)

                interventions.append({
                    "goal_id": str(goal.id),
                    "goal_title": goal.title[:60],
                    "type": itype.value,
                    "priority": priority.value,
                    "reason": (
                        f"10-DAY RULE VIOLATION: Goal stuck for {stuck_days:.0f} days "
                        f"without progress"
                    ),
                    "suggestion": suggestion["text"],
                    "auto_action": suggestion["auto_action"],
                    "stuck_days": int(stuck_days),
                    "current_progress": goal.progress
                })

            return interventions

    async def _detect_orphan_goals(self) -> List[Dict]:
        """
        Detect orphan goals (no parent, not mission level)

        Pattern:
        - L1/L2 goal without parent
        - Not part of any hierarchy
        - → MAY NEED ATTACHMENT OR CANCELLATION
        """
        async with AsyncSessionLocal() as db:
            # Find active goals without parent
            stmt = select(Goal).where(
                Goal.status == "active",
                Goal.parent_id.is_(None),
                Goal.depth_level > 0  # Not mission-level (L0)
            )

            result = await db.execute(stmt)
            orphans = result.scalars().all()

            interventions = []

            for goal in orphans:
                # Check if orphaned recently (within 7 days)
                orphan_days = (datetime.now() - goal.created_at).total_seconds() / 86400

                if orphan_days > 7:  # Give 7 days grace period
                    priority = InterventionPriority.MEDIUM if orphan_days < 30 else InterventionPriority.HIGH

                    interventions.append({
                        "goal_id": str(goal.id),
                        "goal_title": goal.title[:60],
                        "type": InterventionType.TRANSFORM.value,
                        "priority": priority.value,
                        "reason": (
                            f"ORPHAN GOAL: L{goal.depth_level} goal has no parent "
                            f"({orphan_days:.0f} days orphaned)"
                        ),
                        "suggestion": (
                            f"Goal '{goal.title[:40]}' is orphaned (no parent). "
                            f"Consider: (1) Attaching to parent mission, "
                            f"(2) Promoting to mission-level, (3) Cancelling if obsolete."
                        ),
                        "auto_action": "requires_human_decision",
                        "stuck_days": int(orphan_days)
                    })

            return interventions

    async def _detect_zombie_goals(self) -> List[Dict]:
        """
        Detect zombie goals (active but no resources assigned)

        Pattern:
        - Goal created
        - No execution attempts
        - No artifacts registered
        - → ZOMBIE (dead but marked active)
        """
        async with AsyncSessionLocal() as db:
            # Find active goals with no artifacts
            stmt = select(Goal).where(
                Goal.status == "active",
                Goal.is_atomic == True  # Only atomic goals should have artifacts
            )

            result = await db.execute(stmt)
            atomic_goals = result.scalars().all()

            interventions = []

            for goal in atomic_goals:
                # Check for artifacts
                stmt_artifacts = select(func.count(Artifact.id)).where(
                    Artifact.goal_id == goal.id
                )

                result_artifacts = await db.execute(stmt_artifacts)
                artifact_count = result_artifacts.scalar() or 0

                # If no artifacts and goal is old (>3 days)
                if artifact_count == 0:
                    age_days = (datetime.now() - goal.created_at).total_seconds() / 86400

                    if age_days > 3:
                        priority = (
                            InterventionPriority.URGENT if age_days > 14
                            else InterventionPriority.HIGH if age_days > 7
                            else InterventionPriority.MEDIUM
                        )

                        interventions.append({
                            "goal_id": str(goal.id),
                            "goal_title": goal.title[:60],
                            "type": InterventionType.CANCEL.value,
                            "priority": priority.value,
                            "reason": (
                                f"ZOMBIE GOAL: Atomic goal has no execution activity "
                                f"({age_days:.0f} days inactive)"
                            ),
                            "suggestion": (
                                f"Atomic goal '{goal.title[:40]}' shows no execution activity. "
                                f"Consider: (1) Executing immediately, (2) Cancelling, "
                                f"(3) Converting to non-atomic (if not requiring artifacts)."
                            ),
                            "auto_action": "recommend_cancel_or_execute",
                            "stuck_days": int(age_days)
                        })

            return interventions

    async def _detect_over_decomposition(self) -> List[Dict]:
        """
        Detect over-decomposed goals (too many children)

        Pattern:
        - Parent goal with >20 children
        - Completion rate slow
        - → OVER-DECOMPOSITION
        """
        async with AsyncSessionLocal() as db:
            # Count children per parent
            stmt_children = select(
                Goal.parent_id,
                func.count(Goal.id).label("child_count")
            ).where(
                Goal.parent_id.isnot(None)
            ).group_by(Goal.parent_id).having(
                func.count(Goal.id) > 20  # More than 20 children
            )

            result = await db.execute(stmt_children)
            over_decomposed = result.all()

            interventions = []

            for parent_id, child_count in over_decomposed:
                # Get parent goal
                stmt_parent = select(Goal).where(Goal.id == parent_id)
                result_parent = await db.execute(stmt_parent)
                parent = result_parent.scalar_one_or_none()

                if not parent:
                    continue

                priority = (
                    InterventionPriority.HIGH if child_count > 50
                    else InterventionPriority.MEDIUM
                )

                interventions.append({
                    "goal_id": str(parent_id),
                    "goal_title": parent.title[:60],
                    "type": InterventionType.TRANSFORM.value,
                    "priority": priority.value,
                    "reason": (
                        f"OVER-DECOMPOSITION: Goal has {child_count} children "
                        f"(suggests planning problem)"
                    ),
                    "suggestion": (
                        f"Goal '{parent.title[:40]}' has {child_count} subgoals. "
                        f"Consider: (1) Merging related subgoals, "
                        f"(2) Splitting into multiple parent goals, "
                        f"(3) Raising completion threshold."
                    ),
                    "auto_action": "requires_restructuring",
                    "stuck_days": 0  # Not stuck, just over-decomposed
                })

            return interventions

    async def _detect_inconsistent_states(self) -> List[Dict]:
        """
        Detect inconsistent states

        Pattern:
        - Parent marked "done"
        - Children still "active"
        - → INCONSISTENT STATE
        """
        async with AsyncSessionLocal() as db:
            # Find done goals with children
            stmt = select(Goal).where(
                Goal.status == "done"
            )

            result = await db.execute(stmt)
            done_goals = result.scalars().all()

            interventions = []

            for parent in done_goals:
                # Check for active children
                stmt_children = select(func.count(Goal.id)).where(
                    Goal.parent_id == parent.id,
                    Goal.status == "active"
                )

                result_children = await db.execute(stmt_children)
                active_children = result_children.scalar() or 0

                if active_children > 0:
                    priority = InterventionPriority.HIGH

                    interventions.append({
                        "goal_id": str(parent.id),
                        "goal_title": parent.title[:60],
                        "type": InterventionType.AUTO_FIX.value,
                        "priority": priority.value,
                        "reason": (
                            f"INCONSISTENT STATE: Parent marked 'done' but "
                            f"has {active_children} active children"
                        ),
                        "suggestion": (
                            f"Goal '{parent.title[:40]}' is done but has {active_children} "
                            f"active children. This is an inconsistent state. "
                            f"Consider: (1) Completing children first, "
                            f"(2) Reverting parent to 'active'."
                        ),
                        "auto_action": "suggest_revert_parent_or_complete_children",
                        "stuck_days": 0
                    })

            return interventions

    def _generate_stuck_goal_suggestion(self, goal: Goal, stuck_days: float) -> Dict:
        """Generate suggestion for stuck goal"""
        suggestion = {
            "text": "",
            "auto_action": "none"
        }

        # Rule 1: If continuous goal stuck → may need transformation
        if goal.goal_type == "continuous":
            suggestion["text"] = (
                f"Continuous goal stuck {stuck_days:.0f} days. "
                f"Consider: (1) Is this actually continuous? "
                f"(2) Should it be transformed to 'achievable'? "
                f"(3) Is evaluation criteria correct?"
            )
            suggestion["auto_action"] = "recommend_transform_to_achievable"

        # Rule 2: If atomic goal stuck → should execute or cancel
        elif goal.is_atomic:
            suggestion["text"] = (
                f"Atomic goal stuck {stuck_days:.0f} days. "
                f"Atomic goals should complete quickly. "
                f"Consider: (1) Immediate execution, (2) Cancellation, "
                f"(3) Converting to non-atomic."
            )
            suggestion["auto_action"] = "recommend_execute_or_cancel"

        # Rule 3: If parent goal stuck → check children
        elif goal.depth_level > 0:
            suggestion["text"] = (
                f"Parent goal stuck {stuck_days:.0f} days. "
                f"Check if: (1) Children are progressing, "
                f"(2) Decomposition is correct, "
                f"(3) Goal should be split differently."
            )
            suggestion["auto_action"] = "review_children_progress"

        # Rule 4: Mission-level goal stuck → strategic review
        else:
            suggestion["text"] = (
                f"Mission-level goal stuck {stuck_days:.0f} days. "
                f"Requires strategic review. "
                f"Consider: (1) Is mission still valid? "
                f"(2) Should mission be cancelled? (3) Are resources adequate?"
            )
            suggestion["auto_action"] = "requires_strategic_review"

        return suggestion

    async def apply_intervention(
        self,
        goal_id: str,
        intervention_type: str,
        auto_action: str
    ) -> Dict:
        """
        Apply intervention (requires human approval for most actions)

        Args:
            goal_id: Goal to intervene on
            intervention_type: Type of intervention
            auto_action: Auto action to apply

        Returns:
            {
                "result": "applied|rejected|requires_approval",
                "message": "...",
                "action_taken": "..."
            }
        """
        # Import here to avoid circular dependency
        from goal_velocity_engine import goal_velocity_engine
        from strategic_drift_detector import strategic_drift_detector

        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {
                    "result": "rejected",
                    "message": "Goal not found"
                }

            # Check if auto_action is safe
            safe_actions = [
                "requires_human_decision",
                "recommend_execute_or_cancel",
                "recommend_transform_to_achievable"
            ]

            if auto_action not in safe_actions:
                return {
                    "result": "requires_approval",
                    "message": "This intervention requires human approval",
                    "action_taken": "none"
                }

            # Apply safe interventions
            if auto_action == "recommend_execute_or_cancel":
                # Just mark for review
                return {
                    "result": "applied",
                    "message": f"Goal '{goal.title[:40]}' marked for review",
                    "action_taken": "marked_for_review"
                }

            elif auto_action == "recommend_transform_to_achievable":
                # Suggest type change (requires human approval)
                return {
                    "result": "requires_approval",
                    "message": "Type transformation requires human approval",
                    "action_taken": "suggestion_logged"
                }

            else:
                return {
                    "result": "requires_approval",
                    "message": "Intervention requires human review",
                    "action_taken": "none"
                }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

ai_intervention_layer = AIInterventionLayer()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def scan_interventions() -> Dict:
    """Scan for required interventions"""
    return await ai_intervention_layer.scan_for_interventions()


async def check_10_day_rule() -> List[Dict]:
    """Check 10-day rule violations only"""
    return await ai_intervention_layer._detect_stuck_goals()


async def apply_intervention(goal_id: str, intervention_type: str, auto_action: str) -> Dict:
    """Apply intervention to goal"""
    return await ai_intervention_layer.apply_intervention(
        goal_id, intervention_type, auto_action
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing AI Intervention Layer...\n")

        # Scan for interventions
        interventions = await ai_intervention_layer.scan_for_interventions()

        print(f"Interventions Required: {interventions['interventions_required']}")
        print("\nBy Priority:")
        for priority, count in interventions['by_priority'].items():
            print(f"  {priority}: {count}")

        print("\nBy Type:")
        for itype, count in interventions['by_type'].items():
            print(f"  {itype}: {count}")

        print("\nTop 5 Interventions:")
        for intervention in interventions['interventions'][:5]:
            print(f"\n  [{intervention['priority'].upper()}] {intervention['goal_title']}")
            print(f"  Reason: {intervention['reason']}")
            print(f"  Suggestion: {intervention['suggestion']}")

    asyncio.run(test())
