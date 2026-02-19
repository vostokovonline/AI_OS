"""
OUTCOME VALIDATOR v1.0
=====================

Validates goal OUTCOMES (state changes) not just ARTIFACTS

This replaces "artifact = done" with "outcome validation":
- Atomic goals: artifact-based validation
- Achievable (L0-L2): children aggregation
- Continuous: trend-based validation
- Directional: alignment-based validation

Author: AI-OS Core Team
Date: 2026-02-11
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select, func
from database import AsyncSessionLocal
from models import Goal, GoalState
import json


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

class OutcomeValidator:
    """
    Validates goal OUTCOMES instead of simple completion

    Key principle:
    - Artifact ≠ Outcome
    - Document ≠ Change
    - Task ≠ Result
    """

    async def validate(
        self,
        goal_id: str,
        force_revalidation: bool = False
    ) -> Dict:
        """
        Main validation entry point

        Args:
            goal_id: Goal to validate
            force_revalidation: Force re-validation even if recently validated

        Returns:
            {
                "lifecycle_state": "completed|ongoing|permanent|active",
                "evaluation_state": "validated|improving|stable|degrading|aligned|drifting",
                "passed": bool,
                "progress": 0.0-1.0,
                "confidence": 0.0-1.0,
                "evidence": [...],
                "recommendations": "...",
                "validation_mode": "artifact|aggregate|trend|alignment|hypothesis"
            }
        """
        # Get goal
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {
                    "error": "Goal not found",
                    "lifecycle_state": "unknown",
                    "passed": False
                }

            # Determine validation mode from goal properties
            validation_mode = self._determine_validation_mode(goal)

            # Route to appropriate validation method
            if validation_mode == "artifact":
                return await self._validate_atomic_artifact(goal, db)

            elif validation_mode == "aggregate":
                return await self._validate_aggregate(goal, db)

            elif validation_mode == "trend":
                return await self._validate_trend(goal, db)

            elif validation_mode == "alignment":
                return await self._validate_alignment(goal, db)

            elif validation_mode == "hypothesis":
                return await self._validate_hypothesis(goal, db)

            else:
                return {
                    "error": f"Unknown validation mode: {validation_mode}",
                    "lifecycle_state": "unknown",
                    "passed": False
                }

    def _determine_validation_mode(self, goal: Goal) -> str:
        """Determine validation mode from goal properties"""
        if goal.is_atomic:
            return "artifact"
        elif goal.goal_type == "continuous":
            return "trend"
        elif goal.goal_type == "directional":
            return "alignment"
        elif goal.goal_type == "exploratory":
            return "hypothesis"
        else:
            # achievable, meta (non-atomic)
            return "aggregate"

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    async def _validate_atomic_artifact(self, goal: Goal, db) -> Dict:
        """
        L3 Atomic goals: artifact-based validation

        Simple but correct for execution tasks
        """
        from artifact_registry import artifact_registry

        artifact_check = await artifact_registry.check_goal_artifacts(str(goal.id))

        has_passed = artifact_check.get("goal_complete", False)
        total = artifact_check.get("total_count", 0)
        passed = artifact_check.get("passed_count", 0)

        return {
            "lifecycle_state": "completed" if has_passed else "active",
            "evaluation_state": "validated" if has_passed else "not_evaluated",
            "passed": has_passed,
            "progress": 1.0 if has_passed else 0.0,
            "confidence": 0.95,
            "validation_mode": "artifact",
            "evidence": [
                f"Artifacts: {total} total, {passed} passed"
            ],
            "recommendations": "Goal complete" if has_passed else "Need passed artifact"
        }

    async def _validate_aggregate(self, goal: Goal, db) -> Dict:
        """
        L0-L2 Achievable goals: children aggregation

        Checks if all subgoals are done
        """
        # Get children
        children_stmt = select(Goal).where(Goal.parent_id == goal.id)
        children_result = await db.execute(children_stmt)
        children = children_result.scalars().all()

        if not children or len(children) == 0:
            return {
                "lifecycle_state": "active",
                "evaluation_state": "not_evaluated",
                "passed": False,
                "progress": 0.0,
                "confidence": 0.5,
                "validation_mode": "aggregate",
                "evidence": ["Goal has no children"],
                "recommendations": "Decompose goal into subgoals"
            }

        # Count progress
        total = len(children)
        done_count = sum(
            1 for c in children
            if c.status in ["done", "completed"]
        )
        progress = done_count / total if total > 0 else 0.0

        # Check completion mode
        if goal.completion_mode == "manual":
            # Manual approval required - never auto-complete
            passed = False
            lifecycle = "active"
        elif goal.completion_mode == "aggregate":
            # All children must be done
            passed = progress >= 1.0
            lifecycle = "completed" if passed else "active"
        else:
            # strict mode - custom evaluator
            passed = progress >= 1.0
            lifecycle = "completed" if passed else "active"

        return {
            "lifecycle_state": lifecycle,
            "evaluation_state": "validated" if passed else "not_evaluated",
            "passed": passed,
            "progress": progress,
            "confidence": 0.8,
            "validation_mode": "aggregate",
            "evidence": [
                f"Children: {done_count}/{total} completed"
            ],
            "recommendations": (
                "Goal complete" if passed else f"Wait for children ({done_count}/{total})"
            )
        }

    async def _validate_trend(self, goal: Goal, db) -> Dict:
        """
        Continuous goals: trend-based validation

        Analyzes historical measurements to determine if improving
        """
        # Get recent state measurements
        states_stmt = select(GoalState)\
            .where(GoalState.goal_id == goal.id)\
            .order_by(GoalState.measured_at.desc())\
            .limit(5)

        states_result = await db.execute(states_stmt)
        states = states_result.scalars().all()

        if not states or len(states) < 2:
            # Need at least 2 measurements to determine trend
            return {
                "lifecycle_state": "ongoing",
                "evaluation_state": "not_evaluated",
                "passed": False,
                "progress": 0.5,
                "confidence": 0.2,
                "validation_mode": "trend",
                "evidence": [
                    f"Insufficient data: {len(states) if states else 0} measurements (need min 2)"
                ],
                "recommendations": "Need more state measurements to determine trend"
            }

        # Analyze trend
        latest = states[0]
        previous = states[-1]

        latest_delta = latest.progress_delta
        previous_delta = previous.progress_delta

        # Determine trend direction
        if latest_delta > previous_delta + 0.05:  # +5% improvement
            trend = "improving"
            passed = True
        elif abs(latest_delta - previous_delta) < 0.05:  # Stable (within ±5%)
            trend = "stable"

            # Check if meeting threshold
            if latest_delta >= (goal.success_threshold or 0.8):
                passed = True
            else:
                passed = False
        else:  # Declining
            trend = "degrading"
            passed = False

        # Calculate progress score (normalize -1..1 to 0..1)
        progress_score = (latest_delta + 1.0) / 2.0

        return {
            "lifecycle_state": "ongoing",
            "evaluation_state": trend,
            "passed": passed,
            "progress": progress_score,
            "confidence": 0.7 if len(states) >= 3 else 0.5,
            "validation_mode": "trend",
            "evidence": [
                f"Latest delta: {latest_delta:+.1%}",
                f"Previous delta: {previous_delta:+.1%}",
                f"Trend: {trend}",
                f"Measurements: {len(states)}"
            ],
            "recommendations": self._trend_recommendation(trend, latest_delta)
        }

    def _trend_recommendation(self, trend: str, current_delta: float) -> str:
        """Generate recommendation based on trend"""
        if trend == "improving":
            if current_delta >= 0.9:
                return "✅ Excellent improvement. Consider raising target."
            return "✅ Good improvement. Continue current strategy."
        elif trend == "stable":
            if current_delta >= 0.8:
                return "✅ On target. Maintain level."
            return "⏳ Stable but not improving. Consider adjustment."
        else:  # degrading
            return "🔴 Declining! Strategy change needed immediately."

    async def _validate_alignment(self, goal: Goal, db) -> Dict:
        """
        Directional goals: alignment-based validation

        Checks if goal direction is being followed
        """
        # Directional goals should have success_definition or target_state
        if not goal.success_definition and not goal.target_state:
            return {
                "lifecycle_state": "permanent",
                "evaluation_state": "not_evaluated",
                "passed": False,
                "progress": 0.5,
                "confidence": 0.3,
                "validation_mode": "alignment",
                "evidence": ["No success criteria defined"],
                "recommendations": "Define success criteria or target state"
            }

        # Get latest state measurement if available
        states_stmt = select(GoalState)\
            .where(GoalState.goal_id == goal.id)\
            .order_by(GoalState.measured_at.desc())\
            .limit(1)

        states_result = await db.execute(states_stmt)
        latest_state = states_result.scalar_one_or_none()

        if not latest_state:
            # No measurements yet - need manual input
            return {
                "lifecycle_state": "permanent",
                "evaluation_state": "not_evaluated",
                "passed": False,
                "progress": 0.5,
                "confidence": 0.3,
                "validation_mode": "alignment",
                "evidence": ["No state measurements"],
                "recommendations": "Measure current state to evaluate alignment"
            }

        # For now, use simple heuristic
        # TODO: Implement proper alignment scoring from success_definition
        current_metrics = latest_state.current_metrics
        target_metrics = goal.target_state or {}

        # Calculate alignment score
        alignments = []
        total_score = 0.0
        metric_count = 0

        for metric_name, target_value in target_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                # Simple alignment: is current moving toward target?
                if target_value > current_value:
                    # Target is higher
                    score = 0.5  # Partial
                elif abs(target_value - current_value) < (target_value * 0.1):
                    # Within ±10%
                    score = 1.0
                else:
                    score = 0.7

                alignments.append({
                    "metric": metric_name,
                    "current": current_value,
                    "target": target_value,
                    "score": score
                })
                total_score += score
                metric_count += 1

        if metric_count == 0:
            return {
                "lifecycle_state": "permanent",
                "evaluation_state": "not_evaluated",
                "passed": False,
                "progress": 0.5,
                "confidence": 0.3,
                "validation_mode": "alignment",
                "evidence": ["Target metrics don't match current measurements"],
                "recommendations": "Align metric definitions"
            }

        avg_score = total_score / metric_count

        # Determine state from score
        if avg_score >= 0.8:
            state = "aligned"
            passed = True
        elif avg_score >= 0.5:
            state = "drifting"
            passed = False
        else:
            state = "critical"
            passed = False

        return {
            "lifecycle_state": "permanent",
            "evaluation_state": state,
            "passed": passed,
            "progress": avg_score,
            "confidence": 0.6,
            "validation_mode": "alignment",
            "evidence": [
                f"Alignment score: {avg_score:.2f}",
                f"Metrics: {len(alignments)} measured"
            ],
            "recommendations": self._alignment_recommendation(state, avg_score),
            "metric_breakdown": alignments
        }

    def _alignment_recommendation(self, state: str, score: float) -> str:
        """Generate recommendation based on alignment"""
        if state == "aligned":
            return "✅ Well aligned. Maintain current direction."
        elif state == "drifting":
            return "⚠️ Slight drift. Re-align to direction."
        else:  # critical
            return "🔴 Critical misalignment! Immediate correction needed."

    async def _validate_hypothesis(self, goal: Goal, db) -> Dict:
        """
        Exploratory goals: hypothesis validation

        Checks if exploration hypothesis was confirmed
        """
        # TODO: Implement hypothesis tracking
        # For now, always require explicit validation
        return {
            "lifecycle_state": "completed",
            "evaluation_state": "not_evaluated",
            "passed": False,
            "progress": 0.5,
            "confidence": 0.3,
            "validation_mode": "hypothesis",
            "evidence": ["Hypothesis validation not yet implemented"],
            "recommendations": "Manual validation required for exploratory goals"
        }


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

outcome_validator = OutcomeValidator()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def validate_goal_outcome(goal_id: str) -> Dict:
    """
    Convenience function to validate goal outcome

    Usage:
        from outcome_validator import validate_goal_outcome

        result = await validate_goal_outcome("goal-uuid")
        if result["passed"]:
            print(f"Goal passed: {result['recommendations']}")
    """
    return await outcome_validator.validate(goal_id)


async def batch_validate_goals(goal_ids: List[str]) -> Dict:
    """
    Validate multiple goals at once

    Returns:
        {
            "total": int,
            "passed": int,
            "failed": int,
            "results": {...}
        }
    """
    results = {}
    passed = 0
    failed = 0

    for goal_id in goal_ids:
        try:
            result = await outcome_validator.validate(goal_id)
            results[goal_id] = result

            if result.get("passed", False):
                passed += 1
            else:
                failed += 1

        except Exception as e:
            results[goal_id] = {
                "error": str(e),
                "passed": False
            }
            failed += 1

    return {
        "total": len(goal_ids),
        "passed": passed,
        "failed": failed,
        "results": results
    }


# =============================================================================
# STATE MEASUREMENT HELPERS
# =============================================================================

async def record_goal_state(
    goal_id: str,
    current_metrics: Dict,
    measured_at: Optional[datetime] = None
) -> Dict:
    """
    Record a state measurement for a goal

    Args:
        goal_id: Goal to measure
        current_metrics: Current state metrics
        measured_at: When measurement was taken (default: now)

    Returns:
        Created GoalState record
    """
    async with AsyncSessionLocal() as db:
        # Get goal
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            raise ValueError(f"Goal not found: {goal_id}")

        # Get baseline (target_state or first measurement)
        target = goal.target_state or {}

        # Calculate delta from baseline
        # For now, use first measurement as baseline
        baseline_stmt = select(GoalState)\
            .where(GoalState.goal_id == goal_id)\
            .order_by(GoalState.measured_at.asc())\
            .limit(1)

        baseline_result = await db.execute(baseline_stmt)
        baseline_state = baseline_result.scalar_one_or_none()

        if baseline_state:
            baseline_metrics = baseline_state.baseline_metrics
        else:
            # No baseline yet, use current as baseline
            baseline_metrics = current_metrics

        # Calculate progress delta
        progress_deltas = {}
        for metric_name, target_value in target.items():
            if metric_name in current_metrics and metric_name in baseline_metrics:
                current = current_metrics[metric_name]
                baseline = baseline_metrics[metric_name]

                if target != baseline:
                    if target > baseline:
                        # Increasing
                        expected_delta = target - baseline
                        actual_delta = current - baseline
                        progress = actual_delta / expected_delta if expected_delta > 0 else 0
                    else:
                        # Decreasing
                        expected_delta = baseline - target
                        actual_delta = baseline - current
                        progress = actual_delta / expected_delta if expected_delta > 0 else 0

                    progress_deltas[metric_name] = progress

        # Create state record
        state = GoalState(
            goal_id=goal.id,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
            target_metrics=target,
            progress_delta=sum(progress_deltas.values()) / len(progress_deltas) if progress_deltas else 0.0,
            measured_at=measured_at or datetime.now()
        )

        db.add(state)
        await db.commit()
        await db.refresh(state)

        return {
            "state_id": str(state.id),
            "goal_id": goal_id,
            "measured_at": state.measured_at.isoformat(),
            "progress_delta": state.progress_delta
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from database import AsyncSessionLocal
    import uuid

    async def test():
        print("Testing OutcomeValidator...")

        # Create test goal
        goal_id = str(uuid.uuid4())

        print(f"Test goal ID: {goal_id}")
        print("Validator initialized successfully")

    asyncio.run(test())
