"""
GOAL VELOCITY ENGINE v1.0
=========================

Measures how fast goals are being achieved.

Key question: "Is the system moving forward?"

Author: AI-OS Core Team
Date: 2026-02-11
Philosophy: Movement over Stagnation
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_
from database import AsyncSessionLocal
from models import Goal, GoalState
from enum import Enum


# =============================================================================
# VELOCITY METRICS
# =============================================================================

class VelocityMetric(str, Enum):
    """Types of velocity measurements"""
    CYCLE_TIME = "cycle_time"  # Days from creation to completion
    COMPLETION_RATE = "completion_rate"  # Goals completed per time
    ACTIVE_GOAL_COUNT = "active_goal_count"  # Current active goals
    STAGNATION_RATIO = "stagnation_ratio"  # Goals > 30 days inactive


class VelocityState(str, Enum):
    """System health based on velocity"""
    ACCELERATING = "accelerating"  # Getting faster
    HEALTHY = "healthy"  # Steady good pace
    SLOWING = "slowing"  # Slowing down
    STAGNANT = "stagnant"  # No progress
    OVERWHELMED = "overwhelmed"  # Too many goals


# =============================================================================
# VELOCITY ENGINE
# =============================================================================

class GoalVelocityEngine:
    """
    Tracks goal velocity and detects stagnation

    Key insight: "What gets measured, gets managed"
    """

    def __init__(self):
        self.metrics_cache = {}
        self.baseline_window_days = 30  # 30-day rolling baseline

    async def calculate_velocity_metrics(
        self,
        goal_type: Optional[str] = None
    ) -> Dict:
        """
        Calculate overall system velocity metrics

        Args:
            goal_type: Filter by goal type (optional)

        Returns:
            {
                "velocity_state": "accelerating|healthy|slowing|stagnant",
                "metrics": {
                    "avg_cycle_time_days": float,
                    "completion_rate_per_month": float,
                    "active_goal_count": int,
                    "stagnation_ratio": float
                },
                "recommendations": [...],
                "timestamp": "..."
            }
        """
        async with AsyncSessionLocal() as db:
            now = datetime.now()

            # Metric 1: Average cycle time
            # Time from creation to completion for completed goals
            stmt_time = select(
                func.avg(
                    func.extract(Goal.created_at, Goal.completed_at) -
                    func.extract(Goal.created_at, Goal.created_at)
                )
            ).where(
                Goal.status == "done",
                Goal.completed_at.isnot(None)
            )

            if goal_type:
                stmt_time = stmt_time.where(Goal.goal_type == goal_type)

            result_time = await db.execute(stmt_time)
            avg_cycle_time = result_time.scalar() or 0

            # Convert days
            avg_cycle_time_days = avg_cycle_time.total_seconds() / 86400

            # Metric 2: Completion rate
            # Goals completed in last 30 days
            thirty_days_ago = now - timedelta(days=30)

            stmt_rate = select(func.count(Goal.id)).where(
                Goal.status == "done",
                Goal.completed_at >= thirty_days_ago
            )

            if goal_type:
                stmt_rate = stmt_rate.where(Goal.goal_type == goal_type)

            result_rate = await db.execute(stmt_rate)
            completed_count = result_rate.scalar() or 0
            completion_rate = completed_count / 30  # per month

            # Metric 3: Active goal count
            stmt_active = select(func.count(Goal.id)).where(
                Goal.status == "active"
            )

            result_active = await db.execute(stmt_active)
            active_count = result_active.scalar() or 0

            # Metric 4: Stagnation ratio
            # Goals that haven't moved in 30 days
            thirty_days_ago = now - timedelta(days=30)

            stmt_stagnant = select(func.count(Goal.id)).where(
                Goal.status == "active",
                Goal.created_at < thirty_days_ago
            )

            result_stagnant = await db.execute(stmt_stagnant)
            stagnant_count = result_stagnant.scalar() or 0

            stagnation_ratio = stagnant_count / active_count if active_count > 0 else 0

            # Determine velocity state
            velocity_state = self._determine_velocity_state(
                avg_cycle_time_days,
                completion_rate,
                active_count,
                stagnation_ratio
            )

            return {
                "velocity_state": velocity_state.value,
                "metrics": {
                    "avg_cycle_time_days": round(avg_cycle_time_days, 2),
                    "completion_rate_per_month": round(completion_rate, 2),
                    "active_goal_count": active_count,
                    "stagnation_ratio": round(stagnation_ratio, 3)
                },
                "recommendations": self._generate_recommendations(
                    velocity_state,
                    avg_cycle_time_days,
                    completion_rate,
                    active_count,
                    stagnation_ratio
                ),
                "timestamp": now.isoformat()
            }

    def _determine_velocity_state(
        self,
        avg_cycle_time: float,
        completion_rate: float,
        active_count: int,
        stagnation_ratio: float
    ) -> VelocityState:
        """
        Determine if system is accelerating, healthy, slowing, or stagnant

        Thresholds based on baseline (last 30 days)
        """
        # Stagnation check
        if stagnation_ratio > 0.5:
            return VelocityState.STAGNANT

        # Slowing check
        if avg_cycle_time > 30:  # 30 days to complete
            return VelocityState.SLOWING

        # Accelerating check
        if avg_cycle_time < 7 and completion_rate > 4:  # < 7 days, > 4/month
            return VelocityState.ACCELERATING

        # Healthy (default)
        return VelocityState.HEALTHY

    def _generate_recommendations(
        self,
        state: VelocityState,
        avg_cycle_time: float,
        completion_rate: float,
        active_count: int,
        stagnation_ratio: float
    ) -> List[str]:
        """Generate recommendations based on velocity state"""
        recommendations = []

        if state == VelocityState.STAGNANT:
            recommendations.append(
                "CRITICAL: System stagnation detected. "
                f"{stagnation_ratio:.1%} of goals inactive >30 days. "
                "Consider: pausing new goal creation, focusing on completions, "
                "reviewing why goals are not progressing."
            )

        elif state == VelocityState.SLOWING:
            recommendations.append(
                f"WARNING: Goal completion slowing down. "
                f"Average cycle time: {avg_cycle_time:.1f} days. "
                "Consider: breaking down complex goals, checking for blockers, "
                "reducing goal scope."
            )

        elif state == VelocityState.ACCELERATING:
            recommendations.append(
                f"GOOD: System accelerating. "
                f"Completion rate: {completion_rate:.1f}/month. "
                "Maintain current approach, but watch for quality degradation."
            )

        elif state == VelocityState.OVERWHELMED:
            recommendations.append(
                f"CRITICAL: Too many active goals: {active_count}. "
                "System overwhelmed. Consider: goal prioritization, "
                "pausing new goal creation, focusing on completions first."
            )

        else:  # HEALTHY
            if active_count > 20:
                recommendations.append(
                    f"INFO: {active_count} active goals. "
                    "Ensure sufficient resources for current load."
                )

        return recommendations

    async def get_goal_velocity(
        self,
        goal_id: str
    ) -> Dict:
        """
        Get velocity metrics for specific goal

        Returns:
            {
                "cycle_time_days": float,
                "is_stagnant": bool,
                "velocity_percentile": "0.0-1.0"
            }
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

            created = goal.created_at
            completed = goal.completed_at

            if not completed:
                # Still active - return creation time
                age_days = (datetime.now() - created).total_seconds() / 86400
                return {
                    "cycle_time_days": age_days,
                    "is_stagnant": age_days > 30,
                    "velocity_percentile": 0.5  # Assume median
                }

            # Calculate cycle time
            cycle_time = (completed - created).total_seconds() / 86400

            # Check stagnation
            is_stagnant = cycle_time > 30

            # Get percentile (simplified)
            # Faster = better
            if cycle_time < 7:
                percentile = 0.9
            elif cycle_time < 14:
                percentile = 0.7
            elif cycle_time < 30:
                percentile = 0.5
            else:
                percentile = 0.3

            return {
                "cycle_time_days": round(cycle_time, 2),
                "is_stagnant": is_stagnant,
                "velocity_percentile": percentile
            }

    async def detect_system_stagnation(self) -> Dict:
        """
        Detect if entire system is stagnating

        Returns:
            {
                "is_stagnant": bool,
                "evidence": {...}
            }
        """
        async with AsyncSessionLocal() as db:
            now = datetime.now()
            thirty_days_ago = now - timedelta(days=30)

            # Count goals created > 30 days ago still active
            stmt_old_active = select(func.count(Goal.id)).where(
                Goal.status == "active",
                Goal.created_at < thirty_days_ago
            )

            result_old_active = await db.execute(stmt_old_active)
            old_active_count = result_old_active.scalar() or 0

            # Count all active
            stmt_all_active = select(func.count(Goal.id)).where(
                Goal.status == "active"
            )

            result_all_active = await db.execute(stmt_all_active)
            all_active_count = result_all_active.scalar() or 0

            # Stagnation ratio
            stagnation_ratio = old_active_count / all_active_count if all_active_count > 0 else 0

            is_stagnant = stagnation_ratio > 0.3

            return {
                "is_stagnant": is_stagnant,
                "evidence": {
                    "old_active_count": old_active_count,
                    "all_active_count": all_active_count,
                    "stagnation_ratio": round(stagnation_ratio, 3),
                    "threshold": "> 30%"
                },
                "recommendations": self._get_stagnation_recommendations(is_stagnant, stagnation_ratio)
            }

    def _get_stagnation_recommendations(self, is_stagnant: bool, ratio: float) -> List[str]:
        """Generate recommendations for stagnation"""
        recommendations = []

        if is_stagnant:
            recommendations.append(
                f"SYSTEM STAGNANT: {ratio:.1%} of goals inactive >30 days."
            )
            recommendations.append("PAUSE: Stop creating new goals until backlog cleared.")
            recommendations.append("FOCUS: Prioritize completing existing active goals.")
            recommendations.append("ANALYZE: Review why 30+ day goals are not progressing.")
        else:
            recommendations.append(f"System healthy: {ratio:.1%} old goals (acceptable).")

        return recommendations


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

goal_velocity_engine = GoalVelocityEngine()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_system_velocity() -> Dict:
    """Get overall system velocity metrics"""
    return await goal_velocity_engine.calculate_velocity_metrics()


async def get_goal_velocity(goal_id: str) -> Dict:
    """Get velocity for specific goal"""
    return await goal_velocity_engine.get_goal_velocity(goal_id)


async def check_stagnation() -> Dict:
    """Check if system is stagnating"""
    return await goal_velocity_engine.detect_system_stagnation()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing Goal Velocity Engine...\n")

        # Test velocity calculation
        metrics = await goal_velocity_engine.calculate_velocity_metrics()

        print("Velocity State:", metrics["velocity_state"])
        print("Metrics:", metrics["metrics"])
        print("Recommendations:", metrics["recommendations"])

    asyncio.run(test())
