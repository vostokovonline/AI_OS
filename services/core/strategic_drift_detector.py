"""
STRATEGIC DRIFT DETECTOR v1.0
==============================

Detects when system drifts from reality.

Key question: "Is the system learning or repeating mistakes?"

Author: AI-OS Core Team
Date: 2026-02-11
Philosophy: Reality over Self-Deception
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, or_
from database import AsyncSessionLocal
from models import Goal, GoalRelation, Artifact
from enum import Enum
import re


# =============================================================================
# DRIFT TYPES
# =============================================================================

class DriftType(str, Enum):
    """Types of strategic drift"""
    RECURRENT_FAILURE = "recurrent_failure"  # Same goal repeats without success
    OVERESTIMATION = "overestimation"  # System estimates > capability
    REALITY_DEVIATION = "reality_deviation"  # Planned ≠ Actual
    SCOPE_CREEP = "scope_creep"  # Goals keep expanding
    ABANDONMENT_PATTERN = "abandonment_pattern"  # Similar goals repeatedly abandoned


class DriftSeverity(str, Enum):
    """Severity levels for drift"""
    LOW = "low"  # Monitor
    MEDIUM = "medium"  # Warning
    HIGH = "high"  # Action required
    CRITICAL = "critical"  # Immediate intervention


# =============================================================================
# DRIFT DETECTOR
# =============================================================================

class StrategicDriftDetector:
    """
    Detects strategic drift in goal system

    Key insight: "Without feedback, systems drift into delusion"
    """

    def __init__(self):
        self.pattern_cache = {}
        self.similarity_threshold = 0.7  # For goal similarity

    async def detect_all_drifts(self) -> Dict:
        """
        Run all drift detection algorithms

        Returns:
            {
                "overall_status": "healthy|drifting|critical",
                "drifts_detected": int,
                "drifts_by_type": {...},
                "drifts": [
                    {
                        "type": "...",
                        "severity": "...",
                        "description": "...",
                        "evidence": {...},
                        "recommendation": "..."
                    }
                ],
                "timestamp": "..."
            }
        """
        drifts = []

        # Detection 1: Recurrent failures
        recurrent_drifts = await self._detect_recurrent_failures()
        drifts.extend(recurrent_drifts)

        # Detection 2: Overestimation by goal type
        overestimation_drifts = await self._detect_overestimation()
        drifts.extend(overestimation_drifts)

        # Detection 3: Reality deviation
        reality_drifts = await self._detect_reality_deviation()
        drifts.extend(reality_drifts)

        # Detection 4: Scope creep
        scope_drifts = await self._detect_scope_creep()
        drifts.extend(scope_drifts)

        # Detection 5: Abandonment patterns
        abandonment_drifts = await self._detect_abandonment_patterns()
        drifts.extend(abandonment_drifts)

        # Calculate overall status
        overall_status = self._calculate_overall_status(drifts)

        # Group by type
        drifts_by_type = {}
        for drift in drifts:
            drift_type = drift["type"]
            if drift_type not in drifts_by_type:
                drifts_by_type[drift_type] = []
            drifts_by_type[drift_type].append(drift)

        return {
            "overall_status": overall_status,
            "drifts_detected": len(drifts),
            "drifts_by_type": {
                dt: len(drifts_by_type.get(dt, []))
                for dt in [d.value for d in DriftType]
            },
            "drifts": drifts,
            "timestamp": datetime.now().isoformat()
        }

    async def _detect_recurrent_failures(self) -> List[Dict]:
        """
        Detect: Same goal type repeats without success

        Pattern:
        - Goal created → failed → similar goal created → failed ...
        """
        async with AsyncSessionLocal() as db:
            # Find failed goals in last 90 days
            ninety_days_ago = datetime.now() - timedelta(days=90)

            stmt_failed = select(Goal).where(
                Goal.status == "failed",
                Goal.created_at >= ninety_days_ago
            ).order_by(Goal.created_at.desc())

            result_failed = await db.execute(stmt_failed)
            failed_goals = result_failed.scalars().all()

            # Group by similarity
            recurrence_groups = []

            for failed_goal in failed_goals:
                # Find similar goals that also failed
                similar = await self._find_similar_failed_goals(
                    db, failed_goal, failed_goals
                )

                if len(similar) >= 2:  # At least 2 similar failures
                    recurrence_groups.append({
                        "goal_pattern": failed_goal.title[:50],
                        "recurrence_count": len(similar) + 1,
                        "failed_goals": [g.id for g in similar] + [failed_goal.id],
                        "goal_type": failed_goal.goal_type
                    })

            # Generate drift alerts
            drifts = []

            for group in recurrence_groups:
                if group["recurrence_count"] >= 3:
                    severity = DriftSeverity.CRITICAL
                elif group["recurrence_count"] == 2:
                    severity = DriftSeverity.HIGH
                else:
                    severity = DriftSeverity.MEDIUM

                drifts.append({
                    "type": DriftType.RECURRENT_FAILURE.value,
                    "severity": severity.value,
                    "description": (
                        f"Recurrent failure pattern: '{group['goal_pattern']}' "
                        f"has failed {group['recurrence_count']} times"
                    ),
                    "evidence": {
                        "pattern": group["goal_pattern"],
                        "recurrence_count": group["recurrence_count"],
                        "failed_goal_ids": group["failed_goals"],
                        "goal_type": group["goal_type"]
                    },
                    "recommendation": self._get_recurrent_failure_recommendation(
                        group["goal_type"], group["recurrence_count"]
                    )
                })

            return drifts

    async def _find_similar_failed_goals(
        self,
        db,
        goal: Goal,
        failed_goals: List[Goal]
    ) -> List[Goal]:
        """Find goals similar to given goal (simple title match)"""
        similar = []

        # Extract keywords from title
        keywords = set(re.findall(r'\w+', goal.title.lower()))

        for other in failed_goals:
            if other.id == goal.id:
                continue

            other_keywords = set(re.findall(r'\w+', other.title.lower()))

            # Jaccard similarity
            intersection = keywords & other_keywords
            union = keywords | other_keywords

            if len(union) > 0:
                similarity = len(intersection) / len(union)

                if similarity >= self.similarity_threshold:
                    similar.append(other)

        return similar

    async def _detect_overestimation(self) -> List[Dict]:
        """
        Detect: Goal types with low completion rates

        Pattern:
        - Type X created 100 times
        - Type X completed 20 times
        - → 80% failure rate = OVERESTIMATION
        """
        async with AsyncSessionLocal() as db:
            # Count goals by type
            stmt_by_type = select(
                Goal.goal_type,
                func.count(Goal.id).label("total_count"),
                func.sum(func.cast(Goal.status == "done", db.integer())).label("completed_count")
            ).group_by(Goal.goal_type)

            result_by_type = await db.execute(stmt_by_type)
            type_stats = result_by_type.all()

            drifts = []

            for goal_type, total, completed in type_stats:
                if total < 5:  # Skip if too few data points
                    continue

                completion_rate = completed / total

                # Threshold: < 50% completion = overestimation
                if completion_rate < 0.5:
                    if completion_rate < 0.2:
                        severity = DriftSeverity.CRITICAL
                    elif completion_rate < 0.35:
                        severity = DriftSeverity.HIGH
                    else:
                        severity = DriftSeverity.MEDIUM

                    drifts.append({
                        "type": DriftType.OVERESTIMATION.value,
                        "severity": severity.value,
                        "description": (
                            f"Overestimation detected for '{goal_type}' goals: "
                            f"{completion_rate:.1%} completion rate ({completed}/{total})"
                        ),
                        "evidence": {
                            "goal_type": goal_type,
                            "completion_rate": round(completion_rate, 3),
                            "total_goals": total,
                            "completed_goals": completed,
                            "threshold": "< 50%"
                        },
                        "recommendation": self._get_overestimation_recommendation(
                            goal_type, completion_rate
                        )
                    })

            return drifts

    async def _detect_reality_deviation(self) -> List[Dict]:
        """
        Detect: Planned estimates ≠ Actual reality

        Pattern:
        - Goal estimated: 1 day
        - Goal actual: 14 days
        - → 13x deviation = REALITY DEVIATION
        """
        async with AsyncSessionLocal() as db:
            # Find completed goals with created_at and completed_at
            stmt = select(Goal).where(
                Goal.status == "done",
                Goal.completed_at.isnot(None)
            )

            result = await db.execute(stmt)
            completed_goals = result.scalars().all()

            drifts = []

            for goal in completed_goals:
                # Calculate actual duration
                actual_days = (goal.completed_at - goal.created_at).total_seconds() / 86400

                # Expected duration (simplified: atomic = 1 day, others = 7 days)
                if goal.is_atomic:
                    expected_days = 1
                else:
                    expected_days = 7

                # Deviation ratio
                if expected_days > 0:
                    deviation_ratio = actual_days / expected_days
                else:
                    deviation_ratio = 1.0

                # Threshold: > 3x deviation
                if deviation_ratio > 3.0:
                    if deviation_ratio > 10.0:
                        severity = DriftSeverity.CRITICAL
                    elif deviation_ratio > 5.0:
                        severity = DriftSeverity.HIGH
                    else:
                        severity = DriftSeverity.MEDIUM

                    drifts.append({
                        "type": DriftType.REALITY_DEVIATION.value,
                        "severity": severity.value,
                        "description": (
                            f"Reality deviation for '{goal.title[:40]}': "
                            f"took {actual_days:.1f} days vs {expected_days} expected "
                            f"({deviation_ratio:.1f}x slower)"
                        ),
                        "evidence": {
                            "goal_id": str(goal.id),
                            "goal_title": goal.title[:50],
                            "expected_days": expected_days,
                            "actual_days": round(actual_days, 1),
                            "deviation_ratio": round(deviation_ratio, 1)
                        },
                        "recommendation": self._get_reality_deviation_recommendation(
                            deviation_ratio
                        )
                    })

            # Only return top 10 most severe
            drifts.sort(key=lambda d: d["evidence"]["deviation_ratio"], reverse=True)
            return drifts[:10]

    async def _detect_scope_creep(self) -> List[Dict]:
        """
        Detect: Goals that keep expanding

        Pattern:
        - Goal created with 3 subgoals
        - Later: 10 subgoals
        - Later: 25 subgoals
        - → SCOPE CREEP
        """
        async with AsyncSessionLocal() as db:
            # Find goals with many children
            stmt = select(
                Goal.parent_id,
                func.count(Goal.id).label("child_count")
            ).where(
                Goal.parent_id.isnot(None)
            ).group_by(Goal.parent_id).having(
                func.count(Goal.id) > 10  # More than 10 children
            )

            result = await db.execute(stmt)
            parent_goals = result.all()

            drifts = []

            for parent_id, child_count in parent_goals:
                # Get parent goal
                stmt_parent = select(Goal).where(Goal.id == parent_id)
                result_parent = await db.execute(stmt_parent)
                parent = result_parent.scalar_one_or_none()

                if not parent:
                    continue

                # Check if parent is still active (shouldn't have this many children)
                if parent.status == "active":
                    severity = DriftSeverity.HIGH if child_count > 20 else DriftSeverity.MEDIUM

                    drifts.append({
                        "type": DriftType.SCOPE_CREEP.value,
                        "severity": severity.value,
                        "description": (
                            f"Scope creep detected: '{parent.title[:40]}' has "
                            f"{child_count} subgoals (suggests goal decomposition problem)"
                        ),
                        "evidence": {
                            "parent_goal_id": str(parent_id),
                            "parent_title": parent.title[:50],
                            "child_count": child_count,
                            "parent_status": parent.status
                        },
                        "recommendation": (
                            f"Review goal '{parent.title[:40]}' for over-decomposition. "
                            f"Consider: (1) Merging related subgoals, "
                            f"(2) Raising completion threshold, "
                            f"(3) Splitting into multiple parent goals."
                        )
                    })

            return drifts

    async def _detect_abandonment_patterns(self) -> List[Dict]:
        """
        Detect: Similar goals repeatedly abandoned

        Pattern:
        - Goal created → abandoned
        - Similar goal created → abandoned
        - → ABANDONMENT PATTERN
        """
        async with AsyncSessionLocal() as db:
            # Find goals stuck in "active" > 60 days
            sixty_days_ago = datetime.now() - timedelta(days=60)

            stmt_abandoned = select(Goal).where(
                Goal.status == "active",
                Goal.created_at < sixty_days_ago
            )

            result_abandoned = await db.execute(stmt_abandoned)
            abandoned_goals = result_abandoned.scalars().all()

            # Group by domain (simplified: by goal_type)
            abandoned_by_type = {}

            for goal in abandoned_goals:
                if goal.goal_type not in abandoned_by_type:
                    abandoned_by_type[goal.goal_type] = []
                abandoned_by_type[goal.goal_type].append(goal)

            drifts = []

            for goal_type, goals in abandoned_by_type.items():
                if len(goals) >= 3:  # At least 3 abandoned of same type
                    severity = (
                        DriftSeverity.CRITICAL if len(goals) >= 10
                        else DriftSeverity.HIGH if len(goals) >= 5
                        else DriftSeverity.MEDIUM
                    )

                    drifts.append({
                        "type": DriftType.ABANDONMENT_PATTERN.value,
                        "severity": severity.value,
                        "description": (
                            f"Abandonment pattern: {len(goals)} '{goal_type}' goals "
                            f"abandoned (>60 days inactive)"
                        ),
                        "evidence": {
                            "goal_type": goal_type,
                            "abandoned_count": len(goals),
                            "abandoned_goal_ids": [str(g.id) for g in goals[:5]]
                        },
                        "recommendation": (
                            f"Review {len(goals)} abandoned '{goal_type}' goals. "
                            f"Consider: (1) Completing or cancelling them, "
                            f"(2) Identifying why they're stuck, "
                            f"(3) Adjusting goal type or completion criteria."
                        )
                    })

            return drifts

    def _calculate_overall_status(self, drifts: List[Dict]) -> str:
        """Calculate overall system health based on drifts"""
        if not drifts:
            return "healthy"

        critical_count = sum(1 for d in drifts if d["severity"] == "critical")
        high_count = sum(1 for d in drifts if d["severity"] == "high")

        if critical_count >= 2:
            return "critical"
        elif critical_count >= 1 or high_count >= 3:
            return "drifting"
        else:
            return "monitoring"

    def _get_recurrent_failure_recommendation(self, goal_type: str, count: int) -> str:
        """Generate recommendation for recurrent failures"""
        return (
            f"CRITICAL: Goal pattern has failed {count} times. "
            f"STOP creating similar goals. Root cause analysis required. "
            f"Consider: (1) Is this goal achievable? (2) Is execution flawed? "
            f"(3) Should goal type be changed from '{goal_type}'?"
        )

    def _get_overestimation_recommendation(self, goal_type: str, rate: float) -> str:
        """Generate recommendation for overestimation"""
        return (
            f"System consistently overestimates capability for '{goal_type}' goals. "
            f"Current completion rate: {rate:.1%}. "
            f"Consider: (1) Reducing max_depth for this type, "
            f"(2) Increasing resource allocation, (3) Re-classifying goal type."
        )

    def _get_reality_deviation_recommendation(self, ratio: float) -> str:
        """Generate recommendation for reality deviation"""
        return (
            f"Planning estimates deviate {ratio:.1f}x from reality. "
            f"Estimation system needs calibration. "
            f"Consider: (1) Tracking actual vs estimated times, "
            f"(2) Adjusting planning heuristics, (3) Adding buffer to estimates."
        )


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

strategic_drift_detector = StrategicDriftDetector()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def detect_all_drifts() -> Dict:
    """Run all drift detection algorithms"""
    return await strategic_drift_detector.detect_all_drifts()


async def get_recurrent_failures() -> List[Dict]:
    """Get only recurrent failure patterns"""
    return await strategic_drift_detector._detect_recurrent_failures()


async def get_overestimation_by_type() -> List[Dict]:
    """Get only overestimation patterns"""
    return await strategic_drift_detector._detect_overestimation()


async def get_reality_deviation() -> List[Dict]:
    """Get only reality deviation patterns"""
    return await strategic_drift_detector._detect_reality_deviation()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        logger.info("Testing Strategic Drift Detector...\n")

        # Run all detections
        drifts = await strategic_drift_detector.detect_all_drifts()

        logger.info(f"Overall Status: {drifts['overall_status']}")
        logger.info(f"Drifts Detected: {drifts['drifts_detected']}")
        logger.info("\nDrifts by Type:")
        for drift_type, count in drifts['drifts_by_type'].items():
            logger.info(f"  {drift_type}: {count}")

        logger.info("\nTop Drifts:")
        for drift in drifts['drifts'][:5]:
            logger.info(f"  [{drift['severity'].upper()}] {drift['description']}")

    asyncio.run(test())
