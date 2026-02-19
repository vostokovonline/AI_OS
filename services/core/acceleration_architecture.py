"""
ACCELERATION ARCHITECTURE v1.0
==============================

Orchestrates velocity, drift detection, and intervention.
Provides 30% control / 70% acceleration balance.

Key principle: "Movement over Perfection, Discipline over Chaos"

Author: AI-OS Core Team
Date: 2026-02-11
Philosophy: Controlled Acceleration
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum

# Import our three layers
from goal_velocity_engine import goal_velocity_engine, VelocityState
from strategic_drift_detector import strategic_drift_detector, DriftSeverity
from ai_intervention_layer import ai_intervention_layer, InterventionPriority


# =============================================================================
# ACCELERATION STATE
# =============================================================================

class AccelerationState(str, Enum):
    """Overall system acceleration state"""
    CRITICAL_BLOCKED = "critical_blocked"  # Stop everything, fix problems
    DECELERATING = "decelerating"  # Slow down, fix drift
    STABLE = "stable"  # Normal operation
    ACCELERATING = "accelerating"  # Speed up, good velocity
    MAXIMUM_THRUST = "maximum_thrust"  # Full speed ahead


class AccelerationAction(str, Enum):
    """Types of acceleration actions"""
    PAUSE_NEW_GOALS = "pause_new_goals"  # Stop creating goals
    FOCUS_COMPLETIONS = "focus_completions"  # Finish existing goals
    INCREASE_DEPTH = "increase_depth"  # Allow deeper decomposition
    REDUCE_DEPTH = "reduce_depth"  # Limit decomposition
    INCREASE_PARALLEL = "increase_parallel"  # More parallel execution
    RESTRUCTURE = "restructure"  # Restructure goals
    TRANSFORM_TYPE = "transform_type"  # Change goal types
    NORMAL = "normal"  # No special action


# =============================================================================
# ACCELERATION ORCHESTRATOR
# =============================================================================

class AccelerationArchitecture:
    """
    Orchestrates velocity, drift, and intervention layers

    Implements 30% control / 70% acceleration:
    - 30%: Protection (invariants, checks)
    - 70%: Acceleration (velocity, optimization, movement)
    """

    def __init__(self):
        self.velocity_weight = 0.4  # 40% weight to velocity
        self.drift_weight = 0.35  # 35% weight to drift detection
        self.intervention_weight = 0.25  # 25% weight to interventions

    async def get_system_health(self) -> Dict:
        """
        Get complete system health report

        Returns:
            {
                "overall_state": "critical_blocked|decelerating|stable|accelerating|maximum_thrust",
                "confidence": 0.0-1.0,
                "velocity": {...},
                "drift": {...},
                "interventions": {...},
                "recommended_actions": [...],
                "control_vs_acceleration": {
                    "control": 0.0-1.0,
                    "acceleration": 0.0-1.0
                },
                "timestamp": "..."
            }
        """
        # Gather data from all three layers
        velocity_report = await goal_velocity_engine.calculate_velocity_metrics()
        drift_report = await strategic_drift_detector.detect_all_drifts()
        intervention_report = await ai_intervention_layer.scan_for_interventions()

        # Calculate overall state
        overall_state = self._calculate_overall_state(
            velocity_report, drift_report, intervention_report
        )

        # Calculate control/acceleration balance
        balance = self._calculate_balance(
            velocity_report, drift_report, intervention_report
        )

        # Generate recommended actions
        actions = self._generate_actions(
            overall_state, velocity_report, drift_report, intervention_report
        )

        return {
            "overall_state": overall_state.value,
            "confidence": self._calculate_confidence(velocity_report, drift_report),
            "velocity": velocity_report,
            "drift": drift_report,
            "interventions": intervention_report,
            "recommended_actions": actions,
            "control_vs_acceleration": balance,
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_overall_state(
        self,
        velocity: Dict,
        drift: Dict,
        interventions: Dict
    ) -> AccelerationState:
        """
        Calculate overall system state

        Priority order:
        1. Critical issues → CRITICAL_BLOCKED
        2. Many drifts/interventions → DECELERATING
        3. Good velocity → ACCELERATING
        4. Normal → STABLE
        """
        # Check 1: Critical drift or urgent interventions
        critical_drifts = sum(
            1 for d in drift.get("drifts", [])
            if d.get("severity") == "critical"
        )

        urgent_interventions = interventions.get("by_priority", {}).get("urgent", 0)

        if critical_drifts >= 2 or urgent_interventions >= 3:
            return AccelerationState.CRITICAL_BLOCKED

        # Check 2: High drifts or high-priority interventions
        high_drifts = sum(
            1 for d in drift.get("drifts", [])
            if d.get("severity") in ["high", "critical"]
        )

        high_interventions = (
            interventions.get("by_priority", {}).get("high", 0) +
            interventions.get("by_priority", {}).get("urgent", 0)
        )

        if high_drifts >= 3 or high_interventions >= 5:
            return AccelerationState.DECELERATING

        # Check 3: Excellent velocity
        velocity_state = velocity.get("velocity_state")

        if velocity_state == "accelerating":
            # Check if it's safe to accelerate
            if critical_drifts == 0 and urgent_interventions == 0:
                return AccelerationState.MAXIMUM_THRUST
            else:
                return AccelerationState.ACCELERATING

        # Check 4: Healthy velocity
        if velocity_state == "healthy":
            return AccelerationState.STABLE

        # Check 5: Slowing or stagnant
        if velocity_state in ["slowing", "stagnant"]:
            # If slowing but no critical issues, still stable
            if critical_drifts == 0:
                return AccelerationState.STABLE
            else:
                return AccelerationState.DECELERATING

        # Default: stable
        return AccelerationState.STABLE

    def _calculate_balance(
        self,
        velocity: Dict,
        drift: Dict,
        interventions: Dict
    ) -> Dict:
        """
        Calculate control vs acceleration balance

        Returns:
            {
                "control": 0.0-1.0,  # How much protection needed
                "acceleration": 0.0-1.0,  # How much acceleration possible
                "ratio": "control:acceleration"
            }
        """
        # Control factors (higher = need more control)
        drift_count = drift.get("drifts_detected", 0)
        intervention_count = interventions.get("interventions_required", 0)
        stagnation_ratio = velocity.get("metrics", {}).get("stagnation_ratio", 0)

        # Normalize to 0-1
        control_score = min(1.0, (
            (drift_count / 10) * 0.4 +
            (intervention_count / 20) * 0.4 +
            stagnation_ratio * 0.2
        ))

        # Acceleration factors (higher = can accelerate)
        velocity_state = velocity.get("velocity_state")
        completion_rate = velocity.get("metrics", {}).get("completion_rate_per_month", 0)

        # Velocity score (accelerating = high, stagnant = low)
        if velocity_state == "accelerating":
            velocity_score = 1.0
        elif velocity_state == "healthy":
            velocity_score = 0.7
        elif velocity_state == "slowing":
            velocity_score = 0.4
        elif velocity_state == "stagnant":
            velocity_score = 0.2
        else:  # overwhelmed
            velocity_score = 0.0

        # Normalize completion rate (assume 10/month = excellent)
        completion_score = min(1.0, completion_rate / 10)

        acceleration_score = (
            velocity_score * 0.7 +
            completion_score * 0.3
        )

        # Ensure control + acceleration = 1 (normalize)
        total = control_score + acceleration_score
        if total > 0:
            control_normalized = control_score / total
            acceleration_normalized = acceleration_score / total
        else:
            control_normalized = 0.3  # Default 30% control
            acceleration_normalized = 0.7  # Default 70% acceleration

        # Format ratio
        control_pct = int(control_normalized * 100)
        accel_pct = int(acceleration_normalized * 100)

        return {
            "control": round(control_normalized, 2),
            "acceleration": round(acceleration_normalized, 2),
            "ratio": f"{control_pct}:{accel_pct}",
            "interpretation": self._interpret_balance(
                control_normalized, acceleration_normalized
            )
        }

    def _interpret_balance(self, control: float, acceleration: float) -> str:
        """Interpret control/acceleration balance"""
        if control > 0.7:
            return "HIGH CONTROL: System needs correction, limit new goals"
        elif control > 0.5:
            return "MODERATE CONTROL: Some issues, proceed with caution"
        elif control > 0.3:
            return "BALANCED: 30% control / 70% acceleration (optimal)"
        elif control > 0.2:
            return "LEANING ACCELERATION: Good velocity, can optimize"
        else:
            return "MAXIMUM ACCELERATION: Clear path, full speed ahead"

    def _generate_actions(
        self,
        state: AccelerationState,
        velocity: Dict,
        drift: Dict,
        interventions: Dict
    ) -> List[Dict]:
        """
        Generate recommended actions based on state

        Returns:
            [
                {
                    "action": "...",
                    "priority": "...",
                    "reason": "...",
                    "expected_effect": "..."
                }
            ]
        """
        actions = []

        # State-based actions
        if state == AccelerationState.CRITICAL_BLOCKED:
            actions.append({
                "action": AccelerationAction.PAUSE_NEW_GOALS.value,
                "priority": "URGENT",
                "reason": "Critical drift/interventions detected",
                "expected_effect": "Stabilize system, focus on fixing problems"
            })
            actions.append({
                "action": AccelerationAction.FOCUS_COMPLETIONS.value,
                "priority": "URGENT",
                "reason": "Too many stuck goals",
                "expected_effect": "Clear backlog before continuing"
            })

        elif state == AccelerationState.DECELERATING:
            actions.append({
                "action": AccelerationAction.REDUCE_DEPTH.value,
                "priority": "HIGH",
                "reason": "System drifting, simplify decomposition",
                "expected_effect": "Reduce complexity, focus on completions"
            })
            actions.append({
                "action": AccelerationAction.FOCUS_COMPLETIONS.value,
                "priority": "HIGH",
                "reason": "Drift detected",
                "expected_effect": "Stabilize before accelerating"
            })

        elif state == AccelerationState.STABLE:
            actions.append({
                "action": AccelerationAction.NORMAL.value,
                "priority": "INFO",
                "reason": "System stable, continue normal operation",
                "expected_effect": "Maintain current pace"
            })

        elif state == AccelerationState.ACCELERATING:
            actions.append({
                "action": AccelerationAction.INCREASE_PARALLEL.value,
                "priority": "MEDIUM",
                "reason": "Good velocity, safe to increase parallelism",
                "expected_effect": "Increase throughput without breaking stability"
            })

        elif state == AccelerationState.MAXIMUM_THRUST:
            actions.append({
                "action": AccelerationAction.INCREASE_DEPTH.value,
                "priority": "MEDIUM",
                "reason": "Excellent velocity, can handle more complexity",
                "expected_effect": "Optimize for maximum output"
            })
            actions.append({
                "action": AccelerationAction.INCREASE_PARALLEL.value,
                "priority": "MEDIUM",
                "reason": "Maximum thrust mode",
                "expected_effect": "Full speed ahead"
            })

        # Add specific actions based on metrics
        stagnation_ratio = velocity.get("metrics", {}).get("stagnation_ratio", 0)

        if stagnation_ratio > 0.3:
            actions.append({
                "action": AccelerationAction.RESTRUCTURE.value,
                "priority": "HIGH",
                "reason": f"{stagnation_ratio:.1%} goals stagnating",
                "expected_effect": "Clear stuck goals, improve flow"
            })

        # Add drift-specific actions
        recurrent_failures = [
            d for d in drift.get("drifts", [])
            if d.get("type") == "recurrent_failure"
        ]

        if len(recurrent_failures) >= 2:
            actions.append({
                "action": AccelerationAction.TRANSFORM_TYPE.value,
                "priority": "HIGH",
                "reason": f"{len(recurrent_failures)} recurrent failure patterns",
                "expected_effect": "Fix goal type mismatches"
            })

        return actions

    def _calculate_confidence(self, velocity: Dict, drift: Dict) -> float:
        """
        Calculate confidence in overall state assessment

        Higher confidence when:
        - More data points (more goals)
        - Consistent signals across layers
        """
        # Factor 1: Data volume (goal count)
        active_goals = velocity.get("metrics", {}).get("active_goal_count", 0)
        data_score = min(1.0, active_goals / 50)  # 50 goals = full confidence

        # Factor 2: Signal consistency
        # Do velocity and drift agree on system state?
        velocity_state = velocity.get("velocity_state")
        drift_status = drift.get("overall_status")

        if velocity_state == "accelerating" and drift_status == "healthy":
            consistency_score = 1.0
        elif velocity_state == "healthy" and drift_status == "monitoring":
            consistency_score = 0.8
        elif velocity_state == "stagnant" and drift_status == "critical":
            consistency_score = 0.9
        else:
            consistency_score = 0.6  # Mixed signals

        # Weighted average
        confidence = (data_score * 0.4) + (consistency_score * 0.6)

        return round(confidence, 2)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

acceleration_architecture = AccelerationArchitecture()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_system_health() -> Dict:
    """Get complete system health report"""
    return await acceleration_architecture.get_system_health()


async def get_acceleration_state() -> AccelerationState:
    """Get current acceleration state only"""
    health = await acceleration_architecture.get_system_health()
    return AccelerationState(health["overall_state"])


async def get_control_acceleration_balance() -> Dict:
    """Get control/acceleration balance only"""
    health = await acceleration_architecture.get_system_health()
    return health["control_vs_acceleration"]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing Acceleration Architecture...\n")
        print("=" * 70)

        # Get full health report
        health = await acceleration_architecture.get_system_health()

        print(f"\nOVERALL STATE: {health['overall_state']}")
        print(f"Confidence: {health['confidence']:.0%}")

        print("\n" + "=" * 70)
        print("CONTROL vs ACCELERATION")
        print("=" * 70)
        balance = health['control_vs_acceleration']
        print(f"Ratio: {balance['ratio']}")
        print(f"Control: {balance['control']:.2%}")
        print(f"Acceleration: {balance['acceleration']:.2%}")
        print(f"\nInterpretation: {balance['interpretation']}")

        print("\n" + "=" * 70)
        print("VELOCITY")
        print("=" * 70)
        print(f"State: {health['velocity']['velocity_state']}")
        print(f"Cycle Time: {health['velocity']['metrics']['avg_cycle_time_days']} days")
        print(f"Completion Rate: {health['velocity']['metrics']['completion_rate_per_month']}/month")
        print(f"Stagnation Ratio: {health['velocity']['metrics']['stagnation_ratio']:.1%}")

        print("\n" + "=" * 70)
        print("DRIFT DETECTION")
        print("=" * 70)
        print(f"Status: {health['drift']['overall_status']}")
        print(f"Drifts Detected: {health['drift']['drifts_detected']}")

        print("\n" + "=" * 70)
        print("INTERVENTIONS")
        print("=" * 70)
        print(f"Required: {health['interventions']['interventions_required']}")
        print(f"By Priority: {health['interventions']['by_priority']}")

        print("\n" + "=" * 70)
        print("RECOMMENDED ACTIONS")
        print("=" * 70)
        for action in health['recommended_actions']:
            print(f"\n[{action['priority']}] {action['action']}")
            print(f"  Reason: {action['reason']}")
            print(f"  Effect: {action['expected_effect']}")

        print("\n" + "=" * 70)

    asyncio.run(test())
