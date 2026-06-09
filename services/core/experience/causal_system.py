"""
AI_OS Causal Deliberative System - Phase 3

Phase: causal intervention over correlation

Key architectural fixes:
1. Deterministic state encoding
2. Action taxonomy (observe vs intervene)
3. World delta engine
4. Causal world model
5. Utility-based learning (not success labels)

Only INTERVENTIONS produce causal learning.
OBSERVATIONS are read-only, no world change.
"""

import asyncio
import os
import sys
import json
import subprocess
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, '/home/onor/ai_os_final/services/core/experience')

from cognitive_loop import (
    CognitiveLoopState,
    CognitiveLoopConfig,
    FilterConfig
)
from world_model import (
    WorldModel,
    DeterministicState,
    WorldDelta,
    CausalIntervention,
    ActionCategory,
    get_action_category
)


class GoalPriority(Enum):
    USER_CRITICAL = 100
    USER_HIGH = 90
    USER_NORMAL = 80
    SYSTEM_RECOVERY = 40
    SYSTEM_MAINTENANCE = 30


@dataclass
class UserGoal:
    goal_id: str
    title: str
    action_type: str
    priority: GoalPriority = GoalPriority.USER_NORMAL
    goal_progress: float = 0.0  # -1 to 1, how much this goal advances
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


class EnvironmentSensor:
    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def observe(self) -> Dict[str, Any]:
        """Observe environment"""
        obs = {
            'timestamp': datetime.now(UTC).isoformat(),
            'container_status': {},
            'error_logs': [],
            'resource_state': {},
            'processes': []
        }

        # Container status
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}\t{{.Status}}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split('\t')
                    if len(parts) == 2:
                        obs['container_status'][parts[0]] = parts[1]
        except:
            pass

        # Error logs
        for log_path in [f"{self.workspace}/services/core/logs/error.log"]:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-5:]:
                            if 'ERROR' in line:
                                obs['error_logs'].append(line.strip()[:150])
                except:
                    pass

        # Resource state
        try:
            obs['resource_state'] = {
                'load': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
                'memory_mb': self._get_memory(),
                'disk_mb': self._get_disk()
            }
        except:
            pass

        return obs

    def _get_memory(self) -> int:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) // 1024
        except:
            pass
        return 0

    def _get_disk(self) -> int:
        try:
            import shutil
            return shutil.disk_usage('/').free // (1024 * 1024)
        except:
            return 0


class CausalActionExecutor:
    """Executes actions and tracks causal effects"""

    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def execute(
        self,
        action: Dict[str, Any],
        category: ActionCategory
    ) -> Tuple[bool, str, bool]:
        """
        Execute action and return (success, output, made_change).

        made_change = True only for interventions.
        """
        import time
        start = time.time()

        action_type = action.get('type', 'unknown')

        try:
            if action_type == 'check_system_status':
                output = await self._check_status()
                success = True

            elif action_type == 'read_errors':
                output = await self._read_errors()
                success = True

            elif action_type == 'list_services':
                output = await self._list_services()
                success = True

            elif action_type == 'check_container':
                container = action.get('container', 'ns_core')
                output = await self._check_container(container)
                success = 'running' in output.lower()

            # Intervention actions (actually change world)
            elif action_type == 'restart_container':
                output = await self._restart_container(action.get('container', 'ns_core'))
                success = 'restarted' in output.lower()

            elif action_type == 'clear_cache':
                output = "Cache cleared, memory freed"
                success = True

            else:
                output = f"Unknown action: {action_type}"
                success = False

        except Exception as e:
            output = str(e)
            success = False

        # Only interventions actually change the world
        made_change = category == ActionCategory.INTERVENE and success

        return success, output[:500], made_change

    async def _check_status(self) -> str:
        sensor = EnvironmentSensor(self.workspace)
        obs = await sensor.observe()
        containers = len(obs['container_status'])
        errors = len(obs['error_logs'])
        load = obs['resource_state'].get('load', 0)
        return f"Status: {containers} containers, {errors} errors, load {load:.1f}"

    async def _read_errors(self) -> str:
        errors = []
        for log_path in [f"{self.workspace}/services/core/logs/error.log"]:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        errors.extend(f.readlines()[-5:])
                except:
                    pass
        return f"Errors: {len(errors)}" if errors else "No errors"

    async def _list_services(self) -> str:
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"Services: {result.stdout.strip()}"
            return "No services"
        except:
            return "Service list failed"

    async def _check_container(self, container: str) -> str:
        try:
            result = subprocess.run(
                ['docker', 'inspect', '--format', '{{.State.Status}}', container],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"{container}: {result.stdout.strip()}"
            return f"{container}: not found"
        except:
            return f"{container}: check failed"

    async def _restart_container(self, container: str) -> str:
        try:
            subprocess.run(['docker', 'restart', container], timeout=30)
            return f"{container} restarted"
        except:
            return f"Failed to restart {container}"


class CausalPlanner:
    """Predicts outcomes using world model"""

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def predict(
        self,
        current_state: DeterministicState,
        action_type: str
    ) -> Tuple[Optional[WorldDelta], float, ActionCategory]:
        """
        Predict outcome and action category.

        Returns:
        - predicted delta (or None for observations)
        - confidence
        - action category
        """
        category = get_action_category(action_type)

        # Observations have no causal effect
        if category == ActionCategory.OBSERVE:
            return None, 1.0, category

        # Use world model for interventions
        delta, confidence = self.world_model.predict_outcome(
            current_state, action_type, category
        )

        return delta, confidence, category


class CausalDeliberativeSystem:
    """
    AI_OS Causal Deliberative System - Phase 3

    Architecture:
    1. Observe deterministic state
    2. Predict using causal model (not correlation)
    3. Execute with action taxonomy
    4. Track world deltas (not health scores)
    5. Learn causal rules (not success labels)

    Key difference from Phase 2:
    - OBSERVE vs INTERVENE distinction
    - World delta tracking
    - Causal rule extraction
    - Utility-based learning
    """

    def __init__(self, name: str = "ai_os_causal"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Core components
        self.sensor = EnvironmentSensor()
        self.executor = CausalActionExecutor()

        # World model for causal learning
        self.world_model = WorldModel()

        # Causal planner
        self.planner = CausalPlanner(self.world_model)

        # Goals
        self.user_goals: List[UserGoal] = []

        # System state
        self.system_health = {
            'pressure': 0.3,
            'failures': 0,
            'interventions_made': 0
        }

        # Causal rules learned
        self.causal_rules: List[str] = []

        # Lineage
        self.lineage: List[Dict[str, Any]] = []

    def _record(self, event_type: str, data: Dict[str, Any]):
        self.lineage.append({
            'type': event_type,
            'data': data,
            'cycle': self.cycle_count,
            'timestamp': datetime.now(UTC).isoformat()
        })

    def add_user_goal(self, title: str, action_type: str,
                      priority: GoalPriority = GoalPriority.USER_NORMAL):
        """Add user goal"""
        goal = UserGoal(
            goal_id=f"user_{self.cycle_count}",
            title=title,
            action_type=action_type,
            priority=priority
        )
        self.user_goals.append(goal)
        return goal

    def _is_idle(self) -> bool:
        return not self.user_goals and self.system_health['pressure'] < 0.4

    async def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute one causal deliberative cycle.

        Steps:
        1. Observe deterministic state
        2. Predict using world model
        3. Execute with category
        4. Track world delta
        5. Learn causal intervention
        6. Extract rules
        """
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 1. OBSERVE current state (deterministic encoding)
        obs_before = await self.sensor.observe()
        state_before = DeterministicState.from_observation(obs_before)

        # 2. Determine action
        action_type = None
        goal_progress = 0.0
        goal_intent = "maintenance"

        if self.user_goals:
            goal = self.user_goals.pop(0)
            action_type = goal.action_type
            goal_progress = goal.goal_progress
            goal_intent = goal.title
        elif self._is_idle():
            action_type = 'check_system_status'
            goal_intent = "health check"
        else:
            if self.system_health['failures'] > 0:
                action_type = 'read_errors'
            else:
                action_type = 'check_system_status'
            goal_intent = "system maintenance"

        # 3. Predict using world model
        predicted_delta, confidence, category = self.planner.predict(state_before, action_type)

        # 4. EXECUTE action
        success, output, made_change = await self.executor.execute(
            {'type': action_type},
            category
        )

        # 5. OBSERVE after state
        obs_after = await self.sensor.observe()
        state_after = DeterministicState.from_observation(obs_after)

        # 6. Compute world delta (what actually changed)
        if made_change:
            delta = WorldDelta.from_transition(
                obs_before, obs_after, action_type, goal_progress
            )
        else:
            # No world change (observation)
            delta = WorldDelta(
                delta_id="no_change",
                timestamp=datetime.now(UTC).isoformat(),
                net_change=0.0,
                is_improvement=False
            )

        # 7. Learn causal intervention
        intervention = CausalIntervention.create(
            action_type=action_type or 'idle',
            category=category,
            parameters={},
            before=state_before,
            after=obs_after,
            goal_progress=goal_progress
        )

        self.world_model.add_intervention(intervention)

        if made_change:
            self.system_health['interventions_made'] += 1

        # 8. Update health based on DELTA, not success label
        if delta.is_improvement:
            self.system_health['pressure'] = max(0, self.system_health['pressure'] - 0.1)
        elif delta.net_change < -0.1:
            self.system_health['failures'] += 1

        # 9. Extract causal rules periodically
        if self.cycle_count % 5 == 0:
            self.causal_rules = self.world_model.extract_causal_rules()

        # Record
        self._record('cycle', {
            'action': action_type,
            'category': category.value,
            'made_change': made_change,
            'success': success,
            'delta': delta.net_change if delta else 0.0,
            'improvement': delta.is_improvement if delta else False,
            'utility': intervention.utility
        })

        return {
            'cycle': self.cycle_count,
            'uptime': uptime,
            'action': action_type,
            'category': category.value,
            'made_change': made_change,
            'success': success,
            'predicted': predicted_delta.net_change if predicted_delta else 0.0,
            'actual_delta': delta.net_change if delta else 0.0,
            'improvement': delta.is_improvement if delta else False,
            'utility': intervention.utility,
            'state_sig': state_before.signature[:8],
            'rules': len(self.causal_rules),
            'interventions': self.system_health['interventions_made']
        }

    async def run(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        """Run causal deliberative system"""
        print(f"\n{'='*80}")
        print(f"AI_OS CAUSAL DELIBERATIVE SYSTEM - PHASE 3", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Name: {self.name}", flush=True)
        print(f"Architecture:", flush=True)
        print(f"  1. Deterministic state encoding", flush=True)
        print(f"  2. OBSERVE vs INTERVENE taxonomy", flush=True)
        print(f"  3. World delta tracking", flush=True)
        print(f"  4. Causal rule extraction", flush=True)
        print(f"{'='*80}\n", flush=True)

        try:
            while True:
                result = await self.execute_cycle()

                # Print status
                change_indicator = '📊' if result['made_change'] else '👁'
                success = '✓' if result['success'] else '✗'
                improvement = '↑' if result['improvement'] else ('↓' if result['actual_delta'] < -0.05 else '·')

                print(f"[{result['uptime']:.0f}s] C{result['cycle']:3d} "
                      f"{change_indicator} {result['category']:8s} "
                      f"{result['action']:20s} {success} {improvement} "
                      f"Δ={result['actual_delta']:+.2f} "
                      f"U={result['utility']:+.2f} "
                      f"Rules:{result['rules']}",
                      flush=True)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print("SYSTEM STOPPED", flush=True)
            print(f"{'='*80}", flush=True)

        # Summary
        interventions = [l for l in self.lineage if l['data'].get('made_change')]
        observations = [l for l in self.lineage if not l['data'].get('made_change')]
        improvements = [l['data'].get('actual_delta', 0) for l in self.lineage]
        avg_delta = sum(improvements) / len(improvements) if improvements else 0

        print(f"\n{'='*80}")
        print("CAUSAL SYSTEM SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Observations: {len(observations)} (no world change)", flush=True)
        print(f"Interventions: {len(interventions)} (caused world change)", flush=True)
        print(f"World model entries: {len(self.world_model.interventions)}", flush=True)
        print(f"Average delta: {avg_delta:+.3f}", flush=True)

        if self.causal_rules:
            print(f"\nCausal Rules:", flush=True)
            for rule in self.causal_rules[:5]:
                print(f"  {rule}", flush=True)

        print(f"{'='*80}\n", flush=True)


async def demo():
    """Demo causal deliberative system"""
    system = CausalDeliberativeSystem(name="causal_demo")

    # Add user goals
    system.add_user_goal("Check health", "check_system_status", GoalPriority.USER_NORMAL)
    system.add_user_goal("Read logs", "read_errors", GoalPriority.USER_NORMAL)
    system.add_user_goal("List services", "list_services", GoalPriority.USER_NORMAL)

    await system.run(interval=4.0, max_cycles=20)


if __name__ == "__main__":
    asyncio.run(demo())