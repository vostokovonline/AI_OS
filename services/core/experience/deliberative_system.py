"""
AI_OS Deliberative System - From Reactive to Deliberative

Phase transition: reactive → deliberative

Key changes:
1. Experience Memory (episodes, state transitions)
2. Predictive planning (simulate before execute)
3. Causal understanding (state → action → outcome)
4. Heuristic extraction (learn from patterns)
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
from experience_model import (
    ExperienceMemory,
    EnvironmentState,
    StateTransition,
    Episode,
    OutcomeType,
    create_episode_from_execution
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
    created_at: str = ""
    predicted_outcome: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


class EnvironmentSensor:
    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def observe(self) -> Dict[str, Any]:
        """Observe current environment state"""
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


class ActionExecutor:
    """Executes actions and returns results"""

    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def execute(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute action, return (success, output)"""
        import time
        start = time.time()

        action_type = action.get('type', 'unknown')
        goal_id = action.get('goal_id', 'unknown')

        try:
            if action_type == 'check_system_status':
                output = await self._check_status()
                success = True

            elif action_type == 'read_errors':
                output = await self._read_errors()
                success = True

            elif action_type == 'check_container':
                container = action.get('container', 'ns_core')
                output = await self._check_container(container)
                success = 'running' in output.lower()

            elif action_type == 'list_services':
                output = await self._list_services()
                success = True

            else:
                output = f"Unknown action: {action_type}"
                success = False

        except Exception as e:
            output = str(e)
            success = False

        return success, output[:500]

    async def _check_status(self) -> str:
        sensor = EnvironmentSensor(self.workspace)
        obs = await sensor.observe()
        containers = len(obs['container_status'])
        errors = len(obs['error_logs'])
        load = obs['resource_state'].get('load', 0)
        return f"Containers:{containers} Errors:{errors} Load:{load:.1f}"

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


class DeliberativePlanner:
    """
    Deliberative planning - BEFORE action, predict outcome.

    Key difference from reactive:
    - Simulate before execute
    - Predict outcome
    - Estimate risk
    - Compare strategies
    """

    def __init__(self, experience: ExperienceMemory):
        self.experience = experience
        self.prediction_count = 0

    def predict_outcome(
        self,
        current_state: EnvironmentState,
        action_type: str,
        action_params: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Predict outcome of action.

        Returns:
        - predicted outcome type
        - confidence (0-1)
        """
        self.prediction_count += 1

        # Use experience to predict
        similar = self.experience.get_similar_episodes(current_state, action_type, limit=3)

        if similar:
            # Base prediction on similar episodes
            best = similar[0]
            episode, similarity = best

            if episode.success:
                predicted = OutcomeType.IMPROVEMENT.value
                confidence = similarity * episode.success_score
            else:
                predicted = OutcomeType.DEGRADATION.value
                confidence = similarity * (1.0 - episode.success_score)
        else:
            # No experience - assume neutral
            predicted = OutcomeType.NO_CHANGE.value
            confidence = 0.3

        return predicted, min(1.0, confidence + 0.2)

    def estimate_risk(
        self,
        current_state: EnvironmentState,
        action_type: str
    ) -> float:
        """Estimate risk of action (0-1, higher = riskier)"""
        similar = self.experience.get_similar_episodes(current_state, action_type, limit=5)

        if not similar:
            return 0.5  # Unknown = moderate risk

        # Compute risk from failures
        failures = sum(1 for ep, _ in similar if not ep.success)
        return failures / len(similar)

    def select_best_strategy(
        self,
        current_state: EnvironmentState,
        possible_actions: List[str]
    ) -> Tuple[Optional[str], float]:
        """Select best action based on experience and predictions"""
        best_action = None
        best_score = 0.0

        for action in possible_actions:
            # Get success rate from experience
            success_rate = self.experience.get_pattern_success_rate(action)

            # Get risk
            risk = self.estimate_risk(current_state, action)

            # Score = success - risk penalty
            score = success_rate - (risk * 0.3)

            if score > best_score:
                best_score = score
                best_action = action

        return best_action, best_score


class AIAutonomousDeliberativeSystem:
    """
    AI_OS Deliberative Cognitive System

    From reactive to deliberative:
    1. Observe state
    2. Predict outcomes
    3. Select best strategy
    4. Execute
    5. Learn from transitions
    6. Extract heuristics

    Key innovation: experience memory enables learning from past.
    """

    def __init__(self, name: str = "ai_os_deliberative"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Core components
        self.sensor = EnvironmentSensor()
        self.executor = ActionExecutor()
        self.experience = ExperienceMemory()

        # Deliberative planner
        self.planner = DeliberativePlanner(self.experience)

        # Goal management
        self.user_goals: List[UserGoal] = []
        self.completed_goals: List[Dict[str, Any]] = []

        # System health
        self.system_health = {
            'pressure': 0.3,
            'failures': 0
        }

        # Learned heuristics
        self.heuristics: List[str] = []

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
        """Add user goal for execution"""
        goal = UserGoal(
            goal_id=f"user_{self.cycle_count}_{len(self.user_goals)}",
            title=title,
            action_type=action_type,
            priority=priority
        )
        self.user_goals.append(goal)
        self._record('goal_added', {'title': title, 'action': action_type})
        return goal

    def _is_idle(self) -> bool:
        return not self.user_goals and self.system_health['pressure'] < 0.4

    async def execute_deliberative_cycle(self) -> Dict[str, Any]:
        """
        Execute one deliberative cycle.

        Steps:
        1. Observe current state
        2. (If goal) Predict outcome, select strategy
        3. Execute
        4. Observe new state
        5. Learn from transition
        """
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 1. Observe BEFORE state
        obs_before = await self.sensor.observe()
        state_before = EnvironmentState.from_observation(obs_before)

        # Determine action
        action_type = None
        goal_intent = "maintenance"
        predicted_outcome = "unknown"

        if self.user_goals:
            # User goal - highest priority
            goal = self.user_goals.pop(0)
            action_type = goal.action_type
            goal_intent = goal.title
        elif self._is_idle():
            # Idle - check system health
            if obs_before['error_logs']:
                action_type = 'read_errors'
            else:
                action_type = 'check_system_status'
            goal_intent = "health check"
        else:
            # System maintenance needed
            if self.system_health['failures'] > 0:
                action_type = 'read_errors'
                goal_intent = "recovery"
            else:
                action_type = 'check_system_status'
                goal_intent = "maintenance"

        # 2. Predict outcome BEFORE execution
        if action_type:
            predicted_outcome, confidence = self.planner.predict_outcome(
                state_before, action_type, {}
            )
        else:
            predicted_outcome = "idle"
            confidence = 1.0

        # 3. Execute action
        success, output = await self.executor.execute({
            'type': action_type,
            'goal_id': f"cycle_{self.cycle_count}"
        })

        # 4. Observe AFTER state
        obs_after = await self.sensor.observe()
        state_after = EnvironmentState.from_observation(obs_after)

        # 5. Learn from state transition
        episode = create_episode_from_execution(
            before_state=state_before,
            after_state=state_after,
            action_type=action_type or 'idle',
            action_params={},
            goal_intent=goal_intent
        )
        self.experience.add_episode(episode)

        # Update health
        if success:
            self.system_health['pressure'] = max(0, self.system_health['pressure'] - 0.1)
            if self.system_health['failures'] > 0:
                self.system_health['failures'] -= 1
        else:
            self.system_health['failures'] += 1

        # Extract heuristics periodically
        if self.cycle_count % 5 == 0:
            self.heuristics = self.experience.extract_heuristics()

        # Record in lineage
        self._record('cycle_completed', {
            'action': action_type,
            'success': success,
            'predicted': predicted_outcome,
            'outcome': episode.transition.outcome_type.value,
            'improvement': episode.transition.improvement_score
        })

        return {
            'cycle': self.cycle_count,
            'action': action_type,
            'success': success,
            'predicted': predicted_outcome,
            'actual_outcome': episode.transition.outcome_type.value,
            'improvement': episode.transition.improvement_score,
            'confidence': confidence,
            'experiences': len(self.experience.episodes),
            'heuristics': len(self.heuristics)
        }

    async def run(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        """Run the deliberative system"""
        print(f"\n{'='*75}")
        print(f"AI_OS DELIBERATIVE COGNITIVE SYSTEM", flush=True)
        print(f"{'='*75}", flush=True)
        print(f"Name: {self.name}", flush=True)
        print(f"Architecture: Observe → Predict → Execute → Learn", flush=True)
        print(f"Experience Memory: {len(self.experience.episodes)} episodes", flush=True)
        print(f"{'='*75}\n", flush=True)

        try:
            while True:
                result = await self.execute_deliberative_cycle()

                # Print status
                uptime = (datetime.now(UTC) - self.start_time).total_seconds()
                action = result['action'] or 'idle'
                success = '✓' if result['success'] else '✗'
                predicted = result['predicted'][:10]
                actual = result['actual_outcome'][:10]
                exp = result['experiences']
                heur = result['heuristics']

                print(f"[{uptime:.0f}s] C{self.cycle_count:3d} | "
                      f"{action:20s} | {success} | "
                      f"Pred:{predicted:10s} Act:{actual:10s} | "
                      f"Exp:{exp} Heur:{heur}",
                      flush=True)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*75}")
            print("SYSTEM STOPPED", flush=True)
            print(f"{'='*75}", flush=True)

        # Summary
        success_rate = sum(1 for l in self.lineage if l['data'].get('success')) / max(1, len(self.lineage)) * 100
        improvements = [l['data'].get('improvement', 0) for l in self.lineage]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0

        print(f"\n{'='*75}")
        print("DELIBERATIVE SYSTEM SUMMARY", flush=True)
        print(f"{'='*75}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Success rate: {success_rate:.1f}%", flush=True)
        print(f"Experiences: {len(self.experience.episodes)}", flush=True)
        print(f"Heuristics learned: {len(self.heuristics)}", flush=True)
        print(f"Average improvement: {avg_improvement:.3f}", flush=True)
        print(f"Lineage events: {len(self.lineage)}", flush=True)
        if self.heuristics:
            print(f"\nLearned Heuristics:", flush=True)
            for h in self.heuristics[:5]:
                print(f"  - {h}", flush=True)
        print(f"{'='*75}\n", flush=True)


async def demo():
    """Demo deliberative system"""
    system = AIAutonomousDeliberativeSystem(name="deliberative_demo")

    # Add user goals
    system.add_user_goal("Check system health", "check_system_status", GoalPriority.USER_NORMAL)
    system.add_user_goal("Read error logs", "read_errors", GoalPriority.USER_HIGH)
    system.add_user_goal("Verify services", "list_services", GoalPriority.USER_NORMAL)

    await system.run(interval=4.0, max_cycles=18)


if __name__ == "__main__":
    asyncio.run(demo())