"""
AI_OS Phase 5 - Predictive Cognitive Architecture

Phase: From retrospective to predictive cognition

Key architectural changes:
1. World Transition Model: P(next_state | current_state, action)
2. Prediction BEFORE execution
3. Compare predictions vs reality
4. Learn from prediction errors (not reward shaping)
5. Temporal episodes with causal chains

This is the CORE of cognitive architecture:
- Predict before act
- Compare prediction with reality
- Learn from error
- Update model
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

from three_layer_state import (
    TripleLayerState,
    EnvironmentState,
    BeliefState,
    CognitiveState
)
from prediction_engine import (
    PredictionEngine,
    TemporalMemory,
    TemporalEpisode,
    Prediction,
    PredictionError
)
from world_model import ActionCategory, get_action_category


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
            'resource_state': {}
        }

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


class Phase5Executor:
    """Executes actions"""

    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def execute(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Execute action, return (success, output)"""
        action_type = action.get('type', 'unknown')

        try:
            if action_type == 'check_system_status':
                output = await self._check_status()
            elif action_type == 'read_errors':
                output = await self._read_errors()
            elif action_type == 'list_services':
                output = await self._list_services()
            else:
                output = f"Unknown: {action_type}"

            return True, output[:500]

        except Exception as e:
            return False, str(e)

    async def _check_status(self) -> str:
        sensor = EnvironmentSensor(self.workspace)
        obs = await sensor.observe()
        return f"Containers:{len(obs['container_status'])} Errors:{len(obs['error_logs'])}"

    async def _read_errors(self) -> str:
        errors = []
        for log_path in [f"{self.workspace}/services/core/logs/error.log"]:
            if os.path.exists(log_path):
                try:
                    with open(log_path, 'r') as f:
                        errors.extend(f.readlines()[-5:])
                except:
                    pass
        return f"Errors: {len(errors)}"

    async def _list_services(self) -> str:
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return f"Services: {len(result.stdout.strip().split())}"
            return "No services"
        except:
            return "Check failed"


class Phase5PredictiveSystem:
    """
    AI_OS Phase 5 - Predictive Cognitive Architecture

    Architecture:
    1. PREDICT: Before action, predict outcomes
    2. ACT: Execute action
    3. OBSERVE: See actual outcome
    4. COMPARE: Compute prediction error
    5. LEARN: Update transition model, revise beliefs

    Key innovation: PREDICTION BEFORE EXECUTION
    - This makes cognition "predictive" not just "reactive"
    - Prediction error is THE learning signal
    - Not reward shaping, not utility scalars
    """

    def __init__(self, name: str = "ai_os_phase5"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Three-layer state
        self.state = TripleLayerState.initial()

        # Prediction engine - THE CORE innovation
        self.prediction_engine = PredictionEngine()

        # Temporal memory
        self.temporal_memory = TemporalMemory()

        # Components
        self.sensor = EnvironmentSensor()
        self.executor = Phase5Executor()

        # Goals
        self.user_goals: List[UserGoal] = []

        # Episode tracking
        self.episode_actions: List[str] = []
        self.episode_predictions: List[Prediction] = []
        self.episode_initial_state: Optional[Dict] = None

        # Metrics
        self.predictions_made = 0
        self.prediction_errors_detected = 0
        self.episodes_completed = 0

        # Lineage
        self.lineage: List[Dict[str, Any]] = []

    def _record(self, event_type: str, data: Dict[str, Any]):
        self.lineage.append({
            'type': event_type,
            'data': data,
            'cycle': self.cycle_count,
            'timestamp': datetime.now(UTC).isoformat()
        })

    def add_user_goal(self, title: str, action_type: str):
        goal = UserGoal(
            goal_id=f"goal_{len(self.user_goals)}",
            title=title,
            action_type=action_type
        )
        self.user_goals.append(goal)

    async def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute one Phase 5 predictive cycle.

        Steps:
        1. Observe environment state
        2. Update beliefs
        3. PREDICT before action
        4. Execute action
        5. Observe outcome
        6. Compare prediction vs reality
        7. Learn from prediction error
        8. Create temporal episode
        """
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 1. OBSERVE environment
        obs_before = await self.sensor.observe()
        env_before = EnvironmentState.from_observation(obs_before)

        # 2. Update BELIEF state (observation updates beliefs)
        belief_before = self.state.belief
        belief_after = belief_before.with_observation(env_before, 0, len(env_before.containers))
        self.state = self.state.with_belief_revision(belief_after)

        # 3. Determine action
        action_type = None
        if self.user_goals:
            goal = self.user_goals.pop(0)
            action_type = goal.action_type
        else:
            action_type = 'check_system_status'

        category = get_action_category(action_type)

        # 4. PREDICT BEFORE EXECUTING (this is the key!)
        aspects_to_predict = ['error_count', 'container_count', 'resource_load']
        current_state = {
            'error_count': len(obs_before.get('error_logs', [])),
            'container_count': len(obs_before.get('container_status', {})),
            'resource_load': obs_before.get('resource_state', {}).get('load', 0)
        }

        predictions = self.prediction_engine.predict_before_action(
            aspects=aspects_to_predict,
            action=action_type,
            current_state=current_state,
            state_signature=env_before.signature()
        )

        self.episode_predictions = predictions
        self.episode_actions.append(action_type)
        self.predictions_made += len(predictions)

        if self.episode_initial_state is None:
            self.episode_initial_state = current_state.copy()

        # 5. EXECUTE action
        success, output = await self.executor.execute({'type': action_type})

        # 6. OBSERVE after state
        obs_after = await self.sensor.observe()
        env_after = EnvironmentState.from_observation(obs_after)

        # 7. COMPARE predictions vs reality
        actual_state = {
            'error_count': len(obs_after.get('error_logs', [])),
            'container_count': len(obs_after.get('container_status', {})),
            'resource_load': obs_after.get('resource_state', {}).get('load', 0)
        }

        errors = self.prediction_engine.evaluate_predictions(
            actual_state, env_after.signature()
        )

        self.prediction_errors_detected += len(errors)

        # 8. LEARN from prediction errors
        for aspect in aspects_to_predict:
            before_val = current_state.get(aspect)
            after_val = actual_state.get(aspect)
            if before_val is not None and after_val is not None:
                self.prediction_engine.learn_from_transition(
                    aspect, action_type, before_val, after_val
                )

        # 9. Learn for transition model
        for aspect in aspects_to_predict:
            self.prediction_engine.learn_from_transition(
                aspect, action_type,
                current_state.get(aspect, 0),
                actual_state.get(aspect, 0)
            )

        # 10. Create temporal episode every N cycles
        if self.cycle_count % 5 == 0 and self.episode_initial_state:
            episode = TemporalEpisode.create(
                initial_state=self.episode_initial_state,
                state_signature=env_before.signature(),
                actions=self.episode_actions,
                predictions=self.episode_predictions,
                outcomes=actual_state,
                errors=errors,
                episode_type="exploration"
            )
            self.temporal_memory.add_episode(episode)
            self.episodes_completed += 1
            self.episode_actions = []
            self.episode_predictions = []
            self.episode_initial_state = None

        # Get prediction accuracy
        accuracy, error_count = self.prediction_engine.get_prediction_accuracy()

        # Record
        self._record('predictive_cycle', {
            'action': action_type,
            'predictions': len(predictions),
            'errors': len(errors),
            'accuracy': accuracy,
            'confidence': sum(p.confidence for p in predictions) / max(1, len(predictions))
        })

        # Icon based on prediction quality
        if errors:
            worst_error = max(errors, key=lambda e: e.error_magnitude)
            if worst_error.error_type == "catastrophic":
                icon = "💥"
            elif worst_error.error_type == "surprise":
                icon = "❓"
            else:
                icon = "📊"
        elif predictions:
            icon = "🎯"
        else:
            icon = "👁"

        return {
            'cycle': self.cycle_count,
            'uptime': uptime,
            'action': action_type,
            'category': category.value,
            'predictions_made': self.predictions_made,
            'errors_detected': self.prediction_errors_detected,
            'prediction_accuracy': accuracy,
            'confidence': sum(p.confidence for p in predictions) / max(1, len(predictions)),
            'episodes': self.episodes_completed,
            'success': success
        }

    async def run(self, interval: float = 4.0, max_cycles: Optional[int] = None):
        """Run Phase 5 predictive system"""
        print(f"\n{'='*80}")
        print(f"AI_OS PHASE 5 - PREDICTIVE COGNITIVE ARCHITECTURE", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Core Innovation: PREDICTION BEFORE EXECUTION", flush=True)
        print(f"Learning Signal: Prediction Error (not reward)", flush=True)
        print(f"Architecture:", flush=True)
        print(f"  1. PREDICT: Before action, predict outcomes", flush=True)
        print(f"  2. ACT: Execute action", flush=True)
        print(f"  3. OBSERVE: See actual outcome", flush=True)
        print(f"  4. COMPARE: Compute prediction error", flush=True)
        print(f"  5. LEARN: Update transition model", flush=True)
        print(f"{'='*80}\n", flush=True)

        try:
            while True:
                result = await self.execute_cycle()

                success = '✓' if result['success'] else '✗'
                acc = f"{result['prediction_accuracy']:.0%}"

                print(f"[{result['uptime']:.0f}s] C{result['cycle']:3d} "
                      f"{result['action']:20s} {success} "
                      f"P:{result['predictions_made']} E:{result['errors_detected']} "
                      f"Acc:{acc} Conf:{result['confidence']:.1f} "
                      f"Episodes:{result['episodes']}",
                      flush=True)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print("SYSTEM STOPPED", flush=True)
            print(f"{'='*80}", flush=True)

        # Summary
        accuracy, error_count = self.prediction_engine.get_prediction_accuracy()

        print(f"\n{'='*80}")
        print("PHASE 5 SUMMARY - PREDICTIVE COGNITIVE ARCHITECTURE", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Predictions made: {self.predictions_made}", flush=True)
        print(f"Prediction errors: {self.prediction_errors_detected}", flush=True)
        print(f"Overall accuracy: {accuracy:.1%}", flush=True)
        print(f"Temporal episodes: {self.episodes_completed}", flush=True)

        # Model confidence for each aspect
        for aspect in ['error_count', 'container_count', 'resource_load']:
            conf = self.prediction_engine.get_model_confidence(aspect, 'check_system_status')
            print(f"  {aspect} model confidence: {conf:.1%}", flush=True)

        print(f"{'='*80}\n", flush=True)


async def demo():
    """Demo Phase 5 predictive system"""
    system = Phase5PredictiveSystem(name="phase5_demo")

    system.add_user_goal("Check system", "check_system_status")
    system.add_user_goal("Read errors", "read_errors")
    system.add_user_goal("List services", "list_services")

    await system.run(interval=4.0, max_cycles=25)


if __name__ == "__main__":
    asyncio.run(demo())