"""
AI_OS Phase 6 - Latent Causal World Model

Phase: From prediction statistics → generative causal model

Key innovations:
1. Latent causes (not just observed metrics)
2. Causal graph (cause → effect relationships)
3. Generative simulation (counterfactual futures)
4. Active inference (act to reduce uncertainty)
5. Ontology revision (model restructuring on catastrophe)

NOT just: error_count += avg_delta
BUT: "service_instability_causing_cascade_failures"
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
    BeliefState
)
from prediction_engine import PredictionEngine, TemporalMemory
from phase6_world_model import (
    Phase6WorldModel,
    LatentCause,
    LatentState,
    GenerativeSimulator,
    ActiveInferenceEngine,
    OntologyRevisionEngine
)


class GoalPriority(Enum):
    USER_CRITICAL = 100
    USER_HIGH = 90
    USER_NORMAL = 80


@dataclass
class UserGoal:
    goal_id: str
    title: str
    action_type: str
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


class EnvironmentSensor:
    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def observe(self) -> Dict[str, Any]:
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


class Phase6Executor:
    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def execute(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        action_type = action.get('type', 'unknown')

        try:
            if action_type == 'check_system_status':
                output = await self._check_status()
            elif action_type == 'read_errors':
                output = await self._read_errors()
            elif action_type == 'restart_service':
                output = "Service restart simulated"
            elif action_type == 'diagnose':
                output = await self._diagnose()
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

    async def _diagnose(self) -> str:
        sensor = EnvironmentSensor(self.workspace)
        obs = await sensor.observe()
        return f"Diagnostic: {len(obs['error_logs'])} errors, {len(obs['container_status'])} containers"


class Phase6LatentCausalSystem:
    """
    AI_OS Phase 6 - Latent Causal World Model

    Architecture:
    1. INFER LATENT CAUSE from observations
    2. SIMULATE counterfactual futures
    3. SELECT ACTION using active inference
    4. EXECUTE and OBSERVE
    5. HANDLE prediction error (ontology revision if needed)

    Key innovations:
    - NOT just metrics, but CAUSAL INTERPRETATION
    - NOT single prediction, but SIMULATION TREE
    - NOT just exploit, but ACTIVE INFERENCE
    - NOT confidence adjustment, but ONTOLOGY REVISION
    """

    def __init__(self, name: str = "ai_os_phase6"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Phase 5 components
        self.prediction_engine = PredictionEngine()
        self.temporal_memory = TemporalMemory()

        # Phase 6: Latent causal world model
        self.world_model = Phase6WorldModel()

        # Components
        self.sensor = EnvironmentSensor()
        self.executor = Phase6Executor()

        # Goals
        self.user_goals: List[UserGoal] = []

        # Metrics
        self.latent_inferences = 0
        self.simulations_run = 0
        self.active_inference_actions = 0
        self.ontology_revisions = 0
        self.explorations = 0
        self.exploitations = 0

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
        Execute Phase 6 cognitive cycle.

        Steps:
        1. OBSERVE environment
        2. INFER LATENT CAUSE
        3. SIMULATE counterfactual futures
        4. SELECT ACTION (active inference)
        5. EXECUTE
        6. OBSERVE result
        7. HANDLE prediction error / ontology revision
        """
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 1. OBSERVE environment
        obs = await self.sensor.observe()
        env_state = EnvironmentState.from_observation(obs)

        # 2. INFER LATENT CAUSE
        error_count = len(obs.get('error_logs', []))
        container_count = len(obs.get('container_status', {}))
        resource_load = obs.get('resource_state', {}).get('load', 0)

        latent = self.world_model.infer_latent_state(
            error_count=error_count,
            container_count=container_count,
            resource_load=resource_load
        )
        self.latent_inferences += 1

        # 3. SIMULATE counterfactual futures
        available_actions = ['check_system_status', 'read_errors', 'restart_service', 'diagnose']

        if self.user_goals:
            action_type = self.user_goals.pop(0).action_type
        else:
            # Use active inference to select action
            action_type, was_exploration = self.world_model.select_action_active_inference(
                available_actions
            )
            if was_exploration:
                self.explorations += 1
            else:
                self.exploitations += 1

        self.simulations_run += 1

        # Get semantic prediction
        semantic_pred = self.world_model.get_semantic_prediction(action_type)

        # 4. EXECUTE action
        success, output = await self.executor.execute({'type': action_type})

        # 5. OBSERVE result
        obs_after = await self.sensor.observe()
        error_after = len(obs_after.get('error_logs', []))

        # 6. HANDLE prediction error
        prediction_error = abs(error_after - error_count) / max(1, error_count + 1)

        needs_revision = self.world_model.handle_prediction_error(
            error_magnitude=prediction_error,
            failed_prediction=semantic_pred
        )

        if needs_revision:
            self.ontology_revisions += 1

        # Learn from transition
        self.prediction_engine.learn_from_transition(
            'error_count', action_type, error_count, error_after
        )

        # Record
        self._record('latent_causal_cycle', {
            'latent_cause': latent.primary_cause,
            'instability': latent.instability_score,
            'cascade_risk': latent.cascade_risk,
            'action': action_type,
            'semantic_prediction': semantic_pred,
            'prediction_error': prediction_error,
            'ontology_revision': needs_revision
        })

        # Icon
        if needs_revision:
            icon = "💥"
        elif latent.primary_cause == LatentCause.STABLE.value:
            icon = "✓"
        elif latent.instability_score > 0.5:
            icon = "⚠"
        else:
            icon = "🎯"

        return {
            'cycle': self.cycle_count,
            'uptime': uptime,
            'latent_cause': latent.primary_cause,
            'cause_confidence': latent.cause_confidence,
            'instability': latent.instability_score,
            'cascade_risk': latent.cascade_risk,
            'action': action_type,
            'semantic_prediction': semantic_pred[:40],
            'prediction_error': prediction_error,
            'ontology_revisions': self.ontology_revisions,
            'explorations': self.explorations,
            'exploitations': self.exploitations,
            'success': success
        }

    async def run(self, interval: float = 4.0, max_cycles: Optional[int] = None):
        """Run Phase 6 system"""
        print(f"\n{'='*80}")
        print(f"AI_OS PHASE 6 - LATENT CAUSAL WORLD MODEL", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Key Innovations:", flush=True)
        print(f"  1. Latent causes (not just metrics)", flush=True)
        print(f"  2. Causal graph (cause → effect)", flush=True)
        print(f"  3. Generative simulation (counterfactual futures)", flush=True)
        print(f"  4. Active inference (epistemic actions)", flush=True)
        print(f"  5. Ontology revision (model restructuring)", flush=True)
        print(f"{'='*80}\n", flush=True)

        try:
            while True:
                result = await self.execute_cycle()

                success = '✓' if result['success'] else '✗'
                cause = result['latent_cause'].replace('_', ' ')[:20]

                print(f"[{result['uptime']:.0f}s] C{result['cycle']:3d} {result['action']:15s} {success} "
                      f"Cause:{cause:20s} Instab:{result['instability']:.1f} "
                      f"Sim:{self.simulations_run} "
                      f"Expl:{result['explorations']} Exp:{result['exploitations']} "
                      f"Rev:{result['ontology_revisions']}",
                      flush=True)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print("SYSTEM STOPPED", flush=True)
            print(f"{'='*80}", flush=True)

        print(f"\n{'='*80}")
        print("PHASE 6 SUMMARY - LATENT CAUSAL WORLD MODEL", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Latent inferences: {self.latent_inferences}", flush=True)
        print(f"Simulations run: {self.simulations_run}", flush=True)
        print(f"Active inference actions: {self.explorations} explorations, {self.exploitations} exploitations", flush=True)
        print(f"Ontology revisions: {self.ontology_revisions}", flush=True)
        print(f"{'='*80}\n", flush=True)


async def demo():
    """Demo Phase 6 system"""
    system = Phase6LatentCausalSystem(name="phase6_demo")

    system.add_user_goal("Check system", "check_system_status")
    system.add_user_goal("Read errors", "read_errors")
    system.add_user_goal("Diagnose", "diagnose")

    await system.run(interval=4.0, max_cycles=25)


if __name__ == "__main__":
    asyncio.run(demo())