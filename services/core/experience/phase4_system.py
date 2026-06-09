"""
AI_OS Phase 4 - Three-Layer Causal System

Phase: Three-layer state model with proper belief revision

Key architectural fixes:
1. Environment State - physical world (only interventions change)
2. Belief State - agent's beliefs (observations change this)
3. Cognitive State - internal processing

Critical insight:
- OBSERVE: world unchanged, BELIEF revised
- INTERVENE: world changed, BELIEF revised (indirectly)

No more reward shaping - counterfactual outcomes instead.
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
    CognitiveState,
    CounterfactualOutcome,
    CausalGraph
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


class Phase4Executor:
    """Executes actions and tracks effects"""

    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def execute(self, action: Dict[str, Any], category: ActionCategory):
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


class Phase4System:
    """
    AI_OS Phase 4 - Three-Layer Causal System

    Architecture:
    1. Observe environment -> update BELIEF state
    2. Plan intervention -> predict with causal graph
    3. Execute intervention -> update ENVIRONMENT state
    4. Observe result -> update BELIEF state
    5. Learn counterfactuals -> update causal graph

    Key innovations:
    - Belief state updated by OBSERVE (not just world)
    - Counterfactual outcomes (not utility scalar)
    - Causal graph for intervention chains
    - Uncertainty tracking
    """

    def __init__(self, name: str = "ai_os_phase4"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Three-layer state
        self.state = TripleLayerState.initial()

        # Components
        self.sensor = EnvironmentSensor()
        self.executor = Phase4Executor()
        self.causal_graph = CausalGraph()

        # Goals
        self.user_goals: List[UserGoal] = []

        # Tracking
        self.observations_made = 0
        self.interventions_made = 0
        self.belief_revisions = 0
        self.last_error_count = 0
        self.last_container_count = 0

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
        Execute one Phase 4 cycle.

        Key insight:
        - OBSERVE: updates BELIEF state, not environment
        - INTERVENE: updates ENVIRONMENT, then BELIEF (indirectly)
        """
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 1. OBSERVE environment
        obs = await self.sensor.observe()
        env_state = EnvironmentState.from_observation(obs)

        # 2. Update BELIEF state from observation
        # THIS IS THE KEY: observation changes beliefs, not environment
        old_belief = self.state.belief
        prev_errors = self.last_error_count
        prev_containers = self.last_container_count
        
        new_belief = old_belief.with_observation(
            env_state, prev_errors, prev_containers
        )
        self.state = self.state.with_belief_revision(new_belief)
        self.last_error_count = env_state.error_count
        self.last_container_count = len(env_state.containers)
        self.belief_revisions += 1

        self.observations_made += 1

        self._record('observation', {
            'env_signature': env_state.signature(),
            'belief_confidence': self.state.belief.infrastructure_confidence,
            'active_hypotheses': len(self.state.belief.active_hypotheses)
        })

        # 3. Determine action
        action_type = None

        if self.user_goals:
            goal = self.user_goals.pop(0)
            action_type = goal.action_type
        elif self.state.belief.anomaly_confidence > 0.6:
            # High anomaly - investigate
            action_type = 'read_errors'
        else:
            action_type = 'check_system_status'

        category = get_action_category(action_type)

        # 4. Execute action
        success, output = await self.executor.execute(
            {'type': action_type}, category
        )

        # 5. For INTERVENTIONS: observe result, create counterfactual
        if category == ActionCategory.INTERVENE:
            # Observe after state
            obs_after = await self.sensor.observe()
            env_after = EnvironmentState.from_observation(obs_after)

            # Update environment state
            self.state = self.state.with_environment_update(env_after)

            # Create counterfactual outcome
            counterfactual = CounterfactualOutcome.from_intervention(
                action_type, env_state, env_after, self.state.belief
            )

            # Add to causal graph
            self.causal_graph.add_intervention(action_type, counterfactual)

            self.interventions_made += 1

            self._record('intervention', {
                'action': action_type,
                'counterfactual': counterfactual.causal_effect,
                'world_delta': counterfactual.world_delta
            })

        # 6. Update cognitive state based on outcomes
        cognitive = self.state.cognitive.with_goal_update(
            goal_added=bool(self.user_goals),
            progress=0.0
        )
        self.state = self.state.with_cognitive_update(cognitive)

        return {
            'cycle': self.cycle_count,
            'uptime': uptime,
            'action': action_type,
            'category': category.value,
            'observation': self.observations_made,
            'intervention': self.interventions_made,
            'belief_conf': self.state.belief.infrastructure_confidence,
            'anomaly_conf': self.state.belief.anomaly_confidence,
            'hypotheses': len(self.state.belief.active_hypotheses),
            'belief_revisions': self.belief_revisions,
            'success': success
        }

    async def run(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        """Run Phase 4 system"""
        print(f"\n{'='*80}")
        print(f"AI_OS PHASE 4 - THREE-LAYER CAUSAL SYSTEM", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Architecture:", flush=True)
        print(f"  Layer 1: Environment State (physical world)", flush=True)
        print(f"  Layer 2: Belief State (agent's beliefs) ← OBSERVE changes this", flush=True)
        print(f"  Layer 3: Cognitive State (internal processing)", flush=True)
        print(f"Key insight: OBSERVE → belief revision, not world change", flush=True)
        print(f"{'='*80}\n", flush=True)

        try:
            while True:
                result = await self.execute_cycle()

                icon = '📊' if result['category'] == 'intervene' else '🧠'
                success = '✓' if result['success'] else '✗'

                print(f"[{result['uptime']:.0f}s] C{result['cycle']:3d} "
                      f"{icon} {result['category']:10s} {result['action']:20s} {success} "
                      f"Obs:{result['observation']} Int:{result['intervention']} "
                      f"Belief:{result['belief_conf']:.1f} Anomaly:{result['anomaly_conf']:.1f} "
                      f"Hypoth:{result['hypotheses']}",
                      flush=True)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print("SYSTEM STOPPED", flush=True)
            print(f"{'='*80}", flush=True)

        print(f"\n{'='*80}")
        print("PHASE 4 SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Observations: {self.observations_made}", flush=True)
        print(f"Interventions: {self.interventions_made}", flush=True)
        print(f"Belief revisions: {self.belief_revisions}", flush=True)
        print(f"Final belief confidence: {self.state.belief.infrastructure_confidence:.2f}", flush=True)
        print(f"Active hypotheses: {list(self.state.belief.active_hypotheses)}", flush=True)
        print(f"{'='*80}\n", flush=True)


async def demo():
    """Demo Phase 4 system"""
    system = Phase4System(name="phase4_demo")

    system.add_user_goal("Check system", "check_system_status")
    system.add_user_goal("Read errors", "read_errors")
    system.add_user_goal("List services", "list_services")

    await system.run(interval=4.0, max_cycles=20)


if __name__ == "__main__":
    asyncio.run(demo())