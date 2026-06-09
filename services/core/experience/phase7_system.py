"""
AI_OS Phase 7 - Learned Latent Dynamics + Symbolic Reflection

Hybrid architecture combining:
- Subsymbolic: continuous embeddings, learned transitions
- Symbolic: reflective reasoning, interpretable causes

Key innovations:
1. Continuous latent space (not enum causes)
2. Learned transition model P(z_next | z, action)
3. Information gain planning (true active inference)
4. Self-model (own competence and limits)
5. Contrastive temporal learning
"""

import asyncio
import os
import sys
import subprocess
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

sys.path.insert(0, '/home/onor/ai_os_final/services/core/experience')

from phase7_hybrid import (
    Phase7HybridSystem,
    LatentEmbedding,
    SymbolicCause,
    LearnedTransitionModel,
    SelfModel,
    InformationGainPlanner,
    ContrastiveTemporalLearner
)


@dataclass
class UserGoal:
    goal_id: str
    title: str
    action_type: str


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


class Phase7Executor:
    def __init__(self, workspace: str = "/home/onor/ai_os_final"):
        self.workspace = workspace

    async def execute(self, action: str) -> Tuple[bool, str]:
        try:
            if action == 'check_system_status':
                output = await self._check_status()
            elif action == 'read_errors':
                output = await self._read_errors()
            elif action == 'restart_service':
                output = "Service restart simulated"
            elif action == 'diagnose':
                output = "Diagnostic action executed"
            elif action == 'explore':
                output = "Exploration action (information gathering)"
            else:
                output = f"Unknown: {action}"

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


class Phase7HybridCognitiveSystem:
    """
    AI_OS Phase 7 - Hybrid Latent-Symbolic Cognitive System

    Architecture:
    - Subsymbolic Layer: Continuous embeddings, learned transitions
    - Symbolic Layer: Interpretable causes, reflection

    Innovation:
    NOT just symbolic enum → learned latent embeddings
    NOT scripted transitions → learned transition model
    NOT random exploration → information gain planning
    """

    def __init__(self, name: str = "ai_os_phase7"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Phase 7 hybrid system
        self.hybrid = Phase7HybridSystem()

        # Components
        self.sensor = EnvironmentSensor()
        self.executor = Phase7Executor()

        # Goals
        self.user_goals: List[UserGoal] = []

        # Metrics
        self.actions_taken = 0
        self.transitions_learned = 0
        self.exploration_actions = 0
        self.exploitation_actions = 0

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
        self.user_goals.append(UserGoal(
            goal_id=f"goal_{len(self.user_goals)}",
            title=title,
            action_type=action_type
        ))

    async def execute_cycle(self) -> Dict[str, Any]:
        """
        Execute Phase 7 hybrid cognitive cycle.

        Steps:
        1. OBSERVE → create latent embedding
        2. INFER symbolic cause from latent
        3. SELECT ACTION using information gain
        4. EXECUTE
        5. RECORD outcome → learn transition
        """
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        # 1. OBSERVE and create latent embedding
        obs = await self.sensor.observe()
        error_count = len(obs.get('error_logs', []))
        container_count = len(obs.get('container_status', {}))
        resource_load = obs.get('resource_state', {}).get('load', 0)
        memory_mb = obs.get('resource_state', {}).get('memory_mb', 0)

        embedding = self.hybrid.update_from_observation(
            error_count=error_count,
            container_count=container_count,
            resource_load=resource_load,
            memory_mb=memory_mb
        )

        # 2. Get symbolic interpretation
        semantic = self.hybrid.get_semantic_description()

        # 3. SELECT ACTION using information gain planning
        available_actions = ['check_system_status', 'read_errors', 'restart_service', 'diagnose', 'explore']

        if self.user_goals:
            action = self.user_goals.pop(0).action_type
            is_exploration = False
        else:
            action, pragmatic, epistemic = self.hybrid.select_action(
                available_actions,
                situation='diagnostic'
            )
            # Track exploration vs exploitation
            if epistemic > 0.1:
                self.exploration_actions += 1
                is_exploration = True
            else:
                self.exploitation_actions += 1
                is_exploration = False

        # 4. EXECUTE
        success, output = await self.executor.execute(action)
        self.actions_taken += 1

        # 5. OBSERVE result and record transition
        obs_after = await self.sensor.observe()
        error_after = len(obs_after.get('error_logs', []))
        container_after = len(obs_after.get('container_status', {}))
        load_after = obs_after.get('resource_state', {}).get('load', 0)
        mem_after = obs_after.get('resource_state', {}).get('memory_mb', 0)

        new_embedding = self.hybrid.update_from_observation(
            error_count=error_after,
            container_count=container_after,
            resource_load=load_after,
            memory_mb=mem_after
        )

        # Learn transition
        self.hybrid.record_outcome(action, success, new_embedding)
        self.transitions_learned += 1

        # Self-awareness
        self_awareness = self.hybrid.get_self_awareness()

        # Record
        self._record('hybrid_cycle', {
            'latent_norm': self.hybrid._embedding_norm(),
            'symbolic_cause': self.hybrid.symbolic_belief.value,
            'embedding_confidence': embedding.confidence,
            'embedding_novelty': embedding.novelty,
            'action': action,
            'success': success,
            'planning_uncertainty': self_awareness['planning_uncertainty'],
            'exploration_urgency': self_awareness['exploration_urgency']
        })

        # Icon
        if self.hybrid.symbolic_belief == SymbolicCause.STABLE:
            icon = "✓"
        elif self.hybrid.symbolic_belief == SymbolicCause.SERVICE_INSTABILITY:
            icon = "⚠"
        else:
            icon = "?"

        return {
            'cycle': self.cycle_count,
            'uptime': uptime,
            'latent_norm': self.hybrid._embedding_norm(),
            'symbolic_cause': self.hybrid.symbolic_belief.value,
            'embedding_conf': embedding.confidence,
            'embedding_novel': embedding.novelty,
            'action': action,
            'success': success,
            'semantic': semantic[:35],
            'planning_unc': self_awareness['planning_uncertainty'],
            'expl_urgency': self_awareness['exploration_urgency'],
            'transitions': self.transitions_learned,
            'expl': self.exploration_actions,
            'exp': self.exploitation_actions
        }

    async def run(self, interval: float = 4.0, max_cycles: Optional[int] = None):
        """Run Phase 7 hybrid system"""
        print(f"\n{'='*80}")
        print(f"AI_OS PHASE 7 - LEARNED LATENT DYNAMICS + SYMBOLIC REFLECTION", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Hybrid Architecture:", flush=True)
        print(f"  Subsymbolic: continuous embeddings, learned transitions", flush=True)
        print(f"  Symbolic: interpretable causes, reflection", flush=True)
        print(f"Key Innovations:", flush=True)
        print(f"  1. Continuous latent space (not fixed enum)", flush=True)
        print(f"  2. Learned transition model P(z_next | z, action)", flush=True)
        print(f"  3. Information gain planning (true active inference)", flush=True)
        print(f"  4. Self-model (own competence)", flush=True)
        print(f"{'='*80}\n", flush=True)

        try:
            while True:
                result = await self.execute_cycle()

                success = '✓' if result['success'] else '✗'
                cause = result['symbolic_cause'].replace('_', ' ')[:18]
                norm = result['latent_norm']

                print(f"[{result['uptime']:.0f}s] C{result['cycle']:3d} {result['action']:12s} {success} "
                      f"Z:{norm:.2f} Cause:{cause:18s} "
                      f"Conf:{result['embedding_conf']:.1f} "
                      f"Expl:{result['expl']} Exp:{result['exp']}",
                      flush=True)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print("SYSTEM STOPPED", flush=True)
            print(f"{'='*80}", flush=True)

        # Summary
        total_learned = self.hybrid.transition_model.transition_params.__len__() if hasattr(self.hybrid.transition_model.transition_params, '__len__') else 0

        print(f"\n{'='*80}")
        print("PHASE 7 SUMMARY - HYBRID LATENT-SYMBOLIC", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Actions taken: {self.actions_taken}", flush=True)
        print(f"Transitions learned: {self.transitions_learned}", flush=True)
        print(f"Exploration actions: {self.exploration_actions}", flush=True)
        print(f"Exploitation actions: {self.exploitation_actions}", flush=True)
        print(f"Embedding dimensions: {len(self.hybrid.current_embedding.vector) if self.hybrid.current_embedding else 0}", flush=True)
        print(f"Self-model competence: {self.hybrid.self_model.competence}", flush=True)
        print(f"{'='*80}\n", flush=True)


async def demo():
    """Demo Phase 7 hybrid system"""
    system = Phase7HybridCognitiveSystem(name="phase7_demo")

    system.add_user_goal("Check system", "check_system_status")
    system.add_user_goal("Read errors", "read_errors")

    await system.run(interval=4.0, max_cycles=25)


if __name__ == "__main__":
    asyncio.run(demo())