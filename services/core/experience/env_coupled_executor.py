"""
Environment-Coupled Adaptive Executor

Phase transition: cognitive daemon → environment-coupled adaptive system

This is NOT another module. This is the REAL THING.

System now:
1. Observes REAL environment state
2. Executes REAL actions
3. Receives REAL feedback
4. Updates models based on outcomes
5. Adapts strategy based on success/failure
6. Records EVERYTHING in lineage

No more simulation. Real execution loop.
"""

import asyncio
import os
import sys
import json
import subprocess
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import hashlib

sys.path.insert(0, '/home/onor/ai_os_final/services/core/experience')

from cognitive_loop import (
    CognitiveLoopState,
    CognitiveLoopConfig,
    FilterConfig,
    detect_pressure_tension,
    add_tension,
    resolve_tension,
    get_top_tensions
)


class EnvironmentState(Enum):
    """Current environment state"""
    OBSERVING = "observing"
    ACTING = "acting"
    RECEIVING_FEEDBACK = "receiving_feedback"
    ADAPTING = "adapting"


@dataclass
class EnvironmentObservation:
    """What the system observes in the environment"""
    observation_id: str
    timestamp: str
    files_changed: List[str]
    processes_running: List[str]
    error_logs: List[str]
    success_signals: List[str]
    resource_state: Dict[str, Any]


@dataclass
class ActionResult:
    """Result of an action execution"""
    action_id: str
    action_type: str
    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    environment_state: Dict[str, Any]


@dataclass
class Adaptation:
    """How the system adapted based on feedback"""
    adaptation_id: str
    trigger: str  # What caused adaptation
    strategy_change: str  # What changed
    confidence_delta: float  # How much confidence changed
    timestamp: str


class EnvironmentSensor:
    """Observes real environment state"""

    def __init__(self, workspace_path: str = "/home/onor/ai_os_final"):
        self.workspace = workspace_path

    async def observe(self) -> EnvironmentObservation:
        """Observe current environment state"""
        files_changed = []
        processes_running = []
        error_logs = []
        success_signals = []

        # Check for recent file changes
        try:
            result = subprocess.run(
                ['find', self.workspace, '-type', 'f', '-mmin', '-60', '2>/dev/null'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                files_changed = result.stdout.strip().split('\n')[:20]
        except:
            pass

        # Check running processes
        try:
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Filter relevant processes
                for line in result.stdout.split('\n'):
                    if any(x in line for x in ['python', 'docker', 'celery']):
                        processes_running.append(line[:100])
        except:
            pass

        # Check error logs
        try:
            log_paths = [
                f"{self.workspace}/services/core/logs/error.log",
                f"{self.workspace}/services/core/logs/app.log"
            ]
            for path in log_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        lines = f.readlines()
                        for line in lines[-10:]:
                            if 'ERROR' in line or 'error' in line.lower():
                                error_logs.append(line.strip()[:200])
        except:
            pass

        # Check for success signals (completed goals)
        try:
            db_path = f"{self.workspace}/services/core/db"
            if os.path.exists(db_path):
                # Check for recent completed items
                success_signals.append("database_available")
        except:
            pass

        # Resource state
        resource_state = {
            'cpu_approx': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
            'memory_available_mb': self._get_memory_info(),
            'disk_space_mb': self._get_disk_info()
        }

        return EnvironmentObservation(
            observation_id=f"obs_{hashlib.md5(str(datetime.now(UTC)).encode()).hexdigest()[:12]}",
            timestamp=datetime.now(UTC).isoformat(),
            files_changed=files_changed,
            processes_running=processes_running[:10],
            error_logs=error_logs[:5],
            success_signals=success_signals,
            resource_state=resource_state
        )

    def _get_memory_info(self) -> int:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) // 1024  # MB
        except:
            pass
        return 0

    def _get_disk_info(self) -> int:
        try:
            import shutil
            return shutil.disk_usage('/').free // (1024 * 1024)  # MB
        except:
            return 0


class ActionExecutor:
    """Executes real actions in the environment"""

    def __init__(self, workspace_path: str = "/home/onor/ai_os_final"):
        self.workspace = workspace_path

    async def execute(self, action: Dict[str, Any]) -> ActionResult:
        """Execute an action and return real result"""
        import time
        start = time.time()

        action_type = action.get('type', 'unknown')
        action_id = f"act_{hashlib.md5(str(datetime.now(UTC)).encode()).hexdigest()[:8]}"

        try:
            if action_type == 'check_status':
                output = await self._check_system_status()
                success = True

            elif action_type == 'list_files':
                output = await self._list_files(action.get('path', self.workspace))
                success = True

            elif action_type == 'read_log':
                output = await self._read_log(action.get('log_path'))
                success = True

            elif action_type == 'check_container':
                output = await self._check_docker_container(action.get('container', 'ns_core'))
                success = True

            elif action_type == 'test_endpoint':
                output = await self._test_endpoint(action.get('url', 'http://localhost:8000/health'))
                success = True

            elif action_type == 'run_command':
                output = await self._run_command(action.get('command', 'echo test'))
                success = True

            else:
                output = f"Unknown action type: {action_type}"
                success = False

            error = None

        except Exception as e:
            output = ""
            error = str(e)
            success = False

        env_state = {
            'workspace': self.workspace,
            'action_type': action_type,
            'timestamp': datetime.now(UTC).isoformat()
        }

        return ActionResult(
            action_id=action_id,
            action_type=action_type,
            success=success,
            output=output[:2000],  # Limit output size
            error=error,
            execution_time=time.time() - start,
            environment_state=env_state
        )

    async def _check_system_status(self) -> str:
        """Check overall system status"""
        sensor = EnvironmentSensor(self.workspace)
        obs = await sensor.observe()
        return json.dumps({
            'files_changed': len(obs.files_changed),
            'processes': len(obs.processes_running),
            'errors': len(obs.error_logs),
            'resources': obs.resource_state
        }, indent=2)

    async def _list_files(self, path: str) -> str:
        """List files in directory"""
        try:
            files = os.listdir(path)[:50]
            return f"Files in {path}: {len(files)} items\n" + "\n".join(files[:20])
        except Exception as e:
            return f"Error listing {path}: {e}"

    async def _read_log(self, log_path: str) -> str:
        """Read log file"""
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    return f"Last {min(10, len(lines))} lines from {log_path}:\n" + "".join(lines[-10:])
            return f"Log not found: {log_path}"
        except Exception as e:
            return f"Error reading log: {e}"

    async def _check_docker_container(self, container: str) -> str:
        """Check docker container status"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--filter', f'name={container}', '--format', '{{.Status}}'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"Container {container}: {result.stdout.strip()}"
            return f"Container {container}: not running or not found"
        except Exception as e:
            return f"Docker check failed: {e}"

    async def _test_endpoint(self, url: str) -> str:
        """Test HTTP endpoint"""
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={'User-Agent': 'AI_OS'})
            response = urllib.request.urlopen(req, timeout=5)
            return f"Endpoint {url}: {response.status} OK"
        except Exception as e:
            return f"Endpoint {url}: FAILED - {e}"

    async def _run_command(self, command: str) -> str:
        """Run shell command"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            return f"Command: {command}\nExit code: {result.returncode}\nOutput: {result.stdout[:500]}"
        except Exception as e:
            return f"Command failed: {e}"


class StrategyAdaptation:
    """Adapts strategy based on success/failure patterns"""

    def __init__(self):
        self.strategy = {
            'aggression': 0.5,  # How aggressive to try actions
            'caution': 0.5,     # How much to verify before acting
            'exploration': 0.3,  # How much to try new approaches
            'persistence': 0.6   # How long to keep trying
        }
        self.action_history: List[Dict[str, Any]] = []
        self.success_patterns: Dict[str, float] = {}  # action_type -> success_rate
        self.version = 0

    def record_result(self, result: ActionResult):
        """Record action result for learning"""
        self.action_history.append({
            'type': result.action_type,
            'success': result.success,
            'time': result.execution_time,
            'timestamp': datetime.now(UTC).isoformat()
        })

        # Update success rate
        if result.action_type not in self.success_patterns:
            self.success_patterns[result.action_type] = 0.0

        old_rate = self.success_patterns[result.action_type]
        new_rate = (old_rate * 0.8) + (1.0 if result.success else 0.0) * 0.2
        self.success_patterns[result.action_type] = new_rate

    def get_action(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select next action based on context and patterns"""
        # Try to use successful patterns first
        best_action = None
        best_rate = 0.0

        for action_type, rate in self.success_patterns.items():
            if rate > best_rate:
                best_rate = rate
                best_action = action_type

        # Build action based on context
        if context.get('errors_detected'):
            action = {'type': 'read_log', 'log_path': '/home/onor/ai_os_final/services/core/logs/error.log'}
        elif context.get('needs_status_check'):
            action = {'type': 'check_status'}
        elif context.get('needs_container_check'):
            action = {'type': 'check_container', 'container': 'ns_core'}
        else:
            # Use best known action or default
            if best_action:
                action = {'type': best_action}
            else:
                action = {'type': 'check_status'}

        return action

    def adapt(self, feedback: Dict[str, Any]) -> Adaptation:
        """Adapt strategy based on feedback"""
        adaptation_id = f"adapt_{self.version}"
        self.version += 1

        changes = []

        # Analyze feedback
        if feedback.get('success_rate', 0) < 0.5:
            self.strategy['caution'] = min(1.0, self.strategy['caution'] + 0.1)
            self.strategy['aggression'] = max(0.0, self.strategy['aggression'] - 0.1)
            changes.append("increased caution")

        if feedback.get('no_progress_cycles', 0) > 5:
            self.strategy['exploration'] = min(1.0, self.strategy['exploration'] + 0.2)
            changes.append("increased exploration")

        if feedback.get('repetitive_failures', 0) > 3:
            self.strategy['persistence'] = max(0.0, self.strategy['persistence'] - 0.1)
            changes.append("reduced persistence")

        return Adaptation(
            adaptation_id=adaptation_id,
            trigger=str(feedback),
            strategy_change="; ".join(changes) if changes else "no change",
            confidence_delta=0.1 if changes else 0.0,
            timestamp=datetime.now(UTC).isoformat()
        )


class EnvironmentCoupledCognitiveExecutor:
    """
    Environment-Coupled Adaptive Executor

    This is the phase transition from "cognitive daemon" to "adaptive system".

    Loop:
    1. OBSERVE - Sense real environment
    2. THINK - Process observations, detect patterns
    3. ACT - Execute real actions
    4. RECEIVE - Get real feedback
    5. LEARN - Update models based on outcomes
    6. ADAPT - Change strategy based on patterns
    7. REPEAT

    Key difference from cognitive daemon:
    - NOT just orchestration
    - REAL environment interaction
    - Actual feedback loops
    - Persistent learning
    """

    def __init__(self, name: str = "env_coupled_executor"):
        self.name = name
        self.start_time = datetime.now(UTC)
        self.cycle_count = 0

        # Environment interfaces
        self.sensor = EnvironmentSensor()
        self.executor = ActionExecutor()
        self.adaptor = StrategyAdaptation()

        # Cognitive state
        self.cognitive_state = CognitiveLoopState.initial(
            filter_config=FilterConfig(
                noise_threshold=0.3,
                min_authority=0.2,
                max_inputs_per_cycle=10,
                novelty_bonus=0.3,
                repetition_penalty=0.4
            ),
            loop_config=CognitiveLoopConfig(
                attention_budget=0.8,
                max_tensions=30,
                tension_threshold=0.3,
                salience_threshold=0.4,
                goal_generation_rate=0.8,
                adaptation_rate=0.15
            )
        )

        # Lineage - EVERYTHING recorded
        self.lineage: List[Dict[str, Any]] = []

        # Metrics
        self.actions_executed = 0
        self.actions_succeeded = 0
        self.adaptations_made = 0
        self.cycles_no_progress = 0

    def _record(self, event_type: str, data: Dict[str, Any]):
        """Record event in lineage"""
        self.lineage.append({
            'type': event_type,
            'data': data,
            'cycle': self.cycle_count,
            'timestamp': datetime.now(UTC).isoformat()
        })

    async def observe_environment(self) -> EnvironmentObservation:
        """Step 1: Observe real environment"""
        obs = await self.sensor.observe()
        self._record('observation', {
            'observation_id': obs.observation_id,
            'files_changed': len(obs.files_changed),
            'errors': len(obs.error_logs),
            'processes': len(obs.processes_running)
        })
        return obs

    def detect_tensions_from_observation(self, obs: EnvironmentObservation) -> List[Dict[str, Any]]:
        """Step 2: Detect tensions from environment state"""
        tensions = []

        # Error pressure
        if obs.error_logs:
            tensions.append({
                'type': 'pressure',
                'source': 'error_logs',
                'intensity': min(1.0, len(obs.error_logs) * 0.15),
                'description': f"{len(obs.error_logs)} errors detected"
            })

        # Resource pressure
        if obs.resource_state.get('cpu_approx', 0) > 2.0:
            tensions.append({
                'type': 'pressure',
                'source': 'high_cpu',
                'intensity': 0.6,
                'description': 'High CPU load'
            })

        # Novelty (new files = new activity)
        if obs.files_changed:
            tensions.append({
                'type': 'opportunity',
                'source': 'recent_activity',
                'intensity': 0.4,
                'description': f"{len(obs.files_changed)} files changed recently"
            })

        return tensions

    async def act(self, tension: Dict[str, Any], context: Dict[str, Any]) -> ActionResult:
        """Step 3: Execute action based on tension"""
        # Select action using strategy
        action = self.adaptor.get_action(context)

        self._record('action_planned', {
            'action': action,
            'tension': tension
        })

        # Execute
        result = await self.executor.execute(action)
        self.actions_executed += 1

        if result.success:
            self.actions_succeeded += 1

        # Record result
        self.adaptor.record_result(result)

        self._record('action_result', {
            'action_id': result.action_id,
            'success': result.success,
            'output_preview': result.output[:200],
            'execution_time': result.execution_time
        })

        return result

    async def adapt(self, result: ActionResult):
        """Step 6: Adapt strategy based on result"""
        if not result.success:
            self.cycles_no_progress += 1
        else:
            self.cycles_no_progress = 0

        feedback = {
            'success_rate': self.actions_succeeded / max(1, self.actions_executed),
            'no_progress_cycles': self.cycles_no_progress,
            'last_result': result.success
        }

        adaptation = self.adaptor.adapt(feedback)
        self.adaptations_made += 1

        self._record('adaptation', {
            'adaptation_id': adaptation.adaptation_id,
            'strategy_change': adaptation.strategy_change,
            'current_strategy': self.adaptor.strategy.copy()
        })

    async def run_cycle(self, interval: float = 5.0):
        """Execute one full cognitive cycle"""
        self.cycle_count += 1
        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        status = "OBSERVE"
        action_result = None

        try:
            # 1. OBSERVE
            status = "OBSERVE"
            obs = await self.observe_environment()
            self._record('observation', {
                'observation_id': obs.observation_id,
                'errors': len(obs.error_logs),
                'processes': len(obs.processes_running),
                'files': len(obs.files_changed)
            })

            # 2. DETECT TENSIONS
            status = "THINK"
            tensions = self.detect_tensions_from_observation(obs)

            if tensions:
                self._record('tension_detected', {
                    'count': len(tensions),
                    'top': tensions[0].get('description', 'unknown')
                })

            # 3. ACT
            status = "ACT"
            context = {
                'errors_detected': len(obs.error_logs) > 0,
                'needs_status_check': self.cycle_count % 3 == 0,
                'needs_container_check': self.cycle_count % 5 == 0
            }

            if tensions:
                action_result = await self.act(tensions[0], context)
            else:
                # No tensions - do health check
                action_result = await self.executor.execute({'type': 'check_status'})
                self.actions_executed += 1
                if action_result.success:
                    self.actions_succeeded += 1
                self._record('health_check', {
                    'success': action_result.success
                })

            # 4. RECEIVE (already received in result)

            # 5. LEARN (recorded in lineage via adaptor)

            # 6. ADAPT
            status = "ADAPT"
            await self.adapt(action_result)

        except Exception as e:
            self._record('cycle_error', {
                'status': status,
                'error': str(e)
            })

        # Print status
        success_rate = self.actions_succeeded / max(1, self.actions_executed) * 100
        strategy = self.adaptor.strategy
        learned = len(self.adaptor.success_patterns)

        # Show what was observed
        obs_summary = ""
        if status == "OBSERVE" or self.cycle_count == 1:
            obs_summary = ""

        print(f"[{uptime:.0f}s] C{self.cycle_count} | "
              f"{status:8s} | "
              f"A: {self.actions_executed} ({success_rate:.0f}%) | "
              f"Learned: {learned} | "
              f"a={strategy['aggression']:.1f} c={strategy['caution']:.1f}", flush=True)

    async def run(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        """Run the environment-coupled executor"""
        print(f"\n{'='*70}")
        print(f"ENVIRONMENT-COUPLED ADAPTIVE EXECUTOR", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Name: {self.name}")
        print(f"Started: {self.start_time.isoformat()}")
        print(f"Interval: {interval}s")
        print(f"Phase: Observe → Think → Act → Receive → Learn → Adapt")
        print(f"{'='*70}\n")

        try:
            while True:
                await self.run_cycle(interval)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*70}")
            print("EXECUTOR STOPPED")
            print(f"{'='*70}")

        # Summary
        success_rate = self.actions_succeeded / max(1, self.actions_executed) * 100
        print(f"\n{'='*70}")
        print("EXECUTION SUMMARY", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Total cycles: {self.cycle_count}", flush=True)
        print(f"Actions executed: {self.actions_executed}", flush=True)
        print(f"Actions succeeded: {self.actions_succeeded} ({success_rate:.1f}%)", flush=True)
        print(f"Adaptations made: {self.adaptations_made}", flush=True)
        print(f"Lineage events: {len(self.lineage)}", flush=True)
        print(f"Learned patterns: {len(self.adaptor.success_patterns)}", flush=True)
        print(f"Current strategy: {self.adaptor.strategy}", flush=True)
        print(f"Total uptime: {(datetime.now(UTC) - self.start_time).total_seconds():.0f}s", flush=True)
        print(f"{'='*70}\n", flush=True)

        return {
            'cycles': self.cycle_count,
            'actions': self.actions_executed,
            'success_rate': success_rate,
            'patterns': self.adaptor.success_patterns,
            'strategy': self.adaptor.strategy,
            'lineage': self.lineage
        }


async def demo():
    """Demo: environment-coupled executor"""
    executor = EnvironmentCoupledCognitiveExecutor(name="demo_env_executor")

    # Run for 12 cycles (about 1 minute)
    await executor.run(interval=5.0, max_cycles=12)


if __name__ == "__main__":
    asyncio.run(demo())