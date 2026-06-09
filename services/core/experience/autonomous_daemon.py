"""
Minimal Autonomous Cognitive Loop - The Living Core

This is NOT another module. This is the FIRST LIVING CYCLE.

Pipeline:
1. observe world
2. generate tensions
3. prioritize goals
4. execute
5. verify
6. mutate identity/genome
7. reflect
8. repeat

No more abstraction layers. No more "architecture discussions".
Just a loop that runs and demonstrates autonomy.
"""

import asyncio
import sys
import time
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import uuid

# Core imports - these MUST exist
sys.path.insert(0, '/home/onor/ai_os_final/services/core/experience')

from cognitive_loop import (
    CognitiveLoopState,
    CognitiveLoopConfig,
    execute_full_cycle,
    get_top_tensions,
    get_loop_metrics,
    InputSource,
    FilterConfig,
    detect_pressure_tension,
    add_tension,
    resolve_tension,
    TensionState,
    TrendDirection
)

from domain.events import DomainEvent, create_event
from domain.reducers import reduce_event, reduce_events
from cognition.memory import (
    MemoryState,
    MemoryType,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    ReflectiveMemory
)
from cognition.world import WorldModel, Entity, Relation
from cognition.contradiction import ContradictionEngine, detect_contradictions
from cognition.self_model import SelfModel, IdentityAxes
from cognition.goals import GoalEconomy, Goal, GoalStatus
from kernel.policies import IdentityDrivenPolicy, IdentityParameters
from kernel.adapters import ExecutionAdapter


@dataclass
class AutonomousCycleResult:
    """Result of one autonomous cycle"""
    cycle_id: str
    timestamp: str
    tensions_detected: int
    goals_generated: int
    actions_taken: List[str]
    identity_delta: float
    genome_mutations: int
    execution_success: bool


class MinimalAutonomousDaemon:
    """
    The living cognitive loop.

    This is the first autonomous system that:
    - Generates its own goals from tensions
    - Executes them in real world
    - Evolves its identity based on outcomes
    - Maintains persistent state over time

    NOT a module. NOT a function. A DAEMON.
    """

    def __init__(self, name: str = "autonomous_core"):
        self.name = name
        self.cycle_count = 0
        self.start_time = datetime.now(UTC)

        # Initialize all cognitive components
        self.cognitive_state = CognitiveLoopState.initial(
            filter_config=FilterConfig(
                noise_threshold=0.3,
                min_authority=0.2,
                max_inputs_per_cycle=15,
                novelty_bonus=0.25,
                repetition_penalty=0.3
            ),
            loop_config=CognitiveLoopConfig(
                attention_budget=0.9,
                max_tensions=40,
                tension_threshold=0.25,
                salience_threshold=0.45,
                goal_generation_rate=1.2,
                adaptation_rate=0.15
            )
        )

        # Memory systems
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.reflective = ReflectiveMemory()

        # World model
        self.world_model = WorldModel()

        # Self model (identity)
        self.identity = SelfModel(
            identity_id=str(uuid.uuid4()),
            autonomy=0.7,
            curiosity=0.6,
            stability=0.5,
            coherence=0.8,
            created_at=datetime.now(UTC).isoformat()
        )

        # Goal economy
        self.goal_economy = GoalEconomy()

        # Contradiction engine
        self.contradiction_engine = ContradictionEngine()

        # Execution adapter
        self.execution_adapter = ExecutionAdapter()

        # Lineage (event log)
        self.lineage: List[DomainEvent] = []

        # Metrics
        self.total_tensions_resolved = 0
        self.total_goals_completed = 0
        self.total_genome_mutations = 0

    async def observe_world(self) -> List[InputSource]:
        """
        Observe the world and generate inputs.

        For now: simulated observations.
        Later: real sensor data, API calls, environment changes.
        """
        # Simulate world observations
        observations = []

        # Time-based pressures
        hour = datetime.now(UTC).hour
        if hour in [9, 10, 14, 15]:  # Work hours
            observations.append(InputSource(
                source_id=f"work_pressure_{hour}",
                source_type="temporal",
                content=f"Time pressure at {hour}:00",
                raw_data=frozenset(),
                timestamp=datetime.now(UTC).isoformat(),
                authority_score=0.8,
                novelty_score=0.3,
                relevance_score=0.6
            ))

        # Cycle-based novelty (every 10 cycles = new situation)
        if self.cycle_count % 10 == 0:
            observations.append(InputSource(
                source_id=f"novel_event_{self.cycle_count}",
                source_type="environmental",
                content=f"New situation detected",
                raw_data=frozenset(),
                timestamp=datetime.now(UTC).isoformat(),
                authority_score=0.7,
                novelty_score=0.9,
                relevance_score=0.5
            ))

        # Goal completion signals
        if self.total_goals_completed > 0 and self.total_goals_completed % 3 == 0:
            observations.append(InputSource(
                source_id=f"goal_completed_{self.total_goals_completed}",
                source_type="internal",
                content=f"Goal system progress",
                raw_data=frozenset(),
                timestamp=datetime.now(UTC).isoformat(),
                authority_score=0.9,
                novelty_score=0.2,
                relevance_score=0.8
            ))

        return observations

    async def generate_tensions(self, inputs: List[InputSource]) -> CognitiveLoopState:
        """
        Generate tensions from observations.

        Tensions emerge from:
        - High relevance + novelty = pressure
        - Unresolved contradictions = conflict
        - Missing knowledge = vacuum
        """
        state = self.cognitive_state

        for inp in inputs:
            if inp.relevance_score > 0.5 and inp.novelty_score > 0.4:
                # Create tension
                tension = detect_pressure_tension(
                    source=inp.source_id,
                    intensity=inp.relevance_score * inp.novelty_score,
                    pressure_type="cognitive"
                )

                # Add to state
                state = CognitiveLoopState(
                    tension_state=add_tension(state.tension_state, tension),
                    trajectory_state=state.trajectory_state,
                    emergence_state=state.emergence_state,
                    filter_config=state.filter_config,
                    loop_config=state.loop_config,
                    current_phase=state.current_phase,
                    cycle_count=state.cycle_count,
                    last_cycle_time=state.last_cycle_time,
                    adaptation_metrics=state.adaptation_metrics
                )

        return state

    async def prioritize_goals(self, state: CognitiveLoopState) -> List[Dict[str, Any]]:
        """
        Prioritize goals from tensions.

        Returns list of {goal_id, tension_id, priority, action}
        """
        top_tensions = get_top_tensions(state, n=3)

        goals = []
        for tension in top_tensions:
            urgency = tension.urgency()
            if urgency > 0.3:
                goals.append({
                    'goal_id': f"autonomous_goal_{tension.tension_id}",
                    'tension_id': tension.tension_id,
                    'priority': urgency,
                    'action': self._select_action_for_tension(tension.type),
                    'expected_impact': tension.intensity
                })

        return sorted(goals, key=lambda x: x['priority'], reverse=True)

    def _select_action_for_tension(self, tension_type: str) -> str:
        """Select action based on tension type"""
        action_map = {
            'pressure': 'process_and_integrate',
            'contradiction': 'analyze_and_resolve',
            'vacuum': 'explore_and_learn',
            'instability': 'stabilize_and_cohere',
            'divergence': 'realign_and_correct',
            'opportunity': 'exploit_and_expand',
            'attention': 'focus_and_execute'
        }
        return action_map.get(tension_type, 'general_processing')

    async def execute(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a goal.

        For now: simulated execution with realistic outcomes.
        Later: real execution via adapters.
        """
        action = goal['action']
        priority = goal['priority']

        # Simulate execution with varying success
        success_probability = 0.6 + (priority * 0.3)  # Higher priority = higher success
        success = (hash(goal['goal_id']) % 100) < (success_probability * 100)

        result = {
            'goal_id': goal['goal_id'],
            'action': action,
            'success': success,
            'tension_id': goal['tension_id'],
            'quality': 0.5 + (priority * 0.5) if success else 0.3,
            'timestamp': datetime.now(UTC).isoformat()
        }

        # Record in episodic memory
        await self.episodic.add_episode(
            event_type='goal_execution',
            content=result,
            emotional_valence=0.7 if success else 0.3,
            importance=priority
        )

        return result

    async def verify(self, result: Dict[str, Any]) -> bool:
        """Verify execution success"""
        return result.get('success', False)

    async def mutate_identity(self, result: Dict[str, Any]) -> float:
        """
        Mutate identity based on execution outcome.

        Returns identity_delta (change magnitude)
        """
        if not result.get('success'):
            # Failed execution -> increase caution
            self.identity = SelfModel(
                identity_id=self.identity.identity_id,
                autonomy=self.identity.autonomy * 0.95,  # Decrease autonomy
                curiosity=self.identity.curiosity * 0.98,  # Slight decrease
                stability=self.identity.stability * 1.02,  # Increase stability
                coherence=self.identity.coherence * 0.97,
                created_at=self.identity.created_at,
                version=self.identity.version + 1
            )
            return 0.05

        # Success -> slight evolution
        self.identity = SelfModel(
            identity_id=self.identity.identity_id,
            autonomy=min(1.0, self.identity.autonomy * 1.01),
            curiosity=min(1.0, self.identity.curiosity * 1.02),  # Increase curiosity
            stability=self.identity.stability * 0.99,  # Slight decrease (more risk-taking)
            coherence=min(1.0, self.identity.coherence * 1.01),
            created_at=self.identity.created_at,
            version=self.identity.version + 1
        )

        return 0.02

    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reflect on execution and extract patterns.
        """
        # Extract pattern from execution
        if result.get('success'):
            pattern = {
                'type': 'successful_execution',
                'action': result['action'],
                'quality': result['quality'],
                'context': {
                    'priority': result.get('priority', 0.5),
                    'cycle': self.cycle_count
                }
            }
            await self.semantic.add_knowledge(
                concept=result['action'],
                attributes=pattern
            )

        # Record reflection
        reflection = {
            'result': result,
            'identity_state': {
                'autonomy': self.identity.autonomy,
                'curiosity': self.identity.curiosity,
                'stability': self.identity.stability
            },
            'timestamp': datetime.now(UTC).isoformat()
        }

        await self.reflective.add_reflection(
            content=reflection,
            depth=0.5 if result.get('success') else 0.8
        )

        return reflection

    def record_lineage(self, event_type: str, data: Dict[str, Any]):
        """Record event in lineage for replay"""
        event = create_event(
            event_type=event_type,
            data=data,
            metadata={
                'cycle': self.cycle_count,
                'identity_version': self.identity.version,
                'timestamp': datetime.now(UTC).isoformat()
            }
        )
        self.lineage.append(event)

    async def run_cycle(self) -> AutonomousCycleResult:
        """Execute one full autonomous cycle"""
        cycle_id = f"cycle_{self.cycle_count}_{uuid.uuid4().hex[:8]}"

        # 1. Observe
        observations = await self.observe_world()
        self.record_lineage('world_observation', {
            'count': len(observations),
            'sources': [o.source_id for o in observations]
        })

        # 2. Generate tensions
        self.cognitive_state = await self.generate_tensions(observations)
        self.record_lineage('tensions_generated', {
            'count': len(self.cognitive_state.tension_state.tensions)
        })

        # 3. Prioritize goals
        goals = await self.prioritize_goals(self.cognitive_state)

        # 4. Execute top goal
        actions_taken = []
        execution_success = False

        if goals:
            top_goal = goals[0]
            result = await self.execute(top_goal)
            actions_taken.append(f"{top_goal['action']}: {'success' if result['success'] else 'failed'}")

            # 5. Verify
            verified = await self.verify(result)
            execution_success = verified

            # 6. Resolve tension if successful
            if verified:
                self.cognitive_state = CognitiveLoopState(
                    tension_state=resolve_tension(
                        self.cognitive_state.tension_state,
                        top_goal['tension_id'],
                        'autonomous_resolution'
                    ),
                    trajectory_state=self.cognitive_state.trajectory_state,
                    emergence_state=self.cognitive_state.emergence_state,
                    filter_config=self.cognitive_state.filter_config,
                    loop_config=self.cognitive_state.loop_config,
                    current_phase=self.cognitive_state.current_phase,
                    cycle_count=self.cognitive_state.cycle_count + 1,
                    last_cycle_time=datetime.now(UTC).isoformat(),
                    adaptation_metrics=self.cognitive_state.adaptation_metrics
                )
                self.total_tensions_resolved += 1
                self.total_goals_completed += 1

            # 7. Mutate identity
            identity_delta = await self.mutate_identity(result)

            # 8. Reflect
            await self.reflect(result)

            self.record_lineage('goal_executed', {
                'goal_id': top_goal['goal_id'],
                'success': result['success'],
                'identity_delta': identity_delta
            })
        else:
            identity_delta = 0.0

        # Update cycle count
        self.cycle_count += 1

        return AutonomousCycleResult(
            cycle_id=cycle_id,
            timestamp=datetime.now(UTC).isoformat(),
            tensions_detected=len(observations),
            goals_generated=len(goals),
            actions_taken=actions_taken,
            identity_delta=identity_delta,
            genome_mutations=1 if identity_delta > 0 else 0,
            execution_success=execution_success
        )

    async def run_continuous(self, interval: float = 5.0, max_cycles: Optional[int] = None):
        """
        Run autonomous loop continuously.

        Args:
            interval: Seconds between cycles
            max_cycles: Optional max cycles (None = infinite)
        """
        print(f"\n{'='*60}")
        print(f"MINIMAL AUTONOMOUS DAEMON STARTED")
        print(f"{'='*60}")
        print(f"Name: {self.name}")
        print(f"Start time: {self.start_time.isoformat()}")
        print(f"Cycle interval: {interval}s")
        print(f"{'='*60}\n")

        results = []

        try:
            while True:
                # Run one cycle
                result = await self.run_cycle()
                results.append(result)

                # Print status
                uptime = (datetime.now(UTC) - self.start_time).total_seconds()
                print(f"[{uptime:.0f}s] Cycle {self.cycle_count} | "
                      f"Tensions: {result.tensions_detected} | "
                      f"Goals: {result.goals_generated} | "
                      f"Action: {result.actions_taken[0] if result.actions_taken else 'idle'} | "
                      f"Identity Δ: {result.identity_delta:.3f}")

                # Check max cycles
                if max_cycles and self.cycle_count >= max_cycles:
                    break

                # Sleep before next cycle
                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print("AUTONOMOUS DAEMON STOPPED (Keyboard Interrupt)")
            print(f"{'='*60}")

        # Summary
        print(f"\n{'='*60}")
        print("RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Total cycles: {self.cycle_count}")
        print(f"Tensions resolved: {self.total_tensions_resolved}")
        print(f"Goals completed: {self.total_goals_completed}")
        print(f"Identity versions: {self.identity.version}")
        print(f"Lineage events: {len(self.lineage)}")
        print(f"Episodic episodes: {len(self.episodic.episodes)}")
        print(f"Semantic concepts: {len(self.semantic.concepts)}")
        print(f"Final identity: autonomy={self.identity.autonomy:.3f}, "
              f"curiosity={self.identity.curiosity:.3f}, "
              f"stability={self.identity.stability:.3f}")
        print(f"{'='*60}\n")

        return results


async def main():
    """Main entry point"""
    daemon = MinimalAutonomousDaemon(name="ai_os_autonomous_core")

    # Run for 20 cycles (about 100 seconds) for demo
    # In production: run indefinitely
    await daemon.run_continuous(interval=5.0, max_cycles=20)


if __name__ == "__main__":
    asyncio.run(main())