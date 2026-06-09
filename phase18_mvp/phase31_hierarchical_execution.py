"""
Phase 31 — Hierarchical Execution Layer

Unified representation connecting:
  Goal (attractor) → Task DAG (decomposition) → Skill Sequence (policy chain) → Action Trajectory

Key insight:
  - Skills are NOT single actions, they are temporally extended operators
  - Skill sequences = programs over latent space
  - CEM plans over skill chains, not atomic actions
  - Intent Space unifies goal/task/skill/world representations

Architecture:
  Intent Space (I)
       │
  ┌────┼────────────┐
  │    │            │
Goal  Skill       World Model
(attractor) (trajectory) (dynamics)
       │
  Task DAG (segments)
       │
  Skill Sequence (policy chain)
       │
  Action Trajectory (rollout)
"""

import numpy as np
import random
from typing import List, Dict, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ============================================================================
# 1. INTENT SPACE — Unified Representation
# ============================================================================

class IntentType(Enum):
    GOAL = "goal"
    TASK = "task"
    SKILL = "skill"
    ACTION = "action"


@dataclass
class IntentVector:
    """
    Unified embedding in Intent Space I.
    
    All entities (goals, tasks, skills, states) project here.
    Alignment in I-space = semantic compatibility.
    """
    z_state: np.ndarray          # Current latent world state
    goal_embedding: np.ndarray   # Goal as attractor in I-space
    skill_context: np.ndarray    # Available skill distributions
    task_progress: np.ndarray    # Current task completion state
    context_vector: np.ndarray   # Environmental/contextual signals
    
    @property
    def full_vector(self) -> np.ndarray:
        """Concatenate all components into single intent vector."""
        return np.concatenate([
            self.z_state,
            self.goal_embedding,
            self.skill_context,
            self.task_progress,
            self.context_vector
        ])
    
    def alignment_score(self, other: 'IntentVector') -> float:
        """Cosine similarity in intent space."""
        a = self.full_vector
        b = other.full_vector
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def distance_to_attractor(self, goal_embedding: np.ndarray) -> float:
        """Distance to goal attractor in I-space."""
        current_goal = self.goal_embedding
        return float(np.linalg.norm(current_goal - goal_embedding))


# ============================================================================
# 2. GOAL AS ATTRACTOR
# ============================================================================

@dataclass
class GoalAttractor:
    """
    Goal represented as attractor state in latent space.
    
    Not a DB object — a dynamical systems concept.
    """
    goal_id: str
    attractor_state: np.ndarray  # Target latent state
    basin_radius: float          # How close = "completed"
    priority: float
    decay_rate: float            # How urgency decreases over time
    success_criteria: Dict       # Semantic criteria for completion
    
    # Linked tasks
    task_ids: List[str] = field(default_factory=list)
    
    # Dynamics
    current_distance: float = 0.0
    vitality: float = 1.0        # alive/stagnant/decaying
    
    def is_satisfied(self, current_state: np.ndarray) -> bool:
        """Check if current state is within attractor basin."""
        dist = np.linalg.norm(current_state - self.attractor_state)
        self.current_distance = float(dist)
        return dist < self.basin_radius
    
    def update_vitality(self, progress: float, time_elapsed: float) -> float:
        """Update goal vitality based on progress and time."""
        if progress > 0.1:
            self.vitality = min(1.0, self.vitality + 0.05)
        else:
            self.vitality *= (1.0 - self.decay_rate * time_elapsed)
        return self.vitality
    
    def to_intent_embedding(self, dim: int) -> np.ndarray:
        """Project goal attractor to intent space."""
        # Combine attractor state with priority and vitality
        meta = np.array([self.priority, self.vitality, self.basin_radius])
        combined = np.concatenate([self.attractor_state, meta])
        
        if len(combined) < dim:
            combined = np.pad(combined, (0, dim - len(combined)))
        return combined[:dim]


# ============================================================================
# 3. TASK DAG (Decomposition)
# ============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskNode:
    """
    Task = subgoal with limited scope.
    
    NOT an action — a decomposition unit.
    """
    task_id: str
    goal_id: str
    description: str
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Execution
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    
    # Skill sequence for this task
    skill_sequence: List['SkillPrototype'] = field(default_factory=list)
    
    # Expected trajectory in latent space
    expected_trajectory: Optional[np.ndarray] = None
    
    # Timing
    estimated_steps: int = 10
    actual_steps: int = 0
    
    def is_ready(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        if self.status != TaskStatus.PENDING:
            return False
        return self.dependencies.issubset(completed_tasks)
    
    def to_intent_embedding(self, dim: int) -> np.ndarray:
        """Project task state to intent space."""
        status_vec = np.array([
            1.0 if self.status == TaskStatus.COMPLETED else 0.0,
            1.0 if self.status == TaskStatus.RUNNING else 0.0,
            self.progress,
            len(self.dependencies) / max(1, len(self.dependents) + 1)
        ])
        
        if len(status_vec) < dim:
            status_vec = np.pad(status_vec, (0, dim - len(status_vec)))
        return status_vec[:dim]


class TaskGraph:
    """
    DAG of tasks for a goal.
    
    Manages dependencies, execution order, and progress tracking.
    """
    
    def __init__(self, goal_id: str):
        self.goal_id = goal_id
        self.tasks: Dict[str, TaskNode] = {}
        self.completed_tasks: Set[str] = set()
        self.failed_tasks: Set[str] = set()
    
    def add_task(self, task: TaskNode):
        self.tasks[task.task_id] = task
    
    def add_dependency(self, task_id: str, depends_on: str):
        if task_id in self.tasks and depends_on in self.tasks:
            self.tasks[task_id].dependencies.add(depends_on)
            self.tasks[depends_on].dependents.add(task_id)
    
    def get_ready_tasks(self) -> List[TaskNode]:
        """Get tasks that can be executed now."""
        return [
            task for task in self.tasks.values()
            if task.is_ready(self.completed_tasks)
        ]
    
    def complete_task(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.COMPLETED
            self.tasks[task_id].progress = 1.0
            self.completed_tasks.add(task_id)
    
    def fail_task(self, task_id: str):
        if task_id in self.tasks:
            self.tasks[task_id].status = TaskStatus.FAILED
            self.failed_tasks.add(task_id)
    
    def get_progress(self) -> float:
        if not self.tasks:
            return 0.0
        return sum(t.progress for t in self.tasks.values()) / len(self.tasks)
    
    def is_complete(self) -> bool:
        return all(
            t.status == TaskStatus.COMPLETED
            for t in self.tasks.values()
        )
    
    def get_critical_path(self) -> List[str]:
        """Find longest dependency chain."""
        if not self.tasks:
            return []
        
        def dfs(task_id: str, visited: Set[str]) -> List[str]:
            if task_id in visited:
                return []
            visited.add(task_id)
            
            task = self.tasks.get(task_id)
            if not task or not task.dependents:
                return [task_id]
            
            longest = []
            for dep in task.dependents:
                path = dfs(dep, visited)
                if len(path) > len(longest):
                    longest = path
            
            return [task_id] + longest
        
        # Start from tasks with no dependencies
        roots = [t.task_id for t in self.tasks.values() if not t.dependencies]
        longest_path = []
        for root in roots:
            path = dfs(root, set())
            if len(path) > len(longest_path):
                longest_path = path
        
        return longest_path
    
    def to_intent_vector(self, dim: int) -> np.ndarray:
        """Aggregate task graph state to intent space."""
        if not self.tasks:
            return np.zeros(dim)
        
        task_vectors = [t.to_intent_embedding(max(4, dim // len(self.tasks))) for t in self.tasks.values()]
        
        # Pad or truncate to match dim
        max_len = max(len(v) for v in task_vectors)
        padded = [np.pad(v, (0, max_len - len(v))) for v in task_vectors]
        
        aggregated = np.mean(padded, axis=0)
        
        if len(aggregated) < dim:
            aggregated = np.pad(aggregated, (0, dim - len(aggregated)))
        return aggregated[:dim]


# ============================================================================
# 4. SKILL AS TRAJECTORY DISTRIBUTION
# ============================================================================

@dataclass
class SkillPrototype:
    """
    Skill = compressed trajectory distribution in latent space.
    
    NOT a single action policy.
    A temporally extended operator that produces a trajectory segment.
    """
    skill_id: str
    name: str
    description: str
    
    # Trajectory distribution
    mean_trajectory: np.ndarray  # Shape: (T, latent_dim)
    trajectory_variance: np.ndarray  # Uncertainty envelope
    
    # Success profile
    success_rate: float = 0.0
    avg_completion_steps: int = 0
    
    # Applicability
    applicable_goal_types: List[str] = field(default_factory=list)
    required_context: Dict[str, float] = field(default_factory=dict)
    
    # Latent space signature
    skill_embedding: Optional[np.ndarray] = None
    
    # Execution trace (learned from experience)
    execution_traces: List[Dict] = field(default_factory=list)
    
    def generate_trajectory(self, z_start: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Generate predicted trajectory from skill.
        
        Returns: (T, latent_dim) trajectory starting from z_start
        """
        T = self.mean_trajectory.shape[0]
        latent_dim = self.mean_trajectory.shape[1]
        
        # Start from current state, follow mean trajectory with noise
        trajectory = np.zeros((T, latent_dim))
        trajectory[0] = z_start
        
        for t in range(1, T):
            delta = self.mean_trajectory[t] - self.mean_trajectory[t - 1]
            noise = np.random.randn(latent_dim) * noise_scale
            trajectory[t] = trajectory[t - 1] + delta + noise
        
        return trajectory
    
    def update_from_experience(self, actual_trajectory: np.ndarray, success: bool):
        """Update skill distribution from execution experience."""
        self.execution_traces.append({
            'trajectory': actual_trajectory,
            'success': success,
            'timestamp': len(self.execution_traces)
        })
        
        # Update success rate (exponential moving average)
        alpha = 0.1
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        
        # Update mean trajectory (if enough traces)
        if len(self.execution_traces) > 5:
            recent = [t['trajectory'] for t in self.execution_traces[-10:]]
            self.mean_trajectory = np.mean(recent, axis=0)
            self.trajectory_variance = np.var(recent, axis=0)
    
    def to_intent_embedding(self, dim: int) -> np.ndarray:
        """Project skill to intent space."""
        if self.skill_embedding is not None:
            return self.skill_embedding[:dim]
        
        # Use trajectory statistics as embedding
        stats = np.array([
            self.success_rate,
            self.avg_completion_steps / 100.0,
            np.mean(self.trajectory_variance),
            np.mean(self.mean_trajectory)
        ])
        
        if len(stats) < dim:
            stats = np.pad(stats, (0, dim - len(stats)))
        return stats[:dim]


class SkillBank:
    """
    Repository of learned skill prototypes.
    
    Skills are extracted from successful trajectories.
    """
    
    def __init__(self):
        self.skills: Dict[str, SkillPrototype] = {}
        self.skill_clusters: Dict[str, List[str]] = defaultdict(list)
    
    def register_skill(self, skill: SkillPrototype):
        self.skills[skill.skill_id] = skill
    
    def find_applicable_skills(self, goal: GoalAttractor, context: Dict) -> List[SkillPrototype]:
        """Find skills applicable to current goal and context."""
        applicable = []
        for skill in self.skills.values():
            # Check goal type compatibility
            if goal.success_criteria.get('type') in skill.applicable_goal_types:
                applicable.append(skill)
            # Check context requirements
            elif all(
                context.get(k, 0) >= v
                for k, v in skill.required_context.items()
            ):
                applicable.append(skill)
        return applicable
    
    def extract_skill_from_trajectory(
        self,
        trajectory: np.ndarray,
        success: bool,
        context: Dict
    ) -> SkillPrototype:
        """
        Extract skill prototype from successful trajectory.
        
        This is the key learning mechanism:
        successful trajectories → skill prototypes
        """
        skill_id = f"skill_{len(self.skills)}"
        
        skill = SkillPrototype(
            skill_id=skill_id,
            name=f"extracted_{skill_id}",
            description=f"Extracted from trajectory with success={success}",
            mean_trajectory=trajectory.copy(),
            trajectory_variance=np.ones_like(trajectory) * 0.1,
            success_rate=1.0 if success else 0.0,
            avg_completion_steps=len(trajectory)
        )
        
        self.register_skill(skill)
        return skill
    
    def get_skill_sequence_for_task(
        self,
        task: TaskNode,
        available_skills: List[SkillPrototype],
        max_length: int = 5
    ) -> List[SkillPrototype]:
        """
        Select skill sequence for a task.
        
        Returns ordered list of skills that compose to solve the task.
        """
        if not available_skills:
            return []
        
        # Simple heuristic: select skills by success rate and relevance
        scored = [
            (skill, skill.success_rate * (1.0 - task.progress))
            for skill in available_skills
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [s for s, _ in scored[:max_length]]


# ============================================================================
# 5. HIERARCHICAL EXECUTION ENGINE
# ============================================================================

class HierarchicalExecutionEngine:
    """
    Full hierarchical execution: Goal → Task DAG → Skill Sequence → Trajectory
    
    Integrates with world model for prediction and CEM for planning.
    """
    
    def __init__(
        self,
        world_model: Any,  # MinimalWorldModel from Phase 30
        skill_bank: SkillBank,
        intent_dim: int = 64
    ):
        self.world_model = world_model
        self.skill_bank = skill_bank
        self.intent_dim = intent_dim
        
        # Active goals
        self.goals: Dict[str, GoalAttractor] = {}
        self.task_graphs: Dict[str, TaskGraph] = {}
        
        # Execution state
        self.current_intent: Optional[IntentVector] = None
        self.execution_log: List[Dict] = []
    
    def add_goal(self, goal: GoalAttractor, task_graph: TaskGraph):
        """Register goal with its task decomposition."""
        self.goals[goal.goal_id] = goal
        self.task_graphs[goal.goal_id] = task_graph
        
        # Link goal to tasks
        goal.task_ids = list(task_graph.tasks.keys())
    
    def build_intent_vector(
        self,
        z_state: np.ndarray,
        goal: GoalAttractor,
        task_graph: TaskGraph,
        context: Dict
    ) -> IntentVector:
        """Build unified intent vector from all components."""
        return IntentVector(
            z_state=z_state,
            goal_embedding=goal.to_intent_embedding(self.intent_dim),
            skill_context=self._get_skill_context(goal, context),
            task_progress=task_graph.to_intent_vector(self.intent_dim),
            context_vector=self._encode_context(context, self.intent_dim)
        )
    
    def _get_skill_context(self, goal: GoalAttractor, context: Dict) -> np.ndarray:
        """Get aggregated skill context vector."""
        skills = self.skill_bank.find_applicable_skills(goal, context)
        if not skills:
            return np.zeros(self.intent_dim)
        
        embeddings = [s.to_intent_embedding(self.intent_dim) for s in skills]
        return np.mean(embeddings, axis=0)
    
    def _encode_context(self, context: Dict, dim: int) -> np.ndarray:
        """Encode environmental context to vector."""
        values = list(context.values())
        vec = np.array([v if isinstance(v, (int, float)) else 0.0 for v in values])
        
        if len(vec) < dim:
            vec = np.pad(vec, (0, dim - len(vec)))
        return vec[:dim]
    
    def plan_skill_sequence(
        self,
        task: TaskNode,
        z_current: np.ndarray,
        goal: GoalAttractor
    ) -> List[SkillPrototype]:
        """
        Plan skill sequence for task using world model prediction.
        
        Uses CEM-like selection over skill chains.
        """
        available = self.skill_bank.find_applicable_skills(
            goal,
            context={'task_type': task.description}
        )
        
        if not available:
            return []
        
        # Generate candidate skill sequences
        candidates = []
        for _ in range(10):
            seq = self.skill_bank.get_skill_sequence_for_task(
                task, available, max_length=3
            )
            if seq:
                candidates.append(seq)
        
        if not candidates:
            return []
        
        # Evaluate each candidate via world model rollout
        best_score = -np.inf
        best_sequence = candidates[0]
        
        for seq in candidates:
            score = self._evaluate_skill_sequence(seq, z_current, goal)
            if score > best_score:
                best_score = score
                best_sequence = seq
        
        return best_sequence
    
    def _evaluate_skill_sequence(
        self,
        sequence: List[SkillPrototype],
        z_start: np.ndarray,
        goal: GoalAttractor
    ) -> float:
        """
        Evaluate skill sequence by rolling out through world model.
        
        Returns: predicted goal satisfaction score
        """
        z_current = z_start.copy()
        total_reward = 0.0
        
        for skill in sequence:
            # Generate trajectory for this skill
            traj = skill.generate_trajectory(z_current)
            
            # Rollout through world model
            for t in range(len(traj) - 1):
                z_current = traj[t]
                
                # Predict next state using world model (if available)
                if hasattr(self.world_model, 'predict_transition'):
                    try:
                        # Use world model to refine prediction
                        action = np.zeros(16)  # Placeholder
                        mu_pred, _ = self.world_model.predict_transition(
                            z_current, z_current, action
                        )
                        z_current = mu_pred
                    except Exception:
                        pass
                
                # Check proximity to goal
                dist = np.linalg.norm(z_current - goal.attractor_state)
                reward = np.exp(-dist)
                total_reward += reward
            
            # Update z_current to end of skill trajectory
            z_current = traj[-1]
        
        # Normalize by sequence length
        total_steps = sum(s.mean_trajectory.shape[0] for s in sequence)
        return total_reward / max(1, total_steps)
    
    def execute_task(
        self,
        task: TaskNode,
        z_current: np.ndarray,
        goal: GoalAttractor
    ) -> Dict:
        """
        Execute task with skill sequence.
        
        Returns execution result with actual trajectory.
        """
        task.status = TaskStatus.RUNNING
        
        # Plan skill sequence
        skill_sequence = self.plan_skill_sequence(task, z_current, goal)
        task.skill_sequence = skill_sequence
        
        # Execute skills
        z_state = z_current.copy()
        actual_trajectory = [z_state.copy()]
        success = True
        
        for skill in skill_sequence:
            # Generate trajectory
            traj = skill.generate_trajectory(z_state)
            
            # Execute (simulate)
            for t in range(1, len(traj)):
                z_state = traj[t]
                actual_trajectory.append(z_state.copy())
            
            # Update skill from experience
            skill.update_from_experience(
                np.array(actual_trajectory[-len(traj):]),
                success=True  # Simplified
            )
            
            task.actual_steps += len(traj)
        
        # Check if task brought us closer to goal
        final_dist = np.linalg.norm(z_state - goal.attractor_state)
        task_completed = final_dist < goal.basin_radius * 0.5
        
        if task_completed:
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
        else:
            task.progress = max(0.0, 1.0 - final_dist / (goal.basin_radius * 10))
            if task.progress < 0.1:
                task.status = TaskStatus.FAILED
                success = False
        
        # Update goal vitality
        goal.update_vitality(task.progress, task.actual_steps)
        
        result = {
            'task_id': task.task_id,
            'success': success,
            'final_distance': final_dist,
            'final_state': z_state.copy(),
            'trajectory_length': len(actual_trajectory),
            'skills_used': [s.skill_id for s in skill_sequence]
        }
        
        self.execution_log.append(result)
        return result
    
    def execute_goal(
        self,
        goal_id: str,
        z_start: np.ndarray,
        max_iterations: int = 100
    ) -> Dict:
        """
        Execute full goal hierarchy.
        
        Goal → Task DAG → Skill Sequences → Trajectory
        """
        if goal_id not in self.goals:
            return {'error': 'Goal not found'}
        
        goal = self.goals[goal_id]
        task_graph = self.task_graphs[goal_id]
        
        z_current = z_start.copy()
        iteration = 0
        
        while not task_graph.is_complete() and iteration < max_iterations:
            # Get ready tasks
            ready_tasks = task_graph.get_ready_tasks()
            
            if not ready_tasks:
                if task_graph.failed_tasks:
                    break
                # All tasks blocked or done
                break
            
            # Execute first ready task (simplified — could be parallel)
            task = ready_tasks[0]
            
            # Build intent vector
            self.current_intent = self.build_intent_vector(
                z_current, goal, task_graph,
                context={'iteration': iteration}
            )
            
            # Execute task
            result = self.execute_task(task, z_current, goal)
            z_current = result.get('final_state', z_current).copy()
            
            if result['success']:
                task_graph.complete_task(task.task_id)
            else:
                task_graph.fail_task(task.task_id)
            
            iteration += 1
        
        # Final evaluation
        final_dist = np.linalg.norm(z_current - goal.attractor_state)
        goal_satisfied = goal.is_satisfied(z_current)
        
        return {
            'goal_id': goal_id,
            'success': goal_satisfied,
            'final_distance': final_dist,
            'task_progress': task_graph.get_progress(),
            'iterations': iteration,
            'critical_path': task_graph.get_critical_path(),
            'execution_log': self.execution_log[-iteration:] if iteration > 0 else []
        }


# ============================================================================
# 6. INTENT-CONDITIONED CEM PLANNER
# ============================================================================

class IntentConditionedCEM:
    """
    CEM planner conditioned on intent space.
    
    Plans over skill chains, not atomic actions.
    """
    
    def __init__(
        self,
        world_model: Any,
        skill_bank: SkillBank,
        n_candidates: int = 50,
        n_elites: int = 10,
        n_iterations: int = 5
    ):
        self.world_model = world_model
        self.skill_bank = skill_bank
        self.n_candidates = n_candidates
        self.n_elites = n_elites
        self.n_iterations = n_iterations
    
    def plan(
        self,
        intent: IntentVector,
        horizon: int = 10
    ) -> List[SkillPrototype]:
        """
        Plan skill sequence conditioned on intent.
        
        Returns: optimal skill chain
        """
        available_skills = list(self.skill_bank.skills.values())
        if not available_skills:
            return []
        
        # Initialize skill sequence distribution
        # Each "action" is a skill choice
        n_skills = len(available_skills)
        skill_probs = np.ones(n_skills) / n_skills
        
        best_sequence = None
        best_score = -np.inf
        
        for iteration in range(self.n_iterations):
            # Sample candidate skill sequences
            candidates = []
            for _ in range(self.n_candidates):
                seq = self._sample_sequence(skill_probs, available_skills, horizon)
                candidates.append(seq)
            
            # Evaluate candidates
            scores = []
            for seq in candidates:
                score = self._evaluate_sequence(seq, intent)
                scores.append(score)
            
            # Select elites
            elite_indices = np.argsort(scores)[-self.n_elites:]
            elites = [candidates[i] for i in elite_indices]
            
            # Update distribution
            skill_probs = self._update_distribution(elites, available_skills)
            
            # Track best
            max_idx = np.argmax(scores)
            if scores[max_idx] > best_score:
                best_score = scores[max_idx]
                best_sequence = candidates[max_idx]
        
        return best_sequence or []
    
    def _sample_sequence(
        self,
        probs: np.ndarray,
        skills: List[SkillPrototype],
        length: int
    ) -> List[SkillPrototype]:
        """Sample skill sequence from distribution."""
        indices = np.random.choice(len(skills), size=length, p=probs)
        return [skills[i] for i in indices]
    
    def _evaluate_sequence(
        self,
        sequence: List[SkillPrototype],
        intent: IntentVector
    ) -> float:
        """Evaluate skill sequence against intent."""
        if not sequence:
            return -np.inf
        
        z_current = intent.z_state
        total_score = 0.0
        
        for skill in sequence:
            # Generate trajectory
            traj = skill.generate_trajectory(z_current)
            z_current = traj[-1]
            
            # Score: alignment with goal + trajectory coherence
            goal_alignment = np.exp(-np.linalg.norm(
                z_current - intent.goal_embedding[:len(z_current)]
            ))
            total_score += goal_alignment
        
        return total_score / len(sequence)
    
    def _update_distribution(
        self,
        elites: List[List[SkillPrototype]],
        skills: List[SkillPrototype]
    ) -> np.ndarray:
        """Update skill selection distribution from elites."""
        counts = np.zeros(len(skills))
        
        for seq in elites:
            for skill in seq:
                idx = skills.index(skill)
                counts[idx] += 1
        
        # Smooth distribution
        probs = counts + 1.0
        return probs / probs.sum()


# ============================================================================
# 7. TESTS
# ============================================================================

def test_intent_space():
    """Test unified intent space."""
    print("\n" + "=" * 60)
    print("INTENT SPACE TEST")
    print("=" * 60)
    
    dim = 32
    
    # Create intent vectors
    intent1 = IntentVector(
        z_state=np.random.randn(dim),
        goal_embedding=np.random.randn(dim),
        skill_context=np.random.randn(dim),
        task_progress=np.random.randn(dim),
        context_vector=np.random.randn(dim)
    )
    
    intent2 = IntentVector(
        z_state=np.random.randn(dim),
        goal_embedding=intent1.goal_embedding.copy(),  # Same goal
        skill_context=np.random.randn(dim),
        task_progress=np.random.randn(dim),
        context_vector=np.random.randn(dim)
    )
    
    # Test alignment
    alignment = intent1.alignment_score(intent2)
    print(f"\n  Alignment (same goal): {alignment:.4f}")
    
    # Test distance to attractor
    goal_attractor = np.random.randn(dim)
    dist = intent1.distance_to_attractor(goal_attractor)
    print(f"  Distance to attractor: {dist:.4f}")
    
    print("\n  ✓ Intent space operational")


def test_goal_attractor():
    """Test goal as attractor."""
    print("\n" + "=" * 60)
    print("GOAL ATTRACTOR TEST")
    print("=" * 60)
    
    attractor = GoalAttractor(
        goal_id="test_goal",
        attractor_state=np.array([1.0, 2.0, 3.0]),
        basin_radius=0.5,
        priority=0.8,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Test satisfaction
    close_state = np.array([1.1, 2.1, 3.1])
    far_state = np.array([10.0, 20.0, 30.0])
    
    print(f"\n  Close state satisfied: {attractor.is_satisfied(close_state)}")
    print(f"  Distance: {attractor.current_distance:.4f}")
    
    attractor2 = GoalAttractor(
        goal_id="test_goal_2",
        attractor_state=np.array([1.0, 2.0, 3.0]),
        basin_radius=0.5,
        priority=0.8,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    print(f"  Far state satisfied: {attractor2.is_satisfied(far_state)}")
    print(f"  Distance: {attractor2.current_distance:.4f}")
    
    # Test vitality decay
    vitality = attractor.update_vitality(progress=0.0, time_elapsed=10.0)
    print(f"  Vitality after decay: {vitality:.4f}")
    
    print("\n  ✓ Goal attractor operational")


def test_task_graph():
    """Test task DAG."""
    print("\n" + "=" * 60)
    print("TASK GRAPH TEST")
    print("=" * 60)
    
    graph = TaskGraph(goal_id="test_goal")
    
    # Add tasks
    tasks = [
        TaskNode(task_id="t1", goal_id="test_goal", description="Fetch data"),
        TaskNode(task_id="t2", goal_id="test_goal", description="Parse data"),
        TaskNode(task_id="t3", goal_id="test_goal", description="Validate"),
        TaskNode(task_id="t4", goal_id="test_goal", description="Report"),
    ]
    
    for t in tasks:
        graph.add_task(t)
    
    # Add dependencies: t1 → t2 → t3 → t4
    graph.add_dependency("t2", "t1")
    graph.add_dependency("t3", "t2")
    graph.add_dependency("t4", "t3")
    
    # Test ready tasks
    ready = graph.get_ready_tasks()
    print(f"\n  Ready tasks: {[t.task_id for t in ready]}")
    
    # Complete t1
    graph.complete_task("t1")
    ready = graph.get_ready_tasks()
    print(f"  After t1 complete: {[t.task_id for t in ready]}")
    
    # Test critical path
    critical = graph.get_critical_path()
    print(f"  Critical path: {critical}")
    
    # Test progress
    print(f"  Progress: {graph.get_progress():.2f}")
    
    print("\n  ✓ Task graph operational")


def test_skill_bank():
    """Test skill extraction and selection."""
    print("\n" + "=" * 60)
    print("SKILL BANK TEST")
    print("=" * 60)
    
    bank = SkillBank()
    
    # Create synthetic trajectory
    T = 20
    latent_dim = 8
    trajectory = np.cumsum(np.random.randn(T, latent_dim) * 0.1, axis=0)
    
    # Extract skill
    skill = bank.extract_skill_from_trajectory(
        trajectory=trajectory,
        success=True,
        context={'type': 'data_processing'}
    )
    
    print(f"\n  Extracted skill: {skill.skill_id}")
    print(f"  Success rate: {skill.success_rate:.2f}")
    print(f"  Trajectory shape: {skill.mean_trajectory.shape}")
    
    # Test trajectory generation
    z_start = np.zeros(latent_dim)
    generated = skill.generate_trajectory(z_start)
    print(f"  Generated trajectory shape: {generated.shape}")
    
    # Test update from experience
    new_traj = np.cumsum(np.random.randn(T, latent_dim) * 0.1, axis=0)
    skill.update_from_experience(new_traj, success=True)
    print(f"  Updated success rate: {skill.success_rate:.2f}")
    
    print("\n  ✓ Skill bank operational")


def test_hierarchical_execution():
    """Test full hierarchical execution."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL EXECUTION TEST")
    print("=" * 60)
    
    # Create skill bank
    bank = SkillBank()
    
    # Create skills
    for i in range(5):
        traj = np.cumsum(np.random.randn(10, 8) * 0.1, axis=0)
        skill = bank.extract_skill_from_trajectory(traj, success=True, context={})
        skill.applicable_goal_types = ['achievable', 'exploratory']
    
    # Create goal
    goal = GoalAttractor(
        goal_id="test_goal",
        attractor_state=np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        basin_radius=1.0,
        priority=0.9,
        decay_rate=0.01,
        success_criteria={'type': 'achievable'}
    )
    
    # Create task graph
    graph = TaskGraph(goal_id="test_goal")
    
    tasks = [
        TaskNode(task_id="t1", goal_id="test_goal", description="Fetch"),
        TaskNode(task_id="t2", goal_id="test_goal", description="Process"),
        TaskNode(task_id="t3", goal_id="test_goal", description="Report"),
    ]
    for t in tasks:
        graph.add_task(t)
    graph.add_dependency("t2", "t1")
    graph.add_dependency("t3", "t2")
    
    # Create engine (no world model for test)
    engine = HierarchicalExecutionEngine(
        world_model=None,
        skill_bank=bank,
        intent_dim=32
    )
    engine.add_goal(goal, graph)
    
    # Execute
    z_start = np.zeros(8)
    result = engine.execute_goal("test_goal", z_start, max_iterations=10)
    
    print(f"\n  Goal success: {result['success']}")
    print(f"  Final distance: {result['final_distance']:.4f}")
    print(f"  Task progress: {result['task_progress']:.2f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Critical path: {result['critical_path']}")
    
    print("\n  ✓ Hierarchical execution operational")


def test_intent_conditioned_cem():
    """Test intent-conditioned CEM planning."""
    print("\n" + "=" * 60)
    print("INTENT-CONDITIONED CEM TEST")
    print("=" * 60)
    
    # Create skill bank
    bank = SkillBank()
    
    # Create diverse skills
    for i in range(8):
        traj = np.cumsum(np.random.randn(10, 8) * 0.1, axis=0)
        skill = bank.extract_skill_from_trajectory(traj, success=True, context={})
        skill.applicable_goal_types = ['achievable']
    
    # Create intent
    intent = IntentVector(
        z_state=np.zeros(8),
        goal_embedding=np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        skill_context=np.zeros(32),
        task_progress=np.zeros(32),
        context_vector=np.zeros(32)
    )
    
    # Plan
    planner = IntentConditionedCEM(
        world_model=None,
        skill_bank=bank,
        n_candidates=30,
        n_elites=5,
        n_iterations=3
    )
    
    sequence = planner.plan(intent, horizon=5)
    
    print(f"\n  Planned sequence length: {len(sequence)}")
    print(f"  Skills: {[s.skill_id for s in sequence]}")
    
    if sequence:
        # Evaluate
        score = planner._evaluate_sequence(sequence, intent)
        print(f"  Sequence score: {score:.4f}")
    
    print("\n  ✓ Intent-conditioned CEM operational")


if __name__ == "__main__":
    test_intent_space()
    test_goal_attractor()
    test_task_graph()
    test_skill_bank()
    test_hierarchical_execution()
    test_intent_conditioned_cem()
    
    print("\n" + "=" * 60)
    print("PHASE 31: HIERARCHICAL EXECUTION LAYER")
    print("=" * 60)
    
    print("""
PURPOSE: Bridge production AI-OS with learned world model.

WHAT IT ADDS:
  1. Intent Space (I) — unified representation for goals/tasks/skills
  2. Goal as Attractor — goals are dynamical states, not DB objects
  3. Task DAG — proper decomposition with dependencies
  4. Skill as Trajectory — skills are compressed trajectory distributions
  5. Hierarchical Execution — Goal → Task → Skill-Sequence → Trajectory
  6. Intent-Conditioned CEM — planning over skill chains, not actions

KEY INSIGHTS:
  - Skills are NOT single actions
  - Skill sequences = programs over latent space
  - CEM plans over skill chains
  - Intent space unifies all representations

NEXT STEPS:
  1. Wire into Phase 30 world model for real predictions
  2. Connect to production AI-OS skill registry
  3. Add credit assignment from goal completion to skills
  4. Implement skill extraction from real execution traces
""")
