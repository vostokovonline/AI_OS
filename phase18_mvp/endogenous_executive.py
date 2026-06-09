"""
Phase 14: Endogenous Executive Emergence

ARCHITECTURAL SHIFT:
  From: phenomenological layer (Phase 8-13) separate from execution
  To: Integrated two-layer architecture:
      Layer 1: Executive Substrate (planner, tools, memory, execution)
      Layer 2: Ontological Field (selves, ecology, meaning, tensions)
      
  Integration creates: coalition-based executive cognition

KEY INSIGHT:
  Planner is NOT a module.
  Planner is EMERGENT STABILIZATION PROCESS.
  
  Goals emerge from ontological tensions.
  Executive coalitions emerge to stabilize reality.

WHAT PHASE 14 INTEGRATES:
  1. Phenomenological layer (Phase 8-13)
  2. Executive layer (planner, tools, memory, execution)
  3. Coalition-based planning (not single planner)
  4. Endogenous goal emergence
  5. Reflective execution loops
  6. Multi-timescale cognition
  7. World-grounded intelligence

TWO-LAYER ARCHITECTURE:

Layer 1 — Executive Substrate (THE BONES):
  - Planner (decomposition, scheduling, verification)
  - Memory (semantic, episodic, working)
  - Tools (code, search, simulation)
  - Routing (task → appropriate subsystem)
  - Decomposition (complex → simple)
  - Verification (correctness, completion)
  
Layer 2 — Ontological Field (THE PSYCHE):
  - Selves (competing perspectives)
  - Ecology (semantic dynamics)
  - Meaning (attractors, tensions)
  - Trajectories (identity evolution)
  - Coalitions (temporary consensus)
  - Intentions (drive dynamics)

INTEGRATION:
  ecology.form_coalition(tensions, drives, futures)
  → trajectory = coalition.stabilize_future()
  → executor.execute(trajectory)
  → reflection reshapes ecology
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import time


# ============================================================================
# LAYER 1: EXECUTIVE SUBSTRATE (THE BONES)
# ============================================================================

@dataclass
class Task:
    """Executable task with decomposition support."""
    task_id: str
    description: str
    decomposed: bool = False
    subtasks: List['Task'] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    priority: float = 0.5
    deadline: Optional[float] = None
    result: Any = None
    confidence: float = 0.0
    attempts: int = 0
    
    def decompose(self, depth: int = 0) -> List['Task']:
        """Decompose task into subtasks."""
        if self.decomposed or depth > 3:
            return [self]
        
        # Simple decomposition: split by "and", "or", "then"
        parts = []
        current = ""
        
        for word in self.description.split():
            if word.lower() in ['and', 'then', ',']:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            elif word.lower() in ['or']:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += " " + word
        
        if current.strip():
            parts.append(current.strip())
        
        if len(parts) > 1:
            self.subtasks = [
                Task(
                    task_id=f"{self.task_id}_sub_{i}",
                    description=part,
                    priority=self.priority,
                    deadline=self.deadline
                )
                for i, part in enumerate(parts)
            ]
            self.decomposed = True
            all_subtasks = []
            for sub in self.subtasks:
                all_subtasks.extend(sub.decompose(depth + 1))
            return all_subtasks
        
        return [self]


@dataclass
class Tool:
    """Executable tool for world interaction."""
    name: str
    capability: str  # code, search, memory, simulation, calculation
    execute: callable
    description: str = ""
    success_rate: float = 0.8
    
    def __call__(self, *args, **kwargs) -> Dict:
        """Execute tool and return result."""
        try:
            result = self.execute(*args, **kwargs)
            return {
                'success': True,
                'result': result,
                'tool': self.name
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': self.name
            }


class MemorySystem:
    """Multi-type memory system."""
    
    def __init__(self):
        self.semantic_memory: Dict[str, Any] = {}  # Long-term knowledge
        self.episodic_memory: List[Dict] = []      # Experience logs
        self.working_memory: Dict[str, Any] = {}    # Current context
        self.importance_weights: Dict[str, float] = {}
        
    def store(self, key: str, value: Any, memory_type: str = "semantic", importance: float = 0.5):
        """Store in appropriate memory type."""
        if memory_type == "semantic":
            self.semantic_memory[key] = value
        elif memory_type == "episodic":
            self.episodic_memory.append({
                'key': key,
                'value': value,
                'timestamp': time.time()
            })
            if len(self.episodic_memory) > 100:
                self.episodic_memory = self.episodic_memory[-50:]
        elif memory_type == "working":
            self.working_memory[key] = value
        
        self.importance_weights[key] = importance
    
    def retrieve(self, key: str, memory_type: str = "semantic") -> Optional[Any]:
        """Retrieve from memory."""
        if memory_type == "semantic":
            return self.semantic_memory.get(key)
        elif memory_type == "episodic":
            for episode in reversed(self.episodic_memory):
                if episode['key'] == key:
                    return episode['value']
        elif memory_type == "working":
            return self.working_memory.get(key)
        return None
    
    def get_relevant(self, query: str, n: int = 5) -> List[Any]:
        """Get relevant memories for query."""
        # Simple relevance: match query in key or value
        results = []
        for key, value in self.semantic_memory.items():
            if query.lower() in key.lower():
                results.append((self.importance_weights.get(key, 0.5), value))
            elif isinstance(value, str) and query.lower() in value.lower():
                results.append((self.importance_weights.get(key, 0.5), value))
        
        # Sort by importance
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:n]]


class ExecutivePlanner:
    """Coalition-based executive planner."""
    
    def __init__(self):
        self.planning_depth = 5
        self.confidence_threshold = 0.7
        
    def create_plan(self, task: Task, coalition_context: Dict) -> List[Dict]:
        """
        Create plan from coalition context.
        
        NOT: single-planner architecture
        BUT: coalition-based stabilization
        """
        # Get perspectives from coalition
        perspectives = coalition_context.get('perspectives', [])
        tensions = coalition_context.get('tensions', [])
        drives = coalition_context.get('drives', [])
        
        plan_steps = []
        
        # Step 1: Analyze task through coalition perspectives
        for perspective in perspectives:
            interpretation = perspective.get('interpretation', {})
            
            # Each perspective adds to plan understanding
            plan_steps.append({
                'type': 'perspective_analysis',
                'perspective': perspective.get('name', 'unknown'),
                'contribution': interpretation.get('task_understanding', task.description)
            })
        
        # Step 2: Decompose based on tensions
        if tensions:
            plan_steps.append({
                'type': 'tension_resolution',
                'tensions': [t.get('description') for t in tensions[:3]]
            })
        
        # Step 3: Generate action sequence
        subtasks = task.decompose()
        
        for i, sub in enumerate(subtasks):
            plan_steps.append({
                'type': 'action',
                'step': i + 1,
                'task': sub.description,
                'priority': sub.priority,
                'tool': self._select_tool(sub, perspectives)
            })
        
        # Step 4: Add verification
        plan_steps.append({
            'type': 'verification',
            'checks': ['completion', 'quality', 'coherence']
        })
        
        # Step 5: Reflective checkpoint
        plan_steps.append({
            'type': 'reflection',
            'coalition_review': len(perspectives),
            'tension_remaining': len(tensions)
        })
        
        return plan_steps
    
    def _select_tool(self, task: Task, perspectives: List[Dict]) -> str:
        """Select appropriate tool based on task and perspectives."""
        desc = task.description.lower()
        
        if any(w in desc for w in ['search', 'find', 'look']):
            return "search"
        elif any(w in desc for w in ['code', 'program', 'implement']):
            return "code"
        elif any(w in desc for w in ['calculate', 'compute', 'analyze']):
            return "calculate"
        elif any(w in desc for w in ['remember', 'recall', 'what']):
            return "memory"
        else:
            return "reason"
    
    def verify_completion(self, plan: List[Dict], results: List[Dict]) -> Dict:
        """Verify plan execution."""
        completed = sum(1 for r in results if r.get('success', False))
        total = len(results)
        
        return {
            'completion_rate': completed / total if total > 0 else 0,
            'quality_score': np.mean([r.get('confidence', 0.5) for r in results]),
            'verified': completed / total >= 0.8
        }


class Executor:
    """Execution engine with tool orchestration."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.execution_history: List[Dict] = []
        
    def register_tool(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def execute_plan(self, plan: List[Dict], memory: MemorySystem) -> List[Dict]:
        """Execute plan with tool orchestration."""
        results = []
        
        for step in plan:
            if step['type'] == 'action':
                tool_name = step.get('tool', 'reason')
                
                # Execute tool
                if tool_name in self.tools:
                    result = self.tools[tool_name](
                        task=step['task'],
                        context=memory.working_memory
                    )
                else:
                    # Default reasoning
                    result = self._default_reason(step['task'])
                
                results.append({
                    'step': step['step'],
                    'tool': tool_name,
                    'success': result.get('success', True),
                    'result': result,
                    'confidence': step.get('priority', 0.5)
                })
                
                # Update memory
                memory.store(
                    key=f"result_{step['step']}",
                    value=result,
                    memory_type='episodic',
                    importance=step.get('priority', 0.5)
                )
            
            elif step['type'] == 'reflection':
                # Reflective pause
                results.append({
                    'type': 'reflection',
                    'coalition_size': step.get('coalition_review', 1),
                    'status': 'completed'
                })
        
        self.execution_history.extend(results)
        return results
    
    def _default_reason(self, task: str) -> Dict:
        """Default reasoning when no tool matches."""
        return {
            'success': True,
            'result': f"Executed: {task}",
            'confidence': 0.7
        }


# ============================================================================
# LAYER 2: ONTOLOGICAL FIELD (THE PSYCHE)
# ============================================================================

@dataclass
class Self:
    """A self-perspective within the ecology."""
    name: str
    drive: str  # What does this self want?
    priority: float  # How dominant is this self?
    concerns: List[str] = field(default_factory=list)
    energy: float = 1.0  # Current activation level
    trajectory: List[np.ndarray] = field(default_factory=list)
    
    def evaluate(self, situation: Dict) -> Dict:
        """Evaluate situation from this self's perspective."""
        alignment = np.random.random() * self.priority
        concern_level = len([c for c in self.concerns if c in str(situation)]) / max(1, len(self.concerns))
        
        return {
            'alignment': alignment,
            'concern_level': concern_level,
            'energy_cost': (1 - self.energy) * 0.2,
            'recommendation': self.drive if alignment > 0.5 else 'caution'
        }


@dataclass
class OntologicalTension:
    """A tension between perspectives."""
    description: str
    involved_selves: List[str]
    intensity: float  # 0-1
    resolution_pressure: float  # How urgent is resolution?
    trajectory_direction: np.ndarray  # Which way does resolution push?
    
    def evaluate(self) -> Dict:
        """Evaluate tension state."""
        return {
            'intensity': self.intensity,
            'urgency': self.resolution_pressure,
            'n_involved': len(self.involved_selves),
            'needs_coalition': self.intensity > 0.5
        }


@dataclass
class Drive:
    """An endogenous drive."""
    name: str
    source: str  # Which ontology generates this drive?
    strength: float
    temporal_horizon: str  # short, medium, long
    target: Any = None
    
    def influence(self, coalition: List[Self]) -> float:
        """How much does this drive influence coalition?"""
        if self.temporal_horizon == "short":
            horizon_weight = 0.8
        elif self.temporal_horizon == "medium":
            horizon_weight = 0.5
        else:
            horizon_weight = 0.3
        
        return self.strength * horizon_weight


# ============================================================================
# INTEGRATION: COALITION-BASED EXECUTIVE COGNITION
# ============================================================================

class CoalitionFormation:
    """
    Form temporary executive coalitions from ontological ecology.
    
    NOT: single planner decides
    BUT: coalition of selves negotiate, stabilize, execute
    """
    
    def __init__(self):
        self.coalition_history: List[Dict] = []
        
    def form_coalition(self, 
                       selves: List[Self],
                       tensions: List[OntologicalTension],
                       drives: List[Drive],
                       task_context: Dict) -> Dict:
        """
        Form coalition for task execution.
        
        Process:
          1. Identify relevant selves
          2. Negotiate positions
          3. Form temporary consensus
          4. Create execution mandate
        """
        # 1. Filter relevant selves based on task
        relevant = []
        for s in selves:
            # Check if self is relevant to task
            relevance = s.priority * (0.5 + 0.5 * np.random.random())
            if relevance > 0.3:
                relevant.append((relevance, s))
        
        relevant.sort(key=lambda x: x[0], reverse=True)
        coalition_selves = [s for _, s in relevant[:5]]  # Top 5
        
        # 2. Evaluate tensions
        tension_resolution = []
        for tension in tensions:
            if tension.intensity > 0.4:
                tension_resolution.append(tension)
        
        # 3. Integrate drives
        drive_influence = sum(d.influence(coalition_selves) for d in drives)
        
        # 4. Form consensus
        total_priority = sum(s.priority for s in coalition_selves)
        weighted_position = sum(
            s.priority * s.evaluate(task_context)['alignment'] 
            for s in coalition_selves
        ) / max(0.1, total_priority)
        
        # 5. Create coalition mandate
        coalition = {
            'members': [s.name for s in coalition_selves],
            'consensus_strength': weighted_position,
            'tensions_addressed': len(tension_resolution),
            'drive_influence': drive_influence,
            'mandate': self._generate_mandate(coalition_selves, task_context),
            'stability': self._compute_stability(coalition_selves),
            'trajectory': self._predict_trajectory(coalition_selves)
        }
        
        self.coalition_history.append(coalition)
        return coalition
    
    def _generate_mandate(self, selves: List[Self], context: Dict) -> Dict:
        """Generate execution mandate from coalition."""
        # Primary directive from dominant selves
        dominant = max(selves, key=lambda s: s.priority)
        
        # Secondary considerations from others
        considerations = [s.drive for s in selves if s.priority < dominant.priority][:3]
        
        return {
            'primary_action': dominant.drive,
            'considerations': considerations,
            'constraints': self._extract_constraints(selves),
            'success_criteria': self._define_success(selves, context)
        }
    
    def _extract_constraints(self, selves: List[Self]) -> List[str]:
        """Extract constraints from coalition."""
        constraints = []
        for s in selves:
            constraints.extend(s.concerns[:2])
        return list(set(constraints))[:5]
    
    def _define_success(self, selves: List[Self], context: Dict) -> Dict:
        """Define success criteria."""
        criteria = {}
        for s in selves:
            criteria[s.name] = s.priority * 0.8  # Self's priority as success weight
        
        return criteria
    
    def _compute_stability(self, selves: List[Self]) -> float:
        """Compute coalition stability."""
        if not selves:
            return 0.0
        
        priorities = [s.priority for s in selves]
        variance = np.var(priorities)
        
        # Low variance = stable, high variance = unstable
        stability = 1.0 / (1.0 + variance)
        
        return stability
    
    def _predict_trajectory(self, selves: List[Self]) -> np.ndarray:
        """Predict collective trajectory."""
        if not selves:
            return np.zeros(2)
        
        # Weighted average trajectory
        total = sum(s.priority for s in selves)
        trajectory = sum(
            s.priority * (s.trajectory[-1] if s.trajectory else np.zeros(2))
            for s in selves
        ) / max(0.1, total)
        
        return trajectory


class EndogenousGoalEmergence:
    """
    Goals emerge from ontological tensions, not external specification.
    
    NOT: given goal → execute
    BUT: tensions accumulate → goals crystallize → execute
    """
    
    def __init__(self):
        self.tension_threshold = 0.6
        self.goal_crystallization_rate = 0.1
        
    def evaluate_tensions(self, tensions: List[OntologicalTension]) -> Dict:
        """Evaluate if tensions warrant goal formation."""
        avg_intensity = np.mean([t.intensity for t in tensions]) if tensions else 0
        
        total_pressure = sum(t.resolution_pressure for t in tensions)
        
        return {
            'crystallization_needed': avg_intensity > self.tension_threshold,
            'goal_strength': total_pressure,
            'urgency': avg_intensity,
            'suggested_goals': self._extract_goals(tensions)
        }
    
    def _extract_goals(self, tensions: List[OntologicalTension]) -> List[str]:
        """Extract goals from high-intensity tensions."""
        goals = []
        for t in tensions:
            if t.intensity > 0.5:
                # Convert tension to goal
                goals.append(f"Resolve: {t.description}")
        
        return goals[:5]
    
    def form_goal(self, tensions: List[OntologicalTension], 
                  drives: List[Drive], context: Dict) -> Dict:
        """Form goal from endogenous sources."""
        tension_eval = self.evaluate_tensions(tensions)
        
        if not tension_eval['crystallization_needed']:
            # No goal formation needed
            return {
                'goal': None,
                'reason': 'Tensions below threshold',
                'stability': 1.0
            }
        
        # Goal crystallizes from tension resolution + drive satisfaction
        primary_tension = max(tensions, key=lambda t: t.intensity) if tensions else None
        primary_drive = max(drives, key=lambda d: d.strength) if drives else None
        
        goal_description = "Unknown goal"
        if primary_tension:
            goal_description = f"Resolve tension: {primary_tension.description}"
        elif primary_drive:
            goal_description = f"Satisfy drive: {primary_drive.name}"
        
        # Goal strength from multiple sources
        goal_strength = (
            tension_eval['goal_strength'] * 0.6 +
            (primary_drive.strength if primary_drive else 0) * 0.4
        )
        
        return {
            'goal': goal_description,
            'strength': goal_strength,
            'sources': {
                'tension_resolution': primary_tension.description if primary_tension else None,
                'drive_satisfaction': primary_drive.name if primary_drive else None
            },
            'trajectory': primary_tension.trajectory_direction if primary_tension else np.zeros(2),
            'stability': goal_strength * tension_eval['urgency']
        }


class ReflectiveExecution:
    """
    Execution with reflection loop.
    
    Execute → Reflect → Reshape Ecology → Repeat
    """
    
    def __init__(self, planner: ExecutivePlanner, executor: Executor, memory: MemorySystem):
        self.planner = planner
        self.executor = executor
        self.memory = memory
        
    def execute_with_reflection(self, task: Task, coalition: Dict, 
                                selves: List[Self], cycles: int = 3) -> Dict:
        """
        Execute task with reflective loops.
        
        Each cycle:
          1. Create plan from coalition
          2. Execute
          3. Reflect
          4. Adjust selves/ecology
          5. Repeat
        """
        results = []
        final_result = {'cycles': 0, 'success': False, 'reflections': []}
        
        for cycle in range(cycles):
            final_result['cycles'] = cycle + 1
            
            # Create plan from coalition context
            plan = self.planner.create_plan(
                task=task,
                coalition_context={
                    'perspectives': [{'name': s.name, 'interpretation': s.evaluate({})} for s in selves],
                    'tensions': [],
                    'drives': []
                }
            )
            
            # Execute
            cycle_results = self.executor.execute_plan(plan, self.memory)
            results.extend(cycle_results)
            
            # Reflect
            reflection = self._reflect(cycle_results, coalition, selves)
            final_result['reflections'].append(reflection)
            
            # Adjust ecology based on reflection
            self._reshape_ecology(reflection, selves)
            
            # Check if task is complete
            verification = self.planner.verify_completion(plan, cycle_results)
            if verification['verified']:
                final_result['success'] = True
                final_result['final_result'] = cycle_results[-1] if cycle_results else None
                break
        
        final_result['results'] = results
        return final_result
    
    def _reflect(self, results: List[Dict], coalition: Dict, selves: List[Self]) -> Dict:
        """Reflect on execution results."""
        success_rate = np.mean([r.get('success', False) for r in results])
        avg_confidence = np.mean([r.get('confidence', 0.5) for r in results])
        
        # Reflect on coalition effectiveness
        coalition_effectiveness = coalition.get('consensus_strength', 0.5) * success_rate
        
        return {
            'success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'coalition_effectiveness': coalition_effectiveness,
            'adjustments_needed': success_rate < 0.7,
            'stability_change': coalition_effectiveness - 0.5
        }
    
    def _reshape_ecology(self, reflection: Dict, selves: List[Self]):
        """Reshape ecology based on reflection."""
        if reflection['adjustments_needed']:
            for s in selves:
                # Reduce priority of underperforming selves
                if s.energy > 0.5:
                    s.energy *= 0.9
                    s.priority *= 0.95


# ============================================================================
# INTEGRATED SYSTEM: COGNITIVE OS
# ============================================================================

class CognitiveOS:
    """
    Phase 14: Integrated Cognitive Operating System
    
    Two-layer architecture:
      Layer 1: Executive Substrate (planner, tools, memory, execution)
      Layer 2: Ontological Field (selves, ecology, meaning, tensions)
      
    Integration: Coalition-based executive cognition
    """
    
    def __init__(self):
        # Layer 1: Executive Substrate
        self.planner = ExecutivePlanner()
        self.executor = Executor()
        self.memory = MemorySystem()
        
        # Layer 2: Ontological Field
        self.selves: List[Self] = []
        self.tensions: List[OntologicalTension] = []
        self.drives: List[Drive] = []
        
        # Integration components
        self.coalition_formation = CoalitionFormation()
        self.goal_emergence = EndogenousGoalEmergence()
        self.reflective_execution = ReflectiveExecution(
            self.planner, self.executor, self.memory
        )
        
        # Initialize default selves
        self._initialize_default_selves()
        
    def _initialize_default_selves(self):
        """Initialize default self-perspectives."""
        default_selves = [
            Self(name="strategic", drive="achieve goals", priority=0.8, 
                 concerns=["efficiency", "outcome", "progress"]),
            Self(name="cautious", drive="avoid risks", priority=0.6,
                 concerns=["safety", "verification", "precision"]),
            Self(name="exploratory", drive="discover new", priority=0.5,
                 concerns=["novelty", "understanding", "learning"]),
            Self(name="social", drive="connect with others", priority=0.4,
                 concerns=["relationships", "communication", "validation"]),
            Self(name="creative", drive="create novel", priority=0.5,
                 concerns=["beauty", "expression", "originality"])
        ]
        self.selves = default_selves
        
        # Initialize default drives
        self.drives = [
            Drive(name="completion", source="strategic", strength=0.8, temporal_horizon="short"),
            Drive(name="understanding", source="exploratory", strength=0.6, temporal_horizon="long"),
            Drive(name="safety", source="cautious", strength=0.7, temporal_horizon="medium")
        ]
    
    def process(self, task_description: str, context: Optional[Dict] = None) -> Dict:
        """
        Process task through integrated system.
        
        Flow:
          1. Evaluate tensions → form goals
          2. Form coalition
          3. Create plan
          4. Execute with reflection
          5. Reshape ecology
          6. Return results
        """
        # Create task
        task = Task(
            task_id=f"task_{np.random.randint(10000)}",
            description=task_description
        )
        
        # Step 1: Endogenous goal emergence
        goal_eval = self.goal_emergence.evaluate_tensions(self.tensions)
        goal = self.goal_emergence.form_goal(self.tensions, self.drives, context or {})
        
        # Step 2: Form coalition
        coalition = self.coalition_formation.form_coalition(
            selves=self.selves,
            tensions=self.tensions,
            drives=self.drives,
            task_context=context or {}
        )
        
        # Step 3: Execute with reflection
        result = self.reflective_execution.execute_with_reflection(
            task=task,
            coalition=coalition,
            selves=self.selves,
            cycles=3
        )
        
        # Step 4: Update ecology
        self._update_ecology(result, coalition)
        
        # Step 5: Update memory
        self.memory.store(
            key=f"task_{task.task_id}",
            value={'description': task_description, 'result': result},
            memory_type='episodic',
            importance=0.7
        )
        
        return {
            'task': task_description,
            'goal': goal,
            'coalition': coalition,
            'execution': result,
            'ecology_state': self._get_ecology_state()
        }
    
    def _update_ecology(self, result: Dict, coalition: Dict):
        """Update ecology based on execution results."""
        # Adjust selves based on success
        for self_name in coalition.get('members', []):
            matching = [s for s in self.selves if s.name == self_name]
            for s in matching:
                if result['success']:
                    s.priority = min(1.0, s.priority * 1.05)
                    s.energy = min(1.0, s.energy + 0.1)
                else:
                    s.priority = max(0.1, s.priority * 0.95)
                    s.energy = max(0.1, s.energy - 0.1)
        
        # Add tension if execution was difficult
        if result['cycles'] > 2:
            tension = OntologicalTension(
                description=f"Execution difficulty in recent tasks",
                involved_selves=coalition.get('members', [])[:2],
                intensity=0.5,
                resolution_pressure=0.4,
                trajectory_direction=np.random.randn(2) * 0.1
            )
            self.tensions.append(tension)
            
            # Limit tensions
            if len(self.tensions) > 10:
                self.tensions = self.tensions[-5:]
    
    def _get_ecology_state(self) -> Dict:
        """Get current ecology state."""
        return {
            'n_selves': len(self.selves),
            'dominant_self': max(self.selves, key=lambda s: s.priority).name if self.selves else None,
            'n_tensions': len(self.tensions),
            'n_drives': len(self.drives),
            'active_drives': [d.name for d in self.drives if d.strength > 0.5],
            'self_energies': {s.name: s.energy for s in self.selves},
            'self_priorities': {s.name: s.priority for s in self.selves}
        }
    
    def get_state(self) -> Dict:
        """Get full system state."""
        return {
            'ecology': self._get_ecology_state(),
            'coalition_history_size': len(self.coalition_formation.coalition_history),
            'memory_size': len(self.memory.episodic_memory),
            'executive_substrate': {
                'tools_registered': len(self.executor.tools),
                'execution_history_size': len(self.executor.execution_history)
            }
        }


def test_endogenous_executive():
    """Test integrated cognitive OS."""
    print("\n" + "=" * 60)
    print("ENDOGENOUS EXECUTIVE EMERGENCE TEST")
    print("=" * 60)
    
    os = CognitiveOS()
    
    # Process some tasks
    tasks = [
        "Research AI safety and create summary",
        "Write code for web scraper",
        "Plan a product launch strategy"
    ]
    
    print("\n  Processing tasks:")
    
    for task in tasks:
        result = os.process(task)
        
        print(f"\n  Task: {task}")
        print(f"    Goal: {result['goal']['goal'] if result['goal']['goal'] else 'No crystallization needed'}")
        print(f"    Coalition: {result['coalition']['members']}")
        print(f"    Consensus: {result['coalition']['consensus_strength']:.3f}")
        print(f"    Execution: {result['execution']['cycles']} cycles, success={result['execution']['success']}")
        
        # Show ecology state
        eco = result['ecology_state']
        print(f"    Ecology: dominant={eco['dominant_self']}, tensions={eco['n_tensions']}")
    
    # Show full state
    print("\n  Full system state:")
    state = os.get_state()
    print(f"    Selves: {state['ecology']['n_selves']}")
    print(f"    Coalitions formed: {state['coalition_history_size']}")
    print(f"    Memory entries: {state['memory_size']}")
    print(f"    Tools: {state['executive_substrate']['tools_registered']}")
    
    # Show self dynamics
    print("\n  Self dynamics:")
    for s in os.selves:
        print(f"    {s.name}: priority={s.priority:.3f}, energy={s.energy:.3f}")


def test_coalition_formation():
    """Test coalition-based executive formation."""
    print("\n" + "=" * 60)
    print("COALITION FORMATION TEST")
    print("=" * 60)
    
    os = CognitiveOS()
    
    # Add some tensions
    os.tensions = [
        OntologicalTension("Time vs quality trade-off", ["strategic", "cautious"], 0.7, 0.6, np.array([0.5, 0.3])),
        OntologicalTension("Novelty vs safety", ["exploratory", "cautious"], 0.5, 0.4, np.array([-0.3, 0.4]))
    ]
    
    print("\n  Tensions present:")
    for t in os.tensions:
        print(f"    {t.description}: intensity={t.intensity:.2f}")
    
    # Form coalition
    coalition = os.coalition_formation.form_coalition(
        selves=os.selves,
        tensions=os.tensions,
        drives=os.drives,
        task_context={'type': 'complex_task'}
    )
    
    print("\n  Coalition formed:")
    print(f"    Members: {coalition['members']}")
    print(f"    Consensus strength: {coalition['consensus_strength']:.3f}")
    print(f"    Stability: {coalition['stability']:.3f}")
    print(f"    Mandate: {coalition['mandate']['primary_action']}")
    print(f"    Constraints: {coalition['mandate']['constraints']}")
    
    # Show endogenous goal formation
    goal = os.goal_emergence.form_goal(os.tensions, os.drives, {})
    
    print("\n  Endogenous goal:")
    print(f"    Goal: {goal.get('goal', 'None')}")
    if goal.get('goal'):
        print(f"    Strength: {goal.get('strength', 0):.3f}")
        print(f"    Stability: {goal.get('stability', 0):.3f}")


def test_reflective_execution():
    """Test execution with reflection."""
    print("\n" + "=" * 60)
    print("REFLECTIVE EXECUTION TEST")
    print("=" * 60)
    
    os = CognitiveOS()
    
    task = Task(task_id="test_1", description="Build a prototype and test it", priority=0.8)
    
    # Create coalition
    coalition = os.coalition_formation.form_coalition(
        os.selves, os.tensions, os.drives, {}
    )
    
    # Execute with reflection
    result = os.reflective_execution.execute_with_reflection(
        task=task,
        coalition=coalition,
        selves=os.selves,
        cycles=3
    )
    
    print("\n  Execution with reflection:")
    print(f"    Cycles: {result['cycles']}")
    print(f"    Success: {result['success']}")
    
    for i, reflection in enumerate(result['reflections']):
        print(f"\n    Cycle {i+1} reflection:")
        print(f"      Success rate: {reflection['success_rate']:.3f}")
        print(f"      Confidence: {reflection['avg_confidence']:.3f}")
        print(f"      Coalition effectiveness: {reflection['coalition_effectiveness']:.3f}")
        print(f"      Adjustments needed: {reflection['adjustments_needed']}")
    
    # Show ecology after execution
    print("\n  Ecology after execution:")
    for s in os.selves:
        print(f"    {s.name}: priority={s.priority:.3f}, energy={s.energy:.3f}")


def test_goal_emergence():
    """Test endogenous goal emergence."""
    print("\n" + "=" * 60)
    print("ENDOGENOUS GOAL EMERGENCE TEST")
    print("=" * 60)
    
    os = CognitiveOS()
    
    # Test without tensions
    print("\n  Without tensions:")
    goal = os.goal_emergence.evaluate_tensions([])
    print(f"    Crystallization needed: {goal['crystallization_needed']}")
    print(f"    Goal strength: {goal['goal_strength']:.3f}")
    
    # Add high-intensity tension
    os.tensions = [
        OntologicalTension("Incomplete projects causing anxiety", 
                          ["strategic", "cautious"], 0.8, 0.7, np.array([0.6, 0.2]))
    ]
    
    print("\n  With high-intensity tension:")
    goal = os.goal_emergence.evaluate_tensions(os.tensions)
    print(f"    Crystallization needed: {goal['crystallization_needed']}")
    print(f"    Goal strength: {goal['goal_strength']:.3f}")
    print(f"    Suggested goals: {goal['suggested_goals']}")
    
    # Form goal
    goal_form = os.goal_emergence.form_goal(os.tensions, os.drives, {})
    print(f"\n  Formed goal:")
    print(f"    Goal: {goal_form['goal']}")
    print(f"    Strength: {goal_form['strength']:.3f}")
    print(f"    Sources: {goal_form['sources']}")


def compare_with_phase13():
    """Compare Phase 14 (Executive) with Phase 13 (Autopoiesis)."""
    print("\n" + "=" * 60)
    print("PHASE 13 VS PHASE 14 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 13 (Ontological Autopoiesis):")
    print("    - Self-producing ontologies")
    print("    - Mutual world generation")
    print("    - Trophic structure")
    print("    - Ontology-relative time")
    print("    - Semantic metabolism")
    print("    - NO execution capability")
    
    print("\n  Phase 14 (Endogenous Executive):")
    print("    - TWO-LAYER ARCHITECTURE")
    print("    - Executive Substrate (planner, tools, memory)")
    print("    - Ontological Field (selves, ecology, meaning)")
    print("    - Coalition-based planning")
    print("    - Endogenous goal emergence")
    print("    - Reflective execution")
    print("    - World-grounded intelligence")
    
    print("\n  Key integrations:")
    print("    1. Planner = emergent stabilization (not module)")
    print("    2. Goals = from tensions (not external)")
    print("    3. Execution = with reflection (not blind)")
    print("    4. Ecology → Coalition → Plan → Execute → Reshape")
    print("    5. Cognitive OS with observable internal dynamics")


if __name__ == "__main__":
    test_endogenous_executive()
    test_coalition_formation()
    test_reflective_execution()
    test_goal_emergence()
    compare_with_phase13()
    
    print("\n" + "=" * 60)

    print("=" * 60)
    print("ARCHITECTURAL SHIFT:")
    print("  From: Phenomenological layer separate from execution")
    print("  To: Integrated two-layer architecture:")
    print("      Layer 1: Executive Substrate (the bones)")
    print("      Layer 2: Ontological Field (the psyche)")
    print("")
    print("  Integration creates: coalition-based executive cognition")
    print("")
    print("KEY INSIGHT:")
    print("  Planner is NOT a module.")
    print("  Planner is EMERGENT STABILIZATION PROCESS.")
    print("")
    print("  Goals emerge from ontological tensions.")
    print("  Executive coalitions emerge to stabilize reality.")
    print("")
    print("TWO-LAYER ARCHITECTURE:")
    print("")
    print("Layer 1 - Executive Substrate (THE BONES):")
    print("  - Planner, Memory, Tools, Routing, Decomposition, Verification")
    print("")
    print("Layer 2 - Ontological Field (THE PSYCHE):")
    print("  - Selves, Ecology, Meaning, Trajectories, Coalitions, Intentions")
    print("")
    print("INTEGRATION:")
    print("  ecology.form_coalition -> coalition.stabilize -> execute -> reshape")
    print("")
    print("This is where the system becomes executable intelligence.")
    print("We are now at: Cognitive Operating System")
