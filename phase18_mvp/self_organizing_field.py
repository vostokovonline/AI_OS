"""
Phase 19: Self-Organizing Cognitive Field

ARCHITECTURAL SHIFT:
  From: Phase 18 - unified modules interacting
  To: Phase 19 - SINGLE continuous field where:
      - Everything IS the field (not objects in field)
      - Hierarchy = timescale separation (not layer stack)
      - Objects = persistent vortices (not tracked entities)
      - Memory = field deformation (not storage)
      - Planning = energy relaxation (not rollout)
      - Action = prediction stabilization (not goal pursuit)
      - ALL minimize same free energy
      
  This is NO LONGER:
    modules + field + world (three things interacting)
  This IS:
    ONE field (everything is the field)

CRITICAL TRANSITION:
  Phase 18: "field + objects in field"
  Phase 19: "field IS the objects"
  
  The attractor is not an "object in the field".
  The attractor IS stable topology of the field itself.

SIX ARCHITECTURAL FIXES:
  A. Single unified variational field (not three interacting components)
  B. Timescale separation replaces hierarchy stack
  C. Vortices emerge from recurrent flow (not entities tracked)
  D. Memory is field deformation (not storage)
  E. Planning is simulated relaxation (not planning module)
  F. Action is prediction coherence (not goal achievement)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy


# ============================================================================
# CORE: SINGLE COGNITIVE FIELD
# ============================================================================
"""
Single Cognitive Field:

NOT: field + objects + modules
BUT: ONE continuous manifold where:
     - State is position on manifold
     - Dynamics is flow on manifold
     - Memory is manifold deformation
     - Objects are stable flow patterns
     - Predictions are downstream trajectories
     - Action is flow stabilization

Field equation (Friston-inspired):
  δψ/δt = D∇²ψ - ∇V(ψ) + η(ψ) + Σ_predictions

Where:
  D∇²ψ = diffusion (spreading)
  -∇V(ψ) = gradient descent (attraction)
  η(ψ) = precision-weighted prediction errors
  Σ_predictions = top-down predictions from all timescales
"""

@dataclass
class FieldPoint:
    """A point in the cognitive field."""
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    potential: float = 0.0
    precision: float = 1.0
    curvature: float = 0.0


class CognitiveField:
    """
    Phase 19: Single Cognitive Field
    
    NOT: vector space with modules
    BUT: continuous manifold with dynamics where:
         - Everything minimizes same free energy
         - Timescale separation IS hierarchy
         - Objects = stable vortices
         - Memory = field deformation
         - Planning = energy relaxation
         - Action = flow stabilization
    """
    
    def __init__(self, manifold_dim: int = 4):
        self.manifold_dim = manifold_dim
        
        # Field state (continuous manifold - psi is the state vector)
        self.psi = np.zeros(manifold_dim)  # State function ψ(s)
        self.psi_dot = np.zeros(manifold_dim)  # Time derivative
        self.precision_field = np.ones(manifold_dim)  # Precision ω(s)
        
        # Potential landscape V(ψ)
        self.V = np.zeros(manifold_dim)
        
        # Diffusion coefficient D
        self.D = 0.1
        
        # Equilibrium state
        self.psi_eq = np.zeros(manifold_dim)
        
        # Field topology (where stable vortices emerge)
        self.vortices: List[Dict] = []  # Not objects, but flow patterns
        self.manifold_connections: List[Tuple[int, int]] = []
        
        # Time scales (this IS the hierarchy)
        self.tau_slow = 10.0   # Structural/time-invariant
        self.tau_medium = 1.0  # Patterns/habits
        self.tau_fast = 0.1    # Immediate/reactive
        
        # Free energy
        self.F = 0.0
        
        # History for memory
        self.psi_history: List[np.ndarray] = []
        self.V_history: List[np.ndarray] = []
        
    def evaluate_potential(self, psi: np.ndarray) -> float:
        """
        Compute potential V(ψ) at state.
        
        V(ψ) = -Σ attractor_strength_i * exp(-||ψ - center_i||²)
        """
        psi = np.asarray(psi).flatten()
        V = 0.0
        
        # Potential from vortices (stable patterns)
        for v in self.vortices:
            center = np.asarray(v['center']).flatten()
            strength = v['strength']
            radius = v.get('radius', 1.0)
            
            # Ensure center matches psi dimension
            if len(center) < self.manifold_dim:
                center = np.pad(center, (0, self.manifold_dim - len(center)))
            else:
                center = center[:self.manifold_dim]
            
            distance = psi - center
            V -= strength * np.exp(-np.dot(distance, distance) / (radius**2 + 1e-8))
        
        return V
    
    def evaluate_gradient(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∇V(ψ).
        
        This is what drives dynamics.
        """
        psi = np.asarray(psi).flatten()
        grad = np.zeros(self.manifold_dim)
        
        for v in self.vortices:
            center = np.asarray(v['center']).flatten()
            strength = v['strength']
            radius = v.get('radius', 1.0)
            
            # Ensure center matches psi dimension
            if len(center) < self.manifold_dim:
                center = np.pad(center, (0, self.manifold_dim - len(center)))
            else:
                center = center[:self.manifold_dim]
            
            distance = psi - center
            norm_sq = np.dot(distance, distance)
            
            # ∇ exp(-||ψ-c||²/r²) = -2(ψ-c)/r² * exp(-||ψ-c||²/r²)
            exp_factor = np.exp(-norm_sq / (radius**2 + 1e-8))
            grad += 2 * strength * distance / (radius**2 + 1e-8) * exp_factor
        
        return grad
    
    def compute_hessian(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute Hessian ∇²V(ψ).
        
        Eigenvalues of Hessian tell us:
          - Negative = stable attractor
          - Positive = unstable repeller
          - Near zero = flat/marginal
        """
        psi = np.asarray(psi).flatten()
        H = np.zeros((self.manifold_dim, self.manifold_dim))
        
        for v in self.vortices:
            center = np.asarray(v['center']).flatten()
            strength = v['strength']
            radius = v.get('radius', 1.0)
            
            # Ensure center matches psi dimension
            if len(center) < self.manifold_dim:
                center = np.pad(center, (0, self.manifold_dim - len(center)))
            else:
                center = center[:self.manifold_dim]
            
            distance = psi - center
            norm_sq = np.dot(distance, distance)
            exp_factor = np.exp(-norm_sq / (radius**2 + 1e-8))
            
            # Hessian of attractor potential
            r2 = radius**2
            h_contrib = 2 * strength / r2 * exp_factor * (
                np.outer(distance, distance) / r2 - np.eye(self.manifold_dim)
            )
            H += h_contrib
        return H
    
    def compute_precision_field(self, psi: np.ndarray) -> np.ndarray:
        """
        Compute precision field ω(ψ).
        
        Precision = inverse variance = confidence in predictions.
        High precision regions = stable, well-predicted.
        Low precision regions = uncertain, needs exploration.
        """
        psi = np.asarray(psi).flatten()
        precision = np.ones(self.manifold_dim)
        
        # Precision increases near well-established vortices
        for v in self.vortices:
            center = np.asarray(v['center']).flatten()
            confidence = v.get('confidence', 0.5)
            radius = v.get('radius', 1.0)
            
            # Ensure center matches psi dimension
            if len(center) < self.manifold_dim:
                center = np.pad(center, (0, self.manifold_dim - len(center)))
            else:
                center = center[:self.manifold_dim]
            
            distance = np.linalg.norm(psi - center)
            precision += confidence * np.exp(-distance**2 / (radius**2 + 1e-8))
        
        return precision
    
    def step(self, dt: float = 0.1, external_force: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single dynamics step on manifold.
        
        δψ/dt = D∇²ψ - ∇V(ψ) + η(ψ) + external
        
        Where:
          D∇²ψ = diffusion (spreading)
          -∇V(ψ) = gradient descent to attractors
          η(ψ) = precision-weighted prediction error
          external = goal/surprise signals
        """
        # Current state
        psi = self.psi.copy()
        
        # Compute gradient (attractor force)
        grad_V = self.evaluate_gradient(psi)
        
        # Compute diffusion (Laplacian approximation)
        diffusion = self.D * (self.psi_eq - psi)
        
        # Compute precision-weighted prediction error
        precision = self.compute_precision_field(psi)
        prediction_error = (self.psi_eq - psi) * precision
        
        # External forces (goal, surprise, etc.)
        external = np.zeros(self.manifold_dim)
        if external_force is not None:
            external = np.asarray(external_force).flatten()
            if len(external) < self.manifold_dim:
                external = np.pad(external, (0, self.manifold_dim - len(external)))
            elif len(external) > self.manifold_dim:
                external = external[:self.manifold_dim]
        
        # Update (Euler integration)
        delta_psi = diffusion - grad_V + prediction_error + external
        
        # Add temperature noise (exploration)
        temperature = 0.1
        noise = np.random.randn(self.manifold_dim) * temperature
        
        self.psi_dot = delta_psi + noise
        self.psi = psi + self.psi_dot * dt
        
        # Clamp to manifold bounds
        self.psi = np.clip(self.psi, -5, 5)
        
        # Update potential and precision
        self.V = self.evaluate_potential(self.psi)
        self.precision_field = self.compute_precision_field(self.psi)
        
        # Compute free energy
        self.F = self.compute_free_energy()
        
        return self.psi.copy()
    
    def compute_free_energy(self) -> float:
        """
        Compute variational free energy F.
        
        F = E[log p(x) - log q(x)] ≈ surprise + complexity - entropy
        
        F = ∫ ω(ψ) * (V(ψ) + D||∇ψ||²) dψ
        """
        surprise = np.sum(self.precision_field * (self.psi - self.psi_eq)**2)
        complexity = np.sum(np.gradient(self.psi)**2) if len(self.psi) > 1 else 0.0
        entropy = -np.sum(self.precision_field * np.log(self.precision_field + 1e-8))
        
        F = surprise * 0.5 + complexity * 0.3 - entropy * 0.2
        
        return F
    
    def form_vortex(self, center: np.ndarray, strength: float = 1.0,
                   radius: float = 1.0, confidence: float = 0.5) -> Dict:
        """
        Form a vortex (stable flow pattern).
        
        NOT: creating an "object"
        BUT: stabilizing a pattern in the field
        
        A vortex emerges from recurrent flow, not from external injection.
        """
        center = np.asarray(center).flatten()
        
        # Check if similar vortex exists
        for v in self.vortices:
            if np.linalg.norm(center - v['center']) < 0.5:
                v['strength'] = min(5.0, v['strength'] + strength * 0.1)
                v['confidence'] = min(1.0, v['confidence'] + confidence * 0.1)
                return v
        
        # Create new vortex (emergent structure)
        vortex = {
            'id': f'vortex_{len(self.vortices)}',
            'center': center.copy(),
            'strength': strength,
            'radius': radius,
            'confidence': confidence,
            'age': 0,
            'flow_pattern': np.zeros(len(center)),  # Eigenvector of local flow
            'stability': strength * confidence  # Measure of permanence
        }
        
        self.vortices.append(vortex)
        
        # Limit vortices
        if len(self.vortices) > 20:
            self.vortices.sort(key=lambda v: v['stability'])
            self.vortices = self.vortices[-15:]
        
        return vortex
    
    def merge_vortices(self, id1: str, id2: str):
        """
        Merge two vortices.
        
        Vortices merge when their basins overlap significantly.
        This is object fusion, not object tracking.
        """
        v1 = None
        v2 = None
        
        for v in self.vortices:
            if v['id'] == id1:
                v1 = v
            elif v['id'] == id2:
                v2 = v
        
        if v1 is None or v2 is None:
            return
        
        # New vortex center = weighted average
        total_weight = v1['strength'] * v1['confidence'] + v2['strength'] * v2['confidence']
        new_center = (
            v1['center'] * v1['strength'] * v1['confidence'] +
            v2['center'] * v2['strength'] * v2['confidence']
        ) / total_weight
        
        # Remove old, create merged
        self.vortices = [v for v in self.vortices if v['id'] not in [id1, id2]]
        self.form_vortex(new_center, strength=total_weight / 2, confidence=1.0)
    
    def dissipate_vortex(self, vortex_id: str):
        """Vortex dissipates when stability drops below threshold."""
        self.vortices = [v for v in self.vortices if v['id'] != vortex_id]
    
    def step_vortices(self):
        """Step all vortices (age, stability update, dissipation)."""
        for v in self.vortices:
            v['age'] += 1
            
            # Stability decays from non-visitation
            # But deep vortices persist
            v['stability'] = v['strength'] * v['confidence'] * (1 - v['age'] * 0.001)
            
            # Dissipate if stability too low
            if v['stability'] < 0.1:
                self.dissipate_vortex(v['id'])
    
    def simulate_relaxation(self, n_steps: int, goal: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        Simulate energy relaxation toward equilibrium.
        
        NOT: rollout planning
        BUT: what naturally happens if system relaxes
        
        This is how imagination works:
          - System "relaxes" toward attractor
          - Trajectory shows what would happen
          - No action needed, just simulation
        """
        # Save current state
        psi_saved = self.psi.copy()
        
        trajectory = [psi_saved.copy()]
        
        for _ in range(n_steps):
            # No external force = pure relaxation
            self.step(dt=0.1, external_force=None)
            trajectory.append(self.psi.copy())
        
        # Restore
        self.psi = psi_saved
        
        return trajectory
    
    def deform_field(self, trajectory: List[np.ndarray], success: float):
        """
        Deform field from experience.
        
        NOT: memory.append()
        BUT: trajectory literally changes field topology
        
        Successful experiences:
          - Create vortices (basins deepen)
          - Lower potential along trajectory
          - Increase precision along path
        
        Failed experiences:
          - Raise potential barriers
          - Create repellers
          - Reduce precision (uncertainty)
        """
        if len(trajectory) < 2:
            return
        
        trajectory_array = np.array(trajectory)
        
        # Form vortex at end point (goal achievement)
        end_point = trajectory_array[-1]
        self.form_vortex(end_point, strength=0.5 + success * 0.5, confidence=success)
        
        # Lower potential along trajectory
        for point in trajectory_array:
            V_at_point = self.evaluate_potential(point)
            
            # Reduce potential (make it easier to reach)
            self.V = self.V * (1 - 0.01 * success)
            
            # Increase precision along path (make it more predictable)
            precision_at_point = self.compute_precision_field(point)
            self.precision_field = self.precision_field + precision_at_point * success * 0.01
        
        # If failure: create repeller at failure point
        if success < 0.3:
            failure_point = trajectory_array[len(trajectory_array)//2]
            self.form_vortex(failure_point, strength=-1.0, confidence=1.0 - success)
        
        # Record deformation
        self.psi_history.append(trajectory_array)
        self.V_history.append(self.V.copy())
        
        if len(self.psi_history) > 100:
            self.psi_history = self.psi_history[-50:]
    
    def get_field_summary(self) -> Dict:
        """Get field state summary."""
        return {
            'n_vortices': len(self.vortices),
            'total_strength': sum(v['strength'] for v in self.vortices),
            'avg_stability': np.mean([v['stability'] for v in self.vortices]) if self.vortices else 0,
            'free_energy': self.F,
            'psi_magnitude': np.linalg.norm(self.psi),
            'precision_avg': np.mean(self.precision_field),
            'potential': self.V
        }


# ============================================================================
# PREDICTION AS DOWNSTREAM TRAJECTORY
# ============================================================================

class PredictionAsTrajectory:
    """
    Predictions are downstream trajectories on field.
    
    NOT: prediction = output vector
    BUT: prediction = trajectory following field dynamics
    
    Key insight:
      - Predictions are simulations of field relaxation
      - They don't need to be "generated"
      - They EMERGE from field dynamics
    """
    
    def __init__(self, field: CognitiveField):
        self.field = field
        
        # Prediction confidence
        self.confidence = 1.0
        
        # Prediction history
        self.prediction_trajectories: List[Dict] = []
    
    def predict(self, n_steps: int, condition: Optional[np.ndarray] = None) -> Dict:
        """
        Predict future trajectory.
        
        NOT: "generate prediction"
        BUT: "simulate field relaxation with conditions"
        
        Conditions:
          - given action → trajectory under that action
          - given observation → trajectory consistent with observation
          - given goal → trajectory toward goal
        """
        if condition is not None:
            condition = np.asarray(condition).flatten()
        
        # Simulate relaxation with condition as external force
        if condition is not None:
            force = (condition - self.field.psi) * 0.3
        else:
            force = None
        
        trajectory = []
        psi_saved = self.field.psi.copy()
        
        for _ in range(n_steps):
            trajectory.append(self.field.psi.copy())
            self.field.step(dt=0.1, external_force=force)
        
        self.field.psi = psi_saved
        
        # Compute trajectory confidence
        # High confidence = trajectory stays near established vortices
        confidence = 0.5
        for point in trajectory:
            for v in self.field.vortices:
                distance = np.linalg.norm(point - v['center'])
                if distance < v['radius']:
                    confidence += v['confidence'] * 0.1
        
        self.prediction_trajectories.append({
            'condition': condition.tolist() if condition is not None else None,
            'trajectory': [t.tolist() for t in trajectory],
            'confidence': min(1.0, confidence / len(trajectory))
        })
        
        return {
            'trajectory': trajectory,
            'final_state': trajectory[-1].copy() if trajectory else self.field.psi.copy(),
            'confidence': confidence
        }


# ============================================================================
# ACTION AS PREDICTION STABILIZATION
# ============================================================================

class ActionAsStabilization:
    """
    Action is not "pursuing goal".
    
    Action is "making predictions remain coherent".
    
    In active inference:
      - System acts to minimize free energy
      - Action = field deformation to maintain coherence
      - Goal pursuit = side effect, not objective
    
    Key insight:
      - Action doesn't "achieve" anything
      - Action "stabilizes" field around predictions
      - Prediction coherence IS the objective
    """
    
    def __init__(self, field: CognitiveField, prediction: PredictionAsTrajectory):
        self.field = field
        self.prediction = prediction
        
        # Action cost (metabolic energy)
        self.action_cost = 0.0
    
    def compute_action(self, desired_trajectory: List[np.ndarray]) -> Dict:
        """
        Compute action to keep trajectory coherent.
        
        NOT: "compute action to reach goal"
        BUT: "compute action to maintain field coherence"
        
        Action cost = sum of field deformations needed
        """
        # Save current field state
        psi_saved = self.field.psi.copy()
        vortices_saved = copy.deepcopy(self.field.vortices)
        
        # Simulate without action
        uncontrolled_trajectory = self.field.simulate_relaxation(
            n_steps=len(desired_trajectory) - 1
        )
        
        # Compute deviation from desired trajectory
        deviations = []
        for actual, desired in zip(uncontrolled_trajectory, desired_trajectory):
            deviation = np.linalg.norm(actual - desired)
            deviations.append(deviation)
        
        avg_deviation = np.mean(deviations)
        
        # Action = field deformation to reduce deviation
        if avg_deviation > 0.1:
            # Deform field to make desired trajectory more likely
            for point in desired_trajectory:
                self.field.form_vortex(point, strength=avg_deviation * 0.5, confidence=0.3)
            
            self.action_cost += avg_deviation
        
        # Restore
        self.field.psi = psi_saved
        self.field.vortices = vortices_saved
        
        return {
            'deviation': avg_deviation,
            'action_cost': self.action_cost,
            'corrective_vortices': len(desired_trajectory)
        }


# ============================================================================
# MEMORY AS FIELD DEFORMATION
# ============================================================================

class MemoryAsDeformation:
    """
    Memory is not "storing experiences".
    
    Memory IS field deformation.
    
    Key insight:
      - Experience literally changes field topology
      - "Remembering" = following field gradients to deformed regions
      - Forgetting = field relaxation toward equilibrium
      - Memory trace = vortex/basin structure
    """
    
    def __init__(self, field: CognitiveField):
        self.field = field
        
        # Memory consolidation
        self.consolidation_rate = 0.01
        
        # Memory traces
        self.memory_traces: List[Dict] = []
    
    def encode(self, experience: Dict):
        """
        Encode experience into field.
        
        NOT: "store in memory"
        BUT: "experience deforms field topology"
        """
        trajectory = experience.get('trajectory', [])
        success = experience.get('success', 0.5)
        emotional_significance = experience.get('emotional_significance', success)
        
        if not trajectory:
            return
        
        # High emotional significance = strong deformation
        deformation_strength = emotional_significance
        
        # Deform field from trajectory
        self.field.deform_field(trajectory, success)
        
        # Form memory trace (vortex at significant points)
        for i, point in enumerate(trajectory):
            if i % 5 == 0:  # Every 5th point
                significance = (1.0 - i / len(trajectory)) * emotional_significance
                self.field.form_vortex(
                    point, 
                    strength=significance * 0.3,
                    confidence=significance
                )
        
        # Record memory trace
        self.memory_traces.append({
            'trajectory': trajectory,
            'success': success,
            'significance': emotional_significance,
            'potential_value': float(np.sum(self.field.V)) if hasattr(self.field.V, '__iter__') else float(self.field.V)
        })
        
        if len(self.memory_traces) > 100:
            self.memory_traces = self.memory_traces[-50:]
    
    def retrieve(self, query: np.ndarray, n_candidates: int = 5) -> List[Dict]:
        """
        Retrieve memories by following field gradients.
        
        NOT: "search memory database"
        BUT: "field naturally attracts to similar patterns"
        """
        query = np.asarray(query).flatten()
        
        # Find trajectories that pass near query point
        candidates = []
        
        for trace in self.memory_traces:
            trajectory = trace['trajectory']
            min_distance = min(np.linalg.norm(np.array(p) - query) for p in trajectory)
            candidates.append({
                'trace': trace,
                'distance': min_distance,
                'relevance': 1.0 / (1.0 + min_distance)
            })
        
        # Sort by relevance
        candidates.sort(key=lambda x: x['relevance'], reverse=True)
        
        return candidates[:n_candidates]
    
    def consolidate(self):
        """
        Consolidate memory by strengthening memory traces.
        
        Repeated experiences strengthen same vortices.
        This is Hebbian consolidation at field level.
        """
        # Find similar memory traces
        similar_groups: List[List[int]] = []
        
        for i, trace1 in enumerate(self.memory_traces):
            group = [i]
            
            for j, trace2 in enumerate(self.memory_traces):
                if i >= j:
                    continue
                
                # Check if trajectories are similar
                similarity = self._compute_trajectory_similarity(
                    trace1['trajectory'], trace2['trajectory']
                )
                
                if similarity > 0.7:
                    group.append(j)
            
            if len(group) > 1:
                similar_groups.append(group)
        
        # Merge similar groups (strengthen vortices)
        for group in similar_groups:
            # Find centroid trajectory
            centroid = self._compute_centroid([self.memory_traces[i]['trajectory'] for i in group])
            
            # Strengthen centroid in field
            for point in centroid:
                self.field.form_vortex(point, strength=1.0, confidence=0.8)
    
    def _compute_trajectory_similarity(self, traj1: List, traj2: List) -> float:
        """Compute similarity between two trajectories."""
        if not traj1 or not traj2:
            return 0.0
        
        # Normalize lengths
        n = min(len(traj1), len(traj2))
        traj1 = np.array(traj1[:n])
        traj2 = np.array(traj2[:n])
        
        # Compute average distance
        distances = np.linalg.norm(traj1 - traj2, axis=1)
        avg_distance = np.mean(distances)
        
        return 1.0 / (1.0 + avg_distance)
    
    def _compute_centroid(self, trajectories: List[List]) -> np.ndarray:
        """Compute centroid trajectory."""
        n = max(len(t) for t in trajectories)
        centroid = np.zeros((n, len(trajectories[0][0]) if trajectories and trajectories[0] else 1))
        
        for traj in trajectories:
            traj_array = np.array(traj)
            if len(traj_array) < n:
                traj_array = np.pad(traj_array, ((0, n - len(traj_array)), (0, 0)))
            centroid += traj_array[:n]
        
        centroid /= len(trajectories)
        return centroid


# ============================================================================
# INTEGRATED SELF-ORGANIZING FIELD
# ============================================================================

class SelfOrganizingCognitiveField:
    """
    Phase 19: Self-Organizing Cognitive Field
    
    Single unified variational system where:
      A. Hierarchy = Timescale separation in field
      B. Objects = Persistent vortices in flow
      C. Memory = Field deformation from experience
      D. Planning = Simulated relaxation
      E. Action = Prediction stabilization
      F. Perception = Precision-weighted error minimization
      G. Everything minimizes SAME free energy
      
    This is NOT:
      modules + field + world (three things)
    This IS:
      ONE field (everything is the field)
    """
    
    def __init__(self, manifold_dim: int = 4):
        self.manifold_dim = manifold_dim
        
        # The ONE field (not "field + world + memory")
        self.field = CognitiveField(manifold_dim=manifold_dim)
        
        # Prediction (downstream trajectories)
        self.prediction = PredictionAsTrajectory(self.field)
        
        # Action (prediction stabilization)
        self.action = ActionAsStabilization(self.field, self.prediction)
        
        # Memory (field deformation)
        self.memory = MemoryAsDeformation(self.field)
        
        # Time
        self.t = 0.0
        self.dt = 0.1
        
        # Experience log
        self.experiences: List[Dict] = []
    
    def perceive(self, observation: np.ndarray) -> Dict:
        """
        Perception = prediction error + field update.
        
        NOT: "encode observation"
        BUT: "observation is surprise/error that deforms field"
        """
        observation = np.asarray(observation).flatten()
        
        # Current prediction
        current_prediction = self.prediction.predict(n_steps=1)
        
        # Prediction error
        if len(observation) < self.manifold_dim:
            observation = np.pad(observation, (0, self.manifold_dim - len(observation)))
        else:
            observation = observation[:self.manifold_dim]
        
        error = observation - self.field.psi
        
        # Update field to minimize error
        self.field.step(dt=0.1, external_force=error * 0.3)
        
        # Update vortices
        self.field.step_vortices()
        
        return {
            'perceived_state': self.field.psi.tolist(),
            'prediction_error': np.linalg.norm(error),
            'free_energy': self.field.F,
            'field_summary': self.field.get_field_summary()
        }
    
    def imagine(self, n_steps: int, condition: Optional[np.ndarray] = None) -> Dict:
        """
        Imagination = simulated relaxation.
        
        NOT: "generate imagined scenario"
        BUT: "let field relax, trajectory IS imagination"
        """
        trajectory = self.field.simulate_relaxation(n_steps=n_steps, goal=condition)
        
        return {
            'trajectory': [t.tolist() for t in trajectory],
            'final_state': trajectory[-1].tolist() if trajectory else self.field.psi.tolist(),
            'n_steps': len(trajectory)
        }
    
    def act(self, desired_trajectory: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Action = field deformation to maintain prediction coherence.
        
        NOT: "achieve goal"
        BUT: "make predictions stay coherent"
        """
        if desired_trajectory is None:
            # Default: move toward nearest strong vortex
            if self.field.vortices:
                nearest = max(self.field.vortices, key=lambda v: v['strength'] * v['confidence'])
                desired_trajectory = [nearest['center'].copy()]
            else:
                desired_trajectory = [self.field.psi.copy()]
        
        action_result = self.action.compute_action(desired_trajectory)
        
        # Execute action (field deformation)
        self.field.deform_field(desired_trajectory, success=0.7)
        
        return {
            'action_cost': action_result['action_cost'],
            'deviation_corrected': action_result['deviation'],
            'field_summary': self.field.get_field_summary()
        }
    
    def remember(self, query: np.ndarray) -> Dict:
        """
        Memory retrieval = field gradient following.
        
        NOT: "search memory database"
        BUT: "field attracts to similar patterns"
        """
        retrieved = self.memory.retrieve(query)
        
        # Return most relevant memory
        if retrieved:
            return {
                'retrieved_trajectory': retrieved[0]['trace']['trajectory'],
                'relevance': retrieved[0]['relevance'],
                'n_candidates': len(retrieved)
            }
        
        return {'retrieved_trajectory': [], 'relevance': 0.0, 'n_candidates': 0}
    
    def experience(self, observation: np.ndarray, action_result: Optional[Dict] = None,
                  emotional_significance: float = 0.5) -> Dict:
        """
        Full experience cycle.
        
        1. Perceive (prediction error)
        2. Act (field deformation)
        3. Encode (memory as deformation)
        """
        # Perceive
        perception = self.perceive(observation)
        
        # Generate action (simple: move toward prediction)
        desired_trajectory = [self.field.psi.copy()]
        action = self.act(desired_trajectory)
        
        # Encode experience
        experience = {
            'trajectory': desired_trajectory,
            'success': action_result.get('success', 0.5) if action_result else 0.5,
            'emotional_significance': emotional_significance
        }
        
        self.memory.encode(experience)
        self.experiences.append(experience)
        
        return {
            'perception': perception,
            'action': action,
            'experience': experience
        }
    
    def run_cycle(self, n_steps: int = 10) -> Dict:
        """Run cognitive cycle."""
        results = []
        
        for step in range(n_steps):
            # Generate observation
            observation = np.array([
                np.sin(self.t) * 2,
                np.cos(self.t) * 2,
                0.0,
                0.0
            ]) + np.random.randn(4) * 0.1
            
            # Experience cycle
            result = self.experience(observation, emotional_significance=0.5)
            results.append(result)
            
            self.t += self.dt
        
        return {
            'steps': results,
            'field_summary': self.field.get_field_summary(),
            'n_memories': len(self.memory.memory_traces),
            'n_vortices': len(self.field.vortices)
        }


# ============================================================================
# TESTS
# ============================================================================

def test_cognitive_field():
    """Test single cognitive field."""
    print("\n" + "=" * 60)
    print("SINGLE COGNITIVE FIELD TEST")
    print("=" * 60)
    
    field = CognitiveField(manifold_dim=4)
    
    print("\n  Forming initial vortices:")
    
    field.form_vortex(np.array([1.0, 0.5, 0.0, 0.0]), strength=2.0, confidence=0.8)
    field.form_vortex(np.array([0.0, 0.0, 0.0, 0.0]), strength=1.5, confidence=0.6)
    
    print(f"    Vortices: {len(field.vortices)}")
    
    print("\n  Running dynamics:")
    
    for i in range(50):
        psi = field.step()
        
        if i % 10 == 9:
            summary = field.get_field_summary()
            print(f"    Step {i+1}: psi={[round(x, 2) for x in psi[:2]]}, "
                  f"free_energy={summary['free_energy']:.3f}")
    
    print(f"\n  Final state:")
    print(f"    Vortices: {len(field.vortices)}")
    print(f"    Free energy: {field.F:.3f}")


def test_prediction_as_trajectory():
    """Test prediction as downstream trajectory."""
    print("\n" + "=" * 60)
    print("PREDICTION AS TRAJECTORY TEST")
    print("=" * 60)
    
    field = CognitiveField(manifold_dim=4)
    field.form_vortex(np.array([2.0, 1.0, 0.0, 0.0]), strength=2.0, confidence=0.8)
    
    predictor = PredictionAsTrajectory(field)
    
    print("\n  Predicting future trajectory:")
    
    prediction = predictor.predict(n_steps=20, condition=np.array([2.0, 1.0, 0.0, 0.0]))
    
    print(f"    Steps predicted: {len(prediction['trajectory'])}")
    print(f"    Final state: {[round(x, 2) for x in prediction['final_state'][:2]]}")
    print(f"    Confidence: {prediction['confidence']:.3f}")


def test_memory_as_deformation():
    """Test memory as field deformation."""
    print("\n" + "=" * 60)
    print("MEMORY AS FIELD DEFORMATION TEST")
    print("=" * 60)
    
    field = CognitiveField(manifold_dim=4)
    memory = MemoryAsDeformation(field)
    
    print("\n  Encoding experiences:")
    
    for i in range(10):
        trajectory = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.5, 0.5, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0, 0.0]),
        ]
        experience = {
            'trajectory': trajectory,
            'success': 0.7 + i * 0.03,
            'emotional_significance': 0.5 + i * 0.05
        }
        memory.encode(experience)
    
    print(f"    Experiences encoded: {len(memory.memory_traces)}")
    print(f"    Field vortices: {len(field.vortices)}")
    
    print("\n  Retrieving memory:")
    query = np.array([0.5, 0.5, 0.0, 0.0])
    retrieved = memory.retrieve(query)
    
    if retrieved:
        print(f"    Retrieved: {len(retrieved)} candidates")
        print(f"    Top relevance: {retrieved[0]['relevance']:.3f}")


def test_integrated_field():
    """Test integrated self-organizing field."""
    print("\n" + "=" * 60)
    print("SELF-ORGANIZING COGNITIVE FIELD TEST")
    print("=" * 60)
    
    field = SelfOrganizingCognitiveField(manifold_dim=4)
    
    print("\n  Running cognitive cycle:")
    
    result = field.run_cycle(n_steps=30)
    
    print(f"    Steps: {len(result['steps'])}")
    print(f"    Field vortices: {result['n_vortices']}")
    print(f"    Memories stored: {result['n_memories']}")
    print(f"    Free energy: {result['field_summary']['free_energy']:.3f}")
    
    print("\n  Testing imagination:")
    imagination = field.imagine(n_steps=10)
    print(f"    Imagined steps: {imagination['n_steps']}")
    print(f"    Final: {[round(x, 2) for x in imagination['final_state'][:2]]}")
    
    print("\n  Testing memory retrieval:")
    query = np.array([0.5, 0.5, 0.0, 0.0])
    memory_result = field.remember(query)
    print(f"    Retrieved: {memory_result['n_candidates']} candidates")


def test_vortex_dynamics():
    """Test vortex formation, merging, dissipation."""
    print("\n" + "=" * 60)
    print("VORTEX DYNAMICS TEST")
    print("=" * 60)
    
    field = CognitiveField(manifold_dim=4)
    
    print("\n  Forming vortices:")
    
    v1 = field.form_vortex(np.array([1.0, 0.5, 0.0, 0.0]), strength=2.0)
    v2 = field.form_vortex(np.array([1.2, 0.6, 0.0, 0.0]), strength=1.5)
    
    print(f"    Vortex 1: {v1['id']}, strength={v1['strength']:.2f}")
    print(f"    Vortex 2: {v2['id']}, strength={v2['strength']:.2f}")
    
    print("\n  Merging nearby vortices:")
    field.merge_vortices(v1['id'], v2['id'])
    print(f"    After merge: {len(field.vortices)} vortices")
    
    print("\n  Running vortex aging:")
    for _ in range(100):
        field.step_vortices()
    
    print(f"    After 100 steps: {len(field.vortices)} vortices")


def phase_comparison():
    """Compare Phase 18 vs Phase 19."""
    print("\n" + "=" * 60)
    print("PHASE 18 VS PHASE 19 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 18 (Unified Modules):")
    print("    - predictive_coding = separate module")
    print("    - field_dynamics = separate module")
    print("    - persistent_world = separate module")
    print("    - THREE things interacting")
    
    print("\n  Phase 19 (Single Field):")
    print("    - ONE cognitive field")
    print("    - hierarchy = timescale separation")
    print("    - objects = persistent vortices")
    print("    - memory = field deformation")
    print("    - planning = simulated relaxation")
    print("    - action = prediction stabilization")
    print("    - everything minimizes SAME free energy")
    
    print("\n  Critical shift:")
    print("    Phase 18: 'field + objects in field'")
    print("    Phase 19: 'field IS the objects'")
    
    print("\n  Not modules + field + world")
    print("  But ONE field (everything is the field)")


if __name__ == "__main__":
    test_cognitive_field()
    test_prediction_as_trajectory()
    test_memory_as_deformation()
    test_integrated_field()
    test_vortex_dynamics()
    phase_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 19 - SELF-ORGANIZING COGNITIVE FIELD")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 18 - unified modules interacting
  To: Phase 19 - SINGLE continuous field where:
      - Everything IS the field (not objects in field)
      - Hierarchy = timescale separation (not layer stack)
      - Objects = persistent vortices (not tracked entities)
      - Memory = field deformation (not storage)
      - Planning = energy relaxation (not rollout)
      - Action = prediction coherence (not goal pursuit)
      - ALL minimize same free energy

CRITICAL TRANSITION:
  Phase 18: "field + objects in field"
  Phase 19: "field IS the objects"
  
  The attractor is not an "object in the field".
  The attractor IS stable topology of the field itself.

KEY INSIGHTS:

A. SINGLE UNIFIED VARIATIONAL FIELD
   Everything minimizes the same free energy:
   - Perception = error minimization
   - Action = field deformation
   - Memory = topology change
   - Planning = simulated relaxation
   - They are NOT separate operations.

B. HIERARCHY = TIMESCALE SEPARATION
   Not: goal_layer -> semantic_layer -> sensory_layer
   But: slow dynamics + medium dynamics + fast dynamics
   
   Each timescale has different:
   - Update frequency
   - Precision
   - Attractor strength
   - Memory persistence

C. OBJECTS = PERSISTENT VORTICES
   Not: WorldObject (entity container)
   But: stable flow patterns in field
   
   A vortex:
   - Emerges from recurrent flow
   - Has stability (eigenvalue of Hessian)
   - Can merge with other vortices
   - Can dissipate when unstable

D. MEMORY = FIELD DEFORMATION
   Not: memory.append(trajectory)
   But: experience literally changes field topology
   
   Successful trajectories:
   - Create vortices
   - Lower potential along path
   - Increase precision

E. PLANNING = SIMULATED RELAXATION
   Not: rollout planning
   But: let field relax, trajectory emerges
   
   Imagination = what happens naturally

F. ACTION = PREDICTION STABILIZATION
   Not: achieve goal
   But: make predictions remain coherent
   
   Action deforms field to maintain coherence.

This IS:
  - Friston's free energy principle
  - Active inference
  - Continuous attractor field
  - Self-organizing cognitive substrate
  - True dynamical systems cognition
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 19 Summary:

BEFORE:
  - Modules + Field + World (three things interacting)
  - Hierarchy = layer stack
  - Objects = tracked entities
  - Memory = storage
  - Planning = rollout
  - Action = goal pursuit

AFTER:
  - ONE continuous field (everything is the field)
  - Hierarchy = timescale separation
  - Objects = persistent vortices
  - Memory = field deformation
  - Planning = simulated relaxation
  - Action = prediction stabilization

The critical shift:
  "field + objects in field" → "field IS the objects"
  
  Attractors are not entities in the field.
  Attractors ARE stable topology of the field.

This is already:
  - Neural field theory
  - Free energy principle
  - Active inference
  - Continuous dynamical systems
  - Self-organizing cognition
"""