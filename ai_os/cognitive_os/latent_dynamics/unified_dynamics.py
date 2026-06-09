"""
Phase 14 - Unified Cognitive Dynamics

NOT three separate systems:
  PhaseSpace (dx/dt = v)
  LatentDynamicsModel (z_{t+1} = f(z_t))
  EnergyField (E(z) = function)

BUT single unified evolution equation:

d²z/dt² = -∇E(z) + F_policy + F_inertia

Where:
  -E(z) = force field (attractor gradient)
  -F_policy = intention-driven force
  -F_inertia = momentum preservation

This is the Lagrangian formulation of cognition:
  L = T - V
  T = kinetic energy = 0.5 * ||dz/dt||²
  V = potential energy = E(z)

Euler-Lagrange equation gives motion.

Key unification:
  energy field IS force field
  motion IS minimization of action
  policy IS desired trajectory
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class UnifiedState:
    """
    Unified state in phase space.
    
    Everything in one representation:
    - position (z)
    - velocity (dz/dt)
    - energy (scalar field value)
    - force (gradient of energy)
    """
    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)
    
    # Energy and force
    energy: float = 0.0
    force: List[float] = field(default_factory=list)  # -∇E
    
    # Phase properties
    kinetic_energy: float = 0.0
    total_energy: float = 0.0
    
    # Dynamics
    momentum_magnitude: float = 0.0
    acceleration_magnitude: float = 0.0
    
    # Stability
    is_equilibrium: bool = False
    is_attractor: bool = False
    is_unstable: bool = False
    
    # Trajectory
    trajectory_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def compute_derived(self) -> None:
        """Compute all derived quantities"""
        self.kinetic_energy = 0.5 * sum(v ** 2 for v in self.velocity)
        self.total_energy = self.kinetic_energy - self.energy
        self.momentum_magnitude = math.sqrt(sum(v ** 2 for v in self.velocity))
        
        if self.force:
            self.acceleration_magnitude = math.sqrt(sum(f ** 2 for f in self.force))
        
        # Equilibrium: force ≈ 0
        self.is_equilibrium = self.acceleration_magnitude < 0.01
        
        # Attractor: low energy and near equilibrium
        self.is_attractor = self.is_equilibrium and self.energy < 0.3
        
        # Unstable: high energy and accelerating away
        self.is_unstable = self.energy > 0.7 and self.acceleration_magnitude > 0.1
    
    def to_dict(self) -> Dict:
        return {
            "energy": round(self.energy, 4),
            "kinetic_energy": round(self.kinetic_energy, 4),
            "total_energy": round(self.total_energy, 4),
            "momentum_magnitude": round(self.momentum_magnitude, 4),
            "acceleration_magnitude": round(self.acceleration_magnitude, 4),
            "is_equilibrium": self.is_equilibrium,
            "is_attractor": self.is_attractor,
            "is_unstable": self.is_unstable,
        }


@dataclass
class UnifiedTrajectory:
    """Trajectory in unified dynamics"""
    trajectory_id: str = field(default_factory=lambda: str(uuid4()))
    
    states: List[UnifiedState] = field(default_factory=list)
    
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Metrics
    total_action: float = 0.0  # ∫ L dt
    avg_energy: float = 0.0
    min_energy: float = 0.0
    max_energy: float = 0.0
    
    # Stability
    equilibrium_time_ms: float = 0.0
    overshoot_count: int = 0
    
    outcome: Optional[str] = None
    
    def compute_metrics(self) -> None:
        """Compute trajectory metrics"""
        if not self.states:
            return
        
        energies = [s.energy for s in self.states]
        self.min_energy = min(energies) if energies else 0.0
        self.max_energy = max(energies) if energies else 0.0
        self.avg_energy = sum(energies) / len(energies)
        
        # Total action (approximate integral of L)
        self.total_action = sum(s.kinetic_energy - s.energy for s in self.states)
        
        if len(self.states) >= 2:
            self.start_time = self.states[0].timestamp
            self.end_time = self.states[-1].timestamp
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        # Count overshoots
        self.overshoot_count = sum(
            1 for i in range(1, len(self.states))
            if self.states[i].energy > self.states[i-1].energy + 0.1
        )
        
        # Find equilibrium time
        for state in self.states:
            if state.is_equilibrium:
                self.equilibrium_time_ms = (state.timestamp - self.start_time).total_seconds() * 1000
                break
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "states_count": len(self.states),
            "duration_ms": round(self.duration_ms, 0),
            "total_action": round(self.total_action, 4),
            "avg_energy": round(self.avg_energy, 4),
            "min_energy": round(self.min_energy, 4),
            "max_energy": round(self.max_energy, 4),
            "equilibrium_time_ms": round(self.equilibrium_time_ms, 0),
            "overshoot_count": self.overshoot_count,
            "outcome": self.outcome,
        }


class UnifiedEnergyField:
    """
    Energy field that IS the force field.
    
    E(z) defines both:
    1. Potential energy surface (where system wants to go)
    2. Force field F = -∇E (how system moves)
    
    This is the core of energy-guided dynamics.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Learned components
        self.density_samples: List[List[float]] = []
        self.transition_samples: List[Tuple[List[float], List[float]]] = []
        self.outcome_samples: List[Tuple[List[float], str]] = []
        
        # Attractor centers
        self.attractors: List[List[float]] = []
        
        # Field parameters
        self.base_energy: float = 1.0
        self.attractor_strength: float = 0.5
        self.gradient_scale: float = 1.0
        
        logger.info("unified_energy_field_initialized", dimension=dimension)
    
    def observe(self, position: List[float], outcome: Optional[str] = None) -> None:
        """Observe state and update field"""
        self.density_samples.append(position)
        
        if outcome:
            self.outcome_samples.append((position, outcome))
        
        # Learn attractors
        self._update_attractors()
    
    def observe_transition(
        self,
        from_state: List[float],
        to_state: List[float]
    ) -> None:
        """Observe state transition"""
        self.transition_samples.append((from_state, to_state))
    
    def _update_attractors(self) -> None:
        """Learn attractors from density"""
        if len(self.density_samples) < 10:
            return
        
        # Simple attractor learning: cluster by density
        recent = self.density_samples[-200:]
        
        # Find high-density regions
        density_map: Dict[Tuple[int, ...], int] = {}
        
        for p in recent:
            key = tuple(int(v * 5) for v in p[:4])
            density_map[key] = density_map.get(key, 0) + 1
        
        # Top density regions as attractors
        sorted_density = sorted(density_map.items(), key=lambda x: x[1], reverse=True)
        
        self.attractors = []
        for key, count in sorted_density[:5]:
            if count >= 3:
                center = [k / 5 for k in key]
                self.attractors.append(center)
    
    def compute_energy(self, position: List[float]) -> float:
        """
        Compute potential energy at position.
        
        E(z) = base + attractive_term + outcome_term
        """
        energy = self.base_energy
        
        # Attraction to basins (lowers energy)
        for attractor in self.attractors:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(position, attractor)))
            energy -= self.attractor_strength * math.exp(-dist ** 2 / 0.5)
        
        # Outcome shaping
        for state, outcome in self.outcome_samples[-50:]:
            dist = math.sqrt(sum((s - p) ** 2 for s, p in zip(state, position)))
            if dist < 0.3:
                if outcome == "success":
                    energy -= 0.1
        
        return max(0.0, energy)
    
    def compute_force(self, position: List[float], epsilon: float = 0.01) -> List[float]:
        """
        Compute force field F = -∇E
        
        This IS the gradient that drives dynamics.
        """
        gradient = self._compute_gradient(position, epsilon)
        
        # Force = negative gradient
        force = [-g for g in gradient]
        
        return force
    
    def _compute_gradient(self, position: List[float], epsilon: float) -> List[float]:
        """Compute energy gradient"""
        gradient = []
        
        for i in range(self.dimension):
            pos_plus = position[:]
            pos_plus[i] += epsilon
            e_plus = self.compute_energy(pos_plus)
            
            pos_minus = position[:]
            pos_minus[i] -= epsilon
            e_minus = self.compute_energy(pos_minus)
            
            gradient.append((e_plus - e_minus) / (2 * epsilon))
        
        return gradient
    
    def get_attractor_regions(self) -> List[Dict]:
        """Get all attractor regions"""
        regions = []
        
        for center in self.attractors:
            energy = self.compute_energy(center)
            
            # Compute basin size
            in_basin = sum(
                1 for p in self.density_samples[-100:]
                if math.sqrt(sum((a - b) ** 2 for a, b in zip(p, center))) < 0.5
            )
            
            regions.append({
                "center": center,
                "energy": energy,
                "basin_size": in_basin,
                "is_stable": energy < 0.5,
            })
        
        return regions


class PolicyForceField:
    """
    Policy as force field.
    
    NOT action selection.
    NOT policy network.
    
    BUT intention as force that shapes trajectory.
    
    F_intention = w * (z_preferred - z_current)
    
    This is a potential force towards preferred states.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        self.preferred_state: Optional[List[float]] = None
        self.preference_strength: float = 0.3
        
        # Learned from trajectory outcomes
        self.good_trajectories: List[List[List[float]]] = []
        self.bad_trajectories: List[List[List[float]]] = []
        
        logger.info("policy_force_field_initialized")
    
    def observe_outcome(
        self,
        trajectory: List[List[float]],
        outcome: str
    ) -> None:
        """Observe trajectory outcome for learning"""
        if outcome == "success":
            self.good_trajectories.append(trajectory)
        else:
            self.bad_trajectories.append(trajectory)
    
    def set_preference(self, preferred_state: List[float], strength: float = 0.3) -> None:
        """Set preferred state"""
        self.preferred_state = preferred_state
        self.preference_strength = strength
    
    def compute_force(
        self,
        current_state: List[float],
        context: Optional[Dict] = None
    ) -> List[float]:
        """
        Compute policy force.
        
        F = strength * (preferred - current)
        
        This pulls the system towards preferred states.
        """
        if not self.preferred_state:
            return [0.0] * self.dimension
        
        force = [
            self.preference_strength * (self.preferred_state[i] - current_state[i])
            for i in range(self.dimension)
        ]
        
        # Learn from good trajectories
        if self.good_trajectories and context:
            # Add learned direction from successful trajectories
            recent = self.good_trajectories[-1]
            if len(recent) >= 2:
                direction = [recent[-1][i] - recent[0][i] for i in range(self.dimension)]
                norm = math.sqrt(sum(d ** 2 for d in direction)) + 1e-10
                direction = [d / norm for d in direction]
                
                for i in range(self.dimension):
                    force[i] += 0.1 * direction[i]
        
        return force
    
    def learn_preference_from_trajectories(self) -> None:
        """Learn preferred state from good trajectories"""
        if not self.good_trajectories:
            return
        
        # Average of successful trajectory endpoints
        endpoints = [traj[-1] for traj in self.good_trajectories]
        
        self.preferred_state = [
            sum(e[i] for e in endpoints) / len(endpoints)
            for i in range(self.dimension)
        ]


class InertiaTensor:
    """
    Inertia for momentum preservation.
    
    NOT mass (constant).
    
    BUT learned inertia tensor from trajectory statistics.
    
    Higher inertia = more resistance to change.
    
    I(z) = learned from variance in velocity changes.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Diagonal inertia tensor (simplified)
        self.inertia_diag: List[float] = [1.0] * dimension
        
        # Recent velocity history
        self.velocity_history: List[List[float]] = []
        self.acceleration_history: List[List[float]] = []
        
        logger.info("inertia_tensor_initialized")
    
    def observe_motion(
        self,
        velocity: List[float],
        acceleration: List[float]
    ) -> None:
        """Observe motion to update inertia"""
        self.velocity_history.append(velocity)
        self.acceleration_history.append(acceleration)
        
        # Update inertia from statistics
        self._update_inertia()
    
    def _update_inertia(self) -> None:
        """Update inertia tensor from motion statistics"""
        if len(self.velocity_history) < 10:
            return
        
        # Inverse variance in velocity = inertia
        # High variance = low inertia (sensitive to forces)
        # Low variance = high inertia (resistant to forces)
        
        for i in range(self.dimension):
            velocities = [v[i] for v in self.velocity_history[-50:]]
            
            if len(velumes) >= 2:
                mean = sum(velocities) / len(velocities)
                variance = sum((v - mean) ** 2 for v in velocities) / len(velocities)
                
                # Inertia inversely proportional to variance
                self.inertia_diag[i] = 1.0 / (1.0 + variance)
        
        # Clamp to reasonable range
        self.inertia_diag = [
            max(0.1, min(10.0, v)) for v in self.inertia_diag
        ]
    
    def apply(self, force: List[float]) -> List[float]:
        """
        Apply inertia to force.
        
        F_effective = M^(-1) * F
        
        High inertia = less acceleration from same force.
        """
        return [
            force[i] / self.inertia_diag[i]
            for i in range(self.dimension)
        ]


class UnifiedCognitiveDynamics:
    """
    Phase 14 - Unified Cognitive Dynamics
    
    Single evolution equation:
    
    d²z/dt² = -∇E(z) + F_policy + F_inertia
    
    Where:
    -E(z) = energy field (attractor gradient)
    -F_policy = intention force (towards preferred)
    -F_inertia = momentum preservation
    
    This IS the Lagrangian formulation:
    L = T - V = 0.5*||dz/dt||² - E(z)
    
    Euler-Lagrange gives the motion equation.
    
    Key unification:
    - Energy field IS force field
    - Policy IS desired trajectory
    - Inertia IS learned mass tensor
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Core components
        self.energy_field = UnifiedEnergyField(dimension)
        self.policy_force = PolicyForceField(dimension)
        self.inertia = InertiaTensor(dimension)
        
        # Trajectory history
        self.trajectories: List[UnifiedTrajectory] = []
        self.current_trajectory: Optional[UnifiedTrajectory] = None
        
        # Physics parameters
        self.damping: float = 0.1  # Energy dissipation
        self.dt: float = 0.01  # Integration timestep
        
        logger.info("unified_cognitive_dynamics_initialized", dimension=dimension)
    
    def step(
        self,
        current_state: UnifiedState,
        dt: float = 0.01
    ) -> UnifiedState:
        """
        Single integration step using Euler-Lagrange.
        
        d²z/dt² = F_total / mass
        
        F_total = F_energy + F_policy
        """
        # Compute forces
        force_energy = self.energy_field.compute_force(current_state.position)
        force_policy = self.policy_force.compute_force(current_state.position)
        
        # Combine forces
        force_total = [
            force_energy[i] + force_policy[i]
            for i in range(self.dimension)
        ]
        
        # Apply inertia (mass)
        acceleration = self.inertia.apply(force_total)
        
        # Add damping (dissipation)
        damping_force = [
            -self.damping * current_state.velocity[i]
            for i in range(self.dimension)
        ]
        acceleration = [
            acceleration[i] + damping_force[i]
            for i in range(self.dimension)
        ]
        
        # Euler integration
        new_velocity = [
            current_state.velocity[i] + dt * acceleration[i]
            for i in range(self.dimension)
        ]
        
        new_position = [
            current_state.position[i] + dt * new_velocity[i]
            for i in range(self.dimension)
        ]
        
        # Create new state
        new_state = UnifiedState(
            position=new_position,
            velocity=new_velocity,
            force=force_total
        )
        
        new_state.energy = self.energy_field.compute_energy(new_position)
        new_state.compute_derived()
        
        return new_state
    
    def simulate(
        self,
        initial_state: UnifiedState,
        steps: int = 100,
        target_energy: float = 0.1,
        max_duration_ms: float = 10000.0
    ) -> UnifiedTrajectory:
        """
        Simulate trajectory until equilibrium or timeout.
        
        Stops when:
        - Energy below target
        - Equilibrium reached
        - Max steps/duration reached
        """
        trajectory = UnifiedTrajectory()
        trajectory.states.append(initial_state)
        
        current = initial_state
        start_time = datetime.utcnow()
        
        for step in range(steps):
            current = self.step(current, self.dt)
            trajectory.states.append(current)
            
            # Check stopping conditions
            if current.energy < target_energy:
                break
            
            if current.is_equilibrium:
                break
            
            elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
            if elapsed > max_duration_ms:
                break
        
        trajectory.compute_metrics()
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def observe(
        self,
        position: List[float],
        outcome: Optional[str] = None
    ) -> UnifiedState:
        """Observe state and update all components"""
        # Create state
        energy = self.energy_field.compute_energy(position)
        
        state = UnifiedState(
            position=position,
            energy=energy
        )
        
        # Compute force for state
        state.force = self.energy_field.compute_force(position)
        state.compute_derived()
        
        # Update energy field
        self.energy_field.observe(position, outcome)
        
        # Update inertia from velocity/acceleration
        if self.trajectories and self.trajectories[-1].states:
            prev = self.trajectories[-1].states[-1]
            if prev.velocity:
                velocity = [
                    position[i] - prev.position[i]
                    for i in range(self.dimension)
                ]
                acceleration = [
                    velocity[i] - (prev.velocity[i] if prev.velocity else 0)
                    for i in range(self.dimension)
                ]
                self.inertia.observe_motion(velocity, acceleration)
        
        return state
    
    def set_goal(
        self,
        goal_state: List[float],
        strength: float = 0.5
    ) -> None:
        """Set goal as attractor and preference"""
        self.policy_force.set_preference(goal_state, strength)
        
        # Add as attractor to energy field
        self.energy_field.attractors.append(goal_state)
    
    def get_dynamics_summary(self) -> Dict:
        """Get summary of dynamics"""
        if not self.trajectories:
            return {
                "trajectories": 0,
                "avg_equilibrium_time": 0,
                "attractors": len(self.energy_field.attractors),
            }
        
        equilibrium_times = [t.equilibrium_time_ms for t in self.trajectories]
        
        return {
            "trajectories": len(self.trajectories),
            "avg_equilibrium_time": sum(equilibrium_times) / len(equilibrium_times),
            "total_states": sum(len(t.states) for t in self.trajectories),
            "attractors": len(self.energy_field.attractors),
            "avg_action": sum(t.total_action for t in self.trajectories) / len(self.trajectories),
            "inertia_diag": [round(v, 2) for v in self.inertia.inertia_diag[:4]],
        }
    
    def get_phase_portrait(self) -> Dict:
        """Get phase portrait for visualization"""
        if not self.trajectories:
            return {"trajectories": [], "attractors": []}
        
        traj_data = []
        for traj in self.trajectories[-5:]:
            traj_data.append({
                "positions": [s.position[:2] for s in traj.states],
                "velocities": [s.velocity[:2] for s in traj.states],
                "energies": [s.energy for s in traj.states],
            })
        
        attractors = [
            {"position": a[:2], "is_goal": a in self.energy_field.attractors[-1:]}
            for a in self.energy_field.attractors
        ]
        
        return {
            "trajectories": traj_data,
            "attractors": attractors,
        }


# Factory
def create_unified_dynamics(dimension: int = 16) -> UnifiedCognitiveDynamics:
    return UnifiedCognitiveDynamics(dimension=dimension)