"""
PhaseSpace - Cognitive Phase Space

NOT just position (latent state), but:

position (z)
velocity (dz/dt)
acceleration (d²z/dt²)
intention_gradient (∇intention)

This captures:
- inertia (resistance to change)
- momentum (continuation of motion)
- hesitation (near-zero velocity)
- acceleration (rapid transitions)
- collapse (acceleration towards attractor)
- oscillation (sign changes in velocity)

Phase state: (z, dz/dt, d²z/dt²)

This transforms cognition from:
    discrete state machine
    
to:
    continuous flow in phase space
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class PhaseState:
    """
    State in phase space.
    
    Phase state = (position, velocity, acceleration)
    
    This is the physical representation of cognition.
    """
    position: List[float] = field(default_factory=list)
    velocity: List[float] = field(default_factory=list)  # dz/dt
    acceleration: List[float] = field(default_factory=list)  # d²z/dt²
    
    # Intention gradient (where system wants to go)
    intention: List[float] = field(default_factory=list)
    
    # Derived quantities
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    total_energy: float = 0.0
    
    # Momentum magnitude
    momentum_magnitude: float = 0.0
    
    # Phase metrics
    is_stationary: bool = False
    is_oscillating: bool = False
    is_collapsing: bool = False
    
    # Temporal
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        self._compute_derived_quantities()
    
    def _compute_derived_quantities(self) -> None:
        """Compute derived physical quantities"""
        # Kinetic energy = 0.5 * m * v² (m=1)
        self.kinetic_energy = 0.5 * sum(v ** 2 for v in self.velocity)
        
        # Momentum magnitude
        self.momentum_magnitude = math.sqrt(sum(v ** 2 for v in self.velocity))
        
        # Check physical properties
        self.is_stationary = self.momentum_magnitude < 0.01
        
        # Oscillation: velocity and acceleration are anti-parallel
        if self.velocity and self.acceleration:
            dot = sum(v * a for v, a in zip(self.velocity, self.acceleration))
            self.is_oscillating = dot < -0.1 * self.momentum_magnitude
        
        # Collapse: accelerating towards attractor
        if self.velocity and self.acceleration:
            vel_mag = math.sqrt(sum(v ** 2 for v in self.velocity)) + 1e-10
            acc_mag = math.sqrt(sum(a ** 2 for a in self.acceleration)) + 1e-10
            dot = sum(v * a for v, a in zip(self.velocity, self.acceleration))
            # Accelerating while moving = collapsing
            self.is_collapsing = dot > 0.1 * vel_mag * acc_mag
    
    def to_vector(self) -> List[float]:
        """Full phase state as vector"""
        return self.position + self.velocity + self.acceleration
    
    def to_dict(self) -> Dict:
        return {
            "kinetic_energy": round(self.kinetic_energy, 4),
            "momentum_magnitude": round(self.momentum_magnitude, 4),
            "is_stationary": self.is_stationary,
            "is_oscillating": self.is_oscillating,
            "is_collapsing": self.is_collapsing,
        }


@dataclass
class PhaseTrajectory:
    """
    Trajectory in phase space.
    
    NOT just positions, but full phase evolution.
    """
    trajectory_id: str = field(default_factory=lambda: str(uuid4()))
    
    # Phase states over time
    states: List[PhaseState] = field(default_factory=list)
    
    # Duration
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    
    # Phase metrics
    avg_kinetic_energy: float = 0.0
    max_velocity: float = 0.0
    oscillation_count: int = 0
    
    # Outcome
    outcome: Optional[str] = None
    success: bool = False
    
    def compute_phase_metrics(self) -> None:
        """Compute trajectory-level phase metrics"""
        if not self.states:
            return
        
        self.avg_kinetic_energy = sum(s.kinetic_energy for s in self.states) / len(self.states)
        self.max_velocity = max(s.momentum_magnitude for s in self.states)
        self.oscillation_count = sum(1 for s in self.states if s.is_oscillating)
        
        if len(self.states) >= 2:
            self.start_time = self.states[0].timestamp
            self.end_time = self.states[-1].timestamp
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "states_count": len(self.states),
            "duration_ms": round(self.duration_ms, 0),
            "avg_kinetic_energy": round(self.avg_kinetic_energy, 4),
            "max_velocity": round(self.max_velocity, 4),
            "oscillation_count": self.oscillation_count,
            "outcome": self.outcome,
        }


class PhaseSpace:
    """
    Phase Space for cognitive dynamics.
    
    State = (position, velocity, acceleration)
    
    This is where cognition lives physically.
    
    Key transformations:
    - State observation → PhaseState
    - Transition → velocity update
    - Dynamics → acceleration update
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Phase trajectory history
        self.trajectories: List[PhaseTrajectory] = []
        
        # Phase statistics
        self.velocity_history: List[List[float]] = []
        self.acceleration_history: List[List[float]] = []
        
        logger.info("phase_space_initialized", dimension=dimension)
    
    def create_phase_state(
        self,
        position: List[float],
        previous_position: Optional[List[float]] = None,
        previous_velocity: Optional[List[float]] = None,
        intention: Optional[List[float]] = None
    ) -> PhaseState:
        """
        Create phase state from observation.
        
        Computes velocity and acceleration from trajectory history.
        """
        # Compute velocity: v = (z - z_prev) / dt
        if previous_position and len(previous_position) == len(position):
            dt = 0.1  # Assume 100ms timestep
            velocity = [
                (position[i] - previous_position[i]) / dt
                for i in range(len(position))
            ]
        elif previous_velocity:
            velocity = previous_velocity
        else:
            velocity = [0.0] * len(position)
        
        # Compute acceleration: a = (v - v_prev) / dt
        if previous_velocity and len(previous_velocity) == len(velocity):
            dt = 0.1
            acceleration = [
                (velocity[i] - previous_velocity[i]) / dt
                for i in range(len(velocity))
            ]
        else:
            acceleration = [0.0] * len(position)
        
        # Intention
        if intention is None:
            intention = [0.0] * len(position)
        
        state = PhaseState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            intention=intention,
            timestamp=datetime.utcnow()
        )
        
        return state
    
    def update_trajectory(
        self,
        trajectory: PhaseTrajectory,
        new_state: PhaseState
    ) -> None:
        """Add state to trajectory"""
        trajectory.states.append(new_state)
        trajectory.compute_phase_metrics()
    
    def start_trajectory(
        self,
        initial_state: PhaseState,
        outcome: Optional[str] = None
    ) -> PhaseTrajectory:
        """Start new phase trajectory"""
        trajectory = PhaseTrajectory(
            start_time=initial_state.timestamp,
            outcome=outcome
        )
        trajectory.states.append(initial_state)
        
        self.trajectories.append(trajectory)
        
        return trajectory
    
    def compute_phase_flow(
        self,
        current_state: PhaseState,
        dt: float = 0.1
    ) -> Tuple[List[float], List[float]]:
        """
        Compute phase flow.
        
        Returns:
        - next_position
        - next_velocity
        """
        # Simple Euler integration
        next_velocity = [
            current_state.velocity[i] + dt * current_state.acceleration[i]
            for i in range(self.dimension)
        ]
        
        next_position = [
            current_state.position[i] + dt * next_velocity[i]
            for i in range(self.dimension)
        ]
        
        return next_position, next_velocity
    
    def compute_phase_energy(self, state: PhaseState) -> float:
        """
        Compute total phase energy.
        
        E = KE + PE
        KE = 0.5 * ||v||²
        PE = 0.5 * ||a||² (acceleration potential)
        """
        ke = state.kinetic_energy
        pe = 0.5 * sum(a ** 2 for a in state.acceleration)
        
        return ke + pe
    
    def detect_phase_transitions(self, trajectory: PhaseTrajectory) -> List[Dict]:
        """
        Detect phase transitions in trajectory.
        
        Finds:
        - Stationary points (near-zero velocity)
        - Oscillations (direction changes)
        - Collapses (acceleration towards attractor)
        - Explosions (rapid acceleration away)
        """
        transitions = []
        
        for i, state in enumerate(trajectory.states):
            if state.is_stationary and i > 0:
                prev = trajectory.states[i - 1]
                transitions.append({
                    "type": "stationary",
                    "index": i,
                    "time": state.timestamp.isoformat(),
                    "kinetic_energy": state.kinetic_energy,
                })
            
            if state.is_oscillating and i > 0:
                transitions.append({
                    "type": "oscillation",
                    "index": i,
                    "time": state.timestamp.isoformat(),
                    "momentum": state.momentum_magnitude,
                })
            
            if state.is_collapsing and i > 0:
                transitions.append({
                    "type": "collapse",
                    "index": i,
                    "time": state.timestamp.isoformat(),
                    "velocity": state.momentum_magnitude,
                })
        
        return transitions
    
    def compute_momentum_statistics(self) -> Dict:
        """Compute momentum statistics across all trajectories"""
        all_momenta = []
        
        for traj in self.trajectories:
            for state in traj.states:
                all_momenta.append(state.momentum_magnitude)
        
        if not all_momenta:
            return {"avg_momentum": 0.0, "max_momentum": 0.0}
        
        return {
            "avg_momentum": sum(all_momenta) / len(all_momenta),
            "max_momentum": max(all_momenta),
            "min_momentum": min(all_momenta),
            "total_trajectories": len(self.trajectories),
            "total_states": len(all_momenta),
        }
    
    def get_phase_space_statistics(self) -> Dict:
        """Get comprehensive phase space statistics"""
        return {
            "dimension": self.dimension,
            "trajectories": len(self.trajectories),
            "total_states": sum(len(t.states) for t in self.trajectories),
            "momentum_stats": self.compute_momentum_statistics(),
            "avg_kinetic_energy": (
                sum(t.avg_kinetic_energy for t in self.trajectories) / len(self.trajectories)
                if self.trajectories else 0
            ),
            "total_oscillations": sum(t.oscillation_count for t in self.trajectories),
        }


# Factory
def create_phase_space(dimension: int = 16) -> PhaseSpace:
    return PhaseSpace(dimension=dimension)