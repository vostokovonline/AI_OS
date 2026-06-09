"""
Trajectory Tracking - History and projection of cognitive state.

Key principle:
- Trajectory is the PATTERN of state changes over time
- NOT just a sequence of events
- Enables prediction of future states

Trajectory vs History:
- History: what happened (event log)
- Trajectory: where are we going (pattern analysis)

Components:
1. StateVector: snapshot of current state
2. TrajectorySegment: sequence of states with context
3. TrendAnalyzer: detect patterns in trajectory
4. ProjectionEngine: predict future states
"""

from typing import Dict, Any, Optional, List, Tuple, Set, FrozenSet
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib


class TrendDirection(Enum):
    """Direction of trajectory trend"""
    IMPROVING = "improving"     # Getting better
    STABLE = "stable"          # No change
    DEGRADING = "degrading"    # Getting worse
    UNKNOWN = "unknown"        # Insufficient data


class TrajectoryPhase(Enum):
    """Phase of trajectory lifecycle"""
    EXPLORATION = "exploration"     # Gathering information
    CONSOLIDATION = "consolidation" # Building understanding
    EXECUTION = "execution"         # Acting on knowledge
    TRANSITION = "transition"       # Shifting focus


@dataclass(frozen=True)
class StateVector:
    """
    A snapshot of cognitive state at a point in time.
    
    Unlike an event, this is a STATE representation.
    It captures the current configuration of the system.
    """
    vector_id: str
    timestamp: str
    tension_energy: float  # Total tension intensity
    attention_allocated: float  # How much attention is used
    goal_count: int  # Active goals
    belief_coherence: float  # How coherent beliefs are
    identity_stability: float  # Identity confidence
    knowledge_coverage: float  # How much is known
    trajectory_direction: str  # TrendDirection value
    version: int
    
    @staticmethod
    def from_state(state: Dict[str, Any], version: int) -> 'StateVector':
        """Create StateVector from state dict"""
        return StateVector(
            vector_id=f"vec_{hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:12]}",
            timestamp=datetime.utcnow().isoformat(),
            tension_energy=state.get('tension_energy', 0.0),
            attention_allocated=state.get('attention_allocated', 0.0),
            goal_count=state.get('goal_count', 0),
            belief_coherence=state.get('belief_coherence', 0.0),
            identity_stability=state.get('identity_stability', 0.0),
            knowledge_coverage=state.get('knowledge_coverage', 0.0),
            trajectory_direction=state.get('trajectory_direction', TrendDirection.UNKNOWN.value),
            version=version
        )


@dataclass(frozen=True)
class TrajectorySegment:
    """
    A segment of trajectory with temporal context.
    
    Contains multiple state vectors and metadata about
    the segment as a whole.
    """
    segment_id: str
    start_time: str
    end_time: str
    states: Tuple[StateVector, ...]  # Ordered sequence
    phase: str  # TrajectoryPhase value
    avg_tension_energy: float  # Average tension over segment
    dominant_events: Tuple[str, ...]  # What drove changes
    coherence_score: float  # How coherent this segment is
    version: int
    
    def duration_hours(self) -> float:
        """Duration of this segment in hours"""
        try:
            start = datetime.fromisoformat(self.start_time)
            end = datetime.fromisoformat(self.end_time)
            return (end - start).total_seconds() / 3600
        except:
            return 0.0
    
    def energy_trend(self) -> TrendDirection:
        """Compute energy trend over this segment"""
        if len(self.states) < 2:
            return TrendDirection.UNKNOWN
        
        energies = [s.tension_energy for s in self.states]
        first_half = sum(energies[:len(energies)//2]) / (len(energies)//2)
        second_half = sum(energies[len(energies)//2:]) / (len(energies) - len(energies)//2)
        
        if second_half > first_half * 1.1:
            return TrendDirection.IMPROVING
        elif second_half < first_half * 0.9:
            return TrendDirection.DEGRADING
        else:
            return TrendDirection.STABLE


@dataclass(frozen=True)
class TrajectoryState:
    """
    Complete trajectory state of the cognitive system.
    
    Tracks where we've been and projects where we're going.
    """
    current_vector: Optional[StateVector]
    history: Tuple[TrajectorySegment, ...]  # Past segments
    active_segment: Optional[TrajectorySegment]  # Current segment being built
    projected_vectors: Tuple[StateVector, ...]  # Predicted future
    divergence_detected: bool  # True if off expected trajectory
    divergence_magnitude: float  # How far off course
    version: int
    
    @staticmethod
    def empty() -> 'TrajectoryState':
        """Create empty trajectory state"""
        return TrajectoryState(
            current_vector=None,
            history=(),
            active_segment=None,
            projected_vectors=(),
            divergence_detected=False,
            divergence_magnitude=0.0,
            version=0
        )


def create_state_vector(state: Dict[str, Any], version: int) -> StateVector:
    """Create StateVector from state dict"""
    return StateVector.from_state(state, version)


def add_vector_to_trajectory(
    trajectory: TrajectoryState,
    vector: StateVector
) -> TrajectoryState:
    """Add a new state vector to trajectory"""
    if trajectory.active_segment is None:
        # Start new segment
        new_segment = TrajectorySegment(
            segment_id=f"seg_{vector.vector_id}",
            start_time=vector.timestamp,
            end_time=vector.timestamp,
            states=(vector,),
            phase=TrajectoryPhase.EXPLORATION.value,
            avg_tension_energy=vector.tension_energy,
            dominant_events=(),
            coherence_score=1.0,
            version=0
        )
        
        return TrajectoryState(
            current_vector=vector,
            history=trajectory.history,
            active_segment=new_segment,
            projected_vectors=trajectory.projected_vectors,
            divergence_detected=trajectory.divergence_detected,
            divergence_magnitude=trajectory.divergence_magnitude,
            version=trajectory.version + 1
        )
    else:
        # Add to existing segment
        new_states = trajectory.active_segment.states + (vector,)
        new_segment = TrajectorySegment(
            segment_id=trajectory.active_segment.segment_id,
            start_time=trajectory.active_segment.start_time,
            end_time=vector.timestamp,
            states=new_states,
            phase=trajectory.active_segment.phase,
            avg_tension_energy=sum(s.tension_energy for s in new_states) / len(new_states),
            dominant_events=trajectory.active_segment.dominant_events,
            coherence_score=compute_coherence(new_states),
            version=trajectory.active_segment.version + 1
        )
        
        return TrajectoryState(
            current_vector=vector,
            history=trajectory.history,
            active_segment=new_segment,
            projected_vectors=trajectory.projected_vectors,
            divergence_detected=trajectory.divergence_detected,
            divergence_magnitude=trajectory.divergence_magnitude,
            version=trajectory.version + 1
        )


def compute_coherence(states: Tuple[StateVector, ...]) -> float:
    """Compute coherence score for a sequence of states"""
    if len(states) < 2:
        return 1.0
    
    # Coherence = inverse of variance
    energies = [s.tension_energy for s in states]
    mean_energy = sum(energies) / len(energies)
    variance = sum((e - mean_energy) ** 2 for e in energies) / len(energies)
    
    # Normalize: high variance = low coherence
    max_variance = 1.0
    coherence = 1.0 - min(variance / max_variance, 1.0)
    
    return coherence


def detect_divergence(
    trajectory: TrajectoryState,
    expected_direction: str,
    tolerance: float = 0.2
) -> Tuple[bool, float]:
    """
    Detect if current trajectory diverges from expected.
    
    Returns:
    - divergence_detected: True if diverging
    - divergence_magnitude: How far off course (0-1)
    """
    if trajectory.current_vector is None:
        return False, 0.0
    
    current_direction = trajectory.current_vector.trajectory_direction
    
    if current_direction == expected_direction:
        return False, 0.0
    
    # Compute divergence magnitude
    # (Simplified: actual vs expected alignment)
    magnitude = abs(
        trajectory.current_vector.tension_energy -
        (expected_direction == "improving" and 0.3 or 0.7)
    ) * tolerance
    
    return True, min(magnitude, 1.0)


def project_trajectory(
    current: StateVector,
    trend: TrendDirection,
    steps: int = 5
) -> Tuple[StateVector, ...]:
    """
    Project future states based on current trajectory.
    
    This is a simplification - real projection would use
    learned models of state transitions.
    """
    projections = []
    current_time = datetime.fromisoformat(current.timestamp)
    
    for i in range(steps):
        # Simple linear projection
        time_delta = (i + 1) * 0.1  # Hours
        projected_time = datetime.utcnow()  # Use now for simplicity
        
        if trend == TrendDirection.IMPROVING:
            tension_delta = -0.05 * (i + 1)
        elif trend == TrendDirection.DEGRADING:
            tension_delta = 0.05 * (i + 1)
        else:
            tension_delta = 0.0
        
        projected_energy = max(0.0, min(1.0, current.tension_energy + tension_delta))
        
        projected = StateVector(
            vector_id=f"proj_{current.vector_id}_{i}",
            timestamp=projected_time.isoformat(),
            tension_energy=projected_energy,
            attention_allocated=current.attention_allocated,
            goal_count=current.goal_count,
            belief_coherence=current.belief_coherence,
            identity_stability=current.identity_stability,
            knowledge_coverage=current.knowledge_coverage,
            trajectory_direction=trend.value,
            version=current.version + i + 1
        )
        
        projections.append(projected)
    
    return tuple(projections)


def complete_segment(
    trajectory: TrajectoryState,
    phase: str
) -> TrajectoryState:
    """Complete current segment and start new one"""
    if trajectory.active_segment is None:
        return trajectory
    
    completed = TrajectorySegment(
        segment_id=trajectory.active_segment.segment_id,
        start_time=trajectory.active_segment.start_time,
        end_time=trajectory.active_segment.end_time,
        states=trajectory.active_segment.states,
        phase=phase,
        avg_tension_energy=trajectory.active_segment.avg_tension_energy,
        dominant_events=trajectory.active_segment.dominant_events,
        coherence_score=trajectory.active_segment.coherence_score,
        version=trajectory.active_segment.version
    )
    
    return TrajectoryState(
        current_vector=trajectory.current_vector,
        history=trajectory.history + (completed,),
        active_segment=None,
        projected_vectors=trajectory.projected_vectors,
        divergence_detected=trajectory.divergence_detected,
        divergence_magnitude=trajectory.divergence_magnitude,
        version=trajectory.version + 1
    )