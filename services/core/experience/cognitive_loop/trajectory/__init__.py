"""
Trajectory module - History and projection of cognitive state.

Components:
- tracking.py: State vectors, trajectory segments, divergence detection
"""

from cognitive_loop.trajectory.tracking import (
    TrendDirection,
    TrajectoryPhase,
    StateVector,
    TrajectorySegment,
    TrajectoryState,
    create_state_vector,
    add_vector_to_trajectory,
    compute_coherence,
    detect_divergence,
    project_trajectory,
    complete_segment
)

__all__ = [
    'TrendDirection',
    'TrajectoryPhase',
    'StateVector',
    'TrajectorySegment',
    'TrajectoryState',
    'create_state_vector',
    'add_vector_to_trajectory',
    'compute_coherence',
    'detect_divergence',
    'project_trajectory',
    'complete_segment'
]