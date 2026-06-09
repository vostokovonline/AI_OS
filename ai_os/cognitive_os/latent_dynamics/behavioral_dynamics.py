"""
Phase 12.5 - Behavioral Dynamics Engine

Unified engine integrating:
- TrajectoryMemoryGraph: behavioral continuum
- MotifTransitionMatrix: attractor flow dynamics
- InterpretationLayer: symbolic projection (NOT core!)
- TrajectoryRollout: future prediction manifold
- BehavioralFlowField: continuous flow approximation

Key architecture:
- Clean core: geometric, no labels
- InterpretationLayer: symbolic projection on top
- Future manifold: possible trajectories, not single prediction

This transitions from:
  State → Policy → Action

To:
  Trajectory History
      ↓
  Behavioral Manifold
      ↓
  Motif Attractors
      ↓
  Expected Futures
      ↓
  Policy Biasing
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class BehavioralDynamicsState:
    """
    Current state of behavioral dynamics system.
    
    This is the cognitive substrate - NOT API response.
    """
    active_motifs: int = 0
    total_trajectories: int = 0
    flow_entropy: float = 0.0
    metastability: float = 0.0
    expected_surprise: float = 0.0
    
    # Graph metrics
    loops_detected: int = 0
    drift_episodes: int = 0
    
    # Prediction metrics
    rollout_count: int = 0
    best_utility: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "active_motifs": self.active_motifs,
            "total_trajectories": self.total_trajectories,
            "flow_entropy": round(self.flow_entropy, 3),
            "metastability": round(self.metastability, 3),
            "expected_surprise": round(self.expected_surprise, 3),
            "loops_detected": self.loops_detected,
            "drift_episodes": self.drift_episodes,
            "rollout_count": self.rollout_count,
            "best_utility": round(self.best_utility, 3),
        }


class BehavioralDynamicsEngine:
    """
    Phase 12.5 - Behavioral Dynamics Engine
    
    This IS the cognitive substrate. NOT an API feature.
    
    Integrates trajectory graph, motif transitions, and future prediction
    into unified behavioral dynamics system.
    """
    
    def __init__(self, latent_space=None):
        self.latent_space = latent_space
        
        from .trajectory_graph import TrajectoryMemoryGraph, create_trajectory_graph
        from .transition_matrix import MotifTransitionMatrix, create_transition_matrix
        from .interpretation_layer import InterpretationLayer, create_interpretation_layer
        from .trajectory_rollout import (
            TrajectoryRollouter, 
            BehavioralFlowField,
            create_trajectory_rollouter
        )
        from .latent_space import TransitionModel
        
        self.trajectory_graph = create_trajectory_graph()
        self.transition_matrix = create_transition_matrix()
        self.interpretation_layer = create_interpretation_layer()
        
        self.transition_model = TransitionModel()
        
        self.rollouter = None
        if self.latent_space:
            self.rollouter = create_trajectory_rollouter(
                self.latent_space,
                self.transition_model,
                self.transition_matrix
            )
            self.flow_field = BehavioralFlowField(
                self.latent_space,
                self.transition_matrix
            )
        else:
            self.flow_field = None
        
        self._initialized = datetime.utcnow()
        
        logger.info("behavioral_dynamics_engine_initialized")
    
    def add_trajectory(
        self,
        embedding: List[float],
        start_state: List[float],
        end_state: List[float],
        shape_metrics: Dict[str, float],
        motif_id: Optional[str] = None,
        duration_ms: float = 0.0,
        event_count: int = 0,
        outcome: Optional[str] = None
    ) -> str:
        """Add trajectory to behavioral graph and update dynamics"""
        node = self.trajectory_graph.add_trajectory(
            embedding=embedding,
            start_state=start_state,
            end_state=end_state,
            shape_metrics=shape_metrics,
            motif_id=motif_id,
            duration_ms=duration_ms,
            event_count=event_count
        )
        
        if motif_id and node.previous_nodes:
            prev_node = self.trajectory_graph.nodes.get(node.previous_nodes[0])
            if prev_node and prev_node.motif_id:
                self.transition_matrix.add_observation(
                    from_motif=prev_node.motif_id,
                    to_motif=motif_id,
                    duration_ms=duration_ms,
                    outcome=outcome,
                    trajectory_id=node.trajectory_id
                )
        
        self.trajectory_graph.build_transition_matrix()
        
        return node.trajectory_id
    
    def process_transition(
        self,
        from_state,
        to_state,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        embedding: Optional[List[float]] = None
    ) -> Dict:
        """Process state transition with full behavioral dynamics"""
        from .latent_space import SymbolicState
        
        symbolic_from = SymbolicState.from_unified_state(from_state)
        symbolic_to = SymbolicState.from_unified_state(to_state)
        
        if embedding is None:
            embedding = to_state.get("embedding", [0.0] * 32)
        
        motif_id = self._find_nearest_motif(to_state)
        
        shape_metrics = {
            "curvature": to_state.get("curvature", 0.0),
            "directness": to_state.get("directness", 1.0),
            "volatility": to_state.get("volatility", 0.0),
            "momentum": to_state.get("momentum", 0.0),
        }
        
        trajectory_id = self.add_trajectory(
            embedding=embedding,
            start_state=symbolic_from.to_vector(),
            end_state=symbolic_to.to_vector(),
            shape_metrics=shape_metrics,
            motif_id=motif_id,
            outcome=outcome
        )
        
        return {
            "trajectory_id": trajectory_id,
            "motif_id": motif_id,
            "flow_graph": self.trajectory_graph.get_flow_statistics(),
            "transition_matrix": self.transition_matrix.get_statistics(),
        }
    
    def _find_nearest_motif(self, state) -> Optional[str]:
        """Find nearest motif to current state"""
        return state.get("motif_id")
    
    def predict_future(
        self,
        current_state_id: str,
        possible_actions: List[str],
        max_depth: int = 5
    ) -> Optional[Dict]:
        """Predict future trajectory manifold"""
        if not self.rollouter:
            logger.warning("rollouter_not_initialized")
            return None
        
        plan = self.rollouter.predict_rollouts(
            current_state_id=current_state_id,
            possible_actions=possible_actions,
            include_branches=True
        )
        
        return {
            "plan_id": plan.plan_id,
            "current_state_id": plan.current_state_id,
            "rollouts": [r.to_dict() for r in plan.rollouts],
            "best_rollout_id": plan.best_rollout_id,
            "best_utility": plan.best_utility,
            "expected_surprise": plan.expected_surprise,
        }
    
    def get_flow_direction(
        self,
        state_id: str,
        current_motif: Optional[str] = None
    ) -> Tuple[List[float], float]:
        """Get flow direction and magnitude at state"""
        if not self.flow_field:
            return [0.0] * 16, 0.0
        
        return self.flow_field.compute_flow_at(state_id, current_motif)
    
    def find_attractor_path(
        self,
        from_state_id: str,
        to_motif: str
    ) -> List[str]:
        """Find path to attractor basin"""
        if not self.flow_field:
            return []
        
        return self.flow_field.find_attractor_path(from_state_id, to_motif)
    
    def interpret_motif(self, motif_state) -> Dict:
        """Interpret motif geometry into human-readable form"""
        interpretation = self.interpretation_layer.interpret_motif(motif_state)
        return interpretation.to_dict()
    
    def interpret_trajectory(
        self,
        trajectory_node,
        context: Optional[Dict] = None
    ) -> Dict:
        """Interpret trajectory into human-readable form"""
        interpretation = self.interpretation_layer.interpret_trajectory(
            trajectory_node, context
        )
        return interpretation.to_dict()
    
    def get_dynamics_state(self) -> BehavioralDynamicsState:
        """Get current dynamics state"""
        state = BehavioralDynamicsState()
        
        state.total_trajectories = self.trajectory_graph.total_trajectories
        state.active_motifs = len(self.trajectory_graph.motifs)
        
        state.loops_detected = self.trajectory_graph.loop_count
        state.drift_episodes = self.trajectory_graph.drift_episodes
        
        stats = self.transition_matrix.get_statistics()
        state.flow_entropy = stats.get("avg_entropy", 0.0)
        
        metastability = self.transition_matrix.compute_metastability()
        state.metastability = sum(motifs.values()) / max(1, len(motifs)) if metastability else 0.0
        
        if self.rollouter:
            rollouter_stats = self.rollouter.get_rollout_statistics()
            state.rollout_count = rollouter_stats.get("total_rollouts", 0)
            state.best_utility = rollouter_stats.get("avg_utility", 0.0)
        
        return state
    
    def get_transition_graph(self) -> Dict:
        """Get transition graph for visualization"""
        return self.transition_matrix.get_transition_graph()
    
    def get_recent_flow(self, limit: int = 5) -> List[Dict]:
        """Get recent trajectory flow"""
        return self.trajectory_graph.get_recent_flow(limit)
    
    def detect_loops(self) -> List[List[str]]:
        """Detect recurring trajectory loops"""
        return self.trajectory_graph.detect_loops()
    
    def detect_drift(self) -> List[Dict]:
        """Detect behavioral drift episodes"""
        return self.trajectory_graph.detect_drift()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        dynamics_state = self.get_dynamics_state()
        
        return {
            "dynamics_state": dynamics_state.to_dict(),
            "trajectory_graph": self.trajectory_graph.get_flow_statistics(),
            "transition_matrix": self.transition_matrix.get_statistics(),
            "interpretation_layer": self.interpretation_layer.get_all_interpretations(),
            "rollouter": self.rollouter.get_rollout_statistics() if self.rollouter else {},
            "initialized_at": self._initialized.isoformat(),
        }


# Singleton
_behavioral_dynamics_engine: Optional[BehavioralDynamicsEngine] = None


def get_behavioral_dynamics_engine(latent_space=None) -> BehavioralDynamicsEngine:
    """Get or create behavioral dynamics engine"""
    global _behavioral_dynamics_engine
    if _behavioral_dynamics_engine is None:
        _behavioral_dynamics_engine = BehavioralDynamicsEngine(latent_space)
    return _behavioral_dynamics_engine


def reset_behavioral_dynamics_engine() -> None:
    """Reset the singleton (for testing)"""
    global _behavioral_dynamics_engine
    _behavioral_dynamics_engine = None