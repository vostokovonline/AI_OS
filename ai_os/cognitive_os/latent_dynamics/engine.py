"""
Latent Dynamics Engine - Main Orchestrator

Phase 11-16: From Semantic Causality to Information Geometry

Components:
- Phase 11: Latent State Space, Causal Attribution, Semantic Compression
- Phase 12: Motif Discovery, Trajectory Clustering
- Phase 12.5: Behavioral Dynamics (graph, transitions, interpretation)
- Phase 13: Continuous Latent Dynamics (phase space, learned model, active inference)
- Phase 14: Unified Cognitive Physics (force-based)
- Phase 14.5: Riemannian Cognitive Manifold (variational, single L)
- Phase 15: Variational Geometric Inference (probabilistic energy, proper metrics)
- Phase 16: INFORMATION GEOMETRY (single source for everything)

Phase 16 - Single Generative Source:
  p_θ(z) → ONE SOURCE FOR:
    - Energy: V(z) = -log p_θ(z)
    - Metric (Fisher-Rao): g_ij(z) = E[∂_i log p * ∂_j log p]
    - Natural gradient: ż = -g^ij * ∂_j V = g^ij * ∂_j log p
    - Geodesics in probability space
    - KL divergence as geodesic distance

Key unification:
  Everything derives from single probabilistic model p_θ(z)
  NOT separate metric learning + energy learning + dynamics
  
  This is the TRUE information geometry system:
  - Planning = inference in probability space
  - Geodesics = KL divergence minimizers
  - Metric = Fisher information
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class DynamicsState:
    """Current state of dynamics system"""
    latent_state_id: str = ""
    dominant_causes: List[str] = field(default_factory=list)
    transition_count: int = 0
    novelty_score: float = 0.0
    causal_confidence: float = 0.0
    compression_ratio: float = 0.0
    
    # Behavioral dynamics (Phase 12.5)
    behavioral_state: Optional[Dict] = None


class LatentDynamicsEngine:
    """
    Latent Dynamics Engine - main orchestrator for semantic causality.
    
    Now includes Phase 12.5 behavioral dynamics.
    
    Usage:
        engine = LatentDynamicsEngine(cognitive_os)
        
        # Process transition with semantic causality
        transition = engine.process_transition(
            from_state=unified_state,
            to_state=next_unified_state,
            action="execute",
            outcome="success"
        )
        
        # Get causal analysis
        causes = engine.get_dominant_causes()
        
        # Behavioral dynamics
        dynamics = engine.get_behavioral_dynamics()
        
        # Counterfactual
        counterfactual = engine.reason_counterfactual(
            state_id=current.id,
            actual_action="execute",
            alternative_action="explore"
        )
        
        # Future prediction
        predictions = engine.predict_future(
            state_id=current.id,
            actions=["execute", "explore"]
        )
    """
    
    def __init__(self, cognitive_os):
        self.cognitive_os = cognitive_os
        
        from .latent_space import (
            LatentStateSpace,
            TransitionModel,
            SymbolicState,
            create_latent_space
        )
        from .causal_engine import (
            CausalAttributionEngine,
            CounterfactualReasoner,
            create_causal_attribution_engine
        )
        from .semantic_compression import (
            SemanticCompressor,
            CausalConfidence,
            create_semantic_compressor
        )
        from .continuous_dynamics import (
            ContinuousLatentDynamics,
            create_continuous_latent_dynamics
        )
        from .unified_dynamics import (
            UnifiedCognitiveDynamics,
            UnifiedState,
            UnifiedTrajectory,
            create_unified_dynamics
        )
        from .riemannian_manifold import (
            RiemannianCognitiveManifold,
            GeodesicState,
            create_riemannian_manifold
        )
        from .variational_inference import (
            VariationalGeometricInference,
            GeodesicStateVariational,
            create_variational_inference
        )
        from .information_geometry import (
            InformationGeometrySystem,
            create_information_geometry_system
        )
        
        self.latent_space = create_latent_space(dimension=16)
        self.transition_model = TransitionModel()
        self.causal_engine = create_causal_attribution_engine()
        self.counterfactual = CounterfactualReasoner(self.latent_space, self.transition_model)
        self.compressor = create_semantic_compressor()
        self.causal_confidence = CausalConfidence()
        
        self.behavioral_dynamics = get_behavioral_dynamics_engine(self.latent_space)
        self.energy_landscape = create_energy_landscape(dimension=16)
        
        # Phase 13: Continuous dynamics
        self.continuous_dynamics = create_continuous_latent_dynamics(dimension=16)
        
        # Phase 14: Unified dynamics (force-based)
        self.unified_dynamics = create_unified_dynamics(dimension=16)
        
        # Phase 14.5: Riemannian manifold (variational)
        self.riemannian_manifold = create_riemannian_manifold(dimension=16)
        
        # Phase 15: Variational geometric inference
        self.variational_inference = create_variational_inference(dimension=16)
        
        # Phase 16: Information geometry (single generative source)
        self.information_geometry = create_information_geometry_system(dimension=16, n_components=5)
        
        self._transition_history: List[Any] = []
        
        logger.info("latent_dynamics_engine_initialized_phase_12_5")
    
    def process_transition(
        self,
        from_state,
        to_state,
        action: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> Dict:
        """
        Process a state transition with full semantic causality.
        
        Returns:
            - Latent state IDs
            - Detected causes
            - Causal attribution
            - Transition type
        """
        from .latent_space import SymbolicState
        
        symbolic_from = SymbolicState.from_unified_state(from_state)
        symbolic_to = SymbolicState.from_unified_state(to_state)
        
        latent_from = self.latent_space.add_state(symbolic_from)
        latent_to = self.latent_space.add_state(symbolic_to)
        
        from_data = symbolic_from.to_dict()
        to_data = symbolic_to.to_dict()
        
        attribution = self.causal_engine.attribute_transition(
            from_state_id=latent_from.id,
            to_state_id=latent_to.id,
            from_state=from_data,
            to_state=to_data,
            action=action,
            outcome=outcome
        )
        
        if action:
            predicted = self.transition_model.predict_next_state(latent_from, action)
            actual = latent_to.vector
            pred_error = self.transition_model.compute_prediction_error(predicted, actual)
            
            for cause in attribution.latent_causes:
                self.causal_confidence.update_edge_confidence(cause.id, outcome == "success")
        
        self._transition_history.append({
            "from_state": latent_from.id,
            "to_state": latent_to.id,
            "action": action,
            "outcome": outcome,
            "causes": [c.cause_type for c in attribution.latent_causes]
        })
        
        return {
            "from_state_id": latent_from.id,
            "to_state_id": latent_to.id,
            "transition_id": attribution.id,
            "transition_type": attribution.transition_type,
            "latent_causes": [c.to_dict() for c in attribution.latent_causes],
            "causal_edges": [e.to_dict() for e in attribution.causal_edges],
            "novelty": attribution.novelty,
            "predictability": attribution.predictability,
            "prediction_error": latent_to.prediction_error,
        }
    
    def get_dominant_causes(self, limit: int = 5) -> List[Dict]:
        """Get dominant latent causes"""
        return self.causal_engine.get_dominant_causes(limit)
    
    def get_causal_path(
        self,
        from_state_id: str,
        to_state_id: str
    ) -> Optional[List[Dict]]:
        """Get causal path for a transition"""
        return self.causal_engine.get_causal_path(from_state_id, to_state_id)
    
    def reason_counterfactual(
        self,
        state_id: str,
        actual_action: str,
        alternative_action: str
    ) -> Dict:
        """Reason about counterfactual"""
        return self.counterfactual.reason_counterfactual(
            current_state_id=state_id,
            actual_action=actual_action,
            alternative_action=alternative_action
        )
    
    def predict_branch_outcomes(
        self,
        state_id: str,
        possible_actions: List[str]
    ) -> List[Dict]:
        """Predict outcomes for all possible actions"""
        return self.counterfactual.predict_branch_outcomes(state_id, possible_actions)
    
    def compress_trajectory(self, events: List[Dict]) -> List[Dict]:
        """Compress event stream into motifs"""
        from .semantic_compression import BehaviorMotif, TrajectoryChunk
        
        motifs = self.compressor.compress_events(events, self._transition_history)
        chunks = self.compressor.create_chunks_from_motifs(motifs)
        
        return {
            "motifs": [m.to_dict() for m in motifs],
            "chunks": [c.to_dict() for c in chunks],
            "compression_ratio": len(motifs) / len(events) if events else 0,
        }
    
    def get_state_evolution(self) -> List[Dict]:
        """Get full state evolution trajectory"""
        return self.latent_space.get_evolution_trajectory()
    
    def find_similar_states(self, state_id: str, threshold: float = 0.3) -> List[Dict]:
        """Find states similar to given state"""
        state = self.latent_space.get_state(state_id)
        if not state:
            return []
        
        similar = self.latent_space.find_similar_states(state.vector, threshold)
        
        return [
            {
                "state_id": s.id,
                "distance": dist,
                "timestamp": s.timestamp.isoformat(),
                "causes": s.latent_causes
            }
            for s, dist in similar
        ]
    
    def get_confidence_distribution(self) -> Dict:
        """Get distribution of causal edge confidences"""
        return self.causal_confidence.get_confidence_distribution()
    
    def get_unreliable_causes(self, threshold: float = 0.3) -> List[str]:
        """Get causes with low confidence"""
        return self.causal_confidence.get_unreliable_edges(threshold)
    
    def get_behavioral_dynamics(self) -> Dict:
        """Get behavioral dynamics state (Phase 12.5)"""
        return self.behavioral_dynamics.get_dynamics_state().to_dict()
    
    def predict_future(
        self,
        state_id: str,
        possible_actions: List[str],
        max_depth: int = 5
    ) -> Optional[Dict]:
        """Predict future trajectory manifold"""
        return self.behavioral_dynamics.predict_future(
            current_state_id=state_id,
            possible_actions=possible_actions,
            max_depth=max_depth
        )
    
    def get_flow_direction(
        self,
        state_id: str,
        current_motif: Optional[str] = None
    ) -> Tuple[List[float], float]:
        """Get flow direction and magnitude at state"""
        return self.behavioral_dynamics.get_flow_direction(state_id, current_motif)
    
    def find_attractor_path(
        self,
        from_state_id: str,
        to_motif: str
    ) -> List[str]:
        """Find path to attractor basin"""
        return self.behavioral_dynamics.find_attractor_path(from_state_id, to_motif)
    
    def interpret_motif(self, motif_state) -> Dict:
        """Interpret motif geometry into human-readable form"""
        return self.behavioral_dynamics.interpret_motif(motif_state)
    
    def detect_behavioral_loops(self) -> List[List[str]]:
        """Detect recurring trajectory loops"""
        return self.behavioral_dynamics.detect_loops()
    
    def detect_behavioral_drift(self) -> List[Dict]:
        """Detect behavioral drift episodes"""
        return self.behavioral_dynamics.detect_drift()
    
    def get_transition_graph(self) -> Dict:
        """Get motif transition graph"""
        return self.behavioral_dynamics.get_transition_graph()
    
    def predict_gradient_trajectory(
        self,
        start_position: List[float],
        steps: int = 10
    ) -> List[List[float]]:
        """
        Predict trajectory using gradient descent on energy landscape.
        
        This is NOT motif-level prediction - it's continuous latent prediction.
        """
        return self.energy_landscape.predict_trajectory(start_position, steps)
    
    def compute_landscape_energy(self, position: List[float]) -> float:
        """Compute energy at position in landscape"""
        return self.energy_landscape.field._compute_energy(position)
    
    def get_landscape_flow(self, position: List[float]) -> List[float]:
        """Get flow direction at position"""
        return self.energy_landscape.field.compute_flow(position)
    
    def find_basin_path(
        self,
        from_pos: List[float],
        to_pos: List[float]
    ) -> List[List[float]]:
        """Find minimum energy path between positions"""
        return self.energy_landscape.find_minimum_path(from_pos, to_pos)
    
    def get_attractor_strength(self, position: List[float]) -> float:
        """Compute how strongly attracted to nearest basin"""
        return self.energy_landscape.compute_attractor_strength(position)
    
    def visualize_landscape(self) -> Dict:
        """Get landscape visualization data"""
        return self.energy_landscape.visualize_flow()
    
    def unified_step(
        self,
        current_state: UnifiedState,
        dt: float = 0.01
    ) -> UnifiedState:
        """
        Single step using unified evolution equation.
        
        d²z/dt² = -∇E(z) + F_policy + F_inertia
        """
        return self.unified_dynamics.step(current_state, dt)
    
    def unified_simulate(
        self,
        initial_position: List[float],
        steps: int = 100,
        target_energy: float = 0.1
    ) -> UnifiedTrajectory:
        """
        Simulate trajectory using unified dynamics.
        
        Stops when equilibrium reached or max steps.
        """
        from .unified_dynamics import UnifiedState
        
        initial = UnifiedState(position=initial_position)
        initial.energy = self.unified_dynamics.energy_field.compute_energy(initial_position)
        initial.compute_derived()
        
        return self.unified_dynamics.simulate(initial, steps, target_energy)
    
    def set_unified_goal(
        self,
        goal_state: List[float],
        strength: float = 0.5
    ) -> None:
        """Set goal in unified dynamics"""
        self.unified_dynamics.set_goal(goal_state, strength)
    
    def get_phase_portrait(self) -> Dict:
        """Get phase portrait for visualization"""
        return self.unified_dynamics.get_phase_portrait()
    
    def observe_unified(
        self,
        position: List[float],
        outcome: Optional[str] = None
    ) -> UnifiedState:
        """Observe state and update unified dynamics"""
        return self.unified_dynamics.observe(position, outcome)
    
    def get_unified_dynamics_summary(self) -> Dict:
        """Get unified dynamics summary"""
        return self.unified_dynamics.get_dynamics_summary()
    
    # Phase 14.5: Riemannian Cognitive Manifold
    def observe_manifold(self, z: List[float], outcome: Optional[str] = None) -> GeodesicState:
        """Observe state and update manifold geometry"""
        return self.riemannian_manifold.observe(z, outcome)
    
    def geodesic_step(self, state: GeodesicState, dt: float = 0.01) -> GeodesicState:
        """Single geodesic step on manifold"""
        return self.riemannian_manifold.geodesic_step(state, dt)
    
    def integrate_manifold(
        self,
        initial_z: List[float],
        steps: int = 100,
        target_energy: float = 0.1
    ) -> List[GeodesicState]:
        """Integrate geodesic on manifold"""
        return self.riemannian_manifold.integrate(initial_z, steps, target_energy)
    
    def set_manifold_goal(self, goal: List[float], strength: float = 0.5) -> None:
        """Set goal on manifold (as potential minimum)"""
        self.riemannian_manifold.set_goal(goal, strength)
    
    def compute_geodesic(self, from_z: List[float], to_z: List[float]) -> List[List[float]]:
        """Compute geodesic path between two points"""
        return self.riemannian_manifold.compute_geodesic(from_z, to_z)
    
    def get_manifold_statistics(self) -> Dict:
        """Get Riemannian manifold statistics"""
        return self.riemannian_manifold.get_manifold_statistics()
    
    def get_manifold_phase_flow(self) -> Dict:
        """Get phase flow on manifold"""
        return self.riemannian_manifold.get_phase_flow()
    
    # Phase 15: Variational Geometric Inference (fully consistent)
    def observe_variational(self, z: List[float], outcome: Optional[str] = None) -> GeodesicStateVariational:
        """Observe state with variational inference"""
        return self.variational_inference.observe(z, outcome)
    
    def variational_step(self, state: GeodesicStateVariational, dt: float = 0.01) -> GeodesicStateVariational:
        """
        Single step with proper Riemannian dynamics.
        
        Euler-Lagrange: z̈^i + Γ^i_jk * ż^j * ż^k = -g^ij * ∂_j V
        """
        return self.variational_inference.geodesic_step(state, dt)
    
    def integrate_variational(
        self,
        initial_z: List[float],
        steps: int = 100
    ) -> List[GeodesicStateVariational]:
        """Integrate geodesic with variational consistency"""
        return self.variational_inference.integrate(initial_z, steps)
    
    def compute_variational_action(self, trajectory: List[GeodesicStateVariational]) -> float:
        """Compute action S = ∫ L dt"""
        return self.variational_inference.compute_action(trajectory)
    
    def set_variational_goal(self, goal_z: List[float], strength: float = 1.0) -> None:
        """Set goal in variational inference (as potential minimum)"""
        self.variational_inference.set_goal(goal_z, strength)
    
    def get_variational_statistics(self) -> Dict:
        """Get variational inference statistics"""
        return self.variational_inference.get_statistics()
    
    def get_variational_phase_flow(self) -> Dict:
        """Get variational phase flow"""
        return self.variational_inference.get_phase_flow()
    
    # Phase 16: Information Geometry (single generative source)
    def observe_ig(self, z: List[float], outcome: Optional[str] = None) -> None:
        """Observe state in information geometry system"""
        self.information_geometry.observe(z, outcome)
    
    def train_ig(self) -> None:
        """Train generative model (single source)"""
        self.information_geometry.train()
    
    def compute_ig_energy(self, z: List[float]) -> float:
        """Compute energy V(z) = -log p(z) from single model"""
        return self.information_geometry.compute_energy(z)
    
    def compute_ig_metric(self, z: List[float]) -> List[List[float]]:
        """Compute Fisher metric g_ij(z) from single model"""
        return self.information_geometry.compute_metric(z)
    
    def compute_ig_natural_gradient(self, z: List[float]) -> List[float]:
        """Compute natural gradient g^ij * ∂_j log p"""
        return self.information_geometry.compute_natural_gradient(z)
    
    def integrate_ig(
        self,
        initial_z: List[float],
        steps: int = 100
    ) -> List[Dict]:
        """Integrate natural gradient flow"""
        return self.information_geometry.integrate(initial_z, steps)
    
    def kl_divergence_ig(self, z1: List[float], z2: List[float]) -> float:
        """Compute KL divergence between two points"""
        return self.information_geometry.kl_divergence(z1, z2)
    
    def get_ig_statistics(self) -> Dict:
        """Get information geometry statistics"""
        return self.information_geometry.get_statistics()
    
    def get_ig_phase_flow(self) -> Dict:
        """Get information geometry phase flow"""
        return self.information_geometry.get_phase_flow()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        latent_stats = self.latent_space.get_evolution_trajectory()
        
        return {
            "latent_states": len(self.latent_space.history),
            "transitions_processed": len(self._transition_history),
            "dominant_causes": self.get_dominant_causes(5),
            "transition_types": self.causal_engine.get_statistics(),
            "compression": self.compressor.get_statistics(),
            "causal_confidence": self.get_confidence_distribution(),
            "novel_states": sum(1 for s in self.latent_space.history if s.novelty_score > 0.6),
            "avg_prediction_error": sum(s.prediction_error for s in self.latent_space.history) / len(self.latent_space.history) if self.latent_space.history else 0,
            "behavioral_dynamics": self.behavioral_dynamics.get_statistics(),
            "energy_landscape": self.energy_landscape.get_landscape_statistics(),
            "continuous_dynamics": self.continuous_dynamics.get_statistics(),
            "unified_dynamics": self.get_unified_dynamics_summary(),
            "riemannian_manifold": self.get_manifold_statistics(),
            "variational_inference": self.get_variational_statistics(),
            "information_geometry": self.get_ig_statistics(),
        }


# Singleton
_latent_dynamics_engine: Optional[LatentDynamicsEngine] = None


def get_latent_dynamics_engine(cognitive_os) -> LatentDynamicsEngine:
    """Get or create latent dynamics engine"""
    global _latent_dynamics_engine
    if _latent_dynamics_engine is None:
        _latent_dynamics_engine = LatentDynamicsEngine(cognitive_os)
    return _latent_dynamics_engine