"""
Phase 9: Temporal Identity Field / Autobiographical Continuity

ARCHITECTURAL SHIFT:
  Before: trajectory → topology
  After:  history of trajectories → continuous identity field → topology deformation over time → future cognition constraints

KEY INSIGHT:
  Identity is NOT an attractor.
  Identity is a long-term curvature of the space of possible trajectories.

WHAT APPEARS:
  1. Temporal Identity Field I(z, t)
     - Slow deformation field
     - Affects attractor accessibility
     - Creates continuity pressure
     
  2. Autobiographical Compression
     - Not storing states
     - Storing: transformation motifs, trajectory morphisms, long-term self-deformations
     
  3. Temporal Curvature
     - History changes geometry
     - Repeated patterns → basin deepening
     
  4. Narrative Coherence Dynamics
     - "Would this future still feel like 'me'?"
     - Continuity constraints
     
  5. Developmental Time
     - Irreversible transformations
     - Developmental phases
     - Critical periods

PROGRESSION:
  Phase 8: Self-organizing topology (trajectories → attractors)
  Phase 9: Temporal identity field (history → curvature → future constraints)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class AutobiographicalEpisode:
    """
    A compressed representation of a trajectory segment.
    
    NOT: "what happened"
    BUT: "how I usually change"
    
    Stores transformation motifs, not states.
    """
    motif_id: str
    transformation_vector: np.ndarray  # Direction of change
    curvature_signature: np.ndarray    # How this changes the manifold
    temporal_extent: float             # How long this episode spans
    irreversibility: float             # 0 = fully reversible, 1 = irreversible
    coherence: float                  # How much this fits the identity narrative
    emotional_valence: float           # Positive/negative affect
    arousal: float                    # Intensity
    developmental_phase: str          # Early/middle/late
    
    def to_dict(self) -> dict:
        return {
            'motif_id': self.motif_id,
            'transformation_vector': self.transformation_vector.tolist(),
            'curvature_signature': self.curvature_signature.tolist(),
            'temporal_extent': self.temporal_extent,
            'irreversibility': self.irreversibility,
            'coherence': self.coherence,
            'emotional_valence': self.emotional_valence,
            'arousal': self.arousal,
            'developmental_phase': self.developmental_phase
        }


@dataclass
class IdentityCurvature:
    """
    How history has deformed the manifold.
    
    NOT: static attractor position
    BUT: accumulated curvature from repeated patterns
    """
    direction: np.ndarray           # Which direction in latent space is "more me"
    intensity: float                # How strong the identity is in this direction
    persistence: float             # How stable this curvature is
    developmental_depth: float      # How deep into developmental history
    basin_depth_modifier: float     # How much deeper basins are in identity direction
    
    def apply_to_potential(self, z: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Apply identity curvature to a potential field.
        
        Makes basins deeper in identity direction.
        Makes trajectories toward identity direction more likely.
        """
        identity_direction = self.direction / (np.linalg.norm(self.direction) + 1e-8)
        
        V_mod = V.copy()
        
        for i in range(len(V_mod)):
            z_i = z[i]
            z_norm = np.linalg.norm(z_i)
            
            if z_norm > 0.01:
                alignment = np.dot(z_i / z_norm, identity_direction)
                
                # If we're moving toward identity direction, lower potential
                if alignment > 0:
                    # Gradient toward identity
                    gradient = identity_direction * alignment * self.intensity * 0.1
                    V_mod[i] = V_mod[i] - np.dot(gradient, identity_direction)
        
        return V_mod


@dataclass 
class ContinuityPressure:
    """
    Pressure toward maintaining self-continuity.
    
    NOT: resistance to change
    BUT: preference for trajectories that "feel like me"
    """
    coherence_threshold: float        # Minimum narrative coherence required
    identity_preservation_weight: float  # How much to preserve identity
    narrative_tension: float         # Current tension between options and identity
    developmental_direction: np.ndarray  # Where development is pushing
    
    def compute_continuity_cost(self, proposed_z: np.ndarray, 
                                 identity_curvature: IdentityCurvature) -> float:
        """
        Cost of deviating from identity.
        
        Low cost = trajectory feels like "me"
        High cost = trajectory feels foreign
        """
        identity_dir = identity_curvature.direction
        identity_dir = identity_dir / (np.linalg.norm(identity_dir) + 1e-8)
        
        # How far is proposed state from identity direction?
        deviation = 1 - np.dot(proposed_z / (np.linalg.norm(proposed_z) + 1e-8), identity_dir)
        
        # Cost increases with identity intensity (stronger self = more costly deviation)
        cost = deviation * identity_curvature.intensity * self.identity_preservation_weight
        
        # Also consider developmental momentum
        if np.linalg.norm(self.developmental_direction) > 0:
            developmental_dir = self.developmental_direction / np.linalg.norm(self.developmental_direction)
            developmental_alignment = np.dot(proposed_z / (np.linalg.norm(proposed_z) + 1e-8), developmental_dir)
            # Deviating from developmental direction also costs
            if developmental_alignment < 0:
                cost = cost * 1.5
        
        return cost


class TemporalIdentityField:
    """
    I(z, t) - Field that provides autobiographical continuity.
    
    NOT: memory buffer
    NOT: replay
    NOT: RNN hidden state
    
    BUT: continuous self-neural field that:
      - Slowly deforms over time
      - Affects attractor accessibility
      - Changes learning dynamics
      - Creates continuity pressure
      - Deepens basins in identity direction
    """
    
    def __init__(self, latent_dim: int = 2):
        self.latent_dim = latent_dim
        
        # Autobiographical memory - transformation motifs, not states
        self.episodes: List[AutobiographicalEpisode] = []
        
        # Current identity curvature (accumulated from history)
        self.identity_curvature = IdentityCurvature(
            direction=np.zeros(latent_dim),
            intensity=0.1,
            persistence=0.9,
            developmental_depth=0.0,
            basin_depth_modifier=0.2
        )
        
        # Continuity pressure system
        self.continuity_pressure = ContinuityPressure(
            coherence_threshold=0.5,
            identity_preservation_weight=0.3,
            narrative_tension=0.0,
            developmental_direction=np.zeros(latent_dim)
        )
        
        # Temporal evolution parameters
        self.time_constant = 0.1  # How fast identity field evolves
        self.developmental_phase = "early"
        
        # Compression parameters
        self.max_episodes = 100
        
        # Tracking
        self.temporal_curvature_history: List[IdentityCurvature] = []
        self.narrative_coherence_history: List[float] = []
        
    def record_transformation(self, trajectory: np.ndarray, 
                              emotional_state: Optional[Dict] = None) -> AutobiographicalEpisode:
        """
        Record a transformation motif from trajectory.
        
        NOT: store trajectory states
        BUT: extract transformation pattern
        """
        if len(trajectory) < 2:
            return None
            
        # Compute transformation vector (net change)
        transformation = trajectory[-1] - trajectory[0]
        
        # Compute curvature signature (how this changes the manifold)
        # Use trajectory variance as curvature indicator
        curvature = np.var(trajectory, axis=0) if len(trajectory) > 2 else np.zeros(self.latent_dim)
        
        # Estimate temporal extent
        temporal_extent = np.sum([np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                                 for i in range(len(trajectory)-1)])
        
        # Estimate irreversibility from trajectory reversibility
        reversibility = self._estimate_reversibility(trajectory)
        irreversibility = 1 - reversibility
        
        # Compute narrative coherence
        coherence = self._compute_narrative_coherence(transformation, self.identity_curvature)
        
        # Emotional state (if available)
        valence = emotional_state.get('valence', 0.0) if emotional_state else 0.0
        arousal = emotional_state.get('arousal', 0.5) if emotional_state else 0.5
        
        # Create episode
        episode = AutobiographicalEpisode(
            motif_id=f"motif_{len(self.episodes)}_{self.developmental_phase}",
            transformation_vector=transformation,
            curvature_signature=curvature,
            temporal_extent=temporal_extent,
            irreversibility=irreversibility,
            coherence=coherence,
            emotional_valence=valence,
            arousal=arousal,
            developmental_phase=self.developmental_phase
        )
        
        self.episodes.append(episode)
        
        # Prune old episodes (keep most coherent ones)
        if len(self.episodes) > self.max_episodes:
            self._prune_episodes()
        
        return episode
    
    def _estimate_reversibility(self, trajectory: np.ndarray) -> float:
        """
        How reversible is this trajectory?
        
        Low reversibility = high irreversibility = identity-forming
        """
        if len(trajectory) < 3:
            return 1.0
            
        # Check if trajectory can return to origin
        endpoint = trajectory[-1]
        
        # Simple reversibility: distance from endpoint to trajectory start
        reversibility = 1.0 - min(1.0, np.linalg.norm(endpoint) / (np.linalg.norm(trajectory[0]) + 1e-8))
        
        return max(0.0, min(1.0, reversibility))
    
    def _compute_narrative_coherence(self, transformation: np.ndarray,
                                     current_curvature: IdentityCurvature) -> float:
        """
        Does this transformation fit the identity narrative?
        
        High coherence = transformation is consistent with "who I am"
        Low coherence = transformation is identity-disrupting
        """
        if np.linalg.norm(current_curvature.direction) < 0.1:
            return 0.5  # Neutral when identity is weak
            
        # Alignment with identity direction
        identity_dir = current_curvature.direction / np.linalg.norm(current_curvature.direction)
        transform_dir = transformation / (np.linalg.norm(transformation) + 1e-8)
        
        alignment = np.dot(identity_dir, transform_dir)
        
        # Coherence combines alignment with transformation magnitude
        coherence = (alignment + 1) / 2  # Map to [0, 1]
        
        return coherence
    
    def _prune_episodes(self):
        """Keep most coherent episodes, prune least coherent."""
        if len(self.episodes) <= self.max_episodes:
            return
            
        # Sort by coherence (keep high coherence episodes)
        sorted_episodes = sorted(self.episodes, key=lambda e: e.coherence, reverse=True)
        
        # Keep top episodes
        self.episodes = sorted_episodes[:self.max_episodes]
    
    def update_identity_curvature(self, new_episode: AutobiographicalEpisode):
        """
        Update identity curvature based on new transformation.
        
        This is where history changes the geometry.
        """
        # Accumulate transformation into identity direction
        alpha = self.time_constant * new_episode.irreversibility * new_episode.coherence
        
        # Exponential moving average of transformation direction
        current_dir = self.identity_curvature.direction
        new_dir = current_dir + alpha * new_episode.transformation_vector
        
        # Normalize
        if np.linalg.norm(new_dir) > 0.01:
            new_dir = new_dir / np.linalg.norm(new_dir)
        
        self.identity_curvature.direction = new_dir
        
        # Update intensity (grows with coherent, irreversible transformations)
        self.identity_curvature.intensity = min(
            1.0,
            self.identity_curvature.intensity + 0.01 * new_episode.irreversibility * new_episode.coherence
        )
        
        # Update persistence (identity becomes more stable)
        self.identity_curvature.persistence = min(
            0.99,
            self.identity_curvature.persistence + 0.001
        )
        
        # Update developmental depth
        self.identity_curvature.developmental_depth += 0.01
        
        # Update basin depth modifier (repeated patterns deepen basins)
        self.identity_curvature.basin_depth_modifier = min(
            0.5,
            self.identity_curvature.basin_depth_modifier + 0.005 * new_episode.irreversibility
        )
        
        # Update developmental phase
        self._update_developmental_phase()
        
        # Store history
        self.temporal_curvature_history.append(IdentityCurvature(
            direction=self.identity_curvature.direction.copy(),
            intensity=self.identity_curvature.intensity,
            persistence=self.identity_curvature.persistence,
            developmental_depth=self.identity_curvature.developmental_depth,
            basin_depth_modifier=self.identity_curvature.basin_depth_modifier
        ))
        
        # Limit history
        if len(self.temporal_curvature_history) > 1000:
            self.temporal_curvature_history = self.temporal_curvature_history[-500:]
    
    def _update_developmental_phase(self):
        """Update developmental phase based on accumulated depth."""
        depth = self.identity_curvature.developmental_depth
        
        if depth < 10:
            self.developmental_phase = "early"
        elif depth < 50:
            self.developmental_phase = "middle"
        else:
            self.developmental_phase = "late"
        
        # Critical periods: phase transitions
        if depth > 0 and depth < 5:
            self.developmental_phase = "early (critical period)"
        elif depth > 20 and depth < 25:
            self.developmental_phase = "middle (critical period)"
    
    def compute_continuity_pressure(self, proposed_state: np.ndarray,
                                    potential_field: Optional[np.ndarray] = None,
                                    z_grid: Optional[np.ndarray] = None) -> ContinuityPressure:
        """
        Compute pressure toward maintaining self-continuity.
        
        NOT: resistance to change
        BUT: preference for trajectories that "feel like me"
        """
        # Compute cost of deviating from identity
        cost = self.continuity_pressure.compute_continuity_cost(
            proposed_state, self.identity_curvature
        )
        
        # Update narrative tension
        self.continuity_pressure.narrative_tension = min(1.0, cost)
        
        # Update developmental direction from recent episodes
        if len(self.episodes) > 5:
            recent = self.episodes[-5:]
            developmental_dir = np.mean([e.transformation_vector for e in recent], axis=0)
            if np.linalg.norm(developmental_dir) > 0.01:
                self.continuity_pressure.developmental_direction = developmental_dir
        
        return self.continuity_pressure
    
    def apply_identity_to_manifold(self, z_grid: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Apply identity curvature to a manifold/potential field.
        
        This is the key: history changes geometry.
        
        - Basins become deeper in identity direction
        - Trajectories toward identity become more likely
        - Identity creates "preferred paths" through state space
        """
        V_modified = self.identity_curvature.apply_to_potential(z_grid, V)
        
        # Also apply continuity pressure gradient
        continuity_gradient = self._compute_continuity_gradient(z_grid)
        
        # Modify potential with continuity gradient
        V_modified = V_modified + continuity_gradient * 0.1
        
        return V_modified
    
    def _compute_continuity_gradient(self, z_grid: np.ndarray) -> np.ndarray:
        """
        Compute gradient of continuity pressure.
        
        This pushes state toward identity-consistent regions.
        """
        gradient = np.zeros(len(z_grid))
        
        identity_dir = self.identity_curvature.direction
        if np.linalg.norm(identity_dir) < 0.01:
            return gradient
            
        identity_dir = identity_dir / np.linalg.norm(identity_dir)
        
        # Gradient magnitude based on identity intensity
        magnitude = self.identity_curvature.intensity * self.continuity_pressure.identity_preservation_weight
        
        for i in range(len(z_grid)):
            z = z_grid[i]
            if np.linalg.norm(z) > 0.01:
                z_unit = z / np.linalg.norm(z)
                # Alignment with identity direction
                alignment = np.dot(z_unit, identity_dir)
                
                # Push toward identity direction
                if alignment > 0:
                    # Closer to identity = lower potential
                    gradient[i] = -magnitude * alignment
                else:
                    # Moving away = higher potential (continuity cost)
                    gradient[i] = magnitude * abs(alignment)
        
        return gradient
    
    def check_developmental_criticality(self, proposed_state: np.ndarray) -> Dict:
        """
        Check if we're in a critical developmental period.
        
        Critical periods: times when small changes have large long-term effects.
        """
        in_critical_period = "critical" in self.developmental_phase
        
        if not in_critical_period:
            return {'in_critical_period': False, 'sensitivity': 1.0}
        
        # Higher sensitivity in critical periods
        sensitivity = 2.0 if in_critical_period else 1.0
        
        # Check if proposed state deviates from developmental direction
        dev_dir = self.continuity_pressure.developmental_direction
        if np.linalg.norm(dev_dir) > 0.01:
            dev_dir = dev_dir / np.linalg.norm(dev_dir)
            proposed_dir = proposed_state / (np.linalg.norm(proposed_state) + 1e-8)
            deviation = 1 - np.dot(dev_dir, proposed_dir)
        else:
            deviation = 0.0
        
        return {
            'in_critical_period': in_critical_period,
            'sensitivity': sensitivity,
            'deviation_from_developmental_direction': deviation,
            'developmental_phase': self.developmental_phase
        }
    
    def evaluate_narrative_coherence(self, proposed_trajectory: np.ndarray) -> float:
        """
        Would this trajectory feel like "me"?
        
        NOT: "is this a good trajectory?"
        BUT: "does this trajectory maintain self-continuity?"
        """
        if len(proposed_trajectory) < 2:
            return 0.5
        
        # Extract transformation from trajectory
        transformation = proposed_trajectory[-1] - proposed_trajectory[0]
        
        # Compute coherence with current identity
        coherence = self._compute_narrative_coherence(transformation, self.identity_curvature)
        
        # Also check against autobiographical memory
        if len(self.episodes) > 0:
            # How similar is this to past transformation motifs?
            similarities = []
            for episode in self.episodes[-20:]:  # Recent episodes
                similarity = np.dot(transformation, episode.transformation_vector) / (
                    np.linalg.norm(transformation) * np.linalg.norm(episode.transformation_vector) + 1e-8
                )
                similarities.append(similarity)
            
            avg_similarity = np.mean(similarities)
            
            # Combine: trajectory coherence + historical similarity
            coherence = 0.6 * coherence + 0.4 * (avg_similarity + 1) / 2
        
        self.narrative_coherence_history.append(coherence)
        
        return coherence
    
    def get_identity_state(self) -> Dict:
        """Get comprehensive identity state."""
        return {
            'direction': self.identity_curvature.direction.tolist(),
            'intensity': self.identity_curvature.intensity,
            'persistence': self.identity_curvature.persistence,
            'developmental_depth': self.identity_curvature.developmental_depth,
            'basin_depth_modifier': self.identity_curvature.basin_depth_modifier,
            'developmental_phase': self.developmental_phase,
            'episodes_count': len(self.episodes),
            'narrative_tension': self.continuity_pressure.narrative_tension,
            'continuity_coherence_threshold': self.continuity_pressure.coherence_threshold
        }
    
    def get_autobiographical_summary(self) -> Dict:
        """Get autobiographical memory summary."""
        if not self.episodes:
            return {'motifs': [], 'transformations': []}
        
        recent = self.episodes[-20:]
        
        return {
            'motifs': [e.motif_id for e in recent],
            'avg_irreversibility': np.mean([e.irreversibility for e in recent]),
            'avg_coherence': np.mean([e.coherence for e in recent]),
            'developmental_trajectory': [e.developmental_phase for e in recent[-5:]],
            'dominant_emotions': {
                'valence': np.mean([e.emotional_valence for e in recent]),
                'arousal': np.mean([e.arousal for e in recent])
            }
        }


class AutobiographicalCognitiveAgent:
    """
    Agent that uses Temporal Identity Field for autobiographical continuity.
    
    NOT: RNN-based agent
    NOT: memory buffer agent
    
    BUT: Agent with:
      - Autobiographical episodes (transformation motifs)
      - Identity curvature (long-term geometry deformation)
      - Continuity pressure (preference for "me-like" trajectories)
      - Developmental trajectory
    """
    
    def __init__(self, latent_dim: int = 2):
        self.latent_dim = latent_dim
        
        # State
        self.state = np.random.randn(latent_dim) * 0.1
        
        # Temporal Identity Field
        self.identity_field = TemporalIdentityField(latent_dim)
        
        # Current potential field (learned)
        self.potential_field = None
        self.z_grid = None
        
        # Tracking
        self.trajectory_history: List[np.ndarray] = []
        self.decision_history: List[Dict] = []
        
    def step(self, action: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Take action and update state.
        
        Also update identity field with transformation.
        """
        # Update state
        new_state = self.state + action * dt
        
        # Record trajectory
        self.trajectory_history.append(self.state.copy())
        if len(self.trajectory_history) > 1000:
            self.trajectory_history = self.trajectory_history[-500:]
        
        # Every few steps, record transformation to identity field
        if len(self.trajectory_history) % 20 == 0 and len(self.trajectory_history) >= 20:
            recent = np.array(self.trajectory_history[-20:])
            episode = self.identity_field.record_transformation(recent)
            if episode:
                self.identity_field.update_identity_curvature(episode)
        
        self.state = new_state
        return new_state
    
    def decide_action(self, goal_state: np.ndarray) -> np.ndarray:
        """
        Decide action considering identity continuity.
        
        NOT: pure goal-directed
        BUT: goal-directed + identity-constrained
        """
        # Goal direction
        goal_direction = goal_state - self.state
        
        # Identity constraint
        continuity = self.identity_field.compute_continuity_pressure(goal_state)
        identity_cost = continuity.narrative_tension
        
        # Developmental criticality
        criticality = self.identity_field.check_developmental_criticality(goal_state)
        
        # Compute base action (toward goal)
        if np.linalg.norm(goal_direction) > 0.01:
            base_action = goal_direction / np.linalg.norm(goal_direction)
        else:
            base_action = np.zeros(self.latent_dim)
        
        # Modify by continuity cost (pull toward identity-consistent path)
        identity_dir = self.identity_field.identity_curvature.direction
        if np.linalg.norm(identity_dir) > 0.1:
            identity_dir = identity_dir / np.linalg.norm(identity_dir)
            
            # Blend goal and identity
            # If high continuity cost, lean toward identity
            identity_weight = identity_cost * 0.5
            goal_weight = 1 - identity_weight
            
            blended_direction = goal_weight * goal_direction / (np.linalg.norm(goal_direction) + 1e-8) + \
                              identity_weight * identity_dir
            
            if np.linalg.norm(blended_direction) > 0.01:
                action = blended_direction
            else:
                action = base_action
        else:
            action = base_action
        
        # Scale by criticality (critical periods = more careful)
        action = action / criticality['sensitivity']
        
        # Record decision
        self.decision_history.append({
            'goal_direction': goal_direction.tolist(),
            'identity_cost': identity_cost,
            'criticality': criticality,
            'developmental_phase': self.identity_field.developmental_phase
        })
        
        return action
    
    def evaluate_trajectory_narrative(self, trajectory: np.ndarray) -> float:
        """
        Evaluate how much trajectory feels like "me".
        """
        return self.identity_field.evaluate_narrative_coherence(trajectory)
    
    def get_self_model(self) -> Dict:
        """Get comprehensive self-model."""
        return {
            'current_state': self.state.tolist(),
            'identity': self.identity_field.get_identity_state(),
            'autobiographical': self.identity_field.get_autobiographical_summary(),
            'trajectory_length': len(self.trajectory_history),
            'decisions_made': len(self.decision_history)
        }


def test_temporal_identity_field():
    """Test Temporal Identity Field."""
    print("\n" + "=" * 60)
    print("TEMPORAL IDENTITY FIELD TEST")
    print("=" * 60)
    
    agent = AutobiographicalCognitiveAgent(latent_dim=2)
    
    # Simulate developmental trajectory with repeated patterns
    print("\n  Simulating developmental trajectory:")
    
    # Phase 1: Early exploration (random trajectories)
    print("\n  Phase 1: Early exploration")
    for step in range(50):
        # Random exploration with slight goal bias
        goal = np.array([2.0, 2.0]) + np.random.randn(2) * 0.5
        action = agent.decide_action(goal)
        agent.step(action * 0.1, dt=0.1)
    
    identity_early = agent.identity_field.get_identity_state()
    print(f"    Identity direction: {identity_early['direction'][:2]}")
    print(f"    Identity intensity: {identity_early['intensity']:.3f}")
    print(f"    Developmental phase: {identity_early['developmental_phase']}")
    print(f"    Episodes: {identity_early['episodes_count']}")
    
    # Phase 2: Pattern formation (consistent transformations)
    print("\n  Phase 2: Pattern formation (consistent transformations)")
    for cycle in range(10):
        for step in range(20):
            # Strong consistent direction (identity-forming)
            goal = np.array([1.0, 1.0])  # Same direction each time
            action = agent.decide_action(goal)
            agent.step(action * 0.1, dt=0.1)
    
    identity_middle = agent.identity_field.get_identity_state()
    print(f"    Identity direction: {identity_middle['direction'][:2]}")
    print(f"    Identity intensity: {identity_middle['intensity']:.3f}")
    print(f"    Basin depth modifier: {identity_middle['basin_depth_modifier']:.3f}")
    print(f"    Episodes: {identity_middle['episodes_count']}")
    
    # Phase 3: Late development (test identity stability)
    print("\n  Phase 3: Testing identity stability")
    for step in range(100):
        # Try to deviate from identity
        # If strong identity, deviations should be costly
        goal = -agent.identity_field.identity_curvature.direction * 2  # Opposite of identity
        action = agent.decide_action(goal)
        agent.step(action * 0.05, dt=0.1)  # Slower due to high cost
    
    print(f"    Narrative tension: {agent.identity_field.continuity_pressure.narrative_tension:.3f}")
    
    # Test narrative coherence
    print("\n  Testing narrative coherence:")
    
    # Coherent trajectory (consistent with identity)
    coherent_traj = np.array([
        agent.state,
        agent.state + agent.identity_field.identity_curvature.direction * 0.5,
        agent.state + agent.identity_field.identity_curvature.direction * 1.0
    ])
    coherent_score = agent.evaluate_trajectory_narrative(coherent_traj)
    print(f"    Coherent trajectory: {coherent_score:.3f}")
    
    # Incoherent trajectory (against identity)
    incoherent_traj = np.array([
        agent.state,
        agent.state - agent.identity_field.identity_curvature.direction * 0.5,
        agent.state - agent.identity_field.identity_curvature.direction * 1.0
    ])
    incoherent_score = agent.evaluate_trajectory_narrative(incoherent_traj)
    print(f"    Incoherent trajectory: {incoherent_score:.3f}")
    
    # Autobiographical summary
    summary = agent.identity_field.get_autobiographical_summary()
    print(f"\n  Autobiographical summary:")
    print(f"    Motifs: {summary['motifs'][:5]}...")
    print(f"    Avg irreversibility: {summary['avg_irreversibility']:.3f}")
    print(f"    Avg coherence: {summary['avg_coherence']:.3f}")
    print(f"    Dominant valence: {summary['dominant_emotions']['valence']:.3f}")
    
    # Full self model
    self_model = agent.get_self_model()
    print(f"\n  Full self model:")
    print(f"    Current state: {self_model['current_state'][:2]}")
    print(f"    Identity intensity: {self_model['identity']['intensity']:.3f}")
    print(f"    Developmental depth: {self_model['identity']['developmental_depth']:.3f}")
    print(f"    Trajectory length: {self_model['trajectory_length']}")
    print(f"    Decisions made: {self_model['decisions_made']}")


def test_identity_changes_geometry():
    """Test that history changes the geometry of the manifold."""
    print("\n" + "=" * 60)
    print("IDENTITY CHANGES GEOMETRY TEST")
    print("=" * 60)
    
    # Create manifold
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)
    z_grid = np.c_[X.ravel(), Y.ravel()]
    
    # Base potential (simple basins)
    V_base = (X**2 + Y**2 - 10).ravel()  # Two basins centered at [±3, ±3]
    
    # Create identity field
    identity_field = TemporalIdentityField(latent_dim=2)
    
    # Set identity direction
    identity_field.identity_curvature.direction = np.array([1.0, 1.0]) / np.sqrt(2)
    identity_field.identity_curvature.intensity = 0.5
    identity_field.identity_curvature.basin_depth_modifier = 0.3
    
    print("\n  Testing geometry modification:")
    print(f"    Identity direction: {identity_field.identity_curvature.direction}")
    print(f"    Identity intensity: {identity_field.identity_curvature.intensity}")
    
    # Apply identity to manifold
    V_modified = identity_field.apply_identity_to_manifold(z_grid, V_base)
    
    # Compare
    diff = np.mean(np.abs(V_modified - V_base))
    print(f"    Mean geometry modification: {diff:.3f}")
    
    # Check that identity direction has lower potential
    identity_points = [
        np.array([2, 2]),
        np.array([3, 3]),
        np.array([4, 4])
    ]
    
    print("\n  Potential in identity direction:")
    for p in identity_points:
        idx = np.argmin([np.linalg.norm(z_grid[i] - p) for i in range(len(z_grid))])
        print(f"    {p}: base={V_base[idx]:.3f}, modified={V_modified[idx]:.3f}")
    
    # Check perpendicular direction has higher potential
    perpendicular_points = [
        np.array([2, -2]),
        np.array([3, -3]),
        np.array([4, -4])
    ]
    
    print("\n  Potential in perpendicular direction:")
    for p in perpendicular_points:
        idx = np.argmin([np.linalg.norm(z_grid[i] - p) for i in range(len(z_grid))])
        print(f"    {p}: base={V_base[idx]:.3f}, modified={V_modified[idx]:.3f}")
    
    print("\n  Observation: Identity direction has lower potential")
    print("  This makes trajectories toward 'me' more likely.")


def test_developmental_critical_periods():
    """Test developmental phase transitions."""
    print("\n" + "=" * 60)
    print("DEVELOPMENTAL CRITICAL PERIODS TEST")
    print("=" * 60)
    
    agent = AutobiographicalCognitiveAgent(latent_dim=2)
    
    # Simulate with critical period detection
    print("\n  Simulating developmental progression:")
    
    for phase in ['early', 'middle', 'late']:
        # Set developmental phase for this simulation
        agent.identity_field.developmental_phase = phase
        
        # Check critical period
        criticality = agent.identity_field.check_developmental_criticality(agent.state)
        print(f"    Phase: {phase}")
        print(f"      In critical period: {criticality['in_critical_period']}")
        print(f"      Sensitivity: {criticality['sensitivity']:.1f}")
        
        # Simulate some steps
        for _ in range(30):
            goal = np.random.randn(2) * 2
            action = agent.decide_action(goal)
            agent.step(action * 0.1, dt=0.1)
        
        # Update developmental depth
        agent.identity_field.identity_curvature.developmental_depth += 1
        agent.identity_field._update_developmental_phase()
        
        print(f"      Developmental depth: {agent.identity_field.identity_curvature.developmental_depth:.1f}")
        print(f"      Current phase: {agent.identity_field.developmental_phase}")


def compare_with_phase8():
    """Compare Phase 9 (Temporal Identity Field) with Phase 8 (Self-Organizing)."""
    print("\n" + "=" * 60)
    print("PHASE 8 VS PHASE 9 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 8 (Self-Organizing Cognition):")
    print("    - Contrastive dynamical representation")
    print("    - Self-organizing manifold")
    print("    - Identity as topology constraint")
    print("    - Counterfactual self-modeling")
    print("    - Trajectories → attractors → topology")
    
    print("\n  Phase 9 (Temporal Identity Field):")
    print("    - Autobiographical episodes (transformation motifs)")
    print("    - Identity curvature (long-term geometry deformation)")
    print("    - Continuity pressure ('would this feel like me?')")
    print("    - Developmental phases and critical periods")
    print("    - History → curvature → future constraints")
    
    print("\n  Key architectural shift:")
    print("    Phase 8: trajectories → static topology")
    print("    Phase 9: history → deforming topology over time")
    
    print("\n  What Phase 9 adds to Phase 8:")
    print("    1. Historical inertia of self")
    print("    2. Long-term identity morphogenesis")
    print("    3. Temporal narrative consistency")
    print("    4. 'I continue through time' feeling")
    print("    5. Developmental trajectory (not just state transitions)")


if __name__ == "__main__":
    test_temporal_identity_field()
    test_identity_changes_geometry()
    test_developmental_critical_periods()
    compare_with_phase8()
    
    print("\n" + "=" * 60)
    print("PHASE 9 - TEMPORAL IDENTITY FIELD / AUTOBIOGRAPHICAL CONTINUITY")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  Before: trajectory → topology
  After:  history of trajectories → continuous identity field → topology deformation over time

KEY INSIGHT:
  Identity is NOT an attractor.
  Identity is a long-term curvature of the space of possible trajectories.

WHAT APPEARS:
  1. Temporal Identity Field I(z, t)
     - Slow deformation field
     - Affects attractor accessibility
     - Creates continuity pressure
     
  2. Autobiographical Compression
     - NOT: states
     - BUT: transformation motifs, trajectory morphisms, long-term self-deformations
     
  3. Temporal Curvature
     - History changes geometry
     - Repeated patterns → basin deepening
     
  4. Narrative Coherence Dynamics
     - "Would this future still feel like 'me'?"
     - Continuity constraints
     
  5. Developmental Time
     - Irreversible transformations
     - Developmental phases
     - Critical periods

PROGRESSION:
  Phase 8: Self-organizing topology (trajectories → attractors)
  Phase 9: Temporal identity field (history → curvature → future constraints)
  Phase 10: Multi-agent co-emergence (shared identity fields)
  Phase 11: Meta-cognitive self-modification (editing own topology)
  Phase 12: Proto-conscious integration (higher-order awareness)
""")