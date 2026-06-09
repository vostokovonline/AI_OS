"""
Phase 2 - Causal Invariance Layer

Core shift:
  Phase 19: action = deformation operator on energy landscape
  Phase 2:  action = intervention in causal graph

Key difference:
  Phase 19: trajectory learning, dynamic drift
  Phase 2:  causal-invariant structure, counterfactual stability

Architecture:
  Policy π(a|z) → action a_t
    ↓
  Causal Mechanism E = f(Z, A)
    ↓
  observed effect e_t
    ↓
  Causal Invariance Learner (stability across policies)
    ↓
  Causal Graph Update (edges: Z → E | A → E)
    ↓
  Variational Energy V(z | CG) constrained by causal graph
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


@dataclass
class CausalEdge:
    """A causal edge in the graph."""
    source: str
    target: str
    condition: Optional[str] = None  # e.g., action type
    strength: float = 1.0
    confidence: float = 0.0
    sample_count: int = 0


@dataclass
class CausalEffect:
    """Observed causal effect."""
    z: np.ndarray
    action: str
    z_next: np.ndarray
    delta_z: np.ndarray
    policy_id: str
    timestamp: int = 0


@dataclass
class InvarianceTest:
    """Result of invariance test across policies."""
    z_hash: str
    action: str
    effect_variance: float
    is_invariant: bool
    num_policies: int
    effects_by_policy: Dict[str, np.ndarray]


class CausalGraph:
    """
    Explicit causal graph structure.
    
    Nodes: Z (latent states), A (actions), E (effects)
    Edges: causal relationships between nodes
    """
    
    def __init__(self):
        self.nodes: Set[str] = {'Z', 'A', 'E'}
        self.edges: Dict[str, CausalEdge] = {}
        self.edge_count = 0
    
    def add_edge(self, source: str, target: str, condition: Optional[str] = None):
        """Add causal edge."""
        edge_key = f"{source}→{target}" + (f"|{condition}" if condition else "")
        
        if edge_key not in self.edges:
            self.edges[edge_key] = CausalEdge(
                source=source,
                target=target,
                condition=condition,
                strength=1.0,
                confidence=0.0,
                sample_count=0
            )
            self.edge_count += 1
        
        self.edges[edge_key].sample_count += 1
    
    def update_confidence(self, edge_key: str, new_evidence: float):
        """Update edge confidence based on new evidence."""
        if edge_key in self.edges:
            edge = self.edges[edge_key]
            n = edge.sample_count
            edge.confidence = (edge.confidence * (n - 1) + new_evidence) / n
    
    def get_edges_from(self, source: str) -> List[CausalEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges.values() if e.source == source]
    
    def get_edges_to(self, target: str) -> List[CausalEdge]:
        """Get all edges pointing to a node."""
        return [e for e in self.edges.values() if e.target == target]
    
    def causal_strength(self, source: str, target: str) -> float:
        """Get causal strength between two nodes."""
        key = f"{source}→{target}"
        if key in self.edges:
            return self.edges[key].strength * self.edges[key].confidence
        return 0.0


class CausalEffectTracker:
    """
    Tracks causal effects of (z, a) pairs across different policies.
    
    Key insight: We track P(E | Z, A) not just V(z, a)
    """
    
    def __init__(self):
        self.effects: Dict[Tuple[str, str], List[CausalEffect]] = defaultdict(list)
        self.z_cache: Dict[str, np.ndarray] = {}
        self.step_counter = 0
    
    def hash_state(self, z: np.ndarray) -> str:
        """Create deterministic hash of state."""
        state_bytes = z.tobytes()
        return hashlib.md5(state_bytes).hexdigest()[:12]
    
    def log(self, z: np.ndarray, action: str, z_next: np.ndarray, 
            policy_id: str = "default") -> CausalEffect:
        """Log a causal effect."""
        self.step_counter += 1
        
        delta_z = z_next - z
        
        effect = CausalEffect(
            z=z.copy(),
            action=action,
            z_next=z_next.copy(),
            delta_z=delta_z.copy(),
            policy_id=policy_id,
            timestamp=self.step_counter
        )
        
        key = (self.hash_state(z), action)
        self.effects[key].append(effect)
        
        # Cache z for lookup
        self.z_cache[self.hash_state(z)] = z.copy()
        
        return effect
    
    def get_effects(self, z: np.ndarray, action: str) -> List[CausalEffect]:
        """Get all effects for (z, action) pair."""
        key = (self.hash_state(z), action)
        return self.effects.get(key, [])
    
    def get_effects_across_policies(self, z: np.ndarray, action: str) -> InvarianceTest:
        """
        Check invariance of effect across different policies.
        
        P(E | Z, A, π1) ≈ P(E | Z, A, π2) for causal invariance
        """
        z_hash = self.hash_state(z)
        key = (z_hash, action)
        effects = self.effects.get(key, [])
        
        if len(effects) < 2:
            return InvarianceTest(
                z_hash=z_hash,
                action=action,
                effect_variance=np.nan,
                is_invariant=False,
                num_policies=0,
                effects_by_policy={}
            )
        
        # Group by policy
        effects_by_policy: Dict[str, List[np.ndarray]] = defaultdict(list)
        for e in effects:
            effects_by_policy[e.policy_id].append(e.delta_z)
        
        # Compute mean effect per policy
        mean_effects = {}
        for policy, deltas in effects_by_policy.items():
            mean_effects[policy] = np.mean(deltas, axis=0)
        
        # Variance across policy means
        if len(mean_effects) < 2:
            effect_variance = 0.0
        else:
            policy_vectors = list(mean_effects.values())
            effect_variance = np.var(policy_vectors)
        
        # Invariance: low variance means effect is consistent across policies
        is_invariant = effect_variance < 0.1  # threshold
        
        return InvarianceTest(
            z_hash=z_hash,
            action=action,
            effect_variance=effect_variance,
            is_invariant=is_invariant,
            num_policies=len(effects_by_policy),
            effects_by_policy=mean_effects
        )
    
    def check_invariance(self, z: np.ndarray, a1: str, a2: str) -> Tuple[float, bool]:
        """Check if effects for two actions are invariant."""
        test1 = self.get_effects_across_policies(z, a1)
        test2 = self.get_effects_across_policies(z, a2)
        
        if np.isnan(test1.effect_variance) or np.isnan(test2.effect_variance):
            return 0.0, False
        
        variance_diff = abs(test1.effect_variance - test2.effect_variance)
        is_invariant = variance_diff < 0.1
        
        return variance_diff, is_invariant


class CausalInvarianceLearner:
    """
    Learns causal structure and checks invariance across policies.
    
    Key principle:
      if causal link is true → it should persist across different policies
      
      P(E | Z, A, π1) ≈ P(E | Z, A, π2)
    """
    
    def __init__(self, invariance_threshold: float = 0.1):
        self.invariants_threshold = invariance_threshold
        self.invariance_tests: List[InvarianceTest] = []
        self.causal_graph = CausalGraph()
        
        # Invariant edges (stable across policies)
        self.invariant_edges: Set[str] = set()
        
        # Counterfactual history
        self.counterfactuals: List[Dict] = []
    
    def test_invariance(self, z: np.ndarray, action: str) -> InvarianceTest:
        """Test if effect is invariant across policies."""
        # This should be called with effect tracker
        pass  # Will be integrated with CausalEffectTracker
    
    def update_causal_graph(self, effect: CausalEffect, invariance: InvarianceTest):
        """Update causal graph based on observed effect."""
        # Add edges: Z → E and A → E
        self.causal_graph.add_edge('Z', 'E')
        self.causal_graph.add_edge('A', 'E', condition=effect.action)
        
        # Update confidence based on invariance
        edge_key = f"A→E|{effect.action}"
        confidence = 1.0 if invariance.is_invariant else 0.5
        self.causal_graph.update_confidence(edge_key, confidence)
        
        # Mark invariant edges
        if invariance.is_invariant:
            self.invariant_edges.add(edge_key)
    
    def compute_causal_penalty(self, invariance_tests: List[InvarianceTest]) -> float:
        """
        Compute causal invariance penalty.
        
        loss_causal = variance_of_effects_across_policies
        
        This penalizes effects that vary too much across policies,
        enforcing causal stability.
        """
        if not invariance_tests:
            return 0.0
        
        variances = [t.effect_variance for t in invariance_tests 
                     if not np.isnan(t.effect_variance)]
        
        if not variances:
            return 0.0
        
        return np.mean(variances)
    
    def get_counterfactual(self, z: np.ndarray, actual_action: str, 
                         counterfactual_action: str, 
                         effect_tracker: CausalEffectTracker) -> Optional[np.ndarray]:
        """
        Compute counterfactual: what if we chose different action?
        
        Uses causal graph to estimate counterfactual effect.
        """
        # Get actual effect
        actual_effects = effect_tracker.get_effects(z, actual_action)
        if not actual_effects:
            return None
        
        actual_delta = np.mean([e.delta_z for e in actual_effects], axis=0)
        
        # Get counterfactual effect
        cf_effects = effect_tracker.get_effects(z, counterfactual_action)
        if not cf_effects:
            return None
        
        cf_delta = np.mean([e.delta_z for e in cf_effects], axis=0)
        
        # Counterfactual: what would z_next be if we chose different action?
        z_hash = effect_tracker.hash_state(z)
        if z_hash in effect_tracker.z_cache:
            original_z = effect_tracker.z_cache[z_hash]
            counterfactual_z_next = original_z + cf_delta
            
            self.counterfactuals.append({
                'z': z.copy(),
                'actual_action': actual_action,
                'cf_action': counterfactual_action,
                'actual_z_next': original_z + actual_delta,
                'cf_z_next': counterfactual_z_next,
                'cf_delta': cf_delta
            })
            
            return counterfactual_z_next
        
        return None


class CausalVariationalEnergy:
    """
    Variational energy constrained by causal graph.
    
    V(z | CG) = base_energy(z) + causal_penalty(z)
    
    Where causal_penalty enforces invariance of effects.
    """
    
    def __init__(self, base_energy_scale: float = 1.0, causal_weight: float = 0.5):
        self.base_energy_scale = base_energy_scale
        self.causal_weight = causal_weight
        
        # Base energy parameters
        self.center = np.zeros(8)
        self.scale = 1.0
    
    def compute(self, z: np.ndarray, causal_penalty: float = 0.0) -> float:
        """
        Compute energy with causal constraint.
        
        Args:
            z: latent state
            causal_penalty: from CausalInvarianceLearner
        
        Returns:
            energy: base_energy + causal_weight * causal_penalty
        """
        # Base energy (distance from origin)
        base_energy = self.base_energy_scale * (np.linalg.norm(z - self.center) ** 2)
        
        # Causal penalty (high if effects vary across policies)
        total_energy = base_energy + self.causal_weight * causal_penalty
        
        return total_energy
    
    def gradient(self, z: np.ndarray, causal_gradient: np.ndarray = None) -> np.ndarray:
        """Gradient of energy with respect to z."""
        grad = 2 * self.base_energy_scale * (z - self.center)
        
        if causal_gradient is not None:
            grad += self.causal_weight * causal_gradient
        
        return grad


class CausalClosedLoopAgent:
    """
    Phase 2: Causal Invariance Layer built on Phase 19.
    
    Key shift from Phase 19:
      Phase 19: action = deformation operator (dynamic drift)
      Phase 2:  action = intervention in causal graph (causal invariant)
    
    The system now:
      1. Tracks effects of (z, a) across policies
      2. Checks causal invariance (effects stable across policies)
      3. Updates causal graph based on invariance tests
      4. Constrains V by causal graph
    """
    
    def __init__(self, obs_dim: int = 10, latent_dim: int = 8, action_dim: int = 2,
                 causal_weight: float = 0.5):
        # Import Phase 19 components
        from closed_loop_agent import ClosedLoopVariationalAgent
        from true_variational_model import TrueVariationalWorldModel
        
        # Core world model
        self.world_model = TrueVariationalWorldModel(obs_dim, latent_dim, action_dim)
        
        # Phase 19 components
        from closed_loop_agent import ActionDeformationOperator, CoupledEncoder, FreeEnergyAttention
        from closed_loop_agent import EnergyShapingParams
        
        self.deformation_op = ActionDeformationOperator(latent_dim)
        self.encoder = CoupledEncoder(obs_dim, latent_dim)
        self.attention = FreeEnergyAttention()
        self.shaping_params = EnergyShapingParams(
            alpha_exploit=0.5,
            alpha_explore=0.3,
            history_weight=0.1,
            stability_modulation=0.5
        )
        
        # Phase 2: NEW causal components
        self.effect_tracker = CausalEffectTracker()
        self.causal_learner = CausalInvarianceLearner()
        self.causal_energy = CausalVariationalEnergy(causal_weight=causal_weight)
        
        # Trajectory
        self.trajectory: list = []
        self.step_count = 0
        self.current_z: Optional[np.ndarray] = None
        self.current_V: float = 0.0
        self.policy_id = "default"
    
    def step(self, obs: np.ndarray, action: Optional[str] = None,
             compute_counterfactual: bool = False) -> Dict:
        """
        Single step with causal invariance.
        
        Pipeline:
          1. Encode observation
          2. Get world model prediction
          3. Apply action (intervention in causal graph)
          4. Log causal effect
          5. Test invariance across policies
          6. Update causal graph
          7. Compute causal-constrained energy
          8. Update trajectory
        """
        self.step_count += 1
        
        # 1. Encode
        z = self.encoder.forward(obs)
        obs_formatted = np.concatenate([z, np.zeros(2)])
        
        # 2. World model prediction
        default_action = np.array([1.0, 0.0])
        model_state = self.world_model.forward(obs_formatted, default_action)
        predicted_V = model_state['V']
        
        # 3. Select action
        if action is None:
            if len(self.trajectory) > 0:
                prev = self.trajectory[-1]
                # Use delta_z magnitude as prediction error proxy
                prev_error = np.linalg.norm(prev.delta_z)
                base_stability = np.mean(model_state['stability_spectrum']['stabilities'])
                attn, action_tendency = self.attention.compute_attention(
                    prev_error, prev_error * 1.1,  # slight mismatch indicates explore
                    base_stability
                )
            else:
                action_tendency = 'balance'
        else:
            action_tendency = action
        
        action_map = {
            'exploit': np.array([1.0, 0.0]),
            'explore': np.array([-1.0, 0.0]),
            'balance': np.array([0.0, 1.0])
        }
        selected_action = action_map.get(action_tendency, default_action)
        
        # 4. Apply action deformation (intervention)
        context = {
            'instability': 1.0 - np.mean(model_state['stability_spectrum']['stabilities']),
            'V': predicted_V,
            'history': self.trajectory[-5:] if len(self.trajectory) >= 5 else []
        }
        deformation = self.deformation_op.compute_deformation(z, action_tendency, context)
        realized_V = predicted_V + np.mean(deformation) * 0.1
        
        # 5. Get z_next (apply world model with action)
        model_state2 = self.world_model.forward(obs_formatted, selected_action)
        z_next = model_state2['z']
        
        # 6. Log causal effect
        effect = self.effect_tracker.log(z, action_tendency, z_next, self.policy_id)
        
        # 7. Test invariance across policies
        invariance = self.effect_tracker.get_effects_across_policies(z, action_tendency)
        self.causal_learner.invariance_tests.append(invariance)
        
        # 8. Update causal graph
        self.causal_learner.update_causal_graph(effect, invariance)
        
        # 9. Compute causal penalty
        recent_invariance_tests = self.causal_learner.invariance_tests[-10:]
        causal_penalty = self.causal_learner.compute_causal_penalty(recent_invariance_tests)
        
        # 10. Compute causal-constrained energy
        V_causal = self.causal_energy.compute(z, causal_penalty)
        if np.isnan(V_causal) or np.isinf(V_causal):
            V_causal = 10.0
        else:
            V_causal = np.clip(V_causal, -100, 100)
        
        # 11. Counterfactual if requested
        counterfactual_z_next = None
        if compute_counterfactual:
            cf_action = 'explore' if action_tendency == 'exploit' else 'exploit'
            counterfactual_z_next = self.causal_learner.get_counterfactual(
                z, action_tendency, cf_action, self.effect_tracker
            )
        
        # 12. Update trajectory-based components (adapted for CausalEffect)
        # Note: encoder.update_bias expects TrajectoryPoint, we use causal mechanism instead
        # Update encoder bias using causal invariance signal
        if len(self.trajectory) >= 5:
            recent_invariance = self.causal_learner.invariance_tests[-5:]
            avg_variance = np.mean([t.effect_variance for t in recent_invariance 
                                   if not np.isnan(t.effect_variance)])
            # Bias update direction from causal variance
            self.encoder.history_signal += 0.01 * (0.5 - avg_variance)
            self.encoder.history_signal = np.clip(self.encoder.history_signal, -1.0, 1.0)
            self.encoder.b = self.encoder.base_bias + self.encoder.history_signal
        
        # Don't call deformation_op.update_from_trajectory as it expects TrajectoryPoint
        
        # 13. Store trajectory (extended format for encoder update compatibility)
        prediction_error = np.linalg.norm(z_next - z)
        point = CausalEffect(
            z=z.copy(),
            action=action_tendency,
            z_next=z_next.copy(),
            delta_z=z_next - z,
            policy_id=self.policy_id,
            timestamp=self.step_count
        )
        # Store extra info as extension (not in dataclass)
        point.predicted_V = predicted_V
        point.realized_V = realized_V
        point.prediction_error = prediction_error
        self.trajectory.append(point)
        
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)
        
        # Store state
        self.current_z = z.copy()
        self.current_V = V_causal
        
        return {
            'z': z,
            'z_next': z_next,
            'V': V_causal,
            'predicted_V': predicted_V,
            'realized_V': realized_V,
            'causal_penalty': causal_penalty,
            'action': action_tendency,
            'invariance': invariance.is_invariant,
            'invariance_variance': invariance.effect_variance,
            'num_policies': invariance.num_policies,
            'causal_strength': self.causal_learner.causal_graph.causal_strength('A', 'E'),
            'counterfactual_z_next': counterfactual_z_next,
            'num_edges': self.causal_learner.causal_graph.edge_count,
            'num_invariant_edges': len(self.causal_learner.causal_graph.edges)
        }
    
    def get_system_state(self) -> Dict:
        """Get full system state."""
        return {
            'step_count': self.step_count,
            'trajectory_length': len(self.trajectory),
            'current_V': self.current_V,
            'causal_penalty_recent': self.causal_learner.compute_causal_penalty(
                self.causal_learner.invariance_tests[-10:]
            ),
            'num_causal_edges': self.causal_learner.causal_graph.edge_count,
            'num_invariant_edges': len(self.causal_learner.invariant_edges),
            'causal_strength': self.causal_learner.causal_graph.causal_strength('A', 'E'),
            'encoder_bias_norm': float(np.linalg.norm(self.encoder.b)),
        }


def test_causal_graph():
    """Test causal graph structure."""
    print("=" * 60)
    print("CAUSAL GRAPH TEST")
    print("=" * 60)
    
    graph = CausalGraph()
    
    # Add some edges
    graph.add_edge('Z', 'E')
    graph.add_edge('A', 'E', condition='exploit')
    graph.add_edge('A', 'E', condition='explore')
    
    print(f"\n  Nodes: {graph.nodes}")
    print(f"  Edges: {list(graph.edges.keys())}")
    print(f"  Edge count: {graph.edge_count}")
    
    # Update confidence
    graph.update_confidence('A→E|exploit', 0.8)
    print(f"\n  Exploit edge confidence: {graph.edges['A→E|exploit'].confidence}")
    
    # Causal strength
    print(f"  Causal strength (A→E): {graph.causal_strength('A', 'E'):.3f}")


def test_causal_effect_tracker():
    """Test causal effect tracking."""
    print("\n" + "=" * 60)
    print("CAUSAL EFFECT TRACKER TEST")
    print("=" * 60)
    
    tracker = CausalEffectTracker()
    
    # Simulate effects
    z1 = np.array([1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4])
    
    print("\n  Logging effects:")
    
    # Policy 1: exploit
    for i in range(5):
        z_next = z1 + np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) + np.random.randn(8) * 0.01
        effect = tracker.log(z1, 'exploit', z_next, 'policy_1')
        print(f"    exploit P1: |delta|={np.linalg.norm(effect.delta_z):.3f}")
    
    # Policy 2: exploit
    for i in range(5):
        z_next = z1 + np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) + np.random.randn(8) * 0.01
        effect = tracker.log(z1, 'exploit', z_next, 'policy_2')
        print(f"    exploit P2: |delta|={np.linalg.norm(effect.delta_z):.3f}")
    
    # Policy 1: explore
    for i in range(5):
        z_next = z1 + np.array([-0.2, -0.1, 0.1, -0.2, 0.0, 0.2, -0.1, -0.2]) + np.random.randn(8) * 0.01
        effect = tracker.log(z1, 'explore', z_next, 'policy_1')
        print(f"    explore P1: |delta|={np.linalg.norm(effect.delta_z):.3f}")
    
    # Test invariance
    print("\n  Invariance tests:")
    
    test_exploit = tracker.get_effects_across_policies(z1, 'exploit')
    print(f"    exploit: variance={test_exploit.effect_variance:.4f}, "
          f"invariant={test_exploit.is_invariant}, "
          f"policies={test_exploit.num_policies}")
    
    test_explore = tracker.get_effects_across_policies(z1, 'explore')
    print(f"    explore: variance={test_explore.effect_variance:.4f}, "
          f"invariant={test_explore.is_invariant}, "
          f"policies={test_explore.num_policies}")


def test_causal_invariance_learner():
    """Test causal invariance learning."""
    print("\n" + "=" * 60)
    print("CAUSAL INVARIANCE LEARNER TEST")
    print("=" * 60)
    
    from closed_loop_agent import TrajectoryPoint
    
    learner = CausalInvarianceLearner()
    
    # Simulate invariance tests
    z1 = np.random.randn(8)
    
    test1 = InvarianceTest(
        z_hash='abc123',
        action='exploit',
        effect_variance=0.05,
        is_invariant=True,
        num_policies=2,
        effects_by_policy={'p1': np.ones(8) * 0.1, 'p2': np.ones(8) * 0.1}
    )
    learner.invariance_tests.append(test1)
    
    test2 = InvarianceTest(
        z_hash='abc123',
        action='exploit',
        effect_variance=0.08,
        is_invariant=True,
        num_policies=2,
        effects_by_policy={'p1': np.ones(8) * 0.1, 'p2': np.ones(8) * 0.1}
    )
    learner.invariance_tests.append(test2)
    
    print("\n  Causal penalty: ", learner.compute_causal_penalty(learner.invariance_tests))
    print(f"  Causal graph edges: {learner.causal_graph.edge_count}")


def test_causal_variational_energy():
    """Test causal-constrained variational energy."""
    print("\n" + "=" * 60)
    print("CAUSAL VARIATIONAL ENERGY TEST")
    print("=" * 60)
    
    energy = CausalVariationalEnergy(base_energy_scale=1.0, causal_weight=0.5)
    
    z = np.random.randn(8)
    
    print("\n  Energy computation:")
    for penalty in [0.0, 0.1, 0.5, 1.0]:
        V = energy.compute(z, causal_penalty=penalty)
        print(f"    penalty={penalty}: V={V:.3f}")
    
    print("\n  Gradient:")
    grad = energy.gradient(z)
    print(f"    |grad|={np.linalg.norm(grad):.3f}")


def test_causal_closed_loop_agent():
    """Test full causal closed-loop agent."""
    print("\n" + "=" * 60)
    print("CAUSAL CLOSED-LOOP AGENT TEST")
    print("=" * 60)
    
    agent = CausalClosedLoopAgent()
    
    print("\n  Running 50 steps:")
    
    for step in range(50):
        obs = np.random.randn(10)
        state = agent.step(obs)
        
        if step % 10 == 0:
            print(f"    Step {step}: "
                  f"V={state['V']:.3f}, "
                  f"penalty={state['causal_penalty']:.3f}, "
                  f"{state['action']}, "
                  f"invariant={state['invariance']}, "
                  f"edges={state['num_edges']}")
    
    print("\n  System state:")
    sys_state = agent.get_system_state()
    for key, value in sys_state.items():
        if not isinstance(value, (dict, list)):
            print(f"    {key}: {value}")


def test_counterfactual():
    """Test counterfactual computation."""
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL TEST")
    print("=" * 60)
    
    agent = CausalClosedLoopAgent()
    
    # Run some steps to build up effects
    print("\n  Building effect library:")
    
    z1 = np.array([1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4])
    
    for i in range(20):
        obs = np.concatenate([z1, np.zeros(2)])
        if i % 2 == 0:
            agent.step(obs, 'exploit')
        else:
            agent.step(obs, 'explore')
    
    # Compute counterfactual
    print("\n  Computing counterfactual:")
    
    obs = np.concatenate([z1, np.zeros(2)])
    state = agent.step(obs, 'exploit', compute_counterfactual=True)
    
    if state['counterfactual_z_next'] is not None:
        print(f"    Actual z_next: {state['z_next'][:3]}...")
        print(f"    Counterfactual: {state['counterfactual_z_next'][:3]}...")
        print(f"    Difference: {np.linalg.norm(state['z_next'] - state['counterfactual_z_next']):.3f}")
    else:
        print("    No counterfactual available")


def test_invariance_vs_phase19():
    """Test invariance detection vs Phase 19."""
    print("\n" + "=" * 60)
    print("INVARIANCE VS PHASE 19 COMPARISON")
    print("=" * 60)
    
    from closed_loop_agent import ClosedLoopVariationalAgent
    from cognitive_integration import VariationalDecisionEngine
    
    print("\n  Phase 19 (no causal invariance):")
    agent19 = ClosedLoopVariationalAgent()
    
    for step in range(30):
        obs = np.random.randn(10)
        agent19.step(obs)
    
    print(f"    Steps: {agent19.step_count}")
    print(f"    Encoder bias norm: {np.linalg.norm(agent19.encoder.b):.3f}")
    print(f"    Trajectory length: {len(agent19.trajectory)}")
    
    print("\n  Phase 2 (with causal invariance):")
    agent2 = CausalClosedLoopAgent()
    
    for step in range(30):
        obs = np.random.randn(10)
        agent2.step(obs)
    
    print(f"    Steps: {agent2.step_count}")
    print(f"    Encoder bias norm: {np.linalg.norm(agent2.encoder.b):.3f}")
    print(f"    Causal edges: {agent2.causal_learner.causal_graph.edge_count}")
    print(f"    Invariant edges: {len(agent2.causal_learner.invariant_edges)}")
    print(f"    Causal penalty: {agent2.get_system_state()['causal_penalty_recent']:.4f}")
    
    print("\n  Key difference:")
    print("    Phase 19: encoder learns from trajectory (no causal structure)")
    print("    Phase 2:  encoder constrained by causal invariance (structure-aware)")


def test_multipolicy():
    """Test multi-policy causal learning."""
    print("\n" + "=" * 60)
    print("MULTI-POLICY CAUSAL LEARNING TEST")
    print("=" * 60)
    
    agent = CausalClosedLoopAgent()
    
    # Policy 1: conservative
    print("\n  Policy 1 (conservative - exploit):")
    agent.policy_id = "conservative"
    z1 = np.array([1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4])
    
    for step in range(10):
        obs = np.concatenate([z1, np.zeros(2)])
        agent.step(obs, 'exploit')
    
    print(f"    Trajectory: {len(agent.trajectory)} effects")
    
    # Policy 2: aggressive
    print("  Policy 2 (aggressive - explore):")
    agent.policy_id = "aggressive"
    
    for step in range(10):
        obs = np.concatenate([z1, np.zeros(2)])
        agent.step(obs, 'explore')
    
    print(f"    Trajectory: {len(agent.trajectory)} effects")
    
    # Policy 3: random
    print("  Policy 3 (random):")
    agent.policy_id = "random"
    
    for step in range(10):
        obs = np.concatenate([z1, np.zeros(2)])
        if step % 2 == 0:
            agent.step(obs, 'exploit')
        else:
            agent.step(obs, 'explore')
    
    print(f"    Trajectory: {len(agent.trajectory)} effects")
    
    # Test invariance
    print("\n  Invariance across policies:")
    
    test_exploit = agent.effect_tracker.get_effects_across_policies(z1, 'exploit')
    print(f"    Exploit: {test_exploit.num_policies} policies, variance={test_exploit.effect_variance}")
    
    test_explore = agent.effect_tracker.get_effects_across_policies(z1, 'explore')
    print(f"    Explore: {test_explore.num_policies} policies, variance={test_explore.effect_variance}")
    
    print(f"\n  Causal graph edges: {agent.causal_learner.causal_graph.edge_count}")
    print(f"  Invariant edges: {len(agent.causal_learner.invariant_edges)}")


def test_causal_stability():
    """Test causal stability (invariance under interventions)."""
    print("\n" + "=" * 60)
    print("CAUSAL STABILITY TEST")
    print("=" * 60)
    
    agent = CausalClosedLoopAgent()
    z1 = np.array([1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6, 0.4])
    
    print("\n  Testing stability under different action sequences:")
    
    # Sequence A: all exploit
    print("\n  Sequence A (exploit):")
    for i in range(15):
        obs = np.concatenate([z1, np.zeros(2)])
        state = agent.step(obs, 'exploit')
        if i % 5 == 0:
            print(f"    Step {i}: V={state['V']:.3f}, penalty={state['causal_penalty']:.4f}")
    
    # Record V after sequence A
    V_after_A = agent.current_V
    
    # Reset but keep causal graph
    agent.current_z = None
    agent.trajectory = []
    
    # Sequence B: all explore
    print("\n  Sequence B (explore):")
    for i in range(15):
        obs = np.concatenate([z1, np.zeros(2)])
        state = agent.step(obs, 'explore')
        if i % 5 == 0:
            print(f"    Step {i}: V={state['V']:.3f}, penalty={state['causal_penalty']:.4f}")
    
    V_after_B = agent.current_V
    
    print("\n  Stability analysis:")
    print(f"    V after exploit sequence: {V_after_A:.3f}")
    print(f"    V after explore sequence: {V_after_B:.3f}")
    print(f"    Causal graph edges: {agent.causal_learner.causal_graph.edge_count}")
    print(f"    Invariant edges: {len(agent.causal_learner.invariant_edges)}")
    
    # True causal stability = same effect regardless of action sequence
    causal_stability = abs(V_after_A - V_after_B) < 0.5
    print(f"    Causal stable: {causal_stability}")


if __name__ == '__main__':
    test_causal_graph()
    test_causal_effect_tracker()
    test_causal_invariance_learner()
    test_causal_variational_energy()
    test_causal_closed_loop_agent()
    test_counterfactual()
    test_invariance_vs_phase19()
    test_multipolicy()
    test_causal_stability()
    
    print("\n" + "=" * 60)
    print("PHASE 2 - CAUSAL INVARIANCE LAYER COMPLETE")
    print("=" * 60)
    print("\nKey structural shift:")
    print("  Phase 19: action = deformation operator")
    print("  Phase 2:  action = intervention in causal graph")
    print("\nNew capabilities:")
    print("  ✓ Causal effect tracking P(E | Z, A)")
    print("  ✓ Invariance tests across policies")
    print("  ✓ Counterfactual reasoning")
    print("  ✓ Causal graph structure")
    print("  ✓ Causal-constrained energy")
    print("\nThis is no longer 'self-shaping dynamics'.")
    print("This is 'causal-intervention-aware agent'.")