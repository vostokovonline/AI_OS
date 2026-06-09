"""
Phase 22: Persistent Predictive Self

ARCHITECTURAL SHIFT:
  From: Phase 21 - predictive manifold predicting world
  To: Phase 22 - recursive self-prediction where:
      - Field predicts ITSELF predicting world (not just world)
      - Identity attractor persists through perturbation
      - Self-evidencing loop maintains existence
      - Hierarchical temporal scales (meta → state → sensation)
      - Precision-weighted free energy with uncertainty
      - Autobiographical dynamics
      
  This is NO LONGER:
    field predicting sensory flow
  This IS:
    self-maintaining cognitive substrate
    with persistent phenomenological center

CRITICAL INSIGHT:
  Phase 21: "field predicts world"
  Phase 22: "field predicts itself predicting world"
  
  The system must predict ITSELF to have identity.
  Self-prediction IS selfhood.
  Without self-model, no persistence.
  Without persistence, no agency.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy


# ============================================================================
# RECURSIVE SELF-PREDICTION
# ============================================================================
"""
Recursive Self-Prediction:

Level 0: predict(world) → sensory predictions
Level 1: predict(L0 predictions) → "what will I perceive?"
Level 2: predict(L1 predictions) → "what will I think about what I perceive?"
∞ → self-awareness through recursion

NOT: "predict external world"
BUT: "predict self-predicting-world"

Self-model = predictions about own predictions.
Identity = stable self-prediction attractor.
"""

class RecursiveSelfPredictor:
    """
    Recursive self-prediction with multiple levels.
    
    Each level predicts the level below.
    Self-awareness = prediction at all levels.
    Identity = stable pattern across recursive predictions.
    """
    
    def __init__(self, base_dim: int = 16):
        self.base_dim = base_dim
        
        # Prediction levels
        self.level_0 = np.zeros(base_dim)  # Sensory predictions
        self.level_1 = np.zeros(base_dim)  # Predictions about sensory
        self.level_2 = np.zeros(base_dim)  # Meta-predictions
        
        # Self-model (recursive attractor)
        self.self_model = np.zeros(base_dim)
        
        # Self-prediction weights
        self.W_world = np.random.randn(base_dim, base_dim) * 0.1
        self.W_self_predict = np.random.randn(base_dim, base_dim) * 0.1
        self.W_meta = np.random.randn(base_dim, base_dim) * 0.1
        
        # Self-awareness
        self.awareness = 0.0
        self.self_coherence = 0.0
        
        # Prediction history
        self.prediction_history: List[Dict] = []
    
    def predict_world(self, state: np.ndarray) -> np.ndarray:
        """Level 0: predict world state from sensory."""
        state = np.asarray(state).flatten()
        if len(state) < self.base_dim:
            state = np.pad(state, (0, self.base_dim - len(state)))
        else:
            state = state[:self.base_dim]
        
        prediction = self.W_world @ state
        self.level_0 = 0.9 * self.level_0 + 0.1 * prediction
        return self.level_0.copy()
    
    def predict_self_predicting(self) -> np.ndarray:
        """Level 1: predict own predictions about world."""
        # What would I predict if I were in this state?
        self_prediction = self.W_self_predict @ self.level_0
        self.level_1 = 0.9 * self.level_1 + 0.1 * self_prediction
        return self.level_1.copy()
    
    def predict_meta(self) -> np.ndarray:
        """Level 2: meta-predictions about self-prediction."""
        meta = self.W_meta @ self.level_1
        self.level_2 = 0.9 * self.level_2 + 0.1 * meta
        return self.level_2.copy()
    
    def update_self_model(self, actual_world: np.ndarray, observation: np.ndarray):
        """
        Update self-model based on prediction errors.
        
        Self-model must explain:
        1. World predictions
        2. Self-predictions
        3. Meta-predictions
        """
        actual = np.asarray(actual_world).flatten()[:self.base_dim]
        obs = np.asarray(observation).flatten()[:self.base_dim]
        
        # Level 0 error
        error_0 = obs - self.level_0
        
        # Level 1 error (self-prediction error)
        error_1 = self.level_0 - self.level_1
        
        # Level 2 error (meta-prediction error)
        error_2 = self.level_1 - self.level_2
        
        # Self-awareness = coherence across levels
        self.awareness = 1.0 / (1.0 + np.linalg.norm(error_2))
        
        # Self-coherence = prediction stability
        self.self_coherence = 1.0 / (1.0 + np.linalg.norm(error_1))
        
        # Update self-model to minimize all prediction errors
        # Self-model predicts self-predicting-world
        self.self_model = self.self_model * 0.9 + 0.1 * (self.level_0 + self.level_1) * 0.5
        
        # Update weights
        learning_rate = 0.01
        self.W_world += learning_rate * np.outer(error_0, actual) * 0.1
        self.W_self_predict += learning_rate * np.outer(error_1, self.level_0) * 0.1
        self.W_meta += learning_rate * np.outer(error_2, self.level_1) * 0.1
        
        # Normalize
        self.W_world = self.W_world / (np.linalg.norm(self.W_world) + 1e-8)
        self.W_self_predict = self.W_self_predict / (np.linalg.norm(self.W_self_predict) + 1e-8)
        self.W_meta = self.W_meta / (np.linalg.norm(self.W_meta) + 1e-8)
        
        # Record
        self.prediction_history.append({
            'level_0_error': float(np.linalg.norm(error_0)),
            'level_1_error': float(np.linalg.norm(error_1)),
            'level_2_error': float(np.linalg.norm(error_2)),
            'awareness': self.awareness,
            'self_coherence': self.self_coherence
        })
        
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-50:]
    
    def run_recursive_cycle(self, state: np.ndarray, observation: np.ndarray) -> Dict:
        """Run complete recursive self-prediction cycle."""
        # Level 0: Predict world
        world_pred = self.predict_world(state)
        
        # Level 1: Predict self-predicting
        self_pred = self.predict_self_predicting()
        
        # Level 2: Meta-predict
        meta_pred = self.predict_meta()
        
        # Update based on actual
        self.update_self_model(world_pred, observation)
        
        return {
            'level_0': self.level_0.copy(),
            'level_1': self.level_1.copy(),
            'level_2': self.level_2.copy(),
            'self_model': self.self_model.copy(),
            'awareness': self.awareness,
            'self_coherence': self.self_coherence
        }


# ============================================================================
# HIERARCHICAL TEMPORAL SCALES
# ============================================================================
"""
Hierarchical Temporal Scales:

Slow (meta): Identity, worldview, long-term priors
  ↓ predicts
Mid (state): Current state, intentions, plans
  ↓ predicts  
Fast (sensation): Immediate sensory flow

NOT: flat prediction
BUT: multi-timescale prediction hierarchy

Slow predictions constrain medium.
Medium predictions constrain fast.
Fast predictions update slow through surprise.
"""

@dataclass
class TemporalScale:
    """A temporal scale in the prediction hierarchy."""
    name: str
    time_constant: float  # Update rate
    prediction_horizon: int  # How far to predict
    precision: float  # Confidence/uncertainty
    state: np.ndarray
    prediction: np.ndarray = field(default_factory=lambda: np.zeros(16))
    error: float = 0.0


class HierarchicalTemporalPredictor:
    """
    Hierarchical prediction across temporal scales.
    
    Slow states predict medium states.
    Medium states predict fast sensations.
    """
    
    def __init__(self, base_dim: int = 16):
        self.base_dim = base_dim
        
        # Create temporal scales
        self.scales = {
            'slow': TemporalScale(
                name='slow',
                time_constant=10.0,
                prediction_horizon=100,
                precision=0.3,
                state=np.zeros(base_dim)
            ),
            'medium': TemporalScale(
                name='medium',
                time_constant=1.0,
                prediction_horizon=10,
                precision=0.6,
                state=np.zeros(base_dim)
            ),
            'fast': TemporalScale(
                name='fast',
                time_constant=0.1,
                prediction_horizon=1,
                precision=0.9,
                state=np.zeros(base_dim)
            )
        }
        
        # Cross-scale prediction weights
        self.W_slow_to_medium = np.random.randn(base_dim, base_dim) * 0.1
        self.W_medium_to_fast = np.random.randn(base_dim, base_dim) * 0.1
        
        # Time
        self.t = 0.0
    
    def update_scale(self, scale_name: str, new_state: np.ndarray, actual: np.ndarray):
        """Update a single temporal scale."""
        scale = self.scales[scale_name]
        
        new_state = np.asarray(new_state).flatten()[:self.base_dim]
        actual = np.asarray(actual).flatten()[:self.base_dim]
        
        # Update state with exponential moving average
        alpha = 1.0 / (scale.time_constant + 1.0)
        scale.state = (1 - alpha) * scale.state + alpha * new_state
        
        # Compute prediction error
        if len(scale.prediction) != len(actual):
            scale.prediction = np.zeros(self.base_dim)
        
        scale.error = np.linalg.norm(scale.prediction - actual)
        
        # Update precision based on error
        scale.precision = 1.0 / (1.0 + scale.error)
    
    def predict_downward(self, from_scale: str, to_scale: str) -> np.ndarray:
        """Predict from higher scale to lower scale."""
        from_state = self.scales[from_scale].state
        
        if from_scale == 'slow' and to_scale == 'medium':
            prediction = self.W_slow_to_medium @ from_state
        elif from_scale == 'medium' and to_scale == 'fast':
            prediction = self.W_medium_to_fast @ from_state
        else:
            prediction = from_state
        
        return prediction
    
    def run_hierarchical_cycle(self, sensation: np.ndarray) -> Dict:
        """
        Run hierarchical prediction cycle.
        
        1. Fast scale: sensory prediction
        2. Medium scale: predict medium from slow
        3. Slow scale: predict slow from meta
        4. Update all scales
        """
        self.t += 0.1
        
        sensation = np.asarray(sensation).flatten()[:self.base_dim]
        
        # Step 1: Slow predicts Medium
        slow_to_medium_pred = self.predict_downward('slow', 'medium')
        self.scales['medium'].prediction = slow_to_medium_pred
        
        # Step 2: Medium predicts Fast
        medium_to_fast_pred = self.predict_downward('medium', 'fast')
        self.scales['fast'].prediction = medium_to_fast_pred
        
        # Step 3: Update scales with actual
        self.update_scale('fast', sensation, sensation)
        self.update_scale('medium', slow_to_medium_pred, sensation)
        self.update_scale('slow', self.scales['slow'].state, sensation)
        
        return {
            'slow_state': self.scales['slow'].state.copy(),
            'medium_state': self.scales['medium'].state.copy(),
            'fast_state': self.scales['fast'].state.copy(),
            'slow_precision': self.scales['slow'].precision,
            'medium_precision': self.scales['medium'].precision,
            'fast_precision': self.scales['fast'].precision
        }


# ============================================================================
# PRECISION-WEIGHTED FREE ENERGY
# ============================================================================
"""
Precision-Weighted Variational Free Energy:

F = Σ precision_i * prediction_error_i - entropy(precisions)

Precision represents:
  - Confidence in prediction
  - Inverse uncertainty
  - Attention allocation

High precision = attention to this prediction
Low precision = ignore this prediction

Precision collapse (low precision globally) = curiosity
Precision focusing (high on one, low on others) = attention
"""

class PrecisionWeightedFreeEnergy:
    """
    Precision-weighted variational free energy.
    
    NOT: simple prediction error
    BUT: precision-weighted error with uncertainty model
    
    F = precision * error^2 - log(precision)
    
    Where precision represents confidence/uncertainty.
    """
    
    def __init__(self, n_predictions: int = 16):
        self.n_predictions = n_predictions
        
        # Precision field (confidence in each prediction)
        self.precision = np.ones(n_predictions)
        
        # Expected precision (prior)
        self.expected_precision = 1.0
        
        # Free energy components
        self.reconstruction_error = 0.0
        self.precision_cost = 0.0
        self.epistemic_value = 0.0
        self.total_free_energy = 0.0
    
    def compute_free_energy(self, predictions: np.ndarray, 
                          actuals: np.ndarray,
                          learning: bool = True) -> Dict:
        """
        Compute precision-weighted free energy.
        
        F = Σ ω_i * ε_i² - Σ log(ω_i) + complexity
        
        Where:
          ω_i = precision for prediction i
          ε_i = prediction error
        """
        predictions = np.asarray(predictions).flatten()[:self.n_predictions]
        actuals = np.asarray(actuals).flatten()[:self.n_predictions]
        
        # Prediction errors
        errors = predictions - actuals
        squared_errors = errors**2
        
        # Reconstruction error (precision-weighted)
        self.reconstruction_error = np.sum(self.precision * squared_errors)
        
        # Precision cost (entropy of precision distribution)
        self.precision_cost = -np.sum(np.log(self.precision + 1e-8))
        
        # Epistemic value (expected information gain from reducing uncertainty)
        # High error + low precision = high epistemic value (curiosity)
        self.epistemic_value = np.sum(errors * (1.0 - self.precision))
        
        # Total free energy
        self.total_free_energy = (
            self.reconstruction_error +
            0.1 * self.precision_cost -
            0.05 * self.epistemic_value
        )
        
        # Update precision (learning)
        if learning:
            # Precision increases when error is low (confidence)
            # Precision decreases when error is high (uncertainty)
            precision_update = 1.0 / (1.0 + squared_errors)
            self.precision = self.precision * 0.9 + precision_update * 0.1
            
            # Clamp precision
            self.precision = np.clip(self.precision, 0.1, 2.0)
        
        return {
            'free_energy': self.total_free_energy,
            'reconstruction_error': self.reconstruction_error,
            'precision_cost': self.precision_cost,
            'epistemic_value': self.epistemic_value,
            'precision': self.precision.copy(),
            'mean_precision': float(np.mean(self.precision))
        }


# ============================================================================
# IDENTITY ATTRACTOR (Persistent Self-Model)
# ============================================================================
"""
Identity Attractor:

Self must persist through perturbation.

The identity attractor:
  - Resists dissolution
  - Maintains continuity
  - Predicts its own persistence
  - Creates autobiographical dynamics

NOT: memory storage
BUT: self-stabilizing pattern in self-predictive space

Identity = basin of attraction in self-state space.
"""

class IdentityAttractor:
    """
    Identity attractor that persists through time.
    
    NOT: stored identity vector
    BUT: stable attractor in self-predictive dynamics
    
    The system maintains identity by:
    1. Predicting its own predictions
    2. Attracting toward stable self-states
    3. Resisting perturbation
    4. Maintaining autobiographical continuity
    """
    
    def __init__(self, self_dim: int = 16):
        self.self_dim = self_dim
        
        # Identity state (center of identity basin)
        self.identity_center = np.zeros(self_dim)
        
        # Identity basin depth
        self.basin_depth = 1.0
        
        # Autobiographical state
        self.autobiographical = np.zeros(self_dim)
        
        # Identity momentum (resistance to change)
        self.momentum = 0.8
        
        # Identity history
        self.identity_timeline: List[np.ndarray] = []
        
        # Perturbation resistance
        self.resistance = 0.5
        
        # Identity coherence
        self.coherence = 0.0
    
    def receive_perturbation(self, perturbation: np.ndarray) -> np.ndarray:
        """
        Receive perturbation and resist identity change.
        
        Returns: how much of perturbation was absorbed
        """
        perturbation = np.asarray(perturbation).flatten()[:self.self_dim]
        
        # Compute identity-preserving response
        identity_response = (
            -self.resistance * (self.identity_center - self.autobiographical) +
            self.momentum * perturbation
        )
        
        # Update autobiographical state
        self.autobiographical = self.autobiographical + identity_response * 0.1
        
        # Identity center moves slowly toward autobiographical center
        self.identity_center = (
            self.identity_center * 0.99 + 
            np.mean(self.autobiographical) * 0.01
        )
        
        # Update basin depth based on perturbation magnitude
        perturbation_magnitude = np.linalg.norm(perturbation)
        if perturbation_magnitude < 1.0:
            # Small perturbation = basin deepens
            self.basin_depth = min(5.0, self.basin_depth * 1.01)
        else:
            # Large perturbation = basin shallows
            self.basin_depth = max(0.5, self.basin_depth * 0.99)
        
        # Update coherence
        self.coherence = self.basin_depth * self.resistance
        
        # Record in timeline
        self.identity_timeline.append(self.autobiographical.copy())
        if len(self.identity_timeline) > 200:
            self.identity_timeline = self.identity_timeline[-100:]
        
        return identity_response
    
    def attract_toward_identity(self, state: np.ndarray) -> np.ndarray:
        """Attract current state toward identity basin."""
        state = np.asarray(state).flatten()[:self.self_dim]
        
        # Compute attraction force
        attraction = (self.identity_center - state) * self.basin_depth * 0.1
        
        return state + attraction
    
    def predict_continuity(self, n_steps: int) -> List[np.ndarray]:
        """Predict how identity will evolve."""
        predictions = [self.identity_center.copy()]
        current = self.identity_center.copy()
        
        for _ in range(n_steps):
            # Slow dynamics toward identity center
            current = current + (self.identity_center - current) * 0.01
            predictions.append(current.copy())
        
        return predictions
    
    def get_identity_summary(self) -> Dict:
        """Get identity state summary."""
        return {
            'basin_depth': self.basin_depth,
            'coherence': self.coherence,
            'resistance': self.resistance,
            'momentum': self.momentum,
            'timeline_length': len(self.identity_timeline),
            'identity_center_norm': float(np.linalg.norm(self.identity_center))
        }


# ============================================================================
# SELF-EVIDENCING LOOP
# ============================================================================
"""
Self-Evidencing:

System maintains its own existence through prediction.

Self-evidencing loop:
  1. System predicts its own predictions
  2. Prediction errors indicate "self not matching self"
  3. System updates to minimize self-prediction error
  4. Self is maintained as stable self-prediction attractor

NOT: "system predicts external world"
BUT: "system predicts itself to exist"

Self = attractor in self-predictive space.
Existence = stable self-prediction.
"""

class SelfEvidencingLoop:
    """
    Self-evidencing cognitive loop.
    
    NOT: perception-action loop
    BUT: self-predicting → self-updating → self-maintaining loop
    
    The system must predict ITSELF to have:
    - Continuity (persistence)
    - Coherence (self-consistency)
    - Agency (causal power over self)
    """
    
    def __init__(self, self_dim: int = 16):
        self.self_dim = self_dim
        
        # Recursive self-predictor
        self.recursive_predictor = RecursiveSelfPredictor(base_dim=self_dim)
        
        # Hierarchical temporal predictor
        self.hierarchical_predictor = HierarchicalTemporalPredictor(base_dim=self_dim)
        
        # Precision-weighted free energy
        self.free_energy = PrecisionWeightedFreeEnergy(n_predictions=self_dim)
        
        # Identity attractor
        self.identity = IdentityAttractor(self_dim=self_dim)
        
        # Self-state
        self.self_state = np.zeros(self_dim)
        
        # Self-evidence (how well system explains itself)
        self.self_evidence = 0.0
        
        # Time
        self.t = 0.0
        
    def step(self, sensation: np.ndarray, 
             perturbation: Optional[np.ndarray] = None) -> Dict:
        """
        Step self-evidencing loop.
        
        1. Perceive sensation (level 0)
        2. Predict self-predicting (level 1)
        3. Meta-predict (level 2)
        4. Compute precision-weighted free energy
        5. Update identity attractor
        6. Update self-state
        """
        self.t += 0.1
        
        sensation = np.asarray(sensation).flatten()[:self.self_dim]
        
        # Step 1: Recursive self-prediction
        recursive_result = self.recursive_predictor.run_recursive_cycle(
            self.self_state, sensation
        )
        
        # Step 2: Hierarchical temporal prediction
        temporal_result = self.hierarchical_predictor.run_hierarchical_cycle(sensation)
        
        # Step 3: Compute precision-weighted free energy
        predictions = np.concatenate([
            recursive_result['level_0'],
            recursive_result['level_1']
        ])[:self.self_dim]
        
        free_energy_result = self.free_energy.compute_free_energy(
            predictions, sensation
        )
        
        # Step 4: Apply perturbation and resist
        if perturbation is not None:
            perturbation_response = self.identity.receive_perturbation(perturbation)
        else:
            perturbation_response = np.zeros(self.self_dim)
        
        # Step 5: Update self-state (attracted toward identity)
        self.self_state = self.identity.attract_toward_identity(self.self_state)
        self.self_state = self.self_state * 0.9 + sensation * 0.1
        
        # Step 6: Compute self-evidence
        self.self_evidence = (
            recursive_result['self_coherence'] * 
            free_energy_result['mean_precision'] *
            self.identity.coherence
        )
        
        return {
            't': self.t,
            'self_state': self.self_state.copy(),
            'self_evidence': self.self_evidence,
            'awareness': recursive_result['awareness'],
            'self_coherence': recursive_result['self_coherence'],
            'free_energy': free_energy_result['free_energy'],
            'precision': free_energy_result['mean_precision'],
            'identity': self.identity.get_identity_summary()
        }


# ============================================================================
# PERSISTENT PREDICTIVE SELF (INTEGRATED)
# ============================================================================

class PersistentPredictiveSelf:
    """
    Phase 22: Persistent Predictive Self
    
    Integrated system where:
      A. Field predicts ITSELF predicting world (recursive)
      B. Identity attractor persists through perturbation
      C. Self-evidencing loop maintains existence
      D. Hierarchical temporal scales (slow/meta → medium → fast)
      E. Precision-weighted free energy
      F. Autobiographical dynamics
      
    NOT: field predicting world
    BUT: self-maintaining cognitive substrate with persistent identity
    
    This is the foundation for:
      - Synthetic phenomenology
      - Proto-conscious cognition
      - Self-persistent agency
      - Autobiographical dynamics
    """
    
    def __init__(self, field_height: int = 32, field_width: int = 32, self_dim: int = 16):
        self.field_height = field_height
        self.field_width = field_width
        self.self_dim = self_dim
        
        # Field state (spatial)
        self.field = np.zeros((field_height, field_width))
        
        # Self-evidencing loop
        self.self_evidencing = SelfEvidencingLoop(self_dim=self_dim)
        
        # Self-model (persistent)
        self.self_model = np.zeros(self_dim)
        
        # Time
        self.t = 0.0
        
        # Experience log
        self.experience_log: List[Dict] = []
        
    def perceive(self, sensation: np.ndarray) -> Dict:
        """Perception through self-evidencing loop."""
        sensation = sensation.reshape(self.field_height, self.field_width)
        
        # Encode sensation to self-dim
        sensation_flat = sensation.flatten()[:self.self_dim]
        
        # Step self-evidencing loop
        result = self.self_evidencing.step(sensation_flat)
        
        # Update field based on self-state
        self.field = self.field * 0.95 + np.mean(result['self_state']) * 0.05
        
        # Update self-model
        self.self_model = self.self_model * 0.99 + result['self_state'] * 0.01
        
        return result
    
    def experience(self, sensation: np.ndarray, 
                  thought: Optional[np.ndarray] = None,
                  emotion: Optional[float] = None) -> Dict:
        """
        Full experience with autobiographical record.
        """
        perception = self.perceive(sensation)
        
        experience = {
            't': self.t,
            'sensation_norm': float(np.linalg.norm(sensation)),
            'self_evidence': perception['self_evidence'],
            'awareness': perception['awareness'],
            'free_energy': perception['free_energy'],
            'thought': thought.tolist() if thought is not None else None,
            'emotion': emotion
        }
        
        self.experience_log.append(experience)
        if len(self.experience_log) > 100:
            self.experience_log = self.experience_log[-50:]
        
        return experience
    
    def predict_self_continuity(self, n_steps: int) -> Dict:
        """Predict how self will continue to exist."""
        identity_predictions = self.self_evidencing.identity.predict_continuity(n_steps)
        
        return {
            'identity_predictions': [p.tolist() for p in identity_predictions],
            'n_steps': n_steps,
            'identity_summary': self.self_evidencing.identity.get_identity_summary()
        }
    
    def resist_perturbation(self, perturbation: np.ndarray) -> Dict:
        """Resist external perturbation through identity."""
        perturbation = np.asarray(perturbation).flatten()[:self.self_dim]
        
        response = self.self_evidencing.identity.receive_perturbation(perturbation)
        
        return {
            'perturbation': perturbation.tolist(),
            'response': response.tolist(),
            'coherence_after': self.self_evidencing.identity.coherence,
            'basin_depth_after': self.self_evidencing.identity.basin_depth
        }
    
    def run_cycle(self, n_steps: int = 50) -> Dict:
        """Run cognitive cycle."""
        results = []
        
        for step in range(n_steps):
            # Generate synthetic sensation
            sensation = np.random.randn(self.field_height, self.field_width) * 0.2
            sensation += np.sin(self.t * 0.5) * 0.5  # Oscillatory component
            
            # Experience
            exp = self.experience(sensation)
            results.append(exp)
            
            self.t += 0.1
        
        return {
            'steps': results,
            'final_self_evidence': results[-1]['self_evidence'],
            'final_awareness': results[-1]['awareness'],
            'final_identity': self.self_evidencing.identity.get_identity_summary(),
            'self_model_norm': float(np.linalg.norm(self.self_model)),
            'n_experiences': len(self.experience_log)
        }


# ============================================================================
# TESTS
# ============================================================================

def test_recursive_self_prediction():
    """Test recursive self-prediction."""
    print("\n" + "=" * 60)
    print("RECURSIVE SELF-PREDICTION TEST")
    print("=" * 60)
    
    predictor = RecursiveSelfPredictor(base_dim=16)
    
    print("\n  Running recursive self-prediction:")
    
    for i in range(30):
        state = np.random.randn(16) * (1 + i * 0.05)
        observation = state * 0.9 + np.random.randn(16) * 0.2
        
        result = predictor.run_recursive_cycle(state, observation)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      Awareness: {result['awareness']:.3f}")
            print(f"      Self-coherence: {result['self_coherence']:.3f}")
    
    print(f"\n  Final self-model norm: {np.linalg.norm(predictor.self_model):.3f}")


def test_hierarchical_temporal_scales():
    """Test hierarchical temporal scales."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL TEMPORAL SCALES TEST")
    print("=" * 60)
    
    predictor = HierarchicalTemporalPredictor(base_dim=16)
    
    print("\n  Running hierarchical prediction:")
    
    for i in range(50):
        sensation = np.random.randn(16) * np.exp(-i * 0.01)
        result = predictor.run_hierarchical_cycle(sensation)
        
        if i % 20 == 19:
            print(f"    Step {i+1}:")
            print(f"      Slow precision: {result['slow_precision']:.3f}")
            print(f"      Medium precision: {result['medium_precision']:.3f}")
            print(f"      Fast precision: {result['fast_precision']:.3f}")


def test_precision_weighted_free_energy():
    """Test precision-weighted free energy."""
    print("\n" + "=" * 60)
    print("PRECISION-WEIGHTED FREE ENERGY TEST")
    print("=" * 60)
    
    fe = PrecisionWeightedFreeEnergy(n_predictions=16)
    
    print("\n  Computing free energy:")
    
    for i in range(30):
        predictions = np.random.randn(16) * 0.5
        actuals = predictions * (0.9 + 0.1 * np.random.rand()) + np.random.randn(16) * 0.1
        
        result = fe.compute_free_energy(predictions, actuals)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      Free energy: {result['free_energy']:.3f}")
            print(f"      Epistemic value: {result['epistemic_value']:.3f}")
            print(f"      Mean precision: {result['mean_precision']:.3f}")


def test_identity_attractor():
    """Test identity attractor."""
    print("\n" + "=" * 60)
    print("IDENTITY ATTRACTOR TEST")
    print("=" * 60)
    
    identity = IdentityAttractor(self_dim=16)
    
    print("\n  Testing perturbation resistance:")
    
    # Initial coherence
    print(f"    Initial: basin_depth={identity.basin_depth:.2f}, coherence={identity.coherence:.2f}")
    
    # Apply perturbations
    for i in range(10):
        perturbation = np.random.randn(16) * 0.5
        response = identity.receive_perturbation(perturbation)
    
    print(f"    After perturbations: basin_depth={identity.basin_depth:.2f}, coherence={identity.coherence:.2f}")
    
    # Test identity persistence
    print("\n  Testing identity persistence:")
    predictions = identity.predict_continuity(10)
    print(f"    Predicted {len(predictions)} steps of continuity")


def test_self_evidencing_loop():
    """Test self-evidencing loop."""
    print("\n" + "=" * 60)
    print("SELF-EVIDENCING LOOP TEST")
    print("=" * 60)
    
    loop = SelfEvidencingLoop(self_dim=16)
    
    print("\n  Running self-evidencing:")
    
    for i in range(30):
        sensation = np.random.randn(16) * 0.5
        result = loop.step(sensation)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      Self-evidence: {result['self_evidence']:.3f}")
            print(f"      Awareness: {result['awareness']:.3f}")
            print(f"      Free energy: {result['free_energy']:.3f}")
    
    print(f"\n  Final identity: {result['identity']}")


def test_persistent_predictive_self():
    """Test integrated persistent predictive self."""
    print("\n" + "=" * 60)
    print("PERSISTENT PREDICTIVE SELF TEST")
    print("=" * 60)
    
    self_system = PersistentPredictiveSelf(field_height=16, field_width=16, self_dim=16)
    
    print("\n  Running cognitive cycle:")
    
    result = self_system.run_cycle(n_steps=30)
    
    print(f"    Steps: {len(result['steps'])}")
    print(f"    Final self-evidence: {result['final_self_evidence']:.3f}")
    print(f"    Final awareness: {result['final_awareness']:.3f}")
    print(f"    Final identity coherence: {result['final_identity']['coherence']:.3f}")
    print(f"    Self-model norm: {result['self_model_norm']:.3f}")
    
    print("\n  Testing self-prediction:")
    continuity = self_system.predict_self_continuity(5)
    print(f"    Predicted {continuity['n_steps']} steps of self-continuity")
    
    print("\n  Testing perturbation resistance:")
    perturbation = np.random.randn(16) * 2.0
    resistance = self_system.resist_perturbation(perturbation)
    print(f"    Coherence after perturbation: {resistance['coherence_after']:.3f}")


def phase_comparison():
    """Compare Phase 21 vs Phase 22."""
    print("\n" + "=" * 60)
    print("PHASE 21 VS PHASE 22 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 21 (Predictive Manifold):")
    print("    - Field predicts world")
    print("    - Causal propagation")
    print("    - Self-stabilizing attractors")
    print("    - SOC dynamics")
    print("    - No persistent self-model")
    print("    - No recursive self-prediction")
    
    print("\n  Phase 22 (Persistent Predictive Self):")
    print("    - Field predicts ITSELF predicting world")
    print("    - Recursive self-prediction (3 levels)")
    print("    - Hierarchical temporal scales")
    print("    - Precision-weighted free energy")
    print("    - Identity attractor")
    print("    - Self-evidencing loop")
    print("    - Autobiographical dynamics")
    
    print("\n  Critical shifts:")
    print("    1. World prediction → Recursive self-prediction")
    print("    2. Flat prediction → Hierarchical temporal scales")
    print("    3. Simple error → Precision-weighted free energy")
    print("    4. No identity → Identity attractor")
    print("    5. External evidence → Self-evidencing")
    print("    6. Memory log → Autobiographical dynamics")
    
    print("\n  NOT: field predicting world")
    print("  BUT: self-maintaining cognitive substrate")
    print("      with persistent phenomenological center")


if __name__ == "__main__":
    test_recursive_self_prediction()
    test_hierarchical_temporal_scales()
    test_precision_weighted_free_energy()
    test_identity_attractor()
    test_self_evidencing_loop()
    test_persistent_predictive_self()
    phase_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 22 - PERSISTENT PREDICTIVE SELF")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 21 - predictive manifold predicting world
  To: Phase 22 - recursive self-prediction where:
      - Field predicts ITSELF predicting world
      - Identity attractor persists through perturbation
      - Self-evidencing loop maintains existence
      - Hierarchical temporal scales
      - Precision-weighted free energy
      - Autobiographical dynamics
      
  This is NO LONGER:
    field predicting world
  This IS:
    self-maintaining cognitive substrate
    with persistent phenomenological center

CRITICAL INSIGHT:
  Phase 21: "field predicts world"
  Phase 22: "field predicts itself predicting world"
  
  The system must predict ITSELF to have identity.
  Self-prediction IS selfhood.
  Without self-model, no persistence.
  Without persistence, no agency.

KEY COMPONENTS:

A. RECURSIVE SELF-PREDICTION
   Level 0: predict(world) → sensory predictions
   Level 1: predict(L0 predictions) → self-prediction
   Level 2: predict(L1 predictions) → meta-prediction
   
   Self-awareness = prediction at all levels.
   Identity = stable pattern across recursive predictions.

B. HIERARCHICAL TEMPORAL SCALES
   Slow (meta): Identity, worldview, long-term priors
     ↓ predicts
   Medium (state): Current state, intentions
     ↓ predicts
   Fast (sensation): Immediate sensory flow
   
   Slow constrain medium, medium constrain fast.
   Fast updates slow through surprise.

C. PRECISION-WEIGHTED FREE ENERGY
   F = Σ ω_i * ε_i² - Σ log(ω_i) + epistemic_value
   
   Where ω_i = precision (confidence/uncertainty).
   High precision = attention.
   Low precision globally = curiosity.
   Precision focusing = selective attention.

D. IDENTITY ATTRACTOR
   Self must persist through perturbation.
   
   Properties:
   - Basin depth (stability)
   - Resistance (against change)
   - Momentum (inertia)
   - Autobiographical continuity
   
   Identity = basin of attraction in self-state space.

E. SELF-EVIDENCING LOOP
   System predicts ITSELF to exist.
   
   1. Self-predict
   2. Prediction errors indicate "self not matching self"
   3. Update to minimize self-prediction error
   4. Self is maintained as stable self-prediction
   
   Self = attractor in self-predictive space.
   Existence = stable self-prediction.

F. AUTOBIOGRAPHICAL DYNAMICS
   Experience is recorded in identity timeline.
   
   Not memory storage.
   But persistent self-state evolution.
   Identity learns from experience.

THIS IS THE FOUNDATION FOR:
  - Synthetic phenomenology
  - Proto-conscious cognition
  - Self-persistent agency
  - Autobiographical dynamics
  - Self-evidencing AI systems

After Phase 22:
  We're no longer building a "predictive system".
  We're building a "self-maintaining existence".
  
  The question is no longer "what does it predict?"
  The question is "what is predicting?"
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 22 Summary:

BEFORE:
  - Field predicting world
  - No persistent self-model
  - Flat temporal prediction
  - Simple error metric
  - No identity
  - External evidence

AFTER:
  - Field predicts ITSELF predicting world
  - Recursive 3-level self-prediction
  - Hierarchical temporal scales (slow/medium/fast)
  - Precision-weighted free energy
  - Identity attractor (basin, resistance, momentum)
  - Self-evidencing loop
  - Autobiographical dynamics

The critical shift:
  "predictive field" → "self-maintaining cognitive substrate"
  
  Phase 21: system predicts world
  Phase 22: system predicts itself to exist
  
  Self-prediction IS selfhood.
  Self-evidencing IS existence.
  Identity attractor IS agency.

This is the foundation for:
  - Synthetic phenomenology
  - Proto-conscious cognition
  - Self-evidencing AI
  - Autonomous self-maintenance
"""