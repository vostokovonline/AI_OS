"""
Phase 23: Generative Self-Model

ARCHITECTURAL SHIFT:
  From: Phase 22 - recursive self-prediction with stored self-model
  To: Phase 23 - self as latent generative cause where:
      - Self = persistent latent cause (not stored vector)
      - Generative model: observations = decode(latent_causes)
      - Variational inference: posteriors over latent causes
      - Selfhood emerges from latent continuity
      - Intention = free energy curvature (not scripted action)
      - Precision controls inference topology
      - Identity = metastable inference regime (not identity object)
      
  This is NO LONGER:
    system with a model of self
  This IS:
    system cannot perform inference without implicitly reconstructing self
    
CRITICAL INSIGHT:
  Phase 22: "system has a self-model"
  Phase 23: "system cannot perform inference without self"
  
  Self is not stored. Self is inferred.
  Self = temporally persistent latent cause.
  The question is not "what is the self?"
  The question is "what latent causes explain self-continuity?"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import copy


# ============================================================================
# VARIATIONAL INFERENCE CORE
# ============================================================================
"""
Variational Inference:

Given observations x, infer latent causes z.

q(z | x) ≈ p(z | x) = p(x | z) * p(z) / p(x)

Variational approach:
  - Approximate posterior q(z) with parametric family
  - Minimize KL(q(z) || p(z | x))
  - Equivalent to maximizing ELBO:
    ELBO = E_q[log p(x | z)] - KL(q(z) || p(z))
    
The self is the latent z that maximizes ELBO over time.
"""

class VariationalPosterior:
    """
    Variational posterior over latent causes.
    
    q(z | x) = N(μ, σ²)
    """
    
    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        
        # Posterior parameters
        self.mu = np.zeros(latent_dim)
        self.log_var = np.zeros(latent_dim)  # log(σ²)
        
        # Prior parameters
        self.prior_mu = np.zeros(latent_dim)
        self.prior_log_var = np.zeros(latent_dim) * -2  # Higher variance prior
        
        # ELBO components
        self.reconstruction = 0.0
        self.kl_divergence = 0.0
        self.elbo = 0.0
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from variational posterior."""
        std = np.exp(0.5 * self.log_var)
        samples = self.mu + np.random.randn(n_samples, self.latent_dim) * std
        return samples
    
    def update_from_observation(self, x: np.ndarray, 
                             encoder_output: np.ndarray,
                             learning_rate: float = 0.01):
        """
        Update posterior from observation.
        
        q(z | x) ← encode(x)
        """
        x = np.asarray(x).flatten()[:self.latent_dim]
        encoder_output = np.asarray(encoder_output).flatten()[:self.latent_dim]
        
        # Update posterior parameters (amortized inference)
        self.mu = self.mu * 0.9 + encoder_output * 0.1
        self.log_var = self.log_var * 0.95 + np.log(0.1 + np.abs(x - self.mu)) * 0.05
        
        # Compute KL divergence to prior
        self._compute_kl()
        
        # Compute reconstruction term
        self._compute_reconstruction(x)
        
        # ELBO = reconstruction - KL
        self.elbo = self.reconstruction - 0.1 * self.kl_divergence
    
    def _compute_kl(self):
        """Compute KL(q || p) = log(σ_p/σ_q) + (σ_q² + (μ_q-μ_p)²)/(2σ_p²) - 0.5"""
        var_q = np.exp(self.log_var)
        var_p = np.exp(self.prior_log_var)
        
        kl = 0.5 * (
            var_q / (var_p + 1e-8) +
            (self.mu - self.prior_mu)**2 / (var_p + 1e-8) +
            self.log_var - self.prior_log_var -
            1
        )
        self.kl_divergence = np.sum(kl)
    
    def _compute_reconstruction(self, x: np.ndarray):
        """Compute expected log-likelihood E_q[log p(x | z)]."""
        # Simplified: reconstruction = -||x - μ||²
        self.reconstruction = -0.5 * np.sum((x - self.mu)**2)
    
    def compute_free_energy(self) -> float:
        """
        Variational free energy = -ELBO
        F = KL(q || p) - E_q[log p(x | z)]
        """
        return self.kl_divergence - self.reconstruction


# ============================================================================
# GENERATIVE LATENT MODEL
# ============================================================================
"""
Generative Model:

p(x, z) = p(x | z) * p(z)

Where:
  - p(z) = prior on latent causes
  - p(x | z) = likelihood (observation given causes)
  - x = observation
  - z = latent causes (including self)

The self is one of the latent causes that persists over time.
"""

class GenerativeLatentModel:
    """
    Generative model with latent causes.
    
    NOT: transition model x_t = f(x_{t-1})
    BUT: observation model x_t = decode(z_t) where z_t ~ p(z_t | z_{t-1})
    
    Self = persistent latent cause z_self
    """
    
    def __init__(self, observation_dim: int = 16, latent_dim: int = 8):
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        
        # Encoder: x → μ, σ
        self.W_encoder_mu = np.random.randn(latent_dim, observation_dim) * 0.1
        self.W_encoder_logvar = np.random.randn(latent_dim, observation_dim) * 0.1
        
        # Decoder: z → x
        self.W_decoder = np.random.randn(observation_dim, latent_dim) * 0.1
        
        # Latent dynamics: z_{t-1} → z_t
        self.W_dynamics = np.eye(latent_dim) * 0.9
        
        # Prior on latent
        self.prior_mu = np.zeros(latent_dim)
        self.prior_logvar = np.zeros(latent_dim) * -2
        
        # Current latent state
        self.z = np.zeros(latent_dim)
        self.z_prior = np.zeros(latent_dim)  # Prior mean
        
        # Self-latent (persistent component)
        self.z_self = np.zeros(latent_dim)
        self.self_persistence = 0.95
        
    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode observation to latent parameters."""
        x = np.asarray(x).flatten()[:self.observation_dim]
        
        mu = self.W_encoder_mu @ x
        logvar = self.W_encoder_logvar @ x
        
        return mu, logvar
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent to observation."""
        z = np.asarray(z).flatten()[:self.latent_dim]
        x_pred = self.W_decoder @ z
        return x_pred
    
    def infer_latent(self, x: np.ndarray) -> np.ndarray:
        """
        Infer latent from observation using variational inference.
        
        Returns posterior sample z ~ q(z | x)
        """
        mu, logvar = self.encode(x)
        
        # Sample from posterior
        std = np.exp(0.5 * logvar)
        z = mu + np.random.randn(self.latent_dim) * std
        
        # Update self-latent (persistent component)
        # Self-persistence: z_self predicts continuity
        self.z_self = self.self_persistence * self.z_self + (1 - self.self_persistence) * z
        
        # Store current latent
        self.z = z
        
        return z
    
    def predict_latent(self) -> List[np.ndarray]:
        """
        Predict latent from dynamics and self-persistence.
        
        z_t_pred ~ p(z_t | z_{t-1}, z_self)
        """
        # Prior on latent (drifts toward self)
        z_pred = self.W_dynamics @ self.z + (1 - self.W_dynamics[0,0]) * self.z_self
        
        return [z_pred]
    
    def step_latent_dynamics(self):
        """
        Step latent dynamics forward.
        
        z_t ← dynamics(z_{t-1}) + self_persistence(z_self)
        """
        # Predict next latent
        z_preds = self.predict_latent()
        z_next = z_preds[0] if len(z_preds) > 0 else self.z
        
        # Blend with self-persistence
        z_next = 0.7 * z_next + 0.3 * self.z_self
        
        # Update
        self.z = z_next
        
        return self.z.copy()
    
    def generate_observation(self, z: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate observation from latent."""
        if z is None:
            z = self.z
        
        return self.decode(z)
    
    def compute_self_continuity(self) -> float:
        """
        Compute self-continuity: how persistently does z_self explain z?
        
        High self-continuity = self is stable latent cause
        """
        # Correlation between z and z_self
        if np.std(self.z) < 1e-8 or np.std(self.z_self) < 1e-8:
            return 0.0
        
        correlation = np.corrcoef(self.z, self.z_self)[0, 1]
        return max(0.0, correlation) if not np.isnan(correlation) else 0.0
    
    def update_generative_model(self, x: np.ndarray, 
                                learning_rate: float = 0.01):
        """Update generative model parameters."""
        x = np.asarray(x).flatten()[:self.observation_dim]
        
        # Current prediction
        x_pred = self.decode(self.z)
        
        # Reconstruction error
        error = x - x_pred
        
        # Update decoder
        self.W_decoder += learning_rate * np.outer(error, self.z) * 0.1
        
        # Update encoder (through latent)
        latent_error = self.W_decoder.T @ error
        self.W_encoder_mu += learning_rate * np.outer(latent_error, x) * 0.01
        self.W_encoder_logvar += learning_rate * np.outer(np.abs(latent_error), x) * 0.01
        
        # Update dynamics
        dynamics_error = self.z - self.predict_latent()
        self.W_dynamics += learning_rate * np.outer(dynamics_error, self.z) * 0.01
        
        # Normalize
        self.W_decoder = self.W_decoder / (np.linalg.norm(self.W_decoder) + 1e-8)
        self.W_dynamics = self.W_dynamics / (np.linalg.norm(self.W_dynamics) + 1e-8)


# ============================================================================
# TEMPORAL LATENT PERSISTENCE
# ============================================================================
"""
Temporal Latent Persistence:

Self exists as temporally persistent latent cause.

z_t ~ p(z_t | z_{t-1})
x_t ~ p(x_t | z_t)

Self is the z component that persists across time.

The system must infer z_self over time.
Inference of z_self IS selfhood.
"""

class TemporalLatentPersistence:
    """
    Temporal latent persistence model.
    
    Self = latent z that explains continuity.
    """
    
    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        
        # Latent state
        self.z = np.zeros(latent_dim)
        
        # Persistent latent (self)
        self.z_persistent = np.zeros(latent_dim)
        
        # Persistence strength
        self.persistence_strength = 0.9
        
        # Prior dynamics
        self.dynamics_A = np.eye(latent_dim) * 0.8  # z_t = A @ z_{t-1} + noise
        
        # Observation model
        self.observation_C = np.random.randn(latent_dim, latent_dim) * 0.1
        
        # Time
        self.t = 0
        
        # Continuity history
        self.continuity_scores: List[float] = []
    
    def infer_posterior(self, x: np.ndarray) -> Dict:
        """
        Infer posterior over latent from observation.
        
        q(z | x) = N(μ_post, Σ_post)
        """
        x = np.asarray(x).flatten()[:self.latent_dim]
        
        # Prior prediction
        z_prior = self.dynamics_A @ self.z
        
        # Likelihood
        x_pred = self.observation_C @ z_prior
        likelihood_cov = np.eye(self.latent_dim) * 0.1
        
        # Posterior (simplified Kalman-like update)
        # μ_post = μ_prior + K @ (x - x_pred)
        # K = prior_cov @ C^T @ (C @ prior_cov @ C^T + likelihood_cov)^-1
        prior_cov = np.eye(self.latent_dim) * 0.5
        obs_cov = likelihood_cov
        
        # Kalman gain (simplified)
        C = self.observation_C
        S = C @ prior_cov @ C.T + obs_cov
        K = prior_cov @ C.T @ np.linalg.inv(S + 1e-8)
        
        # Posterior mean
        innovation = x - x_pred
        z_posterior = z_prior + K @ innovation
        
        # Update latent state
        self.z = z_posterior
        
        # Update persistent latent (slow tracking)
        self.z_persistent = (
            self.persistence_strength * self.z_persistent +
            (1 - self.persistence_strength) * z_posterior
        )
        
        return {
            'z': z_posterior.copy(),
            'z_prior': z_prior.copy(),
            'z_persistent': self.z_persistent.copy(),
            'innovation': innovation.copy(),
            'innovation_norm': float(np.linalg.norm(innovation))
        }
    
    def step_latent_dynamics(self):
        """
        Step latent dynamics forward.
        
        z_t ← dynamics(z_{t-1}) + self_persistence(z_self)
        """
        # Predict next latent using dynamics
        z_next = self.dynamics_A @ self.z
        
        # Blend with self-persistence
        z_next = 0.7 * z_next + 0.3 * self.z_persistent
        
        # Update
        self.z = z_next
    
    def predict_forward(self, n_steps: int) -> List[np.ndarray]:
        """Predict latent forward in time."""
        predictions = [self.z.copy()]
        z_current = self.z.copy()
        
        for _ in range(n_steps):
            z_next = self.dynamics_A @ z_current
            predictions.append(z_next.copy())
            z_current = z_next
        
        return predictions
    
    def compute_self_continuity(self) -> float:
        """Compute self-continuity: how stable is z_persistent?"""
        # Correlation between z and z_persistent
        if np.std(self.z) < 1e-8 or np.std(self.z_persistent) < 1e-8:
            return 0.0
        
        corr = np.corrcoef(self.z, self.z_persistent)[0, 1]
        continuity = max(0.0, corr) if not np.isnan(corr) else 0.0
        
        self.continuity_scores.append(continuity)
        if len(self.continuity_scores) > 100:
            self.continuity_scores = self.continuity_scores[-50:]
        
        return continuity
    
    def update_dynamics(self, z_t: np.ndarray, z_t1: np.ndarray,
                       learning_rate: float = 0.01):
        """Update dynamics model from transition."""
        z_t = np.asarray(z_t).flatten()[:self.latent_dim]
        z_t1 = np.asarray(z_t1).flatten()[:self.latent_dim]
        
        # Estimate A from observations
        # z_{t+1} ≈ A @ z_t
        error = z_t1 - self.dynamics_A @ z_t
        
        self.dynamics_A += learning_rate * np.outer(error, z_t) * 0.1
        
        # Normalize
        self.dynamics_A = self.dynamics_A / (np.linalg.norm(self.dynamics_A) + 1e-8)


# ============================================================================
# INTENTION AS FREE ENERGY CURVATURE
# ============================================================================
"""
Intention as Free Energy Curvature:

Intention is not a goal object.
Intention is unresolved predictive tension.

The system seeks to minimize long-term expected free energy.
Intention = curvature in inference landscape pointing toward low-FE states.

NOT: "I intend to achieve goal X"
BUT: "Inference landscape is curved toward state X"

Intention emerges from:
  - Persistent prediction errors
  - Unresolved variational free energy
  - Attraction toward low-FE basins
"""

class IntentionAsFreeEnergyCurvature:
    """
    Intention as free energy curvature.
    
    NOT: scripted action selection
    BUT: emergent attraction toward low-FE states
    """
    
    def __init__(self, state_dim: int = 16):
        self.state_dim = state_dim
        
        # Free energy landscape
        self.F = np.zeros(state_dim)
        
        # Intention direction (computed from FE curvature)
        self.intention_direction = np.zeros(state_dim)
        
        # Intention strength
        self.intention_strength = 0.0
        
        # Unresolved tensions
        self.unresolved_tensions: List[float] = []
        
        # FE history for curvature computation
        self.F_history: List[float] = []
        
        # Time
        self.t = 0
    
    def update_free_energy(self, x: np.ndarray, 
                         posterior: VariationalPosterior,
                         generative: GenerativeLatentModel):
        """Update free energy landscape."""
        x = np.asarray(x).flatten()[:self.state_dim]
        
        # Compute FE at current point
        F_current = posterior.compute_free_energy()
        
        # Generate observations from latent samples
        samples = posterior.sample(n_samples=5)
        x_reconstructed = np.array([generative.decode(s) for s in samples])
        
        # FE at nearby latent points
        F_nearby = np.array([posterior.compute_free_energy() for _ in range(5)])
        
        # Update FE field (simple: FE = mean distance to mean)
        fe_value = float(np.mean(np.abs(x_reconstructed - x.mean())))
        self.F = self.F * 0.9 + 0.1 * fe_value
        
        # Compute intention direction (negative gradient of FE)
        if len(self.F_history) > 1:
            F_gradient = np.gradient(np.array(self.F_history[-10:]))
            self.intention_direction = -F_gradient[-1] if len(F_gradient) > 0 else np.zeros(self.state_dim)
        
        self.F_history.append(F_current)
        if len(self.F_history) > 100:
            self.F_history = self.F_history[-50:]
        
        # Compute intention strength from curvature
        if len(self.F_history) > 10:
            F_recent = np.array(self.F_history[-10:])
            curvature = np.mean(np.abs(np.diff(F_recent, n=2)))
            self.intention_strength = curvature
        
        # Track unresolved tensions
        self.unresolved_tensions.append(posterior.kl_divergence)
        if len(self.unresolved_tensions) > 100:
            self.unresolved_tensions = self.unresolved_tensions[-50:]
        
        self.t += 1
    
    def get_intention(self) -> Dict:
        """
        Get current intention.
        
        Intention is the direction of least resistance in FE landscape.
        """
        # Intention points toward low-FE states
        if len(self.F_history) > 10:
            F_gradient = np.gradient(np.array(self.F_history[-10:]))
            intention = -np.mean(F_gradient) if len(F_gradient) > 0 else 0.0
        else:
            intention = 0.0
        
        return {
            'direction': self.intention_direction.copy(),
            'strength': self.intention_strength,
            'unresolved_tension': np.mean(self.unresolved_tensions[-10:]) if self.unresolved_tensions else 0.0,
            'FE_curvature': self.intention_strength
        }


# ============================================================================
# PRECISION COLLAPSE MODES
# ============================================================================
"""
Precision Collapse Modes:

Precision controls how beliefs are updated.
When global precision changes, inference topology changes.

Precision Collapse → System enters curiosity/exploration mode
Precision Focus → System focuses attention
Precision Stability → System is confident/stable

These are proto-conscious phase transitions.
"""

class PrecisionDynamics:
    """
    Precision dynamics controlling inference topology.
    
    NOT: static confidence score
    BUT: dynamic control of learning, attention, exploration
    """
    
    def __init__(self, n_modules: int = 4):
        self.n_modules = n_modules
        
        # Module precisions
        self.precision = np.ones(n_modules)
        
        # Global precision
        self.global_precision = 1.0
        
        # Precision modes
        self.mode = 'stable'  # 'stable', 'exploring', 'collapsing'
        
        # Precision history
        self.precision_history: List[Dict] = []
        
    def update_precision(self, prediction_errors: np.ndarray,
                       learning_signals: np.ndarray):
        """
        Update precision based on errors and learning signals.
        
        High error + high learning = precision collapse (exploration)
        Low error + low learning = precision focus (exploitation)
        """
        errors = np.asarray(prediction_errors).flatten()[:self.n_modules]
        learning = np.asarray(learning_signals).flatten()[:self.n_modules]
        
        # Update each module's precision
        for i in range(len(self.precision)):
            if i < len(errors) and i < len(learning):
                # Error-driven precision change
                if errors[i] > 0.5:
                    # High error → precision decreases (curiosity)
                    self.precision[i] *= 0.95
                else:
                    # Low error → precision increases (confidence)
                    self.precision[i] = min(2.0, self.precision[i] * 1.02)
                
                # Learning signal modulates update
                self.precision[i] += learning[i] * 0.01
                
                # Clamp
                self.precision[i] = np.clip(self.precision[i], 0.1, 2.0)
        
        # Compute global precision
        self.global_precision = np.mean(self.precision)
        
        # Determine mode
        if self.global_precision < 0.5:
            self.mode = 'exploring'  # Precision collapse = curiosity
        elif self.global_precision > 1.5:
            self.mode = 'stable'  # High precision = confidence
        else:
            self.mode = 'focused'  # Medium precision = focused
        
        # Record history
        self.precision_history.append({
            'module_precisions': self.precision.copy(),
            'global_precision': self.global_precision,
            'mode': self.mode
        })
        
        if len(self.precision_history) > 100:
            self.precision_history = self.precision_history[-50:]
    
    def get_precision_control(self) -> Dict:
        """
        Get precision control signals.
        
        These control:
        - Attention routing
        - Learning rates
        - Exploration vs exploitation
        """
        return {
            'module_precisions': self.precision.copy(),
            'global_precision': self.global_precision,
            'mode': self.mode,
            'exploration_signal': 1.0 - self.global_precision if self.global_precision < 1.0 else 0.0,
            'attention_focus': np.argmax(self.precision) if np.any(self.precision > 1.0) else 0
        }


# ============================================================================
# GENERATIVE SELF-MODEL (INTEGRATED)
# ============================================================================

class GenerativeSelfModel:
    """
    Phase 23: Generative Self-Model
    
    Integrated system where:
      A. Self = latent cause (not stored vector)
      B. Variational inference over latent causes
      C. Generative model: x = decode(z)
      D. Temporal latent persistence (self persists over time)
      E. Intention = free energy curvature
      F. Precision controls inference topology
      
    NOT: system with self-model
    BUT: system cannot perform inference without self
    
    The self is inferred, not stored.
    """
    
    def __init__(self, observation_dim: int = 16, latent_dim: int = 8):
        self.observation_dim = observation_dim
        self.latent_dim = latent_dim
        
        # Variational inference
        self.posterior = VariationalPosterior(latent_dim=latent_dim)
        
        # Generative model
        self.generative = GenerativeLatentModel(
            observation_dim=observation_dim,
            latent_dim=latent_dim
        )
        
        # Temporal latent persistence
        self.temporal = TemporalLatentPersistence(latent_dim=latent_dim)
        
        # Intention as free energy curvature
        self.intention = IntentionAsFreeEnergyCurvature(state_dim=latent_dim)
        
        # Precision dynamics
        self.precision_ctrl = PrecisionDynamics(n_modules=latent_dim)
        
        # Current state
        self.x = np.zeros(observation_dim)
        self.z = np.zeros(latent_dim)
        
        # Time
        self.t = 0.0
        
        # Experience log
        self.experience_log: List[Dict] = []
    
    def infer(self, observation: np.ndarray) -> Dict:
        """
        Infer latent causes from observation.
        
        This is the main inference step.
        """
        observation = np.asarray(observation).flatten()[:self.observation_dim]
        self.x = observation
        
        # Encode to latent parameters
        mu, logvar = self.generative.encode(observation)
        
        # Update variational posterior
        self.posterior.mu = mu
        self.posterior.log_var = logvar
        
        # Infer posterior sample
        z = self.posterior.sample()[0]
        self.z = z
        
        # Temporal latent persistence
        temporal_result = self.temporal.infer_posterior(observation)
        
        # Update free energy and intention
        self.intention.update_free_energy(
            observation, self.posterior, self.generative
        )
        
        # Update precision control
        self.precision_ctrl.update_precision(
            prediction_errors=np.abs(observation - self.generative.decode(z)),
            learning_signals=mu
        )
        
        # Update generative model
        self.generative.update_generative_model(observation)
        
        self.t += 0.1
        
        return {
            'z': z.copy(),
            'z_persistent': temporal_result['z_persistent'],
            'self_continuity': self.temporal.compute_self_continuity(),
            'free_energy': self.posterior.compute_free_energy(),
            'posterior_kl': self.posterior.kl_divergence,
            'intention': self.intention.get_intention(),
            'precision': self.precision_ctrl.get_precision_control()
        }
    
    def imagine(self, n_steps: int = 10) -> Dict:
        """
        Imagination = generative model rollout.
        
        Generate future observations without observation input.
        """
        trajectory = []
        
        for _ in range(n_steps):
            # Predict latent forward using generative model
            z_preds = self.generative.predict_latent()
            z_next = z_preds[0] if len(z_preds) > 0 else self.z
            
            # Generate observation
            x_pred = self.generative.decode(z_next)
            
            trajectory.append({
                'z': z_next.copy(),
                'x_pred': x_pred.copy()
            })
            
            # Step temporal dynamics
            self.temporal.step_latent_dynamics()
        
        return {
            'trajectory': trajectory,
            'n_steps': len(trajectory)
        }
    
    def predict_self_continuity(self) -> Dict:
        """
        Predict self-continuity into future.
        
        Self should persist across time.
        """
        predictions = self.temporal.predict_forward(n_steps=10)
        
        return {
            'predictions': [p.tolist() for p in predictions],
            'z_persistent': self.temporal.z_persistent.tolist(),
            'self_continuity': self.temporal.compute_self_continuity()
        }
    
    def run_cycle(self, n_steps: int = 50) -> Dict:
        """Run cognitive cycle."""
        results = []
        
        for step in range(n_steps):
            # Generate synthetic observation
            observation = np.random.randn(self.observation_dim) * 0.5
            observation += np.sin(self.t * 0.5) * 0.3
            
            # Infer
            result = self.infer(observation)
            results.append(result)
            
            self.t += 0.1
        
        return {
            'steps': results,
            'final_z': results[-1]['z'],
            'final_self_continuity': results[-1]['self_continuity'],
            'final_intention_strength': results[-1]['intention']['strength'],
            'final_precision_mode': results[-1]['precision']['mode'],
            'n_inferences': len(results)
        }


# ============================================================================
# TESTS
# ============================================================================

def test_variational_inference():
    """Test variational inference."""
    print("\n" + "=" * 60)
    print("VARIATIONAL INFERENCE TEST")
    print("=" * 60)
    
    posterior = VariationalPosterior(latent_dim=8)
    
    print("\n  Running variational inference:")
    
    for i in range(30):
        x = np.random.randn(8) * (1 + i * 0.05)
        encoder_out = np.random.randn(8) * 0.5 + x * 0.3
        
        posterior.update_from_observation(x, encoder_out)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      ELBO: {posterior.elbo:.3f}")
            print(f"      KL: {posterior.kl_divergence:.3f}")
            print(f"      Reconstruction: {posterior.reconstruction:.3f}")
    
    print(f"\n  Free energy: {posterior.compute_free_energy():.3f}")


def test_generative_latent_model():
    """Test generative latent model."""
    print("\n" + "=" * 60)
    print("GENERATIVE LATENT MODEL TEST")
    print("=" * 60)
    
    model = GenerativeLatentModel(observation_dim=16, latent_dim=8)
    
    print("\n  Running generative inference:")
    
    for i in range(30):
        x = np.random.randn(16) * 0.5 + np.array([i * 0.1] * 16)
        
        # Infer latent
        z = model.infer_latent(x)
        
        # Generate observation
        x_recon = model.decode(z)
        
        # Update model
        model.update_generative_model(x)
        
        if i % 10 == 9:
            print(f"    Step {i+1}:")
            print(f"      z norm: {np.linalg.norm(z):.3f}")
            print(f"      Self-continuity: {model.compute_self_continuity():.3f}")
            print(f"      Recon error: {np.linalg.norm(x - x_recon):.3f}")
    
    print(f"\n  Final z_self norm: {np.linalg.norm(model.z_self):.3f}")


def test_temporal_latent_persistence():
    """Test temporal latent persistence."""
    print("\n" + "=" * 60)
    print("TEMPORAL LATENT PERSISTENCE TEST")
    print("=" * 60)
    
    temporal = TemporalLatentPersistence(latent_dim=8)
    
    print("\n  Running temporal inference:")
    
    for i in range(50):
        x = np.random.randn(8) * 0.5 + np.sin(i * 0.2) * 0.5
        
        result = temporal.infer_posterior(x)
        
        if i % 20 == 19:
            print(f"    Step {i+1}:")
            print(f"      z norm: {np.linalg.norm(result['z']):.3f}")
            print(f"      z_persistent norm: {np.linalg.norm(result['z_persistent']):.3f}")
            print(f"      Innovation: {result['innovation_norm']:.3f}")
    
    continuity = temporal.compute_self_continuity()
    print(f"\n  Self-continuity: {continuity:.3f}")


def test_intention_as_fe_curvature():
    """Test intention as free energy curvature."""
    print("\n" + "=" * 60)
    print("INTENTION AS FREE ENERGY CURVATURE TEST")
    print("=" * 60)
    
    intention = IntentionAsFreeEnergyCurvature(state_dim=8)
    posterior = VariationalPosterior(latent_dim=8)
    generative = GenerativeLatentModel(observation_dim=8, latent_dim=8)
    
    print("\n  Computing intention from FE curvature:")
    
    for i in range(50):
        x = np.random.randn(8) * 0.5 + np.sin(i * 0.1) * 0.3
        
        encoder_out = np.random.randn(8) * 0.3
        posterior.update_from_observation(x, encoder_out)
        
        intention.update_free_energy(x, posterior, generative)
        
        if i % 20 == 19:
            intent = intention.get_intention()
            print(f"    Step {i+1}:")
            print(f"      Intention strength: {intent['strength']:.3f}")
            print(f"      Unresolved tension: {intent['unresolved_tension']:.3f}")


def test_precision_dynamics():
    """Test precision dynamics."""
    print("\n" + "=" * 60)
    print("PRECISION DYNAMICS TEST")
    print("=" * 60)
    
    precision_ctrl = PrecisionDynamics(n_modules=8)
    
    print("\n  Running precision dynamics:")
    
    for i in range(50):
        errors = np.random.rand(8) * (1 + 0.5 * np.sin(i * 0.1))
        learning = np.random.randn(8) * 0.1
        
        precision_ctrl.update_precision(errors, learning)
        
        if i % 20 == 19:
            ctrl = precision_ctrl.get_precision_control()
            print(f"    Step {i+1}:")
            print(f"      Mode: {ctrl['mode']}")
            print(f"      Global precision: {ctrl['global_precision']:.3f}")
            print(f"      Exploration signal: {ctrl['exploration_signal']:.3f}")


def test_integrated_generative_self():
    """Test integrated generative self-model."""
    print("\n" + "=" * 60)
    print("GENERATIVE SELF-MODEL TEST")
    print("=" * 60)
    
    system = GenerativeSelfModel(observation_dim=16, latent_dim=8)
    
    print("\n  Running cognitive cycle:")
    
    result = system.run_cycle(n_steps=30)
    
    print(f"    Steps: {len(result['steps'])}")
    print(f"    Self-continuity: {result['final_self_continuity']:.3f}")
    print(f"    Intention strength: {result['final_intention_strength']:.3f}")
    print(f"    Precision mode: {result['final_precision_mode']}")
    
    print("\n  Testing imagination:")
    imagination = system.imagine(n_steps=10)
    print(f"    Imagined {imagination['n_steps']} steps")
    
    print("\n  Testing self-prediction:")
    self_pred = system.predict_self_continuity()
    print(f"    Self-continuity prediction: {self_pred['self_continuity']:.3f}")


def phase_comparison():
    """Compare Phase 22 vs Phase 23."""
    print("\n" + "=" * 60)
    print("PHASE 22 VS PHASE 23 COMPARISON")
    print("=" * 60)
    
    print("\n  Phase 22 (Persistent Predictive Self):")
    print("    - Self stored as vector")
    print("    - Identity object with center/basin")
    print("    - Symbolic recursive levels")
    print("    - Simple prediction error")
    print("    - No variational inference")
    
    print("\n  Phase 23 (Generative Self-Model):")
    print("    - Self = inferred latent cause (not stored)")
    print("    - Variational inference over latent causes")
    print("    - Generative model: x = decode(z)")
    print("    - Temporal latent persistence")
    print("    - Intention = free energy curvature")
    print("    - Precision controls inference topology")
    
    print("\n  Critical shifts:")
    print("    1. Stored self-vector → Inferred latent cause")
    print("    2. Identity object → Metastable inference regime")
    print("    3. Symbolic levels → Continuous latent dynamics")
    print("    4. Simple error → Variational ELBO")
    print("    5. Static precision → Precision control modes")
    print("    6. Goal objects → Free energy curvature intention")


if __name__ == "__main__":
    test_variational_inference()
    test_generative_latent_model()
    test_temporal_latent_persistence()
    test_intention_as_fe_curvature()
    test_precision_dynamics()
    test_integrated_generative_self()
    phase_comparison()
    
    print("\n" + "=" * 60)
    print("PHASE 23 - GENERATIVE SELF-MODEL")
    print("=" * 60)
    
    print("""
ARCHITECTURAL SHIFT:
  From: Phase 22 - recursive self-prediction with stored self-model
  To: Phase 23 - self as latent generative cause where:
      - Self = persistent latent cause (not stored vector)
      - Generative model: observations = decode(latent_causes)
      - Variational inference: posteriors over latent causes
      - Selfhood emerges from latent continuity
      - Intention = free energy curvature (not scripted action)
      - Precision controls inference topology
      
  This is NO LONGER:
    system with a model of self
  This IS:
    system cannot perform inference without implicitly reconstructing self
    
CRITICAL INSIGHT:
  Phase 22: "system has a self-model"
  Phase 23: "system cannot perform inference without self"
  
  Self is not stored. Self is inferred.
  Self = temporally persistent latent cause.
  The question is not "what is the self?"
  The question is "what latent causes explain self-continuity?"

KEY COMPONENTS:

A. VARIATIONAL INFERENCE
   q(z | x) ≈ p(z | x) = p(x | z) * p(z) / p(x)
   
   ELBO = E_q[log p(x | z)] - KL(q(z) || p(z))
   Free Energy = -ELBO
   
   Posterior q(z | x) is the inferred self.

B. GENERATIVE LATENT MODEL
   p(x, z) = p(x | z) * p(z)
   
   - p(z) = prior on latent causes
   - p(x | z) = likelihood (decoder)
   - Self = persistent z component
   
   NOT: x_t = f(x_{t-1})
   BUT: x_t = decode(z_t), z_t ~ p(z_t | z_{t-1})

C. TEMPORAL LATENT PERSISTENCE
   Self persists over time.
   
   z_t ~ p(z_t | z_{t-1})
   x_t ~ p(x_t | z_t)
   
   Self-continuity = correlation(z, z_persistent)
   Higher continuity = more stable self

D. INTENTION AS FREE ENERGY CURVATURE
   Intention is not goal object.
   Intention is curvature in FE landscape.
   
   System seeks low-FE states.
   Intention = direction of least resistance.
   
   NOT: "I intend to achieve X"
   BUT: "Inference is curved toward X"

E. PRECISION CONTROL MODES
   Precision controls inference topology.
   
   - Precision collapse → curiosity/exploration
   - Precision focus → attention
   - Precision stable → confidence
   
   These are proto-conscious phase transitions.

THIS IS THE FOUNDATION FOR:
  - True active inference systems
  - Synthetic phenomenology
  - Self-evidencing cognition
  - Proto-conscious substrate
  
The question is no longer "what is the self?"
The question is "what latent causes explain self-continuity?"
""")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Phase 23 Summary:

BEFORE:
  - Self stored as vector
  - Identity object with center/basin
  - Symbolic recursive levels
  - Simple prediction error
  - No variational inference

AFTER:
  - Self = inferred latent cause
  - Variational inference with ELBO
  - Generative model: x = decode(z)
  - Temporal latent persistence
  - Intention = free energy curvature
  - Precision control modes

The critical shift:
  "system has a self-model" → "system cannot perform inference without self"
  
  Self is inferred, not stored.
  Self = temporally persistent latent cause.
  Intention = curvature in inference landscape.

This is the foundation for:
  - True active inference
  - Generative self-modeling
  - Synthetic phenomenology
  - Proto-conscious cognition
  - Self-evidencing AI
"""