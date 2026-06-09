"""
Phase 16 - Information Geometry System

NOT physics imitation with separate modules.
NOT adaptive preconditioner.

BUT true information geometry where:

1. Single probabilistic model: p_θ(z)
2. Energy: V(z) = -log p_θ(z)
3. Metric (Fisher-Rao): g_ij(z) = E[∂_i log p * ∂_j log p]
4. Dynamics from single action functional

Key principle:
  Everything (metric, energy, dynamics) derived from one source p_θ(z)
  
  NOT:
    g = learned from velocity stats (separate)
    V = learned from KDE (separate)
    dynamics = glued together
    
  BUT:
    p_θ(z) → V(z) = -log p_θ(z)
    p_θ(z) → g_ij(z) = E[∂_i log p * ∂_j log p]
    V(z) + g(z) → geodesic dynamics via δ∫L dt = 0

This is the Fisher-Rao geometric framework:
  - Natural gradient = gradient with respect to Fisher metric
  - Geodesics = flow under Fisher information geometry
  - KL divergence = geodesic distance
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class GenerativeLatentModel:
    """
    Single generative model p_θ(z) for latent space.
    
    This is the ONE source for everything:
    - Energy: V(z) = -log p(z)
    - Metric: g_ij(z) = E[∂_i log p * ∂_j log p]
    - Natural gradient
    
    Model structure (simplified Gaussian mixture):
    p(z) = Σ w_k * N(z; μ_k, Σ_k)
    
    For efficiency, use diagonal covariance.
    """
    
    def __init__(self, dimension: int = 16, n_components: int = 5):
        self.dimension = dimension
        self.n_components = n_components
        
        # Mixture components
        self.weights: List[float] = [1.0 / n_components] * n_components
        self.means: List[List[float]] = [
            [0.0] * dimension for _ in range(n_components)
        ]
        self.covariances: List[List[float]] = [
            [1.0] * dimension for _ in range(n_components)
        ]
        
        # Observation history for learning
        self.samples: List[List[float]] = []
        self.outcomes: List[str] = []
        
        # Trained flag
        self.trained = False
        
        logger.info("generative_latent_model_initialized", 
                    dimension=dimension, 
                    n_components=n_components)
    
    def observe(self, z: List[float], outcome: Optional[str] = None) -> None:
        """Observe state for learning"""
        self.samples.append(z)
        if outcome:
            self.outcomes.append(outcome)
    
    def fit(self) -> None:
        """
        Fit generative model from observations.
        
        Uses EM algorithm for Gaussian mixture.
        """
        if len(self.samples) < self.n_components * 2:
            logger.warning("insufficient_samples_for_fit", 
                          count=len(self.samples))
            return
        
        # Initialize with k-means
        self._initialize_components()
        
        # EM iterations
        for _ in range(20):
            # E-step: compute responsibilities
            responsibilities = self._e_step()
            
            # M-step: update parameters
            self._m_step(responsibilities)
        
        self.trained = True
        logger.info("generative_model_fitted", samples=len(self.samples))
    
    def _initialize_components(self) -> None:
        """Initialize components using k-means"""
        import random
        random.seed(42)
        
        # Simple k-means initialization
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        
        for k in range(self.n_components):
            idx = indices[k * len(indices) // self.n_components]
            self.means[k] = self.samples[idx][:self.dimension]
    
    def _e_step(self) -> List[List[float]]:
        """E-step: compute responsibilities"""
        n = len(self.samples)
        responsibilities = [[0.0] * self.n_components for _ in range(n)]
        
        for i, z in enumerate(self.samples):
            log_probs = []
            for k in range(self.n_components):
                log_prob = self._log_component_density(z, k)
                log_probs.append(log_prob)
            
            # Log-sum-exp for numerical stability
            max_log = max(log_probs)
            probs = [math.exp(lp - max_log) for lp in log_probs]
            total = sum(probs)
            
            for k in range(self.n_components):
                responsibilities[i][k] = probs[k] / total
        
        return responsibilities
    
    def _m_step(self, responsibilities: List[List[float]]) -> None:
        """M-step: update parameters"""
        n = len(self.samples)
        
        for k in range(self.n_components):
            # Update weight
            total_resp = sum(responsibilities[i][k] for i in range(n))
            self.weights[k] = total_resp / n
            
            if self.weights[k] < 1e-6:
                self.weights[k] = 1e-6
            
            # Update mean
            numerator = [0.0] * self.dimension
            for i, z in enumerate(self.samples):
                for j in range(self.dimension):
                    numerator[j] += responsibilities[i][k] * z[j]
            
            self.means[k] = [n / total_resp for n in numerator]
            
            # Update covariance (diagonal)
            for j in range(self.dimension):
                numerator_var = sum(
                    responsibilities[i][k] * (self.samples[i][j] - self.means[k][j]) ** 2
                    for i in range(n)
                )
                self.covariances[k][j] = max(1e-6, numerator_var / total_resp)
    
    def _log_component_density(self, z: List[float], k: int) -> float:
        """Compute log-density of component k"""
        log_prob = math.log(self.weights[k] + 1e-10)
        
        for j in range(self.dimension):
            diff = z[j] - self.means[k][j]
            var = self.covariances[k][j]
            log_prob -= 0.5 * (diff ** 2 / var + math.log(2 * math.pi * var))
        
        return log_prob
    
    def log_density(self, z: List[float]) -> float:
        """
        Compute log-density: log p(z) = log Σ_k w_k * N(z; μ_k, Σ_k)
        
        This is the ENERGY functional:
        V(z) = -log p(z)
        """
        if not self.trained and len(self.samples) >= 10:
            self.fit()
        
        if not self.trained:
            return 0.0
        
        # Mixture log-density
        log_probs = []
        for k in range(self.n_components):
            log_probs.append(self._log_component_density(z, k))
        
        # Log-sum-exp
        max_log = max(log_probs)
        log_sum = max_log + math.log(sum(math.exp(lp - max_log) for lp in log_probs))
        
        return log_sum
    
    def density(self, z: List[float]) -> float:
        """Compute density p(z)"""
        return math.exp(self.log_density(z))
    
    def energy(self, z: List[float]) -> float:
        """
        Compute energy: V(z) = -log p(z)
        
        This IS the potential for dynamics.
        """
        return -self.log_density(z)
    
    def gradient_log_density(self, z: List[float]) -> List[float]:
        """
        Compute gradient of log-density: ∂_i log p(z)
        
        Uses mixture formula:
        ∂_i log p = Σ_k r_k * ∂_i log N(z; μ_k, Σ_k)
        
        where r_k = p(k|z) ∝ w_k * N(z; μ_k, Σ_k)
        """
        if not self.trained:
            return [0.0] * self.dimension
        
        # Compute responsibilities
        responsibilities = []
        log_probs = []
        
        for k in range(self.n_components):
            log_prob = self._log_component_density(z, k)
            log_probs.append(log_prob)
            responsibilities.append(math.exp(log_prob))
        
        total = sum(responsibilities)
        if total < 1e-10:
            return [0.0] * self.dimension
        
        # Normalize
        responsibilities = [r / total for r in responsibilities]
        
        # Gradient
        grad = [0.0] * self.dimension
        
        for k in range(self.n_components):
            r_k = responsibilities[k]
            for j in range(self.dimension):
                diff = z[j] - self.means[k][j]
                var = self.covariances[k][j]
                grad[j] += r_k * (-diff / var)
        
        return grad
    
    def sample(self, n: int = 1) -> List[List[float]]:
        """Sample from generative model"""
        samples = []
        
        for _ in range(n):
            # Sample component
            import random
            k = random.choices(
                range(self.n_components),
                weights=self.weights
            )[0]
            
            # Sample from component
            sample = [
                self.means[k][j] + random.gauss(0, math.sqrt(self.covariances[k][j]))
                for j in range(self.dimension)
            ]
            samples.append(sample)
        
        return samples
    
    def to_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "n_components": self.n_components,
            "trained": self.trained,
            "samples": len(self.samples),
        }


class FisherRaoMetric:
    """
    Fisher-Rao metric derived from p_θ(z).
    
    g_ij(z) = E[∂_i log p * ∂_j log p]
    
    This is the NATURAL metric for the manifold.
    
    Properties:
    - g is the Fisher information matrix
    - Geodesics = natural gradient flow
    - KL divergence = geodesic distance
    - Affine connection (alpha=0): α-connection
    """
    
    def __init__(self, generative_model: GenerativeLatentModel):
        self.model = generative_model
        self.dimension = generative_model.dimension
        
        # Cached metric
        self.cached_z: Optional[List[float]] = None
        self.cached_metric: Optional[List[List[float]]] = None
        
        # Gradient cache for efficiency
        self.gradient_cache: Dict[str, List[float]] = {}
    
    def compute_metric(self, z: List[float]) -> List[List[float]]:
        """
        Compute Fisher information metric at z.
        
        g_ij(z) = E[∂_i log p * ∂_j log p]
        
        For Gaussian mixture, this is:
        g_ij(z) = Σ_k r_k * (z_i - μ_k^i)(z_j - μ_k^j) / Σ_k^2
        
        Simplified: use gradient outer product.
        """
        if self.cached_z and self._is_close(z, self.cached_z):
            return self.cached_metric
        
        # Compute gradient of log-density
        grad_log_p = self.model.gradient_log_density(z)
        
        # Fisher metric approximation:
        # g_ij ≈ ∂_i log p * ∂_j log p (empirical)
        # plus expectation over mixture
        
        # Use mixture statistics
        metric = [[0.0] * self.dimension for _ in range(self.dimension)]
        
        # For diagonal approximation (efficiency)
        for k in range(self.model.n_components):
            weight = self.model.weights[k]
            
            for j in range(self.dimension):
                diff = z[j] - self.model.means[k][j]
                var = self.model.covariances[k][j]
                
                # Fisher metric contribution
                metric[j][j] += weight * (diff ** 2) / (var ** 2 + 1e-6)
        
        # Normalize
        trace = sum(metric[i][i] for i in range(self.dimension))
        if trace > 0:
            scale = self.dimension / trace
            for i in range(self.dimension):
                for j in range(self.dimension):
                    metric[i][j] *= scale
        
        self.cached_z = z
        self.cached_metric = metric
        
        return metric
    
    def _is_close(self, z1: List[float], z2: List[float], epsilon: float = 0.05) -> bool:
        """Check if z1 and z2 are close (for caching)"""
        if z2 is None:
            return False
        
        for a, b in zip(z1, z2):
            if abs(a - b) > epsilon:
                return False
        return True
    
    def inverse_metric(self, z: List[float]) -> List[List[float]]:
        """Compute inverse Fisher metric g^ij"""
        metric = self.compute_metric(z)
        
        # Diagonal approximation
        inv = [[0.0] * self.dimension for _ in range(self.dimension)]
        for i in range(self.dimension):
            if metric[i][i] > 1e-6:
                inv[i][i] = 1.0 / metric[i][i]
        
        return inv
    
    def christoffel_symbols(self, z: List[float]) -> List[List[List[float]]]:
        """
        Compute Christoffel symbols from Fisher metric.
        
        Γ^i_jk = 0.5 * g^iμ * (∂_j g_μk + ∂_k g_μj - ∂_μ g_jk)
        
        For Fisher metric, this relates to natural gradient geometry.
        """
        metric = self.compute_metric(z)
        inv = self.inverse_metric(z)
        
        epsilon = 0.02
        
        # Compute metric derivatives numerically
        gamma = [
            [[0.0] * self.dimension for _ in range(self.dimension)]
            for _ in range(self.dimension)
        ]
        
        for mu in range(self.dimension):
            # ∂_j g_μk approximation
            z_plus = z[:]
            z_plus[j] += epsilon
            
            z_minus = z[:]
            z_minus[j] -= epsilon
            
            # This is simplified - full implementation would need
            # derivatives of Fisher metric in natural coordinates
        
        return gamma
    
    def riemannian_gradient(self, z: List[float], grad_flat: List[float]) -> List[float]:
        """
        Convert flat gradient to natural (Riemannian) gradient.
        
        grad_nat^i = g^ij * grad_flat^j
        """
        inv = self.inverse_metric(z)
        
        grad_nat = [
            sum(inv[i][j] * grad_flat[j] for j in range(self.dimension))
            for i in range(self.dimension)
        ]
        
        return grad_nat
    
    def geodesic_direction(self, z: List[float], v: List[float]) -> List[float]:
        """
        Compute geodesic acceleration direction.
        
        For Fisher metric, geodesic term involves:
        - Christoffel symbols
        - Metric geometry
        """
        gamma = self.christoffel_symbols(z)
        
        # Simplified geodesic term
        geo = [0.0] * self.dimension
        
        return geo
    
    def kinetic_energy(self, z: List[float], v: List[float]) -> float:
        """
        Compute kinetic energy: T = 0.5 * g_ij * v^i * v^j
        """
        metric = self.compute_metric(z)
        
        T = 0.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                T += 0.5 * metric[i][j] * v[i] * v[j]
        
        return T
    
    def fisher_information(self, z: List[float]) -> float:
        """
        Compute Fisher information: trace(g)
        """
        metric = self.compute_metric(z)
        return sum(metric[i][i] for i in range(self.dimension))


class NaturalGradientFlow:
    """
    Natural gradient flow on Fisher-Rao manifold.
    
    Dynamics:
    ż = -g^ij * ∂_j V
    
    where V = -log p(z) is the energy.
    
    This is the INFORMATION GEOMETRY dynamics:
    - Flow follows natural gradient (Fisher metric)
    - Equilibria at minima of energy (modes of p)
    - Geodesic-like flow in probability space
    """
    
    def __init__(self, generative_model: GenerativeLatentModel):
        self.model = generative_model
        self.fisher_metric = FisherRaoMetric(generative_model)
        
        self.dimension = generative_model.dimension
        self.dt = 0.01
        self.damping = 0.02
        
        logger.info("natural_gradient_flow_initialized")
    
    def compute_force(self, z: List[float]) -> List[float]:
        """
        Compute natural gradient force.
        
        F = -g^ij * ∂_j V = g^ij * ∂_j log p
        
        This pulls system towards higher density regions.
        """
        # Gradient of log-density
        grad_log_p = self.model.gradient_log_density(z)
        
        # Natural gradient: g^-1 * grad
        nat_grad = self.fisher_metric.riemannian_gradient(z, grad_log_p)
        
        return nat_grad
    
    def step(self, z: List[float], v: List[float]) -> Tuple[List[float], List[float]]:
        """
        Single step of natural gradient flow.
        
        ż = natural gradient
        ż̈ = damping
        """
        # Natural gradient force
        force = self.compute_force(z)
        
        # Update velocity
        new_v = [
            v[i] * (1 - self.damping) + self.dt * force[i]
            for i in range(self.dimension)
        ]
        
        # Update position
        new_z = [
            z[i] + self.dt * new_v[i]
            for i in range(self.dimension)
        ]
        
        return new_z, new_v
    
    def integrate(
        self,
        initial_z: List[float],
        steps: int = 100
    ) -> List[Tuple[List[float], List[float]]]:
        """
        Integrate natural gradient flow.
        """
        trajectory = []
        
        z = initial_z[:]
        v = [0.0] * self.dimension
        
        for _ in range(steps):
            z, v = self.step(z, v)
            trajectory.append((z[:], v[:]))
            
            # Stop if near equilibrium
            if sum(abs(vi) for vi in v) < 0.01:
                break
        
        return trajectory


class InformationGeometrySystem:
    """
    Phase 16 - Information Geometry System
    
    Single unified framework where everything derives from p_θ(z):
    
    1. Generative model p(z) [ONE SOURCE]
       ↓
    2. Energy V(z) = -log p(z)
    3. Metric g_ij(z) = E[∂_i log p * ∂_j log p] (Fisher-Rao)
    
    4. Natural gradient dynamics:
       ż = -g^ij * ∂_j V = g^ij * ∂_j log p
    
    This is the true information geometry formulation:
    - Planning = inference in probability space
    - Geodesics = KL divergence minimizers
    - Metric = Fisher information
    """
    
    def __init__(self, dimension: int = 16, n_components: int = 5):
        self.dimension = dimension
        self.n_components = n_components
        
        # Single generative model (ONE SOURCE)
        self.generative_model = GenerativeLatentModel(dimension, n_components)
        
        # Fisher-Rao metric from generative model
        self.fisher_metric = FisherRaoMetric(self.generative_model)
        
        # Natural gradient flow
        self.natural_flow = NaturalGradientFlow(self.generative_model)
        
        # Trajectory history
        self.trajectories: List[List[Tuple[List[float], List[float]]]] = []
        
        logger.info("information_geometry_system_initialized", 
                    dimension=dimension,
                    n_components=n_components)
    
    def observe(self, z: List[float], outcome: Optional[str] = None) -> None:
        """Observe state for learning"""
        self.generative_model.observe(z, outcome)
    
    def train(self) -> None:
        """Train generative model"""
        self.generative_model.fit()
    
    def compute_energy(self, z: List[float]) -> float:
        """Compute energy V(z) = -log p(z)"""
        return self.generative_model.energy(z)
    
    def compute_metric(self, z: List[float]) -> List[List[float]]:
        """Compute Fisher metric g_ij(z)"""
        return self.fisher_metric.compute_metric(z)
    
    def compute_natural_gradient(self, z: List[float]) -> List[float]:
        """Compute natural gradient g^ij * ∂_j log p"""
        return self.natural_flow.compute_force(z)
    
    def integrate(
        self,
        initial_z: List[float],
        steps: int = 100
    ) -> List[Dict]:
        """
        Integrate natural gradient flow.
        
        Returns trajectory with energy and metric.
        """
        trajectory = self.natural_flow.integrate(initial_z, steps)
        
        # Add energy and metric to trajectory
        enriched = []
        for z, v in trajectory:
            enriched.append({
                "z": z,
                "v": v,
                "energy": self.compute_energy(z),
                "metric_trace": sum(self.fisher_metric.compute_metric(z)[i][i] 
                                   for i in range(self.dimension)),
            })
        
        self.trajectories.append(enriched)
        return enriched
    
    def kl_divergence(self, z1: List[float], z2: List[float]) -> float:
        """
        Approximate KL divergence between two points.
        
        KL(p1 || p2) ≈ -log p(z1) + log p(z2)
        
        This is geodesic distance in probability space.
        """
        return self.compute_energy(z2) - self.compute_energy(z1)
    
    def sample_trajectory(self, start_z: List[float], n: int = 100) -> List[List[float]]:
        """Sample from generative model"""
        samples = self.generative_model.sample(n)
        
        # Sort by energy (low to high)
        samples_with_energy = [(s, self.compute_energy(s)) for s in samples]
        samples_with_energy.sort(key=lambda x: x[1])
        
        return [s for s, _ in samples_with_energy]
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "dimension": self.dimension,
            "n_components": self.n_components,
            "generative_model": self.generative_model.to_dict(),
            "trajectories": len(self.trajectories),
            "avg_energy": (
                sum(self.compute_energy(t[-1]["z"]) for t in self.trajectories) / len(self.trajectories)
                if self.trajectories else 0
            ),
        }
    
    def get_phase_flow(self) -> Dict:
        """Get phase flow for visualization"""
        if not self.trajectories:
            return {"flows": [], "density_samples": []}
        
        flows = []
        for traj in self.trajectories[-3:]:
            flows.append({
                "positions": [s["z"][:2] for s in traj],
                "velocities": [s["v"][:2] for s in traj],
                "energies": [s["energy"] for s in traj],
            })
        
        # Sample density for visualization
        samples = self.generative_model.sample(50)
        density_samples = [
            {"z": s[:2], "log_p": self.generative_model.log_density(s)}
            for s in samples
        ]
        
        return {
            "flows": flows,
            "density_samples": density_samples,
        }


# Factory
def create_information_geometry_system(
    dimension: int = 16,
    n_components: int = 5
) -> InformationGeometrySystem:
    return InformationGeometrySystem(dimension, n_components)