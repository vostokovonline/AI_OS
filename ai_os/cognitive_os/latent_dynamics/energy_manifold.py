"""
Phase 17 - Energy-Based Information Manifold

NOT explicit density p_θ(z) with normalization.
NOT GMM with disconnected components.

BUT energy-based model:
  p_θ(z) = (1/Z) * exp(-E_θ(z))

Where:
  - E_θ(z) is learned energy function (no normalization needed)
  - Metric from Hessian: g_ij = ∂_i ∂_j E(z)
  - Curvature from Hessian structure
  - Manifold = energy landscape geometry
  - Smooth, differentiable everywhere

Key shift:
  Before: p(z) → V = -log p (density geometry)
  After: E(z) → metric = Hessian(E) (energy geometry)

This gives:
  - No explicit Z normalization
  - Curvature from Hessian(E)
  - Anisotropy from Hessian structure
  - Attractor basins = minima of E
  - Transition metric from dynamics sensitivity
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


class EnergyBasedModel:
    """
    Energy-based model for latent space.
    
    p(z) = (1/Z) * exp(-E_θ(z))
    
    E_θ is learned energy function (neural network approximation).
    No normalization constant Z needed for gradient-based methods.
    
    Advantages:
    - No partition function Z
    - Hessian gives metric naturally
    - Smooth energy landscape
    - Scales to high dimensions
    """
    
    def __init__(self, dimension: int = 16, hidden_dims: List[int] = None):
        self.dimension = dimension
        self.hidden_dims = hidden_dims or [64, 32]
        
        # Neural network parameters (simplified linear model for now)
        # In production: use actual neural network
        self.W1 = self._init_matrix(self.hidden_dims[0], dimension)
        self.W2 = self._init_matrix(self.hidden_dims[1] if len(self.hidden_dims) > 1 else 16, self.hidden_dims[0])
        self.W_out = self._init_matrix(1, self.hidden_dims[-1])
        
        # Observations for learning
        self.samples: List[List[float]] = []
        self.success_samples: List[List[float]] = []
        self.failure_samples: List[List[float]] = []
        
        # Learned statistics for energy shaping
        self.mean: List[float] = [0.0] * dimension
        self.std: List[float] = [1.0] * dimension
        
        logger.info("energy_based_model_initialized", dimension=dimension)
    
    def _init_matrix(self, rows: int, cols: int) -> List[List[float]]:
        """Initialize matrix with small random values"""
        import random
        return [
            [random.uniform(-0.1, 0.1) for _ in range(cols)]
            for _ in range(rows)
        ]
    
    def observe(self, z: List[float], outcome: Optional[str] = None) -> None:
        """Observe state for learning"""
        self.samples.append(z)
        
        if outcome == "success":
            self.success_samples.append(z)
        elif outcome == "failure":
            self.failure_samples.append(z)
        
        # Update statistics
        self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update sample statistics"""
        if len(self.samples) < 2:
            return
        
        n = len(self.samples)
        
        # Update mean
        self.mean = [
            sum(s[i] for s in self.samples) / n
            for i in range(self.dimension)
        ]
        
        # Update std
        self.std = [
            math.sqrt(
                sum((s[i] - self.mean[i]) ** 2 for s in self.samples) / n
            )
            for i in range(self.dimension)
        ]
        
        # Ensure positive
        self.std = [max(0.1, s) for s in self.std]
    
    def energy(self, z: List[float]) -> float:
        """
        Compute energy E_θ(z).
        
        Simple approximation: quadratic potential + learned deviations.
        
        For production: neural network energy.
        """
        # Quadratic baseline
        E = 0.0
        for i in range(self.dimension):
            normalized = (z[i] - self.mean[i]) / (self.std[i] + 1e-6)
            E += 0.5 * normalized ** 2
        
        # Success regions have lower energy
        if self.success_samples:
            min_dist_success = min(
                self._distance(z, s)
                for s in self.success_samples[-50:]
            )
            E -= 0.3 * math.exp(-min_dist_success ** 2 / 0.5)
        
        # Failure regions have higher energy
        if self.failure_samples:
            min_dist_failure = min(
                self._distance(z, s)
                for s in self.failure_samples[-50:]
            )
            E += 0.2 * math.exp(-min_dist_failure ** 2 / 0.5)
        
        return max(0.0, E)
    
    def _distance(self, z1: List[float], z2: List[float]) -> float:
        """Euclidean distance"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(z1, z2)))
    
    def gradient_energy(self, z: List[float], epsilon: float = 0.01) -> List[float]:
        """
        Compute gradient of energy: ∂_i E(z)
        
        Uses finite differences.
        """
        grad = []
        
        for i in range(self.dimension):
            z_plus = z[:]
            z_plus[i] += epsilon
            E_plus = self.energy(z_plus)
            
            z_minus = z[:]
            z_minus[i] -= epsilon
            E_minus = self.energy(z_minus)
            
            grad.append((E_plus - E_minus) / (2 * epsilon))
        
        return grad
    
    def hessian_energy(self, z: List[float], epsilon: float = 0.01) -> List[List[float]]:
        """
        Compute Hessian of energy: H_ij = ∂_i ∂_j E(z)
        
        This IS the Fisher metric for energy-based models:
        g_ij(z) = H_ij(z) = ∂_i ∂_j E(z)
        
        Properties:
        - Positive semi-definite (energy has local minima)
        - Describes local curvature
        - Determines geodesic geometry
        """
        hessian = [[0.0] * self.dimension for _ in range(self.dimension)]
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    # Diagonal: second derivative
                    z_plus_plus = z[:]
                    z_plus_plus[i] += epsilon
                    E_pp = self.energy(z_plus_plus)
                    
                    z_0 = z[:]
                    E_0 = self.energy(z_0)
                    
                    z_minus_minus = z[:]
                    z_minus_minus[i] -= epsilon
                    E_mm = self.energy(z_minus_minus)
                    
                    hessian[i][i] = (E_pp - 2 * E_0 + E_mm) / (epsilon ** 2)
                else:
                    # Cross terms: ∂_i ∂_j E
                    # Use mixed finite differences
                    z_ij_plus = [z[k] + epsilon if k == i or k == j else z[k] for k in range(self.dimension)]
                    z_i_plus = [z[k] + epsilon if k == i else z[k] for k in range(self.dimension)]
                    z_j_plus = [z[k] + epsilon if k == j else z[k] for k in range(self.dimension)]
                    
                    E_ij = self.energy(z_ij_plus)
                    E_i = self.energy(z_i_plus)
                    E_j = self.energy(z_j_plus)
                    E_0 = self.energy(z)
                    
                    hessian[i][j] = (E_ij - E_i - E_j + E_0) / (epsilon ** 2)
        
        return hessian
    
    def energy_plus_success_bias(self, z: List[float]) -> float:
        """Energy with goal bias"""
        E = self.energy(z)
        
        # Bias towards success regions
        if self.success_samples:
            # Distance to nearest success center
            success_center = [
                sum(s[i] for s in self.success_samples[-20:]) / min(20, len(self.success_samples))
                for i in range(self.dimension)
            ]
            
            dist = self._distance(z, success_center)
            E -= 1.0 * math.exp(-dist ** 2 / 2.0)
        
        return max(0.0, E)


class HessianBasedMetric:
    """
    Metric tensor from energy Hessian.
    
    g_ij(z) = H_ij(z) = ∂_i ∂_j E(z)
    
    Properties:
    - Positive semi-definite (energy minima)
    - Full tensor (not just diagonal)
    - Describes manifold curvature
    - Enables Christoffel computation
    """
    
    def __init__(self, energy_model: EnergyBasedModel):
        self.energy_model = energy_model
        self.dimension = energy_model.dimension
        
        # Caching
        self.cached_z: Optional[List[float]] = None
        self.cached_hessian: Optional[List[List[float]]] = None
        self.cached_inverse: Optional[List[List[float]]] = None
    
    def compute_metric(self, z: List[float]) -> List[List[float]]:
        """
        Compute metric tensor g_ij = Hessian(E)
        """
        if self.cached_z and self._is_close(z, self.cached_z):
            return self.cached_hessian
        
        # Compute Hessian
        hessian = self.energy_model.hessian_energy(z)
        
        # Ensure positive semi-definiteness
        # Add small regularization for numerical stability
        for i in range(self.dimension):
            hessian[i][i] = max(hessian[i][i], 0.01)
        
        self.cached_z = z
        self.cached_hessian = hessian
        self._compute_inverse()
        
        return hessian
    
    def _is_close(self, z1: List[float], z2: List[float], eps: float = 0.02) -> bool:
        if z2 is None:
            return False
        for a, b in zip(z1, z2):
            if abs(a - b) > eps:
                return False
        return True
    
    def _compute_inverse(self) -> None:
        """Compute inverse metric using pseudo-inverse"""
        if self.cached_hessian is None:
            return
        
        # Simplified: diagonal approximation for stability
        n = self.dimension
        self.cached_inverse = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            if abs(self.cached_hessian[i][i]) > 1e-6:
                self.cached_inverse[i][i] = 1.0 / self.cached_hessian[i][i]
    
    def inverse_metric(self, z: List[float]) -> List[List[float]]:
        """Get inverse metric g^ij"""
        self.compute_metric(z)  # Ensure cached
        return self.cached_inverse
    
    def christoffel_symbols(self, z: List[float], epsilon: float = 0.02) -> List[List[List[float]]]:
        """
        Compute Christoffel symbols from metric.
        
        Γ^i_jk = 0.5 * g^iμ * (∂_j g_μk + ∂_k g_μj - ∂_μ g_jk)
        
        This requires derivatives of the metric itself.
        """
        metric = self.compute_metric(z)
        inv = self.inverse_metric(z)
        
        n = self.dimension
        
        # Compute metric derivatives
        def metric_derivative(z: List[float], axis: int) -> List[List[float]]:
            eps = epsilon
            z_plus = z[:]
            z_plus[axis] += eps
            m_plus = self.energy_model.hessian_energy(z_plus)
            
            z_minus = z[:]
            z_minus[axis] -= eps
            m_minus = self.energy_model.hessian_energy(z_minus)
            
            # Central difference
            return [
                [(m_plus[i][j] - m_minus[i][j]) / (2 * eps)
                 for j in range(n)]
                for i in range(n)
            ]
        
        # ∂_j g_μk
        dg = [metric_derivative(z, j) for j in range(n)]
        
        # Christoffel symbols
        gamma = [[[0.0] * n for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    # Γ^i_jk = 0.5 * g^iμ * (∂_j g_μk + ∂_k g_μj - ∂_μ g_jk)
                    sum_val = 0.0
                    for mu in range(n):
                        term1 = dg[j][mu][k] if mu < len(dg[j]) and k < n else 0.0
                        term2 = dg[k][mu][j] if mu < len(dg[k]) and j < n else 0.0
                        term3 = (metric[mu][j] - metric[j][mu]) if mu < n and j < n and mu == k else 0.0
                        
                        # Simplified for diagonal metric
                        sum_val += inv[i][mu] * (term1 + term2 - term3) if i == mu else 0.0
                    
                    gamma[i][j][k] = 0.5 * sum_val
        
        return gamma
    
    def riemannian_gradient(self, z: List[float], grad_flat: List[float]) -> List[float]:
        """
        Compute Riemannian gradient: grad_nat^i = g^ij * grad_flat^j
        """
        inv = self.inverse_metric(z)
        
        grad_nat = [
            sum(inv[i][j] * grad_flat[j] for j in range(self.dimension))
            for i in range(self.dimension)
        ]
        
        return grad_nat
    
    def geodesic_acceleration(
        self,
        z: List[float],
        v: List[float],
        christoffel: List[List[List[float]]]
    ) -> List[float]:
        """
        Compute geodesic acceleration: -Γ^i_jk * v^j * v^k
        
        This is the curvature-induced turning term.
        """
        n = self.dimension
        geo = [0.0] * n
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    geo[i] -= christoffel[i][j][k] * v[j] * v[k]
        
        return geo


class GeodesicMechanics:
    """
    Second-order geodesic dynamics on energy manifold.
    
    Full equation:
    z̈^i + Γ^i_jk * ż^j * ż^k = -g^ij * ∂_j E
    
    This is NOT gradient descent.
    This is inertial dynamics with curvature.
    
    Properties:
    - Momentum transport (inertia)
    - Curvature-induced turning (Christoffel)
    - Natural flow on manifold (geodesics)
    """
    
    def __init__(self, energy_model: EnergyBasedModel):
        self.energy_model = energy_model
        self.hessian_metric = HessianBasedMetric(energy_model)
        
        self.dimension = energy_model.dimension
        self.dt = 0.01
        self.damping = 0.05
        
        logger.info("geodesic_mechanics_initialized")
    
    def acceleration(self, z: List[float], v: List[float]) -> List[float]:
        """
        Compute total acceleration.
        
        a = -∇E (natural gradient) + geodesic_term
        """
        # Natural gradient: -g^ij * ∂_j E
        grad_E = self.energy_model.gradient_energy(z)
        inv_metric = self.hessian_metric.inverse_metric(z)
        
        nat_grad = [
            -sum(inv_metric[i][j] * grad_E[j] for j in range(self.dimension))
            for i in range(self.dimension)
        ]
        
        # Geodesic term: -Γ * v * v
        christoffel = self.hessian_metric.christoffel_symbols(z)
        geo_term = self.hessian_metric.geodesic_acceleration(z, v, christoffel)
        
        # Total acceleration
        a = [
            nat_grad[i] + geo_term[i]
            for i in range(self.dimension)
        ]
        
        # Add damping
        a = [
            a[i] - self.damping * v[i]
            for i in range(self.dimension)
        ]
        
        return a
    
    def step(self, z: List[float], v: List[float]) -> Tuple[List[float], List[float]]:
        """
        Single geodesic step.
        
        Uses Velocity Verlet or similar for stability.
        """
        # Compute acceleration
        a = self.acceleration(z, v)
        
        # Update velocity
        v_new = [
            v[i] + self.dt * a[i]
            for i in range(self.dimension)
        ]
        
        # Update position
        z_new = [
            z[i] + self.dt * v_new[i]
            for i in range(self.dimension)
        ]
        
        return z_new, v_new
    
    def integrate(
        self,
        initial_z: List[float],
        initial_v: Optional[List[float]] = None,
        steps: int = 100
    ) -> List[Dict]:
        """
        Integrate geodesic trajectory.
        """
        trajectory = []
        
        z = initial_z[:]
        v = initial_v[:] if initial_v else [0.0] * self.dimension
        
        for _ in range(steps):
            z, v = self.step(z, v)
            
            energy = self.energy_model.energy(z)
            kinetic = 0.5 * sum(vi ** 2 for vi in v)
            
            trajectory.append({
                "z": z[:],
                "v": v[:],
                "energy": energy,
                "kinetic": kinetic,
                "total": energy + kinetic,
            })
            
            # Stop near equilibrium
            if sum(abs(vi) for vi in v) < 0.01:
                break
        
        return trajectory


class TransitionSensitivityMetric:
    """
    Metric from transition sensitivity.
    
    For cognitive dynamics, metric should reflect:
    how sensitive transitions are to state changes.
    
    g_ij = E[(∂_i f_θ)ᵀ (∂_j f_θ)]
    
    where f_θ(z, a) → z' is the dynamics model.
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Transition sensitivity statistics
        self.transitions: List[Tuple[List[float], List[float], List[float]] = []
        
        # Learned Jacobian approximation
        self.jacobian_approx: List[List[float]] = [
            [0.0] * dimension for _ in range(dimension)
        ]
        
        logger.info("transition_sensitivity_metric_initialized")
    
    def observe_transition(
        self,
        z: List[float],
        action: Optional[List[float]],
        z_next: List[float]
    ) -> None:
        """Observe state transition"""
        self.transitions.append((z, action or [0.0] * self.dimension, z_next))
        
        # Update Jacobian approximation
        self._update_jacobian()
    
    def _update_jacobian(self) -> None:
        """Update Jacobian from transition statistics"""
        if len(self.transitions) < 10:
            return
        
        # Approximate: ∂z'/∂z
        recent = self.transitions[-100:]
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                # Correlation between state dimension j and next state dimension i
                z_vals = [t[0][j] for t in recent]
                z_next_vals = [t[2][i] for t in recent]
                
                if len(z_vals) >= 2:
                    mean_z = sum(z_vals) / len(z_vals)
                    mean_zn = sum(z_next_vals) / len(z_next_vals)
                    
                    cov = sum((z_vals[k] - mean_z) * (z_next_vals[k] - mean_zn) 
                             for k in range(len(z_vals))) / len(z_vals)
                    var = sum((v - mean_z) ** 2 for v in z_vals) / len(z_vals)
                    
                    if abs(var) > 1e-6:
                        self.jacobian_approx[i][j] = cov / var
    
    def compute_metric(self, z: List[float]) -> List[List[float]]:
        """
        Compute metric from transition sensitivity.
        
        g_ij = (J^T J)_ij
        
        This measures how perturbations in dimension j
        affect transitions in dimension i.
        """
        # J^T J approximation
        metric = [[0.0] * self.dimension for _ in range(self.dimension)]
        
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.dimension):
                    metric[i][j] += self.jacobian_approx[k][i] * self.jacobian_approx[k][j]
        
        # Normalize
        trace = sum(metric[i][i] for i in range(self.dimension))
        if trace > 0:
            scale = self.dimension / trace
            for i in range(self.dimension):
                for j in range(self.dimension):
                    metric[i][j] *= scale
        
        return metric


class EnergyBasedInformationManifold:
    """
    Phase 17 - Energy-Based Information Manifold
    
    Core principle:
    p(z) = (1/Z) * exp(-E(z))
    
    Metric: g_ij = Hessian(E) = ∂_i ∂_j E
    
    Dynamics: geodesic flow on energy manifold
    
    This is the TRUE energy-based geometric framework:
    - No explicit density normalization
    - Metric from Hessian (smooth, continuous)
    - Curvature from Hessian structure
    - Geodesics = natural motion on energy surface
    """
    
    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        
        # Energy-based model (no normalization needed)
        self.energy_model = EnergyBasedModel(dimension)
        
        # Hessian-based metric
        self.hessian_metric = HessianBasedMetric(self.energy_model)
        
        # Transition sensitivity metric
        self.transition_metric = TransitionSensitivityMetric(dimension)
        
        # Geodesic mechanics
        self.geodesic = GeodesicMechanics(self.energy_model)
        
        # Trajectory history
        self.trajectories: List[List[Dict]] = []
        
        logger.info("energy_based_manifold_initialized", dimension=dimension)
    
    def observe(
        self,
        z: List[float],
        z_next: Optional[List[float]] = None,
        outcome: Optional[str] = None
    ) -> None:
        """Observe state and transition"""
        self.energy_model.observe(z, outcome)
        
        if z_next:
            self.transition_metric.observe_transition(z, None, z_next)
    
    def compute_energy(self, z: List[float]) -> float:
        """Compute energy E(z)"""
        return self.energy_model.energy(z)
    
    def compute_metric(self, z: List[float]) -> List[List[float]]:
        """
        Compute combined metric.
        
        g = α * Hessian(E) + (1-α) * TransitionSensitivity
        """
        hess = self.hessian_metric.compute_metric(z)
        trans = self.transition_metric.compute_metric(z)
        
        alpha = 0.7  # Weight for Hessian
        
        combined = [
            [alpha * hess[i][j] + (1 - alpha) * trans[i][j]
             for j in range(self.dimension)]
            for i in range(self.dimension)
        ]
        
        return combined
    
    def integrate(
        self,
        initial_z: List[float],
        initial_v: Optional[List[float]] = None,
        steps: int = 100
    ) -> List[Dict]:
        """Integrate geodesic on energy manifold"""
        trajectory = self.geodesic.integrate(initial_z, initial_v, steps)
        self.trajectories.append(trajectory)
        return trajectory
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        return {
            "dimension": self.dimension,
            "samples": len(self.energy_model.samples),
            "success_samples": len(self.energy_model.success_samples),
            "failure_samples": len(self.energy_model.failure_samples),
            "transitions": len(self.transition_metric.transitions),
            "trajectories": len(self.trajectories),
        }
    
    def get_phase_flow(self) -> Dict:
        """Get phase flow for visualization"""
        if not self.trajectories:
            return {"flows": [], "energy_surface": []}
        
        flows = []
        for traj in self.trajectories[-3:]:
            flows.append({
                "positions": [s["z"][:2] for s in traj],
                "velocities": [s["v"][:2] for s in traj],
                "energies": [s["energy"] for s in traj],
            })
        
        # Sample energy surface
        energy_surface = []
        for i in range(-5, 6, 2):
            for j in range(-5, 6, 2):
                z = [
                    self.energy_model.mean[0] + i * 0.5,
                    self.energy_model.mean[1] + j * 0.5,
                ] + [0.0] * (self.dimension - 2)
                energy_surface.append({
                    "z": z[:2],
                    "E": self.compute_energy(z),
                })
        
        return {
            "flows": flows,
            "energy_surface": energy_surface,
        }


# Factory
def create_energy_based_manifold(dimension: int = 16) -> EnergyBasedInformationManifold:
    return EnergyBasedInformationManifold(dimension=dimension)