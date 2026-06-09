"""
Phase 18.5 - Probabilistic Trajectory Model Layer

Path B-lite: Bridge to world-model architecture
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class TransitionMode:
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    weight: float
    covariance: np.ndarray


class ProbabilisticTransitionModel:
    def __init__(self, latent_dim: int, action_dim: int, num_modes: int = 3):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.modes: List[TransitionMode] = []
        self.transitions: List[Tuple] = []
        self._init_modes()
    
    def _init_modes(self):
        for _ in range(self.num_modes):
            mode = TransitionMode(
                A=np.random.randn(self.latent_dim, self.latent_dim) * 0.1,
                b=np.random.randn(self.latent_dim, self.action_dim) * 0.2,
                c=np.zeros(self.latent_dim),
                weight=1.0 / self.num_modes,
                covariance=np.eye(self.latent_dim) * 0.05
            )
            self.modes.append(mode)
    
    def _ensure_vector(self, x, dim):
        if x is None:
            return np.zeros(dim)
        if isinstance(x, list):
            x = np.array(x)
        x = np.asarray(x).flatten()
        if len(x) < dim:
            x = np.pad(x, (0, dim - len(x)))
        elif len(x) > dim:
            x = x[:dim]
        return x
    
    def add_transition(self, z, a, z_next):
        z = self._ensure_vector(z, self.latent_dim)
        a = self._ensure_vector(a, self.action_dim)
        z_next = self._ensure_vector(z_next, self.latent_dim)
        self.transitions.append((z, a, z_next))
        if len(self.transitions) >= 50 and len(self.transitions) % 20 == 0:
            self._fit_modes()
    
    def _fit_modes(self):
        if len(self.transitions) < 50:
            return
        mode_assignments = [[] for _ in range(self.num_modes)]
        for z, a, z_next in self.transitions[-200:]:
            z = self._ensure_vector(z, self.latent_dim)
            a = self._ensure_vector(a, self.action_dim)
            residuals = []
            for i, mode in enumerate(self.modes):
                predicted = mode.A @ z + mode.b @ a + mode.c
                residual = np.linalg.norm(z_next - predicted)
                residuals.append((i, residual))
            nearest = min(residuals, key=lambda x: x[1])[0]
            mode_assignments[nearest].append((z, a, z_next))
        for i, mode in enumerate(self.modes):
            if len(mode_assignments[i]) < 5:
                continue
            X = np.array([np.concatenate([z, a]) for z, a, _ in mode_assignments[i]])
            Y = np.array([z_next for _, _, z_next in mode_assignments[i]])
            try:
                Theta = np.linalg.lstsq(X, Y, rcond=None)[0]
                mode.A = Theta[:self.latent_dim, :self.latent_dim]
                mode.b = Theta[:self.latent_dim, self.latent_dim:]
                mode.weight = len(mode_assignments[i]) / sum(len(m) for m in mode_assignments)
            except:
                pass
    
    def predict_multi_modal(self, z, a, top_k=3):
        z = self._ensure_vector(z, self.latent_dim)
        a = self._ensure_vector(a, self.action_dim)
        mode_predictions = []
        for i, mode in enumerate(self.modes):
            # Ensure mode matrices have correct shapes
            if mode.A.shape != (self.latent_dim, self.latent_dim):
                mode.A = np.random.randn(self.latent_dim, self.latent_dim) * 0.1
            if mode.b.shape != (self.latent_dim, self.action_dim):
                mode.b = np.random.randn(self.latent_dim, self.action_dim) * 0.2
            if mode.c.shape != (self.latent_dim,):
                mode.c = np.zeros(self.latent_dim)
            z_next = mode.A @ z + mode.b @ a + mode.c
            mode_predictions.append((i, z_next, mode.weight))
        mode_predictions.sort(key=lambda x: x[2], reverse=True)
        return mode_predictions[:top_k]
    
    def predict(self, z, a):
        z = self._ensure_vector(z, self.latent_dim)
        a = self._ensure_vector(a, self.action_dim)
        mode_predictions = []
        weighted_sum = np.zeros(self.latent_dim)
        for i, mode in enumerate(self.modes):
            noise = np.random.multivariate_normal(
                np.zeros(self.latent_dim), mode.covariance * 0.1)
            z_next = mode.A @ z + mode.b @ a + mode.c + noise
            mode_predictions.append((i, z_next, mode.weight))
            weighted_sum += mode.weight * z_next
        z_next = weighted_sum / sum(m.weight for m in self.modes)
        return z_next, mode_predictions
    
    def entropy(self, z, a):
        z = self._ensure_vector(z, self.latent_dim)
        a = self._ensure_vector(a, self.action_dim)
        mode_predictions = self.predict_multi_modal(z, a, top_k=self.num_modes)
        weights = np.array([m[2] for m in mode_predictions])
        weights = weights / (weights.sum() + 1e-8)
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        max_entropy = np.log(self.num_modes + 1e-8)
        return entropy / max_entropy if max_entropy > 0 else 0


class MultiModalVField:
    def __init__(self, latent_dim=8, num_modes=3, action_dim=2):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.transition_model = ProbabilisticTransitionModel(latent_dim, action_dim, num_modes)
        self.V_min = 0.3
        self.V_critical = 0.1
        self.current_V = 0.0
        self.current_diversity = 0.0
        self.current_divergence = 0.0
        self.current_entropy = 0.0
        self.V_history = deque(maxlen=100)
        self.diversity_history = deque(maxlen=100)
        self.divergence_history = deque(maxlen=100)
    
    def observe(self, z, a, z_next):
        self.transition_model.add_transition(z, a, z_next)
    
    def compute_V(self, z, a):
        z = self.transition_model._ensure_vector(z, self.latent_dim)
        a = self.transition_model._ensure_vector(a, self.action_dim)
        mode_predictions = self.transition_model.predict_multi_modal(z, a, top_k=self.transition_model.num_modes)
        weights = np.array([m[2] for m in mode_predictions])
        weights_norm = weights / (weights.sum() + 1e-8)
        mode_diversity = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
        mode_diversity = mode_diversity / (np.log(len(weights) + 1e-8) if len(weights) > 1 else 1)
        endpoints = np.array([m[1] for m in mode_predictions])
        if len(endpoints) > 1:
            pairwise_dists = []
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    pairwise_dists.append(np.linalg.norm(endpoints[i] - endpoints[j]))
            mode_divergence = np.mean(pairwise_dists) if pairwise_dists else 0
        else:
            mode_divergence = 0
        mode_entropy = self.transition_model.entropy(z, a)
        mode_divergence_norm = np.tanh(mode_divergence * 2)
        V = 0.4 * mode_diversity + 0.4 * mode_divergence_norm + 0.2 * mode_entropy
        V = float(np.clip(V, 0, 1))
        self.current_V = V
        self.current_diversity = mode_diversity
        self.current_divergence = mode_divergence_norm
        self.current_entropy = mode_entropy
        self.V_history.append(V)
        self.diversity_history.append(mode_diversity)
        self.divergence_history.append(mode_divergence_norm)
        return V
    
    def get_status(self):
        if len(self.V_history) < 5:
            return "WARMUP"
        if self.current_V < self.V_critical:
            return "CRITICAL"
        if self.current_V < self.V_min:
            return "WARNING"
        return "HEALTHY"
    
    def get_signals(self):
        return {
            'V': self.current_V,
            'mode_diversity': self.current_diversity,
            'mode_divergence': self.current_divergence,
            'mode_entropy': self.current_entropy,
            'status': self.get_status(),
        }