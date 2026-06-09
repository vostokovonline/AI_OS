"""
Layer 2: Dynamics Model

z_next = f(z_t, a_t)
Simple linear dynamics + noise.
"""
import numpy as np


class Dynamics:
    """
    Linear dynamics model.
    
    z_next = A @ z_t + B @ a_t + noise
    
    For MVP: random matrices, can be learned.
    """
    
    def __init__(self, latent_dim: int, action_dim: int):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Linear transition matrix (will be learned)
        self.A = np.random.randn(latent_dim, latent_dim) * 0.1
        # Action influence matrix
        self.B = np.random.randn(latent_dim, action_dim) * 0.2
        
        # Add stable eigenvalues (eigenvalues < 1 for stability)
        eigenvalues = np.random.rand(latent_dim) * 0.8
        self.A = self._stabilize_matrix(self.A, eigenvalues)
        
        # Dynamics noise
        self.noise_std = 0.05
    
    def _stabilize_matrix(self, M: np.ndarray, eigenvalues: np.ndarray) -> np.ndarray:
        """Ensure matrix has given eigenvalues (simplified)."""
        # Simple approach: scale matrix to have spectral radius < 1
        spectral_radius = np.max(np.abs(np.linalg.eigvals(M)))
        if spectral_radius > 0.9:
            M = M * 0.9 / spectral_radius
        return M
    
    def predict(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Predict next state given current state and action."""
        noise = np.random.randn(self.latent_dim) * self.noise_std
        z_next = self.A @ z + self.B @ a + noise
        return z_next
    
    def predict_batch(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Predict next states for multiple actions."""
        predictions = []
        for a in actions:
            z_next = self.predict(z, a)
            predictions.append(z_next)
        return np.array(predictions)