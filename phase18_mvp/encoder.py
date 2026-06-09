"""
Layer 1: Encoder

z_t = encode(obs)
Simple structured embedding (not neural yet).
"""
import numpy as np


class Encoder:
    """
    Simple encoder: maps observation to latent state.
    
    For MVP: use PCA-like dimensionality reduction.
    In production: learned encoder.
    """
    
    def __init__(self, obs_dim: int, latent_dim: int):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        
        # Random projection (will be learned in production)
        self.projection = np.random.randn(latent_dim, obs_dim) * 0.1
        
        # Running statistics for normalization
        self.running_mean = np.zeros(obs_dim)
        self.running_std = np.ones(obs_dim)
        self.count = 0
    
    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Map observation to latent state."""
        self.count += 1
        
        # Update running statistics (simple online)
        alpha = 0.01
        self.running_mean = (1 - alpha) * self.running_mean + alpha * obs
        self.running_std = (1 - alpha) * self.running_std + alpha * (obs ** 2)
        self.running_std = np.sqrt(np.maximum(self.running_std, 1e-8))
        
        # Normalize
        normalized = (obs - self.running_mean) / self.running_std
        
        # Project
        z = self.projection @ normalized
        
        # Add small noise for exploration
        z = z + np.random.randn(self.latent_dim) * 0.01
        
        return z
    
    def reset_history(self):
        """Reset encoder statistics."""
        self.count = 0