"""
Trajectory-based Viability Field (MVP v2)

V(z) = diversity of reachable trajectory endpoints
     - collapse penalty
"""
import numpy as np
from collections import deque
from typing import List, Tuple, Optional


class VField:
    """
    Trajectory-based Viability Field.
    
    Key shift:
      OLD: V(z) = quality of next step (single prediction)
      NEW: V(z) = geometry of possible future trajectories (ensemble)
    """
    
    def __init__(self, latent_dim: int = 8):
        self.latent_dim = latent_dim
        
        # V-field thresholds
        self.V_min = 0.3
        self.V_critical = 0.1
        
        # Stochastic ensemble parameters
        self.ensemble_size = 5  # Number of trajectories per action
        self.noise_std = 0.1  # Exploration noise
        
        # History for trend detection
        self.history_len = 100
        self.V_history = deque(maxlen=self.history_len)
        self.diversity_history = deque(maxlen=self.history_len)
        self.collapse_history = deque(maxlen=self.history_len)
        
        # Current signals
        self.current_V = 0.0
        self.current_diversity = 0.0
        self.current_collapse = 0.0
        self.current_instability = 0.0
        self.current_trend = 0.0
    
    def compute_V(self, trajectories: np.ndarray) -> float:
        """
        Compute V-field from trajectory ensemble.
        
        V = diversity + collapse + instability
        
        Args:
            trajectories: shape [num_trajs, horizon, latent_dim]
        
        Returns:
            V: viability score [0, 1]
        """
        # Extract endpoints of trajectories
        endpoints = trajectories[:, -1, :]  # [N, D]
        
        # 1. Diversity of endpoints (how spread are possible futures?)
        diversity = self._mean_pairwise_distance(endpoints)
        
        # 2. Collapse detection (variance of endpoints)
        endpoint_variance = np.mean(np.var(endpoints, axis=0))
        
        # 3. Temporal instability (variance over time steps)
        # Tracks if trajectories are folding/converging over time
        if trajectories.shape[1] > 2:
            time_variances = []
            for t in range(trajectories.shape[1]):
                step_points = trajectories[:, t, :]
                step_variance = np.mean(np.var(step_points, axis=0))
                time_variances.append(step_variance)
            
            # Instability = change in variance over time
            # High instability = trajectories folding/fanning
            trajectory_variance_trend = np.var(time_variances)
            
            # Normalize
            instability = np.tanh(trajectory_variance_trend * 5)
        else:
            instability = 0.0
            trajectory_variance_trend = 0.0
        
        # Also check midpoints variance
        if trajectories.shape[1] > 2:
            midpoints = trajectories[:, trajectories.shape[1] // 2, :]
            mid_variance = np.mean(np.var(midpoints, axis=0))
        else:
            mid_variance = endpoint_variance
        
        # Combine signals
        total_variance = endpoint_variance + mid_variance
        collapse = total_variance  # Higher variance = more spread = less collapse
        
        # Normalize to [0, 1] (heuristic)
        diversity_norm = np.tanh(diversity * 2)
        collapse_norm = np.tanh(collapse * 2)
        
        # V-field: balance diversity, spread, and instability
        V = 0.5 * diversity_norm + 0.3 * collapse_norm + 0.2 * instability
        V = float(np.clip(V, 0, 1))
        
        # Store signals
        self.current_V = V
        self.current_diversity = diversity_norm
        self.current_collapse = collapse_norm
        self.current_instability = instability
        
        # Track history
        self.V_history.append(V)
        self.diversity_history.append(diversity_norm)
        self.collapse_history.append(collapse_norm)
        
        # Compute trend
        if len(self.V_history) >= 5:
            recent = list(self.V_history)[-5:]
            self.current_trend = np.mean(np.diff(recent))
        else:
            self.current_trend = 0.0
        
        return V
    
    def _mean_pairwise_distance(self, X: np.ndarray) -> float:
        """Mean pairwise distance between points."""
        if len(X) <= 1:
            return 0.0
        
        dists = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dist = np.linalg.norm(X[i] - X[j])
                dists.append(dist)
        
        return np.mean(dists)
    
    def compute_V_from_predictions(self, z: np.ndarray, predictions: np.ndarray) -> float:
        """
        Legacy compatibility: compute V from single-step predictions.
        
        For trajectory-based system, use compute_V directly.
        """
        # Wrap single-step predictions as single-length trajectories
        if len(predictions.shape) == 2:
            # predictions shape: [N, D]
            # Convert to trajectories: [N, 1, D]
            trajectories = predictions.reshape(-1, 1, predictions.shape[1])
            return self.compute_V(trajectories)
        return 0.0
    
    def rollout(self, z: np.ndarray, dynamics, actions: np.ndarray, 
               horizon: int = 4, stochastic: bool = True) -> np.ndarray:
        """
        Generate trajectory ensemble from current state.
        
        Key improvement: stochastic ensemble per action.
        This gives true trajectory branching, not just action sensitivity.
        
        Args:
            z: current latent state
            dynamics: dynamics model
            actions: available actions
            horizon: number of steps to simulate
            stochastic: if True, generate multiple trajectories per action
        
        Returns:
            trajectories: shape [num_actions * ensemble_size, horizon+1, latent_dim]
        """
        trajectories = []
        
        for action in actions:
            if stochastic:
                # Generate multiple trajectories per action (true ensemble)
                for e in range(self.ensemble_size):
                    z0 = z.copy()
                    traj = [z0]
                    
                    current_z = z0
                    current_action = action
                    
                    for t in range(horizon):
                        # Predict next state
                        z_next = dynamics.predict(current_z, current_action)
                        
                        # Add exploration noise (model uncertainty)
                        z_next = z_next + np.random.normal(0, self.noise_std, size=z_next.shape)
                        
                        traj.append(z_next)
                        current_z = z_next
                    
                    trajectories.append(traj)
            else:
                # Single deterministic trajectory per action
                z0 = z.copy()
                traj = [z0]
                
                current_z = z0
                current_action = action
                
                for t in range(horizon):
                    z_next = dynamics.predict(current_z, current_action)
                    traj.append(z_next)
                    current_z = z_next
                
                trajectories.append(traj)
        
        return np.array(trajectories)
    
    def compute_V_from_state(self, z: np.ndarray, dynamics, 
                             actions: np.ndarray, horizon: int = 4) -> Tuple[float, np.ndarray]:
        """
        Compute V from state - generates trajectories and evaluates.
        
        Returns:
            V: viability score
            trajectories: generated trajectory ensemble
        """
        trajectories = self.rollout(z, dynamics, actions, horizon)
        V = self.compute_V(trajectories)
        return V, trajectories
    
    def get_status(self) -> str:
        """Get V-field status."""
        if len(self.V_history) < 5:
            return "WARMUP"
        
        V = self.current_V
        trend = self.current_trend
        
        if V < self.V_critical:
            return "CRITICAL - system dying"
        
        if V < self.V_min:
            return "WARNING - viability degrading"
        
        if trend < -0.01:
            return "DEGRADING - V dropping"
        
        return "HEALTHY"
    
    def is_healthy(self) -> bool:
        """Check if V-field is healthy."""
        return self.current_V > self.V_min
    
    def is_critical(self) -> bool:
        """Check if V-field is critical."""
        return self.current_V < self.V_critical
    
    def get_signals(self) -> dict:
        """Get all V-field signals."""
        return {
            'V': self.current_V,
            'diversity': self.current_diversity,
            'collapse': self.current_collapse,
            'instability': self.current_instability,
            'trend': self.current_trend,
            'status': self.get_status(),
            'history_len': len(self.V_history),
        }
    
    def reset_history(self):
        """Reset signal history."""
        self.V_history.clear()
        self.diversity_history.clear()
        self.collapse_history.clear()
    
    def detect_silent_collapse(self, window: int = 20) -> bool:
        """
        Detect silent collapse: V dropping while diversity stable.
        
        This is the key signal for "hidden future narrowing".
        """
        if len(self.V_history) < window:
            return False
        
        recent_V = list(self.V_history)[-window:]
        recent_div = list(self.diversity_history)[-window:]
        
        V_drop = recent_V[0] - recent_V[-1]
        div_change = recent_div[-1] - recent_div[0]
        
        # Silent collapse: V drops but diversity roughly stable
        # This means trajectory space is shrinking without obvious pattern
        if V_drop > 0.1 and abs(div_change) < 0.05:
            return True
        
        return False
    
    def detect_attractor_trap(self, trajectories: np.ndarray) -> float:
        """
        Detect if trajectories are converging to single attractor.
        
        Returns:
            trap_score: 0-1, higher = more trapped
        """
        if len(trajectories) < 2:
            return 0.0
        
        endpoints = trajectories[:, -1, :]
        
        # Check convergence
        spread = np.std(endpoints, axis=0)
        avg_spread = np.mean(spread)
        
        # Low spread = trapped to attractor
        trap_score = 1.0 - np.tanh(avg_spread * 10)
        
        return float(trap_score)