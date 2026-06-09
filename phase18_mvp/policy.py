"""
Trajectory-based Policy

a* = argmax V(trajectory_ensemble)

Now we select action based on trajectory-space viability,
not single-step prediction quality.
"""
import numpy as np
from typing import List, Tuple, Optional


class Policy:
    """
    V-field based policy using trajectory rollout.
    
    Key shift:
      OLD: Select action by max V(single-step prediction)
      NEW: Select action by max V(trajectory ensemble)
    """
    
    def __init__(self, action_dim: int, action_range: float = 1.0):
        self.action_dim = action_dim
        self.action_range = action_range
        
        # Available actions
        self.actions = self._generate_actions(num_actions=5)
        
        # Trajectory horizon
        self.horizon = 4
        
        # Stochastic rollout enabled by default
        self.use_stochastic = True
    
    def _generate_actions(self, num_actions: int) -> np.ndarray:
        """Generate discrete action candidates."""
        actions = []
        for i in range(num_actions):
            action = np.zeros(self.action_dim)
            action[i % self.action_dim] = 1.0
            actions.append(action)
        
        # Add zero action
        actions.append(np.zeros(self.action_dim))
        
        return np.array(actions)
    
    def set_actions(self, actions: np.ndarray):
        """Set available actions."""
        self.actions = actions
    
    def select_action(self, z: np.ndarray, dynamics, vfield) -> Tuple[int, np.ndarray, float]:
        """
        Select action that maximizes stochastic trajectory-based V.
        
        Returns:
            action_idx: index of selected action
            action: action vector
            V: trajectory-space viability score
        """
        best_V = -1
        best_action = None
        best_idx = 0
        best_trajectories = None
        
        for idx, action in enumerate(self.actions):
            # Generate stochastic trajectory ensemble for this action
            trajectories = vfield.rollout(
                z=z,
                dynamics=dynamics,
                actions=np.array([action]),
                horizon=self.horizon,
                stochastic=self.use_stochastic
            )
            
            # Compute V from trajectory ensemble
            V = vfield.compute_V(trajectories)
            
            if V > best_V:
                best_V = V
                best_action = action
                best_idx = idx
                best_trajectories = trajectories
        
        return best_idx, best_action, best_V
    
    def get_candidates(self, z: np.ndarray, dynamics, vfield) -> List[Tuple[int, np.ndarray, float]]:
        """
        Get all action candidates with their V scores.
        
        Useful for analysis and debugging.
        """
        candidates = []
        
        for idx, action in enumerate(self.actions):
            trajectories = vfield.rollout(
                z=z,
                dynamics=dynamics,
                actions=np.array([action]),
                horizon=self.horizon,
                stochastic=self.use_stochastic
            )
            
            V = vfield.compute_V(trajectories)
            candidates.append((idx, action, V))
        
        return candidates
    
    def select_with_depth(self, z: np.ndarray, dynamics, vfield, 
                          horizon: int = 4) -> Tuple[int, np.ndarray, float]:
        """
        Select action with custom trajectory depth.
        """
        best_V = -1
        best_action = None
        best_idx = 0
        
        for idx, action in enumerate(self.actions):
            trajectories = vfield.rollout(
                z=z,
                dynamics=dynamics,
                actions=np.array([action]),
                horizon=horizon
            )
            
            V = vfield.compute_V(trajectories)
            
            if V > best_V:
                best_V = V
                best_action = action
                best_idx = idx
        
        return best_idx, best_action, best_V