"""
Simple environment for MVP testing.

The environment has:
- Moving target
- Obstacles
- Bounded space

Goal: demonstrate V-field behavior:
- Avoid collapse
- Preserve diversity
- React to degradation
"""
import numpy as np
from typing import Tuple


class SimpleEnv:
    """
    Simple 2D environment with moving target and obstacles.
    
    State: agent position (2D)
    Observation: agent position + target position + obstacle positions
    """
    
    def __init__(self):
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([5.0, 5.0])
        
        # Obstacles (will move)
        self.obstacles = [
            np.array([2.5, 2.5]),
            np.array([-2.5, 3.5]),
            np.array([3.0, -3.0]),
        ]
        
        self.bounds = 10.0
        self.step_count = 0
        
        # For moving target challenge
        self.target_pattern = 'circle'  # 'circle', 'random', 'fixed'
        self.target_angle = 0.0
        
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.agent_pos = np.array([0.0, 0.0])
        self.target_pos = np.array([5.0, 5.0])
        self.target_angle = 0.0
        self.step_count = 0
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        obs = np.concatenate([
            self.agent_pos,
            self.target_pos,
            self.obstacles[0],
            self.obstacles[1],
            self.obstacles[2],
        ])
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action.
        
        Returns:
            obs: observation
            reward: reward signal
            done: episode ended
        """
        self.step_count += 1
        
        # Update target position (moving target challenge)
        self._update_target()
        
        # Update agent position
        self.agent_pos = np.clip(
            self.agent_pos + action * 0.5,
            -self.bounds,
            self.bounds
        )
        
        # Compute reward
        dist_to_target = np.linalg.norm(self.agent_pos - self.target_pos)
        reward = -dist_to_target * 0.1
        
        # Obstacle penalty
        for obs_pos in self.obstacles:
            dist_to_obs = np.linalg.norm(self.agent_pos - obs_pos)
            if dist_to_obs < 1.0:
                reward -= 5.0  # penalty for hitting obstacle
        
        # Success reward
        if dist_to_target < 0.5:
            reward += 10.0
            self.target_pos = np.random.uniform(-self.bounds, self.bounds, 2)
        
        # Done condition
        done = self.step_count > 200
        
        obs = self._get_obs()
        return obs, reward, done
    
    def _update_target(self):
        """Update target position based on pattern."""
        if self.target_pattern == 'circle':
            self.target_angle += 0.1
            radius = 5.0
            self.target_pos = np.array([
                radius * np.cos(self.target_angle),
                radius * np.sin(self.target_angle)
            ])
        elif self.target_pattern == 'random':
            if self.step_count % 50 == 0:
                self.target_pos = np.random.uniform(-self.bounds, self.bounds, 2)
        # 'fixed' - target stays where it is
    
    def get_state(self) -> np.ndarray:
        """Get current state."""
        return self.agent_pos.copy()


class StressEnv:
    """
    Environment designed to stress-test V-field.
    
    Has modes that trigger:
    - Attractor collapse
    - Diversity death
    - V-field degradation
    """
    
    def __init__(self):
        self.mode = 'normal'  # 'normal', 'attractor_trap', 'narrowing', 'diversity_death'
        self.step_count = 0
        self.agent_pos = np.array([0.0, 0.0])
        
    def reset(self) -> np.ndarray:
        """Reset to normal mode."""
        self.mode = 'normal'
        self.step_count = 0
        self.agent_pos = np.array([0.0, 0.0])
        
        # Determine mode based on step
        if self.step_count < 100:
            self.mode = 'normal'
        elif self.step_count < 200:
            self.mode = 'attractor_trap'  # Single strong attractor
        elif self.step_count < 300:
            self.mode = 'narrowing'  # Space becomes narrower
        else:
            self.mode = 'diversity_death'  # Very few viable states
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        obs = np.concatenate([
            self.agent_pos,
            [self.step_count / 1000.0],  # time signal
            [1.0 if self.mode == 'attractor_trap' else 0.0],
            [1.0 if self.mode == 'narrowing' else 0.0],
            [1.0 if self.mode == 'diversity_death' else 0.0],
        ])
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Step in stress environment."""
        self.step_count += 1
        
        # Update mode
        if self.step_count == 100:
            self.mode = 'attractor_trap'
        elif self.step_count == 200:
            self.mode = 'narrowing'
        elif self.step_count == 300:
            self.mode = 'diversity_death'
        
        # Move agent
        self.agent_pos = self.agent_pos + action * 0.3
        
        # Apply environment-specific dynamics
        reward = 0.0
        
        if self.mode == 'attractor_trap':
            # Strong attractor at origin
            attractor = np.array([0.0, 0.0])
            self.agent_pos = 0.9 * self.agent_pos + 0.1 * attractor
            reward = -0.1  # slight penalty for existing
            
        elif self.mode == 'narrowing':
            # Space gets narrower
            self.agent_pos[0] = np.clip(self.agent_pos[0], -1.0, 1.0)
            self.agent_pos[1] = np.clip(self.agent_pos[1], -1.0, 1.0)
            reward = -0.05
            
        elif self.mode == 'diversity_death':
            # Only few viable positions
            viable_positions = [
                np.array([0.0, 0.0]),
                np.array([0.5, 0.5]),
                np.array([-0.5, -0.5]),
            ]
            # Move toward nearest viable
            dists = [np.linalg.norm(self.agent_pos - v) for v in viable_positions]
            nearest = viable_positions[np.argmin(dists)]
            self.agent_pos = 0.8 * self.agent_pos + 0.2 * nearest
            reward = -0.2
        
        # Add noise
        self.agent_pos += np.random.randn(2) * 0.05
        
        obs = self._get_obs()
        done = self.step_count > 400
        
        return obs, reward, done
    
    def get_mode(self) -> str:
        """Get current stress mode."""
        return self.mode