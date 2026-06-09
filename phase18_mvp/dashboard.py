"""
V-Field Dashboard Adapter

Provides visualization data for the trajectory-based V-field system.
Connects MVP v3 to visualization layer.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DashboardState:
    """Current state for dashboard visualization."""
    step: int
    V: float
    diversity: float
    collapse: float
    instability: float
    trend: float
    status: str
    
    selected_action: int
    action_candidates: List[Dict]
    
    latent_state: np.ndarray
    trajectory_ensemble: np.ndarray
    
    v_history: List[float]
    diversity_history: List[float]
    collapse_history: List[float]
    
    silent_collapse_detected: bool
    attractor_trap_score: float
    
    reward: float
    total_reward: float


class VFieldDashboardAdapter:
    """
    Adapter layer between V-field system and visualization.
    
    Converts trajectory-based V-field data to dashboard-compatible format.
    """
    
    def __init__(self, vfield, policy, dynamics, encoder):
        self.vfield = vfield
        self.policy = policy
        self.dynamics = dynamics
        self.encoder = encoder
        
        self.history = []
        self.total_reward = 0.0
        
    def update(self, obs: np.ndarray, action_idx: int, reward: float) -> DashboardState:
        """Update dashboard state after action."""
        # Encode observation
        z = self.encoder.encode(obs)
        
        # Get candidates
        candidates = self.policy.get_candidates(z, self.dynamics, self.vfield)
        
        # Generate trajectory ensemble for visualization
        trajectories = self.vfield.rollout(
            z, self.dynamics, self.policy.actions, 
            horizon=4, stochastic=True
        )
        
        # Create dashboard state
        signals = self.vfield.get_signals()
        
        self.total_reward += reward
        
        state = DashboardState(
            step=len(self.history),
            V=signals['V'],
            diversity=signals['diversity'],
            collapse=signals['collapse'],
            instability=signals['instability'],
            trend=signals['trend'],
            status=signals['status'],
            
            selected_action=action_idx,
            action_candidates=[
                {'idx': c[0], 'action': c[1].tolist(), 'V': c[2]}
                for c in candidates
            ],
            
            latent_state=z.copy(),
            trajectory_ensemble=trajectories,
            
            v_history=list(self.vfield.V_history),
            diversity_history=list(self.vfield.diversity_history),
            collapse_history=list(self.vfield.collapse_history),
            
            silent_collapse_detected=self.vfield.detect_silent_collapse(),
            attractor_trap_score=self.vfield.detect_attractor_trap(trajectories),
            
            reward=reward,
            total_reward=self.total_reward
        )
        
        self.history.append(state)
        
        return state
    
    def get_viz_data(self) -> Dict[str, Any]:
        """
        Get all visualization data for dashboard.
        """
        if not self.history:
            return {}
        
        latest = self.history[-1]
        
        # Time series data
        v_series = [h.V for h in self.history]
        diversity_series = [h.diversity for h in self.history]
        collapse_series = [h.collapse for h in self.history]
        
        # Trajectory endpoints for scatter plot
        endpoints = latest.trajectory_ensemble[:, -1, :]
        
        # Action comparison
        action_values = [c['V'] for c in latest.action_candidates]
        
        return {
            'current': {
                'step': latest.step,
                'V': latest.V,
                'diversity': latest.diversity,
                'collapse': latest.collapse,
                'instability': latest.instability,
                'trend': latest.trend,
                'status': latest.status,
            },
            'series': {
                'V': v_series,
                'diversity': diversity_series,
                'collapse': collapse_series,
            },
            'trajectories': {
                'endpoints': endpoints.tolist(),
                'num_trajectories': len(endpoints),
                'horizon': latest.trajectory_ensemble.shape[1],
            },
            'policy': {
                'selected_action': latest.selected_action,
                'action_candidates': latest.action_candidates,
                'max_V': max(action_values) if action_values else 0,
                'min_V': min(action_values) if action_values else 0,
            },
            'detection': {
                'silent_collapse': latest.silent_collapse_detected,
                'attractor_trap': latest.attractor_trap_score,
            },
            'rewards': {
                'current': latest.reward,
                'total': latest.total_reward,
            },
            'alerts': self._get_alerts(latest)
        }
    
    def _get_alerts(self, state: DashboardState) -> List[Dict]:
        """Generate alerts based on state."""
        alerts = []
        
        if state.status == 'CRITICAL':
            alerts.append({
                'type': 'critical',
                'message': 'V-field critical - system dying',
                'timestamp': datetime.now().isoformat()
            })
        elif state.status == 'WARNING':
            alerts.append({
                'type': 'warning',
                'message': 'V-field degrading',
                'timestamp': datetime.now().isoformat()
            })
        
        if state.silent_collapse_detected:
            alerts.append({
                'type': 'silent_collapse',
                'message': 'Silent collapse detected - V dropping while diversity stable',
                'timestamp': datetime.now().isoformat()
            })
        
        if state.attractor_trap_score > 0.5:
            alerts.append({
                'type': 'attractor_trap',
                'message': f'Attractor trap detected (score: {state.attractor_trap_score:.2f})',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts