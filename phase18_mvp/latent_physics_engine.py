"""
Phase 18.9 - Latent Physics Engine (Energy-Consistent Flow Layer)

Key additions vs Phase 18.8:
1. True potential E(z) (not density estimator)
2. Physics-consistent flow F(z,a) = -λ∇E + action_response
3. Consistency loss: L = ||F + ∇E||² + ||z_next - (z + F)||²
4. Attractor detection (stable states)
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Attractor:
    """Stable state (attractor)."""
    position: np.ndarray
    basin_radius: float
    strength: float
    transitions_in: int
    mode_id: int
    
    def attract(self, z: np.ndarray) -> np.ndarray:
        direction = self.position - z
        distance = np.linalg.norm(direction)
        if distance < 1e-6:
            return np.zeros_like(z)
        strength = self.strength * np.exp(-distance / self.basin_radius)
        return strength * direction / distance


class TruePotentialField:
    """True potential field E(z), not density estimator."""
    
    def __init__(
        self,
        latent_dim: int,
        trajectory_weight: float = 0.5,
        stability_weight: float = 0.3,
        reachability_weight: float = 0.2
    ):
        self.latent_dim = latent_dim
        self.trajectory_weight = trajectory_weight
        self.stability_weight = stability_weight
        self.reachability_weight = reachability_weight
        
        self.transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.trajectories: deque = deque(maxlen=100)
        self.attractors: List[Attractor] = []
    
    def add_transition(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray):
        self.transitions.append((
            np.asarray(z).flatten(),
            np.asarray(a).flatten(),
            np.asarray(z_next).flatten()
        ))
        if len(self.transitions) > 1000:
            self.transitions.pop(0)
    
    def compute_trajectory_consistency(self, z: np.ndarray) -> float:
        if len(self.trajectories) < 3:
            return 0.5
        
        passing = 0
        for traj in self.trajectories:
            for point in traj:
                if np.linalg.norm(point - z) < 1.0:
                    passing += 1
                    break
        
        return passing / max(len(self.trajectories), 1)
    
    def compute_stability(self, z: np.ndarray) -> float:
        if len(self.transitions) < 10:
            return 0.5
        
        nearby = []
        for z_i, a_i, z_next_i in self.transitions:
            if np.linalg.norm(z_i - z) < 2.0:
                nearby.append((z_i, z_next_i))
        
        if len(nearby) < 3:
            return 0.5
        
        directions = []
        for z_i, z_next_i in nearby:
            d = z_next_i - z_i
            if np.linalg.norm(d) > 1e-6:
                directions.append(d / np.linalg.norm(d))
        
        if len(directions) < 2:
            return 0.5
        
        mean_dir = np.mean(directions, axis=0)
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)
        
        coherences = [np.dot(d, mean_dir) for d in directions]
        return (np.mean(coherences) + 1) / 2
    
    def compute_reachability(self, z: np.ndarray) -> float:
        if len(self.transitions) < 10:
            return 0.5
        
        ending = sum(1 for t in self.transitions if np.linalg.norm(t[2] - z) < 1.5)
        starting = sum(1 for t in self.transitions if np.linalg.norm(t[0] - z) < 1.5)
        
        return (ending + starting) / (2 * len(self.transitions) + 1e-8)
    
    def compute_E(self, z: np.ndarray) -> float:
        consistency = self.compute_trajectory_consistency(z)
        stability = self.compute_stability(z)
        reachability = self.compute_reachability(z)
        
        P = (
            self.trajectory_weight * consistency +
            self.stability_weight * stability +
            self.reachability_weight * reachability
        )
        P = np.clip(P, 1e-8, 1.0)
        
        E = -np.log(P)
        return float(np.tanh(E / 5.0))
    
    def compute_gradient_E(self, z: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
        z = np.asarray(z).flatten()
        dim = len(z)
        grad = np.zeros(dim)
        
        for i in range(dim):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += epsilon
            z_minus[i] -= epsilon
            
            E_plus = self.compute_E(z_plus)
            E_minus = self.compute_E(z_minus)
            grad[i] = (E_plus - E_minus) / (2 * epsilon)
        
        return grad
    
    def detect_attractors(self):
        if len(self.transitions) < 50:
            return
        
        endpoints = np.array([t[2] for t in self.transitions])
        
        # Simple clustering
        threshold = 2.0
        n = len(endpoints)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            cluster = [i]
            visited[i] = True
            queue = [i]
            
            while queue:
                curr = queue.pop(0)
                for j in range(n):
                    if not visited[j]:
                        dist = np.linalg.norm(endpoints[j] - endpoints[curr])
                        if dist < threshold:
                            cluster.append(j)
                            visited[j] = True
                            queue.append(j)
            
            if len(cluster) >= 5:
                clusters.append(cluster)
        
        for cid, cluster in enumerate(clusters):
            pts = endpoints[cluster]
            center = np.mean(pts, axis=0)
            radius = np.mean(np.linalg.norm(pts - center, axis=1))
            
            self.attractors.append(Attractor(
                position=center,
                basin_radius=radius,
                strength=len(cluster) / n,
                transitions_in=len(cluster),
                mode_id=cid
            ))
        
        if len(self.attractors) > 10:
            self.attractors = self.attractors[-10:]
    
    def get_attractor_force(self, z: np.ndarray) -> np.ndarray:
        force = np.zeros_like(z)
        for attractor in self.attractors:
            force += attractor.attract(z)
        return force


class PhysicsConsistentFlowField:
    """Flow field with physics-consistent dynamics."""
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        lambda_potential: float = 0.7,
        action_strength: float = 0.5
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.lambda_potential = lambda_potential
        self.action_strength = action_strength
        
        self.potential = TruePotentialField(latent_dim)
    
    def add_transition(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray):
        self.potential.add_transition(z, a, z_next)
    
    def compute_F(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        z = np.asarray(z).flatten()
        
        # Conservative part: -grad E
        grad_E = self.potential.compute_gradient_E(z)
        conservative = -self.lambda_potential * grad_E
        
        # Action response
        if a is None:
            a = np.zeros(self.action_dim)
        a = np.asarray(a).flatten()
        
        action_response = self.action_strength * a[:self.latent_dim]
        if len(action_response) < self.latent_dim:
            action_response = np.pad(action_response, (0, self.latent_dim - len(action_response)))
        
        return conservative + action_response
    
    def get_local_physics(self, z: np.ndarray, a: np.ndarray = None) -> Dict:
        z = np.asarray(z).flatten()
        
        E = self.potential.compute_E(z)
        grad_E = self.potential.compute_gradient_E(z)
        F = self.compute_F(z, a)
        attractor_force = self.potential.get_attractor_force(z)
        stability = self.potential.compute_stability(z)
        
        near_attractor = False
        for attractor in self.potential.attractors:
            if np.linalg.norm(z - attractor.position) < attractor.basin_radius:
                near_attractor = True
                break
        
        return {
            'z': z,
            'E': E,
            'grad_E': grad_E,
            'F': F,
            'attractor_force': attractor_force,
            'stability': stability,
            'near_attractor': near_attractor,
            'num_attractors': len(self.potential.attractors)
        }


class ConsistencyLoss:
    """Consistency loss between scalar and vector fields."""
    
    def __init__(self, w_flow: float = 0.4, w_trans: float = 0.4, w_attractor: float = 0.2):
        self.w_flow = w_flow
        self.w_trans = w_trans
        self.w_attractor = w_attractor
    
    def compute(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray, physics: Dict) -> Dict:
        z = np.asarray(z).flatten()
        z_next = np.asarray(z_next).flatten()
        
        # Flow-Potential consistency
        F = physics['F']
        grad_E = physics['grad_E']
        flow_loss = np.sum((F + grad_E) ** 2)
        
        # Transition consistency
        z_pred = z + F
        trans_loss = np.sum((z_next - z_pred) ** 2)
        
        # Attractor consistency
        attract_loss = 0.0
        if physics.get('near_attractor', False):
            af = physics.get('attractor_force', np.zeros_like(z))
            if np.linalg.norm(af) > 1e-6:
                alignment = np.dot(F, af) / (np.linalg.norm(F) * np.linalg.norm(af) + 1e-6)
                attract_loss = max(0, -alignment)
        
        total = self.w_flow * flow_loss + self.w_trans * trans_loss + self.w_attractor * attract_loss
        
        return {
            'total': float(total),
            'flow_potential': float(flow_loss),
            'transition': float(trans_loss),
            'attractor': float(attract_loss)
        }


class LatentPhysicsEngine:
    """
    Latent Physics Engine - energy-consistent flow field.
    
    F(z,a) = -λ∇E(z) + action_response(z,a)
    
    L = ||F + grad E||² + ||z_next - (z + F)||²
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        action_dim: int = 2,
        lambda_potential: float = 0.7,
        action_strength: float = 0.5
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        self.flow_field = PhysicsConsistentFlowField(latent_dim, action_dim, lambda_potential, action_strength)
        self.consistency_loss = ConsistencyLoss()
        self.loss_history: deque = deque(maxlen=100)
        self.step_count = 0
    
    def observe(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray):
        self.flow_field.add_transition(z, a, z_next)
        self.step_count += 1
    
    def get_physics(self, z: np.ndarray, a: np.ndarray = None) -> Dict:
        return self.flow_field.get_local_physics(z, a)
    
    def compute_loss(self, z: np.ndarray, a: np.ndarray, z_next: np.ndarray) -> Dict:
        physics = self.get_physics(z, a)
        loss = self.consistency_loss.compute(z, a, z_next, physics)
        self.loss_history.append(loss['total'])
        return loss
    
    def evolve(self, z: np.ndarray, a: np.ndarray = None, dt: float = 0.1) -> np.ndarray:
        F = self.flow_field.compute_F(z, a)
        z_next = z + F * dt
        
        norm = np.linalg.norm(z_next)
        if norm > 10:
            z_next = z_next * (10 / norm)
        
        return z_next
    
    def simulate_trajectory(self, z_start: np.ndarray, actions: List[np.ndarray], dt: float = 0.1):
        z = z_start.copy()
        trajectory = [z.copy()]
        energies = [self.flow_field.potential.compute_E(z)]
        
        for a in actions:
            z = self.evolve(z, a, dt)
            trajectory.append(z.copy())
            energies.append(self.flow_field.potential.compute_E(z))
        
        return trajectory, energies
    
    def get_state(self) -> Dict:
        return {
            'step_count': self.step_count,
            'num_transitions': len(self.flow_field.potential.transitions),
            'num_attractors': len(self.flow_field.potential.attractors),
            'avg_loss': np.mean(list(self.loss_history)) if self.loss_history else 0,
            'loss_trend': 'decreasing' if len(self.loss_history) >= 10 and 
                          self.loss_history[-1] < self.loss_history[0] else 'stable'
        }