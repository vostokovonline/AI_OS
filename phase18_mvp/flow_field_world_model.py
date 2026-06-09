"""
Phase 18.8 - Flow Field World Model (Geometric Consistency Edition)

Ключевой сдвиг:
  OLD: Dynamic consistency (stabilized ensemble)
  NEW: Geometric consistency (energy-consistent manifold)

Главные изменения:
1. Energy-consistent encoder: z = argmin_z V(z, a) via gradient descent
2. Soft mode mixture: P(mode | z, a) instead of hard assignment
3. True continuous V-field: V(z, a) as scalar field, not samples
4. Gradient flow dynamics: z_next = z - ∇V(z) + noise

Архитектура:
  ┌─────────────────────────────────────────────────────────────┐
  │  PARTICLE IN ENERGY FIELD                                   │
  │                                                              │
  │  z = particle with position in latent manifold             │
  │  V(z) = potential energy at position                       │
  │  grad V = force acting on particle                          │
  │                                                              │
  │  z_next = z - α∇V(z,a) + σ·noise  (gradient flow)          │
  │                                                              │
  │  Particle dynamics = flow field = physics                   │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │  VECTOR BASIS MODES                                         │
  │                                                              │
  │  Не "режимы как отдельные модели"                          │
  │  А "базисные направления в пространстве"                  │
  │                                                              │
  │  f(z,a) = Σ w_i(z) · mode_i(z,a)                           │
  │  где w_i(z) = P(mode_i | z) — soft responsibility          │
  └─────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────┐
  │  TRUE SCALAR FIELD V(z,a)                                   │
  │                                                              │
  │  V(z,a) = -log ∫ P(z_next | z,a) dz_next                  │
  │                                                              │
  │  Это не семплирование, это аналитическое поле             │
  │  ∇V = E[grad_z log P(z_next | z,a)]                       │
  └─────────────────────────────────────────────────────────────┘
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from collections import deque
import copy


@dataclass
class ModeVector:
    """
    Mode как векторный базис в пространстве.
    
    Вместо "отдельная модель перехода" — это:
    - направление в латентном пространстве
    - компонента общего flow field
    - с softmax weight по всему пространству
    """
    id: int
    direction: np.ndarray  # eigen-direction in latent space
    magnitude: float  # strength of this mode
    phase: np.ndarray  # где в пространстве этот mode доминирует
    
    # For soft mixture
    responsibility: float = 0.0
    
    def update(self, z: np.ndarray, residual: float):
        """Обновить mode на основе нового наблюдения."""
        # Exponential moving average
        self.direction = 0.9 * self.direction + 0.1 * residual * z / (np.linalg.norm(z) + 1e-6)
        self.magnitude = 0.9 * self.magnitude + 0.1 * abs(residual)


@dataclass
class EnergyState:
    """
    Состояние в энергетическом поле.
    
    z — позиция частицы
    V — потенциальная энергия
    grad_V — градиент (сила)
    uncertainty — энтропия распределения
    """
    z: np.ndarray
    V: float
    grad_V: np.ndarray
    uncertainty: float
    local_density: float
    
    def flow_direction(self, damping: float = 0.1) -> np.ndarray:
        """Направление градиентного потока."""
        return -self.grad_V * damping


class EnergyConsistentEncoder:
    """
    Encoder который выравнивается с energy landscape.
    
    Вместо: z = encoder(obs)
    
    Нужно: z = argmin_z V(z, a) — но мы аппроксимируем через gradient descent
    
    z_t+1 = z_t - λ * ∇V(z_t, a)
    
    Это amortized inference для energy minimization.
    """
    
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        energy_field,  # will be passed
        lambda_encoder: float = 0.1,  # learning rate for encoder
        encoder_lr: float = 0.01  # slow adaptation
    ):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.energy_field = energy_field
        self.lambda_encoder = lambda_encoder
        self.encoder_lr = encoder_lr
        
        # Base encoder parameters
        self.W = np.random.randn(latent_dim, obs_dim) * 0.1
        self.b = np.zeros(latent_dim)
        
        # Encoder momentum (for gradient descent in latent space)
        self.z_momentum = np.zeros(latent_dim)
        
        # Statistics
        self.alignment_loss_history = deque(maxlen=100)
    
    def encode(self, obs: np.ndarray, a: np.ndarray = None, n_iterations: int = 5) -> np.ndarray:
        """
        Encode с energy-consistent inference.
        
        z = encode_base(obs) → project → minimize_energy(z)
        
        Args:
            obs: observation
            a: action context (optional)
            n_iterations: number of gradient descent steps
        
        Returns:
            Energy-consistent latent state
        """
        obs = np.asarray(obs).flatten()
        
        # Base encoding
        z = self.W @ obs + self.b
        
        # Gradient descent to minimize energy
        for _ in range(n_iterations):
            # Compute gradient of V with respect to z
            grad_V = self.energy_field.estimate_gradient_V(z, a)
            
            # Update with momentum
            self.z_momentum = 0.9 * self.z_momentum + 0.1 * grad_V
            z = z - self.lambda_encoder * self.z_momentum
        
        # Normalize
        z = z / (np.linalg.norm(z) + 1e-6)
        
        return z
    
    def update_alignment(self, z: np.ndarray, V_z: float):
        """
        Обновить encoder параметры для лучшего alignment с energy landscape.
        
       grad_W = -∂V/∂z * ∂z/∂W = -grad_V * obs^T
        """
        # Alignment loss = V(z) — we want to minimize it
        grad_V = self.energy_field.estimate_gradient_V(z, None)
        
        # Gradient for W: dV/dW = dV/dz * dz/dW = -grad_V * obs^T
        obs_placeholder = np.zeros(self.obs_dim)  # Need actual obs
        # This is simplified - in full version, track obs history
        
        # Update W to move toward lower energy regions
        # W ← W - lr * outer(grad_V, obs) — not implemented without obs
    
    def get_state(self) -> Dict:
        """Получить состояние encoder."""
        return {
            'W_norm': float(np.linalg.norm(self.W)),
            'momentum_norm': float(np.linalg.norm(self.z_momentum)),
            'alignment_score': 1.0 - np.mean(list(self.alignment_loss_history)) if self.alignment_loss_history else 0.5
        }


class SoftModeMixture:
    """
    Мягкая смесь режимов вместо hard assignment.
    
    Вместо: mode = argmin error
    
    Теперь: P(mode | z, a) = softmax(-error_i / temperature)
    
    И: z_next = Σ P(mode) * f_mode(z, a)
    
    Это EM-like soft clustering.
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_modes: int = 3,
        temperature: float = 1.0  # softmax temperature
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        self.temperature = temperature
        
        # Mode vectors (eigen-directions)
        self.modes: List[ModeVector] = []
        self._init_modes()
        
        # Responsibility history for EM
        self.responsibility_history: deque = deque(maxlen=1000)
    
    def _init_modes(self):
        """Инициализация модовых направлений."""
        for i in range(self.num_modes):
            mode = ModeVector(
                id=i,
                direction=np.random.randn(self.latent_dim) * 0.1,
                magnitude=1.0 / self.num_modes,
                phase=np.zeros(self.latent_dim)
            )
            self.modes.append(mode)
    
    def compute_responsibilities(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray
    ) -> np.ndarray:
        """
        Вычислить soft responsibilities P(mode | z, a, z_next).
        
        Returns:
            responsibilities: (num_modes,) array of probabilities
        """
        z = np.asarray(z).flatten()
        a = np.asarray(a).flatten()
        z_next = np.asarray(z_next).flatten()
        
        errors = []
        for mode in self.modes:
            # Predict z_next from mode
            predicted = mode.direction * np.dot(z, mode.direction) + \
                       mode.magnitude * np.random.randn(self.latent_dim) * 0.1
            # Simplified: just use distance to mode center
            error = np.linalg.norm(z_next - z - mode.direction)
            errors.append(error)
        
        errors = np.array(errors)
        
        # Softmax for responsibilities
        # P(mode) = exp(-error / T) / sum(exp(-error_j / T))
        neg_errors = -errors / self.temperature
        neg_errors -= np.max(neg_errors)  # numerically stable
        responsibilities = np.exp(neg_errors)
        responsibilities = responsibilities / (responsibilities.sum() + 1e-8)
        
        return responsibilities
    
    def update(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray,
        responsibilities: np.ndarray
    ):
        """
        Обновить modes на основе responsibilities.
        
        Используем EM-like update:
        - M-step: update mode parameters to minimize weighted error
        - Mode becomes weighted combination of assigned transitions
        """
        z = np.asarray(z).flatten()
        z_next = np.asarray(z_next).flatten()
        
        for i, mode in enumerate(self.modes):
            resp = responsibilities[i]
            
            # Residual (how much this transition deviates from mode prediction)
            predicted = z + mode.direction  # simple linear prediction
            residual = z_next - predicted
            
            # Update mode direction (weighted average of residuals)
            if np.linalg.norm(residual) > 1e-6:
                update = resp * residual
                mode.direction = 0.9 * mode.direction + 0.1 * update
                mode.direction = mode.direction / (np.linalg.norm(mode.direction) + 1e-6)
            
            # Update magnitude
            mode.magnitude = 0.9 * mode.magnitude + 0.1 * resp * np.linalg.norm(residual)
            
            # Update phase (center of responsibility)
            mode.phase = 0.9 * mode.phase + 0.1 * resp * z
            
            # Update responsibility
            mode.responsibility = resp
        
        # Store history
        self.responsibility_history.append(responsibilities)
    
    def predict(
        self,
        z: np.ndarray,
        a: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказать z_next с мягкой смесью.
        
        Returns:
            z_next_pred: soft mixture prediction
            responsibilities: P(mode | z, a)
        """
        z = np.asarray(z).flatten()
        
        # Get responsibilities for each mode
        responsibilities = np.array([m.responsibility for m in self.modes])
        responsibilities = responsibilities / (responsibilities.sum() + 1e-8)
        
        # Predict from each mode and combine
        predictions = []
        for mode in self.modes:
            # Mode prediction: z + direction + noise scaled by magnitude
            noise = np.random.randn(self.latent_dim) * mode.magnitude * 0.1
            pred = z + mode.direction + noise
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Weighted average
        z_next_pred = np.sum(predictions * responsibilities[:, None], axis=0)
        
        return z_next_pred, responsibilities
    
    def get_mode_stats(self) -> Dict:
        """Статистика режимов."""
        return {
            'num_modes': len(self.modes),
            'responsibilities': [m.responsibility for m in self.modes],
            'magnitudes': [m.magnitude for m in self.modes],
            'directions_norm': [float(np.linalg.norm(m.direction)) for m in self.modes],
            'total_entropy': -np.sum(np.array([m.responsibility for m in self.modes]) * 
                                     np.log(np.array([m.responsibility + 1e-8 for m in self.modes])))
        }


class ContinuousVField:
    """
    Истинное непрерывное поле V(z,a).
    
    Вместо семплирования:
    V(z,a) = -log ∫ P(z_next | z,a) dz_next
    
    Используем kernel density estimation для непрерывного поля.
    
    Вместо: семплируем N переходов → считаем V
    
    Используем: аналитическое поле через RBF ядро
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        kernel_width: float = 1.0,
        min_points: int = 20
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.kernel_width = kernel_width
        self.min_points = min_points
        
        # Buffer of transitions
        self.transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        
        # Cached kernel matrix (for efficiency)
        self.kernel_cache = None
        self.cache_valid = False
    
    def add_transition(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray
    ):
        """Добавить переход в буфер."""
        self.transitions.append((
            np.asarray(z).flatten(),
            np.asarray(a).flatten(),
            np.asarray(z_next).flatten()
        ))
        self.cache_valid = False
        
        # Limit buffer size
        if len(self.transitions) > 500:
            self.transitions.pop(0)
    
    def _compute_kernel_matrix(self) -> np.ndarray:
        """
        Вычислить матрицу RBF ядер для всех точек.
        
        K[i,j] = exp(-||z_i - z_j||^2 / (2 * sigma^2))
        """
        if len(self.transitions) < self.min_points:
            return np.zeros((0, 0))
        
        Z = np.array([t[0] for t in self.transitions])  # current states
        Z_next = np.array([t[2] for t in self.transitions])  # next states
        
        # Compute pairwise distances for current states
        Z_sq = np.sum(Z ** 2, axis=1, keepdims=True)
        dist_matrix = Z_sq + Z_sq.T - 2 * Z @ Z.T
        dist_matrix = np.clip(dist_matrix, 0, None)
        
        # RBF kernel
        K = np.exp(-dist_matrix / (2 * self.kernel_width ** 2))
        
        self.kernel_cache = K
        self.cache_valid = True
        
        return K
    
    def compute_V(self, z: np.ndarray, a: np.ndarray = None) -> float:
        """
        Вычислить V(z,a) аналитически.
        
        V(z,a) = -log ∫ P(z_next | z,a) dz_next
        
        Аппроксимация через kernel density estimation:
        P(z_next | z,a) ≈ Σ w_i * K(z_next, z_i)
        
        где w_i = P(a | z_i, z_next_i) / Σ w_j
        
        Returns:
            V: potential energy at z
        """
        z = np.asarray(z).flatten()
        
        if len(self.transitions) < self.min_points:
            # Not enough points — return default
            return 0.5
        
        if not self.cache_valid:
            self._compute_kernel_matrix()
        
        Z = np.array([t[0] for t in self.transitions])
        
        # Compute similarity to all transition origins
        dists = np.linalg.norm(Z - z, axis=1)
        
        # Kernel weights
        weights = np.exp(-dists ** 2 / (2 * self.kernel_width ** 2))
        weights = weights / (weights.sum() + 1e-8)
        
        # V = -log weighted average of kernel values
        # Higher density of nearby transitions = lower V (more explored = safer)
        V = -np.log(weights.mean() + 1e-8)
        
        # Normalize to [0, 1]
        V = np.tanh(V / 5.0)
        
        return float(V)
    
    def compute_gradient_V(self, z: np.ndarray, a: np.ndarray = None, epsilon: float = 0.01) -> np.ndarray:
        """
        Вычислить градиент V(z,a) по z.
        
        ∇V(z) = ∂/∂z [-log P(z_next | z,a)]
        
        Returns:
            grad_V: gradient vector in latent space
        """
        z = np.asarray(z).flatten()
        dim = len(z)
        
        if len(self.transitions) < self.min_points:
            return np.zeros(dim)
        
        Z = np.array([t[0] for t in self.transitions])
        
        grad = np.zeros(dim)
        
        for i in range(dim):
            z_plus = z.copy()
            z_minus = z.copy()
            
            z_plus[i] += epsilon
            z_minus[i] -= epsilon
            
            V_plus = self.compute_V(z_plus, a)
            V_minus = self.compute_V(z_minus, a)
            
            grad[i] = (V_plus - V_minus) / (2 * epsilon)
        
        return grad
    
    def get_flow(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        """
        Получить flow vector F(z,a) = -∇V(z,a).
        
        Это направление в котором "течёт" система.
        """
        grad_V = self.compute_gradient_V(z, a)
        flow = -grad_V
        
        return flow
    
    def get_local_field(self, z: np.ndarray, a: np.ndarray = None) -> Dict:
        """Получить полное локальное поле."""
        V = self.compute_V(z, a)
        grad_V = self.compute_gradient_V(z, a)
        flow = self.get_flow(z, a)
        
        # Local density estimate
        Z = np.array([t[0] for t in self.transitions])
        dists = np.linalg.norm(Z - z, axis=1)
        local_density = 1.0 / (np.mean(dists) + 1e-6)
        local_density = np.tanh(local_density / 10)
        
        # Uncertainty = variance of V estimates from nearby points
        V_nearby = [self.compute_V(Z[i], a) for i in range(len(Z)) if dists[i] < 2.0]
        uncertainty = np.std(V_nearby) if V_nearby else 0.5
        
        return {
            'z': z,
            'V': V,
            'grad_V': grad_V,
            'flow': flow,
            'local_density': float(local_density),
            'uncertainty': float(uncertainty),
            'num_transitions': len(self.transitions)
        }


class FlowFieldWorldModel:
    """
    Flow Field World Model — геометрически согласованная система.
    
    Это финальная версия, где:
    1. Encoder выравнивается с energy landscape
    2. Modes — мягкая смесь, не hard assignment
    3. V(z,a) — истинное непрерывное поле
    4. Dynamics = gradient flow, не семплирование
    
    Architecture:
    
    obs → [Energy-Consistent Encoder] → z
                                    ↓
                              [V(z,a) field]
                                    ↓
                        [Soft Mode Mixture] ← responsibilities
                                    ↓
                         z_next = Σ w_i · mode_i(z,a)
                                    ↓
                         grad flow: z_next = z + α·flow
                                    ↓
                              [Transition Buffer]
    
    Usage:
        model = FlowFieldWorldModel()
        
        # Training step
        model.step(obs, a, obs_next)
        
        # Get state
        state = model.get_state()
        # z, V, flow, responsibilities, mode stats
        
        # Predict
        z_next, resp = model.predict(obs, a)
    """
    
    def __init__(
        self,
        obs_dim: int = 10,
        latent_dim: int = 8,
        action_dim: int = 2,
        num_modes: int = 3
    ):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        
        # Core components
        self.V_field = ContinuousVField(latent_dim, action_dim)
        self.mode_mixture = SoftModeMixture(latent_dim, action_dim, num_modes)
        
        # Encoder (energy-consistent)
        self.encoder = EnergyConsistentEncoder(
            obs_dim, latent_dim,
            energy_field=self.V_field,
            lambda_encoder=0.1
        )
        
        # Transition buffer (for V field)
        self.z_history: deque = deque(maxlen=500)
        
        # State
        self.z_current: Optional[np.ndarray] = None
        self.step_count = 0
    
    def step(
        self,
        obs: np.ndarray,
        a: np.ndarray,
        obs_next: np.ndarray = None,
        encode_iterations: int = 5
    ) -> Dict:
        """
        Один шаг обучения/обновления.
        
        Args:
            obs: current observation
            a: action taken
            obs_next: next observation (for training)
            encode_iterations: number of gradient descent steps
        
        Returns:
            Dict с обновлённым состоянием
        """
        self.step_count += 1
        
        # Encode with energy-consistent inference
        z = self.encoder.encode(obs, a, n_iterations=encode_iterations)
        
        # Store in history
        self.z_history.append(z.copy())
        
        if obs_next is not None:
            # Get next latent state
            z_next = self.encoder.encode(obs_next, a, n_iterations=encode_iterations)
            
            # Add to V field
            self.V_field.add_transition(z, a, z_next)
            
            # Get responsibilities
            responsibilities = self.mode_mixture.compute_responsibilities(z, a, z_next)
            
            # Update modes
            self.mode_mixture.update(z, a, z_next, responsibilities)
            
            # Update encoder alignment
            V_z = self.V_field.compute_V(z, a)
            self.encoder.alignment_loss_history.append(V_z)
        
        # Store current state
        self.z_current = z.copy()
        
        # Get full state
        return self.get_state()
    
    def predict(
        self,
        obs: np.ndarray,
        a: np.ndarray,
        encode_iterations: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Предсказать следующее состояние.
        
        Returns:
            z_next: predicted latent state
            responsibilities: P(mode | z, a)
        """
        z = self.encoder.encode(obs, a, n_iterations=encode_iterations)
        
        # Get soft mixture prediction
        z_next, responsibilities = self.mode_mixture.predict(z, a)
        
        return z_next, responsibilities
    
    def get_state(self) -> Dict:
        """Получить полное состояние системы."""
        if self.z_current is None:
            z = np.zeros(self.latent_dim)
        else:
            z = self.z_current
        
        # Local field
        local_field = self.V_field.get_local_field(z, None)
        
        # Mode stats
        mode_stats = self.mode_mixture.get_mode_stats()
        
        # Encoder stats
        encoder_stats = self.encoder.get_state()
        
        # V field stats
        V = self.V_field.compute_V(z, None)
        
        return {
            'z': z,
            'V': V,
            'flow': local_field['flow'],
            'local_density': local_field['local_density'],
            'uncertainty': local_field['uncertainty'],
            'mode_stats': mode_stats,
            'encoder_stats': encoder_stats,
            'step_count': self.step_count,
            'z_history_size': len(self.z_history)
        }
    
    def get_field_data(self, z: np.ndarray = None) -> Dict:
        """Получить данные для визуализации flow field."""
        if z is None:
            z = self.z_current if self.z_current is not None else np.zeros(self.latent_dim)
        
        # Get full local field
        local_field = self.V_field.get_local_field(z, None)
        
        # Get 2D projection data
        Z_hist = np.array(list(self.z_history))
        
        if len(Z_hist) < 3:
            projection_2d = np.array([[0, 0]])
            labels = np.array([0])
        else:
            # PCA projection
            centered = Z_hist - np.mean(Z_hist, axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            if len(S) >= 2:
                projection_2d = centered @ Vt[:2].T
            else:
                projection_2d = np.zeros((len(Z_hist), 2))
            
            # Labels = V values
            labels = np.array([self.V_field.compute_V(z_i, None) for z_i in Z_hist])
        
        # Get flow vectors for visualization
        num_samples = 20
        flow_vectors = []
        positions = []
        
        if len(Z_hist) > 0:
            center = np.mean(Z_hist, axis=0)
            
            for _ in range(num_samples):
                # Random point in local neighborhood
                point = center + np.random.randn(self.latent_dim) * 0.5
                
                flow = self.V_field.get_flow(point, None)
                
                positions.append(point)
                flow_vectors.append(flow)
        
        return {
            'z_current': z,
            'V': local_field['V'],
            'flow': local_field['flow'],
            'grad_V': local_field['grad_V'],
            'local_density': local_field['local_density'],
            'uncertainty': local_field['uncertainty'],
            'projection_2d': projection_2d,
            'labels': labels,
            'flow_vectors': np.array(flow_vectors) if flow_vectors else np.zeros((0, self.latent_dim)),
            'positions': np.array(positions) if positions else np.zeros((0, self.latent_dim)),
            'mode_responsibilities': [m.responsibility for m in self.mode_mixture.modes],
            'num_transitions': len(self.V_field.transitions)
        }