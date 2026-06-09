"""
Phase 18.7 - Stabilized Latent Consistency Layer

Ключевая проблема:
  encoder и transition model не согласованы по целевой функции
  → система "локально согласована", но "глобально не оптимизируема"

Решение:
  1. Latent contract enforcement — фиксированное распределение z
  2. Mode survival policy — жизненный цикл режимов
  3. Unified loss — одна функция потерь
  4. V как potential energy — encoder подстраивается под transition manifold

Архитектура:
  ┌─────────────────────────────────────────────────────────────┐
  │  LATENT CONTRACT                                            │
  │  ├── norm(z) = 1.0 (unit sphere)                           │
  │  ├── drift limit (max change per step)                     │
  │  └── distribution regularization (mean=0, std=1)           │
  ├─────────────────────────────────────────────────────────────┤
  │  MODE POPULATION                                           │
  │  ├── birth: new mode if error > threshold                  │
  │  ├── death: kill if entropy < threshold OR error too high  │
  │  ├── merge: combine similar modes                           │
  │  └── survival buffer: never delete immediately              │
  ├─────────────────────────────────────────────────────────────┤
  │  UNIFIED LOSS                                               │
  │  L = α * reconstruction + β * mode_entropy + γ * stability  │
  ├─────────────────────────────────────────────────────────────┤
  │  ENERGY LANDSCAPE (V как потенциал)                        │
  │  V = -log P(z_next | z, a)                                 │
  │  grad V → direction of least resistance                     │
  └─────────────────────────────────────────────────────────────┘
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class ModeState(Enum):
    """Состояние режима в популяции."""
    ACTIVE = "active"
    DECAYING = "decaying"  # waiting in survival buffer
    MERGING = "merging"
    DEAD = "dead"


@dataclass
class LatentContract:
    """
    Контракт на латентное пространство.
    
    Всё, что касается z, проходит через этот контракт.
    """
    latent_dim: int
    target_norm: float = 1.0
    max_drift: float = 0.5  # max ||z_t - z_{t-1}||
    target_mean: float = 0.0
    target_std: float = 1.0
    
    def enforce(self, z: np.ndarray) -> np.ndarray:
        """Применить контракт к вектору."""
        z = np.asarray(z).flatten()
        
        # Normalize to target norm
        norm = np.linalg.norm(z)
        if norm > 0:
            z = z * (self.target_norm / norm)
        else:
            z = np.zeros(self.latent_dim)
        
        return z
    
    def check_drift(self, z_prev: np.ndarray, z_curr: np.ndarray) -> float:
        """Проверить drift между состояниями."""
        z_prev = np.asarray(z_prev).flatten()
        z_curr = np.asarray(z_curr).flatten()
        drift = np.linalg.norm(z_curr - z_prev)
        return drift
    
    def normalize_distribution(self, z_batch: np.ndarray) -> np.ndarray:
        """Нормализовать батч к целевому распределению."""
        # Center
        mean = np.mean(z_batch, axis=0)
        z_batch = z_batch - mean + self.target_mean
        
        # Scale
        std = np.std(z_batch, axis=0)
        std[std < 1e-6] = 1.0  # avoid division by zero
        z_batch = z_batch * (self.target_std / std)
        
        return z_batch


@dataclass
class ModeIndividual:
    """
    Режим как индивид в популяции.
    
    Полный жизненный цикл: birth → active → decaying → dead
    """
    id: int
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    
    # Lifecycle
    state: ModeState = ModeState.ACTIVE
    age: int = 0
    birth_step: int = 0
    
    # Metrics
    weight: float = 1.0
    reconstruction_error: float = float('inf')
    entropy_score: float = 0.0
    stability: float = 0.0  # 0-1, how stable
    
    # Survival buffer (время до реального удаления)
    decay_steps: int = 0
    max_decay_steps: int = 20
    survival_score: float = 0.0
    
    # Transition history
    history: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)


class ModePopulation:
    """
    Популяция режимов с жизненным циклом.
    
    Режимы не просто "появляются и ломаются" — они живут,
    конкурируют, сливаются, умирают.
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        min_modes: int = 2,
        max_modes: int = 5,
        merge_threshold: float = 0.3,  # similarity threshold for merge
        kill_entropy_threshold: float = 0.2,
        kill_error_threshold: float = 1.5
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        self.min_modes = min_modes
        self.max_modes = max_modes
        self.merge_threshold = merge_threshold
        self.kill_entropy_threshold = kill_entropy_threshold
        self.kill_error_threshold = kill_error_threshold
        
        self.modes: List[ModeIndividual] = []
        self.next_id = 0
        self.step_count = 0
        
        # Initialize with default modes
        self._init_default_modes()
    
    def _init_default_modes(self):
        """Инициализация дефолтных режимов."""
        for _ in range(self.min_modes):
            self._add_mode(
                A=np.random.randn(self.latent_dim, self.latent_dim) * 0.1,
                b=np.random.randn(self.latent_dim, self.action_dim) * 0.2,
                c=np.zeros(self.latent_dim)
            )
    
    def _add_mode(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        from_merger: bool = False
    ) -> ModeIndividual:
        """Добавить новый режим."""
        mode = ModeIndividual(
            id=self.next_id,
            A=A.copy(),
            b=b.copy(),
            c=c.copy() if c is not None else np.zeros(self.latent_dim),
            birth_step=self.step_count
        )
        self.modes.append(mode)
        self.next_id += 1
        return mode
    
    def get_active_modes(self) -> List[ModeIndividual]:
        """Получить активные режимы."""
        return [m for m in self.modes if m.state == ModeState.ACTIVE]
    
    def assign_transition(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray
    ):
        """
        Назначить переход режиму с минимальной ошибкой.
        
        Также обновляет метрики режима.
        """
        z = np.asarray(z).flatten()
        a = np.asarray(a).flatten()
        z_next = np.asarray(z_next).flatten()
        
        # Find best mode
        best_mode = None
        best_error = float('inf')
        
        for mode in self.get_active_modes():
            predicted = mode.A @ z + mode.b @ a + mode.c
            error = np.linalg.norm(z_next - predicted)
            
            if error < best_error:
                best_error = error
                best_mode = mode
        
        if best_mode is None:
            return
        
        # Add to history
        best_mode.history.append((z, a, z_next))
        best_mode.reconstruction_error = best_error
        best_mode.age += 1
        
        # Update weight based on performance
        self._update_mode_weights()
        
        # Check for new mode birth (if no mode fits well)
        if best_error > self.kill_error_threshold and len(self.get_active_modes()) < self.max_modes:
            self._try_birth_mode(z, a, z_next)
    
    def _update_mode_weights(self):
        """Обновить веса режимов на основе ошибок."""
        active = self.get_active_modes()
        if not active:
            return
        
        # Softmax over inverse errors
        errors = np.array([m.reconstruction_error for m in active])
        errors = np.clip(errors, 0.1, 10)  # avoid division by zero
        
        # Convert to weights (lower error = higher weight)
        weights = 1.0 / errors
        weights = weights / weights.sum()
        
        for i, mode in enumerate(active):
            mode.weight = weights[i]
            # Update stability (exponential moving average)
            mode.stability = 0.9 * mode.stability + 0.1 * (1.0 if errors[i] < 0.5 else 0.0)
    
    def _try_birth_mode(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray
    ):
        """Попытка создать новый режим."""
        z = np.asarray(z).flatten()
        a = np.asarray(a).flatten()
        z_next = np.asarray(z_next).flatten()
        
        # Estimate A, b from this single transition
        # For small data, use pseudo-inverse with regularization
        try:
            combined = np.concatenate([z, a])
            # z_next ≈ Theta @ combined
            Theta = np.outer(z_next, combined) / (np.dot(combined, combined) + 1e-6)
            
            A_new = Theta[:, :self.latent_dim] if Theta.shape[1] >= self.latent_dim else np.eye(self.latent_dim) * 0.1
            b_new = Theta[:, self.latent_dim:] if Theta.shape[1] > self.latent_dim else np.zeros((self.latent_dim, self.action_dim))
            
            self._add_mode(A_new, b_new, z_next - A_new @ z - b_new @ a)
        except:
            pass
    
    def update_population(self):
        """
        Обновление популяции: проверка на смерть, слияние.
        
        Вызывается после каждого шага.
        """
        self.step_count += 1
        
        active = self.get_active_modes()
        
        # Check each active mode for death conditions
        for mode in active:
            # Calculate entropy score
            mode.entropy_score = 1.0 - mode.stability
            
            # Death conditions:
            # 1. Very low entropy (too deterministic = collapse)
            # 2. Very high reconstruction error (bad prediction)
            # 3. Has been decaying too long
            
            should_kill = False
            
            if mode.state == ModeState.ACTIVE:
                if mode.entropy_score < self.kill_entropy_threshold:
                    should_kill = True
                if mode.reconstruction_error > self.kill_error_threshold * 2:
                    should_kill = True
                    
            elif mode.state == ModeState.DECAYING:
                mode.decay_steps += 1
                if mode.decay_steps > mode.max_decay_steps:
                    should_kill = True
            
            if should_kill:
                # Don't kill if it would leave us with too few modes
                if len(active) > self.min_modes:
                    mode.state = ModeState.DECAYING
                    mode.decay_steps = 0
        
        # Try to merge similar modes
        self._try_merge()
        
        # Remove dead modes
        self.modes = [m for m in self.modes if m.state != ModeState.DEAD]
    
    def _try_merge(self):
        """Попытка слить похожие режимы."""
        active = self.get_active_modes()
        if len(active) < 2:
            return
        
        # Calculate pairwise similarity (based on A matrices)
        n = len(active)
        similarity = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Compare A matrices
                dA = np.linalg.norm(active[i].A - active[j].A)
                db = np.linalg.norm(active[i].b - active[j].b)
                similarity[i, j] = np.exp(-(dA + db))
                similarity[j, i] = similarity[i, j]
        
        # Find most similar pair above threshold
        max_sim = self.merge_threshold
        merge_pair = None
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarity[i, j] > max_sim:
                    max_sim = similarity[i, j]
                    merge_pair = (i, j)
        
        if merge_pair is not None and len(active) > self.min_modes:
            i, j = merge_pair
            mode_i = active[i]
            mode_j = active[j]
            
            # Merge: weighted average of parameters
            w_i = mode_i.weight / (mode_i.weight + mode_j.weight)
            w_j = 1.0 - w_i
            
            merged_A = w_i * mode_i.A + w_j * mode_j.A
            merged_b = w_i * mode_i.b + w_j * mode_j.b
            merged_c = w_i * mode_i.c + w_j * mode_j.c
            
            # Kill both, create merged
            mode_i.state = ModeState.MERGING
            mode_j.state = ModeState.MERGING
            
            self._add_mode(merged_A, merged_b, merged_c, from_merger=True)
    
    def get_stats(self) -> Dict:
        """Получить статистику популяции."""
        return {
            'total': len(self.modes),
            'active': len(self.get_active_modes()),
            'decaying': len([m for m in self.modes if m.state == ModeState.DECAYING]),
            'avg_error': np.mean([m.reconstruction_error for m in self.get_active_modes()]),
            'avg_stability': np.mean([m.stability for m in self.get_active_modes()]),
            'weights': [m.weight for m in self.modes if m.state == ModeState.ACTIVE]
        }


class UnifiedLoss:
    """
    Единая функция потерь для всей системы.
    
    L = α * reconstruction + β * mode_entropy + γ * stability + δ * latent_regularization
    
    Вместо трёх несогласованных сигналов — один глобальный.
    """
    
    def __init__(
        self,
        w_reconstruction: float = 1.0,
        w_entropy: float = 0.5,
        w_stability: float = 0.3,
        w_latent: float = 0.2
    ):
        self.w_reconstruction = w_reconstruction
        self.w_entropy = w_entropy
        self.w_stability = w_stability
        self.w_latent = w_latent
    
    def compute(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray,
        mode: ModeIndividual,
        latent_contract: LatentContract,
        z_prev: np.ndarray = None
    ) -> Dict:
        """
        Вычислить единую loss и её компоненты.
        
        Returns:
            Dict с 'total', 'reconstruction', 'entropy', 'stability', 'latent'
        """
        # 1. Reconstruction loss
        predicted = mode.A @ z + mode.b @ a + mode.c
        reconstruction = np.sum((z_next - predicted) ** 2)
        
        # 2. Mode entropy (encourage diverse modes)
        active_modes_weight_sum = sum(m.weight for m in [mode] if m.state == ModeState.ACTIVE)
        if active_modes_weight_sum > 0:
            entropy = -np.sum(np.array([mode.weight]) * np.log(np.array([mode.weight]) + 1e-8))
            entropy = entropy / np.log(2)  # normalize to bits
        else:
            entropy = 0.0
        
        # 3. Stability (encourage consistent predictions)
        stability = mode.stability if hasattr(mode, 'stability') else 0.5
        
        # 4. Latent regularization (keep z in target distribution)
        latent_loss = 0.0
        if z_prev is not None:
            drift = latent_contract.check_drift(z_prev, z)
            latent_loss = drift ** 2  # penalize large drifts
        
        # Normalize norm
        norm = np.linalg.norm(z)
        if latent_contract.target_norm > 0:
            norm_loss = (norm - latent_contract.target_norm) ** 2
            latent_loss += norm_loss
        
        # Total loss (lower is better)
        total = (
            self.w_reconstruction * reconstruction +
            self.w_entropy * entropy +
            self.w_stability * (1.0 - stability) +
            self.w_latent * latent_loss
        )
        
        return {
            'total': total,
            'reconstruction': reconstruction,
            'entropy': entropy,
            'stability': stability,
            'latent': latent_loss
        }


class EnergyVField:
    """
    V как потенциал системы (energy landscape).
    
    Вместо V = metric, теперь:
    V = -log P(z_next | z, a)
    
    Это даёт:
    1. V — производная от вероятностной модели
    2. grad V → direction of least resistance
    3. Ландшафт энергии для всей системы
    """
    
    def __init__(
        self,
        population: ModePopulation,
        baseline_energy: float = 1.0
    ):
        self.population = population
        self.baseline_energy = baseline_energy
        self.V_history = deque(maxlen=100)
    
    def compute_energy(
        self,
        z: np.ndarray,
        a: np.ndarray,
        z_next: np.ndarray
    ) -> float:
        """
        Вычислить энергию системы.
        
        V = -log P(z_next | z, a) = reconstruction_error + mode_penalty
        """
        z = np.asarray(z).flatten()
        a = np.asarray(a).flatten()
        z_next = np.asarray(z_next).flatten()
        
        # Find best mode
        best_error = float('inf')
        best_weight = 0.0
        
        for mode in self.population.get_active_modes():
            predicted = mode.A @ z + mode.b @ a + mode.c
            error = np.linalg.norm(z_next - predicted)
            
            if error < best_error:
                best_error = error
                best_weight = mode.weight
        
        # Energy = error + entropy penalty
        entropy_penalty = -np.log(best_weight + 1e-8)
        
        # V = reconstruction + entropy
        V = best_error + 0.1 * entropy_penalty
        
        # Normalize to [0, 1]
        V = np.tanh(V / 5.0)  # compress large values
        V = float(V)
        
        self.V_history.append(V)
        return V
    
    def compute_gradient(
        self,
        z: np.ndarray,
        a: np.ndarray,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """
        Вычислить градиент энергии по z.
        
        grad V ≈ (V(z + ε) - V(z - ε)) / 2ε
        
        Returns:
            direction of decreasing energy
        """
        z = np.asarray(z).flatten()
        a = np.asarray(a).flatten()
        
        grad = np.zeros_like(z)
        
        for i in range(len(z)):
            z_plus = z.copy()
            z_minus = z.copy()
            
            z_plus[i] += epsilon
            z_minus[i] -= epsilon
            
            # We don't have ground truth z_next, so use mode prediction
            # This is an approximation
            best_mode = None
            best_error = float('inf')
            for mode in self.population.get_active_modes():
                pred = mode.A @ z + mode.b @ a + mode.c
                error = np.linalg.norm(pred - z)
                if error < best_error:
                    best_error = error
                    best_mode = mode
            
            if best_mode is None:
                continue
            
            # Approximate gradient from mode
            grad[i] = best_mode.A[i, :] @ z + best_mode.b[i, :] @ a - z[i]
        
        return grad
    
    def get_energy_landscape_stats(self) -> Dict:
        """Статистика энергетического ландшафта."""
        if len(self.V_history) < 2:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        history = list(self.V_history)
        return {
            'mean': np.mean(history),
            'std': np.std(history),
            'min': np.min(history),
            'max': np.max(history),
            'trend': 'increasing' if history[-1] > history[0] else 'decreasing'
        }


class StabilizedWorldModel:
    """
    Полная стабилизированная система.
    
    Объединяет:
    - Latent contract
    - Mode population
    - Unified loss
    - Energy V-field
    
    Usage:
        model = StabilizedWorldModel(latent_dim=8, action_dim=2)
        
        # Training step
        loss = model.step(z, a, z_next, z_prev)
        
        # Get control signal
        control = model.get_control_signal(z, a)
        
        # Get V (energy)
        V = model.compute_energy(z, a, z_next)
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        action_dim: int = 2,
        num_modes: int = 3
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Core components
        self.latent_contract = LatentContract(latent_dim)
        self.population = ModePopulation(
            latent_dim=latent_dim,
            action_dim=action_dim,
            min_modes=2,
            max_modes=num_modes
        )
        self.loss_fn = UnifiedLoss()
        self.energy_field = EnergyVField(self.population)
        
        # State
        self.z_prev: Optional[np.ndarray] = None
        self.step_count = 0
        
        # Statistics
        self.loss_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)
    
    def step(
        self,
        obs: np.ndarray,
        a: np.ndarray,
        z_next_true: np.ndarray = None,
        encoder: Callable = None
    ) -> Dict:
        """
        Один шаг обучения/обновления.
        
        Args:
            obs: observation (raw)
            a: action
            z_next_true: ground truth next latent (if available)
            encoder: function to encode obs → z
        
        Returns:
            Dict с loss components и metrics
        """
        self.step_count += 1
        
        # Encode observation
        if encoder is not None:
            z = encoder(obs)
        else:
            # Default: identity
            z = np.asarray(obs).flatten()
            if len(z) != self.latent_dim:
                z = np.zeros(self.latent_dim)
        
        # Enforce latent contract
        z = self.latent_contract.enforce(z)
        
        # Get predicted z_next from best mode
        best_mode = None
        if self.population.get_active_modes():
            best_error = float('inf')
            for mode in self.population.get_active_modes():
                predicted = mode.A @ z + mode.b @ a + mode.c
                if z_next_true is not None:
                    error = np.linalg.norm(z_next_true - predicted)
                    if error < best_error:
                        best_error = error
                        best_mode = mode
        
        # Use true z_next if available, else predicted
        if z_next_true is not None:
            z_next = self.latent_contract.enforce(z_next_true)
        elif best_mode is not None:
            z_next = best_mode.A @ z + best_mode.b @ a + best_mode.c
            z_next = self.latent_contract.enforce(z_next)
        else:
            z_next = z.copy()
        
        # Assign transition to population
        if z_next_true is not None:
            self.population.assign_transition(z, a, z_next_true)
        
        # Update population (birth/death/merge)
        self.population.update_population()
        
        # Compute unified loss
        if best_mode is not None:
            loss = self.loss_fn.compute(
                z, a, z_next if z_next_true is None else z_next_true,
                best_mode,
                self.latent_contract,
                self.z_prev
            )
        else:
            loss = {'total': 0, 'reconstruction': 0, 'entropy': 0, 'stability': 0, 'latent': 0}
        
        # Update energy field
        energy = self.energy_field.compute_energy(z, a, z_next)
        
        # Store history
        self.loss_history.append(loss['total'])
        self.energy_history.append(energy)
        
        # Update state
        self.z_prev = z.copy()
        
        return {
            'loss': loss,
            'energy': energy,
            'population_stats': self.population.get_stats(),
            'z_norm': float(np.linalg.norm(z)),
            'drift': float(np.linalg.norm(z - self.z_prev)) if self.z_prev is not None else 0
        }
    
    def get_control_signal(self, z: np.ndarray, a: np.ndarray) -> Dict:
        """
        Получить управляющий сигнал для policy.
        
        Returns:
            Dict с V, is_safe, threshold, mode_stats
        """
        energy = 0.0
        
        if self.population.get_active_modes():
            # Use energy as control V
            z_enforced = self.latent_contract.enforce(z)
            a = np.asarray(a).flatten()
            
            # Get best mode prediction
            best_error = float('inf')
            best_weight = 1.0
            
            for mode in self.population.get_active_modes():
                predicted = mode.A @ z_enforced + mode.b @ a + mode.c
                error = np.linalg.norm(predicted - z_enforced)
                if error < best_error:
                    best_error = error
                    best_weight = mode.weight
            
            energy = np.tanh(best_error / 5.0)
        
        # Adaptive threshold based on energy history
        threshold = 0.3
        if len(self.energy_history) >= 20:
            recent = list(self.energy_history)[-20:]
            threshold = np.mean(recent) + np.std(recent)
        
        is_safe = energy < threshold
        
        return {
            'V': 1.0 - energy,  # invert: low energy = high safety
            'energy': energy,
            'is_safe': is_safe,
            'threshold': threshold,
            'mode_stats': self.population.get_stats()
        }
    
    def get_diagnostic_state(self) -> Dict:
        """Получить полное диагностическое состояние."""
        energy_stats = self.energy_field.get_energy_landscape_stats()
        pop_stats = self.population.get_stats()
        
        # Compute trajectory metrics from population
        active = self.population.get_active_modes()
        
        if active:
            weights = np.array([m.weight for m in active])
            weights_norm = weights / weights.sum()
            diversity = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
            max_entropy = np.log(len(weights) + 1e-8)
            trajectory_diversity = diversity / max_entropy if max_entropy > 0 else 0
        else:
            trajectory_diversity = 0.0
        
        return {
            'V': 1.0 - energy_stats.get('mean', 0),
            'trajectory_diversity': trajectory_diversity,
            'trajectory_divergence': energy_stats.get('std', 0),
            'trajectory_entropy': 1.0 - trajectory_diversity,
            'trajectory_collapse': 1.0 - trajectory_diversity,
            'status': 'healthy' if energy_stats.get('mean', 0) < 0.5 else 'warning',
            'energy_stats': energy_stats,
            'population_stats': pop_stats,
            'step_count': self.step_count
        }