"""
Phase 18.6 - Transition-First World Model Kernel

Это НЕ улучшение B-lite. Это новая архитектура.

Ключевые изменения:
1. Жёсткие контракты на размерности (никаких padding внутри ядра)
2. Режимы валидируются перед использованием
3. V разделен на диагностический (наблюдение) и управляющий (policy signal)

Архитектура:
  ┌─────────────────────────────────────────────────────┐
  │           TRANSITION KERNEL (single source of truth)│
  │  ┌─────────────────────────────────────────────────┐ │
  │  │ dim_contract: (z_dim, a_dim) → strict shapes    │ │
  │  │ modes: List[TransitionMode] с валидацией        │ │
  │  │ mode_quality: reconstruction_error, stability   │ │
  │  └─────────────────────────────────────────────────┘ │
  │                         ↓                              │
  │  ┌─────────────────────────────────────────────────┐ │
  │  │ DIAGNOSTIC V (read-only observation)             │ │
  │  │ - trajectory_diversity                          │ │
  │  │ - trajectory_divergence                         │ │
  │  │ - mode_entropy                                  │ │
  │  │ - collapse_indicator                            │ │
  │  │ - NO influence on policy                        │ │
  │  └─────────────────────────────────────────────────┘ │
  │                         ↓                              │
  │  ┌─────────────────────────────────────────────────┐ │
  │  │ CONTROL V (policy signal)                        │ │
  │  │ - computed from Diagnostic V                     │ │
  │  │ - smoothed/filtered                             │ │
  │  │ - used ONLY for action selection                 │ │
  │  └─────────────────────────────────────────────────┘ │
  └─────────────────────────────────────────────────────┘
"""
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class DimContract:
    """Жёсткий контракт на размерности."""
    latent_dim: int
    action_dim: int
    
    def validate_z(self, z) -> np.ndarray:
        z = np.asarray(z).flatten()
        assert len(z) == self.latent_dim, f"z dim mismatch: {len(z)} != {self.latent_dim}"
        return z
    
    def validate_a(self, a) -> np.ndarray:
        a = np.asarray(a).flatten()
        assert len(a) == self.action_dim, f"a dim mismatch: {len(a)} != {self.action_dim}"
        return a
    
    def validate_z_next(self, z_next) -> np.ndarray:
        z_next = np.asarray(z_next).flatten()
        assert len(z_next) == self.latent_dim, f"z_next dim mismatch: {len(z_next)} != {self.latent_dim}"
        return z_next


@dataclass
class ModeQuality:
    """Качество режима - метрики валидности."""
    reconstruction_error: float  # MSE на последних переходах
    weight_stability: float      # Изменение веса между итерациями
    transition_count: int        # Сколько переходов назначено
    is_valid: bool               # Прошёл ли все проверки


@dataclass 
class TransitionMode:
    """Режим перехода с валидацией."""
    A: np.ndarray
    b: np.ndarray
    c: np.ndarray
    weight: float
    previous_weight: float = 0.0
    transition_history: List = None
    quality: Optional[ModeQuality] = None
    
    def __post_init__(self):
        if self.transition_history is None:
            self.transition_history = []
        # Ensure shapes are correct
        assert len(self.A.shape) == 2, f"A should be 2D, got {self.A.shape}"
        assert len(self.b.shape) == 2, f"b should be 2D, got {self.b.shape}"
        assert len(self.c.shape) == 1, f"c should be 1D, got {self.c.shape}"


class TransitionKernel:
    """
    Ядро world model - transition-first архитектура.
    
    Отличается от B-lite:
    1. Жёсткие dim contracts
    2. Валидация режимов перед использованием
    3. Отдельный tracking качества режимов
    """
    
    def __init__(self, latent_dim: int, action_dim: int, num_modes: int = 3):
        self.contract = DimContract(latent_dim, action_dim)
        self.num_modes = num_modes
        self.modes: List[TransitionMode] = []
        self.transitions: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._init_modes()
    
    def _init_modes(self):
        """Инициализация режимов с правильными shapes."""
        for i in range(self.num_modes):
            mode = TransitionMode(
                A=np.random.randn(self.contract.latent_dim, self.contract.latent_dim) * 0.1,
                b=np.random.randn(self.contract.latent_dim, self.contract.action_dim) * 0.2,
                c=np.zeros(self.contract.latent_dim),
                weight=1.0 / self.num_modes
            )
            self.modes.append(mode)
    
    def add_transition(self, z, a, z_next):
        """
        Добавить переход с валидацией контракта.
        
        Raises:
            AssertionError если размерности не совпадают
        """
        z = self.contract.validate_z(z)
        a = self.contract.validate_a(a)
        z_next = self.contract.validate_z_next(z_next)
        
        self.transitions.append((z, a, z_next))
        
        # Назначаем режим с минимальной ошибкой
        best_mode_idx = 0
        best_error = float('inf')
        
        for i, mode in enumerate(self.modes):
            predicted = mode.A @ z + mode.b @ a + mode.c
            error = np.linalg.norm(z_next - predicted)
            if error < best_error:
                best_error = error
                best_mode_idx = i
        
        # Добавляем в историю режима
        self.modes[best_mode_idx].transition_history.append((z, a, z_next))
        
        # Обновляем качество режима
        self._update_mode_quality(best_mode_idx)
        
        # Учим режимы если достаточно данных
        if len(self.transitions) >= 50 and len(self.transitions) % 20 == 0:
            self._fit_modes()
    
    def _update_mode_quality(self, mode_idx: int):
        """Обновить метрики качества режима."""
        mode = self.modes[mode_idx]
        history = mode.transition_history[-50:]  # Последние 50 переходов
        
        if len(history) < 5:
            mode.quality = ModeQuality(
                reconstruction_error=1.0,
                weight_stability=0.0,
                transition_count=len(history),
                is_valid=False
            )
            return
        
        # Reconstruction error
        errors = []
        for z, a, z_next in history:
            predicted = mode.A @ z + mode.b @ a + mode.c
            errors.append(np.linalg.norm(z_next - predicted))
        recon_error = np.mean(errors)
        
        # Weight stability
        weight_change = abs(mode.weight - mode.previous_weight)
        mode.previous_weight = mode.weight
        weight_stability = 1.0 - np.tanh(weight_change * 10)
        
        # Mode valid if:
        # 1. Reconstruction error < threshold (mode predicts well)
        # 2. Has enough transitions
        # 3. Weight stability > threshold (mode is stable)
        is_valid = (
            recon_error < 1.0 and  # Not too high error
            len(history) >= 10 and   # Enough data
            weight_stability > 0.5   # Stable weight
        )
        
        mode.quality = ModeQuality(
            reconstruction_error=recon_error,
            weight_stability=weight_stability,
            transition_count=len(history),
            is_valid=is_valid
        )
    
    def _fit_modes(self):
        """Переобучить режимы с фильтрацией плохих."""
        if len(self.transitions) < 50:
            return
        
        # Собираем переходы по режимам
        mode_assignments = [[] for _ in range(self.num_modes)]
        
        for z, a, z_next in self.transitions[-200:]:
            residuals = []
            for i, mode in enumerate(self.modes):
                # Пропускаем невалидные режимы при назначении
                if mode.quality and not mode.quality.is_valid:
                    continue
                    
                predicted = mode.A @ z + mode.b @ a + mode.c
                residual = np.linalg.norm(z_next - predicted)
                residuals.append((i, residual))
            
            if not residuals:
                continue
                
            nearest = min(residuals, key=lambda x: x[1])[0]
            mode_assignments[nearest].append((z, a, z_next))
        
        # Обновляем режимы и их веса
        total_assignments = sum(len(m) for m in mode_assignments)
        
        for i, mode in enumerate(self.modes):
            if len(mode_assignments[i]) < 5:
                # Недостаточно данных для обучения - помечаем невалидным
                if mode.quality:
                    mode.quality.is_valid = False
                continue
            
            # Собираем данные для регрессии
            # Build X = [z, a] with proper dimensions
            z_dim = self.contract.latent_dim
            a_dim = self.contract.action_dim
            total_dim = z_dim + a_dim
            
            # Stack transitions
            X_list = []
            Y_list = []
            for z, a, z_next in mode_assignments[i]:
                z_v = np.asarray(z).flatten()
                a_v = np.asarray(a).flatten()
                z_next_v = np.asarray(z_next).flatten()
                
                if len(z_v) != z_dim or len(a_v) != a_dim or len(z_next_v) != z_dim:
                    continue
                
                X_list.append(np.concatenate([z_v, a_v]))
                Y_list.append(z_next_v)
            
            if len(X_list) < 5:
                mode.quality.is_valid = False
                continue
            
            X = np.array(X_list)
            Y = np.array(Y_list)
            
            try:
                # Check shapes
                assert X.shape[1] == total_dim, f"X cols: {X.shape[1]} != {total_dim}"
                assert Y.shape[1] == z_dim, f"Y cols: {Y.shape[1]} != {z_dim}"
                
                # Model: z_next = A @ z + b @ a = [A | b] @ [z; a]
                # Y = X @ B, where B is (total_dim, z_dim) = (10, 8)
                # After lstsq: B will be (10, 8)
                # A = B[:z_dim, :] = (8, 8)
                # b = B[z_dim:, :].T = (2, 8).T = (8, 2)
                
                B, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                
                if B.shape != (total_dim, z_dim):
                    print(f"Warning: B shape {B.shape} != ({total_dim}, {z_dim})")
                    continue
                
                A_new = B[:z_dim, :]  # (z_dim, z_dim)
                b_new = B[z_dim:, :].T  # (a_dim, z_dim) -> (z_dim, a_dim)
                
                # Ensure shapes are correct
                if A_new.shape != (z_dim, z_dim):
                    A_new = np.zeros((z_dim, z_dim))
                if b_new.shape != (z_dim, a_dim):
                    b_new = np.zeros((z_dim, a_dim))
                
                mode.A = A_new
                mode.b = b_new
                mode.weight = len(mode_assignments[i]) / max(total_assignments, 1)
                
                # Обновляем качество
                self._update_mode_quality(i)
            except Exception as e:
                print(f"Mode fit error: {e}")
                if mode.quality:
                    mode.quality.is_valid = False
    
    def predict_multi_modal(self, z, a, top_k=3):
        """
        Предсказать с несколькими режимами.
        
        ВАЖНО: Используются только валидные режимы.
        """
        z = self.contract.validate_z(z)
        a = self.contract.validate_a(a)
        
        mode_predictions = []
        valid_count = 0
        
        for i, mode in enumerate(self.modes):
            # Пропускаем невалидные режимы
            if mode.quality and not mode.quality.is_valid:
                continue
            
            # Validate mode shapes - fail fast with assertion
            try:
                assert mode.A.shape == (self.contract.latent_dim, self.contract.latent_dim), \
                    f"Mode {i} A: {mode.A.shape} != ({self.contract.latent_dim}, {self.contract.latent_dim})"
                assert mode.b.shape == (self.contract.latent_dim, self.contract.action_dim), \
                    f"Mode {i} b: {mode.b.shape} != ({self.contract.latent_dim}, {self.contract.action_dim})"
                assert mode.c.shape == (self.contract.latent_dim,), \
                    f"Mode {i} c: {mode.c.shape} != ({self.contract.latent_dim},)"
            except AssertionError as e:
                print(f"Mode validation failed: {e}")
                # Try to fix
                mode.A = np.random.randn(self.contract.latent_dim, self.contract.latent_dim) * 0.1
                mode.b = np.random.randn(self.contract.latent_dim, self.contract.action_dim) * 0.2
                mode.c = np.zeros(self.contract.latent_dim)
                continue
            
            z_next = mode.A @ z + mode.b @ a + mode.c
            mode_predictions.append((i, z_next, mode.weight))
            valid_count += 1
        
        # Если нет валидных режимов - используем все
        if not mode_predictions:
            for i, mode in enumerate(self.modes):
                z_next = mode.A @ z + mode.b @ a + mode.c
                mode_predictions.append((i, z_next, mode.weight))
        
        # Сортируем по весу
        mode_predictions.sort(key=lambda x: x[2], reverse=True)
        return mode_predictions[:min(top_k, len(mode_predictions))]
    
    def get_valid_modes(self) -> List[int]:
        """Получить индексы валидных режимов."""
        return [i for i, m in enumerate(self.modes) 
                if m.quality is None or m.quality.is_valid]
    
    def get_mode_stats(self) -> Dict:
        """Статистика режимов для диагностики."""
        valid_modes = self.get_valid_modes()
        total_weight = sum(self.modes[i].weight for i in valid_modes) if valid_modes else 1.0
        
        return {
            'total_modes': self.num_modes,
            'valid_modes': len(valid_modes),
            'total_transitions': len(self.transitions),
            'mode_weights': [self.modes[i].weight for i in range(self.num_modes)],
            'valid_indices': valid_modes,
            'weights_normalized': [self.modes[i].weight / total_weight for i in valid_modes]
        }


class DiagnosticV:
    """
    Диагностический V - read-only наблюдение состояния.
    
    НЕ влияет на policy напрямую.
    Используется для мониторинга и анализа.
    """
    
    def __init__(self, v_min: float = 0.3, v_critical: float = 0.1):
        self.v_min = v_min
        self.v_critical = v_critical
        self.V_history = deque(maxlen=100)
        self.diversity_history = deque(maxlen=100)
        self.divergence_history = deque(maxlen=100)
        self.entropy_history = deque(maxlen=100)
        self.collapse_history = deque(maxlen=100)
    
    def compute(self, transition_kernel: TransitionKernel, z, a) -> Dict:
        """
        Вычислить диагностические метрики.
        
        Returns:
            Dict с trajectory_diversity, trajectory_divergence,
            trajectory_entropy, trajectory_collapse
        """
        z = transition_kernel.contract.validate_z(z)
        a = transition_kernel.contract.validate_a(a)
        
        # Получаем валидные режимы
        mode_predictions = transition_kernel.predict_multi_modal(z, a)
        
        if not mode_predictions:
            return {
                'V': 0.0,
                'trajectory_diversity': 0.0,
                'trajectory_divergence': 0.0,
                'trajectory_entropy': 0.0,
                'trajectory_collapse': 1.0,
                'status': 'critical'
            }
        
        weights = np.array([m[2] for m in mode_predictions])
        weights_norm = weights / (weights.sum() + 1e-8)
        
        # Trajectory diversity (mode entropy)
        trajectory_diversity = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
        max_entropy = np.log(len(weights) + 1e-8)
        trajectory_diversity = trajectory_diversity / max_entropy if max_entropy > 0 else 0
        
        # Trajectory divergence (mode spread)
        endpoints = np.array([m[1] for m in mode_predictions])
        if len(endpoints) > 1:
            pairwise_dists = []
            for i in range(len(endpoints)):
                for j in range(i + 1, len(endpoints)):
                    pairwise_dists.append(np.linalg.norm(endpoints[i] - endpoints[j]))
            trajectory_divergence = np.mean(pairwise_dists) if pairwise_dists else 0
            trajectory_divergence = np.tanh(trajectory_divergence * 2)
        else:
            trajectory_divergence = 0.0
        
        # Trajectory entropy
        trajectory_entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
        max_entropy = np.log(len(weights) + 1e-8)
        trajectory_entropy = trajectory_entropy / max_entropy if max_entropy > 0 else 0
        
        # Trajectory collapse (1 - diversity)
        trajectory_collapse = 1.0 - trajectory_diversity
        
        # Composite V (Diagnostic)
        V = 0.4 * trajectory_diversity + 0.4 * trajectory_divergence + 0.2 * trajectory_entropy
        V = float(np.clip(V, 0, 1))
        
        # Determine status
        if len(transition_kernel.transitions) < 10:
            status = 'warmup'
        elif V < self.v_critical:
            status = 'critical'
        elif V < self.v_min:
            status = 'warning'
        else:
            status = 'healthy'
        
        # Update history
        self.V_history.append(V)
        self.diversity_history.append(trajectory_diversity)
        self.divergence_history.append(trajectory_divergence)
        self.entropy_history.append(trajectory_entropy)
        self.collapse_history.append(trajectory_collapse)
        
        return {
            'V': V,
            'trajectory_diversity': trajectory_diversity,
            'trajectory_divergence': trajectory_divergence,
            'trajectory_entropy': trajectory_entropy,
            'trajectory_collapse': trajectory_collapse,
            'status': status,
            'valid_modes': len(mode_predictions)
        }


class ControlV:
    """
    Управляющий V - сигнал для policy.
    
    Отличается от DiagnosticV:
    1. Сглаживание (exponential moving average)
    2. Фильтрация выбросов
    3. Адаптивный threshold
    """
    
    def __init__(self, diagnostic_v: DiagnosticV, smoothing: float = 0.3):
        self.diagnostic_v = diagnostic_v
        self.smoothing = smoothing
        self.smoothed_V = 0.5
        self.V_threshold = 0.3  # Адаптивный порог
        self.V_history = deque(maxlen=50)
    
    def compute(self, transition_kernel: TransitionKernel, z, a) -> float:
        """
        Вычислить управляющий V для policy.
        
        Returns:
            Сглаженный, отфильтрованный V-score
        """
        diagnostic = self.diagnostic_v.compute(transition_kernel, z, a)
        raw_V = diagnostic['V']
        
        # Exponential smoothing
        self.smoothed_V = self.smoothing * raw_V + (1 - self.smoothing) * self.smoothed_V
        
        # Store in history for threshold adaptation
        self.V_history.append(self.smoothed_V)
        
        # Adapt threshold based on recent history
        if len(self.V_history) >= 20:
            recent_mean = np.mean(list(self.V_history)[-20:])
            recent_std = np.std(list(self.V_history)[-20:])
            # Threshold = mean - 1 std
            self.V_threshold = max(0.1, recent_mean - recent_std)
        
        # Clip output
        control_V = float(np.clip(self.smoothed_V, 0, 1))
        
        return control_V
    
    def get_signals(self) -> Dict:
        """Получить все управляющие сигналы."""
        return {
            'control_V': self.smoothed_V,
            'threshold': self.V_threshold,
            'is_safe': self.smoothed_V >= self.V_threshold,
            'smoothing': self.smoothing
        }


class WorldModelKernel:
    """
    Полный kernel world model - объединяет все компоненты.
    
    Архитектура:
    1. TransitionKernel - ядро с жёсткими контрактами
    2. DiagnosticV - read-only наблюдение
    3. ControlV - smoothed policy signal
    
    Usage:
        kernel = WorldModelKernel(latent_dim=8, action_dim=2)
        
        # Observation loop
        kernel.observe(z, a, z_next)
        diagnostic = kernel.diagnostic.compute(kernel, z, a)
        control_V = kernel.control.compute(kernel, z, a)
        
        # Policy uses control_V
        if kernel.control.get_signals()['is_safe']:
            execute_action()
        else:
            conservative_mode()
    """
    
    def __init__(self, latent_dim: int = 8, action_dim: int = 2, num_modes: int = 3):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.num_modes = num_modes
        
        # Core components
        self.transition = TransitionKernel(latent_dim, action_dim, num_modes)
        self.diagnostic = DiagnosticV()
        self.control = ControlV(self.diagnostic)
        
        # Stats
        self.step_count = 0
    
    def observe(self, z, a, z_next):
        """Наблюдать переход."""
        self.transition.add_transition(z, a, z_next)
        self.step_count += 1
    
    def get_diagnostic_state(self) -> Dict:
        """Получить полное диагностическое состояние."""
        # Get last transition if exists
        if self.transition.transitions:
            z, a, _ = self.transition.transitions[-1]
        else:
            z = np.zeros(self.latent_dim)
            a = np.zeros(self.action_dim)
        
        diagnostic = self.diagnostic.compute(self.transition, z, a)
        mode_stats = self.transition.get_mode_stats()
        
        return {
            **diagnostic,
            'mode_stats': mode_stats,
            'step_count': self.step_count
        }
    
    def get_control_signal(self) -> Dict:
        """Получить управляющий сигнал для policy."""
        if self.transition.transitions:
            z, a, _ = self.transition.transitions[-1]
        else:
            z = np.zeros(self.latent_dim)
            a = np.zeros(self.action_dim)
        
        return {
            'V': self.control.compute(self.transition, z, a),
            **self.control.get_signals(),
            **self.transition.get_mode_stats()
        }
    
    def get_full_state(self) -> Dict:
        """Получить полное состояние kernel."""
        return {
            'diagnostic': self.get_diagnostic_state(),
            'control': self.get_control_signal(),
            'transition_stats': self.transition.get_mode_stats()
        }