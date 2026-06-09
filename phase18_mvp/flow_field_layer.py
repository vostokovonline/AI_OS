"""
Phase 18.7 - Flow Field Layer (Latent Manifold Edition)

Ключевой сдвиг:
  OLD: V = scalar aggregate
  NEW: V(z) = potential field over latent space

Архитектура:
  ┌─────────────────────────────────────────────────────────────┐
  │  LATENT TRAJECTORY BUFFER                                  │
  │  Z = {z₀, z₁, ..., zₙ} — recent states                    │
  ├─────────────────────────────────────────────────────────────┤
  │  KNN GRAPH BUILDER                                         │
  │  neighborhood(z) → local structure in manifold             │
  ├─────────────────────────────────────────────────────────────┤
  │  FIELD ESTIMATOR                                           │
  │  V(z) = weighted avg of neighbors quality                 │
  │  ∇V(z) = local gradient of potential                      │
  ├─────────────────────────────────────────────────────────────┤
  │  VECTOR FIELD                                              │
  │  F(z, a) = E[z_next - z | neighborhood]                   │
  └─────────────────────────────────────────────────────────────┘

Это НЕ переписывание Kernel — это overlay layer.
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field
from collections import deque


@dataclass
class LatentPoint:
    """Точка в латентном пространстве."""
    z: np.ndarray
    a: Optional[np.ndarray] = None
    z_next: Optional[np.ndarray] = None
    step: int = 0
    V_local: float = 0.0
    quality: float = 1.0  # confidence of this point


@dataclass
class Neighborhood:
    """Локальная окрестность точки."""
    center: int  # index in buffer
    k: int  # number of neighbors
    distances: np.ndarray  # distances to neighbors
    indices: np.ndarray  # indices of neighbors
    local_density: float
    tangent_basis: np.ndarray  # local tangent space


class LatentTrajectoryBuffer:
    """
    Буфер траекторий в латентном пространстве.
    
    Хранит историю состояний для построения структуры manifold.
    """
    
    def __init__(self, latent_dim: int, max_size: int = 500):
        self.latent_dim = latent_dim
        self.max_size = max_size
        
        self.points: List[LatentPoint] = []
        self.step_count = 0
    
    def add(
        self,
        z: np.ndarray,
        a: np.ndarray = None,
        z_next: np.ndarray = None,
        quality: float = 1.0
    ):
        """Добавить точку в буфер."""
        z = np.asarray(z).flatten()
        
        point = LatentPoint(
            z=z,
            a=a,
            z_next=z_next,
            step=self.step_count,
            quality=quality
        )
        
        self.points.append(point)
        self.step_count += 1
        
        # Remove oldest if over capacity
        if len(self.points) > self.max_size:
            self.points.pop(0)
    
    def get_recent(self, n: int) -> np.ndarray:
        """Получить последние n состояний."""
        n = min(n, len(self.points))
        return np.array([p.z for p in self.points[-n:]])
    
    def get_all_z(self) -> np.ndarray:
        """Получить все z как матрицу."""
        return np.array([p.z for p in self.points])
    
    def get_states_as_matrix(self) -> np.ndarray:
        """Получить все состояния (z, a) как матрицу."""
        states = []
        for p in self.points:
            if p.a is not None:
                state = np.concatenate([p.z, p.a])
            else:
                state = p.z
            states.append(state)
        return np.array(states) if states else np.zeros((0, self.latent_dim))
    
    def size(self) -> int:
        return len(self.points)


class KNNGraphBuilder:
    """
    Построение kNN графа в латентном пространстве.
    
    Строит локальную структуру manifold.
    """
    
    def __init__(self, k: int = 10, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        
        # Cache for efficiency
        self.knn_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    def compute_knn(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k-nearest neighbors for all points.
        
        Returns:
            distances: (n, k) array of distances
            indices: (n, k) array of neighbor indices
        """
        if len(Z) == 0:
            return np.array([]), np.array([])
        
        n = len(Z)
        k = min(self.k, n - 1)  # can't be neighbor to yourself
        
        if k == 0:
            return np.zeros((n, 0)), np.zeros((n, 0), dtype=int)
        
        # Compute pairwise distances
        # Using efficient batch computation
        Z_sq = np.sum(Z ** 2, axis=1, keepdims=True)
        distances_matrix = Z_sq + Z_sq.T - 2 * Z @ Z.T
        distances_matrix = np.sqrt(np.clip(distances_matrix, 0, None))
        
        # Set diagonal to inf (exclude self)
        np.fill_diagonal(distances_matrix, np.inf)
        
        # Get k nearest for each point
        indices = np.argsort(distances_matrix, axis=1)[:, :k]
        
        # Gather distances
        row_indices = np.arange(n)[:, None]
        distances = distances_matrix[row_indices, indices]
        
        return distances, indices
    
    def build_neighborhood(
        self,
        Z: np.ndarray,
        center_idx: int
    ) -> Neighborhood:
        """
        Построить локальную окрестность для точки.
        
        Args:
            Z: all points (n, latent_dim)
            center_idx: index of center point
        
        Returns:
            Neighborhood object
        """
        if len(Z) < 2:
            return Neighborhood(
                center=center_idx,
                k=0,
                distances=np.array([]),
                indices=np.array([]),
                local_density=0.0,
                tangent_basis=np.zeros((Z.shape[1], Z.shape[1]))
            )
        
        distances, indices = self.compute_knn(Z)
        
        # Get neighbors of center point
        center_distances = distances[center_idx]
        center_indices = indices[center_idx]
        
        # Compute local density
        local_density = 1.0 / (np.mean(center_distances) + 1e-6)
        
        # Compute local tangent basis
        center = Z[center_idx]
        neighbors = Z[center_indices]
        
        # Tangent space = subspace spanned by neighbor differences
        if len(neighbors) > 1:
            differences = neighbors - center
            # PCA on differences to get tangent basis
            U, S, Vt = np.linalg.svd(differences, full_matrices=False)
            tangent_basis = Vt[:min(len(S), Z.shape[1])].T
        else:
            tangent_basis = np.eye(Z.shape[1])
        
        return Neighborhood(
            center=center_idx,
            k=len(center_indices),
            distances=center_distances,
            indices=center_indices,
            local_density=local_density,
            tangent_basis=tangent_basis
        )


class FieldEstimator:
    """
    Оценка поля V(z) в латентном пространстве.
    
    V(z) = potential field = weighted average of neighbors quality
    
    Uses:
    - Local density (more neighbors = higher confidence)
    - Trajectory quality (consistent paths = higher V)
    - Transition stability (predictable = higher V)
    """
    
    def __init__(
        self,
        knn_builder: KNNGraphBuilder,
        V_decay: float = 0.1,  # how fast V changes in space
        density_weight: float = 0.4,
        trajectory_weight: float = 0.3,
        stability_weight: float = 0.3
    ):
        self.knn_builder = knn_builder
        self.V_decay = V_decay
        self.density_weight = density_weight
        self.trajectory_weight = trajectory_weight
        self.stability_weight = stability_weight
        
        self.Z_history: deque = deque(maxlen=1000)
        self.V_history: deque = deque(maxlen=100)
    
    def estimate_V(self, z: np.ndarray, buffer: LatentTrajectoryBuffer) -> float:
        """
        Оценить V(z) для точки.
        
        V(z) = weighted average of:
        1. Local density (more neighbors = higher confidence)
        2. Trajectory quality (consistent paths)
        3. Transition stability (predictable dynamics)
        """
        if buffer.size() < 5:
            return 0.5  # default
        
        Z = buffer.get_all_z()
        
        # Find k nearest neighbors
        distances, indices = self.knn_builder.compute_knn(Z)
        
        if len(distances) == 0:
            return 0.5
        
        # Compute V based on local structure
        V_components = []
        
        for i in range(len(Z)):
            # 1. Local density
            local_V = 1.0 / (np.mean(distances[i]) + 1e-6)
            V_components.append(local_V)
        
        # Normalize
        V_components = np.array(V_components)
        V_components = V_components / (np.max(V_components) + 1e-6)
        
        # Weight by distance to query point
        z = np.asarray(z).flatten()
        
        # Find closest points to z
        if len(Z) > 0:
            dists_to_z = np.linalg.norm(Z - z, axis=1)
            
            # Gaussian weighting
            weights = np.exp(-dists_to_z / self.V_decay)
            weights = weights / (weights.sum() + 1e-6)
            
            # Weighted V
            V = np.sum(weights * V_components)
        else:
            V = 0.5
        
        return float(np.clip(V, 0, 1))
    
    def estimate_gradient_V(
        self,
        z: np.ndarray,
        buffer: LatentTrajectoryBuffer,
        epsilon: float = 0.01
    ) -> np.ndarray:
        """
        Оценить градиент V(z).
        
        ∇V(z) ≈ (V(z + ε) - V(z - ε)) / 2ε
        
        Returns:
            Direction of decreasing V (direction of steepest descent)
        """
        z = np.asarray(z).flatten()
        dim = len(z)
        
        gradient = np.zeros(dim)
        
        for i in range(dim):
            z_plus = z.copy()
            z_minus = z.copy()
            
            z_plus[i] += epsilon
            z_minus[i] -= epsilon
            
            V_plus = self.estimate_V(z_plus, buffer)
            V_minus = self.estimate_V(z_minus, buffer)
            
            gradient[i] = (V_plus - V_minus) / (2 * epsilon)
        
        return gradient
    
    def get_local_dynamics(
        self,
        z: np.ndarray,
        buffer: LatentTrajectoryBuffer
    ) -> Dict:
        """
        Получить локальную динамику вокруг z.
        
        Returns:
            Dict с локальной статистикой
        """
        if buffer.size() < 5:
            return {
                'local_V': 0.5,
                'local_density': 0.0,
                'dominant_direction': np.zeros_like(z),
                'uncertainty': 0.5
            }
        
        Z = buffer.get_all_z()
        
        # Find closest points
        dists = np.linalg.norm(Z - z, axis=1)
        sorted_idx = np.argsort(dists)[:self.knn_builder.k]
        
        local_Z = Z[sorted_idx]
        local_points = [buffer.points[i] for i in sorted_idx]
        
        # Local V
        local_V = self.estimate_V(z, buffer)
        
        # Local density
        local_density = len(sorted_idx) / (np.mean(dists[sorted_idx]) + 1e-6)
        local_density = np.tanh(local_density / 10)
        
        # Dominant direction (first PC of local cloud)
        if len(local_Z) > 1:
            centered = local_Z - np.mean(local_Z, axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            dominant_direction = Vt[0] if len(Vt) > 0 else np.zeros_like(z)
        else:
            dominant_direction = np.zeros_like(z)
        
        # Uncertainty (spread of local cloud)
        if len(local_Z) > 1:
            spread = np.mean(np.var(local_Z, axis=0))
            uncertainty = np.tanh(spread / 2)
        else:
            uncertainty = 0.5
        
        return {
            'local_V': local_V,
            'local_density': float(local_density),
            'dominant_direction': dominant_direction,
            'uncertainty': float(uncertainty)
        }


class VectorField:
    """
    Векторное поле переходов F(z, a).
    
    F(z, a) = E[z_next - z | neighborhood]
    
    Это поле показывает "куда течёт система" при данном действии.
    """
    
    def __init__(
        self,
        knn_builder: KNNGraphBuilder,
        field_estimator: FieldEstimator,
        num_directions: int = 8
    ):
        self.knn_builder = knn_builder
        self.field_estimator = field_estimator
        self.num_directions = num_directions
        
        # Directions in latent space
        self.directions = self._generate_directions()
    
    def _generate_directions(self) -> np.ndarray:
        """Генерировать равномерно распределённые направления."""
        dim = 8  # assumed latent dim
        directions = []
        
        for i in range(self.num_directions):
            angle = 2 * np.pi * i / self.num_directions
            d = np.zeros(dim)
            d[0] = np.cos(angle)
            d[1] = np.sin(angle)
            directions.append(d)
        
        return np.array(directions)
    
    def compute_flow(
        self,
        z: np.ndarray,
        a: np.ndarray,
        buffer: LatentTrajectoryBuffer
    ) -> np.ndarray:
        """
        Вычислить ожидаемый поток из z при действии a.
        
        F(z, a) = weighted average of (z_next - z) from neighbors
        
        Returns:
            Flow vector (direction and magnitude of expected transition)
        """
        if buffer.size() < 3:
            return np.zeros_like(z)
        
        Z = buffer.get_all_z()
        
        # Find neighbors that had similar action
        neighbors_with_action = []
        
        for i, point in enumerate(buffer.points):
            if point.a is None:
                continue
            
            # Check if action is similar (within 45 degrees or close magnitude)
            a_point = np.asarray(point.a).flatten()
            a_query = np.asarray(a).flatten()
            
            action_similarity = self._action_similarity(a_point, a_query)
            
            if action_similarity > 0.5:
                neighbors_with_action.append(i)
        
        if not neighbors_with_action:
            # No matching actions - use all recent points
            neighbors_with_action = list(range(max(0, buffer.size() - 20), buffer.size()))
        
        if not neighbors_with_action:
            return np.zeros_like(z)
        
        # Compute flow from neighbors
        flows = []
        weights = []
        
        for i in neighbors_with_action:
            point = buffer.points[i]
            
            if point.z_next is None:
                continue
            
            # Flow = z_next - z (expected change)
            flow = point.z_next - point.z
            
            # Weight by recency and distance
            recency_weight = np.exp(-0.01 * (buffer.step_count - point.step))
            distance_weight = np.exp(-np.linalg.norm(point.z - z))
            
            weight = recency_weight * distance_weight
            
            flows.append(flow)
            weights.append(weight)
        
        if not flows:
            return np.zeros_like(z)
        
        flows = np.array(flows)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average flow
        flow = np.sum(flows * weights[:, None], axis=0)
        
        return flow
    
    def _action_similarity(self, a1: np.ndarray, a2: np.ndarray) -> float:
        """Compute similarity between two actions."""
        a1 = a1.flatten()
        a2 = a2.flatten()
        
        if np.linalg.norm(a1) < 1e-6 or np.linalg.norm(a2) < 1e-6:
            return 0.5  # neutral
        
        # Cosine similarity
        cos_sim = np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-6)
        
        # Also check magnitude
        mag_ratio = min(np.linalg.norm(a1), np.linalg.norm(a2)) / max(np.linalg.norm(a1), np.linalg.norm(a2) + 1e-6)
        
        return (cos_sim + 1) / 2 * mag_ratio
    
    def get_flow_field(
        self,
        center: np.ndarray,
        radius: float,
        buffer: LatentTrajectoryBuffer,
        num_samples: int = 20
    ) -> Dict:
        """
        Получить поле потоков вокруг центра.
        
        Для визуализации - генерирует вектора в точках вокруг центра.
        
        Returns:
            Dict с positions, directions, magnitudes
        """
        center = np.asarray(center).flatten()
        
        # Generate sample points
        samples = []
        for i in range(num_samples):
            # Random direction
            direction = np.random.randn(len(center))
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            
            # Random radius (within sphere)
            r = radius * np.random.rand()
            
            point = center + direction * r
            samples.append(point)
        
        samples = np.array(samples)
        
        # Compute flow for each sample
        flows = []
        magnitudes = []
        
        for sample in samples:
            # Assume default action
            a = np.zeros(len(center) // 4)  # action dim
            flow = self.compute_flow(sample, a, buffer)
            flows.append(flow)
            magnitudes.append(np.linalg.norm(flow))
        
        return {
            'positions': samples,
            'directions': np.array(flows),
            'magnitudes': np.array(magnitudes),
            'center': center
        }


class FlowFieldLayer:
    """
    Полный Flow Field слой — overlay на World Model Kernel.
    
    Не переписывает Kernel, а добавляет:
    1. Geometry (latent buffer + kNN)
    2. Field estimation (V(z))
    3. Vector field (F(z,a))
    4. Visualization data
    
    Usage:
        flow_layer = FlowFieldLayer(latent_dim=8, action_dim=2)
        
        # After each kernel step:
        flow_layer.update(z, a, z_next)
        
        # Get field data for visualization:
        field_data = flow_layer.get_field_data(z)
        
        # Or get full field around point:
        flow_field = flow_layer.get_flow_field(z, radius=2.0)
    """
    
    def __init__(
        self,
        latent_dim: int = 8,
        action_dim: int = 2,
        k_neighbors: int = 10,
        buffer_size: int = 500
    ):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Core components
        self.buffer = LatentTrajectoryBuffer(latent_dim, buffer_size)
        self.knn_builder = KNNGraphBuilder(k=k_neighbors)
        self.field_estimator = FieldEstimator(self.knn_builder)
        self.vector_field = VectorField(self.knn_builder, self.field_estimator)
        
        # Metrics
        self.V_global = 0.5
        self.field_history = deque(maxlen=100)
    
    def update(
        self,
        z: np.ndarray,
        a: np.ndarray = None,
        z_next: np.ndarray = None
    ):
        """
        Обновить flow field с новым наблюдением.
        
        Вызывать после каждого step.
        """
        # Add to buffer
        self.buffer.add(z, a, z_next)
        
        # Update global V estimate
        self.V_global = self.field_estimator.estimate_V(z, self.buffer)
        
        # Store history
        self.field_history.append({
            'z': z.copy() if isinstance(z, np.ndarray) else z,
            'V': self.V_global
        })
    
    def get_field_data(self, z: np.ndarray) -> Dict:
        """
        Получить данные поля для визуализации.
        
        Returns:
            Dict с полной информацией о локальной структуре
        """
        z = np.asarray(z).flatten()
        
        # Local dynamics
        local = self.field_estimator.get_local_dynamics(z, self.buffer)
        
        # Flow
        a = np.zeros(self.action_dim)  # default action
        flow = self.vector_field.compute_flow(z, a, self.buffer)
        
        # Gradient of V
        grad_V = self.field_estimator.estimate_gradient_V(z, self.buffer)
        
        # Get neighbors
        Z = self.buffer.get_all_z()
        if len(Z) > 0:
            distances, indices = self.knn_builder.compute_knn(Z)
            
            # Find closest to z
            dists_to_z = np.linalg.norm(Z - z, axis=1)
            closest_idx = np.argsort(dists_to_z)[:5]
            neighbors = Z[closest_idx]
        else:
            neighbors = np.zeros((0, self.latent_dim))
            distances = np.array([])
            indices = np.array([])
        
        return {
            'z': z,
            'V': self.V_global,
            'local_V': local['local_V'],
            'local_density': local['local_density'],
            'dominant_direction': local['dominant_direction'],
            'uncertainty': local['uncertainty'],
            'flow': flow,
            'grad_V': grad_V,
            'neighbors': neighbors,
            'neighbor_distances': distances[:5] if len(distances) > 0 else np.array([]),
            'buffer_size': self.buffer.size()
        }
    
    def get_flow_field(
        self,
        center: np.ndarray,
        radius: float = 2.0,
        num_samples: int = 30
    ) -> Dict:
        """
        Получить поле потоков для области.
        
        Для 2D/3D визуализации.
        """
        return self.vector_field.get_flow_field(center, radius, num_samples, self.buffer)
    
    def get_2d_projection(
        self,
        method: str = 'pca'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Получить 2D проекцию латентного пространства.
        
        Для визуализации на canvas.
        
        Returns:
            (points_2d, labels) — 2D координаты точек
        """
        Z = self.buffer.get_all_z()
        
        if len(Z) < 3:
            # Not enough points
            return np.array([[0, 0]]), np.array([0])
        
        if method == 'pca':
            # PCA projection
            centered = Z - np.mean(Z, axis=0)
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            
            if len(S) >= 2:
                projected = centered @ Vt[:2].T
            else:
                projected = np.zeros((len(Z), 2))
        
        elif method == 'tsne':
            # Simple t-SNE approximation
            # In production: use sklearn.manifold.TSNE
            projected = self._simple_tsne(Z)
        
        else:
            # Just first 2 dimensions
            projected = Z[:, :2]
        
        # Labels = V value for coloring
        labels = np.array([self.field_estimator.estimate_V(z, self.buffer) for z in Z])
        
        return projected, labels
    
    def _simple_tsne(self, Z: np.ndarray, perplexity: float = 10.0) -> np.ndarray:
        """
        Простая t-SNE аппроксимация.
        
        In production: use sklearn.manifold.TSNE
        """
        n = len(Z)
        
        if n < perplexity:
            perplexity = n - 1
        
        # Compute similarities in high-dim
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(Z[i] - Z[j])
        
        # Convert to probabilities
        P = np.zeros((n, n))
        for i in range(n):
            # Exclude self
            row = distances[i].copy()
            row[i] = np.inf
            
            # Sort and find sigmas
            sorted_row = np.sort(row)
            
            # Simple binary search for sigma
            target = np.log(perplexity)
            sigma = 1.0
            
            for _ in range(20):
                prob = np.exp(-row ** 2 / (2 * sigma ** 2))
                prob[i] = 0  # exclude self
                sum_prob = np.sum(prob)
                
                if sum_prob < 1e-6:
                    sigma *= 0.5
                    continue
                
                entropy = -np.sum(prob * np.log(prob + 1e-10)) / sum_prob
                
                if entropy > target:
                    sigma *= 1.5
                else:
                    sigma *= 0.5
                
                if abs(entropy - target) < 0.1:
                    break
            
            P[i] = prob / (sum_prob + 1e-10)
        
        # Symmetrize
        P = (P + P.T) / (2 * n)
        
        # Project to 2D (simple spectral)
        # In production: use proper t-SNE optimization
        M = np.eye(n) - P
        eigenvalues, eigenvectors = np.linalg.eig(M)
        
        idx = np.argsort(eigenvalues.real)[:2]
        projected = eigenvectors.real[:, idx]
        
        return projected
    
    def get_diagnostic_state(self) -> Dict:
        """Получить диагностическое состояние."""
        Z = self.buffer.get_all_z()
        
        # Compute global metrics
        if len(Z) > 10:
            # Spread of points
            spread = np.mean(np.std(Z, axis=0))
            
            # Local density variance
            local_densities = []
            for z in Z[:50]:  # sample
                local = self.field_estimator.get_local_dynamics(z, self.buffer)
                local_densities.append(local['local_density'])
            
            density_variance = np.var(local_densities) if local_densities else 0
        else:
            spread = 0
            density_variance = 0
        
        return {
            'V': self.V_global,
            'buffer_size': self.buffer.size(),
            'spread': float(spread),
            'density_variance': float(density_variance),
            'field_stability': 1.0 - min(density_variance, 1.0)
        }