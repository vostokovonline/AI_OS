"""
EMOTIONAL TRAJECTORY CLUSTERING
Кластеризует формы эмоциональных переходов вместо усреднения пользователей.

Key Idea:
- НЕ среднее по пользователям (плохо - разные паттерны)
- Кластеры траекторий (хорошо - форма перехода, не значения)

Траектория = (emotional_state_before → intermediate states → emotional_state_after)
Форма = shape of curve, не absolute values
"""

import uuid
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from sqlalchemy import select, and_
from database import AsyncSessionLocal
from models import AffectiveMemoryEntry, Goal


class TrajectoryPoint:
    """Точка эмоциональной траектории"""
    def __init__(self, state: Dict[str, float], created_at: datetime, phase: str):
        """
        Args:
            state: {arousal, valence, focus, confidence}
            created_at: Когда это было
            phase: 'start', 'during', 'end'
        """
        self.state = state
        self.created_at = created_at  # ← Используем created_at как в модели
        self.timestamp = created_at  # ← Для совместимости со старым кодом
        self.phase = phase


class EmotionalTrajectory:
    """Эмоциональная траектория - sequence of states through task lifecycle"""

    def __init__(
        self,
        trajectory_id: str,
        user_id: str,
        goal_id: Optional[str],
        action_type: str,
        outcome: str,
        points: List[TrajectoryPoint]
    ):
        self.trajectory_id = trajectory_id
        self.user_id = user_id
        self.goal_id = goal_id
        self.action_type = action_type  # 'deep_goal_decomposition', 'complex_execution', etc
        self.outcome = outcome  # 'success', 'failure', 'aborted'
        self.points = points

    def get_shape_features(self) -> Dict[str, float]:
        """
        Извлекает ФОРМУ траектории (не абсолютные значения!)

        Формовые признаки (shape features):
        - delta_changes: как изменилась каждая размерность
        - volatility: насколько сильно колебалась
        - trend_direction: общий вектор изменения
        - peak_deviation: максимальное отклонение от start
        """
        if len(self.points) < 2:
            return {}

        start_state = self.points[0].state
        end_state = self.points[-1].state

        # Дельты для каждой размерности
        deltas = {
            f"{dim}_delta": end_state.get(dim, 0.5) - start_state.get(dim, 0.5)
            for dim in ["arousal", "valence", "focus", "confidence"]
        }

        # Volatility = среднее изменение между соседними точками
        if len(self.points) > 2:
            changes = []
            for i in range(1, len(self.points)):
                prev_state = self.points[i-1].state
                curr_state = self.points[i].state
                change = sum(
                    abs(curr_state.get(dim, 0.5) - prev_state.get(dim, 0.5))
                    for dim in ["arousal", "valence", "focus", "confidence"]
                )
                changes.append(change)
            volatility = sum(changes) / len(changes)
        else:
            volatility = 0.0

        # Peak deviation = максимальное отклонение от start
        peak_deviations = []
        for point in self.points[1:]:
            deviation = sum(
                abs(point.state.get(dim, 0.5) - start_state.get(dim, 0.5))
                for dim in ["arousal", "valence", "focus", "confidence"]
            )
            peak_deviations.append(deviation)
        peak_deviation = max(peak_deviations) if peak_deviations else 0.0

        # Trend direction = вектор изменений (4D vector)
        trend_vector = [
            end_state.get(dim, 0.5) - start_state.get(dim, 0.5)
            for dim in ["arousal", "valence", "focus", "confidence"]
        ]

        return {
            **deltas,
            "volatility": volatility,
            "peak_deviation": peak_deviation,
            "trend_vector": trend_vector,
            "num_points": len(self.points),
            "duration_hours": (
                (self.points[-1].timestamp - self.points[0].timestamp).total_seconds() / 3600
                if len(self.points) >= 2 else 0.0
            )
        }


class TrajectoryExtractor:
    """Извлекает траектории из Affective Memory"""

    async def extract_trajectories(
        self,
        user_id: Optional[str] = None,
        action_type: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 100
    ) -> List[EmotionalTrajectory]:
        """
        Извлекает траектории из Affective Memory

        Args:
            user_id: Фильтр по пользователю (None = все)
            action_type: Фильтр по типу действия
            outcome: Фильтр по исходу
            limit: Максимум траекторий
        """
        async with AsyncSessionLocal() as db:
            # Получаем записи из Affective Memory
            query = select(AffectiveMemoryEntry).order_by(
                AffectiveMemoryEntry.created_at.desc()
            )

            if user_id:
                query = query.where(AffectiveMemoryEntry.user_id == uuid.UUID(user_id))

            if limit:
                query = query.limit(limit * 2)  # Берем больше, отфильтруем ниже

            result = await db.execute(query)
            entries = result.scalars().all()

            # Группируем по goal_id для построения траекторий
            # Траектория = все записи для одной цели в хронологическом порядке
            # Траектория = все записи для одной цели в хронологическом порядке
            goal_trajectories = {}  # {goal_id: [entries]}

            for entry in entries:
                goal_id = entry.goal_id
                if not goal_id:
                    continue

                if goal_id not in goal_trajectories:
                    goal_trajectories[goal_id] = []
                goal_trajectories[goal_id].append(entry)

            # Строим траектории
            trajectories = []

            for goal_id, goal_entries in goal_trajectories.items():
                # Сортируем по created_at
                goal_entries.sort(key=lambda e: e.created_at)

                # Определяем action_type по цели
                action = await self._infer_action_type(db, goal_id)
                if action_type and action != action_type:
                    continue

                # Определяем outcome
                last_entry = goal_entries[-1]
                outcome_val = last_entry.outcome
                if outcome and outcome_val != outcome:
                    continue

                # Строим точки траектории
                points = []
                for entry in goal_entries:
                    # Фаза определения
                    if entry == goal_entries[0]:
                        phase = 'start'
                    elif entry == goal_entries[-1]:
                        phase = 'end'
                    else:
                        phase = 'during'

                    point = TrajectoryPoint(
                        state=entry.emotional_state_before,
                        created_at=entry.created_at,
                        phase=phase
                    )
                    points.append(point)

                # Создаем траекторию
                trajectory = EmotionalTrajectory(
                    trajectory_id=str(uuid.uuid4()),
                    user_id=str(goal_entries[0].user_id),
                    goal_id=str(goal_id),
                    action_type=action,
                    outcome=outcome_val,
                    points=points
                )
                trajectories.append(trajectory)

                if len(trajectories) >= limit:
                    break

            return trajectories

    async def _infer_action_type(self, db, goal_id: uuid.UUID) -> str:
        """Определяет тип действия по цели"""
        try:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return "unknown"

            # Определяем action_type по свойствам цели
            if goal.is_atomic:
                return "simple_task"
            elif goal.depth_level >= 2:
                return "deep_goal_decomposition"
            elif goal.goal_type == "exploratory":
                return "exploration_task"
            elif goal.goal_type == "continuous":
                return "routine_task"
            else:
                return "complex_execution"

        except:
            return "unknown"


class TrajectoryCluster:
    """Кластер эмоциональных траекторий"""

    def __init__(self, cluster_id: str, action_type: str):
        self.cluster_id = cluster_id
        self.action_type = action_type
        self.trajectories: List[EmotionalTrajectory] = []
        self.centroid_features: Optional[Dict[str, float]] = None
        self.typical_outcome: Optional[str] = None
        self.success_rate: float = 0.0

    def add_trajectory(self, trajectory: EmotionalTrajectory):
        """Добавляет траекторию в кластер"""
        self.trajectories.append(trajectory)
        self._recalculate()

    def _recalculate(self):
        """Пересчитывает характеристики кластера"""
        if not self.trajectories:
            return

        # Вычисляем centroid (средние shape features)
        all_features = [t.get_shape_features() for t in self.trajectories]
        all_features = [f for f in all_features if f]  # убираем пустые

        if not all_features:
            return

        # Усредняем признаки
        self.centroid_features = {}
        for key in all_features[0].keys():
            if key == "trend_vector":
                # Для вектора усредняем покомпонентно
                vectors = [f[key] for f in all_features if isinstance(f[key], list)]
                if vectors:
                    self.centroid_features[key] = [
                        sum(v[i] for v in vectors) / len(vectors)
                        for i in range(len(vectors[0]))
                    ]
            elif isinstance(all_features[0][key], (int, float)):
                values = [f[key] for f in all_features if key in f]
                if values:
                    self.centroid_features[key] = sum(values) / len(values)

        # Вычисляем типичный исход
        outcomes = [t.outcome for t in self.trajectories]
        success_count = outcomes.count("success")
        self.typical_outcome = "success" if success_count > len(outcomes) / 2 else "failure"
        self.success_rate = success_count / len(outcomes) if outcomes else 0.0

    def predict_outcome(self, trajectory: EmotionalTrajectory) -> Tuple[str, float]:
        """
        Предсказывает исход для траектории

        Returns:
            (predicted_outcome, confidence)
        """
        if not self.trajectories:
            return "unknown", 0.0

        # Confidence = размер кластера (чем больше, тем увереннее)
        confidence = min(len(self.trajectories) / 10.0, 1.0)  # нормируем до 1.0

        return self.typical_outcome, confidence


class TrajectoryClusterer:
    """Кластеризует эмоциональные траектории"""

    def __init__(self, num_clusters: int = 5):
        """
        Args:
            num_clusters: Количество кластеров для каждого action_type
        """
        self.num_clusters = num_clusters
        self.extractor = TrajectoryExtractor()
        self.clusters: Dict[str, List[TrajectoryCluster]] = {}  # {action_type: [clusters]}

    async def build_clusters(self, user_id: Optional[str] = None):
        """
        Строит кластеры из Affective Memory

        Args:
            user_id: Если указан, строит кластеры только для пользователя
        """
        # Извлекаем все траектории
        all_trajectories = await self.extractor.extract_trajectories(
            user_id=user_id,
            limit=1000
        )

        # Группируем по action_type
        trajectories_by_action = {}
        for traj in all_trajectories:
            action = traj.action_type
            if action not in trajectories_by_action:
                trajectories_by_action[action] = []
            trajectories_by_action[action].append(traj)

        # Кластеризуем каждую группу отдельно
        self.clusters = {}

        for action_type, trajectories in trajectories_by_action.items():
            if not trajectories:
                continue

            # Используем простой K-means по shape features
            action_clusters = await self._kmeans_clustering(
                trajectories,
                self.num_clusters
            )

            self.clusters[action_type] = action_clusters

            print(f"📊 Built {len(action_clusters)} clusters for action '{action_type}'")

    async def _kmeans_clustering(
        self,
        trajectories: List[EmotionalTrajectory],
        k: int
    ) -> List[TrajectoryCluster]:
        """Простой K-means по shape features"""

        if len(trajectories) < k:
            # Если траекторий мало, создаем по одной на кластер
            k = max(1, len(trajectories))

        # Инициализация: выбираем k случайных траекторий как центроиды
        import random
        initial_indices = random.sample(range(len(trajectories)), k)
        initial_centroids = [
            trajectories[i].get_shape_features()
            for i in initial_indices
        ]

        # Инициализируем кластеры
        clusters = [
            TrajectoryCluster(
                cluster_id=f"{trajectories[initial_indices[i]].action_type}_cluster_{i}",
                action_type=trajectories[0].action_type
            )
            for i in range(k)
        ]

        # K-means итерации
        max_iterations = 10
        for iteration in range(max_iterations):
            # Очищаем кластеры
            for cluster in clusters:
                cluster.trajectories = []

            # Распределяем траектории по ближайшим центроидам
            for traj in trajectories:
                features = traj.get_shape_features()
                if not features:
                    continue

                # Находим ближайший кластер
                best_cluster_idx = self._find_nearest_cluster(features, initial_centroids)
                clusters[best_cluster_idx].add_trajectory(traj)

            # Пересчитываем центроиды
            new_centroids = [
                cluster.centroid_features
                for cluster in clusters
            ]

            # Проверяем сходимость
            if self._centroids_converged(initial_centroids, new_centroids):
                break

            initial_centroids = new_centroids

        return clusters

    def _find_nearest_cluster(
        self,
        features: Dict[str, float],
        centroids: List[Dict[str, float]]
    ) -> int:
        """Находит ближайший центроид"""

        min_distance = float('inf')
        best_idx = 0

        for idx, centroid in enumerate(centroids):
            if not centroid:
                continue

            distance = self._compute_distance(features, centroid)
            if distance < min_distance:
                min_distance = distance
                best_idx = idx

        return best_idx

    def _compute_distance(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """Вычисляет расстояние между двумя sets of features"""

        # Евклидово расстояние по числовым признакам
        distance = 0.0
        count = 0

        for key in features1:
            if key == "trend_vector":
                # Для векторов вычисляем отдельно
                vec1 = features1.get(key, [0, 0, 0, 0])
                vec2 = features2.get(key, [0, 0, 0, 0])
                if isinstance(vec1, list) and isinstance(vec2, list):
                    vec_distance = sum(
                        (vec1[i] - vec2[i]) ** 2
                        for i in range(min(len(vec1), len(vec2)))
                    )
                    distance += vec_distance
                    count += 1
            elif isinstance(features1[key], (int, float)) and isinstance(features2.get(key), (int, float)):
                distance += (features1[key] - features2[key]) ** 2
                count += 1

        if count == 0:
            return float('inf')

        return distance ** 0.5  # sqrt

    def _centroids_converged(
        self,
        old_centroids: List[Dict[str, float]],
        new_centroids: List[Dict[str, float]],
        threshold: float = 0.01
    ) -> bool:
        """Проверяет сходимость центроидов"""

        for old, new in zip(old_centroids, new_centroids):
            if not old or not new:
                return False

            distance = self._compute_distance(old, new)
            if distance > threshold:
                return False

        return True

    def find_similar_trajectories(
        self,
        trajectory: EmotionalTrajectory,
        top_k: int = 5
    ) -> List[Tuple[EmotionalTrajectory, float]]:
        """
        Находит похожие траектории

        Returns:
            List of (trajectory, similarity_score)
        """
        # Ищем в соответствующем кластере
        action_type = trajectory.action_type
        if action_type not in self.clusters:
            return []

        # Находим ближайший кластер
        features = trajectory.get_shape_features()
        if not features:
            return []

        best_cluster = None
        min_distance = float('inf')

        for cluster in self.clusters[action_type]:
            if cluster.centroid_features:
                distance = self._compute_distance(features, cluster.centroid_features)
                if distance < min_distance:
                    min_distance = distance
                    best_cluster = cluster

        if not best_cluster:
            return []

        # Возвращаем top-k траекторий из кластера
        similarities = []
        for traj in best_cluster.trajectories:
            traj_features = traj.get_shape_features()
            if traj_features:
                distance = self._compute_distance(features, traj_features)
                # Конвертируем distance в similarity (1 / (1 + distance))
                similarity = 1.0 / (1.0 + distance)
                similarities.append((traj, similarity))

        # Сортируем по similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def predict_trajectory_outcome(
        self,
        trajectory: EmotionalTrajectory
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Предсказывает исход траектории на основе кластеров

        Returns:
            (predicted_outcome, confidence, expected_delta)
        """
        # Находим похожие траектории
        similar_trajectories = self.find_similar_trajectories(trajectory, top_k=10)

        if not similar_trajectories:
            return "unknown", 0.0, {}

        # Агрегируем исходы
        total_weight = 0.0
        success_weight = 0.0
        expected_deltas = {
            "arousal": 0.0,
            "valence": 0.0,
            "focus": 0.0,
            "confidence": 0.0
        }

        for traj, similarity in similar_trajectories:
            weight = similarity
            total_weight += weight

            if traj.outcome == "success":
                success_weight += weight

            # Вычисляем delta для этой траектории
            if len(traj.points) >= 2:
                start_state = traj.points[0].state
                end_state = traj.points[-1].state

                for dim in expected_deltas:
                    delta = end_state.get(dim, 0.5) - start_state.get(dim, 0.5)
                    expected_deltas[dim] += delta * weight

        # Нормируем
        if total_weight > 0:
            success_rate = success_weight / total_weight
            predicted_outcome = "success" if success_rate > 0.5 else "failure"
            confidence = abs(success_rate - 0.5) * 2.0  # 0 при 0.5, 1 при 0/1

            for dim in expected_deltas:
                expected_deltas[dim] /= total_weight
        else:
            predicted_outcome = "unknown"
            confidence = 0.0

        return predicted_outcome, confidence, expected_deltas


# =============================================================================
# Глобальный экземпляр
# =============================================================================

trajectory_clusterer = TrajectoryClusterer(num_clusters=5)
