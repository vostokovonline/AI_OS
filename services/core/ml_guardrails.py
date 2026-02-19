"""
ML GUARDRAILS - Safety mechanisms for ML-based emotional forecasting

Цель: Не дать ML тихо деградировать систему.

Компоненты:
1. Training Quality Gates - проверка качества после обучения
2. Per-Action Confidence - разные thresholds для разных действий
3. Drift Detection - обнаружение сдвига распределения
4. Forecast Error Tracking - запись и анализ ошибок
"""

import uuid
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select, and_
from database import AsyncSessionLocal
from models import AffectiveMemoryEntry, EmotionalLayerState

# =============================================================================
# TRAINING QUALITY GATES
# =============================================================================

class TrainingQualityGates:
    """
    Проверяет качество обученной модели перед развертыванием.

    Если модель не проходит gate → она не помечается как available.
    """

    # Thresholds для качества модели
    QUALITY_THRESHOLDS = {
        "min_r2_score": 0.4,          # Минимальный R² (объясненная дисперсия)
        "max_mse": 0.05,              # Максимальная MSE (среднеквадратичная ошибка)
        "min_samples": 30,            # Минимальное количество training samples
        "max_train_test_gap": 0.2,    # Максимальный разрыв между train/test R²
    }

    @classmethod
    def evaluate_training_result(
        cls,
        metrics: Dict[str, float],
        training_samples: int
    ) -> Tuple[bool, List[str]]:
        """
        Оценивает результат обучения.

        Args:
            metrics: {'mse': 0.023, 'r2': 0.67, 'test_r2': 0.62}
            training_samples: Количество training samples

        Returns:
            (passed, reasons)
            - passed: True если модель прошла все gates
            - reasons: Список причин если не прошла
        """
        reasons = []

        # Gate 1: Минимальное количество samples
        if training_samples < cls.QUALITY_THRESHOLDS["min_samples"]:
            reasons.append(
                f"Insufficient data: {training_samples} samples "
                f"(minimum {cls.QUALITY_THRESHOLDS['min_samples']} required)"
            )

        # Gate 2: Минимальный R²
        r2_score = metrics.get("r2", 0.0)
        if r2_score < cls.QUALITY_THRESHOLDS["min_r2_score"]:
            reasons.append(
                f"Low R² score: {r2_score:.3f} "
                f"(minimum {cls.QUALITY_THRESHOLDS['min_r2_score']} required)"
            )

        # Gate 3: Максимальная MSE
        mse = metrics.get("mse", 1.0)
        if mse > cls.QUALITY_THRESHOLDS["max_mse"]:
            reasons.append(
                f"High MSE: {mse:.4f} "
                f"(maximum {cls.QUALITY_THRESHOLDS['max_mse']} allowed)"
            )

        # Gate 4: Train/Test gap (overfitting detection)
        train_r2 = metrics.get("train_r2", r2_score)
        test_r2 = metrics.get("test_r2", r2_score)
        gap = abs(train_r2 - test_r2)

        if gap > cls.QUALITY_THRESHOLDS["max_train_test_gap"]:
            reasons.append(
                f"Overfitting detected: train/test R² gap = {gap:.3f} "
                f"(maximum {cls.QUALITY_THRESHOLDS['max_train_test_gap']} allowed)"
            )

        passed = len(reasons) == 0

        return passed, reasons

    @classmethod
    def format_quality_report(
        cls,
        metrics: Dict[str, float],
        training_samples: int,
        passed: bool,
        reasons: List[str]
    ) -> str:
        """Форматирует читаемый отчет о качестве."""
        report = []
        report.append("=" * 60)
        report.append("🛡 ML TRAINING QUALITY REPORT")
        report.append("=" * 60)

        # Status
        status = "✅ PASSED" if passed else "❌ FAILED"
        report.append(f"\nStatus: {status}")
        report.append(f"Training samples: {training_samples}")

        # Metrics
        report.append("\n📊 Metrics:")
        report.append(f"  R² Score:      {metrics.get('r2', 0.0):.4f}")
        report.append(f"  MSE:           {metrics.get('mse', 0.0):.4f}")
        if 'train_r2' in metrics:
            report.append(f"  Train R²:      {metrics.get('train_r2', 0.0):.4f}")
        if 'test_r2' in metrics:
            report.append(f"  Test R²:       {metrics.get('test_r2', 0.0):.4f}")

        # Thresholds
        report.append("\n🎯 Thresholds:")
        for key, value in cls.QUALITY_THRESHOLDS.items():
            report.append(f"  {key}: {value}")

        # Failure reasons
        if not passed:
            report.append("\n❌ Failure Reasons:")
            for reason in reasons:
                report.append(f"  • {reason}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# =============================================================================
# PER-ACTION CONFIDENCE
# =============================================================================

class PerActionConfidence:
    """
    Разные confidence thresholds для разных action types.

    Логика:
    - routine_task → ML может быть уверен (low threshold)
    - complex_execution → ML менее уверен (high threshold)
    - deep_goal_decomposition → средний threshold
    """

    ACTION_CONFIDENCE_THRESHOLDS = {
        # Простые задачи → ML уверен
        "simple_task": 0.3,
        "routine_task": 0.3,

        # Средняя сложность
        "learning_task": 0.4,
        "creative_task": 0.4,

        # Сложные задачи → ML менее уверен
        "deep_goal_decomposition": 0.5,
        "complex_execution": 0.5,

        # Default (fallback)
        "default": 0.4
    }

    @classmethod
    def get_threshold(cls, action_type: str) -> float:
        """Получить confidence threshold для action."""
        return cls.ACTION_CONFIDENCE_THRESHOLDS.get(
            action_type,
            cls.ACTION_CONFIDENCE_THRESHOLDS["default"]
        )

    @classmethod
    def should_use_ml(
        cls,
        action_type: str,
        ml_confidence: float
    ) -> bool:
        """
        Решить использовать ли ML для этого action.

        Args:
            action_type: Тип действия
            ml_confidence: Confidence от ML модели

        Returns:
            True если ML достаточно уверен для этого action
        """
        threshold = cls.get_threshold(action_type)
        return ml_confidence >= threshold


# =============================================================================
# DRIFT DETECTION
# =============================================================================

class DriftDetector:
    """
    Обнаруживает сдвиг распределения features (drift).

    Логика:
    1. При обучении сохраняем distribution stats (mean, std)
    2. При предсказании сравниваем текущие features с training stats
    3. Если сильно отличается → drift detected → отключаем ML
    """

    DRIFT_THRESHOLD = 3.0  # Количество sigma для детекции drift
    DRIFT_FEATURES_TO_CHECK = [
        "arousal", "valence", "focus", "confidence"  # Current state features
    ]

    def __init__(self):
        self.training_stats: Optional[Dict] = None  # Сохраняется при обучении
        self.drift_history: List[Dict] = []

    def save_training_distribution(self, X_train: np.ndarray):
        """
        Сохраняет статистику training distribution.

        Args:
            X_train: Training features matrix (n_samples, n_features)
        """
        self.training_stats = {
            "mean": np.mean(X_train, axis=0).tolist(),
            "std": np.std(X_train, axis=0).tolist(),
            "min": np.min(X_train, axis=0).tolist(),
            "max": np.max(X_train, axis=0).tolist(),
            "n_features": X_train.shape[1],
            "saved_at": datetime.now(timezone.utc).isoformat()
        }

        print(f"📊 [Drift Detection] Saved training distribution stats "
              f"({X_train.shape[0]} samples, {X_train.shape[1]} features)")

    def detect_drift(
        self,
        features: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Проверяет есть ли drift в features.

        Args:
            features: Текущий feature vector (1D или 2D)
            feature_names: Имена features

        Returns:
            (drift_detected, drift_details)
        """
        if self.training_stats is None:
            return False, ["No training stats available"]

        # Convert to 2D if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        drift_detected = False
        drift_details = []

        training_mean = np.array(self.training_stats["mean"])
        training_std = np.array(self.training_stats["std"])

        # Check each feature
        for i, fname in enumerate(feature_names):
            # Only check important features
            if not any(protected in fname for protected in self.DRIFT_FEATURES_TO_CHECK):
                continue

            current_val = features[0, i]
            train_mean = training_mean[i]
            train_std = training_std[i]

            if train_std < 1e-6:
                continue  # Skip constant features

            # Z-score: насколько далеко от training mean
            z_score = abs(current_val - train_mean) / train_std

            if z_score > self.DRIFT_THRESHOLD:
                drift_detected = True
                drift_details.append(
                    f"{fname}: z={z_score:.2f} (val={current_val:.3f}, "
                    f"train_mean={train_mean:.3f} ± {train_std:.3f})"
                )

        if drift_detected:
            # Log to history
            self.drift_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "drift_details": drift_details,
                "features": features[0].tolist()
            })

            print(f"⚠️  [Drift Detection] Drift detected!")
            for detail in drift_details:
                print(f"    • {detail}")

        return drift_detected, drift_details

    def get_drift_summary(self) -> Dict:
        """Возвращает summary drift history."""
        if not self.drift_history:
            return {
                "total_drifts": 0,
                "recent_drifts": []
            }

        return {
            "total_drifts": len(self.drift_history),
            "recent_drifts": self.drift_history[-10:],  # Last 10
            "training_stats": self.training_stats
        }


# =============================================================================
# FORECAST ERROR TRACKING
# =============================================================================

class ForecastErrorTracker:
    """
    Отслеживает ошибки прогнозирования для:
    - Quality monitoring
    - Confidence calibration
    - Retraining decisions
    """

    def __init__(self):
        self.error_history: List[Dict] = []
        self.max_history = 1000  # Храним последние 1000 ошибок

    def record_forecast(
        self,
        user_id: str,
        action_type: str,
        predicted_deltas: Dict[str, float],
        actual_deltas: Dict[str, float],
        ml_confidence: float,
        used_tier: str  # "ML", "Clusters", "Rules"
    ):
        """
        Записывает результат прогнозирования.

        Args:
            user_id: ID пользователя
            action_type: Тип действия
            predicted_deltas: Предсказанные дельты
            actual_deltas: Фактические дельты
            ml_confidence: Confidence от ML (если использовался)
            used_tier: Какой tier был использован
        """
        # Вычисляем ошибки
        errors = {}
        for dim in ["arousal", "valence", "focus", "confidence"]:
            pred = predicted_deltas.get(dim, 0.0)
            actual = actual_deltas.get(dim, 0.0)
            errors[dim] = {
                "absolute_error": abs(pred - actual),
                "squared_error": (pred - actual) ** 2,
                "direction_correct": (pred > 0) == (actual > 0)  # Правильно ли угадали направление
            }

        # Записываем
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "action_type": action_type,
            "predicted_deltas": predicted_deltas,
            "actual_deltas": actual_deltas,
            "ml_confidence": ml_confidence,
            "used_tier": used_tier,
            "errors": errors
        }

        self.error_history.append(record)

        # Ограничиваем размер
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]

    def get_error_metrics(self, action_type: Optional[str] = None) -> Dict:
        """
        Вычисляет метрики ошибок.

        Args:
            action_type: Фильтр по action type (опционально)

        Returns:
            Словарь с метриками
        """
        # Фильтруем по action_type если указан
        records = self.error_history
        if action_type:
            records = [r for r in records if r["action_type"] == action_type]

        if not records:
            return {
                "total_forecasts": 0,
                "action_type": action_type,
                "mean_absolute_error": None,
                "root_mean_squared_error": None,
                "direction_accuracy": None
            }

        # Вычисляем метрики
        mae_sum = 0.0
        mse_sum = 0.0
        direction_correct = 0
        total_dims = 0

        for record in records:
            for dim_errors in record["errors"].values():
                mae_sum += dim_errors["absolute_error"]
                mse_sum += dim_errors["squared_error"]
                if dim_errors["direction_correct"]:
                    direction_correct += 1
                total_dims += 1

        n = total_dims
        return {
            "total_forecasts": len(records),
            "action_type": action_type or "all",
            "mean_absolute_error": mae_sum / n if n > 0 else None,
            "root_mean_squared_error": (mse_sum / n) ** 0.5 if n > 0 else None,
            "direction_accuracy": direction_correct / n if n > 0 else None
        }

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Решить нужно ли переобучить модель.

        Criteria:
        1. Недостаточно данных → нет
        2. MAE > threshold → да
        3. Direction accuracy < threshold → да

        Returns:
            (should_retrain, reason)
        """
        # Минимум данных для решения
        if len(self.error_history) < 20:
            return False, "Insufficient forecast history"

        metrics = self.get_error_metrics()

        # Пороги
        MAX_MAE = 0.15  # Средняя ошибка не должна быть больше 0.15
        MIN_DIRECTION_ACCURACY = 0.6  # Должны быть правы в направлении в 60% случаев

        mae = metrics.get("mean_absolute_error", 0.0)
        direction_acc = metrics.get("direction_accuracy", 1.0)

        if mae > MAX_MAE:
            return True, f"High MAE: {mae:.4f} > {MAX_MAE}"

        if direction_acc < MIN_DIRECTION_ACCURACY:
            return True, f"Low direction accuracy: {direction_acc:.2%} < {MIN_DIRECTION_ACCURACY:.2%}"

        return False, "Model performing adequately"


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

training_quality_gates = TrainingQualityGates()
per_action_confidence = PerActionConfidence()
drift_detector = DriftDetector()
forecast_error_tracker = ForecastErrorTracker()
