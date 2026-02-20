from logging_config import get_logger
logger = get_logger(__name__)

"""
LEARNED EMOTIONAL FORECASTING MODEL
ML-based prediction с rule-based safety net.

Архитектура безопасности:
1. Primary: ML model (sklearn regression/tree)
2. Safety: Rule-based forecasting
3. Fallback: Если ML не уверен или нет данных → rules

Training data: Affective Memory trajectories
Features: Current state + action type + pattern context
Target: Emotional deltas (arousal_Δ, valence_Δ, focus_Δ, confidence_Δ)
"""

import uuid
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select
from database import AsyncSessionLocal
from models import AffectiveMemoryEntry

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.info("⚠️  scikit-learn not available, ML model will be disabled")


# =============================================================================
# Feature Extraction
# =============================================================================

class TrajectoryFeatures:
    """Извлекает features из trajectory для обучения"""

    @staticmethod
    def extract_features(
        current_state: Dict[str, float],
        action_type: str,
        pattern_context: Dict
    ) -> np.ndarray:
        """
        Извлекает feature vector для ML model.

        Features (всего 20+):
        - Current state (4): arousal, valence, focus, confidence
        - Action type (one-hot, 6): deep_goal_decomposition, complex_execution, etc
        - Pattern context (10+): risk_profile, correlations, etc
        """
        features = []

        # 1. Current emotional state (4 features)
        features.extend([
            current_state.get("arousal", 0.5),
            current_state.get("valence", 0.0),
            current_state.get("focus", 0.5),
            current_state.get("confidence", 0.5)
        ])

        # 2. Action type one-hot encoding (6 features)
        action_types = [
            "deep_goal_decomposition",
            "complex_execution",
            "simple_task",
            "creative_task",
            "routine_task",
            "learning_task"
        ]
        for action in action_types:
            features.append(1.0 if action_type == action else 0.0)

        # 3. Pattern context features (10+ features)
        risk_profile = pattern_context.get("risk_profile", {})
        features.extend([
            risk_profile.get("high_arousal_failure_rate", 0.0),
            risk_profile.get("low_focus_failure_rate", 0.0),
            risk_profile.get("low_confidence_failure_rate", 0.0),
        ])

        success_correlations = pattern_context.get("success_correlations", {})
        features.extend([
            success_correlations.get("high_focus_success_rate", 0.5),
            success_correlations.get("positive_valence_success_rate", 0.5),
        ])

        # Dominant patterns (binary features)
        dominant_patterns = pattern_context.get("dominant_patterns", [])
        features.extend([
            1.0 if "success_after_arousal_drop" in dominant_patterns else 0.0,
            1.0 if "failure_when_focus_low" in dominant_patterns else 0.0,
            1.0 if "confidence_builds_on_success" in dominant_patterns else 0.0,
        ])

        return np.array(features)

    @staticmethod
    def extract_target(
        before_state: Dict[str, float],
        after_state: Dict[str, float]
    ) -> np.ndarray:
        """
        Извлекает target vector (emotional deltas).

        Target (4 features):
        - arousal_delta, valence_delta, focus_delta, confidence_delta
        """
        return np.array([
            after_state.get("arousal", 0.5) - before_state.get("arousal", 0.5),
            after_state.get("valence", 0.0) - before_state.get("valence", 0.0),
            after_state.get("focus", 0.5) - before_state.get("focus", 0.5),
            after_state.get("confidence", 0.5) - before_state.get("confidence", 0.5)
        ])


# =============================================================================
# ML Model
# =============================================================================

class EmotionalForecastingModel:
    """
    ML модель для предсказания emotional deltas.

    Использует RandomForestRegressor:
    - Не требует нормализации данных
    - Устойчив к выбросам
    - Предоставляет feature_importance
    """

    MODEL_PATH = Path("/tmp/emotional_forecasting_model.pkl")
    SCALER_PATH = Path("/tmp/emotional_forecasting_scaler.pkl")
    METADATA_PATH = Path("/tmp/emotional_forecasting_metadata.pkl")

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = {
            "trained": False,
            "training_samples": 0,
            "feature_importance": {},
            "metrics": {},
            "trained_at": None
        }

    def is_available(self) -> bool:
        """Проверяет доступность ML модели"""
        return SKLEARN_AVAILABLE and self.metadata["trained"]

    def load(self) -> bool:
        """Загружает модель из диска"""
        try:
            if self.MODEL_PATH.exists():
                with open(self.MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(self.METADATA_PATH, "rb") as f:
                    self.metadata = pickle.load(f)
                logger.info(f"✅ Loaded ML model (trained on {self.metadata['training_samples']} samples)")
                return True
        except Exception as e:
            logger.info(f"⚠️  Could not load ML model: {e}")
        return False

    def save(self):
        """Сохраняет модель на диск"""
        try:
            with open(self.MODEL_PATH, "wb") as f:
                pickle.dump(self.model, f)
            with open(self.SCALER_PATH, "wb") as f:
                pickle.dump(self.scaler, f)
            with open(self.METADATA_PATH, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"✅ Saved ML model to {self.MODEL_PATH}")
        except Exception as e:
            logger.info(f"⚠️  Could not save ML model: {e}")

    async def train(
        self,
        min_samples: int = 20,
        test_size: float = 0.2
    ) -> Dict[str, float]:
        """
        Обучает модель на Affective Memory данных.

        Args:
            min_samples: Минимальное количество samples для обучения
            test_size: Доля test set

        Returns:
            Метрики обучения (MSE, R2)
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not available")

        logger.info("📚 Starting ML model training...")

        # 1. Extract training data from Affective Memory
        X, y = await self._prepare_training_data()

        if len(X) < min_samples:
            raise ValueError(
                f"Insufficient training data: {len(X)} samples "
                f"(minimum {min_samples} required)"
            )

        logger.info(f"📊 Extracted {len(X)} training samples")

        # 2. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # 3. Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 4. Train model (Random Forest)
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)
        logger.info("✅ Model training complete")

        # 5. Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)

        metrics = {
            "mse": float(mse),
            "r2": float(test_r2),
            "train_r2": float(train_r2),
            "test_r2": float(test_r2),
            "training_samples": len(X),
            "test_samples": len(X_test)
        }
        logger.info(f"📈 Metrics: MSE={metrics['mse']:.4f}, R2={metrics['r2']:.4f}")

        # 🆕 QUALITY GATES: Check if model is good enough
        from ml_guardrails import training_quality_gates
        passed, reasons = training_quality_gates.evaluate_training_result(
            metrics, len(X)
        )

        # Print quality report
        report = training_quality_gates.format_quality_report(
            metrics, len(X), passed, reasons
        )
        logger.info(report)

        if not passed:
            # Model failed quality gates → don't mark as available
            self.metadata["trained"] = False
            self.metadata["quality_gate_result"] = "failed"
            self.metadata["quality_gate_reasons"] = reasons

            raise RuntimeError(
                f"Model failed quality gates:\n" + "\n".join(f"  • {r}" for r in reasons)
            )

        # 🆕 DRIFT DETECTION: Save training distribution
        from ml_guardrails import drift_detector
        drift_detector.save_training_distribution(X_train_scaled)

        # 6. Extract feature importance
        feature_names = self._get_feature_names()
        self.metadata["feature_importance"] = dict(zip(
            feature_names,
            self.model.feature_importances_.tolist()
        ))

        # 7. Update metadata
        self.metadata["trained"] = True
        self.metadata["training_samples"] = len(X)
        self.metadata["metrics"] = metrics
        self.metadata["trained_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["quality_gate_result"] = "passed"
        self.metadata["quality_gate_reasons"] = []

        # 8. Save model
        self.save()

        return metrics

    async def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Подготавливает training data из Affective Memory"""
        X_list = []
        y_list = []

        async with AsyncSessionLocal() as db:
            # Get all affective memory entries with after_state
            stmt = select(AffectiveMemoryEntry).where(
                AffectiveMemoryEntry.emotional_state_after.isnot(None)
            ).limit(1000)

            result = await db.execute(stmt)
            entries = result.scalars().all()

            logger.info(f"📊 Found {len(entries)} affective memory entries")

            for entry in entries:
                # Extract features (simplified - no pattern context for historical data)
                before_state = entry.emotional_state_before or {}
                after_state = entry.emotional_state_after or {}

                # Infer action type from goal (simplified)
                action_type = await self._infer_action_type(db, entry.goal_id)

                # Create simplified pattern context (empty for now)
                pattern_context = {
                    "risk_profile": {},
                    "success_correlations": {},
                    "dominant_patterns": []
                }

                # Extract features and target
                try:
                    features = TrajectoryFeatures.extract_features(
                        before_state, action_type, pattern_context
                    )
                    target = TrajectoryFeatures.extract_target(
                        before_state, after_state
                    )

                    X_list.append(features)
                    y_list.append(target)
                except Exception as e:
                    logger.info(f"⚠️  Skipping entry: {e}")
                    continue

        if not X_list:
            raise ValueError("No valid training data found")

        return np.array(X_list), np.array(y_list)

    async def _infer_action_type(self, db, goal_id: uuid.UUID) -> str:
        """Определяет тип действия по цели"""
        try:
            from models import Goal
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return "unknown"

            if goal.is_atomic:
                return "simple_task"
            elif goal.depth_level >= 2:
                return "deep_goal_decomposition"
            else:
                return "complex_execution"
        except Exception as e:
            logger.debug("infer_action_type_failed", goal_id=str(goal_id), error=str(e))
            return "unknown"

    def _get_feature_names(self) -> List[str]:
        """Возвращает имена features"""
        names = []

        # Current state (4)
        names.extend(["arousal", "valence", "focus", "confidence"])

        # Action type (6)
        names.extend([
            "action_decompose", "action_complex", "action_simple",
            "action_creative", "action_routine", "action_learning"
        ])

        # Pattern context (10)
        names.extend([
            "risk_arousal_fail", "risk_focus_fail", "risk_conf_fail",
            "corr_focus_success", "corr_valence_success",
            "pattern_arousal_drop", "pattern_focus_low", "pattern_conf_build"
        ])

        return names

    def predict(
        self,
        current_state: Dict[str, float],
        action_type: str,
        pattern_context: Dict
    ) -> Tuple[Dict[str, float], float]:
        """
        Предсказывает emotional deltas.

        Returns:
            (predicted_deltas, confidence)
        """
        if not self.is_available():
            raise RuntimeError("Model not trained or not available")

        # Extract features
        features = TrajectoryFeatures.extract_features(
            current_state, action_type, pattern_context
        )

        # Scale features
        features_scaled = self.scaler.transform([features])

        # 🆕 DRIFT DETECTION: Check for distribution shift
        from ml_guardrails import drift_detector
        feature_names = self._get_feature_names()
        drift_detected, drift_details = drift_detector.detect_drift(
            features_scaled, feature_names
        )

        if drift_detected:
            # Drift detected → don't use ML
            logger.info("⚠️  [ML Model] Drift detected, disabling ML for this prediction")
            raise RuntimeError(
                f"Drift detected in features: {drift_details}. "
                "ML model disabled for safety."
            )

        # Predict
        deltas = self.model.predict(features_scaled)[0]

        # Calculate confidence (based on tree agreement)
        if hasattr(self.model, 'estimators_'):
            # Individual tree predictions
            tree_preds = np.array([
                tree.predict(features_scaled)[0]
                for tree in self.model.estimators_
            ])

            # Confidence = 1 - std_dev / max_std
            std_dev = np.std(tree_preds, axis=0).mean()
            max_std = 0.5  # heuristic
            confidence = max(0.0, 1.0 - std_dev / max_std)
        else:
            confidence = 0.5  # default

        # 🆕 PER-ACTION CONFIDENCE: Adjust threshold based on action type
        from ml_guardrails import per_action_confidence
        action_threshold = per_action_confidence.get_threshold(action_type)

        # Check if ML is confident enough for this action
        if confidence < action_threshold:
            logger.info(
                f"⚠️  [ML Model] Low confidence ({confidence:.2f}) "
                f"for action '{action_type}' (threshold={action_threshold:.2f})"
            )
            raise RuntimeError(
                f"ML confidence ({confidence:.2f}) below threshold "
                f"for action '{action_type}' ({action_threshold:.2f})"
            )

        predicted_deltas = {
            "arousal": deltas[0],
            "valence": deltas[1],
            "focus": deltas[2],
            "confidence": deltas[3]
        }

        return predicted_deltas, confidence

    def get_feature_importance(self) -> Dict[str, float]:
        """Возвращает важность features"""
        if not self.is_available():
            return {}
        return self.metadata.get("feature_importance", {})

    def get_metadata(self) -> Dict:
        """Возвращает метаданные модели"""
        return self.metadata.copy()


# =============================================================================
# Глобальный экземпляр
# =============================================================================

emotional_forecasting_model = EmotionalForecastingModel()

# Попытка загрузить при импорте
try:
    emotional_forecasting_model.load()
except FileNotFoundError:
    pass  # Model not trained yet, that's ok
except Exception as e:
    logger.debug("model_load_failed_on_import", error=str(e))
    pass  # Model not trained yet, that's ok
