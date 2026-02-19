"""
EVALUATION ENGINE V1
Оценивает выполнение goals и выдает confidence score

Критические принципы:
- Простота: максимум 3 проверки
- Предсказуемость: детерминированный результат
- Объяснимость: понятно ПОЧЕМУ pass/fail
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class EvaluationResult:
    """Результат оценки goal"""
    confidence: float  # 0.0 - 1.0
    passed: bool
    checks: Dict[str, Any]
    summary: str
    evaluation_time: datetime

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "passed": self.passed,
            "checks": self.checks,
            "summary": self.summary,
            "evaluation_time": self.evaluation_time.isoformat()
        }

    def to_display_str(self) -> str:
        """Красивый вывод для UI"""
        status_icon = "✅" if self.passed else "❌"
        confidence_pct = self.confidence * 100

        lines = [
            f"{status_icon} **Evaluation Result** (Confidence: {confidence_pct:.0f}%)",
            "",
            "**Checks:**"
        ]

        for check_name, check_result in self.checks.items():
            check_icon = "✅" if check_result.get("passed") else "❌"
            lines.append(f"  {check_icon} **{check_name}**: {check_result.get('message', 'N/A')}")

        lines.append("")
        lines.append(f"**Summary:** {self.summary}")

        return "\n".join(lines)


class EvaluationEngine:
    """
    Оценивает выполнение goals

    Проверки:
    1. artifact_count - ожидаемое vs фактическое количество
    2. artifact_types - правильные ли типы артефактов
    3. completion_criteria - кастомные критерии из goal
    """

    def __init__(self):
        self.min_confidence_threshold = 0.6  # Ниже - нужен human review

    def evaluate_goal(
        self,
        goal_completion_criteria: Optional[Dict],
        artifacts_produced: List[Dict],
        goal_title: str,
        goal_description: Optional[str] = None
    ) -> EvaluationResult:
        """
        Оценивает выполнение goal

        Args:
            goal_completion_criteria: Критерии выполнения из goal
            artifacts_produced: Список созданных artifacts
            goal_title: Название goal (для объяснений)
            goal_description: Описание goal (для объяснений)

        Returns:
            EvaluationResult с confidence 0.0-1.0
        """
        checks = {}
        total_score = 0.0
        max_score = 0.0

        # 1. Проверка artifact_count
        artifact_check = self._check_artifact_count(
            goal_completion_criteria,
            artifacts_produced
        )
        checks["artifact_count"] = artifact_check
        max_score += 1.0
        if artifact_check["passed"]:
            total_score += 1.0

        # 2. Проверка artifact_types
        types_check = self._check_artifact_types(
            goal_completion_criteria,
            artifacts_produced
        )
        checks["artifact_types"] = types_check
        max_score += 1.0
        if types_check["passed"]:
            total_score += 1.0

        # 3. Проверка completion_criteria (если есть)
        criteria_check = self._check_completion_criteria(
            goal_completion_criteria,
            artifacts_produced
        )
        checks["completion_criteria"] = criteria_check
        max_score += 1.0
        if criteria_check["passed"]:
            total_score += 1.0

        # Вычисляем confidence
        confidence = total_score / max_score if max_score > 0 else 0.0
        passed = confidence >= self.min_confidence_threshold

        # Генерируем summary
        summary = self._generate_summary(checks, confidence, goal_title)

        return EvaluationResult(
            confidence=confidence,
            passed=passed,
            checks=checks,
            summary=summary,
            evaluation_time=datetime.utcnow()
        )

    def _check_artifact_count(
        self,
        completion_criteria: Optional[Dict],
        artifacts_produced: List[Dict]
    ) -> Dict[str, Any]:
        """Проверяет количество артефактов"""
        expected_min = 1  # По умолчанию хотя бы 1

        if completion_criteria:
            artifacts_required = completion_criteria.get("artifacts_required", [])
            expected_min = len(artifacts_required) if artifacts_required else 1

        actual_count = len(artifacts_produced)
        passed = actual_count >= expected_min

        return {
            "passed": passed,
            "expected_min": expected_min,
            "actual": actual_count,
            "message": f"{actual_count}/{expected_min} artifacts produced" if passed else f"Need at least {expected_min} artifacts, got {actual_count}"
        }

    def _check_artifact_types(
        self,
        completion_criteria: Optional[Dict],
        artifacts_produced: List[Dict]
    ) -> Dict[str, Any]:
        """Проверяет типы артефактов"""
        expected_types = set()

        if completion_criteria:
            artifacts_required = completion_criteria.get("artifacts_required", [])
            for req in artifacts_required:
                if isinstance(req, dict):
                    expected_types.add(req.get("type", "FILE"))
                elif isinstance(req, str):
                    expected_types.add(req.upper())

        # Если не указано - ожидаем хотя бы 1 artifact любого типа
        if not expected_types and artifacts_produced:
            return {
                "passed": True,
                "expected_types": "any",
                "actual_types": [a.get("type", "UNKNOWN") for a in artifacts_produced],
                "message": f"Produced: {', '.join(set(a.get('type', 'UNKNOWN') for a in artifacts_produced))}"
            }

        # Проверяем что все ожидаемые типы есть
        actual_types = set(a.get("type", "UNKNOWN") for a in artifacts_produced)
        missing_types = expected_types - actual_types
        passed = len(missing_types) == 0

        return {
            "passed": passed,
            "expected_types": list(expected_types) if expected_types else ["any"],
            "actual_types": list(actual_types),
            "missing_types": list(missing_types),
            "message": f"All required types present: {', '.join(actual_types)}" if passed else f"Missing types: {', '.join(missing_types)}"
        }

    def _check_completion_criteria(
        self,
        completion_criteria: Optional[Dict],
        artifacts_produced: List[Dict]
    ) -> Dict[str, Any]:
        """Проверяет кастомные критерии выполнения"""
        if not completion_criteria:
            return {
                "passed": True,
                "message": "No custom criteria specified"
            }

        # Базовая проверка: artifacts exist
        if not artifacts_produced:
            return {
                "passed": False,
                "message": "No artifacts produced"
            }

        # Проверяем что artifacts имеют content
        has_content = any(
            a.get("content_location") or a.get("content")
            for a in artifacts_produced
        )

        return {
            "passed": has_content,
            "message": "Artifacts have content" if has_content else "Artifacts are empty"
        }

    def _generate_summary(
        self,
        checks: Dict[str, Any],
        confidence: float,
        goal_title: str
    ) -> str:
        """Генерирует текстовое резюме"""
        failed_checks = [name for name, result in checks.items() if not result.get("passed")]

        if not failed_checks:
            return f"✅ Goal '{goal_title}' completed successfully (confidence: {confidence*100:.0f}%)"

        failed_str = ", ".join(failed_checks)
        return f"⚠️ Goal '{goal_title}' has issues: {failed_str} failed (confidence: {confidence*100:.0f}%)"


# Глобальный экземпляр
evaluation_engine = EvaluationEngine()
