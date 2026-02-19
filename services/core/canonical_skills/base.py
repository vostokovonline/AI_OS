"""
SKILL BASE INTERFACE - Canonical Form
Минимальный, но архитектурно правильный интерфейс для всех Skills

Key principle:
Skill = атомарная, проверяемая, воспроизводимая операция
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Artifact:
    """Артефакт, производимый навыком"""
    id: str
    type: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "artifact_type": self.type,
            "content_location": str(self.content) if not isinstance(self.content, dict) else self.content,
            "metadata": self.metadata,
            "created_at": self.created_at
        }


@dataclass
class SkillResult:
    """Результат выполнения навыка"""
    success: bool
    output: Dict[str, Any]
    artifacts: List[Artifact]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "status": "success" if self.success else "failed",
            "artifacts": [a.to_dict() for a in self.artifacts],
            "error": self.error,
            "output": self.output
        }


class Skill(ABC):
    """
    Canonical Skill Interface

    Любой Skill обязан:
    1. Иметь metadata (id, version, description, capabilities)
    2. Реализовать execute() - основную логику
    3. Реализовать verify() - проверку результатов
    4. Возвращать artifacts (обязательно!)
    """

    # ---- REQUIRED METADATA ----
    id: str
    version: str
    description: str

    capabilities: List[str]  # Что умеет делать
    requirements: List[str]  # Что нужно для работы

    input_schema: Dict[str, Any]  # Схема входных данных
    output_schema: Dict[str, Any]  # Схема выходных данных

    produces_artifacts: List[str]  # Какие типы артефактов производит

    # ---- EXECUTION ----
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> SkillResult:
        """
        Выполняет логику навыка

        Requirements:
        - MUST быть детерминированным насколько возможно
        - MUST возвращать artifacts (список может быть пустым, но field обязателен)
        - MUST NOT вызывать Goal Executor
        - MUST NOT ходить в БД напрямую
        - MUST NOT выбирать следующий skill

        Args:
            input_data: Входные данные по input_schema
            context: Контекст выполнения (goal_id, session_id, etc.)

        Returns:
            SkillResult с artifacts
        """
        pass

    # ---- VERIFICATION ----
    @abstractmethod
    def verify(self, result: SkillResult) -> bool:
        """
        Проверяет результаты выполнения

        Args:
            result: SkillResult из execute()

        Returns:
            True если результат валиден, False иначе
        """
        pass

    # ---- HELPERS ----
    def _artifact(
        self,
        type_: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Artifact:
        """Создаёт артефакт"""
        return Artifact(
            id=str(uuid.uuid4()),
            type=type_,
            content=content,
            metadata=metadata or {},
            created_at=datetime.utcnow().isoformat()
        )

    def _error_result(self, message: str) -> SkillResult:
        """Создаёт результат с ошибкой"""
        return SkillResult(
            success=False,
            output={},
            artifacts=[],
            error=message
        )

    def _success_result(
        self,
        output: Dict[str, Any],
        artifacts: List[Artifact]
    ) -> SkillResult:
        """Создаёт успешный результат"""
        return SkillResult(
            success=True,
            output=output,
            artifacts=artifacts,
            error=None
        )
