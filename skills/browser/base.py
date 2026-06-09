"""
Base Browser Executor Interface

Определяет контракт для всех browser executors (Vibium, Playwright, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ExecutorType(Enum):
    """Тип browser executor"""
    VIBIUM = "vibium"
    PLAYWRIGHT = "playwright"
    AUTO = "auto"


class BrowserActionType(Enum):
    """Тип browser действия"""
    NAVIGATE = "navigate"
    CLICK = "click"
    FILL = "fill"
    EXTRACT = "extract"
    SCREENSHOT = "screenshot"
    SEMANTIC = "semantic"  # Произвольная инструкция на естественном языке
    MULTI_STEP = "multi_step"


@dataclass
class BrowserAction:
    """
    Browser действие с контекстом.

    Это ЕДИНСТВЕННЫЙ способ общения с browser executor.
    """
    type: BrowserActionType
    instruction: str  # Семантическая инструкция на естественном языке

    # Контекст выполнения
    url: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Опции
    timeout_ms: int = 30000
    headless: bool = True
    screenshot: bool = True

    # Auth & session
    session_id: Optional[str] = None
    cookies: Optional[List[Dict]] = None

    # Метаданные для decision matrix
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Валидация после инициализации"""
        if not self.instruction:
            raise ValueError("instruction is required")

    def to_dict(self) -> dict:
        """Сериализация для логирования/трейсинга"""
        return {
            "type": self.type.value,
            "instruction": self.instruction,
            "url": self.url,
            "context_keys": list(self.context.keys()),
            "timeout_ms": self.timeout_ms,
            "has_session": self.session_id is not None,
            "metadata": self.metadata
        }


@dataclass
class BrowserArtifact:
    """Артефакт browser действия"""
    type: str  # screenshot, html, text, json, etc.
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Сериализация"""
        return {
            "type": self.type,
            "content": self.content if not isinstance(self.content, bytes) else f"<{len(self.content)} bytes>",
            "metadata": self.metadata
        }


@dataclass
class BrowserResult:
    """
    Результат выполнения browser действия.

    Это то, что возвращается в GoalExecutor.
    """
    success: bool
    action: BrowserAction

    # Артефакты
    artifacts: List[BrowserArtifact] = field(default_factory=list)

    # Execution metadata
    executor_type: ExecutorType = ExecutorType.AUTO
    execution_time_ms: int = 0
    steps_taken: int = 0  # Сколько шагов предпринял Vibium

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None  # timeout, navigation_failed, etc.
    retries: int = 0

    # State
    final_url: Optional[str] = None
    screenshot_taken: bool = False

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Сериализация для логирования"""
        return {
            "success": self.success,
            "action": self.action.to_dict(),
            "artifacts_count": len(self.artifacts),
            "executor_type": self.executor_type.value,
            "execution_time_ms": self.execution_time_ms,
            "steps_taken": self.steps_taken,
            "error": self.error,
            "error_type": self.error_type,
            "retries": self.retries,
            "final_url": self.final_url,
            "screenshot_taken": self.screenshot_taken
        }

    def get_artifact_by_type(self, artifact_type: str) -> Optional[BrowserArtifact]:
        """Получить артефакт по типу"""
        for artifact in self.artifacts:
            if artifact.type == artifact_type:
                return artifact
        return None

    def get_text_content(self) -> Optional[str]:
        """Получить текстовый контент (если есть)"""
        artifact = self.get_artifact_by_type("text")
        return artifact.content if artifact else None

    def get_screenshot(self) -> Optional[bytes]:
        """Получить скриншот (если есть)"""
        artifact = self.get_artifact_by_type("screenshot")
        return artifact.content if artifact else None


class BrowserExecutor(ABC):
    """
    Base interface для всех browser executors.

    КРИТИЧНО: Executor НЕ знает про Goal System.
    Он только исполняет BrowserAction → BrowserResult.
    """

    executor_type: ExecutorType

    @abstractmethod
    async def execute(self, action: BrowserAction) -> BrowserResult:
        """
        Выполнить browser действие.

        Args:
            action: BrowserAction с инструкцией и контекстом

        Returns:
            BrowserResult с артефактами и метаданными
        """
        pass

    @abstractmethod
    def close(self):
        """Закрыть browser и освободить ресурсы"""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Проверить health executor"""
        pass

    # Optional: session management
    async def save_session(self, session_id: str) -> dict:
        """Сохранить текущую сессию (cookies, localStorage)"""
        raise NotImplementedError(f"save_session not implemented for {self.executor_type}")

    async def load_session(self, session_id: str) -> bool:
        """Загрузить сохранённую сессию"""
        raise NotImplementedError(f"load_session not implemented for {self.executor_type}")

    # Optional: capabilities declaration
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Объявить capabilities этого executor.

        Используется decision matrix для выбора.
        """
        return {
            "semantic_navigation": False,  # Понимает ли NL инструкции
            "deterministic": True,         # Детерминированный ли результат
            "ui_robust": True,            # Устойчив к UI changes
            "speed": "medium",            # fast, medium, slow
            "cost_per_action": 0.0,       # Стоимость (USD)
            "supports_auth": True,
            "supports_js": True,
        }
