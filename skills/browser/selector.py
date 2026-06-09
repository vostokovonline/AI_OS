"""
Browser Executor Selector - Decision Matrix

Автоматически выбирает Vibium vs Playwright на основе:
- Типа задачи
- Семантических маркеров
- Истории фейлов
- Context hints
"""

from typing import Optional, Dict, Any
from enum import Enum
from .base import BrowserAction, BrowserActionType, ExecutorType


class SelectionReason(Enum):
    """Причина выбора executor"""
    EXPLICIT_REQUEST = "explicit_request"
    SEMANTIC_INSTRUCTION = "semantic_instruction"
    UI_VOLATILITY = "ui_volatility"
    DETERMINISTIC_REQUIRED = "deterministic_required"
    MASS_OPERATIONS = "mass_operations"
    FALLBACK = "fallback"
    AUTH_COMPLEX = "auth_complex"


class SelectorScore:
    """ скор для каждого executor"""
    def __init__(self):
        self.vibium_score = 0.0
        self.playwright_score = 0.0
        self.reasons: list = []

    def prefer_vibium(self, score: float, reason: SelectionReason):
        """Увеличить скор для Vibium"""
        self.vibium_score += score
        self.reasons.append({
            "executor": "vibium",
            "score": score,
            "reason": reason.value
        })

    def prefer_playwright(self, score: float, reason: SelectionReason):
        """Увеличить скор для Playwright"""
        self.playwright_score += score
        self.reasons.append({
            "executor": "playwright",
            "score": score,
            "reason": reason.value
        })

    def winner(self) -> ExecutorType:
        """Вернуть победителя"""
        if self.vibium_score > self.playwright_score:
            return ExecutorType.VIBIUM
        elif self.playwright_score > self.vibium_score:
            return ExecutorType.PLAYWRIGHT
        else:
            # Ничья → дефолт Vibium (более агentic)
            return ExecutorType.VIBIUM


def select_browser_executor(
    action: BrowserAction,
    failure_history: Optional[Dict[str, Any]] = None
) -> ExecutorType:
    """
    Decision matrix для выбора browser executor.

    Args:
        action: BrowserAction с инструкцией и контекстом
        failure_history: (опционально) история фейлов для адаптации

    Returns:
        ExecutorType - какой executor использовать
    """
    score = SelectorScore()
    instruction = action.instruction.lower()
    context = action.context

    # ========================================================================
    # 1. Явное указание (самый высокий приоритет)
    # ========================================================================
    if "executor" in context:
        explicit = context["executor"]
        if explicit == "vibium":
            score.prefer_vibium(100.0, SelectionReason.EXPLICIT_REQUEST)
            return _return_with_debug(score, action)
        elif explicit == "playwright":
            score.prefer_playwright(100.0, SelectionReason.EXPLICIT_REQUEST)
            return _return_with_debug(score, action)

    # ========================================================================
    # 2. Семантические маркеры (признаки "человекоподобной" задачи)
    # ========================================================================
    semantic_markers = [
        # Навигация и поиск
        "найди", "открой", "перейди", "посмотри", "разберись",
        "найти", "открыть", "перейти", "посмотреть",

        # UI действия
        "в интерфейсе", "в панели", "в админке", "на странице",
        "кликни", "нажми", "выбери", "заполни",

        # Сложные сценарии
        "авторизуй", "войд", "залогинься",
        "разобрался", "пойми", "понять",

        # SaaS-specific
        "в saas", "в dashboard", "в настройках",
    ]

    for marker in semantic_markers:
        if marker in instruction:
            score.prefer_vibium(0.8, SelectionReason.SEMANTIC_INSTRUCTION)
            break

    # ========================================================================
    # 3. Тип URL/домена (признаки сложных UI)
    # ========================================================================
    if action.url:
        url = action.url.lower()

        # SaaS платформы
        saas_domains = [
            "app.", "dashboard.", "admin.", "portal.",
            "my.", "accounts.", "billing."
        ]
        if any(domain in url for domain in saas_domains):
            score.prefer_vibium(0.6, SelectionReason.UI_VOLATILITY)

        # Документация (обычно стабильная)
        if "docs." in url or "documentation" in url:
            score.prefer_playwright(0.3, SelectionReason.DETERMINISTIC_REQUIRED)

    # ========================================================================
    # 4. Context hints
    # ========================================================================

    # UI нестабилен
    if context.get("ui_volatility") == "high":
        score.prefer_vibium(0.9, SelectionReason.UI_VOLATILITY)

    # Требуется детерминизм
    if context.get("deterministic") is True:
        score.prefer_playwright(0.8, SelectionReason.DETERMINISTIC_REQUIRED)

    # Массовые операции
    if context.get("bulk") is True or (context.get("repeat") and context.get("repeat") > 1):
        score.prefer_playwright(0.7, SelectionReason.MASS_OPERATIONS)

    # Сложная авторизация
    if context.get("auth_complex") is True:
        score.prefer_vibium(0.5, SelectionReason.AUTH_COMPLEX)

    # ========================================================================
    # 5. История фейлов (MemorySignal integration)
    # ========================================================================
    if failure_history:
        # Если Playwright часто падал → переключиться на Vibium
        playwright_failures = failure_history.get("playwright", 0)
        if playwright_failures >= 3:
            score.prefer_vibium(0.7, SelectionReason.UI_VOLATILITY)

        # Если Vibium слишком дорог → переключиться на Playwright
        vibium_cost = failure_history.get("vibium_avg_cost", 0.0)
        if vibium_cost > 0.50:  # USD
            score.prefer_playwright(0.4, SelectionReason.MASS_OPERATIONS)

    # ========================================================================
    # 6. Дефолт (если ничего не сработало)
    # ========================================================================
    if score.vibium_score == 0 and score.playwright_score == 0:
        # Ничья → Vibium (агентный подход)
        score.prefer_vibium(0.1, SelectionReason.FALLBACK)

    return _return_with_debug(score, action)


def _return_with_debug(score: SelectorScore, action: BrowserAction) -> ExecutorType:
    """
    Вернуть результат с debug информацией.

    В продакшене это пишется в execution trace.
    """
    winner = score.winner()

    # Логируем decision (в продакшене → structured log)
    debug_info = {
        "action_instruction": action.instruction[:100],  # First 100 chars
        "winner": winner.value,
        "scores": {
            "vibium": round(score.vibium_score, 2),
            "playwright": round(score.playwright_score, 2)
        },
        "decision_reasons": score.reasons
    }

    # TODO: в продакшене писать в execution trace
    # action.metadata["selector_debug"] = debug_info

    return winner


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def should_use_vibium(
    instruction: str,
    url: Optional[str] = None,
    context: Optional[Dict] = None
) -> bool:
    """
    Быстрая проверка: стоит ли использовать Vibium?

    Для простых кейсов без полного action object.
    """
    action = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction=instruction,
        url=url,
        context=context or {}
    )

    result = select_browser_executor(action)
    return result == ExecutorType.VIBIUM


def get_executor_capability_matrix() -> Dict[str, Dict[str, Any]]:
    """
    Матрица capabilities для планировщика.

    Планировщик использует это для оценки:
    - Можно ли выполнить задачу
    - Какой executor выбрать
    """
    return {
        "vibium": {
            "semantic_navigation": True,
            "deterministic": False,
            "ui_robust": True,
            "speed": "slow",
            "cost_per_action": 0.01,  # ~1 cent per action (LLM calls)
            "avg_latency_ms": 5000,
            "supports_auth": True,
            "supports_js": True,
            "best_for": [
                "SaaS panels",
                "Dynamic UI",
                "Complex forms",
                "Semantic tasks",
                "Auth flows"
            ],
            "worst_for": [
                "Mass scraping",
                "High-frequency operations",
                "Fixed scenarios"
            ]
        },
        "playwright": {
            "semantic_navigation": False,
            "deterministic": True,
            "ui_robust": False,
            "speed": "fast",
            "cost_per_action": 0.0,
            "avg_latency_ms": 500,
            "supports_auth": True,
            "supports_js": True,
            "best_for": [
                "Mass scraping",
                "Regression testing",
                "Fixed scenarios",
                "High-frequency operations"
            ],
            "worst_for": [
                "Dynamic UI",
                "Semantic tasks",
                "SaaS panels",
                "Complex auth"
            ]
        }
    }
