"""
Browser Skill Integration with Goal Executor

Показывает как связать Browser Skills с Goal Executor.
"""

import asyncio
from typing import Dict, Any, Optional

from .base import BrowserAction, BrowserResult, BrowserActionType
from .vibium_executor import VibiumExecutor
from .playwright_executor import PlaywrightExecutor
from .selector import select_browser_executor


class BrowserSkillOrchestrator:
    """
    Орхестратор browser skills.

    Задачи:
        1. Выбрать executor (Vibium vs Playwright)
        2. Выполнить действие
        3. Обработать результат
        4. Fallback при ошибках
    """

    def __init__(
        self,
        vibium_config: Optional[Dict] = None,
        playwright_config: Optional[Dict] = None,
        enable_fallback: bool = True
    ):
        """
        Инициализировать orchestrator.

        Args:
            vibium_config: конфиг для Vibium executor
            playwright_config: конфиг для Playwright executor
            enable_fallback: включить fallback (Vibium → Playwright или наоборот)
        """
        self.vibium_config = vibium_config or {}
        self.playwright_config = playwright_config or {}
        self.enable_fallback = enable_fallback

        # Executors (lazy initialization)
        self._vibium = None
        self._playwright = None

        # Failure history (для адаптации)
        self.failure_history: Dict[str, Any] = {
            "vibium": 0,
            "playwright": 0,
            "vibium_avg_cost": 0.0,
            "playwright_avg_cost": 0.0
        }

    async def execute(
        self,
        action: BrowserAction,
        executor_type: Optional[str] = None
    ) -> BrowserResult:
        """
        Выполнить browser action.

        Args:
            action: BrowserAction
            executor_type: (опционально) явно указать executor

        Returns:
            BrowserResult
        """
        # ====================================================================
        # Step 1: Выбрать executor
        # ====================================================================
        if executor_type:
            # Explicit override
            selected = executor_type
        else:
            # Decision matrix
            selected = select_browser_executor(
                action=action,
                failure_history=self.failure_history
            ).value

        # ====================================================================
        # Step 2: Получить executor
        # ====================================================================
        if selected == "vibium":
            executor = self._get_vibium_executor()
        else:
            executor = self._get_playwright_executor()

        # ====================================================================
        # Step 3: Выполнить
        # ====================================================================
        result = await executor.execute(action)

        # ====================================================================
        # Step 4: Fallback (если включён и ошибка)
        # ====================================================================
        if not result.success and self.enable_fallback:
            result = await self._try_fallback(action, selected, result)

        # ====================================================================
        # Step 5: Обновить историю
        # ====================================================================
        self._update_failure_history(selected, result)

        return result

    def _get_vibium_executor(self) -> VibiumExecutor:
        """Lazy инициализация Vibium"""
        if self._vibium is None:
            self._vibium = VibiumExecutor(**self.vibium_config)
        return self._vibium

    def _get_playwright_executor(self) -> PlaywrightExecutor:
        """Lazy инициализация Playwright"""
        if self._playwright is None:
            self._playwright = PlaywrightExecutor(**self.playwright_config)
        return self._playwright

    async def _try_fallback(
        self,
        action: BrowserAction,
        failed_executor: str,
        failed_result: BrowserResult
    ) -> BrowserResult:
        """
        Попробовать fallback executor.

        Логика:
            - Vibium → Playwright (если semantic instruction)
            - Playwright → Vibium (если selector не найден)
        """
        # Не повторяем если это явная ошибка (не UI issue)
        if failed_result.error_type in ["timeout", "unsupported_action"]:
            return failed_result

        # Decide fallback executor
        if failed_executor == "vibium":
            # Vibium → Playwright (только если не semantic)
            if action.type.value == "semantic":
                # Нельзя fallback на Playwright для semantic
                return failed_result

            fallback_executor = self._get_playwright_executor()

        else:  # playwright → vibium
            # Playwright → Vibium (всегда можно)
            fallback_executor = self._get_vibium_executor()

        # Try fallback
        print(f"🔄 Fallback: {failed_executor} → {fallback_executor.executor_type.value}")

        try:
            result = await fallback_executor.execute(action)

            # Add fallback metadata
            result.metadata = {
                "fallback_from": failed_executor,
                "original_error": failed_result.error
            }

            return result

        except Exception as e:
            # Fallback тоже failed
            return failed_result

    def _update_failure_history(self, executor_type: str, result: BrowserResult):
        """Обновить историю фейлов для адаптации"""
        if not result.success:
            self.failure_history[executor_type] += 1

        # TODO: обновить average cost
        # self.failure_history[f"{executor_type}_avg_cost"] = ...

    def close(self):
        """Закрыть все executors"""
        if self._vibium:
            self._vibium.close()
        if self._playwright:
            self._playwright.close()


# ============================================================================
# GOAL EXECUTOR INTEGRATION EXAMPLE
# ============================================================================

class GoalExecutorWithBrowserSkill:
    """
    Пример интеграции Browser Skill с Goal Executor.

    Это показывает, как Goal System использует browser skills.
    """

    def __init__(self):
        self.browser_orchestrator = BrowserSkillOrchestrator(
            vibium_config={
                "llm_provider": "litellm",
                "llm_model": "gpt-4o",
                "headless": False
            },
            playwright_config={
                "headless": True
            },
            enable_fallback=True
        )

    async def execute_goal_with_browser(
        self,
        goal_title: str,
        instruction: str,
        url: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Выполнить goal-задачу с browser action.

        Пример использования:
            executor = GoalExecutorWithBrowserSkill()
            result = await executor.execute_goal_with_browser(
                goal_title="Найти первых-paying клиентов",
                instruction="Войди в аккаунт и открой страницу биллинга",
                url="https://example.com/login",
                context={"auth_complex": True}
            )
        """
        # ====================================================================
        # Step 1: Создать BrowserAction
        # ====================================================================
        action = BrowserAction(
            type=BrowserActionType.SEMANTIC,
            instruction=instruction,
            url=url,
            context=context or {},
            screenshot=True,
            metadata={"goal_title": goal_title}
        )

        # ====================================================================
        # Step 2: Выполнить через orchestrator
        # ====================================================================
        result = await self.browser_orchestrator.execute(action)

        # ====================================================================
        # Step 3: Вернуть результат для Goal System
        # ====================================================================
        return {
            "goal_title": goal_title,
            "success": result.success,
            "executor_used": result.executor_type.value,
            "execution_time_ms": result.execution_time_ms,
            "steps_taken": result.steps_taken,
            "artifacts": {
                "screenshot": result.get_screenshot() is not None,
                "html": len(result.get_artifact_by_type("html").content) if result.get_artifact_by_type("html") else 0,
                "text": len(result.get_text_content() or ""),
            },
            "error": result.error,
            "final_url": result.final_url
        }

    def close(self):
        """Cleanup"""
        self.browser_orchestrator.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """
    Пример использования Browser Skills.
    """
    print("=" * 70)
    print("Browser Skills - Usage Example")
    print("=" * 70)

    orchestrator = BrowserSkillOrchestrator(
        vibium_config={"headless": True},
        playwright_config={"headless": True}
    )

    # ========================================================================
    # Example 1: Semantic instruction (Vibium)
    # ========================================================================
    print("\n📍 Example 1: Semantic instruction")
    print("-" * 70)

    action1 = BrowserAction(
        type=BrowserActionType.SEMANTIC,
        instruction="Найди кнопку входа и авторизуйся",
        url="https://example.com/login",
        context={"auth_complex": True}
    )

    # Decision matrix выберет Vibium
    result1 = await orchestrator.execute(action1)

    print(f"Executor: {result1.executor_type.value}")
    print(f"Success: {result1.success}")
    print(f"Steps taken: {result1.steps_taken}")
    print(f"Artifacts: {len(result1.artifacts)}")

    # ========================================================================
    # Example 2: Deterministic scraping (Playwright)
    # ========================================================================
    print("\n📍 Example 2: Deterministic scraping")
    print("-" * 70)

    action2 = BrowserAction(
        type=BrowserActionType.EXTRACT,
        instruction="Extract price from product page",
        url="https://example.com/product/123",
        context={
            "selector": ".price",
            "deterministic": True
        }
    )

    # Decision matrix выберет Playwright
    result2 = await orchestrator.execute(action2)

    print(f"Executor: {result2.executor_type.value}")
    print(f"Success: {result2.success}")
    print(f"Execution time: {result2.execution_time_ms}ms")

    # ========================================================================
    # Example 3: Goal Executor integration
    # ========================================================================
    print("\n📍 Example 3: Goal Executor integration")
    print("-" * 70)

    goal_executor = GoalExecutorWithBrowserSkill()

    goal_result = await goal_executor.execute_goal_with_browser(
        goal_title="Найти первых-paying клиентов",
        instruction="Войди в аккаунт, открой биллинг и собери данные о платежах",
        url="https://saas.example.com/billing",
        context={
            "auth_complex": True,
            "ui_volatility": "high"
        }
    )

    print(f"Goal: {goal_result['goal_title']}")
    print(f"Success: {goal_result['success']}")
    print(f"Executor: {goal_result['executor_used']}")
    print(f"Time: {goal_result['execution_time_ms']}ms")
    print(f"Artifacts: {goal_result['artifacts']}")

    # Cleanup
    goal_executor.close()
    orchestrator.close()

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    # Запуск примеров
    asyncio.run(example_usage())
