"""
Playwright Executor - Deterministic Browser Automation

Использует Playwright для детерминированных сценариев.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

from .base import (
    BrowserExecutor,
    BrowserAction,
    BrowserResult,
    BrowserArtifact,
    ExecutorType
)


class PlaywrightExecutor(BrowserExecutor):
    """
    Playwright-based browser executor.

    Особенности:
        - Детерминированный
        - Быстрый
        - Требует explicit селекторы
        - Хрупкий к UI changes
    """

    executor_type = ExecutorType.PLAYWRIGHT

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        browser_type: str = "chromium"
    ):
        """
        Инициализировать Playwright executor.

        Args:
            headless: headless mode
            timeout_ms: default timeout
            browser_type: chromium, firefox, webkit
        """
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.browser_type = browser_type

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self._initialized = False

    async def _ensure_browser(self):
        """Ленивая инициализация browser"""
        if self._initialized:
            return

        try:
            from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()

            # Launch browser
            self.browser = await self.playwright[self.browser_type].launch(
                headless=self.headless
            )

            # Create context
            self.context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
            )

            # Create page
            self.page = await self.context.new_page()
            self.page.set_default_timeout(self.timeout_ms / 1000)

            self._initialized = True

        except ImportError:
            raise ImportError(
                "Playwright not installed. Install with: pip install playwright"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Playwright browser: {e}")

    async def execute(self, action: BrowserAction) -> BrowserResult:
        """
        Выполнить browser action через Playwright.

        ВАЖНО: Playwright требует explicit инструкции (селекторы).

        Если action.type == SEMANTIC, вернёт ошибку.
        """
        start_time = time.time()
        started_at = datetime.utcnow()

        try:
            await self._ensure_browser()

            # ====================================================================
            # Step 1: Navigate
            # ====================================================================
            if action.url:
                await self.page.goto(action.url, wait_until="domcontentloaded")

            # ====================================================================
            # Step 2: Execute instruction
            # ====================================================================

            # Playwright НЕ поддерживает семантические инструкции
            if action.type.value == "semantic":
                return BrowserResult(
                    success=False,
                    action=action,
                    artifacts=[],
                    executor_type=self.executor_type,
                    execution_time_ms=int((time.time() - start_time) * 1000),
                    error="Playwright does not support semantic instructions. Use Vibium or provide explicit selectors.",
                    error_type="unsupported_action",
                    started_at=started_at,
                    completed_at=datetime.utcnow()
                )

            # Execute based on action type
            if action.type.value == "navigate":
                # Already navigated above
                pass

            elif action.type.value == "click":
                # Extract selector from instruction/context
                selector = action.context.get("selector")
                if not selector:
                    raise ValueError("Playwright click action requires 'selector' in context")

                await self.page.click(selector)

            elif action.type.value == "fill":
                selector = action.context.get("selector")
                value = action.context.get("value")

                if not selector or value is None:
                    raise ValueError("Playwright fill action requires 'selector' and 'value' in context")

                await self.page.fill(selector, value)

            elif action.type.value == "extract":
                # Extract text from selector
                selector = action.context.get("selector")
                if not selector:
                    raise ValueError("Playwright extract action requires 'selector' in context")

                element = await self.page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                else:
                    text = None

            elif action.type.value == "screenshot":
                # Just screenshot
                pass

            elif action.type.value == "multi_step":
                # Execute sequence of steps
                steps = action.context.get("steps", [])
                for step in steps:
                    step_type = step.get("type")
                    step_selector = step.get("selector")
                    step_value = step.get("value")

                    if step_type == "click":
                        await self.page.click(step_selector)
                    elif step_type == "fill":
                        await self.page.fill(step_selector, step_value)
                    elif step_type == "wait":
                        await self.page.wait_for_selector(step_selector)

            # ====================================================================
            # Step 3: Collect artifacts
            # ====================================================================
            artifacts = []

            # Screenshot
            if action.screenshot:
                screenshot = await self.page.screenshot(full_page=False)
                artifacts.append(BrowserArtifact(
                    type="screenshot",
                    content=screenshot,
                    metadata={
                        "format": "png",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ))

            # HTML content
            html = await self.page.content()
            artifacts.append(BrowserArtifact(
                type="html",
                content=html,
                metadata={"length": len(html)}
            ))

            # Extracted text (если есть)
            if 'text' in locals() and text:
                artifacts.append(BrowserArtifact(
                    type="text",
                    content=text,
                    metadata={}
                ))

            # ====================================================================
            # Step 4: Build result
            # ====================================================================
            execution_time_ms = int((time.time() - start_time) * 1000)

            return BrowserResult(
                success=True,
                action=action,
                artifacts=artifacts,
                executor_type=self.executor_type,
                execution_time_ms=execution_time_ms,
                steps_taken=1,  # Playwright = 1 logical step
                final_url=self.page.url,
                screenshot_taken=action.screenshot,
                started_at=started_at,
                completed_at=datetime.utcnow()
            )

        except Exception as e:
            # Error handling
            execution_time_ms = int((time.time() - start_time) * 1000)

            error_type = "playwright_error"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            elif "selector" in str(e).lower():
                error_type = "selector_not_found"

            return BrowserResult(
                success=False,
                action=action,
                artifacts=[],
                executor_type=self.executor_type,
                execution_time_ms=execution_time_ms,
                error=str(e),
                error_type=error_type,
                started_at=started_at,
                completed_at=datetime.utcnow()
            )

    def close(self):
        """Закрыть browser"""
        if self.page:
            try:
                asyncio.create_task(self.page.close())
            except Exception:
                pass

        if self.context:
            try:
                asyncio.create_task(self.context.close())
            except Exception:
                pass

        if self.browser:
            try:
                asyncio.create_task(self.browser.close())
            except Exception:
                pass

        if self.playwright:
            try:
                asyncio.create_task(self.playwright.stop())
            except Exception:
                pass

        self._initialized = False

    def is_healthy(self) -> bool:
        """Проверить health"""
        return self._initialized and self.page is not None

    # ========================================================================
    # Session management
    # ========================================================================

    async def save_session(self, session_id: str) -> dict:
        """
        Сохранить текущую сессию.
        """
        if not self._initialized:
            raise RuntimeError("Browser not initialized")

        cookies = await self.context.cookies()

        # localStorage
        local_storage = await self.page.evaluate(
            """() => {
            const items = {};
            for (let i = 0; i < window.localStorage.length; i++) {
                const key = window.localStorage.key(i);
                items[key] = window.localStorage.getItem(key);
            }
            return items;
        }"""
        )

        return {
            "session_id": session_id,
            "cookies": cookies,
            "local_storage": local_storage,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def load_session(self, session_id: str, session_data: dict) -> bool:
        """
        Загрузить сохранённую сессию.
        """
        if not self._initialized:
            raise RuntimeError("Browser not initialized")

        # Cookies
        cookies = session_data.get("cookies", [])
        if cookies:
            await self.context.add_cookies(cookies)

        # localStorage
        local_storage = session_data.get("local_storage", {})
        if local_storage:
            await self.page.evaluate(
                """(items) => {
                for (const [key, value] of Object.entries(items)) {
                    window.localStorage.setItem(key, value);
                }
            }""",
                local_storage
            )

        return True

    # ========================================================================
    # Capabilities declaration
    # ========================================================================

    def get_capabilities(self) -> Dict[str, Any]:
        """Объявить capabilities"""
        return {
            "semantic_navigation": False,
            "deterministic": True,
            "ui_robust": False,
            "speed": "fast",
            "cost_per_action": 0.0,
            "avg_latency_ms": 500,
            "supports_auth": True,
            "supports_js": True,
            "supports_session_management": True,
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
