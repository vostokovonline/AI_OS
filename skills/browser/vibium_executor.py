"""
Vibium Executor - Semantic Browser Automation

Использует Vibium (AI-driven browser) для выполнения семантических инструкций.
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


class VibiumExecutor(BrowserExecutor):
    """
    Vibium-based browser executor.

    Особенности:
        - Понимает семантические инструкции
        - Агентный подход (LLM решает КАК выполнить)
        - Устойчив к UI changes
        - Медленнее и дороже Playwright
    """

    executor_type = ExecutorType.VIBIUM

    def __init__(
        self,
        llm_provider: str = "litellm",
        llm_model: str = "gpt-4o",
        headless: bool = False,
        timeout_ms: int = 30000,
        cdp_url: Optional[str] = None  # For remote browser
    ):
        """
        Инициализировать Vibium executor.

        Args:
            llm_provider: litellm, openai, anthropic, etc.
            llm_model: model name
            headless: headless mode
            timeout_ms: default timeout
            cdp_url: (опционально) CDP URL для remote browser
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.cdp_url = cdp_url

        self.browser = None
        self._initialized = False

    async def _ensure_browser(self):
        """Ленивая инициализация browser"""
        if self._initialized:
            return

        try:
            # Import vibium (optional dependency)
            from vibium import Browser

            # Init browser
            if self.cdp_url:
                # Remote browser (CDP)
                self.browser = Browser(
                    llm_provider=self.llm_provider,
                    llm_model=self.llm_model,
                    cdp_url=self.cdp_url,
                    headless=self.headless
                )
            else:
                # Local browser
                self.browser = Browser(
                    llm_provider=self.llm_provider,
                    llm_model=self.llm_model,
                    headless=self.headless
                )

            self._initialized = True

        except ImportError:
            raise ImportError(
                "Vibium not installed. Install with: pip install vibium"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vibium browser: {e}")

    async def execute(self, action: BrowserAction) -> BrowserResult:
        """
        Выполнить browser action через Vibium.

        Args:
            action: BrowserAction с семантической инструкцией

        Returns:
            BrowserResult с артефактами
        """
        start_time = time.time()
        started_at = datetime.utcnow()

        try:
            await self._ensure_browser()

            # ====================================================================
            # Step 1: Navigate (если указан URL)
            # ====================================================================
            if action.url:
                self.browser.go(action.url)

            # ====================================================================
            # Step 2: Execute instruction (семантическое действие)
            # ====================================================================
            result = self.browser.instruct(
                action.instruction,
                timeout=action.timeout_ms / 1000  # Convert to seconds
            )

            steps_taken = result.get("steps", 0)

            # ====================================================================
            # Step 3: Collect artifacts
            # ====================================================================
            artifacts = []

            # Screenshot (если запрошено)
            if action.screenshot:
                screenshot = self.browser.screenshot()
                artifacts.append(BrowserArtifact(
                    type="screenshot",
                    content=screenshot,
                    metadata={
                        "format": "png",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                ))

            # HTML content
            html = self.browser.html()
            artifacts.append(BrowserArtifact(
                type="html",
                content=html,
                metadata={"length": len(html)}
            ))

            # Extracted text (если Vibium вернул)
            if "text" in result:
                artifacts.append(BrowserArtifact(
                    type="text",
                    content=result["text"],
                    metadata={}
                ))

            # Extracted data (если есть)
            if "data" in result:
                artifacts.append(BrowserArtifact(
                    type="json",
                    content=result["data"],
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
                steps_taken=steps_taken,
                final_url=self.browser.current_url(),
                screenshot_taken=action.screenshot,
                started_at=started_at,
                completed_at=datetime.utcnow()
            )

        except Exception as e:
            # Error handling
            execution_time_ms = int((time.time() - start_time) * 1000)

            error_type = "vibium_error"
            if "timeout" in str(e).lower():
                error_type = "timeout"
            elif "navigation" in str(e).lower():
                error_type = "navigation_failed"

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
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass  # Ignore errors on close

        self._initialized = False

    def is_healthy(self) -> bool:
        """Проверить health"""
        return self._initialized and self.browser is not None

    # ========================================================================
    # Session management (опционально)
    # ========================================================================

    async def save_session(self, session_id: str) -> dict:
        """
        Сохранить текущую сессию.

        Returns:
            dict с cookies и localStorage
        """
        if not self._initialized:
            raise RuntimeError("Browser not initialized")

        cookies = self.browser.cookies()

        # TODO: Vibium API for localStorage?
        # 目前 Vibium 可能没有直接的 localStorage API
        # Это placeholder для будущей реализации

        return {
            "session_id": session_id,
            "cookies": cookies,
            "timestamp": datetime.utcnow().isoformat()
        }

    async def load_session(self, session_id: str, session_data: dict) -> bool:
        """
        Загрузить сохранённую сессию.
        """
        if not self._initialized:
            raise RuntimeError("Browser not initialized")

        cookies = session_data.get("cookies", [])
        if cookies:
            self.browser.set_cookies(cookies)

        return True

    # ========================================================================
    # Capabilities declaration
    # ========================================================================

    def get_capabilities(self) -> Dict[str, Any]:
        """Объявить capabilities"""
        return {
            "semantic_navigation": True,
            "deterministic": False,
            "ui_robust": True,
            "speed": "slow",
            "cost_per_action": 0.01,  # ~1 cent per action (LLM calls)
            "avg_latency_ms": 5000,
            "supports_auth": True,
            "supports_js": True,
            "supports_session_management": True,
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
        }


# ============================================================================
# SYNC WRAPPER (для совместимости с sync кодом)
# ============================================================================

class VibiumExecutorSync:
    """
    Sync wrapper для VibiumExecutor.

    Удобно использовать в sync контексте.
    """

    def __init__(self, **kwargs):
        self._async_executor = VibiumExecutor(**kwargs)
        self._loop = None

    def execute_sync(self, action: BrowserAction) -> BrowserResult:
        """Sync версия execute"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._async_executor.execute(action))

    def close(self):
        """Закрыть browser"""
        if self._async_executor:
            self._async_executor.close()

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()
