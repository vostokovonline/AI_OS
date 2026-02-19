"""
Centralized Error Handler for AI-OS v3.0

Replaces all bare 'except:' clauses with proper error handling.
Provides consistent error logging and recovery strategies.

Author: AI-OS Core Team
Date: 2026-02-19
"""

import traceback
import asyncio
from functools import wraps
from typing import Any, Callable, TypeVar
from typing import Coroutine

from logging_config import get_logger, log_error

logger = get_logger(__name__)
T = TypeVar('T')


class ErrorHandler:
    """Centralized error handling with proper logging"""

    @staticmethod
    def safe_execute(
        func: Callable[..., T],
        default: T = None,
        context: dict | None = None,
        log_level: str = "ERROR"
    ) -> T:
        """
        Safely execute a function with error handling.

        Replaces: try: ... except: pass

        Usage:
            result = ErrorHandler.safe_execute(
                lambda: risky_operation(),
                default=None,
                context={"goal_id": goal.id}
            )
        """
        try:
            return func()
        except Exception as e:
            log_error(e, context, log_level)
            return default

    @staticmethod
    async def safe_execute_async(
        coro: Coroutine[Any, Any, T],
        default: T = None,
        context: dict | None = None,
        log_level: str = "ERROR"
    ) -> T:
        """
        Safely execute an async function with error handling.

        Usage:
            result = await ErrorHandler.safe_execute_async(
                risky_async_operation(),
                default=None,
                context={"goal_id": goal.id}
            )
        """
        try:
            return await coro
        except Exception as e:
            log_error(e, context, log_level)
            return default


def handle_errors(
    default: Any = None,
    context: dict | None = None,
    log_level: str = "ERROR",
    reraise: bool = False
):
    """
    Decorator for automatic error handling.

    Replaces: try: ... except: pass patterns

    Usage:
        @handle_errors(default=None, context={"operation": "goal_creation"})
        def create_goal(data):
            # Will never crash, errors are logged
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ctx = context or {}
                ctx.update({"function": func.__name__})
                log_error(e, ctx, log_level)
                if reraise:
                    raise
                return default

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                ctx = context or {}
                ctx.update({"function": func.__name__})
                log_error(e, ctx, log_level)
                if reraise:
                    raise
                return default

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open"""
    pass


class CircuitBreaker:
    """
    Circuit breaker pattern for preventing cascade failures.

    Usage:
        breaker = CircuitBreaker(
            func=external_api_call,
            failure_threshold=5,
            timeout=60
        )

        try:
            result = await breaker.call()
        except CircuitBreakerOpen:
            # Service unavailable, use fallback
            result = get_cached_result()
    """

    def __init__(
        self,
        func: Callable,
        failure_threshold: int = 5,
        timeout: int = 60,
        half_open_attempts: int = 1
    ):
        self.func = func
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_attempts = half_open_attempts

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_success_count = 0

    async def call(self, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                self.half_open_success_count = 0
            else:
                raise CircuitBreakerOpen(
                    f"Circuit breaker is open for {self.func.__name__}"
                )

        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(*args, **kwargs)
            else:
                result = self.func(*args, **kwargs)

            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if timeout has passed"""
        if self.last_failure_time is None:
            return True
        import time
        return time.time() - self.last_failure_time >= self.timeout

    def _on_success(self):
        """Handle successful execution"""
        if self.state == "half_open":
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.half_open_attempts:
                self.state = "closed"
                self.failure_count = 0
                logger.info(
                    "circuit_breaker_closed",
                    function=self.func.__name__
                )
        else:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time() if asyncio._get_running_loop() else 0

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "circuit_breaker_opened",
                function=self.func.__name__,
                failure_count=self.failure_count
            )


# Convenience instances for common operations
http_breaker = CircuitBreaker(
    func=None,  # Will be set per request
    failure_threshold=5,
    timeout=30
)

llm_breaker = CircuitBreaker(
    func=None,
    failure_threshold=3,
    timeout=120
)
