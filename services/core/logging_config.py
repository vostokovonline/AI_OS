"""
Centralized Logging Configuration for AI-OS v3.0

Replaces all logger.info() statements with structured logging.
Uses structlog for production-ready logging.

Author: AI-OS Core Team
Date: 2026-02-19
"""

import logging
import sys
from typing import Any
from datetime import datetime
from pathlib import Path

try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    # Fallback to standard logging
    import logging
    structlog = None


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    json_logs: bool = False
) -> None:
    """
    Configure centralized logging for the entire application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logs
        json_logs: Use JSON format for production (better parsing)
    """
    if STRUCTLOG_AVAILABLE:
        _setup_structlog(level, log_file, json_logs)
    else:
        _setup_standard_logging(level, log_file)


def _setup_structlog(level: str, log_file: str | None, json_logs: bool) -> None:
    """Configure structlog with appropriate processors"""
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        # Production: JSON output
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Development: readable console output
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logging.getLogger().addHandler(file_handler)


def _setup_standard_logging(level: str, log_file: str | None) -> None:
    """Fallback to standard logging when structlog not available"""
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


class StandardLoggerAdapter:
    """
    Wrapper for standard logging.Logger that handles keyword arguments.
    
    Converts logger.info("event", key=value) to:
    logger.info("event | key=value")
    
    This provides compatibility with structlog-style calls when structlog
    is not available.
    """
    
    def __init__(self, logger: logging.Logger):
        self._logger = logger
    
    def _format_kwargs(self, event: str, **kwargs) -> str:
        """Format event and kwargs into a single message string."""
        if not kwargs:
            return event
        
        parts = [event]
        for key, value in kwargs.items():
            # Format value based on type
            if isinstance(value, str):
                formatted_value = value
            elif isinstance(value, (int, float, bool)):
                formatted_value = str(value)
            else:
                formatted_value = repr(value)
            
            parts.append(f"{key}={formatted_value}")
        
        return " | ".join(parts)
    
    def debug(self, event: str, **kwargs) -> None:
        self._logger.debug(self._format_kwargs(event, **kwargs))
    
    def info(self, event: str, **kwargs) -> None:
        self._logger.info(self._format_kwargs(event, **kwargs))
    
    def warning(self, event: str, **kwargs) -> None:
        self._logger.warning(self._format_kwargs(event, **kwargs))
    
    def error(self, event: str, exc_info=None, **kwargs) -> None:
        if exc_info is not None:
            self._logger.error(self._format_kwargs(event, **kwargs), exc_info=exc_info)
        else:
            self._logger.error(self._format_kwargs(event, **kwargs))
    
    def critical(self, event: str, **kwargs) -> None:
        self._logger.critical(self._format_kwargs(event, **kwargs))
    
    # Pass through other logger attributes
    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_logger(name: str | None = None) -> Any:
    """
    Get a logger instance.

    Usage:
        from logging_config import get_logger

        logger = get_logger(__name__)
        logger.info("goal_created", goal_id=goal.id, title=goal.title)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        # Return wrapper that handles keyword args for standard logging
        return StandardLoggerAdapter(logging.getLogger(name))


# Convenience functions for common patterns
def log_goal_transition(
    goal_id: str,
    from_state: str,
    to_state: str,
    actor: str,
    reason: str
) -> None:
    """Log goal state transition with structured data"""
    logger = get_logger("goal_transition")
    logger.info(
        "goal_transition",
        goal_id=goal_id,
        from_state=from_state,
        to_state=to_state,
        actor=actor,
        reason=reason,
        timestamp=datetime.now().isoformat()
    )


def log_error(
    error: Exception,
    context: dict | None = None,
    level: str = "ERROR"
) -> None:
    """Log error with full context and stack trace"""
    logger = get_logger("error_handler")
    log_func = getattr(logger, level.lower(), logger.error)

    log_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if context:
        log_data.update(context)

    log_func("error_occurred", **log_data, exc_info=error)


def http_request_summary(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float
) -> None:
    """Log HTTP request summary"""
    logger = get_logger("http")
    logger.info(
        "http_request",
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms
    )


# Auto-setup on import
setup_logging(
    level="INFO",
    log_file=None,  # Set to "/var/log/ai-os/app.log" in production
    json_logs=False  # Set to True in production
)
