"""
Browser Skills Module

Provides semantic (Vibium) and deterministic (Playwright) browser automation.
"""

from .base import BrowserExecutor, BrowserAction, BrowserResult
from .vibium_executor import VibiumExecutor
from .playwright_executor import PlaywrightExecutor
from .selector import select_browser_executor, ExecutorType

__all__ = [
    "BrowserExecutor",
    "BrowserAction",
    "BrowserResult",
    "VibiumExecutor",
    "PlaywrightExecutor",
    "select_browser_executor",
    "ExecutorType",
]
