"""
Pytest Configuration and Fixtures

Minimal test infrastructure for AI-OS stabilization phase.
"""
import asyncio
import os
import sys
import pytest
from typing import AsyncGenerator, Generator

# Add services/core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'core'))


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mock_redis():
    """Mock Redis for unit tests."""
    class MockRedis:
        def __init__(self):
            self._data = {}
        
        async def get(self, key: str):
            return self._data.get(key)
        
        async def set(self, key: str, value: str, ex: int = None):
            self._data[key] = value
            return True
        
        async def delete(self, *keys: str):
            for key in keys:
                self._data.pop(key, None)
            return len(keys)
    
    return MockRedis()


@pytest.fixture(scope="session")
def mock_llm_response():
    """Mock LLM response for tests."""
    return {
        "choices": [{
            "message": {
                "content": '{"next_node": "FINISH", "reasoning": "Test response"}'
            }
        }],
        "model": "test-model"
    }


@pytest.fixture
def sample_goal_data():
    """Sample goal data for tests."""
    return {
        "title": "Test Goal",
        "description": "A test goal for unit testing",
        "goal_type": "achievable",
        "is_atomic": True,
        "depth_level": 0
    }


@pytest.fixture
def sample_artifact_data():
    """Sample artifact data for tests."""
    return {
        "type": "FILE",
        "content_kind": "file",
        "content_location": "test.md"
    }
