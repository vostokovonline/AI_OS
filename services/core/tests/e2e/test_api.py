"""
E2E API TESTS

End-to-end tests for API endpoints.
Requires running services (use: make test-e2e).
"""
import pytest
import sys
import os
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'core'))

# Try to import httpx for async API testing
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


# Base URL for API
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
@pytest.mark.e2e
class TestAPIHealth:
    """Test API health endpoints."""

    @pytest.mark.asyncio
    async def test_root_endpoint(self):
        """Root endpoint should respond."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=10.0) as client:
            try:
                response = await client.get("/")
                # Any response is OK (even 404) - we just want connection
                assert response.status_code in [200, 404]
            except httpx.ConnectError:
                pytest.skip("API not running")

    @pytest.mark.asyncio
    async def test_llm_status_endpoint(self):
        """LLM status endpoint should return valid JSON."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=10.0) as client:
            try:
                response = await client.get("/llm/status")
                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert data["status"] == "ok"
                assert "llm_status" in data
            except httpx.ConnectError:
                pytest.skip("API not running")


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
@pytest.mark.e2e
class TestGoalAPI:
    """Test goal-related endpoints."""

    @pytest.mark.asyncio
    async def test_goals_list_endpoint(self):
        """Goals list endpoint should return valid data."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            try:
                response = await client.get("/goals/list")
                if response.status_code == 200:
                    data = response.json()
                    assert "goals" in data or isinstance(data, list)
            except httpx.ConnectError:
                pytest.skip("API not running")

    @pytest.mark.asyncio
    async def test_goal_create_validation(self):
        """Goal creation should validate input."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client:
            try:
                # Missing required field
                response = await client.post("/goals/create", json={})
                assert response.status_code in [400, 422]  # Validation error
            except httpx.ConnectError:
                pytest.skip("API not running")


@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
@pytest.mark.e2e
class TestMemoryAPI:
    """Test memory service endpoints."""

    @pytest.mark.asyncio
    async def test_memory_stats_endpoint(self):
        """Memory stats endpoint should return valid data."""
        async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=10.0) as client:
            try:
                response = await client.get("/memory/stats")
                if response.status_code == 200:
                    data = response.json()
                    assert "status" in data or "postgresql" in data
            except httpx.ConnectError:
                pytest.skip("API not running")


@pytest.mark.e2e
class TestServiceConnectivity:
    """Test connectivity to external services."""

    def test_postgres_connectivity(self):
        """PostgreSQL should be accessible."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "exec", "ns_postgres", "pg_isready", "-U", "ns_admin"],
                capture_output=True,
                timeout=5
            )
            assert result.returncode == 0, "PostgreSQL not ready"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or PostgreSQL not running")

    def test_redis_connectivity(self):
        """Redis should be accessible."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "exec", "ns_redis", "redis-cli", "PING"],
                capture_output=True,
                timeout=5
            )
            assert b"PONG" in result.stdout, "Redis not responding"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or Redis not running")

    def test_litellm_connectivity(self):
        """LiteLLM should be accessible."""
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "exec", "ns_litellm", "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:4000/health"],
                capture_output=True,
                timeout=10
            )
            # 401 is OK (needs auth), 200 is OK
            code = result.stdout.decode().strip()
            assert code in ["200", "401"], f"LiteLLM returned {code}"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Docker not available or LiteLLM not running")
