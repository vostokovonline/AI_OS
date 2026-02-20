"""
OCCP v0.3 Compute Assist Sandbox
Pure function. No memory. No goals.

MVP Implementation:
- Pure computation on request payload
- NO access to Goals, Vectors, System State
- Resource enforcement (time, tokens, memory)
- Isolated execution context
"""
import asyncio
import hashlib
from typing import Dict, Optional, List
from datetime import datetime
from pydantic import BaseModel

from occp_sandbox import (
    SandboxExecutor,
    SandboxContract,
    SandboxOp,
    SandboxResult,
    SandboxViolation
)


class ComputeAssistRequest(BaseModel):
    """
    Compute assist request payload

    PRINCIPLE: Request is self-contained
    """
    operation_type: str  # "analyze", "validate", "compute"
    input_data: Dict

    # FORBIDDEN fields (contract violation)
    # goal_id: FORBIDDEN
    # vector_id: FORBIDDEN
    # context_graph: FORBIDDEN


class ComputeAssistSandbox(SandboxExecutor):
    """
    Compute Assist Sandbox

    Pure computation sandbox for:
    - Data analysis
    - Validation checks
    - Algorithm execution
    - Pattern matching

    CRITICAL: This sandbox has NO access to:
    - Goals
    - Vectors
    - Legacy Axis
    - MCL state
    - SK state
    """

    def __init__(self, contract: SandboxContract):
        super().__init__(contract)
        self.operation_count = 0
        self.result_cache = {}  # In-memory only, no persistence

    async def _execute_operation(
        self,
        operation: SandboxOp,
        payload: Dict
    ) -> SandboxResult:
        """
        Execute compute assist operation

        OPERATIONS:
        - COMPUTE: Pure computation
        - ANALYZE: Static analysis
        - VALIDATE: Validation checks
        """
        self.operation_count += 1

        try:
            if operation == SandboxOp.COMPUTE:
                return await self._compute(payload)
            elif operation == SandboxOp.ANALYZE:
                return await self._analyze(payload)
            elif operation == SandboxOp.VALIDATE:
                return await self._validate(payload)
            else:
                return SandboxResult(
                    sandbox_id=self.contract.sandbox_id,
                    success=False,
                    error=f"Unsupported operation: {operation}",
                    aborted=True
                )

        except Exception as e:
            return SandboxResult(
                sandbox_id=self.contract.sandbox_id,
                success=False,
                error=f"Execution error: {str(e)}",
                aborted=True
            )

    async def _compute(self, payload: Dict) -> SandboxResult:
        """
        Pure computation operation

        Example operations:
        - Mathematical computation
        - Data transformation
        - Pattern matching
        """
        operation_type = payload.get("operation_type")
        input_data = payload.get("input_data", {})

        # Resource tracking
        tokens_used = self._estimate_tokens(input_data)
        time_used = self._measure_time(lambda: self._do_compute(operation_type, input_data))

        # Check resource limits
        if tokens_used > self.contract.max_tokens:
            return SandboxResult(
                sandbox_id=self.contract.sandbox_id,
                success=False,
                error=f"Token limit exceeded: {tokens_used}",
                tokens_used=tokens_used,
                aborted=True
            )

        # Execute computation
        result = self._do_compute(operation_type, input_data)

        return SandboxResult(
            sandbox_id=self.contract.sandbox_id,
            success=True,
            output={
                "result": result,
                "operation_type": operation_type
            },
            tokens_used=tokens_used,
            time_ms=time_used,
            completed=True
        )

    async def _analyze(self, payload: Dict) -> SandboxResult:
        """
        Static analysis operation

        Example operations:
        - Structure validation
        - Pattern detection
        - Consistency checks
        """
        operation_type = payload.get("operation_type")
        input_data = payload.get("input_data", {})

        tokens_used = self._estimate_tokens(input_data)

        # Static analysis (compute-bound)
        analysis_result = {
            "structure": self._analyze_structure(input_data),
            "patterns": self._detect_patterns(input_data),
            "consistency": self._check_consistency(input_data)
        }

        return SandboxResult(
            sandbox_id=self.contract.sandbox_id,
            success=True,
            output={
                "analysis": analysis_result,
                "operation_type": operation_type
            },
            tokens_used=tokens_used,
            completed=True
        )

    async def _validate(self, payload: Dict) -> SandboxResult:
        """
        Validation operation

        Example validations:
        - Schema validation
        - Constraint checking
        - Rule verification
        """
        operation_type = payload.get("operation_type")
        input_data = payload.get("input_data", {})

        tokens_used = self._estimate_tokens(input_data)

        # Validation checks
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }

        # Example: check for required fields
        required = payload.get("required_fields", [])
        for field in required:
            if field not in input_data:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Missing required field: {field}")

        return SandboxResult(
            sandbox_id=self.contract.sandbox_id,
            success=True,
            output={
                "validation": validation_result,
                "operation_type": operation_type
            },
            tokens_used=tokens_used,
            completed=True
        )

    # ========================================================================
    # UTILITY METHODS (Pure functions, no side effects)
    # ========================================================================

    def _do_compute(self, operation_type: str, input_data: Dict) -> Dict:
        """
        Actual computation (stub for MVP)

        PRINCIPLE: Pure function, same input → same output
        """
        # Stub implementations for MVP
        if operation_type == "sum":
            values = input_data.get("values", [])
            return {"sum": sum(values)}
        elif operation_type == "count":
            items = input_data.get("items", [])
            return {"count": len(items)}
        elif operation_type == "hash":
            data = str(input_data)
            return {"hash": hashlib.sha256(data.encode()).hexdigest()}
        else:
            return {"error": f"Unknown operation: {operation_type}"}

    def _analyze_structure(self, data: Dict) -> Dict:
        """Analyze data structure"""
        return {
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else [],
            "size": len(data)
        }

    def _detect_patterns(self, data: Dict) -> List[str]:
        """Detect patterns in data (stub)"""
        patterns = []

        # Example: detect numeric sequences
        if isinstance(data, dict):
            values = list(data.values())
            if all(isinstance(v, (int, float)) for v in values):
                patterns.append("all_numeric")

        return patterns

    def _check_consistency(self, data: Dict) -> Dict:
        """Check consistency (stub)"""
        return {
            "consistent": True,
            "issues": []
        }

    def _estimate_tokens(self, data: Dict) -> int:
        """
        Estimate token usage

        PRINCIPLE: Overestimate to be safe
        """
        import json

        # Rough estimate: 1 token ≈ 4 characters
        text = json.dumps(data)
        return len(text) // 4

    def _measure_time(self, func) -> int:
        """
        Measure execution time in milliseconds
        """
        start = datetime.utcnow()

        try:
            func()
        except Exception as e:
            logger.debug("timed_function_error", error=str(e))
            pass  # Error handled elsewhere

        end = datetime.utcnow()
        delta = end - start
        return int(delta.total_seconds() * 1000)


# =============================================================================
# SANDBOX FACTORY
# =============================================================================

def create_compute_assist_sandbox(
    max_tokens: int = 10000,
    max_time_seconds: int = 30
) -> ComputeAssistSandbox:
    """
    Create compute assist sandbox with standard contract

    Args:
        max_tokens: Maximum tokens (default: 10K)
        max_time_seconds: Maximum execution time (default: 30s)

    Returns:
        Configured sandbox instance
    """
    from occp_sandbox import build_compute_assist_contract

    contract = build_compute_assist_contract(
        max_tokens=max_tokens,
        max_time_seconds=max_time_seconds
    )

    return ComputeAssistSandbox(contract)


# =============================================================================
# SANDBOX TESTS (Compliance)
# =============================================================================

async def test_sandbox_isolation():
    """
    Test: Sandbox cannot access system state

    COMPLIANCE: FED-01 (no remote execution)
    """
    sandbox = create_compute_assist_sandbox()

    # Try to access forbidden context (should fail)
    malicious_payload = {
        "operation_type": "compute",
        "input_data": {
            "goal_id": "attempt-to-access-goal"
        }
    }

    result = await sandbox.execute(
        SandboxOp.COMPUTE,
        malicious_payload
    )

    assert result.success == False
    assert "forbidden" in result.error.lower() or "violation" in result.error.lower()
    logger.info("✅ Sandbox isolation test passed")


async def test_resource_enforcement():
    """
    Test: Resource limits are enforced

    COMPLIANCE: FED-05 (survivability > collaboration)
    """
    sandbox = create_compute_assist_sandbox(max_tokens=100)

    # Try to exceed token limit
    large_payload = {
        "operation_type": "compute",
        "input_data": {"data": "x" * 1000}  # Will exceed 100 tokens
    }

    result = await sandbox.execute(
        SandboxOp.COMPUTE,
        large_payload
    )

    # Should be denied
    assert result.success == False
    assert "token" in result.error.lower() or "limit" in result.error.lower()
    logger.info("✅ Resource enforcement test passed")


async def test_pure_function():
    """
    Test: Sandbox is pure function

    COMPLIANCE: No side effects
    """
    sandbox = create_compute_assist_sandbox()

    payload = {
        "operation_type": "sum",
        "input_data": {"values": [1, 2, 3, 4, 5]}
    }

    result1 = await sandbox.execute(SandboxOp.COMPUTE, payload)
    result2 = await sandbox.execute(SandboxOp.COMPUTE, payload)

    # Same input → same output
    assert result1.output == result2.output
    logger.info("✅ Pure function test passed")


if __name__ == "__main__":
    logger.info("OCCP v0.3 Compute Assist Sandbox Tests")
    logger.info("=" * 50)

    asyncio.run(test_sandbox_isolation())
    asyncio.run(test_resource_enforcement())
    asyncio.run(test_pure_function())

    logger.info("=" * 50)
    logger.info("✅ All sandbox compliance tests passed")
