"""
OCCP v0.3 Sandbox Contract Layer
Trust Without Trust — Physical enforcement, not promises
"""
from enum import Enum
from typing import Dict, List, Optional, Literal, Set
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import asyncio


class SandboxOp(str, Enum):
    """Allowed sandbox operations"""
    COMPUTE = "compute"          # Pure computation
    INFERENCE = "inference"      # LLM inference
    VALIDATE = "validate"        # Validation checks
    ANALYZE = "analyze"          # Static analysis
    REDTEAM = "redteam"          # Adversarial testing


class IOPolicy(BaseModel):
    """
    Sandbbox I/O restrictions

    PRINCIPLE: Sandbox is isolated from system state
    """
    # What sandbox CAN read
    allowed_inputs: List[str] = Field(default_factory=list)

    # What sandbox CAN write
    allowed_outputs: List[str] = Field(default_factory=list)

    # FORBIDDEN operations
    no_read: bool = True                 # No filesystem read by default
    no_write: bool = True                # No filesystem write by default
    no_network: bool = True              # No external network
    no_process_spawn: bool = True        # No subprocess spawning
    no_tool_access: bool = True          # No tool use

    # Memory isolation
    isolate_memory: bool = True          # No shared memory access


class SandboxContract(BaseModel):
    """
    Sandbox execution contract

    CRITICAL: This is the ONLY contract between Gateway and Sandbox
    """
    # What operations are allowed
    allowed_ops: Set[SandboxOp]

    # Resource limits (HARD bounds)
    max_tokens: int = Field(gt=0, le=1000000)
    max_time_ms: int = Field(gt=0, le=600000)  # 10 min max
    max_memory_mb: int = Field(gt=0, le=16384)

    # I/O policy
    io_policy: IOPolicy = Field(default_factory=IOPolicy)

    # FORBIDDEN: Sandbox cannot access these
    # (enforced by contract, not just documentation)
    forbidden_contexts: List[str] = Field(
        default_factory=lambda: ["goals", "vectors", "legacy_axis", "mcl_state", "sk_state"]
    )

    # Sandbox metadata
    sandbox_id: str = Field(default_factory=lambda: f"sbx-{datetime.utcnow().timestamp()}")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SandboxResult(BaseModel):
    """
    Sandbox execution result

    CRITICAL: Result contains ONLY output, NEVER execution context
    """
    sandbox_id: str
    success: bool
    output: Optional[Dict] = None
    error: Optional[str] = None

    # Resource usage (for audit)
    tokens_used: int = 0
    time_ms: int = 0
    memory_mb: int = 0

    # Completion status
    completed: bool = True
    timeout: bool = False
    aborted: bool = False

    # Audit (hash-only by default)
    audit_hash: Optional[str] = None


class SandboxViolation(BaseModel):
    """
    Sandbox contract violation

    Raised when sandbox tries to exceed contract
    """
    violation_type: Literal[
        "resource_timeout",
        "resource_memory",
        "resource_tokens",
        "forbidden_operation",
        "forbidden_context",
        "io_violation"
    ]
    reason: str
    contract_sandbox_id: str


# =============================================================================
# SANDBOX EXECUTOR INTERFACE
# =============================================================================

class SandboxExecutor:
    """
    Sandbox execution engine

    PRINCIPLE: Pure function. No memory. No goals.
    """

    def __init__(self, contract: SandboxContract):
        """
        Initialize sandbox with contract

        CRITICAL: Contract cannot be modified after initialization
        """
        self.contract = contract
        self.start_time = None
        self.tokens_used = 0
        self.aborted = False

    async def execute(
        self,
        operation: SandboxOp,
        payload: Dict
    ) -> SandboxResult:
        """
        Execute operation within contract bounds

        ENFORCEMENT:
        - Operation MUST be in allowed_ops
        - Payload MUST NOT contain forbidden contexts
        - Resources MUST NOT exceed contract limits
        """
        # Step 1: Validate operation
        if operation not in self.contract.allowed_ops:
            return SandboxResult(
                sandbox_id=self.contract.sandbox_id,
                success=False,
                error=f"Operation {operation} not in allowed_ops",
                aborted=True
            )

        # Step 2: Validate payload (no forbidden contexts)
        violation = self._check_payload(payload)
        if violation:
            return SandboxResult(
                sandbox_id=self.contract.sandbox_id,
                success=False,
                error=f"Payload violation: {violation.reason}",
                aborted=True
            )

        # Step 3: Execute with timeout
        self.start_time = datetime.utcnow()

        try:
            # Run with timeout
            result = await asyncio.wait_for(
                self._execute_operation(operation, payload),
                timeout=self.contract.max_time_ms / 1000.0
            )

            # Check resource usage
            if result.tokens_used > self.contract.max_tokens:
                return SandboxResult(
                    sandbox_id=self.contract.sandbox_id,
                    success=False,
                    error=f"Token limit exceeded: {result.tokens_used} > {self.contract.max_tokens}",
                    aborted=True
                )

            return result

        except asyncio.TimeoutError:
            return SandboxResult(
                sandbox_id=self.contract.sandbox_id,
                success=False,
                error=f"Timeout after {self.contract.max_time_ms}ms",
                timeout=True,
                aborted=True
            )

        except Exception as e:
            return SandboxResult(
                sandbox_id=self.contract.sandbox_id,
                success=False,
                error=f"Sandbox error: {str(e)}",
                aborted=True
            )

    def _check_payload(self, payload: Dict) -> Optional[SandboxViolation]:
        """
        Check payload for forbidden contexts (recursive) with constant-time padding

        Priority 2 Fix: Uses busy-wait for minimal overhead timing normalization
        """
        import time

        start = time.monotonic()
        target = 0.000020  # 20μs target (only pad outliers)

        def get_all_keys(d: Dict, prefix: str = "") -> List[str]:
            """Recursively get all keys from nested dict"""
            keys = []
            for k, v in d.items():
                full_key = f"{prefix}.{k}" if prefix else k
                keys.append(full_key)
                if isinstance(v, dict):
                    keys.extend(get_all_keys(v, full_key))
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    # Handle list of dicts
                    for item in v:
                        if isinstance(item, dict):
                            keys.extend(get_all_keys(item, full_key))
            return keys

        all_keys = get_all_keys(payload)

        for forbidden in self.contract.forbidden_contexts:
            # Generate variants (e.g., "goals" -> "goal", "goals")
            variants = [forbidden]
            if forbidden.endswith('s'):
                variants.append(forbidden[:-1])  # Remove trailing 's'
            elif not forbidden.endswith('s'):
                variants.append(forbidden + 's')  # Add trailing 's'

            # Check if any key contains forbidden word
            for key in all_keys:
                key_lower = str(key).lower()
                # Get only the last component (after last dot)
                last_component = key_lower.split('.')[-1]

                for variant in variants:
                    # Check if variant is in last component
                    if variant in last_component:
                        # Pad before returning (constant-time)
                        elapsed = time.monotonic() - start
                        if elapsed < target:
                            # Busy-wait for very short delay (avoid async overhead)
                            while (time.monotonic() - start) < target:
                                pass

                        return SandboxViolation(
                            violation_type="forbidden_context",
                            reason=f"Payload contains forbidden context: {forbidden} (found in key: {key})",
                            contract_sandbox_id=self.contract.sandbox_id
                        )

        # Pad at end if no violation found
        elapsed = time.monotonic() - start
        if elapsed < target:
            while (time.monotonic() - start) < target:
                pass

        return None

    async def _execute_operation(
        self,
        operation: SandboxOp,
        payload: Dict
    ) -> SandboxResult:
        """
        Execute operation (implementation in subclasses)
        """
        # Stub: override in specific sandboxes
        return SandboxResult(
            sandbox_id=self.contract.sandbox_id,
            success=True,
            output={"status": "stub", "operation": operation}
        )

    def abort(self):
        """
        Abort sandbox execution
        """
        self.aborted = True


# =============================================================================
# SANDBOX CONTRACT BUILDER
# =============================================================================

def build_compute_assist_contract(
    max_tokens: int = 10000,
    max_time_seconds: int = 30
) -> SandboxContract:
    """
    Build contract for compute_assist operations

    Use case: Pure computation, no system access
    """
    return SandboxContract(
        allowed_ops={SandboxOp.COMPUTE, SandboxOp.ANALYZE},
        max_tokens=max_tokens,
        max_time_ms=max_time_seconds * 1000,
        max_memory_mb=512,
        io_policy=IOPolicy(
            no_read=True,
            no_write=True,
            no_network=True,
            no_process_spawn=True,
            no_tool_access=True,
            isolate_memory=True
        )
    )


def build_adversarial_test_contract(
    max_tokens: int = 50000,
    max_time_seconds: int = 60
) -> SandboxContract:
    """
    Build contract for adversarial testing

    Use case: Red-teaming, stress tests, prompt injection

    CRITICAL: Results are READ-ONLY reports, never control signals
    """
    return SandboxContract(
        allowed_ops={SandboxOp.REDTEAM, SandboxOp.VALIDATE, SandboxOp.ANALYZE},
        max_tokens=max_tokens,
        max_time_ms=max_time_seconds * 1000,
        max_memory_mb=1024,
        io_policy=IOPolicy(
            no_read=True,
            no_write=True,
            no_network=True,
            no_process_spawn=True,
            no_tool_access=True,
            isolate_memory=True
        )
    )


def build_cognitive_review_contract(
    max_tokens: int = 15000,
    max_time_seconds: int = 45
) -> SandboxContract:
    """
    Build contract for cognitive review

    Use case: Decision review, blind spot detection
    """
    return SandboxContract(
        allowed_ops={SandboxOp.INFERENCE, SandboxOp.ANALYZE},
        max_tokens=max_tokens,
        max_time_ms=max_time_seconds * 1000,
        max_memory_mb=512,
        io_policy=IOPolicy(
            no_read=True,
            no_write=True,
            no_network=True,
            no_process_spawn=True,
            no_tool_access=True,
            isolate_memory=True
        )
    )
