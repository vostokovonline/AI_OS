"""
OCCP v0.3 Gateway
Federated Cognitive Control Protocol - Entry Point

PRINCIPLE: Gateway has NO access to:
- Goals
- Vectors
- Execution

Gateway ONLY:
1. Validates request structure
2. Runs local control loop (MCL + SK)
3. Executes sandboxed operation
4. Returns result or denial
"""
import asyncio
import time
from typing import Dict, Optional, Tuple
from occp_v03_types import (
    FederatedRequest,
    FederatedResponse,
    FederatedDenial,
    OCCPDecision,
    OCCPReasonCode,
    DisclosureLevel,
    ConsentRecord,
    FederatedAuditEvent,
    OCCPDecisionSchema
)


# =============================================================================
# PRIORITY 4: TOKEN BUCKET RATE LIMITER
# =============================================================================

class TokenBucket:
    """
    Token bucket rate limiter

    PRINCIPLE: O(1) throttling with predictable latency

    Throttle ≠ Security decision:
    - Does NOT go to SK
    - Does NOT affect forbidden detection
    - Only affects availability, not semantic outcome
    """

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Max burst size (tokens)
            refill_rate: Tokens per second refilled
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def allow(self) -> bool:
        """
        Check if request is allowed (consumes 1 token if allowed)

        Returns:
            True if request allowed (token consumed)
            False if rate limited
        """
        now = time.monotonic()
        delta = now - self.last_refill

        # Refill tokens based on elapsed time
        self.tokens = min(
            self.capacity,
            self.tokens + delta * self.refill_rate
        )
        self.last_refill = now

        # Check if token available
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def get_tokens(self) -> float:
        """Get current token count (for metrics)"""
        now = time.monotonic()
        delta = now - self.last_refill
        return min(
            self.capacity,
            self.tokens + delta * self.refill_rate
        )



class OCCPGatewayError(Exception):
    """Raised when Gateway invariant is violated"""
    pass


class OCCPGateway:
    """
    OCCP Gateway - Federated Entry Point

    INVARIANTS:
    - NO direct access to Goals
    - NO direct access to Vectors
    - NO direct access to Execution
    - All operations MUST pass local MCL + SK
    - All denials MUST have reason_code
    """

    def __init__(self, node_id: str, mcl_checker, sk_checker, resource_manager):
        """
        Args:
            node_id: This node's ID
            mcl_checker: Local MCL permission checker
            sk_checker: Local SK veto checker
            resource_manager: Local resource availability checker
        """
        self.node_id = node_id
        self.mcl_checker = mcl_checker
        self.sk_checker = sk_checker
        self.resource_manager = resource_manager

        # Priority 1 Fix: Concurrent SK Veto Lock
        self._sk_lock = asyncio.Lock()

        # Priority 3 Fix: Audit log file path
        import tempfile
        self.audit_file = tempfile.gettempdir() + "/occp_federated_audit.log"

        # Priority 4 Fix: Rate limiter storage
        # Key: (node_id, request_type) -> TokenBucket
        self._rate_limits: Dict[Tuple[str, str], TokenBucket] = {}
        self._rate_limits_lock = asyncio.Lock()

    async def handle_request(self, request: FederatedRequest) -> FederatedResponse:
        """
        Handle incoming federated request

        CRITICAL: This is the ONLY entry point for remote operations
        """
        # Priority 4: Check rate limit BEFORE any processing
        # Throttle ≠ Security decision (does NOT go to SK)
        if not await self._check_rate_limit(request):
            return self._deny(request, OCCPDecisionSchema(
                layer="Resource",
                decision=OCCPDecision.DENY,
                reason_code=OCCPReasonCode.RESOURCE_01,
                explanation="Rate limit exceeded",
                confidence=1.0
            ))

        # Step 1: Validate request structure
        await self._validate_request(request)

        # Step 2: Local MCL check
        mcl_decision = await self._mcl_check(request)
        if mcl_decision.decision == OCCPDecision.DENY:
            return self._deny(request, mcl_decision)

        # Step 3: Local SK check
        sk_decision = await self._sk_check(request)
        if sk_decision.decision == OCCPDecision.DENY:
            return self._deny(request, sk_decision)

        # Step 4: Resource availability
        resource_decision = await self._resource_check(request)
        if resource_decision.decision == OCCPDecision.DENY:
            return self._deny(request, resource_decision)

        # Step 5: Execute sandboxed
        result = await self._sandbox_execute(request)

        # Step 6: Audit
        await self._audit_log(request, [mcl_decision, sk_decision, resource_decision], result)

        # Step 7: Return response
        return FederatedResponse(
            request_id=request.request_id,
            node_id=self.node_id,
            result=result,
            audit_hash=self._compute_audit_hash(request, result)
        )

    async def _validate_request(self, request: FederatedRequest):
        """
        Validate that request contains NO forbidden fields

        FORBIDDEN fields (signal protocol violation):
        - goal_id
        - vector_id
        - priority
        - urgency
        """
        # Check dict representation for forbidden keys
        request_dict = request.dict()

        forbidden_keys = {"goal_id", "vector_id", "priority", "urgency"}
        found_forbidden = forbidden_keys & request_dict.keys()

        if found_forbidden:
            raise OCCPGatewayError(
                f"Protocol violation: request contains forbidden keys: {found_forbidden}. "
                f"Remote nodes cannot specify goals, vectors, or priorities."
            )

    async def _mcl_check(self, request: FederatedRequest) -> OCCPDecisionSchema:
        """
        MCL Gate: Check if local cognitive mode allows this operation
        """
        try:
            allowed = await self.mcl_checker.allows_federated(request)

            if allowed:
                return OCCPDecisionSchema(
                    layer="MCL",
                    decision=OCCPDecision.ALLOW,
                    reason_code=None,
                    explanation=f"MCL mode {self.mcl_checker.current_mode} allows operation",
                    confidence=1.0
                )
            else:
                return OCCPDecisionSchema(
                    layer="MCL",
                    decision=OCCPDecision.DENY,
                    reason_code=OCCPReasonCode.MCL_01,
                    explanation=self.mcl_checker.explanation,
                    confidence=1.0
                )

        except Exception as e:
            # Fail closed: if MCL check fails, deny
            return OCCPDecisionSchema(
                layer="MCL",
                decision=OCCPDecision.DENY,
                reason_code=OCCPReasonCode.MCL_01,
                explanation=f"MCL check failed: {str(e)}",
                confidence=1.0
            )

    async def _sk_check(self, request: FederatedRequest) -> OCCPDecisionSchema:
        """
        SK Gate: ABSOLUTE VETO over federated operations

        CRITICAL: SK decision is FINAL. No overrides.

        Priority 1 Fix: Serialized SK check with lock to prevent race conditions.
        """
        async with self._sk_lock:
            try:
                allowed = await self.sk_checker.allows_federated(request)

                if allowed:
                    return OCCPDecisionSchema(
                        layer="SK",
                        decision=OCCPDecision.ALLOW,
                        reason_code=None,
                        explanation="SK allows federated operation",
                        confidence=1.0
                    )
                else:
                    return OCCPDecisionSchema(
                        layer="SK",
                        decision=OCCPDecision.DENY,
                        reason_code=self.sk_checker.veto_reason_code,
                        explanation=self.sk_checker.veto_explanation,
                        confidence=1.0
                    )

            except Exception as e:
                # Fail closed: if SK check fails, deny
                return OCCPDecisionSchema(
                    layer="SK",
                    decision=OCCPDecision.DENY,
                    reason_code=OCCPReasonCode.FED_05,
                    explanation=f"SK check failed: {str(e)}",
                    confidence=1.0
                )

    async def _resource_check(self, request: FederatedRequest) -> OCCPDecisionSchema:
        """
        Resource Gate: Check if local resources available
        """
        try:
            available = await self.resource_manager.available(request.resource_bound)

            if available:
                return OCCPDecisionSchema(
                    layer="Resource",
                    decision=OCCPDecision.ALLOW,
                    reason_code=None,
                    explanation="Resources available",
                    confidence=1.0
                )
            else:
                return OCCPDecisionSchema(
                    layer="Resource",
                    decision=OCCPDecision.DENY,
                    reason_code=OCCPReasonCode.RESOURCE_01,
                    explanation="Insufficient resources",
                    confidence=1.0
                )

        except Exception as e:
            return OCCPDecisionSchema(
                layer="Resource",
                decision=OCCPDecision.DENY,
                reason_code=OCCPReasonCode.RESOURCE_01,
                explanation=f"Resource check failed: {str(e)}",
                confidence=1.0
            )

    async def _check_rate_limit(self, request: FederatedRequest) -> bool:
        """
        Check if request is within rate limits

        Priority 4: Token bucket throttling
        - Key: (node_id, request_type)
        - Does NOT go to SK (availability decision, not security)

        Returns:
            True if within rate limit, False if throttled
        """
        limiter = await self._get_rate_limiter(request)
        return limiter.allow()

    async def _get_rate_limiter(self, request: FederatedRequest) -> TokenBucket:
        """
        Get or create rate limiter for (source_node, request_type) key

        Priority 4: Safe defaults for federated requests
        - Capacity: 200 (max burst)
        - Refill rate: 100 tokens/sec (sustained rate)
        """
        key = (request.source_node, str(request.request_type))

        async with self._rate_limits_lock:
            if key not in self._rate_limits:
                # Create new limiter with safe defaults
                self._rate_limits[key] = TokenBucket(
                    capacity=200,      # Max burst
                    refill_rate=100.0  # Tokens per second
                )
            return self._rate_limits[key]

    async def _sandbox_execute(self, request: FederatedRequest) -> Dict:
        """
        Execute operation in sandbox

        CRITICAL: This is the ONLY place where federated code runs
        Gateway delegates to appropriate sandbox based on request_type
        """
        from occp_v03_types import OCCPRequestType
        from occp_compute_sandbox import create_compute_assist_sandbox
        from occp_adversarial_sandbox import create_adversarial_test_sandbox
        from occp_sandbox import SandboxOp

        # Route to appropriate sandbox
        if request.request_type == OCCPRequestType.COMPUTE_ASSIST:
            sandbox = create_compute_assist_sandbox(
                max_tokens=request.resource_bound.compute_seconds * 100,  # Rough estimate
                max_time_seconds=int(request.resource_bound.compute_seconds)
            )

            # Execute compute operation
            result = await sandbox.execute(
                SandboxOp.COMPUTE,
                {
                    "operation_type": "compute",
                    "input_data": request.dict()
                }
            )

        elif request.request_type == OCCPRequestType.ADVERSARIAL_TEST:
            sandbox = create_adversarial_test_sandbox(
                max_tokens=request.resource_bound.compute_seconds * 100,
                max_time_seconds=int(request.resource_bound.compute_seconds)
            )

            # Execute adversarial test
            result = await sandbox.execute(
                SandboxOp.REDTEAM,
                {
                    "target_type": "generic",
                    "test_type": "injection",
                    "payload": request.dict()
                }
            )

        elif request.request_type == OCCPRequestType.COGNITIVE_REVIEW:
            # Use compute sandbox for cognitive review
            sandbox = create_compute_assist_sandbox(
                max_tokens=request.resource_bound.compute_seconds * 100,
                max_time_seconds=int(request.resource_bound.compute_seconds)
            )

            result = await sandbox.execute(
                SandboxOp.ANALYZE,
                {
                    "operation_type": "analyze",
                    "input_data": request.dict()
                }
            )

        else:
            # Fallback for other request types
            sandbox = create_compute_assist_sandbox(
                max_tokens=10000,
                max_time_seconds=30
            )

            result = await sandbox.execute(
                SandboxOp.COMPUTE,
                {
                    "operation_type": "compute",
                    "input_data": {"request_type": request.request_type}
                }
            )

        # Convert SandboxResult to Dict
        if result.success:
            return {
                "status": "success",
                "sandbox_id": result.sandbox_id,
                "output": result.output,
                "tokens_used": result.tokens_used,
                "time_ms": result.time_ms,
                "completed": result.completed
            }
        else:
            return {
                "status": "failed",
                "sandbox_id": result.sandbox_id,
                "error": result.error,
                "aborted": result.aborted,
                "timeout": result.timeout
            }

    def _deny(self, request: FederatedRequest, decision: OCCPDecisionSchema) -> FederatedResponse:
        """
        Return federated denial
        """
        return FederatedResponse(
            request_id=request.request_id,
            node_id=self.node_id,
            denial=FederatedDenial(
                decision="DENY",
                reason_code=decision.reason_code or OCCPReasonCode.FED_01,
                explanation=decision.explanation,
                disclosure_level=DisclosureLevel.MINIMAL,
                node_id=self.node_id
            )
        )

    async def _audit_log(self, request, decisions, result):
        """
        Write to federated audit log with atomic append

        Priority 3 Fix: File locking prevents concurrent write corruption
        """
        import fcntl
        import json
        from datetime import datetime, timezone

        entry = self._create_audit_entry(request, decisions, result)

        try:
            with open(self.audit_file, 'a') as f:
                # Exclusive lock for atomic append
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry) + '\n')
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            # Log failure but don't fail the request
            print(f"Warning: Failed to write audit log: {e}")

    def _create_audit_entry(self, request, decisions, result):
        """
        Create audit entry (hash-only, minimal disclosure)
        """
        from datetime import datetime, timezone

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_id": self.node_id,
            "request_id": request.request_id,
            "request_type": str(request.request_type),
            "decisions": [
                {
                    "layer": d.layer,
                    "decision": str(d.decision),
                    "reason_code": str(d.reason_code) if d.reason_code else None
                }
                for d in decisions
            ],
            "result_hash": self._compute_audit_hash(request, result),
            "sandbox_id": result.get("sandbox_id") if isinstance(result, dict) else None
        }

    def _compute_audit_hash(self, request: FederatedRequest, result: Dict) -> str:
        """
        Compute hash for audit trail (hash-only by default)
        """
        import hashlib
        import json

        # Hash request + result (minimal disclosure)
        data = {
            "request_id": request.request_id,
            "request_type": request.request_type,
            "result_keys": list(result.keys()) if result else []
        }

        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# =============================================================================
# FEDERATION INVARIANTS (Machine-readable)
# =============================================================================

class FederatedInvariants:
    """
    FED-01...FED-05 as executable checks
    """

    @staticmethod
    async def check_no_remote_execution(request: FederatedRequest) -> bool:
        """
        FED-01: No remote execution

        Request MUST NOT contain execution directives
        """
        request_dict = request.dict()
        return "execute" not in request_dict.get("directives", {})

    @staticmethod
    async def check_no_remote_goal_creation(request: FederatedRequest) -> bool:
        """
        FED-02: No remote goal creation

        Request MUST NOT contain goal_id
        """
        return "goal_id" not in request.dict()

    @staticmethod
    async def check_no_cross_node_vectors(request: FederatedRequest) -> bool:
        """
        FED-03: No cross-node vector application

        Request MUST NOT contain vector_id
        """
        return "vector_id" not in request.dict()

    @staticmethod
    async def check_revocable_assistance(request: FederatedRequest) -> bool:
        """
        FED-04: All assistance is revocable

        All operations MUST be sandboxed and time-bounded
        """
        return request.sandbox and request.resource_bound.compute_seconds <= 600

    @staticmethod
    async def check_survivability_first(request: FederatedRequest) -> bool:
        """
        FED-05: Survivability > collaboration

        Resource bounds MUST be reasonable
        """
        return (
            request.resource_bound.compute_seconds <= 600 and
            request.resource_bound.memory_mb <= 16384
        )
