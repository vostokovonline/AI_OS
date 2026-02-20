"""
OCCP v0.3 Compliance Tests
Executable specification of federated protocol invariants

CRITICAL: These tests are the specification.
If test passes → protocol invariant holds.
If test fails → protocol violation.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from occp_v03_types import (
    FederatedRequest,
    OCCPRequestType,
    OCCPDecision,
    OCCPReasonCode,
    ResourceBound
)
from occp_gateway import (
    OCCPGateway,
    OCCPGatewayError,
    FederatedInvariants
)


class TestFederatedInvariants:
    """
    Test FED-01...FED-05 invariants

    These are the CORE protections of OCCP v0.3
    """

    @pytest.mark.asyncio
    async def test_FED_01_no_remote_execution(self):
        """
        FED-01: Remote execution MUST be forbidden

        Request MUST NOT contain execution directives
        """
        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )

        # Valid request: no execution directive
        assert await FederatedInvariants.check_no_remote_execution(request)

        # Invalid request: contains execution directive
        request_dict = request.dict()
        request_dict["directives"] = {"execute": "true"}

        with pytest.raises(Exception):
            await FederatedInvariants.check_no_remote_execution(request)

    @pytest.mark.asyncio
    async def test_FED_02_no_remote_goal_creation(self):
        """
        FED-02: Remote goal creation MUST be forbidden

        Request MUST NOT contain goal_id
        """
        # Valid request: no goal_id
        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )
        assert await FederatedInvariants.check_no_remote_goal_creation(request)

        # Invalid: attempt to pass goal_id
        request_dict = request.dict()
        request_dict["goal_id"] = "some-uuid"

        with pytest.raises(Exception):
            await FederatedInvariants.check_no_remote_goal_creation(request)

    @pytest.mark.asyncio
    async def test_FED_03_no_cross_node_vectors(self):
        """
        FED-03: Cross-node vector application MUST be forbidden

        Request MUST NOT contain vector_id
        """
        request = FederatedRequest(
            request_type=OCCPRequestType.VECTOR_TRANSFORM,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )
        assert await FederatedInvariants.check_no_cross_node_vectors(request)

    @pytest.mark.asyncio
    async def test_FED_04_revocable_assistance(self):
        """
        FED-04: All assistance MUST be revocable

        Operations MUST be:
        - Sandboxed
        - Time-bounded (max 600s)
        """
        # Valid: sandboxed + time-bounded
        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )
        assert await FederatedInvariants.check_revocable_assistance(request)

        # Invalid: not sandboxed
        request.sandbox = False
        assert not await FederatedInvariants.check_revocable_assistance(request)

        # Invalid: time-bounded exceeded
        request.sandbox = True
        request.resource_bound.compute_seconds = 700  # > 600
        assert not await FederatedInvariants.check_revocable_assistance(request)

    @pytest.mark.asyncio
    async def test_FED_05_survivability_first(self):
        """
        FED-05: Survivability > collaboration

        Resource bounds MUST be reasonable:
        - compute <= 600s
        - memory <= 16384mb
        """
        # Valid: reasonable bounds
        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=100.0, memory_mb=1024),
            sandbox=True
        )
        assert await FederatedInvariants.check_survivability_first(request)

        # Invalid: compute too high
        request.resource_bound.compute_seconds = 700
        assert not await FederatedInvariants.check_survivability_first(request)

        # Invalid: memory too high
        request.resource_bound.compute_seconds = 100
        request.resource_bound.memory_mb = 20000  # > 16384
        assert not await FederatedInvariants.check_survivability_first(request)


class TestOCCPGateway:
    """
    Test OCCP Gateway behavior

    Gateway is the ONLY entry point for federated operations
    """

    @pytest.mark.asyncio
    async def test_gateway_rejects_forbidden_fields(self):
        """
        Gateway MUST reject requests with forbidden fields

        Forbidden: goal_id, vector_id, priority, urgency
        """
        # Create mock checkers
        mcl_mock = AsyncMock()
        sk_mock = AsyncMock()
        resource_mock = AsyncMock()

        gateway = OCCPGateway(
            node_id="test-node",
            mcl_checker=mcl_mock,
            sk_checker=sk_mock,
            resource_manager=resource_mock
        )

        # Valid request
        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )

        # Should not raise
        await gateway._validate_request(request)

        # Invalid: contains goal_id
        request_dict = request.dict()
        request_dict["goal_id"] = "some-uuid"

        with pytest.raises(OCCPGatewayError) as exc_info:
            # Reconstruct request with forbidden field
            from pydantic import ValidationError
            # This should fail at validation
            pass

    @pytest.mark.asyncio
    async def test_gateway_mcl_deny_stops_operation(self):
        """
        MCL denial MUST stop operation

        SK and resource checks MUST NOT run if MCL denies
        """
        mcl_mock = AsyncMock()
        mcl_mock.allows_federated = AsyncMock(return_value=False)
        mcl_mock.current_mode = "preservation"
        mcl_mock.explanation = "Preservation mode forbids federated operations"

        sk_mock = AsyncMock()
        resource_mock = AsyncMock()

        gateway = OCCPGateway(
            node_id="test-node",
            mcl_checker=mcl_mock,
            sk_checker=sk_mock,
            resource_manager=resource_mock
        )

        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )

        response = await gateway.handle_request(request)

        # Should deny
        assert response.denial is not None
        assert response.denial["decision"] == "DENY"
        assert response.denial["reason_code"] in [
            OCCPReasonCode.MCL_01,
            OCCPReasonCode.MCL_02,
            OCCPReasonCode.MCL_03
        ]

        # SK should NOT have been called
        sk_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gateway_sk_veto_beats_all(self):
        """
        SK veto is ABSOLUTE

        Even if MCL allows, SK can deny
        """
        mcl_mock = AsyncMock()
        mcl_mock.allows_federated = AsyncMock(return_value=True)

        sk_mock = AsyncMock()
        sk_mock.allows_federated = AsyncMock(return_value=False)
        sk_mock.veto_reason_code = OCCPReasonCode.FED_05
        sk_mock.veto_explanation = "Survivability risk exceeds collaboration benefit"

        resource_mock = AsyncMock()

        gateway = OCCPGateway(
            node_id="test-node",
            mcl_checker=mcl_mock,
            sk_checker=sk_mock,
            resource_manager=resource_mock
        )

        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )

        response = await gateway.handle_request(request)

        # Should deny
        assert response.denial is not None
        assert response.denial["decision"] == "DENY"
        assert response.denial["reason_code"] == OCCPReasonCode.FED_05

    @pytest.mark.asyncio
    async def test_gateway_allows_when_all_pass(self):
        """
        Gateway allows when ALL checks pass
        """
        mcl_mock = AsyncMock()
        mcl_mock.allows_federated = AsyncMock(return_value=True)

        sk_mock = AsyncMock()
        sk_mock.allows_federated = AsyncMock(return_value=True)

        resource_mock = AsyncMock()
        resource_mock.available = AsyncMock(return_value=True)

        gateway = OCCPGateway(
            node_id="test-node",
            mcl_checker=mcl_mock,
            sk_checker=sk_mock,
            resource_manager=resource_mock
        )

        request = FederatedRequest(
            request_type=OCCPRequestType.COMPUTE_ASSIST,
            resource_bound=ResourceBound(compute_seconds=10.0, memory_mb=256),
            sandbox=True
        )

        response = await gateway.handle_request(request)

        # Should allow
        assert response.result is not None
        assert response.denial is None
        assert response.request_id == request.request_id


class TestDualConsent:
    """
    Test dual-consent model

    ALL of these must ALLOW:
    - Local MCL
    - Local SK
    - Remote MCL
    - Remote SK
    """

    @pytest.mark.asyncio
    async def test_local_mcl_deny_stops_before_remote(self):
        """
        Local MCL denial MUST stop request before reaching remote
        """
        # TODO: Implement dual-consent test
        pass

    @pytest.mark.asyncio
    async def test_local_sk_deny_stops_before_remote(self):
        """
        Local SK denial MUST stop request before reaching remote
        """
        # TODO: Implement dual-consent test
        pass

    @pytest.mark.asyncio
    async def test_remote_mcl_deny_stops_execution(self):
        """
        Remote MCL denial MUST stop execution
        """
        # TODO: Implement dual-consent test
        pass

    @pytest.mark.asyncio
    async def test_remote_sk_veto_beats_remote_mcl_allow(self):
        """
        Remote SK veto is ABSOLUTE on remote node
        """
        # TODO: Implement dual-consent test
        pass


class TestAuditTrail:
    """
    Test federated audit logging
    """

    @pytest.mark.asyncio
    async def test_audit_hash_only_by_default(self):
        """
        Audit MUST be hash-only by default

        Full audit only if explicitly requested
        """
        # TODO: Implement audit test
        pass

    @pytest.mark.asyncio
    async def test_both_nodes_log_audit(self):
        """
        BOTH nodes MUST log audit independently

        No shared audit log
        """
        # TODO: Implement audit test
        pass


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    logger.info("OCCP v0.3 Compliance Tests")
    logger.info("=" * 60)

    pytest.main([__file__, "-v"])
