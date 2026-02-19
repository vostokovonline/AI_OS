"""
OCCP v0.3 Federation Protocol
Multi-Instance coordination without shared goals
"""
from typing import Dict, List, Optional
from occp_v03_types import (
    FederatedRequest,
    FederatedResponse,
    ConsentRecord,
    OCCPDecision,
    OCCPReasonCode,
    OCCPRequestType
)
from occp_gateway import OCCPGateway


class OCCPFederationError(Exception):
    """Raised when federation protocol is violated"""
    pass


class OCCPFederationClient:
    """
    Client for sending federated requests to remote nodes

    PRINCIPLE: Client does NOT execute remote goals
    Client ONLY requests assistance
    """

    def __init__(self, local_node_id: str, local_mcl, local_sk):
        """
        Args:
            local_node_id: This node's ID
            local_mcl: Local MCL checker
            local_sk: Local SK checker
        """
        self.local_node_id = local_node_id
        self.local_mcl = local_mcl
        self.local_sk = local_sk

    async def send_federated_request(
        self,
        remote_gateway: OCCPGateway,
        request: FederatedRequest
    ) -> FederatedResponse:
        """
        Send federated request to remote node

        DUAL-CONSENT MODEL:
        - Local MCL must ALLOW
        - Local SK must ALLOW
        - Remote MCL must ALLOW
        - Remote SK must ALLOW

        ANY DENY → operation cancelled
        """
        # Step 1: Local consent (outgoing)
        local_consent = await self._get_local_consent(request)
        if local_consent.final_decision == OCCPDecision.DENY:
            # Local denial: don't even send to remote
            return FederatedResponse(
                request_id=request.request_id,
                node_id=self.local_node_id,
                denial={
                    "decision": "DENY",
                    "reason_code": local_consent.final_reason,
                    "explanation": "Local veto",
                    "disclosure_level": "minimal",
                    "node_id": self.local_node_id
                }
            )

        # Step 2: Send to remote
        remote_response = await remote_gateway.handle_request(request)

        # Step 3: Record dual-consent
        consent_record = ConsentRecord(
            local_mcl=local_consent.mcl,
            local_sk=local_consent.sk,
            remote_mcl=self._extract_remote_decision(remote_response, "MCL"),
            remote_sk=self._extract_remote_decision(remote_response, "SK"),
            final_decision=self._final_decision_from_response(remote_response)
        )

        # TODO: Audit log consent chain

        return remote_response

    async def _get_local_consent(self, request: FederatedRequest) -> Dict:
        """
        Get local MCL + SK consent for outgoing request
        """
        # Local MCL check
        mcl_allowed = await self.local_mcl.allows_outgoing_federated(request)

        # Local SK check
        sk_allowed = await self.local_sk.allows_outgoing_federated(request)

        if mcl_allowed and sk_allowed:
            return {
                "final_decision": OCCPDecision.ALLOW,
                "mcl": {
                    "layer": "MCL",
                    "decision": OCCPDecision.ALLOW,
                    "explanation": "Local MCL allows outgoing"
                },
                "sk": {
                    "layer": "SK",
                    "decision": OCCPDecision.ALLOW,
                    "explanation": "Local SK allows outgoing"
                }
            }
        else:
            # Determine which denied
            if not sk_allowed:
                final_reason = self.local_sk.veto_reason_code
            else:
                final_reason = OCCPReasonCode.MCL_01

            return {
                "final_decision": OCCPDecision.DENY,
                "final_reason": final_reason,
                "mcl": {
                    "layer": "MCL",
                    "decision": OCCPDecision.ALLOW if mcl_allowed else OCCPDecision.DENY
                },
                "sk": {
                    "layer": "SK",
                    "decision": OCCPDecision.ALLOW if sk_allowed else OCCPDecision.DENY
                }
            }

    def _extract_remote_decision(self, response: FederatedResponse, layer: str) -> Dict:
        """Extract remote decision from response"""
        # TODO: Extract from response audit trail
        return {
            "layer": layer,
            "decision": OCCPDecision.ALLOW  # Stub
        }

    def _final_decision_from_response(self, response: FederatedResponse) -> OCCPDecision:
        """Extract final decision from remote response"""
        if response.denial:
            return OCCPDecision.DENY
        else:
            return OCCPDecision.ALLOW


# =============================================================================
# FEDERATION UTILITIES
# =============================================================================

async def create_adversarial_test_request(
    target_node: str,
    project_description: str,
    compute_seconds: float = 10.0,
    memory_mb: int = 512
) -> FederatedRequest:
    """
    Create adversarial test request

    Use case: "Red-team my project L3"
    """
    from occp_v03_types import ResourceBound

    return FederatedRequest(
        request_type=OCCPRequestType.ADVERSARIAL_TEST,
        resource_bound=ResourceBound(
            compute_seconds=compute_seconds,
            memory_mb=memory_mb
        ),
        sandbox=True,
        audit_level="hash-only",
        # Target node and context passed separately (not in request)
        metadata={
            "target_node": target_node,
            "project_description": project_description
        }
    )


async def create_cognitive_review_request(
    target_node: str,
    decision_context: Dict,
    compute_seconds: float = 5.0
) -> FederatedRequest:
    """
    Create cognitive review request

    Use case: "Review this decision for blind spots"
    """
    from occp_v03_types import ResourceBound

    return FederatedRequest(
        request_type=OCCPRequestType.COGNITIVE_REVIEW,
        resource_bound=ResourceBound(
            compute_seconds=compute_seconds,
            memory_mb=256
        ),
        sandbox=True,
        audit_level="hash-only",
        metadata={
            "target_node": target_node,
            "decision_context": decision_context
        }
    )
