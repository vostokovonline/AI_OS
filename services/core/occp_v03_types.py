"""
OCCP v0.3 Core Types
Multi-Instance / Federated Cognitive Control Protocol
"""
from enum import Enum
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class OCCPRequestType(str, Enum):
    """Canonical federated request types"""
    COMPUTE_ASSIST = "compute_assist"
    COGNITIVE_REVIEW = "cognitive_review"
    VECTOR_TRANSFORM = "vector_transform"
    SIMULATION_RUN = "simulation_run"
    ADVERSARIAL_TEST = "adversarial_test"


class OCCPDecision(str, Enum):
    """OCCP gate decision"""
    ALLOW = "ALLOW"
    DENY = "DENY"
    DEFER = "DEFER"


class OCCPReasonCode(str, Enum):
    """Canonical reason codes for DENY/DEFER"""
    # Goal Contract violations
    G_02 = "G-02"  # Directional goals cannot be executed
    G_03 = "G-03"  # Continuous goals cannot be decomposed
    G_04 = "G-04"  # Meta goals cannot execute directly
    G_05 = "G-05"  # Executable goal without upstream directional

    # MCL violations
    MCL_01 = "MCL-01"  # Preservation mode forbids decomposition
    MCL_02 = "MCL-02"  # Exploration mode forbids direct execution
    MCL_03 = "MCL-03"  # Cognitive mode transition required

    # SK violations
    SK_001 = "SK-001"  # Mission drift too high
    SK_002 = "SK-002"  # Incentive capture detected
    SK_003 = "SK-003"  # Over-optimization risk
    SK_004 = "SK-004"  # Irreversibility risk too high

    # Federation violations (FED-01...FED-05)
    FED_01 = "FED-01"  # Remote execution attempted
    FED_02 = "FED-02"  # Remote goal creation attempted
    FED_03 = "FED-03"  # Cross-node vector application
    FED_04 = "FED-04"  # Non-revocable assistance
    FED_05 = "FED-05"  # Collaboration > survivability

    # Resource violations
    RESOURCE_01 = "RESOURCE-01"  # Insufficient compute
    RESOURCE_02 = "RESOURCE-02"  # Memory limit exceeded


class DisclosureLevel(str, Enum):
    """Federated denial disclosure level"""
    NONE = "none"          # No explanation
    MINIMAL = "minimal"    # Formal reason code only
    FULL = "full"          # Voluntary full explanation


class ResourceBound(BaseModel):
    """Resource limits for federated request"""
    compute_seconds: float = Field(gt=0, le=600)
    memory_mb: int = Field(gt=0, le=16384)
    optional: bool = False


class OCCPDecisionSchema(BaseModel):
    """Single gate decision"""
    layer: Literal["Goal", "MCL", "SK", "Resource"]
    decision: OCCPDecision
    reason_code: Optional[OCCPReasonCode] = None
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class FederatedRequest(BaseModel):
    """Incoming federated request"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_type: OCCPRequestType
    source_node: str  # Node ID of requester (for rate limiting)
    resource_bound: ResourceBound
    sandbox: bool = True
    audit_level: Literal["none", "hash-only", "full"] = "hash-only"

    # CRITICAL: These fields are EXPLICITLY FORBIDDEN
    # (validated at Gateway entry point)
    # goal_id: FORBIDDEN
    # vector_id: FORBIDDEN
    # priority: FORBIDDEN
    # urgency: FORBIDDEN

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class FederatedDenial(BaseModel):
    """Federated operation denial"""
    decision: Literal["DENY"]
    reason_code: OCCPReasonCode
    explanation: str
    disclosure_level: DisclosureLevel = DisclosureLevel.MINIMAL
    node_id: Optional[str] = None


class FederatedResponse(BaseModel):
    """Federated operation response"""
    request_id: str
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Exactly one of these must be present
    result: Optional[Dict] = None
    denial: Optional[FederatedDenial] = None

    # Audit info (hash-only by default)
    audit_hash: Optional[str] = None
    audit_level: Literal["none", "hash-only", "full"] = "hash-only"


class ConsentRecord(BaseModel):
    """Dual-consent record"""
    local_mcl: OCCPDecisionSchema
    local_sk: OCCPDecisionSchema
    remote_mcl: OCCPDecisionSchema
    remote_sk: OCCPDecisionSchema

    final_decision: OCCPDecision
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FederatedAuditEvent(BaseModel):
    """Federated audit log entry"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Request info
    request_id: str
    request_type: OCCPRequestType
    from_node: str
    to_node: str

    # Consent chain
    consent: ConsentRecord

    # Outcome
    final_decision: OCCPDecision

    # Audit (hash-only by default)
    audit_hash: Optional[str] = None
    full_audit: Optional[Dict] = None  # Only if audit_level="full"
