from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any
from datetime import datetime

class MessageCreate(BaseModel):
    session_id: Optional[str] = None
    content: str
    image_url: Optional[str] = None

class MessageResponse(BaseModel):
    role: str
    content: str
    created_at: datetime
    class Config: from_attributes = True

class ResumeRequest(BaseModel):
    session_id: str
    action: str
    feedback: Optional[str] = None

class EventRequest(BaseModel):
    source: str
    payload: object

class QualityReview(BaseModel):
    score: int
    is_acceptable: bool
    feedback: str

class SupervisorDecision(BaseModel):
    next_node: Literal["RESEARCHER", "CODER", "DESIGNER", "PM", "INTELLIGENCE", "COACH", "INNOVATOR", "LIBRARIAN", "DEVOPS", "FINISH"]
    reasoning: str

class Step(BaseModel):
    id: int
    description: str
    assigned_role: Literal["CODER", "RESEARCHER", "DESIGNER", "PM", "SKILL"]
    skill_name: Optional[str] = None
    status: str = "pending"

class Plan(BaseModel):
    steps: List[Step]
    final_goal: str
    reasoning: str

class Budget(BaseModel):
    total_steps: int = 15
    steps_used: int = 0
    allow_human_interaction: bool = True

class MetaEvaluation(BaseModel):
    process_score: int
    waste_detected: bool
    better_path_suggestion: str

# =============================================================================
# Emotional Layer Schemas
# =============================================================================

class EmotionalSignals(BaseModel):
    """
    Aggregated signals for emotional inference.

    IMPORTANT: This contains FACTS, not raw events.
    Context Builder collects statistics, Inference Engine interprets them.
    """
    user_text: Optional[str] = None

    # Goal statistics (aggregated, last 24h)
    # Note: Can contain both int (counts) and float (ratios)
    goal_stats: Optional[Dict[str, float]] = None
    # Example:
    # {
    #   "aborted": 3,
    #   "completed": 5,
    #   "created": 7,
    #   "success_ratio": 0.71
    # }

    # System-level metrics
    system_metrics: Optional[Dict[str, float]] = None
    # Example:
    # {
    #   "avg_goal_complexity": 0.8,
    #   "success_ratio": 0.65,
    #   "retry_count": 2
    # }


class EmotionalStateBase(BaseModel):
    """Base emotional state model"""
    arousal: float = Field(ge=0.0, le=1.0, default=0.5)
    valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    focus: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class EmotionalStateCreate(EmotionalStateBase):
    """Create emotional state"""
    user_id: str
    source: str = "inference"
    signals: Optional[Dict[str, Any]] = None


class EmotionalStateResponse(EmotionalStateBase):
    """Emotional state response"""
    id: str
    user_id: str
    timestamp: datetime
    source: str

    class Config:
        from_attributes = True


class EmotionalInfluence(BaseModel):
    """
    How emotions affect decisions.

    All values are relative modifiers (-1.0 to 1.0).
    Applied ON TOP of baseline decision logic.
    """
    # Complexity modifiers
    complexity_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    exploration_bias: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Communication modifiers
    explanation_depth: float = Field(default=0.0, ge=0.0, le=1.0)
    pace_modifier: float = Field(default=0.0, ge=-1.0, le=1.0)

    # Meta
    confidence_modifier: float = Field(default=0.0, ge=-1.0, le=1.0)


class EmotionalContext(BaseModel):
    """
    Agent-friendly context derived from emotional influence.

    String-based hints (better for LLM understanding).
    """
    complexity_limit: float = Field(default=1.0, ge=0.0, le=1.0)
    max_depth: int = Field(default=3, ge=1, le=3)

    exploration: str = Field(default="balanced")  # conservative | balanced | exploratory
    explanation: str = Field(default="normal")    # brief | normal | detailed
    pace: str = Field(default="normal")          # fast | normal | slow
    confidence: str = Field(default="normal")    # low | normal | high


class AffectiveMemoryEntryBase(BaseModel):
    """Base affective memory entry"""
    goal_id: Optional[str] = None
    decision_id: Optional[str] = None
    emotional_state_before: Dict[str, float]
    emotional_state_after: Optional[Dict[str, float]] = None
    outcome: str  # success | partial | fail | unknown
    outcome_metrics: Optional[Dict[str, Any]] = None


class AffectiveMemoryEntryCreate(AffectiveMemoryEntryBase):
    """Create affective memory entry"""
    user_id: str


class AffectiveMemoryEntryResponse(AffectiveMemoryEntryBase):
    """Affective memory entry response"""
    id: str
    user_id: str
    timestamp: datetime

    class Config:
        from_attributes = True


# =============================================================================
# EMOTIONAL INFERENCE ENGINE V2 SCHEMAS
# =============================================================================

class MetaOutcome(BaseModel):
    """Meta-результат выполнения (learning-aware)"""
    outcome: Literal["success", "failure", "aborted"]
    learning_gain: float = Field(default=0.0, ge=0.0, le=1.0)
    unexpected: bool = False
    effort: float = Field(default=0.5, ge=0.0, le=1.0)
    user_reflection: str = ""


class EmotionalIntent(BaseModel):
    """Эмоциональное намерение пользователя"""
    primary: Literal[
        "restore_confidence",
        "reduce_arousal",
        "maintain_focus",
        "increase_engagement",
        "neutral"
    ] = "neutral"
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class EmotionalForecastV2(BaseModel):
    """Прогноз эмоционального состояния"""
    predicted_state: EmotionalStateBase
    risk_flags: List[str] = []
    expected_delta: Dict[str, float] = {}
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PatternContext(BaseModel):
    """Контекст эмоциональных паттернов"""
    risk_profile: Dict[str, float] = {}
    dominant_patterns: List[str] = []
    success_correlations: Dict[str, float] = {}


class DecisionModifiersV2(BaseModel):
    """Модификаторы решений (EIE v2 output)"""
    max_depth: int = Field(default=3, ge=1, le=3)
    pace: Literal["slow", "normal", "fast"] = "normal"
    explanation_level: Literal["minimal", "normal", "detailed"] = "normal"
    style: Literal["direct", "supportive", "collaborative"] = "direct"
    safety_override: bool = False
    recovery_mode: bool = False


class EIEInferenceRequest(BaseModel):
    """Запрос на эмоциональный inference (v2)"""
    user_id: str
    proposed_action: str
    intent: Optional[EmotionalIntent] = None


# =============================================================================
# STEP 2.4: EMOTIONAL FORECAST & OUTCOME SCHEMAS
# =============================================================================

class EmotionalForecastCreate(BaseModel):
    """Создание emotional forecast (внутренний use case)"""
    user_id: str
    goal_id: Optional[str] = None
    action_type: str
    predicted_deltas: Dict[str, float]  # {arousal, valence, focus, confidence}
    forecast_confidence: float = Field(ge=0.0, le=1.0)
    used_tier: Literal["ML", "Clusters", "Rules"]
    risk_flags: Optional[List[str]] = None


class EmotionalForecastResponse(BaseModel):
    """Emotional forecast response"""
    id: str
    user_id: str
    goal_id: Optional[str] = None
    action_type: str
    predicted_deltas: Dict[str, float]
    forecast_confidence: float
    used_tier: str
    risk_flags: Optional[List[str]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class EmotionalOutcomeCreate(BaseModel):
    """Создание emotional outcome"""
    forecast_id: str
    actual_deltas: Dict[str, float]  # {arousal, valence, focus, confidence}
    outcome: Literal["success", "failure", "aborted"]


class EmotionalOutcomeResponse(BaseModel):
    """Emotional outcome response"""
    id: str
    forecast_id: str
    actual_deltas: Dict[str, float]
    outcome: str
    completed_at: datetime

    class Config:
        from_attributes = True


class TierReliabilityCreate(BaseModel):
    """Создание/обновление tier reliability (внутренний use case)"""
    action_type: str
    tier: Literal["ML", "Clusters", "Rules"]
    sample_count: int = Field(ge=0)
    direction_accuracy: float = Field(ge=0.0, le=1.0)
    mae: float = Field(ge=0.0)
    bias: float
    cce: Optional[float] = Field(None, ge=0.0, le=1.0)


class TierReliabilityResponse(BaseModel):
    """Tier reliability response"""
    id: str
    action_type: str
    tier: str
    sample_count: int
    direction_accuracy: float
    mae: float
    bias: float
    cce: Optional[float] = None
    updated_at: datetime

    class Config:
        from_attributes = True


class ForecastWithOutcome(BaseModel):
    """Forecast + outcome together (для self-eval)"""
    forecast: EmotionalForecastResponse
    outcome: Optional[EmotionalOutcomeResponse] = None


# =============================================================================
# Goal Approval Schemas (Phase 2.2.5)
# =============================================================================

class GoalApproveRequest(BaseModel):
    """Request body for MANUAL goal approval"""
    approved_by: str = Field(..., description="Stable identifier of approving authority (user_id/org_id/DAO_id)")
    authority_level: int = Field(..., ge=1, le=4, description="Authority level performing approval")
    comment: Optional[str] = Field(None, max_length=1024, description="Optional audit comment")

    class Config:
        json_schema_extra = {
            "example": {
                "approved_by": "user:123e4567-e89b-12d3-a456-426614174000",
                "authority_level": 2,
                "comment": "Goal verified and completed"
            }
        }


class GoalApprovalResponse(BaseModel):
    """Response after successful approval"""
    goal_id: str
    status: Literal["done"]
    approved_at: str
    approved_by: str
    authority_level: int

    class Config:
        from_attributes = True


class GoalApprovalErrorResponse(BaseModel):
    """Error response for approval failures"""
    error: dict = Field(..., description="Structured error object")


class GoalApprovalSuccess(BaseModel):
    """Success response (200 OK)"""
    goal_id: str
    status: Literal["done"]
    approved_at: str
    approved_by: str
    authority_level: int


# =============================================================================
# Bulk Transition Schemas (UoW Pattern)
# =============================================================================

class BulkTransitionRequest(BaseModel):
    """Request body for bulk goal transitions"""
    goal_ids: List[str] = Field(..., min_length=1, max_length=1000, description="List of goal IDs to transition")
    new_state: str = Field(..., description="Target state: pending, active, done, frozen, archived")
    reason: str = Field(default="Bulk operation", max_length=500, description="Reason for transition")
    actor: str = Field(default="system", max_length=100, description="Who initiated the transition")
    
    class Config:
        json_schema_extra = {
            "example": {
                "goal_ids": ["uuid-1", "uuid-2", "uuid-3"],
                "new_state": "active",
                "reason": "Mass activation for Q1 sprint",
                "actor": "admin:user:123"
            }
        }


class BulkTransitionResponse(BaseModel):
    """Response after bulk transition"""
    total: int = Field(..., description="Total goals requested")
    found: int = Field(..., description="Goals found in database")
    succeeded: int = Field(..., description="Successfully transitioned")
    failed: int = Field(..., description="Failed or blocked")
    results: List[Dict[str, Any]] = Field(..., description="Individual results per goal")
    timestamp: str = Field(..., description="Operation timestamp")


class FreezeTreeRequest(BaseModel):
    """Request body for freezing a goal tree"""
    root_goal_id: str = Field(..., description="Root goal ID")
    reason: str = Field(default="Tree frozen", max_length=500, description="Reason for freeze")
    actor: str = Field(default="system", max_length=100, description="Who initiated the freeze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "root_goal_id": "uuid-root",
                "reason": "Project paused for review",
                "actor": "admin:user:456"
            }
        }
