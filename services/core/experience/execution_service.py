"""
Execution Service - Central execution path with envelope enforcement

All skill execution MUST go through this service.
This is the structural enforcement layer.

Flow:
    GoalExecutor → ExecutionService.execute() → ExecutionEnvelope → ResolvedSkill → SkillExecutor
                                              ↓
                                    (enforcement tracking)
"""
from typing import Dict, Any, Optional
from uuid import uuid4
from dataclasses import dataclass

from experience.execution_envelope import ExecutionEnvelope, ExecutionEnvelopeStore
from experience.enforcement_config import (
    get_enforcement_config,
    get_enforcement_metrics,
    EnforcementMode
)
from experience.skill_registry import (
    SkillRegistry,
    get_skill_registry,
    ResolvedSkill,
    SkillRegistryError
)
from canonical_skills.base import SkillResult


@dataclass
class ExecutionRequest:
    """Immutable execution request"""
    skill_id: str
    goal_id: str
    inputs: Dict[str, Any]
    goal_type: str = "achievable"
    domain: str = "general"
    trace_id: Optional[str] = None
    policy_version: str = "legacy_v1"
    
    def to_context_features(self) -> Dict:
        """Convert to context features (no raw inputs)"""
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type,
            "domain": self.domain,
            "policy_version": self.policy_version,
            "input_keys": list(self.inputs.keys()) if self.inputs else [],
            "input_count": len(self.inputs) if self.inputs else 0,
            "skill_id": self.skill_id,
        }


class ExecutionService:
    """
    Central execution service with envelope enforcement.
    
    This is the ONLY path for skill execution in the system.
    No direct skill.execute() calls allowed outside this service.
    
    Enforcement levels:
    - WARN: Log violations, allow all executions
    - QUARANTINE: Legacy executions don't train policy
    - HARD_FAIL: Only envelope-based execution allowed
    """
    
    def __init__(
        self,
        skill_registry: Optional[SkillRegistry] = None,
        envelope_store: Optional[ExecutionEnvelopeStore] = None
    ):
        self._skill_registry = skill_registry or get_skill_registry()
        self._config = get_enforcement_config()
        self._metrics = get_enforcement_metrics()
        self._envelope_store = envelope_store or ExecutionEnvelopeStore()
        
        # Track envelope-based vs legacy executions
        self._executions_with_envelope = 0
        self._executions_legacy = 0
    
    def _validate_skill_id(self, skill_id: str) -> bool:
        """Validate skill_id format"""
        return skill_id.startswith("core.")
    
    def _create_envelope(self, request: ExecutionRequest) -> ExecutionEnvelope:
        """Create ExecutionEnvelope from request"""
        
        context_features = request.to_context_features()
        
        return ExecutionEnvelope.create(
            trace_id=request.trace_id or uuid4().hex[:8],
            execution_id=uuid4().hex[:8],
            policy_version=request.policy_version,
            selected_skill_id=request.skill_id,
            shadow_skill_id=None,
            candidate_skill_ids=[request.skill_id],
            context_features=context_features,
            goal_type=request.goal_type,
            domain=request.domain
        )
    
    def _resolve_skill(self, skill_id: str) -> ResolvedSkill:
        """Resolve skill through registry (enforces core.* format)"""
        return self._skill_registry.resolve(skill_id)
    
    async def execute(
        self,
        request: ExecutionRequest,
        executor_func: Any = None
    ) -> SkillResult:
        """
        Execute skill with envelope enforcement.
        
        This is the main entry point - all skill execution goes through here.
        
        Args:
            request: ExecutionRequest with skill_id, inputs, etc.
            executor_func: Optional executor function (for legacy compatibility)
            
        Returns:
            SkillResult from skill execution
        """
        # Check if this is envelope-based execution
        is_envelope_based = True
        
        # Validate skill_id format
        if not self._validate_skill_id(request.skill_id):
            is_envelope_based = False
            
            # Log violation for invalid skill_id
            self._metrics.record_legacy_execution()
            self._metrics.record_violation(self._config.mode)
            
            if self._config.mode == EnforcementMode.HARD_FAIL:
                raise RuntimeError(
                    f"Invalid skill_id: {request.skill_id}. "
                    f"Must start with 'core.'. Use ExecutionService for proper execution."
                )
        
        # Create envelope
        envelope = self._create_envelope(request)
        
        # Store envelope for replay
        try:
            self._envelope_store.save(envelope)
        except Exception:
            pass  # Non-critical
        
        # Resolve skill through registry
        try:
            resolved = self._resolve_skill(request.skill_id)
        except SkillRegistryError as e:
            # Invalid skill - always fail in hard_fail mode
            self._metrics.record_violation(self._config.mode)
            
            if self._config.mode == EnforcementMode.HARD_FAIL:
                raise RuntimeError(f"Skill not registered: {request.skill_id}")
            
            # In warn/quarantine, try executor_func if provided
            if executor_func:
                try:
                    return await executor_func(request.inputs)
                except Exception as e:
                    return SkillResult(
                        success=False,
                        error=str(e),
                        artifacts=[]
                    )
            
            return SkillResult(
                success=False,
                error=f"Skill not registered: {request.skill_id}",
                artifacts=[]
            )
        
        # Get executor from skill registry
        try:
            skill_executor = self._skill_registry.get_executor(request.skill_id)
        except Exception as e:
            return SkillResult(
                success=False,
                error=f"Failed to get executor: {e}",
                artifacts=[]
            )
        
        # Execute skill
        try:
            # Call the executor function
            result = await skill_executor(request.inputs)
            
            # Update metrics
            self._executions_with_envelope += 1
            self._metrics.record_envelope_execution()
            
            return result
            
        except Exception as e:
            logger.warning(
                "execution_service_skill_failed",
                skill_id=request.skill_id,
                error=str(e)[:100]
            )
            
            return SkillResult(
                success=False,
                error=str(e)[:100],
                artifacts=[]
            )
    
    def can_train_policy(self) -> bool:
        """Check if current executions can train policy"""
        return self._config.can_train_policy(
            is_envelope_based=(self._executions_with_envelope > 0)
        )
    
    def get_stats(self) -> Dict:
        """Get execution statistics"""
        return {
            "envelope_executions": self._executions_with_envelope,
            "legacy_executions": self._executions_legacy,
            "mode": self._config.mode.value,
            "can_train_policy": self.can_train_policy(),
            "metrics": self._metrics.get_summary()
        }


# Global execution service
_execution_service: Optional[ExecutionService] = None


def get_execution_service() -> ExecutionService:
    """Get or create global execution service"""
    global _execution_service
    if _execution_service is None:
        _execution_service = ExecutionService()
    return _execution_service


# Backward compatibility - execute skill the old way (logs violations)
async def execute_skill_legacy(
    skill_id: str,
    inputs: Dict,
    executor_func: Any
) -> SkillResult:
    """
    Legacy execution path - DO NOT USE IN NEW CODE.
    
    This path logs violations and may block in hard_fail mode.
    Use ExecutionService.execute() instead.
    """
    service = get_execution_service()
    
    request = ExecutionRequest(
        skill_id=skill_id,
        goal_id="unknown",
        inputs=inputs,
        policy_version="legacy_v1"
    )
    
    return await service.execute(request, executor_func)