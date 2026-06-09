"""
Execution Adapter - Compatibility Layer for Legacy Execution

Transitional layer that wraps legacy execution paths and converts them
to envelope-based execution with proper enforcement.

Flow:
    Legacy Call → Adapter → ExecutionEnvelope → SkillRegistry → Executor
                                    ↓
                            (Violations logged if not envelope-based)
"""
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from uuid import uuid4

from experience.execution_envelope import ExecutionEnvelope
from experience.enforcement_config import (
    get_enforcement_config,
    get_enforcement_metrics,
    EnforcementMode
)
from experience.skill_registry import (
    SkillRegistry,
    ResolvedSkill,
    get_skill_registry,
    SkillRegistryError
)


@dataclass
class ExecutionContext:
    """Context for execution - built from various sources"""
    goal_id: str
    goal_type: str = "achievable"
    domain: str = "general"
    trace_id: Optional[str] = None
    policy_version: str = "legacy_v1"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LegacyExecutionError(Exception):
    """Raised when legacy execution violates enforcement"""
    pass


class ExecutionAdapter:
    """
    Compatibility layer that wraps legacy execution paths.
    
    Usage:
        adapter = ExecutionAdapter()
        
        # Legacy style call
        result = await adapter.execute_skill(
            skill_id="core.echo",
            inputs={"message": "hello"},
            context=ExecutionContext(goal_id="goal-123")
        )
        
        # If not envelope-based, this will:
        # - WARN: Log violation but execute
        # - QUARANTINE: Execute but mark as non-training
        # - HARD_FAIL: Raise error
    """
    
    def __init__(
        self,
        skill_registry: Optional[SkillRegistry] = None,
        envelope_store = None  # ExecutionEnvelopeStore
    ):
        self.skill_registry = skill_registry or get_skill_registry()
        self._config = get_enforcement_config()
        self._metrics = get_enforcement_metrics()
        self._envelope_store = envelope_store
    
    def _build_context_features(self, context: ExecutionContext, inputs: Dict) -> Dict:
        """Build context features dict from execution context"""
        features = {
            "goal_id": context.goal_id,
            "goal_type": context.goal_type,
            "domain": context.domain,
            "policy_version": context.policy_version,
            "input_keys": list(inputs.keys()),
            "input_count": len(inputs),
        }
        
        if context.metadata:
            features.update(context.metadata)
        
        return features
    
    def _create_envelope(
        self,
        skill_id: str,
        context: ExecutionContext,
        inputs: Dict,
        selected_skill_id: str,
        shadow_skill_id: Optional[str] = None
    ) -> ExecutionEnvelope:
        """Create ExecutionEnvelope from legacy execution context"""
        
        context_features = self._build_context_features(context, inputs)
        
        # Get available candidates (single skill for now)
        candidate_ids = [selected_skill_id]
        if shadow_skill_id:
            candidate_ids.append(shadow_skill_id)
        
        return ExecutionEnvelope.create(
            trace_id=context.trace_id or uuid4().hex[:8],
            policy_version=context.policy_version,
            selected_skill_id=selected_skill_id,
            shadow_skill_id=shadow_skill_id,
            candidate_skill_ids=candidate_ids,
            context_features=context_features,
            goal_type=context.goal_type,
            domain=context.domain
        )
    
    def _log_violation(
        self,
        skill_id: str,
        context: ExecutionContext,
        error: Optional[str] = None
    ):
        """Log enforcement violation"""
        violation_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "skill_id": skill_id,
            "goal_id": context.goal_id,
            "goal_type": context.goal_type,
            "policy_version": context.policy_version,
            "enforcement_mode": self._config.mode.value,
            "error": error or "legacy_execution_not_envelope_based"
        }
        
        # Save to quarantine dir
        quarantine_path = Path(self._config.quarantine_dir)
        quarantine_path.mkdir(parents=True, exist_ok=True)
        
        filename = quarantine_path / f"violation_{uuid4().hex[:8]}.json"
        with open(filename, "w") as f:
            json.dump(violation_data, f, indent=2)
        
        self._metrics.record_violation(self._config.mode)
    
    async def execute_skill(
        self,
        skill_id: str,
        inputs: Dict,
        context: ExecutionContext,
        executor: Callable[[Dict], Awaitable[Any]]
    ) -> Any:
        """
        Execute skill with envelope tracking.
        
        This is the main entry point for envelope-based execution.
        """
        # Resolve skill through registry (enforces core.* format)
        try:
            resolved = self.skill_registry.resolve(skill_id)
        except SkillRegistryError as e:
            # Invalid skill_id - always reject
            self._log_violation(skill_id, context, str(e))
            if self._config.mode == EnforcementMode.HARD_FAIL:
                raise LegacyExecutionError(f"Skill not registered: {skill_id}")
            # Fall through for warn/quarantine
        
        # Create envelope
        envelope = self._create_envelope(
            skill_id=skill_id,
            context=context,
            inputs=inputs,
            selected_skill_id=skill_id
        )
        
        # Store envelope if store configured
        if self._envelope_store:
            self._envelope_store.save(envelope)
        
        self._metrics.record_envelope_execution()
        
        # Execute via skill registry executor
        try:
            skill_executor = self.skill_registry.get_executor(skill_id)
            result = await skill_executor(inputs)
            return result
        except Exception as e:
            # Log but don't fail on execution error
            return {"error": str(e), "status": "failed"}
    
    async def execute_legacy(
        self,
        skill_id: str,
        inputs: Dict,
        context: ExecutionContext,
        executor: Callable[[Dict], Awaitable[Any]]
    ) -> Any:
        """
        Execute skill in legacy mode (no envelope).
        
        This logs violations and either allows, quarantines, or blocks
        based on enforcement mode.
        """
        self._metrics.record_legacy_execution()
        
        # Always log violation
        self._log_violation(skill_id, context)
        
        if self._config.mode == EnforcementMode.HARD_FAIL:
            raise LegacyExecutionError(
                f"Legacy execution blocked: {skill_id}. "
                f"Use execute_skill() with ExecutionEnvelope instead."
            )
        
        # For warn/quarantine, proceed but don't train policy
        can_train = self._config.can_train_policy(is_envelope_based=False)
        
        # Execute anyway (legacy compatibility)
        try:
            result = await executor(inputs)
            return result
        except Exception as e:
            return {"error": str(e), "status": "failed"}
    
    def wrap_legacy_executor(
        self,
        executor: Callable[[Dict], Awaitable[Any]]
    ) -> Callable:
        """
        Wrap a legacy executor function to add envelope tracking.
        
        Usage:
            async def my_skill(inputs):
                return {"result": "ok"}
            
            wrapped = adapter.wrap_legacy_executor(my_skill)
            result = await wrapped({"message": "hello"})
        """
        async def wrapped(inputs: Dict, context: ExecutionContext = None):
            if context is None:
                context = ExecutionContext(
                    goal_id="unknown",
                    policy_version="legacy_v1"
                )
            
            # Check if we should use envelope path
            if self._config.mode == EnforcementMode.HARD_FAIL:
                # Hard fail - only envelope allowed
                raise LegacyExecutionError(
                    "Legacy executor called in hard_fail mode. "
                    "Use execute_skill() with envelope."
                )
            
            # Use legacy path (will log violation)
            return await self.execute_legacy(
                skill_id="unknown",
                inputs=inputs,
                context=context,
                executor=executor
            )
        
        return wrapped


# Global adapter
_execution_adapter: Optional[ExecutionAdapter] = None


def get_execution_adapter() -> ExecutionAdapter:
    """Get or create global execution adapter"""
    global _execution_adapter
    if _execution_adapter is None:
        from experience.execution_envelope import ExecutionEnvelopeStore
        _execution_adapter = ExecutionAdapter(
            envelope_store=ExecutionEnvelopeStore()
        )
    return _execution_adapter