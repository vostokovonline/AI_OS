"""
GOAL EXECUTOR V2 - Integration with Canonical Skills
Интегрирует Goal Executor с навыками по каноническому интерфейсу

Execution Flow:
1. Parse goal requirements
2. Select skill via registry
3. Execute skill
4. Verify result
5. Register artifacts
6. Check completion

ARCHITECTURE v3.0:
- Uses UnitOfWork pattern for transaction management
- All transactions opened by caller, not internally
"""
import os
from uuid import UUID
from pathlib import Path
from typing import Dict, Optional
from sqlalchemy import select
from models import Goal

# Import canonical skill system
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from canonical_skills.base import Skill, SkillResult, Artifact
from canonical_skills.registry import skill_registry

# Import Experience Engine for learning loop
from experience import experience_engine
from canonical_skills.echo import EchoSkill
from canonical_skills.write_file import WriteFileSkill
from evaluation_engine import evaluation_engine

# Import LLM for content generation
from llm_fallback import chat_with_fallback

# Import production wiring (Phase 2.5.P)
from executor_feedback_wiring import executor_with_feedback, ExecutionMustStopException

# Import UoW infrastructure
from infrastructure.uow import GoalRepository
from goal_transition_service import transition_service

# Import Execution Policy (Phase 2 architectural change)
try:
    from execution_policy import get_execution_policy, ExecutionContext
except ImportError:
    # Fallback if policy not available
    get_execution_policy = None

# Import logging
from logging_config import get_logger
logger = get_logger(__name__)

# Import Skill Evolution Loop for learning
try:
    from skill_evolution_loop import get_skill_evolution_loop
    SKILL_EVOLUTION_LOOP = get_skill_evolution_loop()
except ImportError:
    SKILL_EVOLUTION_LOOP = None
    logger.warning("skill_evolution_loop_not_available")


# =============================================================================
# CRITICAL FIX: normalize_skill_id()
# =============================================================================
def normalize_skill_id(skill) -> str:
    """
    Extract skill_id with multiple fallbacks.
    NEVER returns "unknown" - always returns a valid identifier.

    This is critical because:
    - goal_executions table requires valid skill_id
    - experiences table requires valid skill_used
    - Analytics and learning depend on correct skill_id

    Args:
        skill: Skill instance or class

    Returns:
        str: Valid skill identifier (never "unknown")
    """
    # Try 1: Direct id attribute
    skill_id = getattr(skill, 'id', None)
    if skill_id and skill_id != "unknown":
        return skill_id

    # Try 2: name attribute (some skills use 'name' instead of 'id')
    name = getattr(skill, 'name', None)
    if name and name != "unknown":
        return name

    # Try 3: Class name conversion
    class_name = skill.__class__.__name__
    if "Skill" in class_name:
        # Convert "EchoSkill" → "core.echo"
        # Convert "WriteFileSkill" → "core.write_file"
        base_name = class_name.replace("Skill", "").lower()
        return f"core.{base_name}"

    # Try 4: Module name
    module_name = getattr(skill, '__module__', '')
    if 'canonical_skills' in module_name:
        return f"canonical.{class_name.lower()}"

    # Try 5: Generate consistent hash-based ID
    # This ensures the same skill always gets the same ID
    skill_hash = hash(str(skill))
    skill_id = f"skill_{skill_hash & 0x7fffffff}"  # Positive hash
    logger.warning(
        "skill_id_using_hash",
        skill_class=class_name,
        module=module_name,
        generated_id=skill_id
    )
    return skill_id
    ExecutionContext = None

# Centralized logging
from logging_config import get_logger

logger = get_logger(__name__)


class GoalExecutorV2:
    """
    Goal Executor v2 with Canonical Skills Integration

    Отличие от v1:
    - Работает с навыками по каноническому интерфейсу
    - Skills возвращают SkillResult с Artifact[]
    - Skills самостоятельно verify свои результаты
    - Goal Executor только orchestrates, не выполняет логику
    """

    # Class-level constants (accessible from all methods)
    MAX_ATTEMPTS_DEFAULT = 2
    RETRY_DELAY_SECONDS = 2
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self):
        from infrastructure.uow import GoalRepository
        self._repo = GoalRepository()
        self._init_skills()

    def _init_skills(self):
        """Инициализация и регистрация навыков"""
        # Note: Skills are already loaded by unified_skill_service and registered in skill_router
        # Here we just register in canonical_skills.registry for execution engine
        
        # Register core skills that are needed for execution
        # (these are loaded from canonical_skills by unified_skill_service)
        try:
            from canonical_skills.echo import EchoSkill
            skill_registry.register(EchoSkill())
        except Exception as e:
            logger.warning("echo_load_failed", error=str(e))
        
        try:
            from canonical_skills.write_file import WriteFileSkill
            skill_registry.register(WriteFileSkill())
        except Exception as e:
            logger.warning("write_file_load_failed", error=str(e))
        
        # File operations - may already be loaded by unified_skill_service
        try:
            from canonical_skills.file_read import FileReadSkill
            skill_registry.register(FileReadSkill())
        except Exception as e:
            logger.warning("file_read_load_failed", error=str(e))
        
        try:
            from canonical_skills.file_list import FileListSkill
            skill_registry.register(FileListSkill())
        except Exception as e:
            logger.warning("file_list_load_failed", error=str(e))
        
        try:
            from canonical_skills.file_search import FileSearchSkill
            skill_registry.register(FileSearchSkill())
        except Exception as e:
            logger.warning("file_search_load_failed", error=str(e))
        
        # System skills
        try:
            from canonical_skills.run_command import RunCommandSkill
            skill_registry.register(RunCommandSkill())
        except Exception as e:
            logger.warning("run_command_load_failed", error=str(e))
        
        # AI skills
        try:
            from canonical_skills.summarize_text import SummarizeTextSkill
            skill_registry.register(SummarizeTextSkill())
        except Exception as e:
            logger.warning("summarize_text_load_failed", error=str(e))
        
        try:
            from canonical_skills.analyze_text import AnalyzeTextSkill
            skill_registry.register(AnalyzeTextSkill())
        except Exception as e:
            logger.warning("analyze_text_load_failed", error=str(e))
        
        try:
            from canonical_skills.create_directory import CreateDirectorySkill
            skill_registry.register(CreateDirectorySkill())
        except Exception as e:
            logger.warning("create_directory_load_failed", error=str(e))

        # Web research skill
        try:
            from canonical_skills.web_research import WebResearchSkill
            web_skill = WebResearchSkill()
            logger.info("web_research_skill_loaded", skill_id=str(web_skill.id), capabilities=web_skill.capabilities)
            skill_registry.register(web_skill)
            logger.info("web_research_skill_registered")
        except Exception as e:
            logger.warning("web_research_load_failed", error=str(e))
            import traceback
            logger.error("traceback", exc_info=True)

        logger.info("goal_executor_v2_initialized", skills_count=len(skill_registry.list()))
        for skill in skill_registry.list():
            logger.debug("registered_skill", skill_id=skill.id, version=skill.version, description=skill.description)

    async def execute_goal(
        self,
        *,
        goal_id: str,
        uow: "UnitOfWork",
        session_id: Optional[str] = None,
        max_attempts_override: Optional[int] = None
    ):
        """
        Выполняет цель используя Skills (v6.0 - Pure Function).

        CRITICAL INVARIANT:
        ===================
        Executor is now a PURE FUNCTION.
        Returns ExecutionOutcome, NOT state changes.

        NO:
        - transition_service calls
        - artifact_registry writes
        - DB mutations
        - goal.status changes

        YES:
        - Read goal (via uow)
        - Execute skill with retry
        - Return ExecutionOutcome with artifact data

        ARCHITECTURE v6.0:
        ==================
        Phase 1 (READ):   Get goal snapshot (read-only)
        Phase 2 (EXECUTE): Run skill with retry
        Phase 3 (OUTCOME): Return ExecutionOutcome (no side effects)

        Caller (BulkEngine) decides what to do with outcome.
        """
        from uuid import UUID
        from datetime import datetime
        import asyncio

        # Use class-level constants, with override support for max_attempts
        max_attempts = max_attempts_override if max_attempts_override is not None else self.MAX_ATTEMPTS_DEFAULT

        trace = {
            "goal_id": goal_id,
            "started_at": datetime.utcnow().isoformat(),
            "steps": [],
            "attempts": []
        }

        # =================================================================
        # PHASE 1: READ - Get goal snapshot (uses caller's transaction)
        # =================================================================
        logger.info("execution_phase_1_read", goal_id=goal_id[:8])

        # Use caller's transaction
        goal = await self._repo.get(uow.session, UUID(goal_id))
        if not goal:
            return {"status": "error", "message": f"Goal {goal_id} not found"}

        goal_snapshot = {
            "id": str(goal.id),
            "title": goal.title,
            "description": goal.description,
            "goal_type": goal.goal_type,
            "is_atomic": goal.is_atomic,
            "completion_criteria": goal.completion_criteria,
            "success_definition": goal.success_definition,
            "domains": goal.domains or [],
            "constraints": goal.constraints or {},
            "progress": goal.progress or 0.0,
            "execution_mode": goal.completion_mode or "auto"
        }

        trace["steps"].append({"step": "phase_1_read", "success": True})

        # =================================================================
        # PHASE 2: EXECUTE with RETRY - Run skill WITHOUT transaction
        # =================================================================
        logger.info("execution_phase_2_execute_with_retry", goal_id=goal_id[:8])

        best_result = None
        best_evaluation = None
        attempt_feedback = []

        for attempt_num in range(1, max_attempts + 1):
            attempt_start = datetime.utcnow()
            trace["attempts"].append({"attempt": attempt_num, "started_at": attempt_start.isoformat()})

            try:
                result, evaluation = await self._execute_single_attempt(
                    goal_snapshot=goal_snapshot,
                    goal_id=goal_id,
                    session_id=session_id,
                    attempt_num=attempt_num,
                    previous_feedback=attempt_feedback
                )

                attempt_end = datetime.utcnow()
                trace["attempts"][-1]["duration_ms"] = int((attempt_end - attempt_start).total_seconds() * 1000)
                trace["attempts"][-1]["success"] = result.success
                trace["attempts"][-1]["confidence"] = evaluation.confidence if evaluation else 0.0

                logger.info("attempt_complete",
                           attempt=attempt_num,
                           success=result.success,
                           confidence=evaluation.confidence if evaluation else 0.0)

                # Track best result
                if best_evaluation is None or (evaluation and evaluation.confidence > best_evaluation.confidence):
                    best_result = result
                    best_evaluation = evaluation

                # Check if we can stop (strong pass)
                if evaluation and evaluation.confidence >= 0.8 and result.success:
                    logger.info("strong_pass_achieved", attempt=attempt_num)
                    break

                # Determine if retry is needed
                if not self._should_retry(result, evaluation, attempt_num):
                    break

                # Prepare feedback for next attempt
                attempt_feedback = self._generate_retry_feedback(result, evaluation)
                # attempt_feedback is a list of dicts, get first reason
                reason = "unknown"
                if attempt_feedback and len(attempt_feedback) > 0:
                    reason = attempt_feedback[0].get("type", "unknown")

                logger.info("retry_scheduled",
                           attempt=attempt_num,
                           reason=reason)

                # Exponential backoff
                await asyncio.sleep(self.RETRY_DELAY_SECONDS * attempt_num)

            except Exception as e:
                logger.error("attempt_failed", attempt=attempt_num, error=str(e)[:100])
                trace["attempts"][-1]["error"] = str(e)[:100]
                
                # Check if this is a hard failure
                if self._is_hard_failure(e):
                    break
                
                # Soft failure, prepare for retry
                attempt_feedback = [{"type": "error", "message": str(e)[:100]}]
                await asyncio.sleep(self.RETRY_DELAY_SECONDS * attempt_num)

        # Use best result from all attempts
        result = best_result
        evaluation = best_evaluation

        trace["total_attempts"] = len(trace["attempts"])
        trace["final_confidence"] = evaluation.confidence if evaluation else 0.0

        # =================================================================
        # PHASE 3: OUTCOME - Return pure result (NO state mutations)
        # =================================================================
        logger.info("execution_phase_3_outcome", goal_id=goal_id[:8])

        from application.execution.outcomes import ExecutionOutcome

        # Determine status
        if evaluation and evaluation.passed:
            if evaluation.confidence >= self.CONFIDENCE_THRESHOLD:
                status = "completed"
            else:
                status = "failed"
        else:
            status = "failed"

        # Extract artifact data ( WITHOUT registering in DB)
        artifacts_data = []
        if result and result.artifacts:
            for artifact in result.artifacts:
                artifacts_data.append({
                    "type": artifact.get("type", "FILE"),
                    "content_kind": artifact.get("content_kind", "file"),
                    "content_location": artifact.get("content_location", ""),
                    "verification_rule": artifact.get("verification_rule")
                })

        # Return pure outcome - NO side effects
        return ExecutionOutcome(
            status=status,
            confidence=evaluation.confidence if evaluation else 0.0,
            attempts=len(trace["attempts"]),
            artifacts=artifacts_data,
            execution_trace=trace
        )

    async def _execute_single_attempt(
        self,
        goal_snapshot: dict,
        goal_id: str,
        session_id: Optional[str],
        attempt_num: int,
        previous_feedback: list
    ) -> tuple:
        """Execute single attempt with feedback incorporation."""
        from evaluation_engine import evaluation_engine
        from datetime import datetime

        requirements = self._parse_requirements_from_snapshot(goal_snapshot)
        skill = await self._select_skill(requirements, goal_snapshot)
        
        if not skill:
            from canonical_skills.base import SkillResult
            return SkillResult(success=False, error="No skill found", artifacts=[]), None

        inputs = await self._prepare_inputs_from_snapshot(goal_snapshot, skill)
        
        if previous_feedback and attempt_num > 1:
            inputs = self._incorporate_feedback(inputs, previous_feedback)

        context = {
            "goal_id": goal_id,
            "session_id": session_id or f"goal_{goal_id}",
            "goal_title": goal_snapshot["title"],
            "attempt": attempt_num,
            "previous_failures": len(previous_feedback)
        }

        result = skill.execute(inputs, context)

        evaluation = evaluation_engine.evaluate_goal(
            goal_completion_criteria=goal_snapshot.get("completion_criteria"),
            artifacts_produced=result.artifacts if result.artifacts else [],
            goal_title=goal_snapshot["title"],
            goal_description=goal_snapshot["description"]
        )

        return result, evaluation

    def _should_retry(self, result, evaluation, attempt_num: int) -> bool:
        """Determine if retry is warranted."""
        if attempt_num >= self.MAX_ATTEMPTS_DEFAULT:
            return False

        if not result or not result.success:
            return True

        if not evaluation:
            return True

        if evaluation.confidence < self.CONFIDENCE_THRESHOLD:
            return True

        if evaluation.passed and evaluation.confidence >= 0.8:
            return False

        return evaluation.confidence < 0.7

    def _is_hard_failure(self, error: Exception) -> bool:
        """Check if error is non-retryable."""
        error_str = str(error).lower()
        hard_patterns = [
            "not found",
            "invalid input",
            "constraint violation",
            "unauthorized",
            "forbidden",
            "skill not found"
        ]
        return any(pattern in error_str for pattern in hard_patterns)

    def _generate_retry_feedback(self, result, evaluation) -> list:
        """Generate feedback for next attempt."""
        feedback = []
        
        if not result.success:
            feedback.append({
                "type": "skill_failure",
                "error": result.error if result else "Unknown error"
            })

        if evaluation and not evaluation.passed:
            feedback.append({
                "type": "evaluation_failure",
                "checks_failed": [c for c in evaluation.checks if not c.get("passed")]
            })

        if evaluation and evaluation.confidence < self.CONFIDENCE_THRESHOLD:
            feedback.append({
                "type": "low_confidence",
                "confidence": evaluation.confidence
            })

        return feedback

    def _incorporate_feedback(self, inputs: dict, feedback: list) -> dict:
        """Adjust inputs based on previous feedback."""
        enhanced = inputs.copy()
        
        failure_summary = []
        for f in feedback:
            if f["type"] == "skill_failure":
                failure_summary.append(f"Previous error: {f['error']}")
            elif f["type"] == "evaluation_failure":
                failed_checks = f.get("checks_failed", [])
                if failed_checks:
                    failure_summary.append(f"Failed checks: {len(failed_checks)}")
            elif f["type"] == "low_confidence":
                failure_summary.append(f"Low confidence: {f['confidence']:.2f}")

        if failure_summary:
            enhanced["previous_attempt_feedback"] = "\n".join(failure_summary)
            enhanced["retry_instruction"] = "Previous attempt failed. Please try a different approach."

        return enhanced

    async def _write_execution_trace(self, goal_id: str, trace: dict, status: str, message: str):
        """Write execution trace to goal (separate short transaction)."""
        from infrastructure.uow import create_uow_provider
        from uuid import UUID
        
        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
            goal = await self._repo.get(uow.session, UUID(goal_id))
            if goal:
                goal.execution_trace = trace
                await self._repo.update(uow.session, goal)

    def _parse_requirements_from_snapshot(self, snapshot: dict) -> dict:
        """Parse requirements from goal snapshot (no DB access)."""
        title = snapshot.get("title", "").lower()
        description = snapshot.get("description", "").lower()
        
        # Extract capabilities from title/description
        capabilities = []
        if any(w in title or w in description for w in ['research', 'search', 'find', 'lookup']):
            capabilities.append('research')
            capabilities.append('web-research')
            capabilities.append('search')
        if any(w in title or w in description for w in ['write', 'create', 'generate', 'save']):
            capabilities.append('file-write')
        if any(w in title or w in description for w in ['test', 'check', 'verify']):
            capabilities.append('testing')
        if any(w in title or w in description for w in ['code', 'program', 'implement']):
            capabilities.append('coding')
        
        # Extract expected artifacts
        artifacts = []
        if any(w in title or w in description for w in ['knowledge', 'research', 'information', 'findings']):
            artifacts.append('KNOWLEDGE')
        if any(w in title or w in description for w in ['file', 'config', 'code', 'script']):
            artifacts.append('FILE')
        
        return {
            "goal_type": snapshot.get("goal_type", "achievable"),
            "domains": snapshot.get("domains", []),
            "constraints": snapshot.get("constraints", {}),
            "title": snapshot.get("title", ""),
            "description": snapshot.get("description", ""),
            "capabilities": capabilities,
            "artifacts": artifacts
        }

    async def _select_skill(self, requirements: dict, goal_snapshot: dict):
        """Select appropriate skill with scoring (no DB access)."""
        global skill_registry

        required_capabilities = requirements.get("capabilities", [])
        required_artifacts = requirements.get("artifacts", [])

        all_skills = skill_registry.list()

        if not all_skills:
            # Try MCP for missing capabilities
            if required_capabilities:
                self._try_mcp_generation(required_capabilities, requirements, goal_snapshot)
            return EchoSkill()

        scored_skills = []
        
        # Get skill stats from CACHE (not DB - loop.is_running issue)
        from experience.skill_stats_cache import get_skill_stats_sync
        skill_stats = await get_skill_stats_sync()

        for skill in all_skills:
            if not skill or not hasattr(skill, 'capabilities'):
                continue

            skill_caps = getattr(skill, 'capabilities', [])
            skill_artifacts = getattr(skill, 'produces', [])
            skill_id = getattr(skill, 'id', '')

            # Base score from capability matching
            score = 0
            matched_caps = []
            missed_caps = []

            # Score capability matches
            for req_cap in required_capabilities:
                if req_cap in skill_caps:
                    score += 5
                    matched_caps.append(req_cap)
                elif req_cap in ['research', 'web-research', 'search']:
                    # Penalize if research requested but not available
                    score -= 2
                    missed_caps.append(req_cap)

            # Score artifact type matches
            for req_art in required_artifacts:
                if req_art in skill_artifacts:
                    score += 3

            # EXPERIENCE-BASED SCORING (proper formula)
            if skill_id in skill_stats:
                stats = skill_stats[skill_id]
                success_rate = stats.get('success_rate', 0.5)
                avg_latency = stats.get('avg_latency_ms', 1000)
                exploration_bonus = stats.get('exploration_bonus', 1.0)
                avg_confidence = stats.get('avg_confidence', 0.5)

                # PROPER SCORING FORMULA (as per user specs)
                # success_rate * 0.55 + speed * 0.2 + exploration * 0.15 + confidence * 0.1

                # Speed score: 1 / (1 + latency/1000) = faster is better
                speed_score = 1.0 / (1.0 + avg_latency / 1000.0)

                # Composite experience score (0-20 scale to match capability scoring)
                experience_score = (
                    success_rate * 0.55 * 20 +      # 0-11 points
                    speed_score * 0.2 * 20 +         # 0-4 points
                    exploration_bonus * 0.15 * 20 +  # 0-3 points
                    avg_confidence * 0.1 * 20         # 0-2 points
                )

                score += experience_score

            # Penalize generic skills (echo is fallback, not primary)
            skill_name = getattr(skill, 'name', '')
            if 'echo' in skill_name.lower():
                score -= 10

            # Bonus for exact capability match
            if matched_caps and len(matched_caps) == len(required_capabilities):
                score += 5

            scored_skills.append((score, skill, matched_caps, missed_caps))

        if not scored_skills:
            return EchoSkill()

        # EPSILON-GREEDY EXPLORATION
        # 15% chance to explore random skill instead of best
        import random
        epsilon = 0.15

        if random.random() < epsilon and len(scored_skills) > 1:
            # Explore: pick random from top 3
            top_k = min(3, len(scored_skills))
            chosen = random.randint(0, top_k - 1)
            best_skill = scored_skills[chosen][1]

            logger.info(
                "skill_selection_explore",
                epsilon=epsilon,
                chosen_index=chosen,
                total_skills=len(scored_skills)
            )
        else:
            # Exploit: pick best skill
            scored_skills.sort(key=lambda x: x[0], reverse=True)
            best_skill = scored_skills[0][1]

        # Log top-3 candidates for explainability
        from logging_config import get_logger
        selection_logger = get_logger(__name__)

        top_3 = scored_skills[:3] if len(scored_skills) >= 3 else scored_skills
        for i, (score, skill, matched, missed) in enumerate(top_3):
            skill_name = getattr(skill, 'name', skill.__class__.__name__)
            selection_logger.info(
                "skill_selection_candidate",
                rank=i+1,
                skill_name=skill_name,
                score=score,
                matched_capabilities=matched,
                missed_capabilities=missed
            )

        best_score, best_skill, matched, missed = scored_skills[0]
        skill_name = getattr(best_skill, 'name', best_skill.__class__.__name__)

        selection_logger.info(
            "skill_selection_final",
            selected_skill=skill_name,
            final_score=best_score,
            matched_capabilities=matched,
            missed_capabilities=missed,
            total_candidates=len(scored_skills)
        )

        # If best score is negative or too low, use better skill or fallback
        if best_score < 0 and len(scored_skills) > 1:
            selection_logger.warning(
                "skill_selection_low_score_fallback",
                original_skill=skill_name,
                original_score=best_score,
                falling_back_to_rank=2
            )
            best_score, best_skill, matched, missed = scored_skills[1]
            skill_name = getattr(best_skill, 'name', best_skill.__class__.__name__)
            selection_logger.info(
                "skill_selection_fallback",
                fallback_skill=skill_name,
                fallback_score=best_score
            )

        # If best match is still poor (score < 0 or significant missed capabilities), try MCP
        if best_score < 0 or (matched and len(missed) > len(matched)):
            # Trigger MCP generation for missing capabilities
            if missed:
                self._try_mcp_generation(missed, requirements, goal_snapshot)

        return best_skill

    def _try_mcp_generation(
        self,
        required_capabilities: list,
        requirements: dict,
        goal_snapshot: dict
    ):
        """
        Try to generate skill via MCP for missing capabilities.

        This is a non-blocking trigger - generation happens in background.
        Current execution uses EchoSkill as fallback.
        """
        try:
            import asyncio
            from mcp_manager import mcp_manager

            # Trigger generation in background
            asyncio.create_task(
                mcp_manager.find_or_generate_skill(
                    capabilities=required_capabilities,
                    requirements=requirements,
                    goal_context={
                        "title": goal_snapshot.get("title", ""),
                        "description": goal_snapshot.get("description", "")
                    }
                )
            )

            logger.info(
                "mcp_generation_triggered",
                capabilities=required_capabilities,
                fallback="echo_skill"
            )

        except Exception as e:
            logger.warning(
                "mcp_generation_trigger_failed",
                capabilities=required_capabilities,
                error=str(e)
            )

    async def _select_skill_with_performance(
        self,
        requirements: dict,
        goal_snapshot: dict,
        session
    ):
        """
        PHASE 1: Enhanced skill selection with historical performance metrics.

        Incorporates:
        - Success rate multiplier (high-performing skills get 1.2x boost)
        - Latency penalty (slow skills get penalized)
        - Minimum execution threshold (5+ executions before trusting stats)
        """
        from infrastructure.execution_repositories import SkillStatsRepository

        # Get base skill selection (capability matching)
        skill = await self._select_skill(requirements, goal_snapshot)

        # Try to enhance with performance metrics
        try:
            skill_stats_repo = SkillStatsRepository()
            # Use skill.id (e.g., "core.echo") not skill.name (which may not exist)
            skill_id = getattr(skill, 'id', skill.__class__.__name__)

            # Fetch stats for this skill
            stats = await skill_stats_repo.get(session, skill_id)

            if stats and stats.total_executions >= 5:
                # Apply performance-based adjustments
                success_rate = stats.success_rate or 0.5
                avg_latency = stats.avg_latency_ms or 0

                # Performance multipliers
                performance_multiplier = 1.0

                # Boost high-performing skills
                if success_rate > 0.9:
                    performance_multiplier = 1.2
                    logger.info(
                        "high_performing_skill_boost",
                        skill=skill_id,
                        success_rate=f"{success_rate:.2%}",
                        multiplier=performance_multiplier
                    )
                elif success_rate < 0.7:
                    # Penalize low-performing skills
                    performance_multiplier = 0.5
                    logger.warning(
                        "low_performing_skill_penalty",
                        skill=skill_id,
                        success_rate=f"{success_rate:.2%}",
                        multiplier=performance_multiplier
                    )

                # Latency penalty (skills slower than 5s get penalty)
                latency_penalty = 0
                if avg_latency > 5000:
                    latency_penalty = -2
                    logger.warning(
                        "high_latency_penalty",
                        skill=skill_id,
                        avg_latency_ms=avg_latency,
                        penalty=latency_penalty
                    )

                # Log final decision
                logger.info(
                    "skill_selection_with_performance",
                    skill=skill_id,
                    executions=stats.total_executions,
                    success_rate=f"{success_rate:.2%}",
                    avg_latency_ms=avg_latency,
                    performance_multiplier=performance_multiplier,
                    latency_penalty=latency_penalty
                )

                # Note: In future iterations, we could use these metrics
                # to trigger fallback to alternative skills
            else:
                logger.debug(
                    "insufficient_performance_data",
                    skill=skill_id,
                    executions=stats.total_executions if stats else 0,
                    min_required=5
                )

        except Exception as e:
            logger.warning(
                "performance_metrics_fetch_failed",
                error=str(e),
                falling_back_to="basic_selection"
            )

        return skill

    async def _prepare_inputs_from_snapshot(self, snapshot: dict, skill) -> dict:
        """Prepare skill inputs from snapshot (may involve LLM, but no DB)."""
        return {
            "goal_title": snapshot.get("title", ""),
            "goal_description": snapshot.get("description", ""),
            "goal_id": snapshot.get("id", "")
        }

    async def execute_goal_with_uow(
        self,
        uow: "UnitOfWork",
        goal_id: str,
        session_id: Optional[str] = None
    ) -> dict:
        """
        Выполняет цель используя Skills WITHIN an existing transaction.

        ARCHITECTURE v3.0: Transaction managed by caller via UnitOfWork.

        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели
            session_id: ID сессии

        Returns:
            Результат выполнения с artifacts
        """
        goal = await self._repo.get(uow.session, UUID(goal_id))

        if not goal:
            return {"status": "error", "message": "Goal not found"}

        if not goal.is_atomic:
            raise ValueError(
                f"V2 only handles atomic goals. Goal '{goal.title}' is not atomic. "
                f"Use Orchestrator (V1) for complex goals."
            )

        return await self._execute_atomic_goal_with_uow(uow, goal, session_id)

    async def _execute_atomic_goal_with_uow(
        self,
        uow: "UnitOfWork",
        goal: Goal,
        session_id: Optional[str]
    ) -> dict:
        """
        Выполняет atomic goal via Skills WITHIN existing transaction.

        Flow v3.0:
        1. Parse requirements
        2. Select skill
        3. Prepare inputs
        4. Execute skill (WITH TRACE)
        5. Verify result
        6. Register artifacts
        7. Evaluate
        8. Check completion

        All operations use the passed UoW - NO internal commit/rollback.
        """
        logger.info("atomic_goal_execution_started", goal_title=goal.title, goal_id=str(goal.id))

        # Start execution trace
        from datetime import datetime
        execution_start = datetime.utcnow()
        goal.execution_started_at = execution_start

        trace = {
            "goal_id": str(goal.id),
            "goal_title": goal.title,
            "started_at": execution_start.isoformat(),
            "steps": []
        }

        # PHASE 1: Create execution tracking record
        from execution_models import GoalExecution
        from uuid import uuid4

        execution_rec = GoalExecution(
            execution_id=uuid4(),
            goal_id=goal.id,
            skill_id="",  # Will be set after skill selection
            started_at=execution_start,
            success=False,  # Will be updated on completion
            artifacts_count=0
        )

        # Capture execution engine (v3 vs legacy)
        # This is passed from caller, defaults to legacy if not specified
        execution_engine = getattr(goal, '_execution_engine', None)

        try:
            # Step 1: Parse requirements
            requirements = self._parse_requirements(goal)
            logger.debug("goal_requirements", requirements=requirements)

            trace["steps"].append({
                "step": "parse_requirements",
                "requirements": requirements
            })

            # Step 2: Select skill (PHASE 1: with performance metrics)
            skill = await self._select_skill_with_performance(
                requirements,
                goal,
                uow.session
            )
            if not skill:
                trace["steps"].append({
                    "step": "skill_selection",
                    "success": False,
                    "reason": f"No suitable skill found for requirements: {requirements}"
                })
                goal.execution_trace = trace
                await self._save_goal_with_uow(uow, goal)

                return {
                    "status": "error",
                    "message": f"No suitable skill found for requirements: {requirements}"
                }

            # Record skill selection in trace
            skill_id_normalized = normalize_skill_id(skill)
            trace["steps"].append({
                "step": "skill_selection",
                "success": True,
                "skill_selected": skill_id_normalized,
                "skill_version": skill.version,
                "selection_reason": self._explain_skill_selection(skill, requirements),
                "skill_capabilities": skill.capabilities,
                "artifacts_produced_by_skill": skill.produces_artifacts
            })

            logger.info("skill_selected", skill_id=skill_id_normalized)

            # PHASE 1: Update execution record with skill_id
            execution_rec.skill_id = skill_id_normalized
            execution_rec.execution_engine = execution_engine

            # Step 3: Prepare inputs (async for LLM generation)
            inputs = await self._prepare_inputs(goal, skill)
            logger.debug("skill_inputs", inputs=list(inputs.keys()))

            trace["steps"].append({
                "step": "prepare_inputs",
                "inputs_provided": list(inputs.keys())
            })

            await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="active",
                reason="Starting atomic goal execution",
                actor="goal_executor_v2"
            )

            # Step 4: Execute skill
            context = {
                "goal_id": str(goal.id),
                "session_id": session_id or f"goal_{goal.id}",
                "goal_title": goal.title
            }

            execution_step_start = datetime.utcnow()

            result: SkillResult = skill.execute(inputs, context)

            execution_step_end = datetime.utcnow()

            logger.info("skill_execution_result", success=result.success)
            logger.info("artifacts_count", count=len(result.artifacts))

            # Record execution in trace
            trace["steps"].append({
                "step": "execute_skill",
                "success": result.success,
                "skill_id": skill_id_normalized,
                "started_at": execution_step_start.isoformat(),
                "completed_at": execution_step_end.isoformat(),
                "duration_ms": int((execution_step_end - execution_step_start).total_seconds() * 1000),
                "artifacts_produced": len(result.artifacts),
                "error": result.error if not result.success else None
            })

            if not result.success:
                await transition_service.transition(
                    uow=uow,
                    goal_id=goal.id,
                    new_state="blocked",
                    reason="Skill execution failed",
                    actor="goal_executor_v2"
                )

                goal.progress = 0.0
                goal.execution_trace = trace

                # Phase 2.5.P: Feedback hook on failure
                try:
                    await executor_with_feedback.on_skill_execution_completed(
                        goal_id=str(goal.id),
                        goal_title=goal.title,
                        step_id=f"skill_{skill_id_normalized}",
                        success=False,
                        error=result.error
                    )
                except ExecutionMustStopException as e:
                    await transition_service.transition(
                        uow=uow,
                        goal_id=UUID(str(goal.id)),
                        new_state="frozen",
                        reason=f"Execution stopped by feedback loop: {str(e)}",
                        actor="goal_executor_v2"
                    )

                    goal.execution_trace = trace
                    return {
                        "status": "stopped_by_feedback",
                        "message": f"Execution stopped by feedback loop: {str(e)}",
                        "goal_id": str(goal.id),
                        "trace": trace,
                        "feedback": str(e.safety_level)
                    }

                return {
                    "status": "error",
                    "message": result.error,
                    "goal_id": str(goal.id),
                    "trace": trace
                }

            # Step 5: Verify result
            goal.progress = 0.6
            await self._save_goal_with_uow(uow, goal)

            verification_start = datetime.utcnow()
            is_valid = skill.verify(result)
            verification_end = datetime.utcnow()

            logger.info("artifact_verification", is_valid=is_valid)

            trace["steps"].append({
                "step": "verify_result",
                "verification_passed": is_valid,
                "duration_ms": int((verification_end - verification_start).total_seconds() * 1000)
            })

            # Step 6: Register artifacts in database
            from artifact_registry import ArtifactRegistry

            artifact_registry = ArtifactRegistry()
            registered_artifacts = []

            for artifact in result.artifacts:
                # Convert Artifact to dict format for registry
                # Determine content_kind based on artifact type if not in metadata
                content_kind_value = artifact.metadata.get("content_kind", "unknown")
                if content_kind_value == "unknown":
                    # Auto-determine content_kind based on artifact type
                    if artifact.type == "KNOWLEDGE":
                        content_kind_value = "db"
                    elif artifact.type == "DATASET":
                        content_kind_value = "file"
                    elif artifact.type in ["FILE", "REPORT"]:
                        content_kind_value = "file"
                    else:
                        content_kind_value = "db"  # Default to db for unknown types

                # Convert content to string representation for DB
                if isinstance(artifact.content, (dict, list)):
                    import json
                    content_location = json.dumps(artifact.content, ensure_ascii=False)
                elif artifact.type == "KNOWLEDGE" and content_kind_value == "db":
                    # For KNOWLEDGE artifacts stored in DB, ensure JSON serialization
                    import json
                    if isinstance(artifact.content, dict):
                        content_location = json.dumps(artifact.content, ensure_ascii=False)
                    else:
                        content_location = json.dumps({"content": str(artifact.content)}, ensure_ascii=False)
                else:
                    content_location = str(artifact.content)

                artifact_data = {
                    "artifact_type": artifact.type.upper(),
                    "content_kind": content_kind_value,
                    "content_location": content_location,
                    "domains": artifact.metadata.get("domains", []),
                    "tags": artifact.metadata.get("tags", []),
                    "skill_name": skill_id_normalized,
                    "auto_verify": True  # Auto-verify for atomic goals
                }

                registered = await artifact_registry.register_with_uow(
                    uow=uow,
                    goal_id=str(goal.id),
                    **artifact_data
                )
                registered_artifacts.append(registered)
                logger.debug("artifact_registered", artifact_type=artifact.type)

            # Step 7: Evaluate result
            logger.info("evaluating_goal_completion")

            evaluation_start = datetime.utcnow()
            evaluation_result = evaluation_engine.evaluate_goal(
                goal_completion_criteria=goal.completion_criteria,
                artifacts_produced=registered_artifacts,
                goal_title=goal.title,
                goal_description=goal.description
            )
            evaluation_end = datetime.utcnow()

            logger.info("evaluation_result", summary=evaluation_result.summary)
            logger.info("evaluation_confidence", confidence=f"{evaluation_result.confidence*100:.0f}%")

            # Record evaluation in trace
            trace["steps"].append({
                "step": "evaluate_result",
                "confidence": evaluation_result.confidence,
                "passed": evaluation_result.passed,
                "checks": evaluation_result.checks,
                "duration_ms": int((evaluation_end - evaluation_start).total_seconds() * 1000)
            })

            # Сохраняем evaluation result в goal
            goal.evaluation_result = evaluation_result.to_dict()
            goal.evaluation_confidence = evaluation_result.confidence

            # Step 8: Update goal status based on evaluation
            if evaluation_result.passed:
                # Phase 2.5.P: PRE-COMMIT CHECK (critical!)
                # Before marking as DONE, verify invariants
                try:
                    await executor_with_feedback.before_marking_done(
                        goal_id=str(goal.id),
                        goal_title=goal.title,
                        completion_mode=goal.completion_mode,
                        evaluation_passed=True
                    )
                except ExecutionMustStopException as e:
                    await transition_service.transition(
                        uow=uow,
                        goal_id=UUID(str(goal.id)),
                        new_state="frozen",
                        reason=f"Pre-commit check failed: {str(e)}",
                        actor="goal_executor_v2"
                    )

                    goal.progress = evaluation_result.confidence
                    goal.execution_trace = trace

                    return {
                        "status": "blocked_by_feedback",
                        "message": f"Cannot mark as DONE: {str(e)}",
                        "goal_id": str(goal.id),
                        "trace": trace,
                        "feedback": str(e.safety_level),
                        "invariant_violation": True
                    }

                await transition_service.transition(
                    uow=uow,
                    goal_id=UUID(str(goal.id)),
                    new_state="done",
                    reason="Atomic goal execution complete with safety check passed",
                    actor="goal_executor_v2"
                )

                logger.info("goal_completed", goal_id=str(goal.id))

                # Extract artifact IDs for event emission
                artifact_ids = [str(a.get("artifact_id", str(i))) for i, a in enumerate(registered_artifacts)]
                
                # Phase 2.5.P: Post-completion hook (observer + reflection)
                try:
                    await executor_with_feedback.on_goal_completed(
                        goal_id=str(goal.id),
                        goal_title=goal.title,
                        steps_total=1,
                        steps_completed=1,
                        steps_failed=0,
                        artifacts=artifact_ids,
                        completion_mode=goal.completion_mode
                    )
                except Exception as e:
                    # Log but don't fail (already marked as done)
                    logger.warning("post_completion_hook_error", error=str(e))

            else:
                to_state = "incomplete" if evaluation_result.confidence > 0.3 else "blocked"

                await transition_service.transition(
                    uow=uow,
                    goal_id=UUID(str(goal.id)),
                    new_state=to_state,
                    reason=f"Evaluation confidence {evaluation_result.confidence:.2f} - needs human review",
                    actor="goal_executor_v2"
                )

                logger.warning("goal_state_changed", new_state=to_state.upper())

                # Phase 2.5.P: Hook on incomplete goal
                try:
                    await executor_with_feedback.on_goal_failed(
                        goal_id=str(goal.id),
                        goal_title=goal.title,
                        steps_total=1,
                        steps_completed=0,
                        failure_reason=f"Evaluation failed (confidence: {evaluation_result.confidence:.2f})",
                        error_type="EvaluationFailed",
                        error_message=f"Goal did not meet completion criteria"
                    )
                except Exception as e:
                    # Log but don't fail
                    logger.warning("incomplete_hook_error", error=str(e))

            # Finalize trace
            execution_end = datetime.utcnow()
            goal.execution_completed_at = execution_end
            trace["completed_at"] = execution_end.isoformat()
            trace["total_duration_ms"] = int((execution_end - execution_start).total_seconds() * 1000)
            trace["final_status"] = goal.status
            trace["final_progress"] = goal.progress

            # Add explainability
            trace["explainability"] = self._generate_explanation(trace, evaluation_result)

            goal.execution_trace = trace
            await self._save_goal_with_uow(uow, goal)

            # PHASE 1: Complete execution record on success
            execution_end = datetime.utcnow()
            execution_rec.completed_at = execution_end
            execution_rec.duration_ms = int((execution_end - execution_start).total_seconds() * 1000)
            execution_rec.success = True
            execution_rec.confidence = evaluation_result.confidence
            execution_rec.artifacts_count = len(registered_artifacts)

            # Save execution record to database
            from infrastructure.execution_repositories import GoalExecutionRepository, SkillStatsRepository

            executions_repo = GoalExecutionRepository()
            await executions_repo.add(uow.session, execution_rec)

            # Update skill statistics
            skill_stats_repo = SkillStatsRepository()
            await skill_stats_repo.update_from_execution(
                uow.session,
                skill_id_normalized,
                execution_rec
            )

            logger.info(
                "execution_recorded",
                execution_id=str(execution_rec.execution_id),
                skill_id=skill_id_normalized,
                duration_ms=execution_rec.duration_ms,
                success=True
            )

            # =================================================================
            # RECORD EXPERIENCE - THIS IS THE LEARNING MOMENT
            # =================================================================
            # NOTE: Records in separate transaction via fire-and-forget
            # Using asyncio.create_task() - if it fails, execution continues
            task_type = self._infer_task_type(goal)

            try:
                # Schedule experience recording in background
                # This may fail in Celery workers, but won't block execution
                import asyncio
                asyncio.create_task(
                    self._record_experience_async(
                        goal_id=goal.id,
                        task_type=task_type,
                        skill_id=skill_id_normalized,
                        success=True,
                        confidence=evaluation_result.confidence,
                        latency_ms=execution_rec.duration_ms,
                        goal_title=goal.title,
                        goal_type=goal.goal_type,
                        artifacts_count=len(registered_artifacts)
                    )
                )
                logger.debug(
                    "experience_recording_scheduled",
                    goal_id=str(goal.id),
                    task_type=task_type,
                    skill_id=skill_id_normalized
                )

                # AUTO-LEARNING: Trigger capability gap detection and resolution
                # This is where system becomes self-improving
                asyncio.create_task(
                    self._auto_improve_capabilities(goal.id)
                )
                logger.debug(
                    "auto_improvement_scheduled",
                    goal_id=str(goal.id)
                )
            except Exception as exp_error:
                # Don't fail execution if experience recording fails
                logger.warning(
                    "experience_recording_failed",
                    error=str(exp_error),
                    goal_id=str(goal.id)
                )

            # Add content preview to artifacts for immediate viewing
            artifacts_with_preview = []
            for artifact in registered_artifacts:
                artifact_with_preview = dict(artifact)
                # Add preview for FILE artifacts
                if artifact.get("content_kind") == "file" and artifact.get("content_location"):
                    try:
                        import os
                        file_path = artifact["content_location"]
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                preview = f.read(500)
                                artifact_with_preview["content_preview"] = preview
                    except Exception as e:
                        logger.warning("preview_read_failed", error=str(e))
                        artifact_with_preview["content_preview"] = None
                artifacts_with_preview.append(artifact_with_preview)

            return {
                "status": "success",
                "goal_id": str(goal.id),
                "goal_status": goal.status,
                "skill_used": skill_id_normalized,
                "artifacts_produced": len(registered_artifacts),
                "artifacts": artifacts_with_preview,
                "verification_passed": is_valid,
                "evaluation": evaluation_result.to_dict(),
                "evaluation_passed": evaluation_result.passed,
                "confidence": evaluation_result.confidence,
                "trace": trace
            }

        except Exception as e:
            logger.error("execution_error", error=str(e), exc_info=True)

            # PHASE 1: Complete execution record on failure
            execution_end = datetime.utcnow()
            execution_rec.completed_at = execution_end
            execution_rec.duration_ms = int((execution_end - execution_start).total_seconds() * 1000) if execution_start else None
            execution_rec.success = False
            execution_rec.error_message = str(e)[:1000]  # Truncate long errors
            execution_rec.error_type = type(e).__name__
            execution_rec.artifacts_count = 0

            # Try to save execution record (may fail if transaction is broken)
            try:
                from infrastructure.execution_repositories import GoalExecutionRepository, SkillStatsRepository

                # Get skill_id if skill was selected
                if 'skill' in locals() and skill:
                    # Get skill_id even in exception handler
                    skill_id_from_exception = normalize_skill_id(skill)

                    executions_repo = GoalExecutionRepository()
                    await executions_repo.add(uow.session, execution_rec)

                    # Update skill statistics even on failure
                    skill_stats_repo = SkillStatsRepository()
                    await skill_stats_repo.update_from_execution(
                        uow.session,
                        skill_id_from_exception,
                        execution_rec
                    )

                    logger.info(
                        "execution_recorded_failure",
                        execution_id=str(execution_rec.execution_id),
                        skill_id=skill_id_from_exception,
                        error_type=execution_rec.error_type
                    )

                    # =================================================================
                    # RECORD FAILED EXPERIENCE - ALSO LEARNING MOMENT
                    # =================================================================
                    task_type = self._infer_task_type(goal)

                    try:
                        import asyncio
                        asyncio.create_task(
                            self._record_experience_async(
                                goal_id=goal.id,
                                task_type=task_type,
                                skill_id=skill_id_from_exception,
                                success=False,
                                confidence=0.0,
                                latency_ms=execution_rec.duration_ms or 0,
                                goal_title=goal.title,
                                goal_type=goal.goal_type,
                                artifacts_count=0,
                                error_type=execution_rec.error_type,
                                error_message=execution_rec.error_message
                            )
                        )
                    except Exception as exp_error:
                        logger.warning(
                            "failed_experience_recording_failed",
                            error=str(exp_error),
                            goal_id=str(goal.id)
                        )
            except Exception as db_error:
                logger.warning(
                    "failed_to_record_execution",
                    error=str(db_error)
                )

            await transition_service.transition(
                uow=uow,
                goal_id=UUID(str(goal.id)),
                new_state="blocked",
                reason=f"Execution error: {str(e)[:200]}",
                actor="goal_executor_v2"
            )

            goal.progress = 0.0

            return {
                "status": "error",
                "message": str(e),
                "goal_id": str(goal.id)
            }

    async def _record_experience_async(
        self,
        goal_id,
        task_type,
        skill_id,
        success,
        confidence,
        latency_ms,
        goal_title,
        goal_type,
        artifacts_count,
        error_type=None,
        error_message=None
    ):
        """
        Record experience in separate transaction (async background task).

        This runs AFTER UoW commits to avoid barrier violation.
        """
        try:
            from infrastructure.uow import create_uow_provider

            uow_provider = create_uow_provider()
            async with uow_provider() as uow:
                await experience_engine.record_experience(
                    session=uow.session,
                    goal_id=goal_id,
                    task_type=task_type,
                    skill_id=skill_id,
                    success=success,
                    confidence=confidence,
                    latency_ms=latency_ms,
                    error_type=error_type,
                    error_message=error_message,
                    extra_metadata={
                        "goal_title": goal_title,
                        "goal_type": goal_type,
                        "artifacts_produced": artifacts_count
                    }
                )
                logger.debug(
                    "experience_recorded_async",
                    goal_id=str(goal_id),
                    skill_id=skill_id,
                    success=success
                )
        except Exception as e:
            # Don't fail execution if experience recording fails
            logger.warning(
                "experience_recording_async_failed",
                error=str(e),
                goal_id=str(goal_id)
            )

    async def _auto_improve_capabilities(self, goal_id):
        """
        AUTO-LEARNING: Detect and resolve capability gaps after goal execution.

        This is where AI-OS becomes self-improving:
        1. Analyze goal for missing capabilities
        2. Auto-resolve pipeline gaps
        3. Update capability graph

        Args:
            goal_id: UUID of executed goal
        """
        try:
            from capability import capability_gap_engine
            from models import Goal
            from database import AsyncSessionLocal
            from sqlalchemy import select

            # Load goal
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == goal_id)
                result = await db.execute(stmt)
                goal = result.scalar_one_or_none()

                if not goal:
                    logger.debug("auto_improve_skip", goal_id=str(goal_id), reason="goal_not_found")
                    return

            # Detect gaps
            gaps = await capability_gap_engine.analyze_goal_for_gaps(goal)

            if not gaps:
                logger.debug(
                    "auto_improve_no_gaps",
                    goal_id=str(goal_id),
                    goal_title=goal.title
                )
                return

            # Auto-resolve pipeline gaps
            resolved_count = 0
            for gap in gaps:
                if gap.gap_type == "pipeline":  # Only auto-resolve pipeline gaps
                    resolution = await capability_gap_engine.resolve_gap(gap.gap_id)
                    if resolution and resolution.success:
                        resolved_count += 1

            logger.info(
                "auto_improvement_completed",
                goal_id=str(goal_id),
                goal_title=goal.title,
                gaps_detected=len(gaps),
                gaps_resolved=resolved_count
            )

        except Exception as e:
            # Don't fail if auto-improvement fails
            logger.warning(
                "auto_improvement_failed",
                goal_id=str(goal_id),
                error=str(e)
            )

    def _parse_requirements(self, goal: Goal) -> dict:
        """Парсит требования из completion_criteria и goal title"""
        requirements = {
            "artifacts": [],
            "capabilities": []
        }

        # Extract artifacts from completion_criteria
        if goal.completion_criteria:
            criteria = goal.completion_criteria
            artifacts_req = criteria.get("artifacts_required", [])

            for req in artifacts_req:
                if isinstance(req, dict):
                    req_type = req.get("type")
                    if req_type:
                        requirements["artifacts"].append(req_type.upper())
                elif isinstance(req, str):
                    requirements["artifacts"].append(req.upper())

        # 🔑 Infer capabilities from goal title and description
        title_lower = goal.title.lower()
        desc_lower = (goal.description or "").lower()

        # Research capabilities
        if any(word in title_lower or word in desc_lower for word in ["research", "search", "find", "explore", "discover", "investigate"]):
            requirements["capabilities"].append("research")
            requirements["capabilities"].append("web-research")

        # Write/Create capabilities
        if any(word in title_lower or word in desc_lower for word in ["write", "create", "generate", "produce", "make"]):
            requirements["capabilities"].append("write")
            requirements["capabilities"].append("file-production")

        # Analyze/Summarize capabilities
        if any(word in title_lower or word in desc_lower for word in ["summarize", "analyze", "condense", "summary", "analysis"]):
            requirements["capabilities"].append("analysis")
            requirements["capabilities"].append("summarization")

        # Plan/Structure capabilities
        if any(word in title_lower or word in desc_lower for word in ["plan", "structure", "design", "schema"]):
            requirements["capabilities"].append("planning")
            requirements["capabilities"].append("structured-generation")

        # Verify/Check capabilities (but NOT for research goals)
        is_research_goal = any(word in title_lower for word in ["research", "search", "find"])

        if not is_research_goal and any(word in title_lower or word in desc_lower for word in ["verify", "check", "validate", "inspect"]):
            requirements["capabilities"].append("verification")

        # Test capability - ONLY for explicit test goals
        if ("test" in title_lower or "echo" in title_lower) and "research" not in title_lower:
            requirements["capabilities"].append("test")

        return requirements

    def _infer_task_type(self, goal: Goal) -> str:
        """
        Infer task type from goal for experience tracking.

        Task type is CRITICAL for skill learning and comparison.
        Examples: web_search, summarization, write_file, analysis.

        Args:
            goal: The goal to classify

        Returns:
            Task type string
        """
        title_lower = goal.title.lower()
        desc_lower = (goal.description or "").lower()
        combined = title_lower + " " + desc_lower

        # Priority classification (most specific first)
        task_keywords = {
            "web_search": ["search", "find information", "web research", "google", "lookup"],
            "summarization": ["summarize", "summary", "condense", "brief"],
            "write_file": ["write", "create file", "generate", "produce content"],
            "analysis": ["analyze", "analysis", "evaluate", "assess"],
            "fetch_url": ["fetch", "retrieve", "get url", "download"],
            "test": ["test", "echo", "ping", "verify connection"],
            "planning": ["plan", "design", "structure", "schema"]
        }

        # Find first matching task type
        for task_type, keywords in task_keywords.items():
            if any(keyword in combined for keyword in keywords):
                return task_type

        # Fallback: use goal_type if no specific task matches
        return goal.goal_type or "general"

    async def _generate_content_with_llm(self, goal: Goal) -> str:
        """
        Generate content for goal execution using LLM.

        Args:
            goal: The goal to generate content for

        Returns:
            Generated content as string
        """
        import os
        model = os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile")

        prompt = f"""You are executing a goal. Generate the actual content that accomplishes this goal.

Goal: {goal.title}
Description: {goal.description or 'No description provided'}
Type: {goal.goal_type}
Level: L{goal.depth_level}

Generate a document that accomplishes this goal. Be specific, actionable, and thorough.
Do NOT include meta-commentary like "This is a test file" or "Generated by...".
Just provide the actual content.

Format: Markdown"""

        try:
            response = await chat_with_fallback(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response["choices"][0]["message"]["content"].strip()
            logger.info("llm_content_generated", chars=len(content))
            return content

        except Exception as e:
            logger.warning("llm_generation_failed_using_fallback", error=str(e))
            # Fallback to basic template
            return f"""# {goal.title}

**Type**: {goal.goal_type}
**Level**: L{goal.depth_level}
**Description**: {goal.description or 'No description provided'}

⚠️ LLM generation was unavailable. This is a basic template.
"""

    async def _prepare_inputs(self, goal: Goal, skill: Skill) -> dict:
        """
        Готовит входные данные для skill (now async for LLM support)
        """
        skill_id = normalize_skill_id(skill)

        # EchoSkill inputs
        if skill_id == "core.echo":
            return {
                "text": goal.title
            }

        # WriteFileSkill inputs - use LLM to generate real content
        if skill_id == "core.writefile":
            filename = f"{goal.title.lower().replace(' ', '_')}.md"

            # Generate content using LLM
            logger.info("generating_content_with_llm", goal_title=goal.title)
            text = await self._generate_content_with_llm(goal)

            import os
            return {
                "text": text,
                "filename": filename,
                "directory": os.getenv("ARTIFACTS_PATH", "/data/artifacts")
            }

        # WebResearchSkill inputs
        if skill_id == "core.web_research":
            # Extract keywords from goal title and description
            import re

            # Remove common words and extract key terms
            title_lower = goal.title.lower()
            desc_lower = (goal.description or "").lower()

            # Extract keywords (simple approach)
            keywords = []

            # Add main topic from title
            if "ai" in title_lower or "artificial intelligence" in title_lower:
                keywords.append("artificial intelligence")
            if "machine learning" in title_lower or "machine learning" in desc_lower:
                keywords.append("machine learning")
            if "research" in title_lower:
                keywords.append("research")
            if "latest" in title_lower or "recent" in title_lower:
                keywords.append("latest")

            # Fallback: use title words
            if not keywords:
                words = re.findall(r'\b[a-z]{3,}\b', title_lower)
                # Filter out common words
                stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'with', 'that', 'this'}
                keywords = [w for w in words if w not in stopwords][:5]

            return {
                "keywords": keywords if keywords else ["AI", "developments"]
            }

        # Default
        return {}

    async def _save_goal_with_uow(self, uow: "UnitOfWork", goal: Goal):
        """
        Saves goal within an existing UoW transaction.

        NOTE: Status changes MUST go through transition_service.transition().
        This method only saves non-status fields (progress, traces, timestamps).

        Args:
            uow: UnitOfWork with active transaction
            goal: Goal to save (must be already attached to session)
        """
        await self._repo.update(uow.session, goal)

    def _explain_skill_selection(self, skill: Skill, requirements: dict) -> str:
        """Объясняет почему был выбран этот skill"""
        reasons = []

        # Capability match
        capabilities = requirements.get("capabilities", [])
        skill_capabilities = skill.capabilities

        matching_caps = set(capabilities) & set(skill_capabilities)
        if matching_caps:
            reasons.append(f"Capability match: {list(matching_caps)}")

        # Artifact type match
        artifacts = requirements.get("artifacts", [])
        produces = skill.produces_artifacts

        matching_artifacts = set(artifacts) & set([a.lower() for a in produces])
        if matching_artifacts:
            reasons.append(f"Artifact match: {list(matching_artifacts)}")

        if not reasons:
            return f"Selected as best match (capabilities: {skill_capabilities}, produces: {produces})"

        return ". ".join(reasons)

    def _generate_explanation(self, trace: dict, evaluation_result) -> dict:
        """Генерирует объяснение выполнения goal"""
        explanation = {
            "what_happened": [],
            "why_incomplete": None,
            "recommendations": []
        }

        # Explain what happened
        for step in trace.get("steps", []):
            step_name = step.get("step", "")
            if step_name == "skill_selection":
                if step.get("success"):
                    explanation["what_happened"].append(
                        f"Selected skill: {step.get('skill_selected')} "
                        f"({step.get('selection_reason')})"
                    )
            elif step_name == "execute_skill":
                if step.get("success"):
                    explanation["what_happened"].append(
                        f"Executed skill successfully ({step.get('duration_ms')}ms, "
                        f"{step.get('artifacts_produced')} artifacts)"
                    )
                else:
                    explanation["what_happened"].append(
                        f"Skill execution failed: {step.get('error', 'Unknown error')}"
                    )
            elif step_name == "evaluate_result":
                conf = step.get("confidence", 0) * 100
                explanation["what_happened"].append(
                    f"Evaluation: {conf:.0f}% confidence ({'PASS' if step.get('passed') else 'FAIL'})"
                )

        # Explain why incomplete (if applicable)
        if evaluation_result and not evaluation_result.passed:
            failed_checks = [
                name for name, check in evaluation_result.checks.items()
                if not check.get("passed", False)
            ]

            if failed_checks:
                explanation["why_incomplete"] = (
                    f"Failed checks: {', '.join(failed_checks)}. "
                )
            else:
                explanation["why_incomplete"] = "Confidence below threshold (60%)"

            # Add recommendations
            checks = evaluation_result.checks
            if "artifact_count" in checks:
                check = checks["artifact_count"]
                if not check.get("passed"):
                    expected = check.get("expected_min", 1)
                    actual = check.get("actual", 0)
                    explanation["recommendations"].append(
                        f"Produce {expected} artifacts (currently {actual})"
                    )

            if "artifact_types" in checks:
                check = checks["artifact_types"]
                if not check.get("passed"):
                    missing = check.get("missing_types", [])
                    if missing:
                        explanation["recommendations"].append(
                            f"Add artifacts of types: {', '.join(missing)}"
                        )

        return explanation


# Global instance
goal_executor_v2 = GoalExecutorV2()
