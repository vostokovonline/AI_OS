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
from database import AsyncSessionLocal
from models import Goal

# Import canonical skill system
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from canonical_skills.base import Skill, SkillResult, Artifact
from canonical_skills.registry import skill_registry
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

    def __init__(self):
        # Initialize skills
        self._init_skills()

    def _init_skills(self):
        """Инициализация и регистрация навыков"""
        # Register core skills
        skill_registry.register(EchoSkill())
        skill_registry.register(WriteFileSkill())

        # Register web_research skill with lazy import to avoid circular dependencies
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

    async def execute_goal(self, goal_id: str, session_id: Optional[str] = None) -> dict:
        """
        Выполняет цель используя Skills (legacy wrapper - creates own UoW).

        DEPRECATED: Use execute_goal_with_uow() instead for proper transaction management.

        Args:
            goal_id: ID цели
            session_id: ID сессии

        Returns:
            Результат выполнения с artifacts
        """
        from infrastructure.uow import create_uow_provider

        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
            return await self.execute_goal_with_uow(uow, goal_id, session_id)

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
        repo = GoalRepository()

        goal = await repo.get(uow.session, UUID(goal_id))

        if not goal:
            return {"status": "error", "message": "Goal not found"}

        if not goal.is_atomic:
            raise ValueError(
                f"V2 only handles atomic goals. Goal '{goal.title}' is not atomic. "
                f"Use Orchestrator (V1) for complex goals."
            )

        return await self._execute_atomic_goal_with_uow(uow, goal, session_id)

    async def _execute_atomic_goal(self, goal: Goal, session_id: Optional[str]) -> dict:
        """
        DEPRECATED: Use _execute_atomic_goal_with_uow() instead.

        Executes atomic goal via Skills with self-managed transactions.
        """
        from infrastructure.uow import create_uow_provider

        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
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

        try:
            # Step 1: Parse requirements
            requirements = self._parse_requirements(goal)
            logger.debug("goal_requirements", requirements=requirements)

            trace["steps"].append({
                "step": "parse_requirements",
                "requirements": requirements
            })

            # Step 2: Select skill
            skill = self._select_skill(requirements, goal)
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
            trace["steps"].append({
                "step": "skill_selection",
                "success": True,
                "skill_selected": skill.id,
                "skill_version": skill.version,
                "selection_reason": self._explain_skill_selection(skill, requirements),
                "skill_capabilities": skill.capabilities,
                "artifacts_produced_by_skill": skill.produces_artifacts
            })

            logger.info("skill_selected", skill_id=skill.id)

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
                "skill_id": skill.id,
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
                        step_id=f"skill_{skill.id}",
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
                    "skill_name": skill.id,
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

                # Phase 2.5.P: Post-completion hook (observer + reflection)
                try:
                    await executor_with_feedback.on_goal_completed(
                        goal_id=str(goal.id),
                        goal_title=goal.title,
                        steps_total=1,
                        steps_completed=1,
                        steps_failed=0,
                        artifacts=registered_artifacts,
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
                "skill_used": skill.id,
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

    def _select_skill(self, requirements: dict, goal: Goal) -> Optional[Skill]:
        """
        Выбирает подходящий skill по требованиям

        Priority:
        1. Direct capability match (exact)
        2. AUTO-GENERATE NEW SKILL (for research/complex)
        3. Artifact type match (simple)
        4. Default fallback
        """
        capabilities = requirements.get("capabilities", [])
        artifacts = requirements.get("artifacts", [])

        # 🔑 PRIORITY 1: Exact capability match
        for capability in capabilities:
            skills = skill_registry.find_by_capability(capability)
            if skills:
                logger.debug("skill_found_by_capability", capability=capability)
                return skills[0]

        # 🔑 PRIORITY 2: Check if we need to GENERATE new skill
        has_research = any(cap in capabilities for cap in ["research", "web-research"])
        has_knowledge = "KNOWLEDGE" in artifacts
        is_complex_goal = has_research or has_knowledge

        if is_complex_goal:
            logger.warning("no_suitable_skill_found")
            logger.info("complex_goal_detected_attempting_generation")

            try:
                from skill_generator import skill_generator

                generation_result = skill_generator.generate_skill(
                    requirements=requirements,
                    goal_context={
                        "title": goal.title,
                        "description": goal.description,
                        "goal_type": goal.goal_type
                    }
                )

                if generation_result.get("success"):
                    skill_id = generation_result["skill_id"]
                    new_skill = skill_registry.get(skill_id)

                    if new_skill:
                        logger.info("auto_generated_skill", skill_id=skill_id)
                        return new_skill
                    else:
                        logger.warning("skill_generated_not_found_in_registry")
                else:
                    logger.error("skill_generation_failed", error=generation_result.get('error'))

            except Exception as e:
                logger.error("skill_generation_error", error=str(e), exc_info=True)

        # 🔑 PRIORITY 3: Simple artifact type match
        for artifact_type in artifacts:
            if artifact_type in ["FILE", "KNOWLEDGE", "DATASET"]:
                skills = skill_registry.find_by_artifact(artifact_type.lower())
                if skills:
                    logger.debug("skill_found_by_artifact_type", artifact_type=artifact_type)
                    return skills[0]

        # 🔑 PRIORITY 4: Fallbacks
        if "test" in goal.title.lower():
            return skill_registry.get("core.echo")

        return skill_registry.get("core.write_file")

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
        # EchoSkill inputs
        if skill.id == "core.echo":
            return {
                "text": goal.title
            }

        # WriteFileSkill inputs - use LLM to generate real content
        if skill.id == "core.write_file":
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
        if skill.id == "core.web_research":
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

    async def _save_goal(self, goal: Goal):
        """DEPRECATED: Saves goal in its own transaction. Use _save_goal_with_uow() instead."""
        from infrastructure.uow import create_uow_provider

        uow_provider = create_uow_provider()
        async with uow_provider() as uow:
            await self._save_goal_with_uow(uow, goal)

    async def _save_goal_with_uow(self, uow: "UnitOfWork", goal: Goal):
        """
        Saves goal within an existing UoW transaction.

        NOTE: Status changes MUST go through transition_service.transition().
        This method only saves non-status fields (progress, traces, timestamps).

        Args:
            uow: UnitOfWork with active transaction
            goal: Goal to save (must be already attached to session)
        """
        repo = GoalRepository()
        await repo.update(uow.session, goal)

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
