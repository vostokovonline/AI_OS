"""
VECTOR ENGINE SERVICE
Transformation operators for Goals/Plans/Tasks

Key principle: Vector is NOT a hierarchy level
Vector is an operator: V(x) = x'

Architectural guarantee:
- Applies to: Goal, Plan, Task
- NEVER applies to: Legacy Axis (S0)
- Stateless: no persistence between applications
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import select
from database import AsyncSessionLocal
from models import VectorOperator, VectorApplication, Goal


class VectorEngineError(Exception):
    """Raised when Vector Engine constraint is violated"""
    pass


class VectorEngineService:
    """
    Vector Engine - Applies transformation operators to Goals/Plans/Tasks

    Key principles:
    - Vector V transforms target T: V(T) = T'
    - Vector is stateless (no memory between applications)
    - Vector CANNOT apply to Legacy Axis (S0)
    - Applications are logged for audit only
    """

    VALID_TARGET_TYPES = ["goal", "plan", "task"]
    FORBIDDEN_TARGET_TYPES = ["legacy_axis"]

    async def apply_vector(
        self,
        vector_id: str,
        target_type: str,
        target_id: str,
        applied_by: str = "system"
    ) -> Dict:
        """
        Apply vector to target

        Args:
            vector_id: Vector operator ID
            target_type: Type of target (goal, plan, task)
            target_id: ID of target
            applied_by: Who applied the vector

        Returns:
            {
                "application_id": "...",
                "input_snapshot": {...},
                "output_snapshot": {...},
                "transformation": {...}
            }

        Raises:
            VectorEngineError: if validation fails
        """
        # OCCP Step 1: Check MCL allows this operation
        from mcl_service import mcl_service

        mcl_check = await mcl_service.check_operation_allowed(
            operation="vector.apply",
            component="vector_engine"
        )

        if not mcl_check["allowed"]:
            raise VectorEngineError(
                f"MCL veto: {mcl_check['reason']}"
            )

        # OCCP Step 2: Check SK doesn't veto
        from sk_service import sk_service

        sk_check = await sk_service.check_veto(
            operation="vector.apply",
            component="vector_engine",
            context={"vector_id": vector_id, "target_id": target_id}
        )

        if sk_check["vetoed"]:
            raise VectorEngineError(
                f"SK veto: {sk_check['explanation']}"
            )

        # Validation 1: Target type
        if target_type not in self.VALID_TARGET_TYPES:
            raise VectorEngineError(
                f"Invalid target_type: {target_type}. "
                f"Must be one of: {self.VALID_TARGET_TYPES}"
            )

        # Validation 2: Forbidden targets (Legacy Axis protection)
        if target_type in self.FORBIDDEN_TARGET_TYPES:
            raise VectorEngineError(
                f"Vector CANNOT apply to {target_type}. "
                f"Legacy Axis (S0) is protected from vector transformation."
            )

        async with AsyncSessionLocal() as db:
            # Load vector
            stmt_vec = select(VectorOperator).where(VectorOperator.id == uuid.UUID(vector_id))
            result_vec = await db.execute(stmt_vec)
            vector = result_vec.scalar_one_or_none()

            if not vector:
                raise VectorEngineError(f"Vector not found: {vector_id}")

            if not vector.is_active:
                raise VectorEngineError(f"Vector is not active: {vector.name}")

            # Load target (for now, only goals are supported)
            if target_type == "goal":
                stmt_target = select(Goal).where(Goal.id == uuid.UUID(target_id))
                result_target = await db.execute(stmt_target)
                target = result_target.scalar_one_or_none()

                if not target:
                    raise VectorEngineError(f"Goal not found: {target_id}")

                # Check if goal is Legacy-like (depth 0, directional)
                if target.depth_level == 0 and target.goal_type == "directional":
                    raise VectorEngineError(
                        f"Vector CANNOT apply to Legacy-like goals. "
                        f"Goal {target_id} is depth 0 directional (S0)."
                    )

                # Create input snapshot
                input_snapshot = {
                    "id": str(target.id),
                    "title": target.title,
                    "description": target.description,
                    "goal_type": target.goal_type,
                    "depth_level": target.depth_level
                }

                # Apply transformation
                output_snapshot = await self._transform_goal(target, vector)

                # Log application
                application = VectorApplication(
                    vector_id=vector.id,
                    target_type=target_type,
                    target_id=target.id,
                    input_snapshot=input_snapshot,
                    output_snapshot=output_snapshot,
                    applied_by=applied_by
                )

                db.add(application)
                await db.commit()
                await db.refresh(application)

                return {
                    "application_id": str(application.id),
                    "vector_name": vector.name,
                    "vector_type": vector.vector_type,
                    "input_snapshot": input_snapshot,
                    "output_snapshot": output_snapshot,
                    "transformation": {
                        "rules_applied": vector.transformation_rules,
                        "applied_at": application.applied_at.isoformat()
                    }
                }

            else:
                raise VectorEngineError(
                    f"Target type {target_type} not yet implemented. "
                    f"Only 'goal' is supported currently."
                )

    async def _transform_goal(self, goal: Goal, vector: VectorOperator) -> Dict:
        """
        Transform goal using vector

        This is a simplified version - in production, this would use LLM
        to actually rewrite the goal according to vector's transformation rules.

        For now, returns a modified snapshot with vector's bias applied.
        """
        rules = vector.transformation_rules

        # Simulate transformation (in production: use LLM here)
        modified_title = goal.title
        modified_description = goal.description or ""

        # Apply language style bias
        language_style = rules.get("language_style", "neutral")
        if language_style == "story_driven":
            modified_description = f"[Narrative approach] {modified_description}"
        elif language_style == "precise":
            modified_description = f"[Technical approach] {modified_description}"
        elif language_style == "concise":
            modified_description = f"[Minimal] {modified_description}"
        elif language_style == "critical":
            modified_description = f"[Adversarial review] {modified_description}"

        # Apply priority bias
        priority_bias = rules.get("priority_bias", [])
        if priority_bias:
            modified_description += f"\n\nPriority focus: {', '.join(priority_bias)}"

        return {
            "id": str(goal.id),
            "title": modified_title,
            "description": modified_description,
            "goal_type": goal.goal_type,
            "depth_level": goal.depth_level,
            "vector_applied": vector.name,
            "transformation_type": vector.vector_type
        }

    async def list_vectors(self, active_only: bool = True) -> List[Dict]:
        """
        List all available vectors

        Args:
            active_only: Only return active vectors

        Returns:
            List of vectors
        """
        async with AsyncSessionLocal() as db:
            stmt = select(VectorOperator)

            if active_only:
                stmt = stmt.where(VectorOperator.is_active == True)

            stmt = stmt.order_by(VectorOperator.name)

            result = await db.execute(stmt)
            vectors = result.scalars().all()

            return [
                {
                    "id": str(v.id),
                    "name": v.name,
                    "description": v.description,
                    "vector_type": v.vector_type,
                    "transformation_rules": v.transformation_rules,
                    "forbidden_targets": v.forbidden_targets,
                    "created_at": v.created_at.isoformat() if v.created_at else None
                }
                for v in vectors
            ]

    async def get_vector(self, vector_id: str) -> Optional[Dict]:
        """
        Get vector by ID

        Args:
            vector_id: Vector ID

        Returns:
            Vector or None
        """
        async with AsyncSessionLocal() as db:
            stmt = select(VectorOperator).where(VectorOperator.id == uuid.UUID(vector_id))
            result = await db.execute(stmt)
            vector = result.scalar_one_or_none()

            if not vector:
                return None

            return {
                "id": str(vector.id),
                "name": vector.name,
                "description": vector.description,
                "vector_type": vector.vector_type,
                "transformation_rules": vector.transformation_rules,
                "forbidden_targets": vector.forbidden_targets,
                "created_at": vector.created_at.isoformat() if vector.created_at else None,
                "is_active": vector.is_active
            }

    async def get_application_history(
        self,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Get vector application history (audit log)

        Args:
            target_type: Filter by target type
            target_id: Filter by target ID
            limit: Max results

        Returns:
            List of applications
        """
        async with AsyncSessionLocal() as db:
            stmt = select(VectorApplication)

            if target_type:
                stmt = stmt.where(VectorApplication.target_type == target_type)

            if target_id:
                stmt = stmt.where(VectorApplication.target_id == uuid.UUID(target_id))

            stmt = stmt.order_by(VectorApplication.applied_at.desc()).limit(limit)

            result = await db.execute(stmt)
            applications = result.scalars().all()

            return [
                {
                    "id": str(app.id),
                    "vector_id": str(app.vector_id),
                    "target_type": app.target_type,
                    "target_id": str(app.target_id),
                    "input_snapshot": app.input_snapshot,
                    "output_snapshot": app.output_snapshot,
                    "applied_at": app.applied_at.isoformat() if app.applied_at else None,
                    "applied_by": app.applied_by
                }
                for app in applications
            ]

    def _to_dict(self, vector: VectorOperator) -> Dict:
        """Convert VectorOperator to dict"""
        return {
            "id": str(vector.id),
            "name": vector.name,
            "description": vector.description,
            "vector_type": vector.vector_type,
            "transformation_rules": vector.transformation_rules,
            "forbidden_targets": vector.forbidden_targets,
            "created_at": vector.created_at.isoformat() if vector.created_at else None,
            "is_active": vector.is_active
        }


# Global instance
vector_engine_service = VectorEngineService()
