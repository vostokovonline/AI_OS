"""
LEGACY AXIS SERVICE
Constitutional layer management for AI_OS

This service manages Legacy Axis (S0) - the existential mission layer
that defines WHY the system exists, not WHAT it does.

Key principles:
- Legacy Axis is separate from Goal system
- Cannot be deleted, completed, optimized
- Survives without author (institutions, culture, knowledge)
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from sqlalchemy import select
from database import AsyncSessionLocal
from models import LegacyAxis, Goal


class LegacyAxisService:
    """
    Service for managing Legacy Axis (S0)

    Architectural guarantee:
    - Legacy Axis ≠ Goal
    - Goals are DERIVED FROM Legacy Axis
    - Legacy Axis NEVER participates in decomposition
    """

    VALID_AXIS_TYPES = ["civilizational", "cultural", "technological", "existential"]

    DEFAULT_IMMUTABILITY_POLICY = {
        "cannot_be_deleted": True,
        "cannot_be_completed": True,
        "cannot_have_deadline": True,
        "cannot_be_optimized": True
    }

    DEFAULT_SURVIVABILITY_POLICY = {
        "institutions": False,
        "cultural_artifacts": False,
        "knowledge_systems": False,
        "requires_author": False  # survives without author
    }

    DEFAULT_OPTIMIZATION_CONSTRAINTS = {
        "forbidden_metrics": ["ROI", "velocity", "completion_rate", "profit_only"],
        "allowed_metrics": ["generational_impact", "cultural_resonance", "long_term_survival"]
    }

    async def create(
        self,
        title: str,
        description: str,
        axis_type: str,
        generational_depth: int = 1,
        survivability_policy: Optional[Dict] = None,
        immutability_policy: Optional[Dict] = None,
        optimization_constraints: Optional[Dict] = None
    ) -> Dict:
        """
        Create new Legacy Axis

        Args:
            title: Title of legacy axis
            description: Description
            axis_type: Type (civilizational, cultural, technological, existential)
            generational_depth: Number of generations impacted (1+)
            survivability_policy: Custom survivability policy
            immutability_policy: Custom immutability policy
            optimization_constraints: Custom optimization constraints

        Returns:
            Created Legacy Axis
        """
        # Validation
        if axis_type not in self.VALID_AXIS_TYPES:
            raise ValueError(f"Invalid axis_type: {axis_type}")

        if generational_depth < 1:
            raise ValueError(f"generational_depth must be >= 1, got {generational_depth}")

        # Merge with defaults
        final_survivability = {**self.DEFAULT_SURVIVABILITY_POLICY, **(survivability_policy or {})}
        final_immutability = {**self.DEFAULT_IMMUTABILITY_POLICY, **(immutability_policy or {})}
        final_constraints = {**self.DEFAULT_OPTIMIZATION_CONSTRAINTS, **(optimization_constraints or {})}

        async with AsyncSessionLocal() as db:
            legacy = LegacyAxis(
                title=title,
                description=description,
                axis_type=axis_type,
                generational_depth=generational_depth,
                survivability_policy=final_survivability,
                immutability_policy=final_immutability,
                optimization_constraints=final_constraints
            )

            db.add(legacy)
            await db.commit()
            await db.refresh(legacy)

            return self._to_dict(legacy)

    async def get(self, legacy_id: str) -> Optional[Dict]:
        """
        Get Legacy Axis by ID

        Args:
            legacy_id: Legacy Axis ID

        Returns:
            Legacy Axis or None
        """
        async with AsyncSessionLocal() as db:
            stmt = select(LegacyAxis).where(LegacyAxis.id == uuid.UUID(legacy_id))
            result = await db.execute(stmt)
            legacy = result.scalar_one_or_none()

            if not legacy:
                return None

            return self._to_dict(legacy)

    async def list_all(self, active_only: bool = True) -> List[Dict]:
        """
        List all Legacy Axis

        Args:
            active_only: Only return active axes

        Returns:
            List of Legacy Axis
        """
        async with AsyncSessionLocal() as db:
            stmt = select(LegacyAxis)

            if active_only:
                stmt = stmt.where(LegacyAxis.is_active == True)

            stmt = stmt.order_by(LegacyAxis.created_at)

            result = await db.execute(stmt)
            axes = result.scalars().all()

            return [self._to_dict(ax) for ax in axes]

    async def update_description(self, legacy_id: str, description: str) -> Optional[Dict]:
        """
        Update Legacy Axis description

        NOTE: Only description can be updated
        All other fields are immutable by design

        Args:
            legacy_id: Legacy Axis ID
            description: New description

        Returns:
            Updated Legacy Axis
        """
        async with AsyncSessionLocal() as db:
            stmt = select(LegacyAxis).where(LegacyAxis.id == uuid.UUID(legacy_id))
            result = await db.execute(stmt)
            legacy = result.scalar_one_or_none()

            if not legacy:
                return None

            # Only description can be updated
            legacy.description = description
            await db.commit()
            await db.refresh(legacy)

            return self._to_dict(legacy)

    async def deactivate(self, legacy_id: str) -> bool:
        """
        Deactivate Legacy Axis

        WARNING: This is NOT deletion
        Deactivated axes remain in history but are not used for new goals

        Args:
            legacy_id: Legacy Axis ID

        Returns:
            True if deactivated
        """
        async with AsyncSessionLocal() as db:
            stmt = select(LegacyAxis).where(LegacyAxis.id == uuid.UUID(legacy_id))
            result = await db.execute(stmt)
            legacy = result.scalar_one_or_none()

            if not legacy:
                return False

            legacy.is_active = False
            await db.commit()

            return True

    async def get_derived_goals(self, legacy_id: str) -> List[Dict]:
        """
        Get all goals derived from this Legacy Axis

        Args:
            legacy_id: Legacy Axis ID

        Returns:
            List of goals
        """
        async with AsyncSessionLocal() as db:
            # Assuming we'll add legacy_axis_id FK to Goal model
            # For now, return empty list
            # TODO: Implement after adding Goal.legacy_axis_id

            return []

    def _to_dict(self, legacy: LegacyAxis) -> Dict:
        """Convert LegacyAxis to dict"""
        return {
            "id": str(legacy.id),
            "title": legacy.title,
            "description": legacy.description,
            "axis_type": legacy.axis_type,
            "generational_depth": legacy.generational_depth,
            "survivability_policy": legacy.survivability_policy,
            "immutability_policy": legacy.immutability_policy,
            "optimization_constraints": legacy.optimization_constraints,
            "created_at": legacy.created_at.isoformat() if legacy.created_at else None,
            "updated_at": legacy.updated_at.isoformat() if legacy.updated_at else None,
            "is_active": legacy.is_active
        }


# Global instance
legacy_axis_service = LegacyAxisService()
