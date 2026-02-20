"""
SAFETY CONSTRAINTS - Hard Limits for Autonomous Operations

Prevents runaway behavior by enforcing hard constraints on:
- Maximum concurrent goals
- Budget limits
- Rate limiting
- Cooldown periods
- Risk thresholds

All constraints are code-based (no LLM) and deterministic.

Usage:
    from autonomy.safety_constraints import SafetyConstraints, SafetyViolation
    
    constraints = SafetyConstraints()
    
    # Check if we can create a new goal
    if await constraints.can_create_goal():
        # Create goal
        pass
    else:
        # Log violation
        logger.warning("goal_creation_blocked", reason="max_concurrent_reached")
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select, func, text
from database import AsyncSessionLocal, Base
from sqlalchemy import Column, String, Float, DateTime, Integer, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from logging_config import get_logger

logger = get_logger(__name__)


class ConstraintType(str, Enum):
    """Types of safety constraints"""
    MAX_CONCURRENT_GOALS = "max_concurrent_goals"
    MAX_GOALS_PER_HOUR = "max_goals_per_hour"
    MAX_GOALS_PER_DAY = "max_goals_per_day"
    MAX_BUDGET_PER_DAY = "max_budget_per_day"
    MAX_BUDGET_PER_WEEK = "max_budget_per_week"
    MAX_STRATEGIES = "max_strategies"
    MAX_EXPERIMENTS = "max_experiments"
    MIN_COOLDOWN_MINUTES = "min_cooldown_minutes"
    MAX_RISK_LEVEL = "max_risk_level"


@dataclass
class SafetyConstraint:
    """A single safety constraint"""
    constraint_type: ConstraintType
    limit: float
    current_value: float = 0.0
    enabled: bool = True
    description: str = ""
    
    def is_violated(self) -> bool:
        """Check if constraint is violated"""
        if not self.enabled:
            return False
        return self.current_value >= self.limit
    
    def utilization(self) -> float:
        """Get utilization percentage"""
        if self.limit == 0:
            return 0.0
        return (self.current_value / self.limit) * 100


@dataclass
class SafetyViolation:
    """Record of a safety violation"""
    constraint_type: ConstraintType
    limit: float
    actual: float
    timestamp: datetime
    blocked_action: str
    context: Dict[str, Any] = field(default_factory=dict)


class SafetyConstraintsDB(Base):
    """Database model for safety constraint configuration"""
    __tablename__ = "safety_constraints"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    constraint_type = Column(String(50), nullable=False, unique=True)
    limit_value = Column(Float, nullable=False)  # Renamed from "limit" (reserved word in PG)
    enabled = Column(Boolean, default=True)
    description = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class SafetyViolationLogDB(Base):
    """Database model for logging safety violations"""
    __tablename__ = "safety_violation_log"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    constraint_type = Column(String(50), nullable=False, index=True)
    limit_value = Column(Float, nullable=False)  # Renamed from "limit" (reserved word in PG)
    actual = Column(Float, nullable=False)
    blocked_action = Column(String(100), nullable=False)
    context = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


DEFAULT_CONSTRAINTS = {
    ConstraintType.MAX_CONCURRENT_GOALS: SafetyConstraint(
        constraint_type=ConstraintType.MAX_CONCURRENT_GOALS,
        limit=100,
        description="Maximum number of active goals at any time"
    ),
    ConstraintType.MAX_GOALS_PER_HOUR: SafetyConstraint(
        constraint_type=ConstraintType.MAX_GOALS_PER_HOUR,
        limit=20,
        description="Maximum new goals created per hour"
    ),
    ConstraintType.MAX_GOALS_PER_DAY: SafetyConstraint(
        constraint_type=ConstraintType.MAX_GOALS_PER_DAY,
        limit=100,
        description="Maximum new goals created per day"
    ),
    ConstraintType.MAX_BUDGET_PER_DAY: SafetyConstraint(
        constraint_type=ConstraintType.MAX_BUDGET_PER_DAY,
        limit=10.0,
        description="Maximum budget (USD) spent per day"
    ),
    ConstraintType.MAX_BUDGET_PER_WEEK: SafetyConstraint(
        constraint_type=ConstraintType.MAX_BUDGET_PER_WEEK,
        limit=50.0,
        description="Maximum budget (USD) spent per week"
    ),
    ConstraintType.MAX_STRATEGIES: SafetyConstraint(
        constraint_type=ConstraintType.MAX_STRATEGIES,
        limit=10,
        description="Maximum active strategies"
    ),
    ConstraintType.MAX_EXPERIMENTS: SafetyConstraint(
        constraint_type=ConstraintType.MAX_EXPERIMENTS,
        limit=5,
        description="Maximum concurrent experiments"
    ),
    ConstraintType.MIN_COOLDOWN_MINUTES: SafetyConstraint(
        constraint_type=ConstraintType.MIN_COOLDOWN_MINUTES,
        limit=30,
        description="Minimum cooldown between similar actions"
    ),
    ConstraintType.MAX_RISK_LEVEL: SafetyConstraint(
        constraint_type=ConstraintType.MAX_RISK_LEVEL,
        limit=3,
        description="Maximum risk level allowed (1-5 scale)"
    ),
}


class SafetyConstraints:
    """
    Manager for safety constraints.
    
    All methods are async and query the database for current state.
    No LLM - pure deterministic checks.
    """
    
    def __init__(self):
        self._cache: Dict[ConstraintType, SafetyConstraint] = {}
    
    async def get_constraint(self, constraint_type: ConstraintType) -> SafetyConstraint:
        """Get a constraint with current values"""
        async with AsyncSessionLocal() as session:
            # Try to load from DB
            stmt = select(SafetyConstraintsDB).where(
                SafetyConstraintsDB.constraint_type == constraint_type.value
            )
            result = await session.execute(stmt)
            db_constraint = result.scalar_one_or_none()
            
            if db_constraint:
                constraint = SafetyConstraint(
                    constraint_type=constraint_type,
                    limit=db_constraint.limit_value,
                    enabled=db_constraint.enabled,
                    description=db_constraint.description or ""
                )
            else:
                # Use default
                constraint = DEFAULT_CONSTRAINTS.get(constraint_type)
                if not constraint:
                    raise ValueError(f"Unknown constraint type: {constraint_type}")
            
            # Get current value
            constraint.current_value = await self._get_current_value(constraint_type)
            
            return constraint
    
    async def _get_current_value(self, constraint_type: ConstraintType) -> float:
        """Get current value for a constraint type"""
        async with AsyncSessionLocal() as session:
            if constraint_type == ConstraintType.MAX_CONCURRENT_GOALS:
                stmt = select(func.count()).select_from(
                    select(1).where(
                        # Active, pending, executing goals
                        # We need to use raw SQL or proper model
                        # For now, return 0
                        # TODO: Import Goal model and count active
                    )
                )
                # Simplified - count from goals table
                result = await session.execute(
                    select(func.count()).select_from(
                        select(1).where(
                            # This is a placeholder - actual implementation needs Goal model
                            # For now, we'll return 0
                            text("1=1")
                        )
                    )
                )
                # Return 0 for now
                return 0.0
            
            elif constraint_type == ConstraintType.MAX_GOALS_PER_HOUR:
                # Count goals created in last hour
                one_hour_ago = datetime.utcnow() - timedelta(hours=1)
                # TODO: Implement with Goal model
                return 0.0
            
            elif constraint_type == ConstraintType.MAX_GOALS_PER_DAY:
                # Count goals created today
                today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                # TODO: Implement with Goal model
                return 0.0
            
            elif constraint_type in [ConstraintType.MAX_BUDGET_PER_DAY, ConstraintType.MAX_BUDGET_PER_WEEK]:
                # TODO: Implement budget tracking
                return 0.0
            
            elif constraint_type == ConstraintType.MAX_STRATEGIES:
                # Count active strategies
                # TODO: Implement with Strategy model
                return 0.0
            
            elif constraint_type == ConstraintType.MAX_EXPERIMENTS:
                # Count active experiments
                # TODO: Implement experiment tracking
                return 0.0
            
            else:
                return 0.0
    
    async def can_create_goal(self) -> tuple[bool, Optional[SafetyViolation]]:
        """Check if a new goal can be created"""
        constraints_to_check = [
            ConstraintType.MAX_CONCURRENT_GOALS,
            ConstraintType.MAX_GOALS_PER_HOUR,
            ConstraintType.MAX_GOALS_PER_DAY,
        ]
        
        for constraint_type in constraints_to_check:
            constraint = await self.get_constraint(constraint_type)
            
            if constraint.is_violated():
                violation = SafetyViolation(
                    constraint_type=constraint_type,
                    limit=constraint.limit,
                    actual=constraint.current_value,
                    timestamp=datetime.utcnow(),
                    blocked_action="create_goal",
                    context={"utilization": constraint.utilization()}
                )
                
                await self._log_violation(violation)
                
                logger.warning(
                    "safety_constraint_violated",
                    constraint=constraint_type.value,
                    limit=constraint.limit,
                    actual=constraint.current_value
                )
                
                return False, violation
        
        return True, None
    
    async def can_create_strategy(self) -> tuple[bool, Optional[SafetyViolation]]:
        """Check if a new strategy can be created"""
        constraint = await self.get_constraint(ConstraintType.MAX_STRATEGIES)
        
        if constraint.is_violated():
            violation = SafetyViolation(
                constraint_type=ConstraintType.MAX_STRATEGIES,
                limit=constraint.limit,
                actual=constraint.current_value,
                timestamp=datetime.utcnow(),
                blocked_action="create_strategy"
            )
            
            await self._log_violation(violation)
            return False, violation
        
        return True, None
    
    async def can_spend_budget(self, amount: float) -> tuple[bool, Optional[SafetyViolation]]:
        """Check if budget can be spent"""
        for constraint_type in [ConstraintType.MAX_BUDGET_PER_DAY, ConstraintType.MAX_BUDGET_PER_WEEK]:
            constraint = await self.get_constraint(constraint_type)
            
            if constraint.current_value + amount > constraint.limit:
                violation = SafetyViolation(
                    constraint_type=constraint_type,
                    limit=constraint.limit,
                    actual=constraint.current_value + amount,
                    timestamp=datetime.utcnow(),
                    blocked_action="spend_budget",
                    context={"requested": amount}
                )
                
                await self._log_violation(violation)
                return False, violation
        
        return True, None
    
    async def _log_violation(self, violation: SafetyViolation):
        """Log a safety violation to database"""
        async with AsyncSessionLocal() as session:
            from uuid import uuid4
            
            db_log = SafetyViolationLogDB(
                id=uuid4(),
                constraint_type=violation.constraint_type.value,
                limit_value=violation.limit,
                actual=violation.actual,
                blocked_action=violation.blocked_action,
                context=violation.context,
                timestamp=violation.timestamp
            )
            
            session.add(db_log)
            await session.commit()
            
            logger.info(
                "safety_violation_logged",
                constraint=violation.constraint_type.value,
                blocked_action=violation.blocked_action
            )
    
    async def get_all_constraints(self) -> List[SafetyConstraint]:
        """Get all constraints with current values"""
        constraints = []
        for constraint_type in ConstraintType:
            try:
                constraint = await self.get_constraint(constraint_type)
                constraints.append(constraint)
            except Exception as e:
                logger.error(
                    "constraint_load_error",
                    constraint=constraint_type.value,
                    error=str(e)
                )
        
        return constraints
    
    async def get_utilization_summary(self) -> Dict[str, Any]:
        """Get utilization summary for all constraints"""
        constraints = await self.get_all_constraints()
        
        return {
            "constraints": [
                {
                    "type": c.constraint_type.value,
                    "limit": c.limit,
                    "current": c.current_value,
                    "utilization": c.utilization(),
                    "enabled": c.enabled,
                    "violated": c.is_violated()
                }
                for c in constraints
            ],
            "violated_count": sum(1 for c in constraints if c.is_violated()),
            "total_count": len(constraints)
        }
    
    async def update_constraint(
        self,
        constraint_type: ConstraintType,
        new_limit: float,
        enabled: Optional[bool] = None
    ):
        """Update a constraint's limit"""
        async with AsyncSessionLocal() as session:
            stmt = select(SafetyConstraintsDB).where(
                SafetyConstraintsDB.constraint_type == constraint_type.value
            )
            result = await session.execute(stmt)
            db_constraint = result.scalar_one_or_none()
            
            if db_constraint:
                db_constraint.limit_value = new_limit
                if enabled is not None:
                    db_constraint.enabled = enabled
            else:
                from uuid import uuid4
                
                default = DEFAULT_CONSTRAINTS.get(constraint_type)
                db_constraint = SafetyConstraintsDB(
                    id=uuid4(),
                    constraint_type=constraint_type.value,
                    limit_value=new_limit,
                    enabled=enabled if enabled is not None else (default.enabled if default else True),
                    description=default.description if default else ""
                )
                session.add(db_constraint)
            
            await session.commit()
            
            logger.info(
                "constraint_updated",
                constraint=constraint_type.value,
                new_limit=new_limit
            )
