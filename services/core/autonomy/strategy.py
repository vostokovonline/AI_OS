"""
STRATEGY ENTITY - Strategic Layer for Autonomous Operation

Strategy is an entity ABOVE goals.
Goals execute strategies. Strategies define outcomes.

Strategy Lifecycle:
1. HYPOTHESIS - Proposed strategy, untested
2. ACTIVE - Being executed, under evaluation
3. PAUSED - Temporarily stopped
4. KILLED - Permanently stopped (underperforming)
5. COMPLETED - Successfully achieved expected outcome

Strategy-Goal Relationship:
- Strategy has multiple goals
- Goals contribute to strategy confidence
- Strategy confidence affects goal priorities

Usage:
    from autonomy.strategy import StrategyManager, Strategy, StrategyStatus
    
    manager = StrategyManager()
    
    # Create strategy
    strategy = await manager.create_strategy(
        name="increase_monthly_leads",
        hypothesis="Running ads on channel X will increase leads by ≥15%",
        expected_outcome={
            "entity": "monthly_leads",
            "direction": "increase",
            "min_delta": 15,
            "evaluation_period_days": 30
        }
    )
    
    # Evaluate strategy
    result = await manager.evaluate_strategy(strategy.id)
    # result.confidence = 0.72
    # result.status = ACTIVE | PAUSED | KILLED
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from uuid import UUID, uuid4
from dataclasses import dataclass, field

from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from database import Base, AsyncSessionLocal
from logging_config import get_logger

logger = get_logger(__name__)


class StrategyStatus(str, Enum):
    """Lifecycle states of a strategy"""
    HYPOTHESIS = "hypothesis"    # Proposed, not yet tested
    ACTIVE = "active"            # Being executed
    PAUSED = "paused"            # Temporarily stopped
    KILLED = "killed"            # Permanently stopped (underperforming)
    COMPLETED = "completed"      # Successfully achieved outcome


@dataclass
class ExpectedOutcome:
    """What the strategy is expected to achieve"""
    entity_name: str            # Which state entity to measure
    direction: str              # "increase" | "decrease" | "stabilize"
    min_delta: float            # Minimum acceptable change
    evaluation_period_days: int # How long to evaluate
    baseline_value: Optional[float] = None  # Starting value
    current_value: Optional[float] = None   # Current value
    confidence_threshold: float = 0.7       # Kill if below this
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "direction": self.direction,
            "min_delta": self.min_delta,
            "evaluation_period_days": self.evaluation_period_days,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "confidence_threshold": self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExpectedOutcome":
        return cls(
            entity_name=data["entity_name"],
            direction=data["direction"],
            min_delta=data["min_delta"],
            evaluation_period_days=data["evaluation_period_days"],
            baseline_value=data.get("baseline_value"),
            current_value=data.get("current_value"),
            confidence_threshold=data.get("confidence_threshold", 0.7)
        )


@dataclass
class Strategy:
    """
    In-memory representation of a strategy.
    
    A strategy is a higher-level entity than goals.
    Multiple goals can contribute to one strategy.
    """
    id: UUID
    name: str
    hypothesis: str                        # What we're testing
    expected_outcome: ExpectedOutcome      # Success criteria
    status: StrategyStatus = StrategyStatus.HYPOTHESIS
    confidence: float = 0.5                # 0.0-1.0, how confident we are
    linked_goals: List[UUID] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    evaluation_count: int = 0
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "hypothesis": self.hypothesis,
            "expected_outcome": self.expected_outcome.to_dict(),
            "status": self.status.value,
            "confidence": self.confidence,
            "linked_goals": [str(g) for g in self.linked_goals],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "evaluation_count": self.evaluation_count,
            "extra_data": self.extra_data
        }


class StrategyDB(Base):
    """Database model for strategies"""
    __tablename__ = "strategies"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    hypothesis = Column(Text, nullable=False)
    expected_outcome = Column(JSON, nullable=False)
    status = Column(String(50), default=StrategyStatus.HYPOTHESIS.value, index=True)
    confidence = Column(Float, default=0.5)
    linked_goals = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    evaluated_at = Column(DateTime, nullable=True)
    evaluation_count = Column(Integer, default=0)
    extra_data = Column(JSON, nullable=True)


@dataclass
class StrategyEvaluationResult:
    """Result of evaluating a strategy"""
    strategy_id: UUID
    strategy_name: str
    status: StrategyStatus
    confidence: float
    delta_achieved: Optional[float]
    min_delta_required: float
    evaluation_period_days: int
    days_elapsed: int
    recommendation: str  # "continue" | "pause" | "kill" | "complete"
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class StrategyManager:
    """
    Manager for strategy entities.
    
    Responsibilities:
    - CRUD operations
    - Strategy evaluation
    - Confidence updates
    - Goal linking
    """
    
    async def create_strategy(
        self,
        name: str,
        hypothesis: str,
        expected_outcome: ExpectedOutcome
    ) -> Strategy:
        """Create a new strategy"""
        async with AsyncSessionLocal() as session:
            # Check if exists
            from sqlalchemy import select
            stmt = select(StrategyDB).where(StrategyDB.name == name)
            result = await session.execute(stmt)
            if result.scalar_one_or_none():
                raise ValueError(f"Strategy already exists: {name}")
            
            db_strategy = StrategyDB(
                name=name,
                hypothesis=hypothesis,
                expected_outcome=expected_outcome.to_dict(),
                status=StrategyStatus.HYPOTHESIS.value,
                confidence=0.5
            )
            session.add(db_strategy)
            await session.commit()
            await session.refresh(db_strategy)
            
            logger.info(
                "strategy_created",
                strategy_id=str(db_strategy.id),
                name=name,
                hypothesis=hypothesis[:50]
            )
            
            return Strategy(
                id=db_strategy.id,
                name=db_strategy.name,
                hypothesis=db_strategy.hypothesis,
                expected_outcome=ExpectedOutcome.from_dict(db_strategy.expected_outcome),
                status=StrategyStatus(db_strategy.status),
                confidence=db_strategy.confidence,
                linked_goals=[UUID(g) for g in (db_strategy.linked_goals or [])],
                created_at=db_strategy.created_at,
                started_at=db_strategy.started_at,
                evaluated_at=db_strategy.evaluated_at,
                evaluation_count=db_strategy.evaluation_count,
                extra_data=db_strategy.extra_data or {}
            )
    
    async def get_strategy(self, strategy_id: UUID) -> Optional[Strategy]:
        """Get strategy by ID"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(StrategyDB).where(StrategyDB.id == strategy_id)
            result = await session.execute(stmt)
            db_strategy = result.scalar_one_or_none()
            
            if not db_strategy:
                return None
            
            return Strategy(
                id=db_strategy.id,
                name=db_strategy.name,
                hypothesis=db_strategy.hypothesis,
                expected_outcome=ExpectedOutcome.from_dict(db_strategy.expected_outcome),
                status=StrategyStatus(db_strategy.status),
                confidence=db_strategy.confidence,
                linked_goals=[UUID(g) for g in (db_strategy.linked_goals or [])],
                created_at=db_strategy.created_at,
                started_at=db_strategy.started_at,
                evaluated_at=db_strategy.evaluated_at,
                evaluation_count=db_strategy.evaluation_count,
                extra_data=db_strategy.extra_data or {}
            )
    
    async def get_active_strategies(self) -> List[Strategy]:
        """Get all active strategies"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(StrategyDB).where(
                StrategyDB.status == StrategyStatus.ACTIVE.value
            )
            result = await session.execute(stmt)
            db_strategies = result.scalars().all()
            
            return [
                Strategy(
                    id=s.id,
                    name=s.name,
                    hypothesis=s.hypothesis,
                    expected_outcome=ExpectedOutcome.from_dict(s.expected_outcome),
                    status=StrategyStatus(s.status),
                    confidence=s.confidence,
                    linked_goals=[UUID(g) for g in (s.linked_goals or [])],
                    created_at=s.created_at,
                    started_at=s.started_at,
                    evaluated_at=s.evaluated_at,
                    evaluation_count=s.evaluation_count,
                    extra_data=s.extra_data or {}
                )
                for s in db_strategies
            ]
    
    async def activate_strategy(self, strategy_id: UUID) -> Strategy:
        """Activate a hypothesis strategy"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, update
            
            stmt = select(StrategyDB).where(StrategyDB.id == strategy_id)
            result = await session.execute(stmt)
            db_strategy = result.scalar_one_or_none()
            
            if not db_strategy:
                raise ValueError(f"Strategy not found: {strategy_id}")
            
            db_strategy.status = StrategyStatus.ACTIVE.value
            db_strategy.started_at = datetime.utcnow()
            
            await session.commit()
            await session.refresh(db_strategy)
            
            logger.info(
                "strategy_activated",
                strategy_id=str(strategy_id),
                name=db_strategy.name
            )
            
            return await self.get_strategy(strategy_id)
    
    async def link_goal(self, strategy_id: UUID, goal_id: UUID) -> None:
        """Link a goal to a strategy"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(StrategyDB).where(StrategyDB.id == strategy_id)
            result = await session.execute(stmt)
            db_strategy = result.scalar_one_or_none()
            
            if not db_strategy:
                raise ValueError(f"Strategy not found: {strategy_id}")
            
            linked = set(db_strategy.linked_goals or [])
            linked.add(str(goal_id))
            db_strategy.linked_goals = list(linked)
            
            await session.commit()
            
            logger.info(
                "goal_linked_to_strategy",
                strategy_id=str(strategy_id),
                goal_id=str(goal_id)
            )
    
    async def evaluate_strategy(
        self,
        strategy_id: UUID,
        current_entity_value: Optional[float] = None
    ) -> StrategyEvaluationResult:
        """
        Evaluate a strategy based on current state.
        
        This is the core of strategic autonomy.
        
        Args:
            strategy_id: Strategy to evaluate
            current_entity_value: Current value of the target entity
            
        Returns:
            EvaluationResult with recommendation
        """
        strategy = await self.get_strategy(strategy_id)
        if not strategy:
            raise ValueError(f"Strategy not found: {strategy_id}")
        
        outcome = strategy.expected_outcome
        recommendation = "continue"
        reason = ""
        new_confidence = strategy.confidence
        new_status = strategy.status
        delta_achieved = None
        
        # Calculate delta
        if outcome.baseline_value is not None and current_entity_value is not None:
            delta_achieved = current_entity_value - outcome.baseline_value
            outcome.current_value = current_entity_value
            
            # Calculate progress percentage
            if outcome.min_delta != 0:
                progress = delta_achieved / outcome.min_delta
            else:
                progress = 1.0 if delta_achieved >= 0 else 0.0
            
            # Update confidence based on progress
            if outcome.direction == "increase":
                if progress >= 1.0:
                    new_confidence = min(1.0, strategy.confidence + 0.1)
                    recommendation = "complete"
                    reason = f"Target achieved: {delta_achieved:.1f} >= {outcome.min_delta}"
                    new_status = StrategyStatus.COMPLETED
                elif progress >= 0.5:
                    new_confidence = min(1.0, strategy.confidence + 0.05)
                    reason = f"On track: {progress*100:.0f}% of target"
                elif progress > 0:
                    new_confidence = max(0.1, strategy.confidence - 0.05)
                    reason = f"Below target: {progress*100:.0f}% achieved"
                else:
                    new_confidence = max(0.1, strategy.confidence - 0.15)
                    recommendation = "pause"
                    reason = f"Negative progress: {delta_achieved:.1f}"
            
            # Kill if below threshold
            if new_confidence < outcome.confidence_threshold:
                recommendation = "kill"
                new_status = StrategyStatus.KILLED
                reason = f"Confidence {new_confidence:.2f} below threshold {outcome.confidence_threshold}"
        
        # Calculate days elapsed
        days_elapsed = 0
        if strategy.started_at:
            days_elapsed = (datetime.utcnow() - strategy.started_at).days
        
        # Check if evaluation period exceeded
        if days_elapsed >= outcome.evaluation_period_days:
            if recommendation == "continue":
                recommendation = "kill"
                new_status = StrategyStatus.KILLED
                reason = f"Evaluation period ({outcome.evaluation_period_days}d) exceeded without success"
        
        # Update database
        await self._update_strategy_evaluation(
            strategy_id=strategy_id,
            new_status=new_status,
            new_confidence=new_confidence,
            outcome=outcome
        )
        
        logger.info(
            "strategy_evaluated",
            strategy_id=str(strategy_id),
            confidence=new_confidence,
            recommendation=recommendation,
            reason=reason
        )
        
        return StrategyEvaluationResult(
            strategy_id=strategy_id,
            strategy_name=strategy.name,
            status=new_status,
            confidence=new_confidence,
            delta_achieved=delta_achieved,
            min_delta_required=outcome.min_delta,
            evaluation_period_days=outcome.evaluation_period_days,
            days_elapsed=days_elapsed,
            recommendation=recommendation,
            reason=reason
        )
    
    async def _update_strategy_evaluation(
        self,
        strategy_id: UUID,
        new_status: StrategyStatus,
        new_confidence: float,
        outcome: ExpectedOutcome
    ) -> None:
        """Update strategy after evaluation"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(StrategyDB).where(StrategyDB.id == strategy_id)
            result = await session.execute(stmt)
            db_strategy = result.scalar_one_or_none()
            
            if db_strategy:
                db_strategy.status = new_status.value
                db_strategy.confidence = new_confidence
                db_strategy.evaluated_at = datetime.utcnow()
                db_strategy.evaluation_count += 1
                db_strategy.expected_outcome = outcome.to_dict()
                await session.commit()
    
    async def kill_strategy(self, strategy_id: UUID, reason: str) -> Strategy:
        """Kill an underperforming strategy"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(StrategyDB).where(StrategyDB.id == strategy_id)
            result = await session.execute(stmt)
            db_strategy = result.scalar_one_or_none()
            
            if db_strategy:
                db_strategy.status = StrategyStatus.KILLED.value
                db_strategy.extra_data = db_strategy.extra_data or {}
                db_strategy.extra_data["kill_reason"] = reason
                db_strategy.extra_data["killed_at"] = datetime.utcnow().isoformat()
                await session.commit()
                
                logger.warning(
                    "strategy_killed",
                    strategy_id=str(strategy_id),
                    name=db_strategy.name,
                    reason=reason
                )
            
            return await self.get_strategy(strategy_id)
    
    async def pause_strategy(self, strategy_id: UUID, reason: str) -> Strategy:
        """Pause a strategy"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(StrategyDB).where(StrategyDB.id == strategy_id)
            result = await session.execute(stmt)
            db_strategy = result.scalar_one_or_none()
            
            if db_strategy:
                db_strategy.status = StrategyStatus.PAUSED.value
                db_strategy.extra_data = db_strategy.extra_data or {}
                db_strategy.extra_data["pause_reason"] = reason
                await session.commit()
                
                logger.info(
                    "strategy_paused",
                    strategy_id=str(strategy_id),
                    name=db_strategy.name,
                    reason=reason
                )
            
            return await self.get_strategy(strategy_id)
