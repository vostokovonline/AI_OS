"""
SYSTEM STATE - Central State Management

Single source of truth for all system metrics, strategies, resources, and risks.

Entity Types:
- metric: Numerical measurements (leads, revenue, burn_rate)
- strategy: Active strategies with status
- resource: System resources (budget, compute, team capacity)
- risk: Risk indicators and trends
- hypothesis: Active hypotheses being tested
- constraint: Hard limits that must not be violated

Usage:
    from autonomy import SystemStateManager
    
    manager = SystemStateManager()
    
    # Update state from artifact
    await manager.update_entity(
        entity_name="monthly_leads",
        entity_type="metric",
        new_value={"value": 145, "trend": "up"},
        source_artifact_id=artifact.id
    )
    
    # Get current state
    leads = await manager.get_entity("monthly_leads")
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import json

from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Integer
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from database import Base, AsyncSessionLocal
from logging_config import get_logger

logger = get_logger(__name__)


class EntityType(str, Enum):
    """Types of system state entities"""
    METRIC = "metric"
    STRATEGY = "strategy"
    RESOURCE = "resource"
    RISK = "risk"
    HYPOTHESIS = "hypothesis"
    CONSTRAINT = "constraint"


@dataclass
class SystemStateEntity:
    """In-memory representation of system state entity"""
    id: UUID
    entity_name: str
    entity_type: EntityType
    current_value: Dict[str, Any]
    previous_value: Optional[Dict[str, Any]] = None
    confidence: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    source_artifact_id: Optional[UUID] = None
    
    def get_delta(self, key: str = "value") -> Optional[float]:
        """Calculate delta for a specific key"""
        if self.previous_value is None:
            return None
        if key not in self.current_value or key not in self.previous_value:
            return None
        try:
            return self.current_value[key] - self.previous_value[key]
        except (TypeError, KeyError):
            return None
    
    def get_trend(self, key: str = "value") -> str:
        """Get trend direction for a key"""
        delta = self.get_delta(key)
        if delta is None:
            return "unknown"
        if delta > 0:
            return "up"
        elif delta < 0:
            return "down"
        return "stable"


class SystemStateDB(Base):
    """Database model for system state"""
    __tablename__ = "system_state"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    entity_name = Column(String(255), nullable=False, index=True, unique=True)
    entity_type = Column(String(50), nullable=False, index=True)
    current_value = Column(JSON, nullable=False)
    previous_value = Column(JSON, nullable=True)
    confidence = Column(Float, default=1.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source_artifact_id = Column(PG_UUID(as_uuid=True), nullable=True)
    extra_data = Column(JSON, nullable=True)
    
    # Evaluation window support (Phase 2)
    evaluation_window_days = Column(Integer, default=7)  # How many days to consider for trend
    rolling_average = Column(Float, nullable=True)  # Rolling average over window
    trend = Column(String(20), default="stable")  # up, down, stable
    trend_history = Column(JSON, nullable=True)  # [{"date": "...", "value": 100}, ...]


class SystemStateManager:
    """
    Manager for system state entities.
    
    Responsibilities:
    - CRUD operations on state entities
    - State history tracking
    - Delta calculation
    - Confidence management
    """
    
    async def get_entity(self, entity_name: str) -> Optional[SystemStateEntity]:
        """Get current state of an entity"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(SystemStateDB).where(SystemStateDB.entity_name == entity_name)
            result = await session.execute(stmt)
            db_entity = result.scalar_one_or_none()
            
            if not db_entity:
                return None
            
            return SystemStateEntity(
                id=db_entity.id,
                entity_name=db_entity.entity_name,
                entity_type=EntityType(db_entity.entity_type),
                current_value=db_entity.current_value,
                previous_value=db_entity.previous_value,
                confidence=db_entity.confidence,
                last_updated=db_entity.last_updated,
                source_artifact_id=db_entity.source_artifact_id
            )
    
    async def update_entity(
        self,
        entity_name: str,
        entity_type: EntityType,
        new_value: Dict[str, Any],
        source_artifact_id: Optional[UUID] = None,
        confidence: float = 1.0
    ) -> SystemStateEntity:
        """
        Update or create a state entity.
        
        Args:
            entity_name: Unique name for the entity
            entity_type: Type of entity
            new_value: New value to set
            source_artifact_id: ID of artifact that triggered this update
            confidence: Confidence level of this update (0.0-1.0)
            
        Returns:
            Updated SystemStateEntity
        """
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(SystemStateDB).where(SystemStateDB.entity_name == entity_name)
            result = await session.execute(stmt)
            db_entity = result.scalar_one_or_none()
            
            if db_entity:
                # Update existing - move current to previous
                db_entity.previous_value = db_entity.current_value
                db_entity.current_value = new_value
                db_entity.confidence = confidence
                db_entity.last_updated = datetime.utcnow()
                db_entity.source_artifact_id = source_artifact_id
                
                logger.info(
                    "state_entity_updated",
                    entity_name=entity_name,
                    entity_type=entity_type.value,
                    confidence=confidence
                )
            else:
                # Create new
                db_entity = SystemStateDB(
                    entity_name=entity_name,
                    entity_type=entity_type.value,
                    current_value=new_value,
                    previous_value=None,
                    confidence=confidence,
                    source_artifact_id=source_artifact_id
                )
                session.add(db_entity)
                
                logger.info(
                    "state_entity_created",
                    entity_name=entity_name,
                    entity_type=entity_type.value
                )
            
            await session.commit()
            await session.refresh(db_entity)
            
            return SystemStateEntity(
                id=db_entity.id,
                entity_name=db_entity.entity_name,
                entity_type=EntityType(db_entity.entity_type),
                current_value=db_entity.current_value,
                previous_value=db_entity.previous_value,
                confidence=db_entity.confidence,
                last_updated=db_entity.last_updated,
                source_artifact_id=db_entity.source_artifact_id
            )
    
    async def get_all_metrics(self) -> List[SystemStateEntity]:
        """Get all metric-type entities"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(SystemStateDB).where(
                SystemStateDB.entity_type == EntityType.METRIC.value
            )
            result = await session.execute(stmt)
            db_entities = result.scalars().all()
            
            return [
                SystemStateEntity(
                    id=e.id,
                    entity_name=e.entity_name,
                    entity_type=EntityType(e.entity_type),
                    current_value=e.current_value,
                    previous_value=e.previous_value,
                    confidence=e.confidence,
                    last_updated=e.last_updated,
                    source_artifact_id=e.source_artifact_id
                )
                for e in db_entities
            ]
    
    async def get_entities_by_type(self, entity_type: EntityType) -> List[SystemStateEntity]:
        """Get all entities of a specific type"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(SystemStateDB).where(
                SystemStateDB.entity_type == entity_type.value
            )
            result = await session.execute(stmt)
            db_entities = result.scalars().all()
            
            return [
                SystemStateEntity(
                    id=e.id,
                    entity_name=e.entity_name,
                    entity_type=EntityType(e.entity_type),
                    current_value=e.current_value,
                    previous_value=e.previous_value,
                    confidence=e.confidence,
                    last_updated=e.last_updated,
                    source_artifact_id=e.source_artifact_id
                )
                for e in db_entities
            ]
    
    async def update_with_trend(
        self,
        entity_name: str,
        entity_type: EntityType,
        new_value: Dict[str, Any],
        value_key: str = "value",
        source_artifact_id: Optional[UUID] = None,
        confidence: float = 1.0,
        evaluation_window_days: int = 7
    ) -> SystemStateEntity:
        """
        Update entity with automatic trend calculation.
        
        Args:
            entity_name: Unique name for the entity
            entity_type: Type of entity
            new_value: New value dict (must contain value_key)
            value_key: Key to use for trend calculation (default: "value")
            source_artifact_id: ID of artifact that triggered this update
            confidence: Confidence level of this update (0.0-1.0)
            evaluation_window_days: Days to consider for rolling average
            
        Returns:
            Updated SystemStateEntity
        """
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(SystemStateDB).where(SystemStateDB.entity_name == entity_name)
            result = await session.execute(stmt)
            db_entity = result.scalar_one_or_none()
            
            now = datetime.utcnow()
            new_numeric_value = new_value.get(value_key, 0)
            
            if db_entity:
                # Get or init trend history
                history = db_entity.trend_history or []
                
                # Add new entry
                history.append({
                    "date": now.isoformat(),
                    "value": new_numeric_value
                })
                
                # Filter to window
                cutoff = (now - timedelta(days=evaluation_window_days)).isoformat()
                history = [h for h in history if h["date"] >= cutoff]
                
                # Calculate rolling average
                if history:
                    rolling_avg = sum(h["value"] for h in history) / len(history)
                else:
                    rolling_avg = new_numeric_value
                
                # Calculate trend
                if len(history) >= 2:
                    first_half = history[:len(history)//2]
                    second_half = history[len(history)//2:]
                    first_avg = sum(h["value"] for h in first_half) / len(first_half)
                    second_avg = sum(h["value"] for h in second_half) / len(second_half)
                    
                    delta = second_avg - first_avg
                    threshold = first_avg * 0.05  # 5% threshold for "stable"
                    
                    if delta > threshold:
                        trend = "up"
                    elif delta < -threshold:
                        trend = "down"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"
                
                # Update entity
                db_entity.previous_value = db_entity.current_value
                db_entity.current_value = new_value
                db_entity.confidence = confidence
                db_entity.last_updated = now
                db_entity.source_artifact_id = source_artifact_id
                db_entity.evaluation_window_days = evaluation_window_days
                db_entity.rolling_average = rolling_avg
                db_entity.trend = trend
                db_entity.trend_history = history
                
                logger.info(
                    "state_entity_updated_with_trend",
                    entity_name=entity_name,
                    trend=trend,
                    rolling_average=rolling_avg,
                    window_days=evaluation_window_days
                )
            else:
                # Create new entity
                db_entity = SystemStateDB(
                    entity_name=entity_name,
                    entity_type=entity_type.value,
                    current_value=new_value,
                    previous_value=None,
                    confidence=confidence,
                    source_artifact_id=source_artifact_id,
                    evaluation_window_days=evaluation_window_days,
                    rolling_average=new_numeric_value,
                    trend="stable",
                    trend_history=[{"date": now.isoformat(), "value": new_numeric_value}]
                )
                session.add(db_entity)
                
                logger.info(
                    "state_entity_created_with_trend",
                    entity_name=entity_name,
                    entity_type=entity_type.value
                )
            
            await session.commit()
            await session.refresh(db_entity)
            
            return SystemStateEntity(
                id=db_entity.id,
                entity_name=db_entity.entity_name,
                entity_type=EntityType(db_entity.entity_type),
                current_value=db_entity.current_value,
                previous_value=db_entity.previous_value,
                confidence=db_entity.confidence,
                last_updated=db_entity.last_updated,
                source_artifact_id=db_entity.source_artifact_id
            )
    
    async def get_trend_summary(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """
        Get trend summary for an entity.
        
        Returns:
            {
                "entity_name": str,
                "trend": str,  # up, down, stable
                "rolling_average": float,
                "window_days": int,
                "history_count": int
            }
        """
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(SystemStateDB).where(SystemStateDB.entity_name == entity_name)
            result = await session.execute(stmt)
            db_entity = result.scalar_one_or_none()
            
            if not db_entity:
                return None
            
            return {
                "entity_name": db_entity.entity_name,
                "trend": db_entity.trend,
                "rolling_average": db_entity.rolling_average,
                "window_days": db_entity.evaluation_window_days,
                "history_count": len(db_entity.trend_history or [])
            }
