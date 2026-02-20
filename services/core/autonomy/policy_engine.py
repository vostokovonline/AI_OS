"""
POLICY ENGINE - Rule-Based Decision Making

Evaluates system state and generates actions based on rules.
No ML, no LLM - pure deterministic rule evaluation.

Policy Levels:
- Level 1 (Safety): Hard constraints that must never be violated
- Level 2 (Performance): Reactive rules based on metrics
- Level 3 (Strategic): Long-term strategy adjustments

Usage:
    from autonomy import PolicyEngine, PolicyRule
    
    engine = PolicyEngine()
    
    # Add a rule
    engine.add_rule(PolicyRule(
        name="lead_decline_investigate",
        entity_name="monthly_leads",
        condition="delta < 0",
        action_type="create_goal",
        action_payload={"title": "Investigate lead decline", "priority": "high"},
        priority=2
    ))
    
    # Evaluate
    actions = await engine.evaluate(state_entity)
"""
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from uuid import UUID, uuid4
from dataclasses import dataclass, field
import re

from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from database import Base, AsyncSessionLocal
from logging_config import get_logger

from autonomy.system_state import SystemStateEntity, EntityType

logger = get_logger(__name__)


class ActionType(str, Enum):
    """Types of actions the system can take"""
    CREATE_GOAL = "create_goal"
    CANCEL_GOAL = "cancel_goal"
    REPRIORITIZE = "reprioritize"
    DEACTIVATE_STRATEGY = "deactivate_strategy"
    ACTIVATE_STRATEGY = "activate_strategy"
    SPAWN_EXPERIMENT = "spawn_experiment"
    ALERT = "alert"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass
class PolicyRule:
    """
    A rule that evaluates state and triggers actions.
    
    Rules are evaluated in priority order.
    Higher priority = evaluated first.
    """
    id: UUID
    name: str
    entity_name: str  # Which entity this rule applies to
    entity_type: EntityType
    condition_expression: str  # e.g., "delta < 0", "value > 100"
    action_type: ActionType
    action_payload: Dict[str, Any]
    priority: int = 1
    enabled: bool = True
    cooldown_minutes: int = 60  # Don't re-trigger for this long
    last_triggered: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "entity_name": self.entity_name,
            "entity_type": self.entity_type.value,
            "condition_expression": self.condition_expression,
            "action_type": self.action_type.value,
            "action_payload": self.action_payload,
            "priority": self.priority,
            "enabled": self.enabled,
            "cooldown_minutes": self.cooldown_minutes,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "created_at": self.created_at.isoformat()
        }


class PolicyRuleDB(Base):
    """Database model for policy rules"""
    __tablename__ = "policy_rules"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False, unique=True)
    entity_name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    condition_expression = Column(Text, nullable=False)
    action_type = Column(String(50), nullable=False)
    action_payload = Column(JSON, nullable=False)
    priority = Column(Integer, default=1)
    enabled = Column(Boolean, default=True)
    cooldown_minutes = Column(Integer, default=60)
    last_triggered = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


@dataclass
class PolicyEvaluationResult:
    """Result of evaluating a policy rule"""
    rule: PolicyRule
    entity: SystemStateEntity
    triggered: bool
    condition_met: bool
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PolicyEngine:
    """
    Rule-based policy evaluation engine.
    
    Responsibilities:
    - Load and manage policy rules
    - Evaluate conditions against state
    - Track cooldowns
    - Generate action recommendations
    """
    
    def __init__(self):
        self._rule_cache: Dict[str, PolicyRule] = {}
    
    async def load_rules(self) -> List[PolicyRule]:
        """Load all enabled rules from database"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            
            stmt = select(PolicyRuleDB).where(
                PolicyRuleDB.enabled == True
            ).order_by(PolicyRuleDB.priority.desc())
            
            result = await session.execute(stmt)
            db_rules = result.scalars().all()
            
            rules = []
            for db_rule in db_rules:
                rule = PolicyRule(
                    id=db_rule.id,
                    name=db_rule.name,
                    entity_name=db_rule.entity_name,
                    entity_type=EntityType(db_rule.entity_type),
                    condition_expression=db_rule.condition_expression,
                    action_type=ActionType(db_rule.action_type),
                    action_payload=db_rule.action_payload,
                    priority=db_rule.priority,
                    enabled=db_rule.enabled,
                    cooldown_minutes=db_rule.cooldown_minutes,
                    last_triggered=db_rule.last_triggered,
                    created_at=db_rule.created_at
                )
                rules.append(rule)
                self._rule_cache[rule.name] = rule
            
            logger.info("policy_rules_loaded", count=len(rules))
            return rules
    
    async def get_rules_for_entity(self, entity_name: str) -> List[PolicyRule]:
        """Get all rules that apply to a specific entity"""
        rules = await self.load_rules()
        return [r for r in rules if r.entity_name == entity_name]
    
    def evaluate_condition(
        self,
        condition: str,
        entity: SystemStateEntity
    ) -> tuple[bool, str]:
        """
        Evaluate a condition expression against an entity.
        
        Supported expressions:
        - "delta < 0" - negative change
        - "delta > X" - positive change above threshold
        - "value < X" - value below threshold
        - "value > X" - value above threshold
        - "trend == 'down'" - trend direction
        - "confidence < 0.5" - low confidence
        
        Returns:
            (condition_met, reason) tuple
        """
        try:
            # Get delta
            delta = entity.get_delta("value")
            
            # Get current value
            value = entity.current_value.get("value")
            
            # Get trend
            trend = entity.get_trend("value")
            
            # Parse condition
            condition = condition.strip()
            
            # delta comparisons
            if condition.startswith("delta"):
                if delta is None:
                    return False, "No delta available (first measurement)"
                
                match = re.match(r"delta\s*([<>=!]+)\s*(-?\d+\.?\d*)", condition)
                if match:
                    op, threshold = match.groups()
                    threshold = float(threshold)
                    
                    if op == "<" and delta < threshold:
                        return True, f"delta ({delta:.2f}) < {threshold}"
                    elif op == ">" and delta > threshold:
                        return True, f"delta ({delta:.2f}) > {threshold}"
                    elif op == "<=" and delta <= threshold:
                        return True, f"delta ({delta:.2f}) <= {threshold}"
                    elif op == ">=" and delta >= threshold:
                        return True, f"delta ({delta:.2f}) >= {threshold}"
                    elif op == "==" and delta == threshold:
                        return True, f"delta ({delta:.2f}) == {threshold}"
                    else:
                        return False, f"delta ({delta:.2f}) does not meet: {condition}"
            
            # value comparisons
            elif condition.startswith("value"):
                if value is None:
                    return False, "No 'value' key in current_value"
                
                match = re.match(r"value\s*([<>=!]+)\s*(-?\d+\.?\d*)", condition)
                if match:
                    op, threshold = match.groups()
                    threshold = float(threshold)
                    
                    if op == "<" and value < threshold:
                        return True, f"value ({value}) < {threshold}"
                    elif op == ">" and value > threshold:
                        return True, f"value ({value}) > {threshold}"
                    elif op == "<=" and value <= threshold:
                        return True, f"value ({value}) <= {threshold}"
                    elif op == ">=" and value >= threshold:
                        return True, f"value ({value}) >= {threshold}"
                    elif op == "==" and value == threshold:
                        return True, f"value ({value}) == {threshold}"
                    else:
                        return False, f"value ({value}) does not meet: {condition}"
            
            # trend comparisons
            elif condition.startswith("trend"):
                match = re.match(r"trend\s*==\s*['\"](\w+)['\"]", condition)
                if match:
                    expected_trend = match.group(1)
                    if trend == expected_trend:
                        return True, f"trend is '{trend}'"
                    else:
                        return False, f"trend is '{trend}', expected '{expected_trend}'"
            
            # rolling_average comparisons (Phase 2)
            elif condition.startswith("rolling_average"):
                match = re.match(r"rolling_average\s*([<>=!]+)\s*(-?\d+\.?\d*)", condition)
                if match:
                    op, threshold = match.groups()
                    threshold = float(threshold)
                    
                    # Need to fetch from SystemStateDB since rolling_average isn't in SystemStateEntity
                    # For now, check if it exists in extra_data or current_value
                    rolling_avg = entity.current_value.get("rolling_average")
                    
                    if rolling_avg is None:
                        return False, "No rolling_average available"
                    
                    if op == "<" and rolling_avg < threshold:
                        return True, f"rolling_average ({rolling_avg:.2f}) < {threshold}"
                    elif op == ">" and rolling_avg > threshold:
                        return True, f"rolling_average ({rolling_avg:.2f}) > {threshold}"
                    elif op == "<=" and rolling_avg <= threshold:
                        return True, f"rolling_average ({rolling_avg:.2f}) <= {threshold}"
                    elif op == ">=" and rolling_avg >= threshold:
                        return True, f"rolling_average ({rolling_avg:.2f}) >= {threshold}"
                    else:
                        return False, f"rolling_average ({rolling_avg:.2f}) does not meet: {condition}"
            
            # window_days comparisons (Phase 2)
            elif condition.startswith("window_days"):
                match = re.match(r"window_days\s*([<>=!]+)\s*(\d+)", condition)
                if match:
                    op, threshold = match.groups()
                    threshold = int(threshold)
                    
                    window_days = entity.current_value.get("window_days", 7)
                    
                    if op == "<" and window_days < threshold:
                        return True, f"window_days ({window_days}) < {threshold}"
                    elif op == ">" and window_days > threshold:
                        return True, f"window_days ({window_days}) > {threshold}"
                    elif op == "==" and window_days == threshold:
                        return True, f"window_days ({window_days}) == {threshold}"
                    else:
                        return False, f"window_days ({window_days}) does not meet: {condition}"
            
            # confidence comparisons
            elif condition.startswith("confidence"):
                match = re.match(r"confidence\s*([<>=!]+)\s*(\d+\.?\d*)", condition)
                if match:
                    op, threshold = match.groups()
                    threshold = float(threshold)
                    conf = entity.confidence
                    
                    if op == "<" and conf < threshold:
                        return True, f"confidence ({conf}) < {threshold}"
                    elif op == ">" and conf > threshold:
                        return True, f"confidence ({conf}) > {threshold}"
                    else:
                        return False, f"confidence ({conf}) does not meet: {condition}"
            
            return False, f"Unknown condition format: {condition}"
            
        except Exception as e:
            logger.error("condition_evaluation_error", error=str(e), condition=condition)
            return False, f"Evaluation error: {str(e)}"
    
    def is_in_cooldown(self, rule: PolicyRule) -> bool:
        """Check if rule is in cooldown period"""
        if rule.last_triggered is None:
            return False
        
        from datetime import timedelta
        cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
        return datetime.utcnow() < cooldown_end
    
    async def evaluate(
        self,
        entity: SystemStateEntity
    ) -> List[PolicyEvaluationResult]:
        """
        Evaluate all applicable rules for an entity.
        
        Args:
            entity: The state entity to evaluate
            
        Returns:
            List of evaluation results (including non-triggered)
        """
        results = []
        rules = await self.get_rules_for_entity(entity.entity_name)
        
        for rule in rules:
            condition_met, reason = self.evaluate_condition(
                rule.condition_expression, entity
            )
            
            triggered = condition_met and not self.is_in_cooldown(rule)
            
            result = PolicyEvaluationResult(
                rule=rule,
                entity=entity,
                triggered=triggered,
                condition_met=condition_met,
                reason=reason
            )
            results.append(result)
            
            if triggered:
                logger.info(
                    "policy_rule_triggered",
                    rule_name=rule.name,
                    entity_name=entity.entity_name,
                    action=rule.action_type.value,
                    reason=reason
                )
        
        return results
    
    async def add_rule(
        self,
        name: str,
        entity_name: str,
        entity_type: EntityType,
        condition_expression: str,
        action_type: ActionType,
        action_payload: Dict[str, Any],
        priority: int = 1,
        cooldown_minutes: int = 60
    ) -> PolicyRule:
        """Add a new policy rule"""
        async with AsyncSessionLocal() as session:
            db_rule = PolicyRuleDB(
                name=name,
                entity_name=entity_name,
                entity_type=entity_type.value,
                condition_expression=condition_expression,
                action_type=action_type.value,
                action_payload=action_payload,
                priority=priority,
                cooldown_minutes=cooldown_minutes
            )
            session.add(db_rule)
            await session.commit()
            await session.refresh(db_rule)
            
            rule = PolicyRule(
                id=db_rule.id,
                name=db_rule.name,
                entity_name=db_rule.entity_name,
                entity_type=EntityType(db_rule.entity_type),
                condition_expression=db_rule.condition_expression,
                action_type=ActionType(db_rule.action_type),
                action_payload=db_rule.action_payload,
                priority=db_rule.priority,
                enabled=db_rule.enabled,
                cooldown_minutes=db_rule.cooldown_minutes,
                last_triggered=db_rule.last_triggered,
                created_at=db_rule.created_at
            )
            
            logger.info("policy_rule_added", rule_name=name, entity=entity_name)
            return rule
    
    async def mark_triggered(self, rule_id: UUID) -> None:
        """Mark a rule as triggered (for cooldown tracking)"""
        async with AsyncSessionLocal() as session:
            from sqlalchemy import select, update
            
            stmt = update(PolicyRuleDB).where(
                PolicyRuleDB.id == rule_id
            ).values(last_triggered=datetime.utcnow())
            
            await session.execute(stmt)
            await session.commit()


# Default policy rules for autonomous operation
DEFAULT_POLICY_RULES = [
    {
        "name": "lead_decline_investigate",
        "entity_name": "monthly_leads",
        "entity_type": EntityType.METRIC,
        "condition_expression": "delta < 0",
        "action_type": ActionType.CREATE_GOAL,
        "action_payload": {
            "title": "Investigate lead decline",
            "description": "Leads decreased. Investigate root cause.",
            "goal_type": "exploratory",
            "priority": "high"
        },
        "priority": 2,
        "cooldown_minutes": 1440  # 24 hours
    },
    {
        "name": "lead_growth_scale",
        "entity_name": "monthly_leads",
        "entity_type": EntityType.METRIC,
        "condition_expression": "delta > 20",
        "action_type": ActionType.CREATE_GOAL,
        "action_payload": {
            "title": "Scale successful lead channel",
            "description": "Leads increased significantly. Consider scaling.",
            "goal_type": "achievable",
            "priority": "medium"
        },
        "priority": 1,
        "cooldown_minutes": 1440
    },
    {
        "name": "low_confidence_alert",
        "entity_name": "*",  # Any metric
        "entity_type": EntityType.METRIC,
        "condition_expression": "confidence < 0.5",
        "action_type": ActionType.ALERT,
        "action_payload": {
            "message": "Low confidence measurement detected",
            "severity": "warning"
        },
        "priority": 3,
        "cooldown_minutes": 60
    }
]
