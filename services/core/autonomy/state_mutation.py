"""
STATE MUTATION - State Change Proposals from Artifacts

Artifacts propose state mutations. The policy engine approves them.
This ensures controlled state updates with audit trail.

Mutation Types:
- UPDATE: Change current value
- INCREMENT: Add to current value
- DECREMENT: Subtract from current value
- SET_IF_HIGHER: Only update if new value is higher
- SET_IF_LOWER: Only update if new value is lower

Usage:
    from autonomy import StateMutation, MutationType
    
    mutation = StateMutation(
        entity_name="monthly_leads",
        mutation_type=MutationType.UPDATE,
        new_value={"value": 145},
        confidence=0.92,
        source_artifact_id=artifact.id
    )
"""
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from uuid import UUID
from dataclasses import dataclass, field


class MutationType(str, Enum):
    """Types of state mutations"""
    UPDATE = "update"           # Direct update
    INCREMENT = "increment"     # Add to current value
    DECREMENT = "decrement"     # Subtract from current value
    SET_IF_HIGHER = "set_if_higher"  # Only if new > current
    SET_IF_LOWER = "set_if_lower"    # Only if new < current
    MERGE = "merge"             # Merge dicts


@dataclass
class StateMutation:
    """
    Proposal to mutate system state.
    
    Artifacts create these proposals.
    PolicyEngine validates and applies them.
    """
    entity_name: str
    mutation_type: MutationType
    new_value: Dict[str, Any]
    confidence: float = 1.0
    source_artifact_id: Optional[UUID] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    reason: Optional[str] = None
    approved: bool = False
    applied: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage in artifact"""
        return {
            "entity_name": self.entity_name,
            "mutation_type": self.mutation_type.value,
            "new_value": self.new_value,
            "confidence": self.confidence,
            "source_artifact_id": str(self.source_artifact_id) if self.source_artifact_id else None,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "approved": self.approved,
            "applied": self.applied
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateMutation":
        """Deserialize from dictionary"""
        return cls(
            entity_name=data["entity_name"],
            mutation_type=MutationType(data["mutation_type"]),
            new_value=data["new_value"],
            confidence=data.get("confidence", 1.0),
            source_artifact_id=UUID(data["source_artifact_id"]) if data.get("source_artifact_id") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            reason=data.get("reason"),
            approved=data.get("approved", False),
            applied=data.get("applied", False)
        )


async def apply_state_mutation(
    mutation: StateMutation,
    current_value: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply a state mutation to calculate new value.
    
    This does NOT update the database - only calculates the new value.
    The DecisionEngine is responsible for actual state updates.
    
    Args:
        mutation: The mutation to apply
        current_value: Current state value (if exists)
        
    Returns:
        New state value after mutation
    """
    if mutation.mutation_type == MutationType.UPDATE:
        return mutation.new_value
    
    if current_value is None:
        # No current value - just set
        return mutation.new_value
    
    if mutation.mutation_type == MutationType.INCREMENT:
        result = dict(current_value)
        for key, delta in mutation.new_value.items():
            if key in result and isinstance(result[key], (int, float)):
                result[key] += delta
            else:
                result[key] = delta
        return result
    
    if mutation.mutation_type == MutationType.DECREMENT:
        result = dict(current_value)
        for key, delta in mutation.new_value.items():
            if key in result and isinstance(result[key], (int, float)):
                result[key] -= delta
            else:
                result[key] = -delta
        return result
    
    if mutation.mutation_type == MutationType.SET_IF_HIGHER:
        result = dict(current_value)
        for key, new_val in mutation.new_value.items():
            if key not in result:
                result[key] = new_val
            elif isinstance(new_val, (int, float)) and isinstance(result[key], (int, float)):
                if new_val > result[key]:
                    result[key] = new_val
            else:
                result[key] = new_val
        return result
    
    if mutation.mutation_type == MutationType.SET_IF_LOWER:
        result = dict(current_value)
        for key, new_val in mutation.new_value.items():
            if key not in result:
                result[key] = new_val
            elif isinstance(new_val, (int, float)) and isinstance(result[key], (int, float)):
                if new_val < result[key]:
                    result[key] = new_val
            else:
                result[key] = new_val
        return result
    
    if mutation.mutation_type == MutationType.MERGE:
        result = dict(current_value)
        result.update(mutation.new_value)
        return result
    
    return mutation.new_value
