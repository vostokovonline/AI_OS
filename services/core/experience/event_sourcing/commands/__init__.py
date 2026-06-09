"""
Command Definitions - Intent-oriented operations.

Commands represent intent, not facts.
They are processed by CommandHandler → produce Events.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import hashlib
import json


class CommandType(Enum):
    """Command types"""
    ADD_BELIEF = "add_belief"
    UPDATE_BELIEF = "update_belief"
    REMOVE_BELIEF = "remove_belief"
    ADD_CAUSAL_EDGE = "add_causal_edge"
    REGISTER_CONTRADICTION = "register_contradiction"
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    COMMIT_TRANSACTION = "commit_transaction"
    COMPENSATE_TRANSACTION = "compensate_transaction"
    MUTATE_IDENTITY = "mutate_identity"
    EVOLVE_GENOME = "evolve_genome"
    RECORD_LINEAGE = "record_lineage"


@dataclass
class Command:
    """Base command structure"""
    command_type: str
    stream_id: str
    correlation_id: str = ""
    causation_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def command_id(self) -> str:
        content = {
            "type": self.command_type,
            "stream": self.stream_id,
            "payload": self.payload,
            "ts": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:32]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "stream_id": self.stream_id,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload": self.payload,
            "metadata": self.metadata
        }


@dataclass
class AddBelief(Command):
    """Command to add new belief"""
    command_type: str = "add_belief"
    stream_id: str = "belief-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "proposition": "",
        "confidence": 0.5,
        "entropy": 0.0,
        "source": "unknown",
        "incoming_causes": [],
        "outgoing_effects": []
    })


@dataclass
class UpdateBelief(Command):
    """Command to update belief"""
    command_type: str = "update_belief"
    stream_id: str = "belief-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "belief_id": "",
        "confidence": 0.5,
        "entropy": 0.0,
        "attractor_state": None
    })


@dataclass
class RemoveBelief(Command):
    """Command to remove belief"""
    command_type: str = "remove_belief"
    stream_id: str = "belief-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "belief_id": ""
    })


@dataclass
class RegisterContradiction(Command):
    """Command to register contradiction"""
    command_type: str = "register_contradiction"
    stream_id: str = "contradiction-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "belief_ids": [],
        "contradiction_type": "unknown",
        "severity": "medium"
    })


@dataclass
class ResolveContradiction(Command):
    """Command to resolve contradiction"""
    command_type: str = "resolve_contradiction"
    stream_id: str = "contradiction-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "episode_id": "",
        "resolution_type": "manual"
    })


@dataclass
class CompensateTransaction(Command):
    """Command to compensate transaction"""
    command_type: str = "compensate_transaction"
    stream_id: str = "transaction-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "original_transaction_id": "",
        "reason": "manual_compensation"
    })


@dataclass
class MutateIdentity(Command):
    """Command to mutate identity"""
    command_type: str = "mutate_identity"
    stream_id: str = "identity-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "mutation_type": "strengthen",
        "target_id": "",
        "parameters": {}
    })


@dataclass
class EvolveGenome(Command):
    """Command to evolve genome"""
    command_type: str = "evolve_genome"
    stream_id: str = "genome-stream"
    payload: Dict[str, Any] = field(default_factory=lambda: {
        "selection_pressure": 0.5,
        "mutation_rate": 0.1,
        "fitness_data": {}
    })