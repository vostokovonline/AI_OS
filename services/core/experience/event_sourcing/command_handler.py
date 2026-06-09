"""
Command Handler - Thin layer that routes commands through policies to events.

Command Handler responsibilities:
1. Validate command structure
2. Evaluate policies
3. Convert command to event
4. Return event for storage (does NOT store directly)

This keeps the handler thin and testable.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from commands import Command
from events import CognitiveEvent, EventTypes, SchemaVersion
from policies import PolicyEngine, PolicyResult, PolicyDecision


@dataclass
class CommandResult:
    """Result of command processing"""
    success: bool
    event: Optional[CognitiveEvent] = None
    error: str = ""
    policy_result: Optional[PolicyResult] = None


class CommandHandler:
    """
    Thin command handler.
    
    Flow: Command → Policy Check → Event Creation
    Does NOT store events - returns them for storage.
    """
    
    def __init__(self):
        self._event_counter = 0
    
    def _generate_event_id(self) -> str:
        """Generate deterministic event ID"""
        self._event_counter += 1
        content = {
            "counter": self._event_counter,
            "time": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()[:32]
    
    def handle(self, command: Command, current_state: Dict[str, Any]) -> CommandResult:
        """
        Handle command.
        
        Returns CommandResult with event if successful.
        """
        command_type = command.command_type
        
        policy_result = PolicyEngine.evaluate(command.to_dict(), current_state)
        
        if policy_result.decision == PolicyDecision.DENY:
            return CommandResult(
                success=False,
                error=policy_result.reason,
                policy_result=policy_result
            )
        
        event = self._command_to_event(command, policy_result)
        
        return CommandResult(
            success=True,
            event=event,
            policy_result=policy_result
        )
    
    def _command_to_event(
        self, 
        command: Command, 
        policy_result: PolicyResult
    ) -> CognitiveEvent:
        """Convert command to event"""
        
        event_type_map = {
            "add_belief": EventTypes.BELIEF_ADDED,
            "update_belief": EventTypes.BELIEF_UPDATED,
            "remove_belief": EventTypes.BELIEF_REMOVED,
            "register_contradiction": EventTypes.CONTRADICTION_REGISTERED,
            "resolve_contradiction": EventTypes.CONTRADICTION_RESOLVED,
            "compensate_transaction": EventTypes.TRANSACTION_COMPENSATED,
            "mutate_identity": EventTypes.IDENTITY_MUTATED,
            "evolve_genome": EventTypes.GENOME_EVOLVED,
        }
        
        event_type = event_type_map.get(command.command_type, command.command_type)
        
        payload = dict(command.payload)
        if policy_result.conditions:
            payload["_policy_conditions"] = policy_result.conditions
        
        return CognitiveEvent(
            event_type=event_type,
            stream_id=command.stream_id,
            position=0,
            schema_version=SchemaVersion.V1.value,
            event_id=self._generate_event_id(),
            timestamp=datetime.utcnow().isoformat(),
            causation_id=command.causation_id,
            correlation_id=command.correlation_id,
            payload=payload
        )
    
    def handle_batch(
        self, 
        commands: List[Command], 
        current_state: Dict[str, Any]
    ) -> List[CommandResult]:
        """Handle batch of commands"""
        results = []
        state = dict(current_state)
        
        for command in commands:
            result = self.handle(command, state)
            results.append(result)
            
            if result.success and result.event:
                state = self._apply_event_preview(state, result.event)
        
        return results
    
    def _apply_event_preview(self, state: Dict[str, Any], event: CognitiveEvent) -> Dict[str, Any]:
        """
        Preview state change for policy evaluation in batch.
        
        This is a lightweight preview, not the full reducer.
        """
        event_type = event.event_type
        
        new_state = dict(state)
        
        if event_type == EventTypes.BELIEF_ADDED:
            belief_id = event.payload.get("belief_id", event.event_id)
            new_state.setdefault("beliefs", {})
            new_state["beliefs"][belief_id] = {
                "proposition": event.payload.get("proposition", ""),
                "confidence": event.payload.get("confidence", 0.5),
                "source": event.payload.get("source", "unknown")
            }
        
        elif event_type == EventTypes.BELIEF_REMOVED:
            belief_id = event.payload.get("belief_id", event.event_id)
            if "beliefs" in new_state and belief_id in new_state["beliefs"]:
                del new_state["beliefs"][belief_id]
        
        elif event_type == EventTypes.CONTRADICTION_REGISTERED:
            episode_id = event.payload.get("episode_id", event.event_id)
            new_state.setdefault("contradictions", {})
            new_state["contradictions"][episode_id] = {
                "belief_ids": event.payload.get("belief_ids", []),
                "status": "unresolved"
            }
        
        elif event_type == EventTypes.CONTRADICTION_RESOLVED:
            episode_id = event.payload.get("episode_id", event.event_id)
            if "contradictions" in new_state and episode_id in new_state["contradictions"]:
                new_state["contradictions"][episode_id]["status"] = "resolved"
        
        elif event_type == EventTypes.TRANSACTION_COMPENSATED:
            tx_id = event.payload.get("original_transaction_id", event.event_id)
            new_state.setdefault("transactions", {})
            new_state["transactions"][tx_id] = {"status": "compensated"}
        
        return new_state