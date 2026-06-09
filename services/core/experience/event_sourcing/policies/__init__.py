"""
Domain Policies - Business logic separated from command handling.

Policies define the rules for when and how state can be mutated.
They are pure functions that take command + state and return decision.
"""
from dataclasses import dataclass
from typing import Dict, Any


class PolicyDecision:
    """Policy decision outcomes"""
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


@dataclass
class PolicyResult:
    """Result of policy evaluation"""
    decision: str
    reason: str = ""
    conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}


class BeliefPolicies:
    """Policies for belief management"""
    
    @staticmethod
    def add_belief_policy(command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Policy for adding beliefs."""
        payload = command.get("payload", {})
        proposition = payload.get("proposition", "")
        confidence = payload.get("confidence", 0.5)
        source = payload.get("source", "unknown")
        
        if not proposition:
            return PolicyResult(PolicyDecision.DENY, "Proposition cannot be empty")
        
        if not 0 <= confidence <= 1:
            return PolicyResult(PolicyDecision.DENY, "Confidence must be in [0, 1]")
        
        beliefs = state.get("beliefs", {})
        for bid, belief in beliefs.items():
            if hasattr(belief, 'proposition') and hasattr(belief, 'source'):
                if belief.proposition == proposition and belief.source == source:
                    return PolicyResult(
                        PolicyDecision.CONDITIONAL, 
                        "Duplicate belief, will update",
                        {"existing_id": bid}
                    )
        
        return PolicyResult(PolicyDecision.ALLOW, "Belief addition allowed")
    
    @staticmethod
    def update_belief_policy(command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Policy for updating beliefs."""
        payload = command.get("payload", {})
        belief_id = command.get("target_id", payload.get("belief_id", ""))
        new_confidence = payload.get("confidence", 0.5)
        
        beliefs = state.get("beliefs", {})
        if belief_id not in beliefs:
            return PolicyResult(PolicyDecision.DENY, "Belief does not exist")
        
        belief = beliefs[belief_id]
        old_confidence = belief.confidence if hasattr(belief, 'confidence') else 0.5
        delta = abs(new_confidence - old_confidence)
        
        if delta > 0.5:
            return PolicyResult(
                PolicyDecision.DENY, 
                f"Confidence delta too large: {delta}. Max allowed: 0.5"
            )
        
        return PolicyResult(PolicyDecision.ALLOW, "Belief update allowed")
    
    @staticmethod
    def remove_belief_policy(command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Policy for removing beliefs."""
        payload = command.get("payload", {})
        belief_id = command.get("target_id", payload.get("belief_id", ""))
        
        beliefs = state.get("beliefs", {})
        if belief_id not in beliefs:
            return PolicyResult(PolicyDecision.DENY, "Belief does not exist")
        
        belief = beliefs[belief_id]
        confidence = belief.confidence if hasattr(belief, 'confidence') else 0.5
        if confidence >= 0.9:
            return PolicyResult(
                PolicyDecision.CONDITIONAL,
                "High-confidence belief, requires explicit confirmation",
                {"requires_confirmation": True}
            )
        
        return PolicyResult(PolicyDecision.ALLOW, "Belief removal allowed")


class ContradictionPolicies:
    """Policies for contradiction handling"""
    
    @staticmethod
    def register_contradiction_policy(command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Policy for registering contradictions."""
        payload = command.get("payload", {})
        belief_ids = payload.get("belief_ids", [])
        
        if len(belief_ids) < 2:
            return PolicyResult(PolicyDecision.DENY, "Need at least 2 beliefs for contradiction")
        
        beliefs = state.get("beliefs", {})
        for bid in belief_ids:
            if bid not in beliefs:
                return PolicyResult(PolicyDecision.DENY, f"Belief {bid} does not exist")
        
        contradictions = state.get("contradictions", {})
        for cid, contr in contradictions.items():
            if hasattr(contr, 'belief_ids') and set(contr.belief_ids) == set(belief_ids):
                return PolicyResult(
                    PolicyDecision.CONDITIONAL,
                    "Similar contradiction exists",
                    {"existing_id": cid}
                )
        
        return PolicyResult(PolicyDecision.ALLOW, "Contradiction registration allowed")
    
    @staticmethod
    def resolve_contradiction_policy(command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Policy for resolving contradictions."""
        payload = command.get("payload", {})
        episode_id = command.get("target_id", payload.get("episode_id", ""))
        
        contradictions = state.get("contradictions", {})
        if episode_id not in contradictions:
            return PolicyResult(PolicyDecision.DENY, "Contradiction does not exist")
        
        contr = contradictions[episode_id]
        status = contr.resolution_status if hasattr(contr, 'resolution_status') else "unresolved"
        if status == "resolved":
            return PolicyResult(PolicyDecision.DENY, "Contradiction already resolved")
        
        return PolicyResult(PolicyDecision.ALLOW, "Contradiction resolution allowed")


class TransactionPolicies:
    """Policies for transaction management"""
    
    @staticmethod
    def compensate_transaction_policy(command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Policy for compensating transactions."""
        payload = command.get("payload", {})
        tx_id = payload.get("original_transaction_id", command.get("target_id", ""))
        
        transactions = state.get("transactions", {})
        if tx_id not in transactions:
            return PolicyResult(PolicyDecision.DENY, "Transaction does not exist")
        
        tx = transactions[tx_id]
        status = tx.status if hasattr(tx, 'status') else "committed"
        if status == "compensated":
            return PolicyResult(PolicyDecision.DENY, "Transaction already compensated")
        
        return PolicyResult(PolicyDecision.ALLOW, "Transaction compensation allowed")


class PolicyEngine:
    """Central policy engine."""
    
    POLICY_MAP = {
        "add_belief": BeliefPolicies.add_belief_policy,
        "update_belief": BeliefPolicies.update_belief_policy,
        "remove_belief": BeliefPolicies.remove_belief_policy,
        "register_contradiction": ContradictionPolicies.register_contradiction_policy,
        "resolve_contradiction": ContradictionPolicies.resolve_contradiction_policy,
        "compensate_transaction": TransactionPolicies.compensate_transaction_policy,
    }
    
    @classmethod
    def evaluate(cls, command: Dict[str, Any], state: Dict[str, Any]) -> PolicyResult:
        """Evaluate policy for command"""
        command_type = command.get("command_type", "")
        
        policy = cls.POLICY_MAP.get(command_type)
        if policy is None:
            return PolicyResult(PolicyDecision.ALLOW, "No policy defined, allowing")
        
        return policy(command, state)