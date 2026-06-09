"""
Immutable State Structures for Cognitive Runtime.

All state is represented as frozen dataclasses.
No mutations - only new instances.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import hashlib
import json


@dataclass(frozen=True)
class BeliefState:
    """Immutable belief structure"""
    belief_id: str
    proposition: str
    confidence: float
    entropy: float
    source: str
    created_at: str
    last_updated: str
    incoming_causes: Tuple[str, ...]
    outgoing_effects: Tuple[str, ...]
    attractor_state: Optional[str] = None


@dataclass(frozen=True)
class CausalEdgeState:
    """Immutable causal edge"""
    edge_id: str
    cause_ids: Tuple[str, ...]
    effect_ids: Tuple[str, ...]
    weight: float
    evidence_strength: float
    temporal_distance: int
    created_at: str
    policy_mediation: Optional[str] = None


@dataclass(frozen=True)
class ContradictionState:
    """Immutable contradiction"""
    episode_id: str
    belief_ids: Tuple[str, ...]
    contradiction_type: str
    first_seen: str
    last_seen: str
    recurrence_count: int
    stability_score: float
    resolution_status: str
    severity: str


@dataclass(frozen=True)
class TransactionRecord:
    """Immutable transaction record"""
    transaction_id: str
    status: str
    created_at: str
    compensated_at: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class CognitiveState:
    """
    Root state for the cognitive runtime.
    
    Uses dicts internally for fast access.
    Tuples are derived views for immutability semantics.
    """
    beliefs: Tuple[str, ...] = field(default_factory=tuple)
    causal_edges: Tuple[str, ...] = field(default_factory=tuple)
    contradictions: Tuple[str, ...] = field(default_factory=tuple)
    transactions: Tuple[str, ...] = field(default_factory=tuple)
    total_entropy: float = 0.0
    belief_count: int = 0
    version: int = 0
    
    _beliefs_dict: Dict[str, BeliefState] = field(default_factory=dict)
    _causal_edges_dict: Dict[str, CausalEdgeState] = field(default_factory=dict)
    _contradictions_dict: Dict[str, ContradictionState] = field(default_factory=dict)
    _transactions_dict: Dict[str, TransactionRecord] = field(default_factory=dict)
    
    def __post_init__(self):
        if self._beliefs_dict and not self.beliefs:
            object.__setattr__(self, 'beliefs', tuple(sorted(self._beliefs_dict.keys())))
        if self._causal_edges_dict and not self.causal_edges:
            object.__setattr__(self, 'causal_edges', tuple(sorted(self._causal_edges_dict.keys())))
        if self._contradictions_dict and not self.contradictions:
            object.__setattr__(self, 'contradictions', tuple(sorted(self._contradictions_dict.keys())))
        if self._transactions_dict and not self.transactions:
            object.__setattr__(self, 'transactions', tuple(sorted(self._transactions_dict.keys())))
    
    @staticmethod
    def compute_hash(state: 'CognitiveState') -> str:
        """Deterministic state hash for replay verification"""
        data = {
            "version": state.version,
            "belief_count": state.belief_count,
            "total_entropy": state.total_entropy,
            "beliefs": sorted(state.beliefs),
            "causal_edges": sorted(state.causal_edges),
            "contradictions": sorted(state.contradictions),
            "transactions": sorted(state.transactions)
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()


def initial_state() -> CognitiveState:
    """Create initial empty state"""
    return CognitiveState(
        beliefs=(),
        causal_edges=(),
        contradictions=(),
        transactions=(),
        total_entropy=0.0,
        belief_count=0,
        version=0,
        _beliefs_dict={},
        _causal_edges_dict={},
        _contradictions_dict={},
        _transactions_dict={}
    )


def create_state_from_dicts(
    beliefs: Dict[str, BeliefState],
    causal_edges: Dict[str, CausalEdgeState],
    contradictions: Dict[str, ContradictionState],
    transactions: Dict[str, TransactionRecord],
    total_entropy: float = 0.0,
    version: int = 0
) -> CognitiveState:
    """Create CognitiveState from dictionaries"""
    return CognitiveState(
        beliefs=tuple(sorted(beliefs.keys())),
        causal_edges=tuple(sorted(causal_edges.keys())),
        contradictions=tuple(sorted(contradictions.keys())),
        transactions=tuple(sorted(transactions.keys())),
        total_entropy=total_entropy,
        belief_count=len(beliefs),
        version=version,
        _beliefs_dict=dict(beliefs),
        _causal_edges_dict=dict(causal_edges),
        _contradictions_dict=dict(contradictions),
        _transactions_dict=dict(transactions)
    )