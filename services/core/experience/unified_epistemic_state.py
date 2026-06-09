"""
Unified Epistemic State (UES) - Canonical world-state for cognitive operations

Provides:
- Immutable snapshots (version graph) with deep copy
- Pure vector clock for causal ordering
- Proper state transitions (derive → apply → commit)
- Invariant validation before commit
- WAL-based event sourcing for deterministic replay

Key principle:
    Everything reads/writes through UES.
    No more stale snapshots between layers.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from copy import deepcopy
import hashlib
import json

from wal_engine import WALEngine, WALEventType, get_wal_engine
import hashlib
import json


class EpistemicClock:
    """
    Pure vector clock for causal ordering.
    
    No mixed scalar/vector semantics - pure Lamport-style vector clock.
    """
    
    def __init__(self, initial: Optional[Dict[str, int]] = None):
        self._vector: Dict[str, int] = initial or {}
    
    def tick(self, node_id: str) -> int:
        """Advance clock for node (Lamport increment)"""
        current = self._vector.get(node_id, 0)
        self._vector[node_id] = current + 1
        return self._vector[node_id]
    
    def increment(self, node_id: str) -> int:
        """Alias for tick - increment node's local counter"""
        return self.tick(node_id)
    
    def get(self, node_id: str) -> int:
        """Get node's clock value"""
        return self._vector.get(node_id, 0)
    
    def merge(self, other: "EpistemicClock") -> "EpistemicClock":
        """Merge two clocks (take max of each component)"""
        merged = EpistemicClock()
        all_nodes = set(self._vector.keys()) | set(other._vector.keys())
        for node in all_nodes:
            merged._vector[node] = max(
                self._vector.get(node, 0),
                other._vector.get(node, 0)
            )
        return merged
    
    def happens_before(self, other: "EpistemicClock") -> bool:
        """Check if this clock happens before other"""
        for node_id in self._vector:
            if self._vector[node_id] > other._vector.get(node_id, 0):
                return False
        return True
    
    def copy(self) -> "EpistemicClock":
        """Create copy of clock"""
        return EpistemicClock(self._vector.copy())
    
    def to_dict(self) -> dict:
        return self._vector.copy()


class EpistemicEventType(Enum):
    """Types of epistemic events for event sourcing"""
    BELIEF_ADDED = "belief_added"
    BELIEF_UPDATED = "belief_updated"
    BELIEF_REMOVED = "belief_removed"
    CONTRADICTION_DETECTED = "contradiction_detected"
    CONTRADICTION_RESOLVED = "contradiction_resolved"
    CAUSAL_EDGE_ADDED = "causal_edge_added"
    CAUSAL_EDGE_STRENGTHENED = "causal_edge_strengthened"
    REFLECTION_APPLIED = "reflection_applied"
    CONSTRAINT_ADDED = "constraint_added"
    STATE_COMMITTED = "state_committed"
    
    def increment(self, node_id: str) -> int:
        """Increment node's local counter"""
        if node_id not in self._vector:
            self._vector[node_id] = 0
        self._vector[node_id] += 1
        return self._vector[node_id]
    
    def get(self, node_id: str) -> int:
        """Get node's clock value"""
        return self._vector.get(node_id, 0)
    
    def happens_before(self, other: "EpistemicClock") -> bool:
        """Check if this clock happens before other"""
        for node_id in self._vector:
            if self._vector[node_id] > other._vector.get(node_id, 0):
                return False
        return True
    
    def merge(self, other: "EpistemicClock") -> "EpistemicClock":
        """Merge two clocks (take max of each)"""
        merged = EpistemicClock()
        all_nodes = set(self._vector.keys()) | set(other._vector.keys())
        for node in all_nodes:
            merged._vector[node] = max(
                self._vector.get(node, 0),
                other._vector.get(node, 0)
            )
        return merged
    
    def to_dict(self) -> dict:
        return self._vector.copy()


@dataclass
class BeliefState:
    """Belief snapshot in UES"""
    belief_id: str
    proposition: str
    confidence: float
    entropy: float  # Uncertainty measure
    source: str  # "experience", "inference", "reflection"
    created_at: str
    last_updated: str
    version: int  # Epistemic version
    
    # For hypergraph
    incoming_causes: List[str] = field(default_factory=list)
    outgoing_effects: List[str] = field(default_factory=list)
    
    # Attractor state
    attractor_state: str = "stable"  # stable, oscillating, converging, diverging


@dataclass
class ConstraintState:
    """Constraint snapshot in UES"""
    constraint_id: str
    belief_id: str
    constraint_type: str
    predicate: str
    domain: str
    confidence: float = 1.0
    operator: Optional[str] = None
    value: Optional[Any] = None


@dataclass
class ContradictionState:
    """Contradiction snapshot in UES"""
    episode_id: str
    belief_ids: List[str]
    contradiction_type: str
    first_seen: str
    last_seen: str
    recurrence_count: int
    stability_score: float
    resolution_status: str
    severity: str


@dataclass
class CausalEdgeState:
    """Causal edge snapshot in UES"""
    edge_id: str
    cause_ids: List[str]  # Multiple causes (hyperedge)
    effect_ids: List[str]  # Multiple effects
    weight: float
    evidence_strength: float
    temporal_distance: int
    created_at: str
    policy_mediation: Optional[str] = None


@dataclass
class StateDiff:
    """Diff between two UES snapshots"""
    diff_id: str
    from_version: int
    to_version: int
    trigger: str
    timestamp: str
    added_beliefs: List[str] = field(default_factory=list)
    removed_beliefs: List[str] = field(default_factory=list)
    modified_beliefs: List[Dict[str, Any]] = field(default_factory=list)
    added_constraints: List[str] = field(default_factory=list)
    resolved_contradictions: List[str] = field(default_factory=list)
    new_contradictions: List[str] = field(default_factory=list)
    new_causal_edges: List[str] = field(default_factory=list)
    strengthened_edges: List[str] = field(default_factory=list)
    confidence_delta: float = 0.0
    entropy_delta: float = 0.0
    contradiction_density: float = 0.0
    causal_chain: List[str] = field(default_factory=list)


@dataclass 
class InvariantCheck:
    """Invariant validation result"""
    invariant_name: str
    passed: bool
    details: str
    affected_beliefs: List[str] = field(default_factory=list)


@dataclass
class UnifiedEpistemicState:
    """
    Unified Epistemic State - CANONICAL immutable world-state.
    
    Immutable snapshots via version graph.
    Canonical state is NEVER modified - only new versions are created.
    """
    state_id: str
    version: int
    beliefs: Dict[str, BeliefState]
    constraints: Dict[str, ConstraintState]
    contradictions: Dict[str, ContradictionState]
    causal_edges: Dict[str, CausalEdgeState]
    epistemic_clock: Dict[str, int]
    created_at: str
    parent_version: Optional[int]
    total_entropy: float
    belief_count: int
    contradiction_count: int
    causal_density: float
    state_hash: str
    invariant_results: List[InvariantCheck] = field(default_factory=list)
    
    def clone(self) -> "UnifiedEpistemicState":
        """Create MUTABLE working copy from canonical state."""
        return UnifiedEpistemicState(
            state_id=str(uuid4()),
            version=-1,  # Working copy has negative version
            beliefs=deepcopy(self.beliefs),
            constraints=deepcopy(self.constraints),
            contradictions=deepcopy(self.contradictions),
            causal_edges=deepcopy(self.causal_edges),
            epistemic_clock=self.epistemic_clock.copy(),
            created_at=datetime.utcnow().isoformat(),
            parent_version=self.version,
            total_entropy=self.total_entropy,
            belief_count=self.belief_count,
            contradiction_count=self.contradiction_count,
            causal_density=self.causal_density,
            state_hash="",  # Will compute on commit
            invariant_results=[]
        )
    
    def compute_hash(self) -> str:
        """Compute complete state hash for integrity (includes ALL state components)"""
        content = {
            "version": self.version,
            "parent": self.parent_version,
            "beliefs": {
                k: {
                    "proposition": v.proposition,
                    "confidence": v.confidence,
                    "entropy": v.entropy,
                    "attractor": v.attractor_state,
                    "version": v.version
                } for k, v in self.beliefs.items()
            },
            "contradictions": {
                k: {
                    "belief_ids": v.belief_ids,
                    "status": v.resolution_status,
                    "stability": v.stability_score
                } for k, v in self.contradictions.items()
            },
            "causal_edges": {
                k: {
                    "cause_ids": v.cause_ids,
                    "effect_ids": v.effect_ids,
                    "weight": v.weight
                } for k, v in self.causal_edges.items()
            },
            "total_entropy": self.total_entropy,
            "causal_density": self.causal_density
        }
        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "version": self.version,
            "belief_count": len(self.beliefs),
            "constraint_count": len(self.constraints),
            "contradiction_count": len(self.contradictions),
            "edge_count": len(self.causal_edges),
            "total_entropy": self.total_entropy,
            "created_at": self.created_at,
            "parent_version": self.parent_version
        }


class UnifiedEpistemicStateManager:
    """
    UES Manager - Single source of truth for all cognitive layers.
    
    Architecture: Canonical/Working Copy pattern
    - Canonical state: immutable, stored in version graph
    - Working copy: mutable, cloned from canonical for mutations
    - Commit: atomic pointer swap from working copy to canonical
    - Rollback: discard working copy, canonical untouched
    
    Key invariant:
        All mutations happen on working copy, never directly on canonical.
        Commit validates then atomically replaces canonical.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Version graph (canonical immutable states)
        self._versions: Dict[int, UnifiedEpistemicState] = {}
        self._current_version: int = 0
        self._clock = EpistemicClock()
        
        # Working copy - the ONLY mutable state
        self._working_copy: Optional[UnifiedEpistemicState] = None
        self._working_copy_base_version: int = -1
        
        # WAL engine for deterministic replay
        self._wal = get_wal_engine()
        
        # Initial state
        self._create_initial_state()
        
        # Invariant checkers
        self._invariant_checkers: List[Callable] = []
    
    def has_working_copy(self) -> bool:
        """Check if working copy exists"""
        return self._working_copy is not None
    
    def create_working_copy(self) -> UnifiedEpistemicState:
        """
        Create working copy from current canonical state.
        
        This is the ONLY way to get a mutable state.
        """
        canonical = self.get_current_state()
        
        self._working_copy = canonical.clone()
        self._working_copy_base_version = canonical.version
        
        return self._working_copy
    
    def get_working_copy(self) -> Optional[UnifiedEpistemicState]:
        """Get current working copy"""
        return self._working_copy
    
    def discard_working_copy(self) -> bool:
        """
        Discard working copy - rollback semantics.
        
        True rollback: just discard, don't touch canonical.
        """
        if self._working_copy is None:
            return False
        
        self._working_copy = None
        self._working_copy_base_version = -1
        return True
    
    def commit_working_copy(self, reason: str = "mutation") -> bool:
        """
        Commit working copy to canonical - atomic pointer swap.
        
        Steps:
        1. Validate working copy invariants
        2. If valid: create new canonical version from working copy
        3. If invalid: raise error (working copy still exists)
        """
        if self._working_copy is None:
            raise ValueError("No working copy to commit")
        
        # Validate invariants before commit
        invariants = self._validate_invariants(self._working_copy)
        self._working_copy.invariant_results = invariants
        
        critical_failures = [i for i in invariants if not i.passed and i.details.startswith("CRITICAL")]
        if critical_failures:
            raise ValueError(f"Invariant validation failed on commit: {critical_failures}")
        
        # Compute metrics for new version
        self._working_copy.total_entropy = sum(b.entropy for b in self._working_copy.beliefs.values())
        self._working_copy.belief_count = len(self._working_copy.beliefs)
        edge_count = len(self._working_copy.causal_edges)
        self._working_copy.causal_density = edge_count / max(self._working_copy.belief_count, 1)
        
        # Create new canonical version (atomic pointer swap)
        new_version = self._current_version + 1
        
        # Get state hash BEFORE commit for WAL
        old_state = self.get_current_state()
        state_hash_before = old_state.state_hash
        
        self._working_copy.version = new_version
        self._working_copy.state_hash = self._working_copy.compute_hash()
        state_hash_after = self._working_copy.state_hash
        
        # Serialize full state for snapshot
        full_state = {
            "beliefs": {k: {"proposition": v.proposition, "confidence": v.confidence, "entropy": v.entropy, "source": v.source} 
                       for k, v in self._working_copy.beliefs.items()},
            "contradictions": {k: {"belief_ids": v.belief_ids, "contradiction_type": v.contradiction_type, "stability_score": v.stability_score} 
                              for k, v in self._working_copy.contradictions.items()},
            "causal_edges": {k: {"cause_ids": v.cause_ids, "effect_ids": v.effect_ids, "weight": v.weight} 
                           for k, v in self._working_copy.causal_edges.items()},
            "total_entropy": self._working_copy.total_entropy,
            "belief_count": self._working_copy.belief_count,
            "causal_density": self._working_copy.causal_density
        }
        
        # Log STATE_COMMITTED to WAL with hash chain
        self._wal.log_event(
            event_type=WALEventType.STATE_COMMITTED,
            version=new_version,
            operation="commit",
            target_id=f"state_v{new_version}",
            payload={
                "belief_count": self._working_copy.belief_count,
                "total_entropy": self._working_copy.total_entropy,
                "causal_density": self._working_copy.causal_density,
                "contradiction_count": len(self._working_copy.contradictions),
                "edge_count": len(self._working_copy.causal_edges)
            },
            parent_version=self._current_version,
            actor="ues",
            entropy_delta=0.0,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
            full_state=full_state
        )
        
        # Store in version graph
        self._versions[new_version] = self._working_copy
        self._current_version = new_version
        
        # Clear working copy
        self._working_copy = None
        self._working_copy_base_version = -1
        
        return True
    
    def _create_initial_state(self):
        """Create initial empty state"""
        state = UnifiedEpistemicState(
            state_id=str(uuid4()),
            version=0,
            beliefs={},
            constraints={},
            contradictions={},
            causal_edges={},
            epistemic_clock={},
            created_at=datetime.utcnow().isoformat(),
            parent_version=None,
            total_entropy=0.0,
            belief_count=0,
            contradiction_count=0,
            causal_density=0.0,
            state_hash=""
        )
        state.state_hash = state.compute_hash()
        self._versions[0] = state
    
    def get_current_state(self) -> UnifiedEpistemicState:
        """Get current state snapshot"""
        return self._versions[self._current_version]
    
    def get_state(self, version: int) -> Optional[UnifiedEpistemicState]:
        """Get specific version"""
        return self._versions.get(version)
    
    def add_belief(
        self,
        belief_id: str,
        proposition: str,
        confidence: float,
        entropy: float,
        source: str = "experience",
        causes: Optional[List[str]] = None
    ) -> BeliefState:
        """
        Add belief through UES using canonical/working copy pattern.
        
        Steps:
        1. Create/get working copy
        2. Mutate working copy
        3. Commit working copy to canonical
        """
        
        # Create or get working copy
        if self._working_copy is None:
            self.create_working_copy()
        
        working = self._working_copy
        
        belief = BeliefState(
            belief_id=belief_id,
            proposition=proposition,
            confidence=confidence,
            entropy=entropy,
            source=source,
            created_at=datetime.utcnow().isoformat(),
            last_updated=datetime.utcnow().isoformat(),
            version=-1,  # Working copy version
            incoming_causes=causes or [],
            outgoing_effects=[]
        )
        
        # Mutate working copy (NOT canonical)
        working.beliefs[belief_id] = belief
        
        # Log to WAL BEFORE commit (Write-Ahead Log guarantee)
        self._wal.log_event(
            event_type=WALEventType.BELIEF_ADDED,
            version=self._current_version + 1,
            operation="add",
            target_id=belief_id,
            payload={
                "proposition": proposition,
                "confidence": confidence,
                "entropy": entropy,
                "source": source,
                "created_at": belief.created_at,
                "incoming_causes": causes or [],
                "outgoing_effects": []
            },
            parent_version=self._current_version,
            actor="ues",
            entropy_delta=entropy
        )
        
        # Update clock
        self._clock.tick(belief_id)
        
        # Multi-factor causal linking for identity cohesion
        self._reinforce_causal_links(working, belief_id, source)
        
        # Commit working copy to canonical (atomic pointer swap)
        self.commit_working_copy(reason=f"belief_added:{belief_id}")
        
        return belief
    
    def _reinforce_causal_links(
        self,
        state,
        new_belief_id: str,
        source: str
    ):
        """
        Multi-factor causal reinforcement for semantic identity.
        
        Links based on:
        1. Temporal proximity (temporal)
        2. Shared contradiction participation (contradiction)  
        3. Reflection co-activation (reflection)
        4. Shared source (origin)
        """
        
        if len(state.beliefs) <= 1:
            return
        
        candidate_scores: Dict[str, float] = {}
        
        existing_beliefs = {k: v for k, v in state.beliefs.items() if k != new_belief_id}
        
        for bid, belief in existing_beliefs.items():
            score = 0.0
            
            # Factor 1: Temporal proximity (last 5 beliefs)
            recent_keys = list(state.beliefs.keys())[:-1][-5:]
            if bid in recent_keys:
                idx = recent_keys.index(bid)
                score += (0.5 - idx * 0.1)  # Decaying score
            
            # Factor 2: Shared contradiction participation
            for epid, contra in state.contradictions.items():
                if new_belief_id in contra.belief_ids and bid in contra.belief_ids:
                    score += 0.3 * contra.stability_score  # Stronger for stable contradictions
            
            # Factor 3: Shared source (origin)
            if belief.source == source:
                score += 0.2
            
            # Factor 4: Attractor similarity (same attractor = stronger cohesion)
            if belief.attractor_state == state.beliefs.get(new_belief_id, belief).attractor_state:
                score += 0.15
            
            if score > 0.1:
                candidate_scores[bid] = score
        
        # Create edges to top candidates (max 3)
        top_candidates = sorted(candidate_scores.items(), key=lambda x: -x[1])[:3]
        
        for bid, score in top_candidates:
            edge = CausalEdgeState(
                edge_id=str(uuid4()),
                cause_ids=[bid],
                effect_ids=[new_belief_id],
                weight=min(score, 1.0),
                evidence_strength=score / 1.0,  # Normalized
                temporal_distance=1,
                created_at=datetime.utcnow().isoformat()
            )
            state.causal_edges[edge.edge_id] = edge
            
            # Update node references
            state.beliefs[bid].outgoing_effects.append(new_belief_id)
            state.beliefs[new_belief_id].incoming_causes.append(bid)
    
    def update_belief(
        self,
        belief_id: str,
        new_confidence: float,
        new_entropy: float,
        new_attractor: Optional[str] = None
    ) -> Optional[BeliefState]:
        """
        Update belief through UES using canonical/working copy pattern.
        """
        
        # Create or get working copy
        if self._working_copy is None:
            self.create_working_copy()
        
        working = self._working_copy
        
        if belief_id not in working.beliefs:
            return None
        
        belief = working.beliefs[belief_id]
        
        belief.confidence = new_confidence
        belief.entropy = new_entropy
        belief.last_updated = datetime.utcnow().isoformat()
        belief.version = -1  # Working copy
        
        if new_attractor:
            belief.attractor_state = new_attractor
        
        # Log to WAL
        entropy_delta = new_entropy - belief.entropy
        self._wal.log_event(
            event_type=WALEventType.BELIEF_UPDATED,
            version=self._current_version + 1,
            operation="update",
            target_id=belief_id,
            payload={
                "confidence": new_confidence,
                "entropy": new_entropy,
                "attractor_state": new_attractor or belief.attractor_state
            },
            parent_version=self._current_version,
            actor="ues",
            entropy_delta=entropy_delta
        )
        
        # Update clock
        self._clock.tick(belief_id)
        
        # Commit working copy to canonical
        self.commit_working_copy(reason=f"belief_updated:{belief_id}")
        
        return belief
    
    def add_causal_edge(
        self,
        cause_ids: List[str],
        effect_ids: List[str],
        weight: float = 0.5,
        evidence_strength: float = 0.5,
        policy_mediation: Optional[str] = None
    ) -> CausalEdgeState:
        """
        Add hyperedge (multiple causes -> multiple effects) using canonical/working copy pattern.
        """
        
        # Create or get working copy
        if self._working_copy is None:
            self.create_working_copy()
        
        working = self._working_copy
        
        edge = CausalEdgeState(
            edge_id=str(uuid4()),
            cause_ids=cause_ids,
            effect_ids=effect_ids,
            weight=weight,
            evidence_strength=evidence_strength,
            temporal_distance=1,
            created_at=datetime.utcnow().isoformat(),
            policy_mediation=policy_mediation
        )
        
        # Mutate working copy (NOT canonical)
        working.causal_edges[edge.edge_id] = edge
        
        # Log to WAL
        self._wal.log_event(
            event_type=WALEventType.CAUSAL_EDGE_ADDED,
            version=self._current_version + 1,
            operation="add",
            target_id=edge.edge_id,
            payload={
                "cause_ids": cause_ids,
                "effect_ids": effect_ids,
                "weight": weight,
                "evidence_strength": evidence_strength,
                "created_at": edge.created_at
            },
            parent_version=self._current_version,
            actor="ues",
            entropy_delta=0.0
        )
        
        # Update incoming/effects for beliefs in working copy
        for cause_id in cause_ids:
            if cause_id in working.beliefs:
                for e in effect_ids:
                    if e not in working.beliefs[cause_id].outgoing_effects:
                        working.beliefs[cause_id].outgoing_effects.append(e)
        for effect_id in effect_ids:
            if effect_id in working.beliefs:
                for c in cause_ids:
                    if c not in working.beliefs[effect_id].incoming_causes:
                        working.beliefs[effect_id].incoming_causes.append(c)
        
        # Commit working copy to canonical
        self.commit_working_copy(reason=f"causal_edge_added:{edge.edge_id}")
        
        return edge
    
    def register_contradiction(
        self,
        episode_id: str,
        belief_ids: List[str],
        contradiction_type: str,
        severity: str
    ) -> ContradictionState:
        """
        Register contradiction through UES using canonical/working copy pattern.
        """
        
        # Create or get working copy
        if self._working_copy is None:
            self.create_working_copy()
        
        working = self._working_copy
        
        now = datetime.utcnow().isoformat()
        
        contradiction = ContradictionState(
            episode_id=episode_id,
            belief_ids=belief_ids,
            contradiction_type=contradiction_type,
            first_seen=now,
            last_seen=now,
            recurrence_count=1,
            stability_score=0.5,
            resolution_status="unresolved",
            severity=severity
        )
        
        # Mutate working copy (NOT canonical)
        working.contradictions[episode_id] = contradiction
        
        # Log to WAL
        self._wal.log_event(
            event_type=WALEventType.CONTRADICTION_REGISTERED,
            version=self._current_version + 1,
            operation="add",
            target_id=episode_id,
            payload={
                "belief_ids": belief_ids,
                "contradiction_type": contradiction_type,
                "severity": severity,
                "first_seen": now,
                "recurrence_count": 1,
                "stability_score": 0.5
            },
            parent_version=self._current_version,
            actor="ues",
            entropy_delta=0.0
        )
        
        # Commit working copy to canonical
        self.commit_working_copy(reason=f"contradiction_detected:{episode_id}")
        
        return contradiction
    
    def _validate_invariants(self, state: UnifiedEpistemicState) -> List[InvariantCheck]:
        """Validate state invariants before commit"""
        
        results = []
        
        # 1. Confidence range: all beliefs must have confidence in [0, 1]
        for bid, belief in state.beliefs.items():
            if not (0 <= belief.confidence <= 1):
                results.append(InvariantCheck(
                    invariant_name="confidence_range",
                    passed=False,
                    details=f"CRITICAL: Belief {bid} has confidence {belief.confidence} outside [0,1]",
                    affected_beliefs=[bid]
                ))
            else:
                results.append(InvariantCheck(
                    invariant_name="confidence_range",
                    passed=True,
                    details="OK",
                    affected_beliefs=[bid]
                ))
        
        # 2. Entropy non-negative
        if state.total_entropy < 0:
            results.append(InvariantCheck(
                invariant_name="entropy_non_negative",
                passed=False,
                details=f"CRITICAL: Total entropy is negative: {state.total_entropy}",
                affected_beliefs=list(state.beliefs.keys())
            ))
        
        # 3. Causal edges reference existing beliefs
        for eid, edge in state.causal_edges.items():
            for cause in edge.cause_ids:
                if cause not in state.beliefs:
                    results.append(InvariantCheck(
                        invariant_name="causal_edge_integrity",
                        passed=False,
                        details=f"CRITICAL: Edge {eid} references non-existent cause {cause}",
                        affected_beliefs=[cause]
                    ))
            for effect in edge.effect_ids:
                if effect not in state.beliefs:
                    results.append(InvariantCheck(
                        invariant_name="causal_edge_integrity",
                        passed=False,
                        details=f"CRITICAL: Edge {eid} references non-existent effect {effect}",
                        affected_beliefs=[effect]
                    ))
        
        # 4. Contradictions reference existing beliefs
        for epid, contra in state.contradictions.items():
            for bid in contra.belief_ids:
                if bid not in state.beliefs:
                    results.append(InvariantCheck(
                        invariant_name="contradiction_integrity",
                        passed=False,
                        details=f"CRITICAL: Contradiction {epid} references non-existent belief {bid}",
                        affected_beliefs=[bid]
                    ))
        
        return results
    
    def compute_diff(self, from_version: int, to_version: int) -> Optional[StateDiff]:
        """
        Compute SEMANTIC diff between two versions.
        
        Not just entropy delta, but actual belief changes.
        """
        
        if from_version >= to_version:
            return None
        
        from_state = self._versions.get(from_version)
        to_state = self._versions.get(to_version)
        
        if not from_state or not to_state:
            return None
        
        # Compute semantic changes
        added = [b for b in to_state.beliefs if b not in from_state.beliefs]
        removed = [b for b in from_state.beliefs if b not in to_state.beliefs]
        
        # Find modified beliefs with semantic delta
        modified = []
        for bid in from_state.beliefs:
            if bid in to_state.beliefs:
                old = from_state.beliefs[bid]
                new = to_state.beliefs[bid]
                if old.confidence != new.confidence or old.entropy != new.entropy:
                    modified.append({
                        "belief_id": bid,
                        "old_confidence": old.confidence,
                        "new_confidence": new.confidence,
                        "confidence_delta": new.confidence - old.confidence,
                        "old_entropy": old.entropy,
                        "new_entropy": new.entropy,
                        "attractor_changed": old.attractor_state != new.attractor_state,
                        "semantic_change": self._classify_belief_change(old, new)
                    })
        
        # Find new/resolved contradictions
        new_contradictions = [c for c in to_state.contradictions if c not in from_state.contradictions]
        resolved_contradictions = [c for c in from_state.contradictions if c not in to_state.contradictions]
        
        # Find causal edge changes
        new_edges = [e for e in to_state.causal_edges if e not in from_state.causal_edges]
        
        diff = StateDiff(
            diff_id=str(uuid4()),
            from_version=from_version,
            to_version=to_version,
            trigger="semantic_diff",
            timestamp=datetime.utcnow().isoformat(),
            added_beliefs=added,
            removed_beliefs=removed,
            modified_beliefs=modified,
            new_contradictions=new_contradictions,
            resolved_contradictions=resolved_contradictions,
            new_causal_edges=new_edges
        )
        
        # Compute proper metrics
        diff.entropy_delta = to_state.total_entropy - from_state.total_entropy
        
        # Compute contradiction density
        total_beliefs = max(len(to_state.beliefs), 1)
        diff.contradiction_density = len(to_state.contradictions) / total_beliefs
        
        return diff
    
    def _classify_belief_change(self, old: BeliefState, new: BeliefState) -> str:
        """Classify semantic change type"""
        
        conf_delta = new.confidence - old.confidence
        
        if abs(conf_delta) < 0.1:
            return "stable"
        elif conf_delta > 0.3:
            return "strengthened"
        elif conf_delta < -0.3:
            return "weakened"
        elif old.attractor_state != new.attractor_state:
            return f"attractor_shift_{old.attractor_state}_to_{new.attractor_state}"
        
        return "adjusted"
    
    def get_history(self, limit: int = 10) -> List[UnifiedEpistemicState]:
        """Get recent state history"""
        
        versions = sorted(self._versions.keys(), reverse=True)
        return [self._versions[v] for v in versions[:limit]]
    
    def get_clock(self) -> Dict[str, int]:
        """Get current epistemic clock"""
        return self._clock.to_dict()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        
        state = self.get_current_state()
        return {
            "version": self._current_version,
            "belief_count": state.belief_count,
            "constraint_count": len(state.constraints),
            "contradiction_count": state.contradiction_count,
            "edge_count": len(state.causal_edges),
            "total_entropy": state.total_entropy,
            "causal_density": state.causal_density,
            "epistemic_clock": self._clock.to_dict()
        }
    
    def verify_replay(self) -> tuple[bool, str, Dict[str, Any]]:
        """
        Verify deterministic replay via WAL.
        
        Rebuilds state from event log and compares with canonical.
        
        Returns: (verified, message, details)
        """
        current = self.get_current_state()
        
        # Verify via WAL engine
        verified, msg = self._wal.verify_replay(current, from_version=0)
        
        # Get WAL info
        wal_info = self._wal.get_replay_info()
        
        details = {
            "canonical_version": self._current_version,
            "canonical_belief_count": current.belief_count,
            "canonical_entropy": current.total_entropy,
            "canonical_hash": current.state_hash,
            "wal_event_count": wal_info["event_count"],
            "wal_snapshot_count": wal_info["snapshot_count"],
            "rebuild_verified": verified
        }
        
        return verified, msg, details
    
    def get_wal_info(self) -> Dict[str, Any]:
        """Get WAL engine information"""
        return self._wal.get_replay_info()


# Global instance
_ues_manager: Optional[UnifiedEpistemicStateManager] = None


def get_ues_manager(config: Optional[Dict] = None) -> UnifiedEpistemicStateManager:
    """Get global UES manager"""
    global _ues_manager
    if _ues_manager is None:
        _ues_manager = UnifiedEpistemicStateManager(config)
    return _ues_manager


def get_current_state() -> UnifiedEpistemicState:
    """Get current epistemic state"""
    return get_ues_manager().get_current_state()