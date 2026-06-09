"""
Contradiction Memory - Track persistent, recurring, and unresolved contradictions

Provides:
- Contradiction episode tracking
- Recurrence detection
- Resolution attempts
- Stability scoring
- Memory vs snapshot distinction

Key distinction:
- snapshot contradiction: "A contradicts B right now"
- persistent contradiction: "A has contradicted B for 5 cycles"
- recurring contradiction: "A ↔ B contradiction resolved 3 times, keeps returning"
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


class ResolutionStatus(Enum):
    """Contradiction resolution status"""
    UNRESOLVED = "unresolved"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"  # Temporarily ignored
    RECURRING = "recurring"  # Keeps returning


class ContradictionType(Enum):
    """Types of contradictions"""
    DIRECT = "direct"  # A says X, B says not-X
    IMPLICIT = "implicit"  # A implies X, B implies not-X
    TEMPORAL = "temporal"  # A was true, now B is true
    POLICY = "policy"  # Policies conflict
    CONTEXT = "context"  # Same belief, different contexts
    META = "meta"  # Self-contradiction


@dataclass
class ResolutionAttempt:
    """Single resolution attempt"""
    attempt_id: str
    strategy: str
    hypothesis: str
    success: bool
    attempted_at: str
    belief_mutations: List[Dict[str, Any]] = field(default_factory=list)
    outcome_summary: Optional[str] = None


@dataclass
class ContradictionEpisode:
    """
    Persistent contradiction with full history.
    
    This is NOT a snapshot - it's tracked over time.
    """
    episode_id: str
    belief_ids: List[str]
    contradiction_type: ContradictionType
    first_seen: str
    last_seen: str
    recurrence_count: int
    resolution_status: ResolutionStatus
    stability_score: float
    activation_count: int
    severity: str
    resolution_attempts: List[ResolutionAttempt] = field(default_factory=list)
    dormant_since: Optional[str] = None
    affects_policies: List[str] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "belief_ids": self.belief_ids,
            "contradiction_type": self.contradiction_type.value,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "recurrence_count": self.recurrence_count,
            "resolution_status": self.resolution_status.value,
            "stability_score": self.stability_score,
            "activation_count": self.activation_count,
            "severity": self.severity,
            "affects_policies": self.affects_policies
        }


class ContradictionDetector:
    """Detect contradictions between beliefs (from SemanticDiffEngine)"""
    
    def __init__(self):
        self._contradiction_patterns = [
            (["reliable", "unreliable"], ["dependable", "unpredictable"]),
            (["trust", "distrust"], ["believe", "doubt"]),
            (["safe", "dangerous"], ["secure", "risky"]),
            (["fast", "slow"], ["efficient", "inefficient"]),
        ]
    
    def find_contradictions(
        self,
        beliefs: Dict[str, Dict]
    ) -> List[Tuple[str, str, ContradictionType]]:
        """Find all contradictions between beliefs"""
        
        contradictions = []
        belief_list = list(beliefs.values())
        
        for i, b1 in enumerate(belief_list):
            for b2 in belief_list[i+1:]:
                contra_type = self._check_contradiction(b1, b2)
                if contra_type:
                    contradictions.append((
                        b1.get("belief_id"),
                        b2.get("belief_id"),
                        contra_type
                    ))
        
        return contradictions
    
    def _check_contradiction(
        self,
        b1: Dict,
        b2: Dict
    ) -> Optional[ContradictionType]:
        """Check if two beliefs contradict"""
        
        p1 = b1.get("proposition", "").lower()
        p2 = b2.get("proposition", "").lower()
        
        for pos_group, neg_group in self._contradiction_patterns:
            pos_in_1 = any(kw in p1 for kw in pos_group)
            neg_in_1 = any(kw in p1 for kw in neg_group)
            pos_in_2 = any(kw in p2 for kw in pos_group)
            neg_in_2 = any(kw in p2 for kw in neg_group)
            
            if (pos_in_1 and neg_in_2) or (neg_in_1 and pos_in_2):
                return ContradictionType.DIRECT
        
        return None


class ResolutionEngine:
    """Attempt to resolve contradictions"""
    
    RESOLUTION_STRATEGIES = [
        "context_clarification",  # Different contexts, not contradiction
        "policy_prioritization",  # One policy wins
        "confidence_adjustment",  # Reduce confidence to resolve
        "belief_deprecation",  # Mark one as obsolete
        "meta_belief_creation",  # Create higher-order belief about contradiction
        "temporal_aging",  # Old belief deprecated by new evidence
    ]
    
    def attempt_resolution(
        self,
        episode: ContradictionEpisode,
        hypergraph: Any,  # CausalHypergraph
        current_beliefs: Dict[str, Dict]
    ) -> ResolutionAttempt:
        """Attempt to resolve contradiction"""
        
        # Try each strategy in order until one works
        for strategy in self.RESOLUTION_STRATEGIES:
            attempt = self._try_strategy(
                strategy, episode, hypergraph, current_beliefs
            )
            
            if attempt.success:
                return attempt
        
        # None worked
        return ResolutionAttempt(
            attempt_id=str(uuid4()),
            strategy="exhausted",
            hypothesis="All resolution strategies failed",
            success=False,
            attempted_at=datetime.utcnow().isoformat(),
            outcome_summary="No resolution possible with current strategies"
        )
    
    def _try_strategy(
        self,
        strategy: str,
        episode: ContradictionEpisode,
        hypergraph: Any,
        current_beliefs: Dict[str, Dict]
    ) -> ResolutionAttempt:
        """Try specific resolution strategy"""
        
        if strategy == "context_clarification":
            return self._resolve_context_clarification(episode, current_beliefs)
        elif strategy == "confidence_adjustment":
            return self._resolve_confidence_adjustment(episode, current_beliefs)
        elif strategy == "belief_deprecation":
            return self._resolve_deprecation(episode, current_beliefs)
        
        # Default: not successful
        return ResolutionAttempt(
            attempt_id=str(uuid4()),
            strategy=strategy,
            hypothesis=f"Strategy {strategy} attempted",
            success=False,
            attempted_at=datetime.utcnow().isoformat()
        )
    
    def _resolve_context_clarification(
        self,
        episode: ContradictionEpisode,
        current_beliefs: Dict[str, Dict]
    ) -> ResolutionAttempt:
        """Resolve by finding different contexts"""
        
        # Check if beliefs have different conditions/scopes
        contexts = []
        
        for bid in episode.belief_ids:
            belief = current_beliefs.get(bid, {})
            prop = belief.get("proposition", "")
            
            # Extract context keywords
            if "under" in prop or "when" in prop or "if" in prop:
                contexts.append(bid)
        
        success = len(contexts) >= len(episode.belief_ids)
        
        return ResolutionAttempt(
            attempt_id=str(uuid4()),
            strategy="context_clarification",
            hypothesis="Beliefs have different contextual conditions",
            success=success,
            attempted_at=datetime.utcnow().isoformat(),
            outcome_summary="Resolved via context differentiation" if success else "Not resolved"
        )
    
    def _resolve_confidence_adjustment(
        self,
        episode: ContradictionEpisode,
        current_beliefs: Dict[str, Dict]
    ) -> ResolutionAttempt:
        """Resolve by reducing confidence of one belief"""
        
        # Lower confidence of belief with less evidence
        mutations = []
        
        for bid in episode.belief_ids:
            belief = current_beliefs.get(bid, {})
            current_conf = belief.get("confidence", 0.5)
            
            if current_conf > 0.6:
                new_conf = current_conf * 0.7
                mutations.append({
                    "belief_id": bid,
                    "old_confidence": current_conf,
                    "new_confidence": new_conf
                })
        
        success = len(mutations) > 0
        
        return ResolutionAttempt(
            attempt_id=str(uuid4()),
            strategy="confidence_adjustment",
            hypothesis="Reduced confidence to resolve tension",
            success=success,
            belief_mutations=mutations,
            attempted_at=datetime.utcnow().isoformat(),
            outcome_summary=f"Adjusted {len(mutations)} beliefs" if success else "No adjustments made"
        )
    
    def _resolve_deprecation(
        self,
        episode: ContradictionEpisode,
        current_beliefs: Dict[str, Dict]
    ) -> ResolutionAttempt:
        """Mark one belief as deprecated"""
        
        # Prefer to deprecate older belief
        deprecated = None
        
        for bid in episode.belief_ids:
            belief = current_beliefs.get(bid, {})
            if belief.get("source") == "legacy":
                deprecated = bid
                break
        
        if not deprecated:
            deprecated = episode.belief_ids[0]
        
        return ResolutionAttempt(
            attempt_id=str(uuid4()),
            strategy="belief_deprecation",
            hypothesis="One belief marked as deprecated",
            success=True,
            belief_mutations=[{"belief_id": deprecated, "action": "deprecated"}],
            attempted_at=datetime.utcnow().isoformat(),
            outcome_summary=f"Deprecated belief {deprecated}"
        )


class ContradictionMemory:
    """
    Contradiction Memory - tracks persistent contradictions over time.
    
    Key capabilities:
    - Episode lifecycle tracking (not snapshot)
    - Recurrence detection
    - Resolution attempt history
    - Stability scoring
    - Reflection triggers for persistent contradictions
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        hypergraph: Optional[Any] = None
    ):
        self.config = config or {}
        self._hypergraph = hypergraph
        
        # Episode storage
        self._episodes: Dict[str, ContradictionEpisode] = {}
        
        # Tracking state
        self._detector = ContradictionDetector()
        self._resolution_engine = ResolutionEngine()
        
        # Config
        self._stability_threshold = self.config.get("stability_threshold", 0.7)
        self._recurrence_threshold = self.config.get("recurrence_threshold", 2)
    
    def register_contradiction(
        self,
        belief_ids: List[str],
        contradiction_type: ContradictionType,
        current_beliefs: Dict[str, Dict],
        context: Optional[str] = None
    ) -> ContradictionEpisode:
        """Register new contradiction or update existing"""
        
        # Sort IDs for consistent key
        key = ":".join(sorted(belief_ids))
        
        now = datetime.utcnow().isoformat()
        
        # Check if episode already exists
        if key in self._episodes:
            # Update existing episode
            episode = self._episodes[key]
            episode.last_seen = now
            episode.activation_count += 1
            
            # Check if this is a recurrence (after resolution)
            if episode.resolution_status in [
                ResolutionStatus.RESOLVED,
                ResolutionStatus.SUPPRESSED
            ]:
                episode.recurrence_count += 1
                episode.resolution_status = ResolutionStatus.RECURRING
            
            if context and context not in episode.contexts:
                episode.contexts.append(context)
            
            # Recalculate stability
            episode.stability_score = self._calculate_stability(episode)
            
            return episode
        
        # Create new episode
        episode = ContradictionEpisode(
            episode_id=str(uuid4()),
            belief_ids=belief_ids,
            contradiction_type=contradiction_type,
            first_seen=now,
            last_seen=now,
            recurrence_count=1,
            resolution_status=ResolutionStatus.UNRESOLVED,
            stability_score=0.5,
            activation_count=1,
            severity=self._determine_severity(belief_ids, current_beliefs)
        )
        
        if context:
            episode.contexts = [context]
        
        self._episodes[key] = episode
        
        return episode
    
    def _calculate_stability(self, episode: ContradictionEpisode) -> float:
        """Calculate stability score (how persistent)"""
        
        # Factors:
        # - Recurrence count (more = more stable)
        # - Activation count (more = more stable)
        # - Time since first seen (longer = more stable)
        
        try:
            first = datetime.fromisoformat(episode.first_seen)
            last = datetime.fromisoformat(episode.last_seen)
            days_active = (last - first).total_seconds() / 86400
        except:
            days_active = 1.0
        
        # Normalize to 0-1
        recurrence_factor = min(episode.recurrence_count / 5.0, 1.0)
        activation_factor = min(episode.activation_count / 10.0, 1.0)
        time_factor = min(days_active / 30.0, 1.0)
        
        return (recurrence_factor * 0.4 + activation_factor * 0.4 + time_factor * 0.2)
    
    def _determine_severity(
        self,
        belief_ids: List[str],
        current_beliefs: Dict[str, Dict]
    ) -> str:
        """Determine severity of contradiction"""
        
        avg_confidence = 0.0
        
        for bid in belief_ids:
            belief = current_beliefs.get(bid, {})
            avg_confidence += belief.get("confidence", 0.5)
        
        avg_confidence /= max(len(belief_ids), 1)
        
        if avg_confidence > 0.7:
            return "critical"
        elif avg_confidence > 0.5:
            return "notable"
        else:
            return "minor"
    
    def attempt_resolution(
        self,
        episode_id: str,
        current_beliefs: Dict[str, Dict]
    ) -> Optional[ResolutionAttempt]:
        """Attempt to resolve a contradiction"""
        
        for key, episode in self._episodes.items():
            if episode.episode_id == episode_id:
                attempt = self._resolution_engine.attempt_resolution(
                    episode, self._hypergraph, current_beliefs
                )
                
                episode.resolution_attempts.append(attempt)
                
                if attempt.success:
                    episode.resolution_status = ResolutionStatus.RESOLVED
                else:
                    episode.resolution_status = ResolutionStatus.RESOLVING
                
                return attempt
        
        return None
    
    def get_unresolved(self) -> List[ContradictionEpisode]:
        """Get all unresolved contradictions"""
        
        return [
            ep for ep in self._episodes.values()
            if ep.resolution_status in [
                ResolutionStatus.UNRESOLVED,
                ResolutionStatus.RECURRING
            ]
        ]
    
    def get_persistent(self) -> List[ContradictionEpisode]:
        """Get persistent contradictions (high stability)"""
        
        return [
            ep for ep in self._episodes.values()
            if ep.stability_score >= self._stability_threshold
            and ep.resolution_status != ResolutionStatus.RESOLVED
        ]
    
    def get_recurring(self) -> List[ContradictionEpisode]:
        """Get recurring contradictions"""
        
        return [
            ep for ep in self._episodes.values()
            if ep.resolution_status == ResolutionStatus.RECURRING
            or ep.recurrence_count >= self._recurrence_threshold
        ]
    
    def get_critical(self) -> List[ContradictionEpisode]:
        """Get critical unresolved contradictions"""
        
        return [
            ep for ep in self._episodes.values()
            if ep.severity == "critical"
            and ep.resolution_status != ResolutionStatus.RESOLVED
        ]
    
    def get_reflection_triggers(self) -> List[Dict[str, Any]]:
        """Get triggers for reflection based on contradictions"""
        
        triggers = []
        
        # Persistent contradictions
        for ep in self.get_persistent():
            triggers.append({
                "type": "persistent_contradiction",
                "episode_id": ep.episode_id,
                "belief_ids": ep.belief_ids,
                "stability": ep.stability_score,
                "urgency": "high"
            })
        
        # Recurring contradictions
        for ep in self.get_recurring():
            if ep.resolution_status != ResolutionStatus.RESOLVED:
                triggers.append({
                    "type": "recurring_contradiction",
                    "episode_id": ep.episode_id,
                    "recurrence_count": ep.recurrence_count,
                    "urgency": "medium"
                })
        
        # Critical unresolved
        for ep in self.get_critical():
            triggers.append({
                "type": "critical_contradiction",
                "episode_id": ep.episode_id,
                "belief_ids": ep.belief_ids,
                "urgency": "critical"
            })
        
        return triggers
    
    def get_episode(self, episode_id: str) -> Optional[ContradictionEpisode]:
        """Get specific episode"""
        
        for ep in self._episodes.values():
            if ep.episode_id == episode_id:
                return ep
        
        return None
    
    def get_all_episodes(self) -> List[ContradictionEpisode]:
        """Get all episodes"""
        
        return list(self._episodes.values())
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        
        return {
            "total_episodes": len(self._episodes),
            "unresolved": len(self.get_unresolved()),
            "persistent": len(self.get_persistent()),
            "recurring": len(self.get_recurring()),
            "critical": len(self.get_critical()),
            "resolution_rate": self._calculate_resolution_rate()
        }
    
    def _calculate_resolution_rate(self) -> float:
        """Calculate resolution rate"""
        
        if not self._episodes:
            return 0.0
        
        resolved = sum(
            1 for ep in self._episodes.values()
            if ep.resolution_status == ResolutionStatus.RESOLVED
        )
        
        return resolved / len(self._episodes)


# Global instance
_contradiction_memory: Optional[ContradictionMemory] = None


def get_contradiction_memory(
    config: Optional[Dict] = None,
    hypergraph: Optional[Any] = None
) -> ContradictionMemory:
    """Get global contradiction memory"""
    global _contradiction_memory
    if _contradiction_memory is None:
        _contradiction_memory = ContradictionMemory(config, hypergraph)
    return _contradiction_memory