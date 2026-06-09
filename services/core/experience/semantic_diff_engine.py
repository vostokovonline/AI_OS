"""
Semantic Diff Engine - Track semantic evolution of internal world model

Provides:
- Semantic mutation classification (not just structural diff)
- Causal attribution (not just event reference)
- Contradiction detection
- Confidence drift analysis
- Policy impact estimation
- Reflection trigger detection

Key distinction:
- event sourcing: "what happened"
- semantic evolution: "what changed in our understanding"

Architecture:
    SemanticDiffEngine
    ├── belief_comparator
    ├── contradiction_detector
    ├── causal_attribution_engine
    ├── confidence_drift_analyzer
    ├── policy_impact_estimator
    └── reflection_trigger_detector
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import re


class SemanticChangeType(Enum):
    """Types of semantic mutations (not just structural changes)"""
    ADDED = "added"  # New belief
    REMOVED = "removed"  # Belief retracted
    
    # Semantic transformations
    WEAKENED = "weakened"  # Confidence decreased (but belief persists)
    STRENGTHENED = "strengthened"  # Confidence increased
    SPECIALIZED = "specialized"  # Scope narrowed (e.g., "file ops" → "file ops under latency")
    GENERALIZED = "generalized"  # Scope broadened
    CONTRADICTED = "contradicted"  # Belief now conflicts with other belief
    MERGED = "merged"  # Multiple beliefs combined
    DEPRECATED = "deprecated"  # Belief obsolete
    REINTERPRETED = "reinterpreted"  # Same facts, different meaning
    CONTEXTUALIZED = "contextualized"  # Added context/condition


class ChangeSeverity(Enum):
    """Severity of semantic change"""
    MINOR = "minor"  # Normal variation
    NOTABLE = "notable"  # Significant shift
    CRITICAL = "critical"  # Major worldview change
    CATASTROPHIC = "catastrophic"  # Contradiction, requires resolution


@dataclass
class CausalAttribution:
    """Rich causal attribution (not just event reference)"""
    intent_id: Optional[str]  # What intent caused this
    policy_version: Optional[str]  # What policy was active
    evidence_ids: List[str] = field(default_factory=list)  # What evidence
    runtime_state: Optional[Dict[str, Any]] = None  # What state
    counterfactual_mismatch: Optional[str] = None  # What expectation failed
    upstream_belief_ids: List[str] = field(default_factory=list)  # Which beliefs led to this
    
    def to_dict(self) -> dict:
        return {
            "intent_id": self.intent_id,
            "policy_version": self.policy_version,
            "evidence_ids": self.evidence_ids,
            "runtime_state": self.runtime_state,
            "counterfactual_mismatch": self.counterfactual_mismatch,
            "upstream_belief_ids": self.upstream_belief_ids
        }


@dataclass
class SemanticMutation:
    """Single semantic change with full attribution"""
    mutation_id: str
    change_type: SemanticChangeType
    severity: ChangeSeverity
    belief_id: str
    old_belief: Optional[Dict[str, Any]]
    new_belief: Optional[Dict[str, Any]]
    old_proposition: Optional[str]
    new_proposition: Optional[str]
    semantic_delta: str
    causal_attribution: CausalAttribution
    timestamp: str
    affected_policies: List[str] = field(default_factory=list)
    policy_confidence_change: Dict[str, float] = field(default_factory=dict)


@dataclass
class SemanticDiff:
    """Complete semantic diff between two belief states"""
    diff_id: str
    total_changes: int
    severity_distribution: Dict[str, int]
    change_types: Dict[str, int]
    mutations: List[SemanticMutation]
    belief_topology_change: str
    reflection_needed: bool
    causal_chain_length: int
    timestamp: str
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    trust_shifts: Dict[str, float] = field(default_factory=dict)
    reflection_triggers: List[str] = field(default_factory=list)
    root_causes: List[CausalAttribution] = field(default_factory=list)


class BeliefComparator:
    """Compare beliefs semantically, not just structurally"""
    
    def __init__(self):
        self._keywords_trust = {"trust", "reliable", "dependable", "confident", "proven", "stable"}
        self._keywords_untrust = {"unreliable", "broken", "fails", "unpredictable", "risky", "slow"}
        self._keywords_scope = {"under", "when", "during", "if", "with", "given"}
    
    def compare(self, old: Dict[str, Any], new: Dict[str, Any]) -> Tuple[SemanticChangeType, str]:
        """Compare two beliefs and determine semantic change type"""
        
        old_prop = old.get("proposition", "").lower()
        new_prop = new.get("proposition", "").lower()
        
        old_conf = old.get("confidence", 0.5)
        new_conf = new.get("confidence", 0.5)
        
        # Check for removed
        if new is None:
            return SemanticChangeType.REMOVED, "Belief retracted"
        
        # Check for added
        if old is None:
            return SemanticChangeType.ADDED, "New belief emerged"
        
        # Check for specialization (narrowed scope)
        old_scope = self._extract_scope(old_prop)
        new_scope = self._extract_scope(new_prop)
        
        if new_scope and old_scope and len(new_scope) > len(old_scope):
            added_conditions = set(new_scope) - set(old_scope)
            if added_conditions:
                return SemanticChangeType.SPECIALIZED, f"Added conditions: {', '.join(added_conditions)}"
        
        # Check for generalization (broadened scope)
        if old_scope and new_scope and len(new_scope) < len(old_scope):
            removed_conditions = set(old_scope) - set(new_scope)
            if removed_conditions:
                return SemanticChangeType.GENERALIZED, f"Relaxed constraints: {', '.join(removed_conditions)}"
        
        # Check for trust shift
        old_is_trusting = any(kw in old_prop for kw in self._keywords_trust)
        new_is_trusting = any(kw in new_prop for kw in self._keywords_trust)
        old_is_distrusting = any(kw in old_prop for kw in self._keywords_untrust)
        new_is_distrusting = any(kw in new_prop for kw in self._keywords_untrust)
        
        if old_is_trusting and new_is_distrusting:
            return SemanticChangeType.CONTRADICTED, "Trust completely reversed"
        if old_is_distrusting and new_is_trusting:
            return SemanticChangeType.REINTERPRETED, "Distrust reinterpreted as trust"
        
        # Check for confidence drift
        conf_delta = abs(new_conf - old_conf)
        if conf_delta > 0.3:
            if new_conf > old_conf:
                return SemanticChangeType.STRENGTHENED, f"Confidence increased: {old_conf:.2f} → {new_conf:.2f}"
            else:
                return SemanticChangeType.WEAKENED, f"Confidence decreased: {old_conf:.2f} → {new_conf:.2f}"
        elif conf_delta > 0.15:
            return SemanticChangeType.CONTEXTUALIZED, f"Confidence adjusted: {old_conf:.2f} → {new_conf:.2f}"
        
        # Check for reinterpretation (same facts, different meaning)
        if self._has_semantic_shift(old_prop, new_prop):
            return SemanticChangeType.REINTERPRETED, "Semantic interpretation changed"
        
        return SemanticChangeType.ADDED, "Belief present"
    
    def _extract_scope(self, proposition: str) -> List[str]:
        """Extract conditional scopes from proposition"""
        scopes = []
        for kw in self._keywords_scope:
            if kw in proposition:
                idx = proposition.find(kw)
                # Extract context around keyword
                start = max(0, idx - 20)
                end = min(len(proposition), idx + 30)
                scopes.append(proposition[start:end])
        return scopes
    
    def _has_semantic_shift(self, old_prop: str, new_prop: str) -> bool:
        """Check if proposition has semantic shift without structural change"""
        # Simple heuristic: if same words but different order/context
        old_words = set(old_prop.split())
        new_words = set(new_prop.split())
        
        # If mostly same words but different arrangement
        if len(old_words & new_words) / max(len(old_words), 1) > 0.7:
            # Check if sentiment changed
            old_sentiment = self._compute_sentiment(old_prop)
            new_sentiment = self._compute_sentiment(new_prop)
            if old_sentiment * new_sentiment < 0:  # Sign change
                return True
        return False
    
    def _compute_sentiment(self, text: str) -> float:
        """Simple sentiment score"""
        score = 0.0
        for kw in self._keywords_trust:
            if kw in text: score += 0.1
        for kw in self._keywords_untrust:
            if kw in text: score -= 0.1
        return score


class ContradictionDetector:
    """Detect contradictions between beliefs"""
    
    def __init__(self):
        self._contradiction_patterns = [
            (["reliable", "unreliable"], ["dependable", "unpredictable"]),
            (["fast", "slow"], ["efficient", "inefficient"]),
            (["trust", "distrust"], ["believe", "doubt"]),
            (["safe", "dangerous"], ["secure", "risky"]),
        ]
    
    def detect_contradictions(
        self,
        old_beliefs: Dict[str, Dict],
        new_beliefs: Dict[str, Dict],
        mutations: List[SemanticMutation]
    ) -> List[Dict[str, Any]]:
        """Find contradictions between beliefs"""
        contradictions = []
        
        # Check for direct contradictions in mutations
        for i, m1 in enumerate(mutations):
            if m1.change_type != SemanticChangeType.CONTRADICTED:
                continue
            
            for m2 in mutations[i+1:]:
                if self._are_contradicting(m1, m2):
                    contradictions.append({
                        "mutation_1": m1.mutation_id,
                        "mutation_2": m2.mutation_id,
                        "type": "direct_contradiction",
                        "severity": "critical",
                        "description": f"{m1.semantic_delta} vs {m2.semantic_delta}"
                    })
        
        # Check for implicit contradictions (belief A contradicts belief B)
        belief_list = list(new_beliefs.values())
        for i, b1 in enumerate(belief_list):
            for b2 in belief_list[i+1:]:
                if self._beliefs_contradict(b1, b2):
                    contradictions.append({
                        "belief_1": b1.get("belief_id"),
                        "belief_2": b2.get("belief_id"),
                        "type": "belief_conflict",
                        "severity": "notable",
                        "propositions": [b1.get("proposition"), b2.get("proposition")]
                    })
        
        return contradictions
    
    def _are_contradicting(self, m1: SemanticMutation, m2: SemanticMutation) -> bool:
        """Check if two mutations contradict each other"""
        if not m1.new_belief or not m2.new_belief:
            return False
        
        p1 = m1.new_belief.get("proposition", "").lower()
        p2 = m2.new_belief.get("proposition", "").lower()
        
        return self._beliefs_contradict({"proposition": p1}, {"proposition": p2})
    
    def _beliefs_contradict(self, b1: Dict, b2: Dict) -> bool:
        """Check if two beliefs contradict"""
        p1 = b1.get("proposition", "").lower()
        p2 = b2.get("proposition", "").lower()
        
        for pos_group, neg_group in self._contradiction_patterns:
            pos_in_1 = any(kw in p1 for kw in pos_group)
            neg_in_1 = any(kw in p1 for kw in neg_group)
            pos_in_2 = any(kw in p2 for kw in pos_group)
            neg_in_2 = any(kw in p2 for kw in neg_group)
            
            # If one has positive, other has negative of same category
            if (pos_in_1 and neg_in_2) or (neg_in_1 and pos_in_2):
                return True
        
        return False


class CausalAttributionEngine:
    """Build causal hypergraph, not event chain"""
    
    def attribute(
        self,
        mutation: SemanticMutation,
        prior_beliefs: Dict[str, Dict],
        intent_id: Optional[str] = None,
        policy_version: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        runtime_state: Optional[Dict[str, Any]] = None,
        counterfactual_mismatch: Optional[str] = None
    ) -> CausalAttribution:
        """Build rich causal attribution"""
        
        # Find upstream beliefs that could have caused this
        upstream = []
        if mutation.old_belief:
            # Belief derived from prior beliefs
            for bid, belief in prior_beliefs.items():
                if self._may_have_caused(belief, mutation):
                    upstream.append(bid)
        
        return CausalAttribution(
            intent_id=intent_id,
            policy_version=policy_version,
            evidence_ids=evidence_ids or [],
            runtime_state=runtime_state,
            counterfactual_mismatch=counterfactual_mismatch,
            upstream_belief_ids=upstream[:5]  # Limit to top 5
        )
    
    def _may_have_caused(self, prior_belief: Dict, mutation: SemanticMutation) -> bool:
        """Heuristic: could this belief have caused the mutation?"""
        if not mutation.old_belief:
            return False
        
        # Simple semantic similarity check
        old_prop = prior_belief.get("proposition", "").lower()
        new_prop = mutation.new_belief.get("proposition", "").lower() if mutation.new_belief else ""
        
        # Share keywords
        old_words = set(old_prop.split())
        new_words = set(new_prop.split())
        
        return len(old_words & new_words) >= 2


class ConfidenceDriftAnalyzer:
    """Analyze confidence drift patterns"""
    
    def analyze(
        self,
        old_beliefs: Dict[str, Dict],
        new_beliefs: Dict[str, Dict]
    ) -> Dict[str, float]:
        """Analyze confidence drift per entity"""
        drifts = {}
        
        all_belief_ids = set(old_beliefs.keys()) | set(new_beliefs.keys())
        
        for bid in all_belief_ids:
            old_conf = old_beliefs.get(bid, {}).get("confidence", 0.5)
            new_conf = new_beliefs.get(bid, {}).get("confidence", 0.5)
            
            delta = new_conf - old_conf
            if abs(delta) > 0.1:  # Significant drift
                drifts[bid] = delta
        
        return drifts


class PolicyImpactEstimator:
    """Estimate impact of changes on policies"""
    
    def estimate(
        self,
        mutations: List[SemanticMutation],
        active_policies: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, float]]:
        """Estimate which policies are affected and how"""
        affected = []
        impacts = {}
        
        policy_keywords = {
            "skill_selection": ["skill", "select", "choose", "prefer"],
            "risk_tolerance": ["risk", "safe", "cautious", "aggressive"],
            "latency_preference": ["fast", "slow", "latency", "speed"],
            "reliability_priority": ["reliable", "dependable", "backup", "fallback"],
        }
        
        for mutation in mutations:
            if not mutation.new_belief:
                continue
            
            prop = mutation.new_belief.get("proposal", "").lower()
            
            for policy, keywords in policy_keywords.items():
                if any(kw in prop for kw in keywords):
                    if policy not in affected:
                        affected.append(policy)
                    
                    # Estimate confidence change impact
                    conf_change = abs(
                        mutation.new_belief.get("confidence", 0.5) -
                        mutation.old_belief.get("confidence", 0.5)
                    ) if mutation.old_belief else 0.3
                    
                    impacts[policy] = impacts.get(policy, 0) + conf_change
        
        return affected, impacts


class ReflectionTriggerDetector:
    """Detect when reflection is needed"""
    
    def __init__(self):
        self._critical_changes = {
            SemanticChangeType.CONTRADICTED,
            SemanticChangeType.DEPRECATED,
        }
        self._notable_changes = {
            SemanticChangeType.SPECIALIZED,
            SemanticChangeType.REINTERPRETED,
        }
    
    def should_trigger(
        self,
        diff: SemanticDiff,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Determine if reflection should be triggered"""
        triggers = []
        
        config = config or {}
        threshold = config.get("severity_threshold", ChangeSeverity.CRITICAL)
        
        # High severity changes
        if diff.severity_distribution.get("critical", 0) > 0:
            triggers.append("critical_belief_change")
        
        if diff.severity_distribution.get("catastrophic", 0) > 0:
            triggers.append("worldview_inconsistency")
        
        # Contradictions
        if len(diff.contradictions) > 0:
            triggers.append("belief_contradiction_detected")
        
        # Major trust shifts
        for skill_id, shift in diff.trust_shifts.items():
            if abs(shift) > 0.4:
                triggers.append(f"major_trust_shift_{skill_id}")
        
        # Policy impact
        for mutation in diff.mutations:
            if len(mutation.affected_policies) > 2:
                triggers.append("multi_policy_impact")
        
        # Causal chain depth
        if diff.causal_chain_length > 5:
            triggers.append("complex_causal_chain")
        
        should_reflect = len(triggers) > 0
        
        return should_reflect, triggers


class SemanticDiffEngine:
    """
    Semantic Diff Engine - tracks semantic evolution of internal world model.
    
    Key capabilities:
    - Semantic mutation classification (not just structural diff)
    - Causal hypergraph attribution (not event chain)
    - Contradiction detection
    - Policy impact estimation
    - Reflection trigger detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self._comparator = BeliefComparator()
        self._contradiction_detector = ContradictionDetector()
        self._causal_engine = CausalAttributionEngine()
        self._drift_analyzer = ConfidenceDriftAnalyzer()
        self._impact_estimator = PolicyImpactEstimator()
        self._trigger_detector = ReflectionTriggerDetector()
    
    def compute_diff(
        self,
        old_beliefs: Dict[str, Dict],
        new_beliefs: Dict[str, Dict],
        context: Optional[Dict[str, Any]] = None
    ) -> SemanticDiff:
        """Compute semantic diff between two belief states"""
        
        context = context or {}
        
        mutations = []
        severity_counts = {"minor": 0, "notable": 0, "critical": 0, "catastrophic": 0}
        type_counts = {t.value: 0 for t in SemanticChangeType}
        
        # Compare each belief
        all_belief_ids = set(old_beliefs.keys()) | set(new_beliefs.keys())
        
        for bid in all_belief_ids:
            old = old_beliefs.get(bid)
            new = new_beliefs.get(bid)
            
            change_type, delta_desc = self._comparator.compare(old, new) if old or new else (SemanticChangeType.ADDED, "present")
            
            # Determine severity
            severity = self._classify_severity(change_type, old, new)
            
            # Create mutation placeholder
            mutation = SemanticMutation(
                mutation_id=str(uuid4()),
                change_type=change_type,
                severity=severity,
                belief_id=bid,
                old_belief=old,
                new_belief=new,
                old_proposition=old.get("proposition") if old else None,
                new_proposition=new.get("proposition") if new else None,
                semantic_delta=delta_desc,
                causal_attribution=CausalAttribution(
                    intent_id=None,
                    policy_version=None,
                    evidence_ids=[],
                    runtime_state=None,
                    counterfactual_mismatch=None,
                    upstream_belief_ids=[]
                ),
                timestamp=datetime.utcnow().isoformat()
            )
            
            # Build causal attribution now that we have mutation
            attribution = self._causal_engine.attribute(
                mutation=mutation,
                prior_beliefs=old_beliefs,
                intent_id=context.get("intent_id"),
                policy_version=context.get("policy_version"),
                evidence_ids=context.get("evidence_ids"),
                runtime_state=context.get("runtime_state"),
                counterfactual_mismatch=context.get("counterfactual_mismatch")
            )
            mutation.causal_attribution = attribution
            
            # Estimate policy impact
            affected, impacts = self._impact_estimator.estimate(
                [mutation],
                context.get("active_policies", {})
            )
            mutation.affected_policies = affected
            mutation.policy_confidence_change = impacts
            
            mutations.append(mutation)
            severity_counts[severity.value] += 1
            type_counts[change_type.value] += 1
        
        # Detect contradictions
        contradictions = self._contradiction_detector.detect_contradictions(
            old_beliefs, new_beliefs, mutations
        )
        
        # Analyze confidence drift
        trust_shifts = self._drift_analyzer.analyze(old_beliefs, new_beliefs)
        
        # Determine topology change
        topology = self._classify_topology(type_counts)
        
        # Check reflection triggers
        diff = SemanticDiff(
            diff_id=str(uuid4()),
            total_changes=len(mutations),
            severity_distribution=severity_counts,
            change_types=type_counts,
            mutations=mutations,
            contradictions=contradictions,
            belief_topology_change=topology,
            trust_shifts=trust_shifts,
            reflection_needed=False,
            reflection_triggers=[],
            root_causes=[],
            causal_chain_length=len(mutations),
            timestamp=datetime.utcnow().isoformat()
        )
        
        reflection_needed, triggers = self._trigger_detector.should_trigger(diff, self.config)
        diff.reflection_needed = reflection_needed
        diff.reflection_triggers = triggers
        
        # Find root causes (mutations without upstream)
        diff.root_causes = [
            m.causal_attribution for m in mutations
            if not m.causal_attribution.upstream_belief_ids
        ]
        
        return diff
    
    def _classify_severity(
        self,
        change_type: SemanticChangeType,
        old: Optional[Dict],
        new: Optional[Dict]
    ) -> ChangeSeverity:
        """Classify severity of change"""
        if change_type in [SemanticChangeType.CONTRADICTED, SemanticChangeType.DEPRECATED]:
            return ChangeSeverity.CRITICAL
        
        if change_type in [SemanticChangeType.REMOVED, SemanticChangeType.REINTERPRETED]:
            return ChangeSeverity.NOTABLE
        
        if old and new:
            conf_delta = abs(new.get("confidence", 0.5) - old.get("confidence", 0.5))
            if conf_delta > 0.4:
                return ChangeSeverity.CRITICAL
            elif conf_delta > 0.25:
                return ChangeSeverity.NOTABLE
        
        return ChangeSeverity.MINOR
    
    def _classify_topology(self, type_counts: Dict[str, int]) -> str:
        """Classify overall belief topology change"""
        total = sum(type_counts.values())
        if total == 0:
            return "stable"
        
        specialized = type_counts.get(SemanticChangeType.SPECIALIZED.value, 0)
        generalized = type_counts.get(SemanticChangeType.GENERALIZED.value, 0)
        contradicted = type_counts.get(SemanticChangeType.CONTRADICTED.value, 0)
        
        if contradicted > 0:
            return "restructured"
        if specialized > generalized and specialized > total * 0.3:
            return "narrowed"
        if generalized > specialized and generalized > total * 0.3:
            return "broadened"
        
        return "stable"


# Global instance
_diff_engine: Optional[SemanticDiffEngine] = None


def get_semantic_diff_engine(config: Optional[Dict] = None) -> SemanticDiffEngine:
    """Get global semantic diff engine"""
    global _diff_engine
    if _diff_engine is None:
        _diff_engine = SemanticDiffEngine(config)
    return _diff_engine


def compute_semantic_diff(
    old_beliefs: Dict[str, Dict],
    new_beliefs: Dict[str, Dict],
    context: Optional[Dict] = None
) -> SemanticDiff:
    """Convenience function to compute semantic diff"""
    engine = get_semantic_diff_engine()
    return engine.compute_diff(old_beliefs, new_beliefs, context)