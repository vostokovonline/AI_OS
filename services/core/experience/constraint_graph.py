"""
Constraint Graph - Belief decomposition into predicates, assumptions, invariants

Provides:
- Belief decomposition into constraints
- Constraint incompatibility detection (not keyword polarity)
- Confidence regions
- Invariant tracking

Architecture:
    Belief -> {predicates, assumptions, invariants, scope, confidence_region}
    Constraint -> {type, value, domain, comparison}
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import re


class ConstraintType(Enum):
    """Types of constraints extracted from beliefs"""
    COMPARISON = "comparison"  # >, <, >=, <=, ==, !=
    RANGE = "range"  # range [min, max]
    BOOLEAN = "boolean"  # true/false condition
    TEMPORAL = "temporal"  # time-based constraint
    POLICY = "policy"  # policy requirement
    CONTEXT = "context"  # contextual condition
    PROBABILISTIC = "probabilistic"  # confidence region


class ConstraintOperator(Enum):
    """Comparison operators"""
    GT = ">"  # Greater than
    LT = "<"  # Less than
    GTE = ">="  # Greater or equal
    LTE = "<="  # Less or equal
    EQ = "=="  # Equal
    NEQ = "!="  # Not equal
    IN = "in"  # In set
    NOT_IN = "not_in"  # Not in set


@dataclass
class Constraint:
    """Single constraint extracted from belief"""
    constraint_id: str
    constraint_type: ConstraintType
    predicate: str
    domain: str
    extracted_at: str
    operator: Optional[ConstraintOperator] = None
    value: Optional[Any] = None
    value2: Optional[Any] = None
    unit: Optional[str] = None
    constraint_confidence: float = 1.0
    source_belief_id: Optional[str] = None


@dataclass
class ConfidenceRegion:
    """Confidence region for belief"""
    center: float  # Most likely value
    lower_bound: float  # Lower bound (5th percentile)
    upper_bound: float  # Upper bound (95th percentile)
    distribution: str  # "normal", "uniform", "bimodal"
    entropy: float  # Uncertainty measure


@dataclass
class BeliefConstraints:
    """Full constraint decomposition of a belief"""
    belief_id: str
    proposition: str
    
    # Extracted constraints
    constraints: List[Constraint] = field(default_factory=list)
    
    # Assumptions (things taken as given)
    assumptions: List[str] = field(default_factory=list)
    
    # Invariants (must hold for belief to be true)
    invariants: List[str] = field(default_factory=list)
    
    # Scope conditions
    scope: List[str] = field(default_factory=list)  # "under latency < 100ms"
    
    # Confidence region
    confidence_region: Optional[ConfidenceRegion] = None
    
    # Meta-beliefs about this belief
    meta_constraints: List[str] = field(default_factory=list)


class ConstraintExtractor:
    """Extract constraints from natural language propositions"""
    
    # Regex patterns for constraint extraction
    PATTERNS = {
        "comparison": r"(?P<pred>\w+)\s+(?P<op>>|<|>=|<=|==|!=|is|are)\s+(?P<val>\d+(?:\.\d+)?|[\w]+)",
        "range": r"(?P<pred>\w+)\s+(?:between|from)\s+(?P<v1>\d+(?:\.\d+)?)\s+(?:to|and)\s+(?P<v2>\d+(?:\.\d+)?)",
        "temporal": r"(when|during|after|before|in)\s+(?P<temp>\w+)",
        "context": r"(under|if|given|with)\s+(?P<ctx>[^,]+)",
        "policy": r"(must|should|require|forbid|allow)\s+(?P<pol>[^,]+)",
        "probability": r"(likely|unlikely|probable|confident)\s+(?P<prob>\d+(?:\.\d+)?)",
    }
    
    def extract(self, belief_id: str, proposition: str) -> BeliefConstraints:
        """Extract all constraints from proposition"""
        
        constraints = BeliefConstraints(
            belief_id=belief_id,
            proposition=proposition
        )
        
        # Extract comparison constraints
        for match in re.finditer(self.PATTERNS["comparison"], proposition, re.IGNORECASE):
            pred = match.group("pred")
            op_str = match.group("op")
            val = match.group("val")
            
            operator = self._map_operator(op_str)
            
            try:
                value = float(val) if val.replace(".", "").isdigit() else val
            except:
                value = val
            
            constraint = Constraint(
                constraint_id=str(uuid4()),
                constraint_type=ConstraintType.COMPARISON,
                predicate=pred,
                operator=operator,
                value=value,
                domain=self._infer_domain(pred),
                source_belief_id=belief_id,
                extracted_at=datetime.utcnow().isoformat()
            )
            constraints.constraints.append(constraint)
        
        # Extract range constraints
        for match in re.finditer(self.PATTERNS["range"], proposition, re.IGNORECASE):
            pred = match.group("pred")
            v1 = float(match.group("v1"))
            v2 = float(match.group("v2"))
            
            constraint = Constraint(
                constraint_id=str(uuid4()),
                constraint_type=ConstraintType.RANGE,
                predicate=pred,
                operator=None,
                value=v1,
                value2=v2,
                domain=self._infer_domain(pred),
                source_belief_id=belief_id,
                extracted_at=datetime.utcnow().isoformat()
            )
            constraints.constraints.append(constraint)
        
        # Extract contextual scope
        for match in re.finditer(self.PATTERNS["context"], proposition, re.IGNORECASE):
            scope = match.group("ctx").strip()
            constraints.scope.append(scope)
        
        # Extract temporal constraints
        for match in re.finditer(self.PATTERNS["temporal"], proposition, re.IGNORECASE):
            temp = match.group("temp")
            constraint = Constraint(
                constraint_id=str(uuid4()),
                constraint_type=ConstraintType.TEMPORAL,
                predicate="time",
                value=temp,
                domain="temporal",
                source_belief_id=belief_id,
                extracted_at=datetime.utcnow().isoformat()
            )
            constraints.constraints.append(constraint)
        
        # Extract policy constraints
        for match in re.finditer(self.PATTERNS["policy"], proposition, re.IGNORECASE):
            pol = match.group("pol")
            constraint = Constraint(
                constraint_id=str(uuid4()),
                constraint_type=ConstraintType.POLICY,
                predicate="policy",
                value=pol,
                domain="policy",
                source_belief_id=belief_id,
                extracted_at=datetime.utcnow().isoformat()
            )
            constraints.constraints.append(constraint)
        
        # Extract assumptions (simple heuristic: "assuming", "given that", "provided")
        assumptions = re.findall(r"(?:assuming|given that|provided|given)\s+([^,\.]+)", proposition, re.IGNORECASE)
        constraints.assumptions = [a.strip() for a in assumptions]
        
        return constraints
    
    def _map_operator(self, op_str: str) -> ConstraintOperator:
        """Map string operator to enum"""
        mapping = {
            ">": ConstraintOperator.GT,
            "<": ConstraintOperator.LT,
            ">=": ConstraintOperator.GTE,
            "<=": ConstraintOperator.LTE,
            "==": ConstraintOperator.EQ,
            "!=": ConstraintOperator.NEQ,
            "is": ConstraintOperator.EQ,
            "are": ConstraintOperator.EQ,
        }
        return mapping.get(op_str.lower(), ConstraintOperator.EQ)
    
    def _infer_domain(self, predicate: str) -> str:
        """Infer domain from predicate"""
        domain_keywords = {
            "latency": "latency_ms",
            "speed": "latency_ms",
            "reliable": "reliability",
            "reliability": "reliability",
            "trust": "confidence",
            "confidence": "confidence",
            "risk": "risk",
            "cost": "cost",
            "accuracy": "accuracy",
            "fast": "latency_ms",
            "slow": "latency_ms",
        }
        return domain_keywords.get(predicate.lower(), "unknown")


class ConstraintIncompatibilityDetector:
    """
    Detect constraint incompatibilities (not keyword polarity).
    
    Examples:
    - latency < 100ms vs latency > 500ms
    - policy requires exploration vs policy forbids uncertainty
    - trust > 0.8 vs trust < 0.3
    """
    
    def detect_incompatibilities(
        self,
        constraints1: BeliefConstraints,
        constraints2: BeliefConstraints
    ) -> List[Dict[str, Any]]:
        """Find incompatible constraints between two beliefs"""
        
        incompatibilities = []
        
        for c1 in constraints1.constraints:
            for c2 in constraints2.constraints:
                conflict = self._check_constraint_conflict(c1, c2)
                if conflict:
                    incompatibilities.append(conflict)
        
        return incompatibilities
    
    def _check_constraint_conflict(
        self,
        c1: Constraint,
        c2: Constraint
    ) -> Optional[Dict[str, Any]]:
        """Check if two constraints conflict"""
        
        # Same predicate, different domain - potential conflict
        if c1.predicate != c2.predicate:
            return None
        
        # Both are comparison constraints
        if c1.constraint_type == ConstraintType.COMPARISON and c2.constraint_type == ConstraintType.COMPARISON:
            return self._check_comparison_conflict(c1, c2)
        
        # Both are range constraints
        if c1.constraint_type == ConstraintType.RANGE and c2.constraint_type == ConstraintType.RANGE:
            return self._check_range_conflict(c1, c2)
        
        # Policy conflicts
        if c1.constraint_type == ConstraintType.POLICY and c2.constraint_type == ConstraintType.POLICY:
            return self._check_policy_conflict(c1, c2)
        
        return None
    
    def _check_comparison_conflict(self, c1: Constraint, c2: Constraint) -> Optional[Dict[str, Any]]:
        """Check comparison constraint conflicts"""
        
        if not c1.operator or not c2.operator:
            return None
        
        # Direct value conflicts
        if c1.operator == ConstraintOperator.GT and c2.operator == ConstraintOperator.LT:
            if c1.value and c2.value and c1.value > c2.value:
                return {
                    "type": "comparison_conflict",
                    "predicate": c1.predicate,
                    "constraint1": f"{c1.predicate} {c1.operator.value} {c1.value}",
                    "constraint2": f"{c2.predicate} {c2.operator.value} {c2.value}",
                    "severity": "critical"
                }
        
        if c1.operator == ConstraintOperator.GTE and c2.operator == ConstraintOperator.LTE:
            if c1.value and c2.value and c1.value >= c2.value:
                return {
                    "type": "comparison_conflict",
                    "predicate": c1.predicate,
                    "constraint1": f"{c1.predicate} {c1.operator.value} {c1.value}",
                    "constraint2": f"{c2.predicate} {c2.operator.value} {c2.value}",
                    "severity": "critical"
                }
        
        # Opposite directions
        if c1.operator in [ConstraintOperator.GT, ConstraintOperator.GTE] and c2.operator in [ConstraintOperator.LT, ConstraintOperator.LTE]:
            if c1.value and c2.value and c1.value >= c2.value:
                return {
                    "type": "range_overlap",
                    "predicate": c1.predicate,
                    "constraint1": f"{c1.predicate} {c1.operator.value} {c1.value}",
                    "constraint2": f"{c2.predicate} {c2.operator.value} {c2.value}",
                    "severity": "notable"
                }
        
        return None
    
    def _check_range_conflict(self, c1: Constraint, c2: Constraint) -> Optional[Dict[str, Any]]:
        """Check range constraint conflicts"""
        
        if not c1.value or not c2.value or not c1.value2 or not c2.value2:
            return None
        
        # Non-overlapping ranges
        if c1.value2 < c2.value or c2.value2 < c1.value:
            return {
                "type": "range_disjoint",
                "predicate": c1.predicate,
                "range1": f"[{c1.value}, {c1.value2}]",
                "range2": f"[{c2.value}, {c2.value2}]",
                "severity": "critical"
            }
        
        # Partial overlap
        return None
    
    def _check_policy_conflict(self, c1: Constraint, c2: Constraint) -> Optional[Dict[str, Any]]:
        """Check policy constraint conflicts"""
        
        # Simple heuristic: "require X" vs "forbid X"
        v1 = str(c1.value).lower()
        v2 = str(c2.value).lower()
        
        if "require" in v1 and "forbid" in v2:
            return {
                "type": "policy_conflict",
                "policy1": v1,
                "policy2": v2,
                "severity": "critical"
            }
        
        if "allow" in v1 and "forbid" in v2:
            return {
                "type": "policy_conflict",
                "policy1": v1,
                "policy2": v2,
                "severity": "critical"
            }
        
        return None


class ConstraintGraph:
    """
    Constraint Graph - decompose beliefs into constraints for proper contradiction detection.
    
    Provides:
    - Belief -> constraints decomposition
    - Constraint-based incompatibility detection
    - Confidence regions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self._extractor = ConstraintExtractor()
        self._detector = ConstraintIncompatibilityDetector()
        
        # Belief constraints storage
        self._belief_constraints: Dict[str, BeliefConstraints] = {}
    
    def add_belief(self, belief_id: str, proposition: str, confidence: float) -> BeliefConstraints:
        """Add belief and decompose into constraints"""
        
        constraints = self._extractor.extract(belief_id, proposition)
        
        # Add confidence region
        conf_region = self._compute_confidence_region(confidence)
        constraints.confidence_region = conf_region
        
        self._belief_constraints[belief_id] = constraints
        
        return constraints
    
    def _compute_confidence_region(self, confidence: float) -> ConfidenceRegion:
        """Compute confidence region with proper Shannon entropy (0-1 range)"""
        
        # Spread increases as confidence moves away from edges
        spread = 0.15 + abs(confidence - 0.5) * 0.2
        
        # Proper Shannon entropy: H = -p*log2(p) - (1-p)*log2(1-p)
        # Normalized: divide by max entropy (1.0 at p=0.5) to get 0-1 range
        if confidence <= 0 or confidence >= 1:
            entropy = 0.0
        else:
            # Shannon entropy formula (base 2 for bits)
            import math
            p = confidence
            h = -p * math.log2(p) - (1-p) * math.log2(1-p)
            # Normalize: max entropy at p=0.5 is 1.0
            entropy = h / 1.0  # Already in 0-1 range
        
        return ConfidenceRegion(
            center=confidence,
            lower_bound=max(0, confidence - spread),
            upper_bound=min(1, confidence + spread),
            distribution="normal",
            entropy=entropy
        )
    
    def update_belief(self, belief_id: str, new_confidence: float) -> Optional[BeliefConstraints]:
        """Update belief confidence region"""
        
        if belief_id not in self._belief_constraints:
            return None
        
        constraints = self._belief_constraints[belief_id]
        constraints.confidence_region = self._compute_confidence_region(new_confidence)
        
        return constraints
    
    def get_constraints(self, belief_id: str) -> Optional[BeliefConstraints]:
        """Get constraint decomposition for belief"""
        return self._belief_constraints.get(belief_id)
    
    def detect_contradictions(
        self,
        belief_id1: str,
        belief_id2: str
    ) -> List[Dict[str, Any]]:
        """Detect contradictions via constraint incompatibility"""
        
        c1 = self._belief_constraints.get(belief_id1)
        c2 = self._belief_constraints.get(belief_id2)
        
        if not c1 or not c2:
            return []
        
        return self._detector.detect_incompatibilities(c1, c2)
    
    def get_all_constraints(self) -> Dict[str, BeliefConstraints]:
        """Get all belief constraints"""
        return self._belief_constraints.copy()
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of constraints"""
        
        total_constraints = sum(
            len(c.constraints) for c in self._belief_constraints.values()
        )
        
        by_type = {}
        for c in self._belief_constraints.values():
            for const in c.constraints:
                t = const.constraint_type.value
                by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total_beliefs": len(self._belief_constraints),
            "total_constraints": total_constraints,
            "by_type": by_type
        }


# Global instance
_constraint_graph: Optional[ConstraintGraph] = None


def get_constraint_graph(config: Optional[Dict] = None) -> ConstraintGraph:
    """Get global constraint graph"""
    global _constraint_graph
    if _constraint_graph is None:
        _constraint_graph = ConstraintGraph(config)
    return _constraint_graph