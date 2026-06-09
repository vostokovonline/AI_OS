"""
Policy Attribution Graph - Why specific features led to actions

Traces the causal chain from state features → scoring → decision.

Components:
- AttributionEdge: Causal link between feature and decision
- AttributionNode: Feature or decision in the graph
- AttributionGraph: Complete causal graph
- AttributionAnalyzer: Explain decisions
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttributionNode:
    """Node in the attribution graph"""
    id: str
    node_type: str  # feature, score, candidate, decision
    name: str
    value: float
    
    contributions_from: List[str] = field(default_factory=list)
    contributions_to: List[str] = field(default_factory=list)
    
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionEdge:
    """Edge in the attribution graph"""
    from_node: str
    to_node: str
    weight: float
    contribution: float
    
    reason: str = ""
    polarity: str = "positive"  # positive, negative, neutral


@dataclass
class AttributionPath:
    """Complete causal path from feature to action"""
    path: List[str]  # Node IDs
    features: List[str]
    total_contribution: float
    
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class DecisionAttribution:
    """Full attribution for a decision"""
    decision_id: str
    timestamp: datetime
    
    action: str
    confidence: float
    
    primary_features: List[Tuple[str, float]]  # (feature, contribution)
    causal_paths: List[AttributionPath]
    
    alternative_attributions: List[Dict] = field(default_factory=list)
    
    final_explanation: str = ""


class AttributionGraph:
    """
    Causal attribution graph for decisions.
    
    Builds a graph where:
    - Nodes are features, scores, candidates, decisions
    - Edges are causal relationships
    """
    
    def __init__(self, decision_id: str):
        self.decision_id = decision_id
        self.nodes: Dict[str, AttributionNode] = {}
        self.edges: List[AttributionEdge] = []
        self.created_at = datetime.utcnow()
    
    def add_feature(self, name: str, value: float, weight: float = 1.0) -> str:
        """Add a feature node"""
        node_id = f"feature_{name}"
        
        self.nodes[node_id] = AttributionNode(
            id=node_id,
            node_type="feature",
            name=name,
            value=value,
            weight=weight,
            metadata={"original_value": value}
        )
        
        return node_id
    
    def add_score(self, name: str, score: float) -> str:
        """Add a score node"""
        node_id = f"score_{name}"
        
        self.nodes[node_id] = AttributionNode(
            id=node_id,
            node_type="score",
            name=name,
            value=score
        )
        
        return node_id
    
    def add_candidate(self, name: str, value: float) -> str:
        """Add a candidate/action node"""
        node_id = f"candidate_{name}"
        
        self.nodes[node_id] = AttributionNode(
            id=node_id,
            node_type="candidate",
            name=name,
            value=value
        )
        
        return node_id
    
    def add_decision(self, name: str, value: float) -> str:
        """Add decision node"""
        node_id = "decision_final"
        
        self.nodes[node_id] = AttributionNode(
            id=node_id,
            node_type="decision",
            name=name,
            value=value
        )
        
        return node_id
    
    def link(self, from_id: str, to_id: str, weight: float, reason: str = "") -> None:
        """Link two nodes with contribution"""
        if from_id not in self.nodes:
            logger.warning("node_not_found", node_id=from_id)
            return
        if to_id not in self.nodes:
            logger.warning("node_not_found", node_id=to_id)
            return
        
        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]
        
        contribution = self._compute_contribution(from_node, to_node, weight)
        
        polarity = "positive" if contribution >= 0 else "negative"
        
        edge = AttributionEdge(
            from_node=from_id,
            to_node=to_id,
            weight=weight,
            contribution=contribution,
            reason=reason,
            polarity=polarity
        )
        
        self.edges.append(edge)
        
        from_node.contributions_to.append(to_id)
        to_node.contributions_from.append(from_id)
        
        logger.debug("edge_added", from_node=from_id, to_node=to_id, contribution=contribution)
    
    def _compute_contribution(
        self,
        from_node: AttributionNode,
        to_node: AttributionNode,
        weight: float
    ) -> float:
        """Compute how much from_node contributed to to_node"""
        base = from_node.value * weight
        
        if from_node.node_type == "feature":
            base *= from_node.weight
        
        if to_node.node_type == "score":
            base *= 0.5
        
        if to_node.node_type == "candidate":
            base *= 0.8
        
        return base
    
    def get_primary_features(self) -> List[Tuple[str, float]]:
        """Get features with highest contribution to decision"""
        feature_contributions: Dict[str, float] = {}
        
        for edge in self.edges:
            if edge.from_node.startswith("feature_"):
                if edge.to_node == "decision_final":
                    feature_contributions[edge.from_node] = feature_contributions.get(edge.from_node, 0) + edge.contribution
        
        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return [(self.nodes[fid].name, contrib) for fid, contrib in sorted_features[:5]]
    
    def trace_causal_paths(self) -> List[AttributionPath]:
        """Trace all causal paths from features to decision"""
        paths = []
        
        feature_nodes = [nid for nid, n in self.nodes.items() if n.node_type == "feature"]
        decision_node = "decision_final"
        
        for feature_id in feature_nodes:
            path = self._trace_path(feature_id, decision_node, [feature_id])
            
            if path:
                features = [self.nodes[nid].name for nid in path if nid.startswith("feature_")]
                total = sum(
                    e.contribution for e in self.edges
                    if e.from_node in path and e.to_node in path
                )
                
                paths.append(AttributionPath(
                    path=path,
                    features=features,
                    total_contribution=total,
                    explanation=self._explain_path(path),
                    confidence=min(1.0, abs(total) + 0.3)
                ))
        
        return sorted(paths, key=lambda p: abs(p.total_contribution), reverse=True)
    
    def _trace_path(
        self,
        current: str,
        target: str,
        visited: List[str]
    ) -> Optional[List[str]]:
        """Recursively trace path"""
        if current == target:
            return visited
        
        next_nodes = [e.to_node for e in self.edges if e.from_node == current]
        
        for next_node in next_nodes:
            if next_node not in visited:
                path = self._trace_path(next_node, target, visited + [next_node])
                if path:
                    return path
        
        return None
    
    def _explain_path(self, path: List[str]) -> str:
        """Generate explanation for a path"""
        if len(path) < 2:
            return ""
        
        nodes = [self.nodes[nid] for nid in path if nid in self.nodes]
        feature_nodes = [n for n in nodes if n.node_type == "feature"]
        decision_node = [n for n in nodes if n.node_type == "decision"]
        
        if not feature_nodes or not decision_node:
            return ""
        
        feature_names = [n.name for n in feature_nodes]
        decision_name = decision_node[0].name if decision_node else "unknown"
        
        return f"{', '.join(feature_names)} → ... → {decision_name}"
    
    def get_alternative_attributions(self) -> List[Dict]:
        """Get why alternative actions were not chosen"""
        alternatives = []
        
        candidate_nodes = [nid for nid, n in self.nodes.items() if n.node_type == "candidate"]
        candidate_nodes = [cid for cid in candidate_nodes if cid != "candidate_final"]
        
        decision_node = self.nodes.get("decision_final")
        if not decision_node:
            return alternatives
        
        decision_value = decision_node.value
        
        for candidate_id in candidate_nodes:
            candidate = self.nodes[candidate_id]
            
            edges_to_decision = [
                e for e in self.edges
                if e.from_node == candidate_id and e.to_node == "decision_final"
            ]
            
            contribution = sum(e.contribution for e in edges_to_decision)
            
            alternatives.append({
                "action": candidate.name,
                "value": candidate.value,
                "contribution": contribution,
                "lost_by": decision_value - candidate.value,
                "key_features": self._get_candidate_key_features(candidate_id)
            })
        
        return sorted(alternatives, key=lambda x: x["lost_by"], reverse=True)


class AttributionAnalyzer:
    """
    Analyze and explain decisions using attribution graphs.
    
    Usage:
        analyzer = AttributionAnalyzer()
        
        # After a decision
        attribution = analyzer.attribute_decision(decision_id, state, action, alternatives)
        
        # Get explanation
        explanation = analyzer.explain(attribution)
    """
    
    def __init__(self):
        self.attributions: Dict[str, DecisionAttribution] = {}
        logger.info("attribution_analyzer_initialized")
    
    def attribute_decision(
        self,
        decision_id: str,
        timestamp: datetime,
        state: Dict[str, float],
        action: str,
        confidence: float,
        scoring_details: Dict[str, float],
        alternatives: List[Dict]
    ) -> DecisionAttribution:
        """Create attribution for a decision"""
        graph = AttributionGraph(decision_id)
        
        for feature_name, feature_value in state.items():
            if feature_name not in ["timestamp", "state_type"]:
                weight = self._get_feature_weight(feature_name)
                graph.add_feature(feature_name, feature_value, weight)
        
        for score_name, score_value in scoring_details.items():
            graph.add_score(score_name, score_value)
            for feature_id, node in graph.nodes.items():
                if feature_id.startswith("feature_"):
                    graph.link(feature_id, f"score_{score_name}", weight=0.5)
        
        graph.add_decision(action, confidence)
        
        for feature_id in graph.nodes:
            if feature_id.startswith("feature_"):
                graph.link(feature_id, "decision_final", weight=0.8)
        
        primary_features = graph.get_primary_features()
        causal_paths = graph.trace_causal_paths()
        alternative_attributions = graph.get_alternative_attributions()
        
        attribution = DecisionAttribution(
            decision_id=decision_id,
            timestamp=timestamp,
            action=action,
            confidence=confidence,
            primary_features=primary_features,
            causal_paths=causal_paths,
            alternative_attributions=alternative_attributions
        )
        
        self.attributions[decision_id] = attribution
        
        logger.info("decision_attributed", decision_id=decision_id, action=action)
        return attribution
    
    def _get_feature_weight(self, feature_name: str) -> float:
        """Get importance weight for a feature"""
        weights = {
            "confidence": 1.0,
            "stress_level": 0.9,
            "action_readiness": 0.85,
            "arousal": 0.7,
            "valence": 0.7,
            "focus": 0.6,
            "bias_awareness": 0.5,
            "reflection_depth": 0.5,
            "exploration_tendency": 0.6,
            "task_complexity": 0.7,
            "task_urgency": 0.6,
            "task_novelty": 0.5,
        }
        return weights.get(feature_name, 0.5)
    
    def _get_candidate_key_features(self, candidate_id: str) -> List[str]:
        """Get features that most influenced a candidate"""
        edges = [
            e for e in self.attributions[self._last_decision_id()].causal_paths[0].path
            if e.startswith("feature_")
        ]
        return edges[:3] if edges else []
    
    def _last_decision_id(self) -> str:
        """Get last decision ID"""
        return list(self.attributions.keys())[-1] if self.attributions else ""
    
    def explain(self, attribution: DecisionAttribution) -> str:
        """Generate human-readable explanation"""
        parts = []
        
        if attribution.primary_features:
            top_feature = attribution.primary_features[0]
            parts.append(
                f"Primary driver: {top_feature[0]} (contribution: {top_feature[1]:.2f})"
            )
        
        if len(attribution.primary_features) > 1:
            features_text = ", ".join(
                f"{f[0]}={f[1]:.2f}" for f in attribution.primary_features[:3]
            )
            parts.append(f"Other factors: {features_text}")
        
        if attribution.alternative_attributions:
            best_alternative = attribution.alternative_attributions[0]
            parts.append(
                f"Alternative '{best_alternative['action']}' scored {best_alternative['lost_by']:.2f} lower"
            )
        
        return ". ".join(parts)
    
    def get_decision_explanation(self, decision_id: str) -> Dict[str, Any]:
        """Get full explanation for a decision"""
        attribution = self.attributions.get(decision_id)
        if not attribution:
            return {"error": "Decision not found"}
        
        return {
            "decision_id": decision_id,
            "action": attribution.action,
            "confidence": attribution.confidence,
            "explanation": self.explain(attribution),
            "primary_features": [
                {"feature": f[0], "contribution": f[1]}
                for f in attribution.primary_features
            ],
            "causal_paths": [
                {
                    "features": p.features,
                    "contribution": p.total_contribution,
                    "explanation": p.explanation
                }
                for p in attribution.causal_paths[:3]
            ],
            "alternatives": attribution.alternative_attributions,
            "timestamp": attribution.timestamp.isoformat()
        }


class PolicyAttributionSystem:
    """
    Main interface for policy attribution.
    
    Usage:
        system = PolicyAttributionSystem()
        
        # After each decision
        system.record(state, action, confidence, scoring, alternatives)
        
        # Get explanations
        explanation = system.explain_last()
        full_report = system.get_report(decision_id)
    """
    
    def __init__(self):
        self.analyzer = AttributionAnalyzer()
        self.current_graph: Optional[AttributionGraph] = None
        self.decision_counter = 0
        logger.info("policy_attribution_system_initialized")
    
    def record(
        self,
        state: Dict[str, float],
        action: str,
        confidence: float,
        scoring: Dict[str, float],
        alternatives: List[Dict]
    ) -> str:
        """Record a decision and create attribution"""
        self.decision_counter += 1
        decision_id = f"decision_{self.decision_counter}"
        
        attribution = self.analyzer.attribute_decision(
            decision_id=decision_id,
            timestamp=datetime.utcnow(),
            state=state,
            action=action,
            confidence=confidence,
            scoring_details=scoring,
            alternatives=alternatives
        )
        
        logger.info("decision_recorded", decision_id=decision_id, action=action)
        return decision_id
    
    def explain_last(self) -> str:
        """Explain the most recent decision"""
        if not self.analyzer.attributions:
            return "No decisions recorded"
        
        last_attribution = list(self.analyzer.attributions.values())[-1]
        return self.analyzer.explain(last_attribution)
    
    def get_report(self, decision_id: str) -> Dict[str, Any]:
        """Get full attribution report"""
        return self.analyzer.get_decision_explanation(decision_id)
    
    def get_feature_importance(self, window: int = 100) -> Dict[str, float]:
        """Get average feature importance over recent decisions"""
        if not self.analyzer.attributions:
            return {}
        
        all_features: Dict[str, List[float]] = {}
        
        for attr in list(self.analyzer.attributions.values())[-window:]:
            for feature, contribution in attr.primary_features:
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(contribution)
        
        return {
            feature: sum(values) / len(values) if values else 0
            for feature, values in all_features.items()
        }