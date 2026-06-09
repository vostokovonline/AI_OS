"""
InterpretationLayer - Symbolic Interpretation of Clean Core State

CRITICAL ARCHITECTURAL CONSTRAINT:

MotifState in core has NO labels. It has:
- centroid
- density
- stability
- entropy
- basin_radius
- transitions

This layer provides human-readable interpretations WITHOUT
changing the core representation.

This prevents symbolic leakage from contaminating the learned representations.
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class MotifInterpretation:
    """
    Human-readable interpretation of a motif.
    
    This is NOT part of the core motif representation.
    It's a projection layer for humans/external systems.
    """
    motif_id: str = ""
    
    # Inferred semantics (from geometry, not labels)
    inferred_type: str = ""  # exploration, execution, recovery, etc.
    type_confidence: float = 0.0
    
    # Semantic description
    description: str = ""
    short_label: str = ""
    
    # Behavioral interpretation
    behavioral_signature: str = ""
    cognitive_function: str = ""
    
    # Interpretation quality
    confidence: float = 0.0
    basis: List[str] = field(default_factory=list)  # Why we think this
    
    def to_dict(self) -> Dict:
        return {
            "motif_id": self.motif_id,
            "inferred_type": self.inferred_type,
            "type_confidence": round(self.type_confidence, 3),
            "description": self.description,
            "short_label": self.short_label,
            "behavioral_signature": self.behavioral_signature,
            "cognitive_function": self.cognitive_function,
            "confidence": round(self.confidence, 3),
            "basis": self.basis,
        }


@dataclass
class TrajectoryInterpretation:
    """
    Human-readable interpretation of a trajectory.
    
    NOT stored in core - this is an interpretation layer.
    """
    trajectory_id: str = ""
    
    # What this trajectory "means"
    trajectory_type: str = ""
    interpretation: str = ""
    
    # How unusual this is
    typicality: float = 0.0  # How typical for this motif
    novelty_score: float = 0.0
    
    # Causal interpretation
    likely_causes: List[str] = field(default_factory=list)
    likely_effects: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "type": self.trajectory_type,
            "interpretation": self.interpretation,
            "typicality": round(self.typicality, 3),
            "novelty_score": round(self.novelty_score, 3),
            "likely_causes": self.likely_causes,
            "likely_effects": self.likely_effects,
        }


class MotifInterpreter:
    """
    Interprets motif geometry into human-readable semantics.
    
    Uses geometric features to infer behavioral meaning.
    """
    
    def __init__(self):
        self.interpretations: Dict[str, MotifInterpretation] = {}
        self.type_patterns = self._init_patterns()
        logger.info("motif_interpreter_initialized")
    
    def _init_patterns(self) -> Dict[str, Dict]:
        """Initialize pattern matching for motif types"""
        return {
            "high_entropy": {
                "condition": lambda m: m.entropy > 0.7,
                "type": "exploration",
                "confidence_fn": lambda m: m.entropy
            },
            "low_entropy_high_stability": {
                "condition": lambda m: m.entropy < 0.3 and m.stability > 0.7,
                "type": "execution",
                "confidence_fn": lambda m: m.stability
            },
            "high_self_loop": {
                "condition": lambda m: m.outgoing_transitions.get(m.motif_id, 0) > 0.6,
                "type": "persistence",
                "confidence_fn": lambda m: m.outgoing_transitions.get(m.motif_id, 0)
            },
            "high_decay": {
                "condition": lambda m: m.decay_rate > 0.5,
                "type": "transient",
                "confidence_fn": lambda m: m.decay_rate
            },
            "small_basin": {
                "condition": lambda m: m.basin_radius < 0.3,
                "type": "specific",
                "confidence_fn": lambda m: 1.0 - m.basin_radius
            },
            "large_basin": {
                "condition": lambda m: m.basin_radius > 0.7,
                "type": "general",
                "confidence_fn": lambda m: m.basin_radius
            },
        }
    
    def interpret_motif(self, motif_state) -> MotifInterpretation:
        """Interpret a motif's geometry into semantics"""
        motif_id = motif_state.motif_id
        
        if motif_id in self.interpretations:
            return self.interpretations[motif_id]
        
        interpretation = MotifInterpretation(motif_id=motif_id)
        
        best_pattern = None
        best_confidence = 0.0
        
        for pattern_name, pattern in self.type_patterns.items():
            if pattern["condition"](motif_state):
                confidence = pattern["confidence_fn"](motif_state)
                if confidence > best_confidence:
                    best_pattern = pattern_name
                    best_confidence = confidence
                    interpretation.inferred_type = pattern["type"]
                    interpretation.type_confidence = confidence
        
        interpretation.description = self._generate_description(motif_state, interpretation)
        interpretation.short_label = self._generate_label(interpretation)
        interpretation.behavioral_signature = self._compute_signature(motif_state)
        interpretation.cognitive_function = self._infer_function(interpretation)
        interpretation.confidence = best_confidence
        interpretation.basis = self._generate_basis(motif_state, interpretation)
        
        self.interpretations[motif_id] = interpretation
        return interpretation
    
    def _generate_description(
        self,
        motif_state,
        interpretation: MotifInterpretation
    ) -> str:
        """Generate semantic description"""
        templates = {
            "exploration": "Cognition explores diverse behavioral paths with high uncertainty",
            "execution": "Cognition executes focused, stable behavior patterns",
            "persistence": "Cognition tends to stay within this attractor basin",
            "transient": "Cognition passes through this region quickly",
            "specific": "Cognition exhibits highly specific behavioral patterns",
            "general": "Cognition has broad behavioral flexibility in this region",
        }
        
        base = templates.get(interpretation.inferred_type, "Cognition in mixed state")
        
        if motif_state.density > 0.5:
            base += " (frequently visited)"
        elif motif_state.density < 0.1:
            base += " (rarely visited)"
        
        return base
    
    def _generate_label(self, interpretation: MotifInterpretation) -> str:
        """Generate short label"""
        labels = {
            "exploration": "EXPLORE",
            "execution": "EXEC",
            "persistence": "STAY",
            "transient": "PASS",
            "specific": "FOCUS",
            "general": "FLEX",
        }
        return labels.get(interpretation.inferred_type, "MIXED")
    
    def _compute_signature(self, motif_state) -> str:
        """Compute behavioral signature from geometry"""
        parts = []
        
        if motif_state.entropy > 0.5:
            parts.append("variable")
        else:
            parts.append("consistent")
        
        if motif_state.stability > 0.7:
            parts.append("stable")
        elif motif_state.stability < 0.3:
            parts.append("unstable")
        
        if motif_state.density > 0.5:
            parts.append("common")
        elif motif_state.density < 0.1:
            parts.append("rare")
        
        return "_".join(parts)
    
    def _infer_function(self, interpretation: MotifInterpretation) -> str:
        """Infer cognitive function"""
        functions = {
            "exploration": "Discovery and information gathering",
            "execution": "Goal achievement and task completion",
            "persistence": "Maintaining cognitive state",
            "transient": "Transition and movement",
            "specific": "Targeted problem solving",
            "general": "Adaptive behavior",
        }
        return functions.get(interpretation.inferred_type, "Mixed function")
    
    def _generate_basis(
        self,
        motif_state,
        interpretation: MotifInterpretation
    ) -> List[str]:
        """Generate explanation of why we think this"""
        basis = []
        
        if motif_state.entropy > 0.5:
            basis.append(f"High entropy ({motif_state.entropy:.2f}) indicates exploration")
        elif motif_state.entropy < 0.3:
            basis.append(f"Low entropy ({motif_state.entropy:.2f}) indicates focused behavior")
        
        if motif_state.stability > 0.7:
            basis.append(f"High stability ({motif_state.stability:.2f}) indicates reliable patterns")
        
        if motif_state.density > 0.5:
            basis.append(f"High density ({motif_state.density:.2f}) indicates frequently used")
        
        self_loop = motif_state.outgoing_transitions.get(motif_state.motif_id, 0)
        if self_loop > 0.5:
            basis.append(f"Strong self-loop ({self_loop:.2f}) indicates persistence tendency")
        
        return basis
    
    def batch_interpret(self, motif_states: List) -> List[MotifInterpretation]:
        """Interpret multiple motifs"""
        return [self.interpret_motif(m) for m in motif_states]


class TrajectoryInterpreter:
    """
    Interprets individual trajectories.
    
    NOT stored in core - interpretation layer only.
    """
    
    def __init__(self, motif_interpreter: MotifInterpreter):
        self.motif_interpreter = motif_interpreter
        self.interpretations: Dict[str, TrajectoryInterpretation] = {}
        logger.info("trajectory_interpreter_initialized")
    
    def interpret_trajectory(
        self,
        trajectory_node,
        context: Optional[Dict] = None
    ) -> TrajectoryInterpretation:
        """Interpret a trajectory"""
        traj_id = trajectory_node.trajectory_id
        
        if traj_id in self.interpretations:
            return self.interpretations[traj_id]
        
        interpretation = TrajectoryInterpretation(trajectory_id=traj_id)
        
        interpretation.trajectory_type = self._classify_trajectory(trajectory_node)
        interpretation.interpretation = self._generate_interpretation(trajectory_node, interpretation)
        interpretation.typicality = self._compute_typicality(trajectory_node)
        interpretation.novelty_score = self._compute_novelty(trajectory_node)
        interpretation.likely_causes = self._infer_causes(trajectory_node, context)
        interpretation.likely_effects = self._infer_effects(trajectory_node, context)
        
        self.interpretations[traj_id] = interpretation
        return interpretation
    
    def _classify_trajectory(self, node) -> str:
        """Classify trajectory type"""
        if node.curvature > 0.5 and node.volatility > 0.4:
            return "exploratory"
        elif node.directness > 0.8:
            return "direct"
        elif node.momentum > 0.7:
            return "momentum"
        elif node.divergence_from_previous > 0.5:
            return "divergent"
        else:
            return "standard"
    
    def _generate_interpretation(
        self,
        node,
        interpretation: TrajectoryInterpretation
    ) -> str:
        """Generate human-readable interpretation"""
        templates = {
            "exploratory": "System explored alternative paths with high curvature",
            "direct": "System pursued direct path to goal",
            "momentum": "System maintained high behavioral momentum",
            "divergent": "System deviated from previous trajectory pattern",
            "standard": "System executed standard behavioral pattern",
        }
        
        base = templates.get(interpretation.trajectory_type, "System performed behavior")
        
        if node.motif_id:
            motif_interp = self.motif_interpreter.interpretations.get(node.motif_id)
            if motif_interp:
                base += f" within {motif_interp.short_label} motif"
        
        return base
    
    def _compute_typicality(self, node) -> float:
        """How typical is this trajectory for its motif"""
        if not node.motif_id:
            return 0.5
        
        motif_interp = self.motif_interpreter.interpretations.get(node.motif_id)
        if not motif_interp:
            return 0.5
        
        if motif_interp.inferred_type == "execution":
            return node.directness
        elif motif_interp.inferred_type == "exploration":
            return 1.0 - node.directness
        else:
            return 0.5
    
    def _compute_novelty(self, node) -> float:
        """How novel is this trajectory"""
        novelty = node.divergence_from_previous
        
        if node.motif_confidence < 0.5:
            novelty += 0.3
        
        return min(1.0, novelty)
    
    def _infer_causes(
        self,
        node,
        context: Optional[Dict]
    ) -> List[str]:
        """Infer likely causes of this trajectory"""
        causes = []
        
        if node.divergence_from_previous > 0.5:
            causes.append("Previous trajectory completed, transition occurred")
        
        if node.previous_nodes:
            causes.append("Continuation of recent cognitive pattern")
        
        if context and context.get("novel_context"):
            causes.append("Novel situation triggered adaptation")
        
        if node.curvature > 0.5:
            causes.append("System exploring alternative paths")
        
        return causes
    
    def _infer_effects(
        self,
        node,
        context: Optional[Dict]
    ) -> List[str]:
        """Infer likely effects of this trajectory"""
        effects = []
        
        if node.next_nodes:
            effects.append("Will influence next trajectory's starting point")
        
        if node.convergence_towards_next > 0.7:
            effects.append("High convergence suggests stable next state")
        
        if node.motif_id:
            effects.append(f"Reinforces {node.motif_id} attractor basin")
        
        return effects


class InterpretationLayer:
    """
    Top-level interpretation layer.
    
    Provides unified interface for all interpretations
    WITHOUT modifying core representations.
    """
    
    def __init__(self):
        self.motif_interpreter = MotifInterpreter()
        self.trajectory_interpreter = TrajectoryInterpreter(self.motif_interpreter)
        logger.info("interpretation_layer_initialized")
    
    def interpret_motif(self, motif_state) -> MotifInterpretation:
        """Interpret a motif"""
        return self.motif_interpreter.interpret_motif(motif_state)
    
    def interpret_trajectory(
        self,
        trajectory_node,
        context: Optional[Dict] = None
    ) -> TrajectoryInterpretation:
        """Interpret a trajectory"""
        return self.trajectory_interpreter.interpret_trajectory(trajectory_node, context)
    
    def get_all_interpretations(self) -> Dict:
        """Get all interpretations"""
        return {
            "motifs": [i.to_dict() for i in self.motif_interpreter.interpretations.values()],
            "trajectories": [i.to_dict() for i in self.trajectory_interpreter.interpretations.values()],
        }


# Factory
def create_interpretation_layer() -> InterpretationLayer:
    return InterpretationLayer()