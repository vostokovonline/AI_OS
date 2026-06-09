"""
Emergence Detection - Recognizing novel patterns from composition.

Key principle:
- Emergence = properties that arise from interactions, not from individual parts
- Emergence detection = identifying when the whole becomes greater than sum of parts

Types of emergence:
1. Conceptual - New concepts from combining beliefs
2. Behavioral - New behaviors from combining capabilities
3. Systemic - New properties from combining elements

NOT the same as novelty:
- Novelty: something new arrived
- Emergence: something NEW ARISES from existing things
"""

from typing import Dict, Any, Optional, List, Set, Tuple, FrozenSet
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib


class EmergenceType(Enum):
    """Types of emergent properties"""
    CONCEPTUAL = "conceptual"       # New concepts from beliefs
    BEHAVIORAL = "behavioral"       # New behaviors from capabilities
    SYSTEMIC = "systemic"           # New properties from elements
    RELATIONAL = "relational"        # New relationships from connections
    TEMPORAL = "temporal"           # New patterns over time


class EmergenceStrength(Enum):
    """Strength of emergent property"""
    WEAK = "weak"              # Marginal emergence
    MODERATE = "moderate"      # Clear emergence
    STRONG = "strong"          # Significant emergence
    CRITICAL = "critical"      # Paradigm shift


@dataclass(frozen=True)
class ComponentInteraction:
    """
    Represents an interaction between components that may produce emergence.
    """
    interaction_id: str
    component_a: str  # Component identifier
    component_b: str  # Component identifier
    interaction_type: str  # "synergy", "conflict", "neutral"
    strength: float  # 0-1, how strong this interaction is
    context: FrozenSet[str]  # What makes this interaction relevant


@dataclass(frozen=True)
class EmergentProperty:
    """
    A property that emerged from component interactions.
    
    Emergent properties are NOT reducible to their causes.
    They are irreducible higher-level phenomena.
    """
    property_id: str
    name: str
    emergence_type: str  # EmergenceType value
    components: FrozenSet[str]  # What emerged FROM
    interaction_context: Tuple[ComponentInteraction, ...]
    strength: float  # 0-1, how strong this emergence
    novelty: float  # 0-1, how novel this property is
    first_detected: str
    stability: float  # 0-1, how stable this property is
    version: int
    
    def classify_strength(self) -> EmergenceStrength:
        """Classify emergence strength"""
        if self.strength >= 0.8:
            return EmergenceStrength.CRITICAL
        elif self.strength >= 0.6:
            return EmergenceStrength.STRONG
        elif self.strength >= 0.3:
            return EmergenceStrength.MODERATE
        else:
            return EmergenceStrength.WEAK
    
    def is_stable(self) -> bool:
        """Check if this emergent property is stable"""
        return self.stability > 0.5


@dataclass(frozen=True)
class EmergenceState:
    """
    Complete emergence state of the cognitive system.
    
    Tracks what has emerged and what might emerge.
    """
    detected_properties: Tuple[EmergentProperty, ...]
    pending_interactions: Tuple[ComponentInteraction, ...]
    systemic_coherence: float  # 0-1, how coherent the system is
    emergence_rate: float  # How fast new properties are emerging
    dominant_emergence_type: str  # What type dominates
    version: int
    
    @staticmethod
    def empty() -> 'EmergenceState':
        """Create empty emergence state"""
        return EmergenceState(
            detected_properties=(),
            pending_interactions=(),
            systemic_coherence=0.5,
            emergence_rate=0.0,
            dominant_emergence_type="",
            version=0
        )
    
    def get_by_type(self, emergence_type: str) -> Tuple[EmergentProperty, ...]:
        """Get properties of a specific type"""
        return tuple(
            p for p in self.detected_properties
            if p.emergence_type == emergence_type
        )
    
    def get_strong(self, min_strength: float = 0.6) -> Tuple[EmergentProperty, ...]:
        """Get properties above strength threshold"""
        return tuple(
            p for p in self.detected_properties
            if p.strength >= min_strength
        )


# Emergence Detection Functions

def detect_interaction(
    component_a: str,
    component_b: str,
    interaction_type: str,
    strength: float,
    context: Set[str]
) -> ComponentInteraction:
    """Detect an interaction between components"""
    interaction_id = hashlib.md5(
        f"{component_a}:{component_b}:{interaction_type}".encode()
    ).hexdigest()[:12]
    
    return ComponentInteraction(
        interaction_id=interaction_id,
        component_a=component_a,
        component_b=component_b,
        interaction_type=interaction_type,
        strength=strength,
        context=frozenset(context)
    )


def analyze_interaction_potential(
    components: Tuple[str, ...],
    component_properties: Dict[str, Any]
) -> List[ComponentInteraction]:
    """
    Analyze potential interactions between components.
    
    This is a simplified analysis - real emergence detection
    would use learned models of component interactions.
    """
    interactions = []
    
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1:]:
            # Get properties
            props_a = component_properties.get(comp_a, {})
            props_b = component_properties.get(comp_b, {})
            
            # Check for synergy (complementary properties)
            synergy_score = 0.0
            if 'capability' in props_a and 'capability' in props_b:
                # Complementary capabilities = synergy
                synergy_score = 0.6
            
            if 'goal' in props_a and 'goal' in props_b:
                # Aligned goals = synergy
                if props_a['goal'] == props_b['goal']:
                    synergy_score = max(synergy_score, 0.8)
            
            if 'belief' in props_a and 'belief' in props_b:
                # Related beliefs = moderate synergy
                synergy_score = max(synergy_score, 0.4)
            
            if synergy_score > 0.3:
                interaction = detect_interaction(
                    comp_a, comp_b,
                    "synergy",
                    synergy_score,
                    {comp_a, comp_b}
                )
                interactions.append(interaction)
    
    return interactions


def detect_emergence(
    state: EmergenceState,
    interaction: ComponentInteraction,
    prior_properties: Optional[Dict[str, Any]] = None
) -> Optional[EmergentProperty]:
    """
    Detect if interaction produces emergent property.
    
    This is where the magic happens:
    - When components interact strongly enough
    - And the context is right
    - A new property emerges that is NOT in either component
    
    Returns None if no emergence detected.
    """
    # Need strong interaction
    if interaction.strength < 0.5:
        return None
    
    # Check if this interaction is novel (not seen before)
    for prop in state.detected_properties:
        if interaction.component_a in prop.components and interaction.component_b in prop.components:
            # Already detected this combination
            return None
    
    # Emergence detected!
    property_id = f"emergent_{interaction.interaction_id}"
    
    # Determine emergence type from context
    if "belief" in str(interaction.context):
        emergence_type = EmergenceType.CONCEPTUAL.value
    elif "capability" in str(interaction.context):
        emergence_type = EmergenceType.BEHAVIORAL.value
    elif "relationship" in str(interaction.context):
        emergence_type = EmergenceType.RELATIONAL.value
    else:
        emergence_type = EmergenceType.SYSTEMIC.value
    
    # Compute novelty (relative to existing properties)
    novelty = 0.5  # Simplified
    if prior_properties:
        existing = set(prior_properties.keys())
        new_components = interaction.context - existing
        novelty = min(1.0, len(new_components) / max(1, len(interaction.context)))
    
    return EmergentProperty(
        property_id=property_id,
        name=f"Emergent from {interaction.component_a} + {interaction.component_b}",
        emergence_type=emergence_type,
        components=frozenset([interaction.component_a, interaction.component_b]),
        interaction_context=(interaction,),
        strength=interaction.strength,
        novelty=novelty,
        first_detected=datetime.utcnow().isoformat(),
        stability=0.5,  # Initial stability
        version=0
    )


def add_emergent_property(
    state: EmergenceState,
    property: EmergentProperty
) -> EmergenceState:
    """Add newly detected emergent property"""
    new_properties = state.detected_properties + (property,)
    
    # Update emergence rate
    time_since_start = 1.0  # Simplified
    emergence_rate = len(new_properties) / time_since_start
    
    # Determine dominant type
    type_counts: Dict[str, int] = {}
    for p in new_properties:
        type_counts[p.emergence_type] = type_counts.get(p.emergence_type, 0) + 1
    
    dominant = max(type_counts, key=type_counts.get) if type_counts else ""
    
    return EmergenceState(
        detected_properties=new_properties,
        pending_interactions=state.pending_interactions,
        systemic_coherence=state.systemic_coherence,
        emergence_rate=emergence_rate,
        dominant_emergence_type=dominant,
        version=state.version + 1
    )


def compute_systemic_coherence(
    properties: Tuple[EmergentProperty, ...]
) -> float:
    """Compute how coherent the emergent properties are"""
    if not properties:
        return 0.5  # Neutral
    
    # Coherence = average stability of strong properties
    strong_properties = [p for p in properties if p.strength >= 0.5]
    
    if not strong_properties:
        return 0.3  # Low coherence
    
    return sum(p.stability for p in strong_properties) / len(strong_properties)


def update_stability(
    property: EmergentProperty,
    observed_stability: float
) -> EmergentProperty:
    """Update stability of emergent property based on observations"""
    # Moving average of stability
    new_stability = (property.stability + observed_stability) / 2
    
    return EmergentProperty(
        property_id=property.property_id,
        name=property.name,
        emergence_type=property.emergence_type,
        components=property.components,
        interaction_context=property.interaction_context,
        strength=property.strength,
        novelty=property.novelty,
        first_detected=property.first_detected,
        stability=new_stability,
        version=property.version + 1
    )


def detect_temporal_emergence(
    states: List[EmergenceState],
    time_window: int = 5
) -> Tuple[EmergentProperty, ...]:
    """
    Detect temporal emergence - patterns that emerge over time.
    
    This detects when a sequence of states produces something
    that was not present in any individual state.
    """
    if len(states) < time_window:
        return ()
    
    recent = states[-time_window:]
    
    # Look for properties that emerged and strengthened over time
    emergent_properties = []
    seen_ids = set()
    
    for state in recent:
        for prop in state.detected_properties:
            if prop.property_id not in seen_ids:
                seen_ids.add(prop.property_id)
                emergent_properties.append(prop)
    
    # Filter to those that show strengthening trend
    strengthening = []
    for prop_id in seen_ids:
        strengths = []
        for state in recent:
            for p in state.detected_properties:
                if p.property_id == prop_id:
                    strengths.append(p.strength)
        
        if len(strengths) >= 2:
            if strengths[-1] > strengths[0]:  # Strengthening
                # Find the property with highest recent strength
                strongest = max(
                    (p for state in recent for p in state.detected_properties if p.property_id == prop_id),
                    key=lambda x: x.strength
                )
                strengthening.append(strongest)
    
    return tuple(strengthening)