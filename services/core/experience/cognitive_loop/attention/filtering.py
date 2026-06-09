"""
Attention Filtering - Determines which inputs deserve processing.

Attention filtering is NOT about prioritization (that's salience).
It's about deciding WHAT enters the cognitive system at all.

Key principle:
- Not all inputs are created equal
- Some inputs are noise, some are signal
- Filtering prevents cognitive overload BEFORE salience computation

Types of filtering:
1. Signal/Noise - Is this worth processing?
2. Relevance - Does this relate to current goals/tensions?
3. Novelty - Has this been seen before?
4. Authority - Should this be trusted?
"""

from typing import Dict, Any, Optional, Set, FrozenSet, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib


class InputSignal(Enum):
    """Classification of input signal quality"""
    HIGH_VALUE = "high_value"       # Direct relevance, strong signal
    MEDIUM_VALUE = "medium_value"  # Some relevance, moderate signal
    LOW_VALUE = "low_value"        # Weak signal, consider discarding
    NOISE = "noise"                # Not worth processing
    UNKNOWN = "unknown"            # Unable to classify


@dataclass(frozen=True)
class InputSource:
    """
    Represents an input to the cognitive system.
    
    Inputs can be:
    - External events (user commands, environment changes)
    - Internal events (goal completions, belief updates)
    - Observations (sensory data, measurements)
    """
    source_id: str
    source_type: str  # "external", "internal", "observation"
    content: str  # Human-readable description
    raw_data: FrozenSet[Tuple[str, Any]]  # Structured key-value pairs
    timestamp: str
    authority_score: float  # 0-1, trust level of source
    novelty_score: float  # 0-1, how novel is this
    relevance_score: float  # 0-1, relevance to current state
    
    def classify(self) -> InputSignal:
        """Classify this input's signal quality"""
        total = self.authority_score + self.novelty_score + self.relevance_score
        
        if total >= 2.1:  # Average >= 0.7
            return InputSignal.HIGH_VALUE
        elif total >= 1.2:  # Average >= 0.4
            return InputSignal.MEDIUM_VALUE
        elif total >= 0.3:  # Average >= 0.1
            return InputSignal.LOW_VALUE
        else:
            return InputSignal.NOISE


@dataclass(frozen=True)
class FilterConfig:
    """
    Configuration for attention filtering.
    
    These values are identity-driven (set by self model).
    """
    noise_threshold: float = 0.2  # Below this, classified as noise
    min_authority: float = 0.1  # Must have at least this trust
    max_inputs_per_cycle: int = 20  # Max inputs to consider per cycle
    novelty_bonus: float = 0.2  # Bonus for novel inputs
    repetition_penalty: float = 0.3  # Penalty for repeated inputs
    
    @staticmethod
    def from_identity(autonomy: float, curiosity: float) -> 'FilterConfig':
        """Create config from identity parameters"""
        return FilterConfig(
            noise_threshold=0.2 - (autonomy * 0.1),  # Higher autonomy = stricter
            min_authority=0.1 + (curiosity * 0.05),  # Higher curiosity = more open
            max_inputs_per_cycle=int(20 + (curiosity * 10)),  # More inputs for curious
            novelty_bonus=0.2 + (curiosity * 0.1),  # Higher curiosity = more novelty bonus
            repetition_penalty=max(0.1, 0.5 - (autonomy * 0.2))  # Higher autonomy = less penalty
        )


@dataclass(frozen=True)
class FilteredInputs:
    """
    Result of attention filtering.
    
    Contains only inputs that passed the filter.
    """
    inputs: Tuple[InputSource, ...]
    total_considered: int
    discarded_count: int
    filter_version: int
    
    def get_by_signal(self, signal: InputSignal) -> Tuple[InputSource, ...]:
        """Get inputs of a specific signal classification"""
        return tuple(i for i in self.inputs if i.classify() == signal)
    
    def get_high_value(self) -> Tuple[InputSource, ...]:
        return self.get_by_signal(InputSignal.HIGH_VALUE)
    
    def get_medium_value(self) -> Tuple[InputSource, ...]:
        return self.get_by_signal(InputSignal.MEDIUM_VALUE)


def filter_inputs(
    inputs: Tuple[InputSource, ...],
    config: FilterConfig,
    seen_sources: Optional[FrozenSet[str]] = None
) -> FilteredInputs:
    """
    Filter inputs based on configured thresholds.
    
    This is the core filtering function.
    It removes noise before salience computation.
    """
    seen = seen_sources or frozenset()
    
    filtered = []
    discarded = 0
    
    for inp in inputs[:config.max_inputs_per_cycle]:
        # Check noise threshold
        signal = inp.classify()
        
        if signal == InputSignal.NOISE:
            discarded += 1
            continue
        
        # Check authority minimum
        if inp.authority_score < config.min_authority:
            discarded += 1
            continue
        
        # Apply novelty bonus
        adjusted_relevance = inp.relevance_score
        if inp.source_id not in seen:
            adjusted_relevance += config.novelty_bonus
        else:
            adjusted_relevance -= config.repetition_penalty
        
        if adjusted_relevance < config.noise_threshold:
            discarded += 1
            continue
        
        # Input passed filter
        filtered.append(inp)
    
    return FilteredInputs(
        inputs=tuple(filtered),
        total_considered=len(inputs),
        discarded_count=discarded,
        filter_version=len(inputs)
    )


def compute_input_features(
    source_id: str,
    content: str,
    metadata: Dict[str, Any]
) -> Tuple[str, FrozenSet[Tuple[str, Any]]]:
    """
    Compute features for an input source.
    
    Returns:
    - source_id (possibly hashed for anonymity)
    - structured raw_data
    """
    # Hash source for privacy/identity
    hashed_id = hashlib.sha256(source_id.encode()).hexdigest()[:16]
    
    # Convert metadata to frozen tuples
    raw_data = frozenset(
        (str(k), str(v)) for k, v in metadata.items()
    )
    
    return hashed_id, raw_data