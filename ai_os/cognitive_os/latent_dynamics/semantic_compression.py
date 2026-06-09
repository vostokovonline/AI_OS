"""
Semantic Compression - Macro-events, Trajectory Chunks, Behavior Motifs

Сжимает event stream в семантически значимые единицы:

Event Stream (syntactic) → Semantic Compression → Behavior Macros

Предотвращает "log explosion" и создаёт actionable units.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import uuid4
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class BehaviorMotif:
    """
    Поведенческий мотив - семантически сжатая единица.
    
    Вместо списка событий, теперь:
    - Название (what happened)
    - Ключевые характеристики
    - Вход/выход состояния
    - Семантическое описание
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Identification
    motif_type: str = ""  # "exploration", "exploitation", "recovery", "escalation"
    name: str = ""
    description: str = ""
    
    # Compression
    event_count: int = 0
    duration_ms: float = 0.0
    event_ids: List[str] = field(default_factory=list)
    
    # Semantic markers
    key_causes: List[str] = field(default_factory=list)
    key_outcome: Optional[str] = None
    key_action: Optional[str] = None
    
    # Statistics
    avg_confidence: float = 0.0
    confidence_trajectory: List[float] = field(default_factory=list)
    stress_trajectory: List[float] = field(default_factory=list)
    
    # Quality
    compression_ratio: float = 0.0  # events / macro_events
    coherence_score: float = 0.0  # How coherent the motif is
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "motif_type": self.motif_type,
            "name": self.name,
            "description": self.description,
            "event_count": self.event_count,
            "duration_ms": self.duration_ms,
            "key_causes": self.key_causes,
            "key_outcome": self.key_outcome,
            "key_action": self.key_action,
            "compression_ratio": self.compression_ratio,
            "coherence_score": self.coherence_score,
        }


@dataclass
class TrajectoryChunk:
    """
    Trajectory chunk - законченный кусок поведения.
    
    Это группа мотивов, образующая законченное поведение.
    """
    id: str = field(default_factory=lambda: str(uuid4()))
    
    # Content
    motifs: List[BehaviorMotif] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Semantics
    goal: Optional[str] = None
    outcome: Optional[str] = None
    
    # Metrics
    total_events: int = 0
    total_duration_ms: float = 0.0
    success_score: float = 0.0
    
    # Learning
    pattern_signature: str = ""  # Hash of the pattern
    frequency: int = 0  # How often this pattern occurs
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "motif_count": len(self.motifs),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "goal": self.goal,
            "outcome": self.outcome,
            "total_events": self.total_events,
            "duration_ms": self.total_duration_ms,
            "success_score": self.success_score,
            "frequency": self.frequency,
        }


class SemanticCompressor:
    """
    Semantic Compressor - сжимает event stream.
    
    Преобразует низкоуровневые события в семантически значимые макро-события.
    """
    
    MOTIF_PATTERNS = {
        "exploration": {
            "events": ["candidate_generated", "simulation_branch"],
            "indicators": {"exploration_tendency": ">0.6"},
            "outcome": "novel_action"
        },
        "exploitation": {
            "events": ["candidate_selected"],
            "indicators": {"confidence": ">0.7"},
            "outcome": "proven_action"
        },
        "recovery": {
            "events": ["reconsider", "retry"],
            "indicators": {"stress_level": ">0.5"},
            "outcome": "stable_state"
        },
        "escalation": {
            "events": ["decompose"],
            "indicators": {"task_complexity": ">0.6"},
            "outcome": "hierarchical_action"
        },
        "stabilization": {
            "events": ["wait"],
            "indicators": {"action_readiness": "<0.4"},
            "outcome": "waited"
        },
    }
    
    def __init__(self):
        self.motifs: List[BehaviorMotif] = []
        self.chunks: List[TrajectoryChunk] = []
        self.active_chunk: Optional[TrajectoryChunk] = None
        logger.info("semantic_compressor_initialized")
    
    def compress_events(
        self,
        events: List[Dict],
        transitions: List[Dict]
    ) -> List[BehaviorMotif]:
        """Compress raw events into motifs"""
        if not events:
            return []
        
        motifs = []
        
        current_motif = None
        motif_events = []
        
        for event in events:
            event_type = event.get("event_type", "")
            
            if self._is_boundary_event(event_type):
                if current_motif and motif_events:
                    motif = self._create_motif_from_events(
                        motif_events,
                        self._classify_motif_type(motif_events)
                    )
                    motifs.append(motif)
                    self.motifs.append(motif)
                    motif_events = []
                    current_motif = None
            else:
                motif_events.append(event)
        
        if motif_events:
            motif = self._create_motif_from_events(
                motif_events,
                self._classify_motif_type(motif_events)
            )
            motifs.append(motif)
            self.motifs.append(motif)
        
        logger.info("events_compressed", raw_events=len(events), motifs=len(motifs))
        
        return motifs
    
    def _is_boundary_event(self, event_type: str) -> bool:
        """Check if event is a motif boundary"""
        return event_type in ["transaction_start", "transaction_commit", "transaction_rollback"]
    
    def _classify_motif_type(self, events: List[Dict]) -> str:
        """Classify the type of motif based on events"""
        event_types = [e.get("event_type", "") for e in events]
        
        if "simulation_branch" in event_types:
            return "exploration"
        if "candidate_selected" in event_types:
            selected = [e for e in events if e.get("event_type") == "candidate_selected"]
            if selected and selected[0].get("data", {}).get("confidence", 0) > 0.7:
                return "exploitation"
        if "reconsider" in event_types or "retry" in event_types:
            return "recovery"
        if "decompose" in event_types:
            return "escalation"
        if "wait" in event_types:
            return "stabilization"
        
        return "general"
    
    def _create_motif_from_events(
        self,
        events: List[Dict],
        motif_type: str
    ) -> BehaviorMotif:
        """Create a motif from a sequence of events"""
        timestamps = [datetime.fromisoformat(e.get("timestamp", datetime.utcnow().isoformat())) for e in events]
        duration_ms = (max(timestamps) - min(timestamps)).total_seconds() * 1000
        
        # Extract key data
        confidences = []
        stresses = []
        actions = []
        causes = []
        
        for event in events:
            if "candidate_selected" in event.get("event_type", ""):
                data = event.get("data", {})
                confidences.append(data.get("confidence", 0.5))
                actions.append(data.get("action"))
            if "state_snapshot" in event.get("event_type", ""):
                data = event.get("data", {})
                stresses.append(data.get("stress_level", 0))
        
        outcome_events = [e for e in events if e.get("event_type") == "outcome_recorded"]
        outcome = outcome_events[-1].get("data", {}).get("outcome") if outcome_events else None
        
        motif_names = {
            "exploration": "Exploration Burst",
            "exploitation": "Goal-Oriented Execution",
            "recovery": "State Recovery",
            "escalation": "Complexity Escalation",
            "stabilization": "Wait and Stabilize",
            "general": "Mixed Behavior"
        }
        
        motif = BehaviorMotif(
            motif_type=motif_type,
            name=motif_names.get(motif_type, "Unknown"),
            description=self._generate_motif_description(events, motif_type),
            event_count=len(events),
            duration_ms=duration_ms,
            event_ids=[e.get("id", "") for e in events],
            key_outcome=outcome,
            key_action=actions[-1] if actions else None,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.5,
            confidence_trajectory=confidences,
            stress_trajectory=stresses,
            compression_ratio=1.0 / len(events) if events else 1.0,
            coherence_score=self._compute_coherence(events)
        )
        
        return motif
    
    def _generate_motif_description(self, events: List[Dict], motif_type: str) -> str:
        """Generate semantic description of motif"""
        descriptions = {
            "exploration": "System explored alternative action paths with simulation",
            "exploitation": "System executed a high-confidence action",
            "recovery": "System attempted to recover from stress or failure",
            "escalation": "System decomposed complex task into subcomponents",
            "stabilization": "System paused to stabilize state",
            "general": "System performed mixed cognitive operations"
        }
        return descriptions.get(motif_type, "")
    
    def _compute_coherence(self, events: List[Dict]) -> float:
        """Compute how coherent the motif is"""
        if len(events) < 2:
            return 1.0
        
        confidence_scores = []
        for event in events:
            if "candidate_selected" in event.get("event_type", ""):
                confidence_scores.append(event.get("data", {}).get("confidence", 0.5))
        
        if not confidence_scores:
            return 0.5
        
        variance = sum((c - sum(confidence_scores) / len(confidence_scores)) ** 2 for c in confidence_scores) / len(confidence_scores)
        coherence = max(0, 1 - variance)
        
        return coherence
    
    def create_chunks_from_motifs(
        self,
        motifs: List[BehaviorMotif],
        goal: Optional[str] = None,
        outcome: Optional[str] = None
    ) -> List[TrajectoryChunk]:
        """Group motifs into trajectory chunks"""
        if not motifs:
            return []
        
        chunk = TrajectoryChunk(
            start_time=motifs[0].start_time if hasattr(motifs[0], 'start_time') else datetime.utcnow(),
            goal=goal,
            outcome=outcome
        )
        
        for motif in motifs:
            chunk.motifs.append(motif)
        
        chunk.end_time = motifs[-1].end_time if hasattr(motifs[-1], 'end_time') else datetime.utcnow()
        chunk.total_events = sum(m.event_count for m in chunk.motifs)
        chunk.total_duration_ms = sum(m.duration_ms for m in chunk.motifs)
        chunk.success_score = 1.0 if outcome == "success" else 0.5 if outcome else 0.0
        
        self.chunks.append(chunk)
        
        logger.info("chunk_created", motif_count=len(motifs), total_events=chunk.total_events)
        
        return [chunk]
    
    def get_statistics(self) -> Dict:
        """Get compression statistics"""
        return {
            "total_motifs": len(self.motifs),
            "total_chunks": len(self.chunks),
            "motif_types": self._count_motif_types(),
            "compression_ratio": len(self.motifs) / sum(m.event_count for m in self.motifs) if self.motifs else 0,
        }
    
    def _count_motif_types(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for motif in self.motifs:
            counts[motif.motif_type] += 1
        return dict(counts)


class CausalConfidence:
    """
    Causal Confidence - confidence in causal edges.
    
    Каждая causal edge имеет:
    - causal_strength (насколько сильна причина)
    - counterfactual_support (насколько хорошо counterfactual подтверждает)
    - temporal_confidence (насколько стабильно во времени)
    """
    
    def __init__(self):
        self.edge_confidences: Dict[str, float] = {}
        self.observation_history: Dict[str, List[bool]] = defaultdict(list)
        logger.info("causal_confidence_initialized")
    
    def update_edge_confidence(
        self,
        edge_id: str,
        observed_outcome: bool
    ) -> float:
        """Update confidence for a causal edge based on observation"""
        history = self.observation_history[edge_id]
        history.append(observed_outcome)
        
        if len(history) > 20:
            history = history[-20:]
        
        conf = sum(history) / len(history) if history else 0.5
        
        self.edge_confidences[edge_id] = conf
        
        return conf
    
    def get_edge_confidence(self, edge_id: str) -> float:
        """Get current confidence for edge"""
        return self.edge_confidences.get(edge_id, 0.5)
    
    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get distribution of edge confidences"""
        if not self.edge_confidences:
            return {"high": 0, "medium": 0, "low": 0}
        
        high = sum(1 for c in self.edge_confidences.values() if c > 0.7)
        medium = sum(1 for c in self.edge_confidences.values() if 0.4 < c <= 0.7)
        low = sum(1 for c in self.edge_confidences.values() if c <= 0.4)
        total = len(self.edge_confidences)
        
        return {
            "high": high / total if total else 0,
            "medium": medium / total if total else 0,
            "low": low / total if total else 0,
        }
    
    def get_unreliable_edges(self, threshold: float = 0.3) -> List[str]:
        """Get edges with low confidence"""
        return [e for e, c in self.edge_confidences.items() if c < threshold]


def create_semantic_compressor() -> SemanticCompressor:
    return SemanticCompressor()