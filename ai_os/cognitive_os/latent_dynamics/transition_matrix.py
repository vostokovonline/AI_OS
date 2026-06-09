"""
MotifTransitionMatrix - Behavioral Flow Dynamics (Fixed)

CRITICAL FIX: Transition probabilities must be normalized!

Old bug:
    trans.probability = (trans.probability * (n - 1) + 1.0) / n
    
This makes each edge converge to 1.0 independently - mathematically broken!

Correct approach:
    P(B | A) = count(A->B) / sum(counts[A->*])
    
Store counts separately, compute probability at query time.
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TransitionStats:
    """
    Raw transition statistics - counts and metadata.
    
    Probability is computed at query time, not stored.
    """
    from_motif: str = ""
    to_motif: str = ""
    
    # Raw counts (NOT probabilities)
    count: int = 0
    
    # Confidence in estimate (based on sample size)
    confidence: float = 0.0
    
    # Temporal dynamics (still valid)
    avg_duration_ms: float = 0.0
    variance_ms: float = 0.0
    duration_samples: List[float] = field(default_factory=list)
    
    # Outcome correlation
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "from": self.from_motif,
            "to": self.to_motif,
            "count": self.count,
            "confidence": round(self.confidence, 3),
            "avg_duration_ms": round(self.avg_duration_ms, 0),
            "success_rate": round(self.success_rate, 3),
        }


@dataclass
class MotifFlowStats:
    """
    Flow statistics for a single motif.
    
    Describes how this attractor behaves in the cognitive flow.
    """
    motif_id: str = ""
    
    # Flow distribution (computed at query time)
    outgoing: Dict[str, float] = field(default_factory=dict)
    incoming: Dict[str, float] = field(default_factory=dict)
    
    # Raw counts
    outgoing_total: int = 0
    incoming_total: int = 0
    
    # Attractor properties
    entropy: float = 0.0
    stability: float = 0.0
    
    # Behavioral inertia
    self_loop_prob: float = 0.0
    self_loop_count: int = 0
    preferred_exit: str = ""
    preferred_entry: str = ""
    
    # Metastability metrics
    metastability: float = 0.0
    escape_likelihood: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "motif_id": self.motif_id,
            "outgoing_count": len(self.outgoing),
            "incoming_count": len(self.incoming),
            "outgoing_total": self.outgoing_total,
            "entropy": round(self.entropy, 3),
            "stability": round(self.stability, 3),
            "self_loop_prob": round(self.self_loop_prob, 3),
            "preferred_exit": self.preferred_exit,
            "preferred_entry": self.preferred_entry,
            "metastability": round(self.metastability, 3),
            "escape_likelihood": round(self.escape_likelihood, 3),
        }


class MotifTransitionMatrix:
    """
    Motif transition probability matrix.
    
    THIS IS the behavioral world model in primitive form.
    
    Key architectural decision:
    - Store raw counts, not probabilities
    - Compute P(B|A) = count(A->B) / sum(counts[A->*]) at query time
    
    This ensures proper normalization and mathematical correctness.
    """
    
    def __init__(self):
        # CRITICAL: Store counts, NOT probabilities
        self._counts: Dict[str, Dict[str, TransitionStats]] = defaultdict(dict)
        
        # Outgoing total per motif
        self._outgoing_totals: Dict[str, int] = defaultdict(int)
        
        # Stats cache
        self.motif_stats: Dict[str, MotifFlowStats] = {}
        
        # Observation history
        self.observations: List[Dict] = []
        
        # Computed properties
        self.total_observations: int = 0
        
        logger.info("motif_transition_matrix_initialized_fixed")
    
    def add_observation(
        self,
        from_motif: str,
        to_motif: str,
        duration_ms: float = 0.0,
        outcome: Optional[str] = None,
        trajectory_id: Optional[str] = None
    ) -> None:
        """Add a transition observation"""
        observation = {
            "from_motif": from_motif,
            "to_motif": to_motif,
            "duration_ms": duration_ms,
            "outcome": outcome,
            "trajectory_id": trajectory_id,
            "timestamp": datetime.utcnow()
        }
        
        self.observations.append(observation)
        self.total_observations += 1
        
        self._add_count(from_motif, to_motif, duration_ms, outcome)
        self._prune_old_observations()
    
    def _add_count(
        self,
        from_motif: str,
        to_motif: str,
        duration_ms: float,
        outcome: Optional[str]
    ) -> None:
        """Add count to transition tensor"""
        if to_motif not in self._counts[from_motif]:
            self._counts[from_motif][to_motif] = TransitionStats(
                from_motif=from_motif,
                to_motif=to_motif
            )
        
        stats = self._counts[from_motif][to_motif]
        stats.count += 1
        
        # Update duration statistics
        if duration_ms > 0:
            stats.duration_samples.append(duration_ms)
            n = len(stats.duration_samples)
            stats.avg_duration_ms = (
                stats.avg_duration_ms * (n - 1) + duration_ms
            ) / n
            if n > 1:
                # Online variance update
                old_var = stats.variance_ms
                stats.variance_ms = old_var + (duration_ms - stats.avg_duration_ms) ** 2 / n
        
        # Update outcome counts
        if outcome == "success":
            stats.success_count += 1
        elif outcome == "failure":
            stats.failure_count += 1
        
        # Update confidence
        stats.confidence = min(1.0, stats.count / 10)
        
        # Update outgoing total
        self._outgoing_totals[from_motif] += 1
        
        # Invalidate stats cache
        if from_motif in self.motif_stats:
            del self.motif_stats[from_motif]
        if to_motif in self.motif_stats:
            del self.motif_stats[to_motif]
    
    def _prune_old_observations(self) -> None:
        """Keep observation window bounded"""
        if len(self.observations) > 2000:
            self.observations = self.observations[-1000:]
    
    def get_transition_probability(
        self,
        from_motif: str,
        to_motif: str
    ) -> float:
        """
        Get P(to_motif | from_motif).
        
        Computed at query time: count(A->B) / sum(counts[A->*])
        """
        if from_motif not in self._counts:
            return 0.0
        
        if to_motif not in self._counts[from_motif]:
            return 0.0
        
        count = self._counts[from_motif][to_motif].count
        total = self._outgoing_totals[from_motif]
        
        return count / total if total > 0 else 0.0
    
    def get_next_motif_distribution(
        self,
        current_motif: str
    ) -> List[Tuple[str, float]]:
        """
        Get probability distribution over next motifs.
        
        Returns normalized P(next | current).
        """
        if current_motif not in self._counts:
            return []
        
        total = self._outgoing_totals[current_motif]
        if total == 0:
            return []
        
        dist = []
        for to_motif, stats in self._counts[current_motif].items():
            prob = stats.count / total
            dist.append((to_motif, prob))
        
        dist.sort(key=lambda x: x[1], reverse=True)
        return dist
    
    def predict_next_motif(
        self,
        current_motif: str,
        strategy: str = "expected"
    ) -> Optional[str]:
        """
        Predict next motif based on strategy.
        
        Strategies:
        - expected: Highest probability
        - optimistic: Best success rate
        - safe: Most consistent (highest count)
        """
        dist = self.get_next_motif_distribution(current_motif)
        
        if not dist:
            return None
        
        if strategy == "expected":
            return dist[0][0]
        
        elif strategy == "optimistic":
            best = None
            best_rate = -1
            for motif, prob in dist:
                stats = self._counts[current_motif][motif]
                if stats.success_rate > best_rate:
                    best_rate = stats.success_rate
                    best = motif
            return best
        
        elif strategy == "safe":
            best = None
            best_count = -1
            for motif, prob in dist:
                stats = self._counts[current_motif][motif]
                if stats.count > best_count:
                    best_count = stats.count
                    best = motif
            return best
        
        return None
    
    def compute_entropy(self, motif_id: str) -> float:
        """
        Compute entropy of motif's transitions.
        
        H = -sum(P * log(P))
        """
        dist = self.get_next_motif_distribution(motif_id)
        
        if not dist:
            return 0.0
        
        entropy = 0.0
        for _, prob in dist:
            if prob > 0:
                entropy -= prob * math.log(prob + 1e-10)
        
        return entropy
    
    def compute_stability(self, motif_id: str) -> float:
        """
        Compute stability of motif (inverse of entropy).
        
        High stability = low entropy = predictable transitions
        """
        dist = self.get_next_motif_distribution(motif_id)
        
        if not dist:
            return 1.0
        
        max_entropy = math.log(len(dist))
        entropy = self.compute_entropy(motif_id)
        
        return 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0
    
    def compute_metastability(self) -> Dict[str, float]:
        """
        Compute metastability for all motifs.
        
        Metastability = self_loop_prob * (1 - entropy)
        
        High metastability = system tends to stay in attractor
        """
        metastability = {}
        
        for motif_id in self._counts.keys():
            dist = self.get_next_motif_distribution(motif_id)
            
            if not dist:
                metastability[motif_id] = 0.0
                continue
            
            # Self-loop probability
            self_prob = 0.0
            for motif, prob in dist:
                if motif == motif_id:
                    self_prob = prob
                    break
            
            entropy = self.compute_entropy(motif_id)
            
            # Metastability: tendency to stay + low entropy
            metastability[motif_id] = self_prob * (1.0 - entropy / math.log(len(dist) + 1))
        
        return metastability
    
    def get_flow_statistics(self, motif_id: str) -> MotifFlowStats:
        """Get comprehensive flow statistics for motif"""
        if motif_id in self.motif_stats:
            return self.motif_stats[motif_id]
        
        stats = MotifFlowStats(motif_id=motif_id)
        
        # Outgoing distribution
        dist = self.get_next_motif_distribution(motif_id)
        stats.outgoing = {m: p for m, p in dist}
        stats.outgoing_total = self._outgoing_totals.get(motif_id, 0)
        
        if dist:
            stats.entropy = self.compute_entropy(motif_id)
            stats.stability = self.compute_stability(motif_id)
            
            # Self-loop
            for motif, prob in dist:
                if motif == motif_id:
                    stats.self_loop_prob = prob
                    stats.self_loop_count = self._counts[motif_id][motif_id].count
                    break
            
            # Preferred exit
            stats.preferred_exit = dist[0][0]
        
        # Incoming distribution
        incoming_dist = self.get_incoming_distribution(motif_id)
        stats.incoming = {m: p for m, p in incoming_dist}
        stats.incoming_total = sum(s.count for s in self._counts.values() if motif_id in s)
        
        if incoming_dist:
            stats.preferred_entry = incoming_dist[0][0]
        
        # Metastability
        metastability = self.compute_metastability()
        if motif_id in metastability:
            stats.metastability = metastability[motif_id]
            stats.escape_likelihood = 1.0 - metastability[motif_id]
        
        self.motif_stats[motif_id] = stats
        return stats
    
    def get_incoming_distribution(self, target_motif: str) -> List[Tuple[str, float]]:
        """Get P(source | target) - probability of being in target from source"""
        incoming_counts = {}
        
        for from_m, to_dict in self._counts.items():
            if target_motif in to_dict:
                incoming_counts[from_m] = to_dict[target_motif].count
        
        if not incoming_counts:
            return []
        
        total = sum(incoming_counts.values())
        
        dist = [(m, c / total) for m, c in incoming_counts.items()]
        dist.sort(key=lambda x: x[1], reverse=True)
        
        return dist
    
    def get_transition_graph(self) -> Dict:
        """Get transition graph for visualization"""
        edges = []
        
        for from_motif, to_dict in self._counts.items():
            total = self._outgoing_totals[from_motif]
            
            for to_motif, stats in to_dict.items():
                prob = stats.count / total if total > 0 else 0.0
                
                edges.append({
                    "from": from_motif,
                    "to": to_motif,
                    "probability": round(prob, 3),
                    "count": stats.count,
                    "confidence": round(stats.confidence, 3),
                    "success_rate": round(stats.success_rate, 3),
                })
        
        nodes = list(set(
            list(self._counts.keys()) + 
            [t for d in self._counts.values() for t in d.keys()]
        ))
        
        return {
            "nodes": [{"id": m} for m in nodes],
            "edges": edges
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive matrix statistics"""
        metastability = self.compute_metastability()
        
        total_edges = sum(len(t) for t in self._counts.values())
        
        return {
            "total_observations": self.total_observations,
            "unique_motifs": len(self._counts),
            "total_edges": total_edges,
            "avg_entropy": sum(self.compute_entropy(m) for m in self._counts.keys()) / max(1, len(self._counts)),
            "avg_metastability": sum(m.values()) / max(1, len(m)) if metastability else 0,
            "highest_entropy_motif": (
                max(self._counts.keys(), key=self.compute_entropy) 
                if self._counts else None
            ),
            "most_stable_motif": (
                max(self._counts.keys(), key=self.compute_stability)
                if self._counts else None
            ),
        }
    
    def sample_next_motif(
        self,
        current_motif: str,
        temperature: float = 1.0
    ) -> Optional[str]:
        """
        Sample next motif from distribution.
        
        temperature:
        - 0 = deterministic (most likely)
        - 1 = natural distribution
        - >1 = more random
        """
        dist = self.get_next_motif_distribution(current_motif)
        
        if not dist:
            return None
        
        if temperature == 0:
            return dist[0][0]
        
        # Apply temperature
        weights = [p ** (1/temperature) for _, p in dist]
        total = sum(weights)
        
        import random
        r = random.random() * total
        cumsum = 0
        
        for motif, weight in zip([m for m, _ in dist], weights):
            cumsum += weight
            if r <= cumsum:
                return motif
        
        return dist[-1][0]
    
    def get_transition_tensor(self) -> Dict[str, Dict[str, int]]:
        """
        Get raw count tensor for external use.
        
        counts[from_motif][to_motif] = count
        """
        return {
            from_m: {
                to_m: stats.count 
                for to_m, stats in to_dict.items()
            }
            for from_m, to_dict in self._counts.items()
        }


# Factory
def create_transition_matrix() -> MotifTransitionMatrix:
    return MotifTransitionMatrix()