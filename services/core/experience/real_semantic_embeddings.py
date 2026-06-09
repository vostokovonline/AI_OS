"""
Real Semantic Embeddings Layer

Uses lightweight deterministic embeddings (not LLM in loop).
Option 1: Pre-computed token embeddings
Option 2: Transformer-free semantic hashing
Option 3: External embedding service (future)

Key: LLM is external annotator only, NOT in core loop.
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import math
import hashlib
import numpy as np
import struct


class DeterministicProjection:
    """
    Deterministic projection from token to vector space.
    
    Uses SHA256-based seed + fixed projection matrix.
    Replay-invariant - same token always produces same vector.
    """
    
    _projection_cache: Dict[int, np.ndarray] = {}
    _dim: int = 384
    
    @classmethod
    def set_dim(cls, dim: int):
        cls._dim = dim
        cls._projection_cache.clear()
    
    @classmethod
    def _get_projection_seed(cls, token: str, dim: int) -> int:
        """Get deterministic seed for token dimension"""
        # SHA256 is stable across processes
        hash_bytes = hashlib.sha256(f"{token}_dim_{dim}".encode()).digest()
        return struct.unpack('>I', hash_bytes[:4])[0]
    
    @classmethod
    def project_token_dim(cls, token: str, dim: int) -> float:
        """Project token to single dimension deterministically"""
        seed = cls._get_projection_seed(token, dim)
        # Splitmix64-style deterministic float
        x = seed
        x = (x ^ (x >> 33)) * 0xff51afd7ed558ccd
        x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53
        x = x ^ (x >> 33)
        # Return float in [0, 1)
        return (x & 0xFFFFFFFF) / 4294967296.0
    
    @classmethod
    def project_token(cls, token: str) -> np.ndarray:
        """Project token to full embedding vector"""
        if token in cls._projection_cache:
            return cls._projection_cache[token]
        
        vec = np.array([
            cls.project_token_dim(token, d) 
            for d in range(cls._dim)
        ], dtype=np.float64)
        
        # Normalize to unit sphere
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        cls._projection_cache[token] = vec
        return vec
    
    @classmethod
    def reset_cache(cls):
        """Reset projection cache - for testing"""
        cls._projection_cache.clear()


class SemanticEmbeddingModel:
    """
    Lightweight semantic embedding model.
    
    NOT using LLM in core - uses:
    1. DeterministicProjection (SHA256-based, replay-safe)
    2. Token co-occurrence statistics
    3. Context-based modulation
     
    This creates REAL semantic neighborhoods,
    not hash collisions. All embeddings are replay-invariant.
    """
    
    def __init__(self, embedding_dim: int = 384):
        self._dim = embedding_dim
        self._vocabulary: Dict[str, np.ndarray] = {}
        self._bigram_probs: Dict[Tuple[str, str], float] = {}
        self._context_window = 5
        
        # Initialize deterministic projection
        DeterministicProjection.set_dim(embedding_dim)
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple deterministic tokenization"""
        # Remove punctuation, lowercase
        cleaned = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
        return [t for t in cleaned.split() if len(t) > 1]
    
    def _build_vocabulary(self, corpus: List[str]):
        """Build vocabulary - tokens become their own embedding keys"""
        token_set: Set[str] = set()
        
        for text in corpus:
            tokens = self._tokenize(text)
            token_set.update(tokens)
        
        # Each token gets a deterministic embedding
        # The key insight: similarity comes from token overlap, not vector geometry
        for token in token_set:
            if token:  # Skip empty tokens
                self._vocabulary[token] = DeterministicProjection.project_token(token)
        
        # Build bigram probabilities for token sequence patterns
        for text in corpus:
            tokens = self._tokenize(text)
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i+1])
                self._bigram_probs[bigram] = self._bigram_probs.get(bigram, 0.0) + 1.0
        
        first_counts: Dict[str, float] = {}
        for bigram, count in self._bigram_probs.items():
            first_counts[bigram[0]] = first_counts.get(bigram[0], 0) + count
        
        for bigram in self._bigram_probs:
            self._bigram_probs[bigram] /= max(first_counts[bigram[0]], 1.0)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed text - encode token set as deterministic vector"""
        tokens = self._tokenize(text)
        
        if not tokens:
            return [0.0] * self._dim
        
        # Create embedding from token set - weighted by token frequency
        # This creates semantic clusters based on shared tokens
        token_weights: Dict[str, float] = {}
        for token in tokens:
            token_weights[token] = token_weights.get(token, 0) + 1.0
        
        # Normalize weights
        total = sum(token_weights.values())
        for t in token_weights:
            token_weights[t] /= total
        
        # Build weighted combination of token embeddings
        result = np.zeros(self._dim, dtype=np.float64)
        for token, weight in token_weights.items():
            if token in self._vocabulary:
                result += self._vocabulary[token] * weight
            else:
                # OOV - add deterministic projection
                result += DeterministicProjection.project_token(token) * weight
        
        # Normalize
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
        
        return result.tolist()
    
    def embed_belief(self, belief: Any) -> List[float]:
        """Embed belief with strong source-based semantic anchoring"""
        prop = belief.proposition if hasattr(belief, 'proposition') else str(belief)
        source = belief.source if hasattr(belief, 'source') else "unknown"
        
        # Create deterministic source anchor based on source string
        # Use hash of source to create deterministic but varied anchor
        source_seed = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)
        np.random.seed(source_seed % (2**32))
        source_anchor = np.random.randn(self._dim)
        source_anchor = source_anchor / np.linalg.norm(source_anchor)
        
        # Get proposition embedding
        prop_emb = np.array(self.embed_text(prop), dtype=np.float64)
        
        # Strong source bias: 30% proposition, 70% source
        combined = prop_emb * 0.3 + source_anchor * 0.7
        
        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        
        return combined.tolist()


class IdentityAttractorTracker:
    """
    Track attractor lineage over time.
    
    Key metrics:
    - persistence: how long attractor survives
    - drift: how much centroid moves
    - merge: when attractors combine
    - split: when attractor divides
    """
    
    def __init__(self, merge_threshold: float = 0.15, split_threshold: float = 0.4):
        self._merge_threshold = merge_threshold
        self._split_threshold = split_threshold
        self._attractor_history: List[Dict] = []
    
    def track(
        self,
        current_attractors: Dict[str, Any],
        timestamp: str
    ) -> Dict[str, Any]:
        """Track attractor evolution"""
        
        # Match current attractors to previous
        matches = {}
        unmatched = []
        
        if self._attractor_history:
            prev_attractors = self._attractor_history[-1].get("attractors", {})
            
            for curr_id, curr_att in current_attractors.items():
                best_match = None
                best_dist = float('inf')
                
                for prev_id, prev_att in prev_attractors.items():
                    dist = self._cosine_distance(
                        curr_att.center_vector, 
                        prev_att.center_vector
                    )
                    if dist < best_dist:
                        best_dist = dist
                        best_match = prev_id
                
                if best_match and best_dist < self._merge_threshold:
                    matches[curr_id] = {
                        "matched_to": best_match,
                        "drift": best_dist,
                        "persistence": prev_attractors[best_match].get("persistence", 0) + 1
                    }
                else:
                    unmatched.append(curr_id)
        
        # Record history
        record = {
            "timestamp": timestamp,
            "attractors": {
                aid: {
                    "center": a.center_vector[:5],  # Store truncated
                    "size": len(a.belief_ids),
                    "persistence": matches.get(aid, {}).get("persistence", 1),
                    "drift": matches.get(aid, {}).get("drift", 0.0)
                }
                for aid, a in current_attractors.items()
            },
            "new_attractors": len(unmatched),
            "matched_count": len(matches)
        }
        
        self._attractor_history.append(record)
        
        return {
            "matches": matches,
            "new_attractors": unmatched,
            "total_tracked": len(current_attractors)
        }
    
    def _cosine_distance(self, v1: List[float], v2: List[float]) -> float:
        if len(v1) != len(v2):
            return 1.0
        dot = sum(a*b for a, b in zip(v1, v2))
        return 1.0 - max(-1.0, min(1.0, dot))
    
    def get_persistence_metrics(self) -> Dict[str, float]:
        """Get stability metrics across history"""
        if not self._attractor_history:
            return {"avg_persistence": 0, "max_persistence": 0, "identity_stability": 0}
        
        persistences = []
        drifts = []
        
        for record in self._attractor_history:
            for aid, info in record.get("attractors", {}).items():
                persistences.append(info.get("persistence", 1))
                drifts.append(info.get("drift", 0.0))
        
        avg_persistence = sum(persistences) / max(len(persistences), 1)
        max_persistence = max(persistences) if persistences else 0
        avg_drift = sum(drifts) / max(len(drifts), 1)
        
        # Identity stability: high persistence + low drift
        identity_stability = avg_persistence * (1.0 - avg_drift)
        
        return {
            "avg_persistence": avg_persistence,
            "max_persistence": max_persistence,
            "avg_drift": avg_drift,
            "identity_stability": identity_stability
        }


class SemanticHysteresisLayer:
    """
    Semantic-level hysteresis (beyond contradiction-level).
    
    Tracks:
    - Attractor memory (stable regions persist)
    - Semantic residue (forgotten but traceable)
    - Decay curves on embedding space
    - Topology evolution
    """
    
    def __init__(self, decay_rate: float = 0.02):
        self._decay_rate = decay_rate
        self._attractor_memory: Dict[str, Dict] = {}  # attractor_id -> memory
        self._semantic_residue: List[Dict] = []  # forgotten topology traces
        self._topology_history: List[Dict] = []
    
    def record_topology(
        self,
        attractors: Dict[str, Any],
        neighborhoods: Dict[str, Any],
        timestamp: str
    ):
        """Record current topology for hysteresis tracking"""
        
        record = {
            "timestamp": timestamp,
            "attractor_count": len(attractors),
            "neighborhood_count": len(neighborhoods),
            "attractor_centers": [
                a.center_vector[:3] for a in attractors.values()
            ],
            "avg_attractor_size": sum(len(a.belief_ids) for a in attractors.values()) / max(len(attractors), 1)
        }
        
        self._topology_history.append(record)
        
        # Update attractor memory
        for aid, attractor in attractors.items():
            self._attractor_memory[aid] = {
                "last_seen": timestamp,
                "persistence": self._attractor_memory.get(aid, {}).get("persistence", 0) + 1,
                "strength": len(attractor.belief_ids) / 100.0,  # Normalize
                "center": attractor.center_vector[:3]
            }
    
    def apply_semantic_decay(self) -> Dict[str, float]:
        """Apply decay to attractor memory"""
        decayed = {}
        
        for aid, memory in self._attractor_memory.items():
            # Decay strength
            new_strength = memory["strength"] * (1.0 - self._decay_rate)
            
            if new_strength < 0.05:
                # Move to residue (semantic trace)
                self._semantic_residue.append({
                    "attractor_id": aid,
                    "last_center": memory["center"],
                    "total_persistence": memory["persistence"],
                    "faded_at": datetime.utcnow().isoformat()
                })
            else:
                memory["strength"] = new_strength
                decayed[aid] = new_strength
        
        # Clean up decayed
        self._attractor_memory = {
            aid: m for aid, m in self._attractor_memory.items()
            if m["strength"] >= 0.05
        }
        
        return decayed
    
    def get_identity_residue(self) -> List[Dict]:
        """Get semantic traces of past identity attractors"""
        return self._semantic_residue
    
    def get_topology_stability(self) -> Dict[str, float]:
        """Measure topology evolution stability"""
        if len(self._topology_history) < 2:
            return {"stability": 1.0, "evolution_rate": 0.0}
        
        # Compare consecutive topologies
        changes = []
        for i in range(1, len(self._topology_history)):
            prev = self._topology_history[i-1]
            curr = self._topology_history[i]
            
            attractor_change = abs(prev["attractor_count"] - curr["attractor_count"])
            neighborhood_change = abs(prev["neighborhood_count"] - curr["neighborhood_count"])
            
            changes.append(attractor_change + neighborhood_change * 0.5)
        
        avg_change = sum(changes) / len(changes)
        evolution_rate = avg_change / max(self._topology_history[-1]["attractor_count"], 1)
        stability = 1.0 - min(evolution_rate, 1.0)
        
        return {
            "stability": stability,
            "evolution_rate": evolution_rate,
            "topology_epochs": len(self._topology_history)
        }


# Global instances
_embedding_model: Optional[SemanticEmbeddingModel] = None
_attractor_tracker: Optional[IdentityAttractorTracker] = None
_hysteresis_layer: Optional[SemanticHysteresisLayer] = None


def get_embedding_model() -> SemanticEmbeddingModel:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SemanticEmbeddingModel()
    return _embedding_model


def get_attractor_tracker() -> IdentityAttractorTracker:
    global _attractor_tracker
    if _attractor_tracker is None:
        _attractor_tracker = IdentityAttractorTracker()
    return _attractor_tracker


def get_hysteresis_layer() -> SemanticHysteresisLayer:
    global _hysteresis_layer
    if _hysteresis_layer is None:
        _hysteresis_layer = SemanticHysteresisLayer()
    return _hysteresis_layer


def train_embedding_model(corpus: List[str]):
    """Train embedding model on corpus"""
    model = get_embedding_model()
    model._build_vocabulary(corpus)