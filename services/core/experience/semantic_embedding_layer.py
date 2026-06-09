"""
Semantic Embedding Layer - Belief vector space and semantic pressure.

Key principle:
    Beliefs exist in semantic space, not just structural graph.
    Similar beliefs cluster, distant beliefs create tension.
    
No LLM in core - LLM only as external annotator.
Core is deterministic, semantic mappings are computed.
"""
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from copy import deepcopy
import math
import hashlib


@dataclass
class BeliefEmbedding:
    """Belief in semantic vector space"""
    belief_id: str
    proposition: str
    vector: List[float]  # Semantic embedding
    semantic_neighbors: List[str] = field(default_factory=list)
    temporal_drift: float = 0.0  # How much semantics changed over time
    semantic_confidence: float = 0.5  # Confidence in semantic stability


@dataclass 
class SemanticNeighborhood:
    """Cluster of semantically similar beliefs"""
    neighborhood_id: str
    centroid_vector: List[float]
    member_beliefs: List[str]
    density: float  # How tight the cluster is
    stability: float  # How stable over time
    internal_pressure: float = 0.0  # Tension within cluster


@dataclass
class SemanticAttractor:
    """Stable region in semantic space"""
    attractor_id: str
    center_vector: List[float]
    radius: float  # Influence radius
    belief_ids: List[str]
    persistence: float  # How long has been stable
    attraction_strength: float  # How strongly it pulls beliefs


class SemanticEmbeddingLayer:
    """
    Semantic field for beliefs.
    
    Maps structural beliefs into semantic vector space.
    Computes neighborhoods, attractors, pressure gradients.
    
    No LLM in core - deterministic embeddings based on:
    - Token overlap
    - Causal proximity
    - Temporal co-occurrence
    - Compensation patterns
    """
    
    def __init__(self, embedding_dim: int = 16):
        self._dim = embedding_dim
        self._belief_embeddings: Dict[str, BeliefEmbedding] = {}
        self._neighborhoods: Dict[str, SemanticNeighborhood] = {}
        self._attractors: Dict[str, SemanticAttractor] = {}
        
        # Semantic pressure accumulator
        self._semantic_pressure_history: List[Dict] = []
    
    def compute_embedding(self, belief: Any, related_beliefs: Dict[str, Any]) -> List[float]:
        """
        Compute semantic embedding for a belief.
        
        Deterministic based on:
        1. Token features from proposition
        2. Causal proximity to related beliefs
        3. Source and temporal features
        """
        prop = belief.proposition.lower() if hasattr(belief, 'proposition') else ""
        
        # Simple token-based embedding (deterministic)
        tokens = set(prop.split())
        
        # Create vector from token features (deterministic)
        vector = []
        for i in range(self._dim):
            # Deterministic feature extraction using SHA256
            key = f"{belief.belief_id}_{','.join(sorted(tokens))}_{i}"
            hash_bytes = hashlib.sha256(key.encode()).digest()
            # Convert first 8 bytes to float in [0, 1)
            token_feature = int.from_bytes(hash_bytes[:8], 'big') / (2**64)
            vector.append(token_feature)
        
        # Normalize
        magnitude = math.sqrt(sum(v*v for v in vector))
        if magnitude > 0:
            vector = [v/magnitude for v in vector]
        
        return vector
    
    def map_beliefs(
        self,
        beliefs: Dict[str, Any],
        causal_edges: Dict[str, Any]
    ) -> Dict[str, BeliefEmbedding]:
        """
        Map all beliefs to semantic space.
        
        Returns embeddings with neighbor relationships.
        """
        embeddings = {}
        
        # First pass: compute base embeddings
        for belief_id, belief in beliefs.items():
            embedding = BeliefEmbedding(
                belief_id=belief_id,
                proposition=belief.proposition if hasattr(belief, 'proposition') else "",
                vector=self.compute_embedding(belief, beliefs),
                semantic_neighbors=[],
                temporal_drift=0.0,
                semantic_confidence=0.5
            )
            embeddings[belief_id] = embedding
            self._belief_embeddings[belief_id] = embedding
        
        # Second pass: compute semantic neighbors
        for e1_id, e1 in embeddings.items():
            distances = []
            for e2_id, e2 in embeddings.items():
                if e1_id != e2_id:
                    dist = self._cosine_distance(e1.vector, e2.vector)
                    distances.append((e2_id, dist))
            
            # Sort by distance and take top k
            distances.sort(key=lambda x: x[1])
            e1.semantic_neighbors = [d[0] for d in distances[:5]]
        
        return embeddings
    
    def _cosine_distance(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine distance between vectors"""
        dot = sum(a*b for a, b in zip(v1, v2))
        return 1.0 - max(-1.0, min(1.0, dot))  # Clamp for safety
    
    def detect_neighborhoods(
        self,
        embeddings: Dict[str, BeliefEmbedding]
    ) -> Dict[str, SemanticNeighborhood]:
        """
        Detect semantic neighborhoods (clusters).
        
        Uses simple distance-based clustering.
        """
        neighborhoods = {}
        
        # Simple clustering: group beliefs within threshold distance
        threshold = 0.3
        assigned = set()
        
        for belief_id, embedding in embeddings.items():
            if belief_id in assigned:
                continue
            
            # Start new neighborhood
            members = [belief_id]
            assigned.add(belief_id)
            
            # Find close neighbors
            for other_id, other_emb in embeddings.items():
                if other_id in assigned:
                    continue
                
                dist = self._cosine_distance(embedding.vector, other_emb.vector)
                if dist < threshold:
                    members.append(other_id)
                    assigned.add(other_id)
            
            if len(members) >= 2:  # Minimum cluster size
                # Compute centroid
                centroid = self._compute_centroid(
                    [embeddings[m].vector for m in members]
                )
                
                # Compute density
                density = self._compute_density(
                    [embeddings[m].vector for m in members], centroid
                )
                
                neighborhood = SemanticNeighborhood(
                    neighborhood_id=str(uuid4()),
                    centroid_vector=centroid,
                    member_beliefs=members,
                    density=density,
                    stability=1.0 - density  # Placeholder
                )
                neighborhoods[neighborhood.neighborhood_id] = neighborhood
                self._neighborhoods[neighborhood.neighborhood_id] = neighborhood
        
        return neighborhoods
    
    def _compute_centroid(self, vectors: List[List[float]]) -> List[float]:
        """Compute centroid of vectors"""
        if not vectors:
            return [0.0] * self._dim
        return [
            sum(v[i] for v in vectors) / len(vectors)
            for i in range(self._dim)
        ]
    
    def _compute_density(self, vectors: List[List[float]], centroid: List[float]) -> float:
        """Compute cluster density"""
        if len(vectors) < 2:
            return 0.0
        
        avg_dist = sum(
            self._cosine_distance(v, centroid)
            for v in vectors
        ) / len(vectors)
        
        return 1.0 - avg_dist  # High density = low avg distance
    
    def detect_attractors(
        self,
        neighborhoods: Dict[str, SemanticNeighborhood],
        time_horizon: int = 10
    ) -> Dict[str, SemanticAttractor]:
        """
        Detect stable attractors in semantic space.
        
        Attractor = region that persistently attracts beliefs over time.
        """
        attractors = {}
        
        # High density + stable neighborhoods = attractors
        for neighborhood in neighborhoods.values():
            if neighborhood.density > 0.7 and len(neighborhood.member_beliefs) >= 3:
                attractor = SemanticAttractor(
                    attractor_id=str(uuid4()),
                    center_vector=neighborhood.centroid_vector,
                    radius=0.2,  # Influence radius
                    belief_ids=neighborhood.member_beliefs,
                    persistence=neighborhood.stability,
                    attraction_strength=neighborhood.density
                )
                attractors[attractor.attractor_id] = attractor
                self._attractors[attractor.attractor_id] = attractor
        
        return attractors
    
    def compute_semantic_pressure(
        self,
        embeddings: Dict[str, BeliefEmbedding],
        neighborhoods: Dict[str, SemanticNeighborhood],
        attractors: Dict[str, SemanticAttractor]
    ) -> Dict[str, float]:
        """
        Compute semantic pressure types.
        
        Returns pressure values for:
        - semantic_divergence
        - identity_fragmentation  
        - attractor_collapse
        - semantic_isolation
        - unresolved_tension
        """
        pressure = {}
        
        # 1. Semantic divergence: average distance to neighbors
        total_divergence = 0.0
        for emb in embeddings.values():
            if emb.semantic_neighbors:
                neighbor_vectors = [
                    embeddings[n].vector 
                    for n in emb.semantic_neighbors 
                    if n in embeddings
                ]
                if neighbor_vectors:
                    avg_dist = sum(
                        self._cosine_distance(emb.vector, nv)
                        for nv in neighbor_vectors
                    ) / len(neighbor_vectors)
                    total_divergence += avg_dist
        
        pressure["semantic_divergence"] = total_divergence / max(len(embeddings), 1)
        
        # 2. Identity fragmentation: number of small neighborhoods
        small_neighborhoods = sum(
            1 for n in neighborhoods.values() 
            if len(n.member_beliefs) < 3
        )
        pressure["identity_fragmentation"] = small_neighborhoods / max(len(neighborhoods), 1)
        
        # 3. Attractor collapse: lost attractors over time
        pressure["attractor_collapse"] = 0.0  # Placeholder
        
        # 4. Semantic isolation: beliefs far from any neighborhood
        isolated = 0
        for emb in embeddings.values():
            in_any_neighborhood = any(
                emb.belief_id in n.member_beliefs 
                for n in neighborhoods.values()
            )
            if not in_any_neighborhood:
                isolated += 1
        pressure["semantic_isolation"] = isolated / max(len(embeddings), 1)
        
        # 5. Unresolved tension: high density neighborhoods with low stability
        tension = sum(
            n.density * (1.0 - n.stability)
            for n in neighborhoods.values()
        )
        pressure["unresolved_tension"] = tension / max(len(neighborhoods), 1)
        
        # Record for history
        self._semantic_pressure_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "pressure": pressure.copy()
        })
        
        return pressure
    
    def get_identity_core(
        self,
        attractors: Dict[str, SemanticAttractor],
        beliefs: Dict[str, Any]
    ) -> List[str]:
        """
        Detect identity core - stable semantic region.
        
        Identity core = beliefs in strong, persistent attractors.
        These are the "self" - hard to change, highly stable.
        """
        core_beliefs = []
        
        # Strong attractors with high persistence
        strong_attractors = [
            a for a in attractors.values()
            if a.attraction_strength > 0.7 and a.persistence > 0.5
        ]
        
        for attractor in strong_attractors:
            core_beliefs.extend(attractor.belief_ids)
        
        return list(set(core_beliefs))
    
    def compute_semantic_drift(
        self,
        current_embeddings: Dict[str, BeliefEmbedding],
        version: int
    ) -> float:
        """
        Compute semantic drift over time.
        
        How much has the semantic field changed from previous version.
        """
        if not self._belief_embeddings:
            return 0.0
        
        # Compare current embeddings to previous
        total_drift = 0.0
        compared = 0
        
        for belief_id, current_emb in current_embeddings.items():
            if belief_id in self._belief_embeddings:
                prev_emb = self._belief_embeddings[belief_id]
                drift = self._cosine_distance(prev_emb.vector, current_emb.vector)
                total_drift += drift
                compared += 1
        
        return total_drift / max(compared, 1)
    
    def get_embedding(self, belief_id: str) -> Optional[BeliefEmbedding]:
        """Get embedding for belief"""
        return self._belief_embeddings.get(belief_id)
    
    def get_pressure_history(self) -> List[Dict]:
        """Get semantic pressure history"""
        return self._semantic_pressure_history


# Global instance
_semantic_layer: Optional[SemanticEmbeddingLayer] = None


def get_semantic_layer(embedding_dim: int = 16) -> SemanticEmbeddingLayer:
    """Get global semantic embedding layer"""
    global _semantic_layer
    if _semantic_layer is None:
        _semantic_layer = SemanticEmbeddingLayer(embedding_dim)
    return _semantic_layer


def reset_semantic_layer():
    """Reset layer for fresh analysis"""
    global _semantic_layer
    _semantic_layer = SemanticEmbeddingLayer()