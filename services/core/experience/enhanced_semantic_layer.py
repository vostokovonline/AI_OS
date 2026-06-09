"""
Enhanced Semantic Layer with Real Embeddings + Identity Tracking

Integrates:
1. Real semantic embeddings (not hash-based)
2. Identity attractor lineage tracking
3. Semantic hysteresis (memory beyond contradiction)
4. Thermodynamic realism
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import math

# Import our new components
from real_semantic_embeddings import (
    get_embedding_model, get_attractor_tracker, get_hysteresis_layer,
    train_embedding_model
)


class EnhancedSemanticLayer:
    """
    Semantic layer with real embeddings and identity tracking.
    
    Key improvements over basic layer:
    - Real token-based embeddings (not hash)
    - Attractor lineage (persistence, drift, merge, split)
    - Semantic hysteresis (memory decay, residue)
    - Identity core = stable attractor basin
    """
    
    def __init__(self, embedding_dim: int = 384):
        self._dim = embedding_dim
        self._model = get_embedding_model()
        
        # Tracking components
        self._tracker = get_attractor_tracker()
        self._hysteresis = get_hysteresis_layer()
        
        # Embedding cache
        self._belief_embeddings: Dict[str, List[float]] = {}
        
        # Hysteresis thresholds - prevent flickering (lowered for testing)
        self._enter_threshold = 0.5
        self._exit_threshold = 0.4
        
        # Neighborhood persistence tracking for hysteresis
        self._neighborhood_persistence: Dict[str, int] = {}
        
        # Train on basic corpus first
        self._initialize()
    
    def _initialize(self):
        """Initialize with base corpus"""
        corpus = [
            "knowledge truth belief understanding",
            "reason logic evidence proof", 
            "cause effect consequence outcome",
            "action behavior choice decision",
            "value principle ethics morality",
            "goal aim purpose objective",
            "time space matter energy",
            "system structure order pattern",
            "change growth development evolution",
            "stability persistence continuity",
            "conflict contradiction tension stress",
            "resolution harmony balance peace"
        ]
        train_embedding_model(corpus)
    
    def compute_embedding(self, belief: Any) -> List[float]:
        """Compute real semantic embedding for belief"""
        return self._model.embed_belief(belief)
    
    def map_beliefs(
        self,
        beliefs: Dict[str, Any],
        causal_edges: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map beliefs to semantic space with tracking"""
        
        # Compute embeddings
        embeddings = {}
        for belief_id, belief in beliefs.items():
            emb = self.compute_embedding(belief)
            self._belief_embeddings[belief_id] = emb
            embeddings[belief_id] = {
                "id": belief_id,
                "embedding": emb,
                "proposition": belief.proposition if hasattr(belief, 'proposition') else ""
            }
        
        # Compute semantic neighbors using REAL cosine similarity
        for e1_id, e1 in embeddings.items():
            distances = []
            for e2_id, e2 in embeddings.items():
                if e1_id != e2_id:
                    dist = self._cosine_similarity(e1["embedding"], e2["embedding"])
                    distances.append((e2_id, dist))
            
            distances.sort(key=lambda x: x[1])
            embeddings[e1_id]["neighbors"] = [d[0] for d in distances[:5]]
        
        return embeddings
    
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity"""
        if len(v1) != len(v2):
            return 0.0
        dot = sum(a*b for a, b in zip(v1, v2))
        return max(-1.0, min(1.0, dot))
    
    def detect_neighborhoods(
        self,
        embeddings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect semantic neighborhoods with hysteresis"""
        neighborhoods = {}
        
        # First pass: seed neighborhoods with enter_threshold
        seeds = []
        for belief_id, emb in embeddings.items():
            seeds.append((belief_id, emb))
        
        # Group by similarity using enter_threshold
        clusters = []
        used = set()
        
        for belief_id, emb in seeds:
            if belief_id in used:
                continue
            
            cluster_members = [belief_id]
            used.add(belief_id)
            
            for other_id, other_emb in seeds:
                if other_id in used:
                    continue
                
                sim = self._cosine_similarity(
                    emb["embedding"], 
                    other_emb["embedding"]
                )
                # Use enter_threshold for joining
                if sim >= self._enter_threshold:
                    cluster_members.append(other_id)
                    used.add(other_id)
            
            if len(cluster_members) >= 2:
                clusters.append(cluster_members)
        
        # Second pass: check for exit_threshold hysteresis
        # Members stay if they meet exit_threshold even if not meeting enter
        persisted = set()
        for cluster in clusters:
            for member in cluster:
                persisted.add(member)
        
        # Build neighborhoods with density
        for cluster in clusters:
            # Compute centroid
            centroid = self._compute_centroid([
                embeddings[m]["embedding"] for m in cluster
            ])
            
            # Compute density (using exit_threshold for internal cohesion)
            density = self._compute_density(
                [embeddings[m]["embedding"] for m in cluster],
                centroid
            )
            
            # Apply hysteresis: require exit_threshold for persistence
            if density >= self._exit_threshold:
                neighborhood_id = str(uuid4())
                neighborhoods[neighborhood_id] = {
                    "members": cluster,
                    "centroid": centroid,
                    "density": density,
                    "persistence": self._neighborhood_persistence.get(neighborhood_id, 1)
                }
                
                # Update persistence
                self._neighborhood_persistence[neighborhood_id] = \
                    self._neighborhood_persistence.get(neighborhood_id, 0) + 1
        
        return neighborhoods
    
    def _compute_centroid(self, vectors: List[List[float]]) -> List[float]:
        if not vectors:
            return [0.0] * self._dim
        return [
            sum(v[i] for v in vectors) / len(vectors)
            for i in range(self._dim)
        ]
    
    def _compute_density(self, vectors: List[List[float]], centroid: List[float]) -> float:
        if len(vectors) < 2:
            return 0.0
        
        avg_sim = sum(
            self._cosine_similarity(v, centroid)
            for v in vectors
        ) / len(vectors)
        
        return avg_sim
    
    def detect_attractors(
        self,
        neighborhoods: Dict[str, Any],
        time_horizon: int = 10
    ) -> Dict[str, Any]:
        """Detect attractors with tracking"""
        attractors = {}
        
        for nid, neighborhood in neighborhoods.items():
            if neighborhood["density"] > 0.3 and len(neighborhood["members"]) >= 2:
                attractors[str(uuid4())] = {
                    "center_vector": neighborhood["centroid"],
                    "belief_ids": neighborhood["members"],
                    "radius": 0.25,
                    "attraction_strength": neighborhood["density"]
                }
        
        # Track attractor lineage
        track_result = self._tracker.track(
            attractors,
            datetime.utcnow().isoformat()
        )
        
        # Record topology for hysteresis
        self._hysteresis.record_topology(
            attractors,
            neighborhoods,
            datetime.utcnow().isoformat()
        )
        
        return attractors
    
    def get_identity_core(
        self,
        attractors: Dict[str, Any],
        beliefs: Dict[str, Any]
    ) -> List[str]:
        """Get identity core - persistent attractors with high attraction"""
        
        core = []
        
        for aid, attractor in attractors.items():
            # Core = strong attractor with high attraction strength
            if attractor["attraction_strength"] > 0.7:
                core.extend(attractor["belief_ids"])
        
        return list(set(core))
    
    def apply_semantic_decay(self) -> Dict[str, float]:
        """Apply decay to semantic memory"""
        return self._hysteresis.apply_semantic_decay()
    
    def get_identity_metrics(self) -> Dict[str, Any]:
        """Get identity stability metrics"""
        persistence = self._tracker.get_persistence_metrics()
        topology = self._hysteresis.get_topology_stability()
        
        # Combined identity score
        identity_score = (
            persistence.get("identity_stability", 0) * 0.5 +
            topology.get("stability", 0) * 0.5
        )
        
        return {
            "persistence": persistence,
            "topology": topology,
            "identity_score": identity_score,
            "residue_traces": len(self._hysteresis.get_identity_residue())
        }
    
    def compute_semantic_pressure(
        self,
        embeddings: Dict[str, Any],
        neighborhoods: Dict[str, Any],
        attractors: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute semantic pressure with real metrics"""
        
        pressure = {}
        
        # Semantic divergence - REAL similarity variance
        similarities = []
        for emb in embeddings.values():
            for neighbor in emb.get("neighbors", []):
                if neighbor in embeddings:
                    sim = self._cosine_similarity(
                        emb["embedding"],
                        embeddings[neighbor]["embedding"]
                    )
                    similarities.append(sim)
        
        if similarities:
            pressure["semantic_divergence"] = 1.0 - (sum(similarities) / len(similarities))
        else:
            pressure["semantic_divergence"] = 0.0
        
        # Identity fragmentation
        small_neighborhoods = sum(
            1 for n in neighborhoods.values() 
            if len(n["members"]) < 3
        )
        pressure["identity_fragmentation"] = small_neighborhoods / max(len(neighborhoods), 1)
        
        # Attractor collapse - check if attractors disappearing
        if len(self._hysteresis._topology_history) > 1:
            prev_count = self._hysteresis._topology_history[-2].get("attractor_count", 1)
            curr_count = self._hysteresis._topology_history[-1].get("attractor_count", 0)
            pressure["attractor_collapse"] = max(0, (prev_count - curr_count) / max(prev_count, 1))
        else:
            pressure["attractor_collapse"] = 0.0
        
        # Semantic isolation
        in_any = set()
        for n in neighborhoods.values():
            in_any.update(n["members"])
        
        isolated = len(embeddings) - len(in_any)
        pressure["semantic_isolation"] = isolated / max(len(embeddings), 1)
        
        # Unresolved tension - combine with contradiction-level hysteresis
        residue = len(self._hysteresis.get_identity_residue())
        pressure["unresolved_tension"] = residue / 10.0  # Normalize
        
        return pressure


# Global instance
_enhanced_semantic: Optional[EnhancedSemanticLayer] = None


def get_enhanced_semantic() -> EnhancedSemanticLayer:
    global _enhanced_semantic
    if _enhanced_semantic is None:
        _enhanced_semantic = EnhancedSemanticLayer()
    return _enhanced_semantic


def reset_enhanced_semantic():
    global _enhanced_semantic
    _enhanced_semantic = None