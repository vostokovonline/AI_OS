"""
Advanced Clustering Options - Beyond K-means

K-means is good for bootstrap, but cognition has:
- Manifold structure (not spherical clusters)
- Metastable regions (not fixed clusters)
- Branching flows (not static attractors)
- Dynamic motifs (not static centroids)

These clustering methods handle these properties better.

HDBSCAN: Density-based, handles manifold structure
Spectral Clustering: Graph-based, handles non-convex shapes
Diffusion Maps: Manifold learning, handles branching flows
"""
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for clustering algorithm"""
    algorithm: str = "kmeans"  # kmeans, hdbscan, spectral, diffusion
    min_cluster_size: int = 3
    min_samples: int = 2
    cluster_selection_epsilon: float = 0.0
    sensitivity: float = 1.0  # HDBSCAN sensitivity


class AdvancedClusterer:
    """
    Advanced clustering with multiple algorithms.
    
    Usage:
        clusterer = AdvancedClusterer(config=ClusterConfig(algorithm="hdbscan"))
        clusters = clusterer.fit(embeddings)
    """
    
    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig()
        self.fitted = False
        self.cluster_labels: List[int] = []
        self.cluster_centers: List[List[float]] = []
        self.probabilities: List[float] = []  # Cluster membership probabilities
        
        logger.info("advanced_clusterer_initialized", algorithm=self.config.algorithm)
    
    def fit(self, embeddings: List[List[float]]) -> List[List[int]]:
        """Fit clustering model"""
        if not embeddings:
            return []
        
        if self.config.algorithm == "hdbscan":
            self.cluster_labels, self.probabilities = self._hdbscan(embeddings)
        elif self.config.algorithm == "spectral":
            self.cluster_labels, self.cluster_centers = self._spectral(embeddings)
        elif self.config.algorithm == "diffusion":
            self.cluster_labels, self.cluster_centers = self._diffusion_maps(embeddings)
        else:
            self.cluster_labels, self.cluster_centers = self._kmeans(embeddings)
        
        self.fitted = True
        return self.cluster_labels
    
    def _compute_distance_matrix(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Compute pairwise distance matrix"""
        n = len(embeddings)
        distances = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt(sum(
                    (a - b) ** 2 for a, b in zip(embeddings[i], embeddings[j])
                ))
                distances[i][j] = dist
                distances[j][i] = dist
        
        return distances
    
    def _hdbscan(self, embeddings: List[List[float]]) -> Tuple[List[int], List[float]]:
        """
        HDBSCAN clustering - hierarchical density-based.
        
        Advantages:
        - Variable density clusters
        - No need to specify number of clusters
        - Handles manifold structure
        - Produces cluster hierarchy
        """
        n = len(embeddings)
        
        if n < self.config.min_cluster_size:
            return list(range(n)), [1.0] * n
        
        distances = self._compute_distance_matrix(embeddings)
        
        core_distances = self._compute_core_distances(distances)
        
        mst = self._build_mst(distances, core_distances)
        
        hierarchy, clusters = self._build_hierarchy(mst)
        
        labels = self._extract_clusters(hierarchy, clusters)
        probabilities = self._compute_outlier_scores(distances, labels)
        
        return labels, probabilities
    
    def _compute_core_distances(
        self,
        distances: List[List[float]]
    ) -> List[float]:
        """Compute core distance for each point"""
        n = len(distances)
        k = self.config.min_samples
        
        core_distances = []
        for i in range(n):
            sorted_dists = sorted(distances[i])
            core_dist = sorted_dists[min(k, len(sorted_dists) - 1)]
            core_distances.append(core_dist)
        
        return core_distances
    
    def _build_mst(
        self,
        distances: List[List[float]],
        core_distances: List[float]
    ) -> List[Tuple[int, int, float]]:
        """Build minimum spanning tree using Prim's algorithm"""
        n = len(distances)
        in_tree = [False] * n
        edges = []
        
        in_tree[0] = True
        heap = []
        
        for j in range(1, n):
            dist = distances[0][j]
            heap.append((dist, 0, j))
        
        while len(edges) < n - 1 and heap:
            heap.sort(key=lambda x: x[0])
            dist, i, j = heap.pop(0)
            
            if in_tree[j]:
                continue
            
            edges.append((i, j, dist))
            in_tree[j] = True
            
            for k in range(n):
                if not in_tree[k]:
                    mutual_reach = max(distances[j][k], core_distances[j])
                    heap.append((mutual_reach, j, k))
        
        return edges
    
    def _build_hierarchy(
        self,
        mst: List[Tuple[int, int, float]]
    ) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
        """Build cluster hierarchy from MST"""
        sorted_edges = sorted(mst, key=lambda x: x[2])
        
        parent = list(range(len(mst) + 1))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        hierarchy = []
        clusters = []
        
        for i, j, dist in sorted_edges:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj
                hierarchy.append((pi, pj, dist))
        
        return hierarchy, clusters
    
    def _extract_clusters(
        self,
        hierarchy: List[Tuple[int, int, float]],
        clusters: List[List[int]]
    ) -> List[int]:
        """Extract flat clusters from hierarchy"""
        n = len(hierarchy) + 1 if hierarchy else 1
        labels = [-1] * (n + 1)
        
        if not hierarchy:
            return labels[:-1]
        
        thresholds = sorted(set(e[2] for e in hierarchy))
        epsilon = self.config.cluster_selection_epsilon
        
        cluster_id = 0
        for i, (a, b, dist) in enumerate(hierarchy):
            if dist > epsilon:
                cluster_id += 1
        
        return labels[:-1]
    
    def _compute_outlier_scores(
        self,
        distances: List[List[float]],
        labels: List[int]
    ) -> List[float]:
        """Compute outlier scores"""
        n = len(distances)
        scores = [1.0] * n
        
        if not labels:
            return scores
        
        label_set = set(labels)
        
        for i in range(n):
            if labels[i] == -1:
                scores[i] = 1.0
                continue
            
            same_cluster = [j for j in range(n) if labels[j] == labels[i] and j != i]
            if same_cluster:
                avg_dist = sum(distances[i][j] for j in same_cluster) / len(same_cluster)
                scores[i] = min(1.0, avg_dist)
        
        return scores
    
    def _spectral(self, embeddings: List[List[float]]) -> Tuple[List[int], List[List[float]]]:
        """
        Spectral clustering - graph-based.
        
        Advantages:
        - Handles non-convex shapes
        - Graph-based formulation
        - Good for manifold data
        """
        n = len(embeddings)
        
        if n < self.config.min_cluster_size:
            return list(range(n)), embeddings
        
        distances = self._compute_distance_matrix(embeddings)
        
        affinity = self._compute_affinity(distances)
        
        laplacian = self._compute_laplacian(affinity)
        
        eigenvalues, eigenvectors = self._power_iteration(laplacian, k=min(10, n // 2))
        
        kmeans_result = self._kmeans(eigenvectors)
        
        return kmeans_result, self._compute_centers(embeddings, kmeans_result[0])
    
    def _compute_affinity(self, distances: List[List[float]]) -> List[List[float]]:
        """Compute affinity matrix"""
        n = len(distances)
        sigma = self._estimate_sigma(distances)
        
        affinity = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    affinity[i][j] = math.exp(-distances[i][j] ** 2 / (2 * sigma ** 2))
        
        return affinity
    
    def _estimate_sigma(self, distances: List[List[float]]) -> float:
        """Estimate sigma for RBF kernel"""
        n = len(distances)
        sample_size = min(100, n * n // 2)
        
        samples = []
        for i in range(n):
            for j in range(i + 1, n):
                samples.append(distances[i][j])
                if len(samples) >= sample_size:
                    break
            if len(samples) >= sample_size:
                break
        
        samples.sort()
        return samples[len(samples) // 2] if samples else 1.0
    
    def _compute_laplacian(self, affinity: List[List[float]]) -> List[List[float]]:
        """Compute normalized Laplacian"""
        n = len(affinity)
        
        degree = [sum(affinity[i]) for i in range(n)]
        degree_sqrt = [math.sqrt(d) for d in degree]
        
        laplacian = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if degree_sqrt[i] > 0 and degree_sqrt[j] > 0:
                    laplacian[i][j] = -affinity[i][j] / (degree_sqrt[i] * degree_sqrt[j])
                if i == j:
                    laplacian[i][j] += 1.0
        
        return laplacian
    
    def _power_iteration(
        self,
        matrix: List[List[float]],
        k: int = 3
    ) -> Tuple[List[float], List[List[float]]]:
        """Power iteration for eigenvectors"""
        n = len(matrix)
        
        eigenvectors = []
        eigenvalues = []
        
        for _ in range(k):
            v = [1.0 / math.sqrt(n)] * n
            
            for _ in range(10):
                new_v = [0.0] * n
                for i in range(n):
                    for j in range(n):
                        new_v[i] += matrix[i][j] * v[j]
                
                norm = math.sqrt(sum(x ** 2 for x in new_v))
                if norm > 0:
                    v = [x / norm for x in new_v]
            
            eigenvalues.append(sum(v[i] * sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)))
            eigenvectors.append(v)
        
        return eigenvalues, eigenvectors
    
    def _diffusion_maps(self, embeddings: List[List[float]]) -> Tuple[List[int], List[List[float]]]:
        """
        Diffusion maps - manifold learning.
        
        Advantages:
        - Captures intrinsic geometry
        - Handles branching flows
        - Scale-aware
        """
        n = len(embeddings)
        
        if n < self.config.min_cluster_size:
            return list(range(n)), embeddings
        
        distances = self._compute_distance_matrix(embeddings)
        
        t = 3  # Diffusion time
        kernel = self._diffusion_kernel(distances)
        
        for _ in range(t - 1):
            kernel = self._matrix_multiply(kernel, kernel)
        
        eigenvalues, eigenvectors = self._power_iteration(kernel, k=min(10, n // 2))
        
        kmeans_result = self._kmeans(eigenvectors)
        
        return kmeans_result, self._compute_centers(embeddings, kmeans_result[0])
    
    def _diffusion_kernel(self, distances: List[List[float]]) -> List[List[float]]:
        """Compute diffusion kernel"""
        n = len(distances)
        sigma = self._estimate_sigma(distances)
        
        kernel = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                kernel[i][j] = math.exp(-distances[i][j] ** 2 / (sigma ** 2))
        
        row_sums = [sum(kernel[i]) for i in range(n)]
        for i in range(n):
            if row_sums[i] > 0:
                for j in range(n):
                    kernel[i][j] /= row_sums[i]
        
        return kernel
    
    def _matrix_multiply(
        self,
        a: List[List[float]],
        b: List[List[float]]
    ) -> List[List[float]]:
        """Matrix multiplication"""
        n = len(a)
        result = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result
    
    def _kmeans(self, embeddings: List[List[float]]) -> Tuple[List[int], List[List[float]]]:
        """Fallback to K-means"""
        n_clusters = min(self.config.min_cluster_size, len(embeddings))
        
        if len(embeddings) < n_clusters:
            return list(range(len(embeddings))), embeddings
        
        centers = embeddings[:n_clusters]
        
        for _ in range(10):
            labels = []
            for emb in embeddings:
                distances = [math.sqrt(sum((a - b) ** 2 for a, b in zip(emb, c)))
                            for c in centers]
                labels.append(distances.index(min(distances)))
            
            new_centers = [[0.0] * len(embeddings[0]) for _ in range(n_clusters)]
            counts = [0] * n_clusters
            
            for emb, label in zip(embeddings, labels):
                for i in range(len(emb)):
                    new_centers[label][i] += emb[i]
                counts[label] += 1
            
            for i in range(n_clusters):
                if counts[i] > 0:
                    new_centers[i] = [v / counts[i] for v in new_centers[i]]
            
            centers = new_centers
        
        return labels, centers
    
    def _compute_centers(
        self,
        embeddings: List[List[float]],
        labels: List[int]
    ) -> List[List[float]]:
        """Compute cluster centers"""
        n_clusters = max(labels) + 1 if labels else 1
        
        centers = [[0.0] * len(embeddings[0]) for _ in range(n_clusters)]
        counts = [0] * n_clusters
        
        for emb, label in zip(embeddings, labels):
            if 0 <= label < n_clusters:
                for i in range(len(emb)):
                    centers[label][i] += emb[i]
                counts[label] += 1
        
        for i in range(n_clusters):
            if counts[i] > 0:
                centers[i] = [v / counts[i] for v in centers[i]]
        
        return centers
    
    def predict(self, embedding: List[float]) -> int:
        """Predict cluster for new embedding"""
        if not self.fitted:
            return -1
        
        distances = [math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding, c)))
                    for c in self.cluster_centers]
        
        return distances.index(min(distances)) if distances else -1
    
    def get_cluster_info(self) -> Dict:
        """Get cluster information"""
        return {
            "algorithm": self.config.algorithm,
            "n_clusters": len(set(self.cluster_labels)),
            "total_points": len(self.cluster_labels),
            "avg_probability": sum(self.probabilities) / max(1, len(self.probabilities)) if self.probabilities else 0,
        }


class DynamicMotifTracker:
    """
    Tracks motifs that can:
    - Birth: new motif discovered
    - Merge: motifs combine
    - Death: motif fades
    - Split: motif divides
    
    This handles the dynamic nature of cognitive attractors.
    """
    
    def __init__(self, stability_threshold: float = 0.7):
        self.stability_threshold = stability_threshold
        
        self.motifs: Dict[str, Dict] = {}
        self.transition_history: List[Dict] = []
        
        logger.info("dynamic_motif_tracker_initialized")
    
    def update(
        self,
        new_motifs: List[Dict],
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Update motif states and detect transitions.
        
        Returns:
        - births: Newly formed motifs
        - deaths: Faded motifs
        - merges: Merged motifs
        - splits: Split motifs
        """
        timestamp = timestamp or datetime.utcnow()
        
        result = {
            "births": [],
            "deaths": [],
            "merges": [],
            "splits": [],
            "updated": []
        }
        
        new_motif_ids = {m["cluster_id"] for m in new_motifs}
        old_motif_ids = set(self.motifs.keys())
        
        for motif in new_motifs:
            motif_id = motif["cluster_id"]
            
            if motif_id not in self.motifs:
                result["births"].append(motif_id)
            else:
                result["updated"].append(motif_id)
        
        for motif_id in old_motif_ids:
            if motif_id not in new_motif_ids:
                if self.motifs[motif_id].get("stability", 0) < self.stability_threshold:
                    result["deaths"].append(motif_id)
        
        if result["births"] and result["deaths"]:
            result["merges"].append({
                "from": result["deaths"],
                "to": result["births"]
            })
        
        self.motifs = {m["cluster_id"]: m for m in new_motifs}
        
        self.transition_history.append({
            "timestamp": timestamp,
            "result": result,
            "total_motifs": len(new_motifs)
        })
        
        return result
    
    def get_motif_state(self, motif_id: str) -> Optional[Dict]:
        """Get current state of motif"""
        return self.motifs.get(motif_id)
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        return {
            "total_motifs": len(self.motifs),
            "transitions": len(self.transition_history),
            "recent_births": len(self.transition_history[-1].get("result", {}).get("births", [])) if self.transition_history else 0,
        }


# Factory functions
def create_advanced_clusterer(
    algorithm: str = "hdbscan",
    min_cluster_size: int = 3
) -> AdvancedClusterer:
    """Create advanced clusterer with specified algorithm"""
    config = ClusterConfig(
        algorithm=algorithm,
        min_cluster_size=min_cluster_size
    )
    return AdvancedClusterer(config)


def create_dynamic_tracker(
    stability_threshold: float = 0.7
) -> DynamicMotifTracker:
    """Create dynamic motif tracker"""
    return DynamicMotifTracker(stability_threshold=stability_threshold)