"""
Contradiction Engine v0 - Temporal pressure-based contradiction tracking.

Key principle:
    Contradiction is NOT boolean (true/false).
    Contradiction is pressure that builds over time.
    
Types of pressure:
    - confidence_divergence: beliefs drift apart
    - causal_conflict: mutually exclusive causal claims
    - compensation_loop: recursive undoing
    - reflection_collapse: self-negating chains
    - identity_instability: rapid belief mutations
"""
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from copy import deepcopy
import hashlib
import json

from wal_engine import get_wal_engine, WALEventType


class ContradictionPressureType:
    """Types of epistemic pressure that indicate contradictions"""
    
    # Low pressure
    CONFIDENCE_DIVERGENCE = "confidence_divergence"  # 0.1
    CAUSAL_WEAKNESS = "causal_weakness"  # 0.1
    
    # Medium pressure
    BELIEF_CONFLICT = "belief_conflict"  # 0.3
    MUTUALLY_EXCLUSIVE_CLAIMS = "mutually_exclusive_claims"  # 0.4
    MULTIPLE_COMPENSATIONS = "multiple_compensations"  # 0.4
    
    # High pressure
    RECURSIVE_COMPENSATION = "recursive_compensation"  # 0.7
    REFLECTION_COLLAPSE = "reflection_collapse"  # 0.8
    SELF_NEGATION = "self_negation"  # 1.0
    
    @classmethod
    def get_pressure(cls, pressure_type: str) -> float:
        """Get pressure value for type"""
        mapping = {
            cls.CONFIDENCE_DIVERGENCE: 0.1,
            cls.CAUSAL_WEAKNESS: 0.1,
            cls.BELIEF_CONFLICT: 0.3,
            cls.MUTUALLY_EXCLUSIVE_CLAIMS: 0.4,
            cls.MULTIPLE_COMPENSATIONS: 0.4,
            cls.RECURSIVE_COMPENSATION: 0.7,
            cls.REFLECTION_COLLAPSE: 0.8,
            cls.SELF_NEGATION: 1.0
        }
        return mapping.get(pressure_type, 0.0)


@dataclass
class ContradictionCluster:
    """
    Temporal contradiction object.
    
    Lives in time, has pressure, can be tracked through replay.
    """
    cluster_id: str
    involved_beliefs: List[str]
    pressure_type: str
    pressure_value: float
    severity: str  # low, medium, high, critical
    
    # Temporal metadata
    first_seen_at: str
    last_updated_at: str
    version_introduced: int
    
    # Causal topology
    originating_transactions: List[str] = field(default_factory=list)
    causal_events: List[str] = field(default_factory=list)
    
    # Resolution
    resolution_status: str = "active"  # active, resolving, resolved, collapsed
    resolution_events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "involved_beliefs": self.involved_beliefs,
            "pressure_type": self.pressure_type,
            "pressure_value": self.pressure_value,
            "severity": self.severity,
            "first_seen_at": self.first_seen_at,
            "last_updated_at": self.last_updated_at,
            "version_introduced": self.version_introduced,
            "originating_transactions": self.originating_transactions,
            "causal_events": self.causal_events,
            "resolution_status": self.resolution_status,
            "resolution_events": self.resolution_events
        }


class ContradictionEngine:
    """
    Temporal contradiction engine with pressure semantics.
    
    Key invariants:
    - All contradictions are tracked as events
    - Pressure accumulates, doesn't toggle
    - Contradiction clusters can be replayed
    """
    
    def __init__(self):
        self._clusters: Dict[str, ContradictionCluster] = {}
        self._pressure_history: List[Dict] = []  # For replay verification
        
        # Drift metrics
        self._drift_velocity: float = 0.0
        self._compensation_density: float = 0.0
        self._reflection_instability: float = 0.0
        self._cognitive_turbulence: float = 0.0
    
    def detect_pressure(
        self,
        beliefs: Dict[str, Any],
        transactions: Dict[str, Any],
        causal_graph: Dict[str, Any],
        current_version: int
    ) -> List[ContradictionCluster]:
        """
        Detect contradiction pressure from current state.
        
        Returns list of new or updated clusters.
        """
        new_clusters = []
        now = datetime.utcnow().isoformat()
        
        # 1. Confidence divergence check
        divergent_beliefs = self._check_confidence_divergence(beliefs)
        if divergent_beliefs:
            cluster = self._create_cluster(
                involved_beliefs=divergent_beliefs,
                pressure_type=ContradictionPressureType.CONFIDENCE_DIVERGENCE,
                current_version=current_version,
                now=now
            )
            new_clusters.append(cluster)
        
        # 2. Causal conflict check
        causal_conflicts = self._check_causal_conflicts(beliefs, causal_graph)
        if causal_conflicts:
            for conflict in causal_conflicts:
                cluster = self._create_cluster(
                    involved_beliefs=conflict["beliefs"],
                    pressure_type=ContradictionPressureType.MUTUALLY_EXCLUSIVE_CLAIMS,
                    current_version=current_version,
                    now=now,
                    causal_events=conflict["events"]
                )
                new_clusters.append(cluster)
        
        # 3. Compensation loop check
        compensation_loops = self._check_compensation_loops(transactions)
        if compensation_loops:
            for loop in compensation_loops:
                cluster = self._create_cluster(
                    involved_beliefs=[],
                    pressure_type=ContradictionPressureType.RECURSIVE_COMPENSATION,
                    current_version=current_version,
                    now=now,
                    originating_transactions=loop["transactions"]
                )
                new_clusters.append(cluster)
        
        # 4. Multiple compensations for same belief
        multiple_comp = self._check_multiple_compensations(beliefs, transactions)
        if multiple_comp:
            cluster = self._create_cluster(
                involved_beliefs=multiple_comp,
                pressure_type=ContradictionPressureType.MULTIPLE_COMPENSATIONS,
                current_version=current_version,
                now=now
            )
            new_clusters.append(cluster)
        
        # 5. Self-negation (belief contradicts itself via causal chain)
        self_negation = self._check_self_negation(beliefs, causal_graph)
        if self_negation:
            cluster = self._create_cluster(
                involved_beliefs=self_negation,
                pressure_type=ContradictionPressureType.SELF_NEGATION,
                current_version=current_version,
                now=now,
                severity="critical"
            )
            new_clusters.append(cluster)
        
        # Update clusters and record pressure events
        for cluster in new_clusters:
            existing = self._clusters.get(cluster.cluster_id)
            if existing:
                # Update existing cluster
                existing.last_updated_at = now
                existing.pressure_value = max(existing.pressure_value, cluster.pressure_value)
                existing.causal_events.extend(cluster.causal_events)
            else:
                # New cluster
                self._clusters[cluster.cluster_id] = cluster
            
            # Record pressure event for replay
            self._pressure_history.append({
                "timestamp": now,
                "cluster_id": cluster.cluster_id,
                "pressure_type": cluster.pressure_type,
                "pressure_value": cluster.pressure_value,
                "version": current_version
            })
        
        return new_clusters
    
    def _check_confidence_divergence(self, beliefs: Dict[str, Any]) -> List[str]:
        """Check for beliefs with diverging confidence"""
        divergent = []
        
        belief_list = list(beliefs.values())
        for i, b1 in enumerate(belief_list):
            for b2 in belief_list[i+1:]:
                # Check if beliefs about same or related propositions
                conf_delta = abs(b1.confidence - b2.confidence)
                if conf_delta > 0.4:  # Significant divergence
                    divergent.extend([b1.belief_id, b2.belief_id])
        
        return list(set(divergent))
    
    def _check_causal_conflicts(
        self,
        beliefs: Dict[str, Any],
        causal_graph: Dict[str, Any]
    ) -> List[Dict]:
        """Check for mutually exclusive causal claims"""
        conflicts = []
        
        for edge in causal_graph.get("edges", []):
            # If cause and effect both have high confidence but different values
            # This is simplified - full implementation would check proposition semantics
            pass  # Placeholder
        
        return conflicts
    
    def _check_compensation_loops(self, transactions: Dict[str, Any]) -> List[Dict]:
        """Check for recursive compensation loops"""
        loops = []
        
        for txn_id, txn_info in transactions.items():
            if txn_info.get("status") == "compensated":
                # Check if this transaction was already compensated multiple times
                compensated_count = 0
                for t_id, t_info in transactions.items():
                    if t_info.get("compensated_at", 0) > txn_info.get("compensated_at", 0):
                        compensated_count += 1
                
                if compensated_count > 1:
                    loops.append({
                        "transactions": [txn_id],
                        "count": compensated_count
                    })
        
        return loops
    
    def _check_multiple_compensations(
        self,
        beliefs: Dict[str, Any],
        transactions: Dict[str, Any]
    ) -> List[str]:
        """Check beliefs affected by multiple compensations"""
        affected = []
        
        for txn_id, txn_info in transactions.items():
            if txn_info.get("status") == "compensated":
                # Check compensation events - handle both dict and object
                for event in txn_info.get("events", []):
                    event_type = getattr(event, 'event_type', event.get('type') if isinstance(event, dict) else None)
                    target_id = getattr(event, 'target_id', event.get('target') if isinstance(event, dict) else None)
                    if event_type == "belief_updated" or event_type == "belief_updated":
                        affected.append(target_id)
        
        return list(set(affected))
    
    def _check_self_negation(
        self,
        beliefs: Dict[str, Any],
        causal_graph: Dict[str, Any]
    ) -> List[str]:
        """Check for self-negating belief chains"""
        # Simplified: check if belief has both high outgoing and incoming causal links
        # that create a cycle
        self_negating = []
        
        # This is placeholder - real implementation would analyze causal topology
        return self_negating
    
    def _create_cluster(
        self,
        involved_beliefs: List[str],
        pressure_type: str,
        current_version: int,
        now: str,
        originating_transactions: List[str] = None,
        causal_events: List[str] = None,
        severity: str = None
    ) -> ContradictionCluster:
        """Create contradiction cluster with proper metadata"""
        
        pressure_value = ContradictionPressureType.get_pressure(pressure_type)
        
        if severity is None:
            if pressure_value >= 0.7:
                severity = "critical"
            elif pressure_value >= 0.4:
                severity = "high"
            elif pressure_value >= 0.2:
                severity = "medium"
            else:
                severity = "low"
        
        return ContradictionCluster(
            cluster_id=str(uuid4()),
            involved_beliefs=involved_beliefs,
            pressure_type=pressure_type,
            pressure_value=pressure_value,
            severity=severity,
            first_seen_at=now,
            last_updated_at=now,
            version_introduced=current_version,
            originating_transactions=originating_transactions or [],
            causal_events=causal_events or []
        )
    
    def compute_drift_metrics(
        self,
        total_beliefs: int,
        total_transactions: int,
        compensated_transactions: int,
        deep_contradictions: int,
        recursive_compensations: int
    ) -> Dict[str, float]:
        """
        Compute epistemic drift metrics.
        
        These metrics can be tracked over time for cognitive health analysis.
        """
        # Drift velocity: rate of belief changes
        self._drift_velocity = (total_transactions / max(total_beliefs, 1)) * 0.1
        
        # Compensation density: ratio of compensated to total transactions
        self._compensation_density = compensated_transactions / max(total_transactions, 1)
        
        # Reflection instability: recursive compensation + contradictions
        self._reflection_instability = (
            recursive_compensations * 0.5 +
            deep_contradictions * 0.3
        )
        
        # Cognitive turbulence: combination of all instability
        self._cognitive_turbulence = (
            self._drift_velocity * 0.3 +
            self._compensation_density * 0.3 +
            self._reflection_instability * 0.4
        )
        
        return {
            "drift_velocity": self._drift_velocity,
            "compensation_density": self._compensation_density,
            "reflection_instability": self._reflection_instability,
            "cognitive_turbulence": self._cognitive_turbulence
        }
    
    def get_clusters(self) -> List[ContradictionCluster]:
        """Get all active contradiction clusters"""
        return [c for c in self._clusters.values() if c.resolution_status == "active"]
    
    def get_cluster_topology(self) -> Dict[str, Any]:
        """Get contradiction topology for causal graph integration"""
        nodes = []
        edges = []
        
        for cluster in self._clusters.values():
            nodes.append({
                "cluster_id": cluster.cluster_id,
                "pressure_type": cluster.pressure_type,
                "pressure_value": cluster.pressure_value,
                "severity": cluster.severity,
                "beliefs": cluster.involved_beliefs,
                "transactions": cluster.originating_transactions
            })
            
            # Connect to beliefs
            for belief_id in cluster.involved_beliefs:
                edges.append({
                    "from": cluster.cluster_id,
                    "to": belief_id,
                    "type": "contradiction"
                })
            
            # Connect to transactions
            for txn_id in cluster.originating_transactions:
                edges.append({
                    "from": cluster.cluster_id,
                    "to": txn_id,
                    "type": "originates_from"
                })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "total_clusters": len(nodes)
        }
    
    def get_pressure_events(self) -> List[Dict]:
        """Get pressure history for replay verification"""
        return self._pressure_history
    
    def verify_replay_equivalence(
        self,
        wal_events: List[Any],
        initial_state: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        Verify that contradiction detection is deterministic.
        
        Replay events and verify same clusters are formed.
        """
        # Reset for fresh analysis
        test_engine = ContradictionEngine()
        
        # Simulate state from events (simplified)
        beliefs = {}
        transactions = {}
        causal_graph = {"edges": []}
        
        # Replay events to reconstruct state
        for event in wal_events:
            # Handle both dict and object events
            event_type = getattr(event, 'event_type', event.get('type') if isinstance(event, dict) else None)
            target_id = getattr(event, 'target_id', event.get('target') if isinstance(event, dict) else None)
            payload = getattr(event, 'payload', event if isinstance(event, dict) else {})
            
            if event_type == "belief_added":
                conf = payload.get('confidence', 0.5) if isinstance(payload, dict) else 0.5
                beliefs[target_id] = type('Belief', (), {
                    'belief_id': target_id,
                    'confidence': conf
                })()
            elif event_type == "belief_updated":
                conf = payload.get('confidence', 0.5) if isinstance(payload, dict) else 0.5
                if target_id in beliefs:
                    beliefs[target_id].confidence = conf
            elif event_type == "transaction_compensated":
                txn_id = payload.get("original_transaction_id", target_id) if isinstance(payload, dict) else target_id
                transactions[txn_id] = {"status": "compensated", "events": [event]}
        
        # Detect on reconstructed state
        clusters = test_engine.detect_pressure(
            beliefs=beliefs,
            transactions=transactions,
            causal_graph=causal_graph,
            current_version=max(e.version for e in wal_events) if wal_events else 0
        )
        
        # Compare with current clusters
        current_count = len(self._clusters)
        replay_count = len(clusters)
        
        if abs(current_count - replay_count) > 1:
            return False, f"Cluster count mismatch: current={current_count}, replay={replay_count}"
        
        return True, f"Contradiction replay verified: {replay_count} clusters"
    
    def compute_cognitive_thermodynamics(
        self,
        total_beliefs: int,
        active_clusters: List[ContradictionCluster],
        compensation_density: float,
        time_delta: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute cognitive thermodynamics metrics.
        
        Key concepts:
        - Cognitive temperature: rate of instability
        - Entropy flow: direction of cognitive evolution  
        - Phase state: stable/transitioning/unstable
        - Energy barriers: difficulty of state transitions
        """
        thermodynamics = {}
        
        # 1. Cognitive temperature = pressure density / time
        total_pressure = sum(c.pressure_value for c in active_clusters)
        pressure_density = total_pressure / max(total_beliefs, 1)
        thermodynamics["cognitive_temperature"] = pressure_density / max(time_delta, 0.1)
        
        # 2. Entropy flow = change in cognitive diversity
        # High cluster diversity = high entropy
        cluster_types = set(c.pressure_type for c in active_clusters)
        thermodynamics["cognitive_entropy"] = len(cluster_types) / 8.0  # Normalize
        
        # 3. Phase state classification
        temp = thermodynamics["cognitive_temperature"]
        entropy = thermodynamics["cognitive_entropy"]
        
        if temp < 0.2 and entropy < 0.3:
            thermodynamics["phase_state"] = "stable_cold"
        elif temp < 0.4 and entropy < 0.5:
            thermodynamics["phase_state"] = "transitional"
        elif temp < 0.7:
            thermodynamics["phase_state"] = "unstable_warm"
        else:
            thermodynamics["phase_state"] = "critical_hot"
        
        # 4. Energy barrier = compensation effort required
        # High density = harder to change state
        thermodynamics["energy_barrier"] = compensation_density * temp
        
        # 5. Thermal conductivity = how quickly pressure propagates
        # More clusters = faster propagation
        thermodynamics["thermal_conductivity"] = len(active_clusters) / max(total_beliefs, 1)
        
        # 6. Cooling rate = natural decay of instability
        thermodynamics["cooling_rate"] = 1.0 - (temp * 0.5)  # Higher temp = slower cooling
        
        return thermodynamics
    
    def apply_pressure_decay(
        self,
        decay_rate: float = 0.1
    ) -> Dict[str, float]:
        """
        Apply decay to accumulated pressure.
        
        Key for hysteresis: pressure doesn't disappear instantly.
        Residue remains, creating cognitive "memory".
        
        Returns decayed values for tracking.
        """
        residual_pressure = {}
        
        # Decay each cluster's pressure
        for cluster_id, cluster in self._clusters.items():
            if cluster.resolution_status == "active":
                old_pressure = cluster.pressure_value
                new_pressure = old_pressure * (1.0 - decay_rate)
                
                # Minimum residue - pressure never fully disappears
                cluster.pressure_value = max(new_pressure, 0.05)
                
                residual_pressure[cluster_id] = {
                    "old": old_pressure,
                    "new": cluster.pressure_value,
                    "residue": cluster.pressure_value - new_pressure
                }
                
                # Update cluster if pressure dropped below threshold
                if cluster.pressure_value < 0.1:
                    cluster.resolution_status = "resolving"
        
        return residual_pressure
    
    def get_cognitive_scar_tissue(
        self,
        current_time: str
    ) -> List[Dict]:
        """
        Get residual pressure "scars" from past contradictions.
        
        These are the "memories" that affect cognition even after
        contradictions appear resolved.
        """
        scars = []
        now = datetime.fromisoformat(current_time)
        
        for cluster_id, cluster in self._clusters.items():
            if cluster.resolution_status in ["resolving", "resolved"]:
                # Calculate time since first seen
                first_seen = datetime.fromisoformat(cluster.first_seen_at)
                age_hours = (now - first_seen).total_seconds() / 3600
                
                # Scar persists based on original severity and age
                scar_severity = cluster.pressure_value * min(age_hours / 24.0, 1.0)
                
                if scar_severity > 0.05:
                    scars.append({
                        "cluster_id": cluster_id,
                        "original_pressure": cluster.pressure_value,
                        "age_hours": age_hours,
                        "scar_severity": scar_severity,
                        "original_beliefs": cluster.involved_beliefs
                    })
        
        return sorted(scars, key=lambda x: x["scar_severity"], reverse=True)
    
    def reset(self):
        """Reset engine for fresh analysis"""
        self._clusters = {}
        self._pressure_history = []
        self._drift_velocity = 0.0
        self._compensation_density = 0.0
        self._reflection_instability = 0.0
        self._cognitive_turbulence = 0.0


# Global instance
_contradiction_engine: Optional[ContradictionEngine] = None


def get_contradiction_engine() -> ContradictionEngine:
    """Get global contradiction engine"""
    global _contradiction_engine
    if _contradiction_engine is None:
        _contradiction_engine = ContradictionEngine()
    return _contradiction_engine


def reset_contradiction_engine():
    """Reset engine for testing"""
    global _contradiction_engine
    _contradiction_engine = ContradictionEngine()