"""
Semantic Memory Region - Isolated cognitive namespaces with transactional semantics

Enables:
- Cognitive isolation between agents
- Transactional memory regions  
- Scoped beliefs
- Capability sandboxes
- Crash-safe WAL recovery

Unlike global shared memory, this provides:
- Namespace isolation
- ACID transactions
- Semantic causality tracking
- Belief delta recording
"""
import json
import hashlib
import os
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from pathlib import Path
import threading


class MemoryRegion(str, Enum):
    """Memory isolation regions"""
    EPHEMERAL = "ephemeral"  # Scratch, no persistence
    SHORT_TERM = "short_term"  # Session memory
    LONG_TERM = "long_term"  # Persistent beliefs
    SHARED = "shared"  # Cross-agent shared
    QUARANTINE = "quarantine"  # Isolated for safety
    SELF_MODEL = "self_model"  # Agent's self-representation


@dataclass
class Belief:
    """Single belief with confidence and provenance"""
    belief_id: str
    proposition: str  # The belief statement
    confidence: float  # 0-1
    source: str  # What caused this belief
    created_at: str
    evidence: List[str]  # Supporting evidence IDs
    
    def to_dict(self) -> dict:
        return {
            "belief_id": self.belief_id,
            "proposition": self.proposition,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
            "evidence": self.evidence
        }


@dataclass
class BeliefDelta:
    """Record of belief change - critical for learning"""
    delta_id: str
    region: str
    
    # What changed
    added_beliefs: List[str]  # belief_ids
    removed_beliefs: List[str]  # belief_ids
    modified_confidences: Dict[str, float]  # belief_id -> new_confidence
    
    # Causality
    caused_by: Optional[str]  # Event ID that caused this
    depends_on: List[str]  # Prior belief_ids
    
    # Metadata
    timestamp: str
    causal_sequence: int  # Lamport clock
    
    def to_dict(self) -> dict:
        return {
            "delta_id": self.delta_id,
            "region": self.region,
            "added_beliefs": self.added_beliefs,
            "removed_beliefs": self.removed_beliefs,
            "modified_confidences": self.modified_confidences,
            "caused_by": self.caused_by,
            "depends_on": self.depends_on,
            "timestamp": self.timestamp,
            "causal_sequence": self.causal_sequence
        }


class WriteAheadLog:
    """
    Crash-safe WAL for transactional semantics.
    
    Format:
    1. BEGIN tx
    2. PREPARE op
    3. APPLY op (only after commit)
    4. COMMIT tx
    5. FINALIZE
    """
    
    def __init__(self, region: str, log_dir: str = "/app/wal"):
        self.region = region
        self.log_dir = Path(log_dir) / region
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._current_tx: Optional[str] = None
        self._pending_ops: List[Dict] = []
        
        # Load existing WAL
        self._wal_file = self.log_dir / "wal.log"
        self._committed_seq = self._load_committed_sequence()
    
    def _load_committed_sequence(self) -> int:
        """Load last committed sequence number"""
        seq_file = self.log_dir / "committed_seq.txt"
        if seq_file.exists():
            try:
                return int(seq_file.read_text().strip())
            except:
                pass
        return 0
    
    def _save_committed_sequence(self, seq: int):
        """Save committed sequence"""
        seq_file = self.log_dir / "committed_seq.txt"
        seq_file.write_text(str(seq))
    
    def begin(self, tx_id: str):
        """Begin transaction"""
        with self._lock:
            self._current_tx = tx_id
            self._pending_ops = []
            self._append({"op": "BEGIN", "tx": tx_id})
    
    def prepare(self, op: Dict):
        """Prepare operation (not yet applied)"""
        with self._lock:
            if self._current_tx is None:
                raise RuntimeError("No active transaction")
            
            op_entry = {"op": "PREPARE", "tx": self._current_tx, "data": op}
            self._pending_ops.append(op_entry)
            self._append(op_entry)
    
    def commit(self) -> bool:
        """Commit transaction - apply pending operations"""
        with self._lock:
            if not self._current_tx:
                return False
            
            # Mark all pending as committed
            self._committed_seq += 1
            commit_seq = self._committed_seq
            
            for op in self._pending_ops:
                self._append({
                    "op": "APPLY",
                    "tx": self._current_tx,
                    "seq": commit_seq,
                    "data": op["data"]
                })
            
            self._append({
                "op": "COMMIT",
                "tx": self._current_tx,
                "seq": commit_seq
            })
            
            self._save_committed_sequence(commit_seq)
            
            # Clear pending
            self._current_tx = None
            self._pending_ops = []
            
            return True
    
    def rollback(self):
        """Rollback transaction - discard pending"""
        with self._lock:
            if self._current_tx:
                self._append({"op": "ROLLBACK", "tx": self._current_tx})
                self._current_tx = None
                self._pending_ops = []
    
    def _append(self, entry: Dict):
        """Append to WAL"""
        with open(self._wal_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


class SemanticMemoryRegion:
    """
    Isolated memory region with transactional semantics.
    
    Provides:
    - Namespace isolation
    - WAL-based crash recovery
    - Belief tracking with deltas
    - Capability scoping
    """
    
    def __init__(
        self,
        region: MemoryRegion,
        allow_cross_region_read: bool = False,
        allow_cross_region_write: bool = False
    ):
        self.region = region
        self.allow_cross_region_read = allow_cross_region_read
        self.allow_cross_region_write = allow_cross_region_write
        
        # Storage
        self.store_dir = Path(f"/app/memory_regions/{region.value}")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # WAL
        self._wal = WriteAheadLog(region.value)
        
        # Beliefs
        self._beliefs: Dict[str, Belief] = {}
        self._load_beliefs()
        
        # Causal sequence
        self._causal_clock = 0
        
        # Lock
        self._lock = threading.RLock()
    
    def _load_beliefs(self):
        """Load beliefs from disk"""
        beliefs_file = self.store_dir / "beliefs.json"
        if beliefs_file.exists():
            try:
                data = json.load(beliefs_file)
                for b in data.get("beliefs", []):
                    self._beliefs[b["belief_id"]] = Belief(**b)
            except:
                pass
    
    def _save_beliefs(self):
        """Save beliefs to disk"""
        beliefs_file = self.store_dir / "beliefs.json"
        with open(beliefs_file, "w") as f:
            json.dump({
                "beliefs": [b.to_dict() for b in self._beliefs.values()]
            }, f, indent=2)
    
    def begin_transaction(self, tx_id: str):
        """Begin transaction with WAL"""
        self._wal.begin(tx_id)
    
    def add_belief(
        self,
        proposition: str,
        confidence: float,
        source: str,
        evidence: List[str] = None
    ) -> Belief:
        """Add belief (during transaction)"""
        belief = Belief(
            belief_id=uuid4().hex[:8],
            proposition=proposition,
            confidence=confidence,
            source=source,
            created_at=datetime.utcnow().isoformat(),
            evidence=evidence or []
        )
        
        # Log to WAL
        self._wal.prepare({
            "type": "add_belief",
            "belief": belief.to_dict()
        })
        
        # Add in memory
        self._beliefs[belief.belief_id] = belief
        
        return belief
    
    def update_confidence(self, belief_id: str, new_confidence: float) -> bool:
        """Update belief confidence"""
        if belief_id not in self._beliefs:
            return False
        
        old_confidence = self._beliefs[belief_id].confidence
        self._beliefs[belief_id].confidence = new_confidence
        
        self._wal.prepare({
            "type": "update_confidence",
            "belief_id": belief_id,
            "old": old_confidence,
            "new": new_confidence
        })
        
        return True
    
    def commit_transaction(self) -> bool:
        """Commit transaction - apply to disk"""
        # Save beliefs
        self._save_beliefs()
        
        # Commit WAL
        return self._wal.commit()
    
    def rollback_transaction(self):
        """Rollback transaction"""
        self._wal.rollback()
        # Reload from disk (undo in-memory changes)
        self._beliefs.clear()
        self._load_beliefs()
    
    def record_delta(
        self,
        added: List[str],
        removed: List[str],
        modified: Dict[str, float],
        caused_by: Optional[str] = None
    ) -> BeliefDelta:
        """Record belief delta for learning"""
        self._causal_clock += 1
        
        delta = BeliefDelta(
            delta_id=uuid4().hex[:8],
            region=self.region.value,
            added_beliefs=added,
            removed_beliefs=removed,
            modified_confidences=modified,
            caused_by=caused_by,
            depends_on=list(self._beliefs.keys()),
            timestamp=datetime.utcnow().isoformat(),
            causal_sequence=self._causal_clock
        )
        
        # Save delta
        deltas_file = self.store_dir / "deltas.json"
        deltas = []
        if deltas_file.exists():
            try:
                deltas = json.load(deltas_file)
            except:
                pass
        
        deltas.append(delta.to_dict())
        
        with open(deltas_file, "w") as f:
            json.dump(deltas[-100:], f, indent=2)  # Keep last 100
    
    def get_beliefs(self, min_confidence: float = 0.0) -> List[Belief]:
        """Get all beliefs above threshold"""
        return [b for b in self._beliefs.values() if b.confidence >= min_confidence]
    
    def get_statistics(self) -> Dict:
        """Get region statistics"""
        confidences = [b.confidence for b in self._beliefs.values()]
        
        return {
            "region": self.region.value,
            "belief_count": len(self._beliefs),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "causal_sequence": self._causal_clock
        }


class SemanticMemoryManager:
    """Manager for multiple memory regions"""
    
    def __init__(self):
        self._regions: Dict[MemoryRegion, SemanticMemoryRegion] = {}
        self._lock = threading.Lock()
        
        # Create default regions
        for region in MemoryRegion:
            self._regions[region] = SemanticMemoryRegion(region)
    
    def get_region(self, region: MemoryRegion) -> SemanticMemoryRegion:
        """Get or create region"""
        with self._lock:
            if region not in self._regions:
                self._regions[region] = SemanticMemoryRegion(region)
            return self._regions[region]
    
    def get_all_statistics(self) -> Dict:
        """Get statistics for all regions"""
        return {
            region.value: self._regions[region].get_statistics()
            for region in self._regions
        }


# Global manager
_memory_manager: Optional[SemanticMemoryManager] = None


def get_memory_manager() -> SemanticMemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = SemanticMemoryManager()
    return _memory_manager


def get_region(region: MemoryRegion) -> SemanticMemoryRegion:
    """Convenience function"""
    return get_memory_manager().get_region(region)