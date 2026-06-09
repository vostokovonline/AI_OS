"""
Transactional Overlay Memory - ACID semantics for cognitive operations

Provides:
- Snapshot isolation between transactions
- Tx-local mutations (no premature in-memory changes)
- Commit merge → persistent state
- Rollback discard → clean overlay
- WAL-based recovery
- Overlay reads (merged view)

Architecture:
    SemanticMemoryRegion
    ├── committed_state
    ├── tx_overlays (tx_id -> pending mutations)
    ├── WAL
    └── recovery_engine
"""
import json
import hashlib
import threading
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from copy import deepcopy
from pathlib import Path


class TransactionState(Enum):
    """Transaction lifecycle"""
    ACTIVE = "active"
    PREPARED = "prepared"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class IsolationLevel(Enum):
    """Snapshot isolation levels"""
    READ_COMMITTED = "read_committed"  # See committed + own changes
    REPEATABLE_READ = "repeatable_read"  # See consistent snapshot
    SERIALIZABLE = "serializable"  # Full isolation


@dataclass
class Mutation:
    """Single mutation in transaction overlay"""
    mutation_id: str
    op: str  # "add", "update", "remove"
    key: str  # belief_id or metadata key
    old_value: Optional[Any]  # For undo
    new_value: Optional[Any]
    timestamp: str
    causal_context: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "mutation_id": self.mutation_id,
            "op": self.op,
            "key": self.key,
            "old_value": str(self.old_value) if self.old_value is not None else None,
            "new_value": str(self.new_value) if self.new_value is not None else None,
            "timestamp": self.timestamp,
            "causal_context": self.causal_context
        }


@dataclass
class TransactionOverlay:
    """Transactional overlay - mutations not yet committed"""
    tx_id: str
    state: TransactionState
    isolation: IsolationLevel
    started_at: str
    prepared_at: Optional[str] = None
    committed_at: Optional[str] = None
    
    # Tx-local mutations (not visible to others until commit)
    pending_mutations: List[Mutation] = field(default_factory=list)
    
    # Snapshot for repeatable read
    snapshot: Optional[Dict[str, Any]] = None
    
    # Causality tracking
    caused_by: Optional[str] = None  # Parent transaction
    intent_id: Optional[str] = None
    policy_version: Optional[str] = None


class WALRecord:
    """WAL record for recovery"""
    def __init__(self, record_type: str, tx_id: str, data: Dict, sequence: int):
        self.record_type = record_type  # "BEGIN", "PREPARE", "COMMIT", "ROLLBACK", "MUTATION"
        self.tx_id = tx_id
        self.data = data
        self.sequence = sequence
        self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        content = f"{self.record_type}:{self.tx_id}:{json.dumps(self.data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> dict:
        return {
            "record_type": self.record_type,
            "tx_id": self.tx_id,
            "data": self.data,
            "sequence": self.sequence,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "WALRecord":
        return cls(
            record_type=d["record_type"],
            tx_id=d["tx_id"],
            data=d["data"],
            sequence=d["sequence"]
        )


class TransactionalOverlay:
    """
    Transactional overlay providing ACID semantics.
    
    Key invariants:
    - PREPARE: only writes to overlay + WAL, NOT to committed state
    - COMMIT: merge overlay → committed state + WAL
    - ROLLBACK: discard overlay + WAL log
    - READ: merged view (committed + overlay)
    """
    
    def __init__(self, region_id: str, wal_path: Optional[str] = None):
        self.region_id = region_id
        
        # Persistent (committed) state
        self._committed_state: Dict[str, Any] = {}
        self._committed_version: int = 0
        
        # Transaction overlays (tx_id -> overlay)
        self._tx_overlays: Dict[str, TransactionOverlay] = {}
        
        # WAL for crash recovery
        self._wal_path = wal_path
        self._wal: List[WALRecord] = []
        self._wal_sequence: int = 0
        self._lock = threading.RLock()
        
        # Load from WAL if exists
        if wal_path and Path(wal_path).exists():
            self._load_wal()
    
    def _load_wal(self):
        """Load WAL from disk for recovery"""
        try:
            with open(self._wal_path, 'r') as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        self._wal.append(WALRecord.from_dict(d))
                        self._wal_sequence = max(self._wal_sequence, d["sequence"])
            
            # Recover committed state from WAL
            self._recover_from_wal()
        except Exception as e:
            print(f"WAL recovery failed: {e}")
    
    def _recover_from_wal(self):
        """Recover committed state from WAL log"""
        committed_checkpoint = {}
        
        for record in self._wal:
            if record.record_type == "COMMIT":
                # Apply committed mutations
                for mut_data in record.data.get("mutations", []):
                    if mut_data["op"] == "add":
                        committed_checkpoint[mut_data["key"]] = mut_data["new_value"]
                    elif mut_data["op"] == "update":
                        committed_checkpoint[mut_data["key"]] = mut_data["new_value"]
                    elif mut_data["op"] == "remove":
                        committed_checkpoint.pop(mut_data["key"], None)
        
        self._committed_state = committed_checkpoint
        self._committed_version = self._wal_sequence
    
    def begin_transaction(
        self,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
        caused_by: Optional[str] = None,
        intent_id: Optional[str] = None,
        policy_version: Optional[str] = None
    ) -> str:
        """Begin new transaction - returns tx_id"""
        with self._lock:
            tx_id = str(uuid4())
            
            # Create overlay with snapshot if repeatable read
            snapshot = None
            if isolation == IsolationLevel.REPEATABLE_READ:
                snapshot = deepcopy(self._committed_state)
            
            overlay = TransactionOverlay(
                tx_id=tx_id,
                state=TransactionState.ACTIVE,
                isolation=isolation,
                started_at=datetime.utcnow().isoformat(),
                snapshot=snapshot,
                caused_by=caused_by,
                intent_id=intent_id,
                policy_version=policy_version
            )
            
            self._tx_overlays[tx_id] = overlay
            
            # WAL: BEGIN
            self._append_wal("BEGIN", tx_id, {
                "isolation": isolation.value,
                "caused_by": caused_by,
                "intent_id": intent_id,
                "policy_version": policy_version
            })
            
            return tx_id
    
    def prepare_transaction(self, tx_id: str) -> bool:
        """
        PREPARE: Mark transaction as ready to commit.
        Critical: ONLY writes to overlay + WAL, NOT to committed state.
        """
        with self._lock:
            if tx_id not in self._tx_overlays:
                raise ValueError(f"Transaction {tx_id} not found")
            
            overlay = self._tx_overlays[tx_id]
            
            if overlay.state != TransactionState.ACTIVE:
                raise ValueError(f"Cannot prepare transaction in state {overlay.state}")
            
            # Validate all mutations before preparing
            for mutation in overlay.pending_mutations:
                if not self._validate_mutation(mutation):
                    overlay.state = TransactionState.FAILED
                    return False
            
            # Update state to PREPARED (still in overlay, not committed)
            overlay.state = TransactionState.PREPARED
            overlay.prepared_at = datetime.utcnow().isoformat()
            
            # WAL: PREPARE (still not in committed state!)
            self._append_wal("PREPARE", tx_id, {
                "mutation_count": len(overlay.pending_mutations),
                "prepared_at": overlay.prepared_at
            })
            
            return True
    
    def commit_transaction(self, tx_id: str) -> bool:
        """
        COMMIT: Merge overlay to committed state.
        Only now mutations become visible to other transactions.
        """
        with self._lock:
            if tx_id not in self._tx_overlays:
                raise ValueError(f"Transaction {tx_id} not found")
            
            overlay = self._tx_overlays[tx_id]
            
            if overlay.state != TransactionState.PREPARED:
                raise ValueError(f"Cannot commit transaction in state {overlay.state}")
            
            # Apply mutations to committed state
            for mutation in overlay.pending_mutations:
                self._apply_mutation_to_committed(mutation)
            
            # Update state
            overlay.state = TransactionState.COMMITTED
            overlay.committed_at = datetime.utcnow().isoformat()
            self._committed_version = self._wal_sequence
            
            # WAL: COMMIT (now mutations are durable)
            self._append_wal("COMMIT", tx_id, {
                "mutations": [m.to_dict() for m in overlay.pending_mutations],
                "committed_at": overlay.committed_at
            })
            
            # Clear overlay (transaction complete)
            del self._tx_overlays[tx_id]
            
            # Flush WAL
            self._flush_wal()
            
            return True
    
    def rollback_transaction(self, tx_id: str) -> bool:
        """
        ROLLBACK: Discard overlay without touching committed state.
        True ACID rollback - no reload from disk.
        """
        with self._lock:
            if tx_id not in self._tx_overlays:
                raise ValueError(f"Transaction {tx_id} not found")
            
            overlay = self._tx_overlays[tx_id]
            
            if overlay.state == TransactionState.COMMITTED:
                raise ValueError("Cannot rollback committed transaction")
            
            # WAL: ROLLBACK
            self._append_wal("ROLLBACK", tx_id, {
                "rolled_back_at": datetime.utcnow().isoformat(),
                "discarded_mutations": len(overlay.pending_mutations)
            })
            
            # Simply discard overlay - committed state untouched
            overlay.state = TransactionState.ROLLED_BACK
            del self._tx_overlays[tx_id]
            
            return True
    
    def add_mutation(
        self,
        tx_id: str,
        op: str,
        key: str,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        causal_context: Optional[Dict[str, str]] = None
    ) -> str:
        """Add mutation to transaction overlay (not committed yet)"""
        with self._lock:
            if tx_id not in self._tx_overlays:
                raise ValueError(f"Transaction {tx_id} not found")
            
            overlay = self._tx_overlays[tx_id]
            
            if overlay.state not in [TransactionState.ACTIVE, TransactionState.PREPARED]:
                raise ValueError(f"Cannot add mutation to transaction in state {overlay.state}")
            
            mutation = Mutation(
                mutation_id=str(uuid4()),
                op=op,
                key=key,
                old_value=old_value,
                new_value=new_value,
                timestamp=datetime.utcnow().isoformat(),
                causal_context=causal_context or {}
            )
            
            overlay.pending_mutations.append(mutation)
            
            # WAL: MUTATION (not yet in committed state!)
            self._append_wal("MUTATION", tx_id, mutation.to_dict())
            
            return mutation.mutation_id
    
    def read(
        self,
        tx_id: Optional[str] = None,
        key: Optional[str] = None
    ) -> Any:
        """
        READ: Merged view (committed + overlay for active tx).
        If no tx_id, returns committed state only.
        """
        with self._lock:
            # If no transaction, return committed only
            if tx_id is None:
                if key is None:
                    return deepcopy(self._committed_state)
                return deepcopy(self._committed_state.get(key))
            
            # With transaction - merged view
            if tx_id not in self._tx_overlays:
                raise ValueError(f"Transaction {tx_id} not found")
            
            overlay = self._tx_overlays[tx_id]
            
            # Get base (snapshot or committed)
            if overlay.isolation == IsolationLevel.REPEATABLE_READ and overlay.snapshot:
                base = overlay.snapshot
            else:
                base = self._committed_state
            
            # Apply pending mutations for merged view
            result = deepcopy(base)
            for mut in overlay.pending_mutations:
                if mut.op == "add" or mut.op == "update":
                    result[mut.key] = mut.new_value
                elif mut.op == "remove":
                    result.pop(mut.key, None)
            
            if key is None:
                return result
            return result.get(key)
    
    def _apply_mutation_to_committed(self, mutation: Mutation):
        """Apply mutation to committed state (only on commit)"""
        if mutation.op == "add" or mutation.op == "update":
            self._committed_state[mutation.key] = mutation.new_value
        elif mutation.op == "remove":
            self._committed_state.pop(mutation.key, None)
    
    def _validate_mutation(self, mutation: Mutation) -> bool:
        """Validate mutation before prepare"""
        # Can add custom validation logic here
        return True
    
    def _append_wal(self, record_type: str, tx_id: str, data: Dict):
        """Append to WAL"""
        self._wal_sequence += 1
        record = WALRecord(record_type, tx_id, data, self._wal_sequence)
        self._wal.append(record)
    
    def _flush_wal(self):
        """Flush WAL to disk"""
        if self._wal_path:
            with open(self._wal_path, 'a') as f:
                for record in self._wal[-10:]:  # Keep last 10 records
                    f.write(json.dumps(record.to_dict()) + "\n")
    
    def get_transaction_state(self, tx_id: str) -> Optional[TransactionState]:
        """Get current state of transaction"""
        with self._lock:
            if tx_id not in self._tx_overlays:
                return None
            return self._tx_overlays[tx_id].state
    
    def get_active_transactions(self) -> List[str]:
        """Get list of active transaction IDs"""
        with self._lock:
            return [
                tx_id for tx_id, overlay in self._tx_overlays.items()
                if overlay.state in [TransactionState.ACTIVE, TransactionState.PREPARED]
            ]
    
    def get_committed_state(self) -> Dict[str, Any]:
        """Get committed state (bypassing overlays)"""
        with self._lock:
            return deepcopy(self._committed_state)


class TransactionalMemoryManager:
    """Manager for multiple transactional overlays"""
    
    def __init__(self, base_path: str = "/tmp/transactional_memory"):
        self.base_path = base_path
        self._overlays: Dict[str, TransactionalOverlay] = {}
        self._lock = threading.RLock()
        
        # Ensure base path exists
        Path(base_path).mkdir(parents=True, exist_ok=True)
    
    def get_overlay(self, region_id: str) -> TransactionalOverlay:
        """Get or create overlay for region"""
        with self._lock:
            if region_id not in self._overlays:
                wal_path = f"{self.base_path}/wal_{region_id}.log"
                self._overlays[region_id] = TransactionalOverlay(region_id, wal_path)
            return self._overlays[region_id]
    
    def begin_transaction(
        self,
        region_id: str,
        isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
        caused_by: Optional[str] = None,
        intent_id: Optional[str] = None,
        policy_version: Optional[str] = None
    ) -> str:
        """Begin transaction in region"""
        overlay = self.get_overlay(region_id)
        return overlay.begin_transaction(
            isolation=isolation,
            caused_by=caused_by,
            intent_id=intent_id,
            policy_version=policy_version
        )
    
    def read(self, region_id: str, tx_id: Optional[str] = None, key: Optional[str] = None) -> Any:
        """Read from region"""
        overlay = self.get_overlay(region_id)
        return overlay.read(tx_id=tx_id, key=key)
    
    def get_active_transactions(self, region_id: str) -> List[str]:
        """Get active transactions in region"""
        overlay = self.get_overlay(region_id)
        return overlay.get_active_transactions()


# Global instance
_manager: Optional[TransactionalMemoryManager] = None


def get_transactional_manager(base_path: str = "/tmp/transactional_memory") -> TransactionalMemoryManager:
    """Get global transactional memory manager"""
    global _manager
    if _manager is None:
        _manager = TransactionalMemoryManager(base_path)
    return _manager


def begin_isolated_transaction(
    region_id: str,
    isolation: IsolationLevel = IsolationLevel.READ_COMMITTED,
    caused_by: Optional[str] = None,
    intent_id: Optional[str] = None,
    policy_version: Optional[str] = None
) -> str:
    """Convenience function to begin transaction"""
    manager = get_transactional_manager()
    return manager.begin_transaction(
        region_id=region_id,
        isolation=isolation,
        caused_by=caused_by,
        intent_id=intent_id,
        policy_version=policy_version
    )