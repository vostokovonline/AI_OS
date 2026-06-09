"""
Reflection Kernel - Speculative cognitive sandbox for epistemic analysis

Provides:
- Analysis without direct state mutation
- EpistemicMutationProposal generation
- Conflict resolution
- Belief compression
- Causal reconciliation
- Attractor stabilization

Key principle:
    Reflection NEVER directly modifies UES.
    Only generates proposals for separate commit layer.
"""
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from copy import deepcopy


class MutationOperation(Enum):
    """Types of epistemic mutations"""
    UPDATE_CONFIDENCE = "update_confidence"
    ADD_BELIEF = "add_belief"
    REMOVE_BELIEF = "remove_belief"
    MERGE_BELIEFS = "merge_beliefs"
    SPLIT_BELIEF = "split_belief"
    ADD_CONSTRAINT = "add_constraint"
    REMOVE_CONSTRAINT = "remove_constraint"
    ADD_CAUSAL_EDGE = "add_causal_edge"
    REMOVE_CAUSAL_EDGE = "remove_causal_edge"
    STABILIZE_ATTRACTOR = "stabilize_attractor"
    RESOLVE_CONTRADICTION = "resolve_contradiction"
    RECOMPUTE_ENTROPY = "recompute_entropy"


class ReflectionDepth(Enum):
    """Reflection nesting depth"""
    SURFACE = "surface"  # Direct analysis
    CAUSAL = "causal"    # Why analysis
    META = "meta"        # Thinking about thinking
    RECURSIVE = "recursive"  # Deep recursion (dangerous)


@dataclass
class MutationOperationDetail:
    """Single mutation operation"""
    operation: MutationOperation
    target_id: str  # belief_id, edge_id, etc.
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    confidence: float = 1.0  # How confident we are in this operation
    reason: str = ""  # Why this operation


@dataclass
class EpistemicMutationProposal:
    """
    Proposed mutation from reflection.
    
    NOT applied directly - must go through validation.
    """
    proposal_id: str
    reflection_depth: ReflectionDepth
    triggered_by: str
    operations: List[MutationOperationDetail]
    expected_entropy_delta: float
    created_at: str
    trigger_beliefs: List[str] = field(default_factory=list)
    trigger_contradictions: List[str] = field(default_factory=list)
    expected_confidence_changes: Dict[str, float] = field(default_factory=dict)
    justification_trace: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    rollback_strategy: Optional[str] = None
    expires_at: Optional[str] = None
    proposal_confidence: float = 0.5
    requires_human_review: bool = False
    
    # Confidence in proposal
    proposal_confidence: float = 0.5  # How confident we are in entire proposal
    requires_human_review: bool = False


@dataclass
class ReflectionAnalysis:
    """Complete analysis from reflection"""
    analysis_id: str
    state_version: int
    state_hash: str
    analysis_depth: ReflectionDepth
    beliefs_analyzed: int
    contradictions_analyzed: int
    completed_at: str
    contradictions_found: List[Dict[str, Any]] = field(default_factory=list)
    unstable_beliefs: List[str] = field(default_factory=list)
    causal_gaps: List[Dict[str, Any]] = field(default_factory=list)
    entropy_clusters: List[str] = field(default_factory=list)
    attractor_issues: List[Dict[str, Any]] = field(default_factory=list)
    recommended_operations: List[MutationOperationDetail] = field(default_factory=list)


class ReflectionSandbox:
    """
    Speculative sandbox for reflection.
    
    Creates isolated analysis environment that never touches canonical UES.
    """
    
    def __init__(self, max_depth: int = 3):
        self._max_depth = max_depth
        self._current_depth: int = 0
        self._proposals: List[EpistemicMutationProposal] = []
    
    def analyze(
        self,
        state: Any,  # UnifiedEpistemicState
        ues_manager: Any,
        reason: str,
        depth: ReflectionDepth = ReflectionDepth.SURFACE
    ) -> ReflectionAnalysis:
        """
        Analyze current state WITHOUT modification.
        
        Returns frozen analysis with recommendations.
        """
        
        if self._current_depth >= self._max_depth:
            # Hard limit - return empty analysis
            return ReflectionAnalysis(
                analysis_id=str(uuid4()),
                state_version=state.version,
                state_hash=state.state_hash,
                contradictions_found=[],
                unstable_beliefs=[],
                causal_gaps=[],
                entropy_clusters=[],
                attractor_issues=[],
                recommended_operations=[],
                analysis_depth=ReflectionDepth.RECURSIVE,
                beliefs_analyzed=0,
                contradictions_analyzed=0,
                completed_at=datetime.utcnow().isoformat()
            )
        
        # Increment depth for this analysis
        self._current_depth += 1
        
        try:
            # Perform analysis
            analysis = self._perform_analysis(state, reason, depth)
            
            # Generate proposal from analysis
            if analysis.recommended_operations:
                proposal = self._create_proposal(analysis, reason, depth)
                self._proposals.append(proposal)
            
            return analysis
        
        finally:
            self._current_depth -= 1
    
    def _perform_analysis(
        self,
        state: Any,
        reason: str,
        depth: ReflectionDepth
    ) -> ReflectionAnalysis:
        """Perform actual analysis"""
        
        analysis = ReflectionAnalysis(
            analysis_id=str(uuid4()),
            state_version=state.version,
            state_hash=state.state_hash,
            analysis_depth=depth,
            beliefs_analyzed=len(state.beliefs),
            contradictions_analyzed=len(state.contradictions),
            completed_at=datetime.utcnow().isoformat()
        )
        
        # 1. Find contradictions
        for epid, contra in state.contradictions.items():
            if contra.stability_score > 0.7:  # Persistent
                analysis.contradictions_found.append({
                    "episode_id": epid,
                    "belief_ids": contra.belief_ids,
                    "stability": contra.stability_score,
                    "severity": contra.severity
                })
        
        # 2. Find unstable beliefs (oscillating, high entropy change)
        for bid, belief in state.beliefs.items():
            if belief.attractor_state == "oscillating":
                analysis.unstable_beliefs.append(bid)
            elif belief.entropy > 0.8:
                analysis.entropy_clusters.append(bid)
        
        # 3. Find attractor issues
        attractor_counts = {}
        for bid, belief in state.beliefs.items():
            at = belief.attractor_state
            attractor_counts[at] = attractor_counts.get(at, 0) + 1
        
        for at, count in attractor_counts.items():
            if at in ["oscillating", "diverging", "recursive"]:
                analysis.attractor_issues.append({
                    "attractor_type": at,
                    "belief_count": count
                })
        
        # 4. Generate recommended operations
        analysis.recommended_operations = self._generate_operations(analysis)
        
        return analysis
    
    def _generate_operations(
        self,
        analysis: ReflectionAnalysis
    ) -> List[MutationOperationDetail]:
        """Generate mutation operations based on analysis"""
        
        operations = []
        
        # Resolve contradictions
        for contra in analysis.contradictions_found[:3]:  # Limit per cycle
            # Strategy: reduce confidence of lower-confidence belief
            operations.append(MutationOperationDetail(
                operation=MutationOperation.RESOLVE_CONTRADICTION,
                target_id=contra["episode_id"],
                reason=f"Resolve persistent contradiction between {contra['belief_ids']}",
                confidence=0.7
            ))
        
        # Stabilize oscillating beliefs
        for bid in analysis.unstable_beliefs[:5]:  # Limit
            operations.append(MutationOperationDetail(
                operation=MutationOperation.STABILIZE_ATTRACTOR,
                target_id=bid,
                reason="Reduce oscillation by averaging confidence",
                confidence=0.6
            ))
        
        # Compress high-entropy beliefs
        for bid in analysis.entropy_clusters[:3]:
            operations.append(MutationOperationDetail(
                operation=MutationOperation.RECOMPUTE_ENTROPY,
                target_id=bid,
                reason="Recalculate belief entropy with new evidence",
                confidence=0.5
            ))
        
        return operations
    
    def _create_proposal(
        self,
        analysis: ReflectionAnalysis,
        reason: str,
        depth: ReflectionDepth
    ) -> EpistemicMutationProposal:
        """Create proposal from analysis"""
        
        # Compute expected entropy delta
        entropy_delta = 0.0
        for op in analysis.recommended_operations:
            if op.operation == MutationOperation.RESOLVE_CONTRADICTION:
                entropy_delta -= 0.1  # Resolving contradictions reduces entropy
            elif op.operation == MutationOperation.STABILIZE_ATTRACTOR:
                entropy_delta -= 0.05
        
        return EpistemicMutationProposal(
            proposal_id=str(uuid4()),
            reflection_depth=depth,
            triggered_by=reason,
            trigger_beliefs=analysis.unstable_beliefs,
            trigger_contradictions=[c["episode_id"] for c in analysis.contradictions_found],
            operations=analysis.recommended_operations,
            expected_entropy_delta=entropy_delta,
            expected_confidence_changes={},
            justification_trace=[
                f"Analyzed {analysis.beliefs_analyzed} beliefs",
                f"Found {len(analysis.contradictions_found)} persistent contradictions",
                f"Identified {len(analysis.unstable_beliefs)} oscillating beliefs",
                f"Generated {len(analysis.recommended_operations)} operations"
            ],
            created_at=datetime.utcnow().isoformat(),
            proposal_confidence=0.6
        )
    
    def get_pending_proposals(self) -> List[EpistemicMutationProposal]:
        """Get all proposals generated in this sandbox"""
        return self._proposals.copy()
    
    def clear_proposals(self):
        """Clear proposals (after commit or rollback)"""
        self._proposals = []


class EpistemicTransaction:
    """
    Transactional batch for epistemic mutations.
    
    Provides:
    - Atomic commit or rollback
    - Snapshot before mutation
    - Inverse operation tracking for rollback
    """
    
    def __init__(self, ues_manager: Any, proposal_id: str):
        self._ues = ues_manager
        self._proposal_id = proposal_id
        
        # Snapshot before mutation
        self._snapshot_version = ues_manager.get_current_state().version
        self._snapshot_hash = ues_manager.get_current_state().state_hash
        
        # Pending mutations (not yet applied)
        self._pending_ops: List[MutationOperationDetail] = []
        
        # Inverse operations for rollback
        self._inverse_ops: List[MutationOperationDetail] = []
        
        # State
        self._applied = False
        self._rolled_back = False
        self._errors: List[str] = []
    
    def add_operation(self, op: MutationOperationDetail, inverse: MutationOperationDetail):
        """Add operation to transaction with its inverse"""
        self._pending_ops.append(op)
        self._inverse_ops.append(inverse)
    
    def commit(self) -> tuple[bool, str]:
        """
        Commit all operations atomically using canonical/working copy pattern.
        
        Steps:
        1. Create working copy from canonical
        2. Validate operations against working copy
        3. Apply operations to working copy
        4. Validate working copy post-commit
        5. Commit working copy to canonical (atomic pointer swap)
        """
        
        if self._applied:
            return False, "Transaction already committed"
        
        if self._rolled_back:
            return False, "Transaction already rolled back"
        
        # Create working copy from canonical
        working = self._ues.create_working_copy()
        
        # Validate all operations before applying any
        for op in self._pending_ops:
            if not self._validate_operation(op, working):
                self._ues.discard_working_copy()
                self._rolled_back = True
                return False, f"Pre-commit validation failed: {self._errors}"
        
        # Apply all operations to working copy
        for op in self._pending_ops:
            try:
                self._apply_single_to_working_copy(op, working)
            except Exception as e:
                self._errors.append(f"Apply failed: {str(e)}")
                self._ues.discard_working_copy()
                self._rolled_back = True
                return False, f"Commit failed, rolled back: {self._errors}"
        
        # Post-commit validation on working copy
        if not self._validate_post_commit(working):
            self._ues.discard_working_copy()
            self._rolled_back = True
            return False, f"Post-commit validation failed: {self._errors}"
        
        # Commit working copy to canonical (atomic pointer swap)
        self._ues.commit_working_copy(reason=f"proposal_committed:{self._proposal_id}")
        
        self._applied = True
        return True, f"Committed {len(self._pending_ops)} operations"
    
    def compensate(self) -> tuple[bool, str]:
        """
        Compensate transaction by adding inverse events to WAL.
        
        Unlike "rollback" (which pretends events never happened),
        compensation explicitly records that the transaction was undone.
        
        Key philosophy:
        - History is primary - we never erase events
        - Compensation is an explicit forward event
        - Replay can choose historical (show all) or effective (mask compensated)
        """
        
        if self._rolled_back:
            return True, "Already compensated"
        
        # If operations were never applied, just mark as compensated
        if not self._applied:
            self._rolled_back = True
            return True, "Compensated (no operations applied)"
        
        # Add compensation event to WAL (not restoring snapshot!)
        from wal_engine import WALEventType, get_wal_engine
        wal = get_wal_engine()
        
        # Log compensation event with full metadata
        wal.log_event(
            event_type=WALEventType.TRANSACTION_COMPENSATED,
            version=self._ues._current_version + 1,
            operation="compensate",
            target_id=self._proposal_id,
            payload={
                "original_transaction_id": self._proposal_id,
                "reason": "compensation_requested",
                "operations_count": len(self._pending_ops)
            },
            parent_version=self._ues._current_version,
            actor="reflection",
            entropy_delta=0.0,
            compensates_transaction_id=self._proposal_id,
            transaction_id=self._proposal_id,
            reflection_depth=1,
            origin_agent="reflection_kernel"
        )
        
        # Discard any working copy
        if self._ues.has_working_copy():
            self._ues.discard_working_copy()
        
        self._rolled_back = True
        
        if self._errors:
            return False, f"Compensation completed with warnings: {self._errors}"
        
        return True, f"Compensated transaction {self._proposal_id}"
    
    def rollback(self) -> tuple[bool, str]:
        """Backward compatibility - alias for compensate"""
        return self.compensate()
    
    def _validate_operation(self, op: MutationOperationDetail, state: Any) -> bool:
        """Pre-commit validation for single operation"""
        
        if op.operation == MutationOperation.UPDATE_CONFIDENCE:
            if not 0 <= op.new_value <= 1:
                self._errors.append(f"Invalid confidence {op.new_value}")
                return False
        
        elif op.operation == MutationOperation.REMOVE_BELIEF:
            if op.target_id not in state.beliefs:
                self._errors.append(f"Belief {op.target_id} not found")
                return False
        
        # Add more validations as needed
        return True
    
    def _validate_post_commit(self, state: Any) -> bool:
        """Post-commit validation"""
        
        # Check entropy non-negative
        if state.total_entropy < 0:
            self._errors.append("CRITICAL: Negative entropy")
            return False
        
        # Check all beliefs have valid confidence
        for bid, belief in state.beliefs.items():
            if not 0 <= belief.confidence <= 1:
                self._errors.append(f"Invalid confidence for {bid}")
                return False
        
        return True
    
    def _apply_single_to_working_copy(self, op: MutationOperationDetail, working: Any):
        """
        Apply single operation directly to working copy.
        
        This operates on the working copy passed as parameter,
        NOT through UES methods which would create their own working copies.
        
        Also logs to WAL for deterministic replay.
        """
        
        if op.operation == MutationOperation.UPDATE_CONFIDENCE:
            existing = working.beliefs.get(op.target_id)
            old_confidence = existing.confidence if existing else 0.5
            if existing:
                existing.confidence = op.new_value if op.new_value is not None else 0.5
                existing.last_updated = datetime.utcnow().isoformat()
            
            # Log to WAL for replay
            from wal_engine import WALEventType, get_wal_engine
            wal = get_wal_engine()
            wal.log_event(
                event_type=WALEventType.BELIEF_UPDATED,
                version=self._ues._current_version + 1,
                operation="update",
                target_id=op.target_id,
                payload={"confidence": op.new_value, "reason": op.reason},
                parent_version=self._ues._current_version,
                actor="reflection",
                entropy_delta=0.0,
                transaction_id=self._proposal_id,
                confidence_before=old_confidence,
                confidence_after=op.new_value,
                reflection_depth=1
            )
        
        elif op.operation == MutationOperation.ADD_BELIEF:
            from unified_epistemic_state import BeliefState
            new_belief = BeliefState(
                belief_id=op.target_id,
                proposition=op.new_value if isinstance(op.new_value, str) else f"Belief {op.target_id}",
                confidence=0.5,
                entropy=0.5,
                source="reflection",
                created_at=datetime.utcnow().isoformat(),
                last_updated=datetime.utcnow().isoformat(),
                version=-1,
                incoming_causes=[],
                outgoing_effects=[]
            )
            working.beliefs[op.target_id] = new_belief
        
        elif op.operation == MutationOperation.REMOVE_BELIEF:
            if op.target_id in working.beliefs:
                working.beliefs[op.target_id].confidence = 0.01
                working.beliefs[op.target_id].entropy = 0.99
        
        elif op.operation == MutationOperation.RESOLVE_CONTRADICTION:
            if op.target_id in working.contradictions:
                working.contradictions[op.target_id].resolution_status = "resolved"
        
        elif op.operation == MutationOperation.STABILIZE_ATTRACTOR:
            if op.target_id in working.beliefs:
                working.beliefs[op.target_id].attractor_state = "stable"
    
    def _rollback_partial(self):
        """Rollback applied operations - just discard working copy"""
        if self._ues.has_working_copy():
            self._ues.discard_working_copy()
        self._rolled_back = True


class ReflectionCommitLayer:
    """
    Validates and commits proposals from reflection sandbox.
    
    Provides:
    - Transactional mutation batches
    - Snapshot-based rollback
    - Conflict detection
    - Invariant validation
    """
    
    def __init__(self, ues_manager: Any):
        self._ues = ues_manager
        self._active_transactions: Dict[str, EpistemicTransaction] = {}
        self._transaction_proposals: Dict[str, EpistemicMutationProposal] = {}  # Store proposal for conflict detection
        self._proposal_locks: Set[str] = set()  # Proposals being processed
    
    def validate_proposal(
        self,
        proposal: EpistemicMutationProposal
    ) -> tuple[bool, List[str]]:
        """
        Validate proposal before commit.
        
        Returns: (valid, validation_errors)
        """
        
        errors = []
        
        # Check proposal not already locked
        if proposal.proposal_id in self._proposal_locks:
            errors.append("CRITICAL: Proposal already being processed")
            return False, errors
        
        # 1. Check entropy bounds
        current_state = self._ues.get_current_state()
        new_entropy = current_state.total_entropy + proposal.expected_entropy_delta
        
        if new_entropy < 0:
            errors.append("CRITICAL: Entropy would go negative")
        
        if new_entropy > len(current_state.beliefs) * 1.5:
            errors.append("WARNING: Entropy would significantly exceed belief count")
        
        # 2. Check operation confidence thresholds
        low_confidence_ops = []
        for op in proposal.operations:
            if op.confidence < 0.3:
                low_confidence_ops.append(f"{op.operation.value} on {op.target_id}")
        
        if len(low_confidence_ops) > len(proposal.operations) * 0.5:
            errors.append(f"WARNING: {len(low_confidence_ops)} operations have low confidence")
        
        # 3. Check for contradiction amplification
        current_contradictions = len(current_state.contradictions)
        resolve_ops = [o for o in proposal.operations 
                      if o.operation == MutationOperation.RESOLVE_CONTRADICTION]
        
        if len(resolve_ops) < current_contradictions * 0.3 and current_contradictions > 2:
            errors.append("WARNING: Proposal may not resolve enough contradictions")
        
        # 4. Check rollback strategy exists for complex proposals
        if not proposal.rollback_strategy and len(proposal.operations) > 3:
            errors.append("WARNING: Complex proposal without explicit rollback strategy")
        
        # 5. Check depth not exceeded
        if proposal.reflection_depth == ReflectionDepth.RECURSIVE:
            errors.append("CRITICAL: Cannot commit recursive reflection proposal")
        
        # 6. Check conflict with pending proposals
        conflicts = self._detect_conflicts(proposal)
        if conflicts:
            errors.append(f"WARNING: Conflicts with {len(conflicts)} pending proposals")
        
        # Check for critical errors only
        critical = [e for e in errors if e.startswith("CRITICAL")]
        if critical:
            return False, critical
        
        return len(errors) == 0, errors
    
    def _detect_conflicts(self, proposal: EpistemicMutationProposal) -> List[str]:
        """Detect conflicts with other pending proposals"""
        
        conflicts = []
        
        # Get target IDs from this proposal
        targets = set()
        for op in proposal.operations:
            targets.add(op.target_id)
        
        if not targets:
            return conflicts
        
        # Check against active transactions - use stored proposal targets
        for tx_id, tx_proposal in self._transaction_proposals.items():
            tx_targets = set()
            for op in tx_proposal.operations:
                tx_targets.add(op.target_id)
            
            # If overlapping targets, potential conflict
            if targets & tx_targets:
                conflicts.append(tx_id)
        
        return conflicts
    
    def begin_transaction(
        self,
        proposal: EpistemicMutationProposal
    ) -> EpistemicTransaction:
        """Begin new epistemic transaction"""
        
        # Check for conflicts
        conflicts = self._detect_conflicts(proposal)
        if conflicts:
            raise ValueError(f"Proposal conflicts with active transactions: {conflicts}")
        
        # Lock proposal
        self._proposal_locks.add(proposal.proposal_id)
        
        # Create transaction
        tx = EpistemicTransaction(self._ues, proposal.proposal_id)
        self._active_transactions[proposal.proposal_id] = tx
        self._transaction_proposals[proposal.proposal_id] = proposal  # Store for conflict detection
        
        return tx
    
    def commit_proposal(
        self,
        proposal: EpistemicMutationProposal
    ) -> tuple[bool, str]:
        """
        Commit validated proposal via transactional batch.
        
        Returns: (success, commit_summary)
        """
        
        # Validate first
        valid, errors = self.validate_proposal(proposal)
        
        if not valid:
            critical = [e for e in errors if e.startswith("CRITICAL")]
            if critical:
                return False, f"Validation failed: {critical}"
            # Warnings - may proceed
            print(f"Proposal warnings: {errors}")
        
        # Begin transaction
        tx = self.begin_transaction(proposal)
        
        try:
            # Add all operations with inverses
            for op in proposal.operations:
                inverse = self._create_inverse(op)
                tx.add_operation(op, inverse)
            
            # Commit transaction
            success, msg = tx.commit()
            
            if success:
                # Release lock
                self._proposal_locks.discard(proposal.proposal_id)
                del self._active_transactions[proposal.proposal_id]
                self._transaction_proposals.pop(proposal.proposal_id, None)
            
            return success, msg
        
        except Exception as e:
            # Rollback on exception
            tx.rollback()
            self._proposal_locks.discard(proposal.proposal_id)
            if proposal.proposal_id in self._active_transactions:
                del self._active_transactions[proposal.proposal_id]
                self._transaction_proposals.pop(proposal.proposal_id, None)
            return False, f"Transaction failed: {str(e)}"
    
    def _create_inverse(self, op: MutationOperationDetail) -> MutationOperationDetail:
        """Create inverse operation for rollback"""
        
        if op.operation == MutationOperation.UPDATE_CONFIDENCE:
            return MutationOperationDetail(
                operation=MutationOperation.UPDATE_CONFIDENCE,
                target_id=op.target_id,
                old_value=op.new_value,
                new_value=op.old_value,
                confidence=op.confidence,
                reason=f"Inverse of: {op.reason}"
            )
        
        elif op.operation == MutationOperation.ADD_BELIEF:
            return MutationOperationDetail(
                operation=MutationOperation.REMOVE_BELIEF,
                target_id=op.target_id,
                reason=f"Inverse of: {op.reason}"
            )
        
        elif op.operation == MutationOperation.REMOVE_BELIEF:
            # Can't truly restore removed belief - this is simplification
            return MutationOperationDetail(
                operation=MutationOperation.ADD_BELIEF,
                target_id=op.target_id,
                old_value=None,
                new_value=op.old_value,
                reason=f"Rollback of removal: {op.reason}"
            )
        
        # Default: no inverse possible
        return MutationOperationDetail(
            operation=op.operation,
            target_id=op.target_id,
            reason=f"No inverse for: {op.operation.value}"
        )
    
    def rollback_proposal(self, proposal_id: str) -> tuple[bool, str]:
        """Rollback active transaction"""
        
        if proposal_id not in self._active_transactions:
            return False, "No active transaction found"
        
        tx = self._active_transactions[proposal_id]
        success, msg = tx.rollback()
        
        self._proposal_locks.discard(proposal_id)
        del self._active_transactions[proposal_id]
        self._transaction_proposals.pop(proposal_id, None)
        
        return success, msg
    
    def get_active_transactions(self) -> List[str]:
        """Get list of active transaction IDs"""
        return list(self._active_transactions.keys())


class ReflectionKernel:
    """
    Reflection Kernel - orchestrates reflection analysis and proposal generation.
    
    Architecture:
        UES Snapshot → Reflection Sandbox → Proposal → Validation → Commit
    """
    
    def __init__(self, ues_manager: Any, config: Optional[Dict] = None):
        self.config = config or {}
        self._ues = ues_manager
        self._sandbox = ReflectionSandbox(
            max_depth=self.config.get("max_reflection_depth", 3)
        )
        self._commit_layer = ReflectionCommitLayer(ues_manager)
        
        # History
        self._analysis_history: List[ReflectionAnalysis] = []
        self._proposal_history: List[EpistemicMutationProposal] = []
    
    def reflect(
        self,
        reason: str,
        depth: ReflectionDepth = ReflectionDepth.SURFACE
    ) -> Optional[EpistemicMutationProposal]:
        """
        Main entry point: perform reflection and generate proposal.
        
        Returns proposal (NOT applied) - must be committed separately.
        """
        
        # Get current state
        state = self._ues.get_current_state()
        
        # Analyze in sandbox (no mutation)
        analysis = self._sandbox.analyze(state, self._ues, reason, depth)
        
        # Store in history
        self._analysis_history.append(analysis)
        
        # Get proposals from sandbox
        proposals = self._sandbox.get_pending_proposals()
        
        if not proposals:
            return None
        
        # Return most recent proposal
        proposal = proposals[-1]
        self._proposal_history.append(proposal)
        
        return proposal
    
    def commit_proposal(
        self,
        proposal: EpistemicMutationProposal
    ) -> tuple[bool, str]:
        """Commit validated proposal"""
        
        result = self._commit_layer.commit_proposal(proposal)
        
        # Clear sandbox proposals after commit attempt
        self._sandbox.clear_proposals()
        
        return result
    
    def get_analysis_history(self, limit: int = 10) -> List[ReflectionAnalysis]:
        """Get recent analysis history"""
        return self._analysis_history[-limit:]
    
    def get_proposal_history(self, limit: int = 10) -> List[EpistemicMutationProposal]:
        """Get recent proposal history"""
        return self._proposal_history[-limit:]


# Global instance
_kernel: Optional[ReflectionKernel] = None


def get_reflection_kernel(
    ues_manager: Any = None,
    config: Optional[Dict] = None
) -> ReflectionKernel:
    """Get global reflection kernel"""
    global _kernel
    
    # Need to import here to avoid circular dependency
    if _kernel is None and ues_manager is not None:
        _kernel = ReflectionKernel(ues_manager, config)
    elif _kernel is None:
        # Placeholder until initialized with UES
        pass
    
    return _kernel


def init_reflection_kernel(ues_manager: Any, config: Optional[Dict] = None):
    """Initialize reflection kernel with UES manager"""
    global _kernel
    _kernel = ReflectionKernel(ues_manager, config)
    return _kernel