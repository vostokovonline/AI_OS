"""
Epistemic Kernel — append-only, replayable, invariant-validated governance layer
for beliefs, motifs, attractors, and semantic state.

MIRRORS Execution Kernel architecture:

  Execution Kernel          Epistemic Kernel
  ─────────────             ───────────────
  dispatch journal          semantic journal
  WAL                       observation log
  lease epoch               interpretation epoch
  capability provenance     belief provenance
  invariant engine           semantic invariant engine
  replay recovery           semantic replay
  fencing token             contradiction threshold
  snapshot                  grounding checkpoint

PRINCIPLES:
  - Append-only: interpretations are journaled, never mutated in place
  - Deterministic replay: semantic state₀ + observation log == current state
  - Governed: all belief transitions validated by invariant engine
  - Observable: provenance chain answers "why does system believe X?"

USAGE:
    from epistemic_kernel import EpistemicKernel

    ek = EpistemicKernel()
    ek.record_observation("sleep_fragmentation", confidence=0.7)
    ek.update_belief("burnout_risk", 0.65, provenance="motif_detection")
    report = ek.verify()  # semantic invariants
"""

from .epoch import InterpretationEpoch
from .provenance import ProvenanceGraph
from .semantic_journal import SemanticJournal, InterpretationEvent
from .drift import DriftDetector, DriftReport
from .grounding import GroundingManager
from .semantic_invariants import SemanticInvariantEngine, SemanticInvariantReport

import logging
logger = logging.getLogger(__name__)


class EpistemicKernel:
    """
    Facade for the epistemic kernel — governed semantic evolution.

    Manages:
      - interpretation epochs (versioning the semantic model)
      - belief provenance chain (why each belief exists)
      - semantic journal (append-only interpretation events)
      - drift detection + attenuation (anti-collapse)
      - grounding checkpoints (replay anchors)
      - semantic invariants (epistemic correctness)
    """

    def __init__(self):
        # Epoch — version counter for the semantic model
        self.epoch = InterpretationEpoch()

        # Journal — append-only log of all interpretation events
        self.journal = SemanticJournal()

        # Provenance — why each belief is held
        self.provenance = ProvenanceGraph()

        # Drift detector — monitors semantic divergence
        self.drift = DriftDetector()

        # Grounding — replay anchors
        self.grounding = GroundingManager()

        # Invariant engine — semantic correctness
        self._invariants = SemanticInvariantEngine(self)

        self._beliefs: dict = {}       # belief_name -> {confidence, epoch, ...}
        self._motifs: dict = {}        # motif_name -> {strength, recurrence, ...}
        self._attractors: dict = {}    # attractor_id -> {weight, ...}
        self._observation_count = 0

        logger.info("epistemic_kernel_initialized")

    # ------------------------------------------------------------------
    # OBSERVATIONS — raw input to the semantic layer
    # ------------------------------------------------------------------

    def record_observation(
        self,
        signal: str,
        value: float = 1.0,
        source: str = "",
        context: dict = None,
    ) -> str:
        """
        Record a raw observation. Observations are the semantic analog
        of execution dispatch requests — they enter the journal and
        may trigger belief/motif updates.

        Args:
            signal: observation type (e.g. "sleep_fragmentation")
            value: observed intensity (0.0-1.0)
            source: origin (e.g. "wearable", "self_report", "app_usage")
            context: optional structured data

        Returns:
            observation_id
        """
        self._observation_count += 1
        event = self.journal.append_observation(
            signal=signal,
            value=value,
            source=source,
            context=context or {},
            epoch=self.epoch.current,
            observation_index=self._observation_count,
        )
        self.epoch.touch()
        return event.event_id

    # ------------------------------------------------------------------
    # BELIEFS — interpreted state
    # ------------------------------------------------------------------

    def update_belief(
        self,
        name: str,
        confidence: float,
        provenance: str = "",
        source_event_id: str = "",
    ) -> dict:
        """
        Update a belief with provenance tracking.

        Args:
            name: belief identifier (e.g. "burnout_risk", "focus_quality")
            confidence: new confidence value (0.0-1.0)
            provenance: why this belief changed (e.g. "motif_detection")
            source_event_id: linking to a specific journal event

        Returns:
            current belief state dict
        """
        confidence = max(0.0, min(1.0, confidence))
        prev = self._beliefs.get(name, {})
        prev_conf = prev.get('confidence', 0.0)

        # Journal the belief transition
        event = self.journal.append_belief_update(
            belief_name=name,
            previous_confidence=round(prev_conf, 4),
            new_confidence=round(confidence, 4),
            provenance=provenance,
            epoch=self.epoch.current,
        )

        # Provenance chain
        self.provenance.record_belief_provenance(
            belief_name=name,
            confidence_delta=round(confidence - prev_conf, 4),
            provenance_label=provenance,
            event_id=event.event_id,
            epoch=self.epoch.current,
        )

        # Store belief
        self._beliefs[name] = {
            'confidence': round(confidence, 4),
            'updated_at_epoch': self.epoch.current,
            'updated_at_event': event.event_id,
            'provenance': provenance,
        }

        self.epoch.touch()
        return dict(self._beliefs[name])

    def get_belief(self, name: str) -> dict:
        """Get current state of a belief."""
        return dict(self._beliefs.get(name, {'confidence': 0.0}))

    def get_all_beliefs(self) -> dict:
        """Get all current beliefs."""
        return {k: dict(v) for k, v in self._beliefs.items()}

    # ------------------------------------------------------------------
    # MOTIFS — recurring semantic patterns
    # ------------------------------------------------------------------

    def update_motif(
        self,
        name: str,
        strength: float,
        recurrence: int = 1,
        provenance: str = "",
    ) -> dict:
        """
        Update or reinforce a motif (recurring interpretation pattern).

        Args:
            name: motif identifier
            strength: current strength (0.0-1.0)
            recurrence: how many times this motif has been detected
            provenance: detection source

        Returns:
            current motif state
        """
        strength = max(0.0, min(1.0, strength))
        prev = self._motifs.get(name, {})
        prev_strength = prev.get('strength', 0.0)

        event = self.journal.append_motif_update(
            motif_name=name,
            previous_strength=round(prev_strength, 4),
            new_strength=round(strength, 4),
            recurrence=recurrence,
            provenance=provenance,
            epoch=self.epoch.current,
        )

        self.provenance.record_belief_provenance(
            belief_name=f"motif:{name}",
            confidence_delta=round(strength - prev_strength, 4),
            provenance_label=provenance,
            event_id=event.event_id,
            epoch=self.epoch.current,
        )

        self._motifs[name] = {
            'strength': round(strength, 4),
            'recurrence': recurrence,
            'updated_at_epoch': self.epoch.current,
            'provenance': provenance,
        }

        self.epoch.touch()
        return dict(self._motifs[name])

    def get_motif(self, name: str) -> dict:
        return dict(self._motifs.get(name, {'strength': 0.0, 'recurrence': 0}))

    def get_all_motifs(self) -> dict:
        return {k: dict(v) for k, v in self._motifs.items()}

    # ------------------------------------------------------------------
    # ATTRACTORS — stable semantic fixed points
    # ------------------------------------------------------------------

    def update_attractor(
        self,
        attractor_id: str,
        weight: float,
        label: str = "",
    ) -> dict:
        """
        Update attractor weight. Attractors are stable semantic fixed
        points in the interpretation landscape.

        Args:
            attractor_id: attractor identifier
            weight: attractor strength (0.0-1.0)
            label: optional human-readable label

        Returns:
            current attractor state
        """
        weight = max(0.0, min(1.0, weight))
        prev = self._attractors.get(attractor_id, {})
        prev_weight = prev.get('weight', 0.0)

        self.journal.append_event(
            event_type='ATTRACTOR_UPDATE',
            detail=f"attractor={attractor_id} label={label or attractor_id} from={prev_weight} to={weight}",
            epoch=self.epoch.current,
            data={
                'attractor_id': attractor_id,
                'previous_weight': round(prev_weight, 4),
                'weight': round(weight, 4),
                'label': label or attractor_id,
            },
        )

        self._attractors[attractor_id] = {
            'weight': round(weight, 4),
            'label': label or attractor_id,
            'updated_at_epoch': self.epoch.current,
        }
        self.epoch.touch()
        return dict(self._attractors[attractor_id])

    def get_attractor(self, attractor_id: str) -> dict:
        return dict(self._attractors.get(attractor_id, {'weight': 0.0}))

    def get_all_attractors(self) -> dict:
        return {k: dict(v) for k, v in self._attractors.items()}

    # ------------------------------------------------------------------
    # DRIFT DETECTION
    # ------------------------------------------------------------------

    def check_drift(self) -> 'DriftReport':
        """
        Check for semantic drift in beliefs and motifs.

        Drift is detected when:
          - Confidence grows without new observations
          - Motifs self-reinforce without fresh evidence
          - Attractor weights diverge from observation frequency
          - Interpretation epoch advances faster than observation rate

        Returns:
            DriftReport with violations and recommendations.
        """
        beliefs_snapshot = dict(self._beliefs)
        motifs_snapshot = dict(self._motifs)
        attractors_snapshot = dict(self._attractors)

        return self.drift.evaluate(
            beliefs=beliefs_snapshot,
            motifs=motifs_snapshot,
            attractors=attractors_snapshot,
            epoch=self.epoch.current,
            observation_count=self._observation_count,
            journal_event_count=len(self.journal._events),
        )

    def attenuate_drift(self, report: 'DriftReport') -> int:
        """
        Apply drift attenuation: reduce confidence/strength for drifted beliefs.

        Args:
            report: DriftReport from check_drift()

        Returns:
            number of attenuated beliefs
        """
        count = 0
        for violation in report.violations:
            target = violation.target
            if violation.type == 'belief_drift' and target in self._beliefs:
                current = self._beliefs[target]['confidence']
                attenuation = violation.attenuation
                new_conf = max(0.0, current - attenuation)
                self._beliefs[target]['confidence'] = round(new_conf, 4)
                self._beliefs[target]['provenance'] = 'drift_attenuation'
                count += 1
                self.journal.append_event(
                    event_type='DRIFT_ATTENUATION',
                    detail=f"attenuated belief={target} from={current} to={new_conf}",
                    epoch=self.epoch.current,
                )
            elif violation.type == 'motif_drift' and target in self._motifs:
                current = self._motifs[target]['strength']
                attenuation = violation.attenuation
                new_str = max(0.0, current - attenuation)
                self._motifs[target]['strength'] = round(new_str, 4)
                self._motifs[target]['provenance'] = 'drift_attenuation'
                count += 1
                self.journal.append_event(
                    event_type='DRIFT_ATTENUATION',
                    detail=f"attenuated motif={target} from={current} to={new_str}",
                    epoch=self.epoch.current,
                )

        if count:
            logger.info(f"drift_attenuation_applied count={count}")
        return count

    # ------------------------------------------------------------------
    # GROUNDING — replay anchors
    # ------------------------------------------------------------------

    def create_grounding_checkpoint(self) -> str:
        """
        Create a grounding checkpoint — a snapshot of the current
        epistemic state for deterministic replay.

        Returns:
            checkpoint_id
        """
        return self.grounding.create(
            epoch=self.epoch.current,
            observation_count=self._observation_count,
            beliefs=dict(self._beliefs),
            motifs=dict(self._motifs),
            attractors=dict(self._attractors),
            journal_size=len(self.journal._events),
        )

    def list_grounding_checkpoints(self) -> list:
        """List all grounding checkpoints."""
        return self.grounding.list_checkpoints()

    # ------------------------------------------------------------------
    # INVARIANT VERIFICATION
    # ------------------------------------------------------------------

    def verify(self, *names: str) -> 'SemanticInvariantReport':
        """
        Run semantic invariants against current epistemic state.

        Pure operation — no state mutation.
        Mirrors ExecutionKernel.verify().

        Args:
            *names: specific invariants (all if empty)

        Returns:
            SemanticInvariantReport
        """
        if names:
            return self._invariants.verify(*names)
        return self._invariants.verify_all()

    def assert_invariants(self, *names: str):
        """Verify invariants and raise on violation."""
        from .semantic_invariants import SemanticInvariantViolationError
        report = self.verify(*names)
        if not report.passed:
            for v in report.violations:
                logger.error(
                    f"semantic_invariant_violation name={v.name} "
                    f"severity={v.severity.name} detail={v.detail}"
                )
            raise SemanticInvariantViolationError(report)

    # ------------------------------------------------------------------
    # SNAPSHOT — deterministic state export / restore / recovery
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        """
        Serialize full epistemic state for checkpointing.

        Deterministic: same state -> same serialized dict.
        MIRRORS ExecutionKernel.export_state().
        """
        return {
            'beliefs': {k: dict(v) for k, v in self._beliefs.items()},
            'motifs': {k: dict(v) for k, v in self._motifs.items()},
            'attractors': {k: dict(v) for k, v in self._attractors.items()},
            'epoch': self.epoch.current,
            'observation_count': self._observation_count,
            'provenance_entries': [e.to_dict() for e in self.provenance._entries],
            'journal_events': [e.to_dict() for e in self.journal._events],
            'grounding_checkpoints': self.grounding.list_checkpoints(),
            'journal_sequence': self.journal._sequence,
        }

    def restore_state(self, state: dict):
        """
        Restore full epistemic state from serialized dict.

        Deterministic: same dict -> identical in-memory state.
        MIRRORS ExecutionKernel.restore_state().
        """
        self._beliefs.clear()
        self._motifs.clear()
        self._attractors.clear()
        self.provenance._entries.clear()
        self.journal._events.clear()
        self.grounding._checkpoints.clear()

        for name, data in state.get('beliefs', {}).items():
            self._beliefs[name] = dict(data)
        for name, data in state.get('motifs', {}).items():
            self._motifs[name] = dict(data)
        for aid, data in state.get('attractors', {}).items():
            self._attractors[aid] = dict(data)

        self.epoch._current = state.get('epoch', 0)
        self._observation_count = state.get('observation_count', 0)

        from .provenance import ProvenanceEntry
        for entry_data in state.get('provenance_entries', []):
            entry = ProvenanceEntry(
                belief_name=entry_data['belief_name'],
                confidence_delta=entry_data['confidence_delta'],
                provenance_label=entry_data['provenance_label'],
                event_id=entry_data['event_id'],
                epoch=entry_data['epoch'],
                timestamp=entry_data['timestamp'],
            )
            self.provenance._entries.append(entry)

        from .semantic_journal import InterpretationEvent
        for event_data in state.get('journal_events', []):
            event = InterpretationEvent(
                event_id=event_data['event_id'],
                event_type=event_data['event_type'],
                timestamp=event_data['timestamp'],
                epoch=event_data['epoch'],
                detail=event_data.get('detail', ''),
                data=dict(event_data.get('data', {})),
                prev_event_id=event_data.get('prev_event_id'),
            )
            self.journal._events.append(event)

        self.journal._sequence = state.get('journal_sequence', len(self.journal._events))

        from .grounding import GroundingCheckpoint
        for cp_data in state.get('grounding_checkpoints', []):
            cp = GroundingCheckpoint(
                checkpoint_id=cp_data['checkpoint_id'],
                epoch=cp_data['epoch'],
                observation_count=cp_data['observation_count'],
                journal_size=cp_data['journal_size'],
                created_at=cp_data['created_at'],
                beliefs=dict(cp_data.get('beliefs', {})),
                motifs=dict(cp_data.get('motifs', {})),
                attractors=dict(cp_data.get('attractors', {})),
            )
            self.grounding._checkpoints.append(cp)

        logger.info(
            f"epistemic_state_restored "
            f"beliefs={len(self._beliefs)} "
            f"motifs={len(self._motifs)} "
            f"epoch={self.epoch.current}"
        )

    def recover(self) -> dict:
        """
        Deterministic recovery from journal replay.

        RECOVERY ALGORITHM:
          1. Clear all in-memory state (beliefs, motifs, attractors)
          2. Replay all journal events in order
          3. For each event, reconstruct the corresponding state
          4. Restore epoch from last event
          5. Rebuild provenance graph
          6. Return recovery summary

        Determinism guarantee: same journal -> identical state.
        MIRRORS ExecutionKernel.recover().
        """
        recovered = {
            'state_cleared': True,
            'journal_events_replayed': 0,
            'beliefs_recovered': 0,
            'motifs_recovered': 0,
            'attractors_recovered': 0,
            'observations_recovered': 0,
            'provenance_entries_recovered': 0,
            'final_epoch': 0,
            'status': 'ok',
        }

        self._beliefs.clear()
        self._motifs.clear()
        self._attractors.clear()
        self._observation_count = 0
        self.epoch._current = 0
        self.provenance._entries.clear()

        import re
        _re_belief = re.compile(r'belief=(\S+)\s+from=[\d.]+\s+to=([\d.]+)')
        _re_motif = re.compile(r'motif=(\S+)\s+from=[\d.]+\s+to=([\d.]+)')
        touch_events = 0  # only event types that call epoch.touch()
        for event in self.journal._events:
            event_type = event.event_type
            data = event.data or {}

            if event_type == 'BELIEF_UPDATE':
                touch_events += 1
                name = data.get('belief_name', event.detail)
                new_conf = data.get('new_confidence', 0.0)
                provenance = data.get('provenance', 'journal_replay')
                self._beliefs[name] = {
                    'confidence': round(new_conf, 4),
                    'updated_at_epoch': event.epoch,
                    'updated_at_event': event.event_id,
                    'provenance': provenance,
                }
                recovered['beliefs_recovered'] += 1

            elif event_type == 'MOTIF_UPDATE':
                touch_events += 1
                name = data.get('motif_name', event.detail)
                new_str = data.get('new_strength', 0.0)
                recurrence = data.get('recurrence', 1)
                provenance = data.get('provenance', 'journal_replay')
                self._motifs[name] = {
                    'strength': round(new_str, 4),
                    'recurrence': recurrence,
                    'updated_at_epoch': event.epoch,
                    'provenance': provenance,
                }
                recovered['motifs_recovered'] += 1

            elif event_type == 'ATTRACTOR_UPDATE':
                touch_events += 1
                aid = data.get('attractor_id', event.detail)
                weight = data.get('weight', 0.0)
                label = data.get('label', aid)
                self._attractors[aid] = {
                    'weight': round(weight, 4),
                    'label': label,
                    'updated_at_epoch': event.epoch,
                }
                recovered['attractors_recovered'] += 1

            elif event_type == 'OBSERVATION':
                touch_events += 1
                self._observation_count += 1
                recovered['observations_recovered'] += 1

            elif event_type == 'DRIFT_ATTENUATION':
                detail = event.detail or ''
                bm = _re_belief.search(detail)
                if bm and bm.group(1) in self._beliefs:
                    self._beliefs[bm.group(1)]['confidence'] = round(float(bm.group(2)), 4)
                    self._beliefs[bm.group(1)]['provenance'] = 'drift_attenuation'
                mm = _re_motif.search(detail)
                if mm and mm.group(1) in self._motifs:
                    self._motifs[mm.group(1)]['strength'] = round(float(mm.group(2)), 4)
                    self._motifs[mm.group(1)]['provenance'] = 'drift_attenuation'

            if event_type in ('BELIEF_UPDATE', 'MOTIF_UPDATE'):
                name = data.get('belief_name', data.get('motif_name', 'unknown'))
                prev = data.get('previous_confidence', data.get('previous_strength', 0.0))
                curr = data.get('new_confidence', data.get('new_strength', 0.0))
                self.provenance.record_belief_provenance(
                    belief_name=name,
                    confidence_delta=round(curr - prev, 4),
                    provenance_label=data.get('provenance', 'journal_replay'),
                    event_id=event.event_id,
                    epoch=event.epoch,
                )
                recovered['provenance_entries_recovered'] += 1

            if event.epoch > self.epoch._current:
                self.epoch._current = event.epoch

        # Only event types that call epoch.touch() advance the counter
        self.epoch._current = touch_events

        recovered['journal_events_replayed'] = len(self.journal._events)
        recovered['final_epoch'] = self.epoch.current
        recovered['status'] = 'ok'

        logger.info(
            f"epistemic_recovery_complete "
            f"events={recovered['journal_events_replayed']} "
            f"beliefs={recovered['beliefs_recovered']} "
            f"epoch={recovered['final_epoch']}"
        )

        return recovered

    # ------------------------------------------------------------------
    # PROVENANCE QUERIES
    # ------------------------------------------------------------------

    def get_belief_provenance(self, belief_name: str) -> list:
        """Get full provenance chain for a belief."""
        return self.provenance.get_belief_chain(belief_name)

    def get_provenance_graph(self) -> dict:
        """Get full provenance graph."""
        return self.provenance.get_graph()

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Full epistemic kernel diagnostics."""
        violations = []
        for name in ('belief_confidence_in_bounds', 'motif_strength_in_bounds',
                      'observation_precedes_belief'):
            v = self._invariants.verify(name)
            if v is not None:
                violations.append(v)
        inv_report = SemanticInvariantReport(
            invariants_checked=3,
            invariants_passed=3 - len(violations),
            violations=violations,
            fatal=False,
        )
        return {
            'epoch': self.epoch.current,
            'observation_count': self._observation_count,
            'journal_events': len(self.journal._events),
            'beliefs': len(self._beliefs),
            'motifs': len(self._motifs),
            'attractors': len(self._attractors),
            'provenance_entries': len(self.provenance._entries),
            'grounding_checkpoints': len(self.grounding._checkpoints),
            'drift_violations': len(self.drift._violation_history),
            'invariants': inv_report.to_dict(),
        }
