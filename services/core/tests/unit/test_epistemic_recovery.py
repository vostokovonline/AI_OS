"""
EpistemicKernel Recovery Tests — verifies determinism of
export_state / restore_state / recover().

MIRRORS execution kernel recovery tests in test_execution_recovery.py.

Test categories:
  1. export_state() → restore_state() round-trip
  2. recover() from journal replay produces identical state
  3. Crash recovery: export → modify → restore → verify original
"""

import pytest
from epistemic_kernel import EpistemicKernel


def populated_kernel():
    """Build an epistemic kernel with diverse events."""
    ek = EpistemicKernel()
    ek.record_observation("sleep_fragmentation", value=0.7, source="wearable")
    ek.record_observation("focus_duration", value=0.4, source="app_usage")
    ek.update_belief("burnout_risk", 0.65, provenance="motif_detection")
    ek.update_belief("focus_quality", 0.45, provenance="observation")
    ek.update_motif("fatigue_signal", 0.72, recurrence=3, provenance="pattern_match")
    ek.update_motif("productivity_drop", 0.55, recurrence=2, provenance="trend_detection")
    ek.update_attractor("low_energy", weight=0.6, label="Low Energy State")
    ek.update_attractor("high_focus", weight=0.3, label="High Focus State")
    ek.update_belief("burnout_risk", 0.58, provenance="observation_decay")
    ek.create_grounding_checkpoint()
    ek.update_belief("stress_level", 0.5, provenance="self_report")
    dr = ek.check_drift()
    if dr.violations:
        ek.attenuate_drift(dr)
    return ek


class TestExportRestoreRoundtrip:

    def test_export_restore_produces_identical_beliefs(self):
        ek1 = populated_kernel()
        state = ek1.export_state()

        ek2 = EpistemicKernel()
        ek2.restore_state(state)

        assert ek1.get_all_beliefs() == ek2.get_all_beliefs()

    def test_export_restore_produces_identical_motifs(self):
        ek1 = populated_kernel()
        state = ek1.export_state()

        ek2 = EpistemicKernel()
        ek2.restore_state(state)

        assert ek1.get_all_motifs() == ek2.get_all_motifs()

    def test_export_restore_produces_identical_attractors(self):
        ek1 = populated_kernel()
        state = ek1.export_state()

        ek2 = EpistemicKernel()
        ek2.restore_state(state)

        assert ek1.get_all_attractors() == ek2.get_all_attractors()

    def test_export_restore_produces_identical_epoch(self):
        ek1 = populated_kernel()
        state = ek1.export_state()

        ek2 = EpistemicKernel()
        ek2.restore_state(state)

        assert ek1.epoch.current == ek2.epoch.current

    def test_export_restore_produces_identical_journal(self):
        ek1 = populated_kernel()
        state = ek1.export_state()

        ek2 = EpistemicKernel()
        ek2.restore_state(state)

        j1 = [e.to_dict() for e in ek1.journal._events]
        j2 = [e.to_dict() for e in ek2.journal._events]
        assert len(j1) == len(j2)
        for e1, e2 in zip(j1, j2):
            assert e1['event_type'] == e2['event_type']
            assert e1['epoch'] == e2['epoch']
            assert e1['detail'] == e2['detail']

    def test_export_restore_full_state_equality(self):
        ek1 = populated_kernel()
        state = ek1.export_state()

        ek2 = EpistemicKernel()
        ek2.restore_state(state)

        s1 = ek1.get_stats()
        s2 = ek2.get_stats()
        for key in ('epoch', 'observation_count', 'journal_events',
                     'beliefs', 'motifs', 'attractors', 'provenance_entries',
                     'grounding_checkpoints'):
            assert s1[key] == s2[key], f"Mismatch on {key}: {s1[key]} != {s2[key]}"


class TestRecoverFromJournal:

    def test_recover_produces_identical_beliefs(self):
        original = populated_kernel()

        recovered = EpistemicKernel()
        # Copy the journal events to the recovered kernel
        recovered.journal._events = list(original.journal._events)
        recovered.journal._sequence = original.journal._sequence
        result = recovered.recover()

        assert result['status'] == 'ok'
        assert result['beliefs_recovered'] > 0
        assert original.get_all_beliefs() == recovered.get_all_beliefs()

    def test_recover_produces_identical_motifs(self):
        original = populated_kernel()

        recovered = EpistemicKernel()
        recovered.journal._events = list(original.journal._events)
        recovered.journal._sequence = original.journal._sequence
        recovered.recover()

        assert original.get_all_motifs() == recovered.get_all_motifs()

    def test_recover_produces_identical_attractors(self):
        original = populated_kernel()

        recovered = EpistemicKernel()
        recovered.journal._events = list(original.journal._events)
        recovered.journal._sequence = original.journal._sequence
        recovered.recover()

        assert original.get_all_attractors() == recovered.get_all_attractors()

    def test_recover_produces_identical_epoch(self):
        original = populated_kernel()

        recovered = EpistemicKernel()
        recovered.journal._events = list(original.journal._events)
        recovered.journal._sequence = original.journal._sequence
        recovered.recover()

        assert original.epoch.current == recovered.epoch.current

    def test_recover_restores_observation_count(self):
        original = populated_kernel()

        recovered = EpistemicKernel()
        recovered.journal._events = list(original.journal._events)
        recovered.journal._sequence = original.journal._sequence
        recovered.recover()

        assert original._observation_count == recovered._observation_count

    def test_recover_empty_journal(self):
        """Empty journal → empty state after recover."""
        ek = EpistemicKernel()
        result = ek.recover()

        assert result['status'] == 'ok'
        assert result['journal_events_replayed'] == 0
        assert ek.get_all_beliefs() == {}
        assert ek.get_all_motifs() == {}
        assert ek.get_all_attractors() == {}


class TestCrashRecovery:

    def test_export_modify_restore_recovers_original(self):
        """export → modify → restore → state matches original export."""
        ek = populated_kernel()
        original_state = ek.export_state()

        # Modify kernel
        ek.update_belief("burnout_risk", 0.99, provenance="crash_test")
        ek.update_motif("fatigue_signal", 0.99, recurrence=99, provenance="crash_test")
        ek.record_observation("noise", value=1.0, source="crash_test")

        # Restore original
        ek.restore_state(original_state)

        # Verify restored to original
        original = EpistemicKernel()
        original.restore_state(original_state)
        assert ek.get_all_beliefs() == original.get_all_beliefs()
        assert ek.get_all_motifs() == original.get_all_motifs()
        assert ek.epoch.current == original.epoch.current

    def test_export_restore_twice_is_idempotent(self):
        """restore() twice → same state both times."""
        ek = EpistemicKernel()
        ek.record_observation("test", value=0.5, source="test")
        ek.update_belief("test_belief", 0.7, provenance="test")
        state = ek.export_state()

        # Restore twice
        ek2 = EpistemicKernel()
        ek2.restore_state(state)
        first = ek2.export_state()
        ek2.restore_state(state)
        second = ek2.export_state()

        assert first == second

    def test_cross_instance_determinism(self):
        """Same events → same state after recover() on two instances."""
        events = None

        def build():
            ek = EpistemicKernel()
            ek.record_observation("a", value=0.5, source="s1")
            ek.record_observation("b", value=0.3, source="s2")
            ek.update_belief("x", 0.8, provenance="p1")
            ek.update_motif("m1", 0.6, recurrence=2, provenance="p2")
            ek.update_attractor("at1", weight=0.7, label="Test")
            ek.update_belief("x", 0.75, provenance="p3")
            return ek

        e1, e2 = build(), build()
        e2_recovered = EpistemicKernel()
        e2_recovered.journal._events = list(e2.journal._events)
        e2_recovered.journal._sequence = e2.journal._sequence
        e2_recovered.recover()

        assert e1.get_all_beliefs() == e2_recovered.get_all_beliefs()
        assert e1.get_all_motifs() == e2_recovered.get_all_motifs()
        assert e1.get_all_attractors() == e2_recovered.get_all_attractors()
        assert e1.epoch.current == e2_recovered.epoch.current
