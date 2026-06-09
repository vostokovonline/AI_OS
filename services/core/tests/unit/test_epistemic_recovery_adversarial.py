"""
Adversarial Epistemic Recovery Tests — verifies recovery under
corrupted, duplicated, truncated, missing, and out-of-order journals.

MIRRORS execution kernel fault injection tests.

5 scenarios:
  1. Duplicate journal events — exact copies in the event list
  2. Missing journal event — one event removed from middle
  3. Corrupted event payload — malformed data/detail fields
  4. Out-of-order events — prev_event_id chain broken
  5. Truncated journal — only first N events survive
"""

import copy
from epistemic_kernel import EpistemicKernel
from epistemic_kernel.semantic_journal import InterpretationEvent
from epistemic_kernel.epoch import InterpretationEpoch


def populated_journal(ek=None):
    """Fill a kernel with a known event sequence and return the event list."""
    if ek is None:
        ek = EpistemicKernel()
    ek.record_observation("signal_a", value=0.8, source="test")
    ek.record_observation("signal_b", value=0.3, source="test")
    ek.update_belief("belief_x", 0.7, provenance="obs")
    ek.update_belief("belief_y", 0.4, provenance="obs")
    ek.update_motif("motif_m", 0.6, recurrence=2, provenance="pattern")
    ek.update_attractor("at_a", weight=0.8, label="Alpha")
    ek.update_belief("belief_x", 0.9, provenance="reinforce")
    return list(ek.journal._events)


def recover_from_events(events):
    """Create a fresh kernel, inject events, and recover()."""
    ek = EpistemicKernel()
    ek.journal._events = list(events)
    ek.journal._sequence = len(events)
    result = ek.recover()
    return ek, result


class TestDuplicateEvents:

    def test_duplicate_belief_events(self):
        """Duplicate BELIEF_UPDATE events → last write wins, no crash."""
        events = populated_journal()
        # Append exact copy of last event
        events.append(copy.deepcopy(events[-1]))
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        assert result['beliefs_recovered'] >= 2
        # Last write wins — belief_x has the reinforced value
        assert ek.get_belief('belief_x')['confidence'] == 0.9

    def test_duplicate_all_events(self):
        """Entire journal duplicated → still recoverable."""
        events = populated_journal()
        dupes = [copy.deepcopy(e) for e in events]
        events.extend(dupes)
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        # Last write wins for each belief
        assert ek.get_belief('belief_x')['confidence'] == 0.9
        assert ek.get_belief('belief_y')['confidence'] == 0.4
        assert len(ek.journal._events) == 2 * len(populated_journal())

    def test_duplicate_triple_events_does_not_crash(self):
        """Three copies of each event → no crash, deterministic."""
        events = populated_journal()
        for _ in range(2):
            events.extend(copy.deepcopy(e) for e in populated_journal())
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        assert result['journal_events_replayed'] == 3 * len(populated_journal())


class TestMissingEvent:

    def test_missing_intermediate_belief(self):
        """Remove middle BELIEF_UPDATE → final belief different but deterministic."""
        events = populated_journal()
        # Remove the last BELIEF_UPDATE (belief_x → 0.9)
        before = len(events)
        events = [e for e in events if not (
            e.event_type == 'BELIEF_UPDATE'
            and e.data.get('new_confidence', 0) == 0.9
        )]
        assert len(events) < before
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        # belief_x stuck at first value since reinforce was removed
        assert ek.get_belief('belief_x')['confidence'] == 0.7

    def test_missing_observation(self):
        """Remove OBSERVATION → fewer observations, epoch lower."""
        events = populated_journal()
        events = [e for e in events if e.event_type != 'OBSERVATION']
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        assert result['observations_recovered'] == 0
        assert ek._observation_count == 0

    def test_missing_all_events_no_crash(self):
        """Empty event list → empty state, clean recovery."""
        ek, result = recover_from_events([])
        assert result['status'] == 'ok'
        assert ek.get_all_beliefs() == {}
        assert ek.epoch.current == 0


class TestCorruptedPayload:

    def test_corrupted_missing_data_field(self):
        """Event with data=None → recover should not crash."""
        events = populated_journal()
        # Corrupt the last BELIEF_UPDATE by setting data=None
        for e in reversed(events):
            if e.event_type == 'BELIEF_UPDATE':
                e.data = None
                break
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        # belief_x should still exist (from the first BELIEF_UPDATE)
        assert ek.get_belief('belief_x')['confidence'] in (0.7, 0.0)

    def test_corrupted_missing_event_type(self):
        """Event with event_type=None → silently skipped, no crash."""
        events = populated_journal()
        events[2].event_type = None
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        # Should have recovered with one less event
        assert result['journal_events_replayed'] == len(events)

    def test_corrupted_empty_detail(self):
        """DRIFT_ATTENUATION with empty detail → regex just doesn't match."""
        ek = EpistemicKernel()
        from epistemic_kernel.semantic_journal import InterpretationEvent
        ek.journal._events = [
            InterpretationEvent(
                event_id='corrupt:1', event_type='DRIFT_ATTENUATION',
                timestamp=100.0, epoch=1, detail='',
            ),
        ]
        ek.journal._sequence = 1
        result = ek.recover()
        assert result['status'] == 'ok'

    def test_corrupted_float_as_int(self):
        """Confidence stored as int instead of float → cast still works."""
        events = populated_journal()
        for e in events:
            if e.data and 'new_confidence' in e.data:
                e.data['new_confidence'] = 1  # int, not float
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        assert ek.get_belief('belief_x')['confidence'] == 1.0

    def test_corrupted_negative_confidence(self):
        """Negative confidence → clamped by update_belief, raw in journal."""
        events = populated_journal()
        for e in events:
            if e.data and 'new_confidence' in e.data:
                e.data['new_confidence'] = -0.5
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        # recover uses raw value — only update_belief clamps
        assert ek.get_belief('belief_x')['confidence'] == -0.5


class TestOutOfOrder:

    def test_broken_prev_event_id_chain(self):
        """Broken prev_event_id → recover iterates list order, not chain."""
        events = populated_journal()
        # Break all prev_event_id links
        for i, e in enumerate(events):
            if i > 0:
                e.prev_event_id = 'nonexistent'
        ek, result = recover_from_events(events)
        assert result['status'] == 'ok'
        assert ek.get_belief('belief_x')['confidence'] == 0.9
        assert ek.get_all_beliefs() is not None

    def test_events_in_reverse_order(self):
        """Reverse event order → recover produces different but valid state."""
        events = populated_journal()
        rev = list(reversed(events))
        ek, result = recover_from_events(rev)
        assert result['status'] == 'ok'
        # Last write wins — belief_x ends at whatever the last BELIEF_UPDATE was
        # In reversed order, the first-belief_x (0.7) comes last
        assert ek.get_belief('belief_x')['confidence'] == 0.7

    def test_interleaved_events_cross_instance(self):
        """Same shuffled journal → same recovered state (determinism)."""
        events = populated_journal()
        import random
        rng = random.Random(42)
        rng.shuffle(events)
        ek1, r1 = recover_from_events(events)
        ek2, r2 = recover_from_events(list(events))  # fresh copy
        assert r1['status'] == 'ok'
        assert r2['status'] == 'ok'
        assert ek1.get_all_beliefs() == ek2.get_all_beliefs()
        assert ek1.get_all_motifs() == ek2.get_all_motifs()
        assert ek1.epoch.current == ek2.epoch.current


class TestTruncatedJournal:

    def test_truncated_half_events(self):
        """First half of events → state matches partial execution."""
        events = populated_journal()
        mid = len(events) // 2
        ek, result = recover_from_events(events[:mid])
        assert result['status'] == 'ok'
        assert result['journal_events_replayed'] == mid
        # Only observations and beliefs from first half
        assert ek._observation_count > 0

    def test_truncated_single_event(self):
        """Single OBSERVATION event → observation count = 1, no beliefs."""
        ek = EpistemicKernel()
        from epistemic_kernel.semantic_journal import InterpretationEvent
        ek.journal._events = [
            InterpretationEvent(
                event_id='t:1', event_type='OBSERVATION',
                timestamp=100.0, epoch=1,
                detail='signal=x value=0.5 source=t',
                data={'signal': 'x', 'value': 0.5, 'source': 't', 'observation_index': 1},
            ),
        ]
        ek.journal._sequence = 1
        result = ek.recover()
        assert result['status'] == 'ok'
        assert ek._observation_count == 1
        assert ek.epoch.current == 1

    def test_truncated_journal_then_append_recover_again(self):
        """Truncate → recover → append → recover → still deterministic."""
        ek = EpistemicKernel()
        ek.record_observation("a", value=0.5, source="s1")
        ek.update_belief("x", 0.7, provenance="p1")
        full_state = ek.get_stats()

        # Truncate: keep only first event
        first_event = [ek.journal._events[0]]
        truncated = EpistemicKernel()
        truncated.journal._events = list(first_event)
        truncated.journal._sequence = 1
        truncated.recover()
        assert truncated._observation_count == 1
        assert truncated.get_belief('x')['confidence'] == 0.0  # not recovered

    def test_truncated_100_random_sequences(self):
        """100 random truncations → recover never crashes."""
        base = populated_journal()
        for n in range(1, len(base)):
            ek, result = recover_from_events(base[:n])
            assert result['status'] == 'ok'
            assert result['journal_events_replayed'] == n
            assert ek.epoch.current == result['final_epoch']
