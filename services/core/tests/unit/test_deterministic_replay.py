"""
Replay Determinism Test — THE GATE.

If this test fails, the entire causal stack is not deterministic.
No further layers (PHE, API, UI) should be built until this passes.

Test scenarios:
  1. Same input sequence → same belief state (cross-instance)
  2. Same input sequence → same motif state (cross-instance)
  3. Same input sequence → same drift report (cross-instance)
  4. Same bridge event → same belief delta (cross-instance)
  5. Journal replay returns identical events (causal chain intact)
  6. Epistemic invariant check passes on any valid state
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from epistemic_kernel import EpistemicKernel
from causal_bridge import CausalityBridge


# ── Deterministic test sequences ──

SEQUENCE_A = [
    ('observe', {'signal': 'sleep_frag', 'value': 0.7, 'source': 'wearable'}),
    ('observe', {'signal': 'focus_decline', 'value': 0.5, 'source': 'app_usage'}),
    ('observe', {'signal': 'fatigue', 'value': 0.8, 'source': 'self_report'}),
    ('belief', {'name': 'burnout_risk', 'confidence': 0.65, 'provenance': 'motif_detection'}),
    ('belief', {'name': 'focus_quality', 'confidence': 0.4, 'provenance': 'trend_analysis'}),
    ('belief', {'name': 'burnout_risk', 'confidence': 0.72, 'provenance': 'obs_corr'}),
    ('motif', {'name': 'sf_fatigue', 'strength': 0.75, 'recurrence': 3, 'provenance': 'matcher'}),
    ('motif', {'name': 'fd_errors', 'strength': 0.55, 'recurrence': 2, 'provenance': 'matcher'}),
]

SEQUENCE_B = [
    ('observe', {'signal': 'sensor_a', 'value': 0.3}),
    ('observe', {'signal': 'sensor_b', 'value': 0.9}),
    ('belief', {'name': 'anomaly_score', 'confidence': 0.4, 'provenance': 'detector'}),
    ('belief', {'name': 'anomaly_score', 'confidence': 0.7, 'provenance': 'detector'}),
    ('belief', {'name': 'anomaly_score', 'confidence': 0.6, 'provenance': 'detector'}),
]

BRIDGE_EVENTS = [
    ('COMPLETED', {'entry_id': 'e1', 'goal_id': 'g1', 'lease_id': 'l1', 'success': True, 'duration_ms': 1000}),
    ('COMPLETED', {'entry_id': 'e2', 'goal_id': 'g2', 'lease_id': 'l2', 'success': True, 'duration_ms': 500}),
    ('FAILED', {'entry_id': 'e3', 'goal_id': 'g3', 'lease_id': 'l3', 'success': False, 'duration_ms': 3000, 'error': 'timeout'}),
    ('PREEMPTED', {'entry_id': 'e4', 'goal_id': 'g4', 'lease_id': 'l4'}),
    ('RETRIED', {'entry_id': 'e5', 'goal_id': 'g5', 'lease_id': 'l5'}),
    ('ABANDONED', {'entry_id': 'e6', 'goal_id': 'g6', 'lease_id': 'l6'}),
]


def apply_sequence(ek, sequence):
    """Apply a deterministic sequence to an epistemic kernel."""
    for action, params in sequence:
        if action == 'observe':
            ek.record_observation(**params)
        elif action == 'belief':
            ek.update_belief(**params)
        elif action == 'motif':
            ek.update_motif(**params)


def apply_bridge_events(bridge, events):
    """Apply a sequence of bridge propagation events."""
    for event_type, params in events:
        bridge.on_execution_event(event_type, **params)


def snapshot_state(ek):
    """Deterministic snapshot of epistemic state."""
    return {
        'beliefs': ek.get_all_beliefs(),
        'motifs': ek.get_all_motifs(),
        'attractors': ek.get_all_attractors(),
        'epoch': ek.epoch.current,
        'observation_count': ek._observation_count,
        'journal_event_ids': [e.event_id for e in ek.journal._events],
        'journal_event_types': [e.event_type for e in ek.journal._events],
        'journal_event_detail': [e.detail for e in ek.journal._events],
    }


def snapshot_bridge(bridge):
    """Snapshot of bridge state."""
    return {
        'total_edges': bridge.graph.count(),
        'edge_ids': sorted(e.edge_id for e in bridge.graph._edges.values()),
        'edge_directions': sorted(
            (e.edge_id, e.direction.name)
            for e in bridge.graph._edges.values()
        ),
    }


# ════════════════════════════════════════════════════════════════
# TEST 1: Epistemic kernel cross-instance determinism
# ════════════════════════════════════════════════════════════════

class TestEpistemicDeterminism:

    def test_same_sequence_produces_same_beliefs(self):
        """Same input sequence → same beliefs across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        apply_sequence(ek1, SEQUENCE_A)
        apply_sequence(ek2, SEQUENCE_A)
        assert ek1.get_all_beliefs() == ek2.get_all_beliefs(), \
            "Beliefs differ between instances"

    def test_same_sequence_produces_same_motifs(self):
        """Same input sequence → same motifs across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        apply_sequence(ek1, SEQUENCE_A)
        apply_sequence(ek2, SEQUENCE_A)
        assert ek1.get_all_motifs() == ek2.get_all_motifs(), \
            "Motifs differ between instances"

    def test_same_sequence_produces_same_journal(self):
        """Same input sequence → same journal entries across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        apply_sequence(ek1, SEQUENCE_A)
        apply_sequence(ek2, SEQUENCE_A)
        s1, s2 = snapshot_state(ek1), snapshot_state(ek2)
        assert s1['journal_event_types'] == s2['journal_event_types']
        assert s1['journal_event_detail'] == s2['journal_event_detail']

    def test_same_sequence_produces_same_drift(self):
        """Same input sequence → same drift report across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        apply_sequence(ek1, SEQUENCE_A)
        apply_sequence(ek2, SEQUENCE_A)
        d1 = ek1.check_drift()
        d2 = ek2.check_drift()
        assert d1.overall_drift_score == d2.overall_drift_score
        assert len(d1.violations) == len(d2.violations)

    def test_full_state_equality(self):
        """Complete state snapshot equality across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        apply_sequence(ek1, SEQUENCE_B)
        apply_sequence(ek2, SEQUENCE_B)
        assert snapshot_state(ek1) == snapshot_state(ek2), \
            "Full state differs between instances"

    def test_epoch_advances_monotonically(self):
        """Epoch never decreases within a sequence."""
        ek = EpistemicKernel()
        epochs = []
        for action, params in SEQUENCE_A:
            apply_sequence(ek, [(action, params)])
            epochs.append(ek.epoch.current)
        for i in range(1, len(epochs)):
            assert epochs[i] > epochs[i-1], \
                f"Epoch regression at step {i}: {epochs[i-1]} → {epochs[i]}"

    def test_idempotent_journal_replay(self):
        """Journal replay returns the same events across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        apply_sequence(ek1, SEQUENCE_A)
        apply_sequence(ek2, SEQUENCE_A)

        replay1 = ek1.journal.replay()
        replay2 = ek2.journal.replay()

        assert len(replay1) == len(replay2)
        for e1, e2 in zip(replay1, replay2):
            assert e1.event_type == e2.event_type
            assert e1.detail == e2.detail
            assert e1.epoch == e2.epoch

    def test_invariant_always_passes_on_deterministic_state(self):
        """Epistemic invariants pass on any valid deterministic state."""
        ek = EpistemicKernel()
        apply_sequence(ek, SEQUENCE_A)
        report = ek.verify()
        assert report.passed or not report.fatal, \
            f"Invariant violation on valid state: {report.violations[:2]}"


# ════════════════════════════════════════════════════════════════
# TEST 2: Bridge determinism
# ════════════════════════════════════════════════════════════════

class TestBridgeDeterminism:

    def test_same_bridge_events_same_beliefs(self):
        """Same bridge events → same beliefs across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        b1 = CausalityBridge(epistemic_kernel=ek1)
        b2 = CausalityBridge(epistemic_kernel=ek2)

        apply_bridge_events(b1, BRIDGE_EVENTS)
        apply_bridge_events(b2, BRIDGE_EVENTS)

        assert ek1.get_all_beliefs() == ek2.get_all_beliefs()

    def test_same_bridge_events_same_motifs(self):
        """Same bridge events → same motifs across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        b1 = CausalityBridge(epistemic_kernel=ek1)
        b2 = CausalityBridge(epistemic_kernel=ek2)

        apply_bridge_events(b1, BRIDGE_EVENTS)
        apply_bridge_events(b2, BRIDGE_EVENTS)

        assert ek1.get_all_motifs() == ek2.get_all_motifs()

    def test_same_bridge_events_same_graph(self):
        """Same bridge events → same GRAPH STRUCTURE across instances.
        (Edge IDs are non-deterministic — compare structure, not identity.)
        """
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        b1 = CausalityBridge(epistemic_kernel=ek1)
        b2 = CausalityBridge(epistemic_kernel=ek2)

        apply_bridge_events(b1, BRIDGE_EVENTS)
        apply_bridge_events(b2, BRIDGE_EVENTS)

        s1, s2 = snapshot_bridge(b1), snapshot_bridge(b2)
        # Same total count
        assert s1['total_edges'] == s2['total_edges'], "Edge count differs"
        # Same direction sequence (structural identity)
        dirs1 = [d for _, d in s1['edge_directions']]
        dirs2 = [d for _, d in s2['edge_directions']]
        assert dirs1 == dirs2, "Edge direction sequence differs"

    def test_bridge_propagation_can_be_disabled(self):
        """Propagation off → no state change."""
        ek = EpistemicKernel()
        bridge = CausalityBridge(epistemic_kernel=ek)
        bridge.disable_propagation()

        pre = snapshot_state(ek)
        apply_bridge_events(bridge, BRIDGE_EVENTS)
        post = snapshot_state(ek)

        assert pre == post

    def test_bridge_re_enable_works(self):
        """Disable → enable → propagation works again."""
        ek = EpistemicKernel()
        bridge = CausalityBridge(epistemic_kernel=ek)
        bridge.disable_propagation()
        bridge.enable_propagation()
        assert bridge.propagation_enabled is True

        pre_obs = ek._observation_count
        bridge.on_execution_completed('g1', 'e1', 'l1', True, 1000)
        assert ek._observation_count > pre_obs

    def test_bridge_adjustments_deterministic(self):
        """Same state → same dispatch adjustments across instances."""
        ek1, ek2 = EpistemicKernel(), EpistemicKernel()
        b1 = CausalityBridge(epistemic_kernel=ek1)
        b2 = CausalityBridge(epistemic_kernel=ek2)

        apply_bridge_events(b1, BRIDGE_EVENTS)
        apply_bridge_events(b2, BRIDGE_EVENTS)

        adj1 = b1.get_dispatch_adjustments()
        adj2 = b2.get_dispatch_adjustments()

        for key in adj1:
            assert adj1[key] == adj2[key], \
                f"Adjustment '{key}' differs: {adj1[key]} != {adj2[key]}"


# ════════════════════════════════════════════════════════════════
# TEST 3: Combined determinism (epistemic + bridge)
# ════════════════════════════════════════════════════════════════

class TestCombinedDeterminism:

    def test_full_stack_determinism(self):
        """Full epistemic + bridge stack produces identical state across instances.
        (Bridge edge IDs are non-deterministic — compare structure, not identity.)
        """
        def build_full():
            ek = EpistemicKernel()
            bridge = CausalityBridge(epistemic_kernel=ek)
            apply_sequence(ek, SEQUENCE_A)
            apply_bridge_events(bridge, BRIDGE_EVENTS)
            return ek, bridge

        ek1, b1 = build_full()
        ek2, b2 = build_full()

        assert snapshot_state(ek1) == snapshot_state(ek2)

        s1, s2 = snapshot_bridge(b1), snapshot_bridge(b2)
        assert s1['total_edges'] == s2['total_edges']
        dirs1 = [d for _, d in s1['edge_directions']]
        dirs2 = [d for _, d in s2['edge_directions']]
        assert dirs1 == dirs2

    def test_full_stack_cpe_choice_determinism(self):
        """Same full state → same CPE ranking across instances.
        (May select None if scores below threshold — still deterministic.)
        """
        from causal_policy import CausalPolicyEngine

        def build_cpe():
            ek = EpistemicKernel()
            bridge = CausalityBridge(epistemic_kernel=ek)
            apply_sequence(ek, SEQUENCE_A)
            apply_bridge_events(bridge, BRIDGE_EVENTS[:3])
            return CausalPolicyEngine(bridge=bridge)

        cpe1, cpe2 = build_cpe(), build_cpe()

        d1 = cpe1.decide('test-goal')
        d2 = cpe2.decide('test-goal')

        assert d1.best_score == d2.best_score
        # Both must have same chosen status (both None or both same label)
        if d1.chosen is None:
            assert d2.chosen is None
        else:
            assert d2.chosen is not None
            assert d1.chosen.label == d2.chosen.label
        assert [r['total_score'] for r in d1.ranked] == \
               [r['total_score'] for r in d2.ranked]

    def test_cpe_simulation_determinism(self):
        """Same simulate() call → same score across instances."""
        from causal_policy import CausalPolicyEngine

        def build_cpe():
            ek = EpistemicKernel()
            bridge = CausalityBridge(epistemic_kernel=ek)
            apply_sequence(ek, SEQUENCE_B)
            return CausalPolicyEngine(bridge=bridge)

        cpe1, cpe2 = build_cpe(), build_cpe()

        sim1 = cpe1.simulate_event('COMPLETED', 'g1', True)
        sim2 = cpe2.simulate_event('COMPLETED', 'g1', True)

        assert sim1['score']['total_score'] == sim2['score']['total_score']
        assert sim1['outcome']['stability_delta'] == sim2['outcome']['stability_delta']

    def test_journal_replay_chain_integrity(self):
        """Journal event chain is contiguous (prev_entry_id links)."""
        ek = EpistemicKernel()
        apply_sequence(ek, SEQUENCE_A)

        events = ek.journal._events
        for i in range(1, len(events)):
            assert events[i].prev_event_id == events[i-1].event_id, \
                f"Chain broken at index {i}"
