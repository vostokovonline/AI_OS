"""
Simple test runner for SafeAutoTuner (no pytest required)
"""
import sys
sys.path.insert(0, '/app')

from uuid import uuid4
from application.policies.safe_auto_tuner import SafeAutoTuner, TuningMode
from application.policies.arbitration_trace import ArbitrationTrace
from application.policies.decision_policies import (
    PassThroughPolicy,
    GreedyUtilityPolicy,
    ScoredIntent
)
from application.policies.decision_feedback import DecisionFeedback


def test_register_policy():
    """Test policy registration"""
    tuner = SafeAutoTuner()
    
    tuner.register_policy(
        "GreedyUtilityPolicy",
        {"budget": 10},
        mode=TuningMode.SUGGEST
    )
    
    state = tuner.get_state("GreedyUtilityPolicy")
    
    assert state is not None, "State should not be None"
    assert state["current_params"] == {"budget": 10}, f"Expected budget=10, got {state['current_params']}"
    assert state["mode"] == "suggest", f"Expected suggest mode, got {state['mode']}"
    print("✓ test_register_policy passed")


def test_step_limit():
    """Test max step size is enforced"""
    tuner = SafeAutoTuner()
    
    old = {"budget": 10}
    new = {"budget": 20}
    
    limited = tuner._apply_step_limit(old, new, max_step=2)
    
    assert limited["budget"] == 12, f"Expected 12, got {limited['budget']}"
    print("✓ test_step_limit passed")


def test_step_limit_negative():
    """Test negative step is also limited"""
    tuner = SafeAutoTuner()
    
    old = {"budget": 10}
    new = {"budget": 0}
    
    limited = tuner._apply_step_limit(old, new, max_step=2)
    
    assert limited["budget"] == 8, f"Expected 8, got {limited['budget']}"
    print("✓ test_step_limit_negative passed")


def test_process_cycle_not_enough_data():
    """Test need minimum cycles"""
    tuner = SafeAutoTuner()
    tuner.register_policy("GreedyUtilityPolicy", {"budget": 10})
    
    action = tuner.process_cycle(
        policy_name="GreedyUtilityPolicy",
        regret_history=[0.1, 0.2],  # Only 2 cycles
        current_regret=0.3
    )
    
    assert action["type"] == "none", f"Expected none, got {action['type']}"
    assert "need_5_cycles" in action["reason"], f"Expected need_5_cycles, got {action['reason']}"
    print("✓ test_process_cycle_not_enough_data passed")


def test_process_cycle_suggest_mode():
    """Test suggestion mode doesn't apply"""
    tuner = SafeAutoTuner()
    # Must use exact policy name that tuner recognizes
    tuner.register_policy(
        "GreedyUtilityPolicy",
        {"budget": 10},
        mode=TuningMode.SUGGEST
    )
    
    # Very high regret to ensure it triggers
    action = tuner.process_cycle(
        policy_name="GreedyUtilityPolicy",
        regret_history=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],  # Clearly rising, avg > 0.3
        current_regret=0.65
    )
    
    print(f"DEBUG: action = {action}")
    
    # Should suggest but not apply
    assert action["type"] == "suggest", f"Expected suggest, got {action['type']}"
    assert action["params"]["budget"] == 12, f"Expected budget=12, got {action['params']}"
    print("✓ test_process_cycle_suggest_mode passed")


def test_process_cycle_auto_mode():
    """Test auto mode applies changes"""
    tuner = SafeAutoTuner()
    tuner.register_policy(
        "GreedyUtilityPolicy",
        {"budget": 10},
        mode=TuningMode.AUTO
    )
    
    action = tuner.process_cycle(
        policy_name="GreedyUtilityPolicy",
        regret_history=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        current_regret=0.65
    )
    
    # Should apply
    assert action["type"] == "apply", f"Expected apply, got {action['type']}"
    assert action["params"]["budget"] == 12, f"Expected budget=12, got {action['params']}"
    
    # State should be updated
    state = tuner.get_state("GreedyUtilityPolicy")
    assert state["current_params"]["budget"] == 12, f"Expected budget=12, got {state['current_params']['budget']}"
    print("✓ test_process_cycle_auto_mode passed")


def test_rollback():
    """Test rollback functionality"""
    tuner = SafeAutoTuner()
    tuner.register_policy("GreedyUtilityPolicy", {"budget": 10}, mode=TuningMode.AUTO)
    
    # Apply change
    tuner.process_cycle(
        "GreedyUtilityPolicy",
        [0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
        0.65
    )
    
    state = tuner.get_state("GreedyUtilityPolicy")
    assert state["current_params"]["budget"] == 12, f"Expected 12, got {state['current_params']['budget']}"
    
    # Rollback
    tuner.rollback("GreedyUtilityPolicy")
    
    state = tuner.get_state("GreedyUtilityPolicy")
    assert state["current_params"]["budget"] == 10, f"Expected 10, got {state['current_params']['budget']}"
    print("✓ test_rollback passed")


def test_regret_analysis():
    """Test DecisionFeedback regret analysis"""
    feedback = DecisionFeedback()
    
    trace = ArbitrationTrace(uuid4(), "test")
    
    # Selected: low utility
    trace.record(
        intent_id=uuid4(),
        utility=0.3,
        cost=0.5,
        risk=0.1,
        selected=True
    )
    
    # Rejected: high utility (this creates regret)
    trace.record(
        intent_id=uuid4(),
        utility=0.9,
        cost=0.3,
        risk=0.1,
        selected=False,
        rejection_reason="budget_limit:1"
    )
    
    analysis = feedback.analyze(trace.get_records())
    
    assert analysis.selected_count == 1, f"Expected 1 selected, got {analysis.selected_count}"
    assert analysis.rejected_count == 1, f"Expected 1 rejected, got {analysis.rejected_count}"
    assert analysis.potential_utility_lost > 0, "Should have regret"
    print("✓ test_regret_analysis passed")


def main():
    print("=" * 60)
    print("Running SafeAutoTuner Tests")
    print("=" * 60)
    
    test_register_policy()
    test_step_limit()
    test_step_limit_negative()
    test_process_cycle_not_enough_data()
    test_process_cycle_suggest_mode()
    test_process_cycle_auto_mode()
    test_rollback()
    test_regret_analysis()
    
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
