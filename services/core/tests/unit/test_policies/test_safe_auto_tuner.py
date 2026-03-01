"""
Unit tests for SafeAutoTuner
"""
import pytest
import sys
from uuid import uuid4
from datetime import datetime

# Import directly without going through application/__init__
sys.path.insert(0, '/app')

from application.policies.safe_auto_tuner import (
    SafeAutoTuner,
    TuningMode,
    TuningState
)
from application.policies.arbitration_trace import ArbitrationTrace
from application.policies.decision_policies import (
    PassThroughPolicy,
    GreedyUtilityPolicy,
    ScoredIntent
)
from application.policies.decision_feedback import DecisionFeedback


class TestSafeAutoTuner:
    """Test SafeAutoTuner"""
    
    def test_register_policy(self):
        """Test policy registration"""
        tuner = SafeAutoTuner()
        
        tuner.register_policy(
            "GreedyUtilityPolicy",
            {"budget": 10},
            mode=TuningMode.SUGGEST
        )
        
        state = tuner.get_state("GreedyUtilityPolicy")
        
        assert state is not None
        assert state["current_params"] == {"budget": 10}
        assert state["mode"] == "suggest"
    
    def test_step_limit(self):
        """Test max step size is enforced"""
        tuner = SafeAutoTuner()
        
        old = {"budget": 10}
        new = {"budget": 20}
        
        limited = tuner._apply_step_limit(old, new, max_step=2)
        
        assert limited["budget"] == 12  # 10 + 2
    
    def test_step_limit_negative(self):
        """Test negative step is also limited"""
        tuner = SafeAutoTuner()
        
        old = {"budget": 10}
        new = {"budget": 0}
        
        limited = tuner._apply_step_limit(old, new, max_step=2)
        
        assert limited["budget"] == 8  # 10 - 2
    
    def test_step_limit_no_change_needed(self):
        """Test when change is within limit"""
        tuner = SafeAutoTuner()
        
        old = {"budget": 10}
        new = {"budget": 12}
        
        limited = tuner._apply_step_limit(old, new, max_step=2)
        
        assert limited["budget"] == 12  # No change needed
    
    def test_process_cycle_not_enough_data(self):
        """Test need minimum cycles"""
        tuner = SafeAutoTuner()
        tuner.register_policy("Greedy", {"budget": 10})
        
        action = tuner.process_cycle(
            policy_name="Greedy",
            regret_history=[0.1, 0.2],  # Only 2 cycles
            current_regret=0.3
        )
        
        assert action["type"] == "none"
        assert "need_5_cycles" in action["reason"]
    
    def test_process_cycle_below_threshold(self):
        """Test no action when regret acceptable"""
        tuner = SafeAutoTuner()
        tuner.register_policy("Greedy", {"budget": 10})
        
        action = tuner.process_cycle(
            policy_name="Greedy",
            regret_history=[0.1, 0.15, 0.2, 0.22, 0.25],  # avg ~0.2
            current_regret=0.25
        )
        
        assert action["type"] == "none"
        assert "regret_acceptable" in action["reason"]
    
    def test_process_cycle_suggest_mode(self):
        """Test suggestion mode doesn't apply"""
        tuner = SafeAutoTuner()
        tuner.register_policy(
            "Greedy",
            {"budget": 10},
            mode=TuningMode.SUGGEST
        )
        
        action = tuner.process_cycle(
            policy_name="Greedy",
            regret_history=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45],  # Rising trend
            current_regret=0.45
        )
        
        # Should suggest but not apply
        assert action["type"] == "suggest"
        assert action["params"]["budget"] == 12  # Limited to +2
    
    def test_process_cycle_auto_mode(self):
        """Test auto mode applies changes"""
        tuner = SafeAutoTuner()
        tuner.register_policy(
            "Greedy",
            {"budget": 10},
            mode=TuningMode.AUTO
        )
        
        action = tuner.process_cycle(
            policy_name="Greedy",
            regret_history=[0.1, 0.2, 0.3, 0.35, 0.4, 0.45],
            current_regret=0.45
        )
        
        # Should apply
        assert action["type"] == "apply"
        assert action["params"]["budget"] == 12
        
        # State should be updated
        state = tuner.get_state("Greedy")
        assert state["current_params"]["budget"] == 12
    
    def test_rollback(self):
        """Test rollback functionality"""
        tuner = SafeAutoTuner()
        tuner.register_policy("Greedy", {"budget": 10}, mode=TuningMode.AUTO)
        
        # Apply change
        tuner.process_cycle(
            "Greedy",
            [0.1, 0.2, 0.3, 0.35, 0.4, 0.45],
            0.45
        )
        
        state = tuner.get_state("Greedy")
        assert state["current_params"]["budget"] == 12
        
        # Rollback
        tuner.rollback("Greedy")
        
        state = tuner.get_state("Greedy")
        assert state["current_params"]["budget"] == 10
    
    def test_rollback_no_history(self):
        """Test rollback with no history"""
        tuner = SafeAutoTuner()
        tuner.register_policy("Greedy", {"budget": 10})
        
        result = tuner.rollback("Greedy")
        
        assert result is None  # No history


class TestDecisionFeedback:
    """Test DecisionFeedback"""
    
    def test_regret_analysis_empty(self):
        """Test empty trace"""
        feedback = DecisionFeedback()
        analysis = feedback.analyze([])
        
        assert analysis.selected_count == 0
        assert analysis.regret_ratio == 0
    
    def test_regret_analysis_selected_only(self):
        """Test with only selected (no rejected)"""
        # Create mock records
        from dataclasses import replace
        from application.policies.arbitration_trace import ArbitrationRecord
        
        trace = ArbitrationTrace(uuid4(), "test")
        
        # Add selected records
        for i in range(3):
            trace.record(
                intent_id=uuid4(),
                utility=0.8,
                cost=0.2,
                risk=0.1,
                selected=True
            )
        
        analysis = feedback.analyze(trace.get_records())
        
        assert analysis.selected_count == 3
        assert analysis.rejected_count == 0
        assert analysis.regret_ratio == 0
    
    def test_regret_analysis_with_rejection(self):
        """Test with rejected items"""
        feedback = DecisionFeedback()
        
        from application.policies.arbitration_trace import ArbitrationRecord
        
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
        
        assert analysis.selected_count == 1
        assert analysis.rejected_count == 1
        assert analysis.potential_utility_lost > 0  # Should have regret
    
    def test_should_improve_policy(self):
        """Test threshold"""
        feedback = DecisionFeedback()
        
        # Create analysis with high regret
        from application.policies.arbitration_trace import ArbitrationTrace
        trace = ArbitrationTrace(uuid4(), "test")
        trace.record(uuid4(), 0.9, 0.1, 0.1, True)
        trace.record(uuid4(), 0.1, 0.9, 0.1, False)
        
        analysis = feedback.analyze(trace.get_records())
        
        assert analysis.regret_ratio > 0.3
        assert feedback.should_improve_policy(analysis) is True
        
        # Low regret
        trace2 = ArbitrationTrace(uuid4(), "test")
        trace2.record(uuid4(), 0.8, 0.1, 0.1, True)
        trace2.record(uuid4(), 0.2, 0.8, 0.1, False)
        
        analysis2 = feedback.analyze(trace2.get_records())
        
        assert analysis2.regret_ratio < 0.3
        assert feedback.should_improve_policy(analysis2) is False


class TestDecisionPolicies:
    """Test decision policies"""
    
    @pytest.mark.asyncio
    async def test_pass_through_policy(self):
        """Test PassThroughPolicy"""
        policy = PassThroughPolicy()
        
        # Create scored intents
        intents = [
            BulkExecutionIntent(
                goal_id=uuid4(),
                transition=StateTransitionIntent(
                    goal_id=uuid4(),
                    from_status="active",
                    to_status="done",
                    reason="test",
                    actor="test"
                )
            )
            for _ in range(5)
        ]
        
        scored = [
            ScoredIntent(intent=i, utility=0.5 + i*0.1, cost=0.3, risk=0.1)
            for i, i in enumerate(intents)
        ]
        
        selected = await policy.select(scored, budget=None)
        
        # Should return all
        assert len(selected) == 5
    
    @pytest.mark.asyncio
    async def test_greedy_utility_policy(self):
        """Test GreedyUtilityPolicy sorts by utility"""
        policy = GreedyUtilityPolicy()
        
        intents = [
            BulkExecutionIntent(
                goal_id=uuid4(),
                transition=StateTransitionIntent(
                    goal_id=uuid4(),
                    from_status="active",
                    to_status="done",
                    reason="test",
                    actor="test"
                )
            )
            for _ in range(5)
        ]
        
        # Mixed utilities
        scored = [
            ScoredIntent(intent=intents[2], utility=0.3, cost=0.3, risk=0.1),  # Low
            ScoredIntent(intent=intents[0], utility=0.9, cost=0.3, risk=0.1),  # High
            ScoredIntent(intent=intents[1], utility=0.6, cost=0.3, risk=0.1),  # Medium
        ]
        
        selected = await policy.select(scored, budget=2)
        
        # Should select top 2 by utility
        assert selected[0].utility == 0.9
        assert selected[1].utility == 0.6
    
    @pytest.mark.asyncio
    async def test_policy_with_trace(self):
        """Test policies can use trace"""
        policy = PassThroughPolicy()
        
        trace = ArbitrationTrace(uuid4(), "PassThroughPolicy")
        
        intents = [
            BulkExecutionIntent(
                goal_id=uuid4(),
                transition=StateTransitionIntent(
                    goal_id=uuid4(),
                    from_status="active",
                    to_status="done",
                    reason="test",
                    actor="test"
                )
            )
        ]
        
        scored = [ScoredIntent(intent=intents[0], utility=0.8, cost=0.2, risk=0.1)]
        
        selected = await policy.select(scored, budget=None, trace=trace)
        
        # Should have recorded
        assert trace.count() == 1
        record = trace.get_records()[0]
        assert record.selected is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
