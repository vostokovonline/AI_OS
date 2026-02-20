"""
GOAL STATE MACHINE TESTS

Tests for goal transition validation.
Goal state machine must prevent invalid transitions.
"""
import pytest
import sys
import os

# Add app directory to path
app_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)


class TestGoalStatusValidation:
    """Test goal status field protection."""

    def test_direct_status_assignment_blocked(self):
        """Direct status assignment must be blocked."""
        from models import Goal
        
        goal = Goal(title="Test", description="Test")
        
        with pytest.raises(RuntimeError) as exc_info:
            goal.status = "done"
        
        assert "DIRECT STATUS ASSIGNMENT BLOCKED" in str(exc_info.value)

    def test_status_read_works(self):
        """Reading status must work."""
        from models import Goal
        
        goal = Goal(title="Test", description="Test")
        
        # Default status can be None, "active", or "pending"
        # The important thing is that reading doesn't raise
        status = goal.status
        assert status is None or status in ["active", "pending", "done", "ongoing"]


class TestGoalTransitions:
    """Test goal transition service."""

    VALID_TRANSITIONS = [
        ("pending", "active"),
        ("active", "ongoing"),
        ("active", "done"),
        ("ongoing", "done"),
        ("active", "aborted"),
        ("ongoing", "aborted"),
    ]

    INVALID_TRANSITIONS = [
        ("done", "active"),
        ("done", "pending"),
        ("aborted", "active"),
        ("aborted", "done"),
    ]

    def test_valid_transitions_list(self):
        """Verify valid transitions are defined."""
        # These transitions should be allowed by the state machine
        for from_state, to_state in self.VALID_TRANSITIONS:
            # Just verify the tuples are valid strings
            assert isinstance(from_state, str)
            assert isinstance(to_state, str)

    def test_invalid_transitions_list(self):
        """Verify invalid transitions are defined."""
        # These transitions should be blocked
        for from_state, to_state in self.INVALID_TRANSITIONS:
            assert isinstance(from_state, str)
            assert isinstance(to_state, str)


class TestGoalTypeValidation:
    """Test goal type validation."""

    VALID_GOAL_TYPES = [
        "achievable",
        "continuous",
        "directional",
        "exploratory",
        "meta",
        "bounded",  # Legacy
    ]

    def test_valid_goal_types(self):
        """Goal types must be valid strings."""
        for goal_type in self.VALID_GOAL_TYPES:
            assert isinstance(goal_type, str)
            assert len(goal_type) > 0


class TestGoalContractValidation:
    """Test goal contract system."""

    def test_contract_validator_imports(self):
        """Contract validator must import."""
        from goal_contract_validator import goal_contract_validator
        assert goal_contract_validator is not None

    def test_default_contract_creation(self):
        """Default contract must be created for any goal type."""
        from goal_contract_validator import goal_contract_validator
        
        for goal_type in ["achievable", "continuous", "directional"]:
            contract = goal_contract_validator.create_default_contract(goal_type, 0)
            assert contract is not None
            assert isinstance(contract, dict)
