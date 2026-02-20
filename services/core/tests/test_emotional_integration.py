"""
Integration tests for Emotional Layer

Tests full cycle: signals → inference → aggregation → influence
"""

import pytest
from datetime import datetime
from sqlalchemy import delete
from database import AsyncSessionLocal
from models import EmotionalLayerState
from schemas import EmotionalSignals
from emotional_layer import emotional_layer
from emotional_inference import EmotionalInferenceEngine
from emotional_aggregation import EmotionalLayerStateAggregator
from emotional_influence import EmotionalInfluenceEngine


@pytest.mark.asyncio
class TestEmotionalIntegration:
    """Test full emotional layer cycle."""

    async def test_full_emotional_cycle_tired_user(self):
        """Test complete cycle for tired user."""

        user_id = "test-user-001"

        # Clean up
        async with AsyncSessionLocal() as db:
            await db.execute(delete(EmotionalLayerState).where(EmotionalLayerState.user_id == user_id))
            await db.commit()

        # 1. User says they're tired
        signals = EmotionalSignals(
            user_text="Я устал, давай попроще",
            goal_stats={"aborted": 2},
            system_metrics={"avg_goal_complexity": 0.8}
        )

        # 2. Get influence
        influence = await emotional_layer.get_influence(user_id, signals)

        # 3. Verify influence
        assert influence.complexity_penalty > 0.3, "Should reduce complexity"
        assert influence.pace_modifier < -0.1, "Should slow down"

        # 4. Verify state was saved
        state = await emotional_layer.get_current_state(user_id)
        assert state["valence"] < 0.0, "Should be negative"
        assert state["focus"] < 0.5, "Should have low focus"

    async def test_emotional_stability(self):
        """Test that emotions don't jump around too much."""

        user_id = "test-user-stability"

        # Clean up
        async with AsyncSessionLocal() as db:
            await db.execute(delete(EmotionalLayerState).where(EmotionalLayerState.user_id == user_id))
            await db.commit()

        # Create initial state
        signals1 = EmotionalSignals(
            user_text="Я устал"
        )
        await emotional_layer.get_influence(user_id, signals1)

        # Get state again
        state1 = await emotional_layer.get_current_state(user_id)

        # Similar input again
        signals2 = EmotionalSignals(
            user_text="Я всё ещё устал"
        )
        await emotional_layer.get_influence(user_id, signals2)

        state2 = await emotional_layer.get_current_state(user_id)

        # Should be similar (EMA smoothing)
        assert abs(state2["arousal"] - state1["arousal"]) < 0.3
        assert abs(state2["valence"] - state1["valence"]) < 0.3

    async def test_context_mapping_integration(self):
        """Test influence → context conversion in real scenario."""

        user_id = "test-user-context"

        # Stressed user
        signals = EmotionalSignals(
            user_text="Это слишком сложно",
            goal_stats={"aborted": 3},
        )

        # Get influence as context
        context = await emotional_layer.get_influence_context(user_id, signals)

        # Verify context
        assert context["complexity_limit"] < 0.8, "Should limit complexity"
        assert context["max_depth"] <= 2, "Should reduce depth"
        assert context["pace"] == "slow", "Should slow pace"
        assert context["explanation"] in ["normal", "detailed"], "Should explain more"

    async def test_new_user_gets_baseline(self):
        """New user should start from baseline."""

        user_id = "test-user-new-001"

        # Clean up if exists
        async with AsyncSessionLocal() as db:
            await db.execute(delete(EmotionalLayerState).where(EmotionalLayerState.user_id == user_id))
            await db.commit()

        # Get current state (should be baseline)
        state = await emotional_layer.get_current_state(user_id)

        assert state["arousal"] == 0.5
        assert state["valence"] == 0.0
        assert state["focus"] == 0.5
        assert state["confidence"] == 0.5

    async def test_multiple_users_dont_interfere(self):
        """Different users should have independent emotional states."""

        user1 = "test-user-001"
        user2 = "test-user-002"

        # Clean up
        async with AsyncSessionLocal() as db:
            await db.execute(delete(EmotionalLayerState).where(EmotionalLayerState.user_id == user1))
            await db.execute(delete(EmotionalLayerState).where(EmotionalLayerState.user_id == user2))
            await db.commit()

        # User 1 is tired
        signals1 = EmotionalSignals(user_text="Я устал")
        await emotional_layer.get_influence(user1, signals1)

        # User 2 is happy
        signals2 = EmotionalSignals(user_text="Отлично, всё получилось!")
        await emotional_layer.get_influence(user2, signals2)

        # Check states are different
        state1 = await emotional_layer.get_current_state(user1)
        state2 = await emotional_layer.get_current_state(user2)

        assert state1["valence"] < 0.0, "User 1 should be negative"
        assert state2["valence"] > 0.0, "User 2 should be positive"

    async def test_history_tracking(self):
        """Test that emotional history is recorded."""

        user_id = "test-user-history"

        # Clean up
        async with AsyncSessionLocal() as db:
            await db.execute(delete(EmotionalLayerState).where(EmotionalLayerState.user_id == user_id))
            await db.commit()

        # Create multiple states
        for i in range(3):
            signals = EmotionalSignals(user_text=f"Message {i}")
            await emotional_layer.get_influence(user_id, signals)

        # Get history
        history = await emotional_layer.get_history(user_id, limit=10)

        assert len(history) >= 3, "Should have at least 3 states"
        assert all("arousal" in h for h in history), "All entries should have arousal"

        # Should be ordered by timestamp (newest first)
        timestamps = [h["timestamp"] for h in history]
        assert timestamps == sorted(timestamps, reverse=True)
