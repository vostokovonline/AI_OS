"""
Emotional Layer - Main Facade (MVP)

Orchestrates:
1. Signal collection
2. Inference (rule-based)
3. Aggregation (EMA smoothing)
4. Influence mapping
5. State persistence

Usage:
    emotional_layer = EmotionalLayer()

    # Get influence for decision
    influence = await emotional_layer.get_influence(user_id, signals)

    # Or full cycle
    result = await emotional_layer.process(user_id, signals)
"""

from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy import select, desc
from database import AsyncSessionLocal
from models import EmotionalLayerState
from schemas import EmotionalSignals
from emotional_config import EMOTIONAL_BASELINE
from emotional_inference import EmotionalInferenceEngine
from emotional_aggregation import EmotionalStateAggregator
from emotional_influence import (
    EmotionalInfluenceEngine,
    InfluenceContextMapper,
    EmotionalInfluence,
)


class EmotionalLayer:
    """
    Main facade for Emotional Layer.
    """

    def __init__(self):
        self.inference_engine = EmotionalInferenceEngine()
        self.aggregator = EmotionalStateAggregator()
        self.influence_engine = EmotionalInfluenceEngine()
        self.context_mapper = InfluenceContextMapper()

    # =========================
    # Main API
    # =========================

    async def get_influence(
        self,
        user_id: str,
        signals: EmotionalSignals
    ) -> EmotionalInfluence:
        """
        Get emotional influence for decision-making.

        This is the PRIMARY integration point.

        Args:
            user_id: User identifier
            signals: Aggregated signals (from Context Builder)

        Returns:
            EmotionalInfluence with decision modifiers
        """

        logger.info(f"🔍 [Emotional Layer] Getting influence for user {user_id}")

        # 1. Get current state (or baseline)
        current_state = await self._get_current_state(user_id)
        logger.info(f"   Current state: {current_state}")

        # 2. Infer new state from signals
        inferred_state = self.inference_engine.infer(signals)
        logger.info(f"   Inferred state: {inferred_state}")

        # 3. Aggregate (EMA smoothing)
        aggregated_state = self.aggregator.aggregate(current_state, inferred_state)
        logger.info(f"   Aggregated state: {aggregated_state}")

        # 4. Save to DB
        await self._save_state(user_id, aggregated_state, signals)

        # 5. Map to influence
        influence = self.influence_engine.map_to_influence(aggregated_state)

        logger.info(f"   Influence: {influence}")
        return influence

    async def get_influence_context(
        self,
        user_id: str,
        signals: EmotionalSignals
    ) -> Dict[str, any]:
        """
        Get influence as agent-friendly context dict.

        Convenience method for integration with agents.

        Returns:
        {
            "complexity_limit": 0.6,
            "max_depth": 1,
            "exploration": "conservative",
            "explanation": "detailed",
            "pace": "slow",
            "confidence": "low"
        }
        """

        influence = await self.get_influence(user_id, signals)
        context = self.context_mapper.to_context(influence)

        # Log for debugging
        logger.info(f"💾 [Emotional Layer] Saved state for user {user_id}: {influence}")

        return context

    # =========================
    # State management
    # =========================

    async def get_current_state(self, user_id: str) -> Dict[str, float]:
        """
        Get current emotional state from DB.

        Returns baseline if no history.
        """

        return await self._get_current_state(user_id)

    async def get_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, any]]:
        """
        Get emotional state history.
        """

        async with AsyncSessionLocal() as db:
            stmt = select(EmotionalLayerState).where(
                EmotionalLayerState.user_id == user_id
            ).order_by(
                desc(EmotionalLayerState.created_at)
            ).limit(limit)

            result = await db.execute(stmt)
            states = result.scalars().all()

            return [
                {
                    "arousal": s.arousal,
                    "valence": s.valence,
                    "focus": s.focus,
                    "confidence": s.confidence,
                    "timestamp": s.created_at,  # Fixed: was s.timestamp
                    "source": s.source,
                }
                for s in states
            ]

    # =========================
    # Event tracking
    # =========================

    async def track_event(
        self,
        user_id: str,
        event_type: str,
        metadata: Optional[Dict] = None
    ):
        """
        Track an event for future emotional inference.

        Note: This is optional for MVP.
        Events are primarily collected by Context Builder.
        """

        # TODO: Implement event storage for pattern detection
        # For MVP, this is a no-op
        pass

    # =========================
    # Internal helpers
    # =========================

    async def _get_current_state(self, user_id: str) -> Dict[str, float]:
        """
        Get most recent state from DB or baseline.
        """

        async with AsyncSessionLocal() as db:
            stmt = select(EmotionalLayerState).where(
                EmotionalLayerState.user_id == user_id
            ).order_by(
                desc(EmotionalLayerState.created_at)
            ).limit(1)

            result = await db.execute(stmt)
            state = result.scalar_one_or_none()

            if state:
                return {
                    "arousal": state.arousal,
                    "valence": state.valence,
                    "focus": state.focus,
                    "confidence": state.confidence,
                }
            else:
                # Return baseline for new users
                return dict(EMOTIONAL_BASELINE)

    async def _save_state(
        self,
        user_id: str,
        state: Dict[str, float],
        signals: EmotionalSignals
    ):
        """
        Save emotional state to DB.
        """

        try:
            async with AsyncSessionLocal() as db:
                emotional_state = EmotionalLayerState(
                    user_id=user_id,
                    arousal=state["arousal"],
                    valence=state["valence"],
                    focus=state["focus"],
                    confidence=state["confidence"],
                    created_at=datetime.utcnow(),
                    source="inference",
                    signals=signals.dict() if signals else None,
                )

                db.add(emotional_state)
                await db.commit()

                logger.info(f"✅ [Emotional Layer] State saved to DB: user={user_id}, arousal={state['arousal']:.2f}, valence={state['valence']:.2f}")
        except Exception as e:
            logger.info(f"❌ [Emotional Layer] Failed to save state: {e}")
            # Don't raise - allow system to continue without emotional tracking


# =========================
# Singleton instance
# =========================

emotional_layer = EmotionalLayer()
