"""
Learning Pipeline - Separates learning updates from execution

Execution → LearningEvent → LearningPipeline → Policy Update

This ensures:
- Replay doesn't affect live policy
- Offline evaluation is isolated
- Debugging is clean
"""
from typing import List, Optional, Dict
from datetime import datetime

from .learning_event import LearningEvent
from .thompson_sampling import ThompsonSamplingBandit, get_bandit


class LearningPipeline:
    """
    Decoupled learning pipeline.
    
    Events are processed here - not in execution or shadow evaluator.
    This enables:
    - Replay without affecting live policy
    - Offline evaluation
    - Clean debugging
    """
    
    def __init__(self, bandit: ThompsonSamplingBandit = None):
        self.bandit = bandit or get_bandit()
        self._event_queue: List[LearningEvent] = []
    
    def process_event(self, event: LearningEvent) -> Dict:
        """
        Process a learning event through the pipeline.
        
        Returns processing result with metrics.
        """
        # Only update legacy executed arm (not shadow)
        executed = event.executed_arm
        if executed and executed.startswith("core."):
            self.bandit.update(executed, event.reward)
            update_result = {
                "arm": executed,
                "updated": True,
                "new_alpha": self.bandit.arms.get(executed, {}).get("alpha", 0),
                "new_beta": self.bandit.arms.get(executed, {}).get("beta", 0)
            }
        else:
            update_result = {
                "arm": executed,
                "updated": False,
                "reason": "invalid_skill_id"
            }
        
        # Queue event for batch processing
        self._event_queue.append(event)
        
        return {
            "event_id": event.event_id,
            "timestamp": datetime.utcnow().isoformat(),
            "policy_update": update_result,
            "queue_size": len(self._event_queue)
        }
    
    def process_batch(self, events: List[LearningEvent]) -> Dict:
        """Process multiple events"""
        results = [self.process_event(e) for e in events]
        
        return {
            "processed_count": len(results),
            "results": results,
            "final_queue_size": len(self._event_queue)
        }
    
    def replay_events(
        self,
        events: List[LearningEvent],
        apply_updates: bool = False
    ) -> Dict:
        """
        Replay historical events without affecting live policy.
        
        Args:
            events: Historical events to replay
            apply_updates: If True, actually apply updates (for training)
                         If False, just simulate (for evaluation)
        """
        # Create temporary bandit for replay
        from .thompson_sampling import ThompsonSamplingBandit
        replay_bandit = ThompsonSamplingBandit()
        
        # Simulate replay
        results = []
        for event in events:
            executed = event.executed_arm
            if executed and executed.startswith("core."):
                if apply_updates:
                    self.bandit.update(executed, event.reward)
                else:
                    replay_bandit.update(executed, event.reward)
                
                results.append({
                    "event_id": event.event_id,
                    "arm": executed,
                    "reward": event.reward,
                    "regret": event.regret
                })
        
        # Get final stats
        if apply_updates:
            final_stats = self.bandit.get_stats()
        else:
            final_stats = replay_bandit.get_stats()
        
        return {
            "events_replayed": len(results),
            "applied_updates": apply_updates,
            "final_policy_stats": final_stats,
            "results": results
        }
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "queued_events": len(self._event_queue),
            "bandit_stats": self.bandit.get_stats()
        }


# Global pipeline
_pipeline: Optional[LearningPipeline] = None


def get_learning_pipeline() -> LearningPipeline:
    """Get or create global learning pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = LearningPipeline()
    return _pipeline