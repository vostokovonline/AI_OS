"""
Shadow Mode Evaluator - Compares bandit vs legacy selection without execution

Logs:
{
  "legacy_choice": "core.echo",
  "bandit_choice": "core.write_file", 
  "observed_reward": -1.0,
  "regret": 0.5
}

Does NOT affect actual execution - purely observational.
"""
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from .thompson_sampling import ThompsonSamplingBandit, get_bandit
from .trajectory_dataset import LearningSample


class ShadowEvaluator:
    """
    Shadow mode evaluator for bandit vs legacy comparison.
    
    Usage:
        evaluator = ShadowEvaluator()
        decision = evaluator.evaluate_shadow(
            candidates=["core.echo", "core.write_file"],
            context={"goal_type": "achievable"}
        )
        # ... execute with legacy choice ...
        evaluator.record_outcome(decision, reward=-1.0, latency_ms=100)
    """
    
    def __init__(self, bandit: ThompsonSamplingBandit = None, log_file: str = "/app/experience/shadow_evaluations.jsonl"):
        self.bandit = bandit or get_bandit()
        self.log_file = Path(log_file)
        self.evaluations: List[Dict] = []
        
        # Load existing evaluations
        self._load_evaluations()
    
    def _load_evaluations(self) -> None:
        """Load previous evaluations from file"""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, "r") as f:
                self.evaluations = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            print(f"[SHADOW_EVAL] Failed to load: {e}", flush=True)
    
    def evaluate_shadow(
        self,
        candidates: List[str],
        context: Dict,
        legacy_choice: str = None
    ) -> Dict:
        """
        Evaluate bandit vs legacy without execution.
        
        Returns:
            dict with both choices for logging
        """
        if not candidates:
            return {"error": "No candidates provided"}
        
        # Legacy choice (first candidate or provided)
        legacy = legacy_choice or candidates[0]
        
        # Get bandit recommendation
        recommendation = self.bandit.get_recommendation(candidates)
        bandit_choice = recommendation.get("bandit_choice", legacy)
        
        evaluation = {
            "timestamp": datetime.utcnow().isoformat(),
            "context": context,
            "candidates": candidates,
            "legacy_choice": legacy,
            "bandit_choice": bandit_choice,
            "bandit_confidence": recommendation.get("bandit_confidence", 0.5),
            "arm_stats": recommendation.get("arm_stats"),
            "observed_reward": None,
            "regret": None,
            "phase": "pending"
        }
        
        # Log to file
        self._log_evaluation(evaluation)
        self.evaluations.append(evaluation)
        
        return evaluation
    
    def record_outcome(
        self,
        evaluation: Dict,
        reward: float,
        latency_ms: int,
        success: bool
    ) -> Dict:
        """
        Record observed outcome and calculate regret.
        
        Args:
            evaluation: Previous evaluation dict
            reward: Observed reward
            latency_ms: Execution latency
            success: Whether execution succeeded
        
        Returns:
            Updated evaluation with regret
        """
        # Calculate regret: expected_best - actual
        # Expected best is bandit choice's mean reward
        bandit_choice = evaluation.get("bandit_choice")
        arm_stats = evaluation.get("arm_stats", {})
        expected_best = arm_stats.get("mean", 0.5) if arm_stats else 0.5
        
        # Simple regret: difference between expected and actual
        regret = expected_best - reward
        
        # Update evaluation
        evaluation["observed_reward"] = reward
        evaluation["regret"] = round(regret, 3)
        evaluation["latency_ms"] = latency_ms
        evaluation["success"] = success
        evaluation["phase"] = "completed"
        
        # Update bandit with observed reward
        self.bandit.update(bandit_choice, reward)
        
        # Log updated evaluation
        self._log_evaluation(evaluation)
        
        return evaluation
    
    def _log_evaluation(self, evaluation: Dict) -> None:
        """Log evaluation to file"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(evaluation) + "\n")
    
    def get_statistics(self) -> Dict:
        """
        Get shadow evaluation statistics.
        """
        if not self.evaluations:
            return {"total_evaluations": 0}
        
        completed = [e for e in self.evaluations if e.get("phase") == "completed"]
        
        if not completed:
            return {
                "total_evaluations": len(self.evaluations),
                "completed": 0,
                "pending": len(self.evaluations)
            }
        
        regrets = [e.get("regret", 0) for e in completed if e.get("regret") is not None]
        bandit_wins = sum(
            1 for e in completed
            if e.get("bandit_choice") != e.get("legacy_choice")
            and e.get("observed_reward", 0) > 0
        )
        
        return {
            "total_evaluations": len(self.evaluations),
            "completed": len(completed),
            "pending": len(self.evaluations) - len(completed),
            "avg_regret": sum(regrets) / len(regrets) if regrets else 0,
            "min_regret": min(regrets) if regrets else 0,
            "max_regret": max(regrets) if regrets else 0,
            "bandit_wins": bandit_wins,
            "bandit_total_diffs": sum(
                1 for e in completed if e.get("bandit_choice") != e.get("legacy_choice")
            )
        }
    
    def offline_replay(self, samples: List[LearningSample]) -> Dict:
        """
        Run offline replay to initialize bandit and measure performance.
        
        Args:
            samples: Historical learning samples
        
        Returns:
            Performance metrics
        """
        print(f"[SHADOW_EVAL] Running offline replay with {len(samples)} samples", flush=True)
        
        # Create temporary evaluations from historical data
        for sample in samples:
            context = {
                "goal_type": sample.context.goal_type,
                "domain": sample.context.domain,
                "goal_length": sample.context.goal_length
            }
            
            evaluation = self.evaluate_shadow(
                candidates=sample.candidates or [sample.chosen_skill],
                context=context,
                legacy_choice=sample.chosen_skill
            )
            
            # Record outcome
            self.record_outcome(
                evaluation=evaluation,
                reward=sample.reward,
                latency_ms=sample.latency_ms,
                success=sample.success
            )
        
        return self.get_statistics()


# Global evaluator instance
_evaluator: Optional[ShadowEvaluator] = None


def get_shadow_evaluator() -> ShadowEvaluator:
    """Get or create global shadow evaluator"""
    global _evaluator
    if _evaluator is None:
        _evaluator = ShadowEvaluator()
    return _evaluator