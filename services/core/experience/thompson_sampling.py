"""
Thompson Sampling MVP - Simple Beta distribution bandit for skill selection

Beta(a, b) sampling where:
- a = successes + 1
- b = failures + 1

No contextual features yet - plain Thompson Sampling.
"""
import random
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from .trajectory_dataset import LearningSample


@dataclass
class SkillArm:
    """Single skill arm for Thompson Sampling"""
    skill_id: str
    alpha: float = 1.0  # successes + 1
    beta: float = 1.0   # failures + 1
    total_pulls: int = 0
    
    def sample(self) -> float:
        """Sample from Beta distribution"""
        return random.betavariate(self.alpha, self.beta)
    
    def update(self, reward: float) -> None:
        """Update arm based on reward"""
        self.total_pulls += 1
        if reward > 0:
            # Success - increase alpha
            self.alpha += reward
        else:
            # Failure - increase beta
            self.beta += abs(reward)
    
    def get_mean(self) -> float:
        """Get expected value E[a/(a+b)]"""
        return self.alpha / (self.alpha + self.beta) if (self.alpha + self.beta) > 0 else 0.5
    
    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "alpha": round(self.alpha, 3),
            "beta": round(self.beta, 3),
            "mean": round(self.get_mean(), 3),
            "total_pulls": self.total_pulls
        }


class ThompsonSamplingBandit:
    """
    Simple Thompson Sampling bandit for skill selection.
    
    Usage:
        bandit = ThompsonSamplingBandit()
        selected = bandit.select(candidates=["core.echo", "core.write_file"])
        bandit.update("core.echo", reward=0.8)
    """
    
    def __init__(self, state_file: Optional[str] = None):
        self.arms: Dict[str, SkillArm] = {}
        self.state_file = state_file
        self.selection_history: List[Dict] = []
        
        if state_file:
            self.load_state()
    
    def select(self, candidates: List[str], context: Dict = None) -> str:
        """
        Select skill using Thompson Sampling.
        
        Args:
            candidates: List of available skill IDs
            context: Context features (ignored in v1 MVP)
        
        Returns:
            Selected skill ID
        """
        # Initialize arms for new candidates
        for skill_id in candidates:
            if skill_id not in self.arms:
                self.arms[skill_id] = SkillArm(skill_id=skill_id)
        
        # Sample from each arm
        samples = {skill_id: arm.sample() for skill_id, arm in self.arms.items() if skill_id in candidates}
        
        # Select arm with highest sample
        selected = max(samples, key=samples.get)
        
        # Record selection
        self.selection_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "candidates": candidates,
            "selected": selected,
            "samples": {k: round(v, 3) for k, v in samples.items()},
            "context": context or {}
        })
        
        return selected
    
    def update(self, skill_id: str, reward: float) -> None:
        """
        Update arm with observed reward.
        
        Args:
            skill_id: The skill that was executed
            reward: Observed reward in range [-1, 1]
        """
        if skill_id not in self.arms:
            self.arms[skill_id] = SkillArm(skill_id=skill_id)
        
        self.arms[skill_id].update(reward)
        
        # Save state after update
        if self.state_file:
            self.save_state()
    
    def get_stats(self) -> Dict:
        """Get current bandit statistics"""
        return {
            "total_arms": len(self.arms),
            "total_selections": len(self.selection_history),
            "arms": {skill_id: arm.to_dict() for skill_id, arm in self.arms.items()}
        }
    
    def get_recommendation(self, candidates: List[str]) -> Dict:
        """
        Get recommendation with reasoning (for shadow mode).
        
        Returns dict with both legacy and bandit choices.
        """
        if not candidates:
            return {"error": "No candidates provided"}
        
        # Legacy selection (first candidate)
        legacy_choice = candidates[0]
        
        # Bandit selection
        bandit_choice = self.select(candidates)
        
        # Get arm stats
        arm = self.arms.get(bandit_choice)
        arm_stats = arm.to_dict() if arm else None
        
        return {
            "candidates": candidates,
            "legacy_choice": legacy_choice,
            "bandit_choice": bandit_choice,
            "bandit_confidence": arm_stats.get("mean", 0.5) if arm_stats else 0.5,
            "arm_stats": arm_stats,
            "total_pulls": arm.total_pulls if arm else 0
        }
    
    def save_state(self) -> None:
        """Save bandit state to file"""
        if not self.state_file:
            return
        
        state = {
            "arms": {skill_id: {"alpha": arm.alpha, "beta": arm.beta, "total_pulls": arm.total_pulls}
                     for skill_id, arm in self.arms.items()},
            "selection_count": len(self.selection_history)
        }
        
        Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self) -> None:
        """Load bandit state from file"""
        if not self.state_file:
            return
        
        path = Path(self.state_file)
        if not path.exists():
            return
        
        try:
            with open(path, "r") as f:
                state = json.load(f)
            
            for skill_id, data in state.get("arms", {}).items():
                self.arms[skill_id] = SkillArm(
                    skill_id=skill_id,
                    alpha=data.get("alpha", 1.0),
                    beta=data.get("beta", 1.0),
                    total_pulls=data.get("total_pulls", 0)
                )
        except Exception as e:
            print(f"[THOMPSON_BANDIT] Failed to load state: {e}", flush=True)
    
    def train_from_samples(self, samples: List[LearningSample]) -> None:
        """
        Train bandit from historical samples (offline replay).
        
        Args:
            samples: List of historical learning samples
        """
        for sample in samples:
            # Update with historical reward
            self.update(sample.chosen_skill, sample.reward)
        
        print(f"[THOMPSON_BANDIT] Trained from {len(samples)} samples", flush=True)


# Global bandit instance for easy access
_bandit: Optional[ThompsonSamplingBandit] = None


def get_bandit(state_file: str = "/app/experience/bandit_state.json") -> ThompsonSamplingBandit:
    """Get or create global bandit instance"""
    global _bandit
    if _bandit is None:
        _bandit = ThompsonSamplingBandit(state_file=state_file)
    return _bandit