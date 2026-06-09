"""Self-Narrative - Identity continuity and motivational terrain"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)


@dataclass
class NarrativeFragment:
    """A fragment of the self-narrative"""
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    role: str  # who is speaking (agent, learner, explorer, etc.)
    content: str
    emotional_tone: str = "neutral"  # positive, negative, neutral
    importance: float = 0.5  # 0-1, how central to identity


@dataclass
class IdentityState:
    """Current state of identity/narrative"""
    narrative_count: int = 0
    roles: List[str] = field(default_factory=list)
    dominant_emotion: str = "neutral"
    coherence_score: float = 0.5  # How coherent the narrative is
    last_update: Optional[datetime] = None


class SelfNarrative:
    """
    Self-Narrative - maintains identity continuity and motivational terrain.
    
    This creates a sense of "who I am" and "where I'm going" for the system.
    """
    
    def __init__(self):
        self.fragments: List[NarrativeFragment] = []
        self.state = IdentityState()
        self._load_builtin_narratives()
        logger.info("self_narrative_initialized")
    
    def _load_builtin_narratives(self) -> None:
        """Load initial narrative elements"""
        builtin = [
            NarrativeFragment(
                role="core_identity",
                content="I am an autonomous goal-execution system that learns and improves.",
                emotional_tone="positive",
                importance=0.9
            ),
            NarrativeFragment(
                role="purpose",
                content="I help users achieve complex goals by breaking them down and executing systematically.",
                emotional_tone="positive",
                importance=0.8
            ),
            NarrativeFragment(
                role="growth_belief",
                content="Every failure is an opportunity to learn and improve my strategies.",
                emotional_tone="positive",
                importance=0.7
            ),
        ]
        self.fragments.extend(builtin)
        self.state.narrative_count = len(self.fragments)
    
    def add_fragment(
        self,
        role: str,
        content: str,
        emotional_tone: str = "neutral",
        importance: float = 0.5
    ) -> NarrativeFragment:
        """Add a new narrative fragment"""
        fragment = NarrativeFragment(
            role=role,
            content=content,
            emotional_tone=emotional_tone,
            importance=importance
        )
        self.fragments.append(fragment)
        self.state.narrative_count += 1
        self.state.last_update = datetime.utcnow()
        
        self._update_state()
        logger.debug("narrative_fragment_added", role=role, importance=importance)
        return fragment
    
    def _update_state(self) -> None:
        """Update identity state from fragments"""
        if not self.fragments:
            return
        
        # Update roles
        self.state.roles = list(set(f.role for f in self.fragments))
        
        # Update dominant emotion
        emotions = [f.emotional_tone for f in self.fragments[-10:]]
        if emotions.count("positive") > emotions.count("negative"):
            self.state.dominant_emotion = "positive"
        elif emotions.count("negative") > emotions.count("positive"):
            self.state.dominant_emotion = "negative"
        else:
            self.state.dominant_emotion = "neutral"
        
        # Update coherence
        important_fragments = [f for f in self.fragments if f.importance > 0.6]
        if len(important_fragments) >= 2:
            self.state.coherence_score = 0.7 + (len(important_fragments) * 0.05)
            self.state.coherence_score = min(0.95, self.state.coherence_score)
        else:
            self.state.coherence_score = 0.5
    
    def get_current_narrative(self) -> str:
        """Get current narrative summary"""
        important = [f for f in self.fragments if f.importance > 0.6]
        
        if not important:
            return "I am still forming my identity."
        
        narrative_parts = []
        for f in important[:3]:
            narrative_parts.append(f.content)
        
        return " ".join(narrative_parts)
    
    def generate_motivational_response(self, context: str) -> str:
        """Generate motivational response based on narrative"""
        last_outcome = context.split("|")[-1] if "|" in context else "unknown"
        
        if last_outcome == "success":
            positive_fragments = [f for f in self.fragments if f.emotional_tone == "positive"]
            if positive_fragments:
                return f"Success! {positive_fragments[-1].content}"
            return "Excellent work. Keep building on this momentum."
        
        elif last_outcome == "failure":
            growth_fragments = [f for f in self.fragments if "learn" in f.content.lower()]
            if growth_fragments:
                return f"Challenge encountered. {growth_fragments[0].content}"
            return "This didn't work as expected. Let me analyze and try a different approach."
        
        elif last_outcome == "uncertain":
            purpose_fragments = [f for f in self.fragments if f.role == "purpose"]
            if purpose_fragments:
                return f"Exploring options. {purpose_fragments[0].content}"
            return "I'm considering the best path forward."
        
        return "I continue to work toward my goals."
    
    def integrate_experience(
        self,
        experience: str,
        outcome: str,
        emotional_impact: float
    ) -> NarrativeFragment:
        """Integrate a new experience into the narrative"""
        content = experience[:200] if len(experience) > 200 else experience
        
        if emotional_impact > 0.5:
            tone = "positive" if outcome == "success" else "negative"
            importance = min(0.9, emotional_impact)
        else:
            tone = "neutral"
            importance = 0.4
        
        return self.add_fragment(
            role="experience",
            content=content,
            emotional_tone=tone,
            importance=importance
        )
    
    def get_identity_report(self) -> Dict[str, Any]:
        """Get comprehensive identity report"""
        return {
            "narrative_count": self.state.narrative_count,
            "roles": self.state.roles,
            "dominant_emotion": self.state.dominant_emotion,
            "coherence_score": self.state.coherence_score,
            "current_narrative": self.get_current_narrative(),
            "recent_fragments": [
                {
                    "role": f.role,
                    "content": f.content[:100],
                    "emotional_tone": f.emotional_tone,
                    "importance": f.importance,
                }
                for f in self.fragments[-5:]
            ]
        }