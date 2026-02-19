from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict

class QualityReview(BaseModel):
    score: int
    is_acceptable: bool
    feedback: str

class SupervisorDecision(BaseModel):
    next_node: Literal["RESEARCHER", "CODER", "DESIGNER", "PM", "INTELLIGENCE", "COACH", "INNOVATOR", "LIBRARIAN", "DEVOPS", "FINISH"]
    reasoning: str

class Step(BaseModel):
    id: int
    description: str
    assigned_role: Literal["CODER", "RESEARCHER", "DESIGNER", "PM", "SKILL"]
    skill_name: Optional[str] = None
    status: str = "pending"

class Plan(BaseModel):
    steps: List[Step]
    final_goal: str
    reasoning: str

class Budget(BaseModel):
    total_steps: int = 15
    steps_used: int = 0
    allow_human_interaction: bool = True

class MetaEvaluation(BaseModel):
    process_score: int
    waste_detected: bool
    better_path_suggestion: str
