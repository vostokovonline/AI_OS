"""
Emotional Context Helpers

Integration layer between Emotional Layer and Agent System.
"""

from typing import Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, case
from database import AsyncSessionLocal
from models import Goal
from schemas import EmotionalSignals


async def collect_emotional_signals(
    user_id: str,
    user_message: Optional[str] = None
) -> EmotionalSignals:
    """
    Collect aggregated signals for emotional inference.

    This is called by Context Builder before agent invocation.
    
    Args:
        user_id: User identifier
        user_message: Optional user text message
    
    Returns:
        EmotionalSignals with aggregated facts
    """

    async with AsyncSessionLocal() as db:
        # 1. Goal statistics (last 24h)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        
        # Count goals by status
        stmt_total = select(func.count(Goal.id)).where(
            and_(
                Goal.user_id == user_id,
                Goal.created_at >= cutoff
            )
        )
        
        result_total = await db.execute(stmt_total)
        total_count = result_total.scalar() or 0
        
        # Count completed
        stmt_completed = select(func.count(Goal.id)).where(
            and_(
                Goal.user_id == user_id,
                Goal.created_at >= cutoff,
                Goal.status == 'done'
            )
        )
        result_completed = await db.execute(stmt_completed)
        completed_count = result_completed.scalar() or 0
        
        # Count aborted
        stmt_aborted = select(func.count(Goal.id)).where(
            and_(
                Goal.user_id == user_id,
                Goal.created_at >= cutoff,
                Goal.status == 'aborted'
            )
        )
        result_aborted = await db.execute(stmt_aborted)
        aborted_count = result_aborted.scalar() or 0
        
        # Count active
        stmt_active = select(func.count(Goal.id)).where(
            and_(
                Goal.user_id == user_id,
                Goal.created_at >= cutoff,
                Goal.status == 'active'
            )
        )
        result_active = await db.execute(stmt_active)
        active_count = result_active.scalar() or 0
        
        # Build goal stats
        goal_stats = {
            "created": total_count,
            "completed": completed_count,
            "aborted": aborted_count,
            "active": active_count,
        }
        
        # Calculate success ratio
        goal_stats["success_ratio"] = completed_count / total_count if total_count > 0 else 0.5

        # 2. System metrics
        system_metrics = {
            "avg_goal_complexity": 0.5,  # TODO: Calculate from actual complexity
            "success_ratio": goal_stats.get("success_ratio", 0.5),
        }

        # 3. User text
        user_text = user_message

        return EmotionalSignals(
            user_text=user_text,
            goal_stats=goal_stats,
            system_metrics=system_metrics
        )


def format_emotional_context(emotional_context: Dict) -> str:
    """
    Format emotional context as string hints for LLM prompts.
    
    Args:
        emotional_context: {
            "complexity_limit": 0.6,
            "max_depth": 1,
            "exploration": "conservative",
            "explanation": "detailed",
            "pace": "slow"
        }
    
    Returns:
        Formatted string for prompt injection
    """
    
    hints = []
    
    if emotional_context.get("pace") == "slow":
        hints.append("- Be patient and supportive. Break down complex tasks into simple steps.")
    
    if emotional_context.get("explanation") == "detailed":
        hints.append("- Provide detailed explanations for each step.")
    
    if emotional_context.get("exploration") == "conservative":
        hints.append("- Stick to proven approaches. Avoid experimentation.")
    
    if emotional_context.get("max_depth", 3) <= 1:
        hints.append("- Keep decomposition simple. Maximum 1-2 subgoals only.")
    
    if emotional_context.get("complexity_limit", 1.0) < 0.7:
        hints.append("- Reduce task complexity. Focus on essential steps only.")
    
    if hints:
        return "\n".join(hints)
    else:
        return ""
