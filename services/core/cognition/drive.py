import uuid
from sqlalchemy import select
from database import AsyncSessionLocal
from models import Goal, Thought

async def generate_internal_drive():
    """
    Cognitive Loop: Generates thoughts based on goals without user input.
    """
    async with AsyncSessionLocal() as db:
        # 1. Check Active Goals
        try:
            stmt = select(Goal).where(Goal.status == "active")
            result = await db.execute(stmt)
            goals = result.scalars().all()
        except Exception as e:
            return f"Drive Error: {e}"
        
        if not goals:
            return "No active goals to drive."
            
        # 2. Pick a goal to focus on (Simplified: Pick first)
        focus_goal = goals[0]
        
        # 3. Generate a 'Curiosity' or 'Action' thought
        thought_content = f"INTERNAL DRIVE: I need to advance goal '{focus_goal.title}'. What is the next logical step?"
        
        # 4. Inject into Stream
        thought = Thought(content=thought_content, source="drive_engine", status="pending")
        db.add(thought)
        await db.commit()
        
        return f"Generated thought for {focus_goal.title}"
