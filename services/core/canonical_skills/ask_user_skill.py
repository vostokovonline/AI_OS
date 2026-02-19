"""
ASK_USER SKILL - Universal Human-in-the-Loop Primitive

Architectural guarantees:
- NO Telegram-specific code
- NO waiting for answers
- NO decision-making
- Only initiates observation
"""

import uuid
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from models import DecompositionSession, DecompositionQuestion, Goal


class AskUserSkill:
    """
    Universal skill for asking user questions

    Usage:
        skill = AskUserSkill()
        result = await skill.run(
            goal_id=uuid.uuid4(),
            question_text="Какой первый шаг?",
            question_type="first_step",
            context={}
        )
    """

    name = "ask_user"

    def __init__(self):
        pass

    async def run(
        self,
        *,
        goal_id: uuid.UUID,
        question_text: str,
        question_type: Optional[str] = None,
        session_id: Optional[uuid.UUID] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run ask_user skill

        Args:
            goal_id: Target goal UUID
            question_text: Question text to ask
            question_type: Type of question (exploration, criteria, constraints, first_step)
            session_id: Existing session UUID (if continuing)
            context: Additional metadata

        Returns:
            {
                "type": "ask_user",
                "question_id": UUID,
                "session_id": UUID,
                "question_text": str,
                "question_index": int,
                "status": "awaiting_user"
            }
        """

        async with AsyncSessionLocal() as db:
            # Step 1: Find or create session
            if session_id:
                # Load existing session
                stmt = select(DecompositionSession).where(
                    DecompositionSession.id == session_id
                )
                result = await db.execute(stmt)
                session = result.scalar_one_or_none()

                if not session:
                    raise ValueError(f"Session {session_id} not found")

            else:
                # Create new session
                session = DecompositionSession(
                    goal_id=goal_id,
                    status="awaiting_user",
                    initiated_by="human"  # Explicitly human-initiated
                )
                db.add(session)
                await db.flush()  # Get session_id

            # Step 2: Get next question index
            stmt = select(DecompositionQuestion).where(
                DecompositionQuestion.session_id == session.id
            )
            result = await db.execute(stmt)
            existing_questions = result.scalars().all()
            question_index = len(existing_questions) + 1

            # Step 3: Create question
            question = DecompositionQuestion(
                session_id=session.id,
                question_text=question_text,
                question_index=question_index,
                asked_by="system",  # System asks, human answers
                question_type=question_type
            )
            db.add(question)
            await db.flush()  # Get question_id

            # Step 4: Commit
            await db.commit()

            # Step 5: Return result (NO waiting for answer!)
            return {
                "type": "ask_user",
                "question_id": str(question.id),
                "session_id": str(session.id),
                "question_text": question.question_text,
                "question_index": question.question_index,
                "question_type": question.question_type,
                "status": "awaiting_user",
                "created_at": question.created_at.isoformat()
            }


# Singleton instance
ask_user_skill = AskUserSkill()
