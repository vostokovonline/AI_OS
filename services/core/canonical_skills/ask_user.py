"""
ask_user - примитив для взаимодействия с человеком

КОНТРАКТ:
- НЕ вызывает LLM
- НЕ меняет граф
- НЕ принимает решения
- Только отправляет вопрос и фиксирует ответ
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class AskUser:
    """
    Примитив ask_user - механизм диалога с человеком
    
    Использует существующую систему вопросов (questions_store)
    для общения через Telegram бота
    """
    
    async def __call__(
        self,
        subject_type: str,  # "goal", "alert", "intervention"
        subject_id: str,
        question: str,
        options: Optional[List[str]] = None,
        timeout_action: str = "continue_with_default",
        default_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Задать вопрос человеку и ждать ответа
        
        Args:
            subject_type: тип сущности
            subject_id: ID сущности
            question: текст вопроса
            options: варианты ответа (опционально)
            timeout_action: что делать при таймауте
            default_answer: ответ по умолчанию
        
        Returns:
            {
                "status": "pending" | "answered" | "timeout",
                "question_id": "...",
                "answer": "...",  # если отвечен
                "free_text": "...",  # если есть свободный текст
                "created_at": "..."
            }
        """
        from models import Question
        from database import AsyncSessionLocal
        from sqlalchemy import select
        
        question_id = str(uuid.uuid4())
        
        async with AsyncSessionLocal() as db:
            # Создаём вопрос в БД
            db_question = Question(
                question_id=question_id,
                artifact_id=subject_id,  # временно используем artifact_id
                goal_id=subject_id,
                question=question,
                context=f"subject_type={subject_type}",
                options=options,
                priority="high",
                timeout_at=datetime.utcnow(),
                timeout_action=timeout_action,
                default_answer=default_answer,
                status="pending"
            )
            
            db.add(db_question)
            await db.commit()
        
        # TODO: Отправить уведомление через Telegram
        # Это делается отдельным процессом
        
        return {
            "status": "pending",
            "question_id": question_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def get_answer(self, question_id: str) -> Optional[Dict[str, Any]]:
        """
        Получить ответ на вопрос
        
        Returns:
            {
                "status": "answered",
                "answer": "...",
                "free_text": "...",
                "answered_at": "..."
            }
            или None если ещё не отвечен
        """
        from models import Question
        from database import AsyncSessionLocal
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as db:
            stmt = select(Question).where(Question.question_id == question_id)
            result = await db.execute(stmt)
            question = result.scalar_one_or_none()
            
            if not question:
                return None
            
            if question.status == "answered":
                return {
                    "status": "answered",
                    "question_id": question.question_id,
                    "answer": question.answer,
                    "free_text": question.free_text,
                    "answered_at": question.answered_at.isoformat() if question.answered_at else None
                }
            
            return {
                "status": question.status,
                "question_id": question.question_id
            }

# Singleton
ask_user = AskUser()
