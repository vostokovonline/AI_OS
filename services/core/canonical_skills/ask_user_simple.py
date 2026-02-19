"""
ask_user - простейший примитив для взаимодействия с человеком

Версия MVP без БД - просто генерирует question_id
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

class AskUserSimple:
    """
    Простейший ask_user - без БД
    
    В реальности:
    - Question создастся когда Telegram bot получит webhook
    - Или через UI
    """
    
    async def __call__(
        self,
        subject_type: str,
        subject_id: str,
        question: str,
        options: Optional[List[str]] = None,
        timeout_action: str = "continue_with_default",
        default_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Генерирует вопрос и возвращает question_id
        
        Вопрос "создаётся" когда пользователь отвечает через UI/Telegram
        
        Returns:
            {
                "status": "pending",
                "question_id": "...",
                "question": "...",
                "options": [...]
            }
        """
        question_id = str(uuid.uuid4())
        
        return {
            "status": "pending",
            "question_id": question_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "question": question,
            "options": options,
            "created_at": datetime.utcnow().isoformat()
        }

# Singleton
ask_user_simple = AskUserSimple()
