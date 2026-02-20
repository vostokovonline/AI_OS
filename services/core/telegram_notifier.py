"""
TELEGRAM NOTIFIER MODULE

Отправляет уведомления о вопросах в Telegram.
"""
import os
import json
import logging
from typing import Optional, Dict, Any
import redis
import httpx

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CORE_API_URL = os.getenv("CORE_URL", "http://ns_core:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://ns_redis:6379/0")


class TelegramNotifier:
    """Отправляет уведомления в Telegram"""

    def __init__(self):
        self.enabled = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "your_bot_token_here")
        self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    def is_enabled(self) -> bool:
        """Проверяет включен ли Telegram"""
        return self.enabled

    def is_user_linked(self, user_id: str) -> bool:
        """Проверяет привязан ли пользователь к Telegram"""
        try:
            chat_id = self.redis_client.get(f"telegram:user_chat:{user_id}")
            return chat_id is not None
        except redis.RedisError as e:
            logger.debug("redis_error_checking_user_link", user_id=user_id, error=str(e))
            return False
        except Exception as e:
            logger.warning("unexpected_error_checking_user_link", user_id=user_id, error=str(e))
            return False

    async def send_question(self, user_id: str, question_data: Dict[str, Any]) -> bool:
        """
        Отправляет вопрос пользователю в Telegram

        Args:
            user_id: ID пользователя в системе
            question_data: Данные вопроса

        Returns:
            True если отправлено успешно
        """
        if not self.enabled:
            logger.debug("Telegram not enabled, skipping notification")
            return False

        if not self.is_user_linked(user_id):
            logger.info(f"User {user_id} not linked to Telegram")
            return False

        try:
            # Отправляем через API Telegram бота (internal endpoint)
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{CORE_API_URL}/telegram/send_question",
                    json={
                        "user_id": user_id,
                        "question_data": question_data
                    },
                    timeout=10.0
                )
                return response.status_code == 200

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False


# Глобальный инстанс
notifier = TelegramNotifier()


async def send_question_notification(user_id: str, question_data: Dict[str, Any]) -> bool:
    """
    Отправляет уведомление о вопросе в Telegram

    Usage:
        from telegram_notifier import send_question_notification

        await send_question_notification(
            user_id="user-123",
            question_data={
                "question": "What format?",
                "context": "Creating report",
                "priority": "normal",
                "artifact_id": "uuid-123",
                "timeout_at": "2026-01-18T20:00:00"
            }
        )
    """
    return await notifier.send_question(user_id, question_data)


if __name__ == "__main__":
    # Тест
    import asyncio

    async def test():
        tn = TelegramNotifier()
        logger.info(f"Telegram enabled: {tn.is_enabled()}")

    asyncio.run(test())
