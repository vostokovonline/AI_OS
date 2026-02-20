"""
Groq Cooldown Manager
Отслеживает ошибки Groq и вводит 10-часовой cooldown при проблемах
"""

import os
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

COOLDOWN_FILE = "/tmp/groq_cooldown.json"
COOLDOWN_HOURS = int(os.getenv("GROQ_COOLDOWN_HOURS", "10"))


class GroqCooldownManager:
    """Управляет cooldown периодом для Groq API"""

    def __init__(self):
        self.cooldown_file = Path(COOLDOWN_FILE)
        self._ensure_cooldown_file()

    def _ensure_cooldown_file(self):
        """Создает файл cooldown если не существует"""
        if not self.cooldown_file.exists():
            self._save_cooldown_data({
                "last_error_time": None,
                "error_count": 0,
                "in_cooldown": False
            })

    def _load_cooldown_data(self):
        """Загружает данные о cooldown"""
        try:
            with open(self.cooldown_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {
                "last_error_time": None,
                "error_count": 0,
                "in_cooldown": False
            }

    def _save_cooldown_data(self, data):
        """Сохраняет данные о cooldown"""
        with open(self.cooldown_file, 'w') as f:
            json.dump(data, f, indent=2)

    def is_in_cooldown(self):
        """Проверяет, находится ли Groq в cooldown периоде"""
        data = self._load_cooldown_data()

        if not data.get("in_cooldown"):
            return False

        last_error = data.get("last_error_time")
        if not last_error:
            return False

        # Проверяем, прошло ли 10 часов
        last_error_dt = datetime.fromisoformat(last_error)
        cooldown_end = last_error_dt + timedelta(hours=COOLDOWN_HOURS)

        if datetime.now() >= cooldown_end:
            # Cooldown закончился, сбрасываем
            self._save_cooldown_data({
                "last_error_time": None,
                "error_count": 0,
                "in_cooldown": False
            })
            return False

        return True

    def get_cooldown_remaining(self):
        """Возвращает время до конца cooldown в секундах"""
        data = self._load_cooldown_data()

        if not data.get("in_cooldown"):
            return 0

        last_error = data.get("last_error_time")
        if not last_error:
            return 0

        last_error_dt = datetime.fromisoformat(last_error)
        cooldown_end = last_error_dt + timedelta(hours=COOLDOWN_HOURS)
        remaining = (cooldown_end - datetime.now()).total_seconds()

        return max(0, int(remaining))

    def record_error(self):
        """Записывает ошибку Groq и активирует cooldown"""
        data = self._load_cooldown_data()
        error_count = data.get("error_count", 0) + 1

        self._save_cooldown_data({
            "last_error_time": datetime.now().isoformat(),
            "error_count": error_count,
            "in_cooldown": True
        })

        logger.info(f"⚠️  Groq error #{error_count} recorded. Activating {COOLDOWN_HOURS}-hour cooldown.")
        return COOLDOWN_HOURS * 3600  # Время cooldown в секундах

    def reset(self):
        """Сбрасывает cooldown вручную"""
        self._save_cooldown_data({
            "last_error_time": None,
            "error_count": 0,
            "in_cooldown": False
        })
        logger.info("✅ Groq cooldown reset manually")


# Global instance
_groq_cooldown = None


def get_groq_cooldown():
    """Возвращает глобальный экземпляр GroqCooldownManager"""
    global _groq_cooldown
    if _groq_cooldown is None:
        _groq_cooldown = GroqCooldownManager()
    return _groq_cooldown


def should_use_groq():
    """Проверяет, можно ли использовать Groq сейчас"""
    cooldown = get_groq_cooldown()
    return not cooldown.is_in_cooldown()


def record_groq_error():
    """Записывает ошибку Groq"""
    cooldown = get_groq_cooldown()
    return cooldown.record_error()


def get_groq_cooldown_status():
    """Возвращает статус cooldown для отображения"""
    cooldown = get_groq_cooldown()
    in_cooldown = cooldown.is_in_cooldown()
    remaining = cooldown.get_cooldown_remaining()

    if in_cooldown:
        hours = remaining // 3600
        minutes = (remaining % 3600) // 60
        return {
            "in_cooldown": True,
            "remaining_seconds": remaining,
            "remaining_human": f"{hours}h {minutes}m",
            "cooldown_hours": COOLDOWN_HOURS
        }
    else:
        return {
            "in_cooldown": False,
            "remaining_seconds": 0,
            "cooldown_hours": COOLDOWN_HOURS
        }


if __name__ == "__main__":
    # Тест
    cooldown = get_groq_cooldown()
    status = get_groq_cooldown_status()

    logger.info(f"Groq Cooldown Status: {json.dumps(status, indent=2)}")

    if status["in_cooldown"]:
        logger.info(f"⚠️  Groq is in cooldown for {status['remaining_human']}")
    else:
        logger.info("✅ Groq is available")
