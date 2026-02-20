from logging_config import get_logger
logger = get_logger(__name__)

#!/usr/bin/env python3
"""
Периодическое обслуживание системы
Обновляет застаревшие цели чтобы предотвратить их зависание
"""

import asyncio
import aiohttp
import os
from datetime import datetime

CORE_URL = os.getenv("CORE_URL", "http://ns_core:8000")

async def maintenance_loop():
    """Основной цикл обслуживания"""

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                logger.info(f"[{datetime.utcnow().isoformat()}] Running maintenance...")

                # Обновляем застаревшие цели
                async with session.post(f"{CORE_URL}/goals/auto-update-stale") as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        logger.info(f"  ✅ Auto-updated {result.get('updated', 0)} stale goals")
                    else:
                        logger.info(f"  ⚠️  Failed to update stale goals: {resp.status}")

                # Проверяем застрявшие цели
                async with session.post(f"{CORE_URL}/goals/resume-all-stuck") as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        if result.get('resumed', 0) > 0:
                            logger.info(f"  ✅ Resumed {result.get('resumed', 0)} stuck goals")
                        else:
                            logger.info(f"  ✅ No stuck goals found")
                    else:
                        logger.info(f"  ⚠️  Failed to resume stuck goals: {resp.status}")

                logger.info(f"[{datetime.utcnow().isoformat()}] Maintenance complete")

                # Запуск каждые 30 минут
                await asyncio.sleep(1800)

            except Exception as e:
                logger.info(f"❌ Maintenance error: {e}")
                await asyncio.sleep(300)  # Retry after 5 minutes on error


if __name__ == "__main__":
    logger.info("🔧 AI-OS Maintenance Service")
    logger.info(f"Core URL: {CORE_URL}")
    logger.info("Starting maintenance loop...")
    asyncio.run(maintenance_loop())
