import uuid
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from tasks import run_cron_task
from resource_manager import SystemMonitor
from cognition.drive import generate_internal_drive

# Centralized logging
from logging_config import get_logger

logger = get_logger(__name__)

scheduler = AsyncIOScheduler()
monitor = SystemMonitor()

async def cognitive_heartbeat():
    thought = await generate_internal_drive()
    if "No active goals" not in thought:
        logger.info("cognitive_heartbeat", thought=thought)
        run_cron_task.delay(f"internal_{uuid.uuid4()}", thought)

async def execute_atomic_goals():
    """Автоматически выполняет незавершенные atomic goals каждые 5 минут"""
    from goal_executor_v2 import GoalExecutorV2
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select

    logger.info("atomic_scheduler_check_incomplete")

    executor = GoalExecutorV2()
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(
            Goal.is_atomic == True
        ).where(
            Goal.progress < 1.0
        ).limit(3)  # Выполняем по 3 за раз

        goals = (await db.execute(stmt)).scalars().all()

        if not goals:
            logger.debug("no_incomplete_atomic_goals")
            return

        logger.info("found_incomplete_atomic_goals", count=len(goals))

        for goal in goals:
            progress_pct = int(goal.progress * 100) if goal.progress else 0
            logger.info("executing_atomic_goal", title=goal.title[:60], progress=f"{progress_pct}%")

            try:
                result = await executor.execute_goal(str(goal.id))

                if result.get("goal_complete"):
                    logger.info("atomic_goal_completed", title=goal.title[:50])
                else:
                    new_progress = int(result.get("progress", 0) * 100) if "progress" in result else progress_pct
                    logger.debug("atomic_goal_in_progress", progress=f"{new_progress}%")
            except Exception as e:
                logger.error("atomic_goal_execution_error", error=str(e)[:100])


async def auto_resume_pending_goals():
    """
    🔒 STATE-MACHINE FIX: Автоматически decomposes pending non-atomic цели без подцелей

    Инвариант: pending && is_atomic=false && child_count=0 → needs decomposition
    """
    import httpx
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select, func

    logger.info("auto_resume_scheduler_check_pending")

    async with AsyncSessionLocal() as db:
        # 🔒 FIX: Убрали .limit(5) - теперь обрабатываем ВСЕ цели
        # Ищем только pending цели БЕЗ подцелей (state-machine инвариант)
        subquery = select(func.count(Goal.id)).where(Goal.parent_id == Goal.id)
        stmt = select(Goal).where(
            Goal.is_atomic == False
        ).where(
            Goal.status == 'pending'
        ).where(
            subquery == 0  # Только без подцелей!
        ).order_by(Goal.created_at.desc())  # Сначала старые

        goals = (await db.execute(stmt)).scalars().all()

        if not goals:
            logger.debug("no_pending_goals_without_children")
            return

        logger.info("found_pending_goals_to_decompose", count=len(goals))

        # Decompose through API
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout for decomposition
            for goal in goals:
                logger.info("decomposing_pending_goal", title=goal.title[:60])

                try:
                    response = await client.post(
                        f"http://localhost:8000/goals/{goal.id}/decompose",
                        json={"max_depth": 1},
                        timeout=600.0
                    )

                    if response.status_code == 200:
                        result = response.json()
                        subgoals_count = result.get("subgoals_created", 0)
                        logger.info("subgoals_created", count=subgoals_count)
                    else:
                        logger.warning("decompose_http_error",
                                      status_code=response.status_code,
                                      response=response.text[:100])
                except Exception as e:
                    logger.error("decompose_error", error=str(e)[:100])


async def decompose_non_atomic_goals():
    """
    🔒 STATE-MACHINE FIX: Автоматически decomposes active non-atomic цели без подцелей

    Инвариант: active && is_atomic=false && child_count=0 && depth<3 → needs decomposition
    """
    import httpx
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select, func

    logger.info("decomposition_scheduler_check_active")

    async with AsyncSessionLocal() as db:
        # 🔒 FIX: Убрали .limit(5), добавили subquery в WHERE
        # Ищем только active цели БЕЗ подцелей (state-machine инвариант)
        subquery = select(func.count(Goal.id)).where(Goal.parent_id == Goal.id)
        stmt = select(Goal).where(
            Goal.is_atomic == False
        ).where(
            Goal.status == 'active'  # Только active (не pending, те обрабатывает auto_resume)
        ).where(
            Goal.depth_level < 3  # Не глубже L3
        ).where(
            subquery == 0  # Только без подцелей!
        ).order_by(Goal.created_at.desc())

        goals = (await db.execute(stmt)).scalars().all()

        if not goals:
            logger.debug("all_active_goals_have_subgoals")
            return

        logger.info("found_active_goals_to_decompose", count=len(goals))

        for goal in goals:
            logger.info("decomposing_active_goal", title=goal.title[:60])

            try:
                # Use HTTP API with timeout instead of direct function call
                async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min for decomposition
                    response = await client.post(
                        f"http://localhost:8000/goals/{goal.id}/decompose",
                        json={"max_depth": 3},
                        timeout=600.0  # 10 min timeout for complex decomposition
                    )

                    if response.status_code == 200:
                        result = response.json()
                        subgoals_count = result.get("subgoals_created", 0)
                        if subgoals_count > 0:
                            logger.info("subgoals_created_for_active", count=subgoals_count)
                        else:
                            logger.warning("no_subgoals_created_for_active")
                    else:
                        logger.warning("decompose_active_http_error",
                                      status_code=response.status_code,
                                      response=response.text[:100])

            except Exception as e:
                logger.error("decompose_active_error", error=str(e)[:100])

async def run_nightly_invariants_check():
    """
    🔒 NIGHTLY: Проверка всех инвариантов state-machine

    Запускается раз в сутки для проверки целостности системы.
    """
    from invariants_checker import run_invariants_check

    logger.info("nightly_invariants_check_started")

    try:
        result = await run_invariants_check()

        if result["overall_status"] == "PASS":
            logger.info("all_invariants_pass",
                       passed=result['summary']['passed'],
                       total=result['summary']['total_checks'])
        elif result["overall_status"] == "VIOLATION":
            logger.warning("invariants_violation_detected",
                          violations=result['summary']['violations'])

            # Log detailed violations
            for check in result['invariant_checks']:
                if check['status'] == 'VIOLATION':
                    logger.error("invariant_violation",
                                invariant=check['invariant'],
                                message=check['message'])
        else:  # ERROR
            logger.error("invariants_check_error", errors=result['summary']['errors'])

    except Exception as e:
        logger.error("invariants_check_exception", error=str(e))


async def cleanup_memory_patterns():
    """
    🧠 MEMORY: Очистка старых паттернов с low confidence
    
    Запускается раз в сутки для удаления устаревших паттернов.
    """
    from semantic_memory import semantic_memory

    logger.info("memory_cleanup_started")

    try:
        # Cleanup patterns older than 30 days with low confidence
        deleted = await semantic_memory.cleanup_old_patterns(days=30)
        
        logger.info("memory_cleanup_completed", deleted_count=deleted)
        
        return {"deleted": deleted}
        
    except Exception as e:
        logger.error("memory_cleanup_error", error=str(e))
        return {"error": str(e)}


async def decay_memory_signals():
    """
    🧠 MEMORY: Decay всех MemorySignal
    
    Запускается каждый час для уменьшения TTL сигналов.
    """
    from memory_signal import memory_registry, persistent_memory_registry

    logger.info("memory_signal_decay_started")

    try:
        # Decay in-memory registry
        memory_registry.decay_all()
        local_count = len(memory_registry.get_active())
        
        # Redis registry handles TTL automatically
        redis_summary = persistent_memory_registry.summary()
        
        logger.info("memory_signal_decay_completed",
                   local_signals=local_count,
                   redis_signals=redis_summary.get("total_signals", 0))
        
        return {"local": local_count, "redis": redis_summary.get("total_signals", 0)}
        
    except Exception as e:
        logger.error("memory_signal_decay_error", error=str(e))
        return {"error": str(e)}


def start_scheduler():
    # Cognitive Loop every 10 mins
    scheduler.add_job(cognitive_heartbeat, 'interval', minutes=10)

    # Atomic Goals Executor every 5 mins
    scheduler.add_job(execute_atomic_goals, 'interval', minutes=5, id='atomic_executor')

    # Pending Goals Auto-Resume every 5 mins
    scheduler.add_job(auto_resume_pending_goals, 'interval', minutes=5, id='auto_resume')

    # Decomposition Scheduler every 10 mins
    scheduler.add_job(decompose_non_atomic_goals, 'interval', minutes=10, id='decompose_executor')

    # 🔒 STATE-MACHINE: Invariants check nightly (every 24h at 3 AM)
    scheduler.add_job(
        run_nightly_invariants_check,
        'cron',
        hour=3,
        minute=0,
        id='invariants_check'
    )

    # 🧠 MEMORY: Pattern cleanup nightly (every 24h at 4 AM)
    scheduler.add_job(
        cleanup_memory_patterns,
        'cron',
        hour=4,
        minute=0,
        id='memory_cleanup'
    )

    # 🧠 MEMORY: Signal decay hourly
    scheduler.add_job(
        decay_memory_signals,
        'interval',
        hours=1,
        id='memory_decay'
    )

    scheduler.start()
    logger.info("scheduler_started",
               cognitive_heartbeat="every 10 min",
               atomic_executor="every 5 min",
               auto_resume="every 5 min",
               decomposition="every 10 min",
               invariants_check="daily at 3:00 AM",
               memory_cleanup="daily at 4:00 AM",
               memory_decay="hourly")

