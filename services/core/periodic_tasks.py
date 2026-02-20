"""
PERIODIC TASKS - Background jobs for goal system maintenance
==============================================================

Scheduled tasks:
- Auto-decompose stuck goals (hourly)
- Parent progress aggregation (on child complete)
- System health checks (daily)

Author: AI-OS Core Team
Date: 2026-02-11
"""

from celery import Celery
from celery.schedules import crontab
import os

# Centralized logging
from logging_config import get_logger

logger = get_logger(__name__)

# Initialize Celery app
celery_app = Celery(
    'periodic_tasks',
    broker=os.getenv('CELERY_BROKER_URL', 'redis://ns_redis:6379/0'),
    backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://ns_redis:6379/1')
)


# =============================================================================
# PERIODIC TASKS
# =============================================================================

@celery_app.task(name='auto_decompose_stuck_goals')
def auto_decompose_stuck_goals():
    """
    Auto-decompose pending non-atomic goals

    Runs every hour
    Finds goals stuck > 1 hour in pending and decomposes them
    """
    import asyncio
    from auto_decomposer import auto_decomposer

    logger.info("auto_decompose_stuck_goals_started")

    async def run_decompose():
        report = await auto_decomposer.scan_and_decompose_stuck_goals()

        logger.info("auto_decompose_report",
                   scanned=report['scanned'],
                   decomposed=report['decomposed'],
                   skipped=report['skipped'],
                   failed=report['failed'])

        return report

    # Fix asyncio.run() anti-pattern
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_decompose())
                return future.result()
        else:
            return loop.run_until_complete(run_decompose())
    except RuntimeError:
        return asyncio.run(run_decompose())


@celery_app.task(name='recalculate_parent_progress')
def recalculate_parent_progress():
    """
    Recalculate parent progress for all parent goals

    Runs every 6 hours
    Ensures parent.progress reflects actual children status
    """
    import asyncio
    from parent_progress_aggregator import parent_progress_aggregator

    logger.info("recalculate_parent_progress_started")

    async def run_recalculate():
        report = await parent_progress_aggregator.recalculate_all_parents()

        logger.info("recalculate_parent_progress_report",
                   total_parents=report['total_parents'],
                   updated=report['updated'],
                   completed=report['completed'],
                   errors=report['errors'])

        return report

    # Fix asyncio.run() anti-pattern
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_recalculate())
                return future.result()
        else:
            return loop.run_until_complete(run_recalculate())
    except RuntimeError:
        return asyncio.run(run_recalculate())


@celery_app.task(name='update_parent_on_child_complete')
def update_parent_on_child_complete(child_goal_id: str):
    """
    Update parent progress when child goal completes

    Should be called from goal_executor when child goal done
    """
    import asyncio
    from parent_progress_aggregator import parent_progress_aggregator

    async def run_update():
        report = await parent_progress_aggregator.update_parent_progress(child_goal_id)

        if report.get('updated'):
            logger.info("parent_updated_on_child_complete",
                       parent_id=str(report.get('parent_id')),
                       old_progress=f"{report.get('old_progress'):.0%}",
                       new_progress=f"{report.get('new_progress'):.0%}",
                       children_stats=report['children_stats'])

        return report

    # Fix asyncio.run() anti-pattern
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_update())
                return future.result()
        else:
            return loop.run_until_complete(run_update())
    except RuntimeError:
        return asyncio.run(run_update())


# =============================================================================
# SCHEDULE CONFIGURATION
# =============================================================================

celery_app.conf.beat_schedule = {
    # Every hour: Auto-decompose stuck goals
    'auto-decompose-hourly': {
        'task': 'auto_decompose_stuck_goals',
        'schedule': crontab(minute=0),  # Every hour at XX:00
    },

    # Every 6 hours: Recalculate parent progress
    'recalculate-progress-every-6h': {
        'task': 'recalculate_parent_progress',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
    },

    # Daily at 9 AM: System health check
    'daily-health-check': {
        'task': 'acceleration_system_health_check',
        'schedule': crontab(minute=0, hour=9),  # Daily at 9:00 AM
    },
}


# =============================================================================
# SYSTEM HEALTH CHECK TASK
# =============================================================================

@celery_app.task(name='acceleration_system_health_check')
def acceleration_system_health_check():
    """
    Daily system health check using Acceleration Layer

    Collects metrics and logs summary
    """
    import asyncio
    from acceleration_architecture import acceleration_architecture

    logger.info("daily_system_health_check_started")

    async def run_health_check():
        health = await acceleration_architecture.get_system_health()

        logger.info("system_health_overview",
                   overall_state=health['overall_state'],
                   confidence=f"{health['confidence']:.0%}")

        balance = health['control_vs_acceleration']
        logger.info("control_acceleration_balance",
                   ratio=balance['ratio'],
                   control=f"{balance['control']:.0%}",
                   acceleration=f"{balance['acceleration']:.0%}",
                   interpretation=balance['interpretation'])

        velocity = health['velocity']
        logger.info("velocity_metrics",
                   state=velocity['velocity_state'],
                   cycle_time_days=velocity['metrics']['avg_cycle_time_days'],
                   completion_rate=velocity['metrics']['completion_rate_per_month'],
                   stagnation=f"{velocity['metrics']['stagnation_ratio']:.1%}")

        drift = health['drift']
        logger.info("drift_status",
                   overall_status=drift['overall_status'],
                   drifts_detected=drift['drifts_detected'])

        interventions = health['interventions']
        logger.info("interventions_required",
                   required=interventions['interventions_required'],
                   by_priority=interventions['by_priority'])

        # Log recommended actions
        if health['recommended_actions']:
            for action in health['recommended_actions']:
                logger.info("recommended_action",
                           priority=action['priority'],
                           action=action['action'],
                           reason=action['reason'])

        return health

    # Fix asyncio.run() anti-pattern
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, run_health_check())
                return future.result()
        else:
            return loop.run_until_complete(run_health_check())
    except RuntimeError:
        return asyncio.run(run_health_check())


# =============================================================================
# CONVENIENCE FUNCTION FOR MANUAL TRIGGER
# =============================================================================

def trigger_auto_decompose():
    """Manually trigger auto-decompose (for testing)"""
    auto_decompose_stuck_goals.delay()


def trigger_progress_recalc():
    """Manually trigger progress recalculation (for testing)"""
    recalculate_parent_progress.delay()


# =============================================================================
# STARTER
# =============================================================================

if __name__ == '__main__':
    # Run beat scheduler
    celery_app.start()
