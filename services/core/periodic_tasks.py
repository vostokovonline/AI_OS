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

    print("\n" + "="*70)
    print("[PERIODIC] Auto-decompose stuck goals")
    print("="*70)

    async def run_decompose():
        report = await auto_decomposer.scan_and_decompose_stuck_goals()

        print(f"\nScanned: {report['scanned']}")
        print(f"Decomposed: {report['decomposed']}")
        print(f"Skipped: {report['skipped']}")
        print(f"Failed: {report['failed']}")

        return report

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

    print("\n" + "="*70)
    print("[PERIODIC] Recalculate parent progress")
    print("="*70)

    async def run_recalculate():
        report = await parent_progress_aggregator.recalculate_all_parents()

        print(f"\nTotal parents: {report['total_parents']}")
        print(f"Updated: {report['updated']}")
        print(f"Completed: {report['completed']}")
        print(f"Errors: {report['errors']}")

        return report

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
            print(f"\n[TRIGGER] Updated parent {report.get('parent_id')}")
            print(f"  Progress: {report.get('old_progress'):.0%} → {report.get('new_progress'):.0%}")
            print(f"  Children: {report['children_stats']}")

        return report

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

    print("\n" + "="*70)
    print("[PERIODIC] Daily system health check")
    print("="*70)

    async def run_health_check():
        health = await acceleration_architecture.get_system_health()

        print(f"\nOverall State: {health['overall_state']}")
        print(f"Confidence: {health['confidence']:.0%}")

        balance = health['control_vs_acceleration']
        print(f"\nBalance: {balance['ratio']}")
        print(f"  Control: {balance['control']:.0%}")
        print(f"  Acceleration: {balance['acceleration']:.0%}")
        print(f"  {balance['interpretation']}")

        velocity = health['velocity']
        print(f"\nVelocity: {velocity['velocity_state']}")
        print(f"  Cycle time: {velocity['metrics']['avg_cycle_time_days']} days")
        print(f"  Completion rate: {velocity['metrics']['completion_rate_per_month']}/month")
        print(f"  Stagnation: {velocity['metrics']['stagnation_ratio']:.1%}")

        drift = health['drift']
        print(f"\nDrift: {drift['overall_status']}")
        print(f"  Drifts detected: {drift['drifts_detected']}")

        interventions = health['interventions']
        print(f"\nInterventions: {interventions['interventions_required']}")
        print(f"  By priority: {interventions['by_priority']}")

        # Log recommended actions
        if health['recommended_actions']:
            print(f"\n📋 Recommended Actions:")
            for action in health['recommended_actions']:
                print(f"  [{action['priority']}] {action['action']}")
                print(f"    {action['reason']}")

        return health

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
