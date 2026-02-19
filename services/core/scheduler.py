import uuid
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from tasks import run_cron_task
from resource_manager import SystemMonitor
from cognition.drive import generate_internal_drive

scheduler = AsyncIOScheduler()
monitor = SystemMonitor()

async def cognitive_heartbeat():
    thought = await generate_internal_drive()
    if "No active goals" not in thought:
        print(f"💓 Heartbeat: {thought}")
        run_cron_task.delay(f"internal_{uuid.uuid4()}", thought)

async def execute_atomic_goals():
    """Автоматически выполняет незавершенные atomic goals каждые 5 минут"""
    from goal_executor_v2 import GoalExecutorV2
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select

    print(f"🎯 [Atomic Scheduler] Checking for incomplete atomic goals...")

    executor = GoalExecutorV2()
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(
            Goal.is_atomic == True
        ).where(
            Goal.progress < 1.0
        ).limit(3)  # Выполняем по 3 за раз

        goals = (await db.execute(stmt)).scalars().all()

        if not goals:
            print(f"   ℹ️  No incomplete atomic goals found")
            return

        print(f"   🎯 Found {len(goals)} incomplete atomic goals")

        for goal in goals:
            progress_pct = int(goal.progress * 100) if goal.progress else 0
            print(f"   ⚡ Executing: {goal.title[:60]}... ({progress_pct}%)")

            try:
                result = await executor.execute_goal(str(goal.id))

                if result.get("goal_complete"):
                    print(f"      ✅ COMPLETED: {goal.title[:50]}")
                else:
                    new_progress = int(result.get("progress", 0) * 100) if "progress" in result else progress_pct
                    print(f"      ⏳ In progress: {new_progress}%")
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:100]}")


async def auto_resume_pending_goals():
    """
    🔒 STATE-MACHINE FIX: Автоматически decomposes pending non-atomic цели без подцелей

    Инвариант: pending && is_atomic=false && child_count=0 → needs decomposition
    """
    import httpx
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select, func

    print(f"🔄 [Auto-Resume Scheduler] Checking for pending goals to decompose...")

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
            print(f"   ℹ️  No pending goals without sub-goals found")
            return

        print(f"   🔄 Found {len(goals)} pending goals to decompose")

        # Decompose through API
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout for decomposition
            for goal in goals:
                print(f"   ⚡ Decomposing: {goal.title[:60]}...")

                try:
                    response = await client.post(
                        f"http://localhost:8000/goals/{goal.id}/decompose",
                        json={"max_depth": 1},
                        timeout=600.0
                    )

                    if response.status_code == 200:
                        result = response.json()
                        subgoals_count = result.get("subgoals_created", 0)
                        print(f"      ✅ Created {subgoals_count} sub-goals")
                    else:
                        print(f"      ⚠️  HTTP {response.status_code}: {response.text[:100]}")
                except Exception as e:
                    print(f"      ❌ Error: {str(e)[:100]}")


async def decompose_non_atomic_goals():
    """
    🔒 STATE-MACHINE FIX: Автоматически decomposes active non-atomic цели без подцелей

    Инвариант: active && is_atomic=false && child_count=0 && depth<3 → needs decomposition
    """
    import httpx
    from database import AsyncSessionLocal
    from models import Goal
    from sqlalchemy import select, func

    print(f"🧩 [Decomposition Scheduler] Checking for active non-atomic goals to decompose...")

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
            print(f"   ℹ️  All active non-atomic goals have sub-goals")
            return

        print(f"   🧩 Found {len(goals)} active goals to decompose")

        for goal in goals:
            print(f"   🧩 Decomposing: {goal.title[:60]}...")

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
                            print(f"      ✅ Created {subgoals_count} sub-goals")
                        else:
                            print(f"      ⚠️  No sub-goals created")
                    else:
                        print(f"      ⚠️  HTTP {response.status_code}: {response.text[:100]}")

            except Exception as e:
                print(f"      ❌ Error: {str(e)[:100]}")

async def run_nightly_invariants_check():
    """
    🔒 NIGHTLY: Проверка всех инвариантов state-machine

    Запускается раз в сутки для проверки целостности системы.
    """
    from invariants_checker import run_invariants_check

    print(f"🔍 [Invariants Checker] Running nightly state-machine verification...")

    try:
        result = await run_invariants_check()

        if result["overall_status"] == "PASS":
            print(f"   ✅ All invariants PASS")
            print(f"   📊 {result['summary']['passed']}/{result['summary']['total_checks']} checks passed")
        elif result["overall_status"] == "VIOLATION":
            print(f"   ⚠️  INVARIANTS VIOLATION DETECTED!")
            print(f"   📊 {result['summary']['violations']} violations found")

            # Выводим детальную информацию
            for check in result['invariant_checks']:
                if check['status'] == 'VIOLATION':
                    print(f"\n   🔴 {check['invariant']}:")
                    print(f"      {check['message']}")
        else:  # ERROR
            print(f"   ❌ INvariants check ERROR: {result['summary']['errors']} errors")

    except Exception as e:
        print(f"   ❌ Error running invariants check: {str(e)}")


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

    scheduler.start()
    print("✅ Scheduler started:")
    print("   - Cognitive heartbeat: every 10 min")
    print("   - Atomic goals executor: every 5 min")
    print("   - Pending goals auto-resume: every 5 min")
    print("   - Decomposition scheduler: every 10 min")
    print("   🔒 Invariants check: daily at 3:00 AM")

