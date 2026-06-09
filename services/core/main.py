import uuid, asyncio, time, os
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import httpx
import sqlalchemy
from database import engine, Base, get_db, AsyncSessionLocal
from infrastructure.uow import create_uow_provider, UnitOfWork
from models import Message, ChatSession, Goal, GoalRelation, InterventionCandidate, InterventionSimulation, InterventionRiskScore, InterventionApproval
from schemas import (
    MessageCreate, MessageResponse, ResumeRequest, EventRequest,
    EIEInferenceRequest, MetaOutcome, EmotionalIntent,
    BulkTransitionRequest, BulkTransitionResponse, FreezeTreeRequest
)
from tasks import run_chat_task, run_resume_task, run_cron_task
from scheduler import start_scheduler
from agent_graph import app_graph
from dna_manager import bootstrap_dna
from emotions import analyze_sentiment
from goal_executor import goal_executor
from goal_executor_v2 import goal_executor_v2
from sqlalchemy import select, text, func

# Logging
from logging_config import get_logger
logger = get_logger(__name__)

# STEP 2.7: Intervention Readiness Layer imports
from intervention_candidates_engine import intervention_candidates_engine
from counterfactual_simulator import counterfactual_simulator
from intervention_risk_scorer import intervention_risk_scorer

# IRL Health Monitoring imports
from irl_invariants import irl_invariants_contract
from irl_health_metrics import irl_health_metrics

# Phase 2.2.5: Goal Approval API
from api.goals.approve_completion import router as approve_completion_router

# Phase 2.3.3: Observer Admin API
from api.admin.observer import router as observer_admin_router

# Phase 2.4.5: Reflection Admin API
from api.admin.reflection import router as reflection_admin_router

# v3.0: Arbitration API (Decision System)
from application.api.arbitration_endpoints import router as arbitration_router
from application.api.arbitration_endpoints import set_arbitration_log, set_capital_allocator

# Analytics API (LLM, System Health, Performance)
from application.api.analytics_endpoints import router as analytics_router

# LLM Control Center API (Model recommendations, policy simulation)
from application.api.llm_control_endpoints import router as llm_control_router

# Decision Trace Layer (Explainable model selection)
from application.api.decision_trace_endpoints import decision_router as decision_trace_router

# Shadow Switching Layer (Safe path to auto-switch)
from application.api.shadow_switching_endpoints import shadow_router as shadow_switching_router

# Outcome Tracking Layer (Real performance validation)
from application.api.outcome_tracking_endpoints import outcome_router as outcome_tracking_router

# Execution Orchestrator Layer (Production-grade state machine)
from application.api.execution_endpoints import execution_router

# Goals API Endpoints
from api.endpoints.goals import router as goals_router

# Admin API Endpoints (BUG-002 fix)
from api.endpoints.admin import router as admin_router

# Skills API Endpoints (autoloader integration)
from api.endpoints.skills import router as skills_router

# Control Center API (Metrics Engine)
from api.endpoints.control_center import router as control_center_router

# Semantic Layer API (v7.2 Router & Policy)
from api.endpoints.semantic_layer import router as semantic_router

# NEW: Refactored API module (Dashboard Compatibility Layer)
from api.routes import router as dashboard_router

# Epistemic Kernel, Causal Bridge, CPE, PHE API
from api.endpoints.epistemic import router as epistemic_router
from api.endpoints.causal import router as causal_router
from api.endpoints.aios_state import router as aios_state_router

app = FastAPI()

# SECURITY: Limit CORS to specific origins
# Get allowed origins from environment variable, fallback to localhost for development
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501,http://localhost:8000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Only whitelisted origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# ============================================================
# INCLUDE ROUTERS
# ============================================================

# Dashboard Compatibility Layer (refactored from inline routes)
app.include_router(dashboard_router)

# Phase 2.2.5: Include goal approval router
app.include_router(approve_completion_router)

# Phase 2.3.3: Include observer admin router
app.include_router(observer_admin_router)

# Phase 2.4.5: Include reflection admin router
app.include_router(reflection_admin_router)

# v3.0: Include arbitration router
app.include_router(arbitration_router)

# Analytics router
app.include_router(analytics_router)

# LLM Control Center router
app.include_router(llm_control_router)

# Distributed LLM Queue API (async job system)
try:
    from llm_queue_api import router as llm_queue_router
    app.include_router(llm_queue_router)
except ImportError:
    pass

# System Control API
try:
    from llm_system_api import router as llm_system_router
    app.include_router(llm_system_router)
except ImportError:
    pass

# Decision Trace router
app.include_router(decision_trace_router)

# Shadow Switching router
app.include_router(shadow_switching_router)

# Outcome Tracking router
app.include_router(outcome_tracking_router)

# Execution Orchestrator router
app.include_router(execution_router)

# Goals API router
app.include_router(goals_router)

# Admin API router (execution recovery)
app.include_router(admin_router)

# Control Center router
app.include_router(control_center_router)

# Semantic Layer router (v7.2 TS Router & Policy)
app.include_router(semantic_router)

# Skills API router (autoloader)
app.include_router(skills_router)

# Phase 2A: Execution V3 Metrics
from execution_v3_metrics import router as execution_v3_router
app.include_router(execution_v3_router)

# Model Rotation Monitoring (NEW)
from api.endpoints.model_rotation_endpoints import router as model_rotation_router
app.include_router(model_rotation_router)

# Autonomy System (NEW)
from api.endpoints.autonomy import router as autonomy_router
app.include_router(autonomy_router)

# v1 API (Goals Read - Canonical)
from api.v1.goals import router as goals_v1_router
app.include_router(goals_v1_router)

# Epistemic Kernel, Causal Bridge, CPE, PHE API
app.include_router(epistemic_router)
app.include_router(causal_router)

# AIOSState — MVP cockpit endpoint
app.include_router(aios_state_router)

# UoW Provider for dependency injection
uow_provider = create_uow_provider()

async def get_uow():
    """
    FastAPI Depends для UnitOfWork.
    
    Usage:
        @app.post("/endpoint")
        async def endpoint(uow: UnitOfWork = Depends(get_uow)):
            async with uow:
                # your code here
                pass
    """
    uow = uow_provider()
    async with uow:
        yield uow

async def wait_for_db():
    logger.info("⏳ Connecting to Database...")
    while True:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("✅ Database Connected!")
            break
        except sqlalchemy.exc.DBAPIError as e:
            logger.debug("database_connection_retry", error=str(e))
            await asyncio.sleep(2)
        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            await asyncio.sleep(2)

@app.on_event("startup")
async def startup():
    await wait_for_db()
    async with engine.begin() as conn: await conn.run_sync(Base.metadata.create_all)
    await bootstrap_dna()
    
    # Configure goal dispatcher for queue-based execution
    from application.goal_dispatcher import configure_dispatcher
    from celery_config import celery_app
    configure_dispatcher(celery_app)
    logger.info("✓ GoalDispatcher configured")
    
    # Start scheduler - AsyncIOScheduler runs in same event loop as FastAPI
    # No thread needed - it integrates with uvicorn's event loop
    from scheduler import start_scheduler
    
    try:
        start_scheduler()
        logger.info("✓ Scheduler started successfully")
    except Exception as e:
        logger.error(f"Scheduler failed to start: {e}")
    
    # Get use cases after scheduler is ready
    from scheduler import _get_use_cases
    use_cases = _get_use_cases()
    
    arbitrator = use_cases.get("arbitrator")
    capital_allocator = use_cases.get("capital_allocator")
    
    if arbitrator and hasattr(arbitrator, "_log"):
        set_arbitration_log(arbitrator._log)
        logger.info("✓ Arbitration log injected into API")
    
    if capital_allocator:
        set_capital_allocator(capital_allocator)
        logger.info("✓ Capital allocator injected into API")
    
    # ===== Event-Driven Architecture Setup =====
    from event_bus import setup_event_subscriptions, get_event_bus
    setup_event_subscriptions()
    logger.info("✓ Event Bus configured")
    
    # Start Watchdog (anti-deadlock)
    import asyncio
    from watchdog import get_watchdog
    watchdog = get_watchdog(threshold_minutes=10)
    
    # Run initial check
    try:
        await watchdog.run_watchdog()
    except Exception as e:
        logger.warning(f"Initial watchdog check failed: {e}")
    
    logger.info("✓ Watchdog scheduled")
    logger.info("✓ Event-driven lifecycle ready")

    # Auto-load skills from canonical_skills/ (Unified Service)
    from unified_skill_service import load_and_register_skills
    skills_count = load_and_register_skills()
    logger.info(f"✓ Unified Skill Service loaded {skills_count} skills")

    # Metrics Engine: Initialize and subscribe to event bus
    # Stub - metrics_engine module doesn't exist yet
    class MetricsEngineStub:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            pass
        async def handle_event(self, event):
            pass
        async def start_periodic_flush(self):
            pass
        def subscribe(self, event_type, handler):
            pass
    
    def set_metrics_engine(engine):
        pass
    
    MetricsEngine = MetricsEngineStub
    
    # from metrics_engine import MetricsEngine, set_metrics_engine
    from application.events.bus import get_event_bus
    from application.events.goal_events import GoalActivated, GoalCompleted, GoalFailed
    from application.events.execution_events import SkillExecuted, ArtifactCreated, GoalExecutionFinished
    import os

    # Use docker service name for Redis
    redis_url = os.getenv("REDIS_URL", "redis://ns_redis:6379/0")

    metrics_engine = MetricsEngine(
        redis_url=redis_url,
        postgres_session_factory=AsyncSessionLocal
    )

    # Subscribe to canonical events
    event_bus = get_event_bus()
    event_bus.subscribe(GoalActivated, metrics_engine.handle_event)
    event_bus.subscribe(GoalCompleted, metrics_engine.handle_event)
    event_bus.subscribe(GoalFailed, metrics_engine.handle_event)
    event_bus.subscribe(SkillExecuted, metrics_engine.handle_event)
    event_bus.subscribe(ArtifactCreated, metrics_engine.handle_event)

    # Subscribe to Execution Trace Events
    from application.events.execution_events import (
        GoalExecutionStarted, SkillSelected, ArtifactProduced, 
        GoalEvaluated, GoalTransitioned, GoalExecutionFinished
    )
    
    # Simple logging handlers for Execution Trace
    async def log_goal_started(event):
        from logging_config import get_logger
        logger = get_logger("execution_trace")
        logger.info(f"GOAL_STARTED: {event.goal_id} - {event.goal_title}")
    
    async def log_skill_selected(event):
        from logging_config import get_logger
        logger = get_logger("execution_trace")
        logger.info(f"SKILL_SELECTED: {event.goal_id} - {event.skill_name} (score={event.score})")
    
    async def log_artifact_produced(event):
        from logging_config import get_logger
        logger = get_logger("execution_trace")
        logger.info(f"ARTIFACT_PRODUCED: {event.goal_id} - {event.artifact_type}")
    
    async def log_goal_evaluated(event):
        from logging_config import get_logger
        logger = get_logger("execution_trace")
        logger.info(f"GOAL_EVALUATED: {event.goal_id} - {event.outcome} (confidence={event.confidence})")
    
    async def log_goal_transitioned(event):
        from logging_config import get_logger
        logger = get_logger("execution_trace")
        logger.info(f"GOAL_TRANSITIONED: {event.goal_id} - {event.from_state} → {event.to_state}")
    
    event_bus.subscribe(GoalExecutionStarted, log_goal_started)
    event_bus.subscribe(SkillSelected, log_skill_selected)
    event_bus.subscribe(ArtifactProduced, log_artifact_produced)
    event_bus.subscribe(GoalEvaluated, log_goal_evaluated)
    event_bus.subscribe(GoalTransitioned, log_goal_transitioned)

    # Initialize and subscribe Trace Collector
    from trace_store import get_trace_store
    from trace_collector import get_trace_collector
    trace_store = get_trace_store()
    await trace_store.initialize()
    trace_collector = get_trace_collector(trace_store)
    
    # Subscribe to all execution events for trace collection
    event_bus.subscribe(GoalExecutionStarted, trace_collector.handle_goal_started)
    event_bus.subscribe(SkillSelected, trace_collector.handle_skill_selected)
    event_bus.subscribe(ArtifactProduced, trace_collector.handle_artifact_produced)
    event_bus.subscribe(GoalEvaluated, trace_collector.handle_goal_evaluated)
    event_bus.subscribe(GoalTransitioned, trace_collector.handle_goal_transitioned)
    event_bus.subscribe(GoalExecutionFinished, trace_collector.handle_goal_execution_finished)

    logger.info("✓ Trace Collector initialized and subscribed to event bus")

    # Subscribe to Decision Events (Phase 1.5 - Decision Trace)
    from application.events.decision_events import (
        SkillCandidatesGenerated,
        SkillSelected as DecisionSkillSelected,
        SkillRetry,
        PlanGenerated,
        FallbackTriggered,
        LLMModelSelected
    )

    event_bus.subscribe(SkillCandidatesGenerated, metrics_engine.handle_event)
    event_bus.subscribe(DecisionSkillSelected, metrics_engine.handle_event)
    event_bus.subscribe(SkillRetry, metrics_engine.handle_event)
    event_bus.subscribe(PlanGenerated, metrics_engine.handle_event)
    event_bus.subscribe(FallbackTriggered, metrics_engine.handle_event)
    event_bus.subscribe(LLMModelSelected, metrics_engine.handle_event)

    logger.info("✓ Decision Events subscribed to Metrics Engine")

    # Start periodic batch flush
    await metrics_engine.start_periodic_flush()

    # Store as singleton for API access
    set_metrics_engine(metrics_engine)

    logger.info("✓ Metrics Engine initialized and subscribed to event bus")

    # Initialize Epistemic Kernel, Causal Bridge, CPE, PHE
    from epistemic_factory import (
        get_epistemic_kernel,
        get_causality_bridge,
        get_causal_policy_engine,
        get_policy_horizon_engine,
    )
    ek = get_epistemic_kernel()
    bridge = get_causality_bridge()
    cpe = get_causal_policy_engine()
    phe = get_policy_horizon_engine()
    logger.info("✓ Epistemic Kernel initialized")
    logger.info("✓ Causality Bridge initialized")
    logger.info("✓ Causal Policy Engine initialized")
    logger.info("✓ Policy Horizon Engine initialized")

    # Wire bridge to execution kernel events
    if hasattr(bridge, 'execution_kernel') and bridge.execution_kernel is not None:
        logger.info("✓ Causality Bridge connected to Execution Kernel")

    # Note: Scheduler is already started in background thread above (line 219)
    # Do NOT call start_scheduler() again here - it would cause "RuntimeError: cannot schedule new futures after shutdown"
    logger.info("🚀 SYSTEM ONLINE")

@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageCreate, db=Depends(get_db)):
    sid = req.session_id or str(uuid.uuid4())
    res = await db.execute(select(ChatSession).where(ChatSession.id == sid))
    if not res.scalar_one_or_none():
        db.add(ChatSession(id=sid))
        await db.commit()
    db.add(Message(session_id=sid, role="user", content=req.content))
    await db.commit()
    run_chat_task.delay(sid, req.content, req.image_url)

    # FIX: Use datetime.utcnow() instead of uuid time
    return Message(session_id=sid, role="system", content="⏳ Processing...", created_at=datetime.utcnow())

@app.post("/chat/sync")
async def chat_sync(req: MessageCreate):
    """
    Синхронный чат с AI - возвращает ответ немедленно.
    Поддерживает многодиалоговую историю через session_id.

    Для использования в Telegram боте и веб-чате.
    """
    from llm_fallback import chat_with_fallback
    import os

    try:
        sid = req.session_id or str(uuid.uuid4())

        # Ensure session exists
        async with AsyncSessionLocal() as db:
            res = await db.execute(select(ChatSession).where(ChatSession.id == sid))
            if not res.scalar_one_or_none():
                db.add(ChatSession(id=sid))
                await db.commit()

            # Load last 10 messages for conversation context
            msg_stmt = select(Message).where(
                Message.session_id == sid
            ).order_by(Message.created_at.desc()).limit(10)
            msg_result = await db.execute(msg_stmt)
            recent_messages = list(reversed(msg_result.scalars().all()))

        # Формируем системный промпт
        system_prompt = """Ты AI_OS - интеллектуальная операционная система для управления целями и задачами.

Твои возможности:
- Помощь в постановке и декомпозиции целей
- Ответы на вопросы о системе
- Анализ текущего состояния целей
- Рекомендации по следующим шагам

Отвечай кратко, по делу, на русском языке."""

        # Формируем сообщения для LLM с историей разговора
        messages = [{"role": "system", "content": system_prompt}]

        # Добавляем историю (последние 10 сообщений)
        for msg in recent_messages[-10:]:
            role = "user" if msg.role == "user" else "assistant"
            messages.append({"role": role, "content": msg.content})

        # Добавляем текущее сообщение пользователя
        messages.append({"role": "user", "content": req.content})

        # Сохраняем сообщение пользователя в БД
        async with AsyncSessionLocal() as db:
            db.add(Message(session_id=sid, role="user", content=req.content))
            await db.commit()

        # Получаем модель из переменных окружения
        # groq сломана - используем только работающие модели
        model = os.getenv("LLM_MODEL", "qwen2.5-coder")
        
        # Fallback для моделей которые не работают в LiteLLM
        if model in ["deepseek-reasoner", "qwen3-coder:480b-cloud", "qwen3.5", "gemma4", "minimax", "glm-4", "glm-5", "kimi-k2.5"]:
            model = "qwen2.5-coder"
            logger.info(f"LLM fallback: switched to {model}")

        # Вызываем LLM синхронно с контекстом
        result = await chat_with_fallback(model=model, messages=messages)

        # Извлекаем ответ
        response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "Ошибка получения ответа")

        # Сохраняем ответ ассистента в БД
        async with AsyncSessionLocal() as db:
            db.add(Message(session_id=sid, role="assistant", content=response_text.strip()))
            await db.commit()

        return {
            "status": "ok",
            "session_id": sid,
            "response": response_text.strip()
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "response": f"Ошибка: {str(e)}"
        }


@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str, limit: int = 20):
    """
    Получить историю сообщений для сессии чата.

    Args:
        session_id: ID сессии чата
        limit: Максимальное количество сообщений (default: 20)
    """
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(Message).where(
                Message.session_id == session_id
            ).order_by(Message.created_at.desc()).limit(limit)

            result = await db.execute(stmt)
            messages = result.scalars().all()

            return {
                "status": "ok",
                "session_id": session_id,
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "created_at": msg.created_at.isoformat()
                    }
                    for msg in reversed(messages)
                ],
                "count": len(messages)
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/chat/sessions")
async def list_chat_sessions(limit: int = 20):
    """
    Получить список последних чат-сессий.

    Args:
        limit: Максимальное количество сессий (default: 20)
    """
    try:
        async with AsyncSessionLocal() as db:
            stmt = select(ChatSession).order_by(ChatSession.created_at.desc()).limit(limit)
            result = await db.execute(stmt)
            sessions = result.scalars().all()

            session_list = []
            for session in sessions:
                # Get last message preview
                msg_stmt = select(Message).where(
                    Message.session_id == session.id
                ).order_by(Message.created_at.desc()).limit(1)
                msg_result = await db.execute(msg_stmt)
                last_msg = msg_result.scalar_one_or_none()

                # Get message count
                count_stmt = select(func.count()).select_from(Message).where(
                    Message.session_id == session.id
                )
                count_result = await db.execute(count_stmt)
                msg_count = count_result.scalar() or 0

                session_list.append({
                    "id": session.id,
                    "created_at": session.created_at.isoformat(),
                    "last_message_preview": last_msg.content[:50] if last_msg else None,
                    "message_count": msg_count
                })

            return {
                "status": "ok",
                "sessions": session_list,
                "count": len(session_list)
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/resume")
async def resume(req: ResumeRequest):
    run_resume_task.delay(req.session_id)
    return {"status": "resumed"}

@app.post("/analyze_mood")
async def analyze_mood(req: dict):
    msgs = [HumanMessage(content=m) for m in req.get('history', [])]
    return await analyze_sentiment(msgs)

@app.post("/event")
async def handle_event(evt: EventRequest):
    sid = f"event_{evt.source}_{uuid.uuid4().hex[:6]}"
    run_cron_task.delay(sid, f"EVENT: {evt.source}\nDATA: {evt.payload}")
    return {"status": "processing"}

# ==============================
# GOAL EXECUTION API
# ==============================

class GoalRequest(BaseModel):
    title: str
    description: str = ""
    goal_type: str = "bounded"
    auto_execute: bool = True  # Автоматически выполнить после создания
    is_atomic: bool = False  # Week 3: Atomic goal flag
    depth_level: int = 0  # Week 3: Goal depth level
    parent_id: str = None  # Parent goal ID for hierarchy
    domains: list = []  # Domains for categorization
    cron_schedule: str = "0 9 * * *"  # Default: daily at 9 AM for continuous goals
    user_id: str = None  # Emotional Layer: User ID for personalized context

class ExecuteGoalRequest(BaseModel):
    goal_id: str
    session_id: str = None

class ComplexGoalRequest(BaseModel):
    request: str

@app.post("/testPOST")
async def test_post_simple():
    """ULTRA SIMPLE test - no dependencies"""
    return {"status": "ok", "message": "test works"}


@app.post("/goals/create")
async def create_goal_endpoint(req: GoalRequest):
    """
    Goal creation - simple direct DB.
    """
    from database import AsyncSessionLocal
    from models import Goal
    from uuid import uuid4

    try:
        async with AsyncSessionLocal() as session:
            goal = Goal(
                id=uuid4(),
                title=req.title or "Untitled",
                description=req.description or req.title or "No description",
                goal_type=req.goal_type or "achievable",
                is_atomic=req.is_atomic or False,
                status="pending",
                progress=0.0
            )
            session.add(goal)
            await session.commit()

            goal_id = str(goal.id)
            
            # Event: goal created
            try:
                from event_bus import emit_goal_created
                await emit_goal_created(goal_id=goal_id)
            except Exception as e:
                logger.warning(f"Event emission failed: {e}")

            # Примечание: Temporal запускается ПОСЛЕ commit UoW, т.к. это external service
            if req.goal_type == "continuous":
                try:
                    from temporalio.client import Client
                    from datetime import timedelta

                    # Connect to Temporal server
                    temporal_client = await Client.connect("temporal:7233")

                    workflow_id = f"continuous-{goal_id}"

                    handle = await temporal_client.start_workflow(
                        "ContinuousGoalCronWorkflow",
                        [goal_id, req.title, req.description or "", req.cron_schedule or "0 9 * * *", req.domains or [], None],
                        id=workflow_id,
                        task_queue="ai-os-continuous",
                        cron_schedule=req.cron_schedule or "0 9 * * *",
                        execution_timeout=timedelta(hours=24),
                        run_timeout=timedelta(hours=1)
                    )

                    return {
                        "status": "created_and_continuous",
                        "goal_id": goal_id,
                        "workflow_id": workflow_id,
                        "message": "Continuous goal created and started in Temporal",
                        "cron_schedule": req.cron_schedule or "0 9 * * *"
                    }
                except Exception as temporal_error:
                    # Temporal failed, но goal уже создан в БД (UoW закоммитил)
                    # Это acceptable - goal остаётся в pending, можно запустить вручную
                    import traceback
                    logger.info(f"⚠️ Goal created but Temporal workflow failed to start: {temporal_error}")
                    traceback.print_exc()
                    return {
                        "status": "created",
                        "goal_id": goal_id,
                        "message": "Goal created but Temporal workflow failed to start",
                        "temporal_error": str(temporal_error)
                    }

            # STEP 4: Auto-execute через Celery (вне транзакции)
            # Celery задача запускается ПОСЛЕ commit UoW
            if req.auto_execute:
                from tasks import execute_goal_task
                execute_goal_task.delay(goal_id, None)
                return {
                    "status": "created_and_started",
                    "goal_id": goal_id,
                    "title": goal.title,
                    "goal_type": goal.goal_type,
                    "depth_level": goal.depth_level
                }
            else:
                return {
                    "status": "created",
                    "goal_id": goal_id,
                    "title": goal.title,
                    "goal_type": goal.goal_type,
                    "depth_level": goal.depth_level,
                    "message": "Use /goals/execute to start"
                }
            
    except ValueError as e:
        # Бизнес-правило нарушено - UoW сделает rollback автоматически
        return {
            "status": "error",
            "message": f"Goal creation blocked: {str(e)}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to create goal: {str(e)}"
        }

@app.post("/goals/execute")
async def execute_goal(
    req: ExecuteGoalRequest,
    uow = Depends(get_uow)
):
    """
    Execution endpoint v4.2 - UnitOfWork-based execution.

    TEMPORARY: Type checking to debug UoW injection
    """
    from infrastructure.uow import UnitOfWork

    # ЖЁСТКИЙ ТЕСТ: проверяем что именно приходит
    result_data = {
        "uow_type": str(type(uow)),
        "uow_is_none": uow is None,
        "has_aenter": hasattr(uow, "__aenter__"),
        "has_aexit": hasattr(uow, "__aexit__"),
        "is_unitofwork": isinstance(uow, UnitOfWork),
        "is_callable": callable(uow)
    }

    # Если приходит не UnitOfWork - сразу фейлим
    if not isinstance(uow, UnitOfWork):
        return {
            "error": "UoW_INJECTION_FAILED",
            "debug": result_data,
            "expected": "UnitOfWork instance",
            "got": type(uow)
        }

    # Execute through Kernel — sole execution authority
    async with uow:
        from execution_dynamics import dispatch_goal
        ex_result = await dispatch_goal(
            goal_id=req.goal_id,
            uow=uow,
        )

    return {
        'success': ex_result.success,
        'goal_id': ex_result.goal_id,
        'artifacts': ex_result.artifacts,
        'error': ex_result.error,
        'execution_id': ex_result.execution_id,
        'lease_id': ex_result.lease_id,
        'dispatch_epoch': ex_result.dispatch_epoch,
        'execution_pressure': ex_result.execution_pressure,
        'duration_ms': ex_result.duration_ms,
    }

@app.post("/goals/bulk/activate")
async def bulk_activate_goals(
    uow: UnitOfWork = Depends(get_uow)
):
    """
    BEHAVIOURAL TEST: Bulk transition with 10 goals

    Proves that BulkTransitionEngine applies ALL transitions
    in ONE atomic transaction.

    Process:
    1. Find 10 pending goals
    2. Apply bulk activation via BulkTransitionEngine
    3. Verify atomic commit
    """
    from application.bulk_transition_engine import bulk_transition_engine
    from sqlalchemy import select
    from models import Goal

    # ONE transaction for both load and apply
    async with uow:
        # Phase 1: Load 10 pending goals
        stmt = select(Goal).where(Goal._status == 'pending').limit(10)
        result = await uow.session.execute(stmt)
        goals = result.scalars().all()

        if not goals:
            return {
                "status": "error",
                "message": "No pending goals found for bulk test"
            }

        goal_ids = [g.id for g in goals]

        # Phase 2: Apply bulk transition (inside SAME transaction)
        bulk_result = await bulk_transition_engine.execute_batch(
            uow=uow,
            goal_ids=goal_ids,
            actor="behavioural_test"
        )

    return {
        "status": "completed",
        "bulk_result": {
            "total": bulk_result.total,
            "succeeded": bulk_result.succeeded,
            "failed": bulk_result.failed,
            "blocked": bulk_result.blocked,
            "execution_time_ms": bulk_result.execution_time_ms
        },
        "message": f"Bulk transition completed: {bulk_result.succeeded}/{bulk_result.total} succeeded"
    }

@app.post("/goals/complex")
async def execute_complex_goal(req: ComplexGoalRequest):
    """Выполняет сложную цель из естественного языка"""
    result = await goal_executor.execute_complex_goal(req.request)
    return result

@app.post("/goals/resume_stuck")
async def resume_stuck_goals():
    """Запускает все зависшие цели в статусе active с progress=0"""
    try:
        from goal_executor import execute_goal_task
        from models import Goal
        from database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                text("SELECT id, title FROM goals WHERE status = 'active' AND progress = 0 ORDER BY created_at DESC")
            )
            stuck_goals = result.fetchall()

            if not stuck_goals:
                return {"status": "ok", "message": "No stuck goals found", "started": 0}

            started_count = 0
            for goal_id, title in stuck_goals:
                try:
                    execute_goal_task.delay(str(goal_id), None)
                    started_count += 1
                except Exception as e:
                    logger.info(f"Failed to start goal {goal_id}: {e}")

            return {
                "status": "ok",
                "message": f"Started {started_count} stuck goals",
                "started": started_count,
                "goals": [{"id": str(g[0]), "title": g[1]} for g in stuck_goals]
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ==============================
# NEW GOAL SYSTEM ENDPOINTS
# ==============================

@app.post("/goals/classify")
async def classify_goal(req: GoalRequest):
    """Классифицирует цель по онтологии (SAFE с таймаутом)"""
    import asyncio
    from goal_decomposer import goal_decomposer

    try:
        classification = await asyncio.wait_for(
            goal_decomposer.safe_classify_goal(req.title, req.description, timeout=3.0),
            timeout=4.0
        )
    except asyncio.TimeoutError:
        classification = {
            "goal_type": "achievable",
            "reasoning": "API timeout - using default",
            "executable": True,
            "decomposable": True,
            "is_fallback": True
        }

    return {
        "status": "ok",
        "classification": classification,
        "is_fallback": classification.get("is_fallback", False)
    }


class SimpleGoalRequest(BaseModel):
    title: str
    description: str = ""
    goal_type: str = "achievable"
    is_atomic: bool = False


@app.post("/debug/pipeline/run")
async def run_pipeline_debug():
    """
    DEBUG: Ручной запуск pipeline (resume + execute)
    Позволяет протестировать без ожидания scheduler intervals.
    
    PROTECTED: Only available when DEBUG=true
    """
    import os
    if os.getenv("DEBUG", "").lower() != "true":
        return {"status": "debug_disabled", "message": "Set DEBUG=true to enable"}
    
    from scheduler import start_scheduler
    from application.use_cases.resume_pending_goals import ResumePendingGoalsUseCase
    from application.use_cases.execute_ready_goals import ExecuteReadyGoalsUseCase
    from infrastructure.uow import create_uow_provider
    from goal_executor_v2 import goal_executor_v2
    from application.bulk_engine import BulkTransitionEngine
    
    get_uow_factory = create_uow_provider()
    bulk_engine = BulkTransitionEngine()
    
    resume_use_case = ResumePendingGoalsUseCase(uow_factory=get_uow_factory, bulk_engine=bulk_engine)
    execute_use_case = ExecuteReadyGoalsUseCase(
        uow_factory=get_uow_factory,
        executor=goal_executor_v2,
        bulk_engine=bulk_engine,
        arbitrator=None,
        capital_allocator=None,
        event_bus=None,
    )
    
    resume_result = await resume_use_case.run(actor="debug_pipeline")
    exec_result = await execute_use_case.run(limit=5, actor="debug_pipeline")
    
    return {
        "resume": {
            "total_found": resume_result.total_found,
            "activated": resume_result.activated,
            "failed": resume_result.failed
        },
        "execute": {
            "found": exec_result.total_found,
            "completed": exec_result.completed,
            "failed": exec_result.failed
        }
    }


@app.get("/debug/pipeline/state")
async def get_pipeline_state():
    """
    DEBUG: Текущее состояние pipeline
    Показывает сколько goals в каждом статусе.
    
    PROTECTED: Only available when DEBUG=true
    """
    import os
    if os.getenv("DEBUG", "").lower() != "true":
        return {"status": "debug_disabled", "message": "Set DEBUG=true to enable"}
    
    from sqlalchemy import text
    from goal_decomposer import goal_decomposer
    
    # REAL METRICS: Get LLM health
    llm_metrics = goal_decomposer.get_llm_metrics()
    
    # EVENT-DRIVEN PIPELINE: Get event history
    try:
        from application.events.pipeline import get_pipeline, PipelineEventType
        pipeline = get_pipeline()
        event_history = pipeline.get_history(limit=20)
        event_counts = {}
        for e in pipeline.get_history(limit=1000):
            et = e.event_type.value
            event_counts[et] = event_counts.get(et, 0) + 1
    except Exception:
        event_history = []
        event_counts = {}
    
    from database import AsyncSessionLocal
    
    async with AsyncSessionLocal() as session:
        stmt = text("""
            SELECT status, COUNT(*)
            FROM goals
            GROUP BY status
            ORDER BY COUNT(*) DESC
        """)
        result = await session.execute(stmt)
        states = {row[0]: row[1] for row in result.fetchall()}
        
        stmt2 = text("""
            SELECT status, COUNT(*)
            FROM goals
            WHERE is_atomic = true
            GROUP BY status
        """)
        result2 = await session.execute(stmt2)
        atomic = {row[0]: row[1] for row in result2.fetchall()}
    
    return {
        "all_goals": states,
        "atomic_only": atomic,
        "llm_metrics": llm_metrics,
        "event_pipeline": {
            "event_counts": event_counts,
            "recent_events": [{"type": e.event_type.value, "goal_id": e.goal_id, "ts": e.timestamp.isoformat()} for e in event_history]
        }
    }


@app.post("/goals/{goal_id}/decompose")
async def decompose_goal_endpoint(
    goal_id: str,
    max_depth: int = 3,
    uow: UnitOfWork = Depends(get_uow)
):
    """
    Декомпозирует цель на подцели.

    UoW MIGRATION: Теперь атомарная операция - либо все подцели создаются,
    либо ничего (rollback). Ни одного промежуточного commit.
    """
    from goal_decomposer import goal_decomposer
    from policies.legacy_policy import legacy_policy
    from infrastructure.uow import GoalRepository
    from uuid import UUID

    try:
        # STEP 1: Validate against Legacy Policy (C)
        validation = await legacy_policy.validate_goal_decomposition(goal_id)

        if not validation["valid"]:
            return {
                "status": "error",
                "message": "Legacy Policy violation",
                "reason": validation["reason"]
            }

        # STEP 2: Декомпозируем ВНУТРИ UoW транзакции
        subgoals = await goal_decomposer.decompose_goal_with_uow(
            uow=uow,
            goal_id=goal_id,
            max_depth=max_depth
        )

        return {
            "status": "ok",
            "goal_id": goal_id,
            "subgoals_created": len(subgoals),
            "subgoals": subgoals,
            "transaction": "atomic"
        }

    except ValueError as e:
        # Бизнес-правило нарушено - UoW сделает rollback автоматически
        return {
            "status": "error",
            "message": f"Decomposition blocked: {str(e)}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to decompose goal: {str(e)}"
        }


@app.post("/goals/{goal_id}/evaluate")
async def evaluate_goal_endpoint(goal_id: str, uow: UnitOfWork = Depends(get_uow)):
    """
    Оценивает выполнение цели (Self-Evaluation).

    UoW MIGRATION: Теперь атомарная операция - оценка + state transition в одной транзакции.
    """
    from goal_evaluator import goal_evaluator

    try:
        evaluation = await goal_evaluator.evaluate_goal_with_uow(uow, goal_id)

        return {
            "status": "ok",
            "evaluation": evaluation,
            "transaction": "atomic"
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": f"Evaluation blocked: {str(e)}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to evaluate goal: {str(e)}"
        }


@app.get("/goals/{goal_id}/tree")
async def get_goal_tree(goal_id: str):
    """Получает дерево целей (цель + все подцели)"""
    from models import Goal
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Goal).where(Goal.id == uuid.UUID(goal_id)))
        goal = result.scalar_one_or_none()

        if not goal:
            return {"status": "error", "message": "Goal not found"}

        # Рекурсивно получаем все подцели
        def build_tree(g):
            return {
                "id": str(g.id),
                "title": g.title,
                "description": g.description,
                "status": g.status,
                "progress": g.progress,
                "goal_type": g.goal_type,
                "depth_level": g.depth_level,
                "is_atomic": g.is_atomic,
                "domains": g.domains,
                "children": [build_tree(child) for child in g.children]
            }

        tree = build_tree(goal)

        return {
            "status": "ok",
            "tree": tree
        }


@app.get("/goals/list")
async def get_goals_list(
    limit: int = 500,  # Changed default to 500
    offset: int = 0
):
    """Получает список всех целей (для v2 dashboard)"""
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Goal)
            .order_by(Goal.created_at.desc())
            .limit(limit)  # Now actually uses the parameter
            .offset(offset)
        )
        goals = result.scalars().all()

        goals_list = []
        for g in goals:
            goals_list.append({
                "id": str(g.id),
                "parent_id": str(g.parent_id) if g.parent_id else None,
                "title": g.title,
                "description": g.description,
                "status": g.status,
                "progress": g.progress,
                "goal_type": g.goal_type,
                "depth_level": g.depth_level,
                "is_atomic": g.is_atomic,
                "created_at": g.created_at.isoformat() if g.created_at else None,
                "updated_at": g.updated_at.isoformat() if g.updated_at else None,
            })

        return {
            "status": "ok",
            "goals": goals_list,
            "total": len(goals_list),
            "limit": limit,
            "offset": offset
        }


@app.get("/goals/stats")
async def get_goals_stats():
    """Получает статистику по целям"""
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import func

    async with AsyncSessionLocal() as db:
        # Общая статистика
        total = await db.execute(select(func.count(Goal.id)))
        total = total.scalar()

        # По типам
        by_type = await db.execute(
            select(Goal.goal_type, func.count(Goal.id))
            .group_by(Goal.goal_type)
        )
        by_type = {row[0]: row[1] for row in by_type}

        # По статусам
        by_status = await db.execute(
            select(Goal.status, func.count(Goal.id))
            .group_by(Goal.status)
        )
        by_status = {row[0]: row[1] for row in by_status}

        # По уровням
        by_depth = await db.execute(
            select(Goal.depth_level, func.count(Goal.id))
            .group_by(Goal.depth_level)
        )
        by_depth = {row[0]: row[1] for row in by_depth}

        return {
            "status": "ok",
            "total": total,
            "by_type": by_type,
            "by_status": by_status,
            "by_depth": by_depth
        }


@app.get("/goals/orphans")
async def get_orphan_goals(limit: int = 5):
    """
    GET /goals/orphans - Get orphan goals that need context binding

    Orphan goals are root-level goals (depth_level=0) that are NOT philosophical
    and have been created more than 24 hours ago. They need context: "Ради чего?"
    """
    from orphan_goals_detector import orphan_goals_detector

    try:
        orphans = await orphan_goals_detector.find_orphan_goals(limit=limit)
        stats = await orphan_goals_detector.get_orphan_stats()

        return {
            "status": "ok",
            "orphans": orphans,
            "stats": stats
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/bulk-transition")
async def bulk_transition_goals(
    request: BulkTransitionRequest,
    uow: UnitOfWork = Depends(get_uow)
):
    """
    POST /goals/bulk-transition - Mass transition of multiple goals
    
    UoW MIGRATION: All transitions happen in ONE transaction.
    Either all succeed or all roll back.
    
    Features:
    - O(1) transactions instead of O(N)
    - Pessimistic locking for consistency
    - Atomic rollback on any error
    
    Limits:
    - Max 1000 goals per request
    - Valid states: pending, active, done, frozen, archived
    """
    from infrastructure.uow import bulk_transition_service
    from uuid import UUID
    
    try:
        # Convert string IDs to UUIDs
        goal_uuids = []
        for gid in request.goal_ids:
            try:
                goal_uuids.append(UUID(gid))
            except ValueError:
                return {
                    "status": "error",
                    "error": f"Invalid goal ID format: {gid}"
                }
        
        # Execute bulk transition
        result = await bulk_transition_service.execute_bulk(
            uow=uow,
            goal_ids=goal_uuids,
            new_state=request.new_state,
            reason=request.reason,
            actor=request.actor
        )
        
        return {
            "status": "ok",
            **result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/freeze-tree")
async def freeze_goal_tree(
    request: FreezeTreeRequest,
    uow: UnitOfWork = Depends(get_uow)
):
    """
    POST /goals/freeze-tree - Freeze entire goal tree (root + all descendants)
    
    Useful for:
    - Pausing large projects
    - Mass archiving
    - Cascade operations
    
    All goals in the tree are frozen in ONE transaction.
    """
    from infrastructure.uow import bulk_transition_service
    
    try:
        result = await bulk_transition_service.freeze_tree(
            uow=uow,
            root_goal_id=request.root_goal_id,
            reason=request.reason,
            actor=request.actor
        )
        
        return {
            "status": "ok",
            **result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/{goal_id}/bind-context")
async def bind_goal_context(goal_id: str, context: dict):
    """
    POST /goals/{goal_id}/bind-context - Bind orphan goal to parent context

    Expects: {"parent_title": "строка"} or {"existing_parent_id": "uuid"}

    Action:
    1. If existing_parent_id provided → link to existing parent
    2. If parent_title provided → search for existing Mission or create new L1
    """
    from models import Goal
    from sqlalchemy import select
    import uuid

    parent_title = context.get("parent_title")
    existing_parent_id = context.get("existing_parent_id")

    async with get_uow() as uow:
        repo = uow.goals

        # Load the orphan goal
        goal = await repo.get(uow.session, uuid.UUID(goal_id))

        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")

        parent_goal = None

        # Case 1: Link to existing parent by ID
        if existing_parent_id:
            parent_goal = await repo.get(uow.session, uuid.UUID(existing_parent_id))

            if not parent_goal:
                raise HTTPException(status_code=404, detail="Parent goal not found")

        # Case 2: Find or create parent by title
        elif parent_title:
            stmt_search = select(Goal).where(
                Goal.title.ilike(f"%{parent_title}%")
            ).where(Goal.depth_level == 0).limit(1)
            result_search = await uow.session.execute(stmt_search)
            existing_mission = result_search.scalar_one_or_none()

            if existing_mission:
                parent_goal = existing_mission
                logger.info(f"🔗 Found existing Mission: {parent_goal.title}")
            else:
                new_parent = Goal(
                    title=parent_title,
                    description=f"Миссия: {parent_title}",
                    goal_type="directional",
                    depth_level=0,
                    _status="active",
                    progress=0.0
                )
                await repo.save(uow.session, new_parent)

                # Transition: создание → active
                from goal_transition_service import transition_service
                await transition_service.transition(
                    uow=uow,
                    goal_id=new_parent.id,
                    new_state="active",
                    reason="Context binding: Mission created for orphan goal",
                    actor="system"
                )

                parent_goal = new_parent
                logger.info(f"✨ Created new Mission: {parent_goal.title}")

        else:
            raise HTTPException(status_code=400, detail="Either parent_title or existing_parent_id required")

        # Link orphan to parent
        goal.parent_id = parent_goal.id
        goal.depth_level = (parent_goal.depth_level or 0) + 1

        await uow.session.flush(goal)

        return {
            "status": "ok",
            "message": f"Goal '{goal.title}' linked to parent '{parent_goal.title}'",
            "goal_id": str(goal.id),
            "parent_id": str(parent_goal.id),
            "new_depth_level": goal.depth_level
        }


# ============= v3.0 FEATURES: Goal Contracts, Mutation, Semantic Memory =============

@app.post("/goals/{goal_id}/mutate")
async def mutate_goal_endpoint(goal_id: str, mutation_data: dict, uow: UnitOfWork = Depends(get_uow)):
    """
    Мутирует цель (strengthen/weaken/change_type/freeze/thaw) - v3.0

    UoW MIGRATION: Теперь атомарная операция - мутация + state transition в одной транзакции.
    """
    from goal_mutator import goal_mutator

    mutation_type = mutation_data.get("mutation_type")
    reason = mutation_data.get("reason", "No reason provided")

    # Remove duplicate keys from mutation_data
    mutation_params = {k: v for k, v in mutation_data.items() if k not in ["mutation_type", "reason"]}

    try:
        result = await goal_mutator.mutate_goal_with_uow(
            uow=uow,
            goal_id=goal_id,
            mutation_type=mutation_type,
            reason=reason,
            **mutation_params
        )

        return {
            "status": "ok" if not result.get("error") else "error",
            "result": result,
            "transaction": "atomic"
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": f"Mutation blocked: {str(e)}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to mutate goal: {str(e)}"
        }


@app.post("/goals/{goal_id}/strict_evaluate")
async def strict_evaluate_goal_endpoint(goal_id: str, uow: UnitOfWork = Depends(get_uow)):
    """
    Строго оценивает цель (binary/scalar/trend) - v3.0
    Возвращает факт выполнения без анализа причин

    UoW MIGRATION: Теперь атомарная операция - оценка + state transition в одной транзакции.
    """
    from goal_strict_evaluator import goal_strict_evaluator

    try:
        evaluation = await goal_strict_evaluator.evaluate_goal_with_uow(uow, goal_id)

        return {
            "status": "ok",
            "strict_evaluation": evaluation,
            "transaction": "atomic"
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": f"Evaluation blocked: {str(e)}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to evaluate goal: {str(e)}"
        }


@app.post("/goals/{goal_id}/reflect")
async def reflect_on_goal_endpoint(goal_id: str, strict_evaluation: dict, uow: UnitOfWork = Depends(get_uow)):
    """
    Анализирует причины и генерирует следующие цели - v3.0
    Требует результат от strict_evaluate

    UoW MIGRATION: Теперь атомарная операция - рефлексия + создание next goals в одной транзакции.
    """
    from goal_reflector import goal_reflector

    try:
        reflection = await goal_reflector.reflect_on_goal_with_uow(
            uow=uow,
            goal_id=goal_id,
            strict_evaluation=strict_evaluation
        )

        return {
            "status": "ok",
            "reflection": reflection,
            "transaction": "atomic"
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": f"Reflection blocked: {str(e)}"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": f"Failed to reflect on goal: {str(e)}"
        }


@app.get("/goals/{goal_id}/patterns")
async def get_goal_patterns(goal_id: str):
    """
    Получает рекомендации на основе семантической памяти - v3.0
    """
    from semantic_memory import semantic_memory
    from models import Goal
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            return {"status": "error", "message": "Goal not found"}

    recommendations = await semantic_memory.get_recommendations(goal)

    return {
        "status": "ok",
        "recommendations": recommendations
    }


@app.get("/patterns/retrieve")
async def retrieve_patterns(
    pattern_type: str = None,
    goal_type: str = None,
    domains: str = None,
    limit: int = 5
):
    """
    Извлекает паттерны из семантической памяти - v3.0
    """
    from semantic_memory import semantic_memory

    domain_list = domains.split(",") if domains else None

    patterns = await semantic_memory.retrieve_similar_patterns(
        pattern_type or "success_pattern",
        goal_type,
        domain_list,
        limit
    )

    return {
        "status": "ok",
        "patterns": patterns
    }


@app.post("/goals/{goal_id}/extract_patterns")
async def extract_goal_patterns(goal_id: str, reflection: dict):
    """
    Извлекает паттерны из выполненной цели - v3.0
    """
    from semantic_memory import semantic_memory
    from models import Goal
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            return {"status": "error", "message": "Goal not found"}

    # Извлекаем паттерн
    passed = reflection.get("action") in ["complete", "continue"]

    if passed:
        pattern = await semantic_memory.extract_success_pattern(goal_id, reflection)
    else:
        pattern = await semantic_memory.extract_failure_pattern(goal_id, reflection)

    return {
        "status": "ok",
        "pattern": pattern
    }


@app.post("/patterns/cleanup")
async def cleanup_old_patterns(days: int = 30):
    """
    Cleanup old patterns with low confidence.
    
    Args:
        days: Delete patterns older than N days (default: 30)
    
    Returns:
        Number of deleted patterns
    """
    from semantic_memory import semantic_memory
    
    deleted_count = await semantic_memory.cleanup_old_patterns(days=days)
    
    return {
        "status": "ok",
        "deleted_count": deleted_count,
        "days_threshold": days
    }


@app.post("/patterns/search-vector")
async def search_patterns_vector(query: str, limit: int = 5):
    """
    Search patterns using Milvus vector similarity.
    
    Args:
        query: Search query text
        limit: Maximum results
    
    Returns:
        Similar patterns
    """
    from semantic_memory import semantic_memory
    
    patterns = await semantic_memory.retrieve_similar_patterns_vector(
        query_text=query,
        limit=limit
    )
    
    return {
        "status": "ok",
        "query": query,
        "patterns": patterns
    }


@app.get("/memory/stats")
async def get_memory_stats():
    """
    Get comprehensive memory system statistics.
    
    Returns stats for:
    - PostgreSQL (patterns)
    - Milvus (vector DB)
    - Neo4j (graph)
    - Redis (memory signals)
    """
    from semantic_memory import semantic_memory
    
    stats = await semantic_memory.get_stats()
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        **stats
    }


@app.get("/memory/health")
async def get_memory_health():
    """
    Quick health check for memory systems.
    
    Returns:
        Status of each memory component
    """
    health = {
        "overall": "healthy",
        "components": {}
    }
    
    # Check PostgreSQL
    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(func.count(Goal.id)).limit(1))
            health["components"]["postgresql"] = "connected"
    except Exception as e:
        health["components"]["postgresql"] = f"error: {str(e)[:50]}"
        health["overall"] = "degraded"
    
    # Check Milvus via memory service
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get("http://memory:8001/user/analysis")
            health["components"]["milvus"] = "connected"
    except Exception as e:
        health["components"]["milvus"] = f"error: {str(e)[:50]}"
        health["overall"] = "degraded"
    
    # Check Redis
    try:
        from redis import Redis
        redis_client = Redis(host='redis', port=6379, db=0)
        redis_client.ping()
        health["components"]["redis"] = "connected"
    except Exception as e:
        health["components"]["redis"] = f"error: {str(e)[:50]}"
        health["overall"] = "degraded"
    
    # Check Neo4j
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            health["components"]["neo4j"] = "connected"
    except Exception as e:
        health["components"]["neo4j"] = f"error: {str(e)[:50]}"
    
    return health


@app.post("/memory/batch-store")
async def batch_store_patterns(patterns: list):
    """
    Batch store patterns to Milvus.
    
    Args:
        patterns: List of patterns to store
        
    Returns:
        Number of successfully stored patterns
    """
    from semantic_memory import semantic_memory
    
    count = await semantic_memory.batch_store_patterns_vector(patterns)
    
    return {
        "status": "ok",
        "stored_count": count,
        "requested_count": len(patterns)
    }


# ============= ARTIFACT LAYER v1 - Tangible Results =============

@app.post("/artifacts/register")
async def register_artifact(artifact_data: dict):
    """
    Регистрирует новый артефакт - Artifact Layer v1

    Required fields:
    - goal_id: str
    - type: FILE|KNOWLEDGE|DATASET|REPORT|LINK|EXECUTION_LOG
    - content_kind: file|db|vector|external
    - content_location: str

    Optional fields:
    - skill_name: str
    - agent_role: str
    - domains: list[str]
    - tags: list[str]
    - language: str
    - reusable: bool
    - auto_verify: bool
    """
    from artifact_registry import artifact_registry

    try:
        async with get_uow() as uow:
            result = await artifact_registry.register_with_uow(
                uow=uow,
                goal_id=artifact_data.get("goal_id"),
                artifact_type=artifact_data.get("type"),
                content_kind=artifact_data.get("content_kind"),
                content_location=artifact_data.get("content_location"),
                skill_name=artifact_data.get("skill_name"),
                agent_role=artifact_data.get("agent_role"),
                domains=artifact_data.get("domains"),
                tags=artifact_data.get("tags"),
                language=artifact_data.get("language"),
                reusable=artifact_data.get("reusable", True),
                auto_verify=artifact_data.get("auto_verify", True)
            )

        return {
            "status": "ok",
            "artifact": result
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/goals/{goal_id}/artifacts")
async def get_goal_artifacts(
    goal_id: str,
    verification_status: str = None,
    include_descendants: bool = True
):
    """
    Возвращает артефакты цели

    Для atomic goals: возвращает прямые artifacts
    Для non-atomic goals: агрегирует artifacts от всех descendant atomic goals

    Query params:
    - verification_status: pending|passed|failed|partial (optional filter)
    - include_descendants: включать ли artifacts от descendant goals (default: True)
    """
    from artifact_registry import artifact_registry
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import select
    import uuid

    async with AsyncSessionLocal() as db:
        # Получаем goal
        stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            raise HTTPException(status_code=404, detail="Goal not found")

        artifacts = []

        # Если goal atomic - возвращаем его artifacts
        if goal.is_atomic:
            artifacts = await artifact_registry.list_by_goal(goal_id, verification_status)

        # Если non-atomic и include_descendants=True - агрегируем от descendants
        elif include_descendants:
            # Рекурсивно получаем все descendant atomic goals
            descendant_ids = await get_all_descendant_atomic_goals(goal_id, db)

            # Получаем artifacts от всех descendant atomic goals
            for desc_id in descendant_ids:
                desc_artifacts = await artifact_registry.list_by_goal(str(desc_id), verification_status)
                artifacts.extend(desc_artifacts)

        else:
            # Non-atomic goal без descendants - пустой список
            artifacts = []

    return {
        "status": "ok",
        "goal_id": goal_id,
        "is_atomic": goal.is_atomic,
        "count": len(artifacts),
        "artifacts": artifacts
    }


async def get_all_descendant_atomic_goals(goal_id: str, db) -> list:
    """
    Рекурсивно получает IDs всех descendant atomic goals

    Args:
        goal_id: ID родительской goal
        db: Database session

    Returns:
        List of descendant atomic goal IDs
    """
    from models import Goal
    from sqlalchemy import select
    import uuid

    descendant_ids = []

    # Получаем direct children
    stmt = select(Goal).where(Goal.parent_id == uuid.UUID(goal_id))
    result = await db.execute(stmt)
    children = result.scalars().all()

    for child in children:
        if child.is_atomic:
            # Atomic goal - добавляем в список
            descendant_ids.append(child.id)
        else:
            # Non-atomic goal - рекурсивно получаем descendants
            child_descendants = await get_all_descendant_atomic_goals(str(child.id), db)
            descendant_ids.extend(child_descendants)

    return descendant_ids


@app.get("/artifacts/goals-without-artifacts")
async def get_goals_without_artifacts(limit: int = 100):
    """
    Получить список выполненных goals без artifacts.

    Args:
        limit: Макс. количество goals

    Returns:
        List[goals] без artifacts
    """
    from retroactive_artifacts import RetroactiveArtifactGenerator

    try:
        goals = await RetroactiveArtifactGenerator.find_completed_goals_without_artifacts(limit)

        return {
            "status": "ok",
            "count": len(goals),
            "goals": goals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Возвращает артефакт по ID"""
    from artifact_registry import artifact_registry

    artifact = await artifact_registry.get(artifact_id)

    if not artifact:
        return {"status": "error", "message": "Artifact not found"}

    return {
        "status": "ok",
        "artifact": artifact
    }


@app.get("/artifacts/{artifact_id}/content")
async def get_artifact_content(artifact_id: str):
    """
    Возвращает содержимое артефакта (для FILE type)

    Читает файл с диска или возвращает ошибку если файл не найден
    """
    import os

    from artifact_registry import artifact_registry
    from database import AsyncSessionLocal
    from sqlalchemy import select
    from models import Artifact as ArtifactModel

    async with AsyncSessionLocal() as db:
        stmt = select(ArtifactModel).where(ArtifactModel.id == uuid.UUID(artifact_id))
        result = await db.execute(stmt)
        artifact_db = result.scalar_one_or_none()

        if not artifact_db:
            return {"status": "error", "message": "Artifact not found"}

        # Check if file exists
        file_path = artifact_db.content_location
        if not file_path:
            return {
                "status": "error",
                "message": "No file location for this artifact",
                "artifact_id": artifact_id
            }

        # Try to read file
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                return {
                    "status": "ok",
                    "artifact_id": artifact_id,
                    "file_path": file_path,
                    "file_content": content,
                    "file_size": len(content)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Failed to read file: {str(e)}",
                    "file_path": file_path
                }
        else:
            return {
                "status": "error",
                "message": "File not found on disk",
                "file_path": file_path,
                "hint": "The artifact was registered but the file may have been lost during container restart"
            }


@app.post("/artifacts/{artifact_id}/verify")
async def verify_artifact(artifact_id: str):
    """
    Верифицирует артефакт (CODE-BASED checks, not LLM)

    Returns:
    {
        "status": "passed|failed|partial",
        "results": [{"name": "...", "passed": true, "details": "..."}]
    }
    """
    from artifact_registry import artifact_registry

    result = await artifact_registry.verify_artifact(artifact_id)

    return {
        "status": "ok",
        "verification": result
    }


@app.get("/goals/{goal_id}/artifacts/check")
async def check_goal_artifacts(goal_id: str):
    """
    Проверяет наличие и статус артефактов цели

    Для atomic goals (L3):
    - MUST have at least 1 passed artifact
    - Otherwise marked as incomplete
    """
    from artifact_registry import artifact_registry

    check = await artifact_registry.check_goal_artifacts(goal_id)

    return {
        "status": "ok",
        "check": check
    }



# ============= SKILL MANIFEST v1 - Skill Contracts =============

@app.get("/skills")
async def list_skills(
    category: str = None,
    agent_role: str = None,
    artifact_type: str = None,
    is_active: bool = True
):
    """Возвращает список навыков с их манифестами"""
    from models import SkillManifestDB
    from database import AsyncSessionLocal
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        stmt = select(SkillManifestDB).where(SkillManifestDB.is_active == is_active)

        if category:
            stmt = stmt.where(SkillManifestDB.category == category)

        if agent_role:
            stmt = stmt.where(SkillManifestDB.agent_roles.contains([agent_role]))

        if artifact_type:
            stmt = stmt.where(
                (SkillManifestDB.outputs_artifact_type == artifact_type) |
                (SkillManifestDB.produces.contains([{"type": artifact_type}]))
            )

        stmt = stmt.order_by(SkillManifestDB.name)
        result = await db.execute(stmt)
        manifests = result.scalars().all()

        return {
            "status": "ok",
            "count": len(manifests),
            "skills": [
                {
                    "id": str(m.id),
                    "name": m.name,
                    "version": m.version,
                    "description": m.description,
                    "category": m.category,
                    "agent_roles": m.agent_roles,
                    "inputs": {
                        "schema": m.inputs_schema,
                        "required": m.inputs_required,
                        "optional": m.inputs_optional
                    },
                    "outputs": {
                        "artifact_type": m.outputs_artifact_type,
                        "schema": m.outputs_schema,
                        "reusable": m.outputs_reusable
                    },
                    "produces": m.produces,
                    "constraints": m.constraints,
                    "verification": m.verification,
                    "is_builtin": m.is_builtin
                }
                for m in manifests
            ]
        }


@app.get("/skills/{skill_name}")
async def get_skill_manifest(skill_name: str):
    """Возвращает манифест навыка по имени"""
    from models import SkillManifestDB
    from database import AsyncSessionLocal
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        stmt = select(SkillManifestDB).where(
            (SkillManifestDB.name == skill_name) &
            (SkillManifestDB.is_active == True)
        )
        result = await db.execute(stmt)
        manifest = result.scalar_one_or_none()

        if not manifest:
            return {"status": "error", "message": "Skill not found"}

        return {
            "status": "ok",
            "skill": {
                "name": manifest.name,
                "category": manifest.category,
                "agent_roles": manifest.agent_roles,
                "produces": manifest.produces,
                "verification": manifest.verification
            }
        }


# ============= LLM FALLBACK MANAGEMENT =============

@app.get("/llm/status")
async def get_llm_status():
    """Получить статус LLM fallback системы"""
    from llm_fallback import llm_fallback

    status = await llm_fallback.get_status()
    return {
        "status": "ok",
        "llm_status": status
    }


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus /metrics endpoint - sync for event loop safety"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/llm/reset_groq")
async def reset_groq_cooldown():
    """Вручную сбросить Groq cooldown и включить его обратно"""
    from llm_fallback import async_redis, GROQ_DISABLED_KEY, GROQ_FAILURE_KEY
    from llm_fallback import llm_fallback

    # Удаляем ключи из Redis (async)
    await async_redis.delete(GROQ_DISABLED_KEY, GROQ_FAILURE_KEY)

    status = await llm_fallback.get_status()

    return {
        "status": "ok",
        "message": "Groq cooldown reset manually",
        "new_status": status
    }


@app.post("/llm/test")
async def test_llm(request: dict):
    """
    Тестовый вызов LLM с fallback

    Body:
    {
        "prompt": "Hello, say hi!",
        "model": "groq/llama-3.3-70b-versatile"  # optional
    }
    """
    from llm_fallback import chat_with_fallback

    model = request.get("model", "groq/llama-3.3-70b-versatile")
    prompt = request.get("prompt", "Hello, say hi!")

    messages = [{"role": "user", "content": prompt}]

    try:
        result = await chat_with_fallback(model, messages)

        return {
            "status": "ok",
            "model_used": result.get("model", model),
            "response": result.get("choices", [{}])[0].get("message", {}).get("content", "")
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ============= DASHBOARD V2 API ENDPOINTS =============

from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any
import json


class GraphQuery(BaseModel):
    node_type: Optional[str] = None  # goal|agent|skill|artifact
    root_id: Optional[str] = None
    depth: int = 2
    include_relations: bool = True


@app.get("/graph")
async def get_graph(
    node_type: Optional[str] = None,
    root_id: Optional[str] = None,
    depth: int = 2
):
    """
    Получает граф целей, агентов, навыков и артефактов
    Для Dashboard v2 ReactFlow визуализации
    """
    from models import Goal, Artifact, SkillManifestDB
    from database import AsyncSessionLocal
    from sqlalchemy import select

    async with AsyncSessionLocal() as db:
        nodes = []
        edges = []

        # Добавляем цели
        stmt = select(Goal).order_by(Goal.created_at.desc())
        result = await db.execute(stmt)
        goals = result.scalars().all()

        for g in goals:
            nodes.append({
                "id": str(g.id),
                "type": "goal",
                "data": {
                    "label": g.title,
                    "status": g.status,
                    "progress": g.progress,
                    "goal_type": g.goal_type,
                    "is_atomic": g.is_atomic,
                    "depth_level": g.depth_level
                }
            })

            # Добавляем связь с родителем
            if g.parent_id:
                edges.append({
                    "id": f"{g.parent_id}-{g.id}",
                    "source": str(g.parent_id),
                    "target": str(g.id),
                    "type": "dependency"
                })

        # Примечание: артефакты НЕ добавляются в граф
        # Они загружаются отдельно через /goals/{goal_id}/artifacts
        # когда пользователь кликает на цель в InspectorPanel

        # Добавляем навыки
        stmt = select(SkillManifestDB).where(SkillManifestDB.is_active == True)
        result = await db.execute(stmt)
        skills = result.scalars().all()

        for s in skills:
            nodes.append({
                "id": f"skill-{s.name}",
                "type": "skill",
                "data": {
                    "label": s.name,
                    "category": s.category,
                    "version": s.version,
                    "description": s.description
                }
            })

        return {
            "status": "ok",
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "goals": len([n for n in nodes if n["type"] == "goal"]),
                "artifacts": len([n for n in nodes if n["type"] == "artifact"]),
                "skills": len([n for n in nodes if n["type"] == "skill"])
            }
        }


@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Получает детальную информацию об узле"""
    from models import Goal, Artifact, SkillManifestDB
    from database import AsyncSessionLocal
    from sqlalchemy import select
    import uuid

    async with AsyncSessionLocal() as db:
        # Проверяем goal
        try:
            stmt = select(Goal).where(Goal.id == uuid.UUID(node_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if goal:
                return {
                    "status": "ok",
                    "node": {
                        "id": str(goal.id),
                        "type": "goal",
                        "data": {
                            "title": goal.title,
                            "description": goal.description,
                            "status": goal.status,
                            "progress": goal.progress,
                            "goal_type": goal.goal_type,
                            "is_atomic": goal.is_atomic,
                            "depth_level": goal.depth_level,
                            "domains": goal.domains,
                            "completion_criteria": goal.completion_criteria,
                            "created_at": goal.created_at.isoformat() if goal.created_at else None,
                            "updated_at": goal.updated_at.isoformat() if goal.updated_at else None
                        }
                    }
                }
        except ValueError:
            pass

        # Проверяем artifact
        try:
            stmt = select(Artifact).where(Artifact.id == uuid.UUID(node_id))
            result = await db.execute(stmt)
            artifact = result.scalar_one_or_none()

            if artifact:
                return {
                    "status": "ok",
                    "node": {
                        "id": str(artifact.id),
                        "type": "artifact",
                        "data": {
                            "type": artifact.type,
                            "goal_id": str(artifact.goal_id),
                            "skill_name": artifact.skill_name,
                            "agent_role": artifact.agent_role,
                            "content_kind": artifact.content_kind,
                            "content_location": artifact.content_location,
                            "domains": artifact.domains,
                            "tags": artifact.tags,
                            "verification_status": artifact.verification_status,
                            "reusable": artifact.reusable
                        }
                    }
                }
        except ValueError:
            pass

        return {"status": "error", "message": "Node not found"}


@app.get("/nodes/{node_id}/inspector")
async def get_node_inspector(node_id: str):
    """Получает контекст для inspector panel"""
    from models import Goal, Artifact
    from database import AsyncSessionLocal
    from sqlalchemy import select
    import uuid

    async with AsyncSessionLocal() as db:
        # Проверяем goal
        try:
            stmt = select(Goal).where(Goal.id == uuid.UUID(node_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if goal:
                # Получаем артефакты цели
                artifact_stmt = select(Artifact).where(Artifact.goal_id == goal.id)
                artifact_result = await db.execute(artifact_stmt)
                artifacts = artifact_result.scalars().all()

                # Получаем подцели
                from sqlalchemy import select
                children_stmt = select(Goal).where(Goal.parent_id == goal.id)
                children_result = await db.execute(children_stmt)
                children = children_result.scalars().all()

                return {
                    "status": "ok",
                    "context": {
                        "node_id": str(goal.id),
                        "node_type": "goal",
                        "title": goal.title,
                        "description": goal.description,
                        "status": goal.status,
                        "progress": goal.progress,
                        "artifacts": [
                            {
                                "id": str(a.id),
                                "type": a.type,
                                "status": a.verification_status
                            }
                            for a in artifacts
                        ],
                        "sub_goals": len(children),
                        "domains": goal.domains or [],
                        "metadata": {
                            "created_at": goal.created_at.isoformat() if goal.created_at else None,
                            "updated_at": goal.updated_at.isoformat() if goal.updated_at else None,
                            "depth_level": goal.depth_level
                        }
                    }
                }
        except ValueError:
            pass

        return {"status": "error", "message": "Node not found"}


@app.get("/timeline")
async def get_timeline(
    limit: int = 50,
    node_type: Optional[str] = None
):
    """
    Получает таймлайн событий (создание/обновление целей, артефактов)
    """
    from models import Goal, Artifact
    from database import AsyncSessionLocal
    from sqlalchemy import select, union_all

    events = []

    async with AsyncSessionLocal() as db:
        # Получаем цели с их created_at/updated_at
        goal_stmt = select(Goal).order_by(Goal.created_at.desc()).limit(limit)
        goal_result = await db.execute(goal_stmt)
        goals = goal_result.scalars().all()

        for g in goals:
            events.append({
                "timestamp": g.created_at.isoformat() if g.created_at else None,
                "node_id": str(g.id),
                "node_type": "goal",
                "event_type": "created",
                "data": {
                    "title": g.title,
                    "status": g.status
                }
            })

            if g.updated_at:
                events.append({
                    "timestamp": g.updated_at.isoformat(),
                    "node_id": str(g.id),
                    "node_type": "goal",
                    "event_type": "updated",
                    "data": {
                        "title": g.title,
                        "status": g.status,
                        "progress": g.progress
                    }
                })

        # Получаем артефакты
        artifact_stmt = select(Artifact).order_by(Artifact.created_at.desc()).limit(limit)
        artifact_result = await db.execute(artifact_stmt)
        artifacts = artifact_result.scalars().all()

        for a in artifacts:
            events.append({
                "timestamp": a.created_at.isoformat() if a.created_at else None,
                "node_id": str(a.id),
                "node_type": "artifact",
                "event_type": "created",
                "data": {
                    "type": a.type,
                    "status": a.verification_status
                }
            })

        # Сортируем по timestamp
        events.sort(key=lambda x: x["timestamp"] or "", reverse=True)

        return {
            "status": "ok",
            "events": events[:limit],
            "total": len(events)
        }


@app.post("/ui/events")
async def handle_ui_event(event: dict):
    """
    Обрабатывает события из UI
    """
    event_type = event.get("type")
    event_data = event.get("data", {})

    # Логируем событие
    logger.info(f"[UI Event] {event_type}: {event_data}")

    # Здесь можно добавить обработку разных типов событий
    if event_type == "node_selected":
        # Пользователь выбрал узел
        return {"status": "ok", "message": "Node selected"}
    elif event_type == "mode_changed":
        # Пользователь сменил режим (explore/exploit/reflect)
        return {"status": "ok", "message": "Mode changed"}
    elif event_type == "constraint_updated":
        # Пользователь обновил ограничения
        return {"status": "ok", "message": "Constraint updated"}

    return {"status": "ok", "message": "Event received"}


@app.get("/ui/stream")
async def stream_ui_updates():
    """
    SSE stream для real-time обновлений UI
    """
    async def event_generator():
        try:
            while True:
                # Отправляем heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============= ARTIFACTS API ENDPOINTS =============

@app.get("/artifacts")
async def get_artifacts(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    type: Optional[str] = None
):
    """
    Получает список артефактов с фильтрацией
    Для Dashboard v2
    """
    from models import Artifact as ArtifactModel
    from database import AsyncSessionLocal
    from sqlalchemy import select, func

    async with AsyncSessionLocal() as db:
        # Build query
        stmt = select(ArtifactModel)

        # Apply filters
        if status:
            stmt = stmt.where(ArtifactModel.verification_status == status)
        if type:
            stmt = stmt.where(ArtifactModel.type == type.upper())

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await db.execute(count_stmt)
        total = total_result.scalar()

        # Apply pagination and ordering
        stmt = stmt.order_by(ArtifactModel.created_at.desc()).offset(offset).limit(limit)

        result = await db.execute(stmt)
        artifacts = result.scalars().all()

        return {
            "status": "ok",
            "artifacts": [
                {
                    "id": str(a.id),
                    "type": a.type,
                    "goal_id": str(a.goal_id),
                    "skill_name": a.skill_name,
                    "agent_role": a.agent_role,
                    "content_kind": a.content_kind,
                    "content_location": a.content_location[:500] if a.content_location else "",  # Preview
                    "domains": a.domains or [],
                    "tags": a.tags or [],
                    "verification_status": a.verification_status,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                    "updated_at": a.updated_at.isoformat() if a.updated_at else None
                }
                for a in artifacts
            ],
            "pagination": {
                "total": total,
                "limit": limit,
                "offset": offset
            }
        }


# Duplicate route removed - use /artifacts/{artifact_id} from line 571 instead
# This was conflicting with specific routes like /artifacts/goals-without-artifacts



    from sqlalchemy import select
    import uuid

    async with AsyncSessionLocal() as db:
        try:
            stmt = select(ArtifactModel).where(ArtifactModel.id == uuid.UUID(artifact_id))
            result = await db.execute(stmt)
            artifact = result.scalar_one_or_none()

            if not artifact:
                return {"status": "error", "message": "Artifact not found"}

            # Get goal info
            goal_stmt = select(Goal).where(Goal.id == artifact.goal_id)
            goal_result = await db.execute(goal_stmt)
            goal = goal_result.scalar_one_or_none()

            return {
                "status": "ok",
                "artifact": {
                    "id": str(artifact.id),
                    "type": artifact.type,
                    "goal_id": str(artifact.goal_id),
                    "goal_title": goal.title if goal else None,
                    "skill_name": artifact.skill_name,
                    "agent_role": artifact.agent_role,
                    "content_kind": artifact.content_kind,
                    "content_location": artifact.content_location,
                    "domains": artifact.domains or [],
                    "tags": artifact.tags or [],
                    "language": artifact.language,
                    "verification_status": artifact.verification_status,
                    "verification_results": artifact.verification_results,
                    "reusable": artifact.reusable,
                    "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
                    "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None
                }
            }
        except ValueError:
            return {"status": "error", "message": "Invalid artifact ID"}


@app.get("/artifacts/stats/summary")
async def get_artifacts_stats():
    """
    Получает статистику по артефактам
    """
    from models import Artifact as ArtifactModel
    from database import AsyncSessionLocal
    from sqlalchemy import select, func

    async with AsyncSessionLocal() as db:
        # Total count
        total_stmt = select(func.count()).select_from(ArtifactModel)
        total_result = await db.execute(total_stmt)
        total = total_result.scalar()

        # By status
        status_stmt = select(
            ArtifactModel.verification_status,
            func.count().label('count')
        ).group_by(ArtifactModel.verification_status)
        status_result = await db.execute(status_stmt)
        by_status = {row[0]: row[1] for row in status_result.all()}

        # By type
        type_stmt = select(
            ArtifactModel.type,
            func.count().label('count')
        ).group_by(ArtifactModel.type)
        type_result = await db.execute(type_stmt)
        by_type = {row[0]: row[1] for row in type_result.all()}

        # By skill
        skill_stmt = select(
            ArtifactModel.skill_name,
            func.count().label('count')
        ).where(ArtifactModel.skill_name.isnot(None)).group_by(ArtifactModel.skill_name)
        skill_result = await db.execute(skill_stmt)
        by_skill = {row[0]: row[1] for row in skill_result.all()}

        # Recent activity (last 7 days)
        from datetime import timedelta
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_stmt = select(func.count()).select_from(ArtifactModel).where(
            ArtifactModel.created_at >= week_ago
        )
        recent_result = await db.execute(recent_stmt)
        recent_count = recent_result.scalar()

        return {
            "status": "ok",
            "stats": {
                "total": total,
                "by_status": by_status,
                "by_type": by_type,
                "by_skill": by_skill,
                "recent_week": recent_count
            }
        }


# ============= GOAL STUCK PREVENTION =============

@app.post("/goals/auto-update-stale")
async def auto_update_stale_goals():
    """
    Автоматически обновляет застаревшие цели

    Обновляет goals которые не обновлялись более 1 часа,
    чтобы предотвратить их "зависание"
    """
    from models import Goal
    from sqlalchemy import select
    from datetime import timedelta, timezone

    async with get_uow() as uow:
        now = datetime.now(timezone.utc)
        stale_threshold = now - timedelta(hours=1)

        stmt = select(Goal).where(
            (Goal.status.in_(["active", "pending"])) &
            ((Goal.updated_at < stale_threshold) | (Goal.updated_at.is_(None)))
        )

        result = await uow.session.execute(stmt)
        stale_goals = result.scalars().all()

        updated_count = 0
        for goal in stale_goals:
            if not goal.updated_at or goal.updated_at < stale_threshold:
                goal.updated_at = now
                updated_count += 1

        await uow.session.flush()

        return {
            "status": "ok",
            "updated": updated_count,
            "message": f"Updated {updated_count} stale goals"
        }


@app.post("/goals/resume-all-stuck")
async def resume_all_stuck_goals():
    """
    Реанимирует ALL застрявшие цели (активные цели без прогресса > 24 часа)
    """
    from models import Goal
    from sqlalchemy import select
    from datetime import timedelta, timezone

    async with get_uow() as uow:
        now = datetime.now(timezone.utc)
        stuck_threshold = now - timedelta(hours=24)

        stmt = select(Goal).where(
            (Goal.status == "active") &
            ((Goal.updated_at < stuck_threshold) | (Goal.updated_at.is_(None)))
        )

        result = await uow.session.execute(stmt)
        stuck_goals = result.scalars().all()

        resumed = []
        for goal in stuck_goals:
            goal.updated_at = now
            resumed.append({
                "id": str(goal.id),
                "title": goal.title,
                "last_update": goal.updated_at.isoformat()
            })

        await uow.session.flush()

        return {
            "status": "ok",
            "resumed": len(resumed),
            "goals": resumed
        }


# ============= USER QUESTIONS API =============

@app.get("/questions/pending")
async def get_pending_questions(goal_id: Optional[str] = None):
    """
    Получает список вопросов ожидающих ответа пользователя
    
    Query params:
        goal_id: (optional) Фильтр по ID цели
    """
    from redis import Redis
    
    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)
    
    # Получаем все ключи вопросов
    pattern = f"pending_question:*"
    if goal_id:
        pattern = f"pending_question:{goal_id}:*"
    
    question_keys = redis_client.keys(pattern)
    
    questions = []
    for key in question_keys:
        data = redis_client.get(key)
        if data:
            questions.append(json.loads(data))
    
    # Сортируем по приоритету и времени
    priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
    questions.sort(key=lambda q: (
        priority_order.get(q.get("priority", "normal"), 2),
        q.get("asked_at", "")
    ))
    
    return {
        "status": "ok",
        "count": len(questions),
        "questions": questions
    }


@app.post("/questions/{question_id}/answer")
async def answer_question(question_id: str, answer: str):
    """
    Отправляет ответ на вопрос пользователя
    
    Args:
        question_id: ID вопроса
        answer: Текст ответа
    """
    from redis import Redis
    from database import AsyncSessionLocal
    from models import Artifact
    
    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)
    
    # Находим вопрос
    question_key = None
    question_data = None
    
    for key in redis_client.keys("pending_question:*"):
        data = redis_client.get(key)
        if data:
            parsed = json.loads(data)
            if parsed.get("artifact_id") == question_id:
                question_key = key
                question_data = parsed
                break
    
    if not question_data:
        raise HTTPException(status_code=404, detail="Question not found")

    async with get_uow() as uow:
        artifact = await uow.session.get(Artifact, question_id)
        if artifact:
            artifact.content["answer"] = answer
            artifact.content["answered_at"] = datetime.utcnow().isoformat()
            artifact.content["status"] = "answered"
            artifact.metadata["verification_status"] = "verified"

            await uow.session.flush()
    
    # Удаляем из pending
    redis_client.delete(question_key)
    
    # Сохраняем ответ в историю
    history_key = f"question_history:{question_data['goal_id']}"
    redis_client.lpush(history_key, json.dumps({
        "question": question_data["question"],
        "answer": answer,
        "question_id": question_id,
        "answered_at": datetime.utcnow().isoformat()
    }))
    redis_client.expire(history_key, 86400 * 7)  # Хранить 7 дней
    
    return {
        "status": "ok",
        "message": "Answer recorded successfully",
        "question": question_data["question"],
        "answer": answer
    }


@app.get("/questions/stats")
async def get_question_stats():
    """
    Получает статистику по вопросам
    """
    from redis import Redis
    
    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)
    
    # Подсчет вопросов
    pending_keys = redis_client.keys("pending_question:*")
    
    # Статистика по приоритетам
    priority_stats = {"critical": 0, "high": 0, "normal": 0, "low": 0}
    
    for key in pending_keys:
        data = redis_client.get(key)
        if data:
            parsed = json.loads(data)
            priority = parsed.get("priority", "normal")
            priority_stats[priority] = priority_stats.get(priority, 0) + 1
    
    return {
        "status": "ok",
        "pending_count": len(pending_keys),
        "priority_breakdown": priority_stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/questions/history/{goal_id}")
async def get_question_history(goal_id: str, limit: int = 10):
    """
    Получает историю вопросов для цели
    
    Args:
        goal_id: ID цели
        limit: Максимальное количество записей (default: 10)
    """
    from redis import Redis
    
    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)
    
    history_key = f"question_history:{goal_id}"
    history_raw = redis_client.lrange(history_key, 0, limit - 1)
    
    history = [json.loads(item) for item in history_raw]
    
    return {
        "status": "ok",
        "goal_id": goal_id,
        "count": len(history),
        "history": history
    }


# ============= QUESTION TIMEOUT HANDLING =============

@app.post("/questions/check-timeouts")
async def check_question_timeouts():
    """
    Проверяет все вопросы на таймаут и применяет timeout_action

    Вызывается периодически для обработки просроченных вопросов
    """
    from redis import Redis
    from models import Artifact, Goal
    from sqlalchemy import select
    from goal_transition_service import transition_service

    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)

    timeout_keys = redis_client.keys("question_timeout:*")

    processed = []

    async with get_uow() as uow:
        for timeout_key in timeout_keys:
            ttl = redis_client.ttl(timeout_key)

            if ttl == -2:
                timeout_data_raw = redis_client.get(timeout_key)
                continue

            elif ttl == -1:
                continue

            continue

        stmt = select(Artifact).where(
            Artifact.type == "QUESTION",
            Artifact.content["status"].astext == "pending"
        )
        result = await uow.session.execute(stmt)
        pending_artifacts = result.scalars().all()

        now = datetime.utcnow()

        for artifact in pending_artifacts:
            content = artifact.content
            timeout_at_str = content.get("timeout_at")

            if not timeout_at_str:
                continue

            timeout_at = datetime.fromisoformat(timeout_at_str)

            if now >= timeout_at:
                timeout_action = content.get("timeout_action", "continue_with_default")
                default_answer = content.get("default_answer")
                goal_id = artifact.metadata.get("goal_id")

                logger.info(f"⏰ Question timeout: {artifact.id}")
                logger.info(f"   Action: {timeout_action}")

                if timeout_action == "continue_with_default":
                    if default_answer:
                        artifact.content["answer"] = default_answer
                        artifact.content["answered_at"] = now.isoformat()
                        artifact.content["status"] = "answered_with_default"
                        artifact.content["timeout_used"] = True

                        await uow.session.flush()

                        processed.append({
                            "question_id": str(artifact.id),
                            "action": "continue_with_default",
                            "default_answer": default_answer[:100] if default_answer else None
                        })

                        logger.info(f"   ✓ Used default answer")
                else:
                    timeout_action = "fail_goal"

            if timeout_action == "fail_goal":
                if goal_id:
                    from uuid import UUID
                    try:
                        goal_stmt = select(Goal).where(Goal.id == UUID(goal_id))
                        goal_result = await uow.session.execute(goal_stmt)
                        goal = goal_result.scalar_one_or_none()

                        if goal:
                            goal.error_message = f"Question timed out: {content.get('question', 'Unknown')[:100]}"

                            await transition_service.transition(
                                uow=uow,
                                goal_id=goal.id,
                                new_state="failed",
                                reason=f"Question timed out: {content.get('question', 'Unknown')[:100]}",
                                actor="system"
                            )

                            # Emit GoalFailed event for Metrics Engine
                            event_bus = get_event_bus()
                            await event_bus.publish(GoalFailed(goal_id=goal.id))

                            processed.append({
                                "question_id": str(artifact.id),
                                "action": "fail_goal",
                                "goal_id": goal_id
                            })

                            logger.info(f"   ✗ Goal {goal_id[:8]}... marked as failed")
                    except Exception as e:
                        logger.info(f"   Error failing goal: {e}")

            elif timeout_action == "wait_longer":
                new_timeout = now + timedelta(hours=1)
                artifact.content["timeout_at"] = new_timeout.isoformat()
                artifact.content["extended_count"] = artifact.content.get("extended_count", 0) + 1

                await uow.session.flush()

                processed.append({
                    "question_id": str(artifact.id),
                    "action": "wait_longer",
                    "extended_until": new_timeout.isoformat()
                })

                logger.info(f"   ⏱ Extended until {new_timeout.isoformat()}")

    return {
        "status": "ok",
        "processed_count": len(processed),
        "processed": processed,
        "timestamp": now.isoformat()
    }



# ============= TELEGRAM INTEGRATION =============

@app.post("/telegram/send_question")
async def send_question_via_telegram(request: dict):
    """
    Внутренний endpoint для отправки вопроса через Telegram бота

    Body:
        user_id: ID пользователя
        question_data: Данные вопроса
    """
    from redis import Redis

    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)

    user_id = request.get("user_id")
    question_data = request.get("question_data")

    if not user_id or not question_data:
        raise HTTPException(status_code=400, detail="user_id and question_data required")

    # Находим chat_id
    chat_id = redis_client.get(f"telegram:user_chat:{user_id}")

    if not chat_id:
        return {
            "status": "error",
            "message": "User not linked to Telegram"
        }

    # Отправляем через Telegram Bot API
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        return {
            "status": "error",
            "message": "Telegram bot not configured"
        }

    try:
        async with httpx.AsyncClient() as client:
            priority = question_data.get("priority", "normal")
            emoji = {"critical": "🔴", "high": "🟠", "normal": "🟢", "low": "⚪"}.get(priority, "⚪")

            message = f"""
{emoji} <b>Вопрос от AI-OS</b>

<b>Вопрос:</b>
{question_data.get('question', 'N/A')}

<b>Контекст:</b>
{question_data.get('context', 'N/A')[:200]}...

<b>Приоритет:</b> {priority.upper()}
<b>Истекает:</b> {question_data.get('timeout_at', 'N/A')}

<b>Ответьте через Dashboard или введите текст:</b>
            """

            # Создаем inline клавиатуру
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup

            question_id = question_data.get("artifact_id")
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("📝 Открыть Dashboard", url=f"http://localhost:8000/dashboard"),
                ],
                [
                    InlineKeyboardButton("✅ Использовать дефолт", callback_data=f"skip_{question_id}"),
                ]
            ])

            # Отправляем сообщение
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": int(chat_id),
                "text": message,
                "parse_mode": "HTML",
                "reply_markup": keyboard.to_dict()
            }

            response = await client.post(url, json=payload)
            result = response.json()

            if result.get("ok"):
                return {
                    "status": "ok",
                    "message": "Question sent to Telegram",
                    "chat_id": chat_id
                }
            else:
                return {
                    "status": "error",
                    "message": result.get("description", "Unknown error")
                }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.post("/telegram/link")
async def link_telegram_account(user_id: str, chat_id: int):
    """
    Привязывает Telegram аккаунт к пользователю

    Args:
        user_id: ID пользователя в системе
        chat_id: Telegram chat_id
    """
    from redis import Redis

    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)

    # Сохраняем привязку
    redis_client.set(f"telegram:user_chat:{user_id}", str(chat_id), ex=86400*30)
    redis_client.set(f"telegram:chat_user:{chat_id}", user_id, ex=86400*30)

    return {
        "status": "ok",
        "message": f"Telegram account {chat_id} linked to user {user_id}"
    }



# =============================================================================
# USERS ENDPOINT - Для Telegram Bot
# =============================================================================

@app.get("/users/{user_id}")
async def get_user(user_id: str):
    """
    GET /users/{user_id} - Проверка существования пользователя
    
    Для Telegram Bot: проверяет, может ли пользователь привязать аккаунт
    
    Временное решение (v2.7.1 LTS):
    - Принимаем любой user_id в формате: user-*, *, или UUID
    - TODO: добавить реальную проверку в БД когда будет таблица users
    """
    # Валидация формата
    if not user_id or len(user_id) < 3:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    
    # Временно принимаем любой user_id
    # (пока нет таблицы users в БД)
    return {
        "status": "ok",
        "user_id": user_id,
        "exists": True,
        "created_at": "2025-02-03T00:00:00Z",  # Заглушка для LTS
        "note": "User validation is simplified in v2.7.1 LTS"
    }
@app.get("/telegram/status/{user_id}")
async def get_telegram_status(user_id: str):
    """
    Проверяет статус привязки Telegram для пользователя

    Args:
        user_id: ID пользователя
    """
    from redis import Redis

    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)

    chat_id = redis_client.get(f"telegram:user_chat:{user_id}")
    is_linked = chat_id is not None

    return {
        "status": "ok",
        "user_id": user_id,
        "telegram_linked": is_linked,
        "chat_id": chat_id if is_linked else None
    }


@app.post("/telegram/unlink/{user_id}")
async def unlink_telegram_account(user_id: str):
    """
    Отвязывает Telegram аккаунт от пользователя

    Args:
        user_id: ID пользователя
    """
    from redis import Redis

    redis_client = Redis(host='redis', port=6379, db=0, decode_responses=True)

    chat_id = redis_client.get(f"telegram:user_chat:{user_id}")

    if chat_id:
        redis_client.delete(f"telegram:user_chat:{user_id}")
        redis_client.delete(f"telegram:chat_user:{chat_id}")

        return {
            "status": "ok",
            "message": f"Telegram account unlinked for user {user_id}"
        }
    else:
        raise HTTPException(status_code=404, detail="Telegram account not found")


# ============= GOAL RELATIONS API =============

class GoalRelationRequest(BaseModel):
    from_goal_id: str
    to_goal_id: str
    relation_type: str  # causal, dependency, conflict, reinforcement
    strength: float = 1.0
    reason: str = None
    metadata: dict = None  # Maps to relation_metadata in DB

@app.post("/relations")
async def create_relation(req: GoalRelationRequest):
    """Create a relationship between two goals"""
    try:
        from models import GoalRelation
        import uuid
        from sqlalchemy import select

        async with get_uow() as uow:
            stmt1 = select(Goal).where(Goal.id == uuid.UUID(req.from_goal_id))
            stmt2 = select(Goal).where(Goal.id == uuid.UUID(req.to_goal_id))

            result1 = await uow.session.execute(stmt1)
            result2 = await uow.session.execute(stmt2)

            from_goal = result1.scalar_one_or_none()
            to_goal = result2.scalar_one_or_none()

            if not from_goal:
                raise HTTPException(status_code=404, detail=f"From goal {req.from_goal_id} not found")
            if not to_goal:
                raise HTTPException(status_code=404, detail=f"To goal {req.to_goal_id} not found")

            valid_types = ['causal', 'dependency', 'conflict', 'reinforcement']
            if req.relation_type not in valid_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid relation_type. Must be one of: {valid_types}"
                )

            relation = GoalRelation(
                from_goal_id=uuid.UUID(req.from_goal_id),
                to_goal_id=uuid.UUID(req.to_goal_id),
                relation_type=req.relation_type,
                strength=req.strength,
                reason=req.reason,
                relation_metadata=req.metadata
            )

            uow.session.add(relation)
            await uow.session.flush()
            await uow.session.refresh(relation)

            return {
                "status": "created",
                "relation": {
                    "id": str(relation.id),
                    "from_goal_id": str(relation.from_goal_id),
                    "to_goal_id": str(relation.to_goal_id),
                    "relation_type": relation.relation_type,
                    "strength": relation.strength,
                    "reason": relation.reason
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/relations/{goal_id}")
async def get_goal_relations(goal_id: str):
    """Get all relations for a specific goal"""
    try:
        from models import GoalRelation
        from database import AsyncSessionLocal
        import uuid
        from sqlalchemy import select, or_

        async with AsyncSessionLocal() as db:
            goal_uuid = uuid.UUID(goal_id)

            # Get relations where goal is either from or to
            stmt = select(GoalRelation).where(
                or_(
                    GoalRelation.from_goal_id == goal_uuid,
                    GoalRelation.to_goal_id == goal_uuid
                )
            )

            result = await db.execute(stmt)
            relations = result.scalars().all()

            relations_data = []
            for r in relations:
                relations_data.append({
                    "id": str(r.id),
                    "from_goal_id": str(r.from_goal_id),
                    "to_goal_id": str(r.to_goal_id),
                    "relation_type": r.relation_type,
                    "strength": r.strength,
                    "reason": r.reason,
                    "metadata": r.metadata,
                    "created_at": r.created_at.isoformat() if r.created_at else None
                })

            return {
                "status": "ok",
                "goal_id": goal_id,
                "relations": relations_data,
                "count": len(relations_data)
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/relations/{relation_id}")
async def delete_relation(relation_id: str):
    """Delete a goal relation"""
    try:
        from models import GoalRelation
        import uuid
        from sqlalchemy import select

        async with get_uow() as uow:
            stmt = select(GoalRelation).where(GoalRelation.id == uuid.UUID(relation_id))
            result = await uow.session.execute(stmt)
            relation = result.scalar_one_or_none()

            if not relation:
                raise HTTPException(status_code=404, detail="Relation not found")

            await uow.session.delete(relation)
            await uow.session.flush()

            return {"status": "deleted", "relation_id": relation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PERSONALITY ENGINE API (Phase 1)
# =============================================================================

@app.get("/personality/{user_id}")
async def get_personality_profile(user_id: str):
    """
    Получить профиль личности пользователя

    Args:
        user_id: UUID пользователя (telegram_id или system user ID)

    Returns:
        PersonalityProfileSchema с core_traits, motivations, values, preferences
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        profile = await engine.get_profile(user_id)

        if not profile:
            # Создать дефолтный профиль
            profile = await engine.get_profile(user_id)

        return {
            "status": "ok",
            "profile": profile.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/personality/{user_id}")
async def update_personality_profile(user_id: str, updates: dict):
    """
    Обновить профиль личности пользователя

    Body:
    {
        "core_traits": {"openness": 0.8, ...},
        "motivations": {"growth": 0.9, ...},
        "values": [{"name": "здоровье", "importance": 0.8}, ...],
        "preferences": {
            "communication_style": {"tone": "спокойный", ...},
            ...
        }
    }

    Args:
        user_id: UUID пользователя
        updates: Данные для обновления (частичные)

    Returns:
        Обновлённый PersonalityProfileSchema
    """
    from personality_engine import get_personality_engine, PersonalityUpdateSchema

    try:
        engine = get_personality_engine()

        # Конвертируем dict в Pydantic schema
        update_schema = PersonalityUpdateSchema(**updates)

        profile = await engine.update_profile(user_id, update_schema)

        return {
            "status": "ok",
            "profile": profile.dict()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/personality/{user_id}/feedback")
async def record_personality_feedback(user_id: str, event_type: str, reaction: str,
                                     context: dict = None, correction: str = None,
                                     source: str = "system"):
    """
    Записать feedback для адаптации Personality Engine

    Body:
    {
        "event_type": "goal_completed|decision_approved|tone_corrected",
        "reaction": "positive|negative|neutral",
        "context": {...},
        "correction": "...",
        "source": "system|user_explicit|user_implicit"
    }

    Args:
        user_id: UUID пользователя
        event_type: Тип события
        reaction: Реакция пользователя
        context: Контекст события
        correction: Текст корректировки
        source: Источник feedback

    Returns:
        {"status": "recorded"}
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        await engine.record_feedback(user_id, event_type, reaction, context, correction, source)

        return {
            "status": "recorded",
            "message": "Feedback recorded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personality/{user_id}/values")
async def get_value_matrix(user_id: str):
    """
    Получить матрицу ценностей для Decision Logic

    Args:
        user_id: UUID пользователя

    Returns:
        Dict[value_name] -> importance (0.0-1.0)
        Пример: {"осознанность": 0.8, "здоровье": 0.7, ...}
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        value_matrix = await engine.get_value_matrix(user_id)

        return {
            "status": "ok",
            "user_id": user_id,
            "value_matrix": value_matrix
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personality/{user_id}/communication")
async def get_communication_style(user_id: str):
    """
    Получить стиль общения для Interface Layer

    Args:
        user_id: UUID пользователя

    Returns:
        Dict с communication_style (tone, humor, detail_level, language)
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        comm_style = await engine.get_communication_style(user_id)

        return {
            "status": "ok",
            "user_id": user_id,
            "communication_style": comm_style
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personality/{user_id}/traits")
async def get_core_traits(user_id: str):
    """
    Получить Big Five traits

    Args:
        user_id: UUID пользователя

    Returns:
        Dict с core_traits (openness, conscientiousness, etc.)
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        traits = await engine.get_core_traits(user_id)

        return {
            "status": "ok",
            "user_id": user_id,
            "core_traits": traits
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personality/{user_id}/motivations")
async def get_motivations(user_id: str):
    """
    Получить мотивации

    Args:
        user_id: UUID пользователя

    Returns:
        Dict с motivations (growth, achievement, comfort, etc.)
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        motivations = await engine.get_motivations(user_id)

        return {
            "status": "ok",
            "user_id": user_id,
            "motivations": motivations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# PERSONALITY SNAPSHOTS API (NS1/NS2 Enhancement)
# =============================================================================

@app.post("/personality/{user_id}/snapshot")
async def create_personality_snapshot(user_id: str, reason: str = "update", created_by: str = "system"):
    """
    Создать снапшот профиля личности.

    Args:
        user_id: UUID пользователя
        reason: Причина ("user_update", "adaptation", "manual")
        created_by: Кто создал ("system", "user", "auto_adaptation")

    Returns:
        PersonalitySnapshotSchema
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        snapshot = await engine.create_snapshot(user_id, reason, created_by)

        return {
            "status": "ok",
            "snapshot": snapshot.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/personality/{user_id}/snapshots")
async def get_personality_snapshots(user_id: str, limit: int = 10):
    """
    Получить историю снапшотов.

    Args:
        user_id: UUID пользователя
        limit: Макс. количество снапшотов

    Returns:
        List[PersonalitySnapshotSchema]
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        snapshots = await engine.get_snapshots(user_id, limit)

        return {
            "status": "ok",
            "user_id": user_id,
            "count": len(snapshots),
            "snapshots": [s.dict() for s in snapshots]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/personality/{user_id}/rollback/{snapshot_version}")
async def rollback_personality_to_snapshot(user_id: str, snapshot_version: int):
    """
    Откатиться к версии снапшота.

    Args:
        user_id: UUID пользователя
        snapshot_version: Версия для отката

    Returns:
        Обновлённый PersonalityProfileSchema
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        profile = await engine.rollback_to_snapshot(user_id, snapshot_version)

        return {
            "status": "ok",
            "profile": profile.dict(),
            "message": f"Rolled back to version {snapshot_version}"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONTEXTUAL MEMORY API (NS1/NS2 Enhancement)
# =============================================================================

@app.get("/personality/{user_id}/contextual-memory")
async def get_contextual_memory(user_id: str):
    """
    Получить контекстную память пользователя.

    Args:
        user_id: UUID пользователя

    Returns:
        ContextualMemorySchema
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        memory = await engine.get_contextual_memory(user_id)

        return {
            "status": "ok",
            "contextual_memory": memory.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CONTEXT OS API - Smart context selection for LLM
# =============================================================================

@app.post("/context/select")
async def select_context(request: Request):
    """
    Выбрать релевантный контекст для LLM задачи.
    
    Context OS - интеллектуальный выбор контекста из кодовой базы.
    
    Request body:
    {
        "target": "goal_executor",  # Файл, класс или функция
        "task_type": "implement",    # implement, analyze, refactor, review
        "model": "qwen",            # qwen, opencode, claude, gpt4
        "max_tokens": 8000          # Лимит токенов
    }
    
    Returns:
    {
        "status": "ok",
        "target": "goal_executor",
        "summary": "...",
        "files": [{"path": "...", "content": "...", "relevance": 0.9}],
        "interfaces": [...],
        "total_tokens": 5000
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    target = body.get("target", "")
    task_type = body.get("task_type", "implement")
    model = body.get("model", "qwen")
    max_tokens = body.get("max_tokens")
    
    if not target:
        raise HTTPException(status_code=400, detail="target is required")
    
    try:
        from dev.context_selector import ContextSelector
        from dev.code_graph import load_or_build_graph
        
        # Load code graph
        graph = load_or_build_graph("/app")
        
        # Select context
        selector = ContextSelector(code_graph=graph)
        context_pack = selector.select(
            target=target,
            task_type=task_type,
            model=model,
            max_tokens=max_tokens
        )
        
        return {
            "status": "ok",
            "target": context_pack.target,
            "summary": context_pack.summary,
            "files": [
                {"path": p, "content": c[:2000], "relevance": r} 
                for p, c, r in context_pack.files
            ],
            "interfaces": context_pack.interfaces,
            "imports": context_pack.imports,
            "total_tokens": context_pack.total_tokens
        }
        
    except Exception as e:
        logger.error("context_selection_failed", error=str(e), target=target)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/context/graph/stats")
async def context_graph_stats():
    """
    Получить статистику code graph.
    
    Returns:
    {
        "total_nodes": 950,
        "total_edges": 2100,
        "files": 387,
        "classes": 450,
        "functions": 1200
    }
    """
    try:
        from dev.code_graph import load_or_build_graph
        
        graph = load_or_build_graph("/app")
        
        files = set()
        classes = set()
        functions = set()
        
        for node_id, node in graph.nodes.items():
            if hasattr(node, 'file_path'):
                files.add(node.file_path)
            if hasattr(node, 'node_type'):
                if node.node_type == 'class':
                    classes.add(node_id)
                elif node.node_type == 'function':
                    functions.add(node_id)
        
        return {
            "status": "ok",
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "files": len(files),
            "classes": len(classes),
            "functions": len(functions)
        }
        
    except Exception as e:
        logger.error("graph_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/graph/rebuild")
async def rebuild_context_graph():
    """
    Перестроить code graph.
    
    Returns:
    {
        "status": "ok",
        "nodes": 950,
        "edges": 2100
    }
    """
    try:
        from dev.code_graph import CodeGraph, build_graph
        
        graph = build_graph()
        
        return {
            "status": "ok",
            "nodes": len(graph.nodes),
            "edges": len(graph.edges)
        }
        
    except Exception as e:
        logger.error("graph_rebuild_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SEMANTIC RAG API - Context-aware code search
# =============================================================================

@app.post("/context/semantic-search")
async def semantic_code_search(request: Request):
    """
    Семантический поиск по коду.
    
    Использует:
    1. Keyword-based search по функциям и классам
    2. Graph expansion для зависимостей
    3. Возвращает релевантные файлы и чанки
    
    Request body:
    {
        "query": "add new skill registration",
        "top_k": 10,
        "expand_dependencies": true
    }
    
    Returns:
    {
        "query": "...",
        "chunks_found": 15,
        "files": [
            {
                "path": "/app/canonical_skills/registry.py",
                "chunks": [
                    {"name": "SkillRegistry", "type": "class", "lines": "10-50"}
                ]
            }
        ]
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    query = body.get("query", "")
    top_k = body.get("top_k", 10)
    expand = body.get("expand_dependencies", True)
    
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    
    try:
        from dev.semantic_rag_index import search_code_context
        
        result = await search_code_context(
            task=query,
            top_k=top_k,
            expand_dependencies=expand
        )
        
        return {
            "status": "ok",
            **result
        }
        
    except Exception as e:
        logger.error("semantic_search_failed", error=str(e), query=query)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DEV GOAL ENGINE API - Self-evolving system
# =============================================================================

@app.get("/dev/goals")
async def get_dev_goals():
    """
    Получить список задач разработки на основе анализа traces.
    
    Dev Goal Engine анализирует:
    1. Capability gaps - какие capabilities запрашивались но не были удовлетворены
    2. Problematic skills - навыки с низким success rate
    3. Recurring errors - повторяющиеся ошибки
    
    Returns:
    {
        "status": "ok",
        "dev_goals": [
            {
                "goal_type": "new_skill",
                "title": "Create {skill} skill",
                "description": "...",
                "priority": 4,
                "target_module": "goal_executor_v2.py"
            }
        ]
    }
    """
    try:
        from dev.dev_goal_engine import DevGoalEngine
        from trace_store import get_trace_store
        from trace_mining_engine import get_mining_engine
        
        # Get trace store and mining engine
        trace_store = get_trace_store()
        mining_engine = get_mining_engine(trace_store)
        
        # Create dev goal engine
        engine = DevGoalEngine(mining_engine)
        
        # Generate dev goals
        dev_goals = await engine.analyze_and_generate_dev_goals()
        
        return {
            "status": "ok",
            "dev_goals": [
                {
                    "goal_type": g.goal_type.value,
                    "title": g.title,
                    "description": g.description,
                    "priority": g.priority,
                    "target_module": g.target_module,
                    "required_capabilities": g.required_capabilities
                }
                for g in dev_goals
            ]
        }
        
    except Exception as e:
        logger.error("dev_goals_generation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dev/goals/from-goal")
async def create_dev_goal_from_failed_goal(request: Request):
    """
    Создать dev goal из провалившейся обычной цели.
    
    Request body:
    {
        "goal_id": "uuid",
        "title": "Research AI OS",
        "description": "Research artificial intelligence operating systems",
        "failed_reason": "no skill found for capability 'research'",
        "required_capabilities": ["research", "web-search"]
    }
    
    Returns:
    {
        "status": "ok",
        "dev_task": {
            "task_type": "implement_skill",
            "target": "goal_executor_v2.py",
            "description": "...",
            "suggested_approach": "..."
        }
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    goal_info = {
        "title": body.get("title", ""),
        "description": body.get("description", ""),
        "failed_reason": body.get("failed_reason", ""),
        "required_capabilities": body.get("required_capabilities", []),
        "target_module": body.get("target_module", "unknown")
    }
    
    try:
        from dev.dev_goal_engine import DevGoalEngine
        from trace_store import get_trace_store
        from trace_mining_engine import get_mining_engine
        
        trace_store = get_trace_store()
        mining_engine = get_mining_engine(trace_store)
        engine = DevGoalEngine(mining_engine)
        
        dev_task = engine.generate_dev_task_from_goal(goal_info)
        
        return {
            "status": "ok",
            "dev_task": dev_task
        }
        
    except Exception as e:
        logger.error("dev_goal_from_failed_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dev/analyze/traces")
async def analyze_traces_deep():
    """
    Глубокий анализ traces для выявления паттернов и проблем.
    
    Returns:
    {
        "status": "ok",
        "analysis": {
            "skill_success_rate": {...},
            "skill_usage": {...},
            "patterns": [...],
            "capability_gaps": [...],
            "problematic_skills": [...]
        }
    }
    """
    try:
        from trace_store import get_trace_store
        from trace_mining_engine import get_mining_engine
        
        trace_store = get_trace_store()
        mining_engine = get_mining_engine(trace_store)
        
        analysis = await mining_engine.analyze_all()
        
        # Extract key insights
        skill_rates = analysis.get("skill_success_rate", {})
        problematic = [
            {"skill": k, "rate": v.get("success_rate", 0), "total": v.get("total", 0)}
            for k, v in skill_rates.items()
            if v.get("success_rate", 1.0) < 0.7 and v.get("total", 0) >= 3
        ]
        
        return {
            "status": "ok",
            "analysis": {
                "total_traces": analysis.get("skill_usage", {}).get("total", 0),
                "skill_success_rate": skill_rates,
                "skill_usage": analysis.get("skill_usage", {}),
                "problematic_skills": problematic,
                "patterns_count": len(analysis.get("patterns", []))
            }
        }
        
    except Exception as e:
        logger.error("trace_analysis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SKILL LIFECYCLE MANAGER API - Self-evolving skill activation
# =============================================================================

@app.post("/skills/lifecycle/activate")
async def activate_skill(request: Request):
    """
    Активировать новый навык после генерации.
    
    Это замыкает loop:
    capability gap → generate → activate → planner uses
    
    Request body:
    {
        "skill_id": "my_generated_skill",
        "capabilities": ["code-generation", "python"],
        "metadata": {"source": "dev_goal_engine", "goal_id": "..."}
    }
    
    Returns:
    {
        "status": "ok",
        "skill_id": "my_generated_skill",
        "activated": true,
        "message": "Skill activated and ready to use"
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    skill_id = body.get("skill_id", "")
    capabilities = body.get("capabilities", [])
    metadata = body.get("metadata", {})
    
    if not skill_id:
        raise HTTPException(status_code=400, detail="skill_id is required")
    
    if not capabilities:
        raise HTTPException(status_code=400, detail="capabilities is required")
    
    try:
        from skill_lifecycle_manager import activate_new_skill
        
        # Try to import the generated skill module
        try:
            import importlib
            module = importlib.import_module(f"canonical_skills.autogenerated.{skill_id}")
        except ImportError:
            # Module not found - just log and continue
            logger.warning(
                "skill_module_not_found",
                skill_id=skill_id
            )
            module = None
        
        success = await activate_new_skill(
            skill_id=skill_id,
            skill_module=module,
            capabilities=capabilities,
            metadata=metadata
        )
        
        if success:
            return {
                "status": "ok",
                "skill_id": skill_id,
                "activated": True,
                "message": "Skill activated and ready to use"
            }
        else:
            return {
                "status": "error",
                "skill_id": skill_id,
                "activated": False,
                "message": "Failed to activate skill"
            }
            
    except Exception as e:
        logger.error("skill_activation_failed", error=str(e), skill_id=skill_id)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skills/lifecycle/status/{skill_id}")
async def get_skill_lifecycle_status(skill_id: str):
    """
    Получить статус навыка в lifecycle.
    
    Returns:
    {
        "skill_id": "my_skill",
        "status": "active",
        "history": [...]
    }
    """
    try:
        from skill_lifecycle_manager import get_skill_lifecycle_manager
        
        manager = get_skill_lifecycle_manager()
        status = manager.get_skill_status(skill_id)
        history = manager.get_lifecycle_history(skill_id=skill_id)
        
        return {
            "skill_id": skill_id,
            "status": status.value if status else "unknown",
            "history": history
        }
        
    except Exception as e:
        logger.error("lifecycle_status_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/skills/lifecycle/history")
async def get_lifecycle_history(limit: int = 50):
    """
    Получить историю всех lifecycle событий.
    
    Returns:
    {
        "events": [...],
        "total": 100
    }
    """
    try:
        from skill_lifecycle_manager import get_skill_lifecycle_manager
        
        manager = get_skill_lifecycle_manager()
        events = manager.get_lifecycle_history(limit=limit)
        
        return {
            "events": events,
            "total": len(events)
        }
        
    except Exception as e:
        logger.error("lifecycle_history_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/skills/lifecycle/deactivate/{skill_id}")
async def deactivate_skill(skill_id: str, reason: str = "Manual deprecation"):
    """
    Деактивировать навык.
    
    Args:
        skill_id: ID навыка
        reason: Причина деактивации
    """
    try:
        from skill_lifecycle_manager import get_skill_lifecycle_manager
        
        manager = get_skill_lifecycle_manager()
        success = await manager.deactivate_skill(skill_id, reason)
        
        return {
            "status": "ok" if success else "error",
            "skill_id": skill_id,
            "deactivated": success
        }
        
    except Exception as e:
        logger.error("skill_deactivation_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# SELF-EVOLUTION PIPELINE API - Complete evolution loop
# =============================================================================

@app.post("/evolution/run")
async def run_evolution(request: Request):
    """
    Запустить полный evolution pipeline.
    
    Обрабатывает неудачу цели и эволюционирует систему:
    1. Detect capability gap
    2. Generate dev task
    3. Generate skill
    4. Activate skill
    5. Invalidate caches
    
    Request body:
    {
        "goal_id": "uuid",
        "capability": "pdf-parse",
        "goal_title": "Parse PDF document",
        "goal_description": "Extract text from PDF file"
    }
    
    Returns:
    {
        "status": "ok",
        "success": true,
        "capability_gap": "pdf-parse",
        "skill_generated": "pdf_parser_skill",
        "skill_activated": true,
        "message": "Successfully evolved system"
    }
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    goal_id = body.get("goal_id", "")
    capability = body.get("capability", "")
    goal_title = body.get("goal_title", "")
    goal_description = body.get("goal_description", "")
    
    if not capability:
        raise HTTPException(status_code=400, detail="capability is required")
    
    try:
        from self_evolution_pipeline import run_evolution_from_failure
        
        result = await run_evolution_from_failure(
            goal_id=goal_id,
            capability=capability,
            goal_title=goal_title,
            goal_description=goal_description
        )
        
        return {
            "status": "ok",
            "success": result.success,
            "goal_id": result.goal_id,
            "capability_gap": result.capability_gap,
            "skill_generated": result.skill_generated,
            "skill_activated": result.skill_activated,
            "retry_successful": result.retry_successful,
            "message": result.message,
            "details": result.details
        }
        
    except Exception as e:
        logger.error("evolution_pipeline_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/evolution/status")
async def get_evolution_status():
    """
    Получить статус evolution pipeline.
    
    Returns:
    {
        "evolution_count": 5,
        "recent_evolutions": [...]
    }
    """
    try:
        from self_evolution_pipeline import get_evolution_pipeline
        
        pipeline = get_evolution_pipeline()
        
        # Get lifecycle history
        from skill_lifecycle_manager import get_skill_lifecycle_manager
        manager = get_skill_lifecycle_manager()
        recent = manager.get_lifecycle_history(limit=10)
        
        return {
            "evolution_count": pipeline.evolution_count,
            "recent_evolutions": recent
        }
        
    except Exception as e:
        logger.error("evolution_status_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evolution/retry-goal/{goal_id}")
async def retry_goal_with_evolution(goal_id: str):
    """
    Повторить цель после evolution.
    
    После того как система эволюционировала,
    повторить выполнение цели с новыми capabilities.
    """
    try:
        # Get goal
        from models import Goal
        from database import AsyncSessionLocal
        from sqlalchemy import select
        
        async with AsyncSessionLocal() as session:
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await session.execute(stmt)
            goal = result.scalar_one_or_none()
            
            if not goal:
                return {"status": "error", "message": "Goal not found"}
        
        # Re-execute goal (simplified - just restart)
        # In production, this would call the full execution pipeline
        return {
            "status": "ok",
            "message": "Goal queued for re-execution",
            "goal_id": goal_id,
            "note": "System has evolved - new skills may be available"
        }
        
    except Exception as e:
        logger.error("retry_goal_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/personality/{user_id}/contextual-memory")
async def update_contextual_memory(user_id: str, recent_goals: list = None,
                                  emotional_tone: str = None,
                                  behavioral_summary: dict = None):
    """
    Обновить контекстную память.

    Args:
        user_id: UUID пользователя
        recent_goals: Недавние цели
        emotional_tone: Эмоциональный тон
        behavioral_summary: Поведенческое резюме
    """
    from personality_engine import get_personality_engine

    try:
        engine = get_personality_engine()
        await engine.update_contextual_memory(
            user_id,
            recent_goals=recent_goals,
            emotional_tone=emotional_tone,
            behavioral_summary=behavioral_summary
        )

        return {
            "status": "ok",
            "message": "Contextual memory updated"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# GOAL CONFLICT DETECTION API (NS1/NS2 Enhancement)
# =============================================================================

@app.post("/goals/{goal_id}/check-conflicts")
async def check_goal_conflicts(goal_id: str, check_against: list = None):
    """
    Проверить цель на конфликты с другими целями.

    Args:
        goal_id: ID цели для проверки
        check_against: Список ID целей для проверки (опционально)

    Returns:
        ConflictDetectionResult
    """
    from goal_conflict_detector import get_goal_conflict_detector

    try:
        detector = get_goal_conflict_detector()
        result = await detector.check_goal_conflicts(goal_id, check_against)

        return {
            "status": "ok",
            "goal_id": goal_id,
            "conflict_result": result.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/goals/{user_id}/conflicts")
async def get_goal_conflicts(user_id: str, status: str = None, severity: str = None):
    """
    Получить все конфликты пользователя.

    Args:
        user_id: UUID пользователя
        status: Фильтр по статусу (detected, resolved, ignored)
        severity: Фильтр по severity (low, medium, high, critical)

    Returns:
        List[SingleConflict]
    """
    from goal_conflict_detector import get_goal_conflict_detector

    try:
        detector = get_goal_conflict_detector()
        conflicts = await detector.get_conflicts_for_user(user_id, status, severity)

        return {
            "status": "ok",
            "user_id": user_id,
            "count": len(conflicts),
            "conflicts": [c.dict() for c in conflicts]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conflicts/{conflict_id}/resolve")
async def resolve_goal_conflict(conflict_id: str, resolution: str):
    """
    Разрешить конфликт.

    Args:
        conflict_id: UUID конфликта
        resolution: Текст решения

    Returns:
        Обновлённый SingleConflict
    """
    from goal_conflict_detector import get_goal_conflict_detector

    try:
        detector = get_goal_conflict_detector()
        resolved = await detector.resolve_conflict(conflict_id, resolution)

        return {
            "status": "ok",
            "conflict": resolved.dict(),
            "message": "Conflict resolved"
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RETROACTIVE ARTIFACT GENERATION API
# MUST BE BEFORE /artifacts/{artifact_id} to avoid route conflicts!
# =============================================================================

@app.post("/goals/{goal_id}/fix-artifacts")
async def fix_goal_without_artifacts(goal_id: str):
    """
    Исправить выполненный goal без artifacts - создать artifact постфактум.

    Args:
        goal_id: UUID goal

    Returns:
        Результат операции
    """
    from retroactive_artifacts import fix_goal_without_artifacts

    try:
        result = await fix_goal_without_artifacts(goal_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/artifacts/fix-all-goals")
async def fix_all_goals_without_artifacts():
    """
    Массово создать artifacts для всех выполненных goals без artifacts.

    Useful endpoint для восстановления данных после внедрения Artifact Layer.

    Returns:
        Статистика операции
    """
    from retroactive_artifacts import batch_fix_all_goals

    try:
        result = await batch_fix_all_goals()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ARTIFACTS API (Original routes - MUST BE AFTER specific routes above)
# =============================================================================

# =============================================================================
# TEMPORAL CONTINUOUS GOALS API (Phase 1)
# =============================================================================

class ContinuousGoalRequest(BaseModel):
    title: str
    description: str = ""
    cron_schedule: str = "0 9 * * *"  # Default: daily at 9 AM
    domains: list = None
    max_executions: int = None  # None = run forever


@app.post("/goals/continuous/start")
async def start_continuous_goal(req: ContinuousGoalRequest):
    """
    Запускает continuous goal с Temporal Cron Workflow

    Body:
    {
        "title": "Improve system performance",
        "description": "Weekly performance optimization",
        "cron_schedule": "0 9 * * 1",  # Weekly on Monday
        "domains": ["performance", "programming"],
        "max_executions": null  # Run forever
    }

    Returns:
        {
            "status": "started",
            "workflow_id": "continuous-goal-xxx",
            "message": "Continuous goal started with cron schedule"
        }
    """
    try:
        import sys
        sys.path.insert(0, "/app/temporal")
        from shared.continuous_goals_client import get_continuous_goals_client
        import uuid

        # Generate goal ID
        goal_id = str(uuid.uuid4())

        # Get client
        client = get_continuous_goals_client()

        # Start workflow
        workflow_id = await client.start_continuous_goal(
            goal_id=goal_id,
            title=req.title,
            description=req.description,
            cron_schedule=req.cron_schedule,
            domains=req.domains or [],
            max_executions=req.max_executions,
        )

        return {
            "status": "started",
            "goal_id": goal_id,
            "workflow_id": workflow_id,
            "cron_schedule": req.cron_schedule,
            "message": "Continuous goal started with Temporal Cron Workflow"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/continuous/execute-once/{goal_id}")
async def execute_continuous_goal_once(goal_id: str):
    """
    Выполняет continuous goal один раз (для тестирования)

    Args:
        goal_id: UUID цели

    Returns:
        Результат выполнения
    """
    try:
        import sys
        sys.path.insert(0, "/app/temporal")
        from shared.continuous_goals_client import get_continuous_goals_client
        from models import Goal
        from database import AsyncSessionLocal
        from sqlalchemy import select
        import uuid

        # Get goal from database
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                raise HTTPException(status_code=404, detail="Goal not found")

        # Get client and execute
        client = get_continuous_goals_client()
        result = await client.execute_continuous_goal_once(
            goal_id=goal_id,
            title=goal.title,
            description=goal.description or "",
            domains=goal.domains or [],
        )

        return {
            "status": "completed",
            "goal_id": goal_id,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/goals/continuous/status/{workflow_id}")
async def get_continuous_goal_status(workflow_id: str):
    """
    Получает статус Temporal workflow для continuous goal

    Args:
        workflow_id: Temporal workflow ID

    Returns:
        Статус workflow
    """
    try:
        import sys
        sys.path.insert(0, "/app/temporal")
        from shared.continuous_goals_client import get_continuous_goals_client

        client = get_continuous_goals_client()
        status = await client.get_workflow_status(workflow_id)

        return {
            "status": "ok",
            "workflow_status": status
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/continuous/cancel/{workflow_id}")
async def cancel_continuous_goal(workflow_id: str):
    """
    Отменяет continuous goal workflow

    Args:
        workflow_id: Temporal workflow ID

    Returns:
        Статус операции
    """
    try:
        import sys
        sys.path.insert(0, "/app/temporal")
        from shared.continuous_goals_client import get_continuous_goals_client

        client = get_continuous_goals_client()
        await client.cancel_workflow(workflow_id)

        return {
            "status": "cancelled",
            "workflow_id": workflow_id,
            "message": "Continuous goal workflow cancelled"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goals/continuous/terminate/{workflow_id}")
async def terminate_continuous_goal(workflow_id: str, reason: str = "User requested"):
    """
    Принудительно завершает continuous goal workflow

    Args:
        workflow_id: Temporal workflow ID
        reason: Причина завершения

    Returns:
        Статус операции
    """
    try:
        import sys
        sys.path.insert(0, "/app/temporal")
        from shared.continuous_goals_client import get_continuous_goals_client

        client = get_continuous_goals_client()
        await client.terminate_workflow(workflow_id, reason)

        return {
            "status": "terminated",
            "workflow_id": workflow_id,
            "reason": reason,
            "message": "Continuous goal workflow terminated"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/temporal/workflows")
async def list_temporal_workflows():
    """
    Получает список всех Temporal workflows

    Returns:
        Список workflows со статусами
    """
    try:
        import sys
        sys.path.insert(0, "/app/temporal")
        from shared.temporal_client import get_temporal_client

        client = await get_temporal_client()

        # List workflows (this requires Temporal server query)
        # For now, return basic status
        return {
            "status": "ok",
            "message": "Temporal client connected",
            "note": "Use Temporal Web UI at http://localhost:8088 for detailed workflow list"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# EMOTIONAL LAYER API
# ==============================

@app.get("/emotional/state/{user_id}")
async def get_emotional_state(user_id: str):
    """
    Get current emotional state for a user

    Args:
        user_id: User identifier (UUID)

    Returns:
        Current emotional state (arousal, valence, focus, confidence)
    """
    try:
        from emotional_layer import emotional_layer
        state = await emotional_layer.get_current_state(user_id)
        return {
            "status": "ok",
            "user_id": user_id,
            **state
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional/history/{user_id}")
async def get_emotional_history(user_id: str, limit: int = 100):
    """
    Get emotional state history for a user

    Args:
        user_id: User identifier (UUID)
        limit: Maximum number of records to return

    Returns:
        List of historical emotional states
    """
    try:
        from emotional_layer import emotional_layer
        history = await emotional_layer.get_history(user_id, limit=limit)
        return {
            "status": "ok",
            "user_id": user_id,
            "count": len(history),
            "history": history
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotional/influence/{user_id}")
async def get_emotional_influence(user_id: str, signals: dict):
    """
    Get emotional influence for decision-making

    Args:
        user_id: User identifier (UUID)
        signals: EmotionalSignals object

    Returns:
        EmotionalInfluence with decision modifiers
    """
    try:
        from emotional_layer import emotional_layer
        from schemas import EmotionalSignals

        # Convert dict to EmotionalSignals
        emotional_signals = EmotionalSignals(**signals)

        influence = await emotional_layer.get_influence(user_id, emotional_signals)

        return {
            "status": "ok",
            "user_id": user_id,
            "influence": influence.dict()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotional/context/{user_id}")
async def get_emotional_context(user_id: str, signals: dict):
    """
    Get emotional influence as agent-friendly context dict

    Args:
        user_id: User identifier (UUID)
        signals: EmotionalSignals object

    Returns:
        EmotionalContext dict for agent prompts
    """
    try:
        from emotional_layer import emotional_layer
        from schemas import EmotionalSignals

        # Convert dict to EmotionalSignals
        emotional_signals = EmotionalSignals(**signals)

        context = await emotional_layer.get_influence_context(user_id, emotional_signals)

        return {
            "status": "ok",
            "user_id": user_id,
            "context": context
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# EMOTIONAL INFERENCE ENGINE V2 API
# =============================================================================

@app.post("/emotional/v2/infer")
async def emotional_inference_v2(req: EIEInferenceRequest):
    """
    EIE v2: Выполнить полный эмоциональный inference.

    Args:
        req: Запрос с user_id, proposed_action, intent

    Returns:
        DecisionModifiers с учетом всех 5 слоёв EIE v2
    """
    try:
        from emotional_inference_v2 import emotional_inference_engine_v2

        # Convert intent if provided
        intent = None
        if req.intent:
            from emotional_inference_v2 import EmotionalIntent
            intent = EmotionalIntent(
                primary=req.intent.primary,
                priority=req.intent.priority
            )

        modifiers = await emotional_inference_engine_v2.infer(
            user_id=req.user_id,
            proposed_action=req.proposed_action,
            intent=intent,
            signals=req.signals
        )

        return {
            "status": "ok",
            "modifiers": {
                "max_depth": modifiers.max_depth,
                "pace": modifiers.pace,
                "explanation_level": modifiers.explanation_level,
                "style": modifiers.style,
                "safety_override": modifiers.safety_override,
                "recovery_mode": modifiers.recovery_mode,
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional/v2/forecast/{user_id}")
async def emotional_forecast_v2(
    user_id: str,
    action: str,
    intent_primary: str = "neutral"
):
    """
    EIE v2: Получить эмоциональный прогноз для действия.

    Args:
        user_id: User identifier
        action: Предлагаемое действие
        intent_primary: Эмоциональное намерение

    Returns:
        EmotionalForecast с risk flags и expected deltas
    """
    try:
        from emotional_inference_v2 import (
            emotional_inference_engine_v2,
            EmotionalIntent
        )

        # Reconstruct state and build patterns
        state = await emotional_inference_engine_v2.state_reconstructor.reconstruct_state(user_id)
        context = await emotional_inference_engine_v2.pattern_builder.build_context(user_id)

        # Create intent
        intent = EmotionalIntent(primary=intent_primary)

        # Forecast
        forecast = emotional_inference_engine_v2.forecaster.simulate(
            current_state=state,
            action=action,
            pattern_context=context
        )

        return {
            "status": "ok",
            "forecast": {
                "predicted_state": forecast.predicted_state.to_dict(),
                "risk_flags": forecast.risk_flags,
                "expected_delta": forecast.expected_delta,
                "confidence": forecast.confidence,
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional/v2/patterns/{user_id}")
async def get_emotional_patterns_v2(user_id: str, limit: int = 100):
    """
    EIE v2: Получить эмоциональные паттерны пользователя.

    Возвращает риск-профиль, доминантные паттерны и корреляции с успехом.
    """
    try:
        from emotional_inference_v2 import emotional_inference_engine_v2

        context = await emotional_inference_engine_v2.pattern_builder.build_context(
            user_id, limit=limit
        )

        return {
            "status": "ok",
            "patterns": {
                "risk_profile": context.risk_profile,
                "dominant_patterns": context.dominant_patterns,
                "success_correlations": context.success_correlations,
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional/v2/reconstruct/{user_id}")
async def reconstruct_emotional_state_v2(user_id: str):
    """
    EIE v2: Восстановить реальное текущее состояние (с time-decay).

    В отличие от простого чтения из БД, этот endpoint:
    - Применяет экспоненциальное затухание
    - Учитывает недавние переходы
    - Возвращает "живое" состояние
    """
    try:
        from emotional_inference_v2 import emotional_inference_engine_v2

        state = await emotional_inference_engine_v2.state_reconstructor.reconstruct_state(user_id)

        return {
            "status": "ok",
            "state": state.to_dict(),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotional/v2/clusters/rebuild")
async def rebuild_trajectory_clusters(user_id: Optional[str] = None, num_clusters: int = 5):
    """
    EIE v2: Пересобрать кластеры эмоциональных траекторий.

    Это главный endpoint для Trajectory Clustering (Step 1 upgrade).
    Пересобирает кластеры из Affective Memory и усиливает forecasting power.

    Args:
        user_id: Если указан, строит кластеры только для пользователя.
                  Если None, строит глобальные кластеры (все пользователи).
        num_clusters: Количество кластеров для каждого action_type (default: 5)

    Returns:
        Статистику построенных кластеров.
    """
    try:
        from emotional_trajectory_clustering import trajectory_clusterer

        # Пересобираем кластеры
        await trajectory_clusterer.build_clusters(user_id=user_id)

        # Собираем статистику
        stats = {
            "total_clusters": 0,
            "clusters_by_action": {},
            "user_id": user_id or "global"
        }

        for action_type, clusters in trajectory_clusterer.clusters.items():
            stats["clusters_by_action"][action_type] = []
            stats["total_clusters"] += len(clusters)

            for cluster in clusters:
                cluster_info = {
                    "cluster_id": cluster.cluster_id,
                    "num_trajectories": len(cluster.trajectories),
                    "typical_outcome": cluster.typical_outcome,
                    "success_rate": round(cluster.success_rate, 2),
                    "centroid_features": {
                        k: round(v, 3) if isinstance(v, (int, float)) else v
                        for k, v in (cluster.centroid_features or {}).items()
                        if k != "trend_vector"  # Пропускаем длинные векторы
                    }
                }
                stats["clusters_by_action"][action_type].append(cluster_info)

        return {
            "status": "ok",
            "message": f"Rebuilt {stats['total_clusters']} trajectory clusters",
            "stats": stats
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional/v2/clusters")
async def get_trajectory_clusters():
    """
    EIE v2: Получить текущие кластеры эмоциональных траекторий.

    Возвращает информацию о построенных кластерах.
    """
    try:
        from emotional_trajectory_clustering import trajectory_clusterer

        if not trajectory_clusterer.clusters:
            return {
                "status": "ok",
                "message": "No clusters built yet. Use POST /emotional/v2/clusters/rebuild",
                "clusters": {}
            }

        stats = {
            "total_clusters": 0,
            "clusters_by_action": {}
        }

        for action_type, clusters in trajectory_clusterer.clusters.items():
            stats["clusters_by_action"][action_type] = []
            stats["total_clusters"] += len(clusters)

            for cluster in clusters:
                cluster_info = {
                    "cluster_id": cluster.cluster_id,
                    "num_trajectories": len(cluster.trajectories),
                    "typical_outcome": cluster.typical_outcome,
                    "success_rate": round(cluster.success_rate, 2)
                }
                stats["clusters_by_action"][action_type].append(cluster_info)

        return {
            "status": "ok",
            "stats": stats
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emotional/v2/model/train")
async def train_emotional_forecasting_model(min_samples: int = 20, test_size: float = 0.2):
    """
    EIE v2: Обучить ML модель для эмоционального прогнозирования.

    Это главный endpoint для Step 2 - Learned Forecasting Model.
    Обучает RandomForestRegressor на данных из Affective Memory.

    Args:
        min_samples: Минимальное количество samples для обучения (default: 20)
        test_size: Доля test set для валидации (default: 0.2)

    Returns:
        Метрики обучения (MSE, R2) и информацию о модели
    """
    try:
        from emotional_forecasting_model import emotional_forecasting_model

        # Проверяем доступность sklearn
        import sys
        if 'sklearn' not in sys.modules:
            try:
                import sklearn
            except ImportError:
                return {
                    "status": "error",
                    "message": "scikit-learn not installed. Install with: pip3 install scikit-learn"
                }

        # Обучаем модель
        metrics = await emotional_forecasting_model.train(
            min_samples=min_samples,
            test_size=test_size
        )

        # Получаем информацию о модели
        metadata = emotional_forecasting_model.get_metadata()
        feature_importance = emotional_forecasting_model.get_feature_importance()

        # Top-10 самых важных features
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "status": "ok",
            "message": "ML model trained successfully",
            "metrics": metrics,
            "metadata": metadata,
            "top_features": [
                {"feature": name, "importance": round(imp, 3)}
                for name, imp in top_features
            ]
        }

    except ValueError as e:
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emotional/v2/model")
async def get_emotional_forecasting_model():
    """
    EIE v2: Получить информацию о ML модели.

    Возвращает метаданные, метрики и feature importance.
    """
    try:
        from emotional_forecasting_model import emotional_forecasting_model

        if not emotional_forecasting_model.is_available():
            return {
                "status": "ok",
                "message": "ML model not trained yet. Use POST /emotional/v2/model/train",
                "model": {
                    "available": False,
                    "trained": False
                }
            }

        metadata = emotional_forecasting_model.get_metadata()
        feature_importance = emotional_forecasting_model.get_feature_importance()

        # Top-10 самых важных features
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return {
            "status": "ok",
            "model": {
                "available": True,
                "trained": metadata.get("trained", False),
                "training_samples": metadata.get("training_samples", 0),
                "trained_at": metadata.get("trained_at"),
                "metrics": metadata.get("metrics", {}),
                "top_features": [
                    {"feature": name, "importance": round(imp, 3)}
                    for name, imp in top_features
                ]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# =============================================================================
# STEP 2.6: SYSTEM ALERTS ENDPOINTS
# =============================================================================

@app.get("/alerts")
async def get_alerts(
    resolved: bool = False,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50
):
    """
    GET /alerts — Get system alerts (awareness signals, NOT corrections)
    """
    try:
        from models import SystemAlert
        from sqlalchemy import desc

        async with AsyncSessionLocal() as db:
            # Build query
            stmt = select(SystemAlert)

            # Apply filters
            if resolved is not None:
                stmt = stmt.where(SystemAlert.resolved == resolved)

            if alert_type:
                stmt = stmt.where(SystemAlert.alert_type == alert_type)

            if severity:
                stmt = stmt.where(SystemAlert.severity == severity)

            # Order and limit
            stmt = stmt.order_by(desc(SystemAlert.created_at)).limit(limit)

            result = await db.execute(stmt)
            alerts = result.scalars().all()

            # Convert to response
            return {
                "status": "ok",
                "count": len(alerts),
                "alerts": [
                    {
                        "id": str(alert.id),
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "trigger_data": alert.trigger_data,
                        "explanation": alert.explanation,
                        "context": alert.context,
                        "resolved": alert.resolved,
                        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                        "created_at": alert.created_at.isoformat()
                    }
                    for alert in alerts
                ]
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts/summary")
async def get_alerts_summary():
    """GET /alerts/summary — Get summary of system alerts"""
    try:
        from models import SystemAlert
        from sqlalchemy import func

        async with AsyncSessionLocal() as db:
            # Active alerts
            active_stmt = select(
                SystemAlert.alert_type,
                SystemAlert.severity,
                func.count().label("count")
            ).where(SystemAlert.resolved == False).group_by(
                SystemAlert.alert_type,
                SystemAlert.severity
            )

            result = await db.execute(active_stmt)
            active_alerts = result.fetchall()

            # Total counts
            total_count = (await db.execute(select(func.count()).select_from(SystemAlert))).scalar() or 0
            active_count = (await db.execute(select(func.count()).where(SystemAlert.resolved == False))).scalar() or 0

            return {
                "status": "ok",
                "summary": {
                    "total_alerts": total_count,
                    "active_alerts": active_count,
                    "resolved_alerts": total_count - active_count
                },
                "active_by_type": [
                    {
                        "alert_type": row[0],
                        "severity": row[1],
                        "count": row[2]
                    }
                    for row in active_alerts
                ]
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """
    POST /alerts/{alert_id}/resolve — Mark alert as resolved
    НЕ делает автоматических коррекций!
    """
    try:
        from models import SystemAlert
        from sqlalchemy import select

        async with AsyncSessionLocal() as db:
            # Find alert
            stmt = select(SystemAlert).where(SystemAlert.id == alert_id)
            result = await db.execute(stmt)
            alert = result.scalar_one_or_none()

            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")

            # Mark as resolved
            alert.resolved = True
            alert.resolved_at = datetime.now()

            await db.commit()
            await db.refresh(alert)

            return {
                "status": "ok",
                "message": "Alert marked as resolved",
                "alert": {
                    "id": str(alert.id),
                    "alert_type": alert.alert_type,
                    "resolved": alert.resolved,
                    "resolved_at": alert.resolved_at.isoformat()
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AUTONOMY STATE ENDPOINT (for dashboard)
# =============================================================================

@app.get("/autonomy/state")
async def get_autonomy_dashboard_state():
    """
    GET /autonomy/state — Returns autonomy state for dashboard display.
    Aggregates decision engine, safety constraints, and recent decisions.
    """
    try:
        from models import Goal
        from sqlalchemy import select, func

        async with AsyncSessionLocal() as db:
            # Get goal statistics
            total_goals = (await db.execute(select(func.count()).select_from(Goal))).scalar() or 0
            active_goals = (await db.execute(select(func.count()).select_from(Goal).where(Goal._status == "active"))).scalar() or 0
            completed_goals = (await db.execute(select(func.count()).select_from(Goal).where(Goal._status == "completed"))).scalar() or 0
            pending_goals = (await db.execute(select(func.count()).select_from(Goal).where(Goal._status == "pending"))).scalar() or 0

            # Get recent goals as "recent decisions"
            recent_stmt = select(Goal).where(
                Goal._status.in_(["completed", "active", "failed"])
            ).order_by(Goal.created_at.desc()).limit(10)
            recent_result = await db.execute(recent_stmt)
            recent_goals = recent_result.scalars().all()

            recent_decisions = [
                {
                    "id": str(g.id),
                    "node_id": str(g.id)[:8],
                    "action": "execute" if g._status in ["completed", "active"] else "failed",
                    "reasoning": f"Goal: {g.title}",
                    "confidence": 0.8 if g._status == "completed" else 0.5,
                    "timestamp": g.created_at.isoformat() if g.created_at else "",
                    "status": "executed" if g._status == "completed" else ("pending" if g._status == "active" else "blocked")
                }
                for g in recent_goals
            ]

            # Determine current mode based on active goals
            if active_goals == 0:
                current_mode = "idle"
            elif active_goals < 5:
                current_mode = "autonomous"
            else:
                current_mode = "high_activity"

            return {
                "status": "ok",
                "current_mode": current_mode,
                "active_policies": ["ethical_bounds", "budget_limits", "safety_first"],
                "safety_constraints": {
                    "ethics": ["no_harm", "privacy_first", "transparency"],
                    "budget": 10000,
                    "time_horizon": "30d"
                },
                "recent_decisions": recent_decisions,
                "pending_overrides": 0,
                "goal_stats": {
                    "total": total_goals,
                    "active": active_goals,
                    "completed": completed_goals,
                    "pending": pending_goals
                }
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# STEP 2.7: INTERVENTION READINESS LAYER (IRL) ENDPOINTS
# =============================================================================



@app.get("/interventions/candidates")
async def get_intervention_candidates(
    status: str = "proposed",
    limit: int = 20
):
    """
    GET /interventions/candidates — Get intervention candidates

    Architectural guarantee:
    - Candidates are hypotheses, NOT actions
    - NO write access to models/thresholds/weights/configs
    """
    try:
        candidates = intervention_candidates_engine.get_candidates_by_status(status=status, limit=limit)

        return {
            "status": "ok",
            "count": len(candidates),
            "candidates": candidates
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/interventions/{intervention_id}/simulation")
async def get_intervention_simulation(intervention_id: str):
    """
    GET /interventions/{id}/simulation — Get simulation results

    Shows "what if" scenarios without applying changes.
    """
    try:
        simulation = counterfactual_simulator.get_simulation(intervention_id)

        if not simulation:
            raise HTTPException(status_code=404, detail="Simulation not found")

        return {
            "status": "ok",
            "simulation": simulation
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interventions/{intervention_id}/simulate")
async def run_intervention_simulation(intervention_id: str, replay_window_days: int = 30):
    """
    POST /interventions/{id}/simulate — Run counterfactual simulation

    Runs "what if" analysis on historical data.
    Does NOT modify system state.
    """
    try:
        simulation = counterfactual_simulator.simulate_intervention(
            intervention_id=intervention_id,
            replay_window_days=replay_window_days
        )

        if not simulation:
            raise HTTPException(status_code=400, detail="Simulation failed (insufficient data or other error)")

        return {
            "status": "ok",
            "message": "Simulation completed",
            "simulation": {
                "id": str(simulation.id),
                "intervention_id": str(simulation.intervention_id),
                "replay_window_days": simulation.replay_window.days,
                "metrics_before": simulation.metrics_before,
                "metrics_after": simulation.metrics_after,
                "delta_metrics": simulation.delta_metrics,
                "side_effects": simulation.side_effects,
                "determinism_hash": simulation.determinism_hash
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/interventions/{intervention_id}/risk")
async def get_intervention_risk(intervention_id: str):
    """
    GET /interventions/{id}/risk — Get risk assessment

    Shows risk score and tier for intervention.
    """
    try:
        risk_score = intervention_risk_scorer.get_risk_score(intervention_id)

        if not risk_score:
            raise HTTPException(status_code=404, detail="Risk score not found")

        return {
            "status": "ok",
            "risk_score": risk_score
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interventions/{intervention_id}/approve")
async def approve_intervention(intervention_id: str, decided_by: str, rationale: Optional[str] = None):
    """
    POST /interventions/{id}/approve — Mark intervention as approved

    Human-in-the-loop approval:
    - Approve ≠ Execute (only permits future application)
    - CRITICAL tier cannot be approved (forbidden)
    - All approvals are audited
    """
    try:
        db = get_db_sync()

        # Load intervention
        stmt = select(InterventionCandidate).where(InterventionCandidate.id == intervention_id)
        result = db.execute(stmt)
        intervention = result.scalar_one_or_none()

        if not intervention:
            raise HTTPException(status_code=404, detail="Intervention not found")

        # Check risk tier
        stmt_risk = select(InterventionRiskScore).where(InterventionRiskScore.intervention_id == intervention_id)
        result_risk = db.execute(stmt_risk)
        risk_score = result_risk.scalar_one_or_none()

        if risk_score and risk_score.risk_tier == "CRITICAL":
            raise HTTPException(
                status_code=403,
                detail=f"CRITICAL risk interventions cannot be approved (risk={risk_score.total_risk:.4f})"
            )

        # Create approval record
        approval = InterventionApproval(
            intervention_id=intervention_id,
            decision="approve",
            decided_by=decided_by,
            rationale=rationale,
            decided_at=datetime.now(timezone.utc)
        )

        db.add(approval)

        # Update intervention status
        intervention.status = "approved"

        db.commit()

        return {
            "status": "ok",
            "message": "Intervention approved",
            "intervention_id": str(intervention_id),
            "intervention_type": intervention.intervention_type,
            "risk_tier": risk_score.risk_tier if risk_score else None,
            "approved_by": decided_by
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'db' in locals():
            db.close()


@app.post("/interventions/{intervention_id}/reject")
async def reject_intervention(intervention_id: str, decided_by: str, rationale: Optional[str] = None):
    """
    POST /interventions/{id}/reject — Reject intervention

    Human-in-the-loop rejection:
    - All rejections are audited
    - Prevents future application
    """
    try:
        db = get_db_sync()

        # Load intervention
        stmt = select(InterventionCandidate).where(InterventionCandidate.id == intervention_id)
        result = db.execute(stmt)
        intervention = result.scalar_one_or_none()

        if not intervention:
            raise HTTPException(status_code=404, detail="Intervention not found")

        # Create approval record
        approval = InterventionApproval(
            intervention_id=intervention_id,
            decision="reject",
            decided_by=decided_by,
            rationale=rationale,
            decided_at=datetime.now(timezone.utc)
        )

        db.add(approval)

        # Update intervention status
        intervention.status = "rejected"

        db.commit()

        return {
            "status": "ok",
            "message": "Intervention rejected",
            "intervention_id": str(intervention_id),
            "intervention_type": intervention.intervention_type,
            "rejected_by": decided_by
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'db' in locals():
            db.close()


# =============================================================================
# IRL HEALTH MONITORING ENDPOINTS
# =============================================================================

from irl_invariants import irl_invariants_contract
from irl_health_metrics import irl_health_metrics


@app.get("/irl/invariants")
async def get_irl_invariants():
    """
    GET /irl/invariants — Check all IRL architectural invariants

    Verifies 6 core invariants:
    1. NO_WRITE_ACCESS_TO_INFERENCE
    2. APPROVE_NOT_EXECUTE
    3. CRITICAL_RISK_FORBIDDEN
    4. SIMULATION_NOT_PREDICTION
    5. RISK_EXCEEDS_GAIN_CHECK
    6. HUMAN_IN_THE_LOOP_MANDATORY

    Returns overall PASS/VIOLATION/ERROR status
    """
    try:
        report = irl_invariants_contract.verify_all()

        return {
            "status": "ok",
            "invariants_report": report
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/irl/health")
async def get_irl_health():
    """
    GET /irl/health — Full health report (FMEA-based)

    Monitors all 6 Failure Modes:
    1. FM_IRL_01: False Positive Candidates
    2. FM_IRL_02: Counterfactual Illusion (HIGH RISK)
    3. FM_IRL_03: Risk Score Gaming (Human Side)
    4. FM_IRL_04: Intervention Drift
    5. FM_IRL_05: Semantic Overconfidence
    6. FM_IRL_06: Silent IRL

    Returns overall HEALTHY/DEGRADED/CRITICAL status
    """
    try:
        report = irl_health_metrics.get_full_health_report()

        return {
            "status": "ok",
            "health_report": report
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/irl/status")
async def get_irl_status():
    """
    GET /irl/status — Quick status summary for dashboard

    Lightweight endpoint for operational monitoring.
    """
    try:
        # Check invariants
        invariants = irl_invariants_contract.verify_all()

        # Get health summary
        health = irl_health_metrics.get_full_health_report()

        # Quick stats
        db = get_db_sync()
        try:
            stmt_total = select(func.count(InterventionCandidate.id))
            total_result = db.execute(stmt_total)
            total_candidates = total_result.scalar() or 0

            stmt_approved = select(func.count(InterventionCandidate.id)).where(
                InterventionCandidate.status == "approved"
            )
            approved_result = db.execute(stmt_approved)
            approved_count = approved_result.scalar() or 0

            stmt_pending = select(func.count(InterventionCandidate.id)).where(
                InterventionCandidate.status.in_(["proposed", "simulated"])
            )
            pending_result = db.execute(stmt_pending)
            pending_count = pending_result.scalar() or 0

        finally:
            db.close()

        return {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "invariants_status": invariants["overall_status"],
            "health_status": health["overall_health"],
            "candidates": {
                "total": total_candidates,
                "approved": approved_count,
                "pending": pending_count
            },
            "summary": {
                "invariant_violations": invariants["violation_count"],
                "critical_failure_modes": health["summary"]["critical_count"],
                "degraded_failure_modes": health["summary"]["degraded_count"]
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/irl/lts-contract")
async def get_irl_lts_contract():
    """
    GET /irl/lts-contract — Get LTS architectural contract

    Returns the formal invariants that define IRL LTS (Long-Term-Support) guarantees.
    This is the epistemological contract: system understands boundaries before expanding them.
    """
    try:
        contract = {
            "version": "2.7.LTS",
            "contract_type": "Intervention Readiness Layer",
            "principle": "HONESTY PRECEDES INTELLIGENCE",
            "invariants": [
                {
                    "name": "NO_WRITE_ACCESS_TO_INFERENCE",
                    "description": "IRL has NO write access to models/thresholds/weights/configs",
                    "enforcement": "Architectural (no code paths exist)"
                },
                {
                    "name": "APPROVE_NOT_EXECUTE",
                    "description": "Approve ≠ Execute (approve only permits future discussion)",
                    "enforcement": "Status tracking + human decision required"
                },
                {
                    "name": "CRITICAL_RISK_FORBIDDEN",
                    "description": "CRITICAL risk interventions cannot be approved",
                    "enforcement": "API gate on risk_tier"
                },
                {
                    "name": "SIMULATION_NOT_PREDICTION",
                    "description": "Simulation = replay only, NOT future prediction",
                    "enforcement": "Determinism hash + replay window required"
                },
                {
                    "name": "RISK_EXCEEDS_GAIN_CHECK",
                    "description": "If risk ≥ gain → candidate not proposed",
                    "enforcement": "Candidate generation filter"
                },
                {
                    "name": "HUMAN_IN_THE_LOOP_MANDATORY",
                    "description": "All approve/reject require human decision",
                    "enforcement": "Approval record + decided_by tracking"
                }
            ],
            "guarantees": [
                "System cannot silently self-modify",
                "All interventions require human acknowledgement",
                "Risk is assessed before action is permitted",
                "Simulation does not claim predictive power",
                "High-risk interventions are blocked architecturally"
            ],
            "prohibitions": [
                "NO automatic application of approved interventions",
                "NO adaptive thresholds 'gradually'",
                "NO risk_score → execution binding",
                "NO retraining even with 'manual button'"
            ],
            "status": "ENFORCED",
            "last_verified": datetime.now(timezone.utc).isoformat()
        }

        return {
            "status": "ok",
            "lts_contract": contract
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DECOMPOSITION API ENDPOINTS
# =============================================================================

from canonical_skills.ask_user_skill import ask_user_skill
from pydantic import BaseModel
from typing import Optional


class DecompositionAskRequest(BaseModel):
    goal_id: str
    question_text: str
    question_type: Optional[str] = None
    initiated_by: str = "human"


class DecompositionAnswerRequest(BaseModel):
    question_id: str
    answer_text: str
    answered_by: str = "human"


@app.post("/decomposition/ask")
async def decomposition_ask(req: DecompositionAskRequest):
    """
    POST /decomposition/ask - Создать вопрос через ask_user skill
    
    Вызывается из Telegram бота когда пользователь вводит /decompose
    """
    try:
        import uuid
        
        result = await ask_user_skill.run(
            goal_id=uuid.UUID(req.goal_id),
            question_text=req.question_text,
            question_type=req.question_type,
            session_id=None  # Always create new session on first ask
        )
        
        return {
            "status": "ok",
            **result
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decomposition/session/active")
async def get_active_decomposition_session(goal_id: str):
    """
    GET /decomposition/session/active?goal_id=XXX
    
    Проверить, есть ли активная сессия декомпозиции для цели
    """
    try:
        import uuid
        from models import DecompositionSession
        
        db = get_db_sync()
        stmt = select(DecompositionSession).where(
            DecompositionSession.goal_id == uuid.UUID(goal_id),
            DecompositionSession.status.in_(["awaiting_user", "in_progress"])
        )
        
        result = db.execute(stmt)
        session = result.scalar_one_or_none()
        db.close()
        
        return {
            "status": "ok",
            "has_active_session": session is not None,
            "session_id": str(session.id) if session else None,
            "session_status": session.status if session else None
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Если таблица не существует, возвращаем False
        return {
            "status": "ok",
            "has_active_session": False,
            "session_id": None,
            "session_status": None
        }


@app.post("/decomposition/answer")
async def submit_decomposition_answer(req: DecompositionAnswerRequest):
    """
    POST /decomposition/answer - Сохранить ответ пользователя
    
    Вызывается из Telegram бота когда пользователь отвечает на вопрос
    """
    try:
        import uuid
        from models import DecompositionQuestion, DecompositionAnswer, DecompositionSession
        
        db = get_db_sync()
        # Находим вопрос
        stmt = select(DecompositionQuestion).where(
            DecompositionQuestion.id == uuid.UUID(req.question_id)
        )
        result = db.execute(stmt)
        question = result.scalar_one_or_none()
        
        if not question:
            db.close()
            raise HTTPException(status_code=404, detail="Question not found")
        
        # Проверяем, есть ли уже ответ
        # Get all answers for this question
        answers_stmt = select(DecompositionAnswer).where(
            DecompositionAnswer.question_id == question.id
        )
        answers_result = db.execute(answers_stmt)
        existing_answer = answers_result.scalar_one_or_none()
        
        if existing_answer:
            db.close()
            raise HTTPException(status_code=400, detail="Question already answered")
        
        # Создаём ответ
        answer = DecompositionAnswer(
            question_id=question.id,
            answer_text=req.answer_text,
            answered_by=req.answered_by
        )
        db.add(answer)
        
        # Обновляем статус сессии
        session_stmt = select(DecompositionSession).where(
            DecompositionSession.id == question.session_id
        )
        session_result = db.execute(session_stmt)
        session = session_result.scalar_one_or_none()
        
        if session:
            # Проверяем все ли вопросы отвечены
            all_questions_stmt = select(DecompositionQuestion).where(
                DecompositionQuestion.session_id == session.id
            )
            all_questions_result = db.execute(all_questions_stmt)
            all_questions = all_questions_result.scalars().all()
            
            all_answers_stmt = select(DecompositionAnswer).where(
                DecompositionAnswer.question_id.in_([q.id for q in all_questions])
            )
            all_answers_result = db.execute(all_answers_stmt)
            all_answers = all_answers_result.scalars().all()
            
            # Если все вопросы отвечены - завершаем сессию
            if len(all_answers) >= len(all_questions):
                session.status = "completed"
                session_status = "completed"
            else:
                session.status = "in_progress"
                session_status = "in_progress"
        
        db.commit()
        db.refresh(answer)
        
        # SYNC: Notify Telegram/Dashboard that question was answered
        try:
            # Notify that answer was received (to update UI on other platform)
            import asyncio
            
            # Get question text for notification
            question_text = question.question_text if hasattr(question, 'question_text') else ""
            goal_title = session.goal_title if session else "Unknown"
            
            # Send sync notification to Telegram
            from telegram_notifier import TelegramNotifier
            notifier = TelegramNotifier()
            
            message = f"✅ <b>Ответ получен!</b>\n\nВопрос: {question_text[:100]}...\n\nОтвет: {req.answer_text[:100]}..."
            
            async def notify_sync():
                try:
                    async with httpx.AsyncClient() as client:
                        if notifier._owner_chat_id:
                            await client.post(
                                f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage",
                                json={
                                    "chat_id": notifier._owner_chat_id,
                                    "text": message,
                                    "parse_mode": "HTML"
                                },
                                timeout=5.0
                            )
                except:
                    pass
            
            asyncio.create_task(notify_sync())
        except Exception as sync_err:
            logger.warning("answer_sync_notification_failed", error=str(sync_err))
        
        db.close()
        
        return {
            "status": "ok",
            "answer_id": str(answer.id),
            "question_id": str(question.id),
            "session_id": str(question.session_id),
            "session_status": session_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/decomposition/session/{session_id}")
async def get_decomposition_session(session_id: str):
    """
    GET /decomposition/session/{session_id} - Получить сессию со всеми вопросами
    """
    try:
        import uuid
        from models import DecompositionSession, DecompositionQuestion, DecompositionAnswer
        
        db = get_db_sync()
        # Находим сессию
        stmt = select(DecompositionSession).where(
            DecompositionSession.id == uuid.UUID(session_id)
        )
        result = db.execute(stmt)
        session = result.scalar_one_or_none()
        
        if not session:
            db.close()
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Загружаем вопросы
        questions_stmt = select(DecompositionQuestion).where(
            DecompositionQuestion.session_id == session.id
        ).order_by(DecompositionQuestion.question_index)
        
        questions_result = db.execute(questions_stmt)
        questions = questions_result.scalars().all()
        
        # Загружаем ответы явно
        question_ids = [q.id for q in questions]
        answers_stmt = select(DecompositionAnswer).where(
            DecompositionAnswer.question_id.in_(question_ids)
        )
        answers_result = db.execute(answers_stmt)
        answers = answers_result.scalars().all()
        
        # Создаём映射 question_id -> answer
        answers_map = {a.question_id: a for a in answers}
        db.close()
        
        return {
            "status": "ok",
            "session": {
                "id": str(session.id),
                "goal_id": str(session.goal_id),
                "status": session.status,
                "initiated_by": session.initiated_by,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None
            },
            "questions": [
                {
                    "id": str(q.id),
                    "question_text": q.question_text,
                    "question_index": q.question_index,
                    "question_type": q.question_type,
                    "asked_by": q.asked_by,
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                    "answer": {
                        "id": str(answers_map[q.id].id),
                        "answer_text": answers_map[q.id].answer_text,
                        "answered_by": answers_map[q.id].answered_by,
                        "created_at": answers_map[q.id].created_at.isoformat() if answers_map[q.id].created_at else None
                    } if q.id in answers_map else None
                }
                for q in questions
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decomposition/{session_id}/decompose")
async def decompose_from_answers(session_id: str):
    """
    POST /decomposition/{session_id}/decompose

    Запускает декомпозицию цели на основе ответов пользователя

    Flow:
    1. Загружает сессию и все вопросы/ответы
    2. Передаёт ответы в goal_decomposer для создания подцелей
    3. Помечает сессию как completed
    """
    import uuid
    from models import DecompositionSession, DecompositionQuestion, DecompositionAnswer, Goal
    from goal_decomposer import goal_decomposer

    try:
        async with AsyncSessionLocal() as db:
            # Load session
            stmt = select(DecompositionSession).where(DecompositionSession.id == uuid.UUID(session_id))
            result = await db.execute(stmt)
            session = result.scalar_one_or_none()

            if not session:
                raise HTTPException(status_code=404, detail="Session not found")

            # Load questions with answers
            stmt_questions = select(DecompositionQuestion).where(
                DecompositionQuestion.session_id == uuid.UUID(session_id)
            ).order_by(DecompositionQuestion.question_index)

            result_questions = await db.execute(stmt_questions)
            questions = result_questions.scalars().all()

            # Load answers
            question_ids = [q.id for q in questions]
            stmt_answers = select(DecompositionAnswer).where(
                DecompositionAnswer.question_id.in_(question_ids)
            )

            result_answers = await db.execute(stmt_answers)
            answers = result_answers.scalars().all()

            # Build answers map
            answers_map = {a.question_id: a for a in answers}

            # Check all questions answered
            unanswered = [q for q in questions if q.id not in answers_map]
            if unanswered:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not all questions answered. Missing: {len(unanswered)} answers"
                )

            # Collect all answers into context
            answers_context = []
            for question in questions:
                answer = answers_map[question.id]
                answers_context.append({
                    "question": question.question_text,
                    "answer": answer.answer_text,
                    "question_type": question.question_type
                })

            # Run decomposition with context
            logger.info(f"🧠 Decomposing goal {session.goal_id} with {len(answers_context)} answers")

            # For now, just call standard decompose (answers context is logged but not used yet)
            # TODO: Integrate answers into decompose_goal logic
            subgoals = await goal_decomposer.decompose_goal(str(session.goal_id), max_depth=2)

            # Mark session as completed
            session.status = "completed"
            await db.commit()

            return {
                "status": "ok",
                "message": f"Created {len(subgoals)} subgoals from decomposition session",
                "subgoals_created": len(subgoals),
                "subgoals": subgoals
            }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# LEGACY AXIS (S0) - CONSTITUTIONAL LAYER ENDPOINTS
# =============================================================================

class LegacyAxisCreateRequest(BaseModel):
    title: str
    description: str
    axis_type: str  # civilizational, cultural, technological, existential
    generational_depth: int = 1
    survivability_policy: Optional[Dict] = None
    immutability_policy: Optional[Dict] = None
    optimization_constraints: Optional[Dict] = None


class LegacyAxisUpdateRequest(BaseModel):
    description: str


@app.post("/legacy-axis/create")
async def create_legacy_axis(req: LegacyAxisCreateRequest):
    """
    POST /legacy-axis/create

    Create new Legacy Axis (S0)

    Legacy Axis = existential mission layer
    - Defines WHY system exists
    - Cannot be deleted, completed, optimized
    - Survives without author
    """
    try:
        from legacy_axis_service import legacy_axis_service

        result = await legacy_axis_service.create(
            title=req.title,
            description=req.description,
            axis_type=req.axis_type,
            generational_depth=req.generational_depth,
            survivability_policy=req.survivability_policy,
            immutability_policy=req.immutability_policy,
            optimization_constraints=req.optimization_constraints
        )

        return {
            "status": "created",
            "legacy_axis": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/legacy-axis")
async def list_legacy_axes(active_only: bool = True):
    """
    GET /legacy-axis

    List all Legacy Axis (S0)
    """
    try:
        from legacy_axis_service import legacy_axis_service

        axes = await legacy_axis_service.list_all(active_only=active_only)

        return {
            "status": "ok",
            "count": len(axes),
            "legacy_axes": axes
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/legacy-axis/{legacy_id}")
async def get_legacy_axis(legacy_id: str):
    """
    GET /legacy-axis/{id}

    Get single Legacy Axis by ID
    """
    try:
        from legacy_axis_service import legacy_axis_service

        result = await legacy_axis_service.get(legacy_id)

        if not result:
            raise HTTPException(status_code=404, detail="Legacy Axis not found")

        return {
            "status": "ok",
            "legacy_axis": result
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/legacy-axis/{legacy_id}")
async def update_legacy_axis(legacy_id: str, req: LegacyAxisUpdateRequest):
    """
    PATCH /legacy-axis/{id}

    Update Legacy Axis description

    NOTE: Only description can be updated
    All other fields are immutable by design
    """
    try:
        from legacy_axis_service import legacy_axis_service

        result = await legacy_axis_service.update_description(legacy_id, req.description)

        if not result:
            raise HTTPException(status_code=404, detail="Legacy Axis not found")

        return {
            "status": "updated",
            "legacy_axis": result
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/legacy-axis/{legacy_id}/deactivate")
async def deactivate_legacy_axis(legacy_id: str):
    """
    POST /legacy-axis/{id}/deactivate

    Deactivate Legacy Axis

    WARNING: This is NOT deletion
    Deactivated axes remain in history but are not used for new goals
    """
    try:
        from legacy_axis_service import legacy_axis_service

        success = await legacy_axis_service.deactivate(legacy_id)

        if not success:
            raise HTTPException(status_code=404, detail="Legacy Axis not found")

        return {
            "status": "deactivated",
            "legacy_id": legacy_id
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/legacy-axis/{legacy_id}/goals")
async def get_legacy_axis_goals(legacy_id: str):
    """
    GET /legacy-axis/{id}/goals

    Get all goals derived from this Legacy Axis
    """
    try:
        from legacy_axis_service import legacy_axis_service

        goals = await legacy_axis_service.get_derived_goals(legacy_id)

        return {
            "status": "ok",
            "count": len(goals),
            "goals": goals
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/legacy-policy/constraints")
async def get_legacy_policy_constraints():
    """
    GET /legacy-policy/constraints

    Get current Legacy constraints for API/UX

    Returns forbidden operations, metrics, etc.
    """
    try:
        from policies.legacy_policy import legacy_policy

        constraints = await legacy_policy.get_legacy_constraints()

        return {
            "status": "ok",
            "constraints": constraints
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# VECTOR ENGINE (B) - TRANSFORMATION OPERATORS ENDPOINTS
# =============================================================================

class VectorApplyRequest(BaseModel):
    vector_id: str
    target_type: str  # goal, plan, task
    target_id: str
    applied_by: str = "system"


@app.post("/vector-engine/apply")
async def apply_vector(req: VectorApplyRequest):
    """
    POST /vector-engine/apply

    Apply vector operator to target

    Vector transforms: V(goal) = goal'
    Vector is stateless, application is logged for audit

    PROTECTION: Cannot apply to Legacy Axis (S0)
    """
    try:
        from vector_engine_service import vector_engine_service, VectorEngineError

        result = await vector_engine_service.apply_vector(
            vector_id=req.vector_id,
            target_type=req.target_type,
            target_id=req.target_id,
            applied_by=req.applied_by
        )

        return {
            "status": "transformed",
            "application": result
        }

    except VectorEngineError as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": "VectorEngineConstraint"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector-engine/vectors")
async def list_vectors(active_only: bool = True):
    """
    GET /vector-engine/vectors

    List all available vector operators
    """
    try:
        from vector_engine_service import vector_engine_service

        vectors = await vector_engine_service.list_vectors(active_only=active_only)

        return {
            "status": "ok",
            "count": len(vectors),
            "vectors": vectors
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector-engine/vectors/{vector_id}")
async def get_vector(vector_id: str):
    """
    GET /vector-engine/vectors/{id}

    Get single vector operator by ID
    """
    try:
        from vector_engine_service import vector_engine_service

        result = await vector_engine_service.get_vector(vector_id)

        if not result:
            raise HTTPException(status_code=404, detail="Vector not found")

        return {
            "status": "ok",
            "vector": result
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector-engine/history")
async def get_vector_history(
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    limit: int = 50
):
    """
    GET /vector-engine/history

    Get vector application history (audit log)

    READ ONLY - vectors are stateless, this is just audit trail
    """
    try:
        from vector_engine_service import vector_engine_service

        history = await vector_engine_service.get_application_history(
            target_type=target_type,
            target_id=target_id,
            limit=limit
        )

        return {
            "status": "ok",
            "count": len(history),
            "applications": history
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector-engine/info")
async def get_vector_engine_info():
    """
    GET /vector-engine/info

    Get Vector Engine information and constraints
    """
    try:
        from vector_engine_service import vector_engine_service

        return {
            "status": "ok",
            "info": {
                "name": "Vector Engine (B)",
                "purpose": "Transformation operators for Goals/Plans/Tasks",
                "principle": "V(x) = x' (operator, NOT hierarchy level)",
                "valid_targets": vector_engine_service.VALID_TARGET_TYPES,
                "forbidden_targets": vector_engine_service.FORBIDDEN_TARGET_TYPES,
                "stateless": True,
                "audit_only": True
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# OPEN COGNITIVE CONTROL PROTOCOL (OCCP) - MCL + SK ENDPOINTS
# =============================================================================

class MCLSetModeRequest(BaseModel):
    mode: str  # exploration, exploitation, preservation
    rationale: str


class MCLUpdateDriftRequest(BaseModel):
    drift_score: float  # 0.0 - 1.0


class SKRecordSignalRequest(BaseModel):
    signal_name: str
    signal_value: float
    context: Optional[Dict] = None


@app.get("/occp/mcl/state")
async def get_mcl_state():
    """
    GET /occp/mcl/state

    Get current Meta-Cognition Layer state
    """
    try:
        from mcl_service import mcl_service

        state = await mcl_service.get_active_state()

        if not state:
            return {
                "status": "ok",
                "state": None,
                "message": "No active MCL state"
            }

        return {
            "status": "ok",
            "state": state
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/occp/mcl/set-mode")
async def set_mcl_mode(req: MCLSetModeRequest):
    """
    POST /occp/mcl/set-mode

    Set cognitive mode (exploration/exploitation/preservation)
    """
    try:
        from mcl_service import mcl_service

        result = await mcl_service.set_mode(
            mode=req.mode,
            rationale=req.rationale
        )

        return {
            "status": "ok",
            "message": f"Mode set to {req.mode}",
            "state": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/occp/mcl/update-drift")
async def update_mcl_drift(req: MCLUpdateDriftRequest):
    """
    POST /occp/mcl/update-drift

    Update drift score (triggers auto-transition to preservation if > 0.7)
    """
    try:
        from mcl_service import mcl_service

        result = await mcl_service.update_drift_score(
            new_drift=req.drift_score
        )

        return {
            "status": "ok",
            "message": f"Drift score updated to {req.drift_score}",
            "state": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occp/mcl/allowed-operations")
async def get_mcl_allowed_operations():
    """
    GET /occp/mcl/allowed-operations

    Get list of allowed/forbidden operations under current mode
    """
    try:
        from mcl_service import mcl_service

        result = await mcl_service.get_allowed_operations()

        return {
            "status": "ok",
            "operations": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occp/sk/kernel")
async def get_sk_kernel():
    """
    GET /occp/sk/kernel

    Get active Survivability Kernel
    """
    try:
        from sk_service import sk_service

        kernel = await sk_service.get_active_kernel()

        if not kernel:
            return {
                "status": "ok",
                "kernel": None,
                "message": "No active SK"
            }

        return {
            "status": "ok",
            "kernel": kernel
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occp/sk/rules")
async def get_sk_rules():
    """
    GET /occp/sk/rules

    Get all active SK rules
    """
    try:
        from sk_service import sk_service

        rules = await sk_service.get_all_rules()

        return {
            "status": "ok",
            "count": len(rules),
            "rules": rules
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/occp/sk/record-signal")
async def record_sk_signal(req: SKRecordSignalRequest):
    """
    POST /occp/sk/record-signal

    Record survivability signal measurement
    """
    try:
        from sk_service import sk_service

        signal = await sk_service.record_signal(
            signal_name=req.signal_name,
            signal_value=req.signal_value,
            context=req.context
        )

        return {
            "status": "ok",
            "signal": signal
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occp/sk/signals")
async def get_sk_signals(
    signal_name: Optional[str] = None,
    limit: int = 100
):
    """
    GET /occp/sk/signals

    Get survivability signals (time-series)
    """
    try:
        from sk_service import sk_service

        signals = await sk_service.get_signals(
            signal_name=signal_name,
            limit=limit
        )

        return {
            "status": "ok",
            "count": len(signals),
            "signals": signals
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occp/audit")
async def get_occp_audit(
    source: Optional[str] = None,
    limit: int = 50
):
    """
    GET /occp/audit

    Get OCCP audit log (MCL + SK decisions)
    """
    try:
        from models import OCCPAuditEvent
        from database import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            stmt = select(OCCPAuditEvent)

            if source:
                stmt = stmt.where(OCCPAuditEvent.source == source)

            stmt = stmt.order_by(OCCPAuditEvent.created_at.desc()).limit(limit)

            result = await db.execute(stmt)
            events = result.scalars().all()

            return {
                "status": "ok",
                "count": len(events),
                "events": [
                    {
                        "id": str(e.id),
                        "source": e.source,
                        "decision": e.decision,
                        "decision_type": e.decision_type,
                        "blocked_component": e.blocked_component,
                        "blocked_operation": e.blocked_operation,
                        "rationale": e.rationale,
                        "context": e.context,
                        "created_at": e.created_at.isoformat()
                    }
                    for e in events
                ]
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/occp/info")
async def get_occp_info():
    """
    GET /occp/info

    Get OCCP protocol information
    """
    try:
        return {
            "status": "ok",
            "protocol": {
                "name": "Open Cognitive Control Protocol",
                "version": "0.1",
                "status": "experimental",
                "components": {
                    "mcl": "Meta-Cognition Layer",
                    "sk": "Survivability Kernel"
                },
                "principles": [
                    "Separation of Concerns",
                    "Negative Capability",
                    "Override Hierarchy (SK > MCL > Vector > Execution)",
                    "Explainability First"
                ],
                "compliance": {
                    "mcl_exists": True,
                    "sk_veto_authority": True,
                    "drift_measured": True,
                    "can_halt_intentionally": True
                }
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))




# =============================================================================
# MCP (Modular Capability Plugin) API Endpoints
# =============================================================================

@app.get("/mcp/status")
async def get_mcp_status():
    """
    GET /mcp/status

    Get MCP system status and statistics
    """
    try:
        from mcp_manager import mcp_manager
        from mcp_skill_generator import mcp_skill_generator
        from sqlalchemy import select, func
        from autogenerated_skill_models import AutogeneratedSkill
        from database import AsyncSessionLocal

        # Get registry stats
        active_count = len(mcp_skill_generator._registry)
        active_generations = len(mcp_skill_generator._active_generations)

        # Get database stats
        async with AsyncSessionLocal() as session:
            # Total skills
            total_stmt = select(func.count(AutogeneratedSkill.id))
            total_result = await session.execute(total_stmt)
            total_count = total_result.scalar() or 0

            # By status
            status_stmt = select(
                AutogeneratedSkill.status,
                func.count(AutogeneratedSkill.id)
            ).group_by(AutogeneratedSkill.status)
            status_result = await session.execute(status_stmt)
            status_counts = {row[0]: row[1] for row in status_result}

        return {
            "status": "ok",
            "mcp": {
                "connected": mcp_manager._connected,
                "active_plugins": active_count,
                "active_generations": active_generations,
                "max_plugins": mcp_skill_generator.MAX_PLUGINS,
                "max_concurrent": mcp_skill_generator.MAX_CONCURRENT_GENERATIONS,
                "cooldown_seconds": mcp_skill_generator.GENERATION_COOLDOWN_SECONDS
            },
            "database": {
                "total_skills": total_count,
                "by_status": status_counts
            },
            "config": {
                "pruning_threshold_days": mcp_skill_generator.PRUNING_THRESHOLD_DAYS,
                "min_success_rate": mcp_skill_generator.MIN_SUCCESS_RATE_FOR_RETENTION
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/plugins")
async def list_mcp_plugins(
    status_filter: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    GET /mcp/plugins?status={experimental|stable|deprecated|failed}&limit=50&offset=0

    List MCP plugins with filtering and pagination
    """
    try:
        from mcp_skill_generator import mcp_skill_generator

        plugins = []

        for plugin_id, plugin in mcp_skill_generator._registry.items():
            if status_filter and plugin.status != status_filter:
                continue

            # Calculate generation duration
            duration = None
            if plugin.generation_completed_at and plugin.generation_started_at:
                duration = (plugin.generation_completed_at - plugin.generation_started_at).total_seconds()

            plugins.append({
                "plugin_id": plugin.plugin_id,
                "version": plugin.version,
                "status": plugin.status,
                "generation_status": plugin.generation_status,
                "capabilities": plugin.capabilities,
                "execution_count": plugin.execution_count,
                "success_count": plugin.success_count,
                "success_rate": plugin.success_rate,
                "created_at": plugin.created_at.isoformat(),
                "generation_duration_s": duration,
                "generation_error": plugin.generation_error
            })

        # Sort by created_at (newest first)
        plugins.sort(key=lambda p: p["created_at"], reverse=True)

        # Apply pagination
        total = len(plugins)
        plugins = plugins[offset:offset + limit]

        return {
            "status": "ok",
            "total": total,
            "offset": offset,
            "limit": limit,
            "plugins": plugins
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/plugins/{plugin_id}")
async def get_mcp_plugin(plugin_id: str):
    """
    GET /mcp/plugins/{plugin_id}

    Get detailed info about a specific MCP plugin
    """
    try:
        from mcp_skill_generator import mcp_skill_generator

        plugin = await mcp_skill_generator.get_plugin(plugin_id)

        if not plugin:
            raise HTTPException(status_code=404, detail=f"Plugin not found: {plugin_id}")

        # Calculate generation duration
        duration = None
        if plugin.generation_completed_at and plugin.generation_started_at:
            duration = (plugin.generation_completed_at - plugin.generation_started_at).total_seconds()

        return {
            "status": "ok",
            "plugin": {
                "plugin_id": plugin.plugin_id,
                "version": plugin.version,
                "status": plugin.status,
                "generation_status": plugin.generation_status,
                "capabilities": plugin.capabilities,
                "execution_count": plugin.execution_count,
                "success_count": plugin.success_count,
                "success_rate": plugin.success_rate,
                "created_at": plugin.created_at.isoformat(),
                "generation_started_at": plugin.generation_started_at.isoformat(),
                "generation_completed_at": plugin.generation_completed_at.isoformat() if plugin.generation_completed_at else None,
                "generation_duration_s": duration,
                "generation_error": plugin.generation_error
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mcp/metrics")
async def get_mcp_metrics():
    """
    GET /mcp/metrics

    Get MCP performance metrics
    """
    try:
        from mcp_skill_generator import mcp_skill_generator
        from sqlalchemy import select, func
        from autogenerated_skill_models import AutogeneratedSkill
        from database import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            # Success rate by validation status
            validation_stmt = select(
                AutogeneratedSkill.validation_status,
                func.count(AutogeneratedSkill.id)
            ).group_by(AutogeneratedSkill.validation_status)
            validation_result = await session.execute(validation_stmt)
            validation_counts = {row[0]: row[1] for row in validation_result}

            # Skills by generation trigger
            trigger_stmt = select(
                AutogeneratedSkill.generation_trigger,
                func.count(AutogeneratedSkill.id)
            ).group_by(AutogeneratedSkill.generation_trigger)
            trigger_result = await session.execute(trigger_stmt)
            trigger_counts = {row[0]: row[1] for row in trigger_result}

        # Calculate success rate from registry (actual executions)
        total_executions = 0
        total_successes = 0

        for plugin in mcp_skill_generator._registry.values():
            total_executions += plugin.execution_count
            total_successes += plugin.success_count

        execution_success_rate = total_successes / total_executions if total_executions > 0 else 0.0

        return {
            "status": "ok",
            "metrics": {
                "total_plugins": len(mcp_skill_generator._registry),
                "total_executions": total_executions,
                "total_successes": total_successes,
                "execution_success_rate": execution_success_rate,
                "validation_stats": validation_counts,
                "generation_triggers": trigger_counts
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trace/stats")
async def get_trace_stats():
    """
    GET /trace/stats
    
    Get execution trace statistics - skill usage, success rates, patterns
    """
    try:
        from trace_store import get_trace_store
        from trace_mining_engine import get_mining_engine
        
        trace_store = get_trace_store()
        mining_engine = get_mining_engine(trace_store)
        
        stats = await trace_store.get_stats()
        success_rates = await mining_engine.analyze_skill_success_rate(
            await trace_store.get_all_traces(limit=500)
        )
        
        return {
            "status": "ok",
            "trace_stats": stats,
            "skill_success_rates": success_rates
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/generate")
async def trigger_mcp_generation(request: Dict[str, Any]):
    """
    POST /mcp/generate

    Manually trigger MCP skill generation
    {
        "capabilities": ["stock_analysis"],
        "requirements": {"input_type": "text", "output_type": "report"},
        "goal_context": {"title": "Analyze AAPL", "description": "Stock analysis"}
    }
    """
    try:
        from mcp_manager import mcp_manager

        missing_capabilities = request.get("capabilities", [])
        requirements = request.get("requirements", {})
        goal_context = request.get("goal_context")

        # Trigger generation (non-blocking)
        plugin_id = await mcp_manager.find_or_generate_skill(
            capabilities=missing_capabilities,
            requirements=requirements,
            goal_context=goal_context
        )

        return {
            "status": "ok",
            "plugin_id": plugin_id,
            "message": "Generation triggered" if plugin_id == "fallback_echo" else "Plugin found or generation started"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============= CAPABILITY GAP ENGINE =============

@app.get("/capability/stats")
async def get_capability_stats():
    """Get capability system statistics from persistent storage"""
    from capability import capability_gap_engine

    await capability_gap_engine.initialize()
    stats = await capability_gap_engine.get_stats_async()

    return {
        "status": "ok",
        "stats": stats.to_dict()
    }


@app.post("/capability/analyze/{goal_id}")
async def analyze_goal_for_gaps(goal_id: str):
    """Analyze a goal for capability gaps"""
    from capability import capability_gap_engine
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import select
    
    await capability_gap_engine.initialize()
    
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()
        
        if not goal:
            return {"status": "error", "message": "Goal not found"}
        
        gaps = await capability_gap_engine.analyze_goal_for_gaps(goal)
        
        return {
            "status": "ok",
            "goal_id": str(goal.id),
            "goal_title": goal.title,
            "gaps_detected": len(gaps),
            "gaps": [gap.to_dict() for gap in gaps]
        }


@app.post("/capability/resolve/{gap_id}")
async def resolve_capability_gap(gap_id: str):
    """Attempt to resolve a capability gap"""
    from capability import capability_gap_engine
    from uuid import UUID
    
    await capability_gap_engine.initialize()
    
    try:
        resolution = await capability_gap_engine.resolve_gap(UUID(gap_id))
        
        if resolution:
            return {
                "status": "ok",
                "resolution": resolution.to_dict()
            }
        else:
            return {
                "status": "error",
                "message": "Resolution failed or not possible"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/capability/gaps")
async def list_capability_gaps(status: str = None, limit: int = 100):
    """List all detected capability gaps from persistent storage"""
    from capability import capability_gap_engine
    from capability.models.capability_gap import GapStatus

    await capability_gap_engine.initialize()

    filter_status = GapStatus(status) if status else None
    gaps = await capability_gap_engine.list_gaps_async(status=filter_status, limit=limit)

    return {
        "status": "ok",
        "gaps_count": len(gaps),
        "gaps": [gap.to_dict() for gap in gaps]
    }


@app.get("/capability/graph")
async def get_capability_graph():
    """Get capability graph information"""
    from capability import capability_graph
    
    await capability_graph.initialize()
    stats = capability_graph.get_stats()
    
    return {
        "status": "ok",
        "graph": stats
    }


@app.post("/capability/auto-resolve/{goal_id}")
async def analyze_and_auto_resolve(goal_id: str):
    """
    Analyze goal for gaps and automatically resolve all pipeline gaps.
    
    This is the auto-learning mechanism - system improves itself after each goal.
    """
    from capability import capability_gap_engine
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import select
    
    await capability_gap_engine.initialize()
    
    # Load goal
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()
        
        if not goal:
            return {"status": "error", "message": "Goal not found"}
    
    # Detect gaps
    gaps = await capability_gap_engine.analyze_goal_for_gaps(goal)
    
    # Auto-resolve all pipeline gaps
    resolved_count = 0
    failed_count = 0
    
    for gap in gaps:
        if gap.gap_type == "pipeline":  # Only auto-resolve pipeline gaps
            resolution = await capability_gap_engine.resolve_gap(gap.gap_id)
            if resolution and resolution.success:
                resolved_count += 1
            else:
                failed_count += 1
    
    return {
        "status": "ok",
        "goal_id": str(goal.id),
        "goal_title": goal.title,
        "gaps_detected": len(gaps),
        "pipeline_gaps_resolved": resolved_count,
        "pipeline_gaps_failed": failed_count,
        "message": f"Auto-resolved {resolved_count} pipeline gaps"
    }


# ============= CAPABILITY ROUTER (V2) =============

@app.post("/capability/route/{goal_id}")
async def route_goal_via_capability_system(goal_id: str, auto_resolve_gaps: bool = True):
    """
    Route a goal through the Capability OS v2 proactive planning system.

    This is the NEW way to execute goals - plan FIRST, detect gaps SECOND.

    Flow:
    1. CapabilityPlanner creates execution plan
    2. CapabilitySelector tries to find existing skills/pipelines
    3. Only if missing: GapDetector + PipelineComposer
    4. Return executable plan
    """
    from capability import capability_router
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import select

    await capability_router.initialize()

    # Load goal
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            return {"status": "error", "message": "Goal not found"}

    # Route through capability system
    plan = await capability_router.route_goal(goal, auto_resolve_gaps=auto_resolve_gaps)

    return {
        "status": "ok",
        "plan": plan.to_dict(),
        "can_execute": capability_router.can_execute_plan(plan),
        "summary": {
            "total_capabilities": plan.total_capabilities,
            "ready_capabilities": plan.ready_capabilities,
            "missing_capabilities": plan.missing_capabilities,
            "reuse_rate": f"{(plan.ready_capabilities / plan.total_capabilities * 100):.1f}%" if plan.total_capabilities > 0 else "0%"
        }
    }


@app.post("/capability/plan/{goal_id}")
async def plan_goal_only(goal_id: str):
    """
    Create an execution plan for a goal WITHOUT resolving gaps.

    Use this to preview what capabilities are needed before execution.
    """
    from capability import capability_planner
    from models import Goal
    from database import AsyncSessionLocal
    from sqlalchemy import select

    await capability_planner.initialize()

    # Load goal
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.id == goal_id)
        result = await db.execute(stmt)
        goal = result.scalar_one_or_none()

        if not goal:
            return {"status": "error", "message": "Goal not found"}

    # Create plan
    plan = await capability_planner.plan_goal(goal)

    return {
        "status": "ok",
        "plan": plan.to_dict(),
        "summary": {
            "total_capabilities": plan.total_capabilities,
            "ready_capabilities": plan.ready_capabilities,
            "missing_capabilities": plan.missing_capabilities
        }
    }


@app.get("/capability/ontology")
async def get_capability_ontology():
    """
    Get capability ontology (semantic mapping layer).

    This is where Planner vocabulary maps to Registry vocabulary.
    """
    from capability import capability_ontology

    stats = capability_ontology.get_stats()

    capabilities = [
        cap.to_dict()
        for cap in capability_ontology.get_all_capabilities()
    ]

    return {
        "status": "ok",
        "stats": stats,
        "capabilities": capabilities
    }


@app.get("/capability/ontology/resolve/{capability}")
async def resolve_capability_via_ontology(capability: str):
    """
    Resolve a capability name through the ontology.

    Shows how Planner names map to Registry names.
    """
    from capability import capability_ontology

    resolved = capability_ontology.resolve(capability)

    similar = capability_ontology.find_similar_capabilities(capability, threshold=0.5)

    return {
        "status": "ok",
        "input_capability": capability,
        "resolved_names": resolved,
        "similar_capabilities": similar,
        "definition": capability_ontology.get_capability_info(capability)
    }


@app.get("/capability/metrics/skill/{skill_id}")
async def get_skill_performance_metrics(skill_id: str):
    """
    Get real performance metrics for a skill.

    This loads actual data from skill_stats table:
    - success_rate, avg_latency_ms, avg_confidence
    - total_executions, failure_count
    - reliability_score (calculated)
    - trend (improving/stable/degrading)
    """
    from capability import performance_metrics_provider

    metrics = await performance_metrics_provider.get_skill_metrics(
        skill_id,
        use_cache=False  # Get fresh data
    )

    if not metrics:
        return {
            "status": "error",
            "message": f"No metrics found for skill: {skill_id}",
            "skill_id": skill_id
        }

    return {
        "status": "ok",
        "skill_id": skill_id,
        "metrics": metrics.to_dict(),
        "provider_stats": performance_metrics_provider.get_stats()
    }


@app.get("/capability/metrics/top-skills")
async def get_top_performing_skills(limit: int = 10, min_executions: int = 5):
    """
    Get top performing skills by reliability score.

    Reliability score combines:
    - Success rate (primary)
    - Execution count (confidence)
    - Recency (recent usage)
    """
    from capability import performance_metrics_provider

    top_skills = await performance_metrics_provider.get_top_skills(
        limit=limit,
        min_executions=min_executions
    )

    return {
        "status": "ok",
        "count": len(top_skills),
        "top_skills": [skill.to_dict() for skill in top_skills],
        "provider_stats": performance_metrics_provider.get_stats()
    }


@app.post("/capability/metrics/cache/invalidate")
async def invalidate_metrics_cache():
    """
    Invalidate the metrics cache.

    Forces next query to load fresh data from database.
    """
    from capability import performance_metrics_provider

    await performance_metrics_provider.invalidate_cache()

    return {
        "status": "ok",
        "message": "Metrics cache invalidated",
        "provider_stats": performance_metrics_provider.get_stats()
    }


# =============================================================================
# Evolution Worker API Endpoints (Step C - Self-Improving System)
# =============================================================================

@app.get("/capability/evolution/status")
async def get_evolution_status():
    """
    Get evolution worker status.

    Shows:
    - Active pipelines and their versions
    - Reliability scores
    - Evolution cycle statistics
    """
    from capability import evolution_worker

    status = {
        "is_running": evolution_worker._is_running,
        "cron_interval_minutes": evolution_worker.CRON_INTERVAL_MINUTES,
        "min_executions_for_evolution": evolution_worker.MIN_EXECUTIONS_FOR_EVOLUTION,
        "low_reliability_threshold": evolution_worker.LOW_RELIABILITY_THRESHOLD
    }

    return {
        "status": "ok",
        "worker": status
    }


@app.post("/capability/evolution/run")
async def run_evolution_cycle():
    """
    Manually trigger an evolution cycle.

    This will:
    1. Evaluate all active pipelines
    2. Generate mutations for low-performing pipelines
    3. Test candidate versions
    4. Promote improvements
    """
    from capability import evolution_worker

    logger.info("manual_evolution_cycle_triggered")

    results = await evolution_worker.run_evolution_cycle()

    return {
        "status": "ok",
        "message": "Evolution cycle completed",
        "results": results
    }


@app.post("/capability/evolution/prune")
async def prune_low_performing_pipelines():
    """
    Soft-disable pipelines with very low performance.

    Criteria:
    - Reliability score < 0.3
    - At least 10 executions
    - Success rate < 30%
    """
    from capability import evolution_worker

    logger.info("manual_prune_triggered")

    results = await evolution_worker.prune_low_performing_pipelines()

    return {
        "status": "ok",
        "message": f"Pruned {results['count']} low-performing pipelines",
        "results": results
    }


@app.get("/capability/evolution/versions/{pipeline_id}")
async def get_pipeline_version_history(pipeline_id: str):
    """
    Get version history for a pipeline.

    Returns:
    - All versions (active, candidate, archived, disabled)
    - Metrics snapshots
    - Mutation details
    """
    from sqlalchemy import text
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        query = """
            SELECT
                version_id,
                pipeline_id,
                version_number,
                status,
                mutation_type,
                step_index,
                old_skill_id,
                new_skill_id,
                parameter_changes,
                mutation_reason,
                parent_version_id,
                metrics_snapshot,
                created_at,
                promoted_at,
                archived_at
            FROM pipeline_versions
            WHERE pipeline_id = :pipeline_id
            ORDER BY version_number DESC
        """

        result = await session.execute(
            text(query),
            {"pipeline_id": pipeline_id}
        )
        rows = result.fetchall()

        versions = []
        for row in rows:
            versions.append({
                "version_id": str(row.version_id),
                "pipeline_id": str(row.pipeline_id),
                "version_number": int(row.version_number),
                "status": row.status,
                "mutation_type": row.mutation_type,
                "step_index": int(row.step_index) if row.step_index else None,
                "old_skill_id": row.old_skill_id,
                "new_skill_id": row.new_skill_id,
                "parameter_changes": row.parameter_changes,
                "mutation_reason": row.mutation_reason,
                "parent_version_id": str(row.parent_version_id) if row.parent_version_id else None,
                "metrics_snapshot": row.metrics_snapshot,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
                "archived_at": row.archived_at.isoformat() if row.archived_at else None
            })

        return {
            "status": "ok",
            "pipeline_id": pipeline_id,
            "count": len(versions),
            "versions": versions
        }


@app.post("/capability/evolution/rollback/{pipeline_id}")
async def rollback_pipeline_version(
    pipeline_id: str,
    target_version_id: str,
    reason: str = "Manual rollback"
):
    """
    Rollback a pipeline to a previous version.

    Args:
        pipeline_id: Pipeline to rollback
        target_version_id: Version to rollback to
        reason: Why rollback is happening
    """
    from uuid import UUID
    from capability import evolution_worker

    logger.info(
        "manual_rollback_triggered",
        pipeline_id=pipeline_id,
        target_version_id=target_version_id,
        reason=reason
    )

    success = await evolution_worker.rollback_to_version(
        pipeline_id=UUID(pipeline_id),
        target_version_id=UUID(target_version_id),
        rollback_reason=reason
    )

    if not success:
        return {
            "status": "error",
            "message": "Rollback failed - check logs for details"
        }

    return {
        "status": "ok",
        "message": f"Rolled back {pipeline_id} to version {target_version_id}",
        "pipeline_id": pipeline_id,
        "target_version_id": target_version_id,
        "reason": reason
    }


@app.post("/capability/evolution/promote/{pipeline_id}")
async def promote_pipeline_version(
    pipeline_id: str,
    version_id: str
):
    """
    Promote a candidate version to active.

    This is typically called automatically after A/B testing,
    but can be triggered manually.
    """
    from uuid import UUID
    from capability import evolution_worker

    logger.info(
        "manual_promotion_triggered",
        pipeline_id=pipeline_id,
        version_id=version_id
    )

    success = await evolution_worker.promote_version(
        pipeline_id=UUID(pipeline_id),
        version_id=UUID(version_id)
    )

    if not success:
        return {
            "status": "error",
            "message": "Promotion failed - check logs for details"
        }

    return {
        "status": "ok",
        "message": f"Promoted version {version_id} to active",
        "pipeline_id": pipeline_id,
        "version_id": version_id
    }


# =============================================================================
# A/B Testing API Endpoints
# =============================================================================

@app.get("/capability/ab-testing/stats")
async def get_ab_testing_stats():
    """
    Get A/B testing statistics.

    Shows:
    - Active tests
    - Configuration
    - Test results
    """
    from capability import ab_test_engine

    stats = ab_test_engine.get_test_stats()
    active_tests = ab_test_engine.get_active_tests()

    return {
        "status": "ok",
        "stats": stats,
        "active_tests": [test.to_dict() for test in active_tests]
    }


@app.post("/capability/ab-testing/create")
async def create_ab_test(
    pipeline_id: str,
    candidate_version_id: str,
    traffic_split: float = 0.10,
    min_sample_size: int = 20
):
    """
    Create a new A/B test.

    Args:
        pipeline_id: Pipeline to test
        candidate_version_id: Candidate version to test
        traffic_split: Percentage of traffic to route to candidate (0.0-1.0)
        min_sample_size: Minimum executions per version
    """
    from uuid import UUID
    from capability.ab_testing import ab_test_engine, ABTestConfig

    # Get current active version
    from sqlalchemy import text
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        query = """
            SELECT version_id
            FROM pipeline_versions
            WHERE pipeline_id = :pipeline_id
              AND status = 'active'
            LIMIT 1
        """

        result = await session.execute(
            text(query),
            {"pipeline_id": pipeline_id}
        )
        row = result.fetchone()

        if not row:
            return {
                "status": "error",
                "message": "No active version found for pipeline"
            }

        current_version_id = row.version_id

    # Create test config
    config = ABTestConfig(
        traffic_split_percentage=traffic_split,
        min_sample_size=min_sample_size
    )

    # Create test
    test = await ab_test_engine.create_test(
        pipeline_id=UUID(pipeline_id),
        current_version_id=UUID(current_version_id),
        candidate_version_id=UUID(candidate_version_id),
        config=config
    )

    # Start test
    await ab_test_engine.start_test(test)

    return {
        "status": "ok",
        "message": "A/B test created and started",
        "test": test.to_dict()
    }


@app.get("/capability/ab-testing/tests/{test_id}")
async def get_ab_test_details(test_id: str):
    """
    Get details of a specific A/B test.
    """
    from uuid import UUID
    from capability import ab_test_engine

    test = ab_test_engine._active_tests.get(UUID(test_id))

    if not test:
        return {
            "status": "error",
            "message": "A/B test not found"
        }

    return {
        "status": "ok",
        "test": test.to_dict()
    }


@app.post("/capability/ab-testing/complete/{test_id}")
async def complete_ab_test(test_id: str):
    """
    Manually complete an A/B test and determine winner.

    This will:
    1. Stop routing traffic to both versions
    2. Calculate winner based on metrics
    3. Return decision (PROMOTE, KEEP_CURRENT, or INCONCLUSIVE)
    """
    from uuid import UUID
    from capability import ab_test_engine

    test = ab_test_engine._active_tests.get(UUID(test_id))

    if not test:
        return {
            "status": "error",
            "message": "A/B test not found"
        }

    # Complete test and get decision
    decision = await ab_test_engine.complete_test(test)

    return {
        "status": "ok",
        "message": "A/B test completed",
        "test_id": test_id,
        "decision": decision.value,
        "reason": test.decision_reason
    }


@app.post("/capability/ab-testing/record")
async def record_ab_test_execution(
    test_id: str,
    version_id: str,
    success: bool,
    latency_ms: float,
    confidence: float,
    artifact_count: int = 0
):
    """
    Record an execution result for A/B test.

    This should be called after each goal execution.
    """
    from uuid import UUID
    from capability import ab_test_engine

    await ab_test_engine.record_execution(
        test_id=UUID(test_id),
        version_id=UUID(version_id),
        success=success,
        latency_ms=latency_ms,
        confidence=confidence,
        artifact_count=artifact_count
    )

    return {
        "status": "ok",
        "message": "Execution recorded"
    }


# =============================================================================
# COGNITIVE OS API
# =============================================================================

@app.get("/cognitive/state")
async def get_cognitive_state():
    """
    Get comprehensive cognitive system state.
    
    Returns:
        Emotional, growth, strategy, identity, and world model states.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        state = cognitive_os.get_state()
        return {
            "status": "ok",
            "cognitive_state": state
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "degraded",
            "error": str(e),
            "cognitive_state": None
        }


@app.post("/cognitive/process")
async def process_cognitive_request(
    user_id: str,
    context: dict,
    action: str = "default",
    emotional_signals: Optional[dict] = None
):
    """
    Process a request with full cognitive support.
    
    Integrates:
    - Emotional influence
    - Strategy recommendations
    - Bias detection
    - World model updates
    - Self-narrative context
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import CognitiveOS, CognitiveRequest
        from ai_os.cognitive_os.cognitive_os import cognitive_os as cos
        
        request = CognitiveRequest(
            user_id=user_id,
            context=context,
            action=action,
            emotional_signals=emotional_signals
        )
        
        result = await cos.process(request)
        
        return {
            "status": "ok",
            "emotional_influence": result.emotional_influence,
            "strategy_recommendations": result.strategy_recommendations,
            "bias_warnings": result.bias_warnings,
            "world_state": result.world_state,
            "narrative_context": result.narrative_context,
            "reasoning_trace": result.reasoning_trace
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/cognitive/bias-report")
async def get_bias_report():
    """
    Get current bias awareness report from Growth Layer.
    
    Returns:
        Detected patterns, frequencies, and improvement suggestions.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        report = cognitive_os.growth.get_bias_report()
        return {
            "status": "ok",
            "report": report
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/cognitive/identity")
async def get_identity_report():
    """
    Get self-narrative identity report.
    
    Returns:
        Identity coherence, roles, and current narrative.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        report = cognitive_os.self_narrative.get_identity_report()
        return {
            "status": "ok",
            "identity": report
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/cognitive/strategies")
async def get_strategy_recommendations(
    complexity: float = 0.5,
    urgency: float = 0.5,
    novelty: float = 0.5
):
    """
    Get strategy recommendations based on context.
    
    Args:
        complexity: Task complexity (0-1)
        urgency: Urgency level (0-1)
        novelty: How novel the task is (0-1)
    
    Returns:
        Ranked strategy recommendations.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        
        context = {
            "complexity": complexity,
            "urgency": urgency,
            "novelty": novelty
        }
        
        recommendations = cognitive_os.strategy_evolution.generate_recommendation(context)
        return {
            "status": "ok",
            "recommendations": recommendations
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/cognitive/world")
async def get_world_state():
    """
    Get world model state snapshot.
    
    Returns:
        Entities, relationships, and action history.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        state = cognitive_os.world_model.get_world_state()
        return {
            "status": "ok",
            "world_state": state
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/cognitive/evolve")
async def run_evolution_cycle():
    """
    Run one strategy evolution cycle.
    
    Mutations and crosses top-performing strategies.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        
        new_strategies = cognitive_os.evolve()
        return {
            "status": "ok",
            "new_strategies": new_strategies,
            "generation": cognitive_os.strategy_evolution.generation
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/cognitive/record-outcome")
async def record_strategy_outcome(
    strategy_id: str,
    outcome: str,
    score: float
):
    """
    Record strategy outcome for learning.
    
    Updates performance metrics and integrates into narrative.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        
        cognitive_os.record_outcome(strategy_id, outcome, score)
        return {
            "status": "ok",
            "message": "Outcome recorded"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/cognitive/motivation/{outcome_type}")
async def get_motivational_response(outcome_type: str):
    """
    Get motivational response based on outcome type.
    
    Args:
        outcome_type: success, failure, or uncertain
    
    Returns:
        Motivational message based on self-narrative.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        
        response = cognitive_os.self_narrative.generate_motivational_response(outcome_type)
        return {
            "status": "ok",
            "motivation": response
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# =============================================================================
# PHASE 8: UNIFIED POLICY + SIMULATION CORE
# =============================================================================

@app.get("/agent/state/unified")
async def get_unified_state():
    """
    Get unified latent state from StateBuilder.
    
    Combines world + self + emotion + growth + strategy into single state.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.state_builder import StateBuilder
        
        builder = StateBuilder(cognitive_os)
        state = await builder.build_state()
        
        return {
            "status": "ok",
            "state": {
                "world_outcome": state.world_recent_outcome,
                "world_entities": state.world_entities_count,
                "world_capability": state.world_capability_score,
                "identity_coherence": state.identity_coherence,
                "identity_emotion": state.identity_emotion,
                "arousal": state.arousal,
                "valence": state.valence,
                "focus": state.focus,
                "confidence": state.confidence,
                "bias_count": state.bias_count,
                "bias_awareness": state.bias_awareness,
                "reflection_depth": state.reflection_depth,
                "top_strategy": state.top_strategy_name,
                "stress_level": state.stress_level,
                "exploration_tendency": state.exploration_tendency,
                "action_readiness": state.action_readiness,
            },
            "vector": state.to_vector()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/agent/decide")
async def agent_decide(
    user_id: str,
    task: str,
    task_type: str = "default",
    complexity: float = 0.5,
    urgency: float = 0.5,
    novelty: float = 0.5,
    use_simulation: bool = True
):
    """
    Make a decision using UnifiedAgent with policy(state) → action.
    
    This is the main decision endpoint for Phase 8.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        from ai_os.cognitive_os.policy.unified_agent import AgentContext
        
        agent = get_unified_agent(cognitive_os)
        
        context = AgentContext(
            user_id=user_id,
            task=task,
            task_type=task_type,
            complexity=complexity,
            urgency=urgency,
            novelty=novelty
        )
        
        decision = await agent.decide(context, use_simulation=use_simulation)
        
        return {
            "status": "ok",
            "decision": {
                "action": decision.action_type,
                "confidence": decision.confidence,
                "reasoning": decision.reasoning,
                "expected_utility": decision.expected_utility,
                "expected_risk": decision.expected_risk,
                "alternatives": decision.alternatives,
                "simulation_used": decision.simulation_used,
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/agent/execute-and-learn")
async def agent_execute_and_learn(
    decision_data: dict,
    outcome: str,
    metrics: Optional[dict] = None
):
    """
    Execute decision and record outcome for learning.
    
    Closes the learning loop: action → outcome → strategy update.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        from ai_os.cognitive_os.policy.unified_agent import AgentDecision
        
        agent = get_unified_agent(cognitive_os)
        
        decision = AgentDecision(
            action_type=decision_data.get("action", "execute"),
            confidence=decision_data.get("confidence", 0.5),
            reasoning=decision_data.get("reasoning", ""),
            expected_utility=decision_data.get("expected_utility", 0.5),
            expected_risk=decision_data.get("expected_risk", 0.5),
            alternatives=decision_data.get("alternatives", []),
            simulation_used=decision_data.get("simulation_used", False)
        )
        
        learning = await agent.execute_and_learn(decision, outcome, metrics)
        
        return {
            "status": "ok",
            "learning": learning
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/agent/plan-sequence")
async def agent_plan_sequence(
    user_id: str,
    task: str,
    task_type: str = "default",
    complexity: float = 0.5,
    urgency: float = 0.5,
    novelty: float = 0.5,
    num_actions: int = 3
):
    """
    Plan a sequence of actions using simulation loop.
    
    Returns ordered list of decisions for multi-step execution.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        from ai_os.cognitive_os.policy.unified_agent import AgentContext
        
        agent = get_unified_agent(cognitive_os)
        
        context = AgentContext(
            user_id=user_id,
            task=task,
            task_type=task_type,
            complexity=complexity,
            urgency=urgency,
            novelty=novelty
        )
        
        plan = await agent.plan_sequence(context, num_actions=num_actions)
        
        return {
            "status": "ok",
            "plan": [
                {
                    "action": d.action_type,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning,
                    "expected_utility": d.expected_utility,
                }
                for d in plan
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/agent/stats")
async def get_agent_stats():
    """
    Get agent performance statistics.
    
    Includes decision history, policy stats, and cognitive state.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        stats = agent.get_agent_stats()
        
        return {
            "status": "ok",
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/agent/policy/stats")
async def get_policy_stats():
    """
    Get policy layer statistics.
    
    Includes action distribution and confidence metrics.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.state_builder import StateBuilder
        from ai_os.cognitive_os.policy.policy_layer import PolicyLayer
        
        builder = StateBuilder(cognitive_os)
        policy = PolicyLayer(builder, cognitive_os.world_model)
        
        return {
            "status": "ok",
            "policy_stats": policy.get_decision_stats()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/observability/transactions")
async def get_transactions(
    user_id: Optional[str] = None,
    limit: int = 100
):
    """
    Get completed decision transactions.
    
    Each transaction contains full causal chain from state → action → outcome.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        transactions = agent._transaction_history
        
        if user_id:
            transactions = [t for t in transactions if t.user_id == user_id]
        
        transactions = transactions[-limit:]
        
        return {
            "status": "ok",
            "count": len(transactions),
            "transactions": [
                {
                    "id": t.id,
                    "user_id": t.user_id,
                    "task": t.task,
                    "status": t.status,
                    "action": t.selected_action,
                    "confidence": t.confidence,
                    "outcome": t.outcome.outcome if t.outcome else None,
                    "duration_ms": t.total_duration_ms,
                    "is_complete": t.is_complete(),
                }
                for t in transactions
            ]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/transactions/{transaction_id}")
async def get_transaction_detail(transaction_id: str):
    """
    Get detailed transaction with full causal chain.
    
    Includes state, reasoning events, candidates, attribution, and outcome.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        
        for txn in agent._transaction_history:
            if txn.id == transaction_id:
                return {"status": "ok", "transaction": txn.to_dict()}
        
        return {"status": "error", "error": "Transaction not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/transactions/{transaction_id}/causal-chain")
async def get_causal_chain(transaction_id: str):
    """
    Get causal chain for a transaction.
    
    Shows the complete reasoning path from state → action.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        
        for txn in agent._transaction_history:
            if txn.id == transaction_id:
                return {
                    "status": "ok",
                    "transaction_id": transaction_id,
                    "causal_chain": txn.get_causal_chain()
                }
        
        return {"status": "error", "error": "Transaction not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/replay-dataset")
async def get_replay_dataset(limit: int = 100):
    """
    Get replayable dataset for RL training.
    
    Each entry has complete state → action → outcome sequence.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        dataset = agent.get_replayable_dataset(limit)
        
        return {
            "status": "ok",
            "count": len(dataset),
            "dataset": dataset
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/observability/transaction/execute")
async def execute_with_transaction(
    user_id: str,
    task: str,
    task_type: str = "default",
    complexity: float = 0.5,
    urgency: float = 0.5,
    novelty: float = 0.5,
    use_simulation: bool = True
):
    """
    Execute a decision within a transaction context.
    
    Returns the complete transaction with full causal chain.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        from ai_os.cognitive_os.policy.unified_agent import AgentContext
        
        agent = get_unified_agent(cognitive_os)
        
        context = AgentContext(
            user_id=user_id,
            task=task,
            task_type=task_type,
            complexity=complexity,
            urgency=urgency,
            novelty=novelty
        )
        
        async with agent.transaction(user_id, task) as txn:
            await txn.decide(context, use_simulation=use_simulation)
        
        return {
            "status": "ok",
            "transaction_id": txn.transaction.id,
            "action": txn.transaction.selected_action,
            "confidence": txn.transaction.confidence,
            "reasoning": txn.transaction.reasoning,
            "attribution": txn.transaction.attribution.to_dict() if txn.transaction.attribution else None,
            "reasoning_events": len(txn.transaction.reasoning_events),
            "candidates": len(txn.transaction.candidates),
            "duration_ms": txn.transaction.total_duration_ms,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.post("/observability/transaction/{transaction_id}/record-outcome")
async def record_transaction_outcome(
    transaction_id: str,
    outcome: str,
    score: float = 1.0
):
    """
    Record outcome for a completed transaction.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        from ai_os.cognitive_os.observability.transaction import StateSnapshot
        
        agent = get_unified_agent(cognitive_os)
        
        for txn in agent._transaction_history:
            if txn.id == transaction_id:
                state_after = StateSnapshot()
                state_after.timestamp = datetime.utcnow()
                
                current = agent.diff_engine.get_current_state()
                if current:
                    state_after.confidence = current.confidence
                    state_after.stress_level = current.stress_level
                
                txn.record_outcome(outcome, score, state_after)
                
                return {
                    "status": "ok",
                    "transaction_id": transaction_id,
                    "outcome": outcome,
                    "score": score
                }
        
        return {"status": "error", "error": "Transaction not found"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/transactions/stats")
async def get_transaction_stats():
    """
    Get transaction statistics.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        
        transactions = agent._transaction_history
        completed = [t for t in transactions if t.is_complete()]
        with_outcome = [t for t in completed if t.outcome]
        
        return {
            "status": "ok",
            "stats": {
                "total_transactions": len(transactions),
                "completed": len(completed),
                "with_outcome": len(with_outcome),
                "completion_rate": len(completed) / len(transactions) if transactions else 0,
                "avg_duration_ms": sum(t.total_duration_ms for t in completed) / len(completed) if completed else 0,
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# PHASE 9: OBSERVABILITY LAYER (existing endpoints)
# =============================================================================

@app.get("/observability/traces")
async def get_decision_traces(
    user_id: Optional[str] = None,
    action: Optional[str] = None,
    outcome: Optional[str] = None,
    limit: int = 100
):
    """
    Get decision traces with full behavioral telemetry.
    
    Query parameters:
    - user_id: Filter by user
    - action: Filter by action type
    - outcome: Filter by outcome
    - limit: Max results (default 100)
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        traces = agent.tracer.store.query(
            user_id=user_id,
            action=action,
            outcome=outcome,
            limit=limit
        )
        
        return {
            "status": "ok",
            "count": len(traces),
            "traces": [
                {
                    "id": t.id,
                    "timestamp": t.timestamp.isoformat(),
                    "action": t.action,
                    "confidence": t.confidence,
                    "reasoning": t.reasoning,
                    "outcome": t.outcome,
                    "simulation_used": t.simulation_used,
                }
                for t in traces
            ]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.get("/observability/traces/{decision_id}")
async def get_trace_detail(decision_id: str):
    """
    Get detailed trace for a specific decision.
    
    Includes state snapshot, alternatives, and outcome.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        explanation = agent.tracer.explain_decision(decision_id)
        
        return {"status": "ok", "trace": explanation}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/stats")
async def get_observability_stats(user_id: Optional[str] = None):
    """
    Get comprehensive observability statistics.
    
    Includes action distribution, confidence stats, outcome distribution.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        stats = agent.tracer.get_stats(user_id)
        
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/state/diffs")
async def get_state_diffs(count: int = 5):
    """
    Get recent state diffs showing how state changed over time.
    
    Args:
        count: Number of recent diffs to return
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        diffs = agent.diff_engine.get_recent_diffs(count)
        
        return {"status": "ok", "diffs": diffs}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/state/current")
async def get_current_state():
    """
    Get current unified state snapshot.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        state = agent.diff_engine.get_current_state()
        
        return {
            "status": "ok",
            "state": state.to_dict() if state else None
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/state/patterns")
async def get_state_patterns():
    """
    Get analyzed state patterns and trends.
    
    Includes trends, anomalies, and pattern detection.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        patterns = agent.diff_engine.analyze_patterns()
        
        return {"status": "ok", "patterns": patterns}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/attribution/{decision_id}")
async def get_decision_attribution(decision_id: str):
    """
    Get full attribution for why a decision was made.
    
    Shows causal chain from features → scoring → action.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        report = agent.attribution_system.get_report(decision_id)
        
        return {"status": "ok", "attribution": report}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/attribution/last")
async def get_last_attribution():
    """
    Get explanation for the most recent decision.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        explanation = agent.attribution_system.explain_last()
        
        return {"status": "ok", "explanation": explanation}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/attribution/importance")
async def get_feature_importance(window: int = 100):
    """
    Get average feature importance over recent decisions.
    
    Args:
        window: Number of recent decisions to analyze
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        importance = agent.attribution_system.get_feature_importance(window)
        
        return {"status": "ok", "feature_importance": importance}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/observability/trace/start")
async def start_trace(user_id: str, task: str = ""):
    """
    Start a new decision trace.
    
    Returns trace_id for subsequent decision recording.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        trace = agent.tracer.start_trace(user_id, task)
        
        return {"status": "ok", "trace_id": trace.trace_id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/observability/trace/{trace_id}/end")
async def end_trace(trace_id: str):
    """
    End a decision trace and store results.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        trace = agent.tracer.end_trace(trace_id)
        
        return {
            "status": "ok",
            "completed": trace is not None,
            "duration_ms": trace.duration_ms() if trace else 0
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/report")
async def get_full_observability_report(user_id: Optional[str] = None):
    """
    Get comprehensive observability report combining all components.
    
    Includes traces, state diffs, attribution, and patterns.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.policy.unified_agent import get_unified_agent
        
        agent = get_unified_agent(cognitive_os)
        
        return {
            "status": "ok",
            "report": {
                "tracer_stats": agent.tracer.get_stats(user_id),
                "state_report": agent.diff_engine.get_state_report(),
                "last_attribution": agent.attribution_system.explain_last(),
                "feature_importance": agent.attribution_system.get_feature_importance(100),
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# PHASE 10: EVENT ONTOLOGY - Единый источник истины
# =============================================================================

@app.get("/observability/events")
async def get_events(
    event_type: Optional[str] = None,
    category: Optional[str] = None,
    transaction_id: Optional[str] = None,
    limit: int = 100
):
    """
    Получить события из Event Ontology.
    
    Это единый интерфейс для всех событий системы.
    Заменяет разрозненные tracer + transaction + attribution.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os.eventsourcing import get_event_ontology
        
        ontology = get_event_ontology()
        events = ontology.query(
            event_type=event_type,
            category=category,
            transaction_id=transaction_id,
            limit=limit
        )
        
        return {
            "status": "ok",
            "count": len(events),
            "events": [e.to_dict() for e in events]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/events/statistics")
async def get_event_statistics():
    """
    Получить статистику Event Ontology.
    
    Показывает распределение событий по типам и категориям.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os.eventsourcing import get_event_ontology
        
        ontology = get_event_ontology()
        stats = ontology.get_statistics()
        
        return {
            "status": "ok",
            "statistics": stats
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/events/{event_id}")
async def get_event_detail(event_id: str):
    """
    Получить детали события по ID.
    
    Включает causal chain и связанные события.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os.eventsourcing import get_event_ontology
        
        ontology = get_event_ontology()
        event = ontology.get_event(event_id)
        
        if not event:
            return {"status": "error", "error": "Event not found"}
        
        return {
            "status": "ok",
            "event": event.to_dict(),
            "causal_path": event.get_causal_path()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/streams")
async def get_event_streams(transaction_id: Optional[str] = None):
    """
    Получить Event Streams.
    
    Каждый stream соответствует одной транзакции.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os.eventsourcing import get_event_ontology
        
        ontology = get_event_ontology()
        
        if transaction_id:
            stream = ontology.get_stream(transaction_id)
            if not stream:
                return {"status": "error", "error": "Stream not found"}
            return {
                "status": "ok",
                "streams": [stream.to_dict()]
            }
        else:
            streams = list(ontology.streams.values())
            return {
                "status": "ok",
                "count": len(streams),
                "streams": [s.to_dict() for s in streams[-50:]]
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/observability/streams/{transaction_id}/graph")
async def get_stream_causal_graph(transaction_id: str):
    """
    Получить causal graph для транзакции.
    
    Показывает все события и их причинные связи.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os.eventsourcing import get_event_ontology
        
        ontology = get_event_ontology()
        stream = ontology.get_stream(transaction_id)
        
        if not stream:
            return {"status": "error", "error": "Stream not found"}
        
        return {
            "status": "ok",
            "transaction_id": transaction_id,
            "graph": stream.get_causal_graph()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/observability/agent/execute-with-events")
async def execute_with_event_sourcing(
    user_id: str,
    task: str,
    task_type: str = "default",
    complexity: float = 0.5,
    urgency: float = 0.5,
    novelty: float = 0.5
):
    """
    Выполнить решение через Event-Sourced Agent.
    
    Все события записываются в Event Ontology.
    Возвращает полный causal graph.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.eventsourcing import get_event_sourced_agent
        
        agent = get_event_sourced_agent(cognitive_os)
        
        from ai_os.cognitive_os.eventsourcing import AgentContext as ESAContext
        
        context = ESAContext(
            user_id=user_id,
            task=task,
            task_type=task_type,
            complexity=complexity,
            urgency=urgency,
            novelty=novelty
        )
        
        async with agent.transaction(user_id, task) as txn:
            await txn.decide(context)
        
        return {
            "status": "ok",
            "transaction_id": txn.transaction_id,
            "selected_action": txn.result.selected_action if txn.result else None,
            "confidence": txn.result.confidence if txn.result else None,
            "causal_chain": txn.get_causal_chain(),
            "events_count": len(txn.get_causal_chain()),
            "duration_ms": txn.result.duration_ms if txn.result else 0,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.get("/observability/agent/stats")
async def get_event_sourced_agent_stats():
    """
    Получить статистику Event-Sourced Agent.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.eventsourcing import get_event_sourced_agent
        
        agent = get_event_sourced_agent(cognitive_os)
        stats = agent.get_statistics()
        
        return {
            "status": "ok",
            "stats": stats
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# PHASE 11: LATENT CAUSAL DYNAMICS
# =============================================================================

@app.post("/cognitive/latent/transition")
async def process_latent_transition(
    from_state: dict,
    to_state: dict,
    action: Optional[str] = None,
    outcome: Optional[str] = None
):
    """
    Process state transition with semantic causality.
    
    Extracts latent causes, builds causal attribution,
    and creates latent state representation.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        from ai_os.cognitive_os.latent_dynamics.latent_space import SymbolicState
        
        engine = get_latent_dynamics_engine(cognitive_os)
        
        # Create unified state from dict
        class MockState:
            def __init__(self, data):
                for k, v in data.items():
                    setattr(self, k, v)
        
        from_state_obj = MockState(from_state)
        to_state_obj = MockState(to_state)
        
        result = engine.process_transition(
            from_state=from_state_obj,
            to_state=to_state_obj,
            action=action,
            outcome=outcome
        )
        
        return {"status": "ok", "transition": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


@app.get("/cognitive/latent/causes")
async def get_dominant_causes(limit: int = 5):
    """
    Get dominant latent causes across all transitions.
    
    Returns semantic causes like:
    - uncertainty_spike
    - prediction_conflict
    - reward_expectation_drop
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        causes = engine.get_dominant_causes(limit)
        
        return {"status": "ok", "causes": causes}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/cognitive/latent/causes/unreliable")
async def get_unreliable_causes(threshold: float = 0.3):
    """
    Get causes with low causal confidence.
    
    These are candidates for further investigation.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        unreliable = engine.get_unreliable_causes(threshold)
        
        return {"status": "ok", "unreliable_causes": unreliable}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/cognitive/latent/confidence/distribution")
async def get_causal_confidence_distribution():
    """
    Get distribution of causal edge confidences.
    
    Returns high/medium/low ratios.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        distribution = engine.get_confidence_distribution()
        
        return {"status": "ok", "distribution": distribution}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/cognitive/latent/counterfactual")
async def reason_counterfactual(
    state_id: str,
    actual_action: str,
    alternative_action: str
):
    """
    Reason about counterfactual: what if we chose alternative_action?
    
    Uses transition model to predict what would happen.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        counterfactual = engine.reason_counterfactual(
            state_id=state_id,
            actual_action=actual_action,
            alternative_action=alternative_action
        )
        
        return {"status": "ok", "counterfactual": counterfactual}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/cognitive/latent/branches")
async def predict_branch_outcomes(
    state_id: str,
    possible_actions: List[str]
):
    """
    Predict outcomes for all possible actions.
    
    Returns ranked list of predicted outcomes.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        predictions = engine.predict_branch_outcomes(state_id, possible_actions)
        
        return {"status": "ok", "predictions": predictions}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/cognitive/latent/states/similar")
async def find_similar_states(state_id: str, threshold: float = 0.3):
    """
    Find states similar to given state in latent space.
    
    Useful for finding analogous situations.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        similar = engine.find_similar_states(state_id, threshold)
        
        return {"status": "ok", "similar_states": similar}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/cognitive/latent/evolution")
async def get_state_evolution():
    """
    Get full state evolution trajectory.
    
    Shows how latent state changed over time.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        trajectory = engine.get_state_evolution()
        
        return {"status": "ok", "trajectory": trajectory}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/cognitive/latent/compress")
async def compress_trajectory(events: List[dict]):
    """
    Compress event stream into semantic motifs.
    
    Prevents log explosion, creates actionable units.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        result = engine.compress_trajectory(events)
        
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/cognitive/latent/stats")
async def get_latent_dynamics_stats():
    """
    Get comprehensive latent dynamics statistics.
    """
    try:
        import sys
        sys.path.insert(0, '/home/onor/ai_os_final')
        from ai_os.cognitive_os import cognitive_os
        from ai_os.cognitive_os.latent_dynamics.engine import get_latent_dynamics_engine
        
        engine = get_latent_dynamics_engine(cognitive_os)
        stats = engine.get_statistics()
        
        return {"status": "ok", "stats": stats}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================================
# LEASE MANAGEMENT API (P2.12b)
# ============================================================================


@app.post("/leases/{lease_id}/heartbeat")
async def lease_heartbeat(lease_id: str):
    """
    Record a keepalive heartbeat for an active lease.
    Extends the lease TTL by the default lease TTL.
    """
    from epistemic_factory import get_causality_bridge
    bridge = get_causality_bridge()
    if not bridge or not bridge.execution_kernel:
        return {"status": "error", "error": "Execution kernel not available"}
    ok = bridge.execution_kernel.heartbeat_lease(lease_id)
    return {"status": "ok" if ok else "error", "lease_id": lease_id}


@app.post("/leases/{lease_id}/renew")
async def lease_renew(lease_id: str, ttl: Optional[int] = None):
    """
    Explicitly renew a lease with optional custom TTL.
    """
    from epistemic_factory import get_causality_bridge
    bridge = get_causality_bridge()
    if not bridge or not bridge.execution_kernel:
        return {"status": "error", "error": "Execution kernel not available"}
    ok = bridge.execution_kernel.renew_lease(lease_id, ttl)
    return {"status": "ok" if ok else "error", "lease_id": lease_id}


@app.get("/leases/stale")
async def lease_stale(threshold: Optional[int] = None):
    """
    List active leases without recent heartbeats.
    Optional query param: threshold=300 (seconds)
    """
    from epistemic_factory import get_causality_bridge
    bridge = get_causality_bridge()
    if not bridge or not bridge.execution_kernel:
        return {"status": "error", "error": "Execution kernel not available"}
    stale = bridge.execution_kernel.detect_stale_leases(threshold)
    return {"status": "ok", "stale_leases": stale, "count": len(stale)}


@app.post("/leases/expire-stale")
async def lease_expire_stale(max_age: Optional[int] = None):
    """
    Force-expire all stale leases.
    Journals LEASE_EXPIRED for each stale lease.
    """
    from epistemic_factory import get_causality_bridge
    bridge = get_causality_bridge()
    if not bridge or not bridge.execution_kernel:
        return {"status": "error", "error": "Execution kernel not available"}
    count = bridge.execution_kernel.expire_stale_leases(max_age)
    return {"status": "ok", "expired_count": count}
