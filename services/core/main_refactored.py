"""
AI-OS Core API - Refactored Main Application
Modular structure with proper middleware and security
"""
import asyncio
import time
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Database
from database import engine, Base, get_db, close_db_connections, AsyncSessionLocal

# Models & Schemas
from models import Message, ChatSession
from schemas import MessageCreate, MessageResponse, ResumeRequest, EventRequest

# Services
from tasks import run_chat_task, run_resume_task, run_cron_task
from scheduler import start_scheduler
from agent_graph import app_graph
from dna_manager import bootstrap_dna

# API Routers
from api.endpoints import goals, artifacts, skills, llm, graph
from api.middleware import RateLimitMiddleware, LoggingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("⏳ Starting AI-OS Core...")
    
    # Wait for database
    await wait_for_db()
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Bootstrap DNA
    await bootstrap_dna()
    
    # Start scheduler
    start_scheduler()
    
    print("🚀 AI-OS SYSTEM ONLINE")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down AI-OS Core...")
    await close_db_connections()
    print("✅ Shutdown complete")


async def wait_for_db():
    """Wait for database connection"""
    print("⏳ Connecting to Database...")
    max_retries = 30
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            from sqlalchemy import text
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            print("✅ Database Connected!")
            return
        except Exception as e:
            retry_count += 1
            print(f"  Retry {retry_count}/{max_retries}: {e}")
            await asyncio.sleep(2)
    
    raise Exception("Failed to connect to database after maximum retries")


# Create FastAPI app
app = FastAPI(
    title="AI-OS Core API",
    description="Intelligent goal-execution system powered by AI agents",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

# Include routers
app.include_router(goals.router, prefix="/api/v1")
app.include_router(artifacts.router, prefix="/api/v1")
app.include_router(skills.router, prefix="/api/v1")
app.include_router(llm.router, prefix="/api/v1")
app.include_router(graph.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0"
    }


@app.post("/chat", response_model=MessageResponse)
async def chat(req: MessageCreate, db=Depends(get_db)):
    """Send message to AI-OS chat"""
    import uuid
    
    sid = req.session_id or str(uuid.uuid4())
    
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ChatSession).where(ChatSession.id == sid)
        )
        if not result.scalar_one_or_none():
            session.add(ChatSession(id=sid))
            await session.commit()
        
        session.add(Message(session_id=sid, role="user", content=req.content))
        await session.commit()
    
    # Queue task
    run_chat_task.delay(sid, req.content, req.image_url)
    
    return MessageResponse(
        role="system",
        content="⏳ Processing...",
        created_at=datetime.utcnow()
    )


@app.post("/resume")
async def resume(req: ResumeRequest):
    """Resume chat session"""
    run_resume_task.delay(req.session_id)
    return {"status": "resumed"}


@app.post("/event")
async def handle_event(evt: EventRequest):
    """Handle external events"""
    import uuid
    
    sid = f"event_{evt.source}_{uuid.uuid4().hex[:6]}"
    run_cron_task.delay(sid, f"EVENT: {evt.source}\nDATA: {evt.payload}")
    return {"status": "processing"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
