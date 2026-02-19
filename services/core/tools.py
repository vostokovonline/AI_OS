import os, httpx, redis, asyncio, uuid, time
from langchain_core.tools import tool
from database import AsyncSessionLocal
from models import Goal, SystemPrompt, UserFact
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from skill_manager import create_skill
from tools_memory import *
from tools_external import github_action, fast_search, send_email, deep_web_search, browse_and_extract, generate_website, deploy_website
from telemetry import log_action

# Используем redis для блокировок ресурсов (Compute/Browser)
redis_client = redis.from_url(os.getenv("CELERY_BROKER_URL"))
LIMITS = {"compute": 3, "browser": 2}
OPENCODE_URL = os.getenv("OPENCODE_URL", "http://opencode:8002")
WEBSURFER_URL = os.getenv("WEBSURFER_URL", "http://websurfer:8003")
TELEGRAM_URL = os.getenv("TELEGRAM_URL", "http://telegram:8004")
MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8001")

async def acquire_lock(rtype, timeout=60):
    # Простейший семафор через Redis
    if redis_client.get("STATUS_CRITICAL"): return False, "Low RAM"
    if rtype == "compute" and redis_client.get("STATUS_HEAVY_LOAD"): return False, "High CPU"
    key = f"semaphore:{rtype}"
    limit = LIMITS.get(rtype, 1)
    start = asyncio.get_event_loop().time()
    while True:
        current = redis_client.incr(key)
        if current <= limit:
            redis_client.expire(key, 300) # Авто-сброс через 5 минут
            return True, ""
        redis_client.decr(key)
        if asyncio.get_event_loop().time() - start > timeout: return False, "Timeout"
        await asyncio.sleep(1)

async def release_lock(rtype):
    redis_client.decr(f"semaphore:{rtype}")

@tool
async def run_python_code(code: str, session_id: str = "default"):
    """
    Executes Python code in a persistent Jupyter environment.
    Use this for calculations, data analysis, or file manipulation.
    """
    allowed, msg = await acquire_lock("compute")
    if not allowed: return f"System Busy: {msg}"
    start = time.time()
    status = "success"
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{OPENCODE_URL}/run", json={"session_id": session_id, "code": code}, timeout=60)
            data = res.json()
            out = f"STDOUT:\n{data.get('stdout')}" if data["status"] == "success" else f"ERROR:\n{data.get('stderr')}"
            if data["status"] != "success": status = "error"
            return out
    except Exception as e:
        status = "crash"
        return str(e)
    finally:
        await release_lock("compute")
        try:
             await log_action(session_id, "CODER", "run_python", code[:100], "", status, start)
        except: pass

@tool
async def browse_web(url: str):
    """Visits a website and extracts content."""
    allowed, msg = await acquire_lock("browser")
    if not allowed: return f"System Busy: {msg}"
    start = time.time()
    status = "success"
    try:
        async with httpx.AsyncClient() as client:
            # Исправлен URL для Websurfer (эндпоинт /visit)
            res = await client.post(f"{WEBSURFER_URL}/visit", json={"url": url}, timeout=60)
            data = res.json()
            if data.get("status") == "success":
                 out = f"Title: {data.get('title')}\nContent:\n{data.get('content', '')[:4000]}"
            else:
                 out = f"Error: {data.get('detail')}"
                 status = "error"
            return out
    except Exception as e:
        status = "crash"
        return str(e)
    finally:
        await release_lock("browser")
        try:
             await log_action("unknown", "RESEARCHER", "browse_web", url, "", status, start)
        except: pass

@tool
async def ask_web_llm(provider: str, prompt: str):
    """Chat with Web LLMs (ChatGPT). Provider: 'chatgpt'."""
    allowed, msg = await acquire_lock("browser")
    if not allowed: return f"System Busy: {msg}"
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            res = await client.post(f"{WEBSURFER_URL}/chat_web", json={"provider": provider, "prompt": prompt})
            d = res.json()
            if d["status"]=="success": return f"WEB-LLM RESPONSE:\n{d['content']}"
            return f"WEB ERROR: {d.get('detail')}"
    except Exception as e: return str(e)
    finally: await release_lock("browser")

@tool
async def send_notification(message: str):
    """Sends a notification to Telegram."""
    async with httpx.AsyncClient() as client:
        try: await client.post(f"{TELEGRAM_URL}/notify", json={"message": message})
        except: pass
    return "Sent."

@tool
async def generate_image(prompt: str):
    """Generates an image via AI."""
    safe = prompt.replace(" ", "%20")
    return f"Image: https://image.pollinations.ai/prompt/{safe}"

@tool
async def define_new_skill(name: str, python_code: str, description: str, usage_example: str):
    """Creates a new tool."""
    return await create_skill(name, python_code, description, usage_example)

@tool
async def search_skills(query: str):
    """Searches existing skills."""
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{MEMORY_URL}/search", json={"text": query, "top_k": 3})
        found = [m['text'] for m in res.json().get('matches', []) if "Skill:" in m['text']]
        return "Found:\n" + "\n".join(found) if found else "No skills found."

@tool
async def create_goal(
    title: str,
    description: str = "",
    parent_id: str = None,
    goal_type: str = "achievable",
    depth_level: int = 0,
    is_atomic: bool = False,
    domains: list = None,
    completion_criteria: dict = None
):
    """
    Creates a new goal with Goal Ontology v3.0 support.

    Args:
        title: Goal title
        description: Goal description
        parent_id: Optional parent goal ID
        goal_type: Goal type - 'achievable', 'continuous', 'directional', 'exploratory', or 'meta'
        depth_level: Depth level - 0 (mission), 1 (strategic), 2 (operational), 3 (atomic)
        is_atomic: Whether the goal is atomic (can be executed directly)
        domains: List of domains this goal belongs to
        completion_criteria: JSON criteria for completion
    """
    async with AsyncSessionLocal() as db:
        # 🔍 DEDUPLICATION: Check if goal with same title already exists
        from sqlalchemy import func
        stmt = select(Goal).where(
            func.lower(Goal.title) == func.lower(title.strip())
        )
        existing = (await db.execute(stmt)).scalar_one_or_none()

        if existing:
            # Return existing goal instead of creating duplicate
            return f"♻️  Goal already exists: '{existing.title}' (ID: {existing.id}, Status: {existing.status}, Progress: {existing.progress}). Skipping duplicate creation."

        pid = uuid.UUID(parent_id) if parent_id else None

        # Create goal with v3.0 fields
        g = Goal(
            title=title,
            description=description,
            parent_id=pid,
            goal_type=goal_type,
            depth_level=depth_level,
            is_atomic=is_atomic,
            domains=domains or [],
            completion_criteria=completion_criteria or {}
        )
        db.add(g)
        await db.commit()
        await db.refresh(g)

        return f"✅ Goal '{g.title}' created (ID: {g.id}, Type: {g.goal_type}, Level: L{g.depth_level}, Atomic: {g.is_atomic})"

@tool
async def get_goal_tree(root_id: str = None):
    """Gets the goal hierarchy."""
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.parent_id == None).where(Goal.status != "completed")
        goals = (await db.execute(stmt)).scalars().all()
        if not goals: return "No active goals."
        out = "STRATEGY TREE:\n"
        for g in goals: out += f"- {g.title} ({g.status}, ID: {g.id})\n"
        return out

@tool
async def update_goal(goal_id: str, status: str, progress: float):
    """Updates goal status."""
    async with AsyncSessionLocal() as db:
        try:
             uid = uuid.UUID(goal_id)
        except:
             return "Invalid UUID format."
             
        stmt = select(Goal).where(Goal.id == uid)
        g = (await db.execute(stmt)).scalar_one_or_none()
        if g:
            g.status = status
            g.progress = progress
            await db.commit()
            return "Updated."
        return "Not found."

@tool
async def update_system_prompt(agent_role: str, new_prompt: str):
    """Updates system DNA."""
    async with AsyncSessionLocal() as db:
        stmt = insert(SystemPrompt).values(key=agent_role.upper(), content=new_prompt).on_conflict_do_update(index_elements=['key'], set_=dict(content=new_prompt))
        await db.execute(stmt)
        await db.commit()
    return f"DNA Updated for {agent_role}."

@tool
async def add_user_fact(category: str, content: str):
    """Saves user fact."""
    async with AsyncSessionLocal() as db:
        db.add(UserFact(category=category, content=content))
        await db.commit()
    return "User Profile Updated."

AGENT_TOOLS = [
    run_python_code, browse_web, ask_web_llm, send_notification, generate_image, define_new_skill, search_skills,
    create_goal, get_goal_tree, update_goal, update_system_prompt, add_user_fact,
    save_memory, learn_fact, recall, analyze_goal_knowledge_needs, get_random_concepts_for_synthesis,
    register_idea, log_my_state, add_contact, get_social_insights,
    github_action, fast_search, send_email, deep_web_search, browse_and_extract, generate_website, deploy_website
]
