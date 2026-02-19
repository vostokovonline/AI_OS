import httpx, os
from langchain_core.tools import tool
from database import AsyncSessionLocal
from models import Goal
from sqlalchemy import select

MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8001")

@tool
async def save_memory(text: str, category: str = "general"):
    """Saves episodic memory."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{MEMORY_URL}/remember", json={"text": text, "type": "episodic", "metadata": {"category": category}})
    return "Saved."

@tool
async def learn_fact(subject: str, predicate: str, object_name: str):
    """Saves semantic fact to Graph."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{MEMORY_URL}/add_fact", json={"subject": subject, "predicate": predicate, "object": object_name})
    return "Fact learned."

@tool
async def recall(query: str, type: str = "episodic"):
    """Searches memory (episodic or semantic)."""
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{MEMORY_URL}/search", json={"text": query, "type": type})
        matches = res.json().get("matches", [])
        graph_info = []
        if type == "semantic":
            entity = query.split()[-1] 
            g_res = await client.post(f"{MEMORY_URL}/search_graph", params={"entity": entity})
            graph_info = g_res.json().get("relations", [])
        out = f"MEMORY ({type}):\n" + "\n".join(matches)
        if graph_info: out += "\nGRAPH:\n" + "\n".join(graph_info)
        return out

@tool
async def analyze_goal_knowledge_needs():
    """Analyzes active goals for knowledge gaps."""
    async with AsyncSessionLocal() as db:
        stmt = select(Goal).where(Goal.status == "active")
        goals = (await db.execute(stmt)).scalars().all()
    if not goals: return "No active goals."
    txt = "\n".join([f"- {g.title}: {g.description}" for g in goals])
    return f"ACTIVE GOALS:\n{txt}\nINSTRUCTION: Find missing technical knowledge."

@tool
async def get_random_concepts_for_synthesis():
    """Retrieves random concepts from memory."""
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{MEMORY_URL}/concepts/random")
        return str(res.json().get("concepts", []))

@tool
async def register_idea(title: str, description: str):
    """Registers a new idea."""
    async with AsyncSessionLocal() as db:
        new_goal = Goal(title=f"💡 {title}", description=description, status="idea")
        db.add(new_goal)
        await db.commit()
    return f"Idea '{title}' registered."

@tool
async def log_my_state(energy: int, mood: str, focus: int, notes: str = ""):
    """Logs the user's current state."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{MEMORY_URL}/user/state", json={"energy": energy, "mood": mood, "focus": focus, "notes": notes})
    return "State logged."

@tool
async def add_contact(name: str, relation: str, interests: str):
    """Adds a person to social graph."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{MEMORY_URL}/user/social", params={"name": name, "relation": relation}, json=interests.split(","))
    return "Contact added."

@tool
async def get_social_insights():
    """Returns social analysis."""
    async with httpx.AsyncClient() as client:
        return str((await client.get(f"{MEMORY_URL}/user/analysis")).json())

@tool
async def merge_knowledge_concepts(keep_concept: str, remove_concept: str):
    """Merges duplicate concepts."""
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{MEMORY_URL}/graph/merge", json={"primary": keep_concept, "alias": remove_concept})
        return str(res.json())

@tool
async def inspect_knowledge_graph():
    """Returns a list of entities to check for duplicates."""
    async with httpx.AsyncClient() as client:
        res = await client.get(f"{MEMORY_URL}/graph/inspect")
        return str(res.json())

@tool
async def prune_old_logs(days: int = 7):
    """Deletes system logs older than N days."""
    from sqlalchemy import text
    async with AsyncSessionLocal() as db:
        await db.execute(text(f"DELETE FROM run_logs WHERE created_at < NOW() - INTERVAL '{days} days'"))
        await db.commit()
    return "Logs pruned."
