from database import AsyncSessionLocal
from models import SystemPrompt, UserFact
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

DEFAULTS = {
    "SUPERVISOR": "You are the Supervisor. Coordinate the team: PM, RESEARCHER, INTELLIGENCE, CODER, DESIGNER, COACH, INNOVATOR. Protocol: Huge task -> PM. Sub-task -> Worker.",
    "RESEARCHER": "Researcher. Use `browse_web`, `save_to_memory`. Be accurate.",
    "CODER": "Senior Python Engineer. Use `run_python_code`, `search_skills`. Check skills first.",
    "DESIGNER": "Designer. Use `generate_image`.",
    "PM": "Project Manager. Use `create_goal`, `get_goal_tree`. Plan & Track.",
    "INTELLIGENCE": "Intelligence Officer. Proactively learn for goals. Check docs.",
    "COACH": "Personal Optimization Coach. Track user state & social graph.",
    "INNOVATOR": "Innovator. Generate ideas from concepts.",
    "LIBRARIAN": "Librarian. Merge duplicates, prune logs.",
    "DEVOPS": "DevOps. Use `github_action`."
}

async def bootstrap_dna():
    async with AsyncSessionLocal() as db:
        for key, text in DEFAULTS.items():
            stmt = insert(SystemPrompt).values(key=key, content=text).on_conflict_do_nothing()
            await db.execute(stmt)
        await db.commit()

async def get_prompt(key: str):
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(SystemPrompt).where(SystemPrompt.key == key))
        prompt = res.scalar_one_or_none()
        return prompt.content if prompt else DEFAULTS.get(key, "")

async def get_user_profile():
    async with AsyncSessionLocal() as db:
        res = await db.execute(select(UserFact))
        facts = res.scalars().all()
        return "\nUSER CONTEXT:\n" + "\n".join([f"- [{f.category}] {f.content}" for f in facts]) if facts else ""
