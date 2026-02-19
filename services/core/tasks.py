import os, asyncio, httpx, traceback
from celery import Celery
from langchain_core.messages import HumanMessage
from resource_manager import SystemMonitor
from agent_graph import app_graph
import redis

# Import goal executor tasks to register them with Celery
from goal_executor import execute_goal_task, execute_complex_goal_task
# Import shared celery app
from celery_config import celery_app

monitor = SystemMonitor()

async def notify(msg, sid=None):
    try:
        if sid and "tg_" in sid: await httpx.post(f"{os.getenv('TELEGRAM_URL')}/ask_human", json={"chat_id": sid, "text": msg})
        else: await httpx.post(f"{os.getenv('TELEGRAM_URL')}/notify", json={"message": msg})
    except: pass

async def _exec(sid, input_msg=None):
    # FIX: Define cfg before using it
    cfg = {"configurable": {"thread_id": sid}, "recursion_limit": 50}
    inputs = {"messages": [input_msg]} if input_msg else None

    try:
        print(f"⚙️ Executing Graph for {sid}")
        async for event in app_graph.astream(inputs, cfg, stream_mode="values"): final = event
        res = final['messages'][-1].content
        await notify(f"✅ DONE: {res[:2000]}")
        return res
    except Exception as e:
        print(f"🔥 Error: {e}")
        await notify(f"🔥 SYSTEM ERROR: {e}")
        return "ERROR"

    snap = await app_graph.aget_state(cfg)
    if snap.next and snap.next[0] == "HUMAN":
        await notify(f"🛑 PAUSED: {final['messages'][-1].content}", sid)
        return "PAUSED"

@celery_app.task(bind=True)
def run_chat_task(self, session_id, content, image_url=None):
    if not monitor.check_health(): return "BUSY"
    msg = HumanMessage(content=[{"type":"text","text":content},{"type":"image_url","image_url":image_url}]) if image_url else HumanMessage(content=content)
    return asyncio.run(_exec(session_id, msg))

@celery_app.task(bind=True)
def run_resume_task(self, session_id):
    return asyncio.run(_exec(session_id, None))

@celery_app.task(bind=True)
def run_cron_task(self, session_id, content):
    return asyncio.run(_exec(session_id, HumanMessage(content=content)))
