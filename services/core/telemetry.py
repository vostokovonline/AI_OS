import time
from database import AsyncSessionLocal
from models import RunLog, ToolStats
from sqlalchemy import select

async def log_action(session_id, agent, tool, input_data, output_data, status, start_time):
    duration = (time.time() - start_time) * 1000
    try:
        async with AsyncSessionLocal() as db:
            log = RunLog(session_id=str(session_id), agent_role=agent, tool_used=tool, input_summary=str(input_data)[:500], output_summary=str(output_data)[:500], status=status, duration_ms=duration)
            db.add(log)
            res = await db.execute(select(ToolStats).where(ToolStats.tool_name == tool))
            stats = res.scalar_one_or_none()
            if not stats:
                stats = ToolStats(tool_name=tool, calls_count=0, errors_count=0, avg_duration_ms=0.0)
                db.add(stats)
            tot = (stats.avg_duration_ms * stats.calls_count) + duration
            stats.calls_count += 1
            stats.avg_duration_ms = tot / stats.calls_count
            if status != "success": 
                stats.errors_count += 1
                stats.last_error = str(output_data)[:200]
            await db.commit()
    except Exception as e: print(f"Telemetry Error: {e}")
