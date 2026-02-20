
# --- PATCH: RATE LIMIT AWARE SUPERVISOR ---
from litellm.exceptions import RateLimitError
from datetime import datetime

def _safe_run_with_wait(self, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)

    except RateLimitError as e:
        retry_after = getattr(e, "retry_after", 60)

        self.logger.warning(
            f"🚦 RATE LIMIT — retry after {retry_after}s"
        )

        return {
            "status": "WAIT",
            "reason": "rate_limit",
            "retry_after": retry_after,
            "preferred_model": os.getenv("LLM_MODEL", "smart-model"),
            "timestamp": datetime.utcnow().isoformat()
        }
