
# --- PATCH: HANDLE WAIT STATE ---
if isinstance(result, dict) and result.get("status") == "WAIT":
    countdown = int(result.get("retry_after", 60))

    logger.info(
        f"⏳ Task deferred due to rate limit. Retry in {countdown}s"
    )

    raise self.retry(
        countdown=countdown,
        max_retries=10
    )
