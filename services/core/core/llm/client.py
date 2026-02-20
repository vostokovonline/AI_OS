import os
import time
from litellm import completion

DEFAULT_MODEL = os.getenv("LLM_MODEL", "cloud-reasoner")

def call_llm(messages, model=DEFAULT_MODEL, retries=3):
    """
    Call LLM with Groq fallback support.

    This function now uses the llm_fallback manager which automatically
    switches from Groq to Ollama when Groq rate limits are hit (404 errors).
    """
    # Import here to avoid circular import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    try:
        from llm_fallback import chat_with_fallback_sync
        # Use the fallback manager (sync version)
        return chat_with_fallback_sync(model, messages)
    except ImportError:
        # Fallback to original behavior if llm_fallback not available
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return completion(
                    model=model,
                    messages=messages,
                )
            except Exception as e:
                last_error = e
                if "429" in str(e):
                    time.sleep(2 * attempt)
                else:
                    time.sleep(1)

        raise RuntimeError(f"LLM failed after {retries} retries: {last_error}")
