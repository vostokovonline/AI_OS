"""
LLM API Endpoints Module
"""
from fastapi import APIRouter
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/llm", tags=["llm"])


class LLMTestRequest(BaseModel):
    prompt: str
    model: Optional[str] = "groq/llama-3.3-70b-versatile"


@router.get("/status")
async def get_llm_status():
    """Получить статус LLM fallback системы"""
    from llm_fallback import llm_fallback
    
    status = await llm_fallback.get_status()
    return {"status": "ok", "llm_status": status}


@router.post("/reset_groq")
async def reset_groq_cooldown():
    """Вручную сбросить Groq cooldown и включить его обратно"""
    from llm_fallback import async_redis, GROQ_DISABLED_KEY, GROQ_FAILURE_KEY
    from llm_fallback import llm_fallback
    
    await async_redis.delete(GROQ_DISABLED_KEY, GROQ_FAILURE_KEY)
    status = await llm_fallback.get_status()
    
    return {
        "status": "ok",
        "message": "Groq cooldown reset manually",
        "new_status": status
    }


@router.post("/test")
async def test_llm(request: LLMTestRequest):
    """Тестовый вызов LLM с fallback"""
    from llm_fallback import chat_with_fallback
    
    messages = [{"role": "user", "content": request.prompt}]
    
    try:
        result = await chat_with_fallback(request.model, messages)
        return {
            "status": "ok",
            "model_used": result.get("model", request.model),
            "response": result.get("choices", [{}])[0].get("message", {}).get("content", "")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
