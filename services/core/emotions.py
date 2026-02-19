from pydantic import BaseModel
from langchain_core.messages import SystemMessage
import os
from langchain_openai import ChatOpenAI

class EmotionalState(BaseModel):
    user_mood: str
    bot_mood: str
    color_hex: str

async def analyze_sentiment(messages):
    try:
        llm = ChatOpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY", "sk-1234"), model="vision-model", temperature=0.5).with_structured_output(EmotionalState)
        res = await llm.ainvoke([SystemMessage(content="Analyze mood. JSON.")] + messages[-3:])
        return res
    except: return EmotionalState(user_mood="neutral", bot_mood="professional", color_hex="#00FFFF")
