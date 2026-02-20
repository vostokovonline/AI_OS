"""
LangChain LLM with Fallback Support
Интеграция llm_fallback с LangChain ChatOpenAI
"""
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult, LLMResult
import os


class ChatOpenAIWithFallback(ChatOpenAI):
    """
    ChatOpenAI с поддержкой Groq fallback.

    Перехватывает вызовы и использует llm_fallback для детекции
    404 ошибок от Groq и автоматического переключения на Ollama.
    """

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Генерирует ответ с fallback поддержкой.

        LangChain вызывает этот метод. Мы перехватываем и используем
        llm_fallback который умеет переключаться на Ollama при 404 от Groq.
        """
        # Import here to avoid circular import
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

        try:
            from llm_fallback import chat_with_fallback_sync
            from langchain_core.messages import message_to_dict

            # Конвертируем LangChain messages в формат OpenAI
            openai_messages = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    openai_messages.append({
                        "role": msg.type,
                        "content": msg.content
                    })
                else:
                    # Fallback для других типов messages
                    openai_messages.append({
                        "role": "user",
                        "content": str(msg.content)
                    })

            # Вызываем LLM с fallback
            model = self.model_name or os.getenv("LLM_MODEL", "groq/llama-3.3-70b-versatile")

            result = chat_with_fallback_sync(
                model=model,
                messages=openai_messages,
                temperature=self.temperature,
                **kwargs
            )

            # Парсим результат и конвертируем обратно в LangChain формат
            from langchain_core.outputs import ChatGeneration, ChatResult
            from langchain_core.messages import AIMessage

            content = result["choices"][0]["message"]["content"]

            generation = ChatGeneration(
                message=AIMessage(content=content)
            )

            return ChatResult(generations=[generation])

        except ImportError:
            # Если llm_fallback недоступен, используем оригинальный метод
            logger.info("⚠️ llm_fallback not available, using default ChatOpenAI")
            return super()._generate(messages, stop, run_manager, **kwargs)


def get_model_with_fallback(role="DEFAULT"):
    """
    Возвращает LangChain модель с fallback поддержкой и учетом роли.

    Использует разные модели для разных задач.
    """
    # Mapping ролей на модели
    MODEL_MAPPING = {
        "SUPERVISOR": "ollama/gpt-oss:120b-cloud",         # ⚡ Быстрый роутинг (120B)
        "CODER": "ollama/qwen3-coder:480b-cloud",          # 💻 Код (480B)
        "PM": "ollama/gpt-oss:120b-cloud",                 # 🎯 Управление (120B)
        "RESEARCHER": "ollama/qwen3-coder:480b-cloud",     # 🔍 Поиск (480B)
        "INTELLIGENCE": "ollama/deepseek-v3.1:671b-cloud", # 🧠 Сложные рассуждения (671B)
        "DEFAULT": "ollama/qwen3-coder:480b-cloud"
    }

    model_name = MODEL_MAPPING.get(role, os.getenv("LLM_MODEL", "ollama/qwen3-coder:480b-cloud"))

    # Temperature по роли
    if role == "SUPERVISOR":
        temp = 0.1
    elif role == "INTELLIGENCE":
        temp = 0.3
    else:
        temp = 0.2

    return ChatOpenAIWithFallback(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-1234"),
        model=model_name,
        temperature=temp,
        request_timeout=120  # 2 минуты (все модели быстрые)
    )
