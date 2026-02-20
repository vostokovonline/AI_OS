import httpx
import os
from langchain_core.tools import tool

OPENCODE_URL = os.getenv("OPENCODE_URL", "http://opencode:8002")

@tool("run_python_code")
async def run_python_code(code: str) -> str:
    """
    Выполняет код Python в изолированной среде (Jupyter Kernel) для анализа данных, 
    работы с файловой системой и выполнения сложных вычислений.
    Используй `logger.info()` для вывода результата.

    Пример 1: "создай файл test.txt с текстом 'hello'"
    ```python
    with open("test.txt", "w") as f:
        f.write("hello world")
    logger.info("Файл создан.")
    ```

    Пример 2: "узнай текущую дату"
    ```python
    import datetime
    logger.info(datetime.date.today())
    ```
    """
    logger.info(f"👨‍💻 EXECUTING CODE:\n---\n{code}\n---")
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            # Используем фиксированный session_id, чтобы все операции были в одной среде
            payload = {"session_id": "global_ai_os_session", "code": code}
            
            resp = await client.post(f"{OPENCODE_URL}/run", json=payload)
            resp.raise_for_status() # Вызовет ошибку, если статус не 2xx
            
            data = resp.json()
            if data.get("status") == "error":
                return f"❌ Ошибка выполнения:\n{data.get('stderr', 'Неизвестная ошибка')}"
            
            output = data.get("stdout", "Нет вывода.")
            # Возвращаем только первые 2000 символов, чтобы не перегружать контекст
            return f"✅ Результат:\n{output[:2000]}"
            
    except httpx.HTTPStatusError as e:
        return f"❌ Ошибка сети (HTTP {e.response.status_code}): {e.response.text}"
    except Exception as e:
        return f"❌ Неизвестная ошибка: {str(e)}"

# Список для экспорта
OPENCODE_TOOLS = [run_python_code]
