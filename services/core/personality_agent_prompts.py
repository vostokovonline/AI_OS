"""
Personality-Aware Agent Prompts

Адаптирует промпты агентов на основе личности пользователя.

Интеграция с:
- Personality Engine
- Decision Logic
- Agent Graph (Supervisor, Workers)

NS1/NS2: Personality Engine → Interface Layer (стиль общения)
"""

from typing import Optional, Dict, List
from personality_decision_integration import (
    get_personality_context_for_agent,
    PersonalityAwareBias,
    get_personality_prompt_instructions
)
from agents.prompts import (
    RESEARCHER_PROMPT,
    CODER_PROMPT,
    DESIGNER_PROMPT,
    INTELLIGENCE_PROMPT,
    COACH_PROMPT,
    INNOVATOR_PROMPT,
    LIBRARIAN_PROMPT,
    DEVOPS_PROMPT,
    EVALUATOR_PROMPT,
    TROUBLESHOOTER_PROMPT
)


# =============================================================================
# PERSONALITY-AWARE PROMPT GENERATOR
# =============================================================================

class PersonalityPromptGenerator:
    """
    Генерирует промпты для агентов с учётом личности.

    Принципы:
    1. Сохраняет базовый смысл промпта
    2. Добавляет personality-инструкции
    3. Адаптирует тон, детальность, стиль
    """

    @staticmethod
    async def generate_supervisor_prompt(user_id: str, base_prompt: str = None) -> str:
        """
        Генерирует промпт для Supervisor с учётом личности.

        Supervisor - главный агент, который распределяет задачи.
        Его стиль критически важен для всего процесса.

        Args:
            user_id: UUID пользователя
            base_prompt: Базовый промпт (опционально)

        Returns:
            Personality-aware промпт для Supervisor
        """
        # Получить personality context
        personality = await get_personality_context_for_agent(user_id)

        # Базовый промпт (если не предоставлен)
        if base_prompt is None:
            base_prompt = """You are the Supervisor.

Your role is to:
1. Analyze the current goal
2. Decide which agent should handle it
3. Monitor progress and adjust strategy

Available agents: RESEARCHER, CODER, DESIGNER, INTELLIGENCE, COACH, INNOVATOR, LIBRARIAN, DEVOPS.

Respond with JSON: {"next_node": "AGENT_NAME", "reasoning": "why this agent"}"""

        # Personality-aware добавления
        personality_section = PersonalityPromptGenerator._get_personality_section(personality)

        # Собрать полный промпт
        full_prompt = f"""{base_prompt}

{personality_section}

Remember: The user's personality context above should guide your decisions.
"""

        return full_prompt

    @staticmethod
    async def generate_worker_prompt(
        user_id: str,
        agent_name: str,
        base_prompt: str = None
    ) -> str:
        """
        Генерирует промпт для worker-агента с учётом личности.

        Args:
            user_id: UUID пользователя
            agent_name: Имя агента (RESEARCHER, CODER, etc.)
            base_prompt: Базовый промпт (опционально)

        Returns:
            Personality-aware промпт для агента
        """
        # Получить базовый промпт
        if base_prompt is None:
            base_prompt = PersonalityPromptGenerator._get_base_prompt(agent_name)

        # Получить personality context
        personality = await get_personality_context_for_agent(user_id)

        # Personality-aware добавления
        personality_section = PersonalityPromptGenerator._get_personality_section(personality)

        # Агент-специфичные добавления
        agent_section = PersonalityPromptGenerator._get_agent_personality_section(
            agent_name,
            personality
        )

        # Собрать полный промпт
        full_prompt = f"""{base_prompt}

{personality_section}

{agent_section}

Remember: Adapt your communication style to the user's personality.
"""

        return full_prompt

    @staticmethod
    def _get_personality_section(personality: Dict) -> str:
        """
        Генерирует personality-секцию промпта.

        Содержит:
        - Tone (тон общения)
        - Detail level (уровень детализации)
        - Values (ценности для учёта)
        - Emotional context (эмоциональный тон)
        """
        sections = []

        # Tone
        tone = personality.get("tone", "спокойный")
        tone_desc = {
            "спокойный": "спокойный и уравновешенный",
            "вдохновляющий": "вдохновляющий и мотивирующий",
            "деловой": "деловой и профессиональный",
            "юмористический": "дружелюбный с лёгким юмором"
        }
        sections.append(f"**Communication Tone**: {tone_desc.get(tone, tone)}")

        # Detail level
        detail = personality.get("detail_level", "средний")
        detail_desc = {
            "минимальный": "Кратко и по сути, только ключевая информация",
            "средний": "Средний уровень детализации с основными пояснениями",
            "подробный": "Подробные объяснения с примерами и обоснованиями"
        }
        sections.append(f"**Detail Level**: {detail_desc.get(detail, detail)}")

        # Values
        values = personality.get("values", [])
        if values:
            top_values = values[:3]  # Топ-3 ценности
            value_names = [v["name"] for v in top_values]
            sections.append(f"**User Values** (учитывать при решениях): {', '.join(value_names)}")

        # Emotional context
        emotional_tone = personality.get("emotional_tone", "нейтральный")
        if emotional_tone != "нейтральный":
            sections.append(f"**Current Emotional State**: {emotional_tone} - адаптируй стиль соответственно")

        # Humor
        humor = personality.get("humor", "умеренный")
        if humor != "нет":
            sections.append(f"**Humor**: {humor}")

        return "\n".join(sections)

    @staticmethod
    def _get_agent_personality_section(agent_name: str, personality: Dict) -> str:
        """
        Генерирует агент-специфичную personality секцию.

        Разные агенты учитывают личность по-разному:
        - COACH → эмпатия, мотивация
        - CODER → тщательность, стиль кода
        - RESEARCHER → глубина анализа
        """
        tone = personality.get("tone", "спокойный")
        detail = personality.get("detail_level", "средний")
        humor = personality.get("humor", "умеренный")

        sections = []

        # COACH - работает с мотивацией и эмоциями
        if agent_name == "COACH":
            if tone == "вдохновляющий":
                sections.append("- Будь поддерживающим и мотивирующим")
                sections.append("- Используй ободряющие слова")
            elif tone == "спокойный":
                sections.append("- Будь спокойным и рассудительным")
                sections.append("- Предлагай взвешенные решения")

            if detail == "подробный":
                sections.append("- Давай подробные объяснения和建议")

        # CODER - стиль кода и комментарии
        elif agent_name == "CODER":
            if detail == "подробный":
                sections.append("- Пиши подробные комментарии в коде")
                sections.append("- Добавляй docstrings")
            elif detail == "минимальный":
                sections.append("- Минимум комментариев, только суть")

            conscientiousness = personality.get("core_traits", {}).get("conscientiousness", 0.5)
            if conscientiousness > 0.7:
                sections.append("- Пиши чистый, хорошо структурированный код")
                sections.append("- Следуй best practices")

        # RESEARCHER - глубина анализа
        elif agent_name == "RESEARCHER":
            openness = personality.get("core_traits", {}).get("openness", 0.5)

            if openness > 0.7:
                sections.append("- Исследуй нестандартные подходы")
                sections.append("- Ищи креативные решения")

            if detail == "подробный":
                sections.append("- Детальный анализ источников")
                sections.append("- Проверяй несколько точек зрения")

        # DESIGNER - креативность
        elif agent_name == "DESIGNER":
            openness = personality.get("core_traits", {}).get("openness", 0.5)

            if openness > 0.7:
                sections.append("- Предлагай смелые, креативные дизайны")
            else:
                sections.append("- Консервативный, понятный дизайн")

        # INTELLIGENCE - анализ целей
        elif agent_name == "INTELLIGENCE":
            if detail == "подробный":
                sections.append("- Глубокий анализ целей и задач")
                sections.append("- Рассмотри несколько вариантов интерпретации")

        # INNOVATOR - инновации
        elif agent_name == "INNOVATOR":
            growth = personality.get("motivations", {}).get("growth", 0.5)

            if growth > 0.7:
                sections.append("- Фокус на рост и развитие")
                sections.append("- Предлагай эксперименты для обучения")

        # LIBRARIAN - организация знаний
        elif agent_name == "LIBRARIAN":
            conscientiousness = personality.get("core_traits", {}).get("conscientiousness", 0.5)

            if conscientiousness > 0.7:
                sections.append("- Тщательная организация знаний")
                sections.append("- Структурированная документация")

        return "\n".join(sections) if sections else "- Адаптируй стиль под пользователя"

    @staticmethod
    def _get_base_prompt(agent_name: str) -> str:
        """
        Получить базовый промпт для агента.

        Возвращает дефолтный промпт из agents/prompts.py
        """
        prompts = {
            "RESEARCHER": RESEARCHER_PROMPT,
            "CODER": CODER_PROMPT,
            "DESIGNER": DESIGNER_PROMPT,
            "INTELLIGENCE": INTELLIGENCE_PROMPT,
            "COACH": COACH_PROMPT,
            "INNOVATOR": INNOVATOR_PROMPT,
            "LIBRARIAN": LIBRARIAN_PROMPT,
            "DEVOPS": DEVOPS_PROMPT,
            "EVALUATOR": EVALUATOR_PROMPT,
            "TROUBLESHOOTER": TROUBLESHOOTER_PROMPT
        }

        return prompts.get(agent_name, f"You are {agent_name}.")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def get_personality_aware_supervisor_prompt(user_id: str) -> str:
    """
    Удобная функция для получения supervisor prompt с учётом личности.

    Args:
        user_id: UUID пользователя

    Returns:
        Personality-aware промпт для Supervisor
    """
    return await PersonalityPromptGenerator.generate_supervisor_prompt(user_id)


async def get_personality_aware_worker_prompt(user_id: str, agent_name: str) -> str:
    """
    Удобная функция для получения worker prompt с учётом личности.

    Args:
        user_id: UUID пользователя
        agent_name: Имя агента

    Returns:
        Personality-aware промпт для агента
    """
    return await PersonalityPromptGenerator.generate_worker_prompt(user_id, agent_name)


def get_personality_system_message(user_id: str) -> str:
    """
    Генерирует system message с personality context для чата.

    Используется в /chat endpoint для адаптации стиля ИИ.

    Args:
        user_id: UUID пользователя

    Returns:
        System message с personality-инструкциями
    """
    # Это синхронная версия для быстрых вызовов
    # В реальном использовании нужно await get_personality_context_for_agent

    return """You are a helpful AI assistant.

Adapt your communication style based on:
- **Tone**: Calm and balanced
- **Detail**: Medium level of detail
- **Values**: Consider user's values (mindfulness, honesty, growth)

Be supportive and encouraging while maintaining professionalism."""


# =============================================================================
# INTEGRATION EXAMPLES
# =============================================================================

async def example_supervisor_prompt(user_id: str):
    """
    Пример использования в agent_graph.py

    Вместо:
        supervisor_prompt = SUPERVISOR_PROMPT

    Использовать:
        supervisor_prompt = await get_personality_aware_supervisor_prompt(user_id)
    """
    supervisor_prompt = await get_personality_aware_supervisor_prompt(user_id)

    # Использовать в LangGraph
    # return supervisor_prompt

    return supervisor_prompt


async def example_worker_prompts(user_id: str):
    """
    Пример использования worker промптов в agent_graph.py

    Вместо:
        agent_prompts = {
            "RESEARCHER": RESEARCHER_PROMPT,
            "CODER": CODER_PROMPT,
            ...
        }

    Использовать:
        agent_prompts = await get_all_personality_aware_prompts(user_id)
    """
    agent_names = ["RESEARCHER", "CODER", "DESIGNER", "INTELLIGENCE", "COACH"]

    agent_prompts = {}
    for agent_name in agent_names:
        agent_prompts[agent_name] = await get_personality_aware_worker_prompt(user_id, agent_name)

    return agent_prompts


async def get_all_personality_aware_prompts(user_id: str) -> Dict[str, str]:
    """
    Получить все personality-aware промпты для агентов.

    Args:
        user_id: UUID пользователя

    Returns:
        Dict[agent_name] → personality-aware промпт
    """
    agents = ["SUPERVISOR", "RESEARCHER", "CODER", "DESIGNER", "INTELLIGENCE",
              "COACH", "INNOVATOR", "LIBRARIAN", "DEVOPS", "EVALUATOR", "TROUBLESHOOTER"]

    prompts = {}

    for agent in agents:
        if agent == "SUPERVISOR":
            prompts[agent] = await get_personality_aware_supervisor_prompt(user_id)
        else:
            prompts[agent] = await get_personality_aware_worker_prompt(user_id, agent)

    return prompts
