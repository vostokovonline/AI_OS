"""
Personality-Aware Decision Field Integration

Интегрирует Personality Engine с Decision Logic из NS1/NS2:

Decision Logic (из NS1/NS2):
- Context Analyzer - понимание ситуации
- Option Generator - создание вариантов
- Evaluator - оценка последствий
- Ethical Filter - проверка ценностей
- Adaptive Selector - выбор с учётом личности
- Meta-Decision Module - управление процессом

Интеграция:
1. Personality → DecisionField.bias
2. ContextualMemory → DecisionField.memory
3. Values → EthicalFilter
4. CoreTraits → AdaptiveSelector
"""

from typing import Optional, List, Dict, Literal
from decision_field import DecisionField, ExecutionBias, GoalPressure, Constraint, SystemState, DecisionFieldInput
from personality_engine import get_personality_engine, PersonalityProfileSchema, ContextualMemorySchema
from memory_signal import MemorySignal, memory_registry
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# ENHANCED DATA STRUCTURES
# =============================================================================

class PersonalityAwareBias(ExecutionBias):
    """
    Расширенный ExecutionBias с учётом личности.

    Добавляет к ExecutionBias:
    - communication_style (tone, humor, detail_level)
    - value_alignment (степень соответствия ценностям)
    - personality_driven (как личность повлияла на решение)
    """
    # Стиль общения
    tone: str = "спокойный"  # спокойный, вдохновляющий, деловой, юмористический
    humor: str = "умеренный"  # нет, умеренный, высокий
    detail_level: str = "средний"  # минимальный, средний, подробный

    # Соответствие ценностям
    value_alignment: float = 0.5  # 0..1, насколько решение соответствует ценностям

    # Как личность повлияла
    personality_driven: Dict = Field(default_factory=dict)


class PersonalityContext(BaseModel):
    """
    Контекст личности для принятия решений.

    Собирает данные из PersonalityEngine:
    - Core Traits (Big Five)
    - Motivations
    - Values
    - ContextualMemory
    """
    # Core traits (Big Five)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Motivations
    motivation_growth: float = 0.5
    motivation_achievement: float = 0.5
    motivation_comfort: float = 0.5
    motivation_recognition: float = 0.5
    motivation_social_connection: float = 0.5

    # Values (топ-5)
    values: List[Dict] = Field(default_factory=list)  # [{"name": "осознанность", "importance": 0.8}, ...]

    # Contextual memory
    emotional_tone: str = "нейтральный"
    recent_goals: List[Dict] = Field(default_factory=list)
    interaction_streak: int = 0


# =============================================================================
# PERSONALITY-AWARE DECISION FIELD
# =============================================================================

class PersonalityAwareDecisionField:
    """
    DecisionField с учётом личности пользователя.

    Интегрирует Personality Engine в процесс принятия решений:

    1. Получает PersonalityContext
    2. Превращает ценности в constraints
    3. Превращает contextual_memory в MemorySignal
    4. Применяет personality к ExecutionBias

    NS1/NS2 Integration:
    - Value Matrix → Ethical Filter
    - ContextualMemory → Context Analyzer
    - CoreTraits → Adaptive Selector
    """

    @staticmethod
    async def evaluate(
        user_id: str,
        goals: List[GoalPressure],
        constraints: List[Constraint] = None,
        system_state: SystemState = None
    ) -> PersonalityAwareBias:
        """
        Вычислить bias с учётом личности пользователя.

        Args:
            user_id: UUID пользователя
            goals: Список целей с давлением
            constraints: Список ограничений
            system_state: Состояние системы

        Returns:
            PersonalityAwareBias - bias с учётом личности
        """
        engine = get_personality_engine()

        # 1. Получить профиль личности
        profile = await engine.get_profile(user_id)

        # 2. Получить контекстную память
        context_memory = await engine.get_contextual_memory(user_id)

        # 3. Создать PersonalityContext
        personality_ctx = PersonalityContext(
            # Core traits
            openness=profile.core_traits.openness,
            conscientiousness=profile.core_traits.conscientiousness,
            extraversion=profile.core_traits.extraversion,
            agreeableness=profile.core_traits.agreeableness,
            neuroticism=profile.core_traits.neuroticism,

            # Motivations
            motivation_growth=profile.motivations.growth,
            motivation_achievement=profile.motivations.achievement,
            motivation_comfort=profile.motivations.comfort,
            motivation_recognition=profile.motivations.recognition,
            motivation_social_connection=profile.motivations.social_connection,

            # Values
            values=[v.dict() for v in profile.values],

            # Contextual memory
            emotional_tone=context_memory.emotional_tone_recent,
            recent_goals=[g.dict() for g in context_memory.recent_goals],
            interaction_streak=context_memory.interaction_streak
        )

        # 4. Превратить ценности в constraints
        value_constraints = PersonalityAwareDecisionField._values_to_constraints(profile.values)

        # 5. Превратить contextual_memory в MemorySignal
        memory_signals = PersonalityAwareDecisionField._contextual_memory_to_signals(
            context_memory,
            profile.values
        )

        # 6. Объединить constraints
        all_constraints = (constraints or []) + value_constraints

        # 7. Вычислить базовый bias через DecisionField
        decision_input = DecisionFieldInput(
            goals=goals,
            constraints=all_constraints,
            memory=memory_signals,
            system_state=system_state or SystemState()
        )

        base_bias = DecisionField.evaluate(decision_input)

        # 8. Применить personality к bias
        personality_bias = PersonalityAwareDecisionField._apply_personality(
            base_bias,
            personality_ctx,
            profile
        )

        return personality_bias

    @staticmethod
    def _values_to_constraints(values: List) -> List[Constraint]:
        """
        Превратить ценности в constraints для DecisionField.

        Примеры:
        - "здоровье" → resource_limit (на здоровье)
        - "честность" → safety constraint
        - "эффективность" → time_limit
        """
        constraints = []

        value_names = [v.value_name for v in values]

        # Здоровье → ограничение на переработки
        if any("здоровье" in v.lower() for v in value_names):
            health_importance = next((v.importance for v in values if "здоровье" in v.value_name.lower()), 0.5)
            constraints.append(Constraint(
                type="health_limit",
                severity=health_importance
            ))

        # Честность → safety constraint
        if any("честност" in v.lower() for v in value_names):
            constraints.append(Constraint(
                type="safety",
                severity=0.8  # Высокий приоритет
            ))

        # Эффективность → time_pressure
        if any("эффективност" in v.lower() or "продуктивност" in v.lower() for v in value_names):
            constraints.append(Constraint(
                type="time_limit",
                severity=0.6
            ))

        # Осознанность → depth preference
        if any("осознанност" in v.lower() for v in value_names):
            constraints.append(Constraint(
                type="depth_preference",
                severity=0.7
            ))

        return constraints

    @staticmethod
    def _contextual_memory_to_signals(
        contextual_memory,
        values: List
    ) -> List[MemorySignal]:
        """
        Превратить ContextualMemory в MemorySignal.

        Примеры:
        - Emotional tone → mood signal
        - Recent failures → recent_failure signal
        - Interaction streak → engagement signal
        """
        signals = []

        # Emotional tone → mood signal
        tone = contextual_memory.emotional_tone_recent.lower()

        if "уставший" in tone or "апатичн" in tone:
            signals.append(MemorySignal(
                type="resource_exhaustion",
                target="energy",
                intensity=0.6,
                ttl=60  # 1 час
            ))

        elif "вдохновлен" in tone or "оптимистичн" in tone:
            signals.append(MemorySignal(
                type="success_streak",
                target="motivation",
                intensity=0.7,
                ttl=120  # 2 часа
            ))

        # Interaction streak
        if contextual_memory.interaction_streak > 7:
            # Длительный streak → может overfit
            signals.append(MemorySignal(
                type="overfitting",
                target="routine",
                intensity=0.3,
                ttl=240  # 4 часа
            ))

        # Recent goals analysis
        if contextual_memory.behavioral_summary_week:
            summary = contextual_memory.behavioral_summary_week
            completed = summary.completed_tasks if hasattr(summary, 'completed_tasks') else 0
            missed = summary.missed_tasks if hasattr(summary, 'missed_tasks') else 0

            if missed > completed * 0.5:
                # Много пропусков → снижать сложность
                signals.append(MemorySignal(
                    type="recent_failure",
                    target="complexity",
                    intensity=0.5,
                    ttl=180  # 3 часа
                ))

        return signals

    @staticmethod
    def _apply_personality(
        base_bias: ExecutionBias,
        personality_ctx: PersonalityContext,
        profile: PersonalityProfileSchema
    ) -> PersonalityAwareBias:
        """
        Применить личность к ExecutionBias.

        Core Traits влияют:
        - Openness → глубина анализа, креативность
        - Conscientiousness → тщательность, detail_level
        - Extraversion → стиль общения
        - Agreeableness → эмпатия, cooperation
        - Neuroticism → risk_tolerance
        """
        # 1. Communication style из preferences
        tone = profile.preferences.communication_style.tone
        humor = profile.preferences.communication_style.humor
        detail = profile.preferences.communication_style.detail_level

        # 2. Openness → глубина и креативность
        if personality_ctx.openness > 0.7:
            # Высокая открытость → глубокий анализ, креативность
            if base_bias.depth == "shallow":
                base_bias.depth = "medium"
            if base_bias.llm_profile == "fast":
                base_bias.llm_profile = "creative"

        elif personality_ctx.openness < 0.3:
            # Низкая открытость → консервативность
            if base_bias.llm_profile == "creative":
                base_bias.llm_profile = "balanced"

        # 3. Conscientiousness → тщательность
        if personality_ctx.conscientiousness > 0.7:
            # Высокая добросовестность → подробности
            detail = "подробный"
            if base_bias.depth == "shallow":
                base_bias.depth = "medium"

        elif personality_ctx.conscientiousness < 0.3:
            # Низкая добросовестность → краткость
            detail = "минимальный"

        # 4. Extraversion → стиль общения
        if personality_ctx.extraversion > 0.7:
            # Экстраверт → активный, юмористический
            if tone == "спокойный":
                tone = "вдохновляющий"
            if humor == "нет":
                humor = "умеренный"

        elif personality_ctx.extraversion < 0.3:
            # Интроверт → спокойный
            if tone == "вдохновляющий":
                tone = "спокойный"

        # 5. Agreeableness → cooperation
        if personality_ctx.agreeableness > 0.7:
            # Высокая доброжелательность → избегать конфликтов
            base_bias.avoid_skills.append("confront")

        # 6. Neuroticism → risk tolerance
        if personality_ctx.neuroticism > 0.7:
            # Высокий нейротизм → снижение рисков
            base_bias.risk_tolerance = max(0.1, base_bias.risk_tolerance - 0.3)
            base_bias.llm_profile = "paranoid" if base_bias.llm_profile == "balanced" else base_bias.llm_profile

        # 7. Motivations → приоритеты
        if personality_ctx.motivation_growth > 0.7:
            # Рост → предпочтение learning навыков
            base_bias.prefer_skills.extend(["learn", "analyze", "research"])

        if personality_ctx.motivation_achievement > 0.7:
            # Достижения → предпочтение execute навыков
            base_bias.prefer_skills.extend(["execute", "optimize"])

        if personality_ctx.motivation_comfort > 0.7:
            # Комфорт → избегать рискованных
            base_bias.avoid_skills.extend(["risky", "experimental"])

        # 8. Вычислить value_alignment
        # (простая эвристика - можно улучшить)
        value_alignment = 0.5  # базовая

        if personality_ctx.motivation_growth > 0.7:
            value_alignment += 0.2

        if personality_ctx.conscientiousness > 0.7:
            value_alignment += 0.2

        value_alignment = min(1.0, value_alignment)

        # 9. Создать PersonalityAwareBias
        personality_bias = PersonalityAwareBias(
            # Базовые поля из ExecutionBias
            prefer_skills=base_bias.prefer_skills,
            avoid_skills=base_bias.avoid_skills,
            depth=base_bias.depth,
            speed=base_bias.speed,
            risk_tolerance=base_bias.risk_tolerance,
            retry_aggressiveness=base_bias.retry_aggressiveness,
            llm_profile=base_bias.llm_profile,

            # Personality-specific поля
            tone=tone,
            humor=humor,
            detail_level=detail,
            value_alignment=value_alignment,
            personality_driven={
                "openness": personality_ctx.openness,
                "conscientiousness": personality_ctx.conscientiousness,
                "extraversion": personality_ctx.extraversion,
                "agreeableness": personality_ctx.agreeableness,
                "neuroticism": personality_ctx.neuroticism,
            }
        )

        return personality_bias


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def evaluate_with_personality(
    user_id: str,
    goals: List[GoalPressure],
    constraints: List[Constraint] = None
) -> PersonalityAwareBias:
    """
    Удобная функция для получения personality-aware bias.

    Args:
        user_id: UUID пользователя
        goals: Список целей
        constraints: Опциональные ограничения

    Returns:
        PersonalityAwareBias
    """
    return await PersonalityAwareDecisionField.evaluate(
        user_id=user_id,
        goals=goals,
        constraints=constraints,
        system_state=SystemState()
    )


def get_personality_prompt_instructions(bias: PersonalityAwareBias) -> str:
    """
    Генерирует prompt инструкции на основе personality bias.

    Используется в agent_graph.py для настройки промптов агентов.

    Args:
        bias: PersonalityAwareBias

    Returns:
        str - инструкции для промпта
    """
    instructions = []

    # Tone
    tone_instructions = {
        "спокойный": "Спокойный и уравновешенный тон. Избегай излишней эмоциональности.",
        "вдохновляющий": "Вдохновляющий и мотивирующий тон. Используй ободряющие слова.",
        "деловой": "Деловой и профессиональный тон. Чётко и по существу.",
        "юмористический": "Лёгкий юмор и ирония. Не перебарщивай, поддерживай дружелюбную атмосферу."
    }
    instructions.append(tone_instructions.get(bias.tone, tone_instructions["спокойный"]))

    # Detail level
    detail_instructions = {
        "минимальный": "Отвечай кратко и по сути. Только ключевая информация.",
        "средний": "Средний уровень детализации. Основная информация с небольшими пояснениями.",
        "подробный": "Подробные объяснения с примерами и обоснованиями."
    }
    instructions.append(detail_instructions.get(bias.detail_level, detail_instructions["средний"]))

    # Value alignment
    if bias.value_alignment > 0.7:
        instructions.append("ПРИОРИТИЗУЙЙ действия, которые соответствуют ценностям пользователя (осознанность, честность, развитие).")

    # Risk tolerance
    if bias.risk_tolerance < 0.3:
        instructions.append("Будь ОСТОРОЖНЫМ. Избегай рискованных действий. Предпочитай проверенные решения.")
    elif bias.risk_tolerance > 0.7:
        instructions.append("Можешь позволить себе рискованные и креативные решения.")

    # Depth
    depth_instructions = {
        "shallow": "Действуй быстро. Не углубляйся в детали. Предпочитай простые решения.",
        "medium": "Баланс между скоростью и глубиной. Анализируй, но не переусердствуй.",
        "deep": "Тщательный анализ. Рассмотри все аспекты перед принятием решения."
    }
    instructions.append(depth_instructions.get(bias.depth, depth_instructions["medium"]))

    return "\n".join(instructions)


# =============================================================================
# INTEGRATION WITH AGENT GRAPH
# =============================================================================

async def get_personality_context_for_agent(user_id: str) -> Dict:
    """
    Получить personality context для использования в agent_graph.py

    Returns:
        Dict с personality данными для промптов
    """
    engine = get_personality_engine()

    profile = await engine.get_profile(user_id)
    context_memory = await engine.get_contextual_memory(user_id)

    return {
        "user_id": user_id,
        "tone": profile.preferences.communication_style.tone,
        "humor": profile.preferences.communication_style.humor,
        "detail_level": profile.preferences.communication_style.detail_level,
        "core_traits": profile.core_traits.dict(),
        "motivations": profile.motivations.dict(),
        "values": [v.dict() for v in profile.values],
        "emotional_tone": context_memory.emotional_tone_recent,
        "recent_goals": [g.dict() for g in context_memory.recent_goals],
    }
