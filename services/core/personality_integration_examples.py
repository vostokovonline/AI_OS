"""
Personality Integration Examples

Примеры использования Personality Engine в различных компонентах системы.

1. Goal Executor + Personality
2. Agent Graph + Personality
3. Chat + Personality
4. Goal Decomposition + Personality
"""

from typing import Dict, List
from decision_field import GoalPressure
from personality_decision_integration import (
    evaluate_with_personality,
    get_personality_context_for_agent
)
from personality_agent_prompts import (
    get_all_personality_aware_prompts,
    get_personality_aware_supervisor_prompt
)
from goal_conflict_detector import get_goal_conflict_detector
from personality_engine import get_personality_engine


# =============================================================================
# EXAMPLE 1: GOAL EXECUTOR WITH PERSONALITY
# =============================================================================

async def example_goal_executor_with_personality(goal_id: str, user_id: str):
    """
    Пример: Выполнение цели с учётом личности.

    В goal_executor.py или goal_executor_v2.py:
    """
    engine = get_personality_engine()

    # 1. Получить профиль
    profile = await engine.get_profile(user_id)

    # 2. Получить contextual memory
    context = await engine.get_contextual_memory(user_id)

    print(f"Выполняю goal для пользователя с эмоциональным тоном: {context.emotional_tone_recent}")
    print(f"Стиль общения: {profile.preferences.communication_style.tone}")

    # 3. Проверить конфликты
    detector = get_goal_conflict_detector()
    conflicts = await detector.check_goal_conflicts(goal_id)

    if conflicts.has_conflicts:
        print(f"⚠️  Обнаружены конфликты: {len(conflicts.conflicts)}")
        # Предложить пользователю варианты разрешения

    # 4. Вычислить bias с учётом личности
    goals_pressure = [GoalPressure(
        goal_id=goal_id,
        title="Текущая цель",
        priority="high",
        magnitude=0.7,
        risk_tolerance=profile.core_traits.openness  # Риск зависит от открытости
    )]

    bias = await evaluate_with_personality(
        user_id=user_id,
        goals=goals_pressure
    )

    print(f"Коммуникационный стиль: {bias.tone}")
    print(f"Уровень детализации: {bias.detail_level}")
    print(f"LLM профиль: {bias.llm_profile}")

    # 5. Использовать bias для настройки execution
    return {
        "bias": bias,
        "conflicts": conflicts,
        "personality_context": {
            "tone": bias.tone,
            "detail_level": bias.detail_level,
            "value_alignment": bias.value_alignment
        }
    }


# =============================================================================
# EXAMPLE 2: AGENT GRAPH WITH PERSONALITY
# =============================================================================

async def example_agent_graph_with_personality(goal_id: str, user_id: str):
    """
    Пример: Agent Graph с personality-aware промптами.

    В agent_graph.py:
    """
    # Вместо статических промптов:
    # SUPERVISOR_PROMPT = "You are the Supervisor..."
    # CODER_PROMPT = "You are a Coder..."
    # Использовать personality-aware:

    # 1. Получить все промпты с учётом личности
    agent_prompts = await get_all_personality_aware_prompts(user_id)

    # 2. Использовать в LangGraph nodes
    supervisor_node = {
        "name": "supervisor",
        "prompt": agent_prompts["SUPERVISOR"],
        # ... остальная конфигурация
    }

    coder_node = {
        "name": "coder",
        "prompt": agent_prompts["CODER"],
        # ... остальная конфигурация
    }

    # 3. Каждый агент будет адаптировать свой стиль под пользователя
    return agent_prompts


# =============================================================================
# EXAMPLE 3: CHAT WITH PERSONALITY
# =============================================================================

async def example_chat_with_personality(user_id: str, message: str):
    """
    Пример: Chat с адаптацией под личность.

    В main.py, endpoint /chat:
    """
    from personality_agent_prompts import get_personality_system_message
    from personality_decision_integration import get_personality_context_for_agent

    # 1. Получить personality context
    personality = await get_personality_context_for_agent(user_id)

    # 2. Сгенерировать system message
    system_message = f"""You are a helpful AI assistant.

User Personality Context:
- **Tone**: {personality['tone']}
- **Detail Level**: {personality['detail_level']}
- **Values**: {', '.join([v['name'] for v in personality['values'][:3]])}
- **Current Mood**: {personality['emotional_tone']}

Adapt your communication style accordingly.
"""

    # 3. Использовать в LLM вызове
    # response = await llm.generate(
    #     messages=[
    #         {"role": "system", "content": system_message},
    #         {"role": "user", "content": message}
    #     ]
    # )

    return system_message


# =============================================================================
# EXAMPLE 4: GOAL DECOMPOSITION WITH PERSONALITY
# =============================================================================

async def example_goal_decomposition_with_personality(parent_goal_id: str, user_id: str):
    """
    Пример: Декомпозиция цели с учётом личности.

    В goal_decomposer.py:
    """
    engine = get_personality_engine()

    # 1. Получить профиль
    profile = await engine.get_profile(user_id)

    # 2. Проверить мотивацию
    growth_motivation = profile.motivations.growth
    achievement_motivation = profile.motivations.achievement

    # 3. Адаптировать стратегию декомпозиции
    if growth_motivation > 0.7:
        # Высокая мотивация роста → создавать learning subgoals
        decomposition_strategy = {
            "include_learning_goals": True,
            "focus_on_development": True,
            "subgoal_types": ["learn", "practice", "experiment"]
        }
    elif achievement_motivation > 0.7:
        # Высокая мотивация достижения → фокус на measurable results
        decomposition_strategy = {
            "include_measurable_goals": True,
            "focus_on_results": True,
            "subgoal_types": ["execute", "optimize", "deliver"]
        }
    else:
        # Балансированная стратегия
        decomposition_strategy = {
            "balanced": True,
            "subgoal_types": ["plan", "execute", "review"]
        }

    # 4. Проверить конфликты перед созданием subgoals
    detector = get_goal_conflict_detector()

    # 5. Декомпозировать с учётом стратегии
    # subgoals = await decompose_goal(parent_goal_id, strategy=decomposition_strategy)

    return decomposition_strategy


# =============================================================================
# EXAMPLE 5: CONTEXTUAL MEMORY UPDATE AFTER GOAL COMPLETION
# =============================================================================

async def example_update_contextual_memory_after_goal(goal_id: str, user_id: str, result: str):
    """
    Пример: Обновление contextual memory после выполнения цели.

    Вызывать после завершения goal:
    """
    engine = get_personality_engine()

    # 1. Получить текущую контекстную память
    context = await engine.get_contextual_memory(user_id)

    # 2. Обновить recent goals
    # (получить top 5 целей из БД)
    recent_goals = await get_recent_goals_from_db(user_id, limit=5)

    # 3. Определить emotional tone на основе результата
    if result == "success":
        emotional_tone = "вдохновленный"
    elif result == "failure":
        emotional_tone = "разочарованный"
    else:
        emotional_tone = "нейтральный"

    # 4. Обновить behavioral summary
    completed = context.behavioral_summary_week.completed_tasks if context.behavioral_summary_week else 0
    missed = context.behavioral_summary_week.missed_tasks if context.behavioral_summary_week else 0

    if result == "success":
        completed += 1
    elif result == "failure":
        missed += 1

    behavioral_summary = {
        "completed_tasks": completed,
        "missed_tasks": missed,
        "interaction_frequency": "ежедневно" if completed > 5 else "периодически"
    }

    # 5. Сохранить
    await engine.update_contextual_memory(
        user_id=user_id,
        recent_goals=[g.dict() for g in recent_goals],
        emotional_tone=emotional_tone,
        behavioral_summary=behavioral_summary
    )

    print(f"✅ Contextual memory обновлена: {emotional_tone}")


# =============================================================================
# EXAMPLE 6: PERSONALITY FEEDBACK LOOP
# =============================================================================

async def example_personality_feedback_loop(goal_id: str, user_id: str, user_feedback: str):
    """
    Пример: Адаптация личности на основе feedback.

    Когда пользователь даёт feedback на решение ИИ:
    """
    engine = get_personality_engine()

    # 1. Создать snapshot перед адаптацией
    snapshot = await engine.create_snapshot(
        user_id=user_id,
        reason="feedback_adaptation",
        created_by="system"
    )

    # 2. Проанализировать feedback
    if "слишком подробно" in user_feedback.lower():
        # Пользователь хочет меньше деталей
        profile = await engine.get_profile(user_id)
        new_detail = "минимальный" if profile.preferences.communication_style.detail_level == "средний" else "средний"

        await engine.update_profile(user_id, {
            "preferences": {
                "communication_style": {
                    "detail_level": new_detail
                }
            }
        })

        print(f"✅ Детальность снижена до: {new_detail}")

    elif "слишком кратко" in user_feedback.lower():
        # Пользователь хочет больше деталей
        profile = await engine.get_profile(user_id)
        new_detail = "подробный" if profile.preferences.communication_style.detail_level == "средний" else "средний"

        await engine.update_profile(user_id, {
            "preferences": {
                "communication_style": {
                    "detail_level": new_detail
                }
            }
        })

        print(f"✅ Детальность повышена до: {new_detail}")

    elif "слишком эмоционально" in user_feedback.lower():
        # Пользователь хочет более спокойный тон
        await engine.update_profile(user_id, {
            "preferences": {
                "communication_style": {
                    "tone": "спокойный"
                }
            }
        })

        print("✅ Тон изменён на спокойный")

    # 3. Записать feedback
    await engine.record_feedback(
        user_id=user_id,
        event_type="communication_style_feedback",
        reaction="negative" if "слишком" in user_feedback else "positive",
        context={"feedback_text": user_feedback},
        source="user_explicit"
    )

    return {"status": "adapted", "snapshot_version": snapshot.snapshot_version}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_recent_goals_from_db(user_id: str, limit: int = 5) -> List:
    """
    Получить недавние цели из БД.

    В реальной системе это будет запрос к SQLAlchemy:
    """
    from sqlalchemy import select
    from models import Goal
    from database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        stmt = select(Goal)\
            .where(Goal.user_id == user_id)\
            .order_by(Goal.created_at.desc())\
            .limit(limit)

        result = await db.execute(stmt)
        goals = result.scalars().all()

        return [
            {
                "id": str(g.id),
                "title": g.title,
                "status": g.status,
                "progress": g.progress or 0.0
            }
            for g in goals
        ]


# =============================================================================
# FULL INTEGRATION EXAMPLE
# =============================================================================

async def full_personality_integration_example(goal_title: str, user_id: str):
    """
    Полный пример интеграции всех компонентов.

    Показывает как Personality влияет на весь lifecycle цели:
    1. Создание
    2. Декомпозиция
    3. Выполнение
    4. Feedback
    """
    print(f"\n{'='*60}")
    print(f"PERSONALITY-INTEGRATED GOAL EXECUTION")
    print(f"{'='*60}\n")

    # 1. Получить профиль
    engine = get_personality_engine()
    profile = await engine.get_profile(user_id)

    print(f"👤 User Profile:")
    print(f"   - Tone: {profile.preferences.communication_style.tone}")
    print(f"   - Detail: {profile.preferences.communication_style.detail_level}")
    print(f"   - Growth motivation: {profile.motivations.growth}")
    print(f"   - Achievement: {profile.motivations.achievement}")
    print(f"   - Openness: {profile.core_traits.openness}")

    # 2. Contextual memory
    context = await engine.get_contextual_memory(user_id)
    print(f"\n🧠 Context:")
    print(f"   - Emotional tone: {context.emotional_tone_recent}")
    print(f"   - Recent goals: {len(context.recent_goals)}")
    print(f"   - Interaction streak: {context.interaction_streak} days")

    # 3. Decision bias
    goals = [GoalPressure(
        goal_id="example",
        title=goal_title,
        priority="high",
        magnitude=0.7,
        risk_tolerance=profile.core_traits.openness
    )]

    bias = await evaluate_with_personality(user_id, goals)
    print(f"\n🎯 Decision Bias:")
    print(f"   - Depth: {bias.depth}")
    print(f"   - Speed: {bias.speed}")
    print(f"   - LLM Profile: {bias.llm_profile}")
    print(f"   - Risk tolerance: {bias.risk_tolerance}")
    print(f"   - Communication: {bias.tone}, {bias.detail_level}")
    print(f"   - Value alignment: {bias.value_alignment}")

    # 4. Agent prompts
    agent_prompts = await get_all_personality_aware_prompts(user_id)
    print(f"\n🤖 Agent Prompts:")
    print(f"   - Supervisor: {len(agent_prompts['SUPERVISOR'])} chars")
    print(f"   - Coder: {len(agent_prompts['CODER'])} chars")
    print(f"   - Coach: {len(agent_prompts['COACH'])} chars")

    # 5. Conflict check
    detector = get_goal_conflict_detector()
    # conflicts = await detector.check_goal_conflicts(goal_id)
    print(f"\n⚠️  Conflict Detection: Ready")

    print(f"\n{'='*60}")
    print(f"✅ PERSONALITY INTEGRATED")
    print(f"{'='*60}\n")

    return {
        "profile": profile.dict(),
        "bias": bias.dict(),
        "agent_prompts": list(agent_prompts.keys())
    }


if __name__ == "__main__":
    # Test
    import asyncio

    async def test():
        await full_personality_integration_example(
            goal_title="Изучить Temporal.io",
            user_id="test-user-123"
        )

    asyncio.run(test())
