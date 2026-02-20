"""
MemorySignal Integration for Income Goal

Пример использования MemorySignal v4 для адаптации стратегии
получения устойчивого дохода.
"""

from memory_signal import MemorySignal, MemoryRegistry, memory_registry
from memory_generator import MemorySignalGenerator, memory_generator
from decision_field import DecisionField, DecisionFieldInput, GoalPressure, ExecutionBias
from datetime import datetime
import json

# Centralized logging
from logging_config import get_logger

logger = get_logger(__name__)


def example_income_goal_strategy():
    """
    Пример адаптации стратегии получения дохода с помощью MemorySignal.
    """
    logger.info("separator_70")
    logger.info("memory_signal_income_goal_demo")
    logger.info("separator_70")

    # === Шаг 1: Начальная стратегия ===
    logger.info("step_1_initial_strategy")
    logger.info("separator_dash_70")

    initial_goal = GoalPressure(
        goal_id="income_goal",
        title="Получение устойчивого дохода",
        priority="high",
        direction=["exploitation", "legacy"],
        magnitude=0.8,
        risk_tolerance=0.6,
        bias={
            "prefer_skills": ["sales", "marketing", "development"],
            "avoid_skills": []
        }
    )

    initial_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=[],
            system_state=None
        )
    )

    logger.info(f"Initial execution bias:")
    logger.info(f"  Prefer: {initial_bias.prefer_skills}")
    logger.info(f"  Avoid: {initial_bias.avoid_skills}")
    logger.info(f"  Depth: {initial_bias.depth}")
    logger.info(f"  LLM profile: {initial_bias.llm_profile}")

    # === Шаг 2: Провал - costly MVP development ===
    logger.info("\n💥 Шаг 2: Failure - High Cost MVP Development")
    logger.info("separator_dash_70")

    # Симулируем ошибку: MVP обошёлся слишком дорого
    memory_generator.from_high_cost(
        skill_name="MVP development",
        actual_cost=500.0,  # часов
        expected_cost=100.0,
        threshold=2.0
    )

    logger.info("Generated MemorySignal: high_cost_low_gain")
    logger.info("  Target: MVP development")
    logger.info("  Actual cost: 500h vs Expected: 100h")
    logger.info("  Effect: System will prefer cheaper strategies")

    # === Шаг 3: Стратегия адаптировалась ===
    logger.info("\n🔄 Шаг 3: Strategy Adapted (With Memory)")
    logger.info("separator_dash_70")

    adapted_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=memory_registry.get_active(),
            system_state=None
        )
    )

    logger.info(f"Adapted execution bias:")
    logger.info(f"  Prefer: {adapted_bias.prefer_skills}")
    logger.info(f"  Avoid: {adapted_bias.avoid_skills}")
    logger.info(f"  Depth: {adapted_bias.depth} (было {initial_bias.depth})")
    logger.info(f"  Speed: {adapted_bias.speed} (было {initial_bias.speed})")
    logger.info(f"  LLM profile: {adapted_bias.llm_profile} (было {initial_bias.llm_profile})")

    # === Шаг 4: Ещё один провал - sales failed ===
    logger.info("\n💥 Шаг 4: Another Failure - Sales Approach Failed")
    logger.info("separator_dash_70")

    memory_generator.from_executor_failure(
        skill_name="cold_sales",
        error="No conversions from 100 cold calls",
        error_type="low_conversion"
    )

    logger.info("Generated MemorySignal: recent_failure")
    logger.info("  Target: cold_sales")
    logger.info("  Effect: System will avoid aggressive sales")

    # === Шаг 5: Стратегия снова адаптировалась ===
    logger.info("\n🔄 Шаг 5: Strategy Re-Adapted (With 2 Memories)")
    logger.info("separator_dash_70")

    re_adapted_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=memory_registry.get_active(),
            system_state=None
        )
    )

    logger.info(f"Re-adapted execution bias:")
    logger.info(f"  Prefer: {re_adapted_bias.prefer_skills}")
    logger.info(f"  Avoid: {re_adapted_bias.avoid_skills}")
    logger.info(f"  Depth: {re_adapted_bias.depth}")
    logger.info(f"  Speed: {re_adapted_bias.speed}")
    logger.info(f"  Risk tolerance: {re_adapted_bias.risk_tolerance:.2f}")

    # === Шаг 6: Decay и восстановление ===
    logger.info("\n⏳ Шаг 6: Memory Decay and Recovery")
    logger.info("separator_dash_70")

    # Симулируем 5 циклов планирования
    from decision_field import decay_memory_signals

    for i in range(5):
        decay_memory_signals(memory_registry)
        summary = memory_registry.summary()
        logger.info(f"  Cycle {i+1}: {summary['total_signals']} active signals")

    # После decay стратегия становится менее консервативной
    final_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=memory_registry.get_active(),
            system_state=None
        )
    )

    logger.info(f"\nFinal execution bias (after memory decay):")
    logger.info(f"  Prefer: {final_bias.prefer_skills}")
    logger.info(f"  Depth: {final_bias.depth}")
    logger.info(f"  Risk tolerance: {final_bias.risk_tolerance:.2f}")

    # === Итоги ===
    logger.info("\n" + "=" * 70)
    logger.info("📊 RESULTS: Strategy Adaptation Summary")
    logger.info("separator_70")
    logger.info("""
✅ WITHOUT MemorySignal:
   - System repeats same expensive mistakes
   - Needs manual code changes to adapt
   - Accumulates technical debt

✅ WITH MemorySignal:
   - System automatically adapts after failures
   - Becomes more conservative after high-cost errors
   - Avoids failed strategies
   - Gradually recovers as memory decays
   - NO CODE CHANGES NEEDED
    """)


def example_integration_with_goal_executor():
    """
    Пример интеграции MemorySignal с GoalExecutor.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Real Integration Example: GoalExecutor + MemorySignal")
    logger.info("separator_70")

    # Импортируем (в реальном коде это будет в начале файла)
    from v3_v4_integration import V3ExecutorWithV4Memory

    executor = V3ExecutorWithV4Memory()

    # === Пример 1: Подготовка контекста для выполнения цели ===
    logger.info("\n📋 Example 1: Prepare execution context")
    logger.info("separator_dash_70")

    ctx = executor.prepare_execution_context(
        goal_title="Найти первых-paying клиентов",
        goal_priority="high",
        pressure_direction=["exploitation", "legacy"],
        pressure_magnitude=0.7
    )

    logger.info(f"Execution context prepared with bias:")
    logger.info(f"  Prefer skills: {ctx['prefer_skills']}")
    logger.info(f"  Depth: {ctx['depth']}")

    # === Пример 2: Обработка ошибки при выполнении ===
    logger.info("\n❌ Example 2: Handle execution failure")
    logger.info("separator_dash_70")

    executor.handle_execution_failure(
        skill_name="cold_outreach",
        error="No responses from 50 emails",
        retries=3
    )

    # === Пример 3: Проверка влияния на следующий цикл ===
    logger.info("\n🔄 Example 3: Next cycle - memory affected bias")
    logger.info("separator_dash_70")

    ctx_after = executor.prepare_execution_context(
        goal_title="Найти первых-paying клиентов",
        goal_priority="high"
    )

    logger.info(f"New execution context (influenced by memory):")
    logger.info(f"  Avoid skills: {ctx_after['avoid_skills']}")
    logger.info(f"  Risk tolerance: {ctx_after['risk_tolerance']:.2f}")

    # === Пример 4: Ручной override пользователя ===
    logger.info("\n👤 Example 4: Manual override by user")
    logger.info("separator_dash_70")

    executor.handle_manual_override(
        goal_id="3b7c1939-9c5c-4f62-99e7-b790ea569a41",  # L2 goal
        override_type="block"  # Пользователь заблокировал направление
    )

    logger.info("Generated MemorySignal: false_success")
    logger.info("  Effect: System will be more cautious with similar goals")

    # === Пример 5: Decay памяти ===
    logger.info("\n⏳ Example 5: Memory decay")
    logger.info("separator_dash_70")

    executor.decay_memory()


def generate_real_world_signals():
    """
    Генерирует реальные MemorySignal для текущих целей.
    """
    logger.info("\n" + "=" * 70)
    logger.info("Generating Real-World Memory Signals for Income Goals")
    logger.info("separator_70")

    # Симулируем несколько реальных сценариев

    # Сценарий 1: Провал "Изучить конкурентов" - слишком дорого
    logger.info("\n💡 Scenario 1: Expensive market research")
    memory_generator.from_high_cost(
        skill_name="deep_market_research",
        actual_cost=80.0,  # часов
        expected_cost=20.0,
        threshold=2.0
    )
    logger.info("✅ Signal generated: high_cost_low_gain")
    logger.info("   → Future: System will prefer shallow analysis")

    # Сценарий 2: Провал "Написать код MVP" - технические проблемы
    logger.info("\n💡 Scenario 2: Technical failure in MVP development")
    memory_generator.from_executor_failure(
        skill_name="MVP_development",
        error="Integration issues with payment API",
        error_type="technical"
    )
    logger.info("✅ Signal generated: recent_failure")
    logger.info("   → Future: System will avoid similar technical tasks")

    # Сценарий 3: Успех "Создать лендинг" - но конверсия низкая
    logger.info("\n💡 Scenario 3: False success - landing created but no sales")
    memory_generator.from_manual_override(
        target="landing_page_optimization",
        override_type="force_complete"
    )
    logger.info("✅ Signal generated: false_success")
    logger.info("   → Future: System will be less aggressive with optimization")

    logger.info("\n" + "=" * 70)
    logger.info("📊 Current Memory State:")
    logger.info("separator_70")

    summary = memory_registry.summary()
    # Convert datetime objects to strings for JSON serialization
    summary_serializable = {
        k: v.isoformat() if isinstance(v, datetime) else v
        for k, v in summary.items()
    }
    logger.info(json.dumps(summary_serializable, indent=2))


if __name__ == "__main__":
    # Запускаем все примеры
    example_income_goal_strategy()
    example_integration_with_goal_executor()
    generate_real_world_signals()

    logger.info("\n✅ All examples completed!")
    logger.info("\n💡 Key Takeaway:")
    logger.info("   MemorySignal позволяет системе АВТОМАТИЧЕСКИ адаптироваться")
    logger.info("   к ошибкам без изменения кода.")
    logger.info("   Это обучение без обучения - рефлекс, а не интеллект.")
