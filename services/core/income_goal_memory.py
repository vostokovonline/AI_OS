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


def example_income_goal_strategy():
    """
    Пример адаптации стратегии получения дохода с помощью MemorySignal.
    """
    print("=" * 70)
    print("MemorySignal v4 - Income Goal Strategy Adaptation")
    print("=" * 70)

    # === Шаг 1: Начальная стратегия ===
    print("\n📍 Шаг 1: Initial Strategy (No Memory)")
    print("-" * 70)

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

    print(f"Initial execution bias:")
    print(f"  Prefer: {initial_bias.prefer_skills}")
    print(f"  Avoid: {initial_bias.avoid_skills}")
    print(f"  Depth: {initial_bias.depth}")
    print(f"  LLM profile: {initial_bias.llm_profile}")

    # === Шаг 2: Провал - costly MVP development ===
    print("\n💥 Шаг 2: Failure - High Cost MVP Development")
    print("-" * 70)

    # Симулируем ошибку: MVP обошёлся слишком дорого
    memory_generator.from_high_cost(
        skill_name="MVP development",
        actual_cost=500.0,  # часов
        expected_cost=100.0,
        threshold=2.0
    )

    print("Generated MemorySignal: high_cost_low_gain")
    print("  Target: MVP development")
    print("  Actual cost: 500h vs Expected: 100h")
    print("  Effect: System will prefer cheaper strategies")

    # === Шаг 3: Стратегия адаптировалась ===
    print("\n🔄 Шаг 3: Strategy Adapted (With Memory)")
    print("-" * 70)

    adapted_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=memory_registry.get_active(),
            system_state=None
        )
    )

    print(f"Adapted execution bias:")
    print(f"  Prefer: {adapted_bias.prefer_skills}")
    print(f"  Avoid: {adapted_bias.avoid_skills}")
    print(f"  Depth: {adapted_bias.depth} (было {initial_bias.depth})")
    print(f"  Speed: {adapted_bias.speed} (было {initial_bias.speed})")
    print(f"  LLM profile: {adapted_bias.llm_profile} (было {initial_bias.llm_profile})")

    # === Шаг 4: Ещё один провал - sales failed ===
    print("\n💥 Шаг 4: Another Failure - Sales Approach Failed")
    print("-" * 70)

    memory_generator.from_executor_failure(
        skill_name="cold_sales",
        error="No conversions from 100 cold calls",
        error_type="low_conversion"
    )

    print("Generated MemorySignal: recent_failure")
    print("  Target: cold_sales")
    print("  Effect: System will avoid aggressive sales")

    # === Шаг 5: Стратегия снова адаптировалась ===
    print("\n🔄 Шаг 5: Strategy Re-Adapted (With 2 Memories)")
    print("-" * 70)

    re_adapted_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=memory_registry.get_active(),
            system_state=None
        )
    )

    print(f"Re-adapted execution bias:")
    print(f"  Prefer: {re_adapted_bias.prefer_skills}")
    print(f"  Avoid: {re_adapted_bias.avoid_skills}")
    print(f"  Depth: {re_adapted_bias.depth}")
    print(f"  Speed: {re_adapted_bias.speed}")
    print(f"  Risk tolerance: {re_adapted_bias.risk_tolerance:.2f}")

    # === Шаг 6: Decay и восстановление ===
    print("\n⏳ Шаг 6: Memory Decay and Recovery")
    print("-" * 70)

    # Симулируем 5 циклов планирования
    from decision_field import decay_memory_signals

    for i in range(5):
        decay_memory_signals(memory_registry)
        summary = memory_registry.summary()
        print(f"  Cycle {i+1}: {summary['total_signals']} active signals")

    # После decay стратегия становится менее консервативной
    final_bias = DecisionField.evaluate(
        DecisionFieldInput(
            goals=[initial_goal],
            constraints=[],
            memory=memory_registry.get_active(),
            system_state=None
        )
    )

    print(f"\nFinal execution bias (after memory decay):")
    print(f"  Prefer: {final_bias.prefer_skills}")
    print(f"  Depth: {final_bias.depth}")
    print(f"  Risk tolerance: {final_bias.risk_tolerance:.2f}")

    # === Итоги ===
    print("\n" + "=" * 70)
    print("📊 RESULTS: Strategy Adaptation Summary")
    print("=" * 70)
    print("""
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
    print("\n" + "=" * 70)
    print("Real Integration Example: GoalExecutor + MemorySignal")
    print("=" * 70)

    # Импортируем (в реальном коде это будет в начале файла)
    from v3_v4_integration import V3ExecutorWithV4Memory

    executor = V3ExecutorWithV4Memory()

    # === Пример 1: Подготовка контекста для выполнения цели ===
    print("\n📋 Example 1: Prepare execution context")
    print("-" * 70)

    ctx = executor.prepare_execution_context(
        goal_title="Найти первых-paying клиентов",
        goal_priority="high",
        pressure_direction=["exploitation", "legacy"],
        pressure_magnitude=0.7
    )

    print(f"Execution context prepared with bias:")
    print(f"  Prefer skills: {ctx['prefer_skills']}")
    print(f"  Depth: {ctx['depth']}")

    # === Пример 2: Обработка ошибки при выполнении ===
    print("\n❌ Example 2: Handle execution failure")
    print("-" * 70)

    executor.handle_execution_failure(
        skill_name="cold_outreach",
        error="No responses from 50 emails",
        retries=3
    )

    # === Пример 3: Проверка влияния на следующий цикл ===
    print("\n🔄 Example 3: Next cycle - memory affected bias")
    print("-" * 70)

    ctx_after = executor.prepare_execution_context(
        goal_title="Найти первых-paying клиентов",
        goal_priority="high"
    )

    print(f"New execution context (influenced by memory):")
    print(f"  Avoid skills: {ctx_after['avoid_skills']}")
    print(f"  Risk tolerance: {ctx_after['risk_tolerance']:.2f}")

    # === Пример 4: Ручной override пользователя ===
    print("\n👤 Example 4: Manual override by user")
    print("-" * 70)

    executor.handle_manual_override(
        goal_id="3b7c1939-9c5c-4f62-99e7-b790ea569a41",  # L2 goal
        override_type="block"  # Пользователь заблокировал направление
    )

    print("Generated MemorySignal: false_success")
    print("  Effect: System will be more cautious with similar goals")

    # === Пример 5: Decay памяти ===
    print("\n⏳ Example 5: Memory decay")
    print("-" * 70)

    executor.decay_memory()


def generate_real_world_signals():
    """
    Генерирует реальные MemorySignal для текущих целей.
    """
    print("\n" + "=" * 70)
    print("Generating Real-World Memory Signals for Income Goals")
    print("=" * 70)

    # Симулируем несколько реальных сценариев

    # Сценарий 1: Провал "Изучить конкурентов" - слишком дорого
    print("\n💡 Scenario 1: Expensive market research")
    memory_generator.from_high_cost(
        skill_name="deep_market_research",
        actual_cost=80.0,  # часов
        expected_cost=20.0,
        threshold=2.0
    )
    print("✅ Signal generated: high_cost_low_gain")
    print("   → Future: System will prefer shallow analysis")

    # Сценарий 2: Провал "Написать код MVP" - технические проблемы
    print("\n💡 Scenario 2: Technical failure in MVP development")
    memory_generator.from_executor_failure(
        skill_name="MVP_development",
        error="Integration issues with payment API",
        error_type="technical"
    )
    print("✅ Signal generated: recent_failure")
    print("   → Future: System will avoid similar technical tasks")

    # Сценарий 3: Успех "Создать лендинг" - но конверсия низкая
    print("\n💡 Scenario 3: False success - landing created but no sales")
    memory_generator.from_manual_override(
        target="landing_page_optimization",
        override_type="force_complete"
    )
    print("✅ Signal generated: false_success")
    print("   → Future: System will be less aggressive with optimization")

    print("\n" + "=" * 70)
    print("📊 Current Memory State:")
    print("=" * 70)

    summary = memory_registry.summary()
    # Convert datetime objects to strings for JSON serialization
    summary_serializable = {
        k: v.isoformat() if isinstance(v, datetime) else v
        for k, v in summary.items()
    }
    print(json.dumps(summary_serializable, indent=2))


if __name__ == "__main__":
    # Запускаем все примеры
    example_income_goal_strategy()
    example_integration_with_goal_executor()
    generate_real_world_signals()

    print("\n✅ All examples completed!")
    print("\n💡 Key Takeaway:")
    print("   MemorySignal позволяет системе АВТОМАТИЧЕСКИ адаптироваться")
    print("   к ошибкам без изменения кода.")
    print("   Это обучение без обучения - рефлекс, а не интеллект.")
