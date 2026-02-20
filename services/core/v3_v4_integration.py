from logging_config import get_logger
logger = get_logger(__name__)

"""
v3/v4 Integration Example - MemorySignal в реальном executor

Показывает как интегрировать MemorySignal в существующий GoalExecutor
без переписывания всей логики.
"""

from memory_signal import MemorySignal, MemoryRegistry, memory_registry
from memory_generator import MemorySignalGenerator, memory_generator
from decision_field import DecisionField, DecisionFieldInput, ExecutionBias, GoalPressure
from typing import Optional


class V3ExecutorWithV4Memory:
    """
    Пример интеграции v3 executor с v4 MemorySignal.

    Ключевые точки интеграции:
        1. Перед выполнением: compute bias
        2. После ошибки: generate signal
        3. Каждый цикл: decay memory
    """

    def __init__(self):
        self.memory_registry = memory_registry
        self.memory_generator = memory_generator

    def prepare_execution_context(
        self,
        goal_title: str,
        goal_priority: str = "medium",
        pressure_direction: Optional[list[str]] = None,
        pressure_magnitude: float = 0.5
    ) -> dict:
        """
        Подготовить контекст исполнения с учетом bias.

        Это ОДНА точка где v4 влияет на v3.
        """
        # Шаг 1: Собрать входные данные для DecisionField
        goal_pressure = GoalPressure(
            goal_id="current",
            title=goal_title,
            priority=goal_priority,
            direction=pressure_direction or [],
            magnitude=pressure_magnitude
        )

        # TODO: получить реальные constraints и system_state
        constraints = []  # из системы
        memory = self.memory_registry.get_active()

        # Шаг 2: Вычислить bias
        field_input = DecisionFieldInput(
            goals=[goal_pressure],
            constraints=constraints,
            memory=memory,
            system_state=None  # или реальный SystemState
        )

        bias = DecisionField.evaluate(field_input)

        # Шаг 3: Применить bias к execution context
        execution_context = {
            # v3 параметры
            "goal_title": goal_title,
            "goal_priority": goal_priority,

            # v4 bias (ОДНО место влияния!)
            "prefer_skills": bias.prefer_skills,
            "avoid_skills": bias.avoid_skills,
            "depth": bias.depth,
            "speed": bias.speed,
            "risk_tolerance": bias.risk_tolerance,
            "llm_profile": bias.llm_profile,
        }

        return execution_context

    def handle_execution_failure(
        self,
        skill_name: str,
        error: str,
        retries: int
    ):
        """
        Обработать ошибку исполнения.

        Генерирует MemorySignal который повлияет на будущие решения.
        """
        # Генерируем сигнал от ретрая
        if retries > 3:
            self.memory_generator.from_task_retry(
                task_name=skill_name,
                retries=retries,
                skill_name=skill_name
            )

        # Генерируем сигнал от ошибки
        self.memory_generator.from_executor_failure(
            skill_name=skill_name,
            error=error
        )

        logger.info(f"✅ Generated memory signals for failure: {skill_name}")

    def handle_high_cost(
        self,
        skill_name: str,
        actual_cost: float,
        expected_cost: float
    ):
        """
        Обработать перерасход ресурсов.
        """
        signal = self.memory_generator.from_high_cost(
            skill_name=skill_name,
            actual_cost=actual_cost,
            expected_cost=expected_cost
        )

        if signal:
            logger.info(f"✅ Generated high_cost signal: {skill_name} (ratio: {actual_cost/expected_cost:.2f})")

    def handle_manual_override(
        self,
        goal_id: str,
        override_type: str
    ):
        """
        Обработать ручное вмешательство пользователя.
        """
        self.memory_generator.from_manual_override(
            target=goal_id,
            override_type=override_type
        )

        logger.info(f"✅ Generated manual_override signal: {goal_id}")

    def decay_memory(self):
        """
        Вызывается каждый цикл планирования.

        Уменьшает TTL и удаляет истекшие сигналы.
        """
        from decision_field import decay_memory_signals
        decay_memory_signals(self.memory_registry)

        summary = self.memory_registry.summary()
        logger.info(f"📊 Memory signals: {summary['total_signals']} active")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """
    Пример использования v3+v4 гибрида.
    """
    executor = V3ExecutorWithV4Memory()

    # === Шаг 1: Подготовка контекста ===
    logger.info("\n=== Шаг 1: Prepare execution context ===")
    ctx = executor.prepare_execution_context(
        goal_title="Explore atmospheric electricity",
        goal_priority="high",
        pressure_direction=["exploration", "knowledge"],
        pressure_magnitude=0.7
    )

    logger.info(f"Execution bias:")
    logger.info(f"  Prefer skills: {ctx['prefer_skills']}")
    logger.info(f"  Avoid skills: {ctx['avoid_skills']}")
    logger.info(f"  Depth: {ctx['depth']}")
    logger.info(f"  Speed: {ctx['speed']}")
    logger.info(f"  LLM profile: {ctx['llm_profile']}")

    # === Шаг 2: Симуляция ошибки ===
    logger.info("\n=== Шаг 2: Handle failure ===")
    executor.handle_execution_failure(
        skill_name="web_research",
        error="timeout after 120s",
        retries=4
    )

    # === Шаг 3: Симуляция перерасхода ===
    logger.info("\n=== Шаг 3: Handle high cost ===")
    executor.handle_high_cost(
        skill_name="deep_analysis",
        actual_cost=150.0,
        expected_cost=50.0
    )

    # === Шаг 4: Decay ===
    logger.info("\n=== Шаг 4: Decay memory ===")
    executor.decay_memory()

    # === Шаг 5: Следующий цикл - bias изменился! ===
    logger.info("\n=== Шаг 5: Next cycle - bias influenced by memory ===")
    ctx2 = executor.prepare_execution_context(
        goal_title="Explore atmospheric electricity",
        goal_priority="high"
    )

    logger.info(f"NEW Execution bias (influenced by memory):")
    logger.info(f"  Prefer skills: {ctx2['prefer_skills']}")
    logger.info(f"  Avoid skills: {ctx2['avoid_skills']}")
    logger.info(f"  Depth: {ctx2['depth']}")
    logger.info(f"  Speed: {ctx2['speed']}")

    logger.info("\n✅ Memory affected the bias!")


# ============================================================================
# TESTS
# ============================================================================

def test_memory_signal():
    """Тест MemorySignal"""
    from memory_signal import MemorySignal, MemoryRegistry

    registry = MemoryRegistry()

    # Создаем сигнал
    signal = MemorySignal(
        type="recent_failure",
        target="web_research",
        intensity=0.7,
        ttl=5
    )

    registry.add(signal)

    assert len(registry.get_active()) == 1
    logger.info("✅ MemorySignal created and added")

    # Decay
    signal.decay()
    assert signal.ttl == 4
    logger.info("✅ MemorySignal decay works")

    # Полный decay
    for _ in range(4):
        signal.decay()

    assert signal.is_expired()
    assert len(registry.get_active()) == 0
    logger.info("✅ MemorySignal expired and removed")


def test_decision_field():
    """Тест DecisionField"""
    from decision_field import DecisionField, DecisionFieldInput, GoalPressure

    goals = [
        GoalPressure(
            goal_id="G1",
            title="Explore X",
            priority="high",
            direction=["exploration"],
            magnitude=0.7,
            risk_tolerance=0.8
        )
    ]

    input_data = DecisionFieldInput(
        goals=goals,
        constraints=[],
        memory=[],
        system_state=None
    )

    bias = DecisionField.evaluate(input_data)

    assert bias.depth == "shallow"  # Высокое давление
    assert "analyze" in bias.prefer_skills or "explore" in bias.prefer_skills
    logger.info("✅ DecisionField works")
    logger.info(f"  Depth: {bias.depth}")
    logger.info(f"  LLM profile: {bias.llm_profile}")


def test_memory_bias():
    """Тест влияния memory на bias"""
    from memory_signal import MemorySignal, MemoryRegistry
    from decision_field import DecisionField, DecisionFieldInput, ExecutionBias, GoalPressure

    registry = MemoryRegistry()

    # Базовый bias без memory - с prefer_skills который включает web_research
    goals = [
        GoalPressure(
            goal_id="G1",
            title="Explore X",
            priority="high",
            direction=["exploration"],
            magnitude=0.7,
            bias={"prefer_skills": ["web_research", "analyze"]}
        )
    ]

    bias_no_memory = DecisionField.evaluate(
        DecisionFieldInput(
            goals=goals,
            constraints=[],
            memory=[],
            system_state=None
        )
    )

    logger.info(f"\nBias WITHOUT memory:")
    logger.info(f"  Prefer: {bias_no_memory.prefer_skills}")
    logger.info(f"  Avoid: {bias_no_memory.avoid_skills}")

    # Добавляем memory: web_research failed
    registry.add(MemorySignal(
        type="recent_failure",
        target="web_research",
        intensity=0.7,
        ttl=5
    ))

    bias_with_memory = DecisionField.evaluate(
        DecisionFieldInput(
            goals=goals,
            constraints=[],
            memory=registry.get_active(),
            system_state=None
        )
    )

    logger.info(f"\nBias WITH memory (web_research failed):")
    logger.info(f"  Prefer: {bias_with_memory.prefer_skills}")
    logger.info(f"  Avoid: {bias_with_memory.avoid_skills}")
    logger.info(f"  Risk tolerance: {bias_with_memory.risk_tolerance:.2f}")

    # Проверяем что web_research был удален из prefer
    assert "web_research" not in bias_with_memory.prefer_skills, \
        "web_research should be removed from prefer_skills"
    # И добавлен в avoid
    assert "web_research" in bias_with_memory.avoid_skills, \
        "web_research should be in avoid_skills"
    # Risk tolerance снизился
    assert bias_with_memory.risk_tolerance < bias_no_memory.risk_tolerance, \
        "Risk tolerance should decrease after failure"

    logger.info("✅ Memory successfully affected bias")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MemorySignal v4 - Integration Tests")
    logger.info("=" * 60)

    test_memory_signal()
    test_decision_field()
    test_memory_bias()

    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE: Real-world usage")
    logger.info("=" * 60)

    example_usage()

    logger.info("\n✅ All tests passed!")
