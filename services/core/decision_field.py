"""
DecisionField v4 - Field-Driven Bias Generation

DecisionField НЕ принимает решений.
Он искажает среду исполнения на основе давления целей и памяти.

Pressure + Memory → Bias
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from memory_signal import MemorySignal, MemoryRegistry, memory_registry


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class GoalPressure:
    """
    Расширенная цель с давлением.

    Совместимо с существующими Goal - добавка, не замена.
    """
    goal_id: str
    title: str
    priority: str  # high, medium, low

    # v4: Pressure fields (optional, для обратной совместимости)
    direction: list[str] = field(default_factory=list)  # exploration, exploitation, knowledge, etc.
    magnitude: float = 0.5          # 0..1, сила давления
    risk_tolerance: float = 0.5      # 0..1
    bias: dict = field(default_factory=dict)  # prefer_skills, avoid_skills


@dataclass
class Constraint:
    """
    Ограничение системы.
    """
    type: str  # resource_limit, time_limit, safety, etc.
    target: Optional[str] = None
    severity: float = 0.5  # 0..1


@dataclass
class SystemState:
    """
    Текущее состояние системы.
    """
    total_active_goals: int = 0
    resource_usage: float = 0.0  # 0..1
    error_rate: float = 0.0       # 0..1
    recent_failures: int = 0


@dataclass
class DecisionFieldInput:
    """
    Входные данные для DecisionField.evaluate()
    """
    goals: list[GoalPressure]
    constraints: list[Constraint]
    memory: list[MemorySignal]
    system_state: SystemState


@dataclass
class ExecutionBias:
    """
    Результат DecisionField.evaluate()

    Это НЕ план.
    Это искажение среды исполнения.
    """
    prefer_skills: list[str] = field(default_factory=list)
    avoid_skills: list[str] = field(default_factory=list)

    depth: Literal["shallow", "medium", "deep"] = "medium"
    speed: Literal["fast", "normal", "slow"] = "normal"

    risk_tolerance: float = 0.5          # 0..1
    retry_aggressiveness: float = 0.5    # 0..1

    llm_profile: Literal[
        "fast", "balanced", "deep", "creative", "paranoid"
    ] = "balanced"

    # Для отладки
    _debug_info: dict = field(default_factory=dict)


# ============================================================================
# DECISION FIELD
# ============================================================================

class DecisionField:
    """
    Вычисляет ExecutionBias из давления целей и памяти.

    Принципы:
        - Memory НЕ может отменить GoalPressure
        - Memory только ослабляет или смещает bias
        - Goal > Memory всегда
    """

    @staticmethod
    def evaluate(input: DecisionFieldInput) -> ExecutionBias:
        """
        Главный метод - вычислить bias из поля давления.

        Args:
            input: goals, constraints, memory, system_state

        Returns:
            ExecutionBias - искажение среды исполнения
        """
        # Шаг 1: Базовый bias из GoalPressure
        bias = DecisionField._evaluate_goals(input.goals)

        # Шаг 2: Учесть ограничения
        bias = DecisionField._apply_constraints(bias, input.constraints)

        # Шаг 3: Применить MemorySignal (ослабление/смещение)
        bias = DecisionField._apply_memory(bias, input.memory, input.goals)

        # Шаг 4: Учить состояние системы
        bias = DecisionField._apply_system_state(bias, input.system_state)

        return bias

    @staticmethod
    def _evaluate_goals(goals: list[GoalPressure]) -> ExecutionBias:
        """Вычислить базовый bias из давления целей"""
        prefer = set()
        avoid = set()

        pressure_sum = sum(g.magnitude for g in goals if g.magnitude > 0)
        avg_risk = sum(g.risk_tolerance for g in goals) / len(goals) if goals else 0.5

        exploration = any("exploration" in g.direction for g in goals)
        knowledge = any("knowledge" in g.direction for g in goals)
        exploitation = any("exploitation" in g.direction for g in goals)

        # Определяем предпочитаемые skills на основе direction
        if exploration or knowledge:
            prefer.update(["analyze", "research", "explore"])

        if exploitation:
            prefer.update(["execute", "optimize", "commit"])

        # Применяем явный bias из целей
        for goal in goals:
            if goal.bias:
                prefer.update(goal.bias.get("prefer_skills", []))
                avoid.update(goal.bias.get("avoid_skills", []))

        # Определяем глубину
        if pressure_sum >= 0.7:
            depth = "shallow"  # Высокое давление → быстрые решения
        elif pressure_sum < 0.3:
            depth = "deep"     # Низкое давление → глубокий анализ
        else:
            depth = "medium"

        # Определяем скорость
        speed = "fast" if pressure_sum >= 0.7 else "normal"

        # LLM profile
        if exploration:
            llm_profile = "creative"
        elif pressure_sum > 0.8:
            llm_profile = "fast"
        else:
            llm_profile = "balanced"

        return ExecutionBias(
            prefer_skills=list(prefer),
            avoid_skills=list(avoid),
            depth=depth,
            speed=speed,
            risk_tolerance=avg_risk,
            retry_aggressiveness=0.3 if pressure_sum > 0.7 else 0.6,
            llm_profile=llm_profile,
            _debug_info={"source": "goals", "pressure_sum": pressure_sum}
        )

    @staticmethod
    def _apply_constraints(
        bias: ExecutionBias,
        constraints: list[Constraint]
    ) -> ExecutionBias:
        """Учесть ограничения системы"""
        scarcity = any(c.type == "resource_limit" for c in constraints)
        time_pressure = any(c.type == "time_limit" for c in constraints)

        if scarcity:
            # Нехватка ресурсов → более поверхностно
            bias.depth = "shallow" if bias.depth == "deep" else bias.depth
            bias.speed = "fast" if bias.speed == "normal" else bias.speed
            bias.avoid_skills.append("commit")  # Не делать глубоких изменений

        if time_pressure:
            bias.speed = "fast"
            bias.llm_profile = "fast" if bias.llm_profile == "balanced" else bias.llm_profile

        return bias

    @staticmethod
    def _apply_memory(
        bias: ExecutionBias,
        memory: list[MemorySignal],
        goals: list[GoalPressure]
    ) -> ExecutionBias:
        """
        Применить MemorySignal к bias.

        КРИТИЧНО: Memory НЕ может отменить GoalPressure.
        Memory только ослабляет или смещает.
        """
        if not memory:
            return bias

        # Применяем каждый сигнал
        for m in memory:
            if m.is_expired():
                continue

            # recent_failure → избегать target
            if m.type == "recent_failure":
                if m.target in bias.prefer_skills:
                    bias.prefer_skills.remove(m.target)
                    bias.avoid_skills.append(m.target)

                # Снижаем рискованность
                bias.risk_tolerance = max(0.1, bias.risk_tolerance - (0.2 * m.intensity))

            # resource_exhaustion → снижаем глубину
            if m.type == "resource_exhaustion":
                bias.depth = "shallow" if bias.depth == "deep" else bias.depth
                bias.speed = "fast" if bias.speed != "fast" else bias.speed

            # false_success → снижаем trust
            if m.type == "false_success":
                bias.retry_aggressiveness = max(0.1, bias.retry_aggressiveness - (0.3 * m.intensity))

            # overfitting → меняем стратегию
            if m.type == "overfitting":
                if bias.llm_profile == "balanced":
                    bias.llm_profile = "creative"
                elif bias.llm_profile == "fast":
                    bias.llm_profile = "balanced"

                # Избегаем перегруженных skills
                if m.target in bias.prefer_skills:
                    bias.prefer_skills.remove(m.target)

            # high_cost_low_gain → ускоряемся
            if m.type == "high_cost_low_gain":
                bias.depth = "shallow"
                bias.speed = "fast"
                bias.llm_profile = "fast"

        return bias

    @staticmethod
    def _apply_system_state(
        bias: ExecutionBias,
        state: Optional[SystemState]
    ) -> ExecutionBias:
        """Учесть текущее состояние системы"""
        if state is None:
            return bias

        if state.resource_usage > 0.8:
            bias.depth = "shallow"
            bias.speed = "fast"

        if state.error_rate > 0.3:
            bias.llm_profile = "paranoid"  # Осторожнее
            bias.retry_aggressiveness = max(0.1, bias.retry_aggressiveness - 0.3)

        if state.recent_failures > 3:
            bias.risk_tolerance = max(0.1, bias.risk_tolerance - 0.2)

        return bias


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def apply_memory_to_bias(bias: ExecutionBias, memory: list[MemorySignal]) -> ExecutionBias:
    """
    Отдельная функция для применения memory к существующему bias.

    Используется для постепенной интеграции с v3.
    """
    return DecisionField._apply_memory(bias, memory, goals=[])


def decay_memory_signals(registry: MemoryRegistry):
    """
    Уменьшить TTL всех сигналов и удалить истекшие.

    Вызывается каждый цикл планирования.
    """
    registry.decay_all()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def evaluate_decision_field_simple(
    goals: list[GoalPressure],
    constraints: list[Constraint],
    system_state: Optional[SystemState] = None
) -> ExecutionBias:
    """
    Упрощённая версия без MemorySignal.

    Для обратной совместимости с v3.
    """
    return DecisionField.evaluate(
        DecisionFieldInput(
            goals=goals,
            constraints=constraints,
            memory=[],
            system_state=system_state or SystemState()
        )
    )
