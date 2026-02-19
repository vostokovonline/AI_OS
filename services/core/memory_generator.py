"""
MemorySignal Generator - Creates signals from executor events

Автоматически генерирует MemorySignal из:
- Падений executor
- Ретраев
- Тайм-аутов
- Отменённых задач
- Ручных override пользователя

❌ НЕ из LLM рассуждений
❌ НЕ из "ощущений" системы
"""

from typing import Optional, Literal
from datetime import datetime
from memory_signal import MemorySignal, MemoryRegistry, memory_registry


class MemorySignalGenerator:
    """
    Генерирует MemorySignal на основе событий исполнения.

    Использование:
        generator = MemorySignalGenerator()
        generator.from_task_retry(task, retries=4)
        generator.from_executor_failure(skill_name, error="timeout")
    """

    def __init__(self, registry: Optional[MemoryRegistry] = None):
        self.registry = registry or memory_registry

    def from_task_retry(
        self,
        task_name: str,
        retries: int,
        max_retries: int = 3,
        skill_name: Optional[str] = None
    ) -> Optional[MemorySignal]:
        """
        Генерирует сигнал при ретраях задачи.

        Правило:
            retries > max_retries * 0.5 → warning (intensity=0.4)
            retries >= max_retries → recent_failure (intensity=0.7)
        """
        if retries < max_retries * 0.5:
            return None

        intensity = 0.4 if retries < max_retries else 0.7
        ttl = 5 if retries < max_retries else 10

        signal = MemorySignal(
            type="recent_failure",
            target=skill_name or task_name,
            intensity=intensity,
            ttl=ttl,
            metadata={
                "reason": "task_retry",
                "retries": retries,
                "max_retries": max_retries
            }
        )

        self.registry.add(signal)
        return signal

    def from_executor_failure(
        self,
        skill_name: str,
        error: str,
        error_type: Optional[str] = None
    ) -> Optional[MemorySignal]:
        """
        Генерирует сигнал при падении executor.

        Типы ошибок:
            - timeout → resource_exhaustion
            - api_error → recent_failure
            - validation_error → false_success
        """
        error_lower = error.lower()

        if "timeout" in error_lower:
            signal_type = "resource_exhaustion"
            ttl = 8
            intensity = 0.6
        elif "api" in error_lower or "http" in error_lower:
            signal_type = "recent_failure"
            ttl = 5
            intensity = 0.5
        elif "validation" in error_lower:
            signal_type = "false_success"
            ttl = 3
            intensity = 0.4
        else:
            signal_type = "recent_failure"
            ttl = 5
            intensity = 0.5

        signal = MemorySignal(
            type=signal_type,
            target=skill_name,
            intensity=intensity,
            ttl=ttl,
            metadata={
                "reason": "executor_failure",
                "error": error[:200],  # Первые 200 символов
                "error_type": error_type
            }
        )

        self.registry.add(signal)
        return signal

    def from_high_cost(
        self,
        skill_name: str,
        actual_cost: float,
        expected_cost: float,
        threshold: float = 2.0
    ) -> Optional[MemorySignal]:
        """
        Генерирует сигнал при перерасходе ресурсов.

        Правило:
            actual_cost > expected_cost * threshold → high_cost_low_gain
        """
        if actual_cost <= expected_cost * threshold:
            return None

        ratio = actual_cost / expected_cost
        intensity = min(0.3 + (ratio - threshold) * 0.1, 0.9)
        ttl = 10

        signal = MemorySignal(
            type="high_cost_low_gain",
            target=skill_name,
            intensity=intensity,
            ttl=ttl,
            metadata={
                "reason": "high_cost",
                "actual_cost": actual_cost,
                "expected_cost": expected_cost,
                "ratio": round(ratio, 2)
            }
        )

        self.registry.add(signal)
        return signal

    def from_repeated_success(
        self,
        skill_name: str,
        success_count: int,
        success_threshold: int = 10
    ) -> Optional[MemorySignal]:
        """
        Генерирует сигнал при повторяющемся успехе (overfitting).

        Правило:
            Один skill слишком часто используется → overfitting
        """
        if success_count < success_threshold:
            return None

        intensity = min(0.3 + (success_count - success_threshold) * 0.05, 0.8)
        ttl = 7

        signal = MemorySignal(
            type="overfitting",
            target=skill_name,
            intensity=intensity,
            ttl=ttl,
            metadata={
                "reason": "repeated_success",
                "success_count": success_count
            }
        )

        self.registry.add(signal)
        return signal

    def from_manual_override(
        self,
        target: str,
        override_type: Literal["block", "force_complete", "change_priority"]
    ) -> MemorySignal:
        """
        Генерирует сигнал при ручном вмешательстве пользователя.

        Ручной override = система не справилась сама
        """
        intensity_map = {
            "block": 0.7,
            "force_complete": 0.5,
            "change_priority": 0.3
        }

        signal = MemorySignal(
            type="false_success",
            target=target,
            intensity=intensity_map[override_type],
            ttl=5,
            metadata={
                "reason": "manual_override",
                "override_type": override_type
            }
        )

        self.registry.add(signal)
        return signal


# Глобальный инстанс
memory_generator = MemorySignalGenerator()
