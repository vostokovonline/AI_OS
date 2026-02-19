"""
MemorySignal v4 - Memory as Pressure, Not Knowledge

MemorySignal временно искажает DecisionField на основе прошлого опыта.
Это НЕ storage. Это след (trace), который самоудаляется.

Principles:
- Memory cannot override GoalPressure
- Memory only weakens or shifts bias
- TTL is mandatory - memory self-deletes
- Goal > Memory always
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime, timedelta
import json


MemorySignalType = Literal[
    "recent_failure",
    "resource_exhaustion",
    "false_success",
    "overfitting",
    "high_cost_low_gain"
]


@dataclass
class MemorySignal:
    """
    Один сигнал памяти - временный след от события.

    Примеры использования:
        - Ошибка при выполнении skill → recent_failure
        - Перерасход ресурсов → resource_exhaustion
        - Успех, но без реального эффекта → false_success
    """
    type: MemorySignalType
    target: Optional[str]      # skill | goal | llm_profile | strategy
    intensity: float            # 0.0 - 1.0, сила влияния
    ttl: int                    # cycles to live, обязательное поле!
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """Валидация параметров"""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be 0..1, got {self.intensity}")
        if self.ttl <= 0:
            raise ValueError(f"ttl must be positive, got {self.ttl}")

    def is_expired(self) -> bool:
        """Проверка истекло ли время жизни"""
        return self.ttl <= 0

    def decay(self):
        """Уменьшает TTL на 1 цикл"""
        self.ttl -= 1

    def to_dict(self) -> dict:
        """Сериализация для хранения/логирования"""
        return {
            "type": self.type,
            "target": self.target,
            "intensity": self.intensity,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemorySignal":
        """Десериализация"""
        return cls(
            type=data["type"],
            target=data.get("target"),
            intensity=data["intensity"],
            ttl=data["ttl"],
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


@dataclass
class MemoryRegistry:
    """
    Реестр активных сигналов памяти.

    Хранит сигналы в памяти (runtime), не в БД.
    Это временный шум, а не постоянное знание.
    """
    signals: list[MemorySignal] = field(default_factory=list)

    def add(self, signal: MemorySignal):
        """Добавить новый сигнал"""
        self.signals.append(signal)

    def decay_all(self):
        """Уменьшить TTL всех сигналов и удалить истекшие"""
        active_signals = []
        for signal in self.signals:
            signal.decay()
            if not signal.is_expired():
                active_signals.append(signal)
        self.signals = active_signals

    def get_active(self) -> list[MemorySignal]:
        """Получить только активные сигналы"""
        return [s for s in self.signals if not s.is_expired()]

    def get_by_target(self, target: str) -> list[MemorySignal]:
        """Получить сигналы для конкретной цели"""
        return [s for s in self.signals if s.target == target and not s.is_expired()]

    def get_by_type(self, signal_type: MemorySignalType) -> list[MemorySignal]:
        """Получить сигналы конкретного типа"""
        return [s for s in self.signals if s.type == signal_type and not s.is_expired()]

    def clear(self):
        """Полная очистка (для тестов или сброса)"""
        self.signals = []

    def summary(self) -> dict:
        """Статистика для мониторинга"""
        active = self.get_active()
        return {
            "total_signals": len(active),
            "by_type": {t: len([s for s in active if s.type == t])
                       for t in MemorySignalType.__args__},
            "avg_intensity": sum(s.intensity for s in active) / len(active) if active else 0,
            "oldest_signal": min((s.created_at for s in active), default=None),
            "newest_signal": max((s.created_at for s in active), default=None)
        }


# Глобальный инстанс (синглтон для runtime)
memory_registry = MemoryRegistry()
