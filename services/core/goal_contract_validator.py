"""
Goal Contract Validator - v3.0
Formalized constraints on LLM behavior for goal execution
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from models import Goal
from sqlalchemy.orm import Session


class GoalContractValidator:
    """Валидатор и контроллер Goal Contracts - ограничений на поведение LLM"""

    # Доступные действия
    VALID_ACTIONS = [
        "decompose",          # Декомпозиция на подцели
        "spawn_subgoal",      # Создание одной подцели
        "execute",            # Выполнение через агентов
        "evaluate",           # Оценка выполнения
        "spawn_meta_goal",    # Создание мета-цели
        "external_execution", # Выполнение внешними скриптами
        "mutate",             # Мутация цели
        "freeze",             # Заморозка цели
    ]

    # Режимы оценки
    EVALUATION_MODES = ["binary", "scalar", "trend"]

    # Статусы мутации
    MUTATION_STATUSES = ["active", "frozen", "mutated", "deprecated"]

    @staticmethod
    def create_default_contract(goal_type: str = "achievable", depth_level: int = 0) -> Dict:
        """
        Создает дефолтный контракт для типа цели

        Args:
            goal_type: Тип цели
            depth_level: Уровень глубины (для адаптации контракта)

        Returns:
            Дефолтный контракт
        """
        # Mission-level goals (L0) need decomposition capability
        if depth_level == 0:
            mission_continuous = {
                "allowed_actions": ["decompose", "spawn_subgoal", "execute", "evaluate"],
                "forbidden": ["spawn_meta_goal"],
                "max_depth": 2,
                "max_subgoals": 5,
                "evaluation_mode": "trend",
                "timeout_seconds": 600,
                "resource_limits": {
                    "max_tokens": 50000,
                    "max_api_calls": 20
                }
            }
            mission_directional = {
                "allowed_actions": ["decompose", "spawn_subgoal", "evaluate"],
                "forbidden": ["spawn_meta_goal", "execute"],
                "max_depth": 2,
                "max_subgoals": 5,
                "evaluation_mode": "scalar",
                "timeout_seconds": 0,  # No timeout
                "resource_limits": {
                    "max_tokens": 10000,
                    "max_api_calls": 5
                }
            }
        else:
            # Non-mission continuous: only execution + evaluation
            mission_continuous = None
            mission_directional = None

        defaults = {
            "achievable": {
                "allowed_actions": ["decompose", "spawn_subgoal", "execute", "evaluate"],
                "forbidden": ["spawn_meta_goal"],
                "max_depth": 3,
                "max_subgoals": 7,
                "evaluation_mode": "binary",
                "timeout_seconds": 300,
                "resource_limits": {
                    "max_tokens": 100000,
                    "max_api_calls": 50
                }
            },
            "continuous": mission_continuous or {
                "allowed_actions": ["execute", "evaluate"],
                "forbidden": ["decompose", "spawn_meta_goal"],
                "max_depth": 0,
                "max_subgoals": 0,
                "evaluation_mode": "trend",
                "timeout_seconds": 600,
                "resource_limits": {
                    "max_tokens": 50000,
                    "max_api_calls": 20
                }
            },
            "directional": mission_directional or {
                "allowed_actions": ["spawn_subgoal"],
                "forbidden": ["decompose", "execute", "evaluate", "spawn_meta_goal"],
                "max_depth": 1,
                "max_subgoals": 3,
                "evaluation_mode": "scalar",
                "timeout_seconds": 0,  # No timeout
                "resource_limits": {
                    "max_tokens": 10000,
                    "max_api_calls": 5
                }
            },
            "exploratory": {
                "allowed_actions": ["decompose", "spawn_subgoal", "execute", "evaluate"],
                "forbidden": ["spawn_meta_goal"],
                "max_depth": 2,
                "max_subgoals": 5,
                "evaluation_mode": "binary",
                "timeout_seconds": 900,  # 15 min
                "resource_limits": {
                    "max_tokens": 200000,
                    "max_api_calls": 100
                }
            },
            "meta": {
                "allowed_actions": ["decompose", "spawn_subgoal", "evaluate", "mutate"],
                "forbidden": ["execute", "external_execution"],
                "max_depth": 2,
                "max_subgoals": 3,
                "evaluation_mode": "scalar",
                "timeout_seconds": 1200,  # 20 min
                "resource_limits": {
                    "max_tokens": 150000,
                    "max_api_calls": 70
                }
            }
        }

        # === CRITICAL: NO FALLBACK CONTRACTS ===
        # Система больше не может тихо деградировать
        # Если тип цели не определен явно - лучше упасть, чем плодить pending
        if goal_type not in defaults:
            valid_types = ", ".join(defaults.keys())
            raise ValueError(
                f"Goal type '{goal_type}' has no contract defined. "
                f"Valid types: {valid_types}. "
                f"Refusing to create goal without formal contract."
            )

        return defaults[goal_type]

    @staticmethod
    def validate_contract(contract: Dict) -> tuple[bool, Optional[str]]:
        """
        Валидирует структуру контракта

        Args:
            contract: Контракт для валидации

        Returns:
            (is_valid, error_message)
        """
        if not contract:
            return True, None  # Пустой контракт валиден

        # Проверяем allowed_actions
        if "allowed_actions" in contract:
            if not isinstance(contract["allowed_actions"], list):
                return False, "allowed_actions must be a list"

            for action in contract["allowed_actions"]:
                if action not in GoalContractValidator.VALID_ACTIONS:
                    return False, f"Invalid action: {action}"

        # Проверяем forbidden
        if "forbidden" in contract:
            if not isinstance(contract["forbidden"], list):
                return False, "forbidden must be a list"

            for action in contract["forbidden"]:
                if action not in GoalContractValidator.VALID_ACTIONS:
                    return False, f"Invalid forbidden action: {action}"

        # Проверяем пересечения
        if "allowed_actions" in contract and "forbidden" in contract:
            intersection = set(contract["allowed_actions"]) & set(contract["forbidden"])
            if intersection:
                return False, f"Actions in both allowed and forbidden: {intersection}"

        # Проверяем max_depth
        if "max_depth" in contract:
            if not isinstance(contract["max_depth"], int) or contract["max_depth"] < 0:
                return False, "max_depth must be non-negative integer"

        # Проверяем max_subgoals
        if "max_subgoals" in contract:
            if not isinstance(contract["max_subgoals"], int) or contract["max_subgoals"] < 0:
                return False, "max_subgoals must be non-negative integer"

        # Проверяем evaluation_mode
        if "evaluation_mode" in contract:
            if contract["evaluation_mode"] not in GoalContractValidator.EVALUATION_MODES:
                return False, f"Invalid evaluation_mode: {contract['evaluation_mode']}"

        # Проверяем timeout_seconds
        if "timeout_seconds" in contract:
            if not isinstance(contract["timeout_seconds"], int) or contract["timeout_seconds"] < 0:
                return False, "timeout_seconds must be non-negative integer"

        # Проверяем resource_limits
        if "resource_limits" in contract:
            if not isinstance(contract["resource_limits"], dict):
                return False, "resource_limits must be a dict"

            for key, value in contract["resource_limits"].items():
                if key not in ["max_tokens", "max_api_calls"]:
                    continue
                if not isinstance(value, int) or value < 0:
                    return False, f"resource_limits.{key} must be non-negative integer"

        return True, None

    @staticmethod
    def can_execute_action(goal: Goal, action: str) -> tuple[bool, Optional[str]]:
        """
        Проверяет, разрешено ли действие для цели

        Args:
            goal: Цель
            action: Действие

        Returns:
            (is_allowed, reason)
        """
        # Если цель заморожена или deprecated
        mutation_status = getattr(goal, 'mutation_status', 'active')
        if mutation_status in ["frozen", "deprecated"]:
            return False, f"Goal is {mutation_status}"

        # Если нет контракта - разрешаем все
        goal_contract = getattr(goal, 'goal_contract', None)
        if not goal_contract:
            return True, None

        # Проверяем forbidden
        forbidden = goal_contract.get("forbidden", [])
        if action in forbidden:
            return False, f"Action '{action}' is forbidden by goal contract"

        # Проверяем allowed_actions
        allowed = goal_contract.get("allowed_actions", [])
        if allowed and action not in allowed:
            return False, f"Action '{action}' not in allowed_actions"

        return True, None

    @staticmethod
    def check_depth_limit(goal: Goal, current_depth: int) -> tuple[bool, Optional[str]]:
        """
        Проверяет лимит глубины

        Args:
            goal: Цель
            current_depth: Текущая глубина

        Returns:
            (within_limit, reason)
        """
        if not goal.goal_contract:
            return True, None

        max_depth = goal.goal_contract.get("max_depth", 10)
        if current_depth >= max_depth:
            return False, f"Max depth {max_depth} reached"

        return True, None

    @staticmethod
    def check_subgoals_limit(goal: Goal, current_count: int) -> tuple[bool, Optional[str]]:
        """
        Проверяет лимит подцелей

        Args:
            goal: Цель
            current_count: Текущее количество подцелей

        Returns:
            (within_limit, reason)
        """
        if not goal.goal_contract:
            return True, None

        max_subgoals = goal.goal_contract.get("max_subgoals", 100)
        if current_count >= max_subgoals:
            return False, f"Max subgoals {max_subgoals} reached"

        return True, None

    @staticmethod
    def get_evaluation_mode(goal: Goal) -> str:
        """
        Возвращает режим оценки цели

        Args:
            goal: Цель

        Returns:
            Режим оценки (binary|scalar|trend)
        """
        if not goal.goal_contract:
            return "binary"

        return goal.goal_contract.get("evaluation_mode", "binary")


# Singleton instance
goal_contract_validator = GoalContractValidator()
