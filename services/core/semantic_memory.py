"""
SEMANTIC MEMORY - v3.0
Извлечение и хранение паттернов принятия решений
Memory ≠ Logs - это не просто логи, а извлеченные знания
"""
import uuid
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
from sqlalchemy import select, func
from database import AsyncSessionLocal
from models import Goal
from agent_graph import app_graph


class SemanticMemory:
    """
    Семантическая память - хранит извлеченные паттерны

    Types of patterns:
    - success_patterns: Что сработало
    - failure_patterns: Что не сработало и почему
    - decomposition_patterns: Какие паттерны декомпозиции эффективны
    - agent_effectiveness: Какие agent + model комбинации работают
    - domain_patterns: Специфичные для доменов паттерны

    Отличие от logs:
    - Logs: "Запустили agent X в 12:00"
    - Memory: "Agent X хорошо работает для domain Y при условиях Z"
    """

    async def store_pattern(
        self,
        pattern_type: str,
        content: Dict,
        source_goal_id: str,
        confidence: float = 0.5
    ) -> str:
        """
        Сохраняет паттерн в семантическую память

        Args:
            pattern_type: Тип паттерна
            content: Содержимое паттерна
            source_goal_id: ID цели из которой извлечен паттерн
            confidence: Уверенность 0.0-1.0

        Returns:
            ID созданного паттерна
        """
        # Для семантической памяти используем таблицу Thoughts
        # но с категорией "pattern"

        from models import Thought

        async with AsyncSessionLocal() as db:
            thought = Thought(
                content=f"{pattern_type}: {content}",
                source=source_goal_id,
                status="active" if confidence > 0.5 else "tentative"
            )
            db.add(thought)
            await db.commit()
            await db.refresh(thought)

            return str(thought.id)

    async def extract_success_pattern(self, goal_id: str, reflection: Dict) -> Dict:
        """
        Извлекает паттерн успеха из выполненной цели

        Args:
            goal_id: ID выполненной цели
            reflection: Результат рефлексии от GoalReflector

        Returns:
            Извлеченный паттерн
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

        # Формируем паттерн успеха
        success_pattern = {
            "pattern_type": "success",
            "goal_type": goal.goal_type,
            "depth_level": goal.depth_level,
            "domains": goal.domains,
            "success_factors": reflection.get("success_factors", []),
            "lessons_learned": reflection.get("lessons_learned", []),
            "patterns": reflection.get("patterns", []),
            "extracted_at": datetime.now().isoformat(),
            "source_goal_id": goal_id
        }

        # Сохраняем в память
        pattern_id = await self.store_pattern(
            "success_pattern",
            success_pattern,
            goal_id,
            confidence=0.8
        )

        return {
            "pattern_id": pattern_id,
            "pattern": success_pattern
        }

    async def extract_failure_pattern(self, goal_id: str, reflection: Dict) -> Dict:
        """
        Извлекает паттерн неудачи из провальной цели

        Args:
            goal_id: ID цели
            reflection: Результат рефлексии

        Returns:
            Извлеченный паттерн
        """
        async with AsyncSessionLocal() as db:
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return {"error": "Goal not found"}

        # Формируем паттерн неудачи
        failure_pattern = {
            "pattern_type": "failure",
            "goal_type": goal.goal_type,
            "depth_level": goal.depth_level,
            "domains": goal.domains,
            "root_causes": reflection.get("root_causes", []),
            "mistakes": reflection.get("mistakes", []),
            "missing_resources": reflection.get("missing_resources", []),
            "extracted_at": datetime.now().isoformat(),
            "source_goal_id": goal_id
        }

        # Сохраняем в память
        pattern_id = await self.store_pattern(
            "failure_pattern",
            failure_pattern,
            goal_id,
            confidence=0.7
        )

        return {
            "pattern_id": pattern_id,
            "pattern": failure_pattern
        }

    async def extract_decomposition_pattern(
        self,
        parent_goal: Goal,
        subgoals: List[Goal]
    ) -> Dict:
        """
        Извлекает паттерн декомпозиции

        Анализирует:
        - Сколько подцелей создано
        - Какие типы подцелей
        - Какие домены покрыты
        - Насколько это оказалось эффективным
        """
        decomposition_pattern = {
            "pattern_type": "decomposition",
            "parent_goal_type": parent_goal.goal_type,
            "parent_depth": parent_goal.depth_level,
            "parent_domains": parent_goal.domains,
            "subgoals_count": len(subgoals),
            "subgoals_types": [sg.goal_type for sg in subgoals],
            "subgoals_domains": list(set([d for sg in subgoals for d in (sg.domains or [])])),
            "depth_distribution": [sg.depth_level for sg in subgoals],
            "extracted_at": datetime.now().isoformat()
        }

        # Сохраняем
        pattern_id = await self.store_pattern(
            "decomposition_pattern",
            decomposition_pattern,
            str(parent_goal.id),
            confidence=0.6
        )

        return {
            "pattern_id": pattern_id,
            "pattern": decomposition_pattern
        }

    async def track_agent_effectiveness(
        self,
        agent_role: str,
        model_name: str,
        task_type: str,
        success: bool,
        duration_ms: float,
        context: Dict
    ) -> str:
        """
        Отслеживает эффективность агента + модели

        Examples:
        - "Coder + gpt-4 хорошо работает для refactoring"
        - "Researcher + claude-opus лучше для analysis"
        """
        effectiveness_pattern = {
            "pattern_type": "agent_effectiveness",
            "agent_role": agent_role,
            "model_name": model_name,
            "task_type": task_type,
            "success": success,
            "duration_ms": duration_ms,
            "context": context,  # domains, goal_type, etc.
            "extracted_at": datetime.now().isoformat()
        }

        # Сохраняем
        pattern_id = await self.store_pattern(
            "agent_effectiveness",
            effectiveness_pattern,
            context.get("goal_id", ""),
            confidence=0.9 if success else 0.4
        )

        return pattern_id

    async def retrieve_similar_patterns(
        self,
        pattern_type: str,
        goal_type: str = None,
        domains: List[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Извлекает похожие паттерны из памяти

        Args:
            pattern_type: Тип паттерна
            goal_type: Тип цели (опционально)
            domains: Домены (опционально)
            limit: Максимум паттернов

        Returns:
            Список похожих паттернов
        """
        from models import Thought

        async with AsyncSessionLocal() as db:
            # Базовый запрос
            stmt = select(Thought).where(
                Thought.content.like(f"{pattern_type}%")
            )

            # Фильтруем по статусу
            stmt = stmt.where(Thought.status == "active")

            # Сортируем по дате (новые сначала)
            stmt = stmt.order_by(Thought.created_at.desc())

            # Лимит
            stmt = stmt.limit(limit * 2)  # Берем больше, потом отфильтруем

            result = await db.execute(stmt)
            thoughts = result.scalars().all()

            # Парсим и фильтруем
            patterns = []
            for thought in thoughts:
                try:
                    import json
                    # content = "success_pattern: {...}"
                    content = thought.content.split(": ", 1)[1]
                    pattern = json.loads(content)

                    # Фильтруем по goal_type если нужно
                    if goal_type and pattern.get("goal_type") != goal_type:
                        continue

                    # Фильтруем по domains если нужно
                    if domains:
                        pattern_domains = pattern.get("domains", [])
                        if not any(d in pattern_domains for d in domains):
                            continue

                    patterns.append({
                        "id": str(thought.id),
                        "pattern": pattern,
                        "created_at": thought.created_at.isoformat()
                    })

                    if len(patterns) >= limit:
                        break

                except:
                    continue

            return patterns

    async def get_recommendations(
        self,
        goal: Goal,
        task_type: str = None
    ) -> Dict:
        """
        Дает рекомендации на основе накопленных паттернов

        Args:
            goal: Цель для которой даем рекомендации
            task_type: Тип задачи

        Returns:
            Рекомендации
        """
        # Извлекаем релевантные паттерны
        success_patterns = await self.retrieve_similar_patterns(
            "success_pattern",
            goal_type=goal.goal_type,
            domains=goal.domains,
            limit=3
        )

        failure_patterns = await self.retrieve_similar_patterns(
            "failure_pattern",
            goal_type=goal.goal_type,
            domains=goal.domains,
            limit=3
        )

        agent_patterns = await self.retrieve_similar_patterns(
            "agent_effectiveness",
            limit=5
        )

        # Формируем рекомендации
        recommendations = {
            "success_factors": list(set([
                factor
                for p in success_patterns
                for factor in p["pattern"].get("success_factors", [])
            ])),
            "pitfalls": list(set([
                pitfall
                for p in failure_patterns
                for pitfall in p["pattern"].get("mistakes", [])
            ])),
            "effective_agents": [
                {
                    "agent": p["pattern"]["agent_role"],
                    "model": p["pattern"]["model_name"],
                    "success_rate": "high" if p["pattern"]["success"] else "low"
                }
                for p in agent_patterns
                if p["pattern"]["success"]
            ]
        }

        return recommendations


# Глобальный экземпляр
semantic_memory = SemanticMemory()
