"""
SEMANTIC MEMORY - v3.0
Излечение и хранение паттернов принятия решений
Memory ≠ Logs - это не просто логи, а извлеченные знания

v3.1: Added Milvus vector search integration
"""
import uuid
import os
import json
import httpx
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from sqlalchemy import select, func, delete
from database import AsyncSessionLocal
from models import Goal, Thought
from agent_graph import app_graph

MEMORY_URL = os.getenv("MEMORY_URL", "http://memory:8001")


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
        from models import Thought

        # 🆕 DEDUPLICATION: Check for similar existing patterns
        existing = await self._find_similar_pattern(pattern_type, content)
        if existing:
            # Update confidence instead of creating duplicate
            return await self._update_pattern_confidence(existing, confidence)

        async with AsyncSessionLocal() as db:
            thought = Thought(
                content=f"{pattern_type}: {content}",
                source=source_goal_id,
                status="active" if confidence > 0.5 else "tentative"
            )
            db.add(thought)
            await db.commit()
            await db.refresh(thought)
            
            pattern_id = str(thought.id)
            
            # 🆕 Also store in Milvus for vector search
            content["confidence"] = confidence
            await self.store_pattern_vector(pattern_type, content, pattern_id)

            return pattern_id

    async def _find_similar_pattern(
        self,
        pattern_type: str,
        content: Dict,
        similarity_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Ищет похожий паттерн в памяти.
        
        Args:
            pattern_type: Тип паттерна
            content: Содержимое для сравнения
            similarity_threshold: Порог схожести (0.0-1.0)
            
        Returns:
            ID похожего паттерна или None
        """
        from models import Thought
        
        # Use vector search for similarity
        text_repr = self._pattern_to_text(pattern_type, content)
        similar = await self.retrieve_similar_patterns_vector(text_repr, limit=3)
        
        for pattern in similar:
            # Check type match
            if pattern.get("pattern_type") != pattern_type:
                continue
            
            # Check key fields similarity
            if self._calculate_similarity(content, pattern) >= similarity_threshold:
                return pattern.get("id")
        
        return None

    def _calculate_similarity(self, content1: Dict, content2: Dict) -> float:
        """
        Вычисляет схожесть двух паттернов.
        
        Simple Jaccard similarity on key fields.
        """
        key_fields = ["goal_type", "domains", "success_factors", "mistakes"]
        
        matches = 0
        total = 0
        
        for field in key_fields:
            if field in content1 or field in content2:
                total += 1
                val1 = set(content1.get(field, []) or [])
                val2 = set(content2.get(field, []) or [])
                
                if val1 and val2:
                    intersection = len(val1 & val2)
                    union = len(val1 | val2)
                    matches += intersection / union if union > 0 else 0
                elif val1 == val2:
                    matches += 1
        
        return matches / total if total > 0 else 0.0

    async def _update_pattern_confidence(
        self,
        pattern_id: str,
        new_confidence: float
    ) -> str:
        """
        Обновляет confidence существующего паттерна.
        
        Args:
            pattern_id: ID паттерна
            new_confidence: Новое значение confidence
            
        Returns:
            ID паттерна
        """
        from models import Thought
        
        async with AsyncSessionLocal() as db:
            stmt = select(Thought).where(Thought.id == uuid.UUID(pattern_id))
            result = await db.execute(stmt)
            thought = result.scalar_one_or_none()
            
            if thought:
                # Boost confidence if new evidence supports it
                old_confidence = thought.content.get("confidence", 0.5) if isinstance(thought.content, dict) else 0.5
                boosted = min(1.0, (old_confidence + new_confidence) / 2 + 0.1)
                
                # Update status if confidence crosses threshold
                if boosted > 0.5:
                    thought.status = "active"
                
                await db.commit()
                print(f"🔄 Updated pattern {pattern_id}: confidence {old_confidence:.2f} → {boosted:.2f}")
            
            return pattern_id

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

                except (ValueError, KeyError, json.JSONDecodeError) as e:
                    logger.debug("pattern_parse_error", thought_id=str(thought.id), error=str(e))
                    continue
                except Exception as e:
                    logger.warning("unexpected_pattern_error", thought_id=str(thought.id), error=str(e))
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

    async def store_pattern_vector(
        self,
        pattern_type: str,
        content: Dict,
        pattern_id: str
    ) -> bool:
        """
        Сохраняет паттерн в Milvus для векторного поиска.
        
        Args:
            pattern_type: Тип паттерна
            content: Содержимое паттерна
            pattern_id: ID паттерна из PostgreSQL
            
        Returns:
            True если успешно, False иначе
        """
        try:
            # Создаём текстовое представление для embedding
            text_repr = self._pattern_to_text(pattern_type, content)
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{MEMORY_URL}/remember",
                    json={
                        "text": text_repr,
                        "type": "semantic",
                        "metadata": {
                            "pattern_id": pattern_id,
                            "pattern_type": pattern_type,
                            "goal_type": content.get("goal_type"),
                            "domains": content.get("domains", []),
                            "confidence": content.get("confidence", 0.5)
                        }
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.info(f"⚠️ Milvus store error: {e}")
            return False

    async def retrieve_similar_patterns_vector(
        self,
        query_text: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Извлекает похожие паттерны из Milvus по векторному сходству.
        
        Args:
            query_text: Текст для поиска
            limit: Максимум результатов
            
        Returns:
            Список похожих паттернов
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{MEMORY_URL}/search",
                    json={
                        "text": query_text,
                        "type": "semantic",
                        "top_k": limit
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    matches = data.get("matches", [])
                    
                    results = []
                    for match in matches:
                        try:
                            # Пытаемся распарсить как JSON
                            if isinstance(match, str) and match.startswith("{"):
                                pattern = json.loads(match)
                                results.append(pattern)
                        except json.JSONDecodeError as e:
                            logger.debug("milvus_match_json_error", match=match[:100], error=str(e))
                            continue
                        except Exception as e:
                            logger.warning("milvus_match_error", match=match[:100], error=str(e))
                            continue
                    
                    return results
        except Exception as e:
            logger.info(f"⚠️ Milvus search error: {e}")
        
        return []

    async def cleanup_old_patterns(self, days: int = 30) -> int:
        """
        Удаляет старые паттерны с low confidence.
        
        Args:
            days: Удалить паттерны старше N дней
            
        Returns:
            Количество удалённых паттернов
        """
        async with AsyncSessionLocal() as db:
            cutoff = datetime.now() - timedelta(days=days)
            
            # Удаляем tentative паттерны старше cutoff
            stmt = delete(Thought).where(
                Thought.status == "tentative",
                Thought.created_at < cutoff
            )
            
            result = await db.execute(stmt)
            await db.commit()
            
            deleted_count = result.rowcount
            logger.info(f"🧹 Cleaned up {deleted_count} old patterns")
            
            return deleted_count

    def _pattern_to_text(self, pattern_type: str, content: Dict) -> str:
        """
        Преобразует паттерн в текст для embedding.
        
        Args:
            pattern_type: Тип паттерна
            content: Содержимое паттерна
            
        Returns:
            Текстовое представление
        """
        parts = [f"Pattern type: {pattern_type}"]
        
        if "goal_type" in content:
            parts.append(f"Goal type: {content['goal_type']}")
        
        if "domains" in content:
            parts.append(f"Domains: {', '.join(content['domains'])}")
        
        if "success_factors" in content:
            parts.append(f"Success factors: {', '.join(content['success_factors'])}")
        
        if "lessons_learned" in content:
            parts.append(f"Lessons: {', '.join(content['lessons_learned'])}")
        
        if "root_causes" in content:
            parts.append(f"Root causes: {', '.join(content['root_causes'])}")
        
        if "mistakes" in content:
            parts.append(f"Mistakes: {', '.join(content['mistakes'])}")
        
        return " | ".join(parts)

    async def store_pattern_graph(
        self,
        pattern_type: str,
        content: Dict,
        pattern_id: str
    ) -> bool:
        """
        Сохраняет связи паттерна в Neo4j.
        
        Создаёт узлы для:
        - Pattern
        - Domains
        - Goal types
        
        И связи между ними.
        
        Args:
            pattern_type: Тип паттерна
            content: Содержимое паттерна
            pattern_id: ID паттерна
            
        Returns:
            True если успешно
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Create pattern node
                await client.post(
                    f"{MEMORY_URL}/add_fact",
                    json={
                        "subject": f"Pattern:{pattern_id}",
                        "predicate": "TYPE",
                        "object": pattern_type
                    }
                )
                
                # Create domain relationships
                for domain in content.get("domains", []):
                    await client.post(
                        f"{MEMORY_URL}/add_fact",
                        json={
                            "subject": f"Pattern:{pattern_id}",
                            "predicate": "RELATES_TO_DOMAIN",
                            "object": domain
                        }
                    )
                
                # Create goal type relationship
                if "goal_type" in content:
                    await client.post(
                        f"{MEMORY_URL}/add_fact",
                        json={
                            "subject": f"Pattern:{pattern_id}",
                            "predicate": "APPLIES_TO_GOAL_TYPE",
                            "object": content["goal_type"]
                        }
                    )
                
                return True
                
        except Exception as e:
            print(f"⚠️ Neo4j store error: {e}")
            return False

    async def batch_store_patterns_vector(
        self,
        patterns: List[Dict]
    ) -> int:
        """
        Batch сохранение паттернов в Milvus.
        
        Оптимизация: один HTTP запрос вместо N.
        
        Args:
            patterns: Список паттернов
            
        Returns:
            Количество успешно сохранённых
        """
        # Milvus doesn't support batch insert via HTTP API
        # Use sequential for now, but in parallel
        import asyncio
        
        tasks = []
        for p in patterns:
            task = self.store_pattern_vector(
                p["pattern_type"],
                p["content"],
                p["pattern_id"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if r is True)
        return success_count

    async def get_stats(self) -> Dict:
        """
        Возвращает статистику памяти для мониторинга.
        
        Returns:
            Dict со статистикой всех типов памяти
        """
        stats = {
            "postgresql": {},
            "milvus": {},
            "neo4j": {},
            "redis": {}
        }
        
        # PostgreSQL stats
        try:
            async with AsyncSessionLocal() as db:
                # Total patterns
                result = await db.execute(select(func.count(Thought.id)))
                stats["postgresql"]["total_patterns"] = result.scalar() or 0
                
                # By status
                result = await db.execute(
                    select(Thought.status, func.count(Thought.id))
                    .group_by(Thought.status)
                )
                stats["postgresql"]["by_status"] = {
                    row[0]: row[1] for row in result.all()
                }
                
                # By pattern type (parse from content)
                result = await db.execute(
                    select(Thought.content)
                    .where(Thought.content.like("%_pattern:%"))
                    .limit(100)
                )
                pattern_types = {}
                for row in result.scalars().all():
                    try:
                        content = row if isinstance(row, str) else str(row)
                        ptype = content.split(":")[0] if ":" in content else "unknown"
                        pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
                    except:
                        pass
                stats["postgresql"]["by_pattern_type"] = pattern_types
                
        except Exception as e:
            stats["postgresql"]["error"] = str(e)
        
        # Milvus stats
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check if memory service is responding
                response = await client.get(f"{MEMORY_URL}/user/analysis")
                stats["milvus"]["status"] = "connected"
        except Exception as e:
            stats["milvus"]["status"] = "disconnected"
            stats["milvus"]["error"] = str(e)[:100]
        
        # Redis stats (MemorySignal)
        try:
            from memory_signal import persistent_memory_registry
            stats["redis"]["memory_signals"] = persistent_memory_registry.summary()
        except Exception as e:
            stats["redis"]["error"] = str(e)
        
        return stats


# Глобальный экземпляр
semantic_memory = SemanticMemory()
