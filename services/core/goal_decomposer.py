"""
GOAL DECOMPOSER - Система декомпозиции целей
Разбивает цели на подцели согласно онтологии и критериям атомарности

UoW MIGRATION: Декомпозиция теперь атомарна - все операции в одной транзакции.
"""
import os
import uuid
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select, func
from database import AsyncSessionLocal
from models import Goal
from agent_graph import app_graph
from goal_contract_validator import goal_contract_validator

# UoW imports для новой архитектуры
from infrastructure.uow import UnitOfWork, GoalRepository


TELEGRAM_URL = os.getenv("TELEGRAM_URL", "http://telegram:8004")


class GoalDecomposer:
    """Декомпозитор целей - Goal System Layer"""

    # Типология целей
    GOAL_TYPES = {
        "achievable": "Выполнимая цель (может быть завершена)",
        "continuous": "Непрерывная цель (улучшение, нет финальной точки)",
        "directional": "Векторная/ценностная (задает направление, невыполнимая)",
        "exploratory": "Исследовательская (поиск, результат неизвестен)",
        "meta": "Мета-цель (улучшение самой системы)"
    }

    # Уровни глубины
    DEPTH_LEVELS = {
        0: "Mission (намерение)",
        1: "Strategic (стратегическая подцель)",
        2: "Operational (операционная подцель)",
        3: "Tactical/Atomic (тактическая/атомарная цель)"
    }

    def __init__(self):
        self.decomposition_history = {}

    async def classify_goal(self, title: str, description: str = "") -> Dict:
        """
        Классифицирует цель по типологии

        Returns:
            {
                "goal_type": "achievable|continuous|directional|exploratory|meta",
                "reasoning": "...",
                "executable": True/False,
                "decomposable": True/False
            }
        """
        classification_prompt = f"""Классифицируй цель по онтологии:

ЦЕЛЬ: {title}
ОПИСАНИЕ: {description if description else "Не указано"}

Типология целей:
1. achievable - выполнимая цель, есть финальная точка
   Примеры: "Настроить монетизацию", "Найти проекты и провести анализ"

2. continuous - непрерывная цель, нет финальной точки, есть метрика улучшения
   Примеры: "Улучшать состояние пользователя", "Повышать качество"

3. directional - векторная/ценностная, принципиально невыполнимая, задает направление
   Примеры: "Оставить след в истории", "Действовать этично"

4. exploratory - исследовательская, результат неизвестен заранее
   Примеры: "Найти проекты по теме X и провести анализ"

5. meta - мета-цель, управляет самой системой
   Примеры: "Улучшить качество целеполагания"

Верни JSON:
{{
    "goal_type": "тип",
    "reasoning": "обоснование",
    "executable": true/false,
    "decomposable": true/false
}}
"""

        try:
            thread_id = f"classify_{hash(title) % 100000}"
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=classification_prompt)]
            }, config={"configurable": {"thread_id": thread_id}})

            result = response["messages"][-1].content

            # Пытаемся извлечь JSON
            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            classification = json.loads(result)

            # Валидация
            if classification["goal_type"] not in self.GOAL_TYPES:
                classification["goal_type"] = "achievable"  # default

            return classification

        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {
                "goal_type": "achievable",
                "reasoning": "Ошибка классификации, по умолчанию achievable",
                "executable": True,
                "decomposable": True
            }

    async def analyze_domains(self, title: str, description: str = "") -> List[str]:
        """
        Определяет домены цели

        Returns:
            ["nutrition", "light", "temperature", ...]
        """
        domain_prompt = f"""Определи домены для этой цели:

ЦЕЛЬ: {title}
ОПИСАНИЕ: {description}

Возможные домены: программирование, инфраструктура, данные, UI/UX,
монетизация, безопасность, тестирование, исследования, анализ, дизайн,
обучение, здоровье, финансы, коммуникации, автоматизация, и т.д.

Верни JSON с массивом доменов (3-7 доменов):
{{"domains": ["domain1", "domain2", ...]}}
"""

        try:
            thread_id = f"domains_{hash(title) % 100000}"
            response = await app_graph.ainvoke({
                "messages": [HumanMessage(content=domain_prompt)]
            }, config={"configurable": {"thread_id": thread_id}})

            result = response["messages"][-1].content

            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            data = json.loads(result)
            return data.get("domains", [])

        except Exception as e:
            print(f"❌ Domain analysis error: {e}")
            return ["general"]

    async def decompose_goal(self, goal_id: str, max_depth: int = 3) -> List[Dict]:
        """
        Декомпозирует цель на подцели

        Args:
            goal_id: ID цели для декомпозиции
            max_depth: Максимальная глубина (default: 3)

        Returns:
            Список созданных подцелей
        """

        # 🧠 EMOTIONAL INFERENCE ENGINE V2 INTEGRATION
        # Use EIE v2 for intelligent emotion-based decision modification
        try:
            from emotional_inference_v2 import emotional_inference_engine_v2

            # Get goal for user_id
            async with AsyncSessionLocal() as db:
                stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
                result = await db.execute(stmt)
                goal = result.scalar_one_or_none()
                if goal:
                    user_id = str(goal.user_id) if goal.user_id else "00000000-0000-0000-0000-000000000001"
                else:
                    user_id = "00000000-0000-0000-0000-000000000001"

            # Determine action type for forecasting
            action_type = "deep_goal_decomposition" if max_depth > 1 else "simple_task"

            # Use EIE v2 to get decision modifiers
            modifiers = await emotional_inference_engine_v2.infer(
                user_id=user_id,
                proposed_action=action_type,
                intent=None,  # No explicit intent, use neutral
                signals=None
            )

            # Apply modifiers
            max_depth = modifiers.max_depth
            print(f"🧠 [EIE v2] Adjusted max_depth to {max_depth} (pace={modifiers.pace}, style={modifiers.style})")

            if modifiers.safety_override:
                print(f"🔒 [EIE v2] SAFETY OVERRIDE ACTIVE - recovery_mode={modifiers.recovery_mode}")

        except Exception as e:
            # If EIE v2 fails, continue with original max_depth
            print(f"⚠️  EIE v2 error in decomposer (continuing): {e}")

        async with AsyncSessionLocal() as db:
            # Re-fetch goal since we might have used the session above
            stmt = select(Goal).where(Goal.id == uuid.UUID(goal_id))
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return []

            # 🔑 GOAL CONTRACT CHECK v3.0 - Проверяем разрешено ли декомпозировать
            can_decompose, reason = goal_contract_validator.can_execute_action(goal, "decompose")
            if not can_decompose:
                print(f"⛔ Decomposition forbidden: {reason}")
                return []

            # Если цель атомарная - не декомпозируем
            if goal.is_atomic:
                return []

            # 🔑 GOAL CONTRACT CHECK v3.0 - Проверяем лимит глубины
            can_proceed, reason = goal_contract_validator.check_depth_limit(goal, goal.depth_level)
            if not can_proceed:
                print(f"⛔ Depth limit reached: {reason}")
                goal.is_atomic = True
                await db.commit()
                return []

            # Если достигли максимальной глубины - помечаем как atomic
            if goal.depth_level >= max_depth:
                goal.is_atomic = True
                await db.commit()
                return []

            # 🔑 GOAL CONTRACT CHECK v3.0 - Проверяем лимит подцелей
            stmt_count = select(func.count(Goal.id)).where(Goal.parent_id == goal.id)
            result_count = await db.execute(stmt_count)
            current_subgoals_count = result_count.scalar() or 0

            can_proceed, reason = goal_contract_validator.check_subgoals_limit(goal, current_subgoals_count)
            if not can_proceed:
                print(f"⛔ Subgoals limit reached: {reason}")
                return []

            # Генерируем подцели
            subgoals = await self._generate_subgoals(goal)

            # 🔑 GOAL CONTRACT CHECK v3.0 - Проверяем лимит количества подцелей
            can_proceed, reason = goal_contract_validator.check_subgoals_limit(
                goal,
                current_subgoals_count + len(subgoals)
            )
            if not can_proceed:
                print(f"⛔ Would exceed subgoals limit: {reason}")
                # Обрезаем до лимита
                if goal.goal_contract:
                    max_subgoals = goal.goal_contract.get("max_subgoals", 100)
                    subgoals = subgoals[:max(0, max_subgoals - current_subgoals_count)]

            # Сохраняем подцели
            created_subgoals = []
            for subgoal_data in subgoals:
                # 🔑 GOAL CONTRACT v3.0 - Создаем контракт для подцели
                subgoal_type = subgoal_data.get("goal_type", "achievable")
                subgoal_contract = goal_contract_validator.create_default_contract(subgoal_type)

                subgoal = Goal(
                    parent_id=goal.id,
                    title=subgoal_data["title"],
                    description=subgoal_data.get("description", ""),
                    goal_type=subgoal_type,
                    depth_level=goal.depth_level + 1,
                    is_atomic=subgoal_data.get("is_atomic", False),
                    domains=subgoal_data.get("domains", []),
                    completion_criteria=subgoal_data.get("completion_criteria"),
                    success_definition=subgoal_data.get("success_definition"),
                    goal_contract=subgoal_contract,  # 🔑 v3.0
                    status="pending",
                    progress=0.0
                )
                db.add(subgoal)
                await db.commit()
                await db.refresh(subgoal)

                created_subgoals.append({
                    "id": str(subgoal.id),
                    "title": subgoal.title,
                    "depth": subgoal.depth_level,
                    "is_atomic": subgoal.is_atomic
                })

            # 🔒 STATE-MACHINE: Parent with children → active via transition service (Phase 1)
            # Инвариант: is_atomic == false AND child_count > 0 → status != pending
            if created_subgoals and goal.is_atomic is False:
                from infrastructure.uow import create_uow_provider
                from goal_transition_service import transition_service

                uow_provider = create_uow_provider()
                async with uow_provider() as uow:
                    transition_result = await transition_service.transition(
                        uow=uow,
                        goal_id=goal.id,
                        new_state="active",
                        reason=f"Created {len(created_subgoals)} subgoals",
                        actor="goal_decomposer"
                    )

                    if transition_result["result"] != "success":
                        print(f"⚠️  Transition to active failed: {transition_result}")
                        return created_subgoals

                    goal.progress = 0.0
                    goal.status = "active"

                print(f"✅ Parent goal '{goal.title}' → active (created {len(created_subgoals)} subgoals)")

            # Отправляем уведомление
            await self._send_decomposition_notification(goal, created_subgoals)

            return created_subgoals

    async def decompose_goal_with_uow(
        self,
        uow: UnitOfWork,
        goal_id: str,
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Декомпозирует цель на подцели ВНУТРИ существующей UoW транзакции.
        
        UoW MIGRATION: Это атомарная операция - либо все подцели создаются,
        либо ничего (rollback). Ни одного commit() внутри.
        
        Args:
            uow: UnitOfWork с активной транзакцией
            goal_id: ID цели для декомпозиции  
            max_depth: Максимальная глубина
            
        Returns:
            List[Dict]: Созданные подцели
        """
        from goal_transition_service import transition_service
        from uuid import UUID
        
        goal_uuid = UUID(goal_id)
        repo = GoalRepository(uow)
        
        # 1. Получаем parent goal с pessimistic lock (SELECT FOR UPDATE)
        goal = await repo.get_for_update(uow.session, goal_uuid)
        
        if not goal:
            print(f"❌ Goal {goal_id} not found")
            return []
        
        # 2. GOAL CONTRACT CHECK v3.0 - Проверяем разрешено ли декомпозировать
        can_decompose, reason = goal_contract_validator.can_execute_action(goal, "decompose")
        if not can_decompose:
            print(f"⛔ Decomposition forbidden: {reason}")
            return []
        
        # Если цель атомарная - не декомпозируем
        if goal.is_atomic:
            return []
        
        # 3. GOAL CONTRACT CHECK v3.0 - Проверяем лимит глубины
        can_proceed, reason = goal_contract_validator.check_depth_limit(goal, goal.depth_level)
        if not can_proceed:
            print(f"⛔ Depth limit reached: {reason}")
            goal.is_atomic = True
            await repo.update(uow.session, goal)
            return []
        
        # Если достигли максимальной глубины - помечаем как atomic
        if goal.depth_level >= max_depth:
            goal.is_atomic = True
            await repo.update(uow.session, goal)
            return []
        
        # 4. Проверяем лимит подцелей (читаем текущее количество)
        stmt_count = select(func.count(Goal.id)).where(Goal.parent_id == goal.id)
        result_count = await uow.session.execute(stmt_count)
        current_subgoals_count = result_count.scalar() or 0
        
        can_proceed, reason = goal_contract_validator.check_subgoals_limit(goal, current_subgoals_count)
        if not can_proceed:
            print(f"⛔ Subgoals limit reached: {reason}")
            return []
        
        # 5. Генерируем подцели (LLM call - вне транзакции это было бы опасно, но здесь ок)
        subgoals = await self._generate_subgoals(goal)
        
        # 6. Проверяем лимит количества подцелей
        can_proceed, reason = goal_contract_validator.check_subgoals_limit(
            goal,
            current_subgoals_count + len(subgoals)
        )
        if not can_proceed:
            print(f"⛔ Would exceed subgoals limit: {reason}")
            if goal.goal_contract:
                max_subgoals = goal.goal_contract.get("max_subgoals", 100)
                subgoals = subgoals[:max(0, max_subgoals - current_subgoals_count)]
        
        # 7. Создаём все подцели (НИ ОДНОГО commit!)
        created_subgoals = []
        for subgoal_data in subgoals:
            subgoal_type = subgoal_data.get("goal_type", "achievable")
            subgoal_contract = goal_contract_validator.create_default_contract(subgoal_type)
            
            subgoal = Goal(
                parent_id=goal.id,
                title=subgoal_data["title"],
                description=subgoal_data.get("description", ""),
                goal_type=subgoal_type,
                depth_level=goal.depth_level + 1,
                is_atomic=subgoal_data.get("is_atomic", False),
                domains=subgoal_data.get("domains", []),
                completion_criteria=subgoal_data.get("completion_criteria"),
                success_definition=subgoal_data.get("success_definition"),
                goal_contract=subgoal_contract,
                _status="pending",
                progress=0.0
            )
            
            await repo.save(uow.session, subgoal)
            
            # Refresh чтобы получить ID (но НЕ commit!)
            await uow.session.flush([subgoal])
            
            created_subgoals.append({
                "id": str(subgoal.id),
                "title": subgoal.title,
                "depth": subgoal.depth_level,
                "is_atomic": subgoal.is_atomic
            })
        
        # 8. STATE-MACHINE: Parent → active через transition service
        if created_subgoals and goal.is_atomic is False:
            transition_result = await transition_service.transition(
                uow=uow,
                goal_id=goal.id,
                new_state="active",
                reason=f"Created {len(created_subgoals)} subgoals",
                actor="goal_decomposer"
            )
            
            if transition_result["result"] != "success":
                print(f"⚠️  Transition to active failed: {transition_result}")
                return created_subgoals
            
            goal.progress = 0.0
            await repo.update(uow.session, goal)
        
        print(f"✅ Decomposed '{goal.title}' → {len(created_subgoals)} subgoals (atomic transaction)")
        
        # 9. Отправляем уведомление (fire-and-forget, вне транзакции)
        await self._send_decomposition_notification(goal, created_subgoals)
        
        return created_subgoals

    async def _generate_subgoals(self, goal: Goal) -> List[Dict]:
        """Генерирует подцели через прямой LLM вызов (без agent graph)

        Phase 1 Integration: Использует Personality Engine для персонализированной декомпозиции
        """
        import os
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        # Создаем LLM с таймаутом для декомпозиции
        llm = ChatOpenAI(
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY", "sk-1234"),
            model=os.getenv("LLM_MODEL", "ollama/qwen3-coder:480b-cloud"),
            temperature=0.2,
            request_timeout=120  # 2 минуты (qwen3-coder очень быстрый!)
        )

        # Phase 1: Получаем профиль пользователя из Personality Engine
        # Используем дефолтный user_id из .env (TELEGRAM_OWNER_ID)
        default_user_id = os.getenv("TELEGRAM_OWNER_ID", "5503051162")
        # Форматируем как UUID для Personality Engine
        user_id_uuid = f"{default_user_id[:8]}-{default_user_id[8:12]}-0000-0000-000000000000"

        # Загружаем ценности пользователя
        values_context = ""
        value_list = []
        try:
            from personality_engine import get_personality_engine
            engine = get_personality_engine()
            value_matrix = await engine.get_value_matrix(user_id_uuid)

            # Сортируем ценности по важности
            sorted_values = sorted(value_matrix.items(), key=lambda x: x[1], reverse=True)
            value_list = [f"{name}({importance:.1f})" for name, importance in sorted_values]

            if value_list:
                values_context = f"""

ЦЕННОСТИ ПОЛЬЗОВАТЕЛЯ (по важности):
{', '.join(value_list)}

При декомпозиции УЧИТЫВАЙ:
1. Подцели должны быть согласованы с топ-3 ценностями
2. Приоритет отдавай подцелям, соответствующим самым важным ценностям
3. Избегай подцелей, которые конфликтуют с ценностями пользователя
"""
        except Exception as e:
            print(f"⚠️  Could not load personality profile: {e}")
            values_context = "\n(Используй универсальные ценности при декомпозиции)"

        decomposition_prompt = f"""Ты - эксперт по декомпозиции целей. Разбей цель на подцели согласно онтологии Goal System.

ЦЕЛЬ: {goal.title}
ОПИСАНИЕ: {goal.description or 'Не указано'}
ТИП: {goal.goal_type}
УРОВЕНЬ: {goal.depth_level}
ДОМЕНЫ: {goal.domains or []}{values_context}
ПРАВИЛА ДЕКОМПОЗИЦИИ:
1. Дроби до уровня, где можно достичь измеримого результата
2. НЕ включай конкретные действия (HOW) - только состояния (WHAT)
3. Каждая подцель должна быть в одном домене
4. Оптимально: 3-7 подцелей
5. Подцели должны быть согласованы с ценностями пользователя

КРИТЕРИИ АТОМАРНОЙ ЦЕЛИ:
- Есть измеримый результат/состояние
- Не содержит конкретных шагов
- Можно превратить в 3-7 задач
- В рамках одного домена

ВАЖНО - ТЫ ДОЛЖЕН ВЕРНУТЬ ТОЛЬКО JSON БЕЗ ЛИШНЕГО ТЕКСТА:

{{
    "subgoals": [
        {{
            "title": "Название подцели",
            "description": "Описание желаемого состояния",
            "goal_type": "achievable|continuous|exploratory",
            "is_atomic": true/false,
            "domains": ["domain1"],
            "completion_criteria": {{"condition": "..."}},
            "success_definition": "Успех это когда..."
        }}
    ],
    "reasoning": "Обоснование декомпозиции с учетом ценностей пользователя"
}}
"""

        try:
            # Прямой LLM вызов вместо agent graph (избегаем SAFETY BREAK)
            system_msg = SystemMessage(content="Ты эксперт по декомпозиции целей. Отвечай только валидным JSON.")
            user_msg = HumanMessage(content=decomposition_prompt)

            response = await llm.ainvoke([system_msg, user_msg])
            result = response.content

            # Парсим JSON ответ
            import json
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            data = json.loads(result)
            print(f"✅ Decomposition completed with {len(data.get('subgoals', []))} subgoals")
            print(f"   User values: {', '.join(value_list[:3]) if value_list else 'N/A'}")
            return data.get("subgoals", [])

        except Exception as e:
            print(f"❌ Decomposition error: {e}")
            print(f"   Raw response was: {str(result)[:200] if 'result' in locals() else 'No response'}")
            # Если не удалось декомпозировать - помечаем как atomic
            goal.is_atomic = True
            return []

    async def _send_decomposition_notification(self, goal: Goal, subgoals: List[Dict]):
        """Отправляет уведомление о декомпозиции"""
        try:
            import httpx

            message_parts = []
            message_parts.append(f"🎯 ЦЕЛЬ ДЕКОМПОЗОВАНА: {goal.title}")
            message_parts.append(f"📊 Уровень: {goal.depth_level} → {goal.depth_level + 1}")
            message_parts.append(f"🔻 Создано подцелей: {len(subgoals)}")

            if subgoals:
                message_parts.append("\n📍 Подцели:")
                for i, sg in enumerate(subgoals[:5], 1):
                    atomic_mark = "⚛️ " if sg["is_atomic"] else "  "
                    message_parts.append(f"{atomic_mark}{i}. {sg['title']}")

            full_message = "\n".join(message_parts)

            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{TELEGRAM_URL}/notify",
                    json={"message": full_message},
                    timeout=5
                )
        except:
            pass


# Глобальный экземпляр
goal_decomposer = GoalDecomposer()
