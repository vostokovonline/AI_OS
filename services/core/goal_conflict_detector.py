"""
Goal Conflict Detection System

Выявляет конфликты между целями:
- Resource conflicts (время, деньги, энергия)
- Time conflicts (одновременное выполнение)
- Value conflicts (противоречащие ценности)
- Mutually exclusive goals (полная несовместимость)

Интегрируется с:
- Goal Decomposition (проверка перед созданием subgoals)
- Decision Logic (учёт конфликтов при планировании)
- Dashboard v2 (warnings UI)

NS1/NS2: "Goal Linking (Связывание целей)" - строит взаимосвязи между целями
"""

from typing import List, Dict, Optional, Tuple
from sqlalchemy import select, and_
from database import AsyncSessionLocal
from models import Goal, GoalRelation, GoalConflict, UserProfile, UserValue
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# =============================================================================
# Pydantic Schemas
# =============================================================================

class ConflictDetectionResult(BaseModel):
    """Результат проверки конфликтов"""
    has_conflicts: bool
    conflicts: List[Dict] = Field(default_factory=list)
    severity: str = "none"  # none, low, medium, high, critical


class SingleConflict(BaseModel):
    """Один конфликт"""
    id: str
    goal_1_id: str
    goal_1_title: str
    goal_2_id: str
    goal_2_title: str
    conflict_type: str
    severity: str
    description: str
    resolution_suggestion: Optional[str] = None
    status: str


class ConflictResolution(BaseModel):
    """Предложение по разрешению конфликта"""
    prioritize_goal_id: str
    deprioritize_goal_id: str
    reason: str
    alternative_approach: Optional[str] = None


# =============================================================================
# Conflict Detector
# =============================================================================

class GoalConflictDetector:
    """
    Детектор конфликтов между целями.

    Проверяет:
    1. Resource conflicts (ограниченные ресурсы)
    2. Time conflicts (одновременное выполнение)
    3. Value conflicts (противоречие ценностям)
    4. Mutually exclusive (полная несовместимость)
    """

    # Правила для определения типов конфликтов
    RESOURCE_KEYWORDS = [
        "работать больше", "больше работы", "дополнительная работа",
        "экономить", "бюджет", "деньги", "финансы",
        "энергия", "время", "ресурсы"
    ]

    TIME_KEYWORDS = [
        "одновременно", "параллельно", "в одно время",
        "ежедневно", "каждый день", "еженедельно"
    ]

    MUTUALLY_EXCLUSIVE_PAIRS = [
        ("работать больше", "сократить рабочее время"),
        ("экономить деньги", "инвестировать в развитие"),
        ("снизить стресс", "больше ответственности"),
        ("больше отдыха", "повысить продуктивность"),
    ]

    async def check_goal_conflicts(self, goal_id: str,
                                  check_against: List[str] = None) -> ConflictDetectionResult:
        """
        Проверить цель на конфликты с другими целями.

        Args:
            goal_id: ID цели для проверки
            check_against: Список ID целей для проверки (если None - проверить со всеми активными)

        Returns:
            ConflictDetectionResult
        """
        async with AsyncSessionLocal() as db:
            # Получить цель
            stmt = select(Goal).where(Goal.id == goal_id)
            result = await db.execute(stmt)
            goal = result.scalar_one_or_none()

            if not goal:
                return ConflictDetectionResult(has_conflicts=False, conflicts=[])

            # Получить цели для проверки
            if check_against is None:
                # Все активные goals (без user_id - проверяем все топ-уровневые)
                stmt_other = select(Goal)\
                    .where(Goal.id != goal_id)\
                    .where(Goal.parent_id == None)\
                    .where(Goal.status.in_(["active", "pending", "in_progress"]))
                result_other = await db.execute(stmt_other)
                other_goals = result_other.scalars().all()
            else:
                stmt_other = select(Goal).where(Goal.id.in_(check_against))
                result_other = await db.execute(stmt_other)
                other_goals = result_other.scalars().all()

            conflicts = []

            # Проверить каждую пару
            for other_goal in other_goals:
                conflict = await self._detect_conflict_between(db, goal, other_goal)
                if conflict:
                    conflicts.append(conflict)

            # Определить общий severity
            if not conflicts:
                severity = "none"
            elif any(c["severity"] == "critical" for c in conflicts):
                severity = "critical"
            elif any(c["severity"] == "high" for c in conflicts):
                severity = "high"
            elif any(c["severity"] == "medium" for c in conflicts):
                severity = "medium"
            else:
                severity = "low"

            return ConflictDetectionResult(
                has_conflicts=len(conflicts) > 0,
                conflicts=conflicts,
                severity=severity
            )

    async def _detect_conflict_between(self, db, goal_1: Goal, goal_2: Goal) -> Optional[Dict]:
        """
        Обнаружить конфликт между двумя целями.

        Returns:
            Dict с описанием конфликта или None
        """
        # 1. Проверить уже обнаруженные конфликты
        stmt_existing = select(GoalConflict).where(
            and_(
                GoalConflict.goal_1_id == goal_1.id,
                GoalConflict.goal_2_id == goal_2.id,
                GoalConflict.status != "resolved"
            )
        )
        result_existing = await db.execute(stmt_existing)
        existing = result_existing.scalar_one_or_none()

        if existing:
            return {
                "id": str(existing.id),
                "goal_1_id": str(goal_1.id),
                "goal_1_title": goal_1.title,
                "goal_2_id": str(goal_2.id),
                "goal_2_title": goal_2.title,
                "conflict_type": existing.conflict_type,
                "severity": existing.severity,
                "description": existing.description,
                "resolution_suggestion": existing.resolution_suggestion,
                "status": existing.status
            }

        # 2. Проверить mutually exclusive
        for pair in self.MUTUALLY_EXCLUSIVE_PAIRS:
            title_1_lower = goal_1.title.lower()
            title_2_lower = goal_2.title.lower()

            if (pair[0] in title_1_lower and pair[1] in title_2_lower) or \
               (pair[0] in title_2_lower and pair[1] in title_1_lower):
                return await self._create_conflict(
                    db, goal_1, goal_2,
                    conflict_type="mutually_exclusive",
                    severity="critical",
                    description=f"Цели '{goal_1.title}' и '{goal_2.title}' являются взаимоисключающими",
                    resolution_suggestion=f"Выберите одну из целей: '{goal_1.title}' ИЛИ '{goal_2.title}', но не обе одновременно"
                )

        # 3. Проверить resource conflicts
        has_resource_1 = any(kw in goal_1.title.lower() for kw in self.RESOURCE_KEYWORDS)
        has_resource_2 = any(kw in goal_2.title.lower() for kw in self.RESOURCE_KEYWORDS)

        if has_resource_1 and has_resource_2:
            return await self._create_conflict(
                db, goal_1, goal_2,
                conflict_type="resource",
                severity="medium",
                description=f"Обе цели требуют ограниченных ресурсов: '{goal_1.title}' и '{goal_2.title}'",
                resolution_suggestion=f"Приоритизировать одну цель и отложить другую, или найти способ распределить ресурсы"
            )

        # 4. Проверить value conflicts (если есть personality profile)
        # Это требует интеграции с Personality Engine

        # 5. Проверить временные конфликты (если оба require daily/weekly)
        has_time_1 = any(kw in goal_1.title.lower() for kw in self.TIME_KEYWORDS)
        has_time_2 = any(kw in goal_2.title.lower() for kw in self.TIME_KEYWORDS)

        if has_time_1 and has_time_2:
            return await self._create_conflict(
                db, goal_1, goal_2,
                conflict_type="time",
                severity="low",
                description=f"Обе цели требуют регулярного времени: '{goal_1.title}' и '{goal_2.title}'",
                resolution_suggestion=f"Разнести по времени или объединить в один routine"
            )

        return None

    async def _create_conflict(self, db, goal_1: Goal, goal_2: Goal,
                              conflict_type: str, severity: str,
                              description: str, resolution_suggestion: str) -> Dict:
        """
        Создать запись о конфликте в БД.

        Returns:
            Dict с конфликтом
        """
        # Проверить, не существует ли уже
        stmt_check = select(GoalConflict).where(
            and_(
                GoalConflict.goal_1_id == goal_1.id,
                GoalConflict.goal_2_id == goal_2.id
            )
        )
        result_check = await db.execute(stmt_check)
        existing = result_check.scalar_one_or_none()

        if existing:
            # Обновить статус
            existing.status = "detected"
            await db.commit()
            conflict_id = existing.id
        else:
            # Создать новый
            conflict = GoalConflict(
                goal_1_id=goal_1.id,
                goal_2_id=goal_2.id,
                conflict_type=conflict_type,
                severity=severity,
                description=description,
                resolution_suggestion=resolution_suggestion,
                status="detected"
            )
            db.add(conflict)
            await db.commit()
            await db.refresh(conflict)
            conflict_id = conflict.id

        return {
            "id": str(conflict_id),
            "goal_1_id": str(goal_1.id),
            "goal_1_title": goal_1.title,
            "goal_2_id": str(goal_2.id),
            "goal_2_title": goal_2.title,
            "conflict_type": conflict_type,
            "severity": severity,
            "description": description,
            "resolution_suggestion": resolution_suggestion,
            "status": "detected"
        }

    async def get_conflicts_for_user(self, user_id: str,
                                   status: str = None,
                                   severity: str = None) -> List[SingleConflict]:
        """
        Получить все конфликты пользователя.

        Args:
            user_id: UUID пользователя
            status: Фильтр по статусу (detected, resolved, ignored)
            severity: Фильтр по severity (low, medium, high, critical)

        Returns:
            List[SingleConflict]
        """
        async with AsyncSessionLocal() as db:
            # Получить все ID целей (так как нет user_id)
            stmt_goals = select(Goal.id)
            result_goals = await db.execute(stmt_goals)
            goal_ids = [g[0] for g in result_goals.all()]

            if not goal_ids:
                return []

            # Получить конфликты
            stmt = select(GoalConflict).where(
                GoalConflict.goal_1_id.in_(goal_ids)
            )

            if status:
                stmt = stmt.where(GoalConflict.status == status)
            if severity:
                stmt = stmt.where(GoalConflict.severity == severity)

            stmt = stmt.order_by(GoalConflict.detected_at.desc())

            result = await db.execute(stmt)
            conflicts = result.scalars().all()

            # Загрузить данные о целях
            result_list = []
            for c in conflicts:
                # Получить цели
                stmt_g1 = select(Goal).where(Goal.id == c.goal_1_id)
                res_g1 = await db.execute(stmt_g1)
                goal_1 = res_g1.scalar_one_or_none()

                stmt_g2 = select(Goal).where(Goal.id == c.goal_2_id)
                res_g2 = await db.execute(stmt_g2)
                goal_2 = res_g2.scalar_one_or_none()

                if goal_1 and goal_2:
                    result_list.append(SingleConflict(
                        id=str(c.id),
                        goal_1_id=str(c.goal_1_id),
                        goal_1_title=goal_1.title,
                        goal_2_id=str(c.goal_2_id),
                        goal_2_title=goal_2.title,
                        conflict_type=c.conflict_type,
                        severity=c.severity,
                        description=c.description,
                        resolution_suggestion=c.resolution_suggestion,
                        status=c.status
                    ))

            return result_list

    async def resolve_conflict(self, conflict_id: str,
                              resolution: str,
                              resolved_by: str = "user") -> SingleConflict:
        """
        Разрешить конфликт.

        Args:
            conflict_id: UUID конфликта
            resolution: Текст решения
            resolved_by: Кто разрешил (user, system)

        Returns:
            Обновлённый SingleConflict
        """
        async with AsyncSessionLocal() as db:
            stmt = select(GoalConflict).where(GoalConflict.id == conflict_id)
            result = await db.execute(stmt)
            conflict = result.scalar_one_or_none()

            if not conflict:
                raise ValueError(f"Conflict {conflict_id} not found")

            conflict.status = "resolved"
            conflict.resolved_at = datetime.utcnow()
            # Можно добавить resolution текст в отдельное поле или в description

            await db.commit()
            await db.refresh(conflict)

            # Получить цели
            stmt_g1 = select(Goal).where(Goal.id == conflict.goal_1_id)
            res_g1 = await db.execute(stmt_g1)
            goal_1 = res_g1.scalar_one_or_none()

            stmt_g2 = select(Goal).where(Goal.id == conflict.goal_2_id)
            res_g2 = await db.execute(stmt_g2)
            goal_2 = res_g2.scalar_one_or_none()

            return SingleConflict(
                id=str(conflict.id),
                goal_1_id=str(conflict.goal_1_id),
                goal_1_title=goal_1.title if goal_1 else "Unknown",
                goal_2_id=str(conflict.goal_2_id),
                goal_2_title=goal_2.title if goal_2 else "Unknown",
                conflict_type=conflict.conflict_type,
                severity=conflict.severity,
                description=conflict.description,
                resolution_suggestion=conflict.resolution_suggestion,
                status=conflict.status
            )


# =============================================================================
# Singleton instance
# =============================================================================

_conflict_detector_instance = None

def get_goal_conflict_detector() -> GoalConflictDetector:
    """Получить singleton instance GoalConflictDetector"""
    global _conflict_detector_instance
    if _conflict_detector_instance is None:
        _conflict_detector_instance = GoalConflictDetector()
    return _conflict_detector_instance
