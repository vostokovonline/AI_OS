"""
Personality Engine — ядро индивидуальности ИИ.

Управляет моделью личности пользователя:
- Хранит и обновляет профиль (ценности, стиль, предпочтения)
- Предоставляет данные другим модулям (Goal System, Decision Logic)
- Адаптируется на основе feedback

Phase 1: Core Engine
- UserProfile (Big Five traits, motivations)
- UserValue (ценности)
- UserPreference (communication style, boundaries)
- PersonalityFeedback (adaptation loop)
"""
from typing import Optional, Dict, List
from sqlalchemy import select
from database import AsyncSessionLocal
from models import (
    UserProfile, UserValue, UserPreference, PersonalityFeedback,
    PersonalitySnapshot, ContextualMemory
)
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import json


# =============================================================================
# Pydantic Schemas для API
# =============================================================================

class CoreTraitsSchema(BaseModel):
    """Big Five traits"""
    openness: float = Field(default=0.5, ge=0.0, le=1.0, description="Открытость опыту")
    conscientiousness: float = Field(default=0.5, ge=0.0, le=1.0, description="Добросовестность")
    extraversion: float = Field(default=0.5, ge=0.0, le=1.0, description="Экстраверсия")
    agreeableness: float = Field(default=0.5, ge=0.0, le=1.0, description="Доброжелательность")
    neuroticism: float = Field(default=0.5, ge=0.0, le=1.0, description="Нейротизм")


class MotivationsSchema(BaseModel):
    """Motivations"""
    growth: float = Field(default=0.5, ge=0.0, le=1.0, description="Стремление к развитию")
    achievement: float = Field(default=0.5, ge=0.0, le=1.0, description="Достижения")
    comfort: float = Field(default=0.5, ge=0.0, le=1.0, description="Комфорт")
    recognition: float = Field(default=0.5, ge=0.0, le=1.0, description="Признание")
    social_connection: float = Field(default=0.5, ge=0.0, le=1.0, description="Социальные связи")


class ValueSchema(BaseModel):
    """Single value"""
    name: str = Field(..., description="Название ценности")
    importance: float = Field(..., ge=0.0, le=1.0, description="Важность")


class CommunicationStyleSchema(BaseModel):
    """Communication preferences"""
    tone: str = Field(default="спокойный", description="Тон общения")
    humor: str = Field(default="умеренный", description="Юмор")
    detail_level: str = Field(default="средний", description="Детальность")
    language: str = Field(default="ru", description="Язык")


class ActivityPatternsSchema(BaseModel):
    """Activity patterns"""
    active_hours: List[str] = Field(default_factory=list, description="Активные часы")
    focus_span: str = Field(default="45-60 мин", description="Время фокусировки")


class BoundariesSchema(BaseModel):
    """User boundaries"""
    no_autonomous_actions: bool = Field(default=True, description="Запрет автономных действий")
    requires_confirmation_for: List[str] = Field(default_factory=list, description="Действия, требующие подтверждения")


class PreferencesSchema(BaseModel):
    """User preferences"""
    communication_style: CommunicationStyleSchema = Field(default_factory=CommunicationStyleSchema)
    learning_style: str = Field(default="через примеры", description="Стиль обучения")
    activity_patterns: ActivityPatternsSchema = Field(default_factory=ActivityPatternsSchema)
    boundaries: BoundariesSchema = Field(default_factory=BoundariesSchema)


class PersonalityProfileSchema(BaseModel):
    """Полный профиль личности"""
    user_id: str = Field(..., description="ID пользователя")
    core_traits: CoreTraitsSchema = Field(default_factory=CoreTraitsSchema)
    motivations: MotivationsSchema = Field(default_factory=MotivationsSchema)
    values: List[ValueSchema] = Field(default_factory=list, description="Ценности")
    preferences: PreferencesSchema = Field(default_factory=PreferencesSchema)
    version: int = Field(default=1, description="Версия профиля")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class PersonalityUpdateSchema(BaseModel):
    """Schema для обновления профиля"""
    core_traits: Optional[CoreTraitsSchema] = None
    motivations: Optional[MotivationsSchema] = None
    values: Optional[List[ValueSchema]] = None
    preferences: Optional[PreferencesSchema] = None


class RecentGoalSchema(BaseModel):
    """Недавняя цель в контекстной памяти"""
    id: str
    title: str
    status: str
    progress: float


class BehavioralSummarySchema(BaseModel):
    """Поведенческое резюме за неделю"""
    completed_tasks: int = 0
    missed_tasks: int = 0
    interaction_frequency: str = "ежедневно"  # ежедневно, еженедельно, редко


class ContextualMemorySchema(BaseModel):
    """Контекстная память пользователя"""
    user_id: str
    recent_goals: List[RecentGoalSchema] = Field(default_factory=list)
    emotional_tone_recent: str = "нейтральный"
    emotional_tone_confidence: float = 0.5
    behavioral_summary_week: Optional[BehavioralSummarySchema] = None
    last_interaction_at: Optional[datetime] = None
    interaction_streak: int = 0
    updated_at: Optional[datetime] = None


class PersonalitySnapshotSchema(BaseModel):
    """Снапшот профиля"""
    snapshot_version: int
    snapshot_reason: str
    core_traits: Dict
    motivations: Dict
    values: List[Dict]
    preferences: Dict
    created_at: datetime
    created_by: str


# =============================================================================
# Personality Engine Core
# =============================================================================

class PersonalityEngine:
    """
    Главный класс Personality Engine.

    Управляет профилем личности пользователя:
    - Создание/получение/обновление профиля
    - Value Matrix для Decision Logic
    - Communication Style для Interface Layer
    - Feedback collection для адаптации
    """

    async def get_profile(self, user_id: str) -> Optional[PersonalityProfileSchema]:
        """
        Получить профиль пользователя.

        Args:
            user_id: UUID пользователя (telegram_id или system user ID)

        Returns:
            PersonalityProfileSchema или None если не найден
        """
        async with AsyncSessionLocal() as db:
            # Найти профиль
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await db.execute(stmt)
            profile_db = result.scalar_one_or_none()

            if not profile_db:
                # Создать дефолтный профиль
                profile_db = await self._create_default_profile(db, user_id)

            # Загрузить связанные данные
            # Values
            stmt_vals = select(UserValue).where(UserValue.profile_id == profile_db.id)
            result_vals = await db.execute(stmt_vals)
            values_db = result_vals.scalars().all()

            # Preferences
            stmt_prefs = select(UserPreference).where(UserPreference.profile_id == profile_db.id)
            result_prefs = await db.execute(stmt_prefs)
            prefs_db = result_prefs.scalar_one_or_none()

            # Конвертировать в Pydantic schema
            return self._db_to_schema(profile_db, values_db, prefs_db)

    async def _create_default_profile(self, db, user_id: str) -> UserProfile:
        """
        Создать дефолтный профиль для нового пользователя.

        Args:
            db: SQLAlchemy session
            user_id: UUID пользователя

        Returns:
            UserProfile (сохранённый в БД)
        """
        profile = UserProfile(
            user_id=user_id,
            # Средние значения по умолчанию
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            # Мотивации (чуть выше growth)
            motivation_growth=0.7,
            motivation_achievement=0.6,
            motivation_comfort=0.5,
            motivation_recognition=0.4,
            motivation_social_connection=0.6,
        )
        db.add(profile)
        await db.flush()  # Чтобы получить profile.id

        # Дефолтные ценности
        default_values = [
            {"name": "осознанность", "importance": 0.8, "category": "personal"},
            {"name": "здоровье", "importance": 0.7, "category": "personal"},
            {"name": "саморазвитие", "importance": 0.7, "category": "personal"},
            {"name": "честность", "importance": 0.8, "category": "personal"},
            {"name": "эффективность", "importance": 0.6, "category": "professional"},
        ]
        for val in default_values:
            db.add(UserValue(
                profile_id=profile.id,
                value_name=val["name"],
                importance=val["importance"],
                category=val["category"]
            ))

        # Дефолтные предпочтения
        prefs = UserPreference(
            profile_id=profile.id,
            tone="спокойный",
            humor="умеренный",
            detail_level="средний",
            language="ru",
            learning_style="через примеры",
            active_hours=["09:00-18:00"],
            focus_span="45-60 мин",
            no_autonomous_actions=True,
            requires_confirmation_for=["email_send", "financial_ops"]
        )
        db.add(prefs)

        await db.commit()
        await db.refresh(profile)

        return profile

    def _db_to_schema(self, profile_db: UserProfile, values_db: List[UserValue], prefs_db: UserPreference) -> PersonalityProfileSchema:
        """Конвертировать DB модели в Pydantic schema"""
        # Core traits
        core_traits = CoreTraitsSchema(
            openness=profile_db.openness,
            conscientiousness=profile_db.conscientiousness,
            extraversion=profile_db.extraversion,
            agreeableness=profile_db.agreeableness,
            neuroticism=profile_db.neuroticism,
        )

        # Motivations
        motivations = MotivationsSchema(
            growth=profile_db.motivation_growth,
            achievement=profile_db.motivation_achievement,
            comfort=profile_db.motivation_comfort,
            recognition=profile_db.motivation_recognition,
            social_connection=profile_db.motivation_social_connection,
        )

        # Values
        values = [
            ValueSchema(name=v.value_name, importance=v.importance)
            for v in values_db
        ]

        # Preferences
        if prefs_db:
            preferences = PreferencesSchema(
                communication_style=CommunicationStyleSchema(
                    tone=prefs_db.tone,
                    humor=prefs_db.humor,
                    detail_level=prefs_db.detail_level,
                    language=prefs_db.language,
                ),
                learning_style=prefs_db.learning_style,
                activity_patterns=ActivityPatternsSchema(
                    active_hours=prefs_db.active_hours or [],
                    focus_span=prefs_db.focus_span,
                ),
                boundaries=BoundariesSchema(
                    no_autonomous_actions=prefs_db.no_autonomous_actions,
                    requires_confirmation_for=prefs_db.requires_confirmation_for or [],
                )
            )
        else:
            preferences = PreferencesSchema()

        return PersonalityProfileSchema(
            user_id=str(profile_db.user_id),
            core_traits=core_traits,
            motivations=motivations,
            values=values,
            preferences=preferences,
            version=profile_db.version,
            created_at=profile_db.created_at,
            updated_at=profile_db.updated_at,
        )

    async def update_profile(self, user_id: str, updates: PersonalityUpdateSchema) -> PersonalityProfileSchema:
        """
        Обновить профиль.

        Args:
            user_id: UUID пользователя
            updates: Данные для обновления (частичные)

        Returns:
            Обновлённый PersonalityProfileSchema

        Raises:
            ValueError: если профиль не найден
        """
        async with AsyncSessionLocal() as db:
            # Найти профиль
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await db.execute(stmt)
            profile = result.scalar_one_or_none()

            if not profile:
                raise ValueError(f"Profile not found for user {user_id}")

            # Обновить версию
            profile.version += 1
            profile.updated_at = datetime.utcnow()

            # Обновить core_traits
            if updates.core_traits:
                if updates.core_traits.openness is not None:
                    profile.openness = updates.core_traits.openness
                if updates.core_traits.conscientiousness is not None:
                    profile.conscientiousness = updates.core_traits.conscientiousness
                if updates.core_traits.extraversion is not None:
                    profile.extraversion = updates.core_traits.extraversion
                if updates.core_traits.agreeableness is not None:
                    profile.agreeableness = updates.core_traits.agreeableness
                if updates.core_traits.neuroticism is not None:
                    profile.neuroticism = updates.core_traits.neuroticism

            # Обновить motivations
            if updates.motivations:
                if updates.motivations.growth is not None:
                    profile.motivation_growth = updates.motivations.growth
                if updates.motivations.achievement is not None:
                    profile.motivation_achievement = updates.motivations.achievement
                if updates.motivations.comfort is not None:
                    profile.motivation_comfort = updates.motivations.comfort
                if updates.motivations.recognition is not None:
                    profile.motivation_recognition = updates.motivations.recognition
                if updates.motivations.social_connection is not None:
                    profile.motivation_social_connection = updates.motivations.social_connection

            # Обновить values (полная замена)
            if updates.values is not None:
                # Удалить старые
                stmt_del = select(UserValue).where(UserValue.profile_id == profile.id)
                result_del = await db.execute(stmt_del)
                old_values = result_del.scalars().all()
                for v in old_values:
                    await db.delete(v)

                # Добавить новые
                for val in updates.values:
                    db.add(UserValue(
                        profile_id=profile.id,
                        value_name=val.name,
                        importance=val.importance
                    ))

            # Обновить preferences
            if updates.preferences:
                stmt_prefs = select(UserPreference).where(UserPreference.profile_id == profile.id)
                result_prefs = await db.execute(stmt_prefs)
                prefs = result_prefs.scalar_one_or_none()

                if not prefs:
                    prefs = UserPreference(profile_id=profile.id)
                    db.add(prefs)

                if updates.preferences.communication_style:
                    comm = updates.preferences.communication_style
                    if comm.tone:
                        prefs.tone = comm.tone
                    if comm.humor:
                        prefs.humor = comm.humor
                    if comm.detail_level:
                        prefs.detail_level = comm.detail_level
                    if comm.language:
                        prefs.language = comm.language

                if updates.preferences.learning_style:
                    prefs.learning_style = updates.preferences.learning_style

                if updates.preferences.activity_patterns:
                    act = updates.preferences.activity_patterns
                    if act.active_hours is not None:
                        prefs.active_hours = act.active_hours
                    if act.focus_span:
                        prefs.focus_span = act.focus_span

                if updates.preferences.boundaries:
                    bounds = updates.preferences.boundaries
                    if bounds.no_autonomous_actions is not None:
                        prefs.no_autonomous_actions = bounds.no_autonomous_actions
                    if bounds.requires_confirmation_for is not None:
                        prefs.requires_confirmation_for = bounds.requires_confirmation_for

            await db.commit()
            await db.refresh(profile)

            # Вернуть обновлённый профиль
            return await self.get_profile(user_id)

    async def record_feedback(self, user_id: str, event_type: str,
                            reaction: str, context: Dict = None,
                            correction: str = None, source: str = "system"):
        """
        Записать feedback для адаптации.

        Args:
            user_id: UUID пользователя
            event_type: Тип события ("goal_completed", "decision_approved", ...)
            reaction: Реакция ("positive", "negative", "neutral")
            context: Контекст события (dict)
            correction: Текст корректировки (если пользователь исправил)
            source: Источник ("system", "user_explicit", "user_implicit")
        """
        async with AsyncSessionLocal() as db:
            # Найти профиль
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await db.execute(stmt)
            profile = result.scalar_one_or_none()

            if not profile:
                # Если профиля нет, создаём дефолтный
                profile = await self._create_default_profile(db, user_id)

            feedback = PersonalityFeedback(
                profile_id=profile.id,
                event_type=event_type,
                context=context or {},
                reaction=reaction,
                correction=correction,
                source=source
            )
            db.add(feedback)
            await db.commit()

    async def get_value_matrix(self, user_id: str) -> Dict[str, float]:
        """
        Получить матрицу ценностей для Decision Logic.

        Args:
            user_id: UUID пользователя

        Returns:
            Dict[value_name] -> importance (0.0-1.0)
            Пример: {"осознанность": 0.8, "здоровье": 0.7, ...}
        """
        profile = await self.get_profile(user_id)
        return {v.name: v.importance for v in profile.values}

    async def get_communication_style(self, user_id: str) -> Dict:
        """
        Получить стиль общения для Interface Layer.

        Args:
            user_id: UUID пользователя

        Returns:
            Dict с communication_style
        """
        profile = await self.get_profile(user_id)
        return profile.preferences.communication_style.dict()

    async def get_core_traits(self, user_id: str) -> Dict[str, float]:
        """
        Получить Big Five traits.

        Args:
            user_id: UUID пользователя

        Returns:
            Dict с core_traits
        """
        profile = await self.get_profile(user_id)
        return profile.core_traits.dict()

    async def get_motivations(self, user_id: str) -> Dict[str, float]:
        """
        Получить мотивации.

        Args:
            user_id: UUID пользователя

        Returns:
            Dict с motivations
        """
        profile = await self.get_profile(user_id)
        return profile.motivations.dict()

    # =============================================================================
    # VERSIONING & SNAPSHOTS (NS1/NS2 Enhancement)
    # =============================================================================

    async def create_snapshot(self, user_id: str, reason: str = "update",
                            created_by: str = "system") -> PersonalitySnapshotSchema:
        """
        Создать снапшот профиля для версионирования.

        Args:
            user_id: UUID пользователя
            reason: Причина снапшота ("user_update", "adaptation", "manual")
            created_by: Кто создал ("system", "user", "auto_adaptation")

        Returns:
            PersonalitySnapshotSchema
        """
        async with AsyncSessionLocal() as db:
            # Получить или создать профиль
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await db.execute(stmt)
            profile = result.scalar_one_or_none()

            if not profile:
                # Автоматически создать дефолтный профиль
                profile = await self.get_profile(user_id)

            # Получить связанные данные
            stmt_vals = select(UserValue).where(UserValue.profile_id == profile.id)
            result_vals = await db.execute(stmt_vals)
            values_db = result_vals.scalars().all()

            stmt_prefs = select(UserPreference).where(UserPreference.profile_id == profile.id)
            result_prefs = await db.execute(stmt_prefs)
            prefs_db = result_prefs.scalar_one_or_none()

            # Создать snapshot
            snapshot = PersonalitySnapshot(
                profile_id=profile.id,
                snapshot_version=profile.version,
                snapshot_reason=reason,
                # Сериализовать в JSON
                core_traits={
                    "openness": profile.openness,
                    "conscientiousness": profile.conscientiousness,
                    "extraversion": profile.extraversion,
                    "agreeableness": profile.agreeableness,
                    "neuroticism": profile.neuroticism,
                },
                motivations={
                    "growth": profile.motivation_growth,
                    "achievement": profile.motivation_achievement,
                    "comfort": profile.motivation_comfort,
                    "recognition": profile.motivation_recognition,
                    "social_connection": profile.motivation_social_connection,
                },
                values=[{"name": v.value_name, "importance": v.importance} for v in values_db],
                preferences={
                    "communication_style": {
                        "tone": prefs_db.tone if prefs_db else "спокойный",
                        "humor": prefs_db.humor if prefs_db else "умеренный",
                        "detail_level": prefs_db.detail_level if prefs_db else "средний",
                        "language": prefs_db.language if prefs_db else "ru",
                    },
                    "learning_style": prefs_db.learning_style if prefs_db else "через примеры",
                    "activity_patterns": {
                        "active_hours": prefs_db.active_hours if prefs_db else [],
                        "focus_span": prefs_db.focus_span if prefs_db else "45-60 мин",
                    },
                    "boundaries": {
                        "no_autonomous_actions": prefs_db.no_autonomous_actions if prefs_db else True,
                        "requires_confirmation_for": prefs_db.requires_confirmation_for if prefs_db else [],
                    }
                } if prefs_db else {},
                created_by=created_by,
            )

            db.add(snapshot)
            await db.commit()
            await db.refresh(snapshot)

            return PersonalitySnapshotSchema(
                snapshot_version=snapshot.snapshot_version,
                snapshot_reason=snapshot.snapshot_reason,
                core_traits=snapshot.core_traits,
                motivations=snapshot.motivations,
                values=snapshot.values,
                preferences=snapshot.preferences,
                created_at=snapshot.created_at,
                created_by=snapshot.created_by,
            )

    async def get_snapshots(self, user_id: str, limit: int = 10) -> List[PersonalitySnapshotSchema]:
        """
        Получить историю снапшотов.

        Args:
            user_id: UUID пользователя
            limit: Макс. количество снапшотов

        Returns:
            List[PersonalitySnapshotSchema]
        """
        async with AsyncSessionLocal() as db:
            # Найти профиль
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await db.execute(stmt)
            profile = result.scalar_one_or_none()

            if not profile:
                return []

            # Получить снапшоты
            stmt_snaps = select(PersonalitySnapshot)\
                .where(PersonalitySnapshot.profile_id == profile.id)\
                .order_by(PersonalitySnapshot.created_at.desc())\
                .limit(limit)

            result_snaps = await db.execute(stmt_snaps)
            snapshots = result_snaps.scalars().all()

            return [
                PersonalitySnapshotSchema(
                    snapshot_version=s.snapshot_version,
                    snapshot_reason=s.snapshot_reason,
                    core_traits=s.core_traits,
                    motivations=s.motivations,
                    values=s.values,
                    preferences=s.preferences,
                    created_at=s.created_at,
                    created_by=s.created_by,
                )
                for s in snapshots
            ]

    async def rollback_to_snapshot(self, user_id: str, snapshot_version: int) -> PersonalityProfileSchema:
        """
        Откатиться к версии снапшота.

        Args:
            user_id: UUID пользователя
            snapshot_version: Версия для отката

        Returns:
            Обновлённый PersonalityProfileSchema

        Raises:
            ValueError: если снапшот не найден
        """
        async with AsyncSessionLocal() as db:
            # Найти профиль
            stmt = select(UserProfile).where(UserProfile.user_id == user_id)
            result = await db.execute(stmt)
            profile = result.scalar_one_or_none()

            if not profile:
                raise ValueError(f"Profile not found for user {user_id}")

            # Найти снапшот
            stmt_snap = select(PersonalitySnapshot)\
                .where(PersonalitySnapshot.profile_id == profile.id)\
                .where(PersonalitySnapshot.snapshot_version == snapshot_version)

            result_snap = await db.execute(stmt_snap)
            snapshot = result_snap.scalar_one_or_none()

            if not snapshot:
                raise ValueError(f"Snapshot version {snapshot_version} not found")

            # Восстановить из снапшота
            # Core traits
            profile.openness = snapshot.core_traits["openness"]
            profile.conscientiousness = snapshot.core_traits["conscientiousness"]
            profile.extraversion = snapshot.core_traits["extraversion"]
            profile.agreeableness = snapshot.core_traits["agreeableness"]
            profile.neuroticism = snapshot.core_traits["neuroticism"]

            # Motivations
            profile.motivation_growth = snapshot.motivations["growth"]
            profile.motivation_achievement = snapshot.motivations["achievement"]
            profile.motivation_comfort = snapshot.motivations["comfort"]
            profile.motivation_recognition = snapshot.motivations["recognition"]
            profile.motivation_social_connection = snapshot.motivations["social_connection"]

            # Увеличить версию
            profile.version += 1
            profile.updated_at = datetime.utcnow()

            # Values (полная замена)
            stmt_del = select(UserValue).where(UserValue.profile_id == profile.id)
            result_del = await db.execute(stmt_del)
            old_values = result_del.scalars().all()
            for v in old_values:
                await db.delete(v)

            for val in snapshot.values:
                db.add(UserValue(
                    profile_id=profile.id,
                    value_name=val["name"],
                    importance=val["importance"]
                ))

            # Preferences
            stmt_prefs = select(UserPreference).where(UserPreference.profile_id == profile.id)
            result_prefs = await db.execute(stmt_prefs)
            prefs = result_prefs.scalar_one_or_none()

            if not prefs:
                prefs = UserPreference(profile_id=profile.id)
                db.add(prefs)

            comm = snapshot.preferences.get("communication_style", {})
            if comm:
                prefs.tone = comm.get("tone", "спокойный")
                prefs.humor = comm.get("humor", "умеренный")
                prefs.detail_level = comm.get("detail_level", "средний")
                prefs.language = comm.get("language", "ru")

            prefs.learning_style = snapshot.preferences.get("learning_style", "через примеры")

            act = snapshot.preferences.get("activity_patterns", {})
            if act:
                prefs.active_hours = act.get("active_hours", [])
                prefs.focus_span = act.get("focus_span", "45-60 мин")

            bounds = snapshot.preferences.get("boundaries", {})
            if bounds:
                prefs.no_autonomous_actions = bounds.get("no_autonomous_actions", True)
                prefs.requires_confirmation_for = bounds.get("requires_confirmation_for", [])

            await db.commit()
            await db.refresh(profile)

            # Вернуть обновлённый профиль
            return await self.get_profile(user_id)

    # =============================================================================
    # CONTEXTUAL MEMORY (NS1/NS2 Enhancement)
    # =============================================================================

    async def get_contextual_memory(self, user_id: str) -> ContextualMemorySchema:
        """
        Получить контекстную память пользователя.

        Args:
            user_id: UUID пользователя

        Returns:
            ContextualMemorySchema
        """
        async with AsyncSessionLocal() as db:
            stmt = select(ContextualMemory).where(ContextualMemory.user_id == user_id)
            result = await db.execute(stmt)
            mem_db = result.scalar_one_or_none()

            if not mem_db:
                # Создать дефолтную контекстную память
                mem_db = ContextualMemory(user_id=user_id)
                db.add(mem_db)
                await db.commit()
                await db.refresh(mem_db)

            return ContextualMemorySchema(
                user_id=str(mem_db.user_id),
                recent_goals=[
                    RecentGoalSchema(**g) for g in (mem_db.recent_goals or [])
                ],
                emotional_tone_recent=mem_db.emotional_tone_recent,
                emotional_tone_confidence=mem_db.emotional_tone_confidence,
                behavioral_summary_week=BehavioralSummarySchema(
                    **(mem_db.behavioral_summary_week or {})
                ) if mem_db.behavioral_summary_week else None,
                last_interaction_at=mem_db.last_interaction_at,
                interaction_streak=mem_db.interaction_streak,
                updated_at=mem_db.updated_at,
            )

    async def update_contextual_memory(self, user_id: str,
                                      recent_goals: List[Dict] = None,
                                      emotional_tone: str = None,
                                      behavioral_summary: Dict = None):
        """
        Обновить контекстную память.

        Args:
            user_id: UUID пользователя
            recent_goals: Недавние цели
            emotional_tone: Эмоциональный тон
            behavioral_summary: Поведенческое резюме
        """
        async with AsyncSessionLocal() as db:
            stmt = select(ContextualMemory).where(ContextualMemory.user_id == user_id)
            result = await db.execute(stmt)
            mem = result.scalar_one_or_none()

            if not mem:
                mem = ContextualMemory(user_id=user_id)
                db.add(mem)

            if recent_goals is not None:
                mem.recent_goals = recent_goals[:5]  # Хранить только top 5

            if emotional_tone is not None:
                mem.emotional_tone_recent = emotional_tone

            if behavioral_summary is not None:
                mem.behavioral_summary_week = behavioral_summary

            # Обновить last_interaction
            from datetime import datetime
            mem.last_interaction_at = datetime.utcnow()

            await db.commit()


# =============================================================================
# Singleton instance
# =============================================================================

_personality_engine_instance = None

def get_personality_engine() -> PersonalityEngine:
    """Получить singleton instance PersonalityEngine"""
    global _personality_engine_instance
    if _personality_engine_instance is None:
        _personality_engine_instance = PersonalityEngine()
    return _personality_engine_instance
