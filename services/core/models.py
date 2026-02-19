from sqlalchemy import Column, String, DateTime, ForeignKey, Text, Float, Integer, JSON, Boolean, UniqueConstraint, Index, Interval, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import func
import uuid
import enum
from database import Base


# =============================================================================
# GOAL LIFECYCLE v1.1 - Completion Mode
# =============================================================================

class GoalCompletionMode(str, enum.Enum):
    """
    Режим завершения неатомарных целей

    aggregate: done = all children done (DEFAULT)
    manual: done только вручную (authority gate)
    strict: отдельный custom evaluator
    """
    AGGREGATE = "aggregate"
    MANUAL = "manual"
    STRICT = "strict"

class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Message(Base):
    __tablename__ = "messages"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Goal(Base):
    __tablename__ = "goals"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=True)

    # User association - NEW for Emotional Layer integration
    user_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    # Links goal to a specific user for personalized emotional context

    # Basic fields
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    _status = Column('status', String, default="active")  # Internal column name
    progress = Column(Float, default=0.0)

    # 🔒 PROTECTION: Direct status assignment is FORBIDDEN
    # Use GoalTransitionService.transition_goal() instead
    @hybrid_property
    def status(self):
        """Read-only status - use transition_goal() to change"""
        return self._status

    @status.setter
    def status(self, value):
        raise RuntimeError(
            f"🚫 DIRECT STATUS ASSIGNMENT BLOCKED!\n"
            f"   Attempted: goal.status = '{value}'\n"
            f"   Use: await transition_goal(goal_id, '{value}', reason='...')\n"
            f"   File: goal_transition_service.py"
        )

    # Goal Ontology - NEW
    goal_type = Column(String, default="achievable", nullable=False)
    # achievable: выполнимая цель (может быть завершена)
    # continuous: непрерывная цель (улучшение, нет финальной точки)
    # directional: векторная/ценностная (принципиально невыполнимая, задает направление)
    # exploratory: исследовательская (поиск, результат неизвестен)
    # meta: мета-цель (улучшение самой системы)

    depth_level = Column(Integer, default=0)  # 0=mission, 1=strategic, 2=operational, 3=tactical/atomic
    is_atomic = Column(Boolean, default=False)  # Является ли цель атомарной (можно превратить в задачи)

    # 🔒 GOAL LIFECYCLE v1.1 - Completion Mode
    completion_mode = Column(
        String,
        default='aggregate',
        nullable=False
    )
    # Режим автозавершения для неатомарных целей
    # aggregate: done = all children done (default)
    # manual: done только вручную
    # strict: custom evaluator logic

    # Goal Contract - NEW v3.0
    goal_contract = Column(JSON, nullable=True)
    # {
    #   "allowed_actions": ["decompose", "spawn_subgoal", "execute"],
    #   "forbidden": ["spawn_meta_goal", "external_execution"],
    #   "max_depth": 3,
    #   "max_subgoals": 7,
    #   "evaluation_mode": "binary|scalar|trend",
    #   "timeout_seconds": 300,
    #   "resource_limits": {"max_tokens": 100000, "max_api_calls": 50}
    # }

    # Goal Mutation - NEW v3.0
    mutation_history = Column(JSON, nullable=True)
    # [
    #   {"type": "strengthen|weaken|change_type|freeze", "reason": "...", "timestamp": "..."}
    # ]

    mutation_status = Column(String, default="active")  # active, frozen, mutated, deprecated

    # Completion criteria - NEW
    completion_criteria = Column(JSON, nullable=True)  # {"condition": "...", "metrics": {...}}
    success_definition = Column(Text, nullable=True)  # Описание успешного результата

    # Domain analysis - NEW
    domains = Column(JSON, nullable=True)  # ["nutrition", "light", "temperature"]
    constraints = Column(JSON, nullable=True)  # Ограничения для выполнения

    # Evaluation - NEW
    evaluation_metrics = Column(JSON, nullable=True)  # Метрики для self-evaluation
    evaluation_result = Column(JSON, nullable=True)  # Результаты проверки
    evaluation_confidence = Column(Float, nullable=True)  # Уверенность в выполнении 0.0-1.0

    # Execution tracking
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Week 3: Execution Trace
    execution_trace = Column(JSON, nullable=True)
    execution_started_at = Column(DateTime(timezone=True), nullable=True)
    execution_completed_at = Column(DateTime(timezone=True), nullable=True)

    # STEP 2.4: Emotional Forecast Link
    forecast_id = Column(UUID(as_uuid=True), ForeignKey("emotional_forecasts.id"), nullable=True)
    # Links goal to the emotional forecast made before execution

    # Relationships
    children = relationship("Goal", backref=backref('parent', remote_side=[id]))
    # Additional relations will be loaded via GoalRelation model

class GoalRelation(Base):
    """
    Relationships between goals beyond parent-child hierarchy

    Types:
    - causal: A causes or enables B (causality)
    - dependency: A depends on B (B must complete before A)
    - conflict: A conflicts with B (mutually exclusive or competing resources)
    - reinforcement: A reinforces B (progress on A helps B)
    """
    __tablename__ = "goal_relations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False)
    to_goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False)

    # Type of relationship
    relation_type = Column(String, nullable=False)  # causal, dependency, conflict, reinforcement

    # Optional strength/confidence (0.0 to 1.0)
    strength = Column(Float, default=1.0)

    # Additional metadata
    reason = Column(Text, nullable=True)  # Why this relationship exists
    relation_metadata = Column(JSON, nullable=True)  # Any additional data

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    from_goal = relationship("Goal", foreign_keys=[from_goal_id], backref="relations_from")
    to_goal = relationship("Goal", foreign_keys=[to_goal_id], backref="relations_to")

class SystemPrompt(Base):
    __tablename__ = "system_prompts"
    key = Column(String, primary_key=True)
    content = Column(Text, nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

class UserFact(Base):
    __tablename__ = "user_facts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    category = Column(String)
    content = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class RunLog(Base):
    __tablename__ = "run_logs"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String)
    agent_role = Column(String)
    tool_used = Column(String)
    input_summary = Column(Text)
    output_summary = Column(Text)
    status = Column(String)
    duration_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Review(Base):
    __tablename__ = "reviews"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False)
    decision = Column(String, nullable=False)  # 'approve' | 'improve'
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    goal = relationship("Goal", backref=backref('reviews', cascade="all, delete-orphan"))
    status = Column(String)
    duration_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ToolStats(Base):
    __tablename__ = "tool_stats"
    tool_name = Column(String, primary_key=True)
    calls_count = Column(Integer, default=0)
    errors_count = Column(Integer, default=0)
    avg_duration_ms = Column(Float, default=0.0)
    last_error = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

class Thought(Base):
    __tablename__ = "thoughts"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content = Column(Text)
    source = Column(String)
    status = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class ModelUsage(Base):
    """Отслеживание запросов к LLM моделям"""
    __tablename__ = "model_usage"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False, index=True)
    agent_role = Column(String, nullable=True)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Float, default=0.0)
    status = Column(String, default="success")  # success, rate_limited, error
    error_message = Column(String, nullable=True)
    request_params = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

class ModelLimits(Base):
    """Лимиты на LLM модели"""
    __tablename__ = "model_limits"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, unique=True, nullable=False, index=True)
    rpm_limit = Column(Integer, default=100)  # requests per minute
    tpm_limit = Column(Integer, default=100000)  # tokens per minute
    daily_limit = Column(Integer, default=10000)  # requests per day
    daily_tokens = Column(Integer, default=1000000)  # tokens per day
    current_rpm = Column(Integer, default=0)
    current_tpm = Column(Integer, default=0)
    daily_requests = Column(Integer, default=0)
    daily_tokens_used = Column(Integer, default=0)
    last_reset = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String, default="active")  # active, rate_limited, blocked
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())


class Artifact(Base):
    """
    Artifact Layer v1 - tangible results from atomic goals

    Key principle: "If I delete logs - the system's work remains"
    """
    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Type classification
    type = Column(String, nullable=False, index=True)
    # FILE: .md, .py, .json
    # KNOWLEDGE: chunk in vector DB
    # DATASET: csv, table
    # REPORT: structured summary
    # LINK: URL, repo
    # EXECUTION_LOG: run result

    # Created by
    goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False, index=True)
    skill_name = Column(String, nullable=True)
    agent_role = Column(String, nullable=True)

    # Content reference (pointer to actual content)
    content_kind = Column(String, nullable=False)  # file | db | vector | external
    content_location = Column(String, nullable=False)  # path, URL, DB ID
    description = Column(Text, nullable=True)  # Human-readable description of content

    # Metadata
    domains = Column(JSON, nullable=True)  # ["programming", "research"]
    tags = Column(JSON, nullable=True)  # ["bugfix", "feature"]
    language = Column(String, nullable=True)  # for code artifacts

    # Verification (CODE-BASED, not LLM)
    verification_status = Column(String, default="pending", index=True)  # pending | passed | failed
    verification_results = Column(JSON, nullable=True)  # [{"name": "...", "passed": true, "details": "..."}]

    # Reusability
    reusable = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    goal = relationship("Goal", backref=backref('artifacts', cascade="all, delete-orphan"))


class SkillManifestDB(Base):
    """
    Skill Manifest v1 - Database storage

    Stores skill contracts for planner and execution
    """
    __tablename__ = "skill_manifests"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Basic info
    name = Column(String, unique=True, nullable=False, index=True)
    version = Column(String, default="1.0")
    description = Column(Text)

    # Classification
    category = Column(String, nullable=False, index=True)  # research, coding, analysis, etc.
    agent_roles = Column(JSON, nullable=False)  # ["Researcher", "WebSurfer"]

    # Input/Output contract
    inputs_schema = Column(String, nullable=False)  # SearchQuery, CodeAnalysisQuery
    inputs_required = Column(JSON, nullable=False)  # ["query"]
    inputs_optional = Column(JSON, nullable=True)  # ["max_sources"]

    outputs_artifact_type = Column(String, nullable=False, index=True)  # FILE, KNOWLEDGE, etc.
    outputs_schema = Column(String, nullable=False)  # ResearchReport
    outputs_reusable = Column(Boolean, default=True)

    # What artifacts are produced
    produces = Column(JSON, nullable=False)  # [{"type": "FILE", "store": "file", ...}]

    # Constraints
    constraints = Column(JSON, nullable=True)  # {"max_tokens": 4000, ...}

    # Verification rules (CODE-BASED)
    verification = Column(JSON, nullable=False)  # [{"name": "min_sources", "rule": "..."}]

    # Failure modes
    failure_modes = Column(JSON, nullable=True)  # ["no_sources", "timeout"]

    # Metadata
    is_builtin = Column(Boolean, default=False)  # Built-in or custom
    is_active = Column(Boolean, default=True, index=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# =============================================================================
# PERSONALITY ENGINE MODELS (Phase 1)
# =============================================================================

class UserProfile(Base):
    """
    Personality Engine - профиль личности пользователя

    Хранит модель индивидуальности для персонализации:
    - Core Traits (Big Five): openness, conscientiousness, etc.
    - Motivations: growth, achievement, comfort, etc.
    - Relations: values, preferences
    """
    __tablename__ = "user_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)  # reference to telegram_id or system user

    # Core Traits (Big Five) - 0.0 to 1.0
    openness = Column(Float, default=0.5)          # открытость опыту
    conscientiousness = Column(Float, default=0.5) # добросовестность
    extraversion = Column(Float, default=0.5)      # экстраверсия
    agreeableness = Column(Float, default=0.5)     # доброжелательность
    neuroticism = Column(Float, default=0.5)       # нейротизм

    # Motivations - 0.0 to 1.0
    motivation_growth = Column(Float, default=0.5)           # стремление к развитию
    motivation_achievement = Column(Float, default=0.5)     # достижений
    motivation_comfort = Column(Float, default=0.5)         # комфорту
    motivation_recognition = Column(Float, default=0.5)     # признанию
    motivation_social_connection = Column(Float, default=0.5) # социальным связям

    # Version tracking для отката изменений
    version = Column(Integer, default=1)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    values = relationship("UserValue", back_populates="profile", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="profile", uselist=False, cascade="all, delete-orphan")
    feedback_history = relationship("PersonalityFeedback", back_populates="profile", cascade="all, delete-orphan")


class UserValue(Base):
    """
    Ценности пользователя (множественные)

    Примеры: осознанность, здоровье, саморазвитие, честность, эффективность
    """
    __tablename__ = "user_values"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)

    value_name = Column(String(100), nullable=False)  # "осознанность", "здоровье"
    importance = Column(Float, nullable=False, default=0.5)  # 0.0 - 1.0

    # Категория (опционально для группировки)
    category = Column(String(50), nullable=True)  # "personal", "social", "professional"

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    profile = relationship("UserProfile", back_populates="values")


class UserPreference(Base):
    """
    Предпочтения пользователя

    Определяет как ИИ взаимодействует с пользователем:
    - Communication style (тон, юмор, детальность)
    - Learning style
    - Activity patterns (активные часы, фокус)
    - Boundaries (границы автономности)
    """
    __tablename__ = "user_preferences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, unique=True, index=True)

    # ===== Communication Style =====
    tone = Column(String(50), default="спокойный")  # спокойный, вдохновляющий, деловой, юмористический
    humor = Column(String(50), default="умеренный")  # нет, умеренный, высокий
    detail_level = Column(String(50), default="средний")  # минимальный, средний, подробный
    language = Column(String(10), default="ru")  # ru, en

    # ===== Learning Style =====
    learning_style = Column(String(100), default="через примеры")  # через примеры, визуализация, практика

    # ===== Activity Patterns =====
    active_hours = Column(JSON, nullable=True, default=list)  # ["07:00-11:00", "18:00-21:00"]
    focus_span = Column(String(20), default="45-60 мин")  # типичное время фокусировки

    # ===== Boundaries =====
    no_autonomous_actions = Column(Boolean, default=True)  # запрещены ли автономные действия
    requires_confirmation_for = Column(JSON, nullable=True, default=list)  # ["email_send", "financial_ops"]

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationship
    profile = relationship("UserProfile", back_populates="preferences")


class PersonalityFeedback(Base):
    """
    Feedback для адаптации Personality Engine

    Записывает реакции пользователя на решения ИИ для continual learning
    """
    __tablename__ = "personality_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)

    # ===== Событие =====
    event_type = Column(String(50), nullable=False, index=True)  # "goal_completed", "decision_approved", "tone_corrected"
    context = Column(JSON, nullable=True)  # детали события (goal_id, decision_id, etc.)

    # ===== Реакция пользователя =====
    reaction = Column(String(20), nullable=False)  # "positive", "negative", "neutral"
    correction = Column(Text, nullable=True)  # текст корректировки (если пользователь исправил)

    # ===== Метаданные =====
    source = Column(String(50), default="system")  # system, user_explicit, user_implicit
    processed = Column(Boolean, default=False)  # был ли feedback учтён в адаптации

    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationship
    profile = relationship("UserProfile", back_populates="feedback_history")


# =============================================================================
# EMOTIONAL LAYER MODELS (Phase 1 - v4.0)
# =============================================================================

class EmotionalState(Base):
    """
    История эмоциональных состояний пользователя

    Записывает распознанные эмоции для анализа паттернов и адаптации
    """
    __tablename__ = "emotional_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Эмоция
    emotion_type = Column(String(50), nullable=False, index=True)  # joy, sadness, anger, fear, fatigue, stress, motivation
    intensity = Column(Float, nullable=False)  # 0.0 - 1.0
    confidence = Column(Float, default=0.7)  # уверенность распознавания

    # Причина и контекст
    cause = Column(String(200), nullable=True)  # причина эмоции (keywords)
    message_content = Column(Text, nullable=True)  # текст сообщения (короткий)
    goal_context = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=True)

    # Timestamps
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationships
    goal = relationship("Goal", backref="emotional_states")


# =============================================================================
# PERSONALITY ENHANCEMENT MODELS (v4.0)
# =============================================================================

class PersonalitySnapshot(Base):
    """
    Снапшоты профиля личности для версионирования и отката

    Позволяет:
    - Хранить историю изменений личности
    - Откатываться к предыдущим версиям
    - Анализировать динамику изменения личности
    """
    __tablename__ = "personality_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    profile_id = Column(UUID(as_uuid=True), ForeignKey("user_profiles.id"), nullable=False, index=True)

    # Версия
    snapshot_version = Column(Integer, nullable=False)
    snapshot_reason = Column(String(200), nullable=True)  # "user_update", "adaptation", "manual"

    # Полный snapshot состояния (JSON)
    core_traits = Column(JSON, nullable=False)  # {"openness": 0.7, ...}
    motivations = Column(JSON, nullable=False)  # {"growth": 0.9, ...}
    values = Column(JSON, nullable=False)  # [{"name": "осознанность", "importance": 0.8}, ...]
    preferences = Column(JSON, nullable=False)  # {"communication_style": {...}, ...}

    # Метаданные
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(50), default="system")  # system, user, auto_adaptation

    # Relationships
    profile = relationship("UserProfile", backref="snapshots")


class ContextualMemory(Base):
    """
    Короткосрочная контекстная память пользователя

    Хранит:
    - Недавние цели и их прогресс
    - Эмоциональный тон
    - Behavioral summary (completed_tasks, missed_tasks, interaction_frequency)

    Из NS1/NS2: "contextual_memory" блок
    """
    __tablename__ = "contextual_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)

    # Недавние цели (top 5)
    recent_goals = Column(JSON, nullable=True)  # [{"id": "...", "title": "...", "status": "active", "progress": 0.25}]

    # Эмоциональный тон
    emotional_tone_recent = Column(String(50), default="нейтральный")  # оптимистичный, тревожный, апатичный
    emotional_tone_confidence = Column(Float, default=0.5)  # уверенность в оценке

    # Behavioral summary (за последнюю неделю)
    behavioral_summary_week = Column(JSON, nullable=True)  # {
                                                        #   "completed_tasks": 14,
                                                        #   "missed_tasks": 3,
                                                        #   "interaction_frequency": "ежедневно"
                                                        # }

    # Last interaction
    last_interaction_at = Column(DateTime(timezone=True), nullable=True)
    interaction_streak = Column(Integer, default=0)  # дней подряд взаимодействия

    # Timestamps
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), index=True)


class GoalConflict(Base):
    """
    Обнаруженные конфликты между целями

    Выявляет несовместимые цели:
    - "работать больше" ↔ "снизить стресс"
    - "экономить" ↔ "инвестировать в развитие"
    """
    __tablename__ = "goal_conflicts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Конфликтующие цели
    goal_1_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False, index=True)
    goal_2_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False, index=True)

    # Тип конфликта
    conflict_type = Column(String(50), nullable=False)  # "resource", "time", "values", "mutually_exclusive"
    severity = Column(String(20), default="medium")  # "low", "medium", "high", "critical"

    # Описание
    description = Column(Text, nullable=False)  # "Обе цели требуют много времени"
    resolution_suggestion = Column(Text, nullable=True)  # "Приоритизировать goal_1, отложить goal_2"

    # Статус
    status = Column(String(20), default="detected")  # "detected", "resolved", "ignored"

    # Timestamps
    detected_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    goal_1 = relationship("Goal", foreign_keys=[goal_1_id], backref="conflicts_as_goal_1")
    goal_2 = relationship("Goal", foreign_keys=[goal_2_id], backref="conflicts_as_goal_2")


# =============================================================================
# GOAL COMPLETION APPROVAL (Manual Completion Layer)
# =============================================================================

class GoalCompletionApproval(Base):
    """
    🔒 MANUAL Completion Approval Record

    Audit log для ручного подтверждения завершения MANUAL целей.

    Архитектурный контракт:
    - MANUAL goal НЕ МОЖЕТ быть done без approval
    - Один approval на цель (UNIQUE goal_id)
    - Approval однократный и безотзывный (done = terminal)
    - Audit ≠ operational state (история, не бизнес-логика)

    Инвариант I7:
    goal.completion_mode == MANUAL AND goal.status == DONE
    ⇒ EXISTS approval (goal_id)

    Related:
    - invariants_checker.py:_check_manual_completion_has_approval()
    - GOAL_LIFECYCLE.md v1.1
    """
    __tablename__ = "goal_completion_approvals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Цель, которая завершилась
    goal_id = Column(
        UUID(as_uuid=True),
        ForeignKey("goals.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,  # Один approval на цель
        index=True
    )

    # Кто подтвердил (user:<id> | system | dao:<id>)
    approved_by = Column(String(200), nullable=False)

    # Уровень полномочий (1-4)
    # Phase 2.2.3: Authority level for approval governance
    authority_level = Column(Integer, nullable=False, default=1)

    # Когда подтверждено
    approved_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Опциональный комментарий (почему подтверждено)
    comment = Column(Text, nullable=True)

    # Расширенные метаданные (future: multisig, DAO vote, etc.)
    # НЕ называем 'metadata' - зарезервировано SQLAlchemy
    approval_metadata = Column(JSON, nullable=True)

    # Relationships
    goal = relationship("Goal", foreign_keys=[goal_id], backref="completion_approval")

    def __repr__(self):
        return f"<GoalCompletionApproval(goal_id={self.goal_id}, approved_by={self.approved_by}, authority_level={self.authority_level})>"


# =============================================================================
# DECISION LOGGING MODELS (Phase 2)
# =============================================================================

class DecisionLog(Base):
    """
    Лог решений системы для Self-Reflective Layer

    Записывает входные данные, варианты и результаты для анализа паттернов
    """
    __tablename__ = "decision_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Входные данные
    context = Column(JSON, nullable=False)  # ситуация, цель, эмоция, etc.
    options_generated = Column(JSON, nullable=True)  # список вариантов

    # Решение
    selected_option = Column(String(200), nullable=True)
    reasoning = Column(Text, nullable=True)  # почему выбран этот вариант
    confidence = Column(Float, nullable=True)  # уверенность в решении

    # Результат
    outcome = Column(String(20), nullable=True)  # success, failure, partial
    user_feedback = Column(String(20), nullable=True)  # positive, negative, neutral

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    outcome_measured_at = Column(DateTime(timezone=True), nullable=True)


# =============================================================================
# GROWTH LAYER MODELS (Phase 3)
# =============================================================================

class GrowthTrajectory(Base):
    """
    Траектории роста пользователя

    Отслеживает прогресс в различных доменах (когнитивный, эмоциональный, etc.)
    """
    __tablename__ = "growth_trajectories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Домен роста
    domain = Column(String(50), nullable=False, index=True)  # cognitive, emotional, physical, social, professional
    current_level = Column(String(50), default="beginner")  # beginner, intermediate, advanced, expert
    target_level = Column(String(50), nullable=True)

    # Прогресс
    micro_steps = Column(JSON, nullable=True)  # [{step, completed, date, notes}]
    progress = Column(Float, default=0.0)  # 0.0 - 1.0

    # Цель траектории
    description = Column(Text, nullable=True)
    target_date = Column(DateTime(timezone=True), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

# =============================================================================


# =============================================================================
# EMOTIONAL LAYER MODELS (MVP)
# =============================================================================

class EmotionalLayerState(Base):
    """
    Эмоциональное состояние системы (Emotional Layer MVP)

    Хранит текущее эмоциональное состояние для пользователя.
    Используется Emotional Layer для влияния на решения.
    """
    __tablename__ = "emotional_layer_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Core emotional dimensions
    arousal = Column(Float, default=0.5, nullable=False)      # 0..1 (напряжение/возбуждение)
    valence = Column(Float, default=0.0, nullable=False)      # -1..1 (негатив ↔ позитив)
    focus = Column(Float, default=0.5, nullable=False)        # 0..1 (расфокус ↔ поток)
    confidence = Column(Float, default=0.5, nullable=False)   # 0..1 (уверенность)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    source = Column(String(50), default="inference")  # user_input | system_event | inference
    signals = Column(JSON, nullable=True)  # Input signals for debugging

    # Relationships
    # Optional: link to user if User model exists
    # user = relationship("User", back_populates="emotional_states")


class AffectiveMemoryEntry(Base):
    """
    Эмоциональная память

    Записывает связь между эмоциональным состоянием и результатом решения.
    """
    __tablename__ = "affective_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Context
    goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=True)
    decision_id = Column(UUID(as_uuid=True), nullable=True)

    # Emotional states (before and after)
    emotional_state_before = Column(JSON, nullable=False)
    emotional_state_after = Column(JSON, nullable=True)

    # Outcome
    outcome = Column(String(20), nullable=False)  # success | partial | fail | unknown
    outcome_metrics = Column(JSON, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# =============================================================================
# STEP 2.4: EMOTIONAL FORECAST & OUTCOME PERSISTENCE
# =============================================================================

class EmotionalForecast(Base):
    """
    Emotional forecast — ожидание системы ДО исполнения цели.

    Инварианты:
    - immutable (не изменяется после создания)
    - одна запись = одно ожидание системы
    - создаётся в EmotionalInferenceEngine ДО execution
    """
    __tablename__ = "emotional_forecasts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Context
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=True)

    # What was predicted
    action_type = Column(String(50), nullable=False)  # "simple_task", "complex_execution", etc.
    predicted_deltas = Column(JSONB, nullable=False)  # {arousal, valence, focus, confidence}
    forecast_confidence = Column(Float, nullable=False)  # 0..1
    used_tier = Column(String(20), nullable=False)  # "ML" | "Clusters" | "Rules"

    # Risk flags (optional)
    risk_flags = Column(JSONB, nullable=True)  # ["confidence_collapse", "task_abandonment", ...]

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    outcome = relationship("EmotionalOutcome", back_populates="forecast", uselist=False)


class EmotionalOutcome(Base):
    """
    Emotional outcome — реальный результат ПОСЛЕ исполнения цели.

    Инвариант:
    - outcome не существует без forecast
    - создаётся в EmotionalFeedbackLoop ПОСЛЕ execution
    """
    __tablename__ = "emotional_outcomes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to forecast
    forecast_id = Column(UUID(as_uuid=True), ForeignKey("emotional_forecasts.id"), nullable=False)

    # What actually happened
    actual_deltas = Column(JSONB, nullable=False)  # {arousal, valence, focus, confidence}
    outcome = Column(String(20), nullable=False)  # "success" | "failure" | "aborted"

    # Timestamp
    completed_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    forecast = relationship("EmotionalForecast", back_populates="outcome")


class TierReliability(Base):
    """
    Агрегированная надежность tier для каждого action_type.

    Обновляется в Self-Eval Worker после каждого outcome.
    Используется confidence_calibrator для калибровки.
    """
    __tablename__ = "tier_reliability"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Composite key
    action_type = Column(String(50), nullable=False)
    tier = Column(String(20), nullable=False)

    # Metrics
    sample_count = Column(Integer, nullable=False, default=0)
    direction_accuracy = Column(Float, nullable=False, default=0.0)  # 0..1
    mae = Column(Float, nullable=False, default=0.0)  # Mean Absolute Error
    bias = Column(Float, nullable=False, default=0.0)  # Mean Signed Error
    cce = Column(Float, nullable=True)  # Confidence Calibration Error (optional)

    # Timestamp
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Unique constraint: one record per (action_type, tier)
    __table_args__ = (
        UniqueConstraint('action_type', 'tier', name='uq_tier_reliability_action_tier'),
    )


class SystemAlert(Base):
    """
    STEP 2.6: System Alerts — Awareness signals, NOT corrections.

    Принципы:
    - alerts ≠ errors
    - alerts ≠ corrections
    - alerts = awareness signals

    Alert сохраняется когда система обнаруживает проблему,
    но НЕ делает автоматических исправлений.
    """
    __tablename__ = "system_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Alert classification
    alert_type = Column(String(50), nullable=False)  # "ml_underperforming" | "confidence_miscalibration" | ...
    severity = Column(String(20), nullable=False)  # "INFO" | "WARNING" | "CRITICAL"

    # What happened (machine-readable)
    trigger_data = Column(JSON, nullable=False)
    # {
    #   "metric_name": "direction_accuracy",
    #   "current_value": 0.55,
    #   "threshold_value": 0.65,
    #   "sample_count": 50
    # }

    # Why (human-readable)
    explanation = Column(Text, nullable=False)
    # "ML direction accuracy 0.55 < Rules 0.72 (margin 0.17) in last 50 forecasts"

    # Context
    context = Column(JSON, nullable=True)
    # {
    #   "window_size": 50,
    #   "time_period": "last 50 forecasts",
    #   "affected_tiers": ["ML"],
    #   "affected_actions": ["complex_execution", "deep_goal_decomposition"]
    # }

    # Resolution
    resolved = Column(Boolean, nullable=False, default=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    # None — alerts are standalone signals

    # Indexes
    __table_args__ = (
        Index('idx_system_alerts_type', 'alert_type'),
        Index('idx_system_alerts_severity', 'severity'),
        Index('idx_system_alerts_created', 'created_at'),
        Index('idx_system_alerts_resolved', 'resolved'),
    )


# =============================================================================
# STEP 2.7: INTERVENTION READINESS LAYER (IRL)
# =============================================================================

class InterventionCandidate(Base):
    """
    STEP 2.7: Proposed interventions (hypotheses, NOT actions).

    Architectural invariants:
    - IRL has NO write access to models/thresholds/weights/configs
    - Approve ≠ Execute (approve only permits future application)
    """
    __tablename__ = "intervention_candidates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # What intervention is proposed
    intervention_type = Column(String(100), nullable=False)
    # Types: adjust_confidence_scaling | raise_arousal_guardrail |
    #        lower_tier_weight | disable_ml_for_context

    target_scope = Column(JSON, nullable=False)
    # Example: {"tier": "ML", "action": "*"}
    #          OR {"dimension": "arousal", "threshold": 0.75}

    # Why it's proposed
    triggered_by_alerts = Column(JSON, nullable=False)  # Array of UUIDs
    hypothesis = Column(Text, nullable=False)
    # REQUIRED: Human-readable explanation of WHY this might help

    # Expected outcomes
    expected_gain = Column(Float, nullable=False)
    # Expected improvement in accuracy/trust (Δ)

    estimated_risk = Column(Float, nullable=False)
    # Preliminary risk estimate (before simulation)

    confidence = Column(Float, nullable=False)
    # Confidence in hypothesis (0..1)

    # Status tracking
    status = Column(String(20), nullable=False, default="proposed")
    # 'proposed' | 'simulated' | 'rejected' | 'approved'

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    simulations = relationship("InterventionSimulation", back_populates="intervention", uselist=False)
    risk_score = relationship("InterventionRiskScore", back_populates="intervention", uselist=False)
    approvals = relationship("InterventionApproval", back_populates="intervention")

    # Indexes
    __table_args__ = (
        Index('idx_intervention_candidates_type', 'intervention_type'),
        Index('idx_intervention_candidates_status', 'status'),
        Index('idx_intervention_candidates_created', 'created_at'),
    )


class InterventionSimulation(Base):
    """
    STEP 2.7: Counterfactual simulation results (immutable).

    Simulation ≠ Prediction:
    - Only replay + deterministic recompute
    - NO stochasticity
    - NO future prediction
    """
    __tablename__ = "intervention_simulations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Link to intervention
    intervention_id = Column(UUID(as_uuid=True), ForeignKey("intervention_candidates.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Simulation parameters
    replay_window = Column(Interval, nullable=False)
    # Time window used for replay (e.g., "30 days")

    # Results (before vs after)
    metrics_before = Column(JSON, nullable=False)
    # Metrics BEFORE virtual intervention (accuracy, calibration, trust)
    # Example: {"direction_accuracy": 0.65, "calibration_gap": 0.12, "trust_score": 0.71}

    metrics_after = Column(JSON, nullable=False)
    # Metrics AFTER virtual intervention (recomputed)

    delta_metrics = Column(JSON, nullable=False)
    # Difference: after - before
    # Example: {"direction_accuracy": 0.05, "calibration_gap": -0.03, "trust_score": 0.08}

    side_effects = Column(JSON, nullable=True)
    # Negative consequences (where metrics got worse)
    # Example: {"arousal_accuracy": -0.12, "complex_goals_degraded": true}

    # Determinism guarantee
    determinism_hash = Column(String(64), nullable=False)
    # Hash of inputs + intervention; ensures reproducibility

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    intervention = relationship("InterventionCandidate", back_populates="simulations")

    # Indexes
    __table_args__ = (
        Index('idx_intervention_simulations_intervention', 'intervention_id'),
        Index('idx_intervention_simulations_created', 'created_at'),
    )


class InterventionRiskScore(Base):
    """
    STEP 2.7: Risk assessment for interventions.

    Risk tiers determine approval rights:
    - LOW: Approvable
    - MEDIUM: Approvable with rationale
    - HIGH: Review required
    - CRITICAL: Forbidden (even human cannot approve)
    """
    __tablename__ = "intervention_risk_scores"

    intervention_id = Column(UUID(as_uuid=True), ForeignKey("intervention_candidates.id", ondelete="CASCADE"), primary_key=True)

    # Component scores (all 0..1)
    instability_score = Column(Float, nullable=False)
    # Metric instability (higher = more unstable)

    data_sufficiency = Column(Float, nullable=False)
    # Data sufficiency (higher = more data)

    alert_density = Column(Float, nullable=False)
    # Alert density (higher = more alerts)

    arousal_exposure = Column(Float, nullable=False)
    # Exposure to high-arousal zones (0..1)

    scope_blast_radius = Column(Float, nullable=False)
    # How much system would be affected (0..1)

    # Aggregate risk
    total_risk = Column(Float, nullable=False)
    # Aggregate risk score (0..1)

    risk_tier = Column(String(20), nullable=False)
    # 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    intervention = relationship("InterventionCandidate", back_populates="risk_score")

    # Indexes
    __table_args__ = (
        Index('idx_intervention_risk_scores_tier', 'risk_tier'),
        Index('idx_intervention_risk_scores_total', 'total_risk'),
    )


class InterventionApproval(Base):
    """
    STEP 2.7: Human-in-the-loop approval history.

    All decisions are audited.
    """
    __tablename__ = "intervention_approvals"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    intervention_id = Column(UUID(as_uuid=True), ForeignKey("intervention_candidates.id", ondelete="CASCADE"), nullable=False)

    decision = Column(String(20), nullable=False)
    # 'approve' | 'reject'

    decided_by = Column(String(100), nullable=False)
    # Who made decision (user_id or "system" for auto-rejects)

    rationale = Column(Text, nullable=True)
    # Why this decision was made

    decided_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    intervention = relationship("InterventionCandidate", back_populates="approvals")

    # Indexes
    __table_args__ = (
        Index('idx_intervention_approvals_intervention', 'intervention_id'),
        Index('idx_intervention_approvals_decision', 'decision'),
        Index('idx_intervention_approvals_decided_at', 'decided_at'),
    )

# =============================================================================
# DECOMPOSITION: HUMAN-IN-THE-LOOP OBSERVATION
# =============================================================================

class DecompositionSession(Base):
    """
    Сессия декомпозиции цели - контейнер наблюдения
    
    Смысл:
    - Фиксирует точку вмешательства человека
    - Не хранит "мысли", только факты взаимодействия
    - Позволяет анализировать: какие сессии доходят до конца
    """
    __tablename__ = "decomposition_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id"), nullable=False, index=True)
    
    # Статус сессии
    status = Column(String, nullable=False, default="awaiting_user")
    # awaiting_user - ожидает первого ответа от человека
    # in_progress - идёт диалог
    # completed - декомпозиция завершена
    # aborted - прервана
    
    # Кто инициировал
    initiated_by = Column(String, nullable=False)  # 'human' | 'system'
    
    # Метаданные
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    goal = relationship("Goal", backref=backref("decomposition_sessions", cascade="all, delete-orphan"))
    questions = relationship("DecompositionQuestion", backref="session", cascade="all, delete-orphan", order_by="DecompositionQuestion.question_index")


class DecompositionQuestion(Base):
    """
    Вопрос в рамках сессии декомпозиции
    
    Смысл:
    - Порядок важен (question_index)
    - Кто задал - критично для future ML (asked_by)
    - Позволяет понять, какие формулировки работают
    """
    __tablename__ = "decomposition_questions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("decomposition_sessions.id"), nullable=False, index=True)
    
    # Содержание вопроса
    question_text = Column(Text, nullable=False)
    question_index = Column(Integer, nullable=False)  # Порядковый номер в сессии
    
    # Кто задал
    asked_by = Column(String, nullable=False)  # 'system' | 'human'
    
    # Тип вопроса (для будущего анализа)
    question_type = Column(String, nullable=True)
    # 'exploration' - исследование контекста
    # 'criteria' - критерии успеха
    # 'constraints' - ограничения
    # 'first_step' - первый шаг
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    answer = relationship("DecompositionAnswer", backref="question", uselist=False, cascade="all, delete-orphan")


class DecompositionAnswer(Base):
    """
    Ответ человека на вопрос декомпозиции
    
    Смысл:
    - Единственное место, где появляется "смысл"
    - Ground truth для future ML
    - Позволяет понять, где человек "завис"
    """
    __tablename__ = "decomposition_answers"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_id = Column(UUID(as_uuid=True), ForeignKey("decomposition_questions.id"), nullable=False, unique=True)
    
    # Содержание ответа
    answer_text = Column(Text, nullable=False)
    
    # Кто ответил
    answered_by = Column(String, nullable=False)  # 'human' | 'system'
    
    # Время ответа
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# =============================================================================
# LEGACY AXIS (S0) - CONSTITUTIONAL LAYER
# =============================================================================

class LegacyAxis(Base):
    """
    Legacy Axis (S0) - Constitutional layer of AI_OS

    Назначение:
    - Экзистенциальная миссия (20-50+ years)
    - Определяет СМЫСЛ системы, не деятельность
    - Не может быть завершена, оптимизирована, удалена

    Key principles:
    - Immutability: нельзя удалить, завершить, оптимизировать
    - Survivability: переживает автора (институты, культура, знание)
    - No Optimization: KPI forbidden (ROI, velocity, completion_rate)

    Architectural guarantee:
    - Отдельная сущность от Goal
    - Цели (Goals) DERIVED FROM Legacy Axis
    - Legacy Axis NEVER participates in decomposition
    """
    __tablename__ = "legacy_axis"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)

    # Classification
    axis_type = Column(String, nullable=False)  # civilizational, cultural, technological, existential
    generational_depth = Column(Integer, nullable=False)  # 1+ = 20-50+ years

    # Policies (JSONB for flexibility)
    survivability_policy = Column(JSONB, nullable=False, default={})
    # {
    #   "institutions": true/false,
    #   "cultural_artifacts": true/false,
    #   "knowledge_systems": true/false,
    #   "requires_author": false  # survives without author
    # }

    immutability_policy = Column(JSONB, nullable=False, default={})
    # {
    #   "cannot_be_deleted": true,
    #   "cannot_be_completed": true,
    #   "cannot_have_deadline": true,
    #   "cannot_be_optimized": true
    # }

    optimization_constraints = Column(JSONB, nullable=False, default={})
    # {
    #   "forbidden_metrics": ["ROI", "velocity", "completion_rate", "profit_only"],
    #   "allowed_metrics": ["generational_impact", "cultural_resonance", "long_term_survival"]
    # }

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True, index=True)

    # Relationships
    # Goals derive from LegacyAxis via Goal.legacy_axis_id (FK to be added)


# =============================================================================
# VECTOR ENGINE (B) - TRANSFORMATION OPERATORS
# =============================================================================

class VectorOperator(Base):
    """
    Vector Operator - Transformation function for Goals/Plans/Tasks

    Key principle: Vector is NOT a hierarchy level
    Vector is an operator: V(x) = x'

    Architectural guarantee:
    - Applies to: Goal, Plan, Task
    - NEVER applies to: Legacy Axis (S0)
    - Stateless: no persistence between applications
    - No parent-child relations
    """
    __tablename__ = "vectors"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Classification
    vector_type = Column(String, nullable=False)  # narrative, engineering, institutional, minimal, exploratory, adversarial

    # Transformation rules (how to transform)
    transformation_rules = Column(JSONB, nullable=False, default={})
    # {
    #   "decomposition_style": "narrative_first",
    #   "language_style": "story_driven",
    #   "priority_bias": ["emotional", "symbolic"],
    #   "temporal_focus": "long_term",
    #   "success_criteria": "resonance_over_metrics"
    # }

    # Forbidden targets (protection)
    forbidden_targets = Column(JSONB, nullable=False, default={"target_types": ["legacy_axis"]})
    # {
    #   "target_types": ["legacy_axis"]  # Can NEVER apply to Legacy Axis
    # }

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True, index=True)

    # Relationships
    applications = relationship("VectorApplication", backref="vector")


class VectorApplication(Base):
    """
    Vector Application - History log (READ ONLY)

    Records when a vector was applied to a target.
    This is AUDIT ONLY - vector is stateless.

    Key principle:
    - Input snapshot: state BEFORE transformation
    - Output snapshot: state AFTER transformation
    - Vector itself has NO memory
    """
    __tablename__ = "vector_applications"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    vector_id = Column(UUID(as_uuid=True), ForeignKey("vectors.id"), nullable=False)

    # What was transformed
    target_type = Column(String, nullable=False)  # goal, plan, task
    target_id = Column(UUID(as_uuid=True), nullable=False)

    # Snapshots
    input_snapshot = Column(JSONB, nullable=False)  # Before
    output_snapshot = Column(JSONB, nullable=True)  # After

    # Metadata
    applied_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    applied_by = Column(String, nullable=True)  # 'system' | user_id


# =============================================================================
# OPEN COGNITIVE CONTROL PROTOCOL (OCCP) - MCL + SK
# =============================================================================

class MetaCognitionState(Base):
    """
    Meta-Cognition Layer (MCL) - State Management

    OCCP Draft 0.1 Compliance:
    MCL manages cognitive modes, allowable decision entropy,
    and boundaries of goal interpretation.

    MCL does NOT know what to do.
    MCL knows HOW the system is allowed to think.
    """
    __tablename__ = "meta_cognition_states"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cognitive_mode = Column(String, nullable=False)  # exploration, exploitation, preservation

    # Confidence & Entropy
    epistemic_confidence = Column(Float, nullable=False, default=0.5)  # 0.0 - 1.0
    entropy_budget = Column(Float, nullable=False, default=0.5)  # 0.0 - 1.0

    # Drift Detection
    drift_score = Column(Float, nullable=False, default=0.0)  # 0.0 - 1.0

    # Risk Posture
    risk_posture = Column(String, nullable=False, default="balanced")  # conservative, balanced, aggressive

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_reviewed_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True, index=True)


class SurvivabilityKernel(Base):
    """
    Survivability Kernel (SK) - Absolute Authority Layer

    OCCP Draft 0.1 Compliance:
    SK is an immutable rule layer protecting the system from:
    - Semantic degradation
    - Incentive capture
    - Over-optimization
    - "Successful destruction"

    SK has ABSOLUTE veto authority.
    SK cannot be modified once created.
    """
    __tablename__ = "survivability_kernels"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    version = Column(String, nullable=False, default="1.0")
    authority_level = Column(String, nullable=False, default="absolute")  # absolute, override

    # Immutable flags
    is_active = Column(Boolean, default=True)
    self_modifiable = Column(Boolean, default=False)  # SK cannot modify itself

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    rules = relationship("SKRule", backref="kernel")


class SKRule(Base):
    """
    Survivability Kernel Rule

    Individual rules that SK enforces to protect system viability.
    Rules are immutable once created.
    """
    __tablename__ = "sk_rules"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kernel_id = Column(UUID(as_uuid=True), ForeignKey("survivability_kernels.id"), nullable=False)

    # Rule Definition
    rule_id = Column(String, nullable=False, unique=True)
    signal_name = Column(String, nullable=False)  # mission_drift, incentive_capture, etc.
    operator = Column(String, nullable=False)  # >, <, ==, !=, >=, <=
    threshold = Column(Float, nullable=False)

    # Actions (JSONB)
    actions = Column(JSONB, nullable=False, default=[])
    # [
    #   {"action": "freeze_component", "target": "vector.apply"},
    #   {"action": "degrade_functionality", "target": "goal_creation"},
    #   {"action": "forbid_operation", "target": "..."}
    # ]

    # Explanation (human-readable)
    explanation = Column(Text, nullable=False)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    is_active = Column(Boolean, default=True)


class OCCPAuditEvent(Base):
    """
    OCCP Audit Log - All MCL/SK decisions

    OCCP Draft 0.1 Compliance:
    Every decision by MCL or SK must be logged for transparency.
    """
    __tablename__ = "occp_audit_events"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Source
    source = Column(String, nullable=False)  # MCL, SK

    # Decision
    decision = Column(String, nullable=False)
    decision_type = Column(String, nullable=False)

    # Target
    blocked_component = Column(String, nullable=True)
    blocked_operation = Column(String, nullable=True)

    # Rationale
    rationale = Column(Text, nullable=False)

    # Context (JSONB)
    context = Column(JSONB, nullable=False, default={})

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class SurvivabilitySignal(Base):
    """
    Survivability Signals - Time-series metrics

    System health metrics that SK monitors to make veto decisions.
    """
    __tablename__ = "survivability_signals"

    # Primary fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Signal
    signal_name = Column(String, nullable=False)  # mission_drift, incentive_capture, etc.
    signal_value = Column(Float, nullable=False)

    # Metadata
    measured_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    context = Column(JSONB, nullable=False, default={})


# =============================================================================
# GOAL STATUS TRANSITIONS - Audit Trail (Phase 1)
# =============================================================================

class GoalStatusTransition(Base):
    """
    Audit trail for all goal status transitions

    Every transition MUST go through transition_goal() and be logged here.
    No direct status assignments allowed in production code.

    This is Phase 1 of Controlled Evolution - ensuring visibility into
    all goal state changes for debugging and compliance.
    """
    __tablename__ = "goal_status_transitions"

    # Primary Key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Foreign Key to goals table
    goal_id = Column(UUID(as_uuid=True), ForeignKey("goals.id", ondelete="CASCADE"), nullable=False, index=True)

    # Transition Data
    from_status = Column(String(32), nullable=False)  # pending, active, blocked, done, incomplete, frozen
    to_status = Column(String(32), nullable=False)      # where goal is going
    reason = Column(Text, nullable=False)             # WHY this transition happened
    triggered_by = Column(String(32), nullable=False, default="system")  # who triggered: goal_executor | goal_executor_v2 | goal_decomposer
    execution_id = Column(UUID(as_uuid=True), nullable=True)  # optional: links to execution run if applicable

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    # Relationship
    goal = relationship("Goal", backref=backref("status_transitions", cascade="all, delete-orphan"))
