"""
MODEL USAGE TRACKING
Отслеживание использования LLM моделей и их лимитов
"""
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class ModelUsage(Base):
    """Запросы к моделям"""
    __tablename__ = "model_usage"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, nullable=False, index=True)
    agent_role = Column(String, nullable=True)
    tokens_used = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Float, default=0.0)
    status = Column(String, default="success")  # success, rate_limited, error
    error_message = Column(String, nullable=True)
    request_params = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index('idx_model_created', 'model_name', 'created_at'),
    )


class ModelLimits(Base):
    """Лимиты на модели"""
    __tablename__ = "model_limits"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String, unique=True, nullable=False, index=True)
    rpm_limit = Column(Integer, default=100)  # requests per minute
    tpm_limit = Column(Integer, default=100000)  # tokens per minute
    daily_limit = Column(Integer, default=10000)  # requests per day
    daily_tokens = Column(Integer, default=1000000)  # tokens per day
    current_rpm = Column(Integer, default=0)
    current_tpm = Column(Integer, default=0)
    daily_requests = Column(Integer, default=0)
    daily_tokens_used = Column(Integer, default=0)
    last_reset = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="active")  # active, rate_limited, blocked
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_model_tracking(engine):
    """Инициализация таблиц отслеживания"""
    Base.metadata.create_all(engine)
