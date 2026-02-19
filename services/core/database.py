"""
Database Configuration Module
Optimized with connection pooling and async support
"""
import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import QueuePool
from typing import AsyncGenerator

# Database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Create async engine with proper connection pooling
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,              # Verify connections before use
    pool_recycle=3600,               # Recycle connections after 1 hour
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

# Base for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for FastAPI to get database session.
    Automatically handles session lifecycle.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_db_transaction() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with explicit transaction management.
    Use when you need manual commit/rollback control.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# =============================================================================
# SYNC DB ACCESS FOR READ-ONLY OPERATIONS
# =============================================================================

_sync_engine = None
_SyncSessionLocal = None

def get_sync_db():
    """
    Синхронная сессия для read-only операций мониторинга.
    Используется в IRL health metrics и invariants (без write operations).
    """
    global _sync_engine, _SyncSessionLocal
    
    if _sync_engine is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        # Remove asyncpg from URL for sync engine
        sync_url = DATABASE_URL.replace("+asyncpg", "")
        _sync_engine = create_engine(
            sync_url,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10
        )
        _SyncSessionLocal = sessionmaker(bind=_sync_engine, expire_on_commit=False)
    
    return _SyncSessionLocal()


async def close_db_connections():
    """
    Gracefully close all database connections.
    Call this on application shutdown.
    """
    await engine.dispose()


# Alias for backward compatibility
get_db_sync = get_sync_db
