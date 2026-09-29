"""Database engine and session management using SQLAlchemy 2.x."""

from collections.abc import Generator
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine singleton."""
    global _engine
    if _engine is None:
        safe_url = settings.DATABASE_URL.split("@")[-1] if "@" in settings.DATABASE_URL else settings.DATABASE_URL
        logger.info("Initializing SQLAlchemy engine with target: %s", safe_url)

        if settings.DATABASE_URL.startswith("sqlite"):
            _engine = create_engine(
                settings.DATABASE_URL,
                connect_args={"check_same_thread": False},
                echo=settings.DB_ECHO,
            )
        else:
            _engine = create_engine(
                settings.DATABASE_URL,
                echo=settings.DB_ECHO,
                pool_size=settings.DB_POOL_SIZE,
                max_overflow=settings.DB_MAX_OVERFLOW,
                pool_timeout=settings.DB_POOL_TIMEOUT,
                pool_pre_ping=True,
            )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the sessionmaker factory bound to the engine."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = sessionmaker(
            bind=engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _session_factory


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Dispose and reset the current engine (primarily for testing)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None
