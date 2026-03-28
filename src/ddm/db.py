"""Shared async SQLAlchemy engine, session factory, and migration runner for DDM."""

import logging
import os
import pathlib

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

_engine = None
_SessionLocal = None
_MIGRATION_FILE = pathlib.Path(__file__).parent.parent.parent / "migrations" / "001_ddm_schema.sql"


def _async_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _sync_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # psycopg2 wants postgresql:// (not postgres://)
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def reset_engine() -> None:
    """Discard cached engine/session factory so they are recreated in the current event loop.

    Call this once after uvicorn has started its own loop (e.g. from a lifespan handler)
    to ensure the async engine is bound to the correct loop.
    """
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(_async_db_url(), echo=False, pool_size=5, max_overflow=10)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _SessionLocal


def run_migrations_sync() -> None:
    """Run the DDM schema migration synchronously via psycopg2.

    Safe to call before the async event loop starts (e.g. in __main__ before uvicorn).
    Uses IF NOT EXISTS throughout — idempotent on every deploy.
    """
    if not os.environ.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set — skipping DDM migrations")
        return

    if not _MIGRATION_FILE.exists():
        logger.warning(f"Migration file not found: {_MIGRATION_FILE}")
        return

    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(_sync_db_url())
        conn.autocommit = True
        cur = conn.cursor()
        sql = _MIGRATION_FILE.read_text()
        for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
            try:
                cur.execute(stmt)
            except Exception as e:
                logger.debug(f"DDL skipped ({e}): {stmt[:60]}…")
        conn.close()
        logger.info("DDM migrations applied (sync)")
    except ImportError:
        logger.warning("psycopg2 not installed — skipping sync migration")
    except Exception as e:
        logger.error(f"DDM sync migration failed: {e}")
