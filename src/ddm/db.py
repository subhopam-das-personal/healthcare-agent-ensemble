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


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(_db_url(), echo=False, pool_size=5, max_overflow=10)
    return _engine


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            get_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _SessionLocal


async def run_migrations() -> None:
    """Run the DDM schema migration (idempotent — uses IF NOT EXISTS throughout).

    Called once at MCP server startup. Safe to run on every deploy because
    all DDL statements use IF NOT EXISTS or ON CONFLICT DO NOTHING.
    """
    if not os.environ.get("DATABASE_URL"):
        logger.warning("DATABASE_URL not set — skipping DDM migrations")
        return

    if not _MIGRATION_FILE.exists():
        logger.warning(f"Migration file not found: {_MIGRATION_FILE}")
        return

    sql = _MIGRATION_FILE.read_text()

    try:
        engine = get_engine()
        # Use raw connection for DDL (SQLAlchemy ORM doesn't handle multi-statement DDL well)
        async with engine.connect() as conn:
            # Split on semicolons, skip empty statements
            statements = [s.strip() for s in sql.split(";") if s.strip()]
            for stmt in statements:
                try:
                    await conn.exec_driver_sql(stmt)
                except Exception as e:
                    # Log but don't fail — some statements may already exist
                    logger.debug(f"DDL stmt skipped ({e}): {stmt[:60]}…")
            await conn.commit()
        logger.info("DDM migrations applied successfully")
    except Exception as e:
        logger.error(f"DDM migration failed: {e}")
