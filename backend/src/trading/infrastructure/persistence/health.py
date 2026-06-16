"""Database connectivity check for the readiness probe (/readyz).

When ``DATABASE_URL`` is configured the app is only "ready" if it can reach Postgres; this
issues a trivial ``SELECT 1`` against the shared async engine. With no engine (the in-memory
default) the platform has no external dependency to probe, so readiness is trivially true.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


class DatabaseReadiness:
    """Wraps the async engine (or None) behind a single ``check()`` the /readyz route awaits."""

    def __init__(self, engine: AsyncEngine | None) -> None:
        self._engine = engine

    @property
    def configured(self) -> bool:
        return self._engine is not None

    async def check(self) -> bool:
        if self._engine is None:
            return True
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
