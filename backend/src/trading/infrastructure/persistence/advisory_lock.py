"""Advisory-lock adapters for ``CrossProcessLockPort`` (review F6).

``PostgresAdvisoryLock`` serializes the evaluate->place critical section across separate OS
processes (the daily cron and the always-on web app) that a per-process ``asyncio.Lock`` cannot
see. It uses a SESSION-level ``pg_advisory_lock`` held on one dedicated connection for the
duration of the context and released on every exit path (normal return or exception); closing the
connection releases it too, so a crash cannot strand the lock past the session. ``NoOpAdvisoryLock``
is the single-process / in-memory fallback (no database wired) where cross-process arbitration is
unnecessary.
"""

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine


def advisory_lock_key(key: str) -> int:
    """Map an arbitrary lock name to a stable signed 64-bit int for ``pg_advisory_lock``.

    Postgres advisory-lock keys are signed ``bigint``; hashing keeps a readable string name at the
    call site while producing a collision-resistant, deterministic integer the database accepts.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big", signed=True)


class PostgresAdvisoryLock:
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    @asynccontextmanager
    async def guard(self, key: str) -> AsyncIterator[None]:
        lock_key = advisory_lock_key(key)
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": lock_key})
            # A SESSION-level lock survives COMMIT; commit here so the connection is not left
            # idle-in-transaction across the (network-bound) critical section it protects.
            await conn.commit()
            try:
                yield
            finally:
                await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_key})
                await conn.commit()


class NoOpAdvisoryLock:
    """No-op ``CrossProcessLockPort`` for single-process / in-memory deployments (no database)."""

    @asynccontextmanager
    async def guard(self, key: str) -> AsyncIterator[None]:
        yield
