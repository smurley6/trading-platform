"""Postgres advisory lock for evaluate->place serialization (review F6), over real Postgres.

The unit tests cover the use case's coordination against a fake lock; this proves the concrete
``PostgresAdvisoryLock`` actually takes a session-level ``pg_advisory_lock`` that a SEPARATE
connection cannot re-acquire while held, and releases it on every exit path (normal and error) so
a crash cannot strand the lock. Requires TEST_DATABASE_URL.
"""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from trading.infrastructure.persistence.advisory_lock import (
    PostgresAdvisoryLock,
    advisory_lock_key,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL not set",
    ),
]

_KEY = "evaluate-sleeve-political"


async def _try_lock_from_other_session(url: str) -> bool:
    """True iff a fresh session could grab the same advisory key (i.e. it is NOT currently held)."""
    probe = create_async_engine(url)
    try:
        async with probe.connect() as conn:
            got = await conn.scalar(
                text("SELECT pg_try_advisory_lock(:k)"), {"k": advisory_lock_key(_KEY)}
            )
            if got:
                await conn.execute(
                    text("SELECT pg_advisory_unlock(:k)"), {"k": advisory_lock_key(_KEY)}
                )
                await conn.commit()
            return bool(got)
    finally:
        await probe.dispose()


async def test_guard_holds_the_lock_then_releases_it() -> None:
    url = os.environ["TEST_DATABASE_URL"]
    engine = create_async_engine(url)
    lock = PostgresAdvisoryLock(engine)
    try:
        async with lock.guard(_KEY):
            # While the guard is held, no other session can take the same key.
            assert await _try_lock_from_other_session(url) is False
        # After the context exits, the lock is released and re-acquirable.
        assert await _try_lock_from_other_session(url) is True
    finally:
        await engine.dispose()


async def test_guard_releases_the_lock_on_exception() -> None:
    url = os.environ["TEST_DATABASE_URL"]
    engine = create_async_engine(url)
    lock = PostgresAdvisoryLock(engine)
    try:
        with pytest.raises(RuntimeError):
            async with lock.guard(_KEY):
                raise RuntimeError("boom inside the critical section")
        # The finally in guard() ran pg_advisory_unlock despite the error.
        assert await _try_lock_from_other_session(url) is True
    finally:
        await engine.dispose()
