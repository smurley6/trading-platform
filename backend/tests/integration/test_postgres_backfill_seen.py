"""Postgres backfill dedupe store integration tests (Task 02, migration 0004).

Require TEST_DATABASE_URL pointing at a Postgres with migrations applied (the integration job
runs `alembic upgrade head`). Proves the seen/mark round-trip, cross-source isolation, and the
idempotent re-mark that underpins the backfill job's re-run safety.
"""

import os
from uuid import uuid4

import pytest

from trading.infrastructure.persistence.postgres_backfill_seen import PostgresBackfillSeen

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL not set",
    ),
]


async def test_mark_then_seen_round_trip() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresBackfillSeen(engine)
    source = f"sec_form4-{uuid4()}"
    try:
        assert await store.seen(source, ["a", "b"]) == set()
        await store.mark(source, ["a", "b"])
        assert await store.seen(source, ["a", "b", "c"]) == {"a", "b"}
        assert await store.seen(source, ["c"]) == set()
    finally:
        await engine.dispose()


async def test_remark_is_idempotent() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresBackfillSeen(engine)
    source = f"house_disclosures-{uuid4()}"
    try:
        await store.mark(source, ["x"])
        # A second mark of the same pair must not raise (PK conflict is ignored) and must not
        # duplicate — re-running a backfill window depends on this.
        await store.mark(source, ["x", "y"])
        assert await store.seen(source, ["x", "y"]) == {"x", "y"}
    finally:
        await engine.dispose()


async def test_sources_are_isolated() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresBackfillSeen(engine)
    src_a = f"sec_form4-{uuid4()}"
    src_b = f"senate_efd-{uuid4()}"
    try:
        await store.mark(src_a, ["shared-uid"])
        # The same uid under a different source is NOT seen (the key is the pair).
        assert await store.seen(src_b, ["shared-uid"]) == set()
        assert await store.seen(src_a, ["shared-uid"]) == {"shared-uid"}
    finally:
        await engine.dispose()


async def test_empty_inputs_are_safe() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresBackfillSeen(engine)
    try:
        assert await store.seen("any", []) == set()
        await store.mark("any", [])  # no rows -> no-op, must not raise
    finally:
        await engine.dispose()
