"""Postgres event store integration tests. Require TEST_DATABASE_URL with migrations applied."""

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from trading.domain.events import DomainEvent

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TEST_DATABASE_URL"),
        reason="TEST_DATABASE_URL not set",
    ),
]


def _event(stream_id: str) -> DomainEvent:
    return DomainEvent(
        stream_id=stream_id,
        event_type="test.event",
        payload={"value": 1},
        occurred_at=datetime.now(tz=UTC),
        correlation_id=str(uuid4()),
    )


async def test_append_assigns_gap_free_sequence() -> None:
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres import PostgresEventStore

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresEventStore(engine)
    stream_id = f"test-{uuid4()}"
    try:
        first = await store.append(_event(stream_id))
        second = await store.append(_event(stream_id))
        assert (first.seq, second.seq) == (1, 2)
        read = await store.read_stream(stream_id)
        assert [s.seq for s in read] == [1, 2]
    finally:
        await engine.dispose()


async def test_update_and_delete_are_forbidden_by_the_database() -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError
    from sqlalchemy.ext.asyncio import create_async_engine

    from trading.infrastructure.persistence.postgres import PostgresEventStore

    engine = create_async_engine(os.environ["TEST_DATABASE_URL"])
    store = PostgresEventStore(engine)
    stream_id = f"test-{uuid4()}"
    try:
        await store.append(_event(stream_id))
        with pytest.raises(DBAPIError, match="append-only"):
            async with engine.begin() as conn:
                await conn.execute(
                    text("UPDATE events SET event_type = 'tampered' WHERE stream_id = :s"),
                    {"s": stream_id},
                )
        with pytest.raises(DBAPIError, match="append-only"):
            async with engine.begin() as conn:
                await conn.execute(
                    text("DELETE FROM events WHERE stream_id = :s"), {"s": stream_id}
                )
    finally:
        await engine.dispose()
