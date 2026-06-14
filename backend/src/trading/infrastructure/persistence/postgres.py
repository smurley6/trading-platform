"""Postgres event store adapter.

Append-only is enforced in the database itself (migration 0001 installs a trigger that
rejects UPDATE/DELETE), not just by convention here.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from sqlalchemy import Row, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from trading.domain.events import DomainEvent, StoredEvent
from trading.infrastructure.persistence.tables import events_table

_APPEND_RETRIES = 5

# Hook run inside the append transaction, after the event row is inserted, so a read-model
# update commits atomically with the event (CQRS read side stays consistent with the log).
# Receives the connection, the stored event, and the global `events.id` of the inserted row.
AppendHook = Callable[[AsyncConnection, StoredEvent, int], Awaitable[None]]


class EventStoreConcurrencyError(Exception):
    """Could not claim a sequence number after retries (concurrent writers on one stream)."""


class PostgresEventStore:
    def __init__(self, engine: AsyncEngine, *, on_append: AppendHook | None = None) -> None:
        self._engine = engine
        self._on_append = on_append

    async def append(self, event: DomainEvent) -> StoredEvent:
        for _ in range(_APPEND_RETRIES):
            try:
                async with self._engine.begin() as conn:
                    next_seq = await conn.scalar(
                        select(func.coalesce(func.max(events_table.c.seq), 0) + 1).where(
                            events_table.c.stream_id == event.stream_id
                        )
                    )
                    assert next_seq is not None
                    result = await conn.execute(
                        insert(events_table)
                        .values(
                            event_id=event.event_id,
                            stream_id=event.stream_id,
                            seq=next_seq,
                            event_type=event.event_type,
                            schema_version=event.schema_version,
                            occurred_at=event.occurred_at,
                            payload=event.payload,
                            correlation_id=event.correlation_id,
                        )
                        .returning(events_table.c.id, events_table.c.recorded_at)
                    )
                    row = result.one()
                    stored = StoredEvent(
                        event=event, seq=int(next_seq), recorded_at=row.recorded_at
                    )
                    if self._on_append is not None:
                        # Same transaction: read-model rows commit with the event, or not at all.
                        await self._on_append(conn, stored, int(row.id))
                    return stored
            except IntegrityError:
                continue  # another writer claimed this seq; retry with a fresh one
        raise EventStoreConcurrencyError(
            f"failed to append to stream '{event.stream_id}' after {_APPEND_RETRIES} attempts"
        )

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]:
        return await self._read(events_table.c.stream_id == stream_id)

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]:
        return await self._read(events_table.c.correlation_id == correlation_id)

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        query = select(events_table).order_by(events_table.c.id)
        if limit is not None:
            query = select(events_table).order_by(events_table.c.id.desc()).limit(limit)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        stored = [_row_to_stored(row) for row in rows]
        if limit is not None:
            stored.reverse()
        return stored

    async def _read(self, *conditions: Any) -> Sequence[StoredEvent]:
        query = select(events_table).where(*conditions).order_by(events_table.c.id)
        async with self._engine.connect() as conn:
            rows = (await conn.execute(query)).all()
        return [_row_to_stored(row) for row in rows]


def _row_to_stored(row: Row[Any]) -> StoredEvent:
    return StoredEvent(
        event=DomainEvent(
            event_id=row.event_id,
            stream_id=row.stream_id,
            event_type=row.event_type,
            schema_version=row.schema_version,
            occurred_at=row.occurred_at,
            payload=dict(row.payload),
            correlation_id=row.correlation_id,
        ),
        seq=row.seq,
        recorded_at=row.recorded_at,
    )
