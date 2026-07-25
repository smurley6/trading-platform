"""Projecting event store: wraps an EventStorePort and keeps an in-memory read model
current as events are appended.

Persistence stays the inner store's job; this decorator only folds each appended event into
the materialized read model (CQRS read side). The read model is updated synchronously before
`append` returns, so a write is visible to the very next read (read-after-write consistency —
the e2e tests POST then GET). Reads do not touch this class; they query the read model
directly. Mirrors `BroadcastingEventStore`'s wrapping pattern.
"""

from collections.abc import Sequence

from trading.application.ports.event_store import EventStorePort
from trading.domain.events import DomainEvent, StoredEvent
from trading.infrastructure.persistence.memory_read_model import InMemoryReadModel


class ProjectingEventStore:
    def __init__(self, inner: EventStorePort, read_model: InMemoryReadModel) -> None:
        self._inner = inner
        self._read_model = read_model

    async def append(
        self, event: DomainEvent, *, expected_version: int | None = None
    ) -> StoredEvent:
        stored = await self._inner.append(event, expected_version=expected_version)
        await self._read_model.apply(stored)
        return stored

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_stream(stream_id)

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_by_correlation(correlation_id)

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        return await self._inner.read_all(limit=limit)
