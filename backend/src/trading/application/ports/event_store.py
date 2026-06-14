"""Event store port: append-only system of record (spec section 9)."""

from collections.abc import Sequence
from typing import Protocol

from trading.domain.events import DomainEvent, StoredEvent


class EventStorePort(Protocol):
    async def append(self, event: DomainEvent) -> StoredEvent:
        """Durably append one event. Implementations MUST never update or delete."""
        ...

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]: ...

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]: ...

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        """All events in recording order; `limit` returns the most recent N."""
        ...
