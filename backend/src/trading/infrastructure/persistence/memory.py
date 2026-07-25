"""In-memory event store: dev and unit tests only. Same append-only contract."""

import asyncio
from collections.abc import Sequence

from trading.application.ports.clock import ClockPort
from trading.application.ports.event_store import ExpectedVersionError
from trading.domain.events import DomainEvent, StoredEvent


class InMemoryEventStore:
    def __init__(self, clock: ClockPort) -> None:
        self._clock = clock
        self._events: list[StoredEvent] = []
        self._sequences: dict[str, int] = {}
        self._lock = asyncio.Lock()

    async def append(
        self, event: DomainEvent, *, expected_version: int | None = None
    ) -> StoredEvent:
        async with self._lock:
            current = self._sequences.get(event.stream_id, 0)
            if expected_version is not None and current != expected_version:
                # Opt-in compare-and-set (review F9): the stream advanced since the caller read.
                raise ExpectedVersionError(
                    f"stream '{event.stream_id}' advanced past expected version {expected_version}"
                )
            seq = current + 1
            self._sequences[event.stream_id] = seq
            stored = StoredEvent(event=event, seq=seq, recorded_at=self._clock.now())
            self._events.append(stored)
            return stored

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]:
        return [s for s in self._events if s.event.stream_id == stream_id]

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]:
        return [s for s in self._events if s.event.correlation_id == correlation_id]

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        if limit is None:
            return list(self._events)
        return self._events[-limit:]
