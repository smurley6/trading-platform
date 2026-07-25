"""Broadcasting event store: wraps any EventStorePort and fans appended events out to
live subscribers (the WebSocket layer). Persistence remains the inner store's job; this
only adds in-process notification. A slow subscriber is dropped from, not allowed to block,
the append path.
"""

import asyncio
import logging
from collections.abc import Sequence

from trading.application.ports.event_store import EventStorePort
from trading.domain.events import DomainEvent, StoredEvent

logger = logging.getLogger(__name__)

_SUBSCRIBER_QUEUE_SIZE = 1000


class BroadcastingEventStore:
    def __init__(self, inner: EventStorePort) -> None:
        self._inner = inner
        self._subscribers: set[asyncio.Queue[StoredEvent]] = set()

    async def append(
        self, event: DomainEvent, *, expected_version: int | None = None
    ) -> StoredEvent:
        stored = await self._inner.append(event, expected_version=expected_version)
        self._publish(stored)
        return stored

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_stream(stream_id)

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_by_correlation(correlation_id)

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        return await self._inner.read_all(limit=limit)

    def subscribe(self) -> asyncio.Queue[StoredEvent]:
        queue: asyncio.Queue[StoredEvent] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[StoredEvent]) -> None:
        self._subscribers.discard(queue)

    def _publish(self, stored: StoredEvent) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(stored)
            except asyncio.QueueFull:
                logger.warning("dropping event for a slow websocket subscriber")
