"""Event-store decorator that bumps observability counters on append.

Wraps any ``EventStorePort`` and, on each successful append, increments
``events_appended_total`` plus ``orders_placed_total`` when the appended event is an order
submission. Mirrors the ``BroadcastingEventStore`` / ``ProjectingEventStore`` wrapping pattern:
persistence stays the inner store's job; this only adds cheap, side-effect-free metering.
"""

from collections.abc import Sequence

from trading.application.ports.event_store import EventStorePort
from trading.domain.events import DomainEvent, EventType, StoredEvent
from trading.infrastructure.observability.metrics import MetricsRegistry

EVENTS_APPENDED_TOTAL = "events_appended_total"
ORDERS_PLACED_TOTAL = "orders_placed_total"


class CountingEventStore:
    def __init__(self, inner: EventStorePort, metrics: MetricsRegistry) -> None:
        self._inner = inner
        self._metrics = metrics

    async def append(self, event: DomainEvent) -> StoredEvent:
        stored = await self._inner.append(event)
        self._metrics.increment(EVENTS_APPENDED_TOTAL)
        if event.event_type == EventType.ORDER_SUBMITTED:
            self._metrics.increment(ORDERS_PLACED_TOTAL)
        return stored

    async def read_stream(self, stream_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_stream(stream_id)

    async def read_by_correlation(self, correlation_id: str) -> Sequence[StoredEvent]:
        return await self._inner.read_by_correlation(correlation_id)

    async def read_all(self, limit: int | None = None) -> Sequence[StoredEvent]:
        return await self._inner.read_all(limit=limit)
