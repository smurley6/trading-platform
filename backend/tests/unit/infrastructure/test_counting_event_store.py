"""CountingEventStore bumps observability counters on append."""

from datetime import UTC, datetime

from trading.domain.events import DomainEvent, EventType
from trading.infrastructure.clock import SystemClock
from trading.infrastructure.observability.counting_event_store import (
    EVENTS_APPENDED_TOTAL,
    ORDERS_PLACED_TOTAL,
    CountingEventStore,
)
from trading.infrastructure.observability.metrics import MetricsRegistry
from trading.infrastructure.persistence.memory import InMemoryEventStore


def _event(event_type: str) -> DomainEvent:
    return DomainEvent(
        stream_id="order-1",
        event_type=event_type,
        payload={"client_order_id": "1"},
        occurred_at=datetime(2026, 6, 12, tzinfo=UTC),
        correlation_id="corr-1",
    )


async def test_counts_every_append_and_orders_placed() -> None:
    metrics = MetricsRegistry()
    store = CountingEventStore(InMemoryEventStore(SystemClock()), metrics)

    await store.append(_event(EventType.ORDER_INTENT))
    await store.append(_event(EventType.ORDER_SUBMITTED))
    await store.append(_event(EventType.ORDER_FILLED))

    assert metrics.counter_value(EVENTS_APPENDED_TOTAL) == 3
    assert metrics.counter_value(ORDERS_PLACED_TOTAL) == 1


async def test_reads_pass_through() -> None:
    metrics = MetricsRegistry()
    store = CountingEventStore(InMemoryEventStore(SystemClock()), metrics)
    await store.append(_event(EventType.ORDER_SUBMITTED))

    assert len(await store.read_all()) == 1
    assert len(await store.read_stream("order-1")) == 1
    assert len(await store.read_by_correlation("corr-1")) == 1
